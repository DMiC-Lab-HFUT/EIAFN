from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import CLIPModel


@dataclass
class EncodedFeatures:
    text_global: torch.Tensor
    image_global: torch.Tensor
    text_tokens: torch.Tensor
    image_tokens: torch.Tensor


class EIAFNEncoder(nn.Module):
    """CLIP feature encoder at the global and contextual granularities."""

    def __init__(self, config):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(config.pretrained_model)
        self.image_token_projection = nn.Linear(config.model.input_image_hidden_dim, config.model.dv)

    def forward(self, input_ids, attention_mask, pixel_values, **_):
        output = self.clip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values, return_dict=True)
        return EncodedFeatures(
            text_global=output.text_embeds,
            image_global=output.image_embeds,
            text_tokens=output.text_model_output.last_hidden_state,
            image_tokens=self.image_token_projection(output.vision_model_output.last_hidden_state),
        )


class ImplicitTextMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.model.TGLU_hidden_dim
        input_dim = config.model.input_hidden_dim
        self.query = nn.Linear(input_dim, hidden)
        self.key = nn.Linear(input_dim, hidden)
        self.value = nn.Linear(input_dim, hidden)
        self.global_projection = nn.Linear(input_dim, hidden)
        self.context_norm = nn.LayerNorm(hidden)

    def forward(self, mention, entity):
        entity_global = self.global_projection(entity.text_global)
        mention_global = self.global_projection(mention.text_global)
        query = self.query(entity.text_tokens).unsqueeze(0)
        key = self.key(mention.text_tokens).unsqueeze(1)
        value = self.value(mention.text_tokens).unsqueeze(1)
        attention = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(query.size(-1)), dim=-1)
        context = self.context_norm((attention @ value).mean(dim=-2))
        contextual_score = (entity_global.unsqueeze(0) * context).sum(dim=-1)
        global_score = mention_global @ entity_global.transpose(0, 1)
        return global_score + contextual_score


class AdaptiveFeedbackLayer(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.project = nn.Linear(dimension, dimension)
        self.gate = nn.Linear(dimension, 1)
        self.norm = nn.LayerNorm(dimension)

    def forward(self, query_global, key_global, value_tokens):
        pooled_value = value_tokens.mean(dim=1)
        context = self.project(self.norm(query_global.unsqueeze(1) + pooled_value.unsqueeze(0)))
        refined = self.norm(context * torch.tanh(self.gate(context)) + key_global.unsqueeze(0))
        return (refined * self.norm(query_global).unsqueeze(1)).sum(dim=-1)


class ImplicitVisionMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension = config.model.IDLU_hidden_dim
        self.global_projection = nn.Linear(config.model.input_hidden_dim, dimension)
        self.token_projection = nn.Linear(config.model.dv, dimension)
        self.entity_to_mention = AdaptiveFeedbackLayer(dimension)
        self.mention_to_entity = AdaptiveFeedbackLayer(dimension)

    def forward(self, mention, entity):
        mention_global = self.global_projection(mention.image_global)
        entity_global = self.global_projection(entity.image_global)
        mention_tokens = self.token_projection(mention.image_tokens)
        entity_tokens = self.token_projection(entity.image_tokens)
        entity_to_mention = self.entity_to_mention(entity_global, mention_global, mention_tokens).transpose(0, 1)
        mention_to_entity = self.mention_to_entity(mention_global, entity_global, entity_tokens)
        return entity_to_mention + mention_to_entity


class ImplicitCrossModalMatcher(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimension = config.model.CMFU_hidden_dim
        self.text_projection = nn.Linear(config.model.input_hidden_dim, dimension)
        self.image_projection = nn.Linear(config.model.dv, dimension)
        self.gate = nn.Linear(dimension, 1)
        self.norm = nn.LayerNorm(dimension)

    def _context(self, text_global, image_tokens):
        text = self.text_projection(text_global)
        images = self.image_projection(image_tokens)
        weights = torch.softmax(text.unsqueeze(1) @ images.transpose(-1, -2), dim=-1)
        attended = (weights @ images).squeeze(1)
        return self.norm(text * torch.tanh(self.gate(text)) + attended)

    def forward(self, mention, entity):
        mention_context = self._context(mention.text_global, mention.image_tokens)
        entity_context = self._context(entity.text_global, entity.image_tokens)
        return mention_context @ entity_context.transpose(0, 1)


class ExplicitAssociationUnit(nn.Module):
    """Projected cosine similarities, as specified by the explicit unit."""

    def __init__(self, config):
        super().__init__()
        hidden = config.model.hidden_dim
        input_dim = config.model.input_hidden_dim
        self.text_projection = nn.Linear(input_dim, hidden)
        self.image_projection = nn.Linear(input_dim, hidden)
        self.cross_text_projection = nn.Linear(input_dim, hidden)
        self.cross_image_projection = nn.Linear(input_dim, hidden)

    @staticmethod
    def _cosine(left, right):
        return F.normalize(left, dim=-1) @ F.normalize(right, dim=-1).transpose(0, 1)

    def forward(self, mention, entity):
        text = self._cosine(self.text_projection(mention.text_global), self.text_projection(entity.text_global))
        image = self._cosine(self.image_projection(mention.image_global), self.image_projection(entity.image_global))
        cross = self._cosine(self.cross_image_projection(mention.image_global), self.cross_text_projection(entity.text_global))
        return text, image, cross


class EIAFN(nn.Module):
    """Explicit-Implicit Associative Fusion Network, implemented without Lightning."""

    def __init__(self, config):
        super().__init__()
        self.encoder = EIAFNEncoder(config)
        self.implicit_text = ImplicitTextMatcher(config)
        self.implicit_vision = ImplicitVisionMatcher(config)
        self.implicit_cross_modal = ImplicitCrossModalMatcher(config)
        self.explicit = ExplicitAssociationUnit(config)
        gate_input = config.model.input_hidden_dim * 2
        self.fusion_gate = nn.Sequential(nn.Linear(gate_input, config.model.hidden_dim), nn.ReLU(), nn.Linear(config.model.hidden_dim, 3))
        self.loss_weights = nn.Parameter(torch.tensor([0.3, 0.3, 0.4], dtype=torch.float))

    def encode(self, batch):
        return self.encoder(**batch)

    def score_encoded(self, mention, entity):
        implicit_text = self.implicit_text(mention, entity)
        implicit_image = self.implicit_vision(mention, entity)
        implicit_cross = self.implicit_cross_modal(mention, entity)
        explicit_text, explicit_image, explicit_cross = self.explicit(mention, entity)
        text_score = implicit_text * explicit_text
        image_score = implicit_image * explicit_image
        cross_score = implicit_cross * explicit_cross
        weights = torch.softmax(self.fusion_gate(torch.cat((mention.text_global, mention.image_global), dim=-1)), dim=-1)
        fused_score = weights[:, :1] * text_score + weights[:, 1:2] * image_score + weights[:, 2:3] * cross_score
        return fused_score, (text_score, image_score, cross_score)

    def forward(self, mention_batch, entity_batch):
        return self.score_encoded(self.encode(mention_batch), self.encode(entity_batch))

    def contrastive_loss(self, fused_scores, branch_scores):
        labels = torch.arange(fused_scores.size(0), device=fused_scores.device)
        branch_loss = sum(weight * F.cross_entropy(score, labels) for weight, score in zip(self.loss_weights, branch_scores))
        return branch_loss + F.cross_entropy(fused_scores, labels)
