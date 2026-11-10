import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def split_train_batch(batch):
    mention = {}
    entity = {}
    for name, value in batch.items():
        target = entity if name.startswith("entity_") else mention
        target[name[len("entity_"):] if name.startswith("entity_") else name] = value
    return mention, entity


def ranking_metrics(ranks):
    ranks = np.asarray(ranks)
    return {"hits1": float((ranks <= 1).mean()), "hits3": float((ranks <= 3).mean()), "hits5": float((ranks <= 5).mean()), "hits10": float((ranks <= 10).mean()), "hits20": float((ranks <= 20).mean()), "mr": float(ranks.mean()), "mrr": float((1.0 / ranks).mean())}


class EIAFNTrainer:
    def __init__(self, model, data, config, output_dir):
        self.config = config
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        no_decay = ("bias", "LayerNorm.bias", "LayerNorm.weight")
        groups = [{"params": [p for n, p in model.named_parameters() if not any(term in n for term in no_decay)], "weight_decay": 1e-4}, {"params": [p for n, p in model.named_parameters() if any(term in n for term in no_decay)], "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(groups, lr=config.lr, betas=(0.9, 0.999), eps=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def _cache_entities(self):
        parts = {"text_global": [], "image_global": [], "text_tokens": [], "image_tokens": []}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.data.entity_loader(), desc="Encoding entities"):
                features = self.model.encode(move_to_device(batch, self.device))
                for name in parts:
                    parts[name].append(getattr(features, name).cpu())
        from .model import EncodedFeatures
        return EncodedFeatures(**{name: torch.cat(values).to(self.device) for name, values in parts.items()})

    def _evaluate(self, loader, label):
        entities = self._cache_entities()
        ranks, all_scores = [], []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=label):
                answer = batch.pop("answer").to(self.device)
                mention = self.model.encode(move_to_device(batch, self.device))
                chunks = []
                for start in range(0, entities.text_global.size(0), self.config.data.eval_chunk_size):
                    stop = start + self.config.data.eval_chunk_size
                    from .model import EncodedFeatures
                    entity_chunk = EncodedFeatures(*(getattr(entities, field)[start:stop] for field in ("text_global", "image_global", "text_tokens", "image_tokens")))
                    chunks.append(self.model.score_encoded(mention, entity_chunk)[0])
                scores = torch.cat(chunks, dim=1)
                rank = torch.argsort(torch.argsort(scores, dim=1, descending=True), dim=1) + 1
                ranks.extend(rank[torch.arange(answer.size(0), device=self.device), answer].cpu().tolist())
                if label == "test":
                    all_scores.extend(scores.cpu().tolist())
        metrics = ranking_metrics(ranks)
        with open(self.output_dir / f"{label}_ranks.json", "w", encoding="utf-8") as handle:
            json.dump(ranks, handle, indent=2)
        if label == "test":
            with open(self.output_dir / "test_scores.json", "w", encoding="utf-8") as handle:
                json.dump(all_scores, handle)
        return metrics

    def fit(self):
        best_mrr = float("-inf")
        remaining_patience = self.config.trainer.early_stopping_patience
        history_path = self.output_dir / "metrics.csv"
        with open(history_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_mrr"])
            writer.writeheader()
            for epoch in range(1, self.config.trainer.max_epochs + 1):
                self.model.train()
                losses = []
                for batch in tqdm(self.data.train_loader(), desc=f"Train {epoch}"):
                    mention, entity = split_train_batch(move_to_device(batch, self.device))
                    fused, branches = self.model(mention, entity)
                    loss = self.model.contrastive_loss(fused, branches)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                self.scheduler.step()
                train_loss = float(np.mean(losses))
                validation = self._evaluate(self.data.validation_loader(), "validation")
                writer.writerow({"epoch": epoch, "train_loss": train_loss, "validation_mrr": validation["mrr"]})
                handle.flush()
                print(f"epoch={epoch} train_loss={train_loss:.6f} validation_mrr={validation['mrr']:.6f}")
                if validation["mrr"] > best_mrr:
                    best_mrr = validation["mrr"]
                    remaining_patience = self.config.trainer.early_stopping_patience
                    torch.save({"epoch": epoch, "model_state": self.model.state_dict(), "optimizer_state": self.optimizer.state_dict(), "validation": validation}, self.output_dir / "best.pt")
                else:
                    remaining_patience -= 1
                    if remaining_patience == 0:
                        break
        checkpoint = torch.load(self.output_dir / "best.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        return checkpoint["validation"]

    def test(self):
        return self._evaluate(self.data.test_loader(), "test")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
