import copy
import json
import os
import random
from pathlib import Path
from urllib.parse import unquote

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor


def _read_json(paths):
    paths = [paths] if isinstance(paths, str) else paths
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            records.extend(json.load(handle))
    return records


class EIAFNData:
    """Owns MEL preprocessing and the three ordinary PyTorch data loaders."""

    def __init__(self, config):
        self.config = config
        checkpoint = config.pretrained_model
        processor = CLIPProcessor.from_pretrained(checkpoint)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

        with open(config.data.qid2id, "r", encoding="utf-8") as handle:
            self.qid_to_index = json.loads(handle.readline())

        raw_entities = sorted(_read_json(config.data.entity), key=lambda item: item["id"])
        self.entities = [self._encode_entity(item) for item in raw_entities]
        # The supplied qid2id map stores the integer identifiers used by the
        # mention labels; entity records use those identifiers directly.
        self.entity_by_index = {raw["id"]: encoded for raw, encoded in zip(raw_entities, self.entities)}
        self.train = self._encode_mentions(_read_json(config.data.train_file))
        self.validation = self._encode_mentions(_read_json(config.data.dev_file))
        self.test = self._encode_mentions(_read_json(config.data.test_file))

    def _tokenize(self, text):
        return self.tokenizer(text, padding="max_length", max_length=self.config.data.text_max_length, truncation=True)

    def _encode_entity(self, sample):
        item = copy.deepcopy(sample)
        text = f"{unquote(item['entity_name'])} [SEP] {item['attr']}"
        encoded = self._tokenize(text)
        encoded["image_list"] = item.get("image_list", [])
        encoded["sample_type"] = 0
        return encoded

    def _encode_mentions(self, samples):
        encoded_samples = []
        for sample in tqdm(samples, desc="Preprocessing mentions"):
            item = copy.deepcopy(sample)
            answer = item.get("answer")
            if answer == "nil":
                continue
            text = f"{unquote(item['mentions'])} [SEP] {item['sentence']}"
            encoded = self._tokenize(text)
            encoded["image_list"] = [item["imgPath"]] if item.get("imgPath") else []
            encoded["sample_type"] = 1
            if answer is not None:
                encoded["answer"] = self.qid_to_index[answer]
            encoded_samples.append(encoded)
        return encoded_samples

    def _load_image(self, sample_type, image_list, evaluation):
        if not image_list:
            return torch.zeros(3, 224, 224)
        image_name = image_list[0] if evaluation else random.choice(image_list)
        if sample_type == 1:
            image_name = f"{Path(image_name).stem}.jpg"
        folder = self.config.data.kb_img_folder if sample_type == 0 else self.config.data.mention_img_folder
        try:
            with Image.open(os.path.join(folder, image_name)) as image:
                image = image.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
                return self.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except (OSError, ValueError):
            return torch.zeros(3, 224, 224)

    def _collate_inputs(self, samples, evaluation):
        tokenized, images = [], []
        for source in samples:
            item = copy.deepcopy(source)
            image_list = item.pop("image_list")
            sample_type = item.pop("sample_type")
            item.pop("answer", None)
            tokenized.append(item)
            images.append(self._load_image(sample_type, image_list, evaluation))
        batch = self.tokenizer.pad(tokenized, padding="max_length", max_length=self.config.data.text_max_length, return_tensors="pt")
        batch["pixel_values"] = torch.stack(images)
        return batch

    def _train_collate(self, samples):
        mention_batch = self._collate_inputs(samples, evaluation=False)
        entity_samples = [self.entity_by_index[sample["answer"]] for sample in samples]
        entity_batch = self._collate_inputs(entity_samples, evaluation=False)
        mention_batch.update({f"entity_{name}": value for name, value in entity_batch.items()})
        return mention_batch

    def _evaluation_collate(self, samples):
        batch = self._collate_inputs(samples, evaluation=True)
        batch["answer"] = torch.tensor([sample["answer"] for sample in samples], dtype=torch.long)
        return batch

    def _entity_collate(self, samples):
        return self._collate_inputs(samples, evaluation=True)

    def train_loader(self):
        return DataLoader(self.train, batch_size=self.config.data.batch_size, shuffle=True, num_workers=self.config.data.num_workers, collate_fn=self._train_collate)

    def validation_loader(self):
        return DataLoader(self.validation, batch_size=self.config.data.eval_batch_size, shuffle=False, num_workers=self.config.data.num_workers, collate_fn=self._evaluation_collate)

    def test_loader(self):
        return DataLoader(self.test, batch_size=self.config.data.eval_batch_size, shuffle=False, num_workers=self.config.data.num_workers, collate_fn=self._evaluation_collate)

    def entity_loader(self):
        return DataLoader(self.entities, batch_size=self.config.data.embed_update_batch_size, shuffle=False, num_workers=self.config.data.num_workers, collate_fn=self._entity_collate)
