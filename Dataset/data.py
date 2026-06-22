from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.config import project_path


class MultimodalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        image_embeddings: dict[str, torch.Tensor],
        consistency_scores: dict[str, float] | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_embeddings = image_embeddings
        self.consistency_scores = consistency_scores or {}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        tokens = self.tokenizer(
            row["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        sample_id = row["sample_id"]
        image_embedding = self.image_embeddings[sample_id].float()
        consistency = float(self.consistency_scores.get(sample_id, 0.0))
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "image_embedding": image_embedding,
            "consistency_score": torch.tensor([consistency], dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def read_splits(dataset_keys: list[str]) -> dict[str, pd.DataFrame]:
    splits: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for dataset_key in dataset_keys:
        base = project_path("data/processed") / dataset_key
        for split in splits:
            splits[split].append(pd.read_csv(base / f"{split}.csv"))
    return {split: pd.concat(frames, ignore_index=True) for split, frames in splits.items()}


def load_image_embeddings(dataset_keys: list[str]) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for dataset_key in dataset_keys:
        path = project_path("embeddings") / dataset_key / "clip_image_embeddings.pt"
        merged.update(torch.load(path, map_location="cpu"))
    return merged


def load_consistency_scores(dataset_keys: list[str]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for dataset_key in dataset_keys:
        path = project_path("embeddings") / dataset_key / "consistency_scores.csv"
        if path.exists():
            frame = pd.read_csv(path)
            merged.update(dict(zip(frame["sample_id"], frame["consistency_score"])))
    return merged

