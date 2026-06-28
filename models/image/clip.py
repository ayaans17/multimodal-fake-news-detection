from __future__ import annotations

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ClipEmbeddingDataset(Dataset):
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        return {
            "sample_id": row["sample_id"],
            "text": row["text"],
            "image_path": row["image_path"],
        }


def load_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (224, 224), color=(0, 0, 0))


def collate_clip_batch(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "sample_id": [item["sample_id"] for item in batch],
        "text": [item["text"] for item in batch],
        "images": [load_image(str(item["image_path"])) for item in batch],
    }
