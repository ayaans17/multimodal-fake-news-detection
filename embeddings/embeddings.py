from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics.pairwise import cosine_similarity
from src.config import ensure_dirs, load_yaml, project_path


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


@torch.no_grad()
def generate_dataset_embeddings(
    dataset_key: str,
    batch_size: int,
    device: torch.device,
    model_name: str = "openai/clip-vit-base-patch32",
) -> None:
    output_dir = project_path("embeddings") / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for split in ("train", "val", "test"):
        frames.append(pd.read_csv(project_path("data/processed") / dataset_key / f"{split}.csv"))
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("sample_id")

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()

    loader = DataLoader(
        ClipEmbeddingDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_clip_batch,
    )
    image_embeddings: dict[str, torch.Tensor] = {}
    text_embeddings: dict[str, torch.Tensor] = {}
    consistency_rows = []

    for batch in tqdm(loader, desc=f"CLIP embeddings {dataset_key}"):
        inputs = processor(
            text=batch['text'],
            images=batch['images'],
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }
        outputs = model(**inputs)

        text_features = outputs.text_embeds
        image_features = outputs.image_embeds

        scores = cosine_similarity(
                text_features.cpu().numpy(),
                image_features.cpu().numpy(),
            ).tolist()

        scores = np.diag(scores).tolist()

        for sample_id, image_feature, text_feature, score in zip(
            batch["sample_id"],
            image_features,
            text_features,
            scores,
        ):
            image_embeddings[str(sample_id)] = image_feature
            #print('aa',score)
            text_embeddings[str(sample_id)] = text_feature
            consistency_rows.append({"sample_id": sample_id, "consistency_score": float(score)})

    torch.save(image_embeddings, output_dir / "clip_image_embeddings.pt")
    torch.save(text_embeddings, output_dir / "clip_text_embeddings.pt")
    pd.DataFrame(consistency_rows).to_csv(output_dir / "consistency_scores.csv", index=False)


def run_embedding_generation(config_path: str | Path, batch_size: int = 32) -> None:
    ensure_dirs()
    cfg = load_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset_key in cfg["datasets"]:
        generate_dataset_embeddings(dataset_key, batch_size=batch_size, device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_embedding_generation(args.config, batch_size=args.batch_size)


if __name__ == "__main__":
    main()

