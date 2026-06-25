from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer

from src.config import ensure_dirs, load_yaml, project_path
from src.data import load_consistency_scores, load_image_embeddings
from src.models import MultimodalFakeNewsModel
from src.train import build_loader, evaluate, resolve_device


def load_experiment_params(experiments_cfg: dict[str, Any], experiment_name: str) -> dict[str, Any]:
    defaults = experiments_cfg["defaults"]
    experiment = next(item for item in experiments_cfg["experiments"] if item["name"] == experiment_name)
    return {**defaults, **experiment}


def evaluate_cross_dataset(
    dataset_cfg: dict[str, Any],
    experiments_cfg: dict[str, Any],
    source_experiment: str,
    target_dataset: str,
    split: str = "test",
) -> dict[str, Any]:
    ensure_dirs()
    params = load_experiment_params(experiments_cfg, source_experiment)
    device = resolve_device(str(params.get("device", "auto")))

    checkpoint_path = project_path("models") / source_experiment / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_params = {**params, **checkpoint.get("params", {})}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    frame = pd.read_csv(project_path("data/processed") / target_dataset / f"{split}.csv")
    image_embeddings = load_image_embeddings([target_dataset])
    consistency_scores = (
        load_consistency_scores([target_dataset])
        if checkpoint_params.get("use_consistency")
        else {}
    )

    loader = build_loader(
        frame=frame,
        tokenizer=tokenizer,
        max_length=int(dataset_cfg.get("max_text_length", 128)),
        image_embeddings=image_embeddings,
        consistency_scores=consistency_scores,
        batch_size=int(checkpoint_params["batch_size"]),
        shuffle=False,
        num_workers=int(checkpoint_params.get("num_workers", 0)),
    )

    model = MultimodalFakeNewsModel(
        text_encoder_type=checkpoint_params["text_encoder"],
        vocab_size=tokenizer.vocab_size,
        hidden_dim=int(checkpoint_params["hidden_dim"]),
        dropout=float(checkpoint_params["dropout"]),
        use_consistency=bool(checkpoint_params["use_consistency"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()
    metrics, labels, predictions, probabilities = evaluate(model, loader, criterion, device)

    output_dir = project_path("results") / "cross_dataset" / f"{source_experiment}_on_{target_dataset}_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_frame = frame[["dataset", "sample_id", "text", "image_path", "label"]].copy()
    predictions_frame["prediction"] = predictions
    predictions_frame["probability_real"] = probabilities
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    result = {
        "source_experiment": source_experiment,
        "source_datasets": checkpoint_params.get("datasets"),
        "target_dataset": target_dataset,
        "split": split,
        "text_encoder": checkpoint_params["text_encoder"],
        "use_consistency": bool(checkpoint_params["use_consistency"]),
        "metrics": metrics,
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--source-experiment", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    dataset_cfg = load_yaml(args.config)
    experiments_cfg = load_yaml(args.experiments)
    result = evaluate_cross_dataset(
        dataset_cfg=dataset_cfg,
        experiments_cfg=experiments_cfg,
        source_experiment=args.source_experiment,
        target_dataset=args.target_dataset,
        split=args.split,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

