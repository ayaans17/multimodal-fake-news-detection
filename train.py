from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.config import ensure_dirs, load_yaml, project_path
from src.data import MultimodalDataset, load_consistency_scores, load_image_embeddings, read_splits
from src.models import MultimodalFakeNewsModel


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def metrics_from_predictions(labels: list[int], predictions: list[int]) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary", zero_division=0),
        "recall": recall_score(labels, predictions, average="binary", zero_division=0),
        "f1": f1_score(labels, predictions, average="binary", zero_division=0),
    }


def save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    params: dict[str, Any],
    epoch: int,
    best_epoch: int,
    best_f1: float,
    stale_epochs: int,
    history: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "params": params,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "stale_epochs": stale_epochs,
            "history": history,
        },
        path,
    )


def build_loader(
    frame: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_length: int,
    image_embeddings: dict[str, torch.Tensor],
    consistency_scores: dict[str, float],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = MultimodalDataset(
        frame=frame,
        tokenizer=tokenizer,
        max_length=max_length,
        image_embeddings=image_embeddings,
        consistency_scores=consistency_scores,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image_embedding=batch["image_embedding"],
            consistency_score=batch["consistency_score"],
        )
        loss = criterion(output["logits"], batch["label"])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * batch["label"].size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int], list[float]]:
    model.eval()
    total_loss = 0.0
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        batch = move_batch(batch, device)
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image_embedding=batch["image_embedding"],
            consistency_score=batch["consistency_score"],
        )
        logits = output["logits"]
        loss = criterion(logits, batch["label"])
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1)
        total_loss += loss.item() * batch["label"].size(0)
        labels.extend(batch["label"].cpu().tolist())
        predictions.extend(pred.cpu().tolist())
        probabilities.extend(probs[:, 1].cpu().tolist())
    metrics = metrics_from_predictions(labels, predictions)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics, labels, predictions, probabilities


def train_experiment(
    dataset_cfg: dict[str, Any],
    experiments_cfg: dict[str, Any],
    experiment_name: str,
    resume: bool = True,
) -> dict[str, Any]:
    ensure_dirs()
    defaults = experiments_cfg["defaults"]
    experiment = next(item for item in experiments_cfg["experiments"] if item["name"] == experiment_name)
    params = {**defaults, **experiment}
    device = resolve_device(str(params.get("device", "auto")))

    dataset_keys = list(params["datasets"])
    splits = read_splits(dataset_keys)
    image_embeddings = load_image_embeddings(dataset_keys)
    consistency_scores = load_consistency_scores(dataset_keys) if params.get("use_consistency") else {}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loaders = {
        split: build_loader(
            frame,
            tokenizer,
            max_length=int(dataset_cfg.get("max_text_length", 128)),
            image_embeddings=image_embeddings,
            consistency_scores=consistency_scores,
            batch_size=int(params["batch_size"]),
            shuffle=split == "train",
            num_workers=int(params.get("num_workers", 0)),
        )
        for split, frame in splits.items()
    }

    model = MultimodalFakeNewsModel(
        text_encoder_type=params["text_encoder"],
        vocab_size=tokenizer.vocab_size,
        hidden_dim=int(params["hidden_dim"]),
        dropout=float(params["dropout"]),
        use_consistency=bool(params["use_consistency"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )
    total_steps = len(loaders["train"]) * int(params["epochs"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_steps)),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()

    output_model_dir = project_path("models") / experiment_name
    output_result_dir = project_path("results") / experiment_name
    output_model_dir.mkdir(parents=True, exist_ok=True)
    output_result_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    start_epoch = 1
    last_checkpoint_path = output_model_dir / "last_checkpoint.pt"

    if resume and last_checkpoint_path.exists():
        checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_f1 = float(checkpoint.get("best_val_f1", best_f1))
        best_epoch = int(checkpoint.get("best_epoch", best_epoch))
        stale_epochs = int(checkpoint.get("stale_epochs", stale_epochs))
        history = list(checkpoint.get("history", history))
        start_epoch = int(checkpoint["epoch"]) + 1

    for epoch in range(start_epoch, int(params["epochs"]) + 1):
        print("Epoch ", epoch)
        train_loss = train_epoch(model, loaders["train"], optimizer, scheduler, criterion, device)
        val_metrics, _, _, _ = evaluate(model, loaders["val"], criterion, device)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "params": params,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_f1,
                },
                output_model_dir / "best_model.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(params["patience"]):
                save_training_checkpoint(
                    last_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    params,
                    epoch,
                    best_epoch,
                    best_f1,
                    stale_epochs,
                    history,
                )
                break

        save_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            params,
            epoch,
            best_epoch,
            best_f1,
            stale_epochs,
            history,
        )

    checkpoint = torch.load(output_model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics, labels, predictions, probabilities = evaluate(model, loaders["test"], criterion, device)

    predictions_frame = splits["test"][["dataset", "sample_id", "text", "image_path", "label"]].copy()
    predictions_frame["prediction"] = predictions
    predictions_frame["probability_real"] = probabilities
    predictions_frame.to_csv(output_result_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(output_result_dir / "history.csv", index=False)

    result = {
        "experiment": experiment_name,
        "datasets": dataset_keys,
        "text_encoder": params["text_encoder"],
        "use_consistency": bool(params["use_consistency"]),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "test": test_metrics,
    }
    with (output_result_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return result


def run_selected(
    config_path: str | Path,
    experiments_path: str | Path,
    experiment_name: str,
    resume: bool = True,
) -> dict[str, Any]:
    dataset_cfg = load_yaml(config_path)
    experiments_cfg = load_yaml(experiments_path)
    return train_experiment(dataset_cfg, experiments_cfg, experiment_name, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    parser.add_argument("--experiments", default="configs/experiments.yaml")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--no-resume", action="store_true", help="Ignore last_checkpoint.pt and start from epoch 1.")
    args = parser.parse_args()
    dataset_cfg = load_yaml(args.config)
    experiments_cfg = load_yaml(args.experiments)
    result = train_experiment(dataset_cfg, experiments_cfg, args.experiment, resume=not args.no_resume)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
