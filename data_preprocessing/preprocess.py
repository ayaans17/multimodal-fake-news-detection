from __future__ import annotations

import argparse
import re
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ensure_dirs, load_yaml, project_path


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = str(text)
    # remove html
    text = BeautifulSoup(text, "html.parser").get_text()

    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#", "", text)
    #
    # # remove extra spaces/newlines
    text = re.sub(r"\s+", " ", text).strip()
    #
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    return text.strip()


def normalize_label(value: object, label_mapping: dict[str, int] | None) -> int:
    if pd.isna(value):
        raise ValueError("Label contains missing values.")
    if isinstance(value, (int, float)) and int(value) == value:
        return int(value)
    lowered = str(value).strip().lower()
    if label_mapping and lowered in label_mapping:
        return int(label_mapping[lowered])
    if lowered in {"0", "1"}:
        return int(lowered)
    raise ValueError(f"Unknown label {value!r}; add it to label_mapping in configs/datasets.yaml.")


def resolve_image_path(image_dir: Path, image_value: object) -> str:
    raw = "" if pd.isna(image_value) else str(image_value).strip()
    if not raw:
        return ""
    candidate = Path(raw)
    if candidate.is_absolute():
        return str(candidate)
    direct = image_dir / raw
    if direct.exists():
        return str(direct)
    stem = image_dir / raw
    for extension in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        with_ext = stem.with_suffix(extension)
        if with_ext.exists():
            return str(with_ext)
    return str(direct)


def stratified_split(
    frame: pd.DataFrame,
    seed: int,
    test_size: float,
    val_size: float,
) -> dict[str, pd.DataFrame]:
    stratify = frame["label"] if frame["label"].nunique() > 1 else None
    train_val, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    relative_val = val_size / (1.0 - test_size)
    train_stratify = train_val["label"] if train_val["label"].nunique() > 1 else None
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_stratify,
    )
    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def preprocess_dataset(dataset_key: str, dataset_cfg: dict, global_cfg: dict) -> None:
    csv_path = project_path(dataset_cfg["csv_path"])
    image_dir = project_path(dataset_cfg["image_dir"])
    output_dir = project_path("data/processed") / dataset_key
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(csv_path,encoding="latin1")
    #print("aa",frame.head())
    text_col = dataset_cfg.get("text_col", "text")
    image_col = dataset_cfg.get("image_col", "img_id")
    label_col = dataset_cfg.get("label_col", "label")
    frame=frame.dropna()
    processed = pd.DataFrame(
        {
            "dataset": dataset_key,
            "sample_id": [f"{dataset_key}_{idx}" for idx in range(len(frame))],
            "text": frame[text_col].map(clean_text),
            "image_path": frame[image_col].map(lambda value: resolve_image_path(image_dir, value)),
            "label": frame[label_col].map(lambda value: normalize_label(value, global_cfg.get("label_mapping"))),
        }
    )
    processed = processed.dropna(subset=["text", "label"]).reset_index(drop=True)
    processed = processed[processed["text"].str.len() > 0].reset_index(drop=True)
    processed = processed.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print("bb", processed.head())

    for split_name, split_frame in stratified_split(
        processed,
        seed=int(global_cfg.get("seed", 42)),
        test_size=float(global_cfg.get("test_size", 0.15)),
        val_size=float(global_cfg.get("val_size", 0.15)),
    ).items():
        split_frame.to_csv(output_dir / f"{split_name}.csv", index=False)


def run_preprocessing(config_path: str | Path) -> None:
    ensure_dirs()
    cfg = load_yaml(config_path)
    for dataset_key, dataset_cfg in cfg["datasets"].items():
        preprocess_dataset(dataset_key, dataset_cfg, cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets.yaml")
    args = parser.parse_args()
    run_preprocessing(args.config)


if __name__ == "__main__":
    main()

