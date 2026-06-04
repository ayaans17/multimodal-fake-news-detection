import numpy as np
import pandas as pd
import torch

from PIL import Image
from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity

from transformers import (
    CLIPModel,
    CLIPProcessor
)

# ==========================
# DEVICE
# ==========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ==========================
# CLIP
# ==========================

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(device)

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()

# ==========================
# FUNCTION
# ==========================

def extract_scores(
    csv_path,
    output_path
):

    df = pd.read_csv(csv_path)

    scores = []

    with torch.no_grad():

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            text = str(
                row["clean_text"]
            )

            image = Image.open(
                row["path"]
            ).convert("RGB")

            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            inputs = {
                k: v.to(device)
                for k, v in inputs.items()
            }
            #print(inputs.keys())

            outputs = model(**inputs)

            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds

            similarity = cosine_similarity(
                text_embeds.cpu().numpy(),
                image_embeds.cpu().numpy()
            )[0][0]
            #print(text_features.)

            # similarity = cosine_similarity(
            #     text_features.text_embeds.cpu().numpy(),
            #     image_features.image_embeds.cpu().numpy()
            # )[0][0]

            scores.append(
                similarity
            )

    scores = np.array(
        scores,
        dtype=np.float32
    )

    np.save(
        output_path,
        scores
    )

    print(
        f"Saved {output_path}"
    )

# ==========================
# RUN
# ==========================

extract_scores(
    "../Dataset/processed/evons/train.csv",
    "consistency_score/evons/train_consistency.npy"
)

extract_scores(
    "../Dataset/processed/evons/val.csv",
    "consistency_score/evons/val_consistency.npy"
)

extract_scores(
    "../Dataset/processed/evons/test.csv",
    "consistency_score/evons/test_consistency.npy"
)