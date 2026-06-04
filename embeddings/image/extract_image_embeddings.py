import numpy as np
import pandas as pd
import torch
print(np.load("train_ifnd_image.npy"))
from tqdm import tqdm

from transformers import (
    CLIPProcessor,
    CLIPModel
)
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cpu")

processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
)

model.eval()
model.to(device)

df = pd.read_csv(
    "..\\..\\Dataset\\processed\\buzzfeed\\val.csv"
)

embeddings = []
valid_indices=[]
with torch.no_grad():
    i=0

    for idx, image_path in enumerate(tqdm(df["path"])):#for image_path in tqdm(df["path"]):


        # image = Image.open(
        #     "..\\"+image_path
        # ).convert("RGB")

        try:
            image = Image.open( "..\\"+image_path).convert("RGB")

        except Exception as e:

            print("\n===== IMAGE ERROR =====")
            print("Path:", image_path)
            print("Error:", e)
            continue

        inputs = processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(device)
            for k, v in inputs.items()
        }

        image_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"]
        )

        embedding = (
            image_outputs.pooler_output
            .cpu()
            .numpy()
        )

        embeddings.append(
            embedding.squeeze()
        )
        valid_indices.append(idx)

embeddings = np.array(embeddings)

if len(valid_indices) != len(df):
    clean_df = df.iloc[valid_indices]

    clean_df.to_csv(
    "..\\..\\Dataset\\processed\\buzzfeed\\val_clean.csv",
    index=False
)
np.save(
    "train_buzzfeed_image.npy",
    embeddings
)

print(embeddings.shape)