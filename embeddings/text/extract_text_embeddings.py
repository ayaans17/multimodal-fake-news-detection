import numpy as np
import pandas as pd
import torch

#print(np.load("train_buzzfeed_text.npy").shape)
#print(np.load("..\\image\\train_buzzfeed_image.npy").shape)
from tqdm import tqdm
from transformers import (
    DistilBertTokenizer,
    DistilBertModel
)

device = torch.device("cpu")

tokenizer = DistilBertTokenizer.from_pretrained(
    "distilbert-base-uncased"
)

model = DistilBertModel.from_pretrained(
    "distilbert-base-uncased"
)

model.eval()
model.to(device)

df = pd.read_csv("..\\..\\Dataset\\processed\\buzzfeed\\train_clean.csv"
)

embeddings = []

with torch.no_grad():

    for text in tqdm(df["clean_text"]):

        encoding = tokenizer(
            str(text),
            truncation=True,
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )

        encoding = {
            k: v.to(device)
            for k, v in encoding.items()
        }

        outputs = model(**encoding)

        embedding = (
            outputs.last_hidden_state[:, 0, :]
            .cpu()
            .numpy()
        )

        embeddings.append(
            embedding.squeeze()
        )

embeddings = np.array(embeddings)

np.save(
    "train_ifnd_text.npy",
    embeddings
)

print(embeddings.shape)