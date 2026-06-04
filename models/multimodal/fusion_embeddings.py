import torch
import torch.nn as nn


class FusionEmbeddingModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.text_projection = nn.Linear(
            768,
            512
        )

        self.image_projection = nn.Linear(
            768,
            512
        )

        self.classifier = nn.Sequential(

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 2)
        )

    def forward(

        self,
        text_embedding,
        image_embedding

    ):

        text_embedding = self.text_projection(
            text_embedding
        )

        image_embedding = self.image_projection(
            image_embedding
        )

        fused = torch.cat(
            [text_embedding, image_embedding],
            dim=1
        )

        logits = self.classifier(
            fused
        )

        return logits