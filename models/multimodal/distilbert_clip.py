import torch
import torch.nn as nn

from transformers import (
    DistilBertModel,
    CLIPModel
)


class DistilBERTCLIPEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # TEXT ENCODER
        self.text_encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # IMAGE ENCODER
        self.image_encoder = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # PROJECTION LAYERS
        # Make dimensions compatible

        self.text_projection = nn.Linear(
            768,
            512
        )

        self.image_projection = nn.Linear(
            512,
            512
        )

    def forward(

        self,
        input_ids,
        attention_mask,
        pixel_values
    ):

        # ==========================
        # TEXT FEATURES
        # ==========================

        text_outputs = self.text_encoder(

            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # CLS token embedding
        text_features = text_outputs.last_hidden_state[:, 0, :]

        # projection
        text_features = self.text_projection(
            text_features
        )

        # ==========================
        # IMAGE FEATURES
        # ==========================

        image_features = self.image_encoder.get_image_features(
            pixel_values=pixel_values
        )

        # projection
        image_features = self.image_projection(
            image_features
        )

        return text_features, image_features