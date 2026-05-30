import torch
import torch.nn as nn

from transformers import (
    DistilBertModel,
    CLIPModel
)


class MultimodalFakeNewsModel(nn.Module):

    def __init__(self):

        super().__init__()

        # TEXT ENCODER
        self.text_encoder = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )

        # IMAGE ENCODER
        self.clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        )

        # projection layers
        self.text_projection = nn.Linear(768, 512)

        self.image_projection = nn.Linear(768, 512)

        # classifier
        self.classifier = nn.Sequential(

            nn.Linear(1024, 512),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(512, 2)
        )

    def forward(

        self,
        input_ids,
        attention_mask,
        pixel_values
    ):

        # TEXT FEATURES
        text_outputs = self.text_encoder(

            input_ids=input_ids,
            attention_mask=attention_mask
        )


        text_features = text_outputs.last_hidden_state[:, 0, :]

        # print(type(text_outputs))
        #
        # print(type(text_features))
        #
        # print(text_features.shape)

        text_features = self.text_projection(
            text_features
        )

        # IMAGE FEATURES
        image_features = self.clip_model.vision_model(
            pixel_values=pixel_values
        )
        image_features = image_features.pooler_output
        # print("bb", type(image_features))
        #
        # print("cc",image_features.shape)

        image_features = self.image_projection(
            image_features
        )

        # FUSION
        fused = torch.cat(
            [text_features, image_features],
            dim=1
        )

        # CLASSIFICATION
        logits = self.classifier(fused)

        return logits