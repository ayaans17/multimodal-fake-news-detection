from __future__ import annotations

import torch
from torch import nn

from src.models.text import BiLSTMTextEncoder, DistilBertTextEncoder


class AttentionFusion(nn.Module):
    def __init__(self, text_dim: int, image_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_features: torch.Tensor, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        text_projected = torch.relu(self.text_projection(text_features))
        image_projected = torch.relu(self.image_projection(image_features))
        logits = self.attention(torch.cat([text_projected, image_projected], dim=1))
        weights = torch.softmax(logits, dim=1)
        fused = weights[:, :1] * text_projected + weights[:, 1:] * image_projected
        return self.dropout(fused), weights


class MultimodalFakeNewsModel(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        vocab_size: int,
        hidden_dim: int = 256,
        image_dim: int = 512,
        dropout: float = 0.2,
        use_consistency: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.use_consistency = use_consistency
        if text_encoder_type == "distilbert":
            self.text_encoder = DistilBertTextEncoder()
        elif text_encoder_type == "bilstm":
            self.text_encoder = BiLSTMTextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim, dropout=dropout)
        else:
            raise ValueError(f"Unsupported text encoder {text_encoder_type!r}")

        self.fusion = AttentionFusion(
            text_dim=self.text_encoder.output_dim,
            image_dim=image_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        classifier_input = hidden_dim + (1 if use_consistency else 0)
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embedding: torch.Tensor,
        consistency_score: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        fused, attention_weights = self.fusion(text_features, image_embedding)
        if self.use_consistency:
            if consistency_score is None:
                raise ValueError("consistency_score is required when use_consistency=True.")
            fused = torch.cat([fused, consistency_score], dim=1)
        logits = self.classifier(fused)
        return {"logits": logits, "attention_weights": attention_weights}
