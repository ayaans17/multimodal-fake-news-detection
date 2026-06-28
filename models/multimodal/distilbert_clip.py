from __future__ import annotations

from src.models.multimodal.distilbert_clip_fusion import MultimodalFakeNewsModel


class DistilBertClipModel(MultimodalFakeNewsModel):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        image_dim: int = 512,
        dropout: float = 0.2,
        use_consistency: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__(
            text_encoder_type="distilbert",
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            image_dim=image_dim,
            dropout=dropout,
            use_consistency=use_consistency,
            num_classes=num_classes,
        )
