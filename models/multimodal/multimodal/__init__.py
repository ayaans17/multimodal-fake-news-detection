"""Multimodal models."""

from src.models.multimodal.distilbert_clip import DistilBertClipModel
from src.models.multimodal.distilbert_clip_fusion import AttentionFusion, MultimodalFakeNewsModel

__all__ = ["AttentionFusion", "DistilBertClipModel", "MultimodalFakeNewsModel"]
