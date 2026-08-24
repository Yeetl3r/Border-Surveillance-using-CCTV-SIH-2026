"""Tier 5 -- open-set identity. Enrolled friendlies only; never asserts hostility."""

from .embedder import Embedder, OSNetEmbedder, QualityGate, build_embedder
from .gallery import FriendlyGallery, IdentityResolver
from .osnet import OSNet, load_osnet, osnet_x0_25

__all__ = [
    "Embedder",
    "FriendlyGallery",
    "IdentityResolver",
    "OSNet",
    "OSNetEmbedder",
    "QualityGate",
    "build_embedder",
    "load_osnet",
    "osnet_x0_25",
]
