"""Reusable dataset feature extraction."""

# Shared text normalization and feature extraction may also serve future inference.
from .processing.basal_filtering import extract, load_config
from .processing.text import extract_text

__all__ = ["extract", "extract_text", "load_config"]
