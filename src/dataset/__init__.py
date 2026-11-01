"""Reusable dataset feature extraction."""

# Shared text normalization and feature extraction may also serve future inference.
from .extraction.basal_filtering import extract, load_config
from .extraction.text import extract_text

__all__ = ["extract", "extract_text", "load_config"]
