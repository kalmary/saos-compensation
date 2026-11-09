"""Embedding cache isolation shared by encoder and matcher tests."""

import pytest

from dataset.embeddings.encoding import cached_encoder
from dataset.embeddings.target_matching import cached_matcher


@pytest.fixture(autouse=True)
def clear_model_caches():
    """Prevent cached fake encoders and pattern vectors from crossing test boundaries.

    args:
        None.

    returns:
        None.
    """
    cached_encoder.cache_clear()
    cached_matcher.cache_clear()
    yield
    cached_encoder.cache_clear()
    cached_matcher.cache_clear()
