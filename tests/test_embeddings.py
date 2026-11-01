"""Reusable embedding API boundaries and encoder reuse without downloads."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.embeddings import create_embeddings
from src.dataset.embeddings import encoding

SETTINGS = {
    "model": "test",
    "revision": "pinned",
    "device": "cpu",
    "batch_size": 2,
    "max_length": 128,
}


@pytest.fixture(autouse=True)
def clear_encoder_cache():
    """Keep mocked encoders isolated from other tests.

    args:
        None.

    returns:
        None.
    """
    encoding.cached_encoder.cache_clear()
    yield
    encoding.cached_encoder.cache_clear()


@pytest.mark.parametrize("texts", [[], "text", [None], [123], [""], ["  "]])
def test_invalid_texts_fail_before_model_loading(monkeypatch, texts):
    """Reject malformed batches before importing inference dependencies or loading weights.

    args:
        monkeypatch: Scoped encoder replacement.
        texts: Invalid batch input.

    returns:
        None.
    """
    monkeypatch.setattr(encoding, "SentenceEncoder", lambda *args: pytest.fail("Unexpected model load"))
    with pytest.raises(ValueError, match="texts must"):
        create_embeddings(texts, SETTINGS)


@pytest.mark.parametrize(
    "change",
    [
        {"model": ""},
        {"revision": None},
        {"device": "unknown"},
        {"batch_size": 0},
        {"batch_size": True},
        {"max_length": 3},
    ],
)
def test_invalid_encoder_settings_fail_before_loading(monkeypatch, change):
    """Validate encoder-only settings independently of target roles and patterns.

    args:
        monkeypatch: Scoped encoder replacement.
        change: Invalid encoder setting.

    returns:
        None.
    """
    monkeypatch.setattr(encoding, "SentenceEncoder", lambda *args: pytest.fail("Unexpected model load"))
    with pytest.raises(ValueError):
        create_embeddings(["Text"], {**SETTINGS, **change})


def test_original_texts_and_order_are_preserved_and_models_reused(monkeypatch):
    """Reuse encoder-only settings without altering monetary values or source strings.

    args:
        monkeypatch: Scoped encoder replacement.

    returns:
        None.
    """
    loaded, batches = [], []
    vectors = object()

    def encode(texts):
        """Record original batches and return the model output unchanged.

        args:
            texts: Original strings in input order.

        returns:
            Controlled encoder output.
        """
        batches.append(texts)
        return vectors

    def load(settings):
        """Record normalized settings without downloading model weights.

        args:
            settings: Encoder-only model and runtime configuration.

        returns:
            Controlled encoder.
        """
        loaded.append(settings)
        return SimpleNamespace(encode=encode)

    monkeypatch.setattr(encoding, "SentenceEncoder", load)
    texts = ["Zasądza 10000 zł.", "Ograniczenia w pracy."]
    assert create_embeddings(texts, SETTINGS) is vectors
    assert create_embeddings([texts[1]], {**SETTINGS, "minimum_cosine_similarity": 0.9}) is vectors
    assert loaded == [SETTINGS]
    assert batches == [texts, [texts[1]]]
    create_embeddings(texts, {**SETTINGS, "revision": "other"})
    assert len(loaded) == 2
