"""Embedding routing, winner selection and multi-person principal aggregation."""


import math

import sys

from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


torch = pytest.importorskip("torch")


SETTINGS = {
    "model": "test",
    "revision": "pinned",
    "device": "cpu",
    "batch_size": 8,
    "max_length": 128,
    "minimum_cosine_similarity": 0.7,
    "candidate_keywords": {
        "judgment": ["zasądz"],
        "request": ["żąda"],
        "shared": ["przyzna"],
        "prior_payment": ["wypłac"],
        "procedural_costs": ["koszt"],
        "mixed": ["żąda"],
        "other_payments": ["odset", "rent", "grzywn"],
    },
}


def test_long_sentence_windows_are_pooled_without_dropping_tokens():
    """
    Recombine all windows into sentence vectors and respect inference batch size.

    args:
        None.

    returns:
        None.
    """
    from types import SimpleNamespace

    from src.dataset.embeddings.encoding import SentenceEncoder

    encoder = SentenceEncoder.__new__(SentenceEncoder)
    encoder.settings = {**SETTINGS, "batch_size": 2}
    encoder.tokenizer = lambda *args, **kwargs: {
        "input_ids": torch.tensor([[1, 1], [3, 3], [5, 5]]),
        "attention_mask": torch.ones((3, 2), dtype=torch.long),
        "overflow_to_sample_mapping": torch.tensor([0, 0, 1]),
    }

    class Model:
        config = SimpleNamespace(hidden_size=2)

        def __call__(self, input_ids, attention_mask):
            """
            Supply deterministic token vectors for the pooling calculation.

            args:
                input_ids: Token values for this batch.
                attention_mask: Nonpadding mask.

            returns:
                Token vectors with two identical feature coordinates.
            """
            assert len(input_ids) <= 2
            return SimpleNamespace(
                last_hidden_state=input_ids.float().unsqueeze(-1).repeat(1, 1, 2)
            )

    encoder.model = Model()
    torch.testing.assert_close(
        encoder.encode(["long sentence", "short sentence"]),
        torch.tensor([[2.0, 2.0], [5.0, 5.0]]),
    )

