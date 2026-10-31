"""Embedding routing, winner selection and multi-person principal aggregation."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from src.dataset.processing.target_matching import (
    TargetMatcher,
    prepare_text,
    validate_settings,
)

SETTINGS = {
    "model": "test",
    "revision": "pinned",
    "device": "cpu",
    "batch_size": 8,
    "max_length": 128,
    "minimum_cosine_similarity": 0.7,
}
PATTERNS = {"judgment": ["zasądza {amount}"], "request": ["żąda {amount}"]}


class Encoder:
    def __init__(self, vectors):
        """
        Keep controlled sentence vectors and track encoding batches.

        args:
            vectors: Embeddings returned for source sentences.

        returns:
            None.
        """
        self.vectors = vectors
        self.calls = []

    def encode(self, texts):
        """
        Return orthogonal pattern vectors followed by configured sentence embeddings.

        args:
            texts: Pattern or source texts.

        returns:
            Float32 embedding matrix.
        """
        self.calls.append(texts)
        return torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]] if len(self.calls) == 1 else self.vectors
        )


def test_highest_match_wins_and_request_is_optional():
    """
    Preserve source offsets and select the strongest award without requiring a request.

    args:
        None.

    returns:
        None.
    """
    encoder = Encoder([[0.8, 0.6], [1.0, 0.0]])
    matcher = TargetMatcher(SETTINGS, PATTERNS, encoder)
    text = "Pierwsze zdanie.\nDrugie zdanie."
    result = matcher.match(text)
    assert result["judgment"]["text"] == "Drugie zdanie."
    assert result["judgment"]["similarity"] == 1.0
    assert result["request"] is None
    assert (
        text[result["judgment"]["start"] : result["judgment"]["end"]]
        == "Drugie zdanie."
    )
    matcher.match(text)
    assert len(encoder.calls) == 3
    assert encoder.calls[0] == ["zasądza 100 zł", "żąda 100 zł"]


@pytest.mark.parametrize(
    ("vector", "label"),
    [
        ([0.7, -math.sqrt(0.51)], "judgment"),
        ([0.69, -math.sqrt(1 - 0.69**2)], None),
        ([0.0, 1.0], "request"),
        ([1.0, 1.0], None),
    ],
)
def test_similarity_boundary_and_cross_class_ties(vector, label):
    """
    Admit the configured boundary, abstain below it and do not dual-label ties.

    args:
        vector: Controlled source embedding.
        label: Expected winning class or abstention.

    returns:
        None.
    """
    matcher = TargetMatcher(SETTINGS, PATTERNS, Encoder([vector]))
    result = matcher.match("Zdanie.")
    assert {key for key, value in result.items() if value is not None} == (
        {label} if label else set()
    )


def test_empty_text_does_not_encode_source():
    """
    Return missing classes without an empty inference batch.

    args:
        None.

    returns:
        None.
    """
    encoder = Encoder([])
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match("") == {
        "judgment": None,
        "request": None,
    }
    assert len(encoder.calls) == 1


def test_invalid_settings_and_unknown_slots_fail_before_loading():
    """
    Validate the experiment boundary without invoking model downloads.

    args:
        None.

    returns:
        None.
    """
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_settings({**SETTINGS, "minimum_cosine_similarity": 1.1}, PATTERNS)
    with pytest.raises(ValueError, match="Unknown"):
        prepare_text("zasądza {unsupported}")


def test_long_sentence_windows_are_pooled_without_dropping_tokens():
    """
    Recombine all windows into sentence vectors and respect inference batch size.

    args:
        None.

    returns:
        None.
    """
    from types import SimpleNamespace

    from src.dataset.processing.target_matching import SentenceEncoder

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
