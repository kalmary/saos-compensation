"""Embedding routing, winner selection and multi-person principal aggregation."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from src.dataset.processing.target_matching import (
    PATTERN_CLASSES,
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
PATTERNS = {
    "judgment": ["zasądza {amount}"],
    "request": ["żąda {amount}"],
    "prior_payment": ["wypłacił już {amount}"],
    "procedural_costs": ["zwrot kosztów procesu {amount}"],
    "mixed": ["żądał więcej, zasądzono {amount}"],
    "other_payments": ["odsetki wynoszą {amount}"],
}


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
            torch.eye(len(PATTERN_CLASSES)).tolist()
            if len(self.calls) == 1
            else [
                vector + [0.0] * (len(PATTERN_CLASSES) - len(vector))
                for vector in self.vectors
            ]
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
    text = "Zasądza 100 zł.\nPrzyznaje 200 zł."
    result = matcher.match(text)
    assert result["judgment"]["text"] == "Przyznaje 200 zł."
    assert result["judgment"]["similarity"] == 1.0
    assert result["request"] is None
    assert (
        text[result["judgment"]["start"] : result["judgment"]["end"]]
        == "Przyznaje 200 zł."
    )
    matcher.match(text)
    assert len(encoder.calls) == 3
    assert encoder.calls[0] == [
        prepare_text(phrases[0]) for phrases in PATTERNS.values()
    ]


@pytest.mark.parametrize(
    ("vector", "label"),
    [
        ([0.7, -math.sqrt(0.51)], None),
        ([0.71, -math.sqrt(1 - 0.71**2)], "judgment"),
        ([0.69, -math.sqrt(1 - 0.69**2)], None),
        ([0.0, 1.0], "request"),
        ([1.0, 1.0], "judgment"),
    ],
)
def test_strict_similarity_boundary_and_single_class_winner(vector, label):
    """
    Reject the configured boundary and assign one deterministic class above it.

    args:
        vector: Controlled source embedding.
        label: Expected winning class or abstention.

    returns:
        None.
    """
    matcher = TargetMatcher(SETTINGS, PATTERNS, Encoder([vector]))
    result = matcher.match("Przyznaje 100 zł.")
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
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match("") == dict.fromkeys(
        PATTERN_CLASSES
    )
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


def test_request_only_sentence_cannot_become_judgment_despite_higher_similarity():
    """
    Respect keyword eligibility before choosing a cosine winner.

    args:
        None.

    returns:
        None.
    """
    matcher = TargetMatcher(SETTINGS, PATTERNS, Encoder([[0.72, 0.71]]))
    result = matcher.match("Żąda 100 zł.")
    assert result["judgment"] is None
    assert result["request"]["similarity"] >= 0.7


@pytest.mark.parametrize(
    "groups",
    [
        ["zasądz"],
        {"judgment": ["zasądz"], "request": []},
        {"judgment": [], "request": [], "shared": []},
        {"judgment": "zasądz", "request": [], "shared": []},
        {"judgment": [""], "request": [], "shared": []},
    ],
)
def test_invalid_keyword_groups_fail_before_loading(groups):
    """
    Reject malformed keyword groups at the configuration boundary.

    args:
        groups: Invalid candidate keyword configuration.

    returns:
        None.
    """
    with pytest.raises(ValueError, match="candidate_keywords"):
        validate_settings({**SETTINGS, "candidate_keywords": groups}, PATTERNS)


@pytest.mark.parametrize(
    ("text", "vector", "excluded"),
    [
        ("Zasądza 100 zł kosztów procesu.", [0.70, 0, 0, 0.71, 0], "procedural_costs"),
        ("Zasądza 100 zł odsetek.", [0.70, 0, 0, 0, 0, 0.71], "other_payments"),
        ("Przyznano i wypłacono już 100 zł.", [0.70, 0, 0.71, 0, 0], "prior_payment"),
        ("Żądał zasądzenia 200 zł, zasądzono 100 zł.", [0.70, 0, 0, 0, 0.71], "mixed"),
    ],
)
def test_exclusion_winner_does_not_become_target_and_keeps_other_candidates(
    text, vector, excluded
):
    """
    Keep exclusion diagnostics and find a target in another source sentence.

    args:
        text: Sentence containing an excluded payment context.
        vector: Controlled similarities with a stronger exclusion match.
        excluded: Expected exclusion class.

    returns:
        None.
    """
    matcher = TargetMatcher(SETTINGS, PATTERNS, Encoder([vector, [1.0, 0.0]]))
    result = matcher.match(text + "\nZasądza 300 zł.")
    assert result[excluded]["text"] == text
    assert result["judgment"]["text"] == "Zasądza 300 zł."
    assert result["request"] is None


def test_exclusion_wins_an_exact_tie_instead_of_admitting_a_target():
    """
    Prefer an exclusion class on an exact target/exclusion score tie.

    args:
        None.

    returns:
        None.
    """
    matcher = TargetMatcher(SETTINGS, PATTERNS, Encoder([[1, 0, 0, 1, 0]]))
    result = matcher.match("Zasądza 100 zł kosztów.")
    assert result["judgment"] is None
    assert result["procedural_costs"] is not None
