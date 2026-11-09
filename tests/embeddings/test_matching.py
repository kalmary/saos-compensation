"""Embedding routing, winner selection and multi-person principal aggregation."""

import math

import pytest


torch = pytest.importorskip("torch")


from dataset.embeddings.target_matching import (
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
            vectors: Source vectors in batch order or keyed by normalized text.

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
        if len(self.calls) == 1:
            return torch.eye(len(PATTERN_CLASSES))
        vectors = (
            [self.vectors[text] for text in texts]
            if isinstance(self.vectors, dict)
            else self.vectors
        )
        return torch.tensor([
            vector + [0.0] * (len(PATTERN_CLASSES) - len(vector))
            for vector in vectors
        ])


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
        ("Żądał zasądzenia 200 zł; zasądzono 100 zł.", [0.70, 0, 0, 0, 0.71], "mixed"),
    ],
)
def test_exclusion_winner_does_not_become_target_and_keeps_other_candidates(
    text,
    vector,
    excluded,
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


def test_prefix_without_amount_grows_without_embedding_and_stops_after_match():
    """Encode the first monetary prefix and skip its suffix after acceptance.

    args:
        None.

    returns:
        None.
    """
    prefix = "Żąda zapłaty, po zakończeniu leczenia, kwoty 100 zł,"
    encoder = Encoder({prepare_text(prefix): [0, 1]})
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(
        prefix + " tytułem odszkodowania."
    )
    assert result["request"]["text"] == prefix
    assert result["judgment"] is None
    assert encoder.calls[1:] == [[prepare_text(prefix)]]


@pytest.mark.parametrize("initial_score", [0.6, 0.7])
def test_prefix_grows_only_to_the_right_after_no_strict_match(initial_score):
    """Extend a below-threshold prefix without dropping its beginning.

    args:
        initial_score: First-prefix score below or exactly at the strict threshold.

    returns:
        None.
    """
    prefix = "Zasądza 100 zł,"
    sentence = prefix + " tytułem zadośćuczynienia."
    encoder = Encoder({
        prepare_text(prefix): [initial_score, -math.sqrt(1 - initial_score**2)],
        prepare_text(sentence): [1, 0],
    })
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)
    assert result["judgment"]["text"] == sentence
    assert result["judgment"]["start"] == 0
    assert result["judgment"]["end"] == len(sentence)
    assert encoder.calls[1:] == [
        [prepare_text(prefix)], [prepare_text(sentence)],
    ]


def test_first_matching_prefix_prevents_a_later_higher_score_from_replacing_it():
    """Finish an accepted sentence before embedding its longer prefixes.

    args:
        None.

    returns:
        None.
    """
    prefix = "Zasądza 100 zł,"
    encoder = Encoder({prepare_text(prefix): [0.8, 0.6]})
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(
        prefix + " wraz z odsetkami od 50 zł."
    )
    assert result["judgment"]["text"] == prefix
    assert encoder.calls[1:] == [[prepare_text(prefix)]]


def test_exclusion_stops_its_sentence_without_stopping_other_sentences():
    """Prevent a cost suffix from becoming an award while retaining another award.

    args:
        None.

    returns:
        None.
    """
    prefix = "Koszty procesu wynoszą 100 zł,"
    award = "Zasądza 300 zł."
    encoder = Encoder({
        prepare_text(prefix): [0, 0, 0, 1],
        prepare_text(award): [1, 0],
    })
    text = prefix + " zasądza się ich zwrot.\n" + award
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)
    assert result["procedural_costs"]["text"] == prefix
    assert result["judgment"]["text"] == award
    assert encoder.calls[1:] == [[prepare_text(prefix), prepare_text(award)]]


def test_unmatched_prefixes_end_at_the_sentence_boundary():
    """Exhaust a sentence without crossing into the next sentence for context.

    args:
        None.

    returns:
        None.
    """
    prefix = "Zasądza 100 zł,"
    sentence = prefix + " po zakończeniu leczenia."
    encoder = Encoder({
        prepare_text(prefix): [0.6, 0.8],
        prepare_text(sentence): [0.6, 0.8],
    })
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(
        sentence + " Inny rachunek wynosi 200 zł."
    )
    assert all(match is None for match in result.values())
    assert encoder.calls[1:] == [
        [prepare_text(prefix)], [prepare_text(sentence)],
    ]


def test_recipient_list_before_comma_reaches_amount_parser(monkeypatch):
    """Retain named beneficiaries when the amount occurs after their comma-separated list.

    args:
        monkeypatch: Supply the controlled matcher result to the monetary parser.

    returns:
        None.
    """
    import asyncio

    from dataset.extraction.targets import target_extraction

    text = "Zasądza na rzecz A. B., C. D. kwoty po 100 zł."
    encoder = Encoder({prepare_text(text): [1, 0]})
    matches = TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)
    monkeypatch.setattr(target_extraction, "match_targets", lambda *args: matches)
    result = asyncio.run(
        target_extraction.extract_targets(None, text, {"embedding": SETTINGS})
    )
    assert matches["judgment"]["text"] == text
    assert result["confirmed"]
    assert result["injured_party_count"] == 2
    assert result["suma"] == "200"
