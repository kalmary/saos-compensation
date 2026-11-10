"""Deterministic target-clause expansion with controlled cosine similarities."""

import asyncio
import math

import pytest
import torch

from dataset.embeddings.target_matching import PATTERN_CLASSES, TargetMatcher, prepare_text

SETTINGS = {
    "model": "test", "revision": "pinned", "device": "cpu", "batch_size": 8,
    "max_length": 128, "minimum_cosine_similarity": .7,
    "candidate_keywords": {
        "judgment": ["zasądz"], "request": ["żąda"], "shared": [],
        "prior_payment": ["wypłac"], "procedural_costs": ["koszt"],
        "mixed": [], "other_payments": ["odset"],
    },
}
PATTERNS = {label: [f"{label} {{amount}}"] for label in PATTERN_CLASSES}


class Encoder:
    """Map prepared source spans to controlled class similarities."""

    def __init__(self, scores):
        """Keep expected spans and a trace of embedding batches.

        args:
            scores: Mapping from prepared span to class/score mappings.

        returns:
            None.
        """
        self.scores = {prepare_text(text): value for text, value in scores.items()}
        self.calls = []

    def encode(self, texts):
        """Return orthogonal patterns or unit vectors with exact test similarities.

        args:
            texts: Prepared patterns or source spans in batch order.

        returns:
            Floating-point embedding matrix with a residual non-pattern dimension.
        """
        self.calls.append(texts)
        if len(self.calls) == 1:
            return torch.eye(len(PATTERN_CLASSES) + 1)[:-1]
        vectors = []
        for text in texts:
            scores = self.scores[text]
            vector = [scores.get(label, 0) for label in PATTERN_CLASSES]
            vector.append(math.sqrt(max(0, 1 - sum(value**2 for value in vector))))
            vectors.append(vector)
        return torch.tensor(vectors, dtype=torch.float32)


@pytest.mark.parametrize("label,verb", [("judgment", "Sąd zasądza"), ("request", "Powód żąda")])
def test_right_growth_accepts_money_with_lower_above_threshold_similarity(label, verb, monkeypatch):
    """Recover a decimal amount to the right while retaining exact source offsets.

    args:
        label: Expected target role.
        verb: Role-specific seed wording.
        monkeypatch: Scoped extraction matcher replacement.

    returns:
        None.
    """
    seed = f"{verb} świadczenie,"
    sentence = f"{seed} w kwocie 1 234,56 zł."
    text = "Nagłówek.\n" + sentence
    encoder = Encoder({prepare_text(seed): {label: .95}, prepare_text(sentence): {label: .8}})
    matcher = TargetMatcher(SETTINGS, PATTERNS, encoder)
    matches = matcher.match(text)
    selected = matches[label]
    assert selected["text"] == sentence
    assert selected["similarity"] == pytest.approx(.8)
    assert text[selected["start"]:selected["end"]] == sentence
    assert len(encoder.calls) == 3
    if label == "judgment":
        from dataset.extraction.targets import target_extraction
        monkeypatch.setattr(target_extraction, "match_targets", lambda *args: matches)
        result = asyncio.run(target_extraction.extract_targets(None, text, {"embedding": SETTINGS}))
        assert result["suma"] == "1234.56" and result["confirmed"] is True


def test_improving_money_free_growth_continues_until_amount():
    """Retain an improving intermediate clause before accepting a lower money score.

    args:
        None.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    middle = seed + " na rzecz powoda,"
    sentence = middle + " w kwocie 100 zł."
    encoder = Encoder({seed: {"judgment": .8}, middle: {"judgment": .95}, sentence: {"judgment": .75}})
    selected = TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)["judgment"]
    assert selected["text"] == sentence
    assert encoder.calls[2:] == [[middle], [sentence]]


def test_failed_right_growth_tries_left():
    """Use left monetary context after non-improving money-free right growth.

    args:
        None.

    returns:
        None.
    """
    seed = "sąd zasądza świadczenie,"
    right = seed + " opis bez kwoty."
    left = "Kwota 100 zł, " + seed
    text = left + " opis bez kwoty."
    encoder = Encoder({seed: {"judgment": .95}, right: {"judgment": .9}, left: {"judgment": .8}})
    selected = TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)["judgment"]
    assert selected["text"] == left
    assert encoder.calls[2:] == [[right], [left]]


def test_accepted_left_growth_retries_right_before_earlier_left_amount():
    """Retry right after helpful left context without absorbing a second amount.

    args:
        None.

    returns:
        None.
    """
    seed = "sąd zasądza świadczenie,"
    right = seed + " opis kwoty,"
    left = "dodatkowy opis, " + seed
    both = left + " opis kwoty,"
    final = both + " w wysokości 100 zł."
    text = "Kwota 300 zł, " + final
    encoder = Encoder({seed: {"judgment": .8}, right: {"judgment": .65}, left: {"judgment": .9},
                       both: {"judgment": .95}, final: {"judgment": .85}})
    selected = TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)["judgment"]
    assert selected["text"] == final
    assert "300" not in selected["text"]
    assert encoder.calls[2:] == [[right], [left], [both], [final]]


def test_left_growth_continues_when_no_right_comma_remains():
    """Recover an earlier amount through improving left-only context.

    args:
        None.

    returns:
        None.
    """
    seed = "sąd zasądza świadczenie."
    middle = "na rzecz powoda, " + seed
    text = "Kwota 100 zł, " + middle
    encoder = Encoder({seed: {"judgment": .8}, middle: {"judgment": .9}, text: {"judgment": .75}})
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)["judgment"]["text"] == text


def test_seed_priority_is_preserved_instead_of_ranking_expanded_scores():
    """Start from the highest-scoring seed even when its expanded score is lower.

    args:
        None.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    sentence = seed + " kwota 100 zł."
    other = "Zasądza 200 zł."
    encoder = Encoder({seed: {"judgment": .95}, other: {"judgment": .9}, sentence: {"judgment": .8}})
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence + "\n" + other)["judgment"]["text"] == sentence


@pytest.mark.parametrize("score", [.69, .7])
def test_monetary_growth_must_strictly_exceed_threshold(score):
    """Reject monetary extensions at or below the strict similarity threshold.

    args:
        score: Expanded candidate similarity.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    sentence = seed + " kwota 100 zł."
    encoder = Encoder({seed: {"judgment": .95}, sentence: {"judgment": score}})
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)["judgment"] is None


def test_expansion_rechecks_exclusions_and_preserves_tie_priority():
    """Prevent an expanded costs context from becoming an award on an exact tie.

    args:
        None.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    costs = "koszty procesu 100 zł."
    sentence = seed + " " + costs
    encoder = Encoder({seed: {"judgment": .95}, costs: {"procedural_costs": .9},
                       sentence: {"judgment": .705, "procedural_costs": .705}})
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)
    assert result["judgment"] is None
    assert result["procedural_costs"]["text"] == costs


@pytest.mark.parametrize("boundary", [". ", "! ", "? ", "\n"])
def test_growth_never_borrows_money_from_another_sentence_or_paragraph(boundary):
    """Keep irrelevant neighboring amounts from causing embedding inference.

    args:
        boundary: Full sentence or paragraph boundary.

    returns:
        None.
    """
    encoder = Encoder({})
    text = "Sąd zasądza świadczenie" + boundary + "Kwota 100 zł."
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)["judgment"] is None
    assert len(encoder.calls) == 1


def test_money_free_seed_and_intermediate_growth_can_start_below_threshold():
    """Apply the admission threshold to monetary results rather than incomplete seeds.

    args:
        None.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    middle = seed + " na rzecz powoda,"
    sentence = middle + " w kwocie 100 zł."
    encoder = Encoder({seed: {"judgment": .6}, middle: {"judgment": .65}, sentence: {"judgment": .8}})
    selected = TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)["judgment"]
    assert selected["text"] == sentence and selected["similarity"] > .7


@pytest.mark.parametrize("tail", [" i 200 zł.", "; koszty procesu 200 zł.", " oraz 200 zł i 300 zł."])
def test_right_fragment_with_multiple_amounts_is_cut_before_embedding(tail, monkeypatch):
    """Select the first amount in an appended fragment before scoring or parsing.

    args:
        tail: Additional amounts without a separating comma.
        monkeypatch: Scoped extraction matcher replacement.

    returns:
        None.
    """
    from dataset.extraction.targets import target_extraction

    seed = "Sąd zasądza świadczenie,"
    selected_text = seed + " kwota 1 234,56 zł"
    sentence = selected_text + tail
    text = "Nagłówek.\n" + sentence
    scores = {seed: {"judgment": .9}, selected_text: {"judgment": .8}}
    if "koszty" in tail:
        scores["kwota 1 234,56 zł" + tail] = {"procedural_costs": .75}
    encoder = Encoder(scores)
    matches = TargetMatcher(SETTINGS, PATTERNS, encoder).match(text)
    result = matches["judgment"]
    assert result["text"] == selected_text
    assert text[result["start"]:result["end"]] == selected_text
    assert encoder.calls[-1] == [prepare_text(selected_text)]
    monkeypatch.setattr(target_extraction, "match_targets", lambda *args: matches)
    award = asyncio.run(target_extraction.extract_targets(None, text, {"embedding": SETTINGS}))
    assert award["confirmed"] is True and award["suma"] == "1234.56"


@pytest.mark.parametrize("score,accepted", [(.69, False), (.7, False), (.8, True)])
def test_threshold_is_applied_to_the_trimmed_span(score, accepted):
    """Score the truncated monetary prefix rather than the full multi-amount clause.

    args:
        score: Similarity of the truncated span.
        accepted: Whether the strict monetary threshold is exceeded.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    prefix = seed + " kwota 100 zł"
    encoder = Encoder({seed: {"judgment": .95}, prefix: {"judgment": score}})
    result = TargetMatcher(SETTINGS, PATTERNS, encoder).match(prefix + " oraz 200 zł.")
    assert (result["judgment"] is not None) is accepted
    assert encoder.calls[-1] == [prefix]


def test_single_amount_extension_keeps_its_following_context():
    """Keep the original full fragment when no second amount requires truncation.

    args:
        None.

    returns:
        None.
    """
    seed = "Sąd zasądza świadczenie,"
    sentence = seed + " kwota 100 zł tytułem zadośćuczynienia."
    encoder = Encoder({seed: {"judgment": .9}, sentence: {"judgment": .8}})
    assert TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)["judgment"]["text"] == sentence


def test_left_multi_amount_growth_remains_untrimmed_and_ambiguous(monkeypatch):
    """Preserve left-side context and reject multiple amounts during target parsing.

    args:
        monkeypatch: Scoped extraction matcher replacement.

    returns:
        None.
    """
    from dataset.extraction.targets import target_extraction

    seed = "sąd zasądza świadczenie."
    sentence = "Kwota 100 zł oraz 200 zł, " + seed
    encoder = Encoder({seed: {"judgment": .9}, sentence: {"judgment": .8}})
    matches = TargetMatcher(SETTINGS, PATTERNS, encoder).match(sentence)
    assert matches["judgment"]["text"] == sentence
    monkeypatch.setattr(target_extraction, "match_targets", lambda *args: matches)
    award = asyncio.run(target_extraction.extract_targets(None, sentence, {"embedding": SETTINGS}))
    assert award["confirmed"] is False
    assert "exactly one" in award["status"]
