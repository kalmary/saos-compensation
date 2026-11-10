"""Full-text denial screening before monetary candidate selection."""

import math

import pytest
import torch

from dataset.embeddings import target_matching
from dataset.extraction.model.configuration import load_config


class Encoder:
    """Supply controlled denial similarities without loading model weights."""

    def __init__(self, scores):
        """Store source scores and capture every encoding batch.

        args:
            scores: Source sentences mapped to their denial cosine similarities.

        returns:
            None.
        """
        self.scores = {target_matching.prepare_text(text): score for text, score in scores.items()}
        self.calls = []

    def encode(self, texts):
        """Encode orthogonal patterns and unit vectors for source sentences.

        args:
            texts: Prepared pattern templates or source sentences.

        returns:
            Float32 vectors with a residual dimension outside all pattern axes.
        """
        self.calls.append(texts)
        if len(self.calls) == 1:
            self.width = len(texts) + 1
            return torch.eye(self.width)[:-1]
        vectors = torch.zeros((len(texts), self.width))
        for index, text in enumerate(texts):
            score = self.scores[text]
            vectors[index, 0] = score
            vectors[index, -1] = math.sqrt(1 - score**2)
        return vectors


def make_matcher(scores, batch_size=2):
    """Build a denial matcher from the real pattern configuration and a fake encoder.

    args:
        scores: Sentence-to-denial-similarity mapping.
        batch_size: Maximum sentences evaluated before checking for early rejection.

    returns:
        Matcher, controlled encoder and CPU configuration.
    """
    config = load_config()
    config["embedding"].update(device="cpu", batch_size=batch_size)
    assert next(iter(config["target_patterns"])) == "judgment-denial"
    encoder = Encoder(scores)
    matcher = target_matching.TargetMatcher(
        config["embedding"], config["target_patterns"], encoder
    )
    return matcher, encoder, config


@pytest.mark.parametrize(
    "score,threshold,denied",
    [(0.69, 0.7, False), (0.7, 0.7, False), (0.71, 0.7, True), (0.9, 0.95, False)],
)
def test_denial_uses_strict_configured_similarity_threshold(score, threshold, denied):
    """Reject only similarities strictly exceeding the configured minimum.

    args:
        score: Controlled cosine similarity.
        threshold: Configured minimum cosine similarity.
        denied: Whether the denial gate should match.

    returns:
        None.
    """
    text = "Oddala powództwo w całości."
    matcher, _, _ = make_matcher({text: score})
    matcher.settings["minimum_cosine_similarity"] = threshold
    result = matcher.match_denial(text)
    assert (result is not None) is denied
    if denied:
        assert result["similarity"] == pytest.approx(score)


@pytest.mark.parametrize("first", ["Opis sprawy.", "Zasądza 100 zł."])
def test_denial_checks_full_sentences_without_money_or_keyword_filter(monkeypatch, first):
    """Stop after a money-free denial without generating any monetary candidates.

    args:
        monkeypatch: Scoped matcher cache and monetary candidate replacements.
        first: Background or award sentence preceding the denial.

    returns:
        None.
    """
    denial = "Żadne świadczenie nie przysługuje, wobec niewykazania szkody."
    text = first + "\n" + denial + "\nDalsze ustalenia."
    matcher, encoder, config = make_matcher({first: 0.2, denial: 0.9}, batch_size=1)
    monkeypatch.setattr(target_matching, "cached_matcher", lambda *args: matcher)
    monkeypatch.setattr(
        target_matching, "candidate_sentences",
        lambda *args: pytest.fail("Denied cases must not reach monetary selection"),
    )
    matches = target_matching.match_targets(text, config)
    selected = matches["judgment-denial"]
    assert selected["text"] == denial
    assert text[selected["start"]:selected["end"]] == denial
    assert selected["pattern"] == config["target_patterns"]["judgment-denial"][0]
    assert all(value is None for key, value in matches.items() if key != "judgment-denial")
    assert encoder.calls[1:] == [[first], [denial]]


def test_no_denial_checks_every_sentence_then_runs_monetary_selection(monkeypatch):
    """Scan headings, reasoning and nonmonetary sentences before normal extraction.

    args:
        monkeypatch: Scoped cache and monetary matching replacements.

    returns:
        None.
    """
    sentences = ["Zasądza 100 zł.", "UZASADNIENIE", "Pierwsze ustalenie.", "Drugie ustalenie."]
    text = sentences[0] + "\n" + sentences[1] + "\n" + " ".join(sentences[2:])
    matcher, encoder, config = make_matcher(dict.fromkeys(sentences, 0.3))
    expected = {**dict.fromkeys(target_matching.PATTERN_CLASSES), "judgment": {"text": sentences[0]}}
    monkeypatch.setattr(target_matching, "cached_matcher", lambda *args: matcher)

    def match(source):
        """Require the entire denial pass to finish before monetary matching.

        args:
            source: Full original judgment text.

        returns:
            Controlled monetary match mapping.
        """
        assert source == text
        assert encoder.calls[1:] == [
            [target_matching.prepare_text(row) for row in sentences[:2]], sentences[2:]
        ]
        return expected

    monkeypatch.setattr(matcher, "match", match)
    assert target_matching.match_targets(text, config) == expected


def test_denial_match_records_highest_sentence_and_pattern_in_batch():
    """Retain exact evidence for the strongest denial in an evaluated batch.

    args:
        None.

    returns:
        None.
    """
    sentences = ["Pierwszy opis.", "Oddala powództwo."]
    matcher, encoder, _ = make_matcher(dict(zip(sentences, [0.75, 0.95], strict=True)))
    result = matcher.match_denial(" ".join(sentences))
    assert result["text"] == sentences[1]
    assert result["similarity"] == pytest.approx(0.95)
    assert encoder.calls[1:] == [sentences]


def test_denial_pass_with_empty_text_does_not_encode_source():
    """Avoid empty source batches after pattern initialization.

    args:
        None.

    returns:
        None.
    """
    matcher, encoder, _ = make_matcher({})
    assert matcher.match_denial(" \n ") is None
    assert len(encoder.calls) == 1
