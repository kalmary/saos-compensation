"""Bounded classification inputs without model inference."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset.processing import basal_filtering as basal


@pytest.mark.parametrize(
    ("text", "limit"),
    [
        ("First sentence.\n\nSecond sentence.\nThird.", 33),
        ("a" * 45, 10),
        ("Long sentence with several words and no punctuation", 12),
        ("", 10),
    ],
)
def test_blocks_preserve_source_text_and_character_budget(text, limit):
    """
    Cover every non-whitespace character without oversized or overlapping blocks.

    args:
        text: Source text including short paragraphs or an overlong sentence.
        limit: Maximum block size.

    returns:
        None.
    """
    blocks = list(basal.classification_blocks(text, limit))
    previous_end = 0
    for block in blocks:
        assert previous_end <= block["start"] < block["end"]
        assert not text[previous_end : block["start"]].strip()
        assert block["text"] == text[block["start"] : block["end"]]
        assert len(block["text"]) <= limit
        previous_end = block["end"]
    assert not text[previous_end:].strip()
    if text.startswith("First"):
        assert blocks[0]["text"] == "First sentence.\n\nSecond sentence."


def test_short_paragraphs_share_one_classification_call(monkeypatch):
    """
    Classify adjacent short paragraphs together using original neighboring context.

    args:
        monkeypatch: Scoped inference replacement.

    returns:
        None.
    """
    calls = []

    async def ask(server, state, questions, config):
        """
        Record classification inputs and supply a factual decision.

        args:
            server: Unused local server.
            state: Source block and neighboring context.
            questions: Requested classification and leakage decisions.
            config: Effective extraction configuration.

        returns:
            Accepted factual classification.
        """
        calls.append(state)
        return {
            "sekcja": {"choice": "fakty", "confidence": 0.9},
            "wyciek": {"noul": 0.01},
        }

    monkeypatch.setattr(basal, "ask", ask)
    text = "Pierwsze zdanie.\nDrugie zdanie.\nTrzecie zdanie."
    config = basal.load_config()
    _, rows = asyncio.run(basal.classify_paragraphs(None, text, config, 1))
    assert len(calls) == 1
    assert calls[0]["oceniany_fragment"] == text
    assert rows[0]["unit"] == "block"
    assert rows[0]["start"] == 0
    assert rows[0]["end"] == len(text)


@pytest.mark.parametrize(
    ("section", "confidence", "leak", "accepted"),
    [
        ("fakty", 0.9, 0.01, True),
        ("rozwazania", 0.9, 0.01, False),
        ("fakty", 0.9, 0.9, False),
        ("fakty", 0.1, 0.01, False),
    ],
)
def test_direct_extraction_reviews_only_selected_source_sentences(
    monkeypatch, section, confidence, leak, accepted
):
    """
    Extract before classification and share source reviews across selected fields.

    args:
        monkeypatch: Scoped model replacements.
        section: Selected sentence's semantic role.
        confidence: Confidence in the semantic role.
        leak: Selected sentence's target leakage score.
        accepted: Whether the candidate should become a feature.

    returns:
        None.
    """
    text = "Powódka wymagała rehabilitacji.\nSąd uznał żądanie za zasadne."
    sentence = "Powódka wymagała rehabilitacji."
    config = basal.load_config()
    config["fields"] = {"harm_type": "Krzywda", "life_impact": "Wpływ na życie"}
    events = []

    async def extract_award(server, source, diagnostics, config):
        """
        Supply an accepted target independently of feature extraction.

        args:
            server: Unused server.
            source: Source judgment.
            diagnostics: Unused fragment diagnostics.
            config: Effective configuration.

        returns:
            Confirmed monetary target.
        """
        events.append("target")
        return {
            "confirmed": True,
            "verification": {"noul": 0.99},
            "zadoscuczynienie": "100",
            "pozostale_odszkodowanie": "brak informacji",
        }

    async def ask(server, state, questions, config):
        """
        Supply extracted source spans followed by an independent sentence review.

        args:
            server: Unused server.
            state: Extraction input or reviewed source sentence.
            questions: Requested decisions.
            config: Effective configuration.

        returns:
            Controlled instance, extraction or review response.
        """
        if "instancja" in questions:
            events.append("instance")
            return {"instancja": {"choice": "nieustalona", "confidence": 0.9}}
        if "opis" in questions:
            events.append("extract")
            assert sentence in state
            assert "Sąd uznał" in state
            return {
                "opis": {
                    "data": {field: sentence for field in config["fields"]},
                    "fields": {
                        field: {
                            "start": state.index(sentence),
                            "end": state.index(sentence) + len(sentence),
                            "confidence": 0.9,
                            "line_probability": 0.9,
                        }
                        for field in config["fields"]
                    },
                }
            }
        assert events == ["target", "instance", "extract"]
        assert state["oceniany_fragment"] == sentence
        events.append("review")
        return {
            "sekcja": {"choice": section, "confidence": confidence},
            "wyciek": {"noul": leak},
        }

    monkeypatch.setattr(basal, "extract_award", extract_award)
    monkeypatch.setattr(basal, "ask", ask)
    result = asyncio.run(basal.extract(None, text, config, 42))
    assert events == ["target", "instance", "extract", "review"]
    for field in config["fields"]:
        assert result["features"][field] == (
            sentence if accepted else "brak informacji"
        )
        assert bool(result["evidence"][field]) is accepted
        for quote in result["evidence"][field]:
            assert text[quote["start"] : quote["end"]] == sentence
    assert len(result["description_reviews"]) == 1
