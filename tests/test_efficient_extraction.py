"""Ruling-first targets and source-aligned factual selection."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.processing import basal_filtering as basal
from src.dataset.processing.text import MONEY, amount_in_pln


@pytest.mark.parametrize(
    ("text", "amount"),
    [
        ("65.600,00 (sześćdziesiąt pięć tysięcy sześćset) zł", "65600.00"),
        ("323,29 (trzysta dwadzieścia trzy 29/100) złotych", "323.29"),
    ],
)
def test_parenthesized_written_amount_keeps_numeric_value(text, amount):
    """
    Read numeric values separated from the currency by their written equivalent.

    args:
        text: Monetary source mention.
        amount: Expected decimal PLN value.

    returns:
        None.
    """
    assert amount_in_pln(MONEY.fullmatch(text)) == amount


@pytest.mark.parametrize(
    "heading",
    ["Sąd Okręgowy ustalił, co następuje:", "Sąd ustalił i zważył co następuje:"],
)
def test_factual_heading_preserves_offsets_and_excludes_party_requests(heading):
    """
    Select court findings with exact offsets rather than the preceding request.

    args:
        heading: Supported factual heading.

    returns:
        None.
    """
    text = f"WYROK\nUZASADNIENIE\nPowód żąda 500 zł.\n{heading}\nPowód złamał nogę.\nSąd zważył:\nKwota jest odpowiednia."
    selected = basal.feature_input(text, {})
    assert selected["text"].strip() == "Powód złamał nogę."
    assert text[selected["start"] : selected["end"]] == selected["text"]


@pytest.mark.parametrize("treatment_in_findings", [False, True])
def test_missing_fields_do_not_trigger_fallback(monkeypatch, treatment_in_findings):
    """
    Extract once from findings and leave missing fields unavailable without another pass.

    args:
        monkeypatch: Scoped model replacement.
        treatment_in_findings: Whether all requested fields occur in the primary section.

    returns:
        None.
    """
    text = "UZASADNIENIE\nŻądano zapłaty.\nSąd ustalił:\nPowód złamał nogę.\nSąd zważył:\nWymagał rehabilitacji."
    if treatment_in_findings:
        text = text.replace(
            "Powód złamał nogę.", "Powód złamał nogę. Wymagał rehabilitacji."
        )
    config = basal.load_config()
    config["fields"] = {"harm_type": "Obrażenia", "treatment_assistance": "Leczenie"}
    schemas = []

    async def ask(server, state, questions, config):
        """
        Supply only findings and fail if extraction reaches the reasoning section.

        args:
            server: Unused server.
            state: Extraction text or selected sentence.
            questions: Requested model decisions.
            config: Effective configuration.

        returns:
            Source spans or factual, safe and field-relevant review answers.
        """
        if "opis" in questions:
            assert "Sąd zważył" not in state
            fields = questions["opis"]["schema"]["properties"]
            schemas.append(set(fields))
            data, details = {}, {}
            for field in fields:
                value = (
                    "Powód złamał nogę."
                    if field == "harm_type"
                    else "Wymagał rehabilitacji."
                )
                present = value in state
                data[field] = value if present else None
                details[field] = {
                    "start": state.index(value) if present else None,
                    "end": state.index(value) + len(value) if present else None,
                    "confidence": 0.9,
                    "line_probability": 0.9,
                }
            return {"opis": {"data": data, "fields": details}}
        return {
            "sekcja": {"choice": "fakty", "confidence": 0.99},
            "wyciek": {"noul": 0.01},
            **{key: {"noul": 0.99} for key in questions if key.startswith("fit_")},
        }

    monkeypatch.setattr(basal, "ask", ask)
    result = asyncio.run(basal.extract_features(None, text, config, 1))
    assert schemas == [{"harm_type", "treatment_assistance"}]
    assert result["features"] == {
        "harm_type": "Powód złamał nogę.",
        "treatment_assistance": "Wymagał rehabilitacji."
        if treatment_in_findings
        else "brak informacji",
    }
    for quotes in result["evidence"].values():
        assert all(text[q["start"] : q["end"]] == q["text"] for q in quotes)


@pytest.mark.parametrize(
    ("text", "section", "probability", "leakage", "fit", "reason", "calls"),
    [
        ("Powód złamał nogę.", "fakty", 0.99, 0.01, 0.99, "accepted", 2),
        ("Powód złamał nogę.", "rozwazania", 0.99, 0.01, 0.99, "not factual", 2),
        ("Powód złamał nogę.", "fakty", 0.4, 0.01, 0.99, "factual confidence=0.400", 2),
        ("Powód złamał nogę.", "fakty", 0.99, 0.8, 0.99, "leakage=0.800", 2),
        ("Powód złamał nogę.", "fakty", 0.99, 0.01, 0.3, "field relevance=0.300", 2),
        ("Powód zapłacił 100 zł.", "fakty", 0.99, 0.01, 0.99, "contains currency", 1),
        (None, "fakty", 0.99, 0.01, 0.99, "no description extracted", 1),
    ],
)
def test_feature_logs_report_existing_decision_failures_without_extra_calls(
    monkeypatch, capsys, text, section, probability, leakage, fit, reason, calls
):
    """
    Explain missing features using existing extraction and review decisions only.

    args:
        monkeypatch: Scoped model replacement.
        capsys: Captured progress and rejection details.
        text: Extracted sentence or missing value.
        section: Factuality class returned by the review.
        probability: Factuality confidence.
        leakage: Leakage probability.
        fit: Field relevance probability.
        reason: Expected explanation in console output.
        calls: Expected number of unchanged model decisions.

    returns:
        None.
    """
    source = text or "Powód złamał nogę."
    config = basal.load_config()
    config["fields"] = {"harm_type": "Injury"}
    events = []

    async def ask(server, state, questions, config):
        """
        Supply controlled extraction and quality decisions.

        args:
            server: Unused inference server.
            state: Source text or review context.
            questions: Existing extraction or quality questions.
            config: Effective extraction configuration.

        returns:
            Controlled source span or review answer.
        """
        events.append(set(questions))
        if "opis" in questions:
            return {
                "opis": {
                    "data": {"harm_type": text},
                    "fields": {
                        "harm_type": {
                            "start": 0 if text else None,
                            "end": len(text) if text else None,
                            "confidence": 0.9,
                            "line_probability": 0.9,
                        }
                    },
                }
            }
        return {
            "sekcja": {"choice": section, "confidence": probability},
            "wyciek": {"noul": leakage},
            "fit_harm_type": {"noul": fit},
        }

    monkeypatch.setattr(basal, "ask", ask)
    result = asyncio.run(basal.extract_features(None, source, config, 42))
    assert reason in capsys.readouterr().out
    assert len(events) == calls
    if reason != "accepted":
        assert reason in result["field_status"]["harm_type"]
        assert result["features"]["harm_type"] == "brak informacji"
