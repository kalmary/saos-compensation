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
def test_only_missing_fields_use_fallback(monkeypatch, treatment_in_findings):
    """
    Extract once from findings, abstain on irrelevant fields and use fallback only for gaps.

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
        Supply findings first and support treatment only in the fallback stage.

        args:
            server: Unused server.
            state: Extraction text or selected sentence.
            questions: Requested model decisions.
            config: Effective configuration.

        returns:
            Source spans or factual, safe and field-relevant review answers.
        """
        if "opis" in questions:
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
    assert schemas == (
        [{"harm_type", "treatment_assistance"}]
        if treatment_in_findings
        else [{"harm_type", "treatment_assistance"}, {"treatment_assistance"}]
    )
    assert result["features"] == {
        "harm_type": "Powód złamał nogę.",
        "treatment_assistance": "Wymagał rehabilitacji.",
    }
    for quotes in result["evidence"].values():
        assert all(text[q["start"] : q["end"]] == q["text"] for q in quotes)
