"""Ruling-first targets and source-aligned factual selection."""

import asyncio
import sys
from pathlib import Path

import pytest



from dataset.extraction.source import passages
from dataset.extraction.source.text import MONEY, amount_in_pln


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
    selected = passages.feature_input(text, {})
    assert selected["text"].strip() == "Powód złamał nogę."
    assert text[selected["start"] : selected["end"]] == selected["text"]
