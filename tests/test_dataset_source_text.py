import asyncio
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]




from dataset.extraction.source.text import (
    CURRENCY,
    MONEY,
    amount_in_pln,
    extract_text,
)


@pytest.mark.parametrize(
    "text, amount",
    [
        ("15 000 zł", "15 000"),
        ("15.000,50 PLN", "15.000,50"),
        ("100 złotych", "100"),
        ("2 złote", "2"),
        ("1 złoty", "1"),
        ("100 pln", "100"),
        ("100\u00a0zł", "100"),
    ],
)
def test_pln_amount_formats(text, amount):
    """
    Verify pln amount formats.

    args:
        text: Normalized Polish source text.
        amount: Input for this operation.

    returns:
        None.
    """
    assert MONEY.fullmatch(text)["amount"] == amount
    assert CURRENCY.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "100 EUR",
        "100 euro",
        "100 USD",
        "100 GBP",
        "100 CHF",
        "€100",
        "$100",
        "£100",
        "Powódka doznała złamania nogi.",
    ],
)
def test_other_currencies_and_unrelated_words_are_not_matched(text):
    """
    Verify other currencies and unrelated words are not matched.

    args:
        text: Normalized Polish source text.

    returns:
        None.
    """
    assert not MONEY.search(text)
    assert not CURRENCY.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "4 mln EUR",
        "250 tys USD",
        "4 milionowe zł",
        "12.34 zł",
        "12 34 zł",
        "1,2,3 mln zł",
    ],
)
def test_invalid_amounts_do_not_produce_partial_matches(text):
    """
    Verify invalid amounts do not produce partial matches.

    args:
        text: Normalized Polish source text.

    returns:
        None.
    """
    assert not MONEY.search(text)


@pytest.mark.parametrize("text", ["1 tys. 2 mln zł", "1 mln 2 mln zł"])
def test_compound_amounts_require_descending_units(text):
    """
    Verify compound amounts require descending units.

    args:
        text: Normalized Polish source text.

    returns:
        None.
    """
    with pytest.raises(ValueError, match="largest to smallest"):
        amount_in_pln(MONEY.fullmatch(text))
