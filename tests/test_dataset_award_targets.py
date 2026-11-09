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




from dataset.clients import api
from dataset.extraction.model import configuration as config_module
from dataset.extraction.model import decisions, prompts
from dataset.extraction.source.text import (
    CURRENCY,
    MONEY,
    amount_in_pln,
    extract_text,
)
from dataset.extraction.targets import amounts
from dataset.storage import files as storage


@pytest.mark.parametrize(
    "text, expected",
    [
        ("4 mln zł", "4000000"),
        ("1,5 mln zł", "1500000"),
        ("0,125 mln zł", "125000"),
        ("4mln zł", "4000000"),
        ("4 mln. PLN", "4000000"),
        ("250 tys. zł", "250000"),
        ("2,75 tys złotych", "2750"),
        ("1 tysiąc zł", "1000"),
        ("2 tysiące zł", "2000"),
        ("5 tysięcy zł", "5000"),
        ("1 milion zł", "1000000"),
        ("4 miliony zł", "4000000"),
        ("5 milionów PLN", "5000000"),
        ("1,25 mld zł", "1250000000"),
        ("1 miliard zł", "1000000000"),
        ("2 miliardy zł", "2000000000"),
        ("5 miliardów zł", "5000000000"),
        ("250 TYS. PLN", "250000"),
        ("1\u00a0250,50 zł", "1250.50"),
        ("1\u202f250,50 zł", "1250.50"),
        ("1,23456 tys. zł", "1234.56"),
        ("4 mln 250 tys. zł", "4250000"),
        ("1 mln 250 zł", "1000250"),
        ("1 mld 2 mln 3 tys. 4,50 zł", "1002003004.5"),
    ],
)
def test_scaled_amounts_are_converted_without_losing_source_offsets(text, expected):
    """
    Verify scaled amounts are converted without losing source offsets.

    args:
        text: Normalized Polish source text.
        expected: Expected result.

    returns:
        None.
    """
    match = MONEY.fullmatch(text)
    assert match is not None
    assert amount_in_pln(match) == expected
    source = f"Zasądza {text} na rzecz powódki. Koszty rozlicza osobno."
    rows = list(amounts.amount_contexts(source))
    assert len(rows) == 1
    row = rows[0]
    assert row["kwota"] == expected
    assert source[row["amount_start"] : row["amount_end"]] == text
    assert row["marked"].count("<KWOTA>") == 1
    assert row["text"] == source[: source.index(" Koszty")]
