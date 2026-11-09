"""Cheap target-sentence filtering before model loading or embedding inference."""

import sys
from pathlib import Path

import pytest



@pytest.mark.parametrize("word", ["złodziej", "złoto", "złotnik", "złotawy"])
def test_currency_prefix_does_not_match_unrelated_words(word):
    """Reject unrelated words beginning with the currency prefix.

    args:
        word: Noncurrency word with a similar prefix.

    returns:
        None.
    """
    from dataset.extraction.source.text import CURRENCY, MONEY

    assert not CURRENCY.search(word)
    assert not MONEY.search(f"100 {word}")
