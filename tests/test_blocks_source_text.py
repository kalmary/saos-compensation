"""Bounded classification inputs without model inference."""

import asyncio
import sys
from pathlib import Path

import pytest



from dataset.extraction.source import passages


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
    blocks = list(passages.source_blocks(text, limit))
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
