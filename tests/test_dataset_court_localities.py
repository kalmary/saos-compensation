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


sys.path.insert(0, str(ROOT))


from src.dataset.acquisition import api
from src.dataset.extraction.source.text import (
    CURRENCY,
    MONEY,
    amount_in_pln,
    extract_text,
)
from src.dataset.storage import files as storage


@pytest.mark.parametrize("debug", [False, True])
def test_locality_download_is_retained_only_for_debug(monkeypatch, tmp_path, debug):
    """
    Verify locality download is retained only for debug.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    import sqlite3

    from src.dataset.acquisition import localities

    source = tmp_path / "source.gpkg"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE OT_ADMS_P (NAZWA, RODZAJ, LICZBAMIESZKANCOW, IDENTYFIKATORSIMC, IDENTYFIKATORTERC, IDENTYFIKATORPRNG, WERSJA)"
        )
        connection.execute(
            "INSERT INTO OT_ADMS_P VALUES ('Pisz', 'miasto', 19000, '1', '2', '3', '2026')"
        )
    content = source.read_bytes()

    class Response:
        def __enter__(self):
            """
            Provide the   enter   behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            self.content = content
            return self

        def __exit__(self, *args):
            """
            Provide the   exit   behavior used by this operation.

            args:
                args: Input for this operation.

            returns:
                Result of this operation.
            """
            pass

        def raise_for_status(self):
            """
            Provide the raise for status behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            pass

    monkeypatch.setattr(localities.requests, "get", lambda *args, **kwargs: Response())
    directory = tmp_path / "debug"
    rows, metadata = localities.load_localities(directory if debug else None)
    assert rows[0]["NAZWA"] == "Pisz"
    assert metadata["population_year"] is None
    assert directory.exists() == debug
