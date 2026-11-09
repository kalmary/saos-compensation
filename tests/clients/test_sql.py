"""Read-only locality SQL boundaries using real temporary databases."""

import sqlite3

import pytest

from dataset.clients.sql import read_localities


def test_reads_locality_rows_without_changing_snapshot(locality_database):
    """Read special-character paths safely and leave source bytes unchanged.

    args:
        locality_database: Temporary official-schema locality database.

    returns:
        None.
    """
    content = locality_database.read_bytes()
    rows = read_localities(locality_database)
    assert [(row["NAZWA"], row["RODZAJ"]) for row in rows] == [
        ("Pisz", "miasto"), ("Wieś", "wieś")
    ]
    assert locality_database.read_bytes() == content


def test_missing_database_is_not_created(tmp_path):
    """Fail read-only access rather than creating an empty replacement database.

    args:
        tmp_path: Isolated directory without a locality database.

    returns:
        None.
    """
    path = tmp_path / "missing.gpkg"
    with pytest.raises(sqlite3.OperationalError):
        read_localities(path)
    assert not path.exists()
