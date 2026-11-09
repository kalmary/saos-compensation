"""Read-only SQL access to the official locality GeoPackage."""

import sqlite3
from contextlib import closing
from pathlib import Path


def read_localities(path):
    """Read validated locality attributes and always close the SQLite connection.

    args:
        path: Existing official locality GeoPackage.

    returns:
        Locality attribute dictionaries for cities and villages.
    """
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        if connection.execute("pragma quick_check").fetchone()[0] != "ok":
            raise ValueError("Incomplete or damaged GUGiK GeoPackage")
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "select NAZWA, RODZAJ, LICZBAMIESZKANCOW, IDENTYFIKATORSIMC, "
            "IDENTYFIKATORTERC, IDENTYFIKATORPRNG, WERSJA from OT_ADMS_P "
            "where RODZAJ in ('miasto', 'wieś')"
        ).fetchall()
    return [dict(row) for row in rows]
