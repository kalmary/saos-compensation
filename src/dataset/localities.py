"""Official court-seat locality matching and population classes."""

import hashlib
import json
import re
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import requests

SOURCE = "https://opendata.geoportal.gov.pl/bdot10k/schemat2021/GPKG/OT_ADMS_P.gpkg"


def locality_class(kind, population):
    if kind == "wieś":
        return "village"
    if kind != "miasto" or population is None:
        return "not_available"
    if type(population) not in (int, float) or not 0 < population < float("inf"):
        return "not_available"
    for boundary, label in (
        (20000, "city_under_20000"),
        (100000, "city_20000_to_under_100000"),
        (200000, "city_100000_to_under_200000"),
        (500000, "city_200000_to_under_500000"),
    ):
        if population < boundary:
            return label
    return "city_at_least_500000"


def load_localities(directory=None, timeout=30):
    """Load the official snapshot; keep it on disk only in debug mode."""
    with tempfile.TemporaryDirectory() as temporary:
        path = (
            Path(directory) / "OT_ADMS_P.gpkg"
            if directory
            else Path(temporary) / "OT_ADMS_P.gpkg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = path.with_suffix(".meta.json")
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["url"] != SOURCE
                or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]
            ):
                raise ValueError("GUGiK snapshot integrity check failed")
        else:
            with requests.get(SOURCE, timeout=timeout) as response:
                response.raise_for_status()
                content = response.content
            path.write_bytes(content)
            metadata = {
                "url": SOURCE,
                "sha256": hashlib.sha256(content).hexdigest(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "population_year": None,
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            if connection.execute("pragma quick_check").fetchone()[0] != "ok":
                raise ValueError("Incomplete or damaged GUGiK GeoPackage")
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "select NAZWA, RODZAJ, LICZBAMIESZKANCOW, IDENTYFIKATORSIMC, "
                "IDENTYFIKATORTERC, IDENTYFIKATORPRNG, WERSJA from OT_ADMS_P "
                "where RODZAJ in ('miasto', 'wieś')"
            ).fetchall()
        return [dict(row) for row in rows], metadata


def match_locality(court_name, rows, morphology):
    # Use Morfeusz to match inflected court-seat names against official localities; ambiguous matches remain missing.
    location = re.search(r"\b(?:w|we)\s+([^\n]+)$", court_name)
    if not location:
        return None, "court name does not contain a seat"
    name = location[1].strip()
    exact = [row for row in rows if row["NAZWA"].casefold() == name.casefold()]
    if len(exact) == 1:
        return exact[0], "exact name"
    if len(exact) > 1:
        return None, "multiple localities with the same name"
    words = name.split()
    lemmas = []
    for word in words:
        forms = {word.casefold()}
        for _, _, interpretation in morphology.analyse(word):
            forms.add(interpretation[1].split(":")[0].casefold())
        lemmas.append(forms)
    matches = [
        row
        for row in rows
        if len(row["NAZWA"].split()) == len(words)
        and all(
            word.casefold() in forms
            for word, forms in zip(row["NAZWA"].split(), lemmas)
        )
    ]
    return (
        (matches[0], "unique inflection match using Morfeusz/SIMC")
        if len(matches) == 1
        else (None, f"candidate count: {len(matches)}")
    )
