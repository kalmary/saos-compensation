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
        return "wies"
    if kind != "miasto" or population is None:
        return "brak informacji"
    if type(population) not in (int, float) or not 0 < population < float("inf"):
        return "brak informacji"
    for boundary, label in (
        (20000, "miasto_ponizej_20_tys"),
        (100000, "miasto_20_do_ponizej_100_tys"),
        (200000, "miasto_100_do_ponizej_200_tys"),
        (500000, "miasto_200_do_ponizej_500_tys"),
    ):
        if population < boundary:
            return label
    return "miasto_co_najmniej_500_tys"


def load_localities(directory=None):
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
                raise ValueError("Niezgodny snapshot GUGiK")
        else:
            with requests.get(SOURCE, timeout=60) as response:
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
                raise ValueError("Niekompletny lub uszkodzony GPKG GUGiK")
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "select NAZWA, RODZAJ, LICZBAMIESZKANCOW, IDENTYFIKATORSIMC, "
                "IDENTYFIKATORTERC, IDENTYFIKATORPRNG, WERSJA from OT_ADMS_P "
                "where RODZAJ in ('miasto', 'wieś')"
            ).fetchall()
        return [dict(row) for row in rows], metadata


def match_locality(court_name, rows, morphology):
    # Nazwa sądu jest zwykle w miejscowniku („w Piszu”). Morfeusz rozpoznaje
    # odmianę nazw, a kandydaci nadal pochodzą wyłącznie z urzędowej bazy.
    # Wieloznaczne dopasowanie zostaje brakiem; nie wybieramy największej miejscowości.
    location = re.search(r"\b(?:w|we)\s+([^\n]+)$", court_name)
    if not location:
        return None, "brak nazwy siedziby w nazwie sądu"
    name = location[1].strip()
    exact = [row for row in rows if row["NAZWA"].casefold() == name.casefold()]
    if len(exact) == 1:
        return exact[0], "dokładna nazwa"
    if len(exact) > 1:
        return None, "wiele miejscowości o tej nazwie"
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
        (matches[0], "jednoznaczna odmiana Morfeusz/SIMC")
        if len(matches) == 1
        else (None, f"liczba kandydatów: {len(matches)}")
    )
