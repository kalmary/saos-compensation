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


from src.dataset.storage import files as storage

from src.dataset.storage import records

from src.dataset.acquisition import api

from src.dataset.extraction.targets import amounts

from src.dataset.extraction.model import configuration as config_module

from src.dataset.extraction.model import decisions

from src.dataset.extraction.model import prompts

from src.dataset.extraction.source.text import CURRENCY

from src.dataset.extraction.source.text import MONEY

from src.dataset.extraction.source.text import amount_in_pln

from src.dataset.extraction.source.text import extract_text

@pytest.mark.parametrize(
    "court_name, expected",
    [
        ("Sąd Okręgowy w Płocku", "Płocku"),
        ("Sąd Rejonowy dla Warszawy-Mokotowa w Warszawie", "Warszawie"),
        ("Sąd Rejonowy we Wrocławiu", "Wrocławiu"),
        ("Sąd Rejonowy", "not_available"),
    ],
)
def test_metadata_features_use_judgment_date_instance_and_court_seat(
    court_name, expected
):
    """
    Verify metadata features use judgment date instance and court seat.

    args:
        court_name: Source court name containing its location.
        expected: Expected result.

    returns:
        None.
    """
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "druga", "decision": {"confidence": 0.8}},
    }
    record = {
        "judgmentDate": "2019-12-31",
        "division": {"court": {"name": court_name, "type": "DISTRICT"}},
    }
    records.add_metadata_features(
        result, record, ["judgment_year", "instance", "court_seat"]
    )
    assert result["features"] == {
        "judgment_year": 2019,
        "instance": "druga",
        "court_seat": expected,
    }
    assert result["metadata_sources"]["judgment_year"]["value"] == "2019-12-31"
    assert result["confidence"]["instance"] == 0.8


def test_metadata_selection_and_unknown_instance_do_not_infer_from_court_level():
    """
    Verify metadata selection and unknown instance do not infer from court level.

    args:
        None

    returns:
        None.
    """
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "not_available", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Apelacyjny w Krakowie", "type": "APPEAL"}},
    }
    records.add_metadata_features(result, record, ["instance"])
    assert result["features"] == {"instance": "not_available"}
    assert result["field_status"] == {"instance": "not_available"}


def test_population_feature_uses_official_match_and_preserves_provenance():
    """
    Verify population feature uses official match and preserves provenance.

    args:
        None

    returns:
        None.
    """
    import morfeusz2

    row = {
        "NAZWA": "Warszawa",
        "RODZAJ": "miasto",
        "LICZBAMIESZKANCOW": 1800000,
        "IDENTYFIKATORSIMC": "0918123",
    }
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "not_available", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Okręgowy w Warszawie"}},
    }
    records.add_metadata_features(
        result,
        record,
        ["court_locality_size"],
        ([row], morfeusz2.Morfeusz(), {"population_year": None}),
    )
    assert result["features"] == {"court_locality_size": "city_at_least_500000"}
    assert result["metadata_sources"]["court_locality_size"]["locality"] == row


@pytest.mark.parametrize("debug", [False, True])
def test_export_contains_only_features_targets_and_requested_debug_files(
    tmp_path, debug
):
    """
    Keep diagnostic data out of the feature and target files.

    args:
        tmp_path: Isolated output directory.
        debug: Whether the two debug files are requested.

    returns:
        None.
    """
    config = config_module.load_config(overrides={"debug": debug})
    config["fields"] = {"damage": "Injury"}
    result = {
        "id": 42,
        "features": {"damage": "source sentence"},
        "source_text": "source sentence",
        "evidence": {"damage": [{"start": 0, "end": 15}]},
        "description_reviews": [
            {
                "start": 0,
                "end": 15,
                "answers": {"wyciek": {"noul": 0.01}, "fit_damage": {"noul": 0.9}},
            }
        ],
        "model_calls": [{"request": "input", "response": "output"}],
        "award": {
            "confirmed": True,
            "verification": {"noul": 0.99},
            "suma": "100",
            "evidence": ["private diagnostics"],
        },
    }
    records.write_result(result, tmp_path, config)
    assert json.loads((tmp_path / "record.json").read_text()) == {
        "damage": "source sentence"
    }
    assert json.loads((tmp_path / "award.json").read_text()) == {
        "total": "100",
        "valid": True,
    }
    assert {p.name for p in tmp_path.iterdir()} == (
        {"record.json", "award.json", "model_calls.json", "source.txt"}
        if debug
        else {"record.json", "award.json"}
    )


def test_debug_export_preserves_low_confidence_proposal_with_invalid_flag(tmp_path):
    """
    Keep proposed target values for debug inspection without labeling them valid.

    args:
        tmp_path: Isolated output directory.

    returns:
        None.
    """
    config = config_module.load_config(overrides={"debug": True})
    award = amounts.finish_award(
        {
            "suma": "6300",
        },
        False,
        {"noul": 0.4},
    )
    result = {"features": {}, "award": award, "source_text": "Source."}
    assert records.rejection_reason(result, config) is not None
    records.write_result(result, tmp_path, config)
    assert json.loads((tmp_path / "award.json").read_text()) == {
        "total": "6300",
        "valid": False,
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        "award.json",
        "record.json",
        "model_calls.json",
        "source.txt",
    }

