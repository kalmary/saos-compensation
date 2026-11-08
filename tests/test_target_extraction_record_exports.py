"""Target admission and multiple-recipient arithmetic without model downloads."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.dataset.extraction.model import configuration as config_module
from src.dataset.extraction.model import decisions
from src.dataset.extraction.targets import amounts
from src.dataset.extraction.targets import target_extraction as extraction


@pytest.mark.parametrize(
    ("award", "expected"),
    [
        ({}, 1),
        ({"injured_party_count": None}, 1),
        ({"injured_party_count": "not_available"}, 1),
        ({"injured_party_count": "brak informacji"}, 1),
        ({"injured_party_count": 1}, 1),
        ({"injured_party_count": 2}, 2),
    ],
)
def test_injured_party_count_is_a_numeric_feature(award, expected):
    """
    Default missing recipient counts to one without replacing known counts.

    args:
        award: Resolved or missing recipient count.
        expected: Exported numeric predictor value.

    returns:
        None.
    """
    from src.dataset.storage.records import add_metadata_features

    result = {
        "features": {},
        "evidence": {},
        "confidence": {},
        "field_status": {},
        "award": award,
        "instance": {"value": "pierwsza", "decision": {"confidence": 0.9}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd w Warszawie"}},
    }
    add_metadata_features(result, record, ["injured_party_count"], None)
    assert result["features"] == {"injured_party_count": expected}
    assert type(result["features"]["injured_party_count"]) is int


@pytest.mark.parametrize("prior_payment", [0, 34400])
def test_prior_payment_is_exported_only_in_record(tmp_path, prior_payment):
    """
    Persist the integer feature without adding it to target fields or their total.

    args:
        tmp_path: Isolated output directory.
        prior_payment: Earlier principal feature including the absent-value default.

    returns:
        None.
    """
    import json

    from src.dataset.storage.records import add_metadata_features, write_result

    config = config_module.load_config(overrides={"debug": True})
    result = {
        "features": {},
        "evidence": {},
        "confidence": {},
        "field_status": {},
        "prior_payment": prior_payment,
        "award": {"confirmed": False, "zadoscuczynienie": "65600", "suma": "65600"},
        "instance": {"value": "pierwsza", "decision": {"confidence": 0.9}},
        "source_text": "Source.",
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd w Warszawie"}},
    }
    add_metadata_features(result, record, ["prior_payment"])
    write_result(result, tmp_path, config)
    features = json.loads((tmp_path / "record.json").read_text())
    award = json.loads((tmp_path / "award.json").read_text())
    assert features == {"prior_payment": prior_payment}
    assert type(features["prior_payment"]) is int
    assert "prior_payment" not in award
    assert award["total"] == "65600"
