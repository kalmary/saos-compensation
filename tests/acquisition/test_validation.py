from datetime import date

import pytest

from dataset.acquisition.settings import search_settings
from dataset.acquisition.validation import validate_judgment


CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}

@pytest.mark.parametrize(
    "change",
    [
        {"max_age_years": True},
        {"max_age_years": 3000},
        {"max_pages": -1},
        {"debug": "true"},
        {"download_path": "../outside"},
        {"download_path": ""},
        {"download_path": "/tmp"},
        {"max_gib": -1},
        {"max_gib": True},
        {"max_gib": float("nan")},
        {"timeout": 0},
        {"timeout": float("inf")},
    ],
)
def test_invalid_config_is_rejected(change):
    """
    Verify invalid config is rejected.

    args:
        change: Configuration change under examination.

    returns:
        None.
    """
    with pytest.raises(ValueError):
        search_settings({**CONFIG, **change})


@pytest.mark.parametrize("defect", ["id", "date", "court", "case", "text", "type"])
def test_invalid_judgments_are_rejected(judgment_payload, defect):
    """
    Verify invalid judgments are rejected.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        defect: Malformed field under examination.

    returns:
        None.
    """
    payload = judgment_payload
    data = payload["data"]
    if defect == "id":
        data["id"] = True
    elif defect == "date":
        data["judgmentDate"] = "2010-01-01"
    elif defect == "court":
        data["division"]["court"]["id"] = False
    elif defect == "case":
        data["courtCases"] = []
    elif defect == "text":
        data["textContent"] = "<p> </p>"
    else:
        data["judgmentType"] = "REASONS"
    with pytest.raises(ValueError):
        validate_judgment(payload, 254930, date(2016, 10, 29), date(2026, 10, 29))
