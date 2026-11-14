"""Dataset admission boundaries without model inference."""

from dataset.extraction.quality import rejection_reason, target_rejection_reason


def test_admission_requires_confirmed_targets_and_safe_selected_evidence():
    """
    Reject weak awards and unsafe evidence at configured boundaries.

    args:
        None

    returns:
        None.
    """
    config = {
        "fields": {"damage": "Krzywda"},
        "target_fields": ["zadoscuczynienie"],
        "system_one": {
            "amount_minimum_probability": 0.6,
            "maximum_leak_probability": 0.2,
            "minimum_probability": 0.6,
        },
    }
    result = {
        "award": {
            "confirmed": True,
            "verification": {"noul": 0.6},
            "zadoscuczynienie": "100",
        },
        "features": {"damage": "Source description"},
        "evidence": {"damage": [{"start": 0, "end": 10}]},
        "description_reviews": [
            {
                "start": 0,
                "end": 10,
                "answers": {"wyciek": {"noul": 0.2}, "fit_damage": {"noul": 0.9}},
            }
        ],
    }
    assert rejection_reason(result, config) is None
    result["award"]["verification"]["noul"] = 0.59
    assert rejection_reason(result, config) is not None
    result["award"]["verification"]["noul"] = 0.6
    result["description_reviews"][0]["answers"]["wyciek"]["noul"] = 0.21
    assert rejection_reason(result, config) is not None
    result["description_reviews"][0]["answers"]["wyciek"]["noul"] = 0.2
    result["description_reviews"][0]["answers"]["fit_damage"]["noul"] = 0.59
    assert rejection_reason(result, config) is not None
    result["description_reviews"][0]["answers"]["fit_damage"]["noul"] = 0.6
    assert rejection_reason(result, config) is None


def test_confirmed_component_does_not_require_other_components_to_be_known():
    """
    Retain a confirmed award without interpreting missing components as zero.

    args:
        None.

    returns:
        None.
    """
    config = {
        "fields": {},
        "target_fields": ["zadoscuczynienie", "pozostale_odszkodowanie"],
        "system_one": {
            "amount_minimum_probability": 0.6,
            "maximum_leak_probability": 0.2,
            "minimum_probability": 0.6,
        },
    }
    award = {
        "confirmed": True,
        "verification": {"noul": 0.9},
        "zadoscuczynienie": "6300",
        "pozostale_odszkodowanie": "brak informacji",
    }
    assert target_rejection_reason(award, config) is None
    award["zadoscuczynienie"] = "brak informacji"
    assert target_rejection_reason(award, config) == "no established selected target"
    award["zadoscuczynienie"] = "NaN"
    assert target_rejection_reason(award, config) == "invalid target"


import pytest

from dataset.extraction.model import configuration as config_module


@pytest.mark.parametrize(
    ("confirmed", "verification", "expected"),
    [
        (
            False,
            {"noul": 0.42},
            "low-confidence award (probability=0.42, required >= 0.6)",
        ),
        (
            True,
            {"noul": 0.59},
            "low-confidence award (probability=0.59, required >= 0.6)",
        ),
        (False, {"noul": 0.8}, "unconfirmed award (probability=0.8, required >= 0.6)"),
        (False, {}, "unconfirmed award (probability=unavailable, required >= 0.6)"),
    ],
)
def test_award_rejection_reports_probability_and_threshold(
    confirmed,
    verification,
    expected,
):
    """
    Distinguish weak confirmation from unavailable verification without inventing a score.

    args:
        confirmed: Award confirmation flag.
        verification: Available model verification score.
        expected: Complete user-facing rejection reason.

    returns:
        None.
    """
    config = {"system_one": {"amount_minimum_probability": 0.6}}
    result = {"award": {"confirmed": confirmed, "verification": verification}}
    assert rejection_reason(result, config) == expected


@pytest.mark.parametrize("field", ["cause", "damage", "limitations"])
@pytest.mark.parametrize(
    "value", [None, "", " ", "brak informacji", "not_available", 123, "Source text"]
)
@pytest.mark.parametrize("evidence", [False, True])
def test_source_backed_damage_is_required(field, value, evidence):
    """Require damage even when another text feature has accepted evidence.

    args:
        field: Sole populated text field.
        value: Missing, malformed or usable description.
        evidence: Whether the description has source evidence.

    returns:
        None.
    """
    from dataset.extraction.quality import feature_rejection_reason

    result = {
        "features": {field: value, "judgment_year": 2020, "injured_party_count": 1},
        "evidence": {field: [{"start": 0, "end": 11}] if evidence else []},
    }
    config = {
        "fields": {name: "Question" for name in ("cause", "damage", "limitations")}
    }
    reason = feature_rejection_reason(result, config)
    assert (reason is None) is (field == "damage" and value == "Source text" and evidence)
    if reason is not None:
        assert reason == "no source-backed damage description"


def test_metadata_and_unselected_text_do_not_satisfy_admission():
    """Count only configured text predictors toward the minimum feature requirement.

    args:
        None.

    returns:
        None.
    """
    from dataset.extraction.quality import feature_rejection_reason

    result = {
        "features": {"judgment_year": 2020, "other": "Source text"},
        "evidence": {"other": [{"start": 0, "end": 11}]},
    }
    assert feature_rejection_reason(result, {"fields": {"cause": "Cause"}}) is not None


@pytest.mark.parametrize(("similarity", "accepted"), [(0.7, False), (0.7001, True)])
def test_target_admission_uses_strict_cosine_threshold(similarity, accepted):
    """
    Apply the same strict similarity boundary during matching and target admission.

    args:
        similarity: Winning judgment cosine score.
        accepted: Whether the score exceeds the configured threshold.

    returns:
        None.
    """
    from dataset.extraction.quality import target_rejection_reason

    config = config_module.load_config()
    award = {
        "target_matches": {"judgment": {"similarity": similarity}},
        "confirmed": True,
        "verification": {"noul": 0.9},
        "suma": "100",
    }
    assert (target_rejection_reason(award, config) is None) is accepted


def test_target_parse_failure_reports_status_without_model_probability():
    """Report selected-sentence ambiguity without obsolete model diagnostics.

    args:
        None.

    returns:
        None.
    """
    from dataset.extraction.quality import target_rejection_reason

    award = {
        "target_matches": {"judgment": {"similarity": 0.85}},
        "confirmed": False,
        "status": "selected judgment sentence requires exactly one numeric PLN amount",
    }
    assert (
        target_rejection_reason(award, config_module.load_config()) == award["status"]
    )
