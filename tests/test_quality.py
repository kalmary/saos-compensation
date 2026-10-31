"""Dataset admission boundaries without model inference."""

from src.dataset.processing.quality import rejection_reason


def test_admission_requires_confirmed_targets_and_safe_selected_evidence():
    """
    Reject weak awards and unsafe evidence at configured boundaries.

    args:
        None

    returns:
        None.
    """
    config = {
        "fields": {"harm_type": "Krzywda"},
        "target_fields": ["zadoscuczynienie"],
        "system_one": {
            "amount_minimum_probability": 0.6,
            "maximum_leak_probability": 0.2,
        },
    }
    result = {
        "award": {
            "confirmed": True,
            "verification": {"noul": 0.6},
            "zadoscuczynienie": "100",
        },
        "evidence": {"harm_type": [{"start": 0, "end": 10}]},
        "description_reviews": [
            {"start": 0, "end": 10, "answers": {"wyciek": {"noul": 0.2}}}
        ],
    }
    assert rejection_reason(result, config) is None
    result["award"]["verification"]["noul"] = 0.59
    assert rejection_reason(result, config) is not None
    result["award"]["verification"]["noul"] = 0.6
    result["description_reviews"][0]["answers"]["wyciek"]["noul"] = 0.21
    assert rejection_reason(result, config) is not None
