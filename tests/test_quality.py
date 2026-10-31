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
            "minimum_probability": 0.6,
        },
    }
    result = {
        "award": {
            "confirmed": True,
            "verification": {"noul": 0.6},
            "zadoscuczynienie": "100",
        },
        "features": {"harm_type": "Source description"},
        "evidence": {"harm_type": [{"start": 0, "end": 10}]},
        "description_reviews": [
            {
                "start": 0,
                "end": 10,
                "answers": {"wyciek": {"noul": 0.2}, "fit_harm_type": {"noul": 0.9}},
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
    result["description_reviews"][0]["answers"]["fit_harm_type"]["noul"] = 0.59
    assert rejection_reason(result, config) is not None
    result["description_reviews"][0]["answers"]["fit_harm_type"]["noul"] = 0.6
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
    assert rejection_reason({"award": award}, config) is None
    award["zadoscuczynienie"] = "brak informacji"
    assert (
        rejection_reason({"award": award}, config) == "no established selected target"
    )
    award["zadoscuczynienie"] = "NaN"
    assert rejection_reason({"award": award}, config) == "invalid target"


import asyncio

import pytest

from src.dataset.processing import basal_filtering as basal


@pytest.mark.parametrize(
    ("confirmed", "debug", "probability", "continues"),
    [
        (False, False, 0.9, False),
        (True, False, 0.9, True),
        (False, False, 0.4, False),
        (False, True, 0.4, True),
        (False, True, 0.9, True),
    ],
)
def test_target_is_checked_before_any_feature_inference(
    monkeypatch, confirmed, debug, probability, continues
):
    """
    Stop rejected targets early and run accepted targets before instance inference.

    args:
        monkeypatch: Scoped function replacements.
        confirmed: Whether the extracted target meets admission rules.
        debug: Whether low-confidence proposals may be retained.
        probability: Award verification probability.
        continues: Whether feature inference should run.

    returns:
        None.
    """
    events = []
    award = basal.finish_award(
        {"zadoscuczynienie": "100", "pozostale_odszkodowanie": "brak informacji"},
        confirmed,
        {"noul": probability},
    )

    async def extract_award(server, text, diagnostics, config):
        """
        Supply a target without invoking the model.

        args:
            server: Unused inference server.
            text: Source judgment.
            diagnostics: Unused fragment diagnostics.
            config: Effective configuration.

        returns:
            Controlled award decision.
        """
        events.append("target")
        return award

    async def ask(server, state, questions, config):
        """
        Check that feature inference follows an accepted target.

        args:
            server: Unused inference server.
            state: Model input.
            questions: Requested model decisions.
            config: Effective configuration.

        returns:
            Instance decision.
        """
        assert continues
        assert events == ["target"]
        assert set(questions) == {"instancja"}
        events.append("instance")
        return {"instancja": {"choice": "nieustalona", "confidence": 0.9}}

    monkeypatch.setattr(basal, "extract_award", extract_award)
    monkeypatch.setattr(basal, "ask", ask)
    config = basal.load_config()
    config["fields"] = {}
    config["api"]["debug"] = debug
    result = asyncio.run(basal.extract(None, "Source judgment.", config, 42))
    assert result["award"] == award
    assert events == (["target", "instance"] if continues else ["target"])
    if not continues:
        assert result == {"id": 42, "award": award}


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
    confirmed, verification, expected
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


@pytest.mark.parametrize("found", [0, 2, 3, 4, 6])
def test_text_feature_coverage_requires_strict_majority_and_ignores_metadata(found):
    """
    Reject exactly half of six text fields despite populated numeric metadata.

    args:
        found: Number of source-backed text fields.

    returns:
        None.
    """
    from src.dataset.processing.quality import feature_rejection_reason

    fields = {f"field_{index}": "Description" for index in range(6)}
    result = {
        "features": {
            **{field: "Source text" for field in list(fields)[:found]},
            "judgment_year": 2020,
            "prior_payment": 30000,
        },
        "evidence": {
            field: [{"start": 0, "end": 11}] for field in list(fields)[:found]
        },
    }
    reason = feature_rejection_reason(result, {"fields": fields})
    if found > 3:
        assert reason is None
    else:
        assert reason == f"insufficient text features ({found}/6; required > 50%)"
