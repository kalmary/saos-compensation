import json

import pytest

from dataset.extraction.model import configuration as config_module
from dataset.extraction.targets import amounts
from dataset.storage import records


@pytest.mark.parametrize("debug", [False, True])
def test_export_contains_only_features_targets_and_requested_debug_files(
    tmp_path,
    debug,
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
