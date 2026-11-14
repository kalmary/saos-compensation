import asyncio
import hashlib
import json

import pytest

from dataset.acquisition import pipeline
from dataset.extraction.model import configuration as config_module
from dataset.extraction.targets import amounts


@pytest.mark.parametrize("text_field", ["cause", "damage", "limitations"])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("low_confidence", [False, True])
@pytest.mark.parametrize("missing_target", [False, True])
@pytest.mark.parametrize("sparse_features", [False, True])
def test_pipeline_writes_separate_outputs_and_closes_extractor(
    output_root,
    fake_extractor,
    record_source,
    tmp_path,
    debug,
    low_confidence,
    missing_target,
    sparse_features,
    capsys,
    text_field,
):
    """
    Verify pipeline writes separate outputs and closes the injected extractor.

    args:
        output_root: Isolated pipeline output root.
        fake_extractor: Injected extraction context with cleanup tracking.
        record_source: One-judgment source iterator factory.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.
        low_confidence: Whether target verification falls below the threshold.
        missing_target: Whether no monetary target could be established.
        sparse_features: Whether missing text evidence must discard even debug cases.
        capsys: Captured console output.
        text_field: Sole populated text predictor.

    returns:
        None.
    """
    config = config_module.load_config()
    config["api"]["debug"] = debug
    config["api"]["max_gib"] = 0
    config["metadata_fields"] = ["judgment_year", "instance", "court_seat"]
    config["system_one"]["device"] = "cpu"
    config["embedding"]["device"] = "cpu"
    async def extract(text, identifier):
        """
        Extract source-grounded factual descriptions and separate monetary targets.

        args:
            text: Normalized Polish source text.
            identifier: Stable judgment identifier.

        returns:
            Feature and target results with review evidence.
        """
        await asyncio.sleep(0)
        return {
            "id": identifier,
            "features": {
                **{field: "brak informacji" for field in config["fields"]},
                text_field: "brak informacji" if sparse_features else text,
            },
            "evidence": {}
            if sparse_features
            else {text_field: [{"start": 0, "end": len(text)}]},
            "description_reviews": [
                {
                    "start": 0,
                    "end": len(text),
                    "answers": {
                        "wyciek": {"noul": 0.01},
                        f"fit_{text_field}": {"noul": 0.9},
                    },
                }
            ],
            "award": amounts.finish_award(
                {
                    "suma": "brak informacji" if missing_target else "100",
                },
                not low_confidence and not missing_target,
                {} if missing_target else {"noul": 0.4 if low_confidence else 0.9},
            ),
            "field_status": {},
            "confidence": {},
            "instance": {"value": "pierwsza", "decision": {"confidence": 0.9}},
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "status": "do przeglądu",
            "diagnostics": [],
        }

    fake_extractor.extract = extract
    asyncio.run(
        pipeline.run(config, extractor=fake_extractor.context, record_source=record_source)
    )
    output = capsys.readouterr().out
    if low_confidence and not missing_target:
        assert "low-confidence award" in output
    if missing_target:
        assert "final award verification not run" in output
    directory = tmp_path / "tmp"
    path = directory / "42"
    if text_field != "damage" or sparse_features or ((low_confidence or missing_target) and not debug):
        assert not path.exists()
        assert fake_extractor.stopped
        return
    record = json.loads((path / "record.json").read_text())
    assert record == {
        **{field: "not_available" for field in config["fields"]},
        text_field: "Opis.",
        "judgment_year": 2020,
        "instance": "pierwsza",
        "court_seat": "Płocku",
    }
    assert json.loads((path / "award.json").read_text()) == {
        "total": "not_available" if missing_target else "100",
        "valid": not low_confidence and not missing_target,
    }
    assert {item.name for item in path.iterdir()} == (
        {"record.json", "award.json", "model_calls.json", "source.txt"}
        if debug
        else {"record.json", "award.json"}
    )
    assert fake_extractor.stopped


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
def test_failed_extraction_keeps_pending_state_for_resume(
    output_root,
    fake_extractor,
    record_source,
    failure,
):
    """Preserve unfinished judgment evidence and reset scoped model state on failure.

    args:
        output_root: Isolated dataset output root.
        fake_extractor: Injected extraction context with cleanup tracking.
        record_source: Source factory yielding one judgment.
        failure: Processing error or interruption to propagate.

    returns:
        None.
    """
    from dataset.extraction.model import decisions

    config = config_module.load_config()
    config["api"].update(debug=False, max_gib=0)
    config["metadata_fields"] = []
    config["system_one"]["device"] = "cpu"
    config["embedding"]["device"] = "cpu"

    async def extract(text, identifier):
        """Interrupt inference after the pipeline has persisted pending source state.

        args:
            text: Judgment text.
            identifier: Judgment identifier.

        returns:
            None; raises the selected failure.
        """
        assert identifier == 42 and text == "Opis."
        assert decisions.DECISION_CACHE.get().name == "decisions"
        raise failure()

    fake_extractor.extract = extract
    with pytest.raises(failure):
        asyncio.run(
            pipeline.run(config, extractor=fake_extractor.context, record_source=record_source)
        )
    directory = output_root / config["api"]["download_path"]
    assert {path.name for path in (directory / "42/pending").iterdir()} == {
        "judgment.json", "config.json", "source.txt"
    }
    assert not (directory / "42/record.json").exists()
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    assert checkpoint["next_item"] == 0 and not checkpoint["complete"]
    assert fake_extractor.stopped
    assert decisions.DECISION_CACHE.get() is None
    assert decisions.MODEL_CALLS.get() is None


def test_completed_checkpoint_does_not_start_extraction(output_root, fake_extractor):
    """Leave completed datasets unchanged without checking devices or loading models.

    args:
        output_root: Isolated dataset output root.
        fake_extractor: Context that must remain unopened.

    returns:
        None.
    """
    from dataset.storage.checkpoints import load_checkpoint, save_checkpoint

    config = config_module.load_config()
    config["api"]["max_gib"] = 0
    checkpoint = load_checkpoint(config)
    checkpoint["complete"] = True
    directory = output_root / config["api"]["download_path"]
    save_checkpoint(directory, checkpoint)
    original = (directory / "checkpoint.json").read_bytes()
    asyncio.run(pipeline.run(config, extractor=fake_extractor.context))
    assert (directory / "checkpoint.json").read_bytes() == original
    assert not fake_extractor.stopped


@pytest.mark.parametrize("debug", [False, True])
def test_denied_case_saves_zero_with_damage(
    output_root, fake_extractor, record_source, capsys, debug
):
    """Save denied cases with valid zero targets and remove pending state.

    args:
        output_root: Isolated output root.
        fake_extractor: Controlled extraction context.
        record_source: One-judgment source iterator.
        capsys: Captured rejection summary.
        debug: Whether rejected debug attempts would otherwise be retained.

    returns:
        None.
    """
    config = config_module.load_config(overrides={"debug": debug})
    config["api"]["max_gib"] = 0
    config["metadata_fields"] = []
    config["system_one"]["device"] = config["embedding"]["device"] = "cpu"
    field = "damage"

    async def extract(text, identifier):
        """Return an explicitly denied case with an otherwise populated predictor.

        args:
            text: Original judgment text.
            identifier: Stable source identifier.

        returns:
            Zero-target result with safe damage evidence.
        """
        return {
            "id": identifier,
            "award": {
                "denied": True, "confirmed": True, "suma": "0",
                "target_matches": {"judgment-denial": {"similarity": 0.9}},
            },
            "instance": {"value": "pierwsza"},
            "features": {field: text},
            "evidence": {field: [{"start": 0, "end": len(text)}]},
            "description_reviews": [{
                "start": 0, "end": len(text),
                "answers": {"wyciek": {"noul": 0.01}, "fit_damage": {"noul": 0.9}},
            }],
        }

    fake_extractor.extract = extract
    asyncio.run(pipeline.run(config, extractor=fake_extractor.context, record_source=record_source))
    path = output_root / config["api"]["download_path"] / "42"
    assert json.loads((path / "award.json").read_text()) == {"total": "0", "valid": True}
    assert json.loads((path / "record.json").read_text()) == {"damage": "Opis."}
    assert not (path / "pending").exists()
    output = capsys.readouterr().out
    assert '"rejected_records": 0' in output
    assert fake_extractor.stopped
