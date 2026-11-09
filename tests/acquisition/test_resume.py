from datetime import date
import json

import pytest
import requests

from dataset.acquisition import download as acquisition
from dataset.acquisition.settings import search_settings
from dataset.extraction.model import configuration as config_module
from dataset.storage import checkpoints
from dataset.storage import files as storage
from dataset.storage import records

CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}


@pytest.mark.parametrize("debug", [False, True])
def test_download_deduplicates_filters_and_reports_rejections(
    output_root,
    judgment_payload,
    monkeypatch,
    tmp_path,
    debug,
    capsys,
):
    """
    Verify download deduplicates filters and reports rejections.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.
        capsys: Captured console output.

    returns:
        None.
    """
    data = judgment_payload["data"]
    division = data["division"]
    valid = {**data, "division": {"id": division["id"]}}
    invalid = {**valid, "id": 254931, "textContent": None}
    irrelevant = {**valid, "id": 254932, "textContent": "<p>Inna sprawa.</p>"}
    monkeypatch.setattr(
        acquisition,
        "search_pages",
        lambda *args: iter(
            [(0, {"items": [valid, valid, invalid, irrelevant], "links": []})]
        ),
    )
    monkeypatch.setattr(
        acquisition, "load_divisions", lambda *args: {division["id"]: division}
    )
    config = config_module.load_config(overrides={**CONFIG, "debug": debug})
    checkpoint = checkpoints.load_checkpoint(config)
    records = []
    for record in acquisition.download(config["api"], checkpoint, "run"):
        records.append(record)
        storage.write_json(
            tmp_path / "tmp" / str(record["id"]) / "record.json",
            {
                "id": record["id"],
                "dataset_id": checkpoint["dataset_id"],
                "features": {},
                "targets": {},
            },
        )
    assert [record["id"] for record in records] == [254930]
    assert records[0]["division"] == division
    report = json.loads(capsys.readouterr().out)
    assert (
        report["complete"]
        and report["accepted"] == report["duplicates"] == report["filtered"] == 1
    )
    assert report["rejected"] == [{"id": 254931, "reason": "Missing judgment text"}]
    assert not (tmp_path / "tmp/debug").exists()
    assert checkpoint["complete"]


def test_partial_network_failure_is_visible(output_root, monkeypatch, tmp_path, capsys):
    """
    Verify partial network failure is visible.

    args:
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """

    def fail(*args):
        """
        Provide the fail behavior used by this operation.

        args:
            args: Input for this operation.

        returns:
            Result of this operation.
        """
        raise requests.Timeout("SAOS unavailable")

    monkeypatch.setattr(acquisition, "search_pages", fail)
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    storage.write_json(tmp_path / "tmp/checkpoint.json", checkpoint)
    with pytest.raises(requests.Timeout):
        list(acquisition.download(CONFIG, checkpoint, "run"))
    report = json.loads(capsys.readouterr().out)
    assert not report["complete"] and report["error"] == "SAOS unavailable"
    assert (tmp_path / "tmp/checkpoint.json").exists()
    assert not (tmp_path / "tmp/debug").exists()


def test_checkpoint_freezes_dates_and_allows_operational_overrides(
    output_root,
    monkeypatch,
    tmp_path,
):
    """
    Verify checkpoint freezes dates and allows operational overrides.

    args:
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    checkpoint["request"] = search_settings(CONFIG, date(2024, 2, 29))[2]
    storage.write_json(tmp_path / "tmp/checkpoint.json", checkpoint)
    changed = config_module.load_config(
        overrides={**CONFIG, "debug": True, "timeout": 5, "max_pages": 2, "max_gib": 5}
    )
    assert checkpoints.load_checkpoint(changed) == checkpoint
    changed["system_one"]["minimum_probability"] = 0.8
    with pytest.raises(ValueError, match="Extraction settings changed"):
        checkpoints.load_checkpoint(changed)


def test_resume_skips_committed_records_and_retries_only_unfinished_item(
    output_root,
    judgment_payload,
    monkeypatch,
    tmp_path,
    capsys,
):
    """
    Verify resume skips committed records and retries only unfinished item.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    data = judgment_payload["data"]
    division = data["division"]
    items = [
        {**data, "id": identifier, "division": {"id": division["id"]}}
        for identifier in (1, 2, 3)
    ]
    starts = []

    def pages(*args):
        """
        Provide the pages behavior used by this operation.

        args:
            args: Input for this operation.

        returns:
            Result of this operation.
        """
        starts.append(args[4])
        return iter([(0, {"items": items, "links": [{"rel": "next"}]})])

    monkeypatch.setattr(acquisition, "search_pages", pages)
    monkeypatch.setattr(
        acquisition, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = acquisition.download(config["api"], checkpoint, "first")
    assert next(stream)["id"] == 1
    storage.write_json(
        tmp_path / "tmp/1/record.json",
        {
            "id": 1,
            "dataset_id": checkpoint["dataset_id"],
            "features": {},
            "targets": {},
        },
    )
    assert next(stream)["id"] == 2
    stream.close()
    resumed = checkpoints.load_checkpoint(config)
    assert resumed["next_item"] == 1
    stream = acquisition.download(config["api"], resumed, "second")
    assert next(stream)["id"] == 2
    stream.close()
    assert starts == [0, 0]
    assert (tmp_path / "tmp/1/record.json").exists()
    capsys.readouterr()


def test_resume_recognizes_record_saved_before_checkpoint_advance(
    output_root,
    judgment_payload,
    monkeypatch,
    tmp_path,
    capsys,
):
    """
    Verify resume recognizes record saved before checkpoint advance.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    data = judgment_payload["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}
    monkeypatch.setattr(
        acquisition,
        "search_pages",
        lambda *args: iter([(0, {"items": [item], "links": []})]),
    )
    monkeypatch.setattr(
        acquisition, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = acquisition.download(config["api"], checkpoint, "first")
    assert next(stream)["id"] == data["id"]
    storage.write_json(
        tmp_path / "tmp" / str(data["id"]) / "record.json",
        {
            "id": data["id"],
            "dataset_id": checkpoint["dataset_id"],
            "features": {},
            "targets": {},
        },
    )
    stream.close()
    resumed = checkpoints.load_checkpoint(config)
    assert list(acquisition.download(config["api"], resumed, "second")) == []
    assert resumed["complete"]
    capsys.readouterr()


def test_changed_resume_page_is_rejected(output_root, monkeypatch, tmp_path, capsys):
    """
    Verify changed resume page is rejected.

    args:
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    checkpoint["page_ids"] = [1]
    monkeypatch.setattr(
        acquisition,
        "search_pages",
        lambda *args: iter([(0, {"items": [{"id": 2}], "links": []})]),
    )
    with pytest.raises(ValueError, match="page contents changed"):
        list(acquisition.download(config["api"], checkpoint, "run"))
    assert not checkpoint["complete"]
    capsys.readouterr()


def test_checkpoint_does_not_advance_without_saved_output(
    output_root,
    judgment_payload,
    monkeypatch,
    tmp_path,
    capsys,
):
    """
    Verify checkpoint does not advance without saved output.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    data = judgment_payload["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}
    monkeypatch.setattr(
        acquisition,
        "search_pages",
        lambda *args: iter([(0, {"items": [item], "links": []})]),
    )
    monkeypatch.setattr(
        acquisition, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = acquisition.download(config["api"], checkpoint, "run")
    next(stream)
    with pytest.raises(RuntimeError, match="completed record"):
        next(stream)
    assert checkpoints.load_checkpoint(config)["next_item"] == 0
    capsys.readouterr()


def test_resume_starts_at_next_page_after_network_failure(
    output_root,
    judgment_payload,
    monkeypatch,
    tmp_path,
    capsys,
):
    """
    Keep completed pages committed when fetching the next page fails.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped dependency replacements.
        tmp_path: Isolated output directory.
        capsys: Captured acquisition reports.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    data = judgment_payload["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}

    def pages(*args):
        """
        Yield one page before simulating a network interruption.

        args:
            args: Acquisition arguments including the starting page.

        returns:
            Iterator of the first page.
        """
        assert args[4] == 0
        yield 0, {"items": [item], "links": [{"rel": "next"}]}
        raise requests.Timeout("offline")

    monkeypatch.setattr(acquisition, "search_pages", pages)
    monkeypatch.setattr(
        acquisition, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = acquisition.download(config["api"], checkpoint, "first")
    next(stream)
    storage.write_json(
        tmp_path / "tmp" / str(data["id"]) / "record.json",
        {
            "id": data["id"],
            "dataset_id": checkpoint["dataset_id"],
            "features": {},
            "targets": {},
        },
    )
    with pytest.raises(requests.Timeout):
        next(stream)
    resumed = checkpoints.load_checkpoint(config)
    assert resumed["page"] == 1 and resumed["next_item"] == 0
    starts = []
    monkeypatch.setattr(
        acquisition,
        "search_pages",
        lambda *args: starts.append(args[4]) or iter([(1, {"items": [], "links": []})]),
    )
    assert list(acquisition.download(config["api"], resumed, "second")) == []
    assert starts == [1] and resumed["complete"]
    capsys.readouterr()


def test_rejected_record_advances_checkpoint_without_case_files(
    output_root,
    judgment_payload,
    monkeypatch,
    tmp_path,
    capsys,
):
    """
    Continue after explicit rejection without persisting case artifacts.

    args:
        judgment_payload: Fresh structurally valid SAOS source payload.
        output_root: Isolated acquisition output root.
        monkeypatch: Scoped dependency replacements.
        tmp_path: Isolated dataset directory.
        capsys: Captured progress reports.

    returns:
        None.
    """
    config = config_module.load_config(overrides=CONFIG)
    checkpoint = checkpoints.load_checkpoint(config)
    data = judgment_payload["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}
    monkeypatch.setattr(
        acquisition,
        "search_pages",
        lambda *args: iter([(0, {"items": [item], "links": []})]),
    )
    monkeypatch.setattr(
        acquisition, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = acquisition.download(config["api"], checkpoint, "run")
    record = next(stream)
    record["rejected"] = True
    assert list(stream) == []
    assert checkpoints.load_checkpoint(config)["complete"]
    assert not (tmp_path / "tmp" / str(data["id"])).exists()
    capsys.readouterr()
