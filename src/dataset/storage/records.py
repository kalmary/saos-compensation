"""Local feature/target records, completeness checks and loading."""

import json
from pathlib import Path

from ..extraction.quality import rejection_reason
from .files import remove_tree, write_bytes, write_json


def save_pending(directory, record, config):
    """Persist the source and configuration before starting judgment inference.

    args:
        directory: Dataset output directory.
        record: Validated judgment with its original source payload.
        config: Effective extraction configuration.

    returns:
        Pending directory containing source and resumable model decisions.
    """
    pending = Path(directory) / str(record["id"]) / "pending"
    write_json(pending / "judgment.json", record["source_payload"])
    write_json(pending / "config.json", config)
    write_bytes(pending / "source.txt", (record["text"] + "\n").encode("utf-8"))
    return pending


def remove_pending(pending):
    """Remove finished working state and an empty rejected-case directory.

    args:
        pending: Per-judgment working directory.

    returns:
        None.
    """
    remove_tree(pending)
    if not any(pending.parent.iterdir()):
        pending.parent.rmdir()


def write_result(result, directory, config):
    """
    Write feature and target values, plus only the requested debug files.

    args:
        result: Extracted feature and award values.
        directory: Judgment output directory.
        config: Selected fields and diagnostic flag.

    returns:
        None.
    """
    award = result["award"]
    valid = rejection_reason(result, config) is None
    if config["api"]["debug"] and not valid:
        award = award.get("proposal", award)
    features = {
        field: "not_available" if value == "brak informacji" else value
        for field, value in result["features"].items()
    }
    targets = {
        "total": "not_available"
        if award.get("suma", "brak informacji") == "brak informacji"
        else award["suma"],
    }
    targets["valid"] = valid
    if config["api"]["debug"]:
        write_json(directory / "model_calls.json", result.get("model_calls", []))
        write_bytes(directory / "source.txt", result["source_text"].encode("utf-8"))
    write_json(directory / "award.json", targets)
    write_json(directory / "record.json", features)


def record_complete(path, identifier, dataset_id):
    """
    Check whether a saved record belongs to the current dataset.

    args:
        path: Path to the relevant input or output file.
        identifier: Stable judgment identifier.
        dataset_id: Identifier shared by records in this dataset.

    returns:
        True when a complete record matches the judgment and dataset.
    """
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"Invalid saved record: {path}")
    award_path = path.with_name("award.json")
    if "features" not in record and "dataset_id" not in record and award_path.exists():
        award = json.loads(award_path.read_text(encoding="utf-8"))
        return path.parent.name == str(identifier) and isinstance(award, dict)
    return (
        type(record.get("id")) is int
        and record.get("id") == identifier
        and record.get("dataset_id") == dataset_id
        and isinstance(record.get("features"), dict)
        and isinstance(record.get("targets"), dict)
    )


def load_records(directory, *, include_targets=False):
    """Load complete case folders without touching source or diagnostic files.

    args:
        directory: Dataset root containing SAOS-ID subdirectories.
        include_targets: Require award.json and retain only valid=true cases.

    returns:
        Raw predictor mappings, stable sample_ids and optional unscaled PLN targets.
        This reader never splits samples or fits preprocessing; callers select training
        rows before fitting feature and target preprocessors independently.
    """
    records, ids, targets = [], [], []
    for path in sorted(Path(directory).glob("*/record.json")):
        if include_targets:
            award = json.loads(path.with_name("award.json").read_text(encoding="utf-8"))
            if not isinstance(award, dict) or type(award.get("valid")) is not bool:
                raise ValueError(f"Invalid award flag: {path.parent.name}")
            if not award["valid"]:
                continue
            if "total" not in award:
                raise ValueError(f"Missing award total: {path.parent.name}")
            targets.append(award["total"])
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(f"Invalid predictor record: {path.parent.name}")
        ids.append(path.parent.name)
        records.append(record)
    if not records:
        raise ValueError("No eligible records found")
    return {"records": records, "sample_ids": ids, "targets": targets if include_targets else None}
