"""Load local predictors independently of optional supervised award targets."""

import json
from pathlib import Path


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
