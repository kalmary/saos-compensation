"""Dataset progress and saved-record compatibility checks."""

import hashlib
import json
from datetime import date
from pathlib import Path
from uuid import uuid4

from ..acquisition.settings import search_settings
from .files import write_json

ROOT = Path(__file__).resolve().parents[3]


def save_checkpoint(directory, checkpoint):
    """Persist dataset progress atomically within the output-size budget.

    args:
        directory: Dataset output directory.
        checkpoint: Validated dataset progress.

    returns:
        None.
    """
    write_json(Path(directory) / "checkpoint.json", checkpoint)


def load_checkpoint(config):
    """
    Load compatible progress or initialize a new dataset checkpoint.

    args:
        config: Effective pipeline or acquisition settings.

    returns:
        Compatible progress, or a new unpersisted checkpoint.
    """
    api = config["api"]
    _, _, request = search_settings(api)
    signature = hashlib.sha256(
        json.dumps(
            {key: value for key, value in config.items() if key != "api"},
            ensure_ascii=False,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path = ROOT / api["download_path"] / "checkpoint.json"
    if not path.exists():
        return {
            "dataset_id": uuid4().hex,
            "signature": signature,
            "request": request,
            "page": 0,
            "next_item": 0,
            "page_ids": None,
            "complete": False,
            "localities_sha256": None,
        }
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset_id",
        "signature",
        "request",
        "page",
        "next_item",
        "page_ids",
        "complete",
        "localities_sha256",
    }
    if not isinstance(checkpoint, dict) or set(checkpoint) != required:
        raise ValueError("Invalid checkpoint structure")
    if not isinstance(checkpoint["dataset_id"], str) or not checkpoint["dataset_id"]:
        raise ValueError("Invalid checkpoint dataset ID")
    if checkpoint["signature"] != signature:
        raise ValueError("Extraction settings changed; use a different download_path")
    for key in ("page", "next_item"):
        if type(checkpoint[key]) is not int or checkpoint[key] < 0:
            raise ValueError(f"Invalid checkpoint {key}")
    if type(checkpoint["complete"]) is not bool:
        raise ValueError("Invalid checkpoint completion state")
    ids = checkpoint["page_ids"]
    if checkpoint["complete"] and (ids is not None or checkpoint["next_item"] != 0):
        raise ValueError("Completed checkpoint has an unfinished page")
    if ids is not None and (
        not isinstance(ids, list)
        or any(type(value) is not int or value <= 0 for value in ids)
    ):
        raise ValueError("Invalid checkpoint page IDs")
    if (ids is None and checkpoint["next_item"] != 0) or (
        ids is not None and checkpoint["next_item"] > len(ids)
    ):
        raise ValueError("Checkpoint position exceeds the saved page")
    try:
        end = date.fromisoformat(checkpoint["request"]["judgmentEndDate"])
        expected = search_settings(api, end)[2]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid checkpoint request") from error
    if checkpoint["request"] != expected:
        raise ValueError(
            "Acquisition date settings changed; use a different download_path"
        )
    digest = checkpoint["localities_sha256"]
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64):
        raise ValueError("Invalid checkpoint locality snapshot hash")
    return checkpoint
