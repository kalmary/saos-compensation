"""Resumable acquisition of compensation judgments."""

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from .api import BASE_URL, search_pages, load_divisions, validate_judgment
from .checkpoints import record_complete
from ..storage.files import remove_tree, write_json
from ..extraction.source.text import MONEY, extract_text

ROOT = Path(__file__).resolve().parents[3]


def download(config, checkpoint, run_id):
    """
    Yield pending judgments and advance progress after completed output is saved.

    args:
        config: Effective pipeline or acquisition settings.
        checkpoint: Saved dataset progress and compatibility information.
        run_id: Unique identifier for this execution.

    returns:
        Iterator of pending validated judgments.
    """
    params = checkpoint["request"]
    start = date.fromisoformat(params["judgmentStartDate"])
    end = date.fromisoformat(params["judgmentEndDate"])
    directory = ROOT / config["download_path"]
    checkpoint_path = directory / "checkpoint.json"
    raw = None
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "request": params,
        "start_page": checkpoint["page"],
        "start_item": checkpoint["next_item"],
        "pages": 0,
        "downloaded": 0,
        "accepted": 0,
        "duplicates": 0,
        "saved_records": 0,
        "rejected": [],
        "filtered": 0,
        "complete": False,
    }
    seen = set()
    divisions = None
    try:
        with requests.Session() as session:
            for number, page in search_pages(
                config, None, end, session, checkpoint["page"], params
            ):
                ids = [item["id"] for item in page["items"]]
                if checkpoint["page_ids"] is not None and ids != checkpoint["page_ids"]:
                    raise ValueError(
                        "SAOS page contents changed since the checkpoint; use a different download_path"
                    )
                checkpoint["page_ids"] = ids
                write_json(checkpoint_path, checkpoint)
                manifest["pages"] += 1
                first = checkpoint["next_item"]
                for index, item in enumerate(page["items"]):
                    if index < first:
                        continue
                    identifier = item["id"]
                    checkpoint["next_item"] = index
                    manifest["downloaded"] += 1
                    if identifier in seen:
                        manifest["duplicates"] += 1
                        continue
                    seen.add(identifier)
                    record_path = directory / str(identifier) / "record.json"
                    if record_complete(
                        record_path, identifier, checkpoint["dataset_id"]
                    ):
                        remove_tree(record_path.parent / "pending")
                        manifest["saved_records"] += 1
                        continue
                    if (
                        item.get("courtType") != "COMMON"
                        or item.get("judgmentType") != "SENTENCE"
                    ):
                        manifest["filtered"] += 1
                        continue
                    html = item.get("textContent")
                    if not isinstance(html, str) or not (text := extract_text(html)):
                        manifest["rejected"].append(
                            {"id": identifier, "reason": "Missing judgment text"}
                        )
                        continue
                    if not re.search(r"zadośćuczyn", text, re.I) or not MONEY.search(
                        text
                    ):
                        manifest["filtered"] += 1
                        continue
                    division = item.get("division")
                    division_id = (
                        division.get("id") if isinstance(division, dict) else None
                    )
                    if type(division_id) is not int or division_id <= 0:
                        manifest["rejected"].append(
                            {"id": identifier, "reason": "Invalid division ID"}
                        )
                        continue
                    if divisions is None:
                        divisions = load_divisions(session, raw, config["timeout"])
                    data = {**item, "division": divisions.get(division_id, {})}
                    try:
                        data, text = validate_judgment(
                            {"data": data}, identifier, start, end
                        )
                    except ValueError as error:
                        manifest["rejected"].append(
                            {"id": identifier, "reason": str(error)}
                        )
                        continue
                    record = {
                        "id": identifier,
                        "judgmentDate": data["judgmentDate"],
                        "courtCases": data["courtCases"],
                        "division": data["division"],
                        "text": text,
                        "source_url": f"{BASE_URL}/judgments/{identifier}",
                    }
                    record["source_payload"] = item
                    write_json(checkpoint_path, checkpoint)
                    manifest["accepted"] += 1
                    yield record
                    # Advance only after the consumer has durably saved the final record.
                    if not record.get("rejected") and not record_complete(
                        record_path, identifier, checkpoint["dataset_id"]
                    ):
                        raise RuntimeError(
                            f"Cannot advance checkpoint without a completed record: {identifier}"
                        )
                    checkpoint["next_item"] = index + 1
                    write_json(checkpoint_path, checkpoint)
                checkpoint.update(
                    page=number + 1,
                    next_item=0,
                    page_ids=None,
                    complete=not any(link["rel"] == "next" for link in page["links"]),
                )
                write_json(checkpoint_path, checkpoint)
        manifest["complete"] = True
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(manifest, ensure_ascii=False), flush=True)
