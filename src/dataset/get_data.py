"""Download and filter judgments"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from calendar import monthrange
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.processing.text import MONEY, extract_text
from src.dataset.processing.storage import StorageLimitError, output_limit, write_bytes, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.saos.org.pl/api"


def fetch_json(url, directory=None, session=None, timeout=30):
    """
    Fetch one SAOS response and optionally preserve its original bytes.

    args:
        url: Resource URL.
        directory: Input or output directory for this operation.
        session: Reusable HTTP session, when supplied.
        timeout: Connection/read timeout in seconds.

    returns:
        Decoded JSON response object.
    """
    if not url.startswith(BASE_URL + "/"):
        raise ValueError(f"Expected a SAOS API URL: {url}")
    client = session or requests
    with client.get(
        url, headers={"Accept": "application/json"}, timeout=timeout
    ) as response:
        response.raise_for_status()
        raw = response.content
        content_type = response.headers.get("Content-Type", "")
        if response.status_code != 200 or "json" not in content_type:
            raise ValueError(f"Unexpected JSON response: {url}")
    if directory is not None:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = Path(directory) / f"{key}.json"
        write_bytes(path, raw)
        write_json(
            path.with_suffix(".meta.json"),
            {
                "url": url,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "sha256": hashlib.sha256(raw).hexdigest(),
            },
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {url}")
    return payload


def search_settings(config, today=None):
    """
    Validate acquisition settings and derive inclusive judgment-date bounds.

    args:
        config: Effective pipeline or acquisition settings.
        today: Date used to anchor the acquisition range.

    returns:
        Start date, end date and effective query parameters.
    """
    if not isinstance(config, dict) or set(config) != {
        "max_age_years",
        "max_pages",
        "max_gib",
        "debug",
        "download_path",
        "timeout",
    }:
        raise ValueError(
            "api.yaml requires max_age_years, max_pages, max_gib, debug, download_path and timeout"
        )
    if type(config["max_gib"]) not in (int, float) or not 0 <= config[
        "max_gib"
    ] < float("inf"):
        raise ValueError("max_gib must be a nonnegative finite number")
    if type(config["timeout"]) not in (int, float) or not 0 < config["timeout"] < float(
        "inf"
    ):
        raise ValueError("timeout must be a positive finite number")
    if type(config["debug"]) is not bool:
        raise ValueError("debug must be true or false")
    years, limit = config["max_age_years"], config["max_pages"]
    if type(years) is not int or years <= 0 or type(limit) is not int or limit < 0:
        raise ValueError("max_age_years must be positive and max_pages nonnegative")
    path = config["download_path"]
    if (
        not isinstance(path, str)
        or not path.strip()
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ValueError("download_path must be relative to the project root")
    end = today or date.today()
    if years >= end.year:
        raise ValueError("max_age_years exceeds the supported date range")
    start = end.replace(
        year=end.year - years,
        day=min(end.day, monthrange(end.year - years, end.month)[1]),
    )
    params = {
        "pageSize": 100,
        "pageNumber": 0,
        "judgmentStartDate": start.isoformat(),
        "judgmentEndDate": end.isoformat(),
    }
    return start, end, params


def search_pages(
    config, directory=None, today=None, session=None, start_page=0, params=None
):
    """
    Yield dump pages starting at the saved pagination position.

    args:
        config: Effective pipeline or acquisition settings.
        directory: Input or output directory for this operation.
        today: Date used to anchor the acquisition range.
        session: Reusable HTTP session, when supplied.
        start_page: First page number to request.
        params: Effective SAOS query parameters.

    returns:
        Iterator of page-number/response pairs.
    """
    _, _, defaults = search_settings(config, today)
    params = defaults if params is None else params
    page = start_page
    while config["max_pages"] == 0 or page < config["max_pages"]:
        result = search_judgments(
            {**params, "pageNumber": page}, directory, session, config["timeout"]
        )
        yield page, result
        if not any(link["rel"] == "next" for link in result["links"]):
            break
        page += 1


def search_judgments(params, directory=None, session=None, timeout=30):
    """
    Fetch a dump page and validate its query, identifiers and pagination.

    args:
        params: Effective SAOS query parameters.
        directory: Input or output directory for this operation.
        session: Reusable HTTP session, when supplied.
        timeout: Connection/read timeout in seconds.

    returns:
        Result of this operation.
    """
    url = (
        requests.Request("GET", BASE_URL + "/dump/judgments", params=params)
        .prepare()
        .url
    )
    result = fetch_json(url, directory, session, timeout)
    if not isinstance(result.get("items"), list) or not isinstance(
        result.get("links"), list
    ):
        raise ValueError("Dump response is missing items or pagination links")
    template = result.get("queryTemplate")
    if not isinstance(template, dict):
        raise ValueError("Dump response is missing queryTemplate")
    for key, value in params.items():
        applied = template.get(key)
        if not isinstance(applied, dict) or applied.get("value") != value:
            raise ValueError(f"API did not apply {key}")
    ids = [
        item.get("id") if isinstance(item, dict) else None for item in result["items"]
    ]
    if any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError("Dump returned an invalid judgment ID")
    if len(ids) != len(set(ids)) or len(ids) > params["pageSize"]:
        raise ValueError("Dump returned duplicate IDs or an oversized page")
    relations = set()
    for link in result["links"]:
        if (
            not isinstance(link, dict)
            or link.get("rel")
            not in {"self", "next", "prev", "previous", "first", "last"}
            or not isinstance(link.get("href"), str)
        ):
            raise ValueError("Invalid pagination link")
        if link["rel"] in relations:
            raise ValueError("Duplicate pagination relation")
        relations.add(link["rel"])
    if "self" not in relations or ("next" in relations and not ids):
        raise ValueError("Invalid dump pagination")
    return result


def load_divisions(session, directory=None, timeout=30):
    """
    Build a division-to-court lookup from the official court dump.

    args:
        session: Reusable HTTP session, when supplied.
        directory: Input or output directory for this operation.
        timeout: Connection/read timeout in seconds.

    returns:
        Division records indexed by identifier.
    """
    divisions = {}
    page = 0
    while True:
        url = f"{BASE_URL}/dump/commonCourts?pageSize=100&pageNumber={page}"
        payload = fetch_json(url, directory, session, timeout)
        if not isinstance(payload.get("items"), list) or not isinstance(
            payload.get("links"), list
        ):
            raise ValueError("Invalid common-court dump")
        for court in payload["items"]:
            if (
                not isinstance(court, dict)
                or type(court.get("id")) is not int
                or court["id"] <= 0
                or not isinstance(court.get("name"), str)
                or not court["name"].strip()
                or not isinstance(court.get("divisions"), list)
                or court.get("type") not in {"DISTRICT", "REGIONAL", "APPEAL"}
            ):
                raise ValueError("Invalid court in dump")
            for division in court["divisions"]:
                if not isinstance(division, dict):
                    raise ValueError("Invalid division in court dump")
                identifier = division.get("id")
                if (
                    type(identifier) is not int
                    or identifier <= 0
                    or not isinstance(division.get("name"), str)
                    or identifier in divisions
                ):
                    raise ValueError("Invalid or duplicate division in dump")
                divisions[identifier] = {
                    **division,
                    "court": {key: court[key] for key in ("id", "name", "type")},
                }
        if any(not isinstance(link, dict) for link in payload["links"]):
            raise ValueError("Invalid court pagination link")
        if not any(link.get("rel") == "next" for link in payload["links"]):
            return divisions
        if not payload["items"]:
            raise ValueError("Empty common-court page before next link")
        page += 1


def validate_judgment(payload, identifier, start, end):
    """
    Validate source metadata and normalize the judgment text.

    args:
        payload: Decoded source or model payload.
        identifier: Stable judgment identifier.
        start: Beginning of the permitted source range.
        end: End of the permitted source range.

    returns:
        Validated metadata and normalized text.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(data, dict)
        or type(data.get("id")) is not int
        or data["id"] != identifier
    ):
        raise ValueError("Judgment ID does not match the dump item")
    try:
        judgment_date = date.fromisoformat(data["judgmentDate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid judgment date") from error
    if not start <= judgment_date <= end:
        raise ValueError("Judgment date is outside the requested range")
    if data.get("courtType") != "COMMON" or data.get("judgmentType") != "SENTENCE":
        raise ValueError("Judgment does not match the requested court or judgment type")
    division = data.get("division")
    if not isinstance(division, dict):
        raise ValueError("Missing division metadata")
    court = division.get("court")
    for name, value in (("division", division), ("court", court)):
        if (
            not isinstance(value, dict)
            or type(value.get("id")) is not int
            or value["id"] <= 0
            or not isinstance(value.get("name"), str)
            or not value["name"].strip()
        ):
            raise ValueError(f"Invalid {name} metadata")
    if court.get("type") not in {"DISTRICT", "REGIONAL", "APPEAL"}:
        raise ValueError("Invalid common court level")
    cases = data.get("courtCases")
    if (
        not isinstance(cases, list)
        or not cases
        or any(
            not isinstance(case, dict)
            or not isinstance(case.get("caseNumber"), str)
            or not case["caseNumber"].strip()
            for case in cases
        )
    ):
        raise ValueError("Missing court case number")
    html = data.get("textContent")
    if not isinstance(html, str) or not (text := extract_text(html)):
        raise ValueError("Missing judgment text")
    return data, text


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
    return (
        type(record.get("id")) is int
        and record.get("id") == identifier
        and record.get("dataset_id") == dataset_id
        and isinstance(record.get("features"), dict)
        and isinstance(record.get("targets"), dict)
    )


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
    debug = directory / "debug" / run_id if config["debug"] else None
    raw = debug / "raw" if debug else None
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
                config, raw, end, session, checkpoint["page"], params
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
                    if debug:
                        path = directory / str(identifier) / "debug" / run_id
                        write_json(path / "judgment.json", item)
                        write_json(path / "metadata.json", record)
                        write_bytes(path / "source.txt", (text + "\n").encode("utf-8"))
                    write_json(checkpoint_path, checkpoint)
                    manifest["accepted"] += 1
                    yield record
                    # Advance only after the consumer has durably saved the final record.
                    if not record_complete(
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
        if debug:
            try:
                write_json(debug / "acquisition.json", manifest)
            except StorageLimitError:
                print(
                    "Output limit reached; acquisition report is available in the console.",
                    flush=True,
                )
        print(json.dumps(manifest, ensure_ascii=False), flush=True)


def add_metadata_features(result, record, fields, localities=None):
    """
    Add configured court and date features with their source provenance.

    args:
        result: Extraction result and supporting evidence.
        record: Normalized or saved judgment record.
        fields: Configured feature fields.
        localities: Official locality data and matching context.

    returns:
        None.
    """
    court = record["division"]["court"]
    locality = re.search(r"\b(?:w|we)\s+(.+)$", court["name"])
    values = {
        "judgment_year": date.fromisoformat(record["judgmentDate"]).year,
        "instance": "not_available"
        if result["instance"]["value"] == "brak informacji"
        else result["instance"]["value"],
        "court_seat": locality[1].strip() if locality else "not_available",
    }
    if "court_locality_size" in fields:
        from src.dataset.processing.localities import locality_class, match_locality

        rows, morphology, metadata = localities
        matched, reason = match_locality(court["name"], rows, morphology)
        values["court_locality_size"] = (
            locality_class(matched["RODZAJ"], matched["LICZBAMIESZKANCOW"])
            if matched
            else "not_available"
        )
    sources = {
        "judgment_year": {"field": "judgmentDate", "value": record["judgmentDate"]},
        "instance": result["instance"],
        "court_seat": {"field": "division.court", "value": court},
    }
    if "court_locality_size" in fields:
        sources["court_locality_size"] = {
            "locality": matched,
            "matching": reason,
            "source": metadata,
        }
    result["metadata_sources"] = {field: sources[field] for field in fields}
    for field in fields:
        result["features"][field] = values[field]
        result["evidence"][field] = []
        result["confidence"][field] = (
            result["instance"]["decision"]["confidence"]
            if field == "instance"
            else None
        )
        result["field_status"][field] = (
            "not_available"
            if values[field] == "not_available"
            else "review required"
            if field == "instance"
            else "matched to the current GUGiK snapshot; population year unknown"
            if field == "court_locality_size"
            else "read from SAOS metadata"
        )


def write_result(result, directory, config, run_id, dataset_id):
    """
    Save a primary record and optional run-specific diagnostic snapshots.

    args:
        result: Extraction result and supporting evidence.
        directory: Input or output directory for this operation.
        config: Effective pipeline or acquisition settings.
        run_id: Unique identifier for this execution.
        dataset_id: Identifier shared by records in this dataset.

    returns:
        None.
    """
    award = result["award"]
    record = {
        "id": result["id"],
        "dataset_id": dataset_id,
        "run_id": run_id,
        "features": {
            field: "not_available" if value == "brak informacji" else value
            for field, value in result["features"].items()
        },
        "targets": {
            field: "not_available"
            if award[field] == "brak informacji"
            else award[field]
            for field in config["target_fields"]
        },
        "total": "not_available"
        if award.get("suma", "brak informacji") == "brak informacji"
        else award["suma"],
        "confirmed": award.get("confirmed", False),
        "status": result["status"],
        "target_status": award.get("status", "not_available"),
    }
    if config["api"]["debug"]:
        from src.dataset.processing import basal_filtering as basal

        debug = directory / "debug" / run_id
        basal.write_paragraphs(
            result, debug, config["system_one"]["minimum_probability"], True
        )
        write_json(debug / "config.json", config)
        write_json(debug / "result.json", record)
    write_json(directory / "record.json", record)


def check_device(device):
    """
    Require the requested PyTorch backend to be available before inference.

    args:
        device: Explicit CPU or accelerator selection.

    returns:
        None.
    """
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PyTorch could not be loaded. Install --extra pytorch-macos on macOS or --extra extraction for CPU "
            "or --extra pytorch-cuda-linux for CUDA on Linux."
        ) from error
    if device == "cuda" and (
        torch.version.cuda is None or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable. On Linux, install --extra pytorch-cuda-linux "
            "and check the NVIDIA driver and GPU visibility, or set device: cpu."
        )

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable. Install --extra pytorch-macos on Apple Silicon, or set device: cpu."
        )


async def run(config):
    """
    Execute the configured pipeline within the cumulative output-size budget.

    args:
        config: Effective pipeline or acquisition settings.

    returns:
        None.
    """
    directory = ROOT / config["api"]["download_path"]
    try:
        with output_limit(directory, config["api"]["max_gib"]):
            await process(config)
    except StorageLimitError as error:
        print(
            f"{error}. Increase max_gib to continue from checkpoint.json.", flush=True
        )


async def process(config):
    """
    Resume acquisition and extract configured features and monetary targets.

    args:
        config: Effective pipeline or acquisition settings.

    returns:
        None.
    """
    from src.dataset.processing import basal_filtering as basal

    directory = ROOT / config["api"]["download_path"]
    checkpoint = load_checkpoint(config)
    limit = config["api"]["max_pages"]
    if checkpoint["complete"] or (limit and checkpoint["page"] >= limit):
        print("No pending judgments within the configured page limit.", flush=True)
        return
    model_cache = (
        Path(os.environ.get("HF_HOME", ROOT / "tmp" / "huggingface"))
        .expanduser()
        .resolve()
    )
    if config["api"]["max_gib"] and model_cache.is_relative_to(directory.resolve()):
        raise ValueError(
            "HF_HOME must be outside download_path when max_gib is enabled; use an output subdirectory such as tmp/dataset"
        )
    os.environ.setdefault("HF_HOME", str(model_cache))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    check_device(config["system_one"]["device"])
    write_json(directory / "checkpoint.json", checkpoint)
    localities = None
    if "court_locality_size" in config["metadata_fields"]:
        import morfeusz2

        from src.dataset.processing.localities import load_localities

        rows, metadata = load_localities(
            directory / "debug" / "gugik" if config["api"]["debug"] else None,
            timeout=config["api"]["timeout"],
        )
        digest = metadata["sha256"]
        if checkpoint["localities_sha256"] not in (None, digest):
            raise ValueError(
                "GUGiK snapshot changed since the checkpoint; use a different download_path"
            )
        checkpoint["localities_sha256"] = digest
        write_json(directory / "checkpoint.json", checkpoint)
        localities = rows, morfeusz2.Morfeusz(), metadata
    from basal.server import Server, parser
    from huggingface_hub import snapshot_download

    model = config["system_one"]
    model_path = snapshot_download(
        model["model"], revision=model["revision"], ignore_patterns=["think/*"]
    )
    args = parser().parse_args(
        [
            "--model",
            model_path,
            "--mode",
            model["mode"],
            "--device",
            model["device"],
            "--template-prefix",
            model["template_prefix"],
            "--name",
            model["model"],
            "--context-overflow",
            model["context_overflow"],
        ]
    )
    server = Server(args)
    worker = asyncio.create_task(server.worker())
    basal.REQUEST_COUNTS.update(cached=0, computed=0)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset_id": checkpoint["dataset_id"],
        "config": config,
        "engine_version": "2.4.1",
        "engine_arguments": vars(args),
        "processed": 0,
        "confirmed_awards": 0,
        "complete": False,
    }
    try:
        with closing(download(config["api"], checkpoint, run_id)) as records:
            for record in records:
                identifier = record["id"]
                token = basal.MODEL_CALLS.set([] if config["api"]["debug"] else None)
                try:
                    result = await basal.extract(
                        server, record["text"], config, identifier
                    )
                    if config["api"]["debug"]:
                        result["model_calls"] = basal.MODEL_CALLS.get()
                finally:
                    basal.MODEL_CALLS.reset(token)
                add_metadata_features(
                    result, record, config["metadata_fields"], localities
                )
                write_result(
                    result,
                    directory / str(identifier),
                    config,
                    run_id,
                    checkpoint["dataset_id"],
                )
                summary["processed"] += 1
                summary["confirmed_awards"] += int(
                    result["award"].get("confirmed", False)
                )
                print(f"Filtered judgment {identifier}", flush=True)
        summary["complete"] = True
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        summary["requests"] = basal.REQUEST_COUNTS.copy()
        if config["api"]["debug"]:
            try:
                write_json(directory / "debug" / run_id / "filtering.json", summary)
            except StorageLimitError:
                print(
                    "Output limit reached; filtering report is available in the console.",
                    flush=True,
                )
        print(
            json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"config", "engine_arguments"}
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


def parse_arguments(arguments=None):
    """
    Parse explicit CLI overrides without replacing YAML defaults.

    args:
        arguments: Command-line arguments, or None for process arguments.

    returns:
        Namespace containing explicit CLI overrides.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Explicit CLI options override api.yaml; model settings are in basal.yaml.",
    )
    parser.add_argument(
        "--config-dir",
        help="Directory with api.yaml and basal.yaml (default: configs/dataset)",
    )
    parser.add_argument(
        "--max-age-years",
        type=int,
        default=argparse.SUPPRESS,
        help="Judgment date range in years before today",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum dump pages (100 judgments each); 0 reads all pages",
    )
    parser.add_argument(
        "--max-gib",
        type=float,
        default=argparse.SUPPRESS,
        help="Total output-directory limit in GiB across resumed runs; 0 disables it",
    )
    parser.add_argument(
        "--download-path",
        default=argparse.SUPPRESS,
        help="Output directory relative to the project root",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=argparse.SUPPRESS,
        help="HTTP connection/read timeout in seconds, not a total download limit",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Save sources, evidence and model decisions",
    )
    return parser.parse_args(arguments)


def main():
    """
    Load effective settings and run the dataset pipeline.

    args:
        None

    returns:
        None.
    """
    from src.dataset.processing.basal_filtering import load_config

    options = vars(parse_arguments())
    directory = options.pop("config_dir")
    asyncio.run(run(load_config(directory, options)))


if __name__ == "__main__":
    main()
