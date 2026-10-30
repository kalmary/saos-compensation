"""Download and filter judgments: uv run --extra extraction src/dataset/get_data.py."""

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

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.text import MONEY, extract_text

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.saos.org.pl/api"


def fetch_json(url, directory=None, session=None, timeout=30):
    """Read one response; keep original bytes only when diagnostics are enabled."""
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
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
    if not isinstance(config, dict) or set(config) != {
        "max_age_years",
        "max_pages",
        "debug",
        "download_path",
        "download_only",
        "timeout",
    }:
        raise ValueError(
            "api.yaml wymaga max_age_years, max_pages, debug, download_path, download_only i timeout"
        )
    if type(config["download_only"]) is not bool:
        raise ValueError("download_only musi być wartością true albo false")
    if type(config["timeout"]) not in (int, float) or not 0 < config["timeout"] < float(
        "inf"
    ):
        raise ValueError("timeout musi być dodatnią, skończoną liczbą")
    if type(config["debug"]) is not bool:
        raise ValueError("debug musi być wartością true albo false")
    years, limit = config["max_age_years"], config["max_pages"]
    if type(years) is not int or years <= 0 or type(limit) is not int or limit < 0:
        raise ValueError("max_age_years musi być dodatnie, a max_pages nieujemne")
    path = config["download_path"]
    if (
        not isinstance(path, str)
        or not path.strip()
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ValueError("download_path musi być względną ścieżką w katalogu projektu")
    end = today or date.today()
    if years >= end.year:
        raise ValueError("max_age_years wykracza poza obsługiwane daty")
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


def search_pages(config, directory=None, today=None, session=None):
    _, _, params = search_settings(config, today)
    page = 0
    while config["max_pages"] == 0 or page < config["max_pages"]:
        result = search_judgments(
            {**params, "pageNumber": page}, directory, session, config["timeout"]
        )
        yield page, result
        if not any(link["rel"] == "next" for link in result["links"]):
            break
        page += 1


def search_judgments(params, directory=None, session=None, timeout=30):
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


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def validate_judgment(payload, identifier, start, end):
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


def download(config):
    """Read full dump pages and select common-court compensation judgments."""
    start, end, params = search_settings(config)
    directory = ROOT / config["download_path"]
    debug = (
        directory / "debug" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        if config["debug"]
        else None
    )
    raw = debug / "raw" if debug else None
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "request": params,
        "pages": 0,
        "downloaded": 0,
        "accepted": 0,
        "duplicates": 0,
        "rejected": [],
        "filtered": 0,
        "complete": False,
    }
    seen = set()
    divisions = None
    try:
        with requests.Session() as session:
            for number, page in search_pages(config, raw, end, session):
                manifest["pages"] = number + 1
                for item in page["items"]:
                    identifier = item["id"]
                    if identifier in seen:
                        manifest["duplicates"] += 1
                        continue
                    seen.add(identifier)
                    manifest["downloaded"] += 1
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
                    if config["debug"]:
                        path = directory / str(identifier) / "debug"
                        write_json(path / "judgment.json", item)
                        write_json(path / "metadata.json", record)
                        (path / "source.txt").write_text(text + "\n", encoding="utf-8")
                    manifest["accepted"] += 1
                    yield record
        manifest["complete"] = True
    except Exception as error:
        manifest["error"] = str(error)
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        if debug:
            write_json(debug / "acquisition.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False), flush=True)


def add_metadata_features(result, record, fields, localities=None):
    court = record["division"]["court"]
    locality = re.search(r"\b(?:w|we)\s+(.+)$", court["name"])
    values = {
        "rok_wyroku": date.fromisoformat(record["judgmentDate"]).year,
        "instancja": result["instance"]["value"],
        "miejscowosc_sadu": locality[1].strip() if locality else "brak informacji",
    }
    if "wielkość miejscowości" in fields:
        from src.dataset.localities import locality_class, match_locality

        rows, morphology, metadata = localities
        matched, reason = match_locality(court["name"], rows, morphology)
        values["wielkość miejscowości"] = (
            locality_class(matched["RODZAJ"], matched["LICZBAMIESZKANCOW"])
            if matched
            else "brak informacji"
        )
    sources = {
        "rok_wyroku": {"field": "judgmentDate", "value": record["judgmentDate"]},
        "instancja": result["instance"],
        "miejscowosc_sadu": {"field": "division.court", "value": court},
    }
    if "wielkość miejscowości" in fields:
        sources["wielkość miejscowości"] = {
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
            if field == "instancja"
            else None
        )
        result["field_status"][field] = (
            "brak informacji"
            if values[field] == "brak informacji"
            else "do weryfikacji"
            if field == "instancja"
            else "dopasowano do bieżącego snapshotu GUGiK; brak roku pomiaru ludności"
            if field == "wielkość miejscowości"
            else "odczytano z metadanych SAOS"
        )


def write_result(result, directory, config):
    award = result["award"]
    write_json(
        directory / "record.json",
        {
            "id": result["id"],
            "features": result["features"],
            "targets": {field: award[field] for field in config["target_fields"]},
            "total": award.get("suma", "brak informacji"),
            "confirmed": award.get("confirmed", False),
            "status": result["status"],
            "target_status": award.get("status", "brak informacji"),
        },
    )
    if config["api"]["debug"]:
        from src.dataset import basal_filtering as basal

        debug = directory / "debug"
        basal.write_paragraphs(
            result, debug, config["system_one"]["minimum_probability"], True
        )
        write_json(debug / "config.json", config)


async def run(config, download_only=None):
    from src.dataset import basal_filtering as basal

    directory = ROOT / config["api"]["download_path"]
    download_only = (
        config["api"]["download_only"] if download_only is None else download_only
    )
    if download_only:
        for record in download(config["api"]):
            write_json(directory / str(record["id"]) / "source.json", record)
            print(f"Downloaded judgment {record['id']}", flush=True)
        return

    localities = None
    if "wielkość miejscowości" in config["metadata_fields"]:
        import morfeusz2
        from src.dataset.localities import load_localities

        rows, metadata = load_localities(
            directory / "debug" / "gugik" if config["api"]["debug"] else None,
            timeout=config["api"]["timeout"],
        )
        localities = rows, morfeusz2.Morfeusz(), metadata
    os.environ.setdefault("HF_HOME", str(ROOT / "tmp" / "huggingface"))
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
        "config": config,
        "engine_version": "2.4.1",
        "engine_arguments": vars(args),
        "processed": 0,
        "confirmed_awards": 0,
        "complete": False,
    }
    try:
        with closing(download(config["api"])) as records:
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
                write_result(result, directory / str(identifier), config)
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
            write_json(directory / "debug" / "filtering.json", summary)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-dir", help="Directory containing api.yaml and basal.yaml"
    )
    parser.add_argument(
        "--max-age-years",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum judgment age in years",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum dump pages; 0 means all pages",
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
        help="HTTP connect/read timeout in seconds",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Save sources, evidence and model decisions",
    )
    parser.add_argument(
        "--download-only",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Save source records without running Basal",
    )
    return parser.parse_args(arguments)


def main():
    from src.dataset.basal_filtering import load_config

    options = vars(parse_arguments())
    directory = options.pop("config_dir")
    asyncio.run(run(load_config(directory, options)))


if __name__ == "__main__":
    main()
