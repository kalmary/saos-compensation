"""Validated SAOS dump access and acquisition settings."""

import hashlib
import json
from calendar import monthrange
from datetime import date, datetime, timezone
from pathlib import Path

import requests

from ..storage.files import write_bytes, write_json
from ..extraction.source.text import extract_text

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
