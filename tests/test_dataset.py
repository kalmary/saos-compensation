import asyncio
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import get_data
from src.dataset.processing import basal_filtering as basal
from src.dataset.processing.text import CURRENCY, MONEY, amount_in_pln, extract_text

FIELD_NAMES = {
    "zdarzenie_i_okolicznosci": "event_circumstances",
    "rodzaj_doznanej_krzywdy": "harm_type",
    "przebieg_nasilenie_i_czas_trwania": "harm_severity_duration",
    "konsekwencje_dla_zycia": "life_impact",
    "indywidualne_okolicznosci_poszkodowanego": "claimant_circumstances",
    "podjete_dzialania_i_potrzebna_pomoc": "treatment_assistance",
}


def translate_fields(value):
    """
    Adapt historical fixture schema keys without modifying original artifacts.

    args:
        value: Value to serialize or adapt.

    returns:
        Value with historical schema keys adapted.
    """
    if isinstance(value, dict):
        return {
            FIELD_NAMES.get(key, key): translate_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [translate_fields(item) for item in value]
    return value


CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}


@pytest.mark.parametrize(
    "text, amount",
    [
        ("15 000 zł", "15 000"),
        ("15.000,50 PLN", "15.000,50"),
        ("100 złotych", "100"),
        ("2 złote", "2"),
        ("1 złoty", "1"),
        ("100 pln", "100"),
        ("100\u00a0zł", "100"),
    ],
)
def test_pln_amount_formats(text, amount):
    """
    Verify pln amount formats.

    args:
        text: Normalized Polish source text.
        amount: Input for this operation.

    returns:
        None.
    """
    assert MONEY.fullmatch(text)["amount"] == amount
    assert CURRENCY.search(text)


@pytest.mark.parametrize(
    "text",
    [
        "100 EUR",
        "100 euro",
        "100 USD",
        "100 GBP",
        "100 CHF",
        "€100",
        "$100",
        "£100",
        "Powódka doznała złamania nogi.",
    ],
)
def test_other_currencies_and_unrelated_words_are_not_matched(text):
    """
    Verify other currencies and unrelated words are not matched.

    args:
        text: Normalized Polish source text.

    returns:
        None.
    """
    assert not MONEY.search(text)
    assert not CURRENCY.search(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("4 mln zł", "4000000"),
        ("1,5 mln zł", "1500000"),
        ("0,125 mln zł", "125000"),
        ("4mln zł", "4000000"),
        ("4 mln. PLN", "4000000"),
        ("250 tys. zł", "250000"),
        ("2,75 tys złotych", "2750"),
        ("1 tysiąc zł", "1000"),
        ("2 tysiące zł", "2000"),
        ("5 tysięcy zł", "5000"),
        ("1 milion zł", "1000000"),
        ("4 miliony zł", "4000000"),
        ("5 milionów PLN", "5000000"),
        ("1,25 mld zł", "1250000000"),
        ("1 miliard zł", "1000000000"),
        ("2 miliardy zł", "2000000000"),
        ("5 miliardów zł", "5000000000"),
        ("250 TYS. PLN", "250000"),
        ("1\u00a0250,50 zł", "1250.50"),
        ("1\u202f250,50 zł", "1250.50"),
        ("1,23456 tys. zł", "1234.56"),
        ("4 mln 250 tys. zł", "4250000"),
        ("1 mln 250 zł", "1000250"),
        ("1 mld 2 mln 3 tys. 4,50 zł", "1002003004.5"),
    ],
)
def test_scaled_amounts_are_converted_without_losing_source_offsets(text, expected):
    """
    Verify scaled amounts are converted without losing source offsets.

    args:
        text: Normalized Polish source text.
        expected: Expected result.

    returns:
        None.
    """
    match = MONEY.fullmatch(text)
    assert match is not None
    assert amount_in_pln(match) == expected
    source = f"Zasądza {text} na rzecz powódki. Koszty rozlicza osobno."
    rows = list(basal.amount_contexts(source))
    assert len(rows) == 1
    row = rows[0]
    assert row["kwota"] == expected
    assert source[row["amount_start"] : row["amount_end"]] == text
    assert row["marked"].count("<KWOTA>") == 1
    assert row["text"] == source[: source.index(" Koszty")]


@pytest.mark.parametrize(
    "text",
    [
        "4 mln EUR",
        "250 tys USD",
        "4 milionowe zł",
        "12.34 zł",
        "12 34 zł",
        "1,2,3 mln zł",
    ],
)
def test_invalid_amounts_do_not_produce_partial_matches(text):
    """
    Verify invalid amounts do not produce partial matches.

    args:
        text: Normalized Polish source text.

    returns:
        None.
    """
    assert not MONEY.search(text)


@pytest.mark.parametrize("text", ["1 tys. 2 mln zł", "1 mln 2 mln zł"])
def test_compound_amounts_require_descending_units(text):
    """
    Verify compound amounts require descending units.

    args:
        text: Normalized Polish source text.

    returns:
        None.
    """
    with pytest.raises(ValueError, match="largest to smallest"):
        amount_in_pln(MONEY.fullmatch(text))


def test_config_and_path_are_independent_of_working_directory(monkeypatch, tmp_path):
    """
    Verify config and path are independent of working directory.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    expected = basal.load_config()
    monkeypatch.chdir(tmp_path)
    config = basal.load_config()
    assert config == expected
    assert list(config["fields"]) == list(basal.FEATURE_DESCRIPTIONS)
    assert get_data.ROOT / config["api"]["download_path"] == ROOT / "tmp"
    start, end, params = get_data.search_settings(CONFIG, date(2024, 2, 29))
    assert (start, end) == (date(2014, 2, 28), date(2024, 2, 29))
    assert params["judgmentStartDate"] == "2014-02-28"
    assert "all" not in params


@pytest.mark.parametrize(
    "change",
    [
        {"max_age_years": True},
        {"max_age_years": 3000},
        {"max_pages": -1},
        {"debug": "true"},
        {"download_path": "../outside"},
        {"download_path": ""},
        {"download_path": "/tmp"},
        {"max_gib": -1},
        {"max_gib": True},
        {"max_gib": float("nan")},
        {"timeout": 0},
        {"timeout": float("inf")},
    ],
)
def test_invalid_config_is_rejected(change):
    """
    Verify invalid config is rejected.

    args:
        change: Configuration change under examination.

    returns:
        None.
    """
    with pytest.raises(ValueError):
        get_data.search_settings({**CONFIG, **change})


def search_response(params, identifiers, more=False):
    """
    Provide the search response behavior used by this operation.

    args:
        params: Effective SAOS query parameters.
        identifiers: Input for this operation.
        more: Input for this operation.

    returns:
        Result of this operation.
    """
    return {
        "items": [{"id": identifier} for identifier in identifiers],
        "info": None,
        "queryTemplate": {key: {"value": value} for key, value in params.items()},
        "links": [{"rel": "self", "href": get_data.BASE_URL + "/dump/judgments"}]
        + (
            [
                {
                    "rel": "next",
                    "href": get_data.BASE_URL + "/dump/judgments?pageNumber=1",
                }
            ]
            if more
            else []
        ),
    }


@pytest.mark.parametrize("limit, expected", [(0, [0, 1, 2]), (1, [0]), (2, [0, 1])])
def test_pagination_uses_next_without_result_count(
    monkeypatch, tmp_path, limit, expected
):
    """
    Verify pagination uses next without result count.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        limit: Maximum batch or fragment size.
        expected: Expected result.

    returns:
        None.
    """
    pages = []

    def fetch(url, *args):
        """
        Provide the fetch behavior used by this operation.

        args:
            url: Resource URL.
            args: Input for this operation.

        returns:
            Result of this operation.
        """
        from urllib.parse import parse_qs, urlparse

        page = int(parse_qs(urlparse(url).query)["pageNumber"][0])
        pages.append(page)
        _, _, params = get_data.search_settings(CONFIG)
        params["pageNumber"] = page
        return search_response(params, [page + 1], page < 2)

    monkeypatch.setattr(get_data, "fetch_json", fetch)
    assert len(
        list(get_data.search_pages({**CONFIG, "max_pages": limit}, tmp_path))
    ) == len(expected)
    assert pages == expected


@pytest.mark.parametrize("defect", ["query", "duplicate", "links", "id", "empty_next"])
def test_invalid_dump_response_stops_download(monkeypatch, tmp_path, defect):
    """
    Verify invalid dump response stops download.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        defect: Malformed field under examination.

    returns:
        None.
    """
    _, _, params = get_data.search_settings(CONFIG)
    payload = search_response(params, [1, 2], True)
    if defect == "query":
        payload["queryTemplate"]["judgmentStartDate"]["value"] = "2000-01-01"
    elif defect == "duplicate":
        payload["items"][1] = payload["items"][0]
    elif defect == "id":
        payload["items"][0]["id"] = True
    elif defect == "empty_next":
        payload["items"] = []
    else:
        payload["links"] = []
    monkeypatch.setattr(get_data, "fetch_json", lambda *args: payload)
    with pytest.raises(ValueError):
        get_data.search_judgments(params, tmp_path)


@pytest.mark.parametrize("debug", [False, True])
def test_http_reads_fresh_data_and_saves_raw_only_in_debug(
    monkeypatch, tmp_path, debug
):
    """
    Verify http reads fresh data and saves raw only in debug.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    calls = []
    raw = b'{"data": {"id": 42}, "untouched": [1, null]}'

    class Response:
        content = raw
        headers = {"Content-Type": "application/json"}
        status_code = 200

        def raise_for_status(self):
            """
            Provide the raise for status behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            pass

        def __enter__(self):
            """
            Provide the   enter   behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            return self

        def __exit__(self, *args):
            """
            Provide the   exit   behavior used by this operation.

            args:
                args: Input for this operation.

            returns:
                Result of this operation.
            """
            pass

    def get(url, **kwargs):
        """
        Provide the get behavior used by this operation.

        args:
            url: Resource URL.
            kwargs: Input for this operation.

        returns:
            Result of this operation.
        """
        calls.append(kwargs)
        return Response()

    monkeypatch.setattr(requests, "get", get)
    url = get_data.BASE_URL + "/judgments/42"
    directory = tmp_path if debug else None
    assert get_data.fetch_json(url, directory, timeout=12) == json.loads(raw)
    assert get_data.fetch_json(url, directory, timeout=12) == json.loads(raw)
    assert len(calls) == 2 and calls[0]["timeout"] == 12
    assert bool(list(tmp_path.iterdir())) == debug
    if debug:
        path = tmp_path / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        assert path.read_bytes() == raw
        assert (
            json.loads(path.with_suffix(".meta.json").read_text())["sha256"]
            == hashlib.sha256(raw).hexdigest()
        )


def test_http_timeout_is_not_retried(monkeypatch):
    """
    Verify http timeout is not retried.

    args:
        monkeypatch: Scoped replacements for the test.

    returns:
        None.
    """
    calls = []

    def get(*args, **kwargs):
        """
        Provide the get behavior used by this operation.

        args:
            args: Input for this operation.
            kwargs: Input for this operation.

        returns:
            Result of this operation.
        """
        calls.append(kwargs)
        raise requests.Timeout("offline")

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(requests.Timeout):
        get_data.fetch_json(get_data.BASE_URL + "/dump/judgments", timeout=4)
    assert len(calls) == 1


def fixture():
    """
    Create a structurally valid judgment for isolated validation checks.

    args:
        None

    returns:
        Synthetic SAOS judgment payload.
    """
    return {
        "data": {
            "id": 254930,
            "judgmentDate": "2020-05-12",
            "courtType": "COMMON",
            "judgmentType": "SENTENCE",
            "courtCases": [{"caseNumber": "I C 42/20"}],
            "division": {
                "id": 10,
                "name": "I Wydział Cywilny",
                "court": {
                    "id": 1,
                    "name": "Sąd Okręgowy w Warszawie",
                    "type": "REGIONAL",
                },
            },
            "textContent": "<p>Zasądza 10 000 zł tytułem zadośćuczynienia.</p>",
        }
    }


@pytest.mark.parametrize("defect", ["id", "date", "court", "case", "text", "type"])
def test_invalid_judgments_are_rejected(defect):
    """
    Verify invalid judgments are rejected.

    args:
        defect: Malformed field under examination.

    returns:
        None.
    """
    payload = fixture()
    data = payload["data"]
    if defect == "id":
        data["id"] = True
    elif defect == "date":
        data["judgmentDate"] = "2010-01-01"
    elif defect == "court":
        data["division"]["court"]["id"] = False
    elif defect == "case":
        data["courtCases"] = []
    elif defect == "text":
        data["textContent"] = "<p> </p>"
    else:
        data["judgmentType"] = "REASONS"
    with pytest.raises(ValueError):
        get_data.validate_judgment(
            payload, 254930, date(2016, 10, 29), date(2026, 10, 29)
        )


@pytest.mark.parametrize("debug", [False, True])
def test_download_deduplicates_filters_and_reports_rejections(
    monkeypatch, tmp_path, debug, capsys
):
    """
    Verify download deduplicates filters and reports rejections.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.
        capsys: Captured console output.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    data = fixture()["data"]
    division = data["division"]
    valid = {**data, "division": {"id": division["id"]}}
    invalid = {**valid, "id": 254931, "textContent": None}
    irrelevant = {**valid, "id": 254932, "textContent": "<p>Inna sprawa.</p>"}
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: iter(
            [(0, {"items": [valid, valid, invalid, irrelevant], "links": []})]
        ),
    )
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    config = basal.load_config(overrides={**CONFIG, "debug": debug})
    checkpoint = get_data.load_checkpoint(config)
    records = []
    for record in get_data.download(config["api"], checkpoint, "run"):
        records.append(record)
        get_data.write_json(
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


def test_partial_network_failure_is_visible(monkeypatch, tmp_path, capsys):
    """
    Verify partial network failure is visible.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)

    def fail(*args):
        """
        Provide the fail behavior used by this operation.

        args:
            args: Input for this operation.

        returns:
            Result of this operation.
        """
        raise requests.Timeout("SAOS unavailable")

    monkeypatch.setattr(get_data, "search_pages", fail)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    get_data.write_json(tmp_path / "tmp/checkpoint.json", checkpoint)
    with pytest.raises(requests.Timeout):
        list(get_data.download(CONFIG, checkpoint, "run"))
    report = json.loads(capsys.readouterr().out)
    assert not report["complete"] and report["error"] == "SAOS unavailable"
    assert (tmp_path / "tmp/checkpoint.json").exists()
    assert not (tmp_path / "tmp/debug").exists()


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "fixtures").glob("*/model_calls.json")),
    ids=lambda path: path.parent.name,
)
def test_basal_validates_historical_requests_and_responses(monkeypatch, tmp_path, path):
    """
    Validate historical model responses independently of the current extraction order.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        path: Path to the relevant input or output file.

    returns:
        None.
    """
    config = basal.load_config()
    monkeypatch.setattr(basal, "ROOT", tmp_path)
    calls = translate_fields(json.loads(path.read_text()))
    responses = {
        json.dumps(call["request"], sort_keys=True, ensure_ascii=False): call[
            "response"
        ]
        for call in calls
    }
    seen = []

    class Server:
        async def serve(self, request):
            """
            Provide the serve behavior used by this operation.

            args:
                request: Input for this operation.

            returns:
                Result of this operation.
            """
            key = json.dumps(request, sort_keys=True, ensure_ascii=False)
            assert key in responses, "Extraction changed a saved model request"
            seen.append(key)
            return responses[key]

    for call in calls:
        request = call["request"]
        asyncio.run(basal.ask(Server(), request["state"], request["questions"], config))
    assert len(seen) == len(calls)


@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("low_confidence", [False, True])
@pytest.mark.parametrize("missing_target", [False, True])
def test_pipeline_writes_separate_outputs_and_stops_worker(
    monkeypatch, tmp_path, debug, low_confidence, missing_target
):
    """
    Verify pipeline writes separate outputs and stops worker.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.
        low_confidence: Whether target verification falls below the threshold.
        missing_target: Whether no monetary target could be established.

    returns:
        None.
    """
    import argparse
    import sys
    import types

    from src.dataset import get_data as pipeline

    config = basal.load_config()
    config["api"]["debug"] = debug
    config["api"]["max_gib"] = 0
    config["metadata_fields"] = ["judgment_year", "instance", "court_seat"]
    monkeypatch.setattr(pipeline, "check_device", lambda device: None)
    stopped = []

    class Server:
        def __init__(self, args):
            """
            Initialize the instance state.

            args:
                args: Input for this operation.

            returns:
                None.
            """
            pass

        async def worker(self):
            """
            Provide the worker behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            try:
                await asyncio.Event().wait()
            finally:
                stopped.append(True)

    module = types.ModuleType("basal.server")
    module.Server = Server
    module.parser = lambda: types.SimpleNamespace(
        parse_args=lambda args: argparse.Namespace(model="test")
    )
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(*args, **kwargs):
        """
        Verify the model cache destination without fetching weights.

        args:
            args: Model identifier arguments.
            kwargs: Snapshot download settings.

        returns:
            Placeholder model directory.
        """
        assert kwargs["cache_dir"] == tmp_path / "basal_weights"
        assert kwargs["max_workers"] == 1
        assert "model*.safetensors" in kwargs["allow_patterns"]
        assert "README.md" not in kwargs["allow_patterns"]
        return "test"

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "basal.server", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "download",
        lambda *args: (
            record
            for record in [
                {
                    "id": 42,
                    "text": "Opis.",
                    "source_payload": {"id": 42},
                    "judgmentDate": "2020-05-12",
                    "division": {"court": {"name": "Sąd Okręgowy w Płocku"}},
                }
            ]
        ),
    )

    async def extract(server, text, config, identifier):
        """
        Extract source-grounded factual descriptions and separate monetary targets.

        args:
            server: Local inference server.
            text: Normalized Polish source text.
            config: Effective pipeline or acquisition settings.
            identifier: Stable judgment identifier.

        returns:
            Feature and target results with review evidence.
        """
        await asyncio.sleep(0)
        return {
            "id": identifier,
            "features": {"opis": text},
            "evidence": {},
            "award": basal.finish_award(
                {
                    "zadoscuczynienie": "brak informacji" if missing_target else "100",
                    "pozostale_odszkodowanie": "brak informacji",
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

    monkeypatch.setattr(basal, "extract", extract)
    asyncio.run(pipeline.run(config))
    directory = tmp_path / "tmp"
    path = directory / "42"
    if (low_confidence or missing_target) and not debug:
        assert not path.exists()
        assert stopped == [True]
        return
    record = json.loads((path / "record.json").read_text())
    assert record == {
        "opis": "Opis.",
        "judgment_year": 2020,
        "instance": "pierwsza",
        "court_seat": "Płocku",
    }
    assert json.loads((path / "award.json").read_text()) == {
        "zadoscuczynienie": "not_available" if missing_target else "100",
        "pozostale_odszkodowanie": "not_available",
        "total": "not_available",
        "valid": not low_confidence and not missing_target,
    }
    assert {item.name for item in path.iterdir()} == (
        {"record.json", "award.json", "model_calls.json", "source.txt"}
        if debug
        else {"record.json", "award.json"}
    )
    assert stopped == [True]


def test_notebooks_accept_shared_dataset_config():
    """
    Verify notebooks accept shared dataset config.

    args:
        None

    returns:
        None.
    """
    import importlib

    sys.path.insert(0, str(ROOT / "notebooks"))
    feasibility = importlib.import_module("01_saos_feasibility")
    feasibility.search_settings(CONFIG)


@pytest.mark.parametrize(
    "court_name, expected",
    [
        ("Sąd Okręgowy w Płocku", "Płocku"),
        ("Sąd Rejonowy dla Warszawy-Mokotowa w Warszawie", "Warszawie"),
        ("Sąd Rejonowy we Wrocławiu", "Wrocławiu"),
        ("Sąd Rejonowy", "not_available"),
    ],
)
def test_metadata_features_use_judgment_date_instance_and_court_seat(
    court_name, expected
):
    """
    Verify metadata features use judgment date instance and court seat.

    args:
        court_name: Source court name containing its location.
        expected: Expected result.

    returns:
        None.
    """
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "druga", "decision": {"confidence": 0.8}},
    }
    record = {
        "judgmentDate": "2019-12-31",
        "division": {"court": {"name": court_name, "type": "DISTRICT"}},
    }
    get_data.add_metadata_features(
        result, record, ["judgment_year", "instance", "court_seat"]
    )
    assert result["features"] == {
        "judgment_year": 2019,
        "instance": "druga",
        "court_seat": expected,
    }
    assert result["metadata_sources"]["judgment_year"]["value"] == "2019-12-31"
    assert result["confidence"]["instance"] == 0.8


def test_metadata_selection_and_unknown_instance_do_not_infer_from_court_level():
    """
    Verify metadata selection and unknown instance do not infer from court level.

    args:
        None

    returns:
        None.
    """
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "not_available", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Apelacyjny w Krakowie", "type": "APPEAL"}},
    }
    get_data.add_metadata_features(result, record, ["instance"])
    assert result["features"] == {"instance": "not_available"}
    assert result["field_status"] == {"instance": "not_available"}


def test_config_allows_metadata_only_without_adding_it_to_basal_schema(tmp_path):
    """
    Verify config allows metadata only without adding it to basal schema.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    import yaml

    config = basal.read_config(ROOT / "configs/dataset/basal.yaml")
    config["feature_fields"] = list(basal.METADATA_FIELDS)
    (tmp_path / "basal.yaml").write_text(yaml.safe_dump(config))
    (tmp_path / "api.yaml").write_text(yaml.safe_dump(CONFIG))
    loaded = basal.load_config(tmp_path)
    assert loaded["metadata_fields"] == list(basal.METADATA_FIELDS)
    assert loaded["fields"] == {}


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ([], {}),
        (["--max-pages", "2"], {"max_pages": 2}),
        (["--max-gib", "5"], {"max_gib": 5.0}),
        (["--no-debug"], {"debug": False}),
        (["--debug"], {"debug": True}),
        (
            [
                "--timeout",
                "12.5",
                "--max-age-years",
                "3",
                "--download-path",
                "tmp/test",
            ],
            {"timeout": 12.5, "max_age_years": 3, "download_path": "tmp/test"},
        ),
    ],
)
def test_cli_overrides_only_explicit_arguments(arguments, expected):
    """
    Verify cli overrides only explicit arguments.

    args:
        arguments: Command-line arguments, or None for process arguments.
        expected: Expected result.

    returns:
        None.
    """
    options = vars(get_data.parse_arguments(arguments))
    assert options.pop("config_dir") is None
    assert options == expected
    original = basal.load_config()
    merged = basal.load_config(overrides=options)
    assert merged["api"] == {**original["api"], **expected}


def test_cli_can_override_yaml_before_validation(tmp_path):
    """
    Verify cli can override yaml before validation.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    import yaml

    api = {**CONFIG, "max_pages": -1, "debug": True}
    (tmp_path / "api.yaml").write_text(yaml.safe_dump(api))
    (tmp_path / "basal.yaml").write_text(
        (ROOT / "configs/dataset/basal.yaml").read_text()
    )
    options = vars(
        get_data.parse_arguments(
            ["--config-dir", str(tmp_path), "--max-pages", "1", "--no-debug"]
        )
    )
    config = basal.load_config(options.pop("config_dir"), options)
    assert config["api"] == {**api, "max_pages": 1, "debug": False}


@pytest.mark.parametrize("debug", [False, True])
def test_model_request_cache_is_debug_only(monkeypatch, tmp_path, debug):
    """
    Verify model request cache is debug only.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    config = basal.load_config(overrides={"debug": debug})
    monkeypatch.setattr(basal, "ROOT", tmp_path)
    calls = []

    class Server:
        async def serve(self, request):
            """
            Provide the serve behavior used by this operation.

            args:
                request: Input for this operation.

            returns:
                Result of this operation.
            """
            calls.append(request)
            return {
                "answers": {
                    "q": {"noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}
                }
            }

    questions = {"q": {"type": "noul", "instructions": "Czy to fakt?"}}

    async def ask():
        """
        Request a validated model decision and optionally reuse its diagnostic cache.

        args:
            None

        returns:
            Validated model answers.
        """
        await basal.ask(Server(), "Opis.", questions, config)
        await basal.ask(Server(), "Opis.", questions, config)

    asyncio.run(ask())
    assert len(calls) == (1 if debug else 2)
    assert bool(list(tmp_path.rglob("*.json"))) == debug


def test_population_feature_uses_official_match_and_preserves_provenance():
    """
    Verify population feature uses official match and preserves provenance.

    args:
        None

    returns:
        None.
    """
    import morfeusz2

    row = {
        "NAZWA": "Warszawa",
        "RODZAJ": "miasto",
        "LICZBAMIESZKANCOW": 1800000,
        "IDENTYFIKATORSIMC": "0918123",
    }
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "not_available", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Okręgowy w Warszawie"}},
    }
    get_data.add_metadata_features(
        result,
        record,
        ["court_locality_size"],
        ([row], morfeusz2.Morfeusz(), {"population_year": None}),
    )
    assert result["features"] == {"court_locality_size": "city_at_least_500000"}
    assert result["metadata_sources"]["court_locality_size"]["locality"] == row


@pytest.mark.parametrize("debug", [False, True])
def test_locality_download_is_retained_only_for_debug(monkeypatch, tmp_path, debug):
    """
    Verify locality download is retained only for debug.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    import sqlite3

    from src.dataset.processing import localities

    source = tmp_path / "source.gpkg"
    with sqlite3.connect(source) as connection:
        connection.execute(
            "CREATE TABLE OT_ADMS_P (NAZWA, RODZAJ, LICZBAMIESZKANCOW, IDENTYFIKATORSIMC, IDENTYFIKATORTERC, IDENTYFIKATORPRNG, WERSJA)"
        )
        connection.execute(
            "INSERT INTO OT_ADMS_P VALUES ('Pisz', 'miasto', 19000, '1', '2', '3', '2026')"
        )
    content = source.read_bytes()

    class Response:
        def __enter__(self):
            """
            Provide the   enter   behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            self.content = content
            return self

        def __exit__(self, *args):
            """
            Provide the   exit   behavior used by this operation.

            args:
                args: Input for this operation.

            returns:
                Result of this operation.
            """
            pass

        def raise_for_status(self):
            """
            Provide the raise for status behavior used by this operation.

            args:
                None

            returns:
                Result of this operation.
            """
            pass

    monkeypatch.setattr(localities.requests, "get", lambda *args, **kwargs: Response())
    directory = tmp_path / "debug"
    rows, metadata = localities.load_localities(directory if debug else None)
    assert rows[0]["NAZWA"] == "Pisz"
    assert metadata["population_year"] is None
    assert directory.exists() == debug


def test_court_dump_joins_divisions_across_pages(monkeypatch):
    """
    Verify court dump joins divisions across pages.

    args:
        monkeypatch: Scoped replacements for the test.

    returns:
        None.
    """
    pages = []

    def fetch(url, directory, session, timeout):
        """
        Provide the fetch behavior used by this operation.

        args:
            url: Resource URL.
            directory: Input or output directory for this operation.
            session: Reusable HTTP session, when supplied.
            timeout: Connection/read timeout in seconds.

        returns:
            Result of this operation.
        """
        from urllib.parse import parse_qs, urlparse

        page = int(parse_qs(urlparse(url).query)["pageNumber"][0])
        pages.append(page)
        assert timeout == 7
        return {
            "items": [
                {
                    "id": page + 1,
                    "name": f"Sąd {page}",
                    "type": "REGIONAL",
                    "divisions": [{"id": 10 + page, "name": "Cywilny"}],
                }
            ],
            "links": [{"rel": "next"}] if page == 0 else [],
        }

    monkeypatch.setattr(get_data, "fetch_json", fetch)
    divisions = get_data.load_divisions(object(), timeout=7)
    assert pages == [0, 1]
    assert divisions[11]["court"] == {"id": 2, "name": "Sąd 1", "type": "REGIONAL"}


def test_court_dump_rejects_duplicate_divisions(monkeypatch):
    """
    Verify court dump rejects duplicate divisions.

    args:
        monkeypatch: Scoped replacements for the test.

    returns:
        None.
    """
    payload = {
        "items": [
            {
                "id": 1,
                "name": "Sąd",
                "type": "REGIONAL",
                "divisions": [
                    {"id": 10, "name": "Cywilny"},
                    {"id": 10, "name": "Karny"},
                ],
            }
        ],
        "links": [],
    }
    monkeypatch.setattr(get_data, "fetch_json", lambda *args: payload)
    with pytest.raises(ValueError, match="duplicate division"):
        get_data.load_divisions(object())


@pytest.mark.parametrize(
    "court",
    [
        None,
        {"id": 1, "name": "Sąd", "type": "REGIONAL"},
        {"id": True, "name": "Sąd", "type": "REGIONAL", "divisions": []},
    ],
)
def test_malformed_court_dump_is_rejected(monkeypatch, court):
    """
    Verify malformed court dump is rejected.

    args:
        monkeypatch: Scoped replacements for the test.
        court: Input for this operation.

    returns:
        None.
    """
    monkeypatch.setattr(
        get_data, "fetch_json", lambda *args: {"items": [court], "links": []}
    )
    with pytest.raises(ValueError, match="Invalid court"):
        get_data.load_divisions(object())


@pytest.mark.parametrize(
    ("device", "cuda", "mps", "mode"),
    [
        ("cpu", False, False, "eager"),
        ("cuda", True, False, "eager"),
        ("mps", False, True, "mps"),
    ],
)
def test_explicit_device_configuration_and_availability(
    monkeypatch, tmp_path, device, cuda, mps, mode
):
    """
    Verify explicit device configuration and availability.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        device: Explicit CPU or accelerator selection.
        cuda: Input for this operation.
        mps: Input for this operation.
        mode: Input for this operation.

    returns:
        None.
    """
    import types

    import yaml

    settings = basal.read_config(ROOT / "configs/dataset/basal.yaml")
    settings["device"] = device
    (tmp_path / "basal.yaml").write_text(yaml.safe_dump(settings))
    (tmp_path / "api.yaml").write_text(yaml.safe_dump(CONFIG))
    config = basal.load_config(tmp_path)
    assert config["system_one"]["mode"] == mode
    assert config["system_one"]["device"] == device
    fake = types.SimpleNamespace(
        version=types.SimpleNamespace(cuda="13.0" if cuda else None),
        cuda=types.SimpleNamespace(is_available=lambda: cuda),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: mps)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    get_data.check_device(device)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_accelerator_is_reported(monkeypatch, device):
    """
    Verify unavailable accelerator is reported.

    args:
        monkeypatch: Scoped replacements for the test.
        device: Explicit CPU or accelerator selection.

    returns:
        None.
    """
    import types

    fake = types.SimpleNamespace(
        version=types.SimpleNamespace(cuda=None),
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    with pytest.raises(RuntimeError, match="unavailable"):
        get_data.check_device(device)


def test_missing_pytorch_has_installation_guidance(monkeypatch):
    """
    Verify missing pytorch has installation guidance.

    args:
        monkeypatch: Scoped replacements for the test.

    returns:
        None.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="pytorch-macos"):
        get_data.check_device("cpu")


def test_checkpoint_freezes_dates_and_allows_operational_overrides(
    monkeypatch, tmp_path
):
    """
    Verify checkpoint freezes dates and allows operational overrides.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    checkpoint["request"] = get_data.search_settings(CONFIG, date(2024, 2, 29))[2]
    get_data.write_json(tmp_path / "tmp/checkpoint.json", checkpoint)
    changed = basal.load_config(
        overrides={**CONFIG, "debug": True, "timeout": 5, "max_pages": 2, "max_gib": 5}
    )
    assert get_data.load_checkpoint(changed) == checkpoint
    changed["system_one"]["minimum_probability"] = 0.8
    with pytest.raises(ValueError, match="Extraction settings changed"):
        get_data.load_checkpoint(changed)


def test_resume_skips_committed_records_and_retries_only_unfinished_item(
    monkeypatch, tmp_path, capsys
):
    """
    Verify resume skips committed records and retries only unfinished item.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    data = fixture()["data"]
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

    monkeypatch.setattr(get_data, "search_pages", pages)
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = get_data.download(config["api"], checkpoint, "first")
    assert next(stream)["id"] == 1
    get_data.write_json(
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
    resumed = get_data.load_checkpoint(config)
    assert resumed["next_item"] == 1
    stream = get_data.download(config["api"], resumed, "second")
    assert next(stream)["id"] == 2
    stream.close()
    assert starts == [0, 0]
    assert (tmp_path / "tmp/1/record.json").exists()
    capsys.readouterr()


def test_resume_recognizes_record_saved_before_checkpoint_advance(
    monkeypatch, tmp_path, capsys
):
    """
    Verify resume recognizes record saved before checkpoint advance.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    data = fixture()["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: iter([(0, {"items": [item], "links": []})]),
    )
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = get_data.download(config["api"], checkpoint, "first")
    assert next(stream)["id"] == data["id"]
    get_data.write_json(
        tmp_path / "tmp" / str(data["id"]) / "record.json",
        {
            "id": data["id"],
            "dataset_id": checkpoint["dataset_id"],
            "features": {},
            "targets": {},
        },
    )
    stream.close()
    resumed = get_data.load_checkpoint(config)
    assert list(get_data.download(config["api"], resumed, "second")) == []
    assert resumed["complete"]
    capsys.readouterr()


def test_changed_resume_page_is_rejected(monkeypatch, tmp_path, capsys):
    """
    Verify changed resume page is rejected.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    checkpoint["page_ids"] = [1]
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: iter([(0, {"items": [{"id": 2}], "links": []})]),
    )
    with pytest.raises(ValueError, match="page contents changed"):
        list(get_data.download(config["api"], checkpoint, "run"))
    assert not checkpoint["complete"]
    capsys.readouterr()


def test_checkpoint_does_not_advance_without_saved_output(
    monkeypatch, tmp_path, capsys
):
    """
    Verify checkpoint does not advance without saved output.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        capsys: Captured console output.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    data = fixture()["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: iter([(0, {"items": [item], "links": []})]),
    )
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = get_data.download(config["api"], checkpoint, "run")
    next(stream)
    with pytest.raises(RuntimeError, match="completed record"):
        next(stream)
    assert get_data.load_checkpoint(config)["next_item"] == 0
    capsys.readouterr()


@pytest.mark.parametrize("debug", [False, True])
def test_export_contains_only_features_targets_and_requested_debug_files(
    tmp_path, debug
):
    """
    Keep diagnostic data out of the feature and target files.

    args:
        tmp_path: Isolated output directory.
        debug: Whether the two debug files are requested.

    returns:
        None.
    """
    config = basal.load_config(overrides={"debug": debug})
    result = {
        "id": 42,
        "features": {"harm_type": "source sentence"},
        "source_text": "source sentence",
        "evidence": {"harm_type": [{"start": 0, "end": 15}]},
        "description_reviews": [
            {"start": 0, "end": 15, "answers": {"wyciek": {"noul": 0.01}}}
        ],
        "model_calls": [{"request": "input", "response": "output"}],
        "award": {
            "confirmed": True,
            "verification": {"noul": 0.99},
            "zadoscuczynienie": "100",
            "pozostale_odszkodowanie": "brak informacji",
            "evidence": ["private diagnostics"],
        },
    }
    get_data.write_result(result, tmp_path, config)
    assert json.loads((tmp_path / "record.json").read_text()) == {
        "harm_type": "source sentence"
    }
    assert json.loads((tmp_path / "award.json").read_text()) == {
        "zadoscuczynienie": "100",
        "pozostale_odszkodowanie": "not_available",
        "total": "not_available",
        "valid": True,
    }
    assert {p.name for p in tmp_path.iterdir()} == (
        {"record.json", "award.json", "model_calls.json", "source.txt"}
        if debug
        else {"record.json", "award.json"}
    )


def test_removed_download_only_option_is_rejected():
    """
    Verify removed download only option is rejected.

    args:
        None

    returns:
        None.
    """
    with pytest.raises(SystemExit):
        get_data.parse_arguments(["--download-only"])


def test_output_limit_counts_existing_files_and_persists_across_resumes(tmp_path):
    """
    Verify output limit counts existing files and persists across resumes.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    from src.dataset.processing.storage import (
        StorageLimitError,
        output_limit,
        write_bytes,
    )

    (tmp_path / "existing").write_bytes(b"1234")
    with output_limit(tmp_path, 8 / 1024**3):
        write_bytes(tmp_path / "new", b"5678")
        with pytest.raises(StorageLimitError):
            write_bytes(tmp_path / "overflow", b"9")
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == 8
    assert not (tmp_path / "overflow").exists()
    with pytest.raises(StorageLimitError):
        with output_limit(tmp_path, 8 / 1024**3):
            pass


def test_output_limit_preserves_existing_record_and_counts_atomic_temporary_file(
    tmp_path,
):
    """
    Verify output limit preserves existing record and counts atomic temporary file.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    from src.dataset.processing.storage import (
        StorageLimitError,
        output_limit,
        write_bytes,
    )

    path = tmp_path / "record.json"
    path.write_bytes(b"1234")
    with output_limit(tmp_path, 6 / 1024**3):
        with pytest.raises(StorageLimitError):
            write_bytes(path, b"5678")
        assert path.read_bytes() == b"1234"
        write_bytes(path, b"12")
        write_bytes(tmp_path / "next", b"3456")
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == 6
    assert sorted(path.name for path in tmp_path.iterdir()) == ["next", "record.json"]


def test_resume_starts_at_next_page_after_network_failure(
    monkeypatch, tmp_path, capsys
):
    """
    Keep completed pages committed when fetching the next page fails.

    args:
        monkeypatch: Scoped dependency replacements.
        tmp_path: Isolated output directory.
        capsys: Captured acquisition reports.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    data = fixture()["data"]
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

    monkeypatch.setattr(get_data, "search_pages", pages)
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = get_data.download(config["api"], checkpoint, "first")
    next(stream)
    get_data.write_json(
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
    resumed = get_data.load_checkpoint(config)
    assert resumed["page"] == 1 and resumed["next_item"] == 0
    starts = []
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: starts.append(args[4]) or iter([(1, {"items": [], "links": []})]),
    )
    assert list(get_data.download(config["api"], resumed, "second")) == []
    assert starts == [1] and resumed["complete"]
    capsys.readouterr()


def test_rejected_record_advances_checkpoint_without_case_files(
    monkeypatch, tmp_path, capsys
):
    """
    Continue after explicit rejection without persisting case artifacts.

    args:
        monkeypatch: Scoped dependency replacements.
        tmp_path: Isolated dataset directory.
        capsys: Captured progress reports.

    returns:
        None.
    """
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    config = basal.load_config(overrides=CONFIG)
    checkpoint = get_data.load_checkpoint(config)
    data = fixture()["data"]
    division = data["division"]
    item = {**data, "division": {"id": division["id"]}}
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: iter([(0, {"items": [item], "links": []})]),
    )
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    stream = get_data.download(config["api"], checkpoint, "run")
    record = next(stream)
    record["rejected"] = True
    assert list(stream) == []
    assert get_data.load_checkpoint(config)["complete"]
    assert not (tmp_path / "tmp" / str(data["id"])).exists()
    capsys.readouterr()


def test_interrupted_extraction_reuses_completed_decisions_without_debug(
    monkeypatch, tmp_path
):
    """
    Reuse operational decisions after cancellation with diagnostics disabled.

    args:
        monkeypatch: Scoped dependencies.
        tmp_path: Isolated working-state directory.

    returns:
        None.
    """
    config = basal.load_config(overrides={"debug": False})
    questions = {"q": {"type": "noul", "instructions": "Czy to fakt?"}}
    response = {
        "answers": {"q": {"noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}}
    }
    calls = []

    class Server:
        async def serve(self, request):
            """
            Interrupt the second state once.

            args:
                request: Model decision input.

            returns:
                Deterministic model response.
            """
            calls.append(request["state"])
            if request["state"] == "second" and calls.count("second") == 1:
                raise asyncio.CancelledError()
            return response

    async def execute():
        """
        Re-enter the same working cache after cancellation.

        args:
            None

        returns:
            None.
        """
        token = basal.DECISION_CACHE.set(tmp_path / "pending/decisions")
        try:
            await basal.ask(Server(), "first", questions, config)
            with pytest.raises(asyncio.CancelledError):
                await basal.ask(Server(), "second", questions, config)
            await basal.ask(Server(), "first", questions, config)
            await basal.ask(Server(), "second", questions, config)
        finally:
            basal.DECISION_CACHE.reset(token)

    asyncio.run(execute())
    assert calls == ["first", "second", "second"]
    assert len(list((tmp_path / "pending/decisions").glob("*.json"))) == 2


def test_debug_export_preserves_low_confidence_proposal_with_invalid_flag(tmp_path):
    """
    Keep proposed target values for debug inspection without labeling them valid.

    args:
        tmp_path: Isolated output directory.

    returns:
        None.
    """
    config = basal.load_config(overrides={"debug": True})
    award = basal.finish_award(
        {
            "zadoscuczynienie": "6300",
            "pozostale_odszkodowanie": "brak informacji",
            "suma": "brak informacji",
        },
        False,
        {"noul": 0.4},
    )
    result = {"features": {}, "award": award, "source_text": "Source."}
    assert get_data.rejection_reason(result, config) is not None
    get_data.write_result(result, tmp_path, config)
    assert json.loads((tmp_path / "award.json").read_text()) == {
        "zadoscuczynienie": "6300",
        "pozostale_odszkodowanie": "not_available",
        "total": "not_available",
        "valid": False,
    }
    assert {path.name for path in tmp_path.iterdir()} == {
        "award.json",
        "record.json",
        "model_calls.json",
        "source.txt",
    }
