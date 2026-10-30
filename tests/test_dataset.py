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

from src.dataset import basal_filtering as basal
from src.dataset import get_data
from src.dataset.get_data import run
from src.dataset.text import extract_text
from src.dataset.text import CURRENCY, MONEY, amount_in_pln

CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "debug": False,
    "download_path": "tmp/",
    "download_only": False,
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
    assert not MONEY.search(text)


@pytest.mark.parametrize("text", ["1 tys. 2 mln zł", "1 mln 2 mln zł"])
def test_compound_amounts_require_descending_units(text):
    with pytest.raises(ValueError, match="od największej do najmniejszej"):
        amount_in_pln(MONEY.fullmatch(text))


def test_config_and_path_are_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    config = basal.load_config()
    assert {**config["api"], "debug": False} == CONFIG
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
        {"timeout": 0},
        {"timeout": float("inf")},
        {"download_only": "false"},
    ],
)
def test_invalid_config_is_rejected(change):
    with pytest.raises(ValueError):
        get_data.search_settings({**CONFIG, **change})


def search_response(params, identifiers, more=False):
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
    pages = []

    def fetch(url, *args):
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
    calls = []
    raw = b'{"data": {"id": 42}, "untouched": [1, null]}'

    class Response:
        content = raw
        headers = {"Content-Type": "application/json"}
        status_code = 200

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def get(url, **kwargs):
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
    calls = []

    def get(*args, **kwargs):
        calls.append(kwargs)
        raise requests.Timeout("offline")

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(requests.Timeout):
        get_data.fetch_json(get_data.BASE_URL + "/dump/judgments", timeout=4)
    assert len(calls) == 1


def fixture(identifier=254930):
    return json.loads(
        (ROOT / "fixtures" / str(identifier) / "judgment.json").read_text()
    )


@pytest.mark.parametrize("defect", ["id", "date", "court", "case", "text", "type"])
def test_invalid_judgments_are_rejected(defect):
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
    monkeypatch.setattr(get_data, "ROOT", tmp_path)
    data = fixture()["data"]
    division = data["division"]
    valid = {**data, "division": {"id": division["id"]}}
    invalid = {**valid, "id": 254931, "textContent": None}
    irrelevant = {**valid, "id": 254932, "textContent": "<p>Inna sprawa.</p>"}
    monkeypatch.setattr(
        get_data,
        "search_pages",
        lambda *args: iter([(0, {"items": [valid, valid, invalid, irrelevant]})]),
    )
    monkeypatch.setattr(
        get_data, "load_divisions", lambda *args: {division["id"]: division}
    )
    records = list(get_data.download({**CONFIG, "debug": debug}))
    assert [record["id"] for record in records] == [254930]
    assert records[0]["division"] == division
    report = json.loads(capsys.readouterr().out)
    assert (
        report["complete"]
        and report["accepted"] == report["duplicates"] == report["filtered"] == 1
    )
    assert report["rejected"] == [{"id": 254931, "reason": "Missing judgment text"}]
    assert bool(list(tmp_path.rglob("*.json"))) == debug


def test_partial_network_failure_is_visible(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(get_data, "ROOT", tmp_path)

    def fail(*args):
        raise requests.Timeout("SAOS unavailable")

    monkeypatch.setattr(get_data, "search_pages", fail)
    with pytest.raises(requests.Timeout):
        list(get_data.download(CONFIG))
    report = json.loads(capsys.readouterr().out)
    assert not report["complete"] and report["error"] == "SAOS unavailable"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "path",
    sorted((ROOT / "fixtures").glob("*/model_calls.json")),
    ids=lambda path: path.parent.name,
)
def test_basal_replays_saved_requests_without_behavior_changes(
    monkeypatch, tmp_path, path
):
    config = basal.load_config()
    monkeypatch.setattr(basal, "ROOT", tmp_path)
    calls = json.loads(path.read_text())
    responses = {
        json.dumps(call["request"], sort_keys=True, ensure_ascii=False): call[
            "response"
        ]
        for call in calls
    }
    seen = []

    class Server:
        async def serve(self, request):
            key = json.dumps(request, sort_keys=True, ensure_ascii=False)
            assert key in responses, "Extraction changed a saved model request"
            seen.append(key)
            return responses[key]

    payload = json.loads((path.parent / "judgment.json").read_text())
    text = extract_text(payload["data"]["textContent"])
    result = asyncio.run(basal.extract(Server(), text, config, payload["data"]["id"]))
    assert seen
    assert result["features"] == json.loads((path.parent / "record.json").read_text())
    expected = json.loads((path.parent / "award.json").read_text())
    for key in (*config["target_fields"], "suma", "confirmed"):
        assert result["award"].get(key) == expected.get(key)
    for quotes in result["evidence"].values():
        assert all(text[row["start"] : row["end"]] == row["text"] for row in quotes)


def test_download_only_does_not_load_basal(monkeypatch, tmp_path):
    from src.dataset import get_data as pipeline

    monkeypatch.setattr(pipeline, "download", lambda config: iter([{"id": 42}]))
    asyncio.run(run(basal.load_config(), download_only=True))


@pytest.mark.parametrize("debug", [False, True])
def test_pipeline_writes_separate_outputs_and_stops_worker(
    monkeypatch, tmp_path, debug
):
    import argparse
    import sys
    import types
    from src.dataset import get_data as pipeline

    config = basal.load_config()
    config["api"]["debug"] = debug
    config["metadata_fields"] = ["rok_wyroku", "instancja", "miejscowosc_sadu"]
    stopped = []

    class Server:
        def __init__(self, args):
            pass

        async def worker(self):
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
    hub.snapshot_download = lambda *args, **kwargs: "test"
    monkeypatch.setitem(sys.modules, "basal.server", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(
        pipeline,
        "download",
        lambda config: (
            record
            for record in [
                {
                    "id": 42,
                    "text": "Opis.",
                    "judgmentDate": "2020-05-12",
                    "division": {"court": {"name": "Sąd Okręgowy w Płocku"}},
                }
            ]
        ),
    )

    async def extract(server, text, config, identifier):
        await asyncio.sleep(0)
        return {
            "id": identifier,
            "features": {"opis": text},
            "evidence": {},
            "award": {
                "confirmed": False,
                "zadoscuczynienie": "brak informacji",
                "pozostale_odszkodowanie": "brak informacji",
            },
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
    record = json.loads((path / "record.json").read_text())
    assert record["features"] == {
        "opis": "Opis.",
        "rok_wyroku": 2020,
        "instancja": "pierwsza",
        "miejscowosc_sadu": "Płocku",
    }
    assert record["targets"] == {
        "zadoscuczynienie": "brak informacji",
        "pozostale_odszkodowanie": "brak informacji",
    }
    assert (path / "debug/analysis.json").exists() == debug
    assert (directory / "debug/filtering.json").exists() == debug
    if not debug:
        assert sorted(
            p.relative_to(directory).as_posix() for p in directory.rglob("*")
        ) == ["42", "42/record.json"]
    assert stopped == [True]


def test_notebooks_accept_shared_dataset_config():
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
        ("Sąd Rejonowy", "brak informacji"),
    ],
)
def test_metadata_features_use_judgment_date_instance_and_court_seat(
    court_name, expected
):
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
        result, record, ["rok_wyroku", "instancja", "miejscowosc_sadu"]
    )
    assert result["features"] == {
        "rok_wyroku": 2019,
        "instancja": "druga",
        "miejscowosc_sadu": expected,
    }
    assert result["metadata_sources"]["rok_wyroku"]["value"] == "2019-12-31"
    assert result["confidence"]["instancja"] == 0.8


def test_metadata_selection_and_unknown_instance_do_not_infer_from_court_level():
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "brak informacji", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Apelacyjny w Krakowie", "type": "APPEAL"}},
    }
    get_data.add_metadata_features(result, record, ["instancja"])
    assert result["features"] == {"instancja": "brak informacji"}
    assert result["field_status"] == {"instancja": "brak informacji"}


def test_config_allows_metadata_only_without_adding_it_to_basal_schema(tmp_path):
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
        (["--no-debug"], {"debug": False}),
        (["--debug"], {"debug": True}),
        (["--download-only"], {"download_only": True}),
        (["--no-download-only"], {"download_only": False}),
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
    options = vars(get_data.parse_arguments(arguments))
    assert options.pop("config_dir") is None
    assert options == expected
    original = basal.load_config()
    merged = basal.load_config(overrides=options)
    assert merged["api"] == {**original["api"], **expected}


def test_cli_can_override_yaml_before_validation(tmp_path):
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
    config = basal.load_config(overrides={"debug": debug})
    monkeypatch.setattr(basal, "ROOT", tmp_path)
    calls = []

    class Server:
        async def serve(self, request):
            calls.append(request)
            return {
                "answers": {
                    "q": {"noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}
                }
            }

    questions = {"q": {"type": "noul", "instructions": "Czy to fakt?"}}

    async def ask():
        await basal.ask(Server(), "Opis.", questions, config)
        await basal.ask(Server(), "Opis.", questions, config)

    asyncio.run(ask())
    assert len(calls) == (1 if debug else 2)
    assert bool(list(tmp_path.rglob("*.json"))) == debug


def test_population_feature_uses_official_match_and_preserves_provenance():
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
        "instance": {"value": "brak informacji", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Okręgowy w Warszawie"}},
    }
    get_data.add_metadata_features(
        result,
        record,
        ["wielkość miejscowości"],
        ([row], morfeusz2.Morfeusz(), {"population_year": None}),
    )
    assert result["features"] == {"wielkość miejscowości": "miasto_co_najmniej_500_tys"}
    assert result["metadata_sources"]["wielkość miejscowości"]["locality"] == row


@pytest.mark.parametrize("debug", [False, True])
def test_locality_download_is_retained_only_for_debug(monkeypatch, tmp_path, debug):
    import sqlite3
    from src.dataset import localities

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
            self.content = content
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

    monkeypatch.setattr(localities.requests, "get", lambda *args, **kwargs: Response())
    directory = tmp_path / "debug"
    rows, metadata = localities.load_localities(directory if debug else None)
    assert rows[0]["NAZWA"] == "Pisz"
    assert metadata["population_year"] is None
    assert directory.exists() == debug
