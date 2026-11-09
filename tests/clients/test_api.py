import hashlib
import json

import pytest
import requests

from dataset.acquisition.download import search_pages
from dataset.acquisition.settings import search_settings
from dataset.clients import api

CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}


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
        "links": [{"rel": "self", "href": api.BASE_URL + "/dump/judgments"}]
        + (
            [
                {
                    "rel": "next",
                    "href": api.BASE_URL + "/dump/judgments?pageNumber=1",
                }
            ]
            if more
            else []
        ),
    }


@pytest.mark.parametrize("limit, expected", [(0, [0, 1, 2]), (1, [0]), (2, [0, 1])])
def test_pagination_uses_next_without_result_count(
    monkeypatch,
    tmp_path,
    limit,
    expected,
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
        _, _, params = search_settings(CONFIG)
        params["pageNumber"] = page
        return search_response(params, [page + 1], page < 2)

    monkeypatch.setattr(api, "fetch_json", fetch)
    assert len(list(search_pages({**CONFIG, "max_pages": limit}, tmp_path))) == len(
        expected
    )
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
    _, _, params = search_settings(CONFIG)
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
    monkeypatch.setattr(api, "fetch_json", lambda *args: payload)
    with pytest.raises(ValueError):
        api.search_judgments(params, tmp_path)


@pytest.mark.parametrize("debug", [False, True])
def test_http_reads_fresh_data_and_saves_raw_only_in_debug(
    monkeypatch,
    tmp_path,
    debug,
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
    url = api.BASE_URL + "/judgments/42"
    directory = tmp_path if debug else None
    assert api.fetch_json(url, directory, timeout=12) == json.loads(raw)
    assert api.fetch_json(url, directory, timeout=12) == json.loads(raw)
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
        api.fetch_json(api.BASE_URL + "/dump/judgments", timeout=4)
    assert len(calls) == 1


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

    monkeypatch.setattr(api, "fetch_json", fetch)
    divisions = api.load_divisions(object(), timeout=7)
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
    monkeypatch.setattr(api, "fetch_json", lambda *args: payload)
    with pytest.raises(ValueError, match="duplicate division"):
        api.load_divisions(object())


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
        api, "fetch_json", lambda *args: {"items": [court], "links": []}
    )
    with pytest.raises(ValueError, match="Invalid court"):
        api.load_divisions(object())
