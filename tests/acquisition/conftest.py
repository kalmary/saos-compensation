"""Acquisition output isolation and lightweight extraction dependencies."""

from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from dataset.acquisition import download, pipeline
from dataset.storage import checkpoints


@pytest.fixture
def output_root(monkeypatch, tmp_path):
    """Keep pipeline, downloader and checkpoint output in one isolated root.

    args:
        monkeypatch: Scoped module configuration overrides.
        tmp_path: Isolated test directory.

    returns:
        Root containing the test dataset and temporary state.
    """
    for module in (pipeline, download, checkpoints):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def fake_extractor():
    """Supply an extraction context without importing or loading model dependencies.

    args:
        None.

    returns:
        Configurable extraction callable, context and cleanup status.
    """
    client = SimpleNamespace(extract=None, stopped=False)

    @asynccontextmanager
    async def context():
        """Expose the configured extraction callable and record cleanup.

        args:
            None.

        returns:
            Async context yielding the test extraction callable.
        """
        try:
            yield client.extract
        finally:
            client.stopped = True

    client.context = context()
    return client


@pytest.fixture
def record_source():
    """Provide a judgment stream that never opens an external connection.

    args:
        None.

    returns:
        Iterator factory matching the acquisition pipeline boundary.
    """
    def record_source(*args):
        """Yield one source judgment without external API access.

        args:
            args: Acquisition settings, checkpoint and run identifier.

        returns:
            Iterator containing one structurally valid source judgment.
        """
        yield {
            "id": 42,
            "text": "Opis.",
            "source_payload": {"id": 42},
            "judgmentDate": "2020-05-12",
            "division": {"court": {"name": "Sąd Okręgowy w Płocku"}},
        }

    return record_source
