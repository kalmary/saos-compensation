"""Model snapshot settings and worker lifetime without downloaded weights."""

from types import SimpleNamespace
import asyncio

import pytest

from dataset.clients.basal import load_model, open_extractor
from dataset.extraction.model.configuration import load_config


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_worker_is_closed_on_success_failure_and_cancellation(tmp_path, failure):
    """Release the shared worker even when processing fails or is interrupted.

    args:
        tmp_path: Isolated model cache location.
        failure: Optional exception raised by the consuming pipeline.

    returns:
        None.
    """
    config = load_config()
    started, stopped, loaded = [], [], []

    async def worker():
        """Keep a fake inference worker alive until its context closes.

        args:
            None.

        returns:
            None.
        """
        started.append(True)
        try:
            await asyncio.Event().wait()
        finally:
            stopped.append(True)

    def model_loader(model, cache):
        """Record the configured model and cache without downloading weights.

        args:
            model: Effective pinned model configuration.
            cache: Model cache destination.

        returns:
            Placeholder model path.
        """
        loaded.append((model, cache))
        return "model-path"

    def server_factory(model, path):
        """Provide a controlled worker for the requested model snapshot.

        args:
            model: Effective inference settings.
            path: Loaded snapshot path.

        returns:
            Server exposing only the worker needed by this test.
        """
        assert model == config["system_one"]
        assert path == "model-path"
        return SimpleNamespace(worker=worker)

    async def run():
        """Enter the extraction context and simulate the requested exit condition.

        args:
            None.

        returns:
            None.
        """
        async with open_extractor(
            config, tmp_path, model_loader=model_loader, server_factory=server_factory
        ) as extract:
            assert callable(extract)
            await asyncio.sleep(0)
            if failure is not None:
                raise failure()

    if failure is None:
        asyncio.run(run())
    else:
        with pytest.raises(failure):
            asyncio.run(run())
    assert loaded == [(config["system_one"], tmp_path)]
    assert started == stopped == [True]


def test_model_snapshot_keeps_pinned_revision_and_external_cache(monkeypatch, tmp_path):
    """Preserve restricted snapshot downloads and the one-worker setting.

    args:
        monkeypatch: Replacement for the external snapshot API.
        tmp_path: Weight cache destination.

    returns:
        None.
    """
    import huggingface_hub

    calls = []

    def snapshot_download(model, **kwargs):
        """Capture snapshot arguments without contacting the model host.

        args:
            model: Requested model identifier.
            kwargs: Snapshot revision, cache and file settings.

        returns:
            Placeholder cached snapshot path.
        """
        calls.append((model, kwargs))
        return "model-path"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    model = load_config()["system_one"]
    assert load_model(model, tmp_path) == "model-path"
    name, options = calls.pop()
    assert name == model["model"]
    assert options["revision"] == model["revision"]
    assert options["cache_dir"] == tmp_path
    assert options["max_workers"] == 1
    assert "model*.safetensors" in options["allow_patterns"]
    assert "README.md" not in options["allow_patterns"]
