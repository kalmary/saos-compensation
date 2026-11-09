import asyncio
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]




from dataset.cli import main as cli
from dataset.acquisition import api, checkpoints, pipeline
from dataset.acquisition import download as acquisition
from dataset.extraction import basal_filtering as basal
from dataset.extraction.model import configuration as config_module
from dataset.extraction.model import decisions, prompts
from dataset.extraction.source.text import (
    CURRENCY,
    MONEY,
    amount_in_pln,
    extract_text,
)
from dataset.extraction.targets import amounts
from dataset.storage import files as storage
from dataset.storage import records

CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}


def test_config_and_path_are_independent_of_working_directory(monkeypatch, tmp_path):
    """
    Verify config and path are independent of working directory.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    expected = config_module.load_config()
    monkeypatch.chdir(tmp_path)
    config = config_module.load_config()
    assert config == expected
    assert list(config["fields"]) == ["cause", "damage", "limitations"]
    assert pipeline.ROOT / config["api"]["download_path"] == ROOT / "tmp"
    start, end, params = api.search_settings(CONFIG, date(2024, 2, 29))
    assert (start, end) == (date(2014, 2, 28), date(2024, 2, 29))
    assert params["judgmentStartDate"] == "2014-02-28"
    assert "all" not in params


@pytest.mark.parametrize("text_field", ["cause", "damage", "limitations"])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize("low_confidence", [False, True])
@pytest.mark.parametrize("missing_target", [False, True])
@pytest.mark.parametrize("sparse_features", [False, True])
def test_pipeline_writes_separate_outputs_and_stops_worker(
    monkeypatch,
    tmp_path,
    debug,
    low_confidence,
    missing_target,
    sparse_features,
    capsys,
    text_field,
):
    """
    Verify pipeline writes separate outputs and stops worker.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.
        low_confidence: Whether target verification falls below the threshold.
        missing_target: Whether no monetary target could be established.
        sparse_features: Whether missing text evidence must discard even debug cases.
        capsys: Captured console output.
        text_field: Sole populated text predictor.

    returns:
        None.
    """
    import argparse
    import sys
    import types

    config = config_module.load_config()
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
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
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
            "features": {
                **{field: "brak informacji" for field in config["fields"]},
                text_field: "brak informacji" if sparse_features else text,
            },
            "evidence": {}
            if sparse_features
            else {text_field: [{"start": 0, "end": len(text)}]},
            "description_reviews": [
                {
                    "start": 0,
                    "end": len(text),
                    "answers": {
                        "wyciek": {"noul": 0.01},
                        f"fit_{text_field}": {"noul": 0.9},
                    },
                }
            ],
            "award": amounts.finish_award(
                {
                    "suma": "brak informacji" if missing_target else "100",
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
    output = capsys.readouterr().out
    if low_confidence and not missing_target:
        assert "low-confidence award" in output
    if missing_target:
        assert "final award verification not run" in output
    directory = tmp_path / "tmp"
    path = directory / "42"
    if sparse_features or ((low_confidence or missing_target) and not debug):
        assert not path.exists()
        assert stopped == [True]
        return
    record = json.loads((path / "record.json").read_text())
    assert record == {
        **{field: "not_available" for field in config["fields"]},
        text_field: "Opis.",
        "judgment_year": 2020,
        "instance": "pierwsza",
        "court_seat": "Płocku",
    }
    assert json.loads((path / "award.json").read_text()) == {
        "total": "not_available" if missing_target else "100",
        "valid": not low_confidence and not missing_target,
    }
    assert {item.name for item in path.iterdir()} == (
        {"record.json", "award.json", "model_calls.json", "source.txt"}
        if debug
        else {"record.json", "award.json"}
    )
    assert stopped == [True]


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
    options = vars(cli.parse_arguments(arguments))
    assert options.pop("config_dir") is None
    assert options == expected
    original = config_module.load_config()
    merged = config_module.load_config(overrides=options)
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

    shutil.copytree(ROOT / "configs/dataset/acquisition", tmp_path, dirs_exist_ok=True)
    api = {**CONFIG, "max_pages": -1, "debug": True}
    (tmp_path / "api.yaml").write_text(yaml.safe_dump(api))
    for name in ("features.yaml", "targets.yaml"):
        (tmp_path / name).write_text(
            (ROOT / "configs/dataset/acquisition" / name).read_text()
        )
    (tmp_path / "basal.yaml").write_text(
        (ROOT / "configs/dataset/acquisition/basal.yaml").read_text()
    )
    options = vars(
        cli.parse_arguments(
            ["--config-dir", str(tmp_path), "--max-pages", "1", "--no-debug"]
        )
    )
    config = config_module.load_config(options.pop("config_dir"), options)
    assert config["api"] == {**api, "max_pages": 1, "debug": False}


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

    settings = config_module.read_config(
        ROOT / "configs/dataset/acquisition/basal.yaml"
    )
    settings["device"] = device
    (tmp_path / "basal.yaml").write_text(yaml.safe_dump(settings))
    (tmp_path / "api.yaml").write_text(yaml.safe_dump(CONFIG))
    for name in ("features.yaml", "targets.yaml"):
        (tmp_path / name).write_text(
            (ROOT / "configs/dataset/acquisition" / name).read_text()
        )
    config = config_module.load_config(tmp_path)
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
    pipeline.check_device(device)


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
        pipeline.check_device(device)


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
        pipeline.check_device("cpu")


def test_removed_download_only_option_is_rejected():
    """
    Verify removed download only option is rejected.

    args:
        None

    returns:
        None.
    """
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--download-only"])
