import sys

import pytest

from dataset.acquisition import pipeline
from dataset.extraction.model import configuration as config_module


@pytest.mark.parametrize(
    ("device", "cuda", "mps", "mode"),
    [
        ("cpu", False, False, "eager"),
        ("cuda", True, False, "eager"),
        ("mps", False, True, "mps"),
    ],
)
def test_explicit_device_configuration_and_availability(
    config_directory,
    monkeypatch,
    device,
    cuda,
    mps,
    mode,
):
    """
    Verify explicit device configuration and availability.

    args:
        config_directory: Isolated editable acquisition settings.
        monkeypatch: Scoped replacements for the test.
        device: Explicit CPU or accelerator selection.
        cuda: Input for this operation.
        mps: Input for this operation.
        mode: Input for this operation.

    returns:
        None.
    """
    import types

    import yaml

    settings = config_module.read_config(config_directory / "basal.yaml")
    settings["device"] = device
    (config_directory / "basal.yaml").write_text(yaml.safe_dump(settings))
    config = config_module.load_config(config_directory)
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
