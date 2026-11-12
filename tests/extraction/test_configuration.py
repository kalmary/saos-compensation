"""Acquisition configuration boundaries and checkpoint compatibility."""

import pytest
import yaml

from dataset.extraction.model import configuration


@pytest.mark.parametrize(
    ("file", "key", "value"),
    [
        ("features.yaml", "metadata_fields", ["unknown"]),
        ("features.yaml", "metadata_fields", ["instance", "instance"]),
        ("features.yaml", "text_fields", {}),
        ("features.yaml", "text_fields", {"damage": " "}),
        ("features.yaml", "text_fields", {"damage": 123}),
        ("features.yaml", "text_fields", {"damage": "Szkoda?", "instance": "Etap?"}),
        ("features.yaml", "required_fields", []),
        ("features.yaml", "required_fields", ["damage", "cause"]),
        ("targets.yaml", "patterns", []),
        ("targets.yaml", "unexpected", True),
        ("basal.yaml", "feature_fields", ["damage"]),
    ],
)
def test_invalid_acquisition_configuration_is_rejected(
    config_directory,
    file,
    key,
    value,
):
    """Reject malformed settings before downloads or inference begin.

    args:
        config_directory: Isolated acquisition YAML directory.
        file: File to modify.
        key: Setting to replace or add.
        value: Invalid setting value.

    returns:
        None.
    """
    path = config_directory / file
    settings = configuration.read_config(path)
    settings[key] = value
    path.write_text(yaml.safe_dump(settings))
    with pytest.raises(ValueError):
        configuration.load_config(config_directory)


@pytest.mark.parametrize("phrases", [[], [" "], ["oddala {unknown}"]])
def test_invalid_denial_patterns_fail_before_inference(config_directory, phrases):
    """Validate the active denial category before any model is initialized.

    args:
        config_directory: Isolated acquisition YAML directory.
        phrases: Invalid denial template list.

    returns:
        None.
    """
    path = config_directory / "targets.yaml"
    settings = configuration.read_config(path)
    settings["patterns"]["judgment-denial"] = phrases
    path.write_text(yaml.safe_dump(settings, allow_unicode=True))
    with pytest.raises(ValueError, match="nonempty|Unknown"):
        configuration.load_config(config_directory)
