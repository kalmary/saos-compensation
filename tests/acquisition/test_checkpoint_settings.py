"""Acquisition configuration boundaries and checkpoint compatibility."""

import pytest
import yaml

from dataset.extraction.model import configuration


def test_changed_question_invalidates_checkpoint(
    config_directory,
    monkeypatch,
    tmp_path,
):
    """Resume unchanged settings and reject a changed question on resume.

    args:
        config_directory: Isolated acquisition YAML directory.
        monkeypatch: Scoped repository-root replacement.
        tmp_path: Temporary dataset root.

    returns:
        None.
    """
    from dataset.storage import checkpoints
    from dataset.storage.files import write_json

    config = configuration.load_config(config_directory)
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
    checkpoint = checkpoints.load_checkpoint(config)
    write_json(
        tmp_path / config["api"]["download_path"] / "checkpoint.json", checkpoint
    )
    assert checkpoints.load_checkpoint(config) == checkpoint
    path = config_directory / "features.yaml"
    settings = configuration.read_config(path)
    settings["text_fields"]["damage"] = "Co zostało uszkodzone?"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True))
    changed = configuration.load_config(config_directory)
    with pytest.raises(ValueError, match="Extraction settings changed"):
        checkpoints.load_checkpoint(changed)


@pytest.mark.parametrize("previous_version", [26, 27, 28])
def test_prefix_matching_version_rejects_previous_extraction_checkpoint(
    config_directory, monkeypatch, tmp_path, previous_version
):
    """Keep earlier search policies separate from first-monetary-prefix datasets.

    args:
        previous_version: Earlier clause or prefix search version.
        config_directory: Current acquisition settings.
        monkeypatch: Scoped output-root replacement.
        tmp_path: Isolated dataset root.

    returns:
        None.
    """
    from dataset.storage import checkpoints

    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
    config = configuration.load_config(config_directory)
    previous = {**config, "version": f"saos-extraction-{previous_version}"}
    checkpoint = checkpoints.load_checkpoint(previous)
    checkpoints.save_checkpoint(tmp_path / config["api"]["download_path"], checkpoint)
    assert config["version"] == "saos-extraction-29"
    with pytest.raises(ValueError, match="Extraction settings changed"):
        checkpoints.load_checkpoint(config)
