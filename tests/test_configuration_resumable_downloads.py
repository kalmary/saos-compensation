"""Acquisition configuration boundaries and checkpoint compatibility."""


import asyncio

import shutil

import sys

from pathlib import Path

import pytest

import yaml

ROOT = Path(__file__).resolve().parents[1]


sys.path.insert(0, str(ROOT))


from src.dataset.extraction import features

from src.dataset.extraction.model import configuration

from src.dataset.extraction.model import decisions

@pytest.fixture
def config_directory(tmp_path):
    """Copy acquisition configuration into an isolated editable directory.

    args:
        tmp_path: Temporary test directory.

    returns:
        Directory containing all acquisition YAML files.
    """
    shutil.copytree(ROOT / "configs/dataset/acquisition", tmp_path, dirs_exist_ok=True)
    return tmp_path


def test_changed_question_invalidates_checkpoint(config_directory, monkeypatch, tmp_path):
    """Resume unchanged settings and reject a changed question on resume.

    args:
        config_directory: Isolated acquisition YAML directory.
        monkeypatch: Scoped repository-root replacement.
        tmp_path: Temporary dataset root.

    returns:
        None.
    """
    from src.dataset.acquisition import checkpoints
    from src.dataset.storage.files import write_json

    config = configuration.load_config(config_directory)
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
    checkpoint = checkpoints.load_checkpoint(config)
    write_json(tmp_path / config["api"]["download_path"] / "checkpoint.json", checkpoint)
    assert checkpoints.load_checkpoint(config) == checkpoint
    path = config_directory / "features.yaml"
    settings = configuration.read_config(path)
    settings["text_fields"]["damage"] = "Co zostało uszkodzone?"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True))
    changed = configuration.load_config(config_directory)
    with pytest.raises(ValueError, match="Extraction settings changed"):
        checkpoints.load_checkpoint(changed)

