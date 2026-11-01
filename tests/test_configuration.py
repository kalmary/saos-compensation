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
from src.dataset.extraction.model import configuration, decisions


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


@pytest.mark.parametrize(
    ("file", "key", "value"),
    [
        ("features.yaml", "metadata_fields", ["unknown"]),
        ("features.yaml", "metadata_fields", ["instance", "instance"]),
        ("features.yaml", "text_fields", {"cause": "Co się stało?"}),
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
def test_invalid_acquisition_configuration_is_rejected(config_directory, file, key, value):
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


def test_configured_question_is_used_for_extraction_and_relevance(config_directory, monkeypatch):
    """Use one YAML question for extraction and the subsequent relevance review.

    args:
        config_directory: Isolated acquisition YAML directory.
        monkeypatch: Scoped model replacement.

    returns:
        None.
    """
    question = "Co zostało uszkodzone?"
    path = config_directory / "features.yaml"
    settings = configuration.read_config(path)
    settings["text_fields"] = {"damage": question}
    path.write_text(yaml.safe_dump(settings, allow_unicode=True))
    config = configuration.load_config(config_directory)
    source = "Złamana noga."
    seen = []

    async def ask(server, state, questions, config):
        """Check the configured wording and return source-backed feature evidence.

        args:
            server: Unused inference server.
            state: Source text or review context.
            questions: Extraction or review questions.
            config: Effective acquisition configuration.

        returns:
            Controlled extraction or safe relevance review.
        """
        if "opis" in questions:
            assert questions["opis"]["schema"]["properties"]["damage"]["description"] == question
            seen.append("extract")
            return {"opis": {
                "data": {"damage": source},
                "fields": {"damage": {"start": 0, "end": len(source), "confidence": 0.9}},
            }}
        assert questions["fit_damage"]["instructions"].endswith(question)
        seen.append("review")
        return {
            "sekcja": {"choice": "fakty", "confidence": 0.9},
            "wyciek": {"noul": 0.01},
            "fit_damage": {"noul": 0.9},
        }

    monkeypatch.setattr(decisions, "ask", ask)
    result = asyncio.run(features.extract_features(None, source, config, 42))
    assert result["features"] == {"damage": source}
    assert seen == ["extract", "review"]


def test_reorganized_defaults_resume_existing_checkpoint(config_directory, monkeypatch, tmp_path):
    """Preserve the existing extraction signature and reject changed questions on resume.

    args:
        config_directory: Isolated acquisition YAML directory.
        monkeypatch: Scoped repository-root replacement.
        tmp_path: Temporary dataset root.

    returns:
        None.
    """
    from src.dataset.acquisition import checkpoints
    from src.dataset.storage.files import write_json

    path = config_directory / "features.yaml"
    settings = configuration.read_config(path)
    settings["text_fields"] = {
        "cause": "Co spowodowało szkodę?",
        "damage": "Jakiej szkody doznał poszkodowany?",
        "limitations": "Jakich czynności poszkodowany nie mógł wykonywać?",
    }
    path.write_text(yaml.safe_dump(settings, allow_unicode=True))
    config = configuration.load_config(config_directory)
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)
    checkpoint = checkpoints.load_checkpoint(config)
    assert checkpoint["signature"] == "62304782ddfc5c202232fa69a1e2c5ea57bbb084cd2a45fae048d6a6b221abb1"
    write_json(tmp_path / config["api"]["download_path"] / "checkpoint.json", checkpoint)
    assert checkpoints.load_checkpoint(config) == checkpoint
    path = config_directory / "features.yaml"
    settings = configuration.read_config(path)
    settings["text_fields"]["damage"] = "Co zostało uszkodzone?"
    path.write_text(yaml.safe_dump(settings, allow_unicode=True))
    changed = configuration.load_config(config_directory)
    with pytest.raises(ValueError, match="Extraction settings changed"):
        checkpoints.load_checkpoint(changed)
