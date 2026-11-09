"""Acquisition configuration boundaries and checkpoint compatibility."""

import asyncio
import shutil
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]




from dataset.extraction import features
from dataset.extraction.model import configuration, decisions


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


def test_configured_question_is_used_for_extraction_and_relevance(
    config_directory, monkeypatch
):
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
            assert (
                questions["opis"]["schema"]["properties"]["damage"]["description"]
                == question
            )
            seen.append("extract")
            return {
                "opis": {
                    "data": {"damage": source},
                    "fields": {
                        "damage": {"start": 0, "end": len(source), "confidence": 0.9}
                    },
                }
            }
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
