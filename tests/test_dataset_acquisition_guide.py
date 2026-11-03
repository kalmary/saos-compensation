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


sys.path.insert(0, str(ROOT))


from src.dataset import get_data

from src.dataset.acquisition import checkpoints

from src.dataset.acquisition import pipeline

from src.dataset.storage import files as storage

from src.dataset.storage import records

from src.dataset.acquisition import api

from src.dataset.acquisition import download as acquisition

from src.dataset.extraction import basal_filtering as basal

from src.dataset.extraction.targets import amounts

from src.dataset.extraction.model import configuration as config_module

from src.dataset.extraction.model import decisions

from src.dataset.extraction.model import prompts

from src.dataset.extraction.source.text import CURRENCY

from src.dataset.extraction.source.text import MONEY

from src.dataset.extraction.source.text import amount_in_pln

from src.dataset.extraction.source.text import extract_text

CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}


def test_notebooks_accept_shared_dataset_config():
    """
    Verify notebooks accept shared dataset config.

    args:
        None

    returns:
        None.
    """
    import importlib

    if not (ROOT / "notebooks/01_saos_feasibility.py").is_file() or not (ROOT / "configs/dataset/api.yaml").is_file():
        pytest.skip("Local notebook experiment is unavailable")
    sys.path.insert(0, str(ROOT / "notebooks"))
    feasibility = importlib.import_module("01_saos_feasibility")
    feasibility.search_settings(CONFIG)

