import asyncio
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]




from dataset.cli import main as cli
from dataset.clients import api
from dataset.storage import checkpoints
from dataset.acquisition import pipeline
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


def test_notebooks_accept_shared_dataset_config():
    """
    Verify notebooks accept shared dataset config.

    args:
        None

    returns:
        None.
    """
    import importlib

    if (
        not (ROOT / "notebooks/01_saos_feasibility.py").is_file()
        or not (ROOT / "configs/dataset/api.yaml").is_file()
    ):
        pytest.skip("Local notebook experiment is unavailable")
    sys.path.insert(0, str(ROOT / "notebooks"))
    feasibility = importlib.import_module("01_saos_feasibility")
    feasibility.search_settings(CONFIG)
