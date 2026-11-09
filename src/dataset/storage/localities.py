"""Locality snapshot persistence and integrity checks."""

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ..clients.api import LOCALITIES_URL, fetch_localities
from ..clients.sql import read_localities
from .files import write_bytes, write_json


def load_localities(directory=None, timeout=30):
    """
    Read official locality attributes and verify retained snapshot integrity.

    args:
        directory: Input or output directory for this operation.
        timeout: Connection/read timeout in seconds.

    returns:
        Locality rows and snapshot provenance.
    """
    with tempfile.TemporaryDirectory() as temporary:
        path = (
            Path(directory) / "OT_ADMS_P.gpkg"
            if directory
            else Path(temporary) / "OT_ADMS_P.gpkg"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = path.with_suffix(".meta.json")
        if path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata["url"] != LOCALITIES_URL
                or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]
            ):
                raise ValueError("GUGiK snapshot integrity check failed")
        else:
            content = fetch_localities(timeout=timeout)
            write_bytes(path, content)
            metadata = {
                "url": LOCALITIES_URL,
                "sha256": hashlib.sha256(content).hexdigest(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "population_year": None,
            }
            write_json(metadata_path, metadata)
        return read_localities(path), metadata
