import asyncio
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path
import pytest
import requests
from src.dataset.storage import files as storage
from src.dataset.extraction.source.text import CURRENCY
from src.dataset.extraction.source.text import MONEY
from src.dataset.extraction.source.text import amount_in_pln
from src.dataset.extraction.source.text import extract_text

ROOT = Path(__file__).resolve().parents[1]



sys.path.insert(0, str(ROOT))



def test_output_limit_counts_existing_files_and_persists_across_resumes(tmp_path):
    """
    Verify output limit counts existing files and persists across resumes.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    from src.dataset.storage.files import (
        StorageLimitError,
        output_limit,
        write_bytes,
    )

    (tmp_path / "existing").write_bytes(b"1234")
    with output_limit(tmp_path, 8 / 1024**3):
        write_bytes(tmp_path / "new", b"5678")
        with pytest.raises(StorageLimitError):
            write_bytes(tmp_path / "overflow", b"9")
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == 8
    assert not (tmp_path / "overflow").exists()
    with pytest.raises(StorageLimitError):
        with output_limit(tmp_path, 8 / 1024**3):
            pass



def test_output_limit_preserves_existing_record_and_counts_atomic_temporary_file(
    tmp_path,
):
    """
    Verify output limit preserves existing record and counts atomic temporary file.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    from src.dataset.storage.files import (
        StorageLimitError,
        output_limit,
        write_bytes,
    )

    path = tmp_path / "record.json"
    path.write_bytes(b"1234")
    with output_limit(tmp_path, 6 / 1024**3):
        with pytest.raises(StorageLimitError):
            write_bytes(path, b"5678")
        assert path.read_bytes() == b"1234"
        write_bytes(path, b"12")
        write_bytes(tmp_path / "next", b"3456")
    assert sum(path.stat().st_size for path in tmp_path.iterdir()) == 6
    assert sorted(path.name for path in tmp_path.iterdir()) == ["next", "record.json"]

