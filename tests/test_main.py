import subprocess
import sys
from pathlib import Path


def test_main():
    """
    Verify main.

    args:
        None

    returns:
        None.
    """
    result = subprocess.run(
        [sys.executable, Path(__file__).parents[1] / "src" / "main.py"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == "Hello from saos-compensation!\n"


def test_get_data_help(tmp_path):
    """Verify the installed command exposes acquisition options outside the repository.

    args:
        tmp_path: Working directory without project source files.

    returns:
        None.
    """
    result = subprocess.run(
        [str(Path(sys.executable).parent / "get-data"), "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: get-data" in result.stdout
    for option in (
        "--config-dir",
        "--max-age-years",
        "--max-pages",
        "--max-gib",
        "--download-path",
        "--timeout",
        "--debug",
        "--no-debug",
    ):
        assert option in result.stdout


def test_embeddings_without_basal():
    """Verify reusable embeddings can be imported without optional Basal.

    args:
        None

    returns:
        None.
    """
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['basal'] = None; "
            "from src.dataset.embeddings import create_embeddings; "
            "import torch, transformers; assert callable(create_embeddings)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
