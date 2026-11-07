import subprocess

import sys

from pathlib import Path

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

