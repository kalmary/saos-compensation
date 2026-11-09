import subprocess
import sys


def test_module_help(tmp_path):
    """
    Verify the module CLI works outside the repository.

    args:
        tmp_path: Working directory without project source files.

    returns:
        None.
    """
    result = subprocess.run(
        [sys.executable, "-m", "dataset.cli", "--help"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--download-path" in result.stdout
    assert "--no-debug" in result.stdout
