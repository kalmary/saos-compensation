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
