import subprocess
import sys


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
            "from dataset.embeddings import create_embeddings; "
            "import torch, transformers; assert callable(create_embeddings)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
