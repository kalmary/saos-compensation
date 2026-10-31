"""Pairwise embedding similarity and input boundaries."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from src.dataset.processing.cosine_similarity import cosine_similarity


def test_pairwise_similarity_matches_known_angles_without_gradients():
    """
    Preserve row order and compare identical, opposite and orthogonal vectors.

    args:
        None.

    returns:
        None.
    """
    vectors = torch.tensor([[1.0, 0.0], [0.0, 2.0]], requires_grad=True)
    references = [[3.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
    scores = cosine_similarity(vectors, references)
    torch.testing.assert_close(
        scores, torch.tensor([[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    )
    assert scores.device.type == "cpu"
    assert scores.dtype == torch.float32
    assert not scores.requires_grad
    torch.testing.assert_close(vectors, torch.tensor([[1.0, 0.0], [0.0, 2.0]]))


@pytest.mark.parametrize("magnitude", [1.0, 1e30, 1e-30])
def test_single_vectors_keep_matrix_shape_and_scale_invariance(magnitude):
    """
    Normalize very large and small finite vectors without changing their direction.

    args:
        magnitude: Positive scale applied to both coordinates.

    returns:
        None.
    """
    scores = cosine_similarity([magnitude, magnitude], [1.0, 0.0])
    assert scores.shape == (1, 1)
    assert scores.item() == pytest.approx(2**-0.5)


@pytest.mark.parametrize(
    ("vectors", "references", "message"),
    [
        ([], [1.0, 0.0], "nonempty"),
        ([1j, 0j], [1.0, 0.0], "real-valued"),
        ([[[1.0, 0.0]]], [1.0, 0.0], "vector or matrix"),
        ([1.0, 0.0], [1.0, 0.0, 0.0], "dimensions must match"),
        ([0.0, 0.0], [1.0, 0.0], "zero vectors"),
        ([float("nan"), 1.0], [1.0, 0.0], "finite"),
        ([1.0, 0.0], [float("inf"), 0.0], "finite"),
    ],
)
def test_invalid_embeddings_are_rejected(vectors, references, message):
    """
    Reject undefined cosine similarities and malformed input shapes.

    args:
        vectors: Left input embeddings.
        references: Right input embeddings.
        message: Expected validation message.

    returns:
        None.
    """
    with pytest.raises(ValueError, match=message):
        cosine_similarity(vectors, references)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_devices_do_not_fall_back_to_cpu(monkeypatch, device):
    """
    Fail explicitly when a selected accelerator is unavailable.

    args:
        monkeypatch: Scoped availability replacement.
        device: Requested accelerator.

    returns:
        None.
    """
    backend = torch.cuda if device == "cuda" else torch.backends.mps
    monkeypatch.setattr(backend, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0], device=device)


def test_unknown_device_is_rejected():
    """
    Reject unsupported backend names.

    args:
        None.

    returns:
        None.
    """
    with pytest.raises(ValueError, match="device must be"):
        cosine_similarity([1.0, 0.0], [1.0, 0.0], device="auto")
