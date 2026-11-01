"""Pairwise cosine similarity for existing embeddings on PyTorch devices."""

import torch


def cosine_similarity(embeddings, reference_embeddings, device="cpu"):
    """
    Compare every embedding with every reference embedding without gradients.

    args:
        embeddings: Real-valued vector or matrix shaped (N, D).
        reference_embeddings: Real-valued vector or matrix shaped (M, D).
        device: cpu, mps or cuda; unavailable accelerators raise RuntimeError.

    returns:
        Float32 tensor shaped (N, M) on the requested device, with scores in [-1, 1].
        Single vectors count as one row; empty, nonfinite and zero vectors are rejected.
    """

    if device not in ("cpu", "mps", "cuda"):
        raise ValueError("device must be cpu, mps or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")

    with torch.no_grad():
        normalized = []
        for name, values in (
            ("embeddings", embeddings),
            ("reference_embeddings", reference_embeddings),
        ):
            tensor = torch.as_tensor(values)
            if tensor.is_complex():
                raise ValueError(f"{name} must contain real-valued embeddings")
            tensor = tensor.to(dtype=torch.float32, device=device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 2 or not all(tensor.shape):
                raise ValueError(f"{name} must be a nonempty vector or matrix")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must contain only finite values")
            # Scaling first prevents norm overflow and underflow for finite float32 inputs.
            scale = tensor.abs().amax(dim=1, keepdim=True)
            if (scale == 0).any():
                raise ValueError(f"{name} must not contain zero vectors")
            tensor = tensor / scale
            normalized.append(
                tensor / torch.linalg.vector_norm(tensor, dim=1, keepdim=True)
            )
        left, right = normalized
        if left.shape[1] != right.shape[1]:
            raise ValueError("Embedding dimensions must match")
        return (left @ right.T).clamp(-1, 1)
