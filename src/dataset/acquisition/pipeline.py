"""Model lifecycle and orchestration of acquisition and extraction."""

import asyncio
import json
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from .download import download
from .checkpoints import load_checkpoint
from ..extraction.quality import feature_rejection_reason, rejection_reason
from ..storage.records import add_metadata_features, write_result
from ..storage.files import StorageLimitError, output_limit, remove_tree, write_bytes, write_json

ROOT = Path(__file__).resolve().parents[3]


def check_device(device):
    """
    Require the requested PyTorch backend to be available before inference.

    args:
        device: Explicit CPU or accelerator selection.

    returns:
        None.
    """
    try:
        import torch
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "PyTorch could not be loaded. Install --extra pytorch-macos on macOS or --extra extraction for CPU "
            "or --extra pytorch-cuda-linux for CUDA on Linux."
        ) from error
    if device == "cuda" and (
        torch.version.cuda is None or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA was requested but is unavailable. On Linux, install --extra pytorch-cuda-linux "
            "and check the NVIDIA driver and GPU visibility, or set device: cpu."
        )

    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable. Install --extra pytorch-macos on Apple Silicon, or set device: cpu."
        )


async def run(config):
    """
    Execute the configured pipeline within the cumulative output-size budget.

    args:
        config: Effective pipeline or acquisition settings.

    returns:
        None.
    """
    directory = ROOT / config["api"]["download_path"]
    try:
        with output_limit(directory, config["api"]["max_gib"]):
            await process(config)
    except StorageLimitError as error:
        print(
            f"{error}. Increase max_gib to continue from checkpoint.json.", flush=True
        )


async def process(config):
    """
    Resume acquisition and extract configured features and monetary targets.

    args:
        config: Effective pipeline or acquisition settings.

    returns:
        None.
    """
    from src.dataset.extraction import basal_filtering as basal, decisions
    from src.dataset.extraction.prompts import MODEL_FILES

    directory = ROOT / config["api"]["download_path"]
    checkpoint = load_checkpoint(config)
    limit = config["api"]["max_pages"]
    if checkpoint["complete"] or (limit and checkpoint["page"] >= limit):
        print("No pending judgments within the configured page limit.", flush=True)
        return
    model_cache = ROOT / "basal_weights"
    for cache in (model_cache, ROOT / "embedding_weights"):
        if config["api"]["max_gib"] and cache.resolve().is_relative_to(
            directory.resolve()
        ):
            raise ValueError(
                f"{cache.name} must be outside download_path when max_gib is enabled"
            )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    check_device(config["system_one"]["device"])
    check_device(config["embedding"]["device"])
    write_json(directory / "checkpoint.json", checkpoint)
    localities = None
    if "court_locality_size" in config["metadata_fields"]:
        import morfeusz2

        from src.dataset.acquisition.localities import load_localities

        rows, metadata = load_localities(
            ROOT / "tmp" / "gugik",
            timeout=config["api"]["timeout"],
        )
        digest = metadata["sha256"]
        if checkpoint["localities_sha256"] not in (None, digest):
            raise ValueError(
                "GUGiK snapshot changed since the checkpoint; use a different download_path"
            )
        checkpoint["localities_sha256"] = digest
        write_json(directory / "checkpoint.json", checkpoint)
        localities = rows, morfeusz2.Morfeusz(), metadata
    from basal.server import Server, parser
    from huggingface_hub import snapshot_download

    model = config["system_one"]
    model_path = snapshot_download(
        model["model"],
        revision=model["revision"],
        cache_dir=model_cache,
        allow_patterns=MODEL_FILES,
        max_workers=1,
    )
    args = parser().parse_args(
        [
            "--model",
            model_path,
            "--mode",
            model["mode"],
            "--device",
            model["device"],
            "--template-prefix",
            model["template_prefix"],
            "--name",
            model["model"],
            "--context-overflow",
            model["context_overflow"],
        ]
    )
    server = Server(args)
    worker = asyncio.create_task(server.worker())
    decisions.REQUEST_COUNTS.update(cached=0, computed=0)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "dataset_id": checkpoint["dataset_id"],
        "config": config,
        "engine_version": "2.4.1",
        "engine_arguments": vars(args),
        "processed": 0,
        "confirmed_awards": 0,
        "rejected_records": 0,
        "complete": False,
    }
    try:
        with closing(download(config["api"], checkpoint, run_id)) as records:
            for record in records:
                identifier = record["id"]
                pending = directory / str(identifier) / "pending"
                write_json(pending / "judgment.json", record["source_payload"])
                write_json(pending / "config.json", config)
                write_bytes(
                    pending / "source.txt", (record["text"] + "\n").encode("utf-8")
                )
                cache_token = decisions.DECISION_CACHE.set(pending / "decisions")
                token = decisions.MODEL_CALLS.set([] if config["api"]["debug"] else None)
                try:
                    result = await basal.extract(
                        server, record["text"], config, identifier
                    )
                    if config["api"]["debug"]:
                        result["model_calls"] = decisions.MODEL_CALLS.get()
                finally:
                    decisions.MODEL_CALLS.reset(token)
                    decisions.DECISION_CACHE.reset(cache_token)
                coverage_reason = feature_rejection_reason(result, config)
                reason = rejection_reason(result, config)
                if coverage_reason and reason != coverage_reason:
                    detail = (
                        coverage_reason
                        if result.get("fields_evaluated", bool(result.get("features")))
                        else "text feature extraction skipped"
                    )
                    reason = f"{reason}; {detail}"
                if reason is not None and (
                    not config["api"]["debug"] or coverage_reason
                ):
                    record["rejected"] = True
                    summary["rejected_records"] += 1
                    remove_tree(pending)
                    if not any(pending.parent.iterdir()):
                        pending.parent.rmdir()
                    print(f"Rejected judgment {identifier}: {reason}", flush=True)
                    continue
                result["source_text"] = record["text"]
                add_metadata_features(
                    result, record, config["metadata_fields"], localities
                )
                write_result(result, directory / str(identifier), config)
                remove_tree(pending)
                summary["processed"] += 1
                summary["confirmed_awards"] += int(
                    result["award"].get("confirmed", False)
                )
                if reason is not None:
                    print(
                        f"Saved debug judgment {identifier}: {reason}",
                        flush=True,
                    )
                else:
                    print(f"Filtered judgment {identifier}", flush=True)
        summary["complete"] = True
    except asyncio.CancelledError:
        summary["interrupted"] = True
        raise
    except Exception as error:
        summary["error"] = str(error)
        raise
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        summary["requests"] = decisions.REQUEST_COUNTS.copy()
        print(
            json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"config", "engine_arguments"}
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
