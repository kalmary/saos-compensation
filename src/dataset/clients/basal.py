"""Basal model loading, worker lifetime and extraction integration."""

import asyncio
from contextlib import asynccontextmanager

from ..extraction.basal_filtering import extract
from ..extraction.model.prompts import MODEL_FILES


def load_model(model, cache):
    """Load the pinned model snapshot from the existing external weight cache.

    args:
        model: Effective model settings with a pinned revision.
        cache: Basal weight cache outside size-limited dataset output.

    returns:
        Local model snapshot path.
    """
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model["model"],
        revision=model["revision"],
        cache_dir=cache,
        allow_patterns=MODEL_FILES,
        max_workers=1,
    )


def create_server(model, model_path):
    """Build a Basal server using the configured device and inference arguments.

    args:
        model: Validated inference settings.
        model_path: Local pinned model snapshot.

    returns:
        Configured inference server, without starting its worker.
    """
    from basal.server import Server, parser

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
    return Server(args)


@asynccontextmanager
async def open_extractor(config, cache, *, model_loader=None, server_factory=None):
    """Keep one model worker alive and close it on success, failure or cancellation.

    args:
        config: Effective extraction settings.
        cache: Model snapshot cache directory.
        model_loader: Optional snapshot loader; resolved at runtime when omitted.
        server_factory: Optional inference server factory for the loaded snapshot.

    returns:
        Async context yielding an extraction callable accepting text and identifier.
    """
    model_loader = load_model if model_loader is None else model_loader
    server_factory = create_server if server_factory is None else server_factory
    model = config["system_one"]
    server = server_factory(model, model_loader(model, cache))
    worker = asyncio.create_task(server.worker())

    async def extract_record(text, identifier):
        """Extract a judgment using the shared server and fixed configuration.

        args:
            text: Normalized judgment text.
            identifier: Stable SAOS judgment identifier.

        returns:
            Source-grounded feature and target extraction result.
        """
        return await extract(server, text, config, identifier)

    try:
        yield extract_record
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
