"""Atomic output writes with a cumulative directory size limit."""

import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_OUTPUT = ContextVar("output_budget", default=None)


class StorageLimitError(RuntimeError):
    pass


@contextmanager
def output_limit(directory, max_gib):
    """
    Apply a file-size budget that includes previous output in the directory.

    args:
        directory: Input or output directory for this operation.
        max_gib: Cumulative directory limit in GiB; zero disables it.

    returns:
        Context controlling output writes.
    """
    root = Path(directory).resolve()
    budget = None
    if max_gib:
        used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        budget = {"root": root, "used": used, "limit": int(max_gib * 1024**3)}
    token = _OUTPUT.set(budget)
    try:
        if budget is not None and budget["used"] >= budget["limit"]:
            raise StorageLimitError("Output directory has reached max_gib")
        yield
    finally:
        _OUTPUT.reset(token)


def write_bytes(path, content):
    """
    Atomically replace an output file without exceeding the active size budget.

    args:
        path: Path to the relevant input or output file.
        content: Bytes to write.

    returns:
        None.
    """
    path = Path(path)
    budget = _OUTPUT.get()
    counted = budget is not None and path.resolve().is_relative_to(budget["root"])
    previous = path.stat().st_size if path.exists() else 0
    # Include the temporary file in peak usage before replacing an existing file.
    if counted and budget["used"] + len(content) > budget["limit"]:
        raise StorageLimitError(f"Writing {path.name} would exceed max_gib")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if counted:
        budget["used"] += len(content) - previous


def write_json(path, value):
    """
    Serialize a JSON artifact through the atomic, size-limited writer.

    args:
        path: Path to the relevant input or output file.
        value: Value to serialize or adapt.

    returns:
        None.
    """
    write_bytes(
        path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    )
