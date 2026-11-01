"""Download and filter judgments."""

import argparse
import asyncio
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dataset.acquisition.pipeline import run


def parse_arguments(arguments=None):
    """
    Parse explicit CLI overrides without replacing YAML defaults.

    args:
        arguments: Command-line arguments, or None for process arguments.

    returns:
        Namespace containing explicit CLI overrides.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Explicit CLI options override api.yaml; model settings are in basal.yaml.",
    )
    parser.add_argument(
        "--config-dir",
        help="Acquisition configuration directory (default: configs/dataset/acquisition)",
    )
    parser.add_argument(
        "--max-age-years",
        type=int,
        default=argparse.SUPPRESS,
        help="Judgment date range in years before today",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=argparse.SUPPRESS,
        help="Maximum dump pages (100 judgments each); 0 reads all pages",
    )
    parser.add_argument(
        "--max-gib",
        type=float,
        default=argparse.SUPPRESS,
        help="Total output-directory limit in GiB across resumed runs; 0 disables it",
    )
    parser.add_argument(
        "--download-path",
        default=argparse.SUPPRESS,
        help="Output directory relative to the project root",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=argparse.SUPPRESS,
        help="HTTP connection/read timeout in seconds, not a total download limit",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Save source/model calls and retain failed extraction attempts with valid=false",
    )
    return parser.parse_args(arguments)


def main():
    """
    Load effective settings and run the dataset pipeline.

    args:
        None

    returns:
        None.
    """
    from src.dataset.extraction.model.configuration import load_config

    options = vars(parse_arguments())
    directory = options.pop("config_dir")
    try:
        asyncio.run(run(load_config(directory, options)))
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed model decisions are saved; rerun the same command to resume.",
            flush=True,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
