# SAOS Compensation

Study project for building a regression model that estimates compensation amounts from Polish court judgments available through the SAOS API.

The pipeline acquires date-bounded judgments from the official SAOS dump, selects compensation cases locally, validates and stores them in PostgreSQL with pgvector, extracts a leakage-safe feature schema, vectorizes configured fields, and trains interchangeable regression models from YAML configuration.

## Installation

The project requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:kalmary/saos-compensation.git
cd saos-compensation
```

Install the base project dependencies:

```bash
uv sync
```

The `dev` group is currently empty. The `test` group includes `dev` and currently adds only pytest:

```bash
uv sync --group test
```

## Dataset acquisition and filtering

The entrypoint is [src/dataset/get_data.py](src/dataset/get_data.py). It reads full judgments from the official [SAOS bulk-download API](https://www.saos.org.pl/help/index.php/dokumentacja-api/api-pobierania-danych), joins court/division metadata from `/api/dump/commonCourts`, and separates factual features from monetary targets using Basal.

```bash
uv run src/dataset/get_data.py --help
uv run --extra pytorch-macos src/dataset/get_data.py --max-pages 1 --download-path tmp/dataset
```

[configs/dataset/api.yaml](configs/dataset/api.yaml) supplies the default values. Every key has a matching CLI argument; only explicitly supplied arguments override YAML:

| YAML | CLI | Meaning |
|---|---|---|
| `max_age_years` | `--max-age-years 10` | Judgment date range ending today |
| `max_pages` | `--max-pages 1` | Maximum dump pages; 0 means all pages |
| `max_gib` | `--max-gib 5` | Total output-directory size across resumes; 0 disables the limit |
| `download_path` | `--download-path tmp/experiment` | Output directory relative to the project root |
| `debug` | `--debug` / `--no-debug` | Save diagnostic evidence and intermediate results |
| `timeout` | `--timeout 30` | HTTP connection/read timeout in seconds |

`--config-dir` selects another directory containing `api.yaml` and `basal.yaml`. Paths in YAML resolve from the project root; a relative `--config-dir` resolves from the current working directory.

Dump pages contain up to 100 full judgments. SAOS applies `judgmentStartDate` and `judgmentEndDate`. The dump API has no documented text/court filter: the pipeline locally selects common-court sentences mentioning zadośćuczynienie and a numeric PLN amount. This scans more judgments than the earlier search-plus-detail prototype. A page limit is not a random or representative sample. Pagination follows the presence of `next`; the live dump response does not provide a total count.

One shared HTTP session reuses connections. Each request is made once with an explicit timeout; failures stop the run and are reported.

`<download_path>/checkpoint.json` is operational state saved regardless of debug. It pins the date range and extraction configuration and records the next page/item plus the current page IDs. Restart the same command to resume. Records are saved atomically before advancing the checkpoint; if the process stops between those writes, the saved record is recognized by its dataset ID and is not extracted again. Completed records on earlier pages are not revisited.

You may change debug, timeout, max_gib and the page limit while resuming; increasing the page limit continues the same dataset. Changing extraction settings or the age range requires a different download path. A completed checkpoint performs no further acquisition; use a new path for a fresh dataset/date range. A changed resumed SAOS page or GUGiK snapshot is rejected to avoid silently mixing incompatible data. Existing output folders without a checkpoint are processed as a new dataset. Run one process per output directory.

`max_gib` bounds the sum of all file sizes under download_path, including existing records, checkpoints and debug history from previous executions. Atomic temporary files count toward peak usage. The limit does not reset on resume. Writes that would exceed it are refused and the checkpoint remains resumable; increase max_gib to continue. A partial debug trace may remain for the interrupted judgment, but a partial primary record is never published. Legacy files also count. With a size limit, keep the Hugging Face model cache outside download_path: prefer `tmp/dataset` for output rather than the shared `tmp` directory containing `tmp/huggingface`.

Normal extraction writes one primary record per judgment directly under `download_path`:

```text
<download_path>/
  checkpoint.json
  <SAOS_ID>/
    record.json
```

`record.json` contains judgment, dataset and run IDs, the configured `features` and `targets`, derived `total`, `confirmed` and review statuses. Missing values remain `not_available`, never an assumed zero. These are extraction candidates requiring review, including records where `confirmed` is true.

With `debug: false`, extraction writes only primary records and checkpoint.json. It does not persist raw responses, source exports, model-call caches, quotes, scores or intermediate decisions. Runtime model weights remain in the project model cache. Existing diagnostic files are preserved.

With `debug: true`, additional artifacts explain how each result was produced:

```text
<download_path>/
  checkpoint.json
  debug/
    <RUN_ID>/
      raw/                 # original HTTP responses and acquisition metadata
      acquisition.json
      filtering.json
    decisions/             # shared immutable model-response cache
    gugik/                 # shared official locality snapshot
  <SAOS_ID>/
    record.json            # latest completed result, linked to its RUN_ID
    debug/
      <RUN_ID>/
        judgment.json
        metadata.json
        source.txt
        config.json
        result.json        # exact primary-record snapshot from this run
        analysis.json
        model_calls.json
        facts.txt
        reasoning.txt
        review.txt
        record.json        # extracted descriptions
        award.json
```

Every execution uses a separate UTC run ID; per-judgment diagnostics, result snapshots and acquisition/filtering reports from earlier runs are never overwritten. Acquisition summaries and failures are also printed to the console. Diagnostic scores describe model decisions; they are not causal explanations of a future compensation regressor. Model-call caching is enabled only in debug mode and is keyed by the request and pinned model identity. Cached answers are validated before use.

Install the platform-specific inference dependencies with `uv sync --extra pytorch-macos` on macOS or `uv sync --extra pytorch-cuda-linux` on Linux with an NVIDIA GPU. Both extras include Basal and PyTorch, with platform markers. The Linux PyPI PyTorch package supplies CUDA runtime libraries; a compatible NVIDIA driver and visible GPU are still required. No separate CUDA toolkit or MLX dependency is needed for these PyTorch backends. The existing `extraction` extra remains available for general CPU use.

[configs/dataset/basal.yaml](configs/dataset/basal.yaml) chooses the device, supported features, targets and three confidence thresholds. The pinned Polish model revision and prompts remain fixed in code. `device: cpu` is the default; `mps` selects the Apple GPU and `cuda` selects an NVIDIA GPU. The requested accelerator must be available; unavailable devices fail before source or model downloads. `judgment_year` comes from the judgment date; `instance` requires a confidence-checked procedural-stage decision rather than inference from court level. `court_locality_size` is the population class of the locality where the court is located. It matches the locality name from SAOS court metadata against official GUGiK BDOT10k using Morfeusz inflection analysis and reads LICZBAMIESZKANCOW to choose the class. Ambiguous matches remain missing. This is a current snapshot with no population measurement year, not historical population at judgment time; its provenance is available in debug mode. `court_seat` remains an optional supported source-form seat name.

Numeric PLN amounts support decimal commas, ordinary/nonbreaking thousands separators, `tys.`, `mln`, `mld` and their full Polish unit names. Decimal arithmetic preserves source offsets and handles descending compound units such as `4 mln 250 tys. zł`. Entirely written-out amounts remain unsupported.

The exploratory scripts in `notebooks/` retain their separate research outputs under `tmp/saos_feasibility`, `tmp/system_one` and `tmp/gemma`; the source pipeline does not import them.

Development cases live in [fixtures/](fixtures/). Each folder contains `judgment.json` plus the input text and model outputs. Cases are discovered from the files, not an ID list in config. Tests replay saved model responses offline. These are development snapshots, not independently labeled ground truth.

To run the actual model on those fixture inputs without SAOS acquisition:

```bash
uv run --extra pytorch-macos notebooks/03_system_one_polish.py --fixtures
```

Basal selects factual passages, extracts the configured descriptions, and separately interprets sentences containing monetary amounts. Targets are awarded compensation and other one-off damages; process/court costs, interest and rent are excluded. Ambiguous outputs remain for review. PostgreSQL ingestion, vectorization and regression training are not implemented.

The optional local Gemma fallback remains in `notebooks/04_gemma.py`; it does not download models. Research requirements are in [notebooks/wymagania.md](notebooks/wymagania.md).

Implementation identifiers, diagnostics and comments use English. Feature names are English; their Polish descriptions and source text are sent to Basal. Polish remains in model prompts, monetary decision categories, source evidence and text-matching rules. Source-pipeline missing values use `not_available`; older notebook/fixture snapshots retain their original representation.
