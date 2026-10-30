# SAOS Compensation

Study project for building a regression model that estimates compensation amounts from Polish court judgments available through the SAOS API.

The planned pipeline downloads a narrowly selected set of judgments, validates and stores them in PostgreSQL with pgvector, extracts a leakage-safe feature schema, vectorizes configured fields, and trains interchangeable regression models from YAML configuration.

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
uv run src/dataset/get_data.py --download-only --max-pages 1 --no-debug
uv run --extra extraction src/dataset/get_data.py --max-pages 1
```

[configs/dataset/api.yaml](configs/dataset/api.yaml) supplies the default values. Every key has a matching CLI argument; only explicitly supplied arguments override YAML:

| YAML | CLI | Meaning |
|---|---|---|
| `max_age_years` | `--max-age-years 10` | Judgment date range ending today |
| `max_pages` | `--max-pages 1` | Maximum dump pages; 0 means all pages |
| `download_path` | `--download-path tmp/experiment` | Output directory relative to the project root |
| `debug` | `--debug` / `--no-debug` | Save diagnostic evidence and intermediate results |
| `download_only` | `--download-only` / `--no-download-only` | Save source records without running the model |
| `timeout` | `--timeout 30` | HTTP connection/read timeout in seconds |

`--config-dir` selects another directory containing `api.yaml` and `basal.yaml`. Paths in YAML resolve from the project root; a relative `--config-dir` resolves from the current working directory. The download-only mode still validates both configuration files, but does not load model weights or GUGiK.

Dump pages contain up to 100 full judgments. SAOS applies `judgmentStartDate` and `judgmentEndDate`. The dump API has no documented text/court filter: the pipeline locally selects common-court sentences mentioning zadośćuczynienie and a numeric PLN amount. This scans more judgments than the earlier search-plus-detail prototype. A page limit is not a random or representative sample. Pagination follows the presence of `next`; the live dump response does not provide a total count.

One shared HTTP session reuses connections. Each request is made once with an explicit timeout; failures stop the run and are reported. Source responses are fetched afresh on reruns, so updates under an existing SAOS ID are visible. Duplicate IDs within a run are processed once. Records are replaced atomically by ID. Existing files from previous runs are retained; use a separate `download_path` for separate datasets and run only one process per output directory.

Normal extraction writes one primary record per judgment directly under `download_path`:

```text
<download_path>/
  <SAOS_ID>/
    record.json
```

`record.json` contains `id`, the configured `features`, configured `targets`, the derived `total`, `confirmed`, and review statuses. Missing values remain `brak informacji`, never an assumed zero. These are extraction candidates requiring review, including records where `confirmed` is true. `--download-only` instead writes `<SAOS_ID>/source.json` with the normalized text and metadata as its primary output; it does not produce feature/target records.

With `debug: false`, extraction writes only these primary records. It does not save raw responses, normalized source exports, model-call caches, quotes, scores or intermediate decisions. Runtime model weights remain in the project model cache. Existing diagnostic files are not deleted when debug is disabled.

With `debug: true`, additional artifacts explain how each result was produced:

```text
<download_path>/
  debug/
    <UTC_RUN_TIMESTAMP>/
      raw/                 # original HTTP bytes, URLs, acquisition times, hashes
      acquisition.json     # request, counts, rejections, errors, completion
    decisions/             # exact model requests and validated responses
    gugik/                 # official locality snapshot and source metadata
    filtering.json         # effective configuration, model identity, run outcome
  <SAOS_ID>/
    record.json
    debug/
      judgment.json        # original dump item
      metadata.json        # normalized source and joined court metadata
      source.txt
      config.json
      analysis.json        # source offsets, scores, discarded candidates, metadata provenance
      model_calls.json     # exact requests/responses, including reused model decisions
      facts.txt
      reasoning.txt
      review.txt
      record.json          # extracted feature values
      award.json           # monetary candidates, evidence and final verification
```

Acquisition summaries and failures are also printed to the console. Diagnostic scores describe model decisions; they are not causal explanations of a future compensation regressor. Model-call caching is enabled only in debug mode and is keyed by the request and pinned model identity. Cached answers are validated before use.

[configs/dataset/basal.yaml](configs/dataset/basal.yaml) chooses supported features, targets and three confidence thresholds. Prompts, the pinned Polish model revision and MPS execution remain fixed in code. `rok_wyroku` comes from the judgment date; `instancja` requires a confidence-checked procedural-stage decision rather than inference from court level. `wielkość miejscowości` matches the court seat against official GUGiK BDOT10k using Morfeusz inflection analysis and maps its population to a size class. Ambiguous matches remain missing. This is a current snapshot with no population measurement year, not historical population at judgment time; its provenance is available in debug mode. `miejscowosc_sadu` remains an optional supported source-form seat name.

Numeric PLN amounts support decimal commas, ordinary/nonbreaking thousands separators, `tys.`, `mln`, `mld` and their full Polish unit names. Decimal arithmetic preserves source offsets and handles descending compound units such as `4 mln 250 tys. zł`. Entirely written-out amounts remain unsupported.

The exploratory scripts in `notebooks/` retain their separate research outputs under `tmp/saos_feasibility`, `tmp/system_one` and `tmp/gemma`; the source pipeline does not import them.

Development cases live in [fixtures/](fixtures/). Each folder contains `judgment.json` plus the input text and model outputs. Cases are discovered from the files, not an ID list in config. Tests replay saved model responses offline. These are development snapshots, not independently labeled ground truth.

To run the actual model on those fixture inputs without SAOS acquisition:

```bash
uv run --extra extraction notebooks/03_system_one_polish.py --fixtures
```

Basal selects factual passages, extracts the configured descriptions, and separately interprets sentences containing monetary amounts. Targets are awarded compensation and other one-off damages; process/court costs, interest and rent are excluded. Ambiguous outputs remain for review. PostgreSQL ingestion, vectorization and regression training are not implemented.

The optional local Gemma fallback remains in `notebooks/04_gemma.py`; it does not download models. Research requirements are in [notebooks/wymagania.md](notebooks/wymagania.md).
