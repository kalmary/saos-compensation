# SAOS Compensation

## Repository structure

```text
configs/dataset/
  api.yaml                  # acquisition defaults and output limits
  basal.yaml                # device, feature/target selection and thresholds
src/
  dataset/
    get_data.py             # dataset CLI entry point
    processing/             # Basal extraction, text, locality and storage modules
  main.py                   # package startup placeholder
basal_weights/              # ignored per-model weight cache
notebooks/                  # exploratory scripts
fixtures/                   # development snapshots, when available
tests/                     # automated checks
PROJECT.md                  # workflow and design
STATUS.md                   # current status and dated history
```

## About the repository

A data science study project for predicting compensation amounts from Polish court judgments in SAOS. The implemented stage acquires judgments and extracts factual features and awarded amounts. PostgreSQL/pgvector storage, regression training and inference are planned.

## Installation

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:kalmary/saos-compensation.git
cd saos-compensation
```

Sync the environment before running commands. Choose the inference extra for your platform:

```bash
# macOS: PyTorch with CPU and Apple MPS support
uv sync --locked --extra pytorch-macos --group test

# Linux: PyTorch with NVIDIA CUDA support
uv sync --locked --extra pytorch-cuda-linux --group test
```

Other installation options:

| Selection | Purpose |
|---|---|
| `uv sync --locked` | Base dependencies; no model inference |
| `--group test` | pytest and the currently empty dev group |
| `--group dev` | Currently empty |
| `--extra extraction` | General Basal/PyTorch installation |
| `--extra gemma` | Optional Ollama research fallback |

Subsequent commands use `uv run --no-sync` so execution does not modify dependencies.

```bash
uv run --no-sync pytest -q
```

## Dataset

[src/dataset/get_data.py](src/dataset/get_data.py) is the entry point. Its supporting modules are in `processing/`. `src.dataset` exports `load_config`, `extract_text` and `extract` for reuse in future inference.

```bash
uv run --no-sync src/dataset/get_data.py --help
uv run --no-sync src/dataset/get_data.py --max-pages 1 --download-path tmp/dataset
```

### Configuration

YAML provides defaults; explicit CLI arguments override only matching acquisition settings.

| Configuration | Controls |
|---|---|
| [api.yaml](configs/dataset/api.yaml) | `max_age_years`, `max_pages`, `max_gib`, `download_path`, `debug`, `timeout` |
| [basal.yaml](configs/dataset/basal.yaml) | `model`, `device`, feature/target selection and confidence thresholds |

Acquisition options use hyphens, e.g. `--max-gib 5`; `--debug` and `--no-debug` select diagnostic output. `--config-dir` selects another configuration directory. `device: mps` selects the Apple GPU in the current macOS configuration; use `cpu` or `cuda` on other setups. Unavailable accelerators fail explicitly.

### Acquisition and extraction

- **SAOS:** date-bounded full judgments from the official [dump API](https://www.saos.org.pl/help/index.php/dokumentacja-api/api-pobierania-danych), with court metadata joined from its court dump. Text/court filtering is local because the dump does not expose those filters.
- **HTTP:** one shared Requests session, explicit timeout and no automatic retries.
- **Basal:** pinned Polish model, separate factual-feature and monetary-target extraction; English feature names with Polish descriptions and source text.
- **GUGiK and Morfeusz:** population class of the locality where the court is located; ambiguous matches remain missing. Population measurement year is unknown.

Targets are awarded compensation for non-material harm and other one-off damages. Litigation costs, interest and periodic rent are excluded. Missing values are `not_available`, never an assumed zero. Only records with at least one established selected target, sufficient award confidence and safe selected evidence are persisted. Rejected cases leave no record or per-case diagnostics. Accepted outputs still require independent review.

### Output and resume

```text
<download_path>/
  checkpoint.json
  <SAOS_ID>/
    record.json             # feature values only
    award.json              # selected targets and total only
    source.txt              # debug=true only
    model_calls.json        # debug=true only
```

Restart the same command to resume from the checkpoint. During extraction, `<SAOS_ID>/pending/` stores the source, configuration and completed model decisions even with debug disabled. Restarting reuses these decisions (reported as cached); only the interrupted request may need repeating. Pending state counts toward max_gib and is removed after acceptance or rejection. Saved records are recognized without repeating extraction. Dates and extraction settings stay pinned; debug, timeout, size and page limits may change. A completed dataset or different extraction settings require a new output path. Run one process per output directory.

`max_pages: 0` reads all pages; a page limit is not random sampling. `max_gib` caps all files in the output directory across resumes, including diagnostic history; `0` disables it. Model weights are cached separately under the ignored `basal_weights/models--Remek--basal-2.0-4.5B/` directory. Missing files download automatically through Hugging Face when extraction starts; cached weights are reused. This is handled by the model loader, not PyTorch itself. Only runtime model/tokenizer/configuration/calibration files are requested, with one file-download worker; Xet can still display separate transfer/reconstruction progress.

With `debug: false`, completed cases contain only record.json and award.json. With `debug: true`, only source.txt and model_calls.json are added. Evidence, confidence scores and run metadata are never mixed into the basic files; award.json includes the target validity flag. Existing artifacts from older versions are not automatically removed; use a new output path for the clean layout. Temporary pending state supports interruption recovery and is removed after the outcome is committed.

Target extraction and validation run first. With debug disabled, cases without an admissible target skip instance and feature inference. With debug enabled, every completed extraction attempt is saved, including missing or unconfirmed targets. Features still undergo leakage checks and missing values stay not_available. award.json always includes valid: true when target and feature admission pass, false otherwise; training must filter valid=true. The four case files are committed before temporary pending state is removed. Accepted cases extract feature candidates directly from packed blocks (up to 1200 characters), preserving source offsets and whole sentences where they fit. Only candidate sentences receive factuality and leakage review, shared across features. There is no preliminary paragraph/block classification. Extraction version 8 requires a new output path for datasets created with an older version.

The earlier notebook run used MPS and many fresh model calls; the recent slow run used CPU. The current workflow additionally removes preliminary passage classification. No live speed or quality comparison has been measured.

Research experiments remain in `notebooks/`; their outputs are separate from the dataset CLI. See [PROJECT.md](PROJECT.md) for the full workflow.

## ML pipeline

_To be implemented: reviewed labels, leakage checks, grouped splits and field vectorization._

## Training

_To be implemented: regression baseline and configuration-driven model training._

## Evaluation

_To be implemented: MAE, RMSE and error analysis across case groups._

## Inference

_To be implemented: compensation prediction from model-ready case features._

[Jak to działa](src/dataset/how-it-works.md) describes the complete extraction flow, passes, examples and proposed stratification in Polish.
