# SAOS Compensation

## Repository structure

```text
configs/dataset/
  api.yaml                  # acquisition defaults and output limits
  basal.yaml                # device, feature/target selection and thresholds
src/
  dataset/
    get_data.py             # Dataset CLI entry point
    acquisition/            # SAOS access, pipeline orchestration and checkpoints
    extraction/             # Basal features, targets and text validation
    embeddings/             # Sentence encoding and target matching
    storage/                # Record export, atomic writes and output-size limits
  main.py                   # package startup placeholder
basal_weights/              # ignored per-model weight cache
notebooks/                  # local experiments; only .gitkeep is tracked
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
uv sync --locked --extra pytorch-macos --extra embedding --group test

# Linux: PyTorch with NVIDIA CUDA support
uv sync --locked --extra pytorch-cuda-linux --extra embedding --group test
```

Other installation options:

| Selection | Purpose |
|---|---|
| `uv sync --locked` | Base dependencies; no model inference |
| `--group test` | pytest and the currently empty dev group |
| `--group dev` | Currently empty |
| `--extra extraction` | General Basal/PyTorch installation |
| `--extra embedding` | PyTorch/Transformers sentence embeddings |
| `--extra gemma` | Optional Ollama research fallback |

Subsequent commands use `uv run --no-sync` so execution does not modify dependencies.

```bash
uv run --no-sync pytest -q
```

## Dataset

[src/dataset/get_data.py](src/dataset/get_data.py) is the entry point. Pipeline orchestration and checkpoint validation live under `acquisition/`; record export and atomic writes live under `storage/`. Feature extraction and sentence encoding have their own `extraction/` and `embeddings/` packages. Within `extraction/`, `basal_filtering.py` coordinates extraction and `features.py` reviews predictors. Model configuration, prompts and decision caching live under `model/`; normalization and source batching under `source/`; monetary evidence and target resolution under `targets/`. `src.dataset` exports `load_config`, `extract_text` and `extract` for reuse in future inference.

```bash
uv run --no-sync src/dataset/get_data.py --help
uv run --no-sync src/dataset/get_data.py --max-pages 1 --download-path tmp/dataset
```

### Configuration

YAML provides defaults; explicit CLI arguments override only matching acquisition settings.

| Configuration | Controls |
|---|---|
| [api.yaml](configs/dataset/api.yaml) | `max_age_years`, `max_pages`, `max_gib`, `download_path`, `debug`, `timeout` |
| [basal.yaml](configs/dataset/basal.yaml) | `model`, `device`, general feature selection, single `total` target and confidence thresholds |
| [embedding.yaml](configs/dataset/embedding.yaml) | Encoder, revision, device, batch/token sizes, target/exclusion keyword groups and `minimum_cosine_similarity: 0.7` |
| [target_pattern.yaml](configs/dataset/target_pattern.yaml) | Judgment/request phrase templates |

Acquisition options use hyphens, e.g. `--max-gib 5`; `--debug` and `--no-debug` select diagnostic output. `--config-dir` selects another configuration directory. `device: mps` selects the Apple GPU in the current macOS configuration; use `cpu` or `cuda` on other setups. Unavailable accelerators fail explicitly.

### Acquisition and extraction

- **SAOS:** date-bounded full judgments from the official [dump API](https://www.saos.org.pl/help/index.php/dokumentacja-api/api-pobierania-danych), with court metadata joined from its court dump. Text/court filtering is local because the dump does not expose those filters.
- **HTTP:** one shared Requests session, explicit timeout and no automatic retries.
- **Target matching:** fast Polish-capable multilingual MiniLM embeddings; only sentences containing a numeric PLN amount and a configured target-word stem are encoded in batches; judgment/request keyword groups restrict eligible classes, with shared stems as a fallback; the highest eligible cosine score selects the class. Prior payments, procedural costs, mixed request/outcome sentences and other payments (interest, periodic rent, fines) have competing exclusion groups. A winner must score strictly above the configured 0.7; several qualifying groups still yield one highest-scoring class. Exact ties prefer exclusion, then pattern order. Exclusion winners cannot supply a target. Judgment is mandatory; request is optional.
- **Target parsing:** read exactly one numeric PLN amount from the embedding-selected judgment sentence, without Basal classification or verification and without scanning other ruling clauses. Explicit per-person awards are multiplied only when recipients can be counted from that sentence. Multiple amounts or unresolved per-person counts leave the target missing.
- **Basal:** extracts predictor fields (including prior_payment and procedural instance) and reviews factuality, field relevance and target leakage. It does not determine the regression target.
- **GUGiK and Morfeusz:** population class of the locality where the court is located; ambiguous matches remain missing. Population measurement year is unknown.

The target is the amount in the selected judgment sentence, with per-person multiplication where explicit; separate awards in other sentences are not aggregated. Litigation costs, interest and periodic rent are excluded. Missing values are not_available, never an assumed zero. Normal records require a judgment match and an established principal. Debug preserves failed attempts with valid=false. Cosine similarity is not confidence that a legal role or award is correct; independent evaluation remains necessary.

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

`max_pages: 0` reads all pages; a page limit is not random sampling. `max_gib` caps all files in the output directory across resumes, including diagnostic history; `0` disables it. Model weights are cached separately under the ignored `basal_weights/models--Remek--basal-2.0-4.5B/` directory. Missing files download automatically through Hugging Face when extraction starts; cached weights are reused. This is handled by the model loader, not PyTorch itself. Embedding weights download separately into ignored embedding_weights/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/. Both device settings must match the intended CPU/MPS/CUDA runtime. Only runtime model/tokenizer/configuration/calibration files are requested, with one file-download worker; Xet can still display separate transfer/reconstruction progress.

With `debug: false`, completed cases contain only record.json and award.json. With `debug: true`, only source.txt and model_calls.json are added. Evidence, confidence scores and run metadata are never mixed into the basic files; award.json includes the target validity flag. Existing artifacts from older versions are not automatically removed; use a new output path for the clean layout. Temporary pending state supports interruption recovery and is removed after the outcome is committed.

The only regression target is `total`: the numeric amount from the selected judgment sentence, multiplied by an explicit recipient count for per-person awards. award.json contains only total and valid. Target extraction and validation run first. With debug disabled, cases without an admissible target skip instance and feature inference. Every case must have accepted source-backed descriptions for strictly more than 50% of configured text features (currently at least four of six); metadata and prior_payment do not count. Insufficient coverage discards the case even under debug. Among cases passing this rule, debug also saves attempts with unconfirmed targets. A missing judgment match skips text-feature inference and fails coverage when text features are selected. Multiple beneficiaries are supported. The configured numeric feature `prior_payment` stores confirmed earlier principal paid, rounded to an integer PLN value; `0` means no confirmed amount. It is saved only in record.json and is never added to the target. Features still undergo leakage checks and missing values stay not_available. award.json always includes valid: true when target and feature admission pass, false otherwise; training must filter valid=true. The four case files are committed before temporary pending state is removed. Target sentences are selected by cosine similarity. The target is parsed directly from the selected sentence; Basal neither classifies amounts nor verifies a proposed total. Exactly one numeric PLN occurrence is required. Separate ruling clauses are not scanned or summed. The amount_minimum_probability setting applies only to the prior_payment predictor. Features are extracted together from factual findings in batches of up to 6000 characters. There is no fallback extraction pass; unsupported fields remain missing. Candidate sentences are checked for factuality, field relevance and leakage, preserving source offsets. Console output reports accepted fields, missing descriptions and concrete rejection checks with scores; target failures are distinguished from skipped feature extraction. These diagnostics reuse existing model answers. There is no preliminary paragraph/block classification. Extraction version 20 requires a new output path for datasets created with an older version.

The workflow avoids preliminary passage classification and checks only field candidates. Local comparisons of pinned mini and 4.5B models use the same saved judgments; results are inspection artifacts, not a validated accuracy estimate.

Research experiments can be kept locally in `notebooks/`; only `.gitkeep` is tracked. Tests for those experiments skip when the local files are absent. See [PROJECT.md](PROJECT.md) for the full workflow.

## ML pipeline

_To be implemented: reviewed labels, leakage checks, grouped splits and field vectorization._

## Training

_To be implemented: regression baseline and configuration-driven model training._

## Evaluation

_To be implemented: MAE, RMSE and error analysis across case groups._

## Inference

_To be implemented: compensation prediction from model-ready case features._

[Jak to działa](src/dataset/how-it-works.md) describes the complete extraction flow, passes, examples and proposed stratification in Polish.
