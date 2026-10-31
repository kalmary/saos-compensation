# SAOS Compensation

## Objective

Build a reproducible regression pipeline that estimates the compensation awarded in Polish court judgments available through SAOS. This is a study project for a data science course.

## Delivery plan

### 1. Acquire judgments and produce reviewable records

1. Read full judgments from the official SAOS `/api/dump/judgments` endpoint with a bounded judgment-date range. Resume dump pagination from checkpoint.json until the configured page limit or final page. The dump has no documented text/court filter, so select common-court compensation judgments locally. Join court/division metadata from `/api/dump/commonCourts`.
2. Use one shared HTTP session, an explicit timeout and one attempt per request. Report HTTP, pagination and validation failures; do not silently retry or discard malformed records. Refetch the pending page on resume, verify its IDs against the checkpoint and deduplicate by stable SAOS ID. Skip already committed records from the same dataset.
3. Treat `configs/dataset/api.yaml` as the defaults for acquisition: age range, page limit, total output size (max_gib), download path, debug and timeout. Expose matching CLI arguments; only explicitly supplied arguments override YAML. `--no-debug` must override a true YAML value.
4. Extract factual features separately from monetary targets. Write one `<download_path>/<SAOS_ID>/record.json` containing the configured features and targets, derived total and review statuses. Missing feature values are `not_available`, never an assumed zero. Persist a record only when the selected targets are established and the award confirmation and selected-feature leakage checks meet configured thresholds. Rejected cases produce no record or per-case diagnostic/cache files; only operational progress and aggregate reports remain.
5. Save additional provenance and explanation artifacts only when `debug: true`: accepted judgment payloads, request parameters and acquisition timestamps, normalized source text, source quotes/offsets, accepted/rejected passages, scores, exact model requests/responses, configuration and run errors. These support review of how a record was produced. With debug disabled, persist only primary records and the operational checkpoint; do not delete existing diagnostic files.
6. Keep the device in `configs/dataset/basal.yaml`: CPU by default, with explicit MPS or CUDA selection checked before downloads. Use the `pytorch-macos` dependency extra for local macOS work and `pytorch-cuda-linux` for NVIDIA GPU execution on Linux. Feature names are English while their model-facing descriptions and judgment text are Polish. basal.yaml selects the supported model name, while its revision and prompts remain pinned in code; feature/target selection and confidence thresholds are deliberate configuration choices.
7. Derive judgment year from SAOS metadata, require evidence for procedural instance, and derive the population class of the locality where the court is located from the matched GUGiK locality and its LICZBAMIESZKANCOW field. Preserve ambiguous/missing results. A current population snapshot does not establish population at judgment time.
8. Write `<download_path>/checkpoint.json` independently of debug. Pin the dataset ID, request dates, extraction configuration and GUGiK snapshot hash; store the next page/item and current page IDs. Commit the final record before advancing progress, and recognize an already saved record if interrupted between those writes. Resume automatically with the same command. Debug, timeout, max_gib and page limits may change; a different extraction configuration, age range or fresh dataset requires a new download path. Completed checkpoints perform no further acquisition.
9. Enforce max_gib against all files already in the output directory, including prior executions and atomic temporary writes; zero disables the limit. Stop before a write exceeds the budget, leaving a resumable checkpoint. Keep model weights in the ignored basal_weights/ directory, with one Hugging Face models--<owner>--<model> subfolder per model. Missing weights download automatically when extraction starts; cached files are reused. Keep this cache outside the size-limited output directory.
10. Preserve debug history in `<download_path>/debug/<RUN_ID>/` and `<download_path>/<SAOS_ID>/debug/<RUN_ID>/`, including an exact result.json snapshot linked by run ID from the primary record. Keep existing legacy diagnostics and shared immutable caches. Run one process per target path. Later PostgreSQL JSONB/pgvector ingestion remains separate from this local stage.

The first milestone is complete when a bounded acquisition produces records with explicit accepted/rejected counts, source validation, deliberate missing-value handling and debug-controlled traceability. A complete run does not establish target accuracy or approval for training.

### 2. Define the learning dataset

1. Specify how the awarded amount in PLN is extracted and how multiple awards, claimants, currencies, interest, appeals, and rejected claims are treated.
2. Label a small representative sample manually and use it to measure extraction quality.
3. Define a stable feature schema, beginning with:
   - cause of harm,
   - region,
   - court type and level,
   - judgment date,
   - claimant and harm characteristics supported by the judgment.
4. Evaluate a single open-weight instruction model supporting Polish for extracting and standardizing text features into this schema. Use a fixed model version and prompt, controlled categories, missing-value rules, and source evidence; measure quality on the labeled sample before choosing a model.
5. Store extracted fields together with the extractor name and version.

Label extraction and feature extraction must be separate. Predictor text must exclude the award, requested amount, calculations derived from it, and passages that directly explain the selected award. Related judgments and appeals must remain in one data split.

The second milestone is complete when target extraction and the feature schema reach agreed quality on the labeled sample and pass leakage review.

### 3. Vectorize and train

1. Create reproducible train, validation, and test splits before fitting preprocessing.
2. Encode each configured field appropriately: embeddings for free text, categorical encoders for categories, and scaled numeric values where needed.
3. Train a simple regression baseline first.
4. Select alternative classical or neural regression models through YAML configuration without changing pipeline code.
5. Compare MAE and RMSE and inspect errors by court, region, time range, and compensation band.

The third milestone is complete when a clean checkout can reproduce preprocessing, training, evaluation, and experiment metadata from configuration.

## Configuration layout

```text
configs/
  dataset/
    api.yaml             # acquisition defaults overridden only by explicit CLI arguments
    basal.yaml           # device, selected features/targets and extraction thresholds
  train.yaml             # training controls, when training is implemented
  models/
    <model-name>.yaml    # model choice and parameters, when regression is implemented
```

Expose only values deliberately selected between experiments. Keep model prompts, fixed implementation details and credentials out of experiment controls. Future data access, vectorization and training configuration should be added when implemented, without duplicating values across files. Secrets may be referenced by environment-variable name only.

## Initial technical decisions

- Runtime: Python 3.12 managed with `uv`; sync groups/extras first, then execute with `uv run --no-sync`.
- Dataset entry point: src/dataset/get_data.py; supporting modules live under src/dataset/processing/.
- Current outputs: local per-judgment records and a resume checkpoint; diagnostic artifacts only in debug mode.
- Planned database: PostgreSQL with JSONB and pgvector.
- Configuration: YAML.
- Problem type: supervised regression.
- First model: a simple baseline chosen after the target distribution is inspected.
- Extraction prototype: pinned Basal model with PyTorch; CPU by default, explicit MPS/CUDA when available. Independent labeled evaluation is required before approving extracted training data.

## Main risks

- The judgment text may not identify a single unambiguous award.
- Search terminology may miss relevant judgments or include unrelated uses of compensation terms.
- Reasoning text can reveal the target directly or indirectly.
- Inflation and changes in judicial practice can create temporal drift.
- Related cases can leak across random row-level splits.
- Sparse regions or courts can make subgroup metrics unstable.
