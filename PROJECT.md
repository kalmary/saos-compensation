# SAOS Compensation

## Objective

Build a reproducible regression pipeline that estimates the compensation awarded in Polish court judgments available through SAOS. This is a study project for a data science course.

## Delivery plan

### 1. Acquire judgments and produce reviewable records

1. Read full judgments from the official SAOS `/api/dump/judgments` endpoint with a bounded judgment-date range. Follow dump pagination until the configured page limit or the final page. The dump has no documented text/court filter, so select common-court compensation judgments locally. Join court/division metadata from `/api/dump/commonCourts`.
2. Use one shared HTTP session, an explicit timeout and one attempt per request. Report HTTP, pagination and validation failures; do not silently retry or discard malformed records. Fetch fresh source data on reruns and deduplicate by stable SAOS ID within each run.
3. Treat `configs/dataset/api.yaml` as the defaults for acquisition: age range, page limit, download path, debug, download-only and timeout. Expose matching CLI arguments; only explicitly supplied arguments override YAML. `--no-debug` and `--no-download-only` must override true YAML values.
4. Extract factual features separately from monetary targets. Write one `<download_path>/<SAOS_ID>/record.json` containing the configured features and targets, derived total and review statuses. Missing values are `not_available`, never an assumed zero. In download-only mode, write source.json instead of an extracted record.
5. Save additional provenance and explanation artifacts only when `debug: true`: original response bytes, request parameters and acquisition timestamps, normalized source text, source quotes/offsets, accepted/rejected passages, scores, exact model requests/responses, configuration and run errors. These support review of how a record was produced. With debug disabled, persist only primary records; do not delete existing diagnostic files.
6. Keep the device in `configs/dataset/basal.yaml`: CPU by default, with explicit MPS or CUDA selection checked before downloads. Use the `pytorch-macos` dependency extra for local macOS work and `pytorch-cuda-linux` for NVIDIA GPU execution on Linux. Prompts and the pinned model version remain fixed in code; feature/target selection and confidence thresholds are deliberate configuration choices.
7. Derive judgment year from SAOS metadata, require evidence for procedural instance, and match court-seat population classes against official GUGiK data. Preserve ambiguous/missing results. A current population snapshot does not establish population at judgment time.
8. Replace records atomically by ID and use separate download paths for separate datasets. Raw responses are retained per debug run, while per-judgment diagnostics and filtering.json contain the latest run; reruns overwrite those diagnostics. Later database ingestion should use PostgreSQL JSONB and pgvector; it is separate from this local acquisition/extraction stage.

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

- Runtime: Python 3.12 managed with `uv`.
- Current outputs: local per-judgment records; diagnostic artifacts only in debug mode.
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
