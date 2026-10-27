# SAOS Compensation

## Objective

Build a reproducible regression pipeline that estimates the compensation awarded in Polish court judgments available through SAOS. This is a study project for a data science course.

## Delivery plan

### 1. Acquire and store judgments

1. Inspect the current SAOS API fields and filters.
2. Define the narrowest query that selects judgments relevant to compensation and save the query parameters with each dataset version.
3. Download paginated results with restartable ingestion and deduplicate them by SAOS judgment ID.
4. Validate required metadata and judgment content; reject or quarantine incomplete records.
5. Store source payloads and normalized fields in PostgreSQL. Enable pgvector in the same database for embeddings.

The first milestone is complete when the same request can be rerun without duplicate rows and produces a documented count of downloaded, accepted, and rejected judgments.

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
  data.yaml             # SAOS query, validation, database, and split
  preprocessing.yaml    # target rules, feature fields, leakage rules, encoders
  train.yaml            # training hyperparameters used by trainable models
  models/
    <model-name>.yaml   # model type and model-specific parameters
```

Configuration files may reference secrets through environment-variable names, but must never contain credentials. `train.yaml` is used for neural-network hyperparameters and any shared training controls; model-specific parameters belong in the selected file under `configs/models/`.

## Initial technical decisions

- Runtime: Python 3.12 managed with `uv`.
- Storage: PostgreSQL with JSONB and pgvector.
- Configuration: YAML.
- Problem type: supervised regression.
- First model: a simple baseline chosen after the target distribution is inspected.
- Polish text model: selected only after evaluation on the labeled extraction sample.

## Main risks

- The judgment text may not identify a single unambiguous award.
- Search terminology may miss relevant judgments or include unrelated uses of compensation terms.
- Reasoning text can reveal the target directly or indirectly.
- Inflation and changes in judicial practice can create temporal drift.
- Related cases can leak across random row-level splits.
- Sparse regions or courts can make subgroup metrics unstable.
