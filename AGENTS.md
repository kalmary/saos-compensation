# Project goal

`saos-compensation` is a data science study project for building a regression model that estimates the compensation awarded in Polish court judgments published through the SAOS API.

The project should provide a reproducible pipeline that:

- downloads only relevant judgments through a narrowly scoped SAOS API query,
- validates, filters, and stores the source JSON without losing its structure,
- uses PostgreSQL with pgvector for relational data, JSON payloads, and embeddings,
- extracts the target amount and a repeatable set of judgment features,
- prevents target leakage from passages that disclose or justify the awarded amount,
- converts configured feature fields into model-ready vectors,
- supports interchangeable regression models selected entirely through YAML configuration,
- trains and evaluates experiments reproducibly.

This is a regression problem. Do not turn it into judgment classification or general legal-document search unless explicitly requested.

# Planned workflow

## 1. Data acquisition and storage

- Inspect the current SAOS API schema before implementing the client.
- Select the narrowest supported query that returns judgments concerning compensation; avoid downloading the entire corpus and filtering it only locally.
- Preserve the request parameters and acquisition timestamp so a dataset can be reproduced.
- Validate identifiers, court metadata, dates, judgment text, and pagination before persistence.
- Deduplicate judgments by their stable SAOS identifier and make ingestion restartable.
- Keep the original response payload in a PostgreSQL JSON/JSONB column and materialize fields needed for filtering and analysis as ordinary columns.
- Use pgvector for generated embeddings. Do not introduce a separate vector database without evidence that PostgreSQL is insufficient.
- Keep raw, validated, and model-ready data distinguishable so preprocessing never overwrites source data.

## 2. Target and feature definition

- Define one target value precisely: the awarded compensation amount in PLN for the modeled claim.
- Record how multiple claimants, multiple awards, partial awards, currencies, interest, and rejected claims are handled before training.
- Exclude ambiguous examples rather than silently assigning an uncertain target.
- Build a repeatable representation of facts available for prediction, initially including the cause of harm, region, court type, court level, judgment date, claimant characteristics, injury or harm characteristics, and other supported case facts.
- Treat feature extraction as a separate, versioned preprocessing step. A Polish-language information-extraction model may be evaluated, including a free model with Polish weights, but no specific model is assumed before a small labeled evaluation.

## Leakage prevention

- Extract labels separately from predictor features.
- Do not use the awarded amount, requested amount, calculations based on that amount, or passages explaining why that amount was selected as model input.
- Split data before fitting encoders or other learned preprocessing.
- Group related judgments, appeals, or duplicate proceedings in the same split.
- Review extracted features and nearest-neighbor results for indirect disclosure of the target.
- Keep an explicit allowlist of feature fields presented to the regressor.

## 3. Vectorization and modeling

- Convert every configured feature field to a stable numeric representation appropriate to that field. Text fields may use Polish-language embeddings; categorical and numeric fields should retain their meaning rather than be embedded without need.
- Store the preprocessing version, embedding model name, and embedding dimension with generated features.
- Start with a simple regression baseline before adding a neural network.
- Evaluate with regression metrics such as MAE and RMSE, plus error distributions across courts, regions, time ranges, and target-value bands.
- Fit preprocessing only on the training split and apply the fitted artifacts unchanged to validation and test data.

# Configuration

YAML configuration is the public experiment interface. Keep data access, preprocessing, feature selection, model choice, and training values out of source code when they vary between experiments.

Use this layout as functionality is implemented:

```text
configs/
  data.yaml
  preprocessing.yaml
  train.yaml
  models/
    <model-name>.yaml
```

- `data.yaml` defines the SAOS request, validation rules, database connection environment-variable name, and dataset split.
- `preprocessing.yaml` defines target extraction, the feature allowlist, leakage exclusions, field encoders, and embedding model.
- `train.yaml` defines training-only settings such as seed, batch size, epochs, optimizer, scheduler, and early stopping when the selected model needs them. It must not duplicate model structure or data configuration.
- Each file in `configs/models/` defines one regression model and its parameters, whether it is a classical ML model or a neural network.
- References between configuration files must be explicit. Do not add a configuration option until two real experiments need different values.

# Engineering rules

- Read the relevant implementation, tests, fixtures, configuration, and documentation before editing.
- Make the smallest complete change and reuse existing project patterns.
- Use descriptive, general names. Do not over-modify unrelated code.
- Do not add comments or metadata referring to AI tools or the development conversation.
- Keep API access, database access, feature extraction, vectorization, and model training separable and testable.
- Validate external API data and model configuration at their boundaries. Do not hide malformed records or partial failures.
- Keep secrets out of YAML and source files; configurations may name environment variables only.
- Preserve raw source data and avoid destructive migrations without an explicit migration and backup plan.
- Add focused tests for non-trivial parsing, target extraction, leakage filtering, and configuration-driven model selection.
- Use Python 3.12, the repository `.venv`, `uv`, and the committed `uv.lock`. Run project commands with `uv run`.
- Do not add a dependency until the implementation imports it.
- After each task, update the summary in `STATUS.md` and append a dated entry describing completed changes, findings, and remaining work. Keep only the latest 50 log entries in chronological order.
- Never run Git operations, including commit, push, pull, branch, merge, or reset, without explicit user approval.
