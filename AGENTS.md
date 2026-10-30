# Project goal

`saos-compensation` is a data science study project for building a regression model that estimates the compensation awarded in Polish court judgments published through the SAOS API.

The project should provide a reproducible pipeline that:

- acquires date-bounded judgments through the official SAOS dump API and selects compensation cases,
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

- Read `PROJECT.md` before changing behavior. It is the authoritative description of the current acquisition, output and configuration workflow.
- Use the official SAOS dump endpoints and validate their current schema. Apply supported date filters, select relevant compensation cases locally, join court metadata and deduplicate by stable SAOS ID.
- Keep CLI and `configs/dataset/api.yaml` aligned: YAML supplies defaults; explicit CLI arguments override only their corresponding values.
- Produce per-judgment feature/target records and an atomic checkpoint.json under download_path. Resume from the saved position and advance only after the final record is durable. Preserve the dataset date bounds and reject incompatible extraction settings. Additional source, evidence, model-call and intermediate artifacts must be written only when debug is enabled. Preserve existing artifacts when debug is disabled; store new diagnostics and run reports under a unique run ID so later runs cannot overwrite their history.
- Enforce max_gib on the total output directory across resumptions, including existing files and debug history. Refuse writes that exceed the limit and keep the model cache outside a size-limited output directory.
- Use explicit HTTP timeouts and a shared session. Surface failures without manual retry loops or silently missing records.
- Keep API access, extraction and future PostgreSQL/pgvector ingestion separable. Raw, validated and model-ready representations must not overwrite one another.

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

The current interface is `configs/dataset/api.yaml` for acquisition and `configs/dataset/basal.yaml` for device, feature/target selection and confidence thresholds. Future training settings belong in `configs/train.yaml`, with regression model choices and parameters under `configs/models/` when implemented. Keep references explicit and add a configuration option only when real experiments need different values.

# Engineering rules

- Read `PROJECT.md` and the relevant implementation, tests, fixtures, configuration, and documentation before editing.
- Keep `PROJECT.md` current after behavior or design changes. Edit or replace outdated statements in their existing sections; do not append contradictory updates or maintain a history there. Dated history belongs in `STATUS.md`.
- Use English for implementation identifiers, configuration keys, comments, docstrings, errors and diagnostics. Retain Polish only where required by model-facing prompts/schema identifiers, source-text matching, original evidence or Polish-text analysis. New metadata names and machine-readable status values must be English.
- Use `pytorch-macos` for local PyTorch work on macOS and `pytorch-cuda-linux` for Linux CUDA dependencies. CPU is the default device; explicitly selected accelerators must be available before inference starts.
- Comments and parameter descriptions should explain non-obvious behavior or constraints, not restate names.
- Every function, including helpers and methods, must have a short English triple-quoted docstring with a description, an `args:` section listing its parameters, and a `returns:` section describing its result. Use `None` when there are no arguments or return value.
- Make the smallest complete change and reuse existing project patterns.
- Use descriptive, general names. Do not over-modify unrelated code.
- Do not add comments or metadata referring to AI tools or the development conversation.
- Keep API access, database access, feature extraction, vectorization, and model training separable and testable.
- Validate external API data and model configuration at their boundaries. Do not hide malformed records or partial failures.
- Keep secrets out of YAML and source files; configurations may name environment variables only.
- Preserve existing source and diagnostic artifacts; create additional diagnostic data only when debug is enabled. Avoid destructive migrations without an explicit migration and backup plan.
- Add focused tests for non-trivial parsing, target extraction, leakage filtering, and configuration-driven model selection.
- Use Python 3.12, the repository `.venv`, `uv`, and the committed `uv.lock`. Run project commands with `uv run`.
- Do not add a dependency until the implementation imports it.
- After each task, update the summary in `STATUS.md` and append a dated entry describing completed changes, findings, and remaining work. Keep only the latest 50 log entries in chronological order.
- Never run Git operations, including commit, push, pull, branch, merge, or reset, without explicit user approval.
