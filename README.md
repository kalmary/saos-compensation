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

## Data feasibility

Run the exploratory SAOS check:

```bash
uv run notebooks/01_saos_feasibility.py
```

The script prints its findings and saves a readable report to `tmp/saos_feasibility/report.md`. Raw responses and acquisition metadata are cached in the same directory. Reruns reuse that snapshot; move the directory aside to fetch a fresh one. No database or model inference is required.

