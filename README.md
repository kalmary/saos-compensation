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

