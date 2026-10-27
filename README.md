# SSAOS Compensation

Study project for building a regression model that estimates compensation amounts from Polish court judgments available through the SAOS API.

The planned pipeline downloads a narrowly selected set of judgments, validates and stores them in PostgreSQL with pgvector, extracts a leakage-safe feature schema, vectorizes configured fields, and trains interchangeable regression models from YAML configuration.

## Installation

The project requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/kalmary/ssaos-compensation.git
cd ssaos-compensation
```

Install the base project dependencies:

```bash
uv sync
```

For development with the test group, which currently adds only pytest:

```bash
uv sync --group test
```

See [PROJECT.md](PROJECT.md) for the delivery plan and [AGENTS.md](AGENTS.md) for implementation constraints.
