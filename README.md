# SSAOS Compensation

Study project for building a regression model that estimates compensation amounts from Polish court judgments available through the SAOS API.

The planned pipeline downloads a narrowly selected set of judgments, validates and stores them in PostgreSQL with pgvector, extracts a leakage-safe feature schema, vectorizes configured fields, and trains interchangeable regression models from YAML configuration.

See [PROJECT.md](PROJECT.md) for the delivery plan and [AGENTS.md](AGENTS.md) for implementation constraints.
