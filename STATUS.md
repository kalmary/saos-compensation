# Project status

## Purpose

Maintain a concise record of completed work, findings, and outstanding tasks. After each task, update the summary and append a dated log entry. Keep the latest 50 entries in chronological order, removing the oldest when necessary.

## Summary

The project is at the data-feasibility stage. The exploratory script in `notebooks/01_saos_feasibility.py` checks scoped SAOS queries, pagination, full-text retrieval, and basic record validation. It uses `requests` and stores raw responses, acquisition metadata, and its report under `tmp/saos_feasibility/`.

All contents of `data/` are ignored except `data/.gitkeep`, including package markers and nested directories.

The initial snapshot contains 18 inspected judgments. All passed basic structural checks, but the review found mixed award components, unrelated search results, missing factual detail, and inconsistent dates. Data acquisition is feasible; building a reliable regression dataset remains conditional on a precise target definition, reviewed extraction, and leakage prevention. Search counts do not establish the number of usable training examples.

`PROJECT.md` includes evaluation of one fixed open-weight Polish model for feature extraction and standardization. Model inference, extraction accuracy, and regression performance have not been evaluated. Database ingestion, vectorization, and training are not implemented.

## Next steps

- Select one claim family and define the awarded principal, procedural stage, and treatment of prior payments, rejected claims, and multiple claimants.
- Review a larger sample across courts and years to measure usable-record coverage and refine acquisition filters.
- Label targets and factual evidence separately, then evaluate extraction and leakage controls.
- Compare a fixed Polish model against the labeled feature sample before selecting it.

## Log

- 2026-10-27: Added and ran the SAOS feasibility script. Verified scoped search, pagination, and full-text access; inspected 18 judgments and documented extraction risks in `tmp/saos_feasibility/report.md`.
- 2026-10-27: Documented a fixed Polish-model approach to text-feature extraction and standardization. Kept exploratory outputs under `tmp/` and retained package markers in ignored project folders.
- 2026-10-27: Preserved `data/.gitkeep` while keeping generated data ignored. Updated the status summary, next steps, and log instructions.
- 2026-10-27: Added the status-maintenance rule to `AGENTS.md`, including summary updates, dated entries, and a limit of 50 log entries.
- 2026-10-27: Restricted the `data/` exception to `.gitkeep`; all other contents remain ignored, including `__init__.py` files.
