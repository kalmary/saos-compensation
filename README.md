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

Run acquisition, court metadata, and GUGiK locality matching:

```bash
uv run notebooks/01_saos_feasibility.py --metadata-only
```

The script samples approximately 300 judgments from the last ten years and saves its report to `tmp/saos_feasibility/report.md`. Raw responses, acquisition metadata, and detailed results are cached under `tmp/`. The GUGiK BDOT10k locality layer supplies population classes; its object version does not establish the population measurement year. Ambiguous locality matches remain unresolved.

Run the full prototype, including local Polish semantic extraction:

```bash
uv run --extra extraction notebooks/01_saos_feasibility.py
```

Run only the local model experiment:

```bash
uv run --extra extraction notebooks/03_system_one_polish.py
```

Dataset configuration lives in two files:

- [configs/dataset/basal.yaml](configs/dataset/basal.yaml): model revision, runtime settings, thresholds, input sizes, feature fields, instructions and decision categories.
- [configs/dataset/api.yaml](configs/dataset/api.yaml): SAOS endpoint, request filters, date range, search variants, sampling seed/court levels and development judgment IDs. A null end date means today; a null start date uses the configured number of years. Explicit dates use `YYYY-MM-DD`.

`additional_judgment_ids` adds manually selected examples outside sampling statistics. `judgment_ids` limits which documents Basal processes in the prototype; it is not a SAOS search filter.

[notebooks/preprocessing.yaml](notebooks/preprocessing.yaml) references both files and retains the prototype backend, Gemma fallback and GUGiK source. References resolve relative to that file, independently of the working directory. The pinned basal model runs locally on Apple Silicon; its first run downloads weights. Model inference covers the explicitly configured judgment IDs, not the whole acquisition sample. First, it classifies every paragraph with neighbouring context as facts, party statements, legal reasoning, operative ruling, mixed, ambiguous, or other. Mixed paragraphs are reconsidered sentence by sentence. Second, native basal `extract` fills six broad description fields from a JSON schema, using short batches of accepted factual sentences. Candidate sentences are checked separately for factual content and leakage before selecting one per field for the final record. The displayed descriptions retain the complete source sentence around the selected span, including negations. Section headings are not required. Awards are extracted separately. A numeric PLN match selects a whole source sentence or list item, with only the analyzed amount marked. Basal receives that sentence, up to `context_characters` preceding characters (currently 600), its section, and the operative ruling. One joint choice assigns the marked amount to awarded compensation, other awarded one-off damages, an unallocated principal, exclusion, or uncertainty. No wording pattern such as “tytułem …” is required. Repeated values are consolidated; each target can be retained independently. The ruling has priority when reasoning mentions conflicting values, and a final decision checks the assembled proposal. Uncertain decisions and rejected evidence are recorded separately from the field value `brak informacji`.

Read descriptions in `tmp/system_one/<ID>/record.json` and the separate award proposal in `award.json`. Unknown allocations retain an unallocated amount; missing components are not guessed or filled with zero. Conflicting component values and ambiguous or multiple beneficiaries require review. Process/court costs, interest and periodic rent are excluded. Review the separated source texts in `tmp/system_one/<ID>/facts.txt`, `reasoning.txt`, and `review.txt`; all decisions and offsets remain in `tmp/system_one/results.json`. The old heading-based selection is retained only as a comparison setting (`feature_source: factual_section`); the default is `semantic_paragraphs`.

The configured paragraph threshold is 0.6 and the maximum leakage probability is 0.2. Descriptions with low extraction confidence are retained and flagged for review. Joint assignments of monetary sentences and confirmation of the complete award proposal use a separate threshold, also 0.6; this does not approve them as regression labels. Reports start with actual field values; coverage and probabilities are auxiliary diagnostics. Outputs are candidates for manual review, not training-ready records.

The fallback [notebooks/04_gemma.py](notebooks/04_gemma.py) asks one question per category through the optional `ollama` package. Use it only after evaluating basal, with Ollama running and Gemma installed locally:

```bash
uv run --extra gemma notebooks/04_gemma.py
```

Set `gemma.model` when multiple Gemma models are installed. The script does not download models or use cloud inference. Set `backend: gemma` in the preprocessing configuration only when choosing that fallback for the full prototype.

Research requirements are recorded in [notebooks/wymagania.md](notebooks/wymagania.md). No database ingestion or regression training is performed by these prototypes.

