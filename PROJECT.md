# SAOS Compensation

## Objective

Build a reproducible regression pipeline that estimates the compensation awarded in Polish court judgments available through SAOS. This is a study project for a data science course.

## Delivery plan

### 1. Acquire judgments and produce reviewable records

1. Read full judgments from the official SAOS `/api/dump/judgments` endpoint with a bounded judgment-date range. Resume dump pagination from checkpoint.json until the configured page limit or final page. The dump has no documented text/court filter, so select common-court compensation judgments locally. Join court/division metadata from `/api/dump/commonCourts`.
2. Use one shared HTTP session, an explicit timeout and one attempt per request. Report HTTP, pagination and validation failures; do not silently retry or discard malformed records. Refetch the pending page on resume, verify its IDs against the checkpoint and deduplicate by stable SAOS ID. Skip already committed records from the same dataset.
3. Treat `configs/dataset/api.yaml` as the defaults for acquisition: age range, page limit, total output size (max_gib), download path, debug and timeout. Expose matching CLI arguments; only explicitly supplied arguments override YAML. `--no-debug` must override a true YAML value.
4. Locate target sentences by embedding similarity to configs/dataset/target_pattern.yaml. The judgment class is mandatory; request is optional. Encode each pattern once, first retain only sentences containing a numeric PLN amount (including currency) and a configured candidate_keywords word stem, then compare candidates only with eligible classes: judgment/request stems restrict target eligibility, shared stems enable both only when no target-role stem occurs, and stems from both roles leave the choice to similarity. Additional prior_payment, procedural_costs and mixed keyword/pattern groups compete with those roles; an exclusion winner cannot supply either target role. Keep the best match per exclusion group in debug diagnostics. Require minimum_cosine_similarity (default 0.7), and retain the highest score per class. A sentence belongs only to its highest-scoring eligible class; ties between any classes abstain. Similarity is not a calibrated probability. Basal resolves principal components, beneficiary evidence and amounts only after a judgment match, using the operative ruling around that anchor when available. Multiple beneficiaries are supported: one case record stores the total across them, with per-person amounts multiplied only when their recipient group is established. Unknown counts and amounts remain missing. injured_party_count is a configured numeric feature. prior_payment is an explicitly allowed numeric predictor, resolved separately from the winning prior_payment sentence by source-grounded Basal extraction. It stores an integer PLN amount rounded half up to the nearest złoty, or 0 without a confirmed amount. Select an explicit prior-payment total when available; do not sum repeated mentions or ambiguous separate payments. This feature never enters award.json or the awarded total, and its passage stays excluded from text features. Final Basal amount confidence remains a separate configurable threshold. Normal records require a judgment match, a valid principal and safe features; debug retains failed attempts with valid=false. award.json stores selected components, total, optional request and valid; record.json contains only features. Match sentences, scores and pattern evidence are debug-only in model_calls.json.

5. With debug enabled, add only model_calls.json and source.txt in the same judgment folder. Do not save analysis, raw payloads, reports, config copies or extra debug subdirectories. Retain operational checkpoint.json and temporary pending state for interruption recovery; remove pending state after acceptance or rejection. Rejected cases leave no basic or debug files; all retained attempts failing admission are debug records marked valid=false. Commit the four case files before removing temporary pending state. Summaries remain console-only. Previously created artifacts are preserved rather than deleted automatically.
6. Keep the device in `configs/dataset/basal.yaml`: MPS for the current macOS experiment, with CPU/CUDA alternatives and device availability checked before downloads. Use the `pytorch-macos` dependency extra for local macOS work and `pytorch-cuda-linux` for NVIDIA GPU execution on Linux. Feature names are English while their model-facing descriptions and judgment text are Polish. basal.yaml selects the supported model name, while its revision and prompts remain pinned in code; feature/target selection and confidence thresholds are deliberate configuration choices.
7. Derive judgment year from SAOS metadata, require evidence for procedural instance, and derive the population class of the locality where the court is located from the matched GUGiK locality and its LICZBAMIESZKANCOW field. Count awarded beneficiaries for injured_party_count, distinguishing individual amounts from a joint amount. Preserve ambiguous/missing results. A current population snapshot does not establish population at judgment time.
8. Write `<download_path>/checkpoint.json` independently of debug. Pin the dataset ID, request dates, extraction configuration and GUGiK snapshot hash; store the next page/item and current page IDs. Commit the final record before advancing progress, and recognize an already saved record if interrupted between those writes. Resume automatically with the same command. Debug, timeout, max_gib and page limits may change; a different extraction configuration, age range or fresh dataset requires a new download path. Completed checkpoints perform no further acquisition. Persist completed model decisions, source and configuration in <SAOS_ID>/pending/ independently of debug so interruptions inside a judgment do not repeat completed inference. Retain pending state after interruption and remove it after a successful record commit or explicit rejection; its bytes count toward max_gib.
9. Enforce max_gib against all files already in the output directory, including prior executions and atomic temporary writes; zero disables the limit. Stop before a write exceeds the budget, leaving a resumable checkpoint. Keep model weights in the ignored basal_weights/ directory, with one Hugging Face models--<owner>--<model> subfolder per model. Missing weights download automatically when extraction starts; cached files are reused. Request only root-level weights, tokenizer, configuration and calibration files, with one file-download worker; Xet transport may separately parallelize transfer/reconstruction internally. Embedding model weights use the separate ignored embedding_weights/ folder with .gitkeep retained. Keep both model caches outside the size-limited output directory.
10. Keep normal output to record.json and award.json, adding only source.txt/model_calls.json when debug is enabled. Select the court factual-findings section using explicit headings, falling back to the reasoning body when no heading is available. Extract all selected text fields together in sentence-aligned batches of up to 6000 characters. Review only selected source candidates for factuality, currency exclusion, target leakage and relevance to each proposed field. Share source reviews across fields and try lower-ranked candidates when needed. Only missing fields receive a second extraction pass outside the primary section. Unsupported fields remain not_available. The extractor version is saos-extraction-15; older checkpoints require a separate output path. Keep PostgreSQL/pgvector ingestion separate. Model comparisons use the same saved judgments without changing the active model selection.


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

`src/dataset/processing/cosine_similarity.py` compares existing embedding vectors or matrices using PyTorch on explicit cpu, mps or cuda devices. It returns an N-by-M float32 similarity tensor on the selected device, validates finite nonzero real vectors and matching dimensions, and rejects unavailable accelerators. It does not generate embeddings or connect target phrase templates to acquisition.

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
    basal.yaml           # Basal device, selected features/targets and extraction thresholds
    embedding.yaml       # fast encoder, revision, device, batches and minimum cosine similarity
    target_pattern.yaml  # target roles and excluded payment-context phrases
  train.yaml             # training controls, when training is implemented
  models/
    <model-name>.yaml    # model choice and parameters, when regression is implemented
```

`configs/dataset/target_pattern.yaml` is active input to sentence selection. Keep one representative per distinct construction, without inflection/word-order/ending near-duplicates. Placeholder roles are rendered as Polish words and monetary values are neutralized only for embedding; raw text and numeric amounts retain exact source offsets. The pinned mean-pooled multilingual MiniLM encoder supports Polish, with explicit CPU/MPS/CUDA selection. Long sentences are embedded through all token windows rather than silently truncated. The matcher caches its model and pattern vectors across judgments. Candidate filtering is a cheap case-insensitive word-prefix/PLN check before embedding, with all required cues in the same sentence. No candidate means no embedding model loading; eligible sentences are encoded in batches. Fully written-out amounts remain unsupported. Similar-looking payment sentences can still have different legal roles; independent quality checks remain necessary.

Expose only values deliberately selected between experiments. Keep model prompts, fixed implementation details and credentials out of experiment controls. Future data access, vectorization and training configuration should be added when implemented, without duplicating values across files. Secrets may be referenced by environment-variable name only.

## Initial technical decisions

- Runtime: Python 3.12 managed with `uv`; sync groups/extras first, then execute with `uv run --no-sync`.
- Dataset entry point: src/dataset/get_data.py; supporting modules live under src/dataset/processing/.
- Current outputs: local per-judgment records and a resume checkpoint; diagnostic artifacts only in debug mode.
- Planned database: PostgreSQL with JSONB and pgvector.
- Configuration: YAML.
- Problem type: supervised regression.
- First model: a simple baseline chosen after the target distribution is inspected.
- Extraction prototype: pinned Basal model with PyTorch; MPS for local macOS work, CPU/CUDA when selected. Independent labeled evaluation is required before approving extracted training data.

## Main risks

- The judgment text may not identify a single unambiguous award.
- Search terminology may miss relevant judgments or include unrelated uses of compensation terms.
- Reasoning text can reveal the target directly or indirectly.
- Inflation and changes in judicial practice can create temporal drift.
- Related cases can leak across random row-level splits.
- Sparse regions or courts can make subgroup metrics unstable.
