# SAOS Compensation

## Objective

Build a reproducible regression pipeline that estimates the compensation awarded in Polish court judgments available through SAOS. This is a study project for a data science course.

## Delivery plan

### 1. Acquire judgments and produce reviewable records

1. Read full judgments from the official SAOS `/api/dump/judgments` endpoint with a bounded judgment-date range. Resume dump pagination from checkpoint.json until the configured page limit or final page. The dump has no documented text/court filter, so select common-court compensation judgments locally. Join court/division metadata from `/api/dump/commonCourts`.
2. Use one shared HTTP session, an explicit timeout and one attempt per request. Report HTTP, pagination and validation failures; do not silently retry or discard malformed records. Refetch the pending page on resume, verify its IDs against the checkpoint and deduplicate by stable SAOS ID. Skip already committed records from the same dataset.
3. Treat `configs/dataset/acquisition/api.yaml` as the defaults for acquisition: age range, page limit, total output size (max_gib), download path, debug and timeout. Expose matching CLI arguments; only explicitly supplied arguments override YAML. `--no-debug` must override a true YAML value.
4. Locate target sentences by embedding similarity to configs/dataset/acquisition/targets.yaml. The judgment class is mandatory; request is optional. Encode each pattern once, first retain only sentences containing a numeric PLN amount (including currency) and a configured candidate_keywords word stem, then compare candidates only with eligible classes: judgment/request stems restrict target eligibility, shared stems enable both only when no target-role stem occurs, and stems from both roles leave the choice to similarity. Additional prior_payment, procedural_costs, mixed and other_payments keyword/pattern groups compete with those roles; an exclusion winner cannot supply either target role. Keep the best match per exclusion group in debug diagnostics. Require a score strictly greater than minimum_cosine_similarity (default 0.7), and retain the highest score per class. A sentence belongs only to its highest-scoring eligible class; multiple above-threshold groups still produce only the highest-scoring winner. Exact ties prefer an exclusion class, then the first class in pattern order. Similarity is not a calibrated probability. After a judgment match, parse the numeric PLN value directly from that exact sentence using decimal arithmetic. Do not scan other ruling clauses or ask Basal to classify or verify target amounts. Require exactly one numeric PLN occurrence in the selected sentence; missing or multiple occurrences leave the target unresolved. Multiple beneficiaries are supported within that sentence: multiply an explicit per-person amount only when its recipient group can be counted from source text; preserve unknown counts and reject unresolved per-person totals. Separate awards in other sentences are not aggregated. injured_party_count is derived from explicit recipients in the selected sentence. prior_payment is an explicitly allowed numeric predictor, resolved separately from the winning prior_payment sentence by source-grounded Basal extraction. It stores an integer PLN amount rounded half up to the nearest złoty, or 0 without a confirmed amount. Select an explicit prior-payment total when available; do not sum repeated mentions or ambiguous separate payments. This feature never enters award.json or the awarded total, and its passage stays excluded from text features. amount_minimum_probability applies only to the prior_payment predictor, not the regression target. All records require an accepted source-backed damage description. cause and limitations are optional; there is no minimum feature count or coverage percentage. Missing damage discards the case even under debug. Among cases with accepted damage, normal records require a judgment match, a valid principal and safe features; debug retains other failed attempts with valid=false. award.json stores only total and valid; requested amounts, component evidence and monetary decisions remain internal/debug-only; record.json contains only features. Match sentences, scores and pattern evidence are debug-only in model_calls.json.

5. With debug enabled, add only model_calls.json and source.txt in the same judgment folder. Do not save analysis, raw payloads, reports, config copies or extra debug subdirectories. Retain operational checkpoint.json and temporary pending state for interruption recovery; remove pending state after acceptance or rejection. Rejected cases leave no basic or debug files; all retained attempts failing admission are debug records marked valid=false. Commit the four case files before removing temporary pending state. Summaries remain console-only. Previously created artifacts are preserved rather than deleted automatically.
6. Keep the device in `configs/dataset/acquisition/basal.yaml`: MPS for the current macOS experiment, with CPU/CUDA alternatives and device availability checked before downloads. Use the `pytorch-macos` dependency extra for local macOS work and `pytorch-cuda-linux` for NVIDIA GPU execution on Linux. Feature names are English while their model-facing descriptions and judgment text are Polish. basal.yaml selects the supported model, device and confidence thresholds. Model revisions and technical source/leakage instructions remain pinned in code; features.yaml is the single source of text questions and selected predictors. The same question is used for extraction and field-relevance review.
7. Derive judgment year from SAOS metadata, require evidence for procedural instance, and derive the population class of the locality where the court is located from the matched GUGiK locality and its LICZBAMIESZKANCOW field. Count awarded beneficiaries for injured_party_count, distinguishing individual amounts from a joint amount. Preserve ambiguous/missing results. A current population snapshot does not establish population at judgment time.
8. Write `<download_path>/checkpoint.json` independently of debug. Pin the dataset ID, request dates, extraction configuration and GUGiK snapshot hash; store the next page/item and current page IDs. Commit the final record before advancing progress, and recognize an already saved record if interrupted between those writes. Resume automatically with the same command. Debug, timeout, max_gib and page limits may change; a different extraction configuration, age range or fresh dataset requires a new download path. Completed checkpoints perform no further acquisition. Persist completed model decisions, source and configuration in <SAOS_ID>/pending/ independently of debug so interruptions inside a judgment do not repeat completed inference. Retain pending state after interruption and remove it after a successful record commit or explicit rejection; its bytes count toward max_gib.
9. Enforce max_gib against all files already in the output directory, including prior executions and atomic temporary writes; zero disables the limit. Stop before a write exceeds the budget, leaving a resumable checkpoint. Keep model weights in the ignored basal_weights/ directory, with one Hugging Face models--<owner>--<model> subfolder per model. Missing weights download automatically when extraction starts; cached files are reused. Request only root-level weights, tokenizer, configuration and calibration files, with one file-download worker; Xet transport may separately parallelize transfer/reconstruction internally. Embedding model weights use the separate ignored embedding_weights/ folder with .gitkeep retained. Keep both model caches outside the size-limited output directory.
10. Keep normal output to record.json and award.json, adding only source.txt/model_calls.json when debug is enabled. Select the court factual-findings section using explicit headings, falling back to the reasoning body when no heading is available. Extract all selected text fields together in sentence-aligned batches of up to 6000 characters. Use three simple text features defined in features.yaml: cause asks what caused the harm; damage asks what harm the injured party suffered; limitations asks which activities were difficult or impossible. Questions stay short and general, without examples or enumerated harm types. These describe the cause, the harm/loss itself and unavailable activities respectively, without assuming bodily injury. Missing source information remains missing; semantic distinction does not imply that one multi-fact source sentence can never support two fields. Attribute facts to the injured party. Review only selected source candidates for factuality, currency exclusion, target leakage and relevance to each proposed field. Console diagnostics distinguish no extracted description from currency, non-factual text, low factual confidence, leakage and poor field relevance, reporting existing scores and thresholds without additional model questions. Preserve target rejection reasons before required-feature rejection, and explicitly report when feature extraction was skipped. Report unresolved targets using their selected-sentence parsing status; do not report Basal amount probabilities or final verification for embedding-derived targets. Share source reviews across fields and try lower-ranked candidates when needed. Do not run a second extraction pass outside the primary section. Unsupported fields remain not_available. Require damage in features.yaml text_fields and required_fields and require its accepted source-backed description for every retained record, including debug records. Other features may remain missing; remove the previous majority-coverage rule. Leakage and relevance checks remain active for every populated text feature. The extractor version is saos-extraction-21; older checkpoints require a separate output path. Keep PostgreSQL/pgvector ingestion separate. Model comparisons use the same saved judgments without changing the active model selection.


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

`src/dataset/embeddings/cosine_similarity.py` compares existing embedding vectors or matrices using PyTorch on explicit cpu, mps or cuda devices. It returns an N-by-M float32 similarity tensor on the selected device, validates finite nonzero real vectors and matching dimensions, and rejects unavailable accelerators. It does not generate embeddings or connect target phrase templates to acquisition.

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
    acquisition/
      api.yaml           # acquisition defaults overridden only by explicit CLI arguments
      features.yaml      # metadata/text selection, questions and required damage
      targets.yaml       # embedding settings, keyword groups and phrase patterns
      basal.yaml         # Basal model, device and extraction/review thresholds
    model_input.yaml     # protected future regressor feature selection
  train.yaml             # training controls, when training is implemented
  models/
    <model-name>.yaml    # model choice and parameters, when regression is implemented
```

The sole public target is total; it is fixed by the acquisition contract and is not a Basal setting. The extractor retains the internal field suma as its output adapter; historical notebook experiments retain their component schema. The sole regression label is the principal amount in the embedding-selected judgment sentence, aggregated across explicitly identified recipients when it is stated per person. Other sentences are not scanned or summed. Exclusion patterns aim to filter procedural costs, interest, rent and prior payments; role correctness depends on embedding selection and needs independent evaluation. `configs/dataset/model_input.yaml` remains a protected future scaffold and is not the active extraction allowlist.

`configs/dataset/acquisition/targets.yaml` is active input to sentence selection. Keep one representative per distinct construction, without inflection/word-order/ending near-duplicates. Placeholder roles are rendered as Polish words and monetary values are neutralized only for embedding; raw text and numeric amounts retain exact source offsets. The pinned mean-pooled multilingual MiniLM encoder supports Polish, with explicit CPU/MPS/CUDA selection. Long sentences are embedded through all token windows rather than silently truncated. The matcher caches its model and pattern vectors across judgments. Candidate filtering is a cheap case-insensitive word-prefix/PLN check before embedding, with all required cues in the same sentence. No candidate means no embedding model loading; eligible sentences are encoded in batches. procedural_costs covers court, appeal, enforcement and legal-representation expenses; its keyword group uses procedural context rather than the generic word koszt so clinical expenses remain eligible as damages. other_payments covers interest, periodic rent and fines outside the selected one-off principal targets. Fully written-out amounts remain unsupported. Similar-looking payment sentences can still have different legal roles; independent quality checks remain necessary.

features.yaml contains metadata_fields, text_fields (field-to-question mapping), and required_fields: [damage]. Presence selects a field; there is no duplicate feature list or question mapping in basal.yaml or Python. Metadata names are validated against supported enrichments; text names must be English identifiers with nonempty questions. targets.yaml groups encoder settings and candidate_keywords under embedding and phrase groups under patterns. --config-dir selects a directory containing these four acquisition files and defaults to configs/dataset/acquisition. Reorganization preserves the normalized runtime configuration and version-21 checkpoint signature when values are unchanged. Default questions distinguish cause, harm and activities using general wording without examples or lists of harm types; this wording change requires a new output path for datasets using earlier questions. Question changes remain part of checkpoint compatibility and model decision cache identities. Confidence improvement has not been measured; source, relevance and leakage validation remain mandatory.

Expose only values deliberately selected between experiments. Keep technical source/leakage prompts, fixed implementation details and credentials out of experiment controls. Future data access, vectorization and training configuration should be added when implemented, without duplicating values across files. Secrets may be referenced by environment-variable name only.

## Initial technical decisions

- Runtime: Python 3.12 managed with `uv`; sync groups/extras first, then execute with `uv run --no-sync`.
- Dataset entry point: src/dataset/get_data.py handles CLI arguments only. src/dataset/acquisition/pipeline.py coordinates model lifecycle and extraction; acquisition/api.py validates SAOS responses and settings; acquisition/download.py handles resumable downloads; acquisition/checkpoints.py validates progress; storage/records.py enriches and exports records. Extraction and embedding code have separate packages. Within extraction/, basal_filtering.py coordinates the flow and features.py extracts and reviews predictors. The model/ package contains configuration, prompts and decision caching; source/ contains text normalization and passage batching; targets/ contains monetary evidence, principal resolution and prior-payment extraction; storage/files.py provides shared atomic writes and size limits. CLI flags, checkpoint compatibility and local record schemas are unchanged by this organization.
- Local experiments: notebooks/ contents are ignored; only .gitkeep is tracked. Notebook-specific tests skip when local inputs or their legacy configuration are absent.
- Current outputs: local per-judgment records and a resume checkpoint; diagnostic artifacts only in debug mode.
- Planned database: PostgreSQL with JSONB and pgvector.
- Configuration: YAML.
- Problem type: supervised regression.
- First model: a simple baseline chosen after the target distribution is inspected.
- Extraction prototype: pinned Basal model with PyTorch; MPS for local macOS work, CPU/CUDA when selected. Independent labeled evaluation is required before approving extracted training data.

## Main risks

- The judgment text may not identify a single unambiguous award.
- Direct targets inherit embedding role errors: a cached CPU replay of 257931 selected a 3,000 PLN costs sentence at cosine 0.700052917. Separate awards in unselected sentences are not included.
- Search terminology may miss relevant judgments or include unrelated uses of compensation terms.
- Reasoning text can reveal the target directly or indirectly.
- Inflation and changes in judicial practice can create temporal drift.
- Related cases can leak across random row-level splits.
- Sparse regions or courts can make subgroup metrics unstable.
