"""Encode YAML-selected categories and normalize numeric predictors into stable columns.
Fit missing-value handling, category vocabularies and optional scalers on training data;
reuse saved state for new data and preserve prior_payment=0. Embeddings are handled separately.
"""
