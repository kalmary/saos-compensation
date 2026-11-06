"""Configuration boundary for predictor allowlists and representation choices."""


def load_config(path):
    """Read and validate preprocessing YAML, rejecting target and evidence predictors.

    args:
        path: Explicit preprocessing configuration path.

    returns:
        Validated settings with ordered text, numeric and categorical fields,
        missing/unseen-value policies, encoder identity and target transformation.
    """
    raise NotImplementedError("Preprocessing configuration is not implemented")
