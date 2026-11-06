"""Separate local predictor and optional supervised-target input boundaries."""


def load_records(directory, fields, *, include_targets=False):
    """Load allowlisted predictors with stable IDs and optional valid award targets.

    args:
        directory: Root containing per-judgment record.json files.
        fields: Ordered predictor allowlist; excludes targets and evidence.
        include_targets: Read award.json and retain only valid=true samples when true.
            Prediction input must work without award.json when false.

    returns:
        Mapping with aligned sample_ids, records and optional targets in PLN.
        Missing values retain acquisition semantics, including prior_payment=0.
    """
    raise NotImplementedError("Preprocessing record loading is not implemented")
