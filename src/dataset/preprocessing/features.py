"""Model-ready numeric and categorical predictors, separate from text embeddings."""

from copy import deepcopy

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .io.configuration import validate_config
from .transforms.rules import FIELD_RULES
from .transforms.numeric import numeric_pipeline, numeric_values, transformed_values


class FeatureTransformer(BaseEstimator):
    """Keep fitted field transformations, column meanings and ordering stable."""

    def fit(self, records, settings):
        """Fit each selected predictor on an upstream training split only.

        args:
            records: Nonempty sequence of predictor mappings.
            settings: Ordered original feature allowlist; text is encoded separately.

        returns:
            This fitted transformer.
        """
        config = validate_config(settings)
        rules = {field: deepcopy(FIELD_RULES[field]) for field in config["feature_fields"] if field in FIELD_RULES}
        if not rules:
            raise ValueError("At least one non-embedding predictor is required")
        transformers = {}
        columns = []
        mapping = {}
        for field, options in rules.items():
            values = field_values(records, field, options)
            start = len(columns)
            if options["type"] == "numeric":
                if options["missing"] == "median" and np.isnan(values).all():
                    raise ValueError(f"Cannot fit a median without observations: {field}")
                transformer = numeric_pipeline(options).fit(values)
                if options["transform"] == "year_digits":
                    columns.extend((f"{field}_tens", f"{field}_units"))
                else:
                    columns.append(field)
            else:
                transformer = OneHotEncoder(sparse_output=False, handle_unknown=options["unknown"])
                # Reserve a missing category even when the training field is complete.
                categories = sorted(set(values.ravel()) | {"not_available"})
                transformer.set_params(categories=[categories]).fit(values)
                columns.extend(f"{field}={category}" for category in transformer.categories_[0])
            if not np.isfinite(transformer.transform(values)).all():
                raise ValueError(f"Nonfinite fitted predictor values: {field}")
            transformers[field] = transformer
            mapping[field] = [start, len(columns)]
        self.settings_ = config
        self.field_rules_ = rules
        self.transformers_ = transformers
        self.columns_ = columns
        self.field_columns_ = mapping
        return self

    def transform(self, records, sample_ids):
        """Produce finite predictor blocks without accessing targets or refitting.

        args:
            records: Predictor mappings in sample order.
            sample_ids: Unique stable nonempty string IDs aligned with records.

        returns:
            Mapping containing X, sample_ids, columns and half-open field_columns.
        """
        check_is_fitted(self, "transformers_")
        if len(records) != len(sample_ids) or not sample_ids or any(
            not isinstance(value, str) or not value for value in sample_ids
        ) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Unique string sample IDs must align with nonempty records")
        blocks = []
        for field, options in self.field_rules_.items():
            values = field_values(records, field, options)
            blocks.append(self.transformers_[field].transform(values))
        matrix = np.concatenate(blocks, axis=1)
        if not np.isfinite(matrix).all():
            raise ValueError("Preprocessing produced nonfinite feature values")
        return {"X": matrix, "sample_ids": list(sample_ids), "columns": list(self.columns_),
                "field_columns": deepcopy(self.field_columns_)}


def field_values(records, field, settings):
    """Validate one selected predictor without reading unrelated input fields.

    args:
        records: Nonempty predictor mappings; absent keys are schema errors.
        field: Selected predictor name.
        settings: Field type, transformation and missing-value policies.

    returns:
        Validated column for a numeric pipeline or categorical encoder.
    """
    if not records or any(not isinstance(row, dict) or field not in row for row in records):
        raise ValueError(f"Records require selected field: {field}")
    values = [row[field] for row in records]
    if settings["type"] == "numeric":
        array = transformed_values(numeric_values(values), settings["transform"])
        if settings["missing"] == "error" and np.isnan(array).any():
            raise ValueError(f"Missing numeric predictor: {field}")
        return array
    if any(value is not None and (not isinstance(value, str) or not value.strip()) for value in values):
        raise ValueError(f"Categorical predictors require nonblank strings: {field}")
    return np.asarray(["not_available" if value is None else value for value in values], dtype=object).reshape(-1, 1)
