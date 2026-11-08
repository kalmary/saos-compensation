"""Validated quantities with optional logarithmic transformation and scaling."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


def numeric_values(values, *, monetary=False):
    """Validate numeric values without silently coercing categorical codes.

    args:
        values: One-dimensional input values; None and not_available denote missing.
        monetary: Permit numeric strings and require nonnegative, present amounts.

    returns:
        Float64 column matrix, with NaN only for missing predictors.
    """
    result = []
    for value in values:
        if value is None or value == "not_available":
            if monetary:
                raise ValueError("Target amounts cannot be missing")
            result.append(np.nan)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            if not monetary or not isinstance(value, str):
                raise ValueError("Numeric fields require numbers")
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError("Invalid numeric value") from error
        if not np.isfinite(number) or (monetary and number < 0):
            raise ValueError("Numeric values must be finite; targets must be nonnegative")
        result.append(number)
    if not result:
        raise ValueError("Numeric input cannot be empty")
    return np.asarray(result, dtype=np.float64).reshape(-1, 1)


def transformed_values(values, method):
    """Apply a configured pointwise transformation before fitted scaling.

    args:
        values: Numeric column, possibly containing predictor NaNs.
        method: identity, log1p or year_digits.

    returns:
        Transformed values; year_digits produces two columns without fitted statistics.
    """
    if method == "identity":
        return values
    if method == "year_digits":
        if not np.isfinite(values).all() or np.any(values != np.floor(values)) or np.any(
            (values < 2000) | (values > 2099)
        ):
            raise ValueError("year_digits requires integer years from 2000 through 2099")
        return np.concatenate(((values % 100) // 10, values % 10), axis=1) / 10
    if method != "log1p" or np.any(values < 0):
        raise ValueError("log1p requires nonnegative values")
    return np.log1p(values)


def numeric_pipeline(settings):
    """Build scikit-learn preprocessing without min-max scaling.

    args:
        settings: Validated missing and scaling choices for one numeric field.

    returns:
        Unfitted scikit-learn imputation/scaling pipeline.
    """
    steps = []
    if settings["missing"] != "error":
        steps.append(("imputer", SimpleImputer(
            strategy="median" if settings["missing"] == "median" else "constant",
            fill_value=0, keep_empty_features=True,
        )))
    scaler = {"none": "passthrough", "standard": StandardScaler(), "robust": RobustScaler()}
    steps.append(("scale", scaler[settings["scaling"]]))
    return Pipeline(steps)
