"""Target transformations independent of predictor representation."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .io.configuration import validate_scaling
from ..storage import preprocessing as artifacts
from .transforms.rules import TARGET_RULE
from .transforms.numeric import numeric_pipeline, numeric_values, transformed_values


class TargetPreprocessor(BaseEstimator):
    """Fit an invertible target transformation using training amounts only."""

    def fit(self, targets, settings=None):
        """Fit scaling statistics in the configured target space.

        args:
            targets: Training PLN amounts, including numeric strings.
            settings: Internal policy override; defaults to fixed robust scaling in PLN.

        returns:
            This fitted target preprocessor.
        """
        settings = dict(TARGET_RULE) if settings is None else settings
        if not isinstance(settings, dict) or set(settings) != {"transform", "scaling"}:
            raise ValueError("Target settings require transform and scaling")
        validate_scaling(settings)
        values = transformed_values(numeric_values(targets, monetary=True), settings["transform"])
        scaler = numeric_pipeline({**settings, "missing": "error"})
        scaler.fit(values)
        if not np.isfinite(scaler.transform(values)).all():
            raise ValueError("Target fitting produced nonfinite values")
        self.settings_ = dict(settings)
        self.scaler_ = scaler
        return self

    def transform(self, targets):
        """Convert target amounts using unchanged training statistics.

        args:
            targets: Present nonnegative PLN amounts.

        returns:
            Finite one-dimensional target array.
        """
        check_is_fitted(self, "scaler_")
        values = transformed_values(numeric_values(targets, monetary=True), self.settings_["transform"])
        result = self.scaler_.transform(values).ravel()
        if not np.isfinite(result).all():
            raise ValueError("Target transformation produced nonfinite values")
        return result

    def inverse_transform(self, predictions):
        """Return predictions to PLN without clipping or fitting new statistics.

        args:
            predictions: Finite predictions in transformed target space.

        returns:
            One-dimensional PLN array; rejects numerical overflow.
        """
        check_is_fitted(self, "scaler_")
        values = numeric_values(predictions)
        if not np.isfinite(values).all():
            raise ValueError("Predictions must be present and finite")
        values = self.scaler_.inverse_transform(values)
        if self.settings_["transform"] == "log1p":
            with np.errstate(over="ignore"):
                values = np.expm1(values)
        if not np.isfinite(values).all():
            raise ValueError("Inverse target transformation produced nonfinite amounts")
        return values.ravel()

    def save(self, path=None):
        """Save fitted target scaling separately from predictor preprocessing.

        args:
            path: Trusted artifact destination; defaults to checkpoints/preprocessing/target_preprocessor.pkl.

        returns:
            None.
        """
        check_is_fitted(self, "scaler_")
        artifacts.save_state(artifacts.DEFAULT_TARGET_PATH if path is None else path, self)

    @classmethod
    def load(cls, path=None):
        """Restore target parameters without reading records or fitting new amounts.

        args:
            cls: Target preprocessor class.
            path: Trusted artifact source; defaults to checkpoints/preprocessing/target_preprocessor.pkl.

        returns:
            Fitted target preprocessor for scaling labels or inverse-transforming predictions.
        """
        state = artifacts.load_state(artifacts.DEFAULT_TARGET_PATH if path is None else path)
        if not isinstance(state, cls):
            raise ValueError("Artifact does not contain a target transformer")
        check_is_fitted(state, "scaler_")
        return state
