"""Prepare tabular predictors independently of target scaling and data splitting."""

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .features import FeatureTransformer
from .io.artifacts import load_state, save_state
from .io.configuration import validate_config


class FeaturePreprocessor(BaseEstimator):
    """Fit on training data and reuse fixed transforms for evaluation or prediction."""

    def fit(self, records, settings):
        """Fit predictor transformations after upstream grouped splitting.

        args:
            records: Training predictor mappings.
            settings: Ordered original model feature allowlist.

        returns:
            This fitted preprocessor.
        """
        config = validate_config(settings)
        features = FeatureTransformer().fit(records, config)
        self.features_ = features
        return self

    def transform(self, records, sample_ids):
        """Transform records without fitting or requiring awards for prediction.

        args:
            records: Predictor mappings in sample order.
            sample_ids: Aligned stable judgment IDs.

        returns:
            X, sample_ids, columns and field_columns mapping.
        """
        check_is_fitted(self, "features_")
        return self.features_.transform(records, sample_ids)

    def save(self, path=None):
        """Save fitted settings, transforms and column schema atomically.

        args:
            path: Trusted local destination; defaults to checkpoints/preprocessing/feature_preprocessor.pkl.

        returns:
            None.
        """
        check_is_fitted(self, "features_")
        save_state(path, self)

    @classmethod
    def load(cls, path=None, *, settings=None):
        """Restore trusted state, optionally requiring an exact ordered schema match.

        args:
            cls: FeaturePreprocessor class.
            path: Trusted local artifact path; defaults to checkpoints/preprocessing/feature_preprocessor.pkl.
            settings: Optional expected configuration, including field ordering.

        returns:
            Fitted preprocessor with unchanged transformations and schema.
        """
        state = load_state(path)
        if not isinstance(state, cls):
            raise ValueError("Artifact does not contain a preprocessor")
        check_is_fitted(state, "features_")
        if settings is not None:
            config = validate_config(settings)
            saved = state.features_.settings_
            if config != saved:
                raise ValueError("Incompatible preprocessing configuration or field order")
        return state
