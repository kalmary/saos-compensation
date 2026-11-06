"""Preprocessing lifecycle shared by training, evaluation and prediction."""


class Preprocessor:
    """Coordinate feature and target transformers and their persisted state."""

    def fit(self, records, targets, settings):
        """Fit preprocessing after related judgments have been split into groups.

        args:
            records: Training predictor mappings in sample order.
            targets: Aligned training principal amounts in PLN.
            settings: Validated preprocessing configuration.

        returns:
            Fitted preprocessor; held-out samples must never update its state.
        """
        raise NotImplementedError("Preprocessor fitting is not implemented")

    def transform(self, records, sample_ids, *, targets=None):
        """Transform aligned samples using fixed feature and optional target state.

        args:
            records: Allowlisted predictor mappings in sample order.
            sample_ids: Stable judgment IDs aligned with records.
            targets: Optional aligned PLN targets; absent during prediction.

        returns:
            Mapping containing X, optional y, sample_ids and field-to-column mapping.
        """
        raise NotImplementedError("Preprocessor transformation is not implemented")

    def inverse_transform(self, predictions):
        """Delegate prediction conversion to the fitted target transformer.

        args:
            predictions: Regression outputs in the configured target space.

        returns:
            Predicted amounts in PLN.
        """
        raise NotImplementedError("Preprocessor inverse transformation is not implemented")

    def save(self, path):
        """Persist fitted state with version, schema, encoder and effective settings.

        args:
            path: Destination for preprocessing artifacts, separate from source records.

        returns:
            None.
        """
        raise NotImplementedError("Preprocessor persistence is not implemented")

    @classmethod
    def load(cls, path):
        """Restore fitted state, rejecting incompatible versions and feature schemas.

        args:
            cls: Preprocessor class to restore.
            path: Saved preprocessing artifact location.

        returns:
            Restored preprocessor with identical feature order and transformations.
        """
        raise NotImplementedError("Preprocessor loading is not implemented")
