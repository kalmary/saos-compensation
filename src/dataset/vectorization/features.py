"""Mixed predictor transformation with a stable block for each configured field."""


class FeatureTransformer:
    """Text embeddings and fitted numeric/categorical preprocessing interface."""

    def fit(self, records, settings):
        """Fit imputers, scalers and category vocabularies on training predictors only.

        args:
            records: Training predictor mappings after upstream grouped splitting.
            settings: Validated field order, representations and missing-value policies.

        returns:
            Fitted transformer with field-to-column mapping and encoder metadata.
        """
        raise NotImplementedError("Feature fitting is not implemented")

    def transform(self, records):
        """Apply fitted state and reuse create_embeddings for present text values.

        args:
            records: Predictor mappings in sample order, without required targets.

        returns:
            Finite numeric matrix with fixed column order and dimensions.
            Missing text placeholders must not be embedded as factual descriptions.
        """
        raise NotImplementedError("Feature transformation is not implemented")
