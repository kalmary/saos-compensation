"""Reversible target transformation independent of predictor scaling."""


class TargetTransformer:
    """Identity or configured invertible transformation of PLN targets."""

    def fit(self, targets, settings):
        """Fit target transformation state using training amounts only.

        args:
            targets: Training principal amounts in PLN.
            settings: Validated target transformation settings.

        returns:
            Fitted target transformer.
        """
        raise NotImplementedError("Target fitting is not implemented")

    def transform(self, targets):
        """Convert PLN amounts into the configured training space.

        args:
            targets: Principal amounts in PLN, in sample order.

        returns:
            Finite transformed target vector using unchanged fitted state.
        """
        raise NotImplementedError("Target transformation is not implemented")

    def inverse_transform(self, predictions):
        """Return predictions to PLN before computing MAE or RMSE.

        args:
            predictions: Model outputs in transformed target space.

        returns:
            Predicted PLN amounts without refitting transformation statistics.
        """
        raise NotImplementedError("Inverse target transformation is not implemented")
