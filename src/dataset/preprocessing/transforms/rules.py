"""Fixed preprocessing rules, independent of model feature selection."""

TEXT_FIELDS = ("cause", "damage", "limitations")
FIELD_RULES = {
    "judgment_year": {"type": "numeric", "transform": "year_digits", "scaling": "none", "missing": "error"},
    "injured_party_count": {"type": "numeric", "transform": "identity", "scaling": "none", "missing": "error"},
    "prior_payment": {"type": "numeric", "transform": "log1p", "scaling": "robust", "missing": "error"},
    "instance": {"type": "categorical", "unknown": "ignore"},
    "court_locality_size": {"type": "categorical", "unknown": "ignore"},
}
TARGET_RULE = {"transform": "identity", "scaling": "robust"}
