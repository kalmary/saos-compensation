"""Admission checks for model-derived dataset records."""

from decimal import Decimal, InvalidOperation


def rejection_reason(result, config):
    """
    Reject missing or weak targets and leakage in selected feature evidence.

    args:
        result: Extraction result with model decisions and evidence.
        config: Effective feature selection and probability thresholds.

    returns:
        Rejection reason, or None when the record meets admission rules.
    """
    award = result["award"]
    threshold = config["system_one"]["amount_minimum_probability"]
    if (
        not award.get("confirmed")
        or award.get("verification", {}).get("noul", 0) < threshold
    ):
        return "unconfirmed or low-confidence award"
    for field in config["target_fields"]:
        try:
            value = Decimal(award.get(field, ""))
        except (InvalidOperation, TypeError, ValueError):
            return "missing target"
        if not value.is_finite() or value < 0:
            return "invalid target"
    maximum = config["system_one"]["maximum_leak_probability"]
    for field in config["fields"]:
        quotes = result.get("evidence", {}).get(field, [])
        if (
            result.get("features", {}).get(field, "brak informacji")
            not in ("brak informacji", "not_available")
            and not quotes
        ):
            return "feature description has no source evidence"
        for quote in quotes:
            reviews = [
                row
                for row in result.get("description_reviews", [])
                if row["start"] <= quote["start"] and row["end"] >= quote["end"]
            ]
            if not reviews or any(
                row["answers"]["wyciek"]["noul"] > maximum for row in reviews
            ):
                return (
                    "missing leakage review or excessive leakage in selected features"
                )
    return None
