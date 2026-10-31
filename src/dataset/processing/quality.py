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
    reason = target_rejection_reason(result["award"], config)
    if reason is not None:
        return reason
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


def target_rejection_reason(award, config):
    """
    Check selected monetary targets before spending inference on features.

    args:
        award: Extracted award and verification decision.
        config: Selected targets and confidence threshold.

    returns:
        Rejection reason, or None for an admissible target.
    """
    threshold = config["system_one"]["amount_minimum_probability"]
    probability = award.get("verification", {}).get("noul")
    low_confidence = probability is not None and probability < threshold
    if low_confidence:
        return (
            f"low-confidence award (probability={probability}, required >= {threshold})"
        )
    if not award.get("confirmed") or probability is None:
        level = probability if probability is not None else "unavailable"
        return f"unconfirmed award (probability={level}, required >= {threshold})"
    established = 0
    for field in config["target_fields"]:
        try:
            value = Decimal(award.get(field, ""))
        except (InvalidOperation, TypeError, ValueError):
            if award.get(field) in (None, "brak informacji", "not_available"):
                continue
            return "invalid target"
        if not value.is_finite() or value < 0:
            return "invalid target"
        established += 1
    if not established:
        return "no established selected target"
    return None
