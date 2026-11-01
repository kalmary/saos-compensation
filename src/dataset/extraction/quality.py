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
            minimum = config["system_one"]["minimum_probability"]
            if any(
                row["answers"].get(f"fit_{field}", {}).get("noul", 0) < minimum
                for row in reviews
            ):
                return (
                    "missing field-relevance review or irrelevant feature description"
                )

    return feature_rejection_reason(result, config)


def feature_rejection_reason(result, config):
    """
    Require source-backed values for strictly more than half the selected text features.

    args:
        result: Extracted feature values and their source evidence.
        config: Selected text feature fields; metadata is excluded.

    returns:
        Coverage rejection reason, or None when enough features were found.
    """
    fields = config["fields"]
    found = sum(
        isinstance(result.get("features", {}).get(field), str)
        and result["features"][field].strip()
        not in ("", "brak informacji", "not_available")
        and bool(result.get("evidence", {}).get(field))
        for field in fields
    )
    if fields and found * 2 <= len(fields):
        return f"insufficient text features ({found}/{len(fields)}; required > 50%)"
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
    if "target_matches" in award:
        match = award["target_matches"].get("judgment")
        threshold = config["embedding"]["minimum_cosine_similarity"]
        if match is None or match["similarity"] <= threshold:
            return f"no judgment sentence with cosine similarity > {threshold}"
    threshold = config["system_one"]["amount_minimum_probability"]
    probability = award.get("verification", {}).get("noul")
    low_confidence = probability is not None and probability < threshold
    if low_confidence:
        return (
            f"low-confidence award (probability={probability}, required >= {threshold})"
        )
    if probability is None and award.get("status"):
        reason = award["status"]
        decisions = award.get("amount_decisions", [])
        if decisions:
            best = max(decisions, key=lambda row: row["confidence"])
            reason += (
                f"; best amount classification={best['category']}"
                f" (probability={best['confidence']:.3f}, required >= {threshold})"
            )
        return reason + "; final award verification not run"
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
        if "target_matches" in award:
            try:
                total = Decimal(award.get("suma", ""))
            except (InvalidOperation, TypeError, ValueError):
                return "no established judgment amount"
            if total.is_finite() and total >= 0:
                return None
        return "no established selected target"
    return None
