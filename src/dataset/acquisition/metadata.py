"""Source-backed court, date and recipient feature enrichment."""

import re
from datetime import date


def add_metadata_features(result, record, fields, localities=None):
    """
    Add configured court and date features with their source provenance.

    args:
        result: Extraction result and supporting evidence.
        record: Normalized or saved judgment record.
        fields: Configured feature fields.
        localities: Official locality data and matching context.

    returns:
        None.
    """
    court = record["division"]["court"]
    locality = re.search(r"\b(?:w|we)\s+(.+)$", court["name"])
    values = {
        "judgment_year": date.fromisoformat(record["judgmentDate"]).year,
        "instance": "not_available"
        if result["instance"]["value"] == "brak informacji"
        else result["instance"]["value"],
        "court_seat": locality[1].strip() if locality else "not_available",
    }
    if "injured_party_count" in fields:
        count = result["award"].get("injured_party_count")
        values["injured_party_count"] = (
            1 if count in (None, "not_available", "brak informacji") else count
        )
    if "prior_payment" in fields:
        values["prior_payment"] = result.get("prior_payment", 0)
    if "court_locality_size" in fields:
        from .localities import locality_class, match_locality

        rows, morphology, metadata = localities
        matched, reason = match_locality(court["name"], rows, morphology)
        values["court_locality_size"] = (
            locality_class(matched["RODZAJ"], matched["LICZBAMIESZKANCOW"])
            if matched
            else "not_available"
        )
    sources = {
        "judgment_year": {"field": "judgmentDate", "value": record["judgmentDate"]},
        "instance": result["instance"],
        "court_seat": {"field": "division.court", "value": court},
    }
    if "court_locality_size" in fields:
        sources["court_locality_size"] = {
            "locality": matched,
            "matching": reason,
            "source": metadata,
        }
    if "injured_party_count" in fields:
        sources["injured_party_count"] = {
            "source": "defaulted to one; recipient count unavailable"
            if count in (None, "not_available", "brak informacji")
            else "explicit recipients in the embedding-selected judgment sentence"
        }
    if "prior_payment" in fields:
        sources["prior_payment"] = {
            "source": "embedding-selected prior payment; Basal amount extraction"
        }
    result["metadata_sources"] = {field: sources[field] for field in fields}
    for field in fields:
        result["features"][field] = values[field]
        result["evidence"][field] = []
        result["confidence"][field] = (
            result["instance"]["decision"]["confidence"]
            if field == "instance"
            else None
        )
        result["field_status"][field] = (
            "earlier principal payment in whole PLN; zero without confirmation"
            if field == "prior_payment"
            else sources[field]["source"]
            if field == "injured_party_count"
            else "not_available"
            if values[field] == "not_available"
            else "review required"
            if field == "instance"
            else "matched to the current GUGiK snapshot; population year unknown"
            if field == "court_locality_size"
            else "read from SAOS metadata"
        )
