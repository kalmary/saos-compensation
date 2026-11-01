"""Court metadata enrichment and local record export."""

import re
from datetime import date

from .extraction.quality import rejection_reason
from .storage import write_bytes, write_json


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
        values["injured_party_count"] = result["award"].get(
            "injured_party_count", "not_available"
        )
    if "prior_payment" in fields:
        values["prior_payment"] = result.get("prior_payment", 0)
    if "court_locality_size" in fields:
        from src.dataset.acquisition.localities import locality_class, match_locality

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
            "source": "embedding-selected ruling; Basal beneficiary extraction"
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
            else "not_available"
            if values[field] == "not_available"
            else "review required"
            if field == "instance"
            else "matched to the current GUGiK snapshot; population year unknown"
            if field == "court_locality_size"
            else "read from SAOS metadata"
        )


def write_result(result, directory, config):
    """
    Write feature and target values, plus only the requested debug files.

    args:
        result: Extracted feature and award values.
        directory: Judgment output directory.
        config: Selected fields and diagnostic flag.

    returns:
        None.
    """
    award = result["award"]
    valid = rejection_reason(result, config) is None
    if config["api"]["debug"] and not valid:
        award = award.get("proposal", award)
    features = {
        field: "not_available" if value == "brak informacji" else value
        for field, value in result["features"].items()
    }
    targets = {
        "total": "not_available"
        if award.get("suma", "brak informacji") == "brak informacji"
        else award["suma"],
    }
    targets["valid"] = valid
    if config["api"]["debug"]:
        write_json(directory / "model_calls.json", result.get("model_calls", []))
        write_bytes(directory / "source.txt", result["source_text"].encode("utf-8"))
    write_json(directory / "award.json", targets)
    write_json(directory / "record.json", features)
