"""Judgment metadata validation and source normalization."""

from datetime import date

from ..extraction.source.text import extract_text


def validate_judgment(payload, identifier, start, end):
    """
    Validate source metadata and normalize the judgment text.

    args:
        payload: Decoded source or model payload.
        identifier: Stable judgment identifier.
        start: Beginning of the permitted source range.
        end: End of the permitted source range.

    returns:
        Validated metadata and normalized text.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(data, dict)
        or type(data.get("id")) is not int
        or data["id"] != identifier
    ):
        raise ValueError("Judgment ID does not match the dump item")
    try:
        judgment_date = date.fromisoformat(data["judgmentDate"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Invalid judgment date") from error
    if not start <= judgment_date <= end:
        raise ValueError("Judgment date is outside the requested range")
    if data.get("courtType") != "COMMON" or data.get("judgmentType") != "SENTENCE":
        raise ValueError("Judgment does not match the requested court or judgment type")
    division = data.get("division")
    if not isinstance(division, dict):
        raise ValueError("Missing division metadata")
    court = division.get("court")
    for name, value in (("division", division), ("court", court)):
        if (
            not isinstance(value, dict)
            or type(value.get("id")) is not int
            or value["id"] <= 0
            or not isinstance(value.get("name"), str)
            or not value["name"].strip()
        ):
            raise ValueError(f"Invalid {name} metadata")
    if court.get("type") not in {"DISTRICT", "REGIONAL", "APPEAL"}:
        raise ValueError("Invalid common court level")
    cases = data.get("courtCases")
    if (
        not isinstance(cases, list)
        or not cases
        or any(
            not isinstance(case, dict)
            or not isinstance(case.get("caseNumber"), str)
            or not case["caseNumber"].strip()
            for case in cases
        )
    ):
        raise ValueError("Missing court case number")
    html = data.get("textContent")
    if not isinstance(html, str) or not (text := extract_text(html)):
        raise ValueError("Missing judgment text")
    return data, text
