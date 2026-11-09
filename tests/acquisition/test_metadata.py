
import pytest

from dataset.acquisition.metadata import add_metadata_features


@pytest.mark.parametrize(
    "court_name, expected",
    [
        ("Sąd Okręgowy w Płocku", "Płocku"),
        ("Sąd Rejonowy dla Warszawy-Mokotowa w Warszawie", "Warszawie"),
        ("Sąd Rejonowy we Wrocławiu", "Wrocławiu"),
        ("Sąd Rejonowy", "not_available"),
    ],
)
def test_metadata_features_use_judgment_date_instance_and_court_seat(
    court_name,
    expected,
):
    """
    Verify metadata features use judgment date instance and court seat.

    args:
        court_name: Source court name containing its location.
        expected: Expected result.

    returns:
        None.
    """
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "druga", "decision": {"confidence": 0.8}},
    }
    record = {
        "judgmentDate": "2019-12-31",
        "division": {"court": {"name": court_name, "type": "DISTRICT"}},
    }
    add_metadata_features(
        result, record, ["judgment_year", "instance", "court_seat"]
    )
    assert result["features"] == {
        "judgment_year": 2019,
        "instance": "druga",
        "court_seat": expected,
    }
    assert result["metadata_sources"]["judgment_year"]["value"] == "2019-12-31"
    assert result["confidence"]["instance"] == 0.8


def test_metadata_selection_and_unknown_instance_do_not_infer_from_court_level():
    """
    Verify metadata selection and unknown instance do not infer from court level.

    args:
        None

    returns:
        None.
    """
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "not_available", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Apelacyjny w Krakowie", "type": "APPEAL"}},
    }
    add_metadata_features(result, record, ["instance"])
    assert result["features"] == {"instance": "not_available"}
    assert result["field_status"] == {"instance": "not_available"}


def test_population_feature_uses_official_match_and_preserves_provenance():
    """
    Verify population feature uses official match and preserves provenance.

    args:
        None

    returns:
        None.
    """
    import morfeusz2

    row = {
        "NAZWA": "Warszawa",
        "RODZAJ": "miasto",
        "LICZBAMIESZKANCOW": 1800000,
        "IDENTYFIKATORSIMC": "0918123",
    }
    result = {
        "features": {},
        "field_status": {},
        "evidence": {},
        "confidence": {},
        "instance": {"value": "not_available", "decision": {"confidence": 0.4}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd Okręgowy w Warszawie"}},
    }
    add_metadata_features(
        result,
        record,
        ["court_locality_size"],
        ([row], morfeusz2.Morfeusz(), {"population_year": None}),
    )
    assert result["features"] == {"court_locality_size": "city_at_least_500000"}
    assert result["metadata_sources"]["court_locality_size"]["locality"] == row
