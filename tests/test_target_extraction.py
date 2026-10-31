"""Target admission and multiple-recipient arithmetic without model downloads."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.processing import basal_filtering as basal
from src.dataset.processing import target_extraction as extraction


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("A. B. i C. D.", 2),
        ("A. B., C. D. oraz E. F.", 3),
        ("dwóch poszkodowanych", 2),
        ("12 osób", 12),
        ("poszkodowanych", None),
        ("A. B. reprezentowanego przez C. D.", None),
    ],
)
def test_explicit_beneficiary_count(source, expected):
    """
    Count distinct named recipients without mistaking a representative for another victim.

    args:
        source: Extracted beneficiary span.
        expected: Known count or deliberate missing value.

    returns:
        None.
    """
    assert extraction.count_people(source) == expected


@pytest.mark.parametrize(
    ("per_person", "expected"), [(True, "40000"), (False, "20000")]
)
def test_joint_awards_are_aggregated_without_request(monkeypatch, per_person, expected):
    """
    Distinguish per-person amounts from one shared amount after embedding selection.

    args:
        monkeypatch: Scoped matcher and model replacements.
        per_person: Whether the ruling uses a per-recipient amount.
        expected: Total principal for the case.

    returns:
        None.
    """
    sentence = (
        "Zasądza na rzecz A. B. i C. D. "
        + ("kwoty po " if per_person else "kwotę ")
        + "20000 zł zadośćuczynienia."
    )
    config = basal.load_config()
    monkeypatch.setattr(
        extraction,
        "match_targets",
        lambda text, config: {
            "judgment": {
                "text": sentence,
                "start": 0,
                "end": len(sentence),
                "similarity": 0.9,
            },
            "request": None,
        },
    )
    events = []

    async def ask(server, state, questions, config):
        """
        Supply recipient evidence, principal category and aggregate verification.

        args:
            server: Unused server.
            state: Selected sentence or verification input.
            questions: Requested amount-resolution decision.
            config: Effective settings.

        returns:
            Controlled model answer.
        """
        events.extend(questions)
        if "beneficiaries" in questions:
            return {"beneficiaries": {"data": {"people": "A. B. i C. D."}}}
        if "swiadczenie" in questions:
            return {"swiadczenie": {"choice": "zadoscuczynienie", "confidence": 0.99}}
        assert state["łączna_kwota"] == expected
        assert state["liczba_poszkodowanych"] == 2
        return {"zasadzenie": {"noul": 0.99}}

    monkeypatch.setattr(basal, "ask", ask)
    award = asyncio.run(extraction.extract_targets(None, sentence, config))
    assert award["suma"] == expected
    assert award["injured_party_count"] == 2
    assert award["confirmed"]
    assert events == ["beneficiaries", "swiadczenie", "zasadzenie"]


def test_missing_judgment_never_reaches_basal(monkeypatch):
    """
    Reject an optional-request-only match before any Basal decision.

    args:
        monkeypatch: Scoped matcher replacement.

    returns:
        None.
    """
    monkeypatch.setattr(
        extraction,
        "match_targets",
        lambda text, config: {"judgment": None, "request": {"text": "Żąda zapłaty."}},
    )
    result = asyncio.run(
        extraction.extract_targets(None, "Żąda zapłaty.", basal.load_config())
    )
    assert not result["confirmed"]
    assert result["target_matches"]["judgment"] is None


def test_injured_party_count_is_a_numeric_feature():
    """
    Expose the resolved recipient count as a configured predictor feature.

    args:
        None.

    returns:
        None.
    """
    from src.dataset.get_data import add_metadata_features

    result = {
        "features": {},
        "evidence": {},
        "confidence": {},
        "field_status": {},
        "award": {"injured_party_count": 2},
        "instance": {"value": "pierwsza", "decision": {"confidence": 0.9}},
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd w Warszawie"}},
    }
    add_metadata_features(result, record, ["injured_party_count"], None)
    assert result["features"] == {"injured_party_count": 2}


@pytest.mark.parametrize("second_person", ["A. B.", "C. D."])
def test_distinct_award_clauses_are_summed_without_double_counting_people(
    monkeypatch, second_person
):
    """
    Sum separate operative obligations while deduplicating recipient identities.

    args:
        monkeypatch: Scoped matcher/model replacements.
        second_person: Repeated or distinct beneficiary of the second obligation.

    returns:
        None.
    """
    first = "Zasądza na rzecz A. B. kwotę 100 zł zadośćuczynienia."
    text = (
        first
        + f"\nZasądza na rzecz {second_person} kwotę 100 zł zadośćuczynienia.\nUZASADNIENIE\nOpis."
    )
    config = basal.load_config()
    monkeypatch.setattr(
        extraction,
        "match_targets",
        lambda text, config: {
            "judgment": {
                "text": first,
                "start": 0,
                "end": len(first),
                "similarity": 0.9,
            },
            "request": None,
        },
    )

    async def ask(server, state, questions, config):
        """
        Resolve two principal clauses and confirm their case total.

        args:
            server: Unused server.
            state: Source evidence or assembled proposal.
            questions: Requested resolver decision.
            config: Effective configuration.

        returns:
            Deterministic resolver answer.
        """
        if "beneficiaries" in questions:
            return {"beneficiaries": {"data": {"people": "A. B."}}}
        if "swiadczenie" in questions:
            return {"swiadczenie": {"choice": "zadoscuczynienie", "confidence": 0.99}}
        assert state["łączna_kwota"] == "200"
        assert state["liczba_poszkodowanych"] == (1 if second_person == "A. B." else 2)
        return {"zasadzenie": {"noul": 0.99}}

    monkeypatch.setattr(basal, "ask", ask)
    result = asyncio.run(extraction.extract_targets(None, text, config))
    assert result["suma"] == "200"
    assert result["confirmed"]


@pytest.mark.parametrize(
    ("value", "probability", "expected"),
    [
        ("700 zł", 0.9, 700),
        ("1,5 tys. zł", 0.9, 1500),
        ("700,49 zł", 0.9, 700),
        ("700,50 zł", 0.9, 701),
        ("700 zł", 0.5, 0),
        ("700 zł", None, 0),
        (None, 0.9, 0),
        ("700 EUR", 0.9, 0),
    ],
)
def test_prior_payment_resolves_only_selected_principal_as_integer(
    monkeypatch, value, probability, expected
):
    """
    Keep prior payments numeric and separate from the remaining amount to award.

    args:
        monkeypatch: Scoped model replacement.
        value: Extracted source amount or missing value.
        probability: Model confidence in the selected amount.
        expected: Whole-PLN feature value.

    returns:
        None.
    """

    async def ask(server, state, questions, config):
        """
        Return an earlier-payment span without selecting the outstanding balance.

        args:
            server: Unused inference server.
            state: Earlier-payment sentence.
            questions: Requested feature extraction.
            config: Effective settings.

        returns:
            Controlled source extraction.
        """
        assert set(questions) == {"prior_payment"}
        return {
            "prior_payment": {
                "data": {"amount": value},
                "fields": {"amount": {"confidence": probability}},
            }
        }

    monkeypatch.setattr(basal, "ask", ask)
    result = asyncio.run(
        basal.extract_prior_payment(
            None, {"text": "Wypłacono 700 zł, pozostaje 6300 zł."}, basal.load_config()
        )
    )
    assert type(result) is int
    assert result == expected


def test_missing_prior_payment_requires_no_model_call(monkeypatch):
    """
    Default a missing earlier-payment match to zero without inference.

    args:
        monkeypatch: Scoped model replacement.

    returns:
        None.
    """
    monkeypatch.setattr(
        basal, "ask", lambda *args: pytest.fail("Unexpected model call")
    )
    assert (
        asyncio.run(basal.extract_prior_payment(None, None, basal.load_config())) == 0
    )


@pytest.mark.parametrize("prior_payment", [0, 34400])
def test_prior_payment_is_exported_only_in_record(tmp_path, prior_payment):
    """
    Persist the integer feature without adding it to target fields or their total.

    args:
        tmp_path: Isolated output directory.
        prior_payment: Earlier principal feature including the absent-value default.

    returns:
        None.
    """
    import json

    from src.dataset.get_data import add_metadata_features, write_result

    config = basal.load_config(overrides={"debug": True})
    result = {
        "features": {},
        "evidence": {},
        "confidence": {},
        "field_status": {},
        "prior_payment": prior_payment,
        "award": {"confirmed": False, "zadoscuczynienie": "65600", "suma": "65600"},
        "instance": {"value": "pierwsza", "decision": {"confidence": 0.9}},
        "source_text": "Source.",
    }
    record = {
        "judgmentDate": "2020-01-01",
        "division": {"court": {"name": "Sąd w Warszawie"}},
    }
    add_metadata_features(result, record, ["prior_payment"])
    write_result(result, tmp_path, config)
    features = json.loads((tmp_path / "record.json").read_text())
    award = json.loads((tmp_path / "award.json").read_text())
    assert features == {"prior_payment": prior_payment}
    assert type(features["prior_payment"]) is int
    assert "prior_payment" not in award
    assert award["total"] == "65600"


def test_prior_payment_feature_is_resolved_for_debug_case_without_target(monkeypatch):
    """
    Retain a prior-payment feature even when debug preserves a case without an award.

    args:
        monkeypatch: Scoped target and feature resolvers.

    returns:
        None.
    """

    async def extract_award(*args):
        """
        Supply an excluded earlier payment without any eligible judgment.

        args:
            args: Unused extraction arguments.

        returns:
            Unconfirmed target and embedding diagnostics.
        """
        return {
            "confirmed": False,
            "target_matches": {
                "judgment": None,
                "prior_payment": {"text": "Wypłacono 700 zł."},
            },
        }

    async def prior_payment(server, match, config):
        """
        Resolve the selected feature without modifying the target result.

        args:
            server: Unused inference server.
            match: Earlier-payment source sentence.
            config: Effective pipeline configuration.

        returns:
            Integer feature value.
        """
        assert match["text"] == "Wypłacono 700 zł."
        return 700

    monkeypatch.setattr(basal, "extract_award", extract_award)
    monkeypatch.setattr(basal, "extract_prior_payment", prior_payment)
    result = asyncio.run(
        basal.extract(
            None, "Wypłacono 700 zł.", basal.load_config(overrides={"debug": True}), 42
        )
    )
    assert result["prior_payment"] == 700
    assert result["fields_evaluated"] is False
    assert "prior_payment" not in result["award"]
    assert result["award"]["confirmed"] is False
