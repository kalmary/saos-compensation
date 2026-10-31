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
