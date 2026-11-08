"""Target admission and multiple-recipient arithmetic without model downloads."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.dataset.extraction.model import configuration as config_module
from src.dataset.extraction.model import decisions
from src.dataset.extraction.targets import amounts
from src.dataset.extraction.targets import target_extraction as extraction


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
        extraction.extract_targets(None, "Żąda zapłaty.", config_module.load_config())
    )
    assert not result["confirmed"]
    assert result["target_matches"]["judgment"] is None


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

    monkeypatch.setattr(decisions, "ask", ask)
    result = asyncio.run(
        amounts.extract_prior_payment(
            None,
            {"text": "Wypłacono 700 zł, pozostaje 6300 zł."},
            config_module.load_config(),
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
        decisions, "ask", lambda *args: pytest.fail("Unexpected model call")
    )
    assert (
        asyncio.run(
            amounts.extract_prior_payment(None, None, config_module.load_config())
        )
        == 0
    )


@pytest.mark.parametrize(
    ("sentence", "expected", "count"),
    [
        ("Zasądza na rzecz A. B. kwotę 100 zł.", "100", 1),
        ("Zasądza na rzecz A. B. i C. D. kwotę 20000 zł.", "20000", 2),
        ("Zasądza na rzecz A. B. i C. D. kwoty po 20000 zł.", "40000", 2),
        (
            "Orzeka na rzecz A. B. nawiązkę w wysokości 10.000 (dziesięciu tysięcy) złotych;",
            "10000",
            1,
        ),
        ("Zasądza 1,5 tys. zł.", "1500", "not_available"),
        ("Zasądza 700,50 zł.", "700.50", "not_available"),
        ("Zasądza powodom po 100 zł.", None, "not_available"),
        ("Zasądza 150 zł, w tym 100 zł i 50 zł.", None, "not_available"),
        ("Zasądza 100 zł z odsetkami od 50 zł.", None, "not_available"),
        ("Zasądza kwotę stu złotych.", None, "not_available"),
    ],
)
def test_selected_sentence_is_the_only_target_source(
    monkeypatch, sentence, expected, count
):
    """Parse selected evidence without model decisions or other ruling amounts.

    args:
        monkeypatch: Scoped matcher and model replacements.
        sentence: Embedding-selected judgment sentence.
        expected: Total, or None for an ambiguous or missing amount.
        count: Known recipient count, or missing-value marker.

    returns:
        None.
    """
    text = "Zasądza 999 zł kosztów.\nUZASADNIENIE\n" + sentence + "\nŻąda 90000 zł."
    start = text.index(sentence)
    config = config_module.load_config()
    monkeypatch.setattr(
        extraction,
        "match_targets",
        lambda *args: {
            "judgment": {
                "text": sentence,
                "start": start,
                "end": start + len(sentence),
                "similarity": 0.9,
            },
            "request": {"text": "Żąda 90000 zł."},
        },
    )
    monkeypatch.setattr(
        decisions,
        "ask",
        lambda *args: pytest.fail("Target extraction must not call Basal"),
    )
    result = asyncio.run(extraction.extract_targets(None, text, config))
    assert result["confirmed"] is (expected is not None)
    assert result["suma"] == (expected if expected is not None else "brak informacji")
    assert result["injured_party_count"] == count
    if expected is not None:
        assert result["request"] == "90000"
        assert "verification" not in result
        assert "amount_decisions" not in result


def test_selected_target_must_match_source_offsets(monkeypatch):
    """Reject matcher evidence that does not match the original judgment.

    args:
        monkeypatch: Scoped matcher replacement.

    returns:
        None.
    """
    monkeypatch.setattr(
        extraction,
        "match_targets",
        lambda *args: {
            "judgment": {
                "text": "Zasądza 100 zł.",
                "start": 0,
                "end": 15,
                "similarity": 0.9,
            },
            "request": None,
        },
    )
    with pytest.raises(ValueError, match="does not match its source"):
        asyncio.run(
            extraction.extract_targets(
                None, "Different source.", config_module.load_config()
            )
        )
