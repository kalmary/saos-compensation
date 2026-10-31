"""Cheap target-sentence filtering before model loading or embedding inference."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.processing import target_matching

KEYWORDS = {
    "judgment": ["zasądz"],
    "request": ["żąda"],
    "shared": ["przyzna", "odszkod"],
    "prior_payment": ["wypłac"],
    "procedural_costs": ["koszt"],
    "mixed": ["utrzyma"],
    "other_payments": ["odset", "rent", "grzywn"],
}


@pytest.mark.parametrize(
    ("text", "eligible"),
    [
        ("ZASĄDZA kwotę 20000 zł.", True),
        ("Zasądzono 200 złotych.", True),
        ("Powódka żądała 500 PLN.", True),
        ("Przyznano 100 (sto) zł.", True),
        ("Odszkodowanie wynosi 1,5 tys. zł.", True),
        ("Powód doznał obrażeń.", False),
        ("Zasądza świadczenie bez podania kwoty.", False),
        ("Zasądza świadczenie w złotych.", False),
        ("Zasądza 20000.", False),
        ("Zasądza 200 EUR.", False),
        ("Rachunek opiewa na 300 zł.", False),
        ("Niezasądzona kwota to 100 zł.", False),
        ("Zasądza świadczenie. Rachunek opiewa na 100 zł.", False),
    ],
)
def test_money_currency_and_core_word_must_cooccur_in_one_sentence(text, eligible):
    """
    Require all candidate cues in the same source sentence before embedding.

    args:
        text: Judgment excerpt.
        eligible: Whether a sentence meets all requirements.

    returns:
        None.
    """
    assert bool(target_matching.candidate_sentences(text, KEYWORDS)) is eligible


def test_candidate_filter_preserves_offsets_and_does_not_match_word_interiors():
    """
    Keep exact source positions after rejecting unrelated paragraphs and sentences.

    args:
        None.

    returns:
        None.
    """
    text = "Nagłówek.\nPowód doznał obrażeń. Zasądza 100 zł. Inny opis.\nPowód żąda 200 zł."
    rows = target_matching.candidate_sentences(text, KEYWORDS)
    assert [row["text"] for row in rows] == ["Zasądza 100 zł.", "Powód żąda 200 zł."]
    assert all(text[row["start"] : row["end"]] == row["text"] for row in rows)


def test_no_candidates_does_not_load_the_embedding_model(monkeypatch):
    """
    Return missing matches without loading weights for irrelevant input.

    args:
        monkeypatch: Scoped model-factory replacement.

    returns:
        None.
    """
    monkeypatch.setattr(
        target_matching,
        "cached_matcher",
        lambda *args: pytest.fail("Model must not load"),
    )
    config = {"embedding": {"candidate_keywords": KEYWORDS}}
    assert target_matching.match_targets(
        "Zasądza świadczenie. Inny rachunek: 100 zł.", config
    ) == dict.fromkeys(target_matching.PATTERN_CLASSES)


@pytest.mark.parametrize(
    ("text", "classes"),
    [
        ("Żąda zadośćuczynienia 100 zł.", ("request",)),
        ("Zasądza zadośćuczynienie 100 zł.", ("judgment",)),
        ("Żąda zasądzenia 100 zł.", ("judgment", "request")),
        ("Przyznano 100 zł.", ("judgment", "request")),
    ],
)
def test_role_specific_stems_take_precedence_over_shared_words(text, classes):
    """
    Restrict a known role even when the sentence also contains a shared benefit word.

    args:
        text: Candidate sentence with monetary evidence.
        classes: Eligible embedding classes.

    returns:
        None.
    """
    groups = {
        "judgment": ["zasądz"],
        "request": ["żąda"],
        "shared": ["przyzna", "zadośćuczyn"],
    }
    rows = target_matching.candidate_sentences(text, groups)
    assert rows[0]["classes"] == classes


@pytest.mark.parametrize(
    ("text", "group", "stem"),
    [
        ("Wypłacono już 100 zł.", "prior_payment", "wypłac"),
        ("Koszty procesu wyniosły 100 zł.", "procedural_costs", "koszt"),
        ("Naliczono 100 zł odsetek.", "other_payments", "odset"),
        ("Utrzymał się z żądaniem do 100 zł.", "mixed", "utrzyma"),
    ],
)
def test_exclusion_keywords_also_select_sentences_before_embedding(text, group, stem):
    """
    Admit excluded payment contexts without requiring a positive target keyword.

    args:
        text: Monetary sentence belonging to an exclusion group.
        group: Eligible exclusion class.
        stem: Word prefix selecting that class.

    returns:
        None.
    """
    groups = {key: [] for key in (*target_matching.PATTERN_CLASSES, "shared")}
    groups[group] = [stem]
    (row,) = target_matching.candidate_sentences(text, groups)
    assert row["classes"] == (group,)
    assert text[row["start"] : row["end"]] == text


def test_configured_cost_keywords_distinguish_treatment_from_procedure():
    """
    Keep clinical expenses eligible as damages instead of procedural costs.

    args:
        None.

    returns:
        None.
    """
    from src.dataset.processing.basal_filtering import load_config

    keywords = load_config()["embedding"]["candidate_keywords"]
    (treatment,) = target_matching.candidate_sentences(
        "Zasądza 5000 zł tytułem zwrotu kosztów leczenia.", keywords
    )
    (procedure,) = target_matching.candidate_sentences(
        "Zasądza 2400 zł tytułem zwrotu kosztów procesu.", keywords
    )
    assert "judgment" in treatment["classes"]
    assert "procedural_costs" not in treatment["classes"]
    assert "procedural_costs" in procedure["classes"]
