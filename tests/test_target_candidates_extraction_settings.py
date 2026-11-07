"""Cheap target-sentence filtering before model loading or embedding inference."""


import sys

from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.dataset.embeddings import target_matching

def test_configured_cost_keywords_distinguish_treatment_from_procedure():
    """
    Keep clinical expenses eligible as damages instead of procedural costs.

    args:
        None.

    returns:
        None.
    """
    from src.dataset.extraction.model.configuration import load_config

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


@pytest.mark.parametrize(
    ("text", "group"),
    [
        ("Orzeka na rzecz A. B. nawiązkę w wysokości 100 zł.", "judgment"),
        ("Zasądza na rzecz każdego z powodów po 100 zł.", "judgment"),
        ("Powód ograniczył żądanie do kwoty 100 zł.", "request"),
        ("Poszkodowany otrzymał łącznie z kilku wypłat kwotę 100 zł.", "prior_payment"),
        ("Kwotę 100 zł sąd zasądził zgodnie z przedstawionym spisem kosztów.", "procedural_costs"),
        ("Odstępuje od obciążania powoda kosztami procesu w wysokości 100 zł.", "procedural_costs"),
        ("Oddala w całości żądanie zapłaty kwoty 100 zł.", "mixed"),
        ("Ustala wysokość jednej stawki dziennej grzywny na kwotę 100 zł.", "other_payments"),
        ("Zasądza solidarnie od pozwanych na rzecz powoda 100 zł tytułem zadośćuczynienia.", "judgment"),
        ("Powód dochodzi dopłaty 100 zł ponad kwotę wypłaconą przez ubezpieczyciela.", "request"),
        ("Sprawca zwrócił poszkodowanemu 100 zł przed wniesieniem pozwu i częściowo zaspokoił roszczenie.", "prior_payment"),
        ("Zasądza od pozwanego na rzecz świadka 100 zł tytułem zwrotu kosztów stawiennictwa w postępowaniu.", "procedural_costs"),
        ("Umarza postępowanie co do kwoty 100 zł wskutek cofnięcia powództwa.", "mixed"),
        ("Utrzymuje w mocy wyrok zasądzający kwotę 100 zł.", "mixed"),
        ("Zasądza rentę skapitalizowaną za poprzedni okres w kwocie 100 zł.", "other_payments"),
    ],
)
def test_additional_constructions_reach_their_configured_group(text, group):
    """Keep new structural patterns eligible before embedding comparison.

    args:
        text: Representative monetary sentence.
        group: Expected eligible target or exclusion group.

    returns:
        None.
    """
    from src.dataset.extraction.model.configuration import load_config

    keywords = load_config()["embedding"]["candidate_keywords"]
    (candidate,) = target_matching.candidate_sentences(text, keywords)
    assert group in candidate["classes"]
    assert candidate["text"] == text

