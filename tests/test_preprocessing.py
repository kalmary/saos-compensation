import asyncio
import importlib
import sys
from datetime import date
from pathlib import Path

import morfeusz2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
feasibility = importlib.import_module("01_saos_feasibility")
extraction = importlib.import_module("03_system_one_polish")
gemma = importlib.import_module("04_gemma")


def test_notebook_config_is_independent_of_working_directory(monkeypatch, tmp_path):
    """
    Verify notebook config is independent of working directory.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    expected = extraction.load_config()
    monkeypatch.chdir(tmp_path)
    config = extraction.load_config()
    assert config == expected


@pytest.mark.parametrize(
    ("population", "expected"),
    [
        (19999, "miasto_ponizej_20_tys"),
        (20000, "miasto_20_do_ponizej_100_tys"),
        (100000, "miasto_100_do_ponizej_200_tys"),
        (200000, "miasto_200_do_ponizej_500_tys"),
        (500000, "miasto_co_najmniej_500_tys"),
        (None, "brak informacji"),
        (float("nan"), "brak informacji"),
        (-1, "brak informacji"),
    ],
)
def test_locality_population_boundaries(population, expected):
    """
    Verify locality population boundaries.

    args:
        population: Recorded locality population.
        expected: Expected result.

    returns:
        None.
    """
    assert feasibility.locality_class("miasto", population) == expected
    assert feasibility.locality_class("wieś", population) == "wies"
    assert feasibility.locality_class("część miasta", population) == "brak informacji"


def test_locality_inflection_and_ambiguous_names():
    """
    Verify locality inflection and ambiguous names.

    args:
        None

    returns:
        None.
    """
    morphology = morfeusz2.Morfeusz()
    row = {"NAZWA": "Pisz", "IDENTYFIKATORSIMC": "0977835"}
    assert (
        feasibility.match_locality("Sąd Rejonowy w Piszu", [row], morphology)[0] == row
    )
    second = {**row, "IDENTYFIKATORSIMC": "0000001"}
    assert (
        feasibility.match_locality("Sąd Rejonowy w Piszu", [row, second], morphology)[0]
        is None
    )
    assert feasibility.match_locality("Sąd Rejonowy", [row], morphology)[0] is None


def test_fragment_offsets_preserve_all_non_whitespace_characters():
    """
    Verify fragment offsets preserve all non whitespace characters.

    args:
        None

    returns:
        None.
    """
    text = "Pierwszy akapit.\n\n" + "Długi opis krzywdy i jej następstw. " * 40
    fragments = list(extraction.fragments(text, 100))
    assert all(text[part["start"] : part["end"]] == part["text"] for part in fragments)
    assert all(len(part["text"]) <= 100 for part in fragments)
    assert "".join("".join(part["text"].split()) for part in fragments) == "".join(
        text.split()
    )


def test_html_source_line_breaks_do_not_split_paragraphs():
    """
    Verify html source line breaks do not split paragraphs.

    args:
        None

    returns:
        None.
    """
    html = "<p>Powódka\n <span>J. P.</span>\n złamała nogę.</p>\n<p>Sąd rozważył\n podstawę prawną.</p>"
    assert feasibility.extract_text(html).splitlines() == [
        "Powódka J. P. złamała nogę.",
        "Sąd rozważył podstawę prawną.",
    ]


def test_only_factual_section_is_used_for_features():
    """
    Verify only factual section is used for features.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nZasądza 15000 zł.\nUZASADNIENIE\nPowód żądał zapłaty.\nSąd ustalił:\nPowódka złamała nogę.\nSąd zważył:\nKwota jest odpowiednia."
    result = extraction.feature_input(text, {"feature_source": "factual_section"})
    assert result["text"] == "Powódka złamała nogę."
    assert text[result["start"] : result["end"]] == result["text"]
    assert "15000 zł" in text
    assert (
        extraction.feature_input(
            "UZASADNIENIE\nSam opis bez granic.", {"feature_source": "factual_section"}
        )["text"]
        == ""
    )


def test_semantic_input_does_not_require_section_headings():
    """
    Verify semantic input does not require section headings.

    args:
        None

    returns:
        None.
    """
    text = "Powódka złamała nogę.\nSąd rozważył podstawę prawną."
    selected = extraction.feature_input(text, {"feature_source": "semantic_paragraphs"})
    assert selected["text"] == text
    assert selected["start"] == 0


def test_mixed_paragraph_sentences_keep_source_offsets_and_initials():
    """
    Verify mixed paragraph sentences keep source offsets and initials.

    args:
        None

    returns:
        None.
    """
    text = "Powódka J. P. złamała nogę. Zasądzono 15.000,50 zł."
    chunk = {"text": text, "start": 100, "end": 100 + len(text)}
    sentences = list(extraction.sentence_fragments(chunk))
    assert [row["text"] for row in sentences] == [
        "Powódka J. P. złamała nogę.",
        "Zasądzono 15.000,50 zł.",
    ]
    assert all(
        text[row["start"] - 100 : row["end"] - 100] == row["text"] for row in sentences
    )
    dated = "W dniu 24 czerwca 2023 r. A. B. opublikował wpis. Powód zareagował."
    assert [
        row["text"]
        for row in extraction.sentence_fragments({"text": dated, "start": 0})
    ] == ["W dniu 24 czerwca 2023 r. A. B. opublikował wpis.", "Powód zareagował."]


def test_paragraph_roles_use_context_without_requiring_headings(monkeypatch, tmp_path):
    """
    Verify paragraph roles use context without requiring headings.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    text = "Powódka złamała nogę.\nSąd ustalił, że wymagała pomocy. Zasądzono 10000 zł.\nSąd rozważył podstawę prawną."
    seen = []

    async def answer(server, state, questions, config):
        """
        Provide the answer behavior used by this operation.

        args:
            server: Local inference server.
            state: Input state supplied to the model.
            questions: Model question definitions.
            config: Effective pipeline or acquisition settings.

        returns:
            Result of this operation.
        """
        seen.append(state)
        fragment = state["oceniany_fragment"]
        category = "fakty"
        if "Sąd ustalił" in fragment and "Zasądzono" in fragment:
            category = "mieszany"
        elif fragment.startswith("Zasądzono"):
            category = "rozstrzygniecie"
        elif fragment.startswith("Sąd rozważył"):
            category = "rozwazania"
        return {
            "sekcja": {"choice": category, "confidence": 0.9},
            "wyciek": {"noul": 0.9 if "Zasądzono" in fragment else 0.05},
        }

    monkeypatch.setattr(extraction, "ask", answer)
    config = {
        **extraction.load_config(),
        "feature_source": "semantic_paragraphs",
        "chunk_characters": 1200,
        "context_characters": 600,
        "system_one": {"minimum_probability": 0.3},
    }
    _, rows = asyncio.run(extraction.classify_paragraphs(None, text, config, 1))
    extraction.select_facts(rows, 0.3, 0.15)
    assert [row["text"] for row in rows if row["accepted"]] == [
        "Powódka złamała nogę.",
        "Sąd ustalił, że wymagała pomocy.",
    ]
    assert any(row["unit"] == "sentence" for row in rows)
    assert "Powódka złamała nogę." in seen[1]["poprzedzajacy_kontekst"]
    assert "Sąd rozważył podstawę prawną." in seen[1]["nastepujacy_kontekst"]
    extraction.write_paragraphs(
        {"diagnostics": rows, "features": {"opis": "brak informacji"}},
        tmp_path,
        0.3,
        debug=True,
    )
    facts = (tmp_path / "facts.txt").read_text()
    assert "Powódka złamała nogę." in facts
    assert "Zasądzono" not in facts
    assert "Sąd rozważył" not in facts
    assert "Sąd rozważył" in (tmp_path / "reasoning.txt").read_text()
    assert "Zasądzono" in (tmp_path / "review.txt").read_text()


def test_generated_quotes_must_exist_at_unambiguous_source_offsets():
    """
    Verify generated quotes must exist at unambiguous source offsets.

    args:
        None

    returns:
        None.
    """
    text = "Powódka nie wymagała rehabilitacji ani pomocy innych osób."
    accepted, rejected = gemma.validate_quotes(
        {"cytaty": [text, "Powódka wymagała stałej opieki innych osób."]}, text, 100
    )
    assert accepted == [{"text": text, "start": 100, "end": 100 + len(text)}]
    assert len(rejected) == 1
    assert gemma.validate_quotes({"cytaty": [text]}, text + text)[0] == []


def test_lower_field_threshold_does_not_relax_leakage_filter():
    """
    Verify lower field threshold does not relax leakage filter.

    args:
        None

    returns:
        None.
    """
    row = {
        "text": "Powódka wymagała pomocy przy ubieraniu.",
        "start": 0,
        "end": 37,
        "answers": {
            "sekcja": {"choice": "fakty", "confidence": 0.7},
            "wyciek": {"noul": 0.2},
            "wplyw_na_zycie": {"noul": 0.7},
        },
    }
    assert extraction.select_facts([row], 0.6, 0.15) == []
    row["answers"]["wyciek"]["noul"] = 0.05
    assert len(extraction.select_facts([row], 0.6, 0.15)) == 1
    assert extraction.select_facts([row], 0.85, 0.15) == []
    row["text"] = "Opis zawiera 100 EUR bez wskazania, czy sąd je zasądził."
    assert extraction.select_facts([row], 0.6, 0.15) == []
    assert not feasibility.CURRENCY.search("Powódka doznała złamania nogi.")


@pytest.mark.parametrize(
    "response",
    [
        {"answers": {}},
        {"truncated": True, "answers": {}},
        {"answers": {"q": {"choice": "inna", "probabilities": {"a": 0.8, "b": 0.2}}}},
        {"answers": {"q": {"choice": "a", "probabilities": {"a": 0.9, "b": 0.9}}}},
        {
            "answers": {
                "q": {"choice": "a", "probabilities": {"a": float("nan"), "b": 0.2}}
            }
        },
    ],
)
def test_invalid_model_responses_are_not_cached_as_results(response):
    """
    Verify invalid model responses are not cached as results.

    args:
        response: Raw model response.

    returns:
        None.
    """
    with pytest.raises(ValueError):
        extraction.check_decision(
            response, {"q": extraction.choices("Pytanie", {"a": "A", "b": "B"})}
        )


def test_native_extraction_preserves_negative_sentence_and_low_confidence():
    """
    Verify native extraction preserves negative sentence and low confidence.

    args:
        None

    returns:
        None.
    """
    text = "Nagłówek. Powódka nie wymagała rehabilitacji."
    start = text.index("Powódka")
    source, mapping = extraction.extraction_source(
        text, [{"text": text[start:], "start": start, "end": len(text)}]
    )
    span = "rehabilitacji"
    position = source.index(span)
    answer = {
        "data": {"pomoc": span, "wiek": None},
        "fields": {
            "pomoc": {
                "start": position,
                "end": position + len(span),
                "confidence": 0.12,
            },
            "wiek": {"value": None},
        },
    }
    question = {
        "opis": {"type": "extract", "schema": {"properties": {"pomoc": {}, "wiek": {}}}}
    }
    extraction.check_decision({"answers": {"opis": answer}}, question, source)
    values, evidence, statuses, _ = extraction.describe_record(
        answer, mapping, ["pomoc", "wiek"], 0.3
    )
    assert values == {"pomoc": text[start:], "wiek": "brak informacji"}
    assert "niskiej pewności" in statuses["pomoc"]
    row = evidence["pomoc"][0]
    assert text[row["start"] : row["end"]] == row["text"]
    answer["data"]["pomoc"] = "wymagała rehabilitacji"
    with pytest.raises(ValueError, match="źródłowemu"):
        extraction.check_decision({"answers": {"opis": answer}}, question, source)


def test_extraction_batches_keep_source_offsets_and_all_sentences():
    """
    Verify extraction batches keep source offsets and all sentences.

    args:
        None

    returns:
        None.
    """
    text = "Powódka nie wymagała opieki. Wróciła do pracy. Nadal odczuwała ból."
    passage = {"text": text, "start": 0, "end": len(text)}
    batches = list(extraction.extraction_batches(text, [passage, passage], 50))
    assert len(batches) == 2
    assert sum(len(mapping) for _, mapping in batches) == 3
    for source, mapping in batches:
        assert len(source) <= 50
        for row in mapping:
            assert (
                text[row["start"] : row["end"]]
                == source[row["input_start"] : row["input_end"]]
                == row["text"]
            )


@pytest.mark.parametrize("reject_best", [False, True])
def test_extraction_selects_description_across_short_batches(monkeypatch, reject_best):
    """
    Verify extraction selects description across short batches.

    args:
        monkeypatch: Scoped replacements for the test.
        reject_best: Input for this operation.

    returns:
        None.
    """
    text = "Powódka miała wsparcie rodziny. Potrzebowała rehabilitacji."
    rows = [
        {
            **row,
            "unit": "paragraph",
            "answers": {
                "sekcja": {"choice": "fakty", "confidence": 0.8},
                "wyciek": {"noul": 0.01},
            },
        }
        for row in extraction.sentence_fragments(
            {"text": text, "start": 0, "end": len(text)}
        )
    ]

    async def paragraphs(*args):
        """
        Provide the paragraphs behavior used by this operation.

        args:
            args: Input for this operation.

        returns:
            Result of this operation.
        """
        return {"text": text, "start": 0, "end": len(text), "status": "test"}, rows

    async def answer(server, state, questions, config):
        """
        Provide the answer behavior used by this operation.

        args:
            server: Local inference server.
            state: Input state supplied to the model.
            questions: Model question definitions.
            config: Effective pipeline or acquisition settings.

        returns:
            Result of this operation.
        """
        if "instancja" in questions:
            return {"instancja": {"choice": "nieustalona", "confidence": 0.8}}
        if "sekcja" in questions:
            role = (
                "rozwazania"
                if reject_best and "rehabilitacji" in state["oceniany_fragment"]
                else "fakty"
            )
            return {
                "sekcja": {"choice": role, "confidence": 0.8},
                "wyciek": {"noul": 0.01},
            }
        return {
            "opis": {
                "data": {"pomoc": state},
                "fields": {
                    "pomoc": {
                        "start": 0,
                        "end": len(state),
                        "confidence": 0.05,
                        "line_probability": 0.8 if "rehabilitacji" in state else 0.2,
                    }
                },
            }
        }

    monkeypatch.setattr(extraction, "classify_paragraphs", paragraphs)
    monkeypatch.setattr(extraction, "ask", answer)
    config = {
        **extraction.load_config(),
        "fields": {"pomoc": "Jakiej pomocy potrzebowała?"},
        "chunk_characters": 35,
        "context_characters": 100,
        "system_one": {
            "minimum_probability": 0.3,
            "maximum_leak_probability": 0.15,
            "amount_minimum_probability": 0.3,
        },
    }
    result = asyncio.run(extraction.extract(None, text, config, 1))
    assert result["features"]["pomoc"] == (
        "Powódka miała wsparcie rodziny."
        if reject_best
        else "Potrzebowała rehabilitacji."
    )
    assert result["confidence"]["pomoc"] == 0.05
    assert len(result["native_extraction"]) == 2


AWARD_TEXT = """WYROK
sprawy z powództwa W. W.
I. Zasądza na rzecz powódki W. W. kwotę 17 176 zł z odsetkami od kwoty 15 000 zł i od kwoty 2 176 zł.
II. Zasądza na rzecz powódki W. W. kwotę 6 617 zł tytułem kosztów procesu.
UZASADNIENIE
Powódka żądała 99 000 zł tytułem zadośćuczynienia.
Sąd ustalił:
Powódka złamała nogę.
Sąd zważył:
Sąd uznał, że kwota 15 000 zł tytułem zadośćuczynienia jest odpowiednia.
Roszczenie o zwrot kosztów leczenia w wysokości 2 176 zł zasługiwało na uwzględnienie w całości.
"""


def assigned_amounts(text, categories):
    """
    Provide the assigned amounts behavior used by this operation.

    args:
        text: Normalized Polish source text.
        categories: Input for this operation.

    returns:
        Result of this operation.
    """
    return [
        {
            **row,
            "category": categories.get(row["kwota"], "pomin"),
            "accepted": row["kwota"] in categories,
        }
        for row in extraction.amount_contexts(text)
    ]


def test_amount_input_preserves_sentence_and_marks_only_one_occurrence():
    """
    Verify amount input preserves sentence and marks only one occurrence.

    args:
        None

    returns:
        None.
    """
    text = "Sąd zasądza 15 000 zł tytułem zadośćuczynienia oraz 2 176 zł tytułem odszkodowania. Koszty rozlicza oddzielnie."
    rows = list(extraction.amount_contexts(text))
    assert len(rows) == 2
    for row in rows:
        assert row["text"] == text[: text.index(" Koszty")]
        assert row["marked"].count("<KWOTA>") == 1
        assert (
            row["marked"].replace("<KWOTA>", "").replace("</KWOTA>", "") == row["text"]
        )
        assert text[row["start"] : row["end"]] == row["text"]
        assert feasibility.MONEY.fullmatch(
            text[row["amount_start"] : row["amount_end"]]
        )


@pytest.mark.parametrize(
    "sentence",
    [
        "Zasądza tytułem zadośćuczynienia 445.000 zł.",
        "Zasądza 445 000 zł tytułem zadośćuczynienia.",
        "II. zasądza kwotę 6.400 złotych tytułem zwrotu kosztów instancji odwoławczej.",
    ],
)
def test_amount_candidates_do_not_depend_on_word_order_or_payment_type(sentence):
    """
    Verify amount candidates do not depend on word order or payment type.

    args:
        sentence: Input for this operation.

    returns:
        None.
    """
    rows = list(extraction.amount_contexts(sentence))
    assert len(rows) == 1
    assert rows[0]["text"] == sentence


def test_list_item_keeps_its_request_introduction():
    """
    Verify list item keeps its request introduction.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nUZASADNIENIE\nPowódka wniosła o zapłatę:\n- kwoty 2 176 zł tytułem leczenia."
    row = next(extraction.amount_contexts(text))
    state = extraction.award_state(text, row, 100)
    assert "Powódka wniosła o zapłatę:" in state["poprzedzajacy_kontekst"]
    assert state["sekcja"] == "uzasadnienie"


def test_award_has_two_independent_components_and_does_not_add_repeated_mentions():
    """
    Verify award has two independent components and does not add repeated mentions.

    args:
        None

    returns:
        None.
    """
    rows = assigned_amounts(
        AWARD_TEXT,
        {
            "15000": "zadoscuczynienie",
            "2176": "pozostale_odszkodowanie",
            "17176": "suma",
        },
    )
    result = extraction.propose_award(AWARD_TEXT, rows + rows)
    assert [result[k] for k in (*extraction.AWARD_FIELDS, "suma")] == [
        "15000",
        "2176",
        "17176",
    ]
    assert all(
        AWARD_TEXT[q["start"] : q["end"]] == q["text"] for q in result["evidence"]
    )


def test_single_component_from_reasoning_is_retained_without_filling_other_with_zero():
    """
    Verify single component from reasoning is retained without filling other with zero.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nz powództwa W. W.\nZasądza 6300 zł.\nUZASADNIENIE\nSąd zasądził 6300 zł tytułem zadośćuczynienia."
    rows = assigned_amounts(text, {"6300": "zadoscuczynienie"})
    rows[0]["accepted"] = False
    result = extraction.propose_award(text, rows)
    assert result["zadoscuczynienie"] == "6300"
    assert result["pozostale_odszkodowanie"] == result["suma"] == "brak informacji"


def test_numeric_format_differences_do_not_create_a_false_conflict():
    """
    Verify numeric format differences do not create a false conflict.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nz powództwa W. W.\nUZASADNIENIE\nZasądzono 15000 zł.\nPonownie opisano 15.000,00 zł."
    rows = assigned_amounts(
        text, {"15000": "zadoscuczynienie", "15000.00": "zadoscuczynienie"}
    )
    assert extraction.propose_award(text, rows)["zadoscuczynienie"] == "15000"


def test_ruling_assignment_has_priority_over_a_conflicting_reasoning_amount():
    """
    Verify ruling assignment has priority over a conflicting reasoning amount.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nz powództwa W. W.\nZasądza 6300 zł.\nUZASADNIENIE\nWcześniej wypłacono 700 zł."
    rows = assigned_amounts(
        text, {"6300": "zadoscuczynienie", "700": "zadoscuczynienie"}
    )
    assert extraction.propose_award(text, rows)["zadoscuczynienie"] == "6300"


def test_conflicting_components_are_not_guessed_from_a_total():
    """
    Verify conflicting components are not guessed from a total.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nz powództwa W. W.\nZasądza 17176 zł.\nUZASADNIENIE\nOpisano 15000 zł i 3000 zł."
    rows = assigned_amounts(
        text,
        {
            "17176": "suma",
            "15000": "zadoscuczynienie",
            "3000": "pozostale_odszkodowanie",
        },
    )
    result = extraction.propose_award(text, rows)
    assert all(
        result[k] == "brak informacji" for k in (*extraction.AWARD_FIELDS, "suma")
    )
    assert result["kwota_nierozdzielona"] == "17176"
    assert "conflict" in result


@pytest.mark.parametrize("probability", [0.55, 0.8])
def test_context_decisions_are_followed_by_whole_award_check(monkeypatch, probability):
    """
    Verify context decisions are followed by whole award check.

    args:
        monkeypatch: Scoped replacements for the test.
        probability: Input for this operation.

    returns:
        None.
    """
    seen = []
    text = "WYROK\nz powództwa W. W.\nZasądza tytułem zadośćuczynienia 445000 zł.\nUZASADNIENIE\nOpis."

    async def answer(server, state, questions, config):
        """
        Provide the answer behavior used by this operation.

        args:
            server: Local inference server.
            state: Input state supplied to the model.
            questions: Model question definitions.
            config: Effective pipeline or acquisition settings.

        returns:
            Result of this operation.
        """
        seen.append((state, questions))
        if "swiadczenie" in questions:
            assert (
                "tytułem zadośćuczynienia <KWOTA>445000 zł</KWOTA>" in state["fragment"]
            )
            return {"swiadczenie": {"choice": "zadoscuczynienie", "confidence": 0.9}}
        return {"zasadzenie": {"noul": probability}}

    monkeypatch.setattr(extraction, "ask", answer)
    result = asyncio.run(
        extraction.extract_award(
            None,
            text,
            [],
            {
                **extraction.load_config(),
                "system_one": {"amount_minimum_probability": 0.6},
            },
        )
    )
    assert len(seen) == 2
    assert result["confirmed"] == (probability >= 0.6)
    assert result["zadoscuczynienie"] == (
        "445000" if probability >= 0.6 else "brak informacji"
    )
    if probability < 0.6:
        assert result["proposal"]["zadoscuczynienie"] == "445000"


def test_ignored_cost_context_does_not_become_a_target(monkeypatch):
    """
    Verify ignored cost context does not become a target.

    args:
        monkeypatch: Scoped replacements for the test.

    returns:
        None.
    """
    text = "WYROK\nz powództwa W. W.\nII. zasądza 6400 zł tytułem zwrotu kosztów instancji odwoławczej.\nUZASADNIENIE\nOpis."

    async def answer(server, state, questions, config):
        """
        Provide the answer behavior used by this operation.

        args:
            server: Local inference server.
            state: Input state supplied to the model.
            questions: Model question definitions.
            config: Effective pipeline or acquisition settings.

        returns:
            Result of this operation.
        """
        assert "kosztów instancji odwoławczej" in state["fragment"]
        return {"swiadczenie": {"choice": "pomin", "confidence": 0.9}}

    monkeypatch.setattr(extraction, "ask", answer)
    result = asyncio.run(
        extraction.extract_award(
            None,
            text,
            [],
            {
                **extraction.load_config(),
                "system_one": {"amount_minimum_probability": 0.6},
            },
        )
    )
    assert result["pozostale_odszkodowanie"] == "brak informacji"
    assert not result["confirmed"]


def test_reasoning_total_cannot_replace_the_ruling():
    """
    Verify reasoning total cannot replace the ruling.

    args:
        None

    returns:
        None.
    """
    text = (
        "WYROK\nz powództwa W. W.\nZasądza 445000 zł.\nUZASADNIENIE\nŻądano 1050000 zł."
    )
    rows = assigned_amounts(text, {"445000": "zadoscuczynienie", "1050000": "suma"})
    result = extraction.propose_award(text, rows)
    assert result["zadoscuczynienie"] == "445000"
    assert result["kwota_nierozdzielona"] == "brak informacji"


def test_recognized_ruling_amount_disambiguates_previous_payment_and_total_harm():
    """
    Verify recognized ruling amount disambiguates previous payment and total harm.

    args:
        None

    returns:
        None.
    """
    text = "WYROK\nz powództwa W. W.\nZasądza 6300 zł.\nUZASADNIENIE\nWypłacono 700 zł.\nKrzywdę oceniono na 7000 zł.\nZasądzono 6300 zł tytułem zadośćuczynienia."
    rows = assigned_amounts(
        text,
        {
            "6300": "zadoscuczynienie",
            "700": "zadoscuczynienie",
            "7000": "zadoscuczynienie",
        },
    )
    rows[0]["category"] = "suma"
    result = extraction.propose_award(text, rows)
    assert result["zadoscuczynienie"] == "6300"
    assert result["pozostale_odszkodowanie"] == "brak informacji"


def test_dataset_configs_expose_only_requested_controls():
    """
    Verify dataset configs expose only requested controls.

    args:
        None

    returns:
        None.
    """
    root = Path(__file__).resolve().parents[1]
    basal = extraction.read_config(root / "configs/dataset/basal.yaml")
    api = extraction.read_config(root / "configs/dataset/api.yaml")
    assert set(api) == {
        "max_age_years",
        "max_pages",
        "max_gib",
        "debug",
        "download_path",
        "timeout",
    }
    assert set(basal) == {
        "device",
        "feature_fields",
        "target_fields",
        "minimum_probability",
        "maximum_leak_probability",
        "amount_minimum_probability",
    }
    assert api["max_pages"] == 0


def test_feature_and_target_selection_is_applied(tmp_path):
    """
    Verify feature and target selection is applied.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    import yaml

    root = Path(__file__).resolve().parents[1]
    basal = extraction.read_config(root / "configs/dataset/basal.yaml")
    basal["feature_fields"] = ["event_circumstances"]
    basal["target_fields"] = ["zadoscuczynienie"]
    (tmp_path / "basal.yaml").write_text(yaml.safe_dump(basal))
    manifest = {"basal": "basal.yaml", "api": str(root / "configs/dataset/api.yaml")}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(manifest))
    config = extraction.load_config(path)
    assert list(config["fields"]) == ["event_circumstances"]
    rows = assigned_amounts(
        AWARD_TEXT, {"15000": "zadoscuczynienie", "2176": "pozostale_odszkodowanie"}
    )
    award = extraction.propose_award(AWARD_TEXT, rows, config["target_fields"])
    assert award["zadoscuczynienie"] == "15000"
    assert "pozostale_odszkodowanie" not in award
    assert award["suma"] == "brak informacji"


def test_api_age_limit_and_fixed_query():
    """
    Verify api age limit and fixed query.

    args:
        None

    returns:
        None.
    """
    start, end, params = feasibility.search_settings(
        {"max_age_years": 10, "max_pages": 0, "debug": False}, date(2026, 10, 29)
    )
    assert (start, end) == (date(2016, 10, 29), date(2026, 10, 29))
    assert params["all"] == "zadośćuczynienie zł"
    assert params["courtType"] == "COMMON" and params["judgmentTypes"] == "SENTENCE"
    assert params["judgmentDateFrom"] == "2016-10-29"
    assert feasibility.search_settings(
        {"max_age_years": 10, "max_pages": 1, "debug": False}, date(2024, 2, 29)
    )[0] == date(2014, 2, 28)


@pytest.mark.parametrize(
    ("limit", "expected"), [(0, [0, 1, 2]), (1, [0]), (2, [0, 1]), (8, [0, 1, 2])]
)
def test_page_limit_zero_reads_all_available_pages(monkeypatch, limit, expected):
    """
    Verify page limit zero reads all available pages.

    args:
        monkeypatch: Scoped replacements for the test.
        limit: Maximum batch or fragment size.
        expected: Expected result.

    returns:
        None.
    """
    seen = []

    def search(params):
        """
        Provide the search behavior used by this operation.

        args:
            params: Effective SAOS query parameters.

        returns:
            Result of this operation.
        """
        seen.append(params["pageNumber"])
        return {
            "items": [{"id": params["pageNumber"] + 1}],
            "info": {"totalResults": 250},
        }

    monkeypatch.setattr(feasibility, "search_judgments", search)
    pages = list(
        feasibility.search_pages(
            {"max_age_years": 10, "max_pages": limit, "debug": False}
        )
    )
    assert seen == expected
    assert [number for number, _ in pages] == expected


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("max_age_years", 0),
        ("max_age_years", True),
        ("max_pages", -1),
        ("max_pages", 1.5),
        ("debug", "true"),
    ],
)
def test_invalid_acquisition_configuration_is_rejected(key, value):
    """
    Verify invalid acquisition configuration is rejected.

    args:
        key: Input for this operation.
        value: Value to serialize or adapt.

    returns:
        None.
    """
    config = {"max_age_years": 10, "max_pages": 0, "debug": False, key: value}
    with pytest.raises(ValueError):
        feasibility.search_settings(config)


def test_debug_files_are_optional_and_source_round_trips(tmp_path):
    """
    Verify debug files are optional and source round trips.

    args:
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    result = {
        "features": {"opis": "Opis."},
        "award": {"zadoscuczynienie": "100", "evidence": []},
        "diagnostics": [],
    }
    extraction.write_paragraphs(result, tmp_path / "normal", 0.6)
    assert {p.name for p in (tmp_path / "normal").iterdir()} == {
        "record.json",
        "award.json",
    }
    extraction.write_paragraphs(result, tmp_path / "debug", 0.6, debug=True)
    assert (tmp_path / "debug/facts.txt").exists()
    assert (tmp_path / "debug/analysis.json").exists()
    payload = {"data": {"id": 7, "textContent": "<p>Pełny tekst.</p>"}}
    feasibility.save_source(payload, tmp_path / "debug")
    assert feasibility.fixture_texts(tmp_path) == {7: "Pełny tekst."}


def test_debug_download_propagates_original_judgment(monkeypatch, tmp_path):
    """
    Verify debug download propagates original judgment.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    payload = {
        "data": {
            "id": 42,
            "judgmentDate": date.today().isoformat(),
            "textContent": "<p>Treść.</p>",
        }
    }
    monkeypatch.setattr(feasibility, "OUTPUT", tmp_path)
    monkeypatch.setattr(
        feasibility, "search_pages", lambda config: iter([(0, {"items": [{"id": 42}]})])
    )
    monkeypatch.setattr(feasibility, "fetch_json", lambda url: payload)
    config = {"max_age_years": 10, "max_pages": 1, "debug": False}
    assert feasibility.judgment_texts(config) == {42: "Treść."}
    assert not (tmp_path / "42").exists()
    config["debug"] = True
    feasibility.judgment_texts(config)
    import json

    assert json.loads((tmp_path / "42/judgment.json").read_text()) == payload


def test_debug_trace_contains_real_request_and_response_even_from_cache(
    monkeypatch, tmp_path
):
    """
    Verify debug trace contains real request and response even from cache.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    config = extraction.load_config()
    monkeypatch.setattr(extraction, "ROOT", tmp_path)
    calls = []
    response = {
        "answers": {"q": {"noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}}
    }

    class Server:
        async def serve(self, request):
            """
            Provide the serve behavior used by this operation.

            args:
                request: Input for this operation.

            returns:
                Result of this operation.
            """
            calls.append(request)
            return response

    questions = {"q": {"type": "noul", "instructions": "Czy to fakt?"}}
    trace = []
    token = extraction.MODEL_CALLS.set(trace)
    try:

        async def run():
            """
            Execute the mocked asynchronous operation.

            args:
                None

            returns:
                None.
            """
            await extraction.ask(Server(), "Opis.", questions, config)
            await extraction.ask(Server(), "Opis.", questions, config)

        asyncio.run(run())
    finally:
        extraction.MODEL_CALLS.reset(token)
    assert len(calls) == 1 and len(trace) == 2
    assert trace[0]["request"] == {
        "state": "Opis.",
        "lang": "pl",
        "questions": questions,
    }
    assert trace[1]["response"] == response
