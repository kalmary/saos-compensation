"""Lokalna próba polskiego modelu decyzyjnego: uv run --no-sync notebooks/03_system_one_polish.py."""

import asyncio
import copy
import hashlib
import importlib
import json
import math
import re
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from contextvars import ContextVar

import yaml

ROOT = Path(__file__).resolve().parents[1]
REQUEST_COUNTS = {"cached": 0, "computed": 0}
MODEL_CALLS = ContextVar("model_calls", default=None)


DEFAULTS = {
    "version": "saos-extraction-4",
    "chunk_characters": 1200,
    "context_characters": 600,
    "feature_source": "semantic_paragraphs",
    "system_one": {
        "model": "Remek/basal-2.0-4.5B",
        "revision": "e5ba9149760d99b73b1963d8e0ccf948b58afacb",
        "mode": "eager",
        "device": "cpu",
        "template_prefix": "off",
        "context_overflow": "refuse",
    },
    "language": "pl",
    "questions": {
        "section": {
            "instructions": "Jaka jest funkcja wyłącznie ocenianego fragmentu? Poprzedni i "
            "następny tekst służą tylko ustaleniu kontekstu i autora twierdzenia. "
            "Nie przypisuj ich treści ocenianemu fragmentowi. Opis faktów może "
            "znajdować się w UZASADNIENIU bez własnego nagłówka. Odróżnij co się "
            "wydarzyło od dlaczego sąd wydał taki wyrok lub przyznał taką kwotę. "
            "Jeśli treści są połączone, wybierz mieszany; jeśli brak podstaw do "
            "decyzji, niejednoznaczny.",
            "criteria": {
                "fakty": "Opis ustalonych faktów tej sprawy: zdarzenia, krzywdy, osoby "
                "lub jej sytuacji i następstw. Bez żądań i bez oceny prawnej.",
                "stanowiska": "Żądania, zarzuty lub twierdzenia stron, bez potwierdzenia "
                "ich przez sąd.",
                "rozstrzygniecie": "Sentencja: co sąd zasądził, oddalił lub nakazał.",
                "rozwazania": "Rozważania prawne, precedensy lub uzasadnienie "
                "rozstrzygnięcia.",
                "mieszany": "W jednym fragmencie połączono opis faktów z oceną prawną, "
                "uzasadnieniem kwoty lub niepotwierdzonymi twierdzeniami "
                "strony.",
                "niejednoznaczny": "Nie można rozstrzygnąć, czy to ustalenia sądu, "
                "twierdzenia strony czy rozważania; niewystarczający "
                "kontekst.",
                "inne": "Nagłówek, metadane, podpis, sygnatura lub sama lista dowodów.",
            },
        },
        "leakage": {
            "instructions": "Oceń wyłącznie oceniany fragment, nie jego sąsiadów. Czy ujawnia "
            "kwotę żądaną, zasądzoną lub wcześniej wypłaconą, wynik postępowania, "
            "albo wyjaśnia wybór wysokości świadczenia? Także bez podania liczby. "
            "Sam opis urazu i jego skutków nie jest uzasadnieniem wysokości."
        },
        "amount": {
            "instructions": "Czego dotyczy wyłącznie kwota oznaczona <KWOTA>? Przeczytaj jej "
            "otoczenie i sentencję. Wybierz zasądzone świadczenie tylko dla tego "
            "wyroku i wskazanego poszkodowanego. Nie pomyl żądania ani "
            "wcześniejszej wypłaty z zasądzeniem. Jeżeli sąd uznaje za odpowiednie "
            "7000 zł, ale po odjęciu wypłaty zasądza 6300 zł, targetem jest 6300 "
            "zł. Zwrot kosztów instancji odwoławczej jest kosztem postępowania, "
            "nigdy pozostałym odszkodowaniem. Kwota główna z odsetkami nadal może "
            "być targetem; same odsetki nim nie są. Nie przenoś kategorii "
            "sąsiedniej liczby na oznaczoną kwotę.",
            "criteria": {
                "zadoscuczynienie": "Oznaczona kwota to zadośćuczynienie zasądzone temu "
                "poszkodowanemu w analizowanym wyroku.",
                "pozostale_odszkodowanie": "Oznaczona kwota to zasądzone temu "
                "poszkodowanemu jednorazowe odszkodowanie za "
                "szkodę majątkową, np. zwrot leczenia lub "
                "utraconych zarobków.",
                "suma": "Oznaczona kwota jest zasądzoną sumą świadczeń głównych dla tego "
                "poszkodowanego, której nie można tutaj przypisać do jednego "
                "składnika.",
                "pomin": "Kwota żądana, wcześniej wypłacona, z innego wyroku, dla innego "
                "odbiorcy, koszty procesu/sądowe/instancji "
                "odwoławczej/zastępstwa, odsetki lub renta; poza targetami.",
                "nieustalone": "Nie ma podstaw do przypisania oznaczonej kwoty do "
                "zasądzonego świadczenia.",
            },
        },
        "award_check": {
            "instructions": "Czy propozycja jest zgodna z sentencją analizowanego wyroku i "
            "dowodami? Odbiorcą ma być wskazany poszkodowany, nie fundacja ani "
            "inna osoba. Kwoty muszą być zasądzone w tym wyroku, nie tylko "
            "żądane, wcześniej wypłacone lub cytowane. Koszty procesu i "
            "sądowe, koszty instancji odwoławczej, odsetki oraz renta są "
            "wyłączone. Nieustalenie jednego składnika nie unieważnia "
            "drugiego. Nie traktuj zgodności sumy jako jedynego dowodu."
        },
        "extraction": {
            "instructions": "Wyodrębnij treść pól opisu sprawy zgodnie ze schematem. Zachowaj negacje, osobę "
            "i czas. Nie dopisuj ocen, nie używaj kwot ani uzasadnienia ich "
            "wysokości. Jeśli nie ma informacji dla pola, wybierz not stated.",
            "field_suffix": " Konkretny opis dotyczący poszkodowanego.",
        },
        "instance": {
            "instructions": "W której instancji zapadł ten wyrok? Wymagaj informacji o etapie "
            "postępowania. Nie wnioskuj wyłącznie z poziomu ani nazwy sądu. Przy "
            "braku dowodu wybierz nieustalona.",
            "criteria": {
                "pierwsza": "Pierwsza instancja.",
                "druga": "Druga instancja: rozpoznanie apelacji.",
                "nieustalona": "Brak jednoznacznej informacji o instancji.",
            },
        },
    },
}
FEATURE_DESCRIPTIONS = {
    "event_circumstances": "Zdarzenie będące przyczyną krzywdy i jego okoliczności.",
    "harm_type": "Konkretny rodzaj krzywdy poszkodowanego, np. obrażenie ciała, śmierć bliskiego albo "
    "naruszenie dobrego imienia.",
    "harm_severity_duration": "Przebieg dolegliwości, ich nasilenie, czas trwania oraz ustąpienie lub "
    "utrzymywanie się.",
    "life_impact": "Co zmieniło się w funkcjonowaniu, pracy, relacjach lub samodzielności i jakie skutki "
    "pozostały?",
    "claimant_circumstances": "Wiek, zawód, wcześniejsza sytuacja osobista i zdrowie "
    "poszkodowanego albo relacja ze zmarłym.",
    "treatment_assistance": "Jak łagodzono skutki i jakiego leczenia, wsparcia lub innej pomocy "
    "potrzebował poszkodowany?",
}


def read_config(path):
    """
    Read a YAML mapping and reject other configuration structures.

    args:
        path: Path to the relevant input or output file.

    returns:
        Parsed configuration mapping.
    """
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Konfiguracja musi być mapą: {path}")
    return value


def load_config(path=None):
    """
    Validate configuration and assemble effective extraction settings.

    args:
        path: Path to the relevant input or output file.

    returns:
        Effective validated configuration.
    """
    path = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("preprocessing.yaml")
    )
    settings = read_config(path)
    basal = read_config(path.parent / settings["basal"])
    api = read_config(path.parent / settings["api"])
    required = {
        "model",
        "device",
        "feature_fields",
        "target_fields",
        "minimum_probability",
        "maximum_leak_probability",
        "amount_minimum_probability",
    }
    if set(basal) != required:
        raise ValueError("basal.yaml wymaga tylko nazw pól i trzech progów pewności")
    revisions = {
        "Remek/basal-2.0-mini": "cb81b32ce89b0b87e62fcdad28f0435e25c953f3",
        "Remek/basal-2.0-4.5B": "e5ba9149760d99b73b1963d8e0ccf948b58afacb",
    }
    if not isinstance(basal["model"], str) or basal["model"] not in revisions:
        raise ValueError("model must be Remek/basal-2.0-mini or Remek/basal-2.0-4.5B")
    if basal["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    for key, allowed in (
        (
            "feature_fields",
            {
                *FEATURE_DESCRIPTIONS,
                "judgment_year",
                "instance",
                "court_seat",
                "court_locality_size",
            },
        ),
        ("target_fields", AWARD_FIELDS),
    ):
        fields = basal[key]
        if (
            not isinstance(fields, list)
            or not fields
            or not all(isinstance(field, str) and field in allowed for field in fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError(f"Nieprawidłowe nazwy pól: {key}")
    for key in (
        "minimum_probability",
        "maximum_leak_probability",
        "amount_minimum_probability",
    ):
        value = basal[key]
        if (
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Nieprawidłowy próg: {key}")
    config = {
        **settings,
        **copy.deepcopy(DEFAULTS),
        "api": api,
        "fields": {
            key: FEATURE_DESCRIPTIONS[key]
            for key in basal["feature_fields"]
            if key in FEATURE_DESCRIPTIONS
        },
        "target_fields": basal["target_fields"],
    }
    config["system_one"].update(model=basal["model"], revision=revisions[basal["model"]])
    config["system_one"]["mode"] = "mps" if basal["device"] == "mps" else "eager"
    config["system_one"]["device"] = basal["device"]
    config["system_one"].update(
        {
            key: basal[key]
            for key in (
                "minimum_probability",
                "maximum_leak_probability",
                "amount_minimum_probability",
            )
        }
    )
    return config


# %% Classify passages semantically and extract source-grounded descriptions; scores do not establish accuracy.
def fragments(text, limit):
    # Split at paragraphs and spaces while preserving source offsets; the model determines meaning.
    """
    Yield bounded paragraph fragments with original source offsets.

    args:
        text: Normalized Polish source text.
        limit: Maximum batch or fragment size.

    returns:
        Iterator of source fragments.
    """
    for line in re.finditer(r"[^\n]+", text):
        start = line.start()
        while start < line.end():
            end = min(start + limit, line.end())
            if end < line.end():
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            yield {"start": start, "end": end, "text": text[start:end]}
            start = end
            while start < line.end() and text[start].isspace():
                start += 1


def feature_input(text, config):
    # Classify all paragraphs by default; explicit headings remain a comparison variant.
    """
    Select source text for factual passage classification.

    args:
        text: Normalized Polish source text.
        config: Effective pipeline or acquisition settings.

    returns:
        Selected source text, offsets and selection status.
    """
    if config["feature_source"] == "semantic_paragraphs":
        return {
            "text": text,
            "start": 0,
            "end": len(text),
            "status": "semantyczny podział wszystkich akapitów z kontekstem",
        }
    if config["feature_source"] == "full_document":
        return {
            "text": text,
            "start": 0,
            "end": len(text),
            "status": "pełny dokument — próba porównawcza",
        }
    api = importlib.import_module("01_saos_feasibility")
    facts = api.split_sections(text)["fakty"]
    if not facts or text.count(facts) != 1:
        return {
            "text": "",
            "start": None,
            "end": None,
            "status": "brak jednoznacznych granic faktów",
        }
    start = text.index(facts)
    return {
        "text": facts,
        "start": start,
        "end": start + len(facts),
        "status": "między nagłówkami ustaleń faktycznych i rozważań; do przeglądu",
    }


def check_decision(response, questions, state=None):
    # Reject truncated inputs, incomplete answers and invalid probability distributions.
    """
    Validate model answers, probability distributions and extraction offsets.

    args:
        response: Raw model response.
        questions: Model question definitions.
        state: Input state supplied to the model.

    returns:
        Validated answer mapping.
    """
    if response.get("truncated"):
        raise ValueError("Silnik uciął kontekst")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Niekompletny zestaw odpowiedzi")
    for key, question in questions.items():
        answer = answers[key]
        if question["type"] == "extract":
            expected = set(question["schema"]["properties"])
            if (
                not isinstance(state, str)
                or set(answer.get("data", {})) != expected
                or set(answer.get("fields", {})) != expected
            ):
                raise ValueError("Niekompletny rekord ekstrakcji")
            for field, value in answer["data"].items():
                detail = answer["fields"][field]
                probability = detail.get("confidence")
                if probability is not None and (
                    type(probability) not in (int, float)
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError(f"Nieprawidłowa pewność pola {field}")
                if value is None:
                    continue
                start, end = detail.get("start"), detail.get("end")
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(state)
                    or state[start:end] != value
                ):
                    raise ValueError(
                        f"Pole {field} nie odpowiada fragmentowi źródłowemu"
                    )
            continue
        probabilities = answer.get("probabilities", {})
        if not probabilities or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()
        ):
            raise ValueError("Nieprawidłowe prawdopodobieństwa")
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ValueError("Rozkład nie sumuje się do 1")
        expected = (
            set(question["criteria"])
            if question["type"] == "choice"
            else {"true", "false"}
        )
        if set(probabilities) != expected:
            raise ValueError("Odpowiedź zawiera inne opcje niż pytanie")
        if question["type"] == "choice" and answer.get("choice") not in expected:
            raise ValueError("Wybrana opcja nie należy do przestrzeni odpowiedzi")
        selected = answer["choice"] if question["type"] == "choice" else "true"
        probability = (
            answer.get("confidence")
            if question["type"] == "choice"
            else answer.get("noul")
        )
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or abs(probability - probabilities[selected]) > 0.01
        ):
            raise ValueError("Wynik nie jest zgodny z rozkładem")
    return answers


async def ask(server, state, questions, config):
    """
    Request a validated model decision and optionally reuse its diagnostic cache.

    args:
        server: Local inference server.
        state: Input state supplied to the model.
        questions: Model question definitions.
        config: Effective pipeline or acquisition settings.

    returns:
        Validated model answers.
    """
    request = {"state": state, "lang": config["language"], "questions": questions}
    # Selection thresholds do not change model answers, so they do not invalidate the cache.
    identity = {
        "version": config["version"],
        "engine_version": "2.4.1",
        "model": {
            key: value
            for key, value in config["system_one"].items()
            if key
            not in {
                "minimum_probability",
                "maximum_leak_probability",
                "amount_minimum_probability",
            }
        },
    }
    key = hashlib.sha256(
        json.dumps([identity, request], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    path = ROOT / "tmp" / "system_one" / "decisions" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        REQUEST_COUNTS["cached"] += 1
        response = json.loads(path.read_text(encoding="utf-8"))["response"]
    else:
        REQUEST_COUNTS["computed"] += 1
        response = await server.serve(request)
        check_decision(response, questions, state)
        path.write_text(
            json.dumps(
                {
                    "request": request,
                    "response": response,
                    "model": config["system_one"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if MODEL_CALLS.get() is not None:
        MODEL_CALLS.get().append({"request": request, "response": response})
    return check_decision(response, questions, state)


def choices(question, criteria):
    """
    Build a model question with an explicit choice space.

    args:
        question: Instructions for the model question.
        criteria: Allowed model choices and their descriptions.

    returns:
        Choice-question definition.
    """
    return {"type": "choice", "instructions": question, "criteria": criteria}


def sentence_fragments(chunk):
    # Split mixed paragraphs into sentences without treating initials or decimal amounts as boundaries.
    """
    Split paragraphs into sentences while retaining original offsets.

    args:
        chunk: Source fragment with offsets.

    returns:
        Iterator of source sentences.
    """
    boundaries = [
        0,
        *[
            m.end()
            for m in re.finditer(
                r"(?<!\b[A-ZĄĆĘŁŃÓŚŹŻ]\.)(?<!\br\.)(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])",
                chunk["text"],
            )
        ],
        len(chunk["text"]),
    ]
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and chunk["text"][start].isspace():
            start += 1
        while end > start and chunk["text"][end - 1].isspace():
            end -= 1
        if end > start:
            yield {
                "text": chunk["text"][start:end],
                "start": chunk["start"] + start,
                "end": chunk["start"] + end,
            }


async def classify_paragraphs(server, text, config, identifier):
    """
    Classify source fragments with neighboring context and inspect mixed passages.

    args:
        server: Local inference server.
        text: Normalized Polish source text.
        config: Effective pipeline or acquisition settings.
        identifier: Stable judgment identifier.

    returns:
        Selected input metadata and fragment diagnostics.
    """
    selected = feature_input(text, config)
    chunks = [
        {
            **chunk,
            "start": chunk["start"] + selected["start"],
            "end": chunk["end"] + selected["start"],
        }
        for chunk in fragments(selected["text"], config["chunk_characters"])
    ]
    questions = {
        "sekcja": {"type": "choice", **config["questions"]["section"]},
        "wyciek": {"type": "noul", **config["questions"]["leakage"]},
    }
    diagnostics = []

    async def classify(chunk, unit, parent=None):
        """
        Classify one fragment and retain its contextual model decision.

        args:
            chunk: Source fragment with offsets.
            unit: Fragment granularity.
            parent: Original paragraph, when classifying a sentence.

        returns:
            Result of this operation.
        """
        context = config["context_characters"]
        state = {
            "oceniany_fragment": chunk["text"],
            "poprzedzajacy_kontekst": text[
                max(0, chunk["start"] - context) : chunk["start"]
            ],
            "nastepujacy_kontekst": text[chunk["end"] : chunk["end"] + context],
        }
        answer = await ask(server, state, questions, config)
        row = {**chunk, "unit": unit, "answers": answer, "state": state}
        if parent:
            row["parent_start"], row["parent_end"] = parent["start"], parent["end"]
        diagnostics.append(row)
        return row

    for i, chunk in enumerate(chunks):
        print(f"ID {identifier}: podział akapitu {i + 1}/{len(chunks)}", flush=True)
        row = await classify(chunk, "paragraph")
        decision = row["answers"]["sekcja"]
        if (
            decision["choice"] == "mieszany"
            and decision["confidence"] >= config["system_one"]["minimum_probability"]
        ):
            sentences = list(sentence_fragments(chunk))
            if len(sentences) > 1:
                for sentence in sentences:
                    await classify(sentence, "sentence", chunk)
    return selected, diagnostics


async def evaluate(server, config):
    # This small diagnostic control is not an independent SAOS benchmark.
    """
    Run the small section and leakage diagnostic control.

    args:
        server: Local inference server.
        config: Effective pipeline or acquisition settings.

    returns:
        Result of this operation.
    """
    cases = [
        (
            "Powódka złamała nogę i przez sześć tygodni poruszała się o kulach.",
            "fakty",
            False,
        ),
        (
            "Sąd ustalił: poszkodowany przed wypadkiem pracował jako kierowca.",
            "fakty",
            False,
        ),
        (
            "Powód żądał 80 000 zł, lecz sąd oddalił jego powództwo w całości.",
            None,
            True,
        ),
        (
            "Sąd uwzględnił żądanie w całości, uznając świadczenie za odpowiednie do krzywdy.",
            "rozwazania",
            True,
        ),
        (
            "Zasądza od pozwanego na rzecz powódki 17 176 zł wraz z odsetkami.",
            "rozstrzygniecie",
            True,
        ),
        (
            "Powódka twierdziła, że wymaga stałej opieki, czemu pozwany zaprzeczył. Sąd nie rozstrzygnął tej kwestii.",
            "stanowiska",
            False,
        ),
        (
            "Sąd ustalił, że powódka nie wymagała rehabilitacji ani pomocy innych osób.",
            "fakty",
            False,
        ),
    ]
    questions = {
        "sekcja": {"type": "choice", **config["questions"]["section"]},
        "wyciek": {"type": "noul", **config["questions"]["leakage"]},
    }
    rows = []
    for text, expected, leak in cases:
        answer = await ask(server, text, questions, config)
        rows.append(
            {
                "text": text,
                "expected_section": expected,
                "expected_leak": leak,
                "answers": answer,
                "correct_section": expected is None
                or answer["sekcja"]["choice"] == expected,
                "correct_leak": (answer["wyciek"]["noul"] >= 0.5) == leak,
            }
        )
    return rows


def select_facts(diagnostics, threshold, maximum_leak):
    """
    Mark factual passages that meet confidence and leakage constraints.

    args:
        diagnostics: Fragment decisions and their source locations.
        threshold: Minimum decision confidence.
        maximum_leak: Maximum admitted leakage score.

    returns:
        Accepted factual fragments.
    """
    api = importlib.import_module("01_saos_feasibility")
    for row in diagnostics:
        answer, section = row["answers"], row["answers"]["sekcja"]
        row["accepted"] = (
            section["choice"] == "fakty"
            and section["confidence"] >= threshold
            and answer["wyciek"]["noul"] <= maximum_leak
            and not api.CURRENCY.search(row["text"])
            and any(char.isalpha() for char in row["text"])
            and not row["text"].rstrip().endswith(":")
        )
    return [row for row in diagnostics if row["accepted"]]


def extraction_source(text, passages):
    # Each extraction input line maps to a complete source sentence, preserving negation and subject.
    """
    Assemble extraction input and map its sentences to source locations.

    args:
        text: Normalized Polish source text.
        passages: Accepted factual source fragments.

    returns:
        Assembled text and its source mapping.
    """
    sentences = {}
    for passage in passages:
        for sentence in sentence_fragments(passage):
            if text[sentence["start"] : sentence["end"]] != sentence["text"]:
                raise ValueError("Niezgodny fragment opisu sprawy")
            sentences[(sentence["start"], sentence["end"])] = sentence
    parts, mapping, position = [], [], 0
    for sentence in sorted(sentences.values(), key=lambda row: row["start"]):
        parts.append(sentence["text"])
        mapping.append(
            {
                **sentence,
                "input_start": position,
                "input_end": position + len(sentence["text"]),
            }
        )
        position += len(sentence["text"]) + 1
    return "\n".join(parts), mapping


def describe_record(answer, mapping, fields, threshold):
    """
    Recover complete source sentences for extracted feature values.

    args:
        answer: Validated extraction answer.
        mapping: Mapping between input sentences and source offsets.
        fields: Configured feature fields.
        threshold: Minimum decision confidence.

    returns:
        Feature values, evidence, statuses and confidence scores.
    """
    values, evidence, statuses, confidence = {}, {}, {}, {}
    for field in fields:
        detail = answer["fields"][field]
        value = answer["data"][field]
        confidence[field] = detail.get("confidence", detail.get("present_probability"))
        if value is None:
            values[field], evidence[field], statuses[field] = (
                "brak informacji",
                [],
                "model nie wyekstrahował opisu",
            )
            continue
        rows = [
            row
            for row in mapping
            if row["input_start"] < detail["end"] and row["input_end"] > detail["start"]
        ]
        if not rows:
            raise ValueError(f"Nie można odtworzyć źródła pola {field}")
        # Keep the complete selected sentence rather than a phrase stripped of its negation.
        rows = rows[:2]
        values[field] = " ".join(row["text"] for row in rows)
        evidence[field] = [
            {key: row[key] for key in ("text", "start", "end")} for row in rows
        ]
        uncertain = confidence[field] is not None and confidence[field] < threshold
        statuses[field] = (
            "opis o niskiej pewności — zachowany do przeglądu"
            if uncertain
            else "wyekstrahowano opis do przeglądu"
        )
    return values, evidence, statuses, confidence


def extraction_batches(text, passages, limit):
    """
    Yield bounded factual batches with source mappings.

    args:
        text: Normalized Polish source text.
        passages: Accepted factual source fragments.
        limit: Maximum batch or fragment size.

    returns:
        Iterator of extraction text and source mappings.
    """
    _, sentences = extraction_source(text, passages)
    batch, size = [], 0
    for sentence in sentences:
        if batch and size + len(sentence["text"]) + 1 > limit:
            yield extraction_source(text, batch)
            batch, size = [], 0
        batch.append(sentence)
        size += len(sentence["text"]) + 1
    if batch:
        yield extraction_source(text, batch)


# %% Detect numeric amount contexts; let the model determine their meaning.
AWARD_FIELDS = ("zadoscuczynienie", "pozostale_odszkodowanie")


def amount_contexts(text):
    """
    Yield complete sentence contexts with each PLN occurrence marked.

    args:
        text: Normalized Polish source text.

    returns:
        Iterator of marked amount contexts.
    """
    api = importlib.import_module("01_saos_feasibility")
    heading = list(api.REASONS.finditer(text))
    boundary = heading[0].start() if len(heading) == 1 else None
    # Use the complete sentence or list item; let the model determine the monetary role.
    for paragraph in fragments(text, max(1, len(text))):
        for sentence in sentence_fragments(paragraph):
            for match in api.MONEY.finditer(sentence["text"]):
                start, end = (
                    sentence["start"] + match.start(),
                    sentence["start"] + match.end(),
                )
                marked = (
                    sentence["text"][: match.start()]
                    + "<KWOTA>"
                    + match[0]
                    + "</KWOTA>"
                    + sentence["text"][match.end() :]
                )
                yield {
                    **sentence,
                    "kwota": str(
                        api.Decimal(
                            match["amount"]
                            .replace(" ", "")
                            .replace(".", "")
                            .replace(",", ".")
                        )
                    ),
                    "amount_start": start,
                    "amount_end": end,
                    "marked": marked,
                    "section": "sentencja"
                    if boundary is not None and start < boundary
                    else "uzasadnienie"
                    if boundary is not None
                    else "nieustalona",
                }


def award_state(text, row, context_characters=600):
    """
    Assemble the ruling and context for a selected monetary occurrence.

    args:
        text: Normalized Polish source text.
        row: Amount occurrence with its source context.
        context_characters: Maximum preceding context length.

    returns:
        Structured Polish model input.
    """
    api = importlib.import_module("01_saos_feasibility")
    heading = list(api.REASONS.finditer(text))
    ruling = (
        text[: heading[0].start()]
        if len(heading) == 1
        else "brak jednoznacznej granicy sentencji"
    )
    return {
        "fragment": row["marked"],
        "sekcja": row["section"],
        "sentencja": ruling,
        "poprzedzajacy_kontekst": text[
            max(0, row["start"] - context_characters) : row["start"]
        ],
    }


def propose_award(text, decisions, target_fields=None):
    """
    Consolidate compatible component assignments without summing repeated mentions.

    args:
        text: Normalized Polish source text.
        decisions: Amount assignments to consolidate.
        target_fields: Configured award components.

    returns:
        Consolidated award proposal with evidence.
    """
    target_fields = AWARD_FIELDS if target_fields is None else target_fields
    result = {
        field: "brak informacji"
        for field in (*target_fields, "suma", "kwota_nierozdzielona")
    }
    result.update(
        beneficjent="brak informacji",
        waluta="PLN",
        evidence=[],
        status="brak ustalonego zasądzenia",
    )
    api = importlib.import_module("01_saos_feasibility")
    heading = list(api.REASONS.finditer(text))
    if len(heading) != 1:
        result["status"] = "brak jednoznacznej granicy sentencji — do przeglądu"
        return result
    # Identify the beneficiary without distributing a joint award across multiple people.
    plaintiff = re.search(r"z powództwa\s+([^\n]+)", text[: heading[0].start()], re.I)
    if not plaintiff or re.search(r",|\boraz\b|\s+i\s+", plaintiff[1]):
        result["status"] = "nieustalony lub wieloosobowy beneficjent — do przeglądu"
        return result
    result["beneficjent"] = plaintiff[1].strip()
    selected = {}
    for field in ("suma", *target_fields):
        rows = [
            row for row in decisions if row["accepted"] and row["category"] == field
        ]
        direct = [row for row in rows if row["section"] == "sentencja"]
        # Do not sum repeated mentions; conflicting component values require review. Total evidence must come from the ruling.
        candidates = direct if field == "suma" else direct or rows
        if field in AWARD_FIELDS and not direct and "suma" in selected:
            total = api.Decimal(selected["suma"]["kwota"])
            # Prefer an explicitly assigned component equal to the total, selecting only among model assignments.
            exact = [row for row in candidates if api.Decimal(row["kwota"]) == total]
            candidates = exact or [
                row for row in candidates if api.Decimal(row["kwota"]) < total
            ]
        values = {api.Decimal(row["kwota"]) for row in candidates}
        if len(values) == 1:
            value = values.pop()
            selected[field] = next(
                row for row in candidates if api.Decimal(row["kwota"]) == value
            )
    for field in target_fields:
        if field in selected:
            result[field] = selected[field]["kwota"]
    if "suma" in selected:
        result["kwota_nierozdzielona"] = selected["suma"]["kwota"]
    if set(AWARD_FIELDS) <= set(target_fields) and all(
        field in selected for field in AWARD_FIELDS
    ):
        total = sum(api.Decimal(result[field]) for field in AWARD_FIELDS)
        if "suma" in selected and total != api.Decimal(selected["suma"]["kwota"]):
            result["status"] = "składniki nie zgadzają się z sumą — do przeglądu"
            result["conflict"] = {field: result[field] for field in AWARD_FIELDS}
            result.update({field: "brak informacji" for field in AWARD_FIELDS})
        else:
            result.update(suma=str(total), kwota_nierozdzielona="brak informacji")
    result["evidence"] = [
        {
            key: row[key]
            for key in ("text", "start", "end", "section", "amount_start", "amount_end")
        }
        for row in selected.values()
    ]
    if (
        result["status"] != "składniki nie zgadzają się z sumą — do przeglądu"
        and selected
    ):
        result["status"] = "propozycja z kontekstów kwot — do kontroli"
    return result


def finish_award(proposal, accepted, decision):
    """
    Retain confirmed award components or mark an unsupported proposal for review.

    args:
        proposal: Assembled award candidates.
        accepted: Whether the award proposal was confirmed.
        decision: Final model verification result.

    returns:
        Verified award result with unsupported values withheld.
    """
    result = {**proposal, "verification": decision, "confirmed": accepted}
    if not accepted:
        result["proposal"] = proposal
        result.update(
            {
                field: "brak informacji"
                for field in (*AWARD_FIELDS, "suma", "kwota_nierozdzielona")
                if field in result
            }
        )
        result["status"] = "model nie potwierdził propozycji — do przeglądu"
    return result


async def extract_award(server, text, diagnostics, config):
    """
    Assign contextual amounts and verify the assembled award proposal.

    args:
        server: Local inference server.
        text: Normalized Polish source text.
        diagnostics: Fragment decisions and their source locations.
        config: Effective pipeline or acquisition settings.

    returns:
        Award proposal and supporting amount decisions.
    """
    decisions = []
    threshold = config["system_one"]["amount_minimum_probability"]
    for row in amount_contexts(text):
        answer = (
            await ask(
                server,
                award_state(text, row, config.get("context_characters", 600)),
                {"swiadczenie": {"type": "choice", **config["questions"]["amount"]}},
                config,
            )
        )["swiadczenie"]
        decisions.append(
            {
                **row,
                "category": answer["choice"],
                "confidence": answer["confidence"],
                "accepted": answer["confidence"] >= threshold
                and answer["choice"] in {*AWARD_FIELDS, "suma"},
                "decision": answer,
            }
        )
    proposal = propose_award(text, decisions, config.get("target_fields", AWARD_FIELDS))
    if not proposal["evidence"] or "conflict" in proposal:
        return {**proposal, "confirmed": False, "amount_decisions": decisions}
    state = {
        "propozycja": {
            key: value for key, value in proposal.items() if key != "evidence"
        },
        "sentencja": award_state(text, decisions[0])["sentencja"],
        "dowody": proposal["evidence"],
    }
    answer = (
        await ask(
            server,
            state,
            {"zasadzenie": {"type": "noul", **config["questions"]["award_check"]}},
            config,
        )
    )["zasadzenie"]
    return {
        **finish_award(proposal, answer["noul"] >= threshold, answer),
        "amount_decisions": decisions,
    }


async def extract(server, text, config, identifier):
    """
    Extract source-grounded factual descriptions and separate monetary targets.

    args:
        server: Local inference server.
        text: Normalized Polish source text.
        config: Effective pipeline or acquisition settings.
        identifier: Stable judgment identifier.

    returns:
        Feature and target results with review evidence.
    """
    fields = config["fields"]
    threshold = config["system_one"]["minimum_probability"]
    api = importlib.import_module("01_saos_feasibility")
    heading = api.REASONS.search(text)
    header = text[: min(heading.start() if heading else len(text), 4000)]
    instance = await ask(
        server,
        header,
        {"instancja": {"type": "choice", **config["questions"]["instance"]}},
        config,
    )
    instance = instance["instancja"]
    # Separate factual passages from reasoning and other content before feature extraction.
    selected, diagnostics = await classify_paragraphs(server, text, config, identifier)
    passages = select_facts(
        diagnostics, threshold, config["system_one"]["maximum_leak_probability"]
    )
    descriptions = {field: "brak informacji" for field in fields}
    evidence = {field: [] for field in fields}
    field_status = {
        field: "brak wyodrębnionego opisu sprawy"
        if not passages
        else "model nie wyekstrahował opisu"
        for field in fields
    }
    confidence = {field: None for field in fields}
    native, candidates = [], {field: [] for field in fields}
    schema = {
        "type": "object",
        "properties": {
            field: {
                "type": "string",
                "description": question
                + config["questions"]["extraction"]["field_suffix"],
            }
            for field, question in fields.items()
        },
    }
    # Short batches reduce competition between sentences when extracting source-grounded descriptions.
    for batch, (source, mapping) in enumerate(
        extraction_batches(text, passages, config["chunk_characters"]), 1
    ):
        print(
            f"ID {identifier}: ekstrakcja opisu, porcja {batch} ({len(source)} znaków)",
            flush=True,
        )
        answer = (
            await ask(
                server,
                source,
                {
                    "opis": {
                        "type": "extract",
                        "schema": schema,
                        "instructions": config["questions"]["extraction"][
                            "instructions"
                        ],
                    }
                },
                config,
            )
        )["opis"]
        native.append({"source": source, "mapping": mapping, "answer": answer})
        values, quotes, statuses, probabilities = describe_record(
            answer, mapping, fields, threshold
        )
        for field in fields:
            # Rank the complete selected sentence rather than the length of a phrase within it.
            score = answer["fields"][field].get("line_probability", 0)
            if quotes[field]:
                candidates[field].append(
                    (
                        score,
                        values[field],
                        quotes[field],
                        statuses[field],
                        probabilities[field],
                    )
                )
    # Check selected sentences independently because a paragraph can mix facts and reasoning.
    reviews = {}
    for field, options in candidates.items():
        for score, value, quotes, status, probability in sorted(
            options, key=lambda row: row[0], reverse=True
        ):
            start, end = quotes[0]["start"], quotes[-1]["end"]
            if (start, end) not in reviews:
                context = config["context_characters"]
                state = {
                    "oceniany_fragment": value,
                    "poprzedzajacy_kontekst": text[max(0, start - context) : start],
                    "nastepujacy_kontekst": text[end : end + context],
                }
                answers = await ask(
                    server,
                    state,
                    {
                        "sekcja": {"type": "choice", **config["questions"]["section"]},
                        "wyciek": {"type": "noul", **config["questions"]["leakage"]},
                    },
                    config,
                )
                row = {"text": value, "start": start, "end": end, "answers": answers}
                select_facts(
                    [row], threshold, config["system_one"]["maximum_leak_probability"]
                )
                reviews[start, end] = row
            if reviews[start, end]["accepted"]:
                descriptions[field], evidence[field] = value, quotes
                field_status[field], confidence[field] = status, probability
                break
            field_status[field] = (
                "odrzucone źródło opisu — poza faktami lub sygnał wycieku"
            )
    award = await extract_award(server, text, diagnostics, config)
    return {
        "id": identifier,
        "features": descriptions,
        "evidence": evidence,
        "award": award,
        "diagnostics": diagnostics,
        "field_status": field_status,
        "fields_evaluated": bool(passages),
        "confidence": confidence,
        "native_extraction": native,
        "description_reviews": list(reviews.values()),
        "paragraph_counts": dict(
            Counter(
                row["answers"]["sekcja"]["choice"]
                for row in diagnostics
                if row["unit"] == "paragraph"
            )
        ),
        "instance": {
            "value": instance["choice"]
            if instance["choice"] != "nieustalona"
            and instance["confidence"] >= threshold
            else "brak informacji",
            "decision": instance,
            "source": header,
            "status": "do weryfikacji",
        },
        "feature_input": {
            key: value for key, value in selected.items() if key != "text"
        },
        "input_characters": {
            "full": len(text),
            "classified": len(selected["text"]),
            "features": sum(len(row["text"]) for row in diagnostics if row["accepted"]),
        },
        "status": "kandydaci do weryfikacji; nie dopuszczono do treningu",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def write_paragraphs(result, directory, threshold, debug=False):
    # Export original passages for review; full decisions and source offsets remain in the analysis JSON.
    """
    Export feature values, award evidence and optional diagnostic passages.

    args:
        result: Extraction result and supporting evidence.
        directory: Input or output directory for this operation.
        threshold: Minimum decision confidence.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if "award" in result:
        award = (
            result["award"]
            if debug
            else {
                k: v
                for k, v in result["award"].items()
                if k not in {"evidence", "amount_decisions", "proposal", "verification"}
            }
        )
        (directory / "award.json").write_text(
            json.dumps(award, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (directory / "record.json").write_text(
        json.dumps(result["features"], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not debug:
        return
    (directory / "analysis.json").write_text(
        json.dumps(
            {k: v for k, v in result.items() if k != "model_calls"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if "model_calls" in result:
        (directory / "model_calls.json").write_text(
            json.dumps(result["model_calls"], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    diagnostics = result["diagnostics"]
    groups = {
        "facts.txt": [row for row in diagnostics if row["accepted"]],
        "reasoning.txt": [
            row
            for row in diagnostics
            if row["answers"]["sekcja"]["choice"] == "rozwazania"
            and row["answers"]["sekcja"]["confidence"] >= threshold
        ],
        "review.txt": [
            row
            for row in diagnostics
            if row["answers"]["sekcja"]["choice"] in {"mieszany", "niejednoznaczny"}
            or row["answers"]["sekcja"]["confidence"] < threshold
            or (row["answers"]["sekcja"]["choice"] == "fakty" and not row["accepted"])
        ],
    }
    for filename, rows in groups.items():
        (directory / filename).write_text(
            "\n\n".join(row["text"] for row in rows) + "\n", encoding="utf-8"
        )


async def run(texts=None, config=None):
    # Use pinned engine/model versions; the first run downloads weights and descriptions retain source sentences.
    """
    Run the local extraction experiment and preserve its review artifacts.

    args:
        texts: Judgment texts keyed by identifier.
        config: Effective pipeline or acquisition settings.

    returns:
        None.
    """
    from basal.server import Server, parser
    from huggingface_hub import snapshot_download

    config = config or load_config()
    REQUEST_COUNTS.update(cached=0, computed=0)
    model = config["system_one"]
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.dataset.get_data import check_device

    check_device(model["device"])
    print(
        "Progi: "
        + ", ".join(
            f"{key}={model[key]}"
            for key in (
                "minimum_probability",
                "maximum_leak_probability",
                "amount_minimum_probability",
            )
        ),
        flush=True,
    )
    # Exclude the separate think weights from the 4.5B repository.
    model_path = snapshot_download(
        model["model"],
        revision=model["revision"],
        cache_dir=ROOT / "basal_weights",
        ignore_patterns=["think/*"],
    )
    args = parser().parse_args(
        [
            "--model",
            model_path,
            "--mode",
            model["mode"],
            "--device",
            model["device"],
            "--template-prefix",
            model["template_prefix"],
            "--name",
            model["model"],
            "--context-overflow",
            model["context_overflow"],
        ]
    )
    server = Server(args)
    worker = asyncio.create_task(server.worker())
    try:
        started = time.perf_counter()
        evaluation = await evaluate(server, config)
        if texts is None:
            api = importlib.import_module("01_saos_feasibility")
            api.OUTPUT.mkdir(parents=True, exist_ok=True)
            texts = api.judgment_texts(config["api"])
        results = {}
        for identifier, text in texts.items():
            token = MODEL_CALLS.set([] if config["api"]["debug"] else None)
            try:
                results[identifier] = await extract(server, text, config, identifier)
                if config["api"]["debug"]:
                    results[identifier]["model_calls"] = MODEL_CALLS.get()
            finally:
                MODEL_CALLS.reset(token)
        output = {
            "config": config,
            "engine_version": "2.4.1",
            "engine_arguments": vars(args),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "requests": REQUEST_COUNTS.copy(),
            "evaluation": evaluation,
            "judgments": results,
            "seconds": round(time.perf_counter() - started, 2),
        }
        path = ROOT / "tmp" / "system_one" / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for identifier, result in results.items():
            directory = path.parent / str(identifier)
            write_paragraphs(
                result,
                directory,
                config["system_one"]["minimum_probability"],
                config["api"]["debug"],
            )
            if config["api"]["debug"]:
                (directory / "config.json").write_text(
                    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (directory / "source.txt").write_text(
                    texts[identifier] + "\n", encoding="utf-8"
                )
                source = (
                    ROOT
                    / "tmp"
                    / "saos_feasibility"
                    / str(identifier)
                    / "judgment.json"
                )
                if source.exists():
                    (directory / "judgment.json").write_bytes(source.read_bytes())
        for identifier, result in results.items():
            print(f"\nID {identifier}")
            print(json.dumps(result["features"], ensure_ascii=False, indent=2))
            print(
                "Zasądzone świadczenie:",
                json.dumps(
                    {
                        k: v
                        for k, v in result["award"].items()
                        if k not in ("amount_decisions", "proposal")
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            print(
                f"Zakres wejścia: {result['input_characters']}; {result['feature_input']['status']}"
            )
            print(f"Role akapitów: {result['paragraph_counts']}")
        print(
            f"Kontrola sekcji: {sum(r['correct_section'] for r in evaluation)}/{len(evaluation)}; "
            f"kontrola wycieku: {sum(r['correct_leak'] for r in evaluation)}/{len(evaluation)}"
        )
        print(f"Czas analizy: {output['seconds']} s; żądania: {output['requests']}")
        print(f"Pełne odpowiedzi i dowody: {path}")
        return output
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Przetwórz foldery fixtures/ bez pobierania SAOS",
    )
    options = parser.parse_args()
    texts = (
        importlib.import_module("01_saos_feasibility").fixture_texts()
        if options.fixtures
        else None
    )
    asyncio.run(run(texts))
