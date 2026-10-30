"""Basal fact selection, source-grounded descriptions and separate award extraction."""

import copy
import hashlib
import json
import math
import re
from collections import Counter
from contextvars import ContextVar
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import yaml

from . import text as api
from .get_data import search_settings

ROOT = Path(__file__).resolve().parents[2]
REQUEST_COUNTS = {"cached": 0, "computed": 0}
MODEL_CALLS = ContextVar("model_calls", default=None)


DEFAULTS = {
    "version": "saos-extraction-4",
    "chunk_characters": 1200,
    "context_characters": 600,
    "system_one": {
        "model": "Remek/basal-2.0-4.5B",
        "revision": "e5ba9149760d99b73b1963d8e0ccf948b58afacb",
        "mode": "mps",
        "device": "mps",
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
    "zdarzenie_i_okolicznosci": "Zdarzenie będące przyczyną krzywdy i jego okoliczności.",
    "rodzaj_doznanej_krzywdy": "Konkretny rodzaj krzywdy poszkodowanego, np. obrażenie ciała, śmierć bliskiego albo "
    "naruszenie dobrego imienia.",
    "przebieg_nasilenie_i_czas_trwania": "Przebieg dolegliwości, ich nasilenie, czas trwania oraz ustąpienie lub "
    "utrzymywanie się.",
    "konsekwencje_dla_zycia": "Co zmieniło się w funkcjonowaniu, pracy, relacjach lub samodzielności i jakie skutki "
    "pozostały?",
    "indywidualne_okolicznosci_poszkodowanego": "Wiek, zawód, wcześniejsza sytuacja osobista i zdrowie "
    "poszkodowanego albo relacja ze zmarłym.",
    "podjete_dzialania_i_potrzebna_pomoc": "Jak łagodzono skutki i jakiego leczenia, wsparcia lub innej pomocy "
    "potrzebował poszkodowany?",
}
METADATA_FIELDS = (
    "rok_wyroku",
    "instancja",
    "miejscowosc_sadu",
    "wielkość miejscowości",
)


def read_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Konfiguracja musi być mapą: {path}")
    return value


def load_config(directory=None, overrides=None):
    directory = (
        Path(directory) if directory is not None else ROOT / "configs" / "dataset"
    )
    basal = read_config(directory / "basal.yaml")
    api = read_config(directory / "api.yaml")
    api.update(overrides or {})
    search_settings(api)
    required = {
        "feature_fields",
        "target_fields",
        "minimum_probability",
        "maximum_leak_probability",
        "amount_minimum_probability",
    }
    if set(basal) != required:
        raise ValueError("basal.yaml wymaga tylko nazw pól i trzech progów pewności")
    for key, allowed in (
        ("feature_fields", {*FEATURE_DESCRIPTIONS, *METADATA_FIELDS}),
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
        **copy.deepcopy(DEFAULTS),
        "api": api,
        "fields": {
            key: FEATURE_DESCRIPTIONS[key]
            for key in basal["feature_fields"]
            if key in FEATURE_DESCRIPTIONS
        },
        "metadata_fields": [
            key for key in basal["feature_fields"] if key in METADATA_FIELDS
        ],
        "target_fields": basal["target_fields"],
    }
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


# Extract wypełnia pola fragmentami źródła; choice i noul służą kontroli wejścia.
# Prawdopodobieństwa nie dowodzą poprawności.
def fragments(text, limit):
    # Podział techniczny po akapitach i odstępach; znaczenie ocenia model.
    # Każdy znak niebędący odstępem ma fragment i offset w pełnym tekście.
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
    return {
        "text": text,
        "start": 0,
        "end": len(text),
        "status": "semantyczny podział wszystkich akapitów z kontekstem",
    }


def check_decision(response, questions, state=None):
    # Walidacja odpowiedzi silnika: nie akceptujemy ucięcia ani wadliwego rozkładu.
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
    request = {"state": state, "lang": config["language"], "questions": questions}
    # Progi selekcji nie zmieniają odpowiedzi modelu, więc nie unieważniają cache.
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
    path = ROOT / config["api"]["download_path"] / "debug" / "decisions" / f"{key}.json"
    debug = config["api"]["debug"]
    if debug:
        path.parent.mkdir(parents=True, exist_ok=True)
    if debug and path.exists():
        REQUEST_COUNTS["cached"] += 1
        response = json.loads(path.read_text(encoding="utf-8"))["response"]
    else:
        REQUEST_COUNTS["computed"] += 1
        response = await server.serve(request)
        check_decision(response, questions, state)
        if debug:
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
    return {"type": "choice", "instructions": question, "criteria": criteria}


def sentence_fragments(chunk):
    # Podział techniczny akapitu mieszanego. Reguła nie przypisuje roli zdania.
    # Inicjały osób i liczby z groszami nie powinny wyznaczać granicy.
    boundaries = [
        0,
        *[
            m.end()
            for m in re.finditer(
                r"(?<!\b[A-ZĄĆĘŁŃÓŚŹŻ]\.)(?<!\br\.)"
                r"(?<!(?i:\b(?:tys|mln|mld)\.))(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])",
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


def select_facts(diagnostics, threshold, maximum_leak):
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
    # Każda linia wejścia extract jest zdaniem z dokładnym miejscem w wyroku.
    # Ułatwia to krótki opis i zachowuje negację oraz podmiot wypowiedzi.
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
        # Do opisu włączamy całe wybrane zdanie, nie wyrwany wyraz bez negacji.
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


AWARD_FIELDS = ("zadoscuczynienie", "pozostale_odszkodowanie")


def amount_contexts(text):
    heading = list(api.REASONS.finditer(text))
    boundary = heading[0].start() if len(heading) == 1 else None
    # Całe zdanie lub punkt listy, bez okna słów i bez regexów znaczenia świadczenia.
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
                    "kwota": api.amount_in_pln(match),
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
    heading = list(api.REASONS.finditer(text))
    if len(heading) != 1:
        result["status"] = "brak jednoznacznej granicy sentencji — do przeglądu"
        return result
    # To metadane odbiorcy, nie reguła znaczenia kwoty. Nie dzielimy wspólnego zasądzenia między osoby.
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
        # Nie sumujemy wystąpień. Konflikt kilku wartości w jednej kategorii wymaga przeglądu.
        # Suma zasądzona musi mieć dowód w sentencji, nie w opisie żądań.
        candidates = direct if field == "suma" else direct or rows
        if field in AWARD_FIELDS and not direct and "suma" in selected:
            total = Decimal(selected["suma"]["kwota"])
            # Najpierw jawnie przypisany składnik równy całemu zasądzeniu.
            # Nie podstawiamy kwoty za model: wybieramy spośród jego przypisań.
            exact = [row for row in candidates if Decimal(row["kwota"]) == total]
            candidates = exact or [
                row for row in candidates if Decimal(row["kwota"]) < total
            ]
        values = {Decimal(row["kwota"]) for row in candidates}
        if len(values) == 1:
            value = values.pop()
            selected[field] = next(
                row for row in candidates if Decimal(row["kwota"]) == value
            )
    for field in target_fields:
        if field in selected:
            result[field] = selected[field]["kwota"]
    if "suma" in selected:
        result["kwota_nierozdzielona"] = selected["suma"]["kwota"]
    if set(AWARD_FIELDS) <= set(target_fields) and all(
        field in selected for field in AWARD_FIELDS
    ):
        total = sum(Decimal(result[field]) for field in AWARD_FIELDS)
        if "suma" in selected and total != Decimal(selected["suma"]["kwota"]):
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
    fields = config["fields"]
    threshold = config["system_one"]["minimum_probability"]
    heading = api.REASONS.search(text)
    header = text[: min(heading.start() if heading else len(text), 4000)]
    instance = await ask(
        server,
        header,
        {"instancja": {"type": "choice", **config["questions"]["instance"]}},
        config,
    )
    instance = instance["instancja"]
    # Etap 1: semantyczne oddzielenie opisu sprawy od rozważań i innych treści.
    if fields:
        selected, diagnostics = await classify_paragraphs(
            server, text, config, identifier
        )
    else:
        selected, diagnostics = feature_input(text, config), []
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
    # Krótkie porcje ograniczają liczbę konkurujących zdań. Extract wypełnia
    # schemat treścią; wybieramy zdanie najlepiej pasujące do danego pola.
    for batch, (source, mapping) in enumerate(
        extraction_batches(text, passages, config["chunk_characters"])
        if fields
        else [],
        1,
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
            # Rozwijamy span do zdania, więc ranking dotyczy wyboru zdania,
            # a nie długości konkretnego ciągu wyrazów wewnątrz tego zdania.
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
    # Wybrane zdanie sprawdzamy osobno: akapit mógł mieszać fakty i ocenę sądu.
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
    # Rozdzielone oryginalne teksty do ręcznego porównania. Offsety i wszystkie
    # decyzje, w tym stanowiska stron, pozostają w analysis.json.
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
