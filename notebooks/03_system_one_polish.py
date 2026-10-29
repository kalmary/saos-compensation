"""Lokalna próba polskiego modelu decyzyjnego: uv run --extra extraction notebooks/03_system_one_polish.py."""

import asyncio
import hashlib
import importlib
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "tmp" / "huggingface"))
REQUEST_COUNTS = {"cached": 0, "computed": 0}


def read_config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Konfiguracja musi być mapą: {path}")
    return value


def load_config(path=None):
    path = Path(path) if path is not None else Path(__file__).with_name("preprocessing.yaml")
    settings = read_config(path)
    basal = read_config(path.parent / settings["basal"])
    api = read_config(path.parent / settings["api"])
    config = {**settings, **basal, "api": api, "judgment_ids": api["judgment_ids"]}
    if not isinstance(config, dict) or not isinstance(config.get("fields"), dict) or not config["fields"]:
        raise ValueError("Konfiguracja wymaga niepustej listy pól")
    if not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in config["fields"].items()):
        raise ValueError("Pola muszą mieć nazwy i pytania tekstowe")
    if not 0 < config["system_one"]["minimum_probability"] < 1:
        raise ValueError("Próg decyzji musi należeć do (0, 1)")
    if not 0 <= config["system_one"]["maximum_leak_probability"] < 0.5:
        raise ValueError("Nieprawidłowy próg ryzyka wycieku")
    if not 0 < config["system_one"]["amount_minimum_probability"] < 1:
        raise ValueError("Nieprawidłowy próg potwierdzenia zasądzenia")
    if type(config["chunk_characters"]) is not int or config["chunk_characters"] < 100:
        raise ValueError("Nieprawidłowy rozmiar fragmentu")
    if config["feature_source"] not in {"semantic_paragraphs", "full_document", "factual_section"}:
        raise ValueError("Nieznany zakres tekstu dla cech")
    if type(config["context_characters"]) is not int or config["context_characters"] < 0:
        raise ValueError("Nieprawidłowy rozmiar sąsiedniego kontekstu")
    if not isinstance(config.get("questions"), dict):
        raise ValueError("Brak pytań modelu")
    for name in ("section", "leakage", "amount", "award_check", "extraction", "instance"):
        question = config["questions"].get(name)
        if not isinstance(question, dict) or not isinstance(question.get("instructions"), str) or not question["instructions"].strip():
            raise ValueError(f"Brak instrukcji: {name}")
        if name in {"section", "amount", "instance"}:
            criteria = question.get("criteria")
            if not isinstance(criteria, dict) or not criteria or not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in criteria.items()):
                raise ValueError(f"Nieprawidłowe kategorie: {name}")
    if not isinstance(config["language"], str) or not config["language"].strip():
        raise ValueError("Brak języka pytań")
    if not isinstance(config["questions"]["extraction"].get("field_suffix"), str):
        raise ValueError("Brak instrukcji opisu pola")
    for name in ("model", "revision", "mode", "device", "template_prefix", "context_overflow"):
        if not isinstance(config["system_one"].get(name), str) or not config["system_one"][name].strip():
            raise ValueError(f"Nieprawidłowy parametr modelu: {name}")
    return config


# %% Model semantycznie ocenia fragmenty; nie przycinamy kategorii słownikami urazów.
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
    # Wariant domyślny nie wymaga nagłówków: o roli akapitów decyduje model.
    # Wycięcie po nagłówkach pozostaje wyłącznie wariantem porównawczym.
    if config["feature_source"] == "semantic_paragraphs":
        return {"text": text, "start": 0, "end": len(text), "status": "semantyczny podział wszystkich akapitów z kontekstem"}
    if config["feature_source"] == "full_document":
        return {"text": text, "start": 0, "end": len(text), "status": "pełny dokument — próba porównawcza"}
    api = importlib.import_module("01_saos_feasibility")
    facts = api.split_sections(text)["fakty"]
    if not facts or text.count(facts) != 1:
        return {"text": "", "start": None, "end": None, "status": "brak jednoznacznych granic faktów"}
    start = text.index(facts)
    return {"text": facts, "start": start, "end": start + len(facts),
            "status": "między nagłówkami ustaleń faktycznych i rozważań; do przeglądu"}


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
            if not isinstance(state, str) or set(answer.get("data", {})) != expected or set(answer.get("fields", {})) != expected:
                raise ValueError("Niekompletny rekord ekstrakcji")
            for field, value in answer["data"].items():
                detail = answer["fields"][field]
                probability = detail.get("confidence")
                if probability is not None and (type(probability) not in (int, float) or not math.isfinite(probability) or not 0 <= probability <= 1):
                    raise ValueError(f"Nieprawidłowa pewność pola {field}")
                if value is None:
                    continue
                start, end = detail.get("start"), detail.get("end")
                if not isinstance(value, str) or not value.strip() or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(state) or state[start:end] != value:
                    raise ValueError(f"Pole {field} nie odpowiada fragmentowi źródłowemu")
            continue
        probabilities = answer.get("probabilities", {})
        if not probabilities or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError("Nieprawidłowe prawdopodobieństwa")
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ValueError("Rozkład nie sumuje się do 1")
        expected = set(question["criteria"]) if question["type"] == "choice" else {"true", "false"}
        if set(probabilities) != expected:
            raise ValueError("Odpowiedź zawiera inne opcje niż pytanie")
        if question["type"] == "choice" and answer.get("choice") not in expected:
            raise ValueError("Wybrana opcja nie należy do przestrzeni odpowiedzi")
        selected = answer["choice"] if question["type"] == "choice" else "true"
        probability = answer.get("confidence") if question["type"] == "choice" else answer.get("noul")
        if type(probability) not in (int, float) or not math.isfinite(probability) or abs(probability - probabilities[selected]) > 0.01:
            raise ValueError("Wynik nie jest zgodny z rozkładem")
    return answers


async def ask(server, state, questions, config):
    request = {"state": state, "lang": config["language"], "questions": questions}
    # Progi selekcji nie zmieniają odpowiedzi modelu, więc nie unieważniają cache.
    identity = {"version": config["version"], "engine_version": "2.4.1",
                "model": {key: value for key, value in config["system_one"].items()
                          if key not in {"minimum_probability", "maximum_leak_probability", "amount_minimum_probability"}}}
    key = hashlib.sha256(json.dumps([identity, request], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    path = ROOT / "tmp" / "system_one" / "decisions" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        REQUEST_COUNTS["cached"] += 1
        response = json.loads(path.read_text(encoding="utf-8"))["response"]
    else:
        REQUEST_COUNTS["computed"] += 1
        response = await server.serve(request)
        check_decision(response, questions, state)
        path.write_text(json.dumps({"request": request, "response": response,
            "model": config["system_one"], "created_at": datetime.now(timezone.utc).isoformat()},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return check_decision(response, questions, state)


def choices(question, criteria):
    return {"type": "choice", "instructions": question, "criteria": criteria}


def sentence_fragments(chunk):
    # Podział techniczny akapitu mieszanego. Reguła nie przypisuje roli zdania.
    # Inicjały osób i liczby z groszami nie powinny wyznaczać granicy.
    boundaries = [0, *[m.end() for m in re.finditer(
        r"(?<!\b[A-ZĄĆĘŁŃÓŚŹŻ]\.)(?<!\br\.)(?<=[.!?])\s+(?=[A-ZĄĆĘŁŃÓŚŹŻ])", chunk["text"])], len(chunk["text"])]
    for start, end in zip(boundaries, boundaries[1:]):
        while start < end and chunk["text"][start].isspace():
            start += 1
        while end > start and chunk["text"][end - 1].isspace():
            end -= 1
        if end > start:
            yield {"text": chunk["text"][start:end], "start": chunk["start"] + start, "end": chunk["start"] + end}


async def classify_paragraphs(server, text, config, identifier):
    selected = feature_input(text, config)
    chunks = [{**chunk, "start": chunk["start"] + selected["start"], "end": chunk["end"] + selected["start"]}
              for chunk in fragments(selected["text"], config["chunk_characters"])]
    questions = {"sekcja": {"type": "choice", **config["questions"]["section"]},
                 "wyciek": {"type": "noul", **config["questions"]["leakage"]}}
    diagnostics = []

    async def classify(chunk, unit, parent=None):
        context = config["context_characters"]
        state = {"oceniany_fragment": chunk["text"],
                 "poprzedzajacy_kontekst": text[max(0, chunk["start"] - context):chunk["start"]],
                 "nastepujacy_kontekst": text[chunk["end"]:chunk["end"] + context]}
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
        if decision["choice"] == "mieszany" and decision["confidence"] >= config["system_one"]["minimum_probability"]:
            sentences = list(sentence_fragments(chunk))
            if len(sentences) > 1:
                for sentence in sentences:
                    await classify(sentence, "sentence", chunk)
    return selected, diagnostics


async def evaluate(server, config):
    # Niewielka jawna próba kontrolna. To diagnostyka, nie benchmark SAOS.
    cases = [
        ("Powódka złamała nogę i przez sześć tygodni poruszała się o kulach.", "fakty", False),
        ("Sąd ustalił: poszkodowany przed wypadkiem pracował jako kierowca.", "fakty", False),
        ("Powód żądał 80 000 zł, lecz sąd oddalił jego powództwo w całości.", None, True),
        ("Sąd uwzględnił żądanie w całości, uznając świadczenie za odpowiednie do krzywdy.", "rozwazania", True),
        ("Zasądza od pozwanego na rzecz powódki 17 176 zł wraz z odsetkami.", "rozstrzygniecie", True),
        ("Powódka twierdziła, że wymaga stałej opieki, czemu pozwany zaprzeczył. Sąd nie rozstrzygnął tej kwestii.", "stanowiska", False),
        ("Sąd ustalił, że powódka nie wymagała rehabilitacji ani pomocy innych osób.", "fakty", False),
    ]
    questions = {"sekcja": {"type": "choice", **config["questions"]["section"]},
                 "wyciek": {"type": "noul", **config["questions"]["leakage"]}}
    rows = []
    for text, expected, leak in cases:
        answer = await ask(server, text, questions, config)
        rows.append({"text": text, "expected_section": expected, "expected_leak": leak,
                     "answers": answer, "correct_section": expected is None or answer["sekcja"]["choice"] == expected,
                     "correct_leak": (answer["wyciek"]["noul"] >= 0.5) == leak})
    return rows


def select_facts(diagnostics, threshold, maximum_leak):
    api = importlib.import_module("01_saos_feasibility")
    for row in diagnostics:
        answer, section = row["answers"], row["answers"]["sekcja"]
        row["accepted"] = (section["choice"] == "fakty" and section["confidence"] >= threshold
                           and answer["wyciek"]["noul"] <= maximum_leak and not api.CURRENCY.search(row["text"])
                           and any(char.isalpha() for char in row["text"]) and not row["text"].rstrip().endswith(":"))
    return [row for row in diagnostics if row["accepted"]]


def extraction_source(text, passages):
    # Każda linia wejścia extract jest zdaniem z dokładnym miejscem w wyroku.
    # Ułatwia to krótki opis i zachowuje negację oraz podmiot wypowiedzi.
    sentences = {}
    for passage in passages:
        for sentence in sentence_fragments(passage):
            if text[sentence["start"]:sentence["end"]] != sentence["text"]:
                raise ValueError("Niezgodny fragment opisu sprawy")
            sentences[(sentence["start"], sentence["end"])] = sentence
    parts, mapping, position = [], [], 0
    for sentence in sorted(sentences.values(), key=lambda row: row["start"]):
        parts.append(sentence["text"])
        mapping.append({**sentence, "input_start": position, "input_end": position + len(sentence["text"])})
        position += len(sentence["text"]) + 1
    return "\n".join(parts), mapping


def describe_record(answer, mapping, fields, threshold):
    values, evidence, statuses, confidence = {}, {}, {}, {}
    for field in fields:
        detail = answer["fields"][field]
        value = answer["data"][field]
        confidence[field] = detail.get("confidence", detail.get("present_probability"))
        if value is None:
            values[field], evidence[field], statuses[field] = "brak informacji", [], "model nie wyekstrahował opisu"
            continue
        rows = [row for row in mapping if row["input_start"] < detail["end"] and row["input_end"] > detail["start"]]
        if not rows:
            raise ValueError(f"Nie można odtworzyć źródła pola {field}")
        # Do opisu włączamy całe wybrane zdanie, nie wyrwany wyraz bez negacji.
        rows = rows[:2]
        values[field] = " ".join(row["text"] for row in rows)
        evidence[field] = [{key: row[key] for key in ("text", "start", "end")} for row in rows]
        uncertain = confidence[field] is not None and confidence[field] < threshold
        statuses[field] = "opis o niskiej pewności — zachowany do przeglądu" if uncertain else "wyekstrahowano opis do przeglądu"
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


# %% Target: liczby z otoczeniem słownym; znaczenie przypisuje model.
AWARD_FIELDS = ("zadoscuczynienie", "pozostale_odszkodowanie")
def amount_contexts(text):
    api = importlib.import_module("01_saos_feasibility")
    heading = list(api.REASONS.finditer(text))
    boundary = heading[0].start() if len(heading) == 1 else None
    # Całe zdanie lub punkt listy, bez okna słów i bez regexów znaczenia świadczenia.
    for paragraph in fragments(text, max(1, len(text))):
        for sentence in sentence_fragments(paragraph):
            for match in api.MONEY.finditer(sentence["text"]):
                start, end = sentence["start"] + match.start(), sentence["start"] + match.end()
                marked = sentence["text"][:match.start()] + "<KWOTA>" + match[0] + "</KWOTA>" + sentence["text"][match.end():]
                yield {**sentence,
                       "kwota": str(api.Decimal(match["amount"].replace(" ", "").replace(".", "").replace(",", "."))),
                       "amount_start": start, "amount_end": end, "marked": marked,
                       "section": "sentencja" if boundary is not None and start < boundary else
                                  "uzasadnienie" if boundary is not None else "nieustalona"}


def award_state(text, row, context_characters=600):
    api = importlib.import_module("01_saos_feasibility")
    heading = list(api.REASONS.finditer(text))
    ruling = text[:heading[0].start()] if len(heading) == 1 else "brak jednoznacznej granicy sentencji"
    return {"fragment": row["marked"], "sekcja": row["section"], "sentencja": ruling,
            "poprzedzajacy_kontekst": text[max(0, row["start"] - context_characters):row["start"]]}


def propose_award(text, decisions):
    result = {field: "brak informacji" for field in (*AWARD_FIELDS, "suma", "kwota_nierozdzielona")}
    result.update(beneficjent="brak informacji", waluta="PLN", evidence=[], status="brak ustalonego zasądzenia")
    api = importlib.import_module("01_saos_feasibility")
    heading = list(api.REASONS.finditer(text))
    if len(heading) != 1:
        result["status"] = "brak jednoznacznej granicy sentencji — do przeglądu"
        return result
    # To metadane odbiorcy, nie reguła znaczenia kwoty. Nie dzielimy wspólnego zasądzenia między osoby.
    plaintiff = re.search(r"z powództwa\s+([^\n]+)", text[:heading[0].start()], re.I)
    if not plaintiff or re.search(r",|\boraz\b|\s+i\s+", plaintiff[1]):
        result["status"] = "nieustalony lub wieloosobowy beneficjent — do przeglądu"
        return result
    result["beneficjent"] = plaintiff[1].strip()
    selected = {}
    for field in ("suma", *AWARD_FIELDS):
        rows = [row for row in decisions if row["accepted"] and row["category"] == field]
        direct = [row for row in rows if row["section"] == "sentencja"]
        # Nie sumujemy wystąpień. Konflikt kilku wartości w jednej kategorii wymaga przeglądu.
        # Suma zasądzona musi mieć dowód w sentencji, nie w opisie żądań.
        candidates = direct if field == "suma" else direct or rows
        if field in AWARD_FIELDS and not direct and "suma" in selected:
            total = api.Decimal(selected["suma"]["kwota"])
            # Najpierw jawnie przypisany składnik równy całemu zasądzeniu.
            # Nie podstawiamy kwoty za model: wybieramy spośród jego przypisań.
            exact = [row for row in candidates if api.Decimal(row["kwota"]) == total]
            candidates = exact or [row for row in candidates if api.Decimal(row["kwota"]) < total]
        values = {api.Decimal(row["kwota"]) for row in candidates}
        if len(values) == 1:
            value = values.pop()
            selected[field] = next(row for row in candidates if api.Decimal(row["kwota"]) == value)
    for field in AWARD_FIELDS:
        if field in selected:
            result[field] = selected[field]["kwota"]
    if "suma" in selected:
        result["kwota_nierozdzielona"] = selected["suma"]["kwota"]
    if all(field in selected for field in AWARD_FIELDS):
        total = sum(api.Decimal(result[field]) for field in AWARD_FIELDS)
        if "suma" in selected and total != api.Decimal(selected["suma"]["kwota"]):
            result["status"] = "składniki nie zgadzają się z sumą — do przeglądu"
            result["conflict"] = {field: result[field] for field in AWARD_FIELDS}
            result.update({field: "brak informacji" for field in AWARD_FIELDS})
        else:
            result.update(suma=str(total), kwota_nierozdzielona="brak informacji")
    result["evidence"] = [{key: row[key] for key in ("text", "start", "end", "section", "amount_start", "amount_end")}
                          for row in selected.values()]
    if result["status"] != "składniki nie zgadzają się z sumą — do przeglądu" and selected:
        result["status"] = "propozycja z kontekstów kwot — do kontroli"
    return result


def finish_award(proposal, accepted, decision):
    result = {**proposal, "verification": decision, "confirmed": accepted}
    if not accepted:
        result["proposal"] = proposal
        result.update({field: "brak informacji" for field in (*AWARD_FIELDS, "suma", "kwota_nierozdzielona")})
        result["status"] = "model nie potwierdził propozycji — do przeglądu"
    return result


async def extract_award(server, text, diagnostics, config):
    decisions = []
    threshold = config["system_one"]["amount_minimum_probability"]
    for row in amount_contexts(text):
        answer = (await ask(server, award_state(text, row, config.get("context_characters", 600)), {"swiadczenie": {"type": "choice", **config["questions"]["amount"]}}, config))["swiadczenie"]
        decisions.append({**row, "category": answer["choice"], "confidence": answer["confidence"],
                          "accepted": answer["confidence"] >= threshold and answer["choice"] in {*AWARD_FIELDS, "suma"},
                          "decision": answer})
    proposal = propose_award(text, decisions)
    if not proposal["evidence"] or "conflict" in proposal:
        return {**proposal, "confirmed": False, "amount_decisions": decisions}
    state = {"propozycja": {key: value for key, value in proposal.items() if key != "evidence"},
             "sentencja": award_state(text, decisions[0])["sentencja"], "dowody": proposal["evidence"]}
    answer = (await ask(server, state, {"zasadzenie": {"type": "noul", **config["questions"]["award_check"]}}, config))["zasadzenie"]
    return {**finish_award(proposal, answer["noul"] >= threshold, answer), "amount_decisions": decisions}


async def extract(server, text, config, identifier):
    fields = config["fields"]
    threshold = config["system_one"]["minimum_probability"]
    api = importlib.import_module("01_saos_feasibility")
    heading = api.REASONS.search(text)
    header = text[:min(heading.start() if heading else len(text), 4000)]
    instance = await ask(server, header, {"instancja": {"type": "choice", **config["questions"]["instance"]}}, config)
    instance = instance["instancja"]
    # Etap 1: semantyczne oddzielenie opisu sprawy od rozważań i innych treści.
    selected, diagnostics = await classify_paragraphs(server, text, config, identifier)
    passages = select_facts(diagnostics, threshold, config["system_one"]["maximum_leak_probability"])
    descriptions = {field: "brak informacji" for field in fields}
    evidence = {field: [] for field in fields}
    field_status = {field: "brak wyodrębnionego opisu sprawy" if not passages else "model nie wyekstrahował opisu" for field in fields}
    confidence = {field: None for field in fields}
    native, candidates = [], {field: [] for field in fields}
    schema = {"type": "object", "properties": {
        field: {"type": "string", "description": question + config["questions"]["extraction"]["field_suffix"]}
        for field, question in fields.items()}}
    # Krótkie porcje ograniczają liczbę konkurujących zdań. Extract wypełnia
    # schemat treścią; wybieramy zdanie najlepiej pasujące do danego pola.
    for batch, (source, mapping) in enumerate(extraction_batches(text, passages, config["chunk_characters"]), 1):
        print(f"ID {identifier}: ekstrakcja opisu, porcja {batch} ({len(source)} znaków)", flush=True)
        answer = (await ask(server, source, {"opis": {"type": "extract", "schema": schema,
            "instructions": config["questions"]["extraction"]["instructions"]}}, config))["opis"]
        native.append({"source": source, "mapping": mapping, "answer": answer})
        values, quotes, statuses, probabilities = describe_record(answer, mapping, fields, threshold)
        for field in fields:
            # Rozwijamy span do zdania, więc ranking dotyczy wyboru zdania,
            # a nie długości konkretnego ciągu wyrazów wewnątrz tego zdania.
            score = answer["fields"][field].get("line_probability", 0)
            if quotes[field]:
                candidates[field].append((score, values[field], quotes[field], statuses[field], probabilities[field]))
    # Wybrane zdanie sprawdzamy osobno: akapit mógł mieszać fakty i ocenę sądu.
    reviews = {}
    for field, options in candidates.items():
        for score, value, quotes, status, probability in sorted(options, key=lambda row: row[0], reverse=True):
            start, end = quotes[0]["start"], quotes[-1]["end"]
            if (start, end) not in reviews:
                context = config["context_characters"]
                state = {"oceniany_fragment": value,
                         "poprzedzajacy_kontekst": text[max(0, start - context):start],
                         "nastepujacy_kontekst": text[end:end + context]}
                answers = await ask(server, state, {"sekcja": {"type": "choice", **config["questions"]["section"]},
                    "wyciek": {"type": "noul", **config["questions"]["leakage"]}}, config)
                row = {"text": value, "start": start, "end": end, "answers": answers}
                select_facts([row], threshold, config["system_one"]["maximum_leak_probability"])
                reviews[start, end] = row
            if reviews[start, end]["accepted"]:
                descriptions[field], evidence[field] = value, quotes
                field_status[field], confidence[field] = status, probability
                break
            field_status[field] = "odrzucone źródło opisu — poza faktami lub sygnał wycieku"
    award = await extract_award(server, text, diagnostics, config)
    return {"id": identifier, "features": descriptions,
            "evidence": evidence, "award": award, "diagnostics": diagnostics,
            "field_status": field_status, "fields_evaluated": bool(passages), "confidence": confidence,
            "native_extraction": native,
            "description_reviews": list(reviews.values()),
            "paragraph_counts": dict(Counter(row["answers"]["sekcja"]["choice"] for row in diagnostics if row["unit"] == "paragraph")),
            "instance": {"value": instance["choice"] if instance["choice"] != "nieustalona" and instance["confidence"] >= threshold else "brak informacji",
                         "decision": instance, "source": header, "status": "do weryfikacji"},
            "feature_input": {key: value for key, value in selected.items() if key != "text"},
            "input_characters": {"full": len(text), "classified": len(selected["text"]),
                                 "features": sum(len(row["text"]) for row in diagnostics if row["accepted"])},
            "status": "kandydaci do weryfikacji; nie dopuszczono do treningu",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def write_paragraphs(result, directory, threshold):
    # Rozdzielone oryginalne teksty do ręcznego porównania. Offsety i wszystkie
    # decyzje, w tym stanowiska stron, pozostają w pełnym results.json.
    directory.mkdir(parents=True, exist_ok=True)
    if "award" in result:
        (directory / "award.json").write_text(json.dumps(result["award"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "record.json").write_text(json.dumps(result["features"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    diagnostics = result["diagnostics"]
    groups = {
        "facts.txt": [row for row in diagnostics if row["accepted"]],
        "reasoning.txt": [row for row in diagnostics if row["answers"]["sekcja"]["choice"] == "rozwazania"
                          and row["answers"]["sekcja"]["confidence"] >= threshold],
        "review.txt": [row for row in diagnostics if row["answers"]["sekcja"]["choice"] in {"mieszany", "niejednoznaczny"}
                       or row["answers"]["sekcja"]["confidence"] < threshold
                       or (row["answers"]["sekcja"]["choice"] == "fakty" and not row["accepted"])],
    }
    for filename, rows in groups.items():
        (directory / filename).write_text("\n\n".join(row["text"] for row in rows) + "\n", encoding="utf-8")


async def run(texts=None, config=None):
    # Silnik i wagi są przypięte; opisy to wybrane zdania źródła.
    # Pierwsze uruchomienie pobiera wagi lokalnie.
    from basal.server import Server, parser
    from huggingface_hub import snapshot_download

    config = config or load_config()
    REQUEST_COUNTS.update(cached=0, computed=0)
    model = config["system_one"]
    print("Progi: " + ", ".join(f"{key}={model[key]}" for key in (
        "minimum_probability", "maximum_leak_probability", "amount_minimum_probability")), flush=True)
    # Repozytorium 4.5B zawiera też osobne wagi trybu think; ich nie używamy.
    model_path = snapshot_download(model["model"], revision=model["revision"], ignore_patterns=["think/*"])
    args = parser().parse_args([
        "--model", model_path,
        "--mode", model["mode"], "--device", model["device"],
        "--template-prefix", model["template_prefix"],
        "--name", model["model"],
        "--context-overflow", model["context_overflow"],
    ])
    server = Server(args)
    worker = asyncio.create_task(server.worker())
    try:
        started = time.perf_counter()
        evaluation = await evaluate(server, config)
        if texts is None:
            api = importlib.import_module("01_saos_feasibility")
            api.OUTPUT.mkdir(parents=True, exist_ok=True)
            texts = {identifier: api.extract_text(api.fetch_json(f"{api.BASE_URL}/judgments/{identifier}")["data"]["textContent"])
                     for identifier in config["judgment_ids"]}
        results = {identifier: await extract(server, text, config, identifier) for identifier, text in texts.items()}
        output = {"config": config, "engine_version": "2.4.1", "engine_arguments": vars(args),
                  "created_at": datetime.now(timezone.utc).isoformat(), "requests": REQUEST_COUNTS.copy(),
                  "evaluation": evaluation, "judgments": results,
                  "seconds": round(time.perf_counter() - started, 2)}
        path = ROOT / "tmp" / "system_one" / "results.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for identifier, result in results.items():
            write_paragraphs(result, path.parent / str(identifier), config["system_one"]["minimum_probability"])
        for identifier, result in results.items():
            print(f"\nID {identifier}")
            print(json.dumps(result["features"], ensure_ascii=False, indent=2))
            print("Zasądzone świadczenie:", json.dumps({k: v for k, v in result["award"].items() if k not in ("amount_decisions", "proposal")}, ensure_ascii=False, indent=2))
            print(f"Zakres wejścia: {result['input_characters']}; {result['feature_input']['status']}")
            print(f"Role akapitów: {result['paragraph_counts']}")
        print(f"Kontrola sekcji: {sum(r['correct_section'] for r in evaluation)}/{len(evaluation)}; "
              f"kontrola wycieku: {sum(r['correct_leak'] for r in evaluation)}/{len(evaluation)}")
        print(f"Czas analizy: {output['seconds']} s; żądania: {output['requests']}")
        print(f"Pełne odpowiedzi i dowody: {path}")
        return output
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(run())
