"""Basal fact selection, source-grounded descriptions and separate award extraction."""

import copy
import hashlib
import json
import math
import re
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import text as api
from .quality import target_rejection_reason
from .target_matching import validate_settings
from .text import sentence_fragments
from .storage import write_json

ROOT = Path(__file__).resolve().parents[3]
REQUEST_COUNTS = {"cached": 0, "computed": 0}
MODEL_CALLS = ContextVar("model_calls", default=None)
DECISION_CACHE = ContextVar("decision_cache", default=None)


MODEL_FILES = [
    "model*.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
    "basal.json",
    "CALIBRATION*.json",
]


DEFAULTS = {
    "version": "saos-extraction-18",
    "chunk_characters": 6000,
    "context_characters": 600,
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
            "wyroku i wskazanych poszkodowanych. Zwroty «wnosi o zasądzenie», "
            "«żąda» i «domaga się», lub podobne, opisują żądanie strony, nie zasądzenie przez "
            "sąd; dla takiej wzmianki wybierz pomin. Nie pomyl wcześniejszej "
            "wypłaty z zasądzeniem. Jeżeli sąd uznaje za odpowiednie "
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
            "instructions": "Czy sentencja i przytoczone dowody potwierdzają zasądzenie "
            "wymienionych składników w podanych kwotach dla wskazanego poszkodowanego "
            "w tym wyroku? Odróżnij zasądzenie od żądania, wcześniejszej wypłaty "
            "i kosztów postępowania. Nie oceniaj, czy kwota jest słuszna. "
            "Sprawdzaj tylko wskazane składniki; brak innego składnika nie oznacza "
            "błędu. Nie wymagaj zapisania w sentencji ich obliczonej sumy."
        },
        "extraction": {
            "instructions": "Wyodrębnij treść pól opisu sprawy zgodnie ze schematem. Zachowaj negacje, osobę "
            "i czas. Nie dopisuj ocen, nie używaj kwot ani uzasadnienia ich "
            "wysokości. Każde pole musi odpowiadać własnemu opisowi: leczenie nie "
            "opisuje okoliczności wypadku, a dane spółki nie opisują krzywdy osoby. "
            "Opisuj poszkodowanego, nie jego pełnomocnika ani innych osób. "
            "Nie wypełniaj pola zdaniem tylko dlatego, że jest dostępne. Jeśli brak "
            "informacji odpowiadającej temu polu, wybierz not stated.",
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
    "event_circumstances": "Przyczyna i przebieg zdarzenia, bez obrażeń i leczenia.",
    "harm_type": "Rodzaj krzywdy: uraz, choroba, utrata bliskiego lub naruszenie dóbr osobistych; bez leczenia.",
    "harm_severity_duration": "Nasilenie i czas trwania cierpienia, trwałość uszczerbku i rokowania.",
    "life_impact": "Ograniczenia codziennych czynności, pracy i relacji; nie samo leczenie.",
    "claimant_circumstances": "Wiek, zawód i zdrowie poszkodowanego przed zdarzeniem lub jego więź ze zmarłym.",
    "treatment_assistance": "Zastosowane lub zalecone leczenie, rehabilitacja i pomoc innych osób.",
}

METADATA_FIELDS = (
    "judgment_year",
    "instance",
    "court_seat",
    "court_locality_size",
    "injured_party_count",
    "prior_payment",
)


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
        raise ValueError(f"Configuration must be a mapping: {path}")
    return value


def load_config(directory=None, overrides=None):
    """
    Validate configuration and assemble effective extraction settings.

    args:
        directory: Input or output directory for this operation.
        overrides: Explicit acquisition overrides.

    returns:
        Effective validated configuration.
    """
    from ..get_data import search_settings

    directory = (
        Path(directory) if directory is not None else ROOT / "configs" / "dataset"
    )
    basal = read_config(directory / "basal.yaml")
    api = read_config(directory / "api.yaml")
    embedding = read_config(directory / "embedding.yaml")
    patterns = read_config(directory / "target_pattern.yaml")
    validate_settings(embedding, patterns)
    api.update(overrides or {})
    search_settings(api)
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
        raise ValueError(
            "basal.yaml requires device, feature/target selections and three probability thresholds"
        )
    revisions = {
        "Remek/basal-2.0-mini": "cb81b32ce89b0b87e62fcdad28f0435e25c953f3",
        "Remek/basal-2.0-4.5B": "e5ba9149760d99b73b1963d8e0ccf948b58afacb",
    }
    if not isinstance(basal["model"], str) or basal["model"] not in revisions:
        raise ValueError("model must be Remek/basal-2.0-mini or Remek/basal-2.0-4.5B")
    if basal["device"] not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
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
            raise ValueError(f"Invalid field names: {key}")
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
            raise ValueError(f"Invalid probability threshold: {key}")
    config = {
        **copy.deepcopy(DEFAULTS),
        "api": api,
        "embedding": embedding,
        "target_patterns": patterns,
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
        model=basal["model"], revision=revisions[basal["model"]]
    )
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


# Extract fills fields from source spans; choice and noul check inputs. Scores do not establish accuracy.
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
    """
    Select source text for factual passage classification.

    args:
        text: Normalized Polish source text.
        config: Effective pipeline or acquisition settings.

    returns:
        Selected source text, offsets and selection status.
    """
    reasons = list(api.REASONS.finditer(text))
    body_start = reasons[0].end() if len(reasons) == 1 else 0
    facts = re.search(
        r"^sąd(?:\s+\w+){0,3}\s+ustalił(?:\s+i\s+zważył)?(?:(?:,\s*|\s+)co następuje|\s+(?:następujący\s+)?stan faktyczny)?\s*[:.]?\s*$"
        r"|^(?:ustalenia faktyczne|stan faktyczny)\s*[:.]?\s*$",
        text[body_start:],
        re.IGNORECASE | re.MULTILINE,
    )
    start = body_start + facts.end() if facts else body_start
    reasoning = re.search(
        r"^sąd(?:\s+\w+){0,3}\s+zważył[^\n]*$",
        text[start:],
        re.IGNORECASE | re.MULTILINE,
    )
    end = start + reasoning.start() if reasoning else len(text)
    return {
        "text": text[start:end],
        "start": start,
        "end": end,
        "status": "court factual findings"
        if facts
        else "reasoning body; factual heading unavailable",
        "body_start": body_start,
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
        raise ValueError("The inference engine truncated its input")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Incomplete answer set")
    for key, question in questions.items():
        answer = answers[key]
        if question["type"] == "extract":
            expected = set(question["schema"]["properties"])
            if (
                not isinstance(state, str)
                or set(answer.get("data", {})) != expected
                or set(answer.get("fields", {})) != expected
            ):
                raise ValueError("Incomplete extraction record")
            for field, value in answer["data"].items():
                detail = answer["fields"][field]
                probability = detail.get("confidence")
                if probability is not None and (
                    type(probability) not in (int, float)
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError(f"Invalid field confidence: {field}")
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
                    raise ValueError(f"Field {field} does not match its source span")
            continue
        probabilities = answer.get("probabilities", {})
        if not probabilities or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()
        ):
            raise ValueError("Invalid probabilities")
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ValueError("Probabilities do not sum to one")
        expected = (
            set(question["criteria"])
            if question["type"] == "choice"
            else {"true", "false"}
        )
        if set(probabilities) != expected:
            raise ValueError("Answer options do not match the question")
        if question["type"] == "choice" and answer.get("choice") not in expected:
            raise ValueError("Selected option is outside the answer space")
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
            raise ValueError("Answer score does not match its probability distribution")
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
    path = ROOT / config["api"]["download_path"] / "debug" / "decisions" / f"{key}.json"
    progress = DECISION_CACHE.get()
    if progress is not None:
        path = progress / f"{key}.json"
    cache_enabled = config["api"]["debug"] or progress is not None
    if cache_enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
    if cache_enabled and path.exists():
        REQUEST_COUNTS["cached"] += 1
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["request"] != request:
            raise ValueError("Saved model decision does not match its request")
        response = saved["response"]
    else:
        REQUEST_COUNTS["computed"] += 1
        response = await server.serve(request)
        check_decision(response, questions, state)
        if cache_enabled:
            write_json(
                path,
                {
                    "request": request,
                    "response": response,
                    "model": config["system_one"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    if MODEL_CALLS.get() is not None:
        MODEL_CALLS.get().append({"request": request, "response": response})
    return check_decision(response, questions, state)


def source_blocks(text, limit):
    """
    Pack adjacent source sentences into bounded blocks with exact offsets.

    args:
        text: Normalized judgment text with paragraph boundaries.
        limit: Positive maximum block length in characters.

    returns:
        Iterator of source blocks; overlong sentences are split at spaces.
    """
    if limit <= 0:
        raise ValueError("Block character limit must be positive")
    start, end = None, None
    for paragraph in fragments(text, max(1, len(text))):
        for sentence in sentence_fragments(paragraph):
            for part in fragments(sentence["text"], limit):
                part_start = sentence["start"] + part["start"]
                part_end = sentence["start"] + part["end"]
                if start is not None and part_end - start > limit:
                    yield {"text": text[start:end], "start": start, "end": end}
                    start = None
                if start is None:
                    start = part_start
                end = part_end
    if start is not None:
        yield {"text": text[start:end], "start": start, "end": end}


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
    for row in diagnostics:
        answer, section = row["answers"], row["answers"]["sekcja"]
        checks = [
            (
                section["choice"] != "fakty",
                f"not factual (section={section['choice']})",
            ),
            (
                section["confidence"] < threshold,
                f"factual confidence={section['confidence']:.3f}, required >= {threshold}",
            ),
            (
                answer["wyciek"]["noul"] > maximum_leak,
                f"leakage={answer['wyciek']['noul']:.3f}, required <= {maximum_leak}",
            ),
            (
                bool(api.CURRENCY.search(row["text"])),
                "source sentence contains currency",
            ),
            (not any(char.isalpha() for char in row["text"]), "no descriptive text"),
            (
                row["text"].rstrip().endswith(":"),
                "heading instead of factual description",
            ),
        ]
        row["rejection_reasons"] = [reason for failed, reason in checks if failed]
        row["accepted"] = not row["rejection_reasons"]
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
        for paragraph in fragments(passage["text"], max(1, len(passage["text"]))):
            paragraph["start"] += passage["start"]
            paragraph["end"] += passage["start"]
            for sentence in sentence_fragments(paragraph):
                if text[sentence["start"] : sentence["end"]] != sentence["text"]:
                    raise ValueError("Feature passage does not match its source")
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
                "no description extracted",
            )
            continue
        rows = [
            row
            for row in mapping
            if row["input_start"] < detail["end"] and row["input_end"] > detail["start"]
        ]
        if not rows:
            raise ValueError(f"Cannot recover the source of field: {field}")
        # Keep the complete selected sentence rather than a phrase stripped of its negation.
        rows = rows[:2]
        values[field] = " ".join(row["text"] for row in rows)
        evidence[field] = [
            {key: row[key] for key in ("text", "start", "end")} for row in rows
        ]
        uncertain = confidence[field] is not None and confidence[field] < threshold
        statuses[field] = (
            "low-confidence description retained for review"
            if uncertain
            else "description extracted for review"
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


AWARD_FIELDS = ("zadoscuczynienie", "pozostale_odszkodowanie")


def amount_contexts(text):
    """
    Yield complete sentence contexts with each PLN occurrence marked.

    args:
        text: Normalized Polish source text.

    returns:
        Iterator of marked amount contexts.
    """
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
        result["status"] = "model did not confirm the proposal; review required"
    return result


async def extract_award(server, text, diagnostics, config):
    """
    Resolve monetary targets from embedding-selected sentences.

    args:
        server: Local Basal amount resolver.
        text: Full normalized judgment.
        diagnostics: Retained compatibility argument; not used for target selection.
        config: Effective pipeline settings.

    returns:
        Award values, beneficiary count and target-match diagnostics.
    """
    from .target_extraction import extract_targets

    return await extract_targets(server, text, config)


async def extract_prior_payment(server, match, config):
    """
    Resolve an earlier principal payment as an integer PLN feature, never a target.

    args:
        server: Local Basal extraction server.
        match: Embedding-selected prior-payment sentence, or None.
        config: Effective amount confidence threshold and model settings.

    returns:
        Confirmed prior principal rounded to whole PLN, or zero without confirmation.
    """
    from decimal import ROUND_HALF_UP, Decimal

    if match is None:
        return 0
    answer = (
        await ask(
            server,
            match["text"],
            {
                "prior_payment": {
                    "type": "extract",
                    "instructions": "Wskaż kwotę świadczenia głównego już wypłaconą poszkodowanym "
                    "przez pozwanego lub ubezpieczyciela przed analizowanym wyrokiem. "
                    "Nie wybieraj kwoty żądanej, zasądzonej, pozostałej do zapłaty, odsetek "
                    "ani kosztów postępowania. Wskaż wyłącznie liczbową kwotę wraz z walutą "
                    "jako dokładny fragment źródła. Jeśli podano składniki i ich sumę, wybierz "
                    "tylko sumę. Gdy jest kilka odrębnych wypłat bez jednoznacznej sumy "
                    "lub inny odbiorca niż poszkodowani w tej sprawie, wybierz not stated.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "amount": {
                                "type": "string",
                                "description": "Wcześniej wypłacona kwota główna z walutą PLN.",
                            }
                        },
                    },
                }
            },
            config,
        )
    )["prior_payment"]
    value = answer["data"]["amount"]
    probability = answer["fields"]["amount"].get("confidence")
    if (
        value is None
        or probability is None
        or probability < config["system_one"]["amount_minimum_probability"]
    ):
        return 0
    amount = api.MONEY.fullmatch(value.strip())
    if amount is None:
        return 0
    return int(
        Decimal(api.amount_in_pln(amount)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    )


async def extract_features(server, text, config, identifier):
    """
    Extract only from factual findings and review field relevance of final candidates.

    args:
        server: Local inference server.
        text: Full normalized judgment.
        config: Feature schema and admission thresholds.
        identifier: Stable judgment identifier for progress output.

    returns:
        Features, source evidence and model diagnostics with deliberate missing values.
    """
    fields = config["fields"]
    threshold = config["system_one"]["minimum_probability"]
    selected = feature_input(text, config)
    descriptions = {field: "brak informacji" for field in fields}
    evidence = {field: [] for field in fields}
    statuses = {field: "no description extracted" for field in fields}
    confidence = {field: None for field in fields}
    native, reviews = [], {}
    passages = [
        {
            **block,
            "start": block["start"] + selected["start"],
            "end": block["end"] + selected["start"],
        }
        for block in source_blocks(selected["text"], config["chunk_characters"])
        if fields
    ]
    candidates = {field: [] for field in fields}
    schema = {
        "type": "object",
        "properties": {
            field: {
                "type": "string",
                "description": description
                + config["questions"]["extraction"]["field_suffix"],
            }
            for field, description in fields.items()
        },
    }
    for batch, (source, mapping) in enumerate(
        extraction_batches(text, passages, config["chunk_characters"]), 1
    ):
        print(
            f"ID {identifier}: extracting facts, batch {batch} ({len(source)} characters)",
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
        values, quotes, field_status, probabilities = describe_record(
            answer, mapping, fields, threshold
        )
        for field in fields:
            if quotes[field] and api.CURRENCY.search(values[field]):
                statuses[field] = (
                    "extracted description rejected: source sentence contains currency"
                )
            if quotes[field] and not api.CURRENCY.search(values[field]):
                candidates[field].append(
                    (
                        answer["fields"][field].get("line_probability", 0),
                        values[field],
                        quotes[field],
                        field_status[field],
                        probabilities[field],
                    )
                )
    span_fields = {}
    for field, options in candidates.items():
        for _, value, quotes, _, _ in options:
            span_fields.setdefault((quotes[0]["start"], quotes[-1]["end"]), set()).add(
                field
            )
    for field, options in candidates.items():
        for _, value, quotes, status, probability in sorted(
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
                questions = {
                    "sekcja": {"type": "choice", **config["questions"]["section"]},
                    "wyciek": {"type": "noul", **config["questions"]["leakage"]},
                    **{
                        f"fit_{name}": {
                            "type": "noul",
                            "instructions": "Czy wyłącznie oceniany fragment zawiera konkretną informację "
                            "odpowiadającą temu polu: "
                            + fields[name]
                            + "? Sama prawdziwość zdania nie wystarcza. Nie dopowiadaj informacji "
                            "z sąsiedniego kontekstu. Jeśli fragment dotyczy innej kwestii, odpowiedz nie.",
                        }
                        for name in sorted(span_fields[start, end])
                    },
                }
                answers = await ask(server, state, questions, config)
                row = {
                    "text": value,
                    "start": start,
                    "end": end,
                    "answers": answers,
                }
                select_facts(
                    [row],
                    threshold,
                    config["system_one"]["maximum_leak_probability"],
                )
                reviews[start, end] = row
            review = reviews[start, end]
            if (
                review["accepted"]
                and review["answers"][f"fit_{field}"]["noul"] >= threshold
            ):
                descriptions[field], evidence[field] = value, quotes
                statuses[field], confidence[field] = status, probability
                break
            if not statuses[field].startswith("best candidate rejected:"):
                reasons = list(review["rejection_reasons"])
                relevance = review["answers"][f"fit_{field}"]["noul"]
                if relevance < threshold:
                    reasons.append(
                        f"field relevance={relevance:.3f}, required >= {threshold}"
                    )
                statuses[field] = "best candidate rejected: " + "; ".join(reasons)
    for field in fields:
        status = "accepted" if evidence[field] else statuses[field]
        print(f"ID {identifier}: feature {field}: {status}", flush=True)
    return {
        "features": descriptions,
        "evidence": evidence,
        "field_status": statuses,
        "confidence": confidence,
        "native_extraction": native,
        "diagnostics": list(reviews.values()),
        "description_reviews": list(reviews.values()),
        "fields_evaluated": bool(native),
        "feature_input": {
            key: value for key, value in selected.items() if key != "text"
        },
        "input_characters": {
            "full": len(text),
            "classified": sum(len(row["text"]) for row in reviews.values()),
            "features": sum(
                len(value) for field, value in descriptions.items() if evidence[field]
            ),
        },
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
        Feature and target results, or only id and award when the target is rejected.
    """
    print(f"ID {identifier}: extracting target", flush=True)
    award = await extract_award(server, text, [], config)
    if (
        not config["api"]["debug"]
        and target_rejection_reason(award, config) is not None
    ):
        return {"id": identifier, "award": award}
    prior_features = {}
    if "prior_payment" in config["metadata_fields"]:
        prior_features["prior_payment"] = await extract_prior_payment(
            server, award.get("target_matches", {}).get("prior_payment"), config
        )
    if "target_matches" in award and award["target_matches"]["judgment"] is None:
        return {
            "id": identifier,
            "award": award,
            "fields_evaluated": False,
            **prior_features,
            "features": {field: "brak informacji" for field in config["fields"]},
            "evidence": {field: [] for field in config["fields"]},
            "confidence": {field: None for field in config["fields"]},
            "field_status": {
                field: "no judgment sentence above the similarity threshold"
                for field in config["fields"]
            },
            "instance": {"value": "brak informacji", "decision": {"confidence": 0.0}},
        }
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
    features = await extract_features(server, text, config, identifier)
    return {
        "id": identifier,
        "award": award,
        **prior_features,
        **features,
        "instance": {
            "value": instance["choice"]
            if instance["choice"] != "nieustalona"
            and instance["confidence"] >= threshold
            else "brak informacji",
            "decision": instance,
            "source": header,
            "status": "review required",
        },
        "status": "candidates for review; not approved for training",
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
