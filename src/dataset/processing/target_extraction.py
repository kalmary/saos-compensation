"""Resolve amounts only after embedding selection has located an award sentence."""

import re
from decimal import Decimal

from .target_matching import match_targets
from .text import MONEY, REASONS, amount_in_pln


def count_people(value):
    """
    Count an explicitly extracted beneficiary list without guessing from plural grammar.

    args:
        value: Source span containing beneficiaries or their explicit count.

    returns:
        Positive integer, or None when the source does not establish a count.
    """
    if not value:
        return None
    explicit = re.search(r"\b(\d+)\s+(?:osób|poszkodowan|powod)", value, re.IGNORECASE)
    if explicit:
        return int(explicit[1]) or None
    words = {
        "jedna": 1,
        "jednej": 1,
        "jednego": 1,
        "dwie": 2,
        "dwóch": 2,
        "dwojga": 2,
        "trzech": 3,
        "trojga": 3,
        "czterech": 4,
        "czworga": 4,
        "pięciu": 5,
    }
    for word, count in words.items():
        if re.search(rf"\b{word}\b", value, re.IGNORECASE):
            return count
    if re.search(
        r"reprezentowan|pełnomocnik|opiekun|spółk|fundacj", value, re.IGNORECASE
    ):
        return None
    people = re.split(r"\s*(?:,|\boraz\b|\bi\b)\s*", value)
    if any(not person.strip() for person in people):
        return None
    if len(people) > 1:
        return (
            len(people) if all(count_people(person) == 1 for person in people) else None
        )
    if re.search(
        r"\b(?:powodów|powodowie|powodom|poszkodowanych|poszkodowanym|osób|osobom|osoby)\b",
        value,
        re.IGNORECASE,
    ):
        return None
    if re.search(
        r"\b(?:powód|powoda|powódka|powódki|poszkodowany|poszkodowana)\b",
        value,
        re.IGNORECASE,
    ):
        return 1
    if re.search(
        r"[A-ZĄĆĘŁŃÓŚŹŻ]\.\s*[A-ZĄĆĘŁŃÓŚŹŻ]\.|[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+\s+[A-ZĄĆĘŁŃÓŚŹŻ][a-ząćęłńóśźż-]+",
        value,
    ):
        return 1
    return None


def recipient_group(text):
    """
    Recover an explicit group of recipients from an operative sentence.

    args:
        text: Source ruling sentence.

    returns:
        Recipient list without role prefixes, or None when not explicit.
    """
    match = re.search(r"na rzecz\s+(.+?)\s+(?:kwot[ęy]|tytułem)\b", text, re.IGNORECASE)
    if not match:
        return None
    return re.sub(
        r"\b(?:powoda|powódki|powodów|poszkodowanego|poszkodowanych)\s+",
        "",
        match[1],
        flags=re.IGNORECASE,
    ).strip()


async def extract_targets(server, text, config):
    """
    Find award/request sentences with embeddings and resolve principal amounts with Basal.

    args:
        server: Local Basal server for amount resolution only.
        text: Full normalized judgment text.
        config: Effective pipeline settings, patterns and feature/target schema.

    returns:
        Award components, total across beneficiaries, optional request and match diagnostics.
    """
    from . import basal_filtering as basal

    matches = match_targets(text, config)
    calls = basal.MODEL_CALLS.get()
    if calls is not None:
        calls.append(
            {
                "engine": "embedding",
                "model": config["embedding"]["model"],
                "matches": matches,
            }
        )
    result = {field: "brak informacji" for field in (*config["target_fields"], "suma")}
    result.update(
        confirmed=False,
        request="brak informacji",
        injured_party_count="not_available",
        target_matches=matches,
        amount_decisions=[],
        status="no matching judgment sentence",
    )
    selected = matches["judgment"]
    if selected is None:
        return result
    # Use the ruling around the winning sentence to include awards to other beneficiaries.
    reasons = list(REASONS.finditer(text))
    ruling_end = reasons[0].start() if len(reasons) == 1 else 0
    source = text[:ruling_end] if selected["end"] <= ruling_end else selected["text"]
    source_start = 0 if selected["end"] <= ruling_end else selected["start"]
    request = matches["request"]
    if request is not None:
        requested = list(MONEY.finditer(request["text"]))
        if len(requested) == 1:
            result["request"] = amount_in_pln(requested[0])
    beneficiaries = (
        await basal.ask(
            server,
            source,
            {
                "beneficiaries": {
                    "type": "extract",
                    "instructions": "Wskaż wszystkich poszkodowanych, którym sąd w tym wyroku "
                    "zasądza świadczenie. Nie wliczaj reprezentanta, pełnomocnika ani osoby, której sprawę umorzono. "
                    "Jeśli nie można wskazać listy, wybierz not stated.",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "people": {
                                "type": "string",
                                "description": "Źródłowy fragment z listą odbiorców zasądzonego świadczenia albo ich jawną liczbą.",
                            }
                        },
                    },
                }
            },
            config,
        )
    )["beneficiaries"]
    count = count_people(beneficiaries["data"]["people"])
    result["injured_party_count"] = count if count is not None else "not_available"
    rows = []
    for row in basal.amount_contexts(source):
        state = {
            "fragment": row["marked"],
            "sekcja": "zdanie wybrane przez embedding lub jego sentencja",
            "sentencja": source,
            "żądanie": request["text"] if request else "brak informacji",
        }
        answer = (
            await basal.ask(
                server,
                state,
                {"swiadczenie": {"type": "choice", **config["questions"]["amount"]}},
                config,
            )
        )["swiadczenie"]
        accepted = answer["confidence"] >= config["system_one"][
            "amount_minimum_probability"
        ] and answer["choice"] in {*config["target_fields"], "suma"}
        rows.append(
            {
                **row,
                "category": answer["choice"],
                "confidence": answer["confidence"],
                "accepted": accepted,
                "start": row["start"] + source_start,
                "end": row["end"] + source_start,
                "amount_start": row["amount_start"] + source_start,
                "amount_end": row["amount_end"] + source_start,
            }
        )
    result["amount_decisions"] = rows
    accepted = [row for row in rows if row["accepted"]]
    if not accepted:
        result["status"] = "no established principal amount in the selected ruling"
        return result
    groups = [recipient_group(row["text"]) for row in accepted]
    if all(group and count_people(group) is not None for group in groups):
        names = {
            re.sub(r"\s+", "", name).casefold()
            for group in groups
            for name in re.split(r"\s*(?:,|\boraz\b|\bi\b)\s*", group)
        }
        if all(re.search(r"[A-ZĄĆĘŁŃÓŚŹŻ]", group) for group in groups):
            count = len(names)
            result["injured_party_count"] = count
    totals = {}
    for row in accepted:
        amount = Decimal(row["kwota"])
        offset = row["amount_start"] - row["start"]
        per_person = re.search(r"\bpo\s*$", row["text"][:offset], re.IGNORECASE)
        if per_person:
            group = recipient_group(row["text"])
            group_count = count_people(group) if group else None
            if (
                group_count is None
                and len({(r["start"], r["end"]) for r in accepted}) == 1
            ):
                group_count = count
            if group_count is None:
                result["status"] = (
                    "per-person award has an unresolved beneficiary group"
                )
                return result
            amount *= group_count
        totals[row["category"]] = totals.get(row["category"], Decimal(0)) + amount
    component_total = sum(
        (value for key, value in totals.items() if key != "suma"), Decimal(0)
    )
    if "suma" in totals and component_total:
        complete = all(field in totals for field in config["target_fields"])
        if component_total > totals["suma"] or (
            complete and component_total != totals["suma"]
        ):
            result["status"] = "components conflict with the explicit total"
            return result
    for field in config["target_fields"]:
        if field in totals:
            result[field] = str(totals[field])
    result["suma"] = str(totals.get("suma", component_total))
    result["status"] = "principal summed across awarded beneficiaries"
    verification = (
        await basal.ask(
            server,
            {
                "sentencja": source,
                "skladniki": {key: result[key] for key in config["target_fields"]},
                "łączna_kwota": result["suma"],
                "poszkodowani": groups
                if all(groups)
                else beneficiaries["data"]["people"],
                "liczba_poszkodowanych": result["injured_party_count"],
            },
            {
                "zasadzenie": {
                    "type": "noul",
                    "instructions": "Czy podana kwota łączna odpowiada wyłącznie "
                    "świadczeniom głównym zasądzonym wszystkim wskazanym poszkodowanym w tej sentencji? "
                    "Kwoty «po» wymagają przemnożenia przez liczbę odbiorców; wspólnej kwoty nie mnożymy. "
                    "Nie wliczaj żądań, odsetek, kosztów procesu ani wcześniejszych wypłat. "
                    "Sprawdź też liczbę poszkodowanych, jeśli została podana. Suma może być obliczona, "
                    "nie musi występować dosłownie w tekście.",
                }
            },
            config,
        )
    )["zasadzenie"]
    result["verification"] = verification
    result["confirmed"] = (
        verification["noul"] >= config["system_one"]["amount_minimum_probability"]
    )
    return result
