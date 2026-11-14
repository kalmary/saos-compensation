"""Resolve amounts only after embedding selection has located an award sentence."""

import re
from decimal import Decimal

from ...embeddings.target_matching import match_targets
from ..model import decisions
from ..source.text import MONEY, amount_in_pln


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
    match = re.search(r"na rzecz\s+(.+?)\s+(?:kwot[ęy]|tytułem|nawiązkę|odszkodowanie|zadośćuczynienie)\b", text, re.IGNORECASE)
    if not match:
        return None
    return re.sub(
        r"\b(?:powoda|powódki|powodów|poszkodowanego|poszkodowanych)\s+",
        "",
        match[1],
        flags=re.IGNORECASE,
    ).strip()


async def extract_targets(server, text, config):
    """Read the numeric target only from the embedding-selected judgment sentence.

    args:
        server: Unused inference server; target extraction makes no model requests.
        text: Full normalized judgment text.
        config: Embedding settings and configured target fields.

    returns:
        Parsed award or an explicit missing/ambiguous-target status, plus match diagnostics.
    """
    matches = match_targets(text, config)
    calls = decisions.MODEL_CALLS.get()
    if calls is not None:
        calls.append(
            {
                "engine": "embedding",
                "model": config["embedding"]["model"],
                "matches": matches,
            }
        )
    result = {
        "suma": "brak informacji",
        "confirmed": False,
        "request": "brak informacji",
        "injured_party_count": "not_available",
        "target_matches": matches,
        "status": "no matching judgment sentence",
    }
    denial = matches.get("judgment-denial")
    if denial is not None:
        if text[denial["start"]:denial["end"]] != denial["text"]:
            raise ValueError("Selected denial sentence does not match its source")
        result.update(denied=True, suma="0", confirmed=True, status="judgment denied")
        return result
    selected = matches["judgment"]
    if selected is None:
        return result
    if selected["similarity"] <= config["embedding"]["minimum_cosine_similarity"]:
        result["status"] = "judgment sentence below the similarity threshold"
        return result
    sentence = selected["text"]
    if text[selected["start"] : selected["end"]] != sentence:
        raise ValueError("Selected judgment sentence does not match its source")
    amounts = list(MONEY.finditer(sentence))
    if len(amounts) != 1:
        result["status"] = (
            "selected judgment sentence requires exactly one numeric PLN amount"
        )
        return result
    amount = amounts[0]
    total = Decimal(amount_in_pln(amount))
    group = recipient_group(sentence)
    count = count_people(group) if group else None
    result["injured_party_count"] = count if count is not None else "not_available"
    if re.search(r"\bpo\s*$", sentence[: amount.start()], re.IGNORECASE):
        if count is None:
            result["status"] = "per-person award has an unresolved beneficiary group"
            return result
        total *= count
    result.update(
        suma=str(total),
        confirmed=True,
        status="amount parsed from embedding-selected judgment sentence",
    )
    request = matches.get("request")
    if request is not None:
        requested = list(MONEY.finditer(request["text"]))
        if len(requested) == 1:
            result["request"] = amount_in_pln(requested[0])
    return result
