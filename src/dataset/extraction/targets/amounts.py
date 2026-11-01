"""Monetary contexts, award verification and earlier-payment extraction."""

from ..model import decisions
from ..source import text as api
from ..source.passages import fragments
from ..source.text import sentence_fragments


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
        await decisions.ask(
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
