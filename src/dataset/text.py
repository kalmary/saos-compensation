"""Judgment text normalization and source boundaries."""

import re
from decimal import Decimal
from html.parser import HTMLParser


class JudgmentText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "li", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "h1", "h2", "h3", "h4", "h5", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        # Łamanie linii w źródle HTML nie tworzy akapitu wewnątrz <p>/<span>.
        self.parts.append(re.sub(r"\s+", " ", data))


def extract_text(html):
    parser = JudgmentText()
    parser.feed(html)
    lines = [
        re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()
    ]
    return "\n".join(line for line in lines if line)


REASONS = re.compile(r"^UZASADNIENIE\s*$", re.IGNORECASE | re.MULTILINE)
PLN_UNIT = r"(?:PLN|zł(?:otych|ote|oty)?)"
CURRENCY = re.compile(rf"\b{PLN_UNIT}\b", re.IGNORECASE)

AMOUNT_NUMBER = (
    r"(?:[0-9]{1,3}(?:[ .\u00a0\u202f][0-9]{3})+|[0-9]+)"
    r"(?:,[0-9]+)?"
)
AMOUNT_SCALE = (
    r"(?:tys\.?|tysiąc|tysiące|tysięcy"
    r"|mln\.?|milion(?:y|ów)?"
    r"|mld\.?|miliard(?:y|ów)?)"
)
AMOUNT_PART = re.compile(
    rf"(?P<number>{AMOUNT_NUMBER})\s*(?P<scale>{AMOUNT_SCALE})?",
    re.IGNORECASE,
)

# shortcut: kwoty zapisane w całości słownie wymagają przeglądu; rozszerzyć po ocenie takich przypadków.
MONEY = re.compile(
    r"""
    (?<![\w.,])
    (?<![0-9][ \u00a0\u202f])
    (?P<amount>
    """
    + AMOUNT_NUMBER
    + rf"(?:\s*{AMOUNT_SCALE}(?:\s+{AMOUNT_NUMBER}(?:\s*{AMOUNT_SCALE})?)*)?"
    + r")\s*"
    + PLN_UNIT
    + r"\b",
    re.IGNORECASE | re.VERBOSE,
)


def amount_in_pln(match):
    total = Decimal(0)
    previous_scale = None
    scaled = False
    for part in AMOUNT_PART.finditer(match["amount"]):
        number = re.sub(r"[ .\u00a0\u202f]", "", part["number"]).replace(",", ".")
        scale = (part["scale"] or "").lower()
        scaled = scaled or bool(scale)
        if scale.startswith("tys"):
            multiplier = 1_000
        elif scale.startswith(("mln", "milion")):
            multiplier = 1_000_000
        elif scale.startswith(("mld", "miliard")):
            multiplier = 1_000_000_000
        else:
            multiplier = 1
        if previous_scale is not None and multiplier >= previous_scale:
            raise ValueError(
                "Jednostki kwoty muszą występować od największej do najmniejszej"
            )
        total += Decimal(number) * multiplier
        previous_scale = multiplier
    value = format(total, "f")
    if scaled and "." in value:
        value = value.rstrip("0").rstrip(".")
    return value
