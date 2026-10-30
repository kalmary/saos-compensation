"""Badanie danych: uv run --extra extraction notebooks/01_saos_feasibility.py.

Wynik: odpowiedzi na pięć pytań, dwa targety, cytaty i opisy cech.
To badanie wykonalności, nie gotowy ekstraktor etykiet ani zbiór do treningu.
Zmiany badawcze pozostają w tym pliku; odpowiedzi API i raport trafiają do tmp/.
"""

# %% 1. Scope and acquisition rationale.
# Search for compensation judgments mentioning PLN amounts, distinguishing awards from requests,
# prior payments, costs and references to other cases. Track non-material and material damages
# separately; exclude interest, litigation costs and periodic rent. Examples concern this judgment.
# Request parameters are checked against the live queryTemplate:
# https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przeszukiwania-danych
# https://www.saos.org.pl/help/index.php/search-query-language
# https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przegladania-danych
# courtType=COMMON identifies the court family; ccCourtType identifies its level.
# Search matches alone cannot establish the monetary target.

import asyncio
import hashlib
import importlib
import json
import re
import sqlite3
import time
from collections import Counter
from calendar import monthrange
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "saos_feasibility"
CONFIG = importlib.import_module("03_system_one_polish").load_config()
BASE_URL = "https://www.saos.org.pl/api"

# %% 2. Preserve original JSON and acquisition provenance.
# Cached responses support repeated experiments. Snapshot times may differ between queries.
# Move tmp/saos_feasibility/ to refresh the study. HTTP, JSON and query mismatches stop the run.
# HTML normalization preserves paragraph and heading boundaries.

def fetch_json(url):
    """Preserve the response bytes and request metadata; reuse cached responses."""
    if not url.startswith(BASE_URL + "/"):
        raise ValueError(f"Expected a SAOS API URL: {url}")

    key = hashlib.sha256(url.encode()).hexdigest()
    raw_path = OUTPUT / f"{key}.json"
    metadata_path = OUTPUT / f"{key}.meta.json"
    if raw_path.exists() and metadata_path.exists():
        raw = raw_path.read_bytes()
        metadata = json.loads(metadata_path.read_text())
        if metadata["url"] != url or metadata["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError(f"Cached response failed its integrity check: {raw_path}")
        return json.loads(raw)

    for attempt in range(3):
        try:
            with requests.get(url, headers={"Accept": "application/json"}, timeout=30) as response:
                response.raise_for_status()
                raw = response.content
                content_type = response.headers.get("Content-Type", "")
                status = response.status_code
            payload = json.loads(raw)
            if status != 200 or "json" not in content_type or not isinstance(payload, dict):
                raise ValueError(f"Unexpected JSON response: {url}")
            metadata = {
                "url": url,
                "acquired_at": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "content_type": content_type,
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
            raw_path.write_bytes(raw)
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
            time.sleep(0.3)
            return payload
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as error:
            if isinstance(error, requests.HTTPError) and error.response.status_code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 2:
                raise
            print(f"Request failed; retrying: {error}", flush=True)
            time.sleep(2**attempt)


def save_source(payload, directory=None):
    data = payload["data"]
    if type(data.get("id")) is not int or data["id"] <= 0 or not isinstance(data.get("textContent"), str):
        raise ValueError("Nieprawidłowy wyrok do zapisu")
    directory = Path(directory) if directory is not None else OUTPUT / str(data["id"])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "judgment.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (directory / "source.txt").write_text(extract_text(data["textContent"]) + "\n", encoding="utf-8")


def fixture_texts(directory=None):
    directory = Path(directory) if directory is not None else ROOT / "fixtures"
    texts = {}
    for path in sorted(directory.glob("*/judgment.json")):
        data = json.loads(path.read_text(encoding="utf-8"))["data"]
        if type(data.get("id")) is not int or data["id"] in texts or not isinstance(data.get("textContent"), str):
            raise ValueError(f"Nieprawidłowa lub powtórzona fixtura: {path}")
        text = extract_text(data["textContent"])
        if not text:
            raise ValueError(f"Pusty tekst fixtury: {path}")
        texts[data["id"]] = text
    if not texts:
        raise ValueError(f"Brak fixtur: {directory}")
    return texts


def search_settings(config, today=None):
    if set(config) - {"download_path", "download_only", "timeout"} != {"max_age_years", "max_pages", "debug"}:
        raise ValueError("api.yaml wymaga max_age_years, max_pages i debug")
    if type(config["debug"]) is not bool:
        raise ValueError("debug musi być wartością true albo false")
    years, limit = config["max_age_years"], config["max_pages"]
    if type(years) is not int or years <= 0 or type(limit) is not int or limit < 0:
        raise ValueError("max_age_years musi być dodatnie, a max_pages nieujemne")
    end = today or date.today()
    start = end.replace(year=end.year - years, day=min(end.day, monthrange(end.year - years, end.month)[1]))
    params = {"all": "zadośćuczynienie zł", "courtType": "COMMON", "judgmentTypes": "SENTENCE",
              "pageSize": 100, "pageNumber": 0, "sortingField": "DATABASE_ID", "sortingDirection": "ASC",
              "judgmentDateFrom": start.isoformat(), "judgmentDateTo": end.isoformat()}
    return start, end, params


def search_pages(config, today=None):
    _, _, params = search_settings(config, today)
    page = 0
    while config["max_pages"] == 0 or page < config["max_pages"]:
        result = search_judgments({**params, "pageNumber": page})
        total = result["info"]["totalResults"]
        if not result["items"] and page * params["pageSize"] < total:
            raise ValueError("API zwróciło pustą stronę przed końcem wyników")
        yield page, result
        if (page + 1) * params["pageSize"] >= total:
            break
        page += 1


def judgment_texts(config):
    start, end, _ = search_settings(config)
    texts = {}
    for _, page in search_pages(config):
        for item in page["items"]:
            identifier = item["id"]
            if identifier in texts:
                continue
            payload = fetch_json(f"{BASE_URL}/judgments/{identifier}")
            data = payload["data"]
            if data["id"] != identifier or not start <= date.fromisoformat(data["judgmentDate"]) <= end:
                raise ValueError(f"Nieprawidłowy wyrok: {identifier}")
            if not isinstance(data["textContent"], str):
                raise ValueError(f"Brak tekstu: {identifier}")
            text = extract_text(data["textContent"])
            if not text:
                raise ValueError(f"Pusty tekst: {identifier}")
            if config["debug"]:
                save_source(payload)
            texts[identifier] = text
    return texts


def search_judgments(params):
    url = requests.Request("GET", BASE_URL + "/search/judgments", params=params).prepare().url
    result = fetch_json(url)
    if not isinstance(result.get("items"), list) or not isinstance(result.get("links"), list):
        raise ValueError("Search response is missing items or pagination links")
    if type(result.get("info", {}).get("totalResults")) is not int:
        raise ValueError("Search response is missing the result count")

    template = result["queryTemplate"]
    for key, value in params.items():
        observed = template[key]
        if isinstance(observed, dict):
            observed = observed["value"]
        expected = [value] if key in ("judgmentTypes", "keywords") else value
        if observed != expected:
            raise ValueError(f"API did not apply {key}: expected {expected}, got {observed}")

    ids = [item["id"] for item in result["items"]]
    if any(type(value) is not int or value <= 0 for value in ids):
        raise ValueError("Search returned an invalid judgment ID")
    if len(ids) != len(set(ids)) or len(ids) > params["pageSize"]:
        raise ValueError("Search returned duplicate IDs or an oversized page")
    return result


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
        # HTML source line breaks do not create paragraphs inside inline elements.
        self.parts.append(re.sub(r"\s+", " ", data))


def extract_text(html):
    parser = JudgmentText()
    parser.feed(html)
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


# %% Official court and locality metadata.
# Read the GeoPackage with SQLite; WERSJA is an object version, not the population measurement year.
# shortcut: a single GUGiK snapshot is used; historical analysis needs snapshots from judgment years.
def locality_class(kind, population):
    if kind == "wieś":
        return "wies"
    if kind != "miasto" or population is None:
        return "brak informacji"
    if type(population) not in (int, float) or not 0 < population < float("inf"):
        return "brak informacji"
    for boundary, label in (
        (20000, "miasto_ponizej_20_tys"), (100000, "miasto_20_do_ponizej_100_tys"),
        (200000, "miasto_100_do_ponizej_200_tys"), (500000, "miasto_200_do_ponizej_500_tys"),
    ):
        if population < boundary:
            return label
    return "miasto_co_najmniej_500_tys"


def load_localities():
    path = ROOT / "tmp" / "gugik" / "OT_ADMS_P.gpkg"
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_suffix(".meta.json")
    source = CONFIG["localities"]["url"]
    if not path.exists():
        print("Pobieranie miejscowości GUGiK (około 30 MB)...", flush=True)
        temporary = path.with_suffix(".part")
        digest = hashlib.sha256()
        with requests.get(source, stream=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    digest.update(chunk)
                    output.write(chunk)
        with sqlite3.connect(f"file:{temporary}?mode=ro", uri=True) as connection:
            if connection.execute("pragma quick_check").fetchone()[0] != "ok":
                raise ValueError("Niekompletny lub uszkodzony GPKG GUGiK")
        metadata_path.write_text(json.dumps({"url": source, "sha256": digest.hexdigest(),
            "acquired_at": datetime.now(timezone.utc).isoformat()}, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["url"] != source or hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
        raise ValueError("Niezgodny snapshot GUGiK")
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute("select NAZWA, RODZAJ, LICZBAMIESZKANCOW, IDENTYFIKATORSIMC, "
                                  "IDENTYFIKATORTERC, IDENTYFIKATORPRNG, WERSJA from OT_ADMS_P "
                                  "where RODZAJ in ('miasto', 'wieś')").fetchall()
    return [dict(row) for row in rows], metadata


def match_locality(court_name, rows, morphology):
    # Use Morfeusz to match inflected court-seat names against official localities; ambiguous matches remain missing.
    location = re.search(r"\b(?:w|we)\s+([^\n]+)$", court_name)
    if not location:
        return None, "brak nazwy siedziby w nazwie sądu"
    name = location[1].strip()
    exact = [row for row in rows if row["NAZWA"].casefold() == name.casefold()]
    if len(exact) == 1:
        return exact[0], "dokładna nazwa"
    if len(exact) > 1:
        return None, "wiele miejscowości o tej nazwie"
    words = name.split()
    lemmas = []
    for word in words:
        forms = {word.casefold()}
        for _, _, interpretation in morphology.analyse(word):
            forms.add(interpretation[1].split(":")[0].casefold())
        lemmas.append(forms)
    matches = [row for row in rows if len(row["NAZWA"].split()) == len(words)
               and all(word.casefold() in forms for word, forms in zip(row["NAZWA"].split(), lemmas))]
    return (matches[0], "jednoznaczna odmiana Morfeusz/SIMC") if len(matches) == 1 else (None, f"liczba kandydatów: {len(matches)}")


def court_metadata(record, rows, morphology, courts):
    court_id = record["division"]["court"]["id"]
    if court_id not in courts:
        courts[court_id] = fetch_json(f"{BASE_URL}/commonCourts/{court_id}")["data"]
    court = courts[court_id]
    if court["id"] != court_id or court["type"] not in {"DISTRICT", "REGIONAL", "APPEAL"}:
        raise ValueError("Nieprawidłowe dane sądu")
    hierarchy, visited, current = {}, set(), court
    while current:
        if current["id"] in visited:
            raise ValueError("Cykl w hierarchii sądów")
        visited.add(current["id"])
        hierarchy[current["type"]] = current["name"]
        parent = current.get("parentCourt")
        if not parent:
            break
        if parent["id"] not in courts:
            courts[parent["id"]] = fetch_json(f"{BASE_URL}/commonCourts/{parent['id']}")["data"]
        current = courts[parent["id"]]
        if current["id"] != parent["id"]:
            raise ValueError("Niezgodny identyfikator sądu nadrzędnego")
    division = fetch_json(f"{BASE_URL}/ccDivisions/{record['division']['id']}")["data"]
    if division["id"] != record["division"]["id"] or division["court"]["id"] != court_id:
        raise ValueError("Niezgodne dane wydziału")
    if "locality_match" not in court:
        court["locality_match"] = match_locality(court["name"], rows, morphology)
    locality, reason = court["locality_match"]
    return {
        "sad": {"id": court_id, "nazwa": court["name"]}, "poziom_sadu": court["type"],
        "instancja": "brak informacji", "okreg_sadowy": hierarchy.get("REGIONAL", "brak informacji"),
        "apelacja_sadowa": hierarchy.get("APPEAL", "brak informacji"),
        "rodzaj_wydzialu": division.get("type") or "brak informacji", "wydzial": division["name"],
        "data_wyroku": record["judgmentDate"],
        "klasa_miejscowosci_sadu": locality_class(locality["RODZAJ"], locality["LICZBAMIESZKANCOW"]) if locality else "brak informacji",
        "miejscowosc_zrodlo": locality, "dopasowanie_miejscowosci": reason,
        "rok_pomiaru_ludnosci": "brak informacji",
    }



# %% 3. Compare explicit document-section boundaries.
# A reasoning heading can include both facts and award justification. This comparison requires
# one explicit fact/reasoning boundary pair; it does not establish that the resulting text is safe.
# shortcut: selected explicit headings are recognized; extend after reviewing missed layouts.
REASONS = re.compile(r"(?im)^UZASADNIENIE\s*$")
FACTS = re.compile(
    r"(?im)^(?:Sąd[^\n:]{0,60}\bustalił[^\n:]{0,80}:|"
    r"Ustalenia faktyczne\s*:|Stan faktyczny\s*:)"
)
LAW = re.compile(r"(?im)^(?:Sąd[^\n:]{0,60}\bzważył[^\n:]{0,80}[:.]|Rozważania prawne\s*:)")


def split_sections(text):
    headings = list(REASONS.finditer(text))
    if len(headings) != 1:
        return {"sentencja": "", "uzasadnienie": "", "fakty": "", "rozważania": ""}
    heading = headings[0]
    reasons = text[heading.end():].strip()
    facts, law = list(FACTS.finditer(reasons)), list(LAW.finditer(reasons))
    separated = len(facts) == len(law) == 1 and facts[0].end() < law[0].start()
    return {
        "sentencja": text[:heading.start()].strip(),
        "uzasadnienie": reasons,
        "fakty": reasons[facts[0].end():law[0].start()].strip() if separated else "",
        "rozważania": reasons[law[0].end():].strip() if separated else "",
    }


# %% 4. Numeric PLN mentions measure text availability, not target coverage.
# Module 03 determines awards separately.
# shortcut: written-out amounts, scaled abbreviations and foreign currencies need review here.
MONEY = re.compile(
    r"(?<![\w.,])(?P<amount>(?:\d{1,3}(?:[ .]\d{3})+|\d+)(?:,\d{1,2})?)"
    r"\s*(?:zł(?:ot(?:ych|e|y))?|PLN)\b", re.I
)
CURRENCY = re.compile(r"\b(?:PLN|zł|złotych|złote|złoty|EUR|euro|USD|dolary|dolarów|GBP|funty|funtów|CHF|franki|franków)\b|[€$£]", re.I)
def amounts(text, section):
    return [{"kwota": str(Decimal(match["amount"].replace(" ", "").replace(".", "").replace(",", "."))),
             "waluta": "PLN", "sekcja": section,
             "cytat": text[max(0, match.start() - 100):match.end() + 220].strip()}
            for match in MONEY.finditer(text)]


# %% 5. Extract six broad descriptions with source offsets and scores; missing extraction is not missing source evidence.
FEATURES = CONFIG["fields"]
LEAKAGE = re.compile(
    r"zadośćuczyn|odszkodowan|zasądz|żądan|domaga|wypłac|odset|"
    r"adekwatn|odpowiedni\w* sum|wysokoś\w* świadczeni", re.I
)


def quote(text, pattern, width=300):
    match = re.search(pattern, text, re.I)
    if not match:
        return "brak informacji"
    start, end = max(0, match.start() - 65), min(len(text), match.end() + width)
    return re.sub(r"\s+", " ", text[start:end]).strip()


# %% 6. Offline rule checks cover decimals, zero, missing sections, requests and missing information.
assert extract_text("<p>A <span>B</span>.</p><p>Żółć&nbsp;123</p>") == "A B.\nŻółć 123"
assert [row["kwota"] for row in amounts("15.000,50 zł; 2 000 PLN; 2000zł; 0 zł", "fakty")] == ["15000.50", "2000", "2000", "0"]
assert not amounts("4 mln zł; 15 EUR; 20.10.2020", "fakty")
assert split_sections("WYROK\noddala powództwo")["fakty"] == ""
assert split_sections("WYROK\nUZASADNIENIE\nSąd ustalił:\nUraz.\nSąd zważył:\nOcena.")["fakty"] == "Uraz."


# %% 7. Fixed compensation search with configurable age/page limits; zero pages means unlimited.
if __name__ == "__main__":
    import argparse

    import morfeusz2

    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--metadata-only", action="store_true", help="Pomiń próby modeli lokalnych")
    options = arguments.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    localities, locality_source = load_localities()
    morphology, courts = morfeusz2.Morfeusz(), {}
    start_date, today, params = search_settings(CONFIG["api"])
    report = ["# Badanie SAOS: zadośćuczynienie, kwoty i cechy", "", "## 1. Co i jak wyszukujemy?", "",
              f"Zakres dat orzeczenia: {start_date}–{today}.",
              f"Parametry zapytania: {json.dumps(params, ensure_ascii=False)}.",
              f"Limit stron: {CONFIG['api']['max_pages']} (0 = wszystkie)."]
    sample, sources = {}, {}
    for page, result in search_pages(CONFIG["api"], today):
        print(f"Strona {page}: {len(result['items'])} wyników z {result['info']['totalResults']}", flush=True)
        report.append(f"- Strona {page}: {len(result['items'])} wyników.")
        for item in result["items"]:
            sample[item["id"]] = item
            sources.setdefault(item["id"], []).append(f"strona {page}")
    study_ids = set(sample)

    # %% 8. Fetch full judgment text; search snippets and empty decision/summary metadata cannot replace it.
    analyses, failures = {}, {}
    for identifier in sample:
        print(f"Pełny wyrok {identifier} ({len(analyses) + len(failures) + 1}/{len(sample)})", flush=True)
        payload = fetch_json(f"{BASE_URL}/judgments/{identifier}")
        record = payload.get("data")
        try:
            if not isinstance(record, dict) or type(record.get("id")) is not int or record["id"] != identifier:
                raise ValueError("brak danych lub niezgodne ID")
            date.fromisoformat(record["judgmentDate"])
            if not start_date <= date.fromisoformat(record["judgmentDate"]) <= today:
                raise ValueError("data poza zakresem ostatnich 10 lat")
            if record["courtType"] != "COMMON" or not record["division"]["court"]["name"]:
                raise ValueError("brak metadanych sądu")
            if not record["courtCases"] or any(not c["caseNumber"] for c in record["courtCases"]):
                raise ValueError("brak sygnatury")
            if not isinstance(record["textContent"], str) or not record["textContent"].strip():
                raise ValueError("brak tekstu")
        except (KeyError, TypeError, ValueError) as error:
            failures[identifier] = str(error)
            continue
        text = extract_text(record["textContent"])
        if not text:
            failures[identifier] = "brak tekstu po usunięciu HTML"
            continue
        if CONFIG["api"]["debug"]:
            save_source(payload)
        sections = split_sections(text)
        # Without a unique reasoning boundary, scan all amounts but do not infer a ruling-based award role.
        parts = {key: sections[key] for key in ("sentencja", "uzasadnienie")}
        if not sections["uzasadnienie"]:
            parts = {"nieustalona": text}
        rows = [row for section, content in parts.items() for row in amounts(content, section)]
        metadata = court_metadata(record, localities, morphology, courts)
        analyses[identifier] = {
            "text": text, "sections": sections, "amounts": rows,
            "features": {field: "brak informacji" for field in FEATURES}, "metadata": metadata,
            "feature_status": "nie przetworzono modelem",
        }
    if not analyses:
        raise ValueError(f"Brak poprawnych dokumentów: {failures}")
    study = {identifier: a for identifier, a in analyses.items() if identifier in study_ids}
    if not study:
        raise ValueError("Brak poprawnych dokumentów z losowanej próbki")
    # Model przetwarza wszystkie poprawne dokumenty z pobranych stron.
    model_run = None
    if not options.metadata_only:
        selected = {identifier: analysis["text"] for identifier, analysis in analyses.items()}
        backend = CONFIG["backend"]
        if backend not in {"system_one", "gemma"}:
            raise ValueError("Nieznany backend ekstrakcji")
        module = importlib.import_module("03_system_one_polish" if backend == "system_one" else "04_gemma")
        model_run = asyncio.run(module.run(selected, CONFIG))
        for identifier, result in model_run["judgments"].items():
            analysis = analyses[int(identifier)]
            analysis["features"] = result["features"]
            analysis["feature_status"] = result["status"]
            analysis["model_extraction"] = result
            if result.get("instance", {}).get("value", "brak informacji") != "brak informacji":
                analysis["metadata"]["instancja"] = result["instance"]["value"]
                analysis["metadata"]["instancja_status"] = "model: do weryfikacji"
    # Retain full results for review without treating them as approved training data.
    (OUTPUT / "analysis.json").write_text(json.dumps({
        "status": "kandydaci do ręcznej weryfikacji", "od": str(start_date), "do": str(today), "api_config": CONFIG["api"],
        "próbka": sorted(study_ids), "błędy": failures, "gugik": locality_source,
        "wyroki": {identifier: {"metadane": a["metadata"], "cechy": a["features"],
                    "wzmianki_kwotowe": a["amounts"], "zasadzenie": a.get("model_extraction", {}).get("award"), "podział_sekcji": bool(a["sections"]["fakty"]),
                    "status_cech": a["feature_status"], "model": a.get("model_extraction"),
                    "dobór": sources[identifier]} for identifier, a in analyses.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Show actual extraction results before diagnostics.
    records = ["# Rekordy opisu sprawy", "",
               "Opisy ekstrakcyjne wybierają fragmenty źródła; wariant Basala zachowuje całe wybrane zdanie, w tym negację.",
               "Brak informacji oznacza brak wydobytej treści; pewność i przyczyny pozostają osobno w analysis.json."]
    if model_run:
        for identifier, result in model_run["judgments"].items():
            records += ["", f"## ID {identifier}", "", "```json",
                        json.dumps(result["features"], ensure_ascii=False, indent=2), "```",
                        "", "Zasądzone świadczenie (oddzielnie od cech):", "```json",
                        json.dumps({k: v for k, v in result["award"].items() if k not in ("amount_decisions", "proposal")},
                                   ensure_ascii=False, indent=2), "```"]
    else:
        records += ["", "Model nie został uruchomiony: rekordów opisowych jeszcze nie wyekstrahowano."]
    report = records + ["", *report]

    report += ["", "## 2. Jaka jest struktura wyroków?", "",
               f"Próbka dekady: {len(study_ids)} różnych ID; poprawne strukturalnie: {len(study)}; błędy: {failures or 'brak'}.",
               f"Łącznie przeanalizowano {len(analyses)} pobranych dokumentów.",
               "Strony pobierane są kolejno według ID. Ograniczenie stron nie tworzy próby losowej.",
               "Nagłówek/sentencja -> UZASADNIENIE -> stanowiska stron -> ustalenia -> rozważania.",
               "Porównawczy podział według nagłówków wymaga jednej pary jawnych granic. Model akapitowy nie ma tego warunku.", "",
               "| ID | Znaki | Uzasadnienie | Granice faktów/rozważań | Kwoty PLN |",
               "|---|---:|---|---|---:|"]
    for identifier, analysis in list(analyses.items())[:12]:
        sections = analysis["sections"]
        report.append(f"| {identifier} | {len(analysis['text'])} | {'tak' if sections['uzasadnienie'] else 'brak informacji'} | {'wykryte; do sprawdzenia' if sections['fakty'] else 'brak informacji'} | {len(analysis['amounts'])} |")
    classes = Counter(a["metadata"]["klasa_miejscowosci_sadu"] for a in study.values())
    report += ["", "### Metadane sądu i wielkość siedziby", "",
               "Dane sądu, wydziału i hierarchii pobrano z API SAOS. Poziom sądu nie zastępuje instancji.",
               "Miejscowość dopasowano do GUGiK BDOT10k: nazwa, odmiana Morfeusz, SIMC/TERC, rodzaj, liczba mieszkańców.",
               "Niejednoznaczne dopasowanie pozostaje brakiem informacji. Nie wybieramy arbitralnie największego kandydata.",
               f"Źródło: {locality_source['url']}; pobrano {locality_source['acquired_at']}.",
               "UWAGA: to jeden snapshot BDOT10k. Wersja obiektu nie oznacza roku pomiaru ludności; brak historii populacji.",
               "| Klasa siedziby | Wyroki w próbce |", "|---|---:|"]
    for label, count in sorted(classes.items()):
        report.append(f"| {label} | {count} |")

    # %% 9. Report target candidates and source quotes without choosing the first or largest amount.
    report += ["", "## 3. Czy można znaleźć kwotę i rozdzielić świadczenia?", "",
               "Dwa targety: zadośćuczynienie i łączne pozostałe odszkodowanie. Suma jest wyliczana.",
               "Reguła wykrywa liczby; model ocenia całe zdanie z oznaczoną kwotą oraz kontekstem sentencji. Potem łączymy zgodne przypisania i sprawdzamy propozycję.",
               "Poniższy wykaz liczb to tylko wzmianki, bez kategorii i roli; nie jest wynikiem ekstrakcji targetów.",
               "Rentę okresową i odsetki wyłączamy z sumy jednorazowej; koszty procesu/sądowe również."]
    example_ids = list(dict.fromkeys([554194, 553925, 553824, *list(study)[:4]]))
    for identifier in example_ids:
        if identifier not in analyses:
            continue
        analysis = analyses[identifier]
        report += ["", f"### ID {identifier}: https://www.saos.org.pl/judgments/{identifier}",
                   f"Dobór: {', '.join(sources[identifier])}."]
        for row in analysis["amounts"][:3]:
            report += [f"- Wzmianka: {row['kwota']} PLN | {row['sekcja']}",
                       f"  Cytat: {row['cytat']}"]
        if not analysis["amounts"]:
            report.append("Kwota w obsługiwanym zapisie: brak informacji (to nie jest target=0).")
        if re.search(r"odset", analysis["text"], re.I):
            report.append("Odsetki: są wzmianki; nie wyliczamy ich kwoty.")

    # Keep reviewed interpretations separate from regex results and verify their quotes on each run; these are not accuracy estimates.
    reviewed = {
        554194: {
            "zadośćuczynienie": "15000", "pozostałe_odszkodowanie": "2176", "suma": "17176",
            "sentencja": "kwotę 17 176,00 zł", "rozważania": "15 000 złotych tytułem zadośćuczynienia jest adekwatna",
            "fakt": "izolowane złamanie kości strzałkowej lewej",
            "opis": "W. W.: 15 000 PLN zadośćuczynienia + 2 176 PLN kosztów leczenia = 17 176 PLN. To poprawna suma świadczeń głównych, ale nie samo zadośćuczynienie. Koszty procesu 6 617 PLN i wydatki sądowe 4 030,41 PLN poza sumą.",
        },
    }
    confirmed = []
    report += ["", "### Ręcznie odczytane przykłady — oddzielnie od heurystyk"]
    for identifier, review in reviewed.items():
        if identifier not in analyses:
            report.append(f"ID {identifier}: poza bieżącą próbką; przykład nie został potwierdzony.")
            continue
        analysis = analyses[identifier]
        for section, key in (("sentencja", "sentencja"), ("rozważania", "rozważania"), ("fakty", "fakt")):
            if review[key] not in re.sub(r"\s+", " ", analysis["sections"][section]):
                raise ValueError(f"ID {identifier}: brak oczekiwanego dowodu w sekcji {section}: {review[key]}")
        assert Decimal(review["zadośćuczynienie"]) + Decimal(review["pozostałe_odszkodowanie"]) == Decimal(review["suma"])
        if identifier == 554194:
            assert Decimal("2176") in [Decimal(r["kwota"]) for r in analysis["amounts"]]
        confirmed.append(identifier)
        report += ["", f"ID {identifier}: {review['opis']}",
                   f"- Sentencja: {quote(analysis['sections']['sentencja'], re.escape(review['sentencja']), 150)}",
                   f"- Powiązanie z zadośćuczynieniem: {quote(analysis['sections']['rozważania'], re.escape(review['rozważania']), 180)}",
                   f"- Fakt: {quote(analysis['sections']['fakty'], re.escape(review['fakt']), 120)}"]
    # A payment to a charity is not compensation awarded to the claimant.
    if 553925 in analyses:
        report += ["", "ID 553925 — pułapka beneficjenta: " + quote(analyses[553925]["text"], r"20.000 złotych", 100),
                   "Rozważania wskazują Fundację jako odbiorcę. Nie przypisujemy tej kwoty powodowi jako jego zadośćuczynienia."]

    # %% 10. Report case descriptions and missing fields with source evidence; these are not embeddings or proven predictors.
    report += ["", "## 4. Czy są opis sprawy i osobne kategorie cech?", "",
               "Wartość poniżej to cytat do weryfikacji albo dokładnie 'brak informacji'.",
               "Brak informacji = brak dowodu w wyodrębnionych, przefiltrowanych faktach; nie dowodzi braku zjawiska w sprawie.",
               "Pola są szerokie i pochodzą z YAML. O przypisaniu fragmentu decyduje lokalny model, nie słownik urazów.",
               "Nieprzetworzone dokumenty mają osobny status; brak wykonania modelu nie dowodzi braku cechy w tekście.",
               "Leczenie/rehabilitacja/opieka mogą sugerować potrzebę wydatków, lecz nie dowodzą poniesienia ani zasądzenia kosztów.",
               "Opis może wspierać predykcję; jej jakości nie da się ustalić bez etykiet i testu modelu.", "",
               "| Kategoria cechy | Wyroki z zaakceptowanym fragmentem / przetworzone modelem |", "|---|---:|"]
    processed = [a for a in analyses.values() if a.get("model_extraction", {}).get("fields_evaluated", False)]
    model_document_count = len(model_run["judgments"]) if model_run else 0
    for field in FEATURES:
        count = sum(a["features"][field] != "brak informacji" for a in processed)
        report.append(f"| {field} | {count}/{len(processed)} |")
    if model_run:
        evaluation = model_run.get("evaluation", [])
        report += ["", f"Backend: {CONFIG['backend']}; przetworzono modelem {len(model_run['judgments'])} wyroków.",
                   f"Poprawne decyzje sekcji w jawnej próbie kontrolnej: {sum(r['correct_section'] for r in evaluation)}/{len(evaluation)}.",
                   f"Poprawne decyzje o wycieku: {sum(r['correct_leak'] for r in evaluation)}/{len(evaluation)}.",
                   "To próba diagnostyczna, nie niezależny pomiar jakości na zbiorze SAOS.",
                   "Propozycje dwóch targetów, cytaty i odrzucone fragmenty zapisano w analysis.json oraz wynikach modelu."]
        report += ["", "### Oddzielenie opisu sprawy od rozważań — semantyka akapitów", "",
                   "Każdy akapit jest oceniany z kontekstem poprzedzającym i następującym; nagłówki nie są warunkiem analizy.",
                   "Model rozróżnia fakty, stanowiska stron, rozważania, rozstrzygnięcie, treść mieszaną, niejednoznaczną i inne elementy.",
                   "Akapit mieszany jest ponownie oceniany zdaniami. Do pól trafiają tylko dopuszczone fragmenty faktów.",
                   "Rozdzielone teksty: tmp/system_one/<ID>/facts.txt, reasoning.txt i review.txt. To podział do weryfikacji."]
        for identifier, result in model_run["judgments"].items():
            report += ["", f"ID {identifier}: {result.get('paragraph_counts', {})}"]
            for category in ("fakty", "rozwazania", "stanowiska", "mieszany"):
                examples = [row for row in result.get("diagnostics", []) if row["answers"]["sekcja"]["choice"] == category]
                if category == "fakty":
                    examples = [row for row in examples if row["accepted"]]
                if examples:
                    row = examples[0]
                    report.append(f"- {category} [{row['start']}:{row['end']}]: {row['text'][:450]}")
        report += ["", "### Statusy brakujących i wybranych pól"]
        for identifier, result in model_run["judgments"].items():
            report.append(f"\nID {identifier}: {result.get('feature_input', {}).get('status', 'brak informacji')}")
            for field, status in result.get("field_status", {}).items():
                report.append(f"- {field}: {status}")
    report += ["", "Metadane jako cechy: ID/obszar sądu, poziom sądu, etap postępowania (jeśli jawny), data wyroku.",
               "Poziom sądu NIE oznacza instancji. Obszar sądu NIE oznacza miejsca zdarzenia ani zamieszkania.",
               "Nie badano jeszcze, czy sąd, obszar lub rok poprawiają predykcję. Wszystkie trzy poziomy są w próbie."]
    for identifier in confirmed:
        report += ["", f"### Cechy ID {identifier}"]
        for field, value in analyses[identifier]["metadata"].items():
            report.append(f"- {field}: {value}")
        for field, evidence in analyses[identifier]["features"].items():
            report.append(f"- {field}: {evidence[:500]}")

    # %% 11. Review leakage beyond section headings: factual passages can disclose payments, requests and outcomes.
    report += ["", "## 5. Czy rozdzielenie opisu i uzasadnienia chroni przed wyciekiem?", "",
               "CZĘŚCIOWO. Podział semantyczny pomaga, ale nie zapewnia bezpieczeństwa. Poniżej diagnostyka porównawczego wariantu nagłówków.",
               "Kwoty/ocena świadczenia są zakazane na wejściu. Sama zamiana kwot na maskę jest niewystarczająca.", "",
               "| ID (przykłady) | Akapity faktów | Zawierające kwoty lub sygnały wycieku |", "|---|---:|---:|"]
    for identifier in dict.fromkeys([*list(study)[:8], *confirmed]):
        analysis = analyses[identifier]
        facts = analysis["sections"]["fakty"].splitlines()
        contaminated = [p for p in facts if MONEY.search(p) or LEAKAGE.search(p)]
        report.append(f"| {identifier} | {len(facts)} | {len(contaminated)} |")
        if identifier in confirmed and contaminated:
            report.append(f"\nOdrzucony fragment ID {identifier}: {contaminated[0]}\n")
    report += ["", "Pozostają: pośrednie wskazówki wyniku, cytowane precedensy, ocena sądu i błędne granice sekcji.",
               "Brak nagłówków nie blokuje semantycznej analizy akapitów. Same nagłówki są tylko punktem odniesienia.",
               "Potrzebna ręczna ocena kategorii, osoby, negacji i dostępności faktów w chwili predykcji.",
               "Powiązane postępowania/apelacje muszą trafić do jednego splitu przed uczeniem reprezentacji."]
    split_count = sum(bool(a["sections"]["fakty"]) for a in study.values())
    money_count = sum(bool(a["amounts"]) for a in study.values())
    report += ["", "## Odpowiedzi końcowe", "",
               f"1. Dostępność: w {money_count}/{len(study)} dokumentów próbki dekady znaleziono kwoty PLN; samo trafienie nie potwierdza targetu.",
               f"2. Struktura: parę jawnych granic faktów/rozważań znaleziono w {split_count}/{len(study)} dokumentów. Brak nagłówków nie blokuje semantycznego podziału akapitów.",
               f"3. Target: {len(confirmed)} ręcznie odczytane przykłady potwierdzono cytatami; automatyczne propozycje podziału są oddzielone od wzmiankowanych liczb i nadal wymagają oceny.",
               f"4. Cechy: model analizował {model_document_count} dokumentów; do oceny pól dopuszczono fragmenty z {len(processed)}. Metadane i klasy siedzib obliczono dla całej poprawnej próbki. Nie dowiedziono skuteczności predykcji.",
               "5. Wyciek: sam podział sekcji NIE wystarcza. Nie powstaje tutaj zbiór uznany za bezpieczny do treningu.",
               "", "Pobieranie ograniczone liczbą stron nie daje próby losowej; nie wyliczamy odsetka użytecznych danych w całym SAOS.",
               "Pełne propozycje targetów, cech i cytaty do własnego przeglądu: analysis.json (nie zbiór treningowy).",
               "Snapshoty i daty pobrania: sąsiednie pliki *.meta.json. Zapisane odpowiedzi są ponownie używane.",
               "Źródła: oficjalna dokumentacja SAOS podana na początku skryptu; odnośniki do konkretnych wyroków powyżej."]
    output = "\n".join(report) + "\n"
    (OUTPUT / "report.md").write_text(output, encoding="utf-8")
    print("\n" + output)
    print(f"Raport: {OUTPUT / 'report.md'}")
