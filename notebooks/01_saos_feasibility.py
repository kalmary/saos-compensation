"""Badanie danych: uv run --extra extraction notebooks/01_saos_feasibility.py.

Wynik: odpowiedzi na pięć pytań, dwa targety, cytaty i opisy cech.
To badanie wykonalności, nie gotowy ekstraktor etykiet ani zbiór do treningu.
Zmiany badawcze pozostają w tym pliku; odpowiedzi API i raport trafiają do tmp/.
"""

# %% 1. Czego szukamy i dlaczego właśnie tak?
# Szukamy WYROKÓW zawierających zasądzone zadośćuczynienie z kwotą w PLN.
# Samo słowo + kwota nie wystarcza: kwota może być żądana, odrzucona,
# wcześniej wypłacona, dotyczyć innej osoby, kosztów procesu lub precedensu.
# Badamy osobno zadośćuczynienie, pozostałe odszkodowanie i ich sumę.
# Suma dotyczy jednego poszkodowanego i jednego rozstrzygnięcia;
# nie obejmuje odsetek, kosztów procesu/sądowych ani rent okresowych.
# Pozostałe odszkodowanie może obejmować leczenie, opiekę, utracony dochód itd.
# Docelowo trzeba też zdecydować, czy chodzi o dopłatę czy całe świadczenie
# razem z wcześniejszymi wypłatami. Tutaj przykłady dotyczą kwot z tego wyroku.
#
# Parametry wynikają z dokumentacji i są sprawdzane z żywym queryTemplate:
# https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przeszukiwania-danych
# https://www.saos.org.pl/help/index.php/search-query-language
# https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przegladania-danych
# W dokumentacji courtType miesza rodzaj organu z poziomem sądu;
# odpowiedź API rozróżnia courtType=COMMON i ccCourtType=DISTRICT/REGIONAL/APPEAL.
# Nie ma udokumentowanego filtra „zasądzone zadośćuczynienie w PLN”.
# Dlatego porównujemy zapytania, a znaczenie kwot sprawdzamy w pełnym tekście.

import asyncio
import hashlib
import importlib
import json
import random
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
BASE_URL = CONFIG["api"]["base_url"].rstrip("/")

# %% 2. Pobieranie i HTML: zachowujemy surowy JSON oraz pochodzenie danych.
# Cache pozwala eksperymentować bez ponownego pobierania tych samych wyroków.
# Daty pobrania są w *.meta.json; różne zapytania mogą mieć różne daty snapshotu.
# Aby odświeżyć badanie, przenieś tmp/saos_feasibility/ w inne miejsce.
# Błędy HTTP/JSON i niezgodne parametry zatrzymują badanie, nie znikają po cichu.
# Z HTML usuwamy znaczniki, zachowując akapity i nagłówki do podziału tekstu.

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


def search_settings(config, today=None):
    end = date.fromisoformat(str(config["date_range"]["to"])) if config["date_range"]["to"] else today or date.today()
    lower = config["date_range"]["from"]
    years = config["date_range"]["years"]
    if type(years) is not int or years <= 0:
        raise ValueError("Liczba lat musi być dodatnią liczbą całkowitą")
    start = date.fromisoformat(str(lower)) if lower else end.replace(
        year=end.year - years, day=min(end.day, monthrange(end.year - years, end.month)[1]))
    if start > end:
        raise ValueError("Początek zakresu dat jest późniejszy niż koniec")
    request = config["request"]
    if type(request["pageSize"]) is not int or not 10 <= request["pageSize"] <= 100:
        raise ValueError("pageSize musi należeć do 10..100")
    if type(request["pageNumber"]) is not int or request["pageNumber"] < 0:
        raise ValueError("pageNumber musi być nieujemną liczbą całkowitą")
    if not isinstance(request.get("all"), str) or not request["all"].strip():
        raise ValueError("Brak zawężającej frazy wyszukiwania")
    sampling = config["sampling"]
    if not sampling["court_levels"] or any(level not in {"DISTRICT", "REGIONAL", "APPEAL"} for level in sampling["court_levels"]):
        raise ValueError("Nieprawidłowe poziomy sądów w próbce")
    if type(sampling["seed"]) is not int or type(sampling["examples_per_query"]) is not int or not 0 <= sampling["examples_per_query"] <= request["pageSize"]:
        raise ValueError("Nieprawidłowy dobór próbki")
    for key in ("additional_judgment_ids", "judgment_ids"):
        if not isinstance(config[key], list) or any(type(value) is not int or value <= 0 for value in config[key]):
            raise ValueError(f"Nieprawidłowe ID: {key}")
    params = {**request, "judgmentDateFrom": start.isoformat(), "judgmentDateTo": end.isoformat()}
    if not isinstance(config["queries"], dict) or not config["queries"] or not all(isinstance(v, dict) for v in config["queries"].values()):
        raise ValueError("Warianty zapytań muszą być niepustą mapą filtrów")
    queries = {name: {**params, **overrides} for name, overrides in config["queries"].items()}
    return start, end, params, queries


def yearly_ranges(start, end):
    lower, index = start, 1
    while lower <= end:
        year = start.year + index
        boundary = start.replace(year=year, day=min(start.day, monthrange(year, start.month)[1]))
        upper = end if boundary >= end else boundary - timedelta(days=1)
        yield lower, upper
        lower, index = upper + timedelta(days=1), index + 1


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
        # Łamanie linii w źródle HTML nie tworzy akapitu wewnątrz <p>/<span>.
        self.parts.append(re.sub(r"\s+", " ", data))


def extract_text(html):
    parser = JudgmentText()
    parser.feed(html)
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


# %% Metadane sądu i miejscowości: dane urzędowe, bez zgadywania wielkości miasta.
# GPKG jest bazą SQLite, więc do jej odczytu nie potrzeba biblioteki GIS.
# WERSJA opisuje wersję obiektu BDOT10k, NIE rok pomiaru ludności.
# shortcut: używamy jednego snapshotu GUGiK, do badań historycznych potrzebne są stany z lat wyroków.
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
    # Nazwa sądu jest zwykle w miejscowniku („w Piszu”). Morfeusz rozpoznaje
    # odmianę nazw, a kandydaci nadal pochodzą wyłącznie z urzędowej bazy.
    # Wieloznaczne dopasowanie zostaje brakiem; nie wybieramy największej miejscowości.
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



# %% 3. Gdzie w wyroku szukać kwoty, faktów i rozważań prawnych?
# Typowy układ: nagłówek -> sentencja -> UZASADNIENIE -> stanowiska stron
# -> ustalenia faktyczne -> rozważania prawne. Nie jest to gwarantowany format.
# „UZASADNIENIE” obejmuje często ZARÓWNO opis sprawy, JAK I ocenę wysokości.
# Szukamy jawnych granic „Sąd ... ustalił:” oraz „Sąd ... zważył:”.
# Gdy granic nie ma lub są wielokrotne, nie zgadujemy bezpiecznej części.
# shortcut: rozpoznajemy wybrane jawne nagłówki, rozszerzyć po ręcznej ocenie pominiętych układów.
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


# %% 4. Liczbowe wzmianki PLN: wyłącznie badanie dostępności tekstu.
# Wykaz nie jest tabelą targetów. Rozstrzygnięcie odczytuje osobno moduł 03.
# shortcut: zapis słowny, skróty typu „4 mln zł” i inne waluty wymagają przeglądu.
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


# %% 5. Cechy: sześć szerokich pól z konfiguracji, wypełnianych treścią źródła.
# Nie przypisujemy objawów do pól słownikami regex. Model w 03 wybiera
# oryginalne fragmenty i zachowuje ich offsety oraz prawdopodobieństwa.
# Brak wykonanej analizy nie oznacza braku informacji w źródle:
# status przetworzenia przechowujemy osobno od wartości pola.
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


# %% 6. Małe sprawdzenia reguł, bez pobierania danych i bez dodatkowych plików.
# Ważne przypadki: grosze, zera, brak sekcji, żądanie vs wynik, brak != nie.
assert extract_text("<p>A <span>B</span>.</p><p>Żółć&nbsp;123</p>") == "A B.\nŻółć 123"
assert [row["kwota"] for row in amounts("15.000,50 zł; 2 000 PLN; 2000zł; 0 zł", "fakty")] == ["15000.50", "2000", "2000", "0"]
assert not amounts("4 mln zł; 15 EUR; 20.10.2020", "fakty")
assert split_sections("WYROK\noddala powództwo")["fakty"] == ""
assert split_sections("WYROK\nUZASADNIENIE\nSąd ustalił:\nUraz.\nSąd zważył:\nOcena.")["fakty"] == "Uraz."


# %% 7. Eksperyment wyszukiwania: parametry mają jawny cel, a nie są domysłem.
# all: temat; COMMON: sądy powszechne; SENTENCE: wyroki, nie postanowienia.
# SENTENCE nie gwarantuje uzasadnienia; bywa ono osobnym rekordem REASONS.
# pageSize=10: mała próbka, nie pobranie korpusu; pageNumber=0: pierwsza strona.
# DATABASE_ID + ASC: łatwo powtórzyć kolejność, nie jest to chronologia.
# Dla większej próby losujemy strony w rocznych przedziałach i poziomach sądu.
# „zadośćuczynienie zasądza zł” zawęża tekst, ale może trafić w żądania i koszty.
# keywords zawęża metadane, lecz może zgubić wyroki bez poprawnego tagu.
# Nie wybieramy legalBase=445 na stałe: wykluczyłoby to inne podstawy roszczeń.
# Parametry są lokalnym warsztatem badawczym, nie interfejsem eksperymentu ML.
if __name__ == "__main__":
    import argparse

    import morfeusz2

    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--metadata-only", action="store_true", help="Pomiń próby modeli lokalnych")
    options = arguments.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    localities, locality_source = load_localities()
    morphology, courts = morfeusz2.Morfeusz(), {}
    start_date, today, params, queries = search_settings(CONFIG["api"])
    sampling = CONFIG["api"]["sampling"]
    report = ["# Badanie SAOS: zadośćuczynienie, kwoty i cechy", "", "## 1. Co i jak wyszukujemy?", ""]
    report.append(f"Zakres dat orzeczenia: {start_date}–{today} (według configs/dataset/api.yaml).")
    sample, sources = {}, {}
    for name, query in queries.items():
        print(f"Zapytanie: {name}", flush=True)
        result = search_judgments(query)
        url = requests.Request("GET", BASE_URL + "/search/judgments", params=query).prepare().url
        metadata = json.loads((OUTPUT / f"{hashlib.sha256(url.encode()).hexdigest()}.meta.json").read_text())
        report.append(f"- {name}: {result['info']['totalResults']} trafień; all={query['all']!r}; pobrano {metadata['acquired_at']}.")
        # Każde zapytanie wnosi dwa przykłady, ale deduplikujemy po SAOS ID.
        for item in result["items"][:sampling["examples_per_query"]]:
            sample[item["id"]] = item
            sources.setdefault(item["id"], []).append(name)
    report += [
        "", f"Parametry bazowe: {json.dumps(params, ensure_ascii=False)}.",
        "To filtry tematu, nie potwierdzenie zasądzenia. Liczby nakładają się i NIE oznaczają użytecznych etykiet.",
        "Sprawdzono zastosowanie parametrów w queryTemplate odpowiedzi. Brak filtra semantycznego dla naszego targetu.",
    ]
    # Około 300 wyroków: 10 rocznych przedziałów x 3 poziomy sądu x 10 wyników.
    # W każdej warstwie losujemy jedną PEŁNĄ stronę, z ustalonym ziarnem.
    # To próba grupowa: wyniki na stronie mogą pochodzić z jednego sądu.
    # Nie nadaje się do prostego szacowania udziału użytecznych wyroków w SAOS.
    # Roczne przedziały są rozłączne; ostatni obejmuje również dzisiejszą datę.
    # Poziom sądu nie jest gwarancją pierwszej instancji; apelację ustala się z tekstu.
    random_pages = random.Random(sampling["seed"])
    strata = []
    print("Próbka ostatnich 10 lat: roczne przedziały, wszystkie trzy poziomy sądów...", flush=True)
    for lower, upper in yearly_ranges(start_date, today):
        for level in sampling["court_levels"]:
            query = {**params, **sampling["query"], "ccCourtType": level,
                     "judgmentDateFrom": lower.isoformat(), "judgmentDateTo": upper.isoformat()}
            result = search_judgments(query)
            total = result["info"]["totalResults"]
            page = random_pages.randrange(max(1, total // query["pageSize"]))
            if page:
                query = {**query, "pageNumber": page}
                result = search_judgments(query)
            strata.append((lower, upper, level, total, page, len(result["items"])))
            print(f"  {lower}–{upper} {level}: {total} trafień, strona {page}", flush=True)
            for item in result["items"]:
                sample[item["id"]] = item
                sources.setdefault(item["id"], []).append(f"{lower}–{upper}/{level}/strona {page}")
    study_ids = set(sample)
    report += ["", "### Pobierana próbka a wszystkie trafienia", "",
               "Wyszukiwanie liczy wszystkie trafienia. Pobieramy tylko próbkę pełnych dokumentów.",
               "Jedna losowa pełna strona na warstwę; ostatnia niepełna strona może być pominięta.",
               "| Od | Do | Poziom sądu | Trafienia API | Wylosowana strona | Pobrane ID |",
               "|---|---|---|---:|---:|---:|"]
    for lower, upper, level, total, page, count in strata:
        report.append(f"| {lower} | {upper} | {level} | {total} | {page} | {count} |")
    # Osobny zestaw przykładów do ręcznego objaśnienia, również z ostatnich 10 lat.
    # Przykłady dobrane ręcznie NIE wchodzą do statystyk losowanej próbki.
    for identifier in CONFIG["api"]["additional_judgment_ids"]:
        sample[identifier] = None
        sources.setdefault(identifier, []).append("ręczny przykład; poza statystyką, jeśli nie wylosowany")

    # %% 8. Pełny tekst: search.items.textContent to tylko wycinek z trafieniem.
    # Detail.data.textContent zawiera dokument HTML; target i cechy wyprowadzamy
    # osobno. Puste metadane decision/summary nie zastępują analizy dokumentu.
    analyses, failures = {}, {}
    for identifier in sample:
        print(f"Pełny wyrok {identifier} ({len(analyses) + len(failures) + 1}/{len(sample)})", flush=True)
        record = fetch_json(f"{BASE_URL}/judgments/{identifier}").get("data")
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
        sections = split_sections(text)
        # Gdy nie ma UZASADNIENIA, badamy kwoty w całym tekście, lecz nie nadajemy
        # im roli „zasądzona” na podstawie niepotwierdzonych granic sentencji.
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
    # Próba modelu jest oddzielona od pobrania 300 dokumentów. Wybrane ID są
    # jawne w YAML; nie przedstawiamy ich jako oceny całego zbioru.
    model_run = None
    if not options.metadata_only:
        selected = {}
        for identifier in CONFIG["judgment_ids"]:
            if identifier not in analyses:
                raise ValueError(f"Brak wyroku {identifier} wskazanego w konfiguracji modelu")
            selected[identifier] = analyses[identifier]["text"]
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
    # Pełne wyniki do własnego przeglądu, bez nadawania im statusu danych treningowych.
    (OUTPUT / "analysis.json").write_text(json.dumps({
        "status": "kandydaci do ręcznej weryfikacji", "od": str(start_date), "do": str(today), "api_config": CONFIG["api"],
        "próbka": sorted(study_ids), "błędy": failures, "gugik": locality_source,
        "wyroki": {identifier: {"metadane": a["metadata"], "cechy": a["features"],
                    "wzmianki_kwotowe": a["amounts"], "zasadzenie": a.get("model_extraction", {}).get("award"), "podział_sekcji": bool(a["sections"]["fakty"]),
                    "status_cech": a["feature_status"], "model": a.get("model_extraction"),
                    "dobór": sources[identifier]} for identifier, a in analyses.items()},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Najpierw pokazujemy faktyczny wynik ekstrakcji, dopiero potem diagnostykę.
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
               f"Łącznie przeanalizowano {len(analyses)} dokumentów, w tym dodatkowe przykłady ręczne.",
               "Losowanie stron w rocznych warstwach grupuje sądy; to nie jest prosta próba losowa korpusu.",
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

    # %% 9. Czy można znaleźć target? Kandydaci to jeszcze nie etykiety.
    # Raport pokazuje kilka przykładów; propozycje targetów i cytaty w analysis.json.
    # Nie wybieramy pierwszej/najwyższej kwoty jako targetu.
    # Dwa ręcznie odczytane przykłady dalej pokażą, kiedy target ma uzasadnienie.
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

    # Ręczna interpretacja jest jawna i oddzielona od wyników regexów.
    # Cytaty są wyszukiwane w odpowiedniej sekcji i sprawdzane przy każdym runie.
    # To przykłady wykonalności, nie pomiar dokładności automatycznej ekstrakcji.
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
    # Negatywny przykład: kwota na cel społeczny nie jest świadczeniem dla powoda.
    if 553925 in analyses:
        report += ["", "ID 553925 — pułapka beneficjenta: " + quote(analyses[553925]["text"], r"20.000 złotych", 100),
                   "Rozważania wskazują Fundację jako odbiorcę. Nie przypisujemy tej kwoty powodowi jako jego zadośćuczynienia."]

    # %% 10. Czy jest opis pozwalający wnioskować o kwocie / dodatkowych kosztach?
    # Raportujemy każdą kategorię, w tym brak. To cytaty-kandydaci, nie embeddingi.
    # Dla przejrzystości szczegóły dwóch spraw; liczby dostępności dla całej próbki.
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
        report += ["", f"Próba modelu: {CONFIG['backend']}; ID: {CONFIG['judgment_ids']}.",
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

    # %% 11. Czy opis sprawy można oddzielić od uzasadnienia kwoty? Test wycieku.
    # Oddzielenie sekcji jest dopiero początkiem. Również ustalenia faktyczne mogą
    # zawierać wypłaty, żądania i inne kwoty. Z kolei rozważania powtarzają fakty.
    # Dowód: liczymy skażone akapity i pokazujemy odrzucony fragment dla przykładu.
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
               "", "Próbka stron jest warstwowa/grupowa; nie wyliczamy z niej odsetka użytecznych danych w całym SAOS.",
               "Pełne propozycje targetów, cech i cytaty do własnego przeglądu: analysis.json (nie zbiór treningowy).",
               "Snapshoty i daty pobrania: sąsiednie pliki *.meta.json. Zapisane odpowiedzi są ponownie używane.",
               "Źródła: oficjalna dokumentacja SAOS podana na początku skryptu; odnośniki do konkretnych wyroków powyżej."]
    output = "\n".join(report) + "\n"
    (OUTPUT / "report.md").write_text(output, encoding="utf-8")
    print("\n" + output)
    print(f"Raport: {OUTPUT / 'report.md'}")
