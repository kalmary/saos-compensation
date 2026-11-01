"""Pinned extraction defaults, prompts and feature definitions."""


DEFAULTS = {
    "version": "saos-extraction-19",
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
                "suma": "Oznaczona kwota to zasądzone w tym wyroku jednorazowe świadczenie główne "
                "dla poszkodowanego: zadośćuczynienie, odszkodowanie albo ich łączna kwota.",
                "pomin": "Żądanie, wcześniejsza wypłata, kwota z innego wyroku, odsetki, renta "
                "lub koszty postępowania; poza zasądzonym świadczeniem głównym.",
                "nieustalone": "Nie można potwierdzić roli oznaczonej kwoty.",
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
            "instructions": "Dla każdego pola wybierz fragment źródła odpowiadający na jego pytanie. "
            "Zachowaj negacje, osobę i czas. Nie dopisuj faktów. Opisuj poszkodowanego. "
            "Pomiń kwoty, wynik sprawy i uzasadnienie wysokości świadczenia. "
            "Nie kopiuj fragmentu do pola, na którego pytanie nie odpowiada. "
            "Jeśli brak odpowiedzi, wybierz not stated.",
            "field_suffix": "",
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
    "event_description": "Co doprowadziło do powstania szkody lub krzywdy?",
    "harm_description": "Na czym polegała szkoda lub krzywda poszkodowanego?",
    "harm_duration": "Jak długo trwały negatywne skutki i czy ustąpiły?",
    "practical_impact": "Jakie czynności, obowiązki lub plany poszkodowanego zostały zakłócone?",
    "claimant_context": "Jaka była sytuacja osobista, zdrowotna lub zawodowa poszkodowanego przed zdarzeniem?",
    "remedial_actions": "Jak próbowano usunąć skutki zdarzenia lub ograniczyć ich uciążliwość?",
}



METADATA_FIELDS = (
    "judgment_year",
    "instance",
    "court_seat",
    "court_locality_size",
    "injured_party_count",
    "prior_payment",
)



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
