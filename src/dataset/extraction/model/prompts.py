"""Pinned extraction defaults, prompts and feature definitions."""


DEFAULTS = {
    "version": "saos-extraction-22",
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
