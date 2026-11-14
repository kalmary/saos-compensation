# Od orzeczenia do rekordu — plan prezentacji downloadera

## Cel i granice

Prezentacja HTML w języku polskim dla osób rozumiejących analizę danych, ale nieznających implementacji. Wyjaśnia aktualny przepływ danych, zadania dwóch modeli, przygotowanie ich wejść, interpretację odpowiedzi i warunki zapisania rekordu. To prezentacja koncepcyjna, bez kodu Pythona, list funkcji, instrukcji instalacji i szczegółów bibliotek. Czas: około 20–25 minut, 16 slajdów. Docelowy katalog: `presentation/downloader/`; plik wejściowy: `index.html`.

Stan opisywany: konfiguracja i implementacja `saos-extraction-30`, odczytane 14 listopada 2026, z jednym świadomym wyjątkiem: prezentacja pokazuje docelową politykę „oddalenie → cel 0 zł”. Kod nadal odrzuca takie sprawy; zmiana czeka na wdrożenie. Na slajdzie 7 umieścić dyskretną notę „Polityka 0 zł: do wdrożenia”. Pozostałe etapy opisywać zgodnie z implementacją. Źródłem faktów są bieżący kod i `PROJECT.md`; wcześniejsze wpisy `STATUS.md` opisują także zastąpione warianty. Nie przenosić do prezentacji starszych twierdzeń o wymaganiu większości pól ani o weryfikowaniu celu przez Basal.

Centralna teza: downloader tworzy oddzielnie **cel regresji** i **opis faktów**, a następnie sprawdza, czy mogą znaleźć się w jednym rekordzie. Nie przewiduje jeszcze odszkodowania.

## Język wizualny: redakcyjny atlas danych

### Płótno i siatka

- Format 16:9, projekt bazowy 1600 × 900 px; cały slajd skaluje się proporcjonalnie do okna, bez obcinania treści. Przy wąskim ekranie nadal dostępny tryb slajdowy z czytelnymi kontrolkami oraz opcjonalny widok wszystkich slajdów.
- Marginesy: 88 px lewo/prawo, 60 px góra, 64 px dół. Treść zasadnicza od y=220 do y=788. Szerokość użytkowa 1424 px; 12 kolumn po 96,67 px z odstępami 24 px.
- Górna etykieta rozdziału: x=88, y=60, 15 px, kapitaliki, tracking 0,14 em. Tytuł: x=88, y=100, maks. 2 wiersze, 58 px / 1,06. Podtytuł: 24 px / 1,4, maks. 100 znaków. Slajd z dwuwierszowym tytułem zaczyna treść od y=270.
- Stopka na y=842: cienka linia, lewy napis „SAOS / POZYSKIWANIE DANYCH”, po prawej numer `01 / 16`; 13 px. Kontrolki poza płótnem lub w obszarze stopki, bez nakładania się na treść.
- Rytm odstępów: 8 / 16 / 24 / 40 / 56 px. Karty tylko tam, gdzie przedstawiają odrębny obiekt danych; bez siatki identycznych kart na każdym slajdzie.

### Kolor

- Papier: `#F4F0E7`; atrament: `#172C32`; drugorzędny tekst: `#536466`; linie: `#C8CEC7`.
- Ścieżka celu: cegła `#A84632`, jasne tło `#F1DED5`.
- Ścieżka cech: morski `#166B69`, jasne tło `#DBE8E2`.
- Źródło i infrastruktura: stal `#647885`, jasne tło `#E4E9EA`.
- Ostrzeżenie / odrzucenie: ochra `#946319`, jasne tło `#F2E6CA`.
- Slajdy 1, 3 i 16 na ciemnym tle `#172C32`, tekst `#F4F0E7`; akcenty na nich: jasna cegła `#E4A18A` i mięta `#89C9BA`. Pozostałe na papierze.
- Kolor nigdy nie jest jedynym oznaczeniem: każda ścieżka ma nazwę; odrzucenie ma etykietę i symbol ×, akceptacja ✓. Bez gradientów i ozdobnego glassmorphism.

### Typografia i grafika

- Tytuły: Georgia / „Times New Roman”, serif; bez kursywy poza wyróżnionymi cytatami. Tekst i etykiety: systemowy sans-serif (`-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, sans-serif). Nazwy pól i krótkie wartości: `ui-monospace`, SFMono-Regular, Consolas, monospace.
- Tytuły 58 px, teza okładki 82 px, treść 26 px / 1,35, drobne podpisy diagramów 20 px / 1,3, noty na slajdzie min. 18 px. Nie zmniejszać tekstu w celu upchania treści; przenieść szczegóły do notatek.
- Diagramy jako HTML/CSS lub inline SVG, ostre po skalowaniu; linie 2 px, groty 10 px, zaokrąglenie prostokątów 8 px. Duże numery etapów 64 px z 15% krycia. Brak zdjęć, logotypów usług, emoji, ilustracyjnych stocków.
- Przy każdym diagramie krótka tekstowa etykieta opisująca jego znaczenie. Diagram ma ułatwiać śledzenie danych, a nie udawać kod.
- Cytaty w przykładach są **schematyczne**, podpisane „Przykład schematyczny — nie wynik uruchomienia”. Nigdy nie pokazywać wymyślonego score jako pomiaru modelu.

## Interakcje i sposób oddania

- Jedna strona HTML, 16 sekcji. Lokalny CSS i niewielki JS; bez frameworka, CDN, zewnętrznych fontów i zapytań sieciowych. Prezentacja działa przez otwarcie pliku oraz lokalny serwer.
- Strzałka w prawo / Space / PageDown: następny; w lewo / PageUp: poprzedni; Home/End: pierwszy/ostatni. Nie przechwytywać klawiszy w polach formularzy. Hash `#slide-1`…`#slide-16` wskazuje slajd i umożliwia linkowanie.
- Widoczne przyciski „Poprzedni”, „Następny”, licznik i „Spis slajdów”. Przycisk „Pełny ekran” używa standardowego Fullscreen API i pozwala wrócić do okna; jeśli API niedostępne, ukryć przycisk. Przycisk „Notatki” otwiera panel z komentarzem i źródłami dla bieżącego slajdu. Escape zamyka panel; zarządzanie fokusem i widoczny focus. Spis: lista 16 tytułów, aktywny zaznaczony. Nie używać automatycznego przewijania ani autoplay.
- Strzałki przy skrajnych slajdach są wyłączone. Przy zmianie slajdu aktualizować tytuł dostępności i krótki region live; pozostałe slajdy nie są osiągalne tabulatorem w trybie prezentacji.
- Opcjonalne przejście 150 ms opacity; `prefers-reduced-motion` wyłącza je. Żadnych animowanych etapów ujawniania treści: cały slajd czytelny od razu.
- Druk: wszystkie slajdy widoczne, 16 stron w układzie poziomym 16:9 (160 mm × 90 mm), po jednym slajdzie na stronę, zachowane kolory, brak kontrolek i notatek, żadnego przycinania. Systemowy dialog wydruku wystarcza; nie implementować eksportera PDF.
- Pliki: `index.html`, `styles.css`, `script.js` (jeśli potrzebny); bez dodatkowych zależności. Jeśli używane są funkcje JS, każda ma krótki angielski opis, args i returns w docstringu JSDoc jako odpowiedniku dokumentacji funkcji.

## Slajdy — dosłowna treść i kompozycja

### 01. Od orzeczenia do rekordu

**Etykieta:** „SAOS COMPENSATION / DOWNLOADER”

**Tytuł:** „Od orzeczenia\ndo rekordu”

**Podtytuł:** „Jak tekst uzasadnienia staje się cechami i celem regresji”

**Mały podpis:** „Bieżący przepływ danych · ekstrakcja v30 · 14.11.2026”

**Kompozycja:** ciemne tło. Tytuł x=88, y=220, szer. 850; podtytuł y=430. Po prawej x=1080, y=205 pionowy rysunek trzech arkuszy, 300 × 360: arkusz „ORZECZENIE” rozdziela się na dwa wyraźne zakończenia „FAKTY” (mięta) i „KWOTA” (jasna cegła). Na dole y=680 wiersz 36 px: „Dwa modele. Dwa różne zadania. Jeden kontrolowany rekord.” Bez wykresu prognozy.

**Notatki:** zakres kończy się na zapisaniu zbioru do przeglądu. Regresor, trening i ocena predykcyjna nie są częścią tego uruchomienia.

**Źródła:** `PROJECT.md` → Delivery plan 1; `README.md` → How get-data pipeline works.

### 02. Co ma powstać z jednego wyroku?

**Etykieta:** „01 / PRODUKT”

**Tytuł:** „Fakty i kwota mają różne role”

**Treść dwóch kolumn:**

Lewa, morska, „WEJŚCIE PRZYSZŁEGO MODELU”:
- „Co spowodowało szkodę?”
- „Jakie były jej skutki?”
- „Jakie czynności były utrudnione?”
- „Kontekst sprawy: rok, instancja, miejscowość sądu, liczba poszkodowanych, wcześniejsza wypłata.”

Prawa, ceglana, „CEL REGRESJI”:
- Duża wartość typograficzna: „kwota w PLN”
- „Wyodrębniona z wybranego fragmentu orzeczenia.”
- „Zapisywana osobno od opisu faktów.”

**Dolna teza:** „Opis nie może zdradzać odpowiedzi, której model ma się nauczyć.”

**Kompozycja:** dwie kolumny 840/536 px z pionową linią x=960. Cztery pytania/wiersze po lewej, bez ramek; po prawej jeden duży blok kwoty. U dołu linia oddzielająca tezę (28 px). Nie nazywać wyjściowych rekordów zatwierdzonym zbiorem treningowym.

**Notatki:** `prior_payment` jest świadomie dopuszczonym predyktorem liczbowym, a nie celem; tekstowy opis nie zawiera jego dowodu monetarnego. Docelowe zadanie jest regresją. Obecny parser nie sumuje wszystkich rozstrzygnięć w całym dokumencie.

**Źródła:** `configs/dataset/acquisition/features.yaml`; `extraction/quality.py`; `storage/records.py:write_result` (ścieżki skrócone względem `src/dataset/`).

### 03. Dwa modele, dwa rodzaje odpowiedzi

**Etykieta:** „02 / MODELE”

**Tytuł:** „MiniLM porównuje. Basal wydobywa i ocenia.”

**Lewa kolumna:** „MiniLM” / „paraphrase-multilingual-MiniLM-L12-v2” / „Tekst → wektor → podobieństwo do wzorców” / „Wskazuje fragmenty o określonej roli: zasądzenie, żądanie, wcześniejsza wypłata lub wykluczenie.”

**Prawa kolumna:** „Basal” / „Remek/basal-2.0-4.5B” / „Tekst + pytania → fragmenty źródła lub oceny” / „Wydobywa cechy i sprawdza ich faktyczność, trafność oraz ryzyko ujawnienia celu.”

**Dolny wyróżniony tekst:** „Basal jest tutaj modelem językowym obsługującym zadania ekstrakcji i oceny — nie ma osobnego etapu „LLM → Basal”.”

**Kompozycja:** ciemne tło; pionowy podział w połowie; nazwy 48 px; ikony tylko proste: wektor jako rząd krótkich pionowych kresek, ekstrakcja jako podświetlony wiersz dokumentu. Modele nie są narysowane szeregowo. Dół y=700, szer. 1320, 26 px.

**Notatki:** „Basal” oznacza wybrany checkpoint oraz lokalny silnik jego wywołań. W kodzie ten sam serwer odpowiada na typy `extract`, `choice`, `noul`. Nie przedstawiać go jako chatbota swobodnie piszącego streszczenia ani jako dodatkowego modelu po LLM. Basal nie klasyfikuje i nie potwierdza bieżącego celu pieniężnego. Oba modele używają obecnie MPS, wersje wag są przypięte, wagi są lokalnie cache’owane. Alternatywny wspierany checkpoint mini istnieje, lecz nie jest aktywny; nie sugerować porównania jakości ani przewagi bez pomiarów.

**Źródła:** `configs/dataset/acquisition/basal.yaml`; `targets.yaml`; `src/dataset/clients/basal.py`; `extraction/model/decisions.py`; `extraction/targets/target_extraction.py`.

### 04. Cała droga danych

**Etykieta:** „03 / PRZEPŁYW”

**Tytuł:** „Najpierw źródło, potem dwa odrębne wyniki”

**Diagram, dosłowne węzły:**
1. „SAOS / wyroki + metadane” → 2. „Selekcja i czysty tekst” → 3. „MiniLM / odsiew oddaleń i wybór fragmentów” → 4. „Kwota / odczyt liczbowy”.
Z węzła 3 odgałęzienie w dół: „Basal / cechy i ich kontrola”. Z bocznego małego źródła „SAOS + GUGiK” strzałka do „Metadane”. Kwota, cechy i metadane zbiegają się w „Warunki przyjęcia” → „Pliki rekordu + checkpoint”.
Od etapu 3 alternatywna ceglana ścieżka „Oddalenie → 0 zł” łączy się z torem celu; dalszy tor cech pozostaje potrzebny. Przy tej ścieżce mały dopisek „polityka docelowa”.

**Pod diagramem:** „Brak celu w trybie zwykłym kończy pracę przed ekstrakcją cech.”

**Kompozycja:** diagram ma dwie poziome ścieżki: cel y=345 (cegła), cechy y=570 (morski). Źródła x=88; modele x=520; wyniki x=940; zapis x=1280. Nazwy węzłów najwyżej 3 krótkie wiersze. Nie rysować cech jako bezwarunkowo równoległych do celu.

**Notatki:** debug może kontynuować po nierozwiązanym celu, jeśli wybrano zdanie zasądzenia; brak dopasowania zasądzenia pomija ekstrakcję tekstu i nie pozwala zachować przypadku. Oddalenie w pokazywanej polityce ustala cel 0 zł, ale nie zwalnia z zebrania poprawnych cech. Obecny kod kończy ścieżkę odrzuceniem; to jedyny świadomie docelowy element diagramu. Wzbogacanie końcowe następuje po decyzji o zachowaniu rekordu.

**Źródła:** `acquisition/pipeline.py:process`; `extraction/basal_filtering.py:extract`.

### 05. Co rzeczywiście trafia do analizy?

**Etykieta:** „04 / ŹRÓDŁO”

**Tytuł:** „Zrzut SAOS jest szerszy niż nasz zbiór”

**Treść w czterech kolejnych wierszach:**
1. „Zakres dat” — „Domyślnie 10 lat; granice zostają utrwalone przy pierwszym uruchomieniu.”
2. „Wyroki sądów powszechnych” — „Lokalny filtr rodzaju sądu i orzeczenia.”
3. „Kandydaci do analizy” — „Tekst zawiera rdzeń „zadośćuczyn” oraz liczbową kwotę PLN.”
4. „Poprawne źródło” — „Spójne ID, data, sygnatura, sąd i wydział; rozpoznawanie już zapisanych rekordów.”

**Prawy podpis:** „Dane są pobierane stronami. Liczba pobranych wyroków ≠ liczba zachowanych przypadków.”

**Kompozycja:** cztery poziome pasy x=88, szer. 1000, wys. 105; numery 01–04 po lewej. Po prawej wąska kolumna 330 px z pionową linią i podpisem. Żadnych procentów lub lejkowych szerokości sugerujących niezmierzoną skuteczność.

**Notatki:** oficjalne `/api/dump/judgments`, metadane `/api/dump/commonCourts`. Strona do 100 wyroków; brak zdalnego filtra tekstowego/sądu. Szersze pojęcie „compensation” w celu projektu nie oznacza, że obecny filtr zbiera wszystkie sprawy odszkodowawcze. Limit stron nie jest losowaniem próby. Błędy transportu/schematu zatrzymują pracę; nie wszystkie nieprawidłowe rekordy są wyjątkami — część jest odrzucana z powodem w podsumowaniu.

**Źródła:** `acquisition/download.py`; `acquisition/validation.py`; `acquisition/settings.py`; `clients/api.py`.

### 06. Jeden tekst, zachowana możliwość powrotu do źródła

**Etykieta:** „05 / PRZYGOTOWANIE TEKSTU”

**Tytuł:** „Usuwamy opakowanie, zachowujemy treść i granice”

**Trzy etapy:** „HTML z SAOS” → „Tekst z akapitami” → „Zdania z pozycją w źródle”.

**Przykład schematyczny:**
- panel początkowy: dokument z etykietami „nagłówek”, „sentencja”, „UZASADNIENIE”, „ustalenia”, „rozważania”;
- panel środkowy: te same sekcje jako czysty tekst;
- panel końcowy: podświetlone zdanie „Po zdarzeniu poszkodowana nie mogła wykonywać dotychczasowej pracy.” oraz etykieta „dokładny fragment źródła”.

**Teza dolna:** „Każda wybrana treść ma wskazać miejsce w orzeczeniu — nie tylko brzmieć wiarygodnie.”

**Kompozycja:** trzy arkusze 400 × 390 px, x=88/600/1112 (ostatni szer. 400), y=270; strzałki między nimi. Tekst demonstracyjny max. 22 px; pozostałe wiersze jako neutralne linie. Nie pokazywać surowych tagów, indeksów i regexów.

**Notatki:** normalizacja dekoduje encje, usuwa znaczniki i porządkuje białe znaki, zachowując akapity. Pozycje odnoszą się do tekstu znormalizowanego. Dzielenie respektuje m.in. inicjały i przecinki dziesiętne. To nie jest OCR ani generowanie streszczenia.

**Źródła:** `extraction/source/text.py`; `extraction/source/passages.py`.

### 07. Pierwsze pytanie: czy sprawa została oddalona?

**Etykieta:** „06 / MINILM — ROZSTRZYGNIĘCIE”

**Tytuł:** „Oddalenie oznacza cel 0 zł”

**Diagram:** „Wszystkie pełne zdania” + „12 wzorców oddalenia” → „MiniLM: porównanie znaczenia” → romb „podobieństwo > 0,7?”; TAK → „Cel: 0 zł”; NIE → „Szukaj kwoty zasądzonej”. Obie gałęzie prowadzą dalej do ścieżki przygotowania cech.

**Widoczna nota:** „Bez filtra słów kluczowych i bez wymogu kwoty w zdaniu.”

**Dolny pas:** „Brak zasądzenia jest informacją o wyniku. Opis faktów nadal musi przejść kontrolę.”

**Dyskretna nota w stopce:** „Polityka 0 zł: do wdrożenia”.

**Kompozycja:** dwa wejścia x=88, y=285/445; porównanie x=530, y=365; decyzja x=960; wyniki x=1220, y=300/500. Pod diagramem jedna mała karta ryzyka: „Cytat lub opis wcześniejszego postępowania może prowadzić do błędnej etykiety 0 zł.”

**Notatki:** ten slajd pokazuje zamierzoną zmianę polityki zbioru. Bieżący kod v30 odrzuca oddalenia i nie uruchamia dla nich dalszej ekstrakcji; prezentacja nie zmienia kodu. Odsiew modelowy przebiega partiami 64 zdań i zatrzymuje się na pierwszej partii z trafieniem; zachowuje najsilniejsze trafienie w tej partii, a nie globalny najlepszy wynik dokumentu. Podobieństwo nie jest prawdopodobieństwem. Bez niezależnego sprawdzenia autora wypowiedzi i etapu postępowania. Wdrożenie polityki 0 zł będzie musiało umożliwić dalszą ekstrakcję cech i ich normalną kontrolę.

**Źródła:** `embeddings/target_matching.py:match_denial`; `targets.yaml`; `extraction/basal_filtering.py`; `acquisition/pipeline.py`. Semantyka 0 zł jest planowaną polityką, nie zachowaniem tych plików.

### 08. Jak przygotowujemy zdania do porównania?

**Etykieta:** „07 / MINILM — WEJŚCIE”

**Tytuł:** „Porównujemy rolę wypowiedzi, nie wysokość kwoty”

**Treść i przykład:**
1. „Bierzemy początek zdania do pierwszej części zawierającej kwotę PLN.”
2. „Słowa kluczowe określają, z którymi grupami wolno je porównać.”
3. „Na potrzeby embeddingu zastępujemy kwoty neutralnym „100 zł”.”

**Środkowy przykład, dwie wersje:**
„Sąd zasądza na rzecz powódki 25 000 zł, wraz z odsetkami…”
↓ „fragment do porównania”
„Sąd zasądza na rzecz powódki 100 zł,”

**Dolny podpis:** „Oryginalna kwota i tekst pozostają zachowane do późniejszego odczytu.”

**Kompozycja:** lewa kolumna 550 px zawiera 3 kroki; prawa 790 px zawiera duży cytat na jasnej cegle, przecinek jako widoczna granica. Odcinek po granicy w oryginale szary, etykieta „poza wybranym prefiksem”. Nie przypisywać przykładowi konkretnego score ani zwycięskiej klasy.

**Notatki:** granicą jest koniec części oddzielonej przecinkiem, która jako pierwsza zawiera liczbowe PLN; nie zawsze znak bezpośrednio po kwocie. Jeśli pierwsza część pieniężna nie ma właściwych słów, program nie szuka dalej w tym zdaniu. Wzorce mają neutralizowane placeholdery stron i kwot. Dłuższy tekst kodowany jest przez wszystkie okna tokenów, bez porzucania końca; szczegół 128 tokenów/okno nie jest potrzebny w widocznym slajdzie.

**Źródła:** `embeddings/target_matching.py:candidate_sentences,prepare_text`; `embeddings/encoding.py`.

### 09. Z czym model porównuje fragment?

**Etykieta:** „08 / MINILM — DECYZJA”

**Tytuł:** „Sześć ról konkuruje o ten sam fragment”

**Diagram:** centralny arkusz „Kandydat ze źródła”; wokół sześć nazw: „Zasądzenie”, „Żądanie”, „Wcześniejsza wypłata”, „Koszty postępowania”, „Kontekst mieszany”, „Inne płatności”. Zasądzenie ceglane, żądanie stalowe, pozostałe ochrowe z etykietą „poza celem”.

**Trzy reguły po prawej:**
- „Wygrywa najwyższe podobieństwo wśród dopuszczonych ról.”
- „Wynik musi być ściśle większy niż 0,7.”
- „Z całego dokumentu zostaje najlepsze trafienie każdej roli.”

**Dolna teza:** „Podobieństwo semantyczne nie jest pewnością, że to prawidłowy cel.”

**Kompozycja:** diagram radialny w lewej części 820 × 470 px, reguły w prawej 500 px. Linie tylko do dopuszczonych ról pokazać schematycznie, podpis „porównanie z wzorcami”; bez arbitralnych słupków wyników.

**Notatki:** grupa judgment wymagana, request opcjonalna i nie trafia do predyktorów. Przy dokładnym remisie priorytet ma klasa wykluczająca. Wzorce kodowane raz i ponownie używane. Jest 93 wzorców ról pieniężnych oraz 12 odrębnych wzorców oddalenia. Nie jest to uczenie klasyfikatora na tych wyrokach.

**Źródła:** `embeddings/target_matching.py:TargetMatcher.match`; `configs/dataset/acquisition/targets.yaml`.

### 10. Kwotę odczytuje reguła, nie Basal

**Etykieta:** „09 / CEL”

**Tytuł:** „Wybrany fragment musi dać jednoznaczny wynik”

**Duża sekwencja:** „Wybrane zasądzenie” → „Dokładnie jedna liczbowa kwota PLN” → „Wartość celu”.

**Pod spodem trzy krótkie przypadki:**
- „Kwota łączna” → „Odczyt wartości z tekstu.”
- „Po X zł na osobę” → „Mnożenie tylko przy jawnej liczbie odbiorców.”
- „Kilka kwot lub niejasna grupa” → „Cel nierozwiązany.”

**Dolny pas:** „Oddzielne zasądzenia w innych zdaniach nie są obecnie sumowane.”

**Kompozycja:** główny przepływ y=280, trzy przypadki jako wiersze tabeli y=450–690. Prawy górny mały badge „bez wywołania Basal”.

**Notatki:** parser używa dokładnej arytmetyki dziesiętnej. Kwoty zapisane wyłącznie słownie nie są obsługiwane. Wybór prawidłowej roli pozostaje ograniczeniem podobieństwa; parser nie naprawia błędnej selekcji. Domyślna wartość injured_party_count=1 nie służy do odgadywania mnożnika. Żądana kwota może zostać odczytana pomocniczo, lecz nie wchodzi do award.json. Nie twierdzić, że cel zawsze jest sumą całego wyroku ani że jest zatwierdzany przez drugi model.

**Źródła:** `extraction/targets/target_extraction.py`; `extraction/source/text.py:amount_in_pln`; `extraction/quality.py:target_rejection_reason`.

### 11. Co dostaje Basal przy ekstrakcji cech?

**Etykieta:** „10 / BASAL — WEJŚCIE”

**Tytuł:** „Fragment uzasadnienia i trzy konkretne pytania”

**Lewa kolumna „TEKST”:**
„Ustalenia faktyczne sądu”
„Jeśli brak nagłówka: treść uzasadnienia.”
„Partie do 6000 znaków, z zachowaniem zdań i mapowania do źródła.”

**Prawa kolumna „PYTANIA”:**
- `cause` — „Co spowodowało szkodę? Jaka jest przyczyna zdarzenia?”
- `damage` — „Jakiej szkody doznał poszkodowany? Jakie są skutki zdarzenia dla poszkodowanego?”
- `limitations` — „Jakie czynności były utrudnione lub niemożliwe? Jaki wpływ na poszkodowanego ma doznana szkoda?”

**Pod obiema:** „Wybierz fragment źródła. Zachowaj negację, osobę i czas. Nie dopisuj faktów. Brak odpowiedzi jest dozwolony.”

**Kompozycja:** dokument po lewej 520 px, po prawej pytania 820 px; jeden morski łącznik prowadzi z obu do podpisu „jedno żądanie ekstrakcji na partię”. Trzy pytania w 24 px, klucze 18 px. Zachować dokładne brzmienie pytań z konfiguracji.

**Notatki:** wszystkie wybrane pola pytane razem. To wybór cytatu/fragmentu, nie generowane streszczenie. Wstępna selekcja sekcji jest regułowa; faktyczność ocenia później model. Brak drugiego przebiegu szukającego poza podstawową sekcją. Nie zakładać, że cały wybrany blok zawiera wyłącznie ustalenia sądu.

**Źródła:** `extraction/source/passages.py:feature_input,extraction_batches`; `extraction/features.py:extract_features`; `features.yaml`; `extraction/model/prompts.py`.

### 12. Co zwraca Basal i jak wiążemy to ze źródłem?

**Etykieta:** „11 / BASAL — ODPOWIEDŹ”

**Tytuł:** „Odpowiedzią jest wskazany fragment, nie nowy opis”

**Diagram trzech kolumn:**
„Fragment źródła” → „Pole + tekst + pozycja + ocena” → „Pełne zdanie do kontroli”.

**Przykład schematyczny:** źródło: „Przed zdarzeniem powód pracował fizycznie. Po zdarzeniu nie mógł dźwigać ciężarów.”; wskazanie w środkowej kolumnie: `limitations` / „nie mógł dźwigać ciężarów”; końcowa wartość: „Po zdarzeniu nie mógł dźwigać ciężarów.”

**Dwie kontrole pod diagramem:** „Czy wskazany tekst jest dokładnie w wejściu?” / „Czy odpowiedź jest kompletna i nie została ucięta?”

**Dolna nota:** „Brak dowodu pozostaje brakiem informacji.”

**Kompozycja:** trzy panele po 440 px, y=280–620; podświetlenie konkretnej frazy w środkowym. Nie pokazywać JSON na tym slajdzie; nazwy elementów odpowiedzi wystarczą.

**Notatki:** `extract` zwraca data i fields, z offsetami oraz confidence/miarami wyboru. Kod sprawdza zgodność `state[start:end] == value`. Odzyskuje maksymalnie dwa pełne zdania przecinające wybrany fragment, by nie usuwać negacji/podmiotu. Kandydaci z partii są porządkowani według line_probability; niska confidence ekstrakcji sama nie odrzuca cytatu, bo decyduje późniejszy przegląd. Walidacja formatu, offsetów i rozkładów nie dowodzi prawdziwości interpretacji.

**Źródła:** `extraction/model/decisions.py:check_decision`; `extraction/features.py:describe_record,extract_features`.

### 13. Drugi kontakt z Basal: kontrola kandydata

**Etykieta:** „12 / BASAL — WALIDACJA”

**Tytuł:** „Wybrany cytat musi przejść trzy pytania”

**Pasek wejścia:** „Do 600 znaków przed” | **„OCENIANY FRAGMENT”** | „Do 600 znaków po”. Podpis: „Sąsiedzi służą ustaleniu kontekstu; oceniamy tylko fragment.”

**Trzy pionowe kolumny:**
1. „Czy to ustalone fakty?” — „Wybór kategorii: fakty, stanowiska, rozstrzygnięcie, rozważania lub inne.” — „Wymagane: fakty i ocena ≥ 0,6”.
2. „Czy ujawnia cel?” — „Kwoty, wynik sprawy lub uzasadnienie wysokości — także bez liczby.” — „Wymagane: ryzyko wycieku ≤ 0,35”.
3. „Czy odpowiada na pytanie pola?” — „Ta sama treść pytania co przy ekstrakcji.” — „Wymagane: trafność ≥ 0,6”.

**Dolna teza:** „Nie przechodzi? Sprawdzamy kolejny kandydat albo pozostawiamy brak.”

**Kompozycja:** wejście y=250, wys. 90; trzy kolumny 440 px y=385, separatory pionowe. Progi 32 px morski; brak ozdobnych wskaźników/gauge. Mała dodatkowa etykieta y=755: „Niezależnie: opis z rozpoznaną walutą PLN jest odrzucany.”

**Notatki:** komplet kategorii obejmuje również mieszany i niejednoznaczny. To pytania choice/noul, nie porównywanie embeddingów z przykładami. Noul odpowiada na pytanie binarne oceną true/false; choice zwraca kategorię i rozkład. Rozkłady sprawdzane na kompletność, zakres, sumę i zgodność wyniku. Te same źródłowe zakresy współdzielą review między polami. Próg 0,6 nie oznacza zmierzonej 60% trafności na tych wyrokach. Źródłowe opisy mogą nadal zawierać błędy przypisania do osoby.

**Źródła:** `extraction/features.py:select_facts,extract_features`; `extraction/model/prompts.py`; `extraction/model/decisions.py`; `basal.yaml`.

### 14. Skąd pochodzą pozostałe cechy?

**Etykieta:** „13 / KONTEKST SPRAWY”

**Tytuł:** „Nie wszystko wymaga modelu językowego”

**Tabela, dokładne kolumny:** „Cecha” | „Źródło / sposób uzyskania” | „Ważna granica”.

Wiersze:
- „Rok wyroku” | „Data z metadanych SAOS” | „Bez interpretacji tekstu”.
- „Instancja” | „Basal: początek wyroku i dowód etapu postępowania” | „Nie z samej nazwy sądu”.
- „Wcześniejsza wypłata” | „MiniLM wybiera kontekst, Basal wskazuje kwotę” | „0 oznacza także brak potwierdzenia”.
- „Liczba poszkodowanych” | „Jawni odbiorcy wybranego zasądzenia” | „Domyślnie 1 przy braku ustalenia”.
- „Wielkość miejscowości sądu” | „Siedziba sądu + dane GUGiK” | „Bieżący stan, nie populacja z roku wyroku”.

**Kompozycja:** tabela x=88, y=265, szer.1424; kolumny 320/650/454; wiersze po 88 px, cienkie linie, 23 px. Komórki Basal lekko morsko podkreślone, bez mocnego tła całej tabeli.

**Notatki:** instancja: nagłówek przed UZASADNIENIEM, najwyżej 4000 znaków, wybór pierwsza/druga/nieustalona i próg 0,6. Wcześniejsza wypłata: dokładny span kwoty głównej, confidence ≥0,6, jednoznaczna suma zamiast dodawania wzmianek, zaokrąglenie half-up do pełnego PLN; nigdy nie dodawana do celu. Ten przepływ pokazuje przypadek, gdzie wynik MiniLM rzeczywiście jest wejściem Basal. Lokalność sądu nie jest miejscem zamieszkania powoda ani miejscem zdarzenia.

**Źródła:** `extraction/basal_filtering.py`; `extraction/targets/amounts.py:extract_prior_payment`; `acquisition/metadata.py`; `acquisition/localities.py`.

### 15. Kiedy powstaje rekord i co zostaje na dysku?

**Etykieta:** „14 / ZAPIS”

**Tytuł:** „Przyjęcie rekordu nie oznacza jeszcze poprawnej etykiety”

**Lewa kolumna „WARUNKI”:**
- „Co najmniej jedno źródłowo potwierdzone pole tekstowe.”
- „Kontrola trafności i wycieku dla każdego wypełnionego opisu.”
- „W trybie zwykłym: rozpoznany, dopuszczalny cel.”

**Prawa kolumna „PLIKI”:**
`record.json` — „wartości cech”
`award.json` — „total + valid”
„Tylko debug: source.txt + model_calls.json”

**Dolny diagram trwałości:** „Zapis przypadku” → „Aktualizacja checkpointu” → „Następny wyrok”. Pod nim: „Po przerwaniu wracamy do utrwalonego postępu.”

**Mała nota:** „Debug może zachować nierozwiązany cel jako valid=false. Nie omija wymogu poprawnego opisu. Polityka oddalenie → 0 zł wymaga wdrożenia.”

**Kompozycja:** lewa szer. 690, prawa 650; pionowa linia między. Nazwy plików 28 px monospace, opisy 24 px. Diagram u dołu y=690. `valid=true` nie ma zielonej pieczęci „prawda”.

**Notatki:** brak judgment match pomija ekstrakcję tekstu, więc nie daje zachowanego rekordu nawet w debug. Pending przechowuje ukończone decyzje również bez debug i znika po sukcesie/odrzuceniu. Checkpoint utrwala daty, konfigurację i snapshot GUGiK; niezgodna zmiana wymaga nowego katalogu. Limit obejmuje istniejące pliki, pending i zapisy tymczasowe; cache wag jest osobno. Szczegółów checkpointu nie rozbudowywać na slajdzie ponad jedną linię.

**Źródła:** `extraction/quality.py`; `storage/records.py`; `storage/checkpoints.py`; `acquisition/pipeline.py`; `acquisition/download.py`.

### 16. Co wiemy — i czego trzeba jeszcze dowieść?

**Etykieta:** „15 / WNIOSKI”

**Tytuł:** „Mamy kontrolowany proces. Potrzebujemy pomiaru jakości.”

**Trzy tezy, dosłownie:**
1. „MiniLM rozpoznaje kontekst. Cel to odczytana kwota albo 0 zł przy oddaleniu.”
2. „Basal wydobywa dowody i ocenia ich przydatność.”
3. „Przed treningiem trzeba ręcznie sprawdzić reprezentatywną próbę.”

**Dolne dwie kolumny:**
„Najważniejsze ryzyka” — „Błędna rola płatności · utracony kontekst · błędne rozpoznanie oddalenia · przypisanie faktów niewłaściwej osobie”.
„Następny krok” — „Osobno zmierzyć poprawność kwot, cech, odrzuceń i ochrony przed wyciekiem.”

**Kompozycja:** ciemne tło, tytuł 58 px; trzy duże numerowane linie 32 px y=300/400/500. Dolne kolumny y=650, nagłówki 18 px uppercase, tekst 22 px. Ostatni numer 16/16. Bez przycisku wdrożenia, CTA marketingowego ani twierdzeń o osiągniętej skuteczności.

**Notatki:** testy offline sprawdzają zachowanie kodu, nie precyzję ekstrakcji na zbiorze sądowym. Historyczne odtworzenia pokazywały błędne role celu, ale dotyczyły starszych wersji i nie są pomiarem aktualnego v30. Potrzebna próba z ręcznymi etykietami. Dopiero później grupowany podział danych, dopasowanie preprocessingu wyłącznie na treningu, baseline regresji i MAE/RMSE. PostgreSQL/pgvector i pełna orkiestracja treningu nie są wykonywane przez downloader.

**Źródła:** `PROJECT.md` → Risks/Delivery plan; `README.md` → Current limitations and remaining work; `STATUS.md` → ustalenia jakościowe i ograniczenia walidacji.

## Kontrola merytoryczna przed odbiorem HTML

- Wszędzie rozdzielone MiniLM, Basal i deterministyczny parser. Basal to model językowy w tym przepływie, nie dodatkowy etap po innym LLM.
- Rozpoznawanie oddalenia przed wyborem pieniężnym; pełne zdania, bez PLN/keyword gate. Prezentowana polityka: oddalenie → 0 zł i dalsze wymagania dotyczące cech. Dyskretna nota wskazuje, że ta zmiana czeka na wdrożenie; kod pozostaje niezmieniony.
- Kwoty neutralizowane tylko na potrzeby embeddingu, oryginał pozostaje do odczytu.
- Jeden zwycięzca roli na fragment, wykluczenia konkurują, próg cosinusowy ściśle >0,7; score nie nazywany prawdopodobieństwem.
- Ekstrakcja cech poprzedza review kandydatów; nie twierdzić, że Basal najpierw klasyfikuje każdy fragment całego wyroku.
- Wymagane jedno pole tekstowe, a nie dwa z trzech i nie większość.
- Progi 0,6 / 0,35 / 0,6 mają poprawną funkcję; próg prior_payment nie dotyczy celu.
- Brak agregacji osobnych zdań zasądzeń; domyślna liczba osób nie naprawia nierozwiązanego celu per-person.
- Debug nie gwarantuje zachowania każdego odrzucenia. `valid=true` nie jest ręczną certyfikacją etykiety.
- Wszystkie przykłady oznaczone jako schematyczne; brak wymyślonych wyników, score, wydajności, liczb zachowanych spraw i jakości modeli.

## Kontrola prezentacji przed odbiorem

Otworzyć lokalnie i obejrzeć każdy slajd w 1600×900 oraz 1280×720. Sprawdzić brak nakładania, kontrast, polskie znaki, zawijanie długich nazw modeli, strzałki i drobne podpisy. Przejść wszystkie slajdy klawiaturą, sprawdzić skrajne stany, bezpośredni hash, spis, panel notatek, Escape i przywracanie fokusu. Sprawdzić układ wydruku wszystkich 16 stron. Bez pobierania danych SAOS, pobierania wag lub uruchamiania modeli: prezentacja opisuje implementację, nie jest demonstracją inferencji.
