# Od orzeczenia do rekordu — plan prezentacji downloadera

## Cel i granice

Polska prezentacja HTML dla kolegów znających analizę danych, lecz nieznających implementacji. Wyjaśnia bieżący przepływ od SAOS do osobnych cech i targetu, zadania dwóch modeli oraz granice jakości. Nie opisuje treningu regresora. Stan odniesienia: ekstrakcja v30, 14 listopada 2026. Świadome warianty przyszłe: pełne oddalenie → target 0 zł oraz wymóg zaakceptowanego, źródłowo potwierdzonego `damage` dla ważnego rekordu. Bieżący kod v30 odrzuca oddalenia i dopuszcza dowolne jedno zaakceptowane pole tekstowe; obie zmiany polityki są do wdrożenia. Slajd 5 wyraźnie oznacza 0 zł jako niewdrożone, a slajdy 2, 11 i 12 wyjaśniają status wymogu szkody. Historyczne pliki w `tmp/test` pochodzą ze starszej konfiguracji i nie mierzą poprawności v30.

## Język wizualny

Płótno 1600 × 900 px, 16:9, proporcjonalnie skalowane. Marginesy 88 px po bokach, 60 px u góry; treść między y≈220 i y≈790, stopka przy y=842. Papier `#F4F0E7`, atrament `#172C32`, tekst poboczny `#536466`, linie `#C8CEC7`. Cel: cegła `#A84632`, cechy: morski `#166B69`, źródło i metadane: stal `#647885`, ostrzeżenie: ochra `#946319`. Ciemne slajdy 1, 3 i 11. Tytuły Georgia 58 px, tekst systemowy sans-serif około 26 px, etykiety 18–20 px, kod/klucze monospace. Tło, etykieta i kierunek strzałki wspólnie oznaczają role; sam kolor nie jest jedyną informacją. Diagramy to HTML/CSS, bez zdjęć, ozdobnych animacji, logotypów lub zależności sieciowych. Nie zmniejszać fontu, by zmieścić nieistotne szczegóły: szczegóły techniczne należą do notatek.

## Interakcje

Jedna strona `presentation/downloader/index.html` z lokalnymi `styles.css` i `script.js`. Strzałka w prawo, spacja i PageDown przechodzą dalej; w lewo i PageUp wracają; Home/End skaczą na końce. Widoczne przyciski, licznik 01 / 13, spis slajdów, panel notatek ze źródłami i pełny ekran. Linki `#slide-1` do `#slide-13`. Dialogi zamykane Escape i przywracające fokus. Druk poziomy, jeden slajd na stronę, zachowane kolory, bez kontrolek. Prezentacja działa przez `file://` i lokalny serwer; bez zapytań zewnętrznych.

## Treść i kompozycja slajdów

### 01. Od orzeczenia do rekordu

Ciemna okładka. Tytuł x=88, y≈195, podtytuł „Jak tekst uzasadnienia staje się cechami i celem regresji” pod spodem. Po prawej arkusz „ORZECZENIE” rozdziela się na „FAKTY” i „KWOTA”. Podpis: „Bieżący przepływ danych · ekstrakcja v30 · 14.11.2026”. Bez dolnego sloganu.

### 02. Szukane wartości

Dwie kolumny: szeroka kolumna pytań o przyczynę, skutki, ograniczenia i kontekst sprawy, z zaakceptowaną szkodą jako wymogiem przyszłego ważnego rekordu; węższa ceglana kolumna celu „kwota w PLN”. Teza w stopie: opis faktów nie może zdradzać odpowiedzi. `prior_payment` jest wyłącznie predyktorem, nie celem.

### 03. Role wykorzystanych modeli

Dwie równorzędne kolumny na ciemnym tle. MiniLM: embedding tekstu i podobieństwo do wzorców ról płatności. Basal: tekst i pytania, źródłowe cytaty oraz osobna ocena ich faktograficzności, trafności i wycieku. Modele mają nazwy checkpointów; Basal nie potwierdza bieżącego celu pieniężnego.

### 04. Przepływ danych

Górny poziomy ciąg: SAOS → selekcja i tekst → MiniLM. Pod nim trzy równoległe, jednakowo widoczne gałęzie: „CEL” → odczyt kwoty → kontrola celu; „CECHY” → pełnowymiarowy blok Basal → opis faktów; „KONTEKST” → SAOS + GUGiK → dane sprawy. Na dole wspólny rekord cech, cel i checkpoint. Strzałki i etykiety pokazują rozgałęzienie i połączenie; Basal nie może wyglądać jak dopisek do celu.

### 05. Poszukiwanie targetu

Trzy duże kroki ze strzałkami: schematyczne zdanie z PLN → rzeczywiste wzorce zasądzenia, żądania i kosztów z `targets.yaml` → embeddingi MiniLM, cosine similarity i wybór roli, po czym odczyt kwoty. Kandydaci pieniężni wymagają kwoty PLN i właściwego słowa; grupa pełnego oddalenia jest sprawdzana osobno w pełnych zdaniach. Próg cosine similarity jest ściśle >0,7 i nie oznacza prawdopodobieństwa. W dole dyskretna nota „Polityka 0 zł: do wdrożenia”.

### 06. Rodzaje ról

Dwukolumnowa lista samych kategorii i dosłownych przykładów wzorców: zasądzenie, żądanie, wcześniejsza wypłata, koszty postępowania, kontekst mieszany, inne płatności i pełne oddalenie. Bez kolejnych reguł wyboru, progów lub pseudoocen.

### 07. Fragment uzasadnienia i trzy konkretne pytania

Po lewej autentyczne pełne zdanie SAOS 276190 o rezygnacji z planowanego urlopu w Chorwacji po wypadku; po prawej rzeczywiste pytania `cause`, `damage`, `limitations` z `features.yaml`. Cytat pochodzi z `source.txt` i jest obecny w żądaniu Basal w `model_calls.json`. Bez technicznej instrukcji ekstrakcji na dole.

### 08. Co zwraca Basal?

Ten sam cytat źródłowy co na slajdzie 7 → pytanie `limitations` → dokładny span „powód zmuszony był zrezygnować z planów urlopowych”. Pokazać offset 56–106 względem partii i ekstrakcyjne `confidence≈0,011` jako miarę wyboru spanu, która sama nie rozstrzyga przyjęcia. Pełne zdanie przechodzi osobną kontrolę; późniejszy wynik zgodności z pytaniem ≈0,933 jest odrębną odpowiedzią. Nie nazywać żadnego z tych wyników zmierzoną trafnością.

### 09. Wybrany cytat oceniany pod trzema aspektami

Górna belka kontekstu przed i po fragmencie. Trzy kolumny: czy to ustalone fakty (kategoria fakty i ocena ≥0,6), czy tekst zdradza cel (ryzyko ≤0,35), czy odpowiada na pytanie pola (trafność ≥0,6). Przykład może być odrzucony i zastąpiony kolejnym kandydatem. Osobno wykrywana waluta PLN w opisie.

### 10. Pozostałe cechy

Pięć prostych par nazwa–znaczenie: rok wyroku / data wydania orzeczenia; instancja / etap postępowania; wcześniejsza wypłata / wypłacona kwota główna; liczba poszkodowanych / odbiorcy zasądzenia; wielkość miejscowości sądu / klasa ludności miejscowości. Bez tabeli metod, źródeł i wyjątków.

### 11. Mamy kontrolowany proces. Potrzebujemy pomiaru jakości.

Ciemny slajd wnioskowy. Trzy duże numerowane stwierdzenia: MiniLM rozpoznaje kontekst, Basal wydobywa i ocenia dowody, potwierdzona szkoda jest obowiązkowa w docelowym rekordzie, przed treningiem potrzebna jest ręczna próba. Ryzyka: błędna rola płatności, utracony kontekst, błędne oddalenie, przypisanie faktów innej osobie. Napis „0 zł przy oddaleniu” wyłącznie jako planowana polityka.

### 12. Poprawnie wydobyta szkoda

Autentyczna sprawa SAOS 276190. W nagłówku pytanie `damage`; w dwóch kolumnach identyczny pełny cytat o złamaniu kości śródręcza i urazach oraz zapisana wartość `damage`. Wartość odpowiada pytaniu i jest wsparta cytatem. Jasne zastrzeżenie: poprawność tej jednej cechy nie potwierdza zapisanej kwoty. To nie jest przykład w pełni poprawnego rekordu. Użyty `record.json` ma `valid=true`, lecz ta flaga nie jest ręczną oceną.

### 13. Błędnie rozpoznany target

Autentyczna sprawa SAOS 275707. Pytanie: ile zasądzono na rzecz poszkodowanej? Pokazać wybrany fragment „po rozliczeniu wzajemnym Sąd zasądził od powódki 470,50 zł na korzyść pozwanego” i `award.json` z `total: 470.50`, `valid: true`. Wniosek: jest to rozliczenie kosztów, którego płatnikiem jest powódka, a odbiorcą pozwany; nie może być targetem kompensacji dla powódki. Nie wyznaczać alternatywnej etykiety bez pełnego przeglądu sprawy.

## Weryfikacja przykładów

Źródła: `tmp/test/276190/{source.txt,record.json,award.json,model_calls.json}` oraz analogiczne pliki SAOS 275707. Sprawdzono dosłowne wystąpienie cytatów w `source.txt`; 276190 ma zapisane pełne zdanie `damage` i `limitations`, a odpowiedź Basal zawiera span 56–106 i późniejszy wynik `fit_limitations=0,9333`. Dla 275707 odpowiedź MiniLM wybrała zdanie z 470,50 zł, a `award.json` oznaczył tę kwotę `valid=true`, mimo odwrotnego kierunku płatności. To przykłady jakościowe ze starszych zapisów, nie pomiar v30.
