# Wymagania badania wykonalności SAOS

## 1. Cel i zakres

Celem jest ocena, czy orzeczenia SAOS pozwalają zbudować zbiór do regresji kwoty zasądzonego zadośćuczynienia, z odrębnym rozpoznaniem pozostałego odszkodowania. Badanie ma dostarczać dowodów z dokumentów, a nie tylko liczby wyników wyszukiwania.

- Uwzględniać wyroki z ostatnich 10 lat. Zapisać dokładne daty początku i końca oraz datę pobrania danych.
- Nie ograniczać docelowego zbioru do jednego poziomu sądu lub obszaru geograficznego. Te właściwości mają być potencjalnymi cechami.
- Rozpocząć od sądów powszechnych: rejonowych, okręgowych i apelacyjnych.
- Zachować charakter problemu regresyjnego. Kategoryzacja służy uporządkowaniu targetów i cech.
- W badaniu oddzielać możliwość odczytania informacji, jakość automatycznej ekstrakcji i skuteczność przyszłego modelu. Jedno nie dowodzi pozostałych.

## 2. Forma prototypu

- Pobieranie danych, metadane i raport badania utrzymywać w `01_saos_feasibility.py`.
- Lokalny eksperyment polskiego modelu decyzyjnego utrzymywać w `03_system_one_polish.py`.
- Generatywny wariant awaryjny Gemmy przez pakiet Ollama utrzymywać w `04_gemma.py`; uruchamiać dopiero po ocenie polskiego modelu i udostępnieniu lokalnych wag Gemmy.
- Skrypt może mieć formę kolejnych sekcji badawczych; nie wymaga architektury aplikacji produkcyjnej.
- Dodawać komentarze wyjaśniające cel funkcjonalny sekcji, badane pytanie i ograniczenia zastosowanych reguł.
- Stosować możliwie prosty kod i istniejące zależności.
- W `configs/dataset/basal.yaml` konfigurować wyłącznie nazwy wybieranych cech, nazwy targetów i trzy progi pewności. Model, instrukcje i sposób cięcia tekstu są stałe w kodzie. `configs/dataset/api.yaml` zawiera tylko `max_age_years`, `max_pages` i `debug`; zero stron oznacza wszystkie wyniki. Manifest notebooków wskazuje oba pliki niezależnie od katalogu roboczego.
- Pełną próbę uruchamiać przez `uv run --no-sync notebooks/01_saos_feasibility.py`; wariant bez modeli przez `uv run --no-sync notebooks/01_saos_feasibility.py --metadata-only`.
- Surowe odpowiedzi, metadane pobrania i wyniki zapisywać w projektowym `tmp/`.
- Nie zastępować wyników analizy założonymi z góry odpowiedziami. Ręcznie opracowane przykłady oznaczać osobno.
- Nie wykonywać operacji Git bez osobnej zgody.

## 3. Pytania, na które ma odpowiedzieć program

1. **Dostępność:** czy można znaleźć wyroki, w których występuje zadośćuczynienie z kwotą, i jak odróżnić zasądzenie od samej wzmianki?
2. **Struktura:** jak zbudowane są dokumenty i gdzie szukać sentencji, stanowisk stron, ustaleń faktycznych oraz rozważań sądu?
3. **Target:** czy można ustalić kwotę zasądzoną dla konkretnego poszkodowanego i rozdzielić składniki świadczenia?
4. **Opis sprawy:** czy dokument zawiera informacje mogące służyć do wnioskowania o wysokości zadośćuczynienia i możliwości wystąpienia innych świadczeń?
5. **Wyciek:** czy można oddzielić fakty od fragmentów ujawniających wynik lub uzasadniających wysokość świadczenia?

Odpowiedzi mają być jasne, poparte krótkimi cytatami i identyfikatorami wyroków. Gdy dowody są niewystarczające, wynik ma to wskazywać wprost.

## 4. Wyszukiwanie i zakres pobierania

- Stosować jedno stałe zapytanie SAOS: sądy powszechne, wyroki, fraza „zadośćuczynienie zł”, sortowanie rosnące po ID, 100 wyników na stronę.
- Konfigurować tylko maksymalny wiek danych (`max_age_years`) oraz limit stron (`max_pages`). Zero stron oznacza pobieranie do końca wyników.
- Pobierać strony kolejno; nie losować warstw ani nie dodawać ręcznych ID. Przetwarzać modelem wszystkie poprawne dokumenty z pobranych stron.
- Sprawdzać zastosowane parametry i paginację, zachowywać czas i pełną odpowiedź API. Trafienie zapytania nie potwierdza poprawnego targetu.
- `debug: true` zachowuje dane źródłowe i dodatkowe wyniki pośrednie w folderze danego wyroku. `debug: false` pozostawia podstawowy eksport cech i targetów; istniejących plików diagnostycznych nie usuwać.
- Przypadki rozwojowe trzymać w folderach `fixtures/`; wykrywać je po plikach, bez listy ID w konfiguracji. Zachować oryginalny wyrok, tekst wejściowy, wyniki i dokładne żądania/odpowiedzi modelu. Zapisany wynik modelu nie jest automatycznie ręcznie potwierdzoną etykietą.

## 5. Pobieranie i jakość danych

- Wynik wyszukiwania traktować jako metadane i wycinek tekstu. Pełny dokument pobierać z endpointu szczegółów wyroku.
- Zachowywać oryginalny JSON bez nadpisywania go wersją przetworzoną.
- Zapisywać adres żądania, parametry i czas pozyskania; umożliwić ponowne użycie zapisanej odpowiedzi.
- Walidować identyfikator, datę, sąd, sygnaturę, tekst i strukturę paginacji.
- Deduplikować po stabilnym identyfikatorze SAOS.
- Jawnie raportować błędy i niezgodności; nie ukrywać brakujących dokumentów ani częściowych niepowodzeń.
- Oddzielać poprawność strukturalną od zgodności treści. Poprawny format daty nie dowodzi zgodności daty z nagłówkiem wyroku.
- Uwzględnić możliwość osobnego opublikowania wyroku i jego uzasadnienia.

## 6. Dwa targety zasądzone przez sąd

Ekstraktor odpowiada na pytanie: **co analizowany wyrok zasądził na rzecz konkretnego poszkodowanego?** Nie wymagać sklasyfikowania wszystkich wystąpień kwot w dokumencie jako warunku otrzymania wyniku.

| Target | Zakres |
|---|---|
| `zadoscuczynienie` | Kwota zasądzona za krzywdę |
| `pozostale_odszkodowanie` | Łączna kwota zasądzonych jednorazowych świadczeń za szkodę majątkową |

- Nie dzielić pozostałego odszkodowania na osobne kwotowe kategorie leczenia, rehabilitacji, opieki, dojazdów czy zarobków. Opis sprawy może zawierać te okoliczności, ale target pozostaje jeden.
- `suma` jest wyliczana z dwóch ustalonych targetów i nie jest trzecim niezależnym targetem.
- Koszty procesu, sądowe, zastępstwa prawnego, odsetki i renta okresowa są poza zakresem. Rozpoznawać je tylko w celu wykluczenia; nie tworzyć ich kwotowej klasyfikacji.
- Kwoty żądane, wcześniej wypłacone i przywołane z innych spraw nie są targetami. Nie budować obowiązkowej wieloklasowej anotacji ich wszystkich wystąpień.

### Kolejność ekstrakcji

1. Reguła znajduje zapis liczbowy kwoty PLN. Nie musi dopasować wyrażenia „tytułem…” ani ustalić kategorii.
2. Dla każdej kwoty przygotować całe oryginalne zdanie lub punkt listy, z kwotą oznaczoną jako analizowana. Nie używać arbitralnego okna słów. Dodać poprzedzający kontekst do limitu `context_characters` (aktualnie 600 znaków), aby zachować np. wprowadzenie do listy żądań.
3. Basal otrzymuje zdanie, poprzedzający kontekst, informację o sekcji oraz sentencję jako kontekst rozstrzygnięcia. Jeśli zdanie zawiera kilka kwot, każdą ocenia osobno, oznaczając tylko analizowane wystąpienie. Jedną decyzją przypisuje oznaczoną kwotę do zasądzonego zadośćuczynienia, zasądzonego pozostałego odszkodowania, nierozdzielonego świadczenia głównego, pominięcia albo braku podstaw do decyzji. To nie jest rozbudowana taksonomia świadczeń ani osobna klasyfikacja wszystkich ról kwot.
4. Żądania, wcześniejsze wypłaty, kwoty z innych spraw i koszty postępowania mają być pomijane. Wzmianka o odsetkach po kwocie głównej nie oznacza, że sama kwota główna jest odsetkami.
5. Łączyć powtórzenia tej samej wartości. Każdy z dwóch targetów ustalać niezależnie; przy konflikcie wartości kierować pole do przeglądu. Jawne przypisanie z sentencji ma pierwszeństwo przed wzmianką w uzasadnieniu.
6. Sumę wyliczać dopiero po ustaleniu obu składników. Jeśli rozpoznano sumę zasądzoną w sentencji, sprawdzić zgodność składników. Nie zgadywać brakującego składnika przez odejmowanie ani nie zamieniać braku informacji na zero.
7. Model dodatkowo kontroluje całą propozycję razem z dowodami. Zachować wszystkie decyzje dla poszczególnych kontekstów, także odrzucone, żeby móc ocenić skąd wynika błąd.

### Dowody i ograniczenia

- Dla targetów zachować wartość w PLN, beneficjenta, cytaty, sekcje i offsety oraz wynik pomocniczej kontroli modelu.
- Próg `amount_minimum_probability` dotyczy łącznego przypisania kontekstowej kwoty do zasądzonego świadczenia oraz końcowego potwierdzenia propozycji. Aktualna wartość: 0,6. Niepotwierdzoną propozycję zachować w diagnostyce.
- Nie sumować powtórzeń tej samej kwoty w sentencji i uzasadnieniu. Nie wybierać pierwszej ani największej liczby w całym dokumencie.
- Suma składników musi odpowiadać zasądzeniu dla tej samej osoby. Zgodność arytmetyczna nie zastępuje zgodności rodzaju świadczenia, osoby i rozstrzygnięcia.
- Brak informacji nie jest zerem. Zero wymaga wyraźnego dowodu; prototyp może pozostawić brak zamiast wnioskować z milczenia dokumentu.
- Kwot należnych Fundacji, Skarbowi Państwa lub innym osobom nie przypisywać powodowi.
- Rozpoznawanie kwot nie zależy od formatu numeracji punktów sentencji ani kolejności słów „tytułem zadośćuczynienia” względem liczby. Wiele niejednoznacznych kwot jednego składnika lub wielu beneficjentów wymaga przeglądu.
- Wielu poszkodowanych, kilka odrębnych świadczeń głównych bez jednoznacznej alokacji, apelacje zmieniające wcześniejsze rozstrzygnięcia i waluty obce kierować do przeglądu. Nie rekonstruować automatycznie końcowego wyniku całego postępowania.
- API SAOS nie dostarcza gotowych dwóch targetów. Pola `decision` i `summary` są tekstowe, a wzbogacenie `MAX_REFERENCED_MONEY` oznacza największą wspomnianą kwotę, nie zasądzone świadczenie.
- Pomocniczy wykaz znalezionych liczb może służyć badaniu dostępności danych; nie jest tabelą targetów i nie wymaga kategorii ani roli.

## 7. Potencjalne cechy dotyczące sądu i postępowania

Najpierw uporządkować dane strukturalne, a następnie opis sprawy. Nie zakładać z góry, które cechy wpływają na target.

| Pole | Znaczenie |
|---|---|
| `sad` | Identyfikator i nazwa konkretnego sądu |
| `poziom_sadu` | Rejonowy, okręgowy lub apelacyjny |
| `instancja` | Pierwsza lub druga, ustalana na podstawie wiarygodnych danych o postępowaniu |
| `okreg_sadowy` | Obszar właściwego sądu okręgowego |
| `apelacja_sadowa` | Obszar właściwego sądu apelacyjnego |
| `rodzaj_wydzialu` | Np. cywilny, karny lub pracy |
| `data_wyroku` | Data rozstrzygnięcia, umożliwiająca analizę zmian w czasie |
| `klasa_miejscowosci_sadu` | Klasa wielkości miejscowości będącej siedzibą sądu |

- Nie utożsamiać poziomu sądu z instancją. Sąd okręgowy może orzekać w różnych instancjach.
- Nie utożsamiać siedziby lub obszaru sądu z miejscem zdarzenia czy zamieszkania poszkodowanego.
- Nazwę miejscowości zachować w danych źródłowych i do kontroli dopasowania; klasę wielkości przewidzieć jako potencjalną cechę modelu.
- Identyfikatory sądów traktować jako kategorie, nie liczby o znaczeniu ilościowym.

## 8. Miejscowość a klasa wielkości

Preferować urzędowe bazy lub usługi GUGiK, zamiast ręcznie tworzonego słownika nazw i liczebności.

### Klasy

- `wies`
- `miasto_ponizej_20_tys`
- `miasto_20_do_ponizej_100_tys`
- `miasto_100_do_ponizej_200_tys`
- `miasto_200_do_ponizej_500_tys`
- `miasto_co_najmniej_500_tys`
- `brak informacji`

### Zasady dopasowania

- Wykorzystywać oficjalną nazwę, rodzaj miejscowości i identyfikatory, np. PRNG/SIMC oraz dane administracyjne pozwalające rozróżniać powtarzające się nazwy.
- Najpierw zweryfikować rzeczywisty schemat wybranej bazy GUGiK. Sama obecność miejscowości w PRNG nie potwierdza dostępności jej liczby mieszkańców.
- Sprawdzić możliwość wykorzystania danych BDOT10k dotyczących miejscowości i liczby mieszkańców.
- Jeśli wybrane źródło nie zawiera odpowiedniej liczby ludności, wskazać brak i opisać potrzebę uzupełnienia z urzędowych danych statystycznych, zamiast zgadywać klasę.
- Zachować źródło, datę pozyskania oraz rok lub datę aktualności danych o ludności, o ile są dostępne.
- Przy analizie historycznej preferować dane odpowiadające dacie wyroku; zastosowanie jednego współczesnego stanu dla całej dekady wymaga jawnego oznaczenia ograniczenia.
- Nie klasyfikować miejscowości jako wsi wyłącznie na podstawie małej liczby mieszkańców. Rodzaj miejscowości wymaga potwierdzenia.
- Klasyfikować siedzibę sądu jako miejscowość, nie dzielnicę ani cały obszar właściwości.
- Nie wymuszać jednoznacznego dopasowania przy wielu kandydatach. Nierozstrzygnięte dopasowanie oznaczać jako `brak informacji` dla cechy, zachowując informację diagnostyczną o przyczynie.

## 9. Wysokopoziomowe kategorie opisu sprawy

Głównym zadaniem modelu jest **ekstrakcja treści do zdefiniowanej formy**: dla każdego wyroku ma powstać rekord z sześcioma krótkimi opisami albo wartością `brak informacji`. Sprawdzenie obecności informacji, jej pewność i statystyka pokrycia są pomocnicze; nie zastępują tego rekordu.

Schemat jest roboczym podziałem analitycznym, a nie oficjalnym katalogiem sekcji lub kryteriów stosowanych przez wszystkie sądy. Ma działać w różnych sprawach, np. dotyczących urazu, śmierci bliskiego lub naruszenia dobrego imienia. Nie może zakładać, że każda krzywda wymaga leczenia.

| Pole | Pytanie |
|---|---|
| `zdarzenie_i_okolicznosci` | Co się wydarzyło i w jakich okolicznościach? |
| `rodzaj_doznanej_krzywdy` | Na czym polegało naruszenie dotyczące tej osoby i czego doświadczyła? |
| `przebieg_nasilenie_i_czas_trwania` | Jak krzywda się przejawiała, jak była dotkliwa, jak długo trwała i czy ustąpiła? |
| `konsekwencje_dla_zycia` | Co zmieniło się w funkcjonowaniu, pracy, relacjach lub samodzielności i jakie skutki pozostały? |
| `indywidualne_okolicznosci_poszkodowanego` | Jaki był wiek, wcześniejsza sytuacja, relacje i inne istotne okoliczności osobiste? |
| `podjete_dzialania_i_potrzebna_pomoc` | Jak łagodzono skutki i jakiego leczenia, wsparcia lub innej pomocy potrzebowano? |

### Wymagany wynik ekstrakcji

- Każde z sześciu pól ma mieć wartość tekstową. Treścią ma być krótki, konkretny opis dotyczący tej sprawy, nie odpowiedź „tak/nie”, prawdopodobieństwo, liczba trafień ani kopia całego uzasadnienia.
- Dla lokalnego modelu decyzyjnego dopuszczalny jest opis ekstrakcyjny: krótki fragment lub niewielki zestaw zdań skopiowanych ze źródła. Nie wymagać swobodnej parafrazy, której taki model nie generuje.
- Można dodatkowo wybierać wartości z jawnie zdefiniowanych kategorii, jeśli słownik rzeczywiście pasuje do różnych spraw. Kategorie nie zastępują opisów i nie mogą być dopisywane bez dowodu.
- Gdy model nie wydobędzie treści do pola, wpisać `brak informacji`, zachowując diagnostykę przyczyny. Nie wymuszać wypełnienia wszystkich pól.
- Cytaty, offsety, pewność, odrzucenia i kontrolę wycieku przechowywać obok rekordu, jako metadane. Użytkownik ma móc najpierw przeczytać sam rekord.
- Dla każdego badanego wyroku zapisywać czytelny rekord JSON. Raport ma pokazywać jego rzeczywiste wartości; tabela pokrycia typu `2/2` może być wyłącznie dodatkiem.

- Następstwa psychiczne mogą pojawić się w opisie krzywdy, jej przebiegu lub konsekwencji; nie tworzyć z nich obowiązkowej odrębnej kategorii głównej.
- Leczenie i rehabilitacja są przykładami treści pola o działaniach i pomocy, nie obowiązkowymi kategoriami głównymi. Nie rozdzielać dublujących się pól „następstwa” i „wpływ na życie”.
- Nie narzucać rozbudowanej specjalistycznej taksonomii objawów i urazów.
- Nie przypisywać samodzielnie ocen typu „lekka/ciężka krzywda”. Zachowywać ustalenia sądu i ich źródło.
- To roboczy schemat porządkowania treści, którego uniwersalność trzeba sprawdzić na zróżnicowanych wyrokach.
- Zachować dostatecznie dużo informacji o leczeniu, pomocy, pracy i ograniczeniach życia, aby później badać możliwość wystąpienia pozostałego odszkodowania.
- Opis leczenia nie dowodzi poniesienia kosztów, zgłoszenia roszczenia ani zasądzenia ich zwrotu.

## 10. Dowody i brak informacji

- Dla wyodrębnionego opisu zachować cytat źródłowy, identyfikator wyroku i miejsce w dokumencie.
- Dla nieustalonej cechy używać dokładnie wartości `brak informacji`.
- Wyraźne stwierdzenie braku danego zjawiska zachować jako informację z dokumentu. Nie utożsamiać go z brakiem informacji.
- Rozróżniać twierdzenia strony, ustalenia sądu, opinie przywołane w sprawie oraz cytaty z innych orzeczeń.
- Sprawdzać, kogo dotyczy informacja, kiedy wystąpiła i czy została zanegowana.
- Samo trafienie regexu nie jest potwierdzonym faktem. Reguły wyszukujące fragmenty mogą wspierać przegląd, ale nie zastępują oceny znaczenia.
- Odróżniać brak dowodu w źródle od nierozpoznania przez ekstraktor; przyczynę braku lub niepewności zachować diagnostycznie.

### Model lokalny i progi

- Najpierw ocenić lokalny polski model podobny do Jev, np. basal oparty na Bieliku. Wykorzystać jego funkcje ekstrakcji według schematu i wyboru krótkich fragmentów, a nie ograniczać zadania do serii pytań o obecność informacji.
- Zamiast słowników specjalistycznych objawów używać pytań semantycznych odpowiadających szerokim kategoriom z sekcji 9.
- Ekstrakcję prowadzić na krótkich porcjach zaakceptowanych faktów, żeby mały model nie musiał wybierać spośród wszystkich zdań wyroku naraz. Wynik dla pola wybrać spośród opisów z tych porcji; zachować także pozostałe odpowiedzi w diagnostyce.
- Oddzielać opis sprawy od rozważań semantycznie na poziomie akapitów, z kontekstem poprzedzającym i następującym. Rozróżniać fakty, stanowiska stron, rozważania, rozstrzygnięcie, treść mieszaną, niejednoznaczną i inne elementy.
- Brak nagłówków nie może blokować analizy akapitów. Podział według nagłówków może pozostać wariantem porównawczym; sam nagłówek „UZASADNIENIE” jest niewystarczający.
- Akapity mieszane podzielić technicznie na zdania i ponownie sklasyfikować albo skierować do przeglądu. Wybrane do rekordu zdanie sprawdzić również osobno, ponieważ ocena całego akapitu może przeoczyć mieszanie faktów i rozważań. Z opisu sprawy wyekstrahować treść sześciu pól, zamiast jedynie stwierdzać ich obecność.
- Zachowywać rozdzielone oryginalne fragmenty opisu sprawy i rozważań oraz przypadki wymagające przeglądu. Dla targetu zachować pełny tekst.
- Progi odczytywać z configs/dataset/basal.yaml. Aktualny minimum_probability wynosi 0,6 i dotyczy dopuszczenia źródła jako faktów oraz oznaczania niepewnych opisów. Niska pewność ekstrakcji ma przede wszystkim oznaczać potrzebę przeglądu, a nie automatyczne usunięcie opisu.
- To eksperyment NLP na małym modelu. Drobne błędy przypisania, pominięcia i niedoskonałość skrótu są akceptowalne na etapie badania; nie budować systemu odrzucającego większość informacji w celu pozornego uniknięcia błędów.
- Nie traktować poprawnej struktury JSON, wysokiego wyniku pewności ani niskiego wyniku pewności jako dowodu poprawności lub niepoprawności opisu. Oceniać rzeczywistą treść na przykładach.
- Aktualny maximum_leak_probability wynosi 0,2. Dla potwierdzenia całego proponowanego rozstrzygnięcia stosować osobno konfigurowany amount_minimum_probability, aktualnie 0,6. Są to kandydaci do przeglądu, nie automatycznie zatwierdzone etykiety regresji.
- Filtr wycieku i decyzje o kwotach mają osobne progi; obniżenie progu cech nie może automatycznie ich poluzować.
- Wartość `brak informacji` uzupełniać statusem: niepewność, odrzucone źródło, brak potwierdzenia w analizowanym tekście albo niewykonanie analizy.
- Zapisywać identyfikator i rewizję wag, wersję silnika, pytania, rozkłady odpowiedzi, offsety cytatów i ustawienia selekcji.
- Pomiar czasu odróżniać od pobierania i ładowania wag; pokazywać wykorzystanie cache oraz zakres faktycznie przeanalizowanego tekstu.
- Gemmie zadawać pytania osobno dla każdej kategorii. Wygenerowane cytaty weryfikować względem źródła; poprawny JSON nie dowodzi poprawności odpowiedzi.

## 11. Rozdzielenie targetu i cech — ochrona przed wyciekiem

- Target i cechy wyodrębniać w osobnych krokach.
- Nie używać pełnego uzasadnienia jako wejścia modelu bez selekcji treści.
- Rozróżniać sentencję, stanowiska stron, ustalenia faktyczne i rozważania prawne. Nagłówek „UZASADNIENIE” sam nie wyznacza bezpiecznego opisu sprawy.
- Sprawdzać, czy w części faktycznej nie występują wypłaty, żądania, kwoty lub inne informacje o wyniku.
- Do predyktorów nie włączać kwoty zasądzonej, żądanej ani wcześniej wypłaconej, wyliczeń opartych na tych kwotach i fragmentów uzasadniających wybór wysokości świadczenia.
- Sama zamiana kwot na maskę nie wystarcza. Opis „uwzględniono żądanie w całości” także może ujawniać wynik.
- Zachować cechy opisujące krzywdę, lecz wykluczać ich prezentację w kontekście uzasadniającym konkretną kwotę.
- Brak rozpoznanych granic sekcji nie uprawnia do uznania całego uzasadnienia za bezpieczne.
- Sprawdzać dostępność faktów w zakładanym momencie predykcji.
- Stosować jawną listę pól dopuszczonych do modelu.
- Powiązane postępowania, apelacje i duplikaty grupować w jednym podzbiorze danych.
- Dane dzielić przed dopasowaniem uczonych reprezentacji; przeglądać cechy i podobne przykłady pod kątem pośredniego ujawniania targetu.

## 12. Oczekiwany wynik i weryfikacja

- Raport ma najpierw pokazywać rzeczywiste rekordy z sześcioma krótkimi opisami, następnie odpowiedzi na pięć pytań z sekcji 3, przykłady i odnośniki do źródeł. Liczebności i miary pewności są dodatkiem.
- Pokazać przykłady pozytywne oraz pułapki: kwotę żądaną, inne świadczenie, mieszaną sumę, innego beneficjenta i brak opisu sprawy, jeśli występują w próbie.
- Oddzielać wyniki automatyczne od przykładów ręcznie zinterpretowanych.
- Pokazywać dostępność poszczególnych pól i ograniczenia ekstrakcji, bez przedstawiania kandydatów jako gotowych danych treningowych.
- Zachować pełniejsze wyniki do samodzielnego przeglądu w `tmp/`.
- Proste sprawdzenia powinny obejmować parsowanie pieniędzy, rozdzielenie składników i ról, brak informacji oraz przypadki potencjalnego wycieku.
- Nie deklarować jakości automatycznej ekstrakcji, braku wycieku ani jakości regresji bez odpowiedniej oceny.

## 13. Elementy wymagające dalszej weryfikacji

- Źródło GUGiK i dostępność liczby ludności wraz z jej aktualnością.
- Skuteczność dopasowania siedzib sądów do miejscowości i klas wielkości.
- Dostępność hierarchii sądów, rodzaju wydziału i danych pozwalających ustalić instancję.
- Uniwersalność wysokopoziomowych kategorii opisu sprawy.
- Jakość ekstrakcji targetów, alokacji do osób i rozdzielania świadczeń.
- Polityka wobec wcześniejszych wypłat, oddalonych roszczeń, wielu poszkodowanych i zmian rozstrzygnięcia w apelacji.
- Bezpieczeństwo zestawu cech oraz ich rzeczywista przydatność predykcyjna.

## Źródła referencyjne

- [SAOS — wyszukiwanie](https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przeszukiwania-danych)
- [SAOS — język zapytań](https://www.saos.org.pl/help/index.php/search-query-language)
- [SAOS — szczegóły wyroków, sądów i wydziałów](https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przegladania-danych)
- [GUGiK — Państwowy Rejestr Nazw Geograficznych](https://www.geoportal.gov.pl/pl/dane/panstwowy-rejestr-nazw-geograficznych-prng/)
- [GUGiK — BDOT10k](https://www.geoportal.gov.pl/pl/dane/baza-danych-obiektow-topograficznych-bdot10k/)
- [GUS — przykład klas wielkości miejscowości](https://stat.gov.pl/files/gfx/portalinformacyjny/pl/defaultaktualnosci/5486/37/1/1/korzystanie_z_transportu_publicznego_przez_gosp._domowe.pdf)
