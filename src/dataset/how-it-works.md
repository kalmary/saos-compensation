# Jak powstaje rekord zbioru SAOS

## Co robi model

Basal jest tu ekstraktorem informacji, a nie modelem przewidującym wysokość świadczenia. Czyta polski wyrok, rozpoznaje funkcję fragmentów, wybiera źródłowe opisy faktów i interpretuje wzmianki o kwotach. Model regresyjny uczący się przewidywania zadośćuczynienia nie jest jeszcze zaimplementowany.

`configs/dataset/basal.yaml` wybiera model przez identyfikator Hugging Face, np. `Remek/basal-2.0-mini`. Kod przypisuje obsługiwanym modelom konkretne rewizje. Nazwa odpowiada katalogowi `basal_weights/models--Remek--basal-2.0-mini/`. Brakujące pliki są pobierane automatycznie przez bibliotekę Hugging Face przy rozpoczęciu ekstrakcji; sam import pakietu ani PyTorch nie pobiera wag. `.gitkeep` zachowuje katalog w repozytorium; pozostała zawartość jest ignorowana.

Instrukcje, opisy pól i tekst wyroku są po polsku. Nazwy cech są angielskie: `event_circumstances`, `harm_type`, `harm_severity_duration`, `life_impact`, `claimant_circumstances`, `treatment_assistance`. Nazwa klucza schematu nie musi być w języku dokumentu. Celem jest wybór konkretnych zdań źródła, a nie swobodne dopisywanie okoliczności.

## Wejście i pobieranie

Wejściem programu jest konfiguracja YAML z ewentualnymi jawnymi nadpisaniami CLI. SAOS dump ogranicza daty wyroków; wybór sądów powszechnych i tekstów zawierających wzmiankę o zadośćuczynieniu oraz liczbową kwotę PLN odbywa się lokalnie. Pełne wyroki znajdują się już w stronach dumpu. Osobny słownik sądów pozwala połączyć ID wydziału z nazwą sądu.

HTML jest normalizowany do tekstu z granicami akapitów. Program sprawdza ID, datę, typ orzeczenia, sąd, sygnaturę i treść. Powtórzenia ID nie tworzą osobnych rekordów. Trafienie wyszukiwania lub regexu nie oznacza jeszcze poprawnego targetu.

```mermaid
flowchart TD
    A[YAML i jawne argumenty CLI] --> B[Checkpoint i stały zakres dat]
    B --> C[Strona SAOS dump]
    C --> D[Walidacja i wybór tematyczny]
    D --> E[Polski tekst i metadane]
    E --> F[Ekstrakcja faktów]
    E --> G[Oddzielna ekstrakcja targetów]
    F --> H[Kontrola progów jakości]
    G --> H
    H -->|odrzucenie| I[Brak rekordu i diagnostyki przypadku]
    H -->|akceptacja| J[Metadane i zapis record.json]
    J --> K[Opcjonalna historia debug]
    I --> L[Przesunięcie checkpointu]
    J --> L
```

## Ekstrakcja cech

1. Basal określa instancję na podstawie początku dokumentu. Poziom sądu nie zastępuje informacji o instancji.
2. Każdy fragment akapitu jest oceniany z poprzedzającym i następującym kontekstem. Jedno żądanie zawiera pytanie o funkcję fragmentu i pytanie o wyciek informacji o świadczeniu. Funkcje to m.in. fakty, stanowiska stron, rozstrzygnięcie i rozważania.
3. Akapit rozpoznany jako mieszany może zostać podzielony na zdania i oceniony ponownie.
4. Do ekstrakcji trafiają fakty spełniające `minimum_probability` i `maximum_leak_probability`, bez oznaczeń waluty. Sam nagłówek uzasadnienia nie daje takiej gwarancji.
5. Krótkie porcje faktów wypełniają schemat wybranych cech. Wartości są dokładnymi fragmentami źródła. Wybrany fragment jest rozszerzany do pełnego zdania, żeby nie zgubić osoby ani negacji.
6. Kandydaci są porządkowani według wyniku modelu. Wybrane zdania przechodzą ponowną kontrolę funkcji i wycieku. Brak opisu oznacza brak wydobytej informacji, a nie potwierdzenie, że dane zjawisko nie wystąpiło.

`judgment_year` pochodzi z daty SAOS. `court_locality_size` jest klasą populacji miejscowości, w której znajduje się sąd. Nazwa miejscowości jest dopasowywana do GUGiK przez Morfeusza, a klasa wynika z `LICZBAMIESZKANCOW`. Nie jest to populacja obszaru właściwości sądu ani miejsce zamieszkania powoda. Dopasowania niejednoznaczne pozostają brakujące. Nie znamy roku pomiaru ludności, więc współczesnego źródła nie należy traktować jako zweryfikowanych danych historycznych.

## Ekstrakcja kwot

Regex znajduje liczbową kwotę PLN, ale nie przypisuje jej znaczenia. Obsługiwane są m.in. grosze, separatory tysięcy i skale typu `tys.`, `mln`, `mld`. Kwoty zapisane w całości słownie nie są obsługiwane.

Dla każdego wystąpienia Basal otrzymuje całe zdanie lub punkt listy, jedną oznaczoną kwotę, poprzedzający kontekst i sentencję. Rozróżnia zasądzone zadośćuczynienie, jednorazowe odszkodowanie, nierozdzieloną sumę, pominięcie i niepewność. Kwoty żądane, wcześniejsze wypłaty, koszty procesu, odsetki i renta nie są targetami.

Powtórzeń tej samej wartości nie sumujemy. Jawny dowód z sentencji ma pierwszeństwo; konflikt wartości albo wielu beneficjentów wymaga odrzucenia lub przeglądu. Suma jest wyliczana tylko ze zgodnych, ustalonych składników. Osobne żądanie sprawdza całą propozycję wraz z dowodami.

## Ile jest przebiegów

Nie ma stałej liczby wywołań Basala. Dla dokumentu z cechami jest to w przybliżeniu:

`1 + P + S + B + R + M + V`

- `1`: decyzja o instancji;
- `P`: liczba fragmentów akapitów;
- `S`: dodatkowe zdania z fragmentów mieszanych;
- `B`: porcje dopuszczonych faktów do ekstrakcji;
- `R`: kontrole zdań-kandydatów, współdzielone, gdy to samo zdanie pasuje do kilku pól;
- `M`: liczba wystąpień kwot;
- `V`: końcowa weryfikacja propozycji, zwykle 0 albo 1.

To liczba żądań aplikacji, nie liczba wewnętrznych przebiegów sieci. Silnik może wykonywać kilka operacji dla jednego żądania. Wyłączenie cech tekstowych pomija ich klasyfikację i ekstrakcję. Czas zależy również od długości tekstu, urządzenia i modelu.

## Warunek zapisu

Rekord zostaje zapisany dopiero wtedy, gdy całe zasądzenie jest potwierdzone co najmniej na poziomie `amount_minimum_probability`, wszystkie wybrane targety mają ustalone nieujemne wartości i użyte cytaty cech mają kontrolę wycieku zgodną z `maximum_leak_probability`.

Podczas obliczeń katalog `<ID>/pending/` przechowuje źródło, konfigurację i zakończone decyzje modelu. To tymczasowy stan wznowienia, również przy wyłączonym debugowaniu. Po przerwaniu te same żądania korzystają z zapisanych odpowiedzi; tylko przerwane wywołanie może wymagać powtórzenia. Po odrzuceniu albo zapisaniu wyniku katalog roboczy jest usuwany. Odrzucony przypadek nie pozostawia `record.json` ani diagnostyki sprawy. Tymczasowy stan wlicza się do max_gib. Strony dumpu z odrzuconymi wyrokami nie są zachowywane jako surowe pliki. Checkpoint i zbiorcze raporty nadal opisują postęp; ich techniczny rozmiar wlicza się do limitu całego katalogu. Nie da się zarazem zachować pliku w katalogu i nie liczyć jego rzeczywistych bajtów.

Dla zaakceptowanych rekordów `debug: true` zapisuje źródło, offsety, decyzje, konfigurację i wynik pod osobnym ID uruchomienia. `debug: false` pozostawia rekordy i checkpoint. Starsze pliki nie są usuwane automatycznie. Nowa polityka jakości zmienia wersję ekstrakcji i wymaga nowego katalogu dla zbiorów utworzonych wcześniej.

## Przykład poglądowy

To przykład syntetyczny, nie wynik pomiaru jakości modelu.

Wejście: „Powódka złamała nogę wskutek upadku na oblodzonym chodniku. Przez sześć tygodni chodziła o kulach”. W sentencji: „Zasądza 15 000 zł zadośćuczynienia i 2 000 zł odszkodowania”.

Po przejściu kontroli przykładowy rekord zawiera:

```json
{
  "features": {
    "event_circumstances": "Powódka złamała nogę wskutek upadku na oblodzonym chodniku.",
    "harm_type": "Powódka złamała nogę wskutek upadku na oblodzonym chodniku.",
    "harm_severity_duration": "Przez sześć tygodni chodziła o kulach"
  },
  "targets": {"zadoscuczynienie": "15000", "pozostale_odszkodowanie": "2000"},
  "total": "17000",
  "confirmed": true
}
```

Pełny rekord zawiera również ID sprawy, zbioru i uruchomienia, statusy oraz pozostałe wybrane cechy. Predyktory nie zawierają zdań o kwotach. Wysoki wynik kontroli Basala nie zastępuje ręcznie oznaczonej próby ani niezależnej oceny błędów.

## Stratyfikacja i zrównoważenie danych

Obecnie program nie równoważy zbioru. Kolejność ID i limit stron mogą silnie zniekształcać próbę; sama klasa populacji GUGiK nie zapewnia równowagi geograficznej.

Proponowany kolejny etap:

1. Ustalić populację badania i cel: reprezentatywność orzeczeń albo celowo wyrównaną próbę eksperymentalną. Są to różne rozkłady, więc nie należy traktować ich zamiennie.
2. Na zaakceptowanej puli policzyć liczebności i braki według roku, konkretnego sądu/regionu, poziomu, instancji i klasy populacji. Najpierw ocenić szerokie warstwy rok × region; drobne kombinacje szybko stają się puste.
3. Oznaczyć małą próbę rodzaju krzywdy i zdarzenia, ocenić jej ekstrakcję, a dopiero później używać kategorii do doboru danych. Nie tworzyć niewiarygodnych etykiet tylko dla równoważenia.
4. Powiązane postępowania i apelacje grupować przed podziałem. Test najlepiej wydzielić czasowo i pozostawić zgodny z naturalnym rozkładem. Wyrównywanie wykonać wyłącznie na treningu przez losowanie z zapisanym ziarnem lub wagi obserwacji.
5. Dla ciągłego targetu porównywać stałe pasma kwot i rozkład logarytmu kwoty, nie tworzyć klasyfikacji wyroków. Nie usuwać automatycznie rzadkich wysokich zasądzeń. Braki targetu nie są zerem.
6. Raportować MAE/RMSE i pokrycie osobno dla warstw oraz dla naturalnego testu. Równowaga liczebności nie dowodzi braku wycieku ani lepszej regresji.

Nie dodajemy jeszcze arbitralnych kwot ani wag do downloadera. Najpierw potrzebny jest audyt rozkładu i decyzja o docelowej populacji; limity rozmiaru, progi jakości i czasowy zakres pobierania same tego nie rozwiązują.
