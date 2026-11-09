"""Official court-seat locality matching and population classes."""

import re




def locality_class(kind, population):
    """
    Map an official locality type and population to a size category.

    args:
        kind: Official locality type.
        population: Recorded locality population.

    returns:
        Population category or a missing-value marker.
    """
    if kind == "wieś":
        return "village"
    if kind != "miasto" or population is None:
        return "not_available"
    if type(population) not in (int, float) or not 0 < population < float("inf"):
        return "not_available"
    for boundary, label in (
        (20000, "city_under_20000"),
        (100000, "city_20000_to_under_100000"),
        (200000, "city_100000_to_under_200000"),
        (500000, "city_200000_to_under_500000"),
    ):
        if population < boundary:
            return label
    return "city_at_least_500000"


def match_locality(court_name, rows, morphology):
    """
    Match an inflected court-location name without guessing ambiguous results.

    args:
        court_name: Source court name containing its location.
        rows: Candidate source records.
        morphology: Polish inflection analyzer.

    returns:
        Matched locality or None, and a diagnostic reason.
    """
    location = re.search(r"\b(?:w|we)\s+([^\n]+)$", court_name)
    if not location:
        return None, "court name does not contain a seat"
    name = location[1].strip()
    exact = [row for row in rows if row["NAZWA"].casefold() == name.casefold()]
    if len(exact) == 1:
        return exact[0], "exact name"
    if len(exact) > 1:
        return None, "multiple localities with the same name"
    words = name.split()
    lemmas = []
    for word in words:
        forms = {word.casefold()}
        for _, _, interpretation in morphology.analyse(word):
            forms.add(interpretation[1].split(":")[0].casefold())
        lemmas.append(forms)
    matches = [
        row
        for row in rows
        if len(row["NAZWA"].split()) == len(words)
        and all(
            word.casefold() in forms
            for word, forms in zip(row["NAZWA"].split(), lemmas)
        )
    ]
    return (
        (matches[0], "unique inflection match using Morfeusz/SIMC")
        if len(matches) == 1
        else (None, f"candidate count: {len(matches)}")
    )
