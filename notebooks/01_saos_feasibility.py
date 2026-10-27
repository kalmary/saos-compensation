"""Check SAOS data availability and write a feasibility report to tmp/."""

import hashlib
import json
import re
import time
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tmp" / "saos_feasibility"
BASE_URL = "https://www.saos.org.pl/api"


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
        self.parts.append(data)


def extract_text(html):
    parser = JudgmentText()
    parser.feed(html)
    lines = [re.sub(r"\s+", " ", line).strip() for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def validate_judgment(record, expected_id):
    errors = []
    if type(record.get("id")) is not int or record["id"] != expected_id:
        errors.append("invalid or mismatched ID")
    try:
        date.fromisoformat(record.get("judgmentDate", ""))
    except (TypeError, ValueError):
        errors.append("invalid judgment date")
    if not isinstance(record.get("textContent"), str) or not extract_text(record["textContent"]):
        errors.append("missing judgment text")
    court = (record.get("division") or {}).get("court") or {}
    if record.get("courtType") != "COMMON" or not court.get("id") or not court.get("name"):
        errors.append("missing common-court metadata")
    cases = record.get("courtCases")
    if not isinstance(cases, list) or not cases or any(not case.get("caseNumber") for case in cases):
        errors.append("missing case number")
    return errors


def build_report(results, strata, records, errors):
    texts = {identifier: extract_text(record["textContent"]) for identifier, record in records.items()}
    timestamps = [json.loads(path.read_text())["acquired_at"] for path in OUTPUT.glob("*.meta.json")]
    lines = [
        "# SAOS data feasibility",
        "",
        "**Overall: MODERATE / conditional.** Data access works and some judgments contain explicit awards and useful case facts. A reliable regression dataset still requires a narrower claim definition, reviewed labels, and leakage-safe features.",
        "",
        f"Snapshot acquisition: {min(timestamps)} to {max(timestamps)}.",
        "",
        "## 1. Download and search",
        "",
        "All queries restrict courtType=COMMON and judgmentTypes=SENTENCE. Counts overlap and are search matches, not counts of usable training examples.",
        "",
        "| Query | Matches |",
        "|---|---:|",
    ]
    for name, result in results.items():
        lines.append(f"| {name} | {result['info']['totalResults']:,} |")
    lines.extend([
        "",
        "Search -> page through IDs -> fetch /api/judgments/{id} for full text. Search textContent contains excerpts, not full judgments. Raw JSON and request URLs, timestamps and checksums are retained in tmp/saos_feasibility/.",
        "",
        "The keyword and legalBase filters can narrow acquisition, but missing metadata can exclude relevant cases. Compare them against text-only retrieval before choosing the final query. Do not mix odszkodowanie and zadośćuczynienie as one unspecified label.",
        "",
        "The live queryTemplate distinguishes courtType=COMMON from ccCourtType=DISTRICT/REGIONAL/APPEAL; the published search documentation conflates these fields.",
        "",
        "Pagination check: pages 0 and 1 have distinct IDs, and the server next link advances to page 1.",
        "",
        "## 2. Sample and validation",
        "",
        "Purposive sample: four oldest and four newest results for each of DISTRICT and REGIONAL, plus two broad-query results. Date ties and court clustering make this unsuitable for estimating corpus-wide quality.",
        "",
        "| Stratum | IDs |",
        "|---|---|",
    ])
    for name, identifiers in strata.items():
        lines.append(f"| {name} | {', '.join(map(str, identifiers))} |")
    lines.extend([
        "",
        f"Downloaded {len(records) + len(errors)} judgments; {len(records)} passed basic ID, date-format, court, case-number and nonempty-text checks; {len(errors)} failed. Passing these checks does not establish semantic consistency or suitability for modeling.",
        "",
        "| Field | Nonempty in validated sample |",
        "|---|---:|",
    ])
    for field in ("textContent", "decision", "summary", "legalBases", "keywords", "referencedCourtCases"):
        count = sum(bool(record.get(field)) for record in records.values())
        lines.append(f"| {field} | {count}/{len(records)} |")
    if errors:
        lines.extend(["", "Validation failures:", "", json.dumps(errors, ensure_ascii=False, indent=2)])
    lines.extend([
        "",
        f"Plain-text length: {min(map(len, texts.values())):,}–{max(map(len, texts.values())):,} characters. Long documents need section selection or chunking; short ones may lack facts entirely.",
        "",
        "## 3. Reviewed examples and extraction risks",
        "",
        "These are evidence checks for feasibility, not a validated label dataset. IDs link to the source judgments.",
        "",
        "| Judgment | Observation | Consequence |",
        "|---|---|---|",
    ])
    examples = [
        (4, "80.000 zł", "80,000 PLN appears as a request; the appeal is dismissed.", "Do not treat a requested sum or rejected appeal as an awarded amount."),
        (6, "1230 zł", "The operative amount is payment for legal aid.", "An amount in the operative section is not necessarily compensation."),
        (143825, "15.000,00 zł", "15,000 PLN is awarded to one claimant for bereavement; prior payment is discussed separately.", "Awarded principal is recoverable; distinguish it from total compensation including previous payments."),
        (144313, "2000zł", "2,000 PLN is awarded to the identified beneficiary; costs and interest are separate.", "A claim-specific target is recoverable, subject to review."),
        (554194, "17 176,00 zł", "17,176 PLN combines 15,000 PLN compensation and 2,176 PLN treatment costs.", "Separate claim components using the reasoning; the operative total alone is wrong for a compensation-only target."),
        (144381, "135,00 złotych", "The awarded sum concerns property loss; the moral-loss claim is rejected.", "A compensation search term does not establish an eligible compensation award."),
        (144754, "Powództwo oddala", "Only 568 characters of operative text are available in this snapshot.", "A valid API record can still be unusable for feature extraction."),
        (126364, "ochronę konkurencji", "The case concerns competition protection.", "Text-only search includes unrelated subject matter."),
    ]
    for identifier, evidence, observation, consequence in examples:
        if identifier in texts:
            if evidence not in texts[identifier]:
                raise ValueError(f"Reviewed evidence is missing from judgment {identifier}")
            lines.append(f"| [{identifier}](https://www.saos.org.pl/judgments/{identifier}) | {observation} | {consequence} |")
    lines.extend([
        "",
        "Observed date inconsistencies also need review before temporal splits:",
        "",
    ])
    for identifier, evidence in ((553925, "09 września 2026"), (553920, "20 marca 2026"), (554072, "24 kwietnia 2026")):
        if identifier in texts:
            if evidence not in texts[identifier]:
                raise ValueError(f"Date evidence changed in judgment {identifier}")
            lines.append(f"- ID {identifier}: API judgmentDate={records[identifier]['judgmentDate']}; document heading says {evidence}.")
    lines.extend([
        "- ID 406488: metadata and heading say 2010, but the case number ends /19 and the hearing date is in 2020. Do not silently correct the date.",
        "",
        "## 4. Necessary data and proposed extraction",
        "",
        "| Information | Source and approach | Feasibility |",
        "|---|---|---|",
        "| Awarded principal in PLN | Separate label process using operative section and claim-specific reasoning; identify claimant, claim and procedural stage | Moderate; demonstrated examples, no extraction accuracy measured |",
        "| Court, level, judgment date | Structured metadata; verify conflicts against text | High availability; semantic checks required |",
        "| Region | Map court location to a documented region | Feasible for court region; not equivalent to claimant/accident region |",
        "| Cause, harm, injury, lasting effects | Extract explicit facts with source evidence, preserving nulls | Present in some cases; completeness and accuracy unmeasured |",
        "| Claimant age and characteristics | Match facts to the correct claimant; retain time reference | Uneven availability and anonymization |",
        "| Related proceedings | Source ID, court and case number, appeal text, verified relationships | Partial; citation lists also contain unrelated precedents |",
        "",
        "Example of available facts: ID 143825 explicitly gives claimant age 42 at his mother's death and describes the fatal traffic accident and psychological effects. This is bereavement, not the claimant's bodily injury; claim categories need separate treatment. Both age and event timing must retain their meaning.",
        "",
        "Start with one claim family and one claimant per example. Define whether the target is the additional principal awarded by this judgment or total compensation across prior payments. Exclude inseparable mixed awards, ambiguous allocations, foreign currencies and unclear appeal adjustments. Define rejected-claim handling explicitly; never infer zero from a missing sum.",
        "",
        "## 5. Leakage prevention",
        "",
        "Remove the operative section, requests, previous payments, calculations and amount-selection reasoning from feature inputs. Removing currency strings alone is insufficient. Extract only allowlisted facts from reviewed factual passages, with evidence and nulls. A party's allegation, quoted law or precedent is not automatically an established case fact.",
        "",
        "Define the prediction time: facts established later by an expert may be unavailable when a real prediction would be requested. Group verified related proceedings before splitting, then fit learned encoders on training data only. Feature output and source evidence still require a leakage audit before embeddings or regression.",
        "",
        "## 6. Polish model for standardization",
        "",
        "One fixed open-weight Polish instruction model is a plausible candidate for mapping facts to controlled fields. Bielik-11B-v3.0-Instruct and the smaller Bielik-1.5B-v3.0-Instruct have Polish support and downloadable weights subject to their release access conditions. The 11B model card lists Apache 2.0 and gated access; an anonymous request for the small model's README also returned an authentication requirement.",
        "",
        "Keep checkpoint, prompt, quantization and decoding settings fixed. Request JSON with value and exact source evidence for cause_of_harm, injury_description, severity, lasting_effects and claimant_age. Use null for unsupported facts. Numeric/categorical encoders and text embeddings remain separate later steps.",
        "",
        "Next experiment: label a small sample of factual passages, compare the model's per-field accuracy/F1, null handling, unsupported statements, leakage and latency against that reference. Polish support and valid JSON do not prove extraction quality. No weights were downloaded and no inference was run here.",
        "",
        "## 7. Decision",
        "",
        "- Data acquisition: HIGH feasibility; live scoped queries, pagination and full-text retrieval work.",
        "- Building a trustworthy learning dataset: MODERATE, conditional on claim selection, reviewed labels, semantic validation and leakage controls.",
        "- Automated feature standardization: PLAUSIBLE, NOT VERIFIED; evaluate a fixed Polish model on labeled evidence.",
        "- Predictive model performance: UNKNOWN; availability does not establish predictive value or generalization.",
        "",
        "Proceed with a narrowly defined compensation cohort and a larger reviewed sample across courts and years. Do not interpret the search counts as a usable-dataset estimate; this sample is too small and deliberately selected for that calculation. No database, vectorization or training was performed.",
        "",
        "## Sources",
        "",
        "- https://github.com/kalmary/saos-compensation/issues/1",
        "- https://www.saos.org.pl/help/index.php/dokumentacja-api",
        "- https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przeszukiwania-danych",
        "- https://www.saos.org.pl/help/index.php/dokumentacja-api/api-przegladania-danych",
        "- https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct",
        "- https://huggingface.co/speakleash/Bielik-1.5B-v3.0-Instruct",
    ])
    return "\n".join(lines) + "\n"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = {
        "all": "zadośćuczynienie",
        "courtType": "COMMON",
        "judgmentTypes": "SENTENCE",
        "pageSize": 10,
        "pageNumber": 0,
        "sortingField": "DATABASE_ID",
        "sortingDirection": "ASC",
    }
    queries = {
        "odszkodowanie, text": {**base, "all": "odszkodowanie"},
        "zadośćuczynienie, text": base,
        "zadośćuczynienie, text + keyword": {**base, "keywords": "zadośćuczynienie"},
        "zadośćuczynienie, text + legalBase=445": {**base, "legalBase": "445"},
    }
    results = {}
    print("Checking scoped SAOS queries...", flush=True)
    for name, params in queries.items():
        results[name] = search_judgments(params)
        print(f"  {name}: {results[name]['info']['totalResults']:,} matches", flush=True)

    first_page = results["zadośćuczynienie, text"]
    next_links = [link["href"] for link in first_page["links"] if link["rel"] == "next"]
    if len(next_links) != 1:
        raise ValueError("Expected exactly one next-page link")
    second_page = fetch_json(next_links[0])
    if second_page["queryTemplate"]["pageNumber"]["value"] != 1:
        raise ValueError("Next link did not advance to page 1")
    first_ids = {item["id"] for item in first_page["items"]}
    second_ids = {item["id"] for item in second_page["items"]}
    if first_ids & second_ids or len(second_ids) != len(second_page["items"]):
        raise ValueError("Pagination returned duplicate IDs")

    sample = {item["id"]: item for item in first_page["items"][:2]}
    strata = {}
    for level in ("DISTRICT", "REGIONAL"):
        for direction in ("ASC", "DESC"):
            params = {**base, "ccCourtType": level, "sortingField": "JUDGMENT_DATE", "sortingDirection": direction}
            result = search_judgments(params)
            selected = result["items"][:4]
            strata[f"{level.lower()}_{direction.lower()}"] = [item["id"] for item in selected]
            sample.update({item["id"]: item for item in selected})

    records, errors = {}, {}
    for identifier, item in sample.items():
        payload = fetch_json(item["href"])
        if not isinstance(payload.get("data"), dict):
            raise ValueError(f"Missing judgment object for {identifier}")
        problems = validate_judgment(payload["data"], identifier)
        if problems:
            errors[identifier] = problems
        else:
            records[identifier] = payload["data"]
        print(f"  Judgment {identifier}: {'INVALID: ' + ', '.join(problems) if problems else 'basic validation passed'}", flush=True)
    if not records:
        raise ValueError("No valid full judgments available")

    report = build_report(results, strata, records, errors)
    report_path = OUTPUT / "report.md"
    report_path.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    assert extract_text("<p>A <span>B</span>.</p><p>Żółć&nbsp;123</p>") == "A B.\nŻółć 123"
    assert validate_judgment({}, 1)
    main()
