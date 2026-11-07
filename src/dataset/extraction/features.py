"""Feature extraction with source, relevance and leakage checks."""

from .model import decisions
from .source import text as api
from .source.passages import extraction_batches, feature_input, source_blocks


def select_facts(diagnostics, threshold, maximum_leak):
    """
    Mark factual passages that meet confidence and leakage constraints.

    args:
        diagnostics: Fragment decisions and their source locations.
        threshold: Minimum decision confidence.
        maximum_leak: Maximum admitted leakage score.

    returns:
        Accepted factual fragments.
    """
    for row in diagnostics:
        answer, section = row["answers"], row["answers"]["sekcja"]
        checks = [
            (
                section["choice"] != "fakty",
                f"not factual (section={section['choice']})",
            ),
            (
                section["confidence"] < threshold,
                f"factual confidence={section['confidence']:.3f}, required >= {threshold}",
            ),
            (
                answer["wyciek"]["noul"] > maximum_leak,
                f"leakage={answer['wyciek']['noul']:.3f}, required <= {maximum_leak}",
            ),
            (
                bool(api.CURRENCY.search(row["text"])),
                "source sentence contains currency",
            ),
            (not any(char.isalpha() for char in row["text"]), "no descriptive text"),
            (
                row["text"].rstrip().endswith(":"),
                "heading instead of factual description",
            ),
        ]
        row["rejection_reasons"] = [reason for failed, reason in checks if failed]
        row["accepted"] = not row["rejection_reasons"]
    return [row for row in diagnostics if row["accepted"]]



def describe_record(answer, mapping, fields, threshold):
    """
    Recover complete source sentences for extracted feature values.

    args:
        answer: Validated extraction answer.
        mapping: Mapping between input sentences and source offsets.
        fields: Configured feature fields.
        threshold: Minimum decision confidence.

    returns:
        Feature values, evidence, statuses and confidence scores.
    """
    values, evidence, statuses, confidence = {}, {}, {}, {}
    for field in fields:
        detail = answer["fields"][field]
        value = answer["data"][field]
        confidence[field] = detail.get("confidence", detail.get("present_probability"))
        if value is None:
            values[field], evidence[field], statuses[field] = (
                "brak informacji",
                [],
                "no description extracted",
            )
            continue
        rows = [
            row
            for row in mapping
            if row["input_start"] < detail["end"] and row["input_end"] > detail["start"]
        ]
        if not rows:
            raise ValueError(f"Cannot recover the source of field: {field}")
        # Keep the complete selected sentence rather than a phrase stripped of its negation.
        rows = rows[:2]
        values[field] = " ".join(row["text"] for row in rows)
        evidence[field] = [
            {key: row[key] for key in ("text", "start", "end")} for row in rows
        ]
        uncertain = confidence[field] is not None and confidence[field] < threshold
        statuses[field] = (
            "low-confidence description retained for review"
            if uncertain
            else "description extracted for review"
        )
    return values, evidence, statuses, confidence



async def extract_features(server, text, config, identifier):
    """
    Extract only from factual findings and review field relevance of final candidates.

    args:
        server: Local inference server.
        text: Full normalized judgment.
        config: Feature schema and admission thresholds.
        identifier: Stable judgment identifier for progress output.

    returns:
        Features, source evidence and model diagnostics with deliberate missing values.
    """
    fields = config["fields"]
    threshold = config["system_one"]["minimum_probability"]
    selected = feature_input(text, config)
    descriptions = {field: "brak informacji" for field in fields}
    evidence = {field: [] for field in fields}
    statuses = {field: "no description extracted" for field in fields}
    confidence = {field: None for field in fields}
    native, reviews = [], {}
    passages = [
        {
            **block,
            "start": block["start"] + selected["start"],
            "end": block["end"] + selected["start"],
        }
        for block in source_blocks(selected["text"], config["chunk_characters"])
        if fields
    ]
    candidates = {field: [] for field in fields}
    schema = {
        "type": "object",
        "properties": {
            field: {
                "type": "string",
                "description": description
                + config["questions"]["extraction"]["field_suffix"],
            }
            for field, description in fields.items()
        },
    }
    for batch, (source, mapping) in enumerate(
        extraction_batches(text, passages, config["chunk_characters"]), 1
    ):
        print(
            f"ID {identifier}: extracting facts, batch {batch} ({len(source)} characters)",
            flush=True,
        )
        answer = (
            await decisions.ask(
                server,
                source,
                {
                    "opis": {
                        "type": "extract",
                        "schema": schema,
                        "instructions": config["questions"]["extraction"][
                            "instructions"
                        ],
                    }
                },
                config,
            )
        )["opis"]
        native.append({"source": source, "mapping": mapping, "answer": answer})
        values, quotes, field_status, probabilities = describe_record(
            answer, mapping, fields, threshold
        )
        for field in fields:
            if quotes[field] and api.CURRENCY.search(values[field]):
                statuses[field] = (
                    "extracted description rejected: source sentence contains currency"
                )
            if quotes[field] and not api.CURRENCY.search(values[field]):
                candidates[field].append(
                    (
                        answer["fields"][field].get("line_probability", 0),
                        values[field],
                        quotes[field],
                        field_status[field],
                        probabilities[field],
                    )
                )
    span_fields = {}
    for field, options in candidates.items():
        for _, value, quotes, _, _ in options:
            span_fields.setdefault((quotes[0]["start"], quotes[-1]["end"]), set()).add(
                field
            )
    for field, options in candidates.items():
        for _, value, quotes, status, probability in sorted(
            options, key=lambda row: row[0], reverse=True
        ):
            start, end = quotes[0]["start"], quotes[-1]["end"]
            if (start, end) not in reviews:
                context = config["context_characters"]
                state = {
                    "oceniany_fragment": value,
                    "poprzedzajacy_kontekst": text[max(0, start - context) : start],
                    "nastepujacy_kontekst": text[end : end + context],
                }
                questions = {
                    "sekcja": {"type": "choice", **config["questions"]["section"]},
                    "wyciek": {"type": "noul", **config["questions"]["leakage"]},
                    **{
                        f"fit_{name}": {
                            "type": "noul",
                            "instructions": "Czy oceniany fragment odpowiada na pytanie: "
                            + fields[name],
                        }
                        for name in sorted(span_fields[start, end])
                    },
                }
                answers = await decisions.ask(server, state, questions, config)
                row = {
                    "text": value,
                    "start": start,
                    "end": end,
                    "answers": answers,
                }
                select_facts(
                    [row],
                    threshold,
                    config["system_one"]["maximum_leak_probability"],
                )
                reviews[start, end] = row
            review = reviews[start, end]
            if (
                review["accepted"]
                and review["answers"][f"fit_{field}"]["noul"] >= threshold
            ):
                descriptions[field], evidence[field] = value, quotes
                statuses[field], confidence[field] = status, probability
                break
            if not statuses[field].startswith("best candidate rejected:"):
                reasons = list(review["rejection_reasons"])
                relevance = review["answers"][f"fit_{field}"]["noul"]
                if relevance < threshold:
                    reasons.append(
                        f"field relevance={relevance:.3f}, required >= {threshold}"
                    )
                statuses[field] = "best candidate rejected: " + "; ".join(reasons)
    for field in fields:
        status = "accepted" if evidence[field] else statuses[field]
        print(f"ID {identifier}: feature {field}: {status}", flush=True)
    return {
        "features": descriptions,
        "evidence": evidence,
        "field_status": statuses,
        "confidence": confidence,
        "native_extraction": native,
        "diagnostics": list(reviews.values()),
        "description_reviews": list(reviews.values()),
        "fields_evaluated": bool(native),
        "feature_input": {
            key: value for key, value in selected.items() if key != "text"
        },
        "input_characters": {
            "full": len(text),
            "classified": sum(len(row["text"]) for row in reviews.values()),
            "features": sum(
                len(value) for field, value in descriptions.items() if evidence[field]
            ),
        },
    }
