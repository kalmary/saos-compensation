"""Validated Basal requests, persistent decision cache and call tracing."""

import hashlib
import json
import math
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from ...storage.files import write_json

ROOT = Path(__file__).resolve().parents[4]
REQUEST_COUNTS = {"cached": 0, "computed": 0}
MODEL_CALLS = ContextVar("model_calls", default=None)
DECISION_CACHE = ContextVar("decision_cache", default=None)


def check_decision(response, questions, state=None):
    # Reject truncated inputs, incomplete answers and invalid probability distributions.
    """
    Validate model answers, probability distributions and extraction offsets.

    args:
        response: Raw model response.
        questions: Model question definitions.
        state: Input state supplied to the model.

    returns:
        Validated answer mapping.
    """
    if response.get("truncated"):
        raise ValueError("The inference engine truncated its input")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Incomplete answer set")
    for key, question in questions.items():
        answer = answers[key]
        if question["type"] == "extract":
            expected = set(question["schema"]["properties"])
            if (
                not isinstance(state, str)
                or set(answer.get("data", {})) != expected
                or set(answer.get("fields", {})) != expected
            ):
                raise ValueError("Incomplete extraction record")
            for field, value in answer["data"].items():
                detail = answer["fields"][field]
                probability = detail.get("confidence")
                if probability is not None and (
                    type(probability) not in (int, float)
                    or not math.isfinite(probability)
                    or not 0 <= probability <= 1
                ):
                    raise ValueError(f"Invalid field confidence: {field}")
                if value is None:
                    continue
                start, end = detail.get("start"), detail.get("end")
                if (
                    not isinstance(value, str)
                    or not value.strip()
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(state)
                    or state[start:end] != value
                ):
                    raise ValueError(f"Field {field} does not match its source span")
            continue
        probabilities = answer.get("probabilities", {})
        if not probabilities or any(
            type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
            for p in probabilities.values()
        ):
            raise ValueError("Invalid probabilities")
        if abs(sum(probabilities.values()) - 1) > 0.01:
            raise ValueError("Probabilities do not sum to one")
        expected = (
            set(question["criteria"])
            if question["type"] == "choice"
            else {"true", "false"}
        )
        if set(probabilities) != expected:
            raise ValueError("Answer options do not match the question")
        if question["type"] == "choice" and answer.get("choice") not in expected:
            raise ValueError("Selected option is outside the answer space")
        selected = answer["choice"] if question["type"] == "choice" else "true"
        probability = (
            answer.get("confidence")
            if question["type"] == "choice"
            else answer.get("noul")
        )
        if (
            type(probability) not in (int, float)
            or not math.isfinite(probability)
            or abs(probability - probabilities[selected]) > 0.01
        ):
            raise ValueError("Answer score does not match its probability distribution")
    return answers



async def ask(server, state, questions, config):
    """
    Request a validated model decision and optionally reuse its diagnostic cache.

    args:
        server: Local inference server.
        state: Input state supplied to the model.
        questions: Model question definitions.
        config: Effective pipeline or acquisition settings.

    returns:
        Validated model answers.
    """
    request = {"state": state, "lang": config["language"], "questions": questions}
    # Selection thresholds do not change model answers, so they do not invalidate the cache.
    identity = {
        "version": config["version"],
        "engine_version": "2.4.1",
        "model": {
            key: value
            for key, value in config["system_one"].items()
            if key
            not in {
                "minimum_probability",
                "maximum_leak_probability",
                "amount_minimum_probability",
            }
        },
    }
    key = hashlib.sha256(
        json.dumps([identity, request], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    path = ROOT / config["api"]["download_path"] / "debug" / "decisions" / f"{key}.json"
    progress = DECISION_CACHE.get()
    if progress is not None:
        path = progress / f"{key}.json"
    cache_enabled = config["api"]["debug"] or progress is not None
    if cache_enabled:
        path.parent.mkdir(parents=True, exist_ok=True)
    if cache_enabled and path.exists():
        REQUEST_COUNTS["cached"] += 1
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["request"] != request:
            raise ValueError("Saved model decision does not match its request")
        response = saved["response"]
    else:
        REQUEST_COUNTS["computed"] += 1
        response = await server.serve(request)
        check_decision(response, questions, state)
        if cache_enabled:
            write_json(
                path,
                {
                    "request": request,
                    "response": response,
                    "model": config["system_one"],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                },
            )
    if MODEL_CALLS.get() is not None:
        MODEL_CALLS.get().append({"request": request, "response": response})
    return check_decision(response, questions, state)
