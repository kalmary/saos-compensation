import asyncio
import json
import shutil

import pytest

from dataset.extraction.model import configuration as config_module
from dataset.extraction.model import decisions, prompts

FIELD_NAMES = {
    "zdarzenie_i_okolicznosci": "event_circumstances",
    "rodzaj_doznanej_krzywdy": "harm_type",
    "przebieg_nasilenie_i_czas_trwania": "harm_severity_duration",
    "konsekwencje_dla_zycia": "life_impact",
    "indywidualne_okolicznosci_poszkodowanego": "claimant_circumstances",
    "podjete_dzialania_i_potrzebna_pomoc": "treatment_assistance",
}


def translate_fields(value):
    """
    Adapt historical fixture schema keys without modifying original artifacts.

    args:
        value: Value to serialize or adapt.

    returns:
        Value with historical schema keys adapted.
    """
    if isinstance(value, dict):
        return {
            FIELD_NAMES.get(key, key): translate_fields(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [translate_fields(item) for item in value]
    return value


def test_basal_validates_historical_requests_and_responses(
    monkeypatch,
    tmp_path,
    model_calls_path,
):
    """
    Validate historical model responses independently of the current extraction order.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        model_calls_path: Historical model request/response fixture.

    returns:
        None.
    """
    config = config_module.load_config()
    monkeypatch.setattr(decisions, "ROOT", tmp_path)
    calls = translate_fields(json.loads(model_calls_path.read_text()))
    responses = {
        json.dumps(call["request"], sort_keys=True, ensure_ascii=False): call[
            "response"
        ]
        for call in calls
    }
    seen = []

    class Server:
        async def serve(self, request):
            """
            Provide the serve behavior used by this operation.

            args:
                request: Input for this operation.

            returns:
                Result of this operation.
            """
            key = json.dumps(request, sort_keys=True, ensure_ascii=False)
            assert key in responses, "Extraction changed a saved model request"
            seen.append(key)
            return responses[key]

    for call in calls:
        request = call["request"]
        asyncio.run(
            decisions.ask(Server(), request["state"], request["questions"], config)
        )
    assert len(seen) == len(calls)


def test_config_requires_text_fields_with_metadata_selection(project_root, tmp_path):
    """
    Require damage questions alongside the metadata selection.

    args:
        project_root: Repository configuration and fixture root.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    import yaml

    shutil.copytree(project_root / "configs/dataset/acquisition", tmp_path, dirs_exist_ok=True)
    config = config_module.read_config(tmp_path / "features.yaml")
    config["metadata_fields"] = list(prompts.METADATA_FIELDS)
    config["text_fields"] = {}
    (tmp_path / "features.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="must be nonempty"):
        config_module.load_config(tmp_path)
    config["text_fields"]["damage"] = "Jakiej szkody doznał poszkodowany?"
    (tmp_path / "features.yaml").write_text(yaml.safe_dump(config))
    loaded = config_module.load_config(tmp_path)
    assert loaded["metadata_fields"] == list(prompts.METADATA_FIELDS)
    assert loaded["fields"] == config["text_fields"]


@pytest.mark.parametrize("debug", [False, True])
def test_model_request_cache_is_debug_only(monkeypatch, tmp_path, debug):
    """
    Verify model request cache is debug only.

    args:
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.
        debug: Whether to save diagnostic artifacts.

    returns:
        None.
    """
    config = config_module.load_config(overrides={"debug": debug})
    monkeypatch.setattr(decisions, "ROOT", tmp_path)
    calls = []

    class Server:
        async def serve(self, request):
            """
            Provide the serve behavior used by this operation.

            args:
                request: Input for this operation.

            returns:
                Result of this operation.
            """
            calls.append(request)
            return {
                "answers": {
                    "q": {"noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}
                }
            }

    questions = {"q": {"type": "noul", "instructions": "Czy to fakt?"}}

    async def ask():
        """
        Request a validated model decision and optionally reuse its diagnostic cache.

        args:
            None

        returns:
            Validated model answers.
        """
        await decisions.ask(Server(), "Opis.", questions, config)
        await decisions.ask(Server(), "Opis.", questions, config)

    asyncio.run(ask())
    assert len(calls) == (1 if debug else 2)
    assert bool(list(tmp_path.rglob("*.json"))) == debug


def test_interrupted_extraction_reuses_completed_decisions_without_debug(
    monkeypatch,
    tmp_path,
):
    """
    Reuse operational decisions after cancellation with diagnostics disabled.

    args:
        monkeypatch: Scoped dependencies.
        tmp_path: Isolated working-state directory.

    returns:
        None.
    """
    config = config_module.load_config(overrides={"debug": False})
    questions = {"q": {"type": "noul", "instructions": "Czy to fakt?"}}
    response = {
        "answers": {"q": {"noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}}
    }
    calls = []

    class Server:
        async def serve(self, request):
            """
            Interrupt the second state once.

            args:
                request: Model decision input.

            returns:
                Deterministic model response.
            """
            calls.append(request["state"])
            if request["state"] == "second" and calls.count("second") == 1:
                raise asyncio.CancelledError()
            return response

    async def execute():
        """
        Re-enter the same working cache after cancellation.

        args:
            None

        returns:
            None.
        """
        token = decisions.DECISION_CACHE.set(tmp_path / "pending/decisions")
        try:
            await decisions.ask(Server(), "first", questions, config)
            with pytest.raises(asyncio.CancelledError):
                await decisions.ask(Server(), "second", questions, config)
            await decisions.ask(Server(), "first", questions, config)
            await decisions.ask(Server(), "second", questions, config)
        finally:
            decisions.DECISION_CACHE.reset(token)

    asyncio.run(execute())
    assert calls == ["first", "second", "second"]
    assert len(list((tmp_path / "pending/decisions").glob("*.json"))) == 2
