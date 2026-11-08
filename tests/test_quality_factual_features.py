"""Dataset admission boundaries without model inference."""

import asyncio

import pytest

from src.dataset.extraction import basal_filtering as basal
from src.dataset.extraction.model import configuration as config_module
from src.dataset.extraction.model import decisions
from src.dataset.extraction.quality import rejection_reason, target_rejection_reason
from src.dataset.extraction.targets import amounts


@pytest.mark.parametrize(
    ("confirmed", "debug", "probability", "continues"),
    [
        (False, False, 0.9, False),
        (True, False, 0.9, True),
        (False, False, 0.4, False),
        (False, True, 0.4, True),
        (False, True, 0.9, True),
    ],
)
def test_target_is_checked_before_any_feature_inference(
    monkeypatch, confirmed, debug, probability, continues
):
    """
    Stop rejected targets early and run accepted targets before instance inference.

    args:
        monkeypatch: Scoped function replacements.
        confirmed: Whether the extracted target meets admission rules.
        debug: Whether low-confidence proposals may be retained.
        probability: Award verification probability.
        continues: Whether feature inference should run.

    returns:
        None.
    """
    events = []
    award = amounts.finish_award(
        {"suma": "100"},
        confirmed,
        {"noul": probability},
    )

    async def extract_award(server, text, diagnostics, config):
        """
        Supply a target without invoking the model.

        args:
            server: Unused inference server.
            text: Source judgment.
            diagnostics: Unused fragment diagnostics.
            config: Effective configuration.

        returns:
            Controlled award decision.
        """
        events.append("target")
        return award

    async def ask(server, state, questions, config):
        """
        Check that feature inference follows an accepted target.

        args:
            server: Unused inference server.
            state: Model input.
            questions: Requested model decisions.
            config: Effective configuration.

        returns:
            Instance decision.
        """
        assert continues
        assert events == ["target"]
        assert set(questions) == {"instancja"}
        events.append("instance")
        return {"instancja": {"choice": "nieustalona", "confidence": 0.9}}

    monkeypatch.setattr(amounts, "extract_award", extract_award)
    monkeypatch.setattr(decisions, "ask", ask)
    config = config_module.load_config()
    config["fields"] = {}
    config["api"]["debug"] = debug
    result = asyncio.run(basal.extract(None, "Source judgment.", config, 42))
    assert result["award"] == award
    assert events == (["target", "instance"] if continues else ["target"])
    if not continues:
        assert result == {"id": 42, "award": award}
