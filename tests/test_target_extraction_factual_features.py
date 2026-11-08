"""Target admission and multiple-recipient arithmetic without model downloads."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.dataset.extraction import basal_filtering as basal
from src.dataset.extraction.model import configuration as config_module
from src.dataset.extraction.model import decisions
from src.dataset.extraction.targets import amounts
from src.dataset.extraction.targets import target_extraction as extraction


def test_prior_payment_feature_is_resolved_for_debug_case_without_target(monkeypatch):
    """
    Retain a prior-payment feature even when debug preserves a case without an award.

    args:
        monkeypatch: Scoped target and feature resolvers.

    returns:
        None.
    """

    async def extract_award(*args):
        """
        Supply an excluded earlier payment without any eligible judgment.

        args:
            args: Unused extraction arguments.

        returns:
            Unconfirmed target and embedding diagnostics.
        """
        return {
            "confirmed": False,
            "target_matches": {
                "judgment": None,
                "prior_payment": {"text": "Wypłacono 700 zł."},
            },
        }

    async def prior_payment(server, match, config):
        """
        Resolve the selected feature without modifying the target result.

        args:
            server: Unused inference server.
            match: Earlier-payment source sentence.
            config: Effective pipeline configuration.

        returns:
            Integer feature value.
        """
        assert match["text"] == "Wypłacono 700 zł."
        return 700

    monkeypatch.setattr(amounts, "extract_award", extract_award)
    monkeypatch.setattr(amounts, "extract_prior_payment", prior_payment)
    result = asyncio.run(
        basal.extract(
            None,
            "Wypłacono 700 zł.",
            config_module.load_config(overrides={"debug": True}),
            42,
        )
    )
    assert result["prior_payment"] == 700
    assert result["fields_evaluated"] is False
    assert "prior_payment" not in result["award"]
    assert result["award"]["confirmed"] is False
