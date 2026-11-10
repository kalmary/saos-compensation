"""Denial rejection before amount parsing, predictors and Basal requests."""

import asyncio

import pytest

from dataset.extraction import basal_filtering
from dataset.extraction.model import configuration, decisions
from dataset.extraction.quality import target_rejection_reason
from dataset.extraction.targets import amounts, target_extraction


@pytest.mark.parametrize("debug", [False, True])
def test_denied_case_skips_amount_and_feature_extraction(monkeypatch, debug):
    """Return a rejected result without parsing money or asking for predictors.

    args:
        monkeypatch: Scoped matcher and downstream extraction replacements.
        debug: Whether model diagnostics are enabled.

    returns:
        None.
    """
    text = "Oddala powództwo w całości."
    denial = {"text": text, "start": 0, "end": len(text), "similarity": 0.9}
    config = configuration.load_config(overrides={"debug": debug})
    monkeypatch.setattr(
        target_extraction, "match_targets",
        lambda *args: {"judgment-denial": denial, "judgment": None},
    )
    for module, name in (
        (target_extraction, "amount_in_pln"),
        (amounts, "extract_prior_payment"),
        (decisions, "ask"),
        (basal_filtering, "extract_features"),
    ):
        monkeypatch.setattr(module, name, lambda *args: pytest.fail("Unexpected extraction after denial"))
    calls = [] if debug else None
    token = decisions.MODEL_CALLS.set(calls)
    try:
        result = asyncio.run(basal_filtering.extract(None, text, config, 42))
    finally:
        decisions.MODEL_CALLS.reset(token)
    assert result["fields_evaluated"] is False
    assert result["award"]["denied"] is True
    assert result["award"]["confirmed"] is False
    assert result["award"]["suma"] == "brak informacji"
    assert target_rejection_reason(result["award"], config) == "judgment denied"
    assert "features" not in result and "prior_payment" not in result
    if debug:
        assert calls[0]["matches"]["judgment-denial"] == denial


def test_denial_evidence_must_match_original_source(monkeypatch):
    """Reject denial diagnostics with incorrect source offsets.

    args:
        monkeypatch: Scoped match replacement.

    returns:
        None.
    """
    monkeypatch.setattr(
        target_extraction, "match_targets",
        lambda *args: {
            "judgment-denial": {"text": "Oddala powództwo.", "start": 0, "end": 17},
            "judgment": None,
        },
    )
    with pytest.raises(ValueError, match="denial sentence does not match its source"):
        asyncio.run(target_extraction.extract_targets(None, "Inny tekst.", configuration.load_config()))
