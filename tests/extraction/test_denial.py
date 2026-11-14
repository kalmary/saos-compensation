"""Zero-valued denial targets with ordinary predictor extraction."""

import asyncio

import pytest

from dataset.extraction import basal_filtering
from dataset.extraction.model import configuration, decisions
from dataset.extraction.quality import target_rejection_reason
from dataset.extraction.targets import target_extraction


@pytest.mark.parametrize("debug", [False, True])
def test_denied_case_extracts_features_with_zero_target(monkeypatch, debug):
    """Continue extracting predictors without parsing a monetary award.

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
    monkeypatch.setattr(
        target_extraction, "amount_in_pln",
        lambda *args: pytest.fail("Denial must not parse an award amount"),
    )
    async def ask(*args):
        """Supply an instance decision.

        args:
            args: Inference request arguments.

        returns:
            Controlled instance response.
        """
        return {"instancja": {"choice": "pierwsza", "confidence": 0.9}}

    async def features(*args):
        """Supply factual damage evidence.

        args:
            args: Feature extraction arguments.

        returns:
            Extracted damage and its source evidence.
        """
        return {"features": {"damage": "Opis szkody."}, "fields_evaluated": True}

    monkeypatch.setattr(decisions, "ask", ask)
    monkeypatch.setattr(basal_filtering, "extract_features", features)
    config["metadata_fields"] = []
    calls = [] if debug else None
    token = decisions.MODEL_CALLS.set(calls)
    try:
        result = asyncio.run(basal_filtering.extract(None, text, config, 42))
    finally:
        decisions.MODEL_CALLS.reset(token)
    assert result["fields_evaluated"] is True
    assert result["award"]["denied"] is True
    assert result["award"]["confirmed"] is True
    assert result["award"]["suma"] == "0"
    assert target_rejection_reason(result["award"], config) is None
    assert result["features"]["damage"] == "Opis szkody."
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


@pytest.mark.parametrize("similarity", [0.7, 0.7001])
@pytest.mark.parametrize("total", ["0", "1", "NaN", "brak informacji"])
def test_denial_target_requires_strict_match_and_zero(similarity, total):
    """Validate denial targets without accepting a monetary cost as zero evidence.

    args:
        similarity: Denial similarity at or above the configured threshold.
        total: Proposed denial target value.

    returns:
        None.
    """
    config = configuration.load_config()
    config["embedding"]["minimum_cosine_similarity"] = 0.7
    award = {
        "denied": True, "confirmed": True, "suma": total,
        "target_matches": {"judgment-denial": {"similarity": similarity}},
    }
    assert (target_rejection_reason(award, config) is None) is (
        similarity > 0.7 and total == "0"
    )
