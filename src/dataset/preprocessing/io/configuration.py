"""Validate the ordered allowlist of original model input fields."""

from pathlib import Path

import yaml

from ..transforms.rules import FIELD_RULES, TEXT_FIELDS


def validate_config(settings):
    """Validate feature selection without accepting preprocessing overrides.

    args:
        settings: Mapping containing only feature_fields, an ordered name list.

    returns:
        Independent validated feature selection.
    """
    if not isinstance(settings, dict) or set(settings) != {"feature_fields"}:
        raise ValueError("Model input configuration accepts only feature_fields")
    fields = settings["feature_fields"]
    if not isinstance(fields, list) or not fields or any(not isinstance(field, str) for field in fields):
        raise ValueError("feature_fields must be a nonempty list of names")
    if len(set(fields)) != len(fields):
        raise ValueError("feature_fields cannot contain duplicates")
    for field in fields:
        if field not in FIELD_RULES and field not in TEXT_FIELDS:
            raise ValueError(f"Unsupported predictor: {field}")
    return {"feature_fields": list(fields)}


def validate_scaling(settings):
    """Check internal numeric transformation choices.

    args:
        settings: Numeric or target policy mapping supplied by implementation code.

    returns:
        None.
    """
    if settings["transform"] not in ("identity", "log1p"):
        raise ValueError("transform must be identity or log1p")
    if settings["scaling"] not in ("none", "standard", "robust"):
        raise ValueError("scaling must be none, standard or robust")


def load_config(path):
    """Read the ordered model feature allowlist from YAML.

    args:
        path: Model input YAML path.

    returns:
        Validated feature selection without transformation settings.
    """
    return validate_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
