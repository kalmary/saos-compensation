"""Acquisition configuration and inclusive date bounds."""

from calendar import monthrange
from datetime import date
from pathlib import Path


def search_settings(config, today=None):
    """
    Validate acquisition settings and derive inclusive judgment-date bounds.

    args:
        config: Effective pipeline or acquisition settings.
        today: Date used to anchor the acquisition range.

    returns:
        Start date, end date and effective query parameters.
    """
    if not isinstance(config, dict) or set(config) != {
        "max_age_years",
        "max_pages",
        "max_gib",
        "debug",
        "download_path",
        "timeout",
    }:
        raise ValueError(
            "api.yaml requires max_age_years, max_pages, max_gib, debug, download_path and timeout"
        )
    if type(config["max_gib"]) not in (int, float) or not 0 <= config[
        "max_gib"
    ] < float("inf"):
        raise ValueError("max_gib must be a nonnegative finite number")
    if type(config["timeout"]) not in (int, float) or not 0 < config["timeout"] < float(
        "inf"
    ):
        raise ValueError("timeout must be a positive finite number")
    if type(config["debug"]) is not bool:
        raise ValueError("debug must be true or false")
    years, limit = config["max_age_years"], config["max_pages"]
    if type(years) is not int or years <= 0 or type(limit) is not int or limit < 0:
        raise ValueError("max_age_years must be positive and max_pages nonnegative")
    path = config["download_path"]
    if (
        not isinstance(path, str)
        or not path.strip()
        or Path(path).is_absolute()
        or ".." in Path(path).parts
    ):
        raise ValueError("download_path must be relative to the project root")
    end = today or date.today()
    if years >= end.year:
        raise ValueError("max_age_years exceeds the supported date range")
    start = end.replace(
        year=end.year - years,
        day=min(end.day, monthrange(end.year - years, end.month)[1]),
    )
    params = {
        "pageSize": 100,
        "pageNumber": 0,
        "judgmentStartDate": start.isoformat(),
        "judgmentEndDate": end.isoformat(),
    }
    return start, end, params
