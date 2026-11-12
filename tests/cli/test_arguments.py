from datetime import date

import pytest

from dataset.acquisition import pipeline
from dataset.acquisition.settings import search_settings
from dataset.cli import main as cli
from dataset.clients import api
from dataset.extraction.model import configuration as config_module

CONFIG = {
    "max_age_years": 10,
    "max_pages": 0,
    "max_gib": 0,
    "debug": False,
    "download_path": "tmp/",
    "timeout": 30,
}


def test_config_and_path_are_independent_of_working_directory(
    project_root,
    monkeypatch,
    tmp_path,
):
    """
    Verify config and path are independent of working directory.

    args:
        project_root: Repository configuration and fixture root.
        monkeypatch: Scoped replacements for the test.
        tmp_path: Isolated temporary directory.

    returns:
        None.
    """
    expected = config_module.load_config()
    monkeypatch.chdir(tmp_path)
    config = config_module.load_config()
    assert config == expected
    assert pipeline.ROOT / config["api"]["download_path"] == project_root / expected["api"]["download_path"]
    start, end, params = search_settings(CONFIG, date(2024, 2, 29))
    assert (start, end) == (date(2014, 2, 28), date(2024, 2, 29))
    assert params["judgmentStartDate"] == "2014-02-28"
    assert "all" not in params


@pytest.mark.parametrize(
    "arguments, expected",
    [
        ([], {}),
        (["--max-pages", "2"], {"max_pages": 2}),
        (["--max-gib", "5"], {"max_gib": 5.0}),
        (["--no-debug"], {"debug": False}),
        (["--debug"], {"debug": True}),
        (
            [
                "--timeout",
                "12.5",
                "--max-age-years",
                "3",
                "--download-path",
                "tmp/test",
            ],
            {"timeout": 12.5, "max_age_years": 3, "download_path": "tmp/test"},
        ),
    ],
)
def test_cli_overrides_only_explicit_arguments(arguments, expected):
    """
    Verify cli overrides only explicit arguments.

    args:
        arguments: Command-line arguments, or None for process arguments.
        expected: Expected result.

    returns:
        None.
    """
    options = vars(cli.parse_arguments(arguments))
    assert options.pop("config_dir") is None
    assert options == expected
    original = config_module.load_config()
    merged = config_module.load_config(overrides=options)
    assert merged["api"] == {**original["api"], **expected}


def test_cli_can_override_yaml_before_validation(config_directory):
    """
    Verify cli can override yaml before validation.

    args:
        config_directory: Isolated editable acquisition settings.

    returns:
        None.
    """
    import yaml

    api = {**CONFIG, "max_pages": -1, "debug": True}
    (config_directory / "api.yaml").write_text(yaml.safe_dump(api))
    options = vars(
        cli.parse_arguments(
            ["--config-dir", str(config_directory), "--max-pages", "1", "--no-debug"]
        )
    )
    config = config_module.load_config(options.pop("config_dir"), options)
    assert config["api"] == {**api, "max_pages": 1, "debug": False}


def test_removed_download_only_option_is_rejected():
    """
    Verify removed download only option is rejected.

    args:
        None

    returns:
        None.
    """
    with pytest.raises(SystemExit):
        cli.parse_arguments(["--download-only"])
