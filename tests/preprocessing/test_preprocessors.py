"""Behavioral checks for non-embedding predictor and target preprocessing."""

from copy import deepcopy
import json

import numpy as np
import pytest

from dataset.preprocessing import FeaturePreprocessor, TargetPreprocessor, load_config, load_records


@pytest.fixture
def settings():
    """Provide an ordered original-field selection.

    args:
        None.

    returns:
        Independent feature allowlist.
    """
    return {"feature_fields": ["prior_payment", "instance"]}


def test_fitted_blocks_are_stable_and_prediction_needs_no_target(settings, tmp_path):
    """Keep held-out samples from changing scales, vocabularies or column meanings.

    args:
        settings: Mixed-field policies.
        tmp_path: Temporary artifact destination.

    returns:
        None.
    """
    train = [{"prior_payment": 0, "instance": "pierwsza"}, {"prior_payment": 10, "instance": "druga"}]
    model = FeaturePreprocessor().fit(train, settings)
    assert not hasattr(model, "features_")
    batch = model.transform(train, ["1", "2"])
    np.testing.assert_allclose(batch["X"][:, 0], [-1, 1])
    assert batch["field_columns"] == {"prior_payment": [0, 1], "instance": [1, 4]}
    assert batch["columns"] == ["prior_payment", "instance=druga", "instance=not_available", "instance=pierwsza"]
    np.testing.assert_array_equal(batch["X"][:, 1:], [[0, 0, 1], [1, 0, 0]])
    state = deepcopy(model.transformers_["prior_payment"].named_steps["scale"].center_)
    held_out = model.transform([{"prior_payment": 1000, "instance": "new"}], ["3"])
    assert "y" not in held_out
    np.testing.assert_array_equal(held_out["X"][0, 1:], 0)
    np.testing.assert_array_equal(model.transformers_["prior_payment"].named_steps["scale"].center_, state)
    missing = model.transform([{"prior_payment": 0, "instance": None}], ["4"])
    np.testing.assert_array_equal(missing["X"], [[-1, 0, 1, 0]])
    path = tmp_path / "preprocessing.pkl"
    model.save(path)
    restored = FeaturePreprocessor.load(path, settings=settings)
    np.testing.assert_array_equal(restored.transform(train, ["1", "2"])["X"], batch["X"])
    changed = deepcopy(settings)
    changed["feature_fields"].reverse()
    with pytest.raises(ValueError, match="field order"):
        FeaturePreprocessor.load(path, settings=changed)


@pytest.mark.parametrize("transform", ["identity", "log1p"])
@pytest.mark.parametrize("scaling", ["none", "standard", "robust"])
def test_target_roundtrip_and_constant_values(transform, scaling):
    """Recover PLN values for all supported reversible transformations.

    args:
        transform: Pointwise target transformation.
        scaling: Fitted scaling method.

    returns:
        None.
    """
    settings = {"transform": transform, "scaling": scaling}
    target = TargetPreprocessor().fit([0, 100, 10000], settings)
    values = [0, 15, 200000]
    np.testing.assert_allclose(target.inverse_transform(target.transform(values)), values, atol=1e-8)
    target.fit([100, 100], settings)
    np.testing.assert_allclose(target.inverse_transform(target.transform([100, 200])), [100, 200])


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), "first", None, -1])
def test_invalid_targets_fail(value):
    """Reject malformed and unconfirmed monetary labels.

    args:
        value: Invalid target value.

    returns:
        None.
    """
    with pytest.raises(ValueError):
        TargetPreprocessor().fit([value], {"transform": "identity", "scaling": "none"})


def test_constants_missing_values_and_unknown_categories(settings):
    """Handle constant training columns while preserving real predictor zeros.

    args:
        settings: Mixed-field policies.

    returns:
        None.
    """
    records = [{"prior_payment": 0, "instance": "pierwsza"}] * 2
    model = FeaturePreprocessor().fit(records, settings)
    batch = model.transform([{"prior_payment": 0, "instance": "not_available"}], ["1"])
    assert np.isfinite(batch["X"]).all() and batch["X"][0, 0] == 0
    with pytest.raises(ValueError, match="Missing numeric"):
        model.transform([{"prior_payment": None, "instance": "pierwsza"}], ["2"])
    unknown = model.transform([{"prior_payment": 0, "instance": "new"}], ["3"])
    np.testing.assert_array_equal(unknown["X"][0, 1:], 0)


@pytest.mark.parametrize("field", ["total", "valid", "source", "model_calls", "invented"])
def test_predictor_allowlist_rejects_labels_text_and_unknown_fields(settings, field):
    """Keep targets and monetary/source evidence outside tabular predictors.

    args:
        settings: Valid field policies.
        field: Unsupported predictor name.

    returns:
        None.
    """
    settings["feature_fields"].append(field)
    with pytest.raises(ValueError, match="Unsupported"):
        FeaturePreprocessor().fit([], settings)


def test_input_schema_and_numeric_validation(settings):
    """Reject missing keys, numeric category coercion and misaligned rows.

    args:
        settings: Mixed-field policies.

    returns:
        None.
    """
    model = FeaturePreprocessor().fit([{"prior_payment": 0, "instance": "pierwsza"}], settings)
    for records in ([{"instance": "pierwsza"}], [{"prior_payment": "100", "instance": "pierwsza"}], [{"prior_payment": 0, "instance": 1}]):
        with pytest.raises(ValueError):
            model.transform(records, ["1"])
    with pytest.raises(ValueError, match="sample IDs"):
        model.transform([{"prior_payment": 0, "instance": "pierwsza"}], [])


def test_local_input_filtering_and_prediction_without_awards(tmp_path):
    """Read valid supervised samples and allow prediction without award files.

    args:
        tmp_path: Temporary dataset root.

    returns:
        None.
    """
    for case_id, valid in [("1", True), ("2", False)]:
        folder = tmp_path / case_id
        folder.mkdir()
        (folder / "record.json").write_text(json.dumps({"prior_payment": 0}))
        (folder / "award.json").write_text(json.dumps({"total": "100", "valid": valid}))
    batch = load_records(tmp_path, include_targets=True)
    assert batch["sample_ids"] == ["1"] and batch["targets"] == ["100"]
    for path in tmp_path.glob("*/award.json"):
        path.unlink()
    assert load_records(tmp_path)["sample_ids"] == ["1", "2"]
    with pytest.raises(FileNotFoundError):
        load_records(tmp_path, include_targets=True)


def test_public_yaml_and_nonnegative_logarithm(project_root):
    """Use explicit per-field policies and reject invalid logarithmic inputs.

    args:
        project_root: Repository configuration and fixture root.
        None.

    returns:
        None.
    """
    config = load_config(project_root / "configs/training/model_input.yaml")
    records = [{"judgment_year": 2016, "injured_party_count": 1, "prior_payment": 0,
                "instance": "pierwsza", "court_locality_size": "not_available"}]
    model = FeaturePreprocessor().fit(records, config)
    assert np.isfinite(model.transform(records, ["1"])["X"]).all()
    records[0]["prior_payment"] = -2
    with pytest.raises(ValueError, match="nonnegative"):
        model.transform(records, ["1"])


def test_unfitted_and_incompatible_artifacts_fail(settings, tmp_path):
    """Reject operations before fit and incompatible artifact versions.

    args:
        settings: Mixed-field policies.
        tmp_path: Temporary artifact directory.

    returns:
        None.
    """
    import pickle
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        FeaturePreprocessor().transform([], [])
    records = [{"prior_payment": 0, "instance": "pierwsza"}]
    model = FeaturePreprocessor().fit(records, settings)
    path = tmp_path / "state.pkl"
    model.save(path)
    payload = pickle.loads(path.read_bytes())
    payload["version"] = -1
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError, match="version"):
        FeaturePreprocessor.load(path)


@pytest.mark.parametrize("override", ["scaling", "fields", "target", "transform", "missing"])
def test_yaml_rejects_preprocessing_overrides(settings, override):
    """Keep the model-input interface limited to original feature selection.

    args:
        settings: Ordered model feature selection.
        override: Unsupported preprocessing configuration key.

    returns:
        None.
    """
    settings[override] = "standard"
    with pytest.raises(ValueError, match="only feature_fields"):
        FeaturePreprocessor().fit([], settings)


def test_year_digits_are_fixed_and_keep_column_mapping(project_root, tmp_path):
    """Encode year digits independently of the observed training-year range.

    args:
        project_root: Repository configuration and fixture root.
        tmp_path: Temporary fitted-state destination.

    returns:
        None.
    """
    config = load_config(project_root / "configs/training/model_input.yaml")
    config["feature_fields"] = ["judgment_year"]
    model = FeaturePreprocessor().fit([{"judgment_year": 2016}], config)
    years = [2000, 2009, 2018, 2019, 2020, 2025, 2099]
    records = [{"judgment_year": year} for year in years]
    ids = list(map(str, years))
    result = model.transform(records, ids)
    expected = [[0, 0], [0, .9], [.1, .8], [.1, .9], [.2, 0], [.2, .5], [.9, .9]]
    np.testing.assert_allclose(result["X"], expected)
    assert result["columns"] == ["judgment_year_tens", "judgment_year_units"]
    assert result["field_columns"] == {"judgment_year": [0, 2]}
    path = tmp_path / "year.pkl"
    model.save(path)
    np.testing.assert_array_equal(FeaturePreprocessor.load(path, settings=config).transform(records, ids)["X"], result["X"])
    for value in (1999, 2100, 2025.5, None, "not_available"):
        with pytest.raises(ValueError, match="integer years"):
            model.transform([{"judgment_year": value}], ["invalid"])


def test_original_feature_selection_keeps_text_outside_tabular_scaling():
    """Select raw model fields before their representations expand into columns.

    args:
        None.

    returns:
        None.
    """
    selection = {"feature_fields": ["cause", "instance", "damage", "judgment_year", "limitations"]}
    records = [{"cause": "Opis przyczyny", "damage": "Opis szkody", "limitations": "Opis ograniczeń",
                "instance": "pierwsza", "judgment_year": 2025, "prior_payment": 500}]
    model = FeaturePreprocessor().fit(records, selection)
    result = model.transform(records, ["1"])
    assert list(result["field_columns"]) == ["instance", "judgment_year"]
    assert result["columns"][-2:] == ["judgment_year_tens", "judgment_year_units"]
    np.testing.assert_array_equal(result["X"][0, -2:], [.2, .5])
    assert model.settings_ == selection
    assert records[0]["cause"] == "Opis przyczyny"


@pytest.mark.parametrize("fields", [[], ["instance", "instance"], [1], "instance"])
def test_feature_list_validation(fields):
    """Reject empty, duplicated and malformed original feature selections.

    args:
        fields: Invalid feature list.

    returns:
        None.
    """
    with pytest.raises(ValueError):
        FeaturePreprocessor().fit([], {"feature_fields": fields})


def test_default_checkpoint_path_is_independent_of_working_directory(
    project_root,
    settings,
    tmp_path,
    monkeypatch,
):
    """Save and restore default preprocessing state outside the current directory.

    args:
        project_root: Repository configuration and fixture root.
        settings: Original model feature selection.
        tmp_path: Isolated filesystem root.
        monkeypatch: Scoped checkpoint-path and working-directory replacement.

    returns:
        None.
    """
    from dataset.storage import preprocessing as artifacts

    assert artifacts.DEFAULT_PATH == project_root / "checkpoints/preprocessing/feature_preprocessor.pkl"
    destination = tmp_path / "checkpoints/preprocessing/feature_preprocessor.pkl"
    monkeypatch.setattr(artifacts, "DEFAULT_PATH", destination)
    working = tmp_path / "other"
    working.mkdir()
    monkeypatch.chdir(working)
    records = [{"prior_payment": 0, "instance": "pierwsza"}]
    model = FeaturePreprocessor().fit(records, settings)
    model.save()
    assert destination.is_file()
    assert not (working / "checkpoints").exists()
    restored = FeaturePreprocessor.load(settings=settings)
    np.testing.assert_array_equal(restored.transform(records, ["1"])["X"], model.transform(records, ["1"])["X"])


def test_target_scaling_is_saved_separately_and_never_refits(
    project_root,
    settings,
    tmp_path,
    monkeypatch,
):
    """Keep target statistics independent of predictor artifacts and held-out amounts.

    args:
        project_root: Repository configuration and fixture root.
        settings: Original predictor selection.
        tmp_path: Isolated checkpoint directory.
        monkeypatch: Scoped default-path replacements.

    returns:
        None.
    """
    from dataset.storage import preprocessing as artifacts

    assert artifacts.DEFAULT_TARGET_PATH == project_root / "checkpoints/preprocessing/target_preprocessor.pkl"
    predictor_path = tmp_path / "feature_preprocessor.pkl"
    target_path = tmp_path / "target_preprocessor.pkl"
    monkeypatch.setattr(artifacts, "DEFAULT_PATH", predictor_path)
    monkeypatch.setattr(artifacts, "DEFAULT_TARGET_PATH", target_path)
    records = [{"prior_payment": 0, "instance": "pierwsza"}]
    preprocessor = FeaturePreprocessor().fit(records, settings)
    target = TargetPreprocessor().fit([100, 200, 300])
    np.testing.assert_allclose(target.transform([100, 200, 300]), [-1, 0, 1])
    before = target.transform([100, 200, 300]).copy()
    target.transform([10000000])
    np.testing.assert_array_equal(target.transform([100, 200, 300]), before)
    preprocessor.save()
    target.save()
    assert predictor_path.is_file() and target_path.is_file()
    predictor_bytes = predictor_path.read_bytes()
    restored_target = TargetPreprocessor.load()
    np.testing.assert_allclose(restored_target.inverse_transform([-1, 0, 1]), [100, 200, 300])
    np.testing.assert_array_equal(restored_target.transform([100, 200, 300]), before)
    assert not hasattr(FeaturePreprocessor.load(), "target_")
    target_path.unlink()
    assert "X" in FeaturePreprocessor.load().transform(records, ["1"])
    assert predictor_path.read_bytes() == predictor_bytes
    with pytest.raises(FileNotFoundError):
        TargetPreprocessor.load()
    with pytest.raises(ValueError, match="target transformer"):
        TargetPreprocessor.load(predictor_path)
    target.save()
    with pytest.raises(ValueError, match="preprocessor"):
        FeaturePreprocessor.load(target_path)


@pytest.mark.parametrize("kind", ["features", "target"])
def test_previous_package_artifacts_remain_loadable(tmp_path, kind):
    """Restore fitted version-4 artifacts saved with the previous import namespace.

    args:
        tmp_path: Isolated trusted artifact directory.
        kind: Predictor or target preprocessing artifact to restore.

    returns:
        None.
    """
    import pickle

    import sklearn

    from dataset.storage.preprocessing import VERSION

    records = [{"prior_payment": 0}, {"prior_payment": 100}]
    settings = {"feature_fields": ["prior_payment"]}
    model = (
        FeaturePreprocessor().fit(records, settings)
        if kind == "features"
        else TargetPreprocessor().fit([100, 200])
    )
    payload = {"version": VERSION, "sklearn": sklearn.__version__, "state": model}
    content = pickle.dumps(payload, protocol=0).replace(
        b"cdataset.preprocessing.", b"csrc.dataset.preprocessing."
    )
    assert b"csrc.dataset.preprocessing." in content
    path = tmp_path / "previous.pkl"
    path.write_bytes(content)
    restored = type(model).load(path)
    if kind == "features":
        np.testing.assert_array_equal(
            restored.transform(records, ["1", "2"])["X"],
            model.transform(records, ["1", "2"])["X"],
        )
    else:
        np.testing.assert_array_equal(
            restored.inverse_transform(restored.transform([100, 200])), [100, 200]
        )
    assert path.read_bytes() == content
