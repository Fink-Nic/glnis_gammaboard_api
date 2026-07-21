# type:ignore
import pytest


def test_rust_extension_health_check():
    from glnis import _rust

    assert _rust.health_check() == "glnis-gammaboard-api rust extension ok"


def test_rust_state_folder_classification_for_missing_path(tmp_path):
    from glnis import _rust

    missing_state = tmp_path / "missing-state"

    assert _rust.classify_state_folder(missing_state) == "missing"


def test_state_summary_rejects_non_state_folder(tmp_path):
    from glnis import _rust

    with pytest.raises(ValueError, match="not a saved GammaLoop state folder"):
        _rust.parse_state_summary_json(tmp_path)
