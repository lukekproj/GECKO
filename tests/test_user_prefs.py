"""Tests for utility.user_prefs config.json loading."""

import json
import pytest
from utility import user_prefs

_TRACKED_CONSTANTS = [
    "KINARM_INVALID_ABS_THRESHOLD",
    "AUTO_INTERP_THRESHOLD_FRAMES",
    "DEFAULT_EYE_HEIGHT_M",
    "DEFAULT_VISUAL_ANGLE_DEG",
]

@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(user_prefs, "_exe_dir", lambda: tmp_path)
    for name in _TRACKED_CONSTANTS:
        monkeypatch.setattr(user_prefs, name, getattr(user_prefs, name))
    return tmp_path


def test_no_config_file_uses_defaults(config_dir):
    user_prefs._load_config()
    assert user_prefs.KINARM_INVALID_ABS_THRESHOLD == 99.9


def test_valid_override_is_applied(config_dir):
    (config_dir / "config.json").write_text(
        json.dumps({"AUTO_INTERP_THRESHOLD_FRAMES": 100})
    )
    user_prefs._load_config()
    assert user_prefs.AUTO_INTERP_THRESHOLD_FRAMES == 100


def test_unrecognized_key_is_ignored_not_crashed(config_dir):
    (config_dir / "config.json").write_text(
        json.dumps({"NOT_A_REAL_SETTING": 12345})
    )
    user_prefs._load_config()
    assert not hasattr(user_prefs, "NOT_A_REAL_SETTING")
    assert user_prefs.KINARM_INVALID_ABS_THRESHOLD == 99.9


def test_valid_and_unrecognized_keys_mixed(config_dir):
    (config_dir / "config.json").write_text(
        json.dumps({
            "DEFAULT_EYE_HEIGHT_M": 0.25,
            "Default_Visual_Angle": 7.0,
        })
    )
    user_prefs._load_config()
    assert user_prefs.DEFAULT_EYE_HEIGHT_M == 0.25
    assert user_prefs.DEFAULT_VISUAL_ANGLE_DEG == 5.0


def test_invalid_json_falls_back_to_defaults(config_dir):
    (config_dir / "config.json").write_text("{ this is not valid json, }")
    user_prefs._load_config()
    assert user_prefs.KINARM_INVALID_ABS_THRESHOLD == 99.9
    assert user_prefs.DEFAULT_EYE_HEIGHT_M == 0.2


def test_empty_config_file_uses_defaults(config_dir):
    (config_dir / "config.json").write_text("{}")
    user_prefs._load_config()
    assert user_prefs.KINARM_INVALID_ABS_THRESHOLD == 99.9