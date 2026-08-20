"""Which environment values may become file content, and which never may.

Settings travel in both directions: `apply_settings_to_env` projects the document
into `os.environ` so subprocesses inherit it, and `load_settings` overlays the
environment onto keys the file does not mention. That overlay is what lets a
benchmark or an operator forward a value for one run without editing anyone's
settings; it is also how a forwarded value can end up PERSISTED as an owner
decision by an unrelated save. Owner decision (spec 4.3.7, answer A): the current
split stands, and these tests are the record of it.

The split has three parts:

1. **Alias keys are never written.** A key that exists only as backwards
   compatibility for the environment (`OUROBOROS_MODEL_FALLBACK`, singular) is
   read where it is read and is not part of the settings vocabulary, so no save
   can put it on disk under either name.
2. **A canonical env value may be pinned by an EXPLICIT write.** `load_settings`
   returns it, and a caller that deliberately saves what it loaded persists it.
   That is the owner's escape hatch and stays available.
3. **...except for the keys that are disk-authored, where silence stays silence.**
   `_DISK_AUTHORED_SETTINGS` (the two context-mode keys and the safety mode) and
   `ENDPOINT_AUTHORED_SETTINGS` (the install-time facts) are ratchet or provenance
   surfaces: an environment value there is not an owner decision, so it is neither
   read into the document nor projected back out of one the file does not carry.
   `_settings_file_value` reads DISK ONLY for the same reason — a ratchet whose
   "previous value" came from the environment would let any subprocess open the
   gate by exporting the value it wants to move away from.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from ouroboros import config as cfg

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    settings_path = data_dir / "settings.json"
    monkeypatch.setattr(cfg, "DATA_DIR", data_dir, raising=True)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    for key in cfg.SETTINGS_DEFAULTS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("OUROBOROS_MODEL_FALLBACK", raising=False)
    cfg.reset_runtime_mode_baseline_for_tests()
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


def test_the_singular_fallback_alias_is_read_from_env_and_never_reaches_disk(
        isolated_settings, monkeypatch):
    """The env-only alias: a live benchmark contract on the read side, invisible on
    the write side. It is not in the settings vocabulary, so no save can persist it,
    and it does not silently become a value for the canonical plural key either."""
    from ouroboros import config as cfg

    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACK", "bench/only-chain")

    assert "OUROBOROS_MODEL_FALLBACK" not in cfg.SETTINGS_DEFAULTS
    assert "OUROBOROS_MODEL_FALLBACK" not in cfg.settings_env_keys()
    assert cfg.parse_fallback_chain() == ["bench/only-chain"], "the read-side alias is live"

    loaded = cfg.load_settings()
    assert "OUROBOROS_MODEL_FALLBACK" not in loaded
    assert loaded["OUROBOROS_MODEL_FALLBACKS"] == (
        cfg.SETTINGS_DEFAULTS["OUROBOROS_MODEL_FALLBACKS"]), "the alias leaked into the slot"

    cfg.save_settings(loaded)
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert "OUROBOROS_MODEL_FALLBACK" not in stored


def test_a_forwarded_canonical_value_is_read_and_can_be_pinned_by_an_explicit_write(
        isolated_settings, monkeypatch):
    """The owner's escape hatch, and the reason a forwarded value is not simply
    ignored: it applies for the run, and a deliberate save makes it durable."""
    from ouroboros import config as cfg

    monkeypatch.setenv("OUROBOROS_MODEL", "forwarded/main")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "17")

    loaded = cfg.load_settings()
    assert loaded["OUROBOROS_MODEL"] == "forwarded/main"
    assert loaded["OUROBOROS_MAX_ROUNDS"] == 17
    assert not isolated_settings.exists(), "reading a forwarded value pinned it"

    cfg.save_settings(loaded)
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert stored["OUROBOROS_MODEL"] == "forwarded/main"
    assert stored["OUROBOROS_MAX_ROUNDS"] == 17


def test_a_stored_value_wins_over_the_environment_for_an_ordinary_key(
        isolated_settings, monkeypatch):
    """The overlay fills SILENCE, it does not override the owner's file."""
    from ouroboros import config as cfg

    isolated_settings.write_text(json.dumps({"OUROBOROS_MODEL": "owner/choice"}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MODEL", "forwarded/main")

    assert cfg.load_settings()["OUROBOROS_MODEL"] == "owner/choice"


@pytest.mark.parametrize("key,env_value", [
    ("OUROBOROS_CONTEXT_MODE", "low"),
    ("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false"),
    ("OUROBOROS_SAFETY_MODE", "off"),
])
def test_a_disk_authored_key_is_never_read_out_of_the_environment(
        isolated_settings, monkeypatch, key, env_value):
    """These three are ratchets. An environment value is not authorship, so the
    document never picks one up — otherwise an ordinary load/save round-trip in a
    process whose environment says low/off would launder that value onto disk."""
    from ouroboros import config as cfg

    assert key in cfg._DISK_AUTHORED_SETTINGS
    monkeypatch.setenv(key, env_value)

    loaded = cfg.load_settings()
    assert loaded[key] == cfg.SETTINGS_DEFAULTS[key], "an env ratchet value reached the document"

    cfg.save_settings(loaded)
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert key not in stored, "silence did not stay silence"


def test_a_disk_authored_key_is_not_projected_back_out_of_a_silent_file(
        isolated_settings, monkeypatch):
    """The mirror direction: projecting a default the file never carried would
    clobber a legitimately forwarded value (a benchmark runner has no settings.json
    at all), so an unauthored key is left exactly as the environment has it."""
    from ouroboros import config as cfg

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")

    cfg.apply_settings_to_env(dict(cfg.SETTINGS_DEFAULTS))

    assert os.environ["OUROBOROS_CONTEXT_MODE"] == "low"
    assert os.environ["OUROBOROS_SAFETY_MODE"] == "off"

    # ...and once the FILE carries the key, the file is the authority again.
    isolated_settings.write_text(json.dumps({"OUROBOROS_CONTEXT_MODE": "max"}), encoding="utf-8")
    cfg.apply_settings_to_env(dict(cfg.SETTINGS_DEFAULTS, OUROBOROS_CONTEXT_MODE="max"))
    assert os.environ["OUROBOROS_CONTEXT_MODE"] == "max"


def test_install_time_facts_are_disk_only_in_both_directions(isolated_settings, monkeypatch):
    """`ENDPOINT_AUTHORED_SETTINGS` is stricter than the ratchets: those project
    once the file carries them, these never leave disk at all. An environment
    timestamp alone once closed the onboarding window on a fresh install."""
    from ouroboros import config as cfg

    for key in cfg.ENDPOINT_AUTHORED_SETTINGS:
        monkeypatch.setenv(key, "2020-01-01T00:00:00Z")
        assert key not in cfg.settings_env_keys()

    loaded = cfg.load_settings()
    for key in cfg.ENDPOINT_AUTHORED_SETTINGS:
        assert loaded[key] == cfg.SETTINGS_DEFAULTS[key]

    isolated_settings.write_text(
        json.dumps({key: "2026-01-01T00:00:00Z" for key in cfg.ENDPOINT_AUTHORED_SETTINGS}),
        encoding="utf-8")
    cfg.apply_settings_to_env(cfg.load_settings())
    for key in cfg.ENDPOINT_AUTHORED_SETTINGS:
        assert os.environ[key] == "2020-01-01T00:00:00Z", "a disk-only fact was projected"


def test_the_ratchet_previous_value_is_read_from_disk_only(isolated_settings, monkeypatch):
    """`_settings_file_value` is the ratchet's memory. Reading the environment there
    would turn ``max -> low`` into ``low -> low`` and open the gate for any process
    that can export a variable."""
    from ouroboros import config as cfg

    isolated_settings.write_text(json.dumps({"OUROBOROS_CONTEXT_MODE": "max"}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")

    assert cfg._settings_file_value("OUROBOROS_CONTEXT_MODE", "max") == "max"
    with pytest.raises(PermissionError, match="OUROBOROS_CONTEXT_MODE lowering refused"):
        cfg.save_settings({"OUROBOROS_CONTEXT_MODE": "low"})

    # A key the file does not carry answers the fail-closed default, never the env.
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    assert cfg._settings_file_value("OUROBOROS_SAFETY_MODE", "full") == "full"


def test_the_exemption_sets_are_exactly_the_declared_ones():
    """A structural pin so the two exemptions cannot grow or shrink unnoticed: each
    is a decision about who may author a value, not a convenience list."""
    from ouroboros import config as cfg

    assert cfg._DISK_AUTHORED_SETTINGS == (
        "OUROBOROS_CONTEXT_MODE", "OUROBOROS_CONTEXT_MODE_AUTO_LOW", "OUROBOROS_SAFETY_MODE")
    assert cfg.ENDPOINT_AUTHORED_SETTINGS == frozenset(
        {"OUROBOROS_SUBSCRIPTION_PRESET_VERSION", "OUROBOROS_ONBOARDING_COMPLETED_AT"})
    assert cfg.ENDPOINT_AUTHORED_SETTINGS <= cfg.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV
    # The exported set is DERIVED, never hand-kept: a new key exports by default and
    # an exclusion is a decision written into the one list.
    assert set(cfg.settings_env_keys()) == (
        set(cfg.SETTINGS_DEFAULTS) - set(cfg.SETTINGS_KEYS_NOT_EXPORTED_TO_ENV))
