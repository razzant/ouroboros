"""The ``save_settings`` chokepoint that makes ``OUROBOROS_RUNTIME_MODE`` owner-only.

This module owns the on-disk old-vs-new mode comparison and the ``allow_elevation``
consent it demands, the boot baseline that closes the corrupt-disk roundtrip, the
``_set_tool_timeout`` live-flip chain that bypasses ``/api/settings``, the onboarding
positive where a launcher or wizard may set any initial mode, and the inertness of
the consent flag once the baseline is pinned — in this process and in a subprocess
that inherits its environment.

The remaining layers were split verbatim into ``tests/test_runtime_mode_data_write.py``
(the ``_data_write``/``_data_read`` fence), ``tests/test_runtime_mode_owner_endpoints.py``
(the settings API body and the owner endpoints),
``tests/test_runtime_mode_authorship.py`` (who may author a mode decision),
``tests/test_runtime_mode_launcher_bridges.py`` (the launcher bridges) and
``tests/test_runtime_mode_write_guards.py`` (the deterministic command/write guards);
their shared fixtures live in ``tests/_runtime_mode_elevation_shared.py``.

Hermetic — no network, no supervisor boot. Uses temp dirs for
``DATA_DIR`` / ``SETTINGS_PATH`` overrides via monkeypatching
``ouroboros.config`` module-level constants.
"""
from __future__ import annotations

import json
import os

import pytest

from tests._runtime_mode_elevation_shared import (
    _make_drive_ctx,
    _seed_disk,
)
from tests._runtime_mode_elevation_shared import isolated_settings as _isolated_settings

# The fixture is requested by name as a test parameter, so it is re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
isolated_settings = _isolated_settings


# ---------------------------------------------------------------------------
# 1. save_settings chokepoint
# ---------------------------------------------------------------------------


def test_save_settings_refuses_elevation_without_consent(isolated_settings):
    """Disk has light. Caller tries to save advanced without consent. Refused."""
    from ouroboros.config import save_settings

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})

    with pytest.raises(PermissionError) as exc:
        save_settings({"OUROBOROS_RUNTIME_MODE": "advanced"})
    assert "elevation refused" in str(exc.value)
    assert "light" in str(exc.value) and "advanced" in str(exc.value)
    # On-disk value must NOT have been changed.
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "light"


def test_save_settings_refuses_pro_elevation_from_advanced(isolated_settings):
    from ouroboros.config import save_settings

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "advanced"})

    with pytest.raises(PermissionError):
        save_settings({"OUROBOROS_RUNTIME_MODE": "pro"})


def test_save_settings_allows_elevation_with_explicit_flag(isolated_settings):
    """Owner-driven flow (launcher, onboarding, lifespan) passes ``allow_elevation=True``."""
    from ouroboros.config import save_settings

    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    save_settings(
        {"OUROBOROS_RUNTIME_MODE": "advanced", "OPENAI_API_KEY": "irrelevant"},
        allow_elevation=True,
    )
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "advanced"


def test_save_settings_allows_downgrade_without_consent(isolated_settings):
    """Lowering scope is always free."""
    from ouroboros.config import save_settings

    for old_mode, new_mode in (("pro", "advanced"), ("pro", "light"), ("advanced", "light")):
        _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": old_mode})
        save_settings({"OUROBOROS_RUNTIME_MODE": new_mode})
        on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
        assert on_disk["OUROBOROS_RUNTIME_MODE"] == new_mode


def test_save_settings_allows_same_mode(isolated_settings):
    """No elevation when in == out."""
    from ouroboros.config import save_settings

    for mode in ("light", "advanced", "pro"):
        _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": mode})
        save_settings({"OUROBOROS_RUNTIME_MODE": mode, "TOTAL_BUDGET": "42.0"})
        on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
        assert on_disk["OUROBOROS_RUNTIME_MODE"] == mode
        assert on_disk["TOTAL_BUDGET"] == "42.0"


def test_save_settings_initial_setup_uses_default_baseline(isolated_settings):
    """No on-disk settings yet -> baseline is the default ('advanced').
    Saving 'advanced' is same-mode; saving 'pro' would be elevation."""
    from ouroboros.config import save_settings

    # Initial advanced save (default baseline -> same mode).
    save_settings({"OUROBOROS_RUNTIME_MODE": "advanced"})
    assert isolated_settings.exists()
    # Initial pro save (default baseline -> elevation, blocked without consent).
    isolated_settings.unlink()
    with pytest.raises(PermissionError):
        save_settings({"OUROBOROS_RUNTIME_MODE": "pro"})


# ---------------------------------------------------------------------------
# 4. set_tool_timeout regression: cannot propagate a poisoned disk mode
# ---------------------------------------------------------------------------


def test_set_tool_timeout_cannot_smuggle_elevation(isolated_settings, monkeypatch):
    """Belt-and-braces regression: if a (theoretical) bypass of the
    data_write block ever lands a corrupted runtime_mode on disk, the
    save_settings chokepoint inside _set_tool_timeout still refuses to
    write it back. The function reads disk, modifies timeout only,
    saves — but the save raises PermissionError when the in-memory dict
    carries an elevated mode that the on-disk baseline does not.
    """
    from ouroboros.config import load_settings

    # Step 1: legitimate baseline = light.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    # Step 2: simulate corruption (this is what the attack chain WOULD do):
    #   data_write block now refuses, but if it ever got around it, the
    #   in-memory dict that _set_tool_timeout builds would be:
    #     {OUROBOROS_RUNTIME_MODE: 'advanced', OUROBOROS_TOOL_TIMEOUT_SEC: N}
    #   Manually craft that dict and feed it to save_settings — it must raise.
    from ouroboros.config import save_settings
    poisoned = {"OUROBOROS_RUNTIME_MODE": "advanced", "OUROBOROS_TOOL_TIMEOUT_SEC": 600}
    with pytest.raises(PermissionError):
        save_settings(poisoned)

    # Disk remains at light.
    assert json.loads(isolated_settings.read_text())["OUROBOROS_RUNTIME_MODE"] == "light"

    # And the legitimate _set_tool_timeout flow (load -> mutate timeout
    # -> save) still works because load_settings preserves the on-disk
    # mode unchanged, so the chokepoint sees no elevation.
    settings = load_settings()
    settings["OUROBOROS_TOOL_TIMEOUT_SEC"] = 600
    save_settings(settings)  # no PermissionError
    # JSON preserves the int type — compare against int, not str.
    assert json.loads(isolated_settings.read_text())["OUROBOROS_TOOL_TIMEOUT_SEC"] == 600


# ---------------------------------------------------------------------------
# 5. Onboarding can set initial mode via allow_elevation
# ---------------------------------------------------------------------------


def test_onboarding_can_set_initial_runtime_mode_pro(isolated_settings):
    """First-launch wizard / launcher can choose any starting mode via
    the explicit consent flag."""
    from ouroboros.config import save_settings

    save_settings({"OUROBOROS_RUNTIME_MODE": "pro"}, allow_elevation=True)
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "pro"


# ---------------------------------------------------------------------------
# 7. Boot-time baseline closes the disk-corruption-then-roundtrip loophole
# ---------------------------------------------------------------------------


def test_save_settings_uses_boot_baseline_when_pinned(isolated_settings):
    """Once the boot baseline is pinned, the chokepoint compares against
    that fixed value — out-of-process disk corruption cannot move the
    fence."""
    from ouroboros.config import (
        initialize_runtime_mode_baseline,
        save_settings,
    )

    # Owner started the run in light.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    initialize_runtime_mode_baseline("light")

    # An out-of-process write corrupts disk to "pro" (simulated directly here).
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "pro"})

    # Now any in-process save_settings call that would propagate "pro"
    # (the chain through _set_tool_timeout / api_settings_post / etc.)
    # MUST be refused — even though disk old equals incoming, the boot
    # baseline of "light" still wins.
    with pytest.raises(PermissionError):
        save_settings({"OUROBOROS_RUNTIME_MODE": "pro"})


def test_set_tool_timeout_sanitizes_corrupted_disk_to_env(isolated_settings, monkeypatch):
    """End-to-end regression for the iteration-1 GPT/Gemini finding:
    even if a subprocess managed to write ``pro`` directly to
    settings.json (bypassing both ``_data_write`` and the chokepoint),
    a follow-up ``_set_tool_timeout`` call must NOT propagate the
    elevation into ``os.environ``. ``_set_tool_timeout`` now anchors
    ``OUROBOROS_RUNTIME_MODE`` to the live env value before saving, so
    the corrupted disk gets restored, not propagated."""
    from ouroboros.config import initialize_runtime_mode_baseline
    from ouroboros.tools import control as control_mod

    # Owner started the run in light.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    initialize_runtime_mode_baseline("light")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")

    # Adversary corrupts disk directly to "pro".
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "pro"})

    # Agent calls set_tool_timeout (innocent-looking, POLICY_SKIP).
    ctx = _make_drive_ctx(isolated_settings.parent.parent)
    result = control_mod._set_tool_timeout(ctx, 60)
    assert "OK:" in result, result

    # Disk should now be restored to "light" (the live env value),
    # NOT "pro" (the corrupted disk read).
    on_disk = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert on_disk["OUROBOROS_RUNTIME_MODE"] == "light", (
        "set_tool_timeout propagated corrupted disk back to disk "
        "instead of restoring from env."
    )

    # And the live env must still be light (apply_settings_to_env was
    # called with the sanitised dict).
    assert os.environ.get("OUROBOROS_RUNTIME_MODE") == "light", (
        "set_tool_timeout propagated corrupted disk into os.environ — "
        "elevation chain succeeded."
    )


def test_save_settings_consent_inert_after_boot_baseline_pinned(isolated_settings):
    """Real triad finding T1: ``allow_elevation=True`` MUST become inert
    once the boot baseline is pinned. Otherwise an agent-reachable
    subprocess that imports ``save_settings`` could pass the public
    consent kwarg and bypass the rank check.

    Pre-pin: ``allow_elevation=True`` works (legitimate launcher /
    lifespan path).
    Post-pin: ``allow_elevation=True`` is silently demoted; rank check
    still applies and refuses the elevation.
    """
    from ouroboros.config import (
        initialize_runtime_mode_baseline,
        save_settings,
    )

    # Pre-pin path — owner-driven elevation works.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    save_settings({"OUROBOROS_RUNTIME_MODE": "pro"}, allow_elevation=True)
    assert json.loads(isolated_settings.read_text())["OUROBOROS_RUNTIME_MODE"] == "pro"

    # Pin the boot baseline to the legitimately-elevated mode (simulating
    # a real launcher run that came up in pro). Now further elevation
    # attempts must be refused — even with the consent flag.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    initialize_runtime_mode_baseline("light")
    with pytest.raises(PermissionError) as exc:
        save_settings({"OUROBOROS_RUNTIME_MODE": "pro"}, allow_elevation=True)
    # Error message must clarify that consent is inert after baseline pin.
    assert "boot baseline is pinned" in str(exc.value)


def test_initialize_baseline_prefers_inherited_boot_env_over_runtime_mode(isolated_settings, monkeypatch):
    """Iteration-2 real scope finding SR1: a fresh subprocess inherits
    BOTH ``OUROBOROS_RUNTIME_MODE`` and ``OUROBOROS_BOOT_RUNTIME_MODE``.
    The agent CAN clobber its own ``OUROBOROS_RUNTIME_MODE`` to poison
    the baseline, but ``OUROBOROS_BOOT_RUNTIME_MODE`` is owner-pinned
    and must take precedence in ``initialize_runtime_mode_baseline``."""
    from ouroboros import config as cfg
    from ouroboros.config import (
        BOOT_RUNTIME_MODE_ENV_KEY,
        initialize_runtime_mode_baseline,
        save_settings,
    )

    # Parent ran in light, exported the boot baseline via env.
    monkeypatch.setenv(BOOT_RUNTIME_MODE_ENV_KEY, "light")
    # Adversarial child mutates its own RUNTIME_MODE env to "pro" (this
    # is just the agent's own os.environ — no special privilege).
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")

    # Simulate fresh-subprocess module state (the in-memory pin is
    # absent because the module was just imported).
    monkeypatch.setattr(cfg, "_BOOT_RUNTIME_MODE", None, raising=True)

    # Init must read BOOT_RUNTIME_MODE_ENV_KEY (= "light"), NOT the
    # poisoned RUNTIME_MODE (= "pro"). Otherwise the child poisons its
    # own baseline and the chokepoint becomes ineffective.
    initialize_runtime_mode_baseline()
    assert cfg._BOOT_RUNTIME_MODE == "light", (
        "initialize_runtime_mode_baseline must prefer the inherited "
        "OUROBOROS_BOOT_RUNTIME_MODE over the (clobberable) "
        "OUROBOROS_RUNTIME_MODE."
    )

    # And the chokepoint should now refuse the elevation that the agent
    # tried to set up by poisoning RUNTIME_MODE.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    with pytest.raises(PermissionError):
        save_settings({"OUROBOROS_RUNTIME_MODE": "pro"}, allow_elevation=True)


def test_save_settings_consent_inert_in_subprocess_via_env_propagation(isolated_settings, monkeypatch):
    """Iteration-2 multi-critic finding F2-1 (verified empirically by
    Gemini): a fresh subprocess that re-imports ``ouroboros.config``
    starts with ``_BOOT_RUNTIME_MODE = None``, which previously let
    ``allow_elevation=True`` work again, defeating the chokepoint. The
    fix exports the pinned baseline to ``OUROBOROS_BOOT_RUNTIME_MODE``
    env var so subprocesses inherit it. This test simulates the
    subprocess scenario by clearing the in-memory pin while keeping
    the env var (which is what a fresh subprocess sees)."""
    from ouroboros import config as cfg
    from ouroboros.config import (
        BOOT_RUNTIME_MODE_ENV_KEY,
        initialize_runtime_mode_baseline,
        save_settings,
    )

    # Parent pins the baseline → env var is set.
    _seed_disk(isolated_settings, {"OUROBOROS_RUNTIME_MODE": "light"})
    initialize_runtime_mode_baseline("light")
    assert os.environ.get(BOOT_RUNTIME_MODE_ENV_KEY) == "light"

    # Simulate a fresh subprocess: clear the in-memory module global
    # (this is what a re-imported module looks like) but keep the env
    # var (which subprocess.Popen / mp.spawn inherit).
    monkeypatch.setattr(cfg, "_BOOT_RUNTIME_MODE", None, raising=True)
    assert os.environ.get(BOOT_RUNTIME_MODE_ENV_KEY) == "light"

    # An attempt to elevate via ``allow_elevation=True`` from the
    # "subprocess" must be refused — env-inherited baseline takes over.
    with pytest.raises(PermissionError) as exc:
        save_settings({"OUROBOROS_RUNTIME_MODE": "pro"}, allow_elevation=True)
    assert "env-var" in str(exc.value), (
        "Subprocess save_settings must report the baseline source as "
        "'env-var' so the operator can trace which path refused."
    )
