"""D04 (owner 1B): the flat wall-clock timeout pair is GONE, not inert.

``OUROBOROS_SOFT_TIMEOUT_SEC`` and ``OUROBOROS_HARD_TIMEOUT_SEC`` stopped
terminating anything when the activity model (idle window + subtree liveness +
absolute ceiling) replaced them. What survived was worse than a dead knob: the
Settings UI accepted a number, the save response apologised for it, the
supervisor logged a deprecation row about it, ``/status`` rendered it, and the
queue overwrote whatever was passed with the same two constants. Five surfaces
discussing a value none of them obeyed.

7.0 retires the pair through the existing idiom — membership in
``RETIRED_SETTING_KEYS``, which strips the keys on load — so the assertions here
are about ABSENCE, and absence is what silently comes back. Each one names the
surface that used to carry the ghost.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
RETIRED = ("OUROBOROS_SOFT_TIMEOUT_SEC", "OUROBOROS_HARD_TIMEOUT_SEC")


def test_the_pair_is_retired_rather_than_defaulted():
    from ouroboros.settings_defaults import RETIRED_SETTING_KEYS, SETTINGS_DEFAULTS

    for key in RETIRED:
        assert key in RETIRED_SETTING_KEYS, key
        assert key not in SETTINGS_DEFAULTS, key


def test_a_stored_value_is_stripped_on_load_and_never_reaches_effective_settings(
    tmp_path, monkeypatch,
):
    """The whole migration contract in one pass: an install upgrading with the
    keys on disk loads WITHOUT them, and the env plane cannot smuggle them back
    (only SETTINGS_DEFAULTS keys are overlaid from the environment)."""
    import json

    from ouroboros import config

    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({
        "OUROBOROS_SOFT_TIMEOUT_SEC": 1234,
        "OUROBOROS_HARD_TIMEOUT_SEC": 5678,
        "OUROBOROS_MAX_WORKERS": 7,
    }), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", settings_path)
    for key in RETIRED:
        monkeypatch.setenv(key, "999")

    loaded = config.load_settings()

    assert int(loaded["OUROBOROS_MAX_WORKERS"]) == 7  # the round trip still works
    for key in RETIRED:
        assert key not in loaded, key


@pytest.mark.parametrize("module_name", ["supervisor.queue", "supervisor.workers"])
def test_no_supervisor_module_still_carries_a_timeout_global(module_name):
    """Both modules kept a module-global pinned to a constant, which read as a
    live tunable to anyone grepping for the setting name."""
    import importlib

    module = importlib.import_module(module_name)
    for name in ("SOFT_TIMEOUT_SEC", "HARD_TIMEOUT_SEC"):
        assert not hasattr(module, name), f"{module_name}.{name}"


def test_neither_init_still_asks_a_caller_for_a_value_it_discards():
    """``queue.init`` took the two numbers only to compare them against the
    constants it then wrote anyway; ``workers.init`` existed to forward them."""
    from supervisor import queue, workers

    assert list(inspect.signature(queue.init).parameters) == ["drive_root"]
    assert "soft_timeout" not in inspect.signature(workers.init).parameters
    assert "hard_timeout" not in inspect.signature(workers.init).parameters


def test_the_deprecation_event_path_is_gone_with_its_keys():
    """A retired key has nothing left to be loud about; keeping the emitter
    would leave a dead branch that only a future non-default value could
    reach — and there can be no future value."""
    from supervisor import queue

    source = inspect.getsource(queue)
    assert "deprecated_settings_ignored" not in source
    assert not hasattr(queue, "_emit_timeout_deprecation_once")
    assert not hasattr(queue, "_timeout_deprecation_emitted")


def test_status_no_longer_renders_a_legacy_timeout_line():
    """``/status`` printed ``legacy_timeouts_ignored: soft=600s, hard=1800s``
    on every request — an owner-visible report of two constants."""
    from supervisor import state

    assert list(inspect.signature(state.status_text).parameters) == [
        "workers_dict", "pending_list", "running_dict",
    ]
    assert "legacy_timeouts_ignored" not in inspect.getsource(state.status_text)


def test_no_runtime_or_settings_surface_still_names_either_key():
    """The grep-class assertion. The N−1 audit fixture is the ONE file that
    must keep them: it is a frozen byte-copy of a v6.113.4 settings document,
    and the RC auditor's retired-setting finding is proven against it."""
    allowed = {
        "ouroboros/settings_defaults.py",       # the RETIRED_SETTING_KEYS entry
        "scripts/rc_audit.py",                  # RETIRED_IN_THIS_ABI
        "tests/fixtures/nminus1/settings_v6.113.4.json",
        "tests/test_legacy_timeout_retirement.py",
        "tests/test_rc_audit_fixture_suite.py",
        "tests/test_settings_honesty.py",
        "tests/test_heartbeat_presentation.py",
        "docs/ARCHITECTURE.md",                 # the retirement is documented
        "ADOPTION_v7next.md",                   # ...and adopted: the D04 row names
                                                # what it retired, same as the already
                                                # skipped docs/v7next/ ledger. A record
                                                # of a removal is not a live surface
    }
    offenders = []
    for pattern in ("*.py", "*.js", "*.json", "*.md", "*.html"):
        for path in REPO.rglob(pattern):
            rel = path.relative_to(REPO).as_posix()
            if rel in allowed or rel.startswith(("venv", "node_modules", "docs/v7next/", "docs/archive/")):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if any(key in text for key in RETIRED):
                offenders.append(rel)
    assert sorted(offenders) == [], offenders


def test_the_rc_auditor_reports_the_pair_as_removed_in_this_window():
    """Retirement without a migration surface is a silent data loss. The
    auditor must class these as 7.0's OWN removals, not pre-existing rot."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_rc_audit_d04", REPO / "scripts" / "rc_audit.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = {c["key"]: c for c in module.build_scope()["checks"]
            if c["id"] == "retired-setting"}
    for key in RETIRED:
        assert rows[key]["since"] == "7.0", key
        assert rows[key]["behavior"] == "stripped-on-load"


def test_the_ledger_row_is_not_a_second_dispatcher_for_the_status_line():
    """Guards the shape of the removal itself: ``status_text`` must not have
    grown a replacement knob while losing the old one."""
    tree = ast.parse((REPO / "supervisor" / "state.py").read_text(encoding="utf-8"))
    fn = next(node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name == "status_text")
    names = {node.value for node in ast.walk(fn)
             if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert not any("TIMEOUT_SEC" in name for name in names)
