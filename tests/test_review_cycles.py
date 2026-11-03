"""Shared review-cycle cap (``ouroboros/review_cycles.py``) — Phase A′ of the
plan-review redesign sprint (owner decisions D9/D10/D19/D20; roast F3/F9/F11/F12).

Covers: strict parsing (string-typed key, ``unlimited`` aliases, fail-closed
default), env-or-default precedence, the retired acceptance-pass key migrated into
the shared knob at load, the commit gate reading the live value (and the spent
allowance's message per enforcement), the settings POST boundary, the UI/JS
binding, the durable exhaustion event and the module size bounds. The acceptance
pass formula, its Required+Blocking binding in ``task_pacing`` and the per-root
acceptance wallet left with task acceptance (owner 17.10, Q2-A)."""

from __future__ import annotations

import json
import logging
import pathlib
import types

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import config as cfg
from ouroboros import review_cycles as rc

REPO = pathlib.Path(__file__).resolve().parents[1]
KEY = "OUROBOROS_REVIEW_MAX_CYCLES"
LEGACY = "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv(LEGACY, raising=False)
    monkeypatch.setattr(rc, "_WARNED", set())
    yield


# ---------------------------------------------------------------------------
# Parsing / setting shape (F3: STRING-typed default so "unlimited" survives coercion)


def test_default_is_string_two_and_coercion_keeps_unlimited():
    assert cfg.SETTINGS_DEFAULTS[KEY] == "2"
    assert isinstance(cfg.SETTINGS_DEFAULTS[KEY], str)
    # An int default would silently coerce "unlimited" back to the default.
    assert cfg._coerce_setting_value(KEY, "unlimited") == "unlimited"
    assert cfg._coerce_setting_value(KEY, 3) == "3"
    assert rc.default_review_max_cycles() == 2
    # The key is projected into the environment like every other setting.
    assert KEY in cfg.settings_env_keys()
    # The legacy acceptance key is RETIRED: it has no default and is migrated at load.
    assert LEGACY not in cfg.SETTINGS_DEFAULTS
    assert LEGACY in cfg.RETIRED_SETTING_KEYS


@pytest.mark.parametrize("raw,expected", [
    ("2", 2), (" 5 ", 5), (1, 1), ("+3", 3),
    ("unlimited", None), ("UNLIMITED", None), ("inf", None), ("∞", None),
])
def test_parse_accepts_positive_ints_and_unlimited_aliases(raw, expected):
    assert rc.parse_review_max_cycles(raw) == expected
    assert rc.is_valid_review_max_cycles(raw)
    assert rc.normalize_review_max_cycles(raw) == ("unlimited" if expected is None else str(expected))


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "", None, "1.5", "true", "2 cycles", True, "none"])
def test_parse_rejects_garbage(raw):
    with pytest.raises((TypeError, ValueError)):
        rc.parse_review_max_cycles(raw)
    assert not rc.is_valid_review_max_cycles(raw)


def test_getter_env_or_default_and_fail_closed_logged_once(monkeypatch, caplog):
    assert rc.review_max_cycles() == 2
    monkeypatch.setenv(KEY, "5")
    assert rc.review_max_cycles() == 5
    monkeypatch.setenv(KEY, "Unlimited")
    assert rc.review_max_cycles() is None
    monkeypatch.setenv(KEY, "∞")
    assert rc.review_max_cycles() is None
    caplog.set_level(logging.WARNING, logger=rc.__name__)
    for bad in ("0", "-1", "abc"):
        monkeypatch.setenv(KEY, bad)
        caplog.clear()
        assert rc.review_max_cycles() == 2  # fail-closed to the bounded default
        assert any(KEY in rec.getMessage() and repr(bad) in rec.getMessage() for rec in caplog.records)
        caplog.clear()
        assert rc.review_max_cycles() == 2
        assert not caplog.records  # once per distinct bad value
    # An empty env value means "unset" (settings loader pops empty values).
    monkeypatch.setenv(KEY, "")
    assert rc.review_max_cycles() == 2


def test_settings_file_roundtrip_projects_unlimited_into_env(monkeypatch, tmp_path):
    # apply_settings_to_env writes os.environ directly. SETENV-then-DELENV is
    # what registers the key's original (absent) state with monkeypatch:
    # pytest's delenv on an ABSENT key records NOTHING, so the previous
    # delenv-only registration silently LEAKED the projected "3" into sibling
    # tests on the same xdist worker (observed: improvement-pass caps reading
    # cycles=3 in test_review_verification_v6544).
    monkeypatch.setenv(KEY, "registered-for-restore")
    monkeypatch.delenv(KEY)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({KEY: "unlimited"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    loaded = cfg.load_settings()
    assert loaded[KEY] == "unlimited"
    cfg.apply_settings_to_env(loaded)
    assert rc.review_max_cycles() is None
    settings_path.write_text(json.dumps({KEY: 3}), encoding="utf-8")
    loaded = cfg.load_settings()
    assert loaded[KEY] == "3"
    cfg.apply_settings_to_env(loaded)
    assert rc.review_max_cycles() == 3


# ---------------------------------------------------------------------------
# Commit gate reads the live value: the shared knob bounds PAID triad+scope
# cycles per ROOT task (Q16/Q23); the identical-diff refusal is knob-independent
# and covered in tests/test_commit_gate.py + tests/test_review_cycles_gates.py.


def _seed_paid_attempts(tmp_path, count, *, start=1, root="root-1", status="succeeded",
                        block_reason="", block_class="", phase="commit", paid=True):
    from ouroboros.review_state import CommitAttemptRecord, make_repo_key, update_state, _utc_now

    repo_key = make_repo_key(pathlib.Path(tmp_path))

    def _mutate(state):
        for i in range(count):
            state.attempts.append(CommitAttemptRecord(
                ts=_utc_now(), commit_message="msg", status=status,
                block_reason=block_reason, block_class=block_class,
                repo_key=repo_key, tool_name="commit_reviewed", task_id=root,
                attempt=start + i, phase=phase, paid=paid,
                pre_review_fingerprint=f"fp-{start + i}",
                root_task_id=root,
            ))
    update_state(pathlib.Path(tmp_path), _mutate)


@pytest.mark.parametrize("enforcement, mode, owner_menu", [
    ("advisory", "pro", False), ("blocking", "cyber_pro", False), ("blocking", "pro", True),
])
def test_a_spent_allowance_leaves_the_decision_to_the_mind_unless_blocking(monkeypatch, tmp_path, enforcement, mode, owner_menu):
    """Advisory (Cyber Pro included) decides itself whether what was reviewed is enough; only Blocking
    names the owner's ceiling as the way on (owner 17.10, Q1-A)."""
    from ouroboros.tools.commit_gate import check_review_cycles_ceiling

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    ctx = types.SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
    _seed_paid_attempts(tmp_path, 2)
    message = check_review_cycles_ceiling(ctx, root_task_id="root-1")["message"]
    assert ("escalate to the owner" in message and "3/5/unlimited" in message) is owner_menu, message
    assert ("Decide whether what was reviewed is enough" in message) is not owner_menu, message


def test_commit_gate_ceiling_reads_live_setting(monkeypatch, tmp_path):
    from ouroboros.tools.commit_gate import (
        check_review_cycles_ceiling,
        count_paid_review_cycles,
    )

    ctx = types.SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
    assert rc.review_max_cycles() == 2  # default moved 3 -> 2 (D20, disclosed)
    _seed_paid_attempts(tmp_path, 1)
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 1
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is None
    _seed_paid_attempts(tmp_path, 1, start=2)  # 2 paid == cap: refused
    exhausted = check_review_cycles_ceiling(ctx, root_task_id="root-1")
    assert exhausted is not None and "REVIEW_CYCLES_EXHAUSTED" in exhausted["message"]
    assert exhausted["cycles_paid"] == 2 and exhausted["cap"] == 2
    # Another root task has its own ceiling.
    assert check_review_cycles_ceiling(ctx, root_task_id="root-2") is None
    # An unknown root never gates (fail-open, disclosed).
    assert check_review_cycles_ceiling(ctx, root_task_id="") is None
    # Raise the shared cap: the SAME ledger is below it again.
    monkeypatch.setenv(KEY, "4")
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is None
    _seed_paid_attempts(tmp_path, 2, start=3)
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is not None
    # Unlimited: never a ceiling, however many paid cycles.
    monkeypatch.setenv(KEY, "unlimited")
    _seed_paid_attempts(tmp_path, 10, start=5)
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is None
    # Garbage fails closed to the bounded default (2), never to "no cap".
    monkeypatch.setenv(KEY, "lots")
    assert check_review_cycles_ceiling(ctx, root_task_id="root-1") is not None


@pytest.mark.parametrize("enforcement, reason", [
    ("blocking", "identical_diff_refused"), ("blocking", "review_cycles_exhausted"), ("advisory", "identical_diff_refused"),
])
def test_replay_disclosure_preserves_recorded_outcome(monkeypatch, enforcement, reason):
    """Every replay emits one progress note and one disclosure carrying the prior outcome."""
    from ouroboros.tools.commit_gate import disclose_commit_review_replay

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    notes, advisory = [], []
    ctx = types.SimpleNamespace(_review_advisory=advisory, emit_progress_fn=notes.append)
    disclose_commit_review_replay(ctx, {"replay_reason": reason, "advisory_replay": "Prior outcome: FAIL."})
    [_note], [disclosure] = notes, advisory
    assert reason in disclosure
    assert disclosure.endswith("Prior outcome: FAIL.")


def test_commit_scope_supplies_intent_when_goal_is_empty():
    from ouroboros.tools.review_helpers import resolve_intent

    assert resolve_intent("", "only the gate wording", "subject: x") == ("only the gate wording", "scope")


def test_paid_cycle_count_counts_dispatched_money_only(tmp_path, monkeypatch):
    """machine-5 contract: the ceiling counts MONEY. Every row that physically
    dispatched a wave (paid=True) counts whatever its terminal — verdict-block,
    pass, success, quorum-failed infra-block, crashed/expired failure alike.
    Only undispatched rows (paid=False: legacy history, free replays,
    preflight refusals) stay outside the count."""
    from ouroboros.tools.commit_gate import count_paid_review_cycles

    monkeypatch.delenv(KEY, raising=False)
    ctx = types.SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="root-1")
    _seed_paid_attempts(tmp_path, 1, status="blocked", block_reason="critical_findings",
                        block_class="verdict", phase="blocking_review")
    _seed_paid_attempts(tmp_path, 1, start=2, status="reviewed", phase="review_only")
    _seed_paid_attempts(tmp_path, 1, start=3)  # succeeded
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 3
    # Dispatched-then-degraded waves spent reviewer money: they count.
    _seed_paid_attempts(tmp_path, 1, start=4, status="blocked", block_reason="review_quorum",
                        block_class="infra", phase="blocking_review")
    _seed_paid_attempts(tmp_path, 1, start=5, status="failed", block_reason="infra_failure",
                        phase="expired")
    _seed_paid_attempts(tmp_path, 1, start=6, status="failed",
                        block_reason="post_commit_tests_failed", phase="post_commit_tests")
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 6
    # Undispatched rows (legacy / free replay / refusals) never count.
    _seed_paid_attempts(tmp_path, 1, start=7, paid=False)
    _seed_paid_attempts(tmp_path, 1, start=8, status="blocked", paid=False,
                        block_reason="identical_diff_refused", phase="preflight")
    assert count_paid_review_cycles(ctx, root_task_id="root-1") == 6


def test_commit_gate_has_no_import_time_cap_constant():
    import ouroboros.tools.commit_gate as commit_gate

    assert not hasattr(commit_gate, "BLOCKED_ATTEMPT_FINGERPRINT_CAP")
    # The pre-Q16 paid-identical-re-review API is retired: identical bytes are
    # refused free from the first verdict-block, not re-reviewed up to a cap.
    assert not hasattr(commit_gate, "check_blocked_attempt_cap")
    assert not hasattr(commit_gate, "blocked_attempt_fingerprint_cap")


# ---------------------------------------------------------------------------
# Settings POST boundary (mirrors the post-task evolution cadence rule)


def _settings_client(monkeypatch, tmp_path, current: dict):
    import server as srv
    import ouroboros.gateway.settings as gateway_settings

    monkeypatch.setattr(srv, "load_settings", lambda: dict(current))

    def fake_save_settings(settings, *args, **kwargs):
        current.clear()
        current.update(settings)

    monkeypatch.setattr(srv, "save_settings", fake_save_settings)
    monkeypatch.setattr(gateway_settings, "_owner_write_settings", fake_save_settings)
    monkeypatch.setattr(srv, "_apply_settings_to_env", lambda *_a, **_k: None)
    monkeypatch.setattr(srv, "_start_supervisor_if_needed", lambda *_a, **_k: False)
    monkeypatch.setattr(srv, "apply_runtime_provider_defaults", lambda s: (dict(s), False, []))
    monkeypatch.setattr(srv, "_mcp_reconfigure_startup", lambda *_a, **_k: None, raising=False)
    app = Starlette(routes=[Route("/api/settings", endpoint=srv.api_settings_post, methods=["POST"])])
    app.state.drive_root = tmp_path / "drive"
    app.state.repo_dir = tmp_path / "repo"
    return TestClient(app)


def test_settings_post_accepts_ints_and_unlimited_rejects_garbage(monkeypatch, tmp_path):
    current = dict(cfg.SETTINGS_DEFAULTS)
    client = _settings_client(monkeypatch, tmp_path, current)
    for good, stored in (("3", "3"), (5, "5"), ("unlimited", "unlimited"), ("INF", "unlimited"), ("∞", "unlimited")):
        resp = client.post("/api/settings", json={KEY: good})
        assert resp.status_code == 200, (good, resp.text)
        assert current[KEY] == stored
    current[KEY] = "2"
    for bad in ("0", "-2", "abc", "1.5", "every"):
        resp = client.post("/api/settings", json={KEY: bad})
        assert resp.status_code == 400, (bad, resp.text)
        assert "positive integer or 'unlimited'" in resp.json()["error"]
        assert current[KEY] == "2", bad  # not persisted


# ---------------------------------------------------------------------------
# UI + JS binding parity (owner D11: ONE knob, Behavior tab, 1 / 2 / 3 / 5 / ∞)


def test_settings_ui_knob_and_js_binding():
    ui = (REPO / "web" / "modules" / "settings_ui.js").read_text(encoding="utf-8")
    js = (REPO / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    css = (REPO / "web" / "settings.css").read_text(encoding="utf-8")
    assert '<input id="s-review-max-cycles" type="hidden" value="2">' in ui
    section = ui[ui.index("<h3>Max Review Cycles</h3>"):]
    section = section[:section.index("<h3>", 1)]  # the card alone, up to the next one
    assert "target: 's-review-max-cycles'" in section
    for value in ("'1'", "'2'", "'3'", "'5'", "'unlimited'"):
        assert f"value: {value}" in section
    assert "\\u221E" in section  # the ∞ label maps to the "unlimited" value
    assert "['s-review-max-cycles', 'OUROBOROS_REVIEW_MAX_CYCLES', '2']" in js
    assert "[data-review-cycles-group].settings-effort-group" in css
    # Behavior tab: the knob closes the review trio that opens the tab, before Access.
    assert ui.index("<h3>Review Enforcement</h3>") < ui.index("<h3>Task Result Review</h3>") < ui.index("<h3>Max Review Cycles</h3>") < ui.index("<h3>Access</h3>")


def test_legacy_acceptance_key_migrates_into_the_shared_knob(tmp_path, monkeypatch):
    """The deprecated acceptance key is a RENAME ALIAS, migrated at load like the retention
    keys — never a runtime branch that could not tell a deliberate "2" from an untouched
    default (production gate finding, 2026-08-16)."""
    import json

    from ouroboros import config as cfg
    from ouroboros.review_cycles import review_max_cycles

    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES": 3}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(settings))
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings, raising=False)
    monkeypatch.delenv("OUROBOROS_REVIEW_MAX_CYCLES", raising=False)
    monkeypatch.delenv("OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES", raising=False)
    loaded = cfg.load_settings()
    assert loaded["OUROBOROS_REVIEW_MAX_CYCLES"] == "4"  # cycles = passes + 1
    # the saved file itself is untouched by load (migration happens in the LOADED dict)
    cfg.apply_settings_to_env(loaded)
    assert review_max_cycles() == 4
    # An owner-authored shared value always wins: no "customized?" guessing left.
    settings.write_text(json.dumps({
        "OUROBOROS_ACCEPTANCE_MAX_IMPROVEMENT_PASSES": 3,
        "OUROBOROS_REVIEW_MAX_CYCLES": "2",
    }), encoding="utf-8")
    loaded = cfg.load_settings()
    cfg.apply_settings_to_env(loaded)
    assert review_max_cycles() == 2


def test_exhausted_event_is_durable_even_with_a_live_queue(tmp_path):
    """Review fix 4: the typed D27 escalation must ALWAYS land in events.jsonl —
    the live queue path persists only task_checkpoint rows, so queue-only emission
    silently lost the durable record; the queue additionally gets the UI push."""
    import queue as _queue

    events: _queue.Queue = _queue.Queue()
    rc.emit_review_cycles_exhausted(
        events, tmp_path, surface="plan_review", task_id="t1",
        cycles_paid=2, cap=2, enforcement="blocking", fingerprint="f" * 64)
    rows = [json.loads(line) for line in
            (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["type"] == "review_cycles_exhausted"
    assert rows[0]["task_id"] == "t1" and rows[0]["cap"] == 2
    pushed = events.get_nowait()
    assert pushed["type"] == "log_event"
    assert pushed["data"]["type"] == "review_cycles_exhausted"
    # No queue: the durable append still lands alone (unchanged path).
    rc.emit_review_cycles_exhausted(
        None, tmp_path, surface="plan_review", task_id="t2",
        cycles_paid=1, cap=1, enforcement="advisory")
    lines = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and events.empty()


def test_review_cycle_modules_respect_size_bounds():
    module_lines = (REPO / "ouroboros" / "review_cycles.py").read_text(encoding="utf-8").splitlines()
    # Brief-level target for this module (the repo ratchet is 1000/1600).
    assert len(module_lines) <= 200
    config_lines = (REPO / "ouroboros" / "config.py").read_text(encoding="utf-8").splitlines()
    assert len(config_lines) <= 1600

