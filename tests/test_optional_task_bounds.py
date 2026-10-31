"""Optional native task bounds (#1196): the total round limit and the absolute lifetime.

A fresh install (no settings document) runs with neither; a document an earlier release
wrote without them keeps the finite values it ran under, and reading never rewrites it;
a typo — blank, zero, negative, a word — is never "no bound"; explicit owner/evaluator
values bind exactly; and every operation that used to inherit the task lifetime keeps a
finite window of its own.
"""

from __future__ import annotations

import datetime as dt
import json
import threading
import types

import pytest

BOUNDS = ("OUROBOROS_MAX_ROUNDS", "OUROBOROS_TASK_ABS_CEILING_SEC")


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
    cfg.reset_runtime_mode_baseline_for_tests()
    yield settings_path
    cfg.reset_runtime_mode_baseline_for_tests()


def test_one_strict_vocabulary_shared_with_the_review_cycle_cap():
    from ouroboros import review_cycles
    from ouroboros.settings_scales import UNLIMITED, parse_positive_or_unlimited

    assert parse_positive_or_unlimited("7") == 7 and parse_positive_or_unlimited(12) == 12
    for alias in ("unlimited", " Unlimited ", "inf", "∞"):
        assert parse_positive_or_unlimited(alias) is None
    for bad in ("", None, "0", 0, "-3", "1.5", "none", "true", True):
        with pytest.raises((TypeError, ValueError)):
            parse_positive_or_unlimited(bad)
    assert review_cycles.UNLIMITED == UNLIMITED
    assert review_cycles.parse_review_max_cycles("unlimited") is None
    assert review_cycles.parse_review_max_cycles("3") == 3


def test_a_typo_is_the_finite_legacy_value_never_unlimited():
    from ouroboros import config as cfg
    from ouroboros.settings_scales import OPTIONAL_BOUND_LEGACY, optional_bound_value

    assert OPTIONAL_BOUND_LEGACY == {"OUROBOROS_MAX_ROUNDS": 200, "OUROBOROS_TASK_ABS_CEILING_SEC": 21600}
    for key in BOUNDS:
        legacy = OPTIONAL_BOUND_LEGACY[key]
        for bad in ("", None, 0, "0", -5, "abc", "1.5", True):
            assert optional_bound_value(key, bad) == legacy, (key, bad)
            assert cfg._coerce_setting_value(key, bad) == legacy, (key, bad)
        # The settings-document spelling is the parent's one read seam: "unlimited" or a positive int.
        assert cfg._coerce_setting_value(key, "UNLIMITED") == "unlimited"
        assert cfg._coerce_setting_value(key, 10800.0) == 10800, "a harness-written integral float"
        assert cfg._coerce_setting_value(key, "900") == 900


def test_fresh_install_is_unlimited_and_an_existing_document_keeps_its_bounds(isolated_settings):
    from ouroboros import config as cfg

    fresh = cfg.load_settings()
    assert all(fresh[key] == "unlimited" for key in BOUNDS)
    assert not isolated_settings.exists(), "a read created the document"

    # A document an earlier release wrote without the keys: the finite values it ran
    # under, and an already Safety-migrated read leaves its bytes alone.
    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 10.0,
        "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "full"}), encoding="utf-8")
    before = isolated_settings.read_bytes()
    legacy = cfg.load_settings()
    assert (legacy["OUROBOROS_MAX_ROUNDS"], legacy["OUROBOROS_TASK_ABS_CEILING_SEC"]) == (200, 21600)
    assert isolated_settings.read_bytes() == before, "a read rewrote the document"

    # Saved positive values are preserved exactly; an explicit unlimited is honoured.
    isolated_settings.write_text(json.dumps({
        "OUROBOROS_MAX_ROUNDS": 42, "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited",
    }), encoding="utf-8")
    loaded = cfg.load_settings()
    assert loaded["OUROBOROS_MAX_ROUNDS"] == 42
    assert loaded["OUROBOROS_TASK_ABS_CEILING_SEC"] == "unlimited"

    # An unreadable document is still an existing one: finite, never unlimited.
    isolated_settings.write_text("{not json", encoding="utf-8")
    broken = cfg.load_settings()
    assert (broken["OUROBOROS_MAX_ROUNDS"], broken["OUROBOROS_TASK_ABS_CEILING_SEC"]) == (200, 21600)


def test_a_fresh_install_save_persists_unlimited_so_the_document_keeps_meaning_it(isolated_settings):
    from ouroboros import config as cfg

    cfg.save_settings(cfg.load_settings())
    stored = json.loads(isolated_settings.read_text(encoding="utf-8"))
    assert all(stored[key] == "unlimited" for key in BOUNDS)
    assert all(cfg.load_settings()[key] == "unlimited" for key in BOUNDS)


def test_environment_precedence_is_unchanged_for_a_document_without_the_key(isolated_settings, monkeypatch):
    from ouroboros import config as cfg

    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "")
    assert cfg.load_settings()["OUROBOROS_MAX_ROUNDS"] == 200, "explicit empty on a fresh install is invalid, not unlimited"
    assert not isolated_settings.exists()
    isolated_settings.write_text(json.dumps({"TOTAL_BUDGET": 10.0}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "77")
    assert cfg.load_settings()["OUROBOROS_MAX_ROUNDS"] == 77, "a forwarded value still applies"
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "")
    assert cfg.load_settings()["OUROBOROS_MAX_ROUNDS"] == 200, "an empty variable is not unlimited"


def test_runtime_readers_absent_empty_malformed_and_floors(monkeypatch):
    from ouroboros.config import get_max_rounds, get_task_abs_ceiling_sec

    for key in BOUNDS:
        monkeypatch.delenv(key, raising=False)
    assert get_max_rounds() is None and get_task_abs_ceiling_sec() is None
    for key in BOUNDS:
        monkeypatch.setenv(key, "")
    assert (get_max_rounds(), get_task_abs_ceiling_sec()) == (200, 21600)
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "garbage")
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "-1")
    assert (get_max_rounds(), get_task_abs_ceiling_sec()) == (200, 21600)
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "12")
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "60")
    assert (get_max_rounds(), get_task_abs_ceiling_sec()) == (12, 300), "the 300 s floor stays"
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "unlimited")
    assert get_task_abs_ceiling_sec() is None


def test_operation_window_is_finite_and_every_deadline_still_narrows_it(monkeypatch):
    from ouroboros.config import OPERATION_WINDOW_FALLBACK_SEC, operation_window_sec
    from ouroboros.deadline_utils import review_operation_timeout_sec

    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    assert operation_window_sec(None) == float(OPERATION_WINDOW_FALLBACK_SEC) == 21600.0
    assert operation_window_sec(3600) == 3600.0
    # An agent session with no lifetime bound gets the finite window, never "no window".
    assert review_operation_timeout_sec(10_000_000, route="agent_session") == 21600.0
    soon = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=600)).isoformat()
    narrowed = review_operation_timeout_sec(10_000_000, route="agent_session", deadline_at=soon)
    assert 0 < narrowed <= 600
    monkeypatch.setenv("OUROBOROS_TASK_ABS_CEILING_SEC", "1800")
    assert review_operation_timeout_sec(10_000_000, route="agent_session") == 1800.0


def test_round_limit_resolution_keeps_the_presence_inline_cap(monkeypatch):
    from ouroboros.loop import _resolve_loop_max_rounds

    monkeypatch.delenv("OUROBOROS_MAX_ROUNDS", raising=False)
    assert _resolve_loop_max_rounds() is None
    assert _resolve_loop_max_rounds(types.SimpleNamespace(inline_max_rounds=10)) == 10
    monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "6")
    assert _resolve_loop_max_rounds(types.SimpleNamespace(inline_max_rounds=10)) == 6
    assert _resolve_loop_max_rounds(types.SimpleNamespace()) == 6


def test_presence_inline_cap_stays_finite_without_a_global_limit():
    from tests.frozen_presence_runtime import (
        PresenceRuntimeDefaults, PresenceRuntimeError, PresenceRuntimeOverrides, resolve_presence_runtime,
    )

    free = resolve_presence_runtime(
        PresenceRuntimeDefaults("main", 10), PresenceRuntimeOverrides(inline_max_rounds=14),
        global_max_rounds=None)
    assert (free.inline_max_rounds, free.capped) == (14, False)
    assert resolve_presence_runtime(None, None, global_max_rounds=None).inline_max_rounds == 10
    with pytest.raises(PresenceRuntimeError):  # a malformed global cap still fails closed
        resolve_presence_runtime(None, None, global_max_rounds=0)


def test_facts_line_round_without_a_round_limit_invents_no_remainder(monkeypatch):
    """The 15-round self-check is gone (#1748); the facts line names the round and the
    limit that binds first, or that there is none — never an invented remainder."""
    from types import SimpleNamespace

    from ouroboros import loop
    from ouroboros.loop_messages import _round_fact

    ctx, tool_ctx = SimpleNamespace(round_idx=15), SimpleNamespace()
    monkeypatch.setattr(loop, "_resolve_loop_max_rounds", lambda _ctx=None: None)
    assert _round_fact(ctx, tool_ctx) == "round 15 (no round limit)"
    monkeypatch.setattr(loop, "_resolve_loop_max_rounds", lambda _ctx=None: 200)
    assert _round_fact(ctx, tool_ctx) == "round 15 of 200 max"
    assert not hasattr(loop, "_maybe_inject_self_check")


def test_model_wait_has_no_execution_window_or_ceiling_stop_without_a_lifetime(tmp_path, monkeypatch):
    from ouroboros import config, model_wait

    monkeypatch.setattr("ouroboros.cancel_intents.cancel_pending", lambda *_a, **_k: False)
    monkeypatch.setattr("ouroboros.owner_mailbox.drain_owner_entries", lambda *_a, **_k: [])
    owner = model_wait.TaskModelWait(task={"id": "unbounded"}, drive_root=tmp_path,
                                     event_queue=None, worker_slot_held=False)
    owner.started_monotonic = 0  # "started" a whole monotonic epoch ago
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: None)
    assert owner.execution_window_remaining() is None
    assert owner.control_reason() is None
    monkeypatch.setattr(config, "get_task_abs_ceiling_sec", lambda: 300)
    assert owner.execution_window_remaining() == 0.0, "a spent window is 0.0, not None"
    assert owner.control_reason() == "absolute_ceiling"


def test_owner_stop_hard_bound_skips_an_absent_lifetime_and_keeps_the_deadline():
    from supervisor.owner_stop import _task_hard_bound_reached

    def queue(ceiling, task):
        return types.SimpleNamespace(
            _queue_lock=threading.Lock(), RUNNING={"t": {"task": task, "started_at": 1.0}},
            _task_deadline_ts=lambda item: float(item.get("deadline_ts") or 0.0),
            get_task_abs_ceiling_sec=lambda: ceiling)

    assert _task_hard_bound_reached(queue(None, {}), "t", now=1e9) is False
    assert _task_hard_bound_reached(queue(300, {}), "t", now=1e9) is True
    assert _task_hard_bound_reached(queue(None, {"deadline_ts": 5.0}), "t", now=10.0) is True


def test_quiz_wait_bound_is_capped_only_by_a_finite_lifetime(monkeypatch):
    from ouroboros.tools.core_artifacts import QuizValidationError, _validate_wait_bound

    monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda: None)
    assert _validate_wait_bound(100_000, wait_for_answer=True) == 100_000
    monkeypatch.setattr("ouroboros.config.get_task_abs_ceiling_sec", lambda: 3600)
    assert _validate_wait_bound(60, wait_for_answer=True) == 60
    with pytest.raises(QuizValidationError, match="at most 60"):
        _validate_wait_bound(61, wait_for_answer=True)


def _request():
    from starlette.requests import Request

    return Request({"type": "http", "method": "POST", "path": "/api/settings",
                    "headers": [], "query_string": b""})


def test_settings_save_refuses_blank_zero_and_malformed_bounds_before_persistence():
    from ouroboros.gateway.settings import _api_settings_post_locked

    for key in BOUNDS:
        for bad in ("", None, "0", 0, "-1", "1.5", "none", "soon", True):
            response = _api_settings_post_locked(_request(), {key: bad})
            body = json.loads(response.body)
            assert response.status_code == 400 and body["saved"] is False, (key, bad)
            assert "positive integer or 'unlimited'" in body["error"]


def test_settings_save_persists_ints_and_the_canonical_unlimited(monkeypatch):
    import asyncio

    from starlette.requests import Request

    import ouroboros.gateway.settings as gws
    from ouroboros.config import SETTINGS_DEFAULTS

    saved: dict = {}

    def _fake_write(payload, *, allow_elevation=False, allow_context_lowering=False,
                    authored_keys=(), boundary=None):
        saved.clear()
        saved.update(payload)
        if boundary is not None:
            boundary.commit()
        return payload

    monkeypatch.setattr(gws, "load_settings", lambda: {**SETTINGS_DEFAULTS, **saved})
    monkeypatch.setattr(gws, "_owner_write_settings", _fake_write)
    monkeypatch.setattr(gws, "_unrecognised_review_models", lambda models: [])
    monkeypatch.setattr(gws, "_apply_settings_to_env", lambda *a, **k: None)

    async def _receive():
        payload = {"OUROBOROS_MAX_ROUNDS": "250", "OUROBOROS_TASK_ABS_CEILING_SEC": "INF"}
        return {"type": "http.request", "body": json.dumps(payload).encode()}

    request = Request({"type": "http", "method": "POST", "path": "/api/settings",
                       "headers": [("content-type", "application/json")],
                       "query_string": b"", "app": None}, receive=_receive)
    response = asyncio.run(gws.api_settings_post(request))
    assert response.status_code == 200, response.body
    assert saved["OUROBOROS_MAX_ROUNDS"] == 250
    assert saved["OUROBOROS_TASK_ABS_CEILING_SEC"] == "unlimited"
