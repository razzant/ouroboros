"""Phase 1 (cross-task self-evolution): budget refusal guard, V4 config envelope,
and V5 promotion (durable-signal -> gated supervisor apply)."""
from __future__ import annotations

import json
import pathlib
import types


import ouroboros.config as config
import ouroboros.post_task_evolution as pte
import supervisor.state as state


# --- Budget refusal guard (red-team R2.1 / BIBLE P8) --------------------------

def _seed_state(d: pathlib.Path, spent: float = 7.5) -> None:
    (d / "state").mkdir(parents=True, exist_ok=True)
    (d / "state" / "state.json").write_text(
        json.dumps({"spent_usd": spent, "keep": "me"}), encoding="utf-8")
    # Mark the throwaway root as an isolated benchmark data root (reset guard requires it).
    (d / state.ISOLATED_BENCHMARK_SENTINEL).write_text("isolated\n", encoding="utf-8")


def test_budget_reset_refuses_without_isolated_sentinel(tmp_path, monkeypatch):
    """Even with confirm_isolated + matching OUROBOROS_DATA_DIR + a non-home target, reset
    REFUSES unless the isolated-benchmark sentinel is present — this is what protects a
    custom/Drive-backed live data root (which would not match the ~/Ouroboros/data check)."""
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "state.json").write_text(json.dumps({"spent_usd": 5.0}), encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert state.reset_per_task_budget(tmp_path, confirm_isolated=True) is False  # no sentinel
    (tmp_path / state.ISOLATED_BENCHMARK_SENTINEL).write_text("isolated\n", encoding="utf-8")
    assert state.reset_per_task_budget(tmp_path, confirm_isolated=True) is True   # sentinel present


def test_budget_reset_refuses_live_dir(monkeypatch):
    live = (pathlib.Path.home() / "Ouroboros" / "data")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(live))
    assert state.reset_per_task_budget(live, confirm_isolated=True) is False


def test_budget_reset_refuses_without_confirm(tmp_path, monkeypatch):
    _seed_state(tmp_path)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert state.reset_per_task_budget(tmp_path, confirm_isolated=False) is False


def test_budget_reset_refuses_without_env(tmp_path, monkeypatch):
    _seed_state(tmp_path)
    monkeypatch.delenv("OUROBOROS_DATA_DIR", raising=False)
    assert state.reset_per_task_budget(tmp_path, confirm_isolated=True) is False


def test_budget_reset_allows_isolated(tmp_path, monkeypatch):
    _seed_state(tmp_path, spent=9.9)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    assert state.reset_per_task_budget(tmp_path, confirm_isolated=True) is True
    after = json.loads((tmp_path / "state" / "state.json").read_text())
    assert after["spent_usd"] == 0.0
    assert after["keep"] == "me"  # non-budget state preserved


# --- V4 config envelope -------------------------------------------------------

def test_envelope_defaults_off(monkeypatch):
    monkeypatch.delenv("OUROBOROS_POST_TASK_EVOLUTION", raising=False)
    monkeypatch.delenv("OUROBOROS_POST_TASK_EVOLUTION_CADENCE", raising=False)
    assert config.get_post_task_evolution_enabled() is False
    assert config.get_post_task_evolution_cadence() == "llm"
    assert config.get_post_task_evolution_budget_usd() == 0.0


def test_envelope_enable_parsing(monkeypatch):
    for v in ("true", "1", "yes", "on"):
        monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", v)
        assert config.get_post_task_evolution_enabled() is True
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "false")
    assert config.get_post_task_evolution_enabled() is False


# --- V5 guards ----------------------------------------------------------------

def test_v5_eligibility_and_canonical():
    assert pte._eligible({"type": "task"}) is True
    assert pte._eligible({"type": "evolution"}) is False
    assert pte._eligible({"type": "deep_self_review"}) is False
    assert pte._eligible({"type": "task", "delegation_role": "subagent"}) is False
    env = types.SimpleNamespace(drive_root=pathlib.Path("/x/data"))
    assert pte._is_canonical_run(env, {}) is True
    assert pte._is_canonical_run(env, {"budget_drive_root": "/y/data"}) is False
    assert pte._is_canonical_run(env, {"budget_drive_root": "/x/data"}) is True


def test_v5_every_n_counter(tmp_path):
    # k=2 -> due on the 2nd, 4th call
    assert pte._counter_due(tmp_path, 2) is False
    assert pte._counter_due(tmp_path, 2) is True
    assert pte._counter_due(tmp_path, 2) is False
    assert pte._counter_due(tmp_path, 2) is True


def test_v5_maybe_promote_off_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("OUROBOROS_POST_TASK_EVOLUTION", raising=False)
    env = types.SimpleNamespace(drive_root=tmp_path)
    assert pte.maybe_promote(env, {"type": "task", "id": "t1"}, {"reflection": "x"}) is None


def test_v5_maybe_promote_light_mode_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    env = types.SimpleNamespace(drive_root=tmp_path)
    assert pte.maybe_promote(env, {"type": "task", "id": "t1"}, {"reflection": "x"}) is None


def test_v5_apply_pending_none_when_no_request(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    assert pte.apply_pending_request(tmp_path) is False


def test_v5_apply_pending_request_activates_gated_campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    # request file with requires_plan_review -> objective must carry the obligation
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(json.dumps({
        "objective": "Refactor X for clarity", "requires_plan_review": True,
        "backlog_id": "abc", "source": "post_task",
    }), encoding="utf-8")

    started = {}
    saved = {}

    import supervisor.evolution_lifecycle as lifecycle
    import supervisor.state as st
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(lifecycle, "start_evolution_campaign", lambda objective, source="": started.update(objective=objective, source=source))
    monkeypatch.setattr(st, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(st, "save_state", lambda s: saved.update(s))

    def _fake_update_state(mutator):
        live = {"owner_chat_id": 7}
        mutator(live)
        saved.update(live)
        return live

    monkeypatch.setattr(st, "update_state", _fake_update_state)

    assert pte.apply_pending_request(tmp_path) is True
    assert "plan_task" in started["objective"]  # requires_plan_review carried in
    assert started["source"] == "post_task"
    assert saved["evolution_mode_enabled"] is True
    assert saved["post_task_autostop"] is True
    # one-shot: request consumed
    assert not (tmp_path / "state" / "post_task_evolution_request.json").exists()


def test_evolution_owner_stopped_blocks_post_task_rearm(tmp_path, monkeypatch):
    """v6.52.x owner-stop-persistence fix (CORE regression, inverse of the test above):
    with the durable ``evolution_owner_stopped`` flag set, a queued post-task promotion is
    DROPPED — never used to silently re-arm evolution (no /evolve start). Also covers the
    maybe_promote race where a worker re-wrote the request after the owner-stop cleared it."""
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(json.dumps({
        "objective": "Refactor X for clarity", "requires_plan_review": False, "source": "post_task",
    }), encoding="utf-8")

    started = {}
    saved = {}
    import supervisor.evolution_lifecycle as lifecycle
    import supervisor.state as st
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(lifecycle, "start_evolution_campaign",
                        lambda objective, source="": started.update(objective=objective, source=source))
    # Owner EXPLICITLY stopped: flag set, evolution disabled, owner present.
    monkeypatch.setattr(st, "load_state",
                        lambda: {"owner_chat_id": 7, "evolution_owner_stopped": True, "evolution_mode_enabled": False})

    def _fake_update_state(mutator):
        live = {"owner_chat_id": 7, "evolution_owner_stopped": True}
        mutator(live)
        saved.update(live)
        return live
    monkeypatch.setattr(st, "update_state", _fake_update_state)

    assert pte.apply_pending_request(tmp_path) is False
    assert started == {}, "start_evolution_campaign must NOT run while owner-stopped"
    assert "evolution_mode_enabled" not in saved, "no autonomous re-enable while owner-stopped"
    # the (raced) request is dropped so it cannot re-fire on a later idle tick
    assert not (tmp_path / "state" / "post_task_evolution_request.json").exists()


def test_post_task_apply_still_defers_when_already_enabled(tmp_path, monkeypatch):
    """The pre-existing 'already enabled' guard is NOT stolen by the new owner-stop guard:
    with evolution running (enabled True, NO owner-stop flag) apply DEFERS (returns False)
    and LEAVES the request for the running campaign — the legitimate absorb-restart-verify
    resume relies on enabled staying True and is untouched by this fix."""
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(json.dumps({
        "objective": "X", "requires_plan_review": False,
    }), encoding="utf-8")
    import supervisor.evolution_lifecycle as lifecycle
    import supervisor.state as st
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(st, "load_state", lambda: {"owner_chat_id": 7, "evolution_mode_enabled": True})

    assert pte.apply_pending_request(tmp_path) is False
    # deferred (enabled-guard), NOT dropped — distinct from the owner-stop drop above
    assert (tmp_path / "state" / "post_task_evolution_request.json").exists()


def test_complete_evolution_campaign_is_terminal_and_archives_tx(tmp_path, monkeypatch):
    """complete_evolution_campaign terminally closes (status NOT in {active,paused}),
    archives+pops any in-flight active_transaction, and pops post_task_backlog_id — so a
    stopped campaign carries no dangling commit for a boot reconcile to absorb."""
    import supervisor.queue as queue
    import supervisor.evolution_lifecycle as lifecycle
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    lifecycle._write_evolution_campaign({
        "id": "OLD", "status": "active",
        "active_transaction": {"task_id": "t1", "commit_sha": "abc"},
        "post_task_backlog_id": "b1",
    })
    c = lifecycle.complete_evolution_campaign("disabled via owner chat", status="stopped")
    assert c["status"] == "stopped"
    assert c["status"] not in {"active", "paused"}
    assert c.get("completion_reason") == "disabled via owner chat"
    assert "active_transaction" not in c          # popped: no dangling tx for a boot reconcile
    assert "post_task_backlog_id" not in c
    assert c.get("transaction_history")            # archived, not lost


def test_evolve_start_after_stop_mints_fresh_campaign(tmp_path, monkeypatch):
    """A terminal 'stopped' campaign is NOT resurrected: /evolve start mints a FRESH
    campaign (new id, active). Confirms owner re-engagement + that the fresh-mint branch
    already covers terminal statuses (proposal (c) needs no code change)."""
    import supervisor.queue as queue
    import supervisor.evolution_lifecycle as lifecycle
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    lifecycle._write_evolution_campaign({"id": "OLD", "status": "stopped"})
    fresh = lifecycle.start_evolution_campaign("new objective", source="owner_chat")
    assert fresh["id"] != "OLD"
    assert fresh["status"] == "active"
    assert fresh["objective"] == "new objective"
    assert fresh.get("cycles_done") == 0


def test_toggle_evolution_off_wires_owner_stop(tmp_path, monkeypatch):
    """Owner-stop SITE wiring (closes the review gap that the downstream apply_pending_request
    tests could not): _handle_toggle_evolution(enabled=False) must set the durable
    evolution_owner_stopped flag, clear post_task_autostop, terminally close the campaign
    (complete_evolution_campaign — NOT the resumable pause), and drop the queued promotion
    request. A regression dropping any of these reintroduces the autonomous re-arm bug while
    the flag-preseeded apply_pending_request tests still pass."""
    from supervisor.events import _handle_toggle_evolution
    import supervisor.evolution_lifecycle as lifecycle
    import supervisor.queue as queue

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text("{}", encoding="utf-8")

    captured = {}
    calls = {"complete": [], "start": []}

    def _fake_update_state(mutator):
        live = {"owner_chat_id": 7}
        mutator(live)
        captured.update(live)
        return live

    monkeypatch.setattr(state, "update_state", _fake_update_state)
    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)  # real drop_pending_request target
    monkeypatch.setattr(lifecycle, "complete_evolution_campaign",
                        lambda reason="", *, status="stopped": calls["complete"].append((reason, status)))
    monkeypatch.setattr(lifecycle, "start_evolution_campaign",
                        lambda *a, **k: calls["start"].append((a, k)))
    monkeypatch.setattr(queue, "cancel_running_evolution_tasks", lambda *a, **k: [])

    ctx = types.SimpleNamespace(
        PENDING=[{"type": "evolution"}, {"type": "task"}],
        sort_pending=lambda: None,
        persist_queue_snapshot=lambda reason="": None,
        send_with_budget=lambda cid, text: None,
        load_state=lambda: {"owner_chat_id": 7},
    )
    _handle_toggle_evolution({"enabled": False}, ctx)

    assert captured["evolution_mode_enabled"] is False
    assert captured["evolution_owner_stopped"] is True   # durable owner-stop sentinel SET
    assert captured["post_task_autostop"] is False       # one-shot autostop cleared
    assert calls["complete"] == [("disabled via agent tool", "stopped")]  # terminal, not pause
    assert calls["start"] == []
    assert not (tmp_path / "state" / "post_task_evolution_request.json").exists()  # request dropped
    assert ctx.PENDING == [{"type": "task"}]              # evolution task pruned from the queue


def test_toggle_evolution_on_clears_owner_stop(tmp_path, monkeypatch):
    """The owner /evolve-start counterpart: _handle_toggle_evolution(enabled=True) CLEARS
    evolution_owner_stopped (the only owner-authorized clear) and mints a fresh campaign — so
    re-engaging after a stop works and the durable flag never sticks True."""
    from supervisor.events import _handle_toggle_evolution
    import supervisor.evolution_lifecycle as lifecycle

    captured = {}
    calls = {"complete": [], "start": []}

    def _fake_update_state(mutator):
        live = {"owner_chat_id": 7}
        mutator(live)
        captured.update(live)
        return live

    monkeypatch.setattr(state, "update_state", _fake_update_state)
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")  # not light mode
    monkeypatch.setattr(lifecycle, "complete_evolution_campaign",
                        lambda reason="", *, status="stopped": calls["complete"].append((reason, status)))
    monkeypatch.setattr(lifecycle, "start_evolution_campaign",
                        lambda objective="", *, source="": calls["start"].append((objective, source)))

    ctx = types.SimpleNamespace(
        load_state=lambda: {"owner_chat_id": 7},
        send_with_budget=lambda cid, text: None,
    )
    _handle_toggle_evolution({"enabled": True, "objective": "improve X"}, ctx)

    assert captured["evolution_mode_enabled"] is True
    assert captured["evolution_owner_stopped"] is False  # cleared on owner start
    assert calls["start"] == [("improve X", "agent_tool")]
    assert calls["complete"] == []


def test_apply_pending_request_atomic_recheck_aborts_on_raced_owner_stop(tmp_path, monkeypatch):
    """Atomic re-check: if an owner stop sets evolution_owner_stopped AFTER the load_state()
    snapshot passes the chokepoint but BEFORE the enabling update_state (the panic-from-
    another-thread window), _activate_one_shot honors the LIVE flag and refuses to enable.
    The stale campaign the pre-check path minted is terminal-closed and the request dropped;
    apply returns False. Without the re-check this path would flip evolution_mode_enabled True
    against a fresh owner stop."""
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(json.dumps({
        "objective": "Refactor X", "requires_plan_review": False, "source": "post_task",
    }), encoding="utf-8")

    import supervisor.evolution_lifecycle as lifecycle
    calls = {"start": [], "complete": []}
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(lifecycle, "start_evolution_campaign",
                        lambda objective, source="": calls["start"].append((objective, source)))
    monkeypatch.setattr(lifecycle, "complete_evolution_campaign",
                        lambda reason="", *, status="stopped": calls["complete"].append((reason, status)))
    # Snapshot passes the chokepoint (flag False, not yet enabled)...
    monkeypatch.setattr(state, "load_state",
                        lambda: {"owner_chat_id": 7, "evolution_owner_stopped": False, "evolution_mode_enabled": False})
    saved = {}

    def _fake_update_state(mutator):
        # ...but by the time of the atomic update the owner stop has landed (raced True).
        live = {"owner_chat_id": 7, "evolution_owner_stopped": True}
        mutator(live)
        saved.update(live)
        return live

    monkeypatch.setattr(state, "update_state", _fake_update_state)

    assert pte.apply_pending_request(tmp_path) is False
    assert calls["start"], "the pre-check path still minted a campaign before the race was seen"
    assert saved.get("evolution_mode_enabled") is not True, "atomic re-check must refuse the enable"
    assert calls["complete"] == [("owner stop raced post-task enable", "stopped")]  # stale campaign closed
    assert not (tmp_path / "state" / "post_task_evolution_request.json").exists()  # request dropped


def test_execute_panic_stop_wires_owner_stop(tmp_path, monkeypatch):
    """Panic is an owner stop: execute_panic_stop sets evolution_owner_stopped + clears
    post_task_autostop in state, terminal-closes the campaign, and drops the queued request
    BEFORE the hard exit. Every destructive teardown op (process/port kills, os._exit) is
    neutralized so the test is safe. Closes the stop-site wiring gap for the panic path."""
    import logging
    import os as _os
    import ouroboros.server_control as sc
    import supervisor.evolution_lifecycle as lifecycle
    import ouroboros.post_task_evolution as pte_mod
    import ouroboros.platform_layer as platform_layer

    saved = {}
    calls = {"complete": [], "drop": []}
    monkeypatch.setattr(state, "load_state", lambda: {"evolution_mode_enabled": True, "post_task_autostop": True})
    monkeypatch.setattr(state, "save_state", lambda s: saved.update(s))
    monkeypatch.setattr(lifecycle, "complete_evolution_campaign",
                        lambda reason="", *, status="stopped", cleanup_worktree=True:
                        calls["complete"].append((reason, status, cleanup_worktree)))
    monkeypatch.setattr(pte_mod, "drop_pending_request", lambda dr: calls["drop"].append(dr))
    # Neutralize destructive teardown: real process/port kills and the hard exit must not run.
    monkeypatch.setattr(platform_layer, "kill_process_on_port", lambda *a, **k: None)
    monkeypatch.setattr(platform_layer, "force_kill_pid", lambda *a, **k: None)
    import ouroboros.tools.shell as _shell
    import ouroboros.local_model as _lm
    monkeypatch.setattr(_shell, "kill_all_tracked_subprocesses", lambda *a, **k: None)
    monkeypatch.setattr(_lm, "get_manager", lambda: types.SimpleNamespace(stop_server=lambda: None))

    class _StopPanic(Exception):
        pass

    def _no_exit(code):
        raise _StopPanic(code)

    monkeypatch.setattr(_os, "_exit", _no_exit)

    import pytest
    consciousness = types.SimpleNamespace(stop=lambda: None)
    with pytest.raises(_StopPanic):
        sc.execute_panic_stop(
            consciousness, lambda **k: None,
            data_dir=tmp_path, panic_exit_code=99, log=logging.getLogger("panic-test"),
        )

    assert saved.get("evolution_owner_stopped") is True    # durable owner-stop sentinel SET
    assert saved.get("post_task_autostop") is False         # one-shot autostop cleared
    assert saved.get("evolution_mode_enabled") is False
    # terminal-close + Emergency Stop Invariant: panic must NOT run the mid-cycle git
    # worktree cleanup (cleanup_worktree=False) so nothing delays the panic hard-exit.
    assert calls["complete"] == [("panic stop", "stopped", False)]
    assert calls["drop"] == [tmp_path]                       # queued promotion dropped


def test_complete_evolution_campaign_cleans_worktree_before_popping_tx(tmp_path, monkeypatch):
    """Owner stop mid-cycle: complete_evolution_campaign runs the per-cycle worktree cleanup
    on the in-flight active_transaction BEFORE popping it, so abandoned/unreviewed evolution
    edits are reset instead of left dirty in the live repo. Without this, the pause->complete
    change would skip cleanup — the terminal 'stopped' status makes the cancelled task's
    (async) task_done early-return in update_evolution_campaign_after_task."""
    import supervisor.queue as queue
    import supervisor.evolution_lifecycle as lifecycle
    monkeypatch.setattr(queue, "DRIVE_ROOT", str(tmp_path))
    cleaned = []
    monkeypatch.setattr(lifecycle, "_cleanup_worktree_after_cycle",
                        lambda tx, task_id: cleaned.append((dict(tx), task_id)))
    lifecycle._write_evolution_campaign({
        "id": "OLD", "status": "active",
        "active_transaction": {"task_id": "t9", "commit_sha": "", "base_head": "abc123"},
    })
    c = lifecycle.complete_evolution_campaign("disabled via owner chat", status="stopped")
    assert c["status"] == "stopped"
    assert "active_transaction" not in c            # tx popped AFTER cleanup
    assert len(cleaned) == 1                          # cleanup ran exactly once...
    assert cleaned[0][1] == "t9"                      # ...for the in-flight tx's task
    assert cleaned[0][0].get("task_id") == "t9"       # cleanup saw the tx before it was popped

    # Emergency Stop Invariant: cleanup_worktree=False (panic path) still pops the tx but
    # runs NO git worktree cleanup, so nothing can delay the panic hard-exit.
    lifecycle._write_evolution_campaign({
        "id": "OLD2", "status": "active",
        "active_transaction": {"task_id": "t10", "commit_sha": "", "base_head": "def456"},
    })
    c2 = lifecycle.complete_evolution_campaign("panic stop", status="stopped", cleanup_worktree=False)
    assert c2["status"] == "stopped"
    assert "active_transaction" not in c2
    assert len(cleaned) == 1                          # unchanged: no cleanup under panic


def test_budget_reset_refuses_target_mismatch(tmp_path, monkeypatch):
    _seed_state(tmp_path)
    other = tmp_path.parent / (tmp_path.name + "_other")
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(other))  # env data dir != reset target
    assert state.reset_per_task_budget(tmp_path, confirm_isolated=True) is False


def test_v5_apply_pending_refused_when_budget_floor_unmet(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD", "5.0")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(
        json.dumps({"objective": "x"}), encoding="utf-8")
    import supervisor.evolution_lifecycle as lifecycle
    import supervisor.state as st
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(st, "load_state", lambda: {"owner_chat_id": 7, "spent_usd": 9.0})
    monkeypatch.setattr(st, "budget_remaining", lambda s: 1.0)  # below the 5.0 floor
    assert pte.apply_pending_request(tmp_path) is False


def test_envelope_enable_rides_generic_settings_merge():
    """The post-task self-evolution enable rides the generic owner settings path
    (like ALLOW_MUTATIVE_SUBAGENTS) so the Settings UI On/Off toggle persists. The
    AGENT still cannot self-enable it: shell (_detect_evolution_owner_control_self_change),
    browser JS (_blocks_post_task_evolution_js), POST /api/settings route guard, and
    data_write (DATA_WRITE_BLOCKED) all block agent-originated changes (see test_acting_subagents)."""
    from ouroboros.gateway.settings import _merge_settings_payload

    merged = _merge_settings_payload(
        {"OUROBOROS_POST_TASK_EVOLUTION": "false"},
        {"OUROBOROS_POST_TASK_EVOLUTION": "true"},
    )
    assert merged["OUROBOROS_POST_TASK_EVOLUTION"] == "true"
    # Genuinely owner-endpoint-only keys stay merge-skipped.
    skipped = _merge_settings_payload(
        {"OUROBOROS_RUNTIME_MODE": "advanced"},
        {"OUROBOROS_RUNTIME_MODE": "pro"},
    )
    assert skipped["OUROBOROS_RUNTIME_MODE"] == "advanced"


def test_cadence_normalization_is_strict(monkeypatch):
    """Only off | llm | every_n:<k>=1> are valid; everything else -> llm so a
    malformed value can never force an evolution cycle after every task."""
    from ouroboros.config import get_post_task_evolution_cadence

    for bad in ("every_nonsense", "every_n:", "every_n:0", "every_n:-1", "every:5", "garbage", ""):
        monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION_CADENCE", bad)
        assert get_post_task_evolution_cadence() == "llm", bad
    for good in ("off", "llm", "every_n:1", "every_n:5", "EVERY_N:3"):
        monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION_CADENCE", good)
        assert get_post_task_evolution_cadence() == good.lower(), good


def test_persistent_objective_steers_active_evolution_campaign(monkeypatch):
    """The owner persistent-objective steer is appended (additively) to an ACTIVE
    evolution campaign's task text and the getter round-trips; empty = pure LLM choice."""
    from ouroboros import config
    from supervisor import evolution_lifecycle as lifecycle

    monkeypatch.delenv("OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE", raising=False)
    assert config.get_evolution_persistent_objective() == ""  # default no-op

    monkeypatch.setenv("OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE", "prioritize test coverage")
    assert config.get_evolution_persistent_objective() == "prioritize test coverage"
    monkeypatch.setattr(lifecycle, "_read_evolution_campaign",
                        lambda: {"status": "active", "objective": "Improve X"})
    text = lifecycle.build_evolution_task_text(1)
    assert "prioritize test coverage" in text  # steer appended
    assert "Improve X" in text                  # campaign objective preserved (not overridden)


def test_apply_pending_keeps_unparseable_request(tmp_path, monkeypatch):
    """A partial/corrupt request must not be dropped (avoids the write/read race)."""
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    p = tmp_path / "state" / "post_task_evolution_request.json"
    p.write_text("{partial-json", encoding="utf-8")
    assert pte.apply_pending_request(tmp_path) is False
    assert p.exists()  # retained for the next tick, not unlinked


def _apply_with_request(tmp_path, monkeypatch, backlog_id):
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(json.dumps({
        "objective": "obj", "backlog_id": backlog_id, "requires_plan_review": False,
    }), encoding="utf-8")
    import supervisor.evolution_lifecycle as lifecycle
    import supervisor.state as stt
    camp: dict = {}
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(lifecycle, "start_evolution_campaign", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_read_evolution_campaign", lambda: camp)
    monkeypatch.setattr(lifecycle, "_write_evolution_campaign", lambda c: camp.update(c))
    monkeypatch.setattr(stt, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(stt, "save_state", lambda s: None)
    ok = pte.apply_pending_request(tmp_path)
    return ok, camp


def test_v5_apply_pending_stores_valid_backlog_link(tmp_path, monkeypatch):
    # Exercises the campaign post_task_backlog_id link path (close-on-absorb relies on it).
    from ouroboros.improvement_backlog import append_backlog_items

    append_backlog_items(tmp_path, [{
        "summary": "the promoted fix", "category": "c", "source": "s",
        "evidence": "e", "fingerprint": "fp-7", "id": "ibl-7",
    }])
    ok, camp = _apply_with_request(tmp_path, monkeypatch, "ibl-7")
    assert ok is True
    assert camp.get("post_task_backlog_id") == "ibl-7"


def test_v5_apply_pending_rejects_unknown_backlog_id(tmp_path, monkeypatch):
    # A hallucinated/stale id must NOT be linked (it could later close an unrelated item).
    ok, camp = _apply_with_request(tmp_path, monkeypatch, "ibl-does-not-exist")
    assert ok is True  # the objective still applies
    assert "post_task_backlog_id" not in camp  # but no bogus link is stored


def test_v5_apply_pending_blocked_in_light_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_POST_TASK_EVOLUTION", "true")
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "post_task_evolution_request.json").write_text(
        json.dumps({"objective": "x"}), encoding="utf-8")
    import supervisor.evolution_lifecycle as lifecycle
    monkeypatch.setattr(lifecycle, "evolution_block_reason", lambda: "light mode")
    assert pte.apply_pending_request(tmp_path) is False
    # stale request dropped
    assert not (tmp_path / "state" / "post_task_evolution_request.json").exists()


def test_promotion_chooser_uses_main_model_slot(tmp_path, monkeypatch):
    """Plan 5C owner decision: the promotion chooser runs on the MAIN model
    slot (medium effort, 8192 output budget), not the light/low/2048 lane.
    Pins the upgrade against future cost-optimization regressions."""
    calls = {}

    def fake_chat_observed(_client, **kwargs):
        calls.update(kwargs)
        return {"content": json.dumps({"promote": False, "objective": ""})}, {}

    import ouroboros.llm_observability as obs
    monkeypatch.setattr(obs, "chat_observed", fake_chat_observed)
    monkeypatch.setenv("OUROBOROS_MODEL", "test-provider/main-model")

    env = types.SimpleNamespace(drive_root=str(tmp_path))
    pte._decide_promotion(env, {"id": "t1"}, {"reflection": "r"}, object(), force=False)

    assert calls.get("model") == "test-provider/main-model"
    assert calls.get("reasoning_effort") == "medium"
    assert calls.get("max_tokens") == 8192

    # Empty env falls back to the SETTINGS_DEFAULTS main-slot value.
    calls.clear()
    monkeypatch.setenv("OUROBOROS_MODEL", "")
    pte._decide_promotion(env, {"id": "t2"}, {"reflection": "r"}, object(), force=False)
    assert calls.get("model") == config.SETTINGS_DEFAULTS["OUROBOROS_MODEL"]
