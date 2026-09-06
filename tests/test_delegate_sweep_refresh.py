"""Nanny-leaf S1 + custody-absorption D1 contracts: the periodic sweep's late
settlement refreshes a TERMINAL task's stored custody disclosure (audit-only —
never cancels), the boot backfill heals generation-crossing stale rows, the
kill paths clear a stale list, and the retry-lineage projection stops
resurrecting the cleared run."""

from __future__ import annotations

import pathlib
import types

from ouroboros import delegate_custody as custody
from ouroboros import delegate_terminal
from ouroboros.task_results import (
    STATUS_FAILED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)


def _stale_terminal_result(tmp_path: pathlib.Path, task_id: str, **extra) -> None:
    extra.setdefault("delegated_runs_unreconciled", ["run-stale"])
    write_task_result(
        tmp_path, task_id, STATUS_FAILED,
        reason_code="provider_unavailable",
        **extra,
    )


def _emit_started(tmp_path: pathlib.Path, run_id: str, task_id: str, **extra) -> None:
    payload = {"run_id": run_id, "task_id": task_id, "route": "claude", "shape": {}, **extra}
    assert custody.emit(tmp_path, custody.STARTED, payload)


def _emit_settled(
    tmp_path: pathlib.Path, run_id: str, task_id: str, *, cost: float = 1.25, **extra
) -> None:
    payload = {
        "run_id": run_id, "task_id": task_id, "route": "claude",
        "model": "claude-x", "state": "succeeded", "cost_usd": cost,
        "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
        **extra,
    }
    assert custody.emit(tmp_path, custody.SETTLED, payload)


def test_sweep_refresh_clears_stale_unreconciled_after_settlement(tmp_path):
    _stale_terminal_result(tmp_path, "t-stale")
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-stale") is True
    result = load_task_result(tmp_path, "t-stale")
    assert result["delegated_runs_unreconciled"] == []
    assert result["delegate_terminal_reconciliation"]["trigger"] == "sweep_refresh"
    # Q5=A: the primary reason survives — the refresh never rewrites reason_code.
    assert result["reason_code"] == "provider_unavailable"
    assert result["status"] == STATUS_FAILED


def test_sweep_refresh_skips_running_and_clean_tasks(tmp_path):
    write_task_result(
        tmp_path, "t-running", STATUS_RUNNING,
        delegated_runs_unreconciled=["run-live"],
    )
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-running") is False
    assert load_task_result(tmp_path, "t-running")["delegated_runs_unreconciled"] == ["run-live"]

    write_task_result(tmp_path, "t-clean", STATUS_FAILED)
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-clean") is False
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "") is False


def test_sweep_refresh_keeps_disclosure_while_custody_still_open(tmp_path):
    from ouroboros import delegate_custody as custody

    assert custody.record_start_requested(
        tmp_path, run_id="", task_id="t-open", invocation_id="inv-9",
        idempotency_key="inv-9", max_seconds=30, request={"prompt": "brief"},
        project_id="project-9", project_owned=False, route="codex",
    )
    _stale_terminal_result(tmp_path, "t-open")
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-open") is True
    result = load_task_result(tmp_path, "t-open")
    # The refresh writes the honest CURRENT audit, not a blind clear.
    assert result["delegated_runs_unreconciled"] == ["invocation:inv-9"]


# ---------------------------------------------------------------------------
# D1a — boot backfill (generation-crossing settlements)
# ---------------------------------------------------------------------------


def test_boot_backfill_fixes_row_settled_in_a_previous_generation(tmp_path):
    """A run settled by a PREVIOUS generation's sweep appears in no current
    pass's outcomes; only the boot backfill can heal that stored row. Q2=B: the
    frozen run counters and the primary reason survive the refresh — the
    envelope (trigger + open_run_ids) is the current-liveness surface."""
    _emit_started(tmp_path, "run-1", "t-gen")
    _emit_settled(tmp_path, "run-1", "t-gen")
    write_task_result(
        tmp_path, "t-gen", STATUS_FAILED,
        reason_code="delegated_custody_unreconciled",
        delegated_runs_unreconciled=["run-1"],
        delegated_runs_started=1, delegated_runs_settled=0,
    )

    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == ["t-gen"]

    result = load_task_result(tmp_path, "t-gen")
    assert result["delegated_runs_unreconciled"] == []
    envelope = result["delegate_terminal_reconciliation"]
    assert envelope["trigger"] == "boot_backfill"
    assert envelope["open_run_ids"] == []
    # Owner Q2=B: counters are a historical snapshot at the original terminal
    # write, never recomputed; reason_code stays untouched (Q5=A).
    assert result["delegated_runs_started"] == 1
    assert result["delegated_runs_settled"] == 0
    assert result["reason_code"] == "delegated_custody_unreconciled"
    assert result["status"] == STATUS_FAILED

    # R5: the healed CURRENT state is agent-visible on the full-handoff
    # surface — get_task_result shows the empty list and the envelope's
    # trigger + open_run_ids next to the historical axes.
    import json

    from ouroboros.tools.control import _get_task_result
    from ouroboros.tools.registry import ToolContext

    output = _get_task_result(
        ToolContext(repo_dir=tmp_path, drive_root=tmp_path), "t-gen",
    )
    payload = output.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0]
    custody = json.loads(payload)["delegated_custody"]
    assert custody == {
        "unreconciled": [], "trigger": "boot_backfill", "open_run_ids": [],
    }


def test_truncated_custody_disclosure_names_both_retry_lineage_rows(tmp_path):
    """When the full-handoff custody disclosure truncates AND the row carries
    retry lineage, the resolvable full source names BOTH rows — the union may
    have inherited identifiers from the original row (final-gate sol#1)."""
    import json

    from ouroboros.tools.control import _subtask_outcome_summary

    row = {
        "task_id": "t-retry",
        "original_task_id": "t-orig",
        "status": STATUS_FAILED,
        "delegated_runs_unreconciled": [f"run-{i}" for i in range(12)],
    }
    summary = json.loads(_subtask_outcome_summary(row))
    custody = summary["delegated_custody"]
    assert custody["unreconciled_omitted"] == 2
    assert custody["full_source"] == [
        "read_file(root='runtime_data', path='task_results/t-retry.json')",
        "read_file(root='runtime_data', path='task_results/t-orig.json')",
    ]


def test_stale_child_replica_cannot_reshadow_healed_custody_fields(tmp_path):
    """Split-drive shape: the boot backfill heals only the CANONICAL row, but
    effective reads overlay a retained child replica. The custody disclosure
    pair is canonical-authoritative once present, so a stale replica must not
    re-shadow the healed row (it may still fill a canonical row that lacks the
    fields — the pre-copy-back window)."""
    from ouroboros.task_status import effective_task_result

    child_drive = tmp_path / "child"
    (child_drive / "task_results").mkdir(parents=True)
    write_task_result(
        child_drive, "t-split", STATUS_FAILED,
        reason_code="delegated_custody_unreconciled",
        delegated_runs_unreconciled=["run-9"],
    )
    write_task_result(
        tmp_path, "t-split", STATUS_FAILED,
        reason_code="delegated_custody_unreconciled",
        delegated_runs_unreconciled=[],
        delegate_terminal_reconciliation={
            "task_id": "t-split", "trigger": "boot_backfill", "outcomes": [],
            "unreconciled": [], "audit_status": "ok", "open_run_ids": [],
        },
        child_drive_root=str(child_drive),
    )

    merged = effective_task_result(tmp_path, load_task_result(tmp_path, "t-split"))

    assert merged["delegated_runs_unreconciled"] == []
    assert merged["delegate_terminal_reconciliation"]["trigger"] == "boot_backfill"

    # The pre-copy-back window still fills from the replica when the canonical
    # row carries no custody fields at all.
    write_task_result(
        tmp_path, "t-fill", STATUS_FAILED, reason_code="x",
        child_drive_root=str(child_drive),
    )
    write_task_result(
        child_drive, "t-fill", STATUS_FAILED, reason_code="x",
        delegated_runs_unreconciled=["run-fill"],
    )
    filled = effective_task_result(tmp_path, load_task_result(tmp_path, "t-fill"))
    assert filled["delegated_runs_unreconciled"] == ["run-fill"]


def test_boot_backfill_preserves_undisposed_patch_debt(tmp_path):
    """The incident shape: settled mutating run whose captured patch awaits its
    owner's disposition. The backfill rewrites the honest CURRENT audit —
    ``patch:<run_id>`` — and never blindly clears the disclosure to []."""
    _emit_started(tmp_path, "run-2", "t-incident", snapshot_id="snap-2")
    _emit_settled(tmp_path, "run-2", "t-incident")
    _stale_terminal_result(tmp_path, "t-incident",
                           delegated_runs_unreconciled=["run-2"])

    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == ["t-incident"]
    assert load_task_result(tmp_path, "t-incident")["delegated_runs_unreconciled"] == [
        "patch:run-2",
    ]


def test_second_boot_is_a_byte_level_noop_for_a_permanently_stale_row(tmp_path):
    """A row the audit cannot improve (undisposed patch) must not rewrite the
    result or append custody events on every boot."""
    _emit_started(tmp_path, "run-2", "t-incident", snapshot_id="snap-2")
    _emit_settled(tmp_path, "run-2", "t-incident")
    _stale_terminal_result(tmp_path, "t-incident",
                           delegated_runs_unreconciled=["run-2"])
    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == ["t-incident"]

    row_path = tmp_path / "task_results" / "t-incident.json"
    events_path = tmp_path / "logs" / "events.jsonl"
    row_before = row_path.read_bytes()
    events_before = events_path.read_bytes()

    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == []

    assert row_path.read_bytes() == row_before
    assert events_path.read_bytes() == events_before


def test_boot_backfill_adds_the_envelope_to_a_flat_only_kill_row(tmp_path):
    """R2: a kill-written row carrying only the flat list (no envelope) is NOT
    current — the next boot attaches the envelope once, and the boot after
    that is a byte-level no-op."""
    _emit_started(tmp_path, "run-6", "t-flat")
    write_task_result(
        tmp_path, "t-flat", STATUS_FAILED,
        delegated_runs_unreconciled=["run-6"],
    )

    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == ["t-flat"]
    result = load_task_result(tmp_path, "t-flat")
    assert result["delegated_runs_unreconciled"] == ["run-6"]
    envelope = result["delegate_terminal_reconciliation"]
    assert envelope["trigger"] == "boot_backfill"
    assert envelope["open_run_ids"] == ["run-6"]

    row_path = tmp_path / "task_results" / "t-flat.json"
    events_path = tmp_path / "logs" / "events.jsonl"
    row_before, events_before = row_path.read_bytes(), events_path.read_bytes()
    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == []
    assert row_path.read_bytes() == row_before
    assert events_path.read_bytes() == events_before


def test_lock_timeout_refresh_is_false_and_emits_no_evidence(tmp_path, monkeypatch):
    """R3 honesty: a refresh whose write never landed returns False and emits
    NO custody evidence — the event may not claim a heal the store refused —
    and the row heals on the next boot."""
    import ouroboros.task_results as task_results_mod

    _emit_started(tmp_path, "run-7", "t-lock2")
    _emit_settled(tmp_path, "run-7", "t-lock2")
    _stale_terminal_result(tmp_path, "t-lock2",
                           delegated_runs_unreconciled=["run-7"])
    events_path = tmp_path / "logs" / "events.jsonl"
    events_before = events_path.read_bytes()

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("lock held elsewhere")

    monkeypatch.setattr(task_results_mod, "update_json_locked", _timeout)
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-lock2") is False
    assert events_path.read_bytes() == events_before
    assert load_task_result(tmp_path, "t-lock2")["delegated_runs_unreconciled"] == ["run-7"]

    monkeypatch.undo()
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-lock2") is True
    assert load_task_result(tmp_path, "t-lock2")["delegated_runs_unreconciled"] == []


def test_boot_backfill_shares_one_custody_replay_snapshot(tmp_path, monkeypatch):
    """Auditing N stored rows must not rescan the unbounded event log 4·N
    times: the whole backfill pays exactly one ``replay()`` pass."""
    for index in range(3):
        run_id, task_id = f"run-{index}", f"t-batch-{index}"
        _emit_started(tmp_path, run_id, task_id)
        _emit_settled(tmp_path, run_id, task_id)
        _stale_terminal_result(tmp_path, task_id,
                               delegated_runs_unreconciled=[run_id])

    replays: list = []
    real_replay = custody.replay
    monkeypatch.setattr(
        custody, "replay",
        lambda *args, **kwargs: replays.append(1) or real_replay(*args, **kwargs),
    )
    refreshed = delegate_terminal.backfill_terminal_reconciliations(tmp_path)
    assert sorted(refreshed) == ["t-batch-0", "t-batch-1", "t-batch-2"]
    assert len(replays) == 1


def test_backfill_lock_timeout_is_fail_soft_and_heals_on_the_next_boot(tmp_path, monkeypatch):
    import ouroboros.task_results as task_results_mod

    _emit_started(tmp_path, "run-4", "t-locked")
    _emit_settled(tmp_path, "run-4", "t-locked")
    _stale_terminal_result(tmp_path, "t-locked",
                           delegated_runs_unreconciled=["run-4"])

    def _timeout(*_args, **_kwargs):
        raise TimeoutError("lock held elsewhere")

    monkeypatch.setattr(task_results_mod, "update_json_locked", _timeout)
    delegate_terminal.backfill_terminal_reconciliations(tmp_path)  # must not raise
    assert load_task_result(tmp_path, "t-locked")["delegated_runs_unreconciled"] == ["run-4"]

    monkeypatch.undo()
    assert delegate_terminal.backfill_terminal_reconciliations(tmp_path) == ["t-locked"]
    assert load_task_result(tmp_path, "t-locked")["delegated_runs_unreconciled"] == []


def test_same_boot_sweep_settlement_is_cleared_in_the_same_generation(tmp_path, monkeypatch):
    """Server ordering (fable #9): the backfill runs AFTER the startup orphan
    reconcile, so a settlement performed by THIS boot's sweep — even one whose
    outcome shape the sweep-side refresh filter never sees — is already visible
    to the backfill audit and the stored row heals in the same generation."""
    # Campaign owner: the startup sweep lives in ouroboros.server_maintenance
    # (server.py delegates); DATA_DIR is its module-level binding.
    from ouroboros import delegate_custody as custody_mod
    from ouroboros import server_maintenance as server_mod

    _emit_started(tmp_path, "run-5", "t-late")
    _stale_terminal_result(tmp_path, "t-late", delegated_runs_unreconciled=["run-5"])
    monkeypatch.setattr(server_mod, "DATA_DIR", tmp_path)

    def fake_reconcile(drive_root, *, running_task_ids, gateway_factory,
                       recoverable_task_ids):
        _emit_settled(tmp_path, "run-5", "t-late")
        return []

    monkeypatch.setattr(custody_mod, "reconcile_orphaned_runs", fake_reconcile)
    server_mod._startup_custody_sweep()

    result = load_task_result(tmp_path, "t-late")
    assert result["delegated_runs_unreconciled"] == []
    assert result["delegate_terminal_reconciliation"]["trigger"] == "boot_backfill"


# ---------------------------------------------------------------------------
# D1b — kill-path clear
# ---------------------------------------------------------------------------


def test_recorder_and_refresh_never_mint_a_row_for_an_absent_task(tmp_path):
    """The mandatory D1b pin, unit half: a task with NO stored row can never be
    minted through the recorder's STATUS_RUNNING fallback by a clean audit, and
    the guarded kill-path refresh touches only an existing row with a non-empty
    stored list."""
    clean = {
        "task_id": "t-none", "trigger": "cancel_publication",
        "outcomes": [], "unreconciled": [], "audit_status": "ok",
        "deferred_project_retirements": [],
    }
    delegate_terminal.record_terminal_reconciliation(tmp_path, "t-none", clean)
    assert delegate_terminal.refresh_terminal_reconciliation(
        tmp_path, "t-none", trigger="kill_path_clear") is False
    assert not (tmp_path / "task_results" / "t-none.json").exists()


def _kill_env(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    return types.SimpleNamespace(q=q, tl=task_lifecycle, drive=tmp_path)


def test_kill_fast_lane_clears_a_stale_disclosure(tmp_path, monkeypatch):
    """The fast already-settled cancel lane performs no terminal write of its
    own; the guarded refresh clears a stale stored list there (D1b)."""
    from ouroboros import cancel_intents as ci

    env = _kill_env(tmp_path, monkeypatch)
    write_task_result(
        env.drive, "t-fast", STATUS_FAILED, result="settled long ago",
        reason_code="delegated_custody_unreconciled",
        delegated_runs_unreconciled=["run-stale"], delegated_runs_settled=0,
    )
    ci.request_cancel(env.drive, "t-fast")

    assert env.tl.cancel_task_custody("t-fast", deliver=False) == env.tl.CANCEL_ALREADY_SETTLED

    stored = load_task_result(env.drive, "t-fast")
    assert stored["status"] == STATUS_FAILED  # settled truth survives the kill
    assert stored["delegated_runs_unreconciled"] == []
    assert stored["delegate_terminal_reconciliation"]["trigger"] == "kill_path_clear"
    # Q2=B / Q5=A: frozen counter and primary reason are untouched.
    assert stored["delegated_runs_settled"] == 0
    assert stored["reason_code"] == "delegated_custody_unreconciled"


def test_kill_fast_lane_adds_no_write_when_nothing_is_stale(tmp_path, monkeypatch):
    """The mandatory D1b pin, lane half: an ordinary fast-lane kill (no stale
    stored list) performs ZERO task-result writes — no RUNNING mint, no second
    write per kill."""
    import ouroboros.task_results as task_results_mod
    from ouroboros import cancel_intents as ci

    env = _kill_env(tmp_path, monkeypatch)
    write_task_result(env.drive, "t-clean-kill", STATUS_FAILED, result="settled")
    ci.request_cancel(env.drive, "t-clean-kill")
    row_path = tmp_path / "task_results" / "t-clean-kill.json"
    row_before = row_path.read_bytes()

    writes: list = []
    real_write = task_results_mod.write_task_result
    monkeypatch.setattr(
        task_results_mod, "write_task_result",
        lambda *args, **kwargs: writes.append(args) or real_write(*args, **kwargs),
    )
    assert env.tl.cancel_task_custody(
        "t-clean-kill", deliver=False) == env.tl.CANCEL_ALREADY_SETTLED
    assert writes == []
    assert row_path.read_bytes() == row_before


def test_miss_lane_already_settled_clears_stale_disclosure(tmp_path, monkeypatch):
    """R4: the finalize-on-miss already-settled branch performs no terminal
    write of its own; the guarded kill-path refresh clears a stale stored
    disclosure there, and a fresh row stays byte-identical (never minted)."""
    from supervisor import cancel_publication as cp
    from supervisor import terminal_delivery as td

    env = _kill_env(tmp_path, monkeypatch)
    monkeypatch.setattr(td, "deliver_miss_lane_outcome", lambda *a, **kw: True)
    monkeypatch.setattr(env.tl, "_settle_intent", lambda *a, **kw: None)

    # The incident ordering: the intent was captured while the task still
    # looked live, and the row settled before finalize-on-miss re-checked.
    intent = {"task_id": "t-miss-stale", "request_id": "r1", "generation": 1}
    write_task_result(
        env.drive, "t-miss-stale", STATUS_FAILED, result="settled elsewhere",
        delegated_runs_unreconciled=["run-stale"],
    )
    assert cp._finalize_cancel_intent_on_miss(
        env.q, "t-miss-stale", intent=intent) == cp.CANCEL_ALREADY_SETTLED
    stored = load_task_result(env.drive, "t-miss-stale")
    assert stored["status"] == STATUS_FAILED
    assert stored["delegated_runs_unreconciled"] == []
    assert stored["delegate_terminal_reconciliation"]["trigger"] == "kill_path_clear"

    # Fresh variant: nothing stale — zero task-result writes, byte-identical.
    import ouroboros.task_results as task_results_mod

    write_task_result(env.drive, "t-miss-clean", STATUS_FAILED, result="settled")
    row_path = tmp_path / "task_results" / "t-miss-clean.json"
    row_before = row_path.read_bytes()
    writes: list = []
    real_write = task_results_mod.write_task_result
    monkeypatch.setattr(
        task_results_mod, "write_task_result",
        lambda *args, **kwargs: writes.append(args) or real_write(*args, **kwargs),
    )
    assert cp._finalize_cancel_intent_on_miss(
        env.q, "t-miss-clean",
        intent={"task_id": "t-miss-clean", "request_id": "r2", "generation": 1},
    ) == cp.CANCEL_ALREADY_SETTLED
    assert writes == []
    assert row_path.read_bytes() == row_before


def test_reaper_self_finalized_branch_clears_stale_disclosure(tmp_path, monkeypatch):
    """R4: a reap that finds the worker's OWN terminal result (self-finalized)
    keeps that write untouched but clears a stale stored disclosure through
    the guarded kill-path refresh; a fresh row stays byte-identical."""
    from supervisor import task_reaper, workers
    from supervisor import terminal_delivery as td

    env = _kill_env(tmp_path, monkeypatch)
    monkeypatch.setattr(workers, "get_event_q",
                        lambda: types.SimpleNamespace(put=lambda evt: None),
                        raising=False)
    monkeypatch.setattr(td, "deliver_miss_lane_outcome", lambda *a, **kw: True)
    monkeypatch.setattr("ouroboros.delegate_custody.reconcile_task_runs",
                        lambda *a, **kw: [])

    def _reap(task_id):
        task_reaper.reap_timed_out_task({
            "worker_id": 1, "proc": None, "task_id": task_id,
            "task": {"id": task_id, "chat_id": 3}, "task_type": "chat",
            "terminal_reason": "idle_timeout", "attempt": 1, "owner_chat_id": 3,
            "runtime_sec": 10.0, "will_retry": False, "retry_task_id": "",
        })

    write_task_result(
        env.drive, "t-self-stale", STATUS_FAILED, result="self-finalized",
        delegated_runs_unreconciled=["run-stale"],
    )
    _reap("t-self-stale")
    stored = load_task_result(env.drive, "t-self-stale")
    assert stored["status"] == STATUS_FAILED
    assert stored["delegated_runs_unreconciled"] == []
    assert stored["delegate_terminal_reconciliation"]["trigger"] == "kill_path_clear"

    # Fresh variant: nothing stale — zero task-result writes, byte-identical.
    import ouroboros.task_results as task_results_mod

    write_task_result(env.drive, "t-self-clean", STATUS_FAILED, result="self-finalized")
    row_path = tmp_path / "task_results" / "t-self-clean.json"
    row_before = row_path.read_bytes()
    writes: list = []
    real_write = task_results_mod.write_task_result
    monkeypatch.setattr(
        task_results_mod, "write_task_result",
        lambda *args, **kwargs: writes.append(args) or real_write(*args, **kwargs),
    )
    _reap("t-self-clean")
    assert writes == []
    assert row_path.read_bytes() == row_before


def test_retry_lineage_stops_resurrecting_cleared_run(tmp_path):
    from ouroboros.task_status import effective_task_result

    _stale_terminal_result(tmp_path, "t-orig", retry_task_id="t-retry")
    write_task_result(tmp_path, "t-retry", STATUS_FAILED, reason_code="provider_unavailable")

    before = effective_task_result(tmp_path, load_task_result(tmp_path, "t-orig"))
    assert "run-stale" in (before.get("delegated_runs_unreconciled") or [])

    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-orig") is True
    after = effective_task_result(tmp_path, load_task_result(tmp_path, "t-orig"))
    assert "run-stale" not in (after.get("delegated_runs_unreconciled") or [])

def test_refresh_rewrites_stale_substrate_counters_after_late_settlement(tmp_path):
    """The scout-A contract, reconciled with owner Q2=B by SURFACE: a terminal
    result written BEFORE the run settled must, after refresh, tell the truth
    in the CURRENT-TRUTH fields readers consume — actual_substrate and the
    envelope's evidence (where the chip reads subscription_cost_usd) — while
    the top-level delegated_runs_* counters stay the historical snapshot at
    the original terminal write, never recomputed."""
    _emit_started(tmp_path, "run-1", "t-late")
    write_task_result(
        tmp_path, "t-late", STATUS_FAILED,
        reason_code="provider_unavailable",
        delegated_runs_unreconciled=["run-1"],
        actual_substrate="harness_attempted",
        delegated_runs_started=1, delegated_runs_settled=0,
        delegated_runs_succeeded=0, delegated_runs_failed=0,
        delegated_runs_source_unresolved=0,
        subagent_envelope={
            "executor_route": "claude", "effective_executor": "harness",
            "actual_substrate": "harness_attempted",
            "execution_evidence": {
                "delegated_runs_started": 1, "delegated_runs_settled": 0,
                "delegated_runs_succeeded": 0, "delegated_runs_failed": 0,
                "subscription_cost_usd": None,
            },
        },
    )
    _emit_settled(tmp_path, "run-1", "t-late", cost=1.25)

    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-late") is True
    result = load_task_result(tmp_path, "t-late")
    assert result["delegated_runs_unreconciled"] == []
    assert result["actual_substrate"] == "harness_used"
    # Q2=B: the top-level counters are the frozen historical snapshot.
    assert result["delegated_runs_settled"] == 0
    assert result["delegated_runs_succeeded"] == 0
    ev = result["subagent_envelope"]["execution_evidence"]
    assert ev["delegated_runs_succeeded"] == 1
    assert ev["subscription_cost_usd"] == 1.25
    # Dispatch decisions are never overwritten by evidence reconciliation.
    assert result["subagent_envelope"]["executor_route"] == "claude"
    # Q5=A: the primary reason stays untouched.
    assert result["reason_code"] == "provider_unavailable"


def test_refresh_never_mints_evidence_for_undelegated_tasks(tmp_path):
    """A terminal task that never carried the harness-dispatch mirror gets no
    fabricated substrate block from a refresh."""
    write_task_result(tmp_path, "t-native", STATUS_FAILED)
    assert delegate_terminal.refresh_terminal_reconciliation(tmp_path, "t-native") is False
    result = load_task_result(tmp_path, "t-native")
    assert "actual_substrate" not in result


def test_cursor_pass_heals_tasks_the_outcome_sweep_never_names(tmp_path):
    """A run settled OUTSIDE the sweep's reconcile outcomes (terminal-boundary
    settlement) is found by the cursor pass over newly appended custody rows;
    a second tick with no new rows does no work."""
    _emit_started(tmp_path, "run-9", "t-boundary")
    write_task_result(
        tmp_path, "t-boundary", STATUS_FAILED,
        delegated_runs_unreconciled=[],
        actual_substrate="harness_attempted",
        delegated_runs_started=1, delegated_runs_settled=0,
        delegated_runs_succeeded=0, delegated_runs_failed=0,
        delegated_runs_source_unresolved=0,
    )
    _emit_settled(tmp_path, "run-9", "t-boundary")

    assert delegate_terminal.refresh_recently_settled_terminals(tmp_path) == 1
    result = load_task_result(tmp_path, "t-boundary")
    assert result["actual_substrate"] == "harness_used"
    # Q2=B: top-level counters stay the historical snapshot; the healed
    # current truth lives in the envelope evidence the readers consume.
    assert result["delegated_runs_succeeded"] == 0
    assert result["subagent_envelope"]["execution_evidence"]["delegated_runs_succeeded"] == 1
    # Cursor advanced: the same rows are never reprocessed.
    assert delegate_terminal.refresh_recently_settled_terminals(tmp_path) == 0


def test_cursor_defers_running_task_without_starving_later_settlements(tmp_path):
    """The grok finding: a SETTLED row for a still-RUNNING parent must not pin
    the byte offset — later settlements past the per-tick window would starve.
    The offset always advances; the deferred task rides a durable map and is
    healed on the tick after its result turns terminal."""
    import json

    _emit_started(tmp_path, "run-a", "t-longlived")
    write_task_result(
        tmp_path, "t-longlived", STATUS_RUNNING,
        actual_substrate="harness_attempted",
        delegated_runs_started=1, delegated_runs_settled=0,
        delegated_runs_succeeded=0, delegated_runs_failed=0,
        delegated_runs_source_unresolved=0,
    )
    _emit_settled(tmp_path, "run-a", "t-longlived")
    assert delegate_terminal.refresh_recently_settled_terminals(tmp_path) == 0
    cursor = json.loads(
        (tmp_path / "state/delegate_terminal_refresh_cursor.json").read_text()
    )
    assert cursor["offset"] > 0  # never pinned on the deferred row
    assert list(cursor["deferred"]) == ["t-longlived"]

    # A LATER terminal-boundary settlement heals immediately — not starved
    # behind the still-running parent.
    _emit_started(tmp_path, "run-b", "t-later")
    write_task_result(
        tmp_path, "t-later", STATUS_FAILED,
        actual_substrate="harness_attempted",
        delegated_runs_started=1, delegated_runs_settled=0,
        delegated_runs_succeeded=0, delegated_runs_failed=0,
        delegated_runs_source_unresolved=0,
    )
    _emit_settled(tmp_path, "run-b", "t-later")
    assert delegate_terminal.refresh_recently_settled_terminals(tmp_path) == 1
    assert load_task_result(tmp_path, "t-later")["actual_substrate"] == "harness_used"

    # The deferred parent heals from the map once terminal, long after its
    # rows fell behind the offset.
    write_task_result(
        tmp_path, "t-longlived", STATUS_FAILED,
        actual_substrate="harness_attempted",
        delegated_runs_started=1, delegated_runs_settled=0,
        delegated_runs_succeeded=0, delegated_runs_failed=0,
        delegated_runs_source_unresolved=0,
    )
    assert delegate_terminal.refresh_recently_settled_terminals(tmp_path) == 1
    assert load_task_result(tmp_path, "t-longlived")["actual_substrate"] == "harness_used"
    cursor = json.loads(
        (tmp_path / "state/delegate_terminal_refresh_cursor.json").read_text()
    )
    assert cursor["deferred"] == {}
