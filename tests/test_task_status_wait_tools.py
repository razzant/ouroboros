"""The wait tools: what a waiter is told, and what it refuses to claim.

Split out of ``tests/test_task_status_flow.py`` by theme: the compact structural batch,
the execution evidence projected for harness children, the unknown-id and phantom
handling with its children roster, the polling that never reads ``cancel_requested`` as
completion, and the argument validation both wait tools share.
"""

import json
import time
from types import SimpleNamespace


def test_wait_for_tasks_returns_compact_structural_batch(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.control import _wait_for_tasks
    from ouroboros.tools.join_ledger import _child_result_sha256

    child_drive = tmp_path / "state" / "headless_tasks" / "childdone" / "data"
    child_drive.mkdir(parents=True)
    write_task_result(
        tmp_path,
        "parentdone",
        STATUS_COMPLETED,
        result="parent finished",
        cost_usd=1.25,
        loop_outcome={"result_status": "succeeded", "compat_result_status": "succeeded"},
        verification_ledger={"entries": [{"kind": "objective_outcome"}]},
        trace_refs=[{"path": "logs/trace.jsonl"}],
    )
    write_task_result(tmp_path, "childdone", STATUS_SCHEDULED, child_drive_root=str(child_drive), result="queued")
    write_task_result(child_drive, "childdone", STATUS_COMPLETED, result="child finished", trace_summary="trace")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["parentdone", "childdone"], timeout_sec=0))

    # Wait-envelope keys are preserved unchanged.
    assert payload["all_terminal"] is True
    assert payload["timed_out"] is False
    assert payload["mode"] == "all_terminal"
    assert "elapsed_sec" in payload and "timeout_sec" in payload
    # Disclosed omission: the note points at the full on-disk envelope.
    assert "get_task_result" in payload["tasks_note"]

    parent = payload["tasks"]["parentdone"]
    assert parent["task_id"] == "parentdone"
    assert parent["status"] == STATUS_COMPLETED
    assert parent["result"] == "parent finished"
    assert parent["cost_usd"] == 1.25
    assert parent["outcome_axes"]["lifecycle"]["status"] == STATUS_COMPLETED
    # Forensics stay on disk — not inlined into the batch projection.
    assert "loop_outcome" not in parent
    assert "verification_ledger" not in parent
    assert "trace_refs" not in parent
    assert "duplicate_of" not in parent

    # child_result_sha256 reuses the join-ledger SSOT hash over the effective result.
    assert parent["child_result_sha256"] == _child_result_sha256(
        load_effective_task_result(tmp_path, "parentdone")
    )

    child = payload["tasks"]["childdone"]
    assert child["result"] == "child finished"
    assert child["trace_summary"] == "trace"
    assert child["cost_usd"] is None  # absent accounting -> honest null, not $0
    assert child["child_result_sha256"] == _child_result_sha256(
        load_effective_task_result(tmp_path, "childdone")
    )


def test_wait_for_tasks_projects_execution_evidence_for_harness_children(tmp_path):
    # Q1A (2026-08-10 amendments): the batch projection is the surface a fan-out
    # parent absorbs its children through, and it used to hide whether a
    # harness-dispatched child ever actually delegated (the e9108a09 shape:
    # nine "harness" children, zero delegated runs, invisible in the batch).
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path, "harnesskid", STATUS_COMPLETED, result="done",
        effective_executor="harness", executor_route="codex",
        actual_substrate="native_only",
        subagent_envelope={
            "actual_substrate": "native_only",
            "execution_evidence": {
                "delegated_runs_started": 0, "delegated_runs_settled": 0,
                "delegated_runs_succeeded": 0, "delegated_run_failure_states": [],
                "evidence_read_failed": False, "subscription_cost_usd": None,
                "subscription_cost_estimated": False, "harness_models": [],
            },
        },
    )
    write_task_result(tmp_path, "nativekid", STATUS_COMPLETED, result="done")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["harnesskid", "nativekid"], timeout_sec=0))

    assert payload["tasks"]["harnesskid"]["execution_evidence"] == {
        "delegated_runs_settled": 0,
        "delegated_runs_failed": 0,
        "native_contribution": "unknown",
        "dispatch_executor": "harness",
        "actual_substrate": "native_only",
        "delegated_runs_started": 0,
        "delegated_runs_succeeded": 0,
    }
    # A native child with no custody evidence stays compact — no evidence block.
    assert "execution_evidence" not in payload["tasks"]["nativekid"]


def test_wait_for_tasks_projection_marks_unreadable_evidence(tmp_path):
    # v6.94.0 landing-gate scope fix: unreadable custody evidence means the
    # counts are UNKNOWN — the projection carries ONLY dispatch_executor and
    # the typed evidence_read_failed marker. Emitting the raw zeros beside the
    # marker fabricated a "no runs" receipt for a log that was never read; the
    # substrate claim is likewise dropped even when the stored record carries
    # one (same omission rule subagents.envelope_from_task applies).
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path, "blindkid", STATUS_COMPLETED, result="done",
        effective_executor="harness", executor_route="codex",
        actual_substrate="native_only",
        subagent_envelope={
            "actual_substrate": "native_only",
            "execution_evidence": {
                "delegated_runs_started": 0, "delegated_runs_succeeded": 0,
                "evidence_read_failed": True,
            },
        },
    )
    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["blindkid"], timeout_sec=0))
    assert payload["tasks"]["blindkid"]["execution_evidence"] == {
        "dispatch_executor": "harness",
        "evidence_read_failed": True,
    }


def test_wait_for_tasks_projection_omits_counts_without_envelope_evidence(tmp_path):
    # 6c03c24e corrective wave (LOW b): a stored harness child with NO envelope
    # evidence at all (pre-6.94 records) must not read as a zero-run receipt —
    # absence means "no evidence yet", so no counts and no substrate claim.
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path, "oldkid", STATUS_COMPLETED, result="done",
        effective_executor="harness", executor_route="codex",
    )
    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["oldkid"], timeout_sec=0))
    assert payload["tasks"]["oldkid"]["execution_evidence"] == {
        "dispatch_executor": "harness",
    }


def test_wait_for_tasks_any_terminal_early_return_projects_pending_child(tmp_path):
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_SCHEDULED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(tmp_path, "fastchild", STATUS_COMPLETED, result="done first", cost_usd=0.10)
    write_task_result(tmp_path, "slowchild", STATUS_SCHEDULED, result="")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["fastchild", "slowchild"], timeout_sec=0, mode="any_terminal"))

    assert payload["mode"] == "any_terminal"
    assert payload["all_terminal"] is False
    assert payload["timed_out"] is False
    assert payload["tasks"]["fastchild"]["status"] == STATUS_COMPLETED
    assert payload["tasks"]["fastchild"]["cost_usd"] == 0.10
    # The still-pending child gets the same compact shape with cost present.
    assert payload["tasks"]["slowchild"]["status"] == STATUS_SCHEDULED
    assert "cost_usd" in payload["tasks"]["slowchild"]
    assert "child_result_sha256" in payload["tasks"]["slowchild"]


def test_wait_for_tasks_cost_present_on_cancelled_and_failed(tmp_path):
    from ouroboros.task_results import STATUS_CANCELLED, STATUS_FAILED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(tmp_path, "cancelledchild", STATUS_CANCELLED, result="best-effort partial handoff", cost_usd=0.42)
    write_task_result(tmp_path, "failedchild", STATUS_FAILED, result="provider exploded")

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["cancelledchild", "failedchild"], timeout_sec=0))

    cancelled = payload["tasks"]["cancelledchild"]
    assert cancelled["status"] == STATUS_CANCELLED
    assert cancelled["cost_usd"] == 0.42
    assert cancelled["result"] == "best-effort partial handoff"
    failed = payload["tasks"]["failedchild"]
    assert failed["status"] == STATUS_FAILED
    # Absent accounting projects an honest null — never a confirmed-looking $0
    # (triad v6.71.2 r1; mirrors the ledger's unknown-cost discipline).
    assert "cost_usd" in failed and failed["cost_usd"] is None
    assert "child_result_sha256" in failed


def test_wait_for_tasks_rejected_duplicate_carries_duplicate_of(tmp_path):
    from ouroboros.task_results import STATUS_REJECTED_DUPLICATE, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    write_task_result(
        tmp_path,
        "dupechild",
        STATUS_REJECTED_DUPLICATE,
        result="duplicate of original123",
        duplicate_of="original123",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    payload = json.loads(_wait_for_tasks(ctx, ["dupechild"], timeout_sec=0))

    dupe = payload["tasks"]["dupechild"]
    assert dupe["status"] == STATUS_REJECTED_DUPLICATE
    assert dupe["duplicate_of"] == "original123"
    assert "cost_usd" in dupe


# --- v6.91 wait terminality: cancel_requested is a latch, not a settled record
def test_wait_for_effective_tasks_keeps_polling_cancel_requested(tmp_path):
    """The cancel-INTENT latch is not settled: the worker may still be exiting
    and the supervisor finalizes to `cancelled` shortly after. Returning
    "completed after 0.0s" here (pre-v6.91 FINAL_STATUSES) disagreed with the
    acceptance fence's SETTLED_STATUSES quiescence and looped the parent on the
    gap (wave3's $1.64 endgame loop). The wait stays bounded by its timeout."""
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, STATUS_CANCELLED, write_task_result
    from ouroboros.task_status import wait_for_effective_tasks

    write_task_result(tmp_path, "cancelling1", STATUS_CANCEL_REQUESTED, result="cancel pending")

    waited = wait_for_effective_tasks(tmp_path, ["cancelling1"], timeout_sec=0)
    assert waited["all_terminal"] is False
    assert waited["timed_out"] is True
    # A pending cancellation is reported as the typed state — never terminal/unknown.
    assert waited["live_child_status"]["cancelling1"] == "cancel_pending"

    # Once the supervisor settles it, the same wait completes normally.
    write_task_result(tmp_path, "cancelling1", STATUS_CANCELLED, result="cancelled")
    waited = wait_for_effective_tasks(tmp_path, ["cancelling1"], timeout_sec=0)
    assert waited["all_terminal"] is True


def test_wait_task_does_not_claim_completion_on_cancel_requested(tmp_path):
    from ouroboros.task_results import STATUS_CANCEL_REQUESTED, write_task_result
    from ouroboros.tools.control import _wait_for_task

    write_task_result(tmp_path, "cancelling2", STATUS_CANCEL_REQUESTED, result="cancel pending")

    output = _wait_for_task(SimpleNamespace(drive_root=tmp_path), "cancelling2", timeout_sec=0)
    assert output.startswith("Task wait timed out")
    assert not output.startswith("Task wait completed")


# --- v6.91 wait_tasks typed unknown ids + children roster ---------------------
def test_wait_for_tasks_flags_unknown_ids_and_attaches_children_roster(tmp_path):
    import json as _json

    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _wait_for_tasks

    # A READABLE queue snapshot that does not know the phantom: a MISSING
    # snapshot fail-softs to "known" (never brand a real child unknown on an
    # unreadable surface), so the unknown verdict needs all surfaces present.
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(
        _json.dumps({"pending": [], "running": []}), encoding="utf-8"
    )
    write_task_result(
        tmp_path,
        "realchild1",
        STATUS_COMPLETED,
        result="real child finished",
        cost_usd=0.55,
        parent_task_id="waitparent1",
        root_task_id="waitparent1",
        delegation_role="subagent",
    )

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_id="waitparent1",
        task_metadata={"root_task_id": "waitparent1"},
    )
    payload = json.loads(_wait_for_tasks(ctx, ["realchild1", "phantomid9"], timeout_sec=0))

    # The phantom id gets a TYPED marker row, not a silent empty projection.
    phantom = payload["tasks"]["phantomid9"]
    assert phantom["unknown_task_id"] is True
    assert "not yet registered or never scheduled" in phantom["note"]
    assert payload["unknown_task_ids"] == ["phantomid9"]

    # The real child still projects the normal compact row.
    real = payload["tasks"]["realchild1"]
    assert real["status"] == STATUS_COMPLETED
    assert "unknown_task_id" not in real

    # The repair surface: the ACTUAL direct children, compact v6.71.2 field set
    # only — no result/trace envelope fields, absent accounting projects null.
    roster = payload["children_roster"]
    assert [row["task_id"] for row in roster] == ["realchild1"]
    assert set(roster[0]) == {"task_id", "status", "cost_usd", "accounted_upper_bound_usd",
                              "child_result_sha256", "outcome_axes"}
    assert roster[0]["cost_usd"] == 0.55
    # C2: the additive honest name carries the SAME value as the alias.
    assert roster[0]["accounted_upper_bound_usd"] == 0.55
    # Nothing was capped away, and the projection SAYS so (BIBLE P1).
    assert payload["children_roster_omitted"] == 0


def test_children_roster_projection_discloses_the_capped_tail(tmp_path):
    """A parent with MORE direct children than the roster cap: the repair surface
    stays bounded, but the bound is disclosed — `children_roster_omitted` carries
    the exact count of real children the cap hid. A silent [:30] here could hide
    the very replacement id wait_tasks' unknown-id repair exists to surface."""
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.control import _children_roster_projection

    total = 33
    for idx in range(total):
        write_task_result(
            tmp_path,
            f"bigchild{idx:03d}",
            STATUS_COMPLETED,
            result=f"child {idx} finished",
            parent_task_id="bigparent1",
            root_task_id="bigparent1",
            delegation_role="subagent",
        )

    ctx = SimpleNamespace(
        drive_root=tmp_path,
        task_id="bigparent1",
        task_metadata={"root_task_id": "bigparent1"},
    )
    projected = _children_roster_projection(ctx, tmp_path)
    roster = projected["children_roster"]
    assert len(roster) == 30  # the cap holds — the surface stays compact
    assert projected["children_roster_omitted"] == total - 30  # …and is disclosed
    assert all(
        set(row) == {"task_id", "status", "cost_usd", "accounted_upper_bound_usd",
                     "child_result_sha256", "outcome_axes"}
        for row in roster
    )


def test_wait_for_tasks_phantom_only_set_short_circuits_the_window(tmp_path, monkeypatch):
    """A wait set in which NOTHING was ever minted ends after the registration
    grace instead of blocking the whole requested window — and says so."""
    import json as _json

    from ouroboros.tools import control, control_task_results

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(
        _json.dumps({"pending": [], "running": []}), encoding="utf-8"
    )
    monkeypatch.setattr(control_task_results, "_UNMINTED_WAIT_GRACE_SEC", 0.1)

    ctx = SimpleNamespace(drive_root=tmp_path, task_id="waitparent3", task_metadata={})
    started = time.monotonic()
    payload = json.loads(control._wait_for_tasks(ctx, ["phantomid7", "phantomid8"], timeout_sec=600))
    elapsed = time.monotonic() - started

    assert elapsed < 30, "phantom-only wait must not block for the requested window"
    short = payload["wait_short_circuited"]
    assert short["reason"] == "all_task_ids_unminted"
    assert short["requested_timeout_sec"] == 600.0
    assert sorted(payload["unknown_task_ids"]) == ["phantomid7", "phantomid8"]


def test_wait_for_tasks_id_minted_during_grace_keeps_waiting(tmp_path, monkeypatch):
    """The grace is for the registration race: an id that becomes real during it
    is a genuine child, so the wait resumes with the remaining window."""
    import json as _json

    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools import control, control_task_results

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(
        _json.dumps({"pending": [], "running": []}), encoding="utf-8"
    )
    monkeypatch.setattr(control_task_results, "_UNMINTED_WAIT_GRACE_SEC", 0.1)

    real_calls = {"n": 0}
    original = control_task_results._unminted_wait_ids

    def _mint_after_grace(ctx, drive_root, task_ids):
        real_calls["n"] += 1
        if real_calls["n"] > 1:
            # The child registered during the grace window.
            write_task_result(
                tmp_path, "latechild1", STATUS_COMPLETED, result="registered late",
                parent_task_id="waitparent4", root_task_id="waitparent4",
                delegation_role="subagent",
            )
        return original(ctx, drive_root, task_ids)

    monkeypatch.setattr(control_task_results, "_unminted_wait_ids", _mint_after_grace)
    ctx = SimpleNamespace(drive_root=tmp_path, task_id="waitparent4", task_metadata={})
    payload = json.loads(control._wait_for_tasks(ctx, ["latechild1"], timeout_sec=5))

    assert "wait_short_circuited" not in payload
    assert payload["timeout_sec"] == 5.0
    assert payload["tasks"]["latechild1"]["status"] == STATUS_COMPLETED


def test_wait_for_tasks_queue_scheduled_id_is_not_unknown(tmp_path):
    """An id with a queue-snapshot row but no task result yet is a REAL child
    (just-scheduled), never a phantom — and without unknowns the roster is not
    attached (the compact batch stays compact, v6.71.2)."""
    import json as _json

    from ouroboros.tools.control import _wait_for_tasks

    snapshot = {"pending": [{"id": "queuedonly1", "task": {}}], "running": []}
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "queue_snapshot.json").write_text(_json.dumps(snapshot), encoding="utf-8")

    ctx = SimpleNamespace(drive_root=tmp_path, task_id="waitparent2", task_metadata={})
    payload = json.loads(_wait_for_tasks(ctx, ["queuedonly1"], timeout_sec=0))

    assert "unknown_task_ids" not in payload
    assert "children_roster" not in payload
    assert "unknown_task_id" not in payload["tasks"]["queuedonly1"]


def test_wait_for_task_times_out_when_child_is_not_terminal(tmp_path):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.control import _wait_for_task

    write_task_result(tmp_path, "stillrunning", STATUS_RUNNING, result="working")

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _wait_for_task(ctx, "stillrunning", timeout_sec=0)

    assert "Task wait timed out" in output
    assert "stillrunning [running]" in output


def test_wait_tools_reject_invalid_ids_and_cap_batch(tmp_path):
    from ouroboros.tools.control import _wait_for_task, _wait_for_tasks

    ctx = SimpleNamespace(drive_root=tmp_path)

    assert "TOOL_ARG_ERROR" in _wait_for_task(ctx, "../settings", timeout_sec=0)
    assert "TOOL_ARG_ERROR" in _wait_for_tasks(ctx, ["ok123", "../bad"], timeout_sec=0)
    from ouroboros.config import MAX_ACTIVE_SUBAGENTS_HARD_CAP
    assert MAX_ACTIVE_SUBAGENTS_HARD_CAP == 500
    assert "capped at 500" in _wait_for_tasks(
        ctx, [f"task{i}" for i in range(MAX_ACTIVE_SUBAGENTS_HARD_CAP + 1)], timeout_sec=0
    )


def test_wait_for_task_reports_rejected_duplicate(tmp_path):
    from ouroboros.task_results import STATUS_REJECTED_DUPLICATE, write_task_result
    from ouroboros.tools.control import _wait_for_task

    write_task_result(
        tmp_path,
        "dup123",
        STATUS_REJECTED_DUPLICATE,
        duplicate_of="orig999",
        result="Task was rejected as semantically similar to already active task orig999.",
    )

    ctx = SimpleNamespace(drive_root=tmp_path)
    output = _wait_for_task(ctx, "dup123")

    assert "rejected_duplicate" in output
    assert "duplicate_of=orig999" in output
