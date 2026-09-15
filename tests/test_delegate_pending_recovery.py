"""Pending-invocation recovery must preserve one exact physical intention."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.delegate_shared import _fail, delegate_result


def _session_snapshot() -> dict:
    from ouroboros.subagent_runtime import select_subagent_snapshot

    settings = {
        "OUROBOROS_SUBAGENTS": json.dumps({
            "enabled": True,
            "items": [{
                "subagent_id": "session-builder",
                "name": "Session builder",
                "recommended_use": "Subscription-backed implementation.",
                "route": {
                    "kind": "agent_session",
                    "target_id": "codex=gpt-5.6-sol",
                    "credential_profile_id": "profile-1",
                },
                "effort": "high",
            }],
        }),
    }
    return select_subagent_snapshot(settings, subagent_id="session-builder")[0]


def _pending_handoff(tmp_path, task_id: str):
    import ouroboros.delegate_recovery as recovery
    from ouroboros import delegate_custody as custody
    from ouroboros.subagent_work_order import work_order_fingerprint

    custody._CUSTODY.clear()
    snapshot = _session_snapshot()
    task = {
        "id": task_id,
        "_attempt": 2,
        "configured_subagent": snapshot,
        "drive_root": str(tmp_path),
        "task_constraint": {},
        "task_contract": {"objective": "Build", "expected_output": "Patch"},
    }
    authority = recovery.authority_fingerprint_from_task(task)
    fingerprint = work_order_fingerprint(task)
    invocation_id = f"inv-{task_id}"
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id=task_id,
        invocation_id=invocation_id,
        idempotency_key=invocation_id,
        max_seconds=60,
        request={"prompt": "stored canonical request"},
        project_id="",
        project_owned=False,
        route="codex",
        selected_subagent_id=snapshot["selected_subagent_id"],
        config_fingerprint=snapshot["config_fingerprint"],
        work_order_fingerprint=fingerprint,
        authority_fingerprint=authority,
    )
    handoff = recovery.prepare_handoff(
        tmp_path,
        task,
        cause=recovery.CAUSE_WORKER_CRASH,
        old_attempt=1,
        new_attempt=2,
        worker_id=1,
        exitcode=1,
    )
    assert handoff["pending_invocation_id"] == invocation_id
    return custody, recovery, task, snapshot, invocation_id, fingerprint, authority


def test_exact_pending_retry_bypasses_fresh_zero_run_unknown_fence(
    monkeypatch, tmp_path,
):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate
    from ouroboros import delegate_custody as custody

    calls = []
    monkeypatch.setattr(
        delegate,
        "_delegate_start",
        lambda _ctx, prompt, _max_seconds, retry_of, **_kwargs: (
            calls.append((prompt, retry_of))
            or delegate_result({"status": "started", "run_id": "run-recovered"})
        ),
    )
    ctx = SimpleNamespace(
        task_id="retry-with-zero-run-gap",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "zero_run_evidence_status": "unknown",
            "zero_run_evidence_gaps": ["verification_receipts_incomplete"],
            "exact_start_pending": False,
        },
    )
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id=ctx.task_id,
        invocation_id="inv-recovery",
        idempotency_key="inv-recovery",
        max_seconds=60,
        request={"prompt": "stored canonical request"},
        project_id="",
        project_owned=False,
        route="codex",
    )

    result = json.loads(runtime.exact_start(
        ctx, "stored canonical request", {"retry_of": "inv-recovery"},
    ).text)

    assert result["status"] == "started"
    assert calls == [("stored canonical request", "inv-recovery")]


def test_unknown_retry_token_cannot_bypass_zero_run_unknown_fence(
    monkeypatch, tmp_path,
):
    import ouroboros.subagent_runtime as runtime
    import ouroboros.tools.delegate as delegate

    monkeypatch.setattr(
        delegate,
        "_delegate_start",
        lambda *_args, **_kwargs: pytest.fail(
            "an unowned or missing retry token must not bypass the zero-run fence"
        ),
    )
    ctx = SimpleNamespace(
        task_id="retry-without-pending-invocation",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
        _configured_actor_bootstrap={
            "zero_run_evidence_status": "unknown",
            "zero_run_evidence_gaps": ["verification_receipts_incomplete"],
            "exact_start_pending": False,
        },
    )

    result = json.loads(runtime.exact_start(
        ctx, "stored canonical request", {"retry_of": "unknown-invocation"},
    ).text)

    assert result["status"] == "refused"
    assert result["reason"] == "zero_run_evidence_unavailable"


def test_retry_refusal_preserves_live_owner_handoff_and_pending_invocation(
    monkeypatch, tmp_path,
):
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, _snapshot, invocation_id, _fingerprint, _authority = (
        _pending_handoff(tmp_path, "recovery-deferred")
    )
    monkeypatch.setattr(
        delegate,
        "exact_start",
        lambda *_args, **_kwargs: _fail(
            "delegate_start", "subscription_window_exhausted",
            "the engine subscription window is spent",
        ),
    )
    monkeypatch.setattr(
        custody,
        "reconcile_task_runs",
        lambda *_args, **_kwargs: pytest.fail(
            "a live recovery owner must not enter owner-gone reconciliation"
        ),
    )
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    result = recovery.adopt_handoff(ctx, task)

    assert result["reason"] == "pending_recovery_deferred"
    assert result["pending_invocation_id"] == invocation_id
    assert recovery._read(tmp_path, task["id"])["status"] == "adopted"
    assert task["id"] not in recovery.recoverable_task_ids(tmp_path)
    assert [row["invocation_id"] for row in custody.pending_invocations(tmp_path)] == [
        invocation_id
    ]

    replayed = []

    class OrphanGateway:
        def handshake(self):
            return {}

        def start_run(self, _body, *, idempotency_key):
            replayed.append(idempotency_key)
            return {}

        def close(self):
            pass

    custody.reconcile_orphaned_runs(
        tmp_path,
        running_task_ids=set(),
        recoverable_task_ids=recovery.recoverable_task_ids(tmp_path),
        gateway_factory=lambda: OrphanGateway(),
    )
    assert replayed == [invocation_id]
    custody._CUSTODY.clear()


def test_definite_retry_refusal_retires_handoff_without_false_pending_claim(
    monkeypatch, tmp_path,
):
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, _snapshot, invocation_id, _fingerprint, _authority = (
        _pending_handoff(tmp_path, "recovery-definite")
    )

    def refuse_definitely(*_args, **_kwargs):
        assert custody.emit(tmp_path, custody.START_FAILED, {
            "task_id": task["id"],
            "invocation_id": invocation_id,
            "definite": True,
        })
        return _fail("delegate_start", "route_refused", "the engine refused this route")

    monkeypatch.setattr(delegate, "exact_start", refuse_definitely)
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    result = recovery.adopt_handoff(ctx, task)

    assert result["reason"] == "pending_recovery_definitely_refused"
    assert "pending_invocation_id" not in result
    assert custody.invocation_record(tmp_path, invocation_id)["state"] == "failed_definite"
    assert recovery._read(tmp_path, task["id"])["status"] == "vetoed"
    assert task["id"] not in recovery.recoverable_task_ids(tmp_path)
    custody._CUSTODY.clear()


def test_started_retry_race_adopts_same_durable_invocation(monkeypatch, tmp_path):
    import ouroboros.claudexor_daemon as daemon
    import ouroboros.tools.delegate as delegate
    from ouroboros.tools.registry import ToolContext

    custody, recovery, task, snapshot, invocation_id, fingerprint, authority = (
        _pending_handoff(tmp_path, "recovery-started-race")
    )

    def expose_started_race(*_args, **_kwargs):
        assert custody.record_started(tmp_path, custody.RunCustody(
            run_id="run-started-race",
            task_id=task["id"],
            route_id="codex",
            invocation_id=invocation_id,
            selected_subagent_id=snapshot["selected_subagent_id"],
            config_fingerprint=snapshot["config_fingerprint"],
            work_order_fingerprint=fingerprint,
            authority_fingerprint=authority,
        ))
        return _fail(
            "delegate_start", "invocation_already_started",
            "that invocation already bound a run", run_id="run-started-race",
        )

    class Gateway:
        def get_run(self, run_id):
            assert run_id == "run-started-race"
            return {"id": run_id, "state": "running"}

        def close(self):
            pass

    monkeypatch.setattr(delegate, "exact_start", expose_started_race)
    monkeypatch.setattr(daemon, "ensure_owned_gateway", lambda: Gateway())
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task["id"])
    ctx.budget_drive_root = str(tmp_path)

    result = recovery.adopt_handoff(ctx, task)

    assert result == {
        "status": "adopted",
        "run_id": "run-started-race",
        "cause": recovery.CAUSE_WORKER_CRASH,
    }
    assert recovery._read(tmp_path, task["id"])["status"] == "adopted"
    custody._CUSTODY.clear()


def test_review_substrate_work_does_not_hold_the_actors_zero_run_fence(tmp_path):
    """A pending review invocation and an open review run leave the slot free."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_recovery import unsettled_start_ids
    from ouroboros.outcomes import read_verification_receipts
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.verify import _verify_and_record

    custody._CUSTODY.clear()
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
    (tmp_path / "drive").mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path / "drive"),
    )
    ctx.task_id = "actor-with-review"
    ctx._configured_actor_bootstrap = {
        "route_id": "session-a",
        "work_order_fingerprint": "a" * 64,
        "physical_started": False,
    }
    drive = custody.custody_root(ctx)
    assert custody.record_start_requested(
        drive,
        run_id="",
        task_id=ctx.task_id,
        invocation_id="inv-review",
        idempotency_key="inv-review",
        request={"prompt": "review packet"},
        route="codex",
        source="review_substrate.extraction",
    )
    custody.record_started(drive, custody.RunCustody(
        run_id="run-review", task_id=ctx.task_id, route_id="codex",
        source="review_substrate",
    ))
    custody._CUSTODY.clear()

    assert unsettled_start_ids(drive, ctx.task_id) == {
        "open_run_ids": [],
        "pending_invocation_ids": [],
        "undisposed_patch_run_ids": [],
    }

    written = _verify_and_record(
        ctx,
        contract_kind="delegation_zero_run",
        zero_run_decision="incomplete",
        zero_run_basis="only read-only review runs remain open",
    )

    assert "zero_run_requires_settlement" not in written
    assert [
        row.get("contract_kind")
        for row in read_verification_receipts(drive, ctx.task_id)
    ] == ["delegation_zero_run"]
    custody._CUSTODY.clear()
