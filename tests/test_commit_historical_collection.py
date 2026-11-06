"""Real commit consumers collect historical seats without today's Full roster."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody, review_custody
from ouroboros.observability import persist_call, read_call_payload
from ouroboros.review_dispatch import review_operation_binding, review_reconciliation_identity
from ouroboros.review_execution import ReviewRouteKind
from ouroboros.review_records import ReviewActorRecord, ReviewRequest, ReviewSlot
from ouroboros.review_substrate import run_review_request as _RUN_REVIEW_REQUEST
from ouroboros.tools.parallel_review import run_parallel_review
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests._review_session_route_shared import (  # noqa: F401 - offline transport fixtures
    FakeGateway, _owned_gateway_uses_each_test_transport, _terminal_detail, fake_route,
)


_TASK = "historical-commit"
_RETRY = "commit_review:original-material"
_CONTRACT = "original-contract"


def _seat(drive, repo, surface, slot_id, *, complete=True, custody=True, evidence=None):
    slot = ReviewSlot(slot_id=slot_id, model="old-review=old-model", route=ReviewRouteKind.AGENT_SESSION,
                      session_target="old-review=old-model", session_profile="old-account", effort="high",
                      max_tokens=65536, default_temperature=0.2, role_hint="multi-model review", use_local=False)
    request = ReviewRequest(surface=surface, task_id=_TASK, goal="original review", retry_key=_RETRY,
                            call_type=surface, session_root=str(repo), session_task="original paid instructions",
                            evidence=dict(evidence or {}))
    request.reconciliation_identity = review_reconciliation_identity(
        request, [slot], root_task_id=_TASK, contract=_CONTRACT,
    )
    operation, token = f"op-{slot_id}", f"inv-{slot_id}"
    binding = review_operation_binding(request, slot, operation)
    stored_slot = asdict(slot)
    stored_slot.pop("session_access")  # a genuine pre-rights slot meant readonly
    prompt = {"request": asdict(request), "slot": stored_slot}
    persist_call(drive, task_id=_TASK, call_id=f"{operation}_prompt", call_type=f"{surface}_prompt",
                 payload=prompt, manifest={"review_operation_binding": binding})
    if custody:
        delegate_custody.record_start_requested(
            drive, task_id=_TASK, root_task_id=_TASK, surface=surface, slot_id=slot_id,
            operation_id=operation, invocation_id=token, idempotency_key=f"key-{slot_id}",
            run_id="", project_id="old-project", project_owned=False, max_seconds=60, route="old-review",
            request={"prompt": "original paid instructions", "mode": "ask", "access": "readonly",
                     "harnesses": ["old-review"], "scope": {"kind": "project", "root": str(repo)}, "maxSeconds": 60},
        )
    row = {"slot_id": slot_id, "model_id": slot.model, "route": "agent_session", "status": "error",
           "operation_id": operation, "operation_state": "in_flight", "late_result_pending": True,
           "pending_invocation_id": token, "recovery_binding": {**binding, "pending_invocation_id": token}}
    if complete:
        raw = "[]"
        actor = ReviewActorRecord(slot_id=slot_id, model=slot.model, status="ok", raw_text=raw,
                                  operation_id=operation, recovery_binding=row["recovery_binding"])
        persist_call(drive, task_id=_TASK, call_id=f"{operation}_response", call_type=f"{surface}_response",
                     payload={"producer_outcome": asdict(actor), "message": {"content": raw},
                              "usage": {"invocation_id": token}},
                     manifest={"producer_complete": True, "review_operation_binding": row["recovery_binding"]})
    return row, prompt


@pytest.fixture
def historical_commit(tmp_path, monkeypatch):
    repo, drive = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    ready, prompt = _seat(drive, repo, "multi_model_review", "old-triad")
    unknown, _ = _seat(drive, repo, "multi_model_review", "old-unknown", complete=False, custody=False)
    scope, _ = _seat(drive, repo, "multi_model_review", "old-peer")
    attempt = SimpleNamespace(task_id=_TASK, review_retry_key=_RETRY, review_contract_fingerprint=_CONTRACT,
                              triad_raw_results=[ready, unknown, scope], scope_raw_result={})
    ctx = SimpleNamespace(repo_dir=repo, drive_root=drive, task_id=_TASK, task_metadata={}, pending_events=[],
                          _pending_review_attempt=attempt, _review_reconcile_only=True, _review_iteration_count=0,
                          _current_review_retry_key=_RETRY, _review_history=[], _review_advisory=[],
                          _review_degraded_reasons=[])
    set_review_pool(monkeypatch, pool_roster(pool_seat("new-review", "new-review=new-model", kind="agent_session")))
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    assert commit_triad_delivery()["session_accesses"] == ["full"]
    review_custody._ACTIVE.clear()
    review_custody._NO_RESEND.clear()

    def no_dispatch(*args, **kwargs):
        pytest.fail("collection must neither assemble today's reviewer roster nor dispatch")

    monkeypatch.setattr("ouroboros.reviewer_slot_config.commit_triad_delivery", no_dispatch)
    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", no_dispatch)
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", no_dispatch)
    monkeypatch.setattr("ouroboros.claudexor_daemon.read_owned_gateway", no_dispatch)
    return ctx, ready, unknown, scope, prompt


@pytest.mark.parametrize('child_drive', [False, True])
def test_query_model_collects_pre_rights_cas_under_current_full_slot(historical_commit, monkeypatch, child_drive):
    """Isolate the real rights mismatch before any commit packet/preflight work."""
    import asyncio
    from ouroboros.tools.review_multi_model import _query_model

    ctx, ready, _unknown, _scope, _prompt = historical_commit
    if child_drive:
        ctx.budget_drive_root = ctx.drive_root
        ctx.drive_root = ctx.drive_root / 'child'
        ctx.drive_root.mkdir()
    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _RUN_REVIEW_REQUEST)
    review_custody.prepare_frozen_review_reconciliation(ctx, ctx._pending_review_attempt)

    async def collect():
        return await _query_model(
            object(), "old-review=old-model", [], asyncio.Semaphore(1), ctx,
            slot_id="old-triad", route=ReviewRouteKind.AGENT_SESSION,
            session_target="old-review=old-model", session_profile="old-account",
            session_task="rebuilt instructions", session_root=str(ctx.repo_dir),
            session_access="full", retry_key=_RETRY, effort="high", use_local=False,
        )

    _model, payload, _headers = asyncio.run(collect())
    assert not payload.get("error"), payload
    assert payload["choices"][0]["message"]["content"] == "[]"
    assert payload["operation_id"] == ready["operation_id"]
    assert payload["recovery_binding"] == ready["recovery_binding"]


def test_commit_collects_old_readonly_cas_and_keeps_unknown_seat_under_new_full_roster(historical_commit):
    ctx, ready, unknown, scope, _prompt = historical_commit
    before = read_call_payload(ctx.drive_root, task_id=_TASK, call_id="op-old-triad_prompt")
    _error, scope_result, _reason, _advisory = run_parallel_review(ctx, "collect the original review")
    triad = {row["slot_id"]: row for row in ctx._last_triad_raw_results}
    assert set(triad) == {"old-triad", "old-unknown", "old-peer"}
    assert triad["old-triad"]["status"] == "responded"
    assert triad["old-triad"]["raw_text"] == "[]"
    for source in (ready, unknown):
        collected = triad[source["slot_id"]]
        assert collected["operation_id"] == source["operation_id"]
        assert collected["recovery_binding"] == source["recovery_binding"]
    assert triad["old-triad"]["pending_invocation_id"] == ""
    assert triad["old-unknown"]["pending_invocation_id"] == unknown["pending_invocation_id"]
    assert triad["old-unknown"]["operation_state"] == "custody_lost"
    assert triad["old-unknown"]["late_result_pending"] is True
    assert triad["old-peer"]["status"] == "responded"
    assert triad["old-peer"]["operation_id"] == scope["operation_id"]
    assert read_call_payload(ctx.drive_root, task_id=_TASK, call_id="op-old-triad_prompt") == before


@pytest.mark.parametrize("damage", ["foreign_task", "foreign_roster", "missing_prompt"])
def test_commit_collection_rejects_foreign_or_missing_original_source(historical_commit, damage):
    ctx, ready, _unknown, _scope, prompt = historical_commit
    if damage == "missing_prompt":
        ready["operation_id"] = "operation-with-no-prompt"
    elif damage == "wrong_token":
        ready["pending_invocation_id"] = "foreign-token"
    else:
        if damage == "foreign_task":
            prompt["request"]["task_id"] = "another-task"
        elif damage == "foreign_slot":
            prompt["slot"]["slot_id"] = "another-slot"
        else:
            prompt["request"]["reconciliation_identity"]["roster_hash"] = "foreign-roster"
        manifest, _payload, _refs = read_call_payload(ctx.drive_root, task_id=_TASK, call_id="op-old-triad_prompt")
        persist_call(ctx.drive_root, task_id=_TASK, call_id="op-old-triad_prompt", call_type="multi_model_review_prompt",
                     payload=prompt, manifest={"review_operation_binding": manifest["review_operation_binding"]})
    run_parallel_review(ctx, "collect only")
    triad = {row["slot_id"]: row for row in ctx._last_triad_raw_results}
    rejected = triad["old-triad"]
    assert rejected["status"] != "responded"
    assert rejected["operation_state"] == "custody_lost" and rejected["late_result_pending"] is True
    assert rejected["operation_id"] == ready["operation_id"]
    assert rejected["pending_invocation_id"] == ready["pending_invocation_id"]
    assert triad["old-peer"]["status"] == "responded"  # independent paid answer survives


@pytest.mark.parametrize('child_drive', [False, True])
def test_commit_pending_start_replays_original_token_request_and_rights(historical_commit, monkeypatch, child_drive):
    ctx, _ready, unknown, _scope, _prompt = historical_commit
    FakeGateway.reset()
    FakeGateway.catalog_entry["id"] = "old-review"
    FakeGateway.detail = _terminal_detail("[]", conformance="passed", model="old-model")
    monkeypatch.setattr("ouroboros.claudexor_daemon.ensure_owned_gateway", lambda: FakeGateway())
    monkeypatch.setattr("ouroboros.review_substrate.run_review_request", _RUN_REVIEW_REQUEST)
    raw = {"prompt": "ORIGINAL task password=preserve-me", "instructions": "original instructions",
           "mode": "ask", "access": "readonly", "harnesses": ["old-review"], "primaryHarness": "old-review",
           "scope": {"kind": "project", "root": str(ctx.repo_dir)}, "maxSeconds": 60,
           "authPreference": "subscription"}
    delegate_custody.record_start_requested(
        ctx.drive_root, task_id=_TASK, root_task_id=_TASK, surface="multi_model_review", slot_id="old-unknown",
        operation_id=unknown["operation_id"], invocation_id=unknown["pending_invocation_id"], idempotency_key="original-key",
        run_id="", project_id="old-project", project_owned=False, max_seconds=60, route="old-review", request=raw,
    )
    if child_drive:
        ctx.budget_drive_root = ctx.drive_root
        ctx.drive_root = ctx.drive_root / 'child'
        ctx.drive_root.mkdir()
    run_parallel_review(ctx, "collect only")
    starts = [(key, body) for gateway in FakeGateway.instances for key, body in zip(gateway.start_keys, gateway.start_requests)]
    assert starts == [(unknown["pending_invocation_id"], raw)]
    collected = {row["slot_id"]: row for row in ctx._last_triad_raw_results}["old-unknown"]
    assert collected["operation_id"] == unknown["operation_id"]
    assert collected["recovery_binding"]["pending_invocation_id"] == unknown["pending_invocation_id"]
    assert collected["status"] == "responded"
    assert not any(gateway.cancels for gateway in FakeGateway.instances)


def test_new_explicit_commit_review_uses_new_full_roster_and_new_operation(tmp_path, fake_route, monkeypatch):  # noqa: F811
    import asyncio
    from ouroboros.review_checking import freeze_checking_source
    from ouroboros.reviewer_slot_config import commit_triad_delivery
    from ouroboros.tools.review_multi_model import _query_model

    repo, drive = tmp_path / "new-repo", tmp_path / "new-data"
    repo.mkdir()
    (repo / "source.py").write_text("SOURCE = 'new review'\n")
    set_review_pool(monkeypatch, pool_roster(pool_seat("new-explicit", "fake-review=fake-small", kind="agent_session")))
    plan = commit_triad_delivery()
    FakeGateway.catalog_entry["accessProfilesSupported"] = ["readonly", "workspace_write", "full"]
    FakeGateway.detail = _terminal_detail("[]", conformance="passed")
    original = FakeGateway.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.engine_version = "3.22.0"
        self.ensure_full_access = lambda root: {"allowFullAccess": True}

    monkeypatch.setattr(FakeGateway, "__init__", init)
    ctx = SimpleNamespace(task_id="new-commit", drive_root=drive, repo_dir=repo, task_metadata={}, pending_events=[])
    source = freeze_checking_source(repo, data_root=drive)

    async def send():
        return await _query_model(
            object(), plan["models"][0], [], asyncio.Semaphore(1), ctx,
            slot_id=plan["slot_ids"][0], route=plan["routes"][0],
            session_target=plan["session_targets"][0], session_access=plan["session_accesses"][0],
            session_task="review new material", session_root=str(repo), retry_key="commit_review:new-material",
            checking_source=source,
        )

    _model, payload, _headers = asyncio.run(send())
    assert not payload.get("error"), payload
    requests = [body for gateway in FakeGateway.instances for body in gateway.start_requests]
    assert len(requests) == 1 and requests[0]["access"] == "full"
    assert requests[0]["execution"]["workspaceRoot"] != str(repo)
    assert payload["slot_id"] == "new-explicit"
    assert payload["operation_id"] and payload["operation_id"] != "op-old-triad"
