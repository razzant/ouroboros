"""Canonical delegation requests stay replayable outside the event row."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import observability
from ouroboros.delegate_pending import request_body
from tests._delegated_transport_shared import (
    _LiveRunStub, _nanny_ctx, _select_transport_access,
    _owned_gateway_uses_each_test_transport,  # noqa: F401 -- autouse fixture
)
from tests._review_session_route_shared import fake_route as _fake_route

fake_route = _fake_route


def _start_rows(root):
    return [row for row in custody._iter_rows(custody.event_log_path(root))
            if row["type"] == custody.START_REQUESTED]


def _digest(body):
    # The engine digests parsed JSON with stable keys, not wire whitespace.
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("surface", ["delegate", "review"])
@pytest.mark.parametrize("failure", [None, "blob", "event"])
def test_canonical_body_and_event_precede_dispatch(tmp_path, monkeypatch, request, surface, failure):
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.review_execution import ReviewRouteUnavailable
    from ouroboros.tools import delegate
    from tests._review_session_route_shared import _run_session_directly

    prompt = "Inspect these exact bytes: Привет\n" + "x" * 50_000
    order, bodies = [], []
    real_write, real_emit = observability.write_blob, custody.emit

    def write(root, body, **kwargs):
        if isinstance(body, dict) and body.get("prompt") == prompt:
            assert order == []
            if failure == "blob":
                raise OSError("fixture refuses body storage")
            bodies.append(copy.deepcopy(body))
            ref = real_write(root, body, **kwargs)
            order.append("blob")
            return ref
        return real_write(root, body, **kwargs)

    def emit(root, kind, payload, **kwargs):
        if kind == custody.START_REQUESTED:
            assert order == ["blob"]
            assert "request" not in payload
            assert request_body(root, payload) == bodies[0]
            if failure == "event":
                return False
            order.append("event")
        return real_emit(root, kind, payload, **kwargs)

    if surface == "review":
        stub_type = request.getfixturevalue("fake_route")
    else:
        stub = _LiveRunStub()
        stub_type = type(stub)
        # A READ-ONLY child: technical access alone decides the run shape (#1609),
        # and this readonly-only daemon serves exactly that request.
        _select_transport_access(monkeypatch, "readonly")
        monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
        monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *a, **k: stub)
    original_start = stub_type.start_run

    def dispatch(self, body, *, idempotency_key=""):
        assert order == ["blob", "event"]
        stored = custody.invocation_record(tmp_path, idempotency_key)
        assert stored["request"] == body == bodies[0]
        assert _digest(stored["request"]) == _digest(body)
        order.append("dispatch")
        return original_start(self, body, idempotency_key=idempotency_key)

    monkeypatch.setattr(observability, "write_blob", write)
    monkeypatch.setattr(custody, "emit", emit)
    monkeypatch.setattr(stub_type, "start_run", dispatch)
    if surface == "review":
        if failure:
            with pytest.raises(ReviewRouteUnavailable) as caught:
                _run_session_directly(tmp_path, prompt=prompt)
            assert caught.value.code == "start_request_row_unwritable"
        else:
            _run_session_directly(tmp_path, prompt=prompt)
    else:
        result = json.loads(delegate._delegate_start(_nanny_ctx(tmp_path), prompt).text)
        assert result["status"] == ("refused" if failure else "started")
        if failure:
            assert result["reason"] == "start_request_row_unwritable"
    if failure:
        assert "dispatch" not in order
        assert _start_rows(tmp_path) == []
    else:
        row = _start_rows(tmp_path)[0]
        assert row["prompt_chars"] == len(prompt)
        assert len(json.dumps(row).encode("utf-8")) < 2_000
        assert order == ["blob", "event", "dispatch"]


@pytest.mark.parametrize("access,geometry", [
    ("readonly", None), ("full", None),
    ("full", {"workspaceKind": "directory", "scopePaths": [], "isolation": "live"}),
])
def test_raw_review_envelope_and_thread_projection_round_trip(tmp_path, access, geometry):
    from ouroboros.review_thread_continuity import start_review_thread_turn
    from ouroboros.review_session_custody import recorded_review_shape

    body = {"prompt": "Привет\n" + "x" * 80_000,
            "instructions": "Authorization: Bearer EXAMPLE-NOT-A-REAL-CREDENTIAL",
            "authPreference": "subscription", "mode": "ask" if access == "readonly" else "agent", "access": access,
            "scope": {"kind": "project", "root": str(tmp_path)},
            "harnesses": ["codex"], "primaryHarness": "codex", "maxSeconds": 300,
            "credentialProfileId": None, "_use_thread": True, "_thread_id": "thread-1",
            "outputSchema": {"type": "object", "properties": {"passed": {"type": "boolean"}}}}
    if geometry is not None:
        body["execution"] = {"delegated": True, "workspaceRoot": str(tmp_path / "copy"), **geometry}
    original = copy.deepcopy(body)
    assert custody.record_start_requested(tmp_path, invocation_id="review", task_id="task", request=body)
    row = _start_rows(tmp_path)[0]
    assert row["request_ref"]["kind"] == "json"
    restored = custody.invocation_record(tmp_path, "review")["request"]
    assert restored == body == original
    assert _digest(restored) == _digest(body)
    assert recorded_review_shape(restored) is not None
    projections = []

    class Gateway:
        def start_thread_turn(self, thread_id, request, *, idempotency_key):
            projections.append((thread_id, request, idempotency_key))
            return {"runId": "run"}

    for envelope in (body, restored):
        start_review_thread_turn(Gateway(), envelope["_thread_id"], envelope, idempotency_key="review")
    assert projections[0] == projections[1]
    projected = projections[0][1]
    assert not {"_thread_id", "_use_thread", "scope"} & projected.keys()
    if geometry is None:
        assert "execution" not in projected, "legacy requests acquire no new geometry during replay"
    else:
        assert projected["execution"] == geometry
    assert body == original


def test_legacy_inline_wins_even_beside_an_unreadable_ref(tmp_path, monkeypatch):
    inline = {"prompt": "legacy request"}
    assert custody.emit(tmp_path, custody.START_REQUESTED, {
        "invocation_id": "legacy", "request": inline, "request_ref": {"path": "missing"}})
    def refuse_read(*args, **kwargs):
        raise AssertionError("legacy inline must not read a blob")
    monkeypatch.setattr(observability, "read_blob_ref", refuse_read)
    assert custody.invocation_record(tmp_path, "legacy")["request"] == inline
    assert custody.pending_invocations(tmp_path)[0]["request"] == inline


@pytest.mark.parametrize("damage", ["missing", "invalid_size", "corrupt"])
def test_unreadable_ref_keeps_request_unknown_and_retry_refuses(tmp_path, damage):
    from ouroboros.tools.delegate_integration import _validated_invocation

    assert custody.record_start_requested(tmp_path, invocation_id="lost", task_id="task",
                                          request={"prompt": "exact original"})
    row = _start_rows(tmp_path)[0]
    if damage == "missing":
        Path(row["request_ref"]["path"]).unlink()
    elif damage == "corrupt":
        Path(row["request_ref"]["path"]).write_bytes(b"not gzip")
    else:  # the stored request row itself carries a malformed reference: the pending obligation
        from ouroboros import delegate_invocation, obligations, store  # and the invocation's addressed fold

        row["request_ref"]["size"] = "invalid"
        obligations.add(tmp_path, "custody_open", "invocation:lost", {"request": row})
        with store.open_store(tmp_path, write=True) as txn:
            record = txn.get_row(delegate_invocation.INVOCATIONS, "lost")
            record["request"]["request_ref"]["size"] = "invalid"
            txn.put_row(delegate_invocation.INVOCATIONS, "lost", record)
    assert custody.invocation_record(tmp_path, "lost")["request"] is None
    pending = custody.pending_invocations(tmp_path)
    assert [row["invocation_id"] for row in pending] == ["lost"]
    assert pending[0]["request"] is None and pending[0]["request_ref"] == row["request_ref"]
    assert request_body(tmp_path, pending[0]) is None
    record, refusal = _validated_invocation(tmp_path, "lost", "task", "exact original")
    assert record is None
    assert json.loads(refusal.text)["reason"] == "invocation_request_unrecorded"


@pytest.mark.parametrize("inline_request", [None, {}, "invalid"])
def test_malformed_legacy_inline_without_a_reference_keeps_existing_behavior(tmp_path, inline_request):
    assert custody.emit(tmp_path, custody.START_REQUESTED, {
        "invocation_id": "legacy-empty", "task_id": "owner", "request": inline_request})
    assert custody.pending_invocations(tmp_path) == []


@pytest.mark.parametrize("damage", [None, "missing", "corrupt"])
def test_lost_ack_keeps_the_original_start_claim_when_its_body_is_unreadable(tmp_path, monkeypatch, damage):
    from ouroboros.delegate_recovery import unsettled_start_ids
    from ouroboros.delegate_terminal import _audit_task_custody
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.tools import delegate

    keys = []

    class LostAcknowledgment(_LiveRunStub):
        def start_run(self, body, *, idempotency_key=""):
            keys.append(idempotency_key)
            if len(keys) == 1:
                raise gateway.ClaudexorUnavailable("daemon_unreachable", "fixture accepted start then lost reply")
            return {"runId": f"run-{len(keys)}"}

    stub = LostAcknowledgment()
    _select_transport_access(monkeypatch, "readonly")   # a read-only child (#1609)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *args, **kwargs: stub)
    ctx = _nanny_ctx(tmp_path, task_id="owner")
    first = json.loads(delegate._delegate_start(ctx, "Inspect fixture files.").text)
    token = first["pending_invocation_id"]
    assert keys == [token]
    blob = Path(_start_rows(tmp_path)[0]["request_ref"]["path"])
    if damage == "missing":
        blob.unlink()
    elif damage == "corrupt":
        blob.write_bytes(b"fixture corrupt gzip")
    assert unsettled_start_ids(tmp_path, "owner")["pending_invocation_ids"] == [token]
    audit = {"task_id": "owner", "audit_status": "ok", "unreconciled": []}
    _audit_task_custody(tmp_path, "owner", audit, emit_evidence=False)
    assert audit["pending_invocation_ids"] == [token]
    assert audit["unreconciled"] == [f"invocation:{token}"]
    fresh = json.loads(delegate._delegate_start(ctx, "Inspect other fixture files.").text)
    assert fresh["status"] == "started" and keys[0] == token and keys[1] != token
    new_token = keys[1]
    retry = json.loads(delegate._delegate_start(ctx, "Inspect fixture files.", retry_of=token).text)
    if damage:
        assert retry["reason"] == "invocation_request_unrecorded" and keys == [token, new_token]
        assert "Start a new run" not in retry["detail"]
    else:
        assert retry["status"] == "started" and keys == [token, new_token, token]


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_missing_pending_body_keeps_snapshot_and_orphan_recovery_without_a_post(tmp_path, damage):
    from types import SimpleNamespace
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation

    assert custody.record_start_requested(
        tmp_path, invocation_id="pending", task_id="owner", snapshot_id="snapshot",
        request={"prompt": "Preserve this pending invocation"})
    blob = Path(_start_rows(tmp_path)[0]["request_ref"]["path"])
    if damage == "missing":
        blob.unlink()
    else:
        blob.write_bytes(b"fixture corrupt gzip")
    pending = custody.pending_invocations(tmp_path)
    assert len(pending) == 1 and pending[0]["request"] is None
    assert custody.open_snapshot_ids(tmp_path) == {"snapshot"}
    gateway = SimpleNamespace(start_run=lambda *args, **kwargs: pytest.fail("unknown body must not POST"))
    result = _recover_pending_invocation(tmp_path, gateway, pending[0])
    assert result["action"] == "invocation_retained"
    assert result["reason"] == "invocation_request_unrecorded"
    assert custody.invocation_record(tmp_path, "pending")["state"] == "pending"
    assert custody.open_snapshot_ids(tmp_path) == {"snapshot"}
    review_result = _recover_pending_invocation(tmp_path, gateway, {**pending[0], "source": "review_substrate"})
    assert review_result["reason"] == "review_panel_owns_invocation"
    assert custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "pending", "definite": True})
    assert custody.pending_invocations(tmp_path) == [] and custody.open_snapshot_ids(tmp_path) == set()



def test_a_pending_body_whose_place_cannot_be_learned_now_is_retained_never_unrecorded(tmp_path, monkeypatch):
    """A stored request body whose bytes cannot be located now (``EvidenceUnavailable``: the packs catalog
    cannot answer) is neither missing nor unrecorded: recovery keeps the invocation pending with its address,
    posts nothing and records no ``invocation_request_unrecorded`` reconciliation; once readable the same
    address gives the exact body. The read names the invocation's task."""
    from types import SimpleNamespace

    from ouroboros import evidence_packs
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation

    assert custody.record_start_requested(
        tmp_path, invocation_id="pending", task_id="owner", snapshot_id="snapshot",
        request={"prompt": "Preserve this pending invocation"})
    ref = _start_rows(tmp_path)[0]["request_ref"]
    real, tasks = evidence_packs.blob_bytes, []

    def unlocatable(root, blob, **kwargs):
        tasks.append(kwargs.get("task_id"))
        if blob.get("sha256") == ref["sha256"]:
            raise evidence_packs.EvidenceUnavailable(
                "evidence_unavailable: the packs catalog cannot answer now", reason="busy")
        return real(root, blob, **kwargs)

    gateway = SimpleNamespace(start_run=lambda *args, **kwargs: pytest.fail("an unread body must not POST"))
    with monkeypatch.context() as busy:
        busy.setattr(evidence_packs, "blob_bytes", unlocatable)
        pending = custody.pending_invocations(tmp_path)
        assert len(pending) == 1 and pending[0]["request"] is None and pending[0]["request_ref"] == ref
        result = _recover_pending_invocation(tmp_path, gateway, pending[0])
    assert result["action"] == "invocation_retained" and result["reason"] == "evidence_unavailable", result
    assert "owner" in tasks, tasks
    reconciled = [row for row in custody._iter_rows(custody.event_log_path(tmp_path))
                  if row["type"] == custody.RECONCILED and row.get("invocation_id") == "pending"]
    assert reconciled == [], reconciled
    assert custody.invocation_record(tmp_path, "pending")["state"] == "pending"
    assert request_body(tmp_path, pending[0]) == {"prompt": "Preserve this pending invocation"}


def test_a_legacy_request_whose_body_cannot_be_located_now_keeps_its_shape_owed(tmp_path):
    """The one-time shape backfill reads an older open request's body for its access and mode. While that body,
    held only by another task's pack, cannot be located (the packs catalog is held), the request keeps no shape and
    the job stays pending, never an empty shape recorded as done; the next run records the real one."""
    import threading

    from ouroboros import delegate_pending, evidence_packs, maintenance_jobs, store

    body = {"prompt": "keep", "mode": "ask", "access": "readonly"}
    assert custody.record_start_requested(tmp_path, invocation_id="legacy", task_id="owner", snapshot_id="snapshot",
                                          source="review_substrate", request=body)
    key = "invocation:legacy"
    with store.open_store(tmp_path, write=True) as txn:  # the shape a row written before request shapes has
        row = txn.get_row("custody_open", key)
        row.pop("pending", None)
        row["request"].pop("request_shape", None)
        txn.put_row("custody_open", key, row)
    ref = row["request"]["request_ref"]
    evidence_packs.write_pack(tmp_path, "other", 1, [ref])
    evidence_packs.ensure_index(tmp_path)
    Path(ref["path"]).unlink()
    held, release = threading.Event(), threading.Event()

    def hold():
        with store.name_lock(tmp_path, evidence_packs.INDEX_LOCK_NAME, timeout_sec=5):
            held.set()
            release.wait(600)

    def state():
        job = maintenance_jobs.job(tmp_path, delegate_pending.JOB) or {}
        with store.open_store(tmp_path, write=False) as txn:
            return job, txn.get_row("custody_open", key)

    holder = threading.Thread(target=hold, name="packer-holds-index")
    holder.start()
    try:
        assert held.wait(5)
        with store.acquisition_budget(0):
            assert delegate_pending.backfill(tmp_path, lambda: False) == {"state": "pending", "items": 0}
    finally:
        release.set()
        holder.join(10)
    job, facts = state()
    assert "pending" not in facts and delegate_pending.backfill_due(tmp_path, job, {}, True), (job, facts)
    assert delegate_pending.backfill(tmp_path, lambda: False) == {"state": "done", "items": 1}
    job, facts = state()
    assert facts["pending"][0]["request_shape"] == {"access": "readonly", "mode": "ask", "execution": False}
    assert not delegate_pending.backfill_due(tmp_path, job, {}, True)


@pytest.mark.parametrize("prior", ["absent", "refused", "fenced", "corrupt"])
def test_skill_review_retries_distinguish_known_pending_from_absent_or_refused(tmp_path, fake_route, prior,
                                                                              monkeypatch):
    """``refused`` is an engine 400 answering the POST: as unknown as a lost reply (#3138), so the retry replays
    its recorded start under the same key. Only a start fenced before its POST (``fenced``) is definitely refused,
    and its retry starts anew, like an ``absent`` token."""
    from ouroboros import review_execution
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.owner_pause import OwnerPauseRefused
    from ouroboros.review_execution import ReviewRouteUnavailable
    from tests._review_session_route_shared import _run_session_directly

    state = {"pending_invocation_id": "absent-token"} if prior == "absent" else {}
    if prior == "fenced":
        real, fences = review_execution.run_operation, []

        def fence_once(*args, **kwargs):
            if not fences:
                fences.append(1)
                raise OwnerPauseRefused("owner_pause_fenced")
            return real(*args, **kwargs)
        monkeypatch.setattr(review_execution, "run_operation", fence_once)
    elif prior != "absent":
        fake_route.start_error = ClaudexorUnavailable(
            "fixture_refused" if prior == "refused" else "daemon_unreachable",
            "fixture initial transport outcome", status_code=400 if prior == "refused" else 0)
    if prior != "absent":
        with pytest.raises((ClaudexorUnavailable, ReviewRouteUnavailable)):
            _run_session_directly(tmp_path, surface="skill_review", slot_id="skill-slot", retry_state=state)
        row = _start_rows(tmp_path)[0]
        token = row["invocation_id"]
        state["pending_invocation_id"] = token
        if prior == "missing":
            Path(row["request_ref"]["path"]).unlink()
        elif prior == "corrupt":
            Path(row["request_ref"]["path"]).write_bytes(b"fixture corrupt gzip")
    else:
        token = state["pending_invocation_id"]
    if prior in {"missing", "corrupt"}:
        with pytest.raises(ReviewRouteUnavailable) as caught:
            _run_session_directly(tmp_path, surface="skill_review", slot_id="skill-slot", retry_state=state)
        assert caught.value.code == "review_recovery_request_missing"
        assert state["pending_invocation_id"] == token
        assert [key for gateway in fake_route.instances for key in gateway.start_keys] == [token]
    else:
        _run_session_directly(tmp_path, surface="skill_review", slot_id="skill-slot", retry_state=state)
        keys = [key for gateway in fake_route.instances for key in gateway.start_keys]
        if prior in ("intact", "refused"):
            assert keys == [token, token]
        else:  # absent: no record; fenced: definitely refused before any POST, so only the new start was sent
            assert keys[-1] != token and len(keys) == 1


def test_pending_scan_only_reads_blobs_for_surviving_invocations(tmp_path, monkeypatch):
    for identity in ("started", "refused", "pending"):
        assert custody.record_start_requested(tmp_path, invocation_id=identity,
                                              request={"prompt": identity})
    assert custody.emit(tmp_path, custody.STARTED, {"invocation_id": "started", "run_id": "run"})
    assert custody.emit(tmp_path, custody.START_FAILED, {"invocation_id": "refused", "definite": True})
    reads, original = [], observability.read_blob_ref
    def read(*args, **kwargs):
        body = original(*args, **kwargs)
        reads.append(body["prompt"])
        return body
    monkeypatch.setattr(observability, "read_blob_ref", read)
    pending = custody.pending_invocations(tmp_path)
    assert [row["invocation_id"] for row in pending] == ["pending"]
    assert reads == []  # current inventory reads typed facts, not canonical bodies
    assert pending[0]["request"] is None and pending[0]["request_ref"]
    assert request_body(tmp_path, pending[0]) == {"prompt": "pending"}
    assert reads == ["pending"]  # only an addressed replay resolves the surviving body


@pytest.mark.parametrize("different", [False, True])
def test_task_event_omits_only_an_equal_contract_mirror_without_mutation(tmp_path, different):
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.utils import sanitize_task_for_event

    task = attach_task_contract({"id": "task", "text": "Inspect the files", "metadata": {"source": "web"}})
    if different:
        task["metadata"]["task_contract"] = {"other": "contract"}
    original = copy.deepcopy(task)
    projected = sanitize_task_for_event(task, tmp_path / "logs")
    assert task == original
    assert projected["task_contract"] == original["task_contract"]
    assert projected["metadata"]["source"] == "web"
    assert ("task_contract" in projected["metadata"]) is different
    if different:
        assert projected["metadata"]["task_contract"] == original["metadata"]["task_contract"]


@pytest.mark.parametrize("start_row", ["written", "billing_refused", "unknown_error", "not_written"])
def test_a_review_start_proven_unsent_spends_no_paid_wave_and_any_other_is_paid_before_its_post(
        tmp_path, fake_route, monkeypatch, start_row):
    """F4 (owner, 8 Nov 2026): the paid stamp follows the durable start row and precedes every POST. A start the
    billing authority refused before any row or POST is proven unsent: no stamp, and the seat says unrequested. A
    start row whose landing is unknown (an error, a False write a replay could still POST) is paid; a POST never
    happens without the stamp before it."""
    from ouroboros.review_execution import ReviewRouteUnavailable, SessionInvocation, run_delegated_review_session
    from ouroboros.usage_accounting import BillingAuthorityUnavailable

    order: list = []

    def stamp():
        order.append("stamp")

    real_record, real_start = custody.record_start_requested, fake_route.start_run

    def record(*args, **kwargs):
        order.append("row")
        if start_row == "billing_refused":
            raise BillingAuthorityUnavailable("fixture: billing authority unreadable before any row")
        if start_row == "unknown_error":
            raise OSError("fixture: the row write's landing is unknown")
        return False if start_row == "not_written" else real_record(*args, **kwargs)

    def post(self, request, **kwargs):
        order.append("post")
        return real_start(self, request, **kwargs)

    monkeypatch.setattr(custody, "record_start_requested", record)
    monkeypatch.setattr(fake_route, "start_run", post)
    invocation = SessionInvocation(task_id="t-paid", surface="scope_review", slot_id="scope_slot_1", timeout_sec=30,
                                   dispatch_stamp=stamp)
    if start_row == "written":
        run_delegated_review_session(invocation=invocation, prompt="review this", root="/tmp/fake-repo",
                                     custody_drive=tmp_path)
        assert order == ["row", "stamp", "post"], "paid before the POST, after the start row"
        return
    with pytest.raises((ReviewRouteUnavailable, OSError)) as caught:
        run_delegated_review_session(invocation=invocation, prompt="review this", root="/tmp/fake-repo",
                                     custody_drive=tmp_path)
    assert "post" not in order
    if start_row == "billing_refused":
        assert order == ["row"] and getattr(caught.value, "delegated_start_unrequested", None) is True
        assert getattr(caught.value, "code", "") == "billing_authority_unavailable"
    else:
        assert order == ["row", "stamp"], "an unknown landing is paid, never refunded"
        assert getattr(caught.value, "delegated_start_unrequested", None) is not True


@pytest.mark.parametrize('configured', [None, '', 'standard'])
@pytest.mark.parametrize('refused', [False, True])
def test_pending_recovery_keeps_row_tier_separate_from_execution(tmp_path, monkeypatch, configured, refused):
    from ouroboros import store
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.subagent_history import record_session_execution
    from ouroboros import subagent_metrics
    from ouroboros.subagent_history import subagent_last_delegation

    class Engine(_LiveRunStub):
        store_identity = 'fixture-store'

        def start_run(self, body, *, idempotency_key=''):
            assert body['processingPreference'] == 'fast'
            assert idempotency_key == 'pending-tier'
            if refused:
                raise ClaudexorUnavailable('fixture_refused', 'not started', status_code=400)
            return {'runId': self.run_id, 'storeId': self.store_identity}

    engine = Engine()
    assert custody.record_start_requested(tmp_path, invocation_id='pending-tier', task_id='owner',
        selected_subagent_id='fixture-row', route='some-route', project_id='prj-existing', project_owned=False,
        handshake_store_id=engine.store_identity, daemon_store_id=engine.store_identity,
        idempotency_key='pending-tier',  # receipt-bound replay; this fixture varies only the row's tier
        request={'prompt': 'Fixture inspection', 'model': 'fixture-model', 'access': 'readonly',
                 'mode': 'ask', 'processingPreference': 'fast'}, processing={'requested': 'fast'},
        **({'row_processing': configured} if configured is not None else {}))
    # Both the addressed store reader and the historical fold start from the real recorded request.
    rows = list(custody.custody_rows(tmp_path))
    for record in (custody.invocation_record(tmp_path, 'pending-tier'),
                   custody.invocation_record(tmp_path, 'pending-tier', rows=rows),
                   custody.pending_invocations(tmp_path)[0],
                   custody.pending_invocations(tmp_path, rows=rows)[0]):
        assert ('row_processing' in record) == (configured is not None)
        assert record.get('row_processing') == configured
        assert record['processing']['requested'] == 'fast'
    # Exercise an already cached pending projection from before the new fields existed.
    with store.open_store(tmp_path, write=True) as txn:
        cached = txn.get_row('custody_open', 'invocation:pending-tier')
        for pending in cached.get('pending', []):
            pending.pop('row_processing', None)
            pending.pop('processing', None)
        txn.put_row('custody_open', 'invocation:pending-tier', cached)
    pending = custody.pending_invocations(tmp_path)[0]
    result = _recover_pending_invocation(tmp_path, engine, pending)
    if refused:  # an answered replay retires nothing (#3138); its row keeps the attempt's facts
        assert (result['action'], result['reason']) == ('invocation_retained', 'start_refusal_unproven')
        event = next(row for row in custody.custody_rows(tmp_path) if row['type'] == custody.START_FAILED)
        assert event.get('row_processing') == configured and event['definite'] is False
        assert event['processing_preference'] == 'fast'
    else:
        recovered = custody.replay(tmp_path)[engine.run_id]
        assert recovered.row_processing == configured
        assert recovered.processing_preference == 'fast'
        record_session_execution(tmp_path, recovered, {'summary': {'state': 'succeeded'}}, {})
    # The real history producer and metrics store retain the configured row identity.
    identity = subagent_last_delegation(tmp_path)['latest_by_subagent']['fixture-row']['identity']
    assert identity['processing_preference'] == ('fast' if configured is None else configured)
    if not refused:
        subagent_metrics.catch_up(tmp_path)
        measured = subagent_metrics.read(tmp_path, 'fixture-row')['rows']
        assert len(measured) == 1 and measured[0]['row_identity']['processing_preference'] == identity['processing_preference']


@pytest.mark.parametrize("own", ["economy", ""])  # a row's own tier; a row without one ('' = its decided default)
@pytest.mark.parametrize("recovery", ["retry", "reconcile", "reconcile_retained"])
def test_a_session_start_carries_its_rows_own_tier_from_the_real_producer_through_a_lost_ack(
        tmp_path, monkeypatch, own, recovery):
    """``_delegate_start`` writes the row's own tier on START_REQUESTED beside the run's (a work's Fast), never into
    the engine request; an explicit retry or positive noncreating sweep lookup keeps it on STARTED.
    A sweep with no accepting-store proof retains the original UNKNOWN attempt and its row's tier."""
    import tests._delegated_transport_shared as shared
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation
    from ouroboros.gateways import claudexor as gateway
    from ouroboros.subagent_history import record_session_execution, subagent_last_delegation
    from ouroboros.tools import delegate

    keys = []

    class LostAcknowledgment(_LiveRunStub):
        store_identity = ""

        def agent_capabilities(self):
            return {"harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                                   "accessProfilesSupported": ["readonly"],
                                   "processingPreferences": ["standard", "fast", "economy"]}]}

        def start_run(self, body, *, idempotency_key=""):
            keys.append((copy.deepcopy(body), idempotency_key))
            if len(keys) == 1:
                raise gateway.ClaudexorUnavailable("daemon_unreachable", "fixture accepted start then lost reply")
            if recovery != "retry":
                pytest.fail("a lost identity-less start cannot be replayed automatically")
            return {"runId": self.run_id}

    real = shared._transport_snapshot
    monkeypatch.setattr(shared, "_transport_snapshot", lambda route, access=None: {
        **real(route, access), **({"processing_preference": own} if own else {}), "work_processing": "fast"})
    stub = LostAcknowledgment()
    _select_transport_access(monkeypatch, "readonly")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setattr(gateway, "ClaudexorGateway", lambda *args, **kwargs: stub)
    ctx = _nanny_ctx(tmp_path, task_id="owner")
    first = json.loads(delegate._delegate_start(ctx, "Inspect fixture files.").text)
    token = first["pending_invocation_id"]
    body = keys[0][0]
    assert body["processingPreference"] == "fast", "the run asks the work's tier"
    assert not {"row_processing", "rowProcessing"} & set(body), "the row's own tier is custody, never a request option"
    requested = _start_rows(tmp_path)[0]
    assert requested["row_processing"] == own and requested["processing"]["requested"] == "fast"
    assert custody.invocation_record(tmp_path, token)["row_processing"] == own
    if recovery == "retry":
        retried = json.loads(delegate._delegate_start(ctx, "Inspect fixture files.", retry_of=token).text)
        assert retried["status"] == "started" and keys[1] == (body, token), "byte-identical replay"
        assert all(row.get("row_processing") == own for row in _start_rows(tmp_path))
    else:
        if recovery == "reconcile":
            stub.operations = lambda: [{"method": "POST", "path": "/v2/runs/lookup"}]
            def lookup(sent, *, idempotency_key):
                assert (sent, idempotency_key) == (body, token)
                return {"status": "found", "storeId": "fixture-store", "handle": {
                    "jobId": "accepted-job", "runId": stub.run_id, "state": "running"}}
            stub.lookup_run = lookup
        result = _recover_pending_invocation(tmp_path, stub, custody.pending_invocations(tmp_path)[0],
                                             may_read=recovery == "reconcile")
        assert len(keys) == 1, "an unknown identity-less start is read, never automatically re-POSTed"
        if recovery == "reconcile_retained":
            assert (result["action"], result["reason"]) == ("invocation_retained", "invocation_store_unproven")
            failed = next(row for row in custody.custody_rows(tmp_path) if row["type"] == custody.START_FAILED
                          and row.get("invocation_id") == token)
            assert failed["row_processing"] == own and failed["processing_preference"] == "fast"
            assert failed["definite"] is False and custody.invocation_record(tmp_path, token)["state"] == "pending"
    if recovery != "reconcile_retained":
        entry = custody.replay(tmp_path)[stub.run_id]
        assert (entry.row_processing, entry.processing_preference) == (own, "fast")
        record_session_execution(tmp_path, entry, {"summary": {"state": "succeeded"}}, {})
    identity = subagent_last_delegation(tmp_path)["latest_by_subagent"]["transport-fixture"]["identity"]
    assert identity["processing_preference"] == own, "filed under the row's own tier, not the work's Fast"
