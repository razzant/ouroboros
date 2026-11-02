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


@pytest.mark.parametrize("inline_request", [None, {}, [], "invalid"])
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


@pytest.mark.parametrize("prior", ["absent", "refused", "intact", "missing", "corrupt"])
def test_skill_review_retries_distinguish_known_pending_from_absent_or_refused(tmp_path, fake_route, prior):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.review_execution import ReviewRouteUnavailable
    from tests._review_session_route_shared import _run_session_directly

    state = {"pending_invocation_id": "absent-token"} if prior == "absent" else {}
    if prior != "absent":
        fake_route.start_error = ClaudexorUnavailable(
            "fixture_refused" if prior == "refused" else "daemon_unreachable",
            "fixture initial transport outcome", status_code=400 if prior == "refused" else 0)
        with pytest.raises(ClaudexorUnavailable):
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
        if prior == "intact":
            assert keys == [token, token]
        else:
            assert keys[-1] != token and len(keys) == (1 if prior == "absent" else 2)


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
