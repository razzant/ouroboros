"""Custody and acceptance checks for historical partial-source work orders."""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import pytest
from tests._store_shared import read_result


def _fixture(tmp_path):
    from ouroboros.subagent_work_order import (
        canonical_work_order_source,
        compile_external_work_order,
    )
    from ouroboros.tools.registry import ToolContext

    repo = tmp_path / "repo"
    repo.mkdir()
    contract = {
        "objective": "Build the exact external patch " + ("x" * 250_100),
        "expected_output": "A reviewed patch",
    }
    task = {
        "id": "source-child",
        "objective": contract["objective"],
        "expected_output": contract["expected_output"],
        "task_contract": contract,
        "task_constraint": {},
        "workspace_root": str(repo),
        "workspace_mode": "external_workspace",
    }
    # Historical rows remain readable after new work stopped producing partial
    # source lenses. Build the stored shape directly, without a live size policy.
    rendered = compile_external_work_order(task)
    request = {
        "schema": 1, "kind": "complete_work_order", "coverage": "partial",
        "complete_chars": len(rendered),
        "complete_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "wire_budget_chars": 250_000,
        "source": {
            "kind": "task_result", "task_id": task["id"], "tool": "get_task_result",
            "arguments": {"task_id": task["id"], "include_authority": True,
                          "include_work_order_source": True},
            "projection": "canonical_work_order",
        },
    }
    prompt = "WORK ORDER SOURCE REQUEST\n" + json.dumps(request)
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=tmp_path,
        workspace_root=repo,
        workspace_mode="external_workspace",
        task_id=task["id"],
        task_metadata={
            "task_constraint": {},
            "workspace_root": str(repo),
            "workspace_mode": "external_workspace",
        },
        task_contract=contract,
    )
    full_text, reason = canonical_work_order_source(ctx, request)
    assert not reason
    assert full_text == rendered
    return ctx, request, full_text, prompt


def _started_entry(tmp_path, request, sha):
    from ouroboros import delegate_custody as custody

    custody._CUSTODY.pop("source-run", None)
    entry = custody.RunCustody(
        run_id="source-run",
        task_id="source-child",
        route_id="claude",
        model="claude-fable-5",
        work_order_fingerprint=sha,
        work_order_coverage="partial",
        work_order_source_request=request,
        execution_root=str(tmp_path / "snapshot"),
        target_root=str(tmp_path / "repo"),
        authority_source="external_workspace_root",  # an acting run: its capture may be applied
    )
    assert custody.record_started(tmp_path, entry)
    custody._CUSTODY[entry.run_id] = entry
    return entry


def _source_response(request, text, start, end):
    return {
        "schema": 1,
        "kind": "source_response",
        "complete_sha256": request["complete_sha256"],
        "source": request["source"],
        "start_char": start,
        "end_char": end,
        "text": text,
    }


def test_actor_first_coordination_appendix_preserves_complete_text():
    import ouroboros.tools.delegate as delegate

    authority = SimpleNamespace(delegated=False, access="readonly", investigation=False)
    context = " \n" + "яё𐍈🚀\n" * 55_000 + "DECISIVE_TAIL\n "
    instructions = delegate._host_instructions(authority, coordination_context=context)
    assert instructions.endswith(context)
    assert "git commit" in instructions
    assert "OMISSION NOTE" not in instructions


def test_started_replay_keeps_partial_request_and_verified_ranges(tmp_path):
    from ouroboros import delegate_custody as custody

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    assert custody.replay(tmp_path)[entry.run_id].work_order_source_request == request
    assert custody.work_order_source_verification(entry)["status"] == "cannot_verify"
    assert not custody.record_source_range_verified(
        tmp_path,
        entry,
        start_char=0,
        end_char=len(full_text) + 1,
        complete_sha256=request["complete_sha256"],
        source=request["source"],
        text_sha256="f" * 64,
        text_chars=len(full_text) + 1,
    )
    assert custody.work_order_source_verification(entry)["status"] == "cannot_verify"
    assert custody.emit(tmp_path, custody.SOURCE_RANGE_VERIFIED, {
        "run_id": entry.run_id,
        "task_id": entry.task_id,
        "start_char": 0,
        "end_char": len(full_text) + 1,
        "complete_sha256": request["complete_sha256"],
        "source": request["source"],
        "text_sha256": "f" * 64,
        "text_chars": len(full_text) + 1,
    })
    custody._CUSTODY.clear()
    assert custody.work_order_source_verification(
        custody.replay(tmp_path)[entry.run_id]
    )["status"] == "cannot_verify"
    midpoint = len(full_text) // 2
    assert custody.record_source_range_verified(
        tmp_path,
        entry,
        start_char=0,
        end_char=midpoint,
        complete_sha256=request["complete_sha256"],
        source=request["source"],
        text_sha256="a" * 64,
        text_chars=midpoint,
    )
    assert custody.record_source_range_verified(
        tmp_path,
        entry,
        start_char=midpoint,
        end_char=len(full_text),
        complete_sha256=request["complete_sha256"],
        source=request["source"],
        text_sha256="b" * 64,
        text_chars=len(full_text) - midpoint,
    )
    custody._CUSTODY.clear()
    replayed = custody.replay(tmp_path)[entry.run_id]
    assert custody.work_order_source_verification(replayed)["status"] == "complete"


def test_orphaned_pending_recovery_preserves_partial_source_gate(tmp_path, monkeypatch):
    from ouroboros import delegate_custody as custody

    _ctx, request, _full_text, _prompt = _fixture(tmp_path)
    assert custody.record_start_requested(
        tmp_path,
        run_id="",
        task_id="source-child",
        invocation_id="source-invocation",
        idempotency_key="source-invocation",
        max_seconds=60,
        request={"prompt": "WORK ORDER SOURCE REQUEST", "access": "readonly"},
        project_id="",
        project_owned=False,
        route="claude",
        work_order_fingerprint=request["complete_sha256"],
        work_order_coverage="partial",
        work_order_source_request=request,
    )
    record = custody.pending_invocations(tmp_path)[0]
    monkeypatch.setattr(custody, "_reconcile_one", lambda *_args, **_kwargs: {"ok": True})

    class Gateway:
        def start_run(self, _request, *, idempotency_key=""):
            assert idempotency_key == "source-invocation"
            return {"runId": "source-recovered"}

    assert custody._recover_pending_invocation(tmp_path, Gateway(), record) == {"ok": True}
    recovered = custody.replay(tmp_path)["source-recovered"]
    assert recovered.work_order_coverage == "partial"
    assert recovered.work_order_source_request == request
    assert custody.work_order_source_verification(recovered)["status"] == "cannot_verify"


def test_existing_task_result_reader_returns_the_same_canonical_range(tmp_path):
    from ouroboros.task_results import write_task_result
    from ouroboros.subagent_work_order import canonical_work_order_source
    from ouroboros.tools.control import _get_task_result

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    write_task_result(
        tmp_path,
        "source-child",
        "running",
        objective=ctx.task_contract["objective"],
        expected_output=ctx.task_contract["expected_output"],
        task_contract=dict(ctx.task_contract),
        task_constraint={},
    )
    stored = read_result(tmp_path, "source-child")
    assert "workspace_root" not in stored
    assert "workspace_mode" not in stored
    expected_text, reason = canonical_work_order_source(ctx, request)
    assert not reason
    midpoint = len(full_text) // 2
    payload = json.loads(_get_task_result(
        ctx,
        "source-child",
        include_authority=True,
        include_work_order_source=True,
        source_start_char=0,
        source_end_char=midpoint,
    ))
    source = payload["work_order_source"]
    assert payload["source"] == request["source"]
    assert source["complete_sha256"] == request["complete_sha256"]
    assert source["complete_chars"] == request["complete_chars"]
    assert source["complete_sha256"] == hashlib.sha256(
        expected_text.encode("utf-8")
    ).hexdigest()
    assert source["complete_chars"] == len(expected_text)
    assert source["text"] == expected_text[:midpoint]


def _pending_detail(*interaction_ids, delivery=None):
    """The run detail the engine serves while these questions are pending (legacy rows omit ``delivery``)."""
    return {"summary": {"state": "running", "waitingOnUser": True}, "lastSeq": 1, "pendingInteractions": [
        {"interactionId": iid, "questions": [{"id": "q1", "question": "Send the range."}],
         **({"delivery": delivery} if delivery else {})} for iid in interaction_ids]}


def test_source_answer_is_verified_before_delivery_and_replayed(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    calls = []

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def get_run(self, _run_id, **_kwargs):
            return _pending_detail("interaction-1", "interaction-2")

        def answer_interaction(self, run_id, interaction_id, answers, *, timeout_sec=None):
            calls.append((run_id, interaction_id, answers))
            return {"accepted": True, "status": "delivered"}

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: Gateway())
    midpoint = len(full_text) // 2
    first = _source_response(request, full_text[:midpoint], 0, midpoint)
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-1",
        [{"question_id": "q1", "free_text": "Here is the requested range."}],
        first,
    ).text)
    assert out["status"] == "delivered"
    assert out["work_order_verification"]["status"] == "cannot_verify"
    assert "[SOURCE_RESPONSE]" in calls[0][2][0]["freeText"]
    second = _source_response(request, full_text[midpoint:], midpoint, len(full_text))
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-2",
        [{"question_id": "q1", "free_text": "The remaining range."}],
        second,
    ).text)
    assert out["work_order_verification"]["status"] == "complete"
    custody = __import__("ouroboros.delegate_custody", fromlist=["replay"])
    custody._CUSTODY.clear()
    assert custody.work_order_source_verification(
        custody.replay(tmp_path)[entry.run_id]
    )["status"] == "complete"


def test_source_receipt_retries_after_delivery_when_first_append_fails(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    midpoint = len(full_text) // 2
    response = _source_response(request, full_text[:midpoint], 0, midpoint)
    statuses = iter(("delivered", "already_resolved"))
    append_results = iter((False, True))

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def get_run(self, _run_id, **_kwargs):
            return _pending_detail("interaction-retry")

        def answer_interaction(self, _run_id, _interaction_id, _answers, *, timeout_sec=None):
            status = next(statuses)
            return {"accepted": status == "delivered", "status": status}

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: Gateway())
    monkeypatch.setattr(
        custody,
        "record_source_range_verified",
        lambda *_args, **_kwargs: next(append_results),
    )
    first = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-retry",
        [{"question_id": "q1", "free_text": "range"}], response,
    ).text)
    assert first["status"] == "delivered"
    assert first["work_order_verification"]["status"] == "cannot_verify"
    custody._CUSTODY.clear()
    second = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-retry",
        [{"question_id": "q1", "free_text": "range"}], response,
    ).text)
    assert second["status"] == "already_resolved"
    assert next(append_results, None) is None


def test_already_resolved_without_prior_delivery_keeps_source_unverified(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    midpoint = len(full_text) // 2
    response = _source_response(request, full_text[:midpoint], 0, midpoint)
    calls = []

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def answer_interaction(self, _run_id, _interaction_id, _answers, *, timeout_sec=None):
            calls.append(True)
            return {"accepted": False, "status": "already_resolved"}

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: Gateway())
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-timeout",
        [{"question_id": "q1", "free_text": "range"}], response,
    ).text)
    assert calls == [True]
    assert out["status"] == "already_resolved"
    assert out["work_order_verification"]["status"] == "cannot_verify"


def test_invalid_source_answer_never_posts_to_engine(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    calls = []

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def answer_interaction(self, *_args, **_kwargs):
            calls.append(True)
            return {"accepted": True, "status": "delivered"}

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: Gateway())
    bad = _source_response(request, full_text[:20], 0, 20)
    bad["complete_sha256"] = "0" * 64
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-1", [{"question_id": "q1"}], bad,
    ).text)
    assert out["reason"] == "source_response_invalid"
    assert calls == []
    fractional = _source_response(request, full_text[:20], 0.0, 20)
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-1", [{"question_id": "q1"}], fractional,
    ).text)
    assert out["reason"] == "source_response_invalid"
    assert calls == []


@pytest.mark.parametrize("resolution", ["reject", "complete_then_apply"])
def test_terminal_partial_is_cannot_verify_and_apply_is_refused_but_reject_remains(tmp_path, resolution):
    from ouroboros import delegate_custody as custody
    from ouroboros.headless import ARTIFACT_STATUS_READY_NO_CHANGES
    from ouroboros.tools.delegate import _delivered_terminal_payload
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, request, _full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    detail = {"summary": {"state": "succeeded", "model": "claude-fable-5", "effectiveAccess": "readonly"}, "lastSeq": 1}
    assert custody.settle_run(tmp_path, None, entry, detail)["settled"]
    # Disposition replays custody under its lock. A process-local settled flag
    # is not terminal proof, and rejecting a result needs a usable capture.
    cap_dir = custody.delegated_capture_dir(tmp_path, entry.task_id, entry.run_id)
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / "workspace_patch.json").write_text(json.dumps({"status": ARTIFACT_STATUS_READY_NO_CHANGES}))
    assert custody.record_patch_captured(tmp_path, entry)
    custody._CUSTODY.clear()
    entry = custody.replay(tmp_path)[entry.run_id]
    assert entry.settled and entry.terminal_state == "succeeded"
    terminal = _delivered_terminal_payload(
        ctx,
        entry.run_id,
        detail,
        SimpleNamespace(access="readonly", delegated=False),
        entry,
        None,
    )
    assert terminal["work_order_verification"]["status"] == "cannot_verify"
    assert terminal["acceptance_status"] == "cannot_verify"
    apply_out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
    assert "SOURCE_UNRESOLVED" in apply_out
    assert not custody.replay(tmp_path)[entry.run_id].patch_disposed
    if resolution == "reject":
        reject_out = _integrate_delegated_patch(ctx, entry.run_id, "reject", "not accepted")
        assert "Rejected delegated run" in reject_out
        assert custody.replay(tmp_path)[entry.run_id].patch_disposed == "rejected"
        return
    # Once the durable interval union is complete, the source gate opens and the
    # normal capture/apply guards own the next answer (there is no false permanent
    # refusal just because this was once over budget).
    assert custody.record_source_range_verified(
        tmp_path,
        entry,
        start_char=0,
        end_char=request["complete_chars"],
        complete_sha256=request["complete_sha256"],
        source=request["source"],
        text_sha256="c" * 64,
        text_chars=request["complete_chars"],
    )
    apply_after_complete = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
    assert "no captured file changes to apply" in apply_after_complete
    assert custody.replay(tmp_path)[entry.run_id].patch_disposed == "applied"
    custody._CUSTODY.clear()


@pytest.mark.parametrize("delivery,read", [("async", True), (None, False)], ids=["async-handoff", "unread-row"])
def test_a_source_answer_to_a_question_that_held_nothing_confirms_no_source(tmp_path, monkeypatch, delivery, read):
    """claudexor #628: for an async question ``delivered`` is the engine's handoff to the agent's
    next user message, never its consumption; a row this call could not read proves nothing either.
    Neither records a confirmed delivery, so a later ``already_resolved`` cannot authorize it."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    statuses = iter(("delivered", "already_resolved"))

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def get_run(self, _run_id, **_kwargs):
            if not read:
                raise gateway_module.ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503)
            return _pending_detail("interaction-async", delivery=delivery)

        def answer_interaction(self, _run_id, _interaction_id, _answers, *, timeout_sec=None):
            return {"accepted": True, "status": next(statuses)}

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: Gateway())
    response = _source_response(request, full_text, 0, len(full_text))
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-async", [{"question_id": "q1", "free_text": "all of it"}], response,
    ).text)
    assert out["status"] == "delivered"
    assert out["work_order_verification"] == {"status": "cannot_verify", "reason": "source_delivery_unconfirmed",
                                              "can_authorize": False, "interaction_delivery": delivery or "unknown"}
    assert ("handoff" in out["note"]) is (delivery == "async")
    again = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-async", [{"question_id": "q1", "free_text": "all of it"}], response,
    ).text)
    assert again["status"] == "already_resolved" and again["work_order_verification"]["can_authorize"] is False


@pytest.mark.parametrize("delivery,read,note_key,confirmed", [
    (None, True, "delivered_blocking", True), ("blocking", True, "delivered_blocking", True),
    ("async", True, "delivered_async", False), (None, False, "delivered", False),
], ids=["legacy-row", "blocking", "async", "unread-row"])
def test_a_source_answers_note_and_receipt_rest_on_the_same_row_read(tmp_path, monkeypatch, delivery, read,
                                                                     note_key, confirmed):
    """A verified source answer reads its row once, and both its sentence and its proof come
    from that read: only a question that held the run confirms the range, and an unread row
    gets the plain receipt and no confirmation."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.delegate_interactions import _ANSWER_NOTES
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def get_run(self, _run_id, **_kwargs):
            if not read:
                raise gateway_module.ClaudexorUnavailable("daemon_busy", "local RPC timed out", status_code=503)
            return _pending_detail("interaction-source", delivery=delivery)

        def answer_interaction(self, _run_id, _interaction_id, _answers, *, timeout_sec=None):
            return {"accepted": True, "status": "delivered"}

        def close(self):
            pass

    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: Gateway())
    out = json.loads(delegate._delegate_answer(
        ctx, entry.run_id, "interaction-source", [{"question_id": "q1", "free_text": "all of it"}],
        _source_response(request, full_text, 0, len(full_text)),
    ).text)
    assert out["status"] == "delivered" and out["note"] == _ANSWER_NOTES[note_key]
    assert out["work_order_verification"]["status"] == ("complete" if confirmed else "cannot_verify")
    assert bool(out["work_order_verification"]["can_authorize"]) is confirmed


def _source_answer_on_a_simulated_clock(tmp_path, monkeypatch, *, handshake_sec, read_sec=0.0,
                                        delivery="blocking", read_fails=False, addressed=False):
    """One real ``_delegate_answer`` through the real gateway; every daemon latency is simulated."""
    import httpx

    import ouroboros.delegate_interactions as interactions
    import ouroboros.tools.delegate as delegate
    from ouroboros import delegate_custody as custody
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    before = custody.work_order_source_verification(entry)
    clock = [0.0]
    monkeypatch.setattr(interactions, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    wire = []

    def respond(sent):
        wire.append(sent)
        if sent.method == "POST":
            return httpx.Response(200, json={"accepted": True, "status": "delivered"})
        clock[0] += read_sec
        if read_fails:
            return httpx.Response(503, json={"code": "daemon_busy", "message": "local RPC timed out"})
        return httpx.Response(200, json=_pending_detail("interaction-clocked", delivery=delivery))

    def handshake(**_kwargs):
        clock[0] += handshake_sec
        return {}

    gateway = gateway_module.ClaudexorGateway.__new__(gateway_module.ClaudexorGateway)
    monkeypatch.setattr(gateway, "handshake", handshake)
    monkeypatch.setattr(gateway_module, "ClaudexorGateway", lambda: gateway)
    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://offline.invalid") as client:
        gateway._client = client
        out = json.loads(delegate._delegate_answer(
            ctx, entry.run_id, "interaction-clocked", [{"question_id": "q1", "free_text": "all of it"}],
            _source_response(request, full_text, 0, len(full_text)),
            reply_to={"starter_task_id": entry.task_id, "run_id": entry.run_id,
                      "interaction_id": "interaction-clocked", "question_ids": ["q1"]} if addressed else None,
        ).text)
    custody._CUSTODY.clear()
    return out, wire, clock[0], before, custody.work_order_source_verification(custody.replay(tmp_path)[entry.run_id])


@pytest.mark.parametrize("delivery,read_fails,addressed,note_key,confirmed", [
    ("blocking", False, False, "delivered_blocking", True), ("async", False, False, "delivered_async", False),
    (None, True, False, "delivered", False), ("blocking", False, True, "delivered_blocking", True),
], ids=["blocking", "async", "unread-row", "addressed-reads-once"])
def test_a_source_answers_post_gets_only_what_its_row_read_left(tmp_path, monkeypatch, delivery, read_fails,
                                                                addressed, note_key, confirmed):
    """The row read comes first because only a question that held the run can confirm the range,
    and it spends the same budget: the POST gets the remainder, and the proof still rests on that read."""
    from ouroboros.delegate_interactions import _ANSWER_DEADLINE_SEC, _ANSWER_NOTES

    out, wire, spent, _before, after = _source_answer_on_a_simulated_clock(
        tmp_path, monkeypatch, handshake_sec=30, read_sec=40, delivery=delivery, read_fails=read_fails,
        addressed=addressed)
    assert [sent.method for sent in wire] == ["GET", "POST"]
    post_read = wire[1].extensions["timeout"]["read"]
    assert post_read == 30 and spent + post_read <= _ANSWER_DEADLINE_SEC
    assert out["status"] == "delivered" and out["note"] == _ANSWER_NOTES[note_key]
    assert out["work_order_verification"]["status"] == ("complete" if confirmed else "cannot_verify")
    assert bool(after["can_authorize"]) is confirmed


@pytest.mark.parametrize("handshake_sec,read_sec,reads", [(30, 70, 1), (100, 0, 0)],
                         ids=["row-read-spent-it", "handshake-spent-it"])
def test_a_source_answer_out_of_budget_posts_nothing_and_proves_nothing(tmp_path, monkeypatch, handshake_sec,
                                                                        read_sec, reads):
    """Once the budget is gone: no POST and no further read, the typed outcome of any spent call,
    and the run's source custody exactly as it was."""
    out, wire, _spent, before, after = _source_answer_on_a_simulated_clock(
        tmp_path, monkeypatch, handshake_sec=handshake_sec, read_sec=read_sec)
    assert [sent.method for sent in wire] == ["GET"] * reads
    assert out["status"] == "delivery_unknown" and out["still_pending"] is None
    assert "before the answer POST was sent" in out["transport_error"]
    assert "work_order_verification" not in out
    assert after == before
