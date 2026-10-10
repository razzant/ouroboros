"""A cumulative continuation patch retains every unresolved source obligation."""

import hashlib
import json
from pathlib import Path

import pytest

from ouroboros import delegate_custody as custody
from tests._delegated_transport_shared import (
    _delegating_ctx,
    _owned_gateway_uses_each_test_transport,  # noqa: F401 -- autouse stub transport
)
from tests.test_delegate_continuation import _settle, _start, _write_result


def _partial_source(tmp_path, ctx, task_id):
    from ouroboros.subagent_work_order import canonical_work_order_source, compile_external_work_order

    task = {
        **ctx.task_metadata, "id": task_id,
        "task_contract": {"objective": f"{task_id}: " + "source material " * 20_000,
                          "expected_output": "A verified patch"},
        "workspace_root": str(ctx.workspace_root), "workspace_mode": ctx.workspace_mode,
        "task_constraint": {},
    }
    _write_result(tmp_path, task_id, **{key: value for key, value in task.items() if key != "id"})
    text = compile_external_work_order(task)
    request = {
        "schema": 1, "kind": "complete_work_order", "coverage": "partial",
        "complete_chars": len(text), "complete_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source": {"kind": "task_result", "task_id": task_id, "tool": "get_task_result",
                   "arguments": {"task_id": task_id, "include_authority": True,
                                 "include_work_order_source": True},
                   "projection": "canonical_work_order"},
    }
    assert canonical_work_order_source(ctx, request) == (text, "")
    return request, text


def _record_range(tmp_path, entry, request, text, start, end):
    assert custody.record_source_range_verified(
        tmp_path, entry, start_char=start, end_char=end,
        complete_sha256=request["complete_sha256"], source=request["source"],
        text_sha256=hashlib.sha256(text[start:end].encode()).hexdigest(), text_chars=end - start)


def _answer_range(ctx, run_id, request, text, start, end):
    from ouroboros.tools import delegate

    return json.loads(delegate._delegate_answer(
        ctx, run_id, f"range-{request['source']['task_id']}-{start}",
        [{"question_id": "source", "free_text": "The requested source range."}],
        {"schema": 1, "kind": "source_response", "complete_sha256": request["complete_sha256"],
         "source": request["source"], "start_char": start, "end_char": end, "text": text[start:end]},
    ).text)


@pytest.mark.parametrize("resolution,new_source,start_path", [
    ("reject", False, "direct"),
    ("apply", False, "direct"),
    ("apply", True, "direct"),
    ("apply", False, "retry"),
    ("apply", True, "recovery"),
    ("apply", True, "chain"),
])
def test_snapshot_continuation_preserves_source_coverage(
        tmp_path, monkeypatch, resolution, new_source, start_path):
    from ouroboros.delegate_custody_reconcile import _recover_pending_invocation
    from ouroboros.gateways import claudexor
    from ouroboros.tools.subagent_integration_delegated import _integrate_delegated_patch

    ctx = _delegating_ctx(tmp_path, acting=True, task_id="t-nanny-write")
    old, old_text = _partial_source(tmp_path, ctx, "old-source")
    new, new_text = _partial_source(tmp_path, ctx, "new-source")
    request, payload, ctx, _ = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-first", prompt="partial legacy brief",
        start_kwargs={"_work_order_source_request": old})
    assert payload["status"] == "started"
    snapshot = Path(request["execution"]["workspaceRoot"])
    (snapshot / "retained.txt").write_text("predecessor work\n", encoding="utf-8")
    pred = custody.replay(tmp_path)["run-first"]
    midpoint = len(old_text) // 2
    _record_range(tmp_path, pred, old, old_text, 0, midpoint)
    _settle(tmp_path, pred.run_id, ctx.task_id)
    options = {"continue_from": pred.run_id}
    if new_source:
        options["_work_order_source_request"] = new
    lost = claudexor.ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    request, payload, ctx, _ = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-next", prompt="",
        start_kwargs=options, start_error=lost if start_path in {"retry", "recovery"} else None)
    if start_path == "retry":
        replayed, payload, ctx, _ = _start(
            tmp_path, monkeypatch, acting=True, run_id="run-next", prompt="",
            start_kwargs={"retry_of": payload["pending_invocation_id"]})
        assert replayed == request and payload["status"] == "started", payload
    elif start_path == "recovery":
        pending, = custody.pending_invocations(tmp_path)

        class RecoveryGateway:
            def start_run(self, body, *, idempotency_key):
                assert body == request and idempotency_key == pending["invocation_id"]
                return {"runId": "run-next"}

            def get_run(self, _run_id):
                return {"summary": {"state": "running"}}

        assert _recover_pending_invocation(tmp_path, RecoveryGateway(), pending)["action"] == "left_live"
    else:
        assert payload["status"] == "started"

    successor_id = "run-next"
    if start_path == "chain":
        _settle(tmp_path, successor_id, ctx.task_id)
        _r, payload, ctx, _ = _start(
            tmp_path, monkeypatch, acting=True, run_id="run-last", prompt="",
            start_kwargs={"continue_from": successor_id})
        assert payload["status"] == "started"
        successor_id = "run-last"
    custody._CUSTODY.clear()
    successor = custody.replay(tmp_path)[successor_id]
    verification = custody.work_order_source_verification(successor)
    assert verification["status"] == "cannot_verify" and verification["can_authorize"] is False
    assert successor.execution_root == str(snapshot)
    # Before any more source is delivered, apply refuses but reject remains available.
    if resolution == "reject":
        _settle(tmp_path, successor.run_id, ctx.task_id)
        refused = _integrate_delegated_patch(ctx, successor.run_id, "apply")
        assert "SOURCE_UNRESOLVED" in refused and snapshot.exists()
        rejected = _integrate_delegated_patch(ctx, successor.run_id, "reject")
        assert "Rejected delegated run" in rejected
        assert custody.replay(tmp_path)[successor.run_id].patch_disposed == "rejected"
        return

    deliveries = []

    class AnswerGateway:
        def handshake(self, **_kwargs): return {}
        def answer_interaction(self, run_id, interaction_id, answers):
            deliveries.append((run_id, interaction_id, answers))
            return {"accepted": True, "status": "delivered"}
        def close(self): pass

    monkeypatch.setattr(claudexor, "ClaudexorGateway", lambda: AnswerGateway())
    if new_source:
        answered = _answer_range(ctx, successor.run_id, new, new_text, 0, len(new_text))
        assert answered["status"] == "delivered"
        assert answered["work_order_verification"]["can_authorize"] is False
    # Deliver only the missing half: the predecessor's verified half must have survived.
    answered = _answer_range(ctx, successor.run_id, old, old_text, midpoint, len(old_text))
    assert answered["status"] == "delivered", answered
    assert answered["work_order_verification"]["status"] == "complete"
    assert len(deliveries) == (2 if new_source else 1)
    custody._CUSTODY.clear()
    assert custody.work_order_source_verification(custody.replay(tmp_path)[successor.run_id])["can_authorize"]
    assert custody.work_order_source_verification(custody.replay(tmp_path)[pred.run_id])["can_authorize"] is False
    _settle(tmp_path, successor.run_id, ctx.task_id)
    applied = _integrate_delegated_patch(ctx, successor.run_id, "apply")
    assert "Integrated delegated run" in applied, applied
    assert Path(successor.target_root, "retained.txt").read_text(encoding="utf-8") == "predecessor work\n"


def test_readonly_continuation_leaves_source_obligation_with_predecessor(tmp_path, monkeypatch):
    ctx = _delegating_ctx(tmp_path, acting=True, task_id="t-nanny-write")
    request, _text = _partial_source(tmp_path, ctx, "old-source")
    _start(tmp_path, monkeypatch, acting=True, run_id="run-first",
           start_kwargs={"_work_order_source_request": request})
    _settle(tmp_path, "run-first", ctx.task_id)
    _body, payload, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-reader", access="readonly",
        start_kwargs={"continue_from": "run-first"})
    assert payload["status"] == "started"
    state = custody.replay(tmp_path)
    assert custody.work_order_source_verification(state["run-reader"])["status"] == "not_required"
    assert custody.work_order_source_verification(state["run-first"])["can_authorize"] is False
