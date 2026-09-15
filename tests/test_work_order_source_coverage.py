"""Custody and acceptance checks for historical partial-source work orders."""

from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace


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
        settled=True,
        execution_root=str(tmp_path / "snapshot"),
        target_root=str(tmp_path / "repo"),
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

    authority = SimpleNamespace(delegated=False, access="readonly")
    context = " \n" + "яё𐍈🚀\n" * 55_000 + "DECISIVE_TAIL\n "
    instructions = delegate._build_start_instructions(authority, coordination_context=context)
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
    from ouroboros.task_results import task_result_path, write_task_result
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
    stored = json.loads(task_result_path(tmp_path, "source-child", create=False).read_text())
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


def test_source_answer_is_verified_before_delivery_and_replayed(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gateway_module

    ctx, request, full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    calls = []

    class Gateway:
        def handshake(self, **_kwargs):
            return {}

        def answer_interaction(self, run_id, interaction_id, answers):
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

        def answer_interaction(self, _run_id, _interaction_id, _answers):
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

        def answer_interaction(self, _run_id, _interaction_id, _answers):
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


def test_terminal_partial_is_cannot_verify_and_apply_is_refused_but_reject_remains(tmp_path):
    from ouroboros import delegate_custody as custody
    from ouroboros.tools.delegate import _delivered_terminal_payload
    from ouroboros.tools.subagent_integration import _integrate_delegated_patch

    ctx, request, _full_text, _prompt = _fixture(tmp_path)
    entry = _started_entry(tmp_path, request, request["complete_sha256"])
    terminal = _delivered_terminal_payload(
        ctx,
        entry.run_id,
        {"summary": {"state": "succeeded", "model": "claude-fable-5", "effectiveAccess": "readonly"}, "lastSeq": 1},
        SimpleNamespace(access="readonly", delegated=False),
        entry,
        None,
    )
    assert terminal["work_order_verification"]["status"] == "cannot_verify"
    assert terminal["acceptance_status"] == "cannot_verify"
    apply_out = _integrate_delegated_patch(ctx, entry.run_id, "apply", "")
    assert "SOURCE_UNRESOLVED" in apply_out
    reject_out = _integrate_delegated_patch(ctx, entry.run_id, "reject", "not accepted")
    assert "SOURCE_UNRESOLVED" not in reject_out
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
    assert "SOURCE_UNRESOLVED" not in apply_after_complete
    custody._CUSTODY.clear()
