"""Real skip/persistence/source/tool consumers; no reviewer transport or paid calls."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import artifacts
from ouroboros.acceptance_history import historical_review_controls, read_acceptance_history
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.contracts.task_contract import build_task_contract
from ouroboros.headless import copy_child_task_result, prepare_task_drive, remove_subagent_task_drive, retry_child_task_refs
from ouroboros.loop_acceptance_review import _observe_host_acceptance_request, _skip_task_acceptance_for_launch_reason
from ouroboros.owner_mailbox import write_owner_message, write_task_message
from ouroboros.owner_quiz import record_answered, record_asked
from ouroboros.project_dialogue import build_owner_message_ref
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.registry import ToolRegistry
from ouroboros.tools.review import get_tools
from ouroboros.utils import append_jsonl
from tests._usage_store_testing import ledger_rows


@pytest.fixture(autouse=True)
def review_policy(monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "auto")
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *_a, **_k: pytest.fail("historical preparation must not send a model request"))


def _fixture(tmp_path, *, split=False, retry=False, child=False, cap="finite", deadline="", accounting_running=False):
    root = tmp_path / "canonical"
    tid, accounting = ("retry", "original") if retry else ("historical", "historical")
    worker = prepare_task_drive(root, tid, "empty") if split else root
    contract = build_task_contract({"id": tid, "expected_output": "Original criterion", "deadline_at": deadline})
    lineage = {"root_task_id": accounting, "delegation_role": "subagent" if child else "root",
               "parent_task_id": accounting if child else None}
    if retry:
        lineage.update(original_task_id=accounting, timeout_retry_from=accounting)
    cap_fields = {} if cap == "unknown" else {"acceptance_original_root_cap": {
        "state": cap, **({"usd": 4.0} if cap == "finite" else {}), "source": "task_admission"}}
    write_task_result(root, accounting, "running", task_contract=contract, chat_id=0,
                      root_task_id=accounting, delegation_role="root", **cap_fields)
    if retry and not accounting_running:
        write_task_result(root, accounting, "failed")
    write_task_result(worker, tid, "running", task_contract=contract, chat_id=0, **lineage)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=worker)
    ctx = registry._ctx
    ctx.task_id, ctx.current_chat_id, ctx.task_attempt = tid, 0, 1
    ctx.task_contract = contract
    ctx.task_metadata = {**lineage, "budget_drive_root": str(root)}
    task = {"id": tid, "chat_id": 0, "task_contract": contract, "task_attempt": 1,
            "_skip_post_task_synthesis": True,
            "drive_root": str(worker), "budget_drive_root": str(root), **lineage}
    trace = {"tool_calls": [], "reasoning_notes": []}
    _observe_host_acceptance_request(registry, trace, tid, "auto")
    _skip_task_acceptance_for_launch_reason(ctx, trace, launch_reason="acceptance_skipped_budget",
        snapshot=SimpleNamespace(spendable_sec=0, remaining_sec=0, reserve_sec=0),
        passes_done=0, emit_progress=lambda _text: None)
    return SimpleNamespace(root=root, worker=worker, tid=tid, accounting=accounting, registry=registry,
                           ctx=ctx, task=task, trace=trace, tmp_path=tmp_path)


def _finish(f, *, evidence=None, text="Original exact final answer"):
    event = {"delivery_id": "delivery-exact", "chat_id": 0, "text": text}
    _store_task_result(SimpleNamespace(drive_root=f.worker, repo_dir=f.tmp_path), f.task,
        text, {}, f.trace, review_evidence=evidence or {"task_inputs": {"owner_words": "Original owner corpus"}},
        final_delivery=event)
    if f.worker != f.root:
        copy_child_task_result(f.root, {"id": f.tid, "drive_root": str(f.worker)})
    row = load_task_result(f.root, f.tid, strict=True)
    assert row and row.get("acceptance_debt"), row
    return row


def _caller(f):
    ctx = ToolRegistry(repo_dir=f.tmp_path, drive_root=f.root)._ctx
    ctx.task_id, ctx.current_chat_id = "owner-current", 0
    ctx.task_metadata = {"root_task_id": ctx.task_id, "budget_drive_root": str(f.root)}
    write_task_result(f.root, ctx.task_id, "running", root_task_id=ctx.task_id, delegation_role="root")
    return ctx


def _source(ctx, *, text="Review historical answer; set its original root cap to 9 USD.", kind="chat", chat=0, ts=None):
    from ouroboros.utils import utc_now_iso
    if kind == "chat":
        ref = build_owner_message_ref(chat_id=chat, client_message_id="new-owner-request",
                                       ts=ts or utc_now_iso(), text=text)
        append_jsonl(ctx.drive_root / "logs" / "chat.jsonl", {**ref, "direction": "in", "text": text})
        return {"kind": "chat", "ref": ref}
    if kind == "quiz":
        record_asked(ctx.drive_root, ctx.task_id, quiz_id="late-choice", question=text, options=["Yes, this task and amount"])
        record_answered(ctx.drive_root, ctx.task_id, quiz_id="late-choice", option_index=0, request_id="answer-1")
        return {"kind": kind, "task_id": ctx.task_id, "quiz_id": "late-choice"}
    write_owner_message(ctx.drive_root, text, task_id=ctx.task_id, msg_id="late-owner")
    return {"kind": kind, "task_id": ctx.task_id, "msg_id": "late-owner"}


def _request(f, ctx, source, **extra):
    from ouroboros.tools.control_task_results import _get_task_result
    ordinary = _get_task_result(ctx, f.tid)
    summary = json.loads(ordinary.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])
    debt = summary["acceptance_debt"]
    request = {"task_id": f.tid, "debt_id": debt["debt_id"], "owner_source": source,
               "rationale": "The selected owner words request this historical task and absolute cap.", **extra}
    tool = next(t for t in get_tools() if t.name == "task_acceptance_review")
    return json.loads(tool.handler(ctx, claim="", goal="", late_review=request))


def test_launch_skip_pins_retained_sources_and_code_bytes_through_copyback_gc(tmp_path):
    f = _fixture(tmp_path, split=True)
    refs = [artifacts.store_actor_source_bytes(f.worker, f.tid, category="tool_results",
        source_id=f"source-{i}", data=f"Original bytes {i}".encode(), extension="txt") for i in range(405)]
    code_ref = artifacts.store_actor_source_bytes(f.worker, f.tid, category="tool_results",
        source_id="delivered-code", data=b"print('original')\n", extension="txt")
    artifacts.store_task_artifact_bytes(f.worker, f.tid, "delivered.py", b"print('original')\n", kind="task_artifact")
    workspace = tmp_path / "current.py"
    workspace.write_text("original workspace")
    f.trace["tool_calls"] = [{"tool": "read_file", "result_source_ref": ref} for ref in refs]
    evidence = {"task_inputs": {"text": "Original owner corpus"},
                "repo_diff_source_ref": code_ref,
                "snapshot": {"sha256": hashlib.sha256(workspace.read_bytes()).hexdigest()}}
    row = _finish(f, evidence=evidence)
    assert row["child_ref_promotion"]["pending_refs"]
    assert not remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False)
    retry_child_task_refs(f.root, f.worker, f.tid)
    assert not load_task_result(f.root, f.tid)["child_ref_promotion"]["pending_refs"]
    debt = copy.deepcopy(row["acceptance_debt"])
    workspace.write_text("new unrelated workspace revision")
    write_task_result(f.root, f.tid, "failed", result="changed mutable result", expected_output="changed criterion",
                      task_contract={"expected_output": "changed"})
    from ouroboros.task_custody import settle_child_drive
    assert remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False), settle_child_drive(
        f.root, f.tid, f.worker, live=lambda _task: False)
    assert not f.worker.exists()
    current = load_task_result(f.root, f.tid)
    raw = artifacts.read_task_result_source_bytes(f.root, current, Path(debt["source_ref"]["path"]).name,
                                                  debt["source_ref"]["path"])
    frozen = json.loads(raw)
    assert frozen["answer"] == "Original exact final answer"
    assert frozen["effective_criteria"]["task_contract"]["expected_output"] == "Original criterion"
    assert frozen["owner_corpus"]["text"] == "Original owner corpus"
    captured = {item["location"]: item["source_ref"] for item in frozen["sources"]}
    for i in range(405):
        assert artifacts.read_actor_source_bytes(f.root, f.tid, captured[f"trace.tool_calls[{i}].result_source_ref"]) == f"Original bytes {i}".encode()
    assert artifacts.read_actor_source_bytes(f.root, f.tid, captured["evidence.repo_diff_source_ref"]) == b"print('original')\n"
    assert {"section": "workspace", "reason": "not_a_filesystem_snapshot"} in frozen["gaps"]
    assert debt["delivery_status"] == "unconfirmed"
    ctx = _caller(f)
    prepared = _request(f, ctx, _source(ctx))
    assert prepared["status"] == "owed", prepared
    assert "historical_subject" not in prepared and prepared["dispatched"] is False
    assert prepared["source_ref"]["sha256"] == debt["source_ref"]["sha256"]


def test_forced_round_rail_connects_to_final_result_capture(tmp_path, monkeypatch):
    from tests.test_delivery_forced_finalization import _forced_test_context
    loop, registry, limit, trace = _forced_test_context(tmp_path)
    # External model transport only; the forced rail, seed, final persistence,
    # and source reader are production consumers.
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (
        {"role": "assistant", "content": "Forced final answer"}, 0.0))
    text, usage, trace = loop._handle_round_limit(limit)
    assert trace["acceptance_history_seed"]["cause"] == "acceptance_bypassed_round_limit"
    task = {"id": "parent1", "chat_id": 7, "root_task_id": "parent1", "delegation_role": "root"}
    _store_task_result(SimpleNamespace(drive_root=tmp_path, repo_dir=tmp_path), task, text, usage, trace,
                      final_delivery={"delivery_id": "forced-delivery", "chat_id": 7, "text": text})
    row = load_task_result(tmp_path, "parent1")
    assert read_acceptance_history(tmp_path, "parent1", row["acceptance_debt"])["answer"] == text


@pytest.mark.parametrize("retry,child,expected", [(False, False, True), (True, False, True), (False, True, False)])
def test_root_retry_and_child_eligibility(tmp_path, retry, child, expected):
    f = _fixture(tmp_path, retry=retry, child=child)
    assert bool(f.trace.get("acceptance_history_seed")) is expected
    if expected:
        row = _finish(f)
        assert row["acceptance_debt"]["task_id"] == f.tid
        assert row["acceptance_debt"]["accounting_root_task_id"] == f.accounting


@pytest.mark.parametrize("cap", ["finite", "unlimited", "unknown"])
@pytest.mark.parametrize("retry", [False, True])
def test_historical_caps_never_fall_back_to_current_settings(tmp_path, monkeypatch, cap, retry):
    from ouroboros.usage_accounting import UsageScope, usage_scope
    monkeypatch.setenv("OUROBOROS_PER_TASK_COST_USD", "999")
    with usage_scope(UsageScope(drive_root=tmp_path / "canonical",
            task_id="retry" if retry else "historical", root_task_id="original" if retry else "historical",
            root_limit_usd=None)):
        f = _fixture(tmp_path, retry=retry, cap=cap)
    _finish(f)
    ctx = _caller(f)
    out = _request(f, ctx, _source(ctx))
    assert out["effective_original_root_cap"]["state"] == cap, out
    if cap == "finite":
        assert out["effective_original_root_cap"]["usd"] == 4
    assert not out["dispatched"]


@pytest.mark.parametrize("kind", ["chat", "quiz", "mailbox"])
@pytest.mark.parametrize("retry", [False, True])
def test_owner_cap_amendment_is_absolute_root_bound_idempotent_and_replica_safe(tmp_path, kind, retry):
    f = _fixture(tmp_path, split=True, retry=retry)
    row = _finish(f)
    write_task_result(f.root, f.accounting, load_task_result(f.root, f.accounting)["status"],
                      reserved_usd=1.25, accounted_upper_bound_usd=3.5,
                      budget_pause={"state": "settled", "marker": "keep"})
    original = load_task_result(f.root, f.accounting)
    ctx = _caller(f)
    source = _source(ctx, kind=kind)
    out = _request(f, ctx, source, new_original_root_cap_usd=9)
    # A bare cap (the pre-selector form) is money only: no review is prepared.
    assert (out["status"], out["action"], out["dispatched"]) == ("amended", "amend_cap", False), out
    assert not load_task_result(f.root, f.tid).get("review_operations")
    assert out["owner_source_ref"]["size"] > 0
    alias = copy.deepcopy(source)
    if kind == "chat":
        alias["ref"]["client_message_id"] = ""
        alias["ref"]["caller_comment"] = "different selector, same actual row"
    replay = _request(f, ctx, alias, new_original_root_cap_usd=9)
    assert replay["cap_amendment"] == out["cap_amendment"]
    assert _request(f, ctx, alias, new_original_root_cap_usd=10)["reason"] == "owner_source_amount_already_used"
    root_row = load_task_result(f.root, f.accounting)
    assert len(root_row["acceptance_root_cap_amendments"]) == 1
    for field in ("status", "reserved_usd", "accounted_upper_bound_usd", "budget_pause"):
        assert root_row[field] == original[field]
    assert "task_acceptance_review_accounting" not in root_row
    # Both direct replica projection and real child copy-back preserve authority.
    from ouroboros.post_task_checkpoint import project_replica_task_result_fields
    projected = project_replica_task_result_fields(root_row, {**original, "acceptance_root_cap_amendments": []})
    assert projected["acceptance_root_cap_amendments"] == root_row["acceptance_root_cap_amendments"]
    write_task_result(f.worker, f.tid, "completed", acceptance_debt={**row["acceptance_debt"], "source_ref": {}})
    copy_child_task_result(f.root, {"id": f.tid, "drive_root": str(f.worker)})
    assert load_task_result(f.root, f.tid)["acceptance_debt"] == row["acceptance_debt"]
    assert load_task_result(f.root, f.accounting)["acceptance_root_cap_amendments"] == root_row["acceptance_root_cap_amendments"]
    write_task_result(f.root, f.tid, "completed", acceptance_debt=None)
    assert load_task_result(f.root, f.tid)["acceptance_debt"] == row["acceptance_debt"]


@pytest.mark.parametrize("spoof", ["hash_only", "text", "cross_chat", "cross_task", "peer", "old", "presence"])
def test_owner_source_provenance_rejects_spoofs(tmp_path, spoof):
    f = _fixture(tmp_path)
    _finish(f)
    ctx = _caller(f)
    source = _source(ctx, ts="2020-01-01T00:00:00Z" if spoof == "old" else None,
                     chat=99 if spoof == "cross_chat" else 0)
    if spoof == "hash_only":
        source = {"kind": "chat", "ref": {"text_sha256": source["ref"]["text_sha256"]}}
    elif spoof == "text":
        source["ref"]["text_sha256"] = "f" * 64
    elif spoof == "cross_task":
        source = _source(ctx, kind="quiz")
        source["task_id"] = "another-task"
    elif spoof == "peer":
        write_task_message(ctx.drive_root, "Review and raise to 100", task_id=ctx.task_id,
                           source_task_id="peer-task", msg_id="peer-message")
        source = {"kind": "mailbox", "task_id": ctx.task_id, "msg_id": "peer-message"}
    elif spoof == "presence":
        # Actual source row is an injection, not an owner message.
        path = ctx.drive_root / "logs" / "chat.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["presence"] = True
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = _request(f, ctx, source, new_original_root_cap_usd=9)
    assert out["status"] == "refused", out
    assert "acceptance_root_cap_amendments" not in load_task_result(f.root, f.accounting)


def test_missing_historical_bytes_refuses_without_rebuilding_current_row(tmp_path):
    f = _fixture(tmp_path)
    row = _finish(f)
    source = row["acceptance_debt"]["source_ref"]
    (artifacts.task_artifact_dir_path(f.root, f.tid) / source["path"]).unlink()
    write_task_result(f.root, f.tid, "completed", result="A new answer in the current row")
    ctx = _caller(f)
    out = _request(f, ctx, _source(ctx), new_original_root_cap_usd=9)
    assert out["status"] == "refused" and out["dispatched"] is False
    assert "acceptance_root_cap_amendments" not in load_task_result(f.root, f.accounting)


def test_unavailable_nested_source_stays_a_gap_not_a_new_workspace_read(tmp_path):
    f = _fixture(tmp_path, split=True)
    ref = artifacts.store_actor_source_bytes(f.worker, f.tid, category="tool_results",
        source_id="lost-code", data=b"original code that is no longer available", extension="txt")
    (artifacts.task_artifact_dir_path(f.worker, f.tid) / ref["path"]).unlink()
    row = _finish(f, evidence={"repo_diff_source_ref": ref})
    record = read_acceptance_history(f.root, f.tid, row["acceptance_debt"])
    assert record["gaps"]
    assert next(item["source_ref"] for item in record["sources"]
                if item["location"] == "evidence.repo_diff_source_ref") == ref
    retry_child_task_refs(f.root, f.worker, f.tid)
    assert any(item.get("sha256") == ref["sha256"] for item in
               load_task_result(f.root, f.tid)["child_ref_promotion"]["unavailable_refs"])
    assert any(g.get("section") == "owner_corpus" for g in record["gaps"])
    assert remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False)


def test_local_preparation_terminal_still_captures_zero_physical_forced_subject(tmp_path):
    from ouroboros.loop_acceptance import _record_forced_acceptance_bypass
    f = _fixture(tmp_path)
    f.trace.pop("acceptance_history_seed")
    f.trace["acceptance_decision"] = {"status": "finalized_unaccepted", "origin": "local_acceptance_preparation"}
    _record_forced_acceptance_bypass(SimpleNamespace(tools=f.registry, task_id=f.tid,
        accumulated_usage={"reason_code": "round_limit"}), f.trace, "round_limit")
    assert _finish(f)["acceptance_debt"]["cause"] == "acceptance_bypassed_round_limit"


def test_real_terminal_emitter_freezes_owner_corpus_and_outbox_identity_without_delivery(tmp_path):
    from ouroboros.agent import Env
    from ouroboros.agent_task_pipeline import emit_task_results
    from ouroboros.loop_messages import _record_owner_directive
    f = _fixture(tmp_path)
    f.task.update(workspace_root=str(tmp_path), workspace_mode="external", text="Original owner task", chat_id=42)
    f.ctx.current_chat_id = 42
    _record_owner_directive(f.ctx, source="initial_user", content="Retain this exact owner requirement.", msg_id="initial-owner")
    events = []
    emit_task_results(Env(repo_dir=tmp_path, drive_root=f.root), None, None,
        events, f.task, "Exact emitted answer", {"rounds": 1}, f.trace, 0.0, f.root / "logs", ctx=f.ctx)
    row = load_task_result(f.root, f.tid)
    debt = row["acceptance_debt"]
    send = next(e for e in events if e["type"] == "send_message")
    assert debt["delivery"]["delivery_id"] == send["delivery_id"]
    assert debt["delivery"]["chat_id"] == 42 and debt["delivery_status"] == "unconfirmed"
    record = read_acceptance_history(f.root, f.tid, debt)
    assert "Retain this exact owner requirement." in json.dumps(record["owner_corpus"])
    assert record["answer"] == send["text"] == "Exact emitted answer"
    assert "task_acceptance_review_accounting" not in row


def test_outbound_body_is_frozen_separately_and_missing_delivery_id_is_a_gap(tmp_path):
    f = _fixture(tmp_path)
    _store_task_result(SimpleNamespace(drive_root=f.worker, repo_dir=tmp_path), f.task,
        "Raw solve result", {}, f.trace, final_delivery={"chat_id": 0, "text": "Host-rendered final body"})
    row = load_task_result(f.root, f.tid)
    record = read_acceptance_history(f.root, f.tid, row["acceptance_debt"])
    assert row["result"] == record["result_text"] == "Raw solve result"
    assert record["answer"] == "Host-rendered final body"
    assert record["delivery"]["delivery_id"] == ""
    assert {"section": "delivery", "reason": "final_delivery_identity_unavailable"} in record["gaps"]
    assert any(g.get("reason") == "unretained_external_sources_unavailable" for g in record["gaps"])


def test_claim_after_source_capture_blocks_explicit_new_panel_preparation(tmp_path):
    from ouroboros.review_substrate import build_review_binding
    from ouroboros.task_results import claim_task_acceptance_review_cycle
    f = _fixture(tmp_path)
    _finish(f)
    claim = claim_task_acceptance_review_cycle(f.root, f.accounting,
        build_review_binding(candidate="a", evidence={}, fence_token_or_state="f"), claimed_by_task_id=f.tid)
    assert claim["status"] == "claimed"
    before = load_task_result(f.root, f.accounting)["task_acceptance_review_accounting"]
    ctx = _caller(f)
    out = _request(f, ctx, _source(ctx), action="review", new_original_root_cap_usd=9)
    assert "paid_or_unknown_panel" in out["execution_blocked_by"] and not out["dispatched"]
    after = load_task_result(f.root, f.accounting)
    assert after["task_acceptance_review_accounting"] == before
    assert after["acceptance_root_cap_amendments"][0]["new_cap_usd"] == 9


@pytest.mark.parametrize("state", ["pending", "unknown", "completed", "claimed", "published", "legacy"])
def test_existing_physical_or_claimed_panel_never_produces_a_new_subject(tmp_path, state):
    f = _fixture(tmp_path)
    if state == "claimed":
        from ouroboros.review_substrate import build_review_binding
        from ouroboros.task_results import claim_task_acceptance_review_cycle
        binding = build_review_binding(candidate="original candidate", evidence={}, fence_token_or_state="original fence")
        claim_task_acceptance_review_cycle(f.root, f.accounting, review_binding=binding, claimed_by_task_id=f.tid)
    elif state == "published":
        write_task_result(f.root, f.accounting, "running", review_projection={"panels": [{
            "panel_id": "old", "surface": "task_acceptance", "authority": "host_root",
            "actors": [{"operation_state": "unknown"}]}]})
    elif state == "legacy":
        write_task_result(f.root, f.accounting, "running", review_status={"run_count": 1})
    else:
        f.trace["review_runs"] = [{"authority": "host_root", "actors": [{"operation_state": state}]}]
    _store_task_result(SimpleNamespace(drive_root=f.worker, repo_dir=tmp_path), f.task, "answer", {}, f.trace,
        final_delivery={"delivery_id": "d", "chat_id": 0, "text": "answer"})
    assert not load_task_result(f.root, f.tid).get("acceptance_debt")


@pytest.mark.parametrize("control", ["pause", "stop", "panic", "root_fence", "deadline", "review_prohibited"])
def test_explicit_request_never_clears_owner_controls(tmp_path, control):
    f = _fixture(tmp_path, deadline="2000-01-01T00:00:00Z" if control == "deadline" else "")
    _finish(f)
    if control == "pause":
        write_task_result(f.root, f.tid, "completed", budget_pause={"state": "paused"})
    elif control == "stop":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(f.root, f.tid, allow_settled_target=True)
    elif control == "panic":
        (f.root / "state").mkdir(exist_ok=True)
        (f.root / "state" / "panic_stop.flag").write_text("panic")
    elif control == "root_fence":
        (f.root / "state").mkdir(exist_ok=True)
        (f.root / "state" / "queue_snapshot.json").write_text(json.dumps({
            "budget_root_fences": [{"root_task_id": f.accounting, "status": "active"}]}))
    elif control == "deadline":
        # Today's mutable result must not erase the original owner deadline.
        write_task_result(f.root, f.tid, "completed", task_contract={})
    elif control == "review_prohibited":
        write_task_result(f.root, f.tid, "completed", task_constraint={"allow_review": False})
    ctx = _caller(f)
    out = _request(f, ctx, _source(ctx), new_original_root_cap_usd=9)
    assert out["execution_blocked_by"] and not out["dispatched"], out
    assert load_task_result(f.root, f.accounting)["acceptance_root_cap_amendments"][0]["new_cap_usd"] == 9


@pytest.mark.parametrize("cause", ["owner_hurry", "acceptance_bypassed_owner_requested_finalization"])
def test_new_explicit_request_can_prepare_hurry_wrapup_history_but_automatic_cannot(tmp_path, cause):
    f = _fixture(tmp_path)
    f.trace["acceptance_history_seed"]["cause"] = cause
    row = _finish(f)
    assert "owner_finalization" in historical_review_controls(f.root, row, row["acceptance_debt"],
                                                               caller_task_id="", automatic=True)
    ctx = _caller(f)
    out = _request(f, ctx, _source(ctx))
    assert out["status"] == "owed" and not out["dispatched"], out
    assert load_task_result(f.root, f.tid)["acceptance_debt"]["cause"] == cause


def test_terminal_boundary_never_reads_artifact_bytes_or_promotes_bulk_sources(tmp_path, monkeypatch):
    f = _fixture(tmp_path, split=True)
    artifact = artifacts.store_task_artifact_bytes(f.worker, f.tid, "large.bin", b"x" * 2_000_000)
    ref = artifacts.store_actor_source_bytes(f.worker, f.tid, category="tool_results",
        source_id="already-captured", data=b"retained evidence" * 100_000, extension="txt")
    artifact_path = artifacts.task_artifact_dir_path(f.worker, f.tid) / artifact["path"]
    ref_path = artifacts.task_artifact_dir_path(f.worker, f.tid) / ref["path"]
    read_bytes, path_open = Path.read_bytes, Path.open

    def guarded_read(path):
        assert path not in (artifact_path, ref_path), "terminal capture opened bulk bytes"
        return read_bytes(path)

    def guarded_open(path, *args, **kwargs):
        assert path not in (artifact_path, ref_path), "terminal capture streamed bulk bytes"
        return path_open(path, *args, **kwargs)

    f.trace["tool_calls"] = [{"tool": "read_file", "result_source_ref": ref}]
    with monkeypatch.context() as urgent:
        urgent.setattr(Path, "read_bytes", guarded_read)
        urgent.setattr(Path, "open", guarded_open)
        urgent.setattr(artifacts, "stream_artifact_file", lambda *_a, **_k: pytest.fail("terminal artifact hashing"))
        urgent.setattr(artifacts, "copy_artifact_file", lambda *_a, **_k: pytest.fail("terminal artifact copy"))
        urgent.setattr("ouroboros.observability.promote_child_task_refs",
                       lambda *_a, **_k: pytest.fail("terminal recursive bulk promotion"))
        _store_task_result(SimpleNamespace(drive_root=f.worker, repo_dir=tmp_path), f.task,
            "Urgent final", {}, f.trace, review_evidence={},
            final_delivery={"delivery_id": "urgent", "chat_id": 0, "text": "Urgent final"})
    row = load_task_result(f.worker, f.tid)
    assert row["acceptance_debt"]["source_ref"]["size"] < 15_000
    assert not (artifacts.task_artifact_dir_path(f.root, f.tid) / ref["path"]).exists()
    # Ordinary copyback/GC, outside the urgent answer boundary, owns the bytes.
    copy_child_task_result(f.root, {"id": f.tid, "drive_root": str(f.worker)})
    retry_child_task_refs(f.root, f.worker, f.tid)
    assert remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False)
    canonical = load_task_result(f.root, f.tid)
    record = read_acceptance_history(f.root, f.tid, canonical["acceptance_debt"])
    retained = next(item["source_ref"] for item in record["sources"] if item["location"] == "trace.tool_calls[0].result_source_ref")
    assert artifacts.read_actor_source_bytes(f.root, f.tid, retained) == b"retained evidence" * 100_000
    assert not list(f.root.rglob("historical-artifact-*"))


def test_mutable_artifact_is_a_historical_gap_even_if_current_bytes_match(tmp_path):
    f = _fixture(tmp_path)
    artifact_dir = artifacts.task_artifact_dir_path(f.worker, f.tid, create=True)
    path = artifact_dir / "mutable.py"
    path.write_text("original bytes")
    row = _finish(f)
    record = read_acceptance_history(f.root, f.tid, row["acceptance_debt"])
    assert record["artifact_manifests"]
    assert any(gap["reason"] == "manifest_is_not_historical_bytes" for gap in record["gaps"])
    path.write_text("new bytes")
    ctx = _caller(f)
    assert _request(f, ctx, _source(ctx))["status"] == "owed"
    assert read_acceptance_history(f.root, f.tid, row["acceptance_debt"]) == record


@pytest.mark.parametrize("relayed", [False, True])
def test_main_owner_can_name_project_history_using_current_or_host_relayed_source(tmp_path, relayed):
    f = _fixture(tmp_path)
    _finish(f)
    ctx = _caller(f)
    ctx.current_chat_id = 1  # Main; the historical target belongs to another room.
    source = _source(ctx, chat=42 if relayed else 1,
        text=f"Review the answer of task {f.tid} in the Project; raise its original root cap to 9 USD.")
    if relayed:
        write_task_result(f.root, ctx.task_id, "running", origin_message_ref=source["ref"])
    out = _request(f, ctx, source, new_original_root_cap_usd=9)
    assert out["status"] == "amended" and out["dispatched"] is False, out
    amendment = load_task_result(f.root, f.accounting)["acceptance_root_cap_amendments"][0]
    assert amendment["source"]["ref"] == source["ref"]
    assert amendment["debt_id"] == out["debt_id"] and amendment["accounting_root_task_id"] == f.accounting


def test_target_room_owner_source_outside_caller_is_not_authority(tmp_path):
    f = _fixture(tmp_path)
    _finish(f)
    ctx = _caller(f)
    ctx.current_chat_id = 1
    # A real source in the TARGET's room grants no membership to the caller.
    out = _request(f, ctx, _source(ctx, chat=0), new_original_root_cap_usd=9)
    assert out["reason"] == "owner_source_unavailable" and not out["dispatched"]
    assert "acceptance_root_cap_amendments" not in load_task_result(f.root, f.accounting)


@pytest.mark.parametrize("actor", ["child", "presence"])
def test_restricted_actor_cannot_reuse_real_owner_source(tmp_path, actor):
    f = _fixture(tmp_path)
    _finish(f)
    ctx = _caller(f)
    source = _source(ctx)
    if actor == "child":
        ctx.task_metadata.update(parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    else:
        from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_payload
        from ouroboros.presence_capabilities import PresenceProfileResolution
        from ouroboros.presence_runtime import ResolvedPresenceRuntime

        resolution = PresenceProfileResolution(active=(), missing_required=(), missing_optional=(), orphaned=(),
            runtime=ResolvedPresenceRuntime("main", 10, 10, False), profile_fingerprint="a" * 64,
            selection_fingerprint="b" * 64, required_selections_present=True)
        ceiling = build_presence_capability_ceiling(skill_name="presence", skill_content_hash="c" * 64,
            state_fingerprint="d" * 64, resolution=resolution)
        ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    out = _request(f, ctx, source, new_original_root_cap_usd=9)
    assert out["reason"] == "owner_caller_required" and not out["dispatched"]
    assert "acceptance_root_cap_amendments" not in load_task_result(f.root, f.accounting)


def test_huge_history_returns_compact_refs_and_real_bounded_reader_recovers_all_bytes(tmp_path):
    from ouroboros.acceptance_history import historical_source_reference
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.control_task_results import _get_task_result

    f = _fixture(tmp_path, split=True)
    # A trace producer already retained the full bytes, including a blind tail.
    raw = ("full trace row " + "z" * 140 + "\n") * 3_000 + "BLIND-HISTORICAL-TAIL\n"
    ref = artifacts.store_actor_source_bytes(f.worker, f.tid, category="tool_results",
        source_id="acceptance_tool_trajectory", data=raw.encode(), extension="txt")
    f.trace["tool_calls"] = [{"tool": "read_file", "result": raw, "result_source_ref": ref}]
    refs = [artifacts.store_actor_source_bytes(f.worker, f.tid, category="tool_results",
        source_id=f"ref-{i}", data=f"marker-{i}".encode(), extension="txt") for i in range(405)]
    f.trace["tool_calls"].extend({"tool": "read_file", "result_source_ref": item} for item in refs)
    _finish(f, evidence={"tool_trajectory_source_ref": ref})
    retry_child_task_refs(f.root, f.worker, f.tid)
    assert remove_subagent_task_drive(f.root, f.tid, live=lambda _task: False)
    ctx = _caller(f)
    ordinary = _get_task_result(ctx, f.tid)
    assert "acceptance_debt" in ordinary and len(ordinary) < 20_000
    # Even a long owner message/rationale is retained behind a typed reference.
    out = _request(f, ctx, _source(ctx, text="Review historical task. " * 10_000))
    assert len(json.dumps(out)) < 6_000 and "historical_subject" not in out
    assert "BLIND-HISTORICAL-TAIL" not in json.dumps(out)

    def read_all(source):
        arguments = dict(source["reader"]["arguments"])
        if source["reader"]["tool"] == "get_task_result":
            meta = json.loads(_get_task_result(ctx, **arguments))["review_source"]
            chunks = [json.loads(_get_task_result(ctx, **arguments,
                source_start_char=start, source_end_char=min(start + 8_000, meta["complete_chars"])
            ))["review_source"]["text"] for start in range(0, meta["complete_chars"], 8_000)]
            body = "".join(chunks).encode()
            assert len(body) == source["size"]
            assert hashlib.sha256(body).hexdigest() == source["sha256"]
            return body
        start, chunks = 1, []
        while True:
            rendered = _read_file(ctx, **{**arguments, "start_line": start, "max_lines": 40})
            extent = ctx.last_read_view
            assert extent and extent["body_chars"] < 20_000
            chunks.append(rendered[extent["body_start"]:extent["body_start"] + extent["body_chars"]])
            if extent["end_line"] == extent["total_lines"]:
                break
            start = extent["end_line"] + 1
        body = "".join(chunks).encode()
        assert len(body) == source["size"]
        assert hashlib.sha256(body).hexdigest() == source["sha256"]
        return body

    history = json.loads(read_all(out["source_ref"]))
    assert "trajectory" not in history and len(history["sources"]) >= 406
    selected = next(item["source_ref"] for item in history["sources"]
                    if item["location"] == "evidence.tool_trajectory_source_ref")
    assert read_all(historical_source_reference(f.root, f.tid, selected)) == raw.encode()


@pytest.mark.serial
def test_real_budget_pause_cap_amendment_preserves_all_controls_and_ledger(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, reserve_attempt, mark_dispatched, settle_attempt
    from supervisor.events import _handle_budget_pause
    from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx, _supervisor_ctx

    f = _fixture(tmp_path, retry=True, accounting_running=True)
    _finish(f)
    queue, _state, workers = _install_queue(f.root, monkeypatch)
    spent = reserve_attempt(AttemptRequest(model="synthetic", provider="test", drive_root=f.root,
        task_id=f.accounting, root_task_id=f.accounting, reservation_usd=2, global_limit_usd=100, root_limit_usd=4))
    mark_dispatched(spent)
    settle_attempt(spent, cost_usd=2, cost_final=True)
    hold = reserve_attempt(AttemptRequest(model="synthetic", provider="test", drive_root=f.root,
        task_id=f.accounting, root_task_id=f.accounting, reservation_usd=2, global_limit_usd=100, root_limit_usd=4))
    mark_dispatched(hold)
    settle_attempt(hold, cost_usd=2, cost_final=True)  # known spend reaches the $4 cap (#1487)
    with pytest.raises(BudgetExceeded) as exhausted:
        reserve_attempt(AttemptRequest(model="synthetic", provider="test", drive_root=f.root,
            task_id=f.accounting, root_task_id=f.accounting, reservation_usd=0.1, global_limit_usd=100, root_limit_usd=4))
    pause_ctx, limit = _loop_ctx(f.root, f.accounting)
    try:
        with pytest.raises(budget_pause.BudgetPauseRequested) as caught:
            budget_pause.request_pause(limit, rail=budget_pause.RAIL_DISPATCH_REFUSED,
                                       scope="root", reason_text=str(exhausted.value), root_task_id=f.accounting)
        task = {"id": f.accounting, "type": "task", "chat_id": 0, "root_task_id": f.accounting, "_attempt": 1}
        worker = SimpleNamespace(busy_task_id=f.accounting)
        workers.RUNNING[f.accounting] = {"task": task, "worker_id": 0, "attempt": 1}
        workers.WORKERS[0] = worker
        persisted, pushed = [], []
        supervisor = _supervisor_ctx(f.root, workers, queue, persisted, pushed)
        _handle_budget_pause({**budget_pause.pause_event(task, caught.value.pause), "worker_id": 0}, supervisor)
        assert budget_pause.budget_pause_row(f.root, f.accounting)["state"] == "paused"
        assert workers.PENDING[0]["_budget_pause"]["exact_continuation"]
        # Persist the real queue fence alongside its in-process owner projection.
        snapshot = f.root / "state" / "queue_snapshot.json"
        snapshot.write_text(json.dumps({"budget_root_fences": list(queue.BUDGET_ROOT_FENCES.values()),
                                       "pending": workers.PENDING}))
        old_snapshot = snapshot.read_bytes()
        old_queue = copy.deepcopy((workers.PENDING, workers.RUNNING, queue.BUDGET_ROOT_FENCES))
        old_row = load_task_result(f.root, f.accounting)
        old_ledger = ledger_rows(f.root)
        ctx = _caller(f)
        out = _request(f, ctx, _source(ctx), new_original_root_cap_usd=9)
        assert out["cap_amendment"]["new_cap_usd"] == 9 and not out["dispatched"], out
        assert "pause:" + f.accounting in out["execution_blocked_by"]
        assert "root_budget_fence" in out["execution_blocked_by"]
        new_row = load_task_result(f.root, f.accounting)
        assert {k: v for k, v in new_row.items() if k not in ("acceptance_root_cap_amendments", "updated_at")} == {
            k: v for k, v in old_row.items() if k != "updated_at"}
        assert (workers.PENDING, workers.RUNNING, queue.BUDGET_ROOT_FENCES) == old_queue
        assert snapshot.read_bytes() == old_snapshot
        assert ledger_rows(f.root) == old_ledger
        assert budget_pause.dispatch_fenced(f.accounting) and worker.busy_task_id is None
        assert hold.attempt_id in {row["attempt_id"] for row in old_ledger}
    finally:
        budget_pause.end_dispatch_fence(f.accounting)
