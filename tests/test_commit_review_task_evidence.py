"""Selected commit evidence survives compaction and each reviewer delivery."""
from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import shutil
import subprocess
import time
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.observability import persist_call, promote_child_task_refs, read_blob_ref
from ouroboros.outcomes import collect_trace_refs
from ouroboros.review_evidence import (
    _ACCEPT_NOTES_CAP,
    capture_commit_review_evidence,
    commit_review_evidence_section,
    materialize_commit_review_session_view,
    pending_commit_review_evidence,
    release_commit_review_session_view,
)
from ouroboros.tools.registry import ToolContext
from tests._workspace_executor_shared import _init_repo

pytestmark = pytest.mark.serial


@pytest.fixture
def evidence_context(tmp_path):
    repo, child, canonical = (tmp_path / name for name in ("repo", "child", "canonical"))
    _init_repo(repo)
    (repo / ".gitignore").write_text("/.review-drive/\n")
    child.mkdir()
    canonical.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=child, budget_drive_root=str(canonical), task_id="evidence-task")
    ctx._execution_trace = {"tool_calls": [], "reasoning_notes": []}
    ctx._accumulated_usage = {"llm_call_refs": []}
    return ctx


def model_response(ctx, name, content, *, execution="solve", call_type="llm_response"):
    trace = persist_call(ctx.drive_root, task_id=ctx.task_id, call_id=name + "_response",
                         call_type=call_type, payload={"message": {"content": content}},
                         manifest={"execution_id": execution, "llm_call_id": name})
    row = {"llm_call_id": name, "execution_id": execution, "response_ref": trace["manifest_ref"]}
    ctx._accumulated_usage["llm_call_refs"].append(row)
    return row


def tool_response(ctx, name, parent, *, tool="view_image", result="image attached", args=None):
    args = args or {"path": "/tmp/view.png"}
    trace = persist_call(ctx.drive_root, task_id=ctx.task_id, call_id=name,
                         call_type="tool_call", payload={"tool": tool, "tool_call_id": name,
                         "args": args, "result": result, "parent_call_id": parent,
                         "execution_id": "solve", "round_id": "round-1", "semantic_ok": True,
                         "result_meta": {"status": "ok"}})
    row = {"tool": tool, "tool_call_id": name, "args": args, "result": result, "trace_ref": trace}
    ctx._execution_trace["tool_calls"].append(row)
    return row


def test_large_task_selected_source_is_complete_and_native_readable(evidence_context):
    ctx = evidence_context
    model_response(ctx, "before", "Inspect the screenshot")
    image = tool_response(ctx, "image", "before")
    assessment = "Visible assessment\n" + ("The button aligns with the field.\n" * 900) + "ASSESSMENT_END"
    model_response(ctx, "after", [{"type": "thinking", "text": "PRIVATE_THINKING"}, {"type": "text", "text": assessment}])
    source = capture_commit_review_evidence(ctx)
    exact = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"]).decode()
    assert "ASSESSMENT_END" in exact
    assert "PRIVATE_THINKING" not in exact
    assert source["source_complete"] is True
    assert source["selected_count"] == 1
    for delivery in ("native", "session", "packet"):
        assert len(commit_review_evidence_section(source, delivery=delivery)) <= _ACCEPT_NOTES_CAP
    assert "evidence_delivery=partial" in commit_review_evidence_section(source, delivery="packet")
    assert "host-retained provenance" in commit_review_evidence_section(source, delivery="packet")
    ctx._execution_trace["reasoning_notes"] = ["unrelated" * 300000]
    ctx._execution_trace["tool_calls"].extend([{"tool": "read_file", "result": "unrelated" * 2000}] * 100)
    ctx.messages = []  # Compaction cannot remove the completed tool trace.
    ctx._tool_trace_refs = {}
    again = capture_commit_review_evidence(ctx)
    assert again["source_ref"] == source["source_ref"]
    assert again["unselected_count"] == 100
    assert len(commit_review_evidence_section(again, delivery="native")) <= _ACCEPT_NOTES_CAP
    image["trace_ref"]["call_id"] = "changed-after-freeze"
    assert source["original_refs"][0]["call_id"] == "image"
    from ouroboros.review_native_episode import inspection_registry
    registry, native_ctx, _ = inspection_registry(str(ctx.repo_dir), ctx.budget_drive_root, ctx.task_id)
    result = registry.execute_result("read_file", {"root": "artifact_store", "path": source["source_ref"]["path"], "max_lines": 12})
    assert result.status == "ok", result.text
    assert "Selected browser/vision execution sources" in result.text
    assert native_ctx.last_read_view["opened_root"] == "artifact_store"


@pytest.mark.parametrize("recoverable", [True, False], ids=["projection_recovers", "source_unavailable"])
def test_partial_visual_result_completeness_follows_recovery(evidence_context, recoverable):
    from ouroboros import artifacts

    ctx = evidence_context
    full = "Full observed result\n" + "x" * 20000 + "\nDECISIVE_END"
    model_response(ctx, "before", "Inspect the screen")
    projection = full if recoverable else {"partial": "legacy body unavailable"}
    call = tool_response(ctx, "visual", "before", result=projection)
    _, primary_ref, issue = artifacts.persist_exact_text_source(
        ctx.drive_root, ctx.task_id, source_id="visual", text=full,
    )
    assert not issue
    logged = full[:100]
    call.update(result=logged, result_partial=True, result_source_ref=primary_ref)
    (artifacts.task_artifact_dir_path(ctx.drive_root, ctx.task_id) / primary_ref["path"]).unlink()
    model_response(ctx, "after", "Recorded visible assessment")

    packet = capture_commit_review_evidence(ctx)
    raw = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, packet["source_ref"]).decode("utf-8")
    selected = json.loads(raw.split("\n\n", 2)[2])

    assert selected["result_complete"] is recoverable
    assert selected["result"] == (full if recoverable else logged)
    assert selected["following_visible_text"] == "Recorded visible assessment"
    assert packet["source_status"] == "ready"
    assert packet["source_complete"] is recoverable
    assert packet["gap_count"] == (0 if recoverable else 1)


@pytest.mark.parametrize("shape", ["missing", "empty", "failed", "other_execution", "tampered"])
def test_following_response_gap_never_selects_a_later_success(evidence_context, shape):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "image", "before")
    if shape != "missing":
        row = model_response(ctx, "following", "" if shape == "empty" else "IMMEDIATE_TEXT",
                             execution="other" if shape == "other_execution" else "solve",
                             call_type="llm_error" if shape == "failed" else "llm_response")
        if shape == "tampered":
            pathlib.Path(row["response_ref"]["path"]).write_text("{}")
    if shape not in {"missing", "other_execution"}:
        model_response(ctx, "later", "LATER_SUCCESS_MUST_NOT_BE_SELECTED")
    model_response(ctx, "foreign", "FOREIGN_SUCCESS", execution="other")
    source = capture_commit_review_evidence(ctx)
    exact = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"]).decode()
    assert "LATER_SUCCESS_MUST_NOT_BE_SELECTED" not in exact
    assert "FOREIGN_SUCCESS" not in exact
    if shape == "empty":
        assert '"following_visible_text_status": "no_visible_text"' in exact
    else:
        assert "following_response_gap" in exact
        assert not source["source_complete"]


def test_multiple_tools_share_one_following_response(evidence_context):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before", tool="browse_page")
    tool_response(ctx, "two", "before")
    model_response(ctx, "after", "ONE_SHARED_ASSESSMENT")
    source = capture_commit_review_evidence(ctx)
    raw = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"]).decode()
    assert raw.count("ONE_SHARED_ASSESSMENT") == 1
    assert source["selected_count"] == 2
    assert "evidence_delivery=complete_selected" in commit_review_evidence_section(source, delivery="packet")


def test_session_view_is_identical_ignored_rematerializable_and_disposable(evidence_context):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "Exact assessment")
    source = capture_commit_review_evidence(ctx)
    view = materialize_commit_review_session_view(source, ctx.repo_dir)
    assert view["session_source_status"] == "ready"
    path = pathlib.Path(view["session_path"])
    exact = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"])
    assert path.read_bytes() == exact
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ctx.repo_dir, check=True, capture_output=True, text=True)
    assert ".review-drive" not in status.stdout
    path.unlink()
    # The same recorded source (a pending rejoin re-reads it from the frozen
    # request through pending_commit_review_evidence) materializes the identical
    # bytes again; no separate preflight-view restorer exists any more.
    import ouroboros.review_evidence as evidence_module
    assert not hasattr(evidence_module, "restore_commit_review_evidence")
    materialize_commit_review_session_view(source, ctx.repo_dir)
    assert path.read_bytes() == exact
    release_commit_review_session_view(view)
    assert not path.exists()
    assert read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"]) == exact


def test_unignored_session_root_keeps_a_partial_exhibit_without_widening_policy(evidence_context):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "An assessment")
    source = capture_commit_review_evidence(ctx)
    (ctx.repo_dir / ".gitignore").unlink()
    view = materialize_commit_review_session_view(source, ctx.repo_dir)
    assert view["session_source_status"] == "unavailable"
    assert not (ctx.repo_dir / ".review-drive").exists()
    assert "retrieval is unavailable" in commit_review_evidence_section(view, delivery="session")


def test_original_responses_survive_child_cleanup_without_new_tool_metadata(evidence_context):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "RETAINED_ORIGINAL")
    source = capture_commit_review_evidence(ctx)
    refs = collect_trace_refs(ctx._accumulated_usage, ctx._execution_trace)
    result, promotion = promote_child_task_refs(pathlib.Path(ctx.budget_drive_root), ctx.drive_root, ctx.task_id, {"trace_refs": refs})
    assert promotion["status"] == "complete"
    shutil.rmtree(ctx.drive_root)
    ref = result["trace_refs"]["llm_call_refs"][-1]["response_ref"]
    manifest = json.loads(pathlib.Path(ref["path"]).read_text())
    payload = read_blob_ref(pathlib.Path(ctx.budget_drive_root), manifest["redacted_projection_ref"])
    assert payload["message"]["content"] == "RETAINED_ORIGINAL"
    assert b"RETAINED_ORIGINAL" in read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, source["source_ref"])
    assert "review_evidence_refs" not in ctx._execution_trace["tool_calls"][0]


def test_pending_reconciliation_uses_the_recorded_source(evidence_context):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "Frozen assessment")
    frozen = capture_commit_review_evidence(ctx)
    prompt = persist_call(ctx.drive_root, task_id=ctx.task_id, call_id="review", call_type="review_prompt",
                          payload={"request": {"task_id": ctx.task_id, "evidence": {"task_execution": frozen}}})
    ctx._pending_review_attempt = SimpleNamespace(triad_raw_results=[{"prompt_ref": prompt}], scope_raw_result={})
    ctx._execution_trace["tool_calls"] = []
    assert pending_commit_review_evidence(ctx) == frozen


@pytest.mark.parametrize("delivery", ["packet", "native", "session"])
def test_triad_request_preserves_evidence_and_native_root(evidence_context, monkeypatch, delivery):
    from ouroboros.review_records import ReviewRouteKind
    from ouroboros.tools.review_multi_model import _query_model
    import ouroboros.review_substrate as substrate

    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "Assessment")
    evidence = capture_commit_review_evidence(ctx)
    if delivery == "session":
        evidence = materialize_commit_review_session_view(evidence, ctx.repo_dir)
    captured = []
    def run(request, **kwargs):
        captured.append(request)
        return SimpleNamespace(actors=[{"status": "ok", "raw_text": "[]"}])
    monkeypatch.setattr(substrate, "run_review_request", run)
    route = ReviewRouteKind.AGENT_SESSION if delivery == "session" else ReviewRouteKind.API_CHAT
    messages = [{"role": "user", "content": "packet"}]
    asyncio.run(_query_model(None, "model", messages, asyncio.Semaphore(1), ctx,
                            route=route, session_task="Review", session_root=str(ctx.repo_dir),
                            task_evidence=evidence, subagent_id="native" if delivery == "native" else "",
                            native_retrieval=delivery == "native", use_local=False))  # F8: the delivery fact, not the id
    request = captured[0]
    assert request.evidence["task_execution"]["source_ref"] == evidence["source_ref"]
    assert evidence["source_ref"] in request.evidence_refs
    if delivery == "native":
        assert request.policy["native_data_root"] == ctx.budget_drive_root
        assert "root='artifact_store'" in request.session_task
    elif delivery == "session":
        assert evidence["session_relative_path"] in request.session_task
        assert "native_data_root" not in request.policy
    else:
        assert request.messages == messages
        assert request.session_task == ""


@pytest.mark.parametrize("fail", [False, True])
def test_loop_borrows_trace_through_tool_calls_and_restores_it(tmp_path, monkeypatch, fail):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry
    from tests.test_loop_transport_wait import _loop_kwargs

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    previous = {"prior": "trace"}
    registry._ctx._execution_trace = previous
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    observed = []
    def call(model_call):
        active = registry._ctx._execution_trace
        assert active is not previous
        observed.append(active)
        if fail:
            raise RuntimeError("fixture stop")
        return {"role": "assistant", "content": "done"}, 0.0, model_call.active_context_mode
    monkeypatch.setattr(loop, "_call_round_model", call)
    if fail:
        with pytest.raises(RuntimeError, match="fixture stop"):
            loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    else:
        _, _, trace = loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
        assert observed[0] is trace
    assert registry._ctx._execution_trace is previous


def test_canonical_write_failure_keeps_a_bounded_explicit_gap(evidence_context, monkeypatch):
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "Assessment")
    monkeypatch.setattr("ouroboros.artifacts.store_actor_source_bytes", lambda *a, **kw: (_ for _ in ()).throw(OSError("storage failure")))
    evidence = capture_commit_review_evidence(ctx)
    assert evidence["source_status"] == "unavailable" and evidence["source_ref"] == {}
    assert evidence["source_complete"] is False
    text = commit_review_evidence_section(evidence, delivery="native")
    assert "full source retrieval is unavailable" in text and "Assessment" in text
    assert len(text) <= _ACCEPT_NOTES_CAP


@pytest.mark.parametrize("pending", [False, True])
def test_session_copy_lifetime_follows_existing_review_custody(evidence_context, monkeypatch, pending):
    from ouroboros.tools import git_review_cycle
    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "Assessment")
    ctx._commit_review_evidence = materialize_commit_review_session_view(capture_commit_review_evidence(ctx), ctx.repo_dir)
    path = pathlib.Path(ctx._commit_review_evidence["session_path"])
    monkeypatch.setattr(git_review_cycle, "_review_custody_pending", lambda c: pending)
    git_review_cycle._release_review_evidence_if_settled(ctx)
    assert path.exists() is pending
    assert read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, ctx._commit_review_evidence["source_ref"])


@pytest.mark.parametrize("delivery", ["native", "session"])
def test_retrieving_seat_request_preserves_selected_source(evidence_context, monkeypatch, delivery):
    """A retrieving seat of the one wave (native episode or hosted session)
    carries the frozen task-execution evidence in its request; only the native
    episode gets the data root to page it from."""
    import asyncio

    from ouroboros.review_records import ReviewRouteKind
    from ouroboros.tools.review_multi_model import _query_model
    import ouroboros.review_substrate as substrate

    ctx = evidence_context
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "Assessment")
    evidence = capture_commit_review_evidence(ctx)
    captured = []
    def receive(request, **kwargs):
        captured.append(request)
        return SimpleNamespace(actors=[{"status": "ok", "raw_text": "[]", "usage": {}}])
    monkeypatch.setattr(substrate, "run_review_request", receive)
    asyncio.run(_query_model(
        None, "fixture", [], asyncio.Semaphore(1), ctx, slot_id="seat",
        route=ReviewRouteKind.AGENT_SESSION if delivery == "session" else ReviewRouteKind.API_CHAT,
        session_root=str(ctx.repo_dir), session_task="two-part brief",
        native_retrieval=delivery == "native", task_evidence=evidence, use_local=False))
    request = captured[0]
    assert request.evidence["task_execution"] == evidence
    assert evidence["source_ref"] in request.evidence_refs
    assert (request.policy.get("native_data_root") == ctx.budget_drive_root) is (delivery == "native")


@pytest.mark.parametrize("image_state", ["attached", "missing", "reported_failure"])
@pytest.mark.parametrize("skill", ["unix_computer_use", "another_image_producer"])
def test_autoattached_image_process_trace_reaches_commit_evidence(evidence_context, image_state, skill):
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.loop_tool_execution import process_tool_results
    from ouroboros.tools.tool_result import ToolResult

    ctx = evidence_context
    image = ctx.repo_dir / "captured.png"
    if image_state != "missing":
        image.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j8eUAAAAASUVORK5CYII="))
    model_response(ctx, "before", "Inspect the application screenshot")
    name = extension_surface_name(skill, "screenshot")
    raw = json.dumps({"ok": image_state != "reported_failure", "path": str(image), "auto_attach_image": str(image)})
    call = tool_response(ctx, "screen", "before", tool=name, result=raw)
    ctx._execution_trace["tool_calls"].clear()
    ctx.messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": fn, "arguments": "{}"}}
        for call_id, fn in [("screen", name), ("read", "read_file")]]}]
    rows = [{"fn_name": name, "tool_call_id": "screen", "result": raw, "trace_ref": call["trace_ref"],
             "is_error": False, "tool_args": call["args"], "args_for_log": call["args"], "result_meta": {}},
            {"fn_name": "read_file", "tool_call_id": "read", "result": "ordinary result",
             "is_error": False, "tool_args": {}, "args_for_log": {}, "result_meta": {}}]
    if image_state == "reported_failure":
        rows[0]["tool_result"] = ToolResult(status="error", code="TOOL_REPORTED_FAILURE", text=raw)
    assert process_tool_results(rows, ctx.messages, ctx._execution_trace, lambda _: None,
                                tools=SimpleNamespace(_ctx=ctx)) == 0
    assert [m["role"] for m in ctx.messages[:3]] == ["assistant", "tool", "tool"]
    pictures = [b for m in ctx.messages if isinstance(m.get("content"), list)
                for b in m["content"] if b.get("type") == "image_url"]
    assert bool(pictures) is (image_state == "attached")
    assert ctx._execution_trace["tool_calls"][1].get("image_attachment") is None
    if pictures:
        assert pathlib.Path(pictures[0]["_source_path"]).read_bytes() == image.read_bytes()
    model_response(ctx, "after", [{"type": "thinking", "text": "PRIVATE_THINKING"},
                                  {"type": "text", "text": "IMMEDIATE_VISIBLE_ASSESSMENT"}])
    evidence = capture_commit_review_evidence(ctx)
    if image_state == "reported_failure":
        assert evidence == {}
        assert "image_attachment" not in ctx._execution_trace["tool_calls"][0]
    else:
        assert evidence["selected_count"] == 1 and evidence["unselected_count"] == 1
        exact = read_actor_source_bytes(ctx.budget_drive_root, ctx.task_id, evidence["source_ref"]).decode()
        assert name in exact and raw in json.loads(exact.split("\n\n", 2)[2])["result"]
        assert '"image_attachment"' in exact and "IMMEDIATE_VISIBLE_ASSESSMENT" in exact
        assert "PRIVATE_THINKING" not in exact
        assert evidence["source_complete"] is (image_state == "attached")
        assert "proof of visual inspection" in exact


def test_fresh_stage_captures_current_evidence_over_a_stale_selection(evidence_context, monkeypatch):
    """No preflight rejoin carries a frozen selection any more (3A): every stage reads
    the current trace, so a stale selection left on the context never reaches the panel."""
    from ouroboros.tools import git

    ctx = evidence_context
    (ctx.repo_dir / "README.md").write_text("fresh candidate\n")
    ctx._commit_review_evidence = {"preview": "OLD_REJOIN"}
    model_response(ctx, "before", "Before")
    tool_response(ctx, "new", "before")
    model_response(ctx, "after", "NEW_ASSESSMENT")
    monkeypatch.setattr(git, "_free_cycle_gate", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_preflight_and_tests_gate", lambda *a, **kw: None)
    monkeypatch.setattr(git, "_install_paid_dispatch_stamp", lambda *a: None)
    observed = []
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **kw: (observed.append(ctx._commit_review_evidence) or None, None, "", []))
    result = git._run_reviewed_stage_cycle(ctx, "fresh", time.time(), paths=["README.md"], require_release_tag=False)
    assert result["status"] == "passed"
    assert len(observed) == 1 and "NEW_ASSESSMENT" in observed[0]["preview"]


def test_real_packet_assembly_omits_optional_excerpt_before_required_material(evidence_context, monkeypatch):
    """The triad packet drops the optional evidence excerpt before it degrades
    required material, and before it spends a paid density probe."""
    from ouroboros.review_records import ReviewRouteKind
    from ouroboros.tools import review, review_admission

    ctx = evidence_context
    for path in ("BIBLE.md", "docs/DEVELOPMENT.md", "docs/DESIGN.md", "docs/ARCHITECTURE.md", "docs/CHECKLISTS.md"):
        target = ctx.repo_dir / path
        target.parent.mkdir(exist_ok=True)
        target.write_text("GOVERNANCE_MARKER " + path)
    (ctx.repo_dir / "README.md").write_text("MANDATORY_SNAPSHOT\n")
    subprocess.run(["git", "add", "README.md"], cwd=ctx.repo_dir, check=True)
    model_response(ctx, "before", "Before")
    tool_response(ctx, "one", "before")
    model_response(ctx, "after", "OPTIONAL_IMAGE_EXCERPT\n" * 400)
    evidence = capture_commit_review_evidence(ctx)
    ctx._commit_review_evidence = evidence
    cap = [10**9]
    monkeypatch.setattr(review_admission, "density_probe_before_size_refusal", lambda *a, **kw: pytest.fail("optional excerpt should fit before paid density probe"))
    monkeypatch.setattr(review, "_preflight_check", lambda *a: None)
    monkeypatch.setattr(review, "_load_checklist_section", lambda *_a, **_k: "CHECKLIST_MARKER")
    monkeypatch.setattr("ouroboros.reviewer_slot_config.commit_triad_delivery", lambda: {
        "models": ["fixture"], "routes": [ReviewRouteKind.API_CHAT], "slot_ids": ["triad-one"],
        "session_profiles": [""], "subagent_ids": [""], "use_local": [False]})
    monkeypatch.setattr(review, "reviewer_context_window", lambda *a, **kw: 1000000)
    monkeypatch.setattr(review, "calibrated_input_token_limit", lambda *a, **kw: cap[0])
    monkeypatch.setattr(review, "estimate_tokens", len)

    def build():
        prepared, early, exited = review._prepare_unified_review(ctx, "candidate", goal="INTENT_MARKER", review_rebuttal="REBUTTAL_MARKER")
        assert not exited and early is None
        return prepared["prompt"], prepared["stable_prefix_len"]

    full, prefix = build()
    long_exhibit = commit_review_evidence_section(evidence, delivery="packet")
    short_exhibit = commit_review_evidence_section(evidence, delivery="packet", compact=True)
    assert long_exhibit in full
    expected = full.replace(long_exhibit, short_exhibit)
    cap[0] = len(expected) + 1
    assert len(full) > cap[0]
    fitted, next_prefix = build()
    assert fitted == expected and next_prefix == prefix
    assert full[:prefix] == fitted[:prefix]
    for marker in ("CHECKLIST_MARKER", "INTENT_MARKER", "REBUTTAL_MARKER", "MANDATORY_SNAPSHOT", "+MANDATORY_SNAPSHOT"):
        assert marker in fitted
    # The triad packet no longer pastes the reference books in full: the
    # governance tiers deliver them as navigation, BIBLE.md rides every api
    # row's constitutional head (outside this prompt) and the standing
    # disclosures ride the checklist section. Nothing is dropped silently —
    # every document is named here and dispositioned in the manifest.
    assert "Governance navigation (index of sources not inlined)" in fitted
    assert "docs/ARCHITECTURE.md" in fitted and "docs/DEVELOPMENT.md" in fitted
    assert "OPTIONAL_IMAGE_EXCERPT" not in fitted and "excerpt omitted to fit" in fitted
    assert ctx._commit_review_evidence == evidence
