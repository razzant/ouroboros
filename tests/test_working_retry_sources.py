"""A real new-ID retry reads its inherited sources without inheriting authority."""
from copy import deepcopy
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

from ouroboros import context, context_compaction as cc, working_checkpoint as wc
from ouroboros.agent import Env, OuroborosAgent
from ouroboros.artifacts import read_actor_source_bytes, store_actor_source_bytes, task_artifact_dir_path
from ouroboros.context_source_view import emergency_address_view
from ouroboros.task_results import load_task_result, write_task_result
from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_context_reclaim_materializer import _request
from tests.test_context_source_view import _prefix, _tool
from tests.test_root_effort_ingress import _pool_ready


@pytest.fixture
def restored(tmp_path, monkeypatch, request):
    root = tmp_path / "data"
    root.mkdir()
    q, _state, workers = _install_queue(root, monkeypatch)
    _pool_ready(monkeypatch, workers)
    # Startup and the registry are real; only external metadata/model calls are doubled.
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("No model call"))
    monkeypatch.setattr(cc, "_call_summarizer", lambda *a, **k: pytest.fail("No helper call"))
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": task["model"], "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="test-route")))
    old, limit = _loop_ctx(root, "checkpoint-origin")
    write_task_result(root, old.task_id, "running", task_attempt=1, admitted_dispatch_attempt=1)
    raw = [*_prefix(), *_tool(text="Exact previous result, Ж.\r\n" * 1000)]
    reclaim = _request(raw, 1)
    projected, receipt = emergency_address_view(raw, reclaim, rung="unseen_bodies", drive_root=root, task_id=old.task_id)
    assert receipt.status == "applied"
    limit.messages = projected
    old._task_acceptance_reviewed = True
    old._task_acceptance_reviewed_subject = "old subject"
    limit.llm_trace = {"review_runs": [{"authority": "host_root", "verdict": "PASS"}]}
    wc.save_round(limit, "ready")
    task = {"id": old.task_id, "type": "task", "text": "Keep exact source.", "chat_id": 0,
            "root_task_id": old.task_id, "_attempt": 1}
    from supervisor import task_reaper
    admitted = task_reaper._enqueue_retry(q, task, task_id=old.task_id, retry_task_id="checkpoint-successor",
        attempt=1, terminal_reason="idle_timeout", recon_fields={})
    assert admitted[0], admitted
    [retry] = workers.PENDING
    drive = root / "task_drives" / retry["id"] if getattr(request, "param", "") == "successor_drive" else root
    drive.mkdir(parents=True, exist_ok=True)
    retry = {**retry, "drive_root": str(drive), "budget_drive_root": str(root)}
    agent = OuroborosAgent(Env(repo_dir=Path(__file__).resolve().parents[1], drive_root=drive,
                              budget_drive_root=str(root)))
    agent._task_started_ts, agent._pending_events = time.time(), []
    ctx, _startup, cap = agent._prepare_task_context(retry)
    assert ctx and not cap.get("authority_source_unavailable"), cap
    frozen = wc.load_recovery(ctx)
    saved_messages = deepcopy(frozen["messages"])
    messages, trace, usage, seen = [], {}, {}, set()
    wc.resume_from_working(agent.tools, frozen, messages, trace, usage, seen)
    assert frozen["messages"] == saved_messages  # The saved source is never rewritten.
    assert messages[2:len(projected)] == projected[2:]  # Existing route rebind refreshes the core only.
    assert ctx._task_acceptance_reviewed is False and not ctx._task_acceptance_reviewed_subject
    assert trace["review_runs"][0]["superseded_by_revision"] is True
    assert "selected_review_history_view" not in load_task_result(root, ctx.task_id)
    return SimpleNamespace(root=root, old=old.task_id, ctx=ctx, tools=agent.tools, raw=raw,
                           request=reclaim, receipt=receipt, projected=projected)


@pytest.mark.parametrize("restored", ["canonical", "retained_source", "successor_drive"], indirect=True)
def test_real_new_id_startup_reads_advertised_and_materialized_old_sources_and_own_sources(restored, request):
    f = restored
    ref = f.receipt.checkpoint_ref
    expected = read_actor_source_bytes(f.root, f.old, ref)
    if request.node.callspec.params["restored"] == "retained_source":
        child = f.root / "task_drives" / f.old
        child.mkdir(parents=True)
        old_path = task_artifact_dir_path(f.root, f.old, create=False) / ref["path"]
        target = task_artifact_dir_path(child, f.old, create=True) / ref["path"]
        target.parent.mkdir(parents=True)
        target.write_bytes(old_path.read_bytes())
        old_path.unlink()
    assert read_actor_source_bytes(f.ctx.drive_root, f.ctx.task_id, ref) == expected
    advertised = f.tools.execute("read_file", ref["read"]["arguments"])
    assert "Exact previous result" in advertised and "NOT_FOUND" not in advertised
    refs = [row for row in f.receipt.source_refs if "unit_id" in row]
    source_views, _ = cc._restored_source_views(refs, drive_root=f.ctx.drive_root, task_id=f.ctx.task_id, request=f.request)
    assert source_views and "Exact previous result" in str(source_views)
    with pytest.raises(ValueError, match="identity or full raw hash"):
        cc._restored_source_views([{**refs[0], "raw_sha256": "0" * 64}],
            drive_root=f.root, task_id=f.ctx.task_id, request=f.request)
    own = store_actor_source_bytes(f.root, f.ctx.task_id, category="tool_results", source_id="new-result",
                                  data=b"Exact current result", extension="txt")
    assert read_actor_source_bytes(f.root, f.ctx.task_id, own) == b"Exact current result"
    assert "Exact current result" in f.tools.execute("read_file", own["read"]["arguments"])
    assert f.ctx.task_contract.get("predecessor_authority") is None


def test_own_source_wins_and_wrong_or_corrupt_predecessor_is_not_replaced(restored):
    f = restored
    ref = f.receipt.checkpoint_ref
    old_path = task_artifact_dir_path(f.root, f.old, create=False) / ref["path"]
    expected = old_path.read_bytes()
    with pytest.raises(ValueError, match="sha256"):
        read_actor_source_bytes(f.root, f.ctx.task_id, {**ref, "sha256": "0" * 64})
    old_path.write_bytes(b"Corrupt previous source")
    with pytest.raises(ValueError):
        read_actor_source_bytes(f.root, f.ctx.task_id, ref)
    assert "READ_FILE_ERROR" in f.tools.execute("read_file", ref["read"]["arguments"])
    old_path.write_bytes(expected)
    own_path = task_artifact_dir_path(f.root, f.ctx.task_id, create=True) / ref["path"]
    own_path.parent.mkdir(parents=True)
    own_path.write_bytes(expected)
    old_path.write_bytes(b"The current source must win over this corrupt predecessor")
    assert read_actor_source_bytes(f.root, f.ctx.task_id, ref) == expected
    assert "Exact previous result" in f.tools.execute("read_file", ref["read"]["arguments"])


def test_missing_or_wrong_retry_link_does_not_invent_a_source(restored):
    f = restored
    ref = f.receipt.checkpoint_ref
    write_task_result(f.root, f.ctx.task_id, "running", timeout_retry_from="unrelated")
    with pytest.raises(FileNotFoundError):
        read_actor_source_bytes(f.root, f.ctx.task_id, ref)
    assert "NOT_FOUND" in f.tools.execute("read_file", ref["read"]["arguments"])
    own = store_actor_source_bytes(f.root, f.ctx.task_id, category="tool_results", source_id="own",
                                  data=b"Still reads its own source", extension="txt")
    assert "Still reads its own source" in f.tools.execute("read_file", own["read"]["arguments"])
