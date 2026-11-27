"""A stored new-ID timeout retry reads its inherited sources without inheriting authority.

No producer mints a new-ID successor now (#1376); one admitted before the upgrade is restored
(``tests._silence_reaper_support``) and its real startup reads its predecessor's sources.
"""
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
from tests._silence_reaper_support import historical_successor
from tests.test_context_reclaim_materializer import _request
from tests.test_context_source_view import _prefix, _tool
from tests.test_root_effort_ingress import _pool_ready


@pytest.mark.parametrize(("place", "saved"), [("main", "unreadable"), ("main", "missing"), ("room", "unreadable"),
                                              ("main", "healthy")])
def test_a_restarted_helper_s_first_request_carries_its_held_owner_row_once(tmp_path, monkeypatch, place, saved):
    """#3555: a native helper held my human's carried row on its record and ACKed it in a ready save, then died. The
    same-ID retry runs the real startup, loop, dispatch and fitting; the fake stands at the provider client, so this
    observes what the loop hands it, not a vendor's wire. With the selected checkpoint unreadable or none saved, the
    first request shows the words once: quoted where its built input lacks them, as the addressed fact where its
    composed room rows show them. A usable checkpoint restores its own copy; the held receipt is never restamped."""
    import json
    import queue
    from ouroboros import dialogue_store, loop, owner_relay, usage_accounting as ua
    from ouroboros.chat_chain import format_address, row_address
    from ouroboros.llm_messages import project_wire_messages
    from ouroboros.loop_messages import _extract_plain_text_from_content
    from ouroboros.owner_mailbox import _ack_path
    from ouroboros.projects_registry import bind_task_to_project, create_project

    root = tmp_path / "data"
    root.mkdir()
    _q, _state, workers = _install_queue(root, monkeypatch)
    _pool_ready(monkeypatch, workers)
    for name, value in (("OUROBOROS_CONTEXT_MODE", "max"), ("OUROBOROS_SAFETY_MODE", "off"), ("MCP_ENABLED", "false"),
                        ("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off"), ("OUROBOROS_TASK_REVIEW_MODE", "off"),
                        ("OUROBOROS_MAX_ROUNDS", "12")):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("No real provider call"))
    monkeypatch.setattr("ouroboros.provider_models.model_has_credentials", lambda _model: True)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr(context, "_context_fit_route", lambda task, **kw: (
        {"model": "openai/test-model", "provider": "openai", "use_local": False},
        SimpleNamespace(status="confirmed", stale=False, window_tokens=500_000, route_fp="test-route")))
    chat, words, task_id = 1, "Please make the chart titles blue.", "helper-1"
    # The tree's root row is the owner-Pause authority every lifecycle-bound member reads (``owner_pause.member_fence``);
    # a scheduling root has it, open with no Pause, as ``tests/test_budget_pause_lineage.py`` seeds its root.
    write_task_result(root, "root", "running", chat_id=0, root_task_id="root")
    if place == "room":
        chat = int(create_project(root, "quotas", name="Quotas")["chat_id"])
        bind_task_to_project(root, "root", "quotas", chat, origin={"absent": "system"})
    task = {"id": task_id, "type": "task", "text": "Write the quarterly summary.", "chat_id": chat,
            "model": "openai/test-model", "delegation_role": "subagent", "root_task_id": "root",
            "parent_task_id": "root", "drive_root": str(root), "budget_drive_root": str(root), "_attempt": 1}
    write_task_result(root, task_id, "running", task_attempt=1, admitted_dispatch_attempt=1)

    def boot(attempt_task):
        agent = OuroborosAgent(Env(repo_dir=Path(__file__).resolve().parents[1], drive_root=root,
                                  budget_drive_root=str(root)))
        agent._task_started_ts, agent._pending_events = time.time(), []
        ctx, messages, cap = agent._prepare_task_context(attempt_task)
        assert ctx and not cap.get("authority_source_unavailable"), cap
        return agent, ctx, messages

    def visible(messages):  # the text a model reads: the wire projection, never host-only row metadata
        return "\n".join(_extract_plain_text_from_content(row.get("content"))
                         for row in project_wire_messages(messages) if row.get("role") != "assistant")

    first_agent, first_ctx, messages = boot(task)
    assert words not in visible(messages)  # the helper was born before my human spoke
    row = {"chat_id": chat, "direction": "in", "ts": "2026-10-04T06:00:00+00:00", "text": words, "client_message_id": "msg-2"}
    dialogue_store.append(root, "chat", row)
    address = format_address(row_address(row))
    entry, fresh = owner_relay.write_relay(
        root, task_id, relay_id=f"carry:{address}", note="", chat_id=chat, teller_task_id="root",
        quotes=[owner_relay.owner_quote(root, row, address=address)], canonical_root=root, host_carried={"went_to": ""})
    assert fresh
    messages.append({"role": "assistant", "content": "Working."})  # the carried row reaches a later round
    ids = set()
    loop._drain_incoming_messages(messages, queue.Queue(), root, task_id, None, ids, owner_ctx=first_ctx,
                                  defer_content_ack=True)
    assert visible(messages).count(words) == 1
    wc.save_ready(SimpleNamespace(tools=first_agent.tools, messages=messages, llm_trace={"tool_calls": []}, accumulated_usage={},
                                  round_idx=1, tool_schemas=[], owner_msg_seen=ids, budget_tail="no_tool"))
    acks = [json.loads(line) for line in _ack_path(root, task_id).read_bytes().splitlines()]
    assert [(ack["msg_id"], ack.get("attempt_key")) for ack in acks] == [(entry["msg_id"], "1")]
    receipt = owner_relay._own_held_state(first_ctx)["held_owner_messages"][address]["carried"]
    retry = {**task, "_attempt": 2}
    if saved == "missing":
        wc.checkpoint_path(root, task_id, 1).unlink()  # the worker died with no saved working state
    assert wc.attach_recovery(root, retry, source_task_id=task_id, from_attempt=1, cause="worker_crash") is (saved != "missing")
    if saved == "unreadable":  # the selected frozen source is damaged after selection; its locator stays as selected
        frozen = task_artifact_dir_path(root, task_id, create=False) / retry["_working_recovery"]["source_ref"]["path"]
        frozen.write_bytes(b"{}")
    agent, ctx, composed = boot(retry)
    assert visible(composed).count(words) == int(place == "room")  # the composer reads a room's row, never a Main row
    sent = []

    class Provider:  # the provider client's place: the loop's own dispatch, fitting and shaping run before it
        def default_model(self):
            return "openai/test-model"

        def chat(self, **kwargs):
            ua.adopt_physical_attempt_capture(None)  # no physical receipt is claimed for this fake send
            sent.append(deepcopy(kwargs))
            return {"content": "The summary is ready."}, {"prompt_tokens": 100, "completion_tokens": 10,
                                                           "cost": 0.0, "provider": "openai"}

    loop.run_llm_loop(messages=composed, tools=agent.tools, llm=Provider(), drive_logs=agent.env.drive_path("logs"),
                      emit_progress=lambda text, **meta: None, incoming_messages=queue.Queue(), task_id=task_id,
                      drive_root=root, task_type="task", event_queue=queue.Queue())
    first = visible(sent[0]["messages"])
    quoted = f"carried to this work by the host (reached this work); handle {address}]\n{words}"
    told = f"[Host fact: the row {address} in your starting words was addressed to this work]"
    assert first.count(words) == 1, "my human's words reach the first request to the provider client exactly once"
    assert (first.count(quoted), first.count(told)) == ((0, 1) if place == "room" else (1, 0))
    assert ("could not be restored" in first) is (saved == "unreadable")
    assert ("continued from its working checkpoint" in first) is (saved == "healthy")
    assert owner_relay._own_held_state(ctx)["held_owner_messages"][address]["carried"] == receipt  # kept, not restamped


@pytest.fixture
def restored(tmp_path, monkeypatch, request):
    root = tmp_path / "data"
    root.mkdir()
    q, _state, workers = _install_queue(root, monkeypatch)
    _pool_ready(monkeypatch, workers)
    # Startup and the registry are real; only external metadata/model calls are doubled.
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *a, **k: pytest.fail("No model call"))
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
    limit.llm_trace = {"review_runs": [{"authority": "host_root", "verdict": "PASS"}]}
    wc.save_round(limit, "ready")
    task = {"id": old.task_id, "type": "task", "text": "Keep exact source.", "chat_id": 0,
            "root_task_id": old.task_id, "_attempt": 1}
    historical_successor(task, 1, "checkpoint-successor")
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
    own_path.parent.mkdir(parents=True, exist_ok=True)
    assert not own_path.exists()  # the source this test introduces is not already there
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
