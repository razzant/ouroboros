"""Owner Continue reaches the next full Main request without losing memory sources.

The installed queue, HTTP ingress, context builder, Main loop, Chronicle and
source readers are real. Only model/catalog boundaries are local fixtures; the
plan review history a predecessor carries is an earlier-schema record the retired
plan-review engine left, recorded from its real flow (tests/fixtures/legacy_plan_dispute).
These tests qualify transport and retention, not a model's semantic recall.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import queue

import pytest

from ouroboros import memory_floor, memory_view, review_history_view
from ouroboros.agent import Env
from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.context import build_llm_messages
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
from ouroboros.memory import Memory
from ouroboros.task_results import load_task_result
from ouroboros.tools.registry import ToolRegistry
from tests._budget_pause_exact_helpers import _install_queue
from tests.test_common_story import _journal, _rooms, _snapshot, _story, _write
from tests.test_common_story_composition import _read_all
from tests.test_owner_continue import _interrupted
from tests.test_plan_dispute_history import RATIONALE, RAW_REASON, _state
from tests.test_review_local_accounts import (
    FIRST, OPEN_REQUIREMENT, SECOND, actor_notes, apply_local, open_blocking, review_units,
)
from tests.test_review_view_integration import _capture, harness as _harness

harness = _harness
pytestmark = pytest.mark.serial  # Continue installs the supervisor's module globals.


OWNER = "Finish the authorized chart export without deploying a service."
SPOKEN = "I will retain the chart protocol.\nMay I add a plain-text export?"
ANSWER = "Add the export, keep the chart units, and do not deploy a service."
CURRENT = OPEN_REQUIREMENT  # the recorded fourth round's open blocking finding


def message_text(messages):
    return "\n".join(row["content"] if isinstance(row["content"], str)
                     else "".join(block.get("text", "") for block in row["content"])
                     for row in messages)


@pytest.fixture(autouse=True)
def offline_main(monkeypatch, provider_catalog_offline):
    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setenv("MCP_ENABLED", "false")


def continue_task(root, predecessor, monkeypatch, **fields):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.task_continue import api_task_continue

    _, _, workers = _install_queue(root, monkeypatch)
    _interrupted(root, predecessor, chat_id=1, origin_message_text=OWNER, **fields)
    with TestClient(Starlette(routes=[Route("/api/tasks/{task_id}/continue", api_task_continue,
                                          methods=["POST"])])) as client:
        response = client.post(f"/api/tasks/{predecessor}/continue",
                               json={"action_nonce": "memory-followup-press-0001"})
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt["ok"] and not receipt["held"]
    task = next(row for row in workers.PENDING if row["id"] == receipt["successor_task_id"])
    assert task["id"] != predecessor and task["root_task_id"] == task["id"]
    assert load_task_result(root, task["id"])["status"] == "scheduled"
    return task


def send_next(env, task):
    """Build a fresh root and capture its complete first provider-boundary request."""
    from ouroboros import loop, usage_accounting as ua
    from ouroboros.agent_startup_checks import validate_task_authority_sources
    from ouroboros.contracts.task_contract import attach_task_contract
    from ouroboros.llm_attempt import _attempt_request, _physical_candidate
    from ouroboros.model_send_seal import persist_physical_candidate

    # Worker boot binds the admitted predecessor pointer before building context.
    assert validate_task_authority_sources(env, task) == {}
    task = attach_task_contract(task)
    registry = ToolRegistry(repo_dir=env.repo_dir, drive_root=env.drive_root)
    ctx = registry._ctx
    ctx.task_id, ctx.current_chat_id = task["id"], task["chat_id"]
    ctx.task_metadata = {**task["metadata"], "root_task_id": task["id"], "delegation_role": "root"}
    ctx.task_contract = task.get("task_contract") or {}
    if task.get("workspace_root"):
        ctx.workspace_root = Path(task["workspace_root"])
        ctx.workspace_mode = task["workspace_mode"]
    memory = Memory(drive_root=env.drive_root, repo_dir=env.repo_dir)
    messages, cap = build_llm_messages(env, memory, task, ctx=ctx)
    assembled, sent = deepcopy(messages), []

    class Provider:
        def default_model(self):
            return ctx.context_fit_plan.model

        def chat(self, **kwargs):
            sent.append(deepcopy(kwargs))
            candidate = _physical_candidate({key: kwargs[key] for key in ("messages", "tools", "model")})
            request = _attempt_request({"provider": "openai", "usage_model": kwargs["model"]}, candidate)
            persisted = persist_physical_candidate(env.drive_root, task_id=task["id"],
                attempt_id="memory-followup-send", candidate=candidate, candidate_facts={})
            ua.adopt_physical_attempt_capture(ua.PhysicalAttemptCapture(
                "memory-followup-send", kwargs["model"], "openai", "settled", "canonical_json_v1",
                candidate_manifest_ref=persisted["manifest_ref"], physical_context=request.physical_context,
                candidate_raw_sha256=request.candidate_raw_sha256, candidate_raw_size_bytes=request.candidate_raw_size_bytes,
                candidate_context_sha256=request.candidate_context_sha256,
                candidate_context_size_bytes=request.candidate_context_size_bytes))
            return {"role": "assistant", "content": "The retained work is available."}, {
                "prompt_tokens": 100, "completion_tokens": 10, "cost": 0.0, "provider": "openai"}

    result, _, _ = loop.run_llm_loop(messages, registry, Provider(), env.drive_path("logs"),
        lambda *_a, **_kw: None, queue.Queue(), task_id=task["id"], drive_root=env.drive_root)
    assert result == "The retained work is available." and len(sent) == 1
    def text(content):
        return content if isinstance(content, str) else "".join(block.get("text", "") for block in content)
    # The send path may wrap an input in cache blocks and append round facts.
    # Every assembled body must still arrive whole, in the same role and order.
    for original, transmitted in zip(assembled, sent[0]["messages"]):
        assert transmitted["role"] == original["role"]
        assert text(transmitted["content"]).startswith(text(original["content"]))
    assert len(sent[0]["messages"]) >= len(assembled)
    assert sent[0]["tools"] and sent[0]["model"]
    return ctx, assembled, sent[0], cap


def test_owner_continue_sends_two_local_notes_review_readers_and_exact_dialogue(harness, monkeypatch):
    from ouroboros.context import _task_authority_projection
    from ouroboros.owner_mailbox import KIND_OWNER_TEXT, write_owner_message
    from ouroboros.tools.control_runtime import _send_user_message
    from ouroboros.tools.owner_delivery import publish_pending_owner_dialogue
    from supervisor.message_bus import log_chat

    # Three closed rounds and a fourth with the open blocking CURRENT, as recorded.
    ctx, raw_review = open_blocking(harness)
    substrate = harness.install({})
    before_state = deepcopy(_state(harness))
    before = _capture(ctx)
    ctx.messages, ctx.current_chat_id = before, 1
    assert "OK" in _send_user_message(ctx, SPOKEN)
    publish_pending_owner_dialogue(ctx, before)
    speech = next(row for row in before if row.get(HOST_CONTEXT_KIND_KEY) == "owner_dialogue")
    facts = json.loads(speech["content"].split("\n", 2)[1])
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, facts["source_ref"]).decode("utf-8") == SPOKEN
    # Canonical host logging records both sides of the addressed conversation.
    assert log_chat("out", 1, 0, SPOKEN, task_id=ctx.task_id, record_type="proactive_message",
                    drive_root=ctx.drive_root, require_write=True)
    assert log_chat("in", 1, 0, ANSWER, client_message_id="owner-export-answer",
                    drive_root=ctx.drive_root, require_write=True)
    write_owner_message(ctx.drive_root, ANSWER, ctx.task_id, msg_id="owner-export-answer", kind=KIND_OWNER_TEXT)
    before.append({"role": "user", "content": ANSWER})
    group = [entry["decision_ref"] for entry in review_history_view.review_note_options(ctx)["entries"]
             if entry["groupable"]]
    assert group
    first_critic = next(unit for unit in review_units(before) if RAW_REASON in unit.source_text)
    first, receipt, _ = apply_local(ctx, before, FIRST,
        remove=[first_critic.unit_id], review_notes=[{
            "bound_decisions": group, "remark": "Earlier chart alternatives.",
            "reason": "Charts were chosen under the earlier contract; changed premises may reopen the choice."}])
    assert receipt["status"] == "applied"
    second, receipt, _ = apply_local(ctx, first, SECOND, remove=[review_units(first)[-1].unit_id])
    assert receipt["status"] == "applied" and len(actor_notes(second)) == 2
    assert speech in second and ANSWER in str(second) and CURRENT in str(second)
    assert "actor_history_group" in str(second) and RATIONALE not in str(second)
    selected = review_history_view.load_review_history_view(receipt["selected_review_history_view"],
        lambda ref: read_actor_source_bytes(ctx.drive_root, ctx.task_id, ref))
    # Peer mail of a long-running root: many notes from other tasks, none of them owner words.
    from ouroboros.owner_mailbox import write_task_message
    peer_bodies = [f"peer note {index}: the quota helper reports " + "q" * 400 + f" END{index}" for index in range(30)]
    for index, body in enumerate(peer_bodies):
        assert write_task_message(ctx.drive_root, body, ctx.task_id, source_task_id=f"helper-{index % 5}",
                                  msg_id=f"peer-{index}")

    task = continue_task(ctx.drive_root, ctx.task_id, monkeypatch,
                         workspace_root=str(ctx.workspace_root), workspace_mode="external")
    env = Env(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    successor, assembled, request, _ = send_next(env, task)
    text = message_text(request["messages"])
    for retained in (FIRST, SECOND, ANSWER):
        assert retained in text and retained in message_text(assembled)
    assert CURRENT not in text, "the predecessor's review state stays behind its reader"
    # Peer mail is a count plus the exact reader, never bodies; the work order appears once
    # (the contract objective inside the frozen prefix is a typed reference to the assignment).
    from ouroboros.context_compaction import context_units, unit_kind
    assert "30 messages" in task["text"] and "select='metadata.continuation.peer_context'" in task["text"]
    assert not any(body in text for body in peer_bodies)
    assert text.count("[CONTINUE] Continue was requested for task") == 1
    runtime_block = next(block["text"] for block in assembled[0]["content"] if "## Runtime context\n\n" in block["text"])
    prefix = json.loads(runtime_block.split("## Runtime context\n\n", 1)[1].split("\n\n## ", 1)[0])
    assert prefix["task_contract"]["objective"]["representation"] == "assignment"
    assert "historical_review_context" not in json.dumps(prefix) and "previous_review_context" not in json.dumps(prefix)
    brief = prefix["predecessor_authority"]  # status and the exact reader, once
    assert brief["source"]["arguments"] == {"task_id": ctx.task_id, "include_authority": True}
    assert brief["status"] == "cancelled"
    for field in ("task_contract", "plan_review_state"):
        assert field not in brief and brief["omitted_fields"][field] > 0
    assert "omitted_fields" in brief and "predecessor_authority" not in prefix["task_contract"]
    assert json.dumps(prefix).count(f'"task_id": "{ctx.task_id}"') >= 1 and OWNER in task["text"]
    # The notes row is a foldable dialogue unit of the mind's own view; the assignment is not.
    units = context_units(assembled, scope="dialogue")
    assert 2 in [unit.start for unit in units] and 1 not in [unit.start for unit in units]
    assert unit_kind(assembled, next(unit for unit in units if unit.start == 2)) == "user"
    # Newline escaping varies by provider content wrapper; the room keeps the exact words.
    assert all(line in text for line in SPOKEN.splitlines())
    assert raw_review not in text and RAW_REASON not in text and RATIONALE not in text
    assert _state(harness) == before_state and substrate.calls == []
    assert review_history_view.SELECTED_VIEW_FIELD not in load_task_result(ctx.drive_root, task["id"])
    assert SPOKEN not in str(successor._owner_directives) and ANSWER in str(successor._owner_directives)
    # The two local notes ride as ONE host prose row right after the assignment, never in the
    # frozen prefix, naming the exact retained source; the ancestor history is not walked.
    from ouroboros.main_context_authority import PREDECESSOR_NOTES_NAME
    from ouroboros.tools.control import _get_task_result

    runtime = _task_authority_projection(env, task)
    assert "historical_review_context" not in json.dumps(runtime) and FIRST not in json.dumps(runtime)
    (note,) = [row for row in assembled if row.get(HOST_CONTEXT_KIND_KEY) == PREDECESSOR_NOTES_NAME]
    assert assembled.index(note) == 2 and FIRST in note["content"] and SECOND in note["content"]
    pointer = receipt["selected_review_history_view"]
    from ouroboros.main_context_authority import WORKING_NOTES_FIELD

    assert load_task_result(ctx.drive_root, ctx.task_id)[WORKING_NOTES_FIELD]["path"] in note["content"]  # last fold's
    assert f"get_task_result(task_id={ctx.task_id!r}, include_authority=True)" in note["content"]
    payload = json.loads(_get_task_result(successor, ctx.task_id, include_authority=True))
    assert payload["authority"][review_history_view.SELECTED_VIEW_FIELD] == pointer
    assert payload["authority"]["task_contract"] and payload["authority"]["plan_review_state"]
    assert CURRENT in json.dumps(payload["authority"]["plan_review_state"])
    covered = [read_actor_source_bytes(ctx.drive_root, ctx.task_id, item["source_ref"]).decode("utf-8")
               for item in selected["covered"]]
    assert covered and any(RATIONALE in source for source in covered), "the exact decision source stays readable"
    assert read_actor_source_bytes(ctx.drive_root, ctx.task_id, facts["source_ref"]).decode("utf-8") == SPOKEN


def test_owner_continue_sends_selected_corrected_account_and_retains_old_source_versions(tmp_path, monkeypatch):
    from ouroboros.tools.chronicle import _memory_read
    from tests.test_cache_optimization import _make_env_and_memory
    from tests.test_chronicle_tools import _parse_window

    env, _ = _make_env_and_memory(tmp_path)
    root = env.drive_root
    _, _, _, page, other, _ = _rooms(root)
    old_text = "The rehearsal appeared to finish; the independent garden measurement was real."
    old = _write(root, kind="account", text=old_text, sources=[page, other])["node_id"]
    assert _write(root, kind="selection", target_id=old, replaces=[page, other], reason="My earlier account.")["ok"]
    correction = "The rehearsal was not delivery. The garden's independent measurement remains valid."
    fix = _write(root, kind="correction", target_id=page, text=correction)["node_id"]
    new_text = "I withdrew the rehearsal as delivery; the garden result remains valid and export work is still open."
    new = _write(root, kind="account", text=new_text, sources=[{"id": page, "revision": fix}, other])["node_id"]
    assert new_text not in _story(root), "Publishing understanding alone must not replace the current story"
    assert correction in _story(root)
    assert _write(root, kind="selection", target_id=new, replaces=[old, page, other],
                  reason="My revised account over the corrected sources.")["ok"]
    journal = _journal(root)
    task = continue_task(root, "story-predecessor", monkeypatch)
    successor, assembled, request, cap = send_next(env, task)
    for messages in (assembled, request["messages"]):
        text = message_text(messages)
        assert new_text in text and old_text not in text
        assert "later correction of its source" not in text
        assert f"memory_read(node_id='{new}')" in text
    assert cap["memory_view"]["floor"]["pointer_records"] == []
    exact_new = _memory_read(successor, node_id=new)
    assert f"source page {page}" in exact_new and f"revision {fix} used" in exact_new
    exact_old = _memory_read(successor, node_id=old)
    assert old_text in exact_old and f"revision {page} used; later corrections not in this account: {fix}" in exact_old
    exact_source = _memory_read(successor, node_id=page, revision=fix)
    assert correction in exact_source and "Alpha asked again and I counted." in exact_source
    assert all(_parse_window(result)[-1] is None for result in (exact_new, exact_old, exact_source))
    assert _journal(root) == journal and successor.task_id == task["id"]


def test_physical_floor_keeps_account_and_pending_correction_without_claiming_incorporation(tmp_path):
    _, _, _, page, other, _ = _rooms(tmp_path)
    account_text = "I had treated the rehearsal as delivery, while the garden result had independent evidence."
    account = _write(tmp_path, kind="account", text=account_text, sources=[page, other])["node_id"]
    assert _write(tmp_path, kind="selection", target_id=account, replaces=[page, other], reason="Together.")["ok"]
    correction = ("The rehearsal was withdrawn, so the export still needs a new observation. The workshop count is unverified.\n" * 950
                  + "The garden's independent measurement remains valid. Final exception Ω.")
    fix = _write(tmp_path, kind="correction", target_id=page, text=correction)["node_id"]
    snapshot, journal = _snapshot(tmp_path), _journal(tmp_path)
    full, _, facts = memory_floor.render_view_for_mode(snapshot, mode="max", owner_mode="max",
        window_tokens=1_000_000, known_window=True, output_reserve=512, ratio=1.0, non_memory_tokens=0)
    assert account_text in full and "Final exception Ω." in full
    assert f"later correction of its source {page} (not in this account)" in full
    assert facts["floor"]["pointer_records"] == []
    short, room, facts = memory_floor.render_view_for_mode(snapshot, mode="max", owner_mode="max",
        window_tokens=4096, known_window=True, output_reserve=512, ratio=1.0, non_memory_tokens=0)
    assert account not in facts["floor"]["pointer_records"]
    assert account_text in short and "Final exception Ω." in short
    assert memory_floor.minimal_view_tokens(snapshot) > 4096 - 512
    assert "### Physical floor" in room and "reduced forms" in room
    assert "2026-09-03 00:00 → 2026-09-03 00:03" in short
    assert "incorporated" not in short and "not in this account" in short
    exact, _ = _read_all(tmp_path, account)
    assert f"revision {page} used; later corrections not in this account: {fix}" in exact
    exact_correction, windows = _read_all(tmp_path, fix)
    assert windows > 1 and exact_correction.split("text:\n", 1)[1] == correction
    assert _journal(tmp_path) == journal
    assert memory_view.snapshot_from_json(memory_view.snapshot_json(snapshot)) == snapshot
