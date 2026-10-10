"""``/review`` reaches the worker through ONE queue door for every entry (decision 3A).

The agent's tool event, the Web selector (``/review <row>`` over the message bus)
and Telegram's bare ``/review`` all end in
``queue_deep_self_review_task(..., reviewer=...)``: the named row rides the task
to the worker, an empty name runs the Main model, and the acknowledgement — or
the refusal of a row that is not enabled — lands in the chat that asked.
"""

from __future__ import annotations

import types

import pytest

from tests.test_git_review_preflight_gate import _roster


@pytest.fixture()
def door(tmp_path, monkeypatch):
    from supervisor import queue, state

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    pending: list = []
    monkeypatch.setattr(queue, "PENDING", pending)
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    state.update_state(lambda live: live.update(owner_chat_id=1))
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    sent: list = []
    monkeypatch.setattr(queue, "send_with_budget",
                        lambda chat_id, text, **kw: sent.append((chat_id, text, kw.get("system_type"))))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("supervisor.workers._worker_pool_execution_state",
                        lambda: {"available": True, "disabled_reason": ""})
    return types.SimpleNamespace(queue=queue, pending=pending, sent=sent)


def test_a_named_row_rides_the_task_and_the_ack_names_it(door, monkeypatch):
    _roster(monkeypatch)
    tid = door.queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=7, reviewer=" api-scout ")
    [task] = door.pending
    assert (task["id"], task["type"], task["chat_id"], task["reviewer"]) == (tid, "deep_self_review", 7, "api-scout")
    assert door.sent == [(7, f"🔎 Deep self-review queued: {tid} (owner:/review; reviewer: api-scout)",
                          "deep_self_review_queued")]


def test_no_name_runs_the_main_model(door):
    tid = door.queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=7)
    assert door.pending[0]["reviewer"] == ""
    assert door.sent == [(7, f"🔎 Deep self-review queued: {tid} (owner:/review; reviewer: Main)",
                          "deep_self_review_queued")]


@pytest.mark.parametrize("enabled, name", [(False, "api-scout"), (True, "nobody")])
def test_a_row_that_is_not_enabled_is_refused_in_the_asking_chat(door, monkeypatch, enabled, name):
    _roster(monkeypatch, enabled=enabled)
    assert door.queue.queue_deep_self_review_task("owner:/review", force=True, chat_id=7, reviewer=name) is None
    assert door.pending == []
    [(chat, text, kind)] = door.sent
    assert (chat, kind) == (7, "deep_self_review_unavailable")
    assert text.startswith(f"Deep self-review could not be queued: reviewer {name!r} is not an enabled catalog row")


def test_the_tool_event_hands_the_reviewer_to_the_queue():
    from supervisor.events_runtime_controls import _handle_deep_self_review_request

    handed: list = []
    sup = types.SimpleNamespace(queue_deep_self_review_task=lambda **kw: handed.append(kw))
    _handle_deep_self_review_request({"type": "deep_self_review_request", "reason": "look again",
                                      "reviewer": "api-scout", "model": "openai/fake-reviewer"}, sup)
    _handle_deep_self_review_request({"type": "deep_self_review_request", "reason": "mine"}, sup)
    assert [(kw["reason"], kw["reviewer"]) for kw in handed] == [("look again", "api-scout"), ("mine", "")]


@pytest.mark.parametrize("source, chat, text, reviewer", [
    ("web", 1, "/review api-scout", "api-scout"),  # the Web selector's frame
    ("web", 1, "/review", ""),
    ("telegram", 42, "/review", ""),  # Telegram has no selector: the Main model
])
def test_every_owner_entry_reaches_the_one_queue_door(monkeypatch, source, chat, text, reviewer):
    import server
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "record_inbound_message", lambda *_a, **_k: None)
    state = {"owner_id": 5, "owner_chat_id": 1, "owner_external_id": 5, "owner_external_chat_id": 42}
    handed: list = []
    ctx = types.SimpleNamespace(
        load_state=lambda: dict(state), update_state=lambda fn: fn(dict(state)),
        send_with_budget=lambda *_a, **_k: pytest.fail("no reply: the queue door answers"),
        queue_deep_self_review_task=lambda **kw: handed.append(kw), consciousness=None, kill_workers=None)
    update = {"update_id": 1, "message": {"chat": {"id": chat}, "from": {"id": 5}, "text": text, "source": source}}
    server._handle_bridge_update_batch(types.SimpleNamespace(), [update], 0, ctx, [0])
    assert handed == [{"reason": "owner:/review", "force": True, "chat_id": chat, "reviewer": reviewer}]


def test_the_worker_runs_the_named_row_and_links_its_record(tmp_path, monkeypatch):
    import ouroboros.agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.tools import review_change

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    (drive / "memory").mkdir(parents=True)
    (drive / "logs").mkdir()
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda self: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_k: ([], {}))
    answers: list = []
    monkeypatch.setattr(agent_module, "emit_task_results", lambda *a, **_k: answers.append(a[5]))
    seen: dict = {}

    def system_review(ctx, **kwargs):
        seen.update(kwargs)
        return {"record_id": "rec-1", "report": "REPORT",
                "usage": {"resolved_model": "openai/fake-reviewer", "cost": 0.0}}

    monkeypatch.setattr(review_change, "run_system_review", system_review)
    agent = OuroborosAgent(Env(repo_dir=repo, drive_root=drive))
    task = {"id": "dsr-q", "type": "deep_self_review", "chat_id": 1, "text": "owner:/review", "reviewer": "api-scout"}
    agent.handle_task(task)
    assert seen["reviewer"] == "api-scout"
    assert answers[-1] == "REPORT\n\nReview record: rec-1 (surface=system)"

    # A row disabled between the queue's check and the worker: the typed unavailable text.
    def refused(ctx, **kwargs):
        raise review_change.ReviewChangeArgumentError("reviewer 'api-scout' is not an enabled catalog row (disabled)")

    monkeypatch.setattr(review_change, "run_system_review", refused)
    agent.handle_task({**task, "id": "dsr-q2"})
    assert answers[-1].startswith("❌ Deep self-review unavailable: reviewer 'api-scout' is not an enabled catalog row")
