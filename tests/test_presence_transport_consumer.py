"""A forced Presence final under the Host contract the installed transport actually honours.

The fake transport below mirrors the installed Telegram adapter without importing
Hub code: ``submit`` is ``custody.record_submission`` (a message/deferred turn body
is queued, a deferred turn registers its work_ref) and ``poll`` is
``runtime.process_one_work`` over ``host.poll`` (pending keeps the work; a terminal
result must echo the polled work_ref, closes the work, and only a ``message`` body
is sent). The real /presence/turn and /presence/work endpoints, loop and pipeline
run with a scripted model; nothing leaves the process.

Boundary, not covered here: a polled late result that is itself ``deferred`` (owed
work scheduling further work) is stored truthfully (body and nested work_ref), but
that consumer sends no deferred late body and closes the work, and the poll
response must echo the polled work_ref, so the nested obligation cannot reach it
without a transport protocol change.
"""

from __future__ import annotations

import queue
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline as pipeline, loop
from ouroboros.gateway.host_service import create_host_service_app
from ouroboros.presence_runner import PresenceTurnGate, run_presence_turn
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry
from tests.test_host_service_api import _seed_presence_behavior, _seed_token
from tests.test_presence_forced_delivery import RECORD, _forced, _read

_TOKEN = "presence-token"
_OUTCOMES = {"message", "silent", "tool_delivered", "deferred"}
PARTIAL = "Q1 is ready: 41. Q2 is still being checked."
RESULT = "Q2 is ready too: 43."
NOTE = "sent the table via the transport tool; helper child-7 failed with provider 400"


class _InstalledTransportContract:
    def __init__(self, client: TestClient, binding: str) -> None:
        self.client, self.binding = client, binding
        self.outbox: list[str] = []
        self.open_work: list[str] = []

    def submit(self, body: dict) -> None:
        assert body["ok"] is True and body["status"] == "completed" and body["outcome"] in _OUTCOMES
        if body["text"] and body["outcome"] in {"message", "deferred"}:
            self.outbox.append(body["text"])
        if body["outcome"] == "deferred":
            assert body["work_ref"], "deferred submission requires work_ref"
            self.open_work.append(body["work_ref"])

    def poll(self) -> None:
        for work_ref in list(self.open_work):
            body = self.client.get(f"/presence/work/{work_ref}", params={"binding_id": self.binding},
                                   headers={"X-Skill-Token": _TOKEN}).json()
            if body["status"] == "pending":
                continue
            assert body["status"] in {"completed", "failed", "cancelled"} and body["outcome"] in _OUTCOMES
            assert body.get("work_ref") in {"", work_ref}, "presence Host returned a different work_ref"
            self.open_work.remove(work_ref)
            if body["outcome"] == "message" and body["text"]:
                self.outbox.append(body["text"])


def _scripted_agent(root, monkeypatch, forced, *, handoff=None):
    """The real loop and pipeline for the task it is handed: one tool round, then the ONE forced call."""

    class Agent:
        def handle_task(self, task):
            monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
            monkeypatch.setenv("OUROBOROS_MAX_ROUNDS", "1")
            registry = ToolRegistry(repo_dir=root, drive_root=root)
            ctx = registry._ctx
            ctx.is_direct_chat = bool(task.get("_is_direct_chat"))
            ctx.task_metadata = {**task["metadata"], "inline_max_rounds": 1}
            ctx.task_contract = dict(task["task_contract"])
            if handoff:
                ctx._swarm_handoff_attempt = dict(handoff)
            registry.override_handler("chat_history", lambda *_a, **_kw: "Synthetic history")
            replies = iter([_read(), {"content": forced}])
            monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (next(replies), 0.0))
            task["_skip_post_task_synthesis"] = True
            text, usage, trace = loop.run_llm_loop(
                [{"role": "user", "content": task["text"]}], registry,
                SimpleNamespace(default_model=lambda: "test-model"), root / "logs",
                lambda *_a, **_kw: None, queue.Queue(), task_id=task["id"], drive_root=root,
            )
            events = []
            pipeline.emit_task_results(SimpleNamespace(drive_root=root, repo_dir=root), None, None,
                                       events, task, text, usage, trace, 0.0, root / "logs", ctx=ctx)
            return events

    return Agent()


def _turn(client, binding):
    return client.post("/presence/turn", headers={"X-Skill-Token": _TOKEN}, json={
        "binding_id": binding,
        "event": {
            "source_event_id": "telegram:bot-1:42", "provider": "telegram", "account_id": "bot-1",
            "conversation_id": "room-1", "thread_id": "topic-1", "conversation_key": "ignored",
            "actor": {"platform_actor_id": "user-7"}, "conversation": {"title": "Community"},
            "message": {"message_id": "42"}, "text": "Please compile Q1 and Q2",
        },
    }).json()


@pytest.mark.parametrize("turn_forced,spoken_now", [
    (_forced("message", PARTIAL), [PARTIAL]),  # a declared partial beside owed work
    (_forced("deferred", PARTIAL), [PARTIAL]),
    (_forced("tool_delivered", NOTE), []),  # the note is context, never a reply
    (RECORD, []),  # an undeclared internal record says nothing new
], ids=["message", "deferred", "tool_delivered", "undeclared"])
def test_the_installed_transport_gets_the_partial_now_and_the_owed_result_later(
        tmp_path, monkeypatch, turn_forced, spoken_now):
    _seed_token(tmp_path, skill="telegram-bot", token=_TOKEN, permissions=["presence"],
                manifest_permissions=["presence"])
    binding = _seed_presence_behavior(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    captured = {}

    def runner(**kwargs):
        def factory(**_kwargs):
            agent = _scripted_agent(tmp_path, monkeypatch, turn_forced,
                                    handoff={"status": "scheduled", "task_id": "work-9"})
            original = agent.handle_task

            def handle(task):
                captured["turn"] = task
                # The admitted promotion the handoff names, canonical from admission.
                write_task_result(tmp_path, "work-9", "scheduled", delegation_role="root",
                                  root_task_id="work-9", description="Compile Q2",
                                  metadata={"presence": dict(task["metadata"]["presence"])})
                return original(task)

            agent.handle_task = handle
            return agent

        return run_presence_turn(repo_dir=repo, drive_root=tmp_path, agent_factory=factory,
                                 gate=PresenceTurnGate(1), **kwargs)

    with TestClient(create_host_service_app(tmp_path, presence_runner=runner)) as client:
        transport = _InstalledTransportContract(client, binding)
        transport.submit(_turn(client, binding))
        assert transport.outbox == spoken_now and transport.open_work == ["work-9"]  # speech AND custody

        transport.poll()  # the owed work has not run: nothing is sent and it stays owed
        assert transport.outbox == spoken_now and transport.open_work == ["work-9"]

        turn = captured["turn"]
        work = {"id": "work-9", "type": "task", "chat_id": turn["chat_id"], "text": "Compile Q2",
                "delegation_role": "root", "root_task_id": "work-9", "_presence_origin": True,
                "metadata": {"presence": dict(turn["metadata"]["presence"])},
                "task_contract": dict(turn["task_contract"])}
        _scripted_agent(tmp_path, monkeypatch, _forced("message", RESULT)).handle_task(work)
        transport.poll()

    assert transport.outbox == spoken_now + [RESULT] and transport.open_work == []
    assert not any(RECORD in text or NOTE in text for text in transport.outbox)
