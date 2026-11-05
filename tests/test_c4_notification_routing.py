"""C4 (poltergeist phase C): lifecycle/incident notifications reach the RIGHT chat.

Task-bound skill jobs report to the task's chat; truly unbound jobs stay on the
Skill Review panel (chat 0 — note Main is chat 1); reaper incidents go to the
task's own chat with owner chat only as the absent-binding fallback; negative
A2A ids never reach human streams; a duplicate lifecycle initiator from another
chat gets a typed pointer ack.
"""

from __future__ import annotations

from tests import _dialogue as dialogue_test

import asyncio

import pytest

import ouroboros.skill_lifecycle_queue as q




@pytest.fixture(autouse=True)
def _reset_queue_state():
    q._dedupe_jobs.clear()
    q._events.clear()
    yield
    q._dedupe_jobs.clear()
    q._events.clear()


def _run(coro):
    return asyncio.run(coro)


def _capture_sends(monkeypatch):
    sent = []
    import supervisor.message_bus as bus

    monkeypatch.setattr(bus, "send_with_budget",
                        lambda cid, text, **kw: sent.append((cid, str(text), kw)))
    return sent


class TestLifecycleChatRouting:

    def test_duplicate_initiator_gets_a_typed_pointer_in_its_own_chat(self, monkeypatch):
        sent = _capture_sends(monkeypatch)
        existing = q.LifecycleJob(id="skill-job-1", kind="review", target="delta",
                                  dedupe_key="k", status="running", chat_id=17)
        q._dedupe_jobs["k"] = existing

        async def runner():  # pragma: no cover - never runs
            return "ok"

        with pytest.raises(q.DuplicateLifecycleJobError):
            _run(q.run_lifecycle_job(kind="review", target="delta", runner=runner,
                                     dedupe_key="k", chat_id=25))
        pointers = [entry for entry in sent
                    if "lifecycle_pointer" in entry[2].get("progress_meta", {})]
        assert len(pointers) == 1
        cid, _text, kwargs = pointers[0]
        assert cid == 25  # the DUPLICATE caller's own chat
        assert kwargs["task_id"] == ""  # pointer ack never becomes a second task card
        pointer = kwargs["progress_meta"]["lifecycle_pointer"]
        assert pointer["job_id"] == "skill-job-1"
        assert pointer["chat_id"] == 17  # where the routing actually lives


class TestInterruptedRowChat:
    def test_interrupted_row_carries_the_payload_chat(self, tmp_path):

        from ouroboros.skill_review_runner import _append_interrupted_review_progress

        _append_interrupted_review_progress(
            tmp_path, "alpha", {"job_id": "job-1", "chat_id": 17},
            ts="2026-08-11T00:00:00Z")
        rows = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")
        assert rows[-1]["chat_id"] == 17
