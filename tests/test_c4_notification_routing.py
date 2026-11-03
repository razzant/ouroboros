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


pytestmark = pytest.mark.serial


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
    def test_task_bound_job_reports_to_its_chat(self, monkeypatch):
        sent = _capture_sends(monkeypatch)

        async def runner():
            return "ok"

        _run(q.run_lifecycle_job(
            kind="review", target="alpha", runner=runner, chat_id=17,
            options=q.LifecycleJobOptions(presentation={
                "group_id": "task:root-1:alpha",
                "task_id": "child-1",
                "root_task_id": "root-1",
                "origin_task_id": "child-1",
                "origin_root_task_id": "root-1",
                "presentation_owner_task_id": "root-1",
            }),
        ))
        assert sent, "lifecycle notifications must fire"
        assert {cid for cid, _, _ in sent} == {17}
        lifecycle = sent[0][2]["progress_meta"]["lifecycle"]
        assert lifecycle["chat_id"] == 17
        assert lifecycle["status"] == "queued"
        assert lifecycle["group_id"] == "task:root-1:alpha"
        assert lifecycle["task_id"] == "child-1"
        assert lifecycle["root_task_id"] == "root-1"
        assert lifecycle["presentation_owner_task_id"] == "root-1"

    def test_unbound_job_stays_on_the_panel_chat_zero(self, monkeypatch):
        sent = _capture_sends(monkeypatch)

        async def runner():
            return "ok"

        _run(q.run_lifecycle_job(kind="install", target="beta", runner=runner))
        assert sent and {cid for cid, _, _ in sent} == {0}

    def test_negative_a2a_chat_never_reaches_a_human_stream(self, monkeypatch):
        sent = _capture_sends(monkeypatch)

        async def runner():
            return "ok"

        _run(q.run_lifecycle_job(kind="review", target="gamma", runner=runner, chat_id=-1001))
        assert sent and {cid for cid, _, _ in sent} == {0}

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

    def test_panel_initiator_gets_its_pointer_too(self, monkeypatch):
        # F10: chat 0 is the PANEL, not "no chat". The old `if not requested_chat`
        # guard silently dropped exactly this ack — a panel caller duplicating a
        # task-bound job saw nothing at all.
        sent = _capture_sends(monkeypatch)
        existing = q.LifecycleJob(id="skill-job-3", kind="review", target="zeta",
                                  dedupe_key="k3", status="running", chat_id=17)
        q._dedupe_jobs["k3"] = existing

        async def runner():  # pragma: no cover - never runs
            return "ok"

        with pytest.raises(q.DuplicateLifecycleJobError):
            _run(q.run_lifecycle_job(kind="review", target="zeta", runner=runner,
                                     dedupe_key="k3", chat_id=0))
        pointers = [entry for entry in sent
                    if "lifecycle_pointer" in entry[2].get("progress_meta", {})]
        assert len(pointers) == 1
        assert pointers[0][0] == 0
        assert pointers[0][2]["progress_meta"]["lifecycle_pointer"]["chat_id"] == 17

    def test_duplicate_from_the_same_chat_gets_no_pointer(self, monkeypatch):
        sent = _capture_sends(monkeypatch)
        existing = q.LifecycleJob(id="skill-job-2", kind="review", target="eps",
                                  dedupe_key="k2", status="running", chat_id=17)
        q._dedupe_jobs["k2"] = existing

        async def runner():  # pragma: no cover
            return "ok"

        with pytest.raises(q.DuplicateLifecycleJobError):
            _run(q.run_lifecycle_job(kind="review", target="eps", runner=runner,
                                     dedupe_key="k2", chat_id=17))
        assert not sent


class TestInterruptedRowChat:
    def test_interrupted_row_carries_the_payload_chat(self, tmp_path):

        from ouroboros.skill_review_runner import _append_interrupted_review_progress

        _append_interrupted_review_progress(
            tmp_path, "alpha", {"job_id": "job-1", "chat_id": 17},
            ts="2026-08-11T00:00:00Z")
        rows = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")
        assert rows[-1]["chat_id"] == 17

    def test_interrupted_row_negative_chat_falls_back_to_panel(self, tmp_path):

        from ouroboros.skill_review_runner import _append_interrupted_review_progress

        _append_interrupted_review_progress(
            tmp_path, "alpha", {"job_id": "job-1", "chat_id": -5},
            ts="2026-08-11T00:00:00Z")
        rows = dialogue_test.log_rows(tmp_path / "logs" / "progress.jsonl")
        assert rows[-1]["chat_id"] == 0


class TestNotificationRoute:
    """The ONE normalizer: membership decides, never truthiness (F10)."""

    def test_zero_is_the_panel_route_and_negatives_are_suppressed(self):
        from supervisor.message_bus import notification_chat_route

        assert notification_chat_route(7) == 7
        # 0 is the Skill Review panel — a real destination, not "no chat".
        assert notification_chat_route(0) == 0
        # Absent candidates fall through to the next one; A2A ids are skipped.
        assert notification_chat_route(None, "", 5) == 5
        assert notification_chat_route(-42, 1) == 1
        assert notification_chat_route("junk", 0) == 0
        # Nothing deliverable is None — distinct from the panel's 0.
        assert notification_chat_route(-42, None) is None
        assert notification_chat_route() is None


class TestReaperIncidentChat:
    def test_task_chat_wins_owner_is_fallback(self):
        from supervisor.task_reaper import _incident_chat_id

        assert _incident_chat_id({"chat_id": 7}, 1) == 7
        # A task BOUND to the Skill Review panel keeps its incident there: the
        # old `> 0` test re-routed it to the owner chat, and the `if chat_id:`
        # send guard then dropped it entirely.
        assert _incident_chat_id({"chat_id": 0}, 1) == 0
        assert _incident_chat_id({}, 1) == 1
        assert _incident_chat_id(None, 1) == 1

    def test_project_binding_beats_the_row_chat(self, tmp_path):
        """A reaper incident is a DIRECT send, so the binding must win HERE: a
        task converted into a project mid-run keeps its origin chat on the row."""
        from types import SimpleNamespace

        from ouroboros.projects_registry import bind_task_to_project
        from supervisor.task_reaper import _incident_chat_id

        bind_task_to_project(tmp_path, "root-inc", "reap-proj", 5150, origin={"absent": "system"})
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path)

        assert _incident_chat_id({"id": "root-inc", "chat_id": 1}, 1, ctx) == 5150
        # A child is never bound itself; it inherits the room through lineage.
        assert _incident_chat_id(
            {"id": "child-inc", "root_task_id": "root-inc", "chat_id": 1}, 1, ctx
        ) == 5150
        # An unbound task keeps today's order: its own chat, owner as fallback.
        assert _incident_chat_id({"id": "unbound-inc", "chat_id": 7}, 1, ctx) == 7
        assert _incident_chat_id({"id": "unbound-inc"}, 1, ctx) == 1

    def test_a2a_binding_falls_through_to_the_task_chat(self, tmp_path):
        """Suppression still applies to the new first candidate: a synthetic
        (negative) bound chat is skipped, it does not silence the notice."""
        from types import SimpleNamespace

        from ouroboros.projects_registry import bind_task_to_project
        from supervisor.task_reaper import _incident_chat_id

        bind_task_to_project(tmp_path, "a2a-inc", "a2a-proj", -1001, origin={"absent": "system"})
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path)

        assert _incident_chat_id({"id": "a2a-inc", "chat_id": 7}, 1, ctx) == 7
        assert _incident_chat_id({"id": "a2a-inc"}, 1, ctx) == 1

    def test_negative_task_chat_never_reaches_a_human_stream(self):
        from supervisor.task_reaper import _incident_chat_id

        assert _incident_chat_id({"chat_id": -42}, 1) == 1
        # No owner chat configured and an A2A task chat: no deliverable route at
        # all — reported as None, not as the panel.
        assert _incident_chat_id({"chat_id": -42}, 0) is None
        assert _incident_chat_id({"chat_id": "junk"}, -3) is None


class TestCascadeDeliveryRow:
    """A cancel cascade whose root already left the live maps borrows a live
    descendant's row for its lineage chat. Reading that chat for TRUTH skipped
    every descendant homed in the hidden partition (chat 0), so the cascade
    silently had no row and the notice went nowhere."""

    def _queue(self, pending, running):
        from supervisor import queue as real_queue

        class _Q:
            PENDING = pending
            RUNNING = running
            _is_descendant_of = staticmethod(real_queue._is_descendant_of)

        return _Q

    def test_a_descendant_homed_in_the_partition_is_returned(self):
        from supervisor.cancel_publication import _cascade_delivery_row_locked

        child = {"id": "kid", "root_task_id": "root-9", "chat_id": 0}
        assert _cascade_delivery_row_locked(self._queue([child], {}), "root-9") == child
        assert _cascade_delivery_row_locked(
            self._queue([], {"kid": {"task": child}}), "root-9"
        ) == child

    def test_a_descendant_without_a_chat_is_still_skipped(self):
        from supervisor.cancel_publication import _cascade_delivery_row_locked

        child = {"id": "kid", "root_task_id": "root-9"}
        assert _cascade_delivery_row_locked(self._queue([child], {}), "root-9") == {}
