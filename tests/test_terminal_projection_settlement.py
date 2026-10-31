"""#1154 — a root's Project row and Main's single mirror are ONE owed unit.

Before this, ``task_done`` appended the canonical Project row (or deferred it
behind an open post-task synthesis) and, INDEPENDENTLY, enqueued Main's mirror
right away. So Main heard about a run that had not finished finalizing, and a
crash between the two effects lost whichever half had not happened yet.

The obligation is now a durable row (``canonical_terminal_projection_ready``)
persisted before the effects and cleared only when Main is confirmed delivered
or positively INELIGIBLE. One continuation runs both halves outside the result
lock, ``task_done`` and the post-task callback share it, and existing startup
and off-loop maintenance passes retry it without cognition or a new timer/store.

The disclosed limits stay: the outbox is bounded and external delivery is
at-least-once — this is not exactly-once.
"""
from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest

from ouroboros.post_task_checkpoint import (
    SETTLEMENT_DEFERRED,
    SETTLEMENT_NONE,
    SETTLEMENT_SETTLED,
    settle_terminal_projection,
)
from ouroboros.project_dialogue import append_terminal_task_projection
from ouroboros.task_results import STATUS_COMPLETED, load_task_result, write_task_result
from tests._store_shared import seed_result

DONE = {"chat_id": 1, "status": "completed", "outcome_axes": {"execution": {"status": "ok"}}}


def _chat_rows(root: pathlib.Path):
    from tests._dialogue import chat_rows

    return chat_rows(root)


def _project_rows(root: pathlib.Path, task_id: str):
    return [row for row in _chat_rows(root)
            if row.get("task_id") == task_id and row.get("type") == "task_summary"]


@pytest.fixture()
def project_root(tmp_path, monkeypatch):
    """A registered project whose room actually holds the run's work."""
    from ouroboros.projects_registry import bind_task_to_project, create_project

    project = create_project(tmp_path, "launch", name="Launch")
    bind_task_to_project(tmp_path, "root-1", project["id"], project["chat_id"],
                         origin={"absent": "system"})
    queued: list[dict] = []
    def send(root, event, **_kw):
        from supervisor.terminal_delivery import register_delivery
        queued.append(dict(event))
        register_delivery(root, event["delivery_id"], emitted={**event, "chat_id": 1})
        return True

    monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery", send)
    from supervisor.event_observation import ObservedEventQueue
    bus = ObservedEventQueue()
    monkeypatch.setattr("supervisor.workers.get_event_q", lambda *, create=True: bus)
    task = {"id": "root-1", "project_id": project["id"], "title": "Ship release",
            "chat_id": project["chat_id"], "root_task_id": "root-1"}
    return SimpleNamespace(root=tmp_path, project=project, task=task, queued=queued, send=send, bus=bus)


def _store(root, task_id="root-1", *, project_id="launch", phase=None, **over):
    fields = {"root_task_id": task_id, "project_id": project_id,
              "result": "Owner already has the answer", **over}
    if phase is not None:
        fields["root_phase_checkpoint"] = {"post_task_synthesis": phase}
    return write_task_result(root, task_id, STATUS_COMPLETED, **fields)


class TestOwedBeforeEffects:
    def test_an_open_post_task_defers_both_halves_and_keeps_the_obligation(self, project_root):
        stored = _store(project_root.root, phase="running")
        assert not append_terminal_task_projection(
            project_root.root, "root-1", project_root.task, stored, DONE)
        owed = load_task_result(project_root.root, "root-1")["canonical_terminal_projection_ready"]
        assert owed["summary_id"] == "task-terminal:root-1"
        assert _project_rows(project_root.root, "root-1") == []
        # The continuation refuses to run either half while the synthesis is open:
        # the early answer stays in the Project thread, and Main hears nothing.
        assert settle_terminal_projection(
            project_root.root, "root-1", task=project_root.task) == SETTLEMENT_DEFERRED
        assert project_root.queued == []
        assert _project_rows(project_root.root, "root-1") == []

    def test_a_settled_root_appends_the_row_and_still_owes_main(self, project_root):
        stored = _store(project_root.root)
        assert append_terminal_task_projection(
            project_root.root, "root-1", project_root.task, stored, DONE)
        record = load_task_result(project_root.root, "root-1")
        # BOTH: the Project receipt AND the obligation the mirror still stands on.
        assert record["canonical_terminal_projection"]["summary_id"] == "task-terminal:root-1"
        assert record["canonical_terminal_projection_ready"]["summary_id"] == "task-terminal:root-1"
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert project_root.queued == []

    def test_a_child_owes_main_nothing_and_carries_no_obligation(self, tmp_path):
        child = {"id": "child-1", "parent_task_id": "root-1", "root_task_id": "root-1",
                 "delegation_role": "subagent", "chat_id": 41}
        stored = write_task_result(tmp_path, "child-1", STATUS_COMPLETED,
                                   parent_task_id="root-1", root_task_id="root-1",
                                   delegation_role="subagent")
        assert append_terminal_task_projection(tmp_path, "child-1", child, stored,
                                               {"chat_id": 41, "status": "completed"})
        record = load_task_result(tmp_path, "child-1")
        assert "canonical_terminal_projection_ready" not in record
        assert settle_terminal_projection(tmp_path, "child-1") == SETTLEMENT_NONE


class TestSettlementClosesTheUnit:
    def test_the_post_task_callback_runs_both_halves_exactly_once(self, project_root):
        _store(project_root.root, phase="running")
        stored = load_task_result(project_root.root, "root-1")
        append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)
        _store(project_root.root, phase="completed")

        assert settle_terminal_projection(
            project_root.root, "root-1", task=project_root.task) == SETTLEMENT_SETTLED
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert len(project_root.queued) == 1
        assert project_root.queued[0]["chat_id"] == 1
        assert project_root.queued[0]["delivery_id"] == "project-completion:root-1"
        record = load_task_result(project_root.root, "root-1")
        assert record["canonical_terminal_projection_ready"] is None

        # Repeating the continuation is a no-op: nothing owed, nothing appended.
        assert settle_terminal_projection(
            project_root.root, "root-1", task=project_root.task) == SETTLEMENT_NONE
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert len(project_root.queued) == 1

    def test_a_duplicate_settlement_reuses_the_one_delivery_id(self, project_root):
        _store(project_root.root)
        stored = load_task_result(project_root.root, "root-1")
        append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)
        for _ in range(3):
            settle_terminal_projection(project_root.root, "root-1", task=project_root.task)
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert {row["delivery_id"] for row in project_root.queued} == {"project-completion:root-1"}

    def test_a_positively_ineligible_root_clears_the_obligation_without_sending(self, tmp_path, monkeypatch):
        queued: list[dict] = []
        monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                            lambda _root, event, **_kw: queued.append(dict(event)) or True)
        # A workspace-derived project id has no registry row, so there is no room
        # to open and Main is owed NOTHING — a positive answer, not an unknown.
        stored = write_task_result(tmp_path, "root-2", STATUS_COMPLETED,
                                   root_task_id="root-2", project_id="proj_deadbeef1234")
        task = {"id": "root-2", "project_id": "proj_deadbeef1234", "chat_id": 7}
        append_terminal_task_projection(tmp_path, "root-2", task, stored,
                                        {"chat_id": 7, "status": "completed"})
        assert settle_terminal_projection(tmp_path, "root-2", task=task) == SETTLEMENT_SETTLED
        assert queued == []
        assert load_task_result(tmp_path, "root-2")["canonical_terminal_projection_ready"] is None


class TestPartialWritesAndRetry:
    def test_a_registration_that_is_not_durable_keeps_the_obligation_owed(self, project_root, monkeypatch):
        _store(project_root.root)
        stored = load_task_result(project_root.root, "root-1")
        append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)

        # The live send may still go out; what is NOT durable is the owed row, so
        # a crash before delivery would lose it. UNKNOWN, never a cleared debt.
        monkeypatch.setattr("supervisor.terminal_delivery.register_pending_delivery",
                            lambda *_a, **_kw: False)
        assert settle_terminal_projection(
            project_root.root, "root-1", task=project_root.task) == SETTLEMENT_DEFERRED
        record = load_task_result(project_root.root, "root-1")
        assert record["canonical_terminal_projection_ready"]["summary_id"] == "task-terminal:root-1"

        # A later due pass heals; cache failure consumes backoff, not tick-rate retries.
        ready = record["canonical_terminal_projection_ready"]
        ready["main_delivery"]["last_attempt_at"] = "2000-01-01T00:00:00Z"
        write_task_result(project_root.root, "root-1", "completed", canonical_terminal_projection_ready=ready)
        monkeypatch.undo()
        monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                            project_root.send)
        assert settle_terminal_projection(
            project_root.root, "root-1", task=project_root.task, event_queue=project_root.bus) == SETTLEMENT_SETTLED
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert load_task_result(project_root.root, "root-1")["canonical_terminal_projection_ready"] is None

    def test_an_unreadable_eligibility_answer_is_unknown_not_ineligible(self, project_root, monkeypatch):
        _store(project_root.root)
        stored = load_task_result(project_root.root, "root-1")
        append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)

        def _boom(*_a, **_kw):
            raise OSError("registry unreadable")

        monkeypatch.setattr("ouroboros.projects_registry.task_presentation_snapshot", _boom)
        assert settle_terminal_projection(
            project_root.root, "root-1", task=project_root.task) == SETTLEMENT_DEFERRED
        assert load_task_result(
            project_root.root, "root-1")["canonical_terminal_projection_ready"] is not None

    def test_clearing_is_a_compare_and_swap_against_the_current_marker_and_phase(self, project_root):
        from ouroboros.post_task_checkpoint import _clear_terminal_projection_obligation

        _store(project_root.root)
        stored = load_task_result(project_root.root, "root-1")
        append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)
        # A stale summary id belongs to a different terminal and clears nothing.
        expected = load_task_result(project_root.root, "root-1")
        stale = {**expected, "canonical_terminal_projection_ready": {"summary_id": "task-terminal:other"}}
        _clear_terminal_projection_obligation(project_root.root, "root-1", stale, "ineligible")
        assert load_task_result(
            project_root.root, "root-1")["canonical_terminal_projection_ready"] is not None
        # A post-task pass that re-opened also blocks the clear.
        _store(project_root.root, phase="running")
        _clear_terminal_projection_obligation(project_root.root, "root-1", expected, "ineligible")
        assert load_task_result(
            project_root.root, "root-1")["canonical_terminal_projection_ready"] is not None
        # Matching marker and settled phase: the obligation closes.
        _store(project_root.root, phase="completed")
        expected = load_task_result(project_root.root, "root-1")
        _clear_terminal_projection_obligation(project_root.root, "root-1", expected, "ineligible")
        assert load_task_result(
            project_root.root, "root-1")["canonical_terminal_projection_ready"] is None

    def test_readiness_is_on_disk_before_project_append_and_no_result_lock_is_held(self, project_root, monkeypatch):
        from ouroboros import project_dialogue as dialogue

        stored = _store(project_root.root)
        original = dialogue.append_canonical_task_summary

        def append(root, row):
            current = load_task_result(root, "root-1")
            assert current["canonical_terminal_projection_ready"]["token"] == row["terminal_projection_token"]
            # A real reentrant result write would time out if append held its lock.
            write_task_result(root, "root-1", "completed", reentrant_probe=True)
            return original(root, row)

        monkeypatch.setattr(dialogue, "append_canonical_task_summary", append)
        assert append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)

    def test_initial_marker_write_crash_is_discovered_without_restart(self, project_root, monkeypatch):
        from ouroboros import terminal_projection as projection

        _store(project_root.root, chat_id=project_root.project["chat_id"])
        original = projection.write_task_result
        monkeypatch.setattr(projection, "write_task_result", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("crash")))
        assert settle_terminal_projection(project_root.root, "root-1") == SETTLEMENT_DEFERRED
        stored = load_task_result(project_root.root, "root-1", strict=True)
        assert "canonical_terminal_projection_ready" not in stored
        assert stored["canonical_terminal_projection_origin"] == "terminal_transition"
        monkeypatch.setattr(projection, "write_task_result", original)
        assert projection.reconcile_terminal_projections(project_root.root) == 1
        assert len(_project_rows(project_root.root, "root-1")) == len(project_root.queued) == 1

    def test_append_receipt_crash_dedupes_project_by_the_durable_token(self, project_root, monkeypatch):
        from ouroboros import terminal_projection as projection
        from ouroboros import project_dialogue as dialogue

        _store(project_root.root)
        original = dialogue.append_canonical_task_summary

        def append_then_crash(root, row):
            assert original(root, row)
            raise OSError("process ended after append, before result receipt")

        monkeypatch.setattr(dialogue, "append_canonical_task_summary", append_then_crash)
        assert settle_terminal_projection(project_root.root, "root-1", task=project_root.task) == SETTLEMENT_DEFERRED
        published_at = _project_rows(project_root.root, "root-1")[0]["ts"]
        monkeypatch.setattr(projection, "utc_now_iso", lambda: "2099-01-01T00:00:00Z")
        monkeypatch.setattr(dialogue, "append_canonical_task_summary", original)
        assert projection.reconcile_terminal_projections(project_root.root) == 1
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert load_task_result(project_root.root, "root-1")["canonical_terminal_projection"]["written_at"] == published_at

    def test_queue_failure_keeps_canonical_debt_and_retry_schedule(self, project_root, monkeypatch):
        from supervisor.terminal_delivery import pending_deliveries
        from ouroboros.terminal_projection import reconcile_terminal_projections

        _store(project_root.root)
        monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                            lambda *_a, **_kw: (_ for _ in ()).throw(OSError("queue unavailable")))
        assert settle_terminal_projection(project_root.root, "root-1", task=project_root.task) == SETTLEMENT_DEFERRED
        assert [r["delivery_id"] for r in pending_deliveries(project_root.root)] == ["project-completion:root-1"]
        current = load_task_result(project_root.root, "root-1")
        saved = current["canonical_terminal_projection_ready"]["main_delivery"]
        assert saved["attempts"] == 1
        assert saved["event"]["chat_id"] == 1
        # A missing cache entry does not reset timing or cause a tick-rate send.
        (project_root.root / "state" / "terminal_deliveries.json").write_text('{"delivered": [], "pending": {}}')
        assert reconcile_terminal_projections(project_root.root) == 0

    def test_stale_retirement_cannot_clear_new_token_or_attempt(self, project_root):
        from ouroboros.terminal_projection import clear_terminal_projection_obligation

        _store(project_root.root)
        append_terminal_task_projection(project_root.root, "root-1", project_root.task,
                                        load_task_result(project_root.root, "root-1"), DONE)
        expected = load_task_result(project_root.root, "root-1")
        newer = {**expected["canonical_terminal_projection_ready"], "token": "new-attempt"}
        write_task_result(project_root.root, "root-1", "completed", task_attempt=2,
                          canonical_terminal_projection_ready=newer)
        assert not clear_terminal_projection_obligation(project_root.root, "root-1", expected, "owed")
        assert load_task_result(project_root.root, "root-1")["canonical_terminal_projection_ready"] == newer

    @pytest.mark.parametrize("filename", ["projects.json", "task_project_bindings.json"])
    def test_corrupt_membership_is_unknown_not_ineligible(self, project_root, filename):
        from ouroboros import projects_registry as registry

        _store(project_root.root, _is_direct_chat=True)
        path = registry._registry_path(project_root.root) if filename == "projects.json" else registry._bindings_path(project_root.root)
        # For the registry case test ordinary task membership; direct turns read
        # their binding first and are positively native in this fixture.
        if filename == "projects.json":
            _store(project_root.root, _is_direct_chat=False)
        path.write_text("{unreadable", encoding="utf-8")
        assert settle_terminal_projection(project_root.root, "root-1", task=project_root.task) == SETTLEMENT_DEFERRED
        assert load_task_result(project_root.root, "root-1")["canonical_terminal_projection_ready"]

    def test_concurrent_callbacks_append_and_register_one_projection(self, project_root):
        from concurrent.futures import ThreadPoolExecutor

        _store(project_root.root)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: settle_terminal_projection(
                project_root.root, "root-1", task=project_root.task), range(2)))
        assert SETTLEMENT_SETTLED in results
        assert len(_project_rows(project_root.root, "root-1")) == len(project_root.queued) == 1


class TestRestartRecovery:
    def test_startup_leaves_a_pause_an_earlier_version_saved_for_the_maintenance_pass(self, tmp_path, monkeypatch):
        from ouroboros import agent_task_pipeline as pipeline, post_task_synthesis

        _store(tmp_path, phase="paused")
        monkeypatch.setattr(pipeline, "_settle_terminal_projection",
                            lambda *_a, **_kw: pytest.fail("paused synthesis has not settled"))
        monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                            lambda *_a, **_kw: pytest.fail("a saved Pause never runs"))
        monkeypatch.setattr(post_task_synthesis, "end_saved_late_phase_pause",
                            lambda *_a, **_kw: pytest.fail("boot never converts; the maintenance pass does"))
        assert pipeline.recover_pending_root_post_task_synthesis(tmp_path) == 0
        checkpoint = pipeline.load_task_result(tmp_path, "root-1")["root_phase_checkpoint"]
        assert checkpoint["post_task_synthesis"] == "paused"

    @pytest.mark.parametrize("ready", [False, True])
    def test_startup_selects_publication_debt_before_mail_capture(self, project_root, monkeypatch, ready):
        from ouroboros import agent_task_pipeline as pipeline, task_custody
        from ouroboros.task_result_schema import stamp_task_result_schema

        _store(project_root.root, phase="completed", chat_id=project_root.project["chat_id"])
        if ready:
            stored = load_task_result(project_root.root, "root-1")
            append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)
        # Historical terminals, retired publication obligations and children
        # share the startup scan but owe no Project/Main publication work.
        for index in range(64):
            tid = f"history-{index}"
            row = {"task_id": tid, "status": "completed"}
            if index % 3 == 1:
                row.update(canonical_terminal_projection_origin="terminal_transition",
                           canonical_terminal_projection={"main_disposition": "owed"})
            elif index % 3 == 2:
                row.update(parent_task_id="root-1", root_task_id="root-1",
                           canonical_terminal_projection_origin="terminal_transition")
            seed_result(project_root.root, tid, stamp_task_result_schema(row))
        calls = []
        for name in ("capture_unread_mail", "capture_owner_mail"):
            original = getattr(task_custody, name)

            def capture(root, tid, *args, _name=name, _original=original, **kwargs):
                calls.append((_name, tid))
                return _original(root, tid, *args, **kwargs)

            monkeypatch.setattr(task_custody, name, capture)
        monkeypatch.setattr(pipeline, "_run_post_task_processing_async",
                            lambda *_a, **_kw: pytest.fail("publication buys no cognition"))

        assert pipeline.recover_pending_root_post_task_synthesis(project_root.root) == 0
        assert calls and {tid for _name, tid in calls} == {"root-1"}
        assert len(_project_rows(project_root.root, "root-1")) == len(project_root.queued) == 1
        assert load_task_result(project_root.root, "root-1")["canonical_terminal_projection_ready"] is None
        calls.clear()
        assert pipeline.recover_pending_root_post_task_synthesis(project_root.root) == 0
        assert calls == []

    def test_the_existing_scan_retries_an_owed_terminal_row_without_paid_cognition(
        self, project_root, monkeypatch,
    ):
        from ouroboros import agent_task_pipeline as pipeline

        _store(project_root.root, phase="completed")
        stored = load_task_result(project_root.root, "root-1")
        append_terminal_task_projection(project_root.root, "root-1", project_root.task, stored, DONE)
        # Simulate a crash between the Project row and Main: the obligation is
        # still on disk, the mirror never went out.
        assert load_task_result(
            project_root.root, "root-1")["canonical_terminal_projection_ready"] is not None
        assert project_root.queued == []

        monkeypatch.setattr(
            pipeline, "_run_post_task_processing_async",
            lambda *_a, **_kw: pytest.fail("restart settlement must not buy cognition"))
        # The scan counts RECOVERED SYNTHESES; a settlement is bookkeeping, not one.
        assert pipeline.recover_pending_root_post_task_synthesis(
            project_root.root, repo_dir=project_root.root / "repo") == 0
        assert len(project_root.queued) == 1
        assert len(_project_rows(project_root.root, "root-1")) == 1
        assert load_task_result(
            project_root.root, "root-1")["canonical_terminal_projection_ready"] is None
        # Idempotent across restarts.
        assert pipeline.recover_pending_root_post_task_synthesis(
            project_root.root, repo_dir=project_root.root / "repo") == 0
        assert len(project_root.queued) == 1

    def test_a_split_replica_can_neither_resurrect_nor_replace_the_canonical_facts(self):
        from ouroboros.post_task_checkpoint import project_replica_task_result_fields

        canonical = {
            "canonical_terminal_projection": {"summary_id": "task-terminal:root-1"},
            "canonical_terminal_projection_ready": None,
        }
        replica = {
            "canonical_terminal_projection": {"summary_id": "task-terminal:imposter"},
            "canonical_terminal_projection_ready": {"summary_id": "task-terminal:root-1"},
            "result": "worker copy",
        }
        overlay = project_replica_task_result_fields(canonical, replica)
        assert "canonical_terminal_projection" not in overlay
        assert "canonical_terminal_projection_ready" not in overlay
        assert overlay["result"] == "worker copy"


class TestNoLeakage:
    def test_a_project_native_direct_turn_is_never_mirrored_into_main(self, project_root):
        from ouroboros.project_dialogue import (
            MAIN_MIRROR_INELIGIBLE, project_completion_delivery_outcome,
        )

        stored = _store(project_root.root, _is_direct_chat=True)
        assert project_completion_delivery_outcome(
            project_root.root, {"_is_direct_chat": True}, "root-1",
            project_root.task, stored, DONE) == (MAIN_MIRROR_INELIGIBLE, False)
        assert project_root.queued == []

    def test_an_unbound_run_whose_room_was_registered_late_is_ineligible(self, tmp_path, monkeypatch):
        from ouroboros.project_dialogue import (
            MAIN_MIRROR_INELIGIBLE, project_completion_delivery_outcome,
        )
        from ouroboros.projects_registry import create_project

        queued: list[dict] = []
        monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery",
                            lambda _root, event, **_kw: queued.append(dict(event)) or True)
        create_project(tmp_path, "late", name="Late")
        task = {"id": "root-3", "project_id": "late", "chat_id": 99}
        stored = write_task_result(tmp_path, "root-3", STATUS_COMPLETED,
                                   root_task_id="root-3", project_id="late")
        assert project_completion_delivery_outcome(
            tmp_path, {}, "root-3", task, stored,
            {"status": "completed"}) == (MAIN_MIRROR_INELIGIBLE, False)
        assert queued == []

    def test_a_child_never_reaches_the_main_mirror_seam(self, project_root):
        from ouroboros.project_dialogue import (
            MAIN_MIRROR_INELIGIBLE, project_completion_delivery_outcome,
        )

        child = {**project_root.task, "id": "child-1", "parent_task_id": "root-1",
                 "delegation_role": "subagent"}
        stored = write_task_result(project_root.root, "child-1", STATUS_COMPLETED,
                                   parent_task_id="root-1", root_task_id="root-1",
                                   delegation_role="subagent", project_id="launch")
        assert project_completion_delivery_outcome(
            project_root.root, {}, "child-1", child, stored, DONE) == (MAIN_MIRROR_INELIGIBLE, False)
        assert project_root.queued == []


def test_periodic_maintenance_retries_terminal_only_without_synthesis(project_root, monkeypatch):
    from ouroboros import server_maintenance as maintenance
    from ouroboros import agent_task_pipeline as pipeline

    _store(project_root.root, phase='completed', chat_id=project_root.project['chat_id'])
    monkeypatch.setattr(pipeline, '_run_post_task_processing_async',
                        lambda *_a, **_kw: pytest.fail('maintenance must not run synthesis'))
    monkeypatch.setattr('supervisor.task_lifecycle.sweep_cancel_intents', lambda: [])
    monkeypatch.setattr('supervisor.terminal_delivery.replay_pending_deliveries', lambda *_a: [])
    monkeypatch.setattr('ouroboros.observability.retry_pending_child_ref_promotions', lambda *_a: None)
    monkeypatch.setattr(maintenance, '_reconcile_abandoned_usage', lambda *_a: None)
    real_register = __import__('supervisor.terminal_delivery', fromlist=['register_pending_delivery']).register_pending_delivery
    monkeypatch.setattr('supervisor.terminal_delivery.register_pending_delivery', lambda *_a, **_kw: False)
    assert settle_terminal_projection(project_root.root, 'root-1') == SETTLEMENT_DEFERRED
    assert not project_root.queued
    ready = load_task_result(project_root.root, 'root-1')['canonical_terminal_projection_ready']
    ready['main_delivery']['last_attempt_at'] = '2000-01-01T00:00:00Z'
    write_task_result(project_root.root, 'root-1', 'completed', canonical_terminal_projection_ready=ready)
    monkeypatch.setattr('supervisor.terminal_delivery.register_pending_delivery', real_register)
    monkeypatch.setattr(maintenance, 'DATA_DIR', project_root.root)
    stop_checks = iter([False, True])
    monkeypatch.setattr(maintenance, '_stop_requested', lambda *_a: next(stop_checks))
    assert maintenance._CUSTODY_SWEEP_LOCK.acquire(blocking=False)
    maintenance._run_periodic_custody_sweep()
    assert len(project_root.queued) == len(_project_rows(project_root.root, 'root-1')) == 1
    assert load_task_result(project_root.root, 'root-1')['canonical_terminal_projection']['main_disposition'] == 'delivered'


def test_main_uses_current_canonical_result_after_project_append(project_root, monkeypatch):
    from ouroboros import project_dialogue as dialogue

    _store(project_root.root, phase='completed', terminal_origin='model_final', result='old answer')
    original = dialogue.append_canonical_task_summary

    def append(root, row):
        success = original(root, row)
        write_task_result(root, 'root-1', 'completed', result='fresh canonical answer')
        return success

    monkeypatch.setattr(dialogue, 'append_canonical_task_summary', append)
    assert settle_terminal_projection(project_root.root, 'root-1', task=project_root.task,
                                      event={'result': 'stale event answer'}) == SETTLEMENT_SETTLED
    assert project_root.queued[0]['progress_meta']['completion_answer'] == 'fresh canonical answer'


def test_main_address_and_retry_claim_are_durable_before_live_queue(project_root, monkeypatch):
    _store(project_root.root)

    def queue(root, event, **_kw):
        stored = load_task_result(root, 'root-1')
        ready = stored['canonical_terminal_projection_ready']
        assert ready['main_delivery']['event'] == event
        assert ready['main_delivery']['attempts'] == 1
        assert ready['main_delivery']['event']['chat_id'] == 1
        return project_root.send(root, event)

    monkeypatch.setattr('supervisor.terminal_delivery.enqueue_terminal_delivery', queue)
    assert settle_terminal_projection(project_root.root, 'root-1', task=project_root.task) == SETTLEMENT_SETTLED


def test_append_receipt_crash_dedupes_by_the_stored_publication(project_root, monkeypatch):
    from ouroboros import project_dialogue as dialogue
    from ouroboros.terminal_projection import reconcile_terminal_projections

    _store(project_root.root)
    original = dialogue.append_canonical_task_summary

    def append_then_fail(root, row):
        assert original(root, row)
        raise OSError('append succeeded; result receipt was not written')

    monkeypatch.setattr(dialogue, 'append_canonical_task_summary', append_then_fail)
    assert settle_terminal_projection(project_root.root, 'root-1', task=project_root.task) == SETTLEMENT_DEFERRED
    monkeypatch.setattr(dialogue, 'append_canonical_task_summary', original)
    assert reconcile_terminal_projections(project_root.root) == 1
    assert len(_project_rows(project_root.root, 'root-1')) == 1, "the earlier publication is found in the store, not appended again"
    assert len(project_root.queued) == 1


@pytest.mark.parametrize('artifact_status', ['pending', 'finalizing'])
def test_artifact_finalization_holds_project_and_main(project_root, artifact_status):
    task = {**project_root.task, 'workspace_root': str(project_root.root / 'workspace')}
    _store(project_root.root, workspace_root=task['workspace_root'], artifact_status=artifact_status)
    assert settle_terminal_projection(project_root.root, 'root-1', task=task) == SETTLEMENT_DEFERRED
    assert not _project_rows(project_root.root, 'root-1')
    assert not project_root.queued
    _store(project_root.root, workspace_root=task['workspace_root'], artifact_status='failed',
           artifact_bundle={'status': 'failed'}, outcome_axes={'artifacts': {'status': 'failed'}})
    assert settle_terminal_projection(project_root.root, 'root-1', task=task) == SETTLEMENT_SETTLED
    assert len(_project_rows(project_root.root, 'root-1')) == 1
    assert _project_rows(project_root.root, 'root-1')[0]['outcome'] == 'Failed'
    assert len(project_root.queued) == 1
    assert settle_terminal_projection(project_root.root, 'root-1', task=task) == SETTLEMENT_NONE


def test_terminal_timestamp_enrichment_does_not_remint(project_root):
    _store(project_root.root)
    assert settle_terminal_projection(project_root.root, 'root-1', task=project_root.task) == SETTLEMENT_SETTLED
    before = load_task_result(project_root.root, 'root-1')['canonical_terminal_projection']
    _store(project_root.root, ts='2099-01-01T00:00:00Z')
    assert settle_terminal_projection(project_root.root, 'root-1', task=project_root.task) == SETTLEMENT_NONE
    assert load_task_result(project_root.root, 'root-1')['canonical_terminal_projection'] == before
    assert len(_project_rows(project_root.root, 'root-1')) == 1
    assert len(project_root.queued) == 1


def test_canonical_cancel_keeps_existing_artifact_readiness_exception(project_root):
    write_task_result(project_root.root, 'root-1', 'cancelled', root_task_id='root-1',
        project_id='launch', workspace_root=str(project_root.root/'workspace'),
        artifact_status='pending', result='Cancelled by custody')
    assert settle_terminal_projection(project_root.root, 'root-1') == SETTLEMENT_SETTLED
    assert _project_rows(project_root.root, 'root-1')[0]['outcome'] == 'Stopped'


def test_split_root_uses_stored_adoption_facts_without_task_argument(project_root):
    child = project_root.root/'child-drive'
    _store(project_root.root, child_drive_root=str(child),
           workspace_root=str(project_root.root/'workspace'), artifact_status='ready_with_changes')
    assert settle_terminal_projection(project_root.root, 'root-1') == SETTLEMENT_DEFERRED
    assert not _project_rows(project_root.root, 'root-1')
    _store(project_root.root, headless_child_drive_root=str(child),
           child_ref_promotion={'schema_version': 1, 'status': 'complete'})
    assert settle_terminal_projection(project_root.root, 'root-1') == SETTLEMENT_SETTLED
    assert len(_project_rows(project_root.root, 'root-1')) == 1


def test_child_append_does_not_scan_root_recovery_history(tmp_path, monkeypatch):
    def forbidden(*_a):
        raise AssertionError('child entered root history recovery')
    monkeypatch.setattr('ouroboros.terminal_projection._already_in_chat', forbidden)
    child = {'id': 'child', 'parent_task_id': 'root', 'root_task_id': 'root',
             'delegation_role': 'subagent', 'chat_id': 1}
    stored = write_task_result(tmp_path, 'child', 'completed', parent_task_id='root',
                               root_task_id='root', delegation_role='subagent')
    assert append_terminal_task_projection(tmp_path, 'child', child, stored, DONE)
    assert not append_terminal_task_projection(tmp_path, 'child', child, stored, DONE)
    assert len(_project_rows(tmp_path, 'child')) == 1
