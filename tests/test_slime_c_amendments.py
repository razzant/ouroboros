"""Phase C amendments (slime saga): live final-answer delivery before blocking
post-task cognition, sealed final ground truth into synthesis, the durable
per-project last-result pointer, and the project reflections read-back.

Incident e9108a09: the owner's answer waited 34 minutes behind a hung
backlog groom, the reaper killed the worker, and the buffered send_message
died with it — while the post-task reflection, written from the error trace,
declared the delivered 52-page PDF missing.
"""

import json
import pathlib
import queue
import time


def _make_fake_env(drive_root: pathlib.Path):
    class FakeMemory:
        def load_identity(self):
            return "test identity"

    class FakeCtx:
        pending_restart_reason = None

    class FakeEnv:
        def __init__(self, root):
            self.drive_root = root

        def drive_path(self, sub):
            p = self.drive_root / sub
            p.mkdir(parents=True, exist_ok=True)
            return p

    return FakeEnv(drive_root), FakeMemory(), FakeCtx()


def _make_drive(tmp_path):
    drive_root = tmp_path / "data"
    logs = drive_root / "logs"
    logs.mkdir(parents=True)
    (drive_root / "memory").mkdir()
    (drive_root / "task_results").mkdir()
    return drive_root, logs


def _emit(atp, env, memory, ctx, pending_events, task, event_queue, text="Reply text"):
    class FakeLLM:
        def chat(self, **kwargs):
            return {"content": "summary"}, {"cost": 0}

    atp.emit_task_results(
        env=env, memory=memory, llm=FakeLLM(),
        pending_events=pending_events,
        task=task, text=text,
        usage={"cost": 0.01, "rounds": 3, "prompt_tokens": 100, "completion_tokens": 50},
        llm_trace={"tool_calls": [], "reasoning_notes": []},
        start_time=time.time() - 1.0,
        drive_logs=env.drive_root / "logs",
        ctx=ctx,
        event_queue=event_queue,
    )


class TestLiveFinalAnswerDelivery:
    """The final send_message must be observable supervisor-side BEFORE a
    blocking post-task phase starts, while task_done stays buffered (last)."""

    def test_answer_live_before_blocking_post_task_and_task_done_not_yet_emitted(
        self, tmp_path, monkeypatch,
    ):
        drive_root, logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)
        # An artifact in the store: the sealed package must attest it.
        art = drive_root / "task_results" / "artifacts" / "live1"
        art.mkdir(parents=True)
        (art / "report.pdf").write_bytes(b"x" * 123)

        import ouroboros.agent_task_pipeline as atp

        event_queue = queue.Queue()
        pending_events = []
        observed = {}

        def hanging_post_task(*args, **kwargs):
            # Snapshot at the exact moment the (blocking) post-task hook runs:
            # this is where the incident's groom hung for 34 minutes.
            live = []
            while not event_queue.empty():
                live.append(event_queue.get_nowait())
            observed["live_types"] = [e.get("type") for e in live]
            observed["live_events"] = live
            observed["pending_types_at_hook"] = [e.get("type") for e in pending_events]
            observed["blocking"] = kwargs.get("blocking")
            observed["sealed_final"] = kwargs.get("sealed_final")
            return None

        monkeypatch.setattr(atp, "_run_chat_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_post_task_processing_async", hanging_post_task)

        # type=evolution → the blocking post-task branch (same class as the
        # incident's workspace/project roots).
        task = {"id": "live1", "type": "evolution", "chat_id": 1, "text": "hello"}
        _emit(atp, env, memory, ctx, pending_events, task, event_queue)

        # At hook time the owner's answer had ALREADY left over the live queue…
        assert observed["blocking"] is True
        assert observed["live_types"] == ["send_message"]
        assert observed["live_events"][0]["text"] == "Reply text"
        assert observed["live_events"][0]["task_id"] == "live1"
        assert observed["live_events"][0]["delivery_id"]
        # …and task_done had NOT been emitted by the worker: not on the live
        # queue, still waiting in the buffer for the post-return drain. The
        # buffered send_message copy is KEPT (queue.put is not a delivery
        # receipt) and carries the same delivery_id for supervisor-side dedupe.
        assert "task_done" not in observed["live_types"]
        assert "task_done" in observed["pending_types_at_hook"]
        assert "send_message" in observed["pending_types_at_hook"]

        # Durable result was stored BEFORE the live delivery point.
        stored = json.loads(
            (drive_root / "task_results" / "live1.json").read_text(encoding="utf-8"))
        assert stored["result"] == "Reply text"

    def test_no_duplicate_final_message_when_both_copies_flow(
        self, tmp_path, monkeypatch,
    ):
        """Both the live copy AND the kept buffered copy reach the supervisor;
        the delivery_id dedupe in _handle_send_message delivers exactly one."""
        drive_root, logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)

        import ouroboros.agent_task_pipeline as atp

        event_queue = queue.Queue()
        pending_events = []
        monkeypatch.setattr(atp, "_run_chat_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **kw: None)

        task = {"id": "dedupe1", "type": "evolution", "chat_id": 1, "text": "hi"}
        _emit(atp, env, memory, ctx, pending_events, task, event_queue)

        live = [event_queue.get_nowait() for _ in range(event_queue.qsize())]
        live_sends = [e for e in live if e.get("type") == "send_message"]
        buffered_sends = [e for e in pending_events if e.get("type") == "send_message"]
        # Both copies exist (put() is not a receipt) with one shared delivery_id.
        assert len(live_sends) == 1 and len(buffered_sends) == 1
        assert live_sends[0]["delivery_id"] == buffered_sends[0]["delivery_id"]
        # task_done goes LAST, exactly as today, via the buffered return only.
        assert all(e.get("type") != "task_done" for e in live)
        assert [e.get("type") for e in pending_events].count("task_done") == 1

        # Supervisor side: both copies flow, exactly one message is sent.
        from supervisor.events import _handle_send_message

        sent = []

        class SupCtx:
            RUNNING = {}

            @staticmethod
            def send_with_budget(chat_id, text, **kwargs):
                sent.append(text)

            @staticmethod
            def append_jsonl(path, data):
                pass

        _handle_send_message(live_sends[0], SupCtx())
        _handle_send_message(buffered_sends[0], SupCtx())
        assert sent == ["Reply text"]

    def test_proactive_message_does_not_hijack_live_final_delivery(
        self, tmp_path, monkeypatch,
    ):
        """Regression (review CRITICAL, reproduced live): send_user_message
        (tools/control.py) can fall back into the SAME pending_events buffer
        mid-task. Since the live-first seam its deferred frames carry the
        SAME task_id as the final (plus the proactive_message discriminator),
        so the selector's contract is last-match: the host appends the FINAL
        after all tool-time frames, and only that last row ships live. The
        proactive text must never leave early under a degenerate
        'final:<tid>:<digest>' delivery_id while the owner's answer stays
        hostage to blocking post-task."""
        drive_root, logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)

        import ouroboros.agent_task_pipeline as atp

        event_queue = queue.Queue()
        # Exact deferred shape _send_user_message buffers via the seam: same
        # buffer, SAME task_id as the finalizing task, typed discriminator.
        proactive = {
            "type": "send_message", "chat_id": 1, "text": "Proactive mid-task ping",
            "format": "markdown", "is_progress": False,
            "system_type": "proactive_message", "ts": "2026-08-10T00:00:00Z",
            "task_id": "proact1", "parent_task_id": "", "root_task_id": "",
        }
        pending_events = [proactive]
        monkeypatch.setattr(atp, "_run_chat_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **kw: None)

        task = {"id": "proact1", "type": "evolution", "chat_id": 1, "text": "hi"}
        _emit(atp, env, memory, ctx, pending_events, task, event_queue,
              text="The real final answer.")

        live = [event_queue.get_nowait() for _ in range(event_queue.qsize())]
        # Exactly one live event: the FINAL answer, not the proactive text.
        assert len(live) == 1
        assert live[0]["text"] == "The real final answer."
        assert live[0]["task_id"] == "proact1"
        # The delivery_id carries the real task id (never 'final::<digest>').
        assert live[0]["delivery_id"].startswith("final:proact1:")
        # The proactive frame stays buffered without a delivery_id: it flows
        # through the ordinary post-return drain and is never suppressed as
        # the final's duplicate. Both rows carry the task id now; only the
        # FINAL — appended last by the host — is minted and shipped.
        assert "delivery_id" not in proactive
        tagged = [
            e for e in pending_events
            if e.get("type") == "send_message" and e.get("task_id") == "proact1"
        ]
        assert len(tagged) == 2
        assert tagged[0] is proactive
        assert tagged[-1]["delivery_id"] == live[0]["delivery_id"]

    def test_live_delivery_falls_back_to_last_send_message(self):
        """Defensive fallback: with no task_id match in the buffer, the LAST
        send_message ships (the final is always appended after mid-task
        proactives) and the delivery_id still carries the finalizing task's
        id, never the degenerate empty form."""
        from ouroboros.task_finalization import deliver_final_message_live

        events = [
            {"type": "send_message", "chat_id": 1, "text": "first proactive"},
            {"type": "task_eval", "chat_id": 1},
            {"type": "send_message", "chat_id": 1, "text": "last send"},
            {"type": "task_done", "chat_id": 1},
        ]
        live_queue = queue.Queue()
        assert deliver_final_message_live(live_queue, events, "nomatch1") is True
        live = live_queue.get_nowait()
        assert live["text"] == "last send"
        assert live["delivery_id"].startswith("final:nomatch1:")
        assert "delivery_id" not in events[0]

    def test_failed_live_send_does_not_suppress_buffered_copy(self, tmp_path):
        """Regression (review CRITICAL): the delivery_id used to be registered
        in _DELIVERED_MESSAGE_IDS BEFORE send_with_budget, so a raising live
        send suppressed the buffered copy later — the answer was LOST. The id
        is now registered only AFTER a successful send: first send raises →
        the buffered copy delivers exactly once (and only then is a further
        duplicate suppressed)."""
        from supervisor.events import _handle_send_message

        sent = []
        state = {"raise_next": True}

        class SupCtx:
            RUNNING = {}
            DRIVE_ROOT = tmp_path

            @staticmethod
            def send_with_budget(chat_id, text, **kwargs):
                if state["raise_next"]:
                    state["raise_next"] = False
                    raise RuntimeError("bridge down")
                sent.append(text)

            @staticmethod
            def append_jsonl(path, data):
                pass

        evt = {"type": "send_message", "chat_id": 1, "text": "Final answer",
               "task_id": "sendfail1", "delivery_id": "final:sendfail1:cafe0123"}
        _handle_send_message(dict(evt), SupCtx())  # live copy: send raises
        assert sent == []
        _handle_send_message(dict(evt), SupCtx())  # buffered copy must flow
        assert sent == ["Final answer"]
        _handle_send_message(dict(evt), SupCtx())  # registered now → suppressed
        assert sent == ["Final answer"]

    def test_live_transport_failure_falls_back_to_buffered_path(
        self, tmp_path, monkeypatch,
    ):
        """A broken live channel must never LOSE (or double) the answer."""
        drive_root, logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)

        import ouroboros.agent_task_pipeline as atp

        class BrokenQueue:
            def put(self, item):
                raise RuntimeError("manager gone")

        pending_events = []
        monkeypatch.setattr(atp, "_run_chat_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **kw: None)

        task = {"id": "fallback1", "type": "evolution", "chat_id": 1, "text": "hi"}
        _emit(atp, env, memory, ctx, pending_events, task, BrokenQueue())

        types = [e.get("type") for e in pending_events]
        assert types.count("send_message") == 1  # buffered, not lost
        assert types.index("send_message") < types.index("task_done")

    def test_non_blocking_tasks_keep_buffered_delivery(self, tmp_path, monkeypatch):
        """A plain task (post-task runs async) keeps today's buffered path —
        the live shortcut exists only where delivery used to be held hostage."""
        drive_root, logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)

        import ouroboros.agent_task_pipeline as atp

        event_queue = queue.Queue()
        pending_events = []
        monkeypatch.setattr(atp, "_run_chat_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **kw: None)

        task = {"id": "plain1", "type": "task", "chat_id": 1, "text": "hi"}
        _emit(atp, env, memory, ctx, pending_events, task, event_queue)

        assert event_queue.empty()
        types = [e.get("type") for e in pending_events]
        assert types.count("send_message") == 1
        assert types.index("send_message") < types.index("task_done")


class TestSealedFinalPackage:
    """Summary/reflection receive host-attested final ground truth (Q4A)."""

    def test_sealed_package_built_from_stored_result_artifacts(self, tmp_path):
        """The manifest reuses the durable result's own artifact records —
        no independent filesystem walk (no second source of truth)."""
        from ouroboros.task_finalization import build_sealed_final_package

        drive_root, logs = _make_drive(tmp_path)
        art = drive_root / "task_results" / "artifacts" / "seal1"
        art.mkdir(parents=True)
        (art / "report.pdf").write_bytes(b"x" * 123)

        sealed = build_sealed_final_package({
            "task_id": "seal1",
            "artifacts": [{"name": "report.pdf", "path": str(art / "report.pdf"),
                           "size": 123, "status": "ready"}],
        }, "The PDF is ready.")

        assert sealed["final_result_text"] == "The PDF is ready."
        assert [(i["name"], i["size_bytes"]) for i in sealed["artifact_manifest"]] == [
            ("report.pdf", 123),
        ]

    def test_facts_row_retells_nothing_of_the_sealed_final(self, tmp_path, monkeypatch):
        """No paid narrative: the host facts row neither prompts a model nor
        copies the delivered answer; reflection alone reads the sealed package."""
        import pytest

        import ouroboros.agent_task_pipeline as atp

        drive_root, logs = _make_drive(tmp_path)
        env, _memory, _ctx = _make_fake_env(drive_root)
        monkeypatch.setattr("ouroboros.llm_observability.chat_observed",
                            lambda *_a, **_k: pytest.fail("the facts row buys no model call"))
        atp._record_task_facts(
            env,
            {"id": "sum1", "type": "task", "chat_id": 1, "text": "make a pdf"},
            {"cost": 0.01, "rounds": 3},
            {"tool_calls": [{"tool": "shell", "error": ""}], "reasoning_notes": []},
            logs,
        )

        [row] = [json.loads(line) for line in (logs / "chat.jsonl").read_text(encoding="utf-8").splitlines()]
        assert row["summary_kind"] == "host_task_facts" and row["text"] == ""
        assert row["tool_calls"] == 1 and row["rounds"] == 3
        assert "Delivered" not in json.dumps(row) and "make a pdf" not in json.dumps(row)

    def test_sealed_package_reaches_reflection_prompt(self, tmp_path, monkeypatch):
        import ouroboros.llm_observability as obs
        from ouroboros.reflection import generate_reflection
        from ouroboros.task_finalization import (
            build_sealed_final_package, sealed_final_prompt_section,
        )

        drive_root, logs = _make_drive(tmp_path)
        art = drive_root / "task_results" / "artifacts" / "refl1"
        art.mkdir(parents=True)
        (art / "report.pdf").write_bytes(b"x" * 123)

        captured = {}

        def fake_chat_observed(llm, **kwargs):
            captured["prompt"] = kwargs["messages"][0]["content"]
            return {"content": "reflection\nMEMORY_ACTIONS_JSON: []\n"
                               "BACKLOG_CANDIDATES_JSON: []"}, {"cost": 0}

        monkeypatch.setattr(obs, "chat_observed", fake_chat_observed)

        sealed = build_sealed_final_package(
            {"artifacts": [{"name": "report.pdf", "path": str(art / "report.pdf"),
                            "size": 123, "status": "ready"}]},
            "Delivered the 52-page PDF.")
        task = {"id": "refl1", "type": "task", "chat_id": 1, "text": "make a pdf",
                "drive_root": str(drive_root)}
        entry = generate_reflection(
            task,
            {"tool_calls": [{"tool": "shell", "error": "boom"}], "reasoning_notes": []},
            "trace summary", object(), {},
            usage_snapshot_text="",
            sealed_final_text=sealed_final_prompt_section(sealed),
        )

        assert entry
        assert "Sealed final outcome (host-attested ground truth)" in captured["prompt"]
        assert "Delivered the 52-page PDF." in captured["prompt"]
        assert "report.pdf (123 bytes)" in captured["prompt"]

    def test_emit_task_results_passes_sealed_package_to_post_task(
        self, tmp_path, monkeypatch,
    ):
        drive_root, logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)
        art = drive_root / "task_results" / "artifacts" / "wire1"
        art.mkdir(parents=True)
        (art / "report.pdf").write_bytes(b"x" * 123)

        import ouroboros.agent_task_pipeline as atp

        observed = {}
        monkeypatch.setattr(atp, "_run_chat_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(atp, "_run_scratchpad_consolidation", lambda *a, **kw: None)
        monkeypatch.setattr(
            atp, "_run_post_task_processing_async",
            lambda *a, **kw: observed.update(sealed=kw.get("sealed_final")),
        )

        task = {"id": "wire1", "type": "task", "chat_id": 1, "text": "hi"}
        _emit(atp, env, memory, ctx, [], task, queue.Queue(), text="Final answer.")

        # The manifest travelled from the freshly stored durable result (whose
        # artifacts the pipeline's own store authority collected).
        sealed = observed["sealed"]
        assert sealed["final_result_text"] == "Final answer."
        assert [(i["name"], i["size_bytes"]) for i in sealed["artifact_manifest"]] == [
            ("report.pdf", 123),
        ]


class TestProjectLastResultPointer:
    """Durable registry pointer beats the bounded newest-64 mtime scan."""

    def _write_results(self, tmp_path, n_foreign=70):
        import os

        from ouroboros.task_results import task_result_path, write_task_result

        write_task_result(
            tmp_path, "projres1", "completed", project_id="slime",
            objective="project deliverable", ts="2026-08-10T00:00:01Z",
        )
        os.utime(task_result_path(tmp_path, "projres1", create=False), (100, 100))
        for i in range(n_foreign):
            tid = f"foreign{i:03d}"
            write_task_result(
                tmp_path, tid, "completed", project_id="", objective="noise",
            )
            os.utime(task_result_path(tmp_path, tid, create=False),
                     (200 + i, 200 + i))

    def test_pointer_survives_more_than_64_newer_foreign_results(self, tmp_path):
        import server
        from ouroboros.projects_registry import create_project, update_project

        create_project(tmp_path, "slime", name="Slime")
        self._write_results(tmp_path, n_foreign=70)
        update_project(tmp_path, "slime", last_task_result_id="projres1")

        class Ctx:
            DRIVE_ROOT = tmp_path

        row = server._latest_project_task_result(Ctx(), "slime")
        assert row is not None
        assert str(row.get("task_id") or row.get("id")) == "projres1"

    def test_missing_pointer_self_heals_via_full_scan(self, tmp_path):
        """A pre-pointer project whose newest result sits deeper than the
        bounded 64-window is found via the disclosed full-store scan, and the
        ABSENT pointer is written back so the next call is a direct fetch.
        (With zero matching results nothing is written back, so the scan
        repeats per lookup until a matching result exists.)"""
        import server
        from ouroboros.projects_registry import create_project, get_project

        create_project(tmp_path, "slime", name="Slime")
        self._write_results(tmp_path, n_foreign=70)

        class Ctx:
            DRIVE_ROOT = tmp_path

        row = server._latest_project_task_result(Ctx(), "slime")
        assert row is not None
        assert str(row.get("task_id") or row.get("id")) == "projres1"
        assert get_project(tmp_path, "slime")["last_task_result_id"] == "projres1"

    def test_stale_pointer_falls_back_to_scan(self, tmp_path):
        import server
        from ouroboros.projects_registry import create_project, update_project

        create_project(tmp_path, "slime", name="Slime")
        self._write_results(tmp_path, n_foreign=10)
        update_project(tmp_path, "slime", last_task_result_id="vanished")

        class Ctx:
            DRIVE_ROOT = tmp_path

        # Pointer targets a missing file → the fallback scan still serves the
        # project result (10 foreign rows fit inside the bounded window), but
        # a NON-EMPTY pointer is never overwritten from the scan: it may be a
        # fresh split-drive stamp whose canonical copy-back has not landed yet
        # (write-back is reserved for the ABSENT-pointer self-heal).
        row = server._latest_project_task_result(Ctx(), "slime")
        assert row is not None
        assert str(row.get("task_id") or row.get("id")) == "projres1"
        from ouroboros.projects_registry import get_project
        assert get_project(tmp_path, "slime")["last_task_result_id"] == "vanished"

    def test_copyback_window_lookup_does_not_regress_pointer(self, tmp_path):
        """Regression (review HIGH): for a split-drive root, finalization
        stamps last_task_result_id BEFORE the canonical copy-back materializes
        the result file. A project-room lookup in that window used to write
        the scan hit back over the fresh pointer, permanently regressing it to
        an OLDER result. The scan hit is served for this turn only; the fresh
        pointer survives until the copy-back lands."""
        import server
        from ouroboros.projects_registry import create_project, get_project, update_project

        create_project(tmp_path, "slime", name="Slime")
        self._write_results(tmp_path, n_foreign=10)
        # Freshly stamped pointer; task_results/inflight42.json does not exist
        # yet (the copy-back window).
        update_project(tmp_path, "slime", last_task_result_id="inflight42")

        class Ctx:
            DRIVE_ROOT = tmp_path

        row = server._latest_project_task_result(Ctx(), "slime")
        # This turn is served from the scan (the older durable result)…
        assert row is not None
        assert str(row.get("task_id") or row.get("id")) == "projres1"
        # …and the fresh pointer is NOT regressed to that older result.
        assert get_project(tmp_path, "slime")["last_task_result_id"] == "inflight42"

    def test_record_task_finalization_stamps_pointer(self, tmp_path, monkeypatch):
        import ouroboros.config as cfg
        from ouroboros.projects_registry import create_project, get_project
        from ouroboros.tools.project_journal import record_task_finalization

        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")
        create_project(tmp_path, "slime", name="Slime")

        record_task_finalization(
            "slime", {"id": "task42"},
            objective="finish the game", kind="done", exec_status="ok",
            drive_root=tmp_path,
        )

        assert get_project(tmp_path, "slime")["last_task_result_id"] == "task42"

    def test_project_room_direct_chat_root_stamps_the_pointer(self, tmp_path, monkeypatch):
        """I29: a project ROOM's direct-chat root writes a durable result carrying
        project_id, but wrote no pointer - so the per-project fallback for "continue
        from this result" was empty too, and a promote from that room found nothing.
        The pointer is stamped; the deliberate letters-home exclusion stands."""
        import ouroboros.agent_task_pipeline as atp
        import ouroboros.config as cfg
        from ouroboros.projects_registry import create_project, get_project

        drive_root, _logs = _make_drive(tmp_path)
        env, memory, ctx = _make_fake_env(drive_root)
        monkeypatch.setattr(cfg, "DATA_DIR", drive_root)
        monkeypatch.setattr(atp, "_run_post_task_processing_async", lambda *a, **kw: None)
        project = create_project(drive_root, "slime", name="Slime")
        pending: list = []

        task = {"id": "roomturn1", "type": "task", "chat_id": project["chat_id"],
                "project_id": "slime", "_is_direct_chat": True, "text": "how is it going?"}
        _emit(atp, env, memory, ctx, pending, task, queue.Queue(), text="Fine.")

        assert get_project(drive_root, "slime")["last_task_result_id"] == "roomturn1"
        # Still NOT a letters-home task: no journal milestone, no digest.
        assert not [e for e in pending if e.get("type") == "project_digest"]
        assert not (drive_root / "projects" / "slime" / "journal.jsonl").exists()


class TestProjectReflectionsReadBack:
    """A project-bound context must include the project's OWN full reflections
    (the canonical log only carries bounded pointer rows)."""

    def test_project_bound_context_includes_project_reflection(
        self, tmp_path, monkeypatch,
    ):
        import ouroboros.config as cfg

        monkeypatch.setattr(cfg, "DATA_DIR", tmp_path / "data")

        from ouroboros.context import build_recent_sections
        from ouroboros.memory import Memory
        from ouroboros.project_facts import project_reflections_path
        from ouroboros.utils import append_jsonl

        path = project_reflections_path("slimeproj")
        path.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(path, {
            "ts": "2026-08-10T12:00:00Z", "task_id": "ref42", "task_type": "task",
            "goal": "PROJECT_LESSON_MARKER remember the slime physics fix",
            "key_markers": [],
        })

        drive_root = tmp_path / "drive"
        (drive_root / "logs").mkdir(parents=True)
        sections = build_recent_sections(
            Memory(drive_root=drive_root), env=None, project_id="slimeproj",
        )

        combined = "\n\n".join(sections)
        assert "Project execution reflections (this project's own: slimeproj)" in combined
        assert "PROJECT_LESSON_MARKER" in combined

    def test_non_project_context_has_no_project_section(self, tmp_path):
        from ouroboros.context import build_recent_sections
        from ouroboros.memory import Memory

        drive_root = tmp_path / "drive"
        (drive_root / "logs").mkdir(parents=True)
        combined = "\n\n".join(
            build_recent_sections(Memory(drive_root=drive_root), env=None))
        assert "Project execution reflections" not in combined
