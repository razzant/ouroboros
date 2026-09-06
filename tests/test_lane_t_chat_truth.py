"""Lane T: typed direct/ephemeral conclusions (#369) and honest budget pause (#322)."""

import types

import pytest


class TestStampRootFinalPhase:
    def test_open_post_task_holds_finalizing(self):
        from ouroboros.task_finalization import stamp_root_final_phase

        evt = {"type": "send_message"}
        stamp_root_final_phase(evt, {"_is_direct_chat": True}, post_task_open=True, terminal_status="completed")
        assert evt["progress_meta"] == {"task_phase": "finalizing"}

    def test_settled_direct_bare_final_is_typed_terminal(self):
        from ouroboros.task_finalization import stamp_root_final_phase

        evt = {"type": "send_message"}
        stamp_root_final_phase(evt, {"_is_direct_chat": True}, post_task_open=False, terminal_status="completed")
        assert evt["progress_meta"] == {"task_terminal_status": "completed"}

    def test_settled_direct_final_names_the_durable_status(self):
        """A stopped direct turn settles ``failed``: the terminal word is the row's, never a blanket completed."""
        from ouroboros.task_finalization import stamp_root_final_phase

        evt = {"type": "send_message"}
        stamp_root_final_phase(evt, {"_is_direct_chat": True}, post_task_open=False, terminal_status="failed")
        assert evt["progress_meta"] == {"task_terminal_status": "failed"}

    def test_settled_managed_root_keeps_task_done_conclusion(self):
        from ouroboros.task_finalization import stamp_root_final_phase

        evt = {"type": "send_message"}
        stamp_root_final_phase(evt, {}, post_task_open=False, terminal_status="failed")
        assert "progress_meta" not in evt


class TestEphemeralTerminalStamp:
    def test_ephemeral_final_carries_typed_conclusion(self, tmp_path):
        from ouroboros.task_finalization import prepare_terminal_send_event

        evt = {"type": "send_message", "task_id": "e1"}
        out = prepare_terminal_send_event(
            tmp_path, {"id": "e1", "_ephemeral_turn": True}, "answer", {},
            evt, ephemeral=True, presence=False,
        )
        # The final concludes the activity even when its task_done is missed;
        # the pipeline adds the outcome facts before delivery.
        assert out["progress_meta"]["task_terminal_status"] == "completed"

    def test_presence_frames_stay_unstamped(self, tmp_path):
        from ouroboros.task_finalization import prepare_terminal_send_event

        evt = {"type": "send_message", "task_id": "p1"}
        out = prepare_terminal_send_event(
            tmp_path, {"id": "p1"}, "answer", {}, evt, ephemeral=True, presence=True,
        )
        assert "progress_meta" not in out


class TestBudgetPauseFact:
    def test_own_pause_row_wins(self):
        from supervisor.queue_transitions import budget_pause_fact

        row = {"status": "paused_before_dispatch"}
        assert budget_pause_fact({"_budget_pause": row}, {}) is row

    def test_root_fence_covers_fenceless_sibling(self):
        from supervisor.queue_transitions import budget_pause_fact

        fences = {"r1": {"status": "paused"}}
        assert budget_pause_fact({"id": "child", "root_task_id": "r1"}, fences)
        assert budget_pause_fact({"id": "r1"}, fences)  # the root itself

    def test_inactive_fence_and_foreign_root_are_not_pauses(self):
        from supervisor.queue_transitions import budget_pause_fact

        assert budget_pause_fact({"id": "x", "root_task_id": "r2"}, {"r1": {"status": "paused"}}) is None
        assert budget_pause_fact({"id": "r1"}, {"r1": {"status": "cleared"}}) is None
        assert budget_pause_fact({}, {}) is None


class TestClearBudgetRootFence:
    @pytest.fixture()
    def clean_queue(self, monkeypatch):
        from supervisor import queue as q

        monkeypatch.setattr(q, "PENDING", [])
        monkeypatch.setattr(q, "RUNNING", {})
        monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", {})
        return q

    def test_last_settled_member_releases_the_fence(self, clean_queue):
        from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

        clean_queue.BUDGET_ROOT_FENCES["r1"] = {"status": "paused", "root_task_id": "r1"}
        assert clear_budget_root_fence_for_settled_tree({"id": "t9", "root_task_id": "r1"}) is True
        assert "r1" not in clean_queue.BUDGET_ROOT_FENCES

    def test_live_pending_member_keeps_the_fence(self, clean_queue):
        from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

        clean_queue.BUDGET_ROOT_FENCES["r1"] = {"status": "paused", "root_task_id": "r1"}
        clean_queue.PENDING.append({"id": "sib", "root_task_id": "r1"})
        assert clear_budget_root_fence_for_settled_tree({"id": "t9", "root_task_id": "r1"}) is False
        assert "r1" in clean_queue.BUDGET_ROOT_FENCES

    def test_live_running_member_keeps_the_fence(self, clean_queue):
        from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

        clean_queue.BUDGET_ROOT_FENCES["r1"] = {"status": "paused", "root_task_id": "r1"}
        clean_queue.RUNNING["sib"] = {"task": {"id": "sib", "root_task_id": "r1"}}
        assert clear_budget_root_fence_for_settled_tree({"id": "t9", "root_task_id": "r1"}) is False

    def test_foreign_or_missing_fence_is_a_noop(self, clean_queue):
        from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

        assert clear_budget_root_fence_for_settled_tree({"id": "t9", "root_task_id": "nope"}) is False
        assert clear_budget_root_fence_for_settled_tree({}) is False


class TestStateProjectionBudgetPaused:
    def test_paused_pending_root_projects_budget_paused_phase(self, tmp_path, monkeypatch):
        from supervisor import queue as q
        from ouroboros.gateway.state import _chat_activities_snapshot_safe

        monkeypatch.setattr(q, "PENDING", [
            {"id": "root-paused", "chat_id": 1, "queued_at": "2026-08-31T00:00:00Z",
             "_budget_pause": {"status": "paused_before_dispatch"}},
            {"id": "root-queued", "chat_id": 1, "queued_at": "2026-08-31T00:00:00Z"},
            {"id": "root-fenced", "chat_id": 1, "queued_at": "2026-08-31T00:00:00Z"},
        ])
        monkeypatch.setattr(q, "RUNNING", {})
        monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", {
            "root-fenced": {"status": "paused", "root_task_id": "root-fenced"},
        })
        rows = {r["activity_id"]: r for r in _chat_activities_snapshot_safe(tmp_path, {})}
        assert rows["root-paused"]["phase"] == "budget_paused"
        assert rows["root-fenced"]["phase"] == "budget_paused"
        assert rows["root-queued"]["phase"] == "queued"


class TestFenceReleaseIdentity:
    def test_cancel_task_done_event_carries_root_identity(self, monkeypatch):
        # Pending-cancel and reaper task_done arrive AFTER the row left the
        # queue: the event stamp is the fence-release seam's identity source.
        from supervisor import queue as q
        from supervisor import workers

        captured = []
        monkeypatch.setattr(
            workers, "get_event_q",
            lambda: types.SimpleNamespace(put=captured.append),
        )
        q._emit_cancel_task_done({"id": "child", "root_task_id": "r1", "chat_id": 1}, "child")
        assert captured and captured[0]["root_task_id"] == "r1"

    def test_event_shaped_identity_releases_the_fence(self, monkeypatch):
        # Helper-level pin: the {"id", "root_task_id"} shape the dispatch seam
        # assembles (row -> event -> durable result) releases the latch of a
        # settled tree. The cascade itself lives in _finish_task_done_dispatch
        # and is exercised by the emitter-stamp pin above, not here.
        from supervisor import queue as q
        from supervisor.queue_transitions import clear_budget_root_fence_for_settled_tree

        monkeypatch.setattr(q, "PENDING", [])
        monkeypatch.setattr(q, "RUNNING", {})
        monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", {
            "r1": {"status": "paused", "root_task_id": "r1"},
        })
        assert clear_budget_root_fence_for_settled_tree(
            {"id": "child", "root_task_id": "r1"}
        ) is True
        assert "r1" not in q.BUDGET_ROOT_FENCES


class TestOrphanFenceSweep:
    def test_restore_drops_dead_tree_fence_and_keeps_live_one(self, tmp_path, monkeypatch):
        import json as _json

        from ouroboros.utils import utc_now_iso
        from supervisor import queue as q

        import supervisor.task_lifecycle as tl

        fresh_fences = {}
        fresh_pending = []
        monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(q, "QUEUE_SNAPSHOT_PATH", tmp_path / "state" / "queue_snapshot.json")
        # One shared object per name across BOTH modules: restore_queue_fences
        # mutates task_lifecycle's global, the sweep reads queue's import.
        monkeypatch.setattr(q, "PENDING", fresh_pending)
        monkeypatch.setattr(tl, "PENDING", fresh_pending, raising=False)
        monkeypatch.setattr(q, "RUNNING", {})
        monkeypatch.setattr(q, "BUDGET_ROOT_FENCES", fresh_fences)
        monkeypatch.setattr(tl, "BUDGET_ROOT_FENCES", fresh_fences)
        (tmp_path / "state").mkdir(parents=True)
        (tmp_path / "logs").mkdir(parents=True)
        snapshot = {
            "ts": utc_now_iso(),
            "pending": [{
                "id": "live-root", "task": {
                    "id": "live-root", "type": "task", "text": "x", "chat_id": 1,
                    "_budget_pause": {"status": "paused_before_dispatch"},
                },
            }],
            "running": [],
            "acceptance_fences": [],
            "budget_root_fences": [
                {"status": "paused", "root_task_id": "live-root", "fence_id": "f1"},
                {"status": "paused", "root_task_id": "dead-root", "fence_id": "f2"},
            ],
        }
        q.QUEUE_SNAPSHOT_PATH.write_text(_json.dumps(snapshot), encoding="utf-8")
        q.restore_pending_from_snapshot()
        # The dead tree's latch cannot outlive its members across a restart;
        # the live tree keeps its fence for the explicit resume.
        assert "dead-root" not in q.BUDGET_ROOT_FENCES
        assert "live-root" in q.BUDGET_ROOT_FENCES
