"""A new head sees the roster's dated focus, questions and holds outside its prefix."""
from types import SimpleNamespace

from ouroboros.live_works import live_works_present
from ouroboros.task_results import write_task_result
from ouroboros.utils import utc_now_iso
from tests._shared import write_queue_document


def _fixture(root):
    from ouroboros.projects_registry import bind_task_to_project, create_project
    project = create_project(root, "room", name="Room")
    bind_task_to_project(root, "moved", project["id"], project["chat_id"], origin={"absent": "system"})
    focus = {"text": "Verifying the authoring path", "author_task_id": "moved",
             "authored_at": "2026-10-16T12:00:00Z", "source_ref": {"reader": "workpad_read", "project_id": "room"}}
    write_task_result(root, "moved", "running", title="A moved work", chat_id=1, focus=focus,
        started_at="2026-10-16T11:00:00Z", last_line="Different web narration",
        owner_quiz={"q1": {"state": "open", "question": "Which version?", "options": ["A", "B"]},
                    "q0": {"state": "answered", "question": "Old question"}},
        owner_wait={"state": "waiting", "quiz_id": "q1", "task_attempt": 1, "reason": "owner"})
    write_task_result(root, "sleeper", "running", owner_wait={"state": "waiting", "task_attempt": 1,
        "reason": "sleep", "quiz_id": "", "sleep": {"mode": "warm", "wake_at": "2026-10-16T13:00:00Z"}})
    stamp = utc_now_iso()
    write_queue_document(root / "state/queue_snapshot.json", {"ts": stamp,
        "running": [{"id": "moved", "attempt": 1, "task": {"id": "moved", "chat_id": 1}},
                    {"id": "child", "task": {"id": "child", "parent_task_id": "moved"}}],
        "pending": [{"id": "sleeper", "attempt": 1, "task": {"id": "sleeper", "chat_id": 1,
            "_project_admission_restore_hold": {"reason": "project_dispatch_unconfirmed"}}}]})
    write_queue_document(root / "state/direct_roots.json", {"ts": stamp, "incomplete": False,
        "roots": [{"task_id": "native", "chat_id": 1, "title": "A direct head", "started_at": stamp}]})
    return project, focus


def test_present_uses_roster_focus_questions_holds_and_exact_bound_room(tmp_path):
    from ouroboros.peer_roster import independent_roots, render_roster_note
    project, focus = _fixture(tmp_path)
    present = live_works_present(tmp_path)
    assert present["incomplete"] is False
    works = {row["task_id"]: row for row in present["works"]}
    assert set(works) == {"moved", "sleeper", "native"}
    moved = works["moved"]
    assert moved["title"] == "A moved work" and moved["state"] == "running"
    assert moved["room"] == {"chat_id": project["chat_id"], "project_id": "room", "project_name": "Room"}
    assert moved["last_line"] == focus["text"] and moved["last_line_at"] == focus["authored_at"]
    assert moved["started_at"] == "2026-10-16T11:00:00Z"
    assert moved["open_questions"] == [{"quiz_id": "q1", "state": "open", "question": "Which version?", "options": ["A", "B"]}]
    assert moved["waiting"][0]["kind"] == "owner"
    sleep = works["sleeper"]
    assert sleep["state"] == "pending" and sleep["waiting"][0]["kind"] == "sleep"
    assert sleep["project_admission_hold"]["reason"] == "project_dispatch_unconfirmed"
    assert sleep["open_questions"] == [] and sleep["last_line"] == ""
    note = render_roster_note(independent_roots(tmp_path))
    assert "Which version?" in note and "project_dispatch_unconfirmed" in note
    assert "Different web narration" not in note
    write_task_result(tmp_path, "moved", "completed")
    settled = next(row for row in live_works_present(tmp_path)["works"] if row["task_id"] == "moved")
    assert settled["recorded_status"] == "completed" and settled["last_line"] == ""
    assert settled["open_questions"] == [] and settled["waiting"] == []


def test_missing_roster_is_disclosed_not_an_authoritative_empty_present(tmp_path):
    present = live_works_present(tmp_path)
    assert present["incomplete"] is True and present["works"] == []
    _fixture(tmp_path)
    assert live_works_present(tmp_path)["incomplete"] is False


def test_the_new_head_reads_the_present_at_capture_outside_the_cache_prefix(tmp_path, monkeypatch):
    """Routing hands the head no copy of the roster: its context capture builds the present."""
    from ouroboros import context, server_routing_context as routing
    from ouroboros.llm_messages import split_leading_system_prefix
    from tests.test_cache_optimization import _make_env_and_memory
    env, memory = _make_env_and_memory(tmp_path)
    _fixture(env.drive_root)
    ctx = SimpleNamespace(DRIVE_ROOT=env.drive_root, RUNNING={}, PENDING=[])
    monkeypatch.setattr(routing, "_active_direct_roots", lambda ctx: [])
    # The subagent roster is process-wide state another test may change between the two builds; this test is about
    # where the live-works present lands, so the roster is held still.
    import ouroboros.subagent_runtime as subagent_runtime
    monkeypatch.setattr(subagent_runtime, "current_model_visible_subagent_catalog", lambda: {"rows": []})
    md = routing._decision_turn_metadata(ctx, 1, "new-request", {})
    assert "live_works_present" not in md
    task = {"id": "new-head", "type": "task", "text": "Next request", "metadata": md}
    messages, _ = context.build_llm_messages(env=env, memory=memory, task=task)
    wire, moved = split_leading_system_prefix(messages)
    assert moved > 0
    prefix = [row for row in wire if row["role"] == "system"]
    tail = str([row for row in wire if row["role"] != "system"])
    assert "Verifying the authoring path" in tail and "Which version?" in tail
    assert "live_works_present" not in str(prefix)
    write_task_result(env.drive_root, "moved", "completed")
    newer, _ = context.build_llm_messages(env=env, memory=memory, task=task)
    fresh, _ = split_leading_system_prefix(newer)
    assert [row for row in fresh if row["role"] == "system"] == prefix
    # Queued heads refresh the older metadata snapshot at context construction.
    assert "Verifying the authoring path" not in str([row for row in fresh if row["role"] != "system"])


def test_new_head_discloses_prepared_and_uncertain_intents_without_replaying(tmp_path):
    from ouroboros.promotion_intent import prepare_promotion_intent, claim_promotion_intent, reconcile_promotion_intents
    from ouroboros.server_routing_context import _recent_root_results
    from ouroboros.task_results import load_task_result
    events = [{"type": "promote_chat_to_task", "task_id": task_id, "routing_token": task_id,
               "objective": "Recover work explicitly", "title": task_id} for task_id in ("prepared", "claimed")]
    for event in events:
        prepare_promotion_intent(tmp_path, event)
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path)
    before = {row["task_id"]: row for row in _recent_root_results(ctx)[0]}
    assert before["prepared"]["promotion_admission"]["phase"] == "prepared"
    assert before["prepared"]["promotion_intent_available"] is True
    claim_promotion_intent(tmp_path, events[1])
    reconcile_promotion_intents(tmp_path, protected=set())
    after = {row["task_id"]: row for row in _recent_root_results(ctx)[0]}
    assert after["prepared"]["promotion_admission"]["status"] == "not_started"
    assert after["prepared"]["promotion_admission"]["retry_available"] is True
    assert "explicit retry" in after["prepared"]["admission_notice"]
    assert after["claimed"]["promotion_admission"]["status"] == "unconfirmed"
    assert after["claimed"]["promotion_admission"]["retry_available"] is False
    assert "held for reconciliation" in after["claimed"]["admission_notice"]
    for event in events:
        stored = load_task_result(tmp_path, event["task_id"])
        assert not stored.get("execution_owner") and not stored.get("admitted_dispatch")
        assert stored["promotion_intent"] == event


def test_roster_keeps_available_facts_when_project_store_is_busy(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    from ouroboros.peer_roster import independent_roots
    from ouroboros.store import StoreBusy

    project, focus = _fixture(tmp_path)
    def busy(*_args, **_kwargs):
        raise StoreBusy("dialogue busy")
    monkeypatch.setattr(projects_registry, "get_project", busy)
    roster = independent_roots(tmp_path)
    assert roster["incomplete"] is True
    moved = next(row for row in roster["roots"] if row["task_id"] == "moved")
    assert moved["unavailable"] == {"project": "StoreBusy"}
    assert moved["project_id"] == project["id"] and moved["focus"] == focus
    assert moved["open_questions"][0]["quiz_id"] == "q1"
    present = live_works_present(tmp_path)
    assert present["incomplete"] is True
    assert next(row for row in present["works"] if row["task_id"] == "moved")["unavailable"] == moved["unavailable"]


def test_roster_discloses_both_unreadable_task_and_project_enrichment(tmp_path, monkeypatch):
    from ouroboros import projects_registry, task_results
    from ouroboros.peer_roster import independent_roots
    from ouroboros.store import StoreBusy

    _fixture(tmp_path)
    def busy(*_args, **_kwargs):
        raise StoreBusy("busy")
    monkeypatch.setattr(projects_registry, "project_binding_for_task", busy)
    monkeypatch.setattr(task_results, "load_task_result", busy)
    roster = independent_roots(tmp_path)
    assert roster["incomplete"] is True
    assert all(row["unavailable"] == {"task_result": "StoreBusy", "project": "StoreBusy"}
               for row in roster["roots"])
