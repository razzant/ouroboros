"""A new head sees the roster's dated focus, questions and holds outside its prefix."""
import json
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
                    {"id": "child", "task": {"id": "child", "parent_task_id": "moved"}},
                    # The owner's direct head: a conversation-lane row of the same snapshot, never a separate fragment.
                    {"id": "native", "attempt": 1, "lane": "conversation", "task": {
                        "id": "native", "chat_id": 1, "title": "A direct head", "_is_direct_chat": True,
                        "_execution_owner": {"kind": "direct", "lane": "conversation", "executor": "worker"}}}],
        "pending": [{"id": "sleeper", "attempt": 1, "task": {"id": "sleeper", "chat_id": 1,
            "_project_admission_restore_hold": {"reason": "project_dispatch_unconfirmed"}}}]})
    return project, focus


def _runtime(rows):
    """The Runtime context object as the model reads it: the JSON its section carries, not the text around it."""
    marker = "## Runtime context\n\n"
    for row in rows:
        content = row["content"]
        for text in [content] if isinstance(content, str) else [block.get("text") or "" for block in content]:
            if marker in text:
                return json.JSONDecoder().raw_decode(text.split(marker, 1)[1])[0]
    raise AssertionError("the head carries no Runtime context section")


def _present(wire):
    """The live-works present as the head's own non-system messages carry it, with its works by id."""
    present = _runtime([row for row in wire if row["role"] != "system"])["live_works_present"]
    return present, {work["task_id"]: work for work in present["works"]}


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
    # The dated focus stands once, whole: no second copy of its text and date beside it.
    assert moved["focus"] == focus and not {"last_line", "last_line_at"} & set(moved)
    assert moved["started_at"] == "2026-10-16T11:00:00Z"
    assert moved["open_questions"] == [{"quiz_id": "q1", "state": "open", "question": "Which version?", "options": ["A", "B"]}]
    assert moved["waiting"][0]["kind"] == "owner"
    sleep = works["sleeper"]
    assert sleep["state"] == "pending" and sleep["waiting"][0]["kind"] == "sleep"
    assert sleep["project_admission_hold"]["reason"] == "project_dispatch_unconfirmed"
    assert sleep["open_questions"] == [] and "focus" not in sleep
    note = render_roster_note(independent_roots(tmp_path))
    assert "Which version?" in note and "project_dispatch_unconfirmed" in note
    assert "Different web narration" not in note
    write_task_result(tmp_path, "moved", "completed")
    settled = next(row for row in live_works_present(tmp_path)["works"] if row["task_id"] == "moved")
    assert settled["recorded_status"] == "completed" and "focus" not in settled
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
    _, focus = _fixture(env.drive_root)
    # Another work carries the focus's words as its title: a legitimate coincidence that outlives both builds.
    write_task_result(env.drive_root, "sleeper", "running", title=focus["text"])
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
    # The objects themselves, parsed from the head: the work `moved` stands with its whole dated focus and its open
    # question, and the web card's `last_line` is a field of neither its row nor the present. Where else the same
    # words or a `last_line` stand is another object's fact.
    present, works = _present(wire)
    assert works["moved"]["focus"] == focus and works["moved"]["open_questions"] == [
        {"quiz_id": "q1", "state": "open", "question": "Which version?", "options": ["A", "B"]}]
    assert not {"last_line", "last_line_at"} & set(works["moved"]) and "last_line_source" not in present
    assert works["sleeper"]["title"] == focus["text"]
    assert "live_works_present" not in str(prefix)
    write_task_result(env.drive_root, "moved", "completed")
    newer, _ = context.build_llm_messages(env=env, memory=memory, task=task)
    fresh, _ = split_leading_system_prefix(newer)
    assert [row for row in fresh if row["role"] == "system"] == prefix
    # Queued heads refresh the older metadata snapshot at context construction: the same work, now settled.
    present, works = _present(fresh)
    assert works["moved"].get("recorded_status") == "completed" and "focus" not in works["moved"]
    assert works["moved"]["open_questions"] == [] and not {"last_line", "last_line_at"} & set(works["moved"])
    assert "last_line_source" not in present and works["sleeper"]["title"] == focus["text"]


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
    from ouroboros import projects_registry
    from ouroboros.peer_roster import independent_roots
    from ouroboros.store import StoreBusy

    _fixture(tmp_path)
    def busy(*_args, **_kwargs):
        raise StoreBusy("busy")
    from ouroboros import task_store

    monkeypatch.setattr(projects_registry, "project_binding_for_task", busy)
    monkeypatch.setattr(task_store, "documents_by_ids", busy)  # the roster's one read of every root's result
    roster = independent_roots(tmp_path)
    assert roster["incomplete"] is True
    assert all(row["unavailable"] == {"task_result": "StoreBusy", "project": "StoreBusy"}
               for row in roster["roots"])


def test_a_stop_states_its_own_press_once_and_only_where_that_is_proven(tmp_path):
    """Data as ``owner_pause.next_fence`` records it: the creating press on the fence and again under
    ``requests[request_id]``. The repeated entry keeps its key and points at the stop; every other shape stays whole."""
    from ouroboros.live_works import OWN_PRESS, _stop_shown

    by = {"actor": "ouroboros", "task_id": "m1", "text": "pausing for the update", "at": "2026-11-20T10:00:00Z"}
    other = {"by": {"actor": "owner", "text": "", "at": "2026-11-20T10:05:00Z"}, "fence_id": "f2", "generation": 2}
    earlier = {"fence_id": "f1", "generation": 1}  # an earlier generation's press, recorded without an author
    fence = {"by": by, "fence_id": "f2", "request_id": "r2", "state": "paused", "requested_at": by["at"],
             "generation": 2, "requests": {"r1": earlier, "r2": {"by": by, "fence_id": "f2", "generation": 2}, "r3": other}}
    shown = _stop_shown(fence)
    assert shown["requests"] == {"r1": earlier, "r2": OWN_PRESS, "r3": other}  # every key, every other press
    assert {key: value for key, value in shown.items() if key != "requests"} == {
        key: value for key, value in fence.items() if key != "requests"}
    assert _stop_shown({"scope": "installation", "member_stop": fence})["member_stop"] == shown
    # Not proven, so whole: the own key absent; an own entry that differs; a fence that declares no author, where
    # nothing may read as one; a later press that filled the author of a fence whose creating press named nobody.
    absent = {**fence, "requests": {"r1": earlier, "r3": other}}
    differs = {**fence, "requests": {**fence["requests"], "r2": {**fence["requests"]["r2"], "generation": 1}}}
    nobody = {key: value for key, value in fence.items() if key != "by"}
    nobody["requests"] = {"r1": earlier, "r2": {"fence_id": "f2", "generation": 2}}
    filled = {**nobody, "by": other["by"], "requests": {**nobody["requests"], "r3": other}}
    for stop in (absent, differs, nobody, filled, {}, None):
        assert _stop_shown(stop) == stop
    # Through the roster: the head's record of a paused root carries that one stop.
    write_task_result(tmp_path, "t", "running", title="Paused", chat_id=1, owner_pause=fence)
    write_queue_document(tmp_path / "state/queue_snapshot.json", {"ts": utc_now_iso(), "pending": [], "running": [
        {"id": "t", "attempt": 1, "task": {"id": "t", "chat_id": 1}}]})
    [work] = live_works_present(tmp_path)["works"]
    assert work["stop"]["by"] == by and work["stop"]["requests"] == shown["requests"]
