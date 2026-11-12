"""A wake holds my other conversations as every integrating focus does (owner 06.11, Q2-A): the same block as a
Main turn, running work under its own latest focus; under pressure the groups fold to their lines and, last, to one
bounded count with a reader."""
from ouroboros import memory_view as mv, memory_floor as mf
from ouroboros.focus import normalize_focus
from ouroboros.task_results import write_task_result
from tests.test_memory_view_room import _install, _view, _section, MAIN_TASK
from tests._memory_view_synthetic import conversation, snapshot


def test_a_wake_holds_the_block_a_main_turn_holds_and_running_work_names_its_own_latest_focus(tmp_path):
    rooms, _rows = _install(tmp_path)
    focus = normalize_focus("Waiting for a choice\nthen implement", {"reader": "get_task_result", "task_id": "ta1"},
                            task_id="ta1", authored_at="2026-09-05T10:20:00+00:00")
    write_task_result(tmp_path, "ta1", "running", focus=focus)
    write_task_result(tmp_path, "tb1", "running", focus={**focus, "text": "FOREIGN SIGNATURE"})  # ta1's, not tb1's
    write_task_result(tmp_path, "bound", "completed", focus={**focus, "author_task_id": "bound", "text": "Done focus"})
    main, main_text = _view(tmp_path, MAIN_TASK)
    wake, text = _view(tmp_path, {"id": "wake001", "usage_category": "consciousness"})
    # A wake is one turn beside its whole memory, as a Main turn is: one eighth of work room each (owner, 29.10).
    assert wake.spec.other_conversations and wake.spec.work_room_levels == 1 and main.spec.work_room_levels == 1
    assert _section(text, "## This room (Main)") == _section(main_text, "## This room (Main)")
    block = _section(text, "## Other conversations")
    assert block == _section(main_text, "## Other conversations")
    alpha = _section(block, f"### Project Alpha [chat_id={rooms['alpha']}]")
    assert "running get_task_result(task_id='ta1'): Waiting for a choice then implement (2026-09-05 10:20)" in alpha
    assert "FOREIGN SIGNATURE" not in block and "Done focus" not in block and "running get_task_result(task_id='tb1')" in block
    assert "alpha words" in alpha and f"memory_read(room_id='{rooms['alpha']}', rows=true)" in alpha
    write_task_result(tmp_path, "ta1", "running", focus={**focus, "text": "Changed after capture"})
    assert mv.render_room(wake) == text  # a captured view renders the same bytes
    assert "Changed after capture" in _view(tmp_path, {"id": "wake002", "usage_category": "consciousness"})[1]


def test_164_conversations_fold_to_their_lines_and_last_to_one_bounded_count_under_pressure():
    groups = [conversation(i, notes=1) for i in range(164)]
    snap = snapshot(role="consciousness", conversations=groups)
    whole = mv.render_room(snap)
    assert whole.count("\n### Project room ") == 164 and whole.count("What this project is.") == 164
    level = mf.fit_memory_view(snap, {"margin": 200, "physical": 200, "budget": None})
    small = mv.render_room(snap, level)
    # Each group yields to its emergency line first, then the common count takes conversations one by one (plan
    # commit 2, step 7): here all 164, which still print as one bounded count line with its catalogue reader.
    assert dict(level.steps)["F6g"] == 164 and dict(level.steps)["F6a"] == 164 and "unread" not in dict(level.addressed)["F6a"]
    assert small.count("conversations by count") == 1 and "164 conversations by count: 164 open rooms;" in small
    assert "memory_read(room_id='*', start=0)" in small and len(small) < 1_500 and "What this project is." not in small
    roomy = mf.fit_memory_view(snap, {"margin": 1_000_000, "physical": 1_000_000, "budget": None})
    assert not roomy.addressed and mv.render_room(snap, roomy) == whole
