"""Wake directory stays readable without standing copies of other rooms' words."""
from ouroboros import memory_view as mv, memory_floor as mf
from ouroboros.focus import normalize_focus
from ouroboros.task_results import write_task_result
from tests.test_memory_view_room import _install, _view, _section, MAIN_TASK
from tests._memory_view_synthetic import live_room, snapshot


def test_wake_keeps_ordinary_main_and_only_latest_authored_focus_elsewhere(tmp_path):
    rooms, _rows = _install(tmp_path)
    focus = normalize_focus("Waiting for a choice\nthen implement", {"reader": "get_task_result", "task_id": "ta1"},
                            task_id="ta1", authored_at="2026-09-05T10:20:00+00:00")
    write_task_result(tmp_path, "ta1", "completed", focus=focus)
    write_task_result(tmp_path, "bound", "completed", focus={**focus, "author_task_id": "bound",
        "text": "Earlier bound focus", "authored_at": "2026-09-05T10:19:59+00:00"})
    write_task_result(tmp_path, "tb1", "completed", focus={**focus, "text": "FOREIGN SIGNATURE"})
    main, main_text = _view(tmp_path, MAIN_TASK)
    wake, text = _view(tmp_path, {"id": "wake001", "usage_category": "consciousness"})
    # A wake is one turn beside its whole memory, as a Main turn is: one eighth of work room each (owner, 29.10).
    assert wake.spec.live_rooms == "directory" and wake.spec.work_room_levels == 1
    assert main.spec.work_room_levels == 1
    assert _section(text, "## This room (Main)") == _section(main_text, "## This room (Main)")
    live = _section(text, "## Live rooms")
    lines = [line for line in live.splitlines() if line.startswith("- ")]
    assert len(lines) == len(wake.live_rooms) == 3
    assert "Waiting for a choice then implement" in live and "FOREIGN SIGNATURE" not in live
    assert "Earlier bound focus" not in live
    assert "alpha words" not in live and "beta words" not in live
    assert all("last activity" in line and "people " in line and "memory_read(" in line for line in lines)
    assert f"memory_read(room_id='{rooms['alpha']}', rows=true)" in live
    _snap, explicit = _view(tmp_path, MAIN_TASK, live_rooms="lines_with_words")
    assert "alpha words" in _section(explicit, "## Live rooms")  # a chosen different form remains possible
    write_task_result(tmp_path, "ta1", "completed", focus={**focus, "text": "Changed after capture"})
    assert mv.render_room(wake) == text
    assert "Changed after capture" in _view(tmp_path, {"id": "wake002", "usage_category": "consciousness"})[1]


def test_164_room_directory_collapses_to_one_bounded_reader_under_pressure():
    rooms = [{**live_room(i, notes=1), "directory": True, "focus": "One authored focus", "words": []}
             for i in range(164)]
    snap = snapshot(role="consciousness", live=rooms)
    whole = mv.render_room(snap)
    assert len([line for line in whole.splitlines() if line.startswith("- ")]) == 164
    assert "What this project is." not in whole
    elements = mf.floor_elements(snap)
    assert not any(step in ("F6r", "F6") for step, *_ in elements)
    level = mf.fit_memory_view(snap, {"margin": 200, "physical": 200, "budget": None})
    small = mv.render_room(snap, level)
    assert "164 open rooms;" in small and "memory_read(room_id='*', start=0)" in small
    assert len(small) < 600 and "One authored focus" not in small
    roomy = mf.fit_memory_view(snap, {"margin": 1_000_000, "physical": 1_000_000, "budget": None})
    assert not roomy.addressed and mv.render_room(snap, roomy) == whole
