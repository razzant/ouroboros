"""The incoming content contracts survive the distance forms and their transient caches."""
import dataclasses
import hashlib
from types import SimpleNamespace

import pytest

from ouroboros import memory_floor as mf, memory_view as mv
from ouroboros.project_dialogue import append_canonical_task_summary
from tests.test_memory_view_content_contract import world as world, _request


@pytest.mark.parametrize("folded", [False, True])
def test_working_notes_stay_on_the_task_when_its_gist_moves_to_a_work_line(world, folded):
    from ouroboros.main_context_authority import publish_working_notes
    from tests.test_context_renewal import NOTE, _noted

    root = world.memory.drive_root
    publish_working_notes(SimpleNamespace(task_id="earlier-beta", drive_root=root), _noted(root))
    task = {"id": "reader-beta", "chat_id": world.beta["chat_id"]}
    snap = mv.capture_memory_view(root, task, mv.view_spec_for_task(task, root))
    entry = next(t for t in snap.room["lane2"] if t["task"] == "earlier-beta")
    notes = entry["line"].split("; working notes kept: ", 1)[1]
    assert entry["line_without_gist"].endswith("; working notes kept: " + notes)
    level = mv.FloorLevel((("F2", ("work:earlier-beta:final",)),)) if folded else mv.FULL_VIEW
    text = mv.render_room(snap, level)
    assert "; working notes kept: " + notes in text and NOTE not in text
    assert ("; work earlier-beta]" in text) is folded
    assert "The orchard measurements are complete." in text


@pytest.mark.parametrize("folded", [False, True])
def test_remote_task_keeps_the_cards_outcome_when_fold_read_carries_its_gist(world, folded):
    root = world.memory.drive_root
    assert append_canonical_task_summary(root, {
        "summary_id": "new-ending", "type": "task_summary", "direction": "system",
        "summary_kind": "terminal_root_projection", "status": "cancelled", "outcome": "Stopped",
        "task_id": "earlier-beta", "root_task_id": "earlier-beta", "delegation_role": "root",
        "chat_id": world.beta["chat_id"], "ts": "2026-11-10T09:15:00+00:00",
        "text": "Stopped by my human.", "wake_summary": "The orchard work stopped.", "fold_read": folded,
    })
    task = world.tasks["main"]
    snap = mv.capture_memory_view(root, task, mv.view_spec_for_task(task, root))
    room = next(r for g in snap.conversations for r in g["rooms"] if r["room_id"] == str(world.beta["chat_id"]))
    entry = next(t for t in room["tasks"] if t["task"] == "earlier-beta")
    assert "] Stopped;" in entry["line"] and "] Stopped;" in entry["line_without_gist"]
    text = mv.render_room(snap)
    assert text.count("The orchard work stopped.") == 1
    assert "] Stopped;" in text


def test_form_caches_leave_the_neighbors_capture_hash_and_equality_unchanged(world):
    _messages, _cap, ctx = _request(world, "main")
    core = ctx.context_fit_plan.core
    snap = mv.snapshot_from_json(core.memory_view_json)
    before = mv.snapshot_json(snap)
    other = dataclasses.replace(snap, chronicle_sequence=(core.memory_view_sequence or 0) + 100)
    level = mf.fit_memory_view(snap, {"margin": 0, "physical": 0, "budget": 0})
    mv.story_chars(snap, level)
    mv.room_chars(snap, level)
    assert "_form_catalogue" in snap.__dict__ and "_pieces" in snap.__dict__
    assert snap == other and mv.snapshot_json(snap) == before == mv.snapshot_json(other)
    assert hashlib.sha256(mv.snapshot_json(snap).encode()).digest() == hashlib.sha256(before.encode()).digest()
    assert all(not name.startswith("_") for name in dataclasses.asdict(snap))
    assert not any(name.startswith("_") for name in other.__dict__)
