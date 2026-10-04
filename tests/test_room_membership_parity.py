"""``memory_inventory.rooms_of_row`` puts a row in exactly the rooms ``room_membership`` does.

The inventory decides membership from row metadata (no text), once for all rooms, so
it can be recomputed on every capture; the canonical rule lives in
``project_dialogue.room_membership`` and ``chat_chain.iter_room_rows``. The parity
runs over Main, two Projects, a transport chat and the hidden partition, a task bound
by an owner message written in Main (one row in two rooms), lifecycle rows pinned to
Main and origin-addressed notices, both in the chain and as synthetic rows.
"""
from __future__ import annotations

import itertools

from ouroboros import chat_chain
from ouroboros import memory_inventory as mi
from tests import _memory_inventory_shared as shared


def _canonical(root):
    from ouroboros.project_dialogue import room_membership, source_refs_for_project
    from ouroboros.projects_registry import all_task_bindings, list_reserved_projects

    projects = {int(project["chat_id"]) for project in list_reserved_projects(root)}
    bindings = all_task_bindings(root)

    def member(room: int):
        refs = source_refs_for_project(root, room) if room in projects else []
        return room_membership(room, projects, refs, bindings)

    return member


def _synthetic(rooms):
    from ouroboros.project_dialogue import MAIN_PINNED_ROW_TYPES, ORIGIN_ADDRESSED_NOTICE_TYPES

    kinds = sorted(MAIN_PINNED_ROW_TYPES | ORIGIN_ADDRESSED_NOTICE_TYPES) + ["", "task_summary"]
    lineages = [{}, {"task_id": "bound"}, {"task_id": "kid", "parent_task_id": "bound", "root_task_id": "bound"},
                {"task_id": "free"}]
    chats = [1, rooms["alpha"], rooms["beta"], 777, 0, -5]
    for index, (kind, lineage, chat) in enumerate(itertools.product(kinds, lineages, chats)):
        row = shared.msg(f"2026-09-05T00:{index // 60:02d}:{index % 60:02d}+00:00", f"synthetic {index}",
                         chat_id=chat, direction="system" if kind else "out", **lineage)
        if kind:
            row["type"] = kind
        yield row
    yield shared.msg("2026-09-01T00:05:00+00:00", shared.ORIGIN, client_message_id="origin-a")  # bound origin
    yield shared.msg("2026-09-01T00:05:00+00:00", shared.ORIGIN, client_message_id="")  # ref without client id
    yield shared.msg("2026-09-01T00:05:00+00:00", shared.ORIGIN + "!", client_message_id="origin-a")  # other text


def test_rooms_of_row_matches_room_membership_for_every_row_and_room(tmp_path):
    rooms = shared.world(tmp_path, activate=False)
    member = _canonical(tmp_path)
    facts = mi.membership_facts(tmp_path)
    candidates = [1, rooms["alpha"], rooms["beta"], 777, 0]
    chain = [row for _address, row, _pos in chat_chain.iter_rows(tmp_path)]
    assert len(chain) >= 18
    seen = {room: set() for room in candidates}
    for row in [*chain, *_synthetic(rooms)]:
        found = mi.rooms_of_row(mi.row_meta(row), facts)
        for room in candidates:
            expected = member(room)(chat_chain._row_chat_id(row), row)
            assert (str(room) in found) is expected, (room, row)
            seen[room].add(expected)
    assert all(values == {True, False} for values in seen.values()), seen  # every room both holds and excludes


def test_one_owner_message_is_in_main_and_in_the_project_it_started(tmp_path):
    rooms = shared.world(tmp_path, activate=False)
    facts = mi.membership_facts(tmp_path)
    origin = next(row for _a, row, _p in chat_chain.iter_rows(tmp_path) if row.get("client_message_id") == "origin-a")
    assert mi.rooms_of_row(mi.row_meta(origin), facts) == {"1", str(rooms["alpha"])}
    bound = next(row for _a, row, _p in chat_chain.iter_rows(tmp_path) if row.get("task_id") == "bound"
                 and row.get("direction") == "out")
    assert mi.rooms_of_row(mi.row_meta(bound), facts) == {str(rooms["alpha"])}
    notice = next(row for _a, row, _p in chat_chain.iter_rows(tmp_path) if row.get("type") == "task_not_started")
    assert mi.rooms_of_row(mi.row_meta(notice), facts) == {"1"}  # the refusal stays with the issuing chat
    started = next(row for _a, row, _p in chat_chain.iter_rows(tmp_path) if row.get("type") == "project_started")
    assert mi.rooms_of_row(mi.row_meta(started), facts) == {"1"}


def test_room_sets_equal_the_canonical_room_reader(tmp_path):
    rooms = shared.world(tmp_path, activate=False)
    facts = mi.membership_facts(tmp_path)
    stream = [(mi.rooms_of_row(mi.row_meta(row), facts), pos) for _address, row, pos in chat_chain.iter_rows(tmp_path)]
    for room in ("1", str(rooms["alpha"]), str(rooms["beta"]), "777", "0"):
        canonical = [pos for _a, _r, pos in chat_chain.iter_room_rows(tmp_path, room)]
        assert canonical and [pos for found, pos in stream if room in found] == canonical, room


def test_bound_owner_messages_are_the_project_source_refs(tmp_path):
    from ouroboros.project_dialogue import _source_ref_identity, source_refs_for_project

    rooms = shared.world(tmp_path, activate=False)
    facts = mi.membership_facts(tmp_path)
    assert facts.project_chat_ids == {rooms["alpha"], rooms["beta"]}
    for chat in facts.project_chat_ids:
        canonical = {_source_ref_identity(ref) for ref in source_refs_for_project(tmp_path, chat)}
        inventory = {key for key, chats in facts.projects_by_source_key.items() if chat in chats}
        assert inventory == canonical
    assert any(facts.projects_by_source_key.values())
