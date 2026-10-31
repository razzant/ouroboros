"""One context build parses its captured memory view once; no consumer mutates the shared parse.

The fit plan, the floor's inventory and selection, and the memory line each read the view
from ``core.memory_view_json``; on Main that text is megabytes, parsed three times a turn.
"""

from __future__ import annotations

import json


def test_one_build_parses_its_view_once_and_no_consumer_mutates_it(tmp_path, monkeypatch):
    from ouroboros import context_view_facts, memory_floor, memory_view as mv
    from tests.test_memory_view_modes import _built

    core, plan = _built(tmp_path, monkeypatch, books=0)
    mv._parsed_view.cache_clear()
    parses, real = [], json.loads
    monkeypatch.setattr(mv.json, "loads", lambda raw, *args, **kwargs: (
        parses.append(1) if raw == core.memory_view_json else None) or real(raw, *args, **kwargs))

    built = plan(10_000_000)
    assert memory_floor.view_inventory(built, "max")
    assert context_view_facts.memory_snapshot(built) is mv.snapshot_from_json(core.memory_view_json)
    context_view_facts.memory_facts(built, "max")
    memory_floor.select_view(built, "max", [])

    assert len(parses) == 1
    assert mv.snapshot_json(mv.snapshot_from_json(core.memory_view_json)) == core.memory_view_json
