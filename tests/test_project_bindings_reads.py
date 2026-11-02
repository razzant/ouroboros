"""Every reader of ``state/project_task_bindings.json`` shares one parse; its one writer reads fresh.

The bindings document grows with every bound task (megabytes on long-lived installs) and
routing, the census and the live-log addressing seam read it per root and per frame.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import projects_registry as registry
from ouroboros import utils


@pytest.fixture
def bound(tmp_path, monkeypatch):
    """A room with one bound task; files are trusted as written (the clock-tick rule has its own case)."""
    monkeypatch.setattr(utils, "_RACY_NS", 0)
    registry.bind_task_to_project(tmp_path, "t1", "alpha", origin={"absent": "system"})
    return tmp_path


def _parses_of_bindings(monkeypatch, root) -> list:
    text, seen, real = registry._bindings_path(root).read_text(encoding="utf-8"), [], json.loads
    monkeypatch.setattr(utils.json, "loads", lambda raw, *args, **kwargs: (
        seen.append(raw) if raw == text else None) or real(raw, *args, **kwargs))
    return seen


def _read_everything(root) -> None:
    registry.project_binding_for_task(root, "t1")
    registry.project_binding_for_task(root, "t1", strict=True)
    registry.project_chat_for_task(root, "t1")
    registry.project_id_for_task(root, "t1", strict=True)
    registry.all_task_project_bindings(root)
    registry.all_task_project_bindings(root, strict=True)
    registry.project_task_bindings(root)
    registry.RoomResolver(root)("t1")
    registry.RoomResolver(root, strict=True)("t1")


def test_strict_and_lenient_readers_share_one_parse_until_a_bind_moves_it(bound, monkeypatch):
    parses = _parses_of_bindings(monkeypatch, bound)

    _read_everything(bound)
    assert len(parses) == 1
    shared = registry._load_bindings(bound, strict=True)
    assert registry._load_bindings(bound) is shared and json.loads(
        registry._bindings_path(bound).read_text(encoding="utf-8")) == shared  # no reader mutated it

    registry.bind_task_to_project(bound, "t2", "alpha", origin={"absent": "system"})
    assert registry.project_id_for_task(bound, "t2", strict=True) == "alpha"
    assert registry.project_binding_for_task(bound, "t2")["project_id"] == "alpha"


def test_a_failed_bind_leaves_no_trace_in_any_reader(bound, monkeypatch):
    _read_everything(bound)  # the shared parse is warm

    def refused(_root, _data):
        raise OSError("disk full")

    with monkeypatch.context() as patch:
        patch.setattr(registry, "_save_bindings", refused)
        with pytest.raises(OSError):
            registry.bind_task_to_project(bound, "t2", "alpha", origin={"absent": "system"})

    assert registry.project_binding_for_task(bound, "t2") is None
    assert registry.project_id_for_task(bound, "t2", strict=True) == ""
    assert "t2" not in registry.project_task_bindings(bound) and "t2" not in registry.all_task_project_bindings(bound)
    assert registry.RoomResolver(bound)("t2") == 0
    assert registry.project_chat_for_task(bound, "t1") > 0  # the stored binding still reads
