"""knowledge_list(topics=...) is the read SYSTEM.md points at when the index shows only its outline."""
from __future__ import annotations

import pathlib

import pytest

from ouroboros.tools.knowledge import _knowledge_list, get_tools


class _Ctx:
    def __init__(self, drive_root):
        self.drive_root = pathlib.Path(drive_root)
        self.project_id = ""
        self.task_id = "t1"

    def drive_path(self, rel):
        return self.drive_root / rel


def _shelf(tmp_path):
    ctx = _Ctx(tmp_path / "drive")
    kdir = ctx.drive_root / "memory" / "knowledge"
    (kdir / "people").mkdir(parents=True)
    (kdir / "git-recipes.md").write_text(
        "---\ntype: procedure\nsummary: Use rebase sparingly.\n---\n# Git recipes\n", encoding="utf-8")
    (kdir / "people" / "elena.md").write_text(
        "---\ntype: person\nsummary: Prefers short calls.\n---\n# Elena\n", encoding="utf-8")
    return ctx


def test_topics_narrow_the_listing_and_an_omitted_list_keeps_the_whole_shelf(tmp_path):
    ctx = _shelf(tmp_path)
    whole = _knowledge_list(ctx)
    assert "rebase" in whole and "short calls" in whole
    narrowed = _knowledge_list(ctx, topics=["people/"])
    assert "short calls" in narrowed and "rebase" not in narrowed
    exact = _knowledge_list(ctx, topics=["git-recipes"])
    assert "rebase" in exact and "short calls" not in exact


def test_topics_must_be_a_list_of_strings(tmp_path):
    ctx = _shelf(tmp_path)
    assert "TOOL_ARG_ERROR" in _knowledge_list(ctx, topics="people/")
    assert "TOOL_ARG_ERROR" in _knowledge_list(ctx, topics=[3])


def test_the_tool_schema_offers_topics():
    entry = next(tool for tool in get_tools() if tool.name == "knowledge_list")
    properties = entry.schema["parameters"]["properties"]
    assert properties["topics"]["type"] == "array"
    assert "topics" not in entry.schema["parameters"]["required"]


@pytest.mark.parametrize("topics,unmatched,count", [
    (["people/"], [], 0), (["people"], ["people"], 0),
    (["people/", "missing"], ["missing"], 0), (["old"], ["old"], 1),
])
def test_narrowed_listing_names_selection_unmatched_selectors_and_scoped_archives(tmp_path, topics, unmatched, count):
    import json
    from ouroboros.knowledge import read_knowledge_note, resolve_knowledge_address, write_knowledge_note

    ctx = _shelf(tmp_path)
    old = resolve_knowledge_address(ctx.drive_root, "old", "global")
    assert write_knowledge_note(old, "# Earlier note\n").ok
    assert write_knowledge_note(old, "", mode="archive", reason="Superseded",
                                expected_revision=read_knowledge_note(old).revision).ok
    text = _knowledge_list(ctx, topics=topics)
    assert "Selection: " + json.dumps(topics) in text
    assert f"Archived notes in this selection: {count}." in text
    assert ("Unmatched selectors in active view:" in text) == bool(unmatched)
    if unmatched:
        assert "Unmatched selectors in active view: " + json.dumps(unmatched) in text
    whole = _knowledge_list(ctx)
    assert "Archived notes: 1." in whole and "Selection:" not in whole
    assert "Unmatched selectors" not in whole
    archived = _knowledge_list(ctx, topics=["old"], view="archived")
    assert "Unmatched selectors" not in archived and "Earlier note" in archived
