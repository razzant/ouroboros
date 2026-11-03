"""The durable knowledge index at its outline resolution: every active note keeps its line.

The outline is the index's operative view (``context_sections`` builds both from one
inventory); the host-section floor steps the index to it last, only where the window needs
it, for every caller.
"""
from __future__ import annotations

import pytest

from ouroboros.knowledge import (
    knowledge_index_view, knowledge_index_views, resolve_knowledge_address, write_knowledge_note,
)

RESERVE = 8_192


def _note(root, topic, *, summary, kind="note", title=""):
    head = f"type: {kind}\nsummary: {summary!r}\n" + (f"title: {title!r}\n" if title else "")
    assert write_knowledge_note(resolve_knowledge_address(root, topic, "global"),
                                f"---\n{head}---\n\n# {title or topic}\n\nBody of {topic}.\n").ok


def _shelf(root):
    _note(root, "release-queue-20261016", summary="The queue stalls when two releases race.", title="Release queue")
    _note(root, "patterns", summary="Failure classes and their counters.")
    _note(root, "people/ada", summary="Ada reviews every protocol change on Fridays.", kind="person", title="Ada")
    _note(root, "owner-model", summary="The owner wants one coherent design, not patches.", kind="person")
    _note(root, "recipes/cache", summary="Warm the cache before a benchmark.", title="Cache warm-up")
    archived = resolve_knowledge_address(root, "old-incident", "global")
    _note(root, "old-incident", summary="Fixed in September.")
    from ouroboros.knowledge import read_knowledge_note

    assert write_knowledge_note(archived, "", mode="archive", reason="Resolved and carried into patterns.",
                                expected_revision=read_knowledge_note(archived).revision).ok


def test_the_outline_keeps_every_active_note_and_the_summaries_of_people(tmp_path):
    _shelf(tmp_path)
    whole, outline = knowledge_index_views(resolve_knowledge_address(tmp_path, "overview", "global"))
    for line in ("- release-queue-20261016 — Release queue", "- patterns", "- owner-model",
                 "- people/ada — Ada", "- recipes/cache — Cache warm-up"):
        assert line in outline
    # People's understanding keeps its authored summary; everything else is one read away.
    assert "Ada reviews every protocol change on Fridays." in outline
    assert "The owner wants one coherent design, not patches." in outline
    assert "The queue stalls" not in outline and "Warm the cache" not in outline
    assert "The queue stalls" in whole and "Warm the cache" in whole  # the whole view keeps every summary
    # An archived note leaves both views' rows but is counted with its reader.
    assert "old-incident" not in outline and "Archived notes: 1." in outline
    assert "Outline: every active note (5)" in outline
    # Grouped by the shelf directory the note was written into.
    assert outline.index("people/\n- people/ada") < outline.index("recipes/\n- recipes/cache")
    assert len(outline) < len(whole)


def test_a_group_or_a_topic_lists_with_its_summaries(tmp_path):
    _shelf(tmp_path)
    address = resolve_knowledge_address(tmp_path, "overview", "global")
    people = knowledge_index_view(address, topics=("people/",))
    assert "Ada reviews every protocol change on Fridays." in people and "release-queue" not in people
    one = knowledge_index_view(address, topics=("release-queue-20261016",))
    assert "The queue stalls when two releases race." in one and "people/ada" not in one
    assert "patterns" in knowledge_index_view(address)  # no filter: the whole active index


@pytest.mark.parametrize("kind,kept", [("person", True), ("Person", True), (" PERSON ", True),
                                        ("\tPeRsOn\t", True), ("personal", False), ("note", False)])
def test_person_summary_survives_case_and_whitespace_in_authored_yaml(tmp_path, kind, kept):
    import json

    _note(tmp_path, "entry", summary="The authored understanding of this person.", kind=json.dumps(kind))
    whole, outline = knowledge_index_views(resolve_knowledge_address(tmp_path, "overview", "global"))
    assert "The authored understanding of this person." in whole
    assert ("The authored understanding of this person." in outline) == kept
    assert "- entry" in outline
