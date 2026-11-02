"""The supported memory read (``memory_reader``, ``PluginAPI.read_memory`` 2.1) over memory its real producers wrote.

Every source here is written by its own producer into a temporary data root: chronicle pages, parts, notes,
corrections, decisions, accounts, selections and marks through ``ChronicleStore``; knowledge through
``write_knowledge_note``; identity and WORLD through ``publish_memory_file``; the scratchpad through
``Memory.append_scratchpad_block``; a Project journal through ``append_journal_milestone``. No live memory.
"""
from __future__ import annotations

import hashlib
import pathlib

import pytest

from ouroboros import config
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.contracts.plugin_api import MemoryReadError
from ouroboros.memory_reader import read_memory

MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
LIGHT = {"kind": "helper", "route": "configured-light"}
CHILD = {"kind": "helper", "task_id": "kid1", "focus": {"role": "child", "task_id": "kid1"}}
PROJECT = "atlas-proj"
NESTED = "проекты/память/атлас ✦"


def covers(*refs, at=0):
    return {"mode": "range", "rows": list(refs), "stream_span": [at, at + max(len(refs) - 1, 0)]}


def ok(result):
    assert result.ok, result
    return result.record


def page(store, *refs, room="1", at=0, author=MIND, text=None, **metadata):
    return ok(store.publish_page(room_id=room, text=text or f"page over {refs}", covers=covers(*refs, at=at),
                                 author=author, metadata=metadata or None))


def catalog(root, area, **params):
    return read_memory(root, "catalog", {"area": area, **params})


def every(root, area, **params):
    """Every entry of an area, following ``next`` until ``complete``."""
    entries, cursor, pages = [], None, 0
    while True:
        answer = catalog(root, area, **params, **({"cursor": cursor} if cursor else {}))
        entries += answer["entries"]
        pages += 1
        if answer["complete"]:
            assert answer["next"] is None
            return entries, pages
        cursor = answer["next"]


def document(root, address, **params):
    return read_memory(root, "document", {"address": address, **params})


def whole(root, address, length, **params):
    """A text read whole through ranges, every range bound to the revision the first one named."""
    first = document(root, address, length=length, **params)
    parts, answer = [first["text"]], first
    while not answer["complete"]:
        answer = document(root, address, start=answer["next"]["start"], length=length,
                          revision=answer["next"]["revision"])
        parts.append(answer["text"])
    return "".join(parts), first


@pytest.fixture
def memory(tmp_path, monkeypatch):
    """A synthetic installation: two rooms of chronicle, knowledge on two shelves, plain sources, journals."""
    root = tmp_path / "data"
    root.mkdir()
    store = ChronicleStore(root)
    p1 = page(store, "a", "b", at=0, text="Первая страница: what we decided.")
    d1 = page(store, "c", at=10, author=LIGHT, text="Light's draft that still acts.")
    d2 = page(store, "d", at=20, author=CHILD, text="A child's draft, accepted.")
    d3 = page(store, "e", at=30, author=CHILD, text="A child's draft, rejected.")
    c1 = page(store, "f", at=40, author=LIGHT, text="A candidate awaiting a decision.", candidate=True)
    ok(store.decide(d2["id"], True, MIND, "it holds"))
    ok(store.decide(d3["id"], False, MIND, "wrong room"))
    q1 = page(store, "x", room="2", text="Room two's page.")
    account = ok(store.publish_account(room_id="1", text="Across both rooms.", sources=[p1["id"], q1["id"]], author=MIND))
    fix1 = ok(store.correct(p1["id"], "Correction one.", MIND))
    fix2 = ok(store.correct(p1["id"], "Correction two.", MIND, expected_revision=fix1["id"]))
    part = ok(store.publish_part(room_id="1", text="A part over the first two.", member_ids=[p1["id"], d1["id"]],
                                 author=MIND, expected_sequence=store.room_head("1")))
    draft_part = ok(store.publish_part(room_id="1", text="Light folds the accepted draft.", member_ids=[d2["id"]],
                                       author=LIGHT, expected_sequence=store.room_head("1")))
    ok(store.decide(d1["id"], True, MIND, "accepted while folded"))
    withdrawn = ok(store.publish_account(room_id="2", text="Withdrawn telling.", sources=[q1["id"]], author=MIND))
    ok(store.select_account(account["id"], replaces=[q1["id"]], author=MIND, reason="tell room two through it"))
    ok(store.select_account(withdrawn["id"], replaces=[q1["id"]], author=MIND, reason="shown", shown=True))
    ok(store.select_account(withdrawn["id"], replaces=[], author=MIND, reason="not now", shown=False))
    kept = ok(store.mark({"kind": "chronicle", "id": p1["id"]}, "Keep this in view.", MIND, room_id="1"))
    viewed = ok(store.mark({"kind": "chronicle", "id": q1["id"]}, "Meaning only.", MIND, room_id="2", scope="global"))
    ok(store.set_mark_view(viewed["id"], "meaning", MIND, "enough"))
    gone = ok(store.mark({"kind": "chronicle", "id": d1["id"]}, "Released later.", MIND, room_id="1"))
    ok(store.release_mark(gone["id"], MIND, "done with it"))
    note = ok(store.write_note(room_id="1", task_id="t1", text="A note for later.", author=MIND))

    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    def knowledge(topic, content, scope="global", **kwargs):
        result = write_knowledge_note(resolve_knowledge_address(root, topic, scope), content, writer="turn", **kwargs)
        assert result.ok, result
        return result.current

    knowledge("overview", "---\nsummary: The shelf in one line.\n---\n# Overview\n")
    first = knowledge(NESTED, "---\ntitle: Атлас\nsummary: |\n  Первая строка.\n  Second line, whole.\n---\n"
                              "# Атлас\n\nSee [the overview](../../overview.md), [missing](nowhere.md) "
                              "and [web](https://example.org).\n")
    knowledge(NESTED, "# Атлас\n\nRewritten body, version two.\n", expected_revision=first.revision)
    old = knowledge("old-topic", "---\nsummary: Retired.\n---\n# Old\n")
    knowledge("old-topic", "", mode="archive", expected_revision=old.revision, reason="superseded")
    knowledge("заметки/первая", "# Project note\n", scope=f"project:{PROJECT}")

    from ouroboros.memory import Memory
    from ouroboros.memory_publication import publish_memory_file

    mem = Memory(root)
    author = {"task_id": "t1", "focus": {"role": "root"}, "route": "test"}
    publish_memory_file(mem.identity_path(), None, lambda _old: "# Identity\n\nI am.\n",
                        journal=mem.identity_journal_path(), author=author, source_type="task")
    mem.append_scratchpad_block("A working thought " + "w" * 400)
    monkeypatch.setattr(config, "DATA_DIR", root)
    from ouroboros.tools.project_journal import append_journal_milestone

    append_journal_milestone(PROJECT, "start", "Начали атлас.", task_id="t1")
    (root / "projects" / PROJECT / "workpad.md").write_text("# Workpad\n", encoding="utf-8")
    ids = dict(p1=p1, d1=d1, d2=d2, d3=d3, c1=c1, q1=q1, account=account, withdrawn=withdrawn, fix1=fix1, fix2=fix2,
               part=part, draft_part=draft_part, kept=kept, viewed=viewed, gone=gone, note=note)
    return root, store, {name: record["id"] for name, record in ids.items()}


def by_id(entries):
    return {entry["id"]: entry for entry in entries}


# --- the chronicle ------------------------------------------------------------------------------------

def test_every_draft_state_of_pages_and_parts_is_listed_with_its_status(memory):
    root, _store, ids = memory
    entries, _pages = every(root, "records", room_id="1", kinds=["page", "part"], limit=2)
    status = {entry["id"]: entry["status"] for entry in entries}
    assert status == {ids["p1"]: "final", ids["d1"]: "accepted", ids["d2"]: "accepted", ids["d3"]: "rejected",
                      ids["c1"]: "candidate", ids["part"]: "final", ids["draft_part"]: "draft"}
    helpers, _ = every(root, "records", author_kind="helper", status=["draft", "candidate"], limit=3)
    assert {entry["id"] for entry in helpers} == {ids["c1"], ids["draft_part"]}
    signers = by_id(entries)
    assert signers[ids["d2"]]["signer"] == "child, task kid1" and signers[ids["c1"]]["signer"] == "Light"
    assert signers[ids["part"]]["covers"]["member_ids"] == [ids["p1"], ids["d1"]]  # a short list stays inline
    assert signers[ids["p1"]]["folded_into"] == ids["part"] and signers[ids["p1"]]["corrections"] == 2
    assert signers[ids["p1"]]["preview"] == "Первая страница: what we decided."  # its own words, corrections apart


def test_a_record_reads_whole_at_every_revision_and_names_the_later_ones(memory):
    root, store, ids = memory
    record = {"kind": "record", "id": ids["p1"]}
    now = document(root, record)
    assert now["revision"] == now["current_revision"] == ids["fix2"]
    assert now["text"].startswith("Первая страница") and "Correction one." in now["text"] and "Correction two." in now["text"]
    assert now["later_corrections"] == {"count": 0, "next": None} and now["corrections_shown"] == 2 and now["told_by"] is None
    cited = document(root, record, revision=ids["p1"])
    assert cited["text"] == "Первая страница: what we decided." and cited["later_corrections"]["count"] == 2
    later = catalog(root, **cited["later_corrections"]["next"])
    assert [entry["id"] for entry in later["entries"]] == [ids["fix1"], ids["fix2"]] and later["complete"]
    assert cited["current_revision"] == ids["fix2"]
    middle = document(root, record, revision=ids["fix1"])
    assert "Correction one." in middle["text"] and "Correction two." not in middle["text"]
    # The agent's memory_read composes the same words through the same store rule.
    assert middle["text"] == store.interpret([store.get(ids["p1"])], revision=ids["fix1"])[0]["current_text"]
    with pytest.raises(MemoryReadError) as refused:
        document(root, record, revision=ids["q1"])
    assert refused.value.code == "not_found"
    assert document(root, {"kind": "record", "id": ids["gone"]})["text"] == "Released later."
    decision = by_id(every(root, "history", address=record)[0])
    assert set(decision) == {ids["fix1"], ids["fix2"]}
    assert [entry["kind"] for entry in every(root, "history", address={"kind": "record", "id": ids["d3"]})[0]] == ["decision"]


def test_an_account_keeps_its_frozen_source_versions_apart_from_their_state_now(memory):
    root, _store, ids = memory
    sources, _ = every(root, "relations", address={"kind": "record", "id": ids["account"]}, relation="sources", limit=1)
    frozen = by_id(sources)[ids["p1"]]
    assert frozen["revision"] == ids["p1"] and frozen["revision_known"] is True
    assert frozen["later_corrections"]["count"] == 2 and frozen["current_revision"] == ids["fix2"]
    assert [e["id"] for e in catalog(root, **frozen["later_corrections"]["next"])["entries"]] == [ids["fix1"], ids["fix2"]]
    assert frozen["folded_into"] == ids["part"] and frozen["missing"] is False
    assert by_id(sources)[ids["q1"]]["later_corrections"] == {"count": 0, "next": None}
    cited, _ = every(root, "relations", address={"kind": "record", "id": ids["q1"]}, relation="cited_by", limit=1)
    assert [(row["account"]["id"], row["cited"]["revision"]) for row in cited] == [
        (ids["account"], ids["q1"]), (ids["withdrawn"], ids["q1"])]
    account = document(root, {"kind": "record", "id": ids["account"]})
    assert account["record"]["selection"]["shown"] is True and account["stored"]["sources"][0]["id"] == ids["p1"]


def test_a_part_lists_its_members_without_inventing_their_versions(memory):
    root, _store, ids = memory
    answer = catalog(root, "relations", address={"kind": "record", "id": ids["part"]}, relation="members", limit=1)
    assert answer["member_versions"] == "not_recorded" and answer["count"] == 2 and answer["next"] == "i1"
    members, _ = every(root, "relations", address={"kind": "record", "id": ids["part"]}, relation="members", limit=1)
    assert [row["id"] for row in members] == [ids["p1"], ids["d1"]]
    assert all("revision" not in {k for k in row if k != "member"} for row in members)
    with pytest.raises(MemoryReadError):
        catalog(root, "relations", address={"kind": "record", "id": ids["p1"]}, relation="members")


def test_a_record_carries_its_story_period_and_a_retold_one_its_marked_block_period(tmp_path, monkeypatch):
    """A page, part, account or retold record lists and reads with the ``period``, ``source_period`` and
    ``period_basis`` of its story entry (``memory_inventory.record_period``, printed by the view). No legacy unit is
    computed: a retold record names its block's period, marked, which only the story narrows to its room's rows."""
    from ouroboros import chat_chain, memory_inventory
    from ouroboros.tools.chronicle import page_covers
    from tests import _memory_inventory_shared as shared

    rooms = shared.world(tmp_path)  # the legacy import of two blocks over real rooms
    store = ChronicleStore(tmp_path)
    at = {pos: address for address, _row, pos in chat_chain.iter_rows(tmp_path)}

    def sealed(room, first, last):
        covers = page_covers(tmp_path, room, from_addr=at[first], to_addr=at[last])["covers"]
        return ok(store.publish_page(room_id=room, text=f"Rows {first}-{last}.", covers=covers, author=MIND))["id"]

    early, late, alpha = sealed("1", 10, 12), sealed("1", 19, 19), sealed(str(rooms["alpha"]), 13, 13)
    part = ok(store.publish_part(room_id="1", text="A part.", member_ids=[early, late], author=MIND,
                                 expected_sequence=store.room_head("1")))["id"]
    account = ok(store.publish_account(room_id="1", text="Across.", sources=[alpha, "legacy-b00-r777"], author=MIND))["id"]
    ok(store.select_account(account, replaces=[], author=MIND, reason="show it"))
    undated, fix = page(store, "z")["id"], ok(store.correct(alpha, "A correction.", MIND))["id"]
    keys = ("period", "source_period", "period_basis")
    story = {entry["id"]: {key: entry[key] for key in keys} for entry in every(tmp_path, "story")[0]}
    monkeypatch.setattr(memory_inventory, "legacy_units", lambda *_a, **_k: pytest.fail("the legacy history was read"))
    listed = by_id(every(tmp_path, "records", limit=7)[0])
    dated = {ident: {key: entry[key] for key in keys} for ident, entry in listed.items() if "period" in entry}
    assert {listed[ident]["kind"] for ident in dated} == {"page", "part", "account", "legacy"}
    assert "period" not in listed[fix] and not set(keys) & set(document(tmp_path, {"kind": "record", "id": fix}))
    for ident, fields in dated.items():  # a record reads as it lists, the period beside its room label and teller
        answer = document(tmp_path, {"kind": "record", "id": ident}, length=1)
        assert {key: answer[key] for key in keys} == fields and not set(keys) & set(answer["record"])
    assert dated[part] == story[part] == {"period": "2026-09-03 00:00 → 2026-09-03 00:09", "period_basis": "rows",
                                         "source_period": {"start": "2026-09-03T00:00:00+00:00",
                                                           "end": "2026-09-03T00:09:00+00:00", "incomplete": False}}
    assert dated[undated] == story[undated] == {"period": "period unknown", "source_period": None, "period_basis": "rows"}
    block = {"period": "2026-09-01 00:00 → 2026-09-01 00:05 (block period)", "period_basis": "block",
             "source_period": {"start": "2026-09-01T00:00:00+00:00", "end": "2026-09-01T00:05:00+00:00", "incomplete": False}}
    assert dated["legacy-b00-r777"] == block and story["legacy-b00-r777"]["period"] == "2026-09-01 00:04 → 2026-09-01 00:04"
    assert dated[account]["period"] == "2026-09-01 00:00 → 2026-09-03 00:03 (partly block period)"
    assert (dated[account]["period_basis"], story[account]["period_basis"]) == ("mixed", "rows")
    members = every(tmp_path, "relations", address={"kind": "record", "id": part}, relation="members")[0]
    cited = every(tmp_path, "relations", address={"kind": "record", "id": alpha}, relation="cited_by")[0]
    assert [{key: row["member"][key] for key in keys} for row in members] == [dated[early], dated[late]]
    assert [{key: row["account"][key] for key in keys} for row in cited] == [dated[account]]


def test_the_story_is_the_views_own_story_with_selected_accounts_and_what_they_tell(memory):
    root, store, ids = memory
    assert catalog(root, "story")["state"] == "not_activated"  # as the view: no story before the one legacy import
    store.ensure_activated()
    before = _sources(root)
    story, _ = every(root, "story", limit=2)
    assert _sources(root) == before  # one story capture writes no memory source
    from ouroboros.chronicle_store import capture
    from ouroboros.memory_view import capture_story, room_labeler

    with capture(root, tail="terminal"):
        view = capture_story(store, root, room_labeler(root))[0]
    assert [(e["id"], e["kind"], e["told_by"]) for e in story] == [(e["id"], e["kind"], e.get("told_by")) for e in view]
    shown = by_id(story)
    assert shown[ids["q1"]]["told_by"] == ids["account"] and shown[ids["account"]]["told_by"] is None
    assert ids["withdrawn"] not in shown  # its latest selection withdrew it
    assert ids["d3"] not in shown and ids["c1"] not in shown  # a rejected draft and a candidate do not act
    assert ids["p1"] not in shown and ids["d2"] not in shown  # folded into acting parts
    assert ids["part"] in shown and ids["draft_part"] in shown
    assert catalog(root, "story")["order"] == "story"


def test_marks_are_active_or_released_with_the_views_decided_since(memory):
    root, _store, ids = memory
    active = by_id(every(root, "marks", state="active")[0])
    assert set(active) == {ids["kept"], ids["viewed"]}
    assert active[ids["viewed"]]["visibility"] == "meaning" and active[ids["viewed"]]["scope"] == "global"
    released = by_id(every(root, "marks", state="released")[0])
    assert set(released) == {ids["gone"]} and released[ids["gone"]]["released"]["preview"] == "done with it"


def test_rooms_are_labelled_and_an_unactivated_chronicle_says_so(memory):
    root, _store, _ids = memory
    answer = catalog(root, "rooms")
    assert answer["chronicle"]["state"] == "present" and answer["chronicle"]["activation"] is None
    rooms = {entry["room_id"]: entry for entry in answer["entries"]}
    assert rooms["1"]["label"] == "Main" and rooms["2"]["label"] == "Unknown room [chat_id=2]"
    assert rooms["1"]["head"] > 0 and rooms["1"]["records"] > rooms["2"]["records"]


def test_an_absent_chronicle_is_absent_and_reading_creates_nothing(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    assert catalog(root, "rooms")["chronicle"]["state"] == "absent"
    assert catalog(root, "story")["entries"] == [] and read_memory(root, "changes")["since_known"] is False
    with pytest.raises(MemoryReadError) as missing:
        document(root, {"kind": "file", "source": "identity"})
    assert missing.value.code == "not_found"
    assert not any(root.rglob("*")), "a read created files"


# --- knowledge ----------------------------------------------------------------------------------------

def test_a_shelf_pages_nested_unicode_and_archived_notes_with_their_whole_summary(memory):
    root, _store, _ids = memory
    notes, pages = every(root, "shelf", scope="global", view="all", limit=1)
    assert pages >= 3 and len({note["topic"] for note in notes}) == len(notes) == 3
    nested = {note["topic"]: note for note in notes}[NESTED]
    # An overwrite keeps the authored preamble: the multi-line summary comes whole, never shortened.
    assert nested["title"] == "Атлас" and nested["summary"] == "Первая строка.\nSecond line, whole.\n"
    archived = {note["topic"]: note for note in notes}["old-topic"]
    assert archived["state"] == "archived" and archived["archive"]["reason"] == "superseded"
    assert archived["summary"] == "Retired."
    active, _ = every(root, "shelf", scope="global", limit=1)
    assert {note["topic"] for note in active} == {"overview", NESTED}
    project, _ = every(root, "shelf", scope=f"project:{PROJECT}")
    assert [note["topic"] for note in project] == ["заметки/первая"]
    shelves, _ = every(root, "shelves")
    assert [shelf["scope"] for shelf in shelves] == ["global", f"project:{PROJECT}"]


class _WindowsOrderedPath(type(pathlib.Path())):
    """A local path that sorts as ``WindowsPath`` does: by its lower-cased parts."""

    def __lt__(self, other):
        return [part.lower() for part in self.parts] < [part.lower() for part in other.parts]


def test_a_shelf_pages_every_note_once_where_paths_sort_case_insensitively(memory, monkeypatch):
    """A page ends at a topic and the next one starts after it in the order the cursor compares, never in the
    platform's path order: after ``mixed/alpha`` neither ``mixed/Beta`` nor ``mixed/Zeta`` counts as listed."""
    from dataclasses import replace

    from ouroboros import memory_reader
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    root, _store, _ids = memory
    for topic in ("alpha", "Beta", "gamma", "Zeta"):
        assert write_knowledge_note(resolve_knowledge_address(root, f"mixed/{topic}", "global"), f"# {topic}\n").ok
    resolve = memory_reader._note_address

    def windows_ordered(*args):
        address = resolve(*args)
        return replace(address, shelf=_WindowsOrderedPath(address.shelf))

    monkeypatch.setattr(memory_reader, "_note_address", windows_ordered)
    notes, pages = every(root, "shelf", scope="global", view="all", limit=1)
    topics = [note["topic"] for note in notes]
    assert {"mixed/alpha", "mixed/Beta", "mixed/gamma", "mixed/Zeta"} <= set(topics) and pages == len(topics) == 7
    assert topics == sorted(topics, key=lambda topic: f"{topic}.md".split("/"))  # each once, in the compared order


def test_a_note_keeps_its_source_capture_history_and_each_version_reads_whole(memory):
    root, _store, _ids = memory
    address = {"kind": "note", "scope": "global", "topic": NESTED}
    history, _ = every(root, "history", address=address, limit=1)
    assert [row["fields"]["publication"] for row in history] == ["source_capture", "source_capture"]
    assert history[0]["fields"]["old_sha256"] == "" and history[1]["fields"]["old_sha256"] == history[0]["fields"]["new_sha256"]
    first = document(root, {**history[0]["address"], "field": ["new_content"]})
    assert first["text"].startswith("---\ntitle: Атлас") and "[the overview](../../overview.md)" in first["text"]
    assert first["revision"] == hashlib.sha256(first["text"].encode("utf-8")).hexdigest()
    current = document(root, address)
    assert current["text"].endswith("\n---\n# Атлас\n\nRewritten body, version two.\n") and current["note"]["title"] == "Атлас"
    assert current["revision"] == history[1]["fields"]["new_sha256"]
    assert every(root, "relations", address=address, relation="links")[0] == []  # version two links nothing


def test_note_links_resolve_present_unwritten_and_external(memory):
    root, _store, _ids = memory
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    address = resolve_knowledge_address(root, "linked", "global")
    assert write_knowledge_note(address, "See [o](overview.md), [m](missing.md), [w](https://example.org).\n").ok
    links, _ = every(root, "relations", address={"kind": "note", "scope": "global", "topic": "linked"},
                     relation="links", limit=2)
    assert [(link["status"], (link["address"] or {}).get("topic")) for link in links] == [
        ("present", "overview"), ("unwritten", "missing"), ("external", None)]


def test_a_links_page_of_a_changed_note_is_refused_not_mixed(memory):
    root, _store, _ids = memory
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    target = resolve_knowledge_address(root, "linked", "global")
    address = {"kind": "note", "scope": "global", "topic": "linked"}
    assert write_knowledge_note(target, "[a](a.md) [b](b.md) [c](c.md)\n").ok
    first = catalog(root, "relations", address=address, relation="links", limit=1)
    held = catalog(root, "relations", address=address, relation="links", limit=1, cursor=first["next"])
    assert [link["target"] for link in first["entries"] + held["entries"]] == ["a.md", "b.md"]  # unchanged: it holds
    assert write_knowledge_note(target, "[new](new.md) [a](a.md) [b](b.md) [c](c.md)\n",
                                expected_revision=first["revision"]).ok  # a link lands before the page already read
    with pytest.raises(MemoryReadError) as stale:
        catalog(root, "relations", address=address, relation="links", limit=1, cursor=first["next"])
    assert stale.value.code == "revision_mismatch" and stale.value.current_revision not in (None, first["revision"])
    links, _ = every(root, "relations", address=address, relation="links", limit=1)
    assert [link["target"] for link in links] == ["new.md", "a.md", "b.md", "c.md"]


def test_a_continued_range_of_a_changed_note_is_refused_not_mixed(memory):
    root, _store, _ids = memory
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    address = {"kind": "note", "scope": "global", "topic": "overview"}
    first = document(root, address, length=5)
    assert first["next"] == {"start": 5, "revision": first["revision"]}
    target = resolve_knowledge_address(root, "overview", "global")
    assert write_knowledge_note(target, "# Overview\n\nChanged.\n", expected_revision=first["revision"]).ok
    with pytest.raises(MemoryReadError) as stale:
        document(root, address, start=5, revision=first["next"]["revision"])
    assert stale.value.code == "revision_mismatch" and stale.value.current_revision != first["revision"]


# --- plain sources and journals -------------------------------------------------------------------------

def test_plain_sources_and_their_journals_are_read_without_creating_absent_ones(memory):
    root, _store, _ids = memory
    documents, _ = every(root, "documents", limit=2)
    keyed = {entry["key"]: entry for entry in documents}
    assert keyed["0:identity"]["exists"] and not keyed["0:world"]["exists"]
    assert keyed[f"1:{PROJECT}:journal"]["exists"] and keyed[f"1:{PROJECT}:workpad"]["exists"]
    assert document(root, {"kind": "file", "source": "identity"})["text"] == "# Identity\n\nI am.\n"
    journal, _ = every(root, "history", address={"kind": "file", "source": "identity"})
    assert journal[0]["fields"]["new_content"] == "# Identity\n\nI am.\n"
    scratch, _ = every(root, "history", address={"kind": "file", "source": "scratchpad"})
    long_block = scratch[-1]["fields"]["block"]["content"]
    assert long_block["ref"] == "text" and long_block["chars"] > 400
    assert document(root, long_block["address"])["text"].startswith("A working thought www")
    milestones, _ = every(root, "history", address={"kind": "journal", "journal": f"project:{PROJECT}"})
    assert milestones[0]["fields"]["text"] == "Начали атлас."
    assert document(root, {"kind": "file", "source": f"workpad:{PROJECT}"})["text"] == "# Workpad\n"
    with pytest.raises(MemoryReadError):
        document(root, {"kind": "file", "source": "world"})
    assert not (root / "memory" / "WORLD.md").exists()


def test_documents_page_in_key_order_when_one_project_id_prefixes_another(memory):
    """``atlas`` prefixes ``atlas-2`` and ``atlas-proj``, whose keys sort first (``-`` before ``:``): at every page
    size each document is reached once, in key order."""
    root, _store, _ids = memory
    for pid in ("atlas", "atlas-2"):
        (root / "projects" / pid).mkdir()
    expected = ["0:identity", "0:scratchpad", "0:world",
                *sorted(f"1:{pid}:{part}" for pid in ("atlas", "atlas-2", PROJECT) for part in ("journal", "workpad"))]
    for limit in range(1, len(expected) + 1):
        assert [entry["key"] for entry in every(root, "documents", limit=limit)[0]] == expected, limit


def test_an_unterminated_journal_line_waits_and_is_never_skipped(memory):
    root, _store, _ids = memory
    path = root / "projects" / PROJECT / "journal.jsonl"
    with path.open("ab") as stream:
        stream.write(b'{"ts": "x", "kind": "note", "text": "half')
    answer = catalog(root, "history", address={"kind": "journal", "journal": f"project:{PROJECT}"})
    assert answer["complete"] and answer["unterminated_tail_bytes"] > 0 and len(answer["entries"]) == 1
    with path.open("ab") as stream:
        stream.write(b' done"}\n')
    later = catalog(root, "history", address={"kind": "journal", "journal": f"project:{PROJECT}"},
                    cursor=answer["position"])
    assert [row["fields"]["text"] for row in later["entries"]] == ["half done"]


# --- size and growth ------------------------------------------------------------------------------------

def test_a_text_over_one_mebibyte_is_reached_exactly_through_ranges(memory):
    root, store, _ids = memory
    words = ("Строка атласа ✦ " * 80_000) + "конец"
    big = page(store, "big", room="3", text=words)
    assert len(words.encode("utf-8")) > 1_048_576
    text, first = whole(root, {"kind": "record", "id": big["id"]}, length=300_000)
    assert text == words and first["range"]["total"] == len(words) and not first["complete"]
    entry = by_id(every(root, "records", room_id="3")[0])[big["id"]]
    assert entry["chars"] == len(words) and len(entry["preview"]) < 300


def test_changes_page_through_a_large_unrelated_history_without_loss(memory):
    root, store, ids = memory
    baseline = read_memory(root, "changes")
    assert baseline["since_known"] is False and baseline["entries"] == []
    for batch in range(3):
        assert store.publish([{"kind": "note", "room_id": "9", "task_id": f"t{batch}", "author": MIND,
                               "text": f"unrelated {batch}-{n}", "id": f"u{batch}-{n}"} for n in range(500)]).ok
    fix = ok(store.correct(ids["q1"], "Room two, corrected.", MIND))
    seen, position, rounds = [], baseline["position"], 0
    while True:
        answer = read_memory(root, "changes", {"position": position, "limit": 97})
        assert answer["since_known"] is True
        seen += [entry["id"] for entry in answer["entries"]]
        assert answer["position"] == ({"sequence": answer["entries"][-1]["sequence"], "record_id": answer["entries"][-1]["id"]}
                                      if answer["entries"] else position)
        position, rounds = answer["position"], rounds + 1
        if answer["complete"]:
            break
    assert len(seen) == len(set(seen)) == 1501 and seen[-1] == fix["id"] and rounds == 16
    assert position == answer["newest"]
    with pytest.raises(MemoryReadError) as moved:
        read_memory(root, "changes", {"position": {"sequence": position["sequence"], "record_id": "not-that-one"}})
    assert moved.value.code == "position_mismatch"
    # A room's catalog over that history examines only its own page.
    room_two, pages = every(root, "records", room_id="2", limit=10)
    assert pages == 1 and ids["q1"] in by_id(room_two)


def test_a_catalog_page_bounds_the_index_statement_itself(memory, monkeypatch):
    root, store, _ids = memory
    from ouroboros import chronicle_store

    for n in range(60):  # more history than any page below
        assert store.write_note(room_id="1", task_id="t1", text=f"note {n}", author=MIND).ok
    fetched = []
    original = chronicle_store._IndexLookups.rows

    def spy(self, *args, **kwargs):
        rows = original(self, *args, **kwargs)
        fetched.append((args[3] if len(args) > 3 else kwargs.get("limit", -1), len(rows)))
        return rows

    monkeypatch.setattr(chronicle_store._IndexLookups, "rows", spy)
    catalog(root, "records", limit=3)
    catalog(root, "marks", limit=5)  # (``story`` is one view capture per read, sliced: its cost is the view's)
    paged = [pair for pair in fetched if pair[0] in (4, 6)]  # each page asks one row past itself
    assert [limit for limit, _rows in paged] == [4, 6] and all(rows <= limit for limit, rows in paged)
    assert paged[0][1] == 4  # the statement itself stopped there: the room holds far more


# --- read-only ------------------------------------------------------------------------------------------

def _sources(root: pathlib.Path) -> dict:
    derived = {"index.sqlite3", "index.sqlite3-wal", "index.sqlite3-shm", ".publication.lock"}
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name not in derived}


def test_reading_every_area_writes_no_memory_source(memory):
    root, store, ids = memory
    (root / "memory" / "chronicle" / "index.sqlite3").unlink()  # a cold index catches up: derived, allowed
    before = _sources(root)
    for area, params in (("rooms", {}), ("records", {}), ("story", {}), ("marks", {"state": "all"}), ("shelves", {}),
                         ("shelf", {"scope": "global", "view": "all"}), ("documents", {}),
                         ("history", {"address": {"kind": "record", "id": ids["p1"]}}),
                         ("history", {"address": {"kind": "note", "scope": "global", "topic": NESTED}}),
                         ("relations", {"address": {"kind": "record", "id": ids["account"]}, "relation": "sources"})):
        every(root, area, **params)
    document(root, {"kind": "record", "id": ids["account"]})
    document(root, {"kind": "note", "scope": "global", "topic": NESTED})
    read_memory(root, "changes", {"position": {"sequence": 0, "record_id": None}})
    assert _sources(root) == before
    assert store.activation() is None  # never activated or imported by a read


def test_refusals_are_typed_and_unknown_parameters_are_not_ignored(memory):
    root, _store, _ids = memory
    for operation, params, code in (
            ("search", {}, "invalid_parameter"),
            ("catalog", {"area": "records", "lmit": 3}, "invalid_parameter"),
            ("catalog", {"area": "records", "cursor": "t-not-mine"}, "invalid_parameter"),
            ("catalog", {"area": "shelf", "scope": "project:../x"}, "invalid_parameter"),
            ("document", {"address": {"kind": "file", "source": "../../etc/passwd"}}, "invalid_parameter"),
            ("document", {"address": {"kind": "record", "id": "nope"}}, "not_found"),
            ("document", {"address": {"kind": "journal", "journal": "identity"}}, "invalid_parameter"),
            ("changes", {"position": {"sequence": True}}, "invalid_parameter")):
        with pytest.raises(MemoryReadError) as refused:
            read_memory(root, operation, params)
        assert refused.value.code == code, (operation, params)
