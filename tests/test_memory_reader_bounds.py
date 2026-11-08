"""Bounded answers of ``memory_reader`` over growing records: every list, history and text of one record stays
reachable, by a continuation or a ranged field read, while one answer carries a bounded amount of it.

The records are written by ``ChronicleStore`` (one transaction for a bulk of them, as the importer does); the
told-by precedence is checked against ``memory_view_account.told_map`` over every acting selection.
"""
from __future__ import annotations

import json

import pytest

from ouroboros import memory_reader
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.contracts.plugin_api import MemoryReadError
from ouroboros.memory_reader import read_memory

MIND = {"kind": "mind", "task_id": "t1", "focus": {"role": "root", "task_id": "t1"}}
SMALL = 48 * 1024  # what one answer carries beside its asked-for text range or its entries
PER_ENTRY = 4 * 1024  # one list entry, projected


def size(answer) -> int:
    return len(json.dumps(answer, ensure_ascii=False))


def beside_text(answer) -> int:
    """The answer's size without the text range it was asked for: metadata only."""
    return size(answer) - len(json.dumps(answer.get("text", ""), ensure_ascii=False))


def catalog(root, area, **params):
    return read_memory(root, "catalog", {"area": area, **params})


def document(root, address, **params):
    return read_memory(root, "document", {"address": address, **params})


def every(root, area, **params):
    entries, cursor = [], params.pop("cursor", None)
    while True:
        answer = catalog(root, area, **params, **({"cursor": cursor} if cursor else {}))
        assert size(answer) < SMALL + PER_ENTRY * len(answer["entries"])
        entries += answer["entries"]
        if answer["complete"]:
            return entries
        cursor = answer["next"]


def follow(root, continuation):
    """Every entry a continuation (``{area, ...}``) leads to."""
    params = dict(continuation)
    return every(root, params.pop("area"), **params)


def whole(root, address, length=4096):
    first = document(root, address, length=length)
    assert beside_text(first) < SMALL
    parts, answer = [first["text"]], first
    while not answer["complete"]:
        answer = document(root, address, start=answer["next"]["start"], length=length, revision=answer["next"]["revision"])
        assert beside_text(answer) < SMALL
        parts.append(answer["text"])
    return "".join(parts)


def ok(result):
    assert result.ok, result
    return result.record


def notes(store, count, room="5"):
    ids = [f"n{room}-{n}" for n in range(count)]
    ok(store.publish([{"id": ident, "kind": "note", "room_id": room, "task_id": "t1", "author": MIND,
                       "text": f"note {ident}"} for ident in ids]))
    return ids


def pages(store, count, room="6", rows=1):
    ids = [f"p{room}-{n}" for n in range(count)]
    ok(store.publish([{"id": ident, "kind": "page", "room_id": room, "text": f"page {ident}", "author": MIND,
                       "metadata": {}, "covers": {"room_id": room, "mode": "range", "stream_span": [n, n],
                                                  "rows": [f"{ident}-row{r}" for r in range(rows)]}}
                      for n, ident in enumerate(ids)]))
    return ids


@pytest.fixture
def root(tmp_path):
    path = tmp_path / "data"
    path.mkdir()
    return path


def test_a_record_with_thousands_of_corrections_answers_a_small_range_and_pages_its_history(root):
    store = ChronicleStore(root)
    first = ok(store.write_note(room_id="1", task_id="t1", text="The original.", author=MIND))
    fixes = [f"fix-{n}" for n in range(3000)]
    ok(store.publish([{"id": ident, "kind": "correction", "room_id": "1", "target_id": first["id"], "author": MIND,
                       "text": f"correction {n}"} for n, ident in enumerate(fixes)]))
    cited = document(root, {"kind": "record", "id": first["id"]}, revision=first["id"], length=10)
    assert size(cited) < SMALL and cited["text"] == "The origin"
    assert cited["later_corrections"]["count"] == 3000 and cited["current_revision"] == fixes[-1]
    later = follow(root, cited["later_corrections"]["next"])
    assert [entry["id"] for entry in later] == fixes
    now = document(root, {"kind": "record", "id": first["id"]}, length=10)
    assert size(now) < SMALL and now["corrections_shown"] == 3000 and now["later_corrections"]["count"] == 0
    assert now["range"]["total"] > 3000 * len("correction 0")


def test_a_selection_naming_thousands_of_records_is_compact_and_read_whole_by_ranges(root):
    store = ChronicleStore(root)
    told = notes(store, 3000)
    account = ok(store.publish_account(room_id="5", text="All of them.", sources=told[:2], author=MIND))
    selection = ok(store.select_account(account["id"], replaces=told, author=MIND, reason="tell them all"))
    events = every(root, "history", address={"kind": "record", "id": account["id"]})
    replaces = events[0]["replaces"]
    assert replaces["ref"] == "list" and replaces["count"] == 3000 and len(replaces["head"]) < 10
    assert json.loads(whole(root, replaces["address"], length=20_000)) == told
    shown = document(root, {"kind": "record", "id": account["id"]}, length=5)
    assert size(shown) < SMALL and shown["record"]["selection"]["replaces"]["count"] == 3000
    assert document(root, {"kind": "record", "id": told[1500]}, length=5)["told_by"] == account["id"]
    assert size(document(root, {"kind": "record", "id": selection["id"]}, length=5)) < SMALL


def test_an_account_with_thousands_of_sources_pages_them_and_keeps_the_frozen_list_whole(root):
    store = ChronicleStore(root)
    cited = notes(store, 3000)
    account = ok(store.publish_account(room_id="5", text="Over three thousand.", sources=cited, author=MIND))
    answer = document(root, {"kind": "record", "id": account["id"]}, length=5)
    assert size(answer) < SMALL
    frozen = answer["stored"]["sources"]
    assert frozen["ref"] == "list" and frozen["count"] == 3000
    assert json.loads(whole(root, frozen["address"], length=50_000)) == store.get(account["id"])["sources"]
    sources = every(root, "relations", address={"kind": "record", "id": account["id"]}, relation="sources", limit=500)
    assert [source["id"] for source in sources] == cited and all(s["revision"] == s["id"] for s in sources)


def test_page_rows_and_part_members_by_the_thousand_are_reachable_without_invented_versions(root):
    store = ChronicleStore(root)
    sealed = pages(store, 1, room="7", rows=5000)[0]
    answer = document(root, {"kind": "record", "id": sealed}, length=5)
    rows = answer["stored"]["covers"]["rows"]
    assert size(answer) < SMALL and rows["ref"] == "list" and rows["count"] == 5000
    assert json.loads(whole(root, rows["address"], length=30_000)) == store.get(sealed)["covers"]["rows"]
    members = pages(store, 2000)
    part = ok(store.publish_part(room_id="6", text="Two thousand pages.", member_ids=members, author=MIND,
                                 expected_sequence=store.room_head("6")))
    folded = document(root, {"kind": "record", "id": part["id"]}, length=5)
    assert size(folded) < SMALL and folded["stored"]["covers"]["member_ids"]["count"] == 2000
    listed = every(root, "relations", address={"kind": "record", "id": part["id"]}, relation="members", limit=400)
    assert [row["id"] for row in listed] == members and all("revision" not in row for row in listed)


def test_the_nested_changes_of_one_source_are_counted_and_paged(root):
    store = ChronicleStore(root)
    members = pages(store, 2000)
    part = ok(store.publish_part(room_id="6", text="Folded.", member_ids=members, author=MIND,
                                 expected_sequence=store.room_head("6")))
    account = ok(store.publish_account(room_id="6", text="Told through the part.", sources=[part["id"]], author=MIND))
    ok(store.publish([{"id": f"late-{n}", "kind": "correction", "room_id": "6", "target_id": member, "author": MIND,
                       "text": f"later {n}"} for n, member in enumerate(members)]))
    source = catalog(root, "relations", address={"kind": "record", "id": account["id"]}, relation="sources")
    assert size(source) < SMALL
    changes = source["entries"][0]["nested_changes"]
    assert changes["count"] == 2000
    listed = follow(root, changes["next"])
    assert len(listed) == 2000 and {row["source_id"] for row in listed} == set(members)
    assert all(row["revision"] is None for row in listed)  # a part records no member versions


def test_a_nested_changes_page_of_a_changed_list_is_refused_not_mixed(root):
    store = ChronicleStore(root)
    members = pages(store, 3)
    part = ok(store.publish_part(room_id="6", text="Folded.", member_ids=members, author=MIND,
                                 expected_sequence=store.room_head("6")))
    account = ok(store.publish_account(room_id="6", text="Told through the part.", sources=[part["id"]], author=MIND))
    for member in members[1:]:
        ok(store.correct(member, f"later {member}", MIND))
    nested = {"area": "relations", "address": {"kind": "record", "id": account["id"]}, "relation": "source_changes",
              "source": 0, "limit": 1}
    first = read_memory(root, "catalog", nested)
    held = read_memory(root, "catalog", {**nested, "cursor": first["next"]})
    assert [row["source_id"] for row in first["entries"] + held["entries"]] == members[1:]  # unchanged: it holds
    ok(store.correct(members[0], "earlier", MIND))  # the list grows before the page already read
    with pytest.raises(MemoryReadError) as changed:
        read_memory(root, "catalog", {**nested, "cursor": first["next"]})
    assert changed.value.code == "position_mismatch"
    assert [row["source_id"] for row in follow(root, nested)] == members  # read again: each change once


def _told_everywhere(store) -> dict:
    """The oracle: ``told_map`` over every acting shown selection, as the story view computes it."""
    from ouroboros.memory_view_account import told_map

    latest = {}
    for selection in store.records(kinds=("selection",)):
        latest[selection["target_id"]] = selection
    return told_map([{"id": account, "selection_sequence": chosen["sequence"], "replaces": chosen.get("replaces") or ()}
                     for account, chosen in latest.items() if chosen.get("shown")])


def test_told_by_is_an_indexed_lookup_with_the_story_precedence(root, monkeypatch):
    store = ChronicleStore(root)
    store.ensure_activated()  # the story is read only after the one legacy import, as the view reads it
    x, y, z = notes(store, 3, room="8")
    a, b, c, d = (ok(store.publish_account(room_id="8", text=name, sources=[x], author=MIND))["id"] for name in "ABCD")
    steps = [(a, [x, y], True), (b, [x], True), (c, [b], True), (d, [z], True), (b, [x], True), (a, [x], True),
             (d, [z], False), (c, [b], True)]
    expected_after = []
    for account, replaces, shown in steps:  # the reader agrees with told_map after every single choice
        ok(store.select_account(account, replaces=replaces, author=MIND, reason="step", shown=shown))
        expected_after.append(_told_everywhere(store))
        assert {i: document(root, {"kind": "record", "id": i}, length=1)["told_by"] for i in (x, y, z, a, b, c, d)} == {
            i: expected_after[-1].get(i) for i in (x, y, z, a, b, c, d)}
    unrelated = notes(store, 2500, room="9")
    for n in range(0, 2500, 500):  # a large unrelated selection history the lookups must not read
        extra = ok(store.publish_account(room_id="9", text=f"E{n}", sources=[unrelated[n]], author=MIND))
        ok(store.select_account(extra["id"], replaces=unrelated[n:n + 500], author=MIND, reason="unrelated"))
    oracle = _told_everywhere(store)
    real_records = ChronicleStore.records

    def no_selection_scan(self, *args, **kwargs):
        assert "selection" not in (kwargs.get("kinds") or ()), "a told-by lookup read every selection"
        return real_records(self, *args, **kwargs)

    monkeypatch.setattr(ChronicleStore, "records", no_selection_scan)
    for ident in (x, y, z, a, b, c, d, unrelated[7], unrelated[2499]):
        assert document(root, {"kind": "record", "id": ident}, length=1)["told_by"] == oracle.get(ident), ident
    store.index_path.unlink()  # the told projection is derived: a replayed index answers the same
    assert [document(root, {"kind": "record", "id": i}, length=1)["told_by"] for i in (x, b, unrelated[7])] == [
        oracle.get(x), oracle.get(b), oracle.get(unrelated[7])]
    assert oracle.get(x) == a and oracle.get(b) == c and oracle.get(y) is None and oracle.get(z) is None
    assert expected_after[4].get(b) is None  # B chosen again after C told it: no longer told
    story = {entry["id"]: entry["told_by"] for entry in every(root, "story", limit=100)}
    assert {ident: told for ident, told in story.items() if ident in (a, b, c, d)} == {
        a: None, b: c, c: None}  # D's latest selection withdrew it


def test_the_story_keeps_legacy_coverage_and_refuses_a_page_of_a_changed_story(root):
    store = ChronicleStore(root)
    store.ensure_activated()
    legacy = ok(store.publish([{"id": f"legacy-b0{n}-r1", "kind": "legacy", "room_id": "1", "text": f"retelling {n}",
                                "author": {"kind": "legacy_helper", "writer": "old_consolidator"},
                                "metadata": {"legacy_type": "era", "legacy_block": n}} for n in (1, 2)]))
    first = catalog(root, "story", limit=1)
    assert first["state"] == "active" and [e["kind"] for e in first["entries"]] == ["legacy"] and first["count"] == 2
    whole_story = every(root, "story", limit=1)
    assert [e["id"] for e in whole_story] == ["legacy-b01-r1", "legacy-b02-r1"] and legacy["id"] == "legacy-b02-r1"
    part = ok(store.publish_part(room_id="1", text="The old era, told again.", member_ids=["legacy-b01-r1", "legacy-b02-r1"],
                                 author=MIND, expected_sequence=store.room_head("1")))
    with pytest.raises(MemoryReadError) as changed:
        catalog(root, "story", limit=1, cursor=first["next"])
    assert changed.value.code == "position_mismatch"
    assert [e["id"] for e in every(root, "story")] == [part["id"]]  # folded legacy leaves; the part tells it


def _addresses(value):
    """Every ``address`` an answer hands out (refs, continuations' sources), wherever it sits."""
    if isinstance(value, dict):
        for key in ("address", "derived_from"):  # a derived value names its source: that must resolve too
            if isinstance(value.get(key), dict):
                yield value[key]
        for key, item in value.items():
            if key not in ("address", "derived_from"):
                yield from _addresses(item)
    elif isinstance(value, list):
        for item in value:
            yield from _addresses(item)


def test_a_story_page_after_a_reselection_is_refused_even_when_nothing_visible_moved(root):
    store = ChronicleStore(root)
    store.ensure_activated()
    page = ok(store.publish_page(room_id="1", text="A page.", author=MIND, host_stamp={"tasks": [{"task_id": "t1", "status": "completed"}]},
                                 covers={"mode": "range", "rows": ["r1"], "stream_span": [0, 0]}))
    account = ok(store.publish_account(room_id="1", text="Told once.", sources=[page["id"]], author=MIND))
    first_choice = ok(store.select_account(account["id"], replaces=[], author=MIND, reason="first"))
    first = catalog(root, "story", limit=1)
    assert first["count"] == 2 and first["next"]
    unchanged = catalog(root, "story", limit=1, cursor=first["next"])  # nothing changed: the continuation holds
    assert unchanged["entries"] and unchanged["complete"]
    told = {e["id"]: e for e in first["entries"] + unchanged["entries"]}
    assert told[account["id"]]["selection"] == first_choice["id"]
    ok(store.select_account(account["id"], replaces=[], author=MIND, reason="second"))  # same shown, told_by, text
    with pytest.raises(MemoryReadError) as changed:
        catalog(root, "story", limit=1, cursor=first["next"])
    assert changed.value.code == "position_mismatch"


def test_every_address_a_story_or_record_answer_hands_out_resolves(root):
    store = ChronicleStore(root)
    store.ensure_activated()
    stamp = {"tasks": [{"task_id": f"t{n}", "status": f"status-with-a-long-name-{n}"} for n in range(40)]}  # a long summary
    page = ok(store.publish_page(room_id="1", text="Stamped " * 100, author=MIND, host_stamp=stamp,
                                 covers={"mode": "range", "rows": [f"r{n}" for n in range(30)], "stream_span": [0, 29]}))
    ok(store.publish([{"id": "legacy-gap-1", "kind": "gap", "room_id": "1", "ts": "", "text": "A gap.",
                       "author": {"kind": "host", "operation": "journal_recovery"},
                       "metadata": {"source_gap": "unreadable " * 60}}]))
    account = ok(store.publish_account(room_id="1", text="Across.", sources=[page["id"]], author=MIND))
    ok(store.select_account(account["id"], replaces=[page["id"]], author=MIND, reason="told"))
    answers = [catalog(root, "story", limit=10), catalog(root, "records", room_id="1"),
               document(root, {"kind": "record", "id": page["id"]}, length=10),
               document(root, {"kind": "record", "id": account["id"]}, length=10)]
    found = [address for answer in answers for address in _addresses(answer)]
    assert found and all(address["kind"] in ("record", "note", "file", "entry", "journal", "room", "shelf")
                         for address in found)
    for address in found:
        if address["kind"] in ("record", "note", "file", "entry"):
            document(root, address, length=10)  # resolves, or the answer promised what it cannot keep


def test_a_story_over_dialogue_history_still_importing_is_unavailable_not_imported(root, monkeypatch):
    from ouroboros import dialogue_store
    from ouroboros.store import StoreMigrating

    store = ChronicleStore(root)
    store.ensure_activated()
    ok(store.publish([{"id": "legacy-b01-r1", "kind": "legacy", "room_id": "1", "text": "retelling", "author": {
        "kind": "legacy_helper", "writer": "old_consolidator"}, "metadata": {"legacy_type": "era", "legacy_block": 1}}]))
    asked = []

    def importing(root_, stream, **_bounds):  # the readiness probe's own refusal while an import is pending
        asked.append(stream)
        raise StoreMigrating("Dialogue history is still importing")

    monkeypatch.setattr(dialogue_store, "require_imported", importing)
    with pytest.raises(MemoryReadError) as refused:
        catalog(root, "story")
    assert refused.value.code == "unavailable" and "still importing" in refused.value.message and asked == ["chat"]


def test_the_first_rooms_page_holds_host_records_without_a_room(root):
    store = ChronicleStore(root)
    receipt = store.ensure_activated()  # the real producer: an empty install's activation receipt, room ""
    assert receipt["kind"] == "activation" and receipt["room_id"] == ""
    notes(store, 1, room="1")
    first = catalog(root, "rooms", limit=1)
    assert [entry["room_id"] for entry in first["entries"]] == [""] and first["chronicle"]["activation"]["id"]
    assert first["entries"][0]["label"] == "Host records without a room" and not first["complete"]
    second = catalog(root, "rooms", limit=1, cursor=first["next"])
    assert [entry["room_id"] for entry in second["entries"]] == ["1"] and second["complete"]


def test_the_host_room_is_selected_by_its_exact_empty_id_and_an_omitted_room_is_every_room(root):
    store = ChronicleStore(root)
    receipt = store.ensure_activated()
    (note,) = notes(store, 1, room="1")
    host = ok(store.mark({"kind": "chronicle", "id": receipt["id"]}, "Held by the host.", MIND, room_id=""))
    (room, _one) = catalog(root, "rooms")["entries"]
    records = every(root, "records", room_id=room["address"]["room_id"], limit=1)  # the address the rooms page gave
    assert [entry["id"] for entry in records] == [receipt["id"], host["id"]]
    assert [entry["id"] for entry in every(root, "marks", room_id="")] == [host["id"]]
    assert {entry["id"] for entry in every(root, "records")} == {receipt["id"], note, host["id"]}
    assert {entry["id"] for entry in every(root, "marks", room_id=None)} == {host["id"]}
    for area, params in (("records", {"room_id": 0}), ("marks", {"room_id": "\x00"}), ("shelf", {"scope": ""})):
        with pytest.raises(MemoryReadError) as refused:  # only a room's id may be empty
            catalog(root, area, **params)
        assert refused.value.code == "invalid_parameter"


def test_a_journal_page_stops_before_a_line_beyond_its_window_and_reads_that_line_alone(root, monkeypatch):
    monkeypatch.setattr(memory_reader, "SCAN_BYTES", 4096)
    path = root / "projects" / "atlas-proj" / "journal.jsonl"
    path.parent.mkdir(parents=True)
    huge = "Огромная запись. " * 6000
    rows = [{"ts": "1", "kind": "note", "text": "small one"},
            {"ts": "2", "kind": "note", "text": huge, "items": list(range(5000))},
            {"ts": "3", "kind": "note", "text": "small two"}]
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    address = {"kind": "journal", "journal": "project:atlas-proj"}
    first = catalog(root, "history", address=address)
    assert [row["fields"]["text"] for row in first["entries"]] == ["small one"] and not first["complete"]
    second = catalog(root, "history", address=address, cursor=first["next"])
    assert size(second) < SMALL and len(second["entries"]) == 1  # the one record-sized read
    fields = second["entries"][0]["fields"]
    assert fields["text"]["chars"] == len(huge) and fields["items"]["count"] == 5000
    assert whole(root, fields["text"]["address"], length=40_000) == huge
    assert json.loads(whole(root, fields["items"]["address"])) == list(range(5000))
    third = catalog(root, "history", address=address, cursor=second["next"])
    assert [row["fields"]["text"] for row in third["entries"]] == ["small two"] and third["complete"]


def test_a_long_authored_summary_is_listed_as_a_ref_that_reads_exactly_that_summary(root):
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note

    summary = "Сводка, написанная автором. " * 120  # past the shelf's inline limit
    body = "# Note\n\nThe body, which is not the summary.\n"
    content = "---\nsummary: " + json.dumps(summary, ensure_ascii=False) + "\n---\n" + body
    assert write_knowledge_note(resolve_knowledge_address(root, "long/сводка", "global"), content, writer="turn").ok
    (entry,) = every(root, "shelf", scope="global")
    assert entry["summary"]["ref"] == "text" and entry["summary"]["chars"] == len(summary)
    assert entry["summary"]["address"] == {"kind": "note", "scope": "global", "topic": "long/сводка", "field": ["summary"]}
    assert whole(root, entry["summary"]["address"], length=1000) == summary  # the summary itself, never the body
    short = "One line."
    assert write_knowledge_note(resolve_knowledge_address(root, "short", "global"),
                                f"---\nsummary: {short}\n---\n{body}", writer="turn").ok
    assert {e["topic"]: e["summary"] for e in every(root, "shelf", scope="global")}["short"] == short


@pytest.mark.parametrize("params", [{"area": ["records"]}, {"area": "records", "author_kind": ["helper"]},
                                    {"area": "records", "kinds": [["page"]]}, {"area": "marks", "state": ["all"]},
                                    {"area": "relations", "address": {"kind": "record", "id": "x"}, "relation": ["members"]}])
def test_list_shaped_parameters_are_refused_not_raised(root, params):
    ChronicleStore(root).write_note(room_id="1", task_id="t1", text="x", author=MIND)
    with pytest.raises(MemoryReadError) as refused:
        read_memory(root, "catalog", params)
    assert refused.value.code == "invalid_parameter"
