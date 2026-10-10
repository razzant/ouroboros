"""Compact common stories retain exact composition and unincorporated source meaning."""
from __future__ import annotations

import json

from ouroboros import chat_chain, memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.projects_registry import create_project
from ouroboros.tools.chronicle import _memory_read
from tests import _memory_inventory_shared as shared
from tests.test_chronicle_tools import _parse_window
from tests.test_common_story import KID, _block, _ctx, _journal, _page, _rooms, _snapshot, _story, _write


ACCOUNT = "\n\n".join([
    "I first approached the repair as a question of speed. In the workshop, people asked for a working "
    "instrument and I answered with increasingly elaborate checks. Those checks found useful faults, "
    "but they also kept changing the definition of finished work. I learned to hold the original promise "
    "beside the new evidence, so an interesting problem could be recorded without silently replacing it.",
    "In the archive room I made the opposite mistake. A short account sounded complete because I could "
    "still find the raw conversation. Being able to recover a detail is different from remembering why "
    "it matters now. The account needed the reason for our choice, the conditions under which it held, "
    "and the unresolved question. Exact sources remained necessary when a later conversation challenged it.",
    "The garden discussion helped me connect those experiences. A shared plan did not make every local "
    "observation interchangeable. One room had measured a result; another had only proposed an explanation. "
    "I could tell their common story without treating the proposal as a measurement or taking ownership "
    "of the local records. Each room still needed enough detail to continue its own work.",
    "When the measurements changed, the earlier account did not become dishonest merely by remaining "
    "in the archive. It became misleading when I repeated it as the whole present picture. I want my "
    "current understanding to carry the correction while leaving the earlier words available as part "
    "of how I arrived here. Writing a new account is an act of understanding, not a proof that I understood.",
    "I also noticed that a confident explanation can hide a missing source. A lost record gives me a "
    "gap to acknowledge and investigate, not permission to infer its contents from neighboring entries. "
    "A period describes the span represented by these records; it does not promise that every event "
    "within that period was observed, remembered, or settled.",
    "What carries across these rooms is therefore a practice of keeping promises and evidence distinct. "
    "I can revise my interpretation, return to an earlier one, or hold several accounts together. "
    "The records do not choose among them for me. They let me reconsider with the actual words, dates, "
    "versions and outcomes available, while the common account keeps the experience intelligible.",
])


def _read_all(root, node, **args):
    """Use the real character-window reader to EOF, checking its range continuity."""
    chunks, start = [], 0
    while True:
        text = _memory_read(_ctx(root), node_id=node, start=start, **args)
        begin, end, total, more = _parse_window(text)
        assert begin == start and end >= begin
        chunks.append(text.split("\n", 2)[2])
        if more is None:
            assert end == total
            document = "".join(chunks)
            assert len(document) == total
            return document, len(chunks)
        assert end > start
        start = end


def _many_sources(root, count):
    shared.world(root)
    rows = []
    for n in range(count):
        project = create_project(root, f"room-{n}", name=f"Workshop {n}: a long descriptive room name")
        rows.append(shared.msg(f"2026-10-{n + 1:02d}T10:00:00+00:00", f"Observation from room {n}.",
                               chat_id=project["chat_id"]))
    shared.append(root / "logs" / "chat.jsonl", *rows)
    first = len(list(chat_chain.iter_rows(root))) - count
    pages = [_page(root, str(row["chat_id"]), first + n, first + n,
                   text=f"Local account {n}: We recorded the proposed repair and the observation separately.")
             for n, row in enumerate(rows)]
    account = _write(root, kind="account", text=ACCOUNT, sources=pages)
    assert account["ok"]
    chosen = _write(root, kind="selection", target_id=account["node_id"], replaces=pages,
                    reason="I now tell the experience together while each room retains its own account.")
    assert chosen["ok"]
    return pages, account["node_id"], chosen["node_id"], rows


def test_resident_composition_is_bounded_while_exact_reader_retains_twenty_four_rooms(tmp_path):
    blocks = []
    for count in (2, 24):
        root = tmp_path / str(count)
        pages, account, selection, rows = _many_sources(root, count)
        snapshot = _snapshot(root)
        journal = _journal(root)
        story = mv.render_story(snapshot)
        block = _block(story, account)
        blocks.append(block)
        assert all("  " + paragraph in block for paragraph in ACCOUNT.split("\n\n"))
        assert f"from {count} sources" in block and f"{count} story records told through this account" in block
        assert block.count("memory_read(") == 1
        assert f"memory_read(node_id='{account}')" in block
        assert f"shown by selection {selection}" in block
        assert f"2026-10-01 10:00 → 2026-10-{count:02d} 10:00" in block
        exact, _pages = _read_all(root, account)
        for source, row in zip(pages, rows):
            assert f"source page {source} (room {row['chat_id']}): revision {source} used" in exact
        assert f"replaces {count}: " + ", ".join(pages) + ";" in exact
        assert f"selection {selection} by mind" in exact and ACCOUNT in exact
        # The common view reaches children identically; room detail remains independently usable.
        assert _story(root, KID) == story
        local = mv.render_room(_snapshot(root, {"id": "rootA001", "chat_id": rows[-1]["chat_id"]}))
        assert f"Local account {count - 1}: We recorded the proposed repair" in local
        assert _journal(root) == journal
    # Growing both source and replacement lists changes counts, never the resident line population.
    assert len(blocks[0].splitlines()) == len(blocks[1].splitlines())
    assert abs(len(blocks[1]) - len(blocks[0])) < 20


def test_correction_meaning_stays_until_a_new_direct_source_version_is_authored(tmp_path):
    _rooms_map, _alpha, _beta, page, other, _draft = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    first = _write(tmp_path, kind="correction", target_id=page,
                   text="The initial count was based on a rehearsal, not a completed repair.")
    assert first["ok"]
    old = _write(tmp_path, kind="account", text=ACCOUNT,
                 sources=[{"id": page, "revision": first["node_id"]}, other])
    assert old["ok"]
    old_id = old["node_id"]
    assert _write(tmp_path, kind="selection", target_id=old_id, replaces=[page, other], reason="My account.")["ok"]
    correction = ("The workshop measurement was withdrawn. The planned delivery still needs a new observation.\n" * 950
                  + "The final exception is that the garden's independent measurement remains valid. Ω")
    late = _write(tmp_path, kind="correction", target_id=page, expected_revision=first["node_id"], text=correction)
    assert late["ok"]
    late_id = late["node_id"]
    pending = _block(_story(tmp_path), old_id)
    assert "\n".join("  " + line for line in correction.splitlines()) in pending
    assert first["node_id"] in pending and first["node_id"] != late_id
    assert first["node_id"] == store.get(old_id)["sources"][0]["revision"]
    # Prose about understanding and a re-selection reason cannot rewrite the old source edge.
    assert _write(tmp_path, kind="correction", target_id=old_id, text="I understand the withdrawal now.")["ok"]
    assert _write(tmp_path, kind="selection", target_id=old_id, replaces=[page, other],
                  reason=f"I incorporated {late_id}.")["ok"]
    unchanged = _block(_story(tmp_path), old_id)
    assert "I understand the withdrawal now." in unchanged
    assert "\n".join("  " + line for line in correction.splitlines()) in unchanged
    assert f"I incorporated {late_id}." not in unchanged
    updated = _write(tmp_path, kind="account", text="I no longer treat the workshop's withdrawn measurement as a result. "
                     "The garden's independent result remains valid; the workshop still owes a new observation.",
                     sources=[{"id": page, "revision": late_id}, other])
    assert updated["ok"]
    updated_id = updated["node_id"]
    assert _write(tmp_path, kind="selection", target_id=updated_id, replaces=[old_id, page, other],
                  reason="This new account carries my revised understanding.")["ok"]
    current = _story(tmp_path)
    assert "garden's independent result remains valid" in current and correction.splitlines()[0] not in current
    assert ACCOUNT.split("\n\n")[0] not in current
    exact_late, windows = _read_all(tmp_path, late_id)
    assert exact_late.split("text:\n", 1)[1] == correction and windows > 1
    exact_current, _ = _read_all(tmp_path, updated_id)
    assert f"source page {page}" in exact_current and f"revision {late_id} used" in exact_current
    assert f"replaces 3: {old_id}, {page}, {other};" in exact_current
    exact_old, _ = _read_all(tmp_path, old_id)
    assert f"revision {first['node_id']} used; later corrections not in this account: {late_id}" in exact_old
    assert ACCOUNT in exact_old
    assert "I understand the withdrawal now." in exact_old
    # Citing that old account retains its stale nested edge even alongside a corrected direct edge.
    nested = _write(tmp_path, kind="account", text="A later account that still cites my old account.",
                    sources=[old_id, {"id": page, "revision": late_id}])
    assert nested["ok"]
    assert _write(tmp_path, kind="selection", target_id=nested["node_id"], replaces=[updated_id, old_id, page, other],
                  reason="Retain both bases explicitly.")["ok"]
    still_pending = _block(_story(tmp_path), nested["node_id"])
    assert f"cited revision {first['node_id']}" in still_pending and f"via {old_id}" in still_pending
    assert "\n".join("  " + line for line in correction.splitlines()) in still_pending


def test_rejection_meaning_stays_until_a_new_account_binds_the_rejected_source(tmp_path):
    _rooms_map, _alpha, _beta, _page_id, other, draft = _rooms(tmp_path)
    old = _write(tmp_path, kind="account", text="I relied on a helper's provisional measurement.", sources=[draft, other])
    assert old["ok"]
    assert _write(tmp_path, kind="selection", target_id=old["node_id"], replaces=[draft, other], reason="Together.")["ok"]
    reason = "The draft confused a simulation with an observation. " * 120 + "The local experiment remains undone."
    rejected = _write(tmp_path, kind="decision", target_id=draft, accepted=False, reason=reason)
    assert rejected["ok"]
    assert reason in _block(_story(tmp_path), old["node_id"])
    revised = _write(tmp_path, kind="account", text="The helper's simulation is rejected as measurement; "
                     "the local experiment still needs to run.", sources=[draft, other])
    assert revised["ok"]
    assert _write(tmp_path, kind="selection", target_id=revised["node_id"], replaces=[old["node_id"], draft, other],
                  reason="My new account records the rejection and the remaining work.")["ok"]
    story = _story(tmp_path)
    assert "local experiment still needs to run" in story and reason not in story
    stored = ChronicleStore(tmp_path)
    assert stored.get(old["node_id"])["sources"][0]["status"] == "draft"
    assert stored.get(revised["node_id"])["sources"][0]["status"] == "rejected"
    exact, _ = _read_all(tmp_path, rejected["node_id"])
    assert exact.split("text:\n", 1)[1] == reason
    original, _ = _read_all(tmp_path, draft)
    assert reason in original and "A helper's draft about Beta." in original


def test_unknown_cited_revision_remains_a_gap_beside_the_compact_account(tmp_path):
    _rooms_map, _alpha, _beta, page, other, _draft = _rooms(tmp_path)
    fix = _write(tmp_path, kind="correction", target_id=page, text="A source correction later missing from a copy.")
    assert fix["ok"]
    account = _write(tmp_path, kind="account", text=ACCOUNT, sources=[page, other])
    assert account["ok"]
    assert _write(tmp_path, kind="selection", target_id=account["node_id"], replaces=[page, other], reason="Together.")["ok"]
    records = tmp_path / "memory" / "chronicle" / "records.jsonl"
    copied = tmp_path / "copy"
    target = copied / "memory" / "chronicle" / "records.jsonl"
    target.parent.mkdir(parents=True)
    target.write_text("".join(line for line in records.read_text(encoding="utf-8").splitlines(keepends=True)
                              if all(r["id"] != fix["node_id"] for r in json.loads(line)["records"])), encoding="utf-8")
    store = ChronicleStore(copied)
    snapshot = mv.MemoryViewSnapshot(spec=mv.ROLE_DEFAULTS["integrator"], store_status={"state": "active"}, frontier={},
                                     story=tuple(mv._story_pages(store, lambda room: f"room {room}", {})))
    shown = mv.render_story(snapshot)
    assert f"that revision of {page} is not in this chronicle; its current one is {page}" in shown
    assert f"revision {fix['node_id']}" in shown
    exact, _ = _read_all(copied, account["node_id"])
    assert "that revision is not in this chronicle" in exact and ACCOUNT in exact
