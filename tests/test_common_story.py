"""My account across rooms and its selection for the common view (``chronicle_store``, ``memory_view``, tools).

An account is my own text over exact source versions of several rooms; it is provenance, never
ownership: publishing it seals no row, folds nothing, moves no head and leaves every source free
to inform another account or to be folded locally. Selection is a separate act: it shows the
account in the story in place of the records it names, whose count, period and composition reader stay visible and
keep their detail on their room's page. A later correction, rejection or fold of a source is
visible as not in the account; the account's own basis never changes, and a copied or missing
source is a disclosed gap, not fabricated text. Root and child read the same account; a nanny
reads none. Every rule is pinned in both directions on the shared world fixture; no test calls
a model or the network, and no real record, room or words of the owner's chronicle enters here.
"""
from __future__ import annotations

import hashlib
import json
import re

import pytest

from ouroboros import chat_chain
from ouroboros import memory_floor as mf
from ouroboros import memory_inventory as mi
from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore, body_sha256
from ouroboros.tools.chronicle import _chronicle_write, _memory_read
from ouroboros.tools.registry import ToolContext
from tests import _memory_inventory_shared as shared

MAIN = {"id": "turn0001", "chat_id": 1}
KID = {"id": "kid00001", "chat_id": 1, "delegation_role": "subagent", "root_task_id": "turn0001"}
NANNY = {**KID, "id": "nan00001", "configured_subagent": {"route": {"kind": "agent_session"}}}
HELPER = {"kind": "helper", "route": "configured-light"}
G_TEXT = "Across Alpha and Main I learned the same lesson twice: count before promising."
G2_TEXT = "Alpha's count and Beta's question were one question about trust."
NEXT_TEXT = "Told once more, with both earlier accounts behind me."


def _ctx(root, task_id="turn0001", **fields):
    fields.setdefault("current_chat_id", 1)
    return ToolContext(repo_dir=root, drive_root=root, task_id=task_id, **fields)


def _write(root, **args):
    return json.loads(_chronicle_write(_ctx(root), **args))


def _snapshot(root, task=MAIN):
    return mv.capture_memory_view(root, task, mv.view_spec_for_task(task, root))


def _story(root, task=MAIN):
    return mv.render_story(_snapshot(root, task))


def _page(root, room, first, last, *, author=shared.MIND, text=None):
    from ouroboros.tools.chronicle import page_covers

    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=text or f"Page {room} {first}-{last}.", covers=covers,
                                               author=author)
    assert result.ok, result
    return result.record["id"]


def _part(root, room, members, text="A part of my story."):
    store = ChronicleStore(root)
    result = store.publish_part(room_id=room, text=text, member_ids=members, author=shared.MIND,
                                expected_sequence=store.room_head(room))
    assert result.ok, result
    return result.record["id"]


def _journal(root):
    return hashlib.sha256((root / "memory" / "chronicle" / "records.jsonl").read_bytes()).hexdigest()


def _block(text, record_id):
    """One record's block of the story: from its ``### … <id>`` header to the next ``### `` header."""
    start = text.index(f" {record_id}\n")
    start = text.rindex("\n### ", 0, start)
    end = text.find("\n### ", start + 1)
    return text[start:end if end > 0 else len(text)]


def _rooms(root):
    """The world plus one page of Alpha (A1), one of Main (B1) and a helper's draft in Beta (H1)."""
    rooms = shared.world(root)
    alpha, beta = str(rooms["alpha"]), str(rooms["beta"])
    a1 = _page(root, alpha, 13, 13, text="Alpha asked again and I counted.")
    b1 = _page(root, "1", 10, 11, text="Main started and the owner asked me to go on.")
    h1 = _page(root, beta, 14, 14, author=HELPER, text="A helper's draft about Beta.")
    return rooms, alpha, beta, a1, b1, h1


# --- A. publication is provenance, selection is display -----------------------------------------------

def test_publishing_an_account_freezes_its_sources_and_changes_no_local_fact(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    before = (store.sealed_row_refs(alpha), store.sealed_row_refs("1"), store.room_head(alpha), store.room_head("1"),
              {r["id"]: r["folded_into"] for r in store.pages_of_room(alpha) + store.pages_of_room("1")})
    bytes_before = _journal(tmp_path)
    story_before = _story(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, {"id": b1}], author=shared.MIND)
    assert g.ok and g.current_head is None and g.record["kind"] == "account"
    frozen = g.record["sources"]
    assert [src["id"] for src in frozen] == [a1, b1]
    assert frozen[0] == {"id": a1, "kind": "page", "room_id": alpha, "revision": a1, "sha256": body_sha256(store.get(a1)),
                         "status": "final"}
    assert frozen[1]["room_id"] == "1" and len(frozen[1]["sha256"]) == 64
    # Nothing local moved: sealed sets, heads and folds are as before, and the journal only grew.
    after = (store.sealed_row_refs(alpha), store.sealed_row_refs("1"), store.room_head(alpha), store.room_head("1"),
             {r["id"]: r["folded_into"] for r in store.pages_of_room(alpha) + store.pages_of_room("1")})
    assert after == before and _journal(tmp_path) != bytes_before
    assert (tmp_path / "memory" / "chronicle" / "records.jsonl").read_bytes().count(b"\n") > 0
    # Publication alone is not selection: the story is unchanged but for its accounts line.
    story = _story(tmp_path)
    assert G_TEXT not in story and a1 in story and b1 in story
    assert story.replace("\nMy accounts across rooms: 1 written, 0 in the common view (the rest one memory_read away).", "") == story_before
    # Read back through the actual reader: who wrote it, which versions, how to read the originals.
    read = _memory_read(_ctx(tmp_path), node_id=g.record["id"])
    assert read.startswith(f"[account {g.record['id']}; room 1; mind (root t1); based on 2 sources, ")
    assert f"source page {a1} (room {alpha}): revision {a1} used" in read and "changed since" not in read.split("\n")[0]
    assert f"memory_read(node_id={a1}, revision={a1}) reads that version" in read
    assert "shown in the common view only by my selection" in read and G_TEXT in read


def test_selection_replaces_only_the_named_records_and_the_rest_stay_whole(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    chosen = store.select_account(g, replaces=[a1], author=shared.MIND, reason="Alpha's page is told through it.")
    assert chosen.ok and chosen.record["kind"] == "selection" and chosen.current_head is None
    story = _story(tmp_path)
    block = _block(story, g)
    assert block.startswith(f"\n### My account across rooms · 2026-09-03 00:00 → 2026-09-03 00:03 · account {g}\n")
    assert f"  {G_TEXT}\n- written " in block and "by me (root, task t1) from 2 sources; their rows span the period above, not every event in it" in block
    assert f"exact composition, source versions and selections: memory_read(node_id='{g}')" in block
    assert "1 story records told through this account (including nested selections); 2026-09-03 00:03 → 2026-09-03 00:03" in block
    exact = _memory_read(_ctx(tmp_path), node_id=g)
    assert f"source page {a1} (room {alpha}): revision {a1} used" in exact
    assert f"source page {b1} (room 1): revision {b1} used" in exact
    assert f"shown=True; replaces 1: {a1};" in exact
    assert "Alpha asked again and I counted." not in story  # A1's meaning now lives in the selected account
    assert f"### Main · 2026-09-03 00:00 → 2026-09-03 00:01 · page {b1}\n  Main started" in story  # B1 stays whole
    assert "A helper's draft about Beta." in story  # H1 untouched
    assert "\nMy accounts across rooms: 1 written, 1 in the common view.\n" in story + "\n"
    # Root and child read the same account block; the story is the same bytes; a nanny has none.
    assert _story(tmp_path, KID) == story and mv.render_story(_snapshot(tmp_path, NANNY)) == ""
    # The local reader keeps A1 whole: a root working in Alpha sees its page's detail on the room page.
    room = mv.render_room(_snapshot(tmp_path, {"id": "rootA001", "chat_id": rooms["alpha"]}))
    assert f"### Pages of this room\n#### {a1} — 2026-09-03 00:03 → 2026-09-03 00:03\n  Alpha asked again and I counted." in room
    assert G_TEXT not in room
    main_room = mv.render_room(_snapshot(tmp_path))
    assert "### Pages of this room" not in main_room and "Main started" not in main_room  # B1 is whole in the story
    # An empty replaces list shows the account beside everything; a withdrawn selection takes it out again.
    assert store.select_account(g, replaces=[], author=shared.MIND, reason="beside the pages").ok
    beside = _story(tmp_path)
    assert G_TEXT in beside and "Alpha asked again and I counted." in beside and "told through" not in beside
    assert store.select_account(g, replaces=[a1], shown=False, author=shared.MIND, reason="not yet").ok
    withdrawn = _story(tmp_path)
    assert G_TEXT not in withdrawn and "Alpha asked again and I counted." in withdrawn
    assert "\nMy accounts across rooms: 1 written, 0 in the common view (the rest one memory_read away).\n" in withdrawn + "\n"



@pytest.mark.parametrize("target_kind", ["page", "legacy"])
def test_latest_explicit_selection_wins_overlap_without_reordering_accounts(tmp_path, target_kind):
    _rooms_map, _alpha, _beta, a1, b1, _h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    target = a1 if target_kind == "page" else "legacy-b01-r1"
    original = store.get(target)
    g1 = store.publish_account(room_id="1", text=G_TEXT, sources=[target, b1], author=shared.MIND).record["id"]
    g2 = store.publish_account(room_id="1", text=G2_TEXT, sources=[target, b1], author=shared.MIND).record["id"]
    bases = {ident: store.get(ident) for ident in (g1, g2)}

    def choose(account, *, replaces=None, shown=True):
        result = _write(tmp_path, kind="selection", target_id=account,
                        replaces=[target] if replaces is None else replaces,
                        shown=shown, reason="A new explicit choice of the common view.")
        assert result["ok"], result
        return result

    def assert_told_by(account, other):
        story = _story(tmp_path)
        assert "1 story records told through this account" in _block(story, account)
        assert next(e for e in _snapshot(tmp_path).story if e["id"] == target)["told_by"] == account
        assert f"replaces 1: {target};" in _memory_read(_ctx(tmp_path), node_id=account)
        if other is not None:
            assert "story records told through this account" not in _block(story, other)
            # A new selection changes display ownership, not the accounts' chronology.
            assert story.index(f" · account {g1}\n") < story.index(f" · account {g2}\n")
        assert _story(tmp_path, KID) == story
        return story

    choose(g1)
    choose(g2)
    assert_told_by(g2, g1)  # later publication and later selection happen to agree
    latest = choose(g1)
    assert latest["sequence"] > next(a for a in store.accounts() if a["id"] == g2)["selection"]["sequence"]
    selected = assert_told_by(g1, g2)  # publication order must not defeat this explicit re-selection
    journal = _journal(tmp_path)
    store.index_path.unlink()
    assert _story(tmp_path) == selected and _journal(tmp_path) == journal  # projection replay keeps the choice
    choose(g1, replaces=[])
    assert_told_by(g2, g1)  # G1 is still shown, but no longer claims this source
    choose(g1)
    assert_told_by(g1, g2)
    choose(g1, shown=False)
    assert G_TEXT not in assert_told_by(g2, None)  # withdraw only G1; G2's selection still acts
    choose(g2, shown=False)
    restored = _story(tmp_path)
    assert G2_TEXT not in restored and original["text"] in restored
    assert not any(row.get("told_by") for row in _snapshot(tmp_path).story if row["id"] == target)
    assert store.get(target) == original and all(store.get(ident) == value for ident, value in bases.items())



@pytest.mark.parametrize("chain_length", [2, 3])
def test_reselecting_an_account_reveals_it_after_two_or_three_link_replacement(tmp_path, chain_length):
    _rooms_map, _alpha, _beta, a1, b1, _h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    accounts = []
    for index in range(chain_length):
        sources = [accounts[-1]] if accounts else [a1, b1]
        account = store.publish_account(room_id="1", text=f"Synthetic account body {index + 1}.",
                                        sources=sources, author=shared.MIND).record["id"]
        accounts.append(account)
        assert _write(tmp_path, kind="selection", target_id=account, replaces=sources,
                      reason="The next authored account tells the previous one.")["ok"]
    originals = {ident: store.get(ident) for ident in [a1, b1, *accounts]}
    initial = _story(tmp_path)
    assert f"Synthetic account body {chain_length}." in initial
    assert "Synthetic account body 1." not in initial
    # One composition pointer retains the nested path to every exact source and selection.
    assert f"{chain_length + 1} story records told through this account" in initial
    assert f"memory_read(node_id='{accounts[-1]}')" in initial
    assert all(ident in _memory_read(_ctx(tmp_path), node_id=accounts[0]) for ident in (a1, b1))
    for current, previous in zip(accounts[1:], accounts):
        assert f"replaces 1: {previous};" in _memory_read(_ctx(tmp_path), node_id=current)

    assert _write(tmp_path, kind="selection", target_id=accounts[0], replaces=[accounts[-1], a1, b1],
                  reason="Return to the earlier understanding, keeping its exact sources.")["ok"]
    chosen = _story(tmp_path)
    assert "Synthetic account body 1." in chosen
    assert all(f"Synthetic account body {index + 1}." not in chosen for index in range(1, chain_length))
    assert f"{chain_length + 1} story records told through this account" in chosen
    assert f"memory_read(node_id='{accounts[0]}')" in chosen
    assert f"replaces 3: {accounts[-1]}, {a1}, {b1};" in _memory_read(_ctx(tmp_path), node_id=accounts[0])
    assert not next(e for e in _snapshot(tmp_path).story if e["id"] == accounts[0]).get("told_by")
    assert _story(tmp_path, KID) == chosen

    assert _write(tmp_path, kind="selection", target_id=accounts[0], replaces=[], shown=False,
                  reason="Withdraw this account without changing the other selections.")["ok"]
    withdrawn = _story(tmp_path)
    assert "Synthetic account body 1." not in withdrawn
    assert f"Synthetic account body {chain_length}." in withdrawn  # last remaining selection still acts
    assert originals[a1]["text"] in withdrawn and originals[b1]["text"] in withdrawn
    assert all(store.get(ident) == record for ident, record in originals.items())


def test_selection_and_account_refusals_name_ids_and_land_nothing(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    before = _journal(tmp_path)
    missing = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, "no-such"], author=shared.MIND)
    assert (missing.reason, missing.conflict_ids) == ("target_missing", ("no-such",))
    wrong = store.publish_account(room_id="1", text=G_TEXT, sources=[{"id": a1, "revision": b1}], author=shared.MIND)
    assert (wrong.reason, wrong.current_revision, wrong.conflict_ids) == ("revision_conflict", a1, (a1,))
    assert store.publish_account(room_id="1", text=G_TEXT, sources=[], author=shared.MIND).reason == "invalid"
    assert store.publish_account(room_id="1", text=" ", sources=[a1], author=shared.MIND).reason == "invalid"
    assert store.publish_account(room_id="1", text=G_TEXT, sources=[a1], author=HELPER).reason == "invalid"
    forged = store.publish([{"kind": "account", "room_id": "1", "text": G_TEXT, "author": shared.MIND, "sources": [
        {"id": a1, "kind": "page", "room_id": alpha, "revision": a1, "sha256": "0" * 64}]}])
    assert forged.reason == "invalid" and "sha256" in forged.detail
    assert store.select_account("no-such", replaces=[], author=shared.MIND, reason="x").reason == "target_missing"
    assert store.select_account(a1, replaces=[], author=shared.MIND, reason="x").reason == "invalid"  # a page is no account
    assert _journal(tmp_path) == before
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1], author=shared.MIND).record["id"]
    for bad in (dict(replaces=[a1], reason=" "), dict(replaces=[a1, a1], reason="twice"), dict(replaces=[g], reason="itself"),
                dict(replaces=["gone"], reason="x"), dict(replaces=[a1], reason="x", shown="yes")):
        refused = store.select_account(g, author=shared.MIND, **bad)
        assert not refused.ok and refused.reason in ("invalid", "target_missing"), bad
    assert store.select_account(g, replaces=[a1], author=HELPER, reason="helpers do not choose").reason == "invalid"


# --- B. a source informs several accounts and is still folded locally ----------------------------------

def test_two_accounts_cite_one_source_and_a_local_fold_over_it_still_works(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    c1 = _page(tmp_path, beta, 8, 9, text="Beta asked and I answered.")  # another room's page
    g2 = store.publish_account(room_id="1", text=G2_TEXT, sources=[a1, c1], author=shared.MIND)
    assert g2.ok and [src["id"] for src in g2.record["sources"]] == [a1, c1]
    # A legitimate same-room, same-kind, adjacent fold over A1 and a new page A2 through the ordinary writer.
    a2 = _page(tmp_path, alpha, 2, 3, text="Alpha's first question, sealed late.")  # earlier rows, adjacent in story order
    head = store.room_head(alpha)
    local = store.publish_part(room_id=alpha, text="Alpha, told once.", member_ids=[a1, a2], author=shared.MIND,
                               expected_sequence=head)
    assert local.ok, local  # G's use of A1 adds no already_folded refusal
    assert store.get(a1) is not None and {r["id"]: r["folded_into"] for r in store.pages_of_room(alpha)}[a1] == local.record["id"]
    # Both accounts keep their basis; the fold is visible as a later change of the source, not as ownership.
    for account in store.accounts():
        src = next(s for s in account["sources"] if s["id"] == a1)
        assert src["revision"] == a1 and src["folded_into"] == local.record["id"] and src["later_corrections"] == []
    assert store.select_account(g, replaces=[a1], author=shared.MIND, reason="through G").ok
    assert store.select_account(g2.record["id"], replaces=[c1], author=shared.MIND, reason="through G2").ok
    story = _story(tmp_path)
    assert G_TEXT in story and G2_TEXT in story  # two useful accounts coexist
    assert f"- its source {a1} has since been folded into part {local.record['id']} (not in this account)" in _block(story, g)
    assert f"part {local.record['id']}" in story and "Alpha, told once." in story  # the new local part is new, whole
    assert "Beta asked and I answered." not in story and "1 story records told through this account" in _block(story, g2.record["id"])
    assert f"replaces 1: {c1};" in _memory_read(_ctx(tmp_path), node_id=g2.record["id"])


# --- C. a later correction of a source, and a changed or missing source identity ---------------------

def test_a_correction_after_the_account_keeps_its_basis_and_is_shown_as_not_incorporated(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    assert store.select_account(g, replaces=[a1, b1], author=shared.MIND, reason="both told through it").ok
    fix = store.correct(a1, "I counted twice, not once.", shared.MIND, expected_sequence=store.room_head(alpha))
    assert fix.ok
    saved = store.get(g)["sources"][0]
    assert saved == {"id": a1, "kind": "page", "room_id": alpha, "revision": a1, "sha256": body_sha256(store.get(a1)),
                     "status": "final"}  # the journal's account record is byte-for-byte its first basis
    acting = next(s for s in store.accounts()[0]["sources"] if s["id"] == a1)
    assert acting["later_corrections"] == [fix.record["id"]] and acting["current_revision"] == fix.record["id"]
    assert acting["revision"] == a1 and acting["revision_known"] is True
    story = _story(tmp_path)
    block = _block(story, g)
    assert f"- later correction of its source {a1} (not in this account):\n  I counted twice, not once." in block
    assert "correction by me of" not in block  # not a fold's correction line: the relation is provenance
    assert G_TEXT in block  # the account's own words are untouched
    # The local corrected text is usable: the room page and the reader show the page with its correction.
    room = mv.render_room(_snapshot(tmp_path, {"id": "rootA001", "chat_id": rooms["alpha"]}))
    assert "Alpha asked again and I counted.\n\n  [correction " in room and "I counted twice, not once." in room
    read = _memory_read(_ctx(tmp_path), node_id=g)
    assert f"revision {a1} used; later corrections not in this account: {fix.record['id']}; memory_read(node_id={a1}, revision={a1})" in read
    assert "based on 2 sources, " in read.split("\n")[0] and "; 1 changed since;" in read.split("\n")[0]
    # The exact old version: the original alone, with the later correction named, never shown as that version.
    old = _memory_read(_ctx(tmp_path), node_id=a1, revision=a1)
    assert f"as of revision {a1}: the original alone; later corrections not shown: {fix.record['id']}" in old
    assert "I counted twice" not in old and "text:\nAlpha asked again and I counted." in old
    assert f"informs account {g} (revision {a1} read;" in old
    current = _memory_read(_ctx(tmp_path), node_id=a1, revision=fix.record["id"])
    assert "the original and its corrections up to that one; none later" in current and "I counted twice" in current
    assert "TOOL_ARG_ERROR" in _memory_read(_ctx(tmp_path), node_id=a1, revision=b1)
    assert "TOOL_ARG_ERROR" in _memory_read(_ctx(tmp_path), revision=a1)


def test_a_source_missing_from_a_copied_journal_is_a_gap_not_text(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    assert store.select_account(g, replaces=[], author=shared.MIND, reason="shown").ok
    # A copy of the journal without A1's transaction: the account and its reference survive, the source does not.
    log = tmp_path / "memory" / "chronicle" / "records.jsonl"
    kept = [line for line in log.read_text(encoding="utf-8").splitlines(keepends=True)
            if not any(record["id"] == a1 for record in json.loads(line)["records"])]
    copy = tmp_path / "copy"
    (copy / "memory" / "chronicle").mkdir(parents=True)
    (copy / "memory" / "chronicle" / "records.jsonl").write_text("".join(kept), encoding="utf-8")
    other = ChronicleStore(copy)
    account = other.accounts()[0]
    gone, there = account["sources"]
    assert gone["missing"] is True and gone["revision_known"] is False and gone["revision"] == a1
    assert there["missing"] is False and there["id"] == b1
    assert other.get(g)["sources"][0]["sha256"] == gone["sha256"]  # the frozen identity is kept, not rewritten
    snapshot = mv.MemoryViewSnapshot(spec=mv.ROLE_DEFAULTS["integrator"], store_status={"state": "active"}, frontier={},
                                     story=tuple(mv._story_pages(other, lambda room, sample=None: f"room {room}", {})),
                                     legacy_blocks={"accounts": 1, "accounts_shown": 1})
    story = mv.render_story(snapshot)
    assert f"- source: page {a1} of room {alpha}, revision {a1}; not in this chronicle: its words are not here" in story
    assert "Alpha asked again" not in story and G_TEXT in story
    assert mi.record_period(other, other.get(g), {}).span["incomplete"] is True
    read = _memory_read(_ctx(copy), node_id=g)
    assert f"source page {a1} (room {alpha}): revision {a1} used; NOT in this chronicle: its words cannot be read here" in read


# --- D. an account that cites an undecided draft, rejected afterwards -----------------------------------

def test_rejecting_a_cited_draft_follows_local_authority_and_the_account_shows_it(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    assert next(r for r in store.room_records(beta) if r["id"] == h1)["status"] == "draft"
    synthetic = store.publish_account(room_id="1", text="A synthetic account leaning on a draft.", sources=[b1, h1],
                                      author=shared.MIND).record
    assert synthetic["sources"][1]["status"] == "draft"
    assert store.select_account(synthetic["id"], replaces=[h1], author=shared.MIND, reason="through it").ok
    assert "A helper's draft about Beta." not in _story(tmp_path)
    # The link blocks nothing: the draft is rejected once, with its reason, and its rows reopen.
    rejected = store.decide(h1, False, shared.MIND, "not what happened in Beta")
    assert rejected.ok and all(r["id"] != h1 for r in store.room_records(beta)) and store.sealed_row_refs(beta) == set()
    assert store.decide(h1, True, shared.MIND, "changed my mind").reason == "invalid"  # one-shot, as before
    assert store.decide(b1, False, shared.MIND, "mine").reason == "invalid"  # my own page takes no decision
    src = store.accounts()[0]["sources"][1]
    assert (src["status"], src["status_now"]) == ("draft", "rejected")
    story = _story(tmp_path)
    block = _block(story, synthetic["id"])
    assert f"- my later rejection of its source {h1} (not in this account):\n  not what happened in Beta" in block
    assert f"- source: page {h1} of Project Beta [chat_id={beta}], revision {h1}, a draft then;" in block
    assert "told through this account" not in block  # a rejected draft is no entry of the story
    read = _memory_read(_ctx(tmp_path), node_id=synthetic["id"])
    assert f"revision {h1} used, draft then; now rejected;" in read
    # An accepted draft is said so too, and an account over a draft the mind accepts stays as written.
    h2 = _page(tmp_path, beta, 14, 14, author=HELPER, text="A better draft about Beta.")
    later = store.publish_account(room_id="1", text="Over the better draft.", sources=[h2], author=shared.MIND).record["id"]
    assert store.decide(h2, True, shared.MIND, "right").ok
    assert store.select_account(later, replaces=[], author=shared.MIND, reason="shown").ok
    assert f"- its source {h2}, a draft then, I have since accepted" in _block(_story(tmp_path), later)


# --- E. new records after the account stay new; a later account replaces explicitly ------------------

def test_a_new_record_after_the_account_is_open_and_a_later_account_replaces_only_by_selection(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    assert store.select_account(g, replaces=[a1, b1], author=shared.MIND, reason="through G").ok
    a3 = _page(tmp_path, alpha, 2, 3, text="A3: Alpha's beginning, sealed after the account.")
    units = {unit.record_id: unit for unit in mi.legacy_units(store, tmp_path)}
    period = mi.record_period(store, store.get(g), units)
    assert period.span["end"] == "2026-09-03T00:03:00+00:00" and period.source == "rows"
    assert "stream_span" not in store.get(g) and "covers" not in store.get(g)  # no range, no sealing
    story = _story(tmp_path)
    assert "A3: Alpha's beginning" in story and f"page {a3}" in story  # new and whole: not told through G
    facts = mi.room_facts(store, alpha, mi.open_room_rows(tmp_path, alpha), {}, 0)
    assert facts["pages"] == 2  # A1 and A3: an account is no page of the room
    assert [pos for _a, _m, pos in mi.open_room_rows(tmp_path, alpha)] == []  # alpha's open rows: 13 sealed by A1
    # G_next replaces G and may cite A1 again; G and its sources stay reconstructible under it.
    g_next = store.publish_account(room_id="1", text=NEXT_TEXT, sources=[a1, g, a3], author=shared.MIND).record["id"]
    assert store.select_account(g_next, replaces=[g, a3], author=shared.MIND, reason="told once more").ok
    later = _story(tmp_path)
    block = _block(later, g_next)
    assert NEXT_TEXT in block and G_TEXT not in later
    assert "4 story records told through this account (including nested selections); 2026-09-01 00:02 → 2026-09-03 00:03" in block
    assert f"replaces 2: {g}, {a3};" in _memory_read(_ctx(tmp_path), node_id=g_next)
    assert f"replaces 2: {a1}, {b1};" in _memory_read(_ctx(tmp_path), node_id=g)
    assert "\nMy accounts across rooms: 2 written, 1 in the common view (the rest one memory_read away).\n" in later + "\n"
    assert store.get(g)["text"] == G_TEXT and store.get(g)["sources"][0]["revision"] == a1


# --- F. roles and views ----------------------------------------------------------------------------

def test_every_role_reads_the_same_account_and_its_own_detail(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    assert store.select_account(g, replaces=[a1, b1], author=shared.MIND, reason="through G").ok
    tasks = {"main": MAIN, "root_alpha": {"id": "rootA001", "chat_id": rooms["alpha"]},
             "wake": {"id": "wake0001", "chat_id": 1, "metadata": {"usage_category": "consciousness"}},
             "presence": {"id": "pres0001", "chat_id": 555, "metadata": {"presence": {"binding_id": "b"}}},
             "child_main": KID, "child_alpha": {**KID, "id": "kid00002", "chat_id": rooms["alpha"]}}
    stories = {name: _story(tmp_path, task) for name, task in tasks.items()}
    assert len(set(stories.values())) == 1 and G_TEXT in stories["main"]
    rooms_text = {name: mv.render_room(_snapshot(tmp_path, task)) for name, task in tasks.items()}
    for name in ("main", "wake", "child_main"):
        assert "Main started and the owner asked me to go on." in rooms_text[name], name  # B1's detail in Main's page
        assert "Alpha asked again" not in rooms_text[name], name
    for name in ("root_alpha", "child_alpha"):
        assert "Alpha asked again and I counted." in rooms_text[name] and "Main started" not in rooms_text[name], name
    nanny = _snapshot(tmp_path, NANNY)
    assert mv.render_story(nanny) == "" and "Main started and the owner asked me to go on." in mv.render_room(nanny)
    assert G_TEXT not in mv.render_room(nanny)  # a nanny carries no life account
    # Without any account the view is as before: nothing told, no accounts line.
    plain = tmp_path / "plain"
    plain.mkdir()
    _rooms(plain)
    assert "My accounts across rooms" not in _story(plain) and "told through" not in _story(plain)


# --- G. the floor ----------------------------------------------------------------------------------

def test_the_floor_takes_an_account_like_a_page_and_keeps_what_it_tells_named(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    assert store.select_account(g, replaces=[a1, "legacy-b01-r1"], author=shared.MIND, reason="through G").ok
    snapshot = _snapshot(tmp_path)
    elements = mf.floor_elements(snapshot)
    f5 = [ident for step, ident, *_rest in elements if step == "F5"]
    assert g in f5 and a1 not in f5 and set(f5) <= {g, b1, h1}  # a told record is already an address: not degradable
    f3 = {ident: whole for step, ident, whole, _short in elements if step == "F3"}
    assert alpha in f3 and "Alpha began." in f3[alpha] and "Alpha worked." in f3[alpha]  # an untold room, both records
    assert "1" not in f3  # Main's one untold record is shorter than its room line: not degradable
    assert "legacy-b01-r1" not in "".join(f3.values()) and "Main was quiet." not in "".join(f3.values())
    assert mf.fit_memory_view(snapshot, {"margin": None, "physical": None, "budget": None}) == mv.FULL_VIEW
    full = mv.render_story(snapshot)
    represented = "2 story records told through this account (including nested selections); 2026-09-02 00:00 → 2026-09-03 00:03 (includes block periods)"
    assert "Main was quiet." not in full and represented in full
    level = mv.FloorLevel((("F5", (g,)),))
    short = mv.render_story(snapshot, level)
    assert G_TEXT not in short and "### My older pages, parts and accounts, by address" in short
    label, period, _kind = _block(full, g).split("\n")[1][4:].split(" · ")  # "### <label> · <period> · account <id>"
    assert f"- {label}; {period}; account {g}; memory_read(node_id='{g}')" in short and period == "2026-09-03 00:00 → 2026-09-03 00:03"
    assert represented in short  # the horizon holds even when the account itself is addressed
    exact = _memory_read(_ctx(tmp_path), node_id=g)
    assert f"replaces 2: {a1}, legacy-b01-r1;" in exact
    assert mv.snapshot_from_json(mv.snapshot_json(snapshot)) == snapshot
    note = mf.floor_note(level, window_tokens=100_000, mode="max")
    assert "1 pages, parts or accounts of my story" in note


# --- H. rebuild and replay ---------------------------------------------------------------------------

def test_a_rebuilt_index_replays_accounts_and_selections_and_cannot_rewrite_a_basis(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    g = store.publish_account(room_id="1", text=G_TEXT, sources=[a1, b1], author=shared.MIND).record["id"]
    assert store.select_account(g, replaces=[a1], author=shared.MIND, reason="through G").ok
    assert store.correct(a1, "Counted twice.", shared.MIND, expected_sequence=store.room_head(alpha)).ok
    before = (_story(tmp_path), store.accounts(), store.get(g), store.room_head(alpha), store.sealed_row_refs(alpha))
    bytes_before = (tmp_path / "memory" / "chronicle" / "records.jsonl").read_bytes()
    store.index_path.unlink()
    assert (_story(tmp_path), store.accounts(), store.get(g), store.room_head(alpha), store.sealed_row_refs(alpha)) == before
    assert (tmp_path / "memory" / "chronicle" / "records.jsonl").read_bytes() == bytes_before
    # A record under the account's id with another basis is an identity collision, never a rewrite.
    forged = store.publish([{**{k: v for k, v in store.get(g).items() if k != "sequence"},
                             "sources": [{**store.get(g)["sources"][0], "revision": store.get(a1)["id"] + "x"}]}])
    assert forged.reason == "invalid" and "collision" in forged.detail
    assert store.get(g)["sources"][0]["revision"] == a1
    # A wake observes both records as journal changes.
    events, _current, _window = mi.memory_changes(tmp_path, None, set())
    assert events == []  # no accepted position: the current one is the baseline
    boundary = {"sequence": store.get(b1)["sequence"], "record_id": b1}
    lines = [line for _kind, _none, line in mi.memory_changes(tmp_path, boundary, set())[0]]
    assert any(line.startswith(f"- account {g} in room 1, by mind (root, task t1): is based on 2 sources") for line in lines)
    assert any(": shows the account in the common view in place of 1 record; " in line for line in lines)


# --- tools: the integrating mind writes, a child is refused ----------------------------------------------

def test_chronicle_write_publishes_and_selects_an_account_and_a_child_is_refused(tmp_path):
    rooms, alpha, beta, a1, b1, h1 = _rooms(tmp_path)
    kid = _ctx(tmp_path, "kid00001", task_metadata={"delegation_role": "subagent", "parent_task_id": "turn0001",
                                                     "root_task_id": "turn0001"})
    before = _journal(tmp_path)
    for args in ({"kind": "account", "text": G_TEXT, "sources": [a1, b1]},
                 {"kind": "selection", "target_id": "x", "replaces": [], "reason": "mine"}):
        refused = json.loads(_chronicle_write(kid, **args))
        assert refused["ok"] is False and refused["reason"] == "not_integrator" and args["kind"] in refused["detail"]
    assert _journal(tmp_path) == before
    assert "TOOL_ARG_ERROR" in _chronicle_write(_ctx(tmp_path), kind="account", text=G_TEXT)
    assert "TOOL_ARG_ERROR" in _chronicle_write(_ctx(tmp_path), kind="selection", replaces=[])
    written = _write(tmp_path, kind="account", text=G_TEXT, sources=[a1, {"id": b1, "revision": b1}])
    assert written["ok"] and written["kind"] == "account" and written["sources"] == 2 and written["room_id"] == "1"
    assert "room_head" not in written and written["revision"] == written["node_id"]
    stale = json.loads(_chronicle_write(_ctx(tmp_path), kind="account", text="x", sources=[{"id": a1, "revision": "nope"}]))
    assert stale["ok"] is False and stale["reason"] == "revision_conflict" and stale["current_revision"] == a1
    chosen = _write(tmp_path, kind="selection", target_id=written["node_id"], replaces=[a1], reason="through it")
    assert chosen["ok"] and chosen["kind"] == "selection" and chosen["replaces"] == 1 and chosen["shown"] is True
    assert "A helper's draft" in _story(tmp_path) and G_TEXT in _story(tmp_path) and "Alpha asked again and I counted." not in _story(tmp_path)
    listing = _memory_read(_ctx(tmp_path))
    assert re.search(rf"^\[account {written['node_id']}; room 1; mind \(root turn0001\); based on 2 sources, ", listing, re.M)
    read = _memory_read(_ctx(tmp_path), node_id=written["node_id"])
    assert f"selection {chosen['node_id']} by mind (root turn0001) (seq {chosen['sequence']}):\nshown=True; replaces 1: {a1}; through it" in read
    withdrawn = _write(tmp_path, kind="selection", target_id=written["node_id"], replaces=[], shown=False, reason="later")
    assert withdrawn["ok"] and withdrawn["shown"] is False and G_TEXT not in _story(tmp_path)
    # The selection record itself reads back with its target and list.
    sel = _memory_read(_ctx(tmp_path), node_id=chosen["node_id"])
    assert sel.startswith(f"[selection {chosen['node_id']}; room 1; mind (root turn0001); replaces 1; shown in the common view; target {written['node_id']};")
    assert f"account: memory_read(node_id={written['node_id']}); replaces: {a1}; shown=True" in sel
