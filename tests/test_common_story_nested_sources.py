"""Retained source changes survive common-account replacement without rewriting its basis."""
from __future__ import annotations

import pytest

from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import _memory_read
from tests._memory_inventory_shared import MIND
from tests.test_common_story import HELPER, KID, _block, _ctx, _journal, _page, _part, _rooms, _story


@pytest.mark.parametrize("source_kind,change_kind", [
    ("account", "correction"), ("account", "rejection"), ("account", "acceptance"),
    ("part", "correction"), ("part", "rejection"),
])
def test_selected_account_keeps_changes_inside_its_retained_sources(tmp_path, source_kind, change_kind):
    _world, alpha, beta, page, other, draft = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    target = page if change_kind == "correction" else draft
    if source_kind == "account":
        inner = store.publish_account(room_id="1", text="My first account.", sources=[target, other], author=MIND)
        assert inner.ok
        inner = inner.record["id"]
        assert store.select_account(inner, replaces=[target, other], author=MIND, reason="First choice.").ok
    elif change_kind == "correction":
        second = _page(tmp_path, alpha, 2, 3, text="Another local page.")
        inner = _part(tmp_path, alpha, [page, second], text="My local part.")
    else:
        result = store.publish_part(room_id=beta, text="A helper's local part.", member_ids=[draft], author=HELPER,
                                    expected_sequence=store.room_head(beta))
        assert result.ok
        inner = result.record["id"]
    outer = store.publish_account(room_id="1", text="My later account stays as I wrote it.", sources=[inner, other], author=MIND)
    assert outer.ok
    outer = outer.record["id"]
    original = {ident: store.get(ident) for ident in (outer, inner, target)}
    assert store.select_account(outer, replaces=[inner, other], author=MIND, reason="Tell it through this account.").ok
    assert not next(row for row in store.accounts() if row["id"] == outer)["sources"][0]["nested_changes"]
    assert "nested source" not in _memory_read(_ctx(tmp_path), node_id=outer)

    marker = f"Exact synthetic {change_kind} of the underlying source."
    if change_kind == "correction":
        event = store.correct(target, marker, MIND, expected_sequence=store.room_head(alpha))
    else:
        if source_kind == "part":
            # Follow existing rejection authority: release the helper part before rejecting its member.
            assert store.decide(inner, False, MIND, "The helper part is no longer acting.").ok
        event = store.decide(target, change_kind == "acceptance", MIND, marker)
    assert event.ok
    event = event.record
    journal = _journal(tmp_path)
    selected = _story(tmp_path)
    block = _block(selected, outer)
    document = _memory_read(_ctx(tmp_path), node_id=outer)
    for text in (block, document):
        assert marker in text and f"{change_kind} of nested source page {target}" in text
        assert f"via {inner}" in text and event["id"] in text and event["ts"] in text
        assert f"memory_read(node_id='{event['id']}')" in text and "root, task t1" in text
        assert "original account text unchanged" in text
        if source_kind == "account":
            assert f"cited revision {target}" in text
        else:
            assert "member revision was not recorded" in text
            assert f"cited revision {target}" not in text
    assert _story(tmp_path, KID) == selected
    assert _journal(tmp_path) == journal  # all interpretations and renderings are reads
    assert all(store.get(ident) == record for ident, record in original.items())
    assert marker in _memory_read(_ctx(tmp_path), node_id=target)
    # Replaying the disposable index recovers the same nested facts from unchanged journal authority.
    store.index_path.unlink()
    assert _story(tmp_path) == selected and _journal(tmp_path) == journal
    if source_kind == "account":
        assert store.select_account(outer, replaces=[], shown=False, author=MIND, reason="Read the earlier account.").ok
        direct = _story(tmp_path)
        assert marker in direct if change_kind != "acceptance" else "I have since accepted" in direct


def test_nested_graph_uses_each_frozen_revision_and_keeps_many_to_many_and_local_folding(tmp_path):
    _world, alpha, _beta, page, other, _draft = _rooms(tmp_path)
    store = ChronicleStore(tmp_path)
    first = store.correct(page, "Correction already read by the first account.", MIND,
                          expected_sequence=store.room_head(alpha))
    assert first.ok
    first = first.record
    inner = store.publish_account(room_id="1", text="First account after correction.", sources=[page, other], author=MIND)
    assert inner.ok
    inner = inner.record["id"]
    outer = store.publish_account(room_id="1", text="Account over the first.", sources=[inner], author=MIND).record["id"]
    assert store.select_account(outer, replaces=[page, other, inner], author=MIND, reason="Common story.").ok
    assert "nested source" not in _memory_read(_ctx(tmp_path), node_id=outer)
    last = store.correct(page, "Later correction must stay visible across two account edges.", MIND,
                         expected_sequence=store.room_head(alpha), expected_revision=first["id"])
    assert last.ok
    last = last.record
    peer = store.publish_account(room_id="1", text="Independent interpretation of the old original.",
                                 sources=[{"id": page, "revision": page}, other], author=MIND)
    assert peer.ok  # citing a source is not exclusive ownership
    second = _page(tmp_path, alpha, 2, 3)
    part = _part(tmp_path, alpha, [page, second])  # still free to fold locally after both accounts
    assert next(row for row in store.pages_of_room(alpha) if row["id"] == page)["folded_into"] == part
    before = _journal(tmp_path)
    shown = _block(_story(tmp_path), outer)
    read = _memory_read(_ctx(tmp_path), node_id=outer)
    for text in (shown, read):
        assert last["text"] in text and last["id"] in text
        assert first["text"] not in text
        assert f"cited revision {first['id']}" in text
    assert store.get(inner)["sources"][0]["revision"] == first["id"]
    assert store.get(peer.record["id"])["sources"][0]["revision"] == page
    assert _journal(tmp_path) == before
