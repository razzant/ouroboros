"""Retained source changes survive common-account replacement without rewriting its basis."""
from __future__ import annotations

import re

import pytest

from ouroboros import memory_view as mv
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.tools.chronicle import _memory_read
from tests._memory_inventory_shared import MIND
from tests.test_common_story import HELPER, KID, _block, _ctx, _journal, _page, _part, _rooms, _snapshot, _story


@pytest.mark.parametrize("source_kind,change_kind", [
    ("account", "correction"), ("account", "rejection"), ("account", "acceptance"), ("part", "correction"),
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
        assert "root, task t1" in text and "original account text unchanged" in text
        if source_kind == "account":
            assert f"cited revision {target}" in text
        else:
            assert "member revision was not recorded" in text
            assert f"cited revision {target}" not in text
    # The reader prints each change's own reader; the story states the readers once and prints the id once.
    assert f"memory_read(node_id='{event['id']}')" in document and f"memory_read(node_id='{event['id']}')" not in block
    assert block.count(event["id"]) == 1 and "memory_read(node_id='<change id>') reads a change" in block
    assert _story(tmp_path, KID) == selected
    assert _journal(tmp_path) == journal  # all interpretations and renderings are reads
    assert all(store.get(ident) == record for ident, record in original.items())
    assert marker in _memory_read(_ctx(tmp_path), node_id=target)
    from ouroboros.memory_form_facts import ordinary_composition

    parts = ordinary_composition(_snapshot(tmp_path))["blocks"]["story"]["parts"]
    account = next(p for p in parts if p["kind"] == "account" and p["sources"][0]["id"] == outer)
    nested = [c for c in account.get("changes", ()) if c.get("via")]
    assert [(c["kind"], c["id"], c["target"], c["via"]) for c in nested] == [(change_kind, event["id"], target, [inner])]
    assert nested[0]["read"]["arguments"] == {"node_id": event["id"], "revision": event["id"]}
    assert "TOOL_ARG_ERROR" not in _memory_read(_ctx(tmp_path), **nested[0]["read"]["arguments"])
    if change_kind == "correction":
        assert nested[0]["target_read"]["arguments"] == {"node_id": target, "revision": event["id"]}
        assert marker in _memory_read(_ctx(tmp_path), **nested[0]["target_read"]["arguments"])
    else:
        assert nested[0]["target_navigation"]["arguments"] == {"node_id": target}
    cited = nested[0]["cited_source"]  # the frozen revision the rendered line names, beside the change's reader
    if source_kind == "account":
        assert cited["read"]["arguments"] == {"node_id": target, "revision": target}
        if change_kind == "correction":  # the cited revision, not the later change target_read opens
            assert marker not in _memory_read(_ctx(tmp_path), **cited["read"]["arguments"])
    else:
        assert cited["navigation"]["arguments"] == {"node_id": target} and cited["read_gap"]
    assert marker not in str(account)
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
    from ouroboros.memory_form_facts import ordinary_composition

    parts = ordinary_composition(_snapshot(tmp_path))["blocks"]["story"]["parts"]
    mapped = next(p for p in parts if p["kind"] == "account" and p["sources"][0]["id"] == outer)
    entry = next(c for c in mapped.get("changes", ()) if c.get("via") and c["id"] == last["id"])
    assert entry["target_read"]["arguments"] == {"node_id": page, "revision": last["id"]}
    assert entry["cited_source"]["read"]["arguments"] == {"node_id": page, "revision": first["id"]}
    read = _memory_read(_ctx(tmp_path), **entry["cited_source"]["read"]["arguments"])
    assert first["text"] in read and last["text"] not in read  # the frozen revision A, not the later change B
    assert _journal(tmp_path) == before


def nested_world(root):
    """Accounts nested five deep over pages, a part and a helper's draft, then signed changes after every citation:
    corrections of cited pages (a cited revision), of a part's member (no recorded member revision) and my
    acceptance of the draft. ``{name: id}``; the corrections under ``fixes``."""
    _world, alpha, _beta, a1, b1, h1 = _rooms(root)
    store = ChronicleStore(root)
    p2 = _page(root, alpha, 2, 3, text="Alpha's first question, answered.")
    part = _part(root, alpha, [a1, p2], text="Alpha, told as one part.")

    def account(text, sources):
        result = store.publish_account(room_id="1", text=text, sources=sources, author=MIND)
        assert result.ok, result
        return result.record["id"]

    acc1 = account("How Alpha and Main met.", [a1, b1])
    acc2 = account("Alpha as a part, Beta's draft beside it.", [part, h1, a1])
    acc3 = account("Both tellings at once.", [acc1, acc2])
    acc4 = account("The same, a season later.", [acc3])
    acc5 = account("Again, with Main's page read anew.", [acc4, b1])
    outer = account("The whole arc, told last.", [acc5])
    assert store.select_account(outer, replaces=[acc5], author=MIND, reason="The latest telling.").ok
    fixes = []
    for target, words in ((a1, "Alpha counted twice, not once."), (b1, "The owner asked twice."),
                          (a1, "Alpha's count was eleven.\nThe second line of that correction."),
                          (p2, "The first question was about trust."), (b1, "Main's page: the owner's word was 'go on'."),
                          (a1, "Alpha: the last correction.")):
        revision = store.interpret([store.get(target)])[0]["revision"]
        result = store.correct(target, words, MIND, expected_revision=None if revision == target else revision)
        assert result.ok, result
        fixes.append(result.record["id"])
    assert store.decide(h1, True, MIND, "The helper's draft holds.").ok
    return {"outer": outer, "acc5": acc5, "a1": a1, "b1": b1, "h1": h1, "p2": p2, "part": part, "fixes": fixes}


_CHANGE = re.compile(r"- (?P<kind>correction|acceptance|rejection) of nested source (?P<source_kind>\S+) (?P<source>\S+) "
                     r"\(room (?P<room>[^)]*)\) via (?P<chain>.+?); (?P<basis>cited revision (?P<revision>\S+); later "
                     r"change not in that source account|part member; member revision was not recorded); "
                     r"(?P<change>\S+) by mind \((?P<signer>.*)\) at (?P<ts>.+)")


def _full_form(story_lines):
    """The exact reader's lines rebuilt from the story's lines alone, by the readers and chain rule its first line
    states: what the story prints once each must give back every id, text, path and revision fact."""
    head, rest, full, above = story_lines[0], story_lines[1:], [], []
    assert "original account text unchanged" in head and "(first N as above)" in head
    for line, text in zip(rest[::2], rest[1::2]):
        found = _CHANGE.fullmatch(line)
        assert found, line
        chain = found["chain"].split(" -> ")
        if chain[0].startswith("(first "):
            chain = above[:int(chain[0].split()[1])] + chain[1:]
        above = chain
        event = "correction" if found["kind"] == "correction" else "decision"
        revision = f", revision='{found['revision']}'" if found["revision"] else ""
        full += [f"- {found['kind']} of nested source {found['source_kind']} {found['source']} (room {found['room']}) via "
                 f"{' -> '.join(chain)}; {found['basis']}; original account text unchanged",
                 f"  {event} {found['change']} by mind ({found['signer']}) at {found['ts']}; "
                 f"memory_read(node_id='{found['change']}')", text, f"  source: memory_read(node_id='{found['source']}'{revision})"]
    return full


def test_an_accounts_nested_changes_print_each_id_once_and_give_back_the_readers_lines(tmp_path):
    """Both directions (K7 b): the story prints every nested change of a shown account with its id once, the shared
    chain prefix by count and the readers once; rebuilt by the rule it states, that form is the exact account reader's
    form line for line, and the reader still prints it whole. Every correction id, text, chain and revision fact
    reaches the story, and the story is shorter."""
    from ouroboros.memory_view_account import source_change_lines, story_change_lines
    from ouroboros.tools.chronicle import _memory_read

    ids = nested_world(tmp_path)
    story, document = _story(tmp_path), _memory_read(_ctx(tmp_path), node_id=ids["outer"])
    block = _block(story, ids["outer"])
    shown = next(entry for entry in _snapshot(tmp_path).story if entry["id"] == ids["outer"])
    sources = [src for src in shown["sources"] if src.get("nested_changes")]
    assert sources and sum(len(src["nested_changes"]) for src in sources) >= 12
    bases = {change.get("revision") is None for src in sources for change in src["nested_changes"]}
    assert bases == {True, False}  # cited revisions and part members alike
    assert max(len(change["via"]) for src in sources for change in src["nested_changes"]) >= 5
    for src in sources:
        compact, full = story_change_lines(src), source_change_lines(src)
        assert _full_form(compact) == full
        assert "\n".join(compact) in block and "\n".join(full) in document and "\n".join(full) not in block
        for line in compact[1::2]:  # each change line names its change and its source once
            found = _CHANGE.fullmatch(line)
            assert line.count(found["change"]) == 1 and line.replace(found["basis"], "").count(found["source"]) == 1
    for fix in ids["fixes"]:
        record = ChronicleStore(tmp_path).get(fix)
        assert fix in block and record["text"].split("\n")[0] in block
        assert f"memory_read(node_id='{fix}')" in document and f"memory_read(node_id='{fix}')" not in block
    assert "The helper's draft holds." in block and "part member; member revision was not recorded" in block
    assert mv.json_chars(block) < mv.json_chars("\n".join(line for src in sources for line in source_change_lines(src)))
