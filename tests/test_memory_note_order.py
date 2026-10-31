"""A page of notes joins the current room head, including when folded or read by address."""
from ouroboros.chronicle_store import ChronicleStore
from ouroboros.memory_sealing import terminal_links
from tests.test_chronicle_tools import addr, chat, ctx_for, ts, write


def test_note_page_follows_room_pages_folds_with_latest_and_has_no_false_later_rows(tmp_path):
    store = ChronicleStore(tmp_path)
    store.ensure_activated()
    ctx = ctx_for(tmp_path, "writer01")
    pages = []
    for n in (1, 2):
        row = chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(n), "text": f"Words {n}"}])[0]
        reply = write(ctx, kind="page", text=f"Page {n}", covers={"from": addr(row), "to": addr(row)})
        assert reply["ok"]
        pages.append(reply["node_id"])
    chat(tmp_path, [{"chat_id": 1, "direction": "system", "ts": ts(3), "task_id": "writer01",
                     "type": "task_summary", "status": "completed", "text": "Earlier completion"}])
    note = write(ctx, kind="note", text="A remaining thought")["node_id"]
    result = write(ctx, kind="page", text="The thought settled", covers={"note_ids": [note]})
    assert result["ok"]
    page = store.get(result["node_id"])
    assert page["covers"]["stream_span"] == [2, 2]
    assert [p["id"] for p in store.pages_of_room("1")] == [*pages, page["id"]]
    # Readers of the coverage itself agree with publication metadata about which rows are later.
    assert terminal_links(tmp_path, {**page, "metadata": {}}) == []
    assert write(ctx, kind="page", text="Duplicate", covers={"note_ids": [note]})["reason"] == "already_sealed"
    refused = write(ctx, kind="part", text="Not neighbours", member_ids=[pages[0], page["id"]],
                    expected_sequence=store.room_head("1"))
    assert not refused["ok"] and refused["reason"] == "invalid" and "adjacent" in refused["detail"], refused
    folded = write(ctx, kind="part", text="The latest page and thought", member_ids=[pages[1], page["id"]],
                   expected_sequence=store.room_head("1"))
    assert folded["ok"], folded
    assert store.get(folded["node_id"])["covers"]["stream_span"] == [1, 2]
    assert "Originals remain readable by address" in folded["effect"]
    assert "next start" in folded["effect"] and "from the first changed retelling" in folded["effect"]
    assert store.get(pages[1])["text"] == "Page 2" and store.get(page["id"])["text"] == "The thought settled"
    later = chat(tmp_path, [{"chat_id": 1, "direction": "system", "ts": ts(4), "task_id": "writer01",
                            "type": "task_summary", "status": "failed", "text": "Later completion"}])[0]
    linked = "\n".join(terminal_links(tmp_path, {**page, "metadata": {}}))
    assert "Later terminal rows" in linked and "failed" in linked and "completed" not in linked
    assert addr(later) in linked


def test_note_page_uses_the_stream_head_even_without_words_in_its_room(tmp_path):
    from ouroboros.chat_chain import iter_rows

    chat(tmp_path, [{"chat_id": 1, "direction": "in", "ts": ts(n), "text": f"Other room {n}"}
                   for n in range(1, 4)])
    ctx = ctx_for(tmp_path)
    note = write(ctx, kind="note", room_id="555", text="A room without dialogue")["node_id"]
    reply = write(ctx, kind="page", room_id="555", text="Resolved", covers={"note_ids": [note]})
    head = list(iter_rows(tmp_path))[-1][2]
    assert head == 2
    assert ChronicleStore(tmp_path).get(reply["node_id"])["covers"]["stream_span"] == [head, head]
