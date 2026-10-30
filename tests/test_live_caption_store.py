"""Caption revisions use the dialogue transaction without moving history coordinates."""
import pytest

from ouroboros import dialogue_store as ds
from ouroboros.store import StoreError


def test_text_revision_preserves_coordinates_metadata_and_room_facts(tmp_path, monkeypatch):
    first = ds.append(tmp_path, 'chat', {'direction': 'out', 'chat_id': 1, 'ts': '2026-10-22T01:00:00Z',
        'text': 'Original', 'source': 'live_call', 'spoken': {'item_id': 'item', 'played_ms': 100}}, feed=True)
    following = ds.append(tmp_path, 'chat', {'direction': 'in', 'chat_id': 1, 'ts': '2026-10-22T01:00:01Z',
        'text': 'Following', 'client_message_id': 'owner-next'}, feed=True)
    before = ds.room(tmp_path, 1)
    changed = ds.revise_text(tmp_path, 'chat', first.seq, 'Corrected words.' * 100)
    after = ds.room(tmp_path, 1)
    assert changed == {**first.row, 'text': 'Corrected words.' * 100}
    assert after == {**before, 'revision': before['revision'] + 1}
    rows = list(ds.iter_rows(tmp_path, 'chat'))
    assert [row.seq for row in rows] == [first.seq, following.seq]
    assert [row.length for row in rows] == [first.length, following.length]
    assert rows[1].row == following.row and ds.stream_end(tmp_path, 'chat') == following.end
    assert ds.revise_text(tmp_path, 'chat', first.seq, changed['text']) == changed
    assert ds.room(tmp_path, 1) == after, 'an exact replay must not increment the revision'
    assert int(ds.meta(tmp_path, 'chat_text_revision')) == 1
    with monkeypatch.context() as guarded:
        def fail(*args, **kwargs):
            raise StoreError('revision update failed')
        guarded.setattr(ds, 'touch_room', fail)
        with pytest.raises(StoreError, match='revision update failed'):
            ds.revise_text(tmp_path, 'chat', first.seq, 'Never committed')
    assert ds.row_at(tmp_path, 'chat', first.seq).row == changed, 'body and room revision commit atomically'
    assert int(ds.meta(tmp_path, 'chat_text_revision')) == 1, 'a rolled-back edit must not invalidate the tail'
    with pytest.raises(ValueError, match='dialogue row'):
        ds.revise_text(tmp_path, 'chat', following.end + 1, 'Unknown')
    assert ds.room(tmp_path, 1) == after


def test_caption_revision_refreshes_a_stored_memory_address_before_the_tail_anchor(tmp_path):
    from ouroboros import chat_chain, memory_inventory
    from tests import _memory_inventory_shared as shared

    shared.world(tmp_path)
    assert memory_inventory.fill_tail(tmp_path, lambda: False)['state'] == 'done'
    before = memory_inventory.rows_after_frontier(tmp_path)
    assert len(before) > 1
    original, resolved = chat_chain.resolve_row(tmp_path, before[0][0])
    assert original is not None
    changed = ds.revise_text(tmp_path, 'chat', resolved['seq'], 'Corrected caption before the unchanged last row.')
    after = memory_inventory.rows_after_frontier(tmp_path)
    assert after[0][0]['row_sha256'] == chat_chain.source_row_id(changed)
    assert chat_chain.resolve_row(tmp_path, after[0][0])[0] == changed
    assert after[1:] == before[1:]  # the anchor alone cannot detect this edit
    assert memory_inventory.fill_tail(tmp_path, lambda: False)['state'] == 'done'
    assert memory_inventory.rows_after_frontier(tmp_path) == after
