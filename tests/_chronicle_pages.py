"""Shared synthetic chronicle page publication."""
from ouroboros import chat_chain
from ouroboros.chronicle_store import ChronicleStore
from tests import _memory_inventory_shared as shared


def _page(root, room, first, last, *, author=shared.MIND, text=None):
    from ouroboros.tools.chronicle import page_covers

    addresses = {pos: address for address, _row, pos in chat_chain.iter_rows(root)}
    covers = page_covers(root, room, from_addr=addresses[first], to_addr=addresses[last])["covers"]
    result = ChronicleStore(root).publish_page(room_id=room, text=text or f"Page {room} {first}-{last}.", covers=covers,
                                               author=author)
    assert result.ok, result
    return result.record["id"]
