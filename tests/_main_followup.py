"""Main intake fixtures shared by conversation laws and the queue I/O inventory."""
from types import SimpleNamespace

import pytest

from supervisor import conversation_lane, worker_chat_lane, workers


class _Bridge:
    """The web bridge as the intake reads it: queued owner messages in, typed receipts out."""

    def __init__(self):
        self.updates, self.acks, self.offset = [], [], 0

    def get_updates(self, offset=0, timeout=1):
        return [update for update in self.updates if update["update_id"] >= offset]

    def send_routing_ack(self, chat_id, **fields):
        self.acks.append({"chat_id": chat_id, **fields})

    def broadcast(self, _payload):
        return None


def _say(bridge, ctx, text, client_message_id, *, chat_id=1, source="web", **message):
    """One owner message through the real intake (``server._process_bridge_updates``)."""
    import server

    bridge.updates.append({"update_id": len(bridge.updates) + 1, "message": {
        "chat": {"id": chat_id}, "from": {"id": 1}, "text": text, "source": source,
        "client_message_id": client_message_id, **message}})
    bridge.offset = server._process_bridge_updates(bridge, bridge.offset, ctx)


@pytest.fixture
def main(lane, monkeypatch):
    """Main's intake over the lane's live tables (routing reads them by reference), the pooled fixture row gone."""
    monkeypatch.setattr("supervisor.message_bus.log_chat", lambda *_a, **_k: None)
    workers.RUNNING.pop("owner")
    ctx = SimpleNamespace(DRIVE_ROOT=workers.DRIVE_ROOT, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          BUDGET_ROOT_FENCES={}, load_state=lambda: {"owner_id": 1, "owner_chat_id": 1},
                          update_state=lambda fn: fn({"owner_id": 1, "owner_chat_id": 1}),
                          consciousness=SimpleNamespace(), handle_chat_direct=worker_chat_lane.handle_chat_direct,
                          send_with_budget=lambda *a, **k: None)
    return SimpleNamespace(ctx=ctx, bridge=_Bridge(), root=workers.DRIVE_ROOT, lane=lane)


def _turns(text=None):
    """The lane's owner turns now (one whose words begin with ``text`` when given)."""
    rows = [meta for meta in workers.RUNNING.values() if conversation_lane.conversation_row(meta)]
    return [meta for meta in rows if text is None or str(meta["task"].get("text") or "").startswith(text)]


def _delivered(bridge, client_message_id):
    return [ack for ack in bridge.acks if ack.get("client_message_id") == client_message_id
            and ack.get("action") == "mailbox_delivery"]
