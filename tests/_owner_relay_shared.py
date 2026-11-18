"""A retained owner row and the matching context used for relay tests."""
from types import SimpleNamespace

from ouroboros import dialogue_store
from ouroboros.chat_chain import format_address, row_address


def source(tmp_path):
    row = {"chat_id": 1, "direction": "in", "ts": "2026-10-01T00:00:00Z", "text": "  exact\nowner words  ",
           "client_message_id": "owner-1", "attachments": []}
    dialogue_store.append(tmp_path, "chat", row)
    return row, format_address(row_address(row))


def context(tmp_path, row=None):
    from ouroboros.project_dialogue import build_owner_message_ref
    metadata = {} if row is None else {"origin_message_ref": build_owner_message_ref(
        chat_id=row["chat_id"], client_message_id=row["client_message_id"], ts=row["ts"], text=row["text"])}
    return SimpleNamespace(task_id="teller", drive_root=tmp_path, budget_drive_root=tmp_path,
                           current_chat_id=1, task_metadata=metadata, pending_events=[], event_queue=None,
                           drive_logs=lambda: tmp_path / "logs")
