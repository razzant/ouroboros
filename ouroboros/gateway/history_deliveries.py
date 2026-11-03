"""The ``msg_type`` projection of a stored delivery row for ``GET /api/chat/history``.

A delivery the core put into the chat (a document, photo, video, link buttons or a
living card) is stored as a compact typed row without base64 or body; its history
row names the kind in ``msg_type`` with the fields its bubble needs. Questions keep
their own merge in ``gateway/history.py`` (their lifecycle lives in task results).
"""

from __future__ import annotations

from typing import Any, Dict

from ouroboros.gateway._helpers import coerce_int

_CARD_FIELDS = ("card_id", "kind", "title")


def delivery_fields(entry: Dict[str, Any]) -> Dict[str, Any]:
    """``msg_type`` and its fields for a stored delivery row; empty for every other row."""
    kind = entry.get("type")
    if kind == "document":
        return {"msg_type": "document", "filename": str(entry.get("filename") or "file"),
                "mime": str(entry.get("mime") or "application/octet-stream"),
                "download_url": str(entry.get("download_url") or ""),
                "download_url_compat": str(entry.get("download_url_compat") or ""),
                "caption": str(entry.get("caption") or ""),
                **({"size_bytes": coerce_int(entry.get("size_bytes"), 0)} if "size_bytes" in entry else {})}
    if kind in {"photo", "video"} and entry.get("download_url"):
        return {"msg_type": str(kind), "mime": str(entry.get("mime") or ""),
                "download_url": str(entry["download_url"]),
                "download_url_compat": str(entry.get("download_url_compat") or ""),
                "caption": str(entry.get("caption") or "")}
    if kind == "links":
        return {"msg_type": "links", "actions": list(entry.get("actions") or []), "title": str(entry.get("title") or "")}
    if kind == "card" and isinstance(entry.get("card"), dict):
        # The identity as first published; the current body and revision are read from GET /api/cards/{card_id}.
        card = entry["card"]
        return {"msg_type": "card", "card": {**{key: str(card.get(key) or "") for key in _CARD_FIELDS},
                                             "revision": coerce_int(card.get("revision"), 1)}}
    return {}


__all__ = ["delivery_fields"]
