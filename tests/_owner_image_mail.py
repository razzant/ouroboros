"""An owner message with images exactly as the server routes it to a live task's mailbox (#2941).

``server_owner_routing._stage_mailbox_attachments`` stages the files in the task's own store; the row's text
is the owner's words plus that door's own ``[ATTACHMENTS]`` note, and the row carries the staged manifest.
"""

from __future__ import annotations

import struct
import zlib
from types import SimpleNamespace


def png_bytes(red: int = 255) -> bytes:
    """A valid 2x1 RGB PNG: image preparation passes it through unchanged, so pixels compare byte for byte."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes([0, red, 0, 0, 0, 0, 255]))) + chunk(b"IEND", b""))


def write_owner_images(drive, task_id, words, files, *, msg_id, client_message_id="", reply_reference=None):
    """Stage FILES for TASK_ID through the server door and write the owner's row; returns (exact text, manifest)."""
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.server_owner_routing import _stage_mailbox_attachments

    note, manifest, _report = _stage_mailbox_attachments(SimpleNamespace(DRIVE_ROOT=drive), drive, task_id, {
        "chat_attachment_uploads": [{"path": str(path)} for path in files]})
    assert [row["status"] for row in manifest] == ["staged"] * len(files), manifest
    assert write_owner_message(drive, words + note, task_id, msg_id=msg_id, client_message_id=client_message_id,
                               attachment_manifest=list(manifest), reply_reference=reply_reference)
    return words + note, [dict(row) for row in manifest]
