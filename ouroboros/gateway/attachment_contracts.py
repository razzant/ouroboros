"""Typed owner-attachment shapes, re-exported by the gateway facade (``contracts``).

The inbound reference a chat frame names, the task staging manifest row, and the
one attachment view every chat surface renders. Runtime validation stays at the
ingresses (``gateway.ws``, ``chat_uploads``, ``artifacts.stage_task_attachments``).
"""

from __future__ import annotations

from typing import Optional

try:
    from typing import Literal, TypedDict
except ImportError:  # pragma: no cover - Python 3.10 compatibility.
    from typing_extensions import Literal, TypedDict


class ChatAttachmentInbound(TypedDict, total=False):
    """Reference to a file stored by /api/chat/upload under data/uploads/.
    ``filename`` is its stored basename. Images reach vision models as
    native image blocks."""

    filename: str
    display_name: str
    mime: str


class AttachmentManifestEntry(TypedDict, total=False):
    """One declared task attachment after staging admission."""

    ordinal: int
    status: Literal["staged", "rejected"]
    reason: str
    label: str
    root: str
    relpath: str
    abs_path: str
    mime: str
    is_image: bool
    size: int
    sha256: str
    rule: str


class ChatAttachmentView(TypedDict, total=False):
    """One owner attachment as every chat surface shows it (``chat_uploads.attachment_view``):
    the sender's own bubble (``UploadResponse.view``), the echo and history alike. ``kind`` is
    proven from the bytes; ``url`` (``/api/files/download?upload=<id>``) exists only while
    ``available``. An unavailable attachment keeps its name and offers no action."""

    name: str
    kind: Literal["image", "video", "audio", "file"]
    mime: str
    size: Optional[int]
    available: bool
    url: str
