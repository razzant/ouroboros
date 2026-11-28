"""What the memory view's renderers share without importing one another (``memory_view``, ``memory_view_marks``)."""
from __future__ import annotations

from typing import Any, Mapping


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def rendered(lines, kind="framing", sources=(), form="whole", room_id=None):
    """An optional inspection sink observes these exact lines, without choosing their form."""
    return lines
