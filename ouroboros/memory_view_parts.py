"""The memory view's line sink and mapping read: a renderer's default ``part`` returns the lines it is given, and a
captured fact that is not a mapping reads as an empty one."""
from __future__ import annotations

from typing import Any, Mapping


def mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def rendered(lines, kind="framing", sources=(), form="whole", room_id=None):
    """An optional inspection sink observes these exact lines, without choosing their form."""
    return lines
