"""Reviewed connector marks reduced to bounded monochrome geometry.

The host never inlines an author's SVG document. This parser rebuilds an SVG
from geometry alone; an unsupported element/attribute is an author-facing
repair warning. Bytes and element limits bound a small UI asset, not behavior.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from typing import Any

MAX_ICON_BYTES = 8192
MAX_ICON_ELEMENTS = 128
HOST_ICON_NAMES = (
    "sparkle", "mic", "volume", "list", "plus", "template", "gauge", "check", "book",
    "chat", "projects", "files", "skills", "widgets", "dashboard", "settings",
    "message-circle", "hash", "mail", "network", "inbox", "plug", "globe", "bot", "webhook",
)
_GEOMETRY = {
    "svg": {"viewBox", "width", "height"}, "g": set(), "path": {"d"},
    "circle": {"cx", "cy", "r"}, "ellipse": {"cx", "cy", "rx", "ry"},
    "line": {"x1", "y1", "x2", "y2"}, "polyline": {"points"}, "polygon": {"points"},
    "rect": {"x", "y", "width", "height", "rx", "ry"},
}
_PAINT = {"fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "fill-rule", "clip-rule"}
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_NUMBERS = re.compile(rf"\s*{_NUMBER}(?:[\s,]+{_NUMBER})*\s*")
_PATH = re.compile(rf"(?:[MmZzLlHhVvCcSsQqTtAa\s,]|{_NUMBER})+")
_TRANSFORM = re.compile(rf"(?:\s*(?:matrix|translate|scale|rotate|skewX|skewY)\(\s*{_NUMBER}(?:[\s,]+{_NUMBER})*\s*\)[\s,]*)+")


def validate_icon(value: Any) -> str:
    """A host glyph or the sole reviewed asset path; never a URL or markup."""
    if not isinstance(value, str) or value not in ("", "ui/icon.svg", *HOST_ICON_NAMES):
        raise ValueError("icon must be a host icon name or ui/icon.svg")
    return value


def _attribute(name: str, value: str) -> str:
    if name in ("fill", "stroke"):
        if value not in ("none", "currentColor"):
            raise ValueError("icon fill/stroke must be currentColor or none")
    elif name in ("fill-rule", "clip-rule"):
        if value not in ("evenodd", "nonzero"):
            raise ValueError("invalid icon fill rule")
    elif name in ("stroke-linecap", "stroke-linejoin"):
        if value not in ("butt", "round", "square", "miter", "bevel"):
            raise ValueError("invalid icon stroke geometry")
        return "round"
    elif name == "transform":
        if not _TRANSFORM.fullmatch(value):
            raise ValueError("invalid icon transform")
    elif name == "d":
        if not _PATH.fullmatch(value):
            raise ValueError("invalid icon path geometry")
    elif not _NUMBERS.fullmatch(value):
        raise ValueError(f"invalid icon geometry: {name}")
    if name not in _PAINT and any(not math.isfinite(float(n)) for n in re.findall(_NUMBER, value)):
        raise ValueError("icon coordinates must be finite")
    return "2" if name == "stroke-width" else value


def sanitize_svg(source: bytes | str) -> str:
    """Rebuild a geometry-only SVG or raise ValueError (no partial active content)."""
    raw = source.encode("utf-8") if isinstance(source, str) else source
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_ICON_BYTES or b"<!" in raw:
        raise ValueError("icon must be at most 8 KB of SVG geometry, without declarations")
    try:
        original = ET.fromstring(raw.decode("utf-8"))
    except (UnicodeError, ET.ParseError) as exc:
        raise ValueError("icon must be valid UTF-8 SVG") from exc
    count = 0

    def rebuild(node, depth=0):
        nonlocal count
        count += 1
        tag = node.tag.removeprefix("{http://www.w3.org/2000/svg}")
        if tag not in _GEOMETRY or (tag == "svg") != (depth == 0):
            raise ValueError("icon contains non-geometry elements")
        if count > MAX_ICON_ELEMENTS or depth > 16:
            raise ValueError("icon geometry exceeds element/depth limits")
        if (node.text or "").strip() or (node.tail or "").strip():
            raise ValueError("icon may contain geometry only")
        attrs = {}
        for name, value in node.attrib.items():
            if name not in _GEOMETRY[tag] | _PAINT | {"transform"}:
                raise ValueError(f"icon attribute is not geometry: {name}")
            attrs[name] = _attribute(name, value)
        if depth == 0:
            view = attrs.get("viewBox", "0 0 24 24")
            numbers = re.findall(_NUMBER, view)
            if len(numbers) != 4 or float(numbers[2]) <= 0 or float(numbers[3]) <= 0:
                raise ValueError("icon viewBox needs four numbers and positive dimensions")
            attrs.pop("width", None)
            attrs.pop("height", None)
            attrs.update(viewBox=view, xmlns="http://www.w3.org/2000/svg")
            attrs.setdefault("fill", "currentColor")
            attrs.setdefault("stroke", "none")
            attrs.update({"stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round"})
        clean = ET.Element(tag, dict(sorted(attrs.items())))
        for child in node:
            clean.append(rebuild(child, depth + 1))
        return clean

    result = ET.tostring(rebuild(original), encoding="unicode")
    if count < 2 or len(result.encode("utf-8")) > MAX_ICON_BYTES:
        raise ValueError("icon must contain bounded geometry")
    return result


def icon_spec(value: Any, source: bytes | str | None = None) -> dict[str, str]:
    """A transport-safe mark; unavailable/invalid assets use the host fallback."""
    try:
        name = validate_icon(value)
        if name == "ui/icon.svg":
            return {"kind": "svg", "svg": sanitize_svg(source)} if source is not None else {}
        return {"kind": "host", "name": name} if name else {}
    except ValueError:
        return {}
