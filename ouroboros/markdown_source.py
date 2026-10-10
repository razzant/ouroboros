"""Byte-preserving Markdown structure for reference books and knowledge notes.

Parsing describes physical sources only. Callers decide which links are required,
what note metadata means, and whether a selected view is sufficient for a decision.
The existing tree-sitter Markdown grammars own syntax; PyYAML owns frontmatter.
"""

from __future__ import annotations

import hashlib
import html
import string
from dataclasses import dataclass
from typing import Any, Mapping

import yaml


class MarkdownSourceError(ValueError):
    """A source could not be parsed; its original bytes remain authoritative."""


def _markdown_parser(grammar: str, source_path: str) -> Any:
    """Native parsers are mutable, so each source owns its parser instances.

    Keep the optional native dependency off the context import path. Python
    already caches imported grammar bindings; no shared mutable parser is needed.
    """
    try:
        from tree_sitter_language_pack import get_parser

        return get_parser(grammar)
    except Exception as exc:
        raise MarkdownSourceError(
            f"{source_path}: Markdown parser unavailable ({grammar}: {type(exc).__name__}: {exc})"
        ) from exc


@dataclass(frozen=True)
class SourceRange:
    """Physical UTF-8 byte interval [start, end) and inclusive LF line bounds."""

    start_byte: int
    end_byte: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class MarkdownHeading:
    title: str
    level: int
    span: SourceRange
    section: SourceRange


@dataclass(frozen=True)
class MarkdownLink:
    label: str
    destination: str
    span: SourceRange


@dataclass(frozen=True)
class MarkdownSource:
    source_path: str
    raw: bytes
    sha256: str
    frontmatter: Mapping[str, Any] | None
    frontmatter_span: SourceRange | None
    body_span: SourceRange
    headings: tuple[MarkdownHeading, ...]
    paragraphs: tuple[SourceRange, ...]
    lists: tuple[SourceRange, ...]
    list_items: tuple[SourceRange, ...]
    links: tuple[MarkdownLink, ...]
    code_blocks: tuple[SourceRange, ...]

    @property
    def text(self) -> str:
        return self.raw.decode("utf-8")

    def text_at(self, span: SourceRange) -> str:
        if not 0 <= span.start_byte <= span.end_byte <= len(self.raw):
            raise ValueError("source range is outside the Markdown source")
        return self.raw[span.start_byte:span.end_byte].decode("utf-8")

    def lines(self, start_line: int = 1, max_lines: int | None = None) -> SourceRange:
        """Select physical LF lines, without inventing a composed-book address."""
        starts = [0] + [i + 1 for i, byte in enumerate(self.raw) if byte == 10]
        total = len(starts) - int(bool(self.raw) and self.raw.endswith(b"\n"))
        if isinstance(start_line, bool) or start_line < 1 or start_line > total:
            raise ValueError("start_line is outside the Markdown source")
        if max_lines is not None and (isinstance(max_lines, bool) or max_lines < 1):
            raise ValueError("max_lines must be positive")
        last = total if max_lines is None else min(total, start_line + max_lines - 1)
        end = starts[last] if last < len(starts) else len(self.raw)
        return _range(self.raw, starts[start_line - 1], end)


def _range(raw: bytes, start: int, end: int) -> SourceRange:
    first = raw.count(b"\n", 0, start) + 1
    last = raw.count(b"\n", 0, max(start, end - 1)) + 1
    return SourceRange(start, end, first, last)


def _frontmatter(raw: bytes) -> tuple[Mapping[str, Any] | None, int]:
    lines = raw.split(b"\n")
    if not lines or lines[0].rstrip(b"\r") != b"---":
        return None, 0
    offset = len(lines[0]) + 1
    for line in lines[1:]:
        if line.rstrip(b"\r") in (b"---", b"..."):
            value = yaml.safe_load(raw[len(lines[0]) + 1:offset].decode("utf-8"))
            if value is not None and not isinstance(value, dict):
                raise ValueError("Markdown frontmatter must be a YAML mapping")
            return value or {}, min(len(raw), offset + len(line) + 1)
        offset += len(line) + 1
    # An opening thematic break without a closing delimiter is ordinary Markdown.
    return None, 0


def _markdown_value(value: str) -> str:
    """Decode Markdown destination escapes without normalizing physical paths."""
    out: list[str] = []
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 1 < len(value) and value[index + 1] in string.punctuation:
            index += 1
        out.append(value[index])
        index += 1
    return html.unescape("".join(out))


def _destination(node: Any) -> str:
    value = node.text.decode("utf-8")
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return _markdown_value(value)


def _heading_title(node: Any) -> str:
    content = node.child_by_field_name("heading_content")
    if content is None:
        content = next((c for c in node.named_children if c.type in ("inline", "paragraph")), None)
    return content.text.decode("utf-8").strip() if content else ""


def parse_markdown_navigation(raw: bytes, source_path: str) -> tuple[Mapping[str, Any], str | None]:
    """Read frontmatter and title without building links or physical ranges.

    The same YAML reader and block grammar own syntax. An authored title needs
    no body parse; otherwise visit only as far as the first real heading. This
    projection is not a complete MarkdownSource or evidence of its links.
    """
    try:
        raw.decode("utf-8")
        metadata, body_start = _frontmatter(raw)
        metadata = metadata or {}
        # Missing native grammars must keep the same visible source gap as a
        # complete read, including for a titled archived note.
        block_parser = _markdown_parser("markdown", source_path)
        _markdown_parser("markdown_inline", source_path)
        title = metadata.get("title")
        if isinstance(title, str) and title.strip():
            return metadata, title
        root = block_parser.parse(raw[body_start:]).root_node

        def first_heading(node: Any) -> str | None:
            if node.type in ("atx_heading", "setext_heading"):
                return _heading_title(node)
            for child in node.named_children:
                title = first_heading(child)
                if title is not None:
                    return title
            return None

        return metadata, first_heading(root)
    except MarkdownSourceError:
        raise
    except (UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise MarkdownSourceError(f"{source_path}: {type(exc).__name__}: {exc}") from exc


def parse_markdown_source(raw: bytes, source_path: str) -> MarkdownSource:
    """Parse supplied bytes, including exact Git/index bytes, without any I/O."""
    try:
        return _parse_markdown_source(raw, source_path)
    except MarkdownSourceError:
        raise
    except (UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise MarkdownSourceError(f"{source_path}: {type(exc).__name__}: {exc}") from exc


def _parse_markdown_source(raw: bytes, source_path: str) -> MarkdownSource:
    raw.decode("utf-8")  # Strict decoding; no replacement bytes or newline rewrite.
    frontmatter, body_start = _frontmatter(raw)
    block_parser = _markdown_parser("markdown", source_path)
    inline_parser = _markdown_parser("markdown_inline", source_path)
    root = block_parser.parse(raw[body_start:]).root_node
    nodes: list[Any] = []

    def walk(node: Any) -> None:
        nodes.append(node)
        for child in node.named_children:
            walk(child)

    walk(root)

    def span(node: Any) -> SourceRange:
        return _range(raw, body_start + node.start_byte, body_start + node.end_byte)

    def label(value: bytes) -> str:
        return " ".join(_markdown_value(value.decode("utf-8").strip("[]")).split()).casefold()

    definitions: dict[str, str] = {}
    for node in nodes:
        if node.type != "link_reference_definition":
            continue
        children = {child.type: child for child in node.named_children}
        if "link_label" in children and "link_destination" in children:
            definitions.setdefault(label(children["link_label"].text),
                                   _destination(children["link_destination"]))

    headings: list[MarkdownHeading] = []
    links: list[MarkdownLink] = []
    for node in nodes:
        if node.type in ("atx_heading", "setext_heading"):
            marker = next((c.type for c in node.named_children if "marker" in c.type or "underline" in c.type), "")
            level = int(marker[5]) if marker.startswith("atx_h") else (1 if "h1" in marker else 2)
            headings.append(MarkdownHeading(
                _heading_title(node), level,
                span(node), span(node.parent) if node.parent.type == "section" else span(node),
            ))
        if node.type != "inline":
            continue
        base = body_start + node.start_byte

        def inline_links(part: Any) -> None:
            if part.type == "image":
                return
            if part.type in ("inline_link", "full_reference_link", "collapsed_reference_link", "shortcut_link"):
                children = {child.type: child for child in part.named_children}
                text_node = children.get("link_text")
                label_node = children.get("link_label") or text_node
                destination = children.get("link_destination")
                target = (_destination(destination) if destination else
                          definitions.get(label(label_node.text), "") if label_node else "")
                if target:
                    links.append(MarkdownLink(
                        text_node.text.decode("utf-8") if text_node else "",
                        target, _range(raw, base + part.start_byte, base + part.end_byte),
                    ))
                return
            for child in part.named_children:
                inline_links(child)

        inline_links(inline_parser.parse(node.text).root_node)
    return MarkdownSource(
        str(source_path), raw, hashlib.sha256(raw).hexdigest(), frontmatter,
        _range(raw, 0, body_start) if body_start else None, _range(raw, body_start, len(raw)),
        tuple(headings), tuple(span(n) for n in nodes if n.type == "paragraph"),
        tuple(span(n) for n in nodes if n.type == "list"),
        tuple(span(n) for n in nodes if n.type == "list_item"), tuple(links),
        tuple(span(n) for n in nodes if n.type == "code_fence_content"),
    )
