"""Keep host facts inline when a known physical window needs their details by address.

Producers supply the operative view; the floor never interprets authored prose or
decides which authority matters. Unknown sections stay whole. Captured bytes are
retained before a pointer is published, under an immutable content address in the
actor's runtime data root. No owner target or working margin runs this ladder; a live
call, whose session window always fills, addresses every section that has an operative
view from its first request (``fit_host_sections(operative=True)``).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import pathlib
import re
import tempfile
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Mapping, Optional

from ouroboros.context_layout import TIER0_ALWAYS_FULL
from ouroboros.utils import estimate_tokens

log = logging.getLogger(__name__)
PROTECTED = TIER0_ALWAYS_FULL | {"task_contract_discipline", "task_input", "task_authority"}
LADDER = ("health_invariants", "runtime", "review_records", "registry", "installed_skills",
          "drive_state", "shared_understanding")


@dataclass(frozen=True, eq=False, repr=False)
class HostSection(str):
    """A producer's text plus its explicit window view, still usable by string callers.

    The string value is exactly ``full_text``: legacy joins, cache blocks and
    public builders keep their bytes. The frozen fields survive in ContextCore
    for fitting; plain strings enter the capture with no operative view.
    ``reader`` names the original source; relocation always reads the retained
    capture with read_file, so a later change at that source cannot change it.
    """

    section_id: str
    full_text: str
    operative_view: Optional[str] = None
    reader: str = ""

    def __new__(cls, section_id: str, full_text: str, operative_view: Optional[str] = None, reader: str = ""):
        return str.__new__(cls, full_text)

    def __getnewargs__(self):
        return self.section_id, self.full_text, self.operative_view, self.reader


def capture_sections(parts: Iterable[str], *, prefix: str) -> tuple[HostSection, ...]:
    """Keep structural section boundaries without guessing an unknown producer's meaning."""
    return tuple(part if isinstance(part, HostSection) else HostSection(f"{prefix}_{i}", part)
                 for i, part in enumerate(parts))


def labelled_section(section: HostSection, label: Callable[[str], str]) -> HostSection:
    """Apply the same snapshot label to both forms without losing the captured metadata."""
    return replace(section, full_text=label(section.full_text),
                   operative_view=None if section.operative_view is None else label(section.operative_view))


def _component(value: str) -> str:
    # Task ids are usually already filename components. Unusual ids retain a
    # deterministic identity without introducing traversal or an oversized name.
    return (value if re.fullmatch(r"[A-Za-z0-9_-]{1,120}", value)
            else hashlib.sha256(value.encode("utf-8")).hexdigest())


def section_address(task_id: str, section: HostSection) -> str:
    digest = hashlib.sha256(section.full_text.encode("utf-8")).hexdigest()
    return f"state/context_views/{_component(task_id)}/{digest}/{_component(section.section_id)}.md"


def _retain(root: pathlib.Path, path: str, text: str) -> None:
    """Publish complete bytes atomically without ever replacing an existing capture."""
    target, raw = root / path, text.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError("context_section_capture_mismatch")
        return
    fd, temporary = tempfile.mkstemp(prefix=".capture-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != raw:
                raise ValueError("context_section_capture_mismatch")
    finally:
        os.unlink(temporary)


def relocated_text(section: HostSection, reader: str, *, call: bool = False) -> str:
    """What a request shows of a relocated section: its operative view, why, and the read of its retained bytes."""
    cause = "from this call's view" if call else "for the physical window"
    return (section.operative_view + "\n\n"
            + f"[Host section details omitted {cause}; retained full text: {reader}.]")


def _tokens(text: str) -> int:
    return estimate_tokens(json.dumps(text, ensure_ascii=False))


def fit_host_sections(
    sections: Iterable[HostSection], *, root: str, task_id: str,
    measure: Callable[[Mapping[str, str]], int], window_tokens: int, known_window: bool,
    output_reserve: int, ratio: float, kept: Iterable[Mapping] = (), operative: bool = False,
) -> tuple[dict[str, str], list[dict]]:
    """Address details in producer order only while the complete request exceeds its window.

    ``measure`` includes the chosen memory view, schemas and actual tail. It is
    recomputed after each substitution, so memory subsequently fits the smaller
    fixed part from the same capture. Previously shown addresses stay addressed;
    unknown windows and owner targets never create new ones. ``operative`` (a live
    call, whose session window always fills) addresses every section that has an
    operative view from the first request. A retention failure keeps the complete
    section inline, never a pointer to unavailable bytes.
    """
    addressed: dict[str, str] = {}
    receipts: list[dict] = []
    previous = {row["section_id"] for row in kept}
    order = {name: i for i, name in enumerate(LADDER)}
    # Restore every shown address before measuring pressure for any new step.
    candidates = sorted(sections, key=lambda part: (part.section_id not in previous,
                                                    order.get(part.section_id, len(order))))
    for section in candidates:
        if section.section_id in PROTECTED or section.operative_view is None or not root:
            continue
        carried = section.section_id in previous
        if not carried and not operative and (not known_window or window_tokens <= 0
                                              or math.ceil(measure(addressed) * ratio) + output_reserve <= window_tokens):
            continue
        path = section_address(task_id, section)
        reader = f"read_file(root='runtime_data', path={path!r})"
        short = relocated_text(section, reader, call=operative)
        saved = _tokens(section.full_text) - _tokens(short)
        if len(short) >= len(section.full_text) or saved <= 0:
            continue
        try:
            _retain(pathlib.Path(root), path, section.full_text)
        except (OSError, ValueError):
            log.warning("Host section capture unavailable; keeping %s whole", section.section_id, exc_info=True)
            continue
        addressed[section.section_id] = short
        receipts.append({"section_id": section.section_id, "tokens_saved": saved, "path": path,
                         "sha256": hashlib.sha256(section.full_text.encode("utf-8")).hexdigest(),
                         "reader": reader})
    return addressed, receipts
