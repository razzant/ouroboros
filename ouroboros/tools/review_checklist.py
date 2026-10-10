"""The layered change-review checklist (``docs/CHECKLISTS.md``) as the review
surfaces read it: one ``## Header`` section, the layer a subject is judged by
(`review_body_fact.layer_for`), and the fingerprint recording which rules ran.

A leaf beside ``review_prompt_text`` and ``review_file_pack``: it reads the
checklist file and nothing else, and never imports its parent ``review_helpers``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_checklist_section(section_name: str, checklist_path: Optional[Path] = None) -> str:
    """Extract one ``## Header`` section from docs/CHECKLISTS.md (the host
    repo's by default; ``checklist_path`` reads another tree's copy)."""
    checklist_path = Path(checklist_path) if checklist_path else REPO_ROOT / "docs" / "CHECKLISTS.md"
    text = checklist_path.read_text(encoding="utf-8")

    header = f"## {section_name}"
    start = text.find(header)
    if start == -1:
        raise ValueError(
            f"Section {header!r} not found in {checklist_path}"
        )

    next_header = text.find("\n## ", start + len(header))
    if next_header == -1:
        return text[start:]
    return text[start:next_header]


# The change-review checklist is layered (docs/CHECKLISTS.md): the universal
# core every reviewed change carries, and the Ouroboros body layer appended
# only when the subject IS the body (`review_body_fact.layer_for`).
CORE_CHECKLIST_SECTION = "Change Review Checklist"
BODY_CHECKLIST_SECTION = "Ouroboros Body Layer"
CHECKLIST_LAYERS = ("core", "body")
CHECKLIST_RELATIVE_PATH = "docs/CHECKLISTS.md"


def load_checklist_layers(layer: str, checklist_path: Optional[Path] = None) -> str:
    """The change-review checklist for one layer: ``core`` is the universal
    `Change Review Checklist`; ``body`` appends the `Ouroboros Body Layer` as
    the contiguous file slice (its items continue the core numbering)."""
    if layer not in CHECKLIST_LAYERS:
        raise ValueError(f"unknown checklist layer {layer!r}; expected one of {CHECKLIST_LAYERS}")
    core = load_checklist_section(CORE_CHECKLIST_SECTION, checklist_path)
    if layer == "core":
        return core
    return f"{core}\n{load_checklist_section(BODY_CHECKLIST_SECTION, checklist_path)}"


def checklist_fingerprint(layer: str, checklist_path: Optional[Path] = None) -> dict:
    """Which rules ran: ``checklist_hash`` is the sha256 of the layered text,
    ``rules_source.sha`` the git blob id of the checklist file as read from
    the install (HEAD's blob on a clean tree — the installed body's rules
    execute, D31), computed from the bytes themselves without a git call."""
    path = Path(checklist_path) if checklist_path else REPO_ROOT / "docs" / "CHECKLISTS.md"
    raw = path.read_bytes()
    blob_sha = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    text_hash = hashlib.sha256(load_checklist_layers(layer, path).encode("utf-8")).hexdigest()
    return {"checklist_hash": text_hash,
            "rules_source": {"path": CHECKLIST_RELATIVE_PATH, "sha": blob_sha}}
