"""Owner rows a packet-only body reviewer gets for the paths its change touches.

The book carries no module tree and a packet seat cannot call ``module_map``; the touched files reach it with the
change. What it still needs is who owns each touched path: one row per path of the change, with its Git change in
the reviewed subject (a rename or copy names both sides and is owned where its source was) and its domain and
manuals, its surface's manuals, or the fact that the installed map has no owner for it. All of it is read from the
INSTALLED manifest; no other module of a domain is listed. The block is the first optional evidence of the packet's
fit ladder (``governance_context.yield_module_rows``), and every fixed text here is review-contract material
(``contract_material``).
"""

from __future__ import annotations

import pathlib
import string
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

MODULE_ROWS_PATH = "ouroboros/domains.toml#module_map"

HEADING = ("Owners of the touched paths in the installed system (`ouroboros/domains.toml` and its `[references]`), "
           "one row per path of the change; the change itself follows.")
DOMAIN_ROW = "- {path} ({change} in this change; domain {domain}: {title}; manuals: {manuals})"
SURFACE_ROW = "- {path} ({change} in this change; surface {surface}; manuals: {manuals})"
UNOWNED_ROW = "- {path} ({change} in this change; the installed map has no owner for it)"
NO_MANUALS = "none named"
UNAVAILABLE = ("Owner rows of the touched paths were NOT delivered: the installed `ouroboros/domains.toml` is "
               "unavailable; the review's governance manifest records why.")
YIELDED = ("Owner rows of the {count} touched path(s) were NOT delivered: with them this packet exceeded the strictest "
           "configured reviewer's input limit, and they give way before any part of the change. The installed "
           "`ouroboros/domains.toml` holds them.")
YIELD_REASON = ("gave way in the fit ladder: the packet exceeded the strictest configured reviewer's input limit, "
                "and the change's snapshots and diff context keep their place")
# Git name-status letters and the words a row says; a rename or copy names its source; a path without a status says "touched".
CHANGES = {"A": "added", "D": "deleted", "M": "modified", "T": "type-changed",
           "R": "renamed from {old}", "C": "copied from {old}"}
TOUCHED = "touched"
TEMPLATES = (HEADING, DOMAIN_ROW, SURFACE_ROW, UNOWNED_ROW, UNAVAILABLE, YIELDED)

Change = Tuple[str, str, str]  # (Git status, path, source path of a rename or copy, else "")


@dataclass(frozen=True)
class ModuleRows:
    text: str = ""
    count: int = 0
    problem: str = ""


def _change(status: str, old: str) -> str:
    return CHANGES.get(str(status or "")[:1].upper(), TOUCHED).format(old=old)


def module_rows(root: pathlib.Path, touched: Iterable[str], changes: Optional[Sequence[Change]] = None) -> ModuleRows:
    """One row per path of ``changes`` (the reviewed subject's NUL-decoded name-status), else of ``touched``."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest

    try:
        manifest = load_domain_manifest(pathlib.Path(root))
    except ValueError as exc:
        return ModuleRows(problem=f"owner rows unavailable: {exc}")
    entries = list(changes) if changes else [("", str(path), "") for path in touched]
    lines = []
    for path, status, old in sorted((str(p).replace("\\", "/"), s, o) for s, p, o in entries):
        change, domain = _change(status, old), manifest.modules.get(path) or manifest.modules.get(old)
        surfaces = [key for key in manifest.references if key not in manifest.domains and path.startswith(key)]
        if domain:
            lines.append(DOMAIN_ROW.format(path=path, change=change, domain=domain, title=manifest.domains[domain],
                                           manuals=", ".join(manifest.references.get(domain, ())) or NO_MANUALS))
        elif surfaces:
            surface = max(surfaces, key=len)
            lines.append(SURFACE_ROW.format(path=path, change=change, surface=surface,
                                            manuals=", ".join(manifest.references[surface]) or NO_MANUALS))
        else:
            lines.append(UNOWNED_ROW.format(path=path, change=change))
    return ModuleRows("\n".join([HEADING, *lines]), len(lines)) if lines else ModuleRows()


def placeholders(template: str, **fixed: str) -> str:
    """``template`` with every field a placeholder, or the given fixed word."""
    names = {name for _text, name, _spec, _conv in string.Formatter().parse(template) if name}
    return template.format(**{name: fixed.get(name, f"⟨{name}⟩") for name in names})


def contract_material() -> list:
    """Every fixed text this module can hand a packet seat, over placeholders (review-contract material; no I/O).
    A row's change word and the empty-manuals word are fixed text, so each is rendered."""
    words = [placeholders(word) for word in [*CHANGES.values(), TOUCHED]]
    rows = [placeholders(template, change=word, manuals=manuals) for template in (DOMAIN_ROW, SURFACE_ROW, UNOWNED_ROW)
            for word in words for manuals in ("⟨manuals⟩", NO_MANUALS)]
    return [*(placeholders(template) for template in TEMPLATES), *dict.fromkeys(rows)]


def domains_index(root: pathlib.Path) -> str:
    """``module_map domains`` for a plan seat that cannot call it; an unreadable manifest is named, never dropped."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    try:
        return "\n".join(module_map_rows(pathlib.Path(root), "domains", load_domain_manifest(root)))
    except ValueError as exc:
        return f"⚠️ OMISSION NOTE: the domain index is unavailable ({exc})."
