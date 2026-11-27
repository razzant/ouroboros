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
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

MODULE_ROWS_PATH = "ouroboros/domains.toml#module_map"

HEADING = ("Owners of the touched paths in the installed system (`ouroboros/domains.toml` and its `[references]`), "
           "one row per path of the change; the change itself follows.")
DOMAIN_ROW = "- {path} ({change} in this change; domain {domain}: {title}; manuals: {manuals})"
SURFACE_ROW = "- {path} ({change} in this change; surface {surface}; manuals: {manuals})"
UNOWNED_ROW = "- {path} ({change} in this change; the installed map has no owner for it)"
NO_MANUALS = "none named"
UNDEFINED_TITLE = "the installed map does not define this id"
UNAVAILABLE = ("Owner rows of the touched paths were NOT delivered: the installed `ouroboros/domains.toml` is "
               "unavailable; the review's governance manifest records why.")
YIELDED = ("Owner rows of the {count} touched path(s) were NOT delivered: with them this packet exceeded the strictest "
           "configured reviewer's input limit, and they give way before any part of the change. The installed "
           "`ouroboros/domains.toml` holds them.")
YIELD_REASON = ("removed at the fit ladder's first step, to try to keep the change and its snapshots within the "
                "strictest configured reviewer's input limit")
# Git name-status letters and the words a row says; a rename or copy names its source; a path without a status says "touched".
CHANGES = {"A": "added", "D": "deleted", "M": "modified", "T": "type-changed",
           "R": "renamed from {old}", "C": "copied from {old}"}
TOUCHED = "touched"

Change = Tuple[str, str, str]  # (Git status, path, source path of a rename or copy, else "")


@dataclass(frozen=True)
class ModuleRows:
    text: str = ""
    count: int = 0
    problem: str = ""


def _change(status: str, old: str) -> str:
    return CHANGES.get(str(status or "")[:1].upper(), TOUCHED).format(old=old)


def render_rows(manifest, entries: Iterable[Change]) -> ModuleRows:
    """The block for ``entries`` against ``manifest`` (``domains``, ``modules``, ``references``): the one renderer,
    which ``contract_material`` also runs. A rename or copy is owned where its source was, else where it landed."""
    from ouroboros.code_intelligence_architecture import path_owner as _owner  # the one resolution of ownership

    lines = []
    for path, status, old in sorted((str(p).replace("\\", "/"), s, o) for s, p, o in entries):
        source = _owner(manifest, old) if old and str(status)[:1].upper() in ("R", "C") else ("", "")
        kind, key = source if source[0] else _owner(manifest, path)
        change, manuals = _change(status, old), ", ".join(manifest.references.get(key, ())) or NO_MANUALS
        if kind == "domain":
            title = manifest.domains.get(key) or UNDEFINED_TITLE
            lines.append(DOMAIN_ROW.format(path=path, change=change, domain=key, title=title, manuals=manuals))
        elif kind == "surface":
            lines.append(SURFACE_ROW.format(path=path, change=change, surface=key, manuals=manuals))
        else:
            lines.append(UNOWNED_ROW.format(path=path, change=change))
    return ModuleRows("\n".join([HEADING, *lines]), len(lines)) if lines else ModuleRows()


def module_rows(root: pathlib.Path, touched: Iterable[str], changes: Optional[Sequence[Change]] = None) -> ModuleRows:
    """One row per reviewed path: each of ``changes`` (the subject's NUL-decoded name-status) and each path of
    ``touched`` the change set does not name (a managed resolution's conflict anchor), as "touched"."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest

    try:
        manifest = load_domain_manifest(pathlib.Path(root))
    except ValueError as exc:
        return ModuleRows(problem=f"owner rows unavailable: {exc}")
    entries = list(changes or ())
    named = {path for _status, path, _old in entries}
    entries += [("", str(path), "") for path in dict.fromkeys(touched) if str(path) not in named]
    return render_rows(manifest, entries)


def contract_material() -> tuple:
    """Every fixed text this module can hand a packet seat, as the renderer itself emits it over placeholder rows
    (every row kind and change word, with none, one and two manuals, an undefined domain id), plus the
    notices (review-contract material; no I/O). The section title is added by ``navigation_material``."""
    from types import SimpleNamespace

    manuals = {"none": [], "one": ["⟨manual⟩"], "two": ["⟨manual⟩", "⟨second manual⟩"]}
    manifest = SimpleNamespace(domains={f"⟨{n} domain⟩": "⟨title⟩" for n in manuals},
                               modules={**{f"⟨{n} domain path⟩": f"⟨{n} domain⟩" for n in manuals},
                                        "⟨undefined path⟩": "⟨undefined id⟩"},
                               references={**{f"⟨{n} domain⟩": m for n, m in manuals.items()},
                                           **{f"⟨{n} surface⟩": m for n, m in manuals.items()}})
    paths = [*(f"⟨{n} domain path⟩" for n in manuals), *(f"⟨{n} surface⟩⟨path⟩" for n in manuals),
             "⟨undefined path⟩", "⟨unowned path⟩"]
    blocks = [render_rows(manifest, [(letter, path, "⟨old⟩") for path in paths]).text for letter in [*CHANGES, ""]]
    return blocks, [UNAVAILABLE, YIELDED.format(count="⟨count⟩")]


def domains_index(root: pathlib.Path) -> str:
    """``module_map domains`` for a plan seat that cannot call it; an unreadable manifest is named, never dropped."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    try:
        return "\n".join(module_map_rows(pathlib.Path(root), "domains", load_domain_manifest(root)))
    except ValueError as exc:
        return f"⚠️ OMISSION NOTE: the domain index is unavailable ({exc})."
