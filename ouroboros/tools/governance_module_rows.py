"""Module rows a packet-only body reviewer gets for the domains its change touches.

The architecture book carries no hand-kept module tree; a reviewer whose packet is all it
sees cannot call the ``module_map`` lookup, and the touched files themselves reach it with
the change. What the old tree also gave such a reviewer was the neighbourhood: which domain
owns a touched path, which manuals own that domain, and one line per other module of it,
so that a duplicate mechanism is visible. That neighbourhood is read from the INSTALLED
system only, the root the rules are read from, through the same readers as the lookup:
nothing of the proposed tree is loaded or parsed here.

Every emitted path carries its group (domain id or web folder). The domain block (heading,
each group's head row with its manuals, one row per touched path labelled with the change's
own Git status when the seam supplies it) goes before the chapter sections naming the touched
files; neighbour rows, nearest to a touched file first, go after them and give way from the
far end when the share runs out. The packet says what it lacks. Every fixed text here is a
template rendered over placeholders into the review contract (``contract_material``).
"""

from __future__ import annotations

import pathlib
import string
from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional

MODULE_ROWS_PATH = "ouroboros/domains.toml#module_map"
NEIGHBOUR_ROWS_PATH = "ouroboros/domains.toml#module_map (neighbours)"

HEADING = ("Domains owning the touched paths, as the installed system describes them (`ouroboros/domains.toml`); "
           "the change below carries the touched files' changes, and the files themselves when they fit.")
HEAD_ROW = "- ouroboros/domains.toml — {group}: {title}; {count} files; manuals: {manuals}"
HEAD_ROW_TESTS = ("- ouroboros/domains.toml — {group}: {title}; {count} files, {tests} more under tests/ not "
                  "listed; manuals: {manuals}")
CHANGE_ROW = "- {path} ({group}; {change} in this change)"
UNOWNED_ROW = "- {path} ({change} in this change; the installed map has no owner for it)"
NEIGHBOUR_HEADING = ("Other modules of those domains and folders in the installed system, nearest to the change "
                     "first, each with its own lead:")
NEIGHBOUR_ROW = "- {path} — {group}: {lead}"
NO_LEAD_ROW = "- {path} ({group}; no docstring or header lead)"
UNREADABLE_ROW = "- {path} ({group}; the installed file could not be read)"
CUT_LISTING = "Some rows of these groups are missing: the installed tree's listing stopped short."
UNAVAILABLE = ("Module rows of the touched paths were NOT delivered: the installed `ouroboros/domains.toml` is "
               "unavailable; the review's governance manifest records why.")
DOMAINS_NOT_DELIVERED = ("Module rows of the touched paths were NOT delivered: the inline share was exhausted. The "
                         "installed `ouroboros/domains.toml` and each module's docstring hold them.")
NEIGHBOURS_NOT_DELIVERED = ("{missing} of {total} neighbour module row(s) were NOT delivered: the inline share was "
                            "exhausted after the sections naming the touched files; `module_map` lists them for a "
                            "reader with tools.")
# Git name-status letters and the word a row says; a seam with paths alone says "touched".
CHANGES = {"A": "added", "D": "deleted", "M": "modified", "R": "renamed", "C": "copied", "T": "type-changed"}
TOUCHED = "touched"


@dataclass
class ModuleRows:
    domains: str = ""
    neighbours: List[str] = field(default_factory=list)
    problem: str = ""


def _change(path: str, status: Optional[Mapping[str, str]]) -> str:
    letter = str((status or {}).get(path) or "")[:1].upper()
    return CHANGES.get(letter, TOUCHED)


def module_rows(root: pathlib.Path, touched: Iterable[str], *,
                status: Optional[Mapping[str, str]] = None) -> ModuleRows:
    """Both blocks for ``touched`` (``status``: path -> Git name-status letter), read from the installed tree."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map

    root = pathlib.Path(root)
    try:
        manifest = load_domain_manifest(root)
    except ValueError as exc:
        return ModuleRows(problem=f"module rows unavailable: {exc}")
    touched = sorted({str(path).replace("\\", "/") for path in touched})
    groups: dict = {}  # group (domain id or web folder) -> touched paths it owns
    unowned: List[str] = []
    for path in touched:
        domain = manifest.modules.get(path)
        if domain:
            groups.setdefault(domain, []).append(path)
        elif path.startswith("web/") and "/" in path:
            groups.setdefault(path.rsplit("/", 1)[0], []).append(path)
        elif path.endswith(".py") and path.startswith(("ouroboros/", "supervisor/")):
            unowned.append(path)
    lines, near, far, seen, cut = [HEADING], [], [], set(touched), False
    for group, paths in groups.items():
        answer = module_map(root, group, manifest)
        unreadable = {note.split(": ", 1)[1] for note in answer.limits if note.startswith("module_map_unreadable: ")}
        cut = cut or len(unreadable) < len(answer.limits)
        if group in manifest.domains:
            title, manuals = manifest.domains[group], manifest.references.get(group, ())
        else:
            owners = [key for key in manifest.references if key not in manifest.domains
                      and (group + "/").startswith(key.rstrip("/") + "/")]
            title, manuals = "web folder", manifest.references.get(max(owners, key=len), ()) if owners else ()
        head = HEAD_ROW_TESTS if answer.tests_left_out else HEAD_ROW
        lines.append(head.format(group=group, title=title, count=len(answer.entries), tests=answer.tests_left_out,
                                 manuals=", ".join(manuals) or "none named"))
        lines += [CHANGE_ROW.format(path=path, group=group, change=_change(path, status)) for path in paths]
        folders = {path.rsplit("/", 1)[0] for path in paths}
        for path, _owner, lead in answer.entries:  # each path once, across overlapping groups
            if path in seen:
                continue
            seen.add(path)
            row = (UNREADABLE_ROW if path in unreadable else NEIGHBOUR_ROW if lead else NO_LEAD_ROW).format(
                path=path, group=group, lead=lead)
            (near if path.rsplit("/", 1)[0] in folders else far).append(row)
    lines += [UNOWNED_ROW.format(path=path, change=_change(path, status)) for path in unowned]
    if len(lines) == 1:
        return ModuleRows()
    return ModuleRows("\n".join([*lines, *([CUT_LISTING] if cut else [])]), [*near, *far])


def neighbour_text(rows: List[str]) -> str:
    return "\n".join([NEIGHBOUR_HEADING, *rows]) if rows else ""


def notice(rows: ModuleRows, *, domains_delivered: bool, neighbours_shown: int) -> str:
    """The packet's in-band statement of module evidence it does not carry; ``""`` when all arrived."""
    if rows.problem:
        return UNAVAILABLE
    parts = [DOMAINS_NOT_DELIVERED] if rows.domains and not domains_delivered else []
    if neighbours_shown < len(rows.neighbours):
        parts.append(NEIGHBOURS_NOT_DELIVERED.format(missing=len(rows.neighbours) - neighbours_shown,
                                                     total=len(rows.neighbours)))
    return " ".join(parts)


TEMPLATES = (HEADING, HEAD_ROW, HEAD_ROW_TESTS, CHANGE_ROW, UNOWNED_ROW, NEIGHBOUR_HEADING, NEIGHBOUR_ROW,
             NO_LEAD_ROW, UNREADABLE_ROW, CUT_LISTING, UNAVAILABLE, DOMAINS_NOT_DELIVERED, NEIGHBOURS_NOT_DELIVERED)


def placeholders(template: str, **fixed: str) -> str:
    """``template`` with every field a placeholder, or the given fixed word."""
    names = {name for _text, name, _spec, _conv in string.Formatter().parse(template) if name}
    return template.format(**{name: fixed.get(name, f"⟨{name}⟩") for name in names})


def contract_material() -> list:
    """Every fixed text this module can hand a packet seat, over placeholders (review-contract material; no I/O).
    A row's change word is fixed text, so each one is rendered."""
    words = [*CHANGES.values(), TOUCHED]
    rows = [placeholders(template, change=word) for template in (CHANGE_ROW, UNOWNED_ROW) for word in words]
    neighbours = placeholders(NEIGHBOURS_NOT_DELIVERED)
    return [*(placeholders(template) for template in TEMPLATES if template not in (CHANGE_ROW, UNOWNED_ROW)),
            *rows, DOMAINS_NOT_DELIVERED + " " + neighbours]


def domains_index(root: pathlib.Path) -> str:
    """``module_map domains`` for a plan seat that cannot call it; an unreadable manifest is named, never dropped."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    try:
        return "\n".join(module_map_rows(pathlib.Path(root), "domains", load_domain_manifest(root)))
    except ValueError as exc:
        return f"⚠️ OMISSION NOTE: the domain index is unavailable ({exc})."
