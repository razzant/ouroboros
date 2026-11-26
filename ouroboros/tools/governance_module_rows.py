"""Module rows a packet-only body reviewer gets for the domains its change touches.

The architecture book carries no hand-kept module tree; a reviewer whose packet is all it
sees cannot call the ``module_map`` lookup, and the touched files themselves already reach it
through the change. What the old tree also gave such a reviewer was the neighbourhood: which
domain owns a touched path, which manuals own that domain, and one line per other module of
it, so that a duplicate mechanism is visible. That neighbourhood is read from the INSTALLED
system only, the root the rules are read from, through the same readers as the lookup:
nothing of the proposed tree is loaded or parsed here (the change carries its own text).

Two blocks: the domain block (a heading, each group's head row from ``[references]``, and a
marker per touched path: touched, or new in this change) goes before the chapter sections
naming the touched files; the neighbour block (``path — lead`` rows) after them. When the
share cannot hold them, neighbour rows go first and the domain block last, and the packet
says what it lacks (``notice``). Every fixed sentence here is in ``contract_material``, so
the commit-review contract counts what a seat is handed.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable, List

MODULE_ROWS_PATH = "ouroboros/domains.toml#module_map"
NEIGHBOUR_ROWS_PATH = "ouroboros/domains.toml#module_map (neighbours)"

HEADING = ("Domains owning the touched paths, as the installed system describes them (`ouroboros/domains.toml`); "
           "the touched files themselves are in the change below.")
NEIGHBOUR_HEADING = "Other modules of those domains and folders in the installed system, each with its own lead:"
HEAD_ROW = "- ouroboros/domains.toml ({group}: {title}, {count} files; manuals: {manuals})"
TOUCHED_ROW = "- {path} (touched in this change)"
NEW_ROW = "- {path} (new in this change; its domain assignment, if any, is in the change)"
NO_LEAD_ROW = "- {path} (no docstring or header lead)"
CUT_LISTING = ("Some rows of these groups are missing or have no lead: the installed tree's listing or a file read "
               "stopped short.")
UNAVAILABLE = ("Module rows of the touched paths were NOT delivered: the installed `ouroboros/domains.toml` is "
               "unavailable; the review's governance manifest records why.")
DOMAINS_NOT_DELIVERED = ("Module rows of the touched paths were NOT delivered: the inline share was exhausted. The "
                         "installed `ouroboros/domains.toml` and each module's docstring hold them.")
NEIGHBOURS_NOT_DELIVERED = ("{count} neighbour module row(s) were NOT delivered: the inline share was exhausted after "
                            "the sections naming the touched files.")


@dataclass
class ModuleRows:
    domains: str = ""
    neighbours: str = ""
    neighbour_count: int = 0
    problem: str = ""


def module_rows(root: pathlib.Path, touched: Iterable[str]) -> ModuleRows:
    """Both blocks for ``touched``, read from the installed tree at ``root``."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map

    root = pathlib.Path(root)
    try:
        manifest = load_domain_manifest(root)
    except (ValueError, OSError) as exc:
        return ModuleRows(problem=f"module rows unavailable: {exc}")
    groups: dict = {}  # module_map argument -> touched paths it owns
    new: List[str] = []
    for path in sorted(set(touched)):
        domain = manifest.modules.get(path)
        if domain:
            groups.setdefault(domain, []).append(path)
        elif path.startswith("web/") and "/" in path:
            groups.setdefault(path.rsplit("/", 1)[0], []).append(path)
        elif path.endswith(".py") and path.startswith(("ouroboros/", "supervisor/")):
            new.append(path)
    touched_set, seen = set(touched), set()
    heads, markers, others, cut = [], [], [], False
    for arg, paths in groups.items():
        answer = module_map(root, arg, manifest)
        cut = cut or bool(answer.limits)
        listed = {entry[0] for entry in answer.entries}
        if arg in manifest.domains:
            title, manuals = manifest.domains[arg], manifest.references.get(arg, ())
        else:
            owners = [key for key in manifest.references if key not in manifest.domains
                      and (arg + "/").startswith(key.rstrip("/") + "/")]
            title, manuals = "web folder", manifest.references.get(max(owners, key=len), ()) if owners else ()
        heads.append(HEAD_ROW.format(group=arg, title=title, count=len(answer.entries),
                                     manuals=", ".join(manuals) or "none named"))
        markers += [(TOUCHED_ROW if path in listed or path in manifest.modules else NEW_ROW).format(path=path)
                    for path in paths]
        for path, _owner, lead in answer.entries:  # each path once, across overlapping groups
            if path not in touched_set and path not in seen:
                seen.add(path)
                others.append(f"- {path} — {lead}" if lead else NO_LEAD_ROW.format(path=path))
    markers += [NEW_ROW.format(path=path) for path in new]
    if not heads and not markers:
        return ModuleRows()
    domains = "\n".join([HEADING, *heads, *markers, *([CUT_LISTING] if cut else [])])
    neighbours = "\n".join([NEIGHBOUR_HEADING, *others]) if others else ""
    return ModuleRows(domains, neighbours, len(others))


def notice(rows: ModuleRows, *, domains_delivered: bool, neighbours_delivered: bool) -> str:
    """The packet's in-band statement of module evidence it does not carry; ``""`` when all arrived."""
    if rows.problem:
        return UNAVAILABLE
    if rows.domains and not domains_delivered:
        return DOMAINS_NOT_DELIVERED + (" " + NEIGHBOURS_NOT_DELIVERED.format(count=rows.neighbour_count)
                                        if rows.neighbours else "")
    if rows.neighbours and not neighbours_delivered:
        return NEIGHBOURS_NOT_DELIVERED.format(count=rows.neighbour_count)
    return ""


def contract_material() -> list:
    """Every fixed line this module can hand a seat, over placeholders (commit-review contract material; no I/O)."""
    neighbours = NEIGHBOURS_NOT_DELIVERED.format(count="⟨count⟩")
    return [HEADING, NEIGHBOUR_HEADING, CUT_LISTING, UNAVAILABLE, DOMAINS_NOT_DELIVERED, neighbours,
            DOMAINS_NOT_DELIVERED + " " + neighbours,
            HEAD_ROW.format(group="⟨group⟩", title="⟨title⟩", count="⟨count⟩", manuals="⟨manuals⟩")]


def domains_index(root: pathlib.Path) -> str:
    """``module_map domains`` for a plan seat that cannot call it; an unreadable manifest is named, never dropped."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    try:
        return "\n".join(module_map_rows(pathlib.Path(root), "domains", load_domain_manifest(root)))
    except ValueError as exc:
        return f"⚠️ OMISSION NOTE: the domain index is unavailable ({exc})."
