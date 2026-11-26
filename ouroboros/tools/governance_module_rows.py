"""Module rows a packet-only body reviewer gets for the files its change touches.

The architecture book carries no hand-kept module tree: addresses and leads come from the
code through the ``module_map`` lookup (``code_intelligence_module_map``). A reviewer whose
packet is all it sees cannot call that lookup, so the packet carries the rows itself, built
from structured ``(path, owner, lead)`` entries, never by re-parsing display text.

Two blocks, so they never crowd out the chapter sections that name a touched file: the
touched modules' rows (every touched manifest module or ``web/`` file, each once, with its
own lead or a note that the reviewed tree does not hold it), admitted before those sections,
and the neighbour rows (the rest of each touched domain, or the touched ``web/`` file's
own folder, each path once), admitted after them. The rows are read from the reviewed
subject's tree when the seam supplies it, else from the installed tree, and the block says
which. Whatever is not delivered is said in the packet's navigation (``notice``), and an
unreadable manifest is a named problem. A plan seat without tools gets the compact domain
index instead (``domains_index``).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Iterable

MODULE_ROWS_PATH = "ouroboros/domains.toml#module_map"
NEIGHBOUR_ROWS_PATH = "ouroboros/domains.toml#module_map (neighbours)"


@dataclass
class ModuleRows:
    touched: str = ""
    neighbours: str = ""
    neighbour_count: int = 0
    problem: str = ""
    source: str = ""


def module_rows(root: pathlib.Path, touched: Iterable[str], *, subject_tree: bool) -> ModuleRows:
    """Both blocks for the touched paths, read at ``root`` (the subject's tree when ``subject_tree``)."""
    from ouroboros.code_intelligence_architecture import ARCHITECTURE_LIMIT_MARKER, load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map, render_entry

    root = pathlib.Path(root)
    source = "the reviewed subject's tree" if subject_tree else (
        "the installed tree; this seam did not supply the subject's own tree, so a proposed lead may differ")
    try:
        manifest = load_domain_manifest(root)
    except ValueError as exc:
        return ModuleRows(problem=f"module rows unavailable: {exc}", source=source)
    groups: dict = {}  # module_map argument -> (label, touched paths)
    for path in sorted(set(touched)):
        domain = manifest.modules.get(path)
        if domain:
            groups.setdefault(domain, (f"{domain} — {manifest.domains.get(domain, '')}", []))[1].append(path)
        elif path.startswith("web/") and "/" in path:
            folder = path.rsplit("/", 1)[0]
            groups.setdefault(folder, (folder + "/", []))[1].append(path)
    mine, others, limits, seen = [], [], [], set()
    for arg, (label, paths) in groups.items():
        answer = module_map(root, arg, manifest)
        limits.extend(f"{ARCHITECTURE_LIMIT_MARKER} {note}" for note in answer.limits)
        found = {entry[0]: entry for entry in answer.entries}
        for path in paths:
            if path not in seen:
                seen.add(path)
                mine.append(render_entry(found[path]) if path in found else f"{path} (not in that tree: new or deleted)")
        # A web folder's neighbours are its own files, not its subfolders' (those are their own groups).
        group = [entry for entry in answer.entries if entry[0] not in seen
                 and (arg in manifest.domains or entry[0].rsplit("/", 1)[0] == arg)]
        if group:
            seen.update(entry[0] for entry in group)
            others.append("\n".join([f"### {label}", *(render_entry(entry) for entry in group)]))
    if not mine:
        return ModuleRows(source=source)
    lead = f"Module rows of the touched files, read from {source} (each file's own docstring/header lead)."
    touched_block = "\n".join([lead, *mine, *limits])
    neighbours = ("Neighbour modules of the touched domains and web folders, read from the same tree.\n\n"
                  + "\n\n".join(others)) if others else ""
    count = sum(block.count("\n") for block in others)
    return ModuleRows(touched_block, neighbours, count, "", source)


def notice(rows: ModuleRows, *, touched_delivered: bool, neighbours_delivered: bool) -> str:
    """The packet's in-band statement of module evidence it does not carry; ``""`` when all arrived."""
    if rows.problem:
        return f"Module rows of the touched files were NOT delivered ({rows.problem})."
    parts = []
    if rows.touched and not touched_delivered:
        parts.append("Module rows of the touched files were NOT delivered: the inline share was exhausted; "
                     "the touched files themselves are in this packet.")
    if rows.neighbours and not neighbours_delivered:
        parts.append(f"{rows.neighbour_count} neighbour module row(s) of the touched domains were NOT delivered: "
                     "the inline share was exhausted after the sections naming the touched files.")
    return " ".join(parts)


def domains_index(root: pathlib.Path) -> str:
    """``module_map domains`` for a plan seat that cannot call it; an unreadable manifest is named, never dropped."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    try:
        return "\n".join(module_map_rows(pathlib.Path(root), "domains", load_domain_manifest(root)))
    except ValueError as exc:
        return f"⚠️ OMISSION NOTE: the domain index is unavailable ({exc})."
