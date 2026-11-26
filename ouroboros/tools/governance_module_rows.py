"""Module rows a packet-only body reviewer gets for the domains its change touches.

The architecture book carries no hand-kept module tree: addresses and leads come from the
code through the ``module_map`` lookup (``code_intelligence_module_map``). A reviewer whose
packet is all it sees cannot call that lookup, so the packet carries its rows for the touched
domains (a Python manifest domain, or a touched ``web/`` file's directory): touched modules
first, then their neighbours, each with its own lead. ``blocks`` returns the whole block and a
touched-only form whose text says how many neighbour rows it left out and how to list them;
the caller admits the first that fits its inline share and records the rest as navigation. An
unreadable manifest is a named problem the caller records, never an empty block. A plan seat
without tools gets the compact domain index instead (``domains_index``).
"""

from __future__ import annotations

import pathlib
from typing import Iterable, List, Tuple

MODULE_ROWS_PATH = "ouroboros/domains.toml#module_map"


def _groups(root: pathlib.Path, touched: Iterable[str], manifest) -> List[Tuple[str, str, List[str]]]:
    """``(label, module_map argument, touched paths)`` per touched domain or web directory."""
    groups: dict = {}
    for path in sorted(set(touched)):
        domain = manifest.modules.get(path)
        if domain:
            groups.setdefault(domain, (f"{domain} — {manifest.domains.get(domain, '')}", domain, []))[2].append(path)
        elif path.startswith("web/") and "/" in path:
            folder = path.rsplit("/", 1)[0]
            groups.setdefault(folder, (folder + "/", folder, []))[2].append(path)
    return list(groups.values())


def blocks(root: pathlib.Path, touched: Iterable[str]) -> Tuple[str, str, int, str]:
    """``(whole block, touched-only block, neighbour rows the second leaves out, problem)``; empty
    blocks when no touched path belongs to a domain or a ``web/`` directory."""
    from ouroboros.code_intelligence_architecture import ARCHITECTURE_LIMIT_MARKER, load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    root = pathlib.Path(root)
    try:
        manifest = load_domain_manifest(root)
    except ValueError as exc:
        return "", "", 0, f"module rows unavailable: {exc}"
    whole, reduced, left_out = [], [], 0
    for label, arg, paths in _groups(root, touched, manifest):
        rows = module_map_rows(root, arg, manifest)
        limits = [row for row in rows[1:] if row.startswith(ARCHITECTURE_LIMIT_MARKER)]
        body = [row for row in rows[1:] if row not in limits and "/" in row.split(" ", 1)[0]]
        head = "\n".join(row for row in rows if row not in limits and row not in body)
        mine = [row for row in body if row.split(" ", 1)[0] in paths]
        absent = [p + " (not in the installed tree)" for p in paths
                  if not any(row.split(" ", 1)[0] == p for row in body)]
        others = [row for row in body if row not in mine]
        whole.append("\n".join([f"### {label}", head, *mine, *absent, *others, *limits]))
        left_out += len(others)
        reduced.append("\n".join([f"### {label}", head, *mine, *absent, *limits]
                                 + ([f"{len(others)} neighbour row(s) left out for this packet's inline share; "
                                     f"`module_map {arg}` lists them"] if others else [])))
    if not whole:
        return "", "", 0, ""
    lead = ("Module rows of the touched domains (own docstring/header leads; touched modules first). "
            "The architecture book keeps no module tree.")
    return "\n\n".join([lead, *whole]), "\n\n".join([lead, *reduced]), left_out, ""


def domains_index(root: pathlib.Path) -> str:
    """``module_map domains`` for a plan seat that cannot call it; an unreadable manifest is named, never dropped."""
    from ouroboros.code_intelligence_architecture import load_domain_manifest
    from ouroboros.code_intelligence_module_map import module_map_rows

    try:
        return "\n".join(module_map_rows(pathlib.Path(root), "domains", load_domain_manifest(root)))
    except ValueError as exc:
        return f"⚠️ OMISSION NOTE: the domain index is unavailable ({exc})."
