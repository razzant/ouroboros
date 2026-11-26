"""Module map: live module addresses and their own one-line leads (``query_code`` architecture fact).

The argument is an address, matched literally: ``domains`` (each domain, its module
count and owning manuals from ``ouroboros/domains.toml`` ``[references]``), a domain id
(``D06``: its manifest modules), or a repository path prefix (``web/modules/``,
``ouroboros/tools/``, one file). A lead is the file's own words: the first line of a
Python module docstring or of a JS/TS/CSS leading header comment; other files are
listed by path alone. The map replaces a hand-kept module tree, so it never invents a
purpose and never ranks or filters by meaning.

A prefix is enumerated through the caller's ``list_prefix`` (the tool passes its admitted,
path-only inventory of exactly that prefix, so nothing else is walked or parsed); leads
are read through the caller's admitted byte reader. Every enumeration limit, unreadable
lead or spent ``deadline`` becomes an ``ARCHITECTURE_LIMIT_MARKER`` row, which
``query_code`` reports as an incomplete reply: a cut answer never looks like a complete
count. Files under a ``tests/`` directory below the prefix are counted and left out
unless the prefix itself names that directory.
"""

from __future__ import annotations

import ast
import io
import pathlib
import time
import tokenize
from typing import Callable, List, Tuple

from ouroboros import work_domain
from ouroboros.code_intelligence_architecture import ARCHITECTURE_LIMIT_MARKER

_LEAD_CHARS = 200
_JS_SUFFIXES = (".js", ".mjs", ".cjs", ".ts", ".mts", ".cts", ".jsx", ".tsx", ".css")

ListPrefix = Callable[[str], Tuple[List[str], List[str]]]


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")[:_LEAD_CHARS]


def python_lead(text: str) -> str:
    """The first docstring line of a module, read from its first token; ``""`` when it has none."""
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.ENCODING):
                continue
            if tok.type == tokenize.STRING:
                value = ast.literal_eval(tok.string)
                return _first_line(value) if isinstance(value, str) else ""
            return ""
    except (tokenize.TokenError, SyntaxError, ValueError, IndentationError):
        return ""
    return ""


def header_comment_lead(text: str) -> str:
    """The leading ``//`` run or ``/* */`` block of a JS/TS/CSS file, as one line; ``""`` when absent."""
    body = text.lstrip("﻿").lstrip()
    if body.startswith("#!"):
        body = body.split("\n", 1)[1].lstrip() if "\n" in body else ""
    if body.startswith("/*"):
        block = body[2:].split("*/", 1)[0]
        words = [line.strip().lstrip("*").strip() for line in block.splitlines()]
    elif body.startswith("//"):
        words = []
        for line in body.splitlines():
            if not line.strip().startswith("//"):
                break
            words.append(line.strip()[2:].strip())
    else:
        return ""
    return " ".join(word for word in words if word)[:_LEAD_CHARS]


def _lead(relpath: str, raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if relpath.endswith(".py"):
        return python_lead(text)
    if relpath.endswith(_JS_SUFFIXES):
        return header_comment_lead(text)
    return ""


def _default_list_prefix(root: pathlib.Path) -> ListPrefix:
    def listed(prefix: str) -> Tuple[List[str], List[str]]:
        from ouroboros.code_intelligence import build_code_inventory
        inventory = build_code_inventory(root, scope=prefix, persist=False, parse=False)
        return [file.path for file in inventory.files], list(inventory.enumeration_limits)
    return listed


def _prefix(arg: str) -> str:
    raw = arg.strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part and part != "."]
    if raw.startswith("/") or (len(raw) > 1 and raw[1] == ":") or not parts or ".." in parts:
        raise ValueError(f"module_map takes 'domains', a domain id or a repository-relative path prefix, not {arg!r}")
    return "/".join(parts)


def _under(path: str, prefix: str) -> bool:
    return path == prefix or path.startswith(prefix + "/")


def module_map_rows(root: pathlib.Path, arg: str, manifest, *, list_prefix: ListPrefix | None = None,
                    read_bytes: Callable[[str], bytes] | None = None, deadline: float | None = None) -> List[str]:
    """Rows for one ``module_map`` argument; limit rows start with the architecture limit marker."""
    root = pathlib.Path(root)
    refs = manifest.references
    if arg.strip().lower() == "domains":
        counts: dict = {}
        for domain in manifest.modules.values():
            counts[domain] = counts.get(domain, 0) + 1
        rows = [f"{d} — {manifest.domains[d]} ({counts.get(d, 0)} modules) | manuals: {', '.join(refs.get(d, ())) or '—'}"
                for d in sorted(manifest.domains)]
        return rows + [f"{key} | manuals: {', '.join(paths)}" for key, paths in sorted(refs.items())
                       if key not in manifest.domains]
    domain = arg.strip().upper()
    limits: List[str] = []
    if domain in manifest.domains:
        paths = sorted(path for path, owner in manifest.modules.items() if owner == domain)
        head = [f"{domain} — {manifest.domains[domain]} ({len(paths)} modules) | manuals: "
                f"{', '.join(refs.get(domain, ())) or '—'}"]
    else:
        prefix = _prefix(arg)
        listed, limits = (list_prefix or _default_list_prefix(root))(prefix)
        wants_tests = "tests" in prefix.split("/")
        tests = [p for p in listed if not wants_tests and "tests" in p[len(prefix):].split("/")]
        paths = sorted(p for p in listed if _under(p, prefix) and p not in tests)
        owners = [key for key in refs if key not in manifest.domains and _under(prefix, key.rstrip("/"))]
        domains = sorted({manifest.modules[p] for p in paths if p in manifest.modules})
        if owners:
            manuals = ", ".join(refs[max(owners, key=len)])
        elif len(domains) == 1:
            manuals = ", ".join(refs.get(domains[0], ())) or "—"
        else:
            manuals = "per domain (module_map domains)" if domains else "—"
        head = [f"{prefix} ({len(paths)} file(s)) | manuals: {manuals}"]
        if tests:
            head.append(f"{len(tests)} file(s) under tests/ left out; name that directory to list them")
    rows = []
    for index, path in enumerate(paths):
        if deadline is not None and time.monotonic() >= deadline:
            limits.append(f"module_map_time_limit: leads of {len(paths) - index} of {len(paths)} files not read")
            rows.extend(paths[index:])
            break
        lead = ""
        if path.endswith((".py", *_JS_SUFFIXES)):
            try:
                raw = read_bytes(path) if read_bytes is not None else work_domain.read_bytes(root / path)
                lead = _lead(path, raw)
            except work_domain.WorkDomainUnsettled:
                raise
            except (OSError, PermissionError, ValueError):
                limits.append(f"module_map_unreadable: {path}")
        owner = manifest.modules.get(path)
        rows.append(path + (f" ({owner})" if owner and domain not in manifest.domains else "")
                    + (f" — {lead}" if lead else ""))
    if not paths and not limits:
        head.append("no files under this prefix (enumeration complete)")
    return head + rows + [f"{ARCHITECTURE_LIMIT_MARKER} {note}" for note in limits]
