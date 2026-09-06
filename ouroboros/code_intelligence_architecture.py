"""Architecture facts over the pinned domain/contract/persistence carriers (CPL-3).

Five pure queries whose consumer #1 is Ouroboros itself (self-evolution):

- ``owner_of(path|symbol)`` — the domain owner per ``ouroboros/domains.toml``;
- ``domain_dependencies(d)`` — the manifest-baseline direction edges of one domain;
- ``facade_consumers(sym)`` — who imports through a compatibility facade (the
  ``noqa: F401`` re-export convention the facade inventory documents);
- ``persistence_entities_written_by(sym)`` — the durable entities a writer owns
  per ``docs/PERSISTENCE.md``;
- ``protected_contracts_affected(diff)`` — the protected surfaces
  (``runtime_mode_policy`` inventories) and frozen-contract rows
  (``docs/v7next/FROZEN_CONTRACTS_INVENTORY.md``) a change set touches.

Everything here is a pure function over data the repository already pins as
SSOT — the domain manifest, the generated inventories, and the protected-path
inventories. No LLM, no caches, no ledgers: every reader takes an explicit
``repo_root``, reads the carrier files fresh, and raises a teaching
``ValueError`` when a carrier is missing or an argument is malformed. The
model consumes these through the existing ``query_code`` tool
(``op=architecture``) — the seam decision is recorded in the campaign ledger
(``docs/v7next/LEDGER_CORRECTIONS.md``, F5 lane C section).
"""

from __future__ import annotations

import ast
import pathlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from ouroboros.code_intelligence import CodeInventory, _resolve_relative_import

DOMAIN_MANIFEST_RELPATH = "ouroboros/domains.toml"
PERSISTENCE_DOC_RELPATH = "docs/PERSISTENCE.md"
FROZEN_INVENTORY_RELPATH = "docs/v7next/FROZEN_CONTRACTS_INVENTORY.md"

ARCHITECTURE_FACTS = (
    "owner_of",
    "domain_dependencies",
    "facade_consumers",
    "persistence_entities_written_by",
    "protected_contracts_affected",
)

# The facade convention the generated facade inventory documents: a top-level
# ``from <module> import name`` carrying the F401-noqa marker is a re-export.
_NOQA_F401 = re.compile(r"#\s*noqa(?::[^#]*\bF401\b|\s*$|:\s*$)")
_BACKTICK_SPAN = re.compile(r"`([^`]+)`")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DIFF_GIT = re.compile(r"^diff --git a/(\S+) b/(\S+)$", re.MULTILINE)
_DIFF_FILE = re.compile(r"^(?:\+\+\+|---) (?:[ab]/)?(\S+)", re.MULTILINE)
_DIFF_RENAME = re.compile(r"^rename (?:from|to) (\S+)$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Carrier loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainManifest:
    """The parsed domain manifest: vocabulary, assignment, baseline edges."""

    domains: Dict[str, str]        # "D02" -> title
    modules: Dict[str, str]        # repo-relative path -> domain id
    graph_allowed: Tuple[str, ...]  # strict baseline "D01->D02" pairs
    lazy_only: Tuple[str, ...]      # lazy-only baseline pairs


def _read_carrier(repo_root: pathlib.Path, relpath: str) -> str:
    path = pathlib.Path(repo_root) / relpath
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"architecture carrier {relpath} is unreadable under {repo_root} "
            f"({exc}); architecture facts are defined over an Ouroboros "
            "checkout that carries its pinned inventories"
        ) from exc


def load_domain_manifest(repo_root: pathlib.Path) -> DomainManifest:
    try:  # Python 3.11+
        import tomllib
    except ImportError:  # pragma: no cover - 3.10 venvs ship tomli
        import tomli as tomllib

    raw = _read_carrier(repo_root, DOMAIN_MANIFEST_RELPATH)
    try:
        data = tomllib.loads(raw)
    except Exception as exc:
        raise ValueError(f"{DOMAIN_MANIFEST_RELPATH} does not parse as TOML: {exc}") from exc
    graph = data.get("graph") or {}
    return DomainManifest(
        domains={str(k): str(v) for k, v in (data.get("domains") or {}).items()},
        modules={str(k): str(v) for k, v in (data.get("modules") or {}).items()},
        graph_allowed=tuple(str(p) for p in graph.get("allowed") or ()),
        lazy_only=tuple(str(p) for p in graph.get("lazy_only") or ()),
    )


def _module_dotted(path: str) -> str:
    parts = path[:-3].split("/") if path.endswith(".py") else path.split("/")
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _split_md_row(line: str) -> List[str]:
    """Split one markdown table row on unescaped pipes (inventory convention)."""
    body = line.strip().strip("|")
    cells: List[str] = []
    cur: List[str] = []
    escaped = False
    for ch in body:
        if escaped:
            cur.append(ch)
            escaped = False
        elif ch == "\\":
            cur.append(ch)
            escaped = True
        elif ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur).strip())
    return cells


# ---------------------------------------------------------------------------
# owner_of
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainOwner:
    query: str
    module: str        # the manifest module the query resolved to
    domain: str        # "D02"
    domain_title: str
    via: str           # module_path | dotted_module | symbol_definition


def owner_of(
    repo_root: pathlib.Path,
    query: str,
    *,
    inventory: CodeInventory | None = None,
) -> Tuple[DomainOwner, ...]:
    """Domain owner(s) of a module path, dotted module, or defined symbol.

    A path or dotted module resolves directly against the manifest; a bare
    symbol resolves through the code inventory to its defining module(s) and
    each definition inside the manifest population reports its owner. A target
    outside the runtime module population returns ``()`` — no domain owns it.
    """
    manifest = load_domain_manifest(repo_root)
    text = str(query or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("owner_of requires a module path, dotted module, or symbol name")
    norm = text[2:] if text.startswith("./") else text

    def _owner(path: str, via: str) -> DomainOwner:
        domain = manifest.modules[path]
        return DomainOwner(text, path, domain, manifest.domains.get(domain, ""), via)

    if norm in manifest.modules:
        return (_owner(norm, "module_path"),)
    if "/" in norm or norm.endswith(".py"):
        return ()  # a real path outside the runtime module population
    dotted_map = {_module_dotted(path): path for path in manifest.modules}
    if norm in dotted_map:
        return (_owner(dotted_map[norm], "dotted_module"),)
    if "." in norm:
        return ()  # dotted, but not a population module
    if inventory is None:
        from ouroboros.code_intelligence import build_code_inventory

        inventory = build_code_inventory(pathlib.Path(repo_root), persist=False)
    from ouroboros.code_intelligence import symbol_definitions

    owners = {
        file.path
        for file, _symbol in symbol_definitions(inventory, norm)
        if file.path in manifest.modules
    }
    return tuple(_owner(path, "symbol_definition") for path in sorted(owners))


# ---------------------------------------------------------------------------
# domain_dependencies
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DomainDependencies:
    domain: str
    title: str
    outgoing: Tuple[str, ...]       # strict manifest-baseline edges out of the domain
    incoming: Tuple[str, ...]       # strict edges into the domain
    lazy_outgoing: Tuple[str, ...]  # lazy-only baseline edges out
    lazy_incoming: Tuple[str, ...]  # lazy-only baseline edges in


def domain_dependencies(repo_root: pathlib.Path, domain: str) -> DomainDependencies:
    """The manifest-baseline dependency edges of one domain (both directions)."""
    manifest = load_domain_manifest(repo_root)
    dom = str(domain or "").strip().upper()
    if dom not in manifest.domains:
        known = ", ".join(sorted(manifest.domains))
        raise ValueError(f"unknown domain {domain!r}; the manifest domains are: {known}")

    def _ends(pairs: Iterable[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        out: List[str] = []
        inc: List[str] = []
        for pair in pairs:
            src, _, dst = pair.partition("->")
            if src == dom:
                out.append(dst)
            if dst == dom:
                inc.append(src)
        return tuple(sorted(out)), tuple(sorted(inc))

    outgoing, incoming = _ends(manifest.graph_allowed)
    lazy_out, lazy_in = _ends(manifest.lazy_only)
    return DomainDependencies(
        dom, manifest.domains[dom], outgoing, incoming, lazy_out, lazy_in,
    )


# ---------------------------------------------------------------------------
# facade_consumers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FacadeConsumer:
    facade: str    # the facade module path
    consumer: str  # the module importing through the facade
    name: str      # the imported name ("" = the facade module itself, "*" = star)
    line: int


def _population_resolver(modules: Dict[str, str]):
    dotted_map = {_module_dotted(path): path for path in modules}

    def resolve(dotted: str) -> str | None:
        parts = dotted.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in dotted_map:
                return dotted_map[candidate]
        return None

    return resolve


def facade_reexports(
    repo_root: pathlib.Path,
    manifest: DomainManifest | None = None,
) -> Dict[str, Dict[str, str]]:
    """``facade path -> {exported name: leaf path}`` over the manifest population.

    Same convention the generated facade inventory pins: a top-level
    ``from <population module> import ...`` statement carrying the
    ``noqa: F401`` marker declares re-export bindings.
    """
    root = pathlib.Path(repo_root)
    manifest = manifest or load_domain_manifest(root)
    resolve = _population_resolver(manifest.modules)
    facades: Dict[str, Dict[str, str]] = {}
    for path in sorted(manifest.modules):
        source_path = root / path
        if not source_path.is_file():
            continue
        source = source_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        try:
            tree = ast.parse(source, filename=path)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ImportFrom):
                continue
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if not any(
                _NOQA_F401.search(lines[lineno - 1])
                for lineno in range(node.lineno, min(end, len(lines)) + 1)
            ):
                continue
            base = _resolve_relative_import(
                pathlib.PurePosixPath(path), node.module or "", int(node.level or 0),
            )
            for alias in node.names:
                if alias.name == "*":
                    leaf = resolve(base)
                else:
                    leaf = resolve(f"{base}.{alias.name}") or resolve(base)
                if leaf is None or leaf == path:
                    continue
                exported = alias.asname or alias.name
                facades.setdefault(path, {})[exported] = leaf
    return facades


def facade_consumers(repo_root: pathlib.Path, sym: str) -> Tuple[FacadeConsumer, ...]:
    """Who imports through a facade — for a facade module, or one re-exported name.

    ``sym`` may be a facade module (path or dotted) — every import of that
    facade across the population is a consumer row — or a bare re-exported
    name — the rows narrow to ``from <facade> import <name>`` sites of the
    facades that re-export it. Attribute access on a plain module import
    (``import ouroboros.llm`` then ``llm.chat``) is deliberately out of scope:
    only import statements are counted.
    """
    root = pathlib.Path(repo_root)
    manifest = load_domain_manifest(root)
    reexports = facade_reexports(root, manifest)
    text = str(sym or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("facade_consumers requires a facade module or a re-exported name")

    name_filter = ""
    if text in manifest.modules or "/" in text or text.endswith(".py"):
        targets = {text} if text in reexports else set()
        if not targets:
            raise ValueError(
                f"{text} is not a facade module (no top-level noqa:F401 re-exports); "
                "see docs/v7next/FACADE_INVENTORY.md for the facade list"
            )
    else:
        dotted_map = {_module_dotted(path): path for path in reexports}
        if text in dotted_map:
            targets = {dotted_map[text]}
        else:
            targets = {facade for facade, exports in reexports.items() if text in exports}
            name_filter = text
            if not targets:
                raise ValueError(
                    f"no facade re-exports a name or matches a module {text!r}; "
                    "facade_consumers answers about noqa:F401 facade bindings"
                )

    resolve = _population_resolver(manifest.modules)
    rows: List[FacadeConsumer] = []
    for path in sorted(manifest.modules):
        source_path = root / path
        if path in targets or not source_path.is_file():
            continue
        try:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = resolve(alias.name)
                    if resolved in targets and not name_filter:
                        rows.append(FacadeConsumer(resolved, path, "", node.lineno))
            elif isinstance(node, ast.ImportFrom):
                base = _resolve_relative_import(
                    pathlib.PurePosixPath(path), node.module or "", int(node.level or 0),
                )
                for alias in node.names:
                    if alias.name == "*":
                        resolved = resolve(base)
                        if resolved in targets and not name_filter:
                            rows.append(FacadeConsumer(resolved, path, "*", node.lineno))
                        continue
                    base_module = resolve(base)
                    named_module = resolve(f"{base}.{alias.name}")
                    if named_module in targets and named_module != base_module:
                        # ``from ouroboros import llm`` — the facade module itself.
                        if not name_filter:
                            rows.append(FacadeConsumer(named_module, path, "", node.lineno))
                        continue
                    if base_module in targets:
                        if name_filter and alias.name != name_filter:
                            continue
                        rows.append(FacadeConsumer(base_module, path, alias.name, node.lineno))
    return tuple(sorted(rows, key=lambda r: (r.facade, r.consumer, r.line, r.name)))


# ---------------------------------------------------------------------------
# persistence_entities_written_by
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PersistenceWrite:
    entity: str       # the row's Path cell (entity label, backticks kept)
    section: str      # the "## ..." section the row lives in
    writer_cell: str  # the row's raw Writer cell
    matched: str      # the token of ``sym`` that matched the writer cell


def _persistence_rows(repo_root: pathlib.Path) -> List[Tuple[str, str, str]]:
    """Every ``(section, entity_cell, writer_cell)`` row of docs/PERSISTENCE.md."""
    text = _read_carrier(repo_root, PERSISTENCE_DOC_RELPATH)
    rows: List[Tuple[str, str, str]] = []
    section = ""
    header: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
            header = []
            continue
        if not line.startswith("|"):
            header = []
            continue
        cells = _split_md_row(line)
        if [c.lower() for c in cells[:2]] == ["path", "writer"]:
            header = cells
            continue
        if header and set(line.replace("|", "").strip()) <= {"-", " ", ":"}:
            continue
        if header and len(cells) >= 2:
            rows.append((section, cells[0], cells[1]))
    if not rows:
        raise ValueError(
            f"{PERSISTENCE_DOC_RELPATH} carries no Path|Writer table rows; "
            "the persistence inventory is the carrier this query reads"
        )
    return rows


def persistence_entities_written_by(
    repo_root: pathlib.Path, sym: str,
) -> Tuple[PersistenceWrite, ...]:
    """Durable entities whose PERSISTENCE.md writer cell names ``sym``.

    ``sym`` may be a writer module (path or dotted — matched against the
    backticked module spans of the Writer column) or a bare function name
    (word-matched inside the Writer cell prose, e.g. ``save_settings``).
    """
    text = str(sym or "").strip().replace("\\", "/")
    if not text:
        raise ValueError("persistence_entities_written_by requires a writer module or name")
    if "." in text and "/" not in text and not text.endswith(".py"):
        candidate = text.replace(".", "/") + ".py"
        manifest = load_domain_manifest(repo_root)
        if candidate in manifest.modules:
            text = candidate
    is_path = "/" in text or text.endswith(".py")
    rows: List[PersistenceWrite] = []
    for section, entity, writer in _persistence_rows(repo_root):
        if is_path:
            spans = _BACKTICK_SPAN.findall(writer)
            if not any(span == text or span.endswith("/" + text) for span in spans):
                continue
        elif text not in _WORD.findall(writer):
            continue
        rows.append(PersistenceWrite(entity, section, writer, text))
    return tuple(rows)


# ---------------------------------------------------------------------------
# protected_contracts_affected
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ContractImpact:
    contract: str  # the frozen-contract row label
    path: str      # the changed path that hits the row
    role: str      # owner | anchor


@dataclass(frozen=True)
class ProtectedImpact:
    paths: Tuple[str, ...]                 # normalized changed paths considered
    protected: Tuple[Any, ...]             # runtime_mode_policy.ProtectedPath rows
    contracts: Tuple[ContractImpact, ...]  # frozen-contract rows affected


def _frozen_contract_rows(repo_root: pathlib.Path) -> List[Tuple[str, str, str]]:
    """``(label, file, role)`` triples of the generated frozen-contracts inventory."""
    text = _read_carrier(repo_root, FROZEN_INVENTORY_RELPATH)
    triples: List[Tuple[str, str, str]] = []
    for line in text.splitlines():
        if line.startswith("- browser-envelope prose owners:"):
            for span in _BACKTICK_SPAN.findall(line):
                triples.append(("browser-envelope ABI", span, "owner"))
            continue
        if not line.startswith("|"):
            continue
        cells = _split_md_row(line)
        if len(cells) < 4 or not cells[0].isdigit():
            continue
        label_match = _BACKTICK_SPAN.search(cells[1])
        label = (label_match.group(1) if label_match else cells[1]).replace("\\|", "|")
        for cell, role in ((cells[2], "owner"), (cells[3], "anchor")):
            for span in _BACKTICK_SPAN.findall(cell):
                file_part = span.replace("\\|", "|").split("::", 1)[0].split(" ", 1)[0]
                if file_part:
                    triples.append((label, file_part, role))
    if not triples:
        raise ValueError(
            f"{FROZEN_INVENTORY_RELPATH} carries no contract rows; regenerate the "
            "inventory (scripts/regenerate_inventories.py) before asking for impact"
        )
    return triples


def paths_from_diff(diff: str | Iterable[str]) -> Tuple[str, ...]:
    """Changed paths from a unified diff text, or a pass-through path iterable."""
    from ouroboros.runtime_mode_policy import normalize_repo_path

    if isinstance(diff, str):
        raw: List[str] = []
        if _DIFF_GIT.search(diff) or _DIFF_FILE.search(diff):
            for a_path, b_path in _DIFF_GIT.findall(diff):
                raw.extend((a_path, b_path))
            raw.extend(_DIFF_FILE.findall(diff))
            raw.extend(_DIFF_RENAME.findall(diff))
        else:
            raw.extend(part for part in re.split(r"[,\s]+", diff) if part)
        candidates = raw
    else:
        candidates = [str(p) for p in diff]
    normalized = {
        normalize_repo_path(p)
        for p in candidates
        if p and p != "/dev/null" and not p.startswith("/dev/")
    }
    return tuple(sorted(p for p in normalized if p and p != "."))


def protected_contracts_affected(
    repo_root: pathlib.Path, diff: str | Iterable[str],
) -> ProtectedImpact:
    """Protected surfaces and frozen contracts a change set touches.

    ``diff`` is a unified diff text or an iterable of changed repo-relative
    paths. Protected categories come from the ``runtime_mode_policy``
    inventories (safety-critical / frozen-contract / release-invariant); the
    contract rows come from the generated frozen-contracts inventory — a
    changed path that is an owner or anchor file of a row names that contract.
    """
    from ouroboros.runtime_mode_policy import protected_paths_in

    paths = paths_from_diff(diff)
    if not paths:
        raise ValueError(
            "protected_contracts_affected received no changed paths; pass a "
            "unified diff or a comma/space-separated repo-relative path list"
        )
    protected = tuple(protected_paths_in(paths))
    contracts: List[ContractImpact] = []
    triples = _frozen_contract_rows(repo_root)
    for path in paths:
        for label, file_part, role in triples:
            if file_part == path:
                contracts.append(ContractImpact(label, path, role))
    deduped = tuple(sorted(set(contracts), key=lambda c: (c.contract, c.path, c.role)))
    return ProtectedImpact(paths=paths, protected=protected, contracts=deduped)


# ---------------------------------------------------------------------------
# The query_code op=architecture renderer
# ---------------------------------------------------------------------------

def architecture_fact_rows(
    repo_root: pathlib.Path,
    query: str,
    *,
    inventory: CodeInventory | None = None,
) -> List[str]:
    """Render one architecture fact as compact tool rows.

    ``query`` is ``"<fact> <argument>"`` where fact is one of
    ``ARCHITECTURE_FACTS``; for ``protected_contracts_affected`` the argument
    is a comma/space-separated changed-path list (or a pasted unified diff).
    """
    text = str(query or "").strip()
    fact, _, arg = text.partition(" ")
    fact = fact.strip().lower()
    arg = arg.strip()
    if fact not in ARCHITECTURE_FACTS:
        raise ValueError(
            "op=architecture takes query='<fact> <argument>' with fact one of: "
            + ", ".join(ARCHITECTURE_FACTS)
        )
    if not arg:
        raise ValueError(f"architecture fact {fact} requires an argument after the fact name")
    root = pathlib.Path(repo_root)
    if fact == "owner_of":
        owners = owner_of(root, arg, inventory=inventory)
        if not owners:
            return [f"{arg}: no domain owner — not in the runtime module population "
                    f"({DOMAIN_MANIFEST_RELPATH})"]
        return [
            f"{row.module} -> {row.domain} ({row.domain_title}) [{row.via}]"
            for row in owners
        ]
    if fact == "domain_dependencies":
        deps = domain_dependencies(root, arg)
        return [
            f"{deps.domain} ({deps.title}) — manifest-baseline strict edges",
            f"imports: {', '.join(deps.outgoing) or '—'}",
            f"imported by: {', '.join(deps.incoming) or '—'}",
            f"lazy-only imports: {', '.join(deps.lazy_outgoing) or '—'}",
            f"lazy-only imported by: {', '.join(deps.lazy_incoming) or '—'}",
        ]
    if fact == "facade_consumers":
        rows = facade_consumers(root, arg)
        if not rows:
            return [f"{arg}: facade has no import-statement consumers in the population"]
        return [
            f"{row.consumer}:{row.line} imports "
            + (f"{row.name} from {row.facade}" if row.name else f"{row.facade}")
            for row in rows
        ]
    if fact == "persistence_entities_written_by":
        writes = persistence_entities_written_by(root, arg)
        if not writes:
            return [f"{arg}: no durable entity in {PERSISTENCE_DOC_RELPATH} names this writer"]
        return [
            f"{row.entity} [{row.section}] writer: {row.writer_cell}"
            for row in writes
        ]
    impact = protected_contracts_affected(root, arg)
    rows = [
        f"{len(impact.paths)} changed path(s): {len(impact.protected)} protected, "
        f"{len(impact.contracts)} frozen-contract row(s) affected"
    ]
    rows.extend(f"{p.path} -> {p.category}" for p in impact.protected)
    rows.extend(
        f"frozen contract `{c.contract}` ({c.role}: {c.path})" for c in impact.contracts
    )
    return rows
