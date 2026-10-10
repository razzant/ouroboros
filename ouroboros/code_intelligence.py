"""Internal deterministic code inventory v5.

No embeddings, no LSP, no SQLite, and no raw source cache. This is a compact
structural projection used by digest/review context builders and read-only
code-query tools. The persisted JSON contains local structural facts only.
Cross-file import candidates and name occurrences are joined against current
source at query time; no cached name match is a semantic binding. Full source
buffers are never cached.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List

from ouroboros.code_import_candidates import import_specs, is_import_node
from ouroboros.utils import atomic_write_json, utc_now_iso

CODE_INTELLIGENCE_SCHEMA_VERSION = 5

SKIP_DIRS = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "node_modules", "dist", "build",
    ".tox", ".eggs", "python-standalone", "assets",
})
SEARCH_SKIP_GLOBS = frozenset({
    "*.pyc", "*.pyo", "*.so", "*.dylib", "*.dll", "*.exe",
    "*.bin", "*.o", "*.a", "*.tar", "*.gz", "*.zip",
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.ico", "*.webp",
    "*.woff", "*.woff2", "*.ttf", "*.eot",
    "*.min.js", "*.min.css", "*.map",
    "*.db", "*.sqlite", "*.sqlite3",
    "*.lock",
})
_SKIP_DIRS = set(SKIP_DIRS)
_JS_IMPORT_RE = re.compile(r"""(?m)^\s*(?:import\s+.*?\s+from\s+|export\s+.*?\s+from\s+|import\s*\(|require\s*\()\s*['"]([^'"]+)['"]""")
_ROUTE_RE = re.compile(r"""(?i)(?:route|path)\s*[:=]\s*['"]([^'"]+)['"]|@\w+\.route\(['"]([^'"]+)['"]""")
_SENSITIVE_NAME_RE = re.compile(r"(?i)(token|secret|credential|private[_-]?key|api[_-]?key|password|passwd)")
_SENSITIVE_EXTENSIONS = {".json", ".env", ".key", ".pem", ".p12", ".pfx", ".crt", ".cer"}
_MAX_INDEX_FILE_BYTES = 2_000_000
_MAX_ENUMERATION_PATHS = 20_000
_MAX_ENUMERATION_SECONDS = 45.0
# Bound optional local import evidence independently of source/file limits.
# Overflow requests a source parse, never a silently truncated import answer.
_MAX_LOCAL_IMPORT_FACTS = 2000
_MAX_LOCAL_IMPORT_CHARS = 128_000


@dataclass
class SymbolFact:
    name: str
    kind: str
    line_start: int
    line_end: int
    signature: str = ""


@dataclass
class CallSiteFact:
    name: str
    line: int
    enclosing: str = ""


@dataclass
class ImportFact:
    specifier: str
    line: int
    column: int
    slot: str
    enclosing: str = ""


class _ImportFacts:
    """One bounded file-local projection; no snippets or joined target paths."""

    def __init__(self):
        self.rows: list[ImportFact] = []
        self.complete = True
        self.chars = 0

    def add(self, specifier: str, line: int, column: int, slot: str, enclosing: str):
        if not self.complete:
            return
        self.chars += len(specifier) + len(slot) + len(enclosing)
        if len(self.rows) >= _MAX_LOCAL_IMPORT_FACTS or self.chars > _MAX_LOCAL_IMPORT_CHARS:
            self.complete = False
            self.rows.clear()
            return
        self.rows.append(ImportFact(specifier, line, column, slot, enclosing))


@dataclass
class FileFact:
    path: str
    sha256: str
    size: int
    language: str
    token_estimate: int
    disposition: str = "indexed"
    syntax_error: str = ""
    symbols: List[SymbolFact] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    call_sites: List[CallSiteFact] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    import_facts: List[ImportFact] = field(default_factory=list)
    import_facts_complete: bool = False
    import_facts_method: str = ""


@dataclass
class CodeInventory:
    schema_version: int
    repo_root: str
    git_head: str
    created_at: str
    files: List[FileFact]
    coverage: Dict[str, int]
    nested_roots: List[str] = field(default_factory=list)
    enumeration_limits: List[str] = field(default_factory=list)

    def to_json(self) -> Dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "repo_root": self.repo_root,
            "git_head": self.git_head,
            "created_at": self.created_at,
            "files": [
                {
                    **asdict(file),
                    "symbols": [asdict(symbol) for symbol in file.symbols],
                }
                for file in self.files
            ],
            "coverage": dict(self.coverage),
            "nested_roots": list(self.nested_roots),
            "enumeration_limits": list(self.enumeration_limits),
        }


def inventory_cache_path(repo_root: pathlib.Path, drive_root: pathlib.Path) -> pathlib.Path:
    root = pathlib.Path(repo_root).resolve(strict=False)
    repo_key = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    return pathlib.Path(drive_root) / "state" / "code_intel" / repo_key / "inventory.json"


def prune_stale_code_intel_roots(
    drive_root: pathlib.Path,
    retention_days: "int | None" = None,
    *,
    now: "float | None" = None,
) -> Dict[str, Any]:
    """Age-prune workspace roots whose inventory went stale (CPL4-C14).

    The root-dir key is a one-way path hash, so liveness cannot be probed —
    the ``inventory.json`` mtime is the signal (an active root rewrites it on
    every index). Pure derived cache: a pruned root costs one full re-index,
    never data. Fail-soft per entry; a root whose inventory is missing ages
    by the directory's own mtime.
    """
    import shutil

    from ouroboros.retention import age_cutoff, get_gc_retention_days

    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)
    report: Dict[str, Any] = {"removed": [], "kept": 0, "errors": []}
    cache_root = pathlib.Path(drive_root) / "state" / "code_intel"
    try:
        entries = sorted(p for p in cache_root.iterdir() if p.is_dir())
    except OSError:
        return report
    for entry in entries:
        probe = entry / "inventory.json"
        try:
            mtime = (probe if probe.exists() else entry).stat().st_mtime
        except OSError:
            report["kept"] += 1
            continue
        if mtime >= cutoff:
            report["kept"] += 1
            continue
        try:
            shutil.rmtree(entry)
            report["removed"].append(entry.name)
        except OSError:
            report["errors"].append({"entry": entry.name, "error": "remove_failed"})
    return report


def _symbol_from_json(raw: Any) -> SymbolFact:
    data = raw if isinstance(raw, dict) else {}
    return SymbolFact(
        name=str(data.get("name") or ""),
        kind=str(data.get("kind") or ""),
        line_start=int(data.get("line_start") or 0),
        line_end=int(data.get("line_end") or data.get("line_start") or 0),
        signature=str(data.get("signature") or ""),
    )


def _call_from_json(raw: Any) -> CallSiteFact:
    data = raw if isinstance(raw, dict) else {}
    return CallSiteFact(
        name=str(data.get("name") or ""),
        line=int(data.get("line") or 0),
        enclosing=str(data.get("enclosing") or ""),
    )


def _file_from_json(raw: Any) -> FileFact | None:
    if not isinstance(raw, dict):
        return None
    required = ("path", "sha256", "size", "language", "token_estimate",
                "import_facts", "import_facts_complete", "import_facts_method")
    if any(key not in raw for key in required):
        return None
    try:
        return FileFact(
            path=str(raw.get("path") or ""),
            sha256=str(raw.get("sha256") or ""),
            size=int(raw.get("size") or 0),
            language=str(raw.get("language") or ""),
            token_estimate=int(raw.get("token_estimate") or 0),
            disposition=str(raw.get("disposition") or "indexed"),
            syntax_error=str(raw.get("syntax_error") or ""),
            symbols=[item for item in (_symbol_from_json(s) for s in raw.get("symbols") or []) if item.name],
            imports=sorted({str(item) for item in (raw.get("imports") or []) if str(item)}),
            routes=sorted({str(item) for item in (raw.get("routes") or []) if str(item)}),
            call_sites=[item for item in (_call_from_json(s) for s in raw.get("call_sites") or []) if item.name],
            exports=sorted({str(item) for item in (raw.get("exports") or []) if str(item)}),
            import_facts=[ImportFact(**item) for item in raw.get("import_facts", [])],
            import_facts_complete=raw.get("import_facts_complete") is True,
            import_facts_method=str(raw.get("import_facts_method") or ""),
        )
    except Exception:
        return None


def load_cached_inventory(repo_root: pathlib.Path, drive_root: pathlib.Path) -> CodeInventory | None:
    """Load current local facts; old caches may contain stale cross-file joins."""
    path = inventory_cache_path(repo_root, drive_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict) or raw.get("schema_version") != CODE_INTELLIGENCE_SCHEMA_VERSION:
        return None
    files = []
    for item in raw.get("files") or []:
        fact = _file_from_json(item)
        if fact is None:
            return None
        files.append(fact)
    coverage: Dict[str, int] = {}
    raw_cov = raw.get("coverage")
    if isinstance(raw_cov, dict):
        coverage = {str(k): int(v or 0) for k, v in raw_cov.items()}
    else:
        for file in files:
            coverage[file.disposition] = coverage.get(file.disposition, 0) + 1
    return CodeInventory(
        schema_version=CODE_INTELLIGENCE_SCHEMA_VERSION,
        repo_root=str(raw.get("repo_root") or pathlib.Path(repo_root).resolve(strict=False)),
        git_head=str(raw.get("git_head") or ""),
        created_at=str(raw.get("created_at") or ""),
        files=files,
        coverage=coverage,
        nested_roots=list(raw.get("nested_roots") or []),
        enumeration_limits=list(raw.get("enumeration_limits") or []),
    )


def _git_head(repo_root: pathlib.Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except Exception:
        return ""


def _enumerate_files(
    repo_root: pathlib.Path,
    *,
    scope: str = "",
    path_allowed: Callable[[pathlib.Path], bool] | None = None,
) -> tuple[List[pathlib.Path], List[str], List[str]]:
    """Enumerate Git-visible files, expanding embedded repositories on demand.

    Each nested Git root supplies its own ignore rules. Visibility is checked
    before entering it; directory symlinks are never followed. The fallback
    filesystem walk has no Git ignore semantics and reports that limitation.
    Limits bound visited entries/time, not the number of operation matches.
    A scope changes the initial walk root so unrelated entries spend no budget.
    """
    root = pathlib.Path(repo_root).resolve(strict=False)
    selected = root / pathlib.PurePosixPath(str(scope or "").replace("\\", "/"))
    try:
        selected_parts = selected.relative_to(root).parts
    except ValueError:
        raise ValueError("scope must stay within the inventory root") from None
    if ".." in selected_parts:
        raise ValueError("scope must stay within the inventory root")
    paths: set[pathlib.Path] = set()
    nested: set[str] = set()
    limits: set[str] = set()
    pending: list[tuple[pathlib.Path, str]] = []
    seen: set[pathlib.Path] = set()
    visited = 0
    deadline = time.monotonic() + _MAX_ENUMERATION_SECONDS

    def budget() -> bool:
        if visited >= _MAX_ENUMERATION_PATHS:
            limits.add(f"enumeration_entry_limit:{_MAX_ENUMERATION_PATHS}")
            return False
        if time.monotonic() >= deadline:
            limits.add(f"enumeration_time_limit:{_MAX_ENUMERATION_SECONDS:g}s")
            return False
        return True

    def permitted(path: pathlib.Path) -> bool:
        try:
            rel = path.relative_to(root)
        except ValueError:
            return False
        if ".." in rel.parts or any(part in _SKIP_DIRS for part in rel.parts):
            return False
        return path_allowed is None or path_allowed(path)

    def visible(path: pathlib.Path) -> bool:
        nonlocal visited
        visited += 1
        return permitted(path)

    def add(path: pathlib.Path) -> None:
        if not visible(path):
            return
        if path.is_symlink():
            # File symlinks are classified by _file_fact before any source read.
            # In particular, no traversal may escape through a directory link.
            if path.is_dir():
                limits.add("directory_symlinks_not_followed")
            else:
                paths.add(path)
        elif path.is_dir():
            pending.append((path, ""))
        else:
            paths.add(path)

    if selected != root:
        if not permitted(selected):
            return [], [], []
        # Check each directory before inspecting its Git marker. A scoped walk
        # inside a nested repo still reports that repo, even without visiting
        # the ancestor in its listing. Never enter through a directory symlink.
        directory = root
        for part in selected_parts:
            directory /= part
            if not permitted(directory):
                return [], [], []
            if directory.is_symlink() and (directory != selected or directory.is_dir()):
                return [], [], ["directory_symlinks_not_followed"]
            if directory.is_dir() and (directory / ".git").exists():
                nested.add(directory.relative_to(root).as_posix())
    if selected.is_dir():
        pending.append((selected, ""))
    elif selected.is_file() or selected.is_symlink():
        pending.append((selected.parent, selected.name))
    else:
        return [], sorted(nested), []

    while pending and budget():
        directory, filename = pending.pop()
        if directory in seen:
            continue
        seen.add(directory)
        if directory != root and (directory / ".git").exists():
            nested.add(directory.relative_to(root).as_posix())
        try:
            command = ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
            if filename:
                command.extend(["--", ":(literal)" + filename])
            proc = subprocess.run(
                command,
                cwd=str(directory), capture_output=True,
                timeout=max(0.01, min(10.0, deadline - time.monotonic())),
            )
        except (OSError, subprocess.TimeoutExpired):
            proc = None
        if proc is not None and proc.returncode == 0:
            for part in sorted(set(proc.stdout.split(b"\0"))):
                if not part:
                    continue
                if not budget():
                    break
                add(directory / os.fsdecode(part.rstrip(b"/")))
            continue
        limits.add("filesystem_fallback_without_git_ignores")
        if filename:
            add(directory / filename)
            continue
        # os.walk does not follow directory symlinks. Prune before descending,
        # including caller-invisible roots and nested repos with their own rules.
        for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
            if not budget():
                break
            current = pathlib.Path(dirpath)
            descend = []
            for name in sorted(dirnames):
                if not budget():
                    break
                child = current / name
                if not visible(child):
                    continue
                if child.is_symlink():
                    limits.add("directory_symlinks_not_followed")
                elif (child / ".git").exists():
                    pending.append((child, ""))
                else:
                    descend.append(name)
            dirnames[:] = descend
            for name in sorted(filenames):
                if not budget():
                    break
                add(current / name)
    return sorted(paths), sorted(nested), sorted(limits)


def _tracked_files(repo_root: pathlib.Path) -> List[pathlib.Path]:
    """Compatibility projection; build_code_inventory also retains walk limits."""
    return _enumerate_files(repo_root)[0]


def _language(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".mts": "typescript",
        ".cts": "typescript",
        ".go": "go",
        ".md": "markdown",
        ".json": "json",
        ".toml": "toml",
        ".yaml": "yaml",
        ".yml": "yaml",
    }.get(suffix, suffix.lstrip(".") or "text")


def _signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = [arg.arg for arg in node.args.args]
        return f"{node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    return ""




def _resolve_relative_import(rel_path: pathlib.PurePosixPath, module: str, level: int) -> str:
    if level <= 0:
        return module
    package_parts = list(rel_path.parent.parts)
    keep = max(0, len(package_parts) - level + 1)
    parts = package_parts[:keep]
    if module:
        parts.extend(str(module).split("."))
    return ".".join(part for part in parts if part)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _python_facts(text: str, rel_path: pathlib.PurePosixPath, *,
                  import_facts: _ImportFacts | None = None) -> tuple[List[SymbolFact], List[str], str, List[CallSiteFact]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [], f"{exc.msg} at line {exc.lineno}", []
    symbols: List[SymbolFact] = []
    imports: List[str] = []
    calls: List[CallSiteFact] = []
    stack: list[tuple[ast.AST, tuple[str, ...]]] = [(tree, ())]
    while stack:
        node, enclosing = stack.pop()
        child_enclosing = enclosing
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.append(SymbolFact(
                name=node.name,
                kind="class" if isinstance(node, ast.ClassDef) else ("async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"),
                line_start=int(getattr(node, "lineno", 0) or 0),
                line_end=int(getattr(node, "end_lineno", getattr(node, "lineno", 0)) or 0),
                signature=_signature(node),
            ))
            child_enclosing = (*enclosing, node.name)
        elif isinstance(node, ast.Assign):
            if all(isinstance(target, ast.Name) and target.id.isupper() for target in node.targets):
                for target in node.targets:
                    symbols.append(SymbolFact(target.id, "constant", int(getattr(node, "lineno", 0) or 0), int(getattr(node, "lineno", 0) or 0)))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.isupper():
            symbols.append(SymbolFact(node.target.id, "constant", int(getattr(node, "lineno", 0) or 0), int(getattr(node, "lineno", 0) or 0)))
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level and not node.module:
                imports.extend(
                    _resolve_relative_import(rel_path, alias.name, int(node.level or 0))
                    for alias in node.names
                    if alias.name and alias.name != "*"
                )
            elif node.module or node.level:
                imports.append(_resolve_relative_import(rel_path, node.module or "", int(node.level or 0)))
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name:
                calls.append(CallSiteFact(name, int(getattr(node, "lineno", 0) or 0), enclosing[-1] if enclosing else ""))
        if import_facts is not None and isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                specs = {alias.name for alias in node.names}
            else:
                base = "." * node.level + (node.module or "")
                specs = {base} if base else set()
                specs.update(base + ("" if base.endswith(".") else ".") + alias.name
                             for alias in node.names if base and alias.name != "*")
            for spec in sorted(specs):
                import_facts.add(spec, node.lineno, node.col_offset + 1,
                                 "ast." + type(node).__name__, " / ".join(enclosing[-2:]))
        stack.extend((child, child_enclosing) for child in reversed(list(ast.iter_child_nodes(node))))
    symbols = sorted(symbols, key=lambda item: (item.line_start, item.name))
    calls = sorted(calls, key=lambda item: (item.line, item.enclosing, item.name))
    return symbols, sorted(set(imports)), "", calls


def extract_js_imports(text: str) -> list[str]:
    """Return deterministic JS/TS import specifiers from source text."""
    return sorted(set(_JS_IMPORT_RE.findall(text)))


def extract_routes(text: str) -> list[str]:
    """Return deterministic route/path-like literals from source text."""
    return sorted({match[0] or match[1] for match in _ROUTE_RE.findall(text) if match[0] or match[1]})


# --- Generic polyglot structural extraction (tree-sitter) ----------------------
# ONE META path for every language without a bespoke extractor (Go/Rust/Java/Ruby/
# C/C++/C#/PHP/Kotlin/Swift/Scala/Lua/Bash + JS/TS) — no per-language regex. The
# Python path stays on the stdlib `ast` (the canonical, richer Python parser:
# signatures, relative-import resolution, constant/async kinds). When the grammar
# or the tree-sitter library is unavailable the caller surfaces a VISIBLE
# `structural_unavailable:<lang>` disposition — never a silent regex/AST guess.
_TS_LANGUAGES = {
    # _language() output -> tree-sitter-language-pack grammar name
    "go": "go", "rs": "rust", "rust": "rust", "java": "java", "rb": "ruby",
    "ruby": "ruby", "c": "c", "h": "c", "cpp": "cpp", "cc": "cpp", "cxx": "cpp",
    "hpp": "cpp", "cs": "csharp", "php": "php", "kt": "kotlin", "kts": "kotlin",
    "swift": "swift", "scala": "scala", "lua": "lua", "sh": "bash", "bash": "bash",
    "javascript": "javascript", "typescript": "typescript",
}
_TS_DEF_KINDS = {
    "function_declaration": "function", "function_item": "function", "function_definition": "function",
    "method_declaration": "method", "method_definition": "method", "method_spec": "method",
    "constructor_declaration": "constructor", "singleton_method": "method",
    "class_declaration": "class", "class_definition": "class",
    "struct_item": "struct", "struct_specifier": "struct", "union_specifier": "union",
    "interface_declaration": "interface", "enum_declaration": "enum", "enum_item": "enum",
    "trait_item": "trait", "impl_item": "impl", "protocol_declaration": "protocol",
    "type_declaration": "type", "type_alias_declaration": "type", "type_spec": "type",
    "module": "module", "namespace_declaration": "namespace", "object_declaration": "object",
    "const_item": "constant", "macro_definition": "macro",
}
_TS_NAME_TYPES = ("identifier", "type_identifier", "field_identifier", "property_identifier",
                  "constant", "scoped_identifier", "name", "word")
_TS_CALL_TYPES = {"call", "call_expression", "method_invocation", "function_call_expression",
                  "invocation_expression", "macro_invocation"}
_TS_IMPORT_TYPES = {"import_declaration", "import_spec", "import_statement", "use_declaration",
                    "using_directive", "preproc_include", "package_clause"}


@functools.lru_cache(maxsize=32)
def _ts_parser(grammar: str):
    """Cached tree-sitter parser for a grammar; None when the lib/grammar is absent."""
    try:
        from tree_sitter_language_pack import get_parser
        return get_parser(grammar)  # type: ignore[arg-type]
    except Exception:
        return None


def _ts_node_name(node: Any) -> str:
    named = node.child_by_field_name("name")
    if named is not None and named.text:
        return named.text.decode("utf-8", "replace")
    for child in node.children:
        if child.type in _TS_NAME_TYPES and child.text:
            return child.text.decode("utf-8", "replace")
    return ""


def _ts_callee_leaf(node: Any) -> Any:
    """Terminal identifier of a syntactic callee, never its receiver. Past member/name fields only a sole unlabelled
    operand (parentheses, `x!`) or a final name after labelled qualifiers (PHP ``\\A\\foo``) continues; ``getters[key]`` names none."""
    target = next((t for t in map(node.child_by_field_name, ("function", "name", "method")) if t is not None), None)
    while target is not None:
        if target.child_count == 0:
            return target if target.type in _TS_NAME_TYPES else None
        child = next((c for c in map(target.child_by_field_name, ("attribute", "property", "field", "name", "function")) if c is not None), None)
        if child is None:  # type arguments are no operand, so C# Target<int>() still names Target
            ops = [(target.field_name_for_child(i), c) for i, c in enumerate(target.children) if c.is_named and c.type != "type_argument_list"]
            if ops and ops[-1][0] is None and None not in [label for label, _ in ops[:-1]] and (len(ops) == 1 or ops[-1][1].type in _TS_NAME_TYPES):
                child = ops[-1][1]
        target = child
    return None


def _treesitter_facts(text: str, lang: str, *, raw: bytes | None = None,
                      import_facts: _ImportFacts | None = None):
    """Structural facts via tree-sitter, also used for partial Python imports.

    Returns (symbols, imports, syntax_error, calls), or None without a parser.
    """
    grammar = "python" if lang == "python" else _TS_LANGUAGES.get(lang)
    if not grammar:
        return None
    parser = _ts_parser(grammar)
    if parser is None:
        return None
    try:
        tree = parser.parse(raw if raw is not None else text.encode("utf-8", "replace"))
    except Exception:
        return None
    symbols: List[SymbolFact] = []
    calls: List[CallSiteFact] = []
    imports: List[str] = []
    root = tree.root_node
    # Iterative DFS carrying the enclosing definition name (for call attribution).
    stack: list[tuple[Any, str, tuple[str, ...]]] = [(root, "", ())]
    while stack:
        node, enclosing, ancestors = stack.pop()
        child_enclosing = enclosing
        child_ancestors = ancestors
        ntype = node.type
        if import_facts is not None and is_import_node(node, lang):
            for spec in import_specs(node, lang):
                import_facts.add(spec, node.start_point[0] + 1, node.start_point[1] + 1,
                                 ntype, " / ".join(ancestors[-2:]))
        if ntype in _TS_DEF_KINDS:
            name_node = node.child_by_field_name("name")
            if name_node is not None:
                child_ancestors = (*ancestors, name_node.text.decode("utf-8", "replace"))
        declaration_kind = _TS_DEF_KINDS.get(ntype)
        if ntype == "variable_declarator" and node.parent is not None:
            parent = node.parent
            name_node = node.child_by_field_name("name")
            if (parent.type in {"lexical_declaration", "variable_declaration"}
                    and name_node is not None and name_node.type == "identifier"):
                declaration_kind = "constant" if any(c.type == "const" for c in parent.children) else "variable"
        if declaration_kind:
            name = _ts_node_name(node)
            if name:
                sig = (node.text.decode("utf-8", "replace").splitlines() or [""])[0].strip()[:200] if node.text else ""
                symbols.append(SymbolFact(name, declaration_kind, node.start_point[0] + 1, node.end_point[0] + 1, sig))
                child_enclosing = name
        elif ntype in _TS_CALL_TYPES:
            callee = _ts_callee_leaf(node)
            if callee is not None and callee.text:
                calls.append(CallSiteFact(callee.text.decode("utf-8", "replace"), node.start_point[0] + 1, enclosing))
        elif ntype in _TS_IMPORT_TYPES and node.text:
            spec = (node.text.decode("utf-8", "replace").splitlines() or [""])[0].strip()[:200]
            if spec:
                imports.append(spec)
        stack.extend((child, child_enclosing, child_ancestors) for child in reversed(node.children))
    syntax_error = "syntax error" if root.has_error else ""
    symbols.sort(key=lambda s: (s.line_start, s.name))
    calls.sort(key=lambda c: (c.line, c.enclosing, c.name))
    return symbols, sorted(set(imports)), syntax_error, calls


def _file_fact(repo_root: pathlib.Path, path: pathlib.Path) -> FileFact:
    try:
        rel = path.relative_to(repo_root).as_posix()
    except ValueError:
        try:
            rel = path.resolve(strict=False).relative_to(repo_root).as_posix()
        except ValueError:
            return FileFact(str(path), "", 0, _language(path), 0, disposition="path_escape")
    try:
        path.resolve(strict=False).relative_to(repo_root.resolve(strict=False))
    except (OSError, ValueError):
        return FileFact(rel, "", 0, _language(path), 0, disposition="path_escape")
    if _is_sensitive_inventory_path(rel):
        return FileFact(rel, "", 0, _language(path), 0, disposition="sensitive")
    try:
        stat_size = path.stat().st_size
    except OSError as exc:
        return FileFact(rel, "", 0, _language(path), 0, disposition=f"read_error:{exc}")
    if stat_size > _MAX_INDEX_FILE_BYTES:
        return FileFact(rel, "", stat_size, _language(path), 0, disposition="oversized")
    try:
        with path.open("rb") as source:
            raw = source.read(_MAX_INDEX_FILE_BYTES + 1)
    except OSError as exc:
        return FileFact(rel, "", 0, _language(path), 0, disposition=f"read_error:{exc}")
    if len(raw) > _MAX_INDEX_FILE_BYTES:
        return FileFact(rel, "", len(raw), _language(path), 0, disposition="oversized")
    digest = hashlib.sha256(raw).hexdigest()
    lang = _language(path)
    token_est = max(1, len(raw) // 4)
    if b"\0" in raw[:4096]:
        return FileFact(rel, digest, len(raw), lang, token_est, disposition="binary")
    text = raw.decode("utf-8", errors="replace")
    local_imports = _ImportFacts()
    if lang == "python":
        symbols, imports, syntax_error, calls = _python_facts(text, pathlib.PurePosixPath(rel), import_facts=local_imports)
        import_method = "python ast"
        if syntax_error or text.encode("utf-8") != raw:
            # Preserve partial-tree import syntax in invalid Python as well.
            local_imports = _ImportFacts()
            parsed = _treesitter_facts(text, lang, raw=raw, import_facts=local_imports)
            local_imports.complete &= parsed is not None
            import_method = "tree-sitter/python" if parsed is not None else ""
        return FileFact(
            rel, digest, len(raw), lang, token_est,
            syntax_error=syntax_error,
            symbols=symbols,
            imports=imports,
            call_sites=calls,
            import_facts=local_imports.rows,
            import_facts_complete=local_imports.complete,
            import_facts_method=import_method,
        )
    if lang in _TS_LANGUAGES:
        ts = _treesitter_facts(text, lang, raw=raw, import_facts=local_imports)
        if ts is not None:
            symbols, ts_imports, syntax_error, calls = ts
            # JS/TS keep their dedicated import + route extraction (route detection
            # is framework-shaped, not a tags concern); symbols/calls now come from
            # tree-sitter instead of the old per-line regex.
            if lang in {"javascript", "typescript"}:
                imports = extract_js_imports(text)
                routes = extract_routes(text)[:50]
            else:
                imports = ts_imports
                routes = []
            return FileFact(
                rel, digest, len(raw), lang, token_est,
                syntax_error=syntax_error, symbols=symbols, imports=imports,
                routes=routes, call_sites=calls,
                import_facts=local_imports.rows,
                import_facts_complete=local_imports.complete,
                import_facts_method=f"tree-sitter/{_TS_LANGUAGES[lang]}",
            )
        # A known code language but no grammar/tree-sitter available: surface a
        # VISIBLE structural-unavailable disposition instead of silently guessing.
        return FileFact(rel, digest, len(raw), lang, token_est, disposition=f"structural_unavailable:{lang}")
    # Unknown suffixes still participate in text evidence, but no outline parser
    # ran. Do not report an empty outline as successfully indexed source.
    return FileFact(rel, digest, len(raw), lang, token_est, disposition=f"structural_unavailable:{lang}")


def _is_sensitive_inventory_path(rel_path: str) -> bool:
    rel = str(rel_path or "").replace("\\", "/")
    name = pathlib.PurePosixPath(rel).name
    lower = name.lower()
    if lower == ".env" or lower.startswith(".env."):
        return True
    suffix = pathlib.PurePosixPath(rel).suffix.lower()
    return suffix in _SENSITIVE_EXTENSIONS and bool(_SENSITIVE_NAME_RE.search(name))


def _is_excluded_inventory_path(path: pathlib.Path, excluded_paths: list[pathlib.Path]) -> bool:
    try:
        resolved = pathlib.Path(path).resolve(strict=False)
    except Exception:
        return False
    for excluded in excluded_paths:
        if resolved == excluded:
            return True
        try:
            if excluded.is_dir():
                resolved.relative_to(excluded)
                return True
        except Exception:
            continue
    return False


def build_code_inventory(
    repo_root: pathlib.Path,
    *,
    scope: str = "",
    drive_root: pathlib.Path | None = None,
    persist: bool = True,
    exclude_paths: Iterable[pathlib.Path] | None = None,
    path_allowed: Callable[[pathlib.Path], bool] | None = None,
) -> CodeInventory:
    root = pathlib.Path(repo_root).resolve(strict=False)
    deadline = time.monotonic() + _MAX_ENUMERATION_SECONDS
    cached = load_cached_inventory(root, drive_root) if drive_root is not None else None
    cached_by_path = {file.path: file for file in (cached.files if cached else [])}
    excluded_paths = [
        pathlib.Path(path).expanduser().resolve(strict=False)
        for path in (exclude_paths or [])
    ]
    filtered = pathlib.PurePosixPath(str(scope or "").replace("\\", "/")) != pathlib.PurePosixPath(".")

    def visible(path: pathlib.Path) -> bool:
        nonlocal filtered
        allowed = (not _is_excluded_inventory_path(path, excluded_paths)
                   and (path_allowed is None or path_allowed(path)))
        if not allowed:
            filtered = True
        return allowed

    paths, nested_roots, enumeration_limits = _enumerate_files(root, scope=scope, path_allowed=visible)
    files = []
    for path in paths:
        if time.monotonic() >= deadline:
            enumeration_limits.append(f"inventory_time_limit:{_MAX_ENUMERATION_SECONDS:g}s")
            break
        if not path.is_file() and not path.is_symlink():
            continue
        rel = path.relative_to(root).as_posix()
        # Check sensitivity, size and root containment before cache validation.
        # An oversized cached file must never trigger an unbounded hash read.
        raw = None
        cached_file = cached_by_path.get(rel)
        try:
            contained = path.resolve(strict=False).is_relative_to(root)
            if (cached_file is not None and contained and not _is_sensitive_inventory_path(rel)
                    and path.stat().st_size <= _MAX_INDEX_FILE_BYTES):
                with path.open("rb") as source:
                    raw = source.read(_MAX_INDEX_FILE_BYTES + 1)
        except OSError:
            pass
        if (cached_file is not None and raw is not None
                and len(raw) <= _MAX_INDEX_FILE_BYTES
                and cached_file.language == _language(path)
                and (cached_file.disposition != "indexed"
                     or cached_file.language == "python" or cached_file.language in _TS_LANGUAGES)
                and cached_file.sha256 == hashlib.sha256(raw).hexdigest()):
            files.append(cached_file)
        else:
            files.append(_file_fact(root, path))
    coverage: Dict[str, int] = {}
    for file in files:
        coverage[file.disposition] = coverage.get(file.disposition, 0) + 1
    inventory = CodeInventory(
        schema_version=CODE_INTELLIGENCE_SCHEMA_VERSION,
        repo_root=str(root),
        git_head=_git_head(root),
        created_at=utc_now_iso(),
        files=files,
        coverage=coverage,
        nested_roots=nested_roots,
        enumeration_limits=enumeration_limits,
    )
    # A caller-scoped view must not replace the shared full-source cache.
    if persist and drive_root is not None and not filtered:
        path = inventory_cache_path(root, pathlib.Path(drive_root))
        atomic_write_json(path, inventory.to_json(), trailing_newline=True)
    return inventory


def _in_scope(path: str, scope: str) -> bool:
    scope = str(scope or "").strip().replace("\\", "/").rstrip("/")
    if scope.startswith("./"):
        scope = scope[2:]
    return scope in {"", "."} or path == scope or path.startswith(scope + "/")


def render_digest_entry(file: FileFact) -> str:
    """Render one bounded overview entry, including unavailable-file status."""
    parts = [f"== {file.path} ({file.size} bytes, {file.language}) =="]
    if file.disposition != "indexed":
        parts.append(f"  Status: {file.disposition}")
    if file.syntax_error:
        parts.append(f"  Syntax: {file.syntax_error}")
    for label, items, limit in (
        ("Symbols", [symbol.name for symbol in file.symbols], 20),
        ("Imports", file.imports, 12),
        ("Routes", file.routes, 12),
        ("Calls", [call.name for call in file.call_sites], 12),
    ):
        if items:
            value = ", ".join(items[:limit])
            if len(items) > limit:
                value += f", ... ({len(items)} total)"
            parts.append(f"  {label}: {value}")
    return "\n".join(parts)


def digest_rollups(inventory: CodeInventory, scope: str = "") -> Dict[str, Any]:
    """Scope totals independent of the selected page; no completeness claim."""
    files = [file for file in inventory.files if _in_scope(file.path, scope)]
    languages: Dict[str, int] = {}
    dispositions: Dict[str, int] = {}
    for file in files:
        languages[file.language] = languages.get(file.language, 0) + 1
        dispositions[file.disposition] = dispositions.get(file.disposition, 0) + 1
    indexed = [file for file in files if file.disposition == "indexed"]
    return {
        "files": len(files), "indexed": len(indexed),
        "symbols": sum(len(file.symbols) for file in indexed),
        "line_est": sum(max(1, file.token_estimate // 20) for file in indexed),
        "languages": dict(sorted(languages.items())),
        "dispositions": dict(sorted(dispositions.items())),
    }


def render_codebase_digest(inventory: CodeInventory) -> str:
    """Compatibility whole-inventory renderer; query_code pages entries itself."""
    totals = digest_rollups(inventory)
    return (
        f"Codebase Digest ({totals['files']} files, ~{totals['line_est']} line-est, "
        f"{totals['symbols']} symbols, head={inventory.git_head[:12] or 'unknown'})\n"
        + "\n\n".join(render_digest_entry(file) for file in inventory.files)
    )


def symbol_definitions(inventory: CodeInventory, name: str = "", *, path: str = "", kind: str = "any") -> list[tuple[FileFact, SymbolFact]]:
    matches: list[tuple[FileFact, SymbolFact]] = []
    for file in inventory.files:
        if file.disposition != "indexed":
            continue
        if not _in_scope(file.path, path):
            continue
        for symbol in file.symbols:
            if name and symbol.name != name:
                continue
            if kind and kind != "any" and symbol.kind != kind:
                continue
            matches.append((file, symbol))
    return sorted(matches, key=lambda item: (item[0].path, item[1].line_start, item[1].name))


def relevant_files(inventory: CodeInventory, query: str, *, path: str = "", limit: int | None = 40) -> list[tuple[FileFact, float, str]]:
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(query or ""))
    }
    scored: list[tuple[FileFact, float, str]] = []
    for file in inventory.files:
        if not _in_scope(file.path, path):
            continue
        if file.disposition != "indexed" and not file.disposition.startswith("structural_unavailable:"):
            continue
        reasons: list[str] = []
        score = 0.0
        path_text = file.path.casefold()
        path_hits = sorted(token for token in tokens if token in path_text)
        if path_hits:
            score += 2.0 * len(path_hits)
            reasons.append("path:" + ",".join(path_hits[:3]))
        symbol_hits = sorted({sym.name for sym in file.symbols if sym.name.casefold() in tokens})
        if symbol_hits:
            score += 5.0 * len(symbol_hits)
            reasons.append("symbols:" + ",".join(symbol_hits[:3]))
        import_hits = sorted({imp for imp in file.imports if any(token in imp.casefold() for token in tokens)})
        if import_hits:
            score += 1.5 * len(import_hits)
            reasons.append("imports")
        route_hits = sorted({route for route in file.routes if any(token in route.casefold() for token in tokens)})
        if route_hits:
            score += 3.0 * len(route_hits)
            reasons.append("routes")
        if file.path.startswith("tests/") and path_hits:
            score += 1.0
            reasons.append("test")
        if score > 0:
            scored.append((file, score, "; ".join(reasons)))
    scored.sort(key=lambda item: (-item[1], item[0].path))
    return scored if limit is None else scored[: max(1, int(limit or 40))]
