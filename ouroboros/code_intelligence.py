"""Internal deterministic code inventory v2.

No embeddings, no LSP, no SQLite, and no raw source cache. This is a compact
structural projection used by digest/review context builders and read-only
code-query tools. The persisted JSON index is additive and derived-only:
source text is read for parsing, never cached.
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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List

from ouroboros.utils import atomic_write_json, utc_now_iso


CODE_INTELLIGENCE_SCHEMA_VERSION = 2

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
class ReferenceFact:
    name: str
    line: int
    enclosing: str = ""


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
    resolved_import_paths: List[str] = field(default_factory=list)
    routes: List[str] = field(default_factory=list)
    call_sites: List[CallSiteFact] = field(default_factory=list)
    references: List[ReferenceFact] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)


@dataclass
class CodeInventory:
    schema_version: int
    repo_root: str
    git_head: str
    created_at: str
    files: List[FileFact]
    coverage: Dict[str, int]

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


def _reference_from_json(raw: Any) -> ReferenceFact:
    data = raw if isinstance(raw, dict) else {}
    return ReferenceFact(
        name=str(data.get("name") or ""),
        line=int(data.get("line") or 0),
        enclosing=str(data.get("enclosing") or ""),
    )


def _file_from_json(raw: Any) -> FileFact | None:
    if not isinstance(raw, dict):
        return None
    required = ("path", "sha256", "size", "language", "token_estimate")
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
            resolved_import_paths=sorted({str(item) for item in (raw.get("resolved_import_paths") or []) if str(item)}),
            routes=sorted({str(item) for item in (raw.get("routes") or []) if str(item)}),
            call_sites=[item for item in (_call_from_json(s) for s in raw.get("call_sites") or []) if item.name],
            references=[item for item in (_reference_from_json(s) for s in raw.get("references") or []) if item.name],
            exports=sorted({str(item) for item in (raw.get("exports") or []) if str(item)}),
        )
    except Exception:
        return None


def load_cached_inventory(repo_root: pathlib.Path, drive_root: pathlib.Path) -> CodeInventory | None:
    """Load a v2 derived inventory cache, returning None for v1/malformed data."""
    path = inventory_cache_path(repo_root, drive_root)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict) or int(raw.get("schema_version") or 0) < CODE_INTELLIGENCE_SCHEMA_VERSION:
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


def _tracked_files(repo_root: pathlib.Path) -> List[pathlib.Path]:
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=str(repo_root),
            capture_output=True,
            timeout=10,
        )
        if proc.returncode == 0:
            return [repo_root / part.decode("utf-8", errors="replace") for part in proc.stdout.split(b"\0") if part]
    except Exception:
        pass
    paths: List[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in sorted(dirnames) if d not in _SKIP_DIRS]
        for name in sorted(filenames):
            paths.append(pathlib.Path(dirpath) / name)
    return paths


def _language(path: pathlib.Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
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


def _python_facts(text: str, rel_path: pathlib.PurePosixPath) -> tuple[List[SymbolFact], List[str], str, List[CallSiteFact], List[ReferenceFact]]:
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return [], [], f"{exc.msg} at line {exc.lineno}", [], []
    symbols: List[SymbolFact] = []
    imports: List[str] = []
    calls: List[CallSiteFact] = []
    references: List[ReferenceFact] = []
    stack: list[tuple[ast.AST, str]] = [(tree, "")]
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
            child_enclosing = node.name
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
                calls.append(CallSiteFact(name, int(getattr(node, "lineno", 0) or 0), enclosing))
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            references.append(ReferenceFact(node.id, int(getattr(node, "lineno", 0) or 0), enclosing))
        elif isinstance(node, ast.Attribute):
            references.append(ReferenceFact(node.attr, int(getattr(node, "lineno", 0) or 0), enclosing))
        stack.extend((child, child_enclosing) for child in reversed(list(ast.iter_child_nodes(node))))
    symbols = sorted(symbols, key=lambda item: (item.line_start, item.name))
    calls = sorted(calls, key=lambda item: (item.line, item.enclosing, item.name))
    references = sorted(references, key=lambda item: (item.line, item.enclosing, item.name))
    return symbols, sorted(set(imports)), "", calls, references


def _resolve_python_import(repo_root: pathlib.Path, module: str) -> str:
    rel = pathlib.Path(*str(module or "").split("."))
    for candidate in (repo_root / rel.with_suffix(".py"), repo_root / rel / "__init__.py"):
        if candidate.is_file():
            try:
                return candidate.relative_to(repo_root).as_posix()
            except ValueError:
                return ""
    return ""


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
_TS_MAX_SYMBOLS = 4000


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


def _ts_callee_name(node: Any) -> str:
    for field_name in ("function", "name", "method"):
        target = node.child_by_field_name(field_name)
        if target is not None:
            if target.type in _TS_NAME_TYPES and target.text:
                return target.text.decode("utf-8", "replace").rsplit(".", 1)[-1].rsplit("::", 1)[-1]
            # member/scoped call (obj.method(), pkg.Func(), a::b()): the callee is
            # the FINAL identifier (the method/function), not the receiver/namespace.
            # Iterate in reverse so `obj.doThing()` -> doThing, `fmt.Sprintf()` ->
            # Sprintf (matches the Python ast path and the former JS regex).
            for child in reversed(target.children):
                if child.type in _TS_NAME_TYPES and child.text:
                    return child.text.decode("utf-8", "replace")
    return ""


def _treesitter_facts(text: str, lang: str):
    """Generic structural facts for a non-Python language via tree-sitter.

    Returns (symbols, imports, syntax_error, calls, references) or None when the
    grammar/library is unavailable (the caller then marks structural_unavailable).
    """
    grammar = _TS_LANGUAGES.get(lang)
    if not grammar:
        return None
    parser = _ts_parser(grammar)
    if parser is None:
        return None
    try:
        tree = parser.parse(text.encode("utf-8", "replace"))
    except Exception:
        return None
    symbols: List[SymbolFact] = []
    calls: List[CallSiteFact] = []
    imports: List[str] = []
    root = tree.root_node
    # Iterative DFS carrying the enclosing definition name (for call attribution).
    stack: list[tuple[Any, str]] = [(root, "")]
    while stack:
        node, enclosing = stack.pop()
        child_enclosing = enclosing
        ntype = node.type
        if ntype in _TS_DEF_KINDS:
            name = _ts_node_name(node)
            if name and len(symbols) < _TS_MAX_SYMBOLS:
                sig = (node.text.decode("utf-8", "replace").splitlines() or [""])[0].strip()[:200] if node.text else ""
                symbols.append(SymbolFact(name, _TS_DEF_KINDS[ntype], node.start_point[0] + 1, node.end_point[0] + 1, sig))
                child_enclosing = name
        elif ntype in _TS_CALL_TYPES:
            callee = _ts_callee_name(node)
            if callee:
                calls.append(CallSiteFact(callee, node.start_point[0] + 1, enclosing))
        elif ntype in _TS_IMPORT_TYPES and node.text:
            spec = (node.text.decode("utf-8", "replace").splitlines() or [""])[0].strip()[:200]
            if spec:
                imports.append(spec)
        stack.extend((child, child_enclosing) for child in reversed(node.children))
    syntax_error = "syntax error" if root.has_error else ""
    symbols.sort(key=lambda s: (s.line_start, s.name))
    calls.sort(key=lambda c: (c.line, c.enclosing, c.name))
    return symbols, sorted(set(imports)), syntax_error, calls, []


def _file_fact(repo_root: pathlib.Path, path: pathlib.Path) -> FileFact:
    try:
        rel = path.relative_to(repo_root).as_posix()
    except ValueError:
        try:
            rel = path.resolve(strict=False).relative_to(repo_root).as_posix()
        except ValueError:
            return FileFact(str(path), "", 0, _language(path), 0, disposition="path_escape")
    if path.is_symlink():
        try:
            path.resolve(strict=False).relative_to(repo_root)
        except ValueError:
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
        raw = path.read_bytes()
    except OSError as exc:
        return FileFact(rel, "", 0, _language(path), 0, disposition=f"read_error:{exc}")
    digest = hashlib.sha256(raw).hexdigest()
    lang = _language(path)
    token_est = max(1, len(raw) // 4)
    if b"\0" in raw[:4096]:
        return FileFact(rel, digest, len(raw), lang, token_est, disposition="binary")
    text = raw.decode("utf-8", errors="replace")
    if lang == "python":
        symbols, imports, syntax_error, calls, references = _python_facts(text, pathlib.PurePosixPath(rel))
        resolved = [p for p in (_resolve_python_import(repo_root, module) for module in imports) if p]
        return FileFact(
            rel,
            digest,
            len(raw),
            lang,
            token_est,
            syntax_error=syntax_error,
            symbols=symbols,
            imports=imports,
            resolved_import_paths=resolved,
            call_sites=calls,
            references=references,
        )
    if lang in _TS_LANGUAGES:
        ts = _treesitter_facts(text, lang)
        if ts is not None:
            symbols, ts_imports, syntax_error, calls, references = ts
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
                routes=routes, call_sites=calls, references=references,
            )
        # A known code language but no grammar/tree-sitter available: surface a
        # VISIBLE structural-unavailable disposition instead of silently guessing.
        return FileFact(rel, digest, len(raw), lang, token_est, disposition=f"structural_unavailable:{lang}")
    return FileFact(rel, digest, len(raw), lang, token_est)


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
    drive_root: pathlib.Path | None = None,
    persist: bool = True,
    exclude_paths: Iterable[pathlib.Path] | None = None,
) -> CodeInventory:
    root = pathlib.Path(repo_root).resolve(strict=False)
    cached = load_cached_inventory(root, drive_root) if drive_root is not None else None
    cached_by_path = {file.path: file for file in (cached.files if cached else [])}
    excluded_paths = [
        pathlib.Path(path).expanduser().resolve(strict=False)
        for path in (exclude_paths or [])
    ]
    files = []
    for path in _tracked_files(root):
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            rel_parts = path.parts
        if any(part in _SKIP_DIRS for part in rel_parts):
            continue
        if _is_excluded_inventory_path(path, excluded_paths):
            continue
        if path.is_file():
            rel = ""
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                pass
            try:
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except Exception:
                digest = ""
            cached_file = cached_by_path.get(rel)
            if cached_file is not None and digest and cached_file.sha256 == digest:
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
    )
    if persist and drive_root is not None:
        path = inventory_cache_path(root, pathlib.Path(drive_root))
        atomic_write_json(path, inventory.to_json(), trailing_newline=True)
    return inventory


def render_codebase_digest(inventory: CodeInventory) -> str:
    lines: List[str] = []
    total_lines_est = 0
    total_symbols = 0
    for file in inventory.files:
        if file.disposition != "indexed":
            continue
        line_est = max(1, file.token_estimate // 20)
        total_lines_est += line_est
        total_symbols += len(file.symbols)
        parts = [f"\n== {file.path} ({file.size} bytes, {file.language}) =="]
        if file.symbols:
            names = ", ".join(symbol.name for symbol in file.symbols[:20])
            if len(file.symbols) > 20:
                names += f", ... ({len(file.symbols)} total)"
            parts.append(f"  Symbols: {names}")
        if file.imports:
            imports = ", ".join(file.imports[:12])
            if len(file.imports) > 12:
                imports += f", ... ({len(file.imports)} total)"
            parts.append(f"  Imports: {imports}")
        if file.routes:
            parts.append("  Routes: " + ", ".join(file.routes[:12]))
        if file.call_sites:
            calls = ", ".join(call.name for call in file.call_sites[:12])
            if len(file.call_sites) > 12:
                calls += f", ... ({len(file.call_sites)} total)"
            parts.append(f"  Calls: {calls}")
        lines.append("\n".join(parts))
    return (
        f"Codebase Digest ({len(inventory.files)} files, ~{total_lines_est} line-est, "
        f"{total_symbols} symbols, head={inventory.git_head[:12] or 'unknown'})\n"
        + "\n".join(lines)
    )


def symbol_definitions(inventory: CodeInventory, name: str = "", *, path: str = "", kind: str = "any") -> list[tuple[FileFact, SymbolFact]]:
    matches: list[tuple[FileFact, SymbolFact]] = []
    path_filter = str(path or "").strip().replace("\\", "/")
    for file in inventory.files:
        if file.disposition != "indexed":
            continue
        if path_filter and not (file.path == path_filter or file.path.startswith(path_filter.rstrip("/") + "/")):
            continue
        for symbol in file.symbols:
            if name and symbol.name != name:
                continue
            if kind and kind != "any" and symbol.kind != kind:
                continue
            matches.append((file, symbol))
    return sorted(matches, key=lambda item: (item[0].path, item[1].line_start, item[1].name))


def symbol_references(inventory: CodeInventory, name: str, *, path: str = "") -> list[tuple[FileFact, ReferenceFact]]:
    definition_paths = {file.path for file, _ in symbol_definitions(inventory, name, path=path)}
    matches: list[tuple[FileFact, ReferenceFact]] = []
    for file in inventory.files:
        if file.disposition != "indexed":
            continue
        for ref in file.references:
            if ref.name != name:
                continue
            if definition_paths and file.path in definition_paths and ref.line in {
                symbol.line_start for def_file, symbol in symbol_definitions(inventory, name, path=path) if def_file.path == file.path
            }:
                continue
            matches.append((file, ref))
    return sorted(matches, key=lambda item: (item[0].path, item[1].line, item[1].enclosing))


def symbol_callers(inventory: CodeInventory, name: str, *, path: str = "") -> list[tuple[FileFact, CallSiteFact]]:
    # Best-effort: path disambiguates definitions, but dynamic method resolution is intentionally approximate.
    if path and not {file.path for file, _ in symbol_definitions(inventory, name, path=path)}:
        return []
    matches: list[tuple[FileFact, CallSiteFact]] = []
    for file in inventory.files:
        if file.disposition != "indexed":
            continue
        for call in file.call_sites:
            if call.name == name:
                matches.append((file, call))
    return sorted(matches, key=lambda item: (item[0].path, item[1].line, item[1].enclosing))


def symbol_callees(inventory: CodeInventory, name: str, *, path: str = "") -> list[tuple[FileFact, CallSiteFact]]:
    definition_files = {file.path for file, _ in symbol_definitions(inventory, name, path=path)}
    matches: list[tuple[FileFact, CallSiteFact]] = []
    for file in inventory.files:
        if file.disposition != "indexed":
            continue
        if definition_files and file.path not in definition_files:
            continue
        for call in file.call_sites:
            if call.enclosing == name or call.enclosing.endswith(f".{name}"):
                matches.append((file, call))
    return sorted(matches, key=lambda item: (item[0].path, item[1].line, item[1].name))


def impact_files(inventory: CodeInventory, target: str, *, depth: int = 1) -> list[tuple[FileFact, str]]:
    depth = max(1, min(5, int(depth or 1)))
    target_text = str(target or "").strip().replace("\\", "/")
    impacted: dict[str, str] = {}
    frontier: set[str] = set()
    if target_text.endswith((".py", ".js", ".jsx", ".ts", ".tsx")) or "/" in target_text:
        frontier.add(target_text)
    else:
        frontier.update(file.path for file, _ in symbol_definitions(inventory, target_text))
    path_to_file = {file.path: file for file in inventory.files}
    for hop in range(depth):
        next_frontier: set[str] = set()
        for file in inventory.files:
            if file.disposition != "indexed":
                continue
            if set(file.resolved_import_paths) & frontier:
                impacted.setdefault(file.path, f"imports depth {hop + 1}")
                next_frontier.add(file.path)
            if target_text and any(ref.name == target_text for ref in file.references):
                impacted.setdefault(file.path, f"references {target_text}")
                next_frontier.add(file.path)
        frontier = next_frontier
        if not frontier:
            break
    for path in sorted(frontier):
        impacted.setdefault(path, "target")
    return [(path_to_file[path], reason) for path, reason in sorted(impacted.items()) if path in path_to_file]


def relevant_files(inventory: CodeInventory, query: str, *, limit: int = 40) -> list[tuple[FileFact, float, str]]:
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", str(query or ""))
    }
    scored: list[tuple[FileFact, float, str]] = []
    for file in inventory.files:
        if file.disposition != "indexed":
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
    return scored[: max(1, int(limit or 40))]
