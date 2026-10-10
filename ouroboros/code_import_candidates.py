"""Import syntax and query-time filesystem candidates, never name resolution.

Only recognized grammar shapes produce specifiers. Python ``from`` syntax emits
the module and possible imported submodules; Rust use lists are expanded from
their syntax. Neither operation binds an imported name, follows aliases, consults
package manifests, or evaluates expressions. Unknown grammar shapes yield no
specifier, so callers must disclose extraction coverage rather than infer that
there are no dependencies.

The matcher accepts an already-selected file universe and stores nothing. Its
relative/suffix conventions deliberately retain all matches, including extension
and index siblings. JS runtime extensions may name TS source siblings. Rust use
paths may end in symbols, so every module prefix is a candidate. These are search
heuristics, not compiler or package-manager rules, even for a unique match.
"""

from __future__ import annotations

import posixpath
from pathlib import PurePosixPath
from typing import Any, Iterable

_LANGUAGES = {"py": "python", "js": "javascript", "jsx": "javascript",
              "ts": "typescript", "tsx": "typescript", "rs": "rust"}
_EXTENSIONS = (".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx",
               ".mts", ".cts", ".go", ".java", ".rs", ".rb", ".c", ".h",
               ".cpp", ".hpp", ".cs", ".kt", ".swift", ".php", ".lua")
_SOURCE_SIBLINGS = {".js": (".ts", ".tsx"), ".jsx": (".tsx",),
                    ".mjs": (".mts",), ".cjs": (".cts",)}


def _text(node: Any) -> str:
    raw = getattr(node, "text", b"") or b""
    return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)


def _field(node: Any, name: str) -> Any:
    return node.child_by_field_name(name)


def _literal(node: Any) -> str:
    """Keep literal contents raw: escapes/interpolations are not evaluated."""
    if node is None or node.type not in {
        "string", "string_literal", "interpreted_string_literal", "raw_string_literal",
    }:
        return ""
    text = _text(node)
    if len(text) >= 2 and text[0] in "\"'`" and text[-1] == text[0]:
        return text[1:-1]
    return ""


def _literal_call(node: Any) -> bool:
    if node.type != "call_expression":
        return False
    callee = _field(node, "function")
    if callee is None or callee.type not in {"identifier", "import"}:
        return False
    if _text(callee) not in {"require", "import"}:
        return False
    arguments = _field(node, "arguments")
    return arguments is not None and bool(arguments.named_children) and bool(
        _literal(arguments.named_children[0]))


def is_import_node(node: Any, language: str = "") -> bool:
    """Select one node per statement/specifier, avoiding Go parent/child doubles.

    ``require`` is recognized only by its syntactic callee spelling; it may be
    shadowed. Its presence does not assert that a module loader will run.
    """
    lang = _LANGUAGES.get(language, language)
    kind = node.type
    if kind in {"import_statement", "import_from_statement", "use_declaration", "import_spec"}:
        return True
    if kind == "import_declaration":
        # Go's declaration wraps import_spec; Java's declaration is the unit.
        return not any(c.type in {"import_spec", "import_spec_list"} for c in node.named_children)
    if kind == "export_statement":
        return _field(node, "source") is not None
    return lang in {"", "javascript", "typescript"} and _literal_call(node)


def _python_name(node: Any) -> str:
    if node.type == "aliased_import":
        node = _field(node, "name")
    return _text(node) if node is not None else ""


def _rust_specs(node: Any, prefix: str = "") -> list[str]:
    """Expand use-tree grammar nodes; aliases are discarded, not interpreted."""
    kind = node.type
    if kind == "use_as_clause":
        path = _field(node, "path")
        return _rust_specs(path, prefix) if path is not None else []
    if kind == "scoped_use_list":
        path, items = _field(node, "path"), _field(node, "list")
        base = "::".join(p for p in (prefix, _text(path)) if p)
        return _rust_specs(items, base) if items is not None else []
    if kind == "use_list":
        return [spec for child in node.named_children for spec in _rust_specs(child, prefix)]
    if kind == "use_wildcard":
        # The '*' child is unnamed; its optional named child is the path.
        path = next(iter(node.named_children), None)
        return ["::".join(p for p in (prefix, _text(path)) if p)]
    if kind in {"identifier", "scoped_identifier", "crate", "self", "super"}:
        value = _text(node)
        if value == "self" and prefix:
            return [prefix]
        return ["::".join(p for p in (prefix, value) if p)]
    return []


def import_specs(node: Any, language: str = "") -> list[str]:
    """Return normalized source specifiers for one selected syntax node.

    Python ``from pkg import x`` yields ``pkg`` and ``pkg.x``: the latter is a
    possible submodule path, not an assertion that ``x`` is a module. Relative
    dots are retained. Java static imports also expose their enclosing type.
    Literal escape sequences remain unexpanded and may therefore not match a
    filename. Computed imports and template strings are not extracted.
    """
    if not is_import_node(node, language):
        return []
    lang = _LANGUAGES.get(language, language)
    kind = node.type
    specs: list[str] = []
    if kind == "import_from_statement":
        module = _field(node, "module_name")
        base = _text(module)
        if base:
            specs.append(base)
            for child in node.named_children:
                if child == module or child.type not in {"dotted_name", "aliased_import"}:
                    continue
                name = _python_name(child)
                if name:
                    specs.append(base + ("" if base.endswith(".") else ".") + name)
    elif kind == "import_statement" and (lang == "python" or any(
        c.type in {"dotted_name", "aliased_import"} for c in node.named_children
    )):
        specs = [_python_name(c) for c in node.named_children
                 if c.type in {"dotted_name", "aliased_import"}]
    elif kind in {"import_statement", "export_statement", "import_spec"}:
        source = _field(node, "source") if kind != "import_spec" else _field(node, "path")
        if source is None:
            # TypeScript's `import x = require("x")` has a nested source field.
            clause = next((c for c in node.named_children if c.type == "import_require_clause"), None)
            source = _field(clause, "source") if clause is not None else None
        specs = [_literal(source)]
    elif kind == "call_expression":
        specs = [_literal(_field(node, "arguments").named_children[0])]
    elif kind == "use_declaration":
        argument = _field(node, "argument")
        specs = _rust_specs(argument) if argument is not None else []
    elif kind == "import_declaration" and lang in {"", "java"}:
        name = next((c for c in node.named_children if c.type in {"identifier", "scoped_identifier"}), None)
        if name is not None:
            specs = [_text(name)]
            if any(c.type == "static" for c in node.children) and not any(c.type == "asterisk" for c in node.named_children):
                parent = _field(name, "scope")
                if parent is not None:
                    specs.append(_text(parent))
    return sorted({spec for spec in specs if spec})


def _safe_path(path: str) -> str:
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return ""
    normalized = posixpath.normpath(path)
    if normalized == ".." or normalized.startswith("../"):
        return ""
    return normalized


def _variants(base: str) -> set[str]:
    variants = {base}
    ext = PurePosixPath(base).suffix
    if ext in _EXTENSIONS:
        variants.update(base[:-len(ext)] + sibling for sibling in _SOURCE_SIBLINGS.get(ext, ()))
    else:
        if base != ".":
            variants.update(base + suffix for suffix in _EXTENSIONS)
        variants.update(posixpath.join(base, "index" + suffix) for suffix in _EXTENSIONS)
        variants.add(posixpath.join(base, "__init__.py"))
        variants.add(posixpath.join(base, "mod.rs"))
    return {posixpath.normpath(path) for path in variants}


class ImportCandidateLookup:
    """Request-local path membership, including every complete suffix boundary.

    Normalize the current universe once, not once per import specifier. Values
    retain the original paths and all siblings; this is not a resolved graph.
    """

    def __init__(self, paths: Iterable[str]):
        self.relative: dict[str, set[str]] = {}
        self.suffix: dict[str, set[str]] = {}
        for path in paths:
            clean = _safe_path(path)
            if not clean:
                continue
            self.relative.setdefault(clean, set()).add(path)
            while clean:
                self.suffix.setdefault(clean, set()).add(path)
                _, separator, clean = clean.partition("/")
                if not separator:
                    break

    def matches(self, variants: set[str], relative: bool) -> list[str]:
        lookup = self.relative if relative else self.suffix
        return sorted({path for variant in variants for path in lookup.get(variant, ())})


def import_candidates(
    spec: str, importer_path: str, paths: Iterable[str] | ImportCandidateLookup, language: str = "",
) -> tuple[str, list[str]]:
    """Return ``(relative|suffix, paths)`` from this call's file universe.

    Paths must be repository-relative POSIX names. Root escapes are rejected.
    Bare modules use suffix matching (including monorepo ambiguity); relative
    paths never fall back to suffix matching. Package exports, Go module maps,
    aliases, Python sys.path, Rust cfg/macros and Java classpaths are not read.
    """
    lang = _LANGUAGES.get(language, language)
    if not lang:
        lang = _LANGUAGES.get(PurePosixPath(importer_path).suffix.lstrip("."), "")
    spec = spec.strip()
    python_relative = lang == "python" and spec.startswith(".")
    rust_relative = lang == "rust" and spec.startswith(("self::", "super::"))
    relative = spec.startswith(("./", "../")) or python_relative or rust_relative
    kind = "relative" if relative else "suffix"
    importer = _safe_path(importer_path)
    if not spec or not importer or spec.startswith("/") or "\\" in spec or "\x00" in spec:
        return kind, []
    directory = posixpath.dirname(importer)
    if python_relative:
        dots = len(spec) - len(spec.lstrip("."))
        spec = "../" * (dots - 1) + spec[dots:].replace(".", "/")
    elif lang in {"python", "java"}:
        spec = spec.replace(".", "/")
    elif lang == "rust":
        parts = spec.removeprefix("::").split("::")
        if parts[0] == "crate":
            parts = parts[1:]
        elif rust_relative:
            # Ordinary foo.rs owns module foo/, whereas mod.rs/lib.rs/main.rs
            # owns its directory. This is only the conventional source layout.
            if PurePosixPath(importer).stem not in {"mod", "lib", "main"}:
                directory = posixpath.splitext(importer)[0]
            if parts[0] == "self":
                parts = parts[1:]
            while parts and parts[0] == "super":
                directory += "/.."
                parts = parts[1:]
        spec = "/".join(parts)
    base = _safe_path((posixpath.join(directory, spec) or ".") if relative else spec)
    if not base:
        return kind, []
    bases = {base}
    if lang == "rust":
        # A use path can end in a type/function. Retain each possible module
        # prefix, without deciding which segment denotes a module.
        floor = _safe_path(directory) if relative else ""
        parent = posixpath.dirname(base)
        while parent and parent != "." and parent != floor:
            bases.add(parent)
            parent = posixpath.dirname(parent)
    variants = {v for candidate in bases for v in _variants(candidate)}
    lookup = paths if isinstance(paths, ImportCandidateLookup) else ImportCandidateLookup(paths)
    return kind, lookup.matches(variants, relative)
