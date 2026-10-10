"""On-demand source evidence, never symbol binding or persisted cross-file joins.

The inventory supplies paths and bounded local import facts; current source bytes
validate reuse and supply snippets. Token views parse only prefilter matches.
Filtering happens before the selected-view row cap:
thousands of comments cannot spend a caller's quota. Grammar slots remain visible
even where a familiar syntactic role can be suggested. Pages rescan, not snapshot.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from ouroboros import code_intelligence as ci
from ouroboros.code_import_candidates import import_specs, is_import_node
from ouroboros.code_search_rg import MAX_FILE_SIZE_BYTES, search_skip_reason

MAX_EVIDENCE_ROWS = 20000


def grammar_id(language: str) -> str:
    return ci._TS_LANGUAGES.get(language, language)


def in_scope(path: str, scope: str) -> bool:
    return not scope or path == scope or path.startswith(scope.rstrip("/") + "/")


@dataclass
class EvidenceRow:
    path: str
    line: int
    column: int
    slot: str
    source: str
    hint: str = ""
    enclosing: str = ""
    specifier: str = ""
    language: str = ""

    def render(self, label: str = "") -> str:
        context = f" in {self.enclosing}" if self.enclosing else ""
        tags = " ".join(part for part in (self.hint, label) if part)
        anchor = f"{self.path}:{self.line}" + (f":{self.column}" if self.column else "")
        return f"{anchor} {self.slot}{' ' + tags if tags else ''}{context} | {self.source}"


@dataclass
class EvidenceScan:
    rows: list[EvidenceRow] = field(default_factory=list)
    files_selected: int = 0
    files_scanned: int = 0
    files_matched: int = 0
    files_parsed: int = 0
    occurrences: int = 0
    limits: Counter = field(default_factory=Counter)
    methods: set[str] = field(default_factory=set)
    incomplete: bool = False

    def summary(self) -> str:
        return (f"method: {', '.join(sorted(self.methods)) or 'literal prefilter'}; "
                f"files: {self.files_selected} selected, {self.files_scanned} text-scanned, "
                f"{self.files_matched} matched, {self.files_parsed} parsed")


def _slot(node: Any) -> str:
    parent = node.parent
    if parent is None:
        return node.type
    for index, child in enumerate(parent.children):
        if child == node:
            name = parent.field_name_for_child(index)
            return parent.type + ("." + name if name else "")
    return parent.type


def _source_lines(text: str, *, python_ast: bool = False) -> list[str]:
    """Lines in the coordinate model of the anchor being rendered.

    Tree-sitter rows and literal anchors count LF only (a CRLF's CR is trimmed);
    Python AST lines also end at a bare CR. ``str.splitlines`` would also split
    at form feed, vertical tab, NEL or U+2028 and shift every later slice.
    """
    if python_ast:
        return re.split(r"\r\n?|\n", text)
    return [line[:-1] if line.endswith("\r") else line for line in text.split("\n")]


def _source(lines: list[str], line: int, column: int) -> str:
    text = lines[line - 1] if 0 < line <= len(lines) else ""
    # Column is UTF-8 bytes, as in tree-sitter. Retain a literal source slice
    # surrounding a distant match rather than truncating away the query itself.
    char_column = len(text.encode("utf-8")[:max(0, column - 1)].decode("utf-8", "ignore"))
    start = max(0, char_column - 60)
    end = min(len(text), max(start + 180, char_column + 80))
    return ("…" if start else "") + text[start:end] + ("…" if end < len(text) else "")


def _tree_rows(tree: Any, file: Any, text: str, query: str | None, mode: str,
               deadline: float) -> tuple[list[EvidenceRow], set[tuple[int, int]], bool]:
    rows: list[EvidenceRow] = []
    covered: set[tuple[int, int]] = set()
    lines = _source_lines(text)
    callees: set[tuple[int, int]] = set()
    stack = [(tree.root_node, (), False)]
    visited = 0
    while stack:
        node, ancestors, importing = stack.pop()
        visited += 1
        if visited % 256 == 0 and time.monotonic() > deadline:
            return rows, covered, True
        import_node = is_import_node(node, file.language)
        if mode == "imports" and import_node:
            for specifier in import_specs(node, file.language):
                line, col = node.start_point
                rows.append(EvidenceRow(file.path, line + 1, col + 1, node.type,
                                        _source(lines, line + 1, col + 1), "import?",
                                        " / ".join(ancestors[-2:]), specifier, file.language))
        if mode != "imports" and node.type in ci._TS_CALL_TYPES:
            callee = ci._ts_callee_leaf(node)
            if callee is not None:
                callees.add((callee.start_byte, callee.end_byte))
        if mode != "imports" and node.child_count == 0 and node.is_named:
            value = (node.text or b"").decode("utf-8", "replace")
            if query is None or value == query:
                line, col = node.start_point
                covered.add((line + 1, col + 1))
                slot = _slot(node)
                is_call = (node.start_byte, node.end_byte) in callees
                definition = node.parent is not None and node.parent.type in ci._TS_DEF_KINDS and slot.endswith(".name")
                hint = "call?" if is_call else "import?" if importing else "def?" if definition else ""
                if mode not in {"callers", "callees"} or is_call:
                    if mode != "definition" or slot.endswith(".name"):
                        rows.append(EvidenceRow(file.path, line + 1, col + 1, slot,
                                                _source(lines, line + 1, col + 1), hint,
                                                " / ".join(ancestors[-2:])))
        child_ancestors = ancestors
        if node.type in ci._TS_DEF_KINDS:
            name = node.child_by_field_name("name")
            if name is not None:
                child_ancestors = (*ancestors, name.text.decode("utf-8", "replace"))
        stack.extend((child, child_ancestors, importing or import_node)
                     for child in reversed(node.children))
    return rows, covered, False


def scan_occurrences(inventory: ci.CodeInventory, query: str | None, *, mode: str = "references",
                     scope: str = "", lang: str = "any", ranges: dict | None = None,
                     wall_seconds: float = 45, max_rows: int = MAX_EVIDENCE_ROWS,
                     deadline: float | None = None,
                     select_rows: Callable[[list[EvidenceRow]], list[EvidenceRow]] | None = None,
                     path_allowed: Callable[[pathlib.Path], bool] | None = None) -> EvidenceScan:
    """Select operation rows before counting/paging; no semantic alias following.

    Only selected rows consume max_rows. Text-only rows are eligible exclusively
    for references. Missing grammar in callers is therefore an explicit limit,
    never evidence that the file has no calls. Import extraction is syntactic;
    candidate joins happen separately over the current complete path list.
    Complete local import facts are reused only for these exact source bytes;
    legacy, overflowed or changed facts require a fresh parse.
    Inventory call facts and definition ranges also apply only to the bytes the
    inventory hashed: a changed file is disclosed without them; others still scan.
    A view may select/summarize each file's rows before the selected-result cap.
    """
    scan = EvidenceScan()
    stale = False
    deadline = deadline if deadline is not None else time.monotonic() + wall_seconds
    root = pathlib.Path(inventory.repo_root)
    pattern = re.compile(r"(?<![\w$])" + re.escape(query) + r"(?![\w$])") if query else None
    files = [f for f in inventory.files if in_scope(f.path, scope)
             and (lang == "any" or grammar_id(f.language) == grammar_id(lang))
             and (ranges is None or f.path in ranges)]
    scan.files_selected = len(files)
    for file in sorted(files, key=lambda f: f.path):
        if time.monotonic() > deadline:
            scan.limits["wall budget hit; remaining files not scanned"] += 1
            scan.incomplete = True
            break
        path = root / file.path
        if path_allowed is not None and not path_allowed(path):
            scan.limits["policy excluded"] += 1
            continue
        reason = search_skip_reason(path)
        if not reason and file.disposition not in {"indexed"} and not file.disposition.startswith("structural_unavailable"):
            reason = file.disposition
        if reason:
            scan.limits[reason] += 1
            continue
        try:
            with path.open("rb") as source:
                raw = source.read(MAX_FILE_SIZE_BYTES + 1)
        except OSError:
            scan.limits["read error"] += 1
            continue
        if len(raw) > MAX_FILE_SIZE_BYTES:
            scan.limits["oversized"] += 1
            continue
        if b"\0" in raw:
            scan.limits["binary"] += 1
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", "replace")
            scan.limits["invalid UTF-8 replaced"] += 1
        scan.files_scanned += 1
        matches = list(pattern.finditer(text)) if pattern else []
        if pattern and not matches:
            continue
        scan.files_matched += 1
        scan.occurrences += len(matches)
        grammar = grammar_id(file.language)
        if mode == "imports" and grammar not in {"python", "javascript", "typescript", "go", "java", "rust"}:
            scan.limits[f"import specifier syntax unverified: {grammar}"] += 1
        cached_imports = (mode == "imports" and file.import_facts_complete
                          and file.import_facts_method
                          and file.sha256 == hashlib.sha256(raw).hexdigest())
        parser = (ci._ts_parser(grammar) if not cached_imports
                  and (grammar == "python" or file.language in ci._TS_LANGUAGES) else None)
        tree = None
        if parser is not None:
            try:
                tree = parser.parse(raw)
            except Exception:
                scan.limits["parser failed"] += 1
        rows: list[EvidenceRow] = []
        covered: set[tuple[int, int]] = set()
        if cached_imports:
            lines = _source_lines(text, python_ast=file.import_facts_method == "python ast")
            rows = [EvidenceRow(file.path, fact.line, fact.column, fact.slot,
                                _source(lines, fact.line, fact.column), "import?",
                                fact.enclosing, fact.specifier, file.language)
                    for fact in file.import_facts]
            scan.methods.add(f"{file.import_facts_method} local imports (hash-validated)")
            if file.syntax_error:
                scan.limits["syntax error (partial import facts)"] += 1
        elif tree is not None:
            scan.files_parsed += 1
            scan.methods.add(f"tree-sitter/{grammar}")
            if tree.root_node.has_error:
                scan.limits["syntax error (partial tree)"] += 1
            rows, covered, stopped = _tree_rows(tree, file, text, query, mode, deadline)
            if stopped:
                scan.limits["wall budget hit within syntax tree"] += 1
                scan.incomplete = True
        else:
            scan.limits[f"no grammar: {file.language} (text only)"] += 1
            # Keep useful Python local calls if tree-sitter is unavailable. These
            # need validation against this read, not only the inventory read:
            # edits can happen between inventory construction and this scan.
            if mode in {"callers", "callees"} and file.language == "python" and not file.syntax_error:
                if file.sha256 != hashlib.sha256(raw).hexdigest():
                    scan.limits["file changed since inventory; Python call facts not used"] += 1
                    stale = True
                else:
                    lines = _source_lines(text, python_ast=True)
                    rows = [EvidenceRow(file.path, c.line, 0, "ast.Call.func", _source(lines, c.line, 1),
                                        "call?", c.enclosing) for c in file.call_sites if query is None or c.name == query]
                    scan.methods.add("python ast local calls")
        if mode == "references":
            lines = _source_lines(text)
            text_lines: set[int] = set()
            # Map offsets in one pass; repeated text.count would be quadratic for
            # common tokens in a large file. Text rows are line-level: the first
            # match without syntax context anchors its line; others may follow.
            line = 1
            line_start = 0
            for match in matches:
                while True:
                    newline = text.find("\n", line_start, match.start())
                    if newline < 0:
                        break
                    line += 1
                    line_start = newline + 1
                col = len(text[line_start:match.start()].encode("utf-8")) + 1
                if (line, col) not in covered and line not in text_lines:
                    text_lines.add(line)
                    rows.append(EvidenceRow(file.path, line, col, "text", _source(lines, line, col)))
                    scan.methods.add("literal text")
        if ranges is not None and rows:
            # Outline ranges describe the inventory's bytes in its line model;
            # on other bytes, or Python AST lines (bare CR) against parser rows,
            # they could attribute a moved call to the wrong definition.
            if file.sha256 != hashlib.sha256(raw).hexdigest():
                mismatch = "file changed since inventory"
            elif tree is not None and file.language == "python" and "\r" in text.replace("\r\n", ""):
                mismatch = "bare CR: outline and parser lines differ"
            else:
                mismatch = ""
            if mismatch:
                scan.limits[f"{mismatch}; definition ranges not applied"] += 1
                stale = True
                rows = []
            rows = [r for r in rows if any(start <= r.line <= end for start, end in ranges.get(file.path, []))]
        rows.sort(key=lambda r: (r.line, r.column, r.specifier))
        if select_rows is not None:
            rows = select_rows(rows)
        room = max_rows - len(scan.rows)
        scan.rows.extend(rows[:room])
        if len(rows) > room:
            scan.limits[f"selected row cap {max_rows}"] += 1
            scan.incomplete = True
            break
        if time.monotonic() > deadline:
            scan.limits["wall budget hit during row selection"] += 1
            scan.incomplete = True
        if scan.incomplete:
            break
    scan.incomplete |= stale
    return scan
