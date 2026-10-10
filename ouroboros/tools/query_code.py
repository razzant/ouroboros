"""Read-only structured code queries over the deterministic code inventory."""

from __future__ import annotations

import os
import pathlib
import re
import time
from collections import Counter
from typing import Any, Callable, List

from ouroboros.config import runtime_setting
from ouroboros.protected_artifacts import block_reason_for_path
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    path_is_relative_to,
)
from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.tools.tool_result import completed_local_read

_OPS = (
    "relevant_files",
    "symbols",
    "definition",
    "references",
    "callers",
    "callees",
    "impact",
    "structural",
    "digest",
    "architecture",
)
_MAX_LIMIT = 200
_OP_OPTIONS = {
    "symbols": {"kind", "lang"}, "definition": {"kind", "lang"},
    "references": {"lang"}, "callers": {"lang"}, "callees": {"lang"},
    "impact": {"lang", "depth"}, "structural": {"lang"},
    "digest": {"lang"}, "relevant_files": {"lang"}, "architecture": set(),
}
# Structural walks read every candidate file, so bound them for an arbitrary
# external root (a user_files target like /app or ~) the way search_code bounds
# its scan — a file cap plus the shared wall-clock budget, symlink-confined.
_STRUCTURAL_MAX_FILES = 20000


def _structural_wall_budget() -> float:
    try:
        return max(5.0, float(runtime_setting("OUROBOROS_SEARCH_CODE_WALL_SEC", "45") or 45))
    except Exception:
        return 45.0


def _walk_candidate_files(scope: pathlib.Path, repo_root: pathlib.Path) -> tuple[list[pathlib.Path], str]:
    """Bounded, symlink-safe file enumeration under *scope*. Does NOT follow
    directory symlinks and drops files whose resolved path escapes *repo_root*,
    so a structural query over an external user_files target cannot wander the
    whole filesystem. Stops at a file cap or the wall-clock budget and returns a
    disclosed-truncation note (P1, never silent)."""
    if scope.is_file():
        return [scope], ""
    root_resolved = repo_root.resolve(strict=False)
    deadline = time.monotonic() + _structural_wall_budget()
    files: list[pathlib.Path] = []
    for dirpath, dirnames, filenames in os.walk(scope, followlinks=False):
        if time.monotonic() > deadline:
            return files, f"walk stopped after {_structural_wall_budget():.0f}s wall budget (narrow path=)"
        dirnames.sort()
        for name in sorted(filenames):
            fp = pathlib.Path(dirpath) / name
            try:
                rp = fp.resolve(strict=False)
                if rp != root_resolved and not path_is_relative_to(rp, root_resolved):
                    continue
            except Exception:
                continue
            files.append(fp)
            if len(files) >= _STRUCTURAL_MAX_FILES:
                return files, f"walk stopped at {_STRUCTURAL_MAX_FILES} files (narrow path=)"
    return files, ""


def _safe_path(repo_root: pathlib.Path, path: str) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if not text or text == ".":
        return ""
    target = (repo_root / text).resolve(strict=False)
    try:
        return target.relative_to(repo_root.resolve(strict=False)).as_posix()
    except ValueError as exc:
        raise ValueError(f"path escapes root: {path}") from exc


def _visible_file(
    ctx: ToolContext,
    repo_root: pathlib.Path,
    rel_path: str,
    binding: ResolvedResourceBinding | None = None,
    runtime_check: Callable[[pathlib.Path], str] | None = None,
) -> bool:
    try:
        target = (repo_root / rel_path).resolve(strict=False)
    except Exception:
        return False
    from ouroboros.tools.core_file_tools import _runtime_data_read_block

    return not (
        (runtime_check(target) if runtime_check is not None else _runtime_data_read_block(ctx, target, root=binding.root if binding else ""))
        or block_reason_for_path(ctx, target, "read_bytes", binding)
        or block_reason_for_path(ctx, target, "static_introspection", binding)
    )


def _structural(
    ctx: ToolContext,
    repo_root: pathlib.Path,
    query: str,
    path: str,
    lang: str,
    binding: ResolvedResourceBinding | None = None,
    runtime_check: Callable[[pathlib.Path], str] | None = None,
) -> list[str]:
    # Tree-sitter node types or a Python AST fallback, never literal matching.
    # Query may be a tree-sitter S-expression
    # like "(function_definition)" or a node type such as "FunctionDef".
    import ast

    def _query_node_type(raw: str) -> str:
        text = str(raw or "").strip()
        if text.startswith("("):
            match = re.match(r"\(\s*([A-Za-z_][\w-]*)", text)
            return match.group(1) if match else ""
        return text

    ts_node_type = _query_node_type(query)

    # CW11 (v6.34.0): structural is polyglot via the SAME tree-sitter infrastructure as
    # the symbol inventory — _language (suffix -> language id) + _TS_LANGUAGES (id ->
    # grammar) + the cached _ts_parser. Python keeps a stdlib-ast fallback; every other
    # language is tree-sitter ONLY (no literal/text fallback), so a node-type query never
    # false-matches a comment and never echoes source, and a missing grammar surfaces a
    # visible structural_unavailable:<lang> marker instead of a silent guess.
    from ouroboros.code_intelligence import _TS_LANGUAGES, _language, _ts_parser
    from ouroboros.code_search_rg import MAX_FILE_SIZE_BYTES, search_skip_reason

    deadline = time.monotonic() + _structural_wall_budget()
    skipped = Counter()

    def _file_lang_grammar(fp: pathlib.Path):
        lid = _language(fp)
        if lid == "python":
            return ("python", "python")
        return (lid, _TS_LANGUAGES.get(lid))

    def _filter_grammar(value: str):
        """The user's lang filter normalized to a grammar (or 'python'); None => no filter."""
        v = str(value or "").strip().lower()
        if v in ("", "any"):
            return None
        if v == "python":
            return "python"
        return _TS_LANGUAGES.get(v, v)

    def _ts_rows(grammar: str, rel: str, text: str):
        """tree-sitter node-type matches, or None when the grammar/library is unavailable."""
        parser = _ts_parser(grammar)
        if parser is None:
            return None
        try:
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            skipped["parse_error"] += 1
            return None
        if tree.root_node.has_error:
            skipped["syntax_error_partial_tree"] += 1
        found: list[str] = []
        stack = [tree.root_node]
        while stack:
            if time.monotonic() > deadline:
                skipped["time_limit"] += 1
                break
            node = stack.pop()
            if node.type == ts_node_type:
                found.append(f"{rel}:{int(node.start_point[0]) + 1} {node.type}")
            stack.extend(reversed(list(node.children)))
        return found

    want_grammar = _filter_grammar(lang)
    scope = (repo_root / (path or ".")).resolve(strict=False)
    candidates, walk_note = _walk_candidate_files(scope, repo_root)
    rows: list[str] = []
    unavailable_seen: set = set()
    # Page size is not a collection cap. Keep a distinct, disclosed work bound.
    cap = 20000
    for fp in candidates:
        if time.monotonic() > deadline:
            skipped["time_limit"] += 1
            break
        if len(rows) >= cap:
            break
        if not fp.is_file():
            continue
        lang_id, grammar = _file_lang_grammar(fp)
        # Skip non-code files (no grammar, not python) unless the user explicitly filtered
        # to a language whose grammar happens to be missing (then surface its marker).
        if grammar is None and lang_id != "python" and want_grammar is None:
            continue
        file_grammar = "python" if lang_id == "python" else grammar
        if want_grammar is not None and file_grammar != want_grammar:
            continue
        try:
            rel = fp.relative_to(repo_root).as_posix()
        except ValueError:
            continue
        if not _visible_file(ctx, repo_root, rel, binding, runtime_check):
            skipped["policy_excluded"] += 1
            continue
        if not ts_node_type:
            continue
        reason = search_skip_reason(fp)
        if reason:
            skipped[reason] += 1
            continue
        try:
            with fp.open("rb") as source:
                raw = source.read(MAX_FILE_SIZE_BYTES + 1)
            if len(raw) > MAX_FILE_SIZE_BYTES or b"\0" in raw:
                skipped["oversized_or_binary"] += 1
                continue
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            skipped["read_error"] += 1
            continue
        if lang_id == "python":
            ts = _ts_rows("python", rel, text)
            if ts:
                rows.extend(ts)
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                skipped["python_ast_syntax_error"] += 1
                continue
            for node in ast.walk(tree):
                if node.__class__.__name__.casefold() == ts_node_type.casefold():
                    rows.append(f"{rel}:{int(getattr(node, 'lineno', 0) or 0)} {node.__class__.__name__}")
            continue
        ts = _ts_rows(grammar, rel, text)
        if ts is None:
            if lang_id not in unavailable_seen:
                unavailable_seen.add(lang_id)
                rows.append(f"structural_unavailable:{lang_id} (tree-sitter grammar not loaded)")
        else:
            rows.extend(ts)
    if walk_note:
        rows.append(f"structural_walk_truncated: {walk_note}")
    if len(rows) >= cap:
        rows = rows[:cap] + [f"structural_rows_truncated: selected row cap {cap}"]
    rows.extend(f"structural_limit: {reason}={count}" for reason, count in sorted(skipped.items()))
    return rows


@completed_local_read
def _query_code(
    ctx: ToolContext,
    op: str,
    _resolved_binding: ResolvedResourceBinding | None = None,
    **options: Any,
) -> str:
    query = str(options.get("query") or "")
    path = str(options.get("path") or "")
    lang = str(options.get("lang") or "any")
    kind = str(options.get("kind") or "any")
    depth = int(options.get("depth") or 1)
    root = str(options.get("root") or "active_workspace")
    bucket = str(options.get("bucket") or "")
    skill_name = str(options.get("skill_name") or "")
    limit = int(options.get("limit") or 40)
    offset = int(options.get("offset") or 0)
    op = str(op or "").strip()
    if op not in _OPS:
        return f"⚠️ TOOL_ARG_ERROR (query_code): op must be one of {', '.join(_OPS)}."
    if op not in ("symbols", "digest") and not str(query or "").strip():
        return f"⚠️ TOOL_ARG_ERROR (query_code): op '{op}' requires query."
    for name, value, default in (("kind", kind, "any"), ("lang", lang, "any"), ("depth", depth, 1)):
        if value != default and name not in _OP_OPTIONS[op]:
            return f"⚠️ TOOL_ARG_ERROR (query_code): op '{op}' does not read {name}."
    if op == "digest" and query:
        return "⚠️ TOOL_ARG_ERROR (query_code): digest does not read query; use path to select its scope."
    if op == "architecture" and path not in ("", "."):
        return "⚠️ TOOL_ARG_ERROR (query_code): architecture reads its target from query, not path."
    try:
        binding = _resolved_binding or build_resolved_resource_binding(
            ctx,
            root=root,
            operation="search",
            path=path or ".",
            bucket=bucket,
            skill_name=skill_name,
        )
    except Exception as exc:
        if str(exc).startswith("profile=") and " cannot " in str(exc):
            return f"⚠️ TOOL_ACCESS_BLOCKED: {str(exc).rstrip('.')}."
        return f"⚠️ TOOL_ARG_ERROR (query_code): {exc}"
    try:
        normalized_root = binding.root
        if normalized_root == "system_repo":
            repo_root = binding.base_path
        elif normalized_root == "active_workspace":
            repo_root = binding.base_path
        elif normalized_root == "skill_payload":
            repo_root = binding.base_path
        elif normalized_root == "user_files":
            if not str(path or "").strip():
                raise ValueError(
                    "root=user_files requires an explicit path (e.g. '/app' or a project subdir); "
                    "it will not scan the entire home"
                )
            # Documented external-target contract (v6.47.0): read-only code
            # intelligence over an absolute path OUTSIDE the user_files home
            # (e.g. a benchmark /app) stays supported — opt out of the v6.54.3
            # home-membership rejection; the credential/control-plane block
            # reasons still apply inside resolve_user_file_path.
            target = binding.target_path
            if target.is_dir():
                repo_root = target.resolve(strict=False)
            elif target.is_file():
                repo_root = target.parent.resolve(strict=False)
            else:
                raise ValueError(f"user_files path does not exist: {str(path).strip()}")
        else:
            raise ValueError(
                "root must be active_workspace, system_repo, skill_payload, or user_files"
            )
        # The binding already resolved host/backend addresses and confinement;
        # scope the query to that physical target instead of re-reading raw args.
        path = binding.target_path.relative_to(repo_root).as_posix()
        scoped_path = _safe_path(repo_root, path)
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (query_code): {exc}"

    from ouroboros.tools.core_file_tools import _runtime_data_read_check

    runtime_check = _runtime_data_read_check(ctx, root=binding.root)
    if scoped_path and (repo_root / scoped_path).is_file() and not _visible_file(
        ctx, repo_root, scoped_path, binding, runtime_check,
    ):
        # A scope header must not re-expose a protected operand omitted by the
        # existing resource policy. The reason is independent of its spelling.
        return "⚠️ TOOL_ACCESS_BLOCKED: selected source is unavailable to this resource view."
    limit = min(max(1, int(limit or 40)), _MAX_LIMIT)
    offset = max(0, int(offset or 0))
    from ouroboros.code_navigation import NavigationView

    view = NavigationView()
    deadline = time.monotonic() + _structural_wall_budget()

    def admitted(target: pathlib.Path) -> bool:
        return _visible_file(ctx, repo_root, target.relative_to(repo_root).as_posix(), binding, runtime_check)

    def admitted_inventory(scope: str):
        """This resource view's inventory; every inventory consumer shares its admission."""
        from ouroboros.code_intelligence import build_code_inventory
        from ouroboros.protected_artifacts import protected_artifact_paths
        from ouroboros.tools.core_secret_paths import is_restricted_subagent_profile

        exclude_paths: list[pathlib.Path] = list(protected_artifact_paths(ctx, binding))
        # Do not cache an external/ephemeral user_files target's inventory in the
        # live code-intel cache. Cache writes retain their existing actor/mode
        # contract independently of file visibility; Cyber acting tasks may persist.
        persist = not (exclude_paths or normalized_root == "user_files" or is_restricted_subagent_profile(ctx))
        inventory = build_code_inventory(
            repo_root, drive_root=pathlib.Path(ctx.drive_root), persist=persist, exclude_paths=exclude_paths,
            scope=scope, path_allowed=admitted,
        )
        inventory.files = [
            file for file in inventory.files
            if _visible_file(ctx, repo_root, file.path, binding, runtime_check)
        ]
        return inventory

    try:
        if op == "architecture":
            # Architecture facts (CPL-3) are defined over the Ouroboros repo's
            # pinned inventories — the manifest/inventory carriers live in the
            # code roots, never in a skill payload or an external target.
            if normalized_root not in ("active_workspace", "system_repo"):
                return (
                    "⚠️ TOOL_ARG_ERROR (query_code): op architecture requires "
                    "root=active_workspace or system_repo."
                )
            from ouroboros.code_intelligence_architecture import ARCHITECTURE_LIMIT_MARKER, architecture_fact_rows

            try:
                # A bare-symbol owner_of reads source through the same admission.
                rows = architecture_fact_rows(repo_root, query, inventory=lambda: admitted_inventory(""))
                view.notes = ["method: pinned architecture carriers; scope: repository; path targets are in query"]
            except ValueError as exc:
                return f"⚠️ TOOL_ARG_ERROR (query_code): {exc}"
            view.limits = [row for row in rows if row.startswith(ARCHITECTURE_LIMIT_MARKER)]
            view.incomplete = bool(view.limits)
            rows = [row for row in rows if not row.startswith(ARCHITECTURE_LIMIT_MARKER)]
        elif op == "structural":
            # Collection is bounded by the walk and row work caps, never by page
            # size; offset/limit page the collected rows below (#447 D6).
            rows = _structural(
                ctx, repo_root, query, scoped_path, str(lang or "any"),
                binding, runtime_check
            )
            view.notes = [f"method: tree-sitter / Python ast; scope: {scoped_path or '.'}; lang={lang}",
                          "universe: independent bounded filesystem walk (not the Git inventory)"]
            markers = [row for row in rows if row.startswith(("structural_walk_truncated:", "structural_rows_truncated:", "structural_limit:"))]
            view.limits.extend(markers)
            view.incomplete = bool(markers)
            rows = [row for row in rows if row not in markers]
        else:
            inventory_scope = scoped_path
            if op == "impact" or (op in {"references", "callers"} and (repo_root / scoped_path).is_file()):
                inventory_scope = ""
            inventory = admitted_inventory(inventory_scope)
            from ouroboros.code_navigation import inventory_view

            if op == "impact" and ("/" in query or "\\" in query):
                query = _safe_path(repo_root, query)
            view = inventory_view(
                inventory, op=op, query=query, path=scoped_path, kind=kind, lang=lang,
                depth=depth, deadline=deadline, path_allowed=admitted,
            )
            rows = view.rows
    except ValueError as exc:
        return f"⚠️ TOOL_ARG_ERROR (query_code): {exc}"
    except Exception as exc:
        return f"⚠️ QUERY_CODE_ERROR: {type(exc).__name__}: {exc}"

    total = len(rows)
    shown = rows[offset:offset + limit]
    next_offset = offset + limit
    label = query or scoped_path or "."
    header = f"{op} `{label}` — {len(shown)} of {'at least ' if view.incomplete else ''}{total}"
    if next_offset < total:
        header += f" — next offset={next_offset}"
    if not shown:
        header += (f" — offset={offset} is beyond the {total} collected rows" if offset else
                   " — no selected evidence within the searched coverage")
    if view.incomplete:
        header = "⚠️ QUERY_CODE_TRUNCATED: " + header
    if offset:
        view.notes.append("page re-scanned; rows can shift if files changed (no snapshot)")
    if view.notes:
        header += "\n" + "\n".join("  · " + note for note in view.notes)
    if view.limits:
        header += "\nlimits: " + "; ".join(dict.fromkeys(view.limits))
    return header + ("\n\n" + "\n".join(shown) if shown else "")


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("query_code", {
            "name": "query_code",
            "description": (
                "Read-only code navigation with source evidence and explicit scope/limits. "
                "symbols gives local outlines, including qualified declarators; definition also gives labelled "
                "syntax name-field candidates. references finds exact-token occurrences (including strings/comments); "
                "callers filters syntactic callee positions, callees finds calls in selected definition ranges. "
                "These are not resolved symbol bindings: aliases and dynamic names need separate queries. "
                "impact accepts a file or symbol and shows occurrence evidence and filesystem import candidates; "
                "every depth hop remains a candidate. digest pages scoped file outlines with directory rollups; "
                "relevant_files ranks scoped files by task words. structural queries grammar node types in a "
                "separate bounded filesystem walk (Python also supports AST class names). Missing grammar or "
                "partial coverage is disclosed. architecture reads pinned repository carriers with query='<fact> "
                "<argument>': owner_of, domain_dependencies, facade_consumers, persistence_entities_written_by, "
                "protected_contracts_affected. Pages rescan current files, not an immutable snapshot. "
                "Use search_code for arbitrary literal/regex search; inspect source anchors to assess meaning."
            ),
            "parameters": {"type": "object", "properties": {
                "op": {"type": "string", "enum": list(_OPS), "description": "Operation: relevant_files (where to look), digest (scoped paged outline), symbols, definition, references, callers, callees, impact, structural, architecture (domain/facade/persistence/protected facts)."},
                "query": {"type": "string", "default": "", "description": "Exact name/token (definition/references/callers), file or symbol (impact), AST node type (structural), task text (relevant_files), or '<fact> <argument>' (architecture). Empty for digest."},
                "path": {"type": "string", "default": "", "description": "Directory scopes search. File scopes symbols/definition/callees/digest/relevant_files; references/callers use it as an ordering selector while searching root. Impact target is query; file path may select a symbol definition or repeat the file target. REQUIRED for root=user_files (the explicit target dir/file, e.g. '/app' or '/app/src'); it is never the whole home."},
                "lang": {"type": "string", "enum": ["python", "javascript", "typescript", "go", "rust", "java", "ruby", "c", "cpp", "csharp", "php", "kotlin", "swift", "scala", "lua", "bash", "any"], "default": "any"},
                "kind": {"type": "string", "enum": ["function", "async_function", "class", "constant", "variable", "method", "constructor", "struct", "union", "interface", "enum", "trait", "impl", "protocol", "type", "module", "namespace", "object", "macro", "any"], "default": "any"},
                "depth": {"type": "integer", "default": 1, "description": "1..5 candidate import hops for impact; only unique filesystem candidates expand, never proven binding."},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "skill_payload", "user_files"], "default": "active_workspace", "description": "active_workspace/system_repo are code roots; skill_payload selects one exact skill with bucket + skill_name; user_files runs read-only intelligence over an EXTERNAL target dir/file named by path= (e.g. /app), never the whole home."},
                "bucket": {"type": "string", "enum": ["external", "clawhub", "ouroboroshub", "native", "user_repo"], "description": "Required with root=skill_payload; selects the physical skill source."},
                "skill_name": {"type": "string", "description": "Required with root=skill_payload; exact skill directory identity."},
                "limit": {"type": "integer", "default": 40},
                "offset": {"type": "integer", "default": 0},
            }, "required": ["op"]},
        }, _query_code, timeout_sec=120),
    ]
