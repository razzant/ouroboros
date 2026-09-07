"""Read-only structured code queries over the deterministic code inventory."""

from __future__ import annotations

import os
import pathlib
import re
import time
from typing import Any, Callable, List

from ouroboros.protected_artifacts import block_reason_for_path
from ouroboros.tool_access import (
    ResolvedResourceBinding,
    build_resolved_resource_binding,
    path_is_relative_to,
)
from ouroboros.tools.registry import ToolContext, ToolEntry


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
# Structural walks read every candidate file, so bound them for an arbitrary
# external root (a user_files target like /app or ~) the way search_code bounds
# its scan — a file cap plus the shared wall-clock budget, symlink-confined.
_STRUCTURAL_MAX_FILES = 20000


def _structural_wall_budget() -> float:
    try:
        return max(5.0, float(os.environ.get("OUROBOROS_SEARCH_CODE_WALL_SEC", "45") or 45))
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
    secret_check: Callable[[pathlib.Path], bool] | None = None,
) -> bool:
    try:
        target = (repo_root / rel_path).resolve(strict=False)
    except Exception:
        return False
    try:
        from ouroboros.tools.core import is_restricted_subagent_profile as _is_local_readonly_subagent, _is_subagent_secret_repo_target

        if _is_local_readonly_subagent(ctx) and (
            secret_check(target) if secret_check else _is_subagent_secret_repo_target(target, repo_root, ctx=ctx)
        ):
            return False
    except Exception:
        pass
    return not (
        block_reason_for_path(ctx, target, "read_bytes", binding)
        or block_reason_for_path(ctx, target, "static_introspection", binding)
    )


def _inventory_rows(
    ctx: ToolContext,
    inventory: Any,
    repo_root: pathlib.Path,
    opts: dict[str, Any],
    binding: ResolvedResourceBinding | None = None,
    secret_check: Callable[[pathlib.Path], bool] | None = None,
) -> list[str]:
    from ouroboros.code_intelligence import (
        impact_files,
        relevant_files,
        symbol_callees,
        symbol_callers,
        symbol_definitions,
        symbol_references,
    )

    op = str(opts.get("op") or "")
    query = str(opts.get("query") or "")
    path = str(opts.get("path") or "")
    kind = str(opts.get("kind") or "any")
    depth = int(opts.get("depth") or 1)
    limit = int(opts.get("limit") or 40)
    offset = int(opts.get("offset") or 0)
    rows: list[str] = []
    if op in {"symbols", "definition"}:
        for file, symbol in symbol_definitions(inventory, query, path=path, kind=kind or "any"):
            if _visible_file(ctx, repo_root, file.path, binding, secret_check):
                rows.append(f"{file.path}:{symbol.line_start} {symbol.kind} {symbol.signature or symbol.name}")
    elif op == "references":
        for file, ref in symbol_references(inventory, query, path=path):
            if _visible_file(ctx, repo_root, file.path, binding, secret_check):
                rows.append(f"{file.path}:{ref.line} {query}{' in ' + ref.enclosing if ref.enclosing else ''}")
    elif op in {"callers", "callees"}:
        iterator = symbol_callers(inventory, query, path=path) if op == "callers" else symbol_callees(inventory, query, path=path)
        for file, call in iterator:
            if _visible_file(ctx, repo_root, file.path, binding, secret_check):
                rows.append(f"{file.path}:{call.line} {call.enclosing + ' -> ' if call.enclosing else ''}{call.name}")
    elif op == "impact":
        for file, reason in impact_files(inventory, path or query, depth=depth):
            if _visible_file(ctx, repo_root, file.path, binding, secret_check):
                rows.append(f"{file.path}  {reason}")
    elif op == "relevant_files":
        for idx, (file, score, reason) in enumerate(relevant_files(inventory, query, limit=min(_MAX_LIMIT, offset + limit)), 1):
            if _visible_file(ctx, repo_root, file.path, binding, secret_check):
                top_symbols = ", ".join(symbol.name for symbol in file.symbols[:5])
                rows.append(f"{idx}. {file.path} score={score:.2f} reason={reason}{' symbols=' + top_symbols if top_symbols else ''}")
    return rows


def _structural(
    ctx: ToolContext,
    repo_root: pathlib.Path,
    query: str,
    path: str,
    lang: str,
    limit: int,
    binding: ResolvedResourceBinding | None = None,
    secret_check: Callable[[pathlib.Path], bool] | None = None,
) -> list[str]:
    # Conservative first step: use tree-sitter when available, otherwise a Python
    # ast fallback plus literal matching. Query may be a tree-sitter S-expression
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
            return None
        found: list[str] = []
        stack = [tree.root_node]
        while stack:
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
    cap = min(max(1, limit), _MAX_LIMIT)
    for fp in candidates:
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
        if not _visible_file(ctx, repo_root, rel, binding, secret_check):
            continue
        if not ts_node_type:
            continue
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if lang_id == "python":
            ts = _ts_rows("python", rel, text)
            if ts:
                rows.extend(ts)
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
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
    return rows


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
            try:
                from ouroboros.tool_access import active_tool_profile

                if active_tool_profile(ctx) == "acting_subagent":
                    return "⚠️ TOOL_ACCESS_BLOCKED: query_code root=system_repo is not available to acting subagents."
            except Exception:
                pass
            repo_root = binding.base_path
        elif normalized_root == "active_workspace":
            repo_root = binding.base_path
        elif normalized_root == "skill_payload":
            repo_root = binding.base_path
        elif normalized_root == "user_files":
            # Read-only structured intelligence over an EXTERNAL workspace target
            # (e.g. the SWE-bench dig-direct /app) — R1. Restricted subagents must
            # not read arbitrary owner home; the main/live task is allowed. An
            # empty path is a HARD ERROR: it will not scan the entire home.
            try:
                from ouroboros.tool_access import active_tool_profile

                if active_tool_profile(ctx) in ("acting_subagent", "local_readonly_subagent"):
                    return "⚠️ TOOL_ACCESS_BLOCKED: query_code root=user_files is not available to subagents."
            except Exception:
                pass
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

    limit = min(max(1, int(limit or 40)), _MAX_LIMIT)
    offset = max(0, int(offset or 0))
    secret_check = None
    try:
        from ouroboros.tools.core_secret_paths import is_restricted_subagent_profile, make_subagent_secret_target_check

        if op != "architecture" and is_restricted_subagent_profile(ctx):
            secret_check = make_subagent_secret_target_check(repo_root, ctx=ctx)
    except Exception:
        pass  # Preserve the existing per-target fallback on policy preparation failure.

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
            from ouroboros.code_intelligence_architecture import architecture_fact_rows

            try:
                rows = architecture_fact_rows(repo_root, query)
            except ValueError as exc:
                return f"⚠️ TOOL_ARG_ERROR (query_code): {exc}"
        elif op == "structural":
            # Collect through the requested page (offset+limit, like relevant_files
            # above): collecting only `limit` rows made rows[offset:] empty on every
            # page after the first and blamed the query for it (#447 D6).
            rows = _structural(
                ctx, repo_root, query, scoped_path, str(lang or "any"),
                min(_MAX_LIMIT, offset + limit), binding, secret_check
            )
        else:
            from ouroboros.code_intelligence import build_code_inventory
            from ouroboros.protected_artifacts import protected_artifact_paths

            exclude_paths: list[pathlib.Path] = list(protected_artifact_paths(ctx, binding))
            persist = True
            if exclude_paths or normalized_root == "user_files":
                # Do not cache an external/ephemeral user_files target's inventory
                # in the live code-intel cache.
                persist = False
            try:
                from ouroboros.tools.core import is_restricted_subagent_profile as _is_local_readonly_subagent, _is_subagent_secret_repo_target

                if _is_local_readonly_subagent(ctx):
                    persist = False
                    exclude_paths = [
                        p for p in repo_root.rglob("*")
                        if (secret_check(p) if secret_check else _is_subagent_secret_repo_target(p, repo_root, ctx=ctx))
                    ]
            except Exception:
                pass
            inventory = build_code_inventory(repo_root, drive_root=pathlib.Path(ctx.drive_root), persist=persist, exclude_paths=exclude_paths)
            inventory.files = [
                file for file in inventory.files
                if _visible_file(ctx, repo_root, file.path, binding, secret_check)
            ]
            if op == "digest":
                # Whole-repo map (folded from the former codebase_digest tool):
                # a compact file/symbol inventory to orient in an unfamiliar repo.
                from ouroboros.code_intelligence import render_codebase_digest
                digest_text = render_codebase_digest(inventory)
                if normalized_root == "user_files":
                    # Same В23 egress seam: a secret-shaped file/symbol NAME in
                    # the owner's home must not surface raw (#447 s2r2).
                    from ouroboros.secret_masking import mask_secret_bytes

                    digest_text, _masked = mask_secret_bytes(digest_text)
                return digest_text
            rows = _inventory_rows(ctx, inventory, repo_root, {
                "op": op, "query": query, "path": scoped_path, "kind": kind,
                "depth": depth, "limit": limit, "offset": offset,
            }, binding, secret_check)
    except Exception as exc:
        return f"⚠️ QUERY_CODE_ERROR: {type(exc).__name__}: {exc}"

    total = len(rows)
    shown = rows[offset:offset + limit]
    next_offset = offset + limit
    label = query or scoped_path or "."
    # Collection stops at min(_MAX_LIMIT, offset+limit) for the ops that page a
    # capped collector (structural, relevant_files): a full collection there
    # means more matches MAY exist, so "No results" / "N of N" would be a
    # success-shaped lie about completeness (#447 S3). Ops whose collectors
    # return the complete set must NOT claim a cap they never applied.
    collection_capped = (
        op in ("structural", "relevant_files")
        and total >= min(_MAX_LIMIT, offset + limit)
    )
    if not shown:
        if collection_capped:
            return (
                f"⚠️ QUERY_CODE_TRUNCATED: collection for op `{op}` `{label}` stops at "
                f"{_MAX_LIMIT} rows and offset={offset} lies beyond what was collected. "
                "Narrow the query or path= instead of paging past the cap."
            )
        return f"No results for op `{op}` `{label}`. {_empty_hint(op, label)}"
    def _mask_user_files_rows(text: str) -> str:
        # Same egress seam as read_file/search (#447 В23): query_code snippets
        # over the owner's home must not carry raw credential bytes.
        from ouroboros.tools.core_secret_paths import is_restricted_subagent_profile

        if normalized_root != "user_files" and not is_restricted_subagent_profile(ctx):
            return text
        from ouroboros.secret_masking import mask_secret_bytes

        masked, count = mask_secret_bytes(
            text, mask_opaque=normalized_root not in {"active_workspace", "system_repo"},
        )
        if count:
            masked += (
                f"\n⚠️ SECRET_BYTES_MASKED: {count} secret-shaped span(s) were "
                "replaced with ***; raw credentials never enter model context."
            )
        return masked

    header = f"{op} `{label}` — {len(shown)} of {total}"
    if next_offset < total:
        header += f" — next offset={next_offset}"
    elif collection_capped:
        # The collector filled exactly the requested page: "N of N" would claim
        # completeness it never verified, below the 200 cap included.
        header += (
            f" — collection capped at {_MAX_LIMIT}; more may exist (narrow query/path=)"
            if total >= _MAX_LIMIT
            else f" — more may exist; continue with offset={next_offset}"
        )
    return _mask_user_files_rows(header + "\n\n" + "\n".join(shown)) + _next_step_hint(op)


def _empty_hint(op: str, label: str) -> str:
    """Op-specific recovery hint — do NOT reflexively redirect to search_code."""
    if op in ("definition", "references", "callers", "callees", "impact"):
        return (
            f"Check the exact symbol name (these ops match a defined symbol, not text). "
            f"Use op=relevant_files query=\"{label}\" to find where to look, or op=symbols to list what's defined."
        )
    if op == "symbols":
        return "Narrow with path= to a file/dir, or use op=relevant_files to locate the area first."
    if op == "structural":
        return ("structural needs a node type, not free text — an AST class for Python (FunctionDef/ClassDef) "
                "or a tree-sitter node for other langs (function_declaration for Go, struct_item for Rust, etc.). "
                "Add lang=go|rust|... to filter by language.")
    if op == "relevant_files":
        return "Rephrase the task in domain words, or use search_code for an exact string you expect in the source."
    return "Verify the symbol/path; use search_code only for plain-text/regex matches."


def _next_step_hint(op: str) -> str:
    """Suggest the natural follow-up op so results chain instead of dead-ending."""
    hints = {
        "relevant_files": "\n\nNext: read_file(...) the top hit, or query_code(op=symbols, path=...) to list its symbols.",
        "symbols": "\n\nNext: query_code(op=definition/references, query=<name>) on a symbol of interest.",
        "definition": "\n\nNext: query_code(op=references/callers, query=<name>) to see how it is used.",
        "callers": "\n\nNext: read_file(...) a caller, or query_code(op=impact, query=<name>) for blast radius.",
        "callees": "\n\nNext: query_code(op=definition, query=<callee>) to read what it calls.",
    }
    return hints.get(op, "")


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("query_code", {
            "name": "query_code",
            "description": (
                "Read-only structured code intelligence over the active workspace — prefer this "
                "over grep/find/sed-as-reader for anything symbol-aware. Start with "
                "op=relevant_files (task text -> the files to read) when you don't yet know where "
                "to look; op=digest maps an unfamiliar repo FIRST; then symbols/definition/"
                "references/callers/callees/impact/structural for precise navigation. Use search_code "
                "only for plain text/regex. Symbol intelligence (digest/symbols/definition/references/"
                "callers/callees/impact) is polyglot via tree-sitter (Python/JS/TS/Go/Rust/Java/Ruby/C/"
                "...); op=structural (node-type queries) is polyglot too — tree-sitter for every supported "
                "language (Python/JS/TS/Go/Rust/Java/Ruby/C/C++/C#/PHP/Kotlin/Swift/Scala/Lua/Bash), with a "
                "visible structural_unavailable:<lang> marker when a grammar is missing (Python also has a "
                "stdlib-ast fallback). op=architecture answers over the Ouroboros repo's pinned inventories "
                "(domain manifest, facade/persistence/frozen-contract carriers): query='<fact> <argument>' with "
                "fact one of owner_of, domain_dependencies, facade_consumers, persistence_entities_written_by, "
                "protected_contracts_affected (argument = path/symbol, domain id, facade/name, writer, or a "
                "comma-separated changed-path list). Returns compact file:line anchors and signatures/snippets, "
                "never full bodies."
            ),
            "parameters": {"type": "object", "properties": {
                "op": {"type": "string", "enum": list(_OPS), "description": "Operation: relevant_files (where to look), digest (whole-repo map), symbols, definition, references, callers, callees, impact, structural, architecture (domain/facade/persistence/protected facts)."},
                "query": {"type": "string", "default": "", "description": "Exact symbol name (definition/references/callers/...), AST node type (structural), task text (relevant_files), or '<fact> <argument>' (architecture). Empty for digest."},
                "path": {"type": "string", "default": "", "description": "Optional file/dir scope or definition disambiguator. REQUIRED for root=user_files (the explicit target dir/file, e.g. '/app' or '/app/src'); it is never the whole home."},
                "lang": {"type": "string", "enum": ["python", "javascript", "typescript", "go", "rust", "java", "ruby", "c", "cpp", "csharp", "php", "kotlin", "swift", "scala", "lua", "bash", "any"], "default": "any"},
                "kind": {"type": "string", "enum": ["function", "async_function", "class", "constant", "any"], "default": "any"},
                "depth": {"type": "integer", "default": 1, "description": "Graph depth for impact."},
                "root": {"type": "string", "enum": ["active_workspace", "system_repo", "skill_payload", "user_files"], "default": "active_workspace", "description": "active_workspace/system_repo are code roots; skill_payload selects one exact skill with bucket + skill_name; user_files runs read-only intelligence over an EXTERNAL target dir/file named by path= (e.g. /app), never the whole home."},
                "bucket": {"type": "string", "enum": ["external", "clawhub", "ouroboroshub", "native", "user_repo"], "description": "Required with root=skill_payload; selects the physical skill source."},
                "skill_name": {"type": "string", "description": "Required with root=skill_payload; exact skill directory identity."},
                "limit": {"type": "integer", "default": 40},
                "offset": {"type": "integer", "default": 0},
            }, "required": ["op"]},
        }, _query_code, timeout_sec=120),
    ]
