"""Single source of truth for tool visibility, parallelism, and result limits."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib
from collections.abc import Iterable, Mapping
from typing import Any

# ── RWS v2 §3.3 structural isolation gate ───────────────────────────────
#
# Two directions, one traversal.  FORWARD: nothing in the execd bundle may
# reach Home authority, because the postmortem's whole failure class was an
# execd module that imported Home by accident and only failed on the remote
# host.  REVERSE: the transport/native modules may not import Home POLICY
# authorities either — one brain means Home decides and execd executes, so a
# policy import inside the bundle is a second authority by construction.
#
# The gate reads EVERY import scope, module-level and function-local alike.  It
# used to read module scope only, and claimed in this comment that "a
# clean-subprocess invocation smoke covers those per native operation" — no such
# smoke existed, so a function-local import was simply unchecked, and two real
# violations were living in that gap: `remote_ssh_config` reached
# `ouroboros.config` (settings_or_owner_state) and `ouroboros.utils` — a declared
# bundle member that really travels — reached `ouroboros.observability`
# (home_policy_authority, which does not travel).  Both are fixed at the source
# now, and `tests/test_remote_native_isolation.py` runs the compensating
# clean-subprocess smoke for real.
#
# The two scopes are treated DIFFERENTLY, on purpose:
#
# * module-scope imports are TRAVERSED transitively, because importing a module
#   unconditionally executes them — that is the "arrives merely by importing"
#   class the postmortem named;
# * function-local imports are checked as DIRECT edges of the module that spells
#   them, and not traversed.  They are conditional (they happen only if the
#   function runs) and they are the repo's ordinary cycle-breaker, so following
#   them transitively reaches 226 modules — effectively the whole program — and a
#   gate that always fails teaches nothing.  Checking them directly is what
#   catches the real class: a bundle module NAMING a Home authority.

REMOTE_NATIVE_CLOSURE_SEEDS: frozenset[str] = frozenset({
    "ouroboros.execd",
    "ouroboros.execd_spool",
    "ouroboros.execd_state",
    "ouroboros.execd_task_files",
    "ouroboros.remote_protocol",
    # The transport is seeded too, in the REVERSE direction: it never travels in
    # the bundle, but it must not import a Home policy authority either — that
    # inverted arrow (transport reaching Home to finish a result) is what forced
    # every Home guard to be mirrored inside the donor's remote path.
    "ouroboros.remote_ssh",
    "ouroboros.remote_ssh_config",
    "ouroboros.shell_parse",
    "ouroboros.workspace_diagnostics",
    "ouroboros.workspace_native",
    "ouroboros.workspace_native_contract",
    "ouroboros.workspace_payload_native",
    "ouroboros.workspace_query_native",
    "ouroboros.workspace_snapshot_native",
})

FORBIDDEN_REMOTE_IMPORT_PREFIXES: dict[str, tuple[str, ...]] = {
    "registry": ("ouroboros.tools",),
    "provider_or_model": (
        "ouroboros.llm",
        "ouroboros.local_model",
        "ouroboros.model_",
        "ouroboros.pricing",
        "ouroboros.provider_models",
    ),
    "review_or_planning": (
        "ouroboros.deep_self_review",
        "ouroboros.parallel_review",
        "ouroboros.plan_review",
        "ouroboros.review",
        "ouroboros.review_evidence",
        "ouroboros.review_state",
        "ouroboros.scope_review",
        "ouroboros.triad_review",
    ),
    "server_or_gateway": (
        "server",
        "supervisor",
        "ouroboros.gateway",
        "ouroboros.gateways",
        "ouroboros.server",
        "ouroboros.supervisor",
    ),
    "settings_or_owner_state": (
        "ouroboros.config",
        "ouroboros.owner",
        "ouroboros.settings_setup_contract",
    ),
    "home_task_or_artifact_state": (
        "ouroboros.artifacts",
        "ouroboros.mutation_attribution",
        "ouroboros.outcomes",
        "ouroboros.project_facts",
        "ouroboros.protected_artifacts",
        "ouroboros.task_pacing",
        "ouroboros.task_results",
        "ouroboros.task_status",
    ),
    # The reverse gate proper: Home POLICY authorities.  execd executing a
    # policy decision it made locally would be a second authority even if the
    # module never touched the network.
    "home_policy_authority": (
        "ouroboros.observability",
        "ouroboros.protected_artifacts",
        "ouroboros.remote_export_policy",
        "ouroboros.remote_transfer",
        "ouroboros.safety",
        "ouroboros.tool_access",
        "ouroboros.tool_policy",
        "ouroboros.workspace_admission",
        "ouroboros.workspace_executor",
    ),
}


def remote_native_import_closure(
    repo_root: pathlib.Path,
    *,
    operation_modules: Mapping[str, str] | None = None,
    extra_roots: Iterable[str] = (),
) -> dict[str, Any]:
    """Return the deterministic module-import-time closure for execd kernels.

    Every reached module records the edge that pulled it in, so a violation can
    name the exact `a -> b` import (and the whole chain back to its seed)
    instead of only reporting that the closure is dirty.
    """

    from ouroboros.workspace_native_contract import (
        REMOTE_NATIVE_OPERATION_MODULE,
        validate_remote_native_operation_map,
    )

    native_map = (
        REMOTE_NATIVE_OPERATION_MODULE
        if operation_modules is None
        else operation_modules
    )
    validate_remote_native_operation_map(native_map)
    root = pathlib.Path(repo_root).resolve(strict=False)
    seeds = frozenset({
        *native_map.values(),
        *REMOTE_NATIVE_CLOSURE_SEEDS,
        *(str(extra) for extra in extra_roots),
    })
    initial_modules = {
        module
        for seed in seeds
        for module in (seed, *_parent_packages(seed))
    }
    pending = sorted(initial_modules, reverse=True)
    visited: set[str] = set()
    missing: set[str] = set()
    edges: dict[str, tuple[str, ...]] = {}
    local_edges: dict[str, tuple[str, ...]] = {}
    importer: dict[str, str] = {}
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        module_path = _local_module_path(root, module)
        if module_path is None:
            missing.add(module)
            continue
        scoped = _local_imports_by_scope(root, module, module_path)
        imports = tuple(sorted(scoped["module"]))
        edges[module] = imports
        local_edges[module] = tuple(sorted(scoped["local"]))
        for imported in reversed(imports):
            for dependency in (imported, *_parent_packages(imported)):
                if dependency not in visited:
                    importer.setdefault(dependency, module)
                    pending.append(dependency)

    forbidden: dict[str, list[dict[str, str]]] = {}
    for category, prefixes in FORBIDDEN_REMOTE_IMPORT_PREFIXES.items():
        rows = [
            {
                "module": module,
                "edge": f"{importer.get(module, '<seed>')} -> {module}",
                "path": " -> ".join(_import_path(module, importer)),
                "scope": "module",
            }
            for module in sorted(visited)
            if any(_module_matches_prefix(module, prefix) for prefix in prefixes)
        ]
        # Function-local edges: named by a module IN the closure, judged directly.
        # The chain stops at the naming module because the edge is conditional —
        # what is forbidden is the bundle spelling the Home module's name at all.
        rows.extend(
            {
                "module": imported,
                "edge": f"{module} -> {imported}",
                "path": " -> ".join([*_import_path(module, importer), imported]),
                "scope": "function_local",
            }
            for module in sorted(local_edges)
            for imported in local_edges[module]
            if any(_module_matches_prefix(imported, prefix) for prefix in prefixes)
        )
        if rows:
            forbidden[category] = rows
    return {
        "roots": sorted(seeds),
        "modules": sorted(visited),
        "edges": {module: list(edges[module]) for module in sorted(edges)},
        "function_local_edges": {
            module: list(local_edges[module])
            for module in sorted(local_edges)
            if local_edges[module]
        },
        "importers": dict(sorted(importer.items())),
        "missing_modules": sorted(missing),
        "forbidden": forbidden,
    }


def assert_remote_native_import_closure(
    repo_root: pathlib.Path,
    *,
    operation_modules: Mapping[str, str] | None = None,
    extra_roots: Iterable[str] = (),
    declared_kernel_modules: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a clean closure, otherwise fail before an execd artifact exists."""

    from ouroboros.workspace_native_contract import REMOTE_NATIVE_KERNEL_MODULES

    audit = remote_native_import_closure(
        repo_root,
        operation_modules=operation_modules,
        extra_roots=extra_roots,
    )
    if audit["missing_modules"]:
        raise ValueError(
            "remote native import closure has missing modules: "
            f"{audit['missing_modules']}"
        )
    # A declared bundle module the closure never reaches is a stale declaration
    # (a renamed or deleted module), which would ship a bundle allowlist that
    # does not describe the artifact.
    declared = (
        REMOTE_NATIVE_KERNEL_MODULES
        if declared_kernel_modules is None
        else declared_kernel_modules
    )
    unreached = sorted(set(declared) - set(audit["modules"]))
    if unreached:
        raise ValueError(
            "REMOTE_NATIVE_KERNEL_MODULES declares modules the execd import "
            f"closure never reaches: {unreached}"
        )
    if audit["forbidden"]:
        violations = [
            f"{category}: {row['module']} via {row['edge']} (chain: {row['path']})"
            for category, rows in sorted(audit["forbidden"].items())
            for row in rows
        ]
        raise ValueError(
            "remote native import closure reaches forbidden Home dependencies:\n"
            + "\n".join(violations)
        )
    return audit


def _import_path(module: str, importer: Mapping[str, str]) -> list[str]:
    """Walk the recorded edges back to the seed that pulled ``module`` in."""

    chain = [module]
    seen = {module}
    current = module
    while current in importer:
        current = importer[current]
        if current in seen:
            break
        seen.add(current)
        chain.append(current)
    return list(reversed(chain))


def _module_matches_prefix(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def _parent_packages(module: str) -> tuple[str, ...]:
    parts = module.split(".")
    return tuple(".".join(parts[:index]) for index in range(1, len(parts)))


def _local_module_path(repo_root: pathlib.Path, module: str) -> pathlib.Path | None:
    relative = pathlib.Path(*module.split("."))
    module_file = repo_root / relative.with_suffix(".py")
    if module_file.is_file():
        return module_file
    package_file = repo_root / relative / "__init__.py"
    if package_file.is_file():
        return package_file
    return None


def _local_imports_by_scope(
    repo_root: pathlib.Path,
    module: str,
    module_path: pathlib.Path,
) -> dict[str, set[str]]:
    """Split this module's local-package imports into module scope vs. deeper.

    `ast.walk` reaches EVERY import statement, which is how a function-local
    `from ouroboros.config import …` becomes visible at all — the same traversal
    the platform gate already uses for its own function-local evasion class. The
    two buckets are returned separately because only module-scope edges may be
    followed transitively (see the header note).

    A dynamic import that spells the module's name as a STRING LITERAL counts as
    the same edge. Reading only `ast.Import`/`ast.ImportFrom` meant a bundle module
    could name a Home authority in plain source — `importlib.import_module(
    "ouroboros.artifacts")`, `__import__("ouroboros.tool_access")`, or the same
    through a bare/aliased `import_module` — and pass both this gate and the
    clean-subprocess smoke, which never runs the branch that imports.

    BOUNDARY: only literals are seen. `import_module("ouroboros." + name)`, a name
    read from a table, or `sys.modules[...]` require executing the program and are
    NOT detected here — that residue belongs to review, exactly as the platform
    gate's own boundary note says. What is closed is every form that names the Home
    module in the source, which is every form that can be audited at all.
    """

    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    module_scope: set[int] = set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_scope.add(id(node))
            continue
        if isinstance(node, ast.Call):
            # Recorded AND descended into: a module-scope call can nest another.
            module_scope.add(id(node))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        pending.extend(ast.iter_child_nodes(node))
    package = module if module_path.name == "__init__.py" else module.rpartition(".")[0]
    dynamic_importers = _dynamic_importer_names(tree)
    found: dict[str, set[str]] = {"module": set(), "local": set()}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            candidates = _dynamic_import_literals(node, dynamic_importers)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            candidates = _imported_module_names(node, package)
        else:
            continue
        bucket = "module" if id(node) in module_scope else "local"
        for candidate in candidates:
            if _local_module_path(repo_root, candidate) is not None:
                found[bucket].add(candidate)
    return found


def _dynamic_importer_names(tree: ast.Module) -> frozenset[str]:
    """Local names bound to a dynamic importer, plus the two canonical ones.

    `from importlib import import_module as imp` makes the importer a plain Name,
    so matching only `<x>.import_module(...)` and the builtin `__import__` left the
    bare-name spelling unchecked.
    """

    names = {"__import__", "import_module"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in ("importlib", "builtins"):
            continue
        for alias in node.names:
            if alias.name in ("import_module", "__import__"):
                names.add(alias.asname or alias.name)
    return frozenset(names)


def _dynamic_import_literals(
    node: ast.Call,
    dynamic_importers: frozenset[str],
) -> list[str]:
    """The module names one dynamic-import CALL spells as a string literal."""

    func = node.func
    is_importer = (
        isinstance(func, ast.Attribute) and func.attr in ("import_module", "__import__")
    ) or (isinstance(func, ast.Name) and func.id in dynamic_importers)
    if not is_importer or not node.args:
        return []
    first = node.args[0]
    if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
        return []
    name = first.value
    # `import_module(".sibling", "ouroboros")` — a relative spelling needs its
    # package to mean anything, and only a literal package can be resolved here.
    if name.startswith("."):
        anchor = node.args[1] if len(node.args) > 1 else None
        anchor = anchor if anchor is None else getattr(anchor, "value", None)
        for keyword in node.keywords:
            if keyword.arg == "package":
                anchor = getattr(keyword.value, "value", None)
        if not isinstance(anchor, str):
            return []
        try:
            name = importlib.util.resolve_name(name, anchor)
        except (ImportError, ValueError):
            return []
    return [name]


def _imported_module_names(
    node: ast.Import | ast.ImportFrom,
    package: str,
) -> list[str]:
    """Every dotted module name one import statement could bind."""

    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        relative = "." * node.level + str(node.module or "")
        try:
            base = importlib.util.resolve_name(relative, package)
        except (ImportError, ValueError):
            return []
    elif node.module:
        base = node.module
    else:
        return []
    return [base, *(f"{base}.{alias.name}" for alias in node.names)]



# ── the Home/execd capability contract (RWS v2 §3.1 admission) ──────────
#
# The manifest is HOME's authority and nothing else's: it is what a handshake is
# compared against, so a target that answers with a different one is refused
# rather than trusted. It is built from the registry's UNFILTERED built-in
# schemas, which is why it is deterministic — the same build produces the same
# digest regardless of which task, placement or visibility filter is in play.
#
# `WORKSPACE_TOOL_EXECUTION_AFFINITY` names, per tool, WHOSE facts an operation
# is about. It is not a routing table and deliberately does not decide placement:
# that answer belongs to the sealed `WorkspaceRef` and the root matrix in
# `workspace_ref.py`. Its job here is exhaustiveness — a new workspace-surface
# tool with no declaration fails closed instead of being silently classified.
#
# Exhaustiveness is checked against the REGISTRY, not against a second
# hand-written list. Comparing this table to `_WORKSPACE_ALLOWED_TOOLS` only ever
# proved the two constants agreed with each other; a genuinely new built-in in
# NEITHER of them passed every gate and was classified Home-only by silence.
# `HOME_ONLY_TOOL_NAMES` below closes that by making the Home answer an explicit
# declaration, so adding a tool forces the placement question to be answered.
#
# NOTE for the next phase: this is about BUILT-IN tools. There is no
# `execution_affinity` field on a SKILL manifest — see `HOME_ONLY_TOOL_NAMES`.
WORKSPACE_CAPABILITY_MANIFEST_SCHEMA_VERSION = 1

ROOT_AFFINITY_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "write_file", "edit_text", "search_code", "query_code",
    # Multi-file editors, target-native like their single-file sibling `edit_text`:
    # they are the token-efficient way to make a scattered change, so a remote task
    # having to fall back to one `edit_text` per site would be a real capability loss.
    "apply_patch", "edit_batch",
})
CWD_AFFINITY_TOOL_NAMES: frozenset[str] = frozenset({"run_command", "run_script"})
WORKSPACE_AFFINITY_TOOL_NAMES: frozenset[str] = frozenset({"vcs_status", "vcs_diff"})
SERVICE_AFFINITY_TOOL_NAMES: frozenset[str] = frozenset({
    "start_service", "service_status", "service_logs", "stop_service",
})
HYBRID_AFFINITY_TOOL_NAMES: frozenset[str] = frozenset({
    "verify_and_record",
    "schedule_subagent",
    "integrate_subagent_patch",
    "compare_subagent_patches",
    "browse_page",
    "browser_action",
    "analyze_screenshot",
    "vlm_query",
    "view_image",
    "ocr_pdf",
    "extract_video_frames",
})
HOME_AFFINITY_TOOL_NAMES: frozenset[str] = frozenset({
    "chat_history", "recent_tasks", "plan_task", "task_acceptance_review",
    "wait_task", "wait_tasks", "get_task_result", "peek_task", "cancel_task",
    "discard_child_result", "override_delegation_constraint",
    # The delegate_* quartet is the D10 succession of the retired claude_code_edit,
    # and it is Home-affine for the same reason `wait_task`/`cancel_task` are: these
    # are lifecycle calls against a run Home owns and Home alone can answer about.
    # The delegated CODER itself runs wherever the harness is logged in — putting
    # that on a target is the deferred external-coder bridge, not this change — so a
    # remote task can still start and steer a delegation, its Home half just stays
    # Home. `integrate_delegated_patch` joins them rather than mirroring
    # `integrate_subagent_patch`'s hybrid claim, because that claim was backed by the
    # remote bridge this branch removes.
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
    "integrate_delegated_patch",
    "knowledge_read", "knowledge_list", "knowledge_write",
    "journal_read", "journal_write", "workpad_read", "workpad_write",
    "tree_note", "tree_read", "web_search", "youtube_transcript",
    "list_available_tools", "enable_tools",
})

WORKSPACE_TOOL_EXECUTION_AFFINITY: dict[str, str] = {
    **{name: "root" for name in ROOT_AFFINITY_TOOL_NAMES},
    **{name: "cwd" for name in CWD_AFFINITY_TOOL_NAMES},
    **{name: "workspace" for name in WORKSPACE_AFFINITY_TOOL_NAMES},
    **{name: "service" for name in SERVICE_AFFINITY_TOOL_NAMES},
    **{name: "hybrid" for name in HYBRID_AFFINITY_TOOL_NAMES},
    **{name: "home" for name in HOME_AFFINITY_TOOL_NAMES},
}


# Built-ins that are deliberately NOT part of the workspace capability surface:
# they act on Home state (the repo body, the owner channel, the runtime control
# plane, the skills lifecycle, GitHub, the evolution machinery) and the registry
# filters them out of a workspace task's envelope entirely.
#
# This is a DECLARATION, not a leftover. Before it, "unclassified" and "Home-only"
# were the same state, so a new built-in was classified by omission and the
# exhaustiveness claim above compared two hand-written constants to each other
# instead of to the tools that exist. Adding a built-in now forces a choice: give
# it a workspace affinity, or say here that it is Home's.
#
# There is deliberately NO skill-manifest counterpart. Marrying skills to remote
# placement is a deferred phase (owner decision): `scripts[].execution_affinity`
# and `tool_execution_affinity` are not read by any loader and were removed from
# `docs/CREATING_SKILLS.md`, which had documented them as validated and
# load-blocking when nothing implemented them at all.
HOME_ONLY_TOOL_NAMES: frozenset[str] = frozenset({
    # the repo body and its review/commit gates
    "advisory_review", "commit_reviewed", "review_status", "request_deep_self_review",
    "vcs_commit_reviewed", "vcs_pull_ff", "vcs_restore", "vcs_revert", "vcs_rollback",
    "run_ci_tests", "codebase_health", "stage_adaptations", "promote_to_stable",
    # the owner channel
    "send_user_message", "send_photo", "send_video", "send_file",
    # runtime control plane
    "request_restart", "switch_model", "set_tool_timeout", "compact_context",
    "toggle_consciousness", "toggle_evolution", "generate_evolution_stats",
    # projects / routing / task steering
    "list_projects", "route_to_project", "promote_chat_to_task", "steer_task",
    "ensure_project_scope", "forward_to_worker",
    # cognitive memory
    "update_identity", "update_scratchpad", "memory_map", "memory_update_registry",
    # skills lifecycle
    "list_skills", "skill_exec", "skill_review", "skill_preflight", "toggle_skill",
    "submit_skill_to_hub",
    # GitHub / PR integration
    "create_github_issue", "get_github_issue", "list_github_issues",
    "close_github_issue", "comment_on_issue", "get_github_pr", "list_github_prs",
    "comment_on_pr", "fetch_pr_ref", "cherry_pick_pr_commits",
    "create_integration_branch", "stage_pr_merge",
})


# ── the ROUTING table of the execute phase (RWS v2 §3.1 step 4) ─────────────
#
# The affinity table above says whose FACTS an operation is about; this one says
# which native OPERATION a remote dispatch actually runs, and it is a separate
# table on purpose. Affinity is exhaustive over the whole workspace surface;
# routing is exhaustive over the target's declared operations, and the two names
# differ wherever the tool is a Home contract wrapped around a target check
# (`verify_and_record` records a Home receipt for a `verify_remote_check`).
#
# A tool ABSENT here stays Home-local under every placement. That is the whole
# reason the table exists: without it the prepare phase would ask the target to
# prepare `journal_write`, the target would refuse an operation it never declared,
# and a Home-only faculty would break on a remote task. The exhaustiveness test
# pins this against `MANDATORY_REMOTE_NATIVE_OPERATIONS`, so a new native
# operation cannot become reachable — or stay unreachable — by accident.
REMOTE_NATIVE_TOOL_OPERATION: dict[str, str] = {
    **{
        name: name
        for name in (
            ROOT_AFFINITY_TOOL_NAMES
            | CWD_AFFINITY_TOOL_NAMES
            | WORKSPACE_AFFINITY_TOOL_NAMES
            | SERVICE_AFFINITY_TOOL_NAMES
        )
    },
    "extract_video_frames": "extract_video_frames",
}
# `verify_and_record` is deliberately ABSENT even though the target declares
# `verify_remote_check`, and the absence is the CORRECT routing rather than a gap.
# The tool is a HYBRID whose Home half is the point of it: it writes the durable
# verification receipt the ledger and task acceptance read. Routing the whole tool
# would run the check on the target and record nothing, so the evidence would
# disappear with the session while the tool still reported success. So the two halves
# are wired separately (`tools/verify.py::_verify_on_remote_target`): the check runs on
# the target through the same prepared path, `bytes_equal` is compared THERE (design-
# partner P2 — comparing on Home would transfer both files in full for a fact that is
# one boolean), the after-check existence probe of each declared path is the target's,
# and Home records those attested facts as its own receipt, labelled with the surface
# that produced them so a remote green and a Home green are never silently the same
# evidence.


def remote_native_operation_for_tool(tool_name: str) -> str:
    """The native operation an ssh dispatch executes for ``tool_name``, or ``""``.

    Empty means Home-local under every placement: the tool has no target-side
    counterpart at all. This answers the TOOL half of the routing question; the
    CALL half is the root matrix below, and a dispatch needs both to be true
    before an operation may leave Home.
    """

    return REMOTE_NATIVE_TOOL_OPERATION.get(str(tool_name or "").strip(), "")


# ── the CALL half of routing: which root is this operation about? ────────────
#
# `REMOTE_NATIVE_TOOL_OPERATION` is per TOOL, and per tool is not enough. `read_file`
# is the same tool whether it reads the target's worktree or Home's artifact store,
# and only its `root` argument says which — so a per-tool routing answer necessarily
# sends BOTH to the same place. It sent both to the target, where `root` is not
# modelled at all, and the target answered about its own workspace: the model asked
# for the task's artifact and was handed a same-named file from the remote project.
# Consulting `workspace_ref.root_is_target_native` at the routing point is what gives
# the ratified matrix (Q2а) force over dispatch instead of only over path accessors.
#
# A tool ABSENT here carries no root label, and that is a positive statement rather
# than a gap: its operation is about the active workspace by construction — a process
# cwd (`run_command`, `run_script`, the service quartet), the workspace's own git
# (`vcs_status`, `vcs_diff`), a workspace-relative media path
# (`extract_video_frames`) — so the placement alone decides where it runs.
ROOT_LABELLED_TOOL_ARG: dict[str, str] = {name: "root" for name in ROOT_AFFINITY_TOOL_NAMES}
# The root a root-labelled call falls back to when it names none. It is the schema
# default of all six of those tools, and it has to be spelled HERE too, because
# routing reads the caller's RAW arguments — before any handler applies its own.
IMPLICIT_RESOURCE_ROOT = "active_workspace"


def dispatch_resource_root(tool_name: str, args: Mapping[str, Any] | None = None) -> str:
    """The resource root THIS CALL of ``tool_name`` resolves under."""

    key = ROOT_LABELLED_TOOL_ARG.get(str(tool_name or "").strip(), "")
    if not key:
        return IMPLICIT_RESOURCE_ROOT
    return str((args or {}).get(key) or "").strip() or IMPLICIT_RESOURCE_ROOT


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_copy(value: Any, *, label: str) -> dict[str, Any]:
    try:
        copied = json.loads(_canonical_json_bytes(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON: {exc}") from exc
    if not isinstance(copied, dict):
        raise ValueError(f"{label} must be an object")
    return copied


def build_workspace_capability_manifest(
    public_schemas: Iterable[Mapping[str, Any]],
    *,
    repo_root: pathlib.Path,
    operation_modules: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the canonical Home/execd capability contract from registry schemas.

    The caller supplies the registry's unfiltered built-in schemas. THIS module
    owns which of those tools form the workspace surface; execd owns only the
    explicit native operation allowlist. `native_kernel_modules` is the declared
    bundle kernel, NOT the closure's roots — the closure seeds the Home-side
    transport in the reverse direction, and publishing that as the bundle's
    contents would misdescribe what actually travels.
    """

    from ouroboros.workspace_native_contract import (
        REMOTE_NATIVE_KERNEL_MODULES,
        REMOTE_NATIVE_OPERATION_MODULE,
    )

    native_map = (
        REMOTE_NATIVE_OPERATION_MODULE
        if operation_modules is None
        else operation_modules
    )
    schemas_by_name: dict[str, dict[str, Any]] = {}
    for raw in public_schemas:
        schema = _canonical_json_copy(raw, label="public tool schema")
        function = schema.get("function")
        name = str(function.get("name") or "").strip() if isinstance(function, dict) else ""
        if schema.get("type") != "function" or not name:
            raise ValueError("public tool schema must be a named function envelope")
        if name in schemas_by_name:
            raise ValueError(f"duplicate public tool schema: {name}")
        schemas_by_name[name] = schema

    expected_names = frozenset(WORKSPACE_TOOL_EXECUTION_AFFINITY)
    missing = sorted(expected_names - schemas_by_name.keys())
    if missing:
        raise ValueError(f"workspace capability manifest is missing public schemas: {missing}")
    public_tools = [schemas_by_name[name] for name in sorted(expected_names)]
    import_audit = assert_remote_native_import_closure(
        pathlib.Path(repo_root),
        operation_modules=native_map,
    )
    payload: dict[str, Any] = {
        "schema_version": WORKSPACE_CAPABILITY_MANIFEST_SCHEMA_VERSION,
        "public_tools": public_tools,
        "public_schema_sha256": hashlib.sha256(_canonical_json_bytes(public_tools)).hexdigest(),
        "native_operations": [
            {"name": name, "module": str(native_map[name])} for name in sorted(native_map)
        ],
        "native_kernel_modules": sorted(REMOTE_NATIVE_KERNEL_MODULES),
        "native_import_modules": list(import_audit["modules"]),
        "native_import_edges": dict(import_audit["edges"]),
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _validated_workspace_capability_manifest(
    raw: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    from ouroboros.workspace_native_contract import validate_remote_native_operation_map

    manifest = _canonical_json_copy(raw, label=f"{label} capability manifest")
    if manifest.get("schema_version") != WORKSPACE_CAPABILITY_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"{label} capability manifest has an unsupported schema version")
    claimed_manifest_hash = str(manifest.pop("manifest_sha256", "") or "")
    if claimed_manifest_hash != hashlib.sha256(_canonical_json_bytes(manifest)).hexdigest():
        raise ValueError(f"{label} capability manifest hash is invalid")
    public_tools = manifest.get("public_tools")
    if not isinstance(public_tools, list):
        raise ValueError(f"{label} capability manifest public_tools must be a list")
    if manifest.get("public_schema_sha256") != hashlib.sha256(
        _canonical_json_bytes(public_tools)
    ).hexdigest():
        raise ValueError(f"{label} capability manifest public schema hash is invalid")
    native_operations = manifest.get("native_operations")
    if not isinstance(native_operations, list):
        raise ValueError(f"{label} capability manifest native_operations must be a list")
    operation_map: dict[str, str] = {}
    for item in native_operations:
        if not isinstance(item, dict):
            raise ValueError(f"{label} native operation rows must be objects")
        name = str(item.get("name") or "")
        if not name or name in operation_map:
            raise ValueError(f"{label} native operation names must be non-empty and unique")
        operation_map[name] = str(item.get("module") or "")
    validate_remote_native_operation_map(operation_map)
    manifest["manifest_sha256"] = claimed_manifest_hash
    return manifest


def assert_workspace_capability_compatible(
    home_manifest: Mapping[str, Any],
    backend_manifest: Mapping[str, Any],
) -> None:
    """Fail admission on a missing native operation or public-schema drift."""

    home = _validated_workspace_capability_manifest(home_manifest, label="Home")
    backend = _validated_workspace_capability_manifest(
        backend_manifest,
        label="workspace backend",
    )
    if home["native_operations"] != backend["native_operations"]:
        raise ValueError("workspace backend native capability allowlist differs from Home")
    if home["public_schema_sha256"] != backend["public_schema_sha256"]:
        raise ValueError("workspace backend public tool schema digest differs from Home")
    if home["public_tools"] != backend["public_tools"]:
        raise ValueError("workspace backend public tool schemas differ from Home")
    for field in ("native_kernel_modules", "native_import_modules", "native_import_edges"):
        if home.get(field) != backend.get(field):
            raise ValueError(
                f"workspace backend {field} differs from the Home import closure"
            )


CORE_TOOL_NAMES: frozenset[str] = frozenset({
    "read_file", "list_files", "write_file", "edit_text",
    "apply_patch", "edit_batch",
    "search_code", "query_code", "plan_task",
    "run_command", "run_script",
    "start_service", "service_status", "service_logs", "stop_service",
    "vcs_status", "vcs_diff", "vcs_commit_reviewed", "commit_reviewed",
    "vcs_restore", "vcs_revert", "vcs_pull_ff", "vcs_rollback",
    "schedule_subagent", "integrate_subagent_patch", "compare_subagent_patches",
    "integrate_delegated_patch",
    "wait_task", "wait_tasks", "get_task_result",
    # D#7 soft-join child controls (siblings of steer_task): inspect/decide a child's fate
    # before finalizing (peek = pure read, discard = explicit abandon, cancel = real stop).
    "cancel_task", "peek_task", "discard_child_result", "override_delegation_constraint",
    # Task-tree coordination must be in the round-one envelope so a parent can publish the
    # shared frame BEFORE fanning out interdependent children (no enable_tools detour).
    "tree_note", "tree_read",
    # Main-chat routing capabilities the SYSTEM.md decision turn relies on
    # (kept in the core envelope so the anti-freeze ephemeral turn never needs an
    # enable_tools detour to route — though initial_tool_schemas exposes the full
    # set today, this makes the coupling explicit).
    "list_projects", "route_to_project", "promote_chat_to_task", "steer_task",
    "ensure_project_scope",
    "update_scratchpad", "update_identity",
    "chat_history", "recent_tasks",
    "knowledge_read", "knowledge_write", "knowledge_list",
    "web_search",
    "browse_page", "browser_action", "analyze_screenshot", "view_image",
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
    "send_user_message", "send_photo", "send_video", "send_file",
    "switch_model",
    "request_restart", "promote_to_stable",
    "advisory_review", "review_status", "task_acceptance_review", "verify_and_record",
    # Heal mode blocks enable_tools, so repair/review tools must be core.
    "list_skills", "skill_review", "skill_preflight",
    "submit_skill_to_hub",
})

# Meta-tools: always visible alongside core tools
META_TOOL_NAMES: frozenset[str] = frozenset({
    "list_available_tools", "enable_tools",
})

LOCAL_READONLY_SUBAGENT_MODE: str = "local_readonly_subagent"

# V1 subagents are read-only against local Ouroboros state. Browser interaction
# remains available by explicit product decision, so this mode is not a remote
# website sandbox.
LOCAL_READONLY_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset({
    # switch_model changes COGNITIVE POWER, not authority: a child that started on
    # the cheap lane and finds the work harder raises itself instead of failing or
    # asking the parent to respawn it (BIBLE P5). Nothing about the sandbox changes.
    "switch_model",
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
    "chat_history", "recent_tasks", "get_task_result", "wait_task", "wait_tasks",
    "schedule_subagent",
    # Task-tree coordination: a child reads the shared frame and raises beacons. tree_note
    # is a bounded tree-scoped write; its tagged child-result disposition branch also
    # updates the existing child result through join_ledger's lineage/hash authority.
    # It has no repo/control-plane effect, so remains valid for read-only subagents.
    "tree_note", "tree_read", "override_delegation_constraint",
    # Nanny verbs. The child gets no shell — it gets the right to ASK the host to run a
    # session, and the host derives the access profile from THIS task's authority, so a
    # read-only child can only ever host a read-only session. delegate_answer speaks
    # only to a run this task already owns (custody-gated like cancel).
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
    "web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query", "view_image",
    # Bounded media projection: writes derived frames only under artifact_store/video_frames.
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
})

ACTING_SUBAGENT_MODE: str = "acting_subagent"

# Mutative ("acting") subagents may write inside an isolated write root
# (self_worktree / external_workspace) and run shell/services there.
# They explicitly CANNOT commit the live body (commit_reviewed /
# vcs_commit_reviewed), run runtime control, touch the skills lifecycle, enable
# tools, or write cognitive memory (update_identity/update_scratchpad/
# knowledge_write). The parent integrates and is the sole committer. Extension /
# MCP tools are denied unless explicitly granted per-child via
# TaskConstraint.external_tool_grants.
ACTING_SUBAGENT_TOOL_NAMES: frozenset[str] = frozenset({
    # switch_model changes COGNITIVE POWER, not authority: a child that started on
    # the cheap lane and finds the work harder raises itself instead of failing or
    # asking the parent to respawn it (BIBLE P5). Nothing about the sandbox changes.
    "switch_model",
    "read_file", "list_files", "search_code", "query_code",
    "vcs_status", "vcs_diff",
    "write_file", "edit_text",
    "apply_patch", "edit_batch",
    "run_command", "run_script",
    "start_service", "service_status", "service_logs", "stop_service",
    "integrate_subagent_patch", "compare_subagent_patches",
    "schedule_subagent", "wait_task", "wait_tasks", "get_task_result",
    "verify_and_record",
    "knowledge_read", "knowledge_list",
    "tree_note", "tree_read", "override_delegation_constraint",
    # Same nanny verbs, same host-derived profile — an acting child hosts a
    # workspace_write session confined to a private snapshot of its own write
    # root, and explicitly integrates the captured diff (C1).
    "delegate_start", "delegate_wait", "delegate_cancel", "delegate_answer",
    "integrate_delegated_patch",
    "web_search", "browse_page", "browser_action", "analyze_screenshot", "vlm_query", "view_image",
    "ocr_pdf", "youtube_transcript", "extract_video_frames",
    "list_available_tools",
})

READ_ONLY_PARALLEL_TOOLS: frozenset[str] = frozenset({
    "read_file", "list_files",
    "search_code", "query_code", "recent_tasks",
    "web_search", "chat_history",
    "vcs_status", "vcs_diff", "service_status", "service_logs",
    "get_task_result", "list_projects",
})

# Enqueue-only tools safe to emit in parallel within one tool-call round.
# schedule_subagent is fire-and-forget: it writes a `requested` task result and
# does event_queue.put_nowait(...) with no blocking LLM/RPC on the parent path.
# Parent-side shared ctx state touched during emission is guarded by
# _SCHEDULE_EMIT_LOCK in tools/control.py; the supervisor still drains EVENT_Q
# serially, so cap/dedup/enqueue remain single-threaded and safe.
PARALLEL_SAFE_ENQUEUE_TOOLS: frozenset[str] = frozenset({"schedule_subagent"})

# Stateful browser tools need the thread-sticky executor.
STATEFUL_BROWSER_TOOLS: frozenset[str] = frozenset({
    "browse_page", "browser_action",
})

# Full outputs are semantic (review verdicts, advisory findings, status).
UNTRUNCATED_TOOL_RESULTS: frozenset[str] = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
    "plan_task",
    "task_acceptance_review",
    "advisory_review",
    "skill_review",
    "review_status",
    "get_task_result",
    "wait_task",
    "wait_tasks",
})

# Cognitive artifacts must not be truncated.
UNTRUNCATED_REPO_READ_PATHS: frozenset[str] = frozenset({
    "BIBLE.md",
    "README.md",
    "docs/ARCHITECTURE.md",
    "docs/CHECKLISTS.md",
    "docs/DEVELOPMENT.md",
})

# Per-tool char caps; omitted tools use DEFAULT_TOOL_RESULT_LIMIT.
TOOL_RESULT_LIMITS: dict[str, int] = {
    "read_file": 80_000,
    "recent_tasks": 80_000,
    "knowledge_read": 80_000,
    "run_command": 80_000,
    "run_script": 80_000,
    "search_code": 80_000,
    "query_code": 80_000,
    "service_logs": 80_000,
    # Best-of-N patch comparison shows several candidate diffs side by side; the
    # default 15k cap would truncate after the first one and defeat the tool.
    "compare_subagent_patches": 80_000,
    # skill_exec wraps stdout/stderr; keep the full capped payload visible.
    "skill_exec": 300_000,
    # tree_read returns the shared task-tree coordination tail (up to 200 entries); the 15k
    # default would truncate the swarm blackboard and defeat the coordination contract.
    "tree_read": 80_000,
    # apply_patch results carry per-hunk diagnostics, edit_batch per-edit ones
    # (an aborted batch reports EVERY failed edit so one retry can fix them all);
    # write_file appends the overwrite diff.
    "apply_patch": 80_000,
    "edit_batch": 80_000,
    "write_file": 80_000,
}

DEFAULT_TOOL_RESULT_LIMIT: int = 15_000


def tool_result_limit(tool_name: str) -> int:
    """The char budget a tool's result is delivered under.

    Read by the truncator AND by producers that must fit inside it: a tool whose payload
    is structured JSON has to bound itself, because outer head-truncation cuts mid-string
    and destroys the document. Both sides asking the same function is what keeps a
    producer's idea of "small enough" from drifting away from the cap actually applied.
    """
    return TOOL_RESULT_LIMITS.get(str(tool_name or ""), DEFAULT_TOOL_RESULT_LIMIT)


# Reviewed mutative tools must not end with ambiguous executor timeouts.
REVIEWED_MUTATIVE_TOOLS: frozenset[str] = frozenset({
    "commit_reviewed",
    "vcs_commit_reviewed",
})

# Foreground mutative tools may keep editing files after Python future timeout;
# the loop must wait for terminal completion instead of returning while they run.
# Empty since D10 retired the SDK edit gateway (the one foreground tool that
# kept editing after a Python-future timeout); the seam stays for successors.
FOREGROUND_MUTATIVE_TOOLS: frozenset[str] = frozenset()
