#!/usr/bin/env python3
"""Validate the parseable MIGRATION_v7.md contract against the live tree.

This module owns the migration-table half of the v7 prologue evidence: parsing
the canonical table, resolving symbol references, and demanding rows for every
migration-relevant drift class between the immutable baseline and the current
candidate tree.

Both languages are modelled through one lexical module-surface representation:
``name -> {(kind, provider path, provider symbol), ...}`` where the set holds
every possible binding alternative across straight-line code and module-scope
``if``/``try`` branches, so branch order can never hide an incompatible
alternative. Python kinds are ``class``/``function``/``assignment`` (owned
symbols, dunders included), ``reexport`` (named ImportFrom carrying its exact
resolved provider path and imported symbol) and ``import`` (an ordinary import
binding, which never owns a symbol). JavaScript binding kinds are ``class``,
``function`` (declarations, arrow functions and function expressions alike),
``variable`` and ``import``; the ES export surface is tracked alongside as
``exported name -> (provider specifier, source symbol)`` with the actual local
binding symbol recorded for locally provided exports. Drift demands a
migration row when:

- a tracked path is deleted, renamed away or type-changed (every supported
  baseline symbol identity needs a row unless one path-level row owns the
  whole file),
- a baseline identity vanishes, is masked by a strict-kind-incompatible
  binding, or its conditional binding alternatives change (exported
  JavaScript identities backed by local bindings keep strict kinds too),
- a baseline identity keeps its name but changes provider: extraction to a
  re-export, a re-export pointing at a new owner path or source symbol, and a
  re-export inlined back into a local implementation all demand a row whose
  owner cell equals the exact expected provider identity (``path::symbol``,
  ``path`` for namespace re-exports, or ``external:<specifier>::<symbol>``
  for bare/external JavaScript providers). An extraction to a non-local
  provider also requires a facade cell naming the exact old identity, and
  the old path must structurally re-export exactly the declared owner path
  and source symbol, which a repo-local JavaScript owner must publicly
  export (a copied local implementation, an unrelated symbol or a private
  owner binding is never a facade; spec 4.4 pending rows stay valid before
  their extraction occurs). An inlined re-export is provider drift without an extraction
  facade, so its facade cell stays ``-``.

Symbol resolution is lexical and conservative: Python ``__getattr__`` never
satisfies a specific name, ordinary import bindings are references rather
than owned symbols, and wildcard sources (bare ``export *`` and Python
``from x import *``) make an affected module surface unverifiable, which
fails closed with a deterministic error instead of a silent pass (explicit
``export * as ns``/``import * as ns`` are exact namespace bindings).
JavaScript sources are parsed with the real tree-sitter grammar (a main
runtime dependency), so comments and string literals can never masquerade as
symbol definitions. Any unreadable or unparseable source, missing parser, or
unavailable first-party predicate fails closed with a deterministic visible
error instead of silently skipping.
"""
from __future__ import annotations
import ast
import functools
import json
import os
import pathlib
import posixpath
import re
import subprocess
import sys
import tempfile
from typing import Any, Iterable
# The campaign's immutable provenance anchor: the tree the v7 plan was written
# against. The frozen prologue evidence (census, source hashes, contract and
# safety-differential snapshots) is bound to it and must never move, or the
# campaign loses the baseline its acceptance is measured from.
BASELINE_SHA = "a191e1cc21a380176bcedc9b8edd86078fc87fa1"
# The exact merge-base the branch is currently built on. The migration ledger
# records v7-AUTHORED moves only, so this pin travels with every tactical rebase
# (owner decision, 2026-08-16): left behind, ordinary upstream refactors become
# phantom "missing migration" demands and bury the real rows. Update it in the
# same commit as the rebase.
MERGE_BASE_SHA = "8028f1df864743dcc7543b83b6e23d65db5f9e0c"
MIGRATION_PATH = pathlib.PurePosixPath("MIGRATION_v7.md")
MIGRATION_HEADERS = ("old path/symbol", "new owner/path", "facade/public contract", "semantic delta", "characterization test", "upstream-transfer status/note")
# Semantic delta ids are a shared registry, one per plan §4.3 item (legend duplicated in
# the MIGRATION_v7.md header, which is the reader-facing copy):
#   D02 §4.3.3 typed tool results · D03 §4.3.5 settings seam ·
#   D04 §4.3.6 retired settings knobs · D05 §4.3.8 safety host facts · D06 §4.3.12 events taxonomy ·
#   D07 §4.3.11 Emergency Stop 2A · D08 §4.3.13 cancellation/delegation fail-closed registries ·
#   D09 §4.3.2 LLM local retry (one physical attempt) · D11 §1.9/№8 FUNCTION_DEBT same-qualname
#   relocation rule · D13 §6.4 supervisor/git_ops pre-init roots follow OUROBOROS_* env
#   (hermetic-isolation incident fix; ratified by owner batch №11, spec §1.12) · D18 §1.9/№8 module-handle
#   reads of rebound supervisor globals in extracted leaves · D31 §1.14-2 the contributor review trust
#   boundary (owner decision 2026-08-19, superseding batch №14 answer 2=A): the per-proposal classifier
#   — hand-list, then anchors plus name rule plus base-flow import closure — retires whole, because the
#   contributor lane now hands the review off to the target base's own machinery for every proposal, so
#   there is nothing left to classify · D33 §1.9/№8-pattern
#   module-handle reads of monkeypatchable loop facade bindings in the L-B leaves (the ratified
#   supervisor mechanism applied to the loop stream with its own id per the §1.9-1 "separate delta
#   id" rule; owner-ratified, batch №17 answer 2=A; leaves hold no mutable state, the handle
#   exists so tests patching loop.X keep intercepting) · D34 §1.9-10 carrier-aware update engine
#   (owner-ratified batch №8 answer 6=A / spec §1.9-10): the shared span-substitution resolver
#   (supervisor/update_carriers.py, span descriptors SSOT in ouroboros/tools/release_sync.py) is
#   applied at the three managed-update insertion points before write-tree; malformed/duplicate
#   anchors and conflicts outside a carrier span stay on the assisted path, never whole-file theirs ·
#   D35 §1.9/№8-pattern module-handle reads of rebindable git_ops globals in the G1 leaves
#   (`init` rebinds REPO_DIR/DRIVE_ROOT/BRANCH_* and tests monkeypatch the capture plumbing and
#   sibling members on the parent; the §1.9-1 mechanism with a separate id per stream — the
#   per-leaf `_go()` read sets are pinned in tests/test_module_handle_extraction.py).
#   anchors and conflicts outside a carrier span stay on the assisted path, never whole-file theirs.
#   · D37 §1.9/№8-pattern module-handle reads of monkeypatchable review-stack facade bindings in the
#   L-C leaves (the ratified mechanism applied to the review stream with its own id per the §1.9-1
#   "separate delta id" rule, exactly as D33 did for the loop stream; handles `_rev()`/`_car()`,
#   leaves hold no mutable state, the handle exists so tests patching/rebinding the parent's
#   `tools.review.X` / `tools.claude_advisory_review.X` bindings keep intercepting the moved bodies).
#   D38 §1.9/№8-pattern module-handle reads of monkeypatchable agent.py / usage_accounting.py facade
#   bindings in the L-C2 leaves (the same ratified supervisor mechanism applied to the L-C2 stream
#   with its own id per the §1.9-1 "separate delta id" rule; leaves hold no mutable state, the
#   handle exists so tests patching the parent binding keep intercepting the moved bodies).
# "D01" (reserved for §4.3.1 size-ratchet layers) was retired unused (owner-ratified, batch №11):
# ratchet-layer changes are governed by size_ratchet.json + scripts/regenerate_size_ratchet.py, not
# by ledger rows.
# Before assigning ANY new id, prove it free with `git grep -n "\bDnn\b"`: the runtime prose
# already uses a two-digit sprint-decision namespace ("(D12)" review context on the fly,
# "(D14)".."(D17)", "(D19)"+ in reviewer_slot_config/claudexor_daemon/subagents/review_context_atlas),
# and every collision splits one label across two meanings. Skipped for exactly that reason:
# "D10" (historical claude_code_edit retirement — docs/DEVELOPMENT.md "D10 postmortem"),
# "D12" and "D14"–"D17" (occupied by that prose namespace; D12 was briefly used for the
# module-handle delta by one fix commit before the collision was caught in the delta re-gate).
# Two S3b commit messages say "D10" and one says "D12" for the module-handle delta; commit
# history is immutable — the ledger and this registry are the id authority: it is D18.
#   · D36 §1.9/№8-pattern module-handle reads in the DEL1 delegate-family leaves
#   (delegate_custody / tools.delegate / delegate_integration / subagent_integration;
#   renumbered from the lane's provisional D35 after the G1 collision).
APPROVED_SEMANTIC_DELTAS = frozenset({"none", "D02", "D03", "D04", "D05", "D06", "D07", "D08", "D09", "D11", "D13", "D18", "D31", "D33", "D34", "D35", "D36", "D37", "D38"})
UPSTREAM_STATUSES = frozenset({"not_applicable", "pending", "transferred", "retired"})
APPROVED_PENDING_OWNERS = frozenset({
    "ouroboros/tools/tool_context.py", "ouroboros/tools/tool_catalog.py", "ouroboros/tools/tool_result.py",
    "ouroboros/tools/tool_resolution.py", "ouroboros/tools/registry_core.py", "ouroboros/tools/registry_guards.py", "ouroboros/tools/registry_guard_process.py", "ouroboros/tools/extension_dispatch.py",
    "ouroboros/tools/core_artifacts.py", "ouroboros/tools/core_file_tools.py",
    "ouroboros/tools/git_plumbing.py", "ouroboros/tools/git_review_cycle.py", "ouroboros/tools/git_evolution.py", "ouroboros/tools/git_repo_edit.py", "ouroboros/tools/git_vcs_ops.py",
    "ouroboros/tools/shell_process.py", "ouroboros/tools/shell_outputs.py", "ouroboros/tools/shell_effects.py",
    "ouroboros/tools/scope_review_budget.py", "ouroboros/tools/scope_review_pack.py",
    "ouroboros/tools/review_prompt_text.py", "ouroboros/tools/review_file_pack.py",
    "ouroboros/review_state_records.py", "ouroboros/review_state_model.py",
    "ouroboros/headless_status.py", "ouroboros/workspace_patch_capture.py",
    "ouroboros/settings_defaults.py", "ouroboros/settings_scales.py", "ouroboros/model_slots.py",
    "ouroboros/review_model_routes.py", "ouroboros/runtime_limits.py",
    "ouroboros/tool_access_types.py", "ouroboros/tool_access_paths.py", "ouroboros/tool_access_roots.py", "ouroboros/tool_access_user_files.py",
    "ouroboros/llm_attempt.py", "ouroboros/llm_capability_policy.py", "ouroboros/llm_routing.py",
    "ouroboros/llm_messages.py", "ouroboros/llm_fallback.py", "ouroboros/llm_anthropic.py",
    "ouroboros/llm_gigachat.py", "ouroboros/llm_local.py", "ouroboros/llm_openai_compatible.py",
    "ouroboros/llm_pricing.py",
    "ouroboros/review_records.py", "ouroboros/review_verdict.py", "ouroboros/review_projection.py",
    "ouroboros/review_evidence_sections.py",
    "ouroboros/skill_review_packs.py", "ouroboros/skill_review_rebuttals.py",
    "ouroboros/skill_review_prompt.py", "ouroboros/skill_review_output.py",
    "skills/unix_computer_use/lib/cu_runtime.py", "skills/unix_computer_use/lib/cu_connections.py", "skills/unix_computer_use/lib/cu_remote_backends.py",
    "devtools/benchmarks/osworld/cu_bridge_runtime.py", "devtools/benchmarks/osworld/cu_bridge_prompts.py", "devtools/benchmarks/osworld/cu_bridge_tool_policy.py",
    "devtools/benchmarks/osworld/cu_bridge_gate.py", "devtools/benchmarks/osworld/cu_bridge_budget.py",
    "devtools/benchmarks/osworld/step_agent_common.py", "devtools/benchmarks/osworld/step_agent_env.py", "devtools/benchmarks/osworld/step_agent_claims.py",
    "devtools/benchmarks/osworld/step_agent_actions.py", "devtools/benchmarks/osworld/step_agent_policy.py",
    "web/tests/harness_accounts_helpers.js", "web/tests/harness_accounts_cards.test.js",
    "web/tests/harness_accounts_custody.test.js", "web/tests/harness_accounts_panel.test.js",
    "supervisor/events_chat_delivery.py", "supervisor/events_subagent_admission.py",
    "supervisor/events_schedule_task.py", "supervisor/events_project_routing.py",
    "supervisor/events_coop_checkpoint.py", "supervisor/events_evolution_done.py",
    "supervisor/events_task_done.py", "supervisor/events_budget.py",
    "supervisor/events_worker_reports.py", "supervisor/events_runtime_controls.py",
    "supervisor/cancel_custody.py", "supervisor/worker_process.py",
    "ouroboros/server_process.py", "ouroboros/server_routing_context.py", "ouroboros/server_owner_routing.py",
    "ouroboros/server_liveness.py", "ouroboros/server_maintenance.py", "ouroboros/server_restart.py",
    "ouroboros/tools/control_events.py", "ouroboros/tools/control_routing.py",
    "ouroboros/tools/control_subagent_spec.py", "ouroboros/tools/control_scheduling.py",
    "ouroboros/tools/control_runtime.py", "ouroboros/tools/control_task_results.py",
    "ouroboros/extension_registry_state.py", "ouroboros/extension_surface_names.py",
    "ouroboros/extension_child_catalog.py", "ouroboros/extension_import_staging.py",
    "ouroboros/extension_liveness.py", "ouroboros/extension_plugin_api.py",
    "supervisor/queue_snapshot.py", "supervisor/queue_timeouts.py",
    "supervisor/queue_schedules.py", "supervisor/queue_evolution.py",
    "supervisor/worker_promotion.py", "supervisor/worker_chat_lane.py",
    "supervisor/worker_health.py", "supervisor/worker_pool_lifecycle.py",
    "supervisor/worker_assignment.py",
})
_PY_LOCAL_KINDS = frozenset({"class", "function", "assignment"})
def _git(repo: pathlib.Path, *args: str, text: bool = True) -> str | bytes:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=text).stdout
def _source_text(repo: pathlib.Path, ref: str, path: str) -> str:
    return _git(repo, "show", f"{ref}:{path}", text=False).decode("utf-8", errors="strict")  # type: ignore[union-attr]
def _tracked_paths(repo: pathlib.Path, ref: str) -> list[str]:
    return sorted(line for line in str(_git(repo, "ls-tree", "-r", "--name-only", ref)).splitlines() if line)
def _parse_ref(cell: str) -> tuple[str, str]:
    if "::" not in cell: return cell, ""
    return tuple(cell.split("::", 1))  # type: ignore[return-value]
def _parse_migration(path: pathlib.Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 2:
        raise ValueError("MIGRATION_v7.md has no canonical table")
    header = tuple(cell.strip() for cell in table_lines[0].strip("|").split("|"))
    if header != MIGRATION_HEADERS:
        raise ValueError(f"migration header/order mismatch: {header!r}")
    separator = tuple(cell.strip() for cell in table_lines[1].strip("|").split("|"))
    if len(separator) != len(MIGRATION_HEADERS) or any(not re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        raise ValueError("migration separator is malformed")
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(MIGRATION_HEADERS):
            raise ValueError(f"migration row has {len(cells)} cells: {line}")
        rows.append(dict(zip(MIGRATION_HEADERS, cells)))
    return rows
def _migration_json(cell: str, keys: tuple[str, ...]) -> dict[str, str]:
    value = json.loads(cell)
    if not isinstance(value, dict) or tuple(value) != keys or not all(isinstance(item, str) for item in value.values()):
        raise ValueError(f"expected ordered string object with keys {list(keys)}")
    compact = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if compact != cell or any("|" in item for item in value.values()):
        raise ValueError("cell must be canonical compact JSON without pipes")
    return value
def _assignment_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, ast.Starred):
        yield from _assignment_names(target.value)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _assignment_names(element)
def _python_module_surface(text: str, path: str) -> tuple[dict[str, frozenset[tuple[str, str, str]]], bool]:
    """Map module-scope names to binding-alternative sets; flag wildcard imports.

    Each alternative is ``(kind, provider path, provider symbol)``; providers
    are non-empty only for named ImportFrom re-exports. Both branches of
    module-scope ``if``/``try`` statements contribute alternatives, so the
    comparison is branch-order independent.
    """
    package = list(pathlib.PurePosixPath(path).parent.parts)
    surface: dict[str, set[tuple[str, str, str]]] = {}
    wildcard = False
    def add(name: str, kind: str, provider: str = "", symbol: str = "") -> None:
        surface.setdefault(name, set()).add((kind, provider, symbol))
    def visit(body: list[ast.stmt]) -> None:
        nonlocal wildcard
        for node in body:
            if isinstance(node, ast.ClassDef):
                add(node.name, "class")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                add(node.name, "function")
            elif isinstance(node, ast.Assign) or (isinstance(node, ast.AnnAssign) and node.value is not None):
                for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets):
                    for name in _assignment_names(target): add(name, "assignment")
            elif isinstance(node, ast.Import):
                for alias in node.names: add((alias.asname or alias.name).split(".")[0], "import")
            elif isinstance(node, ast.ImportFrom):
                if any(alias.name == "*" for alias in node.names):
                    wildcard = True
                origin = ""
                if node.module:
                    parts = (package[:max(0, len(package) - node.level + 1)] if node.level else []) + node.module.split(".")
                    origin = "/".join(parts) + ".py"
                for alias in node.names:
                    if alias.name != "*": add(alias.asname or alias.name, "reexport", origin, alias.name)
            elif isinstance(node, ast.If):
                for branch in (node.body, node.orelse): visit(branch)
            elif isinstance(node, ast.Try):
                for branch in (node.body, node.orelse, node.finalbody, *(handler.body for handler in node.handlers)):
                    visit(branch)
    visit(ast.parse(text).body)
    return {name: frozenset(alternatives) for name, alternatives in surface.items()}, wildcard
def _python_tracked_names(surface: dict[str, frozenset[tuple[str, str, str]]], first_party: frozenset[str]) -> list[str]:
    """Owned baseline identities: local bindings (dunders included) and named
    re-exports whose provider resolves to a first-party repo path."""
    return sorted(name for name, alternatives in surface.items()
                  if any(kind in _PY_LOCAL_KINDS or (kind == "reexport" and provider in first_party)
                         for kind, provider, _symbol in alternatives))
def _drift_sources(repo: pathlib.Path, ref: str, path: str, errors: list[str]) -> tuple[str, str] | None:
    """Baseline and candidate text of a surviving file; fail closed on read errors."""
    try:
        base_text = _source_text(repo, ref, path)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        errors.append(f"migration completeness unverifiable for {path}: baseline source unreadable")
        return None
    try:
        current_text = (repo / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        errors.append(f"migration completeness unverifiable for {path}: candidate source unreadable")
        return None
    return base_text, current_text
def _python_symbol_drift(repo: pathlib.Path, ref: str, paths: Iterable[str], first_party: frozenset[str]) -> tuple[dict[str, tuple[str, bool]], set[str], list[str]]:
    """Compare tracked Python surfaces: (provider transitions, moved/removed, errors).

    Identity kinds are strict: only an unchanged binding-alternative set, or a
    same-kind single local binding, preserves a baseline identity without a
    row. A named ImportFrom is a provider transition pinned to the exact
    resolved provider path and imported symbol; a tracked re-export inlined
    into a local binding transitions to the local owner identity without an
    extraction facade. Masking assignments/imports, kind changes and changed
    conditional alternatives demand a row; wildcard imports make the file
    unverifiable.
    """
    transitions: dict[str, tuple[str, bool]] = {}
    vanished: set[str] = set()
    errors: list[str] = []
    for path in sorted(paths):
        if not path.endswith(".py") or not (repo / path).is_file(): continue
        sources = _drift_sources(repo, ref, path, errors)
        if sources is None: continue
        try:
            base_surface, base_wildcard = _python_module_surface(sources[0], path)
        except SyntaxError:
            errors.append(f"migration completeness unverifiable for {path}: baseline python source does not parse")
            continue
        try:
            current_surface, current_wildcard = _python_module_surface(sources[1], path)
        except SyntaxError:
            errors.append(f"migration completeness unverifiable for {path}: candidate python source does not parse")
            continue
        if base_wildcard or current_wildcard:
            errors.append(f"migration completeness unverifiable for {path}: wildcard import obscures the module surface")
            continue
        for name in _python_tracked_names(base_surface, first_party):
            identity = f"{path}::{name}"
            base_alternatives = base_surface[name]
            current_alternatives = current_surface.get(name, frozenset())
            if current_alternatives == base_alternatives: continue
            if len(base_alternatives) == 1 == len(current_alternatives):
                base_kind = next(iter(base_alternatives))[0]
                kind, provider, symbol = next(iter(current_alternatives))
                if kind == "reexport" and provider:
                    transitions[identity] = (f"{provider}::{symbol}", provider != path)
                elif base_kind == "reexport" and kind in _PY_LOCAL_KINDS:
                    transitions[identity] = (f"{path}::{name}", False)  # inlined into a local owner; no extraction facade
                else:
                    vanished.add(identity)  # strict identity kinds: masking/kind changes demand a row
            else:
                vanished.add(identity)  # changed conditional binding alternatives demand a row
    return transitions, vanished, errors
@functools.lru_cache(maxsize=1)
def _js_parser() -> Any:
    try:
        from tree_sitter_language_pack import get_parser
        return get_parser("javascript")
    except Exception:
        return None
_JS_FUNCTION_DECLARATION_TYPES = frozenset({"function_declaration", "generator_function_declaration"})
_JS_FUNCTION_VALUE_TYPES = frozenset({"arrow_function", "function_expression", "function", "generator_function"})
_JS_CLASS_VALUE_TYPES = frozenset({"class", "class_expression"})
def _js_text(node: Any) -> str:
    return node.text.decode("utf-8", "replace") if node is not None and node.text else ""
def _js_pattern_names(node: Any) -> set[str]:
    if node.type in {"identifier", "shorthand_property_identifier_pattern"}:
        return {_js_text(node)}
    names: set[str] = set()
    for child in node.named_children:
        if node.type == "assignment_pattern" and child is node.child_by_field_name("right"):
            continue  # default-value expressions are reads, not bindings
        if node.type == "pair_pattern" and child is node.child_by_field_name("key"):
            continue
        names |= _js_pattern_names(child)
    return names
def _js_declaration_bindings(node: Any) -> dict[str, str]:
    """Map declared top-level names to strict kinds (class/function/variable)."""
    if node.type in _JS_FUNCTION_DECLARATION_TYPES or node.type == "class_declaration":
        name = _js_text(node.child_by_field_name("name"))
        kind = "class" if node.type == "class_declaration" else "function"
        return {name: kind} if name else {}
    bindings: dict[str, str] = {}
    if node.type in {"lexical_declaration", "variable_declaration"}:
        for declarator in node.named_children:
            if declarator.type != "variable_declarator": continue
            target = declarator.child_by_field_name("name")
            value = declarator.child_by_field_name("value")
            if target is None: continue
            if target.type == "identifier":
                kind = "variable"
                if value is not None and value.type in _JS_FUNCTION_VALUE_TYPES: kind = "function"
                elif value is not None and value.type in _JS_CLASS_VALUE_TYPES: kind = "class"
                bindings[_js_text(target)] = kind
            else:
                bindings.update(dict.fromkeys(_js_pattern_names(target), "variable"))
    return {name: kind for name, kind in bindings.items() if name}
def _js_source_literal(node: Any) -> str:
    source = node.child_by_field_name("source")
    if source is None: return ""
    return next((_js_text(child) for child in source.named_children if child.type == "string_fragment"), "")
def _js_import_bindings(clause: Any) -> dict[str, str]:
    """Map local import bindings to source symbols ('default', named, '' = namespace)."""
    names: dict[str, str] = {}
    for child in clause.named_children:
        if child.type == "identifier":
            names[_js_text(child)] = "default"
        elif child.type == "namespace_import":
            for grand in child.named_children:
                if grand.type == "identifier": names[_js_text(grand)] = ""
        elif child.type == "named_imports":
            for spec in child.named_children:
                if spec.type == "import_specifier":
                    source_name = _js_text(spec.child_by_field_name("name"))
                    local = _js_text(spec.child_by_field_name("alias")) or source_name
                    names[local] = source_name
    return {name: symbol for name, symbol in names.items() if name}
def _js_module_surface(text: str) -> tuple[tuple[dict[str, tuple[str, str, str]], dict[str, tuple[str, str]], bool] | None, str]:
    """Return ((bindings, export surface, has bare export *), '') or (None, reason).

    Bindings map top-level names to (kind, specifier, source symbol) with kind
    in class/function/variable/import. The export surface maps every exported
    name (aliased, ``default`` and ``export * as ns`` included) to its exact
    provider identity (specifier, source symbol): locally provided exports
    carry ('' , actual local binding symbol) and namespace re-exports carry an
    empty source symbol. Export resolution runs after the full pass, so
    hoisted declarations and later imports resolve exactly. A bare
    ``export *`` is lexically unenumerable and flags the surface instead.
    """
    parser = _js_parser()
    if parser is None:
        return None, "javascript structural parser unavailable"
    tree = parser.parse(text.encode("utf-8", "replace"))
    if tree.root_node.has_error:
        return None, "javascript source does not parse"
    bindings: dict[str, tuple[str, str, str]] = {}
    imported: dict[str, tuple[str, str]] = {}
    records: list[tuple[str, Any, Any]] = []
    wildcard = False
    for node in tree.root_node.named_children:
        if node.type == "import_statement":
            clause = next((child for child in node.named_children if child.type == "import_clause"), None)
            if clause is not None:
                source = _js_source_literal(node)
                for name, symbol in _js_import_bindings(clause).items():
                    imported[name] = (source, symbol)
                    bindings[name] = ("import", source, symbol)
            continue
        if node.type != "export_statement":
            for name, kind in _js_declaration_bindings(node).items(): bindings[name] = (kind, "", "")
            continue
        declaration = node.child_by_field_name("declaration")
        is_default = any(child.type == "default" for child in node.children)
        if declaration is not None:
            declared = _js_declaration_bindings(declaration)
            for name, kind in declared.items(): bindings[name] = (kind, "", "")
            records.append(("default_declaration" if is_default else "declaration", sorted(declared), None))
            continue
        source = _js_source_literal(node)
        clause = next((child for child in node.named_children if child.type == "export_clause"), None)
        namespace = next((child for child in node.named_children if child.type == "namespace_export"), None)
        if clause is not None:
            specs = []
            for spec in clause.named_children:
                if spec.type != "export_specifier": continue
                local_name = _js_text(spec.child_by_field_name("name"))
                exported_name = _js_text(spec.child_by_field_name("alias")) or local_name
                specs.append((local_name, exported_name))
            records.append(("clause", source, specs))
        elif namespace is not None:
            name = next((_js_text(grand) for grand in namespace.named_children if grand.type == "identifier"), "")
            if name: records.append(("namespace", source, name))
        elif is_default:
            value = node.child_by_field_name("value")
            if value is not None and value.type == "identifier":
                records.append(("default_value", _js_text(value), None))
            else:
                kind = "variable"
                if value is not None and value.type in _JS_FUNCTION_VALUE_TYPES: kind = "function"
                elif value is not None and value.type in _JS_CLASS_VALUE_TYPES: kind = "class"
                records.append(("default_value", "", kind))
        elif source:
            wildcard = True  # bare `export * from ...` is lexically unenumerable
    exports: dict[str, tuple[str, str]] = {}
    for record, first, second in records:
        if record == "default_declaration":
            exports["default"] = ("", first[0] if first else "default")
        elif record == "declaration":
            exports.update({name: ("", name) for name in first})
        elif record == "clause":
            for local_name, exported_name in second:
                if first: exports[exported_name] = (first, local_name)
                else: exports[exported_name] = imported.get(local_name, ("", local_name))
        elif record == "namespace":
            exports[second] = (first, "")
        elif record == "default_value":
            if first and first in imported: exports["default"] = imported[first]
            elif first and first in bindings: exports["default"] = ("", first)
            else:
                exports["default"] = ("", "default")
                # `default` is a reserved word, so this pseudo-binding can never
                # collide with a real top-level name; it carries the strict kind
                # of an anonymous default-exported value.
                if second: bindings.setdefault("default", (second, "", ""))
    return (bindings, exports, wildcard), ""
def _js_resolved_origin(path: str, spec: str) -> str:
    """Resolve a relative specifier to a repo path ('' when not repo-resolvable)."""
    if not spec.startswith(("./", "../")): return ""
    resolved = posixpath.normpath(posixpath.join(str(pathlib.PurePosixPath(path).parent), spec))
    return "" if resolved.split("/", 1)[0] == ".." else resolved
def _js_provider_ref(path: str, name: str, source: str, symbol: str) -> str:
    """Canonical exact owner spelling for a provider identity.

    Local providers spell the actual local binding symbol; repo-resolvable
    specifiers spell ``resolved path::source symbol`` (path only for
    namespace re-exports); bare specifiers spell the one canonical external
    form ``external:<specifier>::<source symbol>``.
    """
    if not source:
        return f"{path}::{symbol or name}"
    origin = _js_resolved_origin(path, source)
    base = origin if origin else f"external:{source}"
    return f"{base}::{symbol}" if symbol else base
def _js_tracked_names(surface: tuple[dict[str, tuple[str, str, str]], dict[str, tuple[str, str]], bool]) -> list[str]:
    """Owned baseline identities: exported names plus top-level declarations."""
    bindings, exports, _wildcard = surface
    return sorted(set(exports) | {name for name, (kind, _, _) in bindings.items() if kind != "import"})
def _js_symbol_drift(repo: pathlib.Path, ref: str, paths: Iterable[str]) -> tuple[dict[str, tuple[str, bool]], set[str], list[str]]:
    """Compare tracked JavaScript surfaces: (provider transitions, moved/removed, errors).

    Every baseline exported identity must keep its exact provider identity
    (specifier and source symbol): extraction to a re-export, an owner path or
    source-symbol change, a re-export inlined back into a local binding, and a
    move to a bare/external provider all transition to the exact expected
    owner spelling; only non-local providers keep the extraction-facade
    requirement. Exported identities backed by local bindings (anonymous
    defaults included) and baseline private declarations must keep their
    strict kind; replacement by an import/re-export transitions, removal or a
    kind change demands a row. Wildcard (bare ``export *``) surfaces are
    unverifiable and fail closed.
    """
    transitions: dict[str, tuple[str, bool]] = {}
    vanished: set[str] = set()
    errors: list[str] = []
    for path in sorted(paths):
        if not (repo / path).is_file(): continue
        sources = _drift_sources(repo, ref, path, errors)
        if sources is None: continue
        base, base_reason = _js_module_surface(sources[0])
        if base is None:
            errors.append(f"migration completeness unverifiable for {path}: baseline {base_reason}")
            continue
        current, current_reason = _js_module_surface(sources[1])
        if current is None:
            errors.append(f"migration completeness unverifiable for {path}: candidate {current_reason}")
            continue
        if base[2] or current[2]:
            errors.append(f"migration completeness unverifiable for {path}: wildcard export obscures the module surface")
            continue
        base_bindings, base_exports, _ = base
        current_bindings, current_exports, _ = current
        for name in sorted(base_exports):
            identity = f"{path}::{name}"
            if name not in current_exports:
                vanished.add(identity)
                continue
            base_ref = _js_provider_ref(path, name, *base_exports[name])
            current_ref = _js_provider_ref(path, name, *current_exports[name])
            if current_ref != base_ref:
                # An extraction to a non-local provider must keep the public facade;
                # a local provider change (inlining/rename) is a row without one.
                transitions[identity] = (current_ref, bool(current_exports[name][0]))
            elif not base_exports[name][0]:
                base_kind = base_bindings.get(base_exports[name][1], ("", "", ""))[0]
                current_kind = current_bindings.get(current_exports[name][1], ("", "", ""))[0]
                if current_kind != base_kind:
                    vanished.add(identity)  # exported local identities keep strict kinds
        for name in sorted(set(base_bindings) - set(base_exports)):
            kind = base_bindings[name][0]
            if kind == "import": continue  # import bindings are references, not owned symbols
            identity = f"{path}::{name}"
            entry = current_bindings.get(name)
            if entry is not None and entry[0] == kind: continue  # strict same-kind local binding
            if entry is not None and entry[0] == "import":
                transitions[identity] = (_js_provider_ref(path, name, entry[1], entry[2]), False)
            elif entry is None and name in current_exports and current_exports[name][0]:
                transitions[identity] = (_js_provider_ref(path, name, *current_exports[name]), False)
            else:
                vanished.add(identity)  # removed or strict-kind-incompatible local change
    return transitions, vanished, errors
def _gated_js_paths(repo: pathlib.Path, paths: Iterable[str]) -> tuple[set[str], list[str]]:
    """Filter to first-party web JavaScript via the production SSOT predicate.

    ``ouroboros.review.is_gated_js_module`` is executed in an isolated
    subprocess with four temporary Ouroboros roots (the `_census` pattern), so
    the checker never imports the runtime package in-process. The predicate
    authority is the checkout this script belongs to.
    """
    js_paths = sorted(path for path in paths if path.endswith(".js"))
    if not js_paths: return set(), []
    script_repo = pathlib.Path(__file__).resolve().parents[1]
    code = ("import json,sys; from ouroboros.review import is_gated_js_module; "
            "print(json.dumps([p for p in json.loads(sys.argv[1]) if is_gated_js_module(p)]))")
    with tempfile.TemporaryDirectory(prefix="ouro-v7-migration-") as temp:
        data = pathlib.Path(temp) / "data"
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(script_repo), "PYTHONDONTWRITEBYTECODE": "1",
               "OUROBOROS_APP_ROOT": temp, "OUROBOROS_REPO_DIR": str(script_repo),
               "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_SETTINGS_PATH": str(data / "settings.json")}
        try:
            output = subprocess.run([sys.executable, "-c", code, json.dumps(js_paths)], cwd=script_repo,
                                    env=env, check=True, capture_output=True, text=True).stdout
        except (OSError, subprocess.CalledProcessError):
            return set(), ["first-party JavaScript predicate unavailable; cannot verify migration completeness: " + ", ".join(js_paths)]
    return set(json.loads(output)), []
def _baseline_symbol_surface(repo: pathlib.Path, path: str, js_supported: bool, first_party: frozenset[str]) -> tuple[list[str] | None, list[str]]:
    """Supported baseline identities of a deleted/renamed/type-changed file."""
    if not (path.endswith(".py") or (path.endswith(".js") and js_supported)):
        return None, []
    try:
        text = _source_text(repo, MERGE_BASE_SHA, path)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return None, [f"migration completeness unverifiable for {path}: baseline source unreadable"]
    if path.endswith(".py"):
        try:
            surface, wildcard = _python_module_surface(text, path)
        except SyntaxError:
            return None, [f"migration completeness unverifiable for {path}: baseline python source does not parse"]
        if wildcard:
            return None, [f"migration completeness unverifiable for {path}: wildcard import obscures the module surface"]
        return _python_tracked_names(surface, first_party), []
    surface_js, reason = _js_module_surface(text)
    if surface_js is None:
        return None, [f"migration completeness unverifiable for {path}: baseline {reason}"]
    if surface_js[2]:
        return None, [f"migration completeness unverifiable for {path}: wildcard export obscures the module surface"]
    return _js_tracked_names(surface_js), []
def _symbol_exists(repo: pathlib.Path, path: str, symbol: str, ref: str = "") -> bool:
    if not symbol: return True
    try:
        text = _source_text(repo, ref, path) if ref else (repo / path).read_text(encoding="utf-8")
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError):
        return False
    if path.endswith(".py"):
        try:
            surface, _wildcard = _python_module_surface(text, path)
        except SyntaxError:
            return False
        found: list[str] = []
        def walk(body: list[ast.stmt], scope: tuple[str, ...] = (), in_class: bool = False) -> None:
            for node in body:
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualname = ".".join((*scope, node.name))
                    if qualname == symbol: found.append(qualname)
                    walk(node.body, (*scope, node.name), isinstance(node, ast.ClassDef))
                elif in_class and (isinstance(node, ast.Assign) or (isinstance(node, ast.AnnAssign) and node.value is not None)):
                    # Class-body attribute assignments are declarations too: a ledger row
                    # may relocate `Owner._ATTR` between class bodies (the llm mixin split
                    # moved 15 of them). Duplicate assignments fail closed via len(found).
                    for target in ([node.target] if isinstance(node, ast.AnnAssign) else node.targets):
                        for name in _assignment_names(target):
                            if ".".join((*scope, name)) == symbol: found.append(symbol)
        walk(ast.parse(text).body)
        if not found and any(kind in _PY_LOCAL_KINDS or kind == "reexport"
                             for kind, _provider, _symbol in surface.get(symbol, frozenset())):
            found.append(symbol)  # module-scope assignment/re-export (ordinary imports stay references)
        return len(found) == 1
    if path.endswith(".js"):
        if "." in symbol: return _js_nested_declaration_exists(text, symbol)
        surface_js, _reason = _js_module_surface(text)
        if surface_js is None: return False  # fail closed: no structural parser, no resolution
        bindings, exports, _wildcard = surface_js
        return symbol in exports or (symbol in bindings and bindings[symbol][0] != "import")
    return False  # Qualified references require a structural parser for their language.
_JS_DECLARATION_STATEMENT_TYPES = _JS_FUNCTION_DECLARATION_TYPES | {"class_declaration", "lexical_declaration", "variable_declaration"}
_JS_SCOPE_NODE_TYPES = _JS_FUNCTION_DECLARATION_TYPES | _JS_FUNCTION_VALUE_TYPES | _JS_CLASS_VALUE_TYPES | {"class_declaration", "class_body", "method_definition"}
def _js_scope_root(node: Any, name: str) -> Any:
    """The node whose children form ``name``'s own scope: the declaration itself for
    ``function``/``class`` declarations, the function/class value for a lexical binding
    (``const f = () => {...}``), or None for a plain value (no scope below it)."""
    if node.type in _JS_FUNCTION_DECLARATION_TYPES or node.type == "class_declaration": return node
    for declarator in node.named_children:
        if declarator.type != "variable_declarator": continue
        target, value = declarator.child_by_field_name("name"), declarator.child_by_field_name("value")
        if target is not None and target.type == "identifier" and _js_text(target) == name:
            return value if value is not None and value.type in (_JS_FUNCTION_VALUE_TYPES | _JS_CLASS_VALUE_TYPES) else None
    return None
def _js_nested_declaration_exists(text: str, qualname: str) -> bool:
    """True when ``_js_declaration_node`` resolves the dotted identity."""
    return _js_declaration_node(text, qualname) is not None
def _js_declaration_node(text: str, qualname: str) -> Any:
    """Resolve a JavaScript identity to its declaration node, or None.

    A bare name resolves to its top-level declaration statement (the
    ``declaration`` child of an ``export`` statement, so an exported and a
    private declaration compare by the same text). A dotted identity
    (``outer.inner[.deeper]``) resolves lexically:

    The JavaScript twin of the Python qualname walk: ``outer`` must be exactly
    one top-level function/class binding, and every further segment exactly one
    function/class/lexical declaration whose NEAREST enclosing function/class
    scope is the previous match (statement blocks such as ``if``/``try`` bodies
    are searched, nested function and class scopes are not — ``f.inner`` does
    not resolve a helper declared inside ``f.g``). This keeps a closure helper
    that moved into an instance factory ledger-addressable without being
    exported. Ambiguity (a name declared twice in the same scope) and parse
    failure resolve to False. Resolution proves that the identity is declared,
    not that an implementation moved: a destructuring re-bind of the same name
    (``const { helper } = makeHelpers(...)``) is a lexical declaration too, so
    move proofs remain the reviewer's byte comparison, not this resolver.
    """
    parser = _js_parser()
    if parser is None: return None
    tree = parser.parse(text.encode("utf-8", "replace"))
    if tree.root_node.has_error: return None
    head, *rest = qualname.split(".")
    if not head or not all(rest): return None
    def declared_kind(node: Any, name: str) -> str:
        return _js_declaration_bindings(node).get(name, "") if node.type in _JS_DECLARATION_STATEMENT_TYPES else ""
    matches, matched_name = [], head
    for statement in tree.root_node.named_children:
        candidate = statement.child_by_field_name("declaration") if statement.type == "export_statement" else statement
        if candidate is None: continue
        if (declared_kind(candidate, head) in {"function", "class"}) if rest else bool(declared_kind(candidate, head)):
            matches.append(candidate)
    for name in rest:
        if len(matches) != 1: return None
        root = _js_scope_root(matches[0], matched_name)
        found, stack = [], (list(root.named_children) if root is not None else [])
        while stack:
            node = stack.pop()
            if declared_kind(node, name): found.append(node)
            if node.type not in _JS_SCOPE_NODE_TYPES: stack.extend(node.named_children)
        matches, matched_name = found, name
    return matches[0] if len(matches) == 1 else None
def _facade_exists(repo: pathlib.Path, path: str, symbol: str) -> bool:
    """Resolve a facade/public-contract cell; JavaScript facades must be exported."""
    if not path.endswith(".js"):
        return _symbol_exists(repo, path, symbol)
    if not symbol: return True
    try:
        text = (repo / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return False
    surface, _reason = _js_module_surface(text)
    return surface is not None and symbol in surface[1]
def _facade_reexport_ref(repo: pathlib.Path, path: str, symbol: str) -> tuple[str, str]:
    """Exact provider identity currently re-exported by a facade binding.

    Returns ``(canonical owner ref, "")`` when the facade is one exact named
    re-export: a single Python ImportFrom alternative or a JavaScript
    re-export/import+export/namespace export. Otherwise returns
    ``("", deterministic reason)``: a copied/local implementation or an
    ordinary import is never an extraction facade.
    """
    try:
        text = (repo / path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return "", "facade source unreadable"
    if path.endswith(".py"):
        try:
            surface, wildcard = _python_module_surface(text, path)
        except SyntaxError:
            return "", "facade python source does not parse"
        alternatives = surface.get(symbol, frozenset())
        if not alternatives and wildcard:
            return "", "facade obscured by a wildcard import"
        if len(alternatives) != 1:
            return "", "facade binding is missing or ambiguous"
        kind, provider, source_symbol = next(iter(alternatives))
        if kind != "reexport":
            return "", "facade binding is a local implementation or ordinary import, not a re-export"
        if not provider:
            return "", "facade re-export provider is not repo-resolvable"
        return f"{provider}::{source_symbol}", ""
    surface_js, reason = _js_module_surface(text)
    if surface_js is None:
        return "", f"facade {reason}"
    source, source_symbol = surface_js[1].get(symbol, ("", ""))
    if not source:
        return "", "facade binding is a local implementation, not a re-export"
    return _js_provider_ref(path, symbol, source, source_symbol), ""
def validate_migration(repo: pathlib.Path) -> list[str]:
    errors: list[str] = []
    path = repo / MIGRATION_PATH
    try:
        rows = _parse_migration(path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return [str(exc)]
    seen: set[str] = set()
    baseline_paths = frozenset(_tracked_paths(repo, MERGE_BASE_SHA))
    for index, row in enumerate(rows, start=1):
        prefix = f"row {index}"
        old = row[MIGRATION_HEADERS[0]]
        owner = row[MIGRATION_HEADERS[1]]
        facade = row[MIGRATION_HEADERS[2]]
        delta_cell = row[MIGRATION_HEADERS[3]]
        test_ref = row[MIGRATION_HEADERS[4]]
        status_cell = row[MIGRATION_HEADERS[5]]
        try: delta = _migration_json(delta_cell, ("id", "note"))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}: invalid semantic delta JSON: {exc}")
            delta = {"id": "", "note": ""}
        try: upstream = _migration_json(status_cell, ("status", "note"))
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{prefix}: invalid upstream status JSON: {exc}")
            upstream = {"status": "", "note": ""}
        if not old or old in seen:
            errors.append(f"{prefix}: old identity is empty or duplicated: {old!r}")
        seen.add(old)
        old_path, old_symbol = _parse_ref(old)
        if old_path not in baseline_paths and not (repo / old_path).exists():
            errors.append(f"{prefix}: old path does not exist at baseline/current: {old_path}")
        elif old_symbol and not _symbol_exists(repo, old_path, old_symbol, MERGE_BASE_SHA):
            errors.append(f"{prefix}: old symbol does not resolve at baseline: {old}")
        retired = owner.startswith("retired:")
        external = owner.startswith("external:")
        owner_path, owner_symbol = _parse_ref(owner)
        pending_owner = (not retired and not external and upstream["status"] == "pending"
                         and owner == owner_path and owner_path in APPROVED_PENDING_OWNERS)
        if (not owner) or ((retired or external) and len(owner.split(":", 1)[1].strip()) == 0):
            errors.append(f"{prefix}: exactly one owner path, external provider or retirement reason is required")
        if retired:
            if upstream["status"] != "retired":
                errors.append(f"{prefix}: retired owner requires retired upstream status")
        elif not external:
            if not (repo / owner_path).exists() and not pending_owner:
                errors.append(f"{prefix}: missing owner is not an approved spec 4.4 pending destination: {owner_path}")
            elif not pending_owner and not _symbol_exists(repo, owner_path, owner_symbol):
                errors.append(f"{prefix}: owner reference does not resolve: {owner}")
        if facade != "-":
            if facade != old:
                errors.append(f"{prefix}: facade must be the exact old identity: {facade}")
            else:
                facade_path, facade_symbol = _parse_ref(facade)
                if not (repo / facade_path).exists() or not _facade_exists(repo, facade_path, facade_symbol):
                    errors.append(f"{prefix}: facade reference does not resolve: {facade}")
                elif not pending_owner:
                    actual_ref, reason = _facade_reexport_ref(repo, facade_path, facade_symbol)
                    if reason:
                        errors.append(f"{prefix}: {reason}: {facade}")
                    elif actual_ref != owner:
                        errors.append(f"{prefix}: facade re-export does not match the declared owner: {facade} -> {actual_ref}")
                    elif (not external and owner_symbol and owner_path.endswith(".js")
                          and not _facade_exists(repo, owner_path, owner_symbol)):
                        # A repo-local JS owner backing a facade must publicly export the
                        # source symbol, or the facade's ES re-export would fail to link.
                        errors.append(f"{prefix}: facade owner does not export the source symbol: {owner}")
            if test_ref == "-":
                errors.append(f"{prefix}: facade requires an identity/signature characterization test")
        if delta["id"] not in APPROVED_SEMANTIC_DELTAS:
            errors.append(f"{prefix}: invalid semantic delta id: {delta['id']}")
        if test_ref != "-":
            test_path, test_symbol = _parse_ref(test_ref)
            if not (repo / test_path).is_file() or not _symbol_exists(repo, test_path, test_symbol):
                errors.append(f"{prefix}: characterization test does not resolve: {test_ref}")
        if upstream["status"] not in UPSTREAM_STATUSES:
            errors.append(f"{prefix}: invalid upstream-transfer status: {upstream['status']}")
        if upstream["status"] == "pending" and not upstream["note"].strip():
            errors.append(f"{prefix}: pending upstream status requires a note")
        for header, cell in row.items():
            if "\n" in cell or "\r" in cell or "|" in cell:
                errors.append(f"{prefix}: {header} is not compact")
    diffs = (
        str(_git(repo, "diff", "--name-status", "-M", f"{MERGE_BASE_SHA}..HEAD", "--")),
        str(_git(repo, "diff", "--name-status", "-M", MERGE_BASE_SHA, "--")),
    )
    candidates: set[str] = set(); modified: set[str] = set()
    for line in "\n".join(diffs).splitlines():
        fields = line.split("\t")
        status = fields[0]
        if status.startswith("R") and len(fields) >= 3:
            candidates.add(fields[1])
        elif status in {"D", "T"} and len(fields) >= 2:
            candidates.add(fields[1])  # deletions and type changes are both losses of the tracked source
        elif status == "M" and len(fields) >= 2: modified.add(fields[1])
    gated_js, predicate_errors = _gated_js_paths(repo, sorted(modified | candidates))
    errors.extend(predicate_errors)
    for old_path in sorted(candidates):
        if old_path in seen: continue  # a path-level row explicitly owns the whole file
        names, surface_errors = _baseline_symbol_surface(repo, old_path, old_path in gated_js, baseline_paths)
        errors.extend(surface_errors)
        if surface_errors: continue
        if names:
            errors.extend(f"tracked migration missing for moved/removed symbol: {old_path}::{name}"
                          for name in names if f"{old_path}::{name}" not in seen)
        elif not any(identity.startswith(old_path + "::") for identity in seen):
            errors.append(f"tracked migration missing for moved/removed path: {old_path}")
    rows_by_old = {row[MIGRATION_HEADERS[0]]: row for row in rows}
    python_transitions, python_vanished, python_errors = _python_symbol_drift(repo, MERGE_BASE_SHA, modified, baseline_paths)
    errors.extend(python_errors)
    js_transitions, js_vanished, js_errors = _js_symbol_drift(repo, MERGE_BASE_SHA, sorted(gated_js & modified))
    errors.extend(js_errors)
    for identity, (owner_ref, facade_required) in sorted({**python_transitions, **js_transitions}.items()):
        row = rows_by_old.get(identity)
        if row is None:
            errors.append(f"tracked migration missing for extracted facade: {identity} -> {owner_ref}")
        elif row[MIGRATION_HEADERS[1]] != owner_ref:
            errors.append(f"tracked migration owner mismatch for extracted facade: {identity} -> {owner_ref}")
        elif facade_required and row[MIGRATION_HEADERS[2]] == "-":
            errors.append(f"tracked migration facade missing for extracted facade: {identity}")
    for identity in sorted(python_vanished | js_vanished):
        if identity not in rows_by_old:
            errors.append(f"tracked migration missing for moved/removed symbol: {identity}")
    return errors
