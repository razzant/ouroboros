#!/usr/bin/env python3
"""Generate and validate the immutable Ouroboros v7 prologue evidence."""
from __future__ import annotations
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import inspect
import io
import json
import os
import pathlib
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable
def _load_migration_module() -> Any:
    """Execute the exact resolved sibling ``v7_migration.py`` directly.

    Every load builds a fresh module object from the sibling path and never
    reads from or writes to ``sys.modules`` (the campaign forbids sys.modules
    proxies/caches), so two evidence checkouts loaded into one process always
    bind their own checkout's contract module.
    """
    target = pathlib.Path(__file__).resolve().with_name("v7_migration.py")
    spec = importlib.util.spec_from_file_location("v7_migration", target)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
_migration = _load_migration_module()
BASELINE_SHA = _migration.BASELINE_SHA
OBSERVED_HEAD_SHA = "d30c560457d6de8cf36fb6339880d228fc740729"
FIXTURE_PATH = pathlib.PurePosixPath("tests/fixtures/v7_prologue_baseline.json")
MIGRATION_HEADERS = _migration.MIGRATION_HEADERS
APPROVED_PENDING_OWNERS = _migration.APPROVED_PENDING_OWNERS
HARD_STREAM_PATHS = {
    "T": "ouroboros/tools/registry.py ouroboros/tool_access.py ouroboros/tools/git.py ouroboros/tools/core.py ouroboros/tools/shell.py ouroboros/headless.py tests/test_tool_capabilities.py tests/test_headless_cli.py tests/test_git_review_pipeline.py".split(),
    "S": "ouroboros/config.py ouroboros/gateway/settings.py server.py supervisor/events.py supervisor/workers.py supervisor/queue.py supervisor/task_lifecycle.py ouroboros/extension_loader.py ouroboros/tools/control.py ouroboros/tools/delegate.py ouroboros/delegate_custody.py ouroboros/tools/subagent_integration.py tests/test_task_status_flow.py tests/test_cancel_intents_phase_a.py tests/test_evolution_state_integrity_v3.py tests/test_runtime_mode_elevation.py tests/test_runtime_mode_core.py tests/test_promote_chat_flow.py tests/test_workspace_executor.py tests/test_extension_loader.py tests/test_extensions_api.py tests/test_delivery_forced_finalization.py tests/test_delegated_subagent_transport.py tests/test_delegated_run_isolation.py tests/test_claudexor_owned_daemon.py tests/test_skill_exec.py tests/test_skill_loader.py tests/test_skill_review.py tests/test_context.py".split(),
    "L": "ouroboros/llm.py ouroboros/loop.py ouroboros/agent.py ouroboros/agent_task_pipeline.py ouroboros/usage_accounting.py ouroboros/tools/review.py ouroboros/tools/review_helpers.py ouroboros/tools/plan_review.py ouroboros/tools/scope_review.py ouroboros/tools/review_synthesis.py ouroboros/tools/claude_advisory_review.py ouroboros/review_state.py ouroboros/review_substrate.py ouroboros/review_evidence.py ouroboros/review_execution.py ouroboros/skill_review.py tests/test_plan_review.py tests/test_scope_review.py tests/test_review_substrate_v2.py tests/test_review_agent_session_route.py tests/test_review_prompt_caching.py tests/test_loop_misc.py tests/test_agent_task_pipeline.py tests/test_preflight_runner.py".split(),
    "W": "web/modules/chat.js web/tests/harness_accounts.test.js skills/unix_computer_use/plugin.py devtools/benchmarks/osworld/run_cu_bridge_agent.py devtools/benchmarks/osworld/run_step_agent.py supervisor/git_ops.py supervisor/update_merge.py tests/test_ui_smoke_playwright.py tests/test_devtools_benchmarks.py tests/test_osworld_cu_bridge.py tests/test_git_ops_recovery.py tests/test_model_slot_role_model.py".split(),
}
HARD_STREAM_BY_PATH = {path: stream for stream, paths in HARD_STREAM_PATHS.items() for path in paths}
def _repo_root(start: pathlib.Path | None = None) -> pathlib.Path:
    candidate = (start or pathlib.Path(__file__)).resolve()
    for parent in (candidate, *candidate.parents):
        if (parent / ".git").exists() and (parent / "BIBLE.md").is_file():
            return parent
    raise RuntimeError("v7 evidence script must run inside an Ouroboros checkout")
_git = _migration._git
_tracked_paths = _migration._tracked_paths
def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))
def _source_bytes(repo: pathlib.Path, ref: str, path: str) -> bytes:
    return _git(repo, "show", f"{ref}:{path}", text=False)  # type: ignore[return-value]
def _source_text(repo: pathlib.Path, ref: str, path: str) -> str:
    return _source_bytes(repo, ref, path).decode("utf-8", errors="strict")
def _band_owner_projection(path: str) -> str:
    lower = path.lower()
    if path.startswith(("web/", "devtools/", "skills/unix_computer_use/")) or any(token in lower for token in (
        "test_ui_smoke", "test_devtools", "test_osworld", "test_git_ops_recovery", "test_model_slot_role_model", "update_merge", "git_ops.py",
    )):
        return "W"
    if path == "server.py" or path.startswith("supervisor/") or any(token in lower for token in (
        "config.py", "gateway/settings.py", "extension_loader", "extensions_api", "task_status_flow", "cancel_intents", "evolution_state", "runtime_mode", "promote_chat", "workspace_executor", "delivery_forced", "delegated_", "skill_", "context.py", "tools/control.py", "tools/delegate.py", "delegate_custody", "subagent_integration",
    )):
        return "S"
    if any(token in lower for token in (
        "tools/registry.py", "tool_access.py", "tools/git.py", "tools/core.py", "tools/shell.py", "headless.py", "loop_tool_execution.py", "tool_capabilities", "test_headless", "test_git_review_pipeline", "test_tool_",
    )):
        return "T"
    return "L"
def _owner_for_path(path: str, stream: str) -> str:
    if not path.startswith(("tests/", "web/tests/")): return path
    lower = path.lower()
    owner_rules = (
        (("devtools", "osworld"), "devtools/benchmarks"),
        (("tool", "headless", "git_review"), "ouroboros/tools/registry.py"),
        (("cancel", "task_status", "delivery", "delegated", "workspace_executor"), "supervisor/task_lifecycle.py"),
        (("extension", "skill_"), "ouroboros/extension_loader.py"),
        (("plan_review", "scope_review", "review_"), "ouroboros/review_substrate.py"),
        (("loop", "acceptance", "nanny"), "ouroboros/loop.py"),
        (("llm", "provider", "model_slot"), "ouroboros/llm.py"),
        (("ui_", "chat", "projects"), "web/modules/chat.js"),
        (("git_ops", "update_merge"), "supervisor/git_ops.py"),
        (("runtime_mode",), "ouroboros/runtime_mode_policy.py"),
        (("context",), "ouroboros/context.py"),
    )
    for needles, owner in owner_rules:
        if any(needle in lower for needle in needles):
            return owner
    return {"T": "ouroboros/tools/registry.py", "S": "supervisor/queue.py", "L": "ouroboros/loop.py", "W": "web/modules/chat.js"}[stream]
def _test_for_path(path: str, stream: str) -> str:
    if path.startswith("tests/") and path.endswith(".py"): return path
    if path.startswith("web/tests/") and path.endswith(".js"): return path
    return {"T": "tests/test_tool_api_v2_public_surface.py", "S": "tests/test_task_status_flow.py", "L": "tests/test_loop_misc.py", "W": "tests/test_devtools_benchmarks.py"}[stream]
def _census(repo: pathlib.Path, ref: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ouro-v7-census-") as temp:
        checkout = pathlib.Path(temp)
        _safe_extract_tar(_git(repo, "archive", "--format=tar", ref, text=False), checkout)  # type: ignore[arg-type]
        code = ("import json,pathlib,sys; from ouroboros.review import iter_gated_modules; "
                "items=iter_gated_modules(pathlib.Path(sys.argv[1]),repo_paths=json.loads(sys.argv[2])); "
                "print(json.dumps([{'path':x.path,'lines':x.line_count,'utf8_bytes':x.utf8_bytes} for x in items]))")
        data = checkout.parent / "data"
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1",
            "OUROBOROS_APP_ROOT": str(checkout.parent), "OUROBOROS_REPO_DIR": str(checkout),
            "OUROBOROS_DATA_DIR": str(data), "OUROBOROS_SETTINGS_PATH": str(data / "settings.json")}
        output = subprocess.run([sys.executable, "-c", code, str(checkout), _canonical_json(_tracked_paths(repo, ref))],
                                cwd=repo, env=env, check=True, capture_output=True, text=True).stdout
        modules = json.loads(output)
    hard_paths = {row["path"] for row in modules if row["lines"] > 1500}
    if hard_paths != set(HARD_STREAM_BY_PATH):
        raise RuntimeError(f"normative hard stream map drifted: missing={sorted(hard_paths - set(HARD_STREAM_BY_PATH))}, extra={sorted(set(HARD_STREAM_BY_PATH) - hard_paths)}")
    over_1000: list[dict[str, Any]] = []
    for row in modules:
        if row["lines"] <= 1000:
            continue
        path = row["path"]
        debt_class = "hard" if row["lines"] > 1500 else "band"
        stream = HARD_STREAM_BY_PATH[path] if debt_class == "hard" else _band_owner_projection(path)
        byte_plan = "split_or_extract_below_200k" if row["utf8_bytes"] > 200_000 else "within_limit"
        over_1000.append({
            **row,
            "debt_class": debt_class,
            "stream": stream,
            "assignment_authority": "normative_spec_7" if debt_class == "hard" else "non_authoritative_evidence_projection",
            "production_owner": _owner_for_path(path, stream),
            "disposition": "split_or_shrink_below_1500" if debt_class == "hard" else "retain_with_growth_rationale",
            "byte_plan": byte_plan,
            "characterization_test": _test_for_path(path, stream),
        })
    return {
        "method": "exact-ref git archive through ouroboros.review.iter_gated_modules with injected tracked paths",
        "module_count": len(modules),
        "python_count": sum(row["path"].endswith(".py") for row in modules),
        "javascript_count": sum(row["path"].endswith(".js") for row in modules),
        "total_lines": sum(row["lines"] for row in modules),
        "hard_count": sum(row["lines"] > 1500 for row in modules),
        "band_count": sum(1000 < row["lines"] <= 1500 for row in modules),
        "byte_debt_count": sum(row["utf8_bytes"] > 200_000 for row in modules),
        "disposition": over_1000,
        "inventory_sha256": _sha256_json(modules),
    }
def _safe_extract_tar(payload: bytes, target: pathlib.Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        root = target.resolve()
        for member in archive.getmembers():
            destination = (target / member.name).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"unsafe git archive member: {member.name}")
        archive.extractall(target)  # noqa: S202 - validated archive from local git object
def _probe_ref(repo: pathlib.Path, ref: str) -> dict[str, Any]:
    archive_bytes = _git(repo, "archive", "--format=tar", ref, text=False)
    with tempfile.TemporaryDirectory(prefix="ouro-v7-evidence-") as temp:
        root = pathlib.Path(temp)
        checkout = root / "repo"
        data = root / "data"
        checkout.mkdir()
        data.mkdir()
        _safe_extract_tar(archive_bytes, checkout)  # type: ignore[arg-type]
        settings = data / "settings.json"
        settings.write_text('{"MCP_ENABLED":false,"MCP_SERVERS":[]}\n', encoding="utf-8")
        env = {
            "HOME": str(root / "home"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": str(checkout),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OUROBOROS_APP_ROOT": str(root),
            "OUROBOROS_REPO_DIR": str(checkout),
            "OUROBOROS_DATA_DIR": str(data),
            "OUROBOROS_SETTINGS_PATH": str(settings),
            "OUROBOROS_BG_WAKEUP_MIN": "60",
            "OUROBOROS_BG_WAKEUP_MAX": "3600",
            "GITHUB_TOKEN": "fixture-not-a-secret",
        }
        completed = subprocess.run(
            [sys.executable, str(pathlib.Path(__file__).resolve()), "_probe"],
            cwd=checkout,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=90,
        )
        return json.loads(completed.stdout)
def _symbol_signature(value: Any) -> str:
    try:
        return str(inspect.signature(value))
    except (TypeError, ValueError):
        return "<module-or-value>"
def _legacy_record(text: str) -> dict[str, Any]:
    return {
        "result_kind": "legacy_text",
        "text": text,
        "code": None,
        "typed_projection": {"state": "pending_stream_T"},
    }
def _probe_safety() -> dict[str, Any]:
    import ouroboros.safety as safety
    cases: list[dict[str, Any]] = []
    original_mode = safety.get_safety_mode
    original_llm_check = safety._run_llm_check
    original_skip = safety._emit_safety_mode_skip
    def policy_case(name: str, tool: str, args: dict[str, Any], mode: str) -> None:
        calls: list[str] = []
        audits: list[dict[str, str]] = []
        safety.get_safety_mode = lambda: mode
        safety._run_llm_check = lambda tool_name, *_a, **_k: (calls.append(tool_name) or (True, ""))
        safety._emit_safety_mode_skip = lambda _ctx, tool_name, value, policy: audits.append({
            "type": "safety_mode_skip", "tool": tool_name, "safety_mode": value, "policy": policy,
        })
        allowed, text = safety.check_safety(tool, args)
        cases.append({
            "case": name,
            "policy": safety.TOOL_POLICY[tool],
            "mode": mode,
            "allowed": allowed,
            "llm_calls": len(calls),
            "audit_events": audits,
            "legacy_result": _legacy_record(text),
        })
    try:
        for mode in ("full", "light", "off"):
            policy_case("delegate_answer_skip", "delegate_answer", {}, mode)
            policy_case("integrate_delegated_patch_check", "integrate_delegated_patch", {"run_id": "fixture"}, mode)
            policy_case("conditional_safe", "run_command", {"cmd": ["python3", "-m", "pytest", "-q"]}, mode)
            policy_case("conditional_unsafe", "run_command", {"cmd": ["curl", "https://example.invalid"]}, mode)
    finally:
        safety.get_safety_mode = original_mode
        safety._run_llm_check = original_llm_check
        safety._emit_safety_mode_skip = original_skip
    import ouroboros.llm_observability as llm_observability
    import ouroboros.model_concurrency as model_concurrency
    original_client = safety.LLMClient
    original_route = safety._resolve_safety_routing
    original_model = safety.get_light_model
    original_chat = llm_observability.chat_observed
    original_slot = model_concurrency.model_call_slot
    safety.LLMClient = lambda: object()
    safety._resolve_safety_routing = lambda: (False, False, None)
    safety.get_light_model = lambda: "fixture/light"
    model_concurrency.model_call_slot = lambda *_a, **_k: contextlib.nullcontext()
    scripted = (
        ("llm_safe", '{"status":"SAFE","reason":"ok"}', None),
        ("llm_suspicious", '{"status":"SUSPICIOUS","reason":"fixture concern"}', None),
        ("llm_dangerous", '{"status":"DANGEROUS","reason":"fixture denial"}', None),
        ("provider_failure", None, RuntimeError("fixture provider unavailable")),
    )
    try:
        for name, response, error in scripted:
            calls: list[str] = []
            def chat_observed(*_a: Any, **_k: Any) -> tuple[dict[str, str], None]:
                calls.append(name)
                if error is not None:
                    raise error
                return {"content": str(response)}, None
            llm_observability.chat_observed = chat_observed
            allowed, text = safety._run_llm_check("create_github_issue", {"title": "fixture"}, None, None)
            cases.append({
                "case": name,
                "policy": "check",
                "mode": "full",
                "allowed": allowed,
                "llm_calls": len(calls),
                "audit_events": [],
                "legacy_result": _legacy_record(text),
            })
    finally:
        safety.LLMClient = original_client
        safety._resolve_safety_routing = original_route
        safety.get_light_model = original_model
        llm_observability.chat_observed = original_chat
        model_concurrency.model_call_slot = original_slot
    return {"owner": "ouroboros/safety.py", "cases": cases}
def _probe_dispatch_cases(repo_dir: pathlib.Path, data_dir: pathlib.Path) -> list[dict[str, Any]]:
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.loop_tool_execution import _extract_result_metadata, _is_tool_execution_failure
    from ouroboros.tools.registry import ToolContext, ToolRegistry, _compose_execute_result
    import ouroboros.safety as safety
    cases: list[dict[str, Any]] = []
    original_check = safety.check_safety
    calls: list[str] = []
    safety.check_safety = lambda tool, *_a, **_k: (calls.append(tool) or (True, ""))
    try:
        registry = ToolRegistry(repo_dir=repo_dir, drive_root=data_dir)
        registry.set_context(ToolContext(
            repo_dir=repo_dir,
            drive_root=data_dir,
            task_constraint=TaskConstraint(mode="acting_subagent", surface="self_worktree", write_root="fixture"),
        ))
        text = registry.execute("integrate_delegated_patch", {"run_id": "fixture"})
        cases.append({
            "case": "acting_integrate_without_workspace",
            "allowed": False,
            "llm_calls": len(calls),
            "audit_events": [],
            "legacy_result": _legacy_record(text),
        })
        calls.clear()
        registry.set_context(ToolContext(repo_dir=repo_dir, drive_root=data_dir))
        text = registry.execute("write_file", {"path": "BIBLE.md", "content": "fixture"})
        cases.append({
            "case": "protected_bible_write",
            "allowed": False,
            "llm_calls": len(calls),
            "audit_events": [],
            "legacy_result": _legacy_record(text),
        })
        warning = "⚠️ SAFETY_WARNING: fixture suspicious action"
        composed = _compose_execute_result("⚠️ TOOL_ERROR: fixture underlying failure", "", warning)
        masked = _is_tool_execution_failure(True, composed)
        cases.append({"case": "safety_warning_masks_tool_error", "allowed": True, "llm_calls": 0,
            "audit_events": [], "surface": "pure_composer", "downstream_failure": masked,
            "downstream_metadata": _extract_result_metadata("fixture_tool", composed, masked),
            "legacy_result": _legacy_record(composed)})
    finally:
        safety.check_safety = original_check
    return cases
def _probe_extension_mcp(repo_dir: pathlib.Path, data_dir: pathlib.Path) -> list[dict[str, Any]]:
    from types import SimpleNamespace
    import ouroboros.extension_loader as extension_loader
    import ouroboros.mcp_client as mcp_client
    import ouroboros.safety as safety
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state, save_skill_grants
    from ouroboros.tools.extension_dispatch import dispatch_extension_tool
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    ctx = ToolContext(repo_dir=repo_dir, drive_root=data_dir)
    cases: list[dict[str, Any]] = []
    original_live = extension_loader.is_extension_live
    original_unload = extension_loader.unload_extension
    original_mode = safety.get_safety_mode
    original_llm_check = safety._run_llm_check
    calls: list[str] = []
    unloaded: list[str] = []
    skill_dir = data_dir / "skills" / "native" / "fixture"; skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text('---\nname: fixture\ndescription: fixture\nversion: "1"\ntype: extension\nentry: plugin.py\npermissions: ["inject_chat"]\n---\nfixture\n', encoding="utf-8")
    (skill_dir / "plugin.py").write_text("def register(api): pass\n", encoding="utf-8")
    skill = find_skill(data_dir, "fixture", repo_path=""); assert skill is not None
    save_enabled(data_dir, "fixture", True); save_review_state(data_dir, "fixture", SkillReviewState(status="pass", content_hash=skill.content_hash))
    skill = find_skill(data_dir, "fixture", repo_path=""); assert skill is not None
    ext_name = extension_loader.extension_surface_name("fixture", "echo")
    with extension_loader._lock: extension_loader._tools[ext_name] = {"name": ext_name, "skill": "fixture", "handler": lambda: "ok", "description": "fixture", "schema": {"type": "object", "properties": {}}}
    missing = extension_loader.runtime_state_for_loaded_skill(skill, data_dir, skills=[skill])
    registry = ToolRegistry(repo_dir=repo_dir, drive_root=data_dir); registry.set_context(ctx)
    missing_visible = ext_name in {schema["function"]["name"] for schema in registry.schemas(core_only=False)}
    save_skill_grants(data_dir, "fixture", [], content_hash=skill.content_hash, requested_keys=[], granted_permissions=["inject_chat"], requested_permissions=["inject_chat"])
    with extension_loader._lock: extension_loader._extensions["fixture"] = extension_loader._ExtensionRegistrations(content_hash=skill.content_hash, skill_dir=str(skill_dir.resolve()))
    granted = extension_loader.runtime_state_for_loaded_skill(skill, data_dir, skills=[skill])
    granted_visible = ext_name in {schema["function"]["name"] for schema in registry.schemas(core_only=False)}
    for case, state, visible in (("extension_missing_grant", missing, missing_visible), ("extension_granted_live", granted, granted_visible)):
        allowed = bool(state["desired_live"] and state["live_loaded"]); grant = state["grant_status"]
        cases.append({"case": case, "allowed": allowed, "llm_calls": 0, "audit_events": [], "visible": visible, "visibility_surface": "ToolRegistry.schemas(core_only=False)", "owner_decision": {"reason": state["reason"], "desired_live": state["desired_live"], "live_loaded": state["live_loaded"], "grant_status": {key: grant[key] for key in ("missing_keys", "missing_permissions", "granted_keys", "granted_permissions", "all_granted", "usable", "content_hash")}}, "legacy_result": _legacy_record("")})
    with extension_loader._lock: extension_loader._extensions.pop("fixture", None); extension_loader._tools.pop(ext_name, None)
    safety.get_safety_mode = lambda: "full"
    safety._run_llm_check = lambda tool, *_a, **_k: (calls.append(tool) or (True, ""))
    extension_loader.unload_extension = lambda skill: unloaded.append(skill)
    try:
        stale_live = False
        extension_loader.is_extension_live = lambda *_a, **_k: stale_live
        stale = dispatch_extension_tool(ctx, "ext_7_fixture_echo", {
            "name": "ext_7_fixture_echo", "skill": "fixture", "handler": lambda: "unused",
        }, {})
        expected_stale = "⚠️ TOOL_ERROR (ext_7_fixture_echo): extension 'fixture' is not allowed to dispatch right now."
        if stale != expected_stale or unloaded != ["fixture"]:
            raise RuntimeError("extension stale characterization drifted")
        cases.append({
            "case": "extension_stale",
            "allowed": stale_live,
            "llm_calls": len(calls),
            "audit_events": [],
            "owner_decision": {
                "owner": "ouroboros.extension_loader.is_extension_live",
                "live": stale_live,
                "dispatch_allowed": stale_live,
            },
            "side_effects": {"unloaded": unloaded},
            "legacy_result": _legacy_record(stale),
        })
        calls.clear()
        extension_live = True
        extension_loader.is_extension_live = lambda *_a, **_k: extension_live
        failed = dispatch_extension_tool(ctx, "ext_7_fixture_echo", {
            "name": "ext_7_fixture_echo",
            "skill": "fixture",
            "handler": lambda: (_ for _ in ()).throw(RuntimeError("fixture extension failure")),
        }, {})
        expected_failed = "⚠️ TOOL_ERROR (ext_7_fixture_echo): extension tool failed: RuntimeError: fixture extension failure"
        if failed != expected_failed or calls != ["ext_7_fixture_echo"]:
            raise RuntimeError("extension exception characterization drifted")
        cases.append({
            "case": "extension_exception",
            "allowed": extension_live,
            "llm_calls": len(calls),
            "audit_events": [],
            "owner_decision": {
                "owner": "ouroboros.extension_loader.is_extension_live + ouroboros.safety.check_safety",
                "live": extension_live,
                "safety_allowed": True,
                "dispatch_allowed": True,
                "handler_outcome": "exception",
            },
            "legacy_result": _legacy_record(failed),
        })
    finally:
        extension_loader.is_extension_live = original_live
        extension_loader.unload_extension = original_unload
        safety.get_safety_mode = original_mode
        safety._run_llm_check = original_llm_check
    manager = mcp_client.MCPManager()
    manager.reconfigure({
        "MCP_ENABLED": True,
        "MCP_TOOL_TIMEOUT_SEC": 60,
        "MCP_SERVERS": [{"id": "fixture", "enabled": True, "transport": "streamable_http", "url": "https://example.invalid/mcp", "allowed_tools": ["ok"]}],
    })
    not_found = manager.call_tool("mcp_fixture__missing", {})
    expected_not_found = (
        "⚠️ MCP_TOOL_NOT_FOUND: 'mcp_fixture__missing'. Refresh the server in "
        "Settings → Advanced or check the allowed_tools allowlist."
    )
    tool_found = False
    if not_found != expected_not_found or not manager.is_enabled():
        raise RuntimeError("MCP not-found characterization drifted")
    cases.append({
        "case": "mcp_not_found", "allowed": manager.is_enabled() and tool_found,
        "llm_calls": 0, "audit_events": [],
        "owner_decision": {
            "owner": "ouroboros.mcp_client.MCPManager.call_tool",
            "manager_enabled": manager.is_enabled(),
            "configured_allowed_tools": ["ok"],
            "tool_found": tool_found,
        },
        "legacy_result": _legacy_record(not_found),
    })
    runtime = manager._servers["fixture"]
    runtime.tools = [mcp_client.MCPTool("fixture", name, f"mcp_fixture__{name}", "", {"type": "object", "properties": {}}) for name in ("ok", "blocked")]
    provider_calls: list[str] = []
    async def local_call(_cfg: Any, name: str, _args: dict[str, Any], _timeout: int) -> str: provider_calls.append(name); return "fixture allowed"
    manager._async_call_tool = local_call
    visible = [tool["name"] for tool in manager.list_tools_for_registry()]; allowed_text = manager.call_tool("mcp_fixture__ok", {}); allowed_calls = list(provider_calls); denied_text = manager.call_tool("mcp_fixture__blocked", {}); denied_calls = provider_calls[len(allowed_calls):]
    cases.extend([{"case": "mcp_allowed_tool", "allowed": True, "llm_calls": 0, "audit_events": [], "visible_names": visible, "provider_calls": allowed_calls, "legacy_result": _legacy_record(allowed_text)}, {"case": "mcp_disallowed_tool", "allowed": False, "llm_calls": 0, "audit_events": [], "visible_names": visible, "provider_calls": denied_calls, "legacy_result": _legacy_record(denied_text)}])
    remote_result = SimpleNamespace(
        content=[SimpleNamespace(text="fixture MCP failure")], isError=True,
    )
    is_error = mcp_client._stringify_call_result(remote_result)
    if is_error != "⚠️ MCP_TOOL_ERROR: fixture MCP failure":
        raise RuntimeError("MCP isError characterization drifted")
    cases.append({
        "case": "mcp_is_error", "allowed": not bool(remote_result.isError),
        "llm_calls": 0, "audit_events": [],
        "owner_decision": {
            "owner": "ouroboros.mcp_client._stringify_call_result",
            "remote_is_error": bool(remote_result.isError),
            "outcome": "error",
        },
        "legacy_result": _legacy_record(is_error),
    })
    return cases
def _probe_runtime() -> dict[str, Any]:
    repo_dir = pathlib.Path(os.environ["OUROBOROS_REPO_DIR"])
    data_dir = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"])
    from ouroboros.contracts.plugin_api import (
        ALWAYS_AVAILABLE_CAPABILITIES,
        MATRIX_CAPABILITIES,
        OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES,
        PLUGIN_API_VERSION,
        ExecutionMode,
        PluginAPI,
        capability_available,
    )
    from ouroboros.contracts.tool_abi import GetToolsProtocol, ToolEntryProtocol
    from ouroboros.contracts.tool_context import ToolContextProtocol
    from ouroboros.gateway import contracts as gateway_contracts
    from ouroboros.runtime_mode_policy import (
        FROZEN_CONTRACT_PATHS,
        FROZEN_CONTRACT_PATH_PREFIXES,
        PROTECTED_RUNTIME_PATHS,
        RELEASE_INVARIANT_PATHS,
        SAFETY_CRITICAL_PATHS,
        protected_path_category,
    )
    from ouroboros.safety import TOOL_POLICY
    from ouroboros.tool_access import _ALL_ROOTS, _POLICY, Operation, active_tool_profile, decide_tool_access
    from ouroboros.tools.registry import ToolContext, ToolRegistry
    import ouroboros.protected_artifacts as protected_artifacts
    plugin_methods = sorted(
        name for name in dir(PluginAPI)
        if not name.startswith("_") and callable(getattr(PluginAPI, name, None))
    )
    plugin_signatures = {name: _symbol_signature(getattr(PluginAPI, name)) for name in plugin_methods}
    plugin_matrix = {
        mode.value: {name: capability_available(name, mode) for name in plugin_methods}
        for mode in ExecutionMode
    }
    registry = ToolRegistry(repo_dir=repo_dir, drive_root=data_dir)
    worktree = repo_dir.parent / "workspace"
    worktree.mkdir()
    workspace_external = all(root.resolve() not in (worktree.resolve(), *worktree.resolve().parents) for root in (repo_dir, data_dir))
    if not workspace_external:
        raise RuntimeError("workspace fixture must be outside repo and data roots")
    contexts = {
        "normal": ToolContext(repo_dir=repo_dir, drive_root=data_dir),
        "workspace": ToolContext(repo_dir=repo_dir, drive_root=data_dir, workspace_root=worktree, workspace_mode="project"),
        "local_readonly": ToolContext(repo_dir=repo_dir, drive_root=data_dir, task_constraint={"mode": "local_readonly_subagent"}),
        "acting": ToolContext(repo_dir=repo_dir, drive_root=data_dir, workspace_root=worktree, workspace_mode="self_worktree", task_constraint={"mode": "acting_subagent", "surface": "self_worktree", "write_root": str(worktree)}),
        "heal": ToolContext(repo_dir=repo_dir, drive_root=data_dir, task_constraint={"mode": "skill_repair", "skill_name": "fixture", "payload_root": "skills/external/fixture"}),
        "ephemeral": ToolContext(repo_dir=repo_dir, drive_root=data_dir, is_ephemeral_turn=True),
    }
    contextual: dict[str, Any] = {}
    expected_profiles = {"normal": "self_modification", "workspace": "workspace_task",
        "local_readonly": "local_readonly_subagent", "acting": "acting_subagent",
        "heal": "skill_repair", "ephemeral": "self_modification"}
    dynamic_schema_hashes: dict[str, dict[str, str]] = {name: {} for name in registry._entries}
    for label, ctx in contexts.items():
        registry.set_context(ctx)
        profile = active_tool_profile(ctx)
        if profile != expected_profiles[label] or (label == "workspace" and not ctx.is_workspace_mode()):
            raise RuntimeError(f"context profile drifted: {label} -> {profile}")
        advertised = registry.schemas(core_only=False)
        visible = sorted(schema["function"]["name"] for schema in advertised)
        contextual[label] = {
            "surface": "ToolRegistry.schemas(core_only=False)",
            "visible_names": visible,
            "count": len(visible),
            "active_profile": profile, "is_workspace_mode": ctx.is_workspace_mode(),
            "workspace_root_external": workspace_external if label == "workspace" else None,
            "capability_omissions": registry.capability_omissions(),
        }
        for name, entry in sorted(registry._entries.items()):
            dynamic_schema_hashes[name][label] = _sha256_json(registry._schema_for_entry(entry))
    inventory = []
    for name, entry in sorted(registry._entries.items()):
        inventory.append({
            "name": name,
            "module": str(getattr(entry.handler, "__module__", "")),
            "schema_sha256": _sha256_json(entry.schema),
            "dynamic_schema_sha256": dynamic_schema_hashes[name],
            "timeout_sec": entry.timeout_sec,
            "is_code_tool": bool(entry.is_code_tool),
            "mutates_worktree": bool(entry.mutates_worktree),
            "policy": TOOL_POLICY.get(name),
        })
    from ouroboros.consciousness import BackgroundConsciousness
    background = BackgroundConsciousness(data_dir, repo_dir, None, lambda: None)
    wake_entry = background._registry._entries["set_next_wakeup"]
    wake_cases: list[dict[str, Any]] = []
    import ouroboros.safety as safety
    original_mode = safety.get_safety_mode
    original_llm_check = safety._run_llm_check
    original_skip = safety._emit_safety_mode_skip
    try:
        for mode in ("full", "light", "off"):
            wake_calls: list[str] = []
            wake_audits: list[dict[str, str]] = []
            safety.get_safety_mode = lambda value=mode: value
            safety._run_llm_check = lambda tool, *_a, **_k: (wake_calls.append(tool) or (True, ""))
            safety._emit_safety_mode_skip = lambda _ctx, tool, value, policy: wake_audits.append({
                "type": "safety_mode_skip", "tool": tool, "safety_mode": value, "policy": policy,
            })
            wake_text = background._registry.execute("set_next_wakeup", {"seconds": 5})
            if wake_text != "OK: next wakeup in 60s" or wake_calls:
                raise RuntimeError(f"set_next_wakeup characterization drifted in {mode}")
            wake_cases.append({
                "case": "set_next_wakeup_scoped",
                "policy": "skip",
                "mode": mode,
                "allowed": True,
                "llm_calls": len(wake_calls),
                "audit_events": wake_audits,
                "legacy_result": _legacy_record(wake_text),
            })
    finally:
        safety.get_safety_mode = original_mode
        safety._run_llm_check = original_llm_check
        safety._emit_safety_mode_skip = original_skip
    scoped = {
        "name": wake_entry.name,
        "module": str(getattr(wake_entry.handler, "__module__", "")),
        "scope": "background_consciousness",
        "schema_sha256": _sha256_json(wake_entry.schema),
        "dynamic_schema_sha256": _sha256_json(background._registry._schema_for_entry(wake_entry)),
        "timeout_sec": wake_entry.timeout_sec,
        "is_code_tool": bool(wake_entry.is_code_tool), "mutates_worktree": bool(wake_entry.mutates_worktree),
        "policy": TOOL_POLICY.get(wake_entry.name),
    }
    access_cells = []
    profiles = sorted(_POLICY)
    roots = sorted(_ALL_ROOTS)
    operations = list(getattr(Operation, "__args__", ()))
    for profile in profiles:
        for root in roots:
            for operation in operations:
                decision = decide_tool_access(profile=profile, root=root, operation=operation)
                access_cells.append({
                    "profile": profile,
                    "root": root,
                    "operation": operation,
                    "allow": decision.allow,
                    "reason": decision.reason,
                    "guard": decision.guard,
                })
    policy_counts = {policy: sum(value == policy for value in TOOL_POLICY.values()) for policy in sorted(set(TOOL_POLICY.values()))}
    runtime_paths = sorted(PROTECTED_RUNTIME_PATHS)
    protected_projection = {
        "owner": "ouroboros/runtime_mode_policy.py",
        "runtime_paths": [{"path": path, "category": protected_path_category(path)} for path in runtime_paths],
        "runtime_prefixes": list(FROZEN_CONTRACT_PATH_PREFIXES),
        "sets": {
            "safety_critical": sorted(SAFETY_CRITICAL_PATHS),
            "frozen_contract": sorted(FROZEN_CONTRACT_PATHS),
            "release_invariant": sorted(RELEASE_INVARIANT_PATHS),
        },
        "task_artifact_owner": "ouroboros/protected_artifacts.py",
        "task_artifact_default_denied_operations": sorted(protected_artifacts._DEFAULT_DENIED_OPERATIONS),
        "channels": {
            "builtin_dispatch": "ouroboros/tools/registry.py::ToolRegistry.execute",
            "extension_dispatch": "ouroboros/tools/extension_dispatch.py::dispatch_extension_tool",
            "mcp_dispatch": "ouroboros/tools/registry.py::ToolRegistry._dispatch_mcp_tool",
            "shell_postcheck": "ouroboros/protected_artifacts.py::shell_block_reason",
        },
        "method": "source-derived constants and dispatch call sites; no filesystem mutation",
    }
    llm_symbols = (
        "cache_ttl_seconds", "LocalContextTooLargeError", "normalize_reasoning_effort",
        "add_usage", "fetch_openrouter_pricing", "fetch_cloudru_pricing", "LLMClient",
        "openrouter_web_search_server_tool", "anthropic_web_search_server_tool",
    )
    loop_symbols = ("DeliveryCandidate", "seal_task_transcript", "run_llm_loop")
    public_facades = []
    facade_modules = {}
    for module_name, symbols in (("ouroboros.llm", llm_symbols), ("ouroboros.loop", loop_symbols)):
        module = __import__(module_name, fromlist=["*"])
        facade_modules[module_name] = module
        for symbol in symbols:
            value = getattr(module, symbol)
            public_facades.append({
                "category": "production_facade",
                "facade": f"{module_name}::{symbol}",
                "owner": f"{module_name}::{symbol}",
                "signature": _symbol_signature(value),
                "identity_preserved": True,
            })
    private_imports: dict[str, list[dict[str, Any]]] = {}
    for path in sorted((repo_dir / "tests").rglob("*.py")):
        relative = path.relative_to(repo_dir).as_posix()
        nodes = ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        for node in (node for node in nodes if isinstance(node, ast.ImportFrom) and node.module == "ouroboros.loop"):
            for alias in (alias for alias in node.names if alias.name.startswith("_")):
                private_imports.setdefault(alias.name, []).append({"importer": relative, "line": node.lineno, "asname": alias.asname})
    loop_module = facade_modules["ouroboros.loop"]
    for symbol, imports in sorted(private_imports.items()):
        public_facades.append({"category": "test_private", "facade": f"ouroboros.loop::{symbol}",
            "owner": f"ouroboros.loop::{symbol}", "signature": _symbol_signature(getattr(loop_module, symbol)),
            "importers": imports, "identity_preserved": True})
    import ouroboros.contracts.api_v1 as api_v1
    import ouroboros.tools as tools_facade
    public_facades.extend([
        {"category": "external_contract", "facade": "ouroboros.contracts.api_v1", "owner": "ouroboros.gateway.contracts", "exports": list(api_v1.__all__), "identity_preserved": True},
        {"category": "production_facade", "facade": "ouroboros.tools", "owner": "ouroboros.tools.registry", "exports": list(tools_facade.__all__), "identity_preserved": True},
        {"category": "production_facade", "facade": "supervisor.queue", "owner": "supervisor.task_lifecycle and supervisor.queue_transitions", "identity_preserved": True},
        {"category": "production_facade", "facade": "web/modules/chat.js::createChatInstance.return", "owner": "web/modules/chat.js::createChatInstance", "exports": ["page", "chatId", "projectId", "restoreScrollPosition", "refreshHistory", "cancelHistoryPaint", "hasPaintedHistory", "hasPendingWork", "getScrollState", "destroy"], "identity_preserved": True},
    ])
    frozen_contracts = {
        "owner": "ouroboros/contracts and ouroboros/gateway/contracts.py",
        "plugin_api": {
            "version": PLUGIN_API_VERSION,
            "methods": plugin_signatures,
            "capability_matrix": plugin_matrix,
            "matrix_capabilities": sorted(MATRIX_CAPABILITIES),
            "always_available_capabilities": sorted(ALWAYS_AVAILABLE_CAPABILITIES),
            "out_of_process_unavailable": sorted(OUT_OF_PROCESS_UNAVAILABLE_CAPABILITIES),
        },
        "tool_context": {
            "fields": sorted(ToolContextProtocol.__annotations__),
            "methods": {name: _symbol_signature(getattr(ToolContextProtocol, name)) for name in ("repo_path", "active_repo_dir", "is_workspace_mode", "drive_path", "drive_logs")},
        },
        "tool_abi": {
            "entry_fields": sorted(ToolEntryProtocol.__annotations__),
            "get_tools_signature": _symbol_signature(GetToolsProtocol.__call__),
        },
        "gateway": {
            "exports": list(gateway_contracts.__all__),
            "http_endpoints": list(gateway_contracts.HTTP_ENDPOINTS),
            "ws_message_types": list(gateway_contracts.WS_MESSAGE_TYPES),
        },
    }
    safety_projection = _probe_safety()
    safety_projection["policy"] = {
        "entries": dict(sorted(TOOL_POLICY.items())),
        "counts": policy_counts,
        "count": len(TOOL_POLICY),
    }
    safety_projection["cases"].extend(_probe_dispatch_cases(repo_dir, data_dir))
    safety_projection["cases"].extend(_probe_extension_mcp(repo_dir, data_dir))
    safety_projection["cases"].extend(wake_cases)
    return {
        "frozen_contracts": frozen_contracts,
        "tool_catalog": {
            "owner": "ouroboros/tools/registry.py",
            "global_entries": inventory,
            "global_count": len(inventory),
            "scoped_entries": [scoped],
            "total_count": len(inventory) + 1,
            "frozen_modules": list(registry._FROZEN_TOOL_MODULES),
            "contextual_visibility": contextual,
            "inventory_sha256": _sha256_json(inventory + [scoped]),
        },
        "tool_access": {
            "owner": "ouroboros/tool_access.py",
            "profiles": profiles,
            "roots": roots,
            "operations": operations,
            "cells": access_cells,
            "cell_count": len(access_cells),
            "matrix_sha256": _sha256_json(access_cells),
        },
        "safety_differential": safety_projection,
        "protected_surfaces": protected_projection,
        "public_facades": {
            "entries": public_facades,
            "unknown_external_consumers": "residual: installed third-party skill/extension import universe is not enumerable from this checkout",
        },
    }
def _source_hashes(repo: pathlib.Path, ref: str) -> dict[str, str]:
    paths = (
        "ouroboros/contracts/plugin_api.py", "ouroboros/contracts/tool_context.py",
        "ouroboros/contracts/tool_abi.py", "ouroboros/contracts/api_v1.py", "ouroboros/contracts/task_contract.py",
        "ouroboros/gateway/contracts.py", "ouroboros/tools/registry.py",
        "ouroboros/tool_access.py", "ouroboros/tool_capabilities.py", "ouroboros/safety.py",
        "ouroboros/runtime_mode_policy.py", "ouroboros/protected_artifacts.py",
        "ouroboros/extension_loader.py", "ouroboros/tools/extension_dispatch.py",
        "ouroboros/mcp_client.py", "ouroboros/llm.py", "ouroboros/loop.py",
        "supervisor/queue.py", "supervisor/update_merge.py", "supervisor/git_ops.py",
        "web/modules/chat.js",
    )
    return {path: _sha256_bytes(_source_bytes(repo, ref, path)) for path in paths}
def _updater_imports(repo: pathlib.Path, ref: str, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    files = ("supervisor/update_merge.py", "supervisor/git_ops.py")
    expected = (
        "server", "ouroboros.gateway.router", "supervisor.queue", "supervisor.events",
        "ouroboros.tools.registry", "ouroboros", "ouroboros.agent",
    )
    derived: list[str] = []
    literals = []
    for path in files:
        source = (overrides or {}).get(path) or _source_text(repo, ref, path)
        found = []
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            for index, item in enumerate(node.elts[:-1]):
                if isinstance(item, ast.Constant) and item.value == "-c" and index == 1 and ast.unparse(node.elts[0]) == "sys.executable":
                    payload = ast.literal_eval(node.elts[index + 1])
                    imports = [alias.name for stmt in ast.parse(payload).body if isinstance(stmt, ast.Import) for alias in stmt.names]
                    found.append((item.lineno, payload, imports))
        if len(found) != 1:
            raise RuntimeError(f"expected exactly one python -c import literal in {path}; found {len(found)}")
        line, payload, imports = found[0]
        derived.extend(imports)
        literals.append({"path": path, "line": line, "python_c": payload, "imports": imports})
    if tuple(derived) != expected:
        raise RuntimeError(f"updater import literals drifted: expected {expected!r}, got {tuple(derived)!r}")
    return {
        "owner": "supervisor/update_merge.py and supervisor/git_ops.py",
        "category": "cross_version_updater",
        "paths": derived,
        "source_literals": literals,
    }
def generate_fixture(repo: pathlib.Path) -> dict[str, Any]:
    baseline = str(_git(repo, "rev-parse", BASELINE_SHA)).strip()
    observed = str(_git(repo, "rev-parse", OBSERVED_HEAD_SHA)).strip()
    if baseline != BASELINE_SHA or observed != OBSERVED_HEAD_SHA:
        raise RuntimeError("v7 evidence source commits are unavailable")
    drift_names = str(_git(repo, "diff", "--name-status", f"{BASELINE_SHA}..{OBSERVED_HEAD_SHA}")).splitlines()
    drift = []
    for line in drift_names:
        fields = line.split("\t")
        drift.append({"status": fields[0], "paths": fields[1:]})
    fixture = {
        "schema_version": 1,
        "campaign": "Ouroboros v7 prologue",
        "baseline_source_sha": BASELINE_SHA,
        "observed_head_sha": OBSERVED_HEAD_SHA,
        "observed_drift": {
            "entries": drift,
            "classification": "packaged CLI install-target fix only; no v7 contract/runtime surface drift",
        },
        "source_hashes": _source_hashes(repo, BASELINE_SHA),
        "baseline_census": _census(repo, BASELINE_SHA),
        "observed_head_census": _census(repo, OBSERVED_HEAD_SHA),
        "updater_imports": _updater_imports(repo, BASELINE_SHA),
        "runtime_probe": _probe_ref(repo, BASELINE_SHA),
        "methods": {
            "runtime": "isolated subprocess from local git archive; four temp Ouroboros roots; no external network calls are made and provider/network boundaries are stubbed or disabled",
            "source": "git show/git ls-tree against immutable object IDs; checkout HEAD is never mutated",
            "safety": "deterministic stubbed provider plus real legacy policy/dispatch composers; no future ToolResult code invented",
        },
    }
    fixture["payload_sha256"] = _sha256_json(fixture)
    return fixture
def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
_parse_ref = _migration._parse_ref
_symbol_exists = _migration._symbol_exists
_parse_migration = _migration._parse_migration
_migration_json = _migration._migration_json
validate_migration = _migration.validate_migration
def command_write(repo: pathlib.Path) -> int:
    fixture = generate_fixture(repo)
    output = repo / FIXTURE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_json_text(fixture), encoding="utf-8")
    print(f"wrote {FIXTURE_PATH} ({fixture['payload_sha256']})")
    return 0
def command_check(repo: pathlib.Path) -> int:
    expected = generate_fixture(repo)
    path = repo / FIXTURE_PATH
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"fixture unreadable: {exc}", file=sys.stderr); return 1
    if actual != expected or path.read_text(encoding="utf-8") != _json_text(expected):
        print(f"{FIXTURE_PATH} is stale; run scripts/v7_evidence.py write", file=sys.stderr); return 1
    errors = validate_migration(repo)
    if errors:
        print("\n".join(errors), file=sys.stderr); return 1
    print(f"v7 evidence OK ({expected['payload_sha256']})")
    return 0
def command_check_migration(repo: pathlib.Path) -> int:
    errors = validate_migration(repo)
    if errors:
        print("\n".join(errors), file=sys.stderr); return 1
    print("MIGRATION_v7.md OK"); return 0
def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "check-migration", "_probe"))
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command == "_probe":
        print(_canonical_json(_probe_runtime())); return 0
    repo = _repo_root()
    if args.command == "write":
        return command_write(repo)
    if args.command == "check":
        return command_check(repo)
    return command_check_migration(repo)
if __name__ == "__main__":
    raise SystemExit(main())
