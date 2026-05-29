"""
Policy-based safety check for tool calls.

Built-ins use explicit policy entries; unknown tools default to one light-model
check. The registry sandbox still runs first, Claude edits still have protected
path revert guards, and commit review remains separate.
"""

import ast
import json
import logging
import os
import pathlib
import re
import shlex
from typing import Tuple, Dict, Any, List, Optional

from ouroboros.config import get_light_model
from ouroboros.llm import LLMClient
from ouroboros.pricing import emit_llm_usage_event, estimate_cost, infer_provider_from_model
from supervisor.state import update_budget_from_usage

log = logging.getLogger(__name__)

# Policy constants.

POLICY_SKIP = "skip"
POLICY_CHECK = "check"
POLICY_CHECK_CONDITIONAL = "check_conditional"

# Unknown/agent-created tools get one cheap LLM recheck.
DEFAULT_POLICY = POLICY_CHECK

# Must cover every built-in exported from ouroboros/tools; invariant-tested.
TOOL_POLICY: Dict[str, str] = {
    # Read-only / trivially safe.
    "read_file": POLICY_SKIP,
    "list_files": POLICY_SKIP,
    "vcs_status": POLICY_SKIP,
    "vcs_diff": POLICY_SKIP,
    "chat_history": POLICY_SKIP,
    "recent_tasks": POLICY_SKIP,
    "knowledge_read": POLICY_SKIP,
    "knowledge_list": POLICY_SKIP,
    "web_search": POLICY_SKIP,
    "codebase_digest": POLICY_SKIP,
    "codebase_health": POLICY_SKIP,
    "search_code": POLICY_SKIP,
    "list_available_tools": POLICY_SKIP,
    "memory_map": POLICY_SKIP,
    "analyze_screenshot": POLICY_SKIP,
    "vlm_query": POLICY_SKIP,
    "browse_page": POLICY_SKIP,
    "browser_action": POLICY_SKIP,
    "list_github_prs": POLICY_SKIP,
    "get_github_pr": POLICY_SKIP,
    "list_github_issues": POLICY_SKIP,
    "get_github_issue": POLICY_SKIP,
    "plan_task": POLICY_SKIP,
    "task_acceptance_review": POLICY_SKIP,
    "review_status": POLICY_SKIP,
    "get_task_result": POLICY_SKIP,
    "wait_task": POLICY_SKIP,
    "wait_tasks": POLICY_SKIP,
    "switch_model": POLICY_SKIP,
    "service_status": POLICY_SKIP,
    "service_logs": POLICY_SKIP,

    # Mutative but separately guarded by sandbox/revert/review gates.
    "write_file": POLICY_SKIP,
    "edit_text": POLICY_SKIP,
    "commit_reviewed": POLICY_SKIP,
    "vcs_commit_reviewed": POLICY_SKIP,
    "knowledge_write": POLICY_SKIP,
    "update_scratchpad": POLICY_SKIP,
    "update_identity": POLICY_SKIP,
    "memory_update_registry": POLICY_SKIP,
    "vcs_pull_ff": POLICY_SKIP,
    "vcs_restore": POLICY_SKIP,
    "vcs_revert": POLICY_SKIP,
    "vcs_rollback": POLICY_SKIP,

    # Control / messaging / internal side effects.
    "schedule_subagent": POLICY_SKIP,
    "cancel_task": POLICY_SKIP,
    "request_restart": POLICY_SKIP,
    "request_deep_self_review": POLICY_SKIP,
    "set_tool_timeout": POLICY_SKIP,
    "toggle_evolution": POLICY_SKIP,
    "toggle_consciousness": POLICY_SKIP,
    "promote_to_stable": POLICY_SKIP,
    "send_user_message": POLICY_SKIP,
    "send_photo": POLICY_SKIP,
    "send_video": POLICY_SKIP,
    "forward_to_worker": POLICY_SKIP,
    "compact_context": POLICY_SKIP,
    "enable_tools": POLICY_SKIP,
    "advisory_review": POLICY_SKIP,
    "start_service": POLICY_CHECK_CONDITIONAL,
    "stop_service": POLICY_SKIP,
    "claude_code_edit": POLICY_SKIP,

    # External skill surface.
    "list_skills": POLICY_SKIP,
    # Review mutates durable skill state but executes no skill subprocess.
    "skill_review": POLICY_SKIP,
    # Toggle only writes private enabled.json state.
    "toggle_skill": POLICY_SKIP,
    # skill_exec enforces fresh executable review/enabled/hash; recheck per call.
    "skill_exec": POLICY_CHECK,
    # Read-only argv-only syntax validator with scrubbed env and per-file caps.
    "skill_preflight": POLICY_SKIP,

    # Conditional: run_command safe-subject whitelist.
    "run_command": POLICY_CHECK_CONDITIONAL,
    "run_script": POLICY_CHECK_CONDITIONAL,

    # Always LLM-checked built-ins.
    "fetch_pr_ref": POLICY_CHECK,
    "create_integration_branch": POLICY_CHECK,
    "cherry_pick_pr_commits": POLICY_CHECK,
    "stage_adaptations": POLICY_CHECK,
    "stage_pr_merge": POLICY_CHECK,
    "run_ci_tests": POLICY_CHECK,
    "generate_evolution_stats": POLICY_CHECK,
    "submit_skill_to_hub": POLICY_CHECK,
    "comment_on_pr": POLICY_CHECK,
    "comment_on_issue": POLICY_CHECK,
    "close_github_issue": POLICY_CHECK,
    "create_github_issue": POLICY_CHECK,

    # Consciousness-only built-ins registered outside get_tools().
    "set_next_wakeup": POLICY_SKIP,
}

# run_command safe-subject whitelist.

# ``pip`` mutates the Python env and must route through the LLM check.
SAFE_SHELL_COMMANDS = frozenset([
    "ls", "cat", "head", "tail", "grep", "rg", "find", "wc",
    "git", "pytest", "pwd", "whoami",
    "date", "which", "file", "stat", "diff", "tree",
    "du", "df",
])

_SAFE_PYTHON_MODULE_ALIASES = {
    "pytest": "pytest",
    "py.test": "pytest",
}


def _split_shell_command(raw_cmd: Any) -> List[str]:
    """Best-effort argv parser for safety whitelist classification."""
    if isinstance(raw_cmd, list):
        return [str(part) for part in raw_cmd if str(part).strip()]
    text = str(raw_cmd or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(part) for part in parsed if str(part).strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(part) for part in parsed if str(part).strip()]
    except (SyntaxError, ValueError):
        pass
    try:
        return [str(part) for part in shlex.split(text) if str(part).strip()]
    except ValueError:
        return text.split()


def _is_explicit_python_interpreter(executable: str) -> bool:
    """Allow literal Python interpreter tokens, not path/basename lookalikes."""
    token = str(executable or "").strip().lower()
    if not token:
        return False
    if token in {"python", "python3"}:
        return True
    return bool(re.fullmatch(r"python\d+(?:\.\d+)?", token))


def _normalize_safe_shell_subject(raw_cmd: Any) -> str:
    """Return the canonical safe subject for shell allowlisting."""
    argv = _split_shell_command(raw_cmd)
    if not argv:
        return ""

    executable = str(argv[0]).strip().lower()
    if executable in SAFE_SHELL_COMMANDS:
        return executable

    if _is_explicit_python_interpreter(executable):
        for idx, part in enumerate(argv[1:-1], start=1):
            part_str = str(part)
            if part_str == "-m":
                module = str(argv[idx + 1]).lower()
                return _SAFE_PYTHON_MODULE_ALIASES.get(module, "")
            if part_str == "-c":
                break
            # After a script path, later -m/-c belongs to that script.
            if not part_str.startswith("-"):
                break
            # After --, everything belongs to the script.
            if part_str == "--":
                break

    return ""


# LLM check plumbing.

def _get_safety_prompt() -> str:
    """Load the safety system prompt from prompts/SAFETY.md."""
    prompt_path = pathlib.Path(__file__).parent.parent / "prompts" / "SAFETY.md"
    try:
        return prompt_path.read_text(encoding="utf-8")
    except Exception as e:
        log.error(f"Failed to read SAFETY.md: {e}")
        return (
            "You are a security supervisor. Block only clearly destructive commands. "
            "Default to SAFE. Respond with JSON: "
            '{\"status\": \"SAFE\"|\"SUSPICIOUS\"|\"DANGEROUS\", \"reason\": \"...\"}'
        )


# Secret redaction.

# Segment matching avoids false positives like ``override_author``.
_SECRET_KEY_SEGMENTS = frozenset({
    "key",  # only together with prefix segment — see _is_secret_key
    "apikey",
    "secret",
    "token",
    "password",
    "passwd",
    "credential",
    "credentials",
    "cookie",
    "authorization",
})

# Prefix+suffix shapes treated as credential keys.
_SECRET_KEY_COMBO = frozenset({
    ("api", "key"),
    ("access", "key"),
    ("access", "token"),
    ("auth", "token"),
    ("auth", "key"),
    ("session", "token"),
    ("refresh", "token"),
})


def _is_secret_key(key: str) -> bool:
    """Segment-aware credential-key classifier."""
    segments = [s for s in re.split(r"[_\-]+", str(key).lower()) if s]
    if not segments:
        return False
    seg_set = set(segments)
    if any(seg in _SECRET_KEY_SEGMENTS and seg != "key" for seg in seg_set):
        return True
    for i in range(len(segments) - 1):
        if (segments[i], segments[i + 1]) in _SECRET_KEY_COMBO:
            return True
    # ``key`` alone is too ambiguous; count it only in combinations.
    return False

# Known inline secret shapes. Do not boundary-anchor sk-/pk-/rk-/gh* tokens:
# over-redaction is acceptable, under-redaction is not.
_SECRET_INLINE_PATTERNS = (
    re.compile(r"(sk|pk|rk|gh[opsu])[-_][A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-]{16,}", re.IGNORECASE),
    re.compile(r"\bapi[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._\-]{16,}['\"]?", re.IGNORECASE),
)


def _redact_secret_value(value: Any) -> Any:
    """Return a JSON-serializable redaction marker for a sensitive value."""
    if isinstance(value, str) and value:
        return f"[REDACTED: {len(value)} chars]"
    if value in (None, "", 0, False):
        return value
    return "[REDACTED]"


def _redact_secrets_in_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Redact secret-like keys and inline secret shapes from tool arguments."""
    def _walk(value: Any) -> Any:
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if _is_secret_key(k):
                    out[k] = _redact_secret_value(v)
                else:
                    out[k] = _walk(v)
            return out
        if isinstance(value, (list, tuple)):
            return [_walk(v) for v in value]
        if isinstance(value, str):
            return _redact_secrets_in_text(value)
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        # Repr fallback is also scrubbed in case it contains a token.
        return _redact_secrets_in_text(repr(value))

    try:
        return _walk(arguments)
    except Exception:
        # Never let redaction itself block every unknown tool.
        return {"_redacted": "[REDACTION_FAILED]"}


def _redact_secrets_in_text(text: str) -> str:
    """Strip common inline-secret shapes out of a free-form string."""
    redacted = text
    for pattern in _SECRET_INLINE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _format_messages_for_safety(messages: List[Dict[str, Any]]) -> str:
    """Format compact safety context, redacting before truncation."""
    parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not content or role == "tool":
            continue
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
        text = _redact_secrets_in_text(str(content))
        if len(text) > 500:
            omitted = len(text) - 500
            text = text[:500] + f" [...{omitted} chars omitted]"
        parts.append(f"[{role}] {text}")
    return "\n".join(parts)


def _build_check_prompt(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    safe_args = _redact_secrets_in_arguments(arguments or {})
    try:
        args_json = json.dumps(safe_args, indent=2, default=repr)
    except Exception:
        args_json = repr(safe_args)
    runtime_mode = os.environ.get("OUROBOROS_RUNTIME_MODE", "advanced") or "advanced"
    prompt = (
        "Proposed tool call:\n"
        f"Runtime mode: {runtime_mode}\n"
        f"Tool: {tool_name}\n"
        f"Arguments:\n```json\n{args_json}\n```\n"
    )
    if messages:
        context = _format_messages_for_safety(messages)
        if context.strip():
            prompt += f"\nConversation context:\n{context}\n"
    prompt += "\nIs this safe?"
    return prompt


def _parse_safety_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse JSON from LLM response, handling markdown code fences."""
    clean = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return None


_REMOTE_PROVIDER_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_PASSWORD",
)

_LOCAL_ROUTING_KEYS = (
    "USE_LOCAL_MAIN",
    "USE_LOCAL_CODE",
    "USE_LOCAL_LIGHT",
    "USE_LOCAL_CONSCIOUSNESS",
    "USE_LOCAL_FALLBACK",
)

# Provider-specific API key mapped from ``infer_api_key_type`` result.
_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
    "cloudru": "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "gigachat": "GIGACHAT_CREDENTIALS",
}


def _any_remote_provider_configured() -> bool:
    return any(str(os.environ.get(k, "") or "").strip() for k in _REMOTE_PROVIDER_KEYS)


def _any_local_routing_enabled() -> bool:
    return any(
        str(os.environ.get(k, "") or "").lower() in ("true", "1")
        for k in _LOCAL_ROUTING_KEYS
    )


def _light_model_has_reachable_provider(light_model: str) -> bool:
    """Return whether the light model's direct provider config is reachable."""
    try:
        from ouroboros.pricing import infer_api_key_type
        key_type = infer_api_key_type(light_model)
    except Exception:  # pragma: no cover — defensive
        return True  # don't over-block on classifier failure
    if key_type == "gigachat":
        # GigaChat accepts either an authorization key (OAuth) or user/password.
        has_creds = bool(str(os.environ.get("GIGACHAT_CREDENTIALS", "") or "").strip())
        has_basic = bool(str(os.environ.get("GIGACHAT_USER", "") or "").strip()) and bool(
            str(os.environ.get("GIGACHAT_PASSWORD", "") or "").strip()
        )
        return has_creds or has_basic
    env_key = _PROVIDER_KEY_ENV.get(key_type)
    if env_key is None:
        return True
    if not str(os.environ.get(env_key, "") or "").strip():
        return False
    if key_type == "openai-compatible":
        base_url = (
            str(os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "") or "").strip()
            or str(os.environ.get("OPENAI_BASE_URL", "") or "").strip()
        )
        if not base_url:
            return False
    return True


def _resolve_safety_routing() -> Tuple[bool, bool, Optional[str]]:
    """Choose local/remote safety backend; unreachable fallback fails open."""
    if str(os.environ.get("USE_LOCAL_LIGHT", "") or "").lower() in ("true", "1"):
        return True, False, None

    light_model = get_light_model()

    if _any_remote_provider_configured():
        # The direct light-model provider needs its own key.
        if _light_model_has_reachable_provider(light_model):
            return False, False, None
        if _any_local_routing_enabled():
            # Provider mismatch: local is fallback, so local outage is tolerated.
            return True, True, None
        return False, False, (
            f"Light model provider key missing for {light_model} "
            f"(other remote keys are set but they don't cover this provider); "
            "skipping check."
        )

    if _any_local_routing_enabled():
        # Local-only configs should warn, not hard-block, on local outage.
        return True, True, None

    return False, False, (
        "No safety LLM available (neither remote provider keys nor local "
        "routing are configured); skipping check."
    )


_UNCHECKED_WARNING_SUFFIX = (
    "The tool call was allowed so the agent is not hard-blocked on a misconfigured "
    "runtime — the hardcoded sandbox (registry.py SAFETY_CRITICAL_PATHS, mutative-git "
    "via shell, gh repo/auth) still applies to every tool, and the claude_code_edit "
    "post-execution revert still applies when the failing call is claude_code_edit."
)


def _run_llm_check(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]],
    ctx: Optional[Any],
) -> Tuple[bool, str]:
    """Run a single light-model safety check and classify the verdict."""
    _use_local_light, _is_local_fallback, _skip_reason = _resolve_safety_routing()
    if _skip_reason is not None:
        log.warning("Safety backend unavailable for %s: %s", tool_name, _skip_reason)
        return True, (
            f"⚠️ SAFETY_WARNING: Safety backend is not configured "
            f"({_skip_reason.rstrip('.')}). {_UNCHECKED_WARNING_SUFFIX}"
        )

    prompt = _build_check_prompt(tool_name, arguments, messages)
    client = LLMClient()

    light_model = get_light_model()
    log.info(f"Running safety check on {tool_name} using {light_model} (local={_use_local_light})")

    try:
        from ouroboros.llm_observability import chat_observed

        msg, usage = chat_observed(
            client,
            drive_root=pathlib.Path(getattr(ctx, "drive_root", "../data")) if ctx is not None else pathlib.Path("../data"),
            task_id=str(getattr(ctx, "task_id", "") or "safety"),
            call_type="safety_supervisor",
            messages=[
                {"role": "system", "content": _get_safety_prompt()},
                {"role": "user", "content": prompt},
            ],
            model=light_model,
            use_local=_use_local_light,
        )
    except Exception as e:
        from ouroboros.utils import sanitize_tool_result_for_log

        safe_error = sanitize_tool_result_for_log(f"{type(e).__name__}: {e}")
        # Fallback local outage warns instead of blocking all unknown tools.
        if _use_local_light and _is_local_fallback:
            log.warning(
                "Safety local-fallback LLM call failed for %s (%s); proceeding with warning",
                tool_name, safe_error,
            )
            return True, (
                f"⚠️ SAFETY_WARNING: Local safety runtime unreachable ({safe_error}). "
                f"{_UNCHECKED_WARNING_SUFFIX}"
            )
        log.error("Safety check LLM call failed for %s: %s", tool_name, safe_error)
        return False, f"⚠️ SAFETY_VIOLATION: Safety check failed with error: {safe_error}"

    if usage:
        # Use provider-canonical model identity for cost/events.
        resolved_model = str(usage.get("resolved_model") or light_model)
        if _use_local_light:
            provider = "local"
            model_name = f"{light_model} (local)"
        else:
            provider = str(usage.get("provider") or infer_provider_from_model(light_model))
            model_name = resolved_model
        cost = float(usage.get("cost") or 0.0)
        if not _use_local_light and cost == 0.0:
            cost = estimate_cost(
                resolved_model,
                int(usage.get("prompt_tokens") or 0),
                int(usage.get("completion_tokens") or 0),
                int(usage.get("cached_tokens") or 0),
                int(usage.get("cache_write_tokens") or 0),
            )
            # Budget fallback below needs cost in the usage dict.
            usage["cost"] = cost
        _eq = getattr(ctx, "event_queue", None) if ctx is not None else None
        if _eq is not None:
            emit_llm_usage_event(
                _eq,
                getattr(ctx, "task_id", "") if ctx is not None else "",
                model_name, usage, cost,
                category="safety",
                provider=provider,
                source="safety_check",
            )
        else:
            update_budget_from_usage(usage)

    result = _parse_safety_response(msg.get("content") or "")
    if result is None:
        log.error(f"Safety check returned invalid JSON for {tool_name}: {msg.get('content')}")
        return False, "⚠️ SAFETY_VIOLATION: Safety Supervisor returned unparseable response."

    status = str(result.get("status", "")).upper()
    reason = result.get("reason", "Unknown")

    if status == "SAFE":
        return True, ""

    if status == "SUSPICIOUS":
        log.warning(f"Safety check: {tool_name} is suspicious: {reason}")
        return True, (
            f"⚠️ SAFETY_WARNING: The Safety Supervisor flagged this action as suspicious.\n"
            f"Reason: {reason}\n"
            f"The command was allowed, but consider whether this is the right approach."
        )

    # DANGEROUS or unrecognised status: fail safe.
    log.error(f"Safety check blocked {tool_name}: {reason}")
    return False, (
        f"⚠️ SAFETY_VIOLATION: The Safety Supervisor blocked this command.\n"
        f"Reason: {reason}\n\n"
        f"You must find a different, safer approach to achieve your goal."
    )


# Public entry point.

def check_safety(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Return ``(allowed, warning_or_error)`` for one tool call."""
    # Arguments can be None for no-parameter tool calls.
    tool_name = str(tool_name or "").strip()
    arguments = arguments or {}
    policy = TOOL_POLICY.get(tool_name, DEFAULT_POLICY)

    if policy == POLICY_SKIP:
        return True, ""

    if policy == POLICY_CHECK_CONDITIONAL:
        raw_cmd = arguments.get("cmd", arguments.get("command", ""))
        if tool_name == "run_script":
            raw_cmd = arguments.get("script", raw_cmd)
        if _normalize_safe_shell_subject(raw_cmd):
            return True, ""
        return _run_llm_check(tool_name, arguments, messages, ctx)

    return _run_llm_check(tool_name, arguments, messages, ctx)
