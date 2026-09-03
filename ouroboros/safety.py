"""
Policy-based safety check for tool calls.

Built-ins use explicit policy entries; unknown tools default to one light-model
check. The registry sandbox still runs first, Claude edits still have protected
path revert guards, and commit review remains separate.
"""

import ast
import json
import logging
import math
import os
import pathlib
import re
import shlex
import time
from typing import Tuple, Dict, Any, List, Optional

from ouroboros.config import get_light_model, get_safety_call_timeout_sec, get_safety_max_tokens, get_safety_mode
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import classify_llm_exception, is_rate_limit_text
from ouroboros.pricing import emit_llm_usage_event, estimate_cost_optional, infer_provider_from_model
from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso
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
    "journal_read": POLICY_SKIP,
    "workpad_read": POLICY_SKIP,
    "web_search": POLICY_SKIP,
    "codebase_health": POLICY_SKIP,
    "search_code": POLICY_SKIP,
    "query_code": POLICY_SKIP,
    "list_available_tools": POLICY_SKIP,
    "memory_map": POLICY_SKIP,
    "analyze_screenshot": POLICY_SKIP,
    "vlm_query": POLICY_SKIP,
    "view_image": POLICY_SKIP,
    "ocr_pdf": POLICY_SKIP,
    "youtube_transcript": POLICY_SKIP,
    "extract_video_frames": POLICY_SKIP,
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
    "peek_task": POLICY_SKIP,
    "wait_task": POLICY_SKIP,
    "wait_tasks": POLICY_SKIP,
    "list_projects": POLICY_SKIP,
    "switch_model": POLICY_SKIP,
    "service_status": POLICY_SKIP,
    "service_logs": POLICY_SKIP,

    # Mutative but separately guarded by sandbox/revert/review gates.
    "write_file": POLICY_SKIP,
    "edit_text": POLICY_SKIP,
    "apply_patch": POLICY_SKIP,
    "edit_batch": POLICY_SKIP,
    "commit_reviewed": POLICY_SKIP,
    "vcs_commit_reviewed": POLICY_SKIP,
    "knowledge_write": POLICY_SKIP,
    "journal_write": POLICY_SKIP,
    "workpad_write": POLICY_SKIP,
    # Bounded tree-scoped coordination. Tagged child-result dispositions are validated
    # and persisted only by join_ledger; neither branch has an external/repo effect.
    "tree_note": POLICY_SKIP,
    "tree_read": POLICY_SKIP,
    "promote_chat_to_task": POLICY_SKIP,
    "ensure_project_scope": POLICY_SKIP,
    "route_to_project": POLICY_SKIP,
    "steer_task": POLICY_SKIP,
    "update_scratchpad": POLICY_SKIP,
    "update_identity": POLICY_SKIP,
    "memory_update_registry": POLICY_SKIP,
    "vcs_pull_ff": POLICY_SKIP,
    "vcs_restore": POLICY_SKIP,
    "vcs_revert": POLICY_SKIP,
    "vcs_rollback": POLICY_SKIP,

    # Control / messaging / internal side effects.
    "schedule_subagent": POLICY_SKIP,
    # One-shot deferred follow-up through the existing supervisor scheduler: the
    # future task re-enters normal admission/safety, so registration itself has
    # no reach beyond what the task already has (same reasoning as schedule_subagent).
    "schedule_followup": POLICY_SKIP,
    # Delegated sessions: the access profile is derived HOST-SIDE from the calling
    # task's own authority and cannot be widened by the model, so the nanny verbs add
    # no reach beyond what the task already has (same reasoning as schedule_subagent).
    "delegate_start": POLICY_SKIP,
    "delegate_wait": POLICY_SKIP,
    "delegate_cancel": POLICY_SKIP,
    # Answering a run's question is custody-gated to the task that started it and
    # carries no authority the task lacks (same reasoning as the verbs above).
    "delegate_answer": POLICY_SKIP,
    "cancel_task": POLICY_SKIP,
    # Parent's explicit decision to abandon a child result: stamps parent_decision +
    # records the reason on the tree ledger; tree-scoped, no external effect (like cancel_task).
    "discard_child_result": POLICY_SKIP,
    "override_delegation_constraint": POLICY_SKIP,
    "request_restart": POLICY_SKIP,
    "request_deep_self_review": POLICY_SKIP,
    "set_tool_timeout": POLICY_SKIP,
    "toggle_evolution": POLICY_SKIP,
    "toggle_consciousness": POLICY_SKIP,
    "promote_to_stable": POLICY_SKIP,
    "send_user_message": POLICY_SKIP,
    "send_photo": POLICY_SKIP,
    "send_video": POLICY_SKIP,
    "send_file": POLICY_SKIP,
    # Structured links cannot exceed the task's existing owner-chat delivery authority.
    "send_links": POLICY_SKIP,
    "presence_finish": POLICY_SKIP,
    "presence_cancel_work": POLICY_SKIP,
    "configure_presence": POLICY_SKIP,
    "initiate_presence": POLICY_SKIP,
    "escalate": POLICY_SKIP,
    "forward_to_worker": POLICY_SKIP,
    "compact_context": POLICY_SKIP,
    "enable_tools": POLICY_SKIP,
    "preflight_review": POLICY_SKIP,
    "advisory_review": POLICY_SKIP,  # compat alias of preflight_review
    "start_service": POLICY_CHECK_CONDITIONAL,
    "stop_service": POLICY_SKIP,

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
    # verify_and_record runs the agent's declared `check` command like run_command,
    # so it carries the same conditional safe-subject gate over that command (FR3).
    "verify_and_record": POLICY_CHECK_CONDITIONAL,

    # Read-only best-of-N comparison of children's returned patches (applies nothing).
    "compare_subagent_patches": POLICY_SKIP,

    # Always LLM-checked built-ins.
    # Applies a subagent's patch into the live repo/worktree (protected-path + no-commit
    # gated inside the tool); keep an extra LLM safety look on the integration itself.
    "integrate_subagent_patch": POLICY_CHECK,
    "integrate_delegated_patch": POLICY_CHECK,
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
# ``find`` is NOT safe: -delete / -exec rm make it a mutator, so it routes
# through the LLM safety check like other mutating commands.
SAFE_SHELL_COMMANDS = frozenset([
    "ls", "cat", "head", "tail", "grep", "rg", "wc",
    "pytest", "pwd", "whoami",
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


def _normalize_resolved_python_subject(raw_cmd: Any, python_resolution: Any) -> str:
    """Reuse the literal-Python fast path only for resolver-attested argv.

    Arbitrary absolute executables remain ineligible.  The registry passes the
    exact resolver result, and this projection restores only its original
    literal ``python`` token for the existing module allowlist decision.
    """

    from ouroboros.process_interpreters import PythonResolutionTrace

    if not isinstance(python_resolution, PythonResolutionTrace) or not python_resolution.verified:
        return ""
    if not isinstance(raw_cmd, (list, tuple)) or not raw_cmd:
        return ""
    argv = [str(part) for part in raw_cmd]
    if argv[0] != python_resolution.resolved_interpreter:
        return ""
    requested = python_resolution.requested_interpreter
    if requested not in {"python", "python3"}:
        return ""
    return _normalize_safe_shell_subject([requested, *argv[1:]])


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


# Safety-supervisor prompt-cache TTL. The SAFETY.md system prefix is byte-stable and
# dominates this lane's input, but the gaps between checks are set by how long the
# PRECEDING tool ran — a subagent wave, a test suite or a review round routinely puts
# minutes between two consecutive checks, so the provider's 5-minute default would
# re-pay the write multiplier several times per task. The 1h tier pays the higher write
# once and reads at 0.1x for the rest of the hour; it breaks even against 5m after the
# second expiry, which a task-length run of ~46 checks passes with room to spare.
_SAFETY_CACHE_TTL = "1h"


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


# Character budget for the CONVERSATION section of the safety prompt. It was
# unbounded: every retained round of a long task rode into every light-model safety
# call, so the prompt grew with the task until the provider rate-limited the safety
# lane. The tool proposal is the SUBJECT of the check and stays outside this budget.
_SAFETY_CONTEXT_CHAR_BUDGET = 4000
_SAFETY_OMISSION_MARKER = "[… {n} older messages omitted]"

# Character budget for the serialized SUBJECT (the proposed call's own arguments).
# The subject is embedded VERBATIM — truncating it would hide the tail from the
# reviewer, so an over-budget subject is REFUSED fail-closed instead: one oversized
# run_script/run_command payload otherwise inflates this prompt past provider
# limits, and the conversation budget above cannot help because the subject rides
# outside it. The refusal is a typed policy denial with a working alternative
# (split the payload into reviewable calls), never a truncated half-reviewed call.
_SAFETY_SUBJECT_CHAR_BUDGET = 250_000


def _format_messages_for_safety(messages: List[Dict[str, Any]]) -> str:
    """Format compact safety context, redacting before truncation.

    Bounded by ``_SAFETY_CONTEXT_CHAR_BUDGET``, selected NEWEST-FIRST (the rounds that
    produced the proposed call are the ones that explain it) and rendered back in
    chronological order. The marker carries the exact count of dropped messages and its
    own space is RESERVED INSIDE the budget, so a bounded transcript is never a silent
    one (BIBLE P1). The per-message 500-char clip keeps its own inline note.
    """
    rendered: List[str] = []
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
        rendered.append(f"[{role}] {text}")
    kept: List[str] = []
    used = 0
    for idx in range(len(rendered) - 1, -1, -1):
        # Worst case if this is the last line taken: exactly ``idx`` older ones dropped.
        reserve = (len(_SAFETY_OMISSION_MARKER.format(n=idx)) + 1) if idx else 0
        cost = len(rendered[idx]) + (1 if kept else 0)
        if kept and used + cost + reserve > _SAFETY_CONTEXT_CHAR_BUDGET:
            break
        used += cost
        kept.append(rendered[idx])
    kept.reverse()
    dropped = len(rendered) - len(kept)
    if dropped:
        kept.insert(0, _SAFETY_OMISSION_MARKER.format(n=dropped))
    return "\n".join(kept)


def _render_subject_json(arguments: Dict[str, Any]) -> str:
    """Serialize the redacted call arguments exactly as the check prompt embeds them.

    ``ensure_ascii=False`` keeps serialized length ≈ input length: the default
    \\uXXXX escaping inflates non-ASCII text 6×, which would make the subject
    budget below a false promise for e.g. Cyrillic scripts. The prompt goes to
    an LLM; UTF-8 is fine.
    """
    safe_args = _redact_secrets_in_arguments(arguments or {})
    try:
        rendered = json.dumps(safe_args, indent=2, default=repr, ensure_ascii=False)
    except Exception:
        rendered = repr(safe_args)
    # A lone surrogate (a validly PARSED \ud800 escape in tool args) survives
    # json.dumps but is unencodable UTF-8 and would explode the provider send
    # downstream; the prompt is a rendering, so substitute rather than crash.
    return rendered.encode("utf-8", "replace").decode("utf-8")


def _build_check_prompt(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
    *,
    args_json: Optional[str] = None,
) -> str:
    if args_json is None:
        args_json = _render_subject_json(arguments)
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
    """Parse a safety JSON object from a model response.

    Safety reviewers occasionally wrap JSON in prose despite the prompt. We
    accept the first object that has the expected shape, but still fail closed
    when no valid object exists.
    """
    clean = text.replace("```json", "").replace("```", "").strip()
    candidates = [clean]
    depth = 0
    start = -1
    in_string = False
    escape = False
    for idx, ch in enumerate(clean):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(clean[start:idx + 1])
                start = -1
    best: Dict[str, Any] | None = None
    rank = {"SAFE": 1, "SUSPICIOUS": 2, "DANGEROUS": 3}
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        status = str(obj.get("status") or "").upper()
        if status in {"SAFE", "SUSPICIOUS", "DANGEROUS"}:
            if best is None or rank[status] > rank[str(best.get("status") or "").upper()]:
                best = obj
    return best


def _classify_safety_parse_failure(msg: Dict[str, Any], usage: Optional[Dict[str, Any]]) -> str:
    """Classify an unparseable safety response for the durable event (v6.54.3).

    ``empty`` (no content came back), ``truncated`` (the output budget was
    exhausted before the JSON closed), or ``unparseable`` (content present but no
    valid status object). Distinct classes need distinct fixes — model routing vs
    output budget vs prompt — so the event must not flatten them."""
    content = str((msg or {}).get("content") or "").strip()
    if not content:
        return "empty"
    if int((usage or {}).get("completion_tokens") or 0) >= get_safety_max_tokens():
        return "truncated"
    return "unparseable"


_REMOTE_PROVIDER_KEYS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "MINIMAX_API_KEY",
    "OPENAI_COMPATIBLE_API_KEY",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "GIGACHAT_PASSWORD",
)

_LOCAL_ROUTING_KEYS = (
    "USE_LOCAL_MAIN",
    "USE_LOCAL_HEAVY",
    "USE_LOCAL_LIGHT",
    "USE_LOCAL_CONSCIOUSNESS",
    "USE_LOCAL_FALLBACK",
)

# Provider-specific API key mapped from ``infer_api_key_type`` result.
_PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "minimax": "MINIMAX_API_KEY",
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


def _safety_deadline_epoch(ctx: Optional[Any]) -> Optional[float]:
    """Task deadline as epoch seconds from the live ToolContext metadata. ToolContext has no
    ``deadline_ts`` field, so derive it the same way loop.py::_task_deadline_epoch does — this
    bounds the model-concurrency slot wait by the REAL task deadline (else the 180s ceiling)."""
    meta = getattr(ctx, "task_metadata", {}) if ctx is not None else {}
    if not isinstance(meta, dict):
        return None
    try:
        from ouroboros.deadline_utils import parse_deadline_ts

        dl = parse_deadline_ts(meta.get("deadline_at"))
        return dl.timestamp() if dl is not None else None
    except Exception:
        return None


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
    "via shell, gh repo/auth) still applies to every tool."
)

# Only a genuine THROUGHPUT signal may wave a safety check through, so the predicate is
# narrower than "retryable". `classify_llm_exception` folds every transient class —
# 408, 5xx and timeouts included — into `provider_transient`, and those are outages, not
# throttling: they keep today's blocking SAFETY_VIOLATION path. A `provider_transient`
# therefore qualifies only when the provider actually said 429, by status code or by the
# shared rate-limit text vocabulary (`is_rate_limit_text` — imported rather than copied,
# so the two lanes cannot fork). Permanent classes are already excluded upstream: a
# structured `insufficient_quota` wins over a 429 status inside that helper and still
# blocks. On the HTTP-200 body lane (`llm._normalize_remote_response`) the qualifying
# kind is exactly `rate_limit`, which that module assigns only to a transient body error
# whose code IS 429; its other body transients stay blocking too.
# Backoff: one 2.0s sleep, inside the range the loop's own transient pattern
# (`2.0 ** attempt`) spans; no new setting, because this lane has exactly one retry.
_SAFETY_BODY_RATE_LIMIT_KIND = "rate_limit"


def _body_is_quota_refusal(body: Dict[str, Any]) -> bool:
    """A body-shaped 429 whose STRUCTURED fields name a QUOTA/BILLING refusal is
    permanent, not throttling — mirror the exception lane's quota-over-429 precedence
    (the markers are loop_llm_call's non-retryable quota vocabulary, imported as the
    SSOT). Only `code`/`type` are scanned and numeric markers are skipped: the shared
    markers are substrings, and free-form `message` text carries request ids and
    hostnames where `402`/`billing` occur incidentally ("request id: req_402ab19",
    "billing-tier throughput cap") — a false quota match here would keep reporting a
    plain throttle as a verdict, the exact bug this lane exists to remove."""
    from ouroboros.loop_llm_call import NON_RETRYABLE_PROVIDER_MARKERS

    text = f"{body.get('code') or ''} {body.get('type') or ''}".lower()
    return any(
        m in text for m in NON_RETRYABLE_PROVIDER_MARKERS["quota_exhausted"] if not m.isdigit()
    )


_SAFETY_RATE_LIMIT_BACKOFF_SEC = 2.0

# Process-local storm latch: after a NON-fallback-lane check (remote, or a local
# PRIMARY whose vLLM/proxy 429s) exhausted both attempts on a provider rate limit, every
# safety check in this process short-circuits to the same blocked SAFETY_UNAVAILABLE
# answer for this window WITHOUT a provider call (the mark/is-cooling shape of
# ouroboros/fallback_cooldown.py, inlined because this lane needs one timestamp, not a
# per-model map). The safety lane is the highest-frequency LIGHT consumer — every guarded
# tool call on every in-process subagent thread — so re-probing a storming route on each
# call would amplify the very storm it reports.
_SAFETY_STORM_COOLDOWN_SEC = 30.0
_SAFETY_STORM_UNTIL = 0.0


class _SafetyRateLimited(Exception):
    """Both safety attempts hit a provider rate limit; the caller BLOCKS this one call
    with the typed SAFETY_UNAVAILABLE outcome (infrastructure fact, not a verdict).
    ``latched`` marks the storm-window short-circuit shape: it carries no NEW storm
    evidence, so the outcome must not extend the window (else the advised retry would
    keep re-arming it forever). ``attempts`` is the PHYSICAL call count behind this
    outcome (0 latched, 1 deadline-expired-before-retry, 2 full), so the durable audit
    row can name what actually happened instead of always claiming a retry."""

    def __init__(self, message: str, *, latched: bool = False, attempts: int = 2) -> None:
        super().__init__(message)
        self.latched = latched
        self.attempts = 0 if latched else attempts


def _is_throughput_rate_limit(kind: str, status_code: Optional[int], safe_error: str) -> bool:
    """True only for provider THROTTLING, never for a plain outage (408/5xx/timeout).

    A KNOWN status code is the authority and the text is never consulted beside it: the
    shared markers are substrings, so `429`, `rpm` and `tpm` occur inside the request
    ids, trace ids and hostnames real outages carry ("503 ... (request id: req_1f429ab0)",
    "504 Gateway Timeout ... rpm-edge-proxy-3"). The text branch therefore runs ONLY when
    no status code was recoverable at all, which is the case it exists for — a transport
    that raised a bare throttling message.
    """
    if kind == _SAFETY_BODY_RATE_LIMIT_KIND:
        return True
    if kind != "provider_transient":
        return False
    return status_code == 429 or (status_code is None and is_rate_limit_text(safe_error))


def _safety_rate_limit_reason(
    exc: Optional[Exception], usage: Optional[Dict[str, Any]], safe_error: str,
) -> Optional[str]:
    """Sanitized bounded description when THIS attempt was rate-limited, else None."""
    if exc is not None:
        found = classify_llm_exception(exc, safe_error)
        if not _is_throughput_rate_limit(found.kind, found.status_code, safe_error):
            return None
        return f"{found.kind}: {safe_error}"[:300]
    body = usage.get("provider_error") if isinstance(usage, dict) else None
    if isinstance(body, dict) and _body_is_quota_refusal(body):
        return None  # structured insufficient-quota is PERMANENT: keep today's blocking path
    if not isinstance(body, dict) or str(body.get("kind") or "") != _SAFETY_BODY_RATE_LIMIT_KIND:
        return None
    return sanitize_tool_result_for_log(
        f"{body.get('kind')} (code={body.get('code')}): {body.get('message')}"
    )[:300]


def _safety_rate_limit_backoff(ctx: Optional[Any]) -> None:
    """One bounded backoff, capped by the task deadline. Called BETWEEN `model_call_slot`
    contexts: that primitive caps CONCURRENT calls, so sleeping inside a held slot would
    park the route for every other thread."""
    delay = _SAFETY_RATE_LIMIT_BACKOFF_SEC
    deadline = _safety_deadline_epoch(ctx)
    if deadline is not None:
        delay = min(delay, max(0.0, float(deadline) - time.time()))
    if delay > 0:
        time.sleep(delay)


def _rate_limited_outcome(
    ctx: Optional[Any], tool_name: str, error: str, *, local_fallback: bool, arm_latch: bool,
    attempts: int = 2,
) -> Tuple[bool, str]:
    """Terminal outcome of a rate-limited safety check, split by lane. The local-FALLBACK
    lane keeps its documented fail-open contract (ARCHITECTURE "Safety and runtime mode" case (c): a broken
    chosen-as-fallback local runtime warns instead of blocking every unknown tool) — a
    429 there must not be stricter than the RuntimeError beside it. Every other lane
    blocks with the typed non-verdict outcome below."""
    if local_fallback:
        log.warning(
            "Safety local-fallback rate-limited for %s; proceeding with warning: %s",
            tool_name, error,
        )
        _emit_durable_safety_event(ctx, {
            "type": "safety_check_rate_limited", "tool": tool_name,
            "action": "fail_open_local_fallback", "error": error,
        })
        return True, (
            f"⚠️ SAFETY_WARNING: The local fallback Safety Supervisor was rate-limited "
            f"and could not check this call ({error}). {_UNCHECKED_WARNING_SUFFIX}"
        )
    return _safety_unavailable_blocked(ctx, tool_name, error, arm_latch=arm_latch, attempts=attempts)


def _safety_unavailable_blocked(
    ctx: Optional[Any], tool_name: str, error: str, *, arm_latch: bool = True, attempts: int = 2,
) -> Tuple[bool, str]:
    """Rate-limited safety lane: BLOCK this one call with a typed non-verdict outcome.

    A 429 is an infrastructure fact about the supervisor, not a verdict about the tool
    call — reporting it as SAFETY_VIOLATION told the agent its own command was unsafe,
    sending it hunting for a "safer" rewording of a benign command. The honest outcome
    keeps `full` mode's owner contract (an unchecked guarded call never executes; the
    existing fail-open cases stay exactly the ARCHITECTURE-documented no-backend three) while
    removing the false accusation: the ⚠️ *_UNAVAILABLE prefix classifies as a plain
    tool ERROR downstream, never as `safety_violation`, and the message itself carries
    the retry contract (P5: the instruction lives with the fact). Disclosed twice: the
    result the agent reads, and the durable event an owner can find later (BIBLE P3)."""
    if arm_latch and attempts >= 2:
        # The latch's documented trigger is a CONFIRMED storm: both attempts 429ed.
        # A deadline-expired single 429 is one data point, not a storm - later
        # checks in this process keep their own probe.
        global _SAFETY_STORM_UNTIL
        _SAFETY_STORM_UNTIL = time.time() + _SAFETY_STORM_COOLDOWN_SEC
    shape = (
        "storm latch, no attempt" if not arm_latch
        else "deadline expired before the retry" if attempts < 2
        else "after one retry"
    )
    action = (
        "blocked_unchecked_storm_latched" if not arm_latch
        else "blocked_unchecked_deadline_expired" if attempts < 2
        else "blocked_unchecked_after_retry"
    )
    log.error("Safety check rate-limited for %s (%s); blocking unchecked: %s", tool_name, shape, error)
    _emit_durable_safety_event(ctx, {
        "type": "safety_check_rate_limited", "tool": tool_name, "action": action, "error": error,
    })
    # The provider error goes on its own line: downstream status classification scans
    # only the FIRST line for outcome markers, and a raw error body could carry one.
    return False, (
        "⚠️ SAFETY_UNAVAILABLE: The Safety Supervisor is rate-limited and could not check "
        "this call. This is infrastructure back-pressure, NOT a verdict about your "
        "command — the call was not executed. Retry the same call shortly; do not "
        f"reword it as if it had been judged unsafe.\n(provider error: {error})"
    )


def _subject_too_large_blocked(
    ctx: Optional[Any], tool_name: str, subject_chars: int,
) -> Tuple[bool, str]:
    """Over-budget subject: BLOCK this one call with a typed policy denial.

    Truncating the subject instead would weaken the check — anything past the cut
    would run unreviewed — and `full` mode's owner contract is that an unchecked
    guarded call never executes. The `_BLOCKED` first-line marker classifies as a
    policy denial downstream (never `safety_violation`), and the message carries
    the working alternative (P5: the instruction lives with the fact). Disclosed
    twice: the result the agent reads, and the durable event an owner can find."""
    log.error(
        "Safety subject for %s is %d chars (budget %d); refusing the unreviewable call",
        tool_name, subject_chars, _SAFETY_SUBJECT_CHAR_BUDGET,
    )
    _emit_durable_safety_event(ctx, {
        "type": "safety_subject_too_large", "tool": tool_name,
        "subject_chars": int(subject_chars),
        "budget_chars": _SAFETY_SUBJECT_CHAR_BUDGET,
    })
    return False, (
        "⚠️ SAFETY_SUBJECT_TOO_LARGE_BLOCKED: this call's arguments serialize to "
        f"{subject_chars:,} characters, above the {_SAFETY_SUBJECT_CHAR_BUDGET:,}-character "
        "budget the Safety Supervisor can review in one prompt. The call was NOT executed "
        "and the subject was NOT truncated (a partially reviewed call must never run). "
        "Reshape it: split the payload into smaller calls that each fit the budget, so "
        "every byte that runs is a byte the supervisor saw."
    )


def _safety_model_call(
    *, client: Any, ctx: Optional[Any], tool_name: str, light_model: str,
    use_local: bool, call_type: str, user_prompt: str, on_usage: Any,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """ONE safety model call, with ONE bounded retry when the provider rate-limits it.

    Both wire shapes reach here: a RAISED provider error, and an HTTP-200 whose body
    carried it (`usage["provider_error"]`) — the production shape, which never raises,
    so an exception-only check would miss it. A non-rate-limit exception is re-raised
    unchanged so every existing lane stays byte-identical; two rate-limited attempts
    raise `_SafetyRateLimited`. A rate-limited attempt's usage is still accounted: the
    call was physical. The v6.40 per-model self-DoS slot is taken PER ATTEMPT (this is
    the highest-frequency LIGHT consumer — every tool call on every in-process subagent
    thread) with the backoff outside it.
    """
    from ouroboros import model_concurrency
    from ouroboros.llm_observability import chat_observed
    from ouroboros.tools.review_helpers import cached_prompt_blocks

    if time.time() < _SAFETY_STORM_UNTIL:
        raise _SafetyRateLimited(
            "safety lane cooling down after a recent rate-limit storm; no attempt made",
            latched=True,
        )
    reason = ""
    for attempt in range(2):
        if attempt:
            deadline = _safety_deadline_epoch(ctx)
            if deadline is not None and time.time() >= deadline:
                break  # the task deadline is spent: the retry would be a paid call past it
            _safety_rate_limit_backoff(ctx)
            if deadline is not None and time.time() >= deadline:
                break  # the backoff sleep itself consumed the deadline
        exc: Optional[Exception] = None
        msg: Dict[str, Any] = {}
        usage: Optional[Dict[str, Any]] = None
        try:
            with model_concurrency.model_call_slot(
                light_model, use_local, _safety_deadline_epoch(ctx)
            ):
                msg, usage = chat_observed(
                    client,
                    drive_root=pathlib.Path(getattr(ctx, "drive_root", "../data")) if ctx is not None else pathlib.Path("../data"),
                    task_id=str(getattr(ctx, "task_id", "") or "safety"),
                    call_type=call_type,
                    # This payload is TOOL-FREE and the send-time cache finalizer may only
                    # mark a tool schema, so the lane is cacheable only because the CALLER
                    # declares its own stable prefix, as every review surface does: the
                    # byte-stable SAFETY.md prompt is the whole prefix (the repair call
                    # reuses it and reads the cache the first attempt wrote) and the tool
                    # proposal stays in the unmarked user turn. Transport shape only — text,
                    # model slot, parsing and fail-closed semantics unchanged; a route that
                    # cannot carry markers has them stripped back to the identical single
                    # block. Disclosed residual: on OpenAI-compatible lanes the system
                    # content goes over the wire as a ONE-ELEMENT block list, so a strict
                    # self-hosted LIGHT endpoint rejecting array content surfaces as
                    # SAFETY_VIOLATION (this lane fails CLOSED there), never as a bypass.
                    messages=[
                        {"role": "system", "content": cached_prompt_blocks(
                            _get_safety_prompt(), ttl=_SAFETY_CACHE_TTL)},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=light_model, use_local=use_local,
                    max_tokens=get_safety_max_tokens(), reasoning_effort="low",
                    timeout=get_safety_call_timeout_sec(),
                    response_format={"type": "json_object"},
                )
        except Exception as e:
            exc = e
        safe_error = sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}") if exc else ""
        rate_limited = _safety_rate_limit_reason(exc, usage, safe_error)
        on_usage(usage)
        if rate_limited is None:
            if exc is not None:
                raise exc
            return msg, usage
        reason = rate_limited
        attempts = attempt + 1
        log.warning(
            "Safety check rate-limited for %s (attempt %d/2): %s", tool_name, attempts, reason,
        )
    raise _SafetyRateLimited(reason, attempts=attempts)


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

    args_json = _render_subject_json(arguments)
    if len(args_json) > _SAFETY_SUBJECT_CHAR_BUDGET:
        return _subject_too_large_blocked(ctx, tool_name, len(args_json))
    prompt = _build_check_prompt(tool_name, arguments, messages, args_json=args_json)
    client = LLMClient()
    light_model = get_light_model()
    log.info(f"Running safety check on {tool_name} using {light_model} (local={_use_local_light})")

    def _emit_safety_usage(usage_payload: Optional[Dict[str, Any]]) -> None:
        if not usage_payload:
            return
        # Use provider-canonical model identity for cost/events.
        resolved_model = str(usage_payload.get("resolved_model") or light_model)
        if _use_local_light:
            provider = "local"
            model_name = f"{light_model} (local)"
        else:
            provider = str(usage_payload.get("provider") or infer_provider_from_model(light_model))
            model_name = resolved_model
        raw_cost = usage_payload.get("cost")
        cost = None
        if raw_cost is not None and not isinstance(raw_cost, bool):
            try:
                cost = float(raw_cost)
            except (TypeError, ValueError):
                cost = None
            if cost is not None and not (math.isfinite(cost) and cost >= 0):
                cost = None
        if raw_cost is not None and cost is None:
            log.warning("safety usage carried an invalid reported cost (%s); treating as unknown",
                        type(raw_cost).__name__)
            usage_payload["cost"] = None
        if _use_local_light:
            cost = 0.0
        elif cost is None:
            cost = estimate_cost_optional(
                resolved_model,
                int(usage_payload.get("prompt_tokens") or 0),
                int(usage_payload.get("completion_tokens") or 0),
                cache_usage={
                    "cached_tokens": int(usage_payload.get("cached_tokens") or 0),
                    "cache_write_tokens": int(usage_payload.get("cache_write_tokens") or 0),
                },
                provider=provider,
            )
            usage_payload["cost"] = cost
        _eq = getattr(ctx, "event_queue", None) if ctx is not None else None
        if _eq is not None:
            emit_llm_usage_event(
                _eq,
                getattr(ctx, "task_id", "") if ctx is not None else "",
                model_name, usage_payload, cost,
                category="safety",
                provider=provider,
                source="safety_check",
            )
        else:
            update_budget_from_usage(usage_payload)

    try:
        msg, usage = _safety_model_call(
            client=client, ctx=ctx, tool_name=tool_name, light_model=light_model,
            use_local=_use_local_light, call_type="safety_supervisor",
            user_prompt=prompt, on_usage=_emit_safety_usage,
        )
    except _SafetyRateLimited as e:
        return _rate_limited_outcome(
            ctx, tool_name, str(e),
            local_fallback=_use_local_light and _is_local_fallback,
            arm_latch=not getattr(e, "latched", False),
            attempts=getattr(e, "attempts", 2),
        )
    except Exception as e:
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

    result = _parse_safety_response(msg.get("content") or "")
    if result is None:
        raw_content = str(msg.get("content") or "")
        failure_class = _classify_safety_parse_failure(msg, usage)
        log.warning(
            "Safety check returned invalid JSON for %s (class=%s); retrying once with repair prompt",
            tool_name, failure_class,
        )
        _emit_durable_safety_event(ctx, {
            "type": "safety_parse_retry",
            "tool": tool_name,
            "failure_class": failure_class,
        })
        try:
            repair_prompt = (
                "Your previous Safety Supervisor response was not parseable as the required JSON object.\n"
                "Return ONLY this strict JSON shape, with no markdown and no prose:\n"
                "{\"status\":\"SAFE|SUSPICIOUS|DANGEROUS\",\"reason\":\"short reason\"}\n\n"
                "Original proposed tool call follows again.\n\n"
                f"{prompt}"
            )
            repair_msg, repair_usage = _safety_model_call(
                client=client, ctx=ctx, tool_name=tool_name, light_model=light_model,
                use_local=_use_local_light, call_type="safety_supervisor_repair",
                user_prompt=repair_prompt, on_usage=_emit_safety_usage,
            )
            result = _parse_safety_response(repair_msg.get("content") or "")
            if result is None:
                failure_class = _classify_safety_parse_failure(repair_msg, repair_usage)
        except _SafetyRateLimited as e:
            return _rate_limited_outcome(
                ctx, tool_name, str(e),
                local_fallback=_use_local_light and _is_local_fallback,
                arm_latch=not getattr(e, "latched", False),
                attempts=getattr(e, "attempts", 2),
            )
        except Exception as exc:
            log.warning("Safety repair retry failed for %s: %s", tool_name, exc, exc_info=True)
        if result is None:
            log.error(f"Safety check returned invalid JSON for {tool_name} (class={failure_class}): {raw_content}")
            _emit_durable_safety_event(ctx, {
                "type": "safety_parse_failed",
                "tool": tool_name,
                "failure_class": failure_class,
            })
            return False, (
                "⚠️ SAFETY_VIOLATION: Safety Supervisor returned unparseable response "
                f"(class={failure_class}) after one repair retry."
            )

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


def _emit_durable_safety_event(ctx: Optional[Any], event: Dict[str, Any]) -> None:
    """Emit a safety audit event durably AT THE MOMENT of the decision (v6.54.3).

    The canonical durable pattern (mirrors control._emit_swarm_fanout): a DIRECT
    append into the drive's events.jsonl — the per-worker log sink forwards
    appended lines to the live dashboard, so no separate queue put is needed and
    a queued-but-undrained event can never be lost to a worker death (review
    round 9, P3). The event_queue is only the last resort for contexts with no
    drive_logs at all."""
    payload = {
        "task_id": str(getattr(ctx, "task_id", "") or "") if ctx is not None else "",
        "ts": utc_now_iso(),
        **event,
    }
    try:
        drive_logs = getattr(ctx, "drive_logs", None) if ctx is not None else None
        if callable(drive_logs):
            from ouroboros.utils import append_jsonl

            if append_jsonl(drive_logs() / "events.jsonl", payload):
                return
            # append_jsonl answers whether the row landed (its documented contract for
            # important events); a False here means the durable half of the disclosure
            # is missing, so fall through to the queue rather than returning silently.
            log.error(
                "durable safety event %s not persisted; falling back to event queue",
                event.get("type") or "?",
            )
        eq = getattr(ctx, "event_queue", None) if ctx is not None else None
        if eq is not None:
            eq.put_nowait(payload)
    except Exception:
        # The durable audit is a P3 invariant for waved-through safety decisions —
        # a lost emit must not be TRULY silent. Never fatal (a failed audit write
        # must not break a task), but escalate to error so the loss is itself
        # traced (adversarial review r1 #21).
        log.error("failed to emit durable safety event %s", event.get("type") or "?", exc_info=True)


def _emit_safety_mode_skip(ctx: Optional[Any], tool_name: str, mode: str, policy: str) -> None:
    """Durable audit for a safety LLM check waved through by a non-full owner mode.

    P3: an advisory/off mode is legitimate ONLY while every decision it waves
    through leaves a loud, durable trace at the moment it happens (review round 1)."""
    log.warning("Safety mode=%s waved through LLM check for %s (policy=%s)", mode, tool_name, policy)
    _emit_durable_safety_event(ctx, {
        "type": "safety_mode_skip",
        "tool": tool_name,
        "safety_mode": mode,
        "policy": policy,
    })


# Public entry point.

def check_safety(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[Any] = None,
    python_resolution: Optional[Any] = None,
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
        elif tool_name == "verify_and_record":
            # A LIST `check` is argv (no shell), so it is safe-subject eligible like
            # run_command. A STRING `check` runs through `sh -c` (shell interpretation),
            # so a safe-looking first word could hide a compound command (`cat x; rm -rf`)
            # — force the full LLM review for string checks (no safe-subject bypass).
            check = arguments.get("check")
            raw_cmd = check if isinstance(check, (list, tuple)) else None
        if _normalize_safe_shell_subject(raw_cmd) or _normalize_resolved_python_subject(
            raw_cmd, python_resolution
        ):
            # Whitelist-safe subject: FULL mode allows it without any LLM call,
            # so non-full modes wave nothing through here — no skip-audit event
            # (adversarial review r1 #19: audit only real deltas vs full mode).
            return True, ""

    # Owner-selected LLM-safety coverage (full | light | off). This gates ONLY the
    # LLM supervisor layer — the deterministic registry sandbox, protected-path
    # policy, and light-mode write guards run in every mode (BIBLE P3: the LLM
    # supervisor is a configurable layer, not the immune floor). Non-full modes
    # emit a durable audit event so a waved-through call is never silent.
    safety_mode = get_safety_mode()
    if safety_mode != "full":
        skip_llm = safety_mode == "off" or (safety_mode == "light" and policy == POLICY_CHECK_CONDITIONAL)
        if skip_llm:
            _emit_safety_mode_skip(ctx, tool_name, safety_mode, policy)
            return True, ""

    return _run_llm_check(tool_name, arguments, messages, ctx)
