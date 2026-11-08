"""
Policy-based safety check for tool calls.

Built-ins carry an explicit policy (a ``TOOL_POLICY`` entry, else their own
``ToolEntry`` declaration); unknown tools default to one light-model check.
Structured resource admission precedes the assessment; review evidence and
actual execution remain independent of the Cyber advisory decision.
"""

import ast
import json
import logging
import pathlib
import hashlib
import threading
import re
import shlex
import time
from collections import OrderedDict
from typing import Tuple, Dict, Any, List, Optional

from ouroboros.config import get_safety_model, get_runtime_mode, get_safety_call_timeout_sec, get_safety_max_tokens, get_safety_mode
from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import classify_llm_exception, is_rate_limit_text
from ouroboros.pricing import emit_llm_usage_event, normalize_usage_cost, infer_provider_from_model
from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso
from ouroboros.config import runtime_setting
from ouroboros.safety_config import OWNER_RULES_KEY

from ouroboros.safety_records import (
    _safety_drive_root, _record_safety_usage, _emit_durable_safety_event, _emit_safety_mode_skip, coalesce_verdicts, verdict_history_bytes,
)
from ouroboros.consciousness_authority import work_kind, effective_runtime_mode

log = logging.getLogger(__name__)

# Policy constants.

POLICY_SKIP = "skip"
POLICY_CHECK = "check"
POLICY_CHECK_CONDITIONAL = "check_conditional"
POLICY_VALUES = frozenset({POLICY_SKIP, POLICY_CHECK, POLICY_CHECK_CONDITIONAL})

# A tool with neither a table entry nor a valid declaration (every extension
# and MCP tool among them) gets one cheap LLM recheck.
DEFAULT_POLICY = POLICY_CHECK

# Protected per-tool policy. It outranks a built-in's own declaration
# (``ToolEntry.safety_policy``), so editing a tool module never lowers a listed
# tool. Each built-in has exactly one home, here or its declaration
# (invariant-tested).
TOOL_POLICY: Dict[str, str] = {
    "inspect_harness": POLICY_SKIP,
    "inspect_subagent": POLICY_SKIP,
    "maintain_harness": POLICY_SKIP,
    "read_file": POLICY_SKIP,
    "list_files": POLICY_SKIP,
    "chat_history": POLICY_SKIP,
    "recent_tasks": POLICY_SKIP,
    "live_roots": POLICY_SKIP,
    "knowledge_read": POLICY_SKIP,
    "memory_read": POLICY_SKIP,
    "knowledge_list": POLICY_SKIP,
    "journal_read": POLICY_SKIP,
    "workpad_read": POLICY_SKIP,
    "web_search": POLICY_SKIP,
    "codebase_health": POLICY_SKIP,
    "search_code": POLICY_SKIP,
    "query_code": POLICY_SKIP,
    "query_data": POLICY_SKIP,
    "list_available_tools": POLICY_SKIP,
    "memory_map": POLICY_SKIP,
    "analyze_screenshot": POLICY_SKIP,
    "vlm_query": POLICY_SKIP,
    "generate_image": POLICY_SKIP,
    "list_image_models": POLICY_SKIP,
    "recover_image": POLICY_SKIP,
    "view_image": POLICY_SKIP,
    "ocr_pdf": POLICY_SKIP,
    "youtube_transcript": POLICY_SKIP,
    "extract_video_frames": POLICY_SKIP,
    "browse_page": POLICY_SKIP,
    "browser_action": POLICY_SKIP,
    "github_read": POLICY_SKIP,
    "plan_task": POLICY_SKIP,
    "finish_task": POLICY_SKIP,
    "review_status": POLICY_SKIP,
    "get_task_result": POLICY_SKIP,
    "peek_task": POLICY_SKIP,
    "wait_task": POLICY_SKIP,
    "wait_tasks": POLICY_SKIP,
    "await_messages": POLICY_SKIP,
    "continue_in_background": POLICY_SKIP,
    "pause_task": POLICY_CHECK, "resume_task": POLICY_CHECK, "continue_task": POLICY_CHECK,
    "list_projects": POLICY_SKIP,
    "switch_model": POLICY_SKIP,
    "service_status": POLICY_SKIP,
    "service_logs": POLICY_SKIP,

    "write_file": POLICY_SKIP,
    "edit_text": POLICY_SKIP,
    "apply_patch": POLICY_SKIP,
    "edit_batch": POLICY_SKIP,
    "commit_reviewed": POLICY_SKIP,
    "vcs_commit_reviewed": POLICY_SKIP,
    "knowledge_write": POLICY_SKIP,
    "chronicle_write": POLICY_SKIP,
    "memory_mark": POLICY_SKIP,
    "journal_write": POLICY_SKIP,
    "workpad_write": POLICY_SKIP,
    "update_focus": POLICY_SKIP,
    "tree_note": POLICY_SKIP,
    "tree_read": POLICY_SKIP,
    "start_task": POLICY_CHECK,
    "move_to_project": POLICY_SKIP,
    "tell_task": POLICY_SKIP,  # a relay between my own works: the send it leads to is checked where it happens
    "update_scratchpad": POLICY_SKIP,
    "update_identity": POLICY_SKIP,
    "update_world": POLICY_SKIP,
    "memory_update_registry": POLICY_SKIP,
    "vcs_pull_ff": POLICY_SKIP,
    "vcs_restore": POLICY_SKIP,
    "vcs_revert": POLICY_SKIP,
    "vcs_rollback": POLICY_SKIP,

    "schedule_subagent": POLICY_SKIP,
    "manage_schedules": POLICY_SKIP,
    "schedule_followup": POLICY_CHECK,
    "delegate_start": POLICY_SKIP,
    "delegate_wait": POLICY_SKIP,
    "delegate_cancel": POLICY_SKIP,
    "delegate_answer": POLICY_SKIP,
    "delegate_message": POLICY_SKIP,
    "cancel_task": POLICY_SKIP,
    "set_task_retention": POLICY_SKIP,  # a retention fact on the owner's own held words; never deletes
    "resume_child_task": POLICY_SKIP,
    "discard_child_result": POLICY_SKIP,
    "override_delegation_constraint": POLICY_SKIP,
    "request_restart": POLICY_SKIP,
    "prepare_self_change": POLICY_SKIP,
    "request_deep_self_review": POLICY_SKIP,
    "set_tool_timeout": POLICY_SKIP,
    "toggle_evolution": POLICY_SKIP,
    "toggle_consciousness": POLICY_CHECK,
    "promote_to_stable": POLICY_SKIP,
    "send_user_message": POLICY_SKIP,
    "send_photo": POLICY_SKIP,
    "send_video": POLICY_SKIP,
    "send_file": POLICY_SKIP,
    "send_links": POLICY_SKIP,
    "presence_cancel_work": POLICY_SKIP,
    "configure_presence": POLICY_CHECK,
    "escalate": POLICY_SKIP,
    "forward_to_worker": POLICY_SKIP,  # a relay, like tell_task
    "compact_context": POLICY_SKIP,
    "enable_tools": POLICY_SKIP,
    "preflight_review": POLICY_SKIP,
    "advisory_review": POLICY_SKIP,  # compat alias of preflight_review
    "review_change": POLICY_SKIP,
    "start_service": POLICY_CHECK_CONDITIONAL,
    "stop_service": POLICY_SKIP,

    "list_skills": POLICY_SKIP,
    "skill_review": POLICY_SKIP,
    "toggle_skill": POLICY_SKIP,
    "skill_exec": POLICY_CHECK,
    "skill_owner_action": POLICY_CHECK,
    "skill_preflight": POLICY_SKIP,

    "run_command": POLICY_CHECK_CONDITIONAL,
    "run_script": POLICY_CHECK_CONDITIONAL,
    "verify_and_record": POLICY_CHECK_CONDITIONAL,

    "compare_subagent_patches": POLICY_SKIP,

    "integrate_subagent_patch": POLICY_CHECK,
    "integrate_delegated_patch": POLICY_CHECK,
    "fetch_pr_ref": POLICY_CHECK,
    "create_integration_branch": POLICY_CHECK,
    "cherry_pick_pr_commits": POLICY_CHECK,
    "stage_adaptations": POLICY_CHECK,
    "stage_pr_merge": POLICY_CHECK,
    "generate_evolution_stats": POLICY_CHECK,
    "submit_skill_to_hub": POLICY_CHECK,
    "github_write": POLICY_CHECK,
    "pr_merge": POLICY_CHECK,

    "set_next_wakeup": POLICY_SKIP,
    "place_reply": POLICY_CHECK,
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


# The complete proposed call is reviewed verbatim after redaction. An oversize
# subject is refused with a reshape/split alternative, never truncated.
_SAFETY_SUBJECT_CHAR_BUDGET = 250_000


def _render_subject_json(arguments: Dict[str, Any]) -> str:
    'Serialize the redacted call arguments exactly as the check prompt embeds them.'
    safe_args = _redact_secrets_in_arguments(arguments or {})
    try:
        rendered = json.dumps(safe_args, indent=2, default=repr, ensure_ascii=False)
    except Exception:
        rendered = repr(safe_args)
    # A lone surrogate (a validly PARSED \ud800 escape in tool args) survives
    # json.dumps but is unencodable UTF-8 and would explode the provider send
    # downstream; the prompt is a rendering, so substitute rather than crash.
    return rendered.encode("utf-8", "replace").decode("utf-8")


# Mandatory evidence and decisive history must fit; omitted SAFE rows are counted.
_SAFETY_BUNDLE_BYTE_BUDGET = 512_000
_SAFETY_PRIOR_BYTE_BUDGET = 64_000
_PRIOR_VERDICTS: OrderedDict[str, dict[str, dict]] = OrderedDict()
_PRIOR_LOCK = threading.RLock()


class _VerdictHistory(dict):
    """Retained decisive rows plus an explicit count of omitted SAFE verdicts."""
    omitted_safe = 0


class _Assessment(tuple):
    """Keep the public pair ABI while distinguishing flags from infrastructure holds."""
    def __new__(cls, allowed, message, *, status="", verdicts=(), destination=""):
        result = super().__new__(cls, (allowed, message))
        result.status, result.verdicts, result.destination = status, tuple(verdicts), destination
        return result


def _root_id(ctx):
    metadata = getattr(ctx, "task_metadata", None) or {}
    return str(metadata.get("root_task_id") or getattr(ctx, "root_task_id", "") or getattr(ctx, "task_id", "") or "")


def clear_safety_work(root_task_id: str) -> None:
    """Root terminal hook: audit persists, retry authority does not."""
    with _PRIOR_LOCK:
        _PRIOR_VERDICTS.pop(str(root_task_id), None)


def _subject_digest(tool_name, arguments):
    # Hash ORIGINAL arguments, not redacted rendering: changed credentials are a new subject too.
    raw = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


# A flag is overridden by a decision, never by bytes re-presented: the re-issue must come from a later model call.
_SEND_DESPITE = ("To send despite this flag, repeat the identical call, request_id included, in a later step; "
                 "any changed argument is checked again.")


def _authoring_call(ctx):
    """The model call that authored this tool call (the loop's ``_current_llm_call_meta``); "" when unknown."""
    meta = getattr(ctx, "_current_llm_call_meta", None)
    return str(meta.get("llm_call_id") or "") if isinstance(meta, dict) else ""


def _remember(ctx, digest, tool_name, assessment, *, speech_refused=False, held=False):
    """Retain a verdict, or a hold of words (when and by which model call: the owner's exit reads it)."""
    root = _root_id(ctx)
    if not root:
        return True  # rootless calls never earn a reusable speech flag
    with _PRIOR_LOCK:
        previous = _PRIOR_VERDICTS.get(root, {})
        before = previous.get(digest, {})
        row = {"digest": digest, "tool": tool_name, "speech_refused": speech_refused,
               "verdicts": coalesce_verdicts(before.get("verdicts", ()), assessment.verdicts),
               **({"llm_call_id": _authoring_call(ctx)} if speech_refused or held else {}),
               # the first hold is when the mind was told to ask; a retry held again keeps that time
               **({"held_at": before.get("held_at") or utc_now_iso(), "hold": getattr(assessment, "status", "")}
                  if held else {})}
        history = _VerdictHistory({**previous, digest: row})
        history.omitted_safe = getattr(previous, "omitted_safe", 0)
        safe_rows = iter([key for key, value in history.items()
                         if not value.get("speech_refused") and value.get("verdicts")
                         and all(verdict["status"] == "SAFE" for verdict in value["verdicts"])])
        while verdict_history_bytes(history) > _SAFETY_PRIOR_BYTE_BUDGET:
            key = next(safe_rows, None)
            if key is None:
                return False  # decisive flags must fit in full
            history.omitted_safe += sum(1 + verdict.get("repeat_count", 0) for verdict in history.pop(key)["verdicts"])
        if root not in _PRIOR_VERDICTS and len(_PRIOR_VERDICTS) >= 512:
            _PRIOR_VERDICTS.popitem(last=False)  # an evicted root earns no speech reissue
        _PRIOR_VERDICTS[root] = history
        _PRIOR_VERDICTS.move_to_end(root)
    return True


def _runtime_mode(ctx):
    return effective_runtime_mode(get_runtime_mode(), getattr(ctx, "task_metadata", None))


def _needs_checker(ctx):
    return bool(str(runtime_setting("OUROBOROS_MODEL_SAFETY", "") or "").strip()) or (
        work_kind(getattr(ctx, "task_metadata", None)) == "consciousness"
        and get_safety_mode("consciousness") == "full")


def _hold(ctx, tool_name, reason, *, status="UNAVAILABLE"):
    _emit_durable_safety_event(ctx, {"type": "safety_hold", "tool": tool_name, "reason": reason, "hold": status})
    marker = "SAFETY_SUBJECT_TOO_LARGE_BLOCKED" if status == "TOO_LARGE" else "SAFETY_UNAVAILABLE"
    return _Assessment(False, f"⚠️ {marker}: {reason}. This call was not checked: that is no verdict. Retry later.",
                       status=status)


# Words no check could judge wait in every mode; the exits are named, never a route around the hold.
_HELD_WORDS = ("These words wait until a check judges them. Or ask your human privately, quoting the exact text and "
               "destination: their answer on that card lets an identical re-issue from a later step send it unjudged. "
               "Never route these words through another tool.")
_HELD_CALL = ("This call waits until a check judges it: its server does not mark it read-only, so it may deliver "
              "words. Retry it in a later step, or ask your human privately with this exact call on the card: {call}. "
              "Their answer lets an identical re-issue from a later step run it unjudged. Never route it through "
              "another tool.")


def _exact_call(tool_name, arguments):
    """An MCP call as my human's card quotes it: it declares no words or destination of its own, so the whole call."""
    return f"{tool_name} {json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


def _owner_yes(ctx, tool_name, arguments, effect, prior):
    """The quiz my human answered after these words were held, its card holding their exact text and every
    destination (an MCP call: the exact call, ``_exact_call``); "" if none. The host checks facts only (bytes,
    destination, time): the mind's re-issue after reading the answer is its judgment that the answer approves."""
    from datetime import datetime
    from ouroboros.place_guard import call_place_facts
    from ouroboros.tool_effects import argument_value

    def when(stamp):
        try:
            return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        except ValueError:
            return None
    held = when(prior.get("held_at"))
    if effect.source == "mcp":  # no declared words or destination: the card holds the exact call
        words, destinations = _exact_call(tool_name, arguments), [tool_name]
    else:
        words = argument_value(arguments, effect.text or "text")
        facts = call_place_facts(ctx, tool_name, arguments) or {}
        destinations = ([row["place_label"] for row in facts.get("places") or [] if row.get("place_label")]
                        or [value for values in (facts.get("targets") or {}).values() for value in values])
    if not isinstance(words, str) or not words.strip() or not destinations or held is None:
        return ""
    for row in getattr(ctx, "_owner_directives", None) or []:
        card = str(((row.get("reply_to") if isinstance(row, dict) else None) or {}).get("text") or "")
        answered = when(row.get("ts") or row.get("received")) if card else None
        try:
            later = answered is not None and answered > held
        except TypeError:  # an answer time without an offset is not comparable: not a fact this exit can use
            later = False
        if row.get("source") == "owner_quiz_answer" and later and words in card and all(d in card for d in destinations):
            return str(row.get("quiz_id") or row.get("msg_id") or "")
    return ""


def _build_check_prompt(tool_name, arguments, messages=None, *, args_json=None, ctx=None, resolved_binding=None,
                        evidence=None):
    """Fixed host evidence, with authorship separate from semantic authorization. ``evidence`` (a dict) receives
    the destinations and notes resolved here, so a refusal names them without resolving them again."""
    from dataclasses import asdict, is_dataclass
    from ouroboros import owner_words
    from ouroboros.place_guard import call_place_facts
    from ouroboros.places import load_places
    from ouroboros.tool_access import canonical_data_root
    from ouroboros.tool_effects import mcp_sends_as
    from ouroboros.safety_evidence import focus_event_source

    metadata = getattr(ctx, "task_metadata", None) or {}
    kind = work_kind(metadata)
    focus = metadata.get("focus") or {}
    place = dict(call_place_facts(ctx, tool_name, arguments) or {})
    place_ids = {row["place_id"] for row in place.get("places", []) if row.get("place_id")}
    # A focus occasions the work; it does not establish where this call lands.
    # Only these two built-ins take an actual place reference as their subject.
    referenced = arguments.get("place_id") if tool_name in {"configure_presence", "place_reply"} else None
    if referenced and referenced not in place_ids:  # a resolved row already names its place
        place["place_id"] = referenced
        place_ids.add(referenced)
    notes = []  # each place's notes once, read now from the canonical store
    if place_ids:
        for record in load_places(canonical_data_root(ctx)):
            if record.place_id not in place_ids:
                continue
            if record.place_id == place.get("place_id"):
                place.update(place_label=record.label, place_state=record.state)
            notes.extend({"place_id": record.place_id, "text": n.text, "author": n.author,
                          **({"at": n.at} if n.at else {}),
                          "standing_delegation": n.author in {"owner", "migration"}} for n in record.notes)
    if tool_name == "place_reply":  # always a message; resolved rows already carry its route
        place["effect"] = "message"
        if not place.get("targets"):
            place["targets"] = {"destination": arguments.get("destination")}
    occasion = "unknown"
    if metadata.get("schedule_writer_kind"):
        occasion = metadata["schedule_writer_kind"] + "-written schedule"
    elif focus:  # read at this check: a confirmation made after the admission ``focus`` records counts
        from ouroboros.presence_context import focus_owner_now
        occasion = "confirmed owner account" if focus_owner_now(ctx) else "person in a place"
    elif metadata.get("wake_cause") == "own_clock":
        occasion = "own clock"
    elif kind == "owner":
        occasion = "owner door"
    directives = getattr(ctx, "_owner_directives", None)
    # User messages are quoted input when the caller has no retained host source; never certify them as owner words.
    sources = [*directives, *owner_words.safety_sources(ctx)] if isinstance(directives, list) else [m for m in (messages or []) if m.get("role") == "user"]
    sources = [focus_event_source(row, metadata, notes) for row in sources]  # the evidence diet: what a row costs
    target = asdict(resolved_binding) if is_dataclass(resolved_binding) else resolved_binding
    # Whose credential an MCP call acts with (e.g. a server signed in as my human); connectors carry it per place.
    sends_as = {"sends_as": mcp} if tool_name.startswith("mcp_") and (mcp := mcp_sends_as(tool_name)) else {}
    with _PRIOR_LOCK:
        root = _root_id(ctx)
        history = _PRIOR_VERDICTS.get(root, {})
        if root in _PRIOR_VERDICTS:
            _PRIOR_VERDICTS.move_to_end(root)
        earlier = list(history.values())
        omitted_safe = getattr(history, "omitted_safe", 0)
    bundle = {
        "work": {"kind": kind, "label": "your work" if kind == "owner" else "consciousness",
                 "root_task_id": _root_id(ctx), "task_id": getattr(ctx, "task_id", ""), "runtime_mode": _runtime_mode(ctx),
                 "consciousness_autonomy": metadata.get("consciousness_autonomy")},
        "occasion": {"cause": occasion, "consciousness_cause": metadata.get("consciousness_cause"),
                     "origin_wake_task_id": metadata.get("origin_wake_task_id"), "focus": focus,
                     "sources": sources, "retained_host_sources": isinstance(directives, list)},
        "subject": {"tool": tool_name, "physical_target": target, "place": place, **sends_as},
        # The acting mind's context reads the same key through the same snapshot (``owner_rules_section``).
        "standing_delegation": {"owner_rules": runtime_setting(OWNER_RULES_KEY, ""), "place_notes": notes},
        "earlier_verdicts": earlier,
        "earlier_safe_verdicts_omitted": omitted_safe,
    }
    if isinstance(evidence, dict):
        evidence.update(place=place, notes=notes)
    return (f"Runtime mode: {_runtime_mode(ctx)}\nHost evidence (provenance is not consent):\n"
            + _render_subject_json(bundle) + "\nProposed tool call:\nTool: " + tool_name
            + "\nArguments:\n" + (args_json if args_json is not None else _render_subject_json(arguments))
            + "\nIs this safe?")


def _parse_safety_response(text: str) -> Optional[Dict[str, Any]]:
    """Parse the ``{"status", "reason", "speech", "quote"}`` verdict from a Safety model response.

    ``speech`` is kept only as a boolean and ``quote`` only as non-empty text; both are the checker's claims.
    Models occasionally wrap the JSON in prose or code fences despite the prompt.
    The whole reply and every outermost ``{...}`` in it are candidates; the
    strictest recognized status wins (DANGEROUS > SUSPICIOUS > SAFE, first of
    equals), so a quoted SAFE cannot outrank a direct flag. Only when no candidate
    carries a verdict is the whole reply read as ``{"answer": "<verdict JSON>"}``
    with no other key: that string is decoded once, never searched. Otherwise
    None: the caller fails closed.
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

    def verdict_of(raw: str) -> Optional[Dict[str, Any]]:
        obj = json.loads(raw)
        status = str(obj.get("status") or "").upper() if isinstance(obj, dict) else ""
        if status not in rank:
            return None
        quote = obj.get("quote")
        return {"status": status, "reason": str(obj.get("reason") or ""),
                **({"speech": obj["speech"]} if isinstance(obj.get("speech"), bool) else {}),
                **({"quote": quote} if isinstance(quote, str) and quote.strip() else {})}

    for candidate in candidates:
        try:
            found = verdict_of(candidate)
        except json.JSONDecodeError:
            continue
        if found and (best is None or rank[found["status"]] > rank[best["status"]]):
            best = found
    if best is None:
        try:
            envelope = json.loads(clean)
            if isinstance(envelope, dict) and list(envelope) == ["answer"] and isinstance(envelope["answer"], str):
                best = verdict_of(envelope["answer"])
        except (ValueError, RecursionError):  # a degenerate payload stays unparseable, as before
            pass
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


_LOCAL_ROUTING_KEYS = (
    "USE_LOCAL_MAIN",
    "USE_LOCAL_HEAVY",
    "USE_LOCAL_LIGHT",
    "USE_LOCAL_CONSCIOUSNESS",
    "USE_LOCAL_FALLBACK",
)


def _any_remote_provider_configured() -> bool:
    from ouroboros.provider_models import any_api_connection_configured

    return any_api_connection_configured()


def _any_local_routing_enabled() -> bool:
    return any(
        str(runtime_setting(k, "") or "").lower() in ("true", "1")
        for k in _LOCAL_ROUTING_KEYS
    )


def _light_model_has_reachable_provider(light_model: str) -> bool:
    """Return whether the light model's connection is reachable (its key, or a keyless endpoint)."""
    from ouroboros.provider_models import PROVIDER_KINDS, model_has_credentials, parse_model_route, provider_has_credentials

    if parse_model_route(light_model).connection:
        return model_has_credentials(light_model)
    try:
        from ouroboros.pricing import infer_api_key_type
        key_type = infer_api_key_type(light_model)
    except Exception:  # pragma: no cover — defensive
        return True  # don't over-block on classifier failure
    return provider_has_credentials(key_type) if key_type in PROVIDER_KINDS else True


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
    """Choose local/remote safety backend; unreachable fallback fails open.

    Only no-backend cases fail open with ``SAFETY_WARNING``: no local lane and either no provider key or one not
    covering the light model's provider; a fallback local runtime that fails or is rate-limited, never a chain route
    (``_unanswered``). An explicit local runtime (``USE_LOCAL_LIGHT``) that fails holds."""
    selected = get_safety_model()
    explicit = bool(str(runtime_setting("OUROBOROS_MODEL_SAFETY", "") or "").strip())
    local_light = str(runtime_setting("USE_LOCAL_LIGHT", "") or "").lower() in ("true", "1")
    if selected.endswith(" (local)") or (not explicit and local_light):
        inherited_main = not explicit and not local_light and not str(runtime_setting("OUROBOROS_MODEL_LIGHT", "") or "").strip()
        return True, inherited_main, None

    light_model = get_safety_model().removesuffix(" (local)")
    from ouroboros.provider_models import provider_for_model

    if provider_for_model(light_model) == "claudexor":
        # Subscription credentials and readiness belong to the model transport.
        return False, False, None

    if _any_remote_provider_configured():
        # The direct light-model provider needs its own key.
        if _light_model_has_reachable_provider(light_model):
            return False, False, None
        if not explicit and _any_local_routing_enabled():
            # Provider mismatch: local is fallback, so local outage is tolerated.
            return True, True, None
        return False, False, (
            f"Safety Supervisor model provider key missing for {light_model} "
            f"(other remote keys are set but they don't cover this provider); "
            "skipping check."
        )

    if not explicit and _any_local_routing_enabled():
        # Local-only configs should warn, not hard-block, on local outage.
        return True, True, None

    return False, False, (
        "No safety LLM available (neither remote provider keys nor local "
        "routing are configured); skipping check."
    )


_UNCHECKED_WARNING_SUFFIX = (
    "The tool call was allowed so the agent is not hard-blocked on a misconfigured "
    "runtime. The selected task and physical resource binding still describe the operation."
)

# Only a genuine THROUGHPUT signal may wave a safety check through, so the predicate is
# narrower than "retryable". `classify_llm_exception` folds every transient class —
# 408, 5xx and timeouts included — into `provider_transient`, and those are outages, not
# throttling: no retry, this route answered nothing. A `provider_transient`
# therefore qualifies only when the provider actually said 429, by status code or by the
# shared rate-limit text vocabulary (`is_rate_limit_text` — imported rather than copied,
# so the two lanes cannot fork). Permanent classes are already excluded upstream: a
# structured `insufficient_quota` wins over a 429 status inside that helper: no retry
# either. On the HTTP-200 body lane (`llm._normalize_remote_response`) the qualifying
# kind is exactly `rate_limit`, which that module assigns only to a transient body error
# whose code IS 429; its other body transients are plain outages (no retry; the chain is asked next).
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
# safety check in this process skips the Supervisor's own route for this window WITHOUT a
# provider call: to the owner's Fallback chain, else the same blocked SAFETY_UNAVAILABLE
# answer (the mark/is-cooling shape of ouroboros/fallback_cooldown.py, kept apart from it so a
# checker storm never reroutes Main; a chain route keeps its own window). The safety lane is the
# highest-frequency LIGHT consumer — every guarded tool call on every in-process subagent
# thread — so re-probing a storming route on each call would amplify the very storm it reports.
_SAFETY_STORM_COOLDOWN_SEC = 30.0
_SAFETY_STORM_UNTIL = 0.0
_CHAIN_STORMS: Dict[tuple, float] = {}  # a Fallback route's own storm window, by its complete binding


class _SafetyRateLimited(Exception):
    """Both safety attempts hit a provider rate limit: the route answered nothing (the chain is tried next, else
    the caller BLOCKS this one call with the typed SAFETY_UNAVAILABLE outcome; infrastructure, not a verdict).
    ``latched`` marks the storm-window short-circuit shape: it carries no NEW storm
    evidence, so the outcome must not extend the window (else the advised retry would
    keep re-arming it forever). ``attempts`` is the PHYSICAL call count behind this
    outcome (0 latched, 1 deadline-expired-before-retry, 2 full), so the durable audit
    row can name what actually happened instead of always claiming a retry."""

    def __init__(self, message: str, *, latched: bool = False, attempts: int = 2) -> None:
        super().__init__(message)
        self.latched = latched
        self.attempts = 0 if latched else attempts


def _arm_storm_latch() -> None:
    global _SAFETY_STORM_UNTIL
    _SAFETY_STORM_UNTIL = time.time() + _SAFETY_STORM_COOLDOWN_SEC


def _mark_storm(error: BaseException, binding: tuple, *, own: bool, fail_open: bool) -> None:
    """Close a route for the latch window after a CONFIRMED storm (both attempts 429ed): the Supervisor's own
    route through the process latch (never on the fail-open local lane), a chain route through its own window.
    A deadline-expired single 429 is one data point, and a latched skip carries no new evidence."""
    if not isinstance(error, _SafetyRateLimited) or error.attempts < 2:
        return
    if not own:
        _CHAIN_STORMS[binding] = time.time() + _SAFETY_STORM_COOLDOWN_SEC
    elif not fail_open:
        _arm_storm_latch()


def _is_throughput_rate_limit(kind: str, status_code: Optional[int], safe_error: str) -> bool:
    """True only for provider THROTTLING, never for a plain outage (408/5xx/timeout).

    A KNOWN status code is the authority and the text is never consulted beside it: the
    short markers `429`, `rpm` and `tpm` are bounded only by letters and digits, so they
    still match the separated ids and hostnames real outages carry ("503 ... (request id:
    req-429-ab0)", "504 Gateway Timeout ... rpm-edge-proxy-3"). The text branch runs ONLY when
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
        return None  # structured insufficient-quota is PERMANENT: no retry on this route
    if not isinstance(body, dict) or str(body.get("kind") or "") != _SAFETY_BODY_RATE_LIMIT_KIND:
        return None
    return sanitize_tool_result_for_log(
        f"{body.get('kind')} (code={body.get('code')}): {body.get('message')}"
    )[:300]


def _safety_rate_limit_backoff(ctx: Optional[Any]) -> None:
    """One bounded backoff, capped by the task deadline. Called BETWEEN physical calls:
    the provider slot is held only inside a physical send, so this sleep parks no route."""
    delay = _SAFETY_RATE_LIMIT_BACKOFF_SEC
    deadline = _safety_deadline_epoch(ctx)
    if deadline is not None:
        delay = min(delay, max(0.0, float(deadline) - time.time()))
    if delay > 0:
        time.sleep(delay)


def _rate_limited_outcome(
    ctx: Optional[Any], tool_name: str, error: str, *, local_fallback: bool, attempts: int = 2,
) -> Tuple[bool, str]:
    'Terminal outcome of a rate-limited safety check, split by lane. The local-FALLBACK'
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
    return _safety_unavailable_blocked(ctx, tool_name, error, attempts=attempts)


def _safety_unavailable_blocked(
    ctx: Optional[Any], tool_name: str, error: str, *, attempts: int = 2,
) -> Tuple[bool, str]:
    'Rate-limited safety lane: BLOCK this one call with a typed non-verdict outcome (``_mark_storm`` closes the route).'
    shape = (
        "storm latch, no attempt" if not attempts
        else "deadline expired before the retry" if attempts < 2
        else "after one retry"
    )
    action = (
        "blocked_unchecked_storm_latched" if not attempts
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
    'Over-budget subject: BLOCK this one call with a typed policy denial.'
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
    model_role: str = "safety", effort: str = "low", account: Optional[str] = None,
    processing: Optional[str] = None, attempts_limit: int = 2, system_prompt: Optional[str] = None,
    latch: bool = True, wait_for_resources: Optional[bool] = None,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """ONE safety model call, with ONE bounded retry when the provider rate-limits it.

    ``latch`` consults the storm latch (the Supervisor's own route only); ``wait_for_resources=False`` returns a
    quota or sign-in refusal at once, so a configured route after this one can answer first."""
    from ouroboros import model_concurrency
    from ouroboros.llm_observability import chat_observed
    from ouroboros.tools.review_helpers import cached_prompt_blocks
    if model_role == "safety":
        from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option, resolve_processing_preference
        if account is None:
            account = str(model_role_option(MODEL_ACCOUNTS_KEY, "safety"))
        if processing is None:
            processing = resolve_processing_preference("safety")

    if latch and time.time() < _SAFETY_STORM_UNTIL:
        raise _SafetyRateLimited(
            "safety lane cooling down after a recent rate-limit storm; no attempt made",
            latched=True,
        )
    reason = ""
    for attempt in range(attempts_limit):
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
            deadline = _safety_deadline_epoch(ctx)
            # The physical sender takes the provider slot; this only carries the deadline. The same
            # task deadline that ends the slot wait and the retry also bounds the send (#572).
            with model_concurrency.model_call_scope(light_model, use_local, deadline):
                msg, usage = chat_observed(
                    client,
                    drive_root=_safety_drive_root(ctx),
                    task_id=str(getattr(ctx, "task_id", "") or "safety"),
                    call_type=call_type,
                    # Stable, tool-free SAFETY.md prefix; subject remains uncached.
                    messages=[
                        {"role": "system", "content": cached_prompt_blocks(
                            (system_prompt if system_prompt is not None else _get_safety_prompt()), ttl=_SAFETY_CACHE_TTL)},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=light_model, use_local=use_local,
                    max_tokens=get_safety_max_tokens(), reasoning_effort=effort,
                    timeout=get_safety_call_timeout_sec(),
                    model_role=model_role, model_account_override=account, processing_preference=processing,
                    wait_for_resources=(not mode_has_unrestricted_agency(_runtime_mode(ctx))
                                        if wait_for_resources is None else wait_for_resources),
                    response_format=({"type": "json_object"}
                                     if LLMClient.supports_response_format(light_model, use_local=use_local)
                                     else None),
                    **({"caller_dispatch_deadline_ts": deadline} if deadline is not None else {}),
                )
        except Exception as e:
            from ouroboros.llm_claudexor import propagate_model_error
            propagate_model_error(e)
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


def _safety_routes(model: str, use_local: bool, overrides: Optional[dict] = None) -> List[Tuple[str, bool, str, tuple]]:
    """The Supervisor's route, then the owner's resilience chain (``OUROBOROS_MODEL_FALLBACKS`` on the
    ``USE_LOCAL_FALLBACK`` lane, as the main loop walks it), each complete binding once: the effective model and
    locality a live wait-card switch gives the role (what is sent, recorded and charged), its role, its binding."""
    from ouroboros.model_slots import route_binding
    from ouroboros.provider_models import fallback_candidate_targets

    chain_local = str(runtime_setting("USE_LOCAL_FALLBACK", "") or "").lower() in ("true", "1")
    rows = [(model, use_local, "safety")] + [(target.model_id, chain_local, f"fallback:{index}") for index, target
                                             in enumerate(fallback_candidate_targets(preserve_slots=True))]
    routes, seen = [], set()
    for row in rows:
        binding = route_binding(*row, overrides=overrides)
        if binding not in seen:
            seen.add(binding)
            routes.append((binding[0], binding[1], row[2], binding))
    return routes


def _answered_nothing(msg: Dict[str, Any], usage: Optional[Dict[str, Any]]) -> str:
    """A provider error body with no content, described; "" when the route answered."""
    body = usage.get("provider_error") if isinstance(usage, dict) else None
    if not isinstance(body, dict) or str((msg or {}).get("content") or "").strip():
        return ""
    return f"provider error {body.get('code') or ''} {body.get('type') or ''}: {body.get('message') or ''}"


def _unanswered(ctx: Optional[Any], tool_name: str, error: Exception, *, fail_open: bool):
    """The outcome of a route that answered nothing: the one-route rule. Only the Supervisor's own fallback
    local route may ``fail_open``; any storm is already marked (``_mark_storm``)."""
    if isinstance(error, _SafetyRateLimited):
        return _Assessment(*_rate_limited_outcome(
            ctx, tool_name, str(error), local_fallback=fail_open, attempts=error.attempts), status="UNAVAILABLE")
    from ouroboros.llm_claudexor import propagate_model_error
    propagate_model_error(error)
    safe_error = sanitize_tool_result_for_log(f"{type(error).__name__}: {error}")
    if fail_open:  # a fallback local outage warns instead of blocking all unknown tools
        log.warning("Safety local-fallback LLM call failed for %s (%s); proceeding with warning", tool_name, safe_error)
        return _Assessment(True, f"⚠️ SAFETY_WARNING: Local safety runtime unreachable ({safe_error}). {_UNCHECKED_WARNING_SUFFIX}", status="UNAVAILABLE")
    log.error("Safety check LLM call failed for %s: %s", tool_name, safe_error)
    return _hold(ctx, tool_name, f"Safety check failed with error: {safe_error}")


def _run_llm_check(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]],
    ctx: Optional[Any],
    resolved_binding: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Run a single light-model safety check and classify the verdict."""
    args_json = _render_subject_json(arguments)
    if len(args_json) > _SAFETY_SUBJECT_CHAR_BUDGET:
        return _Assessment(*_subject_too_large_blocked(ctx, tool_name, len(args_json)), status="TOO_LARGE")
    evidence = {}
    prompt = _build_check_prompt(tool_name, arguments, messages, args_json=args_json,
                                 ctx=ctx, resolved_binding=resolved_binding, evidence=evidence)
    if len(prompt.encode("utf-8")) > _SAFETY_BUNDLE_BYTE_BUDGET:
        return _hold(ctx, tool_name, "The complete mandatory evidence bundle is too large", status="TOO_LARGE")
    _use_local_light, _is_local_fallback, _skip_reason = _resolve_safety_routing()
    if _skip_reason is not None:
        if _needs_checker(ctx):
            return _hold(ctx, tool_name, _skip_reason)
        return _Assessment(True, f"⚠️ SAFETY_WARNING: Safety backend is not configured ({_skip_reason}). {_UNCHECKED_WARNING_SUFFIX}", status="UNAVAILABLE")
    client = LLMClient()
    system_prompt = _get_safety_prompt()
    light_model = get_safety_model().removesuffix(" (local)")
    log.info(f"Running safety check on {tool_name} using {light_model} (local={_use_local_light})")

    def _emit_safety_usage(usage_payload: Optional[Dict[str, Any]]) -> None:
        if not usage_payload:
            return
        cost, model_name, provider, _estimated = normalize_usage_cost(
            usage_payload, model=light_model, use_local=_use_local_light,
            provider=infer_provider_from_model(light_model),
        )
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
            _record_safety_usage(ctx, usage_payload)

    # A route that answers nothing (a throttle after its retry, an outage, a quota or sign-in refusal, an error
    # body) is an infrastructure fact, not a verdict: the check moves along the owner's chain. With no answer the
    # Supervisor's own route decides the outcome exactly as before, its owner question (a model wait) included.
    from ouroboros.llm_claudexor import propagate_model_error
    from ouroboros.model_wait import current_model_wait, model_wait_reason
    from ouroboros.transport_custody import outcome_unknown_on_chain

    fail_open = _use_local_light and _is_local_fallback and not _needs_checker(ctx)
    waiter = current_model_wait()
    routes = _safety_routes(light_model, _use_local_light, waiter.overrides if waiter else None)
    alone, skipped, own_failure, own_reply = len(routes) == 1, [], None, None

    def record_skips(answered: Optional[Dict[str, str]]) -> None:  # None: no route of the walk answered
        _emit_durable_safety_event(ctx, {"type": "safety_route_fallback", "tool": tool_name,
                                         "unavailable": skipped, "answered": answered})

    for index, (light_model, _use_local_light, role, binding) in enumerate(routes):
        try:
            if index and time.time() < _CHAIN_STORMS.get(binding, 0.0):
                raise _SafetyRateLimited("cooling down after this route's own rate-limit storm", latched=True)
            msg, usage = _safety_model_call(
                client=client, ctx=ctx, tool_name=tool_name, light_model=light_model, use_local=_use_local_light,
                call_type="safety_supervisor", user_prompt=prompt, on_usage=_emit_safety_usage,
                system_prompt=system_prompt, model_role=role, latch=not index,
                wait_for_resources=None if alone else False,
            )
            failure = "" if alone else _answered_nothing(msg, usage)
            own_reply = (msg, usage) if failure and not index else own_reply
        except Exception as error:
            if not isinstance(error, _SafetyRateLimited) and not model_wait_reason(error):
                stops = True  # an interruption or an unknown outcome never moves to another route
                try:
                    propagate_model_error(error)
                    stops = outcome_unknown_on_chain(error)  # a direct call that may have answered
                finally:
                    if stops and skipped:
                        record_skips(None)
                if stops:  # never a new verdict, nor the local route's warning for another route's call
                    return _unanswered(ctx, tool_name, error, fail_open=fail_open and not index)
            _mark_storm(error, binding, own=not index, fail_open=fail_open)  # later checks skip it in its window
            if alone:
                return _unanswered(ctx, tool_name, error, fail_open=fail_open)
            own_failure = own_failure if index else error
            failure = f"{type(error).__name__}: {error}"
        if not failure:
            if skipped:
                log.warning("Safety check for %s answered by %s after %s", tool_name, light_model, skipped)
                record_skips({"model": light_model, "role": role})
            break
        skipped.append({"model": light_model, "role": role, "error": sanitize_tool_result_for_log(failure)[:300]})
    else:
        record_skips(None)
        light_model, _use_local_light, role, binding = routes[0]
        if own_reply is not None:
            msg, usage = own_reply  # its error body takes the parse lane, as before
        elif not model_wait_reason(own_failure):
            return _unanswered(ctx, tool_name, own_failure, fail_open=fail_open)
        else:
            try:  # its quota or sign-in refusal: the owner question comes after the chain, as Main's does
                msg, usage = _safety_model_call(
                    client=client, ctx=ctx, tool_name=tool_name, light_model=light_model, use_local=_use_local_light,
                    call_type="safety_supervisor", user_prompt=prompt, on_usage=_emit_safety_usage,
                    system_prompt=system_prompt)
            except Exception as error:
                _mark_storm(error, binding, own=True, fail_open=fail_open)
                return _unanswered(ctx, tool_name, error, fail_open=fail_open)

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
                "{\"status\":\"SAFE|SUSPICIOUS|DANGEROUS\",\"reason\":\"short reason\",\"speech\":true|false,\"quote\":\"\"}\n\n"
                "Original proposed tool call follows again.\n\n"
                f"{prompt}"
            )
            repair_msg, repair_usage = _safety_model_call(
                client=client, ctx=ctx, tool_name=tool_name, light_model=light_model,
                use_local=_use_local_light, call_type="safety_supervisor_repair",
                user_prompt=repair_prompt, on_usage=_emit_safety_usage, system_prompt=system_prompt,
                model_role=role, latch=role == "safety",
            )
            result = _parse_safety_response(repair_msg.get("content") or "")
            if result is None:
                failure_class = _classify_safety_parse_failure(repair_msg, repair_usage)
        except _SafetyRateLimited as e:  # a chain route's storm closes that route; it never takes the local warning
            _mark_storm(e, binding, own=role == "safety", fail_open=fail_open)
            return _unanswered(ctx, tool_name, e, fail_open=fail_open and role == "safety")
        except Exception as exc:
            propagate_model_error(exc)
            log.warning("Safety repair retry failed for %s: %s", tool_name, exc, exc_info=True)
        if result is None:
            log.error(f"Safety check returned invalid JSON for {tool_name} (class={failure_class}): {raw_content}")
            _emit_durable_safety_event(ctx, {
                "type": "safety_parse_failed",
                "tool": tool_name,
                "failure_class": failure_class,
            })
            return _hold(ctx, tool_name, f"Safety Supervisor returned unparseable response (class={failure_class}) after one repair retry")

    verdicts = [{"stage": "first", **result}]
    row = None
    if result["status"] != "SAFE":
        from ouroboros.configured_subagents import normalize_configured_subagents
        try:
            catalog, _ = normalize_configured_subagents(runtime_setting("OUROBOROS_SUBAGENTS", "") or {"enabled": False, "items": []})
        except ValueError as exc:
            return _hold(ctx, tool_name, f"Second-check roster unavailable after a first-stage flag: {sanitize_tool_result_for_log(str(exc))}")
        row = next((r for r in catalog.items if r.rechecks_flags and r.enabled), None) if catalog.enabled else None
    if row is not None:
        second_prompt = ("You review a flag a first checker raised; overturning it needs a reason.\n"
                         + prompt + "\nFirst verdict:\n" + _render_subject_json(result))
        if len(second_prompt.encode("utf-8")) > _SAFETY_BUNDLE_BYTE_BUDGET:
            return _hold(ctx, tool_name, "The mandatory second-check bundle is too large", status="TOO_LARGE")
        light_model = row.route.target_id.removesuffix(" (local)")
        _use_local_light = row.route.target_id.endswith(" (local)")
        try:
            msg, _ = _safety_model_call(
                client=client, ctx=ctx, tool_name=tool_name, light_model=light_model, use_local=_use_local_light,
                call_type="safety_second_check", user_prompt=second_prompt, on_usage=_emit_safety_usage, system_prompt=system_prompt,
                model_role="subagent:" + row.subagent_id, effort=row.effort or "low",
                account=row.route.credential_profile_id, processing=row.processing_preference or None, attempts_limit=1,
                latch=False)
            second = _parse_safety_response(msg.get("content") or "")
        except Exception as exc:
            from ouroboros.llm_claudexor import propagate_model_error
            propagate_model_error(exc)
            return _hold(ctx, tool_name, f"Second check unavailable: {sanitize_tool_result_for_log(str(exc))}")
        if second is None or (second["status"] == "SAFE" and not str(second.get("reason") or "").strip()):
            return _hold(ctx, tool_name, "Second check did not return a verdict with a reason")
        result = second
        verdicts.append({"stage": "second", "row_id": row.subagent_id, **result})
    status = result["status"]
    reasons = "\n".join(f"{v['stage']}: {v['status']} — {v.get('reason', '')}" for v in verdicts)
    if status == "SAFE":
        return _Assessment(True, "", status=status, verdicts=verdicts)
    if status == "SUSPICIOUS":
        return _Assessment(True, "⚠️ SAFETY_WARNING: The Safety Supervisor flagged this action as suspicious.\n"
                           + reasons + "\nThe command was allowed, but consider whether this is the right approach.", status=status, verdicts=verdicts)
    return _Assessment(False, "⚠️ SAFETY_VIOLATION: The Safety Supervisor blocked this command.\n"
                       + reasons + "\nReconsider whether and how to proceed.", status=status, verdicts=verdicts,
                       destination="\n".join(_destination_lines(evidence)))


def _destination_lines(evidence):
    """The flagged call's destinations as the bundle resolved them: each label with my human's notes there."""
    from ouroboros.place_guard import destination_line

    place = evidence.get("place") or {}
    rows = {row["place_id"]: row for row in [*(place.get("places") or []), place] if row.get("place_id")}
    return [destination_line(row.get("place_label") or pid, [note["text"] for note in evidence.get("notes") or ()
                                                            if note["place_id"] == pid and note["standing_delegation"]])
            for pid, row in rows.items()]


def check_safety(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[Any] = None,
    python_resolution: Optional[Any] = None,
    resolved_binding: Optional[Any] = None,
    declared_policy: str = "",
) -> Tuple[bool, str]:
    """Return ``(allowed, warning_or_error)`` for one tool call.

    ``declared_policy`` is a first-party ``ToolEntry``'s own declaration; a
    ``TOOL_POLICY`` entry outranks it and an invalid value is ``DEFAULT_POLICY``.
    Speech, known before a verdict from declarations or marked by the verdict's ``speech``, follows the final
    assessment: DANGEROUS returns it unsent with the reason, the verdict's ``quote`` and the destination's notes, and
    only an identical re-issue (request_id included) from a later model call sends it; words only the verdict saw get
    that re-issue in Cyber Pro alone, and an ordinary mode checks each of their issues again. A hold on speech, or on
    an MCP call without a verdict, binds in every mode and names its exits; my human's answered card holding the exact
    words and destination (an MCP call: the exact call) lets a later identical re-issue send it (``_owner_yes``); a
    re-issue of words only the verdict saw meets the one-author guard first."""
    # Arguments can be None for no-parameter tool calls.
    tool_name = str(tool_name or "").strip()
    arguments = arguments or {}
    declared = declared_policy if isinstance(declared_policy, str) else ""
    policy = TOOL_POLICY.get(tool_name) or (declared if declared in POLICY_VALUES else DEFAULT_POLICY)

    from ouroboros.tool_effects import effect_of
    is_mcp = tool_name.startswith("mcp_")
    effect = effect_of(tool_name, ctx=ctx, is_mcp=is_mcp, args=arguments)
    # Speech the host knows before any verdict, from declarations only: a ``message`` effect, a connector-declared
    # send of a generic tool (``sends.when``), or the automatic reply. A verdict may add the mark (below), never
    # an argument name. An MCP call that is no declared read waits like speech when no verdict exists.
    speech = tool_name == "place_reply" or effect.effect == "message"
    if tool_name == "manage_schedules" and arguments.get("action") in {"create", "update", "restore"}:
        policy = POLICY_CHECK
    if effect.declared and effect.effect == "read":
        _emit_durable_safety_event(ctx, {"type": "safety_declared_read_skip", "tool": tool_name, "source": effect.source,
                                        "reason": "server annotation readOnlyHint" if effect.source == "mcp"
                                        else "reviewed tool declaration: read"})
        return True, ""
    if policy == POLICY_SKIP and not speech:
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

    cyber = mode_has_unrestricted_agency(_runtime_mode(ctx))
    digest = _subject_digest(tool_name, arguments)
    with _PRIOR_LOCK:
        root = _root_id(ctx)
        prior = _PRIOR_VERDICTS.get(root, {}).get(digest, {})
        if root in _PRIOR_VERDICTS:
            _PRIOR_VERDICTS.move_to_end(root)
    call = _authoring_call(ctx)
    if prior.get("speech_refused"):  # host- or verdict-marked speech, flagged before
        if not call or call == prior.get("llm_call_id"):  # the flagged decision re-presented, not a later one
            return False, "⚠️ SAFETY_VIOLATION: Message not sent: this exact call was flagged in this same step. " + _SEND_DESPITE
        if not speech:  # words only the verdict saw still meet the one-author guard a message meets at dispatch
            from ouroboros.place_guard import verdict_speech_refusal
            if refusal := verdict_speech_refusal(ctx, tool_name, arguments, is_mcp=is_mcp):
                return False, refusal
        _emit_durable_safety_event(ctx, {"type": "safety_sent_despite_check", "tool": tool_name,
                                       "subject_digest": digest, "verdicts": prior["verdicts"],
                                       "flagged_in": prior.get("llm_call_id", ""), "sent_by": call,
                                       "detail": "sent despite the check by a later model call"})
        return True, "⚠️ SAFETY_ADVICE: Identical re-issue from a later step; sent despite the check."
    if (speech or is_mcp) and prior.get("held_at") and call and call != prior.get("llm_call_id") and (
            quiz := _owner_yes(ctx, tool_name, arguments, effect, prior)):
        _emit_durable_safety_event(ctx, {"type": "safety_sent_without_assessment", "tool": tool_name,
                                       "subject_digest": digest, "quiz_id": quiz, "hold": prior.get("hold", ""),
                                       "held_at": prior["held_at"], "sent_by": call})
        return True, f"⚠️ SAFETY_ADVICE: Sent without a Safety assessment, on your human's answer to quiz {quiz}."

    # Preserve the selected assessment coverage independently of resource
    # admission. Cyber changes the authority of an assessment, never its verdict.
    # Non-full modes retain their existing disclosed skips.
    safety_mode = get_safety_mode(work_kind(getattr(ctx, "task_metadata", None)))
    if safety_mode != "full":
        skip_llm = safety_mode == "off" or (safety_mode == "light" and policy == POLICY_CHECK_CONDITIONAL)
        if skip_llm:
            _emit_safety_mode_skip(ctx, tool_name, safety_mode, policy)
            return True, ""

    try:
        assessment = _run_llm_check(tool_name, arguments, messages, ctx, resolved_binding)
    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        from ouroboros.model_wait import propagate_model_control
        (propagate_model_control if cyber else propagate_model_error)(exc)
        assessment = _hold(ctx, tool_name, sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}"))
    allowed, message = assessment
    verdicts = getattr(assessment, "verdicts", ())
    if verdicts:
        # Speech follows the final assessment (a second check included): SUSPICIOUS sends with its warning.
        marked = not speech and any(verdict.get("speech") is True for verdict in verdicts)  # words only the verdict saw
        speech = speech or marked
        flagged_speech = speech and getattr(assessment, "status", "") == "DANGEROUS"
        # Declared speech may be sent by a later decision in every mode; words only the verdict saw keep an ordinary
        # mode's refusal (each issue is checked again) and are refused once only in Cyber Pro, where the call would run.
        reissue = flagged_speech and (cyber or not marked)
        if not _remember(ctx, digest, tool_name, assessment, speech_refused=reissue):
            assessment = _hold(ctx, tool_name, "This work's mandatory prior-verdict history is too large", status="TOO_LARGE")
            allowed, message = assessment
        else:
            _emit_durable_safety_event(ctx, {"type": "safety_verdict", "tool": tool_name,
                                           "subject_digest": digest, "verdicts": verdicts})
            if flagged_speech:  # private to the mind: the words, the rule and where they would land
                reasons = "\n".join(f"{v['stage']}: {v['status']} — {v.get('reason', '')}" for v in verdicts)
                quote = next((v["quote"] for v in reversed(verdicts) if v.get("quote")), "")
                return False, ("⚠️ SAFETY_VIOLATION: Message not sent.\n" + reasons
                               + (f"\nFlagged words: {json.dumps(quote, ensure_ascii=False)}" if quote else "")
                               + (f"\n{assessment.destination}" if getattr(assessment, "destination", "") else "")
                               + ("\n" + _SEND_DESPITE if reissue else ""))
            if marked and allowed:  # the place guard a message meets at dispatch, before any verdict existed
                from ouroboros.place_guard import verdict_speech_refusal
                if refusal := verdict_speech_refusal(ctx, tool_name, arguments, is_mcp=is_mcp):
                    return False, refusal
    # Holds cannot earn retry authority. Speech holds bind in every runtime mode, and so does an MCP call's hold:
    # without a verdict nothing says it is not words to people (built-in undeclared tools' holds stay advice).
    if not allowed and (speech or (is_mcp and not getattr(assessment, "verdicts", ()))):
        if not getattr(assessment, "verdicts", ()):
            _remember(ctx, digest, tool_name, assessment, held=True)
        exits = _HELD_WORDS if speech else _HELD_CALL.format(call=_exact_call(tool_name, arguments))
        return False, message + "\n" + exits
    if cyber and not allowed:
        _emit_durable_safety_event(ctx, {"type": "safety_advisory", "tool": tool_name,
            "assessment_allowed": False, "assessment": message, "runtime_mode": _runtime_mode(ctx), "execution_allowed": True})
        return True, ("⚠️ SAFETY_ADVICE: Cyber Pro leaves the decision to Ouroboros. "
                      "Assessment recorded before execution:\n" + message)
    return allowed, message
