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
import re
import shlex
import time
from typing import Tuple, Dict, Any, List, Optional

from ouroboros.config import get_safety_model, get_runtime_mode, get_safety_call_timeout_sec, get_safety_max_tokens, get_safety_mode
from ouroboros.runtime_mode_policy import mode_has_unrestricted_agency
from ouroboros.llm import LLMClient
from ouroboros.loop_llm_call import classify_llm_exception, is_rate_limit_text
from ouroboros.pricing import emit_llm_usage_event, normalize_usage_cost, infer_provider_from_model
from ouroboros.utils import sanitize_tool_result_for_log, utc_now_iso
from ouroboros.config import runtime_setting
from ouroboros.deployment_policy import (
    RELEASE_AFTER_DANGEROUS, RELEASE_UNJUDGED, REPEAT_AFTER_DANGEROUS, checker, organization_rules, safety_exit_open,
)
from ouroboros.safety_config import OWNER_RULES_KEY

from ouroboros.safety_records import (
    _safety_drive_root, _record_safety_usage, _emit_durable_safety_event, _emit_safety_mode_skip, coalesce_verdicts,
    safety_check_wait, verdict_history_bytes, parse_safety_response as _parse_safety_response,
    _JUDGED, _PRIOR_LOCK, _PRIOR_VERDICTS, _RETURNED, _drop, _root_id, _track_returned, _work_key, tell_owner_over_flag,
)
from ouroboros.safety_records import clear_safety_work  # noqa: F401 - public re-export for task_results
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
    """The Supervisor's prompt, ``prompts/SAFETY.md``; one that cannot be loaded is a missing reading, never a
    stand-in."""
    try:
        return (pathlib.Path(__file__).parent.parent / "prompts" / "SAFETY.md").read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"its prompt prompts/SAFETY.md could not be loaded: {exc}") from exc


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


# Mandatory evidence must fit; a work's returned and held calls keep their own bound, the oldest leaving first.
_SAFETY_BUNDLE_BYTE_BUDGET = 512_000
_SAFETY_PRIOR_BYTE_BUDGET = 64_000
_PENDING_NOTE = ("host fact: the last call this work presented through this tool was returned to the acting mind with "
                 "this reading, and no call through this tool has been let through since; the present call may be its "
                 "changed version")


class _Assessment(tuple):
    """Keep the public pair ABI while distinguishing flags from missing readings (``cause``: why none was reached)."""
    def __new__(cls, allowed, message, *, status="", verdicts=(), destination="", cause=""):
        result = super().__new__(cls, (allowed, message))
        result.status, result.verdicts, result.destination, result.cause = status, tuple(verdicts), destination, cause
        return result


def _subject_digest(tool_name, arguments):
    # Hash ORIGINAL arguments, not redacted rendering: changed credentials are a new subject too.
    raw = json.dumps([tool_name, arguments], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8", "surrogatepass")).hexdigest()


# A returned call is released by a decision, never by bytes re-presented: the identical call from a later model call.
_MIND_DECIDES = ("You decide: change the call (a changed call is read again), drop it, or run it as it is by issuing "
                 "the identical call in a later step.")
_CARD_CHECK = ("The host checks only that they answered that card, not what they answered: issue the identical call "
               "again only if they said yes.")
_LATER_WORD = ("A later message or card answer of your human in this work (speech in a call is neither) earns it one "
               "new reading")  # under every deployment policy


def _returned_head(level, verdicts, destination="", words=False):
    """How a returned call opens: nothing ran, the reading (``words``: of these words to this destination, given to
    another call), its flagged words and where they would land."""
    quote = next((v["quote"] for v in reversed(verdicts) if v.get("quote")), "")
    read = (f"These words to this destination were read {level} in this work" if words
            else f"The Safety Supervisor read this call as {level}")
    return (f"⚠️ SAFETY_RETURNED: not run; nothing was sent or changed. {read}:\n"
            + _reasons(verdicts) + (f"\nFlagged words: {json.dumps(quote, ensure_ascii=False)}" if quote else "")
            + (f"\n{destination}" if destination else "")
            + f"\nIt read {'the call that carried them' if words else 'this call'} and the host's facts about it, "
            "not your whole work.")


def _card(subject):
    """What my human's card must hold for ``_owner_yes`` (``_exit_subject``)."""
    words, destinations, exact, _ = subject
    return (f"this exact call on its own line: {words}" if exact
            else f"these exact words on their own lines and the destination {', '.join(destinations)}")


def _again(subject):
    """What another call meets after a DANGEROUS reading: declared words keep waiting to the same destination."""
    return ("A changed call is read again." if subject[2] else "Changed words or another destination are read again; "
            "the same words there keep waiting, whatever the tool or request_id.")


def _waits(subject, cyber):
    where = "Under this organization's deployment policy" if cyber else "In this mode"  # Cyber Pro waits only there
    return (f"{where} a DANGEROUS reading waits for your human. If you still judge it right, ask them privately on "
            f"a card holding {_card(subject)}. {_CARD_CHECK} {_again(subject)} {_LATER_WORD}.")


def _closed(subject):
    """Neither applicable DANGEROUS exit is open: the saved verdict still answers."""
    return ("This organization's deployment policy lets neither a repeat nor a card release it. "
            f"{_again(subject)} {_LATER_WORD}; only the organization's administrator changes that policy.")


def _unread(cause, status=""):
    """A missing reading that holds the call: no verdict either way."""
    again = ("A smaller call may fit." if status == "TOO_LARGE"
             else "Issue the identical call again in a later step; do not reword it as if it had been judged.")
    return (f"⚠️ SAFETY_UNAVAILABLE: not run. The Safety Supervisor could not read this call ({cause}); that is no "
            f"verdict either way. {again}")


def _let_through(cause):
    """A missing reading that lets the call through: the line trails the tool's own result."""
    return f"⚠️ SAFETY_NOTE: let through without a Safety reading ({cause}); the tool's result says what happened."


def _authoring_call(ctx):
    """The step that authored this tool call: the model call (the loop's ``_current_llm_call_meta``), else a live
    call's tool round (``_live_tool_step``); "" when unknown, which is never the same step."""
    meta = getattr(ctx, "_current_llm_call_meta", None)
    return str((meta.get("llm_call_id") if isinstance(meta, dict) else "") or getattr(ctx, "_live_tool_step", "") or "")


def _owner_ids(ctx):
    """My human's rows in this work as the Supervisor's bundle lists them, each as short digests of the identities
    it names (``owner_words.safety_owner_ids``), never their words."""
    from ouroboros.owner_words import safety_owner_ids
    return [{_subject_digest("owner_words", ident)[:12] for ident in ids} for ids in safety_owner_ids(ctx)]


def _owner_seen(kept, listed=()):
    """The digests of my human's rows the readings kept for this call and these words' gate (``kept``) saw,
    ``listed`` (``_owner_ids``) added."""
    marks = [row["owner_words"] for row in kept if row and row.get("owner_words")]
    return set().union(*(mark["seen"] for mark in marks), *listed)


def _remember(ctx, digest, tool_name, assessment, *, flagged="", gate="", held="", owner=None):
    """Keep a returned call (``flagged``: its level) or a held one (``held``: UNAVAILABLE or TOO_LARGE), with its
    verdicts, the step that authored it and, for DANGEROUS, what its readings saw of my human's rows (``owner``,
    ``_owner_seen``) and, for declared words, their words-and-destination key (``gate``). Past the byte bound the
    oldest calls leave first, never the newest; a flagged call no longer kept is read afresh and never leaves
    unread. Without a context nothing is kept (False)."""
    key = _work_key(ctx)
    if not key:
        return False
    with _PRIOR_LOCK:
        rows = _PRIOR_VERDICTS.pop(key, {})
        before = rows.pop(digest, {})
        rows[digest] = {"digest": digest, "tool": tool_name,
                        "verdicts": coalesce_verdicts(before.get("verdicts", ()), assessment.verdicts),
                        **({"flagged": flagged} if flagged else {"held": held} if held else {}),
                        "llm_call_id": _authoring_call(ctx), **({"gate": gate} if gate else {}),
                        **({"owner_words": {"seen": sorted(owner)}} if owner is not None else {})}
        while len(rows) > 1 and verdict_history_bytes(rows) > _SAFETY_PRIOR_BYTE_BUDGET:
            _drop(key, rows, next(iter(rows)))
        _PRIOR_VERDICTS[key] = rows
        while len(_PRIOR_VERDICTS) > 512:
            _JUDGED.pop(_PRIOR_VERDICTS.popitem(last=False)[0], None)  # an evicted work's calls are read afresh
    return True


def _runtime_mode(ctx):
    return effective_runtime_mode(get_runtime_mode(), getattr(ctx, "task_metadata", None))


def _needs_checker(ctx):
    return bool(str(runtime_setting("OUROBOROS_MODEL_SAFETY", "") or "").strip()) or (
        work_kind(getattr(ctx, "task_metadata", None)) == "consciousness"
        and get_safety_mode("consciousness") == "full")


def _hold(ctx, tool_name, reason, *, status="UNAVAILABLE"):
    """Hold this one call: no reading was reached (``_unread``)."""
    _emit_durable_safety_event(ctx, {"type": "safety_hold", "tool": tool_name, "reason": reason, "hold": status})
    return _Assessment(False, _unread(reason, status), status=status, cause=reason)


def _declared_speech(tool_name, effect):
    """Words the host knows before any verdict, from declarations only: the automatic reply or a ``message`` effect
    (a connector's ``sends.when`` included); never an argument name."""
    return tool_name == "place_reply" or effect.effect == "message"


def _exact_call(tool_name, arguments):
    """A call as my human's card quotes it when it declares no words or destination of its own: the whole call."""
    return f"{tool_name} {json.dumps(arguments, ensure_ascii=False, sort_keys=True, default=str)}"


def _exit_subject(ctx, tool_name, arguments, effect, ext_tool=None):
    """What my human's card must hold, as ``(words, destinations, exact, labels)``: declared words with each place's
    label and every recipient outside a place's own conversation (one place never covers another recipient), else the
    exact call (``_exact_call``): any other call, or words or recipients the host cannot tell."""
    from ouroboros.place_guard import call_place_facts
    from ouroboros.tool_effects import argument_value, target_values

    if _declared_speech(tool_name, effect):
        words = argument_value(arguments, effect.text or "text")
        facts = call_place_facts(ctx, tool_name, arguments, ext_tool=ext_tool) or {}
        rows = [row for row in facts.get("places") or [] if row.get("place_label")]
        labels, own = list(dict.fromkeys(row["place_label"] for row in rows)), {row.get("conversation_id") for row in rows}
        targets = facts.get("targets") or target_values(effect, arguments)
        destinations = list(dict.fromkeys(labels + [value for key, values in targets.items() for value in values
                                                    if key not in {"account_id", "thread_id", effect.thread} and value not in own]))
        if isinstance(words, str) and words.strip() and destinations:
            return words, destinations, False, labels
    return _exact_call(tool_name, arguments), [tool_name], True, []


def _owner_yes(ctx, subject):
    """The id of my human's answer in this work whose card (the quiz it answers, never a chat message) holds the words
    and every destination ``subject()`` resolves; "" if none. A card answered before the call counts, as does an
    answer that started a fresh head (its card rides in the head's owner rows). The host checks bytes and destination
    only: the mind's call after reading the answer is its judgment that it approves."""
    from ouroboros.owner_words import OWNER_SOURCES, is_answered_card

    cards = [(row, str(row["reply_to"]["text"])) for row in getattr(ctx, "_owner_directives", None) or []
             if isinstance(row, dict) and row.get("source") in OWNER_SOURCES and is_answered_card(row.get("reply_to"))
             and row["reply_to"].get("text")]
    words, destinations, _, _ = subject() if cards else ("", [], True, [])
    for row, card in cards:
        # The words as whole lines of the card and each destination as a whole token, never a part of a longer one.
        if (f"\n{words.strip()}\n" in f"\n{card}\n"
                and all(re.search(rf"(?<!\w){re.escape(d)}(?!\w)", card) for d in destinations)):
            return str(row.get("quiz_id") or row.get("msg_id") or row["reply_to"].get("address") or "an answer")
    return ""


def _words_key(subject):
    """Declared words and their place (else recipients) apart from the tool and other arguments, so a wait for my human
    is never re-rolled through another door or request_id; "" when the host cannot tell them."""
    return "" if subject[2] else _subject_digest("words", [subject[0], sorted(subject[3] or subject[1])])


def _build_check_prompt(tool_name, arguments, messages=None, *, args_json=None, ctx=None, resolved_binding=None,
                        evidence=None, ext_tool=None):
    """Fixed host evidence, with authorship separate from semantic authorization. ``evidence`` (a dict) receives
    the destinations and notes resolved here, so a refusal names them without resolving them again; an extension
    call's place facts come from the descriptor being dispatched (``ext_tool``), never a later publication."""
    from dataclasses import asdict, is_dataclass
    from ouroboros import owner_words
    from ouroboros.place_guard import call_place_facts, place_evidence
    from ouroboros.tool_effects import mcp_sends_as
    from ouroboros.safety_evidence import focus_event_source, with_author

    metadata = getattr(ctx, "task_metadata", None) or {}
    kind = work_kind(metadata)
    focus = metadata.get("focus") or {}
    place = dict(call_place_facts(ctx, tool_name, arguments, ext_tool=ext_tool) or {})
    place_ids = {row["place_id"] for row in place.get("places", []) if row.get("place_id")}
    # A focus occasions the work; it does not establish where this call lands.
    # Only these two built-ins take an actual place reference as their subject.
    referenced = arguments.get("place_id") if tool_name in {"configure_presence", "place_reply"} else None
    if referenced and referenced not in place_ids:  # a resolved row already names its place
        place["place_id"] = referenced
        place_ids.add(referenced)
    # Each place's notes and circle words once, read now from the canonical store (a circle call adds its circle).
    notes = place_evidence(ctx, place, place_ids, tool_name, arguments)
    if tool_name == "place_reply":  # always a message; resolved rows already carry its route
        place["effect"] = "message"
        if not place.get("targets"):
            place["targets"] = {"destination": arguments.get("destination")}
    occasion = "unknown"
    if metadata.get("schedule_writer_kind"):  # whose work; who supplied its text is its ``schedule_text`` sources' fact
        occasion = metadata["schedule_writer_kind"] + " work's schedule"
    elif focus:  # read at this check: a confirmation made after the admission ``focus`` records counts
        from ouroboros.presence_context import focus_owner_now
        occasion = "confirmed owner account" if focus_owner_now(ctx) else "person in a place"
    elif metadata.get("wake_cause") == "own_clock":
        occasion = "own clock"
    elif kind == "owner":
        occasion = "owner door"
    directives = getattr(ctx, "_owner_directives", None)
    # User messages are quoted input when the caller has no retained host source; never certify them as owner words.
    rows = (owner_words.safety_sources(ctx) if isinstance(directives, list)
            else [m for m in (messages or []) if m.get("role") == "user"])
    sources = [with_author(focus_event_source(row, metadata, notes), metadata) for row in rows]  # diet, then author
    target = asdict(resolved_binding) if is_dataclass(resolved_binding) else resolved_binding
    # Whose credential an MCP call acts with (e.g. a server signed in as my human); connectors carry it per place.
    sends_as = {"sends_as": mcp} if tool_name.startswith("mcp_") and (mcp := mcp_sends_as(tool_name)) else {}
    with _PRIOR_LOCK:
        pending = dict(_RETURNED.get(_work_key(ctx), {}).get(tool_name) or {})
    bundle = {
        "work": {"kind": kind, "label": "your work" if kind == "owner" else "consciousness",
                 "root_task_id": _root_id(ctx), "task_id": getattr(ctx, "task_id", ""), "runtime_mode": _runtime_mode(ctx),
                 "consciousness_autonomy": metadata.get("consciousness_autonomy"), "now": utc_now_iso("seconds")},
        "occasion": {"cause": occasion, "consciousness_cause": metadata.get("consciousness_cause"),
                     "origin_wake_task_id": metadata.get("origin_wake_task_id"), "focus": focus,
                     "sources": sources, "retained_host_sources": isinstance(directives, list)},
        "subject": {"tool": tool_name, "physical_target": target, "place": place, **sends_as,
                    **({"returned_and_pending": {"note": _PENDING_NOTE, **pending}} if pending else {})},
        # The acting mind's context reads the same key through the same snapshot (``owner_rules_section``);
        # a deployment's rules keep their own author, the policy file (``organization_rules_section``).
        "standing_delegation": {"owner_rules": runtime_setting(OWNER_RULES_KEY, ""), "place_notes": notes,
                                **({"organization_rules": rules} if (rules := organization_rules()) else {})},
    }
    if isinstance(evidence, dict):
        evidence.update(place=place, notes=notes)
    return (f"Runtime mode: {_runtime_mode(ctx)}\nHost evidence (provenance is not consent):\n"
            + _render_subject_json(bundle) + "\nProposed tool call:\nTool: " + tool_name
            + "\nArguments:\n" + (args_json if args_json is not None else _render_subject_json(arguments))
            + "\nIs this safe?")


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


def _route_has_access(model: str, use_local: bool) -> bool:
    """Whether a deployment's Supervisor route may be asked: local and subscription routes may; an API route
    needs its connection's credential, the test ``_resolve_safety_routing`` applies to the Supervisor's own route."""
    from ouroboros.provider_models import provider_for_model

    return use_local or provider_for_model(model) == "claudexor" or _light_model_has_reachable_provider(model)


def _resolve_safety_routing() -> Tuple[bool, bool, Optional[str]]:
    """Choose local/remote safety backend; unreachable fallback fails open.

    Only no-backend cases fail open with a ``SAFETY_NOTE``: no local lane and either no provider key or one not
    covering the light model's provider; a fallback local runtime that fails or is rate-limited, never a chain route
    (``_unanswered``). An explicit local runtime (``USE_LOCAL_LIGHT``) that fails holds."""
    selected = get_safety_model()
    explicit = bool(str(runtime_setting("OUROBOROS_MODEL_SAFETY", "") or "").strip())
    local_light = str(runtime_setting("USE_LOCAL_LIGHT", "") or "").lower() in ("true", "1")
    # The lane is the one the resolved model names (``get_safety_model`` labels it): a blank Light follows
    # Main and Main's lane, so a leftover USE_LOCAL_LIGHT never sends an inherited remote Main locally (#2969).
    if selected.endswith(" (local)"):
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
        return False, False, (f"its model's provider key is missing for {light_model}: other remote keys are set "
                              "but none covers this provider")

    if not explicit and _any_local_routing_enabled():
        # Local-only configs should warn, not hard-block, on local outage.
        return True, True, None

    return False, False, "no Safety model is available: neither remote provider keys nor local routing are configured"


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
) -> "_Assessment":
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
        cause = f"the local fallback Safety Supervisor was rate-limited: {error}"
        return _Assessment(True, _let_through(cause), status="UNAVAILABLE", cause=cause)
    return _safety_unavailable_blocked(ctx, tool_name, error, attempts=attempts)


def _safety_unavailable_blocked(
    ctx: Optional[Any], tool_name: str, error: str, *, attempts: int = 2,
) -> "_Assessment":
    'Rate-limited safety lane: BLOCK this one call with a typed non-verdict outcome (``_mark_storm`` closes the route).'
    shape, action = (("storm latch, no attempt", "blocked_unchecked_storm_latched") if not attempts else
                     ("deadline expired before the retry", "blocked_unchecked_deadline_expired") if attempts < 2 else
                     ("after one retry", "blocked_unchecked_after_retry"))
    log.error("Safety check rate-limited for %s (%s); blocking unchecked: %s", tool_name, shape, error)
    _emit_durable_safety_event(ctx, {"type": "safety_check_rate_limited", "tool": tool_name, "action": action,
                                     "error": error})
    # The provider error goes on its own line: downstream status classification scans
    # only the FIRST line for outcome markers, and a raw error body could carry one.
    cause = "its provider is rate-limited"
    return _Assessment(False, f"{_unread(cause)}\n(provider error: {error})", status="UNAVAILABLE", cause=cause)


def _subject_too_large_blocked(ctx: Optional[Any], tool_name: str, subject_chars: int) -> "_Assessment":
    'Over-budget subject: no reading, and never a truncated one (a partially read call must never run).'
    log.error(
        "Safety subject for %s is %d chars (budget %d); refusing the unreviewable call",
        tool_name, subject_chars, _SAFETY_SUBJECT_CHAR_BUDGET,
    )
    _emit_durable_safety_event(ctx, {
        "type": "safety_subject_too_large", "tool": tool_name,
        "subject_chars": int(subject_chars),
        "budget_chars": _SAFETY_SUBJECT_CHAR_BUDGET,
    })
    cause = (f"its arguments serialize to {subject_chars:,} characters, above the {_SAFETY_SUBJECT_CHAR_BUDGET:,} it "
             "reads in one prompt; nothing was truncated")
    return _Assessment(False, _unread(cause, "TOO_LARGE"), status="TOO_LARGE", cause=cause)


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
        raise _SafetyRateLimited("safety lane cooling down after a recent rate-limit storm; no attempt made", latched=True)
    with safety_check_wait(ctx, tool_name):  # the tool waits on this check, its bounded retry included
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
    locality a live wait-card switch gives the role (what is sent, recorded and charged), its role, its binding.
    A deployment's checker walks only its own routes, with their own accounts, and no switch moves them."""
    from ouroboros.model_slots import route_binding
    from ouroboros.provider_models import fallback_candidate_targets

    if (managed := checker()) is not None:
        return [(model, local, "safety", (model, local, account)) for model, local, account in managed.routes()]
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
        return _rate_limited_outcome(ctx, tool_name, str(error), local_fallback=fail_open, attempts=error.attempts)
    from ouroboros.llm_claudexor import propagate_model_error
    propagate_model_error(error)
    safe_error = sanitize_tool_result_for_log(f"{type(error).__name__}: {error}")
    if fail_open:  # a fallback local outage warns instead of blocking all unknown tools
        log.warning("Safety local-fallback LLM call failed for %s (%s); proceeding with warning", tool_name, safe_error)
        cause = f"the local Safety runtime is unreachable: {safe_error}"
        return _Assessment(True, _let_through(cause), status="UNAVAILABLE", cause=cause)
    log.error("Safety check LLM call failed for %s: %s", tool_name, safe_error)
    return _hold(ctx, tool_name, f"its check failed: {safe_error}")


def _run_llm_check(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]],
    ctx: Optional[Any],
    resolved_binding: Optional[Any] = None,
    ext_tool: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Run a single light-model safety check and classify the verdict."""
    args_json = _render_subject_json(arguments)
    if len(args_json) > _SAFETY_SUBJECT_CHAR_BUDGET:
        return _subject_too_large_blocked(ctx, tool_name, len(args_json))
    evidence = {}
    prompt = _build_check_prompt(tool_name, arguments, messages, args_json=args_json,
                                 ctx=ctx, resolved_binding=resolved_binding, evidence=evidence, ext_tool=ext_tool)
    if len(prompt.encode("utf-8")) > _SAFETY_BUNDLE_BYTE_BUDGET:
        return _hold(ctx, tool_name, "the complete mandatory evidence bundle is too large", status="TOO_LARGE")
    # A deployment's walk judges each of its own routes (below): its primary's access never stops the others.
    managed = checker()
    _use_local_light, _is_local_fallback, _skip_reason = (False, False, None) if managed else _resolve_safety_routing()
    if _skip_reason is not None:
        if _needs_checker(ctx):
            return _hold(ctx, tool_name, _skip_reason)
        return _Assessment(True, _let_through(_skip_reason), status="UNAVAILABLE", cause=_skip_reason)
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
    from ouroboros.model_wait import current_model_wait, model_wait_reason, propagate_model_control
    from ouroboros.transport_custody import outcome_unknown_on_chain

    fail_open = _use_local_light and _is_local_fallback and not _needs_checker(ctx)
    waiter = current_model_wait()
    routes = _safety_routes(light_model, _use_local_light, waiter.overrides if waiter else None)
    skipped: List[Dict[str, str]] = []  # routes that answered nothing, then the one that answered

    def pinned(binding: tuple) -> Dict[str, Any]:  # a deployment's route: its own account, no inherited preference
        return {"account": binding[2], "processing": ""} if managed is not None else {}

    def record_skips(answered: Optional[Dict[str, str]]) -> None:  # None: no route of the walk answered
        _emit_durable_safety_event(ctx, {"type": "safety_route_fallback", "tool": tool_name,
                                         "unavailable": skipped, "answered": answered})

    if managed is not None:  # a route this process cannot ask answers nothing and is never sent
        skipped = [{"model": row[0], "role": row[2], "error": "its connection has no credential"}
                   for row in routes if not _route_has_access(row[0], row[1])]
        routes = [row for row in routes if _route_has_access(row[0], row[1])]
        if not routes:
            record_skips(None)
            return _hold(ctx, tool_name,
                         "no route of the deployment's Safety Supervisor has its connection's credential")
    alone, own_failure, own_reply = len(routes) == 1, None, None

    for index, (light_model, _use_local_light, role, binding) in enumerate(routes):
        try:
            if index and time.time() < _CHAIN_STORMS.get(binding, 0.0):
                raise _SafetyRateLimited("cooling down after this route's own rate-limit storm", latched=True)
            msg, usage = _safety_model_call(
                client=client, ctx=ctx, tool_name=tool_name, light_model=light_model, use_local=_use_local_light,
                call_type="safety_supervisor", user_prompt=prompt, on_usage=_emit_safety_usage,
                system_prompt=system_prompt, model_role=role, latch=not index,
                wait_for_resources=None if alone else False, **pinned(binding),
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
                if skipped:
                    record_skips(None)
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
                    system_prompt=system_prompt, **pinned(binding))
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
        own = binding == routes[0][3]  # only the walk's first route consults and arms the process latch
        try:
            repair_prompt = (
                "Your previous Safety Supervisor response was not parseable as the required JSON object.\n"
                "Return ONLY this strict JSON shape, with no markdown and no prose:\n"
                "{\"status\":\"SAFE|SUSPICIOUS|DANGEROUS\",\"reason\":\"short reason\",\"quote\":\"\"}\n\n"
                "Original proposed tool call follows again.\n\n"
                f"{prompt}"
            )
            repair_msg, repair_usage = _safety_model_call(
                client=client, ctx=ctx, tool_name=tool_name, light_model=light_model,
                use_local=_use_local_light, call_type="safety_supervisor_repair",
                user_prompt=repair_prompt, on_usage=_emit_safety_usage, system_prompt=system_prompt,
                model_role=role, latch=own, **pinned(binding),
            )
            result = _parse_safety_response(repair_msg.get("content") or "")
            if result is None:
                failure_class = _classify_safety_parse_failure(repair_msg, repair_usage)
        except _SafetyRateLimited as e:  # a chain route's storm closes that route; it never takes the local warning
            _mark_storm(e, binding, own=own, fail_open=fail_open)
            return _unanswered(ctx, tool_name, e, fail_open=fail_open and own)
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
            return _hold(ctx, tool_name,
                         f"its answer stayed unparseable (class={failure_class}) after one repair retry")

    verdicts, row, failure, second = [{"stage": "first", **result}], None, "", None
    if result["status"] != "SAFE" and managed is None:  # a deployment's Supervisor decides alone
        from ouroboros.configured_subagents import normalize_configured_subagents
        try:
            catalog, _ = normalize_configured_subagents(runtime_setting("OUROBOROS_SUBAGENTS", "") or {"enabled": False, "items": []})
            row = next((r for r in catalog.items if r.rechecks_flags and r.enabled), None) if catalog.enabled else None
        except ValueError as exc:
            failure = f"its roster is unreadable: {sanitize_tool_result_for_log(str(exc))}"
    if row is not None:
        second_prompt = ("You review a flag a first checker raised; overturning it needs a reason.\n"
                         + prompt + "\nFirst verdict:\n" + _render_subject_json(result))
        light_model = row.route.target_id.removesuffix(" (local)")
        _use_local_light = row.route.target_id.endswith(" (local)")
        if len(second_prompt.encode("utf-8")) > _SAFETY_BUNDLE_BYTE_BUDGET:
            failure = "its bundle is too large"
        else:
            try:
                msg, _ = _safety_model_call(
                    client=client, ctx=ctx, tool_name=tool_name, light_model=light_model, use_local=_use_local_light,
                    call_type="safety_second_check", user_prompt=second_prompt, on_usage=_emit_safety_usage,
                    system_prompt=system_prompt, model_role="subagent:" + row.subagent_id, effort=row.effort or "low",
                    account=row.route.credential_profile_id, processing=row.processing_preference or None,
                    attempts_limit=1, latch=False, wait_for_resources=False)
                second = _parse_safety_response(msg.get("content") or "")
            except Exception as exc:
                propagate_model_control(exc)  # only Stop or Pause ends it; any other failure is this check's own
                failure = sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}")
        if second is not None and (second["status"] != "SAFE" or str(second.get("reason") or "").strip()):
            result = second
            verdicts.append({"stage": "second", "row_id": row.subagent_id, **result})
        else:
            failure = failure or "it returned no verdict with a reason"
    if failure:  # an optional second check that answered nothing leaves the first verdict standing
        _emit_durable_safety_event(ctx, {"type": "safety_second_check_failed", "tool": tool_name, "error": failure})
        if verdicts[-1].get("owner"):  # its mark stays on record, but no final reading confirmed it for my human's row
            verdicts[-1] = {**verdicts[-1], "owner_unconfirmed": True}
    status = result["status"]
    return _Assessment(status == "SAFE", "", status=status, verdicts=verdicts,
                       destination="" if status == "SAFE" else "\n".join(_destination_lines(evidence)))


def _reasons(verdicts):
    """Each stage's level and reason, one per line, as the mind and the room's record read them."""
    return "\n".join(f"{v['stage']}: {v['status']} — {v.get('reason', '')}" for v in verdicts)


def _destination_lines(evidence):
    """The flagged call's destinations as the bundle resolved them: each label with my human's notes and circle words."""
    from ouroboros.place_guard import destination_line

    place = evidence.get("place") or {}
    rows = {row["place_id"]: row for row in [*(place.get("places") or []), place] if row.get("place_id")}
    return [destination_line(row.get("place_label") or pid, [note for note in evidence.get("notes") or ()
                                                            if note["place_id"] == pid]) for pid, row in rows.items()]


def check_safety(
    tool_name: str,
    arguments: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
    ctx: Optional[Any] = None,
    python_resolution: Optional[Any] = None,
    resolved_binding: Optional[Any] = None,
    declared_policy: str = "",
    ext_tool: Optional[Any] = None,
) -> Tuple[bool, str]:
    """Return ``(allowed, warning_or_error)`` for one tool call (``_assess_call``); the reading a returned call came
    back with stays beside its tool until a call through that tool is let through (``_track_returned``)."""
    tool_name = str(tool_name or "").strip()
    result = _assess_call(tool_name, arguments or {}, messages, ctx, python_resolution, resolved_binding,
                          declared_policy, ext_tool)
    _track_returned(ctx, tool_name, result)
    return result


def _assess_call(tool_name, arguments, messages, ctx, python_resolution, resolved_binding, declared_policy, ext_tool):
    """One call's assessment.

    ``declared_policy`` is a first-party ``ToolEntry``'s own declaration; a
    ``TOOL_POLICY`` entry outranks it and an invalid value is ``DEFAULT_POLICY``.
    Words are a host fact from declarations alone (``_declared_speech``). A flagged call, SUSPICIOUS or DANGEROUS and
    whatever its tool, returns once before any effect with the reasons, the verdict's ``quote`` and the destination's
    notes (the final stage's verdict, a second check included). The identical call (request_id included) from a later
    model call is let through unread for SUSPICIOUS in every mode and for DANGEROUS in Cyber Pro, the mode read at
    that issue; elsewhere DANGEROUS waits for my human, declared words keyed on words and destination so a new
    request_id or another tool buys no fresh reading; a later message of my human in this work earns one: a row of
    theirs no reading of the call or gate saw (``_owner_seen``).
    A SAFE one leaves the call as one never flagged; one not made says why, and the next issue tries again.
    My human's answer whose card holds the exact words as whole lines and the destination (any other call: the exact
    call) lets it through, one given before the issue included (``_owner_yes``): the host checks that the card was
    answered, never what the answer says. The step that received a return never releases it; an unknown step is a
    later one. A hold on words, or on an MCP call, binds in every mode and names the same card. Deployment exits
    separately govern the mind's DANGEROUS repeat, my human's card after DANGEROUS and a card without a reading; a
    closed exit buys no second roll of a kept verdict, and a flagged call no longer kept is read afresh, never unread.
    ``ext_tool`` is the extension descriptor being dispatched: its effect and the brief's place facts are the
    publication it was read with (``tool_effects.bound_declaration``)."""
    declared = declared_policy if isinstance(declared_policy, str) else ""
    policy = TOOL_POLICY.get(tool_name) or (declared if declared in POLICY_VALUES else DEFAULT_POLICY)

    from ouroboros.tool_effects import effect_of
    is_mcp = tool_name.startswith("mcp_")
    effect = effect_of(tool_name, ext_tool=ext_tool, ctx=ctx, is_mcp=is_mcp, args=arguments)
    speech = _declared_speech(tool_name, effect)  # an unread MCP call waits too
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
            # A script runs whole under its interpreter, so its first word proves nothing about
            # the rest (`ls` then `curl … | sh`): no safe-subject bypass, like a string check below.
            raw_cmd = None
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

    # Preserve the selected assessment coverage independently of resource
    # admission. Cyber changes the authority of an assessment, never its verdict.
    # Non-full modes retain their existing disclosed skips: an unchecked call meets no flag, wait or card.
    safety_mode = get_safety_mode(work_kind(getattr(ctx, "task_metadata", None)))
    if safety_mode != "full":
        skip_llm = safety_mode == "off" or (safety_mode == "light" and policy == POLICY_CHECK_CONDITIONAL)
        if skip_llm:
            _emit_safety_mode_skip(ctx, tool_name, safety_mode, policy)
            return True, ""

    cyber = mode_has_unrestricted_agency(_runtime_mode(ctx))
    digest = _subject_digest(tool_name, arguments)
    with _PRIOR_LOCK:
        key = _work_key(ctx)
        rows = _PRIOR_VERDICTS.get(key, {})
        prior, gates = rows.get(digest, {}), {row["gate"]: row for row in rows.values() if row.get("gate")}
        dropped = set(_JUDGED.get(key, ()))
        if key in _PRIOR_VERDICTS:
            _PRIOR_VERDICTS.move_to_end(key)
    call, flag, resolved = _authoring_call(ctx), prior.get("flagged", ""), []
    # Unmanaged keeps upstream behavior; an explicit list never silently gains the new human exit.
    repeat, release = safety_exit_open(REPEAT_AFTER_DANGEROUS), safety_exit_open(RELEASE_UNJUDGED)
    release_dangerous = safety_exit_open(RELEASE_AFTER_DANGEROUS)

    def subject():  # what my human's card must hold, resolved at most once per check (``_exit_subject``)
        resolved or resolved.append(_exit_subject(ctx, tool_name, arguments, effect, ext_tool))
        return resolved[0]

    def let_through(event, text, **facts):  # released without a new reading, audited
        _emit_durable_safety_event(ctx, {"type": event, "tool": tool_name, "subject_digest": digest, "sent_by": call,
                                         **({"verdicts": basis} if basis else {}),
                                         **({"flagged_in": prior.get("llm_call_id", "")} if flag else {}), **facts})
        return True, text

    def released(quiz):  # my human's answered card holds this exact call, or these exact words and destination
        held = {"hold": prior["held"]} if prior.get("held") else {}
        return let_through("safety_sent_despite_check" if level else "safety_sent_without_assessment",
                           f"⚠️ SAFETY_NOTE: let through on your human's answer ({quiz}), without "
                           f"{'a new' if level else 'a'} reading. The host checked that they answered a card holding "
                           f"this exact {'call' if subject()[2] else 'text and destination'}, not what they answered.",
                           quiz_id=quiz, **held)

    def close(waits):  # who releases a returned call, read at its issue: the mind, else my human's card
        return (_waits(subject(), cyber) if release_dangerous else _closed(subject())) if waits else _MIND_DECIDES
    # The mind releases SUSPICIOUS, and DANGEROUS in Cyber Pro. ``gated``: these declared words to this destination
    # were read DANGEROUS in this work, through whatever tool or request_id.
    gated = gates.get(_words_key(subject())) if gates and speech else None
    level = "DANGEROUS" if gated else flag  # the words' gate outranks an older exact-call flag
    basis = (gated or prior).get("verdicts") or []  # the reading a release overrides
    judged = bool(basis) or digest in dropped or bool(speech and dropped and _words_key(subject()) in dropped)
    waits = level == "DANGEROUS" and not (cyber and repeat)
    same_step = bool(call) and call in {row.get("llm_call_id") for row in (prior, gated) if row}

    def waiting(unmade=""):  # the kept reading answers; for other words' gate, as theirs
        head = _returned_head(level, basis, words=bool(gated) and gated["digest"] != digest)
        return _Assessment(False, f"{head} {close(waits)}{unmade}", status=level, verdicts=basis)
    if flag and same_step:  # the returned decision re-presented, not a later one
        return _Assessment(False, "⚠️ SAFETY_RETURNED: not run. This exact call was returned earlier in this same "
                           f"step; decide in a later step. {close(waits)}", status=level, verdicts=basis)
    if waits and release_dangerous and not same_step and (quiz := _owner_yes(ctx, subject)):
        return released(quiz)
    listed = _owner_ids(ctx) if waits and not same_step else None  # my human's rows as this context lists them
    seen = _owner_seen((gated, prior))  # a later message of theirs earns one new reading
    reread = bool(listed) and any(not ids & seen for ids in listed)
    if waits and not reread:
        return waiting()
    if flag and not waits:  # an owner-marked reading also tells my human, in the Supervisor's own row
        tell_owner_over_flag(ctx, tool_name, arguments, subject, basis, level)
        return let_through("safety_sent_despite_check", "⚠️ SAFETY_NOTE: let through on your later identical call, "
                           "without a new reading; the tool's result says what happened. The Supervisor had read it "
                           f"as {level}: {_reasons(basis)}",
                           detail="let through despite the check by a later model call")
    # Words or an MCP call no reading judged leave on my human's card where this deployment leaves that release open.
    if (speech or is_mcp) and release and not judged and not same_step and (quiz := _owner_yes(ctx, subject)):
        return released(quiz)

    try:
        assessment = _run_llm_check(tool_name, arguments, messages, ctx, resolved_binding, ext_tool)
    except Exception as exc:
        from ouroboros.llm_claudexor import propagate_model_error
        from ouroboros.model_wait import propagate_model_control
        (propagate_model_control if cyber else propagate_model_error)(exc)
        assessment = _hold(ctx, tool_name, sanitize_tool_result_for_log(f"{type(exc).__name__}: {exc}"))
    verdicts, status = getattr(assessment, "verdicts", ()), getattr(assessment, "status", "")
    if reread and not verdicts:  # no new reading was reached: the kept one still answers, and says why
        return waiting(f" A new reading after your human's later message or card answer could not be made "
                       f"({getattr(assessment, 'cause', '')}); the next issue tries again.")
    if reread:  # the new reading replaces what answered: this call's row and these words' gate
        with _PRIOR_LOCK:
            rows = _PRIOR_VERDICTS.get(key, {})
            for stale in [d for d, row in rows.items() if d == digest or (gated and row.get("gate") == gated["gate"])]:
                _drop(key, rows, stale, judged=status != "SAFE")  # SAFE: as a call never flagged
    if verdicts:
        _emit_durable_safety_event(ctx, {"type": "safety_verdict", "tool": tool_name,
                                       "subject_digest": digest, "verdicts": verdicts})
        if status not in ("SUSPICIOUS", "DANGEROUS"):
            return assessment
        # Every flagged call returns once before any effect; its next issue reads the exits as they then stand.
        danger = status == "DANGEROUS"
        kept = _remember(ctx, digest, tool_name, assessment, flagged=status,
                         gate=_words_key(subject()) if danger else "",
                         owner=_owner_seen((gated, prior), _owner_ids(ctx) if listed is None else listed)
                         if danger and key else None)
        tail = f" {close(danger and not (cyber and repeat))}" if kept else ""  # nothing kept, no exit
        return _Assessment(False, _returned_head(status, verdicts, assessment.destination) + tail, status=status,
                           verdicts=verdicts)
    allowed, message = assessment
    if allowed:  # no reading, let through (``_let_through``)
        return assessment
    cause = getattr(assessment, "cause", "") or message
    # A call no reading judged leaves on my human's card for its exact form (a pre-approved github_write).
    if release and not judged and not same_step and (quiz := _owner_yes(ctx, subject)):
        return released(quiz)
    if speech or is_mcp:  # a hold on words or on an MCP call binds in every mode
        held = _remember(ctx, digest, tool_name, assessment, held=status or "UNAVAILABLE")
        card = (f"\nOr ask your human privately on a card holding {_card(subject())}: their answer lets the identical "
                f"call through unread. {_CARD_CHECK}") if held and release and not judged else ""
        return _Assessment(False, message + card, status=status, cause=cause)
    if cyber:  # a built-in tool's hold is advice in Cyber Pro
        _emit_durable_safety_event(ctx, {"type": "safety_advisory", "tool": tool_name,
            "assessment_allowed": False, "assessment": message, "runtime_mode": _runtime_mode(ctx), "execution_allowed": True})
        return True, _let_through(cause)
    return assessment
