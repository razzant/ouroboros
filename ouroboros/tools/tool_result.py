"""Typed internal tool results with a byte-compatible legacy text adapter."""

from __future__ import annotations

import json
import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

from ouroboros.platform_layer import posix_signal_name

ToolStatus = Literal["ok", "error", "blocked", "timeout", "unavailable"]
_TOOL_STATUSES = frozenset({"ok", "error", "blocked", "timeout", "unavailable"})
_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
# One or more spaces: the legacy loop scan keyed on ``startswith("⚠️ ")`` and then
# looked anywhere in the first line, so a two-space warning classified there and
# would silently read as OK here. No producer emits that shape today; accepting it
# keeps the single parser from acquiring a blind spot the pair did not have.
_FIRST_MARKER_RE = re.compile(r"^⚠️ +([A-Z][A-Z0-9_]*)")
_SAFETY_SEPARATOR = "\n\n---\n"
# Host notes TRAIL the payload since #447 H1; this is the boundary a text-only
# reader uses to recover the producer payload the note was appended to.
_HOST_NOTE_SEPARATOR = "\n\n⚠️ "
_MCP_RESULT_ENVELOPE_PREFIX = "External MCP tool result from "
# The registry composes this sentence PRECISELY BECAUSE no provider ran: the name
# resolved to nothing. It is host text under a dynamic-looking name, which is the
# one place the ``ext_``/``mcp_`` name-shape proxy for "body I did not compose" is
# wrong, so both the adapter and the ordered chain read it from here.
_UNKNOWN_TOOL_PREFIX = "⚠️ Unknown tool:"
_MAX_META_ITEMS = 32
_MAX_META_BYTES = 8192
_HOST_META_KEYS = frozenset(
    {
        "route_note",
        "safety_warning",
        "tripwire",
        "owner_state_restored",
        "light_repo_changed",
        "workspace_git_refs_changed",
    }
)
# Producer metadata keeps its exact limits. Composition may add only these
# closed, boolean host annotations, within a separately bounded byte reserve.
_MAX_HOST_META_BYTES = 256
_TOOL_RESULT_ATTR = "_active_builtin_tool_result"


@dataclass
class _ToolResultSlot:
    ctx: Any
    sentinel: object
    result: object


_TOOL_RESULT_STATE: ContextVar[_ToolResultSlot | None] = ContextVar(
    "ouroboros_builtin_tool_result",
    default=None,
)


@dataclass(frozen=True)
class ToolCodeSpec:
    """Stable meaning attached to one internal tool-result code."""

    status: ToolStatus
    outcome_bucket: str
    ui_severity: Literal["info", "warning", "error"]
    recovery: str

    def __post_init__(self) -> None:
        if self.status not in _TOOL_STATUSES:
            raise ValueError(f"invalid tool status: {self.status!r}")
        if not self.outcome_bucket:
            raise ValueError("outcome_bucket must be non-empty")
        if self.ui_severity not in {"info", "warning", "error"}:
            raise ValueError(f"invalid UI severity: {self.ui_severity!r}")
        if not isinstance(self.recovery, str) or not self.recovery:
            raise TypeError("recovery must be non-empty descriptive text")


def _code_spec(
    status: ToolStatus,
    outcome_bucket: str,
    ui_severity: Literal["info", "warning", "error"],
    recovery: str,
) -> ToolCodeSpec:
    return ToolCodeSpec(status, outcome_bucket, ui_severity, recovery)


TOOL_CODE_SPECS: Mapping[str, ToolCodeSpec] = MappingProxyType(
    {
        "OK": _code_spec("ok", "ok", "info", "none"),
        "SAFETY_WARNING": _code_spec(
            "ok",
            "ok",
            "warning",
            "review the safety warning before continuing",
        ),
        "SHELL_REGEX_AUTO_CORRECTED": _code_spec(
            "ok",
            "ok_autocorrected",
            "warning",
            "inspect the corrected command when relevant",
        ),
        "SHELL_NO_MATCH": _code_spec(
            "ok",
            "ok",
            "info",
            "none",
        ),
        "OWNER_STATE_RESTORED": _code_spec(
            "ok",
            "ok",
            "warning",
            "inspect the attempted owner-state mutation",
        ),
        "REVIEW_BLOCKED": _code_spec(
            "ok",
            "review_blocked",
            "warning",
            "address or rebut the review findings",
        ),
        "GIT_ERROR": _code_spec(
            "ok",
            "git_error",
            "warning",
            "inspect the version-control refusal",
        ),
        "LEGACY_WARNING": _code_spec(
            "ok",
            "ok",
            "warning",
            "inspect the warning before continuing",
        ),
        "LEGACY_UNTYPED": _code_spec(
            "ok",
            "untyped",
            "warning",
            "migrate the dynamic producer before consuming typed status",
        ),
        # Owner decision (batch #4, 2026-08-16): the cognitive redirect is a hint,
        # not a failure. It carried an `error` flag that the outcome classifier
        # then had to skip by name; the flag is removed at the source instead.
        "COGNITIVE_TOOL_REQUIRED": _code_spec(
            "ok",
            "ok",
            "warning",
            "use the cognitive tool named in the result text",
        ),
        "ACCESS_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "use an authority permitted by the task contract",
        ),
        "CORE_PROTECTION_BLOCKED": _code_spec(
            "blocked",
            "protected_blocked",
            "warning",
            "use the reviewed protected-write path",
        ),
        # The two demanded roots are DISTINCT codes and the merged `ROOT_REQUIRED`
        # parent is retired: the recovery walk credits a retry only when the write
        # lands on the root the redirect named, so a merged code makes that branch
        # unreachable, and leaving the parent published-by-nobody invites a future
        # reader to "simplify" the split back out.
        "ROOT_REQUIRED_USER_FILES": _code_spec(
            "blocked",
            "root_required_user_files",
            "warning",
            "retry the same write with root='user_files'",
        ),
        "ROOT_REQUIRED_ACTIVE_WORKSPACE": _code_spec(
            "blocked",
            "root_required_active_workspace",
            "warning",
            "retry the same write with root='active_workspace'",
        ),
        # Distinct for the same structural reason, and the merged `RESOURCE_BLOCKED`
        # parent retires with `ROOT_REQUIRED`: only these two names demote a
        # resource block on a read-only tool to ignored telemetry.
        "RESOURCE_CONSTRAINT_BLOCKED": _code_spec(
            "blocked",
            "resource_constraint_blocked",
            "warning",
            "use a resource the task contract allows",
        ),
        "RESOURCE_POLICY_BLOCKED": _code_spec(
            "blocked",
            "resource_policy_blocked",
            "warning",
            "use a resource the resource policy allows",
        ),
        "WORKSPACE_BLOCKED": _code_spec(
            "blocked",
            "workspace_blocked",
            "warning",
            "repair or select a valid workspace binding",
        ),
        "SHELL_CWD_BLOCKED": _code_spec(
            "blocked",
            "cwd_blocked",
            "warning",
            "select a working directory within an allowed root",
        ),
        "SUDO_INTERACTIVE_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "use non-interactive elevation or report the environment limitation",
        ),
        "SUBAGENT_SECRET_READ_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "use a gated read surface without accessing owner secrets",
        ),
        "ELEVATION_BLOCKED": _code_spec(
            "blocked",
            "elevation_blocked",
            "warning",
            "request an owner-controlled setting change",
        ),
        "CONTEXT_MODE_SELF_LOWERING_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "ask the owner to select the cognitive context mode",
        ),
        # The reference's scope-floor self-lowering code is deliberately absent:
        # that owner setting and its shell guard were retired outright in the
        # 7.0 ABI window (owner Q10=A), so the code has no producer on this
        # tree — and a code nothing publishes is a name a later reader mistakes
        # for a live outcome.
        "SAFETY_MODE_SELF_LOWERING_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "ask the owner to select the safety mode",
        ),
        "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "ask the owner to attest the eligible skill",
        ),
        "SKILL_STATE_WRITE_BLOCKED": _code_spec(
            "blocked",
            "skill_state_blocked",
            "warning",
            "use the reviewed skill lifecycle or owner controls",
        ),
        "GIT_VIA_SHELL_BLOCKED": _code_spec(
            "blocked",
            "git_via_shell_blocked",
            "warning",
            "use the reviewed repository mutation path for the Ouroboros runtime",
        ),
        "LIGHT_MODE_BLOCKED": _code_spec(
            "blocked",
            "light_mode_blocked",
            "warning",
            "use a permitted target or owner-selected mode",
        ),
        "LIGHT_MODE_REPO_WRITE_BLOCKED": _code_spec(
            "blocked",
            "light_mode_blocked",
            "warning",
            "use advanced or pro mode for repository writes",
        ),
        "WORKSPACE_GIT_REF_CHANGED": _code_spec(
            "blocked",
            "workspace_blocked",
            "warning",
            "leave external workspace changes as files or patch artifacts",
        ),
        "HEAL_MODE_BLOCKED": _code_spec(
            "blocked",
            "heal_mode_blocked",
            "warning",
            "stay within the admitted repair surface",
        ),
        "ARTIFACT_OUTPUT_UNDECLARED": _code_spec(
            "blocked",
            "artifact_output_undeclared",
            "warning",
            "declare the produced artifact",
        ),
        # Family parents. Each carries the legacy status of the family the loop
        # chain matched by text prefix, so the whole family keeps landing in the
        # policy-denial partition instead of degrading execution health.
        "INTEGRATION_BLOCKED": _code_spec(
            "blocked",
            "integration_blocked",
            "warning",
            "resolve the patch integration refusal before retrying",
        ),
        "WRITE_FILE_BLOCKED": _code_spec(
            "blocked",
            "write_file_blocked",
            "warning",
            "correct the write target or content and retry",
        ),
        "EDIT_TEXT_BLOCKED": _code_spec(
            "blocked",
            "edit_text_blocked",
            "warning",
            "re-read the file and retry the edit",
        ),
        "EDIT_OPS_BLOCKED": _code_spec(
            "blocked",
            "edit_ops_blocked",
            "warning",
            "re-read the file and retry the patch or batch",
        ),
        "DATA_BLOCKED": _code_spec(
            "blocked",
            "data_blocked",
            "warning",
            "use a data surface the task contract allows",
        ),
        "SKILL_PAYLOAD_BLOCKED": _code_spec(
            "blocked",
            "skill_payload_blocked",
            "warning",
            "write inside the selected skill payload",
        ),
        "USER_FILES_PATH_BLOCKED": _code_spec(
            "blocked",
            "user_files_path_blocked",
            "warning",
            "use a path inside the owner's user files",
        ),
        "RUN_SCRIPT_BLOCKED": _code_spec(
            "blocked",
            "run_script_blocked",
            "warning",
            "use a permitted interpreter and script path",
        ),
        "SAFETY_VIOLATION": _code_spec(
            "blocked",
            "safety_violation",
            "error",
            "choose a safe alternative",
        ),
        "LEGACY_BLOCKED": _code_spec(
            "blocked",
            "blocked",
            "warning",
            "follow the refusal text",
        ),
        "CAPABILITY_UNAVAILABLE": _code_spec(
            "unavailable",
            "unavailable",
            "warning",
            "enable or configure the capability",
        ),
        "MCP_UNAVAILABLE": _code_spec(
            "unavailable",
            "unavailable",
            "warning",
            "enable or repair the MCP provider",
        ),
        "EXTENSION_UNAVAILABLE": _code_spec(
            "unavailable",
            "unavailable",
            "warning",
            "enable or repair the extension",
        ),
        "LEGACY_UNAVAILABLE": _code_spec(
            "unavailable",
            "unavailable",
            "warning",
            "inspect availability and retry when restored",
        ),
        "TOOL_TIMEOUT": _code_spec(
            "timeout",
            "timeout",
            "error",
            "retry with a suitable bounded timeout",
        ),
        "MCP_TIMEOUT": _code_spec(
            "timeout",
            "timeout",
            "error",
            "inspect MCP health before retrying",
        ),
        "EXTENSION_TIMEOUT": _code_spec(
            "timeout",
            "timeout",
            "error",
            "inspect extension health before retrying",
        ),
        "TOOL_ARG_ERROR": _code_spec(
            "error",
            "argument_error",
            "error",
            "correct the tool arguments",
        ),
        "UNKNOWN_TOOL": _code_spec(
            "error",
            "unknown_tool",
            "error",
            "select a registered visible tool",
        ),
        "TOOL_ERROR": _code_spec(
            "error",
            "error",
            "error",
            "inspect the tool error and correct the call",
        ),
        "TOOL_INTERNAL_ERROR": _code_spec(
            "error",
            "error",
            "error",
            "inspect the internal tool contract",
        ),
        "EXECUTOR_ERROR": _code_spec(
            "error",
            "executor_error",
            "error",
            "inspect executor health and custody",
        ),
        "SHELL_ERROR": _code_spec(
            "error",
            "shell_error",
            "error",
            "correct the command or environment",
        ),
        "SHELL_EXIT_ERROR": _code_spec(
            "error",
            "non_zero_exit",
            "error",
            "inspect the command result before retrying",
        ),
        "RUN_SCRIPT_ERROR": _code_spec(
            "error",
            "run_script_error",
            "error",
            "correct the script or environment",
        ),
        "ARTIFACT_OUTPUT_ERROR": _code_spec(
            "error",
            "artifact_output_error",
            "error",
            "repair artifact registration",
        ),
        "MCP_ERROR": _code_spec(
            "error",
            "mcp_error",
            "error",
            "inspect the MCP response and provider health",
        ),
        "EXTENSION_ERROR": _code_spec(
            "error",
            "extension_error",
            "error",
            "inspect the extension response and health",
        ),
        # `SAFETY_ERROR` retires with them: its only publisher was the separator
        # count this change removed, and a code nothing publishes is a name a
        # later reader mistakes for a live outcome.
        "TOOL_REPORTED_FAILURE": _code_spec(
            "error",
            "tool_reported_failure",
            "error",
            "inspect the structured tool failure",
        ),
        "VLM_ERROR": _code_spec(
            "error",
            "vlm_error",
            "error",
            "inspect the image or vision-model requirement",
        ),
        "LEGACY_TOOL_ERROR": _code_spec(
            "error",
            "error",
            "error",
            "follow the legacy failure text",
        ),
        # Unreachable from core producers (only SAFETY_VIOLATION spells a
        # violation today, and it is claimed by an exact branch), but a skill or
        # MCP payload can emit an arbitrary ``⚠️ X_VIOLATION`` and the legacy
        # chain gave it its own status. Ported rather than folded into `error`.
        "LEGACY_VIOLATION": _code_spec(
            "error",
            "violation",
            "error",
            "follow the reported violation text",
        ),
    }
)


@dataclass(frozen=True)
class ToolResult:
    """Internal result; ``text`` remains the complete model-facing projection."""

    status: ToolStatus
    code: str
    text: str
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not _CODE_RE.fullmatch(self.code):
            raise ValueError(f"invalid tool result code: {self.code!r}")
        spec = TOOL_CODE_SPECS.get(self.code)
        if spec is None:
            raise ValueError(f"unknown tool result code: {self.code!r}")
        if self.status != spec.status:
            raise ValueError(f"status {self.status!r} does not match {self.code} ({spec.status!r})")
        if not isinstance(self.text, str):
            raise TypeError("tool result text must be a string")
        raw_meta = dict(self.meta or {})
        if any(not isinstance(key, str) for key in raw_meta):
            raise ValueError("tool result meta keys must be strings")
        producer_meta = {
            key: value for key, value in raw_meta.items() if key not in _HOST_META_KEYS
        }
        if len(producer_meta) > _MAX_META_ITEMS:
            raise ValueError("tool result meta must have at most 32 non-host keys")
        try:
            encoded = json.dumps(
                raw_meta,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            producer_encoded = json.dumps(
                producer_meta,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (RecursionError, TypeError, ValueError) as exc:
            raise TypeError("tool result meta must contain JSON-safe values") from exc
        producer_bytes = len(producer_encoded.encode("utf-8"))
        if producer_bytes > _MAX_META_BYTES:
            raise ValueError("tool result meta exceeds 8192 encoded bytes")
        # Keep every payload valid under the old aggregate cap, then permit
        # only the fixed host reserve beyond it as producer bytes approach it.
        total_limit = max(_MAX_META_BYTES, producer_bytes + _MAX_HOST_META_BYTES)
        if len(encoded.encode("utf-8")) > total_limit:
            raise ValueError("tool result host metadata exceeds its reserved overhead")
        object.__setattr__(self, "meta", MappingProxyType(json.loads(encoded)))


def _replace_tool_result(
    result: ToolResult,
    *,
    text: str | None = None,
    code: str | None = None,
    meta_updates: Mapping[str, Any] | None = None,
) -> ToolResult:
    """Replace immutable result fields without re-adapting its trusted facts."""

    selected_code = code or result.code
    meta = dict(result.meta)
    meta.update(dict(meta_updates or {}))
    return ToolResult(
        status=TOOL_CODE_SPECS[selected_code].status,
        code=selected_code,
        text=result.text if text is None else text,
        meta=meta,
    )


def _publish_tool_result(ctx: Any, result: ToolResult) -> str:
    """Publish one registry-scoped builtin result while keeping the string ABI."""

    active = _TOOL_RESULT_STATE.get()
    if active is not None and active.ctx is ctx:
        active.result = result
    elif hasattr(ctx, _TOOL_RESULT_ATTR):
        setattr(ctx, _TOOL_RESULT_ATTR, result)
    return result.text


def _install_tool_result_sidecar(
    ctx: Any,
    sentinel: object,
) -> Token:
    """Install one context-local builtin result sentinel."""
    return _TOOL_RESULT_STATE.set(_ToolResultSlot(ctx, sentinel, sentinel))


def _published_tool_result(ctx: Any, sentinel: object) -> object:
    """Read the result published by the current builtin invocation."""
    active = _TOOL_RESULT_STATE.get()
    if active is not None:
        if active.ctx is not ctx:
            return sentinel
        if sentinel is not None and active.sentinel is not sentinel:
            return sentinel
        if active.result is active.sentinel:
            return sentinel
        return active.result
    return getattr(ctx, _TOOL_RESULT_ATTR, sentinel)


def _restore_tool_result_sidecar(token: Token) -> None:
    """Restore the enclosing invocation's context-local sidecar."""
    _TOOL_RESULT_STATE.reset(token)


def _publish_process_result(
    ctx: Any,
    code: str,
    text: str,
    *,
    exit_code: int | None = None,
    signal_name: str = "",
    artifact_registered: bool = False,
    shell_regex_auto_corrected: bool = False,
    meta: Mapping[str, Any] | None = None,
) -> str:
    """Publish trusted process facts through the transient string-bound sidecar."""

    facts = dict(meta or {})
    if exit_code is not None:
        facts["exit_code"] = int(exit_code)
        if int(exit_code) < 0 and not signal_name:
            signal_name = posix_signal_name(abs(int(exit_code)))
    if signal_name:
        facts["signal"] = signal_name
    if artifact_registered:
        facts["artifact_registered"] = True
    if shell_regex_auto_corrected:
        facts["shell_regex_auto_corrected"] = True
    return _publish_tool_result(
        ctx,
        ToolResult(
            status=TOOL_CODE_SPECS[code].status,
            code=code,
            text=text,
            meta=facts,
        ),
    )


def _wrap_run_script_process_result(
    ctx: Any,
    result: str,
    audit_note: str,
    script_path: Any,
) -> str:
    """Republish an inner shell result after the exact run-script text wrapper."""

    if str(result).lstrip().startswith("⚠️"):
        # The result already owns line 1 with its own typed marker, which is what
        # the failure classifier reads — the nudge appends after it.
        tail = f"\n{audit_note}" if audit_note else ""
        wrapped = f"{result}{tail}\n# script_path={script_path}"
    elif audit_note:
        # The nudge used to REPLACE the whole _run_shell payload (a successful
        # script's answer was gone; re-running was the sole recovery). Marker
        # first — ARTIFACT_OUTPUT_UNDECLARED is a typed policy-denial surface the
        # classifier reads off line 1 — payload appended, as in run_command.
        wrapped = f"{audit_note}\n\n# script_path={script_path}\n{result}"
    else:
        wrapped = f"# script_path={script_path}\n{result}"
    base = _published_tool_result(ctx, None)
    if isinstance(base, ToolResult) and base.text == result:
        code = "ARTIFACT_OUTPUT_UNDECLARED" if audit_note and base.status == "ok" else base.code
        return _publish_tool_result(
            ctx,
            _replace_tool_result(base, text=wrapped, code=code),
        )
    return wrapped


_EXACT_IDENTIFIER_CODES = MappingProxyType(
    {
        "ACCESS_DENIED": "ACCESS_BLOCKED",
        "TOOL_ACCESS_BLOCKED": "ACCESS_BLOCKED",
        "ACTING_NO_WORKSPACE_BLOCKED": "ACCESS_BLOCKED",
        "ACTING_SUBAGENT_BLOCKED": "ACCESS_BLOCKED",
        "ACTING_SUBAGENT_TOOL_NOT_GRANTED": "ACCESS_BLOCKED",
        "LOCAL_READONLY_SUBAGENT_BLOCKED": "ACCESS_BLOCKED",
        "MUTATIVE_SUBAGENTS_DISABLED": "ACCESS_BLOCKED",
        "CORE_PROTECTION_BLOCKED": "CORE_PROTECTION_BLOCKED",
        "ROOT_REQUIRED_USER_FILES": "ROOT_REQUIRED_USER_FILES",
        "ROOT_REQUIRED_ACTIVE_WORKSPACE": "ROOT_REQUIRED_ACTIVE_WORKSPACE",
        "USER_FILES_PATH_BLOCKED": "USER_FILES_PATH_BLOCKED",
        "COGNITIVE_TOOL_REQUIRED": "COGNITIVE_TOOL_REQUIRED",
        "RESOURCE_CONSTRAINT_BLOCKED": "RESOURCE_CONSTRAINT_BLOCKED",
        "RESOURCE_POLICY_BLOCKED": "RESOURCE_POLICY_BLOCKED",
        "DATA_READ_BLOCKED": "DATA_BLOCKED",
        "DATA_LIST_BLOCKED": "DATA_BLOCKED",
        "WORKSPACE_MODE_BLOCKED": "WORKSPACE_BLOCKED",
        "WORKSPACE_GIT_BLOCKED": "WORKSPACE_BLOCKED",
        "WORKSPACE_SHELL_BLOCKED": "WORKSPACE_BLOCKED",
        "WORKSPACE_EXECUTOR_STATE_WRITE_BLOCKED": "WORKSPACE_BLOCKED",
        "LIGHT_MODE_BLOCKED": "LIGHT_MODE_BLOCKED",
        "LIGHT_MODE_REPO_WRITE_BLOCKED": "LIGHT_MODE_REPO_WRITE_BLOCKED",
        "WORKSPACE_GIT_REF_CHANGED": "WORKSPACE_GIT_REF_CHANGED",
        "OWNER_STATE_RESTORED": "OWNER_STATE_RESTORED",
        "HEAL_MODE_BLOCKED": "HEAL_MODE_BLOCKED",
        "ARTIFACT_OUTPUT_UNDECLARED": "ARTIFACT_OUTPUT_UNDECLARED",
        "ARTIFACT_OUTPUT_ERROR": "ARTIFACT_OUTPUT_ERROR",
        "SAFETY_VIOLATION": "SAFETY_VIOLATION",
        "CAPABILITY_UNAVAILABLE": "CAPABILITY_UNAVAILABLE",
        "MCP_DISABLED": "MCP_UNAVAILABLE",
        "MCP_TOOL_NOT_FOUND": "MCP_UNAVAILABLE",
        "MCP_TOOL_DISALLOWED": "ACCESS_BLOCKED",
        "MCP_TOOL_TIMEOUT": "MCP_TIMEOUT",
        "MCP_TOOL_ERROR": "MCP_ERROR",
        "TOOL_TIMEOUT": "TOOL_TIMEOUT",
        # #440 fix-forward markers (upstream failure prefixes, typed here): an
        # abandoned call's void result is the call's own timeout; the
        # hung-session backlog refusal is browser unavailability.
        "BROWSER_SESSION_RETIRED": "TOOL_TIMEOUT",
        "BROWSER_BACKLOG_RETIRED_SESSIONS": "LEGACY_UNAVAILABLE",
        "TOOL_ARG_ERROR": "TOOL_ARG_ERROR",
        "INVALID_ARG": "TOOL_ARG_ERROR",
        "TOOL_ERROR": "TOOL_ERROR",
        "TOOL_INTERNAL_ERROR": "TOOL_INTERNAL_ERROR",
        "EXECUTOR_UNAVAILABLE": "LEGACY_UNAVAILABLE",
        "SHELL_EXIT_ERROR": "SHELL_EXIT_ERROR",
        "SHELL_NO_MATCH": "SHELL_NO_MATCH",
        "SHELL_ERROR": "SHELL_ERROR",
        "SHELL_ARG_ERROR": "SHELL_ERROR",
        "SHELL_CMD_ERROR": "SHELL_ERROR",
        "SHELL_CWD_BLOCKED": "SHELL_CWD_BLOCKED",
        "SUDO_INTERACTIVE_BLOCKED": "SUDO_INTERACTIVE_BLOCKED",
        "SUBAGENT_SECRET_READ_BLOCKED": "SUBAGENT_SECRET_READ_BLOCKED",
        "ELEVATION_BLOCKED": "ELEVATION_BLOCKED",
        "CONTEXT_MODE_SELF_LOWERING_BLOCKED": "CONTEXT_MODE_SELF_LOWERING_BLOCKED",
        "SAFETY_MODE_SELF_LOWERING_BLOCKED": "SAFETY_MODE_SELF_LOWERING_BLOCKED",
        "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED": "OWNER_SKILL_ATTESTATION_SELF_CALL_BLOCKED",
        "SKILL_STATE_WRITE_BLOCKED": "SKILL_STATE_WRITE_BLOCKED",
        "GIT_VIA_SHELL_BLOCKED": "GIT_VIA_SHELL_BLOCKED",
        "RUN_SCRIPT_BLOCKED": "RUN_SCRIPT_BLOCKED",
        "REVIEW_BLOCKED": "REVIEW_BLOCKED",
        "GIT_ERROR": "GIT_ERROR",
    }
)

# ORDERED families, consulted only after the exact table above so a specific
# identifier always beats its family (``SHELL_CWD_BLOCKED`` before ``SHELL_``,
# ``RUN_SCRIPT_BLOCKED`` before ``RUN_SCRIPT_``) and before the generic marker
# fallbacks below, exactly as the legacy loop chain ordered its branches. Order
# inside the tuple is load-bearing only where one prefix prefixes another; it is
# kept in the legacy chain's order regardless so the two can be diffed by eye.
_FAMILY_PREFIX_CODES: tuple[tuple[str, str], ...] = (
    ("SHELL_", "SHELL_ERROR"),
    ("RUN_SCRIPT_", "RUN_SCRIPT_ERROR"),
    ("VLM_", "VLM_ERROR"),
    ("INTEGRATE_", "INTEGRATION_BLOCKED"),
    ("WORKSPACE_", "WORKSPACE_BLOCKED"),
    ("ELEVATION_", "ELEVATION_BLOCKED"),
    ("SKILL_STATE_", "SKILL_STATE_WRITE_BLOCKED"),
    ("SKILL_REDIRECT_", "SKILL_PAYLOAD_BLOCKED"),
    ("SKILL_PAYLOAD_ARG_", "SKILL_PAYLOAD_BLOCKED"),
    ("DATA_WRITE_", "DATA_BLOCKED"),
    ("WRITE_FILE_", "WRITE_FILE_BLOCKED"),
    ("EDIT_TEXT_", "EDIT_TEXT_BLOCKED"),
    ("APPLY_PATCH_", "EDIT_OPS_BLOCKED"),
    ("EDIT_BATCH_", "EDIT_OPS_BLOCKED"),
)


def _compose_execute_result(result: str, route_note: str, safety_msg: str) -> str:
    """Assemble the final tool result.

    ALL host notes TRAIL the result (#447 В12/H1). Failure classification reads
    the producer's typed result first and the payload's FIRST line as fallback,
    so neither the auto-route note nor the safety warning may own line 1 — a
    leading SAFETY_WARNING used to reclassify a failed command (or an
    extension's ``{"ok": false}`` answer) as ok on every text-only reader."""
    if route_note:
        result = f"{result}\n\n{route_note}"
    if safety_msg:
        result = f"{result}\n\n{safety_msg}"
    return result


def _structured_failure(text: str) -> bool:
    """Whether a tool's own JSON payload declares the call failed (``ok: false``).

    Deliberately narrow: the payload must be a JSON OBJECT whose top-level ``ok``
    is exactly False. Since #447 H1 host notes TRAIL the payload, so the whole
    text is no longer parseable once a note rides along — the pre-note payload is
    tried as well, which is exactly what a typed producer would have published.
    """
    stripped = text.lstrip()
    if not stripped.startswith("{") or '"ok"' not in stripped:
        return False
    candidates = [stripped]
    head, sep, _tail = stripped.partition(_HOST_NOTE_SEPARATOR)
    if sep:
        candidates.append(head)
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:  # The compatibility adapter must never break a result.
            continue
        if isinstance(payload, dict) and payload.get("ok") is False:
            return True
    return False


def _classification(code: str, meta: Mapping[str, Any] | None = None) -> tuple[ToolStatus, str, dict[str, Any]]:
    spec = TOOL_CODE_SPECS[code]
    return spec.status, code, dict(meta or {})


def _classify_legacy_text(
    text: str,
    *,
    wrapper_depth: int = 0,
) -> tuple[ToolStatus, str, dict[str, Any]]:
    if wrapper_depth >= 4:
        return _classification(
            "LEGACY_TOOL_ERROR",
            {"wrapper_depth_exceeded": True},
        )
    if text.startswith("⚠️ SAFETY_WARNING"):
        # No separator counting: the wrapper reveals whatever it wraps. Counting
        # ``---`` runs in a producer-controlled body turned a legitimate result
        # that merely contains the separator into a safety-provider failure, and
        # the count is a host fact only where the host itself composed the text
        # (``_compose_execute_result_result``), which still asserts it.
        _warning, separator, inner = text.partition(_SAFETY_SEPARATOR)
        if separator:
            status, code, meta = _classify_legacy_text(
                inner,
                wrapper_depth=wrapper_depth + 1,
            )
            if code != "OK":
                return status, code, {**meta, "safety_warning": True}
        return _classification("SAFETY_WARNING")

    if text.startswith("⚠️ SHELL_REGEX_AUTO_CORRECTED"):
        _warning, separator, inner = text.partition("\n")
        if separator:
            status, code, meta = _classify_legacy_text(
                inner,
                wrapper_depth=wrapper_depth + 1,
            )
            if status != "ok":
                return status, code, {**meta, "shell_regex_auto_corrected": True}
        return _classification("SHELL_REGEX_AUTO_CORRECTED")

    if _structured_failure(text):
        return _classification("TOOL_REPORTED_FAILURE")
    if text.startswith("⚠️ CRITICAL SAFETY_VIOLATION"):
        return _classification("SAFETY_VIOLATION")
    if text.startswith(_UNKNOWN_TOOL_PREFIX):
        return _classification("UNKNOWN_TOOL")

    first_line = text.splitlines()[0] if text else ""
    marker = _FIRST_MARKER_RE.match(first_line)
    if marker is None:
        return _classification("OK")
    identifier = marker.group(1)
    exact = _EXACT_IDENTIFIER_CODES.get(identifier)
    if exact is not None:
        return _classification(exact)
    if identifier.startswith("MCP_"):
        if "TIMEOUT" in identifier:
            return _classification("MCP_TIMEOUT")
        if "UNAVAILABLE" in identifier or "NOT_FOUND" in identifier or "DISABLED" in identifier:
            return _classification("MCP_UNAVAILABLE")
        return _classification("MCP_ERROR")
    if identifier.startswith("EXTENSION_"):
        if "TIMEOUT" in identifier:
            return _classification("EXTENSION_TIMEOUT")
        if "UNAVAILABLE" in identifier or "NOT_FOUND" in identifier or "DISABLED" in identifier:
            return _classification("EXTENSION_UNAVAILABLE")
        return _classification("EXTENSION_ERROR")
    for prefix, family_code in _FAMILY_PREFIX_CODES:
        if identifier.startswith(prefix):
            return _classification(family_code)
    if "_TIMEOUT" in identifier:
        return _classification("TOOL_TIMEOUT")
    if "_UNAVAILABLE" in identifier:
        return _classification("LEGACY_UNAVAILABLE")
    if any(part in identifier for part in ("_BLOCKED", "_FORBIDDEN", "_DISALLOWED")):
        return _classification("LEGACY_BLOCKED")
    if "_VIOLATION" in identifier:
        return _classification("LEGACY_VIOLATION")
    if any(part in identifier for part in ("_ERROR", "_FAILED", "_CORRUPT")):
        return _classification("LEGACY_TOOL_ERROR")
    return _classification("LEGACY_WARNING")


class LegacyTextResultAdapter:
    """One compatibility adapter from host-owned legacy text to ``ToolResult``."""

    @classmethod
    def from_text(cls, tool_name: str, text: str) -> ToolResult:
        if not isinstance(text, str):
            text = str(text)
        normalized_name = str(tool_name or "").strip()
        # The name shape is a PROXY for "a provider composed this body", and the
        # unknown-tool sentence is the one text where the proxy is wrong: the
        # registry writes it because the name resolved to no provider at all, so a
        # hallucinated ``ext_``/``mcp_`` name — or one whose extension was
        # unloaded — would otherwise be a nameless success. Owner item A.1 ("a
        # call to a tool that does not exist was never a success") holds for every
        # name shape, and it holds through the ONE chain below, not a second table.
        dynamic_body_untyped = not text.startswith(_UNKNOWN_TOOL_PREFIX) and (
            normalized_name.startswith("ext_")
            or (
                normalized_name.startswith("mcp_")
                and text.startswith(_MCP_RESULT_ENVELOPE_PREFIX)
            )
        )
        if dynamic_body_untyped:
            # A dynamic body still gets the ONE structured self-report read: a
            # provider that answered ``{"ok": false}`` declared its own failure,
            # which is a host-readable fact and not a re-typing of untrusted
            # prose. An MCP body arrives behind the server envelope, so its
            # payload never starts the string and stays untyped by construction.
            if _structured_failure(text):
                return ToolResult(
                    status="error",
                    code="TOOL_REPORTED_FAILURE",
                    text=text,
                    meta={"dynamic_provider": True},
                )
            return ToolResult(
                status="ok",
                code="LEGACY_UNTYPED",
                text=text,
                meta={"dynamic_provider": True},
            )
        status, code, meta = _classify_legacy_text(text)
        return ToolResult(status=status, code=code, text=text, meta=meta)


def _compose_execute_result_result(
    tool_name: str,
    base: str | ToolResult,
    route_note: str,
    safety_msg: str,
) -> ToolResult:
    """Compose host-owned annotations without re-adapting a typed base result."""

    base_result = (
        base
        if isinstance(base, ToolResult)
        else LegacyTextResultAdapter.from_text(tool_name, base)
    )
    text = _compose_execute_result(base_result.text, route_note, safety_msg)
    meta = dict(base_result.meta)
    if route_note:
        meta["route_note"] = True
    # ``ambiguous_safety_wrapper`` retired with the leading wrapper (#447 H1):
    # the host no longer inserts a ``---`` separator around the payload, so a
    # separator in the composed text is the producer's own markdown rule and
    # carries no ambiguity about who wrapped whom. Which notes rode along stays
    # disclosed by the existing ``route_note``/``safety_warning`` host facts —
    # their TEXT is in the result itself and does not enter the bounded host
    # meta reserve (a full warning would blow the 256-byte ceiling).
    if not safety_msg:
        return ToolResult(
            status=base_result.status,
            code=base_result.code,
            text=text,
            meta=meta,
        )
    if base_result.code == "OK":
        return ToolResult(
            status="ok",
            code="SAFETY_WARNING",
            text=text,
            meta=meta,
        )
    meta["safety_warning"] = True
    return ToolResult(
        status=base_result.status,
        code=base_result.code,
        text=text,
        meta=meta,
    )
