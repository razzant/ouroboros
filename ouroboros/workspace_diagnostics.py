"""Typed internal execution evidence with compatible public rendering."""

from __future__ import annotations

import errno as errno_module
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

# The register of "which owner action removes this refusal". Imported at module
# scope, not lazily: `RemoteWorkspaceError.__init__` consults it on every raise,
# and the register is stdlib-only and travels into the execd bundle exactly like
# this module does, so there is no cycle and no Home dependency to defer.
from ouroboros.remote_refusal_actions import REFUSAL_ACTIONS

DiagnosticDomain = Literal[
    "transport", "protocol", "policy", "filesystem", "process", "artifact"
]
CompletionState = Literal["not_started", "completed", "unknown"]

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)"
    r"\b\s*[:=]\s*)([\"']?)([^\s,\"']+)(\2)"
)
_BEARER_RE = re.compile(r"(?i)(\b(?:authorization\s*:\s*)?bearer\s+)[A-Za-z0-9._~+/=-]+")
_URL_SECRET_RE = re.compile(
    r"(?i)([?&](?:token|access_token|api_key|key|secret|password)=)[^&#\s]+"
)


def sanitize_execution_text(value: Any) -> str:
    """Dependency-light wire/log scrubber that preserves diagnostic structure."""

    text = str(value if value is not None else "")
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _URL_SECRET_RE.sub(r"\1[REDACTED]", text)
    return _SECRET_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_execution_text(value)
    if isinstance(value, dict):
        return {sanitize_execution_text(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_sanitize_value(item) for item in value)
    return value


@dataclass
class ProcessExecutionResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""
    backend_trace: dict[str, Any] = field(default_factory=dict)
    args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stdout = sanitize_execution_text(self.stdout)
        self.stderr = sanitize_execution_text(self.stderr)
        self.backend_trace = _sanitize_value(dict(self.backend_trace))
        self.args = [sanitize_execution_text(item) for item in self.args]


@dataclass(frozen=True)
class ExecutionDiagnostic:
    domain: DiagnosticDomain
    code: str
    message: str
    phase: str
    request_id: str = ""
    operation_id: str = ""
    completion: CompletionState = "not_started"
    retryable: bool = False
    errno: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "message", sanitize_execution_text(self.message))
        object.__setattr__(self, "details", _sanitize_value(dict(self.details)))


@dataclass(frozen=True)
class ToolExecutionEnvelope:
    text: str
    diagnostic: ExecutionDiagnostic | None = None
    process: ProcessExecutionResult | None = None
    artifacts: tuple[dict[str, Any], ...] = ()
    trace: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", sanitize_execution_text(self.text))
        object.__setattr__(
            self,
            "artifacts",
            tuple(_sanitize_value(dict(item)) for item in self.artifacts),
        )
        object.__setattr__(self, "trace", _sanitize_value(dict(self.trace)))


WorkspaceExecutionEnvelope = ToolExecutionEnvelope


class RemoteWorkspaceError(RuntimeError):
    """Typed nonsecret remote-placement error that renders as a diagnostic.

    It lives beside :class:`ExecutionDiagnostic` rather than in the broker
    because every layer that can raise it — the pending-operation journal, the
    OpenSSH transport, the broker, the worker proxy — must be able to construct
    it without importing a layer above itself.  The donor reached the class
    through a function-local ``remote_workspace`` import from inside the
    transport, which is the module cycle this placement removes.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        phase: str,
        completion: CompletionState = "not_started",
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = str(code)
        self.phase = str(phase)
        self.completion = str(completion)
        self.retryable = bool(retryable)
        self.details = dict(details or {})
        # The owner's ACTION, derived from the details rather than added as a second
        # parameter every raiser would have to remember. The Connections projection
        # reads `getattr(error, "action", "retry")`, so before this every typed
        # transport refusal — including "your execd is too old to talk to" — told the
        # owner to RETRY, which is the one thing that cannot work. A refusal that
        # names its action in `details` (the only slot that already reaches the
        # browser and `--json`) now says so at the attribute the surface reads.
        #
        # …and when it does NOT name one, the CODE does, through the register in
        # `remote_refusal_actions`. `details` stayed the only carrier for a while, so
        # the ~40 raise sites in this feature that pass no details all reported
        # `retry` — fourteen of them for codes the owner CLI itself maps to exit 4,
        # "retrying will not help". Deriving from the code fixes every one of them
        # without touching a raise site, and makes "which action removes this" a
        # property of the refusal rather than of whoever remembered to say so.
        # Normalized to the wire spelling on lookup: `code` is raw here, and two of
        # its authorities are upper-case Python constants.
        self.action = str(
            self.details.get("action")
            or REFUSAL_ACTIONS.get(self.code.strip().lower())
            or "retry"
        )
        super().__init__(str(message))

    def diagnostic(
        self,
        *,
        request_id: str = "",
        operation_id: str = "",
    ) -> ExecutionDiagnostic:
        domain: DiagnosticDomain = (
            "transport"
            if self.phase in {"connect", "bootstrap", "stream"}
            else "protocol"
        )
        # The derived action has to be IN the projection, not only on the attribute.
        # `details` is what the comment above calls the only slot that already reaches
        # the browser and `--json`, and this method used to serialize `self.details`
        # UNCHANGED — so a refusal that NAMED its action carried it and the ~40 that
        # derive one from the code carried nothing, which is exactly the case the
        # derivation was written for. It is written here rather than in `__init__` so
        # `self.details` stays the raiser's own words (the durable journal and the
        # private receipt record it verbatim) and the PROJECTION owns the derived fact.
        # Idempotent by construction: when a raiser did name an action, `self.action`
        # IS that string.
        details = dict(self.details)
        details["action"] = self.action
        return ExecutionDiagnostic(
            domain=domain,
            code=self.code,
            message=str(self),
            phase=self.phase,
            request_id=request_id,
            operation_id=operation_id,
            completion=self.completion,  # type: ignore[arg-type]
            retryable=self.retryable,
            details=details,
        )


_ERRNO_CODES = {
    errno_module.ENOENT: "not_found",
    errno_module.EACCES: "permission_denied",
    errno_module.EPERM: "permission_denied",
    errno_module.ENOTDIR: "not_a_directory",
    errno_module.EISDIR: "is_a_directory",
    errno_module.ENOSPC: "no_space",
    errno_module.EROFS: "read_only_filesystem",
}


def diagnostic_from_exception(
    exc: BaseException,
    *,
    request_id: str,
    operation_id: str = "",
    phase: str,
    domain: DiagnosticDomain = "filesystem",
    completion: CompletionState = "not_started",
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> ExecutionDiagnostic:
    """Preserve native errno distinctions without parsing rendered strings."""

    native_errno = getattr(exc, "errno", None)
    try:
        errno_value = int(native_errno) if native_errno is not None else None
    except (TypeError, ValueError):
        errno_value = None
    code = _ERRNO_CODES.get(errno_value, "operation_failed")
    message = str(exc or type(exc).__name__).strip() or type(exc).__name__
    return ExecutionDiagnostic(
        domain=domain,
        code=code,
        message=message[:2000],
        phase=str(phase or "execute"),
        request_id=str(request_id or ""),
        operation_id=str(operation_id or ""),
        completion=completion,
        retryable=bool(retryable),
        errno=errno_value,
        details=dict(details or {}),
    )


def render_diagnostic_text(
    diagnostic: ExecutionDiagnostic,
    *,
    prefix: str = "REMOTE_WORKSPACE_ERROR",
) -> str:
    """Stable fallback text when a typed failure has no legacy rendering."""

    completion = (
        f", completion={diagnostic.completion}"
        if diagnostic.completion != "not_started"
        else ""
    )
    return (
        f"⚠️ {prefix} [{diagnostic.code}] during {diagnostic.phase}"
        f"{completion}: {diagnostic.message}"
    )
