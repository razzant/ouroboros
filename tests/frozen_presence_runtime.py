"""Symbolic runtime settings for reviewed presence profiles."""

from __future__ import annotations

from dataclasses import dataclass

_MODEL_SLOTS = frozenset({"main", "light"})
# How an admitted turn runs: ``event`` is the bounded text turn; ``call`` lets the
# binding open one live audio session (a long turn under the same ceiling).
EXECUTION_EVENT = "event"
EXECUTION_CALL = "call"
_EXECUTIONS = frozenset({EXECUTION_EVENT, EXECUTION_CALL})


class PresenceRuntimeError(ValueError):
    """A presence runtime field failed strict validation."""

    def __init__(self, code: str, field: str) -> None:
        self.code = code
        self.field = field
        super().__init__(f"{code}: {field}")


def _validate_model_slot(value: object, *, field: str = "model_slot") -> str:
    if not isinstance(value, str) or value not in _MODEL_SLOTS:
        raise PresenceRuntimeError("invalid_model_slot", field)
    return value


def _validate_positive_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PresenceRuntimeError("invalid_positive_int", field)
    return value


def validate_execution(value: object, *, field: str = "execution") -> str:
    """One of ``event`` / ``call``; anything else is a typed refusal, never a default."""
    if not isinstance(value, str) or value not in _EXECUTIONS:
        raise PresenceRuntimeError("invalid_execution", field)
    return value


@dataclass(frozen=True)
class PresenceRuntimeDefaults:
    """Reviewed symbolic defaults declared by a presence profile."""

    model_slot: str = "main"
    inline_max_rounds: int = 10
    execution: str = EXECUTION_EVENT

    def __post_init__(self) -> None:
        _validate_model_slot(self.model_slot)
        _validate_positive_int(self.inline_max_rounds, field="inline_max_rounds")
        validate_execution(self.execution)


@dataclass(frozen=True)
class PresenceRuntimeOverrides:
    """Owner-local overrides; ``None`` keeps the reviewed default."""

    model_slot: str | None = None
    inline_max_rounds: int | None = None
    execution: str | None = None

    def __post_init__(self) -> None:
        if self.model_slot is not None:
            _validate_model_slot(self.model_slot)
        if self.inline_max_rounds is not None:
            _validate_positive_int(self.inline_max_rounds, field="inline_max_rounds")
        if self.execution is not None:
            validate_execution(self.execution)


@dataclass(frozen=True)
class ResolvedPresenceRuntime:
    """Effective symbolic runtime values frozen for one admission."""

    model_slot: str
    requested_inline_max_rounds: int
    inline_max_rounds: int
    capped: bool
    execution: str = EXECUTION_EVENT

    def __post_init__(self) -> None:
        validate_execution(self.execution)


_BUILTIN_DEFAULTS = PresenceRuntimeDefaults()


def resolve_presence_runtime(
    defaults: PresenceRuntimeDefaults | None,
    overrides: PresenceRuntimeOverrides | None,
    *,
    global_max_rounds: int | None,
) -> ResolvedPresenceRuntime:
    """Resolve owner override, reviewed default, then built-in fallback.

    The inline cap stays finite: ``global_max_rounds`` (the task round limit) only
    narrows it, and ``None`` — no global limit — leaves the reviewed/owner value."""

    if defaults is None:
        defaults = _BUILTIN_DEFAULTS
    elif not isinstance(defaults, PresenceRuntimeDefaults):
        raise PresenceRuntimeError("invalid_defaults", "defaults")

    if overrides is None:
        overrides = PresenceRuntimeOverrides()
    elif not isinstance(overrides, PresenceRuntimeOverrides):
        raise PresenceRuntimeError("invalid_overrides", "overrides")

    maximum = (None if global_max_rounds is None
               else _validate_positive_int(global_max_rounds, field="global_max_rounds"))
    model_slot = overrides.model_slot or defaults.model_slot
    requested_rounds = (
        overrides.inline_max_rounds if overrides.inline_max_rounds is not None else defaults.inline_max_rounds
    )
    effective_rounds = requested_rounds if maximum is None else min(requested_rounds, maximum)
    return ResolvedPresenceRuntime(
        model_slot=model_slot,
        requested_inline_max_rounds=requested_rounds,
        inline_max_rounds=effective_rounds,
        capped=effective_rounds != requested_rounds,
        execution=overrides.execution or defaults.execution,
    )
