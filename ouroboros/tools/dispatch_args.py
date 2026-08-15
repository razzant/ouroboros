"""The three argument projections of one dispatch (RWS v2 §3.1 step 3).

A tool call carries THREE different argument sets, and today only one of them
has a name:

* ``handler_args`` — the public tool schema, exactly what the handler is bound
  to (post path-arg normalization, post python pre-resolution).
* ``guard_args`` — the shell-guard projection (``process_shell_guard_args``).
  It is NOT the handler set and never was: it adds ``__tool_name``, it drops
  ``verify_and_record``'s contract fields, and it rewrites ``run_script``'s
  ``interpreter``/``script`` pair into an inline ``[interpreter, "-c", script]``
  argv so the guards inspect the script BODY.
* ``execution_args`` — the target-canonical command actually launched: argv plus
  the canonicalized cwd and the resource root it resolved under.

The postmortem shape (codex #2) is a pipeline that cannot say which of the three
it is holding: the guard projection gets passed around as "the args", and one
rename later the authorized set and the executed set are different objects that
merely look alike. So the dispatcher builds ONE immutable record naming all
three, the divergences between them are DECLARED here rather than discovered,
and the equality contracts are tested per pair.

Canonicalization of the cwd goes through the placement-neutral fact accessor
(``execution_facts``), so ``execution_args.cwd`` is a TARGET spelling: for a
local placement that is today's ``resolve(strict=False)`` byte-for-byte, and for
an ssh placement it is the target-canonical path the operation's ONE prepare
resolved — passed in by the caller that holds the prepare bundle, never
re-derived here and never a Home path pretending to be a target one.
"""

from __future__ import annotations

import dataclasses
import pathlib
from types import MappingProxyType
from typing import Any, Mapping

from ouroboros.execution_facts import LOCAL_FACTS
from ouroboros.shell_parse import normalize_check_argv
from ouroboros.tool_access_policy import _ALL_ROOTS
from ouroboros.tool_capabilities import ROOT_AFFINITY_TOOL_NAMES, dispatch_resource_root
from ouroboros.tool_access import resolve_shell_cwd
from ouroboros.tools.shell_guards import (
    CWD_BEARING_TOOLS,
    operation_cwd_keys,
    process_shell_guard_args,
    shell_argv,
)

# Declared, intentional differences between the guard projection and the handler
# set. Anything NOT listed here is a desync, not a design choice — the tests
# assert this table rather than trusting the reader to remember it.
GUARD_PROJECTION_DIVERGENCES: dict[str, tuple[str, ...]] = {
    "run_command": ("adds___tool_name",),
    "start_service": ("adds___tool_name",),
    "verify_and_record": ("adds___tool_name", "drops_contract_fields", "check_normalized_to_argv"),
    "run_script": ("adds___tool_name", "interpreter_script_rewritten_to_inline_argv"),
}

# run_script executes the guarded script BODY from a staged file, so its guard
# argv (inline ``-c``) and its launched argv (interpreter + staged path) differ
# by construction. The invariant that survives is the pair (interpreter, body).
_INLINE_BODY_TOOLS = frozenset({"run_script"})


@dataclasses.dataclass(frozen=True)
class ExecutionArgs:
    """The target-canonical command of one operation.

    ``paths``/``resource_root`` are here for the same reason ``argv`` is. The
    prepared token binds THIS record, so whatever the record omits, the token does
    not commit to — and while the record held only argv and cwd, the hash of a
    `read_file` was a constant per tool. A target could hand back its own
    ``execution_args`` naming a different file and both sides still agreed with
    themselves: Home authorized `write_file path="ok.txt"` and `PWNED.txt` was
    written, `edit_text path="hello.txt"` edited `victim.txt`, `read_file
    path="hello.txt"` returned `SECRET.txt`. The path arguments of a root-labelled
    tool ARE its command, so they are bound like one.
    """

    tool: str
    argv: tuple[str, ...] = ()
    cwd: str = ""
    cwd_root: str = ""
    cwd_error: str = ""
    # run_script only: the body the staged script file will contain.
    inline_body: str = ""
    # Root-labelled tools only: the resource root this call resolves under, and the
    # target-canonical relative path(s) it acts on, in schema order.
    resource_root: str = ""
    paths: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return not self.cwd_error


@dataclasses.dataclass(frozen=True)
class ArgProjections:
    """The three named argument views of one dispatch, immutably bound together."""

    tool: str
    handler_args: Mapping[str, Any]
    guard_args: Mapping[str, Any]
    execution_args: ExecutionArgs

    @property
    def guard_argv(self) -> tuple[str, ...]:
        return tuple(str(token) for token in shell_argv(self.guard_args.get("cmd", "")))

    @property
    def guard_cwd(self) -> str:
        return str(self.guard_args.get("cwd", "") or "")

    def declared_divergences(self) -> tuple[str, ...]:
        return GUARD_PROJECTION_DIVERGENCES.get(self.tool, ())

    def agrees_on_argv(self) -> bool:
        """Whether the authorized argv IS the launched argv.

        False only for the tools whose divergence is declared above; for those,
        ``agrees_on_inline_body`` is the surviving invariant.
        """
        if self.tool in _INLINE_BODY_TOOLS:
            return False
        return self.guard_argv == self.execution_args.argv

    def agrees_on_inline_body(self) -> bool:
        """Guard and execution carry the same interpreter and the same body."""
        argv = self.guard_argv
        if len(argv) < 3 or argv[1] != "-c":
            return False
        return (
            self.execution_args.argv[:1] == argv[:1]
            and self.execution_args.inline_body == argv[2]
        )

    def agrees_on_cwd(self) -> bool:
        """Whether the executed cwd is the resolution of the cwd the guards judged.

        Two facts, actually compared: the guards judge a cwd SPELLING, and the execution
        carries the absolute path that spelling resolved to plus the ROOT LABEL it
        resolved under. When the spelling IS a root label — the only spelling this
        projection can check without becoming a second resolver — the execution must have
        landed under that root; and an operation that has a cwd to resolve must have one.

        It used to read `guard_cwd == handler_args["__requested_cwd"]`, defaulting to
        `guard_cwd`, and NOTHING in the repository ever wrote that key. The method
        reduced to `guard_cwd == guard_cwd`: a projection whose guard cwd was `""` and
        whose executed cwd was `/srv/app/api` asserted agreement, and deleting the
        comparison entirely left every test green. Made to pass rather than made to
        check — in the one place the three-phase protocol says must hold.

        Byte-level anti-drift is not this method's job and never was: `_execution_hash`
        (`dispatch_prepare`) folds the whole `ExecutionArgs`, cwd included, into the
        prepared token, and a re-projection that disagrees refuses with
        `PREPARED_CALL_BINDING_MISMATCH`. This is the readable half of that contract.
        """
        if not self.execution_args.resolved:
            return False
        if not operation_cwd_keys(self.tool, dict(self.guard_args)):
            # Nothing to agree about — and the execution must not have invented a cwd.
            return not self.execution_args.cwd
        if self.guard_cwd in _ALL_ROOTS:
            return self.execution_args.cwd_root == self.guard_cwd
        return bool(self.execution_args.cwd)


def _execution_argv(tool: str, handler_args: Mapping[str, Any]) -> tuple[str, ...]:
    """The launched argv, derived from the HANDLER set — deliberately not from
    the guard projection, so the equality contract compares two independent
    derivations instead of one value with itself."""
    if tool in {"run_command", "start_service"}:
        raw = handler_args.get("cmd")
        return tuple(str(part) for part in raw) if isinstance(raw, list) else tuple(shell_argv(raw or ""))
    if tool == "verify_and_record":
        return tuple(str(part) for part in (normalize_check_argv(handler_args.get("check")) or []))
    if tool == "run_script":
        interpreter = str(handler_args.get("interpreter") or "python3").strip() or "python3"
        return (interpreter,)
    return ()


def _execution_paths(tool: str, handler_args: Mapping[str, Any]) -> tuple[str, ...]:
    """The path argument(s) this call acts on, in schema order.

    Derived from the HANDLER set for the same reason ``_execution_argv`` is: the
    equality contract then compares two independent derivations rather than a value
    with itself. Spellings only — canonicalization against the target is the
    reconciliation step's job, and a Home ``resolve()`` would answer about the wrong
    filesystem.
    """

    if tool not in ROOT_AFFINITY_TOOL_NAMES:
        return ()
    paths: list[str] = []
    if (primary := str(handler_args.get("path") or "")):
        paths.append(primary)
    for row in handler_args.get("files") or []:
        if isinstance(row, Mapping):
            paths.append(str(row.get("path") or ""))
    return tuple(paths)


def _binding_target(binding: Any) -> Any:
    """The dispatch-selected binding item, or ``None`` for an unbound operation."""
    if binding is None:
        return None
    items = binding if isinstance(binding, tuple) else (binding,)
    return items[0] if items else None


def project_dispatch_args(
    ctx: Any,
    tool: str,
    args: Mapping[str, Any],
    *,
    runtime_mode: str = "",
    facts: Any = None,
    target_cwd: str = "",
    binding: Any = None,
) -> ArgProjections:
    """Build the three projections for one dispatch.

    ``facts`` is the operation's already-read fact door and ``target_cwd`` the
    cwd its target already resolved; both come from the ONE prepare, so this
    function never reads the placement itself. Omitting them means "Home is the
    target", which is exactly the local/docker case: there the cwd comes from
    ``binding`` — the dispatch's ONE Home-side resolution — and is canonicalized
    through the Home fact accessor.

    The projection is never a SECOND authority over the cwd. It re-resolves only
    when neither a target cwd nor a binding is supplied, which is the unbound
    operation (no cwd key) and the direct-call test lane; a refusal there is
    captured as ``cwd_error`` rather than raised, because the guard that owns that
    refusal must still be the one to report it.
    """
    handler_args = dict(args)
    guard_args = process_shell_guard_args(tool, handler_args, ctx=ctx, runtime_mode=runtime_mode)
    execution = _project_execution_args(
        ctx, tool, handler_args, guard_args,
        facts=facts, target_cwd=target_cwd, binding=binding,
    )
    return ArgProjections(
        tool=tool,
        handler_args=MappingProxyType(handler_args),
        guard_args=MappingProxyType(guard_args),
        execution_args=execution,
    )


def _project_execution_args(
    ctx: Any,
    tool: str,
    handler_args: Mapping[str, Any],
    guard_args: Mapping[str, Any],
    *,
    facts: Any = None,
    target_cwd: str = "",
    binding: Any = None,
) -> ExecutionArgs:
    argv = _execution_argv(tool, handler_args)
    inline_body = str(handler_args.get("script") or "") if tool in _INLINE_BODY_TOOLS else ""
    paths = _execution_paths(tool, handler_args)
    resource_root = dispatch_resource_root(tool, handler_args) if paths else ""
    keys = operation_cwd_keys(tool, dict(guard_args))
    if not keys:
        return ExecutionArgs(
            tool=tool,
            argv=argv,
            inline_body=inline_body,
            resource_root=resource_root,
            paths=paths,
        )
    if target_cwd:
        # A native route: the cwd is the target's own resolution, and the resource
        # root it resolved under is `active_workspace` by construction — the one
        # SSH-native root (matrix Q2а). Nothing Home-local is consulted, because a
        # Home resolver would answer about the wrong filesystem.
        return ExecutionArgs(
            tool=tool,
            argv=argv,
            cwd=str(target_cwd),
            cwd_root="active_workspace",
            inline_body=inline_body,
            resource_root=resource_root,
            paths=paths,
        )
    bound = _binding_target(binding)
    if bound is not None:
        # Home-local, already resolved. The dispatch resolved this operation's cwd
        # exactly once when it built the binding; calling the resolver again here
        # would make the projection a second authority over the same fact, and the
        # two answers can differ (a relative label re-joined against a different
        # root is the v6.74.0 D1 regression class). Reading the binding is what
        # keeps "one resolution per operation" a property rather than a habit.
        return ExecutionArgs(
            tool=tool,
            argv=argv,
            cwd=(facts or LOCAL_FACTS).canonical_path(pathlib.Path(bound.target_path)),
            cwd_root=str(bound.root or ""),
            inline_body=inline_body,
            resource_root=resource_root,
            paths=paths,
        )
    cwd_text, operation = keys[0]
    try:
        work_dir, cwd_root, _allowed = resolve_shell_cwd(ctx, cwd_text, operation=operation)
        cwd = (facts or LOCAL_FACTS).canonical_path(work_dir)
    except Exception as exc:  # noqa: BLE001 — the refusal is a projected fact
        return ExecutionArgs(
            tool=tool,
            argv=argv,
            inline_body=inline_body,
            resource_root=resource_root,
            paths=paths,
            cwd_error=f"{type(exc).__name__}: {exc}",
        )
    return ExecutionArgs(
        tool=tool,
        argv=argv,
        cwd=cwd,
        cwd_root=cwd_root,
        inline_body=inline_body,
        resource_root=resource_root,
        paths=paths,
    )


__all__ = [
    "ArgProjections",
    "CWD_BEARING_TOOLS",
    "ExecutionArgs",
    "GUARD_PROJECTION_DIVERGENCES",
    "project_dispatch_args",
]
