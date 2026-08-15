"""The EXECUTE phase of one dispatch (RWS v2 §3.1 step 4).

The three-phase protocol only works if something actually runs the third phase.
Prepare gathers the target's facts, authorize runs the full guard pipeline over
them, and then — for an ssh placement — the operation has to leave Home. Without
this module the dispatcher fell through to the LOCAL builtin handler after
authorizing a remote operation, which is not a partial feature but a wrong one:
the handler resolves `active_workspace` as a Home path, so a remote task either
raised a typed placement refusal into the tool loop or answered about the
Ouroboros checkout instead of the project it was pointed at.

So this is the producer, and it is deliberately thin. Everything it needs was
decided upstream:

* WHICH operation — the routing table (`tool_capabilities`), read once during
  prepare and carried on the `PreparedCall`. A tool absent from that table is
  Home-local under every placement and never reaches here.
* WHAT runs — the token-bound `execution_args` the TARGET canonicalized during
  prepare and Home authorized. They are replayed verbatim: passing anything else
  is refused by the broker (arguments mismatch) and by the target (token
  binding), so "authorized" and "executed" cannot drift apart.
* WHAT comes back — the transport already verified every declared blob, applied
  the export policy's return check and published the bytes through the artifact
  authority, so the envelope's `text` is the finished model-facing answer and
  carries no target-native path (D9). Rendering it again here would be a second
  renderer for one result.

What remains is error CLASSIFICATION, and that is the whole reason this is a
named seam rather than three lines in the dispatcher: the three failures a remote
execute can have are three different answers, and flattening them was the defect
that made `SSH_EXECUTOR_UNAVAILABLE` mean "anything remote went wrong".
"""

from __future__ import annotations

from typing import Any

from ouroboros.tools.dispatch_prepare import OutstandingPrepare, PreparedCall

# The model-facing prefixes of the execute phase. Distinct from the prepare
# phase's `REMOTE_EXECUTION_UNAVAILABLE` on purpose: an operation that never
# started and one that may have started are not the same fact, and the
# reconciliation path depends on telling them apart.
REMOTE_EXECUTE_UNAVAILABLE = "⚠️ REMOTE_EXECUTION_UNAVAILABLE"
REMOTE_EXECUTE_REFUSED = "⚠️ REMOTE_EXECUTION_FAILED"


def execute_native_operation(
    ctx: Any,
    prepared: PreparedCall,
    *,
    timeout_sec: float | None = None,
    outstanding: OutstandingPrepare | None = None,
) -> str:
    """Run one authorized native operation on the target; return its tool result.

    The caller has already checked `prepared.native_routed` and the token
    binding; this raises `ValueError` on a prepared call that is not executable,
    because an unbound token reaching the target is a wiring bug and must not be
    reported to the model as a remote condition.

    Reaching here RELEASES the dispatch's abort register: from this line on the
    prepared operation is the target's to close — an execute that is refused or lost
    in flight is a state the target and the reconciliation path own, and withdrawing
    it from Home would be a second authority over one operation's completion.
    """

    from ouroboros.workspace_diagnostics import RemoteWorkspaceError
    from ouroboros.workspace_executor import SshExecutorUnavailableError, execute_prepared

    if not prepared.native_routed:
        raise ValueError("execute_native_operation requires a natively routed PreparedCall")
    if not prepared.bound:
        raise ValueError("execute_native_operation requires an authorized (bound) PreparedCall")
    if outstanding is not None:
        outstanding.release()
    try:
        envelope = execute_prepared(
            prepared.executor,
            prepared.native,
            task_id=str(getattr(ctx, "task_id", "") or ""),
            timeout_sec=timeout_sec,
        )
    except SshExecutorUnavailableError as exc:
        # No broker in this process. The operation never started. The ACTION travels
        # with it, because `dispatch_prepare` prints it for this same exception type
        # and this arm did not — so ONE condition reached the model with a next step
        # or without one depending on which PHASE happened to catch it, and the
        # attribute exists (`workspace_executor.SshExecutorUnavailableError`) exactly
        # because its own docstring says it "reaches the MODEL as prose".
        return f"{REMOTE_EXECUTE_UNAVAILABLE}: {exc} [action: {exc.action}]"
    except RemoteWorkspaceError as exc:
        # The target's (or the broker's) OWN typed refusal. Its code, phase and
        # completion state travel to the model unaltered — a remote refusal
        # rewritten into a Home code is a diagnosis the owner cannot act on. The
        # action rides along on the same terms `dispatch_prepare` uses: printed only
        # when it is not the default `retry`, so nothing gains noise.
        action = getattr(exc, "action", "retry")
        suffix = f" [action: {action}]" if action and action != "retry" else ""
        return (
            f"{REMOTE_EXECUTE_REFUSED}: {exc.code} (phase={exc.phase}, "
            f"completion={exc.completion}): {exc}{suffix}"
        )
    return native_result_text(envelope)


def native_result_text(envelope: Any) -> str:
    """The tool result of a remote envelope.

    ``text`` is authoritative: the target renders it with the SAME renderers Home
    uses (`workspace_native_contract.render_process_result_text`, the per-operation
    error texts), which is what makes the two placements answer in one voice.

    An EMPTY text is therefore an empty RESULT, not a missing one, and speaking in
    one voice means saying so. `vcs_status` and `vcs_diff` on a clean worktree emit
    no bytes — and the Home handler answers `""` for exactly that tree — so a clean
    remote worktree has to answer `""` here as well. It used to answer
    `REMOTE_EXECUTION_FAILED: remote_result_empty`, which is how a remote
    `vcs_status` that had run perfectly read to the owner as a transport failure
    while `run_command(["git","status","--porcelain"])` on the same target worked.
    The shape that refusal claimed to guard — something that is not an envelope —
    never reaches here: it is refused upstream as `remote_result_invalid` when the
    response is decoded (`remote_worker_proxy.validated_envelope_dict`), so the
    fallback guarded nothing and misclassified the one legitimate empty answer.
    """

    text = str(getattr(envelope, "text", "") or "")
    if text:
        return text
    diagnostic = getattr(envelope, "diagnostic", None)
    if diagnostic is not None:
        from ouroboros.workspace_diagnostics import render_diagnostic_text

        return render_diagnostic_text(diagnostic)
    return ""


def withdraw_outstanding_prepare(
    ctx: Any,
    outstanding: OutstandingPrepare,
    *,
    reason: str = "home_refused_before_execute",
) -> bool:
    """Withdraw a prepared operation Home decided NOT to execute (§3.1 step 3 refusal).

    `abort_prepared` is the third member of the prepare/execute pair and had no
    production caller at all: the transport and execd sides were finished, and every
    Home refusal that lands after a successful prepare — the LLM safety supervisor, a
    shell/light-mode/protected-path guard, the argv reconciliation, a
    `PREPARED_CALL_BINDING_MISMATCH`, an argument-schema error — just returned its text
    and left the target holding a reserved token and any blobs the prepare staged until
    the TTL ran out. Now the refusal itself withdraws them.

    Failure here is SWALLOWED, completely and on purpose. The owner is being told WHY
    the operation was refused, and a dead transport is exactly the case where the abort
    fails too — so letting it raise would replace the diagnosis with a cleanup error
    about the wrong thing. The token then expires on its own, which is the state this
    call improves on rather than one it must guarantee away.

    Idempotent: the register is released BEFORE the transport call, so a second
    withdrawal of the same dispatch is a no-op rather than a second abort.
    """

    prepared = getattr(outstanding, "prepared", None)
    if prepared is None or getattr(prepared, "native", None) is None:
        return False
    outstanding.release()
    try:
        from ouroboros.workspace_executor import abort_prepared_operation

        return bool(
            abort_prepared_operation(
                prepared.executor,
                prepared.native,
                task_id=str(getattr(ctx, "task_id", "") or ""),
                reason=str(reason or "home_refused_before_execute"),
            )
        )
    except Exception:  # noqa: BLE001 — see the docstring: the refusal outranks the cleanup
        return False


__all__ = [
    "REMOTE_EXECUTE_REFUSED",
    "REMOTE_EXECUTE_UNAVAILABLE",
    "execute_native_operation",
    "native_result_text",
    "withdraw_outstanding_prepare",
]
