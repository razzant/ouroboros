"""Host projections of existing tool claims and physical custody, never results.

An invocation that unwound cannot launch again. Its known executor or supervisor
receipt can still be pending. These readers preserve that distinction for Pause,
Continue, cold sleep and idle-worker retirement without a second ledger or
business-status classification.
Untracked descendants/remote effects remain the same disclosed gap as exit zero.
"""
from __future__ import annotations

from contextvars import copy_context
import logging
import os
import threading
import time
import uuid
from typing import Any, List

log = logging.getLogger(__name__)


def invocation_binding() -> dict:
    from ouroboros.owner_pause import _TOOL_OPERATION, _member_coordinates

    active = _TOOL_OPERATION.get()
    if not active:
        return {}
    root, tree, task = _member_coordinates(active[0])
    return {"drive_root": root, "root_task_id": tree, "task_id": task,
            "launch_operation_id": active[2].get("operation_id", "")}


def invocation_outcome(source: Any, name: str, operation_id: str = "", *, future_owned: bool = False) -> dict:
    """One invocation, shared by its real worker and settlement owners."""
    from ouroboros.owner_pause import _member_coordinates
    from ouroboros.model_wait import current_model_wait
    from ouroboros.utils import utc_now_iso

    root, tree, task = _member_coordinates(source)
    owner = current_model_wait()
    bound = owner is not None and owner.task_id == task and not owner.closed
    return {"not_started": True, "operation_id": operation_id or uuid.uuid4().hex,
            "claim_root": root, "claim_task": task, "task_id": task, "root_task_id": tree,
            "tool": name, "claimed_at": utc_now_iso(), "source": source,
            "local_owner": ({"pid": os.getpid(), "process_birth": owner.answer_owner_birth,
                             "task_attempt": owner.attempt} if bound else {}),
            "owner_proof": owner_proof(),
            "memory_owned": future_owned and bound, "lock": threading.RLock(),
            # Phase transitions never wait for the durable lock's store I/O.
            "phase_lock": threading.Lock(),
            "admission_context": copy_context()}


def submit_tool_future(ctx: Any, operation_id: str, tool: str, submit: Any, function: Any, *args: Any):
    """The existing invocation owner is bound before its executor can enter."""
    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros.budget_pause import _new_tool_row, _tool_owner_release, _bind_tool_future
    outcome = invocation_outcome(ctx, tool, operation_id, future_owned=True)
    row = _new_tool_row(ctx, operation_id, tool, outcome=outcome)
    outcome["future_row"] = row
    release = _tool_owner_release(row)
    def invoke(*values):
        token = _TOOL_OPERATION.set((ctx, tool, outcome))
        try:
            return function(*values)
        finally:
            _TOOL_OPERATION.reset(token)
    try:
        future = submit(invoke, *args)
    except BaseException:
        row["done"] = True  # positive no executor entry
        release()
        raise
    _bind_tool_future(row, future)
    return future, release


_OWNER_PROOF: dict = {}


def owner_proof() -> dict:
    """This process's death-proof evidence, recorded BESIDE a claim's ``local_owner``.

    ``local_owner`` keeps its exact three keys: retirement compares that dict for
    equality, older releases included. The kernel birth (``process_identity``)
    survives exec, so the custody session tells the image that wrote a claim from
    the one that now runs under the same pid. Older releases ignore this key.
    """
    global _OWNER_PROOF
    from ouroboros.process_custody import current_custody_session_id

    proof = _OWNER_PROOF
    if (proof.get("pid") != os.getpid() or proof.get("session") != current_custody_session_id()
            or not proof.get("kernel_birth")):  # a forked child, an adopted session, or an unread birth
        from ouroboros.process_identity import process_kernel_birth

        _OWNER_PROOF = proof = {"pid": os.getpid(), "kernel_birth": process_kernel_birth(os.getpid()) or "",
                                "session": current_custody_session_id()}
    return dict(proof)


def written_by_this_image(proof: Any) -> bool:
    """A row whose ``owner_proof`` names this very process image: its writer is alive by definition.

    No death proof may select it, whatever pid or birth it shares with an ended
    image (an exec keeps both; only the custody session is new).
    """
    from ouroboros.process_custody import current_custody_session_id

    return (isinstance(proof, dict) and proof.get("pid") == os.getpid()
            and bool(proof.get("session")) and proof.get("session") == current_custody_session_id())


def _claim(outcome: dict) -> dict:
    return {key: outcome[key] for key in
            ("tool", "task_id", "root_task_id", "claimed_at", "local_owner")} | {
                "state": "claimed", "owner_proof": dict(outcome.get("owner_proof") or {})}


def retain_invocation(outcome: dict, update=None) -> None:
    """Publish the SAME existing custody row; its lock coordinates late completion."""
    from ouroboros.task_store import exists, patch_handoff
    from ouroboros.owner_pause import OwnerPauseRefused

    from ouroboros import custody_owed

    if custody_owed.pending():
        custody_owed.drain()  # this process's next write is a chance for what it still owes
    with outcome["lock"]:
        root = outcome["claim_root"]
        if not root or not outcome["task_id"]:
            return
        if not outcome.get("durable_attempted"):
            holder = outcome["task_id"]
            if not exists(root, holder):
                holder = outcome["root_task_id"]
            outcome["claim_task"] = holder
            if not exists(root, holder):
                if getattr(outcome["source"], "task_lifecycle_bound", None) is not False:
                    raise OwnerPauseRefused("owner_pause_authority_unreadable")
                return
        outcome["durable_attempted"] = True  # an unknown COMMIT still needs settlement
        def mutate(current):
            claim = dict(current) if current is not None else _claim(outcome)
            return update(claim) if update is not None else claim
        patch_handoff(root, outcome["claim_task"], "launch_handoffs", outcome["operation_id"], mutate)


def _returned_claim(root: Any, returned_at: str, confirmed: frozenset):
    """The settlement mutator: one positive local return, idempotent and never a replay."""
    from ouroboros.task_store import DELETE_ROW

    def settle(claim):
        if claim is None:
            return None  # an absent claim is never recreated
        pending = {key: value for key, value in (claim.get("external_pending") or {}).items()
                   if key not in confirmed}
        claim = {**claim, "state": "returned", "returned_at": claim.get("returned_at", returned_at),
                 "external_pending": pending}
        return claim if claim_still_owns_effect(root, claim) else DELETE_ROW
    return settle


def settle_invocation(outcome: dict) -> None:
    """Write this invocation's return within one store budget; owe it after that, never forget it.

    Store acquisitions do not wait; the durable lock is released between tries.
    A store that stays busy or refuses hands the SAME fact (stable ``returned_at``,
    confirmed tokens) to ``custody_owed``, whose drivers write it later: a
    returned call never stays «running» because one write lost a race.
    """
    from ouroboros.task_store import patch_handoff
    from ouroboros.utils import utc_now_iso
    from ouroboros.store import StoreBusy, STORE_WAIT_SEC, acquisition_budget
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

    end = time.monotonic() + STORE_WAIT_SEC
    while True:
        try:
            with outcome["lock"], acquisition_budget(0):
                if not outcome.get("durable_attempted") or outcome.get("local_holds", 0):
                    return
                if not any(outcome.get(key) is True for key in
                           ("not_started", "settled", "local_extension_returned", "mcp_call_returned")):
                    return
                returned_at = outcome.setdefault("returned_at", utc_now_iso())
                patch_handoff(outcome["claim_root"], outcome["claim_task"], "launch_handoffs", outcome["operation_id"],
                              _returned_claim(outcome["claim_root"], returned_at,
                                              frozenset(outcome.get("confirmed_external", set()))))
            return
        except StoreBusy as exc:
            remaining = end - time.monotonic()
            if exc.reason == "contention" and remaining > 0:
                time.sleep(min(ACQUISITION_SLICE_SEC, remaining))
                continue
            log.warning("Invocation custody settlement owed for %s; retried by this process",
                        outcome["operation_id"], exc_info=True)
        except Exception:
            log.warning("Invocation custody settlement owed for %s; retried by this process",
                        outcome["operation_id"], exc_info=True)
        _owe_settlement(outcome)
        return


def _owe_settlement(outcome: dict) -> None:
    from ouroboros import custody_owed
    from ouroboros.utils import utc_now_iso

    with outcome["lock"]:
        root, holder, op = outcome["claim_root"], outcome["claim_task"], outcome["operation_id"]
        settle = _returned_claim(root, outcome.setdefault("returned_at", utc_now_iso()),
                                 frozenset(outcome.get("confirmed_external", set())))
    custody_owed.owe(("settle", str(root), str(holder), str(op)),
                     lambda: _patch_owed(root, holder, op, settle), label=str(op))


def _patch_owed(root: Any, holder: str, op: str, change) -> None:
    """One bounded acquisition for an owed row write; a failure raises and the fact stays owed.

    A task whose lifecycle row is gone has nothing left to change: that fact is discharged.
    """
    from ouroboros import task_store
    from ouroboros.store import acquisition_budget
    from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

    try:
        with acquisition_budget(ACQUISITION_SLICE_SEC):
            task_store.patch_handoff(root, holder, "launch_handoffs", op, change)
    except LookupError as exc:
        if type(exc) is not LookupError:  # a KeyError or IndexError is a defect, never "the task is gone"
            raise
        log.warning("Owed custody write for %s dropped: task %s no longer exists", op, holder)


def _update_active_claim(update) -> None:
    from ouroboros.owner_pause import _TOOL_OPERATION

    active = _TOOL_OPERATION.get()
    if active:
        retain_invocation(active[2], update)


def external_handoff(kind: str):
    """Retain duty BEFORE an opaque/process effect; discharge only its exact receipt.

    The local handler returning is not evidence for this duty. No new ledger:
    these facts live on its existing launch_handoffs row.
    """
    from ouroboros._usage_wait import retry_presend
    from ouroboros.owner_pause import _TOOL_OPERATION

    active = _TOOL_OPERATION.get()
    if not active:
        return lambda: None
    outcome, token = active[2], uuid.uuid4().hex
    def add(claim):
        return {**claim, "external_pending": {**(claim.get("external_pending") or {}), token: kind}}
    discharged = False
    def confirmed():
        nonlocal discharged
        with outcome["lock"]:
            outcome.setdefault("confirmed_external", set()).add(token)
            if discharged:
                return
            remove = _without_token(token)
            try:
                retain_invocation(outcome, remove)
                discharged = True
            except Exception:
                log.warning("External custody confirmation owed for %s; retried by this process", token,
                            exc_info=True)
                _owe_confirmation(outcome, token, remove)
        if outcome.get("closed"):
            # This receipt ends only its external duty, not a live runner or
            # callback owned by the enclosing Future.
            row = outcome.get("future_row")
            if row is not None:
                from ouroboros.budget_pause import _finish_tool_row
                _finish_tool_row(row)
            else:
                settle_invocation(outcome)
    try:
        def publish():
            # Existing DB exclusion orders this new duty against settlement,
            # without denying legitimate late updates to earlier duties.
            with outcome["lock"]:
                with outcome["phase_lock"]:
                    if outcome.get("closed"):
                        from ouroboros.owner_pause import OwnerPauseRefused
                        raise OwnerPauseRefused("operation_already_returned")
                retain_invocation(outcome, add)
        retry_presend(publish, check=False)
    except BaseException:
        confirmed()  # no caller acquired this external start
        raise
    return confirmed


def _without_token(token: str):
    """An exact external receipt's removal: idempotent, also once its claim settled away."""
    def remove(claim):
        pending = dict((claim or {}).get("external_pending") or {})
        if claim is None or token not in pending:
            return None  # confirmation cannot manufacture an absent duty
        pending.pop(token, None)
        return {**claim, "external_pending": pending}
    return remove


def _owe_confirmation(outcome: dict, token: str, remove) -> None:
    """An exact external receipt whose write failed: the same removal stays owed."""
    from ouroboros import custody_owed

    root, holder, op = outcome.get("claim_root"), outcome.get("claim_task"), outcome.get("operation_id")
    if not root or not holder:
        return
    custody_owed.owe(("confirm", str(root), str(holder), str(op), token),
                     lambda: _patch_owed(root, holder, op, remove), label=str(op))


def hold_invocation():
    """An actual nested local runner joins this invocation's existing lifetime."""
    from ouroboros.owner_pause import _TOOL_OPERATION
    from ouroboros import budget_pause

    active = _TOOL_OPERATION.get()
    if not active:
        return lambda: None
    outcome = active[2]
    if outcome.get("future_row") is not None:
        return budget_pause._hold_tool_row(outcome["future_row"])
    with outcome["lock"]:
        outcome["local_holds"] = outcome.get("local_holds", 0) + 1
    def release():
        with outcome["lock"]:
            outcome["local_holds"] -= 1
            closed = outcome.get("closed")
        if closed:
            settle_invocation(outcome)
    return release


def local_producers_settled() -> bool:
    """The idle-retirement probe's non-review half, process-wide and O(live rows).

    Every registered tool future, including one its finished attempt forgot, has
    reached its final ``settled`` event (callback/runner holds released: the
    PHYSICAL end), and this process owes no custody write (``custody_owed``: a
    durable settlement or receipt that met a busy store). No sender an owner
    Pause abandoned still settles its paid answer. A returned task is not proof
    of any of these.
    """
    from ouroboros import budget_pause, custody_owed
    from ouroboros._usage_wait import _ABANDONED_SENDS

    with budget_pause._TOOL_LOCK:
        unsettled = any(not row["settled"].is_set()
                        for rows in budget_pause._TOOL_FUTURES.values() for row in rows.values())
    return not unsettled and not _ABANDONED_SENDS and not custody_owed.pending()


def retain_unconfirmed_host_operation(reason: str) -> None:
    """A host owner could not publish its independent custody; never tool meta."""
    _update_active_claim(lambda claim: {**claim, "host_unconfirmed": reason})


def record_control_handoff(event: dict) -> None:
    """Bind the existing supervisor receipt before its event can be emitted."""
    descriptor = {key: str(event.get(key) or "") for key in
                  ("type", "task_id", "client_message_id", "routing_token")}
    _update_active_claim(lambda claim: {**claim, "control_receipts":
                                      [*(claim.get("control_receipts") or []), descriptor]})


def forget_unemitted_control(event: dict) -> None:
    token = str(event.get("routing_token") or "")
    _update_active_claim(lambda claim: {**claim, "control_receipts":
        [row for row in claim.get("control_receipts") or [] if row.get("routing_token") != token]})


def claim_still_owns_effect(root: Any, claim: dict) -> bool:
    if claim.get("host_unconfirmed") or claim.get("external_pending"):
        return True
    from pathlib import Path
    from ouroboros.routing_wait import wait_for_promotion_admission, wait_for_routing_annotation

    for receipt in claim.get("control_receipts") or []:
        token = str(receipt.get("routing_token") or "")
        if not token:
            return True
        if receipt.get("type") == "promote_chat_to_task":
            observed = wait_for_promotion_admission(Path(root), receipt.get("task_id", ""), token,
                client_message_id=receipt.get("client_message_id", ""), timeout_sec=0)
        else:
            observed = wait_for_routing_annotation(Path(root), receipt.get("client_message_id", ""),
                                                   token, timeout_sec=0)
        # ``read`` follows the receiver's durable acknowledgement: delivered and acknowledged.
        if observed.get("status") not in {"scheduled", "delivered", "read", "rejected", "needs_manual_target"}:
            return True
    return False


_EFFECT_WORDS = {"shell_process": "started process", "mcp_transport": "MCP request",
                 "promote_chat_to_task": "task promotion", "steer_task": "message to a task",
                 "tell_task": "message to a task", "ensure_project_scope": "project placement"}


def tool_handoff_reason(claim: dict) -> dict:
    """Why one held claim blocks, as facts the owner can act on; ``work_stop`` and the UI word them.

    ``local_call_live``: its worker still runs (the call, or its unrecorded return);
    ``local_completion_unknown``: no evidence the worker lives or ended;
    ``owner_ended_effect_unknown`` / ``returned_effect_unconfirmed``: the local call is
    over and a named outside effect is not confirmed. A passive read: nothing is written.
    """
    from ouroboros.local_custody_repair import owner_live

    effects = sorted({_EFFECT_WORDS.get(str(kind), str(kind)) for kind in
                      [*(claim.get("external_pending") or {}).values(),
                       *(str((receipt or {}).get("type") or "") for receipt in claim.get("control_receipts") or [])]
                      if kind} | ({"host operation"} if claim.get("host_unconfirmed") else set()))
    state = claim.get("state")
    if state == "returned":
        reason = "returned_effect_unconfirmed"
    elif state == "owner_dead":
        reason = "owner_ended_effect_unknown"
    else:
        owner = claim.get("local_owner") if isinstance(claim.get("local_owner"), dict) else {}
        proof = claim.get("owner_proof") if isinstance(claim.get("owner_proof"), dict) else {}
        try:
            live = owner_live(owner.get("pid") or proof.get("pid"), owner.get("process_birth"), proof)
        except Exception:
            live = False  # an unreadable probe proves nothing either way
        reason = "local_call_live" if live else "local_completion_unknown"
    return {"reason": reason, "tool": str(claim.get("tool") or "tool"), "effects": effects}


def retained_tool_custody(root: Any, task_id: str, row: dict, *, excluding: str = "") -> list[dict]:
    """One read-only projection of still-live local invocations and known effects."""
    blockers = []
    for operation_id, claim in (row.get("launch_handoffs") or {}).items():
        if operation_id == excluding:
            continue
        if (claim.get("state") not in {"returned", "owner_dead"}
                or claim_still_owns_effect(root, claim)):
            blockers.append({"kind": "tool_handoff", "task_id": task_id, "operation": claim,
                             "operation_id": operation_id, **tool_handoff_reason(claim)})
    from ouroboros.merge_receipts import effect_is_definite
    for receipt in row.get("merge_receipts") or []:
        if receipt.get("launch_operation_ids") and not effect_is_definite(receipt):
            blockers.append({"kind": "merge_operation", "task_id": task_id,
                             "receipt_id": receipt.get("receipt_id")})
    return blockers


def _retire_owned_claims(root: Any, holder: str, owned) -> List[str]:
    """Turn the ``claimed`` rows ``owned`` selects into ``owner_dead``: effect unknown, never replayed.

    A known return stays ``returned`` and an already retired row is not rewritten:
    the death of a writer never erases what it had already reported. Returns the
    operation ids this write retired.
    """
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document
    from ouroboros.utils import utc_now_iso

    landed: List[str] = []

    def retire(current):
        require_writable_task_result_schema(current)
        if current.get("task_id") != holder or not current.get("status"):
            raise ValueError("tool_invocation_authority_unreadable")
        landed.clear()
        claims = dict(current.get("launch_handoffs") or {})
        retired = dict(current.get("retired_tool_invocations") or {})
        for op, claim in list(claims.items()):
            if not isinstance(claim, dict) or claim.get("state") != "claimed" or not owned(claim):
                continue
            fact = {**claim, "state": "owner_dead", "retired_at": utc_now_iso(),
                    "effect_outcome": "unknown", "replay_authorized": False}
            retired[op] = fact
            landed.append(op)
            if claim_still_owns_effect(root, fact):
                claims[op] = fact
            else:
                claims.pop(op)
        if not landed:
            return None
        return stamp_task_result_schema({**current, "launch_handoffs": claims,
                                         "retired_tool_invocations": retired})
    patch_document(root, holder, retire)
    return list(landed)


def claim_owned_by_process(claim: dict, *, pid: int, process_birth: str, kernel_birth: str = "") -> bool:
    """Whether ``claim`` was written by the exact process ``(pid, birth)``, kernel evidence first.

    Both kernel births present decide alone (``owner_proof``); otherwise the
    recorded ``local_owner`` must name the same pid and the same non-empty birth
    spelling. Attempt is a scope of the work, not of the process. A claim this
    very image wrote is never selected: an earlier image of this pid shares its
    pid and births, not its session (``written_by_this_image``).
    """
    proof = claim.get("owner_proof") if isinstance(claim.get("owner_proof"), dict) else {}
    owner = claim.get("local_owner") if isinstance(claim.get("local_owner"), dict) else {}
    if written_by_this_image(proof):
        return False
    if kernel_birth and proof.get("kernel_birth"):
        return proof.get("pid") == pid and proof["kernel_birth"] == kernel_birth
    return bool(process_birth) and owner.get("pid") == pid and owner.get("process_birth") == process_birth


def retire_process_invocations(root: Any, *, pid: int, process_birth: str, kernel_birth: str = "") -> List[str]:
    """The one claim retirement: every open claim the ended process wrote, across tasks and attempts.

    Called only with positive evidence that exact process ended (a joined
    retained Process, or ``local_custody_repair.owner_ended``); the open set is
    the claimed rows, never task history. Returns the retired operation ids.
    """
    import json
    import pathlib

    from ouroboros.store import open_store

    def owned(claim):
        return claim_owned_by_process(claim, pid=pid, process_birth=process_birth, kernel_birth=kernel_birth)
    with open_store(pathlib.Path(root), write=False) as txn:
        rows = txn.conn.execute("SELECT task_id, facts FROM task_handoffs "
                                "WHERE kind='launch_handoffs' AND state='claimed'").fetchall()
    holders = set()
    for row in rows:
        try:
            claim = json.loads(row[1])
        except (TypeError, ValueError):
            continue  # an unreadable row is not this process's; it stays held
        if isinstance(claim, dict) and owned(claim):
            holders.add(str(row[0]))
    retired: List[str] = []
    refused: List[Exception] = []
    from ouroboros.store import StoreBusy

    for holder in sorted(holders):  # one refusing row never starves the others; the fact stays owed for it
        try:
            retired += _retire_owned_claims(root, holder, owned)
        except StoreBusy:
            raise  # a busy store refuses every row alike: retry the whole fact later, not each row now
        except Exception as exc:
            log.debug("Claims of ended process %s on %s not retired yet", pid, holder, exc_info=True)
            refused.append(exc)
    if refused:
        raise refused[0]
    return retired

def task_process_blockers(drive_root: Any, task_ids: set[str]) -> list[dict]:
    """Strict read of existing canonical executor custody without backend I/O.

    Kill identity (command hash/signalability) is not a completion predicate:
    failed probes, access denial and exec must not hide a still-live process.
    Legacy unattributed records cannot prove a specific task's custody.
    """
    import pathlib
    from ouroboros.platform_layer import pid_provably_gone
    from ouroboros.workspace_executor import _PROCESS_STATE_DIR, _load_process_record, _valid_process_record

    folder = pathlib.Path(drive_root) / "state" / _PROCESS_STATE_DIR
    try:
        paths = list(folder.iterdir())
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ValueError("workspace_executor_custody_unreadable") from exc
    blockers = []
    for path in paths:
        if path.suffix != ".json":
            continue
        record = _load_process_record(path)
        if record is None:
            raise ValueError("workspace_executor_custody_unreadable")
        task_id = str(record.get("task_id") or "")
        if task_id not in task_ids:
            continue
        if not _valid_process_record(path, record, check_identity=False):
            raise ValueError("workspace_executor_custody_invalid")
        pid = int(record.get("host_pid") or 0)
        if record.get("executor_type") == "local":
            if pid <= 0:
                raise ValueError("workspace_executor_custody_invalid")
            if pid_provably_gone(pid):
                continue
        elif record.get("backend_completed") is True and (pid <= 0 or pid_provably_gone(pid)):
            continue
        blockers.append({"kind": "workspace_executor", "task_id": task_id,
                         "record_id": record["id"], "executor_type": record["executor_type"]})
    return blockers
