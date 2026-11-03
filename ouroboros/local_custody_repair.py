"""Retire a positively ENDED local owner's capability: at exit doors, at startup and at addressed actions.

A task's local tool invocation (``launch_handoffs`` claim) and its model-answer
receiver (``local_answer_consumer_id``) are writers only while their exact local
owner can still act. One writer retires them, by PROCESS: every open claim and
receiver the ended process held, across tasks and attempts
(``retire_process_custody``; never a row this very image wrote). It runs only on
positive evidence:

- the supervisor's joined exit of a pool member: every exit door, replacement
  included, hands the exact process to ``retire_exited_process``, task or no
  task (#1554's missed doors);
- a platform-qualified ABSENCE of the recorded owner (``owner_ended``): its pid
  is positively gone (``platform_layer.pid_provably_gone``: POSIX ESRCH,
  Windows no such pid or a read exit code — never access denial, EPERM or an
  unexplained error), an exited zombie, the pid now belongs to a process whose
  validated birth differs (``process_identity``: kernel-derived, the recorded
  ``owner_proof`` first), or the pid is this very process but the row does not
  carry this image's custody session (an exec keeps pid and births; every row
  this image writes records its session, so another or none is an earlier
  image's).
  A ``ps`` wall-clock ``lstart`` string is rendered in its reader's timezone and
  locale, so two readings of one living process can differ: it never proves a
  different process. A mixed or unreadable identity, a live owner and a row
  without a recorded pid stay held;
- a legacy ``local_custody_witnesses`` row, which older releases kept when a
  retirement write failed: read and discharged, never written any more. A
  failed retirement write is now owed by the process that knows it
  (``custody_owed``).

The proof runs once per generation in the startup custody sweep, after the
process reaper (``reconcile_open_local_custody``, over the open set only), and
at each addressed action — Continue admission, Resume, the held-Continue
selection — for its own tree before it reads custody
(``repair_ended_local_custody``). Passive readers (census, GET) never write.
Generation change, queue absence, task terminality and age prove nothing.
Retirement changes local capability only: the effect stays ``unknown`` and is
never replayed, a known return stays ``returned``; delegated runs, processes,
merge and control receipts, patches and money keep their own custody.

Separately, ``ended_unbound_model_request`` reads an unbound send's retained
raised-failure fact for the passive writer census. It classifies an ended
request, retires nothing and preserves unresolved money and late-receipt rights.
"""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

WITNESS_SET = "local_custody_witnesses"  # legacy rows only: read and discharged, never written


def _exited_unreaped(pid: int) -> bool:
    """A zombie: exited, only its parent has not reaped it. macOS answers from the kernel
    record the birth read already uses (``process_identity.process_zombie``), never a ``ps``
    spawn; elsewhere ``process_containment``. An unreadable state proves nothing."""
    from ouroboros.process_identity import process_zombie

    known = process_zombie(pid)
    if known is not None:
        return known
    try:
        from ouroboros.process_containment import pid_is_zombie

        return pid_is_zombie(pid)
    except Exception:
        return False


def _this_session(proof: Dict[str, Any]) -> bool:
    """Every row this image writes records its custody session; another or none is an earlier
    image of this pid (an exec keeps pid and births), whose memory ended at exec."""
    from ouroboros.process_custody import current_custody_session_id

    return str(proof.get("session") or "") == current_custody_session_id()


def owner_ended(pid: Any, birth: Any, proof: Any = None) -> bool:
    """Positive evidence that the local owner ``(pid, birth)`` no longer runs.

    Custody rows pass their ``owner_proof`` (``{}`` when they carry none): on
    this very pid only the custody session decides. Other stamps (no proof:
    ``workspace_preparation``'s request owners) keep the birth comparison.
    """
    from ouroboros.platform_layer import pid_provably_gone, process_start_time
    from ouroboros.process_identity import birth_identity, different_birth, process_kernel_birth

    try:
        pid = int(pid or 0)
    except (TypeError, ValueError):
        return False
    custody = isinstance(proof, dict)
    proof = proof if custody else {}
    birth, kernel = str(birth or ""), str(proof.get("kernel_birth") or "")
    if pid <= 0 or not (birth or kernel):
        return False
    if pid == os.getpid() and custody:
        return not _this_session(proof)
    if pid != os.getpid() and (pid_provably_gone(pid) or _exited_unreaped(pid)):
        return True
    if kernel and different_birth(process_kernel_birth(pid) or "", kernel):
        return True
    # A ps lstart never validates: only a validated recorded birth is worth reading the live one.
    return bool(birth_identity(birth)) and different_birth(process_start_time(pid), birth)


def owner_live(pid: Any, birth: Any, proof: Any = None) -> bool:
    """Positive evidence a custody row's owner STILL runs: its pid carries the same validated birth.

    Used to word a hold honestly (``tool_custody.tool_handoff_reason``) and to
    refuse a decision to go on beside a live call; retirement never reads it.
    A ``ps`` string, a mixed kind or an unreadable birth is not evidence of
    either life or death; on this very pid the custody session decides.
    """
    from ouroboros.platform_layer import pid_provably_gone, process_start_time
    from ouroboros.process_identity import birth_identity, process_kernel_birth

    try:
        pid = int(pid or 0)
    except (TypeError, ValueError):
        return False
    proof = proof if isinstance(proof, dict) else {}
    if pid <= 0:
        return False
    if pid == os.getpid():
        return _this_session(proof)
    if pid_provably_gone(pid) or _exited_unreaped(pid):
        return False
    kernel = birth_identity(str(proof.get("kernel_birth") or ""))
    if kernel:
        return birth_identity(process_kernel_birth(pid) or "") == kernel
    recorded = birth_identity(str(birth or ""))
    return bool(recorded) and birth_identity(process_start_time(pid)) == recorded


def ended_unbound_model_request(attempt: Dict[str, Any]) -> bool:
    """An unbound attempt whose answer path already ended is not a live writer.

    The ``raised_exception`` failure is retained once the one path that could
    hand this attempt's answer to its caller ended without one: by
    ``usage_accounting._terminalize_failed_attempt`` after the send raised or
    could not start (an abandoned sender's late settlement included), or by
    the Responses search leg after its request or stream raised or closed
    incomplete. No receiver was bound, so nothing local can still take that
    answer; a detached I/O thread may only retain evidence. A missing HTTP
    status or a transport cause describes the remote exchange, whose outcome
    and money stay unresolved with their late-receipt right; nothing is
    retried. Bound or partially bound receivers, still-dispatched rows and
    rows without the retained fact stay held. No writes.
    """
    if (attempt.get("state") != "unresolved" or attempt.get("local_answer_consumer_id")
            or attempt.get("local_answer_owner_birth")
            or attempt.get("local_answer_task_attempt") is not None):
        return False
    failure = attempt.get("physical_failure")
    return isinstance(failure, dict) and failure.get("stage") == "raised_exception"


def retire_process_custody(root: Any, *, pid: int, process_birth: str, kernel_birth: str = "") -> Dict[str, Any]:
    """The one retirement: every open claim and receiver one ended process held, across tasks and
    attempts (kernel birth first, else the exact pid and birth spelling); raises when unwritten.

    Rows this very image wrote are never selected (``tool_custody.written_by_this_image``).
    Returns what changed: the retired operation ids and the number of receivers.
    """
    from ouroboros.model_wait import retire_model_consumers
    from ouroboros.tool_custody import retire_process_invocations, written_by_this_image
    from ouroboros import usage_store

    from ouroboros.store import StoreBusy

    refused = None
    try:
        claims = retire_process_invocations(root, pid=pid, process_birth=process_birth, kernel_birth=kernel_birth)
    except StoreBusy:
        raise  # the receivers would meet the same busy store
    except Exception as exc:  # the receivers below still retire; the fact stays owed for the refusing row
        claims, refused = [], exc
    with usage_store.read(pathlib.Path(root)) as txn:  # captured in one step, decoded after the read closed
        captured = txn.open_attempt_rows()
    rows = [row for row in usage_store.decode_attempts(captured) if row.get("local_answer_owner_pid") == pid]
    by_task: Dict[str, Dict[str, int]] = {}
    for row in rows:
        proof = row.get("local_answer_owner_proof") if isinstance(row.get("local_answer_owner_proof"), dict) else {}
        same = (proof.get("kernel_birth") == kernel_birth if kernel_birth and proof.get("kernel_birth")
                else bool(process_birth) and row.get("local_answer_owner_birth") == process_birth)
        consumer, attempt = row.get("local_answer_consumer_id"), row.get("local_answer_task_attempt")
        if (same and not written_by_this_image(proof) and isinstance(consumer, str) and consumer
                and type(attempt) is int):
            by_task.setdefault(str(row.get("task_id") or ""), {})[consumer] = attempt
    receivers = 0
    for task_id, consumers in _unretired(root, by_task).items():
        try:
            retire_model_consumers(pathlib.Path(root), task_id, consumers)
            receivers += len(consumers)
        except ValueError as exc:
            if type(exc.__cause__) is not LookupError:
                raise
            log.warning("Receivers of ended process %s: task %s is gone, nothing left to retire", pid, task_id)
    if refused is not None:
        raise refused
    return {"claims": claims, "receivers": receivers}


def _exit_key(root: Any, pid: int, process_birth: str, kernel_birth: str) -> tuple:
    return ("exit", str(root), int(pid), str(process_birth), str(kernel_birth))


def retire_exited_process(root: Any, *, pid: int, process_birth: str, kernel_birth: str = "") -> None:
    """An exit door's positive fact, owed in memory: safe under the queue lock; written off it.

    The supervisor confirmed this exact process ended (a joined retained
    Process). Its delivery duty applies the retirement (``custody_owed``).
    """
    from ouroboros import custody_owed

    if not pid or not (process_birth or kernel_birth):
        return

    def write():
        from ouroboros.store import acquisition_budget
        from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

        with acquisition_budget(ACQUISITION_SLICE_SEC):  # a driver's attempt never waits a full store budget
            retire_process_custody(root, pid=int(pid), process_birth=str(process_birth),
                                   kernel_birth=str(kernel_birth))
    custody_owed.owe(_exit_key(root, pid, process_birth, kernel_birth), write, label=f"exit:{pid}")


def retire_ended_process(root: Any, *, pid: int, process_birth: str, kernel_birth: str = "") -> Dict[str, Any]:
    """A proven-ended owner found off the queue lock: retire now, or owe the same fact as its exit door."""
    try:
        return retire_process_custody(root, pid=pid, process_birth=process_birth, kernel_birth=kernel_birth)
    except Exception:
        log.warning("Ended owner %s: retirement owed; retried by this process", pid, exc_info=True)
        retire_exited_process(root, pid=pid, process_birth=process_birth, kernel_birth=kernel_birth)
        return {"claims": [], "receivers": 0, "owed": True}


def _unretired(root: Any, by_task: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Drop consumers whose retirement already landed: an idempotent pass writes nothing twice."""
    from ouroboros.store import open_store

    if not by_task:
        return {}
    with open_store(pathlib.Path(root), write=False) as txn:
        return {task_id: kept for task_id, consumers in by_task.items()
                if (kept := {consumer: attempt for consumer, attempt in consumers.items()
                             if txn.handoff(task_id, consumer) is None})}


def _discharge(root: Any, witness_id: str, facts: Dict[str, Any]) -> bool:
    """A legacy witness row: apply its retirement once, then remove it."""
    from ouroboros import obligations
    from ouroboros.model_wait import retire_model_consumers

    task_id = str(facts.get("task_id") or "")
    consumers = facts.get("model_consumers")
    owner = facts.get("local_owner") if isinstance(facts.get("local_owner"), dict) else {}
    if isinstance(consumers, dict) and consumers:
        retire_model_consumers(pathlib.Path(root), task_id, {str(k): int(v) for k, v in consumers.items()})
    elif owner.get("pid") and owner.get("process_birth"):  # positive proof its owner ended; kept until it lands
        retire_process_custody(root, pid=int(owner["pid"]), process_birth=str(owner["process_birth"]))
    # A witness naming nothing retirable is discharged as it is: there is nothing to apply.
    obligations.remove(root, WITNESS_SET, witness_id)
    return True


def _discharge_witnesses(root: Any, addressed) -> List[Dict[str, Any]]:
    from ouroboros import obligations

    repaired: List[Dict[str, Any]] = []
    try:
        witnesses = obligations.members(root, WITNESS_SET)
    except Exception:
        log.warning("Legacy local-custody witnesses unreadable; repair uses platform evidence only", exc_info=True)
        return repaired
    for witness_id, facts in witnesses.items():
        if not addressed(facts):
            continue
        try:
            if _discharge(root, witness_id, facts):
                repaired.append({"kind": "witness", "task_id": facts.get("task_id"),
                                 "producer": facts.get("producer"), "reason": facts.get("reason")})
        except Exception:
            log.warning("Local-custody witness %s not discharged", witness_id, exc_info=True)
    return repaired


def _repair_holder_claims(root: Any, holder: str, done: set) -> List[Dict[str, Any]]:
    """The claimed rows one task result holds: each recorded owner that positively ended has every
    open claim and receiver of its process retired (``retire_ended_process``), once per pass (``done``)."""
    from ouroboros.task_results import load_task_result

    try:
        row = load_task_result(pathlib.Path(root), holder, strict=True) or {}
    except Exception:
        return []  # unreadable authority: nothing is retired on a guess
    owners: Dict[tuple, Dict[str, Any]] = {}
    for claim in (row.get("launch_handoffs") or {}).values():
        if not isinstance(claim, dict) or claim.get("state") != "claimed":
            continue  # nothing local left to retire
        owner = claim.get("local_owner") if isinstance(claim.get("local_owner"), dict) else {}
        proof = claim.get("owner_proof") if isinstance(claim.get("owner_proof"), dict) else {}
        pid = owner.get("pid") or proof.get("pid")  # an unbound invocation's proof names its process
        owners[(pid, str(owner.get("process_birth") or ""), str(proof.get("kernel_birth") or ""),
                str(proof.get("session") or ""))] = proof if proof.get("pid") == pid else {}
    repaired: List[Dict[str, Any]] = []
    for (pid, birth, kernel, _session), proof in owners.items():
        if (pid, birth, kernel) in done or not owner_ended(pid, birth, proof):
            continue  # one image of a pid may be ended while this one runs: each is judged alone
        done.add((pid, birth, kernel))
        outcome = retire_ended_process(root, pid=int(pid), process_birth=birth, kernel_birth=kernel)
        if outcome["claims"] or outcome["receivers"] or outcome.get("owed"):
            repaired.append({"kind": "tool_owner_absent", "task_id": holder, "pid": pid, **outcome})
    return repaired


def repair_ended_local_custody(root: Any, task_ids: Iterable[str], *, root_task_id: str) -> List[Dict[str, Any]]:
    """The addressed action's repair of ITS tree members; returns what it retired.

    Legacy witnesses first, then recorded owners with platform-qualified
    positive absence. Every failure leaves the custody exactly as held.
    """
    members = {str(task_id) for task_id in task_ids if task_id}

    def addressed(facts):
        holder = str(facts.get("holder") or "")
        return (holder in members and str(facts.get("root_task_id") or "") == str(root_task_id)
                if holder else str(facts.get("task_id") or "") in members)
    repaired, done = _discharge_witnesses(root, addressed), set()
    for holder in sorted(members):
        repaired.extend(_repair_holder_claims(root, holder, done))
    repaired.extend(_repair_absent_receivers(root, members, str(root_task_id)))
    return repaired


def reconcile_open_local_custody(root: Any) -> List[Dict[str, Any]]:
    """Once per generation, after the process reaper: retire open local custody whose owner positively ended.

    The open set only (claimed rows, open receivers, legacy witnesses), never
    task history; a live or unprovable owner stays held and is shown as such.
    """
    from ouroboros.store import open_store

    repaired = _discharge_witnesses(root, lambda _facts: True)
    with open_store(pathlib.Path(root), write=False) as txn:
        holders = sorted({str(row[0]) for row in txn.conn.execute(
            "SELECT DISTINCT task_id FROM task_handoffs WHERE kind='launch_handoffs' AND state='claimed'")})
    done: set = set()
    for holder in holders:
        repaired.extend(_repair_holder_claims(root, holder, done))
    repaired.extend(_repair_absent_receivers(root, None, None))
    return repaired


ACCEPTABLE_UNKNOWN = frozenset({"local_completion_unknown", "owner_ended_effect_unknown",
                                "returned_effect_unconfirmed"})


def receiver_reason(attempt: Dict[str, Any]) -> str:
    """An open model attempt's receiver, as ``tool_custody.tool_handoff_reason`` words a tool call.

    ``no_local_receiver``: no exact consumer and attempt are bound, so nothing local can be retired.
    """
    if not (isinstance(attempt.get("local_answer_consumer_id"), str) and attempt["local_answer_consumer_id"]
            and type(attempt.get("local_answer_task_attempt")) is int):
        return "no_local_receiver"
    proof = attempt.get("local_answer_owner_proof") if isinstance(attempt.get("local_answer_owner_proof"), dict) else {}
    try:
        live = owner_live(attempt.get("local_answer_owner_pid"), attempt.get("local_answer_owner_birth"),
                          proof if proof.get("pid") == attempt.get("local_answer_owner_pid") else {})
    except Exception:
        live = False  # an unreadable probe proves nothing either way
    return "local_call_live" if live else "local_completion_unknown"


def accept_unknown_custody(root: Any, members: Iterable[str], ids: Iterable[str], *, root_task_id: str,
                           by: Dict[str, Any]) -> Dict[str, List[Any]]:
    """The caller's decision to go on past NAMED held calls of one tree whose outcome is unknown.

    Only what the census words as unknown is accepted: a tool call
    (``tool_custody.tool_handoff_reason``) or a model answer receiver
    (``receiver_reason``) in a worker that cannot be proven live or ended, or a
    local call that is over while its outside effect is unconfirmed. A call in a
    live worker is refused, so no second writer starts beside it. The accepted
    tool row leaves ``launch_handoffs`` and stays in ``retired_tool_invocations``
    as observed, with ``effect_outcome="unknown_accepted"``,
    ``replay_authorized=False`` and the decision ``accepted`` (``by``: who, when,
    why, and the reason it was unknown); an accepted receiver is retired with the
    same record. Nothing is re-run and the outside effect is declared neither
    done nor failed; a late return of an accepted call writes nothing.
    """
    from ouroboros.model_wait import retire_model_consumers
    from ouroboros.owner_pause import tree_model_requests
    from ouroboros.task_results import load_task_result, require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document
    from ouroboros.tool_custody import tool_handoff_reason
    from ouroboros.utils import utc_now_iso

    wanted, at = {str(value) for value in ids if value}, utc_now_iso()
    accepted: List[str] = []
    refused: List[Dict[str, str]] = []

    def judged(op: str, task_id: str, reason: str) -> bool:
        if reason not in ACCEPTABLE_UNKNOWN:
            refused.append({"id": op, "task_id": task_id, "reason": reason or "unreadable"})
        return reason in ACCEPTABLE_UNKNOWN
    for holder in sorted({str(task_id) for task_id in members if task_id}):
        try:  # the reasons are probed before the write: no process probe holds the store
            claims = (load_task_result(pathlib.Path(root), holder, strict=True) or {}).get("launch_handoffs") or {}
        except Exception:
            continue  # unreadable authority: nothing is accepted on a guess
        reasons = {op: tool_handoff_reason(claim)["reason"] for op, claim in claims.items()
                   if op in wanted and isinstance(claim, dict)}
        decided = {op: (claims[op].get("state"), why) for op, why in reasons.items() if judged(op, holder, why)}
        landed: List[str] = []

        def accept(current, decided=decided, landed=landed):
            require_writable_task_result_schema(current)
            landed.clear()
            rows = dict(current.get("launch_handoffs") or {})
            retired = dict(current.get("retired_tool_invocations") or {})
            for op, (state, reason) in decided.items():
                if isinstance(rows.get(op), dict) and rows[op].get("state") == state:  # as judged, or left held
                    retired[op] = {**rows.pop(op), "retired_at": at, "effect_outcome": "unknown_accepted",
                                   "replay_authorized": False, "accepted": {**by, "reason": reason}}
                    landed.append(op)
            return stamp_task_result_schema({**current, "launch_handoffs": rows,
                                             "retired_tool_invocations": retired}) if landed else None
        if decided:
            try:  # one refusing row never takes the others' decisions with it
                patch_document(pathlib.Path(root), holder, accept)
                accepted.extend(landed)
            except Exception:
                log.warning("Named decision on %s not recorded; its calls stay held", holder, exc_info=True)
                refused.extend({"id": op, "task_id": holder, "reason": "unwritten"} for op in decided)
    try:
        attempts = [row for row in tree_model_requests(pathlib.Path(root), root_task_id)  # the census's own read
                    if str(row.get("attempt_id") or "") in wanted]
    except Exception:  # what already landed is still reported; the receivers stay held
        log.warning("Open model requests of %s unreadable; named receivers stay held", root_task_id, exc_info=True)
        attempts = []
        refused.extend({"id": op, "task_id": "", "reason": "unreadable"}
                       for op in sorted(wanted - set(accepted) - {row["id"] for row in refused}))
    for attempt in attempts:
        consumer, number = attempt.get("local_answer_consumer_id"), attempt.get("local_answer_task_attempt")
        attempt_id, task, why = str(attempt["attempt_id"]), str(attempt.get("task_id") or ""), receiver_reason(attempt)
        try:
            if why != "no_local_receiver" and not _unretired(root, {task: {consumer: number}}):
                continue  # already retired: nothing held, nothing to record twice
        except Exception:
            refused.append({"id": attempt_id, "task_id": task, "reason": "unreadable"})
            continue
        if judged(attempt_id, task, why):
            try:
                retire_model_consumers(pathlib.Path(root), task, {consumer: number},
                                       record={"accepted": {**by, "reason": why}})
                accepted.append(attempt_id)
            except Exception:
                log.warning("Named decision on receiver %s not recorded", attempt_id, exc_info=True)
                refused.append({"id": attempt_id, "task_id": task, "reason": "unwritten"})
    return {"accepted": accepted, "refused": refused,
            "unmatched": sorted(wanted - set(accepted) - {row["id"] for row in refused})}


def _repair_absent_receivers(root: Any, members: Optional[set], root_task_id: Optional[str]) -> List[Dict[str, Any]]:
    """Open attempts (of this tree, or all) whose recorded answer receiver's process positively ended."""
    from ouroboros.model_wait import retire_model_consumers
    from ouroboros import usage_store

    try:  # the tree's own open rows (root index), or the open set at startup; never history
        with usage_store.read(pathlib.Path(root)) as txn:
            captured = txn.open_attempt_rows(root_task_id=root_task_id)
        rows = [row for row in usage_store.decode_attempts(captured)
                if members is None or str(row.get("task_id") or "") in members]
    except Exception:
        log.warning("Open attempts unreadable; receivers stay held", exc_info=True)
        return []
    by_task: Dict[str, Dict[str, int]] = {}
    pids: Dict[str, Any] = {}
    for row in rows:
        consumer, attempt = row.get("local_answer_consumer_id"), row.get("local_answer_task_attempt")
        if type(attempt) is not int or not isinstance(consumer, str) or not consumer:
            continue
        proof = row.get("local_answer_owner_proof") if isinstance(row.get("local_answer_owner_proof"), dict) else {}
        if owner_ended(row.get("local_answer_owner_pid"), row.get("local_answer_owner_birth"),
                       proof if proof.get("pid") == row.get("local_answer_owner_pid") else {}):
            task_id = str(row.get("task_id") or "")
            by_task.setdefault(task_id, {})[consumer] = attempt
            pids[task_id] = row.get("local_answer_owner_pid")
    repaired: List[Dict[str, Any]] = []
    try:
        for task_id, consumers in _unretired(root, by_task).items():
            try:
                retire_model_consumers(pathlib.Path(root), task_id, consumers)
                repaired.append({"kind": "receiver_owner_absent", "task_id": task_id, "pid": pids[task_id]})
            except Exception:
                log.warning("Ended receivers of %s not retired; they stay held", task_id, exc_info=True)
    except Exception:
        log.warning("Retired receivers unreadable; receivers stay held", exc_info=True)
    return repaired
