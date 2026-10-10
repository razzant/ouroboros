"""Retire a positively ENDED local owner's capability at an addressed action (#1554).

A task's local tool invocation (``launch_handoffs`` claim) and its model-answer
receiver (``local_answer_consumer_id``) are writers only while their exact local
owner can still act. The existing writers retire them (``tool_custody.
retire_tool_invocations``, ``model_wait.retire_model_consumers``); this module
gives those writers two kinds of positive evidence and one consumer:

- A retained WITNESS: the producer that KNEW the owner ended — a confirmed
  worker death (``worker_health``), a closed or abandoned receiver
  (``model_wait``) — but could not publish the retirement keeps one minimal
  addressed obligation (``state/obligations/local_custody_witnesses.json``:
  task, identity, producer, reason). It is discharged only by a successful
  retirement through the same writer.
- A platform-qualified ABSENCE of the recorded owner: its pid is positively
  gone (``platform_layer.pid_provably_gone``: POSIX ESRCH, Windows no such pid
  or a read exit code — never access denial, EPERM or an unexplained error), an
  exited zombie, or the pid now belongs to a process whose VALIDATED birth
  identity differs (``_birth_identity``: Linux boot-qualified or boot-relative
  start ticks, Windows FILETIME — clock- and timezone-independent). A ``ps``
  wall-clock ``lstart`` string is rendered in the reader's timezone and clock,
  so two readings of one living process can differ: it never proves a
  different process. A claim without a recorded pid/birth/attempt, an
  unreadable or unvalidated identity, a live owner and a different-kind birth
  all stay held.

Only an addressed action — Continue admission, Resume, the held-Continue
selection — calls ``repair_ended_local_custody`` for its own tree before it
reads custody; passive readers (census, GET) never write. Generation change,
queue absence, task terminality, age and a mismatched fingerprint prove
nothing here. Retirement changes local capability only: the effect stays
``unknown`` and never replayed; delegated runs, processes, merge and control
receipts, patches and money keep their own custody.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, Iterable, List

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

WITNESS_SET = "local_custody_witnesses"


def _witness_id(task_id: str, identity: Dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{task_id}|{digest}"


def retain_witness(root: Any, task_id: str, *, reason: str, producer: str = "model_wait",
                   model_consumers: Dict[str, int] | None = None, local_owner: Dict[str, Any] | None = None,
                   root_task_id: str = "", holder: str = "") -> bool:
    """Keep the positive fact a failed retirement write would otherwise lose.

    Best effort, never a gate: a failure here leaves the claim/consumer itself
    held (honest unresolved custody), so absence of a witness proves nothing.
    """
    identity = ({"model_consumers": {str(k): int(v) for k, v in (model_consumers or {}).items()}}
                if model_consumers else {"local_owner": dict(local_owner or {})})
    if holder:
        # An unpublished member's claim lives on its root. That address is part
        # of the failed write's identity and must survive for the next action.
        identity["holder"] = str(holder)
    facts = {"task_id": str(task_id), "root_task_id": str(root_task_id or ""), **identity,
             "reason": str(reason), "producer": str(producer), "recorded_at": utc_now_iso()}
    try:
        from ouroboros import obligations

        obligations.add(root, WITNESS_SET, _witness_id(str(task_id), identity), facts)
        return True
    except Exception:
        log.warning("Positive local-custody witness for %s could not be retained; its custody stays held",
                    task_id, exc_info=True)
        return False


def _birth_identity(token: str) -> tuple | None:
    """A clock- and timezone-independent birth identity, or None when unvalidated.

    Only ``platform_layer.process_start_time``'s kernel-derived forms qualify:
    ``win-filetime:<int>`` (UTC FILETIME), ``<ticks>.<boot_id>`` (boot-qualified
    ticks) and ``<ticks>``/``<ticks>.`` (boot-relative ticks). Anything else —
    the ``ps lstart`` wall-clock fallback included — is None: never positive.
    """
    if match := re.fullmatch(r"win-filetime:(\d+)", token):
        return ("windows", int(match.group(1)))
    if match := re.fullmatch(r"(\d+)\.([0-9a-f]{32})", token):
        return ("linux_boot", int(match.group(1)), match.group(2))
    if match := re.fullmatch(r"(\d+)\.?", token):
        return ("ticks", int(match.group(1)))
    return None


def _different_birth(live: str, recorded: str) -> bool:
    """Positive only when both identities are validated, of one kind, and unequal."""
    live_id, recorded_id = _birth_identity(live), _birth_identity(recorded)
    return bool(live_id and recorded_id and live_id[0] == recorded_id[0] and live_id != recorded_id)


def owner_ended(pid: Any, birth: Any) -> bool:
    """Positive evidence that the local owner ``(pid, birth)`` no longer runs."""
    from ouroboros.platform_layer import pid_provably_gone, process_start_time

    try:
        pid = int(pid or 0)
    except (TypeError, ValueError):
        return False
    birth = str(birth or "")
    if pid <= 0 or not birth:
        return False
    if pid == os.getpid():
        return _different_birth(process_start_time(pid), birth)
    if pid_provably_gone(pid):
        return True
    try:
        from ouroboros.process_containment import pid_is_zombie

        if pid_is_zombie(pid):
            return True  # exited; only its parent has not reaped it
    except Exception:
        return False
    return _different_birth(process_start_time(pid), birth)


def retire_ended_owner(root: Any, task_id: str, root_task_id: str, *, pid: int, birth: str,
                       attempt: int, producer: str, holder: str = "") -> bool:
    """Retire every claim and receiver of one ended local owner; witness on failure.

    The exact owner identity (pid, birth, attempt, task and tree) selects; any
    other claim or consumer is untouched. Returns True when both writers landed.
    """
    from ouroboros.model_wait import retire_model_consumers
    from ouroboros.tool_custody import retire_tool_invocations

    landed = True
    try:
        retire_tool_invocations(pathlib.Path(root), task_id, root_task_id, pid=pid, process_birth=birth,
                                task_attempt=attempt, holder=holder)
    except Exception:
        log.warning("Ended owner of %s: tool invocation retirement unwritten", task_id, exc_info=True)
        landed = False
    try:
        from ouroboros import usage_store

        with usage_store.read(pathlib.Path(root)) as txn:  # this task's rows only (task index)
            rows = txn.attempts("task_id = ?", (str(task_id),))
        consumers = {row["local_answer_consumer_id"]: attempt for row in rows
                     if row.get("task_id") == task_id and row.get("root_task_id") == root_task_id
                     and row.get("local_answer_owner_pid") == pid and row.get("local_answer_owner_birth") == birth
                     and type(row.get("local_answer_task_attempt")) is int
                     and row["local_answer_task_attempt"] == attempt
                     and isinstance(row.get("local_answer_consumer_id"), str) and row["local_answer_consumer_id"]}
        if consumers:
            retire_model_consumers(pathlib.Path(root), task_id, consumers)
    except Exception:
        log.warning("Ended owner of %s: model consumer retirement unwritten", task_id, exc_info=True)
        landed = False
    if not landed:
        retain_witness(root, task_id, reason="owner_ended_retirement_unwritten", producer=producer,
                       local_owner={"pid": pid, "process_birth": birth, "task_attempt": attempt},
                       root_task_id=root_task_id, holder=holder)
    return landed


def _discharge(root: Any, witness_id: str, facts: Dict[str, Any]) -> bool:
    from ouroboros import obligations
    from ouroboros.model_wait import retire_model_consumers

    task_id = str(facts.get("task_id") or "")
    consumers = facts.get("model_consumers")
    owner = facts.get("local_owner") if isinstance(facts.get("local_owner"), dict) else {}
    if isinstance(consumers, dict) and consumers:
        retire_model_consumers(pathlib.Path(root), task_id, {str(k): int(v) for k, v in consumers.items()})
    elif not (owner and retire_ended_owner(root, task_id, str(facts.get("root_task_id") or task_id),
                                           pid=int(owner.get("pid") or 0), birth=str(owner.get("process_birth") or ""),
                                           attempt=int(owner.get("task_attempt") or 0), producer="repair",
                                           holder=str(facts.get("holder") or ""))):
        return False
    obligations.remove(root, WITNESS_SET, witness_id)
    return True


def _witnesses(root: Any) -> Dict[str, Dict[str, Any]]:
    from ouroboros import obligations

    with obligations.locked(root):
        return obligations._read(root, WITNESS_SET, missing_ok=True)


def repair_ended_local_custody(root: Any, task_ids: Iterable[str], *, root_task_id: str) -> List[Dict[str, Any]]:
    """The addressed action's repair of ITS tree members; returns what it retired.

    Retained witnesses first, then recorded owners with platform-qualified
    positive absence. Every failure leaves the custody exactly as held.
    """
    from ouroboros.task_results import load_task_result

    members = {str(task_id) for task_id in task_ids if task_id}
    repaired: List[Dict[str, Any]] = []
    try:
        witnesses = _witnesses(root)
    except Exception:
        log.warning("Local-custody witnesses unreadable; repair uses platform evidence only", exc_info=True)
        witnesses = {}
    for witness_id, facts in witnesses.items():
        holder = str(facts.get("holder") or "")
        addressed = (holder in members and str(facts.get("root_task_id") or "") == str(root_task_id)
                     if holder else str(facts.get("task_id") or "") in members)
        if addressed:
            try:
                if _discharge(root, witness_id, facts):
                    repaired.append({"kind": "witness", "task_id": facts.get("task_id"),
                                     "producer": facts.get("producer"), "reason": facts.get("reason")})
            except Exception:
                log.warning("Local-custody witness %s not discharged", witness_id, exc_info=True)
    for holder in sorted(members):
        try:
            row = load_task_result(pathlib.Path(root), holder, strict=True) or {}
        except Exception:
            continue  # unreadable authority: nothing is retired on a guess
        owners = {}
        for claim in (row.get("launch_handoffs") or {}).values():
            if not isinstance(claim, dict):
                continue  # no exact identity; leave this independent claim held
            owner = claim.get("local_owner")
            if claim.get("state") == "claimed" and isinstance(owner, dict):
                key = (str(claim.get("task_id") or ""), str(claim.get("root_task_id") or ""),
                       owner.get("pid"), owner.get("process_birth"), owner.get("task_attempt"))
                owners[key] = owner
        for (task_id, tree, pid, birth, attempt), owner in owners.items():
            if type(attempt) is int and owner_ended(pid, birth) and retire_ended_owner(
                    root, task_id, tree, pid=int(pid), birth=str(birth), attempt=attempt,
                    producer="repair", holder=holder):
                repaired.append({"kind": "tool_owner_absent", "task_id": task_id, "pid": pid})
    repaired.extend(_repair_absent_receivers(root, members, str(root_task_id)))
    return repaired


def _repair_absent_receivers(root: Any, members: set, root_task_id: str) -> List[Dict[str, Any]]:
    """Open attempts of this tree whose recorded answer receiver's process positively ended."""
    from ouroboros import usage_store

    try:  # the tree's own open rows (root index), never the whole open census
        with usage_store.read(pathlib.Path(root)) as txn:
            rows = [row for row in txn.open_attempts(root_task_id=root_task_id)
                    if str(row.get("task_id") or "") in members]
    except Exception:
        log.warning("Open attempts unreadable; receivers stay held", exc_info=True)
        return []
    repaired, seen = [], set()
    for row in rows:
        owner = (str(row.get("task_id") or ""), str(row.get("root_task_id") or ""),
                 row.get("local_answer_owner_pid"), row.get("local_answer_owner_birth"),
                 row.get("local_answer_task_attempt"))
        if owner in seen or type(owner[4]) is not int or not row.get("local_answer_consumer_id"):
            continue
        seen.add(owner)
        if owner_ended(owner[2], owner[3]) and retire_ended_owner(
                root, owner[0], owner[1], pid=int(owner[2]), birth=str(owner[3]), attempt=owner[4],
                producer="repair"):
            repaired.append({"kind": "receiver_owner_absent", "task_id": owner[0], "pid": owner[2]})
    return repaired
