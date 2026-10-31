"""The authority switch of the current-facts store.

A door that moves the checkout to a tree WITHOUT the store (UI Rollback, the automatic
boot-smoke rollback, an Update replace to an older official ref, the agent's rollback
through the UI door) records an intent (``record_switch_intent``, one decision for every
door: ``switch_or_reset``) instead of switching at once; the restart exit, after every
worker is dead (``switch_at_restart`` from ``server._check_restart``), runs
``run_switch_seam``; a boot that finds a pending intent resumes it first
(``start`` -> ``resume_pending_switch``):

preflight (nothing moves on failure; a refusal is a normal restart on the current code
plus a reason for the owner) -> phase ``exporting`` in every store -> facts
(``store_import.export_facts`` per root) -> money (``usage_store.export_journal`` when the
target lacks ``usage_store.py``), idempotent through the switch receipt -> handles (the packing stage
restores every retained checkpoint prefix) -> fsync -> ``export.status=completed``
+ receipt -> the intent is CONSUMED -> the stores are retired -> the SAME switching owner
the door used to call runs (``switching_owner``, bound to the intent's target and reason;
HEAD is verified against the target after it returns) -> the caller exits 42 or re-execs. A
death after the consume and before the switch leaves the new code with a completed export:
the next boot finds no intent and a retired store, re-imports from the complete export and
tells the owner the switch did not happen. An unconfirmed stop of a writer refuses the
switch, never hangs. The switch receipt (``meta`` and ``state/switch_receipt.json``)
records every completed step and the files it wrote or retired; abandonment after a
partial export restores BOTH authorities.

Boot opens the import window on the calling thread before HTTP starts. One thread
resumes a pending switch, requesting restart if switched, otherwise imports in service
mode. Only publication closes the window; failures and switches keep writes migrating
and task reads at 503 for the process lifetime. Supervisor/recovery require publication;
providerless steps run after publication or are dropped. The restart door shares the
kill census that body_adoption.arm uses; it refuses any unconfirmed stop.
Failed outcomes remain in switch_progress.json until the supervisor's startup notice
can reach a bound owner after publication; each successful send durably marks reported.
"""

from __future__ import annotations

import logging
import os
import pathlib
import shutil
import subprocess
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ouroboros import evidence_packs, store, store_import, usage_store
from ouroboros.store import EXPORTING, GENERATION_KEY, PUBLISHED, RETIRED, supervisor_row, utc_stamp as _stamp
from ouroboros.usage_ledger import LEDGER_REL, UsageAccountingError
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

INTENT_REL = pathlib.Path("state/switch_intent.json")
RECEIPT_REL = pathlib.Path("state/switch_receipt.json")
PROGRESS_REL = pathlib.Path("state/switch_progress.json")
DOORS = ("ui_rollback", "vcs_rollback", "boot_smoke_rollback", "update_replace")
_NOTICE_PHASES = ("refused", "switch_failed", "switch_not_performed")
Owner = Callable[[Dict[str, Any]], Tuple[bool, str]]


def _progress(root: pathlib.Path, phase: str, **facts: Any) -> None:
    outcomes = (read_json_dict(root / PROGRESS_REL) or {}).get("outcomes", [])
    current = {"phase": phase, "at": utc_now_iso(), **facts}
    if phase in _NOTICE_PHASES:
        outcomes.append({**current, "reported": False})
    atomic_write_json(root / PROGRESS_REL, {**current, "outcomes": outcomes}, fsync=True)
    supervisor_row(root, "storage_switch", {"phase": phase, **facts})


def startup_notice(root: Any) -> None:
    """After publication and owner binding, deliver every unreported failed switch outcome (never raises)."""
    try:
        from supervisor.state import load_state

        owner_chat = int(load_state().get("owner_chat_id") or 0)
        if owner_chat and store.store_ready(root):
            report_outcomes(pathlib.Path(root), owner_chat)
    except Exception:
        log.warning("Storage switch notice remains owed", exc_info=True)


def report_outcomes(root: pathlib.Path, owner_chat_id: int) -> None:
    """Deliver retained failed outcomes once through the startup-notice seam, then fsync reported.

    The caller supplies a bound owner after publication. A failed send remains owed.
    Like other startup notices, a death between sending and marking may repeat a line.
    """
    from supervisor.message_bus import send_with_budget

    progress = read_json_dict(root / PROGRESS_REL) or {}
    if "outcomes" not in progress and progress.get("phase") in _NOTICE_PHASES:
        # A failure recorded by the previous body also owes its first notice.
        intent = (read_json_dict(root / RECEIPT_REL) or {}).get("intent", {})
        progress["outcomes"] = [{**progress, "intent": intent, "reported": False}]
    doors = dict(zip(DOORS, ("Rollback", "Agent rollback", "Automatic rollback", "Update")))
    reasons = {"quiescence_unconfirmed": "the app could not confirm that all running work had stopped",
               "second_failure": "the export failed twice and was abandoned",
               "abandonment_resumed": "an interrupted export was abandoned and the current data restored"}
    for outcome in progress.get("outcomes", []):
        if outcome.get("reported"):
            continue
        intent = outcome.get("intent") or {}
        door = doors.get(intent.get("door"), "Version switch")
        target = intent.get("target_ref") or intent.get("target_sha") or "the requested version"
        reason = outcome.get("reason") or outcome.get("message") or "the app stopped before changing versions"
        # The headline line only: Git's whole output stays in the receipt and progress record.
        reason = " ".join((str(reasons.get(reason, reason)).splitlines() or [""])[0].split()).rstrip(".")
        msg = (f"{door} to {target} did not happen: {reason}. The app is running the current version. "
               "You can press that door again.")
        send_with_budget(owner_chat_id, msg, role="system", system_type="storage_switch_notice", require_write=True)
        outcome["reported"] = utc_now_iso()
        atomic_write_json(root / PROGRESS_REL, progress, fsync=True)



def record_switch_intent(root: Any, *, target_sha: str, target_ref: str, reason: str, door: str) -> Dict[str, Any]:
    """The door's record of the switch it wants; write-once until consumed or refused."""
    root = pathlib.Path(root)
    if door not in DOORS:
        raise ValueError(f"unknown switch door: {door}")
    if (root / INTENT_REL).exists():
        raise RuntimeError(f"a switch intent is already recorded: {root / INTENT_REL}")
    intent = {"target_sha": str(target_sha), "target_ref": str(target_ref), "reason": str(reason), "door": door,
              "requested_at": utc_now_iso()}
    atomic_write_json(root / INTENT_REL, intent, fsync=True)
    return intent


def read_switch_intent(root: Any) -> Optional[Dict[str, Any]]:
    return read_json_dict(pathlib.Path(root) / INTENT_REL)


def _receipt(root: pathlib.Path, intent: Dict[str, Any]) -> Dict[str, Any]:
    """The receipt of THIS intent (by target and request time); a fresh one otherwise."""
    kept = read_json_dict(root / RECEIPT_REL) or {}
    same = {key: kept.get("intent", {}).get(key) for key in ("target_sha", "requested_at")} == {
        key: intent.get(key) for key in ("target_sha", "requested_at")}
    return kept if same else {"intent": intent, "attempts": 0, "steps": {}, "export": {}}


def _write_receipt(root: pathlib.Path, receipt: Dict[str, Any]) -> None:
    atomic_write_json(root / RECEIPT_REL, receipt, fsync=True)
    try:
        with store.open_store(root, write=True, service=True) as txn:
            txn.set_meta("switch_receipt", receipt)
    except store.StoreError:
        pass  # a retired or absent store: the file is the receipt


def target_file_bytes(repo_dir: Any, sha: str, rel: str) -> Optional[bytes]:
    """The bytes of ``rel`` in the target commit, ``None`` when the target lacks it."""
    result = subprocess.run(["git", "cat-file", "-p", f"{sha}:{rel}"], cwd=str(repo_dir), capture_output=True,
                            env={**os.environ, "LC_ALL": "C", "LANG": "C"})
    return result.stdout if result.returncode == 0 else None


def switch_needs(repo_dir: Any, target_sha: str) -> Dict[str, bool]:
    """What the target tree cannot read, computed from the target before anything moves."""
    return {"facts": target_file_bytes(repo_dir, target_sha, "ouroboros/store.py") is None,
            "money": target_file_bytes(repo_dir, target_sha, "ouroboros/usage_store.py") is None,
            "handles": target_file_bytes(repo_dir, target_sha, "ouroboros/evidence_packs.py") is None,
            "dialogue": target_file_bytes(repo_dir, target_sha, "ouroboros/dialogue_store.py") is None}


def switch_or_reset(root: Any, repo_dir: Any, *, target_sha: str, target_ref: str, reason: str, door: str,
                    reset: Callable[[], Tuple[bool, str]]) -> Tuple[bool, str, bool]:
    """A door's one decision after its writers are stopped: a target WITHOUT the store records
    the intent and moves nothing (the seam exports and switches at the restart exit the door
    requests next); a target with it runs the door's own reset now, as before the store.
    Returns ``(ok, message, deferred)``."""
    needs = switch_needs(repo_dir, target_sha)
    if needs["facts"] or needs.get("dialogue"):
        record_switch_intent(root, target_sha=target_sha, target_ref=target_ref, reason=reason, door=door)
        return True, (f"authority switch to {target_sha[:12]} recorded: the facts are exported and the checkout "
                      "moves at the restart exit"), True
    refusal = _hand_off_stores(pathlib.Path(root))
    if refusal:
        store.forget_handoffs()
        return False, f"switch refused: a store could not be handed off ({refusal})", False
    try:
        ok, message = reset()
    except BaseException as exc:
        if getattr(exc, "before_tree_effects", False):  # the reset changed nothing: this process keeps its stores
            store.forget_handoffs()
        raise
    return ok, message, False


def _hand_off_stores(root: pathlib.Path) -> str:
    """Every present store of the root and its retained drives leaves WAL before the door moves the
    checkout (``store.hand_off``); the refusal text when one cannot."""
    try:
        from ouroboros import dialogue_store

        for drive in (root, *store_import.descendant_drives(root)):
            store.hand_off(drive)
            dialogue_store.hand_off(drive)
        usage_store.hand_off(root)
    except (OSError, store.StoreError, UsageAccountingError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def next_engine_pin(repo_dir: Any, target_sha: str):
    """The engine pin the next generation selects, read from the target as a bound candidate."""
    from ouroboros.claudexor_runtime import _PIN_FILENAME, load_runtime_pin

    raw = target_file_bytes(repo_dir, target_sha, "ouroboros/" + _PIN_FILENAME)
    return None if raw is None else load_runtime_pin(data=raw)


def switching_owner(door: str) -> Owner:
    """The door's original owner, preserving rescue/verification/handoff and the intent's reason.
    Update doors refuse a managed record naming another target; the seam verifies HEAD."""
    if door in {"ui_rollback", "vcs_rollback"}:
        from supervisor.git_ops_reset import IndexLockRefusal, refusal_text
        from supervisor.update_recovery import rollback_to_version

        def restore(intent: Dict[str, Any]) -> Tuple[bool, str]:
            try:
                return rollback_to_version(intent["target_sha"], reason=intent["reason"])
            except IndexLockRefusal as exc:  # the door's typed refusal is this seam's failed switch
                return False, refusal_text(exc)

        return restore
    if door == "boot_smoke_rollback":
        from supervisor.update_merge import read_update_tx_strict, rollback_managed_update

        def rollback(intent: Dict[str, Any]) -> Tuple[bool, str]:
            recorded = str(read_update_tx_strict()[1].get("pre_update_sha") or "")
            return _unbound(recorded, intent, "the managed update transaction rolls back to") or rollback_managed_update(
                intent["reason"], reopen_writer_admission=False)

        return rollback
    if door == "update_replace":
        from supervisor import git_ops
        from supervisor.git_ops_reset import _checkout_outcome

        def replace(intent: Dict[str, Any]) -> Tuple[bool, str]:
            recorded = str((git_ops._read_update_intent() or {}).get("target_sha") or "")
            return _unbound(recorded, intent, "the managed update intent targets") or _checkout_outcome(
                git_ops.BRANCH_DEV, reason="ui_update_apply", unsynced_policy="rescue_and_reset")

        return replace
    raise ValueError(f"unknown switch door: {door}")


def _unbound(recorded: str, intent: Dict[str, Any], what: str) -> Optional[Tuple[bool, str]]:
    """The refusal when the owner's own record would move HEAD elsewhere than the intent."""
    if recorded.strip() == intent["target_sha"]:
        return None
    return False, (f"{what} {recorded.strip()[:12] or 'nothing'}, not the switch intent's {intent['target_sha'][:12]}; "
                   "checkout left unchanged")


def rev_parse(repo_dir: Any, ref: str) -> str:
    """The commit ``ref`` names in ``repo_dir``, ``""`` when it names none."""
    result = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd=str(repo_dir), capture_output=True,
                            text=True, env={**os.environ, "LC_ALL": "C", "LANG": "C"})
    return result.stdout.strip() if result.returncode == 0 else ""



def _roots(root: pathlib.Path) -> Tuple[list, str]:
    """The drives whose store is present, and the refusal when one is LOST (absent beside its
    completed stamp) or not inspectable: a retired drive has nothing left to export, a lost
    one is never «switched» past."""
    present = []
    for drive in (root, *store_import.descendant_drives(root)):
        try:
            state = store.probe(drive)[0]
        except store.StoreError as exc:
            return present, f"{type(exc).__name__}: {exc}"
        if state == "present":
            present.append(drive)
        elif store.generation_stamped(drive):
            return present, f"current store lost at {drive}: missing beside its completed stamp"
    return present, ""


def _largest_result_bytes(roots: Sequence[pathlib.Path]) -> int:
    """The largest single result file the facts export writes: the atomic rewrite of an existing
    result needs one temp file at a time, so the export's peak is one file, not the corpus."""
    largest = 0
    for drive in roots:
        with store.open_store(drive, write=False) as txn:
            largest = max(largest, int(txn.conn.execute(
                "SELECT COALESCE(MAX(LENGTH(CAST(body AS BLOB))), 0) FROM task_bodies").fetchone()[0]))
    return largest


def _money_refusal(root: pathlib.Path, receipt: Dict[str, Any]) -> str:
    """The money exporter's own refusals, checked before anything moves (``usage_store.export_refusal``):
    a completed step whose journal is intact needs nothing; an interrupted one resumes."""
    step = receipt["steps"].get("money") or {}
    if _money_done(root, step) or _nothing_to_export(root):
        return ""
    return usage_store.export_refusal(root, resume=step.get("status") in {"started", "completed"})


def _money_done(root: pathlib.Path, step: Dict[str, Any]) -> bool:
    return step.get("status") == "completed" and usage_store._sha256_file(root / LEDGER_REL) == step["journal"]["sha256"]


def _nothing_to_export(root: pathlib.Path) -> bool:
    """An install that never had money: neither a store nor a journal."""
    return not (root / usage_store.STORE_REL).exists() and not (root / LEDGER_REL).exists()


def _preflight(root: pathlib.Path, needs: Dict[str, bool], receipt: Dict[str, Any], roots: Sequence[pathlib.Path]) -> str:
    """The exporters' own refusals, then the free space the export needs: the handles' raw bytes
    the restore writes, the money journal (bounded by the store file) and the largest single
    result file (one atomic temp file at a time). A real ENOSPC still arrives as ``OSError``."""
    required = {"handles": evidence_packs.pending_restore_bytes(root) if needs["handles"] else 0,
                "largest_result": _largest_result_bytes(roots) if needs["facts"] else 0, "money": 0}
    if needs["facts"]:
        # The custody chain the old body will read must hold every event the store reduced: the
        # outbox is appended and verified empty before any store is set aside.
        from ouroboros.delegate_custody_current import flush_outbox, outbox_depth

        for drive in roots:
            flush_outbox(drive)
            depth = outbox_depth(drive)
            if depth:
                return f"custody outbox under {drive} still holds {depth} unappended event(s)"
    if needs["money"]:
        refusal = _money_refusal(root, receipt)
        if refusal:
            return refusal
        money = root / usage_store.STORE_REL
        required["money"] = money.stat().st_size if money.exists() else 0
    free = shutil.disk_usage(root).free
    if free < sum(required.values()):
        return f"free space {free} below the {sum(required.values())} bytes the export needs ({required})"
    return ""


def _export_money(root: pathlib.Path, receipt: Dict[str, Any]) -> Dict[str, Any]:
    """``usage_store.export_journal`` under the receipt: a completed step with its journal intact
    is skipped; an interrupted step resumes through the store's own export mark (its journal
    recognised by the mark's digest, a landed rename completed from the retired copy)."""
    step = receipt["steps"].get("money") or {}
    if _money_done(root, step):
        return step
    if _nothing_to_export(root):
        return {"status": "nothing_to_export", "completed_at": utc_now_iso()}
    receipt["steps"]["money"] = {"status": "started", "started_at": step.get("started_at") or utc_now_iso()}
    _write_receipt(root, receipt)
    result = usage_store.export_journal(root, resume=True)
    return {"status": "completed", "completed_at": utc_now_iso(), "store_retired_as": result["store_retired_as"],
            "pre_export_as": result["pre_export_as"], "marker": result["marker"],
            "journal": {"path": str(root / LEDGER_REL), "size": result["journal"]["size"],
                        "sha256": result["journal"]["sha256"]}}


def _abandon(root: pathlib.Path, receipt: Dict[str, Any]) -> Dict[str, Any]:
    """Restore both authorities after a partial export, resumably: the receipt records the
    abandonment ``started`` first (a death inside resumes it at the next boot), the money
    store comes back through ``usage_store.abandon_export`` (store, journal, marker, mark),
    the facts stores back to published with their own partial writes absorbed."""
    receipt["abandonment"] = {"status": "started", "at": utc_now_iso()}
    _write_receipt(root, receipt)
    restored: Dict[str, Any] = {}
    money = receipt["steps"].get("money") or {}
    if money.get("status") in {"started", "completed"}:
        moved = usage_store.abandon_export(root, retired_as=money.get("store_retired_as"))
        if moved:
            restored["money"] = moved
    for drive in _roots(root)[0]:
        with store.open_store(drive, write=True, service=True) as txn:
            if txn.phase() != PUBLISHED:
                absorbed = store_import.absorb_own_export(drive, txn)  # its own partial writes join the baseline
                txn.set_meta("phase", PUBLISHED)
                txn.set_meta("export", {"status": "abandoned", "at": utc_now_iso(), **absorbed})
                restored.setdefault("facts", []).append(str(drive))
    receipt["abandonment"] = {"status": "completed", "at": utc_now_iso(), **restored}
    _write_receipt(root, receipt)
    return restored


def _refuse(root: pathlib.Path, intent: Dict[str, Any], receipt: Dict[str, Any], reason: str, **facts: Any) -> Dict[str, Any]:
    """Nothing switches: the intent is set aside, both authorities stay (restored if a previous
    attempt moved them); progress retains the reason for the next startup owner notice."""
    restored = _abandon(root, receipt)
    store.forget_handoffs()  # this process keeps its stores
    refused = root / INTENT_REL.with_name(f"{INTENT_REL.stem}.refused-{_stamp()}.json")
    os.replace(root / INTENT_REL, refused)
    _progress(root, "refused", intent=intent, reason=reason, restored=restored, intent_moved_to=refused.name, **facts)
    return {"status": "refused", "reason": reason, "restored": restored, **facts}



def run_switch_seam(root: Any, *, repo_dir: Any, owner: Optional[Owner] = None, blockers: Sequence[str] = (),
                    restore_handles: Optional[Callable[[pathlib.Path], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Export then switch; order and durability: module docstring. Any unconfirmed blocker refuses.
    Owner defaults to the door; restore_handles provides A+ across retained descendant drives."""
    root = pathlib.Path(root)
    intent = read_switch_intent(root)
    if intent is None:
        return {"status": "no_intent"}
    receipt = _receipt(root, intent)
    if (receipt.get("abandonment") or {}).get("status") == "started":  # a death inside the abandonment: finish it
        return _refuse(root, intent, receipt, "abandonment_resumed")
    receipt["attempts"] = int(receipt.get("attempts") or 0) + 1
    _write_receipt(root, receipt)
    if receipt["attempts"] > 2:
        return _refuse(root, intent, receipt, "second_failure", attempts=receipt["attempts"])
    if blockers:
        return _refuse(root, intent, receipt, "quiescence_unconfirmed", blockers=list(blockers))
    needs = switch_needs(repo_dir, intent["target_sha"])
    roots, refusal = _roots(root) if needs["facts"] else ([], "")
    _progress(root, "preflight", needs=needs, attempt=receipt["attempts"])
    try:
        refusal = refusal or _preflight(root, needs, receipt, roots)
    except (OSError, store.StoreError, UsageAccountingError) as exc:
        refusal = f"{type(exc).__name__}: {exc}"
    if refusal:
        return _refuse(root, intent, receipt, refusal)
    try:
        steps = receipt["steps"]
        if needs["facts"]:
            for drive in roots:
                with store.open_store(drive, write=True, service=True) as txn:
                    txn.set_meta("phase", EXPORTING)
                    txn.set_meta("export", {"status": "exporting", "started_at": utc_now_iso()})
            _progress(root, "exporting", roots=[str(drive) for drive in roots])
            steps["facts"] = {"status": "completed", "roots": {str(drive): store_import.export_facts(drive) for drive in roots},
                              "completed_at": utc_now_iso()}
            _write_receipt(root, receipt)
        if needs["money"]:
            _progress(root, "money")
            steps["money"] = _export_money(root, receipt)
            _write_receipt(root, receipt)
        if needs["handles"]:
            _progress(root, "handles")
            steps["handles"] = (restore_handles or evidence_packs.restore_checkpoint_handles)(root)
            _write_receipt(root, receipt)
        if needs.get("dialogue"):
            from ouroboros.dialogue_import import export_for_older_body

            _progress(root, "dialogue")
            steps["dialogue"] = export_for_older_body(root)  # the post-cutover rows back into the chains
            _write_receipt(root, receipt)
    except (OSError, store.StoreError, UsageAccountingError) as exc:
        receipt["export"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "at": utc_now_iso()}
        _write_receipt(root, receipt)
        _progress(root, "failed", error=receipt["export"]["error"], attempt=receipt["attempts"])
        return {"status": "failed", "error": receipt["export"]["error"], "attempt": receipt["attempts"]}
    receipt["export"] = {"status": "completed", "completed_at": utc_now_iso(), "needs": needs}
    for drive in roots:
        with store.open_store(drive, write=True, service=True) as txn:
            txn.set_meta("export", receipt["export"])
    _write_receipt(root, receipt)
    refusal = _hand_off_stores(root)  # the older body also gets no WAL file (``switch_or_reset``)
    if refusal:
        return _refuse(root, intent, receipt, f"store hand-off: {refusal}")
    os.unlink(root / INTENT_REL)  # consumed BEFORE the switch: the old body never sees an intent
    store.fsync_directory(root / "state")
    receipt["retired"] = {str(drive): _retire(drive) for drive in roots}
    _write_receipt(root, receipt)
    _progress(root, "switching", door=intent["door"], target_sha=intent["target_sha"])
    ok, message = (owner or switching_owner(intent["door"]))(intent)
    head = rev_parse(repo_dir, "HEAD")
    if ok and head != (rev_parse(repo_dir, intent["target_sha"]) or intent["target_sha"]):  # the owner's word is not HEAD
        ok, message = False, f"the switching owner returned but HEAD is {head[:12] or 'unreadable'}, not the target " \
                             f"{intent['target_sha'][:12]}: {message}"
    receipt["switched"] = {"ok": bool(ok), "message": str(message), "head": head, "at": utc_now_iso()}
    _write_receipt(root, receipt)
    _progress(root, "switched" if ok else "switch_failed", intent=intent, message=str(message))
    return {"status": "switched" if ok else "switch_failed", "message": str(message), "needs": needs, "exit": 42}


def _retire(drive: pathlib.Path) -> str:
    """The store LAST: phase retired inside, the stamp cleared, THEN renamed aside (a death
    between the two leaves a present, unstamped, retired store the next import rebuilds from;
    absent beside the stamp therefore always means LOST). Returns the retired name."""
    from ouroboros.startup_migrations import stamp

    with store.open_store(drive, write=True, service=True) as txn:
        txn.set_meta("phase", RETIRED)
    path = store.store_path(drive)
    retired = path.with_name(f"{path.name}.exported-{_stamp()}")
    stamp(drive, **{GENERATION_KEY: None})
    os.replace(path, retired)
    store.fsync_directory(path.parent)
    return retired.name


def resume_pending_switch(root: Any, *, repo_dir: Any, owner: Optional[Owner] = None, blockers: Sequence[str] = (),
                          restore_handles: Optional[Callable[[pathlib.Path], Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The boot door: a pending intent resumes the seam; a completed export whose switch never
    ran is recorded for the startup owner notice (press the door again), never replayed."""
    root = pathlib.Path(root)
    if read_switch_intent(root) is not None:
        return run_switch_seam(root, repo_dir=repo_dir, owner=owner, blockers=blockers,
                               restore_handles=restore_handles or evidence_packs.restore_checkpoint_handles)
    receipt = read_json_dict(root / RECEIPT_REL) or {}
    if (receipt.get("export") or {}).get("status") == "completed" and not receipt.get("switched"):
        receipt["switched"] = {"ok": False, "message": "the switch never ran after the export", "at": utc_now_iso()}
        atomic_write_json(root / RECEIPT_REL, receipt, fsync=True)
        _progress(root, "switch_not_performed", intent=receipt.get("intent", {}),
                  target_sha=receipt.get("intent", {}).get("target_sha"))
        return {"status": "switch_not_performed", "message": "the export completed but the switch did not run; "
                                                             "the current code stays, press the door again"}
    return {"status": "nothing_pending"}


_GUARD = threading.Lock()
_BOOTS: Dict[str, "_Boot"] = {}


class _Boot:
    def __init__(self) -> None:
        self.done = threading.Event()
        self.outcome: Dict[str, Any] = {"status": "importing"}
        self.steps: List[Callable[[], Any]] = []

    @property
    def published(self) -> bool:
        return self.outcome.get("status") == "published"


def start(root: Any, *, repo_dir: Any, request_restart: Callable[[], Any]) -> Dict[str, Any]:
    """Open the import window and hand ROOT to its one import owner (a thread): the pending
    switch, then the import, off the ready path. Returns ``{"status": "importing"}``; the
    outcome is ``wait_ready``'s."""
    root = pathlib.Path(root)
    boot = _Boot()
    store.begin_import(root)  # before the thread exists: no door finds the root idle in between
    with _GUARD:
        _BOOTS[str(root)] = boot
    threading.Thread(target=_run, args=(root, repo_dir, boot, request_restart),
                     daemon=True, name="current-store-import").start()
    return {"status": "importing"}


def _run(root: pathlib.Path, repo_dir: Any, boot: _Boot, request_restart: Callable[[], Any]) -> None:
    from ouroboros.delegate_custody_current import flush_outbox
    from ouroboros.startup_migrations import prepare_startup_state

    outcome: Dict[str, Any] = {"status": "failed"}
    try:
        with store.service_thread():  # the owner's own writes pass the gate; every other thread's wait
            resumed = _resume(root, repo_dir)
            if resumed["status"] == "switched":
                outcome = resumed  # the window stays open: nothing is imported into a retired store
            else:
                report = prepare_startup_state(root, repo_dir=repo_dir, strict=False, initialize_journal=True)
                outcome = {"status": "published" if report.get("status") != "unavailable" and store.published(root)
                           else str(report.get("status") or "unavailable"),
                           "report": report}
                if outcome["status"] == "published":
                    flush_outbox(root)  # a custody event committed before a death reaches its chain at the next boot
                    store.end_import(root)  # publication: the doors open, the waiting steps run
    except Exception as exc:
        log.critical("Boot import of the current-facts store raised; the doors answer migrating until the next boot",
                     exc_info=True)
        outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    with _GUARD:
        boot.outcome = outcome
        boot.done.set()
        steps, boot.steps = boot.steps, []
    if resumed_switched := outcome["status"] == "switched":
        request_restart()  # the switched tree's own code runs next
    if not boot.published:
        log.critical("The current-facts store is not published after boot (%s); %d waiting step(s) dropped%s",
                     outcome["status"], len(steps), ": the restart was requested" if resumed_switched else "")
        return
    while True:
        for step in steps:
            _step(step)
        with _GUARD:
            steps, boot.steps = boot.steps, []
            if not steps:
                return


def _resume(root: pathlib.Path, repo_dir: Any) -> Dict[str, Any]:
    try:
        resumed = resume_pending_switch(
            root, repo_dir=repo_dir, restore_handles=evidence_packs.restore_checkpoint_handles)
    except Exception as exc:
        log.critical("Pending authority switch could not be resumed at boot; the intent waits", exc_info=True)
        resumed = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if resumed["status"] != "nothing_pending":
        log.warning("Authority switch at boot: %s", resumed)
    return resumed


def _step(step: Callable[[], Any]) -> None:
    try:
        step()
    except Exception:
        log.critical("A step waiting on the store's publication failed", exc_info=True)


def wait_ready(root: Any, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Join the boot import of ROOT (at once when none was started here); its outcome."""
    with _GUARD:
        boot = _BOOTS.get(str(pathlib.Path(root)))
    if boot is None:
        return {"status": "not_started"}
    boot.done.wait(timeout)
    return dict(boot.outcome)


def require_published(root: Any) -> Dict[str, Any]:
    """Join the boot import of ROOT and raise ``StoreMigrating`` unless it published (a root no
    boot imported here passes): the supervisor's startup phase and the startup recovery."""
    outcome = wait_ready(root)
    if outcome.get("status") not in ("published", "not_started"):
        raise store.StoreMigrating(f"current store not published at boot: {outcome.get('status')}")
    return outcome


def when_ready(root: Any, *steps: Callable[[], Any]) -> None:
    """Run ``steps`` in order once ROOT's boot import has published (on its thread), or at once
    on a thread of their own when it already published or none runs here (the providerless
    door's recovery and readiness): never on the caller's thread, whose next work a step may
    wait on (the lifespan starts the boot coordinator, and with it the extensions, after this
    call). A boot that did not publish drops them (the process stays not ready)."""
    with _GUARD:
        boot = _BOOTS.get(str(pathlib.Path(root)))
        if boot is not None and not boot.done.is_set():
            boot.steps.extend(steps)
            return
    if boot is not None and not boot.published:
        log.critical("%d step(s) dropped: the current-facts store is not published (%s)", len(steps), boot.outcome.get("status"))
        return
    threading.Thread(target=lambda: [_step(step) for step in steps], daemon=True, name="store-ready-steps").start()


def switch_at_restart(root: Any, repo_dir: Any, cleanup: Any) -> Dict[str, Any]:
    """The seam at the restart exit. ``cleanup`` is ``server._emergency_process_cleanup``'s
    census: ``worker_exits`` the pool's kill census (``supervisor.workers.last_worker_exit_census``),
    ``live_children`` the handles still alive after it, ``owned_stop`` ``stop_owned_work``'s
    outcome; an unconfirmed stop in any of them (a missing census included) is a blocker the
    seam refuses on, never a wait. After that stop the door owner's bounded Git runs as this
    exit's own finalization (``shell_process.exit_finalization``); Panic still ends it. Never raises."""
    root = pathlib.Path(root)
    owner = None
    if read_switch_intent(root) is None:
        head = rev_parse(repo_dir, "HEAD")
        needs = switch_needs(repo_dir, head) if head else {}
        if not (needs.get("facts") or needs.get("dialogue")):  # the deferring needs of ``switch_or_reset``
            return {"status": "no_intent"}
        # T2: the checked-out tree already lacks the store or the dialogue store (the agent's
        # vcs_revert, a manual checkout) and no door recorded the switch: the export runs before
        # this exit with HEAD as the target; nothing resets a second time.
        record_switch_intent(root, target_sha=head, target_ref="HEAD", door="vcs_rollback",
                                          reason="the checked-out tree cannot read the current stores")
        owner = lambda intent: (True, "the checkout already is the target")  # noqa: E731
    wait_ready(root)  # a boot-smoke rollback can arrive while this process still imports: the export reads a published store
    facts = cleanup if isinstance(cleanup, dict) else {}
    stop = facts.get("owned_stop") if isinstance(facts.get("owned_stop"), dict) else {}
    census = facts.get("worker_exits") if isinstance(facts.get("worker_exits"), dict) else None
    blockers = [name for name, blocked in (
        ("worker_exits_unconfirmed", census is None or bool(census.get("unconfirmed"))),
        ("child_processes_alive", bool(facts.get("live_children"))),
        ("owned_work_unconfirmed", stop.get("state") != "completed" or bool(stop.get("unconfirmed"))),
    ) if blocked]
    try:
        from ouroboros.tools.shell_process import exit_finalization

        with exit_finalization():
            outcome = run_switch_seam(root, repo_dir=repo_dir, owner=owner, blockers=blockers,
                                      restore_handles=evidence_packs.restore_checkpoint_handles)
    except Exception as exc:
        log.critical("Authority switch at the restart exit raised; the intent waits for the next boot", exc_info=True)
        outcome = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    log.warning("Authority switch at the restart exit: %s", outcome)
    return outcome
