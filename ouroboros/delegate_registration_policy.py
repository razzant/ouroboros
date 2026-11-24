"""Registration persistence policy and the STARTED-row field tables.

The leaf module the delegate custody core leans on: the #362 persistence
decision and the STARTED-row shape tables live here.

#362 (the f9356572 A3 remediation, ported): on engines that support the
delegated ``workspaceRoot`` field the run registers the user's STABLE target
root as its project — an identity that outlives any individual run. The old
retire class deleted that registration at settlement (custody rows,
settle_run, pending invocations, retry binding, the orphan sweep), so a
user's own project vanished when a delegated run finished. A registration
marked persistent survives every retirement path; only ownership of the
one-shot snapshot registrations is discharged.

A root Ouroboros mints for one task, run or review (``minted_root``: a registry
checkout, a review checkout or input view, a folder under a task's own drive, a
folderless read-only scratch) is not an identity at all: it declares
``scope.ephemeral`` and registers nothing, so it never becomes a permanent engine
project or a ghost later. A sticky thread on it does the same once the engine also
sweeps the lanes such threads leave (``ephemeral_scope``); an older engine keeps
the thread's registration.

One registration serves every start on its folder and engine: the start whose POST
created it holds the duty to retire it, the others adopt it. The duty is acted on
only under the registration's lock and only once nothing else on that engine uses it
(``registration_use``; a start of another project in its folder holds it too,
``delegate_workspace_hold``); a creator that never ran leaves it owed
(``owe_registration``) to the custody sweep, and an adopter's start request is
written while the registration is held (``registration_held``). A release keeps the
registration and its folder marked until a definite answer; after a lost one, a later pass
sends the same bound release (``remove_registration``).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, Iterator, NamedTuple, Optional, Tuple

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS

log = logging.getLogger(__name__)

# Does this engine's ``POST /v2/projects`` answer carry ``created``? Learned per engine
# identity (version, build) from the answers themselves: the field's presence decides,
# never a version. Unobserved or absent keeps the previous GET-then-POST flow.
_CREATED_REPORTED: Dict[Tuple[str, str], bool] = {}


def review_owned_source(source: Any) -> bool:
    """Does this custody row's durable ``source`` name a REVIEW surface?

    True for every ``review_substrate*`` spelling. The reading lives beside the
    STARTED-row tables because ``source`` is one of their first-wins binding
    facts, and every delegation-domain consumer of a custody row shares this
    ONE reading of it (issue #1006).
    """
    return str(source or "").startswith("review_substrate")


# The evidence-only purpose of a private checking copy (#1609): a researcher or an
# external reviewer may run configured checks there, and its copy may name the tree
# it was made from, but nothing it produces is ever integrated into that tree.
INVESTIGATION_AUTHORITY = "investigation"
_NO_APPLY_AUTHORITY_SOURCES = frozenset({INVESTIGATION_AUTHORITY, "readonly"})
INVESTIGATION_NOTE = "Evidence-only investigation: the captured changes are retained as evidence and never applied."
NO_CHANGES_NOTE = ("No changes: the run's result holds exactly the files its copy started from (files Git ignores "
                   "are never part of a result), so it closed itself; there is nothing to apply or reject.")


def no_write_authority(root: str, *, investigation: bool) -> dict:
    """The mutation-authority facts of a run that writes nothing back: read-only, or an
    evidence-only investigation whose root is provenance, never an apply target."""
    return {"target_root": root, "source": INVESTIGATION_AUTHORITY if investigation else "readonly",
            "capture_mode": "evidence" if investigation else "none"}


def _fact(record: Any, key: str) -> Any:
    return record.get(key) if isinstance(record, dict) else getattr(record, key, None)


def evidence_only(record: Any) -> bool:
    """The positive evidence-only purpose (#1609): no read of its capture asks for
    apply/reject. Keyed on the recorded investigation source, not on
    ``candidate_apply_possible`` being false, which also covers degenerate rows
    with no source and no capture binding."""
    return str(_fact(record, "authority_source") or "") == INVESTIGATION_AUTHORITY


def candidate_apply_possible(record: Any) -> bool:
    """Could this run's captured work be integrated into the tree it was made from?

    ONE reading of the first-wins authority facts, shared by every integration
    branch and every observer: a ``RunCustody``, a pending-invocation row or an
    ``invocation_record`` all carry ``authority_source``, ``snapshot_id`` and
    ``resource_ref``. False on the evidence-only (or readonly) authority source,
    or on no recorded source over no capture binding at all: nothing was captured
    that could be applied. That absence does not say the run never wrote, which
    ``independent_review_without_apply`` must prove separately. Every other shape,
    an unknown source included, stays possible: a conservative answer, not
    proof of permission.
    """
    if _fact(record, "execution_mode") == "plan" or _fact(record, "mode") == "plan":
        return False
    source = str(_fact(record, "authority_source") or "")
    if source in _NO_APPLY_AUTHORITY_SOURCES:
        return False
    if source:
        return True
    ref = _fact(record, "resource_ref")
    directory_copy = (isinstance(ref, dict) and ref.get("workspace_kind") == "directory"
                      and ref.get("strategy") == "copy")
    return bool(str(_fact(record, "snapshot_id") or "")) or directory_copy


def _recorded_readonly_ask(record: Any) -> bool:
    """The read-only reviewer's own recorded shape: a STARTED row's granted
    ``access``/``mode``, or a pending row's wire shape, is readonly Ask
    with no private execution binding. Absent shape facts prove nothing."""
    request = _fact(record, "request_shape") or _fact(record, "request")
    request = request if isinstance(request, dict) else {}
    return (str(_fact(record, "access") or request.get("access") or "") == "readonly"
            and str(_fact(record, "mode") or request.get("mode") or "") == "ask"
            and not request.get("execution") and not str(_fact(record, "execution_root") or ""))


def independent_review_without_apply(record: Any) -> bool:
    """A run its REVIEW panel owns that cannot change the reviewed candidate.

    True only for a recognized review-owned ``source`` together with positive
    no-write evidence: the evidence-only (or readonly) authority a #1609 start
    records, or, on a row without that fact (an explicitly read-only reviewer
    seat, or one from before it), its recorded readonly Ask shape with no
    capture binding. An unknown or missing source, or missing shape, is never
    independent.
    """
    if not review_owned_source(_fact(record, "source")):
        return False
    authority = str(_fact(record, "authority_source") or "")
    if authority:
        return authority in _NO_APPLY_AUTHORITY_SOURCES
    return not candidate_apply_possible(record) and _recorded_readonly_ask(record)


def persistent_registration(execution_root: str, access: str) -> bool:
    """Is this run's project registration a durable user identity?

    True exactly when the engine bound a stable execution workspace
    (``workspaceRoot`` supported => non-empty execution root) and the run
    writes for the user's own tree (``workspace_write`` or ``full``): that registration
    names the user's project, not a disposable snapshot, and must outlive
    the run (#362).
    """
    from ouroboros.configured_subagents import SESSION_ACCESS_PROFILES

    return bool(str(execution_root or "").strip()) and access in SESSION_ACCESS_PROFILES


def record_persistent(record) -> bool:
    """Persistence of a DURABLE record, with a pre-marker upgrade fallback.

    Rows written before the marker existed carry no ``project_persistent``
    key; deriving from the immutable stored REQUEST (the delegated
    ``execution.workspaceRoot`` plus ``access``) keeps an old pending retry
    from deleting the user's stable project. Rows that carry the key are
    authoritative (first-wins doctrine: never recompute from a live engine).
    """
    if not isinstance(record, dict):
        return False
    if "project_persistent" in record:
        return bool(record.get("project_persistent"))
    request = record.get("request")
    if not isinstance(request, dict):
        return False
    execution = request.get("execution")
    workspace_root = execution.get("workspaceRoot") if isinstance(execution, dict) else ""
    return persistent_registration(str(workspace_root or ""), str(request.get("access") or ""))


def minted_root(root: str, drive: Any) -> bool:
    """A tree Ouroboros minted for one task, run or review: an isolated review checkout
    (``<data>/state/review_checkouts``), a retained review-input snapshot (``source_handles/review_inputs``
    of a task under the data root), a folder inside a task's own drive (``task_drives/<id>/**``,
    ``state/headless_tasks/<id>/**``, which the drive custody pass retires) or a checkout the subagent
    worktree registry holds (a child's copy, a body candidate). Any other root, an owner's project or
    the system repository, keeps its registration wherever it lives, even under the data root."""
    from ouroboros.subagent_worktrees import registered_checkout

    return minted_shape(root, drive) or registered_checkout(root)


def minted_shape(root: str, drive: Any) -> bool:
    """``minted_root``'s data-root half: the tree's place under the data root alone, no registry read."""
    import pathlib

    from ouroboros.tools.review_subject import CHECKOUT_SUBDIR

    path, data = pathlib.Path(root).resolve(), pathlib.Path(drive).resolve()
    parts = path.relative_to(data).parts if path.is_relative_to(data) else ()
    return (parts[:2] == ("state", CHECKOUT_SUBDIR)
            or any(parts[i:i + 2] == ("source_handles", "review_inputs") for i in range(len(parts) - 1))
            or (len(parts) > 1 and parts[0] == "task_drives")
            or (len(parts) > 2 and parts[:2] == ("state", "headless_tasks")))


def ephemeral_scope(gateway: Any, *, thread: bool = False) -> bool:
    """Does this engine accept ``scope.ephemeral`` for a minted root: on a run
    (``config.CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION``) or on a sticky thread, whose lane only an
    engine that sweeps unregistered namespaces removes (``CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION``)."""
    from ouroboros.config import CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION, CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION
    from ouroboros.gateways.claudexor import engine_at_least

    floor = CLAUDEXOR_EPHEMERAL_THREAD_MIN_VERSION if thread else CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION
    return engine_at_least(str(getattr(gateway, "engine_version", "") or ""), floor)


def resolve_registration(gateway, scope_root: str, execution_root: str, access: str, *, minted: bool = False,
                         thread: bool = False, resolution=None, before_post=None):
    """Register (or adopt) the scope root and decide the markers in one step.

    Returns ``(project_id, owned_project_id, project_persistent, ephemeral)``. A
    ``minted`` root on an engine that accepts ``scope.ephemeral`` for it
    (``ephemeral_scope``; ``thread``: the start binds a sticky thread) registers nothing:
    ``("", "", False, True)``. Otherwise an existing registration is adopted unowned; a
    fresh one is owned by this start; #362 marks the stable-target case
    persistent so no retire path deletes the user's own project. Ownership comes
    from the engine's ``created`` once its answers carry it (one POST, no
    ``GET /v2/projects`` over every project); until then the previous flow runs,
    and on an unobserved engine one optional POST beside a found root learns it.
    A gateway without ``project_registration`` (an older client or a test double)
    keeps the previous flow exactly. ``before_post`` runs right before any request that may create the
    registration; what it raises sends nothing, and a registration the lookup already found is adopted.
    """
    from ouroboros.execution_domain import is_work_ref
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    version = str(getattr(gateway, "engine_version", "") or "")
    if minted and ephemeral_scope(gateway, thread=thread):
        return "", "", False, True
    persistent = persistent_registration(execution_root, access)
    register = getattr(gateway, "project_registration", None)
    receipt = getattr(gateway, "project_registration_receipt", None)
    identity = (version, str(getattr(gateway, "engine_build_sha", "") or ""))
    reported = _CREATED_REPORTED.get(identity) if register or receipt else False
    existing = "" if reported else gateway.find_project_id(scope_root)
    if not (register or receipt) or (existing and reported is False):
        if not existing and before_post is not None:
            before_post()
        project_id = existing or gateway.register_project(scope_root)
        return project_id, ("" if existing else project_id), persistent, False
    try:
        if before_post is not None:
            before_post()
        if receipt:
            answer = receipt(scope_root)
            project_id, created = answer["id"], answer.get("created")
            if resolution is not None and isinstance(created, bool):
                resolution.update(registration_proof(answer))
                resolution["registration_created"] = created
                if store := resolution.get("registration_store_id"):
                    # The registry duty's key, independent of the run's receipt; a work engine's duty names that
                    # store in its own copy of the reference too (the run's reference stays as it was).
                    resolution["daemon_store_id"] = store
                    if is_work_ref(ref := resolution.get("engine_ref")):
                        resolution["engine_ref"] = {**ref, "store": store}
        else:
            project_id, created = register(scope_root)
    except ClaudexorUnavailable:
        if not existing:
            raise
        return existing, "", persistent, False  # the previous flow needed no POST here
    _CREATED_REPORTED[identity] = created is not None
    owned = created if created is not None else not (existing or reported)
    return project_id, (project_id if owned else ""), persistent, False


def resolve_for_start(drive, gateway, root, execution_root, access, *, minted=False, thread=False,
                      persistent=False, task_id="", invocation_id="", preparation=None, used_workspace="",
                      bound_project_id="", bound_resolution=None):
    """Claim preparation on the scope and explicit workspace, each lock held only for its own decision.

    ``bound_project_id`` continues a thread that already owns its scope and never registers a
    substitute; ``bound_resolution`` (``continued_binding``) is that scope's proven registration for
    the same id and engine — never a creator's ``project_resolution``: the successor adopts it, owes
    no retirement, and ``registration_held`` still re-verifies the binding once it changed.

    A start whose registration no persistence keeps may create a temporary one. It holds its route's workspace lock
    once (``delegate_workspace_hold.creation_fence``) around its folder's hold, the live uses observed before any
    request that may create, that request, and the created registration's owed duty with those uses and its own
    preparation: no claim lands between the observation and the duty, and the duty stands before the confirming read,
    which may refuse (the refusal carries no id back to the caller). That lock is released before the project lock is
    taken, the order every retirer keeps (project, then workspace). An unobservable fence or a busy lock sends nothing,
    typed; a registration the lookup already found is adopted.
    """
    from ouroboros.delegate_custody_current import registration_revision
    from ouroboros.delegate_custody_usage import project_preparation, project_retirement_lock
    from ouroboros.delegate_workspace_hold import acquired, creation_fence, prepare_workspace_use

    engine, folder, prepared = engine_facts(gateway), used_workspace or execution_root, False
    if minted and ephemeral_scope(gateway, thread=thread):
        prepare_workspace_use(drive, gateway, folder, engine, invocation_id, preparation)
        return "", "", persistent, True, None
    if bound_project_id:  # a continued thread already owns its scope; never register a substitute
        prepare_workspace_use(drive, gateway, folder, engine, invocation_id, preparation)
        project_id, owned_id, stable, ephemeral, resolution = bound_project_id, "", True, False, bound_resolution
    else:
        stable = persistent or persistent_registration(execution_root, access)
        identity = (str(getattr(gateway, "engine_version", "") or ""), str(getattr(gateway, "engine_build_sha", "") or ""))
        legacy = not ((getattr(gateway, "project_registration_receipt", None) or getattr(gateway, "project_registration", None))
                      and _CREATED_REPORTED.get(identity))
        with contextlib.ExitStack() as held:
            observe = uses = None
            if stable:  # no duty can follow: the folder's hold alone
                prepare_workspace_use(drive, gateway, folder, engine, invocation_id, preparation)
            else:
                observe, uses = held.enter_context(creation_fence(drive, gateway, folder, engine, invocation_id, preparation))
            if legacy:
                acquired(held, project_retirement_lock(drive, "root:" + os.path.normcase(os.path.realpath(root)),
                                                       engine=engine), "The folder's registration is busy")
            resolution = {"removal_stamp": registration_revision(drive), **engine}
            project_id, owned_id, _, ephemeral = resolve_registration(gateway, root, execution_root, access,
                                                                      resolution=resolution, before_post=observe)
            if owned_id and not stable:  # created: its duty, then its preparation, before the fence lets go
                changed = engine != engine_facts(resolution)
                if not owe_registration(drive, project_id, resolution, task_id=task_id, invocation_id=invocation_id,
                                        reason="registration_store_changed" if changed else "start_preparation",
                                        uses=uses):
                    raise OSError(f"the {'original' if changed else 'new'} registration's cleanup could not be recorded")
                resolution["creator_invocation"] = invocation_id
                if preparation is not None:
                    acquired(preparation, project_preparation(drive, project_id, resolution, invocation_id),
                             "The registration's preparation is busy")
                    prepared = True
    if preparation is not None or (owned_id and not stable):
        claim = {"project_id": project_id, "invocation_id": invocation_id, "request": {"scope": {"root": root}}}
        with checked_registration(drive, gateway, claim, resolution):
            resolution = claim["project_resolution"]
            if preparation is not None and project_id and not prepared:
                # The project lock orders admission and retirement; preparations share the
                # registration without keeping that lock across each other's imports.
                acquired(preparation, project_preparation(drive, project_id, resolution, invocation_id),
                         "The registration's preparation is busy")
    return project_id, owned_id, stable, ephemeral, resolution


RETIRED, KEPT = "removed", "kept"
REGISTRATION_IN_USE = "registration_in_use"
PROJECT_HAS_THREADS = "project_has_threads"
REMOVAL_CONFLICTS = frozenset((PROJECT_HAS_THREADS, "project_has_active_run"))
BINDING_DEAD = frozenset(("store_mismatch", "project_root_mismatch"))  # the served base cannot answer for this
# address: the record answers from another store (``store_mismatch``) or its folder differs
# (``project_root_mismatch``). A first request is settled by either; a repeat or an addressed read is
# answered only by the store that took the request — the engine checks the store before the folder, so
# only ``project_root_mismatch`` is that store's own answer, and another store's word keeps the intent
# waiting for the original store's return.
ANSWERED_BY_OWN_STORE = frozenset(("project_root_mismatch",))


def daemon_keeps_project(exc: Any) -> bool:
    """Only this typed code names a project the engine retains for sticky threads."""
    return getattr(exc, "code", "") == PROJECT_HAS_THREADS


def settled_thread_refusal(root, custody, exc) -> bool:
    """Only an actual release refusal can discharge a settled registration. STARTED may race it."""
    from ouroboros.delegate_custody_current import removal_status
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if not (isinstance(exc, ClaudexorUnavailable) and exc.status_code == 409 and daemon_keeps_project(exc)):
        return False
    try:
        if removal_status(root, custody.project_id, custody)[0] != "custody_kept":
            return False
        now = registration_use(root, custody, continuable=False)
        return now is not None and now.settled
    except Exception:
        log.warning("Deferring project retirement: could not reread its users", exc_info=True)
        return False


class RegistrationUse(NamedTuple):
    owned: bool
    persistent: bool
    in_use: bool
    settled: bool
    registration_root: str
    tasks: tuple[str, ...]


def engine_facts(source: Any) -> Dict[str, Any]:
    """The registry's engine, from the serving client or its recorded facts."""
    from ouroboros.execution_domain import is_work_ref, recorded_ref, registration_engine

    ref = recorded_ref(source)
    store_id = registration_engine(source)[1] if isinstance(source, dict) else str(getattr(source, "store_identity", "") or "")
    return {**({"engine_ref": dict(ref)} if is_work_ref(ref) else {}),
            **({"daemon_store_id": store_id} if store_id else {})}


def registration_proof(row):
    """Preserve the engine’s registration receipt separately from the run and handshake."""
    return {key: row[key] for key in ("registration_root", "registration_store_id")
            if isinstance(row.get(key), str) and row[key]}


def continued_binding(source, project_id):
    """A continued thread keeps the registration its scope already proved: the predecessor's recorded
    binding for the same project id and engine route, without the creator's rights — a successor adopts
    and never owns, so ``creator_invocation`` and ``registration_created`` never transfer. The store the
    binding names is checked against the serving engine by ``registration_held``. ``None`` when the
    predecessor recorded no complete binding: the successor then verifies its folder against the
    engine's own list (``registration_matches``)."""
    from ouroboros.execution_domain import engine_key

    origin = source.get("project_resolution") if isinstance(source, dict) else None
    if (not isinstance(origin, dict) or str(source.get("project_id") or "") != str(project_id or "")
            or not origin.get("registration_root") or not origin.get("registration_store_id")
            or engine_key(origin) != engine_key(source)):
        return None
    return {**registration_proof(origin), "removal_stamp": origin.get("removal_stamp") or 0,
            **engine_facts(origin)}


def bind_registration_owner(drive, entry, request):
    """Keep the creator's registry distinct from the admitting store, including a recovered start."""
    from ouroboros.execution_domain import registration_engine

    if not entry.project_id or entry.project_id != request.get("project_id"):
        entry.project_owned = False
        return
    origin = request.get("project_resolution")
    if not entry.project_owned or not request.get("project_owned"):
        entry.project_resolution = dict(origin or {})
        entry.project_owned = False
        return
    if not origin:
        # Old pending requests predate the separate registration receipt. Their handshake
        # can identify the observed registry; it cannot identify the run's admitting store.
        store = request.get("daemon_store_id") or request.get("handshake_store_id")
        origin = {"removal_stamp": "", **engine_facts({**request, "store_identity_unproven": False,
                                                      "daemon_store_id": store})}
    entry.project_resolution = dict(origin)
    if ("creator_invocation" not in origin and not record_persistent(request)
            and (not registration_engine(entry)[1] or registration_engine(entry) != registration_engine(origin))):
        if not owe_registration(drive, entry.project_id, origin, task_id=entry.task_id,
                                invocation_id=entry.invocation_id, reason="registration_origin"):
            raise OSError("the original registration could not be retained")


def remove_registration(root, gateway, project_id, engine, *, registration_root="", lane="custody"):
    """Under the project and folder locks, release the bound registration its caller has just proven unused.

    The registration and its folder stay marked from before the request until a definite answer: no start adopts
    the one or uses the other meanwhile (``registration_held``, ``delegate_workspace_hold``), and while another
    registration's release of the same recorded root is unanswered this one is not sent. After a lost answer a later
    pass, with the same proofs, sends the same bound release: the engine never reuses a released id, so the repeat
    answers ``released`` for this registration or ``absent`` (whichever request removed it), never another one.
    Only a dead binding or the engine's keep answers a repeat; any other refusal leaves the first request unknown.
    A repeat is of the very request that left. Custody repeats a ``historical`` mark only while the folder mark that a
    bound release sets just before it leaves (this store, root and id) is still there; a mark without it, and a
    ``historical_walk`` one (an older walk's id-only DELETE), prove no such request: they hold and send nothing.
    """
    from ouroboros.delegate_custody_current import mark_removal, removal_status
    from ouroboros.delegate_workspace_hold import folder_release, mark_folder_release
    from ouroboros.execution_domain import registration_engine
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.claudexor_maintenance import bound_registration_supported

    if engine != engine_facts(gateway):
        raise ClaudexorUnavailable("registration_store_changed", "The serving registry changed")
    store_id = registration_engine(engine)[1]
    pending, _ = removal_status(root, project_id, engine)
    if pending == "custody_kept" and lane == "custody":
        raise ClaudexorUnavailable(PROJECT_HAS_THREADS, "The engine kept this project", status_code=409)
    supported = registration_root and store_id and bound_registration_supported(gateway)
    holder = folder_release(root, registration_root, engine) if supported else ""  # unreadable: raised, nothing sent
    if pending and not (supported and (pending == lane or (lane == "custody" and pending == "historical"
                                                           and holder == project_id))):
        raise ClaudexorUnavailable("registration_unconfirmed", "An earlier release is unconfirmed")
    if not supported:
        return "kept", "registration_unproven"
    if holder not in ("", project_id):
        raise ClaudexorUnavailable("registration_unconfirmed", "Another release of this folder is unconfirmed")
    if not pending:
        mark_removal(root, project_id, engine, lane)
    if not holder:
        mark_folder_release(root, registration_root, engine, project_id)
    try:
        reply = gateway.bound_registration(project_id, store_id, registration_root, release=True)
    except ClaudexorUnavailable as exc:
        if removal_not_applied(exc, repeated=bool(pending)):
            kept = lane == "custody" and daemon_keeps_project(exc)
            removal_answered(root, project_id, engine, registration_root, "custody_kept" if kept else "")
        raise
    removal_answered(root, project_id, engine, registration_root)  # the caller records the registration's own end
    if pending:
        return ("removed" if reply["state"] == "released" else "gone_after_unknown"), reply
    return {"released": "removed", "absent": "absent"}[reply["state"]], reply


def removal_answered(drive, project_id, engine, registration_root, lane=None):
    """A definite answer frees the folder this registration's release held; ``lane`` (unless None) ends its mark."""
    from ouroboros.delegate_custody_current import mark_removal
    from ouroboros.delegate_workspace_hold import folder_release, mark_folder_release

    if folder_release(drive, registration_root, engine) == project_id:
        mark_folder_release(drive, registration_root, engine, "")
    if lane is not None:
        mark_removal(drive, project_id, engine, lane)


def removal_not_applied(exc, *, repeated=False):
    """Only a definite refusal or a failure before transmission permits another first release. A repeat speaks for
    the unanswered first request only by proving the binding dead in the store that took it or the engine's keep
    for its threads; another store's refusal cannot establish whether that store applied the release."""
    import httpx

    if repeated:
        return exc.status_code == 409 and exc.code in ANSWERED_BY_OWN_STORE | {PROJECT_HAS_THREADS}
    if not exc.status_code:
        return isinstance(exc.__cause__, (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout))
    # Control gates answer before routing; the 409 guards in the bound release
    # all precede unregisterInTx. A generic store/transport error proves neither.
    return exc.code in {
        400: {"invalid_request"}, 401: {"unauthorized"}, 403: {"forbidden"},
        409: {PROJECT_HAS_THREADS, "project_has_active_run", "active_run_scope_unresolved",
              "store_mismatch", "project_root_mismatch", "journal_recovery_required"},
        426: {"handshake_required"},
        503: {"daemon_stopping", "daemon_recovery_only", "store_identity_unavailable"},
    }.get(exc.status_code, ())


def recorded_workspace_key(root):
    """Compare a canonical registry address without resolving it on the current filesystem."""
    return os.path.normcase(root) if isinstance(root, str) and root else ""


def fresh_root_key(path, engine):
    """A fresh scope root as the legacy project list's comparison reads it (``registration_matches``): the owned
    engine resolves it on this filesystem, as the host does here; a work engine's folder is never resolved on this
    host and compares as given. A recorded registry address is never resolved (``recorded_workspace_key``)."""
    from ouroboros.execution_domain import is_work_ref, recorded_ref

    if not isinstance(path, str) or not path:
        return ""
    return recorded_workspace_key(path if is_work_ref(recorded_ref(engine)) else os.path.realpath(path))


def remove_custody_registration(drive, gateway, project_id, engine, use, *, live_task_ids=None, recoverable_handles=()):
    """Project lock first, then the route's workspace lock, under which a preparation (a creating start installs its
    own there, an adopting one under the project lock) is read; a held workspace or a busy or unreadable Keep gate
    sends no release. A Keep already on the registration's own tasks holds it before its folder's uses ask the engine;
    every gate (its own tasks', each related ended user's) is still taken once, right before the release."""
    from ouroboros.delegate_custody_usage import project_preparing
    from ouroboros.delegate_workspace_hold import workspace_holds, workspace_lock
    from ouroboros.owner_keep import OWNER_KEEP, gate, is_kept

    with workspace_lock(drive, engine), contextlib.ExitStack() as gates:
        if project_preparing(drive, project_id, engine):
            return "held", REGISTRATION_IN_USE
        if any(is_kept(drive, task_id) for task_id in use.tasks):  # an early read: the gates below still decide
            return "held", OWNER_KEEP
        in_use, tasks = workspace_holds(drive, gateway, project_id, use.registration_root, engine,
                                        live_task_ids=live_task_ids, recoverable_handles=recoverable_handles)
        if in_use:
            return "held", REGISTRATION_IN_USE
        for task_id in sorted(tasks.union(use.tasks)):
            if why := gates.enter_context(gate(drive, task_id, timeout_sec=0)):
                return "held", why
        return remove_registration(drive, gateway, project_id, engine, registration_root=use.registration_root)


def registration_use(drive_root, record, *, live_task_ids=None, recoverable_handles=(), continuable=True) -> Optional[RegistrationUse]:
    """Possible sharers hold a removal. Only the same route and store can own or discharge the duty."""
    from ouroboros.delegate_continuation import still_continuable
    from ouroboros.delegate_custody_current import registration_snapshot
    from ouroboros.delegate_pending import request_body
    from ouroboros.execution_domain import may_share_registration, registration_engine

    own_run = str(_fact(record, "run_id") or "")
    runs, requests, owed = registration_snapshot(drive_root, str(_fact(record, "project_id") or ""), record)
    rows = [row for row in runs.values() if may_share_registration(row, record)]
    key = registration_engine(record)
    if own_run and (own_run not in runs or registration_engine(runs[own_run]) != key):
        return None
    pending = [row for row in requests if may_share_registration(row, record)]
    persistent = [row for row in rows if row.project_persistent]
    for row in pending:
        if "project_persistent" not in row:
            row = {**row, "request": request_body(drive_root, row)}
        if record_persistent(row):
            persistent.append(row)
    origins = [row.project_resolution for row in rows if row.project_owned and registration_engine(row) == key]
    if owed:
        origins.append(owed)
    # Only a creator can supply the folder. An adopter may be the last run to settle.
    roots = {origin["registration_root"] for origin in origins if origin.get("registration_store_id") == key[1]
             and isinstance(origin.get("registration_root"), str) and origin["registration_root"]}
    bound_root = roots.pop() if len(roots) == 1 else ""
    # An unbound persistent row can protect this registration without ending its owner's duty.
    durable = any(registration_engine(row) == key and
                  (not (origin := _fact(row, "project_resolution")) or registration_engine(origin) == key)
                  for row in persistent)
    live = bool(pending) or any(not row.settled and row.run_id != own_run for row in rows)
    in_use = live or bool(persistent) or (continuable and any(
        still_continuable(drive_root, row, live_task_ids, recoverable_handles=recoverable_handles) for row in rows))
    tasks = {row.task_id for row in rows} | {_fact(record, "task_id"), _fact(owed, "task_id")}
    return RegistrationUse(bool(origins), durable, in_use, not pending and all(row.settled for row in rows), bound_root,
                           tuple(sorted(task for task in tasks if task)))


def owe_registration(drive_root: Any, project_id: str, engine: Dict[str, Any], *, task_id: str = "",
                     invocation_id: str = "", reason: str = "", uses=()) -> bool:
    """Keep the retirement of a registration this host created that no run row owns: its REGISTRATION_OWED row
    (``delegate_custody_current.OWED``), which its engine's custody sweep retires once nothing uses it and its
    PROJECT_RETIRED closes. ``uses``: the live uses its creation met (``delegate_workspace_hold.creation_fence``).
    Whether the row landed; a lost one leaves the registration to the engine."""
    from ouroboros import delegate_custody as custody

    landed = custody.emit(drive_root, custody.REGISTRATION_OWED, {
        "run_id": "", "task_id": task_id, "invocation_id": invocation_id, "project_id": project_id,
        "reason": reason, **engine_facts(engine), **registration_proof(engine), **({"workspace_uses": list(uses)} if uses else {})})
    if not landed:
        log.warning("The owed retirement of project %s was not recorded", project_id)
    return landed


def retire_owed(drive_root: Any, gateway: Any, *, live_task_ids=None, recoverable_handles=(),
                examined=frozenset()) -> None:
    """The custody sweep's pass over the owed retirements of ``gateway``'s engine: each as the duty of a run that
    owns its registration (``delegate_custody.retire_project``: removed or kept once nothing uses it, else left
    owed for the next pass), minus the ids in ``examined`` — this sweep's settled-owner pass just checked them
    on this engine, so their answer belongs to that pass this tick while their duty stays for the next one.
    Fail-soft, like the sweep."""
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_current import owed_registrations
    from ouroboros.execution_domain import registration_engine

    for owed in owed_registrations(drive_root):
        if registration_engine(engine_facts(gateway)) != registration_engine(owed):
            continue  # another engine's registration: that engine's pass retires it
        if str(owed.get("project_id") or "") in examined:
            continue  # the settled-owner pass above just examined this id on this engine
        duty = custody.RunCustody(task_id=str(owed.get("task_id") or ""), project_owned=True,
                                  invocation_id=str(owed.get("invocation_id") or ""),
                                  project_id=str(owed.get("project_id") or ""),
                                  project_resolution={**engine_facts(owed), **registration_proof(owed)},
                                  daemon_store_id=str(owed.get("daemon_store_id") or ""),
                                  engine_ref=dict(owed.get("engine_ref") or {}))
        try:
            custody.retire_project(drive_root, gateway, duty, live_task_ids=live_task_ids,
                                   recoverable_handles=recoverable_handles)
        except Exception:
            log.warning("Registration sweep failed for owed project %s", duty.project_id, exc_info=True)


def retire_unstarted(drive_root: Any, gateway: Any, project_id: str, *, task_id: str = "",
                     invocation_id: str = "", resolution=None, publish=None) -> Tuple[bool, str]:
    """A registration a start created and definitely never used: ``(retired, why it stays)``.

    Removed now, under the registration's lock, only while nothing else on its engine uses it
    (``registration_use``: another start whose POST may land, an unsettled run, a Continue offer); an adopter
    whose row is not written yet checks the removal itself (``registration_held``). A persistent sharer keeps it
    (#362). Other uses and unconfirmed removals leave a duty with the refusal or its own custody event.
    On lock timeout the caller only records the duty. The sweep checks all uses again before removal.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_custody_usage import project_retirement_lock

    engine = engine_facts(resolution if resolution is not None else gateway)
    origin = {**engine, **registration_proof(resolution or {})}
    duty = custody.RunCustody(project_id=project_id, invocation_id=invocation_id, task_id=task_id,
                              project_resolution=origin, **engine)

    def close(reason, *, owes=True):
        if publish is not None:
            return publish(False, reason, origin if owes else None)
        return not owes or owe_registration(drive_root, project_id, origin, task_id=task_id,
                                            invocation_id=invocation_id, reason=reason)

    with contextlib.ExitStack() as held:
        try:
            held.enter_context(project_retirement_lock(drive_root, project_id, engine=engine))
        except TimeoutError:
            # A preparation already has its own durable duty; a stale caller must not mint it again.
            close("retirement_owed", owes=resolution is None or "creator_invocation" not in resolution)
            return False, "retirement_owed"
        if invocation_id:
            from ouroboros import store
            from ouroboros.delegate_invocation import _read as read_invocation

            with store.open_store(drive_root, write=False) as txn:
                previous = read_invocation(txn, invocation_id)
                closed = (txn.get_row("custody_refused", invocation_id) or {}).get("failure") or {}
            # Only the sender's own unsent close (START_UNSENT) still awaits this cleanup's result row.
            if previous and previous.get("state") != "pending" and (
                    previous.get("state") != "failed_definite" or closed.get("type") != custody.START_UNSENT):
                return False, "invocation_closed"
        recorded = resolution is not None and "creator_invocation" in resolution
        if recorded and not still_owed(drive_root, project_id, resolution):
            close("registration_replaced", owes=False)
            return False, "registration_replaced"
        # A START_FAILED with the duty closes the invocation before release. If publication
        # fails the caller has made no external change; the original pending use still protects it.
        if not close("retirement_owed", owes=not recorded):
            return False, "retirement_owed"
        return custody.retire_registration_locked(drive_root, gateway, duty)


def still_owed(drive, project_id, resolution):
    from ouroboros import store
    from ouroboros.delegate_custody_current import OWED, owed_key

    with store.open_store(drive, write=False) as txn:
        duty = txn.get_row(OWED, owed_key(project_id, resolution))
    return duty is not None and duty.get("invocation_id", "") == resolution["creator_invocation"]


@contextlib.contextmanager
def registration_held(drive, gateway, row, resolution=None) -> Iterator[Dict[str, Any]]:
    """Hold the project id through START_REQUESTED. A raced resolution needs the engine's current id. A request
    whose workspace (its ``registration_workspace`` as the START row will record it) the engine does not rule out of
    an unanswered release is refused (``delegate_workspace_hold``)."""
    from ouroboros.delegate_custody_current import removal_status
    from ouroboros.delegate_custody_usage import project_retirement_lock
    from ouroboros.delegate_invocation import invocation_record
    from ouroboros.delegate_pending import replay_store_unproven
    from ouroboros.delegate_workspace_hold import fence_refusal, start_workspace
    from ouroboros.execution_domain import registration_engine

    project_id = str(row.get("project_id") or "")
    request = row.get("request") or {}
    scope = request.get("scope") or {}
    folder = {}
    workspace = str(row.get("registration_workspace") or "") or start_workspace(request, row.get("execution_root"))
    if gateway is not None and workspace:
        try:
            if why := fence_refusal(drive, gateway, workspace, engine_facts(gateway)):
                folder = {"reason": "registration_unconfirmed", "detail": why}
        except Exception as exc:
            folder = {"reason": "registration_unconfirmed", "detail": f"The folder could not be confirmed: {exc}. Nothing was sent."}
    if folder or gateway is None or not project_id or scope.get("ephemeral") or not scope.get("root"):
        yield _first_start_fate(drive, row, resolution, folder)
        return
    with contextlib.ExitStack() as held:
        refusal = {}
        try:
            held.enter_context(project_retirement_lock(drive, project_id, engine=resolution or engine_facts(gateway)))
            prior = invocation_record(drive, row.get("invocation_id", "")) if resolution is None else None
            observed = resolution or (prior or {}).get("project_resolution") or {}
            engine = engine_facts(gateway)
            pending, stamp = removal_status(drive, project_id, engine)
            before = observed.get("removal_stamp") or 0
            # Addressed revisions can predate the global observation taken before resolution.
            # Old event-id stamps have equality only. Zero means no recorded removal.
            change = bool(stamp) and (stamp > before if isinstance(stamp, int) and isinstance(before, int)
                                      else stamp != before)
            # A fresh path resolves at admission; a recorded registry address compares as recorded.
            address = observed.get("registration_root") or fresh_root_key(scope["root"], engine)
            if prior and prior.get("state") != "pending":
                refusal = {"reason": "invocation_closed", "detail": "The recorded invocation cannot be replayed."}
            elif (prior or row.get("daemon_store_id")) and (why := replay_store_unproven(
                    prior or row, getattr(gateway, "store_identity", ""))):
                refusal = {"reason": "invocation_store_unproven", "detail": why}
            elif observed and registration_engine(observed) != registration_engine(engine):
                refusal = {"reason": "registration_store_changed", "detail": "The registration's store changed."}
            elif pending and pending != "custody_kept":
                # ``custody_kept`` is the engine's proven refusal (a typed project_has_threads that
                # removed nothing), not an unconfirmed removal: the registration is alive and a
                # consumer may adopt it — the change gate below re-verifies its binding. Every other
                # lane is a removal whose answer never came and stays held.
                refusal = {"reason": "registration_unconfirmed", "detail": "The earlier removal is still unconfirmed."}
            elif ((resolution is not None and "creator_invocation" in resolution
                   and not still_owed(drive, project_id, resolution))
                  or ((change or (not observed and prior is None))
                      and not registration_matches(gateway, address, project_id,
                                                    observed.get("registration_store_id") or "",
                                                    observed.get("registration_root") or ""))):
                refusal = {"reason": "registration_replaced", "detail":
                           "The folder's registration changed before admission. Nothing was sent."}
            elif engine_facts(gateway) != engine:
                refusal = {"reason": "registration_store_changed", "detail": "The serving registry changed during confirmation."}
            else:
                row["project_resolution"] = {**observed, "removal_stamp": stamp, **engine}
        except Exception as exc:
            refusal = {"reason": getattr(exc, "code", "") or ("registration_busy" if isinstance(exc, TimeoutError)
                                                              else "registration_unconfirmed"),
                       "detail": f"Registration could not be confirmed: {exc}. Nothing was sent."}
        yield _first_start_fate(drive, row, resolution, refusal)


def _first_start_fate(drive, row, resolution, refusal):
    """A refused claim of a start this call resolved and never recorded definitely never ran."""
    from ouroboros.delegate_invocation import invocation_record

    if refusal and resolution is not None:
        try:
            if invocation_record(drive, row.get("invocation_id", "")) is None:
                refusal["definitely_unrun"] = True
        except Exception:
            log.debug("Cannot establish the first start's fate", exc_info=True)
    return refusal


def registration_matches(gateway, root, project_id, store_id="", registration_root=""):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.gateways.claudexor_maintenance import bound_registration_supported

    # Where the creation receipt names the binding (store, id and root), one addressed read answers: a full
    # inventory rechecks every project to answer for one, and an unrelated removal's stamp change need not
    # pay for a list another engine may misform. Legacy engines keep the list (and find_project_id before it).
    if store_id and registration_root and bound_registration_supported(gateway):
        return gateway.bound_registration(project_id, store_id, registration_root)["state"] == "present"
    inventory = getattr(gateway, "project_registrations", None)
    if not callable(inventory):
        # A legacy positive lookup can corroborate this binding; its empty result
        # does not distinguish an absent registration from an unreadable list.
        if gateway.find_project_id(root) == project_id:
            return True
        raise ClaudexorUnavailable("registration_unconfirmed", "No complete project inventory is available")
    rows = inventory()  # the gateway validates every id and root before returning this list
    if not isinstance(rows, list):
        raise ClaudexorUnavailable("malformed_response", "Claudexor project list is malformed")
    # The address arrives resolved or recorded; a registry row may spell the same folder
    # in another case (a Windows drive letter), so both sides compare as recorded keys.
    key = recorded_workspace_key(root)
    return any(recorded_workspace_key(row["root"]) == key and row["id"] == project_id for row in rows)


@contextlib.contextmanager
def checked_registration(drive, gateway, row, resolution):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    with registration_held(drive, gateway, row, resolution) as refusal:
        if refusal:
            exc = ClaudexorUnavailable(refusal["reason"], refusal["detail"])
            exc.registration_claim_refused = True
            if refusal.get("definitely_unrun"):
                exc.delegated_start_unrequested = True
            raise exc
        yield


def confirm_registration(drive, gateway, row, resolution) -> None:
    """The sticky start's confirmation before its thread is created: ``registration_held``'s typed
    checks under the project lock, released before that slow creation runs. The per-call preparation
    ``resolve_for_start`` entered holds the registration until the START row takes it over
    (``project_preparing`` turns every retirer away), so the general lock serializes the checks only."""
    with checked_registration(drive, gateway, row, resolution):
        pass


# How a run's ``maxSeconds`` was decided (#1196), recorded on the START_REQUESTED
# and STARTED rows as ``max_seconds_basis``: a fact beside the number, which no
# admission or continuation decision reads (a continuation follows any stop). A cap
# the nanny asked for, one its own deadline or lifetime derived or narrowed, or
# none (``unbounded``: no ``maxSeconds`` was sent); a row that predates the field is
# unknown, and an older row's ``operation_window`` names the retired 6 h fallback.
CAP_BASIS_REQUESTED = "requested"
CAP_BASIS_REQUESTED_CLAMPED_SCHEMA = "requested_clamped_by_schema"
CAP_BASIS_REQUESTED_CLAMPED_DEADLINE = "requested_clamped_by_deadline"
CAP_BASIS_REQUESTED_CLAMPED_LIFETIME = "requested_clamped_by_lifetime"
CAP_BASIS_DEADLINE_DERIVED = "deadline_derived"
CAP_BASIS_LIFETIME_DERIVED = "lifetime_derived"
CAP_BASIS_UNBOUNDED = "unbounded"

# The STARTED row's string facts as ``(RunCustody attribute, row key)`` pairs —
# one table shared by the replay and the ``record_started`` emit.
STARTED_STR_FIELDS: Tuple[Tuple[str, str], ...] = tuple(
    (attr, "route" if attr == "route_id" else attr) for attr in (
        "task_id", "route_id", "model", "profile_id", "project_id", "root_task_id",
        "parent_task_id", "category", "source", *REVIEW_ATTRIBUTION_KEYS,
        "ledger_root", "idempotency_key", "invocation_id",
        "snapshot_id", "execution_root", "execution_binding_fingerprint", "baseline_sha", "target_root",
        "authority_source", "access", "mode", "isolation",
        "execution_mode", "effective_access", "thread_id",
        "selected_subagent_id", "config_fingerprint", "work_order_fingerprint",
        "work_order_coverage", "authority_fingerprint",
        # The prior run this start explicitly continues ("" for every ordinary
        # start) and, when it continues IN that run's snapshot, its own capture
        # identity and the task that keeps the snapshot's baseline and lock.
        "continuation_of", "capture_id", "snapshot_task_id",
    )
)
# None means no optional fact; '' is a captured default choice, or an unknown recovered
# window origin. ``row_effort`` is the configured pin ('' = Auto), ``effort`` the leaf's level; ``row_processing``
# the row's own tier, ``processing_preference`` the run's.
STARTED_MAPPING_FIELDS = ("authority_ceiling", "promotion", "project_resolution")
STARTED_OPTION_FIELDS = ("effort", "processing_preference", "row_effort", "row_processing", "window_started_at",
                         "resume_start_state")
# Progress carried forward from a previous row: an idempotent re-start writes a
# SECOND started row; replacing wholesale would forget a settlement and put a
# finished run back into the orphan sweep (which would cancel it).
STARTED_PROGRESS_FLAGS: Tuple[str, ...] = (
    "ledger_recorded", "settled", "containment_disclosed", "unread_disclosed",
    "output_artifact", "output_complete", "output_sha", "output_consumed", "output_source",
    "patch_captured", "patch_manifest_sha", "patch_disposed", "patch_disposed_by", "patch_result_lost", "patch_apply_pending",
    "superseded_by")
# The closing facts SETTLED wrote (and the fault CONTAINMENT_FAULT recorded): a later
# STARTED row never carries them, so replacing wholesale would forget a run's cost,
# its spend disclosure, the engine's model/profile/locators and an unresolved fault,
# or the owner's recorded terminal-wake acknowledgement.
STARTED_SETTLEMENT_FACTS: Tuple[str, ...] = (
    "terminal_state", "terminal_reason", "engine_model", "access_profile",
    "spend_disclosed", "spend_estimated", "cost_usd", "cost_partial",
    "settled_at", "engine_locator", "terminal_ack_attempt")
# Binding/authority facts are FIRST-WINS (R1-2): a later idempotent STARTED row
# may be minted by a context that no longer knows the original binding; the
# first recorded fact is authoritative and is never erased or retargeted.
STARTED_FIRST_WINS_FACTS: Tuple[str, ...] = (
    "snapshot_id", "execution_root", "execution_binding_fingerprint", "baseline_sha", "target_root",
    "capture_id", "snapshot_task_id", "authority_source", "resource_ref", "selected_subagent_id",
    "config_fingerprint", "work_order_fingerprint", "work_order_coverage",
    "authority_fingerprint", "work_order_source_request", "category", "source",
    "execution_mode", "effective_access", "thread_id", "engine_ref", *STARTED_MAPPING_FIELDS,
    *REVIEW_ATTRIBUTION_KEYS)
