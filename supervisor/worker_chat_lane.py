"""The conversation lane's doors and the carrier's first step; resume after a restart.

An owner's Main or room turn, a consciousness wake-up and the auto-resume turn are
admitted here and EXECUTE in a conversation-lane worker (``conversation_lane``):
``handle_chat_direct`` and ``handle_wake_direct`` decide admission under Q and the
repo-writer gate lock, write the RUNNING row (origin ``direct``, carrier
``conversation``), publish the authoritative start receipt and return; nothing of
the turn's body — the actor, the context build, the rounds, the tools — runs in
this process. ``prepare_conversation_task`` is the carrier's first step: the owner's
input layered onto the admitted seed (late answers and card presses, the inline
image or staged attachments, the Project binding, consciousness authority, the
task contract, the frozen tool profile, a wake's title), run by
``worker_process.run_worker_task`` before ``handle_task``.

The two disclosed inline executors keep a thread of the server process: a live
realtime call (``executor=inline_call``: its transport authority is process-local,
owner decision 3A) and the degraded carrier chosen only on a physical fact
(``executor=inline_degraded``: a spawn refused, or the repo-writer gate closed in an
assisted-update marker window; owner decision 2A). Both run the worker task body on a
thread of this process over in-memory queues (``inline_carrier``), own a RUNNING row
and a registered actor; ``stop_direct_chat_turn`` and the agent-lock arming
(``workers.arm_direct_chat_turn``) serve only such registered actors — every row,
inline or carried, is stopped by the pooled custody (finalize intent, then the kill).

Turns are refused while the repo-writer gate is closed for a DESTRUCTIVE update
window (apply/replace prologue, materialization, rollback), so a managed update
never races a turn that could touch the checkout mid-reset. While the ONE
authorized assisted resolver holds the repository (``assisted_resolution`` /
``committing_assisted``) the lanes stay open: conversation admission is not
repo-writing permission — the registry's managed-update guard refuses every
repo-mutating tool to any task but that resolver — and an owner line reaches the
resolver through the ordinary ``steer_task`` mailbox path (#283). The server's own
owner-control path is imported BEFORE conflict markers land in the live tree
(``preload_owner_control_path``).
"""

from __future__ import annotations

import logging
import pathlib
import time
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

from supervisor.state import append_jsonl
from ouroboros.consciousness_authority import wake_cause
from ouroboros.task_carrier import EXECUTOR_INLINE_CALL
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)



def _pool():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


# Update phases during which the repository is held by the ONE authorized
# assisted resolver and nothing else moves the tree: conversation stays open.
_CONVERSATION_ADMITTED_PHASES = frozenset({"assisted_resolution", "committing_assisted"})


def conversation_admitted_during_update(gate_reason: str) -> bool:
    """Whether a chat turn may run while ``gate_reason`` closes the repo-writer gate.

    True only while the durable update transaction is VALID and held by the
    authorized assisted resolver — phase ``assisted_resolution`` or
    ``committing_assisted`` — and the process-local latch is either absent (a
    post-restart resume: only the durable marker closes the gate) or the assisted
    latch of that same transaction. Every other closure is a destructive window
    (the apply/replace/rollback prologue, materialization, a corrupt or future
    marker) and keeps refusing. Conversation admission is not repo-writing
    permission: the registry guard still refuses repo tools to a non-resolver.
    """
    from supervisor.update_merge import assisted_writer_gate_reason, read_update_tx_strict

    reason = str(gate_reason or "")
    if not reason:
        return True
    try:
        status, tx = read_update_tx_strict()
    except Exception:
        return False
    if status != "valid" or str(tx.get("phase") or "") not in _CONVERSATION_ADMITTED_PHASES:
        return False
    if reason.startswith("managed_update_tx:"):
        return True
    return reason == assisted_writer_gate_reason(tx)


def wake_gate_refusal() -> str:
    """Why the owner-conversation gate refuses a consciousness wake-up ("" while open), WITHOUT
    the owner's lock notice: a refused wake is the alarm's typed reason, retried quietly, never a
    "🔒" line in Main at every attempt."""
    from ouroboros.server_restart import agent_restart_pending
    if agent_restart_pending():
        return "restart_pending"
    reason = _pool().repo_writer_admission_closed()
    return "" if not reason or conversation_admitted_during_update(reason) else "repo_writer_gate_closed"


def owner_conversation_gate_open() -> bool:
    """The repository-writer gate admits an owner chat turn: open, or a resolver-held update. Quiet."""
    reason = _pool().repo_writer_admission_closed()
    return not reason or conversation_admitted_during_update(reason)


def owner_conversation_admitted(chat_id: int, client_message_id: str = "") -> bool:
    """Admit one owner chat turn (``owner_conversation_gate_open``); a refused turn gets the pool's
    existing lock notice (``_repo_writer_turn_allowed``), naming the submission it answers."""
    return owner_conversation_gate_open() or bool(_pool()._repo_writer_turn_allowed(chat_id, client_message_id))


def marker_window_closed() -> bool:
    """The one closure that admits the conversation yet forbids a fresh carrier: a resolver-held
    update (its conflict markers may be in the tree a spawned child would import). A physical fact
    for the degraded executor (owner decision 2A), never a judgment about load."""
    reason = _pool().repo_writer_admission_closed()
    return bool(reason) and conversation_admitted_during_update(reason)


def preload_owner_control_path() -> list[str]:
    """Import the server's owner-control path while the live tree is still clean.

    The assisted merge is about to write conflict markers into the checkout this
    process imports from. A module already in ``sys.modules`` keeps working; a
    function-local import that first runs AFTER the markers land raises
    SyntaxError on a conflicted file — the late receipt-wait import that broke
    the update conversation (#283). Called right after the resolver readiness
    proof and before a boot re-materialization. Best effort: a failure is logged
    and returned, never a reason to refuse the update (it would degrade the
    conversation, not the update). First-party packages are discovered so newly
    added function-local imports do not escape a manually maintained list.
    The tool catalog still loads through the chat agent's registry, preserving
    its admission rules and module-failure diagnostics.
    """
    import importlib
    import pkgutil

    failed: list[str] = []
    pending = ["ouroboros", "supervisor"]
    while pending:
        name = pending.pop()
        try:
            module = importlib.import_module(name)
            # Discover from the imported package's own path, including nested
            # packages. The packaged server runs embedded Python against the
            # materialized repo; this never depends on the process cwd or a
            # hand-maintained frozen module list.
            if hasattr(module, "__path__"):
                pending.extend(info.name for info in pkgutil.iter_modules(
                    module.__path__, module.__name__ + ".",
                ))
        except Exception:
            log.warning("owner control path preload: %s failed", name, exc_info=True)
            failed.append(name)
    try:
        from ouroboros.tools.registry import ToolRegistry

        ToolRegistry(pathlib.Path(_pool().REPO_DIR), pathlib.Path(_pool().DRIVE_ROOT))
    except Exception:
        log.warning("owner control path preload: tool catalog failed", exc_info=True)
        failed.append("ouroboros.tools.*")
    return failed


# ---------------------------------------------------------------------------
# the worker-side bus (a carrier's broadcasts travel as frames)
# ---------------------------------------------------------------------------

_WORKER_BUS: Any = None


def bind_worker_bus(out_q: Any) -> None:
    """A worker process names its event bus once: a carrier's ``task_named`` goes out as a frame."""
    global _WORKER_BUS
    _WORKER_BUS = out_q


def _broadcast_task_named(msg: dict) -> None:
    """Bridge broadcast callback for admission naming (kept tiny + fail-soft); in a worker, a frame."""
    try:
        if _WORKER_BUS is not None:
            _WORKER_BUS.put(dict(msg))
            return
        from supervisor.message_bus import try_get_bridge

        bridge = try_get_bridge()
        if bridge is not None:
            bridge.broadcast(msg)
    except Exception:
        log.debug("task_named broadcast failed", exc_info=True)


def name_wake_on_admission(drive_root: Any, task: dict, broadcast: Optional[Callable[[dict], None]] = None) -> None:
    """Publish the wake's cause before its first frame, using the namer's enrichment seam."""
    from ouroboros.task_finalization import wake_cause_title
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    name = wake_cause_title(task.get("metadata") or {})
    if not name:
        return
    try:
        task_id = str(task["id"])
        current = load_task_result(drive_root, task_id) or {}
        write_task_result(drive_root, task_id, current.get("status") or STATUS_RUNNING, suggested_name=name)
        (broadcast or _broadcast_task_named)({"type": "task_named", "task_id": task_id, "suggested_name": name})
    except Exception:
        log.debug("wake admission title failed for %s", task.get("id"), exc_info=True)


# ---------------------------------------------------------------------------
# the lane's doors
# ---------------------------------------------------------------------------

def handle_chat_direct(
    chat_id: int,
    text: str,
    image_data: Optional[Union[Tuple[str, str], Tuple[str, str, str]]] = None,
    task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
    realtime_turn: Any = None,
) -> Optional[Dict[str, Any]]:
    """Admit one owner turn: to the conversation lane, or inline for a live call.

    Returns once the turn is admitted (its RUNNING row published, its start
    receipt sent) or refused; the body runs in the carrier. A live call keeps
    this thread (``executor=inline_call``): its host port is process-local.
    The lane's admitted RUNNING meta (its ``task.id``), None when refused, deferred or inline.
    """
    # The admission is a short execution of its own: a door that stalls before
    # deciding still says where it stands (server_liveness reads the registry's
    # executions); the carrier's own receipt covers the turn from its dispatch.
    from supervisor.active_activity import get_direct_activity_registry

    client_msg_id = _client_message_id(task_metadata)
    registry = get_direct_activity_registry()
    execution = registry.begin_execution(chat_id, client_message_id=client_msg_id)
    try:
        if not owner_conversation_admitted(chat_id, client_msg_id):
            return
        if realtime_turn is not None:
            from supervisor.inline_carrier import run_inline_turn

            run_inline_turn(chat_id, text, image_data, task_constraint, task_metadata,
                            realtime_turn=realtime_turn, executor=EXECUTOR_INLINE_CALL, execution=execution)
            return
        return admit_owner_turn(chat_id, text, image_data, task_constraint=task_constraint, task_metadata=task_metadata)
    finally:
        registry.end_execution(execution)


def _client_message_id(task_metadata: Optional[dict]) -> str:
    if not task_metadata:
        return ""
    ref = task_metadata.get("origin_message_ref")
    if isinstance(ref, dict) and ref.get("client_message_id"):
        return str(ref.get("client_message_id") or "")
    return str(task_metadata.get("client_message_id") or "")


def _seed_task(
    chat_id: int, text: str, image_data: Any, task_constraint: Optional[dict], task_metadata: Optional[dict],
) -> Dict[str, Any]:
    """The admitted seed: identity, address, the owner's words and origin facts, the inline image as a durable source,
    the name of a late input's head. Everything else is the carrier's first step (``prepare_conversation_task``)."""
    task: Dict[str, Any] = {
        "id": uuid.uuid4().hex[:8],
        "type": "task",
        "chat_id": chat_id,
        "text": str(text or ""),
        "_is_direct_chat": True,
        "_fresh_identity": True,  # host-only, set at this mint; no task row or client metadata carries it
    }
    if task_constraint:
        task["task_constraint"] = dict(task_constraint)
    if task_metadata:
        task["metadata"] = dict(task_metadata)
        # The ingress-captured origin identity rides on the TASK RECORD so a
        # later post-hoc "Turn into project" reads it from the persisted
        # result instead of re-deriving identity from content.
        origin_ref = task_metadata.get("origin_message_ref")
        if isinstance(origin_ref, dict) and origin_ref:
            task["origin_message_ref"] = dict(origin_ref)
            origin_text = task_metadata.get("origin_message_text")
            if isinstance(origin_text, str) and origin_text:
                task["origin_message_text"] = origin_text
        # Project-thread conversations scope the direct lane to the
        # project's memory (knowledge/journal/workpad sections).
        pid = str(task_metadata.get("project_id") or "").strip()
        if pid:
            task["project_id"] = pid
    from ouroboros.project_admission import fresh_seed_unscoped

    if fresh_seed_unscoped(task):  # this mint is the turn's admission: no enqueue will attest it (#2555)
        task["_project_scope_none"] = True
    if image_data:
        # image_data is (base64, mime) or (base64, mime, caption). The bytes become an
        # actor source the carrier reads back (the row stays small); the caption still
        # seeds task['text'] so a caption-only message keeps working even when nothing stages.
        _store_inline_image(task, image_data)
        if len(image_data) > 2 and image_data[2]:
            task["image_caption"] = image_data[2]
            if not task["text"]:
                task["text"] = image_data[2]
    naming = _late_input_naming(_pool().DRIVE_ROOT, task)  # a late input's head is admitted named by what it answers
    if naming and naming[0]:
        task["suggested_name"] = naming[0]
    return task


def _store_inline_image(task: Dict[str, Any], image_data: Any) -> None:
    import base64

    from ouroboros.artifacts import store_actor_source_bytes

    task["image_source_ref"] = store_actor_source_bytes(
        _pool().DRIVE_ROOT, str(task["id"]), category="context_checkpoints", source_id="inline-image",
        extension="image", data=base64.b64decode(str(image_data[0] or ""), validate=True))
    task["image_mime"] = str(image_data[1] or "image/jpeg")
    task["drive_root"] = str(_pool().DRIVE_ROOT)


def _late_input_naming(root: Any, task: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """``project_naming.late_input_names`` of a late answer's or a press's head: what it answers (the accepted
    question, the card's title as the stored press recorded it) and the owner's words (its text before the
    first step); None without that provenance. Read at admission and by the carrier's namer, never the frame."""
    from ouroboros.gateway.card_decision import card_press_name
    from ouroboros.owner_quiz import late_answer_name
    from ouroboros.project_naming import late_input_names

    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    answered = (late_answer_name(metadata["late_answer"]) if metadata.get("late_answer") is not None
                else card_press_name(root, metadata.get("card_press")))
    return None if answered is None else late_input_names(metadata, answered, str(task.get("text") or ""))


def admit_owner_turn(
    chat_id: int, text: str, image_data: Any = None, *, task_constraint: Optional[dict] = None,
    task_metadata: Optional[dict] = None,
) -> Optional[Dict[str, Any]]:
    """Admit one owner turn to the lane; the RUNNING meta, or None when refused or deferred.

    One transaction with the update owner (Q + the repo-writer gate lock): the gate,
    the restart intake, the row. A Restart accepted meanwhile finds started work; while
    another task's restart is only requested the turn starts and joins that request's
    started work; only the physical drain saves it on the ordinary durable queue. The row's
    barrier and its carrier follow outside that transaction (``dispatch_admitted_turn``).
    """
    from ouroboros.server_restart import agent_restart_draining, agent_restart_pending, note_started_restart_work
    from supervisor import queue
    from supervisor.active_activity import reply_expected
    from supervisor.conversation_lane import admit_turn, dispatch_admitted_turn, replenish_after_claim

    client_msg_id = _client_message_id(task_metadata)
    task = _seed_task(chat_id, text, image_data, task_constraint, task_metadata)
    root = pathlib.Path(str(task.get("drive_root") or "") or _pool().DRIVE_ROOT)
    intake = open_owner_intake(task, root)  # before its row exists: a carrier only ever replaces the fact
    meta = None
    try:
        with queue._queue_lock, _pool()._repo_writer_gate_lock:
            if not owner_conversation_admitted(chat_id, client_msg_id):
                return None
            if agent_restart_pending() and queue.RETURNS_NAMED:
                _pool().send_with_budget(chat_id, "Restart intake is closed. This turn was not accepted; send it after Restart.",
                    **_host_operation_failure(task_metadata), role="system", system_type="task_not_started")
                return None
            deferred = agent_restart_draining()
            meta = None if deferred else admit_turn(
                task, chat_id=chat_id, client_message_id=client_msg_id, quiet=False,
                reply_expected=reply_expected({"metadata": task_metadata or {}}),
            )
            if meta is not None:
                note_started_restart_work(task)
        if deferred:
            _defer_chat_for_restart(task, chat_id, task_metadata)
            return None
        if meta is not None:
            dispatch_admitted_turn(meta)  # the barrier outside Q, then the carrier (or the degraded executor)
            replenish_after_claim()  # the spare the turn took is replaced now, or its own carrier started now (4A)
        _announce_turn_start({"task": task, "chat_id": chat_id, "client_msg_id": client_msg_id, "kind": "direct_chat"})
        return meta
    except Exception as e:
        _report_direct_chat_error({"task": task, "chat_id": chat_id, "client_msg_id": client_msg_id,
                                   "task_metadata": task_metadata}, e)
        return None
    finally:
        if intake and meta is None:  # no admitted row names the fact
            close_owner_intake(task, root)


def open_owner_intake(task: Dict[str, Any], root: Any) -> str:
    """My human's next words in this chat reach this owner turn from here on
    (``conversation_lease.owner_conversation_key``): the host's fact at admission, its carrier's from the moment it
    takes the turn. The key; else "" with no fact of this attempt left standing (a failed take retires the host's,
    ``conversation_lease.retire_author_intake``), so the words start their own turn."""
    from ouroboros.conversation_lease import (
        open_author_intake,
        owner_conversation_key,
        retire_author_intake,
        task_attempt,
    )

    key, attempt = owner_conversation_key(task), task_attempt(task)
    if not key:
        return ""
    try:
        if open_author_intake(root, str(task["id"]), key, attempt=attempt):
            return key
    except Exception:
        log.warning("Owner turn %s reads no follow-up: its intake fact was not written", task.get("id"), exc_info=True)
    try:
        retire_author_intake(root, str(task["id"]), key, attempt=attempt)
    except Exception:
        log.warning("Owner turn %s: an intake fact it could not take may still stand", task.get("id"), exc_info=True)
    return ""


def close_owner_intake(task: Dict[str, Any], root: Any) -> None:
    """The writer's own fact for this attempt goes (``close_author_intake(own=True)``): an arrival starts its own
    turn."""
    from ouroboros.conversation_lease import close_author_intake, task_attempt

    try:
        close_author_intake(root, str(task["id"]), own=True, attempt=task_attempt(task))
    except Exception:
        log.debug("owner intake of %s not closed", task.get("id"), exc_info=True)


def handle_wake_direct(
    chat_id: int,
    text: str,
    task_metadata: Optional[dict],
    on_finished: Optional[Callable[[str, bool], None]] = None,
    bind_input: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Admit a self-initiated Main turn (a consciousness wake-up) to the lane; a typed receipt.

    The same gates as ``handle_chat_direct`` decide admission, but a refused wake
    gets ``{"admitted": False, "task_id": "", "reason": <typed>}`` synchronously
    instead of a chat notice. An admitted wake owns its RUNNING row before this
    returns (the census lists it, Stop reaches it through pooled custody, the
    liveness read sees it); its body runs in a carrier and its end reaches the
    clock as the pooled terminal it is (``consciousness_carriers.finish_pooled_wake``),
    so ``on_finished`` is kept for callers that still pass it and is not called
    from here. ``bind_input(task)`` runs once the wake has its id and before it is
    dispatched — the alarm binds its immutable observation source there; a binder
    failure leaves the complete text as the wake's only input.
    """
    from supervisor import queue
    from supervisor.conversation_lane import admit_turn, dispatch_admitted_turn, replenish_after_claim

    closed = wake_gate_refusal()
    if closed:
        return {"admitted": False, "task_id": "", "reason": closed}
    from supervisor.state import budget_remaining, control_is, load_state
    from ouroboros.consciousness import panic_blocks_wake

    current = load_state()
    if panic_blocks_wake(_pool().DRIVE_ROOT) or not control_is(current, "bg_consciousness_enabled", True):
        return {"admitted": False, "task_id": "", "reason": "consciousness_disabled_or_unknown"}
    try:
        remaining = budget_remaining(current, strict=True)
    except Exception:
        return {"admitted": False, "task_id": "", "reason": "cost_accounting_unavailable"}
    from supervisor.worker_assignment import needs_paid_admission

    # A reached global limit refuses paid admission only: a wake on a $0 subscription or local route runs.
    if remaining <= 0 and needs_paid_admission({"metadata": dict(task_metadata or {})}):
        return {"admitted": False, "task_id": "", "reason": "budget_exhausted"}
    task = _seed_task(int(chat_id), str(text or ""), None, None, dict(task_metadata or {}))
    task_id = str(task["id"])
    if bind_input is not None:
        try:
            bind_input(task)
        except Exception:
            log.warning("wake %s input binding failed; the complete text stays its input", task_id, exc_info=True)
    try:
        with queue._queue_lock, _pool()._repo_writer_gate_lock:
            closed = wake_gate_refusal()  # the gate may have closed since the check above: silent and typed
            if closed:
                return {"admitted": False, "task_id": "", "reason": closed}
            meta = admit_turn(task, chat_id=int(chat_id), quiet=True, reply_expected=False)
    except Exception:
        log.warning("wake %s could not be admitted to the conversation lane", task_id, exc_info=True)
        return {"admitted": False, "task_id": "", "reason": "admission_failed"}
    try:
        dispatch_admitted_turn(meta)  # the barrier outside Q, then the carrier (or the degraded executor)
    except Exception:
        log.warning("wake %s admitted; its carrier could not start now (the lane's next pass retries)", task_id,
                    exc_info=True)
    replenish_after_claim()  # the wake's own carrier starts now; a wake never takes the owner's spare (4A)
    _announce_turn_start({"task": task, "chat_id": int(chat_id), "client_msg_id": "", "kind": "direct_chat"})
    return {"admitted": True, "task_id": task_id, "reason": ""}


def _degrade_if_no_carrier_can_start(meta: Dict[str, Any]) -> None:
    """Under Q (``conversation_lane.dispatch_admitted_turn``, after the row's barrier): a row no carrier holds takes
    the degraded executor only on a physical fact (2A).

    A spare still booting is a carrier that will come: the row waits for it. A
    marker window, a parked carrier or a refused spawn means none will: the same
    worker body runs on a thread of this process, disclosed in the chat and the vitals.
    """
    from supervisor.conversation_lane import lane_carriers
    from supervisor.inline_carrier import EXECUTOR_INLINE_DEGRADED, refused_carrier_reason, start_inline_turn

    if meta.get("worker_id") is not None or meta.get("_unpublished"):  # held, or its row not on disk yet
        return
    if any(w.booting and w.busy_task_id is None for w in lane_carriers()):
        return
    reason = refused_carrier_reason()
    if reason:
        start_inline_turn(meta, executor=EXECUTOR_INLINE_DEGRADED, reason=reason)


# ---------------------------------------------------------------------------
# the carrier's first step
# ---------------------------------------------------------------------------

class ConversationRefusal(Exception):
    """The carrier refuses the admitted turn before any model work: a typed cause and the owner's words."""

    def __init__(self, code: str, owner_text: str) -> None:
        super().__init__(code)
        self.code, self.owner_text = str(code), str(owner_text)


def prepare_conversation_task(agent: Any, task: Dict[str, Any], drive_root: Any, *,
                              broadcast: Optional[Callable[[dict], None]] = None) -> Dict[str, Any]:
    """Layer the owner's input onto the admitted seed, in the process that runs the turn.

    Everything that once happened in the server between the registration and the
    model: the late-answer / card-press text, the inline image, the staged
    attachments (a fully rejected set refuses the turn: ``ConversationRefusal``),
    the Project binding under the room basis the turn then carries like queued
    work (#2547), consciousness authority, the task contract, the frozen tool
    profile and a wake's cause (a late input's head: its admission name) as its title.
    Mutates and returns ``task``; the first-step facts leave it (``PREPARE_KEY``).
    """
    from ouroboros.consciousness_authority import apply_consciousness_authority
    from ouroboros.contracts.task_contract import attach_task_contract
    from supervisor.conversation_lane import PREPARE_KEY

    root = pathlib.Path(drive_root)
    if task.get("_wake_prepare") is not None:
        from ouroboros.consciousness_wake import prepare_wake

        try:
            prepare_wake(task, root)  # observe the window, render, bind the source: the carrier's work
        except Exception:
            task.pop("_wake_prepare", None)
            log.warning("wake %s preparation failed; the admitted text stays its input", task.get("id"), exc_info=True)
    task_metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    text = str(task.get("text") or "")
    if task_metadata.get("late_answer") is not None:
        # A LATE quiz answer: the owner's row holds only their words; the
        # model reads the card they answered, rebuilt from the stored
        # block (disclosed when unreadable) -- the one shared builder.
        from ouroboros.owner_quiz import late_answer_model_text

        task["text"] = late_answer_model_text(root, task_metadata.get("late_answer"), text)
    elif task_metadata.get("card_press") is not None:
        # A press on a finished author's card: the same, with the card frame rebuilt from the stored press.
        from ouroboros.gateway.card_decision import card_press_model_text

        task["text"] = card_press_model_text(root, task_metadata.get("card_press"), text)
    # v6.52.0 (P1, full desktop unify): route the WHOLE desktop attachment set
    # (any type) through the shared staging substrate so the agent gets EVERY
    # attachment — images natively via attachment_images + non-images via the
    # read_file(root='artifact_store', path='attachments/...') manifest — exactly
    # like the CLI/API/GAIA path. The uploads are resolved from data/uploads/ in
    # ws._chat_attachment_uploads and carried as task['metadata'].
    # On a non-empty manifest we DROP the legacy inline image so the same
    # image is not double-injected; on absent/empty uploads (older clients, the
    # single-image seam) the legacy inline source stays untouched.
    uploads = task_metadata.get("chat_attachment_uploads")
    if uploads:
        from ouroboros.artifacts import (
            attachment_manifest_all_rejected,
            attachment_manifest_has_rejections,
            stage_task_attachments,
        )
        from ouroboros.gateway.tasks import _render_attachment_lines

        manifest = stage_task_attachments(root, str(task["id"]), uploads)
        # Partial staging is the default (В25c, capinv-447); a FULLY-rejected
        # set stays atomic — the task would start with none of its material.
        if attachment_manifest_all_rejected(manifest):
            from ouroboros.artifacts import remove_staged_attachments
            from ouroboros.project_dialogue import routing_refusal_cause

            remove_staged_attachments(manifest)
            _record_attachment_rejections(task, int(task.get("chat_id") or 0), task_metadata, manifest, started=False)
            # No mind will read this message: one short typed mark in owner words
            # (closed class), never the model's block; the files are on the owner's
            # bubble and the per-file causes in the diagnostic row.
            cause = routing_refusal_cause("promote_chat_to_task", "rejected", "attachment_admission_rejected")
            count = len(_rejected_rows(manifest))
            raise ConversationRefusal("attachment_admission_rejected",
                                      f"{cause} ({count} file{'' if count == 1 else 's'})" if count else cause)
        from ouroboros.artifacts import attachment_manifest_projection
        authority = attachment_manifest_projection(root, str(task["id"]), manifest)
        rendered = _render_attachment_lines(authority)
        if attachment_manifest_has_rejections(manifest):
            # The turn starts with the rejected rows in its own block and
            # tells the owner itself; the host keeps only the diagnostic.
            _record_attachment_rejections(task, int(task.get("chat_id") or 0), task_metadata, manifest, started=True)
        if manifest:
            manifest = [dict(row) for row in manifest]
            task["drive_root"] = str(root)
            task.update(authority)
            task["attachments"] = authority["attachment_manifest"]
            task["attachment_images"] = [
                m for m in manifest
                if str(m.get("status") or "staged") == "staged" and m.get("is_image")
            ]
            if rendered:
                task["text"] = f"{task.get('text') or ''}\n\n[ATTACHMENTS]\n{rendered}\n[END_ATTACHMENTS]"
            for key in ("image_base64", "image_mime", "image_source_ref"):
                task.pop(key, None)
    # A rejected initial UI task must leave no partial project assignment.
    # Bind only after all declared attachments have passed admission, under the
    # room basis the turn then carries like queued work, so restore and Resume
    # readmit it on that basis (#2547).
    pid = str(task.get("project_id") or "").strip()
    from ouroboros.project_admission import admitted_folder

    # A basis the turn already carries (an earlier attempt's bind, restored with its saved work) is its
    # admission authority: a resumed or retried first step never recaptures it from the room's current
    # row (#2547: a sibling's later folder must not move folderless work). Placement replaces the basis.
    if pid and admitted_folder(task.get("_project_admission"), pid) is None:
        try:
            from ouroboros.project_admission import direct_turn_basis
            from ouroboros.projects_registry import bind_task_to_project

            basis = direct_turn_basis(root, pid)
            bind_task_to_project(
                root, task["id"], pid, int(task.get("chat_id") or 0),
                origin=_pool()._origin_from_mapping(task_metadata, absent="mid_task_no_origin"),
                admission_basis=basis,
            )
            task["_project_admission"] = basis
            task.pop("_project_scope_none", None)
            if broadcast is not None:  # the row (snapshot, pooled park, restore) learns the basis by frame
                broadcast({"type": "direct_turn_bound", "task_id": str(task["id"]), "task_attempt": int(task.get("_attempt") or 1),
                           "facts": {"_project_admission": basis, "project_id": pid}})
        except Exception as exc:
            _pool()._report_binding_failure(task["id"], pid, exc, path="direct_project_turn")
    if not task.get("text"):
        from ouroboros.chat_uploads import attachment_placeholder
        task["text"] = attachment_placeholder(task.get("image_source_ref") or task.get("image_base64"), task_metadata)
    # A Main turn is named lazily: the turn queue fires the namer on the first
    # non-addressing tool call (owner decision Q7=A, 16.09), so a greeting costs
    # no naming call and a working turn gets a title as its block becomes the
    # task card. A Project-room turn is named by its room. A consciousness
    # wake-up derives its level's disabled_tools and mode cap here, before the
    # contract reads them (consciousness_authority), and publishes its cause as its title.
    apply_consciousness_authority(task)
    from ouroboros.settings_integrity import record_subagents_admission
    record_subagents_admission(task, root)
    from ouroboros.route_choice import capture_lead  # the admitted primary (route_choice)
    capture_lead(task, root)
    attach_task_contract(task)
    from ouroboros.tool_access_roots import freeze_tool_profile
    freeze_tool_profile(task)
    if wake_cause(task.get("metadata")):
        name_wake_on_admission(root, task, broadcast)
    elif task.get("suggested_name") and broadcast is not None:  # admitted named (a late input's head): born named
        broadcast({"type": "task_named", "task_id": str(task["id"]), "suggested_name": task["suggested_name"]})
    task.pop(PREPARE_KEY, None)
    return task


def conversation_exception_terminal(agent: Any, task: Dict[str, Any], exc: BaseException, event_queue: Any,
                                    pool_drive_root: Any, *, quiet: bool = False) -> None:
    """A conversation turn's exception is that turn's own terminal, recorded in its carrier before the
    exception unwinds (owner decision 1A): the stored result — typed for a before-start Stop or an
    exhausted budget (the closed class's words, the control's reason), the exception terminal otherwise —
    and the owner's keyed final as a ``send_message`` frame (``task_terminal_status``; a host operation's
    ``origin_message_ref``), as the old lane's ``_report_direct_chat_error`` wrote them. A saved exact pause of this attempt owns
    the id (``death_returns_to_pause``) and unreadable pause evidence is recovery's to judge: neither is
    concluded over. ``quiet`` (a wake) records the terminal and sends no final. Never raises."""
    from ouroboros.budget_pause import budget_pause_row, death_returns_to_pause

    task_id = str(task.get("id") or "")
    try:
        pause = budget_pause_row(task.get("budget_drive_root") or pool_drive_root, task_id)
        if death_returns_to_pause(pause, int(task.get("_attempt") or 1)):
            return
    except Exception:
        log.warning("Pause evidence of turn %s is unreadable; recovery decides its exception", task_id, exc_info=True)
        return
    before_start = bool(getattr(exc, "before_task_start", False))
    control = str(getattr(exc, "control_reason", "") or "")
    stopped = before_start and control in {"cancelled", "finalize_requested"}
    exhausted = before_start and control == "budget_exhausted"
    from ouroboros.deployment_extensions import HOLD_REASON, hold_text, record_held_task

    held = before_start and control == HOLD_REASON  # a required extension this carrier does not run
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    correlation = _host_operation_failure(metadata).get("progress_meta", {})
    origin = {"origin_message_ref": correlation["origin_message_ref"]} if "origin_message_ref" in correlation else {}
    try:
        if held:
            text = hold_text()
            record_held_task(agent.env.drive_root, task, text, **origin)
        elif stopped or exhausted:
            from ouroboros.outcomes import infra_failed_axes
            from ouroboros.subagents import effort_result_fields
            from ouroboros.task_results import STATUS_CANCELLED, STATUS_FAILED, _TRULY_TERMINAL_STATUSES, write_task_result

            text = ("⏹ The owner stopped this chat turn before it started; no work was done." if stopped else
                    "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings.")
            write_task_result(
                agent.env.drive_root, task_id, STATUS_CANCELLED if stopped else STATUS_FAILED,
                _field_projector=lambda current, incoming: (None if current.get("status") in _TRULY_TERMINAL_STATUSES
                                                            else incoming),
                _terminal_observed=True, result=text, reason_code=control,
                outcome_axes=infra_failed_axes(control, lifecycle="cancelled" if stopped else "failed",
                                               review_trigger="before_task_start"),
                **origin, **effort_result_fields(task))
        else:
            from ouroboros.agent import _task_exception_terminal

            text = f"⚠️ Error during processing: {type(exc).__name__}: {exc}"
            _task_exception_terminal(agent.env, task, exc, agent.env.drive_path("logs"), keep_terminal=True)
    except Exception:
        log.warning("Terminal of turn %s was not recorded", task_id, exc_info=True)
        return
    if quiet:  # a wake's failure is the clock's and the supervisor's to report, never a chat final
        return
    progress_meta = {"task_terminal_status": "cancelled" if stopped else "failed"}
    if "origin_message_ref" in correlation:  # an explicit key: the outbound contract scan vouches for the set
        progress_meta["origin_message_ref"] = correlation["origin_message_ref"]
    if metadata.get("initiator"):
        progress_meta["initiator"] = str(metadata["initiator"])
    try:  # the owner reads the failure once, keyed to the turn: the carrier's frame, the supervisor's send
        event_queue.put({"type": "send_message", "task_id": task_id, "chat_id": int(task.get("chat_id") or 0),
                         "text": text, "progress_meta": progress_meta, "role": "system",
                         "system_type": "task_not_started" if stopped or exhausted or held else "task_error"})
    except Exception:
        log.debug("keyed final of turn %s failed", task_id, exc_info=True)


def refusal_terminal(agent: Any, task: Dict[str, Any], refusal: ConversationRefusal, event_queue: Any) -> list:
    """The carrier's refusal as the turn's own terminal: the owner's words as a host notice, no model work."""
    from ouroboros.agent_task_pipeline import emit_task_results

    events: list = []
    emit_task_results(
        agent.env, agent.memory, agent.llm, events, task, refusal.owner_text,
        {"terminal_origin": "host_notice", "execution_status": "infra_failed", "reason_code": refusal.code},
        {"reasoning_notes": [], "tool_calls": []},
        getattr(agent, "_task_started_ts", None) or time.time(), agent.env.drive_path("logs"),
        ctx=getattr(agent.tools, "_ctx", None), event_queue=event_queue)
    return events


class TurnPhaseFrames:
    """The one producer of a task's start phases for the census: its execution (``TurnExecution``) observes the
    task's own frames, and each phase it newly passes leaves as a ``direct_turn_phase`` frame naming its attempt and
    execution, and the worker slot carrying it, which ``conversation_lane.handle_direct_turn_phase`` lands on the
    RUNNING row that slot holds. A conversation turn's carrier (``CarrierTurn``) and a pooled task's endpoint
    (``pooled_turn_phases``) both publish through it."""

    def __init__(self, task_id: str, attempt: int, execution: Any, out_q: Any, worker_id: Any = None,
                 phase_generation: str = "") -> None:
        self.task_id, self.attempt, self.execution, self._out_q = task_id, attempt, execution, out_q
        self.worker_id = worker_id
        self.phase_generation = str(phase_generation or "")

    def observe(self, kind: str, data: Dict[str, Any]) -> None:
        before = self.execution.phase
        self.execution.observe(kind, data)
        if self.execution.phase != before:
            self.publish()

    def publish(self) -> None:
        """The execution's current phase as one frame (each new phase, and a pooled task's first fact)."""
        frame = {"type": "direct_turn_phase", "task_id": self.task_id, "task_attempt": self.attempt,
                 "execution_id": self.execution.execution_id, "phase": self.execution.phase}
        if self.worker_id is not None:
            frame["worker_id"] = self.worker_id
        if self.phase_generation:
            frame["phase_generation"] = self.phase_generation
        try:
            self._out_q.put(frame)
        except Exception:
            log.debug("direct_turn_phase frame failed for %s", self.task_id, exc_info=True)


def pooled_turn_phases(task: Dict[str, Any], out_q: Any, worker_id: Any = None) -> TurnPhaseFrames:
    """A pooled task's start phases (#2862), so its Project card says Preparing with its step until its first Main
    call starts, as a carried turn's does. An inline executor's entry keeps its one execution (as ``CarrierTurn``);
    any other gets one that only observes, never a direct turn's receipt, liveness entry or dispatch fact. Its
    first frame (``started``) leaves at once: the slot's row says Preparing from its start, and this execution,
    not a late frame of an earlier one, owns the row before any other frame of its own. A body the Presence carrier
    runs (``worker_process._run_task_body``) has no such frame: it re-admits from durable facts before any agent
    exists, so a refused or unreadable turn claims no phase and its first frame is its own first context or model
    fact (its phase generation, not that frame, fences an earlier execution)."""
    from supervisor.active_activity import TurnExecution, get_direct_activity_registry

    task_id = str(task.get("id") or "")
    entry = get_direct_activity_registry().get(task_id)
    execution = entry.execution if entry is not None and entry.execution is not None else None
    if execution is None:
        execution = TurnExecution(int(task.get("chat_id") or 0), task_id=task_id)
        execution.advance("started")
    frames = TurnPhaseFrames(task_id, int(task.get("_attempt") or 1), execution, out_q, worker_id,
                             task.get("_phase_generation", ""))
    if not (task.get("_presence_run") or task.get("_presence_turn")):
        frames.publish()
    return frames


class CarrierTurn:
    """A conversation turn's execution facts inside its carrier: the receipt (``TurnExecution``)
    seeded from the admission's wall clock, the endpoint hooks (the Stop marker, the namer, the
    frame observer whose start phases ``TurnPhaseFrames`` publishes), the thread's store meter and
    an owner turn's intake fact, held from the take to the end."""

    def __init__(self, task: Dict[str, Any], drive_root: Any, out_q: Any, worker_id: Any = None) -> None:
        from supervisor.active_activity import TurnExecution, get_direct_activity_registry
        from supervisor.conversation_lane import PREPARE_KEY

        facts = task.get(PREPARE_KEY) if isinstance(task.get(PREPARE_KEY), dict) else {}
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        self.task_id, self.drive_root, self._out_q = str(task.get("id") or ""), drive_root, out_q
        self.attempt = int(task.get("_attempt") or 1)
        self.execution = TurnExecution(int(task.get("chat_id") or 0), str(facts.get("client_message_id") or ""),
                                       task_id=self.task_id, owner_turn=bool(facts.get("reply_expected")))
        entry = get_direct_activity_registry().get(self.task_id)  # an inline executor: this process's own entry
        if entry is not None and entry.execution is not None:
            self.execution = entry.execution  # one execution per turn; the liveness read sees the body's phases
        self.execution.accepted_at = str((task.get("origin_message_ref") or {}).get("ts") or "")
        self.execution.census = dict(facts.get("census") or {})
        self.execution.worker_id = worker_id
        admitted_at = float(facts.get("admitted_at") or 0.0)
        if admitted_at:
            # The owner's wait is measured from the admission's wall clock: the dispatch wait
            # the carrier never saw is part of it, so the first marks are placed back there.
            offset = max(0.0, time.time() - admitted_at)
            then = time.monotonic() - offset
            self.execution.entered_at = admitted_at
            self.execution.marks = [("accepted", then), ("admitted", then)]
        self.execution.advance("started")
        from supervisor.active_activity import carry_turn_execution

        carry_turn_execution(self.task_id, self.execution)  # the turn's own wait fact reads it in this carrier
        self._phases = TurnPhaseFrames(self.task_id, self.attempt, self.execution, out_q, worker_id,
                                       task.get("_phase_generation", ""))
        self.quiet = bool(facts.get("quiet"))
        self.initiator = str(metadata.get("initiator") or "")
        self.intake = open_owner_intake(task, drive_root)  # the host's fact becomes this carrier's, whatever follows
        self._namer = None
        if not task.get("project_id") and not wake_cause(metadata) and not task.get("_presence_turn"):
            from ouroboros.project_naming import spawn_turn_namer

            naming = _late_input_naming(drive_root, task)  # a late input's head: what it answers, never its frame
            if naming is None or naming[1]:  # an authored title needs no namer; Light's failure keeps a standing name
                self._namer = lambda: spawn_turn_namer(  # noqa: E731
                    drive_root, self.task_id, naming[1] if naming else str(task.get("text") or ""),
                    broadcast=lambda msg: out_q.put(dict(msg)), fallback_candidates=(task.get("suggested_name") or "",),
                )

    def observe(self, kind: str, data: Dict[str, Any]) -> None:
        self._phases.observe(kind, data)

    def hooks(self) -> Dict[str, Any]:
        return {"initiator": self.initiator, "on_first_work": self._namer, "on_frame": self.observe}

    def ended(self) -> None:
        """The turn left this carrier: an owner turn that never dispatched records its receipt here."""
        from supervisor.active_activity import carry_turn_execution, record_direct_turn

        carry_turn_execution(self.task_id, None)
        if self.intake:  # however the body ended, no later arrival is accepted for it
            close_owner_intake({"id": self.task_id, "_attempt": self.attempt}, self.drive_root)
        self._namer = None  # the namer's closure holds the whole task: nothing of the turn's input stays reachable
        self.execution.ended_at = time.monotonic()
        if self.execution.owner_turn and self.execution.phase != "dispatched":
            try:
                record_direct_turn(self.execution.receipt(), drive_root=self.drive_root)
            except Exception:
                log.debug("direct_turn receipt failed for %s", self.task_id, exc_info=True)


def _host_operation_failure(metadata: Optional[dict]) -> dict:
    """A refusal's correlation: the owner submission it answers (only that
    ``Sending...`` settles) and, for the host's preaccepted skill messages,
    their terminal correlation."""
    from ouroboros.task_finalization import host_operation_reply_kwargs

    meta = metadata or {}
    source = meta.get("origin_message_ref") if meta.get("_host_operation") else None
    return host_operation_reply_kwargs(source, "failed", str(meta.get("client_message_id") or ""))


def _rejected_rows(manifest: Any) -> list:
    """The rejected declarations of one staging manifest."""
    return [row for row in (manifest or []) if isinstance(row, dict) and row.get("status") == "rejected"]


def _record_attachment_rejections(
    task: Dict[str, Any], chat_id: int, metadata: Optional[dict], manifest: Any, *, started: bool,
) -> None:
    """One failure-only diagnostic row: the per-file reasons the owner's chat no longer carries."""
    rows = [{key: row.get(key) for key in ("ordinal", "label", "status", "reason", "rule") if row.get(key) is not None}
            for row in _rejected_rows(manifest)]
    try:
        append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "attachments_rejected", "task_id": str(task.get("id") or ""),
            "chat_id": int(chat_id or 0), "started": started,
            "origin_message_ref": (metadata or {}).get("origin_message_ref"), "rejected": rows,
        })
    except Exception:
        log.debug("attachments_rejected row failed", exc_info=True)


def _announce_turn_start(admitted: Dict[str, Any]) -> None:
    """Announce the authoritative start immediately (owner decision 2A): the client's `Sending...`
    retires on this receipt (once the census read it triggers has answered), not on a socket echo;
    the census row carries the same activity<->client_message_id link, and a turn the census never
    lists is still settled by that read."""
    try:
        from supervisor.message_bus import get_bridge

        get_bridge().send_chat_action(
            int(admitted["chat_id"] or 0), "typing", activity_id=str(admitted["task"]["id"]),
            client_message_id=admitted["client_msg_id"], phase="thinking", kind=admitted["kind"],
        )
    except Exception:
        log.debug("Direct-turn start typing announce failed", exc_info=True)


def _defer_chat_for_restart(task: dict, chat_id: int, metadata: Optional[dict]) -> None:
    """Use the ordinary durable queue, carrying the same input and attachment sources.

    The seed carries the turn's preparation obligation (``PREPARE_KEY``, kept by the saved queue across the
    Restart): the worker that runs it after Restart takes the conversation's first step (attachments, Project
    binding, contract) as a carrier would, never the server."""
    from supervisor import queue
    from supervisor.active_activity import reply_expected
    from supervisor.conversation_lane import PREPARE_KEY

    if task.get("image_base64"):
        _store_inline_image(task, (task.pop("image_base64"), task.get("image_mime") or "image/jpeg"))
    task[PREPARE_KEY] = {"admitted_at": time.time(), "quiet": False, "census": {},
                         "reply_expected": reply_expected({"metadata": metadata or {}}),
                         "client_message_id": str((metadata or {}).get("client_message_id") or "")}
    task["drive_root"] = str(_pool().DRIVE_ROOT)
    task["root_task_id"] = str(task["id"])
    admitted = queue.enqueue_with_admission_receipt(task, durable=True)
    blocked = (admitted or {}).get("_admission_blocked") or ""
    if blocked:
        _pool().send_with_budget(chat_id, f"This turn was not accepted ({blocked}). Send it after Restart.",
            **_host_operation_failure(metadata), role="system", system_type="task_not_started")
        return
    from ouroboros.task_finalization import host_operation_reply_kwargs

    meta = metadata or {}
    ack = host_operation_reply_kwargs(meta.get("origin_message_ref") if meta.get("_host_operation") else None,
                                     client_message_id=str(meta.get("client_message_id") or ""))
    _pool().send_with_budget(chat_id, "Saved for execution after Restart.",
                            **ack, role="system", system_type="restart_notice")


def _report_direct_chat_error(admitted: Dict[str, Any], e: BaseException) -> None:
    """Record a direct-turn failure and conclude the turn in the chat."""
    import traceback
    task, chat_id = admitted["task"], admitted["chat_id"]
    client_msg_id, task_metadata = admitted["client_msg_id"], admitted.get("task_metadata")
    if admitted.get("realtime_turn") is not None:
        from supervisor.inline_carrier import _close_failed_call, _report_call_context_refusal

        refusal = _close_failed_call(admitted, e)
        if refusal is not None:
            _report_call_context_refusal(admitted, e, refusal)
            return
    stopped = (getattr(e, "before_task_start", False)
               and getattr(e, "control_reason", "") in {"cancelled", "finalize_requested"})
    exhausted = getattr(e, "before_task_start", False) and getattr(e, "control_reason", "") == "budget_exhausted"
    err_msg = ("⏹ The owner stopped this chat turn before it started; no work was done." if stopped else
               "🚫 Budget exhausted. Task rejected. Please increase TOTAL_BUDGET in settings." if exhausted else
               f"⚠️ Error: {type(e).__name__}: {e}")
    append_jsonl(
        _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
        {
            "ts": utc_now_iso(),
            "type": "direct_chat_stopped" if stopped else "direct_chat_error",
            "task_id": str(task.get("id") or ""),
            "chat_id": int(chat_id or 0),
            "error": repr(e),
            "traceback": str(traceback.format_exc())[:2000],
        },
    )
    if admitted.get("activity") is not None:
        admitted["activity"].reply_handed_off = True  # this failure IS the turn's reply
    try:
        # Key the error final with the turn's activity id so the client
        # concludes exactly this turn (active set, 4A) instead of leaving
        # its `Sending.../Thinking...` state to an unkeyed sweep. If the
        # failure happened before the start announce was broadcast, announce
        # it first: the receipt's census read settles the linked `Sending...`
        # and the keyed final right after concludes the turn's census row.
        failed_task_id = str(task.get("id") or "") if isinstance(task, dict) else ""
        if failed_task_id and client_msg_id:
            _announce_turn_start({**admitted, "kind": admitted.get("kind") or "direct_chat"})
        # The keyed final concludes its own turn and linked Sending; it keeps only the host-operation source.
        correlation = _host_operation_failure(task_metadata).get("progress_meta", {})
        progress_meta = {"task_terminal_status": "cancelled" if stopped else "failed"}
        if "origin_message_ref" in correlation:
            progress_meta["origin_message_ref"] = correlation["origin_message_ref"]
        initiator = str((task_metadata or {}).get("initiator") or "")
        if initiator:
            progress_meta["initiator"] = initiator
        _pool().send_with_budget(
            chat_id,
            err_msg,
            task_id=failed_task_id,
            progress_meta=progress_meta,
            role="system", system_type="task_not_started" if stopped or exhausted else "task_error")
    except Exception:
        log.debug("Suppressed exception", exc_info=True)


# ---------------------------------------------------------------------------
# resume after a planned restart
# ---------------------------------------------------------------------------

def conversation_turn_live() -> bool:
    """Whether any owner conversation is live (``live_turn_count``): a correspondent's Presence turn is not one."""
    from supervisor import queue
    from supervisor.conversation_lane import live_turn_count

    with queue._queue_lock:
        return live_turn_count() > 0


def auto_resume_after_restart() -> None:
    """Consume boot doors only after their work stops and controls are durable."""
    try:
        from ouroboros.delegate_recovery import import_legacy_restart, pending_owner_restart, consume_owner_restart
        from ouroboros.installation_stop import (PANIC_RECEIPT, panic_input, panic_received,
                                                remove_received_panic)
        from supervisor.queue import publish_queue_snapshot
        root = _pool().DRIVE_ROOT
        import_legacy_restart(root)
        restart = pending_owner_restart(root)
        if restart:
            if publish_queue_snapshot(reason="owner_restart_holds") is not True:  # barrier (LB)
                log.error("Owner restart holds are not durable; its door is kept.")
                return
            from supervisor.queue_schedules import schedule_transaction
            from supervisor.followup_policy import record_restart
            with schedule_transaction(root):
                record_restart(root)
                consume_owner_restart(root, restart)
        identity = panic_input(root)
        if identity:
            if panic_received(root, identity):
                remove_received_panic(root, identity)
                return
            if publish_queue_snapshot(reason="panic_holds") is not True:  # barrier (LB)
                log.error("Panic holds are not durable; its input is kept.")
                return
            from ouroboros.server_control import PANIC_CONTROL_KEYS, _panic_controls
            from supervisor.state import update_state

            update_state(_panic_controls, confirm=PANIC_CONTROL_KEYS)
            update_state(lambda state: state.update({PANIC_RECEIPT: identity}))
            remove_received_panic(root, identity)
            return
        if restart:
            return

        from supervisor.state import control_value

        chat_known, chat_id = control_value(_pool().load_state(), "owner_chat_id")
        if not chat_known or chat_id in (None, "", 0):
            return  # an autonomous resume turn needs a KNOWN owner chat (#1307)
        from ouroboros.server_process import core_loss_recovery_turn

        if core_loss_recovery_turn(root, int(chat_id), _pool().PENDING, admit=handle_chat_direct,
                                   live=conversation_turn_live, gate=owner_conversation_gate_open):
            return  # a Fault boot: its one recovery turn, or the previous one still open, is the resume (owner 2A)

        # The one real trigger: a planned restart's verify record. A launcher
        # line in the log tail is not evidence of unfinished work (owner Batch4).
        restart_verify_path = _pool().DRIVE_ROOT / "state" / "pending_restart_verify.json"
        if not restart_verify_path.exists():
            return

        if not conversation_turn_live():
            from ouroboros.places import paused_places_sentence

            # The admission is short (the body runs in a carrier): no thread of its own.
            meta = handle_chat_direct(
                int(chat_id),
                "\n".join(filter(None, (
                    "[auto-resume after restart] Continue your work. Read scratchpad and identity — they contain context of what you were doing.",
                    paused_places_sentence(_pool().DRIVE_ROOT)))),
                None,
            )
            append_jsonl(
                _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {
                    "ts": utc_now_iso(),
                    "type": "auto_resume_triggered",
                    "trigger": "planned_restart",
                    "task_id": str(((meta or {}).get("task") or {}).get("id") or ""),
                },
            )
    except Exception as e:
        append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(),
            "type": "auto_resume_error",
            "error": repr(e),
        })


# The inline executors' Stop (``inline_carrier``) stays importable here for its custody callers.
from supervisor.inline_carrier import (  # noqa: E402, F401 -- intentional public re-exports
    DIRECT_TURN_STOP_ENDED, DIRECT_TURN_STOP_GONE, DIRECT_TURN_STOP_LIVE, stop_direct_chat_turn,
)
