"""A reviewed message tool's one local return address. The extension never supplies a room or authority.

An opaque random reference names a short host record for a bound call. The named inbound dialogue row owns
result deduplication and dispatch custody. Nothing wakes at boot just because a call exists; submit and intake
both check the source's room and controls. An indeterminate origin is not Main.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

VERSION = 1
_REF = re.compile(r"[0-9a-f]{32}")
_ID = re.compile(r"late-result-[0-9a-f]{64}")
_LETTER_ID = re.compile(r"[A-Za-z0-9_.:@+\-]{1,256}")
_CALL_REF: ContextVar[list | None] = ContextVar("extension_call_ref", default=None)
_SHARED_EXECUTION = "late_result_execution"


class LateResultUnavailable(ValueError):
    """The source cannot grant automatic attention; the skill keeps its letter."""


class LateResultWaiting(RuntimeError):
    """Retry only the same report: the origin currently does not admit it."""


def restricted_return(task: Any) -> bool:
    """A host-admitted local consideration is not an ordinary owner conversation or work episode."""
    author = task.get("admission_author") if isinstance(task, dict) else None
    return isinstance(author, dict) and bool(author.get("late_report_ref"))


def call_ref(ctx: Any) -> str:
    bound = _CALL_REF.get()
    return bound[1] if bound is not None and bound[0] is ctx else str(getattr(ctx, "_child_call_ref", ""))


@contextmanager
def with_call_ref(ctx: Any):
    token = _CALL_REF.set([ctx, ""])
    try:
        yield
    finally:
        _CALL_REF.reset(token)


def bind_call_ref(ctx: Any, name: str, descriptor: dict, args: dict) -> None:
    """Only the bound invocation's context, including its async handoff, can see this value."""
    bound = _CALL_REF.get()
    if bound is not None and bound[0] is ctx and eligible(name, descriptor, args):
        bound[1] = mint(ctx, descriptor, await_tool_result=True)


def eligible(name: str, descriptor: dict, args: dict) -> bool:
    """A generation-bound reviewed connector declares this exact sending tool (not a read or a gesture)."""
    from ouroboros.tool_effects import EFFECT_MESSAGE, bound_declaration, effect_of

    declaration = bound_declaration(descriptor)
    if not (declaration and declaration.get("provides_place") and declaration["tool"].get("late_result") is True):
        return False
    effect = effect_of(name, ext_tool=descriptor, args=args)
    return effect.declared and effect.effect == EFFECT_MESSAGE and not effect.problem


def _panic_identity(root: pathlib.Path) -> dict:
    from ouroboros.installation_stop import PANIC_RECEIPT, installation_panic

    if installation_panic(root):
        raise LateResultUnavailable("installation stopped")
    with (root / "state" / "state.json").open(encoding="utf-8") as stream:
        state = json.load(stream)
    if not isinstance(state, dict):
        raise LateResultWaiting("installation controls unavailable")
    if state.get("bg_consciousness_stop_source") == "panic":
        raise LateResultUnavailable("installation stopped")
    receipt = state.get(PANIC_RECEIPT, {})
    if not isinstance(receipt, dict):
        raise LateResultWaiting("installation controls unavailable")
    return receipt


def _read_work(root: pathlib.Path, tid: str) -> dict:
    from ouroboros.task_results import load_task_result

    row = load_task_result(root, tid, strict=True)
    if not row or row.get("task_id") != tid:
        raise LateResultWaiting("source work unavailable")
    return row


def _address(root: pathlib.Path, root_id: str, work: dict) -> tuple[int, str, str]:
    """A closed or unknown room, and a Place without a local-only conversation route, grant no wake."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
    from ouroboros.dialogue_provenance import is_presence_task, run_origin
    from ouroboros.project_admission import reserved_project_for_chat
    from ouroboros.projects_registry import PROJECT_ACTIVE, project_binding_for_task

    meta = work.get("metadata") or {}
    if is_presence_task(work) or run_origin(work).get("presence") or meta.get("execution_transport") is not None:
        raise LateResultUnavailable("this Place has no local return route; read the letter in the skill")
    binding = project_binding_for_task(root, root_id, strict=True) or {}
    pid = str(binding.get("project_id") or work.get("project_id") or "")
    chat = binding.get("project_chat_id") if binding else work.get("chat_id")
    if type(chat) is not int or chat <= 0:
        raise LateResultUnavailable("source room unknown")
    incarnation = ""
    if pid:
        room = reserved_project_for_chat(root, chat)
        if room.get("id") != pid or room.get("lifecycle") != PROJECT_ACTIVE or "closed" in room:
            raise LateResultUnavailable("source room is not open")
        incarnation = str(room.get("routing_incarnation") or room.get("created_at") or "")
        if not incarnation:
            raise LateResultUnavailable("source room identity unavailable")
    elif chat != WEB_UI_CHAT_ID:
        raise LateResultUnavailable("source room unknown")
    return chat, pid, incarnation


def _report_id(ref: str) -> str:
    if not isinstance(ref, str) or not _REF.fullmatch(ref):
        raise LateResultUnavailable("invalid call reference")
    return "late-result-" + hashlib.sha256(ref.encode("ascii")).hexdigest()


def _binding_path(root: pathlib.Path, report_id: str) -> pathlib.Path:
    if not _ID.fullmatch(report_id):
        raise LateResultUnavailable("invalid report reference")
    return root / "state" / "extension_late_results" / f"{report_id}.json"


def binding_for_report(root: pathlib.Path, report_id: str) -> dict:
    try:
        with _binding_path(root, report_id).open(encoding="utf-8") as stream:
            binding = json.load(stream)
    except FileNotFoundError as exc:
        raise LateResultUnavailable("call not recorded") from exc
    if not isinstance(binding, dict) or binding.get("version") != VERSION:
        raise LateResultUnavailable("call not recorded")
    return binding


def record_handoff(root: pathlib.Path, ref: str, facts: dict, outcome: str) -> dict:
    """Retain a decision before attempting to project it into a possibly busy dialogue store."""
    from ouroboros.utils import update_json_locked

    record = {key: facts[key] for key in ("claim", "process", "dispatcher")}
    record["outcome"] = outcome

    def decide(binding: dict) -> dict | None:
        prior = binding.get("handoff") or {}
        if prior.get("claim") == record["claim"]:
            if prior != record:
                raise LateResultUnavailable("conflicting local handoff outcome")
            return None
        if prior and prior.get("outcome") != "defer":
            raise LateResultUnavailable("local handoff already decided")
        return {**binding, "handoff": record}

    update_json_locked(_binding_path(root, ref), decide, strict_existing_dict=True)
    return record


def project_handoff(root: pathlib.Path, chat_id: int, ref: str, decision: dict) -> None:
    """Retry the recorded write, never the handoff. A stale deferral cannot release a newer claim."""
    from ouroboros.dialogue_store import complete_dispatch, defer_dispatch
    from ouroboros.store import acquisition_budget

    with acquisition_budget(0):
        if decision["outcome"] == "defer":
            defer_dispatch(root, chat_id, ref, dispatcher=decision["dispatcher"], process=decision["process"],
                           claim=decision["claim"])
        else:
            complete_dispatch(root, chat_id, ref, delivery=decision["outcome"], claim=decision["claim"])


def check_address(root: pathlib.Path, binding: dict, work: dict | None = None) -> None:
    """A letter may be retained only at its still-confirmed local address, without reopening it."""
    from ouroboros.server_control import panic_termination_requested

    if panic_termination_requested() or _panic_identity(root) != binding.get("panic"):
        raise LateResultUnavailable("installation stopped")
    root_id = binding["root_task_id"]
    if _address(root, root_id, work if work is not None else _read_work(root, root_id)) != (
            binding["chat_id"], binding["project_id"], binding["incarnation"]):
        raise LateResultUnavailable("source room changed")


def _source_rows(root: pathlib.Path, binding: dict) -> dict[str, dict]:
    """Read only the addressed lineage, root first. Every enclosing work keeps its own limits."""
    tid, root_id = binding["task_id"], binding["root_task_id"]
    seen = {root_id: _read_work(root, root_id)}
    while tid != root_id:
        if not tid or tid in seen:
            raise LateResultUnavailable("source lineage unavailable")
        row = seen[tid] = _read_work(root, tid)
        if row.get("root_task_id") != root_id:
            raise LateResultUnavailable("source work changed")
        tid = str(row.get("parent_task_id") or (row.get("metadata") or {}).get("parent_task_id") or "")
    return seen


def current_source(root: pathlib.Path, binding: dict) -> tuple[str, dict[str, dict]]:
    """The live originating root or ordinary completion; holds never authorize a new turn."""
    from ouroboros.budget_pause import budget_pause_row, LIVE_PAUSE_STATES
    from ouroboros.cancel_intents import cancel_pending
    from ouroboros.dialogue_provenance import is_presence_task
    from ouroboros.owner_pause import fence_closed, owner_stopped, read_fence
    from ouroboros.outcomes import EXECUTION_OK
    from ouroboros.review_records import recorded_stop_rationale
    from ouroboros.task_finalization import TERMINAL_ORIGIN_MODEL_FINAL
    from ouroboros.work_stop import current_stop

    root_id = binding["root_task_id"]
    rows = _source_rows(root, binding)
    check_address(root, binding, rows[root_id])
    _deadline(binding)
    if fence_closed(read_fence(root, root_id)):
        raise LateResultWaiting("source is held")
    for tid, row in rows.items():
        status = row.get("status")
        meta = row.get("metadata") or {}
        if (is_presence_task(row) or meta.get("execution_transport") is not None or owner_stopped(row)
                or row.get("followup_stop") or row.get("cancel_origin")
                or status in {"cancelled", "cancel_requested", "failed", "interrupted"}
                or cancel_pending(root, tid, strict=True, propagate_busy=True)):
            raise LateResultUnavailable("source no longer admits consideration")
        if (current_stop(row) or budget_pause_row(root, tid).get("state") in LIVE_PAUSE_STATES
                or status not in {"running", "completed"}):
            raise LateResultWaiting("source is held")
        _deadline(row)
        if status == "completed" and (row.get("terminal_origin") != TERMINAL_ORIGIN_MODEL_FINAL
                or (row.get("outcome_axes") or {}).get("execution", {}).get("status") != EXECUTION_OK
                or recorded_stop_rationale(row) is not None):
            raise LateResultUnavailable("source did not finish normally")
    if rows[root_id].get("status") == "running":
        return root_id, rows
    if any(row.get("status") != "completed" for row in rows.values()):
        raise LateResultWaiting("source lineage is still open")
    return "", rows


def _deadline(row: dict) -> str:
    from ouroboros.deadline_utils import parse_deadline_ts, utc_now

    selected = []
    metadata = row.get("metadata") or {}
    for source in (row, metadata, row.get("task_contract") or {}, metadata.get("task_contract") or {}):
        if raw := source.get("deadline_at"):
            deadline = parse_deadline_ts(str(raw))
            if deadline is None or deadline <= utc_now():
                raise LateResultUnavailable("source deadline unavailable or expired")
            selected.append(deadline)
    return min(selected).isoformat() if selected else ""


def current_target(root: pathlib.Path, binding: dict) -> str:
    return current_source(root, binding)[0]


def _route_processing(source: Any) -> str | None:
    """The settled call's canonical host processing preference, or None when unknown."""
    from ouroboros.model_slots import CALL_TIERS

    if not isinstance(source, dict) or "processing_preference" not in source:
        return None
    pref = source["processing_preference"]
    return pref if isinstance(pref, str) and pref in ("", *CALL_TIERS) else None


def return_turn_metadata(root: pathlib.Path, binding: dict, rows: dict[str, dict], *, settling=()) -> dict:
    """Use the originating work's verified wallet and deadline, without reopening that work."""
    from ouroboros.owner_continue import recorded_execution_seconds
    from ouroboros.runtime_limits import resolve_task_execution_timeout_sec
    from supervisor.followup_policy import resolve_relation

    try:
        relation = resolve_relation(root, binding, "related", declared_by="late_result")
    except ValueError as exc:
        raise LateResultUnavailable("source allowance not established") from exc
    route = binding.get("route")
    if not route or _route_processing(route) is None:
        raise LateResultUnavailable("source route not established")
    policies = _source_policies(rows)
    if not all(policy["supported"] for policy in policies.values()) or policies != binding.get("source_policies"):
        raise LateResultUnavailable("source restrictions cannot be carried to a local consideration turn")
    deadlines = [deadline] if (deadline := _deadline(binding)) else []
    windows = {}
    for tid, row in rows.items():
        if deadline := _deadline(row):
            deadlines.append(deadline)
        effective = resolve_task_execution_timeout_sec({**row, "_execution_owner": row.get("execution_owner")})
        limits = [x for x in (effective, (binding.get("execution_limits") or {}).get(tid)) if x is not None]
        if limits:
            used = recorded_execution_seconds(row)
            if used is None and tid in settling:
                raise LateResultWaiting("source execution interval is being closed")
            if used is None or (left := min(limits) - used) <= 0:
                raise LateResultUnavailable("source execution time unavailable or exhausted")
            windows[tid] = left
    return {"billing_group": relation["billing_group"],
            **({"deadline_at": min(deadlines)} if deadlines else {}),
            **({"_late_result_time_scopes": windows} if windows else {}),
            "_late_result_route": route, "_late_result_can_finish": all(policy["finish"] for policy in policies.values()),
            "_late_result_constraints": "\n".join(f"Source work {tid}: {policy['constraints']}"
                for tid, policy in policies.items() if policy["constraints"]),
            "_late_result_origin": binding["task_id"]}


def _source_policies(rows: dict[str, dict]) -> dict:
    """Only the source restrictions that can affect a local, finish-only, no-network turn."""
    from ouroboros.consciousness_authority import task_disabled_tools
    from ouroboros.task_pacing import recorded_producer_ceiling

    result = {}
    for tid, row in rows.items():
        meta = row.get("metadata") or {}
        contract = row.get("task_contract") or meta.get("task_contract")
        if not isinstance(contract, dict):
            result[tid] = {"supported": False}
            continue
        unsupported = (contract.get("capability_ceiling") or contract.get("input_sources") == "declared"
                       or any(source.get("answer_protocol") or source.get("review_answer") for source in (row, meta, contract))
                       or recorded_producer_ceiling(row) or meta.get("root_cost_ceiling_usd") is not None
                       or (contract.get("budget_profile") or {}).get("cost_hard_stop_pct") is not None
                       or ((row.get("configured_subagent") or {}).get("route") or {}).get("kind") == "agent_session")
        result[tid] = {"supported": not unsupported, "finish": "finish_task" not in task_disabled_tools(row),
                       "constraints": contract.get("constraints") or ""}
    return result


def _seconds(value: Any, *, positive: bool = False) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def debit_return_execution(source: dict, scopes: dict, prior_turn: dict | None = None,
                           *, settling=()) -> tuple[dict, float]:
    """Pure remaining-window debit from source custody; never grants credit on a missing closing clock."""
    from ouroboros.owner_continue import recorded_execution_seconds
    from ouroboros.task_status import SETTLED_STATUSES

    prior = source.get(_SHARED_EXECUTION, {})
    spent = prior.get("spent", {}) if isinstance(prior, dict) else None
    if not isinstance(spent, dict) or not all(_seconds(value) for value in spent.values()):
        raise LateResultUnavailable("source time custody is unreadable")
    spent = dict(spent)
    if active := prior.get("active"):
        turn_id, ref, active_scopes = (
            (active.get("task_id"), active.get("report_ref"), active.get("scopes")) if isinstance(active, dict) else (None, None, None))
        if not (isinstance(turn_id, str) and turn_id and isinstance(ref, str) and ref
                and isinstance(active_scopes, list) and all(isinstance(tid, str) and tid for tid in active_scopes)):
            raise LateResultUnavailable("source time custody is unreadable")
        if turn_id in settling:
            raise LateResultWaiting("another return is using the original execution window")
        if not isinstance(prior_turn, dict) or prior_turn.get("task_id") != turn_id:
            raise LateResultWaiting("source work unavailable")
        if prior_turn.get("status") not in SETTLED_STATUSES:
            raise LateResultWaiting("another return still owns the original execution window")
        if ((prior_turn.get("admission_author") or {}).get("late_report_ref") != ref
                or not _seconds(used := recorded_execution_seconds(prior_turn))):
            raise LateResultUnavailable("the preceding return has no verified closing clock")
        for tid in active_scopes:
            spent[tid] = spent.get(tid, 0) + used
    if not scopes or not all(_seconds(seconds, positive=True) for seconds in scopes.values()):
        raise LateResultUnavailable("source execution time unavailable or exhausted")
    left = min(seconds - spent.get(tid, 0) for tid, seconds in scopes.items())
    if not _seconds(left, positive=True):
        raise LateResultUnavailable("the original execution window is exhausted")
    return spent, left


def _remaining_return_execution(root: pathlib.Path, source: dict, scopes: dict, *, settling=()) -> tuple[dict, float]:
    prior = source.get(_SHARED_EXECUTION, {})
    active = prior.get("active") if isinstance(prior, dict) else None
    tid = active.get("task_id") if isinstance(active, dict) else None
    turn = _read_work(root, tid) if isinstance(tid, str) and tid and tid not in settling else None
    return debit_return_execution(source, scopes, turn, settling=settling)


def preview_return_execution(root: pathlib.Path, source: dict, scopes: dict, *, settling=()) -> float | None:
    """Read-only check of the remaining shared execution window before a durable RUNNING row exists."""
    if not scopes:
        return None
    if isinstance(settling, dict):
        for entry in settling.values():
            live = entry.get("task") if isinstance(entry, dict) else None
            meta = live.get("metadata") if restricted_return(live) else None
            live_scopes = meta.get("_late_result_time_scopes") if isinstance(meta, dict) else None
            if isinstance(live_scopes, dict) and (scopes.keys() & live_scopes.keys()):
                raise LateResultWaiting("another return is already admitted for the original execution window")
    return _remaining_return_execution(root, source, scopes, settling=settling)[1]


def reserve_return_execution(root: pathlib.Path, source: dict, scopes: dict, task_id: str, report_ref: str,
                             *, settling=(), expected_seconds: Any = None) -> float | None:
    """Claim one finite window on the source row for an already-durable RUNNING return turn."""
    if not scopes:
        if expected_seconds is not None:
            raise LateResultUnavailable("source execution window changed")
        return None
    from ouroboros.task_results import STATUS_RUNNING, require_writable_task_result_schema, stamp_task_result_schema
    from ouroboros.task_store import patch_document

    prior = source.get(_SHARED_EXECUTION, {})
    spent, left = _remaining_return_execution(root, source, scopes, settling=settling)
    if expected_seconds is not None and (not _seconds(expected_seconds, positive=True) or left != expected_seconds):
        raise LateResultUnavailable("source execution window changed before claim")
    claim = {"task_id": task_id, "report_ref": report_ref, "scopes": sorted(scopes), "seconds": left}

    def claim_on_source(current: dict) -> dict:
        if current.get("status") != "completed" or current.get(_SHARED_EXECUTION, {}) != prior:
            raise LateResultWaiting("source execution custody changed before admission")
        require_writable_task_result_schema(current)
        return stamp_task_result_schema({**current, _SHARED_EXECUTION: {"spent": spent, "active": claim}})

    def require_durable_turn(txn: Any, _current: dict, document: dict) -> dict:
        row = txn.load(str(task_id))
        row_bound = (row.get("metadata") or {}).get("execution_timeout_sec") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or row.get("task_id") != str(task_id) or row.get("status") != STATUS_RUNNING
                or (row.get("admission_author") or {}).get("late_report_ref") != report_ref
                or (expected_seconds is not None and (not _seconds(row_bound, positive=True) or row_bound != expected_seconds))):
            raise LateResultWaiting("return turn must be durably running before reserving execution time")
        return document

    patch_document(root, source["task_id"], claim_on_source, last_word=require_durable_turn)
    return left


def return_time_owned(source: dict, scopes: dict, task_id: str, report_ref: str, seconds: Any) -> bool:
    if not scopes:
        return seconds is None
    active = (source.get(_SHARED_EXECUTION) or {}).get("active")
    return _seconds(seconds, positive=True) and active == {
        "task_id": task_id, "report_ref": report_ref, "scopes": sorted(scopes), "seconds": seconds}


def _return_source_scopes(root: pathlib.Path, task: dict, report_ref: str, *, settling=()) -> tuple[dict, dict, dict]:
    binding = binding_for_report(root, report_ref)
    if task.get("chat_id") != binding["chat_id"] or str(task.get("project_id") or "") != binding["project_id"]:
        raise LateResultUnavailable("the original room changed")
    target, rows = current_source(root, binding)
    if target:
        raise LateResultWaiting("the original reader is still live")
    current = return_turn_metadata(root, binding, rows, settling=settling)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    if any(current.get(key) != metadata.get(key) for key in current):
        raise LateResultUnavailable("the original allowance changed")
    return binding, rows[binding["root_task_id"]], current.get("_late_result_time_scopes") or {}


def claim_running_return_execution(drive_root: Any, task: dict, *, persisted: bool) -> dict | None:
    """Bind shared execution custody to this turn's durable RUNNING row without changing its queued window."""
    if not restricted_return(task):
        return None
    refusal = {"reason_code": "late_result_source_unavailable",
               "human_label": "this late-result turn", "detail": "The letter was kept without starting a turn."}
    report_ref = str((task.get("admission_author") or {}).get("late_report_ref") or "")
    task_id = str(task.get("id") or "")
    root = pathlib.Path(str(task.get("budget_drive_root") or "") or drive_root)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    bound, meta_bound = task.get("execution_timeout_sec"), metadata.get("execution_timeout_sec")
    if (not persisted or not report_ref or not task_id or root.resolve() != pathlib.Path(drive_root).resolve()
            or isinstance(bound, bool) or isinstance(meta_bound, bool) or bound != meta_bound):
        return refusal
    from ouroboros.store import StoreError

    try:
        _binding, source, scopes = _return_source_scopes(root, task, report_ref)
        if not return_time_owned(source, scopes, task_id, report_ref, bound):
            if not scopes or not _seconds(bound, positive=True):
                return refusal
            reserve_return_execution(root, source, scopes, task_id, report_ref, expected_seconds=bound)
        return None
    except (OSError, ValueError, RuntimeError, StoreError):
        return refusal


def _captured_route(ctx: Any) -> dict:
    """The call whose tool invocation owns this address. A later turn never guesses a replacement."""
    call = getattr(ctx, "_current_llm_call_meta", None) or {}
    model, provider = call.get("model"), call.get("physical_provider")
    physical_model, account = call.get("physical_model"), call.get("dispatch_account")
    local, pref = call.get("use_local"), _route_processing(call)
    if not (all(isinstance(value, str) and value for value in (model, provider, physical_model, account))
            and type(local) is bool and pref is not None):
        return {}
    pin = ""
    observed_route = {}
    if provider == "claudexor":
        from ouroboros.provider_models import parse_claudexor_model

        observed = call.get("observed_route") or {}
        source, native = parse_claudexor_model(physical_model)
        pin = observed.get("credentialProfileId")
        if not isinstance(pin, str) or not pin or (account.startswith("pin:") and account != f"pin:{pin}"):
            return {}
        if (observed.get("source"), observed.get("model")) != (source, native):
            return {}
        observed_route = {"source": source, "model": native, "credentialProfileId": pin}
        account = f"pin:{pin}"  # the observed serving account, not an Auto preference on the next turn
    return {"model": model, "provider": provider, "physical_model": physical_model,
            "dispatch_account": account, "use_local": local, "credential_profile_id": pin,
            "processing_preference": pref,
            **({"observed_route": observed_route} if observed_route else {})}


def local_finish_call(ctx: Any, call: dict) -> bool:
    """Only the existing completion decision, never a tool that could act on the letter's behalf."""
    if not getattr(ctx, "_late_report_ref", ""):
        return True
    fn = call.get("function") or {}
    if not ctx.task_metadata.get("_late_result_can_finish") or fn.get("name") != "finish_task":
        return False
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except (TypeError, ValueError):
        return False
    return (isinstance(args, dict) and args.get("action") == "finish"
            and args.keys() <= {"action", "answer", "silent"})


def mint(ctx: Any, descriptor: dict, *, await_tool_result: bool = False) -> str:
    """Mint only after ordinary dispatch admission. Without an exact room, or the serving route its call proved, the
    send stays usable without automatic return: no reference, and the letter stays readable by its id."""
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.utils import atomic_write_json
    from ouroboros.store import StoreError

    if getattr(ctx, "task_lifecycle_bound", False) is not True or not getattr(ctx, "task_id", None):
        return ""
    root = canonical_data_root(ctx)
    tid = str(ctx.task_id)
    skill, digest = str(descriptor.get("skill") or ""), str(descriptor.get("content_hash") or "")
    if not skill or not digest:
        return ""
    try:
        if not (route := _captured_route(ctx)):
            return ""  # no later request could carry this letter, so none is promised
        source = _read_work(root, tid)
        if source.get("status") != "running":
            return ""
        root_id = str(source.get("root_task_id") or tid)
        work = source if root_id == tid else _read_work(root, root_id)
        chat_id, project_id, incarnation = _address(root, root_id, work)
        binding = {"version": VERSION, "task_id": tid, "root_task_id": root_id, "chat_id": chat_id,
                   "project_id": project_id, "incarnation": incarnation, "skill": skill,
                   "content_hash": digest, "panic": _panic_identity(root),
                   **({"inline": {"state": "pending"}} if await_tool_result else {})}
        target, rows = current_source(root, binding)
        if not target:
            return ""
        from ouroboros.runtime_limits import resolve_task_execution_timeout_sec

        binding["route"] = route
        binding["source_policies"] = _source_policies(rows)
        binding["deadline_at"] = min((deadline for row in rows.values() if (deadline := _deadline(row))), default="")
        binding["execution_limits"] = {
            key: resolve_task_execution_timeout_sec({**row, "_execution_owner": row.get("execution_owner")})
            for key, row in rows.items()}
        ref = uuid.uuid4().hex
        atomic_write_json(_binding_path(root, _report_id(ref)), binding)
        return ref
    except (OSError, ValueError, StoreError, LateResultWaiting):
        return ""


def report_input(body: Any) -> tuple[str, str, str, dict, str]:
    from ouroboros.dialogue_provenance import recorded_author
    from ouroboros.owner_mailbox import TASK_MESSAGE_MAX_CHARS

    if not isinstance(body, dict) or set(body) != {"call_ref", "report_id", "label", "text"}:
        raise ValueError("expected call_ref, report_id, label and text")
    ref, message_id, label, text = (body[key] for key in ("call_ref", "report_id", "label", "text"))
    if not isinstance(message_id, str) or not _LETTER_ID.fullmatch(message_id):
        raise ValueError("report_id must be a readable letter id")
    if not isinstance(text, str) or len(text) > TASK_MESSAGE_MAX_CHARS or not text.strip():
        raise ValueError("invalid result text length")
    if not isinstance(label, str) or len(label) > 240:
        raise ValueError("invalid label length")
    author = recorded_author(label)
    fingerprint = hashlib.sha256(json.dumps([message_id, label, text], ensure_ascii=True).encode("ascii")).hexdigest()
    return _report_id(ref), fingerprint, text, author, message_id


def inline_result_fact(ref: str, body: dict, producer_text: str) -> dict:
    """Only the dispatcher that bound the call can stamp a candidate receipt, never the producer alone."""
    if not ref or "late_answer" not in body:
        return {}
    marker = body["late_answer"]
    if marker is None:
        fact = {"state": "none"}
    else:
        if not isinstance(marker, dict) or set(marker) != {"report_id", "label", "text"}:
            return {}
        try:
            _, digest, _, _, _ = report_input({**marker, "call_ref": ref})
        except (TypeError, ValueError):
            return {}
        fact = {"state": "tool_result", "digest": digest}
    return {"ref": _report_id(ref), "producer_sha256": hashlib.sha256(producer_text.encode("utf-8")).hexdigest(),
            **fact}


def acknowledge_inline_result(ctx: Any, result: Any, view: dict) -> None:
    """A completed no-answer receipt is a host fact; a reply needs the full saved transcript."""
    fact = (getattr(result, "meta", None) or {}).get("inline_late_result") or {}
    if ctx is None or not fact:
        return
    from ouroboros.tool_access_paths import canonical_data_root
    from ouroboros.working_checkpoint import defer_content_ack
    from ouroboros.utils import update_json_locked

    producer = result.text if result.producer_text is None else result.producer_text
    if not producer or hashlib.sha256(producer.encode("utf-8")).hexdigest() != fact["producer_sha256"]:
        return
    root, tid = canonical_data_root(ctx), str(ctx.task_id)

    def update(record: dict) -> dict | None:
        if record.get("task_id") != tid or record.get("inline") != {"state": "pending"}:
            return None
        return {**record, "inline": {key: fact[key] for key in ("state", "digest") if key in fact}}

    def save() -> None:
        update_json_locked(_binding_path(root, fact["ref"]), update, strict_existing_dict=True)

    if fact["state"] == "none" or (not view.get("result_partial") and producer in str(view.get("result") or "")):
        defer_content_ack(ctx, save)


def inline_delivery(root: pathlib.Path, binding: dict, fingerprint: str) -> str:
    """An observed ordinary tool result precludes a second delivery of those exact data."""
    inline = binding.get("inline") or {}
    state = inline.get("state")
    if state == "tool_result":
        if inline.get("digest") != fingerprint:
            raise LateResultUnavailable("reported result differs from the retained tool result")
        return state
    if state == "pending":
        if _read_work(root, binding["task_id"]).get("status") == "running":
            raise LateResultWaiting("original tool result has not been accounted for")
        raise LateResultUnavailable("original tool result could not be confirmed")
    if inline and state != "none":
        raise LateResultUnavailable("original tool result could not be confirmed")
    return ""


def return_candidate(ctx: Any, prior: Any) -> Any:
    """A changed physical route or closed source never gains a send from a machine letter."""
    tool_ctx = ctx.tools._ctx
    route = getattr(tool_ctx, "_late_result_route", None)
    if not getattr(tool_ctx, "_late_report_ref", ""):
        return prior
    from ouroboros.tool_access_paths import canonical_data_root

    root = canonical_data_root(tool_ctx)
    report_ref = tool_ctx._late_report_ref
    metadata = tool_ctx.task_metadata

    def allowed(request: Any) -> bool:
        processing = _route_processing(route)
        if (not route or processing is None
                or request.provider != route.get("provider") or request.model != route.get("physical_model")
                or getattr(request.physical_context, "dispatch_account", "") != route.get("dispatch_account")
                or getattr(request, "processing_preference", None) != processing
                or ctx.active_use_local != route.get("use_local")):
            return False
        try:
            binding = binding_for_report(root, report_ref)
            target, rows = current_source(root, binding)
            if target:
                return False
            current = return_turn_metadata(root, binding, rows)
            if not return_time_owned(rows[binding["root_task_id"]], current.get("_late_result_time_scopes") or {},
                                     str(tool_ctx.task_id), report_ref, metadata.get("execution_timeout_sec")):
                return False
            if any(metadata.get(key) != value for key, value in current.items()):
                return False  # a changed source contract, deadline or billing group never authorizes a stale candidate
        except (OSError, ValueError, RuntimeError):
            return False
        return prior is None or prior(request) is not False

    return allowed


def return_account(ctx: Any, default: Any) -> Any:
    return (getattr(ctx.tools._ctx, "_late_result_route", None) or {}).get("credential_profile_id", default)


def return_processing_preference(ctx: Any, default: Any = None) -> Any:
    tool_ctx = ctx.tools._ctx
    if not getattr(tool_ctx, "_late_report_ref", ""):
        return default
    if (processing := _route_processing(getattr(tool_ctx, "_late_result_route", None))) is None:
        raise LateResultUnavailable("source processing preference not established")
    return processing


def stage_live_letter(ctx: Any, entry: dict) -> bool:
    """Keep conditional input outside all ordinary transcript, checkpoint and finalization paths."""
    note = entry["agent_note"]
    ref = note.get("late_report_ref") or ""
    if ctx is None or not _ID.fullmatch(ref) or note.get("late_reader_attempt") != (getattr(ctx, "task_attempt", None) or 1):
        return False
    from ouroboros.tool_access_paths import canonical_data_root

    try:
        binding = binding_for_report(canonical_data_root(ctx), ref)
    except (OSError, ValueError, RuntimeError):
        return False
    if not binding.get("route") or binding.get("send_claim") or binding.get("root_task_id") != str(ctx.task_id):
        return False
    if getattr(ctx, "_late_result_mail", None) is None:
        ctx._late_result_mail = {}
    ctx._late_result_mail[ref] = entry
    return True


_SERVING = ("model", "provider", "physical_model", "dispatch_account", "use_local", "processing_preference")


def owed_letter(tool: Any) -> bool:
    """Whether a finish gives a staged letter one more round: its reader's current serving route, account and
    processing preference included (``""`` is the Default, none recorded is unknown), is the one its call captured,
    and no finish held for it before. Another route, a closed or uncertain source, or one already given that round
    holds nothing."""
    from ouroboros.store import acquisition_budget
    from ouroboros.tool_access_paths import canonical_data_root

    mail = getattr(tool, "_late_result_mail", None)
    try:
        current = _captured_route(tool) if mail else {}
        root = canonical_data_root(tool) if current else None
    except (OSError, ValueError, RuntimeError):
        return False
    if not current:
        return False
    held = getattr(tool, "_late_result_held", None)
    if held is None:
        held = tool._late_result_held = set()
    for ref in [ref for ref in mail if ref not in held]:
        try:
            binding = binding_for_report(root, ref)
            route = binding.get("route") or {}
            if binding.get("send_claim") or any(key not in route or route[key] != current.get(key, ()) for key in _SERVING):
                continue
            with acquisition_budget(0):
                if current_target(root, binding) != str(tool.task_id):
                    continue
        except (OSError, ValueError, RuntimeError):
            continue
        held.add(ref)
        return True
    return False


def take_letter_send(ctx: Any, *, role: str, account: str | None):
    """Offer at most one addressed letter to an ordinary round's optional physical copy. It stays staged until that
    copy's own dispatch is accounted for (``LetterSend.settle``/``observe``); one that can no longer reach this reader
    leaves without holding its ordinary work."""
    from ouroboros.model_slots import send_processing_preference
    from ouroboros.model_wait import current_model_wait
    from ouroboros.store import acquisition_budget
    from ouroboros.tool_access_paths import canonical_data_root

    tool = ctx.tools._ctx
    waiter = current_model_wait()
    if ref := getattr(tool, "_late_report_ref", ""):
        return LetterSend(ctx, canonical_data_root(tool), ref, getattr(tool, "_late_result_route", None) or {})
    mail = getattr(tool, "_late_result_mail", None) or {}
    for ref, entry in list(mail.items()):
        root = canonical_data_root(tool)
        try:
            binding = binding_for_report(root, ref)
            route = binding.get("route") or {}
            if binding.get("send_claim"):
                raise LateResultUnavailable("letter already owns a send")
            if (route.get("model"), route.get("use_local")) != (ctx.active_model, ctx.active_use_local):
                continue
            pin = route.get("credential_profile_id")
            if pin:
                from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option

                current = account if account is not None else model_role_option(MODEL_ACCOUNTS_KEY, role)
                if current and current != pin:
                    continue
            # The preference this round sends (as ``_dispatch_round_model`` computes it) must be the captured one.
            if route.get("processing_preference", ()) != send_processing_preference(
                    getattr(tool, "task_metadata", {}), ctx.active_model, ctx.active_use_local, role,
                    overrides=waiter.overrides if waiter else None):
                continue
            with acquisition_budget(0):
                if current_target(root, binding) != ctx.task_id:
                    raise LateResultUnavailable("source no longer reads here")
        except LateResultUnavailable:
            mail.pop(ref, None)  # it can no longer reach this reader; the room keeps its copy
            continue
        except (OSError, ValueError, RuntimeError):
            continue  # ordinary work doesn't wait on a letter's source
        from ouroboros.owner_mailbox import deliver_task_message

        parts: list[str] = []
        deliver_task_message(entry, ctx.task_id, None, parts.append)
        source = entry["agent_note"]["late_result_origin"]
        return LetterSend(ctx, root, ref, route, live_entry=entry,
                          frame=f"[Local reply to a request from task {source}]\n{parts[0]}")
    return None


class LetterSend:
    """One write-ahead send claim per letter. A response confirms inclusion, not comprehension.

    A live letter waits for a later round only after an explicitly known pre-send refusal; a claim whose attempt the
    ledger proves unsent is returned. Any other end retires it from staging, so nothing possibly sent is offered again."""

    def __init__(self, ctx, root, ref, route, *, live_entry=None, frame=""):
        self.ctx, self.root, self.ref, self.route = ctx, root, ref, route
        self.live_entry, self.frame = live_entry, frame
        self.claim = None
        self.deadline = None
        self.refused = self.unavailable = False

    def dispatch_deadline(self):
        from ouroboros.deadline_utils import parse_deadline_ts

        try:
            deadline = parse_deadline_ts(binding_for_report(self.root, self.ref).get("deadline_at") or "")
        except (OSError, ValueError):
            return 0.0  # a missing binding cannot grant a send
        self.deadline = deadline
        return self.deadline.timestamp() if self.deadline is not None else None

    def admit(self, request, candidate, reservation) -> bool:
        from ouroboros.dialogue_provenance import agent_note_text, declared_author
        from ouroboros.store import acquisition_budget
        from ouroboros.utils import update_json_locked
        from supervisor.message_ingress import accepted_chat_message

        def refused(definite: bool = False) -> bool:  # this admission's own no, before any send of this candidate
            self.refused, self.unavailable = True, self.unavailable or definite  # definite: it cannot ride at all
            return False

        if (self.claim or request.provider != self.route.get("provider") or request.model != self.route.get("physical_model")
                or getattr(request.physical_context, "dispatch_account", "") != self.route.get("dispatch_account")
                or getattr(request, "processing_preference", None) != self.route.get("processing_preference")
                or self.ctx.active_use_local != self.route.get("use_local")):
            return refused()
        tool = self.ctx.tools._ctx
        try:
            with acquisition_budget(0):
                binding = binding_for_report(self.root, self.ref)
                if binding.get("send_claim") or binding.get("route") != self.route:
                    return refused(True)
                row = accepted_chat_message(self.root, binding["chat_id"], self.ref) or {}
                if not row.get("late_report_sha256") or not (body := agent_note_text(row)):
                    return refused(True)
                if not _wire_has_letter(candidate, self.frame or body):
                    return refused()
                if self.live_entry is not None:
                    note = self.live_entry["agent_note"]
                    if (current_target(self.root, binding) != self.ctx.task_id or body != self.live_entry.get("text")
                            or declared_author(row) != note.get("author")
                            or note.get("late_reader_attempt") != (getattr(tool, "task_attempt", None) or 1)):
                        return refused(True)
                elif return_candidate(self.ctx, None)(request) is False:
                    return refused()
            claim = {"attempt_id": reservation.attempt_id, "candidate_sha256": request.candidate_raw_sha256,
                     "task_id": str(tool.task_id)}

            def claim_send(record):
                if record.get("send_claim") or record.get("route") != self.route:
                    raise LateResultUnavailable("letter already owns a send")
                return {**record, "send_claim": claim}

            update_json_locked(_binding_path(self.root, self.ref), claim_send, timeout_sec=0, strict_existing_dict=True)
            self.claim = claim
        except LateResultUnavailable:
            return refused(True)
        except (OSError, ValueError, RuntimeError):
            return refused()
        return True

    def settle(self, capture) -> bool:
        """After the send that may carry this letter raised: True when ``capture`` proves that attempt left unsent.

        Only an explicitly known pre-send refusal keeps the letter waiting: that proven release, or its own
        admission's no. A missing capture proves nothing, so any other end retires it, and a claim stays unless its
        own attempt was proven unsent."""
        released = (capture is not None and capture.state == "released"
                    and (self.claim is None or self.claim.get("attempt_id") == capture.attempt_id))
        if self.claim is not None and released:
            from ouroboros.utils import update_json_locked

            claim = self.claim
            try:
                record = update_json_locked(_binding_path(self.root, self.ref), lambda current: (
                    {key: value for key, value in current.items() if key != "send_claim"}
                    if current.get("send_claim") == claim and "model_inclusion" not in current else None),
                    strict_existing_dict=True)
                self.claim = None if record and "send_claim" not in record else claim
            except (OSError, ValueError, RuntimeError):
                pass  # the claim stands, so its letter is never offered again
        self._restage(keep=(released or self.refused) and self.claim is None and not self.unavailable)
        return released

    def _restage(self, *, keep: bool) -> None:
        """A letter still owed its own send moves behind the other waiting letters; any other leaves its staging."""
        mail = getattr(self.ctx.tools._ctx, "_late_result_mail", None)
        if self.live_entry is not None and mail and self.ref in mail:
            entry = mail.pop(self.ref)
            if keep:
                mail[self.ref] = entry

    def observe(self, capture, usage) -> bool:
        """Only a matching physical candidate and positive response from the bound account earn inclusion.
        Any response means its request may have carried the letter, which is never offered again."""
        import logging

        from ouroboros.deadline_utils import utc_now
        from ouroboros.owner_mailbox import acknowledge_transcript_entry
        from ouroboros.utils import update_json_locked

        self._restage(keep=False)
        claim = self.claim or {}
        if (capture is None or capture.state not in {"dispatched", "settled", "unresolved"} or usage.get("provider_error")
                or capture.attempt_id != claim.get("attempt_id") or capture.candidate_raw_sha256 != claim.get("candidate_sha256")
                or capture.attempt_id != (usage.get("ledger_attempt_ids") or [None])[-1]):
            return False
        if self.deadline is not None and utc_now() >= self.deadline:
            return False  # a late terminal alone cannot date the vendor-side send inside the window
        if expected := self.route.get("observed_route"):
            served = (usage.get("claudexor") or {}).get("route") or {}
            if any(served.get(key) != value for key, value in expected.items()):
                return False
        try:
            record = update_json_locked(_binding_path(self.root, self.ref), lambda record: (
                {**record, "model_inclusion": claim} if record.get("send_claim") == claim else None),
                strict_existing_dict=True)
            if self.live_entry is not None and record.get("model_inclusion") == claim:
                acknowledge_transcript_entry(self.ctx.drive_root or self.ctx.drive_logs.parent,
                                             self.ctx.task_id, self.live_entry)
        except (OSError, ValueError, RuntimeError):
            logging.getLogger(__name__).warning("Letter inclusion receipt could not be retained", exc_info=True)
        return True


def _wire_has_letter(candidate: dict, letter: str) -> bool:
    """Look only at provider user input, where every supported adapter places the intact note: the candidate's own
    Chat or native rows, or its Responses input items (``input_text``), never a union of the two."""
    from ouroboros.context_compaction import WIRE_TEXT_TYPES, physical_message_rows

    for message in physical_message_rows(candidate) or ():
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and letter in content:
                return True
            if isinstance(content, list) and any(isinstance(part, dict) and part.get("type") in WIRE_TEXT_TYPES
                    and letter in str(part.get("text") or "") for part in content):
                return True
    return False
