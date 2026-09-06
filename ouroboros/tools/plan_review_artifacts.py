"""Exact plan-review waves and reviewer-continuation inputs.

The hot task-result state remains bounded. This companion uses the existing
task artifact store for immutable authority bytes and reconstructs either an
API transcript or a Claudexor thread id for the next evidence turn. It adds no
store, transport route, or review policy of its own.
"""

from __future__ import annotations

import copy
from hashlib import sha256
import json
import pathlib
from typing import Any, Dict, List, Optional

from ouroboros.usage_accounting import (
    PHYSICAL_ATTEMPT_STATES, POSITIVE_PHYSICAL_ATTEMPT_STATES,
)


def persist_wave(drive_root: Any, task_id: str, wave: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.observability import redact_projection
    from ouroboros.utils import utc_now_iso

    fingerprint = str(wave.get("request_fingerprint") or "")
    if len(fingerprint) != 64:
        raise ValueError("plan-review wave artifact needs a full fingerprint")
    payload = {
        **wave,
        "artifact_meta": {
            "kind": "plan_review_wave",
            "schema_version": 1,
            "producer_task_id": task_id,
            "producer_root": "artifact_store",
            "source_generation": int(wave.get("cycle_index") or 0),
            "created_at": str(
                wave.get("disposition_recorded_at") or wave.get("reviewed_at") or utc_now_iso()
            ),
            "read_operation": "read_file",
            "retention_owner": "task_artifact_store",
        },
    }
    payload = redact_projection(payload).value
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    digest = sha256(raw).hexdigest()
    cycle = int(wave.get("cycle_index") or 0)
    return store_task_artifact_bytes(
        drive_root, task_id,
        f"plan-review-wave-{cycle:04d}-{fingerprint}-{digest[:12]}.json",
        raw, kind="plan_review_wave",
    )


def read_wave(drive_root: Any, task_id: str, ref: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.artifacts import task_artifact_dir_path

    if not isinstance(ref, dict) or ref.get("root") != "artifact_store":
        raise ValueError("invalid plan-review wave artifact ref")
    name = pathlib.Path(str(ref.get("path") or "")).name
    if not name or name != str(ref.get("path") or ""):
        raise ValueError("invalid plan-review wave artifact path")
    raw = (task_artifact_dir_path(drive_root, task_id, create=False) / name).read_bytes()
    if len(raw) != int(ref.get("bytes") or -1) or sha256(raw).hexdigest() != str(ref.get("sha256") or ""):
        raise ValueError("plan-review wave artifact digest mismatch")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("plan-review wave artifact is not an object")
    return value


def authority_wave(drive_root: Any, task_id: str, hot_wave: Optional[dict]) -> Optional[dict]:
    """Materialize exact authority while retaining newer hot lifecycle stamps."""
    if not isinstance(hot_wave, dict):
        return None
    ref = hot_wave.get("wave_artifact") if isinstance(hot_wave.get("wave_artifact"), dict) else {}
    if not ref:
        return hot_wave
    exact = read_wave(drive_root, task_id, ref)
    return {**exact, **hot_wave, "findings": list(exact.get("findings") or [])}


_PLAN_REVIEW_TRANSPORT_KEYS = frozenset({
    "actors", "actors_degraded", "evidence_manifest", "health_epoch", "reasons", "retry_key",
})


def plan_review_authority_core(
    state: Dict[str, Any], *, source_ref: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Project decision authority ahead of compacted request memory and transport."""
    from ouroboros.task_results import _compact_plan_review_wave

    if not isinstance(state, dict) or not state:
        return state
    try:
        schema_version = int(state.get("schema_version") or 0)
    except (TypeError, ValueError):
        return state
    if schema_version == 1:
        return state
    core = copy.deepcopy(state)
    waves = core.get("waves") if isinstance(core.get("waves"), list) else None
    if not waves:
        return core
    last = len(waves) - 1
    core["waves"] = [
        wave if not isinstance(wave, dict)
        else (wave if wave.get("compact") else _compact_plan_review_wave(wave)) if index < last
        else {k: v for k, v in wave.items() if k not in _PLAN_REVIEW_TRANSPORT_KEYS}
        for index, wave in enumerate(waves)
    ]
    if source_ref is not None:
        latest = core["waves"][-1] if isinstance(core["waves"][-1], dict) else {}
        spec = latest.get("spec") if isinstance(latest.get("spec"), dict) else {}

        def recent(value: Any) -> Dict[str, Any]:
            items = value if isinstance(value, list) else []
            return {"items": copy.deepcopy(items[-4:]), "items_omitted": max(0, len(items) - 4), "total": len(items)}

        core["decision_core"] = {
            "identity": {key: copy.deepcopy(latest[key]) for key in ("cycle_index", "request_fingerprint", "previous_fingerprint", "spec_hash", "evidence_manifest_hash", "aggregate", "closed", "paid") if key in latest},
            "goal": spec.get("goal"), "acceptance_claims": recent(spec.get("acceptance_claims")),
            "findings": recent(latest.get("findings")), "dispositions": recent(latest.get("dispositions")),
        }
        core["waves"] = [_compact_plan_review_wave(wave) if isinstance(wave, dict) and not wave.get("compact") else wave for wave in core["waves"]]
        core["need_evidence_seen"] = recent(core.get("need_evidence_seen"))
        dropped_keys = sorted({key for wave in waves if isinstance(wave, dict)
                               for key in _PLAN_REVIEW_TRANSPORT_KEYS if key in wave})
        core["projection"] = {
            "projected_from": "plan_review_authority_core", "dropped_keys": dropped_keys,
            "full_chars": len(json.dumps(state, ensure_ascii=False, sort_keys=True, default=str)),
            "source_ref": {**copy.deepcopy(source_ref), "field": "authority.plan_review_state"},
        }
    return core


def _row_has_physical_dispatch(row: Dict[str, Any]) -> bool:
    """Identify paid rows from explicit custody facts, with legacy fallback."""
    operation_state = str(row.get("operation_state") or "").strip().lower()
    status = str(row.get("status") or "").strip().lower()
    physical_state = str(row.get("physical_attempt_state") or "").strip().lower()
    if not physical_state and isinstance(row.get("usage"), dict):
        physical_state = str(
            row["usage"].get("physical_attempt_state") or ""
        ).strip().lower()
    # A non-empty state outside the physical-attempt enum is malformed custody,
    # not evidence that the synthetic operation id was free.  Keep it on the
    # paid side until the caller rejects the row rather than laundering it into
    # a $0 pre-dispatch refusal.
    if physical_state and physical_state not in PHYSICAL_ATTEMPT_STATES:
        return True
    if physical_state in {"reserved", "released"}:
        return False
    if physical_state in POSITIVE_PHYSICAL_ATTEMPT_STATES:
        return True
    # With no physical capture, an explicit $0 state wins over the synthetic
    # operation id assigned before provider admission.
    if operation_state == "not_dispatched" or status == "not_dispatched":
        return False
    # Pre-B1 rows and a current substrate omission may lack an operation id.
    # Absence is not proof of $0: only the explicit states above authorize that
    # conclusion, while every ambiguous row stays on the conservative side.
    return True


def in_flight_resume_inputs(
    existing: Dict[str, Any], state: Dict[str, Any], state_root: pathlib.Path,
    task_id: str, configured_slots: list,
) -> Dict[str, Any]:
    """Recover the exact physical set of one already-paid plan-review cycle."""
    from ouroboros.tools.plan_review_runtime import plan_reviewer_config_fingerprint

    stored_roster = str(existing.get("reviewer_config_fingerprint") or "")
    if stored_roster and stored_roster != plan_reviewer_config_fingerprint(configured_slots):
        return {"error": (
            "The reviewer roster changed while the prior paid cycle is still in flight. "
            "Refusing to mix rosters or start a second panel before that cycle settles."
        )}
    previous = None
    previous_fingerprint = str(existing.get("previous_fingerprint") or "")
    if previous_fingerprint:
        from ouroboros.task_results import plan_review_wave

        previous = plan_review_wave(state, previous_fingerprint)
        if previous is not None:
            try:
                previous = authority_wave(state_root, task_id, previous)
            except (OSError, ValueError, json.JSONDecodeError):
                return {"error": (
                    "Prior exact plan-review authority is unreadable; "
                    "in-flight reconciliation is refused."
                )}
    raw_actor_rows = existing.get("actors")
    if not isinstance(raw_actor_rows, list) or any(
        not isinstance(row, dict) for row in raw_actor_rows
    ):
        return {"error": (
            "The prior paid cycle's reviewer roster is malformed. Refusing to "
            "drop rows or infer which physical calls own custody."
        )}
    actor_rows = list(raw_actor_rows)
    for row in actor_rows:
        physical_state = str(row.get("physical_attempt_state") or "").strip().lower()
        if not physical_state and isinstance(row.get("usage"), dict):
            physical_state = str(
                row["usage"].get("physical_attempt_state") or ""
            ).strip().lower()
        if physical_state and physical_state not in PHYSICAL_ATTEMPT_STATES:
            return {"error": (
                "The prior paid cycle contains an unknown physical-attempt state. "
                "Refusing to infer custody from malformed reviewer facts."
            )}
    configured_ids = {str(getattr(slot, "slot_id", "") or "") for slot in configured_slots}
    actor_ids = [str(row.get("slot_id") or "") for row in actor_rows]
    if not actor_rows or any(not slot_id for slot_id in actor_ids) \
            or len(actor_ids) != len(set(actor_ids)) or set(actor_ids) != configured_ids:
        return {"error": (
            "The prior paid cycle's exact reviewer rows do not match its frozen roster. "
            "Refusing to guess which physical calls own custody."
        )}
    dispatched_ids = {
        str(row.get("slot_id") or "") for row in actor_rows
        if _row_has_physical_dispatch(row)
    }
    if not dispatched_ids or any(
        (str(row.get("operation_state") or "") == "in_flight"
         or bool(row.get("late_result_pending")))
        and str(row.get("slot_id") or "") not in dispatched_ids
        for row in actor_rows
    ):
        return {"error": (
            "The prior paid cycle does not contain an exact physical-dispatch set. "
            "Refusing to infer custody from current reviewer health."
        )}
    frozen_rows = []
    for row in actor_rows:
        if str(row.get("slot_id") or "") in dispatched_ids:
            continue
        if bool(row.get("ok")) or not str(row.get("error") or ""):
            return {"error": (
                "A prior reviewer row lacks physical-dispatch custody and is not a frozen "
                "$0 refusal. Reconciliation is refused."
            )}
        frozen = dict(row)
        frozen.setdefault("text", "")
        frozen.setdefault("request_model", str(row.get("model") or ""))
        frozen_rows.append(frozen)
    health_evidence = {
        str(row.get("slot") or ""): {
            "failure_code": str(row.get("code") or ""),
            "reset_at": str(row.get("reset_at") or ""),
        }
        for row in (existing.get("health_epoch") or []) if isinstance(row, dict)
        and str(row.get("slot") or "")
    }
    cycle_index = int(existing.get("cycle_index") or state.get("cycles_paid") or 1)
    return {
        "previous": previous,
        "cycle_index": cycle_index,
        "retry_key": str(existing.get("retry_key") or "")
        or f"plan_review:{existing.get('request_fingerprint')}:{cycle_index}",
        "dispatched_slot_ids": sorted(dispatched_ids),
        "frozen_rows": frozen_rows,
        "health_evidence": health_evidence,
    }


def hot_index_wave(wave: dict, *, page_size: int) -> dict:
    """Keep a bounded per-slot page; exact authority stays in ``wave_artifact``."""
    findings = [dict(row) for row in wave.get("findings") or [] if isinstance(row, dict)]
    counts: Dict[str, int] = {}
    page = []
    for finding in findings:
        slot = str(finding.get("slot") or "")
        count = counts.get(slot, 0)
        counts[slot] = count + 1
        if count < max(1, int(page_size)):
            page.append(finding)
    if len(page) == len(findings):
        return wave
    return {
        **wave, "findings": page, "findings_total": len(findings),
        "findings_paged": True,
    }


def continuation_state(
    state_root: pathlib.Path, task_id: str, previous: Optional[dict], slots: List[Any],
    manifest: dict, *, user_content: str,
) -> tuple[List[Any], Dict[str, List[Dict[str, Any]]], Dict[str, str], str]:
    """Resolve one evidence continuation; the fourth element names a restart cause.

    The guard is load-bearing: a cycle whose manifest names no reviewer-requested
    locator has no prior reviewer thread to continue, so the configured slots are
    returned untouched instead of reporting an absent predecessor wave."""
    if not manifest.get("reviewer_requested"):
        return slots, {}, {}, ""
    return continuation_inputs(
        state_root, task_id, previous, slots, user_content=user_content,
    )


def record_exact_wave(
    state_root: pathlib.Path, task_id: str, wave: dict, exact: dict,
    *, need_evidence_seen: List[str], page_size: int,
) -> dict:
    """Persist exact bytes first, then publish their bounded hot index."""
    from ouroboros.task_results import record_plan_review_wave

    wave["wave_artifact"] = persist_wave(state_root, task_id, exact)
    return record_plan_review_wave(
        state_root, task_id, hot_index_wave(wave, page_size=page_size),
        need_evidence_seen=need_evidence_seen,
    )


def slot_row(slot: Any) -> dict:
    route = getattr(slot, "route", "api_chat")
    return {
        "slot_id": str(getattr(slot, "slot_id", "") or ""),
        "model": str(getattr(slot, "model", "") or ""),
        "effort": str(getattr(slot, "effort", "") or ""),
        "route": str(getattr(route, "value", route) or "api_chat"),
        "session_target": str(getattr(slot, "session_target", "") or ""),
        "session_profile": str(getattr(slot, "session_profile", "") or ""),
    }


def continuation_restart_delta(cause: str) -> dict:
    """The existing-style disclosure of one continuation that restarted fresh."""
    return {
        "kind": "capability_delta",
        "requested": "continuation of prior thread",
        "effective": "fresh session, full packet",
        "reason": str(cause or ""),
    }


def attach_continuation_restart_delta(rows: List[dict], cause: str) -> None:
    """Disclose one fresh continuation restart on every slot row (no-op when
    the continuation held). Thread memory was lost and the wave re-dispatched
    fresh with the full packet: disclosed per slot through the existing
    capability-delta lane."""
    if not cause:
        return
    for row in rows:
        row["capability_delta"] = [
            *(row.get("capability_delta") or []),
            continuation_restart_delta(cause),
        ]


def continuation_inputs(
    state_root: pathlib.Path, task_id: str, previous: Optional[dict], slots: List[Any],
    *, user_content: str,
) -> tuple[List[Any], Dict[str, List[Dict[str, Any]]], Dict[str, str], str]:
    """Rebuild one evidence continuation from the prior exact wave.

    Every miss here is a cache miss, never a validity event: the dispositions
    custody chain is enforced one level up, before this function is reached. An
    absent, unreferenced or unreadable prior exact wave, a changed reviewer
    roster, a prior slot receipt or thread that is gone, an invalid prior API
    transcript — each degrades to a FRESH full-packet dispatch, because the
    packet is self-contained on every send (prior findings, dispositions and
    spec delta already ride it). The fourth element names the typed cause of
    such a restart ('' when continuation held); slots are returned exactly as
    currently configured, never rebound to prior rows."""

    def fresh(cause: str) -> tuple[List[Any], Dict[str, List[Dict[str, Any]]], Dict[str, str], str]:
        return slots, {}, {}, cause

    if not previous:
        return fresh("prior_exact_wave_missing")
    ref = previous.get("wave_artifact") if isinstance(previous.get("wave_artifact"), dict) else {}
    if not ref:
        return fresh("prior_exact_wave_ref_missing")
    try:
        exact = read_wave(state_root, task_id, ref)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return fresh(f"prior_exact_wave_unreadable:{type(exc).__name__}")
    current_rows = [slot_row(slot) for slot in slots]
    if current_rows != [r for r in exact.get("slots") or [] if isinstance(r, dict)]:
        return fresh("prior_reviewer_assignment_set_changed")
    outputs = {str(r.get("slot_id") or ""): r for r in exact.get("reviewer_outputs") or [] if isinstance(r, dict)}
    slot_messages: Dict[str, List[Dict[str, Any]]] = {}
    session_threads: Dict[str, str] = {}
    for config in current_rows:
        sid = str(config.get("slot_id") or "")
        output = outputs.get(sid)
        if not output:
            return fresh(f"prior_slot_receipt_missing:{sid}")
        if str(config.get("route") or "") == "agent_session":
            thread_id = str(output.get("review_thread_id") or "")
            if not thread_id:
                return fresh(f"prior_review_thread_missing:{sid}")
            session_threads[sid] = thread_id
        else:
            prior_messages = output.get("request_messages")
            if (
                not isinstance(prior_messages, list)
                or not prior_messages
                or any(
                    not isinstance(row, dict)
                    or not str(row.get("role") or "").strip()
                    or "content" not in row
                    for row in prior_messages
                )
            ):
                return fresh(f"prior_api_transcript_invalid:{sid}")
            slot_messages[sid] = [
                *[dict(row) for row in prior_messages],
                {"role": "assistant", "content": str(output.get("text") or "")},
                {"role": "user", "content": user_content},
            ]
    return slots, slot_messages, session_threads, ""


def exact_wave(
    wave: dict, *, plan_prose: str, manifest: dict, slots: List[Any], rows: List[dict],
    system_prompt: str, user_content: str, session_task: str,
    slot_messages: Dict[str, List[Dict[str, Any]]],
) -> dict:
    from ouroboros.tools.plan_packet import plan_user_stable_len
    from ouroboros.tools.review_synthesis import build_plan_review_messages

    common = build_plan_review_messages(system_prompt, user_content, plan_user_stable_len(user_content))
    outputs = []
    for row in rows:
        sid, route = str(row.get("slot_id") or ""), str(row.get("route") or "")
        outputs.append({
            "slot_id": sid, "model": str(row.get("model") or ""),
            "request_model": str(row.get("request_model") or ""), "route": route,
            "text": str(row.get("text") or ""), "error": str(row.get("error") or ""),
            "request_messages": (
                list(slot_messages[sid]) if sid in slot_messages else common
            ) if route == "api_chat" else [],
            "session_task": session_task if route == "agent_session" else "",
            "review_thread_id": str(row.get("review_thread_id") or ""),
            "review_turn_id": str(row.get("review_turn_id") or ""),
            "review_thread_receipt": row.get("review_thread_receipt") or {},
            "auth_route_receipt": row.get("auth_route_receipt") or {},
            "profile_continuity_receipt": row.get("profile_continuity_receipt") or {},
            "applied_profile": str(row.get("applied_profile") or ""),
            "prompt_ref": row.get("prompt_ref") or {}, "response_ref": row.get("response_ref") or {},
        })
    return {
        **wave, "plan_prose": plan_prose, "evidence_manifest_full": manifest,
        "slots": [slot_row(slot) for slot in slots], "reviewer_outputs": outputs,
    }
