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


class PlanReviewSourceUnavailable(ValueError):
    """A recorded plan names full authority that this reader cannot resolve."""


def persist_wave(drive_root: Any, task_id: str, wave: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.artifacts import store_actor_source_bytes
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
    cycle = int(wave.get("cycle_index") or 0)
    return store_actor_source_bytes(
        drive_root, task_id, category="context_checkpoints",
        source_id=f"plan-review-wave-{cycle:04d}-{fingerprint}", data=raw, extension="json",
    )



def persist_historical_result(drive_root: Any, task_id: str, wave: dict, result: dict) -> dict:
    """Full late feedback is a source artifact, not a replacement wave verdict."""
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.observability import redact_projection
    from ouroboros.tools import plan_spec

    findings, error = plan_spec.parse_findings(str(result.get("text") or ""))
    payload = redact_projection({
        "kind": "plan_review_historical_supplement", "task_id": task_id,
        "request_fingerprint": wave["request_fingerprint"], "cycle_index": wave["cycle_index"],
        "retry_key": wave.get("retry_key"), "original_wave_artifact": wave.get("wave_artifact") or {},
        "result": result, "parsed_findings": findings, "parse_error": error,
    }).value
    return store_actor_source_bytes(
        drive_root, task_id, category="context_checkpoints",
        source_id=f"plan-review-late-{result['operation_id']}",
        data=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode(),
        extension="json",
    )

def read_wave(drive_root: Any, task_id: str, ref: Dict[str, Any]) -> Dict[str, Any]:
    from ouroboros.artifacts import task_artifact_dir_path

    if not isinstance(ref, dict) or ref.get("root") != "artifact_store":
        raise ValueError("invalid plan-review wave artifact ref")
    if ref.get("kind") == "task_source":
        from ouroboros.artifacts import read_actor_source_bytes
        raw = read_actor_source_bytes(drive_root, task_id, ref)
    else:
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
        if hot_wave.get("compact") or hot_wave.get("spec_in_artifact") or hot_wave.get("spec_body_truncated"):
            raise PlanReviewSourceUnavailable("PLAN_REVIEW_SOURCE_UNAVAILABLE: full spec has no artifact reference")
        return hot_wave
    if not drive_root or not task_id:
        raise PlanReviewSourceUnavailable("PLAN_REVIEW_SOURCE_UNAVAILABLE: artifact owner is unknown")
    try:
        exact = read_wave(drive_root, task_id, ref)
    except (OSError, ValueError) as exc:
        raise PlanReviewSourceUnavailable(
            f"PLAN_REVIEW_SOURCE_UNAVAILABLE: task {task_id}, artifact {ref.get('path')}: {exc}"
        ) from exc
    source = hot_wave.get("spec_source_ref") or exact.get("spec_source_ref")
    if source:
        from ouroboros.artifacts import read_actor_source_bytes
        from ouroboros.tools.plan_spec import spec_hash

        try:
            spec = json.loads(read_actor_source_bytes(drive_root, task_id, source))
            if not isinstance(spec, dict) or spec_hash(spec) != hot_wave.get("spec_hash", exact.get("spec_hash")):
                raise ValueError("operative spec hash mismatch")
        except (OSError, ValueError) as exc:
            raise PlanReviewSourceUnavailable(f"PLAN_REVIEW_SOURCE_UNAVAILABLE: {exc}") from exc
    elif hot_wave.get("spec_in_artifact"):
        raise PlanReviewSourceUnavailable("PLAN_REVIEW_SOURCE_UNAVAILABLE: operative spec reference is missing")
    elif not hot_wave.get("spec_body_truncated") and isinstance(hot_wave.get("spec"), dict):
        spec = hot_wave["spec"]  # Legacy inline authority predates the source handle.
    else:
        from ouroboros.tools.plan_spec import spec_hash

        spec = exact.get("spec")
        if (not isinstance(spec, dict) or exact.get("spec_body_truncated")
                or spec_hash(spec) != exact.get("spec_hash")):
            raise PlanReviewSourceUnavailable("PLAN_REVIEW_SOURCE_UNAVAILABLE: artifact has no complete spec")
    dialogue_ref = hot_wave.get("dialogue_source_ref") or exact.get("dialogue_source_ref")
    if dialogue_ref:
        from ouroboros.artifacts import read_actor_source_bytes
        read_actor_source_bytes(drive_root, task_id, dialogue_ref)
    restored = {
        **exact, **hot_wave,
        "spec": copy.deepcopy(spec), "goal": spec.get("goal") or "",
        "findings": list(exact.get("findings") or []),
    }
    if dialogue_ref and isinstance(restored.get("evidence_manifest_full"), dict):
        from ouroboros.artifacts import task_artifact_dir_path
        restored["evidence_manifest_full"] = copy.deepcopy(restored["evidence_manifest_full"])
        own = restored["evidence_manifest_full"].get("own_dialogue")
        if isinstance(own, dict):
            own.update(source_ref=dialogue_ref, file=str(task_artifact_dir_path(drive_root, task_id, create=False) / dialogue_ref["path"]))
    restored.pop("spec_in_artifact", None)
    restored.pop("spec_body_truncated", None)
    return restored


def authority_state(drive_root: Any, task_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the current spec; historical consumers resolve their selected wave."""
    from ouroboros.task_results import current_plan_review_wave

    current = current_plan_review_wave(state)
    if not current or current.get("compact") or not (
        current.get("spec_source_ref") or current.get("spec_in_artifact")
        or (current.get("spec_body_truncated") and current.get("wave_artifact"))
    ):
        return state
    resolved = authority_wave(drive_root, task_id, current)
    return {**state, "waves": [
        resolved if wave.get("request_fingerprint") == current.get("request_fingerprint") else wave
        for wave in state.get("waves") or []
    ]}


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
            "author_disposition": copy.deepcopy(latest.get("author_disposition"))
            if isinstance(latest.get("author_disposition"), dict) else None,
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
    # operation id assigned before provider admission; a slot released at the
    # dispatch barrier (``pending_dispatch``) is unproven, hence $0 until settled.
    if operation_state in {"not_dispatched", "pending_dispatch"} or status == "not_dispatched":
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
    replaced = existing.get("previous_wave_artifact") if isinstance(existing.get("previous_wave_artifact"), dict) else {}
    if replaced:  # a same-fingerprint re-dispatch replaced its predecessor in the hot index: read the exact copy
        try:
            previous = read_wave(state_root, task_id, replaced)
        except (OSError, ValueError, json.JSONDecodeError):
            return {"error": "Prior exact plan-review authority is unreadable; in-flight reconciliation is refused."}
    elif previous_fingerprint:
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
    if (replaced or previous_fingerprint) and (previous is None or (
            previous.get("cycle_index") == existing.get("cycle_index")
            and str(previous.get("request_fingerprint") or "") == str(existing.get("request_fingerprint") or ""))):
        return {"error": (  # a recorded predecessor that resolves to nothing, or to this very wave, is not a first wave
            "Prior plan-review predecessor cannot be resolved to a distinct wave; in-flight reconciliation is refused."
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
    # The cycle's own physical set: rows proven dispatched plus rows released at
    # the dispatch barrier (awaiting their worker's report) — never re-dispatched.
    dispatched_ids = {
        str(row.get("slot_id") or "") for row in actor_rows
        if _row_has_physical_dispatch(row)
        or str(row.get("operation_state") or "") == "pending_dispatch"
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
        "dispatched_rows": [dict(row) for row in actor_rows if row["slot_id"] in dispatched_ids],
        "frozen_rows": frozen_rows,
        "health_evidence": health_evidence,
    }


def hot_index_wave(wave: dict, *, page_size: int) -> dict:
    """Keep a bounded per-slot page; exact authority stays in ``wave_artifact``."""
    if wave.get("spec_source_ref"):
        # The operative spec has no size limit. Its exact source already exists;
        # never duplicate it inside the bounded task-result index.
        wave = {**wave, "spec": {}, "spec_in_artifact": True}
        wave.pop("goal", None)
        wave.pop("evidence_manifest", None)  # Already preserved by wave_artifact.
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
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.task_results import record_plan_review_wave

    # Raw operative authority already lived in the task result. Keep it exact
    # under the existing source handle; wave evidence/output redaction stays on.
    source = store_actor_source_bytes(
        state_root, task_id, category="context_checkpoints", source_id="plan-spec",
        data=json.dumps(wave["spec"], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        extension="json",
    )
    wave["spec_source_ref"] = source
    manifest = exact.get("evidence_manifest_full") or {}
    own = manifest.get("own_dialogue") or {}
    identity = {"author_request_fingerprint": manifest.get("author_request_fingerprint", ""),
                "dialogue_chat_id": own.get("chat_id")}
    if own.get("source_ref"):
        identity["dialogue_source_ref"] = own["source_ref"]
    wave.update(identity)
    exact = {**exact, **identity, "spec_source_ref": source}
    wave["wave_artifact"] = persist_wave(state_root, task_id, exact)
    stored = record_plan_review_wave(
        state_root, task_id, hot_index_wave(wave, page_size=page_size),
        need_evidence_seen=need_evidence_seen,
    )
    # A slot can settle after the pre-supersede collection but before this
    # publication. Reconcile just the known history at this existing write seam.
    from ouroboros.task_results import load_plan_review_state
    from ouroboros.tools.plan_review_collect import attach_historical_results
    for prior in load_plan_review_state(state_root, task_id).get("waves") or []:
        if prior.get("custody_pending") and (prior.get("closed") or
                prior.get("request_fingerprint") != stored.get("request_fingerprint")):
            try:
                attach_historical_results(state_root, task_id, fingerprint=prior["request_fingerprint"])
            except (OSError, ValueError, TimeoutError):
                # The new wave is already durable. Missing old source is not a
                # failure to record it and must not invite duplicate dispatch.
                import logging
                logging.getLogger(__name__).warning("historical plan source remains unresolved", exc_info=True)
    return authority_wave(state_root, task_id, stored)


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


def frozen_delivery_inputs(wave: dict, slots: list) -> dict:
    """Reuse exact request policy and per-slot inputs; never re-fit live context."""
    policy, sizes = wave.get("request_policy"), wave.get("slot_prompt_chars")
    if not isinstance(policy, dict) or not isinstance(sizes, dict):
        raise PlanReviewSourceUnavailable(
            "PLAN_REVIEW_SOURCE_UNAVAILABLE: original request policy/fit was not recorded; "
            "current values cannot stand in for the paid request")
    outputs = {str(row.get("slot_id") or ""): row for row in wave.get("reviewer_outputs") or []}
    actors = {str(row.get("slot_id") or ""): row for row in wave.get("actors") or []}
    messages, tasks = {}, {}
    for slot in slots:
        sid = str(slot.slot_id)
        if actors.get(sid, {}).get("operation_state") == "not_dispatched":
            continue  # A frozen zero-send refusal has no paid input to rejoin.
        row = outputs.get(sid)
        if not isinstance(row, dict) or sid not in sizes:
            raise PlanReviewSourceUnavailable(f"PLAN_REVIEW_SOURCE_UNAVAILABLE: recorded slot inputs missing: {sid}")
        if bool(getattr(slot, "retrieves", False)):
            if not row.get("session_task"):
                raise PlanReviewSourceUnavailable(f"PLAN_REVIEW_SOURCE_UNAVAILABLE: recorded retrieving task missing: {sid}")
            tasks[sid] = str(row["session_task"])
        else:
            if not row.get("request_messages"):
                raise PlanReviewSourceUnavailable(f"PLAN_REVIEW_SOURCE_UNAVAILABLE: recorded packet missing: {sid}")
            messages[sid] = copy.deepcopy(row["request_messages"])
    return {"request_policy": copy.deepcopy(policy), "slot_messages": messages,
            "slot_session_tasks": tasks, "slot_prompt_chars": dict(sizes),
            "dialogue_delivery": copy.deepcopy(wave.get("dialogue_delivery") or {}),
            "native_mandatory_read_chars": int(policy.get("native_mandatory_read_chars") or 0)}


def exact_wave(
    wave: dict, *, plan_prose: str, manifest: dict, slots: List[Any], rows: List[dict],
    system_prompt: str, user_content: str, session_task: str,
    slot_messages: Dict[str, List[Dict[str, Any]]], dispatched: Optional[dict] = None,
    slot_session_tasks: Optional[dict] = None, dialogue_delivery: Optional[dict] = None,
    request_policy: Optional[dict] = None, slot_prompt_chars: Optional[dict] = None,
) -> dict:
    """``dispatched`` = the exact wave a reconciliation is resuming over.

    A reconcile-only cycle (the $0 collection and the identical-envelope resume)
    physically sends nothing: it re-records the wave the reviewers already answered.
    The packet is rebuilt from the LIVE task context on that path, so a directive that
    arrived after the dispatch would otherwise be written into the reviewers' recorded
    request and, through ``continuation_inputs``, into the prior history of the next
    paid cycle. The recorded request of each slot that already has one is therefore
    carried forward byte for byte; only a slot with no recorded request (a roster row
    the dispatched wave never had) falls back to the rebuilt packet."""
    from ouroboros.tools.plan_packet import plan_user_stable_len
    from ouroboros.tools.review_synthesis import build_plan_review_messages

    common = build_plan_review_messages(system_prompt, user_content, plan_user_stable_len(user_content))
    sent = {
        str(r.get("slot_id") or ""): r
        for r in ((dispatched or {}).get("reviewer_outputs") or []) if isinstance(r, dict)
    }
    native_slots = {str(slot.slot_id) for slot in slots if bool(getattr(slot, "native_retrieval", False))}
    outputs = []
    for row in rows:
        sid, route = str(row.get("slot_id") or ""), str(row.get("route") or "")
        recorded = sent.get(sid) or {}
        outputs.append({
            "slot_id": sid, "model": str(row.get("model") or ""),
            "request_model": str(row.get("request_model") or ""), "route": route,
            "text": str(row.get("text") or ""), "error": str(row.get("error") or ""),
            "request_messages": (
                [dict(m) for m in recorded["request_messages"]]
                if isinstance(recorded.get("request_messages"), list) and recorded["request_messages"]
                else list(slot_messages[sid]) if sid in slot_messages else common
            ) if route == "api_chat" and sid not in native_slots else [],
            "session_task": (
                str(recorded.get("session_task") or "") or (slot_session_tasks or {}).get(sid) or session_task
            ) if route == "agent_session" or sid in native_slots else "",
            "delivery_class": "native_retrieving" if sid in native_slots else route,
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
        "dialogue_delivery": dispatched.get("dialogue_delivery", {}) if dispatched is not None else dialogue_delivery or {},
        "request_policy": copy.deepcopy(dispatched["request_policy"] if dispatched is not None else request_policy),
        "slot_prompt_chars": copy.deepcopy(dispatched["slot_prompt_chars"] if dispatched is not None else slot_prompt_chars),
        "slots": [slot_row(slot) for slot in slots], "reviewer_outputs": outputs,
    }


def compact_wave(wave: Dict[str, Any]) -> Dict[str, Any]:
    """Bounded summary of an older wave (S2): identity, outcome, counts, closure."""
    findings = wave.get("findings") if isinstance(wave.get("findings"), list) else []
    return {
        "compact": True,
        "cycle_index": wave.get("cycle_index"),
        "request_fingerprint": str(wave.get("request_fingerprint") or ""),
        "aggregate": str(wave.get("aggregate") or ""),
        "counts": {
            "findings": int(wave.get("findings_total") or len(findings)),
            "dispositions": len(wave.get("dispositions") or []),
            "blocking": int(wave["counts"].get("blocking") or 0) if isinstance(wave.get("counts"), dict) and "blocking" in wave["counts"] else sum(1 for f in findings if isinstance(f, dict) and f.get("class") == "blocking"),
        },
        "closed": bool(wave.get("closed")),
        "paid": bool(wave.get("paid")),
        "wave_artifact": copy.deepcopy(wave.get("wave_artifact") or {}),
        **{key: copy.deepcopy(wave[key]) for key in ("historical_supplements", "retry_key", "custody_pending", "ordered_weaker", "previous_wave_artifact") if key in wave},
        **({"author_disposition": copy.deepcopy(wave["author_disposition"])}
           if isinstance(wave.get("author_disposition"), dict) else {}),
        **({"spec_source_ref": copy.deepcopy(wave["spec_source_ref"])} if wave.get("spec_source_ref") else {}),
        **{key: copy.deepcopy(wave[key]) for key in ("dialogue_source_ref", "dialogue_chat_id", "author_request_fingerprint") if key in wave},
        **({"reviewed_at": str(wave["reviewed_at"])} if wave.get("reviewed_at") else {}),
    }


def record_plan_review_supplement(
    results_drive_root: Any, task_id: str, *, wave: dict, result: dict, source_ref: dict,
) -> bool:
    """One locked historical write; critic authority and current selection stay intact.

    The caller already verified the original prompt/complete producer CAS. A
    concurrent new cycle/current-wave change is rechecked here, not restored
    from a pre-lock snapshot. The full feedback remains in the source artifact.
    """
    from ouroboros.task_results import _update_plan_review_state

    attached = False
    def _attach(state: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal attached
        target = next((w for w in state["waves"] if w.get("request_fingerprint") == wave["request_fingerprint"]), None)
        if target is None or target.get("cycle_index") != wave.get("cycle_index"):
            return state
        if str(target.get("retry_key") or "") != str(wave.get("retry_key") or ""):
            return state
        if not target.get("closed") and (state.get("current_attempt") or {}).get("fingerprint") == wave["request_fingerprint"]:
            return state
        if not any(row.get("slot_id") == result.get("slot_id") and
                   row.get("operation_id") == result.get("operation_id") for row in wave.get("actors") or []):
            return state
        if isinstance(target.get("actors"), list) and not any(
            row.get("slot_id") == result.get("slot_id") and row.get("operation_id") == result.get("operation_id")
            for row in target["actors"]
        ):
            return state
        supplements = target.setdefault("historical_supplements", [])
        if any(row.get("operation_id") == result["operation_id"] for row in supplements):
            return state
        supplements.append({
            key: copy.deepcopy(result.get(key)) for key in
            ("slot_id", "operation_id", "operation_state", "physical_attempt_state")
        } | {"source_ref": copy.deepcopy(source_ref), "cycle_index": wave["cycle_index"],
             "status": "error" if result.get("error") else "ok"})
        if _row_has_physical_dispatch(result) and not target.get("paid"):
            target["paid"] = True
            state["cycles_paid"] = int(state.get("cycles_paid") or 0) + 1
        settled = {row["operation_id"] for row in supplements}
        target["custody_pending"] = any(
            (row.get("late_result_pending") or row.get("operation_state") in
             {"pending_dispatch", "in_flight", "custody_lost"})
            and row.get("operation_id") not in settled for row in wave.get("actors") or []
        )
        attached = True
        return state

    _update_plan_review_state(results_drive_root, task_id, _attach)
    return attached


def current_author_plan(drive_root: Any, task_id: str, state: dict) -> Optional[dict]:
    """Resolve the selected author source separately from closed reviewer authority."""
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_records import validate_author_disposition

    attempt = state.get("current_attempt") or {}
    subject = attempt.get("author_subject") or {}
    author = validate_author_disposition(subject.get("author_disposition"), subject_hash=str(attempt.get("fingerprint") or ""))
    if not author:
        return None
    try:
        value = json.loads(read_actor_source_bytes(drive_root, task_id, subject["source_ref"]))
        if value.get("kind") != "plan_author_subject" or value.get("fingerprint") != author["subject_hash"] or not isinstance(value.get("spec"), dict):
            raise ValueError("author plan source identity mismatch")
        return {**value, "author_disposition": author, "review_fingerprint": subject["review_fingerprint"]}
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise PlanReviewSourceUnavailable(f"PLAN_AUTHOR_SOURCE_UNAVAILABLE: {exc}") from exc


def row_pending(row: Dict[str, Any]) -> bool:
    """A reviewer row whose answer has not arrived (awaiting, in flight, late-pending): never a terminal absence."""
    return not row.get("ok") and (bool(row.get("late_result_pending")) or str(row.get("operation_state") or "settled") in (
        "pending_dispatch", "in_flight", "custody_lost"))


def _pending_seats(wave: Dict[str, Any]) -> set[str]:
    return {str(r.get("slot_id") or "") for r in wave.get("actors") or [] if isinstance(r, dict) and row_pending(r)}


def _earlier_wave(state_root: Any, task_id: str, state: Dict[str, Any], wave: Dict[str, Any]) -> Optional[dict]:
    """The exact predecessor of ``wave``: by its recorded artifact pointer, else the hot index
    materialized as authority. ``None`` only when the wave names no predecessor; an unreadable,
    evicted or self-naming predecessor is a source failure, never an empty history."""
    ref = wave.get("previous_wave_artifact") if isinstance(wave.get("previous_wave_artifact"), dict) else {}
    fingerprint = str(wave.get("previous_fingerprint") or "")
    if ref:
        try:
            earlier = read_wave(state_root, task_id, ref)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise PlanReviewSourceUnavailable(
                f"PLAN_REVIEW_SOURCE_UNAVAILABLE: predecessor artifact {ref.get('path')}: {exc}") from exc
    elif fingerprint:
        from ouroboros.task_results import plan_review_wave

        hot = plan_review_wave(state, fingerprint)
        if hot is None:
            raise PlanReviewSourceUnavailable(
                f"PLAN_REVIEW_SOURCE_UNAVAILABLE: predecessor wave {fingerprint[:8]} is not in the index")
        earlier = authority_wave(state_root, task_id, hot)
    else:
        return None
    if isinstance(earlier, dict) and earlier.get("cycle_index") == wave.get("cycle_index") and str(
            earlier.get("request_fingerprint") or "") == str(wave.get("request_fingerprint") or ""):
        raise PlanReviewSourceUnavailable("PLAN_REVIEW_SOURCE_UNAVAILABLE: a plan-review wave names itself as its predecessor")
    return earlier if isinstance(earlier, dict) else None


def standing_findings_lineage(state_root: Any, task_id: str, state: Dict[str, Any], previous: Optional[dict],
                              spec: dict, enforcement: str) -> Dict[str, list]:
    """Per-seat standing findings across the same-spec lineage. A seat still pending when its
    wave was superseded gave no terminal answer there, so its obligation comes from the wave
    before, walked back until a real answer, a closed predecessor or a changed spec ends it.
    The set of seats under walk only shrinks: a seat answered in a newer wave is never
    re-added from an older one. History that cannot be read raises (fail closed): an unknown
    obligation is never an empty one."""
    from ouroboros.tools import plan_spec

    target = plan_spec.spec_hash(spec)

    def same_spec(wave: Any) -> bool:
        if not isinstance(wave.get("spec"), dict):  # unresolved spec authority is not a verified mismatch
            raise PlanReviewSourceUnavailable("PLAN_REVIEW_SOURCE_UNAVAILABLE: a plan-review wave carries no operative spec")
        return str(wave.get("spec_hash") or plan_spec.spec_hash(wave["spec"])) == target

    if not isinstance(previous, dict) or not same_spec(previous) or previous.get("closed"):
        return {}
    standing = plan_spec.plan_standing_findings(previous, spec, enforcement)
    pending = _pending_seats(previous) - set(standing)
    wave, seen = previous, set()
    while pending:
        key = (str(wave.get("request_fingerprint") or ""), wave.get("cycle_index"))
        if key in seen:
            raise PlanReviewSourceUnavailable(f"PLAN_REVIEW_SOURCE_UNAVAILABLE: plan-review lineage loops at {key[0][:8]}")
        seen.add(key)
        earlier = _earlier_wave(state_root, task_id, state, wave)
        if earlier is None or earlier.get("closed") or not same_spec(earlier):
            break  # the chain starts here, or a closed / changed-spec predecessor ended every obligation
        step = plan_spec.plan_standing_findings(earlier, spec, enforcement)
        rows = {str(r.get("slot_id") or ""): r for r in earlier.get("actors") or [] if isinstance(r, dict)}
        for sid in sorted(pending):
            if sid in step:
                standing[sid] = step[sid]
                pending.discard(sid)
            elif not (sid in rows and row_pending(rows[sid])):
                pending.discard(sid)  # a real answer (or no seat) in this wave ended the obligation
        wave = earlier
    return standing
