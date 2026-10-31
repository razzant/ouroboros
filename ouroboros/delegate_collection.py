"""Body-owned observation and mechanical collection of exact delegated runs.

The author still owns questions, consumption and disposition. This owner stores
execution facts, full source bytes, captures and supplied money independently of
that author's attention. Observers retain output and money; the existing maintenance
child prepares heavy captures. The server rechecks custody before publishing them.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor

from ouroboros import delegate_custody as c
from ouroboros import delegate_custody_current as current
from ouroboros.delegate_custody_current import collection_pending
from ouroboros.delegate_capture_attempt import (collection_due, record_attempt, GENERATION, failure_retry, output_owed,
                                                capture_required, legacy_cap)
from ouroboros.delegate_directory import is_in_place_run
from ouroboros.delegate_named_evidence import saved_named_evidence
from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS

log = logging.getLogger(__name__)


def _shape(entry):
    from ouroboros.subagents import DelegatedRunShape
    from ouroboros.delegate_registration_policy import INVESTIGATION_AUTHORITY
    return DelegatedRunShape(access=entry.access, mode=entry.mode, isolation=entry.isolation,
                             delegated=entry.delegated,
                             investigation=entry.authority_source == INVESTIGATION_AUTHORITY)


def _retain(root, entry, value, suffix):
    from ouroboros.artifacts import store_actor_source_bytes
    ref = store_actor_source_bytes(root, entry.task_id, category="tool_results",
        source_id=f"delegated-{suffix}-{entry.run_id}",
        data=json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8"), extension="json")
    return {**ref, "task_id": entry.task_id}


def _read_source(root, entry, ref):
    from ouroboros.artifacts import read_actor_source_bytes
    return json.loads(read_actor_source_bytes(root, str(ref.get("task_id") or entry.task_id), ref))


def observe_run(root, entry, detail: dict) -> bool:
    """Publish exact execution/question facts, never collection, control or consumption."""
    summary = c.summary_of(detail)
    if c.names_another_run(detail, entry.run_id):  # a run held by its job id is still this run
        raise ValueError("execution observation belongs to another run")
    state = str(summary.get("state") or "")
    seq = detail.get("lastSeq", 0)
    if not isinstance(seq, int) or isinstance(seq, bool):
        raise ValueError("execution observation has no valid sequence")
    fresh = current.run(root, entry.run_id)
    if fresh is None or fresh.invocation_id != entry.invocation_id or fresh.task_id != entry.task_id:
        return False
    if seq < fresh.observation_seq:
        return False
    if state in c.TERMINAL_STATES:
        if fresh.terminal_state and fresh.terminal_state != state:
            return False
        source = _retain(root, fresh, detail, "terminal")
        if source.get("sha256") == fresh.terminal_source.get("sha256"):
            return False  # Retained revisions are immutable; changed source earns new bytes.
        return c.emit(root, c.TERMINAL_OBSERVED, {
            "run_id": fresh.run_id, "task_id": fresh.task_id, "root_task_id": fresh.root_task_id,
            "parent_task_id": fresh.parent_task_id, "invocation_id": fresh.invocation_id,
            "route": fresh.route_id, "source": fresh.source, "project_id": fresh.project_id,
            "state": state, "reason": str((summary.get("outcomeFacts") or {}).get("reason") or ""),
            "observation_seq": seq, "source_ref": source})
    if fresh.terminal_state:
        return False
    from ouroboros.gateways.claudexor import pending_interactions
    pending = pending_interactions(detail, strict=True)
    ids = [str(item["interaction_id"]) for item in pending]
    if not pending and not fresh.pending_interaction_ids:
        return False
    envelope = {"run_id": fresh.run_id, "pending_interactions": pending}
    source = _retain(root, fresh, envelope, "interactions")
    if source.get("sha256") == fresh.interaction_source.get("sha256"):
        return False
    return c.emit(root, c.INTERACTIONS_OBSERVED, {
        "run_id": fresh.run_id, "task_id": fresh.task_id, "observation_seq": seq,
        "source_ref": source, "interaction_ids": ids})


def _collection_failure(entry, step, exc, failures) -> bool:
    """A changed physical failure is loud; unchanged retries keep the same known debt."""
    signature = (entry.terminal_source.get("sha256"), type(exc).__name__,
                 getattr(exc, "code", None), getattr(exc, "errno", None), getattr(exc, "reason", None), str(exc))
    changed = failures is None or failures.get(step) != signature
    if failures is not None:
        failures[step] = signature
    log.log(logging.WARNING if changed else logging.DEBUG,
            "Delegated %s remains owed for %s: %s", step, entry.run_id, exc, exc_info=True)
    return changed


def record_terminal_money(root, entry, detail: dict, *, failures=None) -> dict:
    """The old exact session accounting, without project retirement or output ACK."""
    if entry.settled:
        return {"settled": True, "ledger_recorded": entry.ledger_recorded, "retried": False}
    from ouroboros.delegate_output import engine_locator
    from ouroboros.gateways.claudexor import final_attempt_facts, run_failure_cause
    from ouroboros.usage_accounting import record_subscription_session

    summary = c.summary_of(detail)
    if not c.is_terminal(detail):
        raise ValueError("cannot record terminal money for a live execution")
    observed = final_attempt_facts(detail, entry.run_id)
    from ouroboros.delegate_custody_usage import disclosed_cost
    cash = disclosed_cost(summary, attempt_execution=detail.get("attemptExecution"))
    spend, estimated = cash["cost_usd"], cash["cost_estimated"]
    profile, access = observed.get("profile_id", ""), str(summary.get("effectiveAccess") or "")
    if not entry.ledger_recorded:
        try:
            from ouroboros.install_identity import require_owned_root

            require_owned_root(entry.ledger_root or root)  # the usage store is not behind ``open_store``'s check
            attempt_id = record_subscription_session(entry.run_id, drive_root=pathlib.Path(entry.ledger_root or root),
                route=entry.route_id, model=observed.get("model", ""), task_id=entry.task_id,
                root_task_id=entry.root_task_id, parent_task_id=entry.parent_task_id,
                category=entry.category, source=entry.source,
                prompt_tokens=c.disclosed_tokens(summary.get("inputTokens")),
                completion_tokens=c.disclosed_tokens(summary.get("outputTokens")),
                cached_tokens=c.disclosed_tokens(summary.get("cachedInputTokens")),
                input_token_usage=summary.get("inputTokenUsage"), attempt_execution=detail.get("attemptExecution"),
                effort_resolution=observed.get("effort_resolution"), spend_usd=spend, spend_estimated=estimated,
                credential_profile_id=profile, access_profile=access,
                **{key: getattr(entry, key) for key in REVIEW_ATTRIBUTION_KEYS})
        except Exception as exc:
            if _collection_failure(entry, "money", exc, failures):
                c.emit(root, c.SETTLE_FAILED, {"run_id": entry.run_id, "task_id": entry.task_id,
                                             "route": entry.route_id, "step": "ledger_write"})
        else:
            if failures is not None:
                failures.pop("money", None)
            entry.ledger_recorded = True
            c.emit(root, c.LEDGER_RECORDED, {"run_id": entry.run_id, "task_id": entry.task_id,
                "root_task_id": entry.root_task_id, "parent_task_id": entry.parent_task_id, "route": entry.route_id})
            if not entry.review_owned:
                _emit_session_usage(root, entry, summary, observed, attempt_id)
    if entry.ledger_recorded:
        failure = summary.get("failure") if isinstance(summary.get("failure"), dict) else {}
        facts = summary.get("outcomeFacts") if isinstance(summary.get("outcomeFacts"), dict) else {}
        failed = {} if str(summary.get("state") or "") in c.SUCCEEDED_STATES else {
            "requested_model": entry.model, "failure_code": str(failure.get("code") or ""),
            "reported_cause": run_failure_cause(failure), "outcome_reason": str(facts.get("reason") or "")}
        entry.settled = c.emit(root, c.SETTLED, {
            "run_id": entry.run_id, "task_id": entry.task_id, "root_task_id": entry.root_task_id,
            "parent_task_id": entry.parent_task_id, "route": entry.route_id,
            "source": entry.source, "category": entry.category, "model": observed.get("model", ""),
            "observed_attempt": observed, "state": str(summary.get("state") or ""), **failed,
            "preserve_containment": True,
            "cost_usd": spend, "cost_final": cash["cost_final"],
            "spend_disclosed": spend is not None, "spend_estimated": estimated, "cost_partial": cash["cost_partial"],
            "credential_profile_id": profile, "access_profile": access, "engine_locator": engine_locator(detail)})
        if entry.settled:
            entry.engine_locator = engine_locator(detail)
            try:
                from ouroboros.subagent_history import record_session_execution
                record_session_execution(root, entry, detail, observed)
            except Exception:
                log.debug("Session history unavailable", exc_info=True)
    return {"settled": entry.settled, "ledger_recorded": entry.ledger_recorded, "retried": True}


def _emit_session_usage(root, entry, summary, observed, attempt_id):
    """Best-effort telemetry after money; a missing event never repeats inference.

    The custody event already represents unknown/legacy sessions. Only a valid
    native split adds token facts. Recollection can repeat this row; consumers
    key it by run, while the ledger remains the sole monetary authority.
    """
    from ouroboros.review_session_usage import session_usage_facts
    from ouroboros.contracts.record_contract import RECORD_CONTRACT_VERSION
    from ouroboros.event_usage import event_token_counts
    from ouroboros.utils import append_jsonl, utc_now_iso

    facts = session_usage_facts(entry.run_id, requested_route=entry.route_id,
                                observed_attempt=observed, summary=summary)
    if "input_token_usage" not in facts:
        return
    try:
        append_jsonl(c.event_log_path(root), {
            "type": "llm_usage", "ts": utc_now_iso(), "record_contract_version": RECORD_CONTRACT_VERSION,
            "task_id": entry.task_id, "root_task_id": entry.root_task_id or entry.task_id,
            "parent_task_id": entry.parent_task_id, "delegation_role": "subagent" if entry.parent_task_id else "",
            "source": entry.source, "category": entry.category, "provider": "claudexor", "api_key_type": "claudexor",
            "model": "", "claudexor": facts["claudexor"], "input_token_usage": facts["input_token_usage"],
            **event_token_counts(facts),
            "accounting_authority": "physical_attempt_ledger", "ledger_attempt_ids": [attempt_id],
        })
    except Exception:
        log.warning("Delegated usage event unavailable for %s", entry.run_id, exc_info=True)


def _retain_output(root, entry, detail, gateway, staging_task_id="", output_authority=None):
    from ouroboros.delegate_output import retain_terminal_bytes, _resolve_full_primary_output
    from ouroboros.tools.delegate_terminal_evidence import resolved_terminal_payload

    full = retained_payload(root, entry, require_complete=False)
    if full is None:
        # A legacy row has no captured grant. Its existing foreground owner
        # supplies its known shape; body-only work keeps the recorded unknown.
        authority = _shape(entry) if entry.access or output_authority is None else output_authority
        full, complete, disclosure = resolved_terminal_payload(entry.run_id, detail, authority, entry, gateway)
    else:
        # The same observed terminal already has an exact presentation source.
        # Retrying its missing output does not re-derive its grant/containment.
        full["primary_output"], complete, disclosure = _resolve_full_primary_output(
            gateway, entry.run_id, full.get("primary_output"))
        full.pop("primary_output_full", None)
    if disclosure is not None:
        full["primary_output_full"] = disclosure
    text = json.dumps(full, ensure_ascii=False, indent=2)
    artifact = retain_terminal_bytes(root, staging_task_id or entry.task_id, entry.run_id, text, text.encode("utf-8"))
    if artifact is None:
        return {"retry": "transient"}
    retry = "" if complete else (disclosure or {}).get("retry", "changed_input")
    if (entry.output_sha == artifact["sha256"] and entry.output_source == artifact["source"]
            and entry.output_complete == complete
            and entry.output_terminal_sha == entry.terminal_source.get("sha256")):
        return {"artifact": artifact, "complete": complete, "published": True, "retry": retry}
    published = c.emit(root, c.OUTPUT_SPILLED, {"run_id": entry.run_id, "task_id": entry.task_id,
        "artifact": artifact["source"]["path"], "source": artifact["source"], "bytes": artifact["bytes"],
        "sha256": artifact["sha256"], "terminal_sha": entry.terminal_source.get("sha256", ""),
        "total_chars": len(text), "staged": True, "full_content": complete})
    return {"artifact": artifact, "complete": complete, "published": published,
            "retry": retry if published else "transient"}


def collect_terminal(root, run_id: str, *, gateway=None, detail: dict | None = None,
                     allow_review: bool = False, staging_task_id: str = "", stopped=None, failures=None,
                     output_authority=None, recapture=False, generation=GENERATION, background=False,
                     output_retry=False, gateway_factory=None) -> dict:
    """One result-owner entry for foreground collection and the body, including recovered debt.

    A contending caller reads committed facts; it never times out an integration
    or evicts a live collector. No queue/root/money lock is held at this entry.
    ``output_retry`` is an explicit wait: owed output and money retry even when this generation's
    attempt already ran; capture and named evidence still follow ``collection_due``.
    """
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock
    from ouroboros.delegate_continuation import disposition_refusal
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from ouroboros.tools.subagent_integration_delegated import retain_run_evidence
    from ouroboros import delegate_capture_job

    root = pathlib.Path(root)
    if stopped is not None and stopped():
        return {"status": "pending", "reason": "runtime_closing"}
    entry = current.run(root, run_id)
    if entry is None or (entry.review_owned and not allow_review):
        return {"status": "unavailable", "reason": "collection_owner_mismatch"}
    if detail is not None:
        observe_run(root, entry, detail)
    entry = current.run(root, run_id)
    if entry is None or not entry.terminal_source:
        return {"status": "pending", "reason": "terminal_not_observed"}
    lock = c.disposition_lock_path(root, entry)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = acquire_exclusive_file_lock(lock, timeout_sec=0, owner_aware_stale=True)
    if fd is None:
        return {"status": "pending", "reason": "collection_in_progress"}
    capture = None
    transient = False
    retention = {}
    errors = []
    owned_gateway = None
    try:
        entry = current.run(root, run_id)
        if entry is None:
            return {"status": "unavailable", "reason": "custody_missing"}
        if entry.named_evidence:
            saved_named_evidence(root, entry)  # addressed validation; never called by tick
        due = allow_review or collection_due(entry, generation=generation, recapture=recapture)
        if not due and not (output_retry and (output_owed(entry) or not entry.settled)):
            return {"status": "pending" if collection_pending(entry) else "collected",
                    "workspace_capture": capture_projection(root, entry), "errors": []}
        output_attempt = entry.collection_attempt.get("output") or {}
        retry_output = output_owed(entry) and (output_retry or recapture or not output_attempt
            or output_attempt.get("terminal_sha") != entry.terminal_source.get("sha256")
            or (output_attempt.get("retry") == "transient" and output_attempt.get("generation") != generation))
        if (gateway is None and gateway_factory is not None
                and (retry_output or (not background and capture_required(entry)))):
            try:
                owned_gateway = gateway = gateway_factory()
            except Exception:
                pass  # Retained output, local capture and money do not require transport.
        retry_capture = recapture or legacy_cap(entry) or (entry.collection_attempt.get("generation") != generation
            and (entry.named_evidence.get("retry") == "transient" or entry.capture_job.get("retry") == "transient"))
        if (background and retry_output and gateway is not None and entry.collection_attempt
                and entry.collection_attempt.get("generation") != generation):
            # One new body may refresh owed output. Money alone never needs engine I/O.
            from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC
            try:
                observe_run(root, entry, gateway.get_run(run_id, timeout_sec=SHORT_POLL_TIMEOUT_SEC))
            except Exception as exc:
                _collection_failure(entry, "observation", exc, failures)
            entry = current.run(root, run_id)
        try:
            detail = _read_source(root, entry, entry.terminal_source)
        except (OSError, ValueError, RuntimeError) as exc:
            _collection_failure(entry, "terminal_source", exc, failures)
            if due:
                record_attempt(root, entry, generation=generation, retry=failure_retry(exc),
                               errors=[f"terminal_source:{type(exc).__name__}"])
            return {"status": "pending", "reason": "terminal_source_unavailable"}
        if retry_output:
            try:
                retention = _retain_output(root, entry, detail, gateway, staging_task_id, output_authority)
            except Exception as exc:
                errors.append(f"output:{type(exc).__name__}")
                retention = {"retry": failure_retry(exc)}
                _collection_failure(entry, "output", exc, failures)
            else:
                if failures is not None:
                    failures.pop("output", None)
            output_attempt = {"terminal_sha": entry.terminal_source.get("sha256"),
                              "generation": generation, "retry": retention.get("retry", "")}
        transient |= output_attempt.get("retry") == "transient"
        entry = current.run(root, run_id)
        closing = stopped is not None and stopped()
        money = ({"settled": entry.settled, "ledger_recorded": entry.ledger_recorded}
                 if closing else record_terminal_money(root, entry, detail, failures=failures))
        entry = current.run(root, run_id)
        # This lock is shared with continue/apply. A possibly-started successor
        # owns the mutable snapshot, even before it has a bound run id.
        if (due and entry is not None and not (stopped is not None and stopped())
                and not disposition_refusal(root, entry)):
            try:
                if background and (entry.execution_root or entry.resource_ref.get("workspace_kind") == "directory") \
                        and not is_in_place_run(entry):  # an in-place run's capture is its fixed projection
                    capture = delegate_capture_job.request_or_result(root, entry, generation=generation, recapture=retry_capture)
                    if capture.get("status") != "pending":
                        if not delegate_capture_job.accept_result(root, entry, capture):
                            capture = {"status": "pending", "note": "The capture binding changed before publication."}
                else:
                    capture = capture_terminal_patch_for_drive(root, entry, gateway=gateway, retain_investigation=False)
                if isinstance(capture, dict) and capture.get("status") == "failed":  # a typed failure stays owed
                    raise RuntimeError(str(capture.get("note") or "patch capture failed"))
            except Exception as exc:
                errors.append(f"capture:{type(exc).__name__}")
                transient |= failure_retry(exc) == "transient" or (capture or {}).get("retry") == "transient"
                _collection_failure(entry, "capture", exc, failures)
            else:
                if failures is not None:
                    failures.pop("capture", None)
        entry = current.run(root, run_id)
        if entry is not None:
            closing = stopped is not None and stopped()
            # Named evidence and evidence-only cleanup come AFTER full output
            # retention. Mutating results remain the author's decision.
            if not closing and capture and entry.output_complete and entry.output_source:
                try:
                    capture = retain_run_evidence(root, entry, capture, gateway, recapture=retry_capture,
                                                  named=(capture.get("named_evidence") or {}) if background else None)
                except Exception as exc:
                    errors.append(f"named_evidence:{type(exc).__name__}")
                    transient |= failure_retry(exc) == "transient"
                    _collection_failure(entry, "named_evidence", exc, failures)
            if not closing and entry.project_persistent and entry.project_owned:
                c.retire_project(root, gateway, entry)  # records kept, never deletes the project
            fresh = current.run(root, run_id)
            named = fresh.named_evidence
            retry = ("transient" if closing or transient or not fresh.settled
                     or named.get("retry") == "transient" else "changed_input" if output_owed(fresh) else "")
            if due or retry_output:
                record_attempt(root, fresh, generation=generation, retry=retry, errors=errors, output=output_attempt)
            shown = capture
            if capture and not is_in_place_run(fresh) and fresh.resource_ref.get("workspace_kind") != "directory":
                # With its recorded decision, as capture_projection shows it. A decided run's capture still owed
                # to maintenance has read nothing yet: its saved capture is shown, with readers for the task this
                # delivery is staged for, and ``capture`` stays pending for the wake below (#1651).
                shown = (capture_projection(root, fresh, reader_task_id=staging_task_id)
                         if fresh.patch_result_lost or (capture.get("status") == "pending"
                                                       and fresh.patch_disposed in ("applied", "rejected"))
                         else _decided(fresh, capture))
            return {"status": "collected" if not collection_pending(fresh) else "pending",
                    "settlement": money, "workspace_capture": shown, "output_retention": retention, "errors": errors}
        return {"status": "unavailable", "reason": "custody_missing"}
    finally:
        release_exclusive_file_lock(lock, fd)
        if background and capture and capture.get("status") == "pending":
            from ouroboros.maintenance import child
            child.request()  # Its zero-wait lookup must follow our final store publication.
        if owned_gateway is not None:
            owned_gateway.close()


def retained_payload(root, entry, *, require_complete: bool = True) -> dict | None:
    """Read verified canonical body bytes, without claiming a reader consumed them."""
    if (not entry.output_source or (require_complete and not entry.output_complete)
            or entry.terminal_source and entry.output_terminal_sha != entry.terminal_source.get("sha256")):
        return None
    try:
        return _read_source(root, entry, entry.output_source)
    except Exception:
        log.warning("Retained output unavailable for %s", entry.run_id, exc_info=True)
        return None


def _decided(entry, block: dict, *, unread=False) -> dict:
    """A recorded apply or reject, with who recorded it (#1651): the current custody fact, never a
    chain replay; a row reduced before the disposer was projected reads ``not_recorded``. Nothing
    is owed after it, so the undecided capture's note no longer applies. A loss over a failed capture
    that saved file changes says they stay (its block counts them); ``unread`` (a saved manifest that
    exists but cannot be read) leaves that count unknown, so no loss note then says nothing was kept."""
    if entry.patch_disposed not in ("applied", "rejected"):
        return block
    by = entry.patch_disposed_by
    who = (f"its starter {by}" if by == entry.task_id else f"task {by}, not its starter {entry.task_id}") \
        if by else "a task not recorded"
    kept = int(block.get("file_output_count") or 0)
    what = (entry.patch_disposed if not entry.patch_result_lost else
            f"rejected as lost: its private copy was gone before a patch was captured; the {kept} file change"
            f"{'s' * (kept != 1)} its failed capture saved {'is' if kept == 1 else 'are'} kept and never applied" if kept else
            "rejected as lost: its private copy was gone before a patch was captured; its saved capture cannot be "
            "read, so whether file changes it saved are kept is unknown, and nothing was applied" if unread else
            "rejected as lost: its private copy was gone and no saved or pinned result named it, so nothing "
            "was kept or applied")
    return {**block, "disposition": entry.patch_disposed, "disposed_by": by or "not_recorded",
            **({"result_lost": True} if entry.patch_result_lost else {}),
            **({"file_output_count": kept} if entry.patch_result_lost and not unread else {}),
            "note": f"Decided by {who}: {what}. No apply or reject is owed."}


def capture_projection(root, entry, *, reader_task_id="") -> dict | None:
    """Existing captured manifest or an honest pending fact; never triggers capture.

    A Git run's saved named-evidence summary rides along, so saved-result and
    retry-successor delivery keep its status and reader. A Git or skill run's recorded
    decision rides with its disposer; an engine directory keeps its own block."""
    from ouroboros.delegate_directory import in_place_block

    if is_in_place_run(entry):
        return in_place_block(entry)
    directory = entry.resource_ref.get("workspace_kind") == "directory"
    if not entry.execution_root and not directory:
        return None
    cap_dir = c.delegated_capture_dir(root, entry.task_id, entry.run_id if directory else c.capture_key(entry))
    try:
        manifest = json.loads((cap_dir / "workspace_patch.json").read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):  # well-formed JSON that is no manifest reads as unreadable
            raise ValueError("a capture manifest is a JSON object")
    except (OSError, ValueError) as exc:
        from ouroboros.delegate_registration_policy import evidence_only

        if evidence_only(entry):  # no author decision is owed
            return {"status": "pending", "note": "Body capture is still owed; evidence only, never applied.",
                    "patch_disposition": entry.patch_disposed or "not_applicable"}
        if not directory and entry.patch_disposed in ("applied", "rejected"):  # decided: nothing is owed
            # Only an absent manifest shows that no capture saved anything.
            return _decided(entry, {"status": "missing"}, unread=not isinstance(exc, FileNotFoundError))
        return {"status": "pending", "note": "Body capture is still owed; no patch was applied. integrate_delegated_patch "
                "prepares a finished run's capture itself and returns it for inspection first.",
                "patch_disposition": entry.patch_disposed or "pending"}
    other_reader = bool(reader_task_id and reader_task_id != entry.task_id)
    if directory:
        from ouroboros.delegate_directory import directory_capture_block
        block = directory_capture_block(root, entry, manifest)
    else:
        from ouroboros.tools.delegate_integration import _capture_block
        block = _capture_block(entry, cap_dir, manifest)
        summary, named = saved_named_evidence(root, entry)
        if named:  # another task reads it by absolute artifact path, as the patch and manifest below
            block["named_evidence"] = ({**named, "summary_read": {"root": "artifact_store", "path": str(summary)}}
                                       if other_reader and named.get("summary_read") else named)
    if other_reader:
        for kind in ("patch", "manifest"):
            if block.get(f"{kind}_read") and block.get(f"{kind}_artifact"):
                block[f"{kind}_read"] = {"root": "artifact_store", "path": block[f"{kind}_artifact"]}
    return block if directory else _decided(entry, block)


_RUNTIMES: dict[str, "BodyRuntime"] = {}
_RUNTIME_LOCK = threading.RLock()


class BodyRuntime:
    """One generation's owned physical jobs. No background job grants actor authority."""
    def __init__(self, root, *, stop_event=None, gateway_factory=None, max_workers=None, collection_ready=None):
        from ouroboros.config import get_max_workers
        from ouroboros.platform_layer import process_start_time

        self.root = pathlib.Path(root).resolve()
        self.external_stop = stop_event
        self.gateway_factory = gateway_factory
        self.collection_ready = collection_ready or (lambda: True)
        self.stop = threading.Event()
        self.changed = threading.Event()
        self.lock = threading.RLock()
        self.runtime_id = uuid.uuid4().hex
        self.birth = process_start_time(os.getpid())
        width = max(2, int(max_workers if max_workers is not None else get_max_workers()))
        self.observers = ThreadPoolExecutor(max_workers=width, thread_name_prefix="run-observe")
        self.reads: dict[str, Future] = {}
        self.collections: dict[str, Future] = {}
        self.requested: set[str] = set()
        self.collection_failures: dict[str, dict] = {}
        # Each held run's latest dated execution fact, as the last observation read it.
        self.observed: dict[str, dict] = {}
        self.observation_gaps: dict[str, tuple] = {}
        # The engine's attention token each held run was last fetched at: echoed as ``seen`` so an
        # unchanged question or terminal does not wake the pass again; set only AFTER the fetch.
        self.seen: dict[str, str] = {}
        self._watched: set[str] = set()
        # Whether the observed engine serves the compact wait, learned from its catalog on the
        # first read; until then (and on engines without it) every held run is read as before.
        self._compact_wait = False
        self._observing = None
        self._connecting = None
        self.thread = threading.Thread(target=self._drive, name="delegated-body", daemon=True)
        try:
            self.thread.start()
        except BaseException:
            self.stop.set()
            self.observers.shutdown(wait=False, cancel_futures=True)
            raise

    @property
    def active(self):
        return self.thread.is_alive() and not self.stop.is_set() and not (
            self.external_stop is not None and self.external_stop.is_set())

    def identity(self):
        return {"active": self.active, "runtime_id": self.runtime_id,
                "pid": os.getpid(), "process_birth": self.birth}

    def ensure_runs(self, pairs, *, task_id: str = "") -> bool:
        if not self.active:
            return False
        ids = []
        for rid, iid in pairs:
            entry = current.run(self.root, str(rid))
            if (entry is None or entry.review_owned or entry.invocation_id != str(iid)
                    or (task_id and entry.task_id != task_id)):
                return False
            ids.append(entry.run_id)
        with self.lock:
            if not self.active:
                return False
            self.requested.update(ids)
            self.changed.set()
        return True

    def _gateway(self):
        if self.gateway_factory is not None:
            return self.gateway_factory()
        from ouroboros.claudexor_daemon import read_owned_gateway
        return read_owned_gateway()

    def _observer_gateway(self):
        """One observation client; discovery and handshake never hold the runtime lock."""
        from ouroboros.gateways.claudexor_metadata import endpoint_generation

        with self.lock:
            gateway = self._observing
        endpoint = getattr(gateway, "_endpoint", None)
        if endpoint is not None and endpoint.descriptor_path and endpoint_generation(endpoint) is None:
            self._drop_observer_gateway(gateway)
        with self.lock:
            if self._observing is not None:
                return self._observing
            owner = self._connecting is None
            if owner:
                self._connecting = Future()
            pending = self._connecting
        if owner:
            try:
                gateway = self._gateway()
                with self.lock:
                    self._observing = gateway
                pending.set_result(gateway)
            except BaseException as exc:
                pending.set_exception(exc)
            finally:
                with self.lock:
                    self._connecting = None
        return pending.result()

    def _observation_key(self, entry):
        from ouroboros.gateways.claudexor_metadata import endpoint_generation

        gateway = self._observing
        endpoint = getattr(gateway, "_endpoint", None)
        generation = endpoint_generation(endpoint) if endpoint is not None else id(gateway)
        return generation, entry.invocation_id, entry.daemon_store_id

    def _drop_observer_gateway(self, gateway=None):
        with self.lock:
            if gateway is not None and self._observing is not gateway:
                return
            gateway, self._observing = self._observing, None
        if gateway is not None:
            gateway.close()

    _WATCH, _PRIME = "*watch", "*prime"  # the compact passes, keyed beside the per-run reads

    def _observe(self, entry):
        """One run's full detail: the addressed read a wake, a fallback beat or a problem row earns.
        The capability question (does this engine serve the compact wait?) is asked beside it, in
        its own guard and off the lock: a question that cannot be asked this beat (a catalog hole)
        is "not supported this beat", nothing is cached, and the old read runs regardless."""
        from ouroboros.gateways import claudexor_observation as observation
        from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC, ClaudexorUnavailable
        gateway = self._observer_gateway()
        try:
            supported = observation.wait_many_supported(gateway, timeout_sec=SHORT_POLL_TIMEOUT_SEC)
        except ClaudexorUnavailable:
            supported = False
        try:
            detail = gateway.get_run(entry.run_id, timeout_sec=SHORT_POLL_TIMEOUT_SEC)
        except BaseException as exc:
            if self._absorb_refusal(entry, gateway, exc):
                return
            raise
        with self.lock:
            self._compact_wait = supported
        self._absorb_detail(entry, gateway, detail)

    def _absorb_refusal(self, entry, gateway, exc) -> bool:
        """Record a refused read; True when the refusal settled the run (an owning store's absence,
        or a terminal fact retention cannot erase), so the reader has nothing left to raise."""
        self._remember(entry, None, error=exc)
        entry = current.run(self.root, entry.run_id) or entry
        if c.daemon_says_absent(exc) and entry.terminal_source:
            return True  # Retention loss cannot erase a terminal fact, including an in-flight publication.
        if c.daemon_says_absent(exc, gateway=gateway, run=entry):
            c.close_absent_run(self.root, gateway, entry, "body_observation_absent")
            return True
        if c.daemon_says_absent(exc):
            c.record_containment_fault(self.root, entry, "run_absence_identity_unproven",
                "The daemon returned 404, but ownership by this store is unproven; run outcome remains unknown.")
            with self.lock:
                self.observation_gaps[entry.run_id] = self._observation_key(entry)
        if not getattr(exc, "status_code", 0) and not getattr(exc, "observation_answered", False):
            self._drop_observer_gateway(gateway)
        return False

    def _absorb_detail(self, entry, gateway, detail):
        with self.lock:
            self.observation_gaps.pop(entry.run_id, None)
        # A late valid source fact remains evidence; it grants no wait.
        observe_run(self.root, entry, detail)
        c.bind_daemon_store(self.root, gateway, entry)
        if entry.containment_fault.get("reason") == "run_absence_identity_unproven":
            c.resolve_containment_fault(self.root, entry, "run_observed",
                                        expected_reason="run_absence_identity_unproven")
        self._remember(entry, detail)

    def _watch(self, entries, *, hold=True):
        """One pass over the watched runs through the observation adapter.

        An engine with the compact wait holds the call (its own default window; ``hold=False``
        reads the present state at once, for runs held since the pass in flight began) and
        names the runs whose addressed detail is now due: a question or a terminal, not yet
        fetched under the token echoed as ``seen``. The detail is read, its facts published,
        and only then the token acknowledged, so a question is never acknowledged unfetched.
        An engine without the wait keeps today's beat: one detail read per run per tick.
        The hold is the observer's own tick window, not the engine's: closing the client does
        not wake a receive blocked in a hold on every platform (Linux ``close(2)`` leaves the
        blocked thread to the socket's own end), so a pass ends within the tick plus the
        transport bound (``SHORT_POLL_TIMEOUT_SEC``: about eight seconds), never the engine's window.
        """
        from ouroboros.delegate_supervision import _TICK_SEC
        from ouroboros.gateways import claudexor_observation as observation
        from ouroboros.gateways.claudexor import SHORT_POLL_TIMEOUT_SEC, ClaudexorUnavailable
        gateway = self._observer_gateway()
        with self.lock:
            targets = [(entry.run_id, self.seen.get(entry.run_id)) for entry in entries]
        try:
            supported = observation.wait_many_supported(gateway, timeout_sec=SHORT_POLL_TIMEOUT_SEC)
            wake = observation.wait_attention(gateway, targets, wait_sec=_TICK_SEC if hold else 0.0,
                                              timeout_sec=SHORT_POLL_TIMEOUT_SEC) if supported else None
        except ClaudexorUnavailable as exc:
            if not observation.is_observation_hole(exc):
                # A refusal of the pass itself, or an answer outside the contract (never a hole or
                # the engine's own retryable 503, which this pass leaves unanswered): this engine's
                # compact wait is unusable, so every run is read as before (logged once: the
                # engine identity is remembered, nothing re-asks it).
                observation.forget_support(gateway)
                with self.lock:
                    self._compact_wait = False
                log.warning("Body compact wait unusable on this engine (%s); each run's detail is read as before", exc)
            self._unanswered(entries, exc, gateway)
            return
        with self.lock:  # the route vanished under us: this pass read details, the next ticks read as before
            self._compact_wait = supported and wake.source != "detail"
        if not supported:
            return  # the next tick reads each run as before
        if wake.hole is not None:
            self._unanswered(entries, wake.hole, gateway)
            return
        for index, (entry, row) in enumerate(zip(entries, wake.rows)):
            if not self.active:
                return
            if row.detail is not None:  # the route vanished under us: this pass read details, as before
                self._absorb_detail(entry, gateway, row.detail)
                continue
            if row.problem is not None or row.error is not None:
                # The engine could not place the run in this pass; only the addressed read, with
                # its store identity, may say the run is absent (``daemon_says_absent``). Once it
                # answered, the problem is acknowledged so the next pass holds again instead of
                # answering the same problem at once.
                try:
                    self._observe(entry)
                except Exception:
                    log.debug("Body observation of %s remains owed after a problem row", entry.run_id, exc_info=True)
                    continue
                if row.problem is not None:
                    with self.lock:
                        self.seen[entry.run_id] = observation.ACKNOWLEDGED_PROBLEM
                continue
            if not wake.needs_attention(index):
                if row.pending is False and entry.pending_interaction_ids:
                    # Answered since this observer last read the run: one addressed read clears
                    # the question from custody. Every other quiet row stays read-free.
                    try:
                        self._observe(entry)
                    except Exception:
                        log.debug("Body observation of %s remains owed after its question was answered",
                                  entry.run_id, exc_info=True)
                    continue
                self._remember(entry, None, row=row)
                c.bind_daemon_store(self.root, gateway, entry)
                continue
            try:
                detail = gateway.get_run(entry.run_id, timeout_sec=SHORT_POLL_TIMEOUT_SEC)
            except BaseException as exc:
                self._absorb_refusal(entry, gateway, exc)
                continue
            self._absorb_detail(entry, gateway, detail)
            with self.lock:
                self.seen[entry.run_id] = row.token

    def _unanswered(self, entries, hole, gateway):
        """Nothing was learned about these runs this pass: the hole is each run's execution fact,
        and a transport that carried no answer is dropped (the next pass opens a fresh one)."""
        for entry in entries:
            self._remember(entry, None, error=hole)
        if not getattr(hole, "status_code", 0) and not getattr(hole, "observation_answered", False):
            self._drop_observer_gateway(gateway)

    def _remember(self, entry, detail, *, error=None, row=None):
        """Keep this read's execution fact for readers that must not poll the engine themselves
        (``gateway/reply_preparation.py``), including a received HTTP refusal; ``row`` is the
        compact wait's answer for a run whose detail this pass did not need."""
        from ouroboros.delegate_supervision import run_activity

        summary = c.summary_of(detail) if detail is not None else {}
        if row is not None:
            summary = {"state": row.status or "", "waitingOnUser": bool(row.pending)}
        fact = {"run_id": entry.run_id, "task_id": entry.task_id, "invocation_id": entry.invocation_id,
                "answered": detail is not None or row is not None or bool(getattr(error, "status_code", 0)),
                "error_code": str(getattr(error, "code", "")), "http_status": getattr(error, "status_code", 0),
                "run_state": str(summary.get("state") or ""),
                "waiting_on_user": bool(summary.get("waitingOnUser") or (detail or {}).get("pendingInteractions")),
                "at": time.monotonic()}  # dated on this process's clock, like every in-process reader
        with self.lock:
            previous, self.observed[entry.run_id] = self.observed.get(entry.run_id), fact
        if run_activity(fact) is True and (previous is None or run_activity(previous) is not True):
            _announce_working(entry)  # first seen working too: a restarted host's idle cue learns of it

    def observation(self, run_id) -> dict:
        with self.lock:
            return dict(self.observed.get(str(run_id)) or {})

    def _collect(self, run_id, recapture=False):
        with self.lock:
            failures = self.collection_failures.setdefault(run_id, {})
        return collect_terminal(self.root, run_id, gateway_factory=self._gateway, stopped=lambda: not self.active,
                                failures=failures, recapture=recapture, generation=self.runtime_id, background=True)

    def tick(self):
        if not self.active:
            return
        entries = current.held_runs(self.root)
        with self.lock:
            for jobs in (self.reads, self.collections):
                for rid, future in list(jobs.items()):
                    if future.done():
                        if not future.cancelled() and future.exception() is not None:
                            log.debug("Body operation for %s remains owed: %s", rid, future.exception())
                        del jobs[rid]  # Completed history is not an in-flight registry.
            requested, self.requested = self.requested, set()
        for rid in requested - entries.keys():
            entry = current.run(self.root, rid)
            if entry is not None:
                entries[rid] = entry
        with self.lock:
            for rid in set(self.observed) - set(entries):
                del self.observed[rid]  # a run no longer held has no current execution fact
                self.observation_gaps.pop(rid, None)
                self.seen.pop(rid, None)
                self._watched.discard(rid)
        with self.lock:
            for rid in list(self.collection_failures):
                if rid not in self.collections and not collection_pending(entries.get(rid)):
                    self.collection_failures.pop(rid, None)
        watch = []
        with self.lock:
            compact = self._compact_wait
        for entry in entries.values():
            if entry.review_owned or not self.active:
                continue
            with self.lock:
                if not entry.terminal_source and (not entry.settled or entry.terminal_state):
                    if self.observation_gaps.get(entry.run_id) != self._observation_key(entry):
                        if compact:
                            watch.append(entry)
                        else:
                            previous = self.reads.get(entry.run_id)
                            if previous is None or previous.done():
                                self.reads[entry.run_id] = self.observers.submit(self._observe, entry)
                if self.collection_ready() and collection_due(entry, generation=self.runtime_id):
                    previous = self.collections.get(entry.run_id)
                    if previous is None or previous.done():
                        self.collections[entry.run_id] = self.observers.submit(self._collect, entry.run_id)
        with self.lock:
            pass_ = self.reads.get(self._WATCH)
            if watch and (pass_ is None or pass_.done()):
                self._watched = {entry.run_id for entry in watch}
                self.reads[self._WATCH] = self.observers.submit(self._watch, watch)
            elif watch:
                # A hold is in flight: a run held since it began gets its present state now,
                # not after the engine's window, so its first observation keeps today's beat.
                fresh = [entry for entry in watch if entry.run_id not in self._watched]
                prime = self.reads.get(self._PRIME)
                if fresh and (prime is None or prime.done()):
                    self._watched.update(entry.run_id for entry in fresh)
                    self.reads[self._PRIME] = self.observers.submit(self._watch, fresh, hold=False)

    def _drive(self):
        from ouroboros.delegate_supervision import _TICK_SEC
        try:
            while not self.stop.is_set() and not (self.external_stop is not None and self.external_stop.is_set()):
                try:
                    self.tick()
                except Exception:
                    log.warning("Body run observation unavailable; custody retained", exc_info=True)
                self.changed.wait(_TICK_SEC)
                self.changed.clear()
        finally:
            self.stop.set()
            # Actual jobs remain owned until their physical return; a caller of
            # close(wait=False) neither joins on the control loop nor forgets them.
            # The observer transport closes first; a compact pass still held ends within its own
            # tick plus the transport bound at the latest (``_watch``), so the join below is
            # bounded by those, never by the engine's window.
            self._drop_observer_gateway()
            self.observers.shutdown(wait=True, cancel_futures=True)
            with _RUNTIME_LOCK:
                if _RUNTIMES.get(str(self.root)) is self:
                    del _RUNTIMES[str(self.root)]

    def close(self, *, wait=False):
        self.stop.set()
        self.changed.set()
        if wait and threading.current_thread() is not self.thread:
            self.thread.join()


def _announce_working(entry) -> None:
    """A held run works now and did not at its last read (or had none): say so the way a
    resumed attempt does, since its task may be a sleeper whose Telegram typing had stopped
    (DESIGN §9). Best effort: a lost frame costs one missed cue."""
    from types import SimpleNamespace

    try:
        from ouroboros.model_sleep import announce_resumed
        from supervisor import queue as q
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.workers import get_event_q

        with q._queue_lock:
            rows = [meta.get("task") for meta in q.RUNNING.values() if isinstance(meta, dict)] + list(q.PENDING)
        task = next((row for row in rows if isinstance(row, dict) and str(row.get("id") or "") == entry.task_id), None)
        turn = get_direct_activity_registry().get(entry.task_id) if task is None else None
        if task is None and turn is None:
            return
        announce_resumed(SimpleNamespace(event_queue=get_event_q(create=False), task_id=entry.task_id,
                                         current_chat_id=task.get("chat_id") if task is not None else turn.chat_id))
    except Exception:
        log.debug("Working-again announce unavailable for %s", entry.run_id, exc_info=True)


def current_activity(root, task_id: str, run_id: str, invocation_id: str = ""):
    """Whether a held run works now by this observer's latest read: the same run, task and
    (when given) invocation, read within a few ticks; anything else is None (unknown)."""
    from ouroboros.delegate_supervision import _OBSERVATION_CURRENT_SEC, run_activity

    runtime = body_runtime(root)
    fact = runtime.observation(run_id) if runtime is not None else {}
    try:
        current = "at" in fact and time.monotonic() - float(fact["at"]) <= _OBSERVATION_CURRENT_SEC
    except (TypeError, ValueError):
        current = False
    if (not current or fact.get("task_id") != str(task_id)
            or (invocation_id and fact.get("invocation_id") != str(invocation_id))):
        return None
    return run_activity(fact)


def start_body_runtime(root, *, stop_event=None, gateway_factory=None, max_workers=None, collection_ready=None) -> BodyRuntime:
    key = str(pathlib.Path(root).resolve())
    with _RUNTIME_LOCK:
        existing = _RUNTIMES.get(key)
        if existing is not None:
            return existing  # A closing generation is not replaced while its jobs remain.
        runtime = BodyRuntime(key, stop_event=stop_event, gateway_factory=gateway_factory, max_workers=max_workers,
                              collection_ready=collection_ready)
        _RUNTIMES[key] = runtime
        return runtime


def body_runtime(root) -> BodyRuntime | None:
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(str(pathlib.Path(root).resolve()))
        return runtime if runtime is not None and runtime.active else None


def stop_body_runtime(root, *, wait=False) -> None:
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(str(pathlib.Path(root).resolve()))
    if runtime is not None:
        runtime.close(wait=wait)
