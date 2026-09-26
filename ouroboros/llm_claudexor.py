"""Caller-owned Claudexor model transport over physical-attempt accounting.
One engine operation rejoins after lost control; bytes enter private CAS before ACK. The live ``ModelTurnState`` belongs to
the caller, never stored assistant history: only a dispatched, durable result
updates it, while unknown/no-start/legacy silence preserves it. Requests priced
ahead of dispatch read the SAME slot; leaving this route clears it. The schema
floor and native-continuation repair: ARCHITECTURE §6 "The live turn slot".
No provider wait changes the task's deadline or Stop.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import replace
import hashlib
import json
import re
import logging
import threading
import time
from typing import Any

from ouroboros import config, context_fit
from ouroboros._usage_response import provider_cost_value
from ouroboros.anthropic_native_custody import scrub_native_custody
from ouroboros.claudexor_daemon import ensure_owned_gateway, owned_engine_version, read_owned_gateway
from ouroboros.deadline_utils import llm_transport_timeout_sec
from ouroboros.gateways.claudexor import (
    ClaudexorUnavailable, engine_at_least, model_failure_evidence_supported, _READ_TIMEOUT_SEC)
from ouroboros.llm_attempt import _attempt_request, _candidate_before_dispatch
from ouroboros.llm_substitution import (
    AccountRotation, SubstitutionBudget, substitution_fact, failed_account_preference,
    take_failed_account_preference)
from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
from ouroboros.model_wait import ModelWaitInterrupted, current_model_wait, prepared_call_scope
from ouroboros.observability import persist_call
from ouroboros.transport_custody import ProviderNotDispatched
from ouroboros.usage_accounting import (
    PhysicalAttemptPreparationFailed, current_physical_attempt_context, current_usage_scope,
    execute_physical_attempt, execute_physical_attempt_async, last_physical_attempt_capture)
from ouroboros.utils import append_jsonl, sanitize_tool_result_for_log, utc_now_iso

log = logging.getLogger(__name__)
def model_catalog(source: str, credential_profile_id: str | None = None, *,
                  requested_model: str | None = None, timeout_sec: float | None = None) -> dict:
    """Metadata-only transport; the capability evidence owner interprets the envelope."""
    gateway = read_owned_gateway()
    try:
        hint = {"requested_model": requested_model} if requested_model is not None else {}
        return gateway.list_source_models(source, credential_profile_id, **hint,
                                          **({"timeout_sec": timeout_sec} if timeout_sec is not None else {}))
    finally:
        gateway.close()


def model_sources(*, processing_view: bool = False) -> dict:
    """Expose declared model sources and their credential owners, without a routing table."""
    gateway = read_owned_gateway()
    try:
        if processing_view:
            try:
                from ouroboros.gateways.claudexor import account_catalog_supported
            except ImportError:
                pass  # An older host facade keeps its strict legacy request.
            else:
                if account_catalog_supported(gateway.operations(), "/v2/model-sources"):
                    return gateway.list_model_sources(view="accounts")
        return gateway.list_model_sources()
    finally:
        gateway.close()


def prepare_processing_target(target: dict) -> dict:
    """Capture the source's advertised transport preference before building bytes.

    No extra polling/cache or model eligibility inference: this reads the existing
    model-source owner once, only for explicit intent. A retained target is reused
    by every physical preparation of this logical call.
    """
    if not target.get("processing_preference") or "processing_preferences" in target:
        return target
    prepared = dict(target)
    prepared["processing_preferences"] = []
    try:
        sources = model_sources(processing_view=True)
    except ClaudexorUnavailable:
        return prepared
    source = next((row for row in sources.get("sources", [])
                   if isinstance(row, dict) and row.get("id") == target.get("source")), {})
    preferences = source.get("processingPreferences")
    if isinstance(preferences, list):
        prepared["processing_preferences"] = [value for value in preferences if isinstance(value, str)]
    return prepared


class ClaudexorModelError(RuntimeError):
    """A model-operation fact with its exact role, route and recovery identity."""

    def __init__(self, problem: dict, *, model_role: str = "", operation_id: str = "",
                 route: dict | None = None, unknown: bool = False):
        self.problem = copy.deepcopy(problem)
        self.code = "model_outcome_unknown" if unknown else str(problem.get("code") or "model_operation_failed")
        super().__init__(f"{self.code}: {problem.get('message') or 'Model operation did not complete'}")
        self.body = {"code": self.code} if unknown else self.problem
        context = problem.get("context") or {}
        # Generic engine wrappers retain their custody identity; existing
        # provider-fact readers use type for the more specific vendor refusal.
        vendor_code = context.get("vendorCode")
        self.type = (vendor_code.strip() if not unknown and self.code in {"provider_failed", "invalid_request"}
                     and isinstance(vendor_code, str) else "")
        self.status_code = 0 if unknown else int(context.get("httpStatus") or 0)
        self.reset_at = str(context.get("resetsAt") or "")
        self.retryable = False if unknown else problem.get("retryable") is True
        if self.code == "response_rejected":
            # Reconstructed process-boundary errors retain the same local
            # rejection: no unknown outcome or same-request wire repair.
            self.stream_rejected = self.stream_incomplete = True
            self.retryable = False
        self.model_role = model_role
        self.operation_id = operation_id
        self.route = copy.deepcopy(route or {})

    @property
    def display_message(self) -> str:
        """Show typed provider details without changing exception classification text."""
        context = self.problem.get("context") or {}
        fields = (("stage", "stage"), ("errorCode", "cause"))
        if self.code != "model_outcome_unknown":
            fields += (("vendorCode", "provider_code"), ("parameter", "parameter"))
        details = [
            f"{label}={value.strip()}"
            for key, label in fields
            if isinstance(value := context.get(key), str) and value.strip()
        ]
        # Details lead so the existing terminal preview can name the refusal.
        return sanitize_tool_result_for_log("; ".join([", ".join(details), str(self)]) if details else str(self))


class ClaudexorModelNotDispatched(ClaudexorModelError, ProviderNotDispatched):
    """Only a terminal engine receipt proving dispatch.state=not_started mints this."""


def presence_refusal_unstarted(error: Exception, round_idx: int) -> bool:
    """Positive no-generation receipt for ALL operations in one physical model call.

    Never fall back to the process-local last capture: it may belong to an
    earlier attempt. The caller separately accumulates this over fallback routes.
    """
    from ouroboros.usage_accounting import PhysicalAttemptCapture

    capture = getattr(error, "physical_attempt_capture", None)
    return (round_idx == 1 and isinstance(error, ClaudexorModelNotDispatched)
            and getattr(error, "presence_all_operations_not_started", False) is True
            and isinstance(capture, PhysicalAttemptCapture) and capture.state == "released")


def propagate_model_error(error: Exception) -> None:
    """Preserve control/resource waits and unknown custody across helper fallbacks.

    A confirmed ordinary provider refusal still belongs to the helper's existing
    retry or disclosed-unavailable path, just as it does for direct API calls.
    """
    from ouroboros.model_wait import ModelWaitInterrupted, model_wait_reason

    if isinstance(error, ModelWaitInterrupted):
        raise error
    if isinstance(error, ClaudexorModelError):
        capture = getattr(error, "physical_attempt_capture", None)
        if (error.code in {"model_outcome_unknown", "model_operation_interrupted"}
                or model_wait_reason(error)
                or getattr(capture, "state", None) in {"dispatched", "unresolved"}):
            raise error


def _usage(result: dict) -> tuple[dict, float | None, bool]:
    """Normalize explicit model usage; never run the generic body-error/free branch."""
    counters = result.get("usage") or {}
    cost_evidence = result.get("cost") or {}
    cash = provider_cost_value(cost_evidence.get("cashUsd"))
    knowledge = cost_evidence.get("knowledge")
    cost = cash if knowledge in {"exact", "estimated"} else None
    if knowledge == "estimated" and cost is None:
        cost = provider_cost_value(cost_evidence.get("estimatedUsd"))
    usage = {
        "prompt_tokens": counters.get("input_tokens"),
        "completion_tokens": counters.get("output_tokens"),
        "cached_tokens": counters.get("cached_input_tokens"),
        "cache_write_tokens": counters.get("cache_write_tokens"),
        "reasoning_tokens": counters.get("reasoning_tokens"),
    }
    if isinstance(result.get("processing"), dict):
        usage["processing"] = copy.deepcopy(result["processing"])
    if cost_evidence:
        usage["cost_evidence"] = copy.deepcopy(cost_evidence)
    usage["total_tokens"] = (
        int(usage["prompt_tokens"] or 0) + int(usage["completion_tokens"] or 0)
        if all(usage[key] is not None for key in ("prompt_tokens", "completion_tokens")) else None
    )
    return usage, cost, cost is not None and knowledge == "exact"


class ModelTurnState:
    """One caller-owned slot holding the engine's current active-turn envelope.

    A reprepared send is the SAME logical turn, so the slot survives the
    existing deep copy of a call's keyword arguments by identity: the copy IS
    this object, which is what lets a quota wait, a connection rejoin or a
    context rebuild update the ORIGINAL owner instead of a fork. Nothing else
    is stored here — no route comparison, no expiry, no history.
    """

    __slots__ = ("envelope",)

    def __init__(self, envelope: dict | None = None):
        self.envelope = envelope

    def __deepcopy__(self, memo):
        return self

    def __repr__(self) -> str:
        # This reaches private call logs; the opaque value itself never does.
        return f"ModelTurnState(active={self.envelope is not None})"


def turn_state_for_route(slot: ModelTurnState | None, provider: str) -> ModelTurnState | None:
    """Keep the slot only while the dispatch stays on this transport.

    A send that leaves for a direct API or local route ends the active turn at
    the caller, and returning later starts a fresh one rather than reviving a
    token the engine no longer owns.
    """
    if slot is None:
        return None
    if str(provider or "") != "claudexor":
        slot.envelope = None
        return None
    return slot


def _requested_turn_state(slot: ModelTurnState | None) -> tuple[bool, dict | None]:
    """(opted in, value to send) for one request, honoring the engine schema floor."""
    if slot is None or not engine_at_least(
        owned_engine_version(), config.CLAUDEXOR_MODEL_TURN_STATE_MIN_VERSION
    ):
        return False, None
    return True, copy.deepcopy(slot.envelope)


def adopt_turn_state(slot: ModelTurnState | None, payload: dict, result: dict) -> None:
    """Take the active-turn envelope from a DISPATCHED durable result.

    This seam adopts a new envelope only for a result the engine proved
    terminal on a request that ASKED about the turn. A legacy-shaped exchange —
    the shape the version floor sends whenever the serving engine is unproven —
    carries no ``nativeContinuation`` field either way, so its result is SILENCE
    about the turn, not a disclaimer that one ended: it leaves a live token
    exactly where it was (BIBLE P1). Within an opted-in request an absent result
    field leaves the turn with no state rather than inventing one, while a
    not-dispatched or unknown outcome never reaches here at all.
    """
    if slot is None or "nativeContinuation" not in payload:
        return
    envelope = result.get("nativeContinuation")
    slot.envelope = copy.deepcopy(envelope) if isinstance(envelope, dict) else None


def cache_key_for_model(model: str) -> str:
    """The Codex prompt-cache key every main-loop execution of this install shares.

    The Codex backend reuses a cached prefix across conversations only when both
    ``prompt_cache_key`` and the ``session_id`` header match (the adapter sets
    both from ``cacheKey``), and per-conversation turn states stay valid under a
    shared session (measured 2026-09-17). One key per data root and model
    therefore lets a new task, child or consciousness cycle be served the
    governance prefix it shares with its predecessors on its very first round,
    instead of paying it cold under a per-execution key — once ``_request`` has
    projected the declared prefix into its own input item. Empty for every other
    provider: API-compatible lanes keep their prefix-derived session identity.
    """
    from ouroboros.provider_models import provider_for_model

    if provider_for_model(model) != "claudexor":
        return ""
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", str(model).rsplit("=", 1)[-1]).strip("-")[:40] or "model"
    digest = hashlib.sha256(f"{config.DATA_DIR}\0{model}".encode("utf-8")).hexdigest()[:16]
    return f"ouroboros-{label}-{digest}"


def _request(target: dict, messages: list, tools: list | None, parameters: dict) -> dict:
    from ouroboros.llm_messages import _MessageShapingMixin, project_declared_system_prefix

    for name in ("response_format", "allow_server_web_search", "bypass_response_cache"):
        if parameters.get(name) or (name == "response_format" and parameters.get(name) is not None):
            raise ClaudexorModelError({"code": "unsupported_parameter", "message": f"Claudexor model transport does not support {name}.",
                                       "context": {"parameter": name}}, model_role=parameters.get("model_role", ""))
    # Only known host and foreign-provider metadata leave the send copy. Native
    # Claudexor payloads and tool schemas are opaque here and are never walked.
    # Every model source (today: Codex) shares a donor's cached prefix only up to an input-item boundary (33,024 vs 213,888).
    prepared = project_declared_system_prefix(target, scrub_native_custody(_MessageShapingMixin._normalize_system_message_placement(messages)))
    for message in prepared:
        for name in ("_context_capsule", "acceptance_observation", "_acceptance_observation", "review_feedback",
                     "reasoning", "reasoning_details", "reasoning_content", "response_id", "stop_reason", "_stable_prefix_blocks"):
            message.pop(name, None)
        # A direct provider's refusal is assistant content, not routing metadata.
        # Preserve both text parts verbatim when a response carries both fields;
        # keep the original history untouched and translate only the send copy.
        refusal = message.pop("refusal", None)
        if refusal:
            if not isinstance(refusal, str):
                raise ClaudexorModelError({"code": "invalid_request", "message": "Assistant refusal must be text."})
            content = message.get("content")
            if not content:
                message["content"] = refusal
            else:
                parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
                message["content"] = [*parts, {"type": "text", "text": refusal}]
        content = message.get("content")
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict):
                for name in ("_caption", "_source_path", "_context_capsule", "cache_control"):
                    block.pop(name, None)
        if message.get("role") == "tool" and isinstance(content, list) and content and all(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        ):
            message["content"] = context_fit.extract_plain_text_from_content(content)
    role = parameters.get("model_role", "")
    override = parameters.get("model_account_override")
    if override is not None and not isinstance(override, str):
        raise ValueError("model_account_override must be a profile name, empty Auto, or None")
    pin = override.strip() if override is not None else model_role_option(MODEL_ACCOUNTS_KEY, role)
    account = {"mode": "pin", "profileId": pin} if pin else {"mode": "auto"}
    # The next matching-route DISPATCH only; Pin still consumes it. A prospective
    # build reads the same fact without spending it, so the priced candidate and
    # the send it admits stay identical.
    failed_profile = (take_failed_account_preference if not parameters.get("prospective")
                      else failed_account_preference)(target, parameters)
    if not pin and not parameters.get("_no_account_preference"):
        # Carry the conversation's last account as a preference, not admission;
        # the engine still chooses. A round already answered by the wrong model
        # carries none, so its ranking decides where every redo lands.
        for message in reversed(prepared):
            native = message.get("nativeContinuation") or {}
            route = native.get("route") or {}
            if route.get("source") == target["source"] and route.get("model") == target["resolved_model"]:
                if route.get("credentialProfileId") and route["credentialProfileId"] != failed_profile:
                    account["preferredProfileId"] = route["credentialProfileId"]
                break
    options = {wire: parameters[key] for key, wire in (
        ("reasoning_effort", "reasoningEffort"), ("temperature", "temperature"),
        ("cache_affinity", "cacheKey"),
        ("service_tier", "serviceTier"),
    ) if parameters.get(key) is not None and parameters.get(key) != ""}
    processing = parameters.get("_processing_submission", target.get("processing_preference"))
    if processing and processing in target.get("processing_preferences", []):
        options["processingPreference"] = processing
    opted_in, turn_state = _requested_turn_state(parameters.get("model_turn_state"))
    return {"source": target["source"], "model": target["resolved_model"], "account": account,
            "messages": prepared, "tools": copy.deepcopy(tools or []),
            # Absent is the legacy stateless shape; explicit null opts into an
            # active turn that has no captured state yet.
            **({"nativeContinuation": turn_state} if opted_in else {}),
            "toolChoice": copy.deepcopy(parameters.get("tool_choice", "auto")), "options": options}


class _ModelInvocation:
    """Custody of one temporary gateway and one caller-identified operation."""

    def __init__(self, target: dict, payload: dict, parameters: dict):
        self.target, self.payload = target, payload
        self.model_turn_state = parameters.get("model_turn_state")
        self.role = str(parameters.get("model_role") or "")
        self.output_reserve = int(parameters.get("max_tokens") or 0)
        self.timeout = llm_transport_timeout_sec(parameters.get("timeout"))
        self.gateway = None
        self.operation_id = ""
        self.invocation_id = ""
        self.request_ref: dict = {}
        self.response_ref: dict = {}
        self.retained: dict = {}
        self.retention_error = ""
        self.root = config.DATA_DIR
        self.task_id = ""
        self.capture = None
        self.detail: dict = {}
        self.poll_control = parameters.get("model_poll_control")
        self.operation_observer = parameters.get("model_operation_observer")
        self.request_manifest_ref: dict = {}
        self.interrupt_reason = ""
        self.create_attempted = False
        self.capture_failure_evidence = False
        self.defer_close = False
        self.io_active = False
        self.io_lock = threading.Lock()
        self.outage_episode = None

    def check_control(self):
        """The caller supplies deadline/cancel policy; this seam transports it."""
        reason = self.interrupt_reason or (self.poll_control() if self.poll_control else None)
        if not reason:
            waiter = current_model_wait()
            reason = waiter.control_reason() if waiter is not None else None
        if not reason:
            return
        cancellation = "not_requested"
        if self.operation_id:
            try:
                self.gateway.cancel_model_operation(self.operation_id, reason_code="host_cancelled")
                cancellation = "requested"  # a POST never proves terminality
            except Exception:
                cancellation = "unconfirmed"
        problem = {"code": "model_operation_interrupted", "message": "The caller interrupted model-result waiting.",
                   "context": {"control_reason": reason, "cancellation": cancellation}}
        cls = ClaudexorModelError if self.create_attempted else ClaudexorModelNotDispatched
        error = cls(problem, model_role=self.role, operation_id=self.operation_id,
                    route=(self.detail.get("dispatch") or {}).get("route"))
        error.control_reason = reason
        if self.capture is not None:
            error.physical_attempt_capture = self.capture
        raise error from None

    def prepare(self, reservation):
        self.invocation_id = reservation.attempt_id
        self.root = reservation.drive_root
        scope = current_usage_scope()
        self.task_id = scope.task_id if scope else ""
        self.check_control()
        try:
            self.gateway = ensure_owned_gateway()
            # Freeze once before create. A lost create reply or replaced gateway
            # must reuse this same operation's diagnostic/idempotency contract.
            self.capture_failure_evidence = model_failure_evidence_supported(self.gateway.operations())
            self.request_ref = self.gateway.upload_model_request(self.payload, idempotency_key=self.invocation_id)
            self.request_manifest_ref = persist_call(self.root, task_id=self.task_id, call_id=f"{self.invocation_id}_model_request",
                         call_type="llm_claudexor_request", payload=self.payload, keep_raw=True,
                         manifest={"invocation_id": self.invocation_id, "request_ref": self.request_ref,
                                   "model_role": self.role,
                                   "capture_failure_evidence": self.capture_failure_evidence})["manifest_ref"]
        except ClaudexorUnavailable as error:
            raise ClaudexorModelError({"code": error.code, "message": str(error)}, model_role=self.role) from None

    def observe_operation(self, *, accepted: bool = False) -> None:
        """Publish custody to an optional process boundary, not provider content."""
        if accepted:
            try:
                # Only this operational request manifest is updated. The
                # ledger's immutable physical-candidate manifest is untouched.
                self.request_manifest_ref = persist_call(
                    self.root, task_id=self.task_id, call_id=f"{self.invocation_id}_model_request",
                    call_type="llm_claudexor_request", payload=self.payload, keep_raw=True,
                    manifest={"invocation_id": self.invocation_id, "request_ref": self.request_ref,
                              "model_role": self.role, "operation_id": self.operation_id,
                              "capture_failure_evidence": self.capture_failure_evidence})["manifest_ref"]
            except Exception as error:
                self.request_manifest_ref = {}
                log.warning("Model custody checkpoint unavailable: %s", type(error).__name__)
        if self.operation_observer is not None:
            try:
                self.operation_observer({"operation_id": self.operation_id, "invocation_id": self.invocation_id,
                                         "request_ref": copy.deepcopy(self.request_ref),
                                         "request_manifest_ref": copy.deepcopy(self.request_manifest_ref)})
            except Exception as error:
                log.warning("Model custody observer unavailable: %s", type(error).__name__)

    def error(self, problem: dict | None, detail: dict | None = None, *, unknown: bool = False):
        detail = detail or {}
        route = (detail.get("dispatch") or {}).get("route") or {}
        cls = ClaudexorModelError if unknown else ClaudexorModelNotDispatched
        return cls(problem or {"code": "model_operation_failed", "message": "The engine returned no model result."},
                   model_role=self.role, operation_id=self.operation_id, route=route, unknown=unknown)

    def receive(self) -> dict:
        outage_started = None
        detail: dict = {}
        while True:
            self.check_control()
            try:
                if not self.operation_id:
                    self.create_attempted = True
                    self.observe_operation()
                    detail = self.gateway.create_model_operation(self.request_ref, idempotency_key=self.invocation_id,
                        **({"capture_failure_evidence": True} if self.capture_failure_evidence else {}))
                    self.operation_id = detail["id"]
                    self.observe_operation(accepted=True)
                else:
                    detail = self.gateway.get_model_operation(self.operation_id, timeout_sec=min(self.timeout, _READ_TIMEOUT_SEC))
                self.detail = detail
                if self.outage_episode is not None:
                    self._control_outage(recovered=True)
                if detail.get("state") not in {"queued", "running"}:
                    self.detail = detail
                    response = detail.get("response") or {}
                    if response.get("state") != "ready":
                        raise self.error(detail.get("problem"), detail,
                                         unknown=(detail.get("dispatch") or {}).get("state") != "not_started")
                    self.response_ref = response["ref"]
                    raw = self.gateway.get_model_result(self.operation_id, expected_ref=self.response_ref,
                                                        timeout_sec=min(self.timeout, _READ_TIMEOUT_SEC), raw_bytes=True)
                    self.retain(raw)
                    result = json.loads(raw.decode("utf-8"))
                    if (detail.get("dispatch") or {}).get("state") == "not_started":
                        error = self.error(result.get("problem"), detail)
                        error.route = copy.deepcopy(result.get("route") or error.route)
                        raise error
                    if result.get("outcome") == "unknown" or (detail.get("dispatch") or {}).get("state") == "unknown":
                        raise self.error(result.get("problem"), detail, unknown=True)
                    if (detail.get("dispatch") or {}).get("state") != "response_received" or result.get("outcome") not in {"completed", "incomplete", "failed"}:
                        raise self.error({"code": "malformed_response", "message": "The engine did not prove a terminal provider response."}, detail, unknown=True)
                    problem = result.get("problem") or {}
                    context = problem.get("context") or {}
                    options = self.payload.get("options") or {}
                    if (result.get("outcome") == "failed" and problem.get("code") == "processing_unavailable"
                            and context.get("generationStarted") is False
                            and context.get("processingFallback") == "standard"
                            and context.get("processingRefusal") in {"capacity", "unsupported"}
                            and self.target.get("processing_preference") in {"fast", "economy"}
                            and options.get("processingPreference") in {"fast", "economy"}
                            and not options.get("serviceTier")):
                        # response_received describes the refusal envelope. The
                        # engine separately proves generation never started.
                        raise ClaudexorModelNotDispatched(problem, model_role=self.role,
                            operation_id=self.operation_id, route=result.get("route") or {})
                    return result
                outage_started = None
            except ClaudexorUnavailable as error:
                if not self.operation_id and error.code == "model_request_invalid":
                    # This exact create refusal is minted by the engine only
                    # after its idempotency lookup proves no accepted command,
                    # and before enqueue. A GET or an arbitrary 4xx cannot mint
                    # non-dispatch authority for an earlier accepted request.
                    raise self.error({"code": error.code, "message": str(error),
                                      "context": {"httpStatus": error.status_code}}) from None
                # A failed read after creation is never a provider connect failure.
                # Drop its causal HTTP chain at this boundary: even ConnectError
                # means only the local control read failed, not inference un-sent.
                if error.code != "daemon_unreachable" and not 500 <= error.status_code < 600:
                    raise self.error({"code": error.code, "message": str(error)}, detail, unknown=True) from None
                if outage_started is None:
                    outage_started = time.monotonic()
                if self._control_outage():
                    continue
                if time.monotonic() - outage_started >= self.timeout:
                    raise self.error({"code": "model_control_unreachable", "message": "Control connection lost; the same model operation may still finish."}, detail, unknown=True) from None
            time.sleep(min(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC, self.timeout))

    def _control_outage(self, *, recovered: bool = False) -> bool:
        """Managed calls keep the same accepted operation through local HTTP loss."""
        from ouroboros.loop_transport import (
            TransportWaitEpisode, emit_network_wait_event, managed_transport_continuation)
        waiter = current_model_wait()
        ctx = getattr(waiter, "tool_context", None)
        if not managed_transport_continuation(ctx):
            return False
        if recovered:
            emit_network_wait_event(self.root / "logs", task_id=self.task_id, phase="recovered",
                elapsed_sec=self.outage_episode.waited_sec, redials=0, model=self.target["usage_model"],
                detail="same_model_operation_rejoined", outcome_custody={"operation_id": self.operation_id})
            self.outage_episode = None
            return True
        if self.outage_episode is None:
            self.outage_episode = TransportWaitEpisode(started_monotonic=time.monotonic())
        episode = self.outage_episode
        backoff = min(config.NETWORK_WAIT_BACKOFF_START_SEC * 2 ** min(episode.wait_iterations, 4),
                      config.NETWORK_WAIT_BACKOFF_MAX_SEC)
        emit_network_wait_event(self.root / "logs", task_id=self.task_id, phase="waiting",
            elapsed_sec=episode.waited_sec, redials=0, model=self.target["usage_model"],
            next_sleep_sec=backoff, detail="same_model_operation_pending",
            outcome_custody={"operation_id": self.operation_id, "invocation_id": self.invocation_id})
        episode.wait_iterations += 1
        try:
            replacement = read_owned_gateway()
        except ClaudexorUnavailable:
            replacement = None
        if replacement is not None:
            previous, self.gateway = self.gateway, replacement
            if previous is not None:
                previous.close()
        def controlled():
            self.check_control()
            return False
        # Unlike an owner-mail peek, check_control's exception must propagate.
        deadline = time.monotonic() + backoff
        while time.monotonic() < deadline:
            controlled()
            time.sleep(min(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC, max(0, deadline - time.monotonic())))
        return True

    def retain(self, raw: bytes) -> None:
        try:
            self.retained = persist_call(
                self.root, task_id=self.task_id, call_id=f"{self.invocation_id}_model_response",
                # A UTF-8 string in the existing JSON CAS is reversible to the
                # verified wire bytes, including whitespace and numeric spelling.
                call_type="llm_claudexor_response", payload={"result_json_utf8": raw.decode("utf-8")}, keep_raw=True,
                manifest={"operation_id": self.operation_id, "invocation_id": self.invocation_id,
                          "response_ref": self.response_ref, "model_role": self.role,
                          "operation_state": self.detail.get("state"),
                          "dispatch_state": (self.detail.get("dispatch") or {}).get("state")},
            )
        except Exception as error:
            # The caller keeps the useful result and the engine keeps its bytes.
            # Lack of local durable custody withholds ACK, never the paid answer.
            self.retention_error = type(error).__name__

    def acknowledge(self) -> dict:
        custody = {"state": "pending", "operation_id": self.operation_id, "response_ref": self.response_ref,
                   "retained_manifest_ref": self.retained.get("manifest_ref")}
        if not self.retained:
            custody["reason"] = f"result_retention_failed:{self.retention_error}"
            return custody
        try:
            receipt = self.gateway.acknowledge_model_result(self.operation_id, self.response_ref["sha256"])
            custody["state"] = (receipt.get("response") or {}).get("state", "pending")
        except Exception as error:
            custody["reason"] = error.code if isinstance(error, ClaudexorUnavailable) else type(error).__name__
        return custody

    def extract_usage(self, result: dict) -> tuple[dict, float | None, bool]:
        usage, cost, final = _usage(result)
        # Settlement reads this row before the caller decides anything, and the
        # density witness must know whose tokenizer it measured: a generation
        # another model produced teaches nothing about the requested one. The
        # ENGINE decided that, here as everywhere; the host compares no models.
        if substitution_fact(result):
            usage["claudexor"] = {"served_other_model": True}
        if "processing" not in usage and self.target.get("processing_preference"):
            options = self.payload.get("options") or {}
            usage["processing"] = {
                "requested": self.target["processing_preference"],
                "submitted": options.get("processingPreference"),
                "submittedNative": options.get("serviceTier"), "observed": "unknown",
                "observedNative": [], "reason": ("processing_not_submitted"
                    if not options.get("processingPreference") and not options.get("serviceTier") else None),
                "source": "host_request",
            }
        return usage, cost, final

    def finish(self, result: dict) -> tuple[dict, dict]:
        usage, cost, final = self.extract_usage(result)
        route = result.get("route") or {}
        requested_options = copy.deepcopy(self.payload.get("options") or {})
        applied_options = copy.deepcopy(result.get("appliedOptions"))
        options_honored = "unknown" if applied_options is None else (
            "mismatch" if any(applied_options[key] != value for key, value in requested_options.items() if key in applied_options) else "confirmed")
        usage.pop("wire_layout", None)  # host-owned: the projection fact of THIS call's target
        usage.update(provider="claudexor", resolved_model=self.target["usage_model"], cost=cost, cost_final=final,
                     cost_estimated=cost is not None and not final,
                     **({"wire_layout": dict(self.target["wire_layout"])} if isinstance(self.target.get("wire_layout"), dict) else {}),
                     claudexor={"operation_id": self.operation_id, "model_role": self.role,
                                "requested_profile": str((self.payload.get("account") or {}).get("profileId") or ""),
                                "route": copy.deepcopy(route), "cost_evidence": copy.deepcopy(result.get("cost")),
                                "outcome": result.get("outcome"), "problem": copy.deepcopy(result.get("problem")),
                                "requested_options": requested_options, "applied_options": applied_options,
                                "options_honored": options_honored,
                                "output_reserve_tokens": self.output_reserve, "output_cap_applied": False,
                                "result_custody": {"state": "pending", "operation_id": self.operation_id,
                                                   "response_ref": self.response_ref,
                                                   "retained_manifest_ref": self.retained.get("manifest_ref")}})
        try:
            self.check_control()
            usage["claudexor"]["result_custody"] = self.acknowledge()
            self.check_control()
        except ClaudexorModelError as error:
            # Control changes authority to continue, never ownership of the
            # already received result or its settled usage.
            error.usage = usage
            error.model_result = copy.deepcopy(result)
            error.route = copy.deepcopy(route)
            raise
        if result.get("outcome") == "failed" or self.detail.get("state") == "cancelled":
            problem = ({"code": "model_operation_cancelled", "message": "The engine cancelled this model operation."}
                       if self.detail.get("state") == "cancelled" else result.get("problem") or {})
            error = ClaudexorModelError(problem, model_role=self.role,
                                       operation_id=self.operation_id, route=route)
            error.physical_attempt_capture = self.capture
            error.usage = usage
            raise error
        message = result.get("message")
        if not isinstance(message, dict):
            error = ClaudexorModelError(result.get("problem") or {
                "code": "response_rejected", "message": "The terminal provider response contained no usable model message."},
                model_role=self.role, operation_id=self.operation_id, route=route)
            error.stream_rejected = error.stream_incomplete = True
            error.physical_attempt_capture = self.capture
            error.usage = usage
            raise error
        if result.get("outcome") == "incomplete":
            usage["response_finish_reason"] = (
                "length" if ((result.get("problem") or {}).get("context") or {}).get("reason") == "max_output_tokens"
                else "incomplete"
            )
        return copy.deepcopy(message), usage
    async def offload(self, function, *args):
        """A cancelled caller leaves the current I/O thread owning its gateway."""
        with self.io_lock:
            self.io_active = True

        def run():
            try:
                return function(*args)
            finally:
                with self.io_lock:
                    self.io_active = False
                    close = self.defer_close
                if close:
                    self.close()

        task = asyncio.create_task(asyncio.to_thread(run))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with self.io_lock:
                self.interrupt_reason = "caller_cancelled"
                self.defer_close = True
                close = not self.io_active
            if close:
                self.close()
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            raise

    def close(self):
        with self.io_lock:
            gateway, self.gateway = self.gateway, None
        if gateway is not None:
            gateway.close()


def _reset_native(payload: dict, error: ClaudexorModelNotDispatched, invocation: _ModelInvocation) -> dict | None:
    from ouroboros.llm_messages import reset_native_payload

    capture = getattr(error, "physical_attempt_capture", None)
    if error.code != "invalid_continuation" or getattr(capture, "state", None) != "released":
        return None
    updated_with_slot = reset_native_payload(
        payload, error.route, source=payload["source"], model=payload["model"],
        turn_state=getattr(invocation, "model_turn_state", None))
    if updated_with_slot is None:
        return None
    updated, changed, surface = updated_with_slot
    append_jsonl(invocation.root / "logs" / "events.jsonl", {
        "ts": utc_now_iso(), "type": "native_continuation_reset", "task_id": invocation.task_id,
        "model_role": invocation.role, "operation_id": invocation.operation_id, "routes": changed,
        **({"surface": surface} if surface else {}),
    })
    return updated


def _accounted_request(invocation: _ModelInvocation):
    request = replace(_attempt_request(invocation.target, invocation.payload),
                      force_unknown_reservation=True, max_completion_tokens=invocation.output_reserve)
    existing = _candidate_before_dispatch(invocation.payload, request)

    def before(reservation):
        manifest = existing(reservation)
        invocation.prepare(reservation)
        return manifest

    return request, before


def _native_retry_preparation(target: dict, payload: dict, parameters: dict, error: ClaudexorModelNotDispatched):
    """Rebind the already-authorized un-sent repair before preparing its next attempt.

    The engine's new account receipt replaces provisional discovery. Pass the
    sanitized send copy to ordinary callers; Main re-applies this attested
    account reset to its canonical source before rebuilding its vision view.
    """
    values = {**parameters, "messages": payload["messages"], "tools": payload["tools"],
              "model": target["usage_model"], "use_local": False}
    waiter = current_model_wait()
    if waiter is None:
        if current_physical_attempt_context() is not None:
            raise ModelWaitInterrupted("model_wait_reprepare_required", role=parameters.get("model_role", ""), cause=error)
        return values  # Bare helpers have no captured Main fit to replace.
    values["_model_observed_route"] = {**error.route, "source": target["source"], "model": target["resolved_model"]}
    if error.code == "invalid_continuation" and getattr(getattr(error, "physical_attempt_capture", None), "state", None) == "released":
        # A processing-only repair may also report another account. Only the
        # typed no-start native reset authorizes Main to scrub its source copy.
        values["_model_observed_route"]["_native_reset"] = True
    return waiter.reprepare(parameters.get("model_role", ""), values)


def _processing_retry_payload(payload: dict, error: ClaudexorModelNotDispatched) -> dict | None:
    """Read the already proved generation refusal; never convert an unknown run."""
    context = error.problem.get("context") or {}
    capture = getattr(error, "physical_attempt_capture", None)
    options = payload.get("options") or {}
    if (error.code != "processing_unavailable" or getattr(capture, "state", None) != "released"
            or context.get("generationStarted") is not False
            or context.get("processingFallback") != "standard"
            or context.get("processingRefusal") not in {"capacity", "unsupported"}
            or options.get("processingPreference") not in {"fast", "economy"}
            or options.get("serviceTier")):
        return None
    updated = copy.deepcopy(payload)
    updated["options"]["processingPreference"] = "standard"
    return updated


def chat_claudexor(target: dict, messages: list, tools: list | None, **parameters: Any) -> tuple[dict, dict]:
    """One generation, one no-start repair per continuation/processing axis, quota re-asks on Auto."""
    target = prepare_processing_target(target)
    payload = _request(target, messages, tools, parameters)
    retry_preparation = None
    native_repaired = processing_repaired = False
    substitution, rotation = SubstitutionBudget(ClaudexorModelError), AccountRotation()
    all_operations_not_started = True
    for _preparation in range(3 + substitution.redos + rotation.CEILING):
        invocation = _ModelInvocation(target, payload, parameters)
        try:
            with prepared_call_scope(retry_preparation or {}) as prepared:
                if prepared:
                    invocation.payload = payload = _request(target, prepared["messages"], prepared.get("tools"), prepared)
                request, before = _accounted_request(invocation)
                result = execute_physical_attempt(request, invocation.receive, extractor=invocation.extract_usage, before_dispatch=before)
                all_operations_not_started = False  # even a discarded substituted response ran
                invocation.capture = last_physical_attempt_capture()
                if substitution.admit(invocation, result):
                    # The same round, asked again naming no account, on this
                    # call's every later request: the engine alone picks.
                    retry_preparation, parameters = None, {**parameters, "_no_account_preference": True}
                    payload = _request(target, payload["messages"], payload["tools"], {**(prepared or parameters), "_no_account_preference": True})
                    continue
                adopt_turn_state((prepared or parameters).get("model_turn_state"),
                                 invocation.payload, result)
                return rotation.disclose(substitution.disclose(invocation.finish(result)))
        except ClaudexorModelNotDispatched as error:
            all_operations_not_started &= getattr(getattr(error, "physical_attempt_capture", None), "state", None) == "released"
            error.presence_all_operations_not_started = all_operations_not_started
            if invocation.response_ref:
                invocation.acknowledge()
            updated = (_processing_retry_payload(payload, error)
                       if not processing_repaired and "standard" in target.get("processing_preferences", []) else None)
            if updated is not None:
                processing_repaired = True
                parameters = {**parameters, "_processing_submission": "standard"}
            elif not native_repaired:
                updated = _reset_native(payload, error, invocation)
                native_repaired = updated is not None
            if updated is None:
                rotation.refused_call(target, parameters, invocation, error)  # an engine verdict is never re-asked
                raise
            payload = updated
            retry_preparation = _native_retry_preparation(target, payload, parameters, error)
        except ClaudexorModelError as error:
            all_operations_not_started = False
            if not rotation.refused_call(target, parameters, invocation, error):
                raise
            retry_preparation, parameters, payload = rotation.reask(parameters, payload)
        except PhysicalAttemptPreparationFailed as error:
            cause = error.__cause__
            if isinstance(cause, ClaudexorModelError):
                cause.physical_attempt_capture = error.physical_attempt_capture
                all_operations_not_started &= (isinstance(cause, ClaudexorModelNotDispatched)
                                               and error.physical_attempt_capture.state == "released")
                if isinstance(cause, ClaudexorModelError):
                    cause.presence_all_operations_not_started = all_operations_not_started
                raise cause from None
            raise
        finally:
            invocation.close()
    raise AssertionError("Unreachable model preparation loop")


def recover_model_attempt(drive_root, row: dict, *, gateway_factory=None):
    """Read retained receipts offline; without dispatch facts, read their exact operation.
    Missing/live/unknown custody defers, unknown price stays unknown; never create or cancel."""
    from ouroboros.observability import call_manifest_path, read_call_payload
    from ouroboros.utils import read_json_dict
    task_id, attempt_id = str(row.get("task_id") or ""), str(row.get("attempt_id") or "")
    call_id = f"{attempt_id}_model_response"
    manifest, payload = {}, {}
    try:
        manifest, payload, _ = read_call_payload(drive_root, task_id=task_id, call_id=call_id)
    except FileNotFoundError:
        pass
    if manifest and manifest.get("invocation_id") != attempt_id:
        return None
    request = read_json_dict(call_manifest_path(drive_root, task_id, f"{attempt_id}_model_request")) or {}
    if request and (request.get("invocation_id") != attempt_id or request.get("task_id") != task_id):
        return None
    operation_id = manifest.get("operation_id") or request.get("operation_id")
    if not operation_id or (request.get("operation_id") and request["operation_id"] != operation_id):
        return None
    operation_state, dispatch_state = manifest.get("operation_state"), manifest.get("dispatch_state")
    raw = payload.get("result_json_utf8") if isinstance(payload, dict) else None
    gateway = None
    try:
        if operation_state not in {"succeeded", "failed", "cancelled"} or not dispatch_state:
            gateway = (gateway_factory or read_owned_gateway)()
            detail = gateway.get_model_operation(operation_id, timeout_sec=_READ_TIMEOUT_SEC)
            operation_state = detail.get("state")
            dispatch_state = (detail.get("dispatch") or {}).get("state")
            if operation_state not in {"succeeded", "failed", "cancelled"}:
                return None
            response = detail.get("response") or {}
            response_ref = response.get("ref") or manifest.get("response_ref") or {}
            if raw is None and response.get("state") == "ready":
                raw = gateway.get_model_result(operation_id, expected_ref=response_ref,
                                               timeout_sec=_READ_TIMEOUT_SEC, raw_bytes=True).decode("utf-8")
            payload = {"result_json_utf8": raw} if raw is not None else {"operation": detail}
            persist_call(drive_root, task_id=task_id, call_id=call_id,
                         call_type="llm_claudexor_response", payload=payload, keep_raw=True,
                         manifest={"operation_id": operation_id, "invocation_id": attempt_id,
                                   "response_ref": response_ref, "operation_state": operation_state,
                                   "dispatch_state": dispatch_state})
            if raw is not None and response.get("state") == "ready":
                try:
                    gateway.acknowledge_model_result(operation_id, response_ref["sha256"])
                except Exception:
                    log.debug("Recovered model response retained without ACK", exc_info=True)
        if dispatch_state == "not_started":
            return "released", {}, None, False
        if dispatch_state != "response_received" or raw is None:
            return "abandoned", {}, None, False
        result = json.loads(raw)
        if not isinstance(result, dict) or result.get("outcome") not in {"completed", "incomplete", "failed"}:
            return "abandoned", {}, None, False
        usage, cost, final = _usage(result)
        return "settled", usage, cost, final
    finally:
        if gateway is not None and gateway_factory is None:
            gateway.close()


async def chat_claudexor_async(target: dict, messages: list, tools: list | None, **parameters: Any) -> tuple[dict, dict]:
    """Keep accounting/capture in the async caller; offload only synchronous I/O."""
    target = (await asyncio.to_thread(prepare_processing_target, target)
              if target.get("processing_preference") and "processing_preferences" not in target else target)
    payload = _request(target, messages, tools, parameters)
    retry_preparation = None
    native_repaired = processing_repaired = False
    substitution, rotation = SubstitutionBudget(ClaudexorModelError), AccountRotation()
    all_operations_not_started = True
    for _preparation in range(3 + substitution.redos + rotation.CEILING):
        invocation = _ModelInvocation(target, payload, parameters)
        try:
            with prepared_call_scope(retry_preparation or {}) as prepared:
                if prepared:
                    invocation.payload = payload = _request(target, prepared["messages"], prepared.get("tools"), prepared)
                request, before = _accounted_request(invocation)

                async def prepare(reservation):
                    return await invocation.offload(before, reservation)

                async def receive():
                    return await invocation.offload(invocation.receive)

                result = await execute_physical_attempt_async(
                    request, receive, extractor=invocation.extract_usage, before_dispatch=prepare)
                all_operations_not_started = False
                invocation.capture = last_physical_attempt_capture()
                if await invocation.offload(substitution.admit, invocation, result):
                    # The same round, asked again naming no account, on this
                    # call's every later request: the engine alone picks.
                    retry_preparation, parameters = None, {**parameters, "_no_account_preference": True}
                    payload = _request(target, payload["messages"], payload["tools"], {**(prepared or parameters), "_no_account_preference": True})
                    continue
                adopt_turn_state((prepared or parameters).get("model_turn_state"),
                                 invocation.payload, result)
                return rotation.disclose(substitution.disclose(await invocation.offload(invocation.finish, result)))
        except ClaudexorModelNotDispatched as error:
            all_operations_not_started &= getattr(getattr(error, "physical_attempt_capture", None), "state", None) == "released"
            error.presence_all_operations_not_started = all_operations_not_started
            if invocation.response_ref:
                await invocation.offload(invocation.acknowledge)
            updated = (_processing_retry_payload(payload, error)
                       if not processing_repaired and "standard" in target.get("processing_preferences", []) else None)
            if updated is not None:
                processing_repaired = True
                parameters = {**parameters, "_processing_submission": "standard"}
            elif not native_repaired:
                updated = _reset_native(payload, error, invocation)
                native_repaired = updated is not None
            if updated is None:
                rotation.refused_call(target, parameters, invocation, error)  # an engine verdict is never re-asked
                raise
            payload = updated
            retry_preparation = _native_retry_preparation(target, payload, parameters, error)
        except ClaudexorModelError as error:
            all_operations_not_started = False
            if not rotation.refused_call(target, parameters, invocation, error):
                raise
            retry_preparation, parameters, payload = rotation.reask(parameters, payload)
        except PhysicalAttemptPreparationFailed as error:
            cause = error.__cause__
            if isinstance(cause, (ClaudexorModelError, asyncio.CancelledError)):
                cause.physical_attempt_capture = error.physical_attempt_capture
                all_operations_not_started &= (isinstance(cause, ClaudexorModelNotDispatched)
                                               and error.physical_attempt_capture.state == "released")
                if isinstance(cause, ClaudexorModelError):
                    cause.presence_all_operations_not_started = all_operations_not_started
                raise cause from None
            raise
        finally:
            if not invocation.defer_close:
                invocation.close()
    raise AssertionError("Unreachable model preparation loop")
