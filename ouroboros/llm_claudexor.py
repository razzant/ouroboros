"""Caller-owned physical accounting and engine continuation; private CAS precedes ACK.
Dispatch, turn-slot, pricing and engine-version semantics: ARCHITECTURE §6."""

from __future__ import annotations

import asyncio
import contextlib
import copy
from dataclasses import dataclass, replace
import hashlib
import json
import re
import logging
import threading
import time
from typing import Any

from ouroboros import _usage_wait, config, context_fit
from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REPLY_SOURCE_SHA_KEY
from ouroboros._usage_response import model_operation_usage as _usage, observe_finish_reason
from ouroboros.anthropic_native_custody import scrub_native_custody
from ouroboros.claudexor_daemon import ensure_owned_gateway, owned_engine_version, read_owned_gateway
from ouroboros.deadline_utils import llm_transport_timeout_sec
from ouroboros.gateways.claudexor import (ClaudexorUnavailable, engine_at_least, model_failure_evidence_supported,
                                          operation_query_supported, _READ_TIMEOUT_SEC)
from ouroboros.gateways.claudexor_metadata import current_engine, model_gateway, model_rejoin_binding
from ouroboros.llm_attempt import _PhysicalSendNotStarted, _attempt_request, _candidate_before_dispatch, effort_request_facts
from ouroboros.llm_capability_policy import model_catalog as model_catalog, catalog_admits_model as catalog_admits_model
from ouroboros.send_clock import stamp_clock_note
from ouroboros.llm_substitution import (
    AccountRotation, CapabilityRetry, RESTORED_PROCESSING_KEY, SubstitutionBudget,
    capability_refusal_text, substitution_fact, send_account)
from ouroboros.model_wait import (ModelWaitInterrupted, accepted_operation_scope, current_model_wait,
                                  dispatch_deadline_remaining_sec, prepared_call_scope)
from ouroboros.observability import persist_call, update_call_manifest_metadata
from ouroboros.context_timing import host_step, mark, span_since, timed
from ouroboros.owner_pause import launch_admission, OwnerPauseRefused, model_handed_off
from ouroboros.transport_custody import ProviderNotDispatched
from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY
from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY
from ouroboros.usage_accounting import (
    PhysicalAttemptCapture, PhysicalAttemptPreparationFailed, adopt_physical_attempt_capture, current_physical_attempt_context,
    current_usage_scope, execute_physical_attempt, execute_physical_attempt_async, last_physical_attempt_capture)
from ouroboros.utils import append_jsonl, read_json_dict, sanitize_tool_result_for_log, utc_now_iso

log = logging.getLogger(__name__)
_PROCESSING_PREFERENCES: dict = {}  # each source's processingPreferences per exact engine build and source id
_ADVISORY_TIERS = ("fast", "economy", "ultrafast")
RESTORED_END_KEY = "_restored_native_end"  # a restored attempt's known end, for its round's own next call (``restored_round``)
_MODEL_STORE_UNVERIFIED = {"code": "model_control_store_unverified",
                          "message": "The original model operation's store could not be verified."}
# A request that recorded no store can never be verified by a later reader: its search has an end, a mismatch may pass.
_MODEL_STORE_UNBOUND = {"code": "model_control_store_unbound", "message":
    "The request recorded no store for its operation, so its result can be neither verified nor read; its cost stays unknown."}


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

    No model eligibility inference: this reads the existing model-source owner, only for explicit intent. A retained
    target is reused by every physical preparation of this logical call. The list is a static declaration of the
    engine build (``claudexor_run_events.declared_live_input`` keeps ``liveInput`` alike): one a read found serves later
    calls on the build that answered it (``current_engine``, the same before and after the read); a failed read, a
    missing list, an unknown build or one replaced during the read is asked again.
    """
    if not target.get("processing_preference") or "processing_preferences" in target:
        return target
    known = _PROCESSING_PREFERENCES.get((engine := current_engine(), target.get("source")))  # an unknown one: none kept
    if known is not None:
        return {**target, "processing_preferences": list(known)}
    prepared = {**target, "processing_preferences": []}
    try:
        sources = timed("claudexor.processing", model_sources, processing_view=True)
    except ClaudexorUnavailable:
        return prepared
    source = next((row for row in sources.get("sources", [])
                   if isinstance(row, dict) and row.get("id") == target.get("source")), {})
    preferences = source.get("processingPreferences")
    if isinstance(preferences, list):
        prepared["processing_preferences"] = [value for value in preferences if isinstance(value, str)]
        if engine and engine == current_engine():  # the build that answered: unchanged across this read
            _PROCESSING_PREFERENCES[(engine, target.get("source"))] = tuple(prepared["processing_preferences"])
    return prepared


class ClaudexorModelError(RuntimeError):
    """A model-operation fact with its exact role, route and recovery identity."""

    def __init__(self, problem: dict, *, model_role: str = "", operation_id: str = "",
                 route: dict | None = None, unknown: bool = False, same_operation_recoverable: bool = False):
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
        self.model_role, self.same_operation_recoverable = model_role, same_operation_recoverable  # the latter: a READ failure; rejoin, never regenerate
        self.operation_id = operation_id
        self.route = copy.deepcopy(route or {})

    @property
    def received_status(self) -> dict:
        receipt = (getattr(self, "usage", None) or {}).get("claudexor") or {}
        return ({"operation_state": receipt["operation_state"], "operation_outcome": receipt["outcome"]}
                if receipt.get("operation_state") and receipt.get("outcome") else {})

    @property
    def display_message(self) -> str:
        """Show typed provider details without changing exception classification text."""
        if self.code == "model_capability_refused":
            return sanitize_tool_result_for_log(capability_refusal_text(self.problem) or str(self))
        context = self.problem.get("context") or {}
        fields = (("stage", "stage"), ("errorCode", "cause"))
        if self.code != "model_outcome_unknown":
            # The event field the code fills when it is this error's own type; otherwise named as reported.
            fields += (("vendorCode", "provider_error_type" if self.type else "vendor_code"), ("parameter", "parameter"))
        details = [
            f"{label}={value.strip()}"
            for key, label in fields
            if isinstance(value := context.get(key), str) and value.strip()
        ]
        # Details lead so a shortened Logs preview still names the refusal.
        return sanitize_tool_result_for_log("; ".join([", ".join(details), str(self)]) if details else str(self))


class ClaudexorRefusalError(ClaudexorModelError):
    """A received refusal cannot yield another send or a verdict for a string-only caller."""

    notice = "The provider returned refusal content that this caller cannot accept."
    terminal_code = "unexpected_refusal_content"  # the problem code, the loop's reason_code and the class are one fact

    @classmethod
    def terminal_problem(cls, **extra) -> dict:
        """The public problem this refusal settles as: its code, no retry and the notice, never the provider's words."""
        return {"code": cls.terminal_code, "retryable": False, "message": cls.notice, **extra}


class ClaudexorContentRefused(ClaudexorRefusalError):
    """The settled Codex operation explicitly declined this request."""

    notice = "Codex declined this request under its content policy."
    terminal_code = "provider_content_refused"

    @classmethod
    def public_problem(cls, result, detail, source):
        """Project a verified, received refusal. The original response stays in the private CAS.

        Both producer shapes are settled and non-retryable: ``terminal_code``, or the legacy ``provider_failed``
        carrying the policy's exact ``vendorCode``. Neither discovery nor the caller's body opt-in classifies them."""
        dispatch = detail.get("dispatch") or {}
        if (source != "codex" or detail.get("state") != "failed" or result.get("outcome") != "failed"
                or dispatch.get("state") != "response_received"):
            return None
        if any(route is not None and (not isinstance(route, dict) or route.get("source", source) != source)
               for route in (result.get("route"), dispatch.get("route"))):
            return None
        problem = result.get("problem") or {}
        if not isinstance(problem, dict) or problem.get("retryable") is not False:
            return None
        context = problem.get("context")
        vendor_code = context.get("vendorCode") if isinstance(context, dict) else None
        legacy = problem.get("code") == "provider_failed" and vendor_code == "cyber_policy"
        if problem.get("code") == cls.terminal_code or legacy:
            return cls.terminal_problem(**({"context": {"vendorCode": vendor_code}} if vendor_code == "cyber_policy" else {}))
        return None


def _explicit_content_refusal(result: object) -> bool:
    problem = result.get("problem") if isinstance(result, dict) else None
    return isinstance(problem, dict) and problem.get("code") == ClaudexorContentRefused.terminal_code


def _operation_receipt_fields(detail: dict) -> dict:
    dispatch = detail.get("dispatch") or {}
    return {"operation_state": detail.get("state"), "dispatch_state": dispatch.get("state"),
            "dispatch_route": dispatch.get("route"),
            **({"operation_problem_code": ClaudexorContentRefused.terminal_code} if _explicit_content_refusal(detail) else {})}


class ClaudexorModelNotDispatched(ClaudexorModelError, ProviderNotDispatched):
    """A failed prepare before create, or an engine receipt proving dispatch.state=not_started."""


# The install's own control daemon left unanswered before any model create (no status: not a received refusal).
DAEMON_UNANSWERED_CODES = frozenset({"daemon_unreachable", "daemon_not_answering", "daemon_starting",
                                     "daemon_descriptor_changed"})


def daemon_not_delivered(failure: object) -> bool:
    """Whether a recorded failure row (``loop_llm_call``'s ``_last_llm_failure``) is a typed Claudexor prepare
    failure: the own daemon did not answer before create, so no operation exists and nothing was sent or charged.
    The model did not fail; its same-route repeats and wait-first gate (``loop_transport``) read this one fact."""
    return (isinstance(failure, dict) and failure.get("not_dispatched_provider") == "claudexor"
            and failure.get("operation_id") == "" and failure.get("provider_code") in DAEMON_UNANSWERED_CODES)


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


def propagate_model_error(error: Exception, *, single_request: bool = False) -> None:
    """Preserve control/resource waits and unknown custody across helper fallbacks.

    A confirmed ordinary provider refusal still belongs to the helper's existing
    retry or disclosed-unavailable path, just as it does for direct API calls. A
    received typed refusal ends its one request: it is raised, so no fallback of the
    helper sends that request again, on another route either. ``single_request``: this
    caller sends nothing more for the request, so the refusal takes its own ordinary
    unavailable outcome instead (a caption, an extraction, one post-task stage).
    """
    from ouroboros.model_wait import ModelWaitInterrupted, model_wait_reason
    from ouroboros.transport_custody import outcome_unknown_on_chain

    if isinstance(error, ModelWaitInterrupted):
        raise error
    if isinstance(error, ClaudexorModelError):
        if (isinstance(error, ClaudexorRefusalError) and not single_request
                or error.code in {"model_outcome_unknown", "model_operation_interrupted"}
                or model_wait_reason(error)
                or outcome_unknown_on_chain(error)):
            raise error


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
    if slot is not None and str(provider or "") != "claudexor":
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
    from ouroboros.llm_messages import (DISPLAY_KEY, _MessageShapingMixin, project_declared_system_prefix,
                                        HOST_ONLY_MESSAGE_KEYS)

    target["requested_reasoning_effort"] = parameters.get("reasoning_effort")
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
                     DISPLAY_KEY,
                     "reasoning", "reasoning_details", "reasoning_content", "response_id", "stop_reason", "_stable_prefix_blocks",
                     HOST_CONTEXT_KIND_KEY, HOST_TEXT_SHA_KEY, REPLY_SOURCE_SHA_KEY, TOOL_RESULT_RECORD_KEY, REVIEW_HISTORY_MESSAGE_KEY,
                     REVIEW_CONTEXT_INDEX_KEY, "_review_image_id", "cache_control"):  # the seal's row mark too
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
                for name in HOST_ONLY_MESSAGE_KEYS:
                    block.pop(name, None)
        if message.get("role") == "tool" and isinstance(content, list) and content and all(
            isinstance(block, dict) and block.get("type") == "text" for block in content
        ):
            message["content"] = context_fit.extract_plain_text_from_content(content)
    # The conversation's account, else the account this task's plan rests on, as a preference the engine may overrule
    # (``send_account``). The failed account is spent by the next matching DISPATCH only; Pin still consumes it. A
    # prospective build reads that fact without spending it, so the priced candidate and the send it admits stay identical.
    plan = getattr(getattr(current_model_wait(), "tool_context", None), "context_fit_plan", None)
    account = send_account(target, prepared, parameters, plan=plan, consume=not parameters.get("prospective"))[0]
    options = {wire: parameters[key] for key, wire in (
        ("reasoning_effort", "reasoningEffort"), ("temperature", "temperature"),
        ("cache_affinity", "cacheKey"),
        ("service_tier", "serviceTier"),
    ) if parameters.get(key) is not None and parameters.get(key) != ""}
    from ouroboros.route_spec import harness_model_named_effort

    # A level in the model name is the route's identity: the wire never carries a
    # different separate effort beside it (the host's request stays recorded as asked).
    named_effort = harness_model_named_effort(target["source"], target["resolved_model"])
    if named_effort:
        options["reasoningEffort"] = named_effort
    processing = parameters.get("_processing_submission", target.get("processing_preference"))
    if (repair := target.get(RESTORED_PROCESSING_KEY)) and not options.get("serviceTier") and (
            target["usage_model"], parameters.get("model_role"), processing, account.get("profileId", "")) == (
            repair["model"], repair["model_role"], repair["preference"], repair["pin"]):
        if "standard" not in target.get("processing_preferences", []):
            raise ClaudexorModelNotDispatched({"code": "unsupported_parameter", "retryable": False,
                "message": "The model source did not advertise Standard processing; no request was sent.",
                "context": {"parameter": "processingPreference"}},
                model_role=parameters.get("model_role", ""))
        processing = "standard"  # the current intent stays on target, just as on the warm repair
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
        self.last_observation: dict = {}
        self.request_manifest_ref: dict = {}
        self.interrupt_reason = ""
        self.create_attempted = False
        self.create_binding = ""
        self.capture_failure_evidence = False
        self.capture_effort_evidence = False
        self.typed_refusals = False
        self.redos_left = None  # served-model redos its call had left when it was admitted (``SubstitutionBudget.restored``)
        self.auth_reissue_left = None  # its call's one no-start ``auth_changed`` reissue, unused (1) or spent (0), as recorded
        self.auth_reissue_of = parameters.get("_auth_reissue_of") or ""  # the released attempt whose allowance this send spends
        self.accepts_typed_refusals = parameters.get("accepts_typed_refusals") is True
        self.defer_close = False
        self.io_active = False
        self.io_lock = threading.Lock()
        self.outage_episode = None

    def check_control(self, *, starting=False, cancel=True):
        """The caller supplies deadline/cancel policy; this seam transports it. Once a create may have been
        accepted, a helper's send window no longer applies: only the caller's own deadlines and controls end it.
        ``cancel=False``: the operation is already terminal, so a control ends the wait without a pointless cancel."""
        if self.gateway is None and self.retained:
            return  # The cold reader classifies its retained terminal result before applying current controls.
        with accepted_operation_scope() if self.create_attempted else contextlib.nullcontext():
            reason = self.interrupt_reason or (self.poll_control() if self.poll_control else None)
            if not reason:
                waiter = current_model_wait()
                reason = waiter.control_reason() if waiter is not None else None
        if not reason and starting and not self.create_attempted:
            # The first create is this route's send, so a caller's bound holds here even with no wait owner to
            # read it. A repeated create only rejoins its idempotent operation; a running one is never cancelled.
            remaining = dispatch_deadline_remaining_sec()
            if remaining is not None and remaining <= 0:
                raise _PhysicalSendNotStarted("deadline")
        if not reason and starting:
            try:
                if model_handed_off(self.invocation_id):
                    self.create_attempted = True
                else:
                    with launch_admission(current_usage_scope()):
                        self.create_attempted = True
            except OwnerPauseRefused as exc:
                reason = str(exc)  # prior create_attempted remains unknown, never unsent
        if not reason:
            return
        cancellation = "not_requested"  # an owner Pause cancels no sent work: it parks the read (``OWNER_PAUSE_CONTROLS``)
        if cancel and reason not in ("owner_pause", "owner_pause_authority_unreadable") and self.operation_id and self._bound_gateway_current():
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

    def prepare(self, reservation, *, canonical_bytes: bytes, sha256: str, size_bytes: int):
        self.invocation_id = reservation.attempt_id
        self.root = reservation.drive_root
        scope = current_usage_scope()
        self.task_id = scope.task_id if scope else ""
        self.check_control()
        deadline = time.monotonic() + self.timeout
        last_error = None
        try:
            while True:
                self.check_control()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise last_error or ClaudexorUnavailable("daemon_not_answering", "Call deadline elapsed before model creation")
                try:
                    self.gateway = timed("daemon.gateway", model_gateway, lambda: ensure_owned_gateway(
                        admission_wait_sec=remaining, startup_wait_sec=remaining))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ClaudexorUnavailable("daemon_not_answering", "Call deadline elapsed before model creation")
                    operations = timed("daemon.operations", self.gateway.operations, timeout_sec=remaining)
                    break
                except ClaudexorUnavailable as error:
                    # No model create has been attempted. Wait on this call's
                    # own transport deadline; received refusals remain refusals.
                    unanswered = not error.status_code and (error.observation_timeout
                                                            or error.code in DAEMON_UNANSWERED_CODES)
                    if not unanswered or time.monotonic() >= deadline:
                        raise
                    last_error = error
                    self.close()
                    time.sleep(min(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC, max(0, deadline - time.monotonic())))
            # Freeze capabilities before the one create attempt.
            self.capture_failure_evidence = model_failure_evidence_supported(operations)
            self.capture_effort_evidence = operation_query_supported(operations, method="POST",
                path="/v2/model-operations", name="captureEffortEvidence", value="true")
            self.typed_refusals = self.accepts_typed_refusals and operation_query_supported(operations, method="POST",
                path="/v2/model-operations", name="typedRefusals", value="true")
            self.create_binding = model_rejoin_binding(self.gateway)
            self.request_ref = timed("daemon.upload", self.gateway.upload_model_request, canonical_bytes, sha256=sha256,
                                     size_bytes=size_bytes, idempotency_key=self.invocation_id)
            self.request_manifest_ref = persist_call(self.root, task_id=self.task_id, call_id=f"{self.invocation_id}_model_request",
                         call_type="llm_claudexor_request", payload=self.payload, keep_raw=True,
                         manifest={"invocation_id": self.invocation_id, "request_ref": self.request_ref,
                                   "model_role": self.role, "model": self.target["usage_model"],
                                   "model_store_binding": self.create_binding,
                                   "typed_refusals": self.typed_refusals, "substitution_redos_left": self.redos_left,
                                   "auth_reissue_left": self.auth_reissue_left,
                                   "capture_failure_evidence": self.capture_failure_evidence,
                                   "capture_effort_evidence": self.capture_effort_evidence})["manifest_ref"]
            if self.auth_reissue_of:  # spent on the released attempt's own record before this reissue's create: never twice
                update_call_manifest_metadata(self.root, task_id=self.task_id, call_id=f"{self.auth_reissue_of}_model_request",
                                              metadata={"auth_reissue_left": 0})
            host_step("daemon_upload")
        except ClaudexorUnavailable as error:
            raise ClaudexorModelNotDispatched({"code": error.code, "message": str(error)}, model_role=self.role) from None

    def observe_operation(self, *, accepted: bool = False, phase: str = "") -> None:
        """Publish changed custody/activity facts, never account or provider content.

        Dispatch started precedes the engine's flush barrier: it proves neither
        delivered HTTP bytes nor provider reasoning. A control hole rejoins the
        same operation. Activity is presentation only, separate from custody.
        """
        if accepted:
            try:
                # Only this operational request manifest is updated. The
                # ledger's immutable physical-candidate manifest is untouched.
                self.request_manifest_ref = update_call_manifest_metadata(
                    self.root, task_id=self.task_id, call_id=f"{self.invocation_id}_model_request",
                    metadata={"operation_id": self.operation_id,
                              "capture_failure_evidence": self.capture_failure_evidence,
                              "capture_effort_evidence": self.capture_effort_evidence})
            except Exception as error:
                self.request_manifest_ref = {}
                log.warning("Model custody checkpoint unavailable: %s", type(error).__name__)
        if self.operation_observer is not None:
            detail, dispatch = self.detail, (self.detail.get("dispatch") or {}).get("state")
            if not phase:
                phase = ("preparing" if not self.operation_id else
                    "response_ready" if (detail.get("response") or {}).get("state") == "ready" else
                    "queued" if detail.get("state") == "queued" and "startedAt" in detail and detail["startedAt"] is None else
                    "dispatching" if detail.get("state") == "running" and dispatch == "not_started" else
                    "awaiting_response" if detail.get("state") == "running" and dispatch == "started" else "unknown")
            activity = {"phase": phase, "operation_id": self.operation_id, "invocation_id": self.invocation_id}
            admission = detail.get("admission") or {}
            blockers = admission.get("blockers") if isinstance(admission, dict) else None
            if phase == "queued" and isinstance(blockers, list):
                activity["blockers"] = [{"kind": row["kind"]} for row in blockers if isinstance(row, dict)
                    and row.get("kind") in {"global_limit", "class_limit", "thread_busy", "stopping"}]
            observation = {"operation_id": self.operation_id, "invocation_id": self.invocation_id,
                           "request_ref": copy.deepcopy(self.request_ref),
                           "request_manifest_ref": copy.deepcopy(self.request_manifest_ref), "model_activity": activity}
            if observation == self.last_observation:
                return
            try:
                self.operation_observer(copy.deepcopy(observation))
                self.last_observation = observation
            except Exception as error:
                log.warning("Model custody observer unavailable: %s", type(error).__name__)

    def error(self, problem: dict | None, detail: dict | None = None, *, unknown: bool = False, rejoinable: bool = False):
        route = ((detail or {}).get("dispatch") or {}).get("route") or {}
        cls = ClaudexorModelError if unknown else ClaudexorModelNotDispatched
        return cls(problem or {"code": "model_operation_failed", "message": "The engine returned no model result."}, model_role=self.role,
                   operation_id=self.operation_id, route=route, unknown=unknown, same_operation_recoverable=rejoinable)

    def _bound_gateway_current(self) -> bool:
        # An unbound original handle can finish in-process, but is never replaceable.
        return self.gateway is not None and (not self.create_binding
                or self.create_binding == model_rejoin_binding(self.gateway))

    def _check_create_binding(self) -> None:
        if not self._bound_gateway_current() or (self.create_attempted and not self.create_binding):
            self.check_control()  # a concurrent Stop wins without marking a create as attempted
            raise self.error(_MODEL_STORE_UNVERIFIED, unknown=self.create_attempted,
                             rejoinable=self.create_attempted) from None

    def _rejoin_gateway(self, gateway):
        if self.create_binding and self.create_binding == model_rejoin_binding(gateway):
            return gateway
        if gateway is not self.gateway:
            gateway.close()
        raise self.error(_MODEL_STORE_UNVERIFIED, unknown=True, rejoinable=True) from None

    def _unconfirmed_refusal(self):
        # No dispatch release, policy verdict or account penalty without a consistent result.
        return self.error({"code": "malformed_response", "message": "The refusal has no consistent response receipt."},
                          unknown=True, rejoinable=True)

    def receive(self) -> dict:
        from ouroboros.model_concurrency import model_call_slot
        with model_call_slot(self.target.get("usage_model", ""), route=self.target, check=self.check_control):
            raw = self._wait_for_result()
            if raw is None:
                if _explicit_content_refusal(self.detail):
                    raise self._unconfirmed_refusal()
                raise self.error(self.detail.get("problem"), self.detail,
                                 unknown=(self.detail.get("dispatch") or {}).get("state") != "not_started")
            self.retain(raw)
            self.observe_operation(phase="response_received")
            return self._terminal_result(json.loads(raw.decode("utf-8")))

    def _wait_for_result(self, *, read_body=True) -> bytes | None:
        """Follow the original operation through live states and control outages. Once bound, only GET it."""
        outage_started = None
        detail: dict = {}
        while True:
            self.check_control()
            try:
                if not self.operation_id:
                    self.observe_operation()
                    self._check_create_binding()
                    self.check_control(starting=True)
                    detail = timed("daemon.create", self.gateway.create_model_operation, self.request_ref,
                        idempotency_key=self.invocation_id,
                        **({"typed_refusals": True} if self.typed_refusals else {}),
                        **({"capture_failure_evidence": True} if self.capture_failure_evidence else {}),
                        **({"capture_effort_evidence": True} if self.capture_effort_evidence else {}))
                    self.operation_id = detail["id"]
                    host_step("create_operation")
                    self.detail = detail
                    self.observe_operation(accepted=True)
                else:
                    if self.gateway is None:
                        if not self.create_binding:
                            raise self.error(_MODEL_STORE_UNBOUND, unknown=True, rejoinable=True)
                        self.gateway = self._rejoin_gateway(read_owned_gateway())
                    if not self._bound_gateway_current():
                        raise ClaudexorUnavailable(**_MODEL_STORE_UNVERIFIED)
                    detail = self.gateway.get_model_operation(self.operation_id, timeout_sec=min(self.timeout, _READ_TIMEOUT_SEC))
                    if detail.get("id") != self.operation_id:
                        raise self.error({"code": "malformed_response", "message": "The operation ID changed during a read."},
                                         unknown=True)
                self.detail = detail
                self.observe_operation()
                if self.outage_episode is not None:
                    self._control_outage(recovered=True)
                if detail.get("state") not in {"queued", "running"}:
                    response = detail.get("response") or {}
                    if read_body and response.get("state") == "ready":
                        self.response_ref = response["ref"]
                        self.check_control(cancel=False)  # a control met during the status read starts no result read
                        if not self._bound_gateway_current():
                            raise ClaudexorUnavailable(**_MODEL_STORE_UNVERIFIED)
                        return self.gateway.get_model_result(self.operation_id, expected_ref=self.response_ref,
                                                             timeout_sec=min(self.timeout, _READ_TIMEOUT_SEC), raw_bytes=True)
                    return None
                outage_started = None
            except ClaudexorUnavailable as error:
                if not self.operation_id and error.code == "model_request_invalid":
                    # This exact create refusal is minted by the engine only
                    # after its idempotency lookup proves no accepted command,
                    # and before enqueue. A GET or an arbitrary 4xx cannot mint
                    # non-dispatch authority for an earlier accepted request.
                    raise self.error({"code": error.code, "message": str(error),
                                      "context": {"httpStatus": error.status_code}}) from None
                self.observe_operation(phase="rejoining")
                # A failed read after creation is never a provider connect failure.
                # Drop its causal HTTP chain at this boundary: even ConnectError
                # means only the local control read failed, not inference un-sent (a refused read, 404 too).
                if error.code != "daemon_unreachable" and not 500 <= error.status_code < 600:
                    if self._control_outage():  # rejoin by operation id OR the same create idempotency key
                        continue
                    raise self.error({"code": error.code, "message": str(error)}, detail, unknown=True, rejoinable=self.create_attempted) from None
                if outage_started is None:
                    outage_started = time.monotonic()
                if self._control_outage():
                    continue
                if time.monotonic() - outage_started >= self.timeout:
                    raise self.error({"code": "model_control_unreachable", "message": "Control connection lost; the same model operation may still finish."}, detail, unknown=True, rejoinable=True) from None
            time.sleep(min(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC, self.timeout))

    def _terminal_result(self, result: dict) -> dict:
        detail = self.detail
        dispatch = (detail.get("dispatch") or {}).get("state")
        if not isinstance(result, dict):
            raise self.error({"code": "malformed_response", "message": "The engine returned no result object."}, detail, unknown=True)
        explicit_refusal = _explicit_content_refusal(result) or _explicit_content_refusal(detail)
        response_received = dispatch == "response_received" and result.get("outcome") in ("completed", "incomplete", "failed")
        if explicit_refusal and not response_received:
            raise self._unconfirmed_refusal()  # The body was retained before classification.
        if dispatch == "not_started":
            error = self.error(result.get("problem"), detail)
            error.route = copy.deepcopy(result.get("route") or error.route)
            raise error
        if result.get("outcome") == "unknown" or dispatch == "unknown":
            raise self.error(result.get("problem"), detail, unknown=True)
        if not response_received:
            raise self.error({"code": "malformed_response", "message": "The engine did not prove a terminal provider response."}, detail, unknown=True)
        if (not explicit_refusal and result["outcome"] == "failed"
                and self.target.get("processing_preference") in _ADVISORY_TIERS
                and (problem := _processing_refusal_proof(result.get("problem")))
                and _sent_advisory_tier(self.payload)):
            # The response is the refusal envelope, not a generated answer.
            raise ClaudexorModelNotDispatched(problem, model_role=self.role,
                                            operation_id=self.operation_id, route=result.get("route"))
        return result

    def _received_reply_error(self, result: dict) -> ClaudexorModelError | None:
        """Decide delivery after settlement and before any substitution or use of the message."""
        state, outcome = self.detail.get("state"), result["outcome"]
        identity = dict(model_role=self.role, operation_id=self.operation_id, route=result.get("route") or {})
        if public := ClaudexorContentRefused.public_problem(result, self.detail, self.payload.get("source")):
            return ClaudexorContentRefused(public, **identity)
        message = result.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if _explicit_content_refusal(result) or _explicit_content_refusal(self.detail):
            return ClaudexorRefusalError(ClaudexorRefusalError.terminal_problem(), **identity)
        if (outcome in ("completed", "incomplete") and isinstance(content, list)
                and any(isinstance(p, dict) and p.get("type") == "refusal" for p in content)):
            accepted = state == "succeeded" and outcome == "completed" and self.typed_refusals and all(
                isinstance(p, dict) and p.get("type") in ("text", "refusal")
                and isinstance(p.get(p["type"]), str) for p in content)
            if not accepted:
                return ClaudexorRefusalError(ClaudexorRefusalError.terminal_problem(), **identity)
        problem = result.get("problem") or {}
        if state == "cancelled":
            return ClaudexorModelError({"code": "model_operation_cancelled",
                "message": "The engine cancelled this model operation."}, **identity)
        if outcome == "failed":
            return ClaudexorModelError(problem, **identity)
        if not isinstance(message, dict):
            error = ClaudexorModelError(problem or {"code": "response_rejected",
                "message": "The terminal provider response contained no usable model message."}, **identity)
            error.stream_rejected = error.stream_incomplete = True
            return error
        # The engine records a received output cut as interrupted, even though its partial text is
        # available to the loop's length recovery. It cannot authorize an unfinished tool call.
        completed = state == "succeeded" and outcome == "completed"
        output_cut = (state == "interrupted" and outcome == "incomplete" and isinstance(content, str)
                      and not message.get("tool_calls") and problem.get("code") == "provider_incomplete"
                      and (problem.get("context") or {}).get("reason") == "max_output_tokens")
        if (self.operation_id or self.detail) and not (completed or output_cut):
            return ClaudexorModelError({"code": "response_rejected",
                "message": "The recorded operation has no deliverable model message."}, **identity)  # rejected by its code (``__init__``)
        return None

    def redo_substitution(self, result: dict, budget: SubstitutionBudget) -> bool:
        if substitution_fact(result) is None or self._received_reply_error(result) is not None:
            return False
        return budget.admit(self, result)

    def _control_outage(self, *, recovered: bool = False) -> bool:
        """Running turns keep the same accepted operation through control loss."""
        from ouroboros.loop_transport import TransportWaitEpisode, emit_network_wait_event, managed_transport_continuation
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
            previous, self.gateway = self.gateway, self._rejoin_gateway(replacement)
            if previous is not None and previous is not self.gateway:
                previous.close()
        # Unlike an owner-mail peek, check_control's exception must propagate.
        deadline = time.monotonic() + backoff
        while time.monotonic() < deadline:
            self.check_control()
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
                          **_operation_receipt_fields(self.detail)},
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
        if self.gateway is None:  # Cold read from the retained manifest; its reader already handled the engine ACK.
            return {**custody, "state": "retained"}
        if not self._bound_gateway_current():
            return {**custody, "reason": _MODEL_STORE_UNVERIFIED["code"]}
        try:
            receipt = self.gateway.acknowledge_model_result(self.operation_id, self.response_ref["sha256"])
            custody["state"] = (receipt.get("response") or {}).get("state", "pending")
        except Exception as error:
            custody["reason"] = error.code if isinstance(error, ClaudexorUnavailable) else type(error).__name__
        return custody

    def extract_usage(self, result: dict) -> tuple[dict, float | None, bool]:
        usage, cost, final = _usage(result, effort_request_facts(self.target, self.payload))
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

    def finish(self, result: dict, *, recovered: bool = False) -> tuple[dict, dict]:
        reply_error = self._received_reply_error(result)
        if recovered and reply_error is None and (mismatch := substitution_fact(result)):
            reply_error = ClaudexorModelError({  # with the redos its admission recorded left, if any: its round decides them
                "code": "model_substituted", "message": "Another model answered the recorded operation.",
                "context": {"requested_model": mismatch["requested"], "observed_model": mismatch["observed"],
                            **({"redos_left": self.redos_left} if type(self.redos_left) is int else {})}},
                model_role=self.role, operation_id=self.operation_id, route=result.get("route"))
        usage, cost, final = self.extract_usage(result)
        # After the host's Standard retry of a refused tier (``_unstarted_repair``) the engine's receipt answers that
        # Standard request; this logical call still asked the work's tier, which its receipt keeps beside what was
        # sent and served (the physical attempt's own row keeps the engine's receipt).
        wanted, sent = self.target.get("processing_preference"), (self.payload.get("options") or {}).get("processingPreference")
        receipt = usage.get("processing")
        if isinstance(receipt, dict) and wanted and sent and sent != wanted and receipt.get("requested") != wanted:
            usage["processing"] = {**receipt, "requested": wanted, "reason": receipt.get("reason") or "submitted_mode_differs"}
        route = result.get("route") or {}
        usage.pop("wire_layout", None)  # host-owned: the projection fact of THIS call's target
        usage.update(provider="claudexor", resolved_model=self.target["usage_model"], cost=cost, cost_final=final,
                     cost_estimated=cost is not None and not final,
                     **({"wire_layout": dict(self.target["wire_layout"])} if isinstance(self.target.get("wire_layout"), dict) else {}),
                     claudexor={"operation_id": self.operation_id, "model_role": self.role,
                                "requested_profile": str((self.payload.get("account") or {}).get("profileId") or ""),
                                **_dispatch_facts(self.payload, self.detail, result),
                                "route": copy.deepcopy(route), "cost_evidence": copy.deepcopy(result.get("cost")),
                                "operation_state": self.detail.get("state"), "dispatch_state": (self.detail.get("dispatch") or {}).get("state"),
                                "outcome": result.get("outcome"), "problem": copy.deepcopy(
                                    reply_error.problem if reply_error is not None else result.get("problem")),
                                "requested_options": copy.deepcopy(self.payload.get("options") or {}),
                                "applied_options": copy.deepcopy(result.get("appliedOptions")),
                                "applied_options_source": "provider_response",
                                "output_reserve_tokens": self.output_reserve, "output_cap_applied": False,
                                "result_custody": {"state": "pending", "operation_id": self.operation_id,
                                                   "response_ref": self.response_ref,
                                                   "retained_manifest_ref": self.retained.get("manifest_ref")}})
        if reply_error is None and result.get("outcome") == "incomplete":  # recorded before a control can take the receiver:
            usage["response_status"] = "incomplete"  # what it carries away says this reply was cut, never whole
            usage["response_finish_reason"] = (
                "length" if ((result.get("problem") or {}).get("context") or {}).get("reason") == "max_output_tokens"
                else "incomplete"
            )
            observe_finish_reason(usage, usage, "response_finish_reason", computed=True)
        refusal = isinstance(reply_error, ClaudexorRefusalError)
        if reply_error is not None:
            usage["response_status"] = result["outcome"]
            reply_error.physical_attempt_capture = self.capture
            reply_error.ledger_attempt_ids = [self.invocation_id] if self.invocation_id else []
            reply_error.usage = usage
        if refusal:
            # A received refusal is its request's terminal: custody is taken before any control is read, and a
            # control read after it keeps its own exit with the refusal beside it (``received_refusal``) for this
            # request's owner alone: a caption keeps it, the loop ends only its own request on it.
            usage["claudexor"]["result_custody"] = self.acknowledge()
        try:
            self.check_control()
            if not refusal:
                usage["claudexor"]["result_custody"] = self.acknowledge()
                self.check_control()
        except ClaudexorModelError as error:
            # Control changes authority to continue, never ownership of the
            # already received result or its settled usage.
            error.usage = usage
            if reply_error is None:
                error.model_result = copy.deepcopy(result)
            error.route = copy.deepcopy(route)
            if refusal:
                error.received_refusal = reply_error
            raise
        if reply_error is not None:
            raise reply_error
        return copy.deepcopy(result["message"]), usage

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


def _dispatch_facts(payload: dict, detail: dict, result: dict) -> dict:
    """What the engine's own operation record says of this send's account and start, beside the round's receipt: the
    profile the send preferred, the engine's ``routing.selectionReason``, and ``engine_until_dispatch_ms``, the time
    until dispatch started by the engine's record (``startedAt`` to ``dispatch.startedAt``; the dispatch mark precedes
    the vendor request's bytes, so it is no wire time). An absent field adds nothing."""
    from ouroboros.deadline_utils import parse_deadline_ts

    preferred = (payload.get("account") or {}).get("preferredProfileId")
    reason = (result.get("routing") or detail.get("routing") or {}).get("selectionReason")
    started, dispatched = (parse_deadline_ts(value) for value in (detail.get("startedAt"),
                                                                    (detail.get("dispatch") or {}).get("startedAt")))
    return {**({"preferred_profile": str(preferred)} if preferred else {}),
            **({"selection_reason": reason} if isinstance(reason, str) and reason else {}),
            **({"engine_until_dispatch_ms": round((dispatched - started).total_seconds() * 1000)}
               if started and dispatched and dispatched >= started else {})}


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
    requested = payload.get("account") or {}
    append_jsonl(invocation.root / "logs" / "events.jsonl", {
        "ts": utc_now_iso(), "type": "native_continuation_reset", "task_id": invocation.task_id,
        "model_role": invocation.role, "operation_id": invocation.operation_id, "routes": changed,
        "requested_profile": str(requested.get("profileId") or requested.get("preferredProfileId") or ""),
        "observed_profile": str((error.route or {}).get("credentialProfileId") or ""),
        **({"surface": surface} if surface else {}),
    })
    return updated


def _accounted_request(invocation: _ModelInvocation):
    # Every invocation is a new host preparation: Main's clock line, kept by the idempotent upload/rejoin.
    if not invocation.payload.get("retryOfOperationId"):
        invocation.payload = stamp_clock_note(invocation.payload)
    # The engine relays to the vendor: its loopback address is not where the model runs (a not-started receipt is a remote-hop fact).
    request = replace(timed("candidate.measure", _attempt_request, invocation.target, invocation.payload), route_is_loopback=False,
                      force_unknown_reservation=True, max_completion_tokens=invocation.output_reserve)
    before, handed = _candidate_before_dispatch(invocation.payload, request, prepare=invocation.prepare), mark()

    def reserved(reservation):  # the ledger reserved this attempt between the hand-over and this call
        span_since("candidate.reserve", handed)
        return before(reservation)
    return request, reserved


def _unstarted_retry_preparation(target: dict, payload: dict, parameters: dict, error: ClaudexorModelNotDispatched):
    """Rebind the already-authorized un-sent repair before preparing its next attempt.

    The engine's new account receipt replaces provisional discovery. Pass the
    sanitized send copy to ordinary callers; Main re-applies an attested
    continuation reset to its canonical source before rebuilding its vision view.
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


def _processing_refusal_proof(problem):
    if isinstance(problem, dict) and problem.get("code") == "processing_unavailable":
        context = problem.get("context")
        if (isinstance(context, dict) and context.get("generationStarted") is False
                and context.get("processingFallback") == "standard"
                and context.get("processingRefusal") in ("capacity", "unsupported")):
            return problem
    return None


def _sent_advisory_tier(payload):
    options = payload.get("options") or {}
    return options.get("processingPreference") in _ADVISORY_TIERS and not options.get("serviceTier")


def processing_recovery_candidate(row):
    """Cheap filter; only the exact request and terminal body can prove the refusal."""
    return (row.get("processing_preference") in _ADVISORY_TIERS
            and row.get("submitted_processing_mode") in _ADVISORY_TIERS)


def _processing_retry_payload(payload: dict, error: ClaudexorModelNotDispatched) -> dict | None:
    if (getattr(getattr(error, "physical_attempt_capture", None), "state", None) != "released"
            or not _processing_refusal_proof(error.problem) or not _sent_advisory_tier(payload)):
        return None
    updated = copy.deepcopy(payload)
    updated["options"]["processingPreference"] = "standard"
    return updated


def _unstarted_repair(target: dict, payload: dict, parameters: dict, invocation: _ModelInvocation,
                      error: ClaudexorModelNotDispatched, repaired: set) -> tuple[dict, dict, Any] | None:
    """(payload, parameters, its prepared next attempt) for the one repair per cause a released not-started receipt
    permits; a preparation that refuses repairs nothing. ``auth_changed`` (the account's authentication changed before
    dispatch) sends the same request once more as a NEW operation on the same route, never a correlated retry,
    recorded once that attempt is prepared. A repaired cause refused again, an unknown outcome and every other code
    return None: the caller's refusal path."""
    capture = getattr(error, "physical_attempt_capture", None)
    if ("processing" not in repaired and "standard" in target.get("processing_preferences", [])
            and (updated := _processing_retry_payload(payload, error)) is not None):
        cause, repair = "processing", (updated, {**parameters, "_processing_submission": "standard"})
    elif "native" not in repaired and (updated := _reset_native(payload, error, invocation)) is not None:
        cause, repair = "native", (updated, parameters)
    elif ("auth" not in repaired and not parameters.get("_auth_reissue_of") and error.code == "auth_changed"
          and getattr(capture, "state", None) == "released"):
        cause, repair = "auth", (payload, {**parameters, "_auth_reissue_of": invocation.invocation_id})
    else:
        return None
    repaired.add(cause)
    prepared = _unstarted_retry_preparation(target, *repair, error)
    if cause == "auth":
        append_jsonl(invocation.root / "logs" / "events.jsonl", {"ts": utc_now_iso(), "type": "model_operation_reissued",
            "cause": error.code, "task_id": invocation.task_id, "model_role": invocation.role, "operation_id": invocation.operation_id})
    return (*repair, prepared)


class RestoredEnd(Exception):
    """A restored read ended known: its round's own next call meets that end before it sends (``restored_end``)."""


@dataclass(frozen=True)
class RestoredCapture(PhysicalAttemptCapture):
    """The capture of an attempt read back from its record. It is opaque: no record proves its request's bytes or token size
    (the engine's words never state a count), and nothing the round measures now describes it (``restored``)."""

    restored = True


def restored_end(usage: dict, attempt: dict, round_idx: int, error: ClaudexorModelError) -> None:
    """End a restored read on its known ``error`` (``RestoredEnd``): the round's own next call meets it (``restored_round``), so
    the wait, retry, route, size and terminal owners decide it as the live one. A refused advisory tier is the exception: its
    Standard repair is kept as ``RESTORED_PROCESSING_KEY``. The record stays in the saved usage until a call meets it, so a
    checkpoint written between the read and that call (or an exact Pause there) carries it and no restart sends over it.
    No transport repair rides it: a restored ``auth_changed``/``invalid_continuation`` is a typed end like any other, because
    nothing durable proves the one repair its cause permits unused and bound to this operation (a repair's own send carries
    no marker, and every cold entry starts a fresh ``repaired`` set); only the live refusal's one-shot repair stands."""
    released, round_id = isinstance(error, ClaudexorModelNotDispatched), f"{usage.get('execution_id')}:round:{round_idx}"
    if repair := getattr(error, "processing_recovery", None):
        usage[RESTORED_PROCESSING_KEY] = {**repair, "round_id": round_id}
        return
    receipt = (getattr(error, "usage", None) or {}).get("claudexor") or {}
    usage[RESTORED_END_KEY] = {  # ``rejected``: a whole response judged unusable stays that, never a same-model repeat
        **{key: attempt.get(key) for key in ("operation_id", "physical_attempt_id", "model", "model_role")},
        "rejected": bool(getattr(error, "stream_rejected", False)),
        "auth_reissue_left": getattr(error, "auth_reissue_left", None),  # its own record's; absent (older): unproven
        "round_id": round_id, "problem": error.problem, "custody": "released" if released else "settled", "route": error.route,
        "receipt": {key: receipt[key] for key in ("operation_state", "outcome", "requested_profile") if key in receipt},
        "origin": {key: attempt[key] for key in ("llm_call_id", "execution_id", "round_id", "round", "attempt", "request_ref",
                   "attempt_budget", "max_retries") if attempt.get(key) is not None}}  # with the burst it was admitted in
    for key in ("_last_llm_error_kind", "execution_status"):
        usage.pop(key, None)  # the round's own call meets the end and rewrites the rest of the failure row at its head
    raise RestoredEnd


def restored_round(usage: dict, round_id: str) -> dict:
    """What this round's restored read leaves its own call: the kept Standard repair of a proven tier refusal and the known end
    (the saved usage's own record, spent where it is met). Another round's record applies nothing and is dropped."""
    repair, end = usage.get(RESTORED_PROCESSING_KEY) or {}, usage.get(RESTORED_END_KEY) or {}
    if end.get("round_id") != round_id:
        usage.pop(RESTORED_END_KEY, None)
    return {**(repair if repair.get("round_id") == round_id else {}), **({"end": end} if end.get("round_id") == round_id else {})}


def _meet_restored_end(target: dict, parameters: dict) -> ClaudexorModelError | None:
    """Raise, in place of this call's first send, the end its round's restored attempt left, as the live call raised it: every
    owner above sees the exact type, code, status (a whole response judged unusable too) and the receipt's own state and outcome.
    Nothing is sent, no account marked and no money counted again (the read counted it once); another model or role never meets
    it. A call that does not meet it still spends it: the end is the evidence of that one restart, never a standing fact a later
    call of the round could replay. Another model's answer whose admission recorded redos left (``redos_left``) is returned
    instead: this call decides that redo by the live policy (``SubstitutionBudget.restored``) and raises it when refused."""
    spent = (target.get(RESTORED_PROCESSING_KEY) or {}).pop("end", None)
    end = dict(spent or {})
    if spent:
        spent.clear()  # met or not, the saved usage's record is spent only here: an exception before this call keeps it
    if end and (end["model"], end["model_role"]) == (target["usage_model"], parameters.get("model_role")):
        error = (ClaudexorModelNotDispatched if end["custody"] == "released" else ClaudexorModelError)(
            end["problem"], model_role=end["model_role"], operation_id=end["operation_id"], route=end["route"])
        if end.get("rejected"):
            error.stream_rejected = error.stream_incomplete = True
        if end.get("receipt"):
            error.usage = {"claudexor": dict(end["receipt"])}  # state, outcome, profile: no counters, so nothing counts twice
        capture = error.physical_attempt_capture = RestoredCapture(  # the last capture is the restored attempt's own,
            end["physical_attempt_id"], end["model"], "claudexor", end["custody"], "opaque")  # never an earlier round's
        error.ledger_attempt_ids = [end["physical_attempt_id"]]
        error.restored_origin = dict(end.get("origin") or {})  # the call and request its own record names (``restored_view``)
        adopt_physical_attempt_capture(capture)
        left = (error.problem.get("context") or {}).get("redos_left") if error.code == "model_substituted" else None
        if type(left) is int and left > 0:
            return error
        if error.code == "auth_changed" and end["custody"] == "released" and end.get("auth_reissue_left") == 1:
            parameters["_auth_reissue_of"] = end["physical_attempt_id"]  # its own record proves the one reissue unused:
            return None  # this call's first send is that reissue, a new operation on the same route, recorded spent
        raise error
    return None


def restored_burst(usage: dict, round_id: str, model: str, role: str, budget: int, max_retries: int) -> tuple[int, int, int]:
    """``(start, budget, max_retries)`` of the call that meets this round's restored end (``call_llm_with_retry``): the burst its
    own attempt (``origin``) was admitted in, recorded before dispatch, within today's and never beyond it, and that attempt's
    place there, so a temporary failure repeats only what its live burst had left and a raised setting renews none. A record that
    names no place or burst proves no unused attempt: the last place. With no end of this round, model and role: unchanged."""
    end = usage.get(RESTORED_END_KEY) or {}
    if end.get("round_id") != round_id or (end.get("model"), end.get("model_role")) != (model, role):
        return 0, budget, max_retries
    ordinal, admitted, retries = ((end.get("origin") or {}).get(key) for key in ("attempt", "attempt_budget", "max_retries"))
    if not (type(ordinal) is int and ordinal > 0 and type(admitted) is int and admitted > 0 and type(retries) is int and retries >= 0):
        return budget - 1, budget, max_retries
    budget, max_retries = min(budget, admitted), min(max_retries, retries)
    return min(ordinal, budget) - 1, budget, max_retries


def restored_view(error: Exception, identity: dict, fit: dict, request_ref: Any) -> tuple:
    """``(identity, fit, request_ref, facts)`` an error event of ``error`` shows, ``facts`` what its failure row adds. A restored
    attempt's end names the call and request its own record names (``restored_fields`` lists them); a field the record names none
    of keeps the evaluating call's own value, the call that also stays as ``current_*``, never an invented origin; none of this
    unsent call's fit facts (its estimate is not that request's size). Its row is a restored attempt's, and a settled one keeps
    its own operation (``recovered_operation_id``): the no-call rail buys no request over that read (``provider_no_call_source``).
    An accepted operation's receipt judged unusable (``stream_rejected``), live or restored, adds ``rejected_receipt``: its own
    known end, which no configured route, wrap-up or grace sends over. Any other error shows what it was given and adds nothing."""
    origin = getattr(error, "restored_origin", None)
    rejected = {"rejected_receipt": True} if isinstance(error, ClaudexorModelError) and getattr(error, "stream_rejected", False) else {}
    if origin is None:
        return identity, fit, request_ref, rejected if getattr(error, "operation_id", "") else {}
    named = {key: origin[key] for key in ("execution_id", "round_id", "llm_call_id", "round", "attempt") if origin.get(key) is not None}
    return ({**identity, **named, "current_llm_call_id": identity["llm_call_id"], "current_attempt": identity["attempt"],
             "restored_fields": sorted(named)}, {}, origin.get("request_ref"),
            {"restored_attempt": True, **({} if isinstance(error, ClaudexorModelNotDispatched) else
                                          {"recovered_operation_id": str(getattr(error, "operation_id", "") or "")}),
             **rejected})


def chat_claudexor(target: dict, messages: list, tools: list | None, **parameters: Any) -> tuple[dict, dict]:
    """One generation, one released no-start repair per cause, quota re-asks on Auto."""
    redo = _meet_restored_end(target, parameters)  # a known end raises; another model's answer with redos left returns
    target = prepare_processing_target(target)
    payload = _request(target, messages, tools, parameters)
    retry_preparation, repaired = None, set()
    substitution, rotation, capability = SubstitutionBudget(ClaudexorModelError), AccountRotation(), CapabilityRetry()
    if redo is not None:  # the live policy decides it on this request; raised when refused, else the round is asked again
        substitution.restored(redo, payload)
        parameters = {**parameters, "_no_account_preference": True}
        payload = _request(target, payload["messages"], payload["tools"], parameters)
    all_operations_not_started = True
    for _preparation in range(5 + substitution.redos + rotation.CEILING):
        invocation = _ModelInvocation(target, payload, parameters)
        # Recorded with its request by admission's own rule (``restored``); a correlated retry is its call's last send.
        invocation.redos_left = 0 if capability.original is not None else substitution.allowance(payload)
        invocation.auth_reissue_left = 0 if "auth" in repaired or invocation.auth_reissue_of else 1
        try:
            with prepared_call_scope(retry_preparation or {}) as prepared:
                if prepared:
                    invocation.payload = payload = _request(target, prepared["messages"], prepared.get("tools"), prepared)
                request, before = _accounted_request(invocation)
                result = execute_physical_attempt(request, invocation.receive, extractor=invocation.extract_usage, before_dispatch=before,
                                                  late_owner=_usage_wait.late_transport_custody(invocation))
                all_operations_not_started = False  # even a discarded substituted response ran
                invocation.capture = last_physical_attempt_capture()
                if capability.original is not None:
                    substitution.redos = substitution.used  # the correlated operation is the last send
                if invocation.redo_substitution(result, substitution):
                    # Every later request in this round leaves account selection to the engine.
                    retry_preparation, parameters = None, {**parameters, "_no_account_preference": True}
                    payload = _request(target, payload["messages"], payload["tools"], {**(prepared or parameters), "_no_account_preference": True})
                    continue
                adopt_turn_state((prepared or parameters).get("model_turn_state"), invocation.payload, result)
                return capability.disclose(rotation.disclose(substitution.disclose(invocation.finish(result))))
        except ClaudexorModelNotDispatched as error:
            all_operations_not_started &= getattr(getattr(error, "physical_attempt_capture", None), "state", None) == "released"
            error.presence_all_operations_not_started = all_operations_not_started
            if invocation.response_ref:
                invocation.acknowledge()
            updated = capability.refused(invocation, error)
            if updated is not None:
                retry_preparation, payload = None, updated
                continue
            if capability.ends_chain(error):
                raise
            repair = _unstarted_repair(target, payload, parameters, invocation, error, repaired)
            if repair is None:
                rotation.refused_call(target, parameters, invocation, error)  # an engine verdict is never re-asked
                raise
            payload, parameters, retry_preparation = repair
        except ClaudexorModelError as error:
            all_operations_not_started = False
            if isinstance(error, ClaudexorRefusalError) or getattr(error, "received_refusal", None) is not None:
                raise
            updated = capability.refused(invocation, error)
            if updated is not None:
                retry_preparation, payload = None, updated
                continue
            if capability.ends_chain(error):
                raise
            if not rotation.refused_call(target, parameters, invocation, error):
                raise
            retry_preparation, parameters, payload = rotation.reask(parameters, payload)
        except PhysicalAttemptPreparationFailed as error:
            cause = error.__cause__
            if isinstance(cause, ClaudexorModelError):
                cause.physical_attempt_capture = error.physical_attempt_capture
                all_operations_not_started &= (isinstance(cause, ClaudexorModelNotDispatched)
                                               and error.physical_attempt_capture.state == "released")
                cause.presence_all_operations_not_started = all_operations_not_started
                raise cause from None
            capability.unsent(error)
            raise
        finally:
            _usage_wait.close_unless_abandoned(invocation)  # an owner Pause's abandoned wait: the sender closes it
    raise AssertionError("Unreachable model preparation loop")


def recover_model_attempt(drive_root, row: dict, *, gateway_factory=None):
    """Reconcile billing from the recorded operation, without submitting work."""
    receipt = _read_model_receipt(drive_root, row, gateway_factory=gateway_factory)
    return receipt[0] if receipt else None


def _read_model_receipt(drive_root, row, *, gateway_factory=None, invocation=None):
    """Use retained terminal facts first. Billing probes once; a restored chat waits on its bound operation."""
    from ouroboros.delegate_custody import TERMINAL_STATES
    from ouroboros.observability import call_manifest_path, read_call_payload
    from ouroboros.usage_ledger import is_abandoned_settlement
    task_id, attempt_id = str(row.get("task_id") or ""), str(row.get("attempt_id") or "")
    call_id = f"{attempt_id}_model_response"
    manifest, payload, ref = {}, {}, {}
    try:
        manifest, payload, ref = read_call_payload(drive_root, task_id=task_id, call_id=call_id)
    except FileNotFoundError:
        pass
    if manifest and manifest.get("invocation_id") != attempt_id:
        return None
    request = read_json_dict(call_manifest_path(drive_root, task_id, f"{attempt_id}_model_request")) or {}
    if request and (request.get("invocation_id") != attempt_id or request.get("task_id") != task_id):
        return None
    operation_id = manifest.get("operation_id") or request.get("operation_id")
    if (not operation_id or (invocation is not None and invocation.operation_id != operation_id)
            or (request.get("operation_id") and request["operation_id"] != operation_id)):
        return None
    operation_state, dispatch_state = manifest.get("operation_state"), manifest.get("dispatch_state")
    image = request.get("request_kind") == "image"
    raw = payload.get("result_json_utf8") if isinstance(payload, dict) else None
    detail_refused = (manifest.get("operation_problem_code") == ClaudexorContentRefused.terminal_code
                      or isinstance(payload, dict) and _explicit_content_refusal(payload.get("operation")))
    if operation_state in TERMINAL_STATES and dispatch_state == "not_started" and detail_refused:
        return ("abandoned", {}, None, False), None
    gateway = None
    try:
        if (operation_state not in TERMINAL_STATES or not dispatch_state
                or (raw is None and (image or dispatch_state == "response_received"))):
            if row.get("state") == "released":  # an old release buys no network read
                return ("released", {}, None, False), None
            binding = request.get("model_store_binding")
            if invocation is None:
                if not image and not binding:
                    raise ClaudexorUnavailable(**_MODEL_STORE_UNBOUND)
                gateway = (gateway_factory or read_owned_gateway)()
                if not image and binding != model_rejoin_binding(gateway):
                    raise ClaudexorUnavailable(**_MODEL_STORE_UNVERIFIED)
                detail = gateway.get_model_operation(operation_id, timeout_sec=_READ_TIMEOUT_SEC)
            else:
                received = invocation._wait_for_result(read_body=raw is None)
                gateway, detail = invocation.gateway, invocation.detail
                if received is not None:
                    raw = received.decode("utf-8")
            operation_state = detail.get("state")
            dispatch_state = (detail.get("dispatch") or {}).get("state")
            if operation_state not in TERMINAL_STATES:
                return None
            response = detail.get("response") or {}
            detail_refused = _explicit_content_refusal(detail)
            response_ref = response.get("ref") or manifest.get("response_ref") or {}
            if invocation is None and raw is None and response.get("state") == "ready":
                if not image and binding != model_rejoin_binding(gateway):
                    raise ClaudexorUnavailable(**_MODEL_STORE_UNVERIFIED)
                raw = gateway.get_model_result(operation_id, expected_ref=response_ref,
                                               timeout_sec=_READ_TIMEOUT_SEC, raw_bytes=True).decode("utf-8")
            payload = {"result_json_utf8": raw} if raw is not None else {"operation": detail}
            manifest = {"operation_id": operation_id, "invocation_id": attempt_id,
                        "response_ref": response_ref, **_operation_receipt_fields(detail),
                        "request_kind": request.get("request_kind", "chat")}
            ref = persist_call(drive_root, task_id=task_id, call_id=call_id, call_type="llm_claudexor_response",
                               payload=payload, keep_raw=True, manifest=manifest)
            # Image ACK belongs to its artifact consumer AFTER ledger reconciliation.
            # Terminal maintenance retains the receipt but cannot claim artifact custody.
            if (not image and raw is not None and response.get("state") == "ready"
                    and binding == model_rejoin_binding(gateway)):
                try:
                    gateway.acknowledge_model_result(operation_id, response_ref["sha256"])
                except Exception:
                    log.debug("Recovered model response retained without ACK", exc_info=True)
        if dispatch_state == "not_started":
            body = json.loads(raw) if raw is not None and not image else payload.get("operation") if isinstance(payload, dict) else None
            if detail_refused or _explicit_content_refusal(body):
                return ("abandoned", {}, None, False), None
            typed = isinstance(body, dict) and isinstance(body.get("problem"), dict)  # the engine's own refusal rides its release
            return ("released", {}, None, False), ((manifest, body, ref) if typed else None)
        if raw is None or (not image and dispatch_state != "response_received"):
            return ("abandoned", {}, None, False), None
        result = json.loads(raw)
        outcomes = {"completed", "failed", "unknown"} if image else {"completed", "incomplete", "failed"}
        if not isinstance(result, dict) or result.get("outcome") not in outcomes:
            return ("abandoned", {}, None, False), None
        receipt = manifest, result, ref
        if (not image and not detail_refused and result["outcome"] == "failed"
                and processing_recovery_candidate(row)
                and manifest.get("model_role", request.get("model_role")) == request.get("model_role")
                and (row.get("state") != "settled" or is_abandoned_settlement(row))
                and _processing_refusal_proof(result.get("problem"))):
            sent = invocation.payload if invocation is not None else read_call_payload(
                drive_root, task_id=task_id, call_id=f"{attempt_id}_model_request")[1]
            if _sent_advisory_tier(sent):
                return ("released", {}, None, False), receipt
        usage, cost, final = _usage(result, row.get("effort"), request_kind=request.get("request_kind", "chat"))
        return ("settled", usage, cost, final), receipt
    finally:
        if invocation is not None:
            invocation.close()
        elif gateway is not None and gateway_factory is None:
            gateway.close()


def recover_chat_reply(root, task_id, pending, turn_state, *, poll_control=None):
    """Read the ready checkpoint's original chat attempt. This path owns no new send allowance."""
    from ouroboros import usage_accounting, usage_store
    from ouroboros.observability import call_manifest_path, read_call_payload
    from ouroboros.transport_custody import release_pre_dispatch_attempt
    from ouroboros.usage_ledger import is_abandoned_settlement

    attempt, operation = pending["physical_attempt_id"], pending["operation_id"]
    with usage_store.read(root) as txn:
        row = txn.attempt(attempt)
    if (not row or row["task_id"] != task_id or row["provider"] != "claudexor"
            or row["state"] not in {"dispatched", "unresolved", "settled", "released"}
            or (pending.get("model") and row["model"] != pending["model"])):
        return None
    unsent = ProviderNotDispatched("The recorded chat operation ended before dispatch.")
    retained = read_json_dict(call_manifest_path(root, task_id, f"{attempt}_model_response")) or {}
    if row["state"] == "released" and not (processing_recovery_candidate(row) or retained.get("dispatch_state") == "not_started"):
        raise unsent  # no retained receipt proves a reason, and an old release buys no network read
    if not operation:
        return None
    request, payload, _ = read_call_payload(root, task_id=task_id, call_id=f"{attempt}_model_request")
    if (request.get("invocation_id") != attempt or request.get("operation_id") not in (None, "", operation)
            or request.get("model_role") != pending.get("model_role")
            or request.get("request_kind", "chat") != "chat"):
        return None
    target = {"usage_model": row["model"], "source": payload["source"], "resolved_model": payload["model"],
              "processing_preference": row.get("processing_preference")}
    invocation = _ModelInvocation(target, payload, {"model_role": request.get("model_role"),
        "max_tokens": row.get("max_completion_tokens"), "model_poll_control": poll_control})
    invocation.root, invocation.task_id = root, task_id
    invocation.invocation_id, invocation.operation_id, invocation.create_attempted = attempt, operation, True
    invocation.create_binding = request.get("model_store_binding") or ""
    receipt = _read_model_receipt(root, row, invocation=invocation)
    if receipt is None:
        return None
    (disposition, counters, cost, final), response = receipt
    if disposition not in {"settled", "released"}:
        return None
    if row["state"] in {"dispatched", "unresolved"} or is_abandoned_settlement(row):
        reservation = usage_accounting.AttemptReservation(attempt, root, row["model"], row["provider"],
            row.get("reservation_upper_bound_usd"), row.get("processing_preference") or "",
            row.get("submitted_processing_mode") or "", row.get("processing_basis"))
        if disposition == "settled":
            usage_accounting.settle_attempt(reservation, counters, cost_usd=cost, cost_final=final, expected_revision=row["revision"])
        else:
            release_pre_dispatch_attempt(reservation, unsent, expected_revision=row["revision"])
        with usage_store.read(root) as txn:
            row = txn.attempt(attempt)  # Another writer may have won the revision check.
    if not row or row["state"] != disposition or is_abandoned_settlement(row):
        return None
    if disposition == "released":
        if response is not None:  # the engine's typed refusal of this unsent operation, as its live call raised it
            manifest, result, _ = response
            if manifest.get("model_role", invocation.role) == invocation.role:
                error = ClaudexorModelNotDispatched(result["problem"], model_role=invocation.role,
                    operation_id=operation, route=result.get("route"))
                error.auth_reissue_left = request.get("auth_reissue_left")  # this attempt's own record; absent: None
                if manifest.get("dispatch_state") != "not_started":  # a refused advisory tier, proved: Standard repairs it
                    error.processing_recovery = {"model": row["model"], "model_role": invocation.role,
                        "preference": row["processing_preference"], "pin": (payload.get("account") or {}).get("profileId", "")}
                raise error
        raise unsent
    manifest, result, ref = response
    if manifest.get("model_role", request.get("model_role")) != request.get("model_role"):
        return None
    invocation.detail = {"state": manifest["operation_state"],
                         "problem": {"code": manifest.get("operation_problem_code")},
                         "dispatch": {"state": manifest["dispatch_state"], "route": manifest.get("dispatch_route")},
                         "response": {"state": "ready", "ref": manifest["response_ref"]}}
    invocation.typed_refusals = request.get("typed_refusals") is True
    invocation.redos_left = request.get("substitution_redos_left")  # absent from an older record: no unused redo proved
    invocation.retained = ref
    invocation.response_ref = manifest["response_ref"]
    reply, usage = invocation.finish(result, recovered=True)
    usage["ledger_attempt_ids"] = [attempt]
    adopt_turn_state(turn_state, payload, result)
    return reply, usage, ref


async def chat_claudexor_async(target: dict, messages: list, tools: list | None, **parameters: Any) -> tuple[dict, dict]:
    """Offload I/O; accounting follows sender custody through cancellation or Pause."""
    target = (await asyncio.to_thread(prepare_processing_target, target)
              if target.get("processing_preference") and "processing_preferences" not in target else target)
    payload = _request(target, messages, tools, parameters)
    retry_preparation, repaired = None, set()
    substitution, rotation, capability = SubstitutionBudget(ClaudexorModelError), AccountRotation(), CapabilityRetry()
    all_operations_not_started = True
    for _preparation in range(5 + substitution.redos + rotation.CEILING):
        invocation = _ModelInvocation(target, payload, parameters)
        invocation.redos_left = 0 if capability.original is not None else substitution.allowance(payload)
        invocation.auth_reissue_left = 0 if "auth" in repaired or invocation.auth_reissue_of else 1
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
                    request, receive, extractor=invocation.extract_usage, before_dispatch=prepare,
                    late_owner=_usage_wait.late_transport_custody(invocation))
                all_operations_not_started = False
                invocation.capture = last_physical_attempt_capture()
                if capability.original is not None:
                    substitution.redos = substitution.used
                if await invocation.offload(invocation.redo_substitution, result, substitution):
                    # Every later request in this round leaves account selection to the engine.
                    retry_preparation, parameters = None, {**parameters, "_no_account_preference": True}
                    payload = _request(target, payload["messages"], payload["tools"], {**(prepared or parameters), "_no_account_preference": True})
                    continue
                adopt_turn_state((prepared or parameters).get("model_turn_state"), invocation.payload, result)
                return capability.disclose(rotation.disclose(substitution.disclose(await invocation.offload(invocation.finish, result))))
        except ClaudexorModelNotDispatched as error:
            all_operations_not_started &= getattr(getattr(error, "physical_attempt_capture", None), "state", None) == "released"
            error.presence_all_operations_not_started = all_operations_not_started
            if invocation.response_ref:
                await invocation.offload(invocation.acknowledge)
            updated = await invocation.offload(capability.refused, invocation, error)
            if updated is not None:
                retry_preparation, payload = None, updated
                continue
            if capability.ends_chain(error):
                raise
            repair = _unstarted_repair(target, payload, parameters, invocation, error, repaired)
            if repair is None:
                rotation.refused_call(target, parameters, invocation, error)  # an engine verdict is never re-asked
                raise
            payload, parameters, retry_preparation = repair
        except ClaudexorModelError as error:
            all_operations_not_started = False
            if isinstance(error, ClaudexorRefusalError) or getattr(error, "received_refusal", None) is not None:
                raise
            updated = await invocation.offload(capability.refused, invocation, error)
            if updated is not None:
                retry_preparation, payload = None, updated
                continue
            if capability.ends_chain(error):
                raise
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
            capability.unsent(error)
            raise
        finally:
            if not invocation.defer_close:
                _usage_wait.close_unless_abandoned(invocation)
    raise AssertionError("Unreachable model preparation loop")
