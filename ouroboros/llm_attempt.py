"""Physical-attempt candidates and send-time prompt-cache policy.

One provider send is a *candidate*: the exact payload object that goes on the
wire. This module owns the facts every lane must state about that object —
the send copy, its canonical digest, the accounting request built from it, the
durable candidate manifest written before dispatch — plus the finalizer that
decides what cache markers the assembled payload actually ships with, since the
finalizer is the only point that sees tools, system and messages together.
Structured provider-overflow predicates live here too: they read the same
candidate-attached facts rather than provider prose.
"""


from __future__ import annotations

import copy
import hashlib
import inspect
import json
import threading
from typing import Any, Dict, List, Optional, Set

from ouroboros.context_budget import CONTEXT_OVERFLOW_CODES
from ouroboros.usage_accounting import (
    AttemptRequest,
    PhysicalAttemptPreconditionFailed,
    PhysicalAttemptPreparationFailed,
    current_physical_attempt_context,
    current_physical_attempt_predicate,
    current_usage_scope,
    execute_physical_attempt,
    execute_physical_attempt_async,
)


# Provider-valid Anthropic ephemeral-cache tiers.
_VALID_CACHE_TTLS = frozenset({"5m", "1h"})


# Only explicit wire tiers have a knowable horizon; bare "default" does not.
_CACHE_TTL_SECONDS = {"5m": 300, "1h": 3600}


PROVIDER_POLICY_REFUSAL = "provider_policy_refusal"


class ProviderPolicyRefusal(RuntimeError):
    """Typed refusal: a policy layer would not let this call reach a provider.

    Not a provider failure — nothing upstream answered — so no rung of the
    recovery ladder can repair it: dropping a parameter, rerouting the endpoint
    or stripping replayed reasoning all re-attempt a call that was refused, and
    the caller ends up seeing whatever the re-attempt produced instead of the
    refusal. It carries the machine-readable ``code`` so the ladder can classify
    it structurally, exactly as the subscription-window refusal is classified in
    ``loop_llm_call.classify_llm_exception`` — never by matching prose.

    A transport that cannot import this class states the same fact by setting
    ``code`` to :data:`PROVIDER_POLICY_REFUSAL` on its own exception type; a
    family of refusals (connection not permitted, egress denied, tenant blocked)
    either subclasses this or carries the same code.
    """

    code = PROVIDER_POLICY_REFUSAL


def _is_provider_policy_refusal(exc: BaseException) -> bool:
    """Structural test: a typed refusal, by class or by the declared ``code``."""
    return isinstance(exc, ProviderPolicyRefusal) or (
        str(getattr(exc, "code", "") or "") == PROVIDER_POLICY_REFUSAL
    )


def _structured_error_values(payload: Any) -> Set[str]:
    if not isinstance(payload, dict):
        return set()
    nodes = [payload]
    if isinstance(payload.get("error"), dict):
        nodes.append(payload["error"])
    return {
        str(node.get(key) or "").strip().lower()
        for node in nodes
        for key in ("code", "type")
        if str(node.get(key) or "").strip()
    }


def _is_structured_context_overflow_exception(exc: BaseException) -> bool:
    """Read only facts attached to this exception; never a stale ContextVar."""
    values = {
        str(getattr(exc, key, "") or "").strip().lower()
        for key in ("code", "type")
        if str(getattr(exc, key, "") or "").strip()
    }
    values.update(_structured_error_values(getattr(exc, "body", None)))
    capture = getattr(exc, "physical_attempt_capture", None)
    if capture is not None:
        values.update({
            str(getattr(capture, key, "") or "").strip().lower()
            for key in ("provider_code", "provider_error_type")
            if str(getattr(capture, key, "") or "").strip()
        })
    return bool(values & CONTEXT_OVERFLOW_CODES)


def _is_structured_context_overflow_body(error: Any) -> bool:
    return bool(_structured_error_values(error) & CONTEXT_OVERFLOW_CODES)


def cache_ttl_seconds(applied_ttl: Any) -> Optional[int]:
    """Return seconds only for an explicit TTL carried by the candidate."""
    return _CACHE_TTL_SECONDS.get(str(applied_ttl or "").strip().lower())


def supports_message_cache_control(model: str) -> bool:
    """Whether the OpenRouter family honors message cache breakpoints."""
    m = str(model or "").strip().lstrip("~")
    return m.startswith("anthropic/") or m.startswith("google/gemini-")


def _route_normalizes_cache_breakpoints(target: Dict[str, Any]) -> bool:
    """Whether the send-time finalizer may normalize cache breakpoints."""
    if str(target.get("provider") or "") == "anthropic":
        return True
    model = str(target.get("resolved_model") or "").strip().lstrip("~")
    return bool(
        target.get("supports_openrouter_extensions")
        and supports_message_cache_control(model)
        and model.startswith("anthropic/")
    )


def _applied_payload_cache_ttl(payload: Dict[str, Any]) -> Optional[str]:
    """Strongest cache TTL carried by THIS exact candidate payload.

    Same reporting rule as the send-time finalizer's return value
    (``_normalize_payload_cache_ttl``: 1h > 5m > bare markers = "default";
    None when the payload carries no markers). Read per candidate rather than
    plumbed from the finalizer because the retry ladder can strip markers
    (``_retry_without_prompt_cache_parameter``) after the finalizer ran — the
    reservation must price the payload actually being sent, not the original.
    """
    breakpoints = _PayloadCachePolicyMixin._payload_cache_breakpoints(payload)
    ttls = {
        str((holder.get("cache_control") or {}).get("ttl") or "").strip().lower()
        for holder in breakpoints
    }
    if "1h" in ttls:
        return "1h"
    if "5m" in ttls:
        return "5m"
    return "default" if breakpoints else None


def _attempt_request(
    target: Dict[str, Any],
    payload: Dict[str, Any],
    *,
    source: Optional[str] = None,
) -> AttemptRequest:
    """Build secret-free facts for one final inspectable candidate."""
    prompt_payload = {
        key: value
        for key, value in payload.items()
        if key not in {
            "model", "max_tokens", "max_completion_tokens", "temperature",
            "top_p", "top_k", "timeout", "stream",
        }
    }
    try:
        prompt_chars = len(json.dumps(prompt_payload, ensure_ascii=False, default=str))
    except Exception:
        prompt_chars = len(str(prompt_payload or ""))
    request_source = source
    if request_source is None:
        bound_scope = current_usage_scope()
        request_source = (
            str(bound_scope.source)
            if bound_scope is not None and bound_scope.source
            else "llm.chat"
        )
    raw = _canonical_candidate_bytes(payload)
    context = _canonical_candidate_bytes({
        key: payload[key] for key in ("system", "messages", "tools", "functions") if key in payload
    })
    return AttemptRequest(
        model=str(target.get("usage_model") or target.get("resolved_model") or payload.get("model") or ""),
        provider=str(target.get("provider") or "unknown"),
        prompt_tokens_estimate=max(0, prompt_chars // 4),
        max_completion_tokens=int(payload.get("max_completion_tokens") or payload.get("max_tokens") or 0),
        source=str(request_source or ""),
        prompt_cache_ttl=_applied_payload_cache_ttl(payload) or "",
        candidate_raw_sha256=hashlib.sha256(raw).hexdigest(),
        candidate_raw_size_bytes=len(raw),
        candidate_context_sha256=hashlib.sha256(context).hexdigest(),
        candidate_context_size_bytes=len(context),
        candidate_measurement_kind="canonical_json_v1",
        physical_context=current_physical_attempt_context(),
    )


def _canonical_candidate_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")


def _physical_candidate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the send copy with capsule metadata removed only from context turns."""
    candidate = copy.deepcopy(payload)

    def _strip(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("_context_capsule", None)
            for child in value.values():
                _strip(child)
        elif isinstance(value, list):
            for child in value:
                _strip(child)

    for key in ("system", "messages"):
        _strip(candidate.get(key))
    return candidate


def _candidate_before_dispatch(candidate: Dict[str, Any], request: AttemptRequest):
    """Close over one final candidate without putting it in accounting rows."""
    predicate = current_physical_attempt_predicate()

    def _persist(reservation):
        fresh = _attempt_request(
            {"provider": request.provider, "usage_model": request.model}, candidate,
            source=request.source,
        )
        identity = (
            fresh.candidate_raw_sha256, fresh.candidate_raw_size_bytes,
            fresh.candidate_context_sha256, fresh.candidate_context_size_bytes,
        )
        expected = (
            request.candidate_raw_sha256, request.candidate_raw_size_bytes,
            request.candidate_context_sha256, request.candidate_context_size_bytes,
        )
        if identity != expected:
            raise PhysicalAttemptPreparationFailed(
                "physical candidate changed before dispatch", attempt_id=reservation.attempt_id,
            )
        from ouroboros.observability import persist_physical_candidate

        scope = current_usage_scope()
        persisted = persist_physical_candidate(
            reservation.drive_root,
            task_id=str(scope.task_id if scope is not None else request.task_id),
            attempt_id=reservation.attempt_id,
            candidate=candidate,
            candidate_facts={
                "candidate_raw_sha256": request.candidate_raw_sha256,
                "candidate_raw_size_bytes": request.candidate_raw_size_bytes,
                "candidate_context_sha256": request.candidate_context_sha256,
                "candidate_context_size_bytes": request.candidate_context_size_bytes,
                "candidate_measurement_kind": request.candidate_measurement_kind,
                "physical_context": (
                    dict(vars(request.physical_context)) if request.physical_context is not None else None
                ),
            },
        )
        if predicate is not None:
            try:
                accepted = predicate(request)
            except BaseException as exc:
                # Persistence already succeeded. Preserve the only durable link
                # even when the host predicate itself raises before returning.
                try:
                    exc.candidate_manifest_ref = persisted["manifest_ref"]
                except Exception:
                    pass
                raise
            if accepted is False:
                failure = PhysicalAttemptPreconditionFailed(
                    "physical candidate precondition rejected dispatch",
                    attempt_id=reservation.attempt_id,
                )
                failure.candidate_manifest_ref = persisted["manifest_ref"]
                raise failure
        return persisted["manifest_ref"]

    return _persist


def _execute_candidate(request: AttemptRequest, send: Any, before_dispatch: Any) -> Any:
    """Keep existing two-argument injected executors usable."""
    if "before_dispatch" not in inspect.signature(execute_physical_attempt).parameters:
        return execute_physical_attempt(request, send)
    return execute_physical_attempt(request, send, before_dispatch=before_dispatch)


async def _execute_candidate_async(request: AttemptRequest, send: Any, before_dispatch: Any) -> Any:
    if "before_dispatch" not in inspect.signature(execute_physical_attempt_async).parameters:
        return await execute_physical_attempt_async(request, send)
    return await execute_physical_attempt_async(request, send, before_dispatch=before_dispatch)


class _PayloadCachePolicyMixin:
    """Send-time cache policy on the fully assembled payload."""

    # Anthropic accepts at most four declared cache breakpoints per request.
    _MAX_CACHE_BREAKPOINTS = 4

    @staticmethod
    def _payload_cache_breakpoints(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Blocks carrying a ``cache_control`` marker, in the real wire prefix order
        ``tools -> system -> messages`` — NOT the order arguments happen to arrive in.

        Descends one level INTO a block's own ``content`` list: a direct-Anthropic
        ``tool_result`` block nests its blocks (``_anthropic_messages`` builds it from a
        ``role="tool"`` message whose content is a list), so the sealed transcript anchor
        (``loop.seal_task_transcript``) sits at ``messages[i].content[j].content[k]``.
        Missing it undercounts the cap and leaves that anchor out of TTL ordering exactly
        on the lane whose provider enforces both. ``tool_result`` is the only nested-content
        shape, and the descent is route-independent because no other payload nests."""
        holders: List[Dict[str, Any]] = []
        for key in ("tools", "system", "messages"):
            part = payload.get(key)
            for item in (part if isinstance(part, list) else [part]):
                if not isinstance(item, dict):
                    continue
                if isinstance(item.get("cache_control"), dict):
                    holders.append(item)
                content = item.get("content")
                if isinstance(content, list):
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if isinstance(block.get("cache_control"), dict):
                            holders.append(block)
                        nested = block.get("content")
                        if isinstance(nested, list):
                            holders.extend(
                                inner for inner in nested
                                if isinstance(inner, dict)
                                and isinstance(inner.get("cache_control"), dict)
                            )
        return holders

    def _normalize_payload_cache_ttl(
        self,
        target: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Optional[str]:
        """Finalize cache policy on the FULLY ASSEMBLED payload; report its strongest TTL.

        The one point where tools, system and messages coexist, hence the single home for
        send-time cache policy (v6.77.0 — replaces two per-builder "mark the last tool"
        copies and restores the TTL ordering guard lost in 176567b BY CONSTRUCTION): a
        ``1h`` breakpoint promotes the earlier EXISTING breakpoints to ``1h`` (a longer TTL
        must precede a shorter one — 5m tools before 1h system is a hard 400) and never
        creates a marker on an earlier segment; a bare marker is the provider default and
        ranks as 5m; the ONLY marker it ever adds is on the last tool schema, and only when
        the tools segment carries none (unconditional on this family in both deleted sites —
        a tool-free payload therefore stays uncached HERE, and system/messages never gain a
        marker they did not declare; a tool-free lane is cached only by DECLARING its stable
        prefix at the caller, as the review surfaces and the safety supervisor do via
        ``review_helpers.cached_prompt_blocks``); above the four-breakpoint cap the four EARLIEST
        (governance-prefix) markers are kept, the tail MARKERS — never content — are dropped
        and the reduction is disclosed in usage (rationale and the builder-side loud layer:
        ``docs/ARCHITECTURE.md``). Only this freshly assembled payload is normalized —
        never caller-owned messages/tools, the canonical transcript, or a route that cannot
        carry these markers (``_route_normalizes_cache_breakpoints``). "1h" wins over
        "default" so pricing bills the extended-tier write multiplier.

        The owner's global TTL (``config.resolve_prompt_cache_ttl``, owner decision
        2026-08-08 Q2=A) has its single WIRE authority here — the one place that decides
        what every marker on this family actually ships as. It is not the only READER:
        ``review_helpers.cached_prompt_blocks(ttl=None)`` projects the same setting into
        the block it owns so a non-normalizing route still carries the owner's tier; on
        this family the finalizer would stamp that block to the same value anyway, so the
        two readers cannot diverge on the wire (``config.resolve_prompt_cache_ttl`` names
        both). When the setting names an explicit tier
        ('5m'/'1h') it is stamped onto EVERY existing breakpoint of this family —
        including caller-declared review/safety prefixes, which is what makes it an
        HONEST override rather than a floor — before the promotion rule runs, so
        ordering stays legal by construction (the 176567b every-call-400 class).
        'default' keeps the pre-setting behavior byte-for-byte: bare markers stay bare
        and a caller-declared ttl stands. It never CREATES a marker (the d32f703d
        empty-block 400 class), and non-Anthropic wire formats are untouched (the
        v5.30.0 Gemini ttl-field class).
        """
        breakpoints = self._payload_cache_breakpoints(payload)
        note: Optional[Dict[str, Any]] = None
        if _route_normalizes_cache_breakpoints(target):
            tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
            if not any(isinstance(t, dict) and isinstance(t.get("cache_control"), dict) for t in tools):
                for tool in reversed(tools):
                    # Schema entries only — skips an appended openrouter:web_search tool.
                    if isinstance(tool, dict) and (
                        isinstance(tool.get("function"), dict)
                        or tool.get("input_schema") is not None
                    ):
                        tool["cache_control"] = {"type": "ephemeral"}
                        breakpoints = self._payload_cache_breakpoints(payload)
                        break
            declared = len(breakpoints)
            if declared > self._MAX_CACHE_BREAKPOINTS:
                for holder in breakpoints[self._MAX_CACHE_BREAKPOINTS:]:
                    holder.pop("cache_control", None)
                breakpoints = breakpoints[:self._MAX_CACHE_BREAKPOINTS]
                note = {"declared": declared, "kept": len(breakpoints),
                        "dropped": declared - len(breakpoints)}
            from ouroboros.config import resolve_prompt_cache_ttl

            global_ttl = resolve_prompt_cache_ttl()
            if global_ttl in _VALID_CACHE_TTLS:
                for holder in breakpoints:
                    holder["cache_control"]["ttl"] = global_ttl
            if any(str(b["cache_control"].get("ttl") or "") == "1h" for b in breakpoints):
                for holder in breakpoints:
                    holder["cache_control"]["ttl"] = "1h"
        if not hasattr(self, "_cache_breakpoint_tls"):
            self._cache_breakpoint_tls = threading.local()
        self._cache_breakpoint_tls.pending = note
        # Report the strongest APPLIED TTL — the value that flows into usage metadata
        # (llm_usage/llm_round events) and prices the write tier. Readers consume this
        # recorded fact; nothing re-derives an "effective TTL" from the route.
        if any(str(b["cache_control"].get("ttl") or "") == "1h" for b in breakpoints):
            return "1h"
        if any(str(b["cache_control"].get("ttl") or "") == "5m" for b in breakpoints):
            return "5m"
        return "default" if breakpoints else None

    def _pop_cache_breakpoint_disclosure(self) -> Optional[Dict[str, Any]]:
        """The pending ≤4-cap reduction record for THIS thread's in-flight call (the
        finalizer writes the slot before every send, so it never mis-attributes)."""
        tls = getattr(self, "_cache_breakpoint_tls", None)
        pending = getattr(tls, "pending", None) if tls is not None else None
        if tls is not None:
            tls.pending = None
        return pending if isinstance(pending, dict) else None
