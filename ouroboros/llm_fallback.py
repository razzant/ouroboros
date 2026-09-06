"""The recovery ladder: what a failed or poisoned send is retried as.

A provider failure arrives as an exception or as an HTTP 200 whose body carries
the error instead. Either way the question is the same and is answered here, in
one ordered ladder: drop a rejected cache parameter, drop or re-floor a rejected
optional parameter, strip replayed reasoning and unpin the endpoint, or reroute
the same model to a healthy sibling. The two drivers (sync and async) are the
only callers that decide how far down the ladder a call walks.
"""


from __future__ import annotations

import copy
import hashlib
import logging
import re
import time
from typing import Any, Dict, Optional

from ouroboros.llm_attempt import (
    _attempt_request,
    _candidate_before_dispatch,
    _execute_candidate,
    _execute_candidate_async,
    _finalized_physical_candidate,
    _is_provider_policy_refusal,
    _is_structured_context_overflow_body,
    _is_structured_context_overflow_exception,
)
from ouroboros.reasoning_artifacts import (
    pop_reasoning_pin_note,
    transcript_has_sealed_reasoning,
)
from ouroboros.request_wire_recovery import (
    note_wire_send_failed,
    note_wire_send_succeeded,
    plan_next_wire_retry,
    request_wire_scoped,
)
from ouroboros.usage_accounting import UsageAccountingError, last_physical_attempt_capture


# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


class _RecoveryLadderMixin:
    """Provider-failure classification, one-shot repairs and the send drivers."""

    @staticmethod
    def _retry_without_prompt_cache_parameter(
        payload: Dict[str, Any],
        target: Dict[str, Any],
        exc: BaseException,
    ) -> Optional[Dict[str, Any]]:
        """Remove only an explicitly rejected cache control or affinity once."""
        if _is_structured_context_overflow_exception(exc) or _is_provider_policy_refusal(exc):
            return None
        provider = str(target.get("provider") or "").strip().lower()
        extra_body = payload.get("extra_body")
        param = ""
        if provider == "openai" and "prompt_cache_key" in payload:
            param = "prompt_cache_key"
        elif (
            bool(target.get("supports_openrouter_extensions"))
            and isinstance(extra_body, dict)
            and "session_id" in extra_body
        ):
            param = "session_id"
        elif (
            provider == "openai-compatible"
            and isinstance(extra_body, dict)
            and "cache" in extra_body
        ):
            param = "cache"
        if not param:
            return None

        text = str(exc or "").lower()
        if param not in text:
            return None
        if not any(
            marker in text
            for marker in (
                "unsupported",
                "not supported",
                "unknown parameter",
                "unrecognized",
                "unexpected keyword",
                "unexpected field",
                "invalid parameter",
                "not permitted",
                "extra inputs",
                "additional properties",
                "no endpoints found",
                "requested parameter",
            )
        ):
            return None

        retry_payload = copy.deepcopy(payload)
        if param == "prompt_cache_key":
            retry_payload.pop(param, None)
        else:
            retry_extra = retry_payload.get("extra_body")
            if isinstance(retry_extra, dict):
                retry_extra.pop(param, None)
            if not retry_extra:
                retry_payload.pop("extra_body", None)
        log.warning(
            "Retrying %s once without unsupported cache parameter %s",
            str(target.get("usage_model") or target.get("resolved_model") or "(unknown model)"),
            param,
        )
        return retry_payload

    @staticmethod
    def _is_http_status(exc: Exception, code: int) -> bool:
        """Structural HTTP-status check on a provider exception (``status_code``
        attribute; falls back to the OpenAI-SDK ``Error code: NNN`` message shape).
        Used instead of error-string matching so the recovery covers every provider
        phrasing of the same status class."""
        sc = getattr(exc, "status_code", None)
        if sc is not None:
            try:
                return int(sc) == int(code)
            except (TypeError, ValueError):
                pass
        # No status_code attr (non-SDK exceptions): match the code only as a
        # STATUS token — leading, or after error/status/http labels — not any bare
        # number, so a token count or id with "400" in it can't false-trigger.
        text = str(exc).strip().lower()
        return bool(re.search(rf"(?:^|error code:?\s*|status(?:[ _]code)?:?\s*|http[\s:]*){int(code)}\b", text))

    def _openrouter_signature_retry_kwargs(
        self,
        target: Dict[str, Any],
        kwargs: Dict[str, Any],
        exc: Exception,
    ) -> Optional[Dict[str, Any]]:
        """Strip replayed reasoning once for a non-overflow OpenRouter 400."""
        if _is_structured_context_overflow_exception(exc) or _is_provider_policy_refusal(exc):
            return None
        if not target.get("supports_openrouter_extensions"):
            return None
        if not self._is_http_status(exc, 400):
            return None
        return self._reroute_same_model_kwargs(target, kwargs)

    @staticmethod
    def _rotate_openrouter_session_affinity(payload: Dict[str, Any]) -> None:
        """A deliberate endpoint reroute must not reuse its sticky session key."""
        extra_body = payload.get("extra_body")
        if not isinstance(extra_body, dict) or not extra_body.get("session_id"):
            return
        previous = str(extra_body["session_id"])
        digest = hashlib.sha256(
            f"{previous}\0reroute\0{time.time_ns()}".encode("utf-8")
        ).hexdigest()[:32]
        extra_body["session_id"] = f"ouroboros-session-{digest}"

    def _reroute_same_model_kwargs(
        self,
        target: Dict[str, Any],
        kwargs: Dict[str, Any],
        *,
        allow_portable_reasoning: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Same-model reroute: strip replayed reasoning metadata and drop the
        provider pin (``allow_fallbacks=false``, set only to preserve reasoning
        continuity) so OpenRouter can route to a HEALTHY endpoint of the SAME
        model. Shared by the 400 signature-rejection path and the transient
        200-body provider-error path. Returns None when no replayed reasoning is
        present (nothing to strip / no continuity pin to drop — default routing can
        already fall back across endpoints). NEVER switches model — only endpoint.

        ``allow_portable_reasoning`` (set ONLY by the transient body-error path): when the
        replayed artifact is NOT sealed (``transcript_has_sealed_reasoning`` — readable
        text/summary, or an opaque form vouched by the signed-portable roster) it survives
        the same-model sibling-provider switch, so PRESERVE it (retry the same payload and
        let OpenRouter route to a healthy endpoint) rather than needlessly dropping
        continuity on the very rate-limit path the failover exists for. The 400
        signature-REJECTION path never sets this: a 400 means the artifact WAS rejected,
        so it must strip regardless of shape. This is the SAME predicate as the proactive
        dispatch pin — one artifact-shape truth for both directions (the pre-#468
        openai/* carve-out here is now absorbed by the roster, which excludes openai/*)."""
        if not target.get("supports_openrouter_extensions"):
            return None
        messages = kwargs.get("messages")
        if not isinstance(messages, list) or not self._has_replayed_reasoning_metadata(messages):
            return None
        model_id = str(kwargs.get("model") or "").strip().lstrip("~")
        preserve_reasoning = (
            allow_portable_reasoning
            and not transcript_has_sealed_reasoning(messages, model_id)
        )
        if preserve_reasoning:
            retry_kwargs = copy.deepcopy(kwargs)
            self._rotate_openrouter_session_affinity(retry_kwargs)
            return retry_kwargs
        retry_kwargs = copy.deepcopy(kwargs)
        retry_kwargs["messages"] = self._strip_openrouter_roundtrip_metadata(messages)
        if not self._has_replayed_reasoning_metadata(retry_kwargs["messages"]):
            extra_body = retry_kwargs.get("extra_body")
            provider = extra_body.get("provider") if isinstance(extra_body, dict) else None
            if isinstance(provider, dict):
                provider.pop("allow_fallbacks", None)
                if not provider:
                    extra_body.pop("provider", None)
                if not extra_body:
                    retry_kwargs.pop("extra_body", None)
        self._rotate_openrouter_session_affinity(retry_kwargs)
        return retry_kwargs

    @staticmethod
    def _provider_body_error(resp_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """An OpenAI-compatible HTTP 200 whose body carries a top-level ``error``
        object instead of a usable completion. OpenRouter passes upstream
        provider errors and its own 429/5xx through the body with status 200; the
        OpenAI SDK builds these leniently, keeping ``error`` and ``choices=None``.
        Returns the error dict, else None (a real completion wins over a
        non-fatal error field)."""
        if not isinstance(resp_dict, dict):
            return None
        err = resp_dict.get("error")
        if not isinstance(err, dict):
            return None
        choices = resp_dict.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if isinstance(choices[0], dict) else {}
            msg = first.get("message") if isinstance(first, dict) else None
            if isinstance(msg, dict) and (msg.get("content") or msg.get("tool_calls")):
                return None
        return err

    @staticmethod
    def _is_transient_body_error(err: Dict[str, Any]) -> bool:
        """Transient body-error = worth a same-model reroute/retry (rate limit,
        overload, upstream 5xx/timeout). Permanent client errors
        (auth/quota/bad-request) are not — they must surface unchanged."""
        try:
            code = int(err.get("code"))
        except (TypeError, ValueError):
            code = 0
        if code in (408, 409, 425, 429, 500, 502, 503, 504, 522, 524, 529):
            return True
        text = str(err.get("message") or "").lower()
        return any(
            marker in text
            for marker in (
                "rate limit", "too many requests", "overloaded", "temporarily",
                "timeout", "timed out", "unavailable", "try again", "capacity",
            )
        )

    def _reroute_kwargs_for_body_error(
        self,
        resp: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """If an HTTP-200 response actually carries a TRANSIENT provider
        body-error, return same-model reroute kwargs (provider unpinned; reasoning
        continuity preserved when the replayed artifact is not sealed, dropped
        otherwise); None when not applicable."""
        try:
            resp_dict = resp.model_dump()
        except Exception:
            return None
        err = self._provider_body_error(resp_dict)
        if not err or _is_structured_context_overflow_body(err):
            return None
        if not self._is_transient_body_error(err):
            return None
        reroute = self._reroute_same_model_kwargs(
            target, kwargs, allow_portable_reasoning=True
        )
        if reroute is None:
            return None
        log.warning(
            "OpenRouter same-model reroute after transient provider body-error "
            "(code=%s); reasoning_continuity_%s",
            err.get("code"),
            "preserved"
            if self._has_replayed_reasoning_metadata(reroute.get("messages") or [])
            else "dropped",
        )
        return reroute

    def _strip_kwargs_for_encrypted_body_error(
        self,
        resp: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Strip replayed encrypted reasoning for a non-overflow body 400."""
        try:
            resp_dict = resp.model_dump()
        except Exception:
            return None
        body_err = self._provider_body_error(resp_dict)
        if not isinstance(body_err, dict) or _is_structured_context_overflow_body(body_err):
            return None
        try:
            code = int(body_err.get("code") or 0)
        except (TypeError, ValueError):
            code = 0
        if code != 400:
            return None
        if "encrypted content" not in str(body_err.get("message") or "").lower():
            return None
        stripped = self._reroute_same_model_kwargs(target, kwargs)
        if stripped is not None:
            log.warning(
                "OpenRouter strip-and-retry after encrypted-reasoning body error (code=400)"
            )
        return stripped

    def _param_retry_kwargs_for_body_error(
        self,
        resp: Any,
        kwargs: Dict[str, Any],
        usage_model: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply exception-path parameter recovery to a non-overflow body 400."""
        try:
            resp_dict = resp.model_dump()
        except Exception:
            return None
        body_err = self._provider_body_error(resp_dict)
        if not isinstance(body_err, dict) or _is_structured_context_overflow_body(body_err):
            return None
        try:
            code = int(body_err.get("code") or 0)
        except (TypeError, ValueError):
            code = 0
        if code != 400:
            return None
        message = str(body_err.get("message") or "")
        if not message:
            return None
        return self._retry_without_optional_sampling(kwargs, usage_model, RuntimeError(message))

    @request_wire_scoped
    def _create_chat_completion_with_retries(
        self,
        create_fn: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Any:
        # Discard a prior aborted ladder's pin note.
        pop_reasoning_pin_note()

        def _send(candidate: Dict[str, Any]) -> Any:
            candidate = _finalized_physical_candidate(target, candidate, "chat.completions")
            request = _attempt_request(target, candidate)
            try:
                result = _execute_candidate(
                    request,
                    lambda: create_fn(**candidate),
                    _candidate_before_dispatch(candidate, request),
                )
                note_wire_send_succeeded(last_physical_attempt_capture())
                self._stage_reasoning_pin_disclosure(candidate)
                return result
            except UsageAccountingError:
                # Admission failure cannot leave its disclosure for a later call.
                self._pop_effort_clamp_disclosure()
                note_wire_send_failed()
                raise
            except Exception:
                note_wire_send_failed()
                raise

        def _body_error(response: Any) -> Optional[Dict[str, Any]]:
            try:
                return self._provider_body_error(response.model_dump())
            except Exception:
                return None

        def _recover_existing(
            candidate: Dict[str, Any],
            *,
            failure: Optional[Exception] = None,
            response: Any = None,
        ) -> Any:
            """One bounded exception/body state machine, then signature recovery."""
            try:
                current_candidate = candidate
                current_failure = failure
                current_response = response
                signature_used = False
                for _ in range(8):
                    if current_failure is None:
                        body = _body_error(current_response)
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=body, body_error=True,
                        )
                        if retry_kwargs is None:
                            return current_response
                    else:
                        if _is_provider_policy_refusal(current_failure):
                            # A typed refusal is permanent by class (D09): no
                            # rung may re-attempt the refused call.
                            raise current_failure
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=current_failure,
                        )
                        if retry_kwargs is None and not signature_used:
                            retry_kwargs = self._openrouter_signature_retry_kwargs(
                                target, current_candidate, current_failure,
                            )
                            signature_used = retry_kwargs is not None
                        if retry_kwargs is None:
                            raise current_failure
                    current_candidate = retry_kwargs
                    try:
                        current_response = _send(retry_kwargs)
                        current_failure = None
                    except UsageAccountingError:
                        raise
                    except Exception as retry_exc:
                        current_failure = retry_exc
                        current_response = None
                if current_failure is not None:
                    raise current_failure
                return current_response
            except Exception:
                # The recovery ladder died terminally: discard any pending
                # effort-clamp note (e.g. the floored learning retry's
                # learned_floor disclosure) so it cannot misattach to a later,
                # unrelated response on this thread (plan-review r3; lanes that
                # never call _clamp_effort_for_model at build time would not
                # reset it).
                self._pop_effort_clamp_disclosure()
                raise

        try:
            resp = _send(kwargs)
        except UsageAccountingError:
            raise  # _send already discarded any pending clamp note (triad r4)
        except Exception as exc:
            cache_retry_kwargs = self._retry_without_prompt_cache_parameter(kwargs, target, exc)
            if cache_retry_kwargs is not None:
                try:
                    resp = _send(cache_retry_kwargs)
                    kwargs = cache_retry_kwargs
                except UsageAccountingError:
                    raise
                except Exception as cache_retry_exc:
                    return _recover_existing(
                        cache_retry_kwargs, failure=cache_retry_exc,
                    )
            else:
                return _recover_existing(kwargs, failure=exc)
        # HTTP-200 success can still carry a transient provider body-error
        # (OpenRouter passes 429/5xx through the body); reroute once to a healthy
        # endpoint of the SAME model while request kwargs are still mutable.
        reroute_kwargs = self._reroute_kwargs_for_body_error(resp, kwargs, target)
        if reroute_kwargs is not None:
            try:
                resp = _send(reroute_kwargs)
            except UsageAccountingError:
                raise
            except Exception as exc:
                if _is_provider_policy_refusal(exc):
                    # A refused call is not a provider answer to fall back FROM.
                    self._pop_effort_clamp_disclosure()
                    raise
                return resp
            kwargs = reroute_kwargs
        # An encrypted-reasoning 400 delivered in the body (directly, or on the
        # response of the reroute above) gets the same one-shot strip-and-retry
        # as the exception path — never a permanent task-killing bad_request.
        strip_kwargs = self._strip_kwargs_for_encrypted_body_error(resp, kwargs, target)
        if strip_kwargs is not None:
            try:
                resp = _send(strip_kwargs)
                kwargs = strip_kwargs
            except UsageAccountingError:
                raise
            except Exception as exc:
                if _is_provider_policy_refusal(exc):
                    self._pop_effort_clamp_disclosure()
                    raise
                return resp
        return _recover_existing(kwargs, response=resp)

    @request_wire_scoped
    async def _create_chat_completion_with_retries_async(
        self,
        create_fn: Any,
        kwargs: Dict[str, Any],
        target: Dict[str, Any],
    ) -> Any:
        # Discard a prior aborted ladder's pin note.
        pop_reasoning_pin_note()

        async def _send(candidate: Dict[str, Any]) -> Any:
            candidate = _finalized_physical_candidate(target, candidate, "chat.completions")
            request = _attempt_request(target, candidate)
            try:
                result = await _execute_candidate_async(
                    request,
                    lambda: create_fn(**candidate),
                    _candidate_before_dispatch(candidate, request),
                )
                note_wire_send_succeeded(last_physical_attempt_capture())
                self._stage_reasoning_pin_disclosure(candidate)
                return result
            except UsageAccountingError:
                # Sync-driver parity: central UAE discard (triad r4).
                self._pop_effort_clamp_disclosure()
                note_wire_send_failed()
                raise
            except Exception:
                note_wire_send_failed()
                raise

        def _body_error(response: Any) -> Optional[Dict[str, Any]]:
            try:
                return self._provider_body_error(response.model_dump())
            except Exception:
                return None

        async def _recover_existing(
            candidate: Dict[str, Any],
            *,
            failure: Optional[Exception] = None,
            response: Any = None,
        ) -> Any:
            """Async twin of the bounded exception/body state machine."""
            try:
                current_candidate = candidate
                current_failure = failure
                current_response = response
                signature_used = False
                for _ in range(8):
                    if current_failure is None:
                        body = _body_error(current_response)
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=body, body_error=True,
                        )
                        if retry_kwargs is None:
                            return current_response
                    else:
                        if _is_provider_policy_refusal(current_failure):
                            # A typed refusal is permanent by class (D09): no
                            # rung may re-attempt the refused call.
                            raise current_failure
                        retry_kwargs = plan_next_wire_retry(
                            current_candidate, error=current_failure,
                        )
                        if retry_kwargs is None and not signature_used:
                            retry_kwargs = self._openrouter_signature_retry_kwargs(
                                target, current_candidate, current_failure,
                            )
                            signature_used = retry_kwargs is not None
                        if retry_kwargs is None:
                            raise current_failure
                    current_candidate = retry_kwargs
                    try:
                        current_response = await _send(retry_kwargs)
                        current_failure = None
                    except UsageAccountingError:
                        raise
                    except Exception as retry_exc:
                        current_failure = retry_exc
                        current_response = None
                if current_failure is not None:
                    raise current_failure
                return current_response
            except Exception:
                self._pop_effort_clamp_disclosure()
                raise

        try:
            resp = await _send(kwargs)
        except UsageAccountingError:
            raise  # _send already discarded any pending clamp note (triad r4)
        except Exception as exc:
            cache_retry_kwargs = self._retry_without_prompt_cache_parameter(kwargs, target, exc)
            if cache_retry_kwargs is not None:
                try:
                    resp = await _send(cache_retry_kwargs)
                    kwargs = cache_retry_kwargs
                except UsageAccountingError:
                    raise
                except Exception as cache_retry_exc:
                    return await _recover_existing(
                        cache_retry_kwargs, failure=cache_retry_exc,
                    )
            else:
                return await _recover_existing(kwargs, failure=exc)
        # HTTP-200 success can still carry a transient provider body-error
        # (OpenRouter passes 429/5xx through the body); reroute once to a healthy
        # endpoint of the SAME model while request kwargs are still mutable.
        reroute_kwargs = self._reroute_kwargs_for_body_error(resp, kwargs, target)
        if reroute_kwargs is not None:
            try:
                resp = await _send(reroute_kwargs)
            except UsageAccountingError:
                raise
            except Exception as exc:
                if _is_provider_policy_refusal(exc):
                    # A refused call is not a provider answer to fall back FROM.
                    self._pop_effort_clamp_disclosure()
                    raise
                return resp
            kwargs = reroute_kwargs
        # An encrypted-reasoning 400 delivered in the body (directly, or on the
        # response of the reroute above) gets the same one-shot strip-and-retry
        # as the exception path — never a permanent task-killing bad_request.
        strip_kwargs = self._strip_kwargs_for_encrypted_body_error(resp, kwargs, target)
        if strip_kwargs is not None:
            try:
                resp = await _send(strip_kwargs)
                kwargs = strip_kwargs
            except UsageAccountingError:
                raise
            except Exception as exc:
                if _is_provider_policy_refusal(exc):
                    self._pop_effort_clamp_disclosure()
                    raise
                return resp
        return await _recover_existing(kwargs, response=resp)
