"""The local llama.cpp lane and its context limit.

Our local lane carries no images, has a small window, and costs nothing. It never cuts a
request to fit: an exact shortfall measured on the serving instance, or the server's own
refusal, is the typed size refusal (``LocalContextTooLargeError``) the common refusal path
recovers from, so BIBLE's principle bodies and every other section reach the model whole or
not at all.
"""


from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.deadline_utils import physical_dispatch_timeout
from ouroboros.model_concurrency import limit_physical_send

from ouroboros.context_budget import (  # the typed overflow lives with the overflow vocabulary
    LocalContextTooLargeError,
    context_overflow_message,
)
from ouroboros.llm_attempt import (
    _attempt_request,
    _candidate_before_dispatch,
    _execute_candidate,
    _finalized_physical_candidate,
    _is_structured_context_overflow_exception,
)
from ouroboros.usage_accounting import UsageAccountingError
from ouroboros.transport_custody import outcome_unknown_on_chain

# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


def local_context_limits(max_tokens: int) -> Tuple[int, int]:
    """Confirmed serving window (0 if unknown) and output cap for caller preflight."""
    ctx_len = 0
    local_max = min(max_tokens, 2048)
    try:
        from ouroboros.local_model import get_manager
        evidence = get_manager().serving_context_evidence() or {}
        if evidence.get("confirmed") is True:
            ctx_len = max(0, int(evidence.get("context_window") or 0))
        if ctx_len > 0:
            local_max = min(max_tokens, max(256, ctx_len // 4))
    except Exception:
        pass
    return ctx_len, local_max


def _split_leading_reasoning(text: str) -> Tuple[str, str]:
    """``(answer, reasoning)`` of a reply whose reasoning comes first; a reply without one is returned whole.

    Two shapes reach this lane: ``<think>…</think>`` the model writes itself (Qwen3), and reasoning that is only
    closed, because the chat template already opened ``<think>`` in the prompt (Qwen3.5 and later)."""
    stripped = text.lstrip()
    paired = re.match(r"<think>(.*?)</think>", stripped, re.DOTALL | re.IGNORECASE)
    if paired:
        return stripped[paired.end():].strip(), paired.group(1).strip()
    if stripped.startswith("<tool_call>"):
        return text, ""  # a reply that opens with a call has no reasoning before it; its arguments are not reasoning
    # The template closes reasoning on a line of its own; a sentence that merely mentions the tag is an answer.
    closed = re.search(r"(?:^|\n)[ \t]*</think>[ \t]*(?:\n|$)", stripped, re.IGNORECASE)
    if closed and not re.search(r"<think>", stripped[:closed.start()], re.IGNORECASE):
        return stripped[closed.end():].strip(), stripped[:closed.start()].strip()
    return text, ""


class _LocalLaneMixin:
    """The local chat request."""

    def _build_local_candidate(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
        max_tokens: int, tool_choice: str, timeout: Optional[float] = None,
        processing_preference: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare the complete local payload for sizing and actual dispatch."""
        messages = self._normalize_system_message_placement(messages)
        clean_messages = self._strip_openrouter_roundtrip_metadata(
            self._copy_messages_with_cache_policy(
                messages, allow_message_cache_control=False, flatten_tool_content_blocks=True,
            )
        )
        # Our llama.cpp lane is launched without a vision handler: keep base64 out
        # of the text prompt and name the lane, never the model.
        clean_messages = self._replace_image_blocks_with_placeholder(clean_messages, "local llama.cpp")
        _ctx_len, local_max = local_context_limits(max_tokens)
        for msg in clean_messages:
            content = msg.get("content")
            if content is None and msg.get("role") == "assistant":
                msg["content"] = ""
            if isinstance(content, list):
                msg["content"] = "\n\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
                )
        clean_tools = None
        if tools:
            clean_tools = [{k: v for k, v in t.items() if k != "cache_control"} for t in tools]
        kwargs: Dict[str, Any] = {"model": "local-model", "messages": clean_messages, "max_tokens": local_max}
        if clean_tools:
            kwargs["tools"] = clean_tools
            kwargs["tool_choice"] = tool_choice
        if timeout and timeout > 0:
            kwargs["timeout"] = float(timeout)

        from ouroboros.model_slots import resolve_processing_preference
        from ouroboros.local_model import get_manager
        evidence = get_manager().serving_context_evidence() or {}
        preference = resolve_processing_preference(override=processing_preference)
        target = {"provider": "local", "resolved_model": "local-model", "usage_model": "local-model",
                  "processing_preference": preference, "context_window_tokens": evidence.get("context_window"),
                  "context_window_confirmed": evidence.get("confirmed") is True}
        return target, kwargs

    def _finalize_local_candidate(self, target: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        """Measure on the serving instance, then bind the output allowance before send.

        This explicit stage may perform bounded, non-generating loopback I/O. The payload
        builder and shared context arithmetic remain pure. An instance that cannot count
        the candidate leaves the estimate in force; an exact shortfall is the typed refusal
        with this candidate's own facts (``llm_attempt.bound_reply_allowance``).
        """
        from ouroboros.local_model import get_manager

        measure = getattr(get_manager(), "measure_prepared_input", None)
        # Bind the provider-clean physical payload before measuring. Host-only
        # metadata must not affect the measured input or candidate hash.
        candidate = _finalized_physical_candidate(target, payload, "chat.completions")
        evidence = measure(candidate) if callable(measure) else None
        if not (isinstance(evidence, dict) and evidence.get("supported")):
            return candidate
        target["local_input_measurement"] = evidence
        # Rebuild from the original prepared source so the exact measured
        # input and output cap are sealed together.
        return _finalized_physical_candidate(target, payload, "chat.completions")

    def _chat_local(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
        max_tokens: int, tool_choice: str, timeout: Optional[float] = None,
        processing_preference: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send exactly the previously prepared complete local candidate."""
        client = self._get_local_client()
        local_target, candidate = self._build_local_candidate(
            messages, tools, max_tokens, tool_choice, timeout, processing_preference)
        # The shared finalizer removes SDK transport options from the measured
        # and sealed payload. Keep the builder's positive caller override (else
        # the client's own phases) for the actual send, which narrows it to the
        # caller's bounds left after preparation and the slot wait (#572),
        # including chat_async's local thread.
        transport_timeout = candidate.get("timeout", getattr(client, "timeout", None))
        local_target["requested_reasoning_effort"] = reasoning_effort
        from ouroboros.send_clock import stamp_clock_note

        # ONE clock line per call, sampled before finalization: the serving
        # instance measures exactly the bytes that are then sealed and sent.
        candidate = self._finalize_local_candidate(local_target, stamp_clock_note(candidate))
        clean_tools = candidate.get("tools")
        # ONE physical attempt per call. Re-sending here spent the caller's
        # physical-attempt budget without the caller authorising it, so a
        # transient local failure now surfaces to the single retry policy that
        # owns the decision (``loop_llm_call.call_llm_with_retry``), which counts
        # the attempts it authorises.
        try:
            request = _attempt_request(local_target, candidate, source="llm.local")
            before = _candidate_before_dispatch(candidate, request)

            def check_instance(reservation):
                measured = local_target.get("local_input_measurement") or {}
                if measured.get("supported"):
                    from ouroboros.local_model import get_manager
                    from ouroboros.usage_accounting import PhysicalAttemptPreparationFailed

                    current = get_manager().serving_context_evidence()
                    if (current.get("process_id") != measured.get("process_id") or not current.get("confirmed")
                            or current.get("context_window") != measured.get("context_window")):
                        raise PhysicalAttemptPreparationFailed("The measured local model instance changed before dispatch")
                return before(reservation)

            def send():
                timeout = physical_dispatch_timeout(transport_timeout)
                return client.chat.completions.create(**candidate, **({"timeout": timeout} if timeout is not None else {}))

            resp = _execute_candidate(request, limit_physical_send(local_target, send), check_instance)
        except UsageAccountingError:
            raise
        except Exception as exc:
            err = str(exc)
            if (_is_structured_context_overflow_exception(exc)
                    or context_overflow_message(err)):
                raise LocalContextTooLargeError(err) from exc
            # Exception-owned capture proves this attempt; prior ContextVar may be unrelated.
            if outcome_unknown_on_chain(exc):
                raise  # Outer custody owns an unknown physical outcome.
            log.warning("Local model request failed: %s", exc)
            raise

        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        if not msg.get("tool_calls") and msg.get("content") and clean_tools:
            functions = [t.get("function") or {} for t in clean_tools if isinstance(t, dict)]
            allowed_tool_names = {str(f.get("name", "")).strip() for f in functions}
            schemas = {str(f.get("name", "")).strip(): f.get("parameters") for f in functions}
            msg = self._parse_tool_calls_from_content(msg, allowed_tool_names, schemas)
        if not msg.get("tool_calls") and msg.get("content") and not msg.get("reasoning_content"):
            # A plain answer's leading reasoning is the model's own, not the reply: keep it beside the answer. An empty
            # reasoning block goes too, so its bare tags never reach the owner.
            content = str(msg["content"])
            answer, reasoning = _split_leading_reasoning(content)
            if answer != content:
                msg = {**msg, "content": answer, **({"reasoning_content": reasoning} if reasoning else {})}

        usage["cost"] = 0.0
        usage["cost_final"] = True
        # CPL6-F1: the same usage provenance every remote lane stamps — the
        # ledger row already carried provider=local, but a consumer reading the
        # returned usage alone could not attribute the call.
        usage["provider"] = "local"
        usage["resolved_model"] = "local-model"
        # The provider's own cut marker: consolidation refuses a truncated
        # summary by usage.response_finish_reason on every lane, this one too.
        first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        finish_reason = first_choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason.strip():
            usage["response_finish_reason"] = finish_reason.strip()[:64]
        from ouroboros.llm_attempt import attach_processing_receipt

        attach_processing_receipt(local_target, usage)
        return msg, usage
