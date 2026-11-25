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
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.deadline_utils import physical_dispatch_timeout
from ouroboros.model_concurrency import limit_physical_send, measurement_slot

from ouroboros.context_budget import LocalContextTooLargeError  # the typed overflow lives with the overflow vocabulary
from ouroboros.llm_messages import DISPLAY_KEY
from ouroboros.llm_attempt import (
    _attempt_request,
    _candidate_before_dispatch,
    _execute_candidate,
    _finalized_physical_candidate,
    is_size_refusal,
)
from ouroboros.usage_accounting import UsageAccountingError
from ouroboros.transport_custody import outcome_unknown_on_chain

# The moved warnings keep the logger identity they were emitted under.
log = logging.getLogger("ouroboros.llm")


def local_context_limits(max_tokens: int, evidence: Optional[Dict[str, Any]] = None) -> Tuple[int, int]:
    """Confirmed serving window (0 if unknown) and output cap for caller preflight.

    ``evidence`` lets a caller supply facts already bound to the endpoint it sends to;
    omitted, the manager's own serving evidence is read."""
    ctx_len = 0
    local_max = min(max_tokens, 2048)
    try:
        if evidence is None:
            from ouroboros.local_model import get_manager
            evidence = get_manager().serving_context_evidence() or {}
        if evidence.get("confirmed") is True:
            ctx_len = max(0, int(evidence.get("context_window") or 0))
        if ctx_len > 0:
            local_max = min(max_tokens, max(256, ctx_len // 4))
    except Exception:
        pass
    return ctx_len, local_max


def _endpoint_port(base_url: Optional[str]) -> Optional[int]:
    """The port an OpenAI-compatible base URL names, or None when it spells none."""
    try:
        from urllib.parse import urlparse
        return urlparse(str(base_url or "")).port
    except Exception:
        return None


def _split_leading_reasoning(text: str) -> Tuple[str, str]:
    """``(answer, reasoning)`` of a reply whose reasoning comes first; a reply without one is returned whole.

    Two shapes reach this lane: ``<think>…</think>`` the model writes itself (Qwen3), and reasoning that is only
    closed, because the chat template already opened ``<think>`` in the prompt (Qwen3.5 and later). Reasoning ends at
    the first ``</think>`` outside code: anywhere after the reply's own opening tag (small models write
    ``<think>brief</think>Answer``), on a line of its own when the template opened it, and then only if no opening tag
    outside code comes first. A reply with no closing tag outside code is returned whole. One pass: code spans are
    read once and met with the tags in order.

    Known limits (review 8): in the template-opened shape, reasoning that names ``<think>`` in plain prose, or a closing
    tag on a line with other text (``done.</think>``), leaves the whole reply as the answer; reasoning with an unclosed
    fence pairs it with a fence of the answer, hiding the closing line; an answer without reasoning whose indented code
    block holds a lone ``</think>`` line loses its beginning."""
    stripped = text.lstrip()
    opened = stripped[:7].lower() == "<think>"
    if not opened and stripped.startswith("<tool_call>"):
        return text, ""  # a reply that opens with a call has no reasoning before it; its arguments are not reasoning
    if "</think>" not in stripped.lower():
        return text, ""  # nothing closes reasoning: the reply is the answer (and its code is never scanned)
    spans, span = _code_spans(stripped), 0
    breaks, line, bounds = [m.start() for m in re.finditer("\n", stripped)], 0, None
    for tag in re.finditer(r"</?think>", stripped, re.IGNORECASE):
        while span < len(spans) and spans[span][1] <= tag.start():
            span += 1
        if (opened and tag.start() == 0) or (span < len(spans) and spans[span][0] <= tag.start()):
            continue  # the reply's own opening tag, or a tag quoted in code
        closing = tag.group(0)[1] == "/"
        if opened:
            if closing:
                return stripped[tag.end():].strip(), stripped[7:tag.start()].strip()
            continue
        if not closing:
            return text, ""
        while line < len(breaks) and breaks[line] < tag.start():
            line, bounds = line + 1, None
        if bounds is None:  # this line's first and last visible characters, read once per line
            first = breaks[line - 1] + 1 if line else 0
            last = breaks[line] if line < len(breaks) else len(stripped)
            visible = re.search(r"\S", stripped[first:last])
            bounds = (first, last, first + visible.start(), first + len(stripped[first:last].rstrip()))
        if bounds[2] == tag.start() and bounds[3] == tag.end():
            return stripped[bounds[1]:].strip(), stripped[:bounds[0]].strip()
    return text, ""


def stamp_display(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp a tool-call reply's display projection (``DISPLAY_KEY``): its leading reasoning apart from its visible
    text.

    History keeps the bytes the server sent (llama.cpp #24890); progress and salvage read the projection and parse no
    grammar. A reply with its own reasoning field is the server's and is left as it came."""
    content = msg.get("content")
    if (msg.get("tool_calls") and isinstance(content, str)
            and not any(msg.get(key) for key in ("reasoning_content", "reasoning", "reasoning_details"))):
        visible, reasoning = _split_leading_reasoning(content)
        if visible != content:
            msg[DISPLAY_KEY] = {"text": visible, "reasoning": reasoning}
    return msg


def _code_spans(text: str) -> List[Tuple[int, int]]:
    """``(start, end)`` of closed fenced blocks and one-line inline code spans, in order and in linear time.

    An inline span opens at a backtick run and closes at the next run of the same length on its line; a run without
    such a partner is literal text."""
    spans: List[Tuple[int, int]] = []
    fence, fence_start, offset = "", 0, 0
    for line in text.splitlines(keepends=True):
        opener = re.match(r"[ \t]{0,3}(`{3,}|~{3,})", line)
        if fence:
            if opener and opener.group(1)[0] == fence[0] and len(opener.group(1)) >= len(fence) \
                    and not line[opener.end():].strip():
                spans.append((fence_start, offset + len(line)))
                fence = ""
        elif opener:
            fence, fence_start = opener.group(1), offset
        else:
            runs = list(re.finditer(r"`+", line))
            later: Dict[int, deque] = {}
            for index, run in enumerate(runs):
                later.setdefault(len(run.group(0)), deque()).append(index)
            index = 0
            while index < len(runs):
                partners = later[len(runs[index].group(0))]
                while partners and partners[0] <= index:
                    partners.popleft()
                if partners:
                    close = partners.popleft()
                    spans.append((offset + runs[index].start(), offset + runs[close].end()))
                    index = close + 1
                else:
                    index += 1
        offset += len(line)
    return spans


class _LocalLaneMixin:
    """The local chat request."""

    def _build_local_candidate(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
        max_tokens: int, tool_choice: str, timeout: Optional[float] = None,
        processing_preference: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare the complete local payload for sizing and actual dispatch.

        ``base_url`` names the endpoint the SDK will actually reach. When given, the
        candidate's serving facts are bound to that endpoint: a count from a manager
        listening elsewhere stays uncertain rather than describing another server."""
        messages = self._normalize_system_message_placement(messages)
        clean_messages = self._strip_openrouter_roundtrip_metadata(
            self._copy_messages_with_cache_policy(
                messages, allow_message_cache_control=False, flatten_tool_content_blocks=True,
            )
        )
        # Our llama.cpp lane is launched without a vision handler: keep base64 out
        # of the text prompt and name the lane, never the model.
        clean_messages = self._replace_image_blocks_with_placeholder(clean_messages, "local llama.cpp")
        from ouroboros.local_model import get_manager
        manager = get_manager()
        expected_port = _endpoint_port(base_url)
        if base_url:
            evidence = manager.serving_context_evidence(bind_endpoint=True) or {}
            if expected_port is None or evidence.get("port") != expected_port:
                evidence = {}
        else:
            evidence = manager.serving_context_evidence() or {}
        _ctx_len, local_max = local_context_limits(max_tokens, evidence)
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
        preference = resolve_processing_preference(override=processing_preference)
        target = {"provider": "local", "resolved_model": "local-model", "usage_model": "local-model",
                  "processing_preference": preference, "context_window_tokens": evidence.get("context_window"),
                  "context_window_confirmed": evidence.get("confirmed") is True,
                  "window_instance_id": str(evidence.get("process_id") or "")}
        if base_url:
            target["local_dispatch_base_url"] = str(base_url)
        return target, kwargs

    def _finalize_local_candidate(self, target: Dict[str, Any], payload: Dict[str, Any],
                                  *, measure: bool = True) -> Dict[str, Any]:
        """Measure on the serving instance, then bind the output allowance before send.

        This explicit stage may perform bounded, non-generating loopback I/O. The payload
        builder and shared context arithmetic remain pure. An instance that cannot count
        the candidate leaves the estimate in force; an exact shortfall is the typed refusal
        with this candidate's own facts (``llm_attempt.bound_reply_allowance``).

        A caller decides ``measure`` from ``model_concurrency.measurement_slot``, which admits this count
        before its connection opens: ``False`` is the once-only form's busy slot, where the count is advice
        and the provider-clean candidate keeps its estimate with no connection at all.
        """
        from ouroboros.local_model import get_manager

        # Bind the provider-clean physical payload before measuring. Host-only
        # metadata must not affect the measured input or candidate hash.
        candidate = _finalized_physical_candidate(target, payload, "chat.completions")
        if not measure:
            return candidate
        measure_input = getattr(get_manager(), "measure_prepared_input", None)
        endpoint_port = _endpoint_port(target.get("local_dispatch_base_url")) if target.get("local_dispatch_base_url") else None
        if target.get("local_dispatch_base_url") and endpoint_port is None:
            return candidate
        # The count must keep the actual SDK port.
        evidence = (measure_input(candidate, expected_port=endpoint_port)
                    if callable(measure_input) and endpoint_port is not None
                    else measure_input(candidate) if callable(measure_input) else None)
        if not (isinstance(evidence, dict) and evidence.get("supported")):
            return candidate
        target["local_input_measurement"] = evidence
        # Rebuild from the original prepared source so the exact measured
        # input and output cap are sealed together.
        return _finalized_physical_candidate(target, payload, "chat.completions")

    def _prepared_local_candidate(self, messages, tools, max_tokens, tool_choice, processing_preference, reasoning_effort,
                                  *, queued=False, base_url=None):
        """The measured, sealed request shared by the built-in sender and its budget lookahead.

        ``queued`` is the sender's own preparation: its count is a request to the built-in server that waits on
        the inference lock like a generation, so it takes the server's slot first and opens no connection before it
        (``model_concurrency.measurement_slot``). The lookahead tries that slot once instead: while a generation
        holds it no connection opens and the candidate keeps its estimate, so a price that is advice never stands
        in line behind the work it prices."""
        from ouroboros.send_clock import stamp_clock_note

        if base_url is None and not queued:
            # The lookahead owns no SDK client: snapshot the built-in route URL from one port read, so the
            # target, its count and the queue admission below name the endpoint the lane dispatches to.
            from ouroboros.local_model import local_dispatch_port
            from ouroboros.model_concurrency import built_in_route

            base_url = built_in_route(local_dispatch_port())["base_url"]

        target, payload = self._build_local_candidate(messages, tools, max_tokens, tool_choice,
                                                      processing_preference=processing_preference,
                                                      base_url=base_url)
        target["requested_reasoning_effort"] = reasoning_effort
        # The clock line is sealed after the wait, and a count the server's one slot declined opens no connection.
        with measurement_slot(target, wait=queued) as admitted:
            return target, self._finalize_local_candidate(target, stamp_clock_note(payload), measure=admitted)

    def _chat_local(
        self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]],
        max_tokens: int, tool_choice: str, timeout: Optional[float] = None,
        processing_preference: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Send exactly the previously prepared complete local candidate."""
        client = self._get_local_client()
        # The queue, the measurement and the generation must name the endpoint this SDK actually
        # reaches: the client is snapshotted once, so its base_url is the address every later
        # env read must not silently replace.
        client_base_url = getattr(client, "base_url", None)
        local_target, candidate = self._prepared_local_candidate(
            messages, tools, max_tokens, tool_choice, processing_preference, reasoning_effort,
            queued=True, base_url=str(client_base_url) if client_base_url else None)
        # The SDK timeout is transport-only and is narrowed again after the slot wait.
        transport_timeout = float(timeout) if timeout and timeout > 0 else getattr(client, "timeout", None)
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
            if is_size_refusal(exc, any_status=True):
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
        msg = stamp_display(dict(msg))
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
