"""Transcript shaping for the wire and the reasoning-artifact contract.

Providers disagree about where a system message may appear, whether a tool
result may carry blocks, what a blind model does with an image, whose
reasoning signatures they can validate, and how much of a leading system
message their prompt cache can reuse. This module owns the send-copy
transforms that answer those disagreements (``split_leading_system_prefix``
included) and the predicates that decide when replayed reasoning is
portable — never the canonical transcript, which every transform copies
before touching.
"""


from __future__ import annotations

import copy
from typing import Any, Dict, List

from ouroboros.anthropic_native_custody import custody_private_key, scrub_native_custody
from ouroboros.llm_attempt import _VALID_CACHE_TTLS
from ouroboros.provider_models import normalize_model_identity

# The Main context builder's declaration on its leading system message: how many leading
# text blocks are byte-stable across conversations (governance). A provider whose prompt
# cache treats the whole leading system section as one unit keeps only those blocks
# there (``split_leading_system_prefix``). Host-only metadata: popped from every send copy.
STABLE_PREFIX_BLOCKS_KEY = "_stable_prefix_blocks"

# Byte-stable provenance header of the projected host-context notice (no clocks, hashes
# or ids: round N+1's send copy must remain a prefix extension of round N's).
HOST_CONTEXT_NOTICE_BEFORE_TASK = (
    "Host context for this turn: memory, knowledge index, runtime facts and recent "
    "activity, rendered by the runtime as a continuation of the system prompt. Not "
    "written by my human and not a message to answer; the message to act on follows next."
)
HOST_CONTEXT_NOTICE_AFTER_TASK = (
    "Host context for this turn: memory, knowledge index, runtime facts and recent "
    "activity, rendered by the runtime as a continuation of the system prompt. Not "
    "written by my human and not a message to answer; the message to act on is the one above."
)
SYSTEM_PREFIX_SPLIT_PLACEMENTS = ("before_task", "after_task")


def split_leading_system_prefix(
    messages: List[Dict[str, Any]], *, placement: str = "before_task",
) -> tuple[List[Dict[str, Any]], int]:
    """Project a declared leading system message for a whole-section prompt cache.

    Returns ``(messages, moved_blocks)``. Applies only when the first message is a
    system message that carries ``STABLE_PREFIX_BLOCKS_KEY`` (the Main context builder's
    declaration, ``context_fit.ContextFitProjection.system_message``), its content is a
    list of text blocks longer than the declared count, and no second system message
    leads the transcript; every other shape — string systems, undeclared multi-block
    review prompts, several leading system messages — comes back unchanged with ``0``.
    The declared blocks stay the system message; the remaining non-empty text blocks
    become ONE ``[SYSTEM NOTICE]`` message with a byte-stable provenance header: a user
    message right before the task (``before_task``) or a developer message right after
    the first user message (``after_task``). A pure function of the canonical messages
    (never mutated), so round N+1's copy extends round N's and the prospective wrap-up
    candidate equals the send. Measured 2026-09-25 on ``openai/gpt-6-sol``: the next
    conversation's first round read 198,797 of 393,676 tokens from cache instead of 0.
    """
    if placement not in SYSTEM_PREFIX_SPLIT_PLACEMENTS:
        raise ValueError(f"unknown system prefix placement: {placement!r}")
    if not messages or not isinstance(messages[0], dict):
        return messages, 0
    leading = messages[0]
    declared = leading.get(STABLE_PREFIX_BLOCKS_KEY)
    if str(leading.get("role") or "") != "system" or not isinstance(declared, int) or declared < 1:
        return messages, 0
    if len(messages) > 1 and isinstance(messages[1], dict) and str(messages[1].get("role") or "") == "system":
        return messages, 0
    content = leading.get("content")
    if not isinstance(content, list) or len(content) <= declared or not all(
        isinstance(block, dict) and str(block.get("type") or "text") == "text"
        and isinstance(block.get("text"), str) for block in content
    ):
        return messages, 0
    moved = [block["text"] for block in content[declared:] if block["text"].strip()]
    if not moved:
        return messages, 0
    system = {key: copy.deepcopy(value) for key, value in leading.items() if key != STABLE_PREFIX_BLOCKS_KEY}
    system["content"] = copy.deepcopy(content[:declared])
    rest = [copy.deepcopy(message) for message in messages[1:]]
    body = "\n\n".join(moved)
    if placement == "after_task":
        first_user = next((i for i, message in enumerate(rest)
                           if isinstance(message, dict) and str(message.get("role") or "") == "user"), None)
        if first_user is not None:
            notice = _MessageShapingMixin._content_with_system_notice_marker(HOST_CONTEXT_NOTICE_AFTER_TASK + "\n\n" + body)
            rest.insert(first_user + 1, {"role": "developer", "content": notice})
            return [system, *rest], len(moved)
    notice = _MessageShapingMixin._content_with_system_notice_marker(HOST_CONTEXT_NOTICE_BEFORE_TASK + "\n\n" + body)
    return [system, {"role": "user", "content": notice}, *rest], len(moved)


def project_declared_system_prefix(target: Dict[str, Any], messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``split_leading_system_prefix`` for one send, its shape stamped on the per-call target.

    ``target["wire_layout"]`` rides into ``usage`` at the route's response normalizer (a
    per-call dict, never a thread-local, so a prospective build can never label another
    route's answer). Callers gate on the route: OpenAI's public API
    (``llm_openai_compatible._project_openai_family_system``) and the Codex backend
    (``llm_claudexor._request``), both of which reuse a donor's cached prefix only up to
    the end of the leading system unit / input item.
    """
    projected, moved_blocks = split_leading_system_prefix(messages)
    if moved_blocks:
        target["wire_layout"] = {"system_prefix_split": True, "moved_blocks": moved_blocks}
    return projected


def reset_native_messages(messages: list, route: dict, *, source: str, model: str) -> tuple[list, list]:
    """Apply an already-authorized account reset without replacing source content."""
    changed = []
    prepared = copy.deepcopy(messages)
    for message in prepared:
        native = message.get("nativeContinuation")
        if not isinstance(native, dict):
            continue
        old = native.get("route") or {}
        if (route.get("source") == old.get("source") == source
                and route.get("model") in (None, model)
                and any(route.get(key) and old.get(key) and route[key] != old[key]
                        for key in ("credentialProfileId", "accountFingerprint"))):
            changed.append({"old_route": old, "new_route": route})
            message.pop("nativeContinuation")
    return prepared, changed


def drop_source_native_messages(messages: list, *, source: str) -> tuple[list, list]:
    """Send one source's history without its continuations, keeping the content.

    The account-reset above answers a route that MOVED. This answers a route
    that refused its own continuation while standing still — an engine that
    binds a continuation to the model that produced it cannot replay it once
    another model answered on the same account. The canonical content and tool
    calls are the message either way, so nothing the caller said is lost.
    """
    changed = []
    prepared = copy.deepcopy(messages)
    for message in prepared:
        native = message.get("nativeContinuation")
        if isinstance(native, dict) and (native.get("route") or {}).get("source") == source:
            changed.append({"old_route": native.get("route") or {}, "new_route": {}})
            message.pop("nativeContinuation")
    return prepared, changed


def reset_native_payload(payload: dict, route: dict, *, source: str, model: str, turn_state: Any = None):
    """Apply an authorized no-start reset to both continuation surfaces."""
    messages, changed = reset_native_messages(
        payload["messages"], route, source=source, model=model)
    if not changed:
        messages, changed = drop_source_native_messages(payload["messages"], source=source)
    slot = payload.get("nativeContinuation")
    if not changed and not isinstance(slot, dict):
        return None
    updated = {**payload, "messages": messages}
    surface = ""
    if isinstance(slot, dict):
        updated.pop("nativeContinuation", None)
        if turn_state is not None and hasattr(turn_state, "envelope"):
            turn_state.envelope = None
        changed = [{"old_route": slot.get("route") or {}, "new_route": {}}, *changed]
        surface = "top_level_turn_slot"
    return updated, changed, surface


class _MessageShapingMixin:
    """Send-copy message transforms and reasoning-artifact predicates."""

    @classmethod
    def _copy_messages_with_cache_policy(
        cls,
        messages: List[Dict[str, Any]],
        *,
        allow_message_cache_control: bool,
        flatten_tool_content_blocks: bool,
        flatten_non_user_content_blocks: bool = False,
        allow_cache_ttl: bool = False,
    ) -> List[Dict[str, Any]]:
        cleaned = scrub_native_custody(messages)
        for msg in cleaned:
            for key in ("acceptance_observation", "_acceptance_observation", "review_feedback",
                        STABLE_PREFIX_BLOCKS_KEY):
                msg.pop(key, None)
            msg.pop("nativeContinuation", None)
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            role = msg.get("role")
            if (role == "tool" and flatten_tool_content_blocks) or (
                role != "user" and flatten_non_user_content_blocks
            ):
                # String-only roles: text blocks fold into one string, so host
                # metadata and cache markers never reach the wire either.
                msg["content"] = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            else:
                for block in content:
                    if isinstance(block, dict):
                        # Strict providers reject cache markers on empty text.
                        empty_text = (
                            block.get("type") == "text"
                            and not str(block.get("text") or "").strip()
                        )
                        if (allow_message_cache_control
                                and isinstance(block.get("cache_control"), dict)
                                and not empty_text):
                            # Keep TTL only where the route documents it.
                            ttl = str(block["cache_control"].get("ttl") or "")
                            block["cache_control"] = (
                                {"type": "ephemeral", "ttl": ttl}
                                if allow_cache_ttl and ttl in _VALID_CACHE_TTLS
                                else {"type": "ephemeral"}
                            )
                        else:
                            block.pop("cache_control", None)
                        # Known host metadata never leaves the send copy.
                        for key in ("_caption", "_source_path", "_context_capsule"):
                            block.pop(key, None)
        return cleaned

    # Provider-private reasoning blocks are valid only on their producing family.
    _REASONING_CONTENT_BLOCK_TYPES = frozenset({"thinking", "reasoning", "redacted_thinking"})

    @classmethod
    def _strip_openrouter_roundtrip_metadata(
        cls,
        messages: List[Dict[str, Any]],
        *,
        keep_reasoning_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """Strip provider-private reasoning round-trip artifacts that a DIFFERENT
        upstream family rejects: assistant-level ``reasoning``/``reasoning_details``/
        ``reasoning_content``/``response_id`` keys AND ``thinking``/``reasoning``
        CONTENT blocks (plus any stray ``signature`` on other blocks). Returns a
        deep copy; the canonical transcript is untouched.

        ``reasoning_content`` is the OpenAI-compatible direct-provider field name
        (GLM / Z.AI / cloud.ru Foundation Models, legacy vLLM) — distinct from the
        OpenRouter/Anthropic ``reasoning``/``reasoning_details`` shapes. Strict
        OpenAI-compatible servers (vLLM/SGLang) reject an echoed ``reasoning_content``
        with HTTP 400 ``Extra inputs are not permitted``, so it must be scrubbed on
        the cloudru / openai-compatible / local lanes too. DeepSeek is the third
        class — a server that REQUIRES its own echo (tool-bearing requests 400
        without the previous turns' ``reasoning_content``) — so its lane passes
        ``keep_reasoning_content=True`` to retain that one field while every
        other round-trip artifact is still stripped."""
        cleaned = scrub_native_custody(messages)
        for msg in cleaned:
            if not isinstance(msg, dict):
                continue
            msg.pop("nativeContinuation", None)
            if msg.get("role") != "assistant":
                continue
            msg.pop("reasoning", None)
            msg.pop("reasoning_details", None)
            if not keep_reasoning_content:
                msg.pop("reasoning_content", None)
            msg.pop("response_id", None)
            content = msg.get("content")
            if isinstance(content, list):
                kept: List[Any] = []
                for block in content:
                    if isinstance(block, dict):
                        btype = str(block.get("type") or "").strip().lower()
                        if btype in cls._REASONING_CONTENT_BLOCK_TYPES:
                            continue
                        block.pop("signature", None)
                    kept.append(block)
                msg["content"] = kept
        return cleaned

    @staticmethod
    def _replace_image_blocks_with_placeholder(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace image content-blocks with an explicit text placeholder for a
        model that has NO native vision — a raw ``image_url`` sent to a blind model
        is silently ignored or 404s. Mirrors the local llama.cpp and GigaChat lanes.
        Returns a deep copy; the canonical transcript is untouched."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for idx, block in enumerate(content):
                if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                    caption = str(block.get("_caption") or "").strip()
                    suffix = f" — {caption}" if caption else ""
                    content[idx] = {"type": "text", "text": f"[image omitted: model has no vision{suffix}]"}
        return cleaned

    @staticmethod
    def _content_with_system_notice_marker(content: Any) -> Any:
        marker = "[SYSTEM NOTICE]\n"
        if isinstance(content, list):
            out = copy.deepcopy(content)
            if out and isinstance(out[0], dict) and str(out[0].get("type") or "") in {"text", "input_text", "output_text"}:
                out[0]["text"] = marker + str(out[0].get("text") or "")
                return out
            return [{"type": "text", "text": marker}] + out
        return marker + str(content or "")

    @staticmethod
    def _is_deferrable_image_user_turn(msg: Dict[str, Any]) -> bool:
        """True for a USER message whose content carries an image block but NO tool_result
        block and NO tool_call_id — i.e. a mid-round injected image (view_image /
        native screenshot) that must not split an assistant tool_use from its matching
        tool_result. A user turn that IS a tool answer (Anthropic-style tool_result content
        block, or an OpenAI tool message) is never deferred (the negative guard)."""
        if str(msg.get("role") or "").strip().lower() != "user":
            return False
        if msg.get("tool_call_id"):
            return False
        content = msg.get("content")
        if not isinstance(content, list):
            return False
        has_image = False
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = str(block.get("type") or "")
            if btype == "tool_result":
                return False  # this user turn answers a tool call — never defer it
            if btype in {"image_url", "image"}:
                has_image = True
        return has_image

    @classmethod
    def _normalize_system_message_placement(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Demote runtime system notices after conversation start.

        (The same ``[SYSTEM NOTICE]`` marker has a second producer on the OpenAI
        family and the Claudexor route: ``split_leading_system_prefix`` projects the LEADING system
        message's declared mutable blocks to a notice BEFORE the first user turn.)

        Providers with strict chat templates require system messages to appear
        only before the first user/assistant/tool turn. Late notices are runtime
        reminders, so they keep recency as user notices. If a notice appears
        between an assistant tool-call message and its tool results, it is
        buffered until after the adjacent tool-result block.

        The same buffer also defers a mid-round image-bearing USER turn (P4a):
        view_image / native-screenshot injection can append a user(image) message
        between an assistant tool_use and its tool_result, which violates every
        provider's tool-call adjacency contract. Buffering it (then flushing after
        the window closes) keeps the tool_result adjacent to its tool_use. This is
        the single send-time chokepoint every provider builder funnels through, so
        the fix covers Anthropic/OpenAI/Gemini/GigaChat at once (Bible P2/P7).
        """
        out: List[Dict[str, Any]] = []
        buffered_notices: List[Dict[str, Any]] = []
        seen_non_system = False
        awaiting_tool_results = False

        def flush_buffered() -> None:
            nonlocal buffered_notices
            if buffered_notices:
                out.extend(buffered_notices)
                buffered_notices = []

        for original in messages:
            msg = copy.deepcopy(original)
            role = str(msg.get("role") or "").strip().lower()

            # P4a: defer an image-bearing user turn that lands inside an open
            # tool_use↔tool_result window — BEFORE the generic clear below, so it is
            # buffered (kept in order with any demoted system notice) rather than
            # inserted between the tool_calls and their results.
            if awaiting_tool_results and cls._is_deferrable_image_user_turn(msg):
                buffered_notices.append(msg)
                continue

            if awaiting_tool_results and role not in {"tool", "system"}:
                awaiting_tool_results = False
                flush_buffered()

            if role == "system" and seen_non_system:
                msg["role"] = "user"
                msg["content"] = cls._content_with_system_notice_marker(msg.get("content"))
                if awaiting_tool_results:
                    buffered_notices.append(msg)
                else:
                    out.append(msg)
                continue

            out.append(msg)
            if role != "system":
                seen_non_system = True
            if role == "assistant" and msg.get("tool_calls"):
                awaiting_tool_results = True

        flush_buffered()
        return out

    @classmethod
    def _has_replayed_reasoning_metadata(cls, messages: List[Dict[str, Any]]) -> bool:
        """PRESENCE predicate: True if the transcript carries ANY provider-private
        reasoning round-trip artifact — assistant ``reasoning``/``reasoning_details``/
        ``reasoning_content``/``response_id`` keys, or ``thinking``/``reasoning``
        CONTENT blocks (or a stray ``signature`` on a content block). Shape-blind on
        purpose: it answers "is there anything to strip?" for the REACTIVE paths
        (400 strip-and-retry, reroute entry). Whether an artifact is actually
        endpoint-BOUND is the separate SEALED question, answered shape-first by
        ``transcript_has_sealed_reasoning`` (ouroboros/reasoning_artifacts.py)."""
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if (
                msg.get("reasoning")
                or msg.get("reasoning_details")
                or msg.get("reasoning_content")
                or msg.get("response_id")
            ):
                return True
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = str(block.get("type") or "").strip().lower()
                    if btype in cls._REASONING_CONTENT_BLOCK_TYPES or block.get("signature"):
                        return True
        return False

    @staticmethod
    def _model_family(model: Any) -> str:
        """The upstream provider FAMILY of a model id — the part before the first
        '/' (``z-ai/glm-5.2`` -> ``z-ai``; ``anthropic/claude-…`` -> ``anthropic``).
        This is the boundary that matters for reasoning-signature validity: GLM and
        Claude both transit OpenRouter, so ``provider=='openrouter'`` is too coarse —
        the FAMILY produces (and alone can validate) a thinking-block signature."""
        norm = (normalize_model_identity(str(model or "")) or str(model or "")).strip().lower().lstrip("~")
        if "/" in norm:
            return norm.split("/", 1)[0]
        return norm

    @classmethod
    def sanitize_reasoning_on_model_switch(
        cls,
        messages: List[Dict[str, Any]],
        from_model: Any,
        to_model: Any,
    ) -> List[Dict[str, Any]]:
        """SSOT for cross-family model switches (cross-model fallback, switch_model,
        per-task model override): when the TARGET model belongs to a DIFFERENT
        provider family than the SOURCE, strip provider-private reasoning artifacts
        the target cannot validate — this is what kills the GLM->Claude fallback
        with a 400 ``Invalid `signature` in `thinking` block``. Same family ->
        return ``messages`` unchanged (preserve reasoning continuity). On a switch
        returns a sanitized COPY; the canonical transcript is never mutated."""
        switched = str(from_model or "").strip() != str(to_model or "").strip()
        has_native_custody = any(
            any(custody_private_key(key) for key in message)
            for message in messages if isinstance(message, dict)
        )
        prepared = scrub_native_custody(messages) if switched and has_native_custody else messages
        if switched and any(isinstance(message, dict) and "nativeContinuation" in message for message in prepared):
            prepared = copy.deepcopy(prepared)
            for message in prepared:
                if isinstance(message, dict):
                    message.pop("nativeContinuation", None)
        if cls._model_family(from_model) == cls._model_family(to_model):
            return prepared
        return cls._strip_openrouter_roundtrip_metadata(prepared)
