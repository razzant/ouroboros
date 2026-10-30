"""Transcript shaping for the wire and the reasoning-artifact contract.

Providers disagree about where a system message may appear, whether a tool
result may carry blocks, whose reasoning signatures they can validate, and how
much of a leading system message their prompt cache can reuse; our own lanes
that cannot carry image bytes leave a marker naming the lane. This module owns
the send-copy transforms that answer those disagreements
(``split_leading_system_prefix`` included) and the predicates that decide when
replayed reasoning is portable — never the canonical transcript, which every
transform copies before touching.
"""


from __future__ import annotations

import copy
from typing import Any, Dict, List

from ouroboros.anthropic_native_custody import custody_private_key, scrub_native_custody
from ouroboros.llm_attempt import _VALID_CACHE_TTLS
from ouroboros.provider_models import normalize_model_identity
from ouroboros.tool_result_record import TOOL_RESULT_RECORD_KEY
from ouroboros.review_history_view import REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY

# The Main context builder's declaration on its leading system message: how many leading
# text blocks are byte-stable across conversations (governance and books; with a memory
# view, identity and my sealed story too: two). A provider that reads a cache only inside
# the leading system group keeps only those blocks there, each as its own system item
# (``split_leading_system_prefix``). Host-only metadata: popped from every send copy.
STABLE_PREFIX_BLOCKS_KEY = "_stable_prefix_blocks"

# One send-copy policy. These are host custody/presentation fields, never model
# content. Strings (including JSON tool results/arguments) remain byte-identical.
HOST_ONLY_MESSAGE_KEYS = frozenset({
    "_host_context_kind", "_host_text_sha256", TOOL_RESULT_RECORD_KEY,
    REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY, STABLE_PREFIX_BLOCKS_KEY,
    "cache_control", "_context_capsule", "_evicted_image", "_caption", "_source_path", "_source_sha256",
    "_review_image_id", "_original_image_url", "acceptance_observation", "_acceptance_observation", "review_feedback",
    "_capture_source", "_canonical", "_requested_look", "_call_frame_source", "_call_frame_digest",
})
DELIVERY_MESSAGE_KEYS = frozenset({"_call_frame_source", "_call_frame_digest"})


def project_wire_messages(messages, *, delivery_marks=False):
    """Copy the structured message tree after host image/cache decisions.

    A live command may retain its source stamp until the adapter has used it to
    replace a native image item. The final provider mapping strips that stamp too.
    Opaque strings and unknown provider fields are never interpreted or rewritten.
    """
    hidden = HOST_ONLY_MESSAGE_KEYS - (DELIVERY_MESSAGE_KEYS if delivery_marks else frozenset())
    def project(value):
        if isinstance(value, dict):
            return {key: project(child) for key, child in value.items() if key not in hidden}
        if isinstance(value, (list, tuple)):
            return [project(child) for child in value]
        return value
    return project(messages)


def own_lane_image_marker(lane: str, caption: str = "") -> str:
    """The text standing where an image was on a lane of ours that cannot carry image bytes.

    It names our transport, never the model: the limit is ours (the local
    llama.cpp lane has no vision handler; the GigaChat lane flattens content to
    text), not evidence about what the model can see.
    """
    suffix = f" — {caption}" if caption else ""
    return f"[image omitted: our {lane} transport lane cannot carry images{suffix}]"


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
    messages: List[Dict[str, Any]], *, placement: str = "before_task", keep_marked: bool = False,
) -> tuple[List[Dict[str, Any]], int]:
    """Project a declared leading system message for a whole-section prompt cache.

    Returns ``(messages, moved_blocks)``. Applies only when the first message is a
    system message that carries ``STABLE_PREFIX_BLOCKS_KEY`` (the Main context builder's
    declaration, ``context_fit.ContextFitProjection.system_message``), its content is a
    list of text blocks longer than the declared count, and no second system message
    leads the transcript; every other shape — string systems, undeclared multi-block
    review prompts, several leading system messages — comes back unchanged with ``0``.
    Each kept block becomes its own system message; the remaining non-empty text blocks
    become ONE ``[SYSTEM NOTICE]`` message with a byte-stable provenance header: a user
    message right before the task (``before_task``) or a developer message right after
    the first user message (``after_task``). A pure function of the canonical messages
    (never mutated), so round N+1's copy extends round N's and the prospective wrap-up
    candidate equals the send. Measured 2026-09-25 on ``openai/gpt-6-sol``: the next
    conversation's first round read 198,797 of 393,676 tokens from cache instead of 0.

    ``keep_marked`` is for a route whose block markers become explicit cache breakpoints
    (OpenRouter's translation for OpenAI): the leading blocks that already carry their own
    ``cache_control`` after the declared ones stay separate system items too, since the
    breakpoint ending each one is written and looked up on its own. Without explicit
    breakpoints only the end of the leading system group is a lookup boundary, so a
    changed knowledge block kept there would also lose the governance prefix before it.
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
    while (keep_marked and declared < len(content) and isinstance(content[declared].get("cache_control"), dict)
           and content[declared]["text"].strip()):  # an empty block is never a system item of its own
        declared += 1
    moved = [block["text"] for block in content[declared:] if block["text"].strip()]
    if not moved:
        return messages, 0
    systems = []
    for block in content[:declared]:
        system = {key: copy.deepcopy(value) for key, value in leading.items() if key != STABLE_PREFIX_BLOCKS_KEY}
        system["content"] = [copy.deepcopy(block)]
        systems.append(system)
    rest = [copy.deepcopy(message) for message in messages[1:]]
    body = "\n\n".join(moved)
    if placement == "after_task":
        first_user = next((i for i, message in enumerate(rest)
                           if isinstance(message, dict) and str(message.get("role") or "") == "user"), None)
        if first_user is not None:
            notice = _MessageShapingMixin._content_with_system_notice_marker(HOST_CONTEXT_NOTICE_AFTER_TASK + "\n\n" + body)
            rest.insert(first_user + 1, {"role": "developer", "content": notice})
            return [*systems, *rest], len(moved)
    notice = _MessageShapingMixin._content_with_system_notice_marker(HOST_CONTEXT_NOTICE_BEFORE_TASK + "\n\n" + body)
    return [*systems, {"role": "user", "content": notice}, *rest], len(moved)


def project_declared_system_prefix(
    target: Dict[str, Any], messages: List[Dict[str, Any]], *, keep_marked: bool = False,
) -> List[Dict[str, Any]]:
    """``split_leading_system_prefix`` for one send, its shape stamped on the per-call target.

    ``target["wire_layout"]`` rides into ``usage`` at the route's response normalizer (a
    per-call dict, never a thread-local, so a prospective build can never label another
    route's answer). Callers gate on the route: OpenAI's public API
    (``llm_openai_compatible._project_openai_family_system``, which alone passes
    ``keep_marked`` for OpenRouter's explicit breakpoints) and the Codex backend
    (``llm_claudexor._request``, markers stripped), both of which otherwise reuse a donor's
    cached prefix only up to the end of the leading system group.
    """
    projected, moved_blocks = split_leading_system_prefix(messages, keep_marked=keep_marked)
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
        sealed = allow_message_cache_control and any(isinstance(m.get("cache_control"), dict) for m in cleaned)
        for msg in cleaned:
            for key in ("acceptance_observation", "_acceptance_observation", "review_feedback",
                        STABLE_PREFIX_BLOCKS_KEY, TOOL_RESULT_RECORD_KEY, REVIEW_HISTORY_MESSAGE_KEY, REVIEW_CONTEXT_INDEX_KEY,
                        "_review_image_id"):
                msg.pop(key, None)
            msg.pop("nativeContinuation", None)
            boundary = msg.pop("cache_control", None)
            # Keep user block boundaries stable after the rolling mark moves.
            # A live OpenRouter/Anthropic probe (2026-10-21) reused the tool prefix
            # but rewrote each tool-free tail when its former marked list became
            # a string; consistent send-copy blocks restored growing cache reads.
            # Unsealed caller prompts and marker-free lanes retain their shape.
            if sealed and msg.get("role") == "user" and isinstance(msg.get("content"), str) and msg["content"].strip():
                msg["content"] = [{"type": "text", "text": msg["content"]}]
            role = msg.get("role")
            if isinstance(msg.get("content"), list) and ((role == "tool" and flatten_tool_content_blocks) or (
                role != "user" and flatten_non_user_content_blocks
            )):
                # String-only roles: text blocks fold into one string, so host
                # metadata and block markers never reach the wire either; a row's
                # seal mark renders after the fold (only that row becomes one block).
                msg["content"] = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in msg["content"]
                )
            if allow_message_cache_control and isinstance(boundary, dict) and role in {"user", "tool"}:
                msg["content"] = cls._content_with_cache_boundary(msg.get("content"), boundary)
            content = msg.get("content")
            if not isinstance(content, list):
                continue
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
                    for key in HOST_ONLY_MESSAGE_KEYS - {"cache_control"}:
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
        keep_reasoning_details: bool = False,
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
        other round-trip artifact is still stripped. Direct MiniMax retains both
        ``reasoning_details`` and ``reasoning_content`` for same-route continuation,
        without converting either carrier. Cross-family switches still scrub them
        (``sanitize_reasoning_on_model_switch``); retention alone proves no
        cross-route portability."""
        cleaned = scrub_native_custody(messages)
        for msg in cleaned:
            if not isinstance(msg, dict):
                continue
            msg.pop("nativeContinuation", None)
            if msg.get("role") != "assistant":
                continue
            msg.pop("reasoning", None)
            if not keep_reasoning_details:
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
    def _replace_image_blocks_with_placeholder(messages: List[Dict[str, Any]], lane: str) -> List[Dict[str, Any]]:
        """Replace image blocks with ``lane``'s marker on a lane of ours that cannot
        carry image bytes (``own_lane_image_marker``). Whether a route receives an
        image is the send policy's decision (``vision_routing``); this transport
        limit only keeps base64 out of a text-only prompt. Returns a deep copy when
        anything changes; the canonical transcript is untouched."""
        if not any(isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image")
                   for msg in messages if isinstance(msg.get("content"), list) for block in msg["content"]):
            return messages
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for idx, block in enumerate(content):
                if isinstance(block, dict) and str(block.get("type") or "") in ("image_url", "image"):
                    content[idx] = {"type": "text", "text": own_lane_image_marker(
                        lane, str(block.get("_caption") or "").strip())}
        return cleaned

    @staticmethod
    def _content_with_cache_boundary(content: Any, marker: Dict[str, Any]) -> Any:
        """A send copy's rendering of the seal's row mark (``context_fit.seal_task_transcript``):
        ``marker`` on the last non-empty text block, a string becoming that one block.
        Content without such a block comes back unchanged; nothing is mutated."""
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
        for index in range(len(blocks) - 1, -1, -1) if isinstance(blocks, list) else ():
            block = blocks[index]
            if isinstance(block, dict) and block.get("type") == "text" and str(block.get("text") or "").strip():
                return [*blocks[:index], {**block, "cache_control": dict(marker)}, *blocks[index + 1:]]
        return content

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
    def _order_complete_tool_episodes(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """On a send copy, put a complete unambiguous batch's answers first.

        An episode ends at the next assistant or EOF. Inspect the whole episode:
        a late duplicate/foreign answer invalidates it even after all expected
        answers arrived. Call IDs are local to that episode. User answers with
        native tool_result blocks move whole, including any attached image.
        Missing/ambiguous IDs and incomplete batches retain their input order.
        No image predicate: eviction/captioning must not reorder earlier answers.
        """
        out = []
        start = 0
        while start < len(messages):
            opening = messages[start]
            out.append(opening)
            start += 1
            if str(opening.get("role") or "").lower() != "assistant":
                continue
            calls = opening.get("tool_calls") or []
            ids = [call.get("id") if isinstance(call, dict) else None for call in calls]
            native_ids = ([block.get("id") for block in opening["content"]
                           if isinstance(block, dict) and block.get("type") == "tool_use"]
                          if isinstance(opening.get("content"), list) else [])
            if ids and native_ids:
                continue  # two competing call declarations are not an unambiguous episode
            ids = ids or native_ids
            if not ids or any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(set(ids)) != len(ids):
                continue
            end = start
            while end < len(messages) and str(messages[end].get("role") or "").lower() != "assistant":
                end += 1
            answers, others, seen = [], [], []
            valid = True
            for msg in messages[start:end]:
                role = str(msg.get("role") or "").lower()
                content = msg.get("content")
                native = ([block.get("tool_use_id") for block in content
                           if isinstance(block, dict) and block.get("type") == "tool_result"]
                          if role == "user" and isinstance(content, list) else [])
                result_ids = [msg.get("tool_call_id")] if role == "tool" else native
                if result_ids:
                    valid = (valid and not (native and msg.get("tool_call_id"))
                             and all(isinstance(cid, str) and cid in ids for cid in result_ids))
                    seen.extend(result_ids)
                    answers.append(msg)
                else:
                    valid = valid and role in {"user", "system"} and not msg.get("tool_call_id")
                    others.append(msg)
            if valid and len(seen) == len(ids) and len(set(seen)) == len(ids):
                out.extend(answers + others)
            else:
                out.extend(messages[start:end])
            start = end
        return out

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

        Complete tool episodes are ordered structurally first, independent of
        image representation. Notice demotion/buffering remains separate and
        also applies to incomplete episodes. Canonical history is never changed.
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

        for original in cls._order_complete_tool_episodes(messages):
            msg = copy.deepcopy(original)
            role = str(msg.get("role") or "").strip().lower()

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
