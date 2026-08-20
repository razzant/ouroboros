"""Transcript shaping for the wire and the reasoning-artifact contract.

Providers disagree about where a system message may appear, whether a tool
result may carry blocks, what a blind model does with an image, and whose
reasoning signatures they can validate. This module owns the send-copy
transforms that answer those disagreements and the predicates that decide when
replayed reasoning is portable — never the canonical transcript, which every
transform copies before touching.
"""


from __future__ import annotations

import copy
from typing import Any, Dict, List

from ouroboros.llm_attempt import _VALID_CACHE_TTLS
from ouroboros.provider_models import normalize_model_identity


def _reasoning_signature_portable_across_or_providers(model: str) -> bool:
    """Whether replay signatures are verified portable across same-model providers."""
    m = str(model or "").strip().lstrip("~")
    return (
        m.startswith("anthropic/")
        or m.startswith("google/gemini-")
        or m.startswith("openai/")
    )


class _MessageShapingMixin:
    """Send-copy message transforms and reasoning-artifact predicates."""

    @classmethod
    def _copy_messages_with_cache_policy(
        cls,
        messages: List[Dict[str, Any]],
        *,
        allow_message_cache_control: bool,
        flatten_tool_content_blocks: bool,
        allow_cache_ttl: bool = False,
    ) -> List[Dict[str, Any]]:
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if msg.get("role") == "tool" and flatten_tool_content_blocks:
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
    def _strip_openrouter_roundtrip_metadata(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        the cloudru / openai-compatible / local lanes too."""
        cleaned = copy.deepcopy(messages)
        for msg in cleaned:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            msg.pop("reasoning", None)
            msg.pop("reasoning_details", None)
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

    @staticmethod
    def _has_openrouter_reasoning_details(messages: List[Dict[str, Any]]) -> bool:
        for msg in messages:
            if isinstance(msg, dict) and msg.get("reasoning_details"):
                return True
        return False

    @classmethod
    def _has_replayed_reasoning_metadata(cls, messages: List[Dict[str, Any]]) -> bool:
        """True if the transcript carries provider-private reasoning artifacts that
        a DIFFERENT upstream family cannot validate: assistant ``reasoning``/
        ``reasoning_details``/``reasoning_content``/``response_id`` keys, or
        ``thinking``/``reasoning`` CONTENT blocks (or a stray ``signature`` on a
        content block). Broader than ``_has_openrouter_reasoning_details`` (which
        only sees the top-level ``reasoning_details`` field)."""
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
        if cls._model_family(from_model) == cls._model_family(to_model):
            return messages
        return cls._strip_openrouter_roundtrip_metadata(messages)
