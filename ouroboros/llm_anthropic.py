"""The native Anthropic lane.

Anthropic is not OpenAI-compatible: system text is its own block list, tool
calls and tool results are content blocks, thinking is a request-level setting,
and cache writes are reported per tier. This module owns that translation in
both directions plus the request that carries it, so the OpenAI-compatible
shape never leaks into the native wire format or back out of it.
"""


from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.anthropic_native_custody import (
    anthropic_replay_scoped,
    mark_replayed_receipts_consumed,
    native_content_for_replay,
    retain_native_assistant_content,
)
from ouroboros.llm_attempt import (
    _VALID_CACHE_TTLS,
    _attempt_request,
    _candidate_before_dispatch,
    _execute_candidate,
    _finalized_physical_candidate,
)
from ouroboros.llm_capability_policy import normalize_reasoning_effort
from ouroboros.request_wire_recovery import (
    finalize_wire_response,
    note_wire_send_failed,
    note_wire_send_succeeded,
    plan_next_wire_retry,
    request_wire_scoped,
)
from ouroboros.usage_accounting import UsageAccountingError, last_physical_attempt_capture


class _AnthropicLaneMixin:
    """Native Anthropic request building, dispatch and response normalisation."""

    @staticmethod
    def _stringify_anthropic_content(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _coalesce_anthropic_message(
        messages: List[Dict[str, Any]],
        role: str,
        content: List[Dict[str, Any]],
    ) -> None:
        if not content:
            return
        if messages and messages[-1].get("role") == role and isinstance(messages[-1].get("content"), list):
            messages[-1]["content"].extend(content)
            return
        messages.append({"role": role, "content": list(content)})

    @staticmethod
    def _anthropic_image_block(image_url: str) -> Optional[Dict[str, Any]]:
        url = str(image_url or "").strip()
        if not url:
            return None
        if url.startswith("data:") and ";base64," in url:
            header, data = url.split(",", 1)
            mime = header[5:].split(";", 1)[0] or "image/png"
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": mime,
                    "data": data,
                },
            }
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": url,
            },
        }

    def _anthropic_blocks_from_content(self, content: Any) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"type": "text", "text": content}] if content else []
        if not isinstance(content, list):
            text = self._stringify_anthropic_content(content)
            return [{"type": "text", "text": text}] if text else []

        blocks: List[Dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                if block:
                    blocks.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                text = self._stringify_anthropic_content(block)
                if text:
                    blocks.append({"type": "text", "text": text})
                continue

            block_type = str(block.get("type") or "").strip()
            if block_type in {"text", "input_text", "output_text"}:
                text = str(block.get("text") or "")
                if text:
                    normalized = {"type": "text", "text": text}
                    if isinstance(block.get("cache_control"), dict):
                        _ttl = str(block["cache_control"].get("ttl") or "")
                        normalized["cache_control"] = (
                            {"type": "ephemeral", "ttl": _ttl}
                            if _ttl in _VALID_CACHE_TTLS
                            else {"type": "ephemeral"}
                        )
                    blocks.append(normalized)
                continue
            if block_type == "image_url":
                image_url = str((block.get("image_url") or {}).get("url") or "")
                image_block = self._anthropic_image_block(image_url)
                if image_block:
                    blocks.append(image_block)
                continue
            if block.get("text"):
                normalized = {"type": "text", "text": str(block.get("text") or "")}
                if isinstance(block.get("cache_control"), dict):
                    _ttl = str(block["cache_control"].get("ttl") or "")
                    normalized["cache_control"] = (
                        {"type": "ephemeral", "ttl": _ttl}
                        if _ttl in _VALID_CACHE_TTLS
                        else {"type": "ephemeral"}
                    )
                blocks.append(normalized)
        return blocks

    @staticmethod
    def _sanitize_anthropic_tool_result_content(content: Any) -> Any:
        """Anthropic rejects empty tool_result content (and 400s on cache_control set
        for an empty text block). Drop empty text blocks, KEEP non-empty / non-text
        (image/document/search) blocks, and substitute a single placeholder only when
        the whole tool result would otherwise be empty (scalar ``""`` or list ``[]``)."""
        placeholder = "(no tool output)"
        if isinstance(content, list):
            cleaned = [
                b for b in content
                if not (
                    isinstance(b, dict)
                    and str(b.get("type") or "") == "text"
                    and not str(b.get("text") or "").strip()
                )
            ]
            return cleaned if cleaned else placeholder
        text = "" if content is None else str(content)
        return text if text.strip() else placeholder

    def _build_anthropic_messages(
        self,
        messages: List[Dict[str, Any]],
        target: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        messages = self._normalize_system_message_placement(messages)
        system_blocks: List[Dict[str, Any]] = []
        anthropic_messages: List[Dict[str, Any]] = []

        for message_index, msg in enumerate(messages):
            role = str(msg.get("role") or "").strip().lower()
            if role == "system":
                system_blocks.extend(self._anthropic_blocks_from_content(msg.get("content")))
                continue

            if role == "user":
                self._coalesce_anthropic_message(
                    anthropic_messages,
                    "user",
                    self._anthropic_blocks_from_content(msg.get("content")),
                )
                continue

            if role == "assistant":
                matching_results = []
                cursor = message_index + 1
                while cursor < len(messages) and str(messages[cursor].get("role") or "") == "tool":
                    matching_results.append(str(messages[cursor].get("tool_call_id") or ""))
                    cursor += 1
                assistant_blocks = (
                    native_content_for_replay(msg, target, matching_results)
                    if target is not None else None
                )
                if assistant_blocks is None:
                    assistant_blocks = self._anthropic_blocks_from_content(msg.get("content"))
                    for tool_call in msg.get("tool_calls") or []:
                        function = tool_call.get("function") or {}
                        raw_args = function.get("arguments")
                        parsed_args: Any = {}
                        if isinstance(raw_args, str):
                            try:
                                parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                            except Exception:
                                parsed_args = {"raw": raw_args}
                        elif raw_args is not None:
                            parsed_args = raw_args
                        if not isinstance(parsed_args, dict):
                            parsed_args = {"value": parsed_args}
                        assistant_blocks.append({
                            "type": "tool_use",
                            "id": str(tool_call.get("id") or ""),
                            "name": str(function.get("name") or ""),
                            "input": parsed_args,
                        })
                self._coalesce_anthropic_message(anthropic_messages, "assistant", assistant_blocks)
                continue

            if role == "tool":
                tool_use_id = str(msg.get("tool_call_id") or "")
                if not tool_use_id:
                    raise ValueError("Anthropic direct tool result is missing tool_call_id.")
                raw_content = msg.get("content")
                # Anthropic accepts list tool_result content; stringify only scalars/dicts.
                if isinstance(raw_content, list):
                    tool_result_content: Any = self._copy_messages_with_cache_policy(
                        [{"role": "tool", "content": raw_content}],
                        allow_message_cache_control=True,
                        flatten_tool_content_blocks=False,
                    )[0]["content"]
                else:
                    tool_result_content = self._stringify_anthropic_content(raw_content)
                tool_result_content = self._sanitize_anthropic_tool_result_content(tool_result_content)
                self._coalesce_anthropic_message(
                    anthropic_messages,
                    "user",
                    [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": tool_result_content,
                    }],
                )

        return system_blocks, anthropic_messages

    @staticmethod
    def _build_anthropic_tool_choice(tool_choice: Any) -> Optional[Dict[str, Any]]:
        if not tool_choice or tool_choice == "auto":
            return None
        if isinstance(tool_choice, dict):
            function = tool_choice.get("function") or {}
            name = str(function.get("name") or "").strip()
            if name:
                return {"type": "tool", "name": name}
            return None
        if not isinstance(tool_choice, str):
            return None
        if tool_choice in {"required", "any"}:
            return {"type": "any"}
        if tool_choice == "none":
            return {"type": "none"}
        return {"type": "tool", "name": tool_choice}

    @staticmethod
    def _cache_write_split(raw_usage: Dict[str, Any]) -> Dict[str, int]:
        """Anthropic's per-tier cache-write counters, when the provider reports them.

        With the extended (1h) tier live, ``usage.cache_creation`` splits
        ``cache_creation_input_tokens`` into ``ephemeral_5m_input_tokens`` /
        ``ephemeral_1h_input_tokens`` — a 1h request can legitimately produce BOTH
        (e.g. a server-tool block cached at the default tier beside the 1h prefix),
        and pricing must bill only the genuine 1h share at the extended ratio.
        Empty dict when the provider reported no split (older shapes) — the caller
        then bills every write at the reported tier, never a loosened ratio.
        """
        split = raw_usage.get("cache_creation") if isinstance(raw_usage, dict) else None
        if not isinstance(split, dict):
            return {}
        out: Dict[str, int] = {}
        for tier, key in (("5m", "ephemeral_5m_input_tokens"), ("1h", "ephemeral_1h_input_tokens")):
            try:
                value = int(split.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value > 0:
                out[tier] = value
        return out

    def _normalize_anthropic_response(
        self,
        resp_dict: Dict[str, Any],
        target: Dict[str, Any],
        prompt_cache_ttl: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        content_blocks = resp_dict.get("content") or []
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "").strip()
            if block_type == "text":
                text = str(block.get("text") or "")
                if text:
                    text_parts.append(text)
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                    },
                })

        raw_usage = resp_dict.get("usage") or {}
        usage: Dict[str, Any] = {
            # v6.77.0: Anthropic EXCLUDES cache reads/writes from `input_tokens`, while
            # `prompt_tokens` is the OpenAI-semantics TOTAL input every consumer assumes —
            # `pricing.regular_input = prompt_tokens - cached - cache_write` clamped fresh
            # input to 0 on a cache-heavy call (and cache_hit_rate could exceed 1.0).
            "prompt_tokens": (
                int(raw_usage.get("input_tokens") or 0)
                + int(raw_usage.get("cache_read_input_tokens") or 0)
                + int(raw_usage.get("cache_creation_input_tokens") or 0)
            ),
            "completion_tokens": int(raw_usage.get("output_tokens") or 0),
            "cached_tokens": int(raw_usage.get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(raw_usage.get("cache_creation_input_tokens") or 0),
            "provider": "anthropic",
            "resolved_model": str(target.get("usage_model") or target.get("resolved_model") or ""),
        }
        if prompt_cache_ttl:
            usage["prompt_cache_ttl"] = prompt_cache_ttl
        write_split = self._cache_write_split(raw_usage)
        if write_split:
            usage["cache_write_tokens_by_ttl"] = write_split
        if usage["prompt_tokens"] or usage["completion_tokens"]:
            from ouroboros.pricing import estimate_cost_optional

            estimated_cost = estimate_cost_optional(
                usage["resolved_model"],
                usage["prompt_tokens"],
                usage["completion_tokens"],
                cache_usage={
                    "cached_tokens": usage["cached_tokens"],
                    "cache_write_tokens": usage["cache_write_tokens"],
                    "prompt_cache_ttl": usage.get("prompt_cache_ttl"),
                    "cache_write_tokens_by_ttl": write_split or None,
                },
                provider="anthropic",
            )
            if estimated_cost is not None:
                usage["cost"] = estimated_cost
                usage["cost_estimated"] = True
        if usage.get("cost") is None:
            usage["cost"] = None
        usage["cost_final"] = bool(
            usage.get("cost") is not None and not usage.get("cost_estimated")
        )
        # Preserve legacy compatibility disclosure; normal adaptation uses request_wire.
        _clamp_note = self._pop_effort_clamp_disclosure()
        if _clamp_note:
            usage["reasoning_effort_clamped"] = _clamp_note
        _cache_note = self._pop_thread_disclosure("_cache_breakpoint_tls")
        if _cache_note:
            usage["prompt_cache_breakpoints_reduced"] = _cache_note

        message: Dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts),
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
        # Surface Anthropic's stop_reason for direct-lane empty-response classification.
        stop_reason = resp_dict.get("stop_reason")
        if stop_reason:
            message["stop_reason"] = str(stop_reason)
        message = retain_native_assistant_content(message, content_blocks, target)
        if tool_calls or str(message.get("content") or "").strip():
            message = mark_replayed_receipts_consumed(message)
        finalize_wire_response(message, usage)
        return message, usage

    def _build_remote_candidate(
        self, target: Dict[str, Any], messages: List[Dict[str, Any]],
        reasoning_effort: str, max_tokens: int, tool_choice: str,
        temperature: Optional[float], tools: Optional[List[Dict[str, Any]]],
        **remote_kwargs: Any,
    ) -> Dict[str, Any]:
        if target.get("provider") != "anthropic":
            return self._build_remote_kwargs(
                target, messages, reasoning_effort, max_tokens, tool_choice,
                temperature, tools, **remote_kwargs,
            )
        system, messages = self._build_anthropic_messages(messages, target)
        payload: Dict[str, Any] = {
            "model": str(target.get("resolved_model") or ""),
            "messages": messages, "max_tokens": max_tokens,
        }
        effort = normalize_reasoning_effort(reasoning_effort)
        if effort == "none":
            payload["thinking"] = {"type": "disabled"}
        elif effort:
            payload["thinking"] = {"type": "adaptive"}
            payload["output_config"] = {"effort": "low" if effort == "minimal" else effort}
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        anthropic_tools = self._build_anthropic_tools(tools)
        if anthropic_tools:
            payload["tools"] = anthropic_tools
            choice = self._build_anthropic_tool_choice(tool_choice)
            if choice:
                payload["tool_choice"] = choice
        return payload

    @request_wire_scoped
    @anthropic_replay_scoped
    def _chat_anthropic(
        self,
        target: Dict[str, Any],
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        reasoning_effort: str,
        max_tokens: int,
        tool_choice: str,
        temperature: Optional[float] = None,
        no_proxy: bool = False,
        timeout: Optional[float] = None,
        allow_server_web_search: bool = False,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        import requests

        payload = self._build_remote_candidate(
            target, messages, reasoning_effort, max_tokens, tool_choice, temperature, tools,
        )
        prompt_cache_ttl = self._normalize_payload_cache_ttl(target, payload)

        url = f"{str(target.get('base_url') or '').rstrip('/')}/messages"
        headers = {
            "x-api-key": str(target.get("api_key") or ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        request_timeout = float(timeout) if timeout and timeout > 0 else 120

        def _send(candidate: Dict[str, Any]):
            candidate = _finalized_physical_candidate(target, candidate, "messages")
            request = _attempt_request(target, candidate, source="llm.anthropic")

            def _post():
                if no_proxy:
                    with requests.Session() as session:
                        session.trust_env = False
                        sent = session.post(url, headers=headers, json=candidate, timeout=request_timeout)
                else:
                    sent = requests.post(url, headers=headers, json=candidate, timeout=request_timeout)
                if sent.status_code >= 400:
                    body_preview = (sent.text or "")[:2000]
                    raise requests.HTTPError(
                        f"{sent.status_code} {sent.reason} for url {sent.url}: {body_preview}",
                        response=sent,
                    )
                return sent

            try:
                result = _execute_candidate(
                    request,
                    _post,
                    _candidate_before_dispatch(candidate, request),
                )
                note_wire_send_succeeded(last_physical_attempt_capture())
                return result
            except UsageAccountingError:
                self._pop_effort_clamp_disclosure()
                note_wire_send_failed()
                raise
            except Exception:
                note_wire_send_failed()
                raise

        try:
            response = _send(payload)
        except UsageAccountingError:
            raise
        except Exception as exc:
            retry_payload = plan_next_wire_retry(payload, error=exc)
            if retry_payload is None:
                self._pop_effort_clamp_disclosure()
                raise
            for _ in range(8):
                try:
                    response = _send(retry_payload)
                    break
                except UsageAccountingError:
                    raise
                except Exception as retry_exc:
                    retry_payload = plan_next_wire_retry(
                        retry_payload, error=retry_exc,
                    )
                    if retry_payload is None:
                        self._pop_effort_clamp_disclosure()
                        raise
            else:
                self._pop_effort_clamp_disclosure()
                raise RuntimeError("request-wire recovery action bound exhausted") from exc
        return self._normalize_anthropic_response(
            response.json(),
            target,
            prompt_cache_ttl=prompt_cache_ttl,
        )
