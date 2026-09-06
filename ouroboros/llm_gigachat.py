"""The native GigaChat lane.

GigaChat owns its own transport library and its own message vocabulary: one
function call per turn, tool results as ``function``-role messages carrying JSON,
a system message only in first position, and a stricter schema validator. This
module owns that translation and the client whose auth the library refreshes.
"""


from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ouroboros.llm_attempt import (
    _attempt_request,
    _candidate_before_dispatch,
    _execute_candidate,
    _physical_candidate,
)


class _GigaChatLaneMixin:
    """GigaChat client construction, message conversion and dispatch."""

    # ------------------------------------------------------------------
    # GigaChat (native `gigachat` library — NOT OpenAI-compatible)
    # ------------------------------------------------------------------
    @staticmethod
    def _new_gigachat_client(
        target: Dict[str, Any],
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        """Build a GigaChat library client for the given target."""
        try:
            from gigachat import GigaChat
        except ImportError as exc:  # pragma: no cover - exercised only without the dep
            raise RuntimeError(
                "The 'gigachat' package is required to use gigachat:: models. "
                "Install it with: pip install gigachat"
            ) from exc
        kwargs: Dict[str, Any] = {
            "scope": str(target.get("scope") or "GIGACHAT_API_PERS"),
            "verify_ssl_certs": bool(target.get("verify_ssl_certs", True)),
        }
        for source, destination in (
            ("api_key", "credentials"), ("user", "user"), ("password", "password"),
            ("base_url", "base_url"),
        ):
            value = str(target.get(source) or "")
            # Provider Test carries an explicit access-token field to suppress
            # inherited auth.  Its empty credential is equally authoritative:
            # omitting it would let the library reload GIGACHAT_CREDENTIALS.
            if value or (source == "api_key" and "access_token" in target):
                kwargs[destination] = value
        if "access_token" in target:
            kwargs["access_token"] = str(target.get("access_token") or "")
        if timeout and timeout > 0:
            kwargs["timeout"] = float(timeout)
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        return GigaChat(**kwargs)

    def _get_gigachat_client(self, target: Dict[str, Any], timeout: Optional[float] = None):
        """Build (and cache) a GigaChat library client for the given target.

        Auth is whatever the env provides: an authorization key (``credentials``
        + ``scope``, OAuth) or ``user``/``password`` (basic auth). The library
        exchanges these for a short-lived access token and refreshes it
        automatically, so caching the client across calls is safe. Any other
        ``GIGACHAT_*`` setting present in the environment (e.g.
        ``GIGACHAT_PROFANITY_CHECK``) is picked up by the library itself.
        A caller-supplied per-request ``timeout`` becomes part of the cache key
        (the library takes it at construction), so the safety-supervisor timeout
        SSOT bounds this lane too (v6.54.3)."""
        credentials = str(target.get("api_key") or "")
        user = str(target.get("user") or "")
        password = str(target.get("password") or "")
        scope = str(target.get("scope") or "GIGACHAT_API_PERS")
        base_url = str(target.get("base_url") or "")
        verify = bool(target.get("verify_ssl_certs", True))
        timeout_key = float(timeout) if timeout and timeout > 0 else None
        cache_key = (credentials, user, password, scope, base_url, verify, timeout_key)

        if cache_key not in self._gigachat_clients:
            self._gigachat_clients[cache_key] = self._new_gigachat_client(target, timeout=timeout)
        return self._gigachat_clients[cache_key]

    @staticmethod
    def _gigachat_text(content: Any) -> str:
        """Flatten OpenAI message content (str or list of blocks) to plain text.

        GigaChat messages carry a plain-string ``content``; multipart blocks and
        any ``cache_control`` markers are collapsed/dropped here.
        """
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    if str(block.get("type") or "") in ("image_url", "image"):
                        # Explicit placeholder instead of a silent drop: the
                        # model (and the transcript reader) must know an image
                        # was present but not deliverable on this lane.
                        caption = str(block.get("_caption") or "").strip()
                        parts.append(f"[image omitted: model has no vision{f' — {caption}' if caption else ''}]")
                        continue
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))
            return "".join(parts)
        return str(content or "")

    @classmethod
    def _gigachat_function_result(cls, content: Any) -> str:
        """Return a function-result string that GigaChat accepts.

        GigaChat requires the ``function``-role message content to be a valid
        JSON document (it parses it server-side). Agent tool results are usually
        plain text (file contents, command output), so anything that isn't
        already valid JSON is wrapped as ``{"result": "<text>"}``.
        """
        text = cls._gigachat_text(content)
        try:
            json.loads(text)
            return text  # already valid JSON — pass through unchanged
        except Exception:
            return json.dumps({"result": text}, ensure_ascii=False)

    @classmethod
    def _gigachat_messages(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI-style messages to GigaChat's message list.

        Differences handled here:
        - role ``tool`` (a tool result) → role ``function`` with the function
          ``name`` resolved from the originating assistant ``tool_call_id``.
        - assistant ``tool_calls`` (a list) → a single ``function_call`` object.
          GigaChat supports ONE function call per turn, so parallel tool calls
          are collapsed to the first one.
        """
        messages = cls._normalize_system_message_placement(messages)
        out: List[Dict[str, Any]] = []
        call_id_to_name: Dict[str, str] = {}
        last_function_name: Optional[str] = None

        for msg in messages:
            role = str(msg.get("role") or "")

            if role == "tool":
                name = (
                    call_id_to_name.get(str(msg.get("tool_call_id") or ""))
                    or last_function_name
                    or "function"
                )
                out.append({
                    "role": "function",
                    "name": name,
                    "content": cls._gigachat_function_result(msg.get("content")),
                })
                continue

            effective_role = role if role in ("system", "user", "assistant") else "user"
            # GigaChat requires the system message to be the FIRST message and
            # rejects any later one ("system message must be the first message").
            # The agent injects system-reminders mid-conversation, so demote any
            # non-leading system message to a user message (keeps its content and
            # recency, which matters for reminders).
            if effective_role == "system" and out:
                effective_role = "user"

            gmsg: Dict[str, Any] = {
                "role": effective_role,
                "content": cls._gigachat_text(msg.get("content")),
            }

            tool_calls = msg.get("tool_calls")
            if role == "assistant" and tool_calls:
                # Record every id→name so following tool results resolve their
                # function name, but only the first call is sent to GigaChat.
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tcid = str(tc.get("id") or "")
                    tcname = str((tc.get("function") or {}).get("name") or "")
                    if tcid and tcname:
                        call_id_to_name[tcid] = tcname

                first = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
                fn = first.get("function") or {}
                name = str(fn.get("name") or "")
                args_raw = fn.get("arguments")
                arguments: Dict[str, Any] = {}
                if isinstance(args_raw, dict):
                    arguments = args_raw
                elif isinstance(args_raw, str) and args_raw.strip():
                    try:
                        arguments = json.loads(args_raw)
                    except Exception:
                        arguments = {}
                gmsg["function_call"] = {"name": name, "arguments": arguments}
                last_function_name = name

            out.append(gmsg)

        return out

    def _chat_gigachat(
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # The gigachat library owns its own httpx transport and proxy handling;
        # no_proxy (a macOS fork-safety flag for the OpenAI/requests paths) does
        # not apply here.
        del no_proxy

        client = self._get_gigachat_client(target, timeout=timeout)

        payload: Dict[str, Any] = {
            "model": str(target.get("resolved_model") or ""),
            "messages": self._gigachat_messages(messages),
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        functions = self._gigachat_functions(tools)
        if functions:
            payload["functions"] = functions
            # GigaChat accepts "auto"/"none" (or a specific {name}); it has no
            # strict "required", so anything else maps to "auto".
            payload["function_call"] = tool_choice if tool_choice in ("auto", "none") else "auto"

        # Current GigaChat-3 models can spend the full max_tokens budget on
        # hidden reasoning and return empty content/tool_calls when
        # reasoning_effort is sent. Keep the native path deterministic.

        candidate = _physical_candidate(payload)
        request = _attempt_request(target, candidate, source="llm.gigachat")
        completion = _execute_candidate(
            request,
            lambda: client.chat(candidate),
            _candidate_before_dispatch(candidate, request),
        )
        return self._normalize_gigachat_response(completion, target)

    def _normalize_gigachat_response(
        self,
        completion: Any,
        target: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Convert a GigaChat ``ChatCompletion`` into (message, usage) dicts.

        A GigaChat ``function_call`` becomes a single OpenAI-style ``tool_calls``
        entry (arguments re-encoded as a JSON string). GigaChat exposes no
        automatic cost source, so the normalized usage reports ``cost=None``.
        """
        choices = getattr(completion, "choices", None) or []
        first = choices[0] if choices else None
        gmsg = getattr(first, "message", None) if first is not None else None

        content = (getattr(gmsg, "content", "") or "") if gmsg is not None else ""
        message: Dict[str, Any] = {"role": "assistant", "content": content}

        function_call = getattr(gmsg, "function_call", None) if gmsg is not None else None
        if function_call is not None:
            name = getattr(function_call, "name", "") or ""
            arguments = getattr(function_call, "arguments", None)
            if not isinstance(arguments, dict):
                arguments = {}
            try:
                args_str = json.dumps(arguments, ensure_ascii=False)
            except Exception:
                args_str = "{}"
            message["tool_calls"] = [{
                "id": "call_0",
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }]
            # OpenAI convention: content is None when the turn is a tool call.
            if not content:
                message["content"] = None

        usage_obj = getattr(completion, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "prompt_tokens", 0) or 0) if usage_obj is not None else 0
        completion_tokens = int(getattr(usage_obj, "completion_tokens", 0) or 0) if usage_obj is not None else 0
        cached_tokens = int(getattr(usage_obj, "precached_prompt_tokens", 0) or 0) if usage_obj is not None else 0

        usage: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cached_tokens": cached_tokens,
            "provider": str(target.get("provider") or "gigachat"),
            "resolved_model": str(target.get("usage_model") or target.get("resolved_model") or ""),
            "cost": None,
            "cost_final": False,
        }

        return message, usage
