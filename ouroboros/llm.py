"""
Ouroboros — LLM client.

The only module that communicates with the LLM API (Google AI Studio).
Contract: chat(), default_model(), available_models(), add_usage().

Uses Google AI Studio's OpenAI-compatible endpoint (free tier).
Base URL: https://generativelanguage.googleapis.com/v1beta/openai/
Auth: GOOGLE_API_KEY (from Google AI Studio: aistudio.google.com/app/apikey)

Free tier rate limits (as of 2025):
  gemini-2.5-pro:  2 RPM, 50 RPD
  gemini-2.0-flash: 15 RPM, 1500 RPD  ← recommended for light/bg model
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# NOTE: Update this to the latest available Gemini model in your AI Studio dashboard.
# Common free-tier model IDs: gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro
DEFAULT_LIGHT_MODEL = "gemini-2.0-flash"


def normalize_reasoning_effort(value: str, default: str = "medium") -> str:
    allowed = {"none", "minimal", "low", "medium", "high", "xhigh"}
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def reasoning_rank(value: str) -> int:
    order = {"none": 0, "minimal": 1, "low": 2, "medium": 3, "high": 4, "xhigh": 5}
    return int(order.get(str(value or "").strip().lower(), 3))


def add_usage(total: Dict[str, Any], usage: Dict[str, Any]) -> None:
    """Accumulate usage from one LLM call into a running total."""
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens", "cache_write_tokens"):
        total[k] = int(total.get(k) or 0) + int(usage.get(k) or 0)
    if usage.get("cost"):
        total["cost"] = float(total.get("cost") or 0) + float(usage["cost"])


def fetch_openrouter_pricing() -> Dict[str, Tuple[float, float, float]]:
    """
    No-op: Google AI Studio free tier has $0 cost.
    Kept for API compatibility with loop.py which calls this function.
    Returns empty dict — all cost calculations will return 0.
    """
    log.debug("fetch_openrouter_pricing: no-op (using Google AI Studio free tier, cost = $0)")
    return {}


class LLMClient:
    """Google AI Studio API wrapper (OpenAI-compatible endpoint). All LLM calls go through this class."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/",
    ):
        self._api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self._base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        reasoning_effort: str = "medium",
        max_tokens: int = 16384,
        tool_choice: str = "auto",
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Single LLM call. Returns: (response_message_dict, usage_dict with cost)."""
        client = self._get_client()
        effort = normalize_reasoning_effort(reasoning_effort)

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        resp = client.chat.completions.create(**kwargs)
        resp_dict = resp.model_dump()
        usage = resp_dict.get("usage") or {}
        choices = resp_dict.get("choices") or [{}]
        msg = (choices[0] if choices else {}).get("message") or {}

        # Google AI Studio free tier: cost is always $0
        usage["cost"] = 0.0
        usage.setdefault("cached_tokens", 0)
        usage.setdefault("cache_write_tokens", 0)

        return msg, usage

    def vision_query(
        self,
        prompt: str,
        images: List[Dict[str, Any]],
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1024,
        reasoning_effort: str = "low",
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Send a vision query to an LLM. Lightweight — no tools, no loop.

        Args:
            prompt: Text instruction for the model
            images: List of image dicts. Each dict must have either:
                - {"url": "https://..."} — for URL images
                - {"base64": "<b64>", "mime": "image/png"} — for base64 images
            model: VLM-capable model ID
            max_tokens: Max response tokens
            reasoning_effort: Effort level

        Returns:
            (text_response, usage_dict)
        """
        # Build multipart content
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            if "url" in img:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img["url"]},
                })
            elif "base64" in img:
                mime = img.get("mime", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img['base64']}"},
                })
            else:
                log.warning("vision_query: skipping image with unknown format: %s", list(img.keys()))

        messages = [{"role": "user", "content": content}]
        response_msg, usage = self.chat(
            messages=messages,
            model=model,
            tools=None,
            reasoning_effort=reasoning_effort,
            max_tokens=max_tokens,
        )
        text = response_msg.get("content") or ""
        return text, usage

    def default_model(self) -> str:
        """Return the single default model from env. LLM switches via tool if needed."""
        # NOTE: Update OUROBOROS_MODEL env var to the latest Gemini model available in AI Studio.
        # Free-tier options: gemini-3.1-pro-preview, gemini-2.5-pro, gemini-2.0-flash, gemini-1.5-pro
        return os.environ.get("OUROBOROS_MODEL", "gemini-3.1-pro-preview")

    def available_models(self) -> List[str]:
        """Return list of available models from env (for switch_model tool schema)."""
        main = os.environ.get("OUROBOROS_MODEL", "gemini-3.1-pro-preview")
        code = os.environ.get("OUROBOROS_MODEL_CODE", "")
        light = os.environ.get("OUROBOROS_MODEL_LIGHT", "")
        models = [main]
        if code and code != main:
            models.append(code)
        if light and light != main and light != code:
            models.append(light)
        return models
