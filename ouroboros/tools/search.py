"""Web search tool — uses DuckDuckGo Instant Answer API (free, no API key required)."""

from __future__ import annotations

import json
import urllib.request
import urllib.parse
from typing import Any, Dict, List

from ouroboros.tools.registry import ToolContext, ToolEntry


def _web_search(ctx: ToolContext, query: str) -> str:
    """
    Search the web using DuckDuckGo Instant Answer API.
    Free, no API key required. Returns a JSON response with answer and related topics.

    Note: For richer results, DuckDuckGo may not always return a direct answer.
    In that case, related topics and abstract text are returned instead.
    """
    try:
        encoded_query = urllib.parse.quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Ouroboros/1.0 (research agent)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        answer = data.get("AbstractText", "").strip()
        answer_source = data.get("AbstractSource", "")
        answer_url = data.get("AbstractURL", "")

        # Collect related topics as supplementary results
        related = []
        for topic in (data.get("RelatedTopics") or [])[:8]:
            if isinstance(topic, dict):
                text = topic.get("Text", "").strip()
                first_url = topic.get("FirstURL", "")
                if text:
                    related.append({"text": text, "url": first_url})

        # Instant Answer (Definition, Calculation, etc.)
        instant = data.get("Answer", "").strip()
        instant_type = data.get("AnswerType", "")

        result: Dict[str, Any] = {}
        if instant:
            result["instant_answer"] = instant
            result["answer_type"] = instant_type
        if answer:
            result["answer"] = answer
            result["source"] = answer_source
            result["url"] = answer_url
        if related:
            result["related"] = related

        if not result:
            result["message"] = (
                f"DuckDuckGo returned no direct answer for: '{query}'. "
                "Try rephrasing your query or use browse_page for a specific URL."
            )

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        return json.dumps({"error": repr(e)}, ensure_ascii=False)


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("web_search", {
            "name": "web_search",
            "description": (
                "Search the web via DuckDuckGo Instant Answer API (free, no API key needed). "
                "Returns JSON with answer, source, and related topics. "
                "For full page content, use browse_page with a specific URL."
            ),
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"},
            }, "required": ["query"]},
        }, _web_search),
    ]
