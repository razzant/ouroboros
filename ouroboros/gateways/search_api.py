"""Built-in web search's provider wire, without policy: the sources a provider's search answer cites, and
DeepSeek's Search API (the query as written, one POST, a typed reading of its answer).

DeepSeek wire facts (api-docs.deepseek.com/guides/search, read 2026-11-11; no live send from this repository):
``POST {DEEPSEEK_BASE_URL}/tools/search`` with the account's Bearer key. ``query`` holds 1..128 Unicode
characters (code points here) and is not all whitespace; operators such as ``site:``, ``"..."`` and
``-term`` are part of it. ``count`` (10) and ``need_content`` (false) keep the provider's defaults. HTTP 200
carries ``results``, an empty list when nothing matched; an item has title, url, snippet (for display),
summary (for a model), content (empty without need_content), site_name, site_icon, date, authority
(0..4, -1 unknown) and rerank_score (0..3). Errors are OpenAI-shaped. A stale limit here either refuses
queries the provider takes or sends ones it refuses with a 400: re-read the guide before changing it.

The caller (``tools/search.py``) owns the client (built never to follow a redirect), its timeout, the key and
the physical-attempt accounting; nothing here retries, prices a request or picks a key.
"""
from __future__ import annotations

from typing import Any, Dict, List

from ouroboros.utils import sanitize_tool_result_for_log

QUERY_MAX_CHARS = 128
_KEPT = ("site_name", "date", "authority", "rerank_score")  # result metadata beside url, title and snippet


def plain(value: Any) -> Any:
    """A provider SDK object as plain data: dicts, lists and scalars; ``model_dump`` or public attributes."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return plain(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {
            str(k): plain(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return str(value)


def cited_sources(resp_obj: Any) -> List[Dict[str, str]]:
    """Each distinct URL a provider's search answer cites (a ``url_citation`` or a url with title or snippet)."""
    root = plain(resp_obj)
    sources: List[Dict[str, str]] = []
    seen: set[str] = set()

    stack: List[Any] = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            ntype = str(node.get("type") or "").lower()
            if "url_citation" in ntype or ("url" in node and ("title" in node or "snippet" in node)):
                url = sanitize_tool_result_for_log(str(node.get("url") or node.get("uri") or "").strip())
                if url and url not in seen:
                    seen.add(url)
                    sources.append({
                        "url": url,
                        "title": sanitize_tool_result_for_log(str(node.get("title") or node.get("name") or "").strip()),
                        "snippet": sanitize_tool_result_for_log(str(
                            node.get("snippet") or node.get("text") or node.get("content")
                            or node.get("cited_text") or node.get("description") or ""
                        ).strip()),
                    })
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)

    return sources


class SearchResponseUnreadable(RuntimeError):
    """An answered request whose body is no results list: never "no matches", and it may have been billed."""

    def __init__(self, response: Any, detail: str) -> None:
        super().__init__(f"DeepSeek Search answered HTTP {response.status_code} without a readable results list "
                         f"({detail}); whether it was billed is unknown")
        self.response, self.status_code = response, response.status_code


def query_refusal(query: Any) -> dict:
    """Why ``query`` cannot be sent as written (the documented contract, or text UTF-8 cannot carry); empty
    when it may be sent unchanged."""
    text = query if isinstance(query, str) else ""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        return {"error": f"DeepSeek Search accepts only text UTF-8 can carry; character {exc.start + 1} of this query "
                         f"is the surrogate code point U+{ord(text[exc.start]):04X}. Nothing was sent.",
                "reason_code": "argument_invalid"}
    if text.strip() and len(text) <= QUERY_MAX_CHARS:
        return {}
    problem = (f"at most {QUERY_MAX_CHARS} characters per query; this one has {len(text)}" if text.strip()
               else "a query that is not only whitespace")
    return {"error": f"DeepSeek Search accepts {problem}. Nothing was sent.", "reason_code": "query_limit",
            "query_chars": len(text), "query_max_chars": QUERY_MAX_CHARS}


def send_search(client: Any, query: str, *, timeout: Any) -> dict:
    """One POST of ``query`` on an OpenAI-SDK ``client`` bound to the DeepSeek base URL; the answer's payload.

    A non-2xx status, a redirect included, is the SDK's typed status error; a 2xx body without a results
    list of objects is ``SearchResponseUnreadable``; transport errors keep their cause chain. Only the DeepSeek
    key goes with it: the SDK's OpenAI account headers (from OPENAI_ORG_ID/OPENAI_PROJECT_ID) are omitted."""
    import httpx
    from openai import Omit

    response = client.post("/tools/search", cast_to=httpx.Response, body={"query": query}, options={
        "timeout": timeout, "headers": {"OpenAI-Organization": Omit(), "OpenAI-Project": Omit()}})
    try:
        payload = response.json()
    except ValueError as exc:
        raise SearchResponseUnreadable(response, type(exc).__name__) from exc
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise SearchResponseUnreadable(response, "no results array of objects")
    return payload


def search_usage(payload: dict) -> tuple[dict, None, bool]:
    """The ledger's evidence of one answered search: what the provider bills per (a request) and how many
    results it returned. The amount stays unknown here; unknown is never free."""
    return {"cost_basis": "search_request", "cost_evidence": {"results": len(payload["results"])}}, None, False


def search_result(payload: dict) -> dict:
    """The answer's tool result: each result's provider-written summary as the lead (not a page Ouroboros
    read), and the sources with their metadata. An empty ``results`` is an answered search with no match."""
    def safe(value: Any) -> Any:
        return sanitize_tool_result_for_log(value) if isinstance(value, str) else value

    sources, lines = [], []
    for item in payload["results"]:
        source = {key: safe(str(item.get(key) or "")) for key in ("url", "title", "snippet")}
        source.update((key, safe(item[key])) for key in _KEPT if item.get(key) is not None)
        summary = safe(str(item.get("summary") or source["snippet"]))
        if source["url"] or summary:
            lines.append(f"- {source['title']}: {summary} ({source['url']})")
        sources.append(source)
    return {"answer": "\n".join(lines) or "(no matches)", "answer_type": "summary", "sources": sources}
