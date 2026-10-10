"""Model-route Settings previews and the maximum-response acknowledgement: their own module beside ``contracts.py``.

Descriptive TypedDicts, the twin of ``web/modules/model_route_types.js``. The producers are
``ouroboros/search_routes.py`` (the built-in ``web_search`` Source/Model) and
``ouroboros/response_limits.py`` (an exact route's maximum response), served by ``GET /api/settings``
with ``websearch_preview=1`` / ``response_limit_preview=1`` and ``POST /api/owner/capability-ack``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore[assignment]

__all__ = ["WebSearchLeg", "WebSearchRoute", "ResponseLimit", "ResponseLimitRoute",
           "ResponseLimitPreview", "ResponseLimitAckResponse"]


class WebSearchLeg(TypedDict):
    """One built-in search transport the resolver may try, in order."""

    source: str  # openai | openrouter | anthropic | ddgs
    backend: str
    model: str  # the wire model; empty for ddgs
    model_source: Literal["selection", "provider_default"]


class WebSearchRoute(TypedDict):
    """``search_routes.resolve_web_search_route``: skills, MCP and browser tools are not part of it."""

    scope: Literal["builtin_web_search"]
    source: str  # auto, or one strict source
    model: str  # the selection as saved or drafted; empty means provider defaults
    legs: List[WebSearchLeg]  # eligible, in order
    unavailable: List[WebSearchLeg]  # credential or package missing
    note: str
    error: NotRequired[str]  # the selection has no built-in transport
    ignored_legacy_model: NotRequired[str]  # a saved bare model the Anthropic source does not apply
    unapplied_model: NotRequired[str]  # under Auto, a saved model no built-in transport serves


class ResponseLimit(TypedDict):
    """``response_limits.ResponseLimit``: one exact route's maximum response, separate from its window."""

    max_output_tokens: int  # 0 = unknown, never a cap
    status: str
    source: str  # owner_ack, a metadata source, or none
    observed_at: str
    stale: bool
    route_fp: str


class ResponseLimitRoute(TypedDict):
    """The endpoint and account the previewed model's send reaches."""

    provider: str
    model: str
    base_url: str
    use_local: bool
    options: Optional[Dict[str, Any]]  # a subscription's exact account binding, else null


class ResponseLimitPreview(TypedDict):
    """``GET /api/settings?response_limit_preview=1``: stores nothing."""

    route: ResponseLimitRoute
    response_limit: ResponseLimit


class ResponseLimitAckResponse(TypedDict):
    """``POST /api/owner/capability-ack`` with ``max_output_tokens`` (0 removes the owner's maximum)."""

    ok: Literal[True]
    ack: ResponseLimit
