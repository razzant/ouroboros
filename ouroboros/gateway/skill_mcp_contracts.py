"""Connect tools of a skill (``GET|POST /api/skills/{skill}/mcp-connect``): its own module beside ``contracts.py``.

Descriptive TypedDicts, the twin of ``web/modules/skill_mcp_types.js``. The producer is ``ouroboros/skill_mcp.py``
behind ``gateway.extensions.api_skill_mcp_connect`` (the confirm runs through ``skill_lifecycle_actions``). The
preview writes nothing; the confirm carries the preview's two source identities and the collateral consent."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore[assignment]

__all__ = ["SkillMcpConnectOtherServer", "SkillMcpConnectPreviewResponse", "SkillMcpConnectRequest",
           "SkillMcpConnectResponse", "SkillMcpConnectServer"]


class SkillMcpConnectServer(TypedDict):
    """One declared server and what Connect does to its Settings row."""

    id: str
    title: str
    change: Literal["add", "enable", "repair", "unchanged", "conflict"]
    conflict: str  # why a same-id row blocks it; "" otherwise
    row: Optional[Dict[str, Any]]  # the exact MCP_SERVERS row after the confirm; null for a conflict
    task_session: bool
    browser_policy: bool
    executable: NotRequired[Dict[str, Any]]  # preview: {command, args, cwd} derived now, never stored


class SkillMcpConnectOtherServer(TypedDict):
    """A row already on that turning MCP on would start too."""

    id: str
    name: str


class SkillMcpConnectPreviewResponse(TypedDict):
    ok: bool
    skill: NotRequired[str]
    content_hash: NotRequired[str]
    ready: NotRequired[bool]
    blockers: NotRequired[List[str]]
    next_action: NotRequired[str]
    servers: NotRequired[List[SkillMcpConnectServer]]
    mcp_enabled: NotRequired[bool]
    enables_mcp: NotRequired[bool]
    other_servers: NotRequired[List[SkillMcpConnectOtherServer]]
    requires_other_servers_confirmation: NotRequired[bool]
    changed: NotRequired[bool]
    conflict: NotRequired[bool]
    settings_digest: NotRequired[str]
    code: NotRequired[str]
    error: NotRequired[str]
    saved: NotRequired[bool]


class SkillMcpConnectRequest(TypedDict):
    expected_content_hash: str
    settings_digest: str
    confirm_other_servers: NotRequired[bool]


class SkillMcpConnectResponse(TypedDict):
    ok: bool
    saved: bool
    changed: NotRequired[bool]
    skill: NotRequired[str]
    mcp_enabled: NotRequired[bool]
    servers: NotRequired[List[SkillMcpConnectServer]]
    other_servers: NotRequired[List[SkillMcpConnectOtherServer]]
    changed_keys: NotRequired[List[str]]
    code: NotRequired[str]
    error: NotRequired[str]
