"""The MCP import preview (``POST /api/mcp/import/preview``): its own module beside ``contracts.py``.

Descriptive TypedDicts, the twin of ``web/modules/mcp_import_types.js``. The producer is
``ouroboros/mcp_import.py`` behind ``gateway.mcp.api_mcp_import_preview``. The request carries
the draft's matching metadata only; header values and addresses stay in the browser, and only
an entry's ``patch`` carries pasted values back to the draft."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

try:  # Python 3.11+
    from typing import Literal, NotRequired, TypedDict  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - CI supports Python 3.10.
    from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore[assignment]

__all__ = ["MCPImportDraftServer", "MCPImportHeaderChanges", "MCPImportPreviewEntry",
           "MCPImportPreviewRequest", "MCPImportPreviewResponse"]


class MCPImportDraftServer(TypedDict, total=False):
    """What import matching reads from one draft server: no header value or address."""

    id: str
    slug: str
    name: str
    transport: str
    header_names: List[str]  # the editor's current header list, the legacy row included
    legacy_header: Optional[str]  # which of those names is the stored legacy pair
    unreadable_headers: bool  # a malformed saved map that a file's headers would replace
    skill: str  # a row Connect tools bound to a skill: import never rewrites it


class MCPImportHeaderChanges(TypedDict):
    """An update whose entry names ``headers``: the file's list replaces the current one."""

    added: List[str]
    replaced: List[str]
    removed: List[str]


class MCPImportPreviewEntry(TypedDict):
    source_name: str
    server_id: str
    action: Literal["", "add", "update"]  # "" while problems keep the entry out
    transport: str
    arg_count: int
    header_names: List[str]
    env_names: List[str]
    unsupported_keys: List[str]
    problems: List[str]
    warnings: List[str]
    patch: Dict[str, Any]  # the only field carrying pasted values; empty while problems exist
    index: NotRequired[int]  # the updated draft position
    header_changes: NotRequired[MCPImportHeaderChanges]
    kept_header_names: NotRequired[List[str]]  # an update without headers keeps these
    legacy_header: NotRequired[str]
    unreadable_headers: NotRequired[bool]


class MCPImportPreviewRequest(TypedDict):
    text: str
    servers: List[Optional[MCPImportDraftServer]]  # null keeps a malformed draft row's position


class MCPImportPreviewResponse(TypedDict):
    ok: bool
    entries: List[MCPImportPreviewEntry]
    wrapper: NotRequired[Literal["mcpServers", "servers"]]
    ignored_keys: NotRequired[List[str]]
    error: NotRequired[str]
