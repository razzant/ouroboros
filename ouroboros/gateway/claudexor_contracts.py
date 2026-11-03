"""Read-side Claudexor browser envelopes, re-exported by gateway.contracts.

Shared facet types live here so the passive reader and the public contract
facade depend on one acyclic owner. These types describe, not validate, JSON.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    from typing import Literal, NotRequired, TypedDict
except ImportError:  # Python 3.10
    from typing_extensions import Literal, NotRequired, TypedDict

ClaudexorReadState = Literal["ok", "not_read", "failed"]


class ClaudexorStatusReads(TypedDict):
    """Independent status facets: harnesses/catalog, profiles/accounts and quota.
    ok means authoritative (including empty); not_read means never asked, including
    a discovery failure before fan-out; failed means no usable answer to a read.
    The login-capability manifest filter fails open and is not a reported facet."""

    catalog: ClaudexorReadState
    accounts: ClaudexorReadState
    quota: ClaudexorReadState


class ClaudexorPassiveReadError(TypedDict):
    """Safe read failure: fixed code, optional HTTP 400–599; no upstream prose."""

    code: str
    status_code: NotRequired[int]


class ClaudexorStatusResponse(TypedDict, total=False):
    """``GET /api/claudexor/status`` — owned-daemon lifecycle plus the daemon's
    own catalog/account/quota truth, each stamped with its read state. Read-only;
    never spawns the daemon (waking it is the owner-initiated POST)."""

    daemon: Dict[str, Any]
    config_dir: str
    harnesses: List[Dict[str, Any]]
    profiles: Dict[str, Any]
    quota: List[Dict[str, Any]]
    quota_absences: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    availability_scope: Dict[str, Any]  # Optional engine binding {model, requestKind, basis: quota_block}.
    resource_capabilities: Dict[str, bool]
    resource_capabilities_read: ClaudexorReadState
    reads: ClaudexorStatusReads
    # True when this read's roster is the unified account model (named rows, routing in
    # `profiles.accountPools`); a pre-unified roster fails `accounts` as `account_model_unsupported`.
    unified_accounts: bool
    # Per facet: `observed_at`, `stale`, `error`; a failed or unasked facet serves its last read (stale).
    facets: Dict[str, Dict[str, Any]]
    subagent_last_delegation: Dict[str, Any]
    error: str


class ClaudexorQuotaResponse(TypedDict):
    """``GET /api/claudexor/status?view=quota`` — passive roster/quota projection."""

    view: Literal["quota"]
    profiles: Dict[str, Any]
    quota: List[Dict[str, Any]]
    quota_absences: List[Dict[str, Any]]
    resources: NotRequired[List[Dict[str, Any]]]
    availability_scope: NotRequired[Dict[str, Any]]
    unified_accounts: bool
    reads: ClaudexorStatusReads
    # A pre-unified roster: accounts `failed` with read_errors.accounts.code `account_model_unsupported`.
    read_errors: Dict[Literal["discovery", "accounts", "quota"], ClaudexorPassiveReadError]
    timings_ms: Dict[str, int]
