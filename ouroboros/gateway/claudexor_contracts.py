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


class ClaudexorQuotaResponse(TypedDict):
    """``GET /api/claudexor/status?view=quota`` — passive roster/quota projection."""

    view: Literal["quota"]
    profiles: Dict[str, Any]
    quota: List[Dict[str, Any]]
    quota_absences: List[Dict[str, Any]]
    unified_accounts: bool
    reads: ClaudexorStatusReads
    read_errors: Dict[Literal["discovery", "accounts", "quota"], ClaudexorPassiveReadError]
    timings_ms: Dict[str, int]
