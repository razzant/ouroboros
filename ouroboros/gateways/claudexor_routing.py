"""Account routing evidence beside the Accounts listing: a passive, optional read.

The engine owns the evidence (model-scoped refusals, temporary capability marks,
last service); this leaf only negotiates the capability rows from the serving
operation catalog and preserves the body. An engine that does not advertise the
query is not asked: older engines keep their strict response shape.
"""
from __future__ import annotations

from typing import Any, Dict, List

ROUTING_EVIDENCE_PATH = "/v2/accounts/routing-evidence"


def capability_marks_supported(operations: List[dict]) -> bool:
    from ouroboros.gateways.claudexor import operation_query_supported

    return operation_query_supported(operations, method="GET", path=ROUTING_EVIDENCE_PATH,
                                     name="includeCapabilities", value="true")


def read_capability_marks(gateway: Any) -> Dict[str, Any]:
    """The evidence with its ``capabilities`` rows; a malformed body is unavailable, never empty."""
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    body = gateway._request("GET", ROUTING_EVIDENCE_PATH + "?includeCapabilities=true")
    if not isinstance(body, dict) or not isinstance(body.get("capabilities"), list):
        raise ClaudexorUnavailable("malformed_response", "Invalid account routing evidence")
    return body
