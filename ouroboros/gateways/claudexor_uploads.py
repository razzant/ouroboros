"""Upload-owner transport over the owned Claudexor HTTP client.

Pure I/O (docs/DEVELOPMENT.md "Gateway Boundary Pattern"): the keyed document import
and the existing byte stage, both bodies preserved verbatim. The caller owns admission,
saved identities, source reading, store comparison and every recovery decision.
"""
from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import quote

# The engine's own feature marker for keyed document import (its catalog descriptor).
UPLOAD_IMPORT_OPERATION = {"method": "POST", "path": "/v2/uploads/imports",
                           "requestSchema": "ControlUploadImportRequest"}


def upload_import_supported(operations: List[Dict[str, Any]]) -> bool:
    """Does this engine's route catalog list the keyed import? No version is inferred."""
    return any(isinstance(row, dict) and all(row.get(name) == value for name, value in UPLOAD_IMPORT_OPERATION.items())
               for row in operations)


class ClaudexorUploadGateway:
    """No upload store, retry or source policy lives in transport."""

    def import_upload(self, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        """POST /v2/uploads/imports with the caller's saved key and exact body."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("POST", UPLOAD_IMPORT_OPERATION["path"], json_body=dict(request),
                             headers={"Idempotency-Key": str(idempotency_key)})
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "Document import returned no object")
        return body

    def put_upload_bytes(self, upload_id: str, data: bytes) -> Dict[str, Any]:
        """PUT /v2/uploads/:id/bytes: exactly the declared bytes, hashed by the engine."""
        body = self._request("PUT", f"/v2/uploads/{quote(str(upload_id), safe='')}/bytes", content_body=bytes(data),
                             headers={"Content-Type": "application/octet-stream"})
        return body if isinstance(body, dict) else {}
