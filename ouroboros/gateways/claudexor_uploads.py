"""Upload-owner transport over the owned Claudexor HTTP client.

Pure I/O (docs/DEVELOPMENT.md "Gateway Boundary Pattern"): the keyed document import, the
release of a managed import's resource and the existing byte stage, every body preserved
verbatim. The caller owns admission, saved identities, source reading, store comparison and
every recovery decision.
"""
from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import quote

# The engine's own feature marker for keyed document import (its catalog descriptor).
UPLOAD_IMPORT_OPERATION = {"method": "POST", "path": "/v2/uploads/imports",
                           "requestSchema": "ControlUploadImportRequest"}
# The marker for managed lifetime: the same row admits ``lifetime: "managed"`` on a keyed import and the
# release route. Without it nothing sends ``lifetime`` and nothing is released (resources stay indefinite).
RESOURCE_RELEASE_OPERATION = {"method": "POST", "path": "/v2/resources/:id/release",
                              "requestSchema": "ControlResourceReleaseRequest"}


def _advertised(operations: List[Dict[str, Any]], descriptor: Dict[str, str]) -> bool:
    return any(isinstance(row, dict) and all(row.get(name) == value for name, value in descriptor.items())
               for row in operations)


def upload_import_supported(operations: List[Dict[str, Any]]) -> bool:
    """Does this engine's route catalog list the keyed import? No version is inferred."""
    return _advertised(operations, UPLOAD_IMPORT_OPERATION)


def resource_release_supported(operations: List[Dict[str, Any]]) -> bool:
    """Does the catalog list the exact release row (and with it the managed import field)? No version is inferred."""
    return _advertised(operations, RESOURCE_RELEASE_OPERATION)


class ClaudexorUploadGateway:
    """No upload store, retry or source policy lives in transport."""

    def _upload_request(self, step: str, method: str, path: str, **request: Any) -> Any:
        """One control request of an upload, its wall time named on the measuring receipt (``daemon.upload.<step>``)."""
        from ouroboros.context_timing import timed

        return timed(f"daemon.upload.{step}", self._request, method, path, **request)

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
        body = self._upload_request("bytes", "PUT", f"/v2/uploads/{quote(str(upload_id), safe='')}/bytes",
                                    content_body=bytes(data), headers={"Content-Type": "application/octet-stream"})
        return body if isinstance(body, dict) else {}

    def cancel_upload(self, upload_id: str) -> Dict[str, Any]:
        """DELETE /v2/uploads/:id: abort the exact unfinished upload; no retry here."""
        body = self._request("DELETE", f"/v2/uploads/{quote(str(upload_id), safe='')}")
        return body if isinstance(body, dict) else {}

    def release_resource(self, resource_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """POST /v2/resources/:id/release with the caller's exact ``{expectedStoreId, sha256}``: naturally idempotent
        (no key), a repeat answers the identical body. It closes future references only; it deletes nothing."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("POST", f"/v2/resources/{quote(str(resource_id), safe='')}/release",
                             json_body=dict(request))
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "Resource release returned no object")
        return body
