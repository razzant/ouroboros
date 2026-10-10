"""Typed maintenance transport over the existing owned Claudexor HTTP client.

The engine owns installation and job facts. This mixin only negotiates the four
routes, preserves their bodies and marks an unconfirmed create for same-key rejoin.
"""
from __future__ import annotations

from urllib.parse import quote, urlencode


def maintenance_request_id(value: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 256
            or any(ord(char) < 32 or ord(char) > 126 for char in value)):
        raise ValueError("Idempotency-Key must be a nonempty printable ASCII request id (at most 256 characters)")
    return value


def _object(value):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if not isinstance(value, dict):
        raise ClaudexorUnavailable("malformed_response", "Claudexor maintenance returned no object")
    return value


class ClaudexorMaintenanceGateway:
    """No daemon lifecycle, installer, cache or retry policy lives in transport."""

    def _maintenance_require(self, operation: str) -> None:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        if not any(row.get("id") == operation for row in self.operations()):
            raise ClaudexorUnavailable("maintenance_unavailable",
                                      "This engine does not provide the requested CLI maintenance operation",
                                      status_code=503)

    def _maintenance_request(self, operation: str, method: str, path: str, **kwargs):
        self._maintenance_require(operation)
        return self._request(method, path, **kwargs)

    def maintenance_harnesses(self, harnesses=(), *, fresh=False, check_latest=False) -> dict:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        query = [("harness", value) for value in harnesses]
        query.extend((("fresh", str(fresh).lower()), ("checkLatest", str(check_latest).lower())))
        body = _object(self._maintenance_request("get:maintenance.harnesses", "GET",
                       "/v2/maintenance/harnesses?" + urlencode(query)))
        if not isinstance(body.get("harnesses"), list):
            raise ClaudexorUnavailable("malformed_response", "Claudexor maintenance inventory is missing")
        return body

    def maintenance_create(self, request: dict, request_id: str) -> dict:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        key = maintenance_request_id(request_id)
        self._maintenance_require("post:maintenance.operations")
        try:
            return self._maintenance_operation(self._request(
                "POST", "/v2/maintenance/operations",
                json_body=request, headers={"Idempotency-Key": key}))
        except ClaudexorUnavailable as exc:
            # A missing capability is a proved pre-submit refusal. Lost transport
            # or unreadable success after POST is not proof that nothing started.
            if exc.status_code < 400 and exc.code in {"daemon_unreachable", "malformed_response"}:
                exc.problem = {"code": exc.code, "message": str(exc), "retryable": False,
                               "context": {"requestId": key, "acceptance": "unknown",
                                           "remedy": "Retry the same body with the same Idempotency-Key"}}
            raise

    def maintenance_operation(self, operation_id: str) -> dict:
        return self._maintenance_operation(self._maintenance_request(
            "get:maintenance.operations.id", "GET",
            "/v2/maintenance/operations/" + quote(operation_id, safe="")), operation_id)

    def maintenance_cancel(self, operation_id: str) -> dict:
        return self._maintenance_operation(self._maintenance_request(
            "post:maintenance.operations.id.cancel", "POST",
            "/v2/maintenance/operations/" + quote(operation_id, safe="") + "/cancel"), operation_id)

    @staticmethod
    def _maintenance_operation(value, operation_id=None) -> dict:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = _object(value)
        if (not isinstance(body.get("id"), str) or not body["id"]
                or not isinstance(body.get("state"), str)
                or (operation_id is not None and body["id"] != operation_id)):
            raise ClaudexorUnavailable("malformed_response", "Claudexor maintenance operation identity is invalid")
        return body
