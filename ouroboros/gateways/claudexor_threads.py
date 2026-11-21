"""Claudexor v3 conversation threads: create, append, turn lookup, read, export and conditional retirement.

Transport only, beside ``gateways/claudexor.py`` (docs/DEVELOPMENT.md "Gateway Boundary
Pattern"): the engine owns continuity, export inventories and the retirement decision; this
mixin carries strict bodies and returns the engine's answers unchanged. Export item bytes are
streamed into a caller-owned sink and hashed on the way, never held whole in memory; a short
body is not an error here, it is a size and hash the caller compares with what it was promised.

Problems keep ``ControlProblem`` authority (``ClaudexorGateway._problem``): a retirement answer's
``context.settled`` rides on ``exc.problem`` for the caller to read, so a 409 the engine bound to
an operation stays distinguishable from a transport or infrastructure failure, which proves
nothing. The capability is the catalog's ``POST /v2/threads/:id/retire`` row, never a version.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict
from urllib.parse import quote, urlencode

from ouroboros.gateways.claudexor_metadata import STORE_IDENTITY
from ouroboros.runtime_limits import CLAUDEXOR_WHOLE_BYTES_READ_SEC

THREAD_RETIRE_OPERATION = ("POST", "/v2/threads/:id/retire")
_CHUNK = 1024 * 1024


def _thread_path(thread_id: str, *rest: str) -> str:
    return "/v2/threads/" + "/".join(quote(str(part), safe="") for part in (thread_id, *rest))


def _object(value: Any, what: str) -> Dict[str, Any]:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if not isinstance(value, dict):
        raise ClaudexorUnavailable("malformed_response", f"Claudexor {what} returned no object")
    return value


class ClaudexorThreadGateway:
    """No continuity, retention or retry policy lives in transport."""

    def create_thread(self, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        """Create one durable v3 conversation thread.

        Claudexor owns continuity and profile routing. This client only carries
        the strict request and the caller's stable idempotency identity.
        """
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("POST", "/v2/threads", json_body=dict(request),
                             headers={"Idempotency-Key": str(idempotency_key)})
        if not isinstance(body, dict) or not str(body.get("id") or ""):
            raise ClaudexorUnavailable("malformed_response", "thread create returned no id")
        return body

    def start_thread_turn(self, thread_id: str, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        """Append one turn through the public v3 thread pipeline; its receipt names the admitting store."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("POST", _thread_path(thread_id, "turns"), json_body=dict(request),
                             headers={"Idempotency-Key": str(idempotency_key), **STORE_IDENTITY})
        if not isinstance(body, dict):
            raise ClaudexorUnavailable("malformed_response", "thread turn returned no handle")
        return body

    def lookup_thread_turn(self, thread_id: str, request: Dict[str, Any], *, idempotency_key: str) -> Dict[str, Any]:
        """``POST /v2/threads/:id/turns/lookup``: the turn a recorded append holds, read with that append's own
        body and key; nothing is appended or enqueued. The engine's answer is returned unprojected."""
        return _object(self._request("POST", _thread_path(thread_id, "turns", "lookup"), json_body=dict(request),
                                     headers={"Idempotency-Key": str(idempotency_key)}), "thread turn lookup")

    def get_thread(self, thread_id: str) -> Dict[str, Any]:
        """Read turns, native-session bindings, and continuity receipts."""
        body = self._request("GET", _thread_path(thread_id))
        return body if isinstance(body, dict) else {}

    def thread_retirement_supported(self) -> bool:
        """Does this engine's own route catalog list the conditional retirement route?"""
        method, path = THREAD_RETIRE_OPERATION
        return any(row.get("method") == method and row.get("path") == path for row in self.operations())

    def thread_export(self, thread_id: str, *, expected_store_id: str) -> bytes:
        """``GET /v2/threads/:id/export``: the exact manifest bytes (the engine writes nothing)."""
        return self._request("GET", _thread_path(thread_id, "export") + "?"
                             + urlencode({"expectedStoreId": str(expected_store_id)}), raw_bytes=True)

    def stream_thread_export_item(self, thread_id: str, export_id: str, item_id: str, sink: Any) -> Dict[str, Any]:
        """Stream one export item's bytes into ``sink``: ``{size, sha256, etag, content_length}`` as received."""
        import httpx

        from ouroboros.gateways.claudexor import _CONNECT_TIMEOUT_SEC, ClaudexorUnavailable

        digest, size, response = hashlib.sha256(), 0, None
        try:
            with self._client.stream("GET", _thread_path(thread_id, "export", export_id, "items", item_id),
                                     timeout=httpx.Timeout(CLAUDEXOR_WHOLE_BYTES_READ_SEC,
                                                           connect=_CONNECT_TIMEOUT_SEC)) as response:
                if response.status_code >= 400:
                    response.read()
                    raise self._problem(response, "GET")
                for chunk in response.iter_bytes(chunk_size=_CHUNK):
                    sink.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                headers = response.headers
        except httpx.HTTPError as exc:
            raise ClaudexorUnavailable(
                "daemon_unreachable", f"Claudexor thread export item read failed: {type(exc).__name__}",
                status_code=response.status_code if response is not None else 0) from exc
        length = headers.get("Content-Length")
        return {"size": size, "sha256": digest.hexdigest(), "etag": str(headers.get("ETag") or ""),
                "content_length": int(length) if str(length or "").isdigit() else None}

    def retire_thread(self, thread_id: str, request: Dict[str, Any]) -> Dict[str, Any]:
        """``POST /v2/threads/:id/retire`` with the strict ``ThreadRetireRequest``."""
        return _object(self._request("POST", _thread_path(thread_id, "retire"), json_body=dict(request),
                                     timeout_sec=CLAUDEXOR_WHOLE_BYTES_READ_SEC), "thread retirement")

    def abandon_thread_retirement(self, thread_id: str, operation_id: str) -> Dict[str, Any]:
        """``POST /v2/threads/:id/retire/abandon``: fence an unsettled operation, or learn it was accepted."""
        return _object(self._request("POST", _thread_path(thread_id, "retire", "abandon"),
                                     json_body={"operationId": str(operation_id)}, timeout_sec=CLAUDEXOR_WHOLE_BYTES_READ_SEC),
                       "thread retirement abandon")

    def thread_retirement(self, thread_id: str, *, operation_id: str, expected_store_id: str) -> Dict[str, Any]:
        """``GET /v2/threads/:id/retirement`` for one operation (passive; ``operation: null`` settles nothing)."""
        query = urlencode({"operationId": str(operation_id), "expectedStoreId": str(expected_store_id)})
        return _object(self._request("GET", _thread_path(thread_id, "retirement") + "?" + query),
                       "thread retirement read")
