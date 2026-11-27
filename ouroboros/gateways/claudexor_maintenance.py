"""Claudexor maintenance and project registry transport over the owned HTTP client."""
from __future__ import annotations

import time
import uuid
from urllib.parse import quote, urlencode

from ouroboros.gateways.claudexor_metadata import STORE_IDENTITY


def bound_registration_supported(gateway) -> bool:
    catalog = getattr(gateway, "operations", None)
    if not callable(catalog):
        return False
    routes = [(op.get("method"), op.get("path")) for op in catalog() if isinstance(op, dict)]
    return ("POST", "/v2/projects/:id/release") in routes and ("GET", "/v2/projects/:id/registration") in routes


def registration_receipt_matches(body, project_id, store_id, root, states):
    """The same bound answer may arrive on the wire or in a persisted walk receipt."""
    return (isinstance(body, dict) and isinstance(root, str) and bool(root)
            and body.get("projectId") == project_id and body.get("storeId") == store_id
            and body.get("root") == root and body.get("state") in states)


CANDIDATE_MATCHES = ("same", "different", "unknown")


def candidate_match(body) -> str:
    """The engine's word on a ``candidate`` folder beside a ``present`` bound read: ``same`` or ``different`` in
    its own domain, else ``unknown`` (no field from an engine that ignores the query, or any other value)."""
    value = body.get("candidateMatch") if isinstance(body, dict) and body.get("state") == "present" else None
    return value if value in CANDIDATE_MATCHES else "unknown"


def supports_program_maintenance(gateway) -> bool:
    from ouroboros.gateways.claudexor import engine_at_least

    return engine_at_least(str(getattr(gateway, "engine_version", "") or ""), "6.0.0")


def maintenance_remaining(deadline):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise ClaudexorUnavailable("observation_timeout", "The maintenance observation deadline expired")
    return remaining


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


def _selected_program(value):
    fields = {"path", "realPath", "version", "versionOutput", "overrideEnv", "overrideValue"}
    return (isinstance(value, dict) and fields <= value.keys()
            and all(value[key] is None or isinstance(value[key], str) for key in fields))


class ClaudexorMaintenanceGateway:
    """No daemon lifecycle, installer, cache or retry policy lives in transport."""

    def _maintenance_require(self, operation: str, *, timeout_sec=None) -> None:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        if not supports_program_maintenance(self):
            raise ClaudexorUnavailable("maintenance_unavailable",
                                      "Update the app or connect an engine "
                                      "at version 6.0.0 or newer to install or update harness programs.", status_code=503)
        if not any(row.get("id") == operation for row in self.operations(**({"timeout_sec": timeout_sec} if timeout_sec is not None else {}))):
            raise ClaudexorUnavailable("maintenance_unavailable",
                                      "This engine does not provide the requested CLI maintenance operation",
                                      status_code=503)

    def _maintenance_request(self, operation: str, method: str, path: str, **kwargs):
        timeout = kwargs.pop("timeout_sec", None)
        deadline = kwargs.pop("deadline", None)
        if deadline is None and timeout is not None:
            deadline = time.monotonic() + timeout
        self._maintenance_require(operation, timeout_sec=maintenance_remaining(deadline))
        if deadline is not None:
            kwargs["timeout_sec"] = maintenance_remaining(deadline)
        return self._request(method, path, **kwargs)

    def maintenance_harnesses(self, harnesses=(), *, fresh=False, check_latest=False) -> dict:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        query = [("harness", value) for value in harnesses]
        query.extend((("fresh", str(fresh).lower()), ("checkLatest", str(check_latest).lower())))
        body = _object(self._maintenance_request("get:maintenance.harnesses", "GET",
                       "/v2/maintenance/harnesses?" + urlencode(query)))
        if not isinstance(body.get("observedAt"), str) or not isinstance(body.get("harnesses"), list):
            raise ClaudexorUnavailable("malformed_response", "Claudexor maintenance inventory is missing")
        for row in body["harnesses"]:
            if not self._maintenance_entry(row):
                raise ClaudexorUnavailable("maintenance_unavailable",
                                          "This engine does not provide the vendor program maintenance contract",
                                          status_code=503)
            if row["operation"] is not None:
                self._maintenance_operation(row["operation"])
        return body

    @staticmethod
    def _maintenance_entry(row) -> bool:
        """Recognize the supported wire shape, never translate legacy ownership/targets."""
        if not isinstance(row, dict) or not {
                "harness", "selection", "newest", "actions", "updateAvailable",
                "releaseNotesUrl", "operation", "observedAt"} <= row.keys():
            return False
        newest, actions = row["newest"], row["actions"]
        if (not isinstance(row["harness"], str) or not _selected_program(row["selection"])
                or row["updateAvailable"] not in ("yes", "no", "cannot_tell")
                or not isinstance(newest, dict) or not isinstance(actions, dict)):
            return False
        fields = {"not_read": {"source"}, "available": {"source", "version", "observedAt"},
                  "unavailable": {"source", "observedAt", "problem"}}.get(str(newest.get("status")))
        return (fields is not None and fields <= newest.keys()
                and all(isinstance(actions.get(name), dict)
                        and type(actions[name].get("available")) is bool
                        and "reason" in actions[name]
                        and isinstance(actions[name].get("commands"), list)
                        for name in ("install", "update", "exactVersion")))

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

    def maintenance_gc_last(self, *, timeout_sec=None) -> dict:
        """``{"last": record | None}``: the engine's last retention pass, read-only (``GET
        /v2/maintenance/gc`` never starts one). An engine without the route (404/405) answers the
        typed ``maintenance_unavailable``: its record is unsupported, never an empty one."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        try:
            body = _object(self._request("GET", "/v2/maintenance/gc", timeout_sec=timeout_sec))
        except ClaudexorUnavailable as exc:
            if exc.status_code in (404, 405):
                raise ClaudexorUnavailable("maintenance_unavailable", "This engine keeps no retention pass record",
                                           status_code=exc.status_code) from exc
            raise
        if "last" not in body or not (body["last"] is None or isinstance(body["last"], dict)):
            raise ClaudexorUnavailable("malformed_response", "Claudexor retention record is missing")
        return body

    def maintenance_operation(self, operation_id: str, *, timeout_sec=None, deadline=None) -> dict:
        return self._maintenance_operation(self._maintenance_request(
            "get:maintenance.operations.id", "GET",
            "/v2/maintenance/operations/" + quote(operation_id, safe=""), timeout_sec=timeout_sec, deadline=deadline), operation_id)

    def maintenance_cancel(self, operation_id: str, *, timeout_sec=None) -> dict:
        return self._maintenance_operation(self._maintenance_request(
            "post:maintenance.operations.id.cancel", "POST",
            "/v2/maintenance/operations/" + quote(operation_id, safe="") + "/cancel", timeout_sec=timeout_sec), operation_id)

    @staticmethod
    def _maintenance_operation(value, operation_id=None) -> dict:
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = _object(value)
        if (not isinstance(body.get("id"), str) or not body["id"]
                or not isinstance(body.get("state"), str)
                or (operation_id is not None and body["id"] != operation_id)):
            raise ClaudexorUnavailable("malformed_response", "Claudexor maintenance operation identity is invalid")
        if (not {"harness", "phase", "target", "requester", "installationHome", "commands",
                 "before", "after", "elsewhere", "activeRunsAtStart", "exitCode", "outcome",
                 "mutation", "termination", "limitations", "progress", "problem",
                 "createdAt", "startedAt", "finishedAt"} <= body.keys()
                or body["outcome"] not in ("changed", "not_changed", "changed_elsewhere", "unknown")
                or not isinstance(body["commands"], list)
                or any(body[key] is not None and not _selected_program(body[key]) for key in ("before", "after"))):
            raise ClaudexorUnavailable("malformed_response", "Claudexor vendor program receipt is unavailable")
        return body

    def project_registration_receipt(self, root: str) -> dict:
        """Keep the canonical folder and store from this POST, including on adoption."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("POST", "/v2/projects", json_body={"root": str(root)},
                             headers={"Idempotency-Key": uuid.uuid4().hex, **STORE_IDENTITY})
        if not isinstance(body, dict) or not isinstance(body.get("id"), str) or not body["id"]:
            raise ClaudexorUnavailable("malformed_response", "Project registration returned no id")
        return {"id": body["id"], "created": body.get("created") if isinstance(body.get("created"), bool) else None,
                "registration_root": body.get("root"), "registration_store_id": body.get("storeId")}

    def bound_registration(self, project_id: str, store_id: str, root: str, *, release: bool = False,
                           candidate: str = "") -> dict:
        """The addressed read or release of one registration bound by store and root. A read may name a
        ``candidate`` execution folder exactly as a start sends it; the engine compares it with this
        registration's folder in its own domain (``candidate_match``). The release never carries one."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        if not all(isinstance(value, str) and value for value in (project_id, store_id, root)):
            raise ClaudexorUnavailable("registration_unproven", "The registration binding is incomplete")
        path = "/v2/projects/" + quote(project_id, safe="")
        fields = {"expectedStoreId": store_id, "expectedRoot": root}
        if release:
            body = self._request("POST", path + "/release", json_body=fields)
        else:
            query = {**fields, **({"candidate": candidate} if candidate else {})}
            body = self._request("GET", path + "/registration?" + urlencode(query))
        states = ("released", "absent") if release else ("present", "absent")
        if not registration_receipt_matches(body, project_id, store_id, root, states):
            raise ClaudexorUnavailable("receipt_unverified", "Claudexor did not confirm the addressed registration")
        return body

    def project_registrations(self) -> list:
        """``GET /v2/projects`` read strictly (``ControlProjectListResponse``): every row an object with a nonempty
        string ``id`` and ``root``, no id twice; an empty list is a valid answer. Any other body raises
        ``malformed_response`` instead of reading as "no projects" (``find_project_id`` keeps its lenient read)."""
        from ouroboros.gateways.claudexor import ClaudexorUnavailable

        body = self._request("GET", "/v2/projects")
        rows = body.get("projects") if isinstance(body, dict) else None
        ids = [row.get("id") for row in rows] if isinstance(rows, list) and all(isinstance(r, dict) for r in rows) else None
        if (ids is None or not all(isinstance(value, str) and value for value in ids) or len(set(ids)) != len(ids)
                or not all(isinstance(row.get("root"), str) and row["root"] for row in rows)):
            raise ClaudexorUnavailable("malformed_response", "Claudexor project list is malformed")
        return rows
