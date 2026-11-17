"""The Logs tail and Raw endpoints over a synthetic drive, as an HTTP client reads them.

Shared by the Raw tests (tests/test_logs_raw.py, tests/test_logs_raw_context_view.py): a test module is not
imported by another test module (docs/development/02-naming-and-boundaries.md).
"""
from __future__ import annotations

import json

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.logs import api_logs_raw, api_logs_tail


def client(root):
    app = Starlette(routes=[Route("/api/logs/{name}", api_logs_tail), Route("/api/logs/{name}/raw", api_logs_raw)])
    app.state.drive_root = root
    return TestClient(app)


def write(root, row, name="events", *, append=False):
    path = root / "logs" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    with path.open("ab" if append else "wb") as handle:
        handle.write(raw)
    return raw


def preview(http, name="events", **params):
    response = http.get(f"/api/logs/{name}", params={"view": "compact", **params})
    assert response.status_code == 200, response.text
    return response.json()["entries"]


def read(http, row, name="events", **params):
    return http.get(f"/api/logs/{name}/raw", params={"ref": row["_raw"]["ref"], **params})
