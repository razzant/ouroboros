"""Public dialogue shapes captured from e817e7ef1 before the SQL reader cutover.

Only installation paths, inode witnesses and the memory snapshot token are
normalized. Row identities, physical positions, cursor payloads, task projections
and human wording remain in the oracle.
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from types import SimpleNamespace

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import dialogue_import
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.logs import api_logs_tail
from ouroboros.gateway.task_events import api_task_events
from ouroboros.memory import Memory
from ouroboros.projects_registry import create_project
from tests._dialogue import seed
from tests._store_shared import seed_result
from tests.test_chat_history_paging import isolated_runtime  # noqa: F401


GOLDEN = Path(__file__).parent / "fixtures" / "dialogue_readers.json"


def conversation(root):
    project = create_project(root, "dialogue-golden", name="Dialogue golden")
    room = project["chat_id"]

    def row(second, text, chat=1, **fields):
        return {"ts": f"2026-10-01T00:00:{second:02d}Z", "chat_id": chat,
                "direction": "out", "text": text, **fields}

    seed(root, archives={
        "chat_20261001T000005.jsonl": [
            row(1, "Начало", direction="in", client_message_id="owner-first"),
            row(2, "Foreign room", room),
            row(3, "Archived answer", task_id="golden-task"),
        ],
        "progress_20261001T000005.jsonl": [
            {"ts": "2026-10-01T00:00:04Z", "chat_id": 1,
             "task_id": "golden-task", "content": "Archived progress"},
        ],
    }, live={
        "chat": [row(6, "Room question", room, direction="in", client_message_id="room-owner"),
                 row(7, "Document", task_id="golden-task", type="document", filename="report.txt",
                     download_url="/api/tasks/golden-task/artifacts/report.txt"),
                 row(8, "Room answer", room),
                 row(9, "Main newest")],
        "progress": [{"ts": "2026-10-01T00:00:06Z", "chat_id": 1,
                      "task_id": "golden-task", "content": "Current progress"}],
    })
    seed_result(root, "golden-task", {"task_id": "golden-task", "status": "completed",
                                    "ts": "2026-10-01T00:00:10Z", "result": "Finished"})
    # The readers read the store: the cutover takes the live files, the archive job the rest.
    assert dialogue_import.run_dialogue_import(root)["state"] == "done"
    # The task store's import receipt in logs/supervisor.jsonl carries a duration, so its length
    # varies with timing; the v2 cursor names that file's position. A known state keeps the oracle
    # independent of the clock and of what ran before it in a shared process.
    (root / "logs" / "supervisor.jsonl").write_bytes(b"")
    return room


def _unrooted(text, root):
    # The v2 event id is a JSON array naming the root, so there the root is JSON-escaped:
    # a Windows root's backslashes are doubled.
    for spelling in (json.dumps(str(root))[1:-1], str(root)):
        text = text.replace(spelling, "<root>")
    return text


def _normalize(value, root):
    if isinstance(value, list):
        return [_normalize(item, root) for item in value]
    if isinstance(value, dict):
        return {_unrooted(key, root): "<chain witness>" if key == "chain" else
                "<root-dependent view>" if key == "view" and value.get("v") == 2 else _normalize(item, root)
                for key, item in value.items()}
    if isinstance(value, str):
        return re.sub(r"snapshot=[0-9a-f]{64}", "snapshot=<snapshot>", _unrooted(value, root))
    return value


def _frames(response):
    assert response.status_code == 200, response.text
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def outputs(root, room):
    endpoint = make_chat_history_endpoint(root)
    result = {}
    for name, params in (("main", {}), ("room", {"chat_id": str(room)})):
        response = asyncio.run(endpoint(SimpleNamespace(query_params=params)))
        assert response.status_code == 200
        first = json.loads(response.body)
        result[name] = first
        replay = asyncio.run(endpoint(SimpleNamespace(query_params={**params, "cursor": first["page_cursor"]})))
        assert replay.status_code == 200
        result[name + "_replay"] = json.loads(replay.body)
        delta = asyncio.run(endpoint(SimpleNamespace(query_params={**params, "cursor": first["after_cursor"]})))
        assert delta.status_code == 200
        result[name + "_delta"] = json.loads(delta.body)
    app = Starlette(routes=[Route("/logs/{name}", api_logs_tail),
                            Route("/tasks/{task_id}/events", api_task_events, methods=["GET", "POST"])])
    app.state.drive_root = root
    with TestClient(app) as client:
        for stream in ("chat", "progress"):
            for task in ("", "golden-task"):
                response = client.get(f"/logs/{stream}", params={"limit": 20, "task_id": task})
                assert response.status_code == 200
                result[f"logs_{stream}_{task}"] = response.json()
        result["events_v1"] = _frames(client.get("/tasks/golden-task/events?wait=0"))
        result["events_v2"] = _frames(client.post("/tasks/golden-task/events", json={"v": 2, "wait": 0}))
    result["memory"] = Memory(root).chat_history(count=20)
    return _normalize(result, root)


def test_dialogue_endpoint_golden(tmp_path):
    room = conversation(tmp_path)
    actual = outputs(tmp_path, room)
    assert actual == json.loads(GOLDEN.read_text(encoding="utf-8"))
