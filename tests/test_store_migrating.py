"""History and its clients wait for publication without certifying absent facts."""
import asyncio
import json
import pathlib
import shutil
from types import SimpleNamespace

import pytest

from ouroboros import store, store_import, task_status
from ouroboros.gateway import history
from tests.group_run import run_in_group
from tests.test_history_tool_carriers import TS, _tool, _write


@pytest.mark.parametrize("fallback", [False, True], ids=["effective_result", "existence_check"])
def test_history_migrating_reaches_http_and_never_imports(tmp_path, monkeypatch, fallback):
    _write(tmp_path / "logs/chat.jsonl", [{"direction": "in", "task_id": "root", "ts": TS,
                                           "text": "Saved words", "chat_id": 1}])
    _write(tmp_path / "logs/tools.jsonl", [_tool("root")])
    _write(tmp_path / "task_results/root.json", [{"_schema_version": 1, "task_id": "root", "status": "completed"}])
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_, **__: [])
    endpoint = history.make_chat_history_endpoint(tmp_path)
    request = SimpleNamespace(query_params={"chat_id": "1"})
    with monkeypatch.context() as patch:
        if fallback:
            patch.setattr(task_status, "load_effective_task_result", lambda *a, **kw: None)
        patch.setattr(store_import, "ensure_current_store", lambda *a, **kw: pytest.fail("GET imported"))
        response = asyncio.run(endpoint(request))
    assert response.status_code == 503
    payload = json.loads(response.body)
    assert payload["reason_code"] == "migrating" and payload["window"]["complete"] is False
    assert store.probe(tmp_path) == ("absent", None) and not store.generation_stamped(tmp_path)
    store_import.ensure_current_store(tmp_path)
    response = asyncio.run(endpoint(request))
    assert response.status_code == 200
    payload = json.loads(response.body)
    carrier = next(row for row in payload["messages"] if row.get("system_type") == "task_evidence")
    assert carrier["task_terminal_status"] == "completed" and payload["window"]["complete"] is True


def test_pre_floor_lineage_waits_for_terminal_truth(tmp_path):
    _write(tmp_path / "task_results/child.json", [{"_schema_version": 1, "task_id": "child", "status": "completed"}])
    rows = [{"is_progress": True, "task_id": "child", "delegation_role": "subagent", "ts": TS},
            {"is_progress": True, "task_id": "recent", "ts": "2026-10-05T09:00:00Z"}]
    with pytest.raises(store.StoreMigrating):
        history._apply_window_quotas(tmp_path, 1, set(), rows, 1, 1)
    store_import.ensure_current_store(tmp_path)
    selected, *_ = history._apply_window_quotas(tmp_path, 1, set(), rows, 1, 1)
    assert [row["task_id"] for row in selected] == ["recent"]


def test_migrating_client_consumers():
    root = pathlib.Path(__file__).resolve().parents[1]
    node = shutil.which("node")
    assert node, "focused frontend verification requires Node"
    files = ["chat_history_loading", "dashboard_read_states", "task_card_recovery", "api_client_errors"]
    result = run_in_group([node, "--test", *[f"web/tests/{name}.test.js" for name in files]], cwd=root, timeout=60)
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
