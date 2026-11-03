"""Wait recovery follows current attempts and preserves other fallback settings."""

import copy
import json
import subprocess
from pathlib import Path


NODE_BIN = str(Path.home() / ".claudexor" / "node" / "bin" / "node") if (Path.home() / ".claudexor" / "node" / "bin" / "node").exists() else "node"

import pytest

from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, apply_model_role_override
from tests.test_llm_claudexor import setup as subscription_transport
from tests.test_model_wait import live_wait as wait_fixture, _action_for

setup = subscription_transport
live_wait = wait_fixture
pytestmark = pytest.mark.serial


@pytest.mark.parametrize("shared_local", [False, True])
def test_fallback_persistence_keeps_shared_local_and_other_rows(shared_local):
    initial = {"OUROBOROS_MODEL_FALLBACKS": "first, second, third",
               "USE_LOCAL_FALLBACK": shared_local,
               MODEL_ACCOUNTS_KEY: {"main": "main-pin", "fallback": ["", "", ""]}}
    original = copy.deepcopy(initial)
    saved = apply_model_role_override(initial, role="fallback:1", model="replacement",
                                      credential_profile_id="", use_local=shared_local)
    assert saved["OUROBOROS_MODEL_FALLBACKS"] == "first, replacement, third"
    assert saved["USE_LOCAL_FALLBACK"] is shared_local
    assert json.loads(saved[MODEL_ACCOUNTS_KEY]) == initial[MODEL_ACCOUNTS_KEY]
    with pytest.raises(ValueError, match="Local applies to all fallbacks"):
        apply_model_role_override(initial, role="fallback:1", model="replacement",
                                  credential_profile_id="", use_local=not shared_local)
    assert initial == original


def test_fallback_account_persistence_does_not_author_an_absent_group_local_key():
    initial = {"OUROBOROS_MODEL_FALLBACKS": "claudexor::codex=first, claudexor::codex=second",
               MODEL_ACCOUNTS_KEY: {"fallback": ["first-pin", "second-pin"]}}
    saved = apply_model_role_override(initial, role="fallback:1", model="claudexor::codex=replacement",
                                      credential_profile_id="new-pin", use_local=False)
    assert "USE_LOCAL_FALLBACK" not in saved
    assert json.loads(saved[MODEL_ACCOUNTS_KEY])["fallback"] == ["first-pin", "new-pin"]


def test_fallback_local_change_remains_task_only_through_real_decision(live_wait, monkeypatch):
    from ouroboros import config, model_wait
    from ouroboros.task_results import load_task_result

    root, _, _, controller, _, decide = live_wait
    monkeypatch.setattr(config, "SETTINGS_PATH", root / "settings.json")
    initial = {"OUROBOROS_MODEL_FALLBACKS": "first, second", "USE_LOCAL_FALLBACK": False}
    (root / "settings.json").write_text(json.dumps(initial))
    original = (root / "settings.json").read_bytes()
    row = {"wait_id": "fallback-wait", "revision": 1, "task_attempt": 1,
           "state": "waiting", "role": "fallback:1"}
    model_wait.mutate_wait(root, "task-one", row["wait_id"], lambda _: row)
    controller.waits[row["wait_id"]] = dict(row)
    body = _action_for({**row, "task_id": "task-one"}, "switch", model="replacement",
                       credential_profile_id="", use_local=True, persist_role=True)
    refused = decide(body)
    assert refused.status_code == 400
    assert json.loads(refused.body)["reason_code"] == "invalid_role_update"
    assert "pending_action" not in load_task_result(root, "task-one")["model_waits"][row["wait_id"]]
    accepted = decide({**body, "persist_role": False})
    assert accepted.status_code == 202 and json.loads(accepted.body)["saved"] is False
    controller._drain_controls()
    assert controller.waits[row["wait_id"]]["_action"]["use_local"] is True
    assert (root / "settings.json").read_bytes() == original


@pytest.mark.parametrize("where", ["running", "pending"])
def test_current_attempt_projection_hides_historical_wait_in_real_js(tmp_path, monkeypatch, where):
    from ouroboros.gateway import state
    from supervisor import queue
    from supervisor.task_model_wait import model_waiting

    old = {"wait_id": "old-wait", "revision": 1, "task_attempt": 1,
           "role": "light", "reason": "quota", "state": "waiting"}
    task = {"id": "retry-task", "root_task_id": "retry-task", "delegation_role": "root",
            "chat_id": 1, "_attempt": 2, "model_waits": {old["wait_id"]: old}}
    meta = {"task": task, "attempt": 2, "started_at": 1}
    monkeypatch.setattr(queue, "RUNNING", {task["id"]: meta} if where == "running" else {})
    monkeypatch.setattr(queue, "PENDING", [task] if where == "pending" else [])
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(state, "_direct_turns_snapshot_safe", lambda **_kwargs: [])
    snapshot = state._chat_activities_snapshot_safe(tmp_path)
    assert len(snapshot) == 1 and snapshot[0]["task_attempt"] == 2
    assert snapshot[0]["model_waits"] == task["model_waits"]  # Preserve history.
    assert model_waiting(meta) is False
    result = subprocess.run([
        NODE_BIN, "--input-type=module", "-e",
        "import {createModelWaitController} from './modules/model_wait.js';"
        "let raw=''; for await (const chunk of process.stdin) raw += chunk;"
        "const activity=JSON.parse(raw)[0];"
        "const view=createModelWaitController({getRecord:()=>null,doc:()=>null});"
        "view.observe(activity.activity_id,activity);"
        "const waiting=view.waiting(activity.activity_id);"
        "view.destroy(); if (waiting) process.exit(1);",
    ], input=json.dumps(snapshot), text=True, capture_output=True,
        cwd=Path(__file__).resolve().parents[1] / "web")
    assert result.returncode == 0, result.stderr
