"""Activity consumes the real gateway census for direct, queued, warm and helper actors."""

import json
import subprocess

from ouroboros import owner_wait
from ouroboros.gateway.tasks import _queue_snapshot
from ouroboros.task_results import write_task_result
from supervisor.active_activity import get_direct_activity_registry
from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree
from tests._budget_pause_exact_helpers import _install_queue
from tests._work_projection_shared import state_fields
from tests.test_direct_model_wait_activity import NODE_BIN, WEB_ROOT
from tests.test_owner_pause import _live_tree


def test_activity_renders_the_real_gateway_members_without_inventing_unknowns(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _live_tree(tmp_path, workers)
    warm = {"id": "warm-author", "type": "task", "chat_id": 7, "root_task_id": "warm-author"}
    workers.RUNNING[warm["id"]] = {"task": warm, "attempt": 1}
    write_task_result(tmp_path, warm["id"], "running", chat_id=7)
    assert request_owner_pause(warm["id"], request_id="warm")["ok"]
    wait = owner_wait.set_owner_wait(tmp_path, warm["id"], {"state": "waiting", "reason": "owner_pause"})
    workers.RUNNING[warm["id"]]["owner_wait"] = wait
    assert refresh_owner_pause_tree(warm["id"]) == "paused"
    sleeper = {"id": "warm-sleeper", "type": "task", "chat_id": 7, "root_task_id": "warm-sleeper"}
    write_task_result(tmp_path, sleeper["id"], "running", chat_id=7)
    sleeping = owner_wait.set_owner_wait(tmp_path, sleeper["id"], {
        "state": "waiting", "reason": "sleep", "sleep": {"mode": "warm"}, "task_attempt": 1})
    workers.RUNNING[sleeper["id"]] = {"task": sleeper, "attempt": 1, "owner_wait": sleeping}
    registry = get_direct_activity_registry()
    registry.register("direct-turn", 42, phase="thinking")
    try:
        availability = {"complete": True}
        body = state_fields(tmp_path, availability=availability)
        body["active_chat_activities_complete"] = availability["complete"]
        assert queue.persist_queue_snapshot(reason="activity-producer-test")
        queue_body = _queue_snapshot(tmp_path)
    finally:
        registry.unregister("direct-turn")
    members = {task: state for work in body["works"] for task, state in work["members"]["states"].items()}
    assert {task: (state["activity"], state["confirmed"]) for task, state in members.items()} == {
        "direct-turn": ("working", True), "root-1": ("working", True), "child-run": ("working", True),
        "child-new": ("queued", True), "warm-author": ("paused", True), "warm-sleeper": ("waiting", True)}
    # Both HTTP bodies are Python-produced. Only HTTP transport/DOM mechanics are stood in for;
    # Activity, workRowView, memberState and controls remain production JS.
    script = """
import { initActivity } from './modules/activity.js';
import { setup } from './tests/dashboard_dom_fixture.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const { state, queue } = JSON.parse(raw), cleanup = [];
const { mount, routes, ws } = setup({after: f => cleanup.push(f)});
const response = data => ({ok: true, status: 200, json: async () => data});
routes.set('/api/tasks?queue_only=1', response({queue}));
routes.set('/api/state', response({...state, bg_consciousness_enabled: false}));
routes.set('/api/schedules', response({tasks: []}));
const activity = initActivity({mount, ws});
try {
    await activity.refresh();
    const rows = mount.querySelector('[data-activity-section="queue"]').querySelectorAll('.activity-row');
    process.stdout.write(JSON.stringify(Object.fromEntries(rows.map(row => {
        const button = row.querySelector('button');
        return [button.dataset.id, {text: row.textContent, resume: button.dataset.resume === '1'}];
    }))));
} finally {activity.destroy(); cleanup.reverse().forEach(f => f());}
"""
    rendered = subprocess.run([NODE_BIN, "--input-type=module", "-e", script],
                              input=json.dumps({"state": body, "queue": queue_body}),
                              text=True, encoding="utf-8", capture_output=True, cwd=WEB_ROOT, timeout=30)
    assert rendered.returncode == 0, rendered.stderr
    rows = json.loads(rendered.stdout)
    assert set(rows) == set(members)
    for task, word in {"direct-turn": "Thinking", "root-1": "Working", "child-run": "Working",
                       "child-new": "Queued", "warm-author": "Paused",
                       "warm-sleeper": "Waiting · Sleeping until it is woken"}.items():
        assert word in rows[task]["text"], rows[task]
        assert "unconfirmed" not in rows[task]["text"]
        assert rows[task]["resume"] is (task == "warm-author")
