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


def test_activity_background_lists_the_gateways_kept_services_and_stops_one_by_its_launch(tmp_path):
    """The real kept-service routes over a real kept launch, rendered by the production Activity module: the row says
    the service, its task's name, its state and printed address; Stop posts that launch with a reason; the stop's
    receipt is a fresh read. A failed list read says unknown, never "none"."""
    import os

    import pytest
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.kept_services import kept_service_routes
    from tests._launch_stop_shared import _answers, _ledgered_service, _settle, write_scripts

    if os.name == "nt":
        pytest.skip("kept-service custody is read by the POSIX launch census")
    leader, (port, pid, birth) = _ledgered_service(tmp_path, write_scripts(tmp_path), "storybook-preview",
                                                   owner_task="producer", drive=tmp_path)
    log = tmp_path / "services" / "producer" / "storybook-preview.log"
    log.parent.mkdir(parents=True)
    log.write_text("Storybook started on http://localhost:6006/\n", encoding="utf-8")
    write_task_result(tmp_path, "producer", "completed", title="Redesign the retention banner")
    app = Starlette(routes=kept_service_routes())
    app.state.drive_root = tmp_path
    try:
        with TestClient(app) as client:
            listed = client.get("/api/services/kept").json()
            (service,) = listed["services"]
            assert service["task_title"] == "Redesign the retention banner" and service["address"] == "http://localhost:6006"
            script = """
import { initActivity, keptServiceRowHtml, keptStopConfirmBody, keptStopToast } from './modules/activity.js';
import { setup } from './tests/dashboard_dom_fixture.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const { kept } = JSON.parse(raw), cleanup = [];
const { mount, routes, ws } = setup({after: f => cleanup.push(f)});
const response = data => ({ok: true, status: 200, json: async () => data});
routes.set('/api/tasks?queue_only=1', response({queue: {running: [], pending: []}}));
routes.set('/api/state', response({bg_consciousness_enabled: false}));
routes.set('/api/schedules', response({tasks: []}));
routes.set('/api/services/kept', response(kept));
const activity = initActivity({mount, ws});
const read = () => mount.querySelector('[data-activity-section="background"]').querySelectorAll('.activity-row')
    .map(row => ({text: row.textContent, stop: row.querySelector('[data-act="kept-stop"]')?.dataset.id || ''}));
try {
    await activity.refresh();
    const shown = read();
    routes.set('/api/services/kept', new Error('unreachable'));
    await activity.refresh();
    const failed = mount.querySelector('[data-activity-section="background"]').textContent;
    const stopped = (state) => keptStopToast({outcome: 'stopped', name: 'x', outputs: {state, detail: 'missing output: a.txt'}});
    process.stdout.write(JSON.stringify({shown, failed, toast: keptStopToast({outcome: 'not_confirmed', name: 'x',
        detail: 'its processes are not confirmed ended (running)'}), stopped: Object.fromEntries(
        ['captured', 'none_declared', 'incomplete', 'unavailable'].map(state => [state, stopped(state)])),
        confirm: keptStopConfirmBody('x'), captureConfirm: keptStopConfirmBody('x', true),
        owed: keptServiceRowHtml({name: 'x', launch_id: 'launch-1', state: 'capture_owed'}),
        retried: [true, false].map(ok => keptStopToast({outcome: 'not_found', name: 'x',
            ok, outputs: {state: ok ? 'captured' : 'incomplete', detail: 'missing output: a.txt'}}))}));
} finally {activity.destroy(); cleanup.reverse().forEach(f => f());}
"""
            rendered = subprocess.run([NODE_BIN, "--input-type=module", "-e", script], input=json.dumps({"kept": listed}),
                                      text=True, encoding="utf-8", capture_output=True, cwd=WEB_ROOT, timeout=30)
            assert rendered.returncode == 0, rendered.stderr
            view = json.loads(rendered.stdout)
            kept_rows = [row for row in view["shown"] if row["stop"]]
            assert [row["stop"] for row in kept_rows] == [service["launch_id"]]
            text = kept_rows[0]["text"]
            for words in ("storybook-preview", "Kept by a task · running", "from Redesign the retention banner",
                          "since ", "http://localhost:6006", "Stop"):
                assert words in text, text
            assert "Consciousness" in view["shown"][0]["text"], "the consciousness row keeps its place first"
            assert "Kept services could not be read. Whether any still run is unknown." in view["failed"]
            assert view["toast"][1] == "warning" and "not confirmed stopped" in view["toast"][0]
            assert view["stopped"]["captured"] == ["Stopped x. Its log and declared outputs are kept.", "success"]
            assert view["stopped"]["none_declared"] == ["Stopped x. Its log is kept.", "success"]
            for state in ("incomplete", "unavailable"):  # a stop whose outputs were not all kept never reads as done
                assert view["stopped"][state][1] == "warning" and "missing output: a.txt" in view["stopped"][state][0]
            # The confirmation promises the attempt, never its success; a later stop that captured says so.
            assert "tries to capture the outputs it declared" in view["confirm"] and "will be kept" not in view["confirm"]
            assert "Kept by a task · ended, declared outputs not captured yet" in view["owed"], view["owed"]
            assert '>Save outputs</button>' in view["owed"] and 'data-capture-only="1"' in view["owed"]
            assert "rechecks it, stops any remaining processes" in view["captureConfirm"]
            assert view["retried"][0] == ["x was no longer running; its declared outputs are now kept.", "success"]
            assert view["retried"][1][1] == "warning" and "missing output: a.txt" in view["retried"][1][0]

            refused = client.post(f"/api/services/kept/{service['launch_id']}/stop", json={})
            assert refused.status_code == 400 and refused.json()["outcome"] == "refused" and _answers(port)
            stopped = client.post(f"/api/services/kept/{service['launch_id']}/stop",
                                  json={"reason": "the owner stopped it from Activity › Background"})
            receipt = stopped.json()
            assert stopped.status_code == 200 and receipt["outcome"] == "stopped" and receipt["actor"] == "owner:gateway"
            # Spawned without the start path's declaration: nothing captured, and the receipt is not done.
            assert receipt["ok"] is False and receipt["outputs"]["state"] == "unavailable", receipt
            assert not _answers(port) and log.exists()
            assert client.get("/api/services/kept").json() == {"services": [], "complete": True, "note": ""}
    finally:
        _settle(leader, identities=[(pid, birth)])
