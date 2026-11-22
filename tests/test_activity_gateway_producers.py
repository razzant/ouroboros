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


def _rendered_queue_rows(state, queue_body):
    """Production Activity over both Python-produced bodies: each queue row's text and control facts, by task id."""
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
        return [button.dataset.id, {text: row.textContent,
            control: [button.dataset.root, button.dataset.budgetPaused, button.dataset.resume, button.textContent]}];
    }))));
} finally {activity.destroy(); cleanup.reverse().forEach(f => f());}
"""
    rendered = subprocess.run([NODE_BIN, "--input-type=module", "-e", script],
                              input=json.dumps({"state": state, "queue": queue_body}),
                              text=True, encoding="utf-8", capture_output=True, cwd=WEB_ROOT, timeout=30)
    assert rendered.returncode == 0, rendered.stderr
    return json.loads(rendered.stdout)


def _reads_around_a_retry(tmp_path, monkeypatch, *, census_first):
    """The census and the queue read with one retry of ``retried`` between them (the same id, its next attempt, started
    now), and the attempt numbers each read names."""
    import time

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    started = time.time() - 1490
    for wid, task_id in enumerate(("retried", "steady")):
        write_task_result(tmp_path, task_id, "running", chat_id=0)
        workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id},
                                    "worker_id": wid, "attempt": 1, "started_at": started}

    def queue_read():
        assert queue.persist_queue_snapshot(reason="activity-retry-test")
        return _queue_snapshot(tmp_path)

    first = state_fields(tmp_path) if census_first else queue_read()
    old = workers.RUNNING["retried"]
    workers.RUNNING["retried"] = {**old, "attempt": 2, "started_at": time.time(), "task": {**old["task"], "_attempt": 2}}
    second = queue_read() if census_first else state_fields(tmp_path)
    body, queue_body = (first, second) if census_first else (second, first)
    numbers = ({row["activity_id"]: row["task_attempt"] for row in body["active_chat_activities"]},
               {row["id"]: row["attempt"] for row in queue_body["running"]})
    return body, queue_body, numbers


def test_a_census_behind_a_retry_lends_the_running_attempt_none_of_its_words(tmp_path, monkeypatch):
    """The census still names attempt 1 of a row the published queue already runs as attempt 2 (an attempt number only
    grows): its words describe the earlier attempt, so the row says Activity unconfirmed beside the running attempt's
    own time. A row whose two reads name one attempt keeps both, and every row keeps its control."""
    import re

    body, queue_body, (census, queued) = _reads_around_a_retry(tmp_path, monkeypatch, census_first=True)
    assert (census["retried"], census["steady"], queued["retried"], queued["steady"]) == (1, 1, 2, 1)
    rows = _rendered_queue_rows(body, queue_body)
    retried, steady = rows["retried"]["text"], rows["steady"]["text"]
    assert "Working" not in retried and re.search(r"Activity unconfirmed · task · \d+s", retried), retried
    assert "Working" in steady and re.search(r"· task · \d+ min", steady), steady
    assert rows["retried"]["control"] == rows["steady"]["control"]


def test_a_queue_behind_a_retry_lends_the_census_words_no_time(tmp_path, monkeypatch):
    """The census already names attempt 2 while the published queue still times attempt 1: the census speaks for the
    current attempt and the queue's time is the earlier one's, so the row keeps the words without that time."""
    import re

    body, queue_body, (census, queued) = _reads_around_a_retry(tmp_path, monkeypatch, census_first=False)
    assert (census["retried"], census["steady"], queued["retried"], queued["steady"]) == (2, 1, 1, 1)
    rows = _rendered_queue_rows(body, queue_body)
    retried, steady = rows["retried"]["text"], rows["steady"]["text"]
    assert "Working · task" in retried and not re.search(r"· \d", retried), retried
    assert "Working" in steady and re.search(r"· task · \d+ min", steady), steady
    assert rows["retried"]["control"] == rows["steady"]["control"]


def test_activity_joins_a_helpers_phase_and_time_only_for_the_same_attempt(tmp_path, monkeypatch):
    import re
    import time

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    root, kid = 'clock-root', 'clock-helper'
    write_task_result(tmp_path, root, 'running', chat_id=7)
    write_task_result(tmp_path, kid, 'running', chat_id=7, root_task_id=root, parent_task_id=root)
    workers.RUNNING[root] = {'task': {'id': root, 'type': 'task', 'chat_id': 7, 'root_task_id': root},
                             'attempt': 1, 'worker_id': 1, 'started_at': time.time() - 1500}
    child = {'id': kid, 'type': 'task', 'chat_id': 7, 'root_task_id': root, 'parent_task_id': root}

    def read(attempt, elapsed, *, stamped=True):
        wait = owner_wait.set_owner_wait(tmp_path, kid, {
            'state': 'waiting', 'reason': 'sleep', 'sleep': {'mode': 'warm'}, 'task_attempt': attempt})
        workers.RUNNING[kid] = {'task': child, 'worker_id': 2, 'started_at': time.time() - elapsed,
                                'owner_wait': wait, **({'attempt': attempt} if stamped else {})}
        body = state_fields(tmp_path)
        assert kid not in {row['activity_id'] for row in body['active_chat_activities']}
        expected = attempt if stamped else None
        member = next(work['members']['states'][kid] for work in body['works']
                      if work['episode']['task_id'] == root)
        assert ('queue_attempt' in member, member.get('queue_attempt')) == (stamped, expected)
        assert queue.persist_queue_snapshot(reason='activity-helper-attempt-test')
        snapshot = _queue_snapshot(tmp_path)
        assert next(row['attempt'] for row in snapshot['running'] if row['id'] == kid) == expected
        return body, snapshot

    old, old_queue = read(1, 1500)
    new, new_queue = read(2, 6)
    unstamped, _ = read(1, 1500, stamped=False)

    def row(census, queue_body):
        return _rendered_queue_rows(census, queue_body)[kid]['text']

    old_phase = row(old, new_queue)
    assert 'Sleeping' not in old_phase and re.search(r'Activity unconfirmed · task · \d+s', old_phase), old_phase
    old_clock = row(new, old_queue)
    assert 'Sleeping' in old_clock and not re.search(r'· \d', old_clock), old_clock
    matched = row(new, new_queue)
    assert 'Sleeping' in matched and re.search(r'· task · \d+s', matched), matched
    unknown_clock = row(unstamped, old_queue)
    assert 'Sleeping' in unknown_clock and not re.search(r'· \d', unknown_clock), unknown_clock


def test_activity_background_lists_the_gateways_kept_services_and_stops_one_by_its_launch(tmp_path):
    """The real kept-service routes over a real kept launch, rendered by the production Activity module: the row says
    the service, its task's name, its state and printed address; Stop posts that launch with a reason; the stop's
    receipt is a fresh read. A failed list read says unknown, never "none"."""
    import os

    import pytest
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros.gateway.kept_services import kept_service_routes
    from ouroboros.launch_custody import open_local_service_log
    from tests._launch_stop_shared import _answers, _ledgered_service, _settle, write_scripts

    if os.name == "nt":
        pytest.skip("kept-service custody is read by the POSIX launch census")
    leader, (port, pid, birth) = _ledgered_service(tmp_path, write_scripts(tmp_path), "storybook-preview",
                                                   owner_task="producer", drive=tmp_path)
    log, stream = open_local_service_log(tmp_path, "producer", "storybook-preview",
                                         leader._ouroboros_custody["scope_marker"], keep_alive=True)
    with stream:
        stream.write(b"Storybook started on http://localhost:6006/\n")
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
