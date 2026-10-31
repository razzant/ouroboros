"""HTTP consumers receive current Main decisions from canonical task/control facts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import obligations, usage_store
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.state import api_state
from ouroboros.model_wait import mutate_wait
from ouroboros.owner_quiz import record_asked, record_publication, reconcile_terminal
from ouroboros.projects_registry import create_project
from ouroboros.task_results import load_task_result, write_task_result
from supervisor.message_bus import log_chat
from tests.test_swarm_host_admission import host  # noqa: F401


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """Real gateway handlers and source readers; no supervisor lifespan or workers."""
    from ouroboros import config
    from supervisor import active_activity, queue, state, workers

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(config, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(state, "DRIVE_ROOT", tmp_path)
    for name, relative in (("STATE_PATH", "state/state.json"),
                           ("STATE_LAST_GOOD_PATH", "state/state.last_good.json"),
                           ("STATE_LOCK_PATH", "locks/state.lock")):
        monkeypatch.setattr(state, name, tmp_path / relative)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    state.save_state({"owner_id": 1, "current_branch": "fixture", "current_sha": "a" * 40,
                      "bg_consciousness_enabled": False, "evolution_mode_enabled": False})
    usage_store.migrate_from_journal(tmp_path)
    pending, running = [], {}
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", tmp_path)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())
    app = Starlette(routes=[Route("/api/state", api_state),
                            Route("/api/chat/history", make_chat_history_endpoint(tmp_path))])
    app.state.drive_root = tmp_path
    app.state.app_start = 0.0
    with TestClient(app) as client:
        yield SimpleNamespace(root=tmp_path, client=client, pending=pending, running=running)


def _project_task(gateway, task_id):
    project = create_project(gateway.root, "project", name="Older Project")
    task = {"id": task_id, "root_task_id": task_id, "project_id": project["id"],
            "chat_id": project["chat_id"], "budget_drive_root": str(gateway.root), "_attempt": 3}
    write_task_result(gateway.root, task_id, "running", root_task_id=task_id,
                      project_id=project["id"], chat_id=project["chat_id"])
    return project, task


def _ask(gateway, task_id="old-task"):
    project, task = _project_task(gateway, task_id)
    block = record_asked(gateway.root, task_id, quiz_id="format", question="Which output format?",
                         options=["PDF", "HTML"], option_details=["Printable", "Searchable"],
                         recommended_index=1, stake="How you will read the result",
                         assumption="I will prepare HTML while you choose", chat_id=project["chat_id"])
    record_publication(gateway.root, task_id, "format", project["chat_id"])
    log_chat("out", project["chat_id"], 1, block["question"], task_id=task_id,
             record_type="quiz", quiz=block, drive_root=gateway.root, require_write=True)
    return project, task


def _state(gateway):
    response = gateway.client.get("/api/state")
    assert response.status_code == 200, response.text
    return response.json()


def test_optional_terminal_question_outside_history_and_sidebar_reaches_state(gateway):
    project, task = _ask(gateway)
    write_task_result(gateway.root, task["id"], "completed", result="Result ready")
    reconcile_terminal(gateway.root, task["id"])
    # The old ask is physically beyond the initial history window and its
    # Project is outside the bounded sidebar; neither bounds current questions.
    for index in range(120):
        log_chat("out", 1, 1, f"Newer Main message {index}", drive_root=gateway.root, require_write=True)
    for index in range(55):
        create_project(gateway.root, f"newer-{index}", name=f"Newer Project {index}")
    history = gateway.client.get("/api/chat/history", params={"chat_id": 1, "limit": 80})
    assert history.status_code == 200
    assert not any(row.get("quiz_id") == "format" for row in history.json()["messages"])

    payload = _state(gateway)
    assert project["id"] not in {row["id"] for row in payload["projects"]}
    assert project["chat_id"] in payload["project_chat_ids"]
    assert payload["current_questions_complete"] is True
    [question] = payload["current_questions"]
    assert question["task_id"] == task["id"] and question["quiz_id"] == "format"
    assert question["quiz_state"] == "expired_terminal"
    assert question["question"] == "Which output format?"
    assert question["options"] == ["PDF", "HTML"]
    assert question["option_details"] == ["Printable", "Searchable"]
    assert question["recommended_index"] == 1
    assert question["assumption"] == "I will prepare HTML while you choose"
    assert question["project_chat_id"] == project["chat_id"]
    assert payload["main_attention"] == []
    assert load_task_result(gateway.root, task["id"])["owner_quiz"]["format"]["state"] == "expired_terminal"


def test_state_carries_current_model_access_and_budget_resume_controls(gateway):
    project, task = _project_task(gateway, "access-task")
    wait = {"wait_id": "sign-in", "revision": 1, "task_attempt": 3, "state": "waiting",
            "role": "main", "reason": "auth", "model": "claudexor::codex=test-model",
            "source": "codex", "credential_harness": "codex", "credential_profile_id": "test-profile",
            "worker_slot_held": True, "auto_continue": True, "requires_owner_action": True}
    mutate_wait(gateway.root, task["id"], wait["wait_id"], lambda _: wait)
    # Previous attempts and waits that can automatically continue stay quiet.
    mutate_wait(gateway.root, task["id"], "stale", lambda _: {**wait, "wait_id": "stale", "task_attempt": 2})
    mutate_wait(gateway.root, task["id"], "automatic", lambda _: {**wait, "wait_id": "automatic", "reason": "quota", "requires_owner_action": False})
    gateway.running[task["id"]] = {"task": task, "attempt": 3, "started_at": 1.0}
    _, parked = _project_task(gateway, "budget-task")
    parked["_budget_pause"] = {"status": "paused_before_dispatch", "reason": "budget",
                               "replay_safe": True, "auto_resume": False}
    gateway.pending.append(parked)
    _, own_pause = _project_task(gateway, "owner-paused-task")
    own_pause["_budget_pause"] = {"status": "paused_before_dispatch", "reason": "owner"}
    gateway.pending.append(own_pause)

    payload = _state(gateway)
    assert payload["active_chat_activities_complete"] is True
    assert payload["main_attention_complete"] is True
    attention = {row["task_id"]: row for row in payload["main_attention"]}
    assert set(attention) == {"access-task", "budget-task", "owner-paused-task"}
    # An owner marker with no current tree fence awaits explicit selection (R3).
    assert attention["owner-paused-task"]["can_resume"] is True
    model = attention["access-task"]
    assert model["model_waits"] == {"sign-in": wait}
    assert model["task_attempt"] == 3 and model["attention_reason"] == "model_access"
    assert model["cancelable"] is True and model["can_resume"] is False
    assert model["project_id"] == project["id"]
    budget = attention["budget-task"]
    assert budget["phase"] == "budget_paused" and budget["pause_cause"] == "budget"
    assert budget["attention_reason"] == "budget"
    assert budget["can_resume"] is True and budget["cancelable"] is True
    assert budget["model_waits"] == {}
    # Reading controls is not permission to act: queue and result truth remain.
    assert gateway.pending == [parked, own_pause]
    assert load_task_result(gateway.root, "access-task")["model_waits"]["sign-in"] == wait


@pytest.mark.parametrize("damage", ["StoreBusy", "StoreMigrating", "StoreLost"])
def test_unavailable_current_question_index_is_incomplete_not_empty_truth(gateway, monkeypatch, damage):
    from ouroboros import store
    _ask(gateway)
    assert _state(gateway)["current_questions_complete"] is True
    with store.open_store(gateway.root, write=False) as txn:
        revision = txn.revision()
    with monkeypatch.context() as patched:
        patched.setattr(obligations, "members", lambda *a: (_ for _ in ()).throw(getattr(store, damage)(damage)))
        payload = _state(gateway)
    assert payload["current_questions"] == []
    assert payload["current_questions_complete"] is False
    assert payload["current_questions_gaps"] == [{"reason": "question_index_unavailable"}]
    assert payload["main_attention_complete"] is True
    assert load_task_result(gateway.root, "old-task")["owner_quiz"]["format"]["state"] == "open"
    with store.open_store(gateway.root, write=False) as txn:
        assert txn.revision() == revision


def test_live_wait_identity_and_revision_survive_older_result_display(gateway):
    _, task = _project_task(gateway, "live-revision")
    old = {"wait_id": "main", "revision": 1, "task_attempt": 3, "state": "waiting",
           "reason": "auth", "requires_owner_action": True, "auto_continue": False}
    mutate_wait(gateway.root, task["id"], "main", lambda _: old)
    settled = {**old, "state": "resolved", "revision": 3, "requires_owner_action": False}
    review = {**old, "wait_id": "review", "revision": 2, "role": "review"}
    task["model_waits"] = {"main": settled, "review": review}
    gateway.running[task["id"]] = {"task": task, "attempt": 3, "started_at": 1.0}
    payload = _state(gateway)
    [attention] = payload["main_attention"]
    assert attention["model_waits"] == {"review": review}
    [activity] = payload["active_chat_activities"]
    assert activity["model_waits"] == {"main": settled, "review": review}


def test_unreadable_live_wait_revision_discloses_gap_instead_of_old_controls(gateway):
    _, task = _project_task(gateway, "unreadable-revision")
    old = {"wait_id": "main", "revision": 1, "task_attempt": 3, "state": "waiting",
           "reason": "auth", "requires_owner_action": True, "auto_continue": False}
    mutate_wait(gateway.root, task["id"], "main", lambda _: old)
    task["model_waits"] = {"main": {**old, "revision": "unreadable"}}
    gateway.running[task["id"]] = {"task": task, "attempt": 3, "started_at": 1.0}
    payload = _state(gateway)
    assert payload["main_attention"] == []
    assert payload["main_attention_complete"] is False
    assert payload["main_attention_gaps"] == [{"reason": "model_wait_action_unknown",
                                               "task_id": task["id"], "wait_id": "main"}]


def _registries(root, pending=(), fences=None, running=None):
    return SimpleNamespace(PENDING=list(pending), BUDGET_ROOT_FENCES=dict(fences or {}),
                           RUNNING=dict(running or {}), DRIVE_ROOT=root)


_EXACT = {"exact_continuation": True, "pause_id": "p", "task_attempt": 1}


def _hold(reason):
    return {"reason": reason, "selected": False, "dispatchable": False}


@pytest.mark.parametrize("row,owner_paused,expected", [
    ({"_budget_pause": {**_EXACT, "reason": "budget"}}, False, True),
    ({"_budget_pause": {"status": "paused_before_dispatch", "reason": "budget", "replay_safe": True}}, False, True),
    ({"_budget_pause": {**_EXACT, "reason": "budget"}}, True, False),
    ({"_budget_pause": {**_EXACT, "reason": "owner"}}, False, True),
    ({"_budget_pause": {**_EXACT, "reason": "sleep", "sleep": {"sleep_id": "s"}}}, False, False),
    ({"_budget_pause": {**_EXACT, "reason": "sleep", "sleep": {"sleep_id": "s"}},
      "_budget_pause_hold": _hold("saved_sleep_recovery")}, False, True),
    ({"_budget_pause": {**_EXACT, "reason": "sleep", "sleep": {"sleep_id": "s"}},
      "_budget_pause_hold": _hold("panic_hold")}, False, True),
    ({"_budget_pause_hold": _hold("owner_restart_hold")}, False, True),
    ({"_budget_pause_hold": _hold("saved_work_hold")}, False, True),
    ({"_budget_pause_hold": _hold("continuation_writer_unsettled")}, False, False),
    ({"_budget_pause_hold": _hold("restore_refused:invalid_budget_fence_snapshot")}, False, True),
    ({"_continuation_prepared": "sha"}, False, True),
    ({"_resume_selection": {"selected_by": "owner"}}, False, False),
    ({}, False, False),
], ids=["exact-budget", "zero-dispatch-budget", "under-owner-fence", "owner-saved", "unheld-sleep",
        "held-saved-sleep", "panic-held-sleep", "restart-hold", "saved-work-hold", "writer-hold",
        "restore-refused", "continue-unconfirmed", "already-selected", "not-parked"])
def test_explicit_resume_follows_each_rows_release_rule(tmp_path, row, owner_paused, expected):
    """Only rows that nothing but an explicit Resume continues need the owner."""
    from ouroboros.gateway.state import _pending_resume_available

    task = {"id": "root", "root_task_id": "root", **row}
    assert _pending_resume_available(_registries(tmp_path, [task]), task, owner_paused=owner_paused) is expected


def test_member_resume_belongs_to_its_root_unless_only_the_owner_can_select_it(tmp_path, monkeypatch):
    from supervisor import events_budget
    from ouroboros.gateway.state import _pending_resume_available

    member = {"id": "child", "root_task_id": "root", "parent_task_id": "root",
              "_budget_pause_hold": _hold("owner_restart_hold")}
    root = {"id": "root", "root_task_id": "root"}
    # A running root with no live Resume grant: the model cannot select it; the owner must.
    assert _pending_resume_available(_registries(tmp_path, [member], running={"root": {"task": root}}),
                                    member, owner_paused=False) is True
    # A root that is itself parked: its own Resume is the one action.
    parked_root = {**root, "_budget_pause": {**_EXACT, "reason": "budget"}}
    assert _pending_resume_available(_registries(tmp_path, [parked_root, member]), member, owner_paused=False) is False
    assert _pending_resume_available(_registries(tmp_path, [member], fences={"root": {"status": "active"}}),
                                    member, owner_paused=False) is False
    # A live root grant lets the root's model select the member (owner Q9).
    monkeypatch.setattr(events_budget, "live_root_resume_grant", lambda *_a, **_k: {"grant_id": "g", "generation": 1})
    assert _pending_resume_available(_registries(tmp_path, [member], running={"root": {"task": root}}),
                                    member, owner_paused=False) is False


def test_census_resume_and_stop_follow_owner_facts_and_children_stay_out_of_the_census(gateway):
    project, held = _project_task(gateway, "held-sleep")
    held["_budget_pause"] = {**_EXACT, "reason": "sleep", "sleep": {"sleep_id": "s"}, "task_attempt": 3}
    held["_budget_pause_hold"] = _hold("saved_sleep_recovery")
    _, sleeping = _project_task(gateway, "unheld-sleep")
    sleeping["_budget_pause"] = {**_EXACT, "reason": "sleep", "sleep": {"sleep_id": "s"}, "task_attempt": 3}
    _, root = _project_task(gateway, "running-root")
    child = {"id": "held-child", "root_task_id": "running-root", "parent_task_id": "running-root",
             "delegation_role": "subagent", "project_id": project["id"], "chat_id": project["chat_id"],
             "budget_drive_root": str(gateway.root), "_budget_pause_hold": _hold("owner_restart_hold")}
    gateway.pending.extend([held, sleeping, child])
    gateway.running["running-root"] = {"task": root, "attempt": 3, "started_at": 1.0}

    payload = _state(gateway)
    census = {row["activity_id"]: row for row in payload["active_chat_activities"]}
    assert set(census) == {"held-sleep", "unheld-sleep", "running-root"}
    assert census["held-sleep"]["can_resume"] is True and census["unheld-sleep"]["can_resume"] is False
    assert all(row["cancelable"] is True for row in census.values())
    assert not any(key in row for row in census.values() for key in ("parent_task_id", "delegation_role"))
    attention = {row["task_id"]: row for row in payload["main_attention"]}
    assert set(attention) == {"held-sleep", "held-child"}
    assert attention["held-sleep"]["attention_reason"] == "restart"  # saved sleep is app_stop in the shared table
    # The child's own Resume is real (no root grant exists); Stop addresses roots only.
    assert attention["held-child"]["can_resume"] is True and attention["held-child"]["cancelable"] is False
    assert attention["held-child"]["root_task_id"] == "running-root"
    assert gateway.pending == [held, sleeping, child]


# --- #1716: an owner Resume of a member, consumed and then interrupted -------------------------
# A real Project child: its root is admitted by the routing promotion event and the child carries
# that root's frozen `_project_admission`, as `events_schedule_task` attaches an inherited basis.
# #1716's own producers park it under the owner's Pause, consume the owner's Resume and lose the
# process at an exact boundary (in-process `ProcessLost`; the real worker death stays outside).

ROOT, CHILD = "proj-root", "proj-child"


def _member_owner(root, monkeypatch):
    """The gateway fixture's owner state, written before any queue history."""
    from ouroboros import config
    from supervisor import active_activity, state

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(root))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(root / "settings.json"))
    monkeypatch.setattr(config, "REPO_DIR", root / "repo")
    state.init(root, total_budget_limit=10.0)
    state.save_state({"owner_id": 1, "current_branch": "fixture", "current_sha": "a" * 40,
                      "bg_consciousness_enabled": False, "evolution_mode_enabled": False})
    usage_store.migrate_from_journal(root)
    monkeypatch.setattr(active_activity, "_DIRECT_ACTIVITY_REGISTRY", active_activity.DirectActivityRegistry())


def _project_member(host, tmp_path, monkeypatch, *, cut="after_cas", resume=True):  # noqa: F811
    """A Project child the owner paused; with ``resume`` its owner Resume is consumed, then lost at ``cut``."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_COMPLETED
    from supervisor import queue, workers
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause
    from tests._budget_pause_exact_helpers import _fast_hold, _mock_pause_observation
    from tests.test_budget_pause_lineage import _member_ctx, _park, _sup
    from tests.test_project_hold_recovery import accepted, worker
    from tests.test_resume_handover import ProcessLost, _start_record

    _member_owner(host.root, monkeypatch)
    accepted(host, tmp_path, tid=ROOT)
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    _start_record(host.root, sent[0])
    parent = workers.RUNNING[ROOT]["task"]
    basis = dict(parent["_project_admission"])
    admitted = queue.enqueue_task({"id": CHILD, "type": "task", "root_task_id": ROOT, "parent_task_id": ROOT,
                                   "depth": 1, "delegation_role": "subagent", "text": "child work",
                                   "project_id": parent["project_id"], "chat_id": parent["chat_id"],
                                   "_project_admission": dict(basis)})
    assert not admitted.get("_admission_blocked"), admitted
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert [row["id"] for row in sent] == [CHILD]
    _start_record(host.root, sent[0])
    assert request_owner_pause(ROOT, request_id="p")["ok"]
    _fast_hold(monkeypatch, budget_pause)
    _mock_pause_observation(monkeypatch, budget_pause, [])
    ctx, limit = _member_ctx(host.root, CHILD, ROOT)
    limit.messages = [{"role": "user", "content": "do it"}, {"role": "assistant", "content": "saved"}]
    pause = _park(limit)
    install_exact_budget_pause(_sup(host.root, queue, workers), CHILD,
                               budget_pause.exact_pause_marker(pause)["checkpoint"])
    workers.RUNNING.pop(ROOT)
    write_task_result(host.root, ROOT, STATUS_COMPLETED, root_task_id=ROOT, project_id=parent["project_id"],
                      chat_id=parent["chat_id"])
    if not resume:
        return "", basis
    sent = worker(host, monkeypatch)
    assert queue.resume_budget_paused_task(CHILD)["ok"]
    workers.assign_tasks()
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    blob = budget_pause.load_budget_pause(ctx)
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    real_reopen = budget_pause._reopen_owner_fence

    def reopen_then_lose(*args, **kwargs):
        real_reopen(*args, **kwargs)
        raise ProcessLost("selected on its fence, nothing saved since")

    def lose(*_args, **_kwargs):
        raise ProcessLost("consumed, before its fence selection")

    with monkeypatch.context() as fault:
        fault.setattr(budget_pause, "_reopen_owner_fence", reopen_then_lose if cut == "after_fence" else lose)
        with pytest.raises(ProcessLost):
            budget_pause.resume_paused_loop(limit.tools, blob, [], {}, {}, set(), budget_remaining_usd=5.0)
    return blob["_pause_row"]["grant"]["grant_id"], basis


def _member_http(root):
    """The real /api/state and owner Resume routes over the queue the producers restored."""
    from ouroboros.gateway.tasks import api_task_resume

    app = Starlette(routes=[Route("/api/state", api_state),
                            Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])
    app.state.drive_root = root
    app.state.app_start = 0.0
    return TestClient(app)


def _member_state(client, *, complete=True):
    response = client.get("/api/state")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["main_attention_complete"] is complete, payload["main_attention_gaps"]
    return payload


def _attention_rows(payload, task_id):
    return [row for row in payload["main_attention"] if row["task_id"] == task_id]


def _main_consumer(payloads, task_id):
    """Main's real consumer reads these exact /api/state payloads in order, as ``chat.js`` hands them over:
    a task that needs the owner is one System message in Main's feed with the task's own controls."""
    script = """
        import fs from 'node:fs';
        import { installDom, restoreDom } from './web/tests/chat_dom_fixture.js';
        import { createMainWaiting } from './web/modules/main_waiting.js';
        import { stampNodeTimestamp } from './web/modules/chat_activity.js';
        const { payloads, taskId } = JSON.parse(fs.readFileSync(0, 'utf8'));
        const posted = [];
        const env = installDom(async (url, init) => {
            posted.push([url, init?.method || 'GET']);
            return { ok: true, json: async () => ({ ok: true }) };
        });
        globalThis.CSS = { escape: value => value };
        const find = (node, predicate) => predicate(node) ? node
            : node.children.reduce((found, child) => found || find(child, predicate), null);
        const page = document.createElement('div');
        const messagesDiv = document.createElement('div');
        const toolbar = document.createElement('div');
        page.append(messagesDiv, toolbar);
        env.mount.append(page);
        document.body = env.mount;
        const waiting = createMainWaiting({ page, messagesDiv, toolbar, stampNodeTimestamp,
            insertMessageNode: node => messagesDiv.append(node) });
        for (const payload of payloads) waiting.snapshot(payload);
        const card = find(messagesDiv, node => node.dataset.ownerAction === taskId);
        const sender = card && find(card, node => node.classList.contains('sender'));
        const resume = card && find(card, node => node.dataset.resumeRun === '1');
        const reason = card && find(card, node => node.classList.contains('owner-action-reason'));
        const stop = card && find(card, node => node.textContent === 'Stop\\u2026');
        const enabled = Boolean(resume && !resume.disabled);
        if (enabled) await resume.listeners.get('click')[0]({ stopPropagation() {} });
        console.log(JSON.stringify({ card: Boolean(card), sender: sender ? sender.textContent : '',
            resume: Boolean(resume), enabled, stop: Boolean(stop), reason: reason ? reason.textContent : '', posted }));
        waiting.destroy(); restoreDom(env.prior);
    """
    result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=Path(__file__).parents[1],
                            input=json.dumps({"payloads": payloads, "taskId": task_id}),
                            capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _settle_owner_fence(root, root_task_id):
    from ouroboros.owner_pause import FENCE_PAUSED, read_fence, set_fence_state

    fence = read_fence(root, root_task_id)
    set_fence_state(root, root_task_id, fence_id=fence["fence_id"], state=FENCE_PAUSED)


@pytest.mark.serial
@pytest.mark.parametrize("settled", [False, True])
@pytest.mark.parametrize("returning", [False, True])
@pytest.mark.parametrize("cut", ["after_cas", "after_fence"])
def test_an_interrupted_owner_resume_of_a_project_member_offers_its_existing_resume_in_main(
    host, tmp_path, monkeypatch, cut, returning, settled,  # noqa: F811
):
    from ouroboros.budget_pause import BUDGET_HOLD_KEY
    from ouroboros.owner_pause import read_fence
    from supervisor import workers
    from tests.test_project_hold_recovery import worker
    from tests.test_resume_handover import _stop_boot

    grant_id, basis = _project_member(host, tmp_path, monkeypatch, cut=cut)
    assert _stop_boot(monkeypatch, returning=returning) == 1
    if settled:  # the tree's own Pause settled while the member waited
        _settle_owner_fence(host.root, ROOT)
    client = _member_http(host.root)
    payload = _member_state(client)
    [row] = _attention_rows(payload, CHILD)
    # Not the owner's deliberate Pause: the parked member's existing Resume is offered, in the
    # Project its admission froze, still waiting for the restore's own Project verification.
    assert row["phase"] == "budget_paused" and row["can_resume"] is True and row["root_task_id"] == ROOT
    assert row["project_id"] == basis["project_id"] and row["cancelable"] is False, "Stop addresses roots only"
    assert row["project_admission_hold"]["label"] == "Waiting for Project verification"
    # The answered fence no longer explains it: the row's own facts do, never "owner".
    assert (row["pause_cause"], row["attention_reason"]) == (("unknown", "resume") if returning
                                                             else ("restart", "restart"))
    # Main's real consumer of this exact payload shows that Resume in its feed message and posts it
    # to the same route.
    assert _main_consumer([payload], CHILD) == {
        "card": True, "sender": "System", "resume": True, "enabled": True, "stop": False,
        "posted": [[f"/api/tasks/{CHILD}/resume", "POST"]],
        "reason": "Needs an explicit Resume." if returning else "Needs Resume after the restart."}
    before = read_fence(host.root, ROOT)
    response = client.post(f"/api/tasks/{CHILD}/resume")
    assert response.status_code == 200 and response.json()["ok"], response.text
    after = read_fence(host.root, ROOT)
    assert after["fence_id"] == before["fence_id"] and after["state"] == before["state"], "the tree stays fenced"
    assert after["selected_members"][CHILD]["grant_id"] == grant_id, "completed on its own consumed grant"
    [held] = workers.PENDING
    assert BUDGET_HOLD_KEY not in held and held["_resume_selection"]["selected_by"] == "owner"
    assert not _attention_rows(_member_state(client), CHILD), "a completed Resume leaves Main"
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [CHILD], "the same owner Resume reaches a worker"
    assert sent[0]["_project_admission"] == basis, "with the Project basis its admission froze"


@pytest.mark.serial
@pytest.mark.parametrize("change", ["newer_pause", "other_attempt", "revoked", "owner_pause"])
def test_main_offers_no_member_resume_beyond_its_own_interrupted_grant(host, tmp_path, monkeypatch, change):  # noqa: F811
    from ouroboros import budget_pause
    from ouroboros.owner_pause import read_fence
    from supervisor import workers
    from tests.test_project_hold_recovery import worker
    from tests.test_resume_handover import _stop_boot

    # owner_pause: the owner's own Pause, never resumed; its Resume stays the Project card's to offer.
    _project_member(host, tmp_path, monkeypatch, resume=change != "owner_pause")
    assert _stop_boot(monkeypatch) == 1
    _settle_owner_fence(host.root, ROOT)  # a settled Pause: the census reads Paused either way
    row = budget_pause.budget_pause_row(host.root, CHILD)
    if change == "newer_pause":  # the closed fence is not the one the consumed grant names
        budget_pause.set_budget_pause(host.root, CHILD, {**row, "grant": {**row["grant"],
                                                                          "owner_pause_fence_id": "older-fence"}})
    elif change == "other_attempt":  # the restored attempt does not resume the grant's saved work
        workers.PENDING[0]["_working_recovery"]["from_attempt"] = 2
    elif change == "revoked":
        budget_pause.set_budget_pause(host.root, CHILD, {**row, "grant": {**row["grant"], "revoked_at": "now"}})
    client = _member_http(host.root)
    payload = _member_state(client)
    assert not _attention_rows(payload, CHILD), "Main stays quiet"
    assert _main_consumer([payload], CHILD)["card"] is False, "its consumer shows nothing"
    if change == "owner_pause":
        return
    before = read_fence(host.root, ROOT)
    sent = worker(host, monkeypatch)
    response = client.post(f"/api/tasks/{CHILD}/resume")
    assert response.status_code != 200 and response.json()["error"] == "selection_owner_held", response.text
    assert read_fence(host.root, ROOT) == before, "the refusal changes no fence"
    workers.assign_tasks()
    assert not sent


@pytest.mark.serial
@pytest.mark.parametrize("settled", [False, True])
def test_an_unreadable_root_refuses_its_members_resume_at_the_action(
    host, tmp_path, monkeypatch, settled,  # noqa: F811
):
    """The root's own document stops decoding after a census read (bytes no UTF-8 decoder accepts,
    written outside the store). The census answers from typed hot facts and never decodes bodies
    (task cards, PERSISTENCE), so that corruption reaches only the action's strict authority read:
    Resume refuses, writes no fence and sends nothing; it never invents a resumable member."""
    import sqlite3

    from ouroboros.owner_pause import read_fence
    from supervisor import workers
    from tests.test_project_hold_recovery import worker
    from tests.test_resume_handover import _stop_boot

    _project_member(host, tmp_path, monkeypatch)
    assert _stop_boot(monkeypatch) == 1
    if settled:
        _settle_owner_fence(host.root, ROOT)
    client = _member_http(host.root)
    assert _attention_rows(_member_state(client), CHILD), "offered while its facts read"
    with sqlite3.connect(host.root / "state" / "current.sqlite") as conn:
        conn.execute("UPDATE task_bodies SET body = CAST(X'7B22FF' AS TEXT) WHERE task_id = ?", (ROOT,))
    with pytest.raises(Exception):
        load_task_result(host.root, ROOT, strict=True)
    before = read_fence(host.root, ROOT)
    sent = worker(host, monkeypatch)
    assert client.post(f"/api/tasks/{CHILD}/resume").status_code != 200, "the API refuses unreadable authority"
    assert read_fence(host.root, ROOT) == before
    workers.assign_tasks()
    assert not sent


# --- A direct Project turn: human-required parks are queue rows; its registry park is its own Pause ----------
# ``_direct_activity_pause_projection`` never offers Resume, and it need not: a direct turn that
# budget-pauses, holds saved work or loses an owner Resume ends its live actor and is parked in
# PENDING under the same id (#1196, restart retention), where the census applies the row's own
# release rule. The registry projection carries only a root parked WARM under the owner's own Pause.

def test_a_budget_parked_direct_project_turn_is_a_queue_row_whose_resume_main_offers(tmp_path, monkeypatch):
    from ouroboros import budget_pause
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.events import _handle_budget_pause
    from tests._budget_pause_exact_helpers import (
        _fast_hold, _install_queue, _loop_ctx, _quiet_external, _supervisor_ctx)

    _member_owner(tmp_path, monkeypatch)
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    project = create_project(tmp_path, "direct-room", name="Direct room")
    ctx, limit_ctx = _loop_ctx(tmp_path, "direct-park", direct=True)
    ctx.current_chat_id = project["chat_id"]
    _fast_hold(monkeypatch, budget_pause)
    _quiet_external(monkeypatch, budget_pause)
    with pytest.raises(budget_pause.BudgetPauseRequested) as raised:
        budget_pause.request_pause(limit_ctx, rail=budget_pause.RAIL_GLOBAL_EXHAUSTED, scope="global",
                                   reason_text="budget reached")
    budget_pause.end_dispatch_fence("direct-park")
    turn = {"id": "direct-park", "type": "task", "chat_id": project["chat_id"], "project_id": project["id"],
            "text": "Summarize the folder", "_is_direct_chat": True}
    _handle_budget_pause(budget_pause.pause_event(turn, raised.value.pause),
                         _supervisor_ctx(tmp_path, workers, queue, [], []))
    assert get_direct_activity_registry().get("direct-park") is None, "no live actor row beside the park"
    payload = _member_state(_member_http(tmp_path))
    [census] = [row for row in payload["active_chat_activities"] if row["activity_id"] == "direct-park"]
    assert (census["kind"], census["phase"], census["pause_cause"], census["can_resume"]) == (
        "direct_chat", "budget_paused", "budget", True)
    [row] = _attention_rows(payload, "direct-park")
    assert (row["attention_reason"], row["project_id"], row["can_resume"]) == ("budget", project["id"], True)
    shown = _main_consumer([payload], "direct-park")
    assert shown["card"] and shown["enabled"] and shown["posted"] == [["/api/tasks/direct-park/resume", "POST"]]
    assert shown["reason"] == "Paused at the spending limit. Change the limit if needed, then Resume."


def test_a_direct_project_roots_own_warm_pause_stays_quiet_in_main(tmp_path, monkeypatch):
    import threading
    import time

    from ouroboros import owner_pause, owner_wait
    from ouroboros.task_results import STATUS_RUNNING
    from supervisor import owner_pause_control
    from supervisor.active_activity import get_direct_activity_registry
    from supervisor.budget_resume import resume_warm_owner_pause_root
    from tests.test_pause_observation_cas import _direct_warm_author

    _member_owner(tmp_path, monkeypatch)
    queue, remote, ctx, limit = _direct_warm_author(tmp_path, monkeypatch)
    project = create_project(tmp_path, "warm-room", name="Warm room")
    write_task_result(tmp_path, "author", STATUS_RUNNING, chat_id=project["chat_id"], project_id=project["id"])
    registry = get_direct_activity_registry()  # the same live turn, working in its Project
    registry.register("author", chat_id=project["chat_id"], project_id=project["id"],
                      actor=registry.get("author").actor)
    real_sleep = time.sleep
    monkeypatch.setattr(owner_wait.time, "sleep", lambda seconds: real_sleep(min(seconds, 0.02)))
    woke = {}
    parked = threading.Thread(target=lambda: woke.update(cause=owner_wait.park_owner_pause_warm(
        limit, ctx, fence=owner_pause.read_fence(tmp_path, "author"), detached=[])), daemon=True)
    parked.start()
    try:
        deadline = time.monotonic() + 10
        while (load_task_result(tmp_path, "author").get("owner_wait") or {}).get("state") != "waiting":
            assert time.monotonic() < deadline and parked.is_alive(), woke
            real_sleep(0.02)
        remote["state"] = "cancelled"  # the run it sent obeyed the Pause's stop
        monkeypatch.setattr(owner_pause_control, "_LAST_SETTLE_CHECK", {})
        assert owner_pause_control.settle_requested_owner_pauses(queue, now=100.0) == ["author"]
        payload = _member_state(_member_http(tmp_path))
        [census] = [row for row in payload["active_chat_activities"] if row["activity_id"] == "author"]
        assert (census["kind"], census["phase"], census["pause_cause"], census.get("can_resume", False)) == (
            "direct_chat", "budget_paused", "owner", False)
        assert census["project_id"] == project["id"]
        assert not _attention_rows(payload, "author"), "the owner's own Pause stays quiet in Main"
        assert resume_warm_owner_pause_root("author")["ok"]
        parked.join(10)
        assert woke.get("cause") == "control:owner_resume"
    finally:
        owner_pause.release_fence(tmp_path, "author", reason="test_cleanup")
        parked.join(10)
