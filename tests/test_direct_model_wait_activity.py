"""Direct Project model-wait census uses the live owner through the JS reducer."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ouroboros import config, model_wait
from ouroboros.gateway import state as gateway_state
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.settings_integrity import TaskSettingsSnapshot
from ouroboros.task_results import write_task_result
from supervisor import queue as supervisor_queue
from supervisor.active_activity import get_direct_activity_registry

NODE_BIN = (
    str(Path.home() / ".claudexor" / "node" / "bin" / "node")
    if (Path.home() / ".claudexor" / "node" / "bin" / "node").exists()
    else "node"
)
WEB_ROOT = Path(__file__).resolve().parents[1] / "web"


def _js_project_summary(rows: list[dict]) -> dict:
    script = """
import { buildProjectActivityIndex } from './modules/project_activity.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const rows = JSON.parse(raw);
const summary = buildProjectActivityIndex(rows).byProject.get('project-1');
process.stdout.write(JSON.stringify(summary || null));
"""
    result = subprocess.run(
        [NODE_BIN, "--input-type=module", "-e", script],
        input=json.dumps(rows), text=True, encoding="utf-8", capture_output=True, cwd=WEB_ROOT,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _js_pause_cards(snapshots: list[list[dict]]) -> list[dict]:
    """Consume the real gateway rows through history, hydration and card rendering."""
    script = """
import { createChatInstance } from './modules/chat.js';
import { installDom, restoreDom, walkCard } from './tests/chat_dom_fixture.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const snapshots = JSON.parse(raw), task = snapshots[0][0].activity_id;
const env = installDom(async url => ({ ok: true, json: async () =>
    String(url).startsWith('/api/chat/history') ? { messages: [{task_id: task, chat_id: 7,
        role: 'assistant', is_progress: true, text: 'Checking retained work',
        ts: '2026-10-08T12:00:00Z', history_id: 'progress:1'}] } : {} }));
const instance = createChatInstance({
    ws: {on() {return () => {};}, isConnected: () => true, send() {}},
    state: {activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0},
    updateUnreadBadge() {}, chatId: 7, idPrefix: 'chat', mountEl: env.mount, asPanel: true,
    stateSnapshots: {begin: () => ({generation: 1, requestedAt: Date.now()}),
        gate() {return Promise.resolve(this.begin());}, isCurrent: () => true, apply() {}},
});
const find = (node, selector) => node.querySelector(selector)
    || node.children.map(child => find(child, selector)).find(Boolean) || null;
try {
    await instance.refreshHistory({revision: 1});
    const results = snapshots.map((rows, generation) => {
        instance.hydrateStateSnapshot({active_chat_activities: rows,
            active_chat_activities_complete: true, supervisor_ready: true}, Infinity, generation + 1);
        const card = walkCard(document.byId.get('chat-messages'), task);
        return {phase: card.querySelector('[data-live-phase]').textContent,
            resume: Boolean(find(card, '[data-resume-run]')),
            typing: card.querySelector('[data-live-typing]').style.display};
    });
    process.stdout.write(JSON.stringify(results));
} finally {instance.destroy(); restoreDom(env.prior);}
"""
    result = subprocess.run([NODE_BIN, "--input-type=module", "-e", script], input=json.dumps(snapshots),
                            text=True, encoding="utf-8", capture_output=True, cwd=WEB_ROOT, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.serial
@pytest.mark.parametrize("kind", ["direct_chat", "managed_task"])
def test_warm_owner_pause_registry_and_fence_reach_the_rendered_card(tmp_path, monkeypatch, kind):
    from ouroboros import owner_pause
    from ouroboros.owner_wait import set_owner_wait

    task = {"id": "paused-root", "root_task_id": "paused-root", "chat_id": 7}
    write_task_result(tmp_path, task["id"], "running", chat_id=7, root_task_id=task["id"])
    monkeypatch.setattr(supervisor_queue, "PENDING", [])
    monkeypatch.setattr(supervisor_queue, "RUNNING", {} if kind == "direct_chat" else {task["id"]: {"task": task}})
    monkeypatch.setattr(supervisor_queue, "BUDGET_ROOT_FENCES", {})
    registry = get_direct_activity_registry()
    registry.clear()
    samples = []
    try:
        with model_wait.task_model_wait_scope(task=task, drive_root=tmp_path,
                                              event_queue=queue.Queue(), worker_slot_held=False) as owner:
            if kind == "direct_chat":
                actor = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(model_wait_context=owner)))
                registry.register(task["id"], 7, actor=actor)
            raw = registry.snapshot()
            fence, _ = owner_pause.install_fence(tmp_path, task["id"], request_id="pause")
            supervisor_queue.BUDGET_ROOT_FENCES[task["id"]] = {"cause": "owner_pause", "fence_id": fence["fence_id"]}
            requested = gateway_state._chat_activities_snapshot_safe(tmp_path)
            assert requested[0]["phase"] == ("thinking" if kind == "direct_chat" else "budget_pausing")
            assert not requested[0].get("finishing_reviews"), "a requested Pause is not yet a paused author"
            set_owner_wait(tmp_path, task["id"], {"wait_id": "pause-wait", "state": "waiting", "reason": "owner_pause"})
            settling = gateway_state._chat_activities_snapshot_safe(tmp_path)[0]
            assert settling["phase"] == "budget_pausing" and not settling.get("finishing_reviews")
            # One reviewer as the census records it: its task (delegated run or live
            # operation) and its own model send. Mixed ids are never a reviewer count.
            owner_pause.set_fence_state(tmp_path, task["id"], fence_id=fence["fence_id"],
                                        state="paused", finishing_reviews=[task["id"], "attempt-review-send"])
            samples.append(gateway_state._chat_activities_snapshot_safe(tmp_path))
            row = samples[-1][0]
            assert row["kind"] == kind and row["phase"] == "budget_paused"
            assert row["pause_cause"] == "owner" and row["finishing_reviews"] is True
            owner_pause.set_fence_state(tmp_path, task["id"], fence_id=fence["fence_id"],
                                        state="paused", finishing_reviews=[])
            samples.append(gateway_state._chat_activities_snapshot_safe(tmp_path))
            owner_pause.release_fence(tmp_path, task["id"], reason="owner_resume")
            samples.append(gateway_state._chat_activities_snapshot_safe(tmp_path))
            assert registry.snapshot() == raw, "the canonical projection never mutates the raw registry"
    finally:
        registry.clear()
    assert _js_pause_cards(samples) == [
        {"phase": "Paused · owner pause · review work finishing", "resume": True, "typing": "none"},
        {"phase": "Paused · owner pause", "resume": True, "typing": "none"},
        # A resumed direct turn reads its own census phase (``thinking``), a managed task ``working``.
        {"phase": "Thinking" if kind == "direct_chat" else "Working", "resume": False, "typing": ""},
    ]


class _ModelCatalog:
    """A deterministic live catalog that resumes one real TaskModelWait owner."""

    available = False

    def claudexor_model_sources(self):
        return {"sources": [{"id": "codex", "credentialHarness": "fixture-harness"}]}

    def claudexor_model_catalog(self, source, _account=None, *, requested_model=None):
        assert source == "codex"
        assert requested_model == "exact-model"
        models = [{"id": "exact-model"}] if self.available else []
        return {"source": source, "credentialProfileId": "fixture", "models": models}


def test_direct_model_wait_owner_gap_keeps_static_row_and_discloses_partial():
    registry = get_direct_activity_registry()
    registry.clear()
    owner = SimpleNamespace(task_id="direct-project", closed=False, snapshot=lambda: (_ for _ in ()).throw(OSError("owner")))
    actor = SimpleNamespace(tools=SimpleNamespace(_ctx=SimpleNamespace(model_wait_context=owner)))
    registry.register("direct-project", 7, project_id="project-1", actor=actor)
    try:
        availability = {"complete": True}
        rows = gateway_state._direct_turns_snapshot_safe(availability=availability)
        assert rows[0]["activity_id"] == "direct-project"
        assert "model_waits" not in rows[0]
        assert availability["complete"] is False
    finally:
        registry.clear()


@pytest.mark.serial
@pytest.mark.parametrize(
    ("refusal_code", "wait_reason"),
    [("subscription_window_exhausted", "quota"), ("auth_required", "auth")],
)
def test_direct_project_model_wait_census_reaches_js_and_resumes(
    tmp_path, monkeypatch, refusal_code, wait_reason,
):
    """The real direct registry -> gateway census carries current wait evidence.

    The wait row is produced by ``TaskModelWait.wait`` from a typed quota/auth
    refusal; the test never seeds a model-wait row or reads history to rebuild
    one.  The same live owner then resumes on a compatible catalog and remains
    open long enough for the current-attempt projection to be observed again.
    """
    registry = get_direct_activity_registry()
    registry.clear()
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(supervisor_queue, "PENDING", [])
    monkeypatch.setattr(supervisor_queue, "RUNNING", {})
    monkeypatch.setattr(supervisor_queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(config, "CLAUDEXOR_MODEL_POLL_INTERVAL_SEC", 0.01)
    monkeypatch.setattr(config, "NETWORK_WAIT_BACKOFF_START_SEC", 0.01)
    monkeypatch.setattr(config, "NETWORK_WAIT_BACKOFF_MAX_SEC", 0.01)

    task = {
        "id": "direct-project",
        "chat_id": 7,
        "_attempt": 2,
        "_is_direct_chat": True,
        "metadata": {"project_id": "project-1"},
    }
    write_task_result(tmp_path, task["id"], "running", chat_id=task["chat_id"])
    actor = SimpleNamespace(
        _busy=True,
        _accepting_owner_messages=True,
        _current_task_id=task["id"],
        _current_chat_id=task["chat_id"],
        _current_task_metadata=task["metadata"],
        _current_task_text="wait for access",
        tools=SimpleNamespace(_ctx=SimpleNamespace(model_wait_context=None)),
    )
    registry.register(task["id"], task["chat_id"], project_id="project-1", actor=actor)

    catalog = _ModelCatalog()
    owner_ready = threading.Event()
    resumed = threading.Event()
    keep_owner_open = threading.Event()
    failures: list[BaseException] = []

    def run_wait():
        try:
            with config.task_settings_scope(TaskSettingsSnapshot(settings={}, environ={})):
                with model_wait.task_model_wait_scope(
                    task=task, drive_root=tmp_path, event_queue=queue.Queue(), worker_slot_held=False,
                ) as owner:
                    actor.tools._ctx.model_wait_context = owner
                    owner_ready.set()
                    owner.wait(
                        catalog,
                        ClaudexorUnavailable(refusal_code, f"fixture {wait_reason}"),
                        {"model": "claudexor::codex=exact-model", "model_role": "main"},
                    )
                    resumed.set()
                    keep_owner_open.wait(5)
        except BaseException as exc:  # surfaced after cleanup below
            failures.append(exc)

    thread = threading.Thread(target=run_wait, name="direct-project-model-wait")
    thread.start()
    try:
        assert owner_ready.wait(2)
        waiting_rows = None
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            availability = {"complete": True}
            direct = gateway_state._direct_turns_snapshot_safe(availability=availability)
            if direct and direct[0].get("model_waits"):
                waiting_rows = gateway_state._chat_activities_snapshot_safe(
                    tmp_path, direct_turns=direct, availability=availability,
                )
                break
            time.sleep(0.01)
        assert waiting_rows is not None
        assert availability["complete"] is True
        assert failures == []
        wait_activity = next(row for row in waiting_rows if row["activity_id"] == task["id"])
        assert wait_activity["project_id"] == "project-1"
        assert wait_activity["task_attempt"] == 2
        wait_row = list(wait_activity["model_waits"].values())[0]
        assert wait_row["state"] == "waiting"
        assert wait_row["reason"] == wait_reason
        assert wait_row["task_attempt"] == 2
        assert _js_project_summary(waiting_rows) == {
            "state": "waiting",
            "motion": False,
            "waiting": True,
            "label": "Waiting for access",
        }

        catalog.available = True
        assert resumed.wait(3), failures
        availability = {"complete": True}
        resumed_direct = gateway_state._direct_turns_snapshot_safe(availability=availability)
        resumed_rows = gateway_state._chat_activities_snapshot_safe(
            tmp_path, direct_turns=resumed_direct, availability=availability,
        )
        assert availability["complete"] is True
        resumed_activity = next(row for row in resumed_rows if row["activity_id"] == task["id"])
        assert resumed_activity["task_attempt"] == 2
        assert list(resumed_activity["model_waits"].values())[0]["state"] == "resolved"
        assert _js_project_summary(resumed_rows)["waiting"] is False
        assert _js_project_summary(resumed_rows)["motion"] is True
    finally:
        catalog.available = True
        keep_owner_open.set()
        thread.join(timeout=5)
        registry.unregister(task["id"])
        assert not thread.is_alive()
    assert failures == []
