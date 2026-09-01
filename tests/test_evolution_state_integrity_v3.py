from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest


def _active_transaction(tmp_path: pathlib.Path, task_id: str = "evo-task"):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    queue.init_queue_refs([], {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({
        "owner_chat_id": 1,
        "evolution_mode_enabled": True,
        "evolution_owner_stopped": False,
    })
    state.save_state(live)
    tx = evolution_lifecycle.begin_evolution_transaction(task_id, cycle=1, campaign=campaign)
    return campaign, tx


class _CaptureQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _assignment_case(tmp_path, monkeypatch, task_id="assign-evo"):
    from supervisor import evolution_lifecycle, queue, state, workers

    state.init(tmp_path)
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    pending, running = [], {}
    monkeypatch.setattr(workers, "PENDING", pending)
    monkeypatch.setattr(workers, "RUNNING", running)
    workers.init(tmp_path, tmp_path, 1, 600, 1800, 0.0)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    state.update_state(lambda live: live.update(
        evolution_mode_enabled=True,
        evolution_owner_stopped=False,
    ))
    tx = evolution_lifecycle.begin_evolution_transaction(task_id, cycle=1, campaign=campaign)
    task = {
        "id": task_id,
        "type": "evolution",
        "text": "Improve",
        "metadata": {"evolution_transaction": dict(tx)},
    }
    pending.append(task)
    inbox, events = _CaptureQueue(), _CaptureQueue()
    worker = SimpleNamespace(wid=1, busy_task_id=None, reaping=False, in_q=inbox)
    monkeypatch.setattr(workers, "WORKERS", {1: worker})
    monkeypatch.setattr(workers, "get_event_q", lambda: events)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    return workers, task, tx, worker, inbox, events


def test_pytest_process_globals_use_the_disposable_root():
    from supervisor import git_ops, queue, state, workers

    expected = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]).resolve(strict=False)
    assert state.DRIVE_ROOT.resolve(strict=False) == expected
    assert queue.DRIVE_ROOT.resolve(strict=False) == expected
    assert git_ops.DRIVE_ROOT.resolve(strict=False) == expected
    assert workers.DRIVE_ROOT.resolve(strict=False) == expected
    assert expected != (pathlib.Path.home() / "Ouroboros" / "data").resolve(strict=False)


def test_pytest_fuse_blocks_state_and_campaign_writes_to_live_data(monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    live = pathlib.Path(os.environ["OUROBOROS_TEST_LIVE_DATA_ROOT"])
    monkeypatch.setattr(state, "STATE_PATH", live / "state" / "state.json")
    monkeypatch.setattr(state, "STATE_LOCK_PATH", live / "locks" / "state.lock")
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        state.save_state({"evolution_mode_enabled": True})

    monkeypatch.setattr(queue, "DRIVE_ROOT", live)
    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        evolution_lifecycle._write_evolution_campaign({"id": "blocked", "status": "active"})


@pytest.mark.serial
def test_pytest_bootstrap_rebinds_preimported_modules_away_from_fake_home(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    fake_home = tmp_path / "home"
    sentinel_path = fake_home / "Ouroboros" / "data" / "state" / "state.json"
    sentinel_path.parent.mkdir(parents=True)
    sentinel = b'{"sentinel":"live"}\n'
    sentinel_path.write_bytes(sentinel)
    env = dict(os.environ)
    for key in (
        "OUROBOROS_DATA_DIR",
        "OUROBOROS_SETTINGS_PATH",
        "OUROBOROS_PYTEST_ACTIVE",
        "OUROBOROS_TEST_LIVE_DATA_ROOT",
    ):
        env.pop(key, None)
    env["HOME"] = str(fake_home)
    env["USERPROFILE"] = str(fake_home)
    env["PYTHONPATH"] = os.pathsep.join((str(repo), str(repo / "tests")))
    code = """
import importlib.util, json, pathlib
from supervisor import git_ops, queue, state, workers
spec = importlib.util.spec_from_file_location('isolated_conftest', pathlib.Path('tests/conftest.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._bind_pytest_runtime_roots()
state.update_state(lambda live: live.update({'probe': 'isolated'}))
print(json.dumps({
    'root': str(state.DRIVE_ROOT),
    'git_ops_root': str(git_ops.DRIVE_ROOT),
    'state': state.load_state(),
}))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert sentinel_path.read_bytes() == sentinel
    assert pathlib.Path(payload["root"]) != sentinel_path.parents[1]
    assert payload["git_ops_root"] == payload["root"]
    assert payload["state"]["probe"] == "isolated"


@pytest.mark.serial
def test_pytest_scrubbed_child_keeps_disposable_state_root(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    fake_home = tmp_path / "home"
    fake_live = fake_home / "Ouroboros" / "data" / "state"
    fake_live.mkdir(parents=True)
    sentinels = {
        fake_live / "state.json": b'{"sentinel":"state"}\n',
        fake_live / "evolution_campaign.json": b'{"sentinel":"campaign"}\n',
    }
    for path, content in sentinels.items():
        path.write_bytes(content)
    disposable_state = pathlib.Path(os.environ["OUROBOROS_DATA_DIR"]) / "state"
    disposable_paths = tuple(
        disposable_state / name
        for name in ("state.json", "state.last_good.json", "evolution_campaign.json")
    )
    disposable_before = {
        path: path.read_bytes() if path.exists() else None for path in disposable_paths
    }
    code = """
import json
from supervisor import evolution_lifecycle, state
state.update_state(lambda live: live.update({'scrubbed_child_probe': True}))
campaign = evolution_lifecycle.start_evolution_campaign('Probe', source='test')
print(json.dumps({
    'campaign_id': campaign.get('id', ''),
    'drive_root': str(state.DRIVE_ROOT),
    'pytest_loaded': 'pytest' in __import__('sys').modules,
}))
"""

    child_env = {"HOME": str(fake_home), "USERPROFILE": str(fake_home)}
    if os.name == "nt" and os.environ.get("SystemRoot"):
        child_env["SystemRoot"] = os.environ["SystemRoot"]
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo,
            env=child_env,
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        assert payload["pytest_loaded"] is False
        assert pathlib.Path(payload["drive_root"]).resolve(strict=False) == pathlib.Path(
            os.environ["OUROBOROS_DATA_DIR"]
        ).resolve(strict=False)
        assert payload["campaign_id"]
        for path, content in sentinels.items():
            assert path.read_bytes() == content
    finally:
        for path, content in disposable_before.items():
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)


@pytest.mark.serial
def test_nested_pytest_keeps_the_original_live_root_marker(tmp_path):
    repo = pathlib.Path(__file__).resolve().parents[1]
    inherited_disposable = tmp_path / "parent-pytest-data"
    env = {
        "OUROBOROS_DATA_DIR": str(inherited_disposable),
        "OUROBOROS_SETTINGS_PATH": str(inherited_disposable / "settings.json"),
    }
    # A Windows child python cannot even boot without SystemRoot, and the nested
    # conftest's fresh mkdtemp needs a real TEMP (the ntpath fallback chain would
    # otherwise land in the repo cwd). POSIX children boot fine with a bare env —
    # same passthrough precedent as the scrubbed-child test above. The conftest
    # Popen patch injects the OUROBOROS_* markers this test is actually about.
    if os.name == "nt":
        for key in ("SystemRoot", "TEMP", "TMP"):
            if os.environ.get(key):
                env[key] = os.environ[key]
    code = """
import importlib.util, json, os, pathlib
spec = importlib.util.spec_from_file_location('nested_conftest', pathlib.Path('tests/conftest.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    'live_root': os.environ['OUROBOROS_TEST_LIVE_DATA_ROOT'],
    'data_root': os.environ['OUROBOROS_DATA_DIR'],
}))
"""

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert pathlib.Path(payload["live_root"]).resolve(strict=False) == pathlib.Path(
        os.environ["OUROBOROS_TEST_LIVE_DATA_ROOT"]
    ).resolve(strict=False)
    assert pathlib.Path(payload["data_root"]).resolve(strict=False) != inherited_disposable.resolve(
        strict=False
    )


def test_scheduler_disables_a_bare_flag_without_campaign(tmp_path, monkeypatch):
    from supervisor import queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    live = state.load_state()
    live.update({
        "owner_chat_id": 1,
        "evolution_mode_enabled": True,
        "post_task_autostop": True,
    })
    state.save_state(live)
    sent = []
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: sent.append(args[1]))

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    assert state.load_state()["evolution_mode_enabled"] is False
    assert state.load_state()["post_task_autostop"] is False
    assert "active campaign authority" in sent[0]
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "evolution_authority_missing"


def test_scheduler_refuses_active_campaign_without_source(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    queue.init_queue_refs([], {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    campaign.pop("source")
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True
    live = state.load_state()
    live.update({"owner_chat_id": 1, "evolution_mode_enabled": True})
    state.save_state(live)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)

    queue.enqueue_evolution_task_if_needed()

    assert state.load_state()["evolution_mode_enabled"] is False


def test_owner_resume_repairs_missing_legacy_campaign_source(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    campaign["status"] = "paused"
    campaign.pop("source")
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True

    resumed = evolution_lifecycle.start_evolution_campaign("", source="owner_chat")

    assert resumed["status"] == "active"
    assert resumed["source"] == "owner_chat"


def test_scheduler_does_not_enqueue_when_transaction_attach_fails(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({"owner_chat_id": 1, "evolution_mode_enabled": True})
    state.save_state(live)
    monkeypatch.setattr(queue, "begin_evolution_transaction", lambda *a, **k: {})
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    assert state.load_state()["evolution_mode_enabled"] is False


def test_transaction_attach_rechecks_owner_stop_under_state_lock(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({"evolution_mode_enabled": False, "evolution_owner_stopped": True})
    state.save_state(live)

    tx = evolution_lifecycle.begin_evolution_transaction(
        "too-late", cycle=1, campaign=campaign,
    )

    assert tx == {}
    assert "active_transaction" not in evolution_lifecycle._read_evolution_campaign()


def test_scheduler_replaces_uncommitted_transaction_lost_before_enqueue(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    live = state.load_state()
    live.update({"owner_chat_id": 1, "evolution_mode_enabled": True})
    state.save_state(live)
    lost = evolution_lifecycle.begin_evolution_transaction(
        "lost-before-enqueue", cycle=1, campaign=campaign,
    )
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)

    queue.enqueue_evolution_task_if_needed()

    assert len(pending) == 1
    replacement = pending[0]["metadata"]["evolution_transaction"]
    assert replacement["transaction_id"] != lost["transaction_id"]
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["active_transaction"]["transaction_id"] == replacement["transaction_id"]
    assert stored["transaction_history"][-1]["abandoned_reason"] == "dispatch_not_persisted"


def test_scheduler_does_not_replace_transaction_while_worker_is_reaping(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    _campaign, tx = _active_transaction(tmp_path, task_id="reaping-evolution")
    pending = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    assert evolution_lifecycle.update_evolution_transaction(
        tx["task_id"], dispatch_status="reaping",
    )
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: None)

    queue.enqueue_evolution_task_if_needed()

    assert pending == []
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["transaction_id"] == tx["transaction_id"]
    assert stored["dispatch_status"] == "reaping"


def test_timeout_marks_evolution_reaping_before_scheduler_can_replace_it(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    _campaign, tx = _active_transaction(tmp_path, task_id="timeout-evolution")
    pending = []
    running = {
        tx["task_id"]: {
            "task": {
                "id": tx["task_id"],
                "type": "evolution",
                "chat_id": 1,
                "metadata": {"evolution_transaction": dict(tx)},
            },
            "started_at": 1.0,
            "last_heartbeat_at": 1.0,
            "worker_id": 7,
            "attempt": 1,
        }
    }
    queue.init_queue_refs(pending, running, {"value": 0})
    worker = SimpleNamespace(busy_task_id=tx["task_id"], proc=None, reaping=False)
    workers_view = SimpleNamespace(WORKERS={7: worker})
    reaper_jobs = _CaptureQueue()
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 0)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_per_call_timeout_ceiling_sec", lambda: 1)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: 10)
    monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
    monkeypatch.setattr(queue, "_reap_queue", reaper_jobs)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": True)
    monkeypatch.setattr(queue, "send_with_budget", lambda *args, **kwargs: None)

    queue._enforce_task_timeouts_locked(
        workers_view, now=1000.0, owner_chat_id=1,
        st={"evolution_mode_enabled": True},
    )

    assert running == {}
    assert worker.reaping is True
    assert len(reaper_jobs.items) == 1
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["dispatch_status"] == "reaping"

    queue.enqueue_evolution_task_if_needed()
    assert pending == []
    assert evolution_lifecycle._read_evolution_campaign()["active_transaction"][
        "transaction_id"
    ] == tx["transaction_id"]


def test_terminal_event_cannot_write_into_a_different_campaign(tmp_path):
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path, task_id="same-task")
    stale = {
        **tx,
        "campaign_id": "old-campaign",
        "transaction_id": "old-transaction",
    }

    result = evolution_lifecycle.update_evolution_campaign_after_task(
        "same-task",
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=stale,
    )

    assert result == {
        "accepted": False,
        "persisted": False,
        "replay": False,
        "reason": "transaction_mismatch",
        "transaction": {},
    }
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["id"] == campaign["id"]
    assert stored["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert stored.get("history", []) == []


def test_metadata_less_terminal_cannot_mutate_active_campaign(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")

    result = evolution_lifecycle.update_evolution_campaign_after_task(
        "stale-task",
        cost_usd=1.25,
        outcome_axes={"execution": {"status": "failed"}},
        rounds=1,
    )

    assert result == {
        "accepted": False,
        "persisted": False,
        "replay": False,
        "reason": "transaction_missing",
        "transaction": {},
    }
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["id"] == campaign["id"]
    assert stored["cycles_done"] == 0
    assert stored["budget_spent_usd"] == 0.0
    assert stored.get("history", []) == []


def test_duplicate_terminal_resumes_pending_cleanup_and_owner_report(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    assert evolution_lifecycle.update_evolution_transaction(
        tx["task_id"], rescue_ref="refs/ouroboros/rescue/test",
    )
    real_resume = evolution_lifecycle._resume_evolution_terminal_effects
    monkeypatch.setattr(
        evolution_lifecycle,
        "_resume_evolution_terminal_effects",
        lambda _campaign_id, _task_id, value: dict(value),
    )

    first = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "failed"}},
        rounds=1,
        transaction=tx,
    )

    assert first["persisted"] is True
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["history"][0]["transaction"]["cleanup_status"] == "pending"
    assert stored["pending_owner_report"]["cycle_outcome"] == "abandoned"

    cleanup_calls = []
    reports = []

    def _cleanup(value, *_args, **_kwargs):
        cleanup_calls.append(value["transaction_id"])
        value["cleanup_status"] = "already_clean"

    monkeypatch.setattr(evolution_lifecycle, "_resume_evolution_terminal_effects", real_resume)
    monkeypatch.setattr(evolution_lifecycle, "_cleanup_worktree_after_cycle", _cleanup)
    monkeypatch.setattr(
        evolution_lifecycle,
        "notify_owner_cycle_outcome",
        lambda campaign, value: reports.append((campaign["id"], value["cycle_outcome"])),
    )

    replay = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "failed"}},
        rounds=1,
        transaction=tx,
    )

    assert replay["replay"] is True
    assert cleanup_calls == [tx["transaction_id"]]
    assert reports == [(_campaign["id"], "abandoned")]
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["history"][0]["transaction"]["cleanup_status"] == "already_clean"
    assert "pending_owner_report" not in stored


def test_duplicate_terminal_resumes_missing_restart_request(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    receipt = evolution_lifecycle.record_evolution_commit(
        campaign_id=tx["campaign_id"],
        transaction_id=tx["transaction_id"],
        task_id=tx["task_id"],
        commit_sha="a" * 40,
    )
    assert receipt["ok"] is True
    real_resume = evolution_lifecycle._resume_evolution_terminal_effects
    monkeypatch.setattr(
        evolution_lifecycle,
        "_resume_evolution_terminal_effects",
        lambda _campaign_id, _task_id, value: dict(value),
    )

    first = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=tx,
    )

    assert first["transaction"]["cycle_outcome"] == "waiting_for_restart"
    restart_calls = []
    monkeypatch.setattr(evolution_lifecycle, "_resume_evolution_terminal_effects", real_resume)
    monkeypatch.setattr(
        evolution_lifecycle,
        "request_evolution_restart",
        lambda drive_root, value, log=None: restart_calls.append(
            (pathlib.Path(drive_root), value["commit_sha"])
        ),
    )

    replay = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=tx,
    )

    assert replay["replay"] is True
    assert restart_calls == [(tmp_path, "a" * 40)]


def test_terminal_restart_preserves_exact_model_reason(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, workers

    campaign, tx = _active_transaction(tmp_path)
    sha = "c" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    current_tx = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({
        "expected_sha": sha,
        "reason": "apply reviewed evolution",
        "evolution_claim": claim,
    }))
    events = _CaptureQueue()
    monkeypatch.setenv("OUROBOROS_EVOLUTION_AUTO_RESTART", "true")
    monkeypatch.setattr(workers, "get_event_q", lambda: events)

    evolution_lifecycle.request_evolution_restart(tmp_path, current_tx)

    assert json.loads(marker.read_text())["reason"] == "apply reviewed evolution"
    assert len(events.items) == 1
    assert events.items[0]["reason"] == "apply reviewed evolution"
    assert events.items[0]["evolution_restart"] is True


def test_terminal_write_serializes_concurrent_campaign_pause(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    real_write = evolution_lifecycle._write_evolution_campaign
    entered = threading.Event()
    release = threading.Event()
    terminal_result = {}
    pause_result = {}

    def _hold_terminal_write(data, **kwargs):
        entered.set()
        assert release.wait(timeout=2)
        return real_write(data, **kwargs)

    monkeypatch.setattr(evolution_lifecycle, "_write_evolution_campaign", _hold_terminal_write)
    monkeypatch.setattr(
        evolution_lifecycle,
        "_cleanup_worktree_after_cycle",
        lambda tx, *_a, **_k: tx.update(cleanup_status="already_clean"),
    )

    def _terminal():
        terminal_result.update(evolution_lifecycle.update_evolution_campaign_after_task(
            tx["task_id"],
            cost_usd=1.0,
            outcome_axes={"execution": {"status": "ok"}},
            rounds=1,
            transaction=tx,
        ))

    terminal_thread = threading.Thread(target=_terminal)
    terminal_thread.start()
    assert entered.wait(timeout=2)

    def _pause():
        pause_result.update(evolution_lifecycle.pause_evolution_campaign("concurrent pause"))

    pause_thread = threading.Thread(target=_pause)
    pause_thread.start()
    pause_thread.join(timeout=0.05)
    assert pause_thread.is_alive()
    release.set()
    terminal_thread.join(timeout=2)
    pause_thread.join(timeout=2)

    assert terminal_result["persisted"] is True
    assert pause_result["status"] == "paused"
    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["status"] == "paused"
    assert stored["pause_reason"] == "concurrent pause"
    assert stored["history"][0]["task_id"] == tx["task_id"]


def test_terminal_write_exception_has_no_lifecycle_side_effects(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle

    _campaign, tx = _active_transaction(tmp_path)
    side_effects = []
    monkeypatch.setattr(
        evolution_lifecycle,
        "_write_evolution_campaign",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "_cleanup_worktree_after_cycle",
        lambda *_a, **_k: side_effects.append("cleanup"),
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "notify_owner_cycle_outcome",
        lambda *_a, **_k: side_effects.append("notify"),
    )

    result = evolution_lifecycle.update_evolution_campaign_after_task(
        tx["task_id"],
        cost_usd=1.0,
        outcome_axes={"execution": {"status": "ok"}},
        rounds=1,
        transaction=tx,
    )

    assert result["persisted"] is False
    assert result["reason"] == "campaign_write_failed"
    assert side_effects == []


def test_rejected_terminal_does_not_consume_global_evolution_state(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, state
    from supervisor.events import _handle_evolution_task_done

    state.init(tmp_path)
    state.update_state(lambda live: live.update(
        evolution_mode_enabled=True,
        post_task_autostop=True,
        evolution_consecutive_failures=4,
    ))
    monkeypatch.setattr(
        evolution_lifecycle,
        "update_evolution_campaign_after_task",
        lambda *_a, **_k: {
            "accepted": True, "persisted": False, "replay": False,
            "reason": "campaign_write_refused", "transaction": {},
        },
    )
    checkpoints = []
    monkeypatch.setattr(
        "ouroboros.evolution_checkpoints.append_evolution_checkpoint",
        lambda *_a, **_k: checkpoints.append(True),
    )
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, REPO_DIR=tmp_path)
    task = {"metadata": {"evolution_transaction": {"transaction_id": "stale"}}}

    _handle_evolution_task_done(
        ctx,
        evt={},
        task_id="stale",
        task=task,
        task_done_event={"status": "failed"},
        outcome_axes={"execution": {"status": "failed"}},
        cost=1.0,
        rounds=1,
    )

    live = state.load_state()
    assert live["evolution_mode_enabled"] is True
    assert live["post_task_autostop"] is True
    assert live["evolution_consecutive_failures"] == 4
    assert checkpoints == []


def test_assignment_dispatches_exact_uncommitted_evolution_claim(tmp_path, monkeypatch):
    workers, task, _tx, worker, inbox, events = _assignment_case(tmp_path, monkeypatch)

    workers.assign_tasks()

    assert inbox.items == [task]
    assert worker.busy_task_id == task["id"]
    assert workers.RUNNING[task["id"]]["task"] == task
    assert events.items == []


def test_assignment_rejects_stale_or_committed_evolution_claim(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from supervisor import evolution_lifecycle

    workers, task, tx, worker, inbox, events = _assignment_case(tmp_path, monkeypatch)
    task["metadata"]["evolution_transaction"]["task_id"] = "other-task"

    workers.assign_tasks()

    assert inbox.items == []
    assert workers.RUNNING == {}
    assert worker.busy_task_id is None
    stored = load_task_result(tmp_path, task["id"])
    assert stored["status"] == "cancelled"
    assert stored["reason_code"] == "evolution_authority_missing"
    assert stored["authority_reason"] == "task_mismatch"
    assert events.items[-1]["metadata"]["evolution_transaction"]["task_id"] == "other-task"

    workers, task, tx, _worker, inbox, _events = _assignment_case(
        tmp_path / "committed", monkeypatch, task_id="committed-evo",
    )
    campaign = evolution_lifecycle._read_evolution_campaign()
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], "a" * 40,
    )["ok"] is True

    workers.assign_tasks()

    assert inbox.items == []
    assert load_task_result(tmp_path / "committed", task["id"])["authority_reason"] == (
        "transaction_already_committed"
    )


def test_assignment_keeps_invalid_evolution_pending_when_cancel_write_fails(
    tmp_path, monkeypatch,
):
    from supervisor import workers as workers_module

    workers, task, _tx, worker, inbox, events = _assignment_case(tmp_path, monkeypatch)
    task["metadata"]["evolution_transaction"]["task_id"] = "other-task"
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )

    workers.assign_tasks()

    assert workers_module.PENDING == [task]
    assert worker.busy_task_id is None
    assert inbox.items == []
    assert events.items == []


def test_evolution_orphan_ref_cannot_be_published_by_later_normal_push(
    tmp_path, monkeypatch,
):
    from ouroboros.tools import git as git_tools
    from supervisor import git_ops

    repo, remote = tmp_path / "repo", tmp_path / "remote.git"

    def _git(*args, cwd=repo, check=True):
        return subprocess.run(
            ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        )

    subprocess.run(
        ["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True,
    )
    repo.mkdir()
    _git("init", "-b", "ouroboros")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    _git("remote", "add", "origin", str(remote))
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    (repo / "peer.txt").write_text("peer-base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    base_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("tag", "-a", "v-base", "-m", "base")
    _git("push", "-u", "origin", "ouroboros")
    _git("push", "origin", "--tags")
    (repo / "file.txt").write_text("orphan\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "orphan")
    orphan_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("tag", "-a", "v-orphan", "-m", "orphan")
    (repo / "peer.txt").write_text("peer-concurrent-edit\n", encoding="utf-8")

    note = git_tools._preserve_evolution_orphan(
        SimpleNamespace(repo_dir=repo), orphan_sha, created_tag="v-orphan",
    )

    assert "CONTAINMENT_FAILED" not in note
    assert _git("rev-parse", "HEAD").stdout.strip() == base_sha
    private_ref = f"refs/ouroboros/evolution-orphans/{orphan_sha}"
    assert _git("rev-parse", private_ref).stdout.strip() == orphan_sha
    assert _git("show-ref", "--verify", "refs/tags/v-orphan", check=False).returncode != 0
    assert _git("rev-parse", "refs/tags/v-base^{commit}").stdout.strip() == base_sha
    assert (repo / "peer.txt").read_text(encoding="utf-8") == "peer-concurrent-edit\n"
    assert _git("status", "--porcelain").stdout.strip()

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    pushed, _message = git_ops.push_to_remote("ouroboros", push_tags=True)

    assert pushed is True
    assert _git("rev-parse", "refs/heads/ouroboros", cwd=remote).stdout.strip() == base_sha
    assert _git("show-ref", "--verify", "refs/tags/v-orphan", cwd=remote, check=False).returncode != 0
    assert _git("show-ref", "--verify", private_ref, cwd=remote, check=False).returncode != 0
    assert _git("cat-file", "-e", orphan_sha, cwd=remote, check=False).returncode != 0

    # A separate Git writer may advance the branch after the atomic containment
    # transaction. Worktree alignment must not move that ref back to the parent.
    (repo / "file.txt").write_text("second orphan\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "second orphan")
    second_orphan = _git("rev-parse", "HEAD").stdout.strip()
    base_tree = _git("rev-parse", f"{base_sha}^{{tree}}").stdout.strip()
    concurrent = subprocess.run(
        ["git", "commit-tree", base_tree, "-p", base_sha],
        cwd=repo,
        input="concurrent branch update\n",
        text=True,
        check=True,
        capture_output=True,
    ).stdout.strip()
    real_subprocess_run = subprocess.run
    interleaved = {"done": False}

    def _interleave_after_ref_transaction(cmd, *args, **kwargs):
        proc = real_subprocess_run(cmd, *args, **kwargs)
        if cmd[:3] == ["git", "update-ref", "--stdin"] and proc.returncode == 0 and not interleaved["done"]:
            real_subprocess_run(
                ["git", "update-ref", "refs/heads/ouroboros", concurrent, base_sha],
                cwd=repo, check=True, capture_output=True, text=True,
            )
            interleaved["done"] = True
        return proc

    monkeypatch.setattr(git_tools.subprocess, "run", _interleave_after_ref_transaction)
    note = git_tools._preserve_evolution_orphan(
        SimpleNamespace(repo_dir=repo), second_orphan,
    )

    assert "CONTAINMENT_FAILED" not in note
    assert "concurrent branch update" in note
    assert _git("rev-parse", "HEAD").stdout.strip() == concurrent
    assert _git(
        "rev-parse", f"refs/ouroboros/evolution-orphans/{second_orphan}",
    ).stdout.strip() == second_orphan


def test_orphan_ref_transaction_failure_falls_back_to_safe_ref_cas(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import git_ops

    repo, remote = tmp_path / "repo", tmp_path / "remote.git"
    real_run = subprocess.run

    def _git(*args, cwd=repo, check=True):
        return real_run(
            ["git", *args], cwd=cwd, check=check, capture_output=True, text=True,
        )

    real_run(["git", "init", "--bare", str(remote)], check=True, capture_output=True, text=True)
    repo.mkdir()
    _git("init", "-b", "ouroboros")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    _git("remote", "add", "origin", str(remote))
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    base_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("push", "-u", "origin", "ouroboros")
    (repo / "file.txt").write_text("orphan\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "orphan")
    orphan_sha = _git("rev-parse", "HEAD").stdout.strip()
    _git("tag", "-a", "v-orphan", "-m", "orphan")

    def _fail_transactions(cmd, *args, **kwargs):
        if cmd[:3] == ["git", "update-ref", "--stdin"]:
            # BYTES streams: the transaction call deliberately runs in binary mode
            # (text-mode pipes CRLF-mangle --stdin commands on Windows).
            return subprocess.CompletedProcess(cmd, 1, b"", b"injected transaction failure")
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(git_tools.subprocess, "run", _fail_transactions)

    note = git_tools._preserve_evolution_orphan(
        SimpleNamespace(repo_dir=repo), orphan_sha, created_tag="v-orphan",
    )

    assert "CONTAINMENT_FAILED" not in note
    assert _git("rev-parse", "HEAD").stdout.strip() == base_sha
    assert _git(
        "rev-parse", f"refs/ouroboros/evolution-orphans/{orphan_sha}",
    ).stdout.strip() == orphan_sha
    assert _git("show-ref", "--verify", "refs/tags/v-orphan", check=False).returncode != 0

    monkeypatch.setattr(git_ops, "REPO_DIR", repo)
    pushed, _message = git_ops.push_to_remote("ouroboros", push_tags=True)

    assert pushed is True
    assert _git("rev-parse", "refs/heads/ouroboros", cwd=remote).stdout.strip() == base_sha
    assert _git("cat-file", "-e", orphan_sha, cwd=remote, check=False).returncode != 0


def test_exact_commit_receipt_is_bound_to_campaign_transaction_and_task(tmp_path):
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
    }
    assert evolution_lifecycle.check_evolution_authority(**claim)["ok"] is True

    receipt = evolution_lifecycle.record_evolution_commit(**claim, commit_sha="a" * 40)

    assert receipt["ok"] is True
    assert receipt["commit_sha"] == "a" * 40
    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["commit_receipt"] == receipt
    assert evolution_lifecycle.check_evolution_authority(
        **claim, commit_sha="b" * 40,
    )["reason"] == "commit_receipt_mismatch"

    campaign_state = evolution_lifecycle._read_evolution_campaign()
    campaign_state["active_transaction"].pop("commit_receipt")
    assert evolution_lifecycle._write_evolution_campaign(campaign_state) is True
    assert evolution_lifecycle.check_evolution_authority(
        **claim, commit_sha="a" * 40,
    )["reason"] == "commit_receipt_missing"


def test_second_evolution_commit_is_blocked_before_review(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], "a" * 40,
    )["ok"] is True
    review_calls = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(
        git_tools,
        "_run_reviewed_stage_cycle",
        lambda *a, **k: review_calls.append(True) or {"status": "passed"},
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        current_task_type="evolution",
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
    )

    result = git_tools._repo_commit_push(ctx, "second commit")

    assert "transaction_already_committed" in result
    assert "No reviewer was called" in result
    assert review_calls == []


def test_receipt_race_blocks_evolution_before_git_commit(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        current_task_type="evolution",
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
    )
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    claim, error = git_tools._check_evolution_commit_stage(
        ctx, "commit", 0.0, phase="pre_review_authority",
    )
    assert error == ""
    assert evolution_lifecycle.record_evolution_commit(
        **claim, commit_sha="b" * 40,
    )["ok"] is True

    _claim, error = git_tools._check_evolution_commit_stage(
        ctx, "commit", 0.0, phase="pre_commit_authority",
    )

    assert "transaction_already_committed" in error
    assert "Nothing was committed" in error


def test_revoked_authority_leaves_commit_unrecorded(tmp_path):
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    live = state.load_state()
    live["evolution_mode_enabled"] = False
    live["evolution_owner_stopped"] = True
    state.save_state(live)

    receipt = evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], "c" * 40,
    )

    assert receipt == {"ok": False, "reason": "owner_stopped", "commit_sha": "c" * 40}
    assert evolution_lifecycle._read_evolution_campaign()["active_transaction"]["commit_sha"] == ""


def test_exact_receipt_remains_authority_after_post_task_autostop(tmp_path):
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "9" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
    }
    assert evolution_lifecycle.record_evolution_commit(**claim, commit_sha=sha)["ok"] is True
    state.update_state(lambda live: live.update(
        evolution_mode_enabled=False,
        post_task_autostop=False,
    ))

    assert evolution_lifecycle.check_evolution_authority(
        **claim, commit_sha=sha,
    )["ok"] is True
    assert evolution_lifecycle.check_evolution_authority(**claim)["reason"] == "evolution_disabled"


@pytest.mark.parametrize("held_lock", ["state", "campaign"])
def test_rescue_link_uses_shared_campaign_cas_and_preserves_commit_receipt(
    tmp_path, monkeypatch, held_lock,
):
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from ouroboros.utils import atomic_write_json
    from supervisor import evolution_lifecycle, git_ops, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "3" * 40
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    monkeypatch.setattr(evolution_lifecycle, "EVOLUTION_CAMPAIGN_CAS_TIMEOUT_SEC", 1.0)
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", tmp_path)
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    if held_lock == "state":
        lock_path = tmp_path / "locks" / "state.lock"
        lock_fd = state.acquire_file_lock(lock_path, timeout_sec=1.0)
        release = state.release_file_lock
    else:
        lock_path = campaign_path.with_name(campaign_path.name + ".lock")
        lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
        release = release_exclusive_file_lock
    assert lock_fd is not None
    done = threading.Event()

    def _link() -> None:
        git_ops._link_rescue_to_evolution_transaction(
            {"rescue_ref": "rescue/test", "path": "/tmp/rescue-test"},
            "test",
        )
        done.set()

    thread = threading.Thread(target=_link, daemon=True)
    thread.start()
    try:
        assert done.wait(0.1) is False
        current = evolution_lifecycle._read_evolution_campaign()
        current["active_transaction"]["interleaved"] = held_lock
        atomic_write_json(campaign_path, current, trailing_newline=True)
    finally:
        release(lock_path, lock_fd)
    assert done.wait(2.0) is True
    thread.join(timeout=1.0)

    stored = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert stored["commit_sha"] == sha
    assert stored["commit_receipt"]["commit_sha"] == sha
    assert stored["rescue_ref"] == "rescue/test"
    assert stored["interleaved"] == held_lock


def test_commit_receipt_uses_campaign_sidecar_before_rescue(tmp_path, monkeypatch):
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    monkeypatch.setattr(evolution_lifecycle, "EVOLUTION_CAMPAIGN_CAS_TIMEOUT_SEC", 1.0)
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    lock_path = campaign_path.with_name(campaign_path.name + ".lock")
    lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=1.0)
    assert lock_fd is not None
    done = threading.Event()
    result = {}

    def _record() -> None:
        result.update(evolution_lifecycle.record_evolution_commit(
            campaign["id"], tx["transaction_id"], tx["task_id"], "4" * 40,
        ))
        done.set()

    thread = threading.Thread(target=_record, daemon=True)
    thread.start()
    try:
        assert done.wait(0.1) is False
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
    assert done.wait(2.0) is True
    thread.join(timeout=1.0)
    assert result["ok"] is True
    assert evolution_lifecycle._read_evolution_campaign()["active_transaction"][
        "commit_receipt"
    ]["commit_sha"] == "4" * 40


def test_campaign_sidecar_contention_releases_state_lock_quickly(tmp_path):
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    sidecar = campaign_path.with_name(campaign_path.name + ".lock")
    sidecar_fd = acquire_exclusive_file_lock(sidecar, timeout_sec=1.0)
    assert sidecar_fd is not None
    try:
        result = evolution_lifecycle.record_evolution_commit(
            campaign["id"], tx["transaction_id"], tx["task_id"], "6" * 40,
        )
        assert result["ok"] is False
        state_fd = state.acquire_file_lock(state.STATE_LOCK_PATH, timeout_sec=0.2)
        assert state_fd is not None
        state.release_file_lock(state.STATE_LOCK_PATH, state_fd)
    finally:
        release_exclusive_file_lock(sidecar, sidecar_fd)


def test_sent_owner_report_clear_cannot_erase_concurrent_commit_receipt(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue

    campaign, tx = _active_transaction(tmp_path)
    report = {"cycle_outcome": "absorbed", "task_id": "previous"}
    current = evolution_lifecycle._read_evolution_campaign()
    current["pending_owner_report"] = report
    assert evolution_lifecycle._write_evolution_campaign(current) is True

    def _send_then_record(*args, **kwargs):
        receipt = evolution_lifecycle.record_evolution_commit(
            campaign["id"], tx["transaction_id"], tx["task_id"], "5" * 40,
        )
        assert receipt["ok"] is True

    monkeypatch.setattr(queue, "notify_owner_cycle_outcome", _send_then_record)

    queue._deliver_pending_owner_report()

    stored = evolution_lifecycle._read_evolution_campaign()
    assert "pending_owner_report" not in stored
    assert stored["active_transaction"]["commit_receipt"]["commit_sha"] == "5" * 40


def test_terminal_campaign_cannot_be_resurrected_by_a_stale_writer(tmp_path):
    from supervisor import evolution_lifecycle

    campaign, _ = _active_transaction(tmp_path)
    stale = dict(campaign)
    evolution_lifecycle.complete_evolution_campaign("owner stop", cleanup_worktree=False)
    stale["status"] = "active"

    assert evolution_lifecycle._write_evolution_campaign(stale) is False
    assert evolution_lifecycle._read_evolution_campaign()["status"] == "stopped"


def test_stale_campaign_cannot_overwrite_a_new_campaign(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    first = evolution_lifecycle.start_evolution_campaign("First", source="test")
    stale = dict(first)
    evolution_lifecycle.complete_evolution_campaign("done", cleanup_worktree=False)
    second = evolution_lifecycle.start_evolution_campaign("Second", source="test")

    stale["status"] = "active"
    assert evolution_lifecycle._write_evolution_campaign(stale) is False
    assert evolution_lifecycle._read_evolution_campaign()["id"] == second["id"]


def test_panic_campaign_close_uses_nonblocking_state_lock(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    timeouts = []
    monkeypatch.setattr(
        state,
        "acquire_file_lock",
        lambda path, timeout_sec=4.0, **kw: timeouts.append(timeout_sec) or None,
    )

    evolution_lifecycle.complete_evolution_campaign(
        "panic stop", status="stopped", cleanup_worktree=False,
    )

    assert timeouts == [0.001]


def test_evolution_commit_refuses_review_when_claim_is_gone(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    reviewed = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(
        git_tools, "_evolution_commit_authority",
        lambda *a, **k: ({}, {"ok": False, "reason": "owner_stopped"}),
    )
    monkeypatch.setattr(
        git_tools, "_run_reviewed_stage_cycle",
        lambda *a, **k: reviewed.append(True) or {"status": "passed"},
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id="evo",
        task_metadata={},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "EVOLUTION_AUTHORITY_REVOKED" in result
    assert reviewed == []


def test_postcommit_cas_failure_returns_local_orphan_after_binding(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools
    from supervisor import evolution_lifecycle

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    tagged, contained = [], []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(git_tools, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    monkeypatch.setattr(git_tools, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        git_tools, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        },
    )
    monkeypatch.setattr(
        git_tools, "run_cmd",
        lambda cmd, cwd=None: "d" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "record_evolution_commit",
        lambda **kwargs: {"ok": False, "reason": "owner_stopped", "commit_sha": kwargs["commit_sha"]},
    )
    monkeypatch.setattr(
        git_tools,
        "_auto_tag_on_version_bump",
        lambda *a, **k: tagged.append(True) or "",
    )
    monkeypatch.setattr(
        git_tools,
        "_preserve_evolution_orphan",
        lambda *a, **k: contained.append((a, k)) or "contained",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "EVOLUTION_COMMIT_ORPHANED" in result
    assert "d" * 40 in result
    assert tagged == [True]
    assert len(contained) == 1


@pytest.mark.parametrize(
    ("task_type", "expected_order"),
    [
        ("evolution", ["authority", "push", "release", "publish"]),
        ("task", ["release", "push", "publish"]),
    ],
)
def test_only_evolution_push_stays_under_git_lock(
    tmp_path, monkeypatch, task_type, expected_order,
):
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    order = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: order.append("release"))
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(git_tools, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    monkeypatch.setattr(git_tools, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    monkeypatch.setattr(
        git_tools,
        "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        },
    )
    monkeypatch.setattr(
        git_tools,
        "run_cmd",
        lambda cmd, cwd=None: "d" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(git_tools, "_auto_tag_on_version_bump", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_evolution_commit_receipt", lambda *a, **k: "")
    monkeypatch.setattr(
        git_tools,
        "_evolution_publication_stopped_result",
        lambda *a, **k: order.append("authority") or "",
    )
    monkeypatch.setattr(git_tools, "_auto_push", lambda *a, **k: order.append("push") or "")
    monkeypatch.setattr(
        git_tools,
        "_publish_reviewed_commit",
        lambda *a, **k: order.append("publish") or "ok",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type=task_type,
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    assert git_tools._repo_commit_push(ctx, "test commit", skip_tests=True) == "ok"
    assert order == expected_order


def test_revoked_publication_does_not_record_or_anchor_success(tmp_path, monkeypatch):
    from ouroboros import mutation_attribution
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    sha = "d" * 40
    attempts, baselines, contained, pushed = [], [], [], []
    authority = iter([
        (claim, {"ok": True}),
        (claim, {"ok": True}),
        (claim, {"ok": True}),
        (claim, {"ok": False, "reason": "owner_stopped"}),
    ])
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", ("root", "task")))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: attempts.append((k.get("status") or a[2], k)))
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(git_tools, "_evolution_commit_authority", lambda *a, **k: next(authority))
    monkeypatch.setattr(git_tools, "_verify_reviewed_commit_binding", lambda *a, **k: (True, ""))
    def reviewed(review_ctx, *args, **kwargs):
        review_ctx._last_triad_raw_results = [{"raw": "triad"}]
        review_ctx._last_scope_raw_result = {"raw": "scope"}
        review_ctx._review_degraded_reasons = ["recorded"]
        return {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        }
    monkeypatch.setattr(git_tools, "_run_reviewed_stage_cycle", reviewed)
    monkeypatch.setattr(git_tools, "run_cmd", lambda cmd, cwd=None: sha if cmd[:3] == ["git", "rev-parse", "HEAD"] else "")
    monkeypatch.setattr(git_tools, "_auto_tag_on_version_bump", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_evolution_commit_receipt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_preserve_evolution_orphan", lambda *a, **k: contained.append(True) or "contained")
    monkeypatch.setattr(git_tools, "_auto_push", lambda *a, **k: pushed.append(True) or "")
    monkeypatch.setattr(mutation_attribution, "advance_mutation_baseline", lambda *a, **k: baselines.append(a))
    ctx = SimpleNamespace(
        repo_dir=tmp_path, drive_root=tmp_path, branch_dev="ouroboros",
        current_task_type="evolution", task_id="evo",
        task_metadata={"evolution_transaction": claim},
        _scope_review_history={"keep": True},
    )

    result = git_tools._repo_commit_push(ctx, "test commit", skip_tests=True)

    assert "EVOLUTION_PUBLICATION_STOPPED" in result
    assert contained == [True]
    assert pushed == []
    assert baselines == []
    statuses = [status for status, _details in attempts]
    assert "succeeded" not in statuses and statuses[-1] == "failed"
    failed = attempts[-1][1]
    assert failed["fingerprint_status"] == "matched"
    assert failed["pre_review_fingerprint"] == "pre"
    assert failed["post_review_fingerprint"] == "post"
    assert failed["triad_raw_results"] == [{"raw": "triad"}]
    assert failed["scope_raw_result"] == {"raw": "scope"}
    assert failed["degraded_reasons"] == ["recorded"]
    assert not getattr(ctx, "last_reviewed_commit_sha", "")
    assert ctx._scope_review_history == {"keep": True}


def test_postcommit_binding_failure_contains_evolution_commit(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    contained = []
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(git_tools, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    monkeypatch.setattr(
        git_tools,
        "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {"fingerprint": "post", "binding": {}},
        },
    )
    monkeypatch.setattr(
        git_tools,
        "run_cmd",
        lambda cmd, cwd=None: "2" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(
        git_tools,
        "_verify_reviewed_commit_binding",
        lambda *a, **k: (False, "tree mismatch"),
    )
    monkeypatch.setattr(
        git_tools,
        "_preserve_evolution_orphan",
        lambda *a, **k: contained.append((a, k)) or "contained",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "REVIEW_BINDING_FAILED" in result
    assert "contained" in result
    assert len(contained) == 1
    assert contained[0][1] == {}


def test_final_tag_binding_failure_cannot_record_restart_receipt(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    claim = {"campaign_id": "camp", "transaction_id": "tx", "task_id": "evo"}
    recorded = []
    contained = []
    binding_results = iter([(True, ""), (False, "tag target mismatch")])
    monkeypatch.setattr(git_tools, "_task_attributed_commit_paths", lambda *a, **k: (None, None, "", None))
    monkeypatch.setattr(git_tools, "_check_overlapping_review_attempt", lambda *a, **k: "")
    monkeypatch.setattr(git_tools, "_record_commit_attempt", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_acquire_git_lock", lambda *a, **k: pathlib.Path("lock"))
    monkeypatch.setattr(git_tools, "_release_git_lock", lambda *a, **k: None)
    monkeypatch.setattr(git_tools, "_prepare_review_commit_worktree", lambda *a, **k: (False, ""))
    monkeypatch.setattr(git_tools, "_evolution_commit_authority", lambda *a, **k: (claim, {"ok": True}))
    monkeypatch.setattr(
        git_tools, "_verify_reviewed_commit_binding", lambda *a, **k: next(binding_results),
    )
    monkeypatch.setattr(
        git_tools, "_run_reviewed_stage_cycle",
        lambda *a, **k: {
            "status": "passed",
            "pre_fingerprint": {"fingerprint": "pre"},
            "post_fingerprint": {
                "fingerprint": "post",
                "binding": {"expected_tag": "v-test"},
            },
        },
    )
    monkeypatch.setattr(
        git_tools, "run_cmd",
        lambda cmd, cwd=None: "1" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setattr(
        git_tools,
        "_auto_tag_on_version_bump",
        lambda *a, **k: " [tagged: v-test]",
    )
    monkeypatch.setattr(
        git_tools,
        "_preserve_evolution_orphan",
        lambda *a, **k: contained.append((a, k)) or "contained",
    )
    monkeypatch.setattr(
        git_tools,
        "_record_evolution_commit_receipt",
        lambda *a, **k: recorded.append(True) or "",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        drive_root=tmp_path,
        branch_dev="ouroboros",
        current_task_type="evolution",
        task_id="evo",
        task_metadata={"evolution_transaction": claim},
    )

    result = git_tools._repo_commit_push(ctx, "test commit")

    assert "REVIEW_BINDING_FAILED" in result
    assert recorded == []
    assert len(contained) == 1
    assert contained[0][1]["created_tag"] == "v-test"


def test_evolution_publication_authority_requires_exact_head(tmp_path, monkeypatch):
    from ouroboros.tools import git as git_tools

    monkeypatch.setattr(
        "supervisor.evolution_lifecycle.check_evolution_authority",
        lambda **kwargs: {"ok": True, "reason": ""},
    )
    monkeypatch.setattr(
        git_tools,
        "run_cmd",
        lambda cmd, cwd=None: "b" * 40 if cmd[:3] == ["git", "rev-parse", "HEAD"] else "",
    )
    ctx = SimpleNamespace(
        repo_dir=tmp_path,
        task_id="evo",
        task_metadata={"evolution_transaction": {
            "campaign_id": "camp",
            "transaction_id": "tx",
            "task_id": "evo",
        }},
    )

    _, authority = git_tools._evolution_commit_authority(ctx, commit_sha="a" * 40)

    assert authority["ok"] is False
    assert authority["reason"] == "head_mismatch"


def test_evolution_promote_event_carries_exact_claim():
    from ouroboros.tools import control

    ctx = SimpleNamespace(
        current_task_type="evolution",
        task_id="evo-task",
        task_metadata={"evolution_transaction": {
            "campaign_id": "campaign",
            "transaction_id": "transaction",
            "task_id": "evo-task",
            "commit_sha": "",
        }},
        last_reviewed_commit_sha="a" * 40,
        pending_events=[],
    )

    control._promote_to_stable(ctx, "reviewed")

    event = ctx.pending_events[0]
    assert event["type"] == "promote_to_stable"
    assert event["reason"] == "reviewed"
    assert event["evolution_claim"] == {
        "campaign_id": "campaign",
        "transaction_id": "transaction",
        "task_id": "evo-task",
        "commit_sha": "a" * 40,
    }


def test_promote_to_stable_rechecks_evolution_claim_without_changing_normal_flow(
    tmp_path, monkeypatch,
):
    from supervisor import events, evolution_lifecycle

    repo = tmp_path / "repo"
    repo.mkdir()

    def _git(*args):
        return subprocess.run(
            ["git", *args], cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

    _git("init", "-b", "ouroboros")
    _git("config", "user.name", "Test")
    _git("config", "user.email", "test@example.com")
    (repo / "file.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "base")
    base_sha = _git("rev-parse", "HEAD")
    _git("branch", "ouroboros-stable", base_sha)
    (repo / "file.txt").write_text("reviewed\n", encoding="utf-8")
    _git("add", ".")
    _git("commit", "-m", "reviewed")
    reviewed_sha = _git("rev-parse", "HEAD")
    sent = []
    ctx = SimpleNamespace(
        REPO_DIR=repo,
        BRANCH_DEV="ouroboros",
        BRANCH_STABLE="ouroboros-stable",
        load_state=lambda: {"owner_chat_id": 1},
        send_with_budget=lambda chat_id, message: sent.append(message),
    )
    monkeypatch.setattr(
        evolution_lifecycle,
        "check_evolution_authority",
        lambda **claim: {
            "ok": claim.get("campaign_id") == "valid",
            "reason": "owner_stopped" if claim.get("campaign_id") != "valid" else "",
        },
    )

    events._handle_promote_to_stable({
        "type": "promote_to_stable",
        "evolution_claim": {
            "campaign_id": "revoked",
            "transaction_id": "tx",
            "task_id": "evo",
            "commit_sha": reviewed_sha,
        },
    }, ctx)
    assert _git("rev-parse", "ouroboros-stable") == base_sha
    assert "owner_stopped" in sent[-1]

    events._handle_promote_to_stable({
        "type": "promote_to_stable",
        "evolution_claim": {
            "campaign_id": "",
            "transaction_id": "",
            "task_id": "",
            "commit_sha": "",
        },
    }, ctx)
    assert _git("rev-parse", "ouroboros-stable") == base_sha
    assert "commit_receipt_missing" in sent[-1]

    events._handle_promote_to_stable({
        "type": "promote_to_stable",
        "evolution_claim": {
            "campaign_id": "valid",
            "transaction_id": "tx",
            "task_id": "evo",
            "commit_sha": reviewed_sha,
        },
    }, ctx)
    assert _git("rev-parse", "ouroboros-stable") == reviewed_sha

    _git("branch", "-f", "ouroboros-stable", base_sha)
    events._handle_promote_to_stable({"type": "promote_to_stable"}, ctx)
    assert _git("rev-parse", "ouroboros-stable") == reviewed_sha


def test_restart_requires_the_exact_active_commit_receipt(tmp_path, monkeypatch):
    from ouroboros.tools import control
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "e" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
    }
    assert evolution_lifecycle.record_evolution_commit(**claim, commit_sha=sha)["ok"] is True
    monkeypatch.setattr(
        control,
        "run_cmd",
        lambda cmd, cwd=None: "" if cmd[:2] == ["git", "status"] else sha,
    )
    ctx = SimpleNamespace(
        current_task_type="evolution",
        repo_dir=tmp_path,
        task_id=tx["task_id"],
        task_metadata={"evolution_transaction": tx},
        last_reviewed_commit_sha=sha,
    )
    assert control._evolution_restart_block_reason(ctx) == ""

    live = state.load_state()
    live["evolution_owner_stopped"] = True
    live["evolution_mode_enabled"] = False
    state.save_state(live)
    assert "owner_stopped" in control._evolution_restart_block_reason(ctx)


def test_boot_restart_verifies_exact_v2_claim_only_after_new_generation(
    tmp_path, monkeypatch,
):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "server-a"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    assert tx["schema_version"] == 2
    sha = "8" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert int(stored.get("absorbed_cycles_done") or 0) == 0
    assert marker.is_file()

    generation["value"] = "server-b"
    agent_startup_checks.verify_restart(env, sha)

    stored = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in stored
    assert stored["transaction_history"][-1]["cycle_outcome"] == "absorbed"
    assert stored["last_boot_reconcile_gen"] == "server-b"
    assert not marker.exists()


def test_boot_restart_rejects_mismatched_claim_without_loser_bypass(
    tmp_path, monkeypatch,
):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "9" * 40
    exact = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**exact)["ok"] is True
    stale = {**exact, "transaction_id": "stale-transaction"}
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"expected_sha": sha, "evolution_claim": stale}))
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)
    agent_startup_checks.verify_restart(env, sha)  # a rename loser must not reconcile it

    stored = evolution_lifecycle._read_evolution_campaign()
    assert stored["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert stored["active_transaction"]["restart_authority_error"] == "restart_claim_mismatch"
    assert int(stored.get("absorbed_cycles_done") or 0) == 0
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "restart_verify"
    assert event["error"] == "restart_claim_mismatch"


def test_boot_markerless_v2_missing_receipt_stays_unresolved(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    stored = evolution_lifecycle._read_evolution_campaign()
    stored["active_transaction"]["commit_sha"] = "a" * 40
    assert evolution_lifecycle._write_evolution_campaign(stored) is True
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen")
    monkeypatch.setattr(
        agent_startup_checks.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, "head")

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert current["active_transaction"]["restart_authority_error"] == "commit_receipt_missing"
    assert int(current.get("absorbed_cycles_done") or 0) == 0
    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "evolution_tx_reconcile_blocked"
    assert event["reason"] == "commit_receipt_missing"


def test_boot_rename_loser_waits_for_claim_winner(tmp_path):
    from ouroboros import agent_startup_checks
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "b" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    claimed = tmp_path / "state" / f"pending_restart_verify.claimed.{os.getpid()}.json"
    claimed.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert int(current.get("absorbed_cycles_done") or 0) == 0


def test_boot_reclaims_dead_restart_claim(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, platform_layer, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "server-a"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "7" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    claimed = tmp_path / "state" / "pending_restart_verify.claimed.999999999.json"
    claimed.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    generation["value"] = "server-b"
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    current = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in current
    assert current["transaction_history"][-1]["cycle_outcome"] == "absorbed"
    assert list((tmp_path / "state").glob("pending_restart_verify.claimed.*.json")) == []


def test_new_campaign_is_stamped_for_same_generation_worker_respawns(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle, queue, state

    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "same-server")
    state.init(tmp_path)
    queue.init(tmp_path, 600, 1800)
    queue.init_queue_refs([], {}, {"value": 0})
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    assert campaign["last_boot_reconcile_gen"] == "same-server"
    state.update_state(lambda live: live.update(evolution_mode_enabled=True))
    tx = evolution_lifecycle.begin_evolution_transaction("respawn", cycle=1, campaign=campaign)
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": "6" * 40,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, claim["commit_sha"])

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["active_transaction"]["transaction_id"] == tx["transaction_id"]
    assert int(current.get("absorbed_cycles_done") or 0) == 0


def test_boot_reconcile_cannot_resurrect_owner_stopped_campaign(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "before-restart"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "4" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    generation["value"] = "after-restart"
    reached = threading.Event()
    release = threading.Event()

    def delayed_merge_base(*args, **kwargs):
        reached.set()
        assert release.wait(timeout=2)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(agent_startup_checks.subprocess, "run", delayed_merge_base)
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )
    worker = threading.Thread(target=agent_startup_checks.verify_restart, args=(env, "5" * 40))
    worker.start()
    assert reached.wait(timeout=2)
    evolution_lifecycle.complete_evolution_campaign("owner stop", cleanup_worktree=False)
    release.set()
    worker.join(timeout=2)

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["status"] == "stopped"
    assert current["completion_reason"] == "owner stop"
    assert "active_transaction" not in current


def test_owner_stop_preserves_prior_boot_reconciliation_evidence(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "before-restart"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "3" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    generation["value"] = "after-restart"
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )
    agent_startup_checks.verify_restart(env, sha)
    evolution_lifecycle.complete_evolution_campaign("owner stop", cleanup_worktree=False)

    current = evolution_lifecycle._read_evolution_campaign()
    assert current["status"] == "stopped"
    assert current["absorbed_cycles_done"] == 1
    assert current["last_boot_reconcile_gen"] == "after-restart"
    assert current["transaction_history"][-1]["cycle_outcome"] == "absorbed"


def test_boot_restart_writers_obey_live_root_fuse(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks
    from supervisor import evolution_lifecycle

    campaign, tx = _active_transaction(tmp_path)
    sha = "2" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    before = campaign_path.read_bytes()
    monkeypatch.setenv("OUROBOROS_PYTEST_ACTIVE", "1")
    monkeypatch.setenv("OUROBOROS_TEST_LIVE_DATA_ROOT", str(tmp_path))
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    with pytest.raises(RuntimeError, match="PYTEST_LIVE_DATA_WRITE_BLOCKED"):
        agent_startup_checks.verify_restart(env, sha)

    assert campaign_path.read_bytes() == before
    assert list((tmp_path / "state").glob("pending_restart_verify.claimed.*.json")) == []


def test_boot_restart_write_failure_restores_claim_for_retry(tmp_path, monkeypatch):
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle

    generation = {"value": "server-a"}
    monkeypatch.setattr(
        process_custody, "current_custody_session_id", lambda: generation["value"],
    )
    campaign, tx = _active_transaction(tmp_path)
    sha = "c" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(**claim)["ok"] is True
    pending = tmp_path / "state" / "pending_restart_verify.json"
    pending.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    real_write = agent_startup_checks.atomic_write_json
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("temporary write failure")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(agent_startup_checks, "atomic_write_json", fail_once)
    generation["value"] = "server-b"
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)
    assert pending.is_file()
    assert list((tmp_path / "state").glob("pending_restart_verify.claimed.*.json")) == []

    agent_startup_checks.verify_restart(env, sha)
    current = evolution_lifecycle._read_evolution_campaign()
    assert "active_transaction" not in current
    assert current["transaction_history"][-1]["cycle_outcome"] == "absorbed"


def test_boot_exact_claim_never_passes_without_active_transaction(tmp_path):
    from ouroboros import agent_startup_checks

    (tmp_path / "state").mkdir(parents=True)
    (tmp_path / "logs").mkdir(parents=True)
    sha = "d" * 40
    claim = {
        "campaign_id": "campaign",
        "transaction_id": "transaction",
        "task_id": "task",
        "commit_sha": sha,
    }
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"expected_sha": sha, "evolution_claim": claim}))
    campaign_path = tmp_path / "state" / "evolution_campaign.json"
    campaign_path.write_text("{bad")
    env = SimpleNamespace(
        drive_path=lambda name: tmp_path / name,
        drive_root=tmp_path,
        repo_dir=tmp_path,
    )

    agent_startup_checks.verify_restart(env, sha)

    event = json.loads((tmp_path / "logs" / "events.jsonl").read_text().splitlines()[-1])
    assert event["type"] == "restart_verify"
    assert event["ok"] is False
    assert event["error"] == "transaction_missing"
    assert marker.exists() is False
    assert campaign_path.read_text() == "{bad"


def test_evolution_restart_write_failure_does_not_become_generic_restart(tmp_path, monkeypatch):
    from ouroboros.tools import control

    monkeypatch.setattr(control, "_evolution_restart_block_reason", lambda ctx: "")
    monkeypatch.setattr(
        control, "run_cmd",
        lambda cmd, cwd=None: "f" * 40 if cmd[-1] == "HEAD" else "ouroboros",
    )
    monkeypatch.setattr(
        control, "atomic_write_json",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )
    ctx = SimpleNamespace(
        current_task_type="evolution",
        repo_dir=tmp_path,
        drive_path=lambda name: tmp_path / name,
        task_id="evo",
        task_metadata={"evolution_transaction": {}},
        pending_restart_reason=None,
        last_push_succeeded=True,
        last_reviewed_commit_sha="f" * 40,
    )

    result = control._request_restart(ctx, "apply reviewed evolution")

    assert "RESTART_BLOCKED" in result
    assert ctx.pending_restart_reason is None


def test_supervisor_rechecks_evolution_claim_immediately_before_restart(tmp_path):
    import server
    from supervisor import evolution_lifecycle, state

    campaign, tx = _active_transaction(tmp_path)
    sha = "1" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": sha,
    }
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], sha,
    )["ok"] is True
    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({
        "reason": "evolution restart",
        "expected_sha": sha,
        "evolution_claim": claim,
    }))
    live = state.load_state()
    live.update({"evolution_mode_enabled": False, "evolution_owner_stopped": True})
    state.save_state(live)
    restarted = []
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        load_state=state.load_state,
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="evolution restart", evolution_restart=True,
    )

    assert restarted == []
    assert "owner_stopped" in messages[0][1]


def test_supervisor_blocks_evolution_restart_if_marker_disappears_during_drain(tmp_path):
    import server

    restarted = []
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        load_state=lambda: {"owner_chat_id": 1},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx,
        restart_reason="agent_requested_restart",
        evolution_restart=True,
    )

    assert restarted == []
    assert "receipt is missing" in messages[0][1]


def test_generic_restart_ignores_stale_evolution_marker(tmp_path, monkeypatch):
    import server

    marker = tmp_path / "state" / "pending_restart_verify.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({
        "reason": "agent_requested_restart",
        "evolution_claim": {"campaign_id": "stale"},
    }))
    restarted = []
    exited = []
    monkeypatch.setattr(server, "_request_restart_exit", lambda: exited.append(True))
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        load_state=lambda: {},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        kill_workers=lambda **k: None,
        save_state=lambda state: None,
        persist_queue_snapshot=lambda **k: None,
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="agent_requested_restart", evolution_restart=False,
    )

    assert restarted
    assert exited == [True]


def test_supervisor_blocks_restart_when_head_moved_after_receipt(tmp_path):
    import server
    from supervisor import evolution_lifecycle

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    (repo / "file.txt").write_text("current\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "current"], cwd=repo, check=True, capture_output=True)
    campaign, tx = _active_transaction(tmp_path)
    reviewed_sha = "2" * 40
    claim = {
        "campaign_id": campaign["id"],
        "transaction_id": tx["transaction_id"],
        "task_id": tx["task_id"],
        "commit_sha": reviewed_sha,
    }
    assert evolution_lifecycle.record_evolution_commit(
        campaign["id"], tx["transaction_id"], tx["task_id"], reviewed_sha,
    )["ok"] is True
    (tmp_path / "state" / "pending_restart_verify.json").write_text(json.dumps({
        "reason": "evolution restart",
        "expected_sha": reviewed_sha,
        "evolution_claim": claim,
    }))
    restarted = []
    messages = []
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        REPO_DIR=repo,
        load_state=lambda: {"owner_chat_id": 1},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"),
        send_with_budget=lambda *a: messages.append(a),
    )

    server._perform_supervisor_restart(
        ctx, restart_reason="evolution restart", evolution_restart=True,
    )

    assert restarted == []
    assert "no longer matches" in messages[0][1]


def test_benchmark_seed_creates_campaign_before_enabling(tmp_path):
    from devtools.benchmarks.common.server_runner import seed_owner_state

    seed_owner_state(tmp_path, evolution_enabled=True)

    state = json.loads((tmp_path / "state" / "state.json").read_text())
    campaign = json.loads((tmp_path / "state" / "evolution_campaign.json").read_text())
    assert campaign["status"] == "active"
    assert campaign["id"]
    assert state["evolution_mode_enabled"] is True
