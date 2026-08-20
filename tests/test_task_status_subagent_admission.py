"""Subagent admission: the lineage, depth and capacity a delegation must satisfy.

Split out of ``tests/test_task_status_flow.py`` by theme: the accepted unique subagent
with its lineage and constraint, the child-drive contract, the chat routing without an
owner, the depth rejection including a configured zero, the legacy event schema, and the
queueing and fail-fast paths around the active-subagent cap and the worker pool.
"""

import json
import pathlib
from types import SimpleNamespace


def test_handle_schedule_task_accepts_unique_subagent_with_lineage_and_constraint(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_SCHEDULED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            self.snapshot_reason = reason

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "child123",
            "objective": "Inspect scheduling",
            "expected_output": "Findings table",
            "constraints": "No writes",
            "role": "reviewer",
            "context": "Parent facts",
            "depth": 1,
            "parent_task_id": "parent123",
            "root_task_id": "root123",
            "session_id": "sess123",
            "actor_id": "subagent:reviewer",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "child123" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "child123" / "data"),
            "budget_drive_root": str(tmp_path),
            "task_constraint": {"mode": "skill_repair", "allow_enable": True, "allow_review": True},
        },
        FakeCtx(),
    )

    assert len(enqueued) == 1
    task = enqueued[0]
    assert task["id"] == "child123"
    assert task["parent_task_id"] == "parent123"
    assert task["root_task_id"] == "root123"
    assert task["session_id"] == "sess123"
    assert task["role"] == "reviewer"
    assert task["memory_mode"] == "forked"
    assert task["child_drive_root"] == task["drive_root"]
    assert task["task_constraint"]["mode"] == "local_readonly_subagent"
    assert task["task_constraint"]["allow_enable"] is False
    assert task["task_constraint"]["allow_review"] is False
    assert "[EXPECTED_OUTPUT]" in task["text"]
    assert "[BEGIN_PARENT_CONTEXT" in task["text"]
    data = json.loads((tmp_path / "task_results" / "child123.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_SCHEDULED
    assert data["expected_output"] == "Findings table"
    assert data["child_drive_root"] == task["drive_root"]
    assert data["task_constraint"]["mode"] == "local_readonly_subagent"
    assert "Do not delegate further" not in task["text"]
    assert "Nested readonly delegation is allowed only through schedule_subagent" in task["text"]
    assert sent and sent[0][2].get("is_progress") is True


def test_handle_schedule_task_rejects_internal_subagent_without_child_drive_contract(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            raise AssertionError("invalid internal subagent should not enqueue")

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "badchild",
            "objective": "Inspect invalid event",
            "expected_output": "Nothing",
            "depth": 1,
            "delegation_role": "subagent",
            "memory_mode": "shared",
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "badchild.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert "memory_mode=forked or empty" in data["result"]
    assert sent and sent[0][2]["progress_meta"]["subagent_event"] == "rejected"
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["parent_task_id"] == ""
    assert sent[0][2]["progress_meta"]["status"] == STATUS_FAILED


def test_handle_schedule_task_uses_event_chat_id_without_owner(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_SCHEDULED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            self.snapshot_reason = reason

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "headless1",
            "objective": "Inspect no-owner path",
            "expected_output": "Findings",
            "depth": 1,
            "chat_id": 44,
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "headless1" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "headless1" / "data"),
        },
        FakeCtx(),
    )

    assert len(enqueued) == 1
    assert enqueued[0]["chat_id"] == 44
    scheduled = json.loads((tmp_path / "task_results" / "headless1.json").read_text(encoding="utf-8"))
    assert scheduled["status"] == STATUS_SCHEDULED
    assert scheduled["chat_id"] == 44
    assert sent and sent[0][0] == 44

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "headless2",
            "objective": "Inspect missing chat target",
            "expected_output": "Findings",
            "depth": 1,
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "headless2" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "headless2" / "data"),
        },
        FakeCtx(),
    )

    # B1 (v6.33.0): a headless subagent with no chat target is no longer
    # rejected — it is enqueued and runs (the live "🗓️ Scheduled" notification is
    # skipped because chat_id is 0). Restores headless/CLI multi-agent.
    assert len(enqueued) == 2
    assert enqueued[1]["id"] == "headless2"
    scheduled2 = json.loads((tmp_path / "task_results" / "headless2.json").read_text(encoding="utf-8"))
    assert scheduled2["status"] == STATUS_SCHEDULED
    # No chat notification was emitted for the chat-less subagent.
    assert all(s[0] != 0 for s in sent)
    assert len(sent) == 1


def test_handle_schedule_task_depth_rejection_writes_failed_status(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.config import get_max_subagent_depth
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            raise AssertionError("depth-rejected task should not enqueue")

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "deep1",
            "objective": "Too deep",
            "expected_output": "Nothing",
            "depth": get_max_subagent_depth() + 1,
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "deep1" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "deep1" / "data"),
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "deep1.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert "depth limit" in data["result"]
    assert sent and "depth limit" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["status"] == STATUS_FAILED


def test_configured_zero_subagent_depth_truly_disables_delegation(tmp_path, monkeypatch):
    """v6.79.0 (owner Q26): a configured depth of 0 means NO delegation.

    Before this, ``_bounded_positive_int_setting`` rewrote a configured 0 to the default 2,
    so every run that asked for "no swarm" silently delegated two levels deep. All three
    facts are pinned together: the resolved setting, the tool-side gate, and the supervisor
    gate — plus the invariant that a ROOT task (depth 0 itself) still runs at depth 0."""
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.config import get_max_subagent_depth
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "0")
    assert get_max_subagent_depth() == 0

    # Tool-side gate: the first child of a root task is already too deep.
    import ouroboros.tools.control as control
    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "root-no-swarm"
    ctx.task_depth = 0
    out = control._schedule_task(ctx, objective="Delegate", expected_output="Something")
    assert "depth limit (0) exceeded" in out

    # Supervisor gate: a depth-1 child event is refused; a depth-0 ROOT task is NOT.
    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            pass

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    def _event(task_id: str, depth: int) -> dict:
        return {
            "type": "schedule_subagent",
            "task_id": task_id,
            "objective": "work",
            "expected_output": "result",
            "depth": depth,
            "delegation_role": "subagent" if depth else "",
            "memory_mode": "forked",
            "chat_id": 1,
            "drive_root": str(tmp_path / "state" / "headless_tasks" / task_id / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / task_id / "data"),
        }

    ev_module._handle_schedule_task(_event("child-at-1", 1), FakeCtx())
    child = json.loads((tmp_path / "task_results" / "child-at-1.json").read_text(encoding="utf-8"))
    assert child["status"] == STATUS_FAILED and "depth limit (0)" in child["result"]
    assert not enqueued

    ev_module._handle_schedule_task(_event("root-at-0", 0), FakeCtx())
    root = json.loads((tmp_path / "task_results" / "root-at-0.json").read_text(encoding="utf-8"))
    assert root["status"] != STATUS_FAILED
    assert enqueued and enqueued[0]["id"] == "root-at-0"


def test_other_bounded_int_settings_keep_their_min_of_one(monkeypatch):
    """``min_value`` defaults to 1, so the depth fix does not leak into sibling settings."""
    from ouroboros.config import get_max_active_subagents_per_root, SETTINGS_DEFAULTS

    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "0")
    assert get_max_active_subagents_per_root() == int(
        SETTINGS_DEFAULTS["OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"]
    )


def test_settings_ui_carries_a_configured_zero_subagent_depth():
    """The runtime honouring 0 is worthless if the Settings page silently reverts it: 0 is FALSY
    in JS, so a stored 0 read through the plain `if (value)` branch displayed the fallback 2, and
    the next Save (which posts every number field unconditionally) wrote 2 back — re-enabling two
    levels of delegation through the UI. All three carriers of the owner's 0 are pinned: the input
    can reach it, the depth entry is falsy-tolerant, and the load path still honours that flag
    (without which the flag is inert)."""
    root = pathlib.Path(__file__).resolve().parents[1]
    settings_js = (root / "web" / "modules" / "settings.js").read_text(encoding="utf-8")
    # The input moved from Advanced -> Runtime Limits to Agents -> Delegation
    # (D-10): the counts bound the agents, not the process pool. Same invariant,
    # new address.
    settings_ui = (root / "web" / "modules" / "subagents_settings.js").read_text(encoding="utf-8")
    assert 'id="s-subagent-depth" type="number" min="0"' in settings_ui
    # The 4th tuple element is the falsy-tolerant flag consumed by the load path below.
    assert "['s-subagent-depth', 'OUROBOROS_MAX_SUBAGENT_DEPTH', 2, true]" in settings_js
    assert (
        "if (allowFalsy ? value !== null && value !== undefined : value) byId(id).value = value;"
        in settings_js
    ), "the load path no longer honours the falsy-tolerant flag, so the entry is inert"


def test_handle_schedule_task_rejects_legacy_subagent_event_schema(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    enqueued = []
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            return None

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "legacy123",
            "description": "Old child form",
            "context": "old reference",
            "parent_task_id": "parent123",
            "delegation_role": "subagent",
        },
        FakeCtx(),
    )

    assert enqueued == []
    data = json.loads((tmp_path / "task_results" / "legacy123.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert "objective and expected_output" in data["result"]
    assert sent and "objective and expected_output" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["parent_task_id"] == "parent123"
    assert sent[0][2]["progress_meta"]["status"] == STATUS_FAILED


def test_handle_schedule_task_queues_when_active_subagent_cap_is_full(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_COMPLETED, STATUS_FAILED, STATUS_SCHEDULED, load_task_result, write_task_result

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "3")  # pin cap (v6.20.0 raised default to 6)
    sent = []
    enqueued = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = [{"id": f"p{i}", "root_task_id": "root123", "delegation_role": "subagent"} for i in range(2)]
        RUNNING = {"r1": {"task": {"id": "r1", "root_task_id": "root123", "delegation_role": "subagent"}}}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "child999",
            "objective": "Too many",
            "expected_output": "Nothing",
            "depth": 1,
            "root_task_id": "root123",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "child999" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "child999" / "data"),
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "child999.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_SCHEDULED
    assert enqueued and enqueued[0]["id"] == "child999"
    assert sent and "queued behind active subagent cap" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["queued_behind_active_cap"] is True

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "child1000",
            "objective": "Too many again",
            "expected_output": "Nothing",
            "depth": 1,
            "root_task_id": "root123",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "child1000" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "child1000" / "data"),
        },
        FakeCtx(),
    )
    data2 = json.loads((tmp_path / "task_results" / "child1000.json").read_text(encoding="utf-8"))
    assert data2["status"] == STATUS_SCHEDULED
    assert any(task["id"] == "child1000" for task in enqueued)

    child_drive = tmp_path / "state" / "headless_tasks" / "childdone" / "data"
    (child_drive / "memory").mkdir(parents=True)
    (child_drive / "memory" / "identity.md").write_text("child identity", encoding="utf-8")
    child_review_projection = {
        "panels": [{
            "panel_id": "child-panel",
            "aggregate_signal": "DEGRADED",
            "actors": [],
        }],
    }
    child_outcome_axes = {
        "lifecycle": {"status": "completed"},
        "execution": {"status": "ok"},
        "objective": {"status": "best_effort"},
        "review": {"status": "degraded"},
        "artifacts": {"status": "ready"},
    }
    write_task_result(
        child_drive,
        "childdone",
        STATUS_COMPLETED,
        result="summary",
        outcome_axes=child_outcome_axes,
        reason_code="acceptance_degraded",
        review_projection=child_review_projection,
    )

    sent = []
    worker = SimpleNamespace(busy_task_id="childdone")
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "childdone": {
                "task": {
                    "id": "childdone",
                    "chat_id": 1,
                    "drive_root": str(child_drive),
                    "delegation_role": "subagent",
                    "role": "reviewer",
                    "root_task_id": "root123",
                    "parent_task_id": "parent123",
                    "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
                }
            }
        },
        WORKERS={7: worker},
        bridge=SimpleNamespace(push_log=lambda _payload: None),
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
        persist_queue_snapshot=lambda reason="": None,
    )

    ev_module._handle_task_done({"task_id": "childdone", "worker_id": 7, "task_type": "task"}, ctx)

    assert load_task_result(tmp_path, "childdone")["result"] == "summary"
    assert not (tmp_path / "task_results" / "artifacts" / "childdone" / "memory_export.json").exists()
    assert sent and sent[-1][2]["progress_meta"]["subagent_role"] == "reviewer"
    terminal_meta = sent[-1][2]["progress_meta"]
    assert terminal_meta["outcome_axes"]["review"]["status"] == "degraded"
    assert terminal_meta["reason_code"] == "acceptance_degraded"
    assert terminal_meta["review_projection"] == child_review_projection

    failed_drive = tmp_path / "state" / "headless_tasks" / "childfail" / "data"
    (failed_drive / "task_results").mkdir(parents=True)
    write_task_result(failed_drive, "childfail", STATUS_FAILED, result="boom")
    sent = []
    worker = SimpleNamespace(busy_task_id="childfail")
    ctx = SimpleNamespace(
        DRIVE_ROOT=tmp_path,
        RUNNING={
            "childfail": {
                "task": {
                    "id": "childfail",
                    "chat_id": 1,
                    "drive_root": str(failed_drive),
                    "delegation_role": "subagent",
                    "role": "reviewer",
                    "root_task_id": "root123",
                    "parent_task_id": "parent123",
                    "task_constraint": {"mode": "local_readonly_subagent", "allow_enable": False},
                }
            }
        },
        WORKERS={8: worker},
        bridge=SimpleNamespace(push_log=lambda _payload: None),
        send_with_budget=lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)),
        persist_queue_snapshot=lambda reason="": None,
    )

    ev_module._handle_task_done({"task_id": "childfail", "worker_id": 8, "task_type": "task"}, ctx)

    assert load_task_result(tmp_path, "childfail")["status"] == STATUS_FAILED
    assert sent and "failed" in sent[-1][1]
    assert sent[-1][2]["progress_meta"]["subagent_event"] == "failed"


def test_handle_schedule_task_fails_fast_when_worker_pool_unavailable(tmp_path, monkeypatch):
    """When the worker pool is empty (e.g. disabled after a crash storm), a
    schedule must NOT be left as a 'scheduled' ghost — it gets a terminal
    workers_unavailable result so the parent can act."""
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_FAILED

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *args, **kwargs: None)
    sent = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {}  # pool disabled / not available

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            sent.append((chat_id, text, kwargs))

        def enqueue_task(self, task):
            raise AssertionError("must not enqueue when worker pool is unavailable")

        def persist_queue_snapshot(self, reason=""):
            pass

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "ghost1",
            "objective": "Work with no workers",
            "expected_output": "Nothing",
            "depth": 1,
            "root_task_id": "rootX",
            "delegation_role": "subagent",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "ghost1" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "ghost1" / "data"),
        },
        FakeCtx(),
    )

    data = json.loads((tmp_path / "task_results" / "ghost1.json").read_text(encoding="utf-8"))
    assert data["status"] == STATUS_FAILED
    assert data.get("reason_code") == "workers_unavailable"
