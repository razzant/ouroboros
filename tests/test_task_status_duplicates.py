"""Duplicate admission: which repeat request is rejected, and which one is a new task.

Split out of ``tests/test_task_status_flow.py`` by theme: the rejected-duplicate row the
scheduler writes, the subagent handoff fields the duplicate finder carries, and the
lineage and role distinctions that keep a genuinely different subagent admissible.
"""

import json
from types import SimpleNamespace


def test_handle_schedule_task_duplicate_writes_rejected_status(tmp_path, monkeypatch):
    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.task_results import STATUS_REJECTED_DUPLICATE

    captured_identity = {}

    def _duplicate(*args, **kwargs):
        captured_identity.update(kwargs.get("dedupe_identity") or {})
        return "orig111"

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", _duplicate)

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

    ev_module._handle_schedule_task(
        {
            "type": "schedule_subagent",
            "task_id": "dup222",
            "objective": "Do the thing",
            "expected_output": "Duplicate verdict",
            "context": "Model focus B",
            "depth": 1,
            "memory_mode": "forked",
            "parent_task_id": "parent111",
            "root_task_id": "root111",
            "drive_root": str(tmp_path / "state" / "headless_tasks" / "dup222" / "data"),
            "child_drive_root": str(tmp_path / "state" / "headless_tasks" / "dup222" / "data"),
            "budget_drive_root": str(tmp_path),
        },
        FakeCtx(),
    )

    path = tmp_path / "task_results" / "dup222.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == STATUS_REJECTED_DUPLICATE
    assert data["duplicate_of"] == "orig111"
    assert sent and "semantically similar" in sent[0][1]
    assert sent[0][2]["is_progress"] is True
    assert sent[0][2]["progress_meta"]["delegation_role"] == "subagent"
    assert sent[0][2]["progress_meta"]["parent_task_id"] == "parent111"
    assert sent[0][2]["progress_meta"]["status"] == STATUS_REJECTED_DUPLICATE
    assert captured_identity == {
        "delegation_role": "subagent",
        "task_id": "dup222",
        "parent_task_id": "parent111",
        "root_task_id": "root111",
        "budget_drive_root": str(tmp_path),
    }


def test_find_duplicate_task_includes_subagent_handoff_fields(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    captured = {}

    class FakeClient:
        def chat(self, messages, **kwargs):
            captured["prompt"] = messages[0]["content"]
            return {"content": "NONE"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Review shared surface",
        "same context",
        [
            {
                "id": "pending1",
                "description": "Review shared surface",
                "context": "same context",
                "expected_output": "Docs table",
                "constraints": "docs only",
                "role": "docs reviewer",
            }
        ],
        {},
        expected_output="Security table",
        constraints="security only",
        role="security reviewer",
    )

    assert result is None
    prompt = captured["prompt"]
    assert "Expected output:\nSecurity table" in prompt
    assert "Expected output:\nDocs table" in prompt
    assert "Constraints:\nsecurity only" in prompt
    assert "Constraints:\ndocs only" in prompt
    assert "Role:\nsecurity reviewer" in prompt
    assert "Role:\ndocs reviewer" in prompt


def test_find_duplicate_task_allows_distinct_subagent_roles(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "pending1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Run nested smoke slot",
        "",
        [
            {
                "id": "pending1",
                "description": "Run nested smoke slot",
                "expected_output": "Smoke handoff",
                "role": "l1-alpha-coordinator",
                "delegation_role": "subagent",
                "parent_task_id": "root1",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="Smoke handoff",
        role="l1-beta-coordinator",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "root1",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_find_duplicate_task_keeps_same_role_subagent_dedupe(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    class FakeClient:
        def chat(self, messages, **kwargs):
            return {"content": "pending1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Run nested smoke slot",
        "",
        [
            {
                "id": "pending1",
                "description": "Run nested smoke slot",
                "expected_output": "Smoke handoff",
                "role": "l1-alpha-coordinator",
                "delegation_role": "subagent",
                "parent_task_id": "root1",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="Smoke handoff",
        role="l1-alpha-coordinator",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "root1",
            "root_task_id": "root1",
        },
    )

    assert result == "pending1"


def test_find_duplicate_task_allows_distinct_subagent_parent_branches(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "pending1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "Run nested branch smoke slot",
        "",
        [
            {
                "id": "pending1",
                "description": "Run nested branch smoke slot",
                "expected_output": "Smoke handoff",
                "role": "shared-l2-role",
                "delegation_role": "subagent",
                "parent_task_id": "l1-alpha",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="Smoke handoff",
        role="shared-l2-role",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "l1-beta",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_find_duplicate_task_allows_subagent_against_running_root_ancestor(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "root1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "You are l1-alpha-coordinator; schedule L2 smoke agents",
        "",
        [],
        {
            "root1": {
                "task": {
                    "id": "root1",
                    "description": "Root coordinator: schedule l1-alpha, l1-beta, l1-gamma subagents",
                    "delegation_role": "root",
                    "parent_task_id": "",
                    "root_task_id": "root1",
                }
            }
        },
        expected_output="L1 handoff",
        role="l1-alpha-coordinator",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "root1",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []


def test_find_duplicate_task_allows_subagent_against_pending_parent_ancestor(monkeypatch):
    from supervisor import events as ev_module
    import ouroboros.config as config_module
    import ouroboros.llm as llm_module

    calls = []

    class FakeClient:
        def chat(self, messages, **kwargs):
            calls.append(messages[0]["content"])
            return {"content": "parent1"}, {}

    monkeypatch.setattr(config_module, "get_light_model", lambda: "test-light")
    monkeypatch.setattr(llm_module, "LLMClient", lambda: FakeClient())

    result = ev_module._find_duplicate_task(
        "You are l1-alpha-coordinator-l2-1; return a smoke handoff",
        "",
        [
            {
                "id": "parent1",
                "description": "You are l1-alpha-coordinator; schedule three L2 smoke subagents",
                "role": "l1-alpha-coordinator",
                "delegation_role": "subagent",
                "parent_task_id": "root1",
                "root_task_id": "root1",
            }
        ],
        {},
        expected_output="L2 handoff",
        role="l1-alpha-coordinator-l2-1",
        dedupe_identity={
            "delegation_role": "subagent",
            "parent_task_id": "parent1",
            "root_task_id": "root1",
        },
    )

    assert result is None
    assert calls == []
