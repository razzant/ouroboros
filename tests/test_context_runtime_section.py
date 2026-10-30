"""The runtime section and the user content the context builder emits.

Split verbatim out of ``tests/test_context.py`` by theme. This module owns the
legacy metadata that must not rewrite the user's text, the light-mode rule and filesystem affordances the runtime section states,
the workspace rules that preserve system review/commit authority, the host routing
manifest and manual contract, the improvement backlog digest, and the runtime_env
block.
"""

from __future__ import annotations

import json
import os

import pytest

from ouroboros.context import build_runtime_section, build_user_content

from tests._context_shared import _make_health_env


@pytest.mark.parametrize("host_fact", [None, {}, {"kind": "other"}, {"kind": "call_open"}])
@pytest.mark.parametrize("attachment", [False, True])
def test_call_open_is_labelled_before_attachment_and_presence_rendering(monkeypatch, host_fact, attachment):
    from copy import deepcopy
    from ouroboros import context

    text = "  A live call opened.\n"
    task = {"text": text, "metadata": {"host_fact": host_fact}}
    original = deepcopy(task)
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}
    monkeypatch.setattr(context, "_build_attachment_image_blocks", lambda _task: [image] if attachment else [])
    seen = []
    monkeypatch.setattr("ouroboros.presence_context.frame_presence_user_content",
                        lambda source, content: seen.append((source, content)) or content)

    expected = ("[Host fact: a live call opened; not the owner's words]\n\n" + text
                if host_fact == {"kind": "call_open"} else text)
    expected = [{"type": "text", "text": expected}, image] if attachment else expected
    assert build_user_content(task) == expected
    assert seen == [(task, expected)]
    assert task == original  # attribution does not rewrite the source or choose different memory


@pytest.mark.parametrize("root_only", [False, True])
def test_schedule_digest_distinguishes_unreadable_from_empty(tmp_path, root_only):
    from types import SimpleNamespace
    from ouroboros.context import _scheduled_tasks_digest

    env = SimpleNamespace(drive_root=tmp_path) if root_only else SimpleNamespace(drive_path=lambda rel: tmp_path / rel)
    path = tmp_path / "state" / "scheduled_tasks.json"
    path.parent.mkdir()
    path.write_text("{unreadable", encoding="utf-8")
    unreadable = _scheduled_tasks_digest(env)
    assert unreadable["unavailable"] and "standing" not in unreadable
    assert unreadable["pointer"] == "manage_schedules(action='list')"
    path.write_text('{"tasks": []}', encoding="utf-8")
    empty = _scheduled_tasks_digest(env)
    assert "unavailable" not in empty
    assert empty["standing"] == empty["attention"] == empty["one_shots"] == []


@pytest.mark.parametrize("enforcement", ["blocking", "advisory"])
def test_legacy_force_plan_metadata_leaves_owner_content_unchanged(monkeypatch, enforcement):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    text = "Fix the marketplace retry flow."
    assert build_user_content({"text": text, "metadata": {"force_plan": True, "force_plan_source": "swarm"}}) == text


@pytest.mark.parametrize("obsolete_marker", [False, True])
def test_managed_legacy_force_plan_leaves_exact_owner_text(obsolete_marker):
    text = "  Investigate the alternatives.\nAsk about the actual constraints.\n"
    assert build_user_content({"text": text, "managed_task": obsolete_marker,
                              "metadata": {"force_plan": True, "force_plan_source": "swarm"}}) == text



def test_runtime_section_includes_light_runtime_mode_rule(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])

    assert payload["runtime_mode"] == "light"
    assert "forbids Ouroboros repo mutation" in payload["runtime_mode_rule"]
    assert "user_files" in payload["runtime_mode_rule"]
    assert "artifact_store" in payload["runtime_mode_rule"]
    assert "explicit scoped skill-payload work/repair" in payload["runtime_mode_rule"]
    assert "runtime_data/uploads" in payload["runtime_mode_rule"]


@pytest.mark.parametrize("trigger", ["once", "cron"])
@pytest.mark.parametrize("previous", [None, {"task_id": "prior", "state": "pending_observation"},
                                     {"task_id": "prior", "state": "completed", "quiet": True, "line": ""}])
def test_runtime_keeps_the_admitted_occurrences_due_time_after_schedule_advances(tmp_path, trigger, previous):
    env = _make_health_env(tmp_path)
    due = "2026-09-29T00:30:00+00:00"
    claimed = "2026-09-30T03:00:00+00:00"
    (tmp_path / "state" / "scheduled_tasks.json").write_text(json.dumps({"tasks": [{
        "id": "follow-up", "enabled": trigger == "cron", "name": "Review results",
        "trigger": {"type": trigger, "run_at": due, "expr": "30 0 * * *"},
        "next_run_at": "2026-10-01T00:30:00+00:00",
        **({"completed_at": claimed} if trigger == "once" else {}),
    }]}), encoding="utf-8")
    task = {"id": "late-run", "type": "task", "metadata": {"schedule_occurrence": {
        "schedule_id": "follow-up", "token": "occurrence-token", "due_at": due, "claimed_at": claimed,
        **({"previous": previous} if previous is not None else {}),
    }}}
    runtime = json.loads(build_runtime_section(env, task).split("\n\n", 1)[1])
    assert runtime["task"]["schedule_occurrence"] == {
        "schedule_id": "follow-up", "due_at": due, "claimed_at": claimed,
        **({"previous": previous} if previous is not None else {}),
    }
    if trigger == "once":
        assert not any(runtime["scheduled_tasks"][group] for group in ("standing", "attention", "one_shots"))
    else:
        assert runtime["scheduled_tasks"]["standing"][0]["next_run_at"] != due


def test_runtime_does_not_invent_a_due_time_for_legacy_or_ordinary_work(tmp_path):
    env = _make_health_env(tmp_path)
    task = {"id": "legacy", "metadata": {"schedule_occurrence": {"schedule_id": "old", "token": "t"}}}
    legacy = json.loads(build_runtime_section(env, task).split("\n\n", 1)[1])
    assert legacy["task"]["schedule_occurrence"] == {"schedule_id": "old", "due_at": None, "claimed_at": None}
    ordinary = json.loads(build_runtime_section(env, {"id": "ordinary"}).split("\n\n", 1)[1])
    assert "schedule_occurrence" not in ordinary["task"]


def test_runtime_section_includes_filesystem_affordances_with_ctx(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext

    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "light")
    ctx = ToolContext(repo_dir=tmp_path / "repo", drive_root=tmp_path)

    section = build_runtime_section(env, {"id": "task-1", "type": "task"}, ctx=ctx)
    payload = json.loads(section.split("\n\n", 1)[1])
    fs = payload["capabilities"]["filesystem"]

    assert fs["profile"] == "self_modification"
    assert "runtime_data" in fs["searchable_roots"]
    assert "task_drive" in fs["searchable_roots"]  # read⇒search closure (TZ-1 E)
    assert "task_drive" in fs["allowed_shell_cwd_roots"]
    assert "status" in fs["git_readonly_subcommands"]
    assert "active_workspace" in fs["light_gated_roots"]


def test_runtime_section_external_workspace_includes_user_files_shell_affordance(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolContext

    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    drive = tmp_path / "data"
    repo = tmp_path / "repo"
    workspace = tmp_path / "workspace"
    drive.mkdir()
    repo.mkdir(exist_ok=True)
    workspace.mkdir(exist_ok=True)
    ctx = ToolContext(
        repo_dir=repo,
        drive_root=drive,
        workspace_root=workspace,
        workspace_mode="external",
    )

    section = build_runtime_section(env, {"id": "task-1", "type": "task"}, ctx=ctx)
    payload = json.loads(section.split("\n\n", 1)[1])
    fs = payload["capabilities"]["filesystem"]

    assert fs["profile"] == "external_workspace_task"
    assert "user_files" in fs["allowed_shell_cwd_roots"]


def test_runtime_section_workspace_rule_preserves_system_review_commit_authority(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    section = build_runtime_section(
        env,
        {
            "id": "task-1",
            "type": "task",
            "workspace_root": str(workspace),
            "workspace_mode": "external",
            "memory_mode": "forked",
        },
    )
    rule = json.loads(section.split("\n\n", 1)[1])["active_workspace"]["rule"]

    assert "default to the active workspace" in rule
    assert "explicit typed root/cwd" in rule
    assert "self-review/commit tools remain available" in rule
    assert "self-review/commit tools are unavailable" not in rule


def test_runtime_section_omits_light_rule_for_advanced(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])

    assert payload["runtime_mode"] == "advanced"
    assert "runtime_mode_rule" not in payload


def test_runtime_section_includes_non_workspace_memory_boundary(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    section = build_runtime_section(
        env,
        {
            "id": "task-1",
            "type": "task",
            "memory_mode": "forked",
            "drive_root": str(tmp_path / "child"),
            "child_drive_root": str(tmp_path / "child"),
            "budget_drive_root": str(tmp_path / "data"),
        },
    )
    payload = json.loads(section.split("\n\n", 1)[1])
    assert payload["task"]["memory_mode"] == "forked"
    assert payload["task"]["child_drive_root"].endswith("child")
    assert payload["task"]["budget_drive_root"].endswith("data")


def test_runtime_section_exposes_host_routing_manifest_and_manual_contract(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    task = {
        "id": "decision-1",
        "type": "task",
        "metadata": {
            "current_chat": {
                "chat_id": 1,
                "running_tasks": [],
                "addressable_root_tasks": [{"task_id": "pending-1", "status": "pending"}],
            },
            "main_routing_manifest": {
                "projects": [{"project_id": "racer", "name": "Racer"}],
                "root_tasks": [{"task_id": "pending-1", "status": "pending"}],
            },
            "routing_contract": {
                "source_lane": "main",
                "on_uncertain_or_invalid_target": "needs_manual_target",
                "manual_options": [{"task_id": "pending-1"}],
            },
        },
    }

    payload = json.loads(build_runtime_section(env, task).split("\n\n", 1)[1])

    assert payload["current_chat"]["addressable_root_tasks"][0]["task_id"] == "pending-1"
    assert payload["main_routing_manifest"]["projects"][0]["project_id"] == "racer"
    assert payload["routing_contract"]["on_uncertain_or_invalid_target"] == "needs_manual_target"


def test_runtime_section_offers_the_rooms_own_continuation_hint(tmp_path, monkeypatch):
    """A project room's routing manifest has to REACH the decision turn.

    While only the pointer row travelled, a room saw exactly ONE continuation
    candidate; when a child had stamped that pointer the room could not name its
    own interrupted root at all and promoted again, minting a duplicate root.
    """
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    room = {
        "final_results": [{"task_id": "racer-old", "status": "completed"}],
        "active_roots": [{"task_id": "racer-live", "status": "running",
                          "cancel_state": "pending"}],
        "omissions": {"final_results": 3, "children": 2, "active_roots": 0},
    }
    task = {
        "id": "decision-room",
        "type": "task",
        "metadata": {
            "current_chat": {
                "chat_id": 7,
                "running_tasks": [],
                "addressable_root_tasks": [{"task_id": "racer-live", "status": "running"}],
            },
            "project_routing_manifest": room,
            "project_last_task_result": {"task_id": "racer-old", "status": "completed"},
        },
    }

    payload = json.loads(build_runtime_section(env, task).split("\n\n", 1)[1])
    assert payload["project_routing_manifest"] == room
    assert payload["project_last_task_result"]["task_id"] == "racer-old"

    # The quiet direction: a lane with nothing to offer states no empty hint.
    task["metadata"]["project_routing_manifest"] = {}
    quiet = json.loads(build_runtime_section(env, task).split("\n\n", 1)[1])
    assert "project_routing_manifest" not in quiet
    assert quiet["project_last_task_result"]["task_id"] == "racer-old"


def test_improvement_backlog_digest_is_actor_scoped(tmp_path):
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    class FakeEnv:
        def drive_path(self, p):
            return tmp_path / p

        def repo_path(self, p):
            return tmp_path / "repo" / p

        @property
        def repo_dir(self):
            return tmp_path / "repo"

        @property
        def drive_root(self):
            return tmp_path

    (tmp_path / "repo" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "repo" / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (tmp_path / "repo" / "README.md").write_text("README", encoding="utf-8")
    (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
    (tmp_path / "repo" / "docs" / "CHECKLISTS.md").write_text('Checklist', encoding="utf-8")
    (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
    (tmp_path / "memory" / "identity.md").write_text("I am Ouroboros", encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
    (tmp_path / "memory" / "knowledge" / "improvement-backlog.md").write_text(
        "# Improvement Backlog\n\n### ibl-1\n- status: open\n- created_at: 2026-04-14T09:00:00+00:00\n- source: execution_reflection\n- category: process\n- task_id: task-1\n- requires_plan_review: yes\n- fingerprint: fp-1\n- summary: Reduce recurring task friction around REVIEW_BLOCKED\n",
        encoding="utf-8",
    )

    ordinary = ({"id": "main", "type": "task", "_is_direct_chat": True},
                {"id": "project", "type": "task", "project_id": "p1"},
                {"id": "managed", "type": "task"},
                {"id": "child", "type": "task", "delegation_role": "subagent"})
    for task in ordinary:
        task["text"] = "hello"
        messages, _ = build_llm_messages(
            env=FakeEnv(), memory=Memory(drive_root=tmp_path), task=task,
        )
        assert "## Improvement Backlog" not in messages[0]["content"][-1]["text"]

    messages, _ = build_llm_messages(
        env=FakeEnv(), memory=Memory(drive_root=tmp_path),
        task={"id": "evolution", "type": "evolution", "text": "improve"})
    dynamic_text = messages[0]["content"][-1]["text"]
    assert "## Improvement Backlog" in dynamic_text
    assert "Reduce recurring task friction around REVIEW_BLOCKED" in dynamic_text
    # A deep self-review reads its own memory packet (deep_self_review), not this section.
    messages, _ = build_llm_messages(
        env=FakeEnv(), memory=Memory(drive_root=tmp_path),
        task={"id": "deep", "type": "deep_self_review", "text": "review"})
    assert "## Improvement Backlog" not in messages[0]["content"][-1]["text"]


class TestRuntimeEnvSection:
    """build_runtime_section: runtime_env carries presentation + platform, and
    the per-message owner_client fact renders beside it (is_desktop retired)."""

    def _make_env(self, tmp_path):
        class FakeEnv:
            repo_dir = tmp_path / "repo"
            drive_root = tmp_path

            def drive_path(self, p):
                return tmp_path / p

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "state" / "state.json").write_text(
            '{"spent_usd": 0}', encoding="utf-8"
        )
        return FakeEnv()

    def test_runtime_env_presentation_absent_means_web(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        monkeypatch.delenv("OUROBOROS_PRESENTATION", raising=False)
        env = self._make_env(tmp_path)
        section = build_runtime_section(env, {"id": "t1", "type": "task"})
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "runtime_env" in data
        assert "platform" in data["runtime_env"]
        assert isinstance(data["runtime_env"]["platform"], str)
        assert data["runtime_env"]["presentation"] == "web"
        # The dead is_desktop flag is retired; presentation replaced it.
        assert "is_desktop" not in data["runtime_env"]

    def test_runtime_env_presentation_from_launcher_export(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        for value in ("desktop_window", "browser_fallback"):
            monkeypatch.setenv("OUROBOROS_PRESENTATION", value)
            env = self._make_env(tmp_path)
            section = build_runtime_section(env, {"id": "t2", "type": "task"})
            data = json.loads(section.split("## Runtime context\n\n", 1)[1])
            assert data["runtime_env"]["presentation"] == value

    def test_owner_client_rendered_from_metadata(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        monkeypatch.delenv("OUROBOROS_PRESENTATION", raising=False)
        env = self._make_env(tmp_path)
        fact = {"pywebview": True, "ua": "TestShell/1.0", "viewport": {"w": 1200, "h": 800}}
        section = build_runtime_section(
            env, {"id": "t3", "type": "task", "metadata": {"client_surface": fact}}
        )
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert data["owner_client"] == fact
        assert "SENT" in data["owner_client_note"]

    def test_owner_client_absent_is_a_gap_not_a_default(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        section = build_runtime_section(env, {"id": "t4", "type": "task"})
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "owner_client" not in data
        assert "owner_client_note" not in data

    def test_ui_language_fact_names_the_tag_and_whether_it_was_chosen(self, tmp_path, monkeypatch):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "")
        data = json.loads(build_runtime_section(env, {"id": "t5", "type": "task"}).split("## Runtime context\n\n", 1)[1])
        assert data["ui_language"] == {"tag": "en", "chosen": False}
        monkeypatch.setenv("OUROBOROS_UI_LANGUAGE", "ru")
        data = json.loads(build_runtime_section(env, {"id": "t6", "type": "task"}).split("## Runtime context\n\n", 1)[1])
        assert data["ui_language"] == {"tag": "ru", "chosen": True}

    def test_owner_client_channel_fact_stamped_by_external_admission(self, tmp_path):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        # /api/tasks and CLI STAMP the channel fact at admission; the renderer
        # reads only the producer-assembled fact.
        section = build_runtime_section(
            env, {"id": "t5", "type": "task", "metadata": {"client_surface": {"channel": "cli"}}}
        )
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert data["owner_client"] == {"channel": "cli"}

    def test_owner_client_never_inferred_from_metadata_source(self, tmp_path):
        from ouroboros.context import build_runtime_section

        env = self._make_env(tmp_path)
        # metadata.source is OVERLOADED (scheduler writes scheduled_task /
        # skill_scheduled_task): the renderer must never dress it up as an
        # owner surface — no producer stamp, no fact (codex scope round 2 N1).
        for source in ("cli", "scheduled_task", "skill_scheduled_task", "web"):
            section = build_runtime_section(
                env, {"id": "t6", "type": "task", "metadata": {"source": source}}
            )
            data = json.loads(section.split("## Runtime context\n\n", 1)[1])
            assert "owner_client" not in data, f"source={source!r} must not render"
        # Internal producers use top-level task["source"], never rendered.
        section = build_runtime_section(
            env, {"id": "t7", "type": "task", "source": "promote_chat_to_task"}
        )
        data = json.loads(section.split("## Runtime context\n\n", 1)[1])
        assert "owner_client" not in data

    def test_the_block_is_compact_json_with_the_same_values(self, tmp_path):
        """No indentation or separator spaces (20.10.2026): the same values, about a fifth fewer tokens."""
        from ouroboros.context import build_runtime_section

        section = build_runtime_section(self._make_env(tmp_path), {"id": "t8", "type": "task"})
        body = section.split("## Runtime context\n\n", 1)[1]
        assert body == json.dumps(json.loads(body), ensure_ascii=False, separators=(",", ":"))


def _delegation_data_root(tmp_path, monkeypatch):
    root = tmp_path / "delegation_data_root"
    (root / "state").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("ouroboros.config.DATA_DIR", root)
    return root


def _delegation_fact(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: "advanced")
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])
    return payload["capabilities"]


def test_delegation_fact_carries_historical_identities_and_profile_readers(tmp_path, monkeypatch):
    root = _delegation_data_root(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "claudexor=opus-5:high")
    (root / "state" / "reviewer_slot_last_execution.json").write_text(json.dumps({
        "triad_1": {
            "ts": "2026-08-18T01:02:03+00:00",
            "surface": "triad",
            "status": "ok",
            "requested": {"profile_id": "requested-review-profile"},
            "effective": {
                "route": "agent_session:claudexor",
                "model": "opus-5",
                "profile_id": "applied-review-profile",
            },
        },
        "triad_2": {
            "ts": "2026-08-18T01:02:04+00:00",
            "surface": "triad",
            "status": "error",
            # B1 typed facts: a dated window carries reset_at, an undated one
            # only the code — both must surface independently.
            "failure_code": "subscription_window_exhausted",
            "reset_at": "2026-08-18T09:20:00+00:00",
        },
    }), encoding="utf-8")
    (root / "state" / "subagent_last_delegation.json").write_text(json.dumps({
        "ts": "2026-08-18T02:00:00+00:00",
        "route": "claudexor",
        "requested_model": "opus-5",
        "applied_model": "claude-opus-5",
        "requested_profile": "requested-delegate-profile",
        "applied_profile": "applied-delegate-profile",
        "selected_subagent_id": "builder",
        "run_id": "run-1",
    }), encoding="utf-8")

    capabilities = _delegation_fact(tmp_path, monkeypatch)
    delegation = capabilities["delegation"]

    assert "configured_route" not in delegation
    rows = {row["slot"]: row for row in delegation["reviewer_slots_last"]}
    assert rows["triad_1"] == {"slot": "triad_1", "outcome": "ok", "observed": "2026-08-18T01:02:03+00:00"}
    assert rows["triad_2"]["outcome"] == "error"
    (last,) = delegation["subagents_last_executions"]
    assert last["selected_subagent_id"] == "claudexor=opus-5" and last["run_id"] == "run-1"
    assert last["observed"] == "2026-08-18T02:00:00+00:00"
    assert "not live health" in delegation["note"]
    assert not any(key in json.dumps(delegation) for key in
                   ("requested_profile", "applied_profile", "failure_code", "reset_at"))
    # The exact readers retain the account and failure evidence instead of copying it inline.
    records = {}
    for reader in delegation["readers"]:
        assert reader["tool"] == "read_file" and reader["arguments"]["root"] == "runtime_data"
        path = reader["arguments"]["path"]
        records[path] = json.loads((root / path).read_text(encoding="utf-8"))
    reviews = records["state/reviewer_slot_last_execution.json"]
    assert reviews["triad_1"]["requested"]["profile_id"] == "requested-review-profile"
    assert reviews["triad_1"]["effective"]["profile_id"] == "applied-review-profile"
    assert reviews["triad_2"]["failure_code"] == "subscription_window_exhausted"
    assert reviews["triad_2"]["reset_at"] == "2026-08-18T09:20:00+00:00"
    saved = records["state/subagent_last_delegation.json"]
    assert saved["requested_profile"] == "requested-delegate-profile"
    assert saved["applied_profile"] == "applied-delegate-profile" and saved["applied_model"] == "claude-opus-5"


def test_delegation_fact_addresses_undated_window_code_without_inventing_reset(tmp_path, monkeypatch):
    root = _delegation_data_root(tmp_path, monkeypatch)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)
    (root / "state" / "reviewer_slot_last_execution.json").write_text(json.dumps({
        "scope": {
            "ts": "2026-08-18T03:00:00+00:00",
            "status": "error",
            "failure_code": "credential_pool_exhausted",
        },
    }), encoding="utf-8")

    delegation = _delegation_fact(tmp_path, monkeypatch)["delegation"]

    (row,) = delegation["reviewer_slots_last"]
    assert "failure_code" not in row and "reset_at" not in row
    assert row["outcome"] == "error"
    reader = delegation["readers"][0]
    stored = json.loads((root / reader["arguments"]["path"]).read_text(encoding="utf-8"))["scope"]
    assert stored["failure_code"] == "credential_pool_exhausted" and "reset_at" not in stored


def test_delegation_fact_absent_files_mean_absent_observations_not_health(tmp_path, monkeypatch):
    _delegation_data_root(tmp_path, monkeypatch)
    monkeypatch.delenv("OUROBOROS_SUBAGENT_HARNESS", raising=False)

    capabilities = _delegation_fact(tmp_path, monkeypatch)

    assert "delegation" not in capabilities


def test_delegation_fact_failure_never_drops_capability_digest(tmp_path, monkeypatch):
    _delegation_data_root(tmp_path, monkeypatch)

    def _boom():
        raise RuntimeError("reader exploded")

    monkeypatch.setattr(
        "ouroboros.reviewer_slot_config.reviewer_slot_last_executions", _boom)

    capabilities = _delegation_fact(tmp_path, monkeypatch)

    assert "delegation" not in capabilities
    # The surrounding digest survives intact.
    assert "allow_mutative_subagents" in capabilities
    assert "write_surfaces" in capabilities


def test_runtime_names_its_capture_instant_and_how_this_run_learns_the_time(tmp_path):
    """#1320: the runtime block is captured once; it says so instead of calling itself "now"."""
    env = _make_health_env(tmp_path)
    main = json.loads(build_runtime_section(env, {"id": "t1", "type": "task"}, captured_at="2027-01-15T12:00:00+00:00")
                      .split("\n\n", 1)[1])
    assert "utc_now" not in main and main["context_captured_at"] == "2027-01-15T12:00:00+00:00"
    assert "not the current time" in main["clock_note"] and "host clock line" in main["clock_note"]
    child = json.loads(build_runtime_section(env, {"id": "t2", "type": "task", "delegation_role": "subagent"})
                       .split("\n\n", 1)[1])
    assert "does not advance during this run" in child["clock_note"] and child["context_captured_at"]


def test_captured_recent_and_drive_sections_carry_one_capture_label(tmp_path, monkeypatch):
    """Recent*/Drive state are labelled with the core's single capture instant, below their headings,
    and are not refreshed: two projections of one core carry byte-identical labels."""
    from ouroboros import context as context_module
    from ouroboros.context import build_llm_messages
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setattr(context_module, "utc_now_iso", lambda: "2027-01-15T12:00:00+00:00")
    env, memory = _make_env_and_memory(tmp_path)
    logs = memory.drive_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "chat.jsonl").write_text(json.dumps({
        "ts": "2027-01-15T11:00:00+00:00", "direction": "in", "chat_id": 1, "text": "hello"}) + "\n",
        encoding="utf-8")
    messages, _info = build_llm_messages(env=env, memory=memory, task={
        "id": "t-labels", "type": "task", "text": "hi", "_is_direct_chat": True, "chat_id": 1, "metadata": {}})
    dynamic = messages[0]["content"][-1]["text"]
    label = "_Snapshot captured at 2027-01-15T12:00:00+00:00 when this context was built; not refreshed during this run._"
    assert "## Drive state\n" + label in dynamic
    # The chat row is the memory view's (the open conversation): no capture-labelled chat tail.
    assert "## Recent chat" not in dynamic and "] hello" in dynamic.split("## This room (Main)", 1)[1]
    runtime = json.loads(dynamic.split("## Runtime context\n\n", 1)[1].split("\n\n## ", 1)[0])
    assert runtime["context_captured_at"] == "2027-01-15T12:00:00+00:00"
    headings = [line for line in dynamic.splitlines() if line.startswith(("## Recent ", "## Drive state"))]
    assert headings and all(dynamic.split(heading + "\n", 1)[1].startswith(label) for heading in headings)
    assert "## Runtime context\n" + label not in dynamic  # labels are for the captured snapshots only


class TestStorageFact:
    """The last finished storage count reaches the Runtime block as one compact fact (Commit 5 of
    the storage plan): present with a count, absent without, never zero, and no threshold, tier or
    WARNING stands in for the mind's judgment."""

    def _env(self, tmp_path):
        class FakeEnv:
            repo_dir = tmp_path / "repo"
            drive_root = tmp_path

            def drive_path(self, p):
                return tmp_path / p

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
        return FakeEnv()

    def _runtime(self, env):
        from ouroboros.context import build_runtime_section

        section = build_runtime_section(env, {"id": "t1", "type": "task"})
        return section, json.loads(section.split("## Runtime context\n\n", 1)[1])

    def test_absent_until_the_owner_counts_then_the_count_with_growth(self, tmp_path, monkeypatch):
        from types import SimpleNamespace

        from ouroboros import storage

        env = self._env(tmp_path)
        _section, data = self._runtime(env)
        assert "storage" not in data, "no count yet: no fact, never zero"

        (tmp_path / "observability").mkdir()
        (tmp_path / "observability" / "a.bin").write_bytes(b"x" * 4096)
        table = [{"id": "model_calls", "group": "records", "roots": [(str(tmp_path), ("observability",))]}]
        first = storage.measure(tmp_path, places=table, sleep=lambda _s: None)
        storage.save_count(tmp_path, first)
        section, data = self._runtime(env)
        fact = data["storage"]
        assert set(fact) == {"counted_at", "bytes", "unreadable", "basis", "previous_at", "growth_bytes", "free_bytes", "volume_total_bytes"}
        assert fact["counted_at"] == first["finished_at"] and fact["bytes"] == first["bytes"] > 0
        assert fact["unreadable"] == 0 and "growth_flag" not in fact
        assert fact["previous_at"] is None and fact["growth_bytes"] is None
        assert isinstance(fact["free_bytes"], int) and isinstance(fact["volume_total_bytes"], int)
        for word in ("WARNING", "threshold", "tier", "HOT STORE"):
            assert word not in section

        (tmp_path / "observability" / "b.bin").write_bytes(b"y" * 8192)
        second = storage.measure(tmp_path, places=table, sleep=lambda _s: None)
        storage.save_count(tmp_path, second)
        _section, data = self._runtime(env)
        assert data["storage"]["previous_at"] == first["finished_at"]
        assert data["storage"]["growth_bytes"] == second["bytes"] - first["bytes"] > 0

        # A helper reads the installation's count; changing free space never needs another count.
        child_env = self._env(tmp_path / "child")
        child_env.budget_drive_root = tmp_path
        monkeypatch.setattr("shutil.disk_usage", lambda path: SimpleNamespace(free=123, total=1000))
        _section, child_data = self._runtime(child_env)
        assert child_data["storage"]["bytes"] == second["bytes"]
        assert child_data["storage"]["free_bytes"] == 123

        def unavailable(_path):
            raise OSError("volume not available")

        monkeypatch.setattr("shutil.disk_usage", unavailable)
        _section, child_data = self._runtime(child_env)
        assert child_data["storage"]["bytes"] == second["bytes"]
        assert child_data["storage"]["free_bytes"] is None
        assert child_data["storage"]["volume_total_bytes"] is None

    def test_an_unreadable_saved_count_is_no_fact(self, tmp_path):
        from ouroboros import storage

        env = self._env(tmp_path)
        (tmp_path / storage.MEASUREMENT_FILE).write_bytes(b"not json")
        _section, data = self._runtime(env)
        assert "storage" not in data

    @pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason="needs a folder this user cannot list")
    def test_unreadable_items_qualify_the_headline_and_the_growth(self, tmp_path):
        """A count that could not read everything reaches the mind with its unreadable total, and
        a growth against a count that had unreadable items carries the same flag the gauge shows."""
        import stat

        from ouroboros import storage

        env = self._env(tmp_path)
        calls = tmp_path / "observability" / "calls"
        calls.mkdir(parents=True)
        (calls / "record.json").write_bytes(b"r" * 4096)
        table = [{"id": "model_calls", "group": "records", "roots": [(str(tmp_path), ("observability",))]}]
        first = storage.measure(tmp_path, places=table, sleep=lambda _s: None)
        storage.save_count(tmp_path, first)
        calls.chmod(0)
        try:
            second = storage.measure(tmp_path, places=table, sleep=lambda _s: None)
            storage.save_count(tmp_path, second)
            _section, data = self._runtime(env)
        finally:
            calls.chmod(stat.S_IRWXU)
        fact = data["storage"]
        assert second["unreadable"] == 1 and fact["unreadable"] == 1 and fact["bytes"] == second["bytes"]
        assert fact["growth_bytes"] == second["bytes"] - first["bytes"] < 0 and fact["growth_flag"] == "unreadable"

    def test_a_malformed_previous_is_no_growth_and_never_breaks_the_section(self, tmp_path):
        from ouroboros import storage
        from ouroboros.utils import atomic_write_json

        env = self._env(tmp_path)
        (tmp_path / "observability").mkdir()
        (tmp_path / "observability" / "a.bin").write_bytes(b"x" * 4096)
        table = [{"id": "model_calls", "group": "records", "roots": [(str(tmp_path), ("observability",))]}]
        latest = storage.measure(tmp_path, places=table, sleep=lambda _s: None)
        atomic_write_json(tmp_path / storage.MEASUREMENT_FILE, {"_schema_version": storage.SCHEMA_VERSION, "latest": latest, "previous": {}})
        section, data = self._runtime(env)
        assert "## Runtime context" in section
        assert data["storage"]["bytes"] == latest["bytes"] and data["storage"]["previous_at"] is None
        assert data["storage"]["growth_bytes"] is None


class TestVitalsFact:
    """The body's own dated facts reach the Runtime block beside ``storage``: the last owner turns from
    the serving process's receipt ring, stall onsets without an observed end, the last busy minute as
    sums/counts/maxima, the runtime, the retained update, the boot row, one reader and a note. Every
    absent source omits its key; nothing here is a verdict or a percentage."""

    def _env(self, tmp_path, monkeypatch):
        from collections import deque

        from supervisor import active_activity

        class FakeEnv:
            repo_dir = tmp_path / "repo"
            drive_root = tmp_path

            def drive_path(self, p):
                return tmp_path / p

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0}', encoding="utf-8")
        monkeypatch.setattr(active_activity, "_DIRECT_TURNS", deque(maxlen=active_activity.DIRECT_TURNS_KEPT))
        monkeypatch.setattr("ouroboros.context_runtime_facts._JOURNAL_MODES", {})
        monkeypatch.delenv("OUROBOROS_IN_WORKER", raising=False)
        return FakeEnv()

    def _runtime(self, env, task=None):
        section = build_runtime_section(env, task or {"id": "t1", "type": "task", "_is_direct_chat": True},
                                        captured_at="2026-10-25T20:00:00+00:00")
        return section, json.loads(section.split("## Runtime context\n\n", 1)[1])

    @staticmethod
    def _journal(tmp_path, rows):
        path = tmp_path / "logs" / "supervisor.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_a_main_direct_turn_reads_its_bodys_dated_facts(self, tmp_path, monkeypatch):
        from supervisor.active_activity import record_direct_turn

        env = self._env(tmp_path, monkeypatch)
        self._journal(tmp_path, [
            {"ts": "2026-10-25T18:00:00+00:00", "type": "boot", "branch": "ouroboros", "sha": "abcdef1234567890"},
            {"ts": "2026-10-25T19:00:00+00:00", "type": "direct_turn_stall", "execution_id": "turn_a", "task_id": "t0",
             "phase": "context_building", "stood_sec": 30.0},
            {"ts": "2026-10-25T19:01:00+00:00", "type": "direct_turn_stall_end", "execution_id": "turn_a", "task_id": "t0",
             "phase": "context_building", "then": "context_built", "stood_sec": 61.0, "samples": 4,
             "top_frames": [{"frame": "ouroboros/memory_view.py:_live_rooms", "samples": 4}]},
            {"ts": "2026-10-25T19:30:00+00:00", "type": "host_duty_stall", "duty": "custody", "running_sec": 120.0},
            {"ts": "2026-10-25T19:40:00+00:00", "type": "direct_turn_stall", "execution_id": "turn_b", "task_id": "t9",
             "phase": "started", "stood_sec": 15.0},
            {"ts": "2026-10-25T19:59:00+00:00", "type": "supervisor_load", "window_sec": 60.0, "max_event_lag_sec": 0.4,
             "handlers": {"llm_usage": {"count": 3, "total_sec": 0.1, "max_sec": 0.05}},
             "store_locks": {"dialog": {"write": {"count": 12, "total_sec": 0.42, "max_sec": 0.2,
                                                   "max_site": "supervisor.message_bus:send:10"}}},
             "queue_lock": {"count": 5, "hold_total_sec": 0.3, "hold_max_sec": 0.2, "hold_max_site": "supervisor.queue:assign:10"}},
        ])
        monkeypatch.setattr("supervisor.state.append_jsonl", lambda path, row, *a, **k: True)
        monkeypatch.setattr("supervisor.active_activity.utc_now_iso", lambda: "2026-10-25T19:55:00+00:00")  # the receipt's stamp
        record_direct_turn({"type": "direct_turn", "execution_id": "turn_a", "chat_id": 1, "task_id": "t0",
                            "accepted_at": "2026-10-25T18:59:00+00:00", "owner_wait_sec": 91.2, "wait_before_entry_sec": 0.8,
                            "handler_wait_sec": 90.4, "phases": {"accepted": 0.1, "context_building": 61.0},
                            "slowest_phase": "context_building", "slowest_phase_sec": 61.0, "dispatch": "observed_send",
                            "meter": {"current.read": 2, "current.read.sec": 0.01, "current.read.max_sec": 0.008,
                                      "chronicle.locked": 7}, "census": {"direct_turns_at_entry": 0}})
        section, data = self._runtime(env)
        vitals = data["vitals"]
        # The fact's instant is the capture instant printed beside it: not written twice.
        assert "observed_at" not in vitals and data["context_captured_at"] == "2026-10-25T20:00:00+00:00"
        from ouroboros.context_runtime_facts import vitals_fact

        stale = vitals_fact(tmp_path, observed_at="2026-10-25T19:00:00+00:00", captured_at="2026-10-25T20:00:00+00:00")
        assert stale["observed_at"] == "2026-10-25T19:00:00+00:00"  # a snapshot older than the capture names its time
        assert all(isinstance(value, str) for key, value in vitals.items()
                   if key not in ("owner_turns", "stalls, no end in window"))
        assert vitals["owner_turns"] == [
            "19:55 chat 1 waited 91.2 s, slowest context_building 61.0 s; txns current 2r · chronicle 7 locked; top memory_view.py:_live_rooms"]
        turn = vitals["owner_turns"][0]
        assert "observed_send" not in turn and "not written" not in turn  # a seen send is the default basis
        assert vitals["stalls, no end in window"] == ["duty custody since 19:30", "turn t9 started since 19:40"]
        assert vitals["busy_minute"] == "19:59: lag ≤0.4 s; holds dialog w12/0.4 s (message_bus:send:10); queue 5/0.3 s"
        assert vitals["runtime"].startswith("python ") and ", sqlite " in vitals["runtime"]
        assert vitals["boot"] == "18:00 ouroboros@abcdef123456"
        assert vitals["note"] == "No verdict here, the weighing is mine; clocks UTC; rows: query_data(kind='supervisor_log', op='jsonl')"
        assert "last_update" not in vitals
        for word in ("WARNING", "CRITICAL", "% busy", "current\"", "threshold"):
            assert word not in json.dumps(vitals), word
        assert "## Supervisor" not in section

    def test_absent_sources_omit_their_keys_and_a_pooled_worker_reads_the_journals_receipts(self, tmp_path, monkeypatch):
        env = self._env(tmp_path, monkeypatch)
        _section, data = self._runtime(env)
        vitals = data["vitals"]
        assert set(vitals) <= {"owner_turns_note", "runtime", "wal_gate", "note"}
        assert {"owner_turns_note", "runtime", "note"} <= set(vitals)
        assert vitals["owner_turns_note"] == "none since this process started"
        monkeypatch.setenv("OUROBOROS_IN_WORKER", "1")  # a deep self-review's pooled worker: no ring of its own
        _section, data = self._runtime(env)
        assert data["vitals"]["owner_turns_note"] == "none in the journal's tail" and "owner_turns" not in data["vitals"]
        self._journal(tmp_path, [{"ts": f"2026-10-25T19:5{n}:00+00:00", "type": "direct_turn", "execution_id": f"turn_{n}",
                                  "chat_id": 1, "owner_wait_sec": float(n), "dispatch": "observed_send"} for n in range(3)])
        _section, data = self._runtime(env)
        assert data["vitals"]["owner_turns"] == ["19:51 chat 1 waited 1.0 s", "19:52 chat 1 waited 2.0 s"]
        assert "owner_turns_note" not in data["vitals"]

    def test_the_journal_per_store_and_the_wal_gate_only_for_a_rollback_store(self, tmp_path, monkeypatch):
        from ouroboros import context_runtime_facts
        from ouroboros.task_results import write_task_result

        env = self._env(tmp_path, monkeypatch)
        write_task_result(tmp_path, "seed", "running")  # a present store, in this test runtime's journal mode
        _section, data = self._runtime(env)
        vitals = data["vitals"]
        journal = vitals["runtime"].split(", journal ", 1)[1]
        assert "current=wal" in journal or "current=rollback" in journal
        assert ("wal_gate" in vitals) == ("=rollback" in journal)
        if "wal_gate" in vitals:
            assert vitals["wal_gate"] == context_runtime_facts.WAL_GATE and "reinstall" not in vitals["wal_gate"]

    def test_an_unreadable_journal_or_a_failing_source_never_breaks_the_section(self, tmp_path, monkeypatch):
        env = self._env(tmp_path, monkeypatch)
        (tmp_path / "logs").mkdir()
        (tmp_path / "logs" / "supervisor.jsonl").write_bytes(b"\xff{not json\n")
        _section, data = self._runtime(env)
        assert "boot" not in data["vitals"] and "stalls" not in data["vitals"]

        def broken(*_a, **_k):
            raise RuntimeError("ring unavailable")

        monkeypatch.setattr("supervisor.active_activity.direct_turns", broken)
        section, data = self._runtime(env)
        assert "vitals" not in data and '"context_captured_at"' in section

    def test_vitals_line_renders_the_same_facts_in_one_line(self):
        from ouroboros.context_runtime_facts import vitals_line

        assert vitals_line(None) == ""
        line = vitals_line({"observed_at": "T", "owner_turns": ["turn one", "turn two"],
                            "stalls, no end in window": ["a stall"], "busy_minute": "M: lag",
                            "runtime": "python 3.10.22, sqlite 3.51.3", "wal_gate": "",
                            "last_update": "applied abc: first request before B → after A", "boot": "B o@s", "note": "n"})
        assert line == ("body as of T; owner turns turn one | turn two; stalls, no end in window a stall; "
                        "busy minute M: lag; runtime python 3.10.22, sqlite 3.51.3; "
                        "last update applied abc: first request before B → after A; boot B o@s")
        assert vitals_line({"runtime": "r", "note": "n"}) == "body; runtime r"  # the capture instant stands beside it
