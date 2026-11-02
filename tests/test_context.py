"""The health invariants ouroboros.context builds, and where they must appear.

This module owns the sent-history changes fact (no provider cache verdict), the remote context overflow,
the rest of the invariant coverage, and the rule that
the invariants come first in both the dynamic and the background-consciousness context.

The runtime section, the advisory review status, the memory/consolidation sections and
the drive-state projection were split verbatim into
``tests/test_context_runtime_section.py``, ``tests/test_context_advisory_review.py``,
``tests/test_context_memory.py`` and ``tests/test_context_drive_state.py``; the health
environment builder they share lives in ``tests/_context_shared.py``.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import dialogue_store
from ouroboros.context import build_health_invariants, build_runtime_section

from tests import _dialogue as dialogue_test
from tests._context_shared import _make_health_env



class TestSentHistoryHealth:
    def _make_env(self, tmp_path, events_lines):
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

        (tmp_path / "state").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "docs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
        (tmp_path / "repo" / "web").mkdir(parents=True, exist_ok=True)
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "1.2.3"}', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text('version-1.2.3', encoding="utf-8")
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text('# Ouroboros v1.2.3', encoding="utf-8")
        (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text('# Dev', encoding="utf-8")
        (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
        (tmp_path / "memory" / "identity.md").write_text('x' * 300, encoding="utf-8")
        (tmp_path / "memory" / "scratchpad.md").write_text('x' * 300, encoding="utf-8")
        (tmp_path / "logs" / "events.jsonl").write_text("\n".join(events_lines) + "\n", encoding="utf-8")
        return FakeEnv()

    def test_health_names_who_changed_sent_history_under_the_running_code(self, tmp_path):
        from ouroboros import transcript_prefix

        env = self._make_env(tmp_path, [])
        assert "SENT HISTORY CHANGES" not in build_health_invariants(env)
        for task_id, origin in (("t1", "superseded_skill_snapshot"), ("t2", "superseded_skill_snapshot"),
                                ("t2", "compaction")):
            transcript_prefix.count_history_change(
                {}, {"round": 3, "kind": "rewritten", "sanctioned_by": origin}, drive_root=tmp_path, task_id=task_id)
        line = next(row for row in build_health_invariants(env).splitlines() if "SENT HISTORY CHANGES" in row)
        assert line.startswith("- INFO: SENT HISTORY CHANGES under the running code ("), line
        assert ", first at " in line and "the round facts name each change" in line
        first = line.split(", first at ", 1)[1].split(")", 1)[0]
        assert len(first) == 17 and first.endswith("Z"), first  # minutes, labelled UTC
        assert "superseded_skill_snapshot 2 in 2 tasks; compaction 1 in 1 task" in line

    def test_another_codes_record_is_not_this_codes_history(self, tmp_path):
        from ouroboros import transcript_prefix
        from ouroboros.utils import atomic_write_json

        env = self._make_env(tmp_path, [])
        path = transcript_prefix.history_changes_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"code": "an-older-commit", "first": "2026-10-20T17:00:00Z", "origins": {
            "superseded_skill_snapshot": {"count": 193, "tasks": ["t1"]}}})
        assert "SENT HISTORY CHANGES" not in build_health_invariants(env)
        transcript_prefix.count_history_change(
            {}, {"round": 2, "kind": "shrunk", "sanctioned_by": "compaction"}, drive_root=tmp_path, task_id="t9")
        result = build_health_invariants(env)
        assert "compaction 1 in 1 task" in result and "superseded_skill_snapshot" not in result

    def test_a_tree_without_git_names_its_version(self, tmp_path, monkeypatch):
        from ouroboros import transcript_prefix, utils

        monkeypatch.setattr(utils, "get_git_info", lambda _root: ("", ""))
        transcript_prefix._running_code.cache_clear()
        try:
            env = self._make_env(tmp_path, [])
            transcript_prefix.count_history_change(
                {}, {"round": 1, "kind": "rewritten", "sanctioned_by": "compaction"}, drive_root=tmp_path, task_id="t1")
            assert "under the running code (version " in build_health_invariants(env)
        finally:
            transcript_prefix._running_code.cache_clear()

    def test_the_task_list_is_capped_and_says_so(self, tmp_path):
        from ouroboros import transcript_prefix

        env = self._make_env(tmp_path, [])
        for index in range(40):
            transcript_prefix.count_history_change(
                {}, {"round": 1, "kind": "rewritten", "sanctioned_by": None}, drive_root=tmp_path, task_id=f"t{index}")
        assert "origin not recorded 40 in 32+ tasks" in build_health_invariants(env)

    @pytest.mark.parametrize("reports", [
        [None] * 6, [0] * 6, [600] * 6, [None] * 10 + [600] * 5,
    ], ids=["unknown", "measured_zero", "measured_hit", "mixed"])
    def test_nullable_adapter_cache_reaches_durable_and_live_rounds(self, tmp_path, reports):
        """The real adapter uses a present null key for an unreported cache.

        Omitted-key fixtures alone missed the producer turning that null into
        zero. Both round events preserve the measurement; health draws no share
        or verdict from them.
        """
        from queue import Queue
        from ouroboros.llm_claudexor import _usage
        from ouroboros.loop_llm_call import call_llm_with_retry

        env = self._make_env(tmp_path, [])
        events = Queue()
        accumulated = {}

        class LLM:
            def chat(self, **_kwargs):
                counters = {"input_tokens": 1000, "output_tokens": 10}
                if reported is not None:
                    counters["cached_input_tokens"] = reported
                usage, cost, final = _usage({"usage": counters})
                usage.update(provider="claudexor", resolved_model="claudexor/probe",
                             cost=cost, cost_final=final)
                return {"content": "Completed synthetic round."}, usage

        for index, reported in enumerate(reports, 1):
            message, _cost = call_llm_with_retry(
                LLM(), [{"role": "user", "content": "Check the report."}],
                "claudexor/probe", None, "medium", 1, tmp_path / "logs",
                "nullable-cache", index, events, accumulated,
            )
            assert message["content"] == "Completed synthetic round."
        rows = [json.loads(line) for line in (tmp_path / "logs/events.jsonl").read_text().splitlines()
                if line.strip()]
        durable = [row["cached_tokens"] for row in rows if row.get("type") == "llm_round"]
        live = []
        while not events.empty():
            event = events.get_nowait()
            if event.get("type") == "log_event" and event["data"].get("type") == "llm_round_finished":
                live.append(event["data"]["cached_tokens"])
        assert durable == reports
        assert live == reports
        assert "cache hit rate" not in build_health_invariants(env).lower()


def test_health_invariants_reports_remote_context_overflow(tmp_path):
    env = _make_health_env(
        tmp_path,
        [json.dumps({"type": "remote_context_overflow", "model": "provider/model"})],
    )

    result = build_health_invariants(env)

    assert "REMOTE CONTEXT OVERFLOW" in result
    assert "provider/model x1" in result


class TestAdditionalHealthInvariantCoverage:
    def test_version_desync_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.4"', encoding="utf-8")

        result = build_health_invariants(env)
        assert "VERSION DESYNC" in result
        assert "pyproject.toml=1.2.4" in result

    def test_web_package_version_desync_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "1.2.4"}', encoding="utf-8")

        result = build_health_invariants(env)
        assert "VERSION DESYNC" in result
        assert "web/package.json=1.2.4" in result

    def test_rc_pep440_pyproject_does_not_warn(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "VERSION").write_text("4.50.0-rc.2", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "4.50.0rc2"', encoding="utf-8")
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "4.50.0-rc.2"}', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text(
            "[![Version 4.50.0-rc.2](https://img.shields.io/badge/version-4.50.0--rc.2-green.svg)](VERSION)",
            encoding="utf-8",
        )
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v4.50.0-rc.2",
            encoding="utf-8",
        )

        result = build_health_invariants(env)
        assert "VERSION DESYNC" not in result

    def test_rc_badge_url_mismatch_warns(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "repo" / "VERSION").write_text("4.50.0-rc.2", encoding="utf-8")
        (tmp_path / "repo" / "pyproject.toml").write_text('version = "4.50.0rc2"', encoding="utf-8")
        (tmp_path / "repo" / "web" / "package.json").write_text('{"version": "4.50.0-rc.2"}', encoding="utf-8")
        (tmp_path / "repo" / "README.md").write_text(
            "[![Version 4.50.0-rc.2](https://img.shields.io/badge/version-4.50.0-rc.2-green.svg)](VERSION)",
            encoding="utf-8",
        )
        (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text(
            "# Ouroboros v4.50.0-rc.2",
            encoding="utf-8",
        )

        result = build_health_invariants(env)
        assert "VERSION DESYNC" in result
        assert "README badge URL token" in result

    def test_duplicate_processing_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        (tmp_path / "logs" / "events.jsonl").write_text(
            json.dumps({
                "type": "owner_message_injected",
                "text": "same message",
                "task_id": "task-a",
            }) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "logs" / "supervisor.jsonl").write_text(
            json.dumps({
                "event_type": "owner_message_injected",
                "text": "same message",
                "task_id": "task-b",
            }) + "\n",
            encoding="utf-8",
        )

        result = build_health_invariants(env)
        assert "DUPLICATE PROCESSING" in result
        assert "task-a" in result
        assert "task-b" in result

    def test_provider_and_overflow_warnings(self, tmp_path):
        env = _make_health_env(
            tmp_path,
            events_lines=[
                json.dumps({"type": "llm_api_error", "model": "openai/gpt-5.5"}),
                json.dumps({"type": "local_context_overflow", "model": "local/qwen"}),
            ],
        )

        result = build_health_invariants(env)
        assert "PROVIDER/ROUTING ERRORS" in result
        assert "openai/gpt-5.5 x1" in result
        assert "LOCAL CONTEXT OVERFLOW" in result
        assert "local/qwen x1" in result

    def test_rescue_snapshot_warning(self, tmp_path):
        env = _make_health_env(tmp_path)
        rescue_dir = tmp_path / "archive" / "rescue" / "2026-04-14-test"
        rescue_dir.mkdir(parents=True, exist_ok=True)
        (rescue_dir / "rescue_meta.json").write_text("{}", encoding="utf-8")
        (rescue_dir / "changes.diff").write_text("diff", encoding="utf-8")

        result = build_health_invariants(env)
        assert "RESCUE SNAPSHOT AVAILABLE" in result
        assert "2026-04-14-test" in result


def test_health_invariants_come_first_in_dynamic_context(tmp_path):
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
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    (tmp_path / "repo" / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (tmp_path / "repo" / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (tmp_path / "repo" / "README.md").write_text("README", encoding="utf-8")
    (tmp_path / "repo" / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "docs" / "DEVELOPMENT.md").write_text(
        "### File Size Budgets\n| Path | Budget chars |\n|------|--------------|\n| memory/identity.md | 1000 |\n",
        encoding="utf-8",
    )
    (tmp_path / "repo" / "docs" / "CHECKLISTS.md").write_text("Checklist", encoding="utf-8")
    (tmp_path / "repo" / "VERSION").write_text("1.2.3", encoding="utf-8")
    (tmp_path / "repo" / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (tmp_path / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (tmp_path / "memory" / "identity.md").write_text("x" * 950, encoding="utf-8")
    (tmp_path / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")

    messages, _cap_info = build_llm_messages(
        env=FakeEnv(),
        memory=Memory(drive_root=tmp_path),
        task={"id": "task-a", "type": "task", "text": "hello"},
    )

    dynamic_text = messages[0]["content"][-1]["text"]
    # Knowledge and my rooms lead the changing block; the runtime sections follow, health first.
    assert dynamic_text.startswith("## Shared understanding")
    assert dynamic_text.index("## This room (Main)") < dynamic_text.index("## Health Invariants")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Scratchpad")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Drive state")
    # #1731: the root keeps its contract discipline, and no root context orders a result review
    # before finalizing: under Auto the review is the author's own typed nomination.
    assert "## Task Contract Discipline" in dynamic_text and "state your success criteria early" in dynamic_text
    assert "task_acceptance_review" not in dynamic_text.split("## Task Contract Discipline", 1)[1].split("\n## ", 1)[0]


def test_health_invariants_come_first_in_a_consciousness_wake_context(tmp_path):
    """A wake-up is an ordinary Main turn: the same builder, the same section order."""
    from ouroboros.context import build_llm_messages
    from ouroboros.memory import Memory

    repo_dir = tmp_path / "repo"
    drive_root = tmp_path / "drive"
    (repo_dir / "prompts").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (drive_root / "memory" / "knowledge").mkdir(parents=True, exist_ok=True)
    (drive_root / "logs").mkdir(parents=True, exist_ok=True)
    (drive_root / "state").mkdir(parents=True, exist_ok=True)

    (repo_dir / "prompts" / "SYSTEM.md").write_text("System prompt", encoding="utf-8")
    (repo_dir / "BIBLE.md").write_text("Bible", encoding="utf-8")
    (repo_dir / "VERSION").write_text("1.2.3", encoding="utf-8")
    (repo_dir / "pyproject.toml").write_text('version = "1.2.3"', encoding="utf-8")
    (repo_dir / "README.md").write_text("README", encoding="utf-8")
    (repo_dir / "docs" / "ARCHITECTURE.md").write_text("# Ouroboros v1.2.3", encoding="utf-8")
    (repo_dir / "docs" / "DEVELOPMENT.md").write_text(
        "### File Size Budgets\n| Path | Budget chars |\n|------|--------------|\n| memory/identity.md | 1000 |\n",
        encoding="utf-8",
    )
    (drive_root / "state" / "state.json").write_text('{"spent_usd": 0, "budget_drift_alert": false}', encoding="utf-8")
    (drive_root / "memory" / "identity.md").write_text("x" * 950, encoding="utf-8")
    (drive_root / "memory" / "scratchpad.md").write_text("scratchpad", encoding="utf-8")
    (drive_root / "logs" / "chat.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "progress.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "tools.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "events.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "supervisor.jsonl").write_text("", encoding="utf-8")
    (drive_root / "logs" / "task_reflections.jsonl").write_text("", encoding="utf-8")

    class FakeEnv:
        def drive_path(self, p):
            return drive_root / p

        def repo_path(self, p):
            return repo_dir / p

        @property
        def repo_dir(self):
            return repo_dir

        @property
        def drive_root(self):
            return drive_root

    messages, _cap_info = build_llm_messages(
        env=FakeEnv(),
        memory=Memory(drive_root=drive_root, repo_dir=repo_dir),
        task={"id": "wake1", "type": "task", "text": "[Wake-up · heartbeat]", "_is_direct_chat": True,
              "metadata": {"initiator": "consciousness", "usage_category": "consciousness"}},
    )

    dynamic_text = messages[0]["content"][-1]["text"]
    assert dynamic_text.startswith("## Shared understanding")
    assert "\n## This room (Main) — head " in dynamic_text  # a wake reads Main as its room, as the Main turn does
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Scratchpad")
    assert dynamic_text.index("## Health Invariants") < dynamic_text.index("## Drive state")


def test_project_room_keeps_an_older_own_directive_beside_sibling_traffic(tmp_path, monkeypatch):
    """Sibling traffic cannot hide an older own-room directive: the room's open rows are its own."""
    from ouroboros.projects_registry import create_project
    from tests._memory_view_context import blocks, section
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "max")
    env, memory = _make_env_and_memory(tmp_path)
    root = memory.drive_root
    own = create_project(root, "own", name="Own")
    sibling = create_project(root, "sibling", name="Sibling")
    own_chat, sibling_chat = int(own["chat_id"]), int(sibling["chat_id"])
    decisive = "CLAUDEXOR_ONLY_AND_THREE_LEVEL_NESTING"
    # The directive is old history (an archived chain segment); the sibling noise is the newer live chain.
    dialogue_test.seed(root, archives={"chat_20260820T010000.jsonl": [
        {"chat_id": own_chat, "direction": "in", "ts": "2026-08-20T01:00:00+00:00", "text": decisive}]},
        live={"chat": [{"chat_id": sibling_chat, "direction": "in", "ts": "2026-08-21T00:00:00+00:00",
                        "text": f"noise-{i}"} for i in range(4500)]})
    dialogue_test.import_all(root)

    _a, _b, changing, _cap = blocks(env, memory, {"id": "own-task", "chat_id": own_chat})
    room = section(changing, f"## This room (Project Own [chat_id={own_chat}])")
    assert decisive in room
    assert "noise-" not in changing  # the sibling room is one line in ## Live rooms
    assert f"Project Sibling [chat_id={sibling_chat}]" in section(changing, "## Live rooms")
    assert str(root) not in room


def test_project_room_keeps_the_retention_proof_cross_room_directive_once(tmp_path):
    """The owner's words that started a Project stay visible once: in the open conversation of the
    room they were written in, else under ``Words that started this work``."""
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from tests._memory_view_context import blocks, section
    from tests.test_cache_optimization import _make_env_and_memory

    env, memory = _make_env_and_memory(tmp_path)
    root = memory.drive_root
    project = create_project(root, "cat-tower", name="Cat tower")
    project_chat = int(project["chat_id"])
    directive = "d" * 500 + " CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3"
    ref = build_owner_message_ref(chat_id=1, client_message_id="cat-origin", ts="2026-08-21T00:00:00Z", text=directive)
    bind_task_to_project(root, "cat-root", "cat-tower", project_chat, origin={"ref": ref, "text": directive})
    source_row = {**ref, "direction": "in", "text": directive}
    follow = {"chat_id": project_chat, "direction": "in", "ts": "2026-08-21T00:01:00Z", "text": "continue"}
    # The chain does not hold the owner's row (an older install never stored it): the binding kept the words.
    dialogue_store.append(root, "chat", follow)
    label = f"## This room (Project Cat tower [chat_id={project_chat}])"
    _a, _b, absent, _cap = blocks(env, memory, {"id": "cat-root", "chat_id": project_chat})
    room = section(absent, label)
    assert room.count("CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3") == 1  # retention-proof: the binding kept it
    assert "### Words that started this work (retention-proof)" in room

    origin = root / "memory" / "chronicle"
    assert origin.is_dir()  # the first capture activated the chronicle
    # Once the owner's row itself is an open row of this room, it is read there, once, and not under the heading.
    dialogue_store.append(root, "chat", source_row)
    _a, _b, present, _cap = blocks(env, memory, {"id": "cat-root", "chat_id": project_chat})
    room = section(present, label)
    assert room.count("CLAUDEXOR_ONLY; L1 MUST ASK L2 TO SPAWN L3") == 1  # an open row of this room
    assert "### Words that started this work" not in room


def test_chat_history_surfaces_malformed_gap_even_when_search_matches_nothing(tmp_path):
    from ouroboros.memory import Memory

    logs = tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / "chat.jsonl").write_bytes(
        b'{"direction":"in","text":"valid"}\n{"direction":"in","text":"decisive-tail"\n'
    )

    result = Memory(tmp_path).chat_history(count=20, search="absent-query")

    assert "no observed messages matching query" in result
    assert "completeness unknown" in result
    assert "jsonl_malformed" in result


def test_archive_only_chat_chain_is_complete_while_live_file_is_absent(tmp_path):
    from ouroboros.memory import Memory

    dialogue_test.seed(tmp_path, archives={"chat_20260820T010000.jsonl": [{"direction": "in", "text": "archive-only"}]})
    dialogue_test.import_all(tmp_path)

    entries, coverage = Memory(drive_root=tmp_path).read_chat_generations()

    assert [entry["text"] for entry in entries] == ["archive-only"]
    assert coverage["gaps"] == []


def test_runtime_section_carries_official_update_fact(tmp_path, monkeypatch):
    env = _make_health_env(tmp_path)
    # Patched WHERE IT IS USED: context.py binds the name at import, so patching the
    # defining module would leave this test asserting the real projection's own answer
    # and proving nothing about the injection.
    monkeypatch.setattr(
        "ouroboros.context.official_update_projection",
        lambda head: {"status": "update_available", "running": {"sha": head}, "letter": {"state": "ready"}},
    )
    section = build_runtime_section(env, {"id": "task-1", "type": "task"})
    payload = json.loads(section.split("\n\n", 1)[1])

    assert payload["official_update"]["status"] == "update_available"
    assert payload["official_update"]["letter"] == {"state": "ready"}
    assert payload["official_update"]["running"]["sha"] == payload["git_head"], "the fact reads THIS repo's HEAD"
