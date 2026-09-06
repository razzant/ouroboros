"""Immune-system hardening invariants (WS-E, v6.26.0).

Covers: triad anti-refusal coverage contract, durable advisory-override
visibility, dispatcher-level worktree-mutation invalidation, and the
`find` removal from the SAFE shell whitelist.
"""

import json
import pathlib

import pytest

from ouroboros.triad_review import parse_model_review_results


def _actor(text: str, model: str = "m1") -> dict:
    return {"model": model, "text": text, "tokens_in": 1, "tokens_out": 1}


class TestTriadRefusalContract:
    def test_bare_empty_array_only_response_is_responded(self):
        parsed = parse_model_review_results({"results": [_actor("[]")]})
        assert parsed.actor_records[0].status == "responded"

    def test_fenced_empty_array_only_response_is_responded(self):
        parsed = parse_model_review_results({"results": [_actor("```json\n[]\n```")]})
        assert parsed.actor_records[0].status == "responded"

    def test_empty_array_with_sentinel_is_responded(self):
        parsed = parse_model_review_results(
            {"results": [_actor("[]\nNO_FINDINGS")]}
        )
        assert parsed.actor_records[0].status == "responded"

    def test_prose_around_the_clean_payload_is_parse_failure(self):
        """Tightened: a preamble cannot be told from a refusal by structure, so
        `I cannot review this diff. []\\nNO_FINDINGS` used to enter quorum as
        clean. The contract asks for the array alone, so prose is refused —
        including a benign-looking preamble."""
        for text in (
            "Everything checks out.\n[]\nNO_FINDINGS",
            "I cannot review this diff. []\nNO_FINDINGS",
        ):
            parsed = parse_model_review_results({"results": [_actor(text)]})
            assert parsed.actor_records[0].status == "parse_failure", text

    def test_empty_array_inside_refusal_prose_is_parse_failure(self):
        refusal = (
            "I'm sorry, but I cannot review this diff because it is too large "
            "and may contain content I cannot process. [] Please try again."
        )
        parsed = parse_model_review_results({"results": [_actor(refusal)]})
        assert parsed.actor_records[0].status == "parse_failure"
        assert not parsed.quorum_met

    def test_refusal_actor_does_not_count_toward_quorum(self):
        good = '[{"item":"code_quality","verdict":"PASS","severity":"advisory","reason":"ok"}]'
        refusal = "I cannot comply with this request. []"
        parsed = parse_model_review_results(
            {"results": [_actor(good, "m1"), _actor(refusal, "m2"), _actor(refusal, "m3")]}
        )
        assert len(parsed.responsive_models) == 1
        assert not parsed.quorum_met

    def test_non_empty_findings_array_still_responded(self):
        text = '[{"item":"security","verdict":"FAIL","severity":"critical","reason":"bad"}]'
        parsed = parse_model_review_results({"results": [_actor(text)]})
        assert parsed.actor_records[0].status == "responded"
        assert parsed.findings and parsed.findings[0]["verdict"] == "FAIL"


class TestAdvisoryOverrideVisibility:
    def test_override_leaves_durable_trace(self, tmp_path, monkeypatch):
        from ouroboros.tools import review as review_mod

        class Ctx:
            drive_root = tmp_path
            task_id = "t1"
            _last_review_block_reason = "critical_findings"

            def drive_logs(self):
                logs = tmp_path / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                return logs

        review_mod._record_advisory_override(Ctx(), "⚠️ REVIEW_BLOCKED: example")

        events = (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8")
        assert "review_advisory_override" in events

        overrides = json.loads(
            (tmp_path / "state" / "advisory_overrides.json").read_text(encoding="utf-8")
        )
        assert overrides["count"] == 1
        assert overrides["recent"][0]["block_reason"] == "critical_findings"

    def test_review_status_payload_surfaces_overrides(self):
        from ouroboros.review_evidence import build_review_status_payload

        projection = {
            "effective_status": "none",
            "effective_hash": None,
            "stale_from_edit": False,
            "stale_from_edit_ts": None,
            "stale_reason": None,
            "filters": {},
            "runs": [],
            "attempts": [],
            "selected_attempt": None,
            "open_obligations": [],
            "open_debts": [],
            "repo_commit_ready": False,
            "retry_anchor": None,
            "advisory_overrides": {"count": 3, "recent": [{"ts": "t", "block_reason": "x"}]},
            "guidance_run": None,
            "state": None,
            "current_hash": "",
            "effective_is_fresh": False,
            "latest_run": None,
            "matching_run": None,
        }
        payload = build_review_status_payload(projection, next_step="n/a")
        assert payload["advisory_overrides_count"] == 3
        assert payload["advisory_overrides_recent"]


class TestCentralWorktreeInvalidation:
    def test_mutating_tool_invalidates_only_on_real_change(self, tmp_path, monkeypatch):
        from ouroboros.tools.registry import ToolRegistry

        registry = ToolRegistry.__new__(ToolRegistry)
        registry._entries = {}
        registry._capability_omissions = []

        class _Ctx:
            repo_dir = tmp_path
            drive_root = tmp_path / "data"

        registry._ctx = _Ctx()

        calls = {"invalidate": 0}
        # Call 1 sees " M a.py" (differs from before="") → invalidate;
        # call 2 sees " M a.py" (equals before) → no invalidate.
        snapshots = iter([" M a.py", " M a.py"])
        monkeypatch.setattr(
            ToolRegistry, "_worktree_status_snapshot", lambda self: next(snapshots)
        )

        def _fake_invalidate(drive_root, **kwargs):
            calls["invalidate"] += 1

        import ouroboros.review_state as review_state

        monkeypatch.setattr(
            review_state, "invalidate_advisory_after_mutation", _fake_invalidate
        )

        # Change observed (before="", after=" M a.py") → invalidate.
        registry._invalidate_advisory_if_worktree_changed("run_command", "")
        # No change (before==after) → no invalidate.
        registry._invalidate_advisory_if_worktree_changed("run_command", " M a.py")
        assert calls["invalidate"] == 1

    @pytest.mark.serial
    def test_worktree_snapshot_reads_real_git_status(self, tmp_path):
        # The test above stubs _worktree_status_snapshot, so it cannot see the
        # snapshot itself failing. This one drives the real method against a real
        # repo: run_cmd used to reject the timeout= kwarg it is called with, so
        # every snapshot degraded to "<status-unavailable>", before == after, and
        # _invalidate_advisory_if_worktree_changed became a permanent no-op.
        import subprocess

        from ouroboros.tools.registry import ToolRegistry

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)

        registry = ToolRegistry.__new__(ToolRegistry)

        class _Ctx:
            repo_dir = tmp_path
            drive_root = tmp_path / "data"

        registry._ctx = _Ctx()

        before = registry._worktree_status_snapshot()
        assert before != "<status-unavailable>"

        (tmp_path / "written_by_tool.py").write_text("x = 1\n", encoding="utf-8")
        after = registry._worktree_status_snapshot()

        assert after != "<status-unavailable>"
        assert "written_by_tool.py" in after
        assert after != before, "a mutated worktree must not look unchanged"

    @pytest.mark.serial
    def test_run_cmd_accepts_and_honors_timeout(self, tmp_path):
        import subprocess
        import sys

        from ouroboros.utils import run_cmd

        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
        assert run_cmd(["git", "status", "--porcelain"], cwd=tmp_path, timeout=20) == ""
        assert run_cmd(["git", "status", "--porcelain"], cwd=tmp_path) == ""

        with pytest.raises(subprocess.TimeoutExpired):
            run_cmd([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.2)

    def test_mutating_tools_are_flagged(self):
        from ouroboros.tools import git_pr, services, shell

        flagged = {
            entry.name
            for mod in (shell, services, git_pr)
            for entry in mod.get_tools()
            if entry.mutates_worktree
        }
        for name in ("run_command", "run_script",
                     "start_service", "stop_service", "cherry_pick_pr_commits",
                     "stage_pr_merge", "stage_adaptations"):
            assert name in flagged, f"{name} must be mutates_worktree-flagged"


class TestFindNotSafe:
    def test_find_routes_through_llm_safety(self):
        from ouroboros.safety import SAFE_SHELL_COMMANDS

        assert "find" not in SAFE_SHELL_COMMANDS
        # Read-only staples stay whitelisted.
        assert "grep" in SAFE_SHELL_COMMANDS
        assert "ls" in SAFE_SHELL_COMMANDS


class TestGitInfo:
    @pytest.mark.parametrize(
        "failed_lookup,failure,expected",
        [
            (None, None, ("ouroboros", "abc123")),
            (0, "timeout", ("", "abc123")),
            (1, "timeout", ("ouroboros", "")),
            (0, "exit", ("", "abc123")),
            (1, "exit", ("ouroboros", "")),
        ],
    )
    def test_bounded_lookups_preserve_partial_results(
        self, tmp_path, monkeypatch, failed_lookup, failure, expected,
    ):
        import subprocess

        from ouroboros.utils import get_git_info

        calls = []
        monkeypatch.setenv("LC_ALL", "ru_RU.UTF-8")
        monkeypatch.setenv("LANG", "ru_RU.UTF-8")

        def fake_run(cmd, **kwargs):
            index = len(calls)
            calls.append(cmd)
            assert kwargs["cwd"] == str(tmp_path)
            assert kwargs["timeout"] == 2
            assert kwargs["env"]["LC_ALL"] == kwargs["env"]["LANG"] == "C"
            if index == failed_lookup:
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(cmd, 2)
                return subprocess.CompletedProcess(cmd, 1, "", "git failed")
            return subprocess.CompletedProcess(cmd, 0, ("ouroboros\n", "abc123\n")[index], "")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert get_git_info(tmp_path) == expected
        assert calls == [
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            ["git", "rev-parse", "HEAD"],
        ]


class TestScopeChecklistFailClosed:
    def test_missing_checklist_raises(self, monkeypatch):
        import pytest

        import ouroboros.tools.scope_review as scope_review

        monkeypatch.setattr(scope_review, "load_checklist_section", lambda *_a, **_k: "")
        with pytest.raises(RuntimeError, match="fail-closed"):
            scope_review._build_scope_prompt(pathlib.Path("."), "msg")


class TestStatusSetSSOT:
    def test_headless_final_statuses_mirror_settled_ssot(self):
        # headless cannot import task_status at module level (import cycle);
        # this pin keeps its mirrored literal equal to the SSOT.
        from ouroboros.headless import _FINAL_STATUSES
        from ouroboros.task_status import SETTLED_STATUSES

        assert _FINAL_STATUSES == SETTLED_STATUSES

    def test_headless_readonly_subagent_mode_mirrors_capability_ssot(self):
        # Same anti-drift pin: the literal headless uses to detect a read-only subagent
        # (it cannot import tool_capabilities at module level) must equal the SSOT constant
        # that the registry/supervisor enforce.
        from ouroboros.headless import _LOCAL_READONLY_SUBAGENT_MODE
        from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE

        assert _LOCAL_READONLY_SUBAGENT_MODE == LOCAL_READONLY_SUBAGENT_MODE
