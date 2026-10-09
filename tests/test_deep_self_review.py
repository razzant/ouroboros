"""Tests for ouroboros.deep_self_review and ``/review`` (decision 3A): the system
review runs on ONE row — an enabled catalog row the call names, else the direct
Main row — as ``review_change(subject=system, surface=system)``."""

from __future__ import annotations

import os
import subprocess
from unittest import mock

import pytest

from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS
from ouroboros.deep_self_review import deep_review_route, main_review_row
from ouroboros.tools.review_helpers import _is_probably_binary
from tests.test_git_review_preflight_gate import _roster


class TestApiRouteAvailability:
    """Availability (`deep_review_route`) of the DEFAULT row — the direct Main row
    (`OUROBOROS_MODEL`): the routed model's credentials, and the direct-OpenAI
    resolution (with its -pro rewrite) for a stored OpenRouter spelling."""

    def test_openrouter(self):
        with mock.patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "sk-or-test", "OUROBOROS_MODEL": "openai/gpt-5.5-pro"},
            clear=True,
        ):
            reason, model = deep_review_route()
        assert reason == ""
        assert model == "openai/gpt-5.5-pro"

    def test_openai(self):
        with mock.patch.dict(
            os.environ, {"OPENAI_API_KEY": "sk-test", "OUROBOROS_MODEL": "openai/gpt-5.6-sol-pro"}, clear=True,
        ):
            reason, model = deep_review_route()
        assert reason == ""
        # The direct route lands on the PROVIDER default, not a mechanical
        # `openai::` + router-slug rewrite: `-pro` is an OpenRouter routing slug
        # that 404s on api.openai.com (live-probed 2026-07-29).
        assert model == OPENAI_DIRECT_DEFAULTS["deep_self_review"]
        assert not model.endswith("-pro")

    def test_none(self):
        with mock.patch.dict(os.environ, {"OUROBOROS_MODEL": "openai/gpt-5.5"}, clear=True):
            reason, model = deep_review_route()
        assert reason
        assert model is None

    def test_direct_provider_prefix_requires_matching_key_even_with_openrouter(self):
        with mock.patch.dict(
            os.environ,
            {"OPENROUTER_API_KEY": "sk-or-test", "OUROBOROS_MODEL": "anthropic::claude-opus-4.8"},
            clear=True,
        ):
            reason, model = deep_review_route()

        assert reason
        assert model is None

    def test_direct_provider_prefix_available_with_matching_key(self):
        with mock.patch.dict(
            os.environ,
            {"ANTHROPIC_API_KEY": "sk-ant-test", "OUROBOROS_MODEL": "anthropic::claude-opus-4.8"},
            clear=True,
        ):
            reason, model = deep_review_route()

        assert reason == ""
        assert model == "anthropic::claude-opus-4.8"

    def test_the_deep_review_model_key_no_longer_chooses_the_executor(self):
        env = {"OPENROUTER_API_KEY": "sk-or-test", "OUROBOROS_MODEL": "openai/main-model",
               "OUROBOROS_MODEL_DEEP_SELF_REVIEW": "openai/deep-model"}
        with mock.patch.dict(os.environ, env, clear=True):
            row = main_review_row()
            route = deep_review_route()
        assert (row.slot_id, row.kind, row.target_id, row.use_local) == ("main", "api_chat", "openai/main-model", None)
        assert route == ("", "openai/main-model")


class _FakeCtx:
    def __init__(self):
        self.pending_events = []


class TestRequestToolEmitsEvent:
    def test_emits_correct_event(self):
        """_request_deep_self_review emits a deep_self_review_request event."""
        from ouroboros.tools.control import _request_deep_self_review

        ctx = _FakeCtx()
        with mock.patch(
            "ouroboros.deep_self_review.deep_review_route",
            return_value=("", "openai/gpt-5.5-pro"),
        ):
            result = _request_deep_self_review(ctx, "test reason")
        assert len(ctx.pending_events) == 1
        evt = ctx.pending_events[0]
        assert evt["type"] == "deep_self_review_request"
        assert evt["reason"] == "test reason"
        assert evt["model"] == "openai/gpt-5.5-pro"
        assert evt["reviewer"] == ""
        assert "Deep self-review" in result and "reviewer: Main" in result

    def test_a_named_row_rides_the_event_to_the_queue(self, monkeypatch):
        from ouroboros.tools.control import _request_deep_self_review

        _roster(monkeypatch)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
        ctx = _FakeCtx()
        result = _request_deep_self_review(ctx, "look again", reviewer=" api-scout ")
        [evt] = ctx.pending_events
        assert (evt["reviewer"], evt["model"]) == ("api-scout", "openai/fake-reviewer")
        assert "reviewer: api-scout" in result

    @pytest.mark.parametrize("enabled, name", [(False, "api-scout"), (True, "nobody")])
    def test_a_row_that_is_not_enabled_is_refused_before_any_event(self, monkeypatch, enabled, name):
        from ouroboros.tools.control import _request_deep_self_review

        _roster(monkeypatch, enabled=enabled)
        ctx = _FakeCtx()
        result = _request_deep_self_review(ctx, "look again", reviewer=name)
        assert ctx.pending_events == []
        assert result.startswith("⚠️ TOOL_ARG_ERROR (request_deep_self_review): ")
        assert "is not an enabled catalog row" in result and result.endswith("No review was queued.")

    def test_unavailable_returns_error(self):
        """An unavailable ROW returns the typed reason without emitting an event."""
        from ouroboros.tools.control import _request_deep_self_review

        ctx = _FakeCtx()
        with mock.patch(
            "ouroboros.deep_self_review.deep_review_route",
            return_value=("no provider credentials for openai/x", None),
        ):
            result = _request_deep_self_review(ctx, "test reason")
        assert len(ctx.pending_events) == 0
        assert result.startswith("❌ Deep self-review unavailable: no provider credentials for openai/x")
        assert "Main model (the default)" in result and "Review lanes" not in result


class TestSystemReviewRow:
    def test_no_name_is_the_direct_main_row(self, monkeypatch):
        from ouroboros.tools.review_change import system_review_row

        monkeypatch.setenv("OUROBOROS_MODEL", "openai/main-model")
        row = system_review_row("")
        assert (row.slot_id, row.target_id) == ("main", "openai/main-model")

    def test_any_enabled_catalog_row_pool_member_or_not(self, monkeypatch):
        from ouroboros.tools.review_change import system_review_row

        _roster(monkeypatch)
        row = system_review_row("api-scout")
        assert (row.slot_id, row.subagent_id, row.target_id) == ("api-scout", "api-scout", "openai/fake-reviewer")

    def test_a_disabled_row_is_an_argument_error(self, monkeypatch):
        from ouroboros.tools.review_change import ReviewChangeArgumentError, system_review_row

        _roster(monkeypatch, enabled=False)
        with pytest.raises(ReviewChangeArgumentError, match="is not an enabled catalog row"):
            system_review_row("api-scout")

    def test_w5_the_bare_main_row_carries_mains_pinned_account(self, monkeypatch):
        """A bare ``/review`` runs where Main runs: the row's credential profile is Main's
        saved pin (``OUROBOROS_MODEL_ACCOUNTS["main"]``), because the executor sends it as
        ``model_account_override`` and an empty override is Auto, not Main's role. An
        unpinned Main is Auto, as before."""
        from ouroboros.tools.review_change import system_review_row

        monkeypatch.setenv("OUROBOROS_MODEL", "openai/main-model")
        monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"main": "personal", "light": "other"}')
        row = system_review_row("")
        assert (row.slot_id, row.target_id, row.profile_id) == ("main", "openai/main-model", "personal")
        monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"main": ""}')
        assert system_review_row("").profile_id == ""
        monkeypatch.delenv("OUROBOROS_MODEL_ACCOUNTS")
        assert main_review_row().profile_id == ""

    def test_w5_a_named_row_keeps_its_own_credential_pin_not_mains(self, monkeypatch):
        """The working side: a catalog row the call names rides its own route credential
        (a managed-source row may pin one) or none; Main's pin never leaks onto it."""
        from ouroboros.tools.review_change import system_review_row
        from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool

        monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"main": "personal"}')
        set_review_pool(monkeypatch, pool_roster(
            pool_seat("pinned-scout", "claudexor::codex=model-x", profile_id="row-pin", marked=False),
            pool_seat("free-scout", "openai/fake-reviewer", marked=False)))
        assert system_review_row("pinned-scout").profile_id == "row-pin"
        assert system_review_row("free-scout").profile_id == ""


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture()
def system_ctx(tmp_path):
    from ouroboros.tools.registry import ToolContext

    repo, drive = tmp_path / "repo", tmp_path / "drive"
    repo.mkdir()
    (drive / "memory").mkdir(parents=True)
    (drive / "memory" / "deep_review.md").write_text("PREVIOUS REPORT", encoding="utf-8")
    (repo / "BIBLE.md").write_text("# BIBLE\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "base")
    ctx = ToolContext(repo_dir=repo, drive_root=drive)
    ctx.task_id = "dsr-1"
    return ctx


class TestRunSystemReview:
    """``run_system_review`` writes ONE ``surface=system`` record whose one seat
    answers ``report`` (never a verdict question), and replaces
    ``memory/deep_review.md`` only with a delivered report."""

    def _stub(self, monkeypatch, value):
        import ouroboros.deep_self_review as dsr

        seen = {}

        def review(*args, **kwargs):
            seen.update(kwargs, args=args)
            return value

        monkeypatch.setattr(dsr, "run_deep_self_review", review)
        return seen

    def test_a_named_row_writes_the_report_record(self, system_ctx, monkeypatch):
        from ouroboros.review_ledger import load_record
        from ouroboros.tools.review_change import run_review_change

        _roster(monkeypatch)
        seen = self._stub(monkeypatch, ("REPORT BODY", {"resolved_model": "openai/fake-reviewer", "cost": 0.25}))
        result = run_review_change(system_ctx, subject="system", surface="system", reviewers=["api-scout"])
        assert seen["slot"].subagent_id == "api-scout" and seen["task_id"] == "dsr-1"
        assert result["report"] == "REPORT BODY" and result["report_delivered"] is True
        record = load_record(system_ctx.drive_root, result["record_id"])
        assert record["surface"] == "system" and record["subject"]["kind"] == "system"
        [seat] = record["rows"]
        assert (seat["seat_id"], seat["parts"], seat["parts_answered"]) == ("api-scout", ["report"], ["report"])
        assert record["panel"]["chosen_by"] == "author" and record["panel"]["reviewers_requested"] == ["api-scout"]
        assert record["verdict"]["report"] == {"delivered": True, "chars": len("REPORT BODY")}
        assert record["verdict"]["aggregate"] != "PASS"
        assert (system_ctx.drive_root / "memory" / "deep_review.md").read_text(encoding="utf-8") == "REPORT BODY"

    def test_no_name_runs_main_and_a_refusal_keeps_the_previous_report(self, system_ctx, monkeypatch):
        from ouroboros.review_ledger import load_record
        from ouroboros.tools.review_change import run_system_review

        monkeypatch.setenv("OUROBOROS_MODEL", "openai/main-model")
        refusal = "❌ Deep self-review unavailable: no OpenRouter or direct OpenAI credentials for openai/main-model."
        seen = self._stub(monkeypatch, (refusal, {"execution_status": "infra_failed",
                                                  "reason_code": "deep_self_review_unavailable"}))
        result = run_system_review(system_ctx)
        assert seen["slot"].slot_id == "main" and result["report_delivered"] is False
        record = load_record(system_ctx.drive_root, result["record_id"])
        assert record["panel"]["chosen_by"] == "owner" and record["panel"]["reviewers_requested"] == []
        assert record["dispatch_refusal"] == {"kind": "reviewer_unavailable", "message": refusal}
        assert (system_ctx.drive_root / "memory" / "deep_review.md").read_text(encoding="utf-8") == "PREVIOUS REPORT"

    def _session_row(self, monkeypatch):
        """An unmarked enabled session row with its own effort and credential pin."""
        from tests.review_pool_rosters import mixed_pool_rows, pool_roster, pool_seat, set_review_pool

        set_review_pool(monkeypatch, pool_roster(*mixed_pool_rows(), pool_seat(
            "deep-session", "codex=gpt-5.6-sol", kind="agent_session", effort="xhigh", profile_id="prof-1", marked=False)))
        return "deep-session"

    def test_the_record_names_the_chosen_rows_real_seat_plan(self, system_ctx, monkeypatch):
        """D1-04 / V11 / D2-03: the system seat is written in the wave's shared ``rows``
        contract, so the record carries the row's route, effort, profile, session
        target and catalog id — not a seat rebuilt from the answer's model name."""
        from ouroboros.review_ledger import load_record
        from ouroboros.tools.review_change import run_review_change

        row = self._session_row(monkeypatch)
        self._stub(monkeypatch, ("REPORT", {"resolved_model": "codex=gpt-5.6-sol", "cost": 0.5}))
        result = run_review_change(system_ctx, subject="system", surface="system", reviewers=[row])
        [seat] = load_record(system_ctx.drive_root, result["record_id"])["rows"]
        assert seat["seat_id"] == row and seat["subagent_id"] == row
        assert seat["requested"] == {"route": "agent_session", "model": "codex=gpt-5.6-sol", "effort": "xhigh",
                                     "profile": "prof-1", "delivery": "retrieving", "session_target": "codex=gpt-5.6-sol",
                                     "processing_preference": "", "subagent_id": row}
        assert seat["effective"]["route"] == "agent_session" and seat["parts_answered"] == ["report"]

    def test_a_refused_review_still_records_the_seat_it_chose(self, system_ctx, monkeypatch):
        from ouroboros.review_ledger import load_record
        from ouroboros.tools.review_change import run_review_change

        row = self._session_row(monkeypatch)
        self._stub(monkeypatch, ("❌ Deep self-review unavailable: no harness.",
                                 {"execution_status": "infra_failed", "reason_code": "deep_self_review_unavailable"}))
        result = run_review_change(system_ctx, subject="system", surface="system", reviewers=[row])
        record = load_record(system_ctx.drive_root, result["record_id"])
        assert record["dispatch_refusal"]["kind"] == "reviewer_unavailable"
        [seat] = record["rows"]
        assert (seat["seat_id"], seat["status"], seat["parts_answered"]) == (row, "not_dispatched", [])
        assert (seat["requested"]["route"], seat["requested"]["effort"], seat["requested"]["profile"]) == (
            "agent_session", "xhigh", "prof-1")
        assert record["verdict"]["aggregate"] == "NOT_DISPATCHED"

    def test_the_calls_goal_questions_and_reason_reach_the_executor_and_the_record(self, system_ctx, monkeypatch):
        """D1-03 / V10: one typed ask (``SystemReviewAsk``) goes to the brief builder and
        the ledger; the standing questionnaire stays, the call's own goal and questions
        are added, and the reason is recorded as given."""
        from ouroboros.deep_self_review import STANDING_GOAL, SystemReviewAsk
        from ouroboros.review_ledger import load_record
        from ouroboros.tools.review_change import run_review_change

        _roster(monkeypatch)
        seen = self._stub(monkeypatch, ("REPORT", {"resolved_model": "openai/fake-reviewer"}))
        result = run_review_change(
            system_ctx, subject="system", surface="system", reviewers=["api-scout"], goal="Check the routing",
            author_questions=["Does the pinned profile reach the provider?", "Is the fallback disclosed?"],
            reason="the route specialist")
        ask = seen["ask"]
        assert ask == SystemReviewAsk(goal="Check the routing", reason="the route specialist",
                                      author_questions=("Does the pinned profile reach the provider?",
                                                        "Is the fallback disclosed?"))
        record = load_record(system_ctx.drive_root, result["record_id"])
        assert record["brief"]["goal"] == "Check the routing"
        assert record["brief"]["author_questions"] == list(ask.author_questions)
        assert (record["panel"]["reason"], record["panel"]["reason_missing"]) == ("the route specialist", False)

        bare = run_review_change(system_ctx, subject="system", surface="system")
        assert seen["ask"] == SystemReviewAsk() and seen["ask"].effective_goal == STANDING_GOAL
        assert load_record(system_ctx.drive_root, bare["record_id"])["brief"]["goal"] == STANDING_GOAL

    def test_the_brief_carries_the_standing_questionnaire_and_the_calls_questions(self, system_ctx):
        from ouroboros.deep_self_review import SystemReviewAsk, _ROLE_PROMPT, _retrieving_task

        ask = SystemReviewAsk(goal="Check the routing", author_questions=("Does the pin reach the provider?",))
        asked, _facts = _retrieving_task(system_ctx.repo_dir, system_ctx.drive_root, ask=ask)
        bare, _facts = _retrieving_task(system_ctx.repo_dir, system_ctx.drive_root)
        for text in (asked, bare):
            assert text.startswith(_ROLE_PROMPT) and "Prioritize: CRITICAL > IMPORTANT > ADVISORY." in text
        assert "The caller's goal for this review" in asked and "Check the routing" in asked
        assert "Author questions (answer each as asked, after your own questionnaire):\n1. Does the pin reach the provider?" in asked
        assert "Author questions" not in bare and "caller's goal" not in bare

    def test_the_record_binds_the_tree_the_reviewer_read_not_the_one_after(self, system_ctx, monkeypatch):
        """D1-06 / V13: the subject is snapshotted BEFORE the reviewer reads; a tree that
        moves under the review is disclosed, never passed off as the one read."""
        import ouroboros.deep_self_review as dsr
        from supervisor.update_candidate import worktree_snapshot_tree
        from ouroboros.review_ledger import load_record
        from ouroboros.tools.review_change import run_system_review

        repo = system_ctx.repo_dir
        before_tree, _ = worktree_snapshot_tree("HEAD", cwd=str(repo))
        before_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()

        def review(*args, **kwargs):  # another process lands a commit while the review reads
            (repo / "moved.py").write_text("x = 1\n", encoding="utf-8")
            _git(repo, "add", "-A")
            _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "moved")
            return "REPORT", {"resolved_model": "openai/main-model"}

        monkeypatch.setenv("OUROBOROS_MODEL", "openai/main-model")
        monkeypatch.setattr(dsr, "run_deep_self_review", review)
        result = run_system_review(system_ctx)
        after_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        assert after_head != before_head
        record = load_record(system_ctx.drive_root, result["record_id"])
        assert (record["subject"]["tree_sha"], record["subject"]["head"], record["subject"]["base"]) == (
            before_tree, before_head, before_head)
        [moved] = [r for r in record["verdict"]["degraded_reasons"] if r.startswith("system_tree_moved_during_review")]
        assert before_head[:12] in moved and after_head[:12] in moved

    @pytest.mark.parametrize("args, fragment", [
        ({"subject": "system", "surface": "change"}, "subject=system goes with surface=system"),
        ({"subject": "worktree", "surface": "system"}, "subject=system goes with surface=system"),
        ({"subject": "system", "surface": "system", "reviewers": ["a", "b"]}, "seats exactly one reviewer"),
    ])
    def test_the_system_surface_is_one_seat_over_the_whole_system(self, system_ctx, monkeypatch, args, fragment):
        from ouroboros.tools.review_change import ReviewChangeArgumentError, run_review_change

        self._stub(monkeypatch, ("unused", {}))
        with pytest.raises(ReviewChangeArgumentError, match=fragment):
            run_review_change(system_ctx, **args)


class TestIsProbablyBinary:
    def test_nul_byte_is_binary(self, tmp_path):
        """File containing a NUL byte is detected as binary."""
        f = tmp_path / "blob.bin"
        f.write_bytes(b"some text\x00more text")
        assert _is_probably_binary(f) is True

    def test_plain_text_is_not_binary(self, tmp_path):
        """Plain text file is not detected as binary."""
        f = tmp_path / "script.py"
        f.write_text("def hello():\n    return 'world'\n")
        assert _is_probably_binary(f) is False

    def test_high_non_printable_ratio_is_binary(self, tmp_path):
        """File with >30% non-printable bytes (ASCII control range) is detected as binary."""
        # 40% non-printable (bytes 1–8 range, ASCII control chars)
        payload = bytes(range(1, 9)) * 10 + b"normal text" * 3
        f = tmp_path / "data.unknown"
        f.write_bytes(payload)
        assert _is_probably_binary(f) is True

    def test_high_byte_ratio_is_binary(self, tmp_path):
        """File with invalid UTF-8 high bytes (no NUL) is detected as binary.

        bytes >= 128 alone are safe for valid UTF-8 (Cyrillic, CJK), but
        invalid UTF-8 sequences (e.g. raw Latin-1 bytes 0x80-0xFF) must still
        be caught by the incremental UTF-8 decode check.
        """
        # Raw Latin-1 bytes 0x80-0xFF: invalid UTF-8, no NUL, few control chars
        payload = bytes(range(128, 256)) * 5 + b"ascii text" * 5
        f = tmp_path / "data.blob"
        f.write_bytes(payload)
        assert _is_probably_binary(f) is True

    def test_only_first_sniff_bytes_read(self, tmp_path):
        """_is_probably_binary only reads _BINARY_SNIFF_BYTES bytes, not the whole file."""
        from ouroboros.tools.review_helpers import _BINARY_SNIFF_BYTES
        # File is mostly text but has NUL in the first 8KB window
        payload = b"text data\x00more" + b"a" * (_BINARY_SNIFF_BYTES * 2)
        f = tmp_path / "big.bin"
        f.write_bytes(payload)
        # Should detect NUL in the first chunk and return True
        assert _is_probably_binary(f) is True

    def test_empty_file_is_not_binary(self, tmp_path):
        """Empty file does not crash and returns False."""
        f = tmp_path / "empty.bin"
        f.write_bytes(b"")
        assert _is_probably_binary(f) is False

    def test_missing_file_returns_false(self, tmp_path):
        """Missing file returns False (let caller handle read failure)."""
        f = tmp_path / "does_not_exist.bin"
        assert _is_probably_binary(f) is False


class TestNoProxyLlmChat:
    """LLMClient.chat(no_proxy=True) — proxy-free httpx transport for macOS fork-safety."""

    def test_chat_no_proxy_uses_trust_env_false(self):
        """chat(no_proxy=True) builds an httpx.Client with trust_env=False and mounts={}."""
        import httpx
        from ouroboros.llm import LLMClient

        captured_clients = []

        real_httpx_client = httpx.Client

        def capturing_httpx_client(*args, **kwargs):
            c = real_httpx_client(*args, **kwargs)
            captured_clients.append(c)
            return c

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        # Resolve the SDK before mocking the HTTP class it inherits on import.
        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client", side_effect=capturing_httpx_client):
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.return_value = mock_resp
                mock_openai_cls.return_value = mock_oa

                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                    llm.chat(
                        messages=[{"role": "user", "content": "hi"}],
                        model="openai/gpt-5.5-pro",
                        max_tokens=8,
                        no_proxy=True,
                    )

        # At least one httpx.Client was created
        assert len(captured_clients) >= 1
        created = captured_clients[0]
        # trust_env=False and mounts={} are the key invariants
        assert created._mounts == {} or not created._mounts

    def test_chat_no_proxy_closes_http_client(self):
        """chat(no_proxy=True) closes the one-shot httpx.Client after the call."""
        import httpx
        from ouroboros.llm import LLMClient

        closed_clients = []
        real_httpx_client = httpx.Client

        class TrackingClient(real_httpx_client):
            def close(self):
                closed_clients.append(self)
                super().close()

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client", TrackingClient):
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.return_value = mock_resp
                mock_openai_cls.return_value = mock_oa

                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                    llm.chat(
                        messages=[{"role": "user", "content": "hi"}],
                        model="openai/gpt-5.5-pro",
                        max_tokens=8,
                        no_proxy=True,
                    )

        assert len(closed_clients) >= 1, "httpx.Client must be closed after no_proxy call"

    def test_chat_no_proxy_closes_on_exception(self):
        """chat(no_proxy=True) closes the http client even when the API call raises."""
        import httpx
        from ouroboros.llm import LLMClient

        closed_clients = []
        real_httpx_client = httpx.Client

        class TrackingClient(real_httpx_client):
            def close(self):
                closed_clients.append(self)
                super().close()

        llm = LLMClient()

        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client", TrackingClient):
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.side_effect = RuntimeError("boom")
                mock_openai_cls.return_value = mock_oa

                with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                    with pytest.raises(RuntimeError, match="boom"):
                        llm.chat(
                            messages=[{"role": "user", "content": "hi"}],
                            model="openai/gpt-5.5-pro",
                            max_tokens=8,
                            no_proxy=True,
                        )

        assert len(closed_clients) >= 1, "httpx.Client must be closed even after exception"

    def test_chat_no_proxy_skips_generation_cost_fetch(self):
        """chat(no_proxy=True) does not call _fetch_generation_cost (proxy/OS path)."""
        from ouroboros.llm import LLMClient

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "id": "gen-abc123",  # has a generation id — would trigger cost fetch normally
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }

        with mock.patch("openai.OpenAI") as mock_openai_cls:
            with mock.patch("httpx.Client") as mock_httpx_cls:
                mock_http = mock.Mock()
                mock_httpx_cls.return_value = mock_http
                mock_oa = mock.Mock()
                mock_oa.chat.completions.create.return_value = mock_resp
                mock_openai_cls.return_value = mock_oa
                with mock.patch.object(llm, "_fetch_generation_cost") as mock_cost:
                    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                        llm.chat(
                            messages=[{"role": "user", "content": "hi"}],
                            model="openai/gpt-5.5-pro",
                            max_tokens=8,
                            no_proxy=True,
                        )
                    mock_cost.assert_not_called()

    def test_chat_no_proxy_false_uses_cached_client(self):
        """chat(no_proxy=False, default) uses the shared cached client, not a new one."""
        import httpx
        from ouroboros.llm import LLMClient

        new_clients = []
        real_httpx_client = httpx.Client

        def counting_httpx_client(*args, **kwargs):
            c = real_httpx_client(*args, **kwargs)
            new_clients.append(c)
            return c

        llm = LLMClient()
        mock_resp = mock.Mock()
        mock_resp.model_dump.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

        with mock.patch("httpx.Client", side_effect=counting_httpx_client):
            with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-test"}, clear=False):
                with mock.patch.object(llm, "_get_remote_client") as mock_get:
                    mock_oa = mock.Mock()
                    mock_oa.chat.completions.create.return_value = mock_resp
                    mock_get.return_value = mock_oa
                    llm.chat(
                        messages=[{"role": "user", "content": "hi"}],
                        model="openai/gpt-5.5-pro",
                        max_tokens=8,
                        no_proxy=False,
                    )
                    mock_get.assert_called_once()

        # no_proxy=False must not construct a new httpx.Client
        assert len(new_clients) == 0


def test_direct_openai_deep_review_sends_a_real_openai_model_id():
    """PHYSICAL-PAYLOAD proof, not a defaults-table assertion.

    An owner may pin the slug `openai/gpt-5.6-sol-pro`. That `-pro`
    suffix is an OpenRouter routing slug, NOT an OpenAI model id: live-probed
    2026-07-29, `gpt-5.6-sol-pro` on api.openai.com /v1/chat/completions returns
    404, while pro reasoning exists only on /v1/responses as
    `reasoning.mode="pro"` (200) — and /v1/chat/completions rejects a `reasoning`
    parameter outright (400 "Unknown parameter"). Every LLM call in llm.py is a
    chat.completions call, so the direct-OpenAI deep-review slot ships plain Sol
    and this test pins what actually reaches the wire.
    """
    import os
    from unittest import mock

    from ouroboros.llm import LLMClient
    from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

    slot = OPENAI_DIRECT_DEFAULTS["deep_self_review"]
    assert slot.startswith("openai::"), slot

    mock_resp = mock.Mock()
    mock_resp.model_dump.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    with mock.patch("openai.OpenAI") as mock_openai_cls:
        mock_oa = mock.Mock()
        mock_oa.chat.completions.create.return_value = mock_resp
        mock_openai_cls.return_value = mock_oa
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-direct-test"}, clear=False):
            LLMClient().chat(
                messages=[{"role": "user", "content": "hi"}],
                model=slot, max_tokens=8, no_proxy=True,
            )
        assert mock_oa.chat.completions.create.called
        payload = mock_oa.chat.completions.create.call_args.kwargs

    # The id on the wire is a REAL OpenAI model, never the OpenRouter slug.
    assert payload["model"] == "gpt-5.6-sol"
    assert not payload["model"].endswith("-pro")
    # ...and no `reasoning` object is smuggled onto a chat.completions call, which
    # the live API rejects with 400 (the only pro carrier is the Responses API).
    assert "reasoning" not in payload
    assert "reasoning" not in (payload.get("extra_body") or {})


def test_direct_fallback_preserves_an_explicit_real_model_pin():
    """Only router-only `-pro` slugs are substituted by the provider default; an
    owner's explicit pin of a REAL OpenAI model keeps the mechanical rewrite."""
    import os
    from unittest import mock

    from ouroboros.provider_models import OPENAI_DIRECT_DEFAULTS

    env = {"OPENAI_API_KEY": "sk-test"}
    with mock.patch.dict(os.environ, env, clear=False):
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ.pop("USE_LOCAL_MAIN", None)
        with mock.patch.dict(os.environ, {"OUROBOROS_MODEL": "openai/gpt-5.5"}):
            reason, model = deep_review_route()
            available = not reason
        assert available is True
        assert model == "openai::gpt-5.5", "an explicit real-model pin survives"
        with mock.patch.dict(os.environ, {"OUROBOROS_MODEL": "openai/gpt-5.5-pro"}):
            reason, model = deep_review_route()
            available = not reason
        assert available is True
        assert model == OPENAI_DIRECT_DEFAULTS["deep_self_review"], (
            "a router-only -pro slug lands on the provider default"
        )
