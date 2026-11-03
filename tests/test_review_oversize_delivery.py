"""Large review subjects travel intact through the configured delivery."""

from dataclasses import replace
from types import SimpleNamespace
import subprocess

import pytest

from ouroboros.artifacts import read_actor_source_bytes
from ouroboros.review_execution import ApiChatReviewExecutor, SessionInvocation
from ouroboros.review_native_episode import NativeToolRoundReviewExecutor
from ouroboros.review_session_preparation import prepare_review_session_request
from tests.test_native_tool_round_executor import _assignment
from tests.test_review_brief_coupling import _staged_repo
from tests._review_session_route_shared import (
    FakeGateway, _owned_gateway_uses_each_test_transport, fake_route,  # noqa: F401
)


LIMIT = {"scope": "turn_text", "unit": "unicode_scalars", "limit": 1_048_576,
         "prompt_budget": 1_040_000, "source": "native.turn/start",
         "askPromptBudget": {"shape": "ordinary_initial_attempt", "engineOverheadMax": 8576}}


@pytest.mark.parametrize("large", [True, False])
@pytest.mark.parametrize("access", ["readonly", "full"])
@pytest.mark.parametrize("workspace", ["git_ignored", "git_unignored", "directory"])
def test_complete_session_turn_pages_only_when_transport_requires_it(tmp_path, large, access, workspace):
    root, checking, data = (tmp_path / name for name in ("repo", "checking", "data"))
    for tree in (root, checking):
        if workspace != "directory":
            _staged_repo(tree)
            if workspace == "git_ignored":
                (tree / ".git" / "info" / "exclude").write_text(".review-drive/\n", encoding="utf-8")
        else:
            tree.mkdir()
    prompt = "Review exact evidence 😀\n" * (60_000 if large else 10)
    facts = {"input_limit": LIMIT, "data_root": str(data)}
    invocation = SessionInvocation(task_id="t-size", surface="multi_model_review", slot_id="s",
        timeout_sec=60, instructions="Read every source before giving your verdict.", access=access,
        source_delivery=facts)
    route = SimpleNamespace(route_id="test-harness", model="model", effort="", profile_id="")
    request = prepare_review_session_request(invocation, route, prompt=prompt, root=str(root),
        thread_id="", schema_asked=False,
        execution={"workspaceRoot": str(checking)} if access == "full" else None)
    assert len(request["prompt"]) + len(request["instructions"]) <= LIMIT["prompt_budget"]
    if large:
        source = facts["turn_sources"]["prompt"]
        assert read_actor_source_bytes(data, "t-size", source).decode("utf-8") == prompt
        target = checking if access == "full" else root
        assert (target / source["session_relative_path"]).read_text(encoding="utf-8") == prompt
        assert "not inlined" in request["prompt"] and source["sha256"] in request["prompt"]
        if workspace != "directory":
            status = subprocess.run(["git", "status", "--porcelain", "--", ".review-drive"],
                cwd=target, capture_output=True, text=True, check=True)
            assert not status.stdout
    else:
        assert request["prompt"] == prompt and not facts.get("turn_sources")


@pytest.mark.parametrize("large", [True, False])
def test_native_initial_subject_is_paged_without_an_episode_stop(tmp_path, monkeypatch, large):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir()
    task = "Required evidence 😀\n" * (65_000 if large else 5)
    assignment = replace(_assignment(repo, None, task), custody_root=data)
    monkeypatch.setattr("ouroboros.review_native_episode.native_episode_transcript_bound", lambda *_a, **_k: 100_000)
    executor = NativeToolRoundReviewExecutor(assignment)
    _registry, schemas, messages, size = executor._open_episode(str(repo), data)
    assert size < 100_000
    if large:
        source = executor._initial_source
        assert read_actor_source_bytes(data, "t-native", source).decode("utf-8") == executor.episode_prompt
        assert source["path"] in str(messages) and "not inlined" in str(messages)
    else:
        assert task.strip() in messages[1]["content"][0]["text"]


def test_packet_delivery_keeps_its_complete_evidence_and_no_file_only_pointer(tmp_path):
    # Tool-free packets cannot dereference a file. Their existing surface fit
    # ladder owns admissible packet size; the transport must not silently shed it.
    repo = tmp_path / "repo"
    repo.mkdir()
    assignment = _assignment(repo, None)
    subject = "packet subject 😀\n" * 65_000
    assignment.request.messages = [{"role": "user", "content": subject}]
    executor = ApiChatReviewExecutor(assignment)
    assert executor.messages == assignment.request.messages


@pytest.mark.parametrize("delegated", [False, True])
def test_history_dominating_brief_is_also_delivered_as_a_source(tmp_path, monkeypatch, delegated):
    from ouroboros.tools import review_brief_coupling as brief

    repo = _staged_repo(tmp_path / "repo")
    (repo / ".git" / "info" / "exclude").write_text(".review-drive/\n", encoding="utf-8")
    data = tmp_path / "data"
    history = "Owner's exact previous finding 😀\n" * 40_000
    monkeypatch.setattr(brief, "first_send_bound", lambda _brief: 100_000)
    text, manifest = brief.build_retrieving_brief(repo, brief.BriefInputs(
        delegated=delegated, task_id="t-size", source_root=str(data), layer="core",
        diff_text="+small diff", intent=brief.BriefIntent(review_rebuttal=history)))
    assert len(text) < 100_000 and manifest["brief_delivery"] == "paged"
    source = manifest["brief_source"]
    original = read_actor_source_bytes(data, "t-size", source).decode("utf-8")
    assert history.strip() in original
    assert read_actor_source_bytes(data, "t-size", manifest["diff_source"]).decode("utf-8") == "+small diff"
    assert source["sha256"] in text


def test_session_dispatch_uses_route_limit_and_rejoins_its_sourced_request(tmp_path, monkeypatch, fake_route):  # noqa: F811
    from ouroboros import review_execution as execution

    repo = _staged_repo(tmp_path / "repo")
    (repo / ".git" / "info" / "exclude").write_text(".review-drive/\n", encoding="utf-8")
    monkeypatch.setattr(FakeGateway, "ask_input_limits", lambda _self: {"fake-review": LIMIT}, raising=False)
    original_poll = execution._poll_session_terminal
    monkeypatch.setattr(execution, "_poll_session_terminal", lambda *_a: (_ for _ in ()).throw(OSError("restart")))
    state = {}
    invocation = SessionInvocation(task_id="t-size", surface="scope_review", slot_id="s",
        operation_id="op-size", timeout_sec=30, retry_state=state)
    prompt = "Exact source 😀\n" * 80_000

    def run():
        return execution.run_delegated_review_session(prompt=prompt, root=str(repo),
            custody_drive=tmp_path / "data", invocation=invocation)

    with pytest.raises(OSError, match="restart"):
        run()
    monkeypatch.setattr(execution, "_poll_session_terminal", original_poll)
    result = run()
    requests = [r for gateway in fake_route.instances for r in gateway.start_requests]
    assert len(requests) == 1 and len(requests[0]["prompt"]) < LIMIT["prompt_budget"]
    assert result["idempotent_recovery"] is True
    assert result["review_source_delivery"]["turn_sources"]["prompt"]["sha256"]
