"""Harness-observed read coverage for a delegated review session.

The host cannot execute a vendor session's reads, so the only witness of what
that reviewer actually opened is the run's own tool-call journal. These tests
pin the grammar that journal is read with (and, just as importantly, what it
refuses to read), the fold over the required-source manifest, and the usage
facts the session executor attaches — the SAME keys a native inspection episode
emits, with `harness_observed` where the episode says `host_observed`.

Fixtures are the shapes observed in live Claudexor journals on this machine:
codex `/bin/bash -lc "…"` command targets, claude `Read:`/`Grep:` targets with
the tool input beside them, cursor bare-path `read` targets.
"""

import hashlib
import itertools
import json

import pytest

from ouroboros.review_session_reads import (
    READ_PROVENANCE,
    fold_session_coverage,
    parse_session_read_receipts,
    session_event_files,
    session_read_facts,
    session_source_reader,
)

from tests._review_session_route_shared import _owned_gateway_uses_each_test_transport as __owned_gateway
from tests._review_session_route_shared import fake_route as __fake_route

# Fixtures are requested by name as test parameters, so they are re-bound
# through a module attribute (the F811 rule the CI ruff gate runs).
_owned_gateway_uses_each_test_transport = __owned_gateway
fake_route = __fake_route

from tests._review_session_route_shared import (  # noqa: E402
    FakeLLM,
    _agent_request,
    _agent_slot,
)


_use_ids = itertools.count(1)


def _command_event(shell, *, wrapped=True, name="command", use_id=None):
    target = f'/bin/bash -lc "{shell}"' if wrapped else shell
    return {"type": "tool_call",
            "tool": {"name": name, "kind": "command", "target": target,
                     "use_id": use_id or f"item_{next(_use_ids)}"}}


def _claude_event(path, *, name="Read", kind="file", use_id=None, **window):
    return {"type": "tool_call",
            "tool": {"name": name, "kind": kind, "target": f"{name}: {path}",
                     "use_id": use_id or f"toolu_{next(_use_ids)}"},
            "payload": {"input": {"file_path": path, **window}}}


def _cursor_event(path, *, name="read", kind="file", use_id=None):
    return {"type": "tool_call", "tool": {"name": name, "kind": kind, "target": path,
                                          "use_id": use_id or f"call_{next(_use_ids)}"}}


def _result(call, **overrides):
    """The `tool_result` the daemon writes when that call finished: same
    `use_id`, an outcome status and (for a command) its exit code."""
    tool = {**call["tool"], "status": "ok", "content_summary": "…"}
    if tool.get("kind") == "command":
        tool["exit_code"] = 0
    return {"type": "tool_result", "tool": {**tool, **overrides}}


def _completed(*events):
    """Every call in `events` followed by its own ok result — the journal shape
    of a session where each call ran to completion."""
    return [row for event in events for row in (event, _result(event))]


def _journal(tmp_path, *events, attempt="a01", run="run-1", paired=True):
    """One attempt journal in the engine's own run layout."""
    directory = tmp_path / run / "attempts" / attempt
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "events.jsonl"
    rows = _completed(*events) if paired else events
    path.write_text("".join(json.dumps(event, ensure_ascii=False) + "\n" for event in rows),
                    encoding="utf-8")
    return path


def _parse(tmp_path, *events, scope_root=None, paired=True, **kwargs):
    """Parse a journal of `events`; by default every call event carries its own
    ok result, because only a completed call can prove a read."""
    return parse_session_read_receipts([_journal(tmp_path, *events, paired=paired, **kwargs)],
                                       scope_root=str(scope_root or tmp_path / "repo"))


def _manifest_row(repo, relative, *, root="system_repo"):
    text = (repo / relative).read_text(encoding="utf-8")
    return {"root": root, "path": relative, "disposition": "modified",
            "source_revision": hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
            "complete_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "complete_chars": len(text), "range_basis": "unicode_text_universal_newlines"}


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "ouroboros").mkdir(parents=True)
    (root / "ouroboros" / "safety.py").write_text(
        "".join(f"line {index}\n" for index in range(1, 41)), encoding="utf-8")
    (root / "prompts").mkdir()
    (root / "prompts" / "SAFETY.md").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# the parser: what the journal proves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shell, expected", [
    ("cat ouroboros/safety.py", {"whole_file": True}),
    ("nl -ba ouroboros/safety.py", {"whole_file": True}),
    ("sed -n '12,40p' ouroboros/safety.py", {"start_line": 12, "end_line": 40}),
    ("sed -n 12,40p ouroboros/safety.py", {"start_line": 12, "end_line": 40}),
    ("sed -n '9p' ouroboros/safety.py", {"start_line": 9, "end_line": 9}),
    ("sed -n '5,$p' ouroboros/safety.py", {"start_line": 5}),
    ("head -n 50 ouroboros/safety.py", {"start_line": 1, "end_line": 50}),
    ("head -50 ouroboros/safety.py", {"start_line": 1, "end_line": 50}),
    ("head ouroboros/safety.py", {"start_line": 1, "end_line": 10}),
    ("tail -n 5 ouroboros/safety.py", {"from_end_lines": 5}),
    ("tail -n +30 ouroboros/safety.py", {"start_line": 30}),
])
def test_codex_shell_read_forms_become_receipts(tmp_path, shell, expected):
    """The command grammar the live codex journals actually carry."""
    receipts = _parse(tmp_path, _command_event(shell))
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["tool"] == "read_file" and receipt["outcome"] == "executed"
    assert receipt["delivered"] is True and receipt["provenance"] == READ_PROVENANCE
    assert receipt["opened_path"] == "ouroboros/safety.py"
    assert receipt["opened_root"] == "session_root"
    assert receipt["evidence"]["harness"] == "codex" and shell in receipt["evidence"]["raw_target"]
    for key, value in expected.items():
        assert receipt[key] == value
    assert "end_line" in receipt or "end_line" not in expected


@pytest.mark.parametrize("shell", [
    "rg -n 'def review' ouroboros/safety.py",      # search, never a delivered extent
    "grep -c review ouroboros/safety.py",
    "wc -l ouroboros/safety.py",
    "pwd",
    "ls ouroboros",
    "git diff --cached -- ouroboros/safety.py",
    "cat ouroboros/safety.py > /tmp/copy",         # redirected away from the model
    "head -c 200 ouroboros/safety.py",             # a byte window, not a line window
    "head -n -3 ouroboros/safety.py",              # all but the last three
    "sed -n '1,3p;9,12p' ouroboros/safety.py",     # several ranges in one script
    "for f in a b; do cat $f; done",               # the path is a variable
    "cat ouroboros/*.py",                          # a glob names no one file
])
def test_unmodelled_and_searching_commands_prove_no_read(tmp_path, shell):
    """Fail toward not proven: what the grammar does not model reads nothing."""
    assert _parse(tmp_path, _command_event(shell)) == []


def test_a_pipeline_proves_nothing_even_though_it_opens_the_file(tmp_path):
    """`cat f | head -20` opens the whole file and shows twenty lines; crediting
    the whole file would be the overclaim this parser exists to avoid."""
    assert _parse(tmp_path, _command_event("cat ouroboros/safety.py | head -20")) == []


def test_unbalanced_quoting_is_skipped_not_raised(tmp_path):
    assert _parse(tmp_path, _command_event("sed -n '1,5p ouroboros/safety.py", wrapped=False)) == []


@pytest.mark.parametrize("command", [
    "false && cat prompts/SAFETY.md ; true",
    "false && cat prompts/SAFETY.md;true",
    "true || cat prompts/SAFETY.md",
    "cat prompts/SAFETY.md >/dev/null",
    "if false; then\ncat prompts/SAFETY.md\nfi",
])
def test_successful_shell_call_does_not_prove_each_read_ran(tmp_path, command):
    assert _parse(tmp_path, _command_event(command)) == []


@pytest.mark.parametrize("result", [None, {"status": "error"}, {"exit_code": 1}])
def test_unfinished_or_failed_read_is_not_credited(tmp_path, result):
    call = _command_event("cat prompts/SAFETY.md")
    events = [call] if result is None else [call, _result(call, **result)]
    assert _parse(tmp_path, *events, paired=False) == []


def test_successful_result_can_precede_its_call_in_the_journal(tmp_path):
    call = _command_event("cat prompts/SAFETY.md")
    assert len(_parse(tmp_path, _result(call), call, paired=False)) == 1


def test_absolute_quoted_and_relative_paths_normalize_to_one_opened_path(tmp_path):
    repo_root = tmp_path / "repo with spaces"
    (repo_root / "ouroboros").mkdir(parents=True)
    absolute_path = (repo_root / "ouroboros/safety.py").as_posix()
    receipts = _parse(
        tmp_path,
        _command_event(f"cat '{absolute_path}'", wrapped=False),
        _command_event("cat ./ouroboros/safety.py"),
        _command_event(f"sed -n '1,4p' '{absolute_path}'"),
        scope_root=repo_root)
    # The first two are the same whole-file extent and fold into ONE receipt.
    assert [r["opened_path"] for r in receipts] == ["ouroboros/safety.py"] * 2
    assert [r["opened_root"] for r in receipts] == ["session_root"] * 2


def test_a_read_outside_the_scope_root_is_not_a_session_root_read(tmp_path):
    outside_path = (tmp_path / "outside.txt").as_posix()
    receipts = _parse(tmp_path, _command_event(f"cat '{outside_path}'", wrapped=False))
    assert [r["opened_root"] for r in receipts] == ["outside_session_root"]


def test_claude_read_without_a_window_is_whole_within_the_readers_line_bound(tmp_path):
    receipt = _parse(tmp_path, _claude_event("/repo/docs/CHECKLISTS.md"))[0]
    assert receipt["whole_file"] is True and receipt["bounded_lines"] == 2000
    assert receipt["evidence"]["harness"] == "claude"


def test_claude_read_window_becomes_an_inclusive_line_range(tmp_path):
    receipt = _parse(tmp_path, _claude_event("ouroboros/safety.py", offset=905, limit=60))[0]
    assert (receipt["start_line"], receipt["end_line"]) == (905, 964)
    assert receipt["whole_file"] is False


def test_claude_search_tools_are_not_reads(tmp_path):
    assert _parse(tmp_path, _claude_event("ouroboros/safety.py", name="Grep", kind="search"),
                  _claude_event("ouroboros", name="Glob", kind="search")) == []


def test_editing_and_writing_file_tools_are_not_reads(tmp_path):
    assert _parse(tmp_path, _claude_event("ouroboros/safety.py", name="edit"),
                  _claude_event("ouroboros/safety.py", name="write_to_file")) == []


def test_cursor_read_records_an_opened_file_of_unproven_extent(tmp_path):
    """The cursor journal names the file and nothing about the delivered range."""
    receipt = _parse(tmp_path, _cursor_event("ouroboros/safety.py"))[0]
    assert receipt["range_unobserved"] is True and receipt["whole_file"] is False
    assert "start_line" not in receipt and receipt["evidence"]["harness"] == "cursor"


def test_every_attempt_journal_of_one_run_is_read(tmp_path):
    _journal(tmp_path, _command_event("cat ouroboros/safety.py"), attempt="a01")
    _journal(tmp_path, _command_event("cat prompts/SAFETY.md"), attempt="a02")
    files = session_event_files(str(tmp_path / "run-1"))
    receipts = parse_session_read_receipts(files, scope_root=str(tmp_path / "repo"))
    assert len(files) == 2
    assert {r["opened_path"] for r in receipts} == {"ouroboros/safety.py", "prompts/SAFETY.md"}


def test_a_missing_run_directory_yields_no_journals(tmp_path):
    assert session_event_files(str(tmp_path / "absent")) == []
    assert session_event_files("") == []


def test_malformed_journal_lines_are_skipped(tmp_path):
    path = _journal(tmp_path, _command_event("cat ouroboros/safety.py"))
    path.write_text("not json\n" + path.read_text(encoding="utf-8") + '{"type": "run_started"}\n',
                    encoding="utf-8")
    receipts = parse_session_read_receipts([path], scope_root=str(tmp_path / "repo"))
    assert [r["opened_path"] for r in receipts] == ["ouroboros/safety.py"]


# ---------------------------------------------------------------------------
# the fold: coverage over the required-source manifest
# ---------------------------------------------------------------------------


def test_whole_file_reads_cover_the_manifest_completely(tmp_path, repo):
    receipts = _parse(tmp_path, _command_event("cat ouroboros/safety.py"),
                      _command_event("nl -ba prompts/SAFETY.md"), scope_root=repo)
    coverage = fold_session_coverage(
        receipts, [_manifest_row(repo, "ouroboros/safety.py"), _manifest_row(repo, "prompts/SAFETY.md")],
        resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "complete" and coverage["required_source_count"] == 2
    assert [row["status"] for row in coverage["sources"]] == ["complete", "complete"]
    assert all(row["missing_ranges"] == [] for row in coverage["sources"])


def test_adjacent_ranges_fold_into_one_complete_source(tmp_path, repo):
    receipts = _parse(tmp_path, _command_event("head -n 20 ouroboros/safety.py"),
                      _command_event("sed -n '21,40p' ouroboros/safety.py"), scope_root=repo)
    coverage = fold_session_coverage(receipts, [_manifest_row(repo, "ouroboros/safety.py")],
                                     resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "complete"
    assert coverage["sources"][0]["covered_chars"] == coverage["sources"][0]["complete_chars"]


def test_a_partial_read_leaves_a_measured_gap(tmp_path, repo):
    receipts = _parse(tmp_path, _command_event("sed -n '1,10p' ouroboros/safety.py"), scope_root=repo)
    coverage = fold_session_coverage(receipts, [_manifest_row(repo, "ouroboros/safety.py")],
                                     resolve_file=session_source_reader(str(repo)))
    row = coverage["sources"][0]
    assert coverage["status"] == "incomplete" and row["status"] == "incomplete"
    # Ten of forty `line N\n` lines: the gap is exact, not a fraction.
    assert row["covered_chars"] == len("".join(f"line {i}\n" for i in range(1, 11)))
    assert row["missing_ranges"] == [[row["covered_chars"], row["complete_chars"]]]


def test_a_required_source_nothing_opened_is_missing_whole(tmp_path, repo):
    coverage = fold_session_coverage([], [_manifest_row(repo, "ouroboros/safety.py")],
                                     resolve_file=session_source_reader(str(repo)))
    row = coverage["sources"][0]
    assert coverage["status"] == "incomplete"
    assert row["missing_ranges"] == [[0, row["complete_chars"]]] and row["covered_chars"] == 0


def test_an_unproven_extent_covers_nothing(tmp_path, repo):
    """A cursor read and a claude whole-file read of a file past the reader's
    own line bound both name the file and prove no delivered range."""
    big = repo / "ouroboros" / "big.py"
    big.write_text("".join(f"line {index}\n" for index in range(1, 2500)), encoding="utf-8")
    receipts = _parse(tmp_path, _cursor_event("ouroboros/safety.py"),
                      _claude_event("ouroboros/big.py"), scope_root=repo)
    coverage = fold_session_coverage(
        receipts, [_manifest_row(repo, "ouroboros/safety.py"), _manifest_row(repo, "ouroboros/big.py")],
        resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "unobserved"
    assert [row["covered_chars"] for row in coverage["sources"]] == [0, 0]


def test_a_claude_whole_file_read_within_the_bound_is_complete(tmp_path, repo):
    receipts = _parse(tmp_path, _claude_event("prompts/SAFETY.md"), scope_root=repo)
    coverage = fold_session_coverage(receipts, [_manifest_row(repo, "prompts/SAFETY.md")],
                                     resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "complete"


def test_tail_ranges_resolve_against_the_current_line_count(tmp_path, repo):
    receipts = _parse(tmp_path, _command_event("head -n 35 ouroboros/safety.py"),
                      _command_event("tail -n 5 ouroboros/safety.py"), scope_root=repo)
    coverage = fold_session_coverage(receipts, [_manifest_row(repo, "ouroboros/safety.py")],
                                     resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "complete"


def test_candidate_drift_is_a_source_gap_never_coverage(tmp_path, repo):
    """The reviewer read a file; the candidate tree no longer holds the declared
    revision, so whatever it read was not this source."""
    row = _manifest_row(repo, "ouroboros/safety.py")
    receipts = _parse(tmp_path, _command_event("cat ouroboros/safety.py"), scope_root=repo)
    (repo / "ouroboros" / "safety.py").write_text("drifted\n", encoding="utf-8")
    coverage = fold_session_coverage(receipts, [row], resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "incomplete"
    assert coverage["sources"][0]["reason"] == "source_gap"
    assert coverage["sources"][0]["covered_chars"] == 0


def test_a_declared_empty_manifest_is_its_own_state(repo):
    coverage = fold_session_coverage([], [], resolve_file=session_source_reader(str(repo)))
    assert coverage["status"] == "complete" and coverage["required_source_count"] == 0
    assert coverage["reason"] == "declared_empty"


def test_inline_delivery_needs_no_duplicate_file_read(repo):
    row = {**_manifest_row(repo, "prompts/SAFETY.md"), "coverage_basis": "delivered_inline"}
    coverage = fold_session_coverage([], [row], resolve_file=lambda source: None)
    assert coverage["status"] == "complete"
    assert coverage["sources"][0]["covered_chars"] == row["complete_chars"]
    assert coverage["sources"][0]["missing_ranges"] == []


@pytest.mark.parametrize("manifest", [None, "", {}, 7])
def test_no_manifest_is_unobserved_never_complete(manifest, repo):
    coverage = fold_session_coverage([], manifest, resolve_file=session_source_reader(str(repo)))
    assert coverage == {"status": "unobserved", "reason": "required_source_manifest_missing",
                        "sources": []}


def test_a_malformed_row_is_unobserved_and_never_blocks_the_others(tmp_path, repo):
    good = _manifest_row(repo, "prompts/SAFETY.md")
    bad = {**_manifest_row(repo, "ouroboros/safety.py"), "complete_sha256": "short"}
    receipts = _parse(tmp_path, _command_event("cat prompts/SAFETY.md"), scope_root=repo)
    coverage = fold_session_coverage(receipts, [bad, good], resolve_file=session_source_reader(str(repo)))
    assert [row["status"] for row in coverage["sources"]] == ["unobserved", "complete"]
    assert coverage["sources"][0]["reason"] == "required_source_identity_unavailable"
    assert coverage["status"] == "unobserved"


def test_a_row_outside_the_repository_roots_is_unreadable_from_a_session(tmp_path, repo):
    row = {**_manifest_row(repo, "prompts/SAFETY.md"), "root": "runtime_data"}
    coverage = fold_session_coverage([], [row], resolve_file=session_source_reader(str(repo)))
    assert coverage["sources"][0]["status"] == "unobserved"
    assert coverage["sources"][0]["reason"] == "required_source_unreadable"


def test_a_deleted_candidate_file_is_unreadable_not_covered(tmp_path, repo):
    row = _manifest_row(repo, "prompts/SAFETY.md")
    (repo / "prompts" / "SAFETY.md").unlink()
    coverage = fold_session_coverage([], [row], resolve_file=session_source_reader(str(repo)))
    assert coverage["sources"][0]["reason"] == "required_source_unreadable"


# ---------------------------------------------------------------------------
# the executor: the usage facts a session row carries
# ---------------------------------------------------------------------------


def _session_result(tmp_path, repo, fake, *events, manifest=None, run_dir=True):
    """Run one delegated review slot whose engine journal carries `events`."""
    from ouroboros.review_substrate import run_review_request

    fake.run_dir = tmp_path / "review-run"
    if run_dir:
        _journal(fake.run_dir.parent, *events, run=fake.run_dir.name)
    policy = {} if manifest is None else {"native_required_sources": manifest}
    request = _agent_request(session_root=str(repo), policy=policy)
    result = run_review_request(request, slots=[_agent_slot()], drive_root=tmp_path, llm=FakeLLM())
    return result.actors[0]


def test_session_usage_carries_harness_observed_coverage(tmp_path, repo, fake_route):
    actor = _session_result(tmp_path, repo, fake_route,
                            _command_event("cat ouroboros/safety.py"),
                            _claude_event("prompts/SAFETY.md"),
                            manifest=[_manifest_row(repo, "ouroboros/safety.py"),
                                      _manifest_row(repo, "prompts/SAFETY.md")])
    usage = actor["usage"]
    assert actor["status"] == "ok"
    assert usage["read_provenance"] == "harness_observed"
    assert usage["native_read_coverage"]["status"] == "complete"
    assert "native_incomplete" not in usage
    # The parsed reads stay retrievable in the operation's own source store.
    from ouroboros.artifacts import task_artifact_dir_path

    handle = usage["native_history_source"]
    assert handle["root"] == "artifact_store"
    stored = json.loads((task_artifact_dir_path(tmp_path, "t-agent")
                         / handle["path"]).read_text(encoding="utf-8"))
    assert stored["read_provenance"] == "harness_observed"
    assert {row["opened_path"] for row in stored["read_receipts"]} == {
        "ouroboros/safety.py", "prompts/SAFETY.md"}
    assert stored["coverage"]["status"] == "complete"


def test_incomplete_session_coverage_is_the_typed_fact_the_native_episode_emits(
        tmp_path, repo, fake_route):
    actor = _session_result(tmp_path, repo, fake_route,
                            _command_event("sed -n '1,10p' ouroboros/safety.py"),
                            manifest=[_manifest_row(repo, "ouroboros/safety.py")])
    usage = actor["usage"]
    assert usage["native_incomplete"] == "required_source_coverage_incomplete"
    assert usage["native_read_coverage"]["status"] == "incomplete"
    assert usage["read_provenance"] == "harness_observed"
    # The verdict itself is paid evidence and survives the coverage gap.
    assert actor["status"] == "ok" and actor["raw_text"] == "[]"


def test_a_session_without_a_manifest_carries_no_coverage_claim(tmp_path, repo, fake_route):
    usage = _session_result(tmp_path, repo, fake_route,
                            _command_event("cat ouroboros/safety.py"))["usage"]
    assert "native_read_coverage" not in usage and "read_provenance" not in usage


def test_an_unreadable_journal_is_disclosed_unobserved_not_a_failed_slot(
        tmp_path, repo, fake_route):
    actor = _session_result(tmp_path, repo, fake_route, run_dir=False,
                            manifest=[_manifest_row(repo, "ouroboros/safety.py")])
    assert actor["status"] == "ok" and actor["raw_text"] == "[]"
    assert actor["usage"]["native_read_coverage"] == {
        "status": "unobserved", "reason": "session_events_unavailable", "sources": []}
    assert actor["usage"]["read_provenance"] == "unobserved"


def test_a_parser_failure_never_leaves_settlement_by_exception(tmp_path, repo, fake_route, monkeypatch):
    """The session's verdict is paid work; a broken fold is a disclosure."""
    import ouroboros.review_session_reads as reads

    def explode(*_args, **_kwargs):
        raise RuntimeError("journal reader is broken")

    monkeypatch.setattr(reads, "parse_session_read_receipts", explode)
    actor = _session_result(tmp_path, repo, fake_route,
                            _command_event("cat ouroboros/safety.py"),
                            manifest=[_manifest_row(repo, "ouroboros/safety.py")])
    assert actor["status"] == "ok" and actor["raw_text"] == "[]"
    assert actor["usage"]["native_read_coverage"]["reason"] == "session_events_unavailable"
    assert actor["usage"]["read_provenance"] == "unobserved"


def test_the_facts_seam_needs_no_store_and_never_raises(tmp_path, repo):
    """`session_read_facts` is the whole seam the executor calls: without a
    place to keep the history the coverage still stands, and an unusable store
    is a lost record, never a lost verdict."""
    _journal(tmp_path, _command_event("cat ouroboros/safety.py"), run="review-run")
    manifest = [_manifest_row(repo, "ouroboros/safety.py")]
    for store in (None, {}, {"root": tmp_path / "missing" / "nested", "task_id": "../escape"}):
        facts = session_read_facts(str(tmp_path / "review-run"),
                                   {"native_required_sources": manifest},
                                   session_root=str(repo), store=store)
        assert facts["native_read_coverage"]["status"] == "complete"
        assert facts["read_provenance"] == "harness_observed"
        assert facts["native_history_source"] == {}
        assert "native_incomplete" not in facts


def test_the_facts_seam_claims_nothing_without_a_declared_manifest(tmp_path, repo):
    assert session_read_facts(str(tmp_path), {}, session_root=str(repo)) == {}
    assert session_read_facts(str(tmp_path), None, session_root=str(repo)) == {}


def test_the_partial_product_fact_rides_the_message_like_an_episodes(tmp_path, repo, fake_route):
    """A consumer reading the verdict text alone must still see that a required
    source went unread — the same carrier the native episode uses."""
    from ouroboros.review_execution import AgentSessionReviewExecutor, ReviewAssignment

    fake_route.run_dir = tmp_path / "review-run"
    _journal(tmp_path, _command_event("sed -n '1,10p' ouroboros/safety.py"), run="review-run")
    request = _agent_request(session_root=str(repo),
                             policy={"native_required_sources": [_manifest_row(repo, "ouroboros/safety.py")]})
    result = AgentSessionReviewExecutor(
        ReviewAssignment(request=request, slot=_agent_slot(), call_id="c-coverage",
                         call_type="scope_review", custody_root=tmp_path),
        llm=FakeLLM()).execute()
    assert result.message["native_incomplete"] == "required_source_coverage_incomplete"
    assert result.raw_text == "[]"  # the paid verdict itself is untouched


def test_the_runner_hands_the_engine_run_directory_to_its_caller(tmp_path, fake_route):
    """Without the run directory on the facts there is no journal to fold."""
    from tests._review_session_route_shared import _run_session_directly

    fake_route.run_dir = tmp_path / "review-run"
    facts = _run_session_directly(tmp_path)
    assert facts["run_dir"] == str(fake_route.run_dir)


def test_the_native_episode_keeps_its_own_host_observed_attestation():
    """This module never relabels a native episode's provenance: the episode
    attests `host_observed` and only a delegated session says `harness_observed`."""
    import inspect

    from ouroboros import review_native_episode

    source = inspect.getsource(review_native_episode.NativeToolRoundReviewExecutor)
    assert '"host_file_read_attestation": "host_observed"' in source
    assert "harness_observed" not in source


# ---------------------------------------------------------------------------
# observed sources: a host-declared absolute file the session may read (plan review's room)
# ---------------------------------------------------------------------------


def _observed_row(path):
    from ouroboros.tools.scope_required_sources import source_text_identity

    return {"root": "artifact_store", "path": path.name, "file": str(path),
            **source_text_identity(path.read_bytes())}


def test_an_observed_source_records_coverage_without_a_capability_delta(tmp_path, repo):
    """A `Read` of the snapshot's absolute path folds into measured coverage under the
    `harness_observed` provenance; an observed source never sets `native_incomplete`
    (quiet side: the same unread row under `native_required_sources` does), and a read of
    an unrelated absolute file contributes nothing. Reverted, the facts are `{}`."""
    snapshot = tmp_path / "artifacts" / "plan-dialogue-1.jsonl"
    snapshot.parent.mkdir()
    snapshot.write_text("".join(f'{{"n": {index}}}\n' for index in range(1, 41)), encoding="utf-8")
    row = _observed_row(snapshot)
    _journal(tmp_path, _claude_event(str(snapshot), offset=1, limit=40), run="run-whole")
    facts = session_read_facts(str(tmp_path / "run-whole"), {"observed_sources": [row]}, session_root=str(repo))
    assert facts["read_provenance"] == READ_PROVENANCE and "native_incomplete" not in facts
    [source] = facts["native_read_coverage"]["sources"]
    assert source["status"] == "complete" and source["covered_chars"] == source["complete_chars"] == row["complete_chars"]
    assert facts["native_read_coverage"]["status"] == "complete"
    _journal(tmp_path, _claude_event(str(snapshot), offset=1, limit=10), run="run-part")
    partial = session_read_facts(str(tmp_path / "run-part"), {"observed_sources": [row]}, session_root=str(repo))
    [source] = partial["native_read_coverage"]["sources"]
    assert source["status"] == "incomplete" and 0 < source["covered_chars"] < row["complete_chars"]
    assert "native_incomplete" not in partial  # observed, never a capability delta
    required = session_read_facts(str(tmp_path / "run-part"), {"native_required_sources": [row]}, session_root=str(repo))
    assert required["native_incomplete"] == "required_source_coverage_incomplete"
    other = tmp_path / "artifacts" / "other.txt"
    other.write_text("unrelated\n", encoding="utf-8")
    _journal(tmp_path, _claude_event(str(other), offset=1, limit=1), run="run-other")
    unrelated = session_read_facts(str(tmp_path / "run-other"), {"observed_sources": [row]}, session_root=str(repo))
    [source] = unrelated["native_read_coverage"]["sources"]
    assert source["status"] == "incomplete" and source["covered_chars"] == 0
