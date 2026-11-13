"""The operator lane of the external review wrapper: the commit gate's own
review-only cycle, run in the runtime's isolated checkout of the staged index
(R3), with its hermetic tests, the author's optional ``surface=preflight`` look
(record + the helper's full answer), drift artifact and custody-bound retention."""

import contextlib
import errno
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import _contributor_packet_shared as shared

REPO = Path(__file__).resolve().parents[1]
_PREFLIGHT_ANSWER = json.dumps([{"item": "code_quality", "verdict": "PASS", "severity": "advisory",
                                 "reason": "the preflight seat read README.md in full"}])


def test_settings_land_before_the_lane_without_a_key_heuristic():
    """The operator lane runs only after the settings landed in the environment, and
    never guesses a reviewer's availability from a key's presence."""
    import inspect
    import scripts.run_external_review as module

    main_source = inspect.getsource(module.main)
    prepare_source = inspect.getsource(module._prepare_review_configuration)
    operator_source = inspect.getsource(module._operator_lane)
    assert "_load_settings_into_env()" in prepare_source
    assert main_source.index("_prepare_review_configuration(args)") < main_source.index("_operator_lane(")
    for source in (main_source, operator_source):
        assert 'os.environ.get("ANTHROPIC_API_KEY"' not in source


def _operator_fixture(tmp_path: Path, monkeypatch, order: list | None = None, *, scout_enabled: bool = True,
                      preflight_answer: str = _PREFLIGHT_ANSWER) -> tuple[Path, list[dict]]:
    """An installed body with a staged README edit; the operator lane's paid seam
    (the review substrate) and hermetic test runner are the golden stand-ins, and
    the gate reads the golden panel from the frozen slot plan. ``order`` records
    every test run (with the tree it ran in) and every paid send."""
    import ouroboros.review_substrate as substrate
    import scripts.run_external_review as module
    from ouroboros.tools import git as git_mod
    from ouroboros.tools import review_helpers

    fixture = shared.init_installed_body(tmp_path)
    repo = Path(fixture["repo"])
    (repo / "README.md").write_text("staged edit\n", encoding="utf-8")
    shared.git(repo, "add", "README.md")
    monkeypatch.setattr(module, "REPO", repo)
    monkeypatch.setattr(module, "_load_settings_into_env", lambda: None)
    monkeypatch.setattr(module, "_resolved_review_config",
                        lambda *, profile="production_commit_gate": json.loads(json.dumps(shared.GOLDEN_CONFIG)))
    monkeypatch.setattr(module, "_select_healthy_openrouter_key", lambda **_kwargs: False)
    # The gate's panel is the review pool; ``api-scout`` is the unmarked row the
    # ``--preflight-reviewer`` lane names (``tests.test_git_review_preflight_gate._roster``).
    monkeypatch.setenv("OUROBOROS_SUBAGENTS", shared.golden_pool(
        {"subagent_id": "api-scout", "name": "API scout", "recommended_use": "An early look.",
         "route": {"kind": "api_model", "target_id": "openai/fake-reviewer"}, "effort": "high",
         "enabled": scout_enabled}))
    monkeypatch.delenv("OUROBOROS_REVIEW_ENFORCEMENT", raising=False)  # retired: the lane needs no pin
    monkeypatch.setenv("OUROBOROS_PRE_PUSH_TESTS", "1")
    briefs: list[dict] = []
    golden = shared.golden_substrate(briefs)
    order = [] if order is None else order

    def run_review_request(request, *, slots, drive_root, llm=None, usage_ctx=None):
        if [slot.slot_id for slot in slots] != ["api-scout"]:
            order.append(("panel", sorted(slot.slot_id for slot in slots)))
            return golden(request, slots=slots, drive_root=drive_root, llm=llm, usage_ctx=usage_ctx)
        order.append(("preflight", request.session_root))
        reserved = (getattr(usage_ctx, "_review_reserved_operations", None) or {}).get(request.surface) or {}
        return SimpleNamespace(actors=[{
            "slot_id": "api-scout", "model": slots[0].model, "status": "ok", "raw_text": preflight_answer,
            "usage": {"provider": "openrouter", "resolved_model": "openai/fake-reviewer",
                      "prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
            "operation_id": str(reserved.get("api-scout") or "op-api-scout"),
            "operation_state": "settled", "late_result_pending": False}])

    def tests(ctx, **kwargs):
        order.append(("tests", Path(ctx.repo_dir)))
        return shared.passing_test_runner(ctx, **kwargs)

    monkeypatch.setattr(substrate, "run_review_request", run_review_request)
    monkeypatch.setattr(review_helpers, "_run_review_preflight_tests", tests)
    monkeypatch.setattr(git_mod, "_run_review_preflight_tests", tests)
    return repo, briefs


def _run_operator_lane(module, monkeypatch, tmp_path: Path, *extra: str) -> tuple[int, Path]:
    output = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}"
    monkeypatch.setattr(module.sys, "argv", [
        "run_external_review.py", f"--output={output}", f"--drive-root={tmp_path / 'drive'}", *extra, "fix: one"])
    return module.main(), output


def test_operator_lane_is_the_commit_gate_dry_run_in_an_isolated_checkout(tmp_path, monkeypatch):
    """R3: the operator lane runs the commit gate's own non-committing cycle over the
    staged index, in the runtime's isolated checkout of the staged patch; without a
    named preflight row the output states the fact; the primary worktree is never touched."""
    import scripts.run_external_review as module

    repo, briefs = _operator_fixture(tmp_path, monkeypatch)
    staged_before = shared.git(repo, "diff", "--cached")
    checkouts = tmp_path / "drive" / "state" / "review_checkouts"

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path)

    assert exit_code == 0, (output / "outcome.json").read_text(encoding="utf-8")
    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["exit_code"] == 0 and outcome["outcome"]["status"] == "passed"
    assert outcome["outcome"]["review_record_id"].startswith("rl-")
    assert "retained_checkout" not in outcome["outcome"]
    # No preflight row named: a stated fact, never a look and never a block.
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["preflight"] == {"status": "not_performed", "record_id": ""}
    assert preflight["seats"] == [] and (output / "preflight.txt").read_text(encoding="utf-8") == "\n"
    sections = shared.full_output_sections((output / "full-output.txt").read_text(encoding="utf-8"))
    seats = json.loads(sections["REVIEW SEAT RECORDS (ledger rows with retained answers, full, untruncated)"])
    assert [(seat["seat_id"], seat["answer"]) for seat in seats] == [
        (slot, shared.ANSWERS[slot]) for slot in ("t1", "t2", "s1")]
    verdict = json.loads(sections["AGGREGATE VERDICT"])
    assert verdict["review_record"]["record_id"] == outcome["outcome"]["review_record_id"]
    assert (verdict["review_record"]["aggregate"], verdict["review_record"]["surface"]) == ("PASS", "commit_gate")
    assert verdict["cost_report"]["unreported_or_unknown_cost_slots"] == ["t2"]
    # Every seat read the isolated checkout of the staged patch, never this worktree.
    assert sorted(brief["slot_id"] for brief in briefs) == ["s1", "t1", "t2"]
    for brief in briefs:
        if brief["session_root"]:
            assert Path(brief["session_root"]).parent.parent == checkouts
            assert Path(brief["session_root"]) != repo
    # Custody settled: the checkout is gone; the primary worktree still holds the staged edit.
    assert not checkouts.exists() or not any(checkouts.iterdir())
    assert shared.git(repo, "diff", "--cached") == staged_before
    assert shared.git(repo, "worktree", "list", "--porcelain").count("worktree ") == 1
    # The cycle's own hygiene (``_ensure_gitignore``) added a file the operator never
    # staged: the untracked-safe comparison surfaces exactly that, loudly, as drift.
    drift = (output / "reviewed-tree-drift.diff").read_text(encoding="utf-8")
    assert "+++ b/.gitignore" in drift and "+staged edit" in drift
    assert not (repo / ".gitignore").exists()


def test_operator_lane_runs_the_named_preflight_through_review_change(tmp_path, monkeypatch, capsys):
    """Contract approval item 3: ``--preflight-reviewer`` runs the commit gate's own look,
    ``review_change(subject=worktree, surface=preflight)`` on the ONE named row, inside the
    isolated checkout and after the hermetic tests there; the output keeps the
    ``surface=preflight`` record beside the helper's full answer, the commit panel still
    reviews on its own record, and the drift check still runs."""
    import scripts.run_external_review as module
    from tests.test_git_review_preflight_gate import _roster

    _roster(monkeypatch)
    order: list = []
    repo, _briefs = _operator_fixture(tmp_path, monkeypatch, order)
    checkouts = tmp_path / "drive" / "state" / "review_checkouts"

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path, "--preflight-reviewer=api-scout")

    assert exit_code == 0, (output / "outcome.json").read_text(encoding="utf-8")
    # Tests first, in the isolated checkout; then the one-seat look; then the panel.
    (first, tested), (second, _root) = order[0], order[1]
    assert (first, second) == ("tests", "preflight")
    assert tested.parent.parent == checkouts and tested != repo
    assert [kind for kind, _ in order[2:]] and {kind for kind, _ in order[2:]} == {"panel"}
    assert sorted(slot for _kind, slots in order[2:] for slot in slots) == ["s1", "t1", "t2"]
    summary = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    assert summary["preflight"]["status"] == "performed" and summary["preflight"]["reviewer"] == "api-scout"
    assert summary["review_record"]["surface"] == "preflight"
    assert [(seat["seat_id"], seat["answer"]) for seat in summary["seats"]] == [("api-scout", _PREFLIGHT_ANSWER)]
    assert (output / "preflight.txt").read_text(encoding="utf-8") == _PREFLIGHT_ANSWER + "\n"
    assert _PREFLIGHT_ANSWER in capsys.readouterr().out
    # The look's record never stands in for the commit panel's.
    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["outcome"]["review_record_id"] != summary["preflight"]["record_id"]
    sections = shared.full_output_sections((output / "full-output.txt").read_text(encoding="utf-8"))
    verdict = json.loads(sections["AGGREGATE VERDICT"])
    assert verdict["review_record"]["surface"] == "commit_gate"
    assert (output / "reviewed-tree-drift.diff").exists()
    assert not checkouts.exists() or not any(checkouts.iterdir())


def test_operator_lane_refuses_a_preflight_row_that_is_not_enabled(tmp_path, monkeypatch):
    """A preflight row that is not an enabled catalog row is the caller's argument
    error: no test run, no look, no panel."""
    import scripts.run_external_review as module
    from tests.test_git_review_preflight_gate import _roster

    _roster(monkeypatch, enabled=False)
    order: list = []
    _operator_fixture(tmp_path, monkeypatch, order, scout_enabled=False)

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path, "--preflight-reviewer=api-scout")

    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert exit_code != 0 and outcome["outcome"]["status"] != "passed"
    assert "is not an enabled catalog row" in json.dumps(outcome)
    assert order == []


def test_operator_lane_retains_the_checkout_while_a_seat_is_open(tmp_path, monkeypatch):
    """R4 (operator lane): a reviewer seat still owed after the cycle keeps the
    isolated checkout alive for reconciliation, named in the typed outcome."""
    import ouroboros.review_substrate as substrate
    import scripts.run_external_review as module

    repo, _briefs = _operator_fixture(tmp_path, monkeypatch)
    settled = substrate.run_review_request

    def one_seat_open(request, *, slots, **kwargs):
        result = settled(request, slots=slots, **kwargs)
        for actor in result.actors:
            if actor["slot_id"] == "t2":
                actor.update(status="error", raw_text="", error="logical wait expired",
                             operation_state="in_flight", late_result_pending=True)
        return result

    monkeypatch.setattr(substrate, "run_review_request", one_seat_open)

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path)

    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert (exit_code, outcome["exit_code"], outcome["outcome"]["status"]) == (3, 3, "blocked")
    retained = Path(outcome["outcome"]["retained_checkout"])
    assert retained.is_dir() and retained.parent.parent == tmp_path / "drive" / "state" / "review_checkouts"
    reviewers = outcome["outcome"]["retained_custody"]["reviewers"]
    assert {row["slot_id"] for _surface, row in reviewers if row.get("late_result_pending")} == {"t2"}
    assert outcome["outcome"]["retention_reason"]
    assert shared.git(repo, "worktree", "list", "--porcelain").count("worktree ") == 2


def test_operator_lane_exits_on_the_verdict_of_a_failure_the_cycle_hands_back(tmp_path, monkeypatch):
    """The review-only cycle hands a FAIL back as ``reviewed`` (it never refuses on a verdict); the lane reads the
    record's verdict and exits 1 as critical findings, while a PASS exits 0
    (``test_operator_lane_is_the_commit_gate_dry_run_in_an_isolated_checkout``). Nothing pins an enforcement."""
    import ouroboros.review_substrate as substrate
    import scripts.run_external_review as module

    _operator_fixture(tmp_path, monkeypatch)
    settled = substrate.run_review_request

    def failing(request, *, slots, **kwargs):
        result = settled(request, slots=slots, **kwargs)
        for actor in result.actors:
            value = json.loads(actor["raw_text"])
            change = value if isinstance(value, list) else value["change"]
            (change or value["coupling"])[0].update(verdict="FAIL", severity="critical",
                                                   reason="README.md loses the staged edit")
            actor["raw_text"] = json.dumps(value, ensure_ascii=False)
        return result

    monkeypatch.setattr(substrate, "run_review_request", failing)

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path)

    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert (exit_code, outcome["exit_code"]) == (1, 1), outcome
    assert (outcome["outcome"]["aggregate"], outcome["outcome"]["block_reason"]) == ("FAIL", "critical_findings")


def test_operator_lane_without_isolation_reviews_this_worktree(tmp_path, monkeypatch):
    import scripts.run_external_review as module

    repo, briefs = _operator_fixture(tmp_path, monkeypatch)

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path, "--no-isolated-checkout")

    assert exit_code == 0, (output / "outcome.json").read_text(encoding="utf-8")
    assert {Path(brief["session_root"]) for brief in briefs if brief["session_root"]} == {repo}
    assert not (tmp_path / "drive" / "state" / "review_checkouts").exists()


# Cyrillic, a supplementary-plane emoji and a BMP arrow: neither cp1252 nor ASCII has any of them.
_UNICODE_REASON = "прочитано целиком 🐍 → готово"


def _unicode_review(tmp_path: Path, monkeypatch) -> tuple[str, str]:
    """The operator fixture with the named preflight look; that seat and panel seat t1 answer in Unicode."""
    from tests.test_git_review_preflight_gate import _roster

    def answer(seat: str) -> str:
        return json.dumps([{"item": "code_quality", "verdict": "PASS", "severity": "advisory",
                            "reason": f"{seat}: {_UNICODE_REASON}"}], ensure_ascii=False)

    _roster(monkeypatch)
    monkeypatch.setitem(shared.ANSWERS, "t1", answer("t1"))
    _operator_fixture(tmp_path, monkeypatch, preflight_answer=answer("api-scout"))
    return answer("t1"), answer("api-scout")


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", "ascii"])
def test_a_console_that_cannot_encode_the_review_shows_escapes_and_the_artifacts_keep_it_exact(
        tmp_path, monkeypatch, encoding):
    """Windows opens a piped stdout in its ANSI code page (cp1252, errors=strict), and a review carries text
    no such page holds (a reviewer's Cyrillic or emoji, the ⚠️ of an omission notice). The console shows it
    as backslash escapes, the UTF-8 artifacts keep the exact text and the exit code stays the verdict's; a
    UTF-8 console shows the text itself (#2464: a passed review died at ``print`` before full-output.txt
    and outcome.json existed)."""
    import scripts.run_external_review as module

    t1, preflight = _unicode_review(tmp_path, monkeypatch)
    # The streams Python opens on such a pipe: stdout strict, stderr backslashreplace.
    stdout, stderr = (io.TextIOWrapper(io.BytesIO(), encoding=encoding, errors=errors)
                      for errors in ("strict", "backslashreplace"))
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path, "--preflight-reviewer=api-scout")
    stdout.flush()
    stderr.flush()
    shown, said = (stream.buffer.getvalue().decode(encoding) for stream in (stdout, stderr))
    assert exit_code == 0, said[-3000:]
    assert json.loads((output / "outcome.json").read_text(encoding="utf-8"))["exit_code"] == 0
    sections = shared.full_output_sections((output / "full-output.txt").read_text(encoding="utf-8"))
    seats = json.loads(sections["REVIEW SEAT RECORDS (ledger rows with retained answers, full, untruncated)"])
    assert {seat["seat_id"]: seat["answer"] for seat in seats}["t1"] == t1
    assert (output / "preflight.txt").read_text(encoding="utf-8") == preflight + "\n"
    assert "PREFLIGHT REVIEW" in shown and "AGGREGATE VERDICT" in shown and "Artifacts: " in said
    if encoding == "utf-8":
        assert f"t1: {_UNICODE_REASON}" in shown and f"api-scout: {_UNICODE_REASON}" in shown
    else:
        assert "\\U0001f40d" in shown and "\\u2192" in shown and "\\u043f" in shown and "🐍" not in shown


def test_a_console_that_fails_after_the_review_raises_with_every_artifact_already_written(tmp_path, monkeypatch):
    """A finished run saves everything before it shows anything: a console that fails (a reader that closed
    the pipe, as ``| head`` does) raises to the caller, never swallowed, after outcome.json, full-output.txt
    and the preflight record already hold the verdict."""
    import scripts.run_external_review as module

    class ClosedPipe(io.StringIO):
        def write(self, _text):
            raise BrokenPipeError(errno.EPIPE, "the reader closed the pipe")

    _t1, preflight = _unicode_review(tmp_path, monkeypatch)
    with contextlib.redirect_stdout(ClosedPipe()), pytest.raises(BrokenPipeError):
        _run_operator_lane(module, monkeypatch, tmp_path, "--preflight-reviewer=api-scout")
    output = tmp_path / "out-0"  # the folder _run_operator_lane named
    outcome = json.loads((output / "outcome.json").read_text(encoding="utf-8"))
    assert outcome["exit_code"] == 0 and outcome["outcome"]["status"] == "passed"
    verdict = json.loads(shared.full_output_sections(
        (output / "full-output.txt").read_text(encoding="utf-8"))["AGGREGATE VERDICT"])
    assert verdict["exit_code"] == 0 and verdict["review_record"]["record_id"] == outcome["outcome"]["review_record_id"]
    assert (output / "preflight.txt").read_text(encoding="utf-8") == preflight + "\n"


def _fresh_operator_lane(root: str) -> None:
    """The child of the fresh-process test below: one operator lane, then the same
    process's proof binding read by the record's own reader. Facts land in ``facts.json``."""
    cold = "ouroboros.tools.browser" not in sys.modules
    from pytest import MonkeyPatch

    import scripts.run_external_review as module
    from ouroboros import review_ledger
    from ouroboros.tools.commit_gate import _review_tests_facts
    from ouroboros.tools.registry import ToolContext

    tmp = Path(root)
    with MonkeyPatch.context() as monkeypatch:
        repo, _briefs = _operator_fixture(tmp, monkeypatch)
        exit_code, output = _run_operator_lane(module, monkeypatch, tmp)
        record_id = json.loads((output / "outcome.json").read_text(encoding="utf-8"))["outcome"]["review_record_id"]
        browsers_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
        facts = {"cold": cold, "exit_code": exit_code, "browsers_path": browsers_path,
                 "record": review_ledger.load_record(tmp / "drive", record_id)["tests"]}
        ctx = ToolContext(repo_dir=repo, drive_root=tmp / "binding-drive")
        shared.passing_test_runner(ctx)
        readme = repo / "README.md"
        staged = readme.read_text(encoding="utf-8")
        facts["same"] = _review_tests_facts(ctx, True)
        readme.write_text("other worktree bytes\n", encoding="utf-8")
        facts["worktree"] = _review_tests_facts(ctx, True)
        readme.write_text("other index bytes\n", encoding="utf-8")
        shared.git(repo, "add", "README.md")
        readme.write_text(staged, encoding="utf-8")  # only the index differs from the tested candidate
        facts["index"] = _review_tests_facts(ctx, True)
        shared.git(repo, "add", "README.md")
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(tmp / "other-browsers"))
        facts["other_browsers_path"] = _review_tests_facts(ctx, True)
        if browsers_path is None:  # the stub armed nothing; the parent reports that fact first
            monkeypatch.delenv("PLAYWRIGHT_BROWSERS_PATH")
        else:
            monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", browsers_path)
        facts["restored"] = _review_tests_facts(ctx, True)
    (tmp / "facts.json").write_text(json.dumps(facts), encoding="utf-8")


@pytest.mark.parametrize("browsers_path", ["packaged", "explicit"])
def test_a_fresh_operator_lane_binds_its_test_proof_after_the_tool_catalog(tmp_path, browsers_path):
    """#2363 in a fresh interpreter whose Playwright ships a browser (a stub package shadows
    the installed one). A worker's registry imports every tool module before its gate mints
    a test proof, so the browser module's packaged-browser default is already part of the
    environment that proof binds. The operator lane loads the catalog first as well. Otherwise
    the panel's first registry changes that environment, and the record says the green run of
    this very candidate does not cover it. An explicit owner path stays as given, and the
    binding stays strict: another worktree, index or browser path is not this candidate."""
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_environment import isolated_environment, retain_tree

    site = tmp_path / "site"
    bundle = site / "playwright" / "driver" / "package" / ".local-browsers" / "webkit-1"
    bundle.mkdir(parents=True)
    (bundle / "pw_run.sh").write_text("", encoding="utf-8")
    (site / "playwright" / "__init__.py").write_text("", encoding="utf-8")
    env = isolated_environment(tmp_path / "env", REPO)
    env.pop("PLAYWRIGHT_BROWSERS_PATH", None)
    env["PYTHONPATH"] = os.pathsep.join([str(site), str(REPO)])
    owner_path = str(tmp_path / "owner-browsers")
    if browsers_path == "explicit":
        env["PLAYWRIGHT_BROWSERS_PATH"] = owner_path
    child = ("import sys\nfrom tests.test_external_review_operator_lane import _fresh_operator_lane\n"
             "_fresh_operator_lane(sys.argv[1])\n")
    container = ProcessContainer(lifeline=True, overload=True)
    proc = None
    try:
        proc = container.spawn([sys.executable, "-c", child, str(tmp_path / "lane")], cwd=REPO, env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output, _ = proc.communicate(timeout=240)
    finally:
        error = container.reap()
        container.close()
        if proc is not None:
            proc.wait(timeout=5)
        if error:
            retain_tree(tmp_path, error)
    assert not error, error
    assert proc.returncode == 0, output.decode(errors="replace")[-6000:]
    facts = json.loads((tmp_path / "lane" / "facts.json").read_text(encoding="utf-8"))
    bound = {"policy": "run", "result": "passed", "proof": "candidate_bound"}
    stale = {"policy": "NOT_RUN", "result": "unknown", "reason": "tests_proof_not_for_this_candidate"}
    assert facts["cold"] and facts["exit_code"] == 0, facts
    assert facts["browsers_path"] == ("0" if browsers_path == "packaged" else owner_path), facts
    assert facts["record"] == bound, facts
    assert facts["same"] == facts["restored"] == bound, facts
    assert facts["worktree"] == facts["index"] == facts["other_browsers_path"] == stale, facts
