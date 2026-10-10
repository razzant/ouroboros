"""The operator lane of the external review wrapper: the commit gate's own
review-only cycle, run in the runtime's isolated checkout of the staged index
(R3), with its hermetic tests, the author's optional ``surface=preflight`` look
(record + the helper's full answer), drift artifact and custody-bound retention."""

import json
from pathlib import Path
from types import SimpleNamespace

from tests import _contributor_packet_shared as shared

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


def _operator_fixture(tmp_path: Path, monkeypatch, order: list | None = None, *,
                      scout_enabled: bool = True) -> tuple[Path, list[dict]]:
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
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking")
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
            "slot_id": "api-scout", "model": slots[0].model, "status": "ok", "raw_text": _PREFLIGHT_ANSWER,
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


def test_operator_lane_without_isolation_reviews_this_worktree(tmp_path, monkeypatch):
    import scripts.run_external_review as module

    repo, briefs = _operator_fixture(tmp_path, monkeypatch)

    exit_code, output = _run_operator_lane(module, monkeypatch, tmp_path, "--no-isolated-checkout")

    assert exit_code == 0, (output / "outcome.json").read_text(encoding="utf-8")
    assert {Path(brief["session_root"]) for brief in briefs if brief["session_root"]} == {repo}
    assert not (tmp_path / "drive" / "state" / "review_checkouts").exists()
