"""A preflight is ``review_change(subject=worktree, surface=preflight, reviewers=[one row])``.

Decision 3A: the author's early look is the same action as any review, over the live
worktree of the system repository, with ONE enabled catalog row the author names (a
review-pool member or not) as the whole panel. The surface is part of every identity,
so a settled preflight record answers only a later preflight of the same subject,
never a ``surface=change`` wave over it (nor the commit panel); an unknown or disabled
row is refused before any wave. The ``preflight_review`` tool is a thin wrapper over
the same call, ``advisory_review`` stays its alias, and the old pipelines are gone.
"""
from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from ouroboros.tools import commit_gate
from ouroboros.tools import preflight_review as pr
from ouroboros.tools import review_change as rc
from tests.test_git_review_preflight_gate import _roster
from tests.test_review_change_tool import Harness, _pool, h  # noqa: F401
from tests.test_review_change_end_to_end import staged_body  # noqa: F401


def _edit(harness: Harness, text: str = "preflight edit\n") -> None:
    (harness.system / "system.txt").write_text(text, encoding="utf-8")


def _look(harness: Harness, reviewer: str, **args):
    return harness.run(root="system_repo", subject="worktree", surface="preflight", reviewers=[reviewer], **args)


@pytest.mark.parametrize("member", [False, True], ids=["catalog-row", "pool-seat"])
def test_one_named_row_is_the_whole_panel(h: Harness, monkeypatch, member) -> None:  # noqa: F811
    _roster(monkeypatch)
    reviewer = _pool()[0] if member else "api-scout"
    _edit(h)
    result = _look(h, reviewer, goal="An early look")

    [call] = h.wave.calls
    assert (call.triad, call.coupling) == ([reviewer], [])
    assert (call.subject.spec.kind, call.subject.spec.surface) == ("worktree", "preflight")
    record = rl.load_record(h.drive, result["record_id"])
    assert record["surface"] == "preflight"
    assert [seat["seat_id"] for seat in record["rows"]] == [reviewer]
    assert (result["panel"]["composition"], result["panel"]["chosen_by"]) == ("composed", "author")


@pytest.mark.parametrize("reviewer, aggregate", [("t1", "NOT_PERFORMED"), ("t2", "PASS"), ("s1", "PASS")])
@pytest.mark.serial
def test_preflight_delivers_one_countercheck_through_the_real_review_wave(
        staged_body, tmp_path, monkeypatch, reviewer, aggregate):  # noqa: F811
    from ouroboros.tools.review_helpers import anti_pattern_lock_guard
    from ouroboros.tools.registry import ToolContext
    from tests.test_review_change_end_to_end import GOAL, SCOPE, _brief_text, shared, substrate

    sent: list[dict] = []
    monkeypatch.setattr(substrate, "run_review_request", shared.golden_substrate(sent))
    ctx = ToolContext(repo_dir=staged_body["repo"], drive_root=tmp_path / "preflight-drive")
    result = rc.run_review_change(ctx, root="system_repo", surface="preflight", subject="worktree",
                               reviewers=[reviewer], goal=GOAL, scope=SCOPE)

    # A packet-only row receives the countercheck but cannot answer coupling.
    assert result["aggregate"] == aggregate and result["state"] == "settled", result
    assert result["per_question"]["change"] == "PASS"
    [given] = sent
    assert given["slot_id"] == reviewer
    text = _brief_text(given)
    assert text.count(anti_pattern_lock_guard("body").strip()) == 1
    assert "deliberate SECOND pass" not in text
    assert "Coupling affects only unchanged code outside the diff" not in " ".join(text.split())


def test_the_preflight_record_never_answers_a_change_wave_over_the_same_subject(h: Harness, monkeypatch) -> None:  # noqa: F811
    _roster(monkeypatch)
    _edit(h)
    first = _look(h, "api-scout")
    again = _look(h, "api-scout")
    assert len(h.wave.calls) == 1 and again["reused"] and again["record_id"] == first["record_id"]

    change = h.run(root="system_repo", subject="worktree", reviewers=["api-scout"])
    assert len(h.wave.calls) == 2, "a surface=change wave over the same worktree is new paid work"
    assert change["record_id"] != first["record_id"] and not change["reused"]
    keys = {rl.load_record(h.drive, rid)["fingerprints"]["reuse_key"] for rid in (first["record_id"], change["record_id"])}
    assert len(keys) == 2

    last = _look(h, "api-scout")
    assert len(h.wave.calls) == 2 and last["record_id"] == first["record_id"], "nor the other way round"


@pytest.mark.parametrize("enabled", [False, None], ids=["disabled", "unknown"])
def test_an_unknown_or_disabled_row_is_refused_before_any_wave(h: Harness, monkeypatch, enabled) -> None:  # noqa: F811
    if enabled is not None:
        _roster(monkeypatch, enabled=enabled)
    _edit(h)
    text = h.tool(root="system_repo", subject="worktree", surface="preflight", reviewers=["api-scout"])
    assert text.startswith("⚠️ TOOL_ARG_ERROR (review_change): ") and "is not an enabled catalog row" in text
    assert h.wave.calls == [] and h.written == {}


def test_an_unparseable_answer_is_performed_never_a_pass(h: Harness, monkeypatch) -> None:  # noqa: F811
    _roster(monkeypatch)
    _edit(h)
    h.wave.status = "parse_failure"
    fact = commit_gate.run_commit_preflight(h.ctx, "api-scout", commit_message="m", goal="", scope="", review_rebuttal="")
    assert fact["status"] == "performed" and fact["record_id"] and fact["aggregate"] != rl.VERDICT_PASS
    [seat] = rl.load_record(h.drive, fact["record_id"])["rows"]
    assert seat["seat_id"] == "api-scout" and seat["status"] == "parse_failure"


def test_a_crashed_look_is_still_performed_and_budget_stops_propagate(monkeypatch) -> None:
    from ouroboros.usage_accounting import BudgetExceeded

    ctx = SimpleNamespace(emit_progress_fn=lambda *a: None)
    monkeypatch.setattr(rc, "run_review_change", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("wave crashed")))
    fact = commit_gate.run_commit_preflight(ctx, "api-scout", commit_message="m", goal="", scope="", review_rebuttal="")
    assert fact == {"status": "performed", "record_id": "", "reviewer": "api-scout", "aggregate": "",
                    "error": "RuntimeError: wave crashed"}
    monkeypatch.setattr(rc, "run_review_change", lambda *a, **kw: (_ for _ in ()).throw(BudgetExceeded("no funds")))
    with pytest.raises(BudgetExceeded):
        commit_gate.run_commit_preflight(ctx, "api-scout", commit_message="m", goal="", scope="", review_rebuttal="")


# --- the ``preflight_review`` tool -----------------------------------------------------

def _ctx(tmp_path):
    return SimpleNamespace(repo_dir=tmp_path, drive_root=tmp_path, task_id="t", emit_progress_fn=lambda *a: None)


def test_the_tool_is_review_change_with_the_one_named_row(tmp_path, monkeypatch) -> None:
    _roster(monkeypatch)
    calls = []
    monkeypatch.setattr(rc, "_handle_review_change", lambda ctx, **kw: calls.append(kw) or "the record")
    ctx = _ctx(tmp_path)
    assert pr._handle_preflight_review(ctx, reviewer="api-scout", commit_message="fix: x", scope="s",
                                       review_rebuttal="r") == "the record"
    pr._handle_preflight_review(ctx, reviewer=" api-scout ", commit_message="fix: x", goal="the goal")
    base = dict(root="system_repo", subject="worktree", surface="preflight", reviewers=["api-scout"])
    assert calls == [{**base, "goal": "fix: x", "scope": "s", "review_rebuttal": "r"},
                     {**base, "goal": "the goal", "scope": "", "review_rebuttal": ""}]


@pytest.mark.parametrize("reviewer, needle", [("", "reviewer is required"), ("nobody", "is not an enabled catalog row")])
def test_the_tool_refuses_without_an_enabled_row(tmp_path, monkeypatch, reviewer, needle) -> None:
    monkeypatch.setattr(rc, "_handle_review_change", lambda *a, **kw: pytest.fail("no wave without an enabled row"))
    text = pr._handle_preflight_review(_ctx(tmp_path), reviewer=reviewer, commit_message="m")
    assert text.startswith("⚠️ TOOL_ARG_ERROR (preflight_review): ") and needle in text
    assert text.endswith("No reviewer was dispatched.")


def test_deterministic_only_is_the_free_diagnostics_and_never_a_wave(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rc, "_handle_review_change", lambda *a, **kw: pytest.fail("deterministic_only pays no reviewer"))
    monkeypatch.setattr(commit_gate, "release_diagnostics", lambda ctx, paths, source: {"source": source, "paths": paths})
    out = json.loads(pr._handle_preflight_review(_ctx(tmp_path), deterministic_only=True, source="index", paths=["VERSION"]))
    assert out == {"source": "index", "paths": ["VERSION"]}


def test_advisory_review_is_an_alias_of_the_same_wrapper() -> None:
    entries = {entry.name: entry for entry in pr.get_tools()}
    assert set(entries) == {"preflight_review", "advisory_review", "review_status"}
    alias, tool = entries["advisory_review"], entries["preflight_review"]
    assert alias.alias_for == "preflight_review" and alias.handler is tool.handler is pr._handle_preflight_review
    assert alias.schema["parameters"] == tool.schema["parameters"]
    assert tool.timeout_sec == alias.timeout_sec == rc._review_change_tool_timeout_sec()
    assert "skip_tests" not in tool.schema["parameters"]["properties"], "the wrapper runs no tests"


def test_the_pipelines_are_gone_and_the_wrapper_is_registered_once() -> None:
    from ouroboros.tools.registry_core import ToolRegistry

    for name in ("claude_advisory_review", "preflight_review_run", "preflight_review_prompt"):
        assert importlib.util.find_spec(f"ouroboros.tools.{name}") is None, name
    modules = ToolRegistry._FROZEN_TOOL_MODULES
    assert modules.count("preflight_review") == 1
    assert not {"claude_advisory_review", "preflight_review_run"} & set(modules)
