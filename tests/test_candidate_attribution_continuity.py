"""#2414: a body candidate's commit attribution survives late attachment, Restart and Continue.

Real Git, the real candidate registry and tool dispatch, the host's own baseline capture
(``OuroborosAgent._capture_mutation_baseline``) and the commit gate's attribution consumer
(``tools.git._task_attributed_commit_paths``). One controlled review stub stands in for paid
seats; the brief they would read is assembled by the real packet builder.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import body_candidate
from ouroboros.mutation_attribution import capture_mutation_baseline
from ouroboros.task_results import load_task_result, write_task_result
from tests.body_candidate_support import candidate_commit, git, isolate, make_ctx, make_serving

EDIT = "ouroboros/mod_a.py"


@pytest.fixture
def body(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    return serving, data


def _start(serving, data, task_id, *, direct=False, predecessor=""):
    """The host's root start; a direct turn records no baseline (its explicit/whole-tree contract)."""
    from ouroboros.agent import OuroborosAgent

    if not load_task_result(data, task_id):
        write_task_result(data, task_id, "running")
    task = {"id": task_id, "root_task_id": task_id, "_execution_owner": {"kind": "direct" if direct else "pooled"},
            **({"predecessor_authority": {"source": {"task_id": predecessor}}} if predecessor else {})}
    agent = SimpleNamespace(env=SimpleNamespace(repo_dir=serving, drive_root=data, budget_drive_root=data))
    OuroborosAgent._capture_mutation_baseline(agent, task, {"root_task_id": task_id})


def _ctx(serving, data, task_id, **overrides):
    return make_ctx(serving, data, task_id, budget_drive_root=data, **overrides)


def _registry(serving, data, ctx):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=serving, drive_root=data)
    registry.set_context(ctx)
    return registry


def _select(ctx, paths=None):
    from ouroboros.tools import git as git_tools

    selected, facts, error, _binding = git_tools._task_attributed_commit_paths(ctx, paths)
    return selected, facts or {}, error


def _baseline(data, task_id):
    return load_task_result(data, task_id)["mutation_evidence"]["baseline"]


def _surface(data, task_id, candidate):
    root = str(pathlib.Path(candidate).resolve())
    return next(row for row in _baseline(data, task_id)["surfaces"] if row["canonical_root"] == root)


def _write(path, rel, text):
    target = pathlib.Path(path) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Late attachment: a direct turn continued in a worker (same task id)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", ["restore_then_capture", "capture_then_restore"])
def test_direct_turn_continued_in_a_worker_commits_its_candidate_edits(body, order):
    serving, data = body
    _start(serving, data, "turn-1", direct=True)
    turn = _ctx(serving, data, "turn-1")
    registry = _registry(serving, data, turn)
    receipt = registry.execute("prepare_self_change", {})
    registry.execute("write_file", {"root": "system_repo", "path": EDIT, "content": "GEN = 'TASK_EDIT'\n"})
    assert _select(turn) == (None, {}, "")  # the direct turn keeps its explicit/whole-tree contract

    worker = _ctx(serving, data, "turn-1")  # continue_in_background: the same id in a pooled worker
    if order == "restore_then_capture":  # the order before this repair; the commit consumer heals it
        body_candidate.restore(worker)
        _start(serving, data, "turn-1")
    else:
        _start(serving, data, "turn-1")
        body_candidate.restore(worker)
    selected, facts, error = _select(worker)
    assert not error and selected == [EDIT], error
    assert facts["copy"]["basis"]["kind"] == "derived" and facts["excluded_preexisting_dirty"] == []
    candidate = pathlib.Path(worker.repo_dir)
    assert _surface(data, "turn-1", candidate)["git"]["base_commit"] == git(serving, "rev-parse", "HEAD")

    later = _ctx(serving, data, "turn-1")  # yet another worker: no late capture turns the edit foreign
    body_candidate.restore(later)
    assert _select(later)[0] == [EDIT]
    assert git(serving, "status", "--porcelain") == ""
    assert "no host mutation baseline" in receipt, receipt  # the direct turn was told which contract it had


def test_same_id_restart_with_its_baseline_keeps_the_record_byte_identical(body):
    serving, data = body
    _start(serving, data, "root-r")
    first = _ctx(serving, data, "root-r")
    _registry(serving, data, first).execute("prepare_self_change", {})
    _write(first.repo_dir, EDIT, "GEN = 'MINE'\n")
    before = _baseline(data, "root-r")

    restarted = _ctx(serving, data, "root-r")
    _start(serving, data, "root-r")
    body_candidate.restore(restarted)
    assert _baseline(data, "root-r") == before
    assert _select(restarted)[0] == [EDIT]


# --------------------------------------------------------------------------- #
# Transfer from an owner killed without a terminal snapshot
# --------------------------------------------------------------------------- #
def _interrupted_owner(serving, data):
    _start(serving, data, "pred")
    bound = body_candidate.prepare(_ctx(serving, data, "pred"))
    _write(bound["path"], EDIT, "GEN = 'PRESERVED_EDIT'\n")
    write_task_result(data, "pred", "failed", result="worker killed")  # no terminal mutation snapshot
    return bound, load_task_result(data, "pred")["mutation_evidence"]


@pytest.mark.parametrize("via", ["continuation", "explicit_resume"])
def test_interrupted_owner_transfer_carries_its_recorded_basis(body, via):
    serving, data = body
    bound, predecessor_evidence = _interrupted_owner(serving, data)
    continuation = via == "continuation"
    _start(serving, data, "succ", predecessor="pred" if continuation else "")
    succ = _ctx(serving, data, "succ", task_contract={"predecessor_authority": {"source": {"task_id": "pred"}}}
                if continuation else {})
    receipt = _registry(serving, data, succ).execute(
        "prepare_self_change", {} if continuation else {"resume": bound["candidate_id"]})

    selected, facts, error = _select(succ)
    assert not error and selected == [EDIT], error
    assert load_task_result(data, "pred")["mutation_evidence"] == predecessor_evidence
    assert facts["copy"]["basis"] == {"kind": "carried", "source": "task pred", "task_id": "pred",
                                      "from_basis": "original"}
    assert "carried from task pred" in receipt and "does not prove" in receipt, receipt


def test_commit_consumer_reattaches_after_a_failed_attach_and_stays_idempotent(body, monkeypatch):
    from ouroboros import mutation_attribution

    serving, data = body
    _interrupted_owner(serving, data)
    _start(serving, data, "succ", predecessor="pred")
    succ = _ctx(serving, data, "succ", task_contract={"predecessor_authority": {"source": {"task_id": "pred"}}})
    real, failures = mutation_attribution.capture_mutation_baseline, []

    def unavailable_once(*args, **kwargs):
        if not failures:
            failures.append(1)
            raise OSError("results store unavailable")
        return real(*args, **kwargs)

    monkeypatch.setattr(mutation_attribution, "capture_mutation_baseline", unavailable_once)
    receipt = _registry(serving, data, succ).execute("prepare_self_change", {})
    assert body_candidate.find("succ")["owners"] == ["pred", "succ"]  # ownership had already moved

    selected, facts, error = _select(succ)
    assert not error and selected == [EDIT], error
    assert facts["copy"]["basis"]["kind"] == "carried"
    assert "was not recorded (OSError: results store unavailable)" in receipt, receipt  # not only a log line
    settled = _baseline(data, "succ")
    again = _ctx(serving, data, "succ")
    body_candidate.restore(again)
    assert _select(again)[0] == [EDIT] and _baseline(data, "succ") == settled


def test_a_predecessors_later_observation_is_not_carried_as_its_start(body):
    serving, data = body
    bound = body_candidate.prepare(_ctx(serving, data, "pred"))  # a direct turn: no baseline yet
    _write(bound["path"], EDIT, "GEN = 'PRESERVED_EDIT'\n")
    _start(serving, data, "pred")  # continued by the earlier release, which recorded the copy late
    capture_mutation_baseline(data, "pred", [{"surface_type": "system_repo", "host_root": bound["path"]}],
                              owner_kind="task_root", owner_id="pred")
    write_task_result(data, "pred", "failed", result="worker killed")
    predecessor_evidence = load_task_result(data, "pred")["mutation_evidence"]
    _start(serving, data, "succ", predecessor="pred")
    succ = _ctx(serving, data, "succ", task_contract={"predecessor_authority": {"source": {"task_id": "pred"}}})
    body_candidate.prepare(succ)

    selected, facts, error = _select(succ)
    assert not error and selected == [EDIT], error
    assert load_task_result(data, "pred")["mutation_evidence"] == predecessor_evidence
    assert facts["copy"]["basis"]["kind"] == "derived"


# --------------------------------------------------------------------------- #
# Rows recorded by the earlier release
# --------------------------------------------------------------------------- #
def _late_row(serving, data, task_id, *, commit_first=False, dirty=(EDIT,), serving_row=True):
    """A direct turn's candidate that a later rebind recorded the old way: today's dirt as its start.
    ``serving_row=False``: the task never recorded an observation of the serving checkout."""
    turn = _ctx(serving, data, task_id)
    bound = body_candidate.prepare(turn)
    candidate = pathlib.Path(bound["path"])
    if commit_first:
        body_candidate.record_reviewed_commit(turn, candidate_commit(candidate, files={"ouroboros/mod_b.py": "GEN = 'C1'\n"}))
    for rel in dirty:
        _write(candidate, rel, "GEN = 'TASK_EDIT'\n")
    if serving_row:
        _start(serving, data, task_id)
    capture_mutation_baseline(data, task_id, [{"surface_type": "system_repo", "host_root": str(candidate)}],
                              owner_kind="task_root", owner_id=task_id)
    return candidate, _surface(data, task_id, candidate)


def test_a_late_dirty_row_of_the_same_creation_is_superseded_and_kept_whole_in_history(body):
    serving, data = body
    candidate, late = _late_row(serving, data, "turn-2")
    assert late["git"]["dirty_paths"] == [EDIT] and "basis" not in late

    worker = _ctx(serving, data, "turn-2")
    body_candidate.restore(worker)
    selected, facts, error = _select(worker)
    assert not error and selected == [EDIT], error
    assert facts["copy"]["basis"]["kind"] == "derived"
    superseded = [row["superseded"] for ext in _baseline(data, "turn-2").get("extensions") or []
                  for row in ext["surfaces"] if "superseded" in row]
    assert superseded == [late]  # the earlier observation, fingerprints included, is never erased


def test_an_unresolved_later_observation_is_kept_and_reaches_review_only_when_named(body):
    serving, data = body
    candidate, late = _late_row(serving, data, "turn-3", commit_first=True)
    assert late["git"]["base_commit"] != body_candidate.find("turn-3")["base_sha"]
    _write(candidate, EDIT, "GEN = 'EDITED_AGAIN'\n")  # its recorded dirt changed: no longer a veto
    _write(candidate, "ouroboros/added.py", "GEN = 'NEW'\n")

    worker = _ctx(serving, data, "turn-3")
    restored = body_candidate.restore(worker)
    selected, facts, error = _select(worker)
    assert not error and selected == ["ouroboros/added.py"], error
    assert facts["selection"]["left_out"] == [EDIT] and facts["excluded_preexisting_dirty"] == []

    selected, facts, error = _select(worker, [EDIT, "ouroboros/added.py"])
    assert not error and selected == ["ouroboros/added.py", EDIT], error
    assert facts["selection"]["uncertain"] == [EDIT]
    row = _surface(data, "turn-3", candidate)  # the observation itself is preserved; it now names its copy
    assert {key: row[key] for key in late} == late and row["candidate"]["candidate_id"] == "body_turn-3"
    assert restored["attribution"] == {"state": "attached", "basis": {"kind": "unrecorded"}, "uncertain": 1}


def test_outcome_remainder_counts_uncertain_copy_dirt_and_never_calls_it_foreign(body):
    from ouroboros.agent_task_pipeline import _attach_host_mutation_projection
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.project_dialogue import _completion_verdict

    serving, data = body
    candidate, late = _late_row(serving, data, "turn-4", commit_first=True)  # kept: an unresolved observation
    _write(candidate, EDIT, "GEN = 'EDITED_AGAIN'\n")
    trace = {}
    _attach_host_mutation_projection(SimpleNamespace(drive_root=data, budget_drive_root=data),
                                     {"id": "turn-4", "root_task_id": "turn-4", "budget_drive_root": str(data)}, trace)
    projection = trace["mutation_attribution"]
    assert projection["uncommitted_remainder"]["state"] == "known", projection
    assert projection["uncommitted_remainder"]["count"] == 1
    assert "preexisting_dirty_changed" not in projection["blockers"] and "attribution_uncertain" in projection["blockers"]
    snapshot = load_task_result(data, "turn-4")["mutation_evidence"]["terminal_candidate_snapshot"]
    row = next(item for item in snapshot["surfaces"] if item["canonical_root"] == str(candidate.resolve()))
    assert row["uncertain"] == [EDIT] and row["excluded_preexisting_dirty"] == [] and row["candidates"] == []
    outcome = derive_loop_outcome("Saved the candidate.", {"tool_calls": 1}, {"mutation_attribution": projection})
    record = {"status": "completed", "outcome_axes": outcome["outcome_axes"], "reason_code": "final_message"}
    assert "1 uncommitted file remains in the saved candidate" in _completion_verdict(record, {})


def test_continue_work_order_observes_the_serving_checkout_not_a_copy_sorting_first(body, monkeypatch, tmp_path):
    from ouroboros.owner_continue import recorded_system_checkout, work_order_text

    serving, data = body
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "0-worktrees"))  # copies sort first
    bound = body_candidate.prepare(_ctx(serving, data, "turn-5"))  # a direct turn: no baseline yet
    candidate_commit(bound["path"], files={"ouroboros/mod_b.py": "GEN = 'C1'\n"})
    (serving / "ouroboros/gone.py").write_text("GEN = 'SERVING_MOVED'\n", encoding="utf-8")
    git(serving, "commit", "-qam", "serving moved")
    _start(serving, data, "turn-5")
    body_candidate.restore(_ctx(serving, data, "turn-5"))  # the copy joins with its creation base
    result = load_task_result(data, "turn-5")
    assert result["mutation_evidence"]["baseline"]["surfaces"][0]["canonical_root"] == str(pathlib.Path(bound["path"]).resolve())

    checkout = recorded_system_checkout(result)
    assert checkout["commit"] == git(serving, "rev-parse", "HEAD"), checkout
    text = work_order_text("turn-5", "worker interrupted", {}, checkout=checkout)
    assert f"system checkout: HEAD {git(serving, 'rev-parse', 'HEAD')}" in text


@pytest.mark.parametrize("serving_row", [True, False])
def test_a_kept_earlier_row_names_its_copy_and_never_stands_for_the_serving_checkout(body, monkeypatch, tmp_path,
                                                                                      serving_row):
    """A row recorded before rows named their copy is kept as recorded: its date, observation and uncertain
    dirt. Attaching the copy joins its identity, so a Continue work order never reports the copy's HEAD as
    the serving checkout, and a task that never observed the serving checkout reports none."""
    from ouroboros.owner_continue import recorded_system_checkout, work_order_text

    serving, data = body
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "0-worktrees"))  # copies sort first
    candidate, late = _late_row(serving, data, "turn-6", commit_first=True, serving_row=serving_row)
    worker = _ctx(serving, data, "turn-6")
    body_candidate.restore(worker)

    checkout = recorded_system_checkout(load_task_result(data, "turn-6"))
    text = work_order_text("turn-6", "worker interrupted", {}, checkout=checkout)
    if serving_row:
        assert checkout.get("commit") == git(serving, "rev-parse", "HEAD"), (checkout, git(candidate, "rev-parse", "HEAD"))
        assert f"system checkout: HEAD {checkout['commit']}" in text
    else:
        assert checkout == {} and "system checkout" not in text, checkout
    row = _surface(data, "turn-6", candidate)
    identity = body_candidate._identity(body_candidate.find("turn-6"))
    assert row.get("candidate") == identity and {key: row[key] for key in late} == late, row
    assert _baseline(data, "turn-6")["extensions"][-1]["surfaces"] == [
        {"surface_type": "system_repo", "canonical_root": row["canonical_root"], "candidate": identity}]
    facts = _select(worker)[1]
    assert facts["uncertain"] == [EDIT] and facts["copy"]["basis"] == {"kind": "unrecorded"}  # still uncertain


def test_a_reused_path_names_another_copy_by_identity(body, monkeypatch):
    import shutil

    from ouroboros import mutation_attribution

    serving, data = body
    _start(serving, data, "root-p")
    first = body_candidate.prepare(_ctx(serving, data, "root-p"))
    _write(first["path"], EDIT, "GEN = 'LOST'\n")
    old = _surface(data, "root-p", first["path"])
    shutil.rmtree(first["path"])
    (serving / EDIT).write_text("GEN = 'NEXT'\n", encoding="utf-8")  # the next copy edits a moved file
    git(serving, "commit", "-qam", "serving moved on")

    real, failures = mutation_attribution.capture_mutation_baseline, []

    def unavailable_once(*args, **kwargs):  # the new copy's own record fails; the old one remains
        if not failures:
            failures.append(1)
            raise OSError("results store unavailable")
        return real(*args, **kwargs)

    monkeypatch.setattr(mutation_attribution, "capture_mutation_baseline", unavailable_once)
    again = _ctx(serving, data, "root-p")
    _registry(serving, data, again).execute("prepare_self_change", {})
    second = body_candidate.descriptor(again)
    assert second["path"] == first["path"] and second["branch"] != first["branch"]
    _write(second["path"], EDIT, "GEN = 'SECOND'\n")
    assert _select(again)[0] == [EDIT]  # the earlier copy's row no longer stands for this one
    row = _surface(data, "root-p", second["path"])
    assert row["git"]["base_commit"] == git(serving, "rev-parse", "HEAD")
    assert row["candidate"]["branch"] == second["branch"] != old["candidate"]["branch"]


# --------------------------------------------------------------------------- #
# Owner work outside the copy, and moved bases
# --------------------------------------------------------------------------- #
def test_owner_serving_dirt_stays_known_foreign_and_explicit_paths_do_not_widen_serving(body):
    serving, data = body
    (serving / "ouroboros/mod_b.py").write_text("GEN = 'OWNER_WIP'\n", encoding="utf-8")
    owner_bytes, owner_index = (serving / "ouroboros/mod_b.py").read_bytes(), git(serving, "ls-files", "-s")
    _start(serving, data, "root-c")
    cand = _ctx(serving, data, "root-c")
    body_candidate.prepare(cand)
    _write(cand.repo_dir, EDIT, "GEN = 'CANDIDATE'\n")
    assert _select(cand)[0] == [EDIT]
    assert (serving / "ouroboros/mod_b.py").read_bytes() == owner_bytes and git(serving, "ls-files", "-s") == owner_index

    _start(serving, data, "root-s")  # an unbound serving author
    in_place = _ctx(serving, data, "root-s")
    (serving / EDIT).write_text("GEN = 'IN_PLACE'\n", encoding="utf-8")
    selected, facts, error = _select(in_place, ["ouroboros/mod_b.py"])
    assert "Outside set: ouroboros/mod_b.py" in error and "copy" not in facts
    assert _select(in_place)[0] == [EDIT]
    (serving / EDIT).write_text("GEN = 'OFFICIAL'\n", encoding="utf-8")
    git(serving, "commit", "-q", "-m", "moved", EDIT)
    (serving / EDIT).write_text("GEN = 'IN_PLACE_AGAIN'\n", encoding="utf-8")
    selected, facts, error = _select(in_place, [EDIT])
    assert "GIT_ATTRIBUTION_BLOCKED" in error and "baseline_stale" in error and selected == []


@pytest.mark.parametrize("overlap", [True, False])
def test_a_moved_candidate_base_over_its_paths_needs_an_explicit_selection(body, overlap):
    serving, data = body
    _start(serving, data, "root-m")
    ctx = _ctx(serving, data, "root-m")
    receipt = _registry(serving, data, ctx).execute("prepare_self_change", {})
    bound = body_candidate.descriptor(ctx)
    official = EDIT if overlap else "ouroboros/mod_b.py"
    (serving / official).write_text("GEN = 'OFFICIAL'\n", encoding="utf-8")
    git(serving, "commit", "-qam", "official line moves")
    fresh = git(serving, "rev-parse", "HEAD")
    git(bound["path"], "merge", "--ff-only", "-q", fresh)
    _write(bound["path"], EDIT, "GEN = 'TASK_ON_FRESH_BASE'\n")

    selected, facts, error = _select(ctx)
    if not overlap:  # the existing disjoint re-anchor still heals automatically
        assert not error and selected == [EDIT] and _surface(data, "root-m", bound["path"])["git"]["base_commit"] == fresh
        return
    assert "baseline_stale" in error and selected == []
    automatic = error
    selected, facts, error = _select(ctx, [EDIT])
    assert not error and selected == [EDIT], error
    assert _surface(data, "root-m", bound["path"])["git"]["base_commit"] == bound["base_sha"]  # never re-anchored
    assert _select(ctx, [])[2].startswith("⚠️ GIT_NO_ATTRIBUTED_CHANGES")  # empty never means everything
    assert facts["selection"]["gaps"] == ["baseline_stale"] and "name its paths explicitly" in automatic
    assert "clean start is recorded" in receipt, receipt


# --------------------------------------------------------------------------- #
# What reviewers and the author receive, and what history keeps
# --------------------------------------------------------------------------- #
def _moved_candidate(serving, data, task_id):
    _start(serving, data, task_id)
    ctx = _ctx(serving, data, task_id)
    bound = body_candidate.prepare(ctx)
    (serving / EDIT).write_text("GEN = 'OFFICIAL'\n", encoding="utf-8")
    git(serving, "commit", "-qam", "official line moves")
    git(bound["path"], "merge", "--ff-only", "-q", git(serving, "rev-parse", "HEAD"))
    _write(bound["path"], EDIT, "GEN = 'TASK_ON_FRESH_BASE'\n")
    return ctx, bound


def test_the_assembled_review_brief_names_the_selected_compositions_gap(body, monkeypatch):
    import ouroboros.tools.review as review
    from tests.review_pool_rosters import set_review_pool

    serving, data = body
    ctx, bound = _moved_candidate(serving, data, "root-brief")
    _select(ctx, [EDIT])  # what commit_reviewed(paths=[...]) computes before staging
    git(bound["path"], "add", EDIT)
    captured = {}

    def panel(*_args, **kwargs):
        plan = kwargs.get("row_plan") or {}
        briefs = [task for task, parts in zip(plan.get("session_tasks") or [], plan.get("parts") or [])
                  if "coupling" in tuple(parts)]
        captured.update(prompt=kwargs["prompt"], task=briefs[0] if briefs else kwargs["session_task"])
        return json.dumps({"results": []})

    monkeypatch.setattr(review, "_preflight_check", lambda *_a, **_k: None)
    monkeypatch.setattr(review, "_load_checklist_section", lambda *_a, **_k: "checklist")
    monkeypatch.setattr(review, "load_governance_doc", lambda *_a, **_k: "governance")
    monkeypatch.setattr(review, "_handle_multi_model_review", panel)
    for delivery in ("packet", "native"):
        set_review_pool(monkeypatch, ["test/reviewer"], **({"delivery": "native"} if delivery == "native" else {}))
        for attr in ("_review_iteration_count", "_review_history", "_review_degraded_reasons"):
            setattr(ctx, attr, 0 if attr == "_review_iteration_count" else [])
        review._run_unified_review(ctx, "Carry the task edit onto the fresh base", goal="carry the edit")
        delivered = captured["prompt"] if delivery == "packet" else captured["task"]
        assert f"selected with uncertain attribution: {EDIT}" not in delivered  # a moved base, not a late row
        assert f"moved from the recorded base {bound['base_sha'][:12]}" in delivered, delivery
        assert "not re-anchored" in delivered


def _commit_until_handback(monkeypatch, reviewer):
    """The real commit gate up to the advisory handback; only seats, tests and publication are stand-ins."""
    from ouroboros import commit_admission
    from ouroboros.tools import git as git_tools
    from ouroboros.tools import review_helpers

    monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **kw: [])
    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics",
                        lambda *a, **kw: {"status": "not_applicable", "findings": [], "unavailable": []})
    monkeypatch.setattr(git_tools, "_managed_candidate_needs_proof", lambda ctx: False)
    monkeypatch.setattr(git_tools, "_run_review_preflight_tests", lambda *_a, **_kw: None)
    monkeypatch.setattr(git_tools, "_post_commit_result", lambda *_a, **_kw: None)
    monkeypatch.setattr(git_tools, "_run_parallel_review", reviewer)
    return git_tools._repo_commit_push


@pytest.mark.parametrize("consumer", ["brief", "handback"])
def test_every_uncertain_name_reaches_the_brief_and_the_author(body, monkeypatch, consumer):
    """Past twelve uncertain paths the brief and the handback still name each one, and never the
    attributable path: the same staged diff with another path uncertain reads differently."""
    import ouroboros.tools.review as review
    from ouroboros.mutation_attribution import selection_disclosure
    from ouroboros.review_dispatch import invoke_review_paid_stamp
    from ouroboros.review_ledger import CouplingOutcome
    from tests.review_pool_rosters import set_review_pool

    serving, data = body
    uncertain = [f"ouroboros/late_{index:02d}.py" for index in range(14)]
    candidate, _late = _late_row(serving, data, "turn-n", commit_first=True, dirty=uncertain)
    worker = _ctx(serving, data, "turn-n")
    body_candidate.restore(worker)
    _write(candidate, "ouroboros/added.py", "GEN = 'NEW'\n")  # attributable: created after the observation
    chosen = [*uncertain[:13], "ouroboros/added.py"]
    selected, facts, error = _select(worker, chosen)
    assert not error and facts["selection"]["uncertain"] == uncertain[:13], error
    disclosure = selection_disclosure(facts["selection"])
    assert all(name in disclosure for name in uncertain) and "ouroboros/added.py" not in disclosure, disclosure

    if consumer == "brief":
        git(candidate, "add", "--", *chosen)
        captured = {}
        monkeypatch.setattr(review, "_preflight_check", lambda *_a, **_k: None)
        monkeypatch.setattr(review, "_load_checklist_section", lambda *_a, **_k: "checklist")
        monkeypatch.setattr(review, "load_governance_doc", lambda *_a, **_k: "governance")
        monkeypatch.setattr(review, "_handle_multi_model_review",
                            lambda *_a, **kw: captured.update(prompt=kw["prompt"]) or json.dumps({"results": []}))
        set_review_pool(monkeypatch, ["test/reviewer"])
        for attr in ("_review_iteration_count", "_review_history", "_review_degraded_reasons"):
            setattr(worker, attr, 0 if attr == "_review_iteration_count" else [])
        review._run_unified_review(worker, "Commit the observed work", goal="commit the observed work")
        delivered = captured["prompt"]
    else:
        def reviewer(_ctx, message, **kw):
            invoke_review_paid_stamp(worker._review_paid_stamp)
            worker._last_review_critical_findings = []
            worker._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [],
                                               "raw_text": "[] PASS"}]
            return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

        commit = _commit_until_handback(monkeypatch, reviewer)
        delivered = commit(worker, "Commit the observed work", paths=chosen, skip_advisory_review=True)
        assert delivered.startswith("Review outcome returned before commit"), delivered
    assert disclosure in delivered


def test_author_handback_and_the_next_epoch_keep_the_disclosed_gap(body, monkeypatch):
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.tools import git as git_tools

    serving, data = body
    ctx, bound = _moved_candidate(serving, data, "root-hand")
    seen = []

    def reviewer(_ctx, message, **kw):
        from ouroboros.mutation_attribution import selection_disclosure
        from ouroboros.review_dispatch import invoke_review_paid_stamp

        invoke_review_paid_stamp(ctx._review_paid_stamp)
        seen.append(selection_disclosure(ctx._commit_attribution))
        ctx._last_review_critical_findings = []
        ctx._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [], "raw_text": "[] PASS"}]
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

    _commit_until_handback(monkeypatch, reviewer)
    assert "baseline_stale" in git_tools._repo_commit_push(ctx, "Carry the edit", skip_advisory_review=True)
    first = git_tools._repo_commit_push(ctx, "Carry the edit", paths=[EDIT], skip_advisory_review=True)
    assert first.startswith("Review outcome returned before commit"), first
    head = first.split("\n", 1)[0]
    assert "not re-anchored" in head and seen and "not re-anchored" in seen[0]
    reference = json.loads(first.split("\n", 1)[1])["review_reference"]
    monkeypatch.setattr(git_tools, "_run_parallel_review", lambda *_a, **_kw: pytest.fail("continuation buys no reviewer"))
    second = git_tools._repo_commit_push(ctx, "Carry the edit", paths=[EDIT], review_reference=reference,
                                         author_disposition={"disposition": "accepted",
                                                             "rationale": "I read the outcome and the disclosed gap."})
    commit = git(bound["path"], "rev-parse", "HEAD")
    assert git(bound["path"], "show", f"HEAD:{EDIT}") == "GEN = 'TASK_ON_FRESH_BASE'", second
    epoch = _baseline(data, "root-hand")["epochs"][-1]
    assert epoch["commit"] == commit and epoch["attribution"]["gaps"] == ["baseline_stale"]
    assert epoch["previous"][0]["git"]["base_commit"] == bound["base_sha"]
    assert _surface(data, "root-hand", bound["path"])["git"]["base_commit"] == commit
