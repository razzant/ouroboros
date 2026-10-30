"""An Update conflict is adapted in its own workspace by one reviewed resolver: real Git, a miniature body.

The resolver itself is a model task; here its work is played by the test (an edit and one
commit on the mechanical result M0) and its review by the commit gate's own record
(``body_candidate.record_reviewed_commit``). What is exercised: the hand-off, the binding,
the settlement rules and the continuation to an activatable candidate.
"""
from __future__ import annotations

import pathlib

import pytest

from ouroboros import body_adoption, body_candidate, body_switch
from supervisor import update_adaptation, update_prepare
from tests.body_candidate_support import git, make_ctx
from tests.test_preflight_receipts import receipt_python  # noqa: F401 -- dependency-only lane fixture
from tests.test_update_activation import Scene

pytestmark = pytest.mark.serial

# Settlement's exact full evidence (o4) has its own end-to-end case below, with the real post-commit gate; the
# other cases here play the resolver without a test run, so the scene stands that one rule in for them.
REAL_QUALIFICATION = update_adaptation._adaptation_unqualified


@pytest.fixture
def scene(tmp_path, monkeypatch):
    import supervisor.queue as queue
    import supervisor.state as state
    import supervisor.workers as workers

    scene = Scene(tmp_path, monkeypatch)
    scene.queued = []
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 10.0)
    monkeypatch.setattr(state, "load_state", lambda *_a, **_kw: {"owner_chat_id": 7})
    monkeypatch.setattr(workers, "ensure_worker_pool_started", lambda **_kw: True)
    monkeypatch.setattr(queue, "DRIVE_ROOT", scene.data)  # the admission receipt lands in this scene's data root
    scene.real_enqueue = queue.enqueue_task
    monkeypatch.setattr(queue, "enqueue_task", lambda task, **kw: scene.queued.append((task, kw)) or task)
    monkeypatch.setattr(update_adaptation, "_adaptation_unqualified", lambda _workspace, _tip: "")
    return scene


def _conflict(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\nOFFICIAL = 1\n"})
    (scene.serving / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_OLD'\nLOCAL = 1\n")
    git(scene.serving, "commit", "-q", "-am", "local feature in mod_a")
    return scene.prepare()


def _resolve(scene, row, text, *, reviewed=True):
    """What the resolver does: resolve on M0 and make one commit, which the commit gate records."""
    workspace = pathlib.Path(row["path"])
    (workspace / "ouroboros" / "mod_a.py").write_text(text)
    git(workspace, "commit", "-q", "-am", "adapt mod_a: keep the official generation and the local feature")
    sha = git(workspace, "rev-parse", "HEAD")
    if reviewed:
        body_candidate.record_reviewed_commit(None, sha, row=update_prepare.find(row["candidate_id"]))
    return sha


def test_a_merge_conflict_starts_one_resolver_bound_to_the_workspace(scene):
    row = _conflict(scene)

    assert row["state"] == "adapting" and row["resolver"]["step"] == "merge"
    [(task, options)] = scene.queued
    # The queue's host-producer admission: front of the queue, persisted with its scheduled receipt.
    assert options["front"] is True and options["durable"] is True and task["chat_id"] == 7
    assert task["metadata"] == {"root_task_id": task["id"], "update_candidate_resolver": row["candidate_id"],
                                "resource_intent": {"kind": "explicit_resource", "root": row["path"]}}
    assert "ouroboros/mod_a.py" in task["text"] and "commit_reviewed" in task["text"]
    workspace = pathlib.Path(row["path"])
    assert git(workspace, "rev-parse", "HEAD") == row["resolver"]["m0"]  # M0, markers and all, is the base
    assert "<<<<<<<" in (workspace / "ouroboros" / "mod_a.py").read_text()
    ctx = make_ctx(scene.serving, scene.data, task["id"])
    ctx.task_metadata = dict(task["metadata"])
    body_candidate.restore(ctx)
    assert pathlib.Path(ctx.repo_dir) == workspace and ctx.branch_dev == row["branch"]
    assert pathlib.Path(body_candidate.serving_repo_dir_for(ctx)) == scene.serving.resolve()
    projection = update_prepare.projection()
    assert projection["state"] == "adapting" and projection["adaptation"]["step"] == "merge"


def test_resolver_followup_keeps_candidate_target_when_the_copy_disappears(scene, monkeypatch):
    from ouroboros.tools.followup import followup_resource_intent
    from supervisor import schedule_occurrence, workers

    row = _conflict(scene)
    [(resolver, _options)] = scene.queued
    workspace = pathlib.Path(row["path"])
    ctx = make_ctx(scene.serving, scene.data, resolver["id"])
    intent = followup_resource_intent(ctx, resolver["metadata"], "")
    assert intent == {"kind": "explicit_resource", "root": str(workspace)}

    monkeypatch.setattr(workers, "REPO_DIR", scene.serving)
    # The occurrence must resolve its original target, not restore the resolver's
    # authority or fall back to the serving body after the temporary copy retires.
    moved = workspace.with_name(workspace.name + "-retained")
    workspace.rename(moved)
    occurrence = {"id": "followup-update-test", "metadata": {}, "text": "continue the adaptation"}
    hold, _basis = schedule_occurrence.resolve_resource(
        occurrence, {"task": {"metadata": {"resource_intent": intent}}})
    # A known folder that is merely absent keeps its address (#1799, owner choice A): the follow-up starts there with
    # that availability fact, never in the serving body and never with the resolver's authority.
    assert hold is None
    assert occurrence["metadata"]["resource_intent"] == intent
    bound = occurrence["metadata"]["workspace_root"]
    assert bound == str(workspace.resolve()) and not pathlib.Path(bound).exists()
    assert occurrence["metadata"]["workspace_availability"]["state"] == "missing"
    assert not pathlib.Path(bound).is_relative_to(scene.serving.resolve())


def test_a_reviewed_resolution_continues_to_a_ready_adapted_merge_that_activates(scene, monkeypatch):
    import supervisor.update_checks as checks

    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": True, "stub": True})
    row = _conflict(scene)
    resolution = _resolve(scene, row, "GEN = 'GEN_CAND'\nOFFICIAL = 1\nLOCAL = 1\n")

    ready = update_adaptation._settle_locked(update_prepare.find(row["candidate_id"]))

    assert ready["state"] == "ready" and ready["merge"]["kind"] == "adapted"
    merged = ready["layers"]["new_commit"]
    assert git(scene.serving, "rev-list", "--parents", "-n", "1", merged).split()[1:] == [
        ready["layers_base"]["head"], scene.target]  # the whole target, as the second parent
    assert git(scene.serving, "rev-parse", f"{merged}^{{tree}}") == git(scene.serving, "rev-parse",
                                                                        f"{resolution}^{{tree}}")
    scene.activate(ready)
    assert scene.boot().returncode == 0
    assert scene.reports()[-1]["a"] == "GEN_CAND"
    assert (scene.serving / "ouroboros" / "mod_a.py").read_text() == "GEN = 'GEN_CAND'\nOFFICIAL = 1\nLOCAL = 1\n"
    body_switch._attest_loaded(str(scene.serving), body_adoption.read(scene.data))
    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "adopted"


def test_the_resolvers_continuation_tells_every_client_over_the_supervisors_bus(scene, monkeypatch):
    """An adaptation ends in the supervisor (its task's settlement), never in a request: the candidate's new state
    reaches every panel and the indicator as the Update's frame, over the bus the server binds to its sockets, once
    the record has changed. A settlement that is not this resolver's changes nothing and says nothing."""
    from types import SimpleNamespace

    import supervisor.message_bus as message_bus
    import supervisor.update_checks as checks

    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": True, "stub": True})
    row = _conflict(scene)
    assert row["state"] == "adapting"
    heard = []
    monkeypatch.setattr(message_bus, "try_get_bridge", lambda: SimpleNamespace(broadcast=lambda payload: heard.append(
        (payload, update_prepare.find(row["candidate_id"])["state"]))))
    _resolve(scene, row, "GEN = 'GEN_CAND'\nOFFICIAL = 1\nLOCAL = 1\n")
    metadata = {"update_candidate_resolver": row["candidate_id"], "root_task_id": row["resolver"]["task_id"]}

    update_adaptation._settle_resolution("update_adapt_someone_else", metadata)
    assert heard == [] and update_prepare.find(row["candidate_id"])["state"] == "adapting"

    update_adaptation._settle_resolution(row["resolver"]["task_id"], metadata)
    assert heard == [({"type": "update_progress_changed"}, "ready")]


def test_a_resolver_that_does_not_commit_leaves_the_owner_its_question(scene, monkeypatch):
    monkeypatch.setattr(update_adaptation, "_resolver_answer", lambda _task: "Keep LOCAL by dropping OFFICIAL, or drop it?")
    row = _conflict(scene)
    decided = update_adaptation._settle_locked(update_prepare.find(row["candidate_id"]))
    assert decided["state"] == "needs_decision" and decided["question"].startswith("Keep LOCAL")
    assert update_prepare.projection()["question"].startswith("Keep LOCAL")


def test_an_unreviewed_or_unresolved_commit_never_becomes_the_update(scene):
    row = _conflict(scene)
    _resolve(scene, row, "GEN = 'GEN_CAND'\nLOCAL = 1\n", reviewed=False)
    refused = update_adaptation._settle_locked(update_prepare.find(row["candidate_id"]))
    assert refused["state"] == "failed" and refused["reason"].startswith("adaptation_unreviewed")


def test_a_resolver_the_real_queue_refuses_leaves_a_retryable_candidate(scene, monkeypatch):
    """Negative control: the real queue closes intake for a restart; nothing claims a resolver started."""
    from supervisor import queue
    from ouroboros import server_restart

    monkeypatch.setattr(queue, "enqueue_task", scene.real_enqueue)
    monkeypatch.setattr(server_restart, "agent_restart_pending", lambda: True)
    monkeypatch.setitem(queue.RETURNS_NAMED, "restart", "named")

    row = _conflict(scene)

    assert row["state"] == "needs_adaptation" and "restart_intake_closed" in row["reason"]
    assert row["resolver"]["refused"].startswith("restart_intake_closed") and not row["resolver"]["task_id"]
    assert not any(task.get("metadata", {}).get("update_candidate_resolver") for task in queue.PENDING)
    projection = update_prepare.projection()
    assert projection["state"] == "needs_adaptation" and "prepare" in projection["actions"]


def test_a_resolver_whose_candidate_moved_on_stays_bound_to_its_own_checkout(scene):
    """A superseded or discarded candidate's resolver never falls back to the serving checkout."""
    row = _conflict(scene)
    [(task, _options)] = scene.queued
    update_prepare._settle(update_prepare.find(row["candidate_id"]), "superseded")

    ctx = make_ctx(scene.serving, scene.data, task["id"])
    ctx.task_metadata = dict(task["metadata"])
    body_candidate.restore(ctx)
    assert pathlib.Path(ctx.repo_dir) == pathlib.Path(row["path"]) != scene.serving

    workspace = pathlib.Path(row["path"])
    workspace.rename(workspace.with_name(workspace.name + "-gone"))  # its checkout vanished meanwhile
    lost = make_ctx(scene.serving, scene.data, task["id"])
    lost.task_metadata = dict(task["metadata"])
    body_candidate.restore(lost)
    assert pathlib.Path(lost.repo_dir) == workspace and not workspace.exists()  # its tools fail; serving untouched
    assert update_adaptation.resolver_row(task["metadata"]) is None  # and its settlement continues nothing


def test_discard_during_adaptation_keeps_the_checkout_until_the_resolver_has_exited(scene, monkeypatch):
    """The resolver is cancelled through its task owner; its files stay while its exit is unproven."""
    from ouroboros import subagent_worktrees
    from ouroboros.task_results import write_task_result
    from supervisor import task_lifecycle

    row = _conflict(scene)
    [(task, _options)] = scene.queued
    workspace = pathlib.Path(row["path"])
    scratch = body_candidate.scratch_root(workspace)  # the resolver's declared process scratch
    (scratch / "resolver-home").mkdir()
    cancelled, alive = [], {"value": True}
    monkeypatch.setattr(task_lifecycle, "cancel_task_by_id", lambda task_id: cancelled.append(task_id) or False)

    discarded = update_prepare.discard(update_prepare.find(row["candidate_id"]))

    assert discarded["state"] == "discarded" and cancelled == [task["id"]]
    assert discarded["resolver_stop"] == {"task_id": task["id"], "confirmed": False, "how": "cancelled"}
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: alive["value"])
    assert workspace.is_dir() and (scratch / "resolver-home").exists()  # a live resolver keeps both
    verdict = update_prepare.retention_verdict(update_prepare.find(row["candidate_id"]), live=lambda _t: None)
    assert verdict == {"remove": False, "reason": f"resolver_unconfirmed:{task['id']}"}

    write_task_result(scene.data, task["id"], "cancelled", result="cancelled by the discard")
    alive["value"] = False
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task_id: alive["value"])
    assert not workspace.exists() and not scratch.exists() and update_prepare.find(row["candidate_id"]) is None


def test_markers_left_in_a_reviewed_commit_are_refused(scene):
    row = _conflict(scene)
    _resolve(scene, row, "<<<<<<< ours\nGEN = 'GEN_OLD'\n=======\nGEN = 'GEN_CAND'\n>>>>>>> theirs\n")
    refused = update_adaptation._settle_locked(update_prepare.find(row["candidate_id"]))
    assert refused["state"] == "failed" and refused["reason"].startswith("adaptation_markers_left")


RESOLVED = "GEN = 'GEN_CAND'\nOFFICIAL = 1\nLOCAL = 1\n"


def test_an_adaptation_settles_only_on_a_complete_green_full_run_of_exactly_its_commit(
        scene, monkeypatch, receipt_python):  # noqa: F811 -- pytest fixture injection
    """Managed adaptation commits need the full suite (DEVELOPMENT 07/14). The resolver is a bound candidate, so
    its commit gate runs what it asked for; a request for full tests is intent, never proof. Settlement accepts
    the resolver's reviewed commit only on a durable receipt of a complete green FULL run of exactly that commit
    (``preflight_qualification.receipt_for_commit``, read under the root its post-commit gate wrote: the
    installation's). No run, a selected run, a full run of other bytes and an incomplete record each fail it with
    their own words; the work stays in the workspace. The REAL post-commit gate runs every time."""
    import json

    import supervisor.update_checks as checks
    from ouroboros import preflight_receipts, test_selection
    from ouroboros.tools import git as git_tools
    from tests.body_candidate_support import candidate_commit
    from tests.test_body_adoption_qualification import FLOOR, SUITE

    monkeypatch.setattr(update_adaptation, "_adaptation_unqualified", REAL_QUALIFICATION)
    for relative, text in SUITE.items():  # the body carries its own tests, as every real one does
        (scene.serving / relative).parent.mkdir(parents=True, exist_ok=True)
        (scene.serving / relative).write_text(text, encoding="utf-8")
    git(scene.serving, "add", "-A")
    git(scene.serving, "commit", "-q", "-m", "suite")
    for key, value in {"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_AGENT_PYTHON": receipt_python,
                       "OUROBOROS_PRE_PUSH_TESTS": "1", "OUROBOROS_PREFLIGHT_TEST_WORKERS": "2",
                       "OURO_TEST_WORKER_BUDGET": "2"}.items():
        monkeypatch.setenv(key, value)
    for key in ("OUROBOROS_PREFLIGHT_SERIAL", "OUROBOROS_PREFLIGHT_TIMEOUT_SEC"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(test_selection, "FLOOR_TESTS", dict(FLOOR))
    monkeypatch.setattr(checks, "workspace_checks", lambda _row: {"ok": True, "stub": True})
    preflight_receipts.capture_executing_source_baseline()  # receipts bind the loaded code, as server.py does
    row = _conflict(scene)
    [(task, _options)] = scene.queued
    ctx = make_ctx(scene.serving, scene.data, task["id"])
    ctx.task_metadata = dict(task["metadata"])
    body_candidate.restore(ctx)  # the resolver's own binding: its tools and gate work in this workspace
    workspace, m0 = pathlib.Path(row["path"]), row["resolver"]["m0"]
    receipts = scene.data / "state" / preflight_receipts.RECEIPTS_DIR

    def commit(text, *, run=True, **request):
        git(workspace, "reset", "-q", "--hard", m0)
        sha = candidate_commit(workspace, "adapt mod_a", files={"ouroboros/mod_a.py": text})
        ctx._commit_test_request = {"extra_tests": (), "full_tests": False, "test_reason": "", **request}
        assert not run or git_tools._run_pre_push_tests(ctx) is None
        body_candidate.record_reviewed_commit(ctx, sha)
        return sha

    def settled():
        return update_adaptation._settle_locked(update_prepare.find(row["candidate_id"]))

    commit(RESOLVED + "# no run\n", run=False)
    refused = settled()
    assert refused["state"] == "failed" and refused["reason"].startswith("adaptation_unqualified: no complete green "
                                                                         "full test run of exactly")
    assert "(no record of that commit)" in refused["reason"] and str(scene.data) in refused["reason"]
    commit(RESOLVED + "# selected\n")
    assert "a selected run, never a full one" in settled()["reason"]
    sha = commit(RESOLVED + "# other bytes\n", run=False, full_tests=True, test_reason="adaptation")
    (workspace / "ouroboros" / "mod_a.py").write_text(RESOLVED + "# edited after the commit\n")
    assert git_tools._run_pre_push_tests(ctx) is None
    git(workspace, "checkout", "-q", "--", "ouroboros/mod_a.py")
    assert "its run carried other bytes than the commit's tree" in settled()["reason"]
    sha = commit(RESOLVED, full_tests=True, test_reason="adaptation of mod_a")
    [path] = [path for path in receipts.glob("*.json") if json.loads(path.read_text())["identity"]["head"] == sha]
    record = json.loads(path.read_text())
    assert record["schema"] == preflight_receipts.RECEIPT_SCHEMA and record["result"] == "passed"
    path.write_text(json.dumps({**record, "lanes": {**record["lanes"], "serial": "failed"}}))
    assert "the parallel/serial lane evidence is incomplete" in settled()["reason"]
    assert git(workspace, "rev-parse", "HEAD") == sha  # the resolver's work stays where it was
    path.write_text(json.dumps(record))

    ready = settled()

    assert ready["state"] == "ready" and ready["merge"]["kind"] == "adapted" and ready["merge"]["resolution"] == sha
    assert git(scene.serving, "show", f"{ready['layers']['new_commit']}:ouroboros/mod_a.py") + "\n" == RESOLVED
