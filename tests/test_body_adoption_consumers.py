"""The existing owners around a body adoption: server restart/bootstrap/exit tail,
evolution, publication and the Runtime fact (#1539). Real Git throughout; the
supervisor context and the stop owners are the same doubles the neighbouring
restart suites use."""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import body_adoption, body_candidate, body_switch
from tests.body_candidate_support import (
    candidate_commit, git, isolate, make_ctx, make_serving, restart_receipt, rich_commit,
)

EXITED = {"doomed": [4001], "dead": [4001], "unconfirmed": [], "cleanup_ok": True, "snapshot_ok": True}


@pytest.fixture
def scene(tmp_path, monkeypatch):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    ctx = make_ctx(serving, data, "root-consumer")
    body_candidate.prepare(ctx)
    cand = rich_commit(ctx.repo_dir)
    body_candidate.record_reviewed_commit(ctx, cand)
    return SimpleNamespace(serving=serving, data=data, ctx=ctx, cand=cand,
                           old=git(serving, "rev-parse", "HEAD"), candidate=pathlib.Path(ctx.repo_dir))


class _GitOps:
    def __init__(self):
        self.safe_restart_calls, self.local_sync_calls = [], []

    def init(self, **_kwargs):
        return None

    def ensure_repo_present(self):
        return None

    def safe_restart(self, **kwargs):
        self.safe_restart_calls.append(kwargs)
        return True, "reset"

    def sync_runtime_dependencies(self, **kwargs):
        self.local_sync_calls.append(kwargs)
        return True, "ok"

    def import_test(self):
        return {"ok": True}


def _managed_server(monkeypatch, scene):
    import server

    monkeypatch.setattr(server, "REPO_DIR", scene.serving)
    monkeypatch.setattr(server, "DATA_DIR", scene.data)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED", True)
    monkeypatch.setattr(server, "_LAUNCHER_MANAGED_REPO_DIR", str(scene.serving.resolve()))
    monkeypatch.setattr(server, "setup_remote_if_configured", lambda *args: None)
    monkeypatch.setattr(server, "_has_active_evolution_transaction", lambda: False)
    monkeypatch.setattr(server, "_safe_restart_serialized", lambda fn, **kwargs: fn(**kwargs))
    return server


def _switch(scene, reason="adopt it"):
    """Authorize, bind, arm and let the REAL helper switch the tree in this process."""
    body_adoption.authorize(scene.ctx, scene.cand, reason=reason)
    restart_receipt(scene.data, scene.cand, reason)
    assert body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, reason)()[0]
    assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owner_restart=False,
                             owned_stop={"state": "completed", "unconfirmed": []}) == "armed"
    handoff = body_adoption.read(scene.data)
    helper = str(body_adoption.helper_dir(scene.data))
    body_switch.record(helper, handoff, "switching", "test")
    root, rows = str(scene.serving), handoff["switch"]
    tracked = body_switch._tracks_filemode(root)
    body_switch._converge(root, rows, "new", body_switch._states(root, rows, tracked), tracked)
    body_switch._set_index(root, rows, "new", body_switch._index_states(root, rows))
    git(scene.serving, "update-ref", "refs/heads/ouroboros", scene.cand, scene.old)
    body_switch.record(helper, handoff, "switched", "test")
    # The fresh process the helper hands the tree to attests what it imports (the hook's
    # ``switched`` branch); here that process is this one.
    body_switch._attest_loaded(root, body_adoption.read(scene.data))


def test_managed_boot_right_after_a_switch_does_not_reset_or_clean_the_checkout(scene, monkeypatch):
    server = _managed_server(monkeypatch, scene)
    ordinary = _GitOps()
    assert server._bootstrap_supervisor_repo({}, ordinary)[0] is True
    assert [call["unsynced_policy"] for call in ordinary.safe_restart_calls] == ["rescue_and_reset"]

    (scene.serving / "owner-notes.txt").write_text("kept through the switch\n")
    _switch(scene)
    adoption_boot = _GitOps()

    ok, message = server._bootstrap_supervisor_repo({}, adoption_boot)

    assert ok is True and "local-dev" in message
    assert adoption_boot.safe_restart_calls == []  # no reset --hard / clean -fd on the tree that just landed
    assert adoption_boot.local_sync_calls == [{"reason": "bootstrap_local_dev"}]  # dependencies for the NEW tree
    assert (scene.serving / "owner-notes.txt").read_text() == "kept through the switch\n"

    assert body_adoption.finalize_on_boot(scene.data, scene.serving, supervisor_ready=True)["outcome"] == "adopted"
    later = _GitOps()
    server._bootstrap_supervisor_repo({}, later)
    assert [call["unsynced_policy"] for call in later.safe_restart_calls] == ["rescue_and_reset"]  # unchanged since


def test_exit_tail_arms_only_with_the_stop_owners_confirmation(scene, monkeypatch):
    import server
    from supervisor import workers

    monkeypatch.setattr(server, "DATA_DIR", scene.data)
    census = {"value": EXITED}
    monkeypatch.setattr(workers, "kill_workers", lambda **kw: True)  # the bool is about cleanup, not readers
    monkeypatch.setattr(workers, "last_worker_exit_census", lambda: census["value"])
    monkeypatch.setattr("ouroboros.platform_layer.kill_process_on_port", lambda _port: None)
    monkeypatch.setattr("ouroboros.extension_companion.panic_kill_all", lambda: None)
    monkeypatch.setattr("ouroboros.gateway.host_service.host_service_port", lambda: 8767)
    monkeypatch.setattr(server, "_stop_owned_daemon_for_new_pin", lambda: None)
    monkeypatch.setattr("multiprocessing.active_children", lambda: [])
    body_adoption.authorize(scene.ctx, scene.cand, reason="adopt it")
    restart_receipt(scene.data, scene.cand, "adopt it")
    assert body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, "adopt it")()[0]
    pointer = pathlib.Path(body_switch.pointer_path(str(scene.serving)))

    # An ordinary shutdown (no restart requested) never arms.
    monkeypatch.setattr(server, "stop_owned_work", lambda _root: {"state": "completed", "unconfirmed": []})
    server._restart_requested.clear()
    server._emergency_process_cleanup(port_sweep=False)
    assert body_adoption.read(scene.data)["phase"] == "authorized" and not pointer.exists()

    server._restart_requested.set()
    try:
        # A custodied survivor of the owned stop defers arming: the next generation boots the old tree.
        monkeypatch.setattr(server, "stop_owned_work",
                            lambda _root: {"state": "unconfirmed", "unconfirmed": [{"pid": 77}]})
        server._emergency_process_cleanup(port_sweep=False)
        handoff = body_adoption.read(scene.data)
        assert handoff["phase"] == "authorized" and "owned_work_unconfirmed" in handoff["events"][-1]["detail"]
        assert not pointer.exists()

        handoff["restart_bound"] = True
        body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), handoff)
        monkeypatch.setattr(server, "stop_owned_work", lambda _root: {"state": "completed", "unconfirmed": []})
        # A kill whose join could not confirm one worker's exit: the census, not kill_workers' bool, decides.
        census["value"] = {**EXITED, "unconfirmed": [4001]}
        server._emergency_process_cleanup(port_sweep=False)
        handoff = body_adoption.read(scene.data)
        assert handoff["phase"] == "authorized" and "worker_exits_unconfirmed" in handoff["events"][-1]["detail"]
        census["value"] = EXITED

        handoff["restart_bound"] = True  # the same adoption, carried by its restart again
        body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), handoff)
        server._owner_restart_requested.set()  # the owner pressed Restart meanwhile: newer control governs
        try:
            server._emergency_process_cleanup(port_sweep=False)
        finally:
            server._owner_restart_requested.clear()
        assert body_adoption.read(scene.data)["phase"] == "authorized" and not pointer.exists()

        handoff = body_adoption.read(scene.data)
        handoff["restart_bound"] = True
        body_switch.write_handoff(str(body_adoption.helper_dir(scene.data)), handoff)
        server._emergency_process_cleanup(port_sweep=False)
    finally:
        server._restart_requested.clear()
    assert body_adoption.read(scene.data)["phase"] == "armed"
    assert pointer.read_text().strip() == str(body_adoption.helper_dir(scene.data))
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old  # the old generation never moved the tree


def test_supervisor_restart_binds_the_adoption_and_a_refused_restart_abandons_it(scene, monkeypatch):
    import server

    monkeypatch.setattr(server, "_safe_restart_serialized", lambda fn, **kwargs: fn(**kwargs))
    exited, messages = [], []
    monkeypatch.setattr(server, "_request_restart_exit", lambda: exited.append(True))

    def ctx(safe_restart):
        return SimpleNamespace(
            DRIVE_ROOT=scene.data, REPO_DIR=scene.serving, RUNNING={}, load_state=lambda: {"owner_chat_id": 1},
            safe_restart=safe_restart, kill_workers=lambda **k: None, update_state=lambda fn: None,
            persist_queue_snapshot=lambda **k: None, send_with_budget=lambda *a, **kw: messages.append(a[1]))

    body_adoption.authorize(scene.ctx, scene.cand, reason="adopt reviewed fix")
    restart_receipt(scene.data, scene.cand, "adopt reviewed fix")
    server._perform_supervisor_restart(ctx(lambda **k: (False, "Unsynced state rescued; restart blocked.")),
                                       restart_reason="adopt reviewed fix")
    assert exited == [] and body_adoption.read(scene.data) == {}
    assert any("was not armed" in text for text in messages)

    body_adoption.authorize(scene.ctx, scene.cand, reason="adopt reviewed fix")
    restart_receipt(scene.data, scene.cand, "adopt reviewed fix")
    server._perform_supervisor_restart(ctx(lambda **k: (True, "ok")), restart_reason="adopt reviewed fix")
    assert exited == [True]
    assert body_adoption.read(scene.data)["restart_bound"] is True
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old


def test_evolution_restart_accepts_a_candidate_claim_at_the_adoption_base(scene, monkeypatch, tmp_path):
    import server

    from supervisor import evolution_lifecycle
    from tests._evolution_state_shared import _active_transaction

    campaign, tx = _active_transaction(scene.data, task_id="root-consumer")
    claim = {"campaign_id": campaign["id"], "transaction_id": tx["transaction_id"], "task_id": tx["task_id"]}
    assert evolution_lifecycle.record_evolution_commit(**claim, commit_sha=scene.cand)["ok"] is True
    claim["commit_sha"] = scene.cand
    marker = scene.data / "state" / "pending_restart_verify.json"
    marker.write_text(json.dumps({"reason": "evolution restart", "expected_sha": scene.cand,
                                  "evolution_claim": claim}))
    monkeypatch.setattr(server, "_safe_restart_serialized", lambda fn, **kwargs: fn(**kwargs))
    exited, messages, restarted = [], [], []
    monkeypatch.setattr(server, "_request_restart_exit", lambda: exited.append(True))
    ctx = SimpleNamespace(
        DRIVE_ROOT=scene.data, REPO_DIR=scene.serving, RUNNING={}, load_state=lambda: {"owner_chat_id": 1},
        safe_restart=lambda **k: restarted.append(k) or (True, "ok"), kill_workers=lambda **k: None,
        update_state=lambda fn: None, persist_queue_snapshot=lambda **k: None,
        send_with_budget=lambda *a, **kw: messages.append(a[1]))

    # The claimed commit lives on the candidate; with no authorized adoption the checkout "does not match".
    server._perform_supervisor_restart(ctx, restart_reason="evolution restart", evolution_restart=True)
    assert restarted == [] and "no longer matches" in messages[-1]

    # The supervisor's own evolution restart authorizes exactly that commit, then the check passes at the base.
    assert body_adoption.authorize_for_task(scene.data, "root-consumer", scene.cand, "evolution restart") is True
    server._perform_supervisor_restart(ctx, restart_reason="evolution restart", evolution_restart=True)
    assert restarted and exited == [True] and body_adoption.read(scene.data)["restart_bound"] is True

    # Owner dirt in the serving tree still stops an evolution restart (unchanged rule).
    exited.clear(), restarted.clear()
    (scene.serving / "owner-notes.txt").write_text("dirt\n")
    server._perform_supervisor_restart(ctx, restart_reason="evolution restart", evolution_restart=True)
    assert restarted == [] and exited == []


def test_evolution_cleanup_never_stashes_or_resets_serving_dirt_of_a_candidate_cycle(scene, monkeypatch):
    from supervisor import evolution_lifecycle, git_ops
    from tests._evolution_state_shared import _active_transaction

    _campaign, tx = _active_transaction(scene.data, task_id="root-consumer")
    monkeypatch.setattr(git_ops, "REPO_DIR", scene.serving)
    (scene.serving / "ouroboros/mod_b.py").write_text("GEN = 'OWNER_EDIT'\n")
    (scene.serving / "owner-notes.txt").write_text("owner untracked\n")
    tx = {**tx, "base_head": scene.old}

    evolution_lifecycle._cleanup_worktree_after_cycle(tx, "root-consumer")

    assert tx["cleanup_status"] == "candidate_retained" and "cleanup_stash" not in tx
    assert (scene.serving / "ouroboros/mod_b.py").read_text() == "GEN = 'OWNER_EDIT'\n"
    assert (scene.serving / "owner-notes.txt").read_text() == "owner untracked\n"
    assert git(scene.serving, "stash", "list") == ""
    assert scene.candidate.is_dir() and git(scene.candidate, "rev-parse", "HEAD") == scene.cand

    # A cycle that ended with an authorization nobody armed leaves no pending adoption behind.
    git(scene.serving, "checkout", "--", "ouroboros/mod_b.py")
    body_adoption.authorize(scene.ctx, scene.cand, reason="evolution restart")
    evolution_lifecycle._cleanup_worktree_after_cycle({**tx}, "root-consumer")
    assert body_adoption.read(scene.data) == {}

    # A cycle that authored the serving checkout itself keeps today's cleanup contract.
    other = {**tx, "task_id": "in-place-task"}
    evolution_lifecycle._cleanup_worktree_after_cycle(other, "in-place-task")
    assert other["cleanup_status"] != "candidate_retained"


def test_serving_push_publishes_reachable_release_tags_and_never_a_candidates(scene, monkeypatch, tmp_path):
    from ouroboros.tools import git as git_tools
    from supervisor import git_ops

    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(scene.serving, "remote", "add", "origin", str(remote))
    git(scene.serving, "tag", "-a", "v1.0.0", "-m", "v1.0.0: base")  # the serving line's own release tag

    # A NUMBERED release committed in the candidate gets its annotated tag at commit time (shared namespace).
    numbered = candidate_commit(scene.candidate, "v1.0.1: numbered release", files={"VERSION": "1.0.1\n"})
    tagged = git_tools._auto_tag_on_version_bump(scene.candidate, "numbered release", expected_commit_sha=numbered,
                                                 expected_tag="v1.0.1")
    assert tagged == " [tagged: v1.0.1]" and git(scene.serving, "rev-parse", "v1.0.1^{commit}") == numbered
    # A version-neutral contribution carries no release tag at all.
    neutral = candidate_commit(scene.candidate, "neutral fix", files={"ouroboros/mod_b.py": "GEN = 'N'\n"})
    assert git_tools._auto_tag_on_version_bump(scene.candidate, "neutral fix", expected_commit_sha=neutral,
                                               expected_tag="") == ""
    assert git(scene.candidate, "tag", "--points-at", neutral) == ""
    git(scene.serving, "update-ref", "refs/ouroboros/candidates/private-pin", neutral)

    monkeypatch.setattr(git_ops, "REPO_DIR", scene.serving)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    pushed, message = git_ops.push_to_remote()

    assert pushed is True and "+ tags" in message
    assert git(remote, "rev-parse", "refs/heads/ouroboros") == scene.old
    assert git(remote, "rev-parse", "refs/tags/v1.0.0^{commit}") == scene.old
    refs = git(remote, "for-each-ref", "--format=%(refname)").splitlines()
    assert sorted(refs) == ["refs/heads/ouroboros", "refs/tags/v1.0.0"]  # no v1.0.1, no candidate branch, no pin

    # Explicit publication of the approved contribution branch stays available and is not forced.
    git(scene.candidate, "push", "-q", "origin", f"{scene.ctx.branch_dev}:refs/heads/contrib/fix")
    assert git(remote, "rev-parse", "refs/heads/contrib/fix") == neutral
    assert "refs/tags/v1.0.1" not in git(remote, "for-each-ref", "--format=%(refname)")

    # Once the numbered release IS the serving line, the same push publishes its tag.
    git(scene.serving, "merge", "-q", "--ff-only", numbered)
    assert git_ops.push_to_remote()[0] is True
    assert git(remote, "rev-parse", "refs/tags/v1.0.1^{commit}") == numbered


def test_boot_settlement_pushes_the_adopted_serving_line_and_tells_the_owner(scene, monkeypatch, tmp_path):
    from supervisor import git_ops, git_ops_reset, message_bus, state

    remote = tmp_path / "origin.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(scene.serving, "remote", "add", "origin", str(remote))
    monkeypatch.setattr(git_ops, "REPO_DIR", scene.serving)
    monkeypatch.setattr(git_ops, "BRANCH_DEV", "ouroboros")
    facts, sent = [], []
    monkeypatch.setattr(git_ops_reset, "_record_checkout_facts", lambda value: facts.append(value))
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat, text, **kw: sent.append((chat, text, kw)))

    assert body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True) == {}  # nothing inherited
    assert sent == [] and git(remote, "for-each-ref") == ""

    _switch(scene)
    unready = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=False)
    assert unready["outcome"] == "unconfirmed" and git(remote, "for-each-ref") == "" and facts == []

    settled = body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    assert settled["outcome"] == "adopted"
    assert facts == [{"current_branch": "ouroboros", "current_sha": scene.cand}]
    assert git(remote, "rev-parse", "refs/heads/ouroboros") == scene.cand  # the serving line, as after any commit
    assert sent and sent[0][0] == 7 and scene.cand[:12] in sent[0][1]
    assert sent[0][2]["system_type"] == "restart_notice"


def test_runtime_fact_lists_retained_candidates_for_a_deliberate_resume(scene):
    assert body_candidate.context_fact("someone-else")[0] == {
        "id": "body_root-consumer", "owner_task": "root-consumer", "branch": "candidate/root-consumer",
        "path": str(scene.candidate), "base": scene.old[:12], "reviewed_commits": 1, "yours": False}
    assert body_candidate.context_fact("root-consumer")[0]["yours"] is True
    for index in range(3):
        body_candidate.prepare(make_ctx(scene.serving, scene.data, f"extra-{index}"))
    limited = body_candidate.context_fact("root-consumer", limit=2)
    assert len(limited) == 3 and limited[-1] == {"omitted": 2, "source": "state/subagent_worktrees.json (kind=body_candidate rows)"}

    from ouroboros.context import build_runtime_section

    env = SimpleNamespace(repo_dir=scene.serving, drive_root=scene.data, budget_drive_root=None,
                          drive_path=lambda name: scene.data / name)
    section = build_runtime_section(env, {"id": "root-consumer", "type": "task"})
    assert '"body_candidates"' in section and "candidate/root-consumer" in section
    assert f'"repo_dir": "{scene.serving}"' in section  # the Runtime block still names the RUNNING body


def test_adoption_state_uses_canonical_budget_root_for_forked_task(scene, tmp_path):
    """A forked execution drive must not hide restart adoption from the server boot."""
    fork = tmp_path / "forked-drive"
    fork.mkdir()
    scene.ctx.drive_root = fork
    scene.ctx.budget_drive_root = scene.data
    handoff = body_adoption.authorize(scene.ctx, scene.cand, reason="canonical-adoption")
    assert handoff["cand"] == scene.cand
    assert body_adoption.read(scene.data)["cand"] == scene.cand
    assert body_adoption.read(fork) == {}


def test_request_restart_from_a_forked_drive_reaches_the_supervisor_binding_and_arming(scene, tmp_path):
    """The restart tool on a project task's forked drive writes the handoff AND its restart receipt
    where the server (its canonical DATA_DIR) binds and arms them."""
    from ouroboros.tools.registry import ToolRegistry

    fork = tmp_path / "forked-drive"
    (fork / "state").mkdir(parents=True)
    scene.ctx.drive_root, scene.ctx.budget_drive_root = fork, scene.data
    registry = ToolRegistry(repo_dir=scene.serving, drive_root=fork)
    registry.set_context(scene.ctx)

    adopting = registry.execute("request_restart", {"reason": "adopt from project", "adopt_commit": scene.cand})

    assert f"adopts candidate commit {scene.cand[:12]}" in adopting, adopting
    assert not (fork / "state" / "pending_restart_verify.json").exists() and body_adoption.read(fork) == {}
    receipt = json.loads((scene.data / "state" / "pending_restart_verify.json").read_text())
    assert (receipt["expected_sha"], receipt["reason"]) == (scene.cand, "adopt from project")
    assert body_adoption.bind_restart(lambda **_kw: (True, "ok"), scene.data, "adopt from project")()[0]
    assert body_adoption.arm(scene.data, worker_exits=EXITED, live_children=[], owner_restart=False,
                             owned_stop={"state": "completed", "unconfirmed": []}) == "armed"
    assert git(scene.serving, "rev-parse", "HEAD") == scene.old  # the old generation wrote nothing


def test_intent_recovery_finds_the_candidate_commit_and_boot_verifies_only_the_serving_sha(
    tmp_path, monkeypatch,
):
    """#1539: an evolution cycle commits in its body candidate, not on the serving HEAD.

    A worker death between that commit and both of its receipts (the transaction's SHA
    and the candidate row's reviewed provenance) must not turn the cycle into ``no_op``
    when its task-done arrives: the post-task classification recovers the commit from
    the task's own candidate by the intent's exact tree and parents, so the evolution
    restart can adopt it. Boot verification stays on the serving SHA and absorbs it once
    the serving checkout holds it. Real Git, distinct checkouts.
    """
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle, git_ops
    from tests._evolution_state_shared import _active_transaction

    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr(git_ops, "REPO_DIR", serving, raising=False)
    campaign, tx = _active_transaction(data, task_id="evo-candidate")
    ctx = make_ctx(serving, data, "evo-candidate")
    body_candidate.prepare(ctx)
    candidate, serving_head = ctx.repo_dir, git(serving, "rev-parse", "HEAD")
    commit_sha = candidate_commit(candidate, "evolution cycle")  # the reviewed commit; then the process died
    assert evolution_lifecycle.update_evolution_transaction("evo-candidate", commit_intent={
        "tree_sha": git(candidate, "rev-parse", "HEAD^{tree}"), "parents": [serving_head]})

    assert evolution_lifecycle.update_evolution_campaign_after_task(
        "evo-candidate", cost_usd=0.0, outcome_axes={}, rounds=1, transaction=tx)["persisted"] is True

    recovered = evolution_lifecycle._read_evolution_campaign()["active_transaction"]
    assert (recovered["commit_sha"], recovered["cycle_outcome"]) == (commit_sha, "waiting_for_restart")
    assert recovered["commit_receipt"]["reason"] == "recovered_from_commit_intent"
    assert body_candidate.find("evo-candidate")["reviewed_commits"] == [commit_sha]
    assert body_adoption.authorize_for_task(data, "evo-candidate", commit_sha, "evolution restart") is True
    assert body_adoption.read(data)["cand"] == commit_sha and git(serving, "rev-parse", "HEAD") == serving_head
    body_adoption.abandon(data, "test")

    git(serving, "merge", "--ff-only", "-q", commit_sha)  # what the adoption lands
    monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: "boot-gen-2")
    agent_startup_checks.verify_restart(SimpleNamespace(
        drive_path=lambda name: data / name, drive_root=data, repo_dir=serving), commit_sha)
    absorbed = evolution_lifecycle._read_evolution_campaign()["transaction_history"][-1]
    assert (absorbed["commit_sha"], absorbed["cycle_outcome"]) == (commit_sha, "absorbed")


def _crash_recovered_cycle(tmp_path, monkeypatch, task_id):
    """Boot A after the server died between the cycle's candidate commit and its receipts."""
    from ouroboros import agent_startup_checks, process_custody
    from supervisor import evolution_lifecycle, git_ops
    from tests._evolution_state_shared import _active_transaction

    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr(git_ops, "REPO_DIR", serving, raising=False)
    _active_transaction(data, task_id=task_id)
    ctx = make_ctx(serving, data, task_id)
    body_candidate.prepare(ctx)
    candidate, serving_head = ctx.repo_dir, git(serving, "rev-parse", "HEAD")
    commit_sha = candidate_commit(candidate, "evolution cycle")  # the reviewed commit; then the server died
    assert evolution_lifecycle.update_evolution_transaction(task_id, commit_intent={
        "tree_sha": git(candidate, "rev-parse", "HEAD^{tree}"), "parents": [serving_head]})
    env = SimpleNamespace(drive_path=lambda name: data / name, drive_root=data, repo_dir=serving)

    def boot(generation, observed=serving_head):
        monkeypatch.setattr(process_custody, "current_custody_session_id", lambda: generation)
        agent_startup_checks.verify_restart(env, observed)
        return evolution_lifecycle._read_evolution_campaign()

    boot("boot-gen-1")
    return SimpleNamespace(serving=serving, data=data, candidate=candidate, serving_head=serving_head,
                           commit_sha=commit_sha, boot=boot)


def _events(data, kind):
    path = data / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == kind]


def _outcome_tags(data):
    path = data / "state" / "evolution_checkpoints.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("kind") == "cycle_outcome"]


def test_boot_after_a_server_crash_recovers_the_candidate_commit_without_absorbing_it(tmp_path, monkeypatch):
    """#1539 R1 Scope: the server died between the cycle's candidate commit and its receipts,
    so no task-done ever classifies the cycle. The worker boot's markerless reconcile
    recovers the exact commit from the task's candidate (the intent's tree and parents)
    together with its reviewed provenance. Only the serving SHA proves a restart, so the
    unadopted commit stays an open transaction across any number of unrelated boots, with
    nothing adopted or restarted, until the deliberate adoption lands it."""
    from ouroboros.utils import update_json_locked

    scene = _crash_recovered_cycle(tmp_path, monkeypatch, "evo-crash")
    commit_sha, data, serving = scene.commit_sha, scene.data, scene.serving
    campaign = update_json_locked(data / "state" / "evolution_campaign.json",
                                  lambda current: {**current, "post_task_backlog_id": "bl-1"})
    open_tx = campaign["active_transaction"]
    assert (open_tx["commit_sha"], open_tx["restart_required"], open_tx["restart_verified"]) == (
        commit_sha, True, False), open_tx
    assert open_tx["commit_receipt"]["reason"] == "recovered_from_commit_intent"
    assert open_tx.get("cycle_outcome") != "absorbed" and int(campaign.get("absorbed_cycles_done") or 0) == 0
    assert not [row for row in campaign.get("transaction_history") or [] if row.get("commit_sha") == commit_sha]
    assert body_candidate.find("evo-crash")["reviewed_commits"] == [commit_sha]
    assert git(serving, "rev-parse", "HEAD") == scene.serving_head and git(serving, "status", "--porcelain") == ""
    assert scene.boot("boot-gen-1")["active_transaction"] == open_tx  # a respawn in the same generation

    # Unrelated new-generation boots without an adoption: the serving HEAD lacking the commit is the
    # awaited state, not a rollback. Nothing is abandoned, reported, counted, adopted or restarted.
    for generation in ("boot-gen-2", "boot-gen-3"):
        kept = scene.boot(generation)
        tx = kept["active_transaction"]
        assert {key: tx[key] for key in open_tx if key != "updated_at"} == {
            key: value for key, value in open_tx.items() if key != "updated_at"}
        assert (tx["restart_required"], tx["restart_verified"], tx["restart_observed_sha"]) == (
            True, False, scene.serving_head)
        assert kept["post_task_backlog_id"] == "bl-1" and kept["last_boot_reconcile_gen"] == generation
        assert not kept.get("transaction_history") and not kept.get("objective_repeat_counts")
        assert "pending_owner_report" not in kept and not _outcome_tags(data)
    assert [(row["reason"], row["commit_sha"]) for row in _events(data, "evolution_tx_awaiting_adoption")] == [
        ("candidate_holds_commit", commit_sha)] * 2
    assert not _events(data, "evolution_tx_abandoned") and not _events(data, "evolution_tx_reconciled")
    assert body_adoption.read(data) == {} and not (data / "state" / "pending_restart_verify.json").exists()
    assert git(serving, "rev-parse", "HEAD") == scene.serving_head and git(serving, "status", "--porcelain") == ""

    # The restored provenance authorizes the restart-bound adoption; the adopted serving SHA absorbs it.
    assert body_adoption.authorize_for_task(data, "evo-crash", commit_sha, "evolution restart") is True
    body_adoption.abandon(data, "test")
    git(serving, "merge", "--ff-only", "-q", commit_sha)  # what the adoption lands
    absorbed_campaign = scene.boot("boot-gen-4", observed=commit_sha)
    absorbed = absorbed_campaign["transaction_history"][-1]
    assert (absorbed["commit_sha"], absorbed["cycle_outcome"]) == (commit_sha, "absorbed")
    assert "active_transaction" not in absorbed_campaign and absorbed_campaign["absorbed_cycles_done"] == 1
    assert [row.get("cycle_outcome") for row in _outcome_tags(data)] == ["absorbed"]


@pytest.mark.parametrize("loss", ["checkout_removed", "branch_rewound"])
def test_a_later_boot_abandons_a_candidate_commit_its_candidate_no_longer_holds(tmp_path, monkeypatch, loss):
    """Only evidence of loss abandons: the candidate checkout is gone, or its branch no longer
    carries the commit (adoption could never name it). The reviewed commit left behind in the
    object store authorizes nothing: no later boot adopts a candidate without its transaction."""
    scene = _crash_recovered_cycle(tmp_path, monkeypatch, "evo-lost")
    if loss == "checkout_removed":
        git(scene.serving, "worktree", "remove", "--force", str(scene.candidate))
    else:
        git(scene.candidate, "reset", "-q", "--hard", scene.serving_head)

    campaign = scene.boot("boot-gen-2")
    assert "active_transaction" not in campaign and "post_task_backlog_id" not in campaign
    lost = campaign["transaction_history"][-1]
    assert (lost["commit_sha"], lost["cycle_outcome"], lost["abandoned_reason"]) == (
        scene.commit_sha, "abandoned", "commit_not_reachable_at_boot")
    assert sum(campaign["objective_repeat_counts"].values()) == 1
    assert campaign["pending_owner_report"]["cycle_outcome"] == "abandoned"
    assert [row.get("cycle_outcome") for row in _outcome_tags(scene.data)] == ["abandoned"]
    assert not _events(scene.data, "evolution_tx_awaiting_adoption")
    scene.boot("boot-gen-3")
    assert body_adoption.read(scene.data) == {} and git(scene.serving, "rev-parse", "HEAD") == scene.serving_head


@pytest.mark.parametrize("unknown", ["registry_unreadable", "git_unanswered"])
def test_an_unreadable_candidate_never_proves_its_commit_lost(tmp_path, monkeypatch, unknown):
    """An unreadable registry or a Git read that does not answer keeps the transaction open for
    this generation; the next boot that can read the candidate finds the commit still held."""
    import subprocess

    scene = _crash_recovered_cycle(tmp_path, monkeypatch, "evo-unknown")
    registry = scene.data / "state" / "subagent_worktrees.json"
    original_registry, original_run = registry.read_text(encoding="utf-8"), subprocess.run
    if unknown == "registry_unreadable":
        registry.write_text("{", encoding="utf-8")
    else:
        def run(cmd, *args, **kwargs):
            if str(kwargs.get("cwd") or "") == str(scene.candidate):
                raise subprocess.TimeoutExpired(cmd, 30)
            return original_run(cmd, *args, **kwargs)

        monkeypatch.setattr(subprocess, "run", run)

    kept = scene.boot("boot-gen-2")["active_transaction"]
    assert (kept["commit_sha"], kept["restart_required"], kept["restart_verified"]) == (scene.commit_sha, True, False)
    assert [row["reason"] for row in _events(scene.data, "evolution_tx_awaiting_adoption")] == ["candidate_unreadable"]
    assert not _events(scene.data, "evolution_tx_abandoned") and not _outcome_tags(scene.data)

    registry.write_text(original_registry, encoding="utf-8")
    monkeypatch.setattr(subprocess, "run", original_run)
    assert scene.boot("boot-gen-3")["active_transaction"]["commit_sha"] == scene.commit_sha
    assert [row["reason"] for row in _events(scene.data, "evolution_tx_awaiting_adoption")] == [
        "candidate_unreadable", "candidate_holds_commit"]
    assert body_adoption.read(scene.data) == {} and git(scene.serving, "rev-parse", "HEAD") == scene.serving_head
