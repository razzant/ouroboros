"""The owner-visible trace of review decisions Blocking would have stopped (``ouroboros/review_overrides.py``).

One writer keeps the event, the install counter and the work's card fact (count, latest reason) on its root task;
a commit Git records later joins ``landed`` only from the call that met such a decision or from the reviewed attempt
its author continuation names; the agent's restart notice and a settled adoption name those commits in one line,
"running" only on the adoption record's word.
"""
from __future__ import annotations

import copy
import json
import subprocess
import time
import uuid
from types import SimpleNamespace

import pytest

from ouroboros.review_overrides import (
    COUNTER_REL, card_line, landed_line, note_override_landing, record_override,
)
from ouroboros.utils import read_json_dict
from tests._git_review_pipeline_shared import _make_ctx
from tests._store_shared import read_result, seed_result

GATE = "over a review Blocking would have stopped (the review gate did not prohibit continuation)"


def _events(ctx):
    path = ctx.drive_root / "logs" / "events.jsonl"
    return [row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if row.get("type") == "review_advisory_override"]


def _decide(ctx, reason, message="finding"):
    ctx._last_review_block_reason = reason
    record_override(ctx, message)


def _commit(repo, name):
    (repo / name).write_text(name, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", name], cwd=repo, check=True, capture_output=True)
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()


def _landed(ctx, *rows):
    (ctx.drive_root / COUNTER_REL).parent.mkdir(parents=True, exist_ok=True)
    (ctx.drive_root / COUNTER_REL).write_text(json.dumps({"count": len(rows), "landed": [
        {"sha": sha, "reason": reason, "task_id": "t", "at": "2026-11-11T10:00:00+00:00"} for sha, reason in rows]}),
        encoding="utf-8")


def test_each_decision_counts_on_its_work_card_beside_the_event_and_the_counter(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    ctx = _make_ctx(tmp_path)
    ctx.task_id = "root-1"
    seed_result(ctx.drive_root, "root-1", {"status": "running", "chat_id": 1})
    child = copy.copy(ctx)
    child.task_id, child.task_metadata = "child-1", {"root_task_id": "root-1"}
    _decide(ctx, "critical_findings", "[CRITICAL] tests_affected: broken")
    _decide(child, "review_quorum", "Too few reviewers answered")
    fact = read_result(ctx.drive_root, "root-1")["review_overrides"]
    assert fact["count"] == 2 and fact["latest"]["reason"] == "review_quorum" and fact["latest"]["task_id"] == "child-1"
    assert fact["latest"]["text"] == "Too few reviewers answered"
    first, second = _events(ctx)
    assert second["task_id"] == "child-1" and second["work_task_id"] == "root-1" and second["review_overrides"] == fact
    assert second["decision_authority"] == "advisory" and second["block_reason"] == "review_quorum"
    assert first["review_overrides"]["count"] == 1, "the live copy is the fact this decision left"
    counter = read_json_dict(ctx.drive_root / COUNTER_REL)
    assert counter["count"] == 2 and [row["block_reason"] for row in counter["recent"]] == ["critical_findings", "review_quorum"]
    assert card_line(fact) == "Review: 2× continued where Blocking would have stopped · latest: review quorum"
    assert card_line({"count": 0}) == card_line(None) == "", "no decision, no field"
    # A work with no record to hold it: no partial row is minted; the event and the counter keep the decision.
    ghost = copy.copy(ctx)
    ghost.task_id = "ghost"
    _decide(ghost, "infra_failure")
    assert read_result(ctx.drive_root, "ghost") is None and "review_overrides" not in _events(ctx)[-1]
    assert read_json_dict(ctx.drive_root / COUNTER_REL)["count"] == 3
    _decide(ctx, "critical_findings", "word " * 100)  # a long reason keeps a head that says it is cut
    latest = read_result(ctx.drive_root, "root-1")["review_overrides"]["latest"]["text"]
    assert len(latest) <= 300 and latest.endswith("word…") and latest.startswith("word word")


def test_a_commit_lands_over_a_decision_only_from_its_call_or_the_attempt_it_continues(tmp_path):
    from ouroboros.review_state import load_state, make_repo_key
    from ouroboros.tools.commit_gate import _record_commit_attempt
    from ouroboros.tools.git_review_cycle import _reset_commit_review_state

    ctx = _make_ctx(tmp_path)
    ctx.task_id, install = "t1", ctx.drive_root

    def reviewed(decision=""):  # one call whose review hands its outcome back before Git (Advisory, Cyber Pro)
        _reset_commit_review_state(ctx)
        _record_commit_attempt(ctx, "change", "reviewing", phase="review")
        if decision:
            _decide(ctx, decision)
        _record_commit_attempt(ctx, "change", "reviewed", phase="review_only", block_reason=decision or "author_decision_required")
        return ctx._current_review_attempt_number

    def continues(attempt=None):  # a later call; its validated reference resolved to that durable attempt
        _reset_commit_review_state(ctx)
        if attempt is not None:
            ctx._author_commit_source = load_state(ctx.drive_root).latest_attempt_for(
                repo_key=make_repo_key(ctx.repo_dir), task_id="t1", tool_name="commit_reviewed", attempt=attempt)

    def landed():
        return [row["sha"][:1] for row in (read_json_dict(install / COUNTER_REL) or {}).get("landed") or []]

    over, clean = reviewed("critical_findings"), reviewed()
    assert load_state(ctx.drive_root).latest_attempt_for(attempt=over).review_override_reason == "critical_findings"
    continues(clean)
    note_override_landing(ctx, "a" * 40)
    assert landed() == [], "a commit over a review that met no decision is not over the earlier one"
    continues(over)
    note_override_landing(ctx, "b" * 40)
    assert landed() == ["b"]
    continues()  # a later call that continues nothing keeps no stale reference alive
    note_override_landing(ctx, "c" * 40)
    assert landed() == ["b"]
    continues(clean)
    _decide(ctx, "preflight")  # the continuation's own preflight, waved through in this very call
    note_override_landing(ctx, "d" * 40)
    assert landed() == ["b", "d"]
    for index in range(70):
        note_override_landing(ctx, f"{index:040d}")
    assert len(landed()) == 64, "the newest 64 landings are kept"


@pytest.mark.parametrize("case", ["warm", "restarted", "successor", "clean"])
def test_the_author_continuation_lands_over_its_attempts_decision_in_any_context(tmp_path, monkeypatch, case):
    """Through the real gate: Advisory hands the reviewed outcome back before Git and the author's continuation commits
    it. Its validated reference names the durable attempt that keeps the decision, so the same context, a fresh one (a
    restarted worker) and a Continue successor land the commit alike; a review that met no decision lands nothing."""
    from ouroboros import commit_admission, owner_continue
    from ouroboros.mutation_attribution import capture_mutation_baseline
    from ouroboros.review_dispatch import invoke_review_paid_stamp
    from ouroboros.review_ledger import CouplingOutcome
    from ouroboros.review_state import load_state
    from ouroboros.task_results import write_task_result
    from ouroboros.tools import git, review, review_helpers
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "advisory")
    monkeypatch.setattr(review_helpers, "check_worktree_readiness", lambda *a, **k: [])
    monkeypatch.setattr(commit_admission, "release_metadata_diagnostics", lambda *a, **k: {
        "status": "not_applicable", "findings": [], "unavailable": []})
    for name, stub in (("_managed_candidate_needs_proof", lambda *a: False), ("_run_review_preflight_tests", lambda *a, **k: None),
                       ("_post_commit_result", lambda *a, **k: None), ("_auto_push", lambda *a, **k: "")):
        monkeypatch.setattr(git, name, stub)
    ctx = _make_ctx(tmp_path)
    ctx.task_id, ctx.branch_dev = "author", "ouroboros"
    (ctx.repo_dir / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    subprocess.run(["git", "add", "VERSION"], cwd=ctx.repo_dir, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "version"], cwd=ctx.repo_dir, check=True, capture_output=True)
    for task_id in ("author", "successor"):
        write_task_result(ctx.drive_root, task_id, "running")
        capture_mutation_baseline(ctx.drive_root, task_id, [{"surface_type": "system_repo", "host_root": str(ctx.repo_dir)}])
    (ctx.repo_dir / "change.py").write_text("value = 2\n", encoding="utf-8")

    def reviewer(actual, message, **kw):  # the paid model alone is scripted; the gate's decision seam is real
        invoke_review_paid_stamp(actual._review_paid_stamp)
        finding = {"item": "code_quality", "severity": "critical", "verdict": "PASS" if case == "clean" else "FAIL",
                   "reason": "wrong value"}
        actual._last_triad_raw_results = [{"slot_id": "critic", "status": "responded", "parsed": [finding], "raw_text": "x"}]
        if case != "clean":
            actual._last_review_critical_findings, actual._last_review_block_reason = [finding], "critical_findings"
            assert review._handle_review_block_or_warning(actual, False, "critical feedback", "") is None
        return None, CouplingOutcome(verdict="PASS", status="responded"), "", []

    monkeypatch.setattr(git, "_run_parallel_review", reviewer)
    first = git._repo_commit_push(ctx, "Change value", skip_advisory_review=True)
    assert first.startswith("Review outcome returned before commit"), first
    reference = json.loads(first.split("\n", 1)[1])["review_reference"]
    author = ctx
    if case in {"restarted", "successor"}:  # a context built afresh, as a restarted worker or the successor builds it
        author = ToolContext(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root,
                             task_id="successor" if case == "successor" else "author")
        author.branch_dev = "ouroboros"
    monkeypatch.setattr(owner_continue, "recorded_continuation", lambda _root, before, after: (before, after) == (
        "author", "successor"))  # the owner's recorded Continue from the author to its successor
    monkeypatch.setattr(git, "_run_parallel_review", lambda *a, **k: pytest.fail("the continuation buys no review"))
    said = git._repo_commit_push(author, "Change value", review_reference=reference, author_disposition={
        "disposition": "rejected", "rationale": "I read the finding and choose to continue."})
    assert load_state(ctx.drive_root).attempts[-1].status == "succeeded", said
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ctx.repo_dir, check=True, capture_output=True,
                         encoding="utf-8").stdout.strip()
    landed = [row["sha"] for row in (read_json_dict(ctx.drive_root / COUNTER_REL) or {}).get("landed") or []]
    assert landed == ([] if case == "clean" else [sha]), (case, said)


def test_the_restart_line_names_only_landed_commits_of_its_range_and_says_running_only_when_adopted(tmp_path):
    ctx = _make_ctx(tmp_path)
    repo = ctx.repo_dir
    old = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    base = _commit(repo, "base")
    first, plain, last = _commit(repo, "one"), _commit(repo, "two"), _commit(repo, "three")
    _landed(ctx, (old, "critical_findings"), (first, "critical_findings"), (last, "review_quorum"))
    line = landed_line(repo, ctx.drive_root, base=base, target="HEAD", running=False)
    assert line == (f"Committed {GATE}: {last[:12]} (review quorum), {first[:12]} (critical findings); deployment "
                    "unconfirmed, as no record yet binds a running process to them.")
    assert old[:12] not in line and plain[:12] not in line
    running = landed_line(repo, ctx.drive_root, base=base, target=last, running=True)
    assert running == f"Now running {GATE}: {last[:12]} (review quorum), {first[:12]} (critical findings)."
    for said in (line, running):
        assert "merged" not in said and "accepted" not in said
    assert landed_line(repo, ctx.drive_root, base=last, target="HEAD", running=False) == ""
    assert landed_line(repo, ctx.drive_root, base="", target="HEAD", running=False) == "", "an unknown base names nothing"


def test_the_agent_restart_notice_carries_the_line_from_the_running_baseline(tmp_path, monkeypatch):
    from ouroboros import server_process, server_restart

    ctx = _make_ctx(tmp_path)
    base = _commit(ctx.repo_dir, "base")
    landed = _commit(ctx.repo_dir, "change")
    _landed(ctx, (landed, "critical_findings"))
    monkeypatch.setattr(server_process, "server_source_baseline", lambda: base)
    monkeypatch.setattr(server_restart, "_AGENT_RESTART", {})
    monkeypatch.setattr(server_restart, "_started_restart_work", lambda: {})
    sent: list = []
    host = SimpleNamespace(REPO_DIR=ctx.repo_dir, DRIVE_ROOT=ctx.drive_root, load_state=lambda: {"owner_chat_id": 7},
                           send_with_budget=lambda chat, text, **kw: sent.append((text, kw.get("system_type"))))
    server = SimpleNamespace(uuid=uuid, time=time, _pending_restart={}, _live_running_task_ids=lambda _ctx: ["busy"],
                             _perform_supervisor_restart=lambda *a, **k: pytest.fail("the drain waits for live work"))
    server_restart._handle_restart_in_supervisor(server, {"type": "restart_request", "reason": "body change"}, host)
    assert sent[0] == (f"♻️ Restart requested by agent: body change\nCommitted {GATE}: {landed[:12]} (critical "
                       "findings); deployment unconfirmed, as no record yet binds a running process to them.",
                       "restart_notice")
    sent.clear()
    _landed(ctx)  # nothing landed over a decision: the notice says only what it always said
    server_restart._AGENT_RESTART.clear()
    server_restart._handle_restart_in_supervisor(server, {"type": "restart_request", "reason": "body change"}, host)
    assert sent[0] == ("♻️ Restart requested by agent: body change", "restart_notice")


@pytest.mark.parametrize("outcome,said", [("adopted", "Now running"), ("unconfirmed", "Committed"), ("not_applied", None)])
def test_a_settled_adoption_names_its_landed_commits_running_only_when_adopted(tmp_path, monkeypatch, outcome, said):
    from ouroboros import body_adoption

    ctx = _make_ctx(tmp_path)
    previous = _commit(ctx.repo_dir, "previous")
    candidate = _commit(ctx.repo_dir, "candidate")
    _landed(ctx, (candidate, "critical_findings"))
    monkeypatch.setattr(body_adoption, "finalize_on_boot", lambda *a, **k: {
        "outcome": outcome, "note": f"Adoption note ({outcome}).", "adoption_id": "a1", "task_id": "",
        "candidate": candidate, "previous": previous, "branch": "ouroboros", "tests": {"tier": "smoke"}})
    monkeypatch.setattr("supervisor.git_ops_reset._record_checkout_facts", lambda facts: None)
    sent: list = []
    monkeypatch.setattr("supervisor.state.load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda chat, text, **kw: sent.append(text))
    result = body_adoption.settle_on_boot(ctx.drive_root, ctx.repo_dir, supervisor_ready=True)
    named = f"{candidate[:12]} (critical findings)"
    if said is None:
        assert named not in result["note"] and named not in "".join(sent)
    else:
        assert f"{said} {GATE}: {named}" in result["note"] and f"{said} {GATE}: {named}" in sent[0]


def test_a_failing_join_or_store_never_fails_the_gate_the_commit_or_the_notice(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    ctx.task_id, ctx._current_review_attempt_number = "t1", 2

    def unreadable(*_args):
        raise OSError("repository identity unreadable")

    monkeypatch.setattr("ouroboros.review_state.make_repo_key", unreadable)
    _decide(ctx, "critical_findings")  # the decision still reaches its event and the counter
    assert _events(ctx)[-1]["block_reason"] == "critical_findings"
    assert read_json_dict(ctx.drive_root / COUNTER_REL)["count"] == 1
    ctx._review_override_this_call = ""
    ctx._author_commit_source = SimpleNamespace(review_override_reason="critical_findings")

    def unwritable(*_args):
        raise OSError("counter unwritable")

    monkeypatch.setattr("ouroboros.review_overrides.update_json_locked", unwritable)
    note_override_landing(ctx, "a" * 40)  # Git already succeeded: the landing's failure is logged, never raised
    assert "landed" not in read_json_dict(ctx.drive_root / COUNTER_REL)
    assert landed_line(ctx.repo_dir, None, base="a", target="b", running=False) == "", "an unreadable counter names nothing"


_CARD_ON_LOAD = """
import { installDom, restoreDom, walkCard } from '%(fixture)s';
import { createChatInstance } from '%(chat)s';
let input = ''; for await (const chunk of process.stdin) input += chunk;
const messages = JSON.parse(input);
const reads = [];
const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () => {
    if (String(url).startsWith('/api/tasks/')) reads.push(String(url));
    return String(url).startsWith('/api/chat/history') ? { messages } : { active_direct_turns: [] };
} }));
const instance = createChatInstance({ ws: { on() { return () => {}; }, isConnected: () => true, send() {} },
    state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
    stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
        gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
    chatId: 1, idPrefix: 'chat', mountEl: mount });
await instance.refreshHistory({ revision: 1 });
const card = (id) => walkCard(globalThis.document.byId.get('chat-messages'), id);
const meta = (id) => card(id)?.querySelector('[data-live-meta]')?.innerHTML ?? null;
const painted = { gate: meta('gate-work'), plain: meta('plain-work'), reads,
    finished: ['gate-work', 'plain-work'].map((id) => card(id)?.dataset.finished === '1') };
instance.destroy(); restoreDom(prior);
process.stdout.write(JSON.stringify(painted));
"""


def test_a_page_load_shows_a_finished_works_field_from_the_history_endpoint(tmp_path):
    """The real history endpoint carries the work's fact on its summary row and the real card paints it before Details,
    with no detail read; a finished work without decisions shows nothing. Without the overlay the line is absent."""
    import asyncio
    import shutil
    from pathlib import Path

    from ouroboros import dialogue_store
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.post_task_checkpoint import set_root_post_task_checkpoint
    from ouroboros.post_task_synthesis import _record_task_facts
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from ouroboros.utils import utc_now_iso

    ctx = _make_ctx(tmp_path)
    root = ctx.drive_root
    env = SimpleNamespace(drive_root=root, repo_dir=ctx.repo_dir)
    for task_id in ("gate-work", "plain-work"):
        task = {"id": task_id, "type": "task", "chat_id": 1}
        write_task_result(root, task_id, STATUS_SCHEDULED, chat_id=1, root_task_id=task_id,
                          result="Task accepted and scheduled.")
        # The work's progress row, as the message bus stores it: the card the page load draws.
        dialogue_store.append(root, "progress", {"ts": utc_now_iso(), "type": "send_message", "task_id": task_id,
                                                 "is_progress": True, "direction": "out", "chat_id": 1,
                                                 "text": "Changing my body", "content": "Changing my body",
                                                 "role": "assistant", "system_type": ""})
        if task_id == "gate-work":
            ctx.task_id = task_id
            _decide(ctx, "critical_findings", "[CRITICAL] tests_affected: the new refusal path has no test")
        _store_task_result(env, task, "Done.", {"rounds": 1, "cost": 0}, {"tool_calls": []})
        _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, root / "logs")
        set_root_post_task_checkpoint(env, task, "completed")  # post-task synthesis settled: the work is finished
    response = asyncio.run(make_chat_history_endpoint(root)(SimpleNamespace(query_params={"chat_id": "1"})))
    messages = json.loads(response.body)["messages"]
    node = shutil.which("node")
    assert node, "Node is required for the Python-to-web page-load witness"
    web = Path(__file__).resolve().parents[1] / "web"
    script = _CARD_ON_LOAD % {"fixture": (web / "tests" / "chat_dom_fixture.js").as_uri(),
                              "chat": (web / "modules" / "chat.js").as_uri()}
    probe = subprocess.run([node, "--input-type=module", "-e", script], input=json.dumps(messages), encoding="utf-8",
                           capture_output=True, timeout=60, check=True)
    painted = json.loads(probe.stdout)
    assert "Review: 1× continued where Blocking would have stopped · latest: critical findings" in painted["gate"], painted
    assert painted["plain"] is not None and "continued where Blocking" not in painted["plain"], painted
    assert painted["reads"] == [], "the page load needs no per-card detail read"
    assert painted["finished"] == [True, True], "both are finished works' cards"
    carried = {(m["task_id"], m["review_overrides"]["count"]) for m in messages if "review_overrides" in m}
    assert carried == {("gate-work", 1)}, "only the work with a decision carries its fact, on its history row"
