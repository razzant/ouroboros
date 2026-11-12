"""#708: what an ended evolution cycle's cleanup moved reaches its card, result and history.

The owner chose to keep the cleanup (the whole dirty checkout goes to a stash, another HEAD to a
local branch, the base comes back) and to show its result. Every scenario runs the real lifecycle
over a real temporary Git repository, the real outbox and send handler, and reads the facts from
the history endpoint and the task result a reload uses, then retrieves the bytes with what the row
printed. A row never claims whose bytes they were, that they stay forever or that it was sent once.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import shutil
import subprocess
from collections import deque
from types import SimpleNamespace

import pytest

TASK = "evo-task"
CHAT = 7
GIT = ("git", "-c", "user.name=Test", "-c", "user.email=test@example.com")


def git(repo, *args, check=True):
    return subprocess.run([*GIT, *args], cwd=str(repo), check=check, capture_output=True, text=True).stdout.strip()


class Bus:
    """The event queue the outbox admits to; ``refuse`` loses a live send."""

    def __init__(self):
        self.items, self.refuse = [], False

    def put(self, item):
        if self.refuse:
            raise OSError("queue refused")
        self.items.append(item)

    def put_unless_outstanding(self, item):
        self.put(item)
        return True

    def delivery_outstanding(self, delivery_id):
        return any(item.get("delivery_id") == delivery_id for item in self.items)

    def drain(self):
        items, self.items = self.items, []
        return items


@pytest.fixture
def cycle(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result
    from supervisor import events_chat_delivery as delivery
    from supervisor import evolution_lifecycle, git_ops, message_bus, queue, state, workers

    repo, data = tmp_path / "repo  with  spaces", tmp_path / "data"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "ouroboros")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    base = git(repo, "rev-parse", "HEAD")

    state.init(data)
    queue.init(data)
    queue.init_queue_refs([], {}, {"value": 0})
    state.save_state({"owner_chat_id": CHAT, "owner_id": CHAT, "evolution_mode_enabled": True,
                      "evolution_owner_stopped": False})
    git_ops.init(repo, data, "")
    queue.RUNNING.clear()
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="test")
    tx = evolution_lifecycle.begin_evolution_transaction(TASK, cycle=1, campaign=campaign)
    assert tx["base_head"] == base
    write_task_result(data, TASK, "running", chat_id=CHAT, root_task_id=TASK, suggested_name="Improve the thing",
                      outcome_axes={"execution": {"status": "ok"}}, result="No change was needed.")

    bus, frames = Bus(), []
    monkeypatch.setattr(workers, "get_event_q", lambda create=True: bus)
    monkeypatch.setattr(delivery, "_DELIVERED_MESSAGE_IDS", deque(maxlen=1000))
    monkeypatch.setattr(message_bus, "DATA_DIR", data)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    ctx = SimpleNamespace(DRIVE_ROOT=data, task_registry={}, send_with_budget=message_bus.send_with_budget,
                          append_jsonl=lambda *a: None, RUNNING=queue.RUNNING)

    def deliver():
        sent = bus.drain()
        for event in sent:
            delivery._handle_send_message(dict(event), ctx)
        return sent

    def finish(chat_id=CHAT):
        write_task_result(data, TASK, "completed", chat_id=chat_id, root_task_id=TASK)

    def end(*, chat_id=CHAT, **overrides):
        finish(chat_id)
        args = {"cost_usd": 0.1, "outcome_axes": {"execution": {"status": "ok"}}, "rounds": 2, "transaction": tx}
        return evolution_lifecycle.update_evolution_campaign_after_task(TASK, **{**args, **overrides})

    def history(chat_id=CHAT):
        from ouroboros.gateway.history import make_chat_history_endpoint

        response = asyncio.run(make_chat_history_endpoint(data)(SimpleNamespace(query_params={"chat_id": str(chat_id)})))
        return json.loads(response.body)["messages"]

    yield SimpleNamespace(repo=repo, data=data, base=base, tx=tx, bus=bus, frames=frames, ctx=ctx, deliver=deliver,
                          end=end, finish=finish, history=history, campaign=evolution_lifecycle._read_evolution_campaign)
    queue.RUNNING.clear()


def rows(scene, chat_id=CHAT):
    return [row for row in scene.history(chat_id) if str(row.get("card_row_id") or "").startswith("evolution-cleanup:")]


def dirty(scene):
    (scene.repo / "tracked.txt").write_text("somebody's edit\n", encoding="utf-8")
    (scene.repo / "untracked.txt").write_text("somebody's new file\n", encoding="utf-8")


def ahead(scene):
    (scene.repo / "wip.txt").write_text("unreviewed\n", encoding="utf-8")
    git(scene.repo, "add", "wip.txt")
    git(scene.repo, "commit", "-qm", "unreviewed leftover")
    return git(scene.repo, "rev-parse", "HEAD")


def stash_oids(scene):
    return git(scene.repo, "stash", "list", "--format=%H").split()


def fault_git(monkeypatch, verb, *, raises=None, when=None):
    """The real Git, except one verb (optionally only ``when``) refuses or ``raises``."""
    from supervisor import git_ops

    real = git_ops.git_capture

    def capture(cmd, **kwargs):
        if verb in cmd[1:3] and (when is None or when(cmd)):
            if raises is not None:
                raise raises
            return 128, "", f"fatal: {verb} refused"
        return real(cmd, **kwargs)

    monkeypatch.setattr(git_ops, "git_capture", capture)


def stash_reads(*failing):
    """A ``when`` for ``fault_git``: only these stash-list reads fail (1 is the read before the push, 2 after it)."""
    seen = []

    def when(cmd):
        if cmd[-1] != "refs/stash":
            return False
        seen.append(cmd)
        return len(seen) in failing
    return when


def nested_only_dirt(scene):
    """Dirt only inside a nested repository: ``git status`` is dirty, ``git stash push`` exits 0 and saves nothing."""
    from supervisor import evolution_lifecycle

    nested = scene.repo / "sub"
    nested.mkdir()
    git(nested, "init", "-q", "-b", "ouroboros")
    (nested / "s.txt").write_text("s\n", encoding="utf-8")
    git(nested, "add", ".")
    git(nested, "commit", "-qm", "s")
    git(scene.repo, "add", "sub")
    git(scene.repo, "commit", "-qm", "gitlink")
    base = git(scene.repo, "rev-parse", "HEAD")
    assert evolution_lifecycle.update_evolution_transaction(TASK, base_head=base)
    (nested / "s.txt").write_text("changed\n", encoding="utf-8")
    assert git(scene.repo, "status", "--porcelain") == "M sub"
    return {**scene.tx, "base_head": base}


def assert_unconfirmed(text):
    """An unread stash list is neither a move nor its absence: the row says Git ran it and how to look."""
    assert "moved local changes" not in text
    assert "(whether it saved anything is unconfirmed)" in text.splitlines()[0]
    assert "was not confirmed" in text and " stash list`" in text


def assert_honest_words(text):
    """A whole-tree stash carries no claim about whose bytes it holds, how long they stay or a single send."""
    assert re.search(r"whole dirty checkout", text)
    assert re.search(r"can include", text)
    lowered = text.lower()
    for claim in ("your edits", "this cycle's edits", "this task's edits", "forever", "permanently",
                  "exactly once", "will stay", "guarantee", "git keeps"):
        assert claim not in lowered, claim


def code_spans(text):
    return re.findall(r"`([^`]+)`", text)


def printed_restore_command(text):
    """The exact command the row printed, the way an owner copies it."""
    return next(part for part in code_spans(text) if part.startswith("git -C "))


def run_printed(command, **kwargs):
    """A printed command as copied, in the POSIX shell the row declares."""
    argv = ["sh", "-c", command] if shutil.which("sh") else shlex.split(command)
    return subprocess.run(argv, capture_output=True, text=True, **kwargs)


@pytest.fixture
def read_only():
    """Closes a directory to real Git, skipping where chmod does not bind (root, Windows); reopened for temp cleanup."""
    closed = []

    def close(path):
        path.chmod(0o555)
        closed.append(path)
        probe = path / ".probe"
        try:
            probe.write_text("", encoding="utf-8")
        except OSError:
            return
        probe.unlink()
        pytest.skip("a read-only directory does not stop this user here")

    yield close
    for path in closed:
        path.chmod(0o755)


def undeletable(scene, close):
    """Dirt real ``git stash push`` saves and cannot clear: it exits 1 with the entry stored, the removable
    untracked file already gone and the tracked edit still on disk (its reset never ran)."""
    dirty(scene)
    locked = scene.repo / "locked"
    locked.mkdir()
    (locked / "held.txt").write_text("held\n", encoding="utf-8")
    close(locked)


def test_no_op_cycle_row_names_the_exact_stash_and_branch_through_reload(cycle):
    dirty(cycle)
    leftover = ahead(cycle)

    recorded = cycle.end()["transaction"]

    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert recorded["cycle_outcome"] == "no_op" and recorded["cleanup_status"] == "reset_to_base"
    assert recorded["cleanup_stash_oids"] == [oid]
    assert recorded["cleanup_preserved_ref"].startswith("evolution-leftover-")
    assert git(cycle.repo, "rev-parse", recorded["cleanup_preserved_ref"]) == leftover == recorded["cleanup_preserved_head"]
    assert git(cycle.repo, "rev-parse", "HEAD") == cycle.base and git(cycle.repo, "status", "--porcelain") == ""

    # The owed row reaches the live frame and the reload, on the cycle's own card, never as a bubble.
    sent = cycle.deliver()
    assert [event["system_type"] for event in sent] == ["evolution_cleanup"]
    live = [frame for frame in cycle.frames if frame.get("card_row_id")]
    assert len(live) == 1 and live[0]["task_id"] == TASK and live[0]["card_row"] == "timeline"
    assert live[0]["terminal_receipt"] == "retained_work", "a transport's progress toggle must not hide it"
    replayed = rows(cycle)
    assert len(replayed) == 1 and replayed[0]["task_id"] == TASK and replayed[0]["role"] == "system"
    assert replayed[0]["card_row_id"] == f"evolution-cleanup:{recorded['transaction_id']}"
    text = replayed[0]["text"]
    for exact in (str(cycle.repo), oid, recorded["cleanup_preserved_ref"], leftover, cycle.base):
        assert exact in text, exact
    assert_honest_words(text)
    assert not [row for row in cycle.history() if row.get("system_type") == "evolution_cycle_outcome"]

    # The same facts on the task result and the campaign's two histories.
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_results import load_task_result

    result = load_task_result(cycle.data, TASK)
    fact = normalize_outcome_axes(result)["execution"]["evolution_cleanup"]
    assert fact["stash_oids"] == [oid] and fact["preserved_ref"] == recorded["cleanup_preserved_ref"]
    assert fact["preserved_head"] == leftover
    assert fact["repo"] == str(cycle.repo) and fact["status"] == "reset_to_base"
    assert "stash" in fact["line"].lower() and "branch" in fact["line"].lower()
    campaign = cycle.campaign()
    for kept in (campaign["history"][-1]["transaction"], campaign["transaction_history"][-1]):
        assert kept["cleanup_stash_oids"] == [oid] and kept["cleanup_status"] == "reset_to_base"
        assert kept["cleanup_repo"] == str(cycle.repo) and kept["cleanup_at"]

    # What the row printed retrieves the bytes.
    assert "(POSIX shell)" in text
    run_printed(printed_restore_command(text), check=True)
    assert (cycle.repo / "tracked.txt").read_text(encoding="utf-8") == "somebody's edit\n"
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"
    assert git(cycle.repo, "show", f"{recorded['cleanup_preserved_ref']}:wip.txt") == "unreviewed"


@pytest.mark.parametrize("mirror", ["off", "on"])
def test_telegram_gets_the_row_whatever_its_progress_toggle_says(cycle, monkeypatch, tmp_path, mirror):
    """The real outbound event reaches the real plugin: this row is typed as retained work, never progress."""
    from ouroboros.event_bus import CHAT_OUTBOUND
    from supervisor import message_bus
    from tests.test_telegram_markdown import _load_converter
    from tests.test_telegram_subagent_cards import _Rec, _setup

    published = []
    monkeypatch.setattr(message_bus, "publish_event", lambda name, event: published.append((name, event)))
    dirty(cycle)
    cycle.end()
    cycle.deliver()
    [event] = [event for name, event in published if name == CHAT_OUTBOUND and event.get("card_row_id")]

    plugin, api = _setup(tmp_path / "telegram", monkeypatch, TELEGRAM_MIRROR_PROGRESS=mirror)
    asyncio.run(plugin._make_outbound(api)(event))

    assert len(_Rec.sent) == 1
    assert stash_oids(cycle)[0] in _Rec.sent[0][1]
    html = _load_converter()(_Rec.sent[0][1])
    assert f"<code>{cycle.repo}</code>" in html, "the real conversion keeps the path's exact spacing"


def test_dirty_checkout_at_base_says_only_the_stash_happened(cycle):
    dirty(cycle)

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "stashed_dirty" and not recorded.get("cleanup_preserved_ref")
    [row] = rows(cycle)
    assert git(cycle.repo, "rev-parse", "refs/stash") in row["text"] and "branch" not in row["text"].lower()
    assert "reset" not in row["text"].lower().splitlines()[0]


def test_abandoned_cycle_keeps_its_report_and_its_row_is_separate(cycle, monkeypatch):
    from supervisor import evolution_lifecycle

    sent = []
    monkeypatch.setattr(evolution_lifecycle, "notify_owner_cycle_outcome", lambda campaign, tx: sent.append(dict(tx)))
    assert evolution_lifecycle.update_evolution_transaction(TASK, rescue_ref="refs/ouroboros/rescue/test")
    dirty(cycle)

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cycle_outcome"] == "abandoned" and recorded["cleanup_status"] == "stashed_dirty"
    assert [row["cycle_outcome"] for row in sent] == ["abandoned"]
    assert "pending_owner_report" not in cycle.campaign()
    assert len(rows(cycle)) == 1


@pytest.mark.parametrize("failure", ["preserve_refused", "preserve_raises", "reset_raises"])
def test_partial_success_keeps_the_stash_and_the_first_failure(cycle, monkeypatch, failure):
    """A later ref or reset failure neither hides the stash that succeeded nor loses its own cause."""
    from supervisor import git_ops

    dirty(cycle)
    leftover = ahead(cycle)
    if failure == "preserve_refused":
        monkeypatch.setattr(git_ops, "preserve_local_ref_branch", lambda *a, **k: (False, "branch refused"))
    elif failure == "preserve_raises":
        def boom(*_a, **_k):
            raise RuntimeError("branch exploded")
        monkeypatch.setattr(git_ops, "preserve_local_ref_branch", boom)
    elif failure == "reset_refused":
        fault_git(monkeypatch, "reset")
    else:
        fault_git(monkeypatch, "reset", raises=RuntimeError("reset exploded"))

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert recorded["cleanup_stash_oids"] == [oid]
    assert recorded["cleanup_status"] == {"preserve_refused": "skipped_preserve_failed", "preserve_raises": "error",
                                          "reset_refused": "reset_failed", "reset_raises": "error"}[failure]
    cause = {"preserve_refused": "branch refused", "preserve_raises": "branch exploded",
             "reset_refused": "reset refused", "reset_raises": "reset exploded"}[failure]
    assert cause in recorded["cleanup_error"]
    assert git(cycle.repo, "rev-parse", "HEAD") == leftover, "the checkout stays where the failure left it"
    [row] = rows(cycle)
    assert oid in row["text"] and cause in row["text"] and "stash" in row["text"].splitlines()[0].lower()
    assert leftover in row["text"], "the commits the checkout still holds are named"
    assert "returned" not in row["text"].splitlines()[0].lower()
    assert_honest_words(row["text"])
    run_printed(printed_restore_command(row["text"]), check=True)
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"


def test_an_unread_stash_list_is_unconfirmed_and_its_lookup_finds_the_saved_bytes(cycle, monkeypatch):
    dirty(cycle)
    fault_git(monkeypatch, "for-each-ref")

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert recorded["cleanup_stash_oids"] == [] and recorded["cleanup_stash_unconfirmed"] is True
    assert recorded["cleanup_status"] == "stash_unconfirmed" and recorded["cleanup_stash"].startswith("evolution-cycle-cleanup-")
    [row] = rows(cycle)
    assert oid not in row["text"] and recorded["cleanup_stash"] in row["text"]
    assert_unconfirmed(row["text"])
    assert_honest_words(row["text"])
    listing = run_printed(printed_restore_command(row["text"]), check=True)
    assert recorded["cleanup_stash"] in listing.stdout, "the printed lookup finds what Git did save"


def test_an_unread_stash_list_keeps_the_rescue_pointer_beside_its_own_note(cycle, monkeypatch):
    from supervisor import evolution_lifecycle

    assert evolution_lifecycle.update_evolution_transaction(TASK, rescue_ref="refs/ouroboros/rescue/test")
    assert evolution_lifecycle.update_evolution_transaction(TASK, recovery_hint="Recover with refs/ouroboros/rescue/test")
    fault_git(monkeypatch, "for-each-ref", when=stash_reads(1))

    recorded = cycle.end()["transaction"]

    assert recorded["cleanup_status"] == "already_clean"
    assert recorded["recovery_hint"].startswith("Recover with refs/ouroboros/rescue/test; the stash list could not be read")


@pytest.mark.parametrize("refusal", ["fake", "fake_long", "real_index_lock"])
def test_a_refused_stash_moved_nothing_and_publishes_no_move(cycle, monkeypatch, refusal):
    dirty(cycle)
    leftover = ahead(cycle)
    if refusal.startswith("fake"):
        fault_git(monkeypatch, "stash", when=lambda cmd: cmd[2] == "push")
    else:  # real Git refuses before it stores anything, so the read back finds no entry
        (cycle.repo / ".git" / "index.lock").touch()

    from supervisor import git_ops
    from ouroboros.utils import truncate_review_artifact
    capture, refused = git_ops.git_capture, []

    def observe(cmd, **kwargs):
        result = capture(cmd, **kwargs)
        if cmd[1:3] == ["stash", "push"]:
            if refusal == "fake_long":
                result = (result[0], result[1], "fatal: stash refused\n" + "diagnostic detail " * 40)
            refused.append(result)
        return result

    monkeypatch.setattr(git_ops, "git_capture", observe)
    recorded = cycle.end()["transaction"]
    cycle.deliver()

    [(code, _stdout, diagnostic)] = refused
    cause = diagnostic.strip()
    assert code != 0 and cause, "the real command's diagnostic is the oracle, not a Git-version-specific phrase"
    retained_cause = truncate_review_artifact(cause, 400)
    if refusal == "fake_long":
        assert retained_cause != cause and retained_cause.startswith(cause[:400])
    assert recorded["cleanup_status"] == "skipped_stash_failed" and retained_cause in recorded["recovery_hint"]
    assert retained_cause == recorded["cleanup_error"]
    assert not recorded.get("cleanup_stash") and not recorded.get("cleanup_stash_oids")
    assert not recorded.get("cleanup_stash_partial")
    assert git(cycle.repo, "rev-parse", "HEAD") == leftover, "never reset over unsaved changes"
    assert (cycle.repo / "untracked.txt").is_file() and rows(cycle) == []


@pytest.mark.parametrize("unread", [(), (1,), (1, 2)], ids=["read", "before_unread", "both_unread"])
def test_dirt_stash_cannot_save_is_never_reported_as_a_move(cycle, monkeypatch, unread):
    """Exit 0 saved nothing here. Only a readable list after the push can say so; an unread one stays unconfirmed."""
    transaction = nested_only_dirt(cycle)
    if unread:
        fault_git(monkeypatch, "for-each-ref", when=stash_reads(*unread))

    recorded = cycle.end(transaction=transaction)["transaction"]
    cycle.deliver()

    assert stash_oids(cycle) == [] and not recorded.get("cleanup_stash_oids")
    assert (cycle.repo / "sub" / "s.txt").read_text(encoding="utf-8") == "changed\n"
    if 2 in unread:  # Git ran it and the result is unread: said so, with the lookup, never as a move
        assert recorded["cleanup_status"] == "stash_unconfirmed" and recorded["cleanup_stash_unconfirmed"] is True
        [row] = rows(cycle)
        assert_unconfirmed(row["text"])
    else:  # the list after the push holds nothing under this label: nothing was saved
        assert recorded["cleanup_status"] == "dirty_not_stashable" and not recorded.get("cleanup_stash")
        assert rows(cycle) == []


def test_an_unread_push_keeps_the_earlier_stash_and_its_partial_outcome(cycle, monkeypatch):
    """A replay: an earlier attempt's stash is read and named; this push's result is unread and said so; the
    reset failed after it. Every saved byte stays findable from what the row printed."""
    leftover = ahead(cycle)
    dirty(cycle)
    label = f"evolution-cycle-cleanup-{cycle.tx['transaction_id']}"
    git(cycle.repo, "stash", "push", "--include-untracked", "-m", label)
    earlier = git(cycle.repo, "rev-parse", "refs/stash")
    (cycle.repo / "tracked.txt").write_text("a later edit\n", encoding="utf-8")
    fault_git(monkeypatch, "for-each-ref", when=stash_reads(2))
    fault_git(monkeypatch, "reset")

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "reset_failed" and recorded["cleanup_stash_oids"] == [earlier]
    assert recorded["cleanup_stash_unconfirmed"] is True and recorded["cleanup_preserved_ref"]
    [row] = rows(cycle)
    first = row["text"].splitlines()[0]
    assert first.startswith("Evolution cleanup moved local changes to a Git stash and kept local commits on a branch")
    assert "but the reset to the cycle's starting commit failed" in first
    assert earlier in row["text"] and leftover in row["text"] and "another entry under" in row["text"]
    listing = run_printed(printed_restore_command(row["text"].split("Look for it", 1)[1]), check=True).stdout
    later = [oid for oid in stash_oids(cycle) if oid != earlier]
    assert len(later) == 1 and listing.count(label) == 2
    git(cycle.repo, "stash", "apply", later[0])
    assert (cycle.repo / "tracked.txt").read_text(encoding="utf-8") == "a later edit\n"


@pytest.mark.parametrize("variant", ["candidate", "disabled", "lock_busy"])
def test_nothing_moved_means_no_row_and_no_claim(cycle, monkeypatch, variant):
    from supervisor import evolution_lifecycle, queue, update_merge

    transaction = cycle.tx
    if variant != "clean":
        dirty(cycle)
    if variant == "candidate":
        from ouroboros import body_adoption, body_candidate

        monkeypatch.setattr(body_candidate, "find", lambda *a, **k: {"candidate_id": "cand-1"})
        monkeypatch.setattr(body_adoption, "abandon", lambda *a, **k: None)
    elif variant == "disabled":
        monkeypatch.setenv("OUROBOROS_EVOLUTION_CYCLE_CLEANUP", "false")
    elif variant == "never_admitted":
        assert evolution_lifecycle.update_evolution_transaction(TASK, admission_refused={"reason": "x", "source": "test"})
        transaction = {**cycle.tx, "admission_refused": {"reason": "x", "source": "test"}}
    elif variant == "other_running":
        queue.RUNNING["other-task"] = {"task": {"id": "other-task"}}
    elif variant == "lock_busy":
        def busy():
            raise RuntimeError("busy")
        monkeypatch.setattr(update_merge, "acquire_update_lock", busy)

    recorded = cycle.end(transaction=transaction)["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == {
        "candidate": "candidate_retained", "clean": "already_clean", "disabled": "disabled",
        "never_admitted": "skipped_never_admitted", "other_running": "skipped_other_tasks_running",
        "lock_busy": "skipped_managed_update_lock_busy"}[variant]
    assert not stash_oids(cycle) and rows(cycle) == []
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_results import load_task_result

    assert "evolution_cleanup" not in normalize_outcome_axes(load_task_result(cycle.data, TASK))["execution"]
    if variant not in {"clean", "candidate"}:
        assert (cycle.repo / "untracked.txt").is_file()


def test_a_candidate_lookup_that_raises_moves_nothing_and_keeps_its_cause(cycle, monkeypatch):
    """The lookup now runs inside the cleanup's own guard: its failure is a recorded cause, never a raise
    that would cost the terminal its restart and report effects, and nothing is moved or said."""
    from ouroboros import body_candidate

    dirty(cycle)

    def unreadable(*_a, **_k):
        raise OSError("copy records unreadable")

    monkeypatch.setattr(body_candidate, "find", unreadable)

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "error" and "copy records unreadable" in recorded["cleanup_error"]
    assert not stash_oids(cycle) and (cycle.repo / "untracked.txt").is_file() and rows(cycle) == []


# --------------------------------------------------------------------------- #
# Live writers: every live focus owns a RUNNING row (no in-process side registry)
# --------------------------------------------------------------------------- #
def test_a_live_conversation_turn_keeps_the_checkout_until_it_ends(cycle):
    """An owner turn runs in a conversation-lane worker and holds a RUNNING row like any task: the shared
    checkout stays as it is while it lives, and the same cleanup applies and says so once it has ended."""
    from supervisor import evolution_lifecycle, queue

    dirty(cycle)
    queue.RUNNING["owner-turn"] = {"task": {"id": "owner-turn", "type": "task", "chat_id": CHAT},
                                   "lane": "conversation", "executor": "worker"}

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "skipped_other_tasks_running"
    assert (cycle.repo / "untracked.txt").is_file() and stash_oids(cycle) == [] and rows(cycle) == []

    queue.RUNNING.pop("owner-turn")
    tx = {"transaction_id": "later-tx", "base_head": cycle.base, "task_id": "evo-later"}
    evolution_lifecycle._cleanup_worktree_after_cycle(tx, "evo-later")
    assert tx["cleanup_status"] == "stashed_dirty" and tx["cleanup_stash_oids"] == stash_oids(cycle) != []


def test_the_settling_cycles_own_running_row_does_not_block_its_cleanup(cycle):
    """At its terminal the cycle's RUNNING row is still present (the dispatcher pops it later)."""
    from supervisor import queue

    dirty(cycle)
    queue.RUNNING[TASK] = {"task": {"id": TASK, "type": "evolution", "chat_id": CHAT}}

    recorded = cycle.end()["transaction"]

    assert recorded["cleanup_status"] == "stashed_dirty" and recorded["cleanup_stash_oids"] == stash_oids(cycle) != []


def test_the_cleanup_owner_never_imports_the_lifecycle_that_calls_it():
    """One direction: ``evolution_lifecycle`` calls this owner and supplies its data root."""
    import ast
    import pathlib

    import supervisor.evolution_cleanup as owner

    tree = ast.parse(pathlib.Path(owner.__file__).read_text(encoding="utf-8"))
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not {name for name in imported if name and "evolution_lifecycle" in name}


# --------------------------------------------------------------------------- #
# Campaign Stop and Panic
# --------------------------------------------------------------------------- #
def test_campaign_stop_discloses_on_the_stopped_cycles_card_and_history(cycle):
    from supervisor import evolution_lifecycle

    dirty(cycle)
    leftover = ahead(cycle)

    ended = evolution_lifecycle.complete_evolution_campaign("disabled via owner chat", status="stopped")
    cycle.deliver()

    kept = ended["transaction_history"][-1]
    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert ended["status"] == "stopped" and kept["cycle_outcome"] == "owner_stopped"
    assert kept["cleanup_status"] == "reset_to_base" and kept["cleanup_stash_oids"] == [oid]
    [row] = rows(cycle)
    assert row["task_id"] == TASK and oid in row["text"] and kept["cleanup_preserved_ref"] in row["text"]
    assert leftover in row["text"]
    assert_honest_words(row["text"])
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_results import load_task_result

    assert normalize_outcome_axes(load_task_result(cycle.data, TASK))["execution"]["evolution_cleanup"]["stash_oids"] == [oid]


def test_stop_keeps_the_stash_and_its_cause_when_the_cleanup_raises_after_it(cycle, monkeypatch):
    from supervisor import evolution_lifecycle, git_ops

    dirty(cycle)
    ahead(cycle)

    def boom(*_a, **_k):
        raise KeyError("after the stash")

    monkeypatch.setattr(git_ops, "preserve_local_ref_branch", boom)

    ended = evolution_lifecycle.complete_evolution_campaign("owner stop", status="stopped")
    cycle.deliver()

    kept = ended["transaction_history"][-1]
    assert kept["cleanup_status"] == "error" and "after the stash" in kept["cleanup_error"]
    assert kept["cleanup_stash_oids"] == stash_oids(cycle)
    [row] = rows(cycle)
    assert "after the stash" in row["text"]


def test_stop_row_survives_a_campaign_write_it_could_not_land(cycle, monkeypatch):
    """The Git move happened whether or not the campaign file took it: the card still says so."""
    from supervisor import evolution_lifecycle, state

    dirty(cycle)
    monkeypatch.setattr(state, "acquire_file_lock", lambda *a, **k: None)

    assert evolution_lifecycle.complete_evolution_campaign("owner stop", status="stopped") == {}
    monkeypatch.undo()

    # undo restored the seams; the owed row is in the registry and the replay delivers it.
    from supervisor import message_bus
    from supervisor import terminal_delivery as td

    monkeypatch.setattr(message_bus, "DATA_DIR", cycle.data)
    monkeypatch.setattr(message_bus, "publish_event", lambda *a, **k: None)
    bridge = message_bus.LocalChatBridge({})
    bridge._broadcast_fn = cycle.frames.append
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0)
    td.replay_pending_deliveries(cycle.data, event_queue=cycle.bus)
    cycle.deliver()
    [row] = rows(cycle)
    assert stash_oids(cycle)[0] in row["text"]


def test_panic_runs_no_git_and_owes_no_row(cycle, monkeypatch):
    from supervisor import evolution_lifecycle, git_ops

    dirty(cycle)
    monkeypatch.setattr(git_ops, "git_capture", lambda *a, **k: pytest.fail("Panic must not run Git"))

    def no_publication(*_a, **_k):
        pytest.fail("Panic must not touch the outbox")

    monkeypatch.setattr("supervisor.terminal_delivery.enqueue_terminal_delivery_outcome", no_publication)

    ended = evolution_lifecycle.complete_evolution_campaign("panic stop", status="stopped", cleanup_worktree=False)

    assert ended["status"] == "stopped" and "cleanup_status" not in ended["transaction_history"][-1]
    assert (cycle.repo / "untracked.txt").is_file() and cycle.bus.items == []


def test_a_failed_publication_cannot_undo_a_stop(cycle, monkeypatch):
    """Publication is an effect of the move, never a gate on the stop that carried it."""
    from ouroboros import task_store
    from supervisor import evolution_lifecycle

    dirty(cycle)

    def broken(*_a, **_k):
        raise OSError("task store down")

    monkeypatch.setattr(task_store, "patch_document", broken)
    monkeypatch.setattr("supervisor.terminal_delivery.register_pending_delivery", broken)

    ended = evolution_lifecycle.complete_evolution_campaign("owner stop", status="stopped")

    assert ended["status"] == "stopped" and ended["transaction_history"][-1]["cleanup_stash_oids"] == stash_oids(cycle)


# --------------------------------------------------------------------------- #
# Crash windows and replays: at-least-once, never a false "already clean"
# --------------------------------------------------------------------------- #
def test_crash_between_the_stash_and_its_evidence_recovers_by_the_exact_label(cycle, monkeypatch):
    """A replay finds a clean tree; the transaction's own stash label says the move happened."""
    from supervisor import evolution_lifecycle

    dirty(cycle)
    tx_id = cycle.tx["transaction_id"]
    real_cleanup = evolution_lifecycle._cleanup_worktree_after_cycle

    class Crash(BaseException):
        pass

    def crash_after_git(tx, task_id):
        git(cycle.repo, "stash", "push", "--include-untracked", "-m", f"evolution-cycle-cleanup-{tx_id}")
        raise Crash()

    monkeypatch.setattr(evolution_lifecycle, "_cleanup_worktree_after_cycle", crash_after_git)
    with pytest.raises(Crash):
        cycle.end()
    monkeypatch.setattr(evolution_lifecycle, "_cleanup_worktree_after_cycle", real_cleanup)
    assert cycle.campaign()["history"][-1]["transaction"]["cleanup_status"] == "pending"
    assert rows(cycle) == []

    replay = cycle.end()
    cycle.deliver()

    assert replay["replay"] is True
    assert replay["transaction"]["cleanup_status"] == "stashed_dirty"
    assert replay["transaction"]["cleanup_stash_oids"] == stash_oids(cycle) != []
    assert len(stash_oids(cycle)) == 1
    [row] = rows(cycle)
    assert stash_oids(cycle)[0] in row["text"]


@pytest.mark.parametrize("guard", ["other_running", "admission_closed", "candidate_unreadable"])
def test_a_replay_that_is_skipped_still_says_what_the_interrupted_attempt_moved(cycle, monkeypatch, guard):
    """Every guard that ends this attempt runs after the read of its own label and branch."""
    from ouroboros import body_adoption, body_candidate
    from supervisor import queue, update_merge, workers

    leftover = ahead(cycle)
    dirty(cycle)
    tx_id = cycle.tx["transaction_id"]
    git(cycle.repo, "stash", "push", "--include-untracked", "-m", f"evolution-cycle-cleanup-{tx_id}")
    kept = f"evolution-leftover-{tx_id}-20261101_000000-aaaaaa"
    git(cycle.repo, "branch", kept)
    if guard == "other_running":
        queue.RUNNING["other-task"] = {"task": {"id": "other-task"}}
    elif guard == "lock_busy":
        def busy():
            raise RuntimeError("busy")
        monkeypatch.setattr(update_merge, "acquire_update_lock", busy)
    elif guard == "admission_closed":
        monkeypatch.setattr(workers, "repo_writer_admission_closed", lambda: "managed_update:smart")
    elif guard == "candidate_found":
        monkeypatch.setattr(body_candidate, "find", lambda *a, **k: {"candidate_id": "cand-1"})
        monkeypatch.setattr(body_adoption, "abandon", lambda *a, **k: None)
    else:
        def unreadable(*_a, **_k):
            raise OSError("copy records unreadable")
        monkeypatch.setattr(body_candidate, "find", unreadable)

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == {"other_running": "skipped_other_tasks_running",
                                          "lock_busy": "skipped_managed_update_lock_busy",
                                          "admission_closed": "skipped_repo_writer_admission_closed",
                                          "candidate_found": "candidate_retained",
                                          "candidate_unreadable": "error"}[guard]
    assert recorded["cleanup_stash_oids"] == stash_oids(cycle) != []
    assert recorded["cleanup_preserved_ref"] == kept and recorded["cleanup_preserved_head"] == leftover
    [row] = rows(cycle)
    assert stash_oids(cycle)[0] in row["text"] and f"- Branch: `{kept}` at `{leftover}`" in row["text"]
    moved = "Evolution cleanup moved local changes to a Git stash and kept local commits on a branch"
    assert row["text"].splitlines()[0] == moved + (", then stopped on an error." if guard == "candidate_unreadable" else ".")
    assert "still at" not in row["text"]


def test_a_nested_dirt_replay_keeps_the_earlier_stash_status(cycle):
    """Only nested dirt is left beside an earlier attempt's stash: no new entry, and the status stays the move."""
    transaction = nested_only_dirt(cycle)
    dirty(cycle)
    git(cycle.repo, "stash", "push", "--include-untracked", "-m", f"evolution-cycle-cleanup-{cycle.tx['transaction_id']}")

    recorded = cycle.end(transaction=transaction)["transaction"]

    assert recorded["cleanup_status"] == "stashed_dirty" and recorded["cleanup_stash_oids"] == stash_oids(cycle)


@pytest.mark.parametrize("commits", [False, True], ids=["stash", "stash_and_branch"])
def test_lost_evidence_write_replays_without_a_second_stash_or_row(cycle, monkeypatch, commits):
    from supervisor import evolution_lifecycle

    leftover = ahead(cycle) if commits else ""
    dirty(cycle)
    real_persist = evolution_lifecycle._persist_evolution_cleanup
    monkeypatch.setattr(evolution_lifecycle, "_persist_evolution_cleanup", lambda *_a: False)
    first = cycle.end()["transaction"]
    cycle.deliver()
    monkeypatch.setattr(evolution_lifecycle, "_persist_evolution_cleanup", real_persist)
    assert cycle.campaign()["history"][-1]["transaction"]["cleanup_status"] == "pending"

    replay = cycle.end()
    cycle.deliver()

    assert replay["replay"] is True and replay["transaction"]["cleanup_stash_oids"] == first["cleanup_stash_oids"]
    status = "reset_to_base" if commits else "stashed_dirty"
    assert cycle.campaign()["history"][-1]["transaction"]["cleanup_status"] == status
    assert len(stash_oids(cycle)) == 1 and len(rows(cycle)) == 1
    if commits:  # the same one branch, found again from Git alone: the lost write left no record of it
        kept = first["cleanup_preserved_ref"]
        assert replay["transaction"]["cleanup_preserved_ref"] == kept
        assert replay["transaction"]["cleanup_preserved_head"] == leftover
        assert git(cycle.repo, "for-each-ref", "--format=%(refname:short)", "refs/heads/evolution-leftover-*") == kept
        assert f"- Branch: `{kept}` at `{leftover}`" in rows(cycle)[0]["text"]


def test_duplicate_terminal_after_delivery_adds_no_second_row(cycle):
    dirty(cycle)
    first = cycle.end()["transaction"]
    cycle.deliver()
    before = len(cycle.frames)

    replay = cycle.end()
    assert cycle.deliver() == []

    assert replay["replay"] is True and replay["transaction"]["cleanup_stash_oids"] == first["cleanup_stash_oids"]
    assert len(cycle.frames) == before and len(rows(cycle)) == 1


def test_a_lost_live_send_is_owed_and_replayed(cycle, monkeypatch):
    from supervisor import terminal_delivery as td

    dirty(cycle)
    cycle.bus.refuse = True
    cycle.end()
    assert rows(cycle) == [] and [row["system_type"] for row in td.pending_deliveries(cycle.data)] == ["evolution_cleanup"]

    cycle.bus.refuse = False
    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0)
    td.replay_pending_deliveries(cycle.data, event_queue=cycle.bus)
    cycle.deliver()

    assert td.pending_deliveries(cycle.data) == [] and len(rows(cycle)) == 1


# --------------------------------------------------------------------------- #
# The card's own line, through the real task-done consumer
# --------------------------------------------------------------------------- #
def _task_done(scene, monkeypatch, **task):
    """The real evolution terminal seam with the event the dispatcher built from the durable result."""
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_results import load_task_result
    from supervisor import events_evolution_done

    scene.finish()
    event = {"type": "task_done", "task_id": TASK, "task_type": "evolution", "status": "completed",
             "chat_id": CHAT, "outcome_axes": normalize_outcome_axes(load_task_result(scene.data, TASK))}
    ctx = SimpleNamespace(DRIVE_ROOT=scene.data, REPO_DIR=scene.repo, append_jsonl=lambda *a: None)
    metadata = {"evolution_transaction": scene.tx}
    events_evolution_done._handle_evolution_task_done(
        ctx, evt={"task_id": TASK, "metadata": metadata}, task_id=TASK, task={"id": TASK, "metadata": metadata},
        task_done_event=event, outcome_axes=event["outcome_axes"], cost=0.1, rounds=2)
    return event


def test_terminal_event_and_durable_row_state_one_short_fact_and_done_stays_done(cycle, monkeypatch):
    from ouroboros.project_dialogue import _completion_verdict, outcome_phase
    from ouroboros.task_results import load_task_result
    from ouroboros.terminal_projection import _project_row

    dirty(cycle)
    event = _task_done(cycle, monkeypatch)

    fact = event["outcome_axes"]["execution"]["evolution_cleanup"]
    assert fact["stash_oids"] == stash_oids(cycle)
    row = load_task_result(cycle.data, TASK)
    assert fact["line"] in _completion_verdict(row, event)
    assert outcome_phase(row, event) == "done"
    projected = _project_row(TASK, row, event, {"chat_id": CHAT})
    assert projected["outcome"] == "Done" and fact["line"] in projected["text"]
    assert stash_oids(cycle)[0] not in projected["text"], "exact ids stay in Details"


def test_stop_during_the_terminal_reaches_the_event_the_dispatcher_publishes(cycle, monkeypatch):
    """An owner stop that closes the campaign inside the terminal's own ``finally`` still informs that terminal."""
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_results import load_task_result
    from supervisor import events_evolution_done, evolution_lifecycle, state

    dirty(cycle)
    assert evolution_lifecycle.record_evolution_stop_intent("owner", "stop")
    state.update_state(lambda live: live.update(evolution_owner_stopped=True))
    # The terminal arrives after the campaign left active: the lifecycle refuses it and the backstop closes.
    campaign = cycle.campaign()
    campaign["status"] = "paused"
    assert evolution_lifecycle._write_evolution_campaign(campaign)
    stale = {**cycle.tx, "transaction_id": "other-transaction"}
    event = {"type": "task_done", "task_id": TASK, "task_type": "evolution", "status": "cancelled",
             "chat_id": CHAT, "outcome_axes": normalize_outcome_axes(load_task_result(cycle.data, TASK))}
    ctx = SimpleNamespace(DRIVE_ROOT=cycle.data, REPO_DIR=cycle.repo, append_jsonl=lambda *a: None)
    metadata = {"evolution_transaction": stale}
    events_evolution_done._handle_evolution_task_done(
        ctx, evt={"task_id": TASK, "metadata": metadata}, task_id=TASK, task={"id": TASK, "metadata": metadata},
        task_done_event=event, outcome_axes=event["outcome_axes"], cost=0.1, rounds=2)

    assert cycle.campaign()["status"] == "stopped"
    assert event["outcome_axes"]["execution"]["evolution_cleanup"]["stash_oids"] == stash_oids(cycle) != []
