"""#708: what real Git's partial failures, an interrupted branch and an odd path leave in the row.

``git stash push`` stores its entry before it clears the tree, so a failure can follow a save and a removed
file; the reset stays refused and the row says what is known, beside the failure. A branch stays paired with
its own commit, apart from where the checkout is. A printed command is run exactly as printed. Same real
lifecycle, outbox, send handler and history as ``tests/test_evolution_cleanup_disclosure.py``, whose fixture this
uses; a read-only directory is the real fault, and every checkout is a temporary one.
"""
from __future__ import annotations

import json
import os
import re

import pytest

from tests import test_evolution_cleanup_disclosure as disclosure
from tests.test_evolution_cleanup_disclosure import (
    TASK,
    ahead,
    assert_honest_words,
    assert_unconfirmed,
    code_spans,
    dirty,
    fault_git,
    git,
    printed_restore_command,
    rows,
    run_printed,
    stash_reads,
    undeletable,
)

cycle, read_only = disclosure.cycle, disclosure.read_only


def result_fact(scene):
    from ouroboros.outcomes import normalize_outcome_axes
    from ouroboros.task_results import load_task_result

    return normalize_outcome_axes(load_task_result(scene.data, TASK))["execution"]["evolution_cleanup"]


def end_cycle(scene, how):
    """The cycle's own terminal, or the campaign Stop that closes it."""
    from supervisor import evolution_lifecycle

    if how == "terminal":
        return scene.end()["transaction"]
    return evolution_lifecycle.complete_evolution_campaign("owner stop", status="stopped")["transaction_history"][-1]


def side_repository(path):
    """A real repository at ``path`` with one saved edit, and the record a cleanup there would leave."""
    path.mkdir()
    git(path, "init", "-q", "-b", "ouroboros")
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-qm", "base")
    (path / "tracked.txt").write_text("saved\n", encoding="utf-8")
    label = "evolution-cycle-cleanup-side"
    git(path, "stash", "push", "-m", label)
    oid = git(path, "rev-parse", "refs/stash")
    return oid, {"transaction_id": "side", "base_head": git(path, "rev-parse", "HEAD"), "cleanup_status": "stashed_dirty",
                 "cleanup_stash": label, "cleanup_stash_oids": [oid], "cleanup_repo": str(path)}


@pytest.mark.parametrize(("where", "how"), [("at_base", "terminal"), ("ahead", "stop")])
def test_a_stash_that_failed_after_saving_names_its_bytes_beside_the_failure(cycle, read_only, how, where):
    """Real Git stores the entry before it clears the tree. Its failure refuses the reset and hides nothing:
    the row names the entry, the cause and the checkout, and what it printed puts the removed file back."""
    leftover = ahead(cycle) if where == "ahead" else ""
    undeletable(cycle, read_only)

    recorded = end_cycle(cycle, how)
    cycle.deliver()

    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert recorded["cleanup_status"] == "skipped_stash_failed" and recorded["cleanup_stash_oids"] == [oid]
    assert recorded["cleanup_stash_partial"] is True and "locked/held.txt" in recorded["cleanup_error"]
    assert not (cycle.repo / "untracked.txt").exists(), "the failed push already removed this file"
    assert (cycle.repo / "tracked.txt").read_text(encoding="utf-8") == "somebody's edit\n"
    assert git(cycle.repo, "rev-parse", "HEAD") == (leftover or cycle.base), "the reset stays refused"

    [row] = rows(cycle)
    text = row["text"]
    assert text.splitlines()[0] == ("Evolution cleanup moved local changes to a Git stash, but the stash command "
                                    "failed, so the cleanup stopped there.")
    assert oid in text and "locked/held.txt" in text and "may already be gone from the checkout" in text
    assert "Branch:" not in text
    assert (f"The checkout is still at `{leftover}`" in text) is bool(leftover)
    assert_honest_words(text)
    fact = result_fact(cycle)
    assert fact["status"] == "skipped_stash_failed" and fact["stash_oids"] == [oid] and fact["stash_partial"] is True

    restore = run_printed(printed_restore_command(text.split("Restore the changes", 1)[1]))
    assert restore.returncode != 0, "with files still in place it reports errors, as the row says"
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"
    assert (cycle.repo / "tracked.txt").read_text(encoding="utf-8") == "somebody's edit\n"
    assert (cycle.repo / "locked" / "held.txt").read_text(encoding="utf-8") == "held\n"


def test_a_stash_that_failed_resetting_gives_back_every_byte_through_both_printed_commands(cycle, read_only):
    """The other partial shape: Git cleared the untracked file and reverted one tracked edit before it could
    not rewrite another. ``stash apply`` leaves tracked files alone there, so the row prints what restores them."""
    from supervisor import evolution_lifecycle

    held, other = cycle.repo / "ro" / "held.txt", cycle.repo / "other.txt"
    held.parent.mkdir()
    held.write_text("base\n", encoding="utf-8")
    other.write_text("base\n", encoding="utf-8")
    git(cycle.repo, "add", ".")
    git(cycle.repo, "commit", "-qm", "two tracked files")
    base = git(cycle.repo, "rev-parse", "HEAD")
    assert evolution_lifecycle.update_evolution_transaction(TASK, base_head=base)
    held.write_text("saved edit\n", encoding="utf-8")
    other.write_text("saved edit\n", encoding="utf-8")
    (cycle.repo / "untracked.txt").write_text("somebody's new file\n", encoding="utf-8")
    read_only(held.parent)

    recorded = cycle.end(transaction={**cycle.tx, "base_head": base})["transaction"]
    cycle.deliver()

    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert recorded["cleanup_status"] == "skipped_stash_failed" and recorded["cleanup_stash_oids"] == [oid]
    assert other.read_text(encoding="utf-8") == "base\n" and not (cycle.repo / "untracked.txt").exists()
    [row] = rows(cycle)
    restore, overwrite = [part for part in code_spans(row["text"]) if part.startswith("git -C ")]

    assert run_printed(restore).returncode != 0
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"
    assert other.read_text(encoding="utf-8") == "base\n", "it leaves tracked files alone, as the row says"

    assert " restore --source=" in overwrite and "overwriting what they hold now" in row["text"]
    run_printed(overwrite)
    assert other.read_text(encoding="utf-8") == "saved edit\n" and held.read_text(encoding="utf-8") == "saved edit\n"
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"


def test_a_refused_push_beside_an_earlier_stash_says_both_and_claims_no_removal(cycle, monkeypatch):
    """The earlier attempt's entry stands; this push saved nothing and removed nothing, and the row keeps them apart."""
    leftover = ahead(cycle)
    dirty(cycle)
    git(cycle.repo, "stash", "push", "--include-untracked", "-m", f"evolution-cycle-cleanup-{cycle.tx['transaction_id']}")
    earlier = git(cycle.repo, "rev-parse", "refs/stash")
    (cycle.repo / "tracked.txt").write_text("a later edit\n", encoding="utf-8")
    fault_git(monkeypatch, "stash", when=lambda cmd: cmd[2] == "push")

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "skipped_stash_failed" and recorded["cleanup_stash_oids"] == [earlier]
    assert not recorded.get("cleanup_stash_partial")
    assert (cycle.repo / "tracked.txt").read_text(encoding="utf-8") == "a later edit\n"
    [row] = rows(cycle)
    text = row["text"]
    assert "but the stash command failed, so the cleanup stopped there." in text.splitlines()[0]
    assert earlier in text and "stash refused" in text and f"The checkout is still at `{leftover}`" in text
    assert "may already be gone" not in text and " restore --source=" not in text


def test_a_failed_stash_whose_list_is_unread_is_unconfirmed_and_its_lookup_finds_the_entry(cycle, monkeypatch, read_only):
    undeletable(cycle, read_only)
    fault_git(monkeypatch, "for-each-ref", when=stash_reads(2))

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    oid = git(cycle.repo, "rev-parse", "refs/stash")
    assert recorded["cleanup_status"] == "skipped_stash_failed" and recorded["cleanup_stash_oids"] == []
    assert recorded["cleanup_stash_unconfirmed"] is True and recorded["cleanup_stash_partial"] is True
    [row] = rows(cycle)
    text = row["text"]
    assert_unconfirmed(text)
    assert "failed, and whether it saved an entry was not confirmed" in text and "may already be gone" in text
    assert "locked/held.txt" in text and oid not in text
    listing = run_printed(printed_restore_command(text), check=True).stdout
    assert recorded["cleanup_stash"] in listing, "the printed lookup finds what the failed push saved"


def test_a_failed_stash_with_an_unread_list_names_no_overwrite_from_an_older_entry(cycle, monkeypatch, read_only):
    """A newer entry may exist beyond the unread list: the earlier one is named, and nothing offers to overwrite from it."""
    dirty(cycle)
    label = f"evolution-cycle-cleanup-{cycle.tx['transaction_id']}"
    git(cycle.repo, "stash", "push", "--include-untracked", "-m", label)
    earlier = git(cycle.repo, "rev-parse", "refs/stash")
    undeletable(cycle, read_only)
    fault_git(monkeypatch, "for-each-ref", when=stash_reads(2))

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_stash_oids"] == [earlier] and recorded["cleanup_stash_unconfirmed"] is True
    [row] = rows(cycle)
    text = row["text"]
    assert earlier in text and "another entry under" in text and "may already be gone" in text
    assert " restore --source=" not in text
    listing = run_printed(printed_restore_command(text.split("Look for it", 1)[1]), check=True).stdout
    assert listing.count(label) == 2 and git(cycle.repo, "rev-parse", "refs/stash") != earlier


def test_a_branch_kept_earlier_stays_at_its_own_commit_when_a_newer_one_cannot_be_kept(cycle, monkeypatch):
    """An interrupted attempt kept commit A; a later writer committed B; this attempt cannot keep B. The row pairs
    the branch with A, as Git says, and names B separately as where the checkout still is."""
    from supervisor import git_ops

    class Crash(BaseException):
        pass

    kept_at = ahead(cycle)
    real_capture = git_ops.git_capture
    fault_git(monkeypatch, "reset", raises=Crash())
    with pytest.raises(Crash):
        cycle.end()
    monkeypatch.setattr(git_ops, "git_capture", real_capture)
    assert cycle.campaign()["history"][-1]["transaction"]["cleanup_status"] == "pending"
    (cycle.repo / "later.txt").write_text("later\n", encoding="utf-8")
    git(cycle.repo, "add", "later.txt")
    git(cycle.repo, "commit", "-qm", "a later writer")
    newer = git(cycle.repo, "rev-parse", "HEAD")
    monkeypatch.setattr(git_ops, "preserve_local_ref_branch", lambda *a, **k: (False, "branch refused"))

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "skipped_preserve_failed" and "branch refused" in recorded["cleanup_error"]
    assert recorded["cleanup_preserved_head"] == kept_at and recorded["cleanup_head"] == newer
    [row] = rows(cycle)
    text = row["text"]
    assert text.splitlines()[0] == ("Evolution cleanup kept local commits on a branch, but could not keep the newer "
                                    "local commits on a branch, so the checkout stays on them.")
    branch, commit = re.search(r"- Branch: `([^`]+)` at `([^`]+)`", text).groups()
    assert git(cycle.repo, "rev-parse", branch) == commit == kept_at, "the printed pair is what Git holds"
    assert f"The checkout is still at `{newer}`" in text and "branch refused" in text
    assert git(cycle.repo, "rev-parse", "HEAD") == newer


def test_a_replay_names_the_newest_branch_its_transaction_kept(cycle):
    """Two interrupted attempts kept two branches and the reset ran: the row names the commits kept last."""
    older = ahead(cycle)
    (cycle.repo / "later.txt").write_text("later\n", encoding="utf-8")
    git(cycle.repo, "add", "later.txt")
    git(cycle.repo, "commit", "-qm", "a later writer")
    newer = git(cycle.repo, "rev-parse", "HEAD")
    prefix = f"evolution-leftover-{cycle.tx['transaction_id']}"
    git(cycle.repo, "branch", f"{prefix}-20261101_000000-bbbbbb", older)
    git(cycle.repo, "branch", f"{prefix}-20261101_000100-aaaaaa", newer)
    git(cycle.repo, "reset", "-q", "--hard", cycle.base)

    recorded = cycle.end()["transaction"]
    cycle.deliver()

    assert recorded["cleanup_status"] == "reset_to_base"
    assert recorded["cleanup_preserved_ref"] == f"{prefix}-20261101_000100-aaaaaa"
    assert recorded["cleanup_preserved_head"] == newer
    [row] = rows(cycle)
    assert f"- Branch: `{prefix}-20261101_000100-aaaaaa` at `{newer}`" in row["text"]


def test_an_older_record_without_a_repository_prints_no_empty_path_and_still_restores(cycle):
    """A pre-upgrade record reaches the row through a duplicate terminal: no fabricated ``git -C ''``, and its
    branch keeps the commit that record paired with it."""
    from supervisor import evolution_cleanup

    dirty(cycle)
    label = "evolution-cycle-cleanup-old"
    git(cycle.repo, "stash", "push", "--include-untracked", "-m", label)
    oid = git(cycle.repo, "rev-parse", "refs/stash")
    tx = {"transaction_id": "old", "base_head": cycle.base, "cleanup_status": "reset_to_base", "cleanup_stash": label,
          "cleanup_stash_oids": [oid], "cleanup_preserved_ref": "old-keep", "cleanup_head": "c" * 40}

    evolution_cleanup.publish_cleanup(tx, TASK, cycle.data)
    cycle.deliver()

    [row] = rows(cycle)
    text = row["text"]
    assert "git -C" not in text and "- Repository: not recorded" in text and "POSIX" not in text
    assert f"- Branch: `old-keep` at `{'c' * 40}`" in text
    assert "from inside that repository" in text
    assert next(part for part in code_spans(text) if " stash apply " in part) == f"git stash apply {oid}"
    run_printed(f"git stash apply {oid}", cwd=cycle.repo, check=True)
    assert (cycle.repo / "untracked.txt").read_text(encoding="utf-8") == "somebody's new file\n"


@pytest.mark.parametrize("name", ["back`tick", "line\nbreak"], ids=["backtick", "line_break"])
def test_a_path_no_code_span_can_carry_is_shown_escaped_with_no_ready_command(cycle, name):
    """The card, a bubble and Telegram read a backtick inside a span differently, and a line break in one becomes a
    space. Such a path is shown escaped, nothing copyable pretends to hold it, and the command runs from inside it."""
    from supervisor import evolution_cleanup

    if os.name == "nt" and "\n" in name:
        pytest.skip("Windows has no such path")
    repo = cycle.repo.parent / name
    oid, tx = side_repository(repo)

    evolution_cleanup.publish_cleanup(tx, TASK, cycle.data)
    cycle.deliver()

    [row] = rows(cycle)
    text = row["text"]
    shown = re.search(r"- Repository, escaped because it holds a backtick or line break: `([^`\n]+)`", text).group(1)
    assert json.loads(shown) == str(repo)
    assert "git -C" not in text and name not in text
    assert "from inside that repository" in text
    assert next(part for part in code_spans(text) if " stash apply " in part) == f"git stash apply {oid}"
    run_printed(f"git stash apply {oid}", cwd=repo, check=True)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "saved\n"


def test_a_path_with_quotes_and_runs_of_spaces_is_exact_in_the_printed_command(cycle):
    from supervisor import evolution_cleanup

    if os.name == "nt":
        pytest.skip("Windows has no such path")
    repo = cycle.repo.parent / "it's   $HOME \"odd\""
    _, tx = side_repository(repo)

    evolution_cleanup.publish_cleanup(tx, TASK, cycle.data)
    cycle.deliver()

    [row] = rows(cycle)
    assert f"- Repository: `{repo}`" in row["text"]
    run_printed(printed_restore_command(row["text"]), check=True)
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "saved\n"
