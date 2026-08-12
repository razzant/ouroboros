"""BRANCH OFF / MERGE BACK against REAL git (A7-A10).

Every test here runs `git init`, `git worktree add` and — where it matters — a
real conflicting merge. Mocking git would pin our beliefs about git rather than
git's behaviour, and the two operations this module owns are exactly the place
where that difference destroys an owner's work.

What is pinned:

* a thread's LOCATION is derived from the worktree's existence, never stored;
* the owner CHOOSES the base, including "exactly as it is now" (a snapshot
  commit that leaves credential-shaped files out);
* a project with no folder and a project with no git are refused with the TYPED
  decisions T2 established, not with an error the owner has to decode;
* merge-back preconditions are the project-WIDE activity query and a clean local
  tree;
* a conflict is SHOWN, the merge is ABORTED, and the thread keeps its branch;
* a successful merge does NOT remove the checkout (A10).
"""

from __future__ import annotations

import subprocess

import pytest

from ouroboros.project_threads_registry import create_thread
from ouroboros.projects_registry import create_project
from ouroboros.thread_branching import (
    BASE_SNAPSHOT,
    REASON_ALREADY_BRANCHED,
    REASON_CHECKOUT_DIRTY,
    REASON_CHECKOUT_HEAD_OFF_BRANCH,
    REASON_GIT_INIT_REQUIRED,
    REASON_LOCAL_TREE_DIRTY,
    REASON_MERGE_ABORT_FAILED,
    REASON_MERGE_CONFLICT,
    REASON_NOT_BRANCHED,
    REASON_NO_FOLDER,
    REASON_PROJECT_BUSY,
    REASON_SNAPSHOT_FAILED,
    REASON_UNKNOWN_BASE,
    branch_off_bases,
    branch_off_thread,
    merge_back_thread,
    thread_location,
)


def _git(cwd, *args, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check)


@pytest.fixture()
def folder(tmp_path):
    """A real git repository standing in for the owner's project folder."""
    root = tmp_path / "owner_folder"
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "T")
    (root / "app.txt").write_text("one\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "seed")
    return root


@pytest.fixture()
def drive(tmp_path):
    return tmp_path / "drive"


@pytest.fixture()
def wt_root(tmp_path):
    return tmp_path / "thread_worktrees"


def _project(drive, folder, pid="racer"):
    create_project(drive, pid, name="Racer", working_dir=str(folder))
    return create_thread(drive, pid, name="Side quest")


def _branch(drive, pid, tid, wt_root, base_ref=""):
    return branch_off_thread(
        drive, pid, tid, base_ref=base_ref, data_dir=drive, worktree_root=wt_root,
    )


# --------------------------------------------------------------------------- #
# Refusals BEFORE any git work
# --------------------------------------------------------------------------- #

def test_a_folderless_project_cannot_branch(drive):
    """A11: branching off needs a place to branch FROM."""
    create_project(drive, "placeless", name="Placeless")
    thread = create_thread(drive, "placeless", name="Side quest")

    out = branch_off_thread(drive, "placeless", thread["id"], data_dir=drive)

    assert out["ok"] is False
    assert out["reason"] == REASON_NO_FOLDER
    assert "no working folder" in out["message"]


def test_a_non_git_folder_gets_T2s_typed_offer_not_an_error(drive, tmp_path):
    """A12/X5: git is OFFERED. The refusal carries the SAME typed decision the
    task-admission path returns, so the owner is asked one question, not two."""
    plain = tmp_path / "plain_folder"
    plain.mkdir()
    (plain / "notes.txt").write_text("hi\n", encoding="utf-8")
    create_project(drive, "plain", name="Plain", working_dir=str(plain))
    thread = create_thread(drive, "plain", name="Side quest")

    out = branch_off_thread(drive, "plain", thread["id"], data_dir=drive)

    assert out["ok"] is False
    assert out["reason"] == REASON_GIT_INIT_REQUIRED
    decision = out["decision"]
    assert decision["decision"] == "git_init_required"
    assert decision["offer"] == "init_git"
    assert decision["project_id"] == "plain"
    assert "branching" in decision["enables"]


def test_an_unknown_base_is_refused_before_anything_is_provisioned(drive, folder, wt_root):
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref="no-such-ref")

    assert out["ok"] is False
    assert out["reason"] == REASON_UNKNOWN_BASE
    assert thread_location(drive, "racer", thread["id"])["where"] == "project_folder"


# --------------------------------------------------------------------------- #
# BRANCH OFF
# --------------------------------------------------------------------------- #

def test_bases_offer_branches_tags_and_as_it_is_now(drive, folder):
    """A8: the list is an OFFER. "As it is now" is one entry in it, always
    present, and it discloses whether choosing it would create a commit."""
    _git(folder, "branch", "experiment")
    _git(folder, "tag", "v1")

    listed = branch_off_bases(folder)

    assert listed["current_branch"] == "main"
    refs = [row["ref"] for row in listed["bases"]]
    assert refs[0] == "main" and "(current)" in listed["bases"][0]["label"]
    assert {"experiment", "v1"} <= set(refs)
    assert {row["kind"] for row in listed["bases"]} == {"branch", "tag"}
    assert listed["snapshot"]["ref"] == BASE_SNAPSHOT
    assert listed["snapshot"]["dirty"] is False
    assert listed["snapshot"]["creates_commit"] is False

    (folder / "app.txt").write_text("edited\n", encoding="utf-8")
    dirty = branch_off_bases(folder)
    assert dirty["snapshot"]["dirty"] is True
    assert dirty["snapshot"]["creates_commit"] is True


def test_branch_off_provisions_a_real_worktree_and_derives_the_location(drive, folder, wt_root):
    from pathlib import Path

    thread = _project(drive, folder)
    assert thread_location(drive, "racer", thread["id"])["where"] == "project_folder"

    out = _branch(drive, "racer", thread["id"], wt_root)

    assert out["ok"] is True, out
    checkout = Path(out["path"])
    assert checkout.is_dir()
    assert (checkout / "app.txt").read_text(encoding="utf-8") == "one\n"
    listed = _git(folder, "worktree", "list").stdout
    assert str(checkout) in listed
    # A7: the location is derived from the worktree existing, not from a flag.
    where = thread_location(drive, "racer", thread["id"])
    assert where["where"] == "worktree"
    assert where["path"] == str(checkout)
    assert where["branch"] == out["branch"]


def test_branch_off_from_a_chosen_branch_uses_that_branchs_content(drive, folder, wt_root):
    from pathlib import Path

    _git(folder, "checkout", "-q", "-b", "experiment")
    (folder / "app.txt").write_text("experimental\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "experiment work")
    _git(folder, "checkout", "-q", "main")
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref="experiment")

    assert out["ok"] is True, out
    assert (Path(out["path"]) / "app.txt").read_text(encoding="utf-8") == "experimental\n"


def test_as_it_is_now_snapshots_uncommitted_work_and_leaves_secrets_out(drive, folder, wt_root):
    """A8's only special case. The snapshot is disclosed: its sha comes back, and
    so do the credential-shaped files deliberately kept out of git history."""
    from pathlib import Path

    (folder / "app.txt").write_text("unsaved edit\n", encoding="utf-8")
    (folder / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is True, out
    snapshot = out["snapshot_commit"]
    assert snapshot["created"] is True
    assert snapshot["sha"]
    assert ".env" in snapshot["skipped_sensitive"]
    checkout = Path(out["path"])
    assert (checkout / "app.txt").read_text(encoding="utf-8") == "unsaved edit\n"
    assert not (checkout / ".env").exists(), "a snapshot must never bake a secret into history"
    tracked = _git(folder, "ls-files").stdout.split()
    assert ".env" not in tracked


def test_a_snapshot_never_deletes_a_TRACKED_credential_shaped_file(drive, folder, wt_root):
    """T3R-1, the regression that matters most in this module.

    ``_unstage_sensitive_paths`` was written for ``attach_snapshot_init``, where
    the repository was created one line earlier and EVERY staged path is untracked
    by construction. On the owner's own pre-existing repository the same call is a
    different operation: ``git diff --cached --name-only`` lists TRACKED
    modifications too, and ``git rm --cached`` on one of those stages a DELETION.
    A fixture the project had tracked for months was committed away on the owner's
    branch, vanished from ``git ls-files``, and the receipt told them it was
    "still in your folder, still untracked" — true only because the snapshot had
    just untracked it.

    Both shapes in one repository, because telling them apart is the fix:
    (a) a TRACKED credential-shaped file the owner modified — snapshotted like any
        other tracked file and DISCLOSED, never deleted;
    (b) an UNTRACKED one — still kept out of history, as it always was.
    """
    fixtures = folder / "tests" / "fixtures"
    fixtures.mkdir(parents=True)
    (fixtures / "token.json").write_text('{"token": "fixture"}\n', encoding="utf-8")
    _git(folder, "add", "-A")
    _git(folder, "commit", "-qm", "the fixture has been tracked for months")
    exclude = folder / ".git" / "info" / "exclude"
    exclude_before = exclude.read_text(encoding="utf-8") if exclude.exists() else ""

    # (a) tracked and modified, (b) untracked and new.
    (fixtures / "token.json").write_text('{"token": "fixture-v2"}\n', encoding="utf-8")
    (folder / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is True, out
    snapshot = out["snapshot_commit"]
    # (b) the untracked secret stays out of history, exactly as before.
    assert snapshot["skipped_sensitive"] == [".env"]
    # (a) the tracked one is disclosed rather than deleted.
    assert snapshot["tracked_sensitive"] == ["tests/fixtures/token.json"]
    tracked = _git(folder, "ls-files").stdout.split()
    assert "tests/fixtures/token.json" in tracked, "a tracked file must survive a snapshot"
    assert ".env" not in tracked
    assert (fixtures / "token.json").is_file()
    # The snapshot commit MODIFIES it; it must never delete it.
    changed = _git(folder, "show", "--name-status", "--format=", "HEAD").stdout
    assert "M\ttests/fixtures/token.json" in changed
    assert "D\t" not in changed
    # The owner's own exclude file is theirs; a pre-existing repo is not rewritten.
    assert (exclude.read_text(encoding="utf-8") if exclude.exists() else "") == exclude_before


def test_a_snapshot_that_cannot_tell_tracked_from_untracked_refuses(drive, folder, wt_root, monkeypatch):
    """"Which of these does HEAD already have" has no safe default: guessing ABSENT
    deletes the owner's tracked file, guessing PRESENT commits a new secret. So the
    snapshot refuses — and hands the INDEX back, because `git add -A` has already
    staged the whole folder by then (T3R-10)."""
    import ouroboros.project_sources as sources

    (folder / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (folder / "app.txt").write_text("unsaved\n", encoding="utf-8")
    thread = _project(drive, folder)
    head_before = _git(folder, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setattr(
        sources, "_staged_sensitive_partition", lambda _p: ([], [], "git ls-tree exploded"),
    )

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is False
    assert out["reason"] == REASON_SNAPSHOT_FAILED
    assert "already tracked" in out["message"]
    assert _git(folder, "rev-parse", "HEAD").stdout.strip() == head_before
    # T3R-10: the folder is handed back as it was — nothing staged, nothing lost.
    staged = _git(folder, "diff", "--cached", "--name-only").stdout.strip()
    assert staged == "", f"the owner's index was left staged: {staged!r}"
    assert (folder / "app.txt").read_text(encoding="utf-8") == "unsaved\n"
    assert (folder / ".env").read_text(encoding="utf-8") == "API_KEY=secret\n"


def test_a_folder_whose_only_change_is_an_untracked_secret_makes_no_commit(drive, folder, wt_root):
    """With the secret left out there is nothing left to commit, and that is read
    from the INDEX rather than from git's English "nothing to commit" (T3R-11)."""
    (folder / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    thread = _project(drive, folder)
    before = _git(folder, "rev-parse", "HEAD").stdout.strip()

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is True, out
    assert out["snapshot_commit"]["created"] is False
    assert out["snapshot_commit"]["sha"] == before
    assert out["snapshot_commit"]["skipped_sensitive"] == [".env"]
    assert _git(folder, "rev-parse", "HEAD").stdout.strip() == before
    assert _git(folder, "diff", "--cached", "--name-only").stdout.strip() == ""


def test_as_it_is_now_on_a_clean_tree_makes_no_commit(drive, folder, wt_root):
    """"Exactly as it is now" of a clean folder is already a commit — HEAD."""
    before = _git(folder, "rev-parse", "HEAD").stdout.strip()
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is True, out
    assert out["snapshot_commit"]["created"] is False
    assert out["snapshot_commit"]["sha"] == before
    assert _git(folder, "rev-parse", "HEAD").stdout.strip() == before


def test_a_renamed_tracked_secret_is_disclosed_and_not_reported_as_untouched(drive, folder, wt_root):
    """T3R2-H1: rename detection made the partition lie about the owner's branch.

    Git detects a staged rename by default and prints it as its DESTINATION alone.
    A tracked `secrets.env` renamed to `secrets2.env` therefore arrived as ONE path
    HEAD has never heard of, was classified "absent from HEAD", and was
    `git rm --cached`-ed — which unstages only the addition. The SOURCE's staged
    deletion, invisible to the partition, was committed: the owner's tracked file
    left their branch while `tracked_sensitive: []` asserted nothing tracked was
    involved. `--no-renames` makes both halves visible as what the index holds.
    """
    (folder / "secrets.env").write_text("API_KEY=old\n", encoding="utf-8")
    _git(folder, "add", "-A")
    _git(folder, "commit", "-qm", "the owner tracked this months ago")
    _git(folder, "mv", "secrets.env", "secrets2.env")
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is True, out
    snapshot = out["snapshot_commit"]
    # The tracked side of the rename is NAMED. Before the fix this was [] while the
    # snapshot commit deleted `secrets.env` from the owner's branch.
    assert snapshot["tracked_sensitive"] == ["secrets.env"]
    assert snapshot["skipped_sensitive"] == ["secrets2.env"]


def test_a_folder_stopped_mid_merge_refuses_to_snapshot(drive, folder, wt_root):
    """T3R2-H2: "exactly as it is now" cannot be committed while git holds the
    folder open.

    A folder stopped part-way through a merge holds conflict markers in tracked
    files, and MERGE_HEAD is the only record of what was being merged. `git add -A`
    + `git commit` there bakes the markers in AND clears MERGE_HEAD, so
    `git merge --abort` stops working and the owner's half-done resolution is gone.
    """
    _git(folder, "checkout", "-q", "-b", "side")
    (folder / "app.txt").write_text("side\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "side edit")
    _git(folder, "checkout", "-q", "main")
    (folder / "app.txt").write_text("main\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "main edit")
    _git(folder, "merge", "side", check=False)
    assert _git(folder, "rev-parse", "--verify", "-q", "MERGE_HEAD", check=False).returncode == 0
    head_before = _git(folder, "rev-parse", "HEAD").stdout.strip()
    thread = _project(drive, folder)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is False
    assert out["reason"] == REASON_SNAPSHOT_FAILED
    assert "stopped part-way through a merge" in out["message"]
    # The owner's conflict is EXACTLY as they left it: same HEAD, merge still open,
    # markers still in the file rather than committed as if they were their code.
    assert _git(folder, "rev-parse", "HEAD").stdout.strip() == head_before
    assert _git(folder, "rev-parse", "--verify", "-q", "MERGE_HEAD", check=False).returncode == 0
    assert "<<<<<<<" in (folder / "app.txt").read_text(encoding="utf-8")


def test_a_snapshot_that_RAISES_still_hands_the_index_back(drive, folder, wt_root, monkeypatch):
    """T3R2-M4: the partition's subprocess calls are unguarded, so a TimeoutExpired
    or an OSError travelled out of `_snapshot_commit` — past a `branch_off_thread`
    whose try covers only provisioning — with `git add -A` already done. The
    returned-error path restored the index; only the raised path leaked it."""
    import subprocess as _subprocess

    import ouroboros.project_sources as sources

    (folder / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    (folder / "app.txt").write_text("unsaved\n", encoding="utf-8")
    thread = _project(drive, folder)

    def _explode(_path):
        raise _subprocess.TimeoutExpired(cmd=["git", "diff"], timeout=60)

    monkeypatch.setattr(sources, "_staged_sensitive_partition", _explode)

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is False
    assert out["reason"] == REASON_SNAPSHOT_FAILED
    assert "TimeoutExpired" in out["message"]
    staged = _git(folder, "diff", "--cached", "--name-only").stdout.strip()
    assert staged == "", f"the owner's index was left staged: {staged!r}"
    assert (folder / "app.txt").read_text(encoding="utf-8") == "unsaved\n"


def test_a_branch_off_that_fails_AFTER_snapshotting_names_the_commit_it_made(drive, folder, wt_root):
    """T3R2-H4: `_snapshot_aborted` covers failures INSIDE the snapshot, but the
    commit precedes provisioning and a provisioning refusal rolled nothing back and
    said nothing. The owner read "branching failed", believed nothing had happened,
    and their uncommitted work had silently become a commit they did not author."""
    thread = _project(drive, folder)
    (folder / "app.txt").write_text("unsaved edit\n", encoding="utf-8")
    # A REAL provisioning refusal on the real path: the thread's branch exists.
    _git(folder, "branch", f"thread/racer__{thread['id']}")

    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)

    assert out["ok"] is False
    assert out["reason"] == "branch_failed"
    head = _git(folder, "rev-parse", "HEAD").stdout.strip()
    assert out["snapshot_commit"]["created"] is True
    assert out["snapshot_commit"]["sha"] == head
    assert head[:12] in out["message"]
    assert "NOT as you left it" in out["message"]


def test_branching_twice_is_refused_rather_than_resetting_the_first(drive, folder, wt_root):
    """The durable registry never clobbers an owner's checkout (X3)."""
    thread = _project(drive, folder)
    first = _branch(drive, "racer", thread["id"], wt_root)
    assert first["ok"] is True

    second = _branch(drive, "racer", thread["id"], wt_root)

    assert second["ok"] is False
    assert second["reason"] == REASON_ALREADY_BRANCHED
    assert second["location"]["path"] == first["path"]


# --------------------------------------------------------------------------- #
# MERGE BACK
# --------------------------------------------------------------------------- #

def _commit_in(checkout, name, body):
    from pathlib import Path

    Path(checkout, name).write_text(body, encoding="utf-8")
    _git(checkout, "config", "user.email", "t@example.com")
    _git(checkout, "config", "user.name", "T")
    _git(checkout, "add", "-A")
    _git(checkout, "commit", "-qm", f"thread work on {name}")


def test_merge_back_brings_the_threads_commits_home_and_keeps_the_checkout(drive, folder, wt_root):
    """A9 happy path + A10: merging never removes the worktree."""
    from pathlib import Path

    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "from the thread\n")

    merged = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert merged["ok"] is True, merged
    assert merged["merged"] is True
    assert (folder / "feature.txt").read_text(encoding="utf-8") == "from the thread\n"
    assert merged["worktree_kept"] is True
    assert Path(out["path"]).is_dir()
    assert thread_location(drive, "racer", thread["id"])["where"] == "worktree"


def test_merge_back_refuses_while_the_project_is_busy(drive, folder, wt_root):
    """A9's first precondition, and A14's honesty: the copy explains WAITING."""
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "from the thread\n")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=True)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_PROJECT_BUSY
    assert "until that task finishes" in refused["message"]
    assert not (folder / "feature.txt").exists()


def test_merge_back_refuses_a_dirty_local_tree_and_names_the_files(drive, folder, wt_root):
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "from the thread\n")
    (folder / "app.txt").write_text("owner is mid-edit\n", encoding="utf-8")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_LOCAL_TREE_DIRTY
    assert any("app.txt" in row for row in refused["dirty_files"])
    assert (folder / "app.txt").read_text(encoding="utf-8") == "owner is mid-edit\n"


def test_a_real_conflict_is_shown_stops_the_merge_and_leaves_both_sides_intact(drive, folder, wt_root):
    """A9's hard rule, against a REAL conflicting merge.

    The owner's folder must come out byte-for-byte as it went in — no conflict
    markers, no half-merge, no MERGE_HEAD — and the thread must keep its branch
    and every commit in it.
    """
    from pathlib import Path

    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "app.txt", "the thread's version\n")
    (folder / "app.txt").write_text("the owner's version\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "owner edit")
    owner_head = _git(folder, "rev-parse", "HEAD").stdout.strip()
    thread_head = _git(out["path"], "rev-parse", "HEAD").stdout.strip()

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_MERGE_CONFLICT
    assert refused["conflicts"] == ["app.txt"]
    # The folder is exactly as it was: same HEAD, clean tree, no merge in flight.
    assert _git(folder, "rev-parse", "HEAD").stdout.strip() == owner_head
    assert _git(folder, "status", "--porcelain").stdout.strip() == ""
    assert (folder / "app.txt").read_text(encoding="utf-8") == "the owner's version\n"
    assert not (Path(folder) / ".git" / "MERGE_HEAD").exists()
    # The thread stays in its branch with its work intact.
    assert _git(out["path"], "rev-parse", "HEAD").stdout.strip() == thread_head
    assert thread_location(drive, "racer", thread["id"])["where"] == "worktree"


def test_a_failed_abort_is_reported_as_a_mid_merge_folder_not_as_success(drive, folder, wt_root, monkeypatch):
    """T3R-2. "The merge was stopped and the folder left as it was" is a CLAIM
    about a git command, and that command can fail.

    The abort's result was never read, so the sentence was asserted rather than
    verified: with a failing abort the owner's folder sat with MERGE_HEAD, `UU`
    entries and conflict markers in the file while the answer said it was
    untouched. The mid-merge state must be NAMED, with what the owner has to do.
    """
    import ouroboros.thread_branching as branching

    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "app.txt", "the thread's version\n")
    (folder / "app.txt").write_text("the owner's version\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "owner edit")

    real_git = branching._git

    def _abort_fails(root, *args):
        if args[:2] == ("merge", "--abort"):
            return subprocess.CompletedProcess(
                ["git", *args], 128, "", "fatal: could not abort",
            )
        return real_git(root, *args)

    monkeypatch.setattr(branching, "_git", _abort_fails)
    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)
    monkeypatch.undo()

    assert refused["ok"] is False
    assert refused["reason"] == REASON_MERGE_ABORT_FAILED
    assert refused["folder_left_mid_merge"] is True
    assert "stopped part-way" in refused["message"]
    assert "git merge --abort" in refused["message"]
    assert refused["conflicts"] == ["app.txt"]
    assert refused["working_dir"] == str(folder)
    # The answer is true: the folder really IS mid-merge, and now says so.
    assert (folder / ".git" / "MERGE_HEAD").exists()
    assert _git(folder, "status", "--porcelain").stdout.strip().startswith("UU")
    assert "<<<<<<<" in (folder / "app.txt").read_text(encoding="utf-8")
    _git(folder, "merge", "--abort")


def test_a_merge_git_refused_outright_is_not_reported_as_a_stuck_folder(drive, folder, wt_root):
    """The abort is only checked when a merge actually STARTED. A merge git
    refused before beginning leaves nothing in progress, so `merge --abort`
    failing there means the folder is FINE — reporting it as stopped part-way
    would send the owner to fix a folder that needs nothing."""
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "from the thread\n")
    # An unrelated history: git refuses the merge outright, without starting one.
    _git(folder, "checkout", "-q", "--orphan", "unrelated")
    _git(folder, "commit", "-qm", "a history with no common ancestor")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] != REASON_MERGE_ABORT_FAILED
    assert not (folder / ".git" / "MERGE_HEAD").exists()


def test_merge_back_refuses_while_the_checkout_still_holds_uncommitted_work(drive, folder, wt_root):
    """T3R-3(a). A merge moves COMMITS. Edits that were never committed in the
    checkout do not travel with it, and answering `ok: true, merged: true` tells
    the owner everything came home while that work sits in a folder they have
    stopped looking at."""
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "committed\n")
    from pathlib import Path

    Path(out["path"], "feature.txt").write_text("committed + more\n", encoding="utf-8")
    Path(out["path"], "brand_new.txt").write_text("never committed\n", encoding="utf-8")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_CHECKOUT_DIRTY
    assert "never committed" in refused["message"]
    assert any("brand_new.txt" in row for row in refused["dirty_files"])
    # Nothing was merged: a refusal must not half-do the operation it refused.
    assert not (folder / "feature.txt").exists()
    assert refused["inspection"]["dirty"] is True


def test_merge_back_refuses_when_the_checkouts_HEAD_is_off_the_threads_branch(drive, folder, wt_root):
    """T3R-3(b), the quieter of the two and the worse one.

    Every commit went to a branch the thread is not bound to, so the bound branch
    never moved, the merge was a no-op, and `ok: true, merged: false` renders as
    "nothing new to merge — the folder already has this work". The folder has none
    of it."""
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _git(out["path"], "checkout", "-q", "-b", "my-side-work")
    _commit_in(out["path"], "feature.txt", "all of the thread's real work\n")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_CHECKOUT_HEAD_OFF_BRANCH
    assert refused["checkout_branch"] == "my-side-work"
    assert refused["branch"] in refused["message"]
    assert "my-side-work" in refused["message"]
    assert not (folder / "feature.txt").exists()


def test_a_DETACHED_checkout_is_not_described_as_a_branch_named_HEAD(drive, folder, wt_root):
    """`git rev-parse --abbrev-ref` answers the literal string "HEAD" for a
    detached head. That is not a branch name and must not be quoted back as one
    in copy the owner is meant to act on."""
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "work made while detached\n")
    _git(out["path"], "checkout", "-q", "--detach")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_CHECKOUT_HEAD_OFF_BRANCH
    assert "detached HEAD" in refused["message"]
    assert "'HEAD'" not in refused["message"], "a detached head is not a branch called HEAD"
    assert refused["branch"] in refused["message"]


def test_a_merge_onto_a_DETACHED_project_folder_is_refused_not_reported_as_done(
    drive, folder, wt_root,
):
    """T3R2-B1: the blocker. A merge needs a branch to LAND on.

    `git merge --no-ff` onto a detached HEAD succeeds and leaves the merge commit
    on no branch at all. Both of this phase's safety judges are then fooled by the
    same wrong reference: `inspect_thread_worktree` counts unmerged commits against
    the project's HEAD — now that dangling merge — and answers zero, and
    `git branch -d` agrees the thread branch is merged, because against that HEAD
    it is. A one-click removal then deletes the checkout AND the branch, and the
    owner's work survives only in the reflog.
    """
    from ouroboros.thread_branching import REASON_PROJECT_HEAD_DETACHED
    from ouroboros.thread_worktrees import get_thread_worktree, inspect_thread_worktree

    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "from the thread\n")
    _git(folder, "checkout", "-q", "--detach")
    head_before = _git(folder, "rev-parse", "HEAD").stdout.strip()

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_PROJECT_HEAD_DETACHED
    assert "not on any branch" in refused["message"]
    assert refused["branch"] in refused["message"]
    # Nothing was merged, so the two safety judges still see the work.
    assert _git(folder, "rev-parse", "HEAD").stdout.strip() == head_before
    assert not (folder / "feature.txt").exists()
    row = get_thread_worktree(drive, "racer", thread["id"])
    assert inspect_thread_worktree(row)["unmerged_commits"] >= 1
    # Deliberately NOT acknowledgeable: this is not work left behind, it is a
    # merge with no destination.
    assert refused.get("acknowledgeable") is None
    still_refused = merge_back_thread(
        drive, "racer", thread["id"], data_dir=drive, busy=False,
        acknowledge_checkout_dirty=True,
    )
    assert still_refused["reason"] == REASON_PROJECT_HEAD_DETACHED
    # And the way OUT is the one the copy names.
    _git(folder, "checkout", "-q", "main")
    merged = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)
    assert merged["ok"] is True, merged
    assert (folder / "feature.txt").read_text(encoding="utf-8") == "from the thread\n"


def test_the_bases_list_reports_no_current_branch_on_a_detached_folder(drive, folder):
    """T3R2-B1, second site. The bases LOOP already guarded `current != "HEAD"`;
    the returned field did not, so a client read `current_branch: "HEAD"` and had
    every reason to display it and to branch off from it as a branch name."""
    _git(folder, "checkout", "-q", "--detach")

    listed = branch_off_bases(folder)

    assert listed["current_branch"] == ""
    assert "HEAD" not in [row["ref"] for row in listed["bases"]]
    # The real branch is still offered, just not as "(current)".
    assert "main" in [row["ref"] for row in listed["bases"]]


def test_an_untracked_file_in_the_owners_folder_does_not_block_a_merge(drive, folder, wt_root):
    """"Clean local tree" is about TRACKED changes.

    An untracked file is not part of a merge and cannot blur which work came from
    where, which is what this precondition protects. Counting it meant a project
    holding one stray `.env` or build artifact could never merge anything back —
    forever — with copy telling the owner to commit or stash a file they
    deliberately keep out of git. It surfaced as collateral of T3R-1: the snapshot
    used to hide such a file by writing the owner's `.git/info/exclude`, which is
    exactly the write T3R-1 stopped.
    """
    (folder / ".env").write_text("API_KEY=secret\n", encoding="utf-8")
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root, base_ref=BASE_SNAPSHOT)
    assert out["ok"] is True, out
    assert out["snapshot_commit"]["skipped_sensitive"] == [".env"]
    _commit_in(out["path"], "feature.txt", "from the thread\n")

    merged = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert merged["ok"] is True, merged
    assert merged["merged"] is True
    assert (folder / "feature.txt").read_text(encoding="utf-8") == "from the thread\n"
    # The owner's file is untouched and still theirs, still untracked.
    assert (folder / ".env").read_text(encoding="utf-8") == "API_KEY=secret\n"
    assert ".env" not in _git(folder, "ls-files").stdout.split()
    # A TRACKED edit still refuses, because that one really would blur it.
    (folder / "app.txt").write_text("owner is mid-edit\n", encoding="utf-8")
    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)
    assert refused["reason"] == REASON_LOCAL_TREE_DIRTY


def test_a_dirty_checkout_can_be_merged_ANYWAY_and_the_answer_says_what_stayed(drive, folder, wt_root):
    """A checkout an agent worked in almost always holds something untracked — a
    log, a build artifact, a scratch file. A refusal with no way past it would
    make merge-back unreachable for exactly the threads that did work, so
    `checkout_dirty` carries A10's consent shape.

    And the success NAMES what stayed: acknowledging that work is left behind is
    not the same as forgetting it was.
    """
    from pathlib import Path

    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "committed\n")
    Path(out["path"], "scratch.log").write_text("agent scratch\n", encoding="utf-8")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)
    assert refused["reason"] == REASON_CHECKOUT_DIRTY
    assert refused["acknowledgeable"] is True
    assert "merge anyway" in refused["message"]

    merged = merge_back_thread(
        drive, "racer", thread["id"], data_dir=drive, busy=False,
        acknowledge_checkout_dirty=True,
    )

    assert merged["ok"] is True, merged
    assert merged["merged"] is True
    assert (folder / "feature.txt").read_text(encoding="utf-8") == "committed\n"
    assert any("scratch.log" in row for row in merged["checkout_left_behind"])
    # The acknowledgement does not cover the WRONG BRANCH case: that is not work
    # left behind, it is a merge that would do nothing while reporting success.
    _git(out["path"], "checkout", "-q", "-b", "elsewhere")
    still = merge_back_thread(
        drive, "racer", thread["id"], data_dir=drive, busy=False,
        acknowledge_checkout_dirty=True,
    )
    assert still["reason"] == REASON_CHECKOUT_HEAD_OFF_BRANCH


def test_a_folder_left_MID_MERGE_is_not_told_to_commit_or_stash(drive, folder, wt_root):
    """After a `merge_abort_failed`, the folder has MERGE_HEAD set and conflict
    markers in the files. The next attempt used to hit the local-tree check and
    answer "commit or stash them first" — advice for a folder with edits in it,
    not one stopped part-way through a merge, and following it would only make
    the state harder to unpick."""
    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "app.txt", "the thread's version\n")
    (folder / "app.txt").write_text("the owner's version\n", encoding="utf-8")
    _git(folder, "commit", "-qam", "owner edit")
    # Put the folder mid-merge for real, and leave it there.
    _git(folder, "-c", "user.name=T", "-c", "user.email=t@example.com",
         "merge", "--no-ff", "--no-edit", out["branch"], check=False)
    assert (folder / ".git" / "MERGE_HEAD").exists()

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["reason"] == REASON_MERGE_ABORT_FAILED
    assert refused["folder_left_mid_merge"] is True
    assert "stash" not in refused["message"]
    assert "git merge --abort" in refused["message"]
    _git(folder, "merge", "--abort")


def test_a_budget_paused_task_does_not_block_merge_back_forever(monkeypatch):
    """T3R-14's PENDING rule, bounded. A budget-exhausted task is parked in
    PENDING with `auto_resume: False` and waits for the owner — possibly forever,
    and across a queue-snapshot restore. Counting it made "this project is busy"
    permanently true, so ONE paused task locked the owner out of merging their own
    work back with nothing on screen to explain why."""
    import ouroboros.thread_branching as branching
    from supervisor import workers

    parked = {
        "id": "p1", "project_id": "racer",
        "_budget_pause": {"status": "paused_before_dispatch", "auto_resume": False},
    }
    monkeypatch.setattr(workers, "PENDING", [parked])
    assert branching.project_is_busy("racer") is False

    # A queued task that CAN still start is a real wait, and still counts.
    monkeypatch.setattr(workers, "PENDING", [{"id": "p2", "project_id": "racer"}])
    assert branching.project_is_busy("racer") is True


def test_an_unbranched_thread_has_nothing_to_merge(drive, folder, wt_root):
    thread = _project(drive, folder)

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert refused["reason"] == REASON_NOT_BRANCHED


def test_project_is_busy_reads_the_project_WIDE_activity_query(monkeypatch):
    """A9 reads "is anything running anywhere in this project", NOT the writer
    lane: a task running in a DIFFERENT folder of the same project still blocks
    a merge, because a merge touches the project as a whole."""
    import ouroboros.project_lease as lease
    import ouroboros.thread_branching as branching
    from supervisor import workers

    elsewhere = {"id": "t1", "project_id": "racer", "workspace_root": "/somewhere/else"}
    monkeypatch.setitem(workers.RUNNING, "t1", {"task": elsewhere})
    try:
        # The lane would say "different folder, go ahead"; the activity query
        # says the project is busy, and that is the one merge-back reads.
        assert lease.running_project_lanes(workers.RUNNING.values()) == {
            ("", __import__("os").path.normcase("/somewhere/else"))
        }
        assert branching.project_is_busy("racer") is True
        assert branching.project_is_busy("other-project") is False
    finally:
        workers.RUNNING.pop("t1", None)


def test_the_activity_query_sees_subagents_and_queued_work(monkeypatch):
    """T3R-14. "Any task running anywhere in this project" was not what the code
    asked: the lane's SUBAGENT exemption came along for the ride and PENDING was
    never read at all.

    The exemption exists so a swarm cannot deadlock against its own parent — a
    scheduling rule about who may be ASSIGNED. A subagent still writes files, and
    a queued task can start the instant after the answer is read, while a merge
    holds no lock against the scheduler.
    """
    import ouroboros.thread_branching as branching
    from supervisor import workers

    member = {"id": "s1", "project_id": "racer", "delegation_role": "subagent",
              "workspace_root": "/w/racer"}
    monkeypatch.setitem(workers.RUNNING, "s1", {"task": member})
    try:
        assert branching.project_is_busy("racer") is True
    finally:
        workers.RUNNING.pop("s1", None)

    assert branching.project_is_busy("racer") is False
    monkeypatch.setattr(workers, "PENDING", [{"id": "q1", "project_id": "racer"}])
    assert branching.project_is_busy("racer") is True
    assert branching.project_is_busy("other") is False


def test_project_is_busy_fails_closed(monkeypatch):
    """"Cannot tell" must never license a merge into a folder something may be
    writing in."""
    import ouroboros.project_lease as lease
    import ouroboros.thread_branching as branching

    def _explode(_running):
        raise RuntimeError("queue unavailable")

    monkeypatch.setattr(lease, "running_project_ids", _explode)

    assert branching.project_is_busy("racer") is True


def test_a_thread_being_deleted_cannot_branch_off_or_merge_back(drive, folder, wt_root):
    """A fenced thread is closed to routing and having its tasks cancelled.
    Provisioning a checkout for it — or merging its branch — would be work on a
    room the owner has already written off."""
    from ouroboros.projects_registry import begin_thread_deletion
    from ouroboros.thread_branching import REASON_THREAD_NOT_LIVE

    thread = _project(drive, folder)
    branched = _branch(drive, "racer", thread["id"], wt_root)
    assert branched["ok"] is True
    begin_thread_deletion(drive, "racer", thread["id"])

    merged = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)
    assert merged["ok"] is False
    assert merged["reason"] == REASON_THREAD_NOT_LIVE

    other = create_thread(drive, "racer", name="Also doomed")
    begin_thread_deletion(drive, "racer", other["id"])
    refused = _branch(drive, "racer", other["id"], wt_root)
    assert refused["ok"] is False
    assert refused["reason"] == REASON_THREAD_NOT_LIVE


def test_an_ARCHIVED_thread_can_still_branch_off(drive, folder, wt_root):
    """Archiving hides a thread; it does not close it."""
    from ouroboros.projects_registry import archive_thread

    thread = _project(drive, folder)
    archive_thread(drive, "racer", thread["id"])

    out = _branch(drive, "racer", thread["id"], wt_root)

    assert out["ok"] is True, out


# --------------------------------------------------------------------------- #
# Branching off must actually BUY concurrency (T3R2-B4)
# --------------------------------------------------------------------------- #

def test_a_branched_threads_task_gets_its_own_folder_lane_and_the_notice_agrees(
    drive, folder, wt_root,
):
    """T3R2-B4, the reproduction. The whole phase exists for this.

    The writer lane is keyed on the FOLDER, so two tasks run at once exactly when
    they name two folders. Nothing bound a THREAD's checkout to the workspace its
    tasks are admitted into: `resolve_room_workspace` read the project's
    `working_dir` and `get_thread_worktree` had ZERO consumers in any admission
    path. So a branched thread's task took the project-folder lane and QUEUED —
    while `queue_notice`, which keys its candidate on `thread_location(...)`,
    answered "this will not wait" and the branch-off copy promised "both can run
    at the same time". The two surfaces disagreed and the owner read the wrong one.

    Both judges are asserted in BOTH directions here, because agreeing by being
    equally wrong is what happened last time.
    """
    from ouroboros.project_lease import candidate_is_leasable, running_project_lanes
    from ouroboros.projects_registry import get_project
    from ouroboros.thread_branching import (
        QUEUE_NOTICE_OWN_CHECKOUT,
        queue_notice,
    )
    from ouroboros.workspace_admission import resolve_room_workspace

    thread = _project(drive, folder)
    branched = _branch(drive, "racer", thread["id"], wt_root)
    assert branched["ok"] is True, branched
    checkout = branched["path"]
    project = get_project(drive, "racer")
    system_repo = drive.parent / "system_repo"

    def _workspace(chat_id):
        return resolve_room_workspace(
            drive_root=drive, system_repo_dir=system_repo,
            project_id="racer", room_chat_id=chat_id,
        )

    # Thread #0 is holding the project folder's lane right now.
    running = [{"task": {"id": "main", "project_id": "racer", "workspace_root": str(folder)}}]
    lanes = running_project_lanes(running)

    # 1. Admission resolves the BRANCHED thread's room to its own checkout.
    resolved, error, decision = _workspace(thread["chat_id"])
    assert (error, decision) == ("", {})
    assert resolved == checkout

    # 2. So the scheduler puts it in a DIFFERENT lane — both run at once.
    assert candidate_is_leasable(
        {"id": "side", "project_id": "racer", "workspace_root": resolved}, lanes,
    ) is True

    # 3. And the sentence the owner reads says the same thing.
    assert queue_notice(
        drive, "racer", thread["id"], data_dir=drive, running=running,
    )["queued"] is False

    # The OTHER direction, same two judges: the unbranched thread #0 resolves to
    # the project folder, queues behind the running task, and is TOLD so.
    zero_ws, zero_error, _ = _workspace(project["chat_id"])
    assert (zero_ws, zero_error) == (str(folder), "")
    assert candidate_is_leasable(
        {"id": "zero", "project_id": "racer", "workspace_root": zero_ws}, lanes,
    ) is False
    zero_notice = queue_notice(drive, "racer", 0, data_dir=drive, running=running)
    assert zero_notice["queued"] is True
    assert zero_notice["remedy"] == "branch_off"

    # And the branched thread waiting on ITSELF: `QUEUE_NOTICE_OWN_CHECKOUT` and
    # its `own` branch were dead code, because nothing could ever occupy that lane.
    own_running = [{"task": {"id": "side", "project_id": "racer", "workspace_root": checkout}}]
    own_notice = queue_notice(
        drive, "racer", thread["id"], data_dir=drive, running=own_running,
    )
    assert own_notice["queued"] is True
    assert own_notice["message"] == QUEUE_NOTICE_OWN_CHECKOUT
    assert own_notice["remedy"] == "", "branching again is advice that does not work"


def test_promoted_admission_binds_the_branched_threads_checkout_as_the_workspace(
    drive, folder, wt_root, tmp_path, monkeypatch,
):
    """The same binding through the REAL admission path, not just its helper.

    `_admit_promoted_workspace` passed only `pid`, so every thread of a project
    was admitted into the project's folder. The room's chat id comes from the
    EVENT: `task["chat_id"]` is rewritten to the project's own chat further up
    when a project is bound during promotion, so by that point it can no longer
    name the room the task was born in.
    """
    from types import SimpleNamespace

    from ouroboros.project_lease import _computed_lane, _task_workspace_root
    from supervisor import workers

    thread = _project(drive, folder)
    branched = _branch(drive, "racer", thread["id"], wt_root)
    assert branched["ok"] is True, branched
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "system_repo")

    task = {"id": "t-side", "chat_id": thread["chat_id"], "text": "do it", "project_id": "racer"}
    outcome = workers._admit_promoted_workspace(
        {"chat_id": thread["chat_id"], "project_id": "racer"},
        SimpleNamespace(), task, pid="racer", tid="t-side",
    )

    assert outcome is None, outcome
    assert task["workspace_root"] == branched["path"]
    assert task["metadata"]["workspace_root"] == branched["path"]
    # Which is the lane it will actually hold — not the project folder's.
    assert _computed_lane(task) == ("", _task_workspace_root(
        {"workspace_root": branched["path"]},
    ))
    assert _computed_lane(task) != _computed_lane(
        {"project_id": "racer", "workspace_root": str(folder)},
    )


def test_the_busy_check_asks_about_the_FOLDER_not_only_the_project_id(monkeypatch, tmp_path):
    """T3R2-M1: T0R2-5 keyed the lane on the folder; this check did not follow.

    Project *alpha* merging into a shared folder while project *beta*'s task
    writes in it reduced to two different project ids and read as idle. So did a
    task carrying a `workspace_root` with NO `project_id` — which holds no lane at
    all and is still in the folder.
    """
    import ouroboros.thread_branching as branching
    from supervisor import workers

    folder = str(tmp_path / "shared")

    def _busy_with(task):
        monkeypatch.setitem(workers.RUNNING, "x", {"task": task})
        try:
            return branching.project_is_busy("alpha", folder)
        finally:
            workers.RUNNING.pop("x", None)

    # Another PROJECT writing in the same folder.
    assert _busy_with({"id": "x", "project_id": "beta", "workspace_root": folder}) is True
    # A task that names the folder and no project at all — it holds no lane, and
    # it is still writing there.
    assert _busy_with({"id": "x", "workspace_root": folder}) is True
    # A different folder in a different project is not this merge's business.
    assert _busy_with(
        {"id": "x", "project_id": "beta", "workspace_root": str(tmp_path / "elsewhere")},
    ) is False
    # ...and the project half still answers on its own, with no folder given.
    assert branching.project_is_busy("alpha") is False


def test_merge_back_HOLDS_the_folder_lane_while_it_rewrites_the_folder(drive, folder, wt_root, monkeypatch):
    """T3R2-M5: the busy check is a bare READ.

    It answers about the instant it ran; the instant after that, the scheduler
    could admit a task straight into the folder being rewritten — the two-writer
    state the lane exists to prevent, reached through the gap between a check and
    the work it checked for.
    """
    import ouroboros.thread_branching as branching
    from ouroboros.project_lease import (
        candidate_is_leasable,
        normalize_workspace_root,
        reserved_folder_lanes,
        running_project_lanes,
    )

    thread = _project(drive, folder)
    out = _branch(drive, "racer", thread["id"], wt_root)
    _commit_in(out["path"], "feature.txt", "from the thread\n")
    lane = ("", normalize_workspace_root(str(folder)))
    newcomer = {"id": "late", "project_id": "racer", "workspace_root": str(folder)}
    assert candidate_is_leasable(newcomer, running_project_lanes([])) is True

    seen = {}
    real_git = branching._git

    def _watch(root, *args):
        if "merge" in args and "--no-ff" in args:
            # Mid-merge: what would the scheduler decide about a task arriving now?
            seen["lanes"] = running_project_lanes([])
            seen["leasable"] = candidate_is_leasable(newcomer, seen["lanes"])
        return real_git(root, *args)

    monkeypatch.setattr(branching, "_git", _watch)
    merged = branching.merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert merged["ok"] is True, merged
    assert lane in seen["lanes"], "the merge must HOLD the folder it is rewriting"
    assert seen["leasable"] is False
    # And it lets go, whatever happened — a failed merge must never strand a
    # folder nobody can schedule into.
    assert reserved_folder_lanes() == set()


def test_a_failed_merge_still_releases_the_folder_it_held(drive, folder, wt_root):
    from ouroboros.project_lease import reserved_folder_lanes

    thread = _project(drive, folder)
    _branch(drive, "racer", thread["id"], wt_root)
    (folder / "app.txt").write_text("owner is mid-edit\n", encoding="utf-8")

    refused = merge_back_thread(drive, "racer", thread["id"], data_dir=drive, busy=False)

    assert refused["ok"] is False
    assert reserved_folder_lanes() == set()


def test_the_git_timeout_is_the_settings_SSOT_not_a_module_local_number(monkeypatch):
    """T3R2-L6: DEVELOPMENT.md's gate says a new numeric timeout is a `config.py`
    setting with a getter and env registration. This module pinned its own 120,
    while the sibling owner-facing git path — the task diff — already read the
    settings getter.

    The gate is satisfied by HAVING a key, not by borrowing the neighbour's: the
    first fix pointed this at `get_task_diff_git_timeout_sec()` and silently
    narrowed branch-off's ceiling from 120s to 30s, on a path that runs
    `git add -A` and `git commit` over a working tree of unknown size. It owns
    `OUROBOROS_THREAD_GIT_TIMEOUT_SEC` now, and the last block is the assertion
    that fails if anyone re-points it at the diff getter.
    """
    import ouroboros.config as config
    import ouroboros.thread_branching as branching

    monkeypatch.delenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", raising=False)
    monkeypatch.delenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", raising=False)

    assert not hasattr(branching, "_GIT_TIMEOUT_SEC")
    assert config.SETTINGS_DEFAULTS["OUROBOROS_THREAD_GIT_TIMEOUT_SEC"] == 120
    assert branching._git_timeout_sec() == config.get_thread_git_timeout_sec() == 120.0

    # Env-overridable and clamped like every sibling reader in `config.py`.
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "77")
    assert branching._git_timeout_sec() == 77.0
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "0")
    assert branching._git_timeout_sec() == 5.0
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "100000")
    assert branching._git_timeout_sec() == 300.0
    monkeypatch.setenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", "not-a-number")
    assert branching._git_timeout_sec() == 120.0

    # NOT the task diff's ceiling: that one bounds a READ against one commit.
    # Turning the diff's knob must not move this path at all.
    monkeypatch.delenv("OUROBOROS_THREAD_GIT_TIMEOUT_SEC", raising=False)
    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "30")
    assert config.get_task_diff_git_timeout_sec() == 30.0
    assert branching._git_timeout_sec() == 120.0
    monkeypatch.setenv("OUROBOROS_TASK_DIFF_GIT_TIMEOUT_SEC", "17")
    assert branching._git_timeout_sec() == 120.0


def test_the_busy_docs_no_longer_claim_that_every_PENDING_task_counts():
    """T3R2-M8 (BIBLE P6): `733b41b8` changed what "busy" means — a budget-paused
    PENDING task no longer counts, because it waits for the owner, possibly
    forever — and touched no doc. `project_is_busy.__doc__` still said a queued
    task "can be assigned at any instant" with no exception, and DEVELOPMENT.md
    still said "so subagents and PENDING work count"."""
    import pathlib

    import ouroboros.thread_branching as branching

    doc = branching.project_is_busy.__doc__ or ""
    assert "BUDGET-PAUSED" in doc
    assert "already taken" in doc
    # ...and the folder half the lane key made necessary.
    assert "FOLDER half" in doc

    development = (
        pathlib.Path(__file__).resolve().parent.parent / "docs" / "DEVELOPMENT.md"
    ).read_text(encoding="utf-8")
    assert "so subagents and PENDING work count" not in development
    assert "EXCEPT what cannot start without the owner" in development
