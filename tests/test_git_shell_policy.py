"""Regression suite for the external-workspace git policy classifier.

These pin the bypasses found in the v6.27.0 audit (NW-4): environment-based
repo retargeting (``GIT_DIR`` / ``GIT_WORK_TREE``) and glued / newline shell
separators that previously let ``cd ws;git -C <runtime> reset`` masquerade as a
single ``cd`` segment with the ``-C`` selector never inspected. The round-1
``cd``-bypass fix shipped with zero behavioural coverage; this file is that
coverage plus the env and nested-shell vectors.
"""
import importlib
import pathlib

import pytest

policy = importlib.import_module("ouroboros.git_shell_policy")

WS = pathlib.Path("/tmp/ws")
REPO = pathlib.Path("/Users/anton/Ouroboros/repo")
DATA = pathlib.Path("/Users/anton/Ouroboros/data")
ROOTS = [REPO, DATA]


def _violation(cmd, *, cwd="/tmp/ws", allow_network=True):
    return policy.external_workspace_git_violation(
        cmd, active_root=WS, cwd=cwd, protected_roots=ROOTS, allow_network=allow_network
    )


# --- bypasses that MUST be blocked -----------------------------------------

BLOCKED_CASES = [
    pytest.param("GIT_DIR=/Users/anton/Ouroboros/repo/.git git reset --hard", id="env_git_dir_bare"),
    pytest.param("env GIT_DIR=/Users/anton/Ouroboros/repo/.git git reset --hard", id="env_git_dir_wrapper"),
    pytest.param("GIT_WORK_TREE=/Users/anton/Ouroboros/repo git checkout .", id="env_work_tree"),
    pytest.param("cd /tmp/ws;git -C /Users/anton/Ouroboros/repo reset --hard", id="glued_semicolon_cd_minusC"),
    pytest.param("cd /tmp/ws\ngit -C /Users/anton/Ouroboros/repo reset --hard", id="newline_cd_minusC"),
    pytest.param("git -C /Users/anton/Ouroboros/repo commit -m x", id="direct_minusC_repo_commit"),
    pytest.param("git --git-dir=/Users/anton/Ouroboros/repo/.git reset --hard", id="git_dir_flag_repo_reset"),
    pytest.param("git --work-tree=/Users/anton/Ouroboros/data checkout .", id="work_tree_flag_data_checkout"),
    pytest.param("sh -c 'git -C /Users/anton/Ouroboros/repo reset --hard'", id="nested_sh_c"),
    pytest.param("true && git -C /Users/anton/Ouroboros/repo clean -fd", id="glued_and_minusC"),
    pytest.param("cd /Users/anton/Ouroboros/repo && git commit -am x", id="cd_into_repo_then_commit"),
    # Bidirectional containment: an ANCESTOR target puts repo/ and data/ inside a
    # task-created work tree even though the target CONTAINS the protected roots.
    pytest.param("git -C /Users/anton/Ouroboros init", id="ancestor_init"),
    pytest.param("cd /Users/anton/Ouroboros && git add -A", id="ancestor_cwd_add"),
    pytest.param("git init /Users/anton", id="home_ancestor_init_arg"),
    # Casefold containment: APFS/NTFS are case-insensitive, so a re-cased spelling
    # of a protected root is the SAME directory there.
    pytest.param("git -C /users/anton/ouroboros/REPO reset --hard", id="casefold_minusC_reset"),
    # `reflog` was in GIT_READONLY_SUBCOMMANDS unconditionally, so its MUTATING
    # verbs (`expire`/`delete`/`drop` rewrite or remove reflog entries) skipped
    # the target checks and were PERMITTED against the runtime (SC-10).
    pytest.param("git -C /Users/anton/Ouroboros/repo reflog expire --expire=now --all", id="reflog_expire_runtime"),
    pytest.param("git -C /Users/anton/Ouroboros/repo reflog delete HEAD@{1}", id="reflog_delete_runtime"),
    pytest.param("git -C /Users/anton/Ouroboros/repo reflog drop --all", id="reflog_drop_runtime"),
    pytest.param(
        "git -C /Users/anton/Ouroboros/repo reflog write refs/heads/x 0000000 1111111 msg",
        id="reflog_write_runtime",
    ),
    pytest.param("cd /Users/anton/Ouroboros/repo && git reflog expire --all", id="reflog_expire_cd_runtime"),
    # `git remote` verbs mutate config/refs; flags BEFORE the verb still dispatch
    # (measured: `git remote -v add origin <url>` ADDS the remote), so the verb is
    # the first non-flag token — `-v` must not read the whole invocation as a listing.
    pytest.param("git -C /Users/anton/Ouroboros/repo remote add origin https://example.com/x.git", id="remote_add_runtime"),
    pytest.param("git -C /Users/anton/Ouroboros/repo remote -v add origin https://example.com/x.git", id="remote_verbose_add_runtime"),
    pytest.param("git -C /Users/anton/Ouroboros/repo remote set-url origin https://example.com/x.git", id="remote_set_url_runtime"),
    pytest.param("git -C /Users/anton/Ouroboros/repo remote prune origin", id="remote_prune_runtime"),
]


@pytest.mark.parametrize("cmd", BLOCKED_CASES)
def test_runtime_targeting_git_is_blocked(cmd):
    assert _violation(cmd), f"expected BLOCK for: {cmd!r}"


def test_network_subcommand_blocked_when_network_disabled():
    assert _violation("git clone https://example.com/x.git", allow_network=False)


def test_network_fence_applies_to_readonly_git_too():
    # ls-remote is read-only for TARGET purposes but still reaches the network:
    # the contract fence must hold independently of the read-only carve-out.
    assert _violation("git ls-remote https://example.com/x.git", allow_network=False)
    # The same per-subcommand fence covers the read-only `remote` forms (`remote
    # show` without `-n` contacts the remote) — in the external resolver's tail
    # AND in the classifier the self_worktree lane asks directly.
    assert _violation("git remote show origin", allow_network=False)
    assert policy.run_shell_git_block_reason(
        "git remote show origin", allow_network=False
    ).startswith("task_contract.allowed_resources")
    assert policy.run_shell_git_block_reason("git remote show origin", allow_network=True) == ""


def test_tag_verify_readonly_does_not_loosen_tag_mutations():
    """Moving `-v`/`--verify` to the read-only set must not weaken the rest of
    the tag table: creation, delete and sign forms stay mutating at the runtime."""
    repo = str(REPO)
    assert _violation("git tag v1.0.0", cwd=repo)
    assert _violation("git tag -d v1.0.0", cwd=repo)
    assert _violation("git tag -s v1.0.0 -m x", cwd=repo)
    assert _violation("git tag -f -v v1.0.0", cwd=repo)  # -f still mutates


def test_self_worktree_lane_classifies_the_same_modes():
    """The strict self_worktree lane funnels through the same classifier: the
    read-only verb forms pass, the mutating verbs keep the blanket block."""
    assert policy.run_shell_git_block_reason("git remote -v") == ""
    assert policy.run_shell_git_block_reason("git tag -v v1.0.0") == ""
    assert policy.run_shell_git_block_reason("git reflog show HEAD") == ""
    assert policy.run_shell_git_block_reason("git reflog expire --all") == "git reflog"
    assert policy.run_shell_git_block_reason("git remote add origin https://e.com/x.git") == "git remote"
    ws_kwargs = dict(active_root=WS, cwd=str(WS))
    assert policy.workspace_git_safety_violation("git remote -v", **ws_kwargs) == ""
    assert policy.workspace_git_safety_violation("git reflog delete HEAD@{1}", **ws_kwargs)


# --- legitimate git that MUST stay allowed -----------------------------------

ALLOWED_CASES = [
    pytest.param("git status", id="status"),
    pytest.param("git commit -m 'work'", id="commit"),
    pytest.param("git clone https://example.com/x.git", id="clone_network_on"),
    pytest.param("cd /tmp/ws/sub && git commit -m x", id="cd_subdir_commit"),
    pytest.param("git -C /tmp/ws/sub status", id="minusC_inside_workspace"),
    pytest.param("echo 'git -C /Users/anton/Ouroboros/repo' > note.txt", id="echo_mentions_git_not_a_git_cmd"),
    pytest.param("grep -r 'git reset' .", id="grep_mentions_git"),
    # READ-ONLY git stays allowed even AT a runtime target (v4.5.1 / f14baf8f
    # false-block line): inspection is the vcs_read status lane.
    pytest.param("git -C /Users/anton/Ouroboros/repo status", id="readonly_minusC_repo_status"),
    pytest.param("git --git-dir=/Users/anton/Ouroboros/repo/.git log", id="readonly_git_dir_log"),
    pytest.param("cd /Users/anton/Ouroboros/repo && git status", id="readonly_cd_repo_status"),
    pytest.param("cd /Users/anton/Ouroboros/repo && git diff", id="readonly_cd_repo_diff"),
    pytest.param("git -C /Users/anton/Ouroboros/repo branch -l", id="readonly_branch_list_repo"),
    # Read-only forms of the verb-dispatched subcommands stay allowed at a runtime
    # target too — the ARCHITECTURE "Safety and runtime mode" contract ("read-only
    # shell git is allowed everywhere"). These
    # were refused before the mode parse: `remote` had no read-only classifier at
    # all, and `tag -v/--verify` (signature check, writes nothing) sat in the
    # mutating flag set (SC-7).
    pytest.param("git -C /Users/anton/Ouroboros/repo remote -v", id="readonly_remote_verbose_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo remote", id="readonly_remote_bare_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo remote show origin", id="readonly_remote_show_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo remote get-url origin", id="readonly_remote_get_url_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo tag -v v1.0.0", id="readonly_tag_verify_short_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo tag --verify v1.0.0", id="readonly_tag_verify_long_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo reflog", id="readonly_reflog_bare_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo reflog show HEAD", id="readonly_reflog_show_repo"),
    pytest.param("git -C /Users/anton/Ouroboros/repo reflog exists refs/heads/main", id="readonly_reflog_exists_repo"),
    # ... while the MUTATING verbs remain free OUTSIDE the runtime (target-aware,
    # not subcommand-blanket): the same spellings blocked above at the repo.
    pytest.param("git remote add origin https://example.com/x.git", id="remote_add_outside_runtime"),
    pytest.param("git reflog expire --expire=now --all", id="reflog_expire_outside_runtime"),
    pytest.param("git tag -d v1.0.0", id="tag_delete_outside_runtime"),
]


@pytest.mark.parametrize("cmd", [
    pytest.param("git -C /tmp/proj commit -m x", id="minusC_outside_from_runtime_cwd"),
    pytest.param("git -C /tmp/proj init", id="minusC_outside_init_from_runtime_cwd"),
    pytest.param("git -C /tmp -C proj add -A", id="chained_minusC_outside_from_runtime_cwd"),
])
def test_minusC_retarget_outside_runtime_allowed_from_runtime_cwd(cmd):
    """Global `-C` chdirs BEFORE git runs: the effective directory — not the
    shell cwd — is the mutating target. The default (non-workspace) lane's
    default cwd IS the system repo, so `git -C <outside-tree> ...` from there
    must stay allowed or the flip re-creates the false-block class."""
    assert _violation(cmd, cwd=str(REPO)) == ""


def test_minusC_into_runtime_still_blocked_from_any_cwd():
    assert _violation("git -C /Users/anton/Ouroboros/repo commit -m x", cwd="/tmp/ws")
    # A RELATIVE -C resolves against the shell cwd and is still caught.
    assert _violation("git -C repo reset --hard", cwd="/Users/anton/Ouroboros")


def test_commit_minusC_message_reuse_is_not_a_path():
    # `git commit -C <commit>` (after the subcommand) reuses a commit message;
    # it must not be parsed as a path selector.
    assert _violation("git commit -C HEAD", cwd="/tmp/ws") == ""


@pytest.mark.parametrize("cmd", ALLOWED_CASES)
def test_legitimate_workspace_git_is_allowed(cmd):
    assert not _violation(cmd), f"expected ALLOW for: {cmd!r}"


def test_clone_allowed_when_network_enabled():
    assert not _violation("git clone https://example.com/x.git", allow_network=True)


# --- destination-aware init/clone (the default lane's cwd IS the system repo) ---

@pytest.mark.parametrize("cmd", [
    pytest.param("git init /tmp/proj", id="init_positional_destination_outside"),
    pytest.param("git init --bare /tmp/proj.git", id="init_bare_destination_outside"),
    pytest.param("git clone https://example.com/x.git /tmp/proj", id="clone_url_destination_outside"),
    pytest.param("git clone git@github.com:o/x.git /tmp/proj", id="clone_scp_url_destination_outside"),
])
def test_init_clone_with_outside_destination_allowed_from_runtime_cwd(cmd):
    """`git init <dir>` / `git clone <url> <dir>` mutate the DESTINATION, not the
    working directory. The default (non-workspace) lane's default cwd IS the system
    repo, so judging these against the cwd refused `git init ~/projects/foo` from
    direct chat and light mode — contradicting the owner contract that mutating git
    is free OUTSIDE the runtime in every lane and mode."""
    assert _violation(cmd, cwd=str(REPO)) == "", cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git init", id="init_no_destination_at_runtime_cwd"),
    pytest.param("git clone https://example.com/x.git", id="clone_no_destination_at_runtime_cwd"),
    pytest.param("git init /Users/anton/Ouroboros/repo/sub", id="init_destination_inside_runtime"),
    pytest.param("git init /Users/anton/Ouroboros", id="init_destination_is_runtime_ancestor"),
    pytest.param("git clone https://example.com/x.git /Users/anton/Ouroboros/data/x", id="clone_destination_inside_data"),
    pytest.param(
        "git init --separate-git-dir=/Users/anton/Ouroboros/repo/x /tmp/proj",
        id="init_separate_git_dir_into_runtime",
    ),
])
def test_init_clone_targeting_the_runtime_still_blocked(cmd):
    """With NO explicit destination the cwd IS the target, and an explicit
    destination (or a `--flag=<path>` retarget) inside/over a runtime root stays
    blocked — the unwind frees the tree, never the runtime."""
    assert _violation(cmd, cwd=str(REPO)), cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git init -b main /tmp/proj", id="init_split_branch_flag"),
    pytest.param("git init --initial-branch=main /tmp/proj", id="init_glued_branch_flag"),
    pytest.param("git init --template /tmp/tpl /tmp/proj", id="init_split_template_flag"),
    pytest.param("git clone --depth 1 https://example.com/x.git /tmp/proj", id="clone_split_depth"),
    pytest.param("git clone -b main https://example.com/x.git /tmp/proj", id="clone_split_branch"),
    pytest.param("git clone -o upstream https://example.com/x.git /tmp/proj", id="clone_split_origin"),
    pytest.param("git clone -j 4 https://example.com/x.git /tmp/proj", id="clone_split_jobs"),
    pytest.param("git clone --filter blob:none https://example.com/x.git /tmp/proj", id="clone_split_filter"),
    pytest.param("git clone --reference /tmp/ref https://example.com/x.git /tmp/proj", id="clone_split_reference"),
])
def test_destination_is_parsed_through_value_taking_flags(cmd):
    """The Q4=A headline capability for its COMMONEST spellings. Positional index
    arithmetic over a diff/branch/tag flag table knew nothing about init/clone's own
    value-taking flags, so `git init -b main /tmp/proj` resolved destination='main'
    and `git clone --depth 1 <url> /tmp/proj` resolved destination='<url>' — both
    relative, both joined onto the default lane's cwd, which IS the system repo."""
    assert _violation(cmd, cwd=str(REPO)) == "", cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git init -b main repo/newtree", id="init_relative_destination_into_repo"),
    pytest.param("git init --initial-branch=main data/newtree", id="init_glued_flag_relative_into_data"),
    pytest.param("git clone --depth 1 https://example.com/x.git repo/newtree", id="clone_relative_into_repo"),
    pytest.param("git clone -b main https://example.com/x.git data/newtree", id="clone_relative_into_data"),
])
def test_relative_destination_into_the_runtime_is_blocked(cmd):
    """Mirror of the mis-parse: with an explicit destination the argument scan was
    narrowed to `--flag=<path>` forms and ABSOLUTE paths, so a RELATIVE destination
    pointing back INTO the runtime was never resolved. Relative candidates in that
    branch must be canonicalized like everywhere else."""
    assert _violation(cmd, cwd="/Users/anton/Ouroboros"), cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git clone -b feature/x https://example.com/x.git /tmp/proj", id="clone_split_slash_branch"),
    pytest.param("git clone --branch=feature/x https://example.com/x.git /tmp/proj", id="clone_glued_slash_branch"),
    pytest.param("git init --initial-branch=feature/x /tmp/proj", id="init_glued_slash_branch"),
    pytest.param("git clone --revision refs/heads/x https://example.com/x.git /tmp/proj", id="clone_split_slash_revision"),
])
def test_nonpath_flag_values_with_slashes_are_not_resolved_as_paths(cmd):
    """A hierarchical ref (`feature/x`, `refs/heads/x`) is path-SHAPED but is a
    NAME to git. The destination-branch argument scan resolved it under the
    effective base — from the default lane's system-repo cwd that lands inside
    the runtime, false-blocking the Q4=A headline spelling with an idiomatic
    branch name."""
    assert _violation(cmd, cwd=str(REPO)) == "", cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git clone -b feature/x https://example.com/x.git repo/newtree", id="slash_branch_does_not_hide_bad_destination"),
    pytest.param("git init --separate-git-dir repo/x /tmp/proj", id="split_separate_git_dir_relative_into_runtime"),
    pytest.param("git init --separate-git-dir repo /tmp/proj", id="split_separate_git_dir_bare_name_into_runtime"),
    pytest.param("git init --separate-git-dir=repo /tmp/proj", id="glued_separate_git_dir_bare_name_into_runtime"),
    pytest.param("git clone --reference repo https://example.com/x.git /tmp/proj", id="split_reference_relative_into_runtime"),
])
def test_nonpath_value_skip_does_not_weaken_containment(cmd):
    """The skip covers only values git reads as names/numbers: the destination
    itself and every PATH-taking flag value (`--separate-git-dir`, `--reference`,
    `--template`) keep meeting the containment scan — by the FLAG's documented
    type, so even a BARE name (`--separate-git-dir repo` from the runtime's
    parent, which the path-SHAPE test alone would skip) is resolved and judged."""
    assert _violation(cmd, cwd="/Users/anton/Ouroboros"), cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git init -b main", id="init_branch_flag_no_destination"),
    pytest.param("git clone --depth 1 /tmp/src", id="clone_local_source_no_destination"),
])
def test_flagged_init_clone_without_a_destination_keeps_the_cwd_check(cmd):
    """A flag VALUE left among the positionals is not a destination: `git init -b
    main` and `git clone <src>` both mutate the CWD, so the cwd containment check
    must still run (the default lane's cwd IS the system repo)."""
    assert _violation(cmd, cwd=str(REPO)), cmd


# --- read-only git that WRITES (the `--output` diff option) --------------------

@pytest.mark.parametrize("cmd", [
    pytest.param("git log --output=/Users/anton/Ouroboros/data/settings.json", id="log_output_into_data"),
    pytest.param("git diff --output=/Users/anton/Ouroboros/repo/BIBLE.md", id="diff_output_into_repo"),
    pytest.param("git show --output /Users/anton/Ouroboros/data/settings.json", id="show_split_output_into_data"),
    pytest.param("git -C /tmp/x log --output=/Users/anton/Ouroboros/data/logs/chat.jsonl", id="minusC_output_into_data"),
])
def test_readonly_git_cannot_write_into_the_runtime_via_output(cmd):
    """`log`/`show`/`diff` all accept the diff option `--output=<file>` and TRUNCATE
    it (measured against real git). The read-only exemption skipped the whole target
    block, so a "read-only" git could overwrite settings.json / any repo file."""
    assert _violation(cmd, cwd="/tmp/ws"), cmd


@pytest.mark.parametrize("cmd", [
    pytest.param("git log --output=/tmp/history.txt", id="output_to_scratch"),
    pytest.param("git diff --output=report.diff", id="output_relative_inside_workspace"),
])
def test_readonly_git_output_outside_the_runtime_stays_allowed(cmd):
    """The mutation lands at the FILE, not at the cwd — judging the cwd instead
    would refuse `git log --output=/tmp/x` from the default lane's system-repo cwd."""
    assert _violation(cmd, cwd="/tmp/ws") == "", cmd
    assert _violation("git log --output=/tmp/history.txt", cwd=str(REPO)) == ""


def test_relative_argument_symlinked_into_the_runtime_is_blocked():
    """A relative arg with no `..` was skipped on the assumption that a plain
    descend from a safe base cannot reach a protected root. A SYMLINK breaks that
    assumption, and resolve() follows it."""
    import os
    import tempfile

    base = pathlib.Path(tempfile.mkdtemp()).resolve()
    runtime = base / "runtime"
    runtime.mkdir()
    tree = base / "tree"
    tree.mkdir()
    os.symlink(runtime, tree / "link")
    violation = policy.external_workspace_git_violation(
        "git init ./link", active_root=tree, cwd="", protected_roots=[runtime], allow_network=True,
    )
    assert violation, "a symlinked relative destination must not bypass containment"


# --- read-only git classification (the runtime-read guard's exemption key) -----

@pytest.mark.parametrize("cmd,expected", [
    pytest.param("git status", True, id="ro_status"),
    pytest.param("git -C /Users/anton/Ouroboros/repo log", True, id="ro_minusC_runtime_log"),
    pytest.param("git log; git diff", True, id="ro_two_git_segments"),
    pytest.param("cd /Users/anton/Ouroboros/repo && git status", True, id="ro_cd_then_git"),
    pytest.param("git branch -l", True, id="ro_branch_list"),
    pytest.param("sh -c 'git log'", True, id="ro_nested_shell"),
    pytest.param("git commit -m x", False, id="mut_commit"),
    pytest.param("git branch -D x", False, id="mut_branch_delete"),
    pytest.param("git status && cat /Users/anton/Ouroboros/data/settings.json", False, id="mixed_git_and_cat"),
    pytest.param("cat /etc/passwd", False, id="not_git_at_all"),
    pytest.param("sh -c 'git commit -m x'", False, id="mut_nested_shell"),
    pytest.param("", False, id="empty"),
    # A read-only-looking invocation that WRITES a file is not read-only: the diff
    # option `--output=<file>` truncates it, so it must meet the runtime/secret guard.
    pytest.param("git log --output=/tmp/x.txt", False, id="output_flag_writes"),
    pytest.param("git diff --output /tmp/x.txt", False, id="output_flag_split_writes"),
    # ... and one that READS ARBITRARY FILES is not repo inspection either:
    # `git diff --no-index /dev/null <secret>` PRINTS the file (measured).
    pytest.param("git diff --no-index /dev/null /Users/anton/Ouroboros/data/settings.json", False, id="no_index_reads_any_file"),
    # `-o` must NOT be read as an output flag: for this family it means something
    # else entirely (`git ls-files -o` == --others) and matching it would refuse an
    # ordinary untracked-file listing.
    pytest.param("git ls-files -o", True, id="ls_files_others_is_not_an_output_flag"),
    # `--output-indicator-new=<char>` is not a file either.
    pytest.param("git diff --output-indicator-new=X", True, id="output_indicator_is_not_a_file"),
    # `pushd`/`popd` are as neutral as `cd`: they read and write nothing.
    pytest.param("pushd /Users/anton/Ouroboros/repo && git log", True, id="ro_pushd_then_git"),
    # Verb-dispatched subcommands: the read-only MODES carry the exemption key,
    # the mutating verbs never do. `git remote -v add ...` really ADDS (measured),
    # so the verb is the first non-flag token, not args[0].
    pytest.param("git remote -v", True, id="ro_remote_verbose"),
    pytest.param("git remote show origin", True, id="ro_remote_show"),
    pytest.param("git remote get-url origin", True, id="ro_remote_get_url"),
    pytest.param("git remote add origin https://example.com/x.git", False, id="mut_remote_add"),
    pytest.param("git remote -v add origin https://example.com/x.git", False, id="mut_remote_verbose_add"),
    pytest.param("git tag -v v1.0.0", True, id="ro_tag_verify_short"),
    pytest.param("git tag --verify v1.0.0", True, id="ro_tag_verify_long"),
    pytest.param("git tag v1.0.0", False, id="mut_tag_create"),
    pytest.param("git tag -s v1.0.0 -m x", False, id="mut_tag_sign"),
    pytest.param("git reflog", True, id="ro_reflog_bare"),
    pytest.param("git reflog show HEAD", True, id="ro_reflog_show"),
    pytest.param("git reflog exists refs/heads/main", True, id="ro_reflog_exists"),
    pytest.param("git reflog expire --expire=now --all", False, id="mut_reflog_expire"),
    pytest.param("git reflog delete HEAD@{1}", False, id="mut_reflog_delete"),
    pytest.param("git reflog drop --all", False, id="mut_reflog_drop"),
])
def test_is_readonly_git_command(cmd, expected):
    """ALL-or-nothing per segment: this is what lets read-only git through the
    external-workspace runtime-read guard WITHOUT opening a shell bypass for the
    secret/credential surface."""
    assert policy.is_readonly_git_command(cmd) is expected, cmd


def test_pushd_moves_the_segment_walker_base_like_cd():
    """Only `cd` moved the walker's base, so `pushd <runtime> && git commit` was
    judged against the ORIGINAL base while the shell had actually chdir'd into the
    runtime. At base the default lane blanket-blocked mutating git, so the flip is
    what opens this; `popd` restores the pushed base."""
    assert _violation("pushd /Users/anton/Ouroboros/repo && git commit -am x", cwd="/tmp/ws")
    assert _violation("pushd /tmp/proj && git commit -am x", cwd=str(REPO)) == ""
    assert _violation("pushd /tmp/proj && popd && git commit -am x", cwd=str(REPO))
