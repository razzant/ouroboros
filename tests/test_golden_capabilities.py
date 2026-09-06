"""Golden capability suite (#447 WS3-c).

Every case here is a POSITIVE pin drawn from a documented capinv-447 incident:
it asserts that a legitimate capability SURVIVES the immune system, not that
something is blocked. A guard change that turns any of these red is a
capability regression by definition (CHECKLISTS item 21: "name the surviving
positive path"). Deliberately NOT a framework — plain parametrized tests over
the existing entry points; the exhaustive per-branch pins live next to their
fixes (test_declared_params_honored, test_capability_effect_predicates,
test_capinv447_ws2e, test_secret_masking_egress, test_runtime_mode_core).
"""

from __future__ import annotations

import pathlib
import subprocess
import types

import pytest

from ouroboros.tool_access import user_files_path_block_reason
from ouroboros.tools.core import _code_search, _list_files, _read_file, _write_file
from ouroboros.tools.registry import ToolContext, ToolRegistry
from ouroboros.tools import registry_guard_process


def _posix(rendered: str) -> str:
    """Listings and search hits spell paths with the host separator (JSON-escaped
    in a listing); compare them separator-agnostically."""
    return rendered.replace("\\\\", "/").replace("\\", "/")


AWS_SECRET_LINE = "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"


@pytest.fixture()
def user_files_ctx(tmp_path, monkeypatch):
    """ROOT-principal context with an isolated user home carrying real
    credential-shaped names (В23=A: the root reads its owner's home in full;
    secret BYTES are masked at egress, names are never a read-authorization
    input)."""
    home = tmp_path / "home"
    (home / ".aws").mkdir(parents=True)
    (home / ".aws" / "credentials").write_text("[default]\n" + AWS_SECRET_LINE, encoding="utf-8")
    (home / ".bashrc").write_text("export EDITOR=vi\n", encoding="utf-8")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(home))
    system = tmp_path / "system"
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    for p in (system, data, workspace):
        p.mkdir()
    ctx = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                      task_id="t-golden")
    return ctx, home


@pytest.fixture()
def repo_ctx(tmp_path):
    """ROOT-principal context over a real git repo (restore/append/read pins)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=repo, check=True)
    drive = tmp_path / "data"
    drive.mkdir()
    return ToolContext(repo_dir=repo, drive_root=drive, task_id="t-golden")


# ---------------------------------------------------------------------------
# Incident X1/В23 — the ROOT principal reads/lists/searches its owner's home;
# credential-shaped NAMES stopped being a read-authorization input (bytes are
# masked at egress instead of the file being refused at ingress).
# ---------------------------------------------------------------------------

def test_root_reads_credential_named_user_file_masked_not_refused(user_files_ctx):
    ctx, _home = user_files_ctx
    out = _read_file(ctx, ".aws/credentials", root="user_files")
    assert not out.startswith("⚠️"), out[:200]          # the read itself succeeds
    assert "[default]" in out                            # non-secret content survives
    assert "wJalrXUtnFEMI" not in out                    # raw key bytes never egress
    assert "SECRET_BYTES_MASKED" in out                  # disclosure, not silence


def test_root_lists_and_searches_credential_named_user_files(user_files_ctx):
    ctx, _home = user_files_ctx
    listing = _list_files(ctx, path=".aws", root="user_files")
    assert ".aws/credentials" in _posix(listing)         # the name is not hidden
    found = _code_search(ctx, "aws_secret_access_key", root="user_files", path=".aws")
    assert ".aws/credentials" in _posix(found)           # search reaches the file
    assert "wJalrXUtnFEMI" not in found                  # match lines are masked
    assert "SECRET_BYTES_MASKED" in found


@pytest.mark.parametrize("operation", ["read", "list", "search"])
@pytest.mark.parametrize("relpath", [".aws/credentials", ".bashrc"])
def test_root_read_authorization_is_location_only(user_files_ctx, operation, relpath):
    # decide-level pin: for read-family operations under the owner's home the
    # authorization answer is location-only — no name-shape veto.
    ctx, home = user_files_ctx
    assert user_files_path_block_reason(ctx, home / relpath, operation=operation) == ""


# ---------------------------------------------------------------------------
# Incident A3 — "sudo" as DATA (a search pattern, a path argument) is not sudo;
# only command-head sudo is judged. `rg sudo README.md` used to be refused.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "rg sudo README.md",
    "grep -rn sudo ouroboros/",
    "ls /usr/bin/sudo",
])
def test_sudo_named_as_data_passes_the_deterministic_prefilter(tmp_path, cmd):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    assert registry_guard_process._run_shell_safety_check(registry, {"cmd": cmd}, "advanced") is None


# ---------------------------------------------------------------------------
# Incident A7 — read-only git modes (stash list, config get, worktree list)
# pass the merged read allowlist instead of being treated as mutations.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "git stash list",
    "git config --get remote.origin.url",
    "git worktree list",
])
def test_git_read_modes_pass_the_read_allowlist(cmd):
    from ouroboros.tools.registry import _is_pure_read_inspection

    assert _is_pure_read_inspection(cmd) is True


# ---------------------------------------------------------------------------
# Incident A2 — INSPECTION of skill owner-state is a read and passes the
# writeish read-carve (the mention of an owner-state filename is not a write).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "rg 'schema_version' data/state/skills/",
    "grep -rn grants.json data/state/skills",
])
def test_skill_owner_state_inspection_read_passes(tmp_path, cmd):
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    assert registry_guard_process._run_shell_safety_check(registry, {"cmd": cmd}, "advanced") is None


# ---------------------------------------------------------------------------
# Incident G10/В25c — a partially-valid attachment set stages its valid
# members; one rejected sibling no longer voids the whole declaration.
# ---------------------------------------------------------------------------

def test_partial_attachment_set_stages_the_valid_member(tmp_path):
    from ouroboros.artifacts import stage_task_attachments

    aws = tmp_path / ".aws"
    aws.mkdir()
    (aws / "credentials").write_text("[default]\n", encoding="utf-8")
    report = tmp_path / "report.txt"
    report.write_text("findings\n", encoding="utf-8")

    drive = tmp_path / "drive"
    manifest = stage_task_attachments(drive, "task-golden",
                                      [str(aws / "credentials"), str(report)])
    assert [row["status"] for row in manifest] == ["rejected", "staged"]
    staged_rel = manifest[1]["relpath"]
    # The valid member physically lands in the artifact store.
    staged_files = [p for p in drive.rglob("*") if p.is_file()]
    assert len(staged_files) == 1 and staged_files[0].name == "report.txt"
    assert staged_rel.endswith("report.txt")


# ---------------------------------------------------------------------------
# Incident D2 — write_file(mode="append") appends on repo roots (it used to
# silently overwrite, destroying every prior chunk of a chunked write).
# ---------------------------------------------------------------------------

def test_append_mode_appends_on_repo_root(repo_ctx):
    ctx = repo_ctx
    assert _write_file(ctx, path="log.txt", content="head ",
                       root="active_workspace").startswith("✅")
    res = _write_file(ctx, path="log.txt", content="tail",
                      root="active_workspace", mode="append")
    assert res.startswith("✅") and "appended" in res
    assert (pathlib.Path(ctx.repo_dir) / "log.txt").read_text(encoding="utf-8") == "head tail"


# ---------------------------------------------------------------------------
# Incident D1 — read_file(start_char=...) is honored on the repo roots (a long
# one-line file used to re-read the identical head forever).
# ---------------------------------------------------------------------------

def test_start_char_honored_on_repo_root(repo_ctx):
    ctx = repo_ctx
    (pathlib.Path(ctx.repo_dir) / "one_line.txt").write_text(
        "0123456789ABCDEFGHIJ\n", encoding="utf-8")
    out = _read_file(ctx, "one_line.txt", root="active_workspace", start_char=10)
    assert "(from char 10 of this window)" in out
    assert out.endswith("ABCDEFGHIJ\n") and "0123456789" not in out


# ---------------------------------------------------------------------------
# Incident D8 — the protected-path gate on vcs_restore judges the DAMAGE SET;
# a directory restore whose dirty files are all unprotected still restores.
# ---------------------------------------------------------------------------

def test_vcs_restore_directory_with_only_unprotected_dirty_files_restores(repo_ctx):
    from ouroboros.tools import git as git_mod

    ctx = repo_ctx
    repo = pathlib.Path(ctx.repo_dir)
    (repo / "docs").mkdir()
    (repo / "docs" / "guide.md").write_text("original\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    (repo / "docs" / "guide.md").write_text("dirty edit\n", encoding="utf-8")
    (repo / "docs" / "scratch.tmp").write_text("junk\n", encoding="utf-8")

    result = git_mod._restore_to_head(ctx, confirm=True, paths=["docs"])
    assert "Restored" in result, result[:200]
    assert "RESTORE_BLOCKED" not in result and "RESTORE_ERROR" not in result
    assert (repo / "docs" / "guide.md").read_text(encoding="utf-8") == "original\n"
    assert not (repo / "docs" / "scratch.tmp").exists()


# ---------------------------------------------------------------------------
# Incident D4 — declared benign dotfiles export; .env.example is a template,
# .gitignore is project config, neither is a credential.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("component", [(".gitignore",), (".env.example",)])
def test_declared_dotfile_outputs_are_exportable(component):
    from ouroboros.tools.shell_outputs import _sensitive_output_component_reason

    assert _sensitive_output_component_reason(component) == ""


# ---------------------------------------------------------------------------
# Incident D3 — a search over a policy-pruned tree still returns the matches
# it CAN see and honestly discloses what it dropped (no silent completeness).
# ---------------------------------------------------------------------------

def test_search_over_pruned_tree_returns_matches_with_honest_drop_receipt(tmp_path, monkeypatch):
    import ouroboros.code_search_rg as rg_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "match.txt").write_text("needle here\n", encoding="utf-8")
    (repo / "big.txt").write_bytes(b"needle" + b"x" * (rg_mod.MAX_FILE_SIZE_BYTES + 1))
    ctx = types.SimpleNamespace(
        drive_root=str(tmp_path / "data"), repo_dir=str(repo),
        workspace_root="", workspace_mode="", task_metadata={},
    )
    # Force the Python fallback lane deterministically (rg availability varies).
    monkeypatch.setattr(rg_mod, "search_with_rg",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("force fallback")))
    result = _code_search(ctx, "needle", root="active_workspace", path=".")
    assert "match.txt" in result                          # the search still works
    assert "were present but not searched" in result      # the drop is disclosed
    assert "oversized=1" in result


# ---------------------------------------------------------------------------
# Incident В27 — review preflight no longer demands tests for a .py change
# (e.g. comment-only); test sufficiency belongs to the reviewers (item 6).
# ---------------------------------------------------------------------------

def test_comment_only_diff_passes_review_preflight():
    from ouroboros.tools import review

    assert review._preflight_check(
        "clarify comments in the tool loop", "M  ouroboros/loop.py", "/tmp",
    ) is None


# ---------------------------------------------------------------------------
# Incident A1 — a credential-shaped NAME on an ordinary source file is not an
# authorization fact: children read it wherever the location is allowed (the
# suffix carve; conventional credential LOCATIONS like auth/ stay denied —
# that half is pinned in test_query_code.py).
# ---------------------------------------------------------------------------

def test_credential_named_ordinary_source_readable_by_children(tmp_path):
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_MODE

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "token_service.py").write_text(
        "def issue_token():\n    return 'ordinary auth logic'\n", encoding="utf-8")
    ctx = ToolContext(
        repo_dir=repo, drive_root=tmp_path / "data",
        task_constraint=TaskConstraint(mode=LOCAL_READONLY_SUBAGENT_MODE),
    )
    out = _read_file(ctx, "token_service.py", root="active_workspace")
    assert "ordinary auth logic" in out
    assert "REPO_READ_BLOCKED" not in out
