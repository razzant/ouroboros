"""Effect/position predicates instead of text-mention denies (#447 WS2-d).

A2: skill owner-state mentions take the family writeish + read-carve;
A3: sudo is judged at command-head position, never as a data token;
A4: interpreter write inference proves the receiver / agrees across regex+AST;
A6: skill preflight permission findings prove the PluginAPI receiver or degrade;
A7: git read-only classification is one SSOT with mode parsers, gh is argv-parsed.
"""

import textwrap

import pytest

from ouroboros.shell_parse import sudo_noninteractive_violation


# ---------------------------------------------------------------------------
# A3 — sudo head-walk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    ["rg", "sudo", "README.md"],
    ["printf", "sudo"],
    ["grep", "-rn", "sudo", "ouroboros/"],
    ["ls", "/usr/bin/sudo"],
    ["grep", "-rn", "sudoedit", "docs/"],
    ["pytest", "tests/test_sudo_option_parser.py"],
    ["sudo", "-n", "true"],
    ["sudo", "--non-interactive", "true"],
])
def test_sudo_named_as_data_or_noninteractive_is_allowed(cmd):
    assert sudo_noninteractive_violation(cmd) is False


@pytest.mark.parametrize("cmd", [
    ["sudo", "true"],
    ["sudoedit", "/etc/hosts"],
    ["sudo", "-S", "true"],
    ["sudo", "-nS", "true"],
    ["sh", "-c", "sudo apt update"],
    ["nohup", "sudo", "apt", "install", "-y", "jq"],
    "echo hi && sudo whoami",
    ["sudo", "-n", "sh", "-c", "sudo whoami"],
])
def test_sudo_at_command_head_is_still_guarded(cmd):
    assert sudo_noninteractive_violation(cmd) is True


# ---------------------------------------------------------------------------
# A2 — skill owner-state mention takes the read-carve
# ---------------------------------------------------------------------------


def _registry(tmp_path):
    from ouroboros.tools.registry import ToolRegistry

    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir(exist_ok=True)
    drive.mkdir(exist_ok=True)
    return ToolRegistry(repo_dir=repo, drive_root=drive)


@pytest.mark.parametrize("cmd", [
    "rg 'review.json' data/state/skills/",
    "grep -rn enabled.json data/state/skills",
])
def test_skill_state_pure_read_inspection_is_allowed(tmp_path, cmd):
    # The runtime_data file plane explicitly allows reading review.json; the
    # shell plane must not refuse the same read with a WRITE-named marker.
    blocked = _registry(tmp_path)._run_shell_safety_check({"cmd": cmd}, "advanced")
    assert blocked is None


@pytest.mark.parametrize("cmd", [
    "rm data/state/skills/weather/review.json",
    "cp payload.json data/state/skills/weather/grants.json",
    'python -c "open(\'data/state/skills/w/enabled.json\', \'w\').write(\'{}\')"',
])
def test_skill_state_write_shapes_stay_blocked(tmp_path, cmd):
    blocked = _registry(tmp_path)._run_shell_safety_check({"cmd": cmd}, "advanced")
    assert blocked is not None and "SKILL_STATE_WRITE_BLOCKED" in blocked


# ---------------------------------------------------------------------------
# A4 — receiver proof in the AST walker + regex/AST agreement
# ---------------------------------------------------------------------------


def _walk(code: str):
    from ouroboros.tools.shell_guards import _python_write_targets_and_unknown

    return _python_write_targets_and_unknown(textwrap.dedent(code))


def test_str_literal_receiver_replace_is_not_a_write():
    targets, unknown = _walk("s = 'a,b'\nprint(s.replace(',', ';'))\n")
    assert targets == [] and unknown is False


def test_list_literal_receiver_remove_is_not_a_write():
    targets, unknown = _walk("xs = [1, 2]\nitem = 2\nxs.remove(item)\n")
    assert targets == [] and unknown is False


def test_local_class_save_call_site_is_not_a_write():
    code = """
    class A:
        def save(self):
            return 1
    A().save()
    """
    targets, unknown = _walk(code)
    assert targets == [] and unknown is False


def test_local_class_save_with_real_write_inside_is_still_seen():
    code = """
    class A:
        def save(self):
            open('out.txt', 'w').write('x')
    A().save()
    """
    targets, unknown = _walk(code)
    assert "out.txt" in targets


def test_path_receiver_rename_is_still_a_write():
    targets, _ = _walk("import pathlib\np = pathlib.Path('a.txt')\np.write_text('x')\n")
    # Name bound via a Path(...) call is NOT carved as a str receiver.
    assert "a.txt" in targets


def test_sqlite_readonly_uri_is_not_a_write():
    targets, unknown = _walk(
        "import sqlite3\nsqlite3.connect('file:/tmp/db.sqlite?mode=ro', uri=True)\n"
    )
    assert targets == [] and unknown is False


def test_sqlite_plain_connect_is_still_a_write():
    targets, unknown = _walk("import sqlite3\nsqlite3.connect('db.sqlite')\n")
    assert "db.sqlite" in targets or unknown


def test_regex_lane_agrees_with_ast_on_list_remove_and_ro_sqlite():
    from ouroboros.tools.write_shape import interpreter_write_shape

    assert interpreter_write_shape(["python", "-c", "xs=[1,2]; xs.remove(1)"]) is False
    assert interpreter_write_shape(
        ["python", "-c", "import sqlite3; sqlite3.connect('file:/tmp/x?mode=ro', uri=True)"]
    ) is False
    assert interpreter_write_shape(["python", "-c", "import os; os.remove('x')"]) is True
    assert interpreter_write_shape(["python", "-c", "import os; os.rename('a','b')"]) is True


# ---------------------------------------------------------------------------
# A7 — git mode parsers, one read-only SSOT, glued -C, gh argv parse
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cmd", [
    "git config --get remote.origin.url",
    "git config --list",
    "git config user.name",
    "git stash list",
    "git stash show",
    "git worktree list",
    "git notes list",
    "git notes",
    "git bisect log",
    "git merge-base HEAD origin/main",
    "git check-ignore -v build/",
    "git name-rev HEAD",
])
def test_git_readonly_modes_are_classified_readonly(cmd):
    from ouroboros.git_shell_policy import is_readonly_git_command

    assert is_readonly_git_command(cmd) is True


@pytest.mark.parametrize("cmd", [
    "git config user.name somebody",
    "git config --unset user.name",
    "git config set user.name x",
    "git config -f /tmp/other --list",  # external-file reader: no exemption ride
    "git stash",
    "git stash pop",
    "git worktree add ../w",
    "git notes add -m x",
    "git bisect reset",
])
def test_git_mutating_or_external_modes_are_not_readonly(cmd):
    from ouroboros.git_shell_policy import is_readonly_git_command

    assert is_readonly_git_command(cmd) is False


def test_pure_read_inspection_uses_the_same_git_ssot():
    from ouroboros.tools.registry import _is_pure_read_inspection

    assert _is_pure_read_inspection("git stash list") is True
    assert _is_pure_read_inspection("git config --get remote.origin.url") is True
    assert _is_pure_read_inspection("git config user.name x") is False
    assert _is_pure_read_inspection("git log --output=/tmp/x") is False


def test_glued_dash_c_selects_the_same_base_as_split(tmp_path):
    from ouroboros.git_shell_policy import external_workspace_git_violation

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "proj"
    outside.mkdir()
    for spelling in (f"git -C {runtime} commit -m x", f"git -C{runtime} commit -m x"):
        assert external_workspace_git_violation(
            spelling,
            active_root=outside,
            cwd=str(outside),
            protected_roots=[runtime],
        ), spelling
    for spelling in (f"git -C {outside} commit -m x", f"git -C{outside} commit -m x"):
        assert external_workspace_git_violation(
            spelling,
            active_root=outside,
            cwd=str(runtime),
            protected_roots=[runtime],
        ) == "", spelling


@pytest.mark.parametrize("cmd", [
    "rg 'gh auth' docs/",
    "echo 'run gh auth login later'",
    "gh auth status",
    "gh auth token",
    "gh pr list",
])
def test_gh_mentions_and_readonly_auth_are_allowed(cmd):
    from ouroboros.git_shell_policy import gh_shell_block_reason

    assert gh_shell_block_reason(cmd) == ""


@pytest.mark.parametrize("cmd", [
    "gh auth login",
    "gh auth logout",
    "gh auth refresh",
    "gh auth setup-git",
    "sh -c 'gh auth login'",
    "gh repo create mine",
    "gh repo delete mine",
])
def test_gh_mutating_verbs_at_head_are_blocked(cmd):
    from ouroboros.git_shell_policy import gh_shell_block_reason

    assert "SAFETY_VIOLATION" in gh_shell_block_reason(cmd)


# ---------------------------------------------------------------------------
# A6 — skill preflight permission findings prove the receiver or degrade
# ---------------------------------------------------------------------------


def _permission_findings(tmp_path, plugin_code: str, permissions=()):
    from ouroboros.contracts.skill_manifest import SkillManifest
    from ouroboros.tools.skill_preflight import _plugin_permission_findings

    skill_dir = tmp_path / "skill"
    skill_dir.mkdir(exist_ok=True)
    (skill_dir / "plugin.py").write_text(textwrap.dedent(plugin_code), encoding="utf-8")
    manifest = SkillManifest(
        name="x", description="d", version="0.1.0", type="extension",
        entry="plugin.py", permissions=list(permissions),
    )
    return _plugin_permission_findings(skill_dir, manifest)


def test_api_receiver_call_without_permission_still_blocks(tmp_path):
    findings = _permission_findings(tmp_path, """
    def register(api):
        api.get_settings(['KEY'])
    """)
    assert [f for f in findings if f["permission"] == "read_settings" and f["ok"] is False]


def test_api_alias_receiver_is_still_proven(tmp_path):
    findings = _permission_findings(tmp_path, """
    def register(api):
        a = api
        a.register_route('/x', None)
    """)
    assert [f for f in findings if f["permission"] == "route" and f["ok"] is False]


def test_foreign_receiver_degrades_to_note_not_block(tmp_path):
    findings = _permission_findings(tmp_path, """
    import otherlib

    def register(api):
        otherlib.OtherLibrary().get_settings()
    """)
    row = [f for f in findings if f["permission"] == "read_settings"]
    assert row and row[0]["ok"] is True and row[0].get("degraded") is True


def test_declared_permission_stays_ok(tmp_path):
    findings = _permission_findings(tmp_path, """
    def register(api):
        api.get_settings(['KEY'])
    """, permissions=["read_settings"])
    row = [f for f in findings if f["permission"] == "read_settings"]
    assert row and row[0]["ok"] is True and "degraded" not in row[0]
