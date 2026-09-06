"""GitHub tools retain Project selection and explicit targets through every subcall."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from ouroboros.tools import github
from ouroboros.tools.registry import ToolContext


def _context(tmp_path, kind):
    system = tmp_path / "system"
    project = tmp_path / "project"
    system.mkdir(exist_ok=True)
    project.mkdir(exist_ok=True)
    ctx = ToolContext(repo_dir=system, drive_root=tmp_path / "data", task_id="task-fixture")
    if kind == "queued":
        ctx.workspace_root, ctx.workspace_mode, ctx.project_id = project, "external", "project-fixture"
    elif kind == "room":
        ctx.is_direct_chat, ctx.project_id = True, "project-fixture"
        ctx.task_metadata = {"_project_room_dir": str(project)}
    return ctx, project if kind != "system" else system


@pytest.fixture
def gh_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        output = ""
        if argv[1:3] in (["issue", "list"], ["pr", "list"]):
            output = "[]"
        elif argv[1:3] in (["issue", "view"], ["pr", "view"]):
            output = json.dumps({"number": 7, "title": "Fixture", "state": "OPEN", "author": {"login": "fixture"}})
        elif argv[1:3] == ["issue", "create"]:
            output = "https://github.com/owner/selected/issues/7"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


_CALLS = [
    ("list_github_issues", {}, 1), ("get_github_issue", {"number": 7}, 1),
    ("comment_on_issue", {"number": 7, "body": "text"}, 1),
    ("close_github_issue", {"number": 7, "comment": "closing"}, 2),
    ("create_github_issue", {"title": "Title", "body": "Body", "labels": "bug"}, 2),
    ("list_github_prs", {}, 1), ("get_github_pr", {"number": 7}, 3),
    ("comment_on_pr", {"number": 7, "body": "text"}, 1),
]


@pytest.mark.parametrize("kind", ["system", "queued", "room"])
@pytest.mark.parametrize("name,args,count", _CALLS)
@pytest.mark.parametrize("repo", ["", "github.example/owner/selected"])
def test_every_repository_tool_keeps_target_in_all_subcalls(tmp_path, gh_calls, monkeypatch, kind, name, args, count, repo):
    ctx, expected_cwd = _context(tmp_path, kind)
    monkeypatch.setenv("GH_REPO", "unrelated/wrong-repo")
    monkeypatch.setenv("GH_HOST", "configured.example")
    entry = next(item for item in github.get_tools() if item.name == name)

    result = entry.handler(ctx, **args, repo=repo)

    assert not result.startswith("⚠️"), result
    assert len(gh_calls) == count
    assert "repo" in entry.schema["parameters"]["properties"]
    for argv, kwargs in gh_calls:
        assert kwargs["cwd"] == str(expected_cwd)
        if repo:
            assert argv[-2:] == ["--repo", repo]
        else:
            assert "--repo" not in argv
        if kind != "system":
            assert "GH_REPO" not in kwargs["env"]
        assert kwargs["env"]["GH_HOST"] == "configured.example"
    if name == "get_github_pr" and (kind != "system" or repo):
        assert "fetch_pr_ref(" not in result
        assert "stage_pr_merge(" not in result


@pytest.mark.parametrize("failure", ["note", "missing-room", "fileless", "missing-workspace", "invalid-workspace-mode"])
def test_unusable_project_never_calls_gh_on_system_repo(tmp_path, gh_calls, failure):
    ctx, project = _context(tmp_path, "queued" if "workspace" in failure else "room")
    if failure == "note":
        ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    elif failure == "fileless":
        ctx.task_metadata = {}
    elif failure == "invalid-workspace-mode":
        ctx.workspace_mode = ""
    else:
        project.rmdir()

    result = github._list_issues(ctx)

    assert "GH_TARGET_" in result
    assert gh_calls == []


def test_fileless_room_accepts_explicit_repo(tmp_path, gh_calls):
    ctx, _ = _context(tmp_path, "room")
    ctx.task_metadata = {}
    assert not github._get_issue(ctx, 7, repo="owner/selected").startswith("⚠️")
    assert gh_calls[0][0][-2:] == ["--repo", "owner/selected"]


def test_generic_hub_transport_keeps_explicit_api_contract(tmp_path, gh_calls, monkeypatch):
    ctx, _ = _context(tmp_path, "room")
    ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    monkeypatch.setenv("GH_REPO", "configured/hub")
    github._gh_cmd(["api", "/repos/owner/hub/contents/catalog.json"], ctx)
    assert gh_calls[0][0] == ["gh", "api", "/repos/owner/hub/contents/catalog.json"]
    assert gh_calls[0][1]["cwd"] == str(ctx.repo_dir)
    assert gh_calls[0][1]["env"]["GH_REPO"] == "configured/hub"


def test_room_registry_failure_is_visible(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    from ouroboros.workspace_admission import room_chat_lens_dir

    def fail(*args):
        raise OSError("fixture registry failure")

    monkeypatch.setattr(projects_registry, "get_project", fail)
    directory, note = room_chat_lens_dir(tmp_path, "project-fixture")
    assert directory == ""
    assert "registry entry is unreadable" in note
    assert "OSError" in note


@pytest.mark.parametrize("token", ["GITHUB_TOKEN", "GH_TOKEN", "settings"])
def test_cli_discovery_accepts_the_execution_token_sources(tmp_path, monkeypatch, token):
    from ouroboros import config
    from ouroboros.tools.registry_guards import _builtin_tool_availability

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(config, "load_settings", lambda: {"GITHUB_TOKEN": "fixture"} if token == "settings" else {})
    if token != "settings":
        monkeypatch.setenv(token, "fixture")
    ctx, _ = _context(tmp_path, "queued")
    assert _builtin_tool_availability("get_github_issue", ctx)[0] is True


def test_cli_store_metadata_enables_only_cli_tools_without_probing(tmp_path, monkeypatch):
    from ouroboros.tools.registry_guards import _builtin_tool_availability

    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GH_CONFIG_DIR", str(tmp_path / "gh"))
    (tmp_path / "gh").mkdir()
    (tmp_path / "gh" / "hosts.yml").write_text("github.com:\n  user: fixture\n", encoding="utf-8")

    def no_probe(*args, **kwargs):
        raise AssertionError("discovery must not spawn an authentication probe")

    monkeypatch.setattr(subprocess, "run", no_probe)
    ctx, _ = _context(tmp_path, "queued")
    for name, _, _ in _CALLS:
        assert _builtin_tool_availability(name, ctx)[0] is True
    for name in ("run_ci_tests", "submit_skill_to_hub", "generate_evolution_stats"):
        assert _builtin_tool_availability(name, ctx) == (False, "missing_credential", "GITHUB_TOKEN")


@pytest.mark.parametrize("bound,explicit", [(False, False), (False, True), (True, True)])
def test_presence_repository_selection_keeps_host_argument_authority(tmp_path, gh_calls, bound, explicit):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload
    from ouroboros.presence_capabilities import PresenceArgumentBinding
    from ouroboros.tools.registry import ToolRegistry

    bindings = (PresenceArgumentBinding(("repo",), "static", static_value="owner/allowed"),) if bound else ()
    ceiling = PresenceCapabilityCeiling(
        skill_name="fixture", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main",
        inline_max_rounds=10, tool_grants=(PresenceToolGrant("get_github_issue", bindings),),
        resource_grants=(), digest="e" * 64,
    )
    ctx, _ = _context(tmp_path, "system")
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    args = {"number": 7, **({"repo": "owner/unselected"} if explicit else {})}

    result = registry.execute("get_github_issue", args)

    if explicit and not bound:
        assert "PRESENCE_ARGUMENT_BINDING_BLOCKED" in result
        assert not gh_calls
    else:
        assert "Issue #7" in result
        assert len(gh_calls) == 1
        if bound:
            assert gh_calls[0][0][-2:] == ["--repo", "owner/allowed"]
        else:
            assert "--repo" not in gh_calls[0][0]


def test_child_does_not_gain_github_access_from_explicit_repo(tmp_path, gh_calls):
    from ouroboros.tools.registry import ToolRegistry

    ctx, _ = _context(tmp_path, "queued")
    ctx.task_metadata = {"delegation_role": "subagent"}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute("get_github_issue", {"number": 7, "repo": "owner/unselected"})
    assert "BLOCKED" in result
    assert not gh_calls
