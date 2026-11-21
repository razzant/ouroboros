"""GitHub tools retain Project selection and explicit targets through every subcall."""

import hashlib
import io
import json
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from ouroboros.artifacts import actor_source_reference
from ouroboros.tools import github, github_read
from ouroboros.tools.registry import ToolContext


def _api_parts(argv):
    """``(route, hostname, typed_fields)`` of one ``gh api`` argv; ``_api_method`` reads its HTTP verb."""
    route, hostname, fields, rest = "", "", {}, list(argv[2:])
    while rest:
        item = rest.pop(0)
        if item in ("--method", "--hostname", "--input"):
            value = rest.pop(0)
            hostname = value if item == "--hostname" else hostname
        elif item in ("-f", "-F"):
            key, _sep, value = rest.pop(0).partition("=")
            fields[key] = value
        elif item != "--include":
            route = item
    return route, hostname, fields


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


SHA, MOVED = "a" * 40, "b" * 40
BASE = "https://github.example/owner/selected"
# `gh pr edit` with a token that lacks read:org (#2424): the CLI's own preparation query, not the edit, is refused.
READ_ORG_REFUSAL = ("GraphQL: Your token has not been granted the required scopes to execute this query. The 'login' "
                    "field requires one of the following scopes: ['read:org'], but your token has only been granted "
                    "the: ['repo'] scopes. Please modify your token's scopes at: https://github.com/settings/tokens.\n")


def _run(run_id, name="CI", status="completed", conclusion="success", **extra):
    return {"databaseId": run_id, "workflowName": name, "event": "pull_request", "status": status,
            "conclusion": conclusion, "attempt": 1, "url": f"{BASE}/actions/runs/{run_id}", "headSha": SHA, **extra}


def _job(job_id, name, status="completed", conclusion="success", steps=(), run_id=11):
    return {"databaseId": job_id, "name": name, "status": status, "conclusion": conclusion,
            "url": f"{BASE}/actions/runs/{run_id}/job/{job_id}",
            "steps": [{"number": number, "name": step, "status": step_status, "conclusion": step_conclusion}
                      for number, step, step_status, step_conclusion in steps]}


@pytest.fixture
def gh_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        output = ""
        if argv[1:3] == ["repo", "view"]:
            output = json.dumps({"nameWithOwner": "owner/selected", "url": BASE})
        elif argv[1] == "api":
            route = _api_parts(argv)[0].split("?", 1)[0]
            method = argv[argv.index("--method") + 1] if "--method" in argv else "GET"
            if route == "graphql":
                output = json.dumps({"data": {"nodes": [{"id": "PR_node7", "commits": {"totalCount": 1}, "reviewDecision": "APPROVED"}]}})
            elif route.endswith(("/issues/7", "/pulls/7")):
                output = json.dumps({"number": 7, "node_id": "PR_node7", "title": "Fixture", "state": "closed" if method == "PATCH" else "open",
                                     "user": {"login": "fixture"}, "head": {"sha": SHA}, "html_url": f"{BASE}/pull/7", "body": ""})
            elif method == "POST" and route.endswith("/issues"):
                output = json.dumps({"number": 7, "title": "Title", "body": "Body", "labels": [{"name": "bug"}],
                                     "user": {"login": "fixture"}, "html_url": f"{BASE}/issues/7"})
            elif route.endswith("/comments"):
                output = json.dumps({"id": 1, "body": "text", "user": {"login": "fixture"}, "created_at": "t",
                                     "html_url": f"{BASE}/issues/7#issuecomment-1", "issue_url": f"{BASE}/issues/7"})
            elif route.endswith("/reviews"):
                output = json.dumps({"id": 1, "body": "text", "state": "COMMENTED", "user": {"login": "fixture"},
                                     "submitted_at": "t", "html_url": f"{BASE}/pull/7#pullrequestreview-1"})
            else:
                output = "[]"
        elif argv[1:3] == ["pr", "view"]:
            output = json.dumps({"number": 7, "title": "Fixture", "state": "OPEN", "author": {"login": "fixture"},
                                 "headRefOid": SHA, "url": f"{BASE}/pull/7"})
        elif argv[1:3] == ["pr", "diff"]:
            output = "diff --git a/x b/x"
        elif argv[1:3] == ["run", "list"]:
            output = json.dumps([_run(11, conclusion="failure")])
        elif argv[1:3] == ["run", "view"]:
            output = json.dumps({"jobs": [_job(55, "test", conclusion="failure")]})
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


_CALLS = [
    ("github_read", {"method": "list", "kind": "issue"}, 1), ("github_read", {"method": "get", "kind": "issue", "target": 7}, 1),
    ("github_write", {"method": "comment", "target": 7, "body": "text"}, 1),
    ("github_write", {"method": "update", "target": 7, "state": "closed"}, 1),
    ("github_write", {"method": "create_issue", "title": "Title", "body": "Body", "labels": ["bug"]}, 1),
    ("github_read", {"method": "list", "kind": "pr"}, 1),
    ("github_read", {"method": "get", "kind": "pr", "target": 7, "include": ["diff"]}, 3),
    ("github_write", {"method": "review", "target": 7, "event": "COMMENT", "body": "text"}, 1),
    ("github_read", {"method": "checks", "target": 7}, 4),
]


_NEW_CALLS = [
    ("github_read", {"method": "get", "kind": "run", "target": 11, "include": ["log"]}, ["run view", "run view"]),
    ("github_read", {"method": "list", "kind": "artifact", "target": 11, "limit": 100}, ["run view", "api"]),
    ("github_read", {"method": "download", "kind": "artifact", "target": 23, "max_bytes": 4096}, ["repo view", "api", "api"]),
    ("github_write", {"method": "create_pr", "title": "New title", "head": "feature", "base": "main",
                      "expect": {"head_sha": SHA, "base_sha": MOVED}, "body": "New body"},
     ["repo view", "api", "api", "pr list", "pr create", "pr view"]),
    ("github_write", {"method": "update", "kind": "pr", "target": 7, "expect": {"head_sha": SHA, "base_sha": MOVED},
                      "title": "New title", "body": "New body", "base": "release"},
     ["pr view", "api", "api", "pr view"]),
    ("github_write", {"method": "update", "kind": "pr", "target": 7, "draft": False, "expect": {"head_sha": SHA}},
     ["pr view", "pr ready", "pr view"]),
]


@pytest.fixture
def complete_gh(monkeypatch):
    """Fake the CLI at subprocess and its byte-capture boundary, including PR read-back."""
    from ouroboros.tools import github_workflows

    calls = []
    pr = {"number": 7, "url": f"{BASE}/pull/7", "state": "OPEN", "title": "Old title",
          "body": "Old body", "isDraft": True, "headRefName": "feature", "baseRefName": "main",
          "headRefOid": SHA, "headRepositoryOwner": {"login": "owner"}}
    archive_file = io.BytesIO()
    with zipfile.ZipFile(archive_file, "w") as archive:
        archive.writestr("results.txt", "passed\n")
    archive_bytes = archive_file.getvalue()
    artifact = {"id": 23, "name": "test-results", "size_in_bytes": len(archive_bytes), "expired": False,
                "created_at": "2026-01-01T00:00:00Z", "expires_at": "2026-02-01T00:00:00Z",
                "digest": f"sha256:{hashlib.sha256(archive_bytes).hexdigest()}",
                "workflow_run": {"id": 11, "head_sha": SHA}}
    monkeypatch.setattr(github, "github_token_from_env_or_settings", lambda: "fixture-token")
    # The ref ``_retain`` returns, priced without writing bytes: these cases judge the gh target, not retention.
    monkeypatch.setattr(github_workflows, "_retain", lambda _ctx, **kw: actor_source_reference(category="tool_results", **kw))

    def record(argv, kwargs):
        calls.append((list(argv), kwargs))
        args = argv[1:]
        if args[:2] == ["repo", "view"] and len(args) > 2 and args[2] not in ("--json",):
            args = args[:2] + args[3:]  # gh repo view takes its explicit repository positionally.
        if args[:2] == ["repo", "view"]:
            return {"nameWithOwner": "owner/selected", "url": BASE}
        if args[:2] == ["run", "view"] and "--json" in args:
            return _run(11, jobs=[_job(55, "test")])
        if args[:2] == ["pr", "list"]:
            return []
        if args[:2] == ["pr", "view"]:
            return dict(pr)
        if args[:2] == ["pr", "create"]:
            assert {"--title=New title", "--head=feature", "--base=main", "--body-file=-"} <= set(args)
            assert kwargs["input"] == "New body"
            pr.update(title="New title", body="New body", isDraft=False)
            return pr["url"]
        if args[:2] == ["pr", "edit"]:  # Its preparation queries organization fields a repo-scoped token cannot read.
            return SimpleNamespace(returncode=1, stdout="", stderr=READ_ORG_REFUSAL)
        if args[:2] == ["pr", "ready"]:
            assert args[2] == "7" and "--undo" not in args
            pr["isDraft"] = False
            return ""
        if args[:1] == ["api"]:
            path = args[1]
            if path in ("repos/owner/selected/git/ref/heads/feature",):
                return {"object": {"sha": SHA}}
            if path in ("repos/owner/selected/git/ref/heads/main",
                        "repos/owner/selected/git/ref/heads/release"):
                return {"object": {"sha": MOVED}}
            if path == "repos/owner/selected/actions/runs/11/artifacts?per_page=100&page=1":
                return {"total_count": 1, "artifacts": [artifact]}
            if path == "repos/owner/selected/actions/artifacts/23":
                return artifact
            if path == "repos/owner/selected/pulls/7" and args[args.index("--method") + 1] == "PATCH":
                fields = json.loads(kwargs["input"])
                assert fields == {"title": "New title", "body": "New body", "base": "release"}
                pr.update(title=fields["title"], body=fields["body"], baseRefName=fields["base"])
                return {"number": 7, "html_url": pr["url"], "title": pr["title"], "body": pr["body"],
                        "base": {"ref": pr["baseRefName"]}, "head": {"sha": pr["headRefOid"]}}
        raise AssertionError(f"unexpected gh command: {argv!r}")

    def run(argv, **kwargs):
        value = record(argv, kwargs)
        if isinstance(value, SimpleNamespace):
            return value
        return SimpleNamespace(returncode=0, stdout=value if isinstance(value, str) else json.dumps(value), stderr="")

    def capture(argv, cwd, env, timeout, path, max_bytes):
        calls.append((list(argv), {"cwd": str(cwd), "env": env, "capture": True, "max_bytes": max_bytes}))
        if argv[1:3] == ["run", "view"] and "--log" in argv:
            data = b"job\tstep\tpassed\n"
        elif argv[1:3] == ["api", "repos/owner/selected/actions/artifacts/23/zip"]:
            data = archive_bytes
        else:
            raise AssertionError(f"unexpected gh capture: {argv!r}")
        assert len(data) <= max_bytes
        path.write_bytes(data)
        return github.GhCapture(path, len(data), hashlib.sha256(data).hexdigest(), False), 0, "", False

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(github, "_capture_stdout", capture)
    return calls


@pytest.mark.parametrize("kind", ["system", "queued", "room"])
@pytest.mark.parametrize("name,args,steps", _NEW_CALLS)
@pytest.mark.parametrize("repo", ["", "github.example/owner/selected"])
def test_new_repository_tools_keep_target_through_read_and_readback(
        tmp_path, complete_gh, monkeypatch, kind, name, args, steps, repo):
    ctx, expected_cwd = _context(tmp_path, kind)
    monkeypatch.setenv("GH_REPO", "unrelated/wrong-repo")
    monkeypatch.setenv("GH_HOST", "configured.example")
    entry = next(item for item in github.get_tools() if item.name == name)

    result = entry.handler(ctx, **args, repo=repo)

    assert not result.startswith("⚠️"), result
    assert "repo" in entry.schema["parameters"]["properties"]
    assert len(complete_gh) == len(steps)
    assert [argv[1] if argv[1] == "api" else " ".join(argv[1:3]) for argv, _ in complete_gh] == steps
    operation = "draft" if "draft" in args else args["method"]
    result = json.loads(result.partition("\n")[2])
    result = result.get("view", "") + result.get("detail", "")
    api_paths = {
        "list": ["repos/owner/selected/actions/runs/11/artifacts?per_page=100&page=1"],
        "download": ["repos/owner/selected/actions/artifacts/23",
                                     "repos/owner/selected/actions/artifacts/23/zip"],
        "create_pr": ["repos/owner/selected/git/ref/heads/feature",
                             "repos/owner/selected/git/ref/heads/main"],
        "update": ["repos/owner/selected/git/ref/heads/release", "repos/owner/selected/pulls/7"],
    }
    assert [argv[2] for argv, _ in complete_gh if argv[1] == "api"] == api_paths.get(operation, [])
    # A read naming its repository runs every leg, the generic API ones included, in the task's own task_drive.
    scratch = ctx.task_drive_root() if name == "github_read" and repo else None
    for argv, kwargs in complete_gh:
        assert kwargs["env"]["GH_HOST"] == "configured.example"
        if argv[1] == "api":
            assert argv[-2:] == ["--hostname", "github.example"]
            assert kwargs["cwd"] == str(scratch or ctx.repo_dir)
            assert kwargs["env"]["GH_REPO"] == "unrelated/wrong-repo"
            continue
        assert kwargs["cwd"] == str(scratch or expected_cwd)
        if repo and argv[1:3] == ["repo", "view"]:
            assert argv[3] == repo and "--repo" not in argv
        elif repo:
            assert argv[-2:] == ["--repo", repo]
        else:
            assert "--repo" not in argv
        if kind != "system":
            assert "GH_REPO" not in kwargs["env"]
        else:
            assert kwargs["env"]["GH_REPO"] == "unrelated/wrong-repo"
    if operation == "get":
        assert "job\tstep\tpassed" in result and "captured 16 bytes exactly" in result
        assert complete_gh[-1][1]["capture"] is True
        assert complete_gh[-1][0][4:7] == ["--log", "--attempt", "1"]
    elif operation == "list":
        assert "id 23: test-results" in result and "Complete captured JSON page" in result
    elif operation == "download":
        assert "results.txt (7 bytes)" in result and "Retained as the archive" in result
        assert complete_gh[-1][1]["capture"] is True
    else:
        assert "Read-back:" in result and "confirmed" in result
        # One mutation; the field edit is a REST PATCH at the read PR's own host, never `gh pr edit`.
        assert sum(argv[1:3] in (["pr", "create"], ["pr", "ready"]) or "PATCH" in argv
                   for argv, _ in complete_gh) == 1
        assert not any(argv[1:3] == ["pr", "edit"] for argv, _ in complete_gh)


@pytest.mark.parametrize("kind", ["system", "queued", "room", "fileless", "broken"])
@pytest.mark.parametrize("name,args,_steps", _NEW_CALLS)
def test_new_repository_tools_reject_null_target_without_gh(tmp_path, complete_gh, monkeypatch, kind, name, args, _steps):
    from ouroboros.tools.registry import ToolRegistry

    ctx, _ = _context(tmp_path, "room" if kind in {"fileless", "broken"} else kind)
    if kind == "fileless":
        ctx.task_metadata = {}
    elif kind == "broken":
        ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    monkeypatch.setenv("GH_REPO", "unrelated/wrong-repo")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)

    result = registry.execute_result(name, {**args, "repo": None})

    assert "GH_TARGET_INVALID" in result.text
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert not complete_gh


@pytest.mark.parametrize("repo", [False, 0, [], {}])
@pytest.mark.parametrize("name,args,_steps", _NEW_CALLS)
def test_new_repository_tools_reject_invalid_target_without_gh(tmp_path, complete_gh, name, args, _steps, repo):
    ctx, _ = _context(tmp_path, "queued")
    entry = next(item for item in github.get_tools() if item.name == name)

    assert "GH_TARGET_INVALID" in entry.handler(ctx, **args, repo=repo)
    assert not complete_gh


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
    # An omitted repo names its one target before the first REST or GraphQL leg; the checks reader has none.
    assert len(gh_calls) == count + (not repo and args["method"] != "checks")
    assert "repo" in entry.schema["parameters"]["properties"]
    # A read naming its repository needs no checkout: every leg, the ones that take no repo too, runs in its task_drive.
    scratch = ctx.task_drive_root() if name == "github_read" and repo else None
    for argv, kwargs in gh_calls:
        if argv[1:3] == ["repo", "view"]:
            assert argv[-2:] == ["--json", "nameWithOwner,url"]
            # An ambient GH_REPO still selects the repository where no Project replaced it, as it does for gh itself.
            assert argv[3:-2] == (["unrelated/wrong-repo"] if kind == "system" else [])
        elif argv[1] == "api":
            route, hostname, fields = _api_parts(argv)
            if route == "graphql":  # Node facts follow the PR URL returned by GitHub, like list(kind='pr').
                assert argv == ["gh", "api", "graphql", "--hostname", "github.example", "--input", "-"]
                assert kwargs["cwd"] == str(scratch or ctx.repo_dir)
                assert json.loads(kwargs["input"])["variables"] == {"ids": ["PR_node7"]}
                continue
            if route.startswith("repos/owner/selected/check-runs/"):  # The checks reader's one literal path: the URL GitHub returned.
                assert argv == ["gh", "api", "repos/owner/selected/check-runs/55/annotations?per_page=100",
                                "--hostname", "github.example"]
                assert kwargs["cwd"] == str(scratch or ctx.repo_dir)
                continue
            assert "--repo" not in argv  # gh api has no --repo: the route or the typed fields name the repository.
            assert route.startswith("repos/owner/selected/") and hostname == "github.example" and "{owner}" not in route
        elif repo:
            assert argv[-2:] == ["--repo", repo]
        else:
            assert "--repo" not in argv
        assert kwargs["cwd"] == str(scratch or expected_cwd)
        if kind != "system":
            assert "GH_REPO" not in kwargs["env"]
        assert kwargs["env"]["GH_HOST"] == "configured.example"


@pytest.mark.parametrize("failure", ["note", "fileless", "invalid-workspace-mode"])
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

    result = github_read.run_read(ctx, method="list", kind="issue")

    assert "GH_TARGET_" in result
    assert gh_calls == []


def test_fileless_room_accepts_explicit_repo(tmp_path, gh_calls):
    ctx, _ = _context(tmp_path, "room")
    ctx.task_metadata = {}
    assert not github_read.run_read(ctx, method="get", kind="issue", target=7, repo="owner/selected").startswith("⚠️")
    assert gh_calls[0][0] == ["gh", "api", "repos/owner/selected/issues/7"]


def test_generic_hub_transport_keeps_explicit_api_contract(tmp_path, gh_calls, monkeypatch):
    ctx, _ = _context(tmp_path, "room")
    ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    monkeypatch.setenv("GH_REPO", "configured/hub")
    github._gh_cmd(["api", "/repos/owner/hub/contents/catalog.json"], ctx)
    assert gh_calls[0][0] == ["gh", "api", "/repos/owner/hub/contents/catalog.json"]
    assert gh_calls[0][1]["cwd"] == str(ctx.repo_dir)
    assert gh_calls[0][1]["env"]["GH_REPO"] == "configured/hub"


@pytest.mark.parametrize("kind", ["system", "queued", "room", "fileless", "broken"])
@pytest.mark.parametrize("name,args,_count", _CALLS)
def test_public_null_target_never_selects_internal_hub_transport(tmp_path, gh_calls, monkeypatch, kind, name, args, _count):
    from ouroboros.tools.registry import ToolRegistry

    ctx, _ = _context(tmp_path, "room" if kind in {"fileless", "broken"} else kind)
    if kind == "fileless":
        ctx.task_metadata = {}
    elif kind == "broken":
        ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    monkeypatch.setenv("GH_REPO", "unrelated/wrong-repo")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    result = registry.execute_result(name, {**args, "repo": None})
    assert "GH_TARGET_INVALID" in result.text
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert not gh_calls


@pytest.mark.parametrize("name,args,_count", _CALLS)
def test_required_target_is_an_error_then_a_valid_call_has_no_stale_failure(tmp_path, gh_calls, monkeypatch, name, args, _count):
    from ouroboros.loop_tool_execution import _typed_execution_failure
    from ouroboros.tools.registry import ToolRegistry

    ctx, _ = _context(tmp_path, "room")
    ctx.task_metadata = {}
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    refused = registry.execute_result(name, args)
    assert "GH_TARGET_REQUIRED" in refused.text
    assert (refused.status, refused.code) == ("error", "TOOL_ARG_ERROR")
    assert _typed_execution_failure(True, refused)
    assert not gh_calls
    valid = registry.execute_result(name, {**args, "repo": "owner/selected"})
    assert valid.status == "ok" and not _typed_execution_failure(True, valid)
    assert len(gh_calls) == _count


@pytest.mark.parametrize("repo", [False, 0, 1])
def test_invalid_public_target_values_cannot_be_generic_transport(tmp_path, gh_calls, repo):
    ctx, _ = _context(tmp_path, "queued")
    assert "GH_TARGET_INVALID" in github_read.run_read(ctx, method="get", kind="issue", target=7, repo=repo)
    assert not gh_calls


def test_room_registry_failure_is_visible(tmp_path, monkeypatch):
    from ouroboros import projects_registry
    from ouroboros.workspace_admission import room_chat_lens_dir

    def fail(*args, **kwargs):
        raise OSError("fixture registry failure")

    monkeypatch.setattr(projects_registry, "get_reserved_project", fail)
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
    assert _builtin_tool_availability("github_read", ctx)[0] is True


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
    assert _builtin_tool_availability("submit_skill_to_hub", ctx) == (False, "missing_credential", "GITHUB_TOKEN")
    assert _builtin_tool_availability("read_file", ctx) == (True, "", "")


@pytest.mark.parametrize("bound,explicit", [(False, False), (False, True), (True, True)])
def test_presence_repository_selection_keeps_host_argument_authority(tmp_path, gh_calls, bound, explicit):
    from ouroboros.presence_authority import PresenceCapabilityCeiling, PresenceToolGrant, presence_ceiling_payload
    from tests.frozen_presence_capabilities import PresenceArgumentBinding
    from ouroboros.tools.registry import ToolRegistry

    bindings = (PresenceArgumentBinding(("repo",), "static", static_value="owner/allowed"),) if bound else ()
    ceiling = PresenceCapabilityCeiling(
        skill_name="fixture", skill_content_hash="a" * 64, profile_fingerprint="b" * 64,
        state_fingerprint="c" * 64, selection_fingerprint="d" * 64, model_slot="main",
        inline_max_rounds=10, tool_grants=(PresenceToolGrant("github_read", bindings),),
        resource_grants=(), digest="e" * 64,
    )
    ctx, _ = _context(tmp_path, "system")
    ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(ceiling)}
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    args = {"method": "get", "kind": "issue", "target": 7, **({"repo": "owner/unselected"} if explicit else {})}

    result = registry.execute("github_read", args)

    if explicit and not bound:
        assert "PRESENCE_ARGUMENT_BINDING_BLOCKED" in result
        assert not gh_calls
    else:
        assert result.startswith("github_read get issue #7")
        assert len(gh_calls) == (1 if bound else 2)
        assert gh_calls[-1][0][2] == ("repos/owner/allowed/issues/7" if bound else "repos/owner/selected/issues/7")
        # The host-bound repo is a named repository too: that read runs in the task's own task_drive.
        assert gh_calls[-1][1]["cwd"] == str(ctx.task_drive_root() if bound else ctx.repo_dir)


def _role_context(tmp_path, kind, role):
    """A Project with its folder, or one without (``folderless``); ``child`` is a delegated read-only subagent."""
    ctx, cwd = _context(tmp_path, "queued" if kind == "project" else "system")
    if kind == "folderless":
        ctx.project_id = "project-fixture"
        cwd = ctx.task_drive_root()
    if role == "child":
        ctx.task_metadata = {"delegation_role": "subagent"}
    return ctx, cwd


def _registry(ctx):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return registry


_CHILD_READS = [  # Representative reads a read-only child is offered, on both fake CLIs.
    ("gh_calls", {"method": "get", "kind": "issue", "target": 7}),
    ("gh_calls", {"method": "get", "kind": "pr", "target": 7, "include": ["diff"]}),
    ("gh_calls", {"method": "api", "route": "repos/{owner}/{repo}/issues/7"}),
    ("gh_calls", {"method": "checks", "target": 7}),
    ("complete_gh", {"method": "get", "kind": "run", "target": 11, "include": ["log"]}),
    ("complete_gh", {"method": "download", "kind": "artifact", "target": 23, "max_bytes": 4096}),
]


@pytest.mark.parametrize("kind", ["project", "folderless"])
@pytest.mark.parametrize("fixture,args", _CHILD_READS)
@pytest.mark.parametrize("repo", ["", "github.example/owner/selected"])
def test_read_only_child_reads_github_where_a_writer_would(tmp_path, request, monkeypatch, kind, fixture, args, repo):
    """github_read is offered to read-only children: it binds gh's directory with read authority, so the child
    launches exactly what an ordinary writer in the same context launches, in the same directory."""
    calls = request.getfixturevalue(fixture)
    monkeypatch.setenv("GH_REPO", "unrelated/wrong-repo")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    observed = {}
    for role in ("writer", "child"):
        ctx, cwd = _role_context(tmp_path, kind, role)
        calls.clear()
        result = _registry(ctx).execute_result("github_read", {**args, "repo": repo})
        observed[role] = (result.status, result.code, [(argv, kw["cwd"], "GH_REPO" in kw["env"]) for argv, kw in calls])
    assert observed["child"] == observed["writer"]
    status, code, launched = observed["child"]
    if kind == "folderless" and not repo:
        assert (status, code, launched) == ("error", "TOOL_ARG_ERROR", [])
        return
    assert status == "ok" and launched
    # The repository subcommands run in the selected folder, or for a read naming its repository in the task's own
    # task_drive, without the ambient GH_REPO default.
    folder = ctx.task_drive_root() if repo else cwd
    assert all((selected, ambient) == (str(folder), False) for argv, selected, ambient in launched if argv[1] != "api")


def test_read_only_child_keeps_every_other_github_gate(tmp_path, gh_calls, monkeypatch):
    """Read authority for gh's directory widens nothing else: writes, merge and the shell stay withheld, and the
    network and disabled-method gates still refuse before any gh launch."""
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    ctx, _ = _role_context(tmp_path, "project", "child")
    registry = _registry(ctx)
    for name, args in (("github_write", {"method": "comment", "target": 7, "body": "text"}),
                       ("pr_merge", {"number": 7, "expected_head_sha": SHA, "method": "merge"}),
                       ("run_command", {"cmd": ["gh", "issue", "list"]})):
        assert registry.execute_result(name, args).status == "blocked", name
    read = {"method": "get", "kind": "issue", "target": 7}
    for contract, fragment in (({"allowed_resources": {"network": False}}, "network"),
                               ({"disabled_tools": ["get_github_issue"]}, "get_github_issue")):
        ctx.task_contract = contract
        blocked = registry.execute_result("github_read", read)
        assert blocked.status == "blocked" and fragment in blocked.text, blocked.text
    assert not gh_calls
    ctx.task_contract = {}
    assert registry.execute_result("github_read", read).status == "ok" and len(gh_calls) == 2


def test_read_only_child_with_an_unavailable_project_folder_sends_nothing(tmp_path, gh_calls):
    ctx, project = _role_context(tmp_path, "project", "child")
    project.rmdir()
    result = _registry(ctx).execute_result("github_read", {"method": "list", "kind": "issue"})
    assert result.text.startswith("⚠️ GH_TARGET_UNAVAILABLE:")
    assert (result.status, result.meta.get("operation_outcome")) == ("unavailable", "completed_no_effect")
    assert not gh_calls


def test_an_observe_wake_reads_a_named_repository_without_its_project_folder(tmp_path, gh_calls, monkeypatch):
    """#3613 under Observe: with the Project folder moved away, a github_read naming its repository passes the registry
    and runs every leg in the task's own task_drive, while the GitHub actions Observe withholds stay withheld."""
    from ouroboros.consciousness_authority import apply_consciousness_authority

    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_k: (True, ""))
    ctx, project = _context(tmp_path, "queued")
    project.rmdir()
    ctx.task_metadata = apply_consciousness_authority({"id": "wake", "metadata": {
        "initiator": "consciousness", "consciousness_autonomy": "observe"}})["metadata"]
    registry = _registry(ctx)
    for name, args in (("github_write", {"method": "comment", "target": 7, "body": "text"}),
                       ("pr_merge", {"number": 7, "expected_head_sha": SHA, "method": "merge"})):
        assert registry.execute_result(name, {**args, "repo": "owner/selected"}).status == "blocked", name
    assert not gh_calls

    read = registry.execute_result("github_read", {"method": "get", "kind": "pr", "target": 7, "include": ["diff"],
                                                   "repo": "owner/selected"})

    assert read.status == "ok", read.text
    assert [argv[1] if argv[1] == "api" else " ".join(argv[1:3]) for argv, _ in gh_calls] == ["api", "pr diff", "api"]
    assert {kw["cwd"] for _argv, kw in gh_calls} == {str(ctx.task_drive_root())} and not project.exists()


@pytest.mark.parametrize("state", ["healthy", "moved-room", "unreadable-registry", "moved-workspace", "folderless"])
def test_a_read_naming_its_repository_needs_no_project_folder(tmp_path, gh_calls, state):
    """#3613: the Project folder moved outside Ouroboros. A read naming its repository runs every leg, the count read
    GitHub's PR URL names included, in the task's own task_drive whatever that folder's state, and so does a GraphQL
    query whose stdin variables carry a literal {branch}. A route with either branch placeholder gh fills from the
    folder still needs it: it runs there, or is refused, typed, before gh starts. An omitted repo still means the
    folder. Nothing falls back to the system repository or recreates the moved folder."""
    ctx, project = _context(tmp_path, "queued" if state in ("healthy", "moved-workspace") else "room")
    if state in ("moved-room", "moved-workspace"):
        project.rmdir()
    if state == "moved-room":  # What room_chat_lens_dir records for a selected folder that no longer exists.
        ctx.task_metadata["_project_room_note"] = f"project 'project-fixture' working_dir {project} is unusable"
    elif state == "unreadable-registry":
        ctx.task_metadata = {"_project_room_note": "registry unavailable"}
    elif state == "folderless":
        ctx.task_metadata = {}
    read = next(entry.handler for entry in github.get_tools() if entry.name == "github_read")
    scratch = str(ctx.task_drive_root())

    named = read(ctx, method="get", kind="pr", target=7, include=["diff"], repo="owner/selected")

    assert named.startswith("github_read get pr #7"), named
    assert [argv[1] if argv[1] == "api" else " ".join(argv[1:3]) for argv, _ in gh_calls] == ["api", "pr diff", "api"]
    assert {kw["cwd"] for _argv, kw in gh_calls} == {scratch}
    gh_calls.clear()
    # A {branch} in the GraphQL stdin body is data, not an endpoint placeholder: the query needs no checkout either.
    query = 'query($head:String!){repository(owner:"owner",name:"selected"){ref(qualifiedName:$head){name}}}'
    graphql = read(ctx, method="api", graphql=query, variables={"head": "{branch}"}, repo="owner/selected")
    assert not graphql.startswith("⚠️"), graphql
    [(argv, kw)] = gh_calls
    assert (argv, kw["cwd"]) == (["gh", "api", "graphql", "--input", "-"], scratch)
    assert json.loads(kw["input"])["variables"] == {"head": "{branch}"}
    gh_calls.clear()
    for form in ("{branch}", ":branch"):  # Both branch forms of gh's endpoint pattern are filled from the folder.
        ctx._active_builtin_tool_result = None
        branch = read(ctx, method="api", route="repos/{owner}/{repo}/compare/main..." + form, repo="owner/selected")
        if state == "healthy":  # The folder exists: gh reads the branch there.
            routes = [(argv[3], kw["cwd"]) for argv, kw in gh_calls]
            assert routes == [("repos/owner/selected/compare/main..." + form, str(project))], branch
        else:
            refused = ctx._active_builtin_tool_result
            assert branch.startswith("⚠️ GH_TARGET_UNAVAILABLE:") and not gh_calls, branch
            assert (refused.status, refused.meta.get("operation_outcome")) == ("unavailable", "completed_no_effect")
        gh_calls.clear()
    implicit = read(ctx, method="get", kind="issue", target=7)
    assert project.exists() is (state not in ("moved-room", "moved-workspace"))
    if state == "healthy":  # The omitted repo names the folder's repository, read in the folder.
        assert not implicit.startswith("⚠️") and {kw["cwd"] for _argv, kw in gh_calls} == {str(project)}, implicit
    else:
        assert implicit.startswith("⚠️ GH_TARGET_") and not gh_calls, implicit


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("intent", ["system_repo", "", "explicit_none"])
def test_a_branch_route_runs_in_the_kept_body_checkout_never_a_folderless_scratch(tmp_path, gh_calls, direct, intent):
    """#3613: readmission that keeps a Project's body work records resource_intent system_repo and no room lens
    (``project_placement.readmit_placement``), so the resolver selects the body checkout and gh fills a named
    repository's {branch} there. The same Project with no folder selects only its own scratch, which has no branch:
    refused, typed, before gh starts. Neither outcome depends on the room or chat fields."""
    ctx, body = _context(tmp_path, "system")
    ctx.is_direct_chat, ctx.project_id = direct, "project-fixture"
    ctx.task_metadata = {"resource_intent": {"kind": intent}} if intent else {}
    read = next(entry.handler for entry in github.get_tools() if entry.name == "github_read")
    for form in ("{branch}", ":branch"):
        ctx._active_builtin_tool_result = None
        result = read(ctx, method="api", route="repos/{owner}/{repo}/compare/main..." + form, repo="owner/selected")
        if intent == "system_repo":
            routes = [(argv[3], kw["cwd"]) for argv, kw in gh_calls]
            assert routes == [("repos/owner/selected/compare/main..." + form, str(body))], result
        else:
            refused = ctx._active_builtin_tool_result
            assert result.startswith("⚠️ GH_TARGET_UNAVAILABLE:") and not gh_calls, result
            assert (refused.status, refused.meta.get("operation_outcome")) == ("unavailable", "completed_no_effect")
        gh_calls.clear()
