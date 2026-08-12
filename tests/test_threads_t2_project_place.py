"""Phase T2 — a project always has a PLACE, and git is offered rather than forced.

A11: attaching an existing folder, cloning, or auto-provisioning are all legitimate
ways to give a project its working folder. A12: an attached folder that is not under
git STAYS not under git until the owner says yes — auto-``git init`` in someone
else's folder is forbidden, so admission stops before the first FILE task with the
typed ``git_init_required`` decision instead of either refusing the folder outright
(what it used to do) or quietly initialising it.

Sibling coverage: the entry-point admissions live in
``tests/test_v6590_projects_entry.py`` and the admission SSOT itself in
``tests/test_v6580_projects_foundation.py``.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import types

from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient


def _init_git_repo(path: pathlib.Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=str(path), check=True)
    (path / "README.md").write_text("x\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=str(path), check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@local", "commit", "-qm", "init"],
        cwd=str(path), check=True,
    )


class _ProjectsReq:
    """The minimal request shape the gateway projects handlers read."""

    def __init__(self, body, *, drive_root, repo_dir, path_params=None):
        self._body = body
        self.path_params = dict(path_params or {})
        self.app = types.SimpleNamespace(
            state=types.SimpleNamespace(drive_root=drive_root, repo_dir=repo_dir)
        )

    async def json(self):
        return self._body


# --- containment: a place is never carved out of somebody else's repository --------

def _repo_with_subdir(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    owner_repo = tmp_path / "owner_repo"
    _init_git_repo(owner_repo)
    subdir = owner_repo / "packages" / "web"
    subdir.mkdir(parents=True)
    (subdir / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    return owner_repo, subdir


def test_attach_refuses_a_folder_nested_in_another_repository(tmp_path):
    """Dropping the git REQUIREMENT must not drop CONTAINMENT with it: they are
    different rules. "Not a git repo" is fine; "a subdirectory of somebody's git
    repo" is not, because the only way to track a place there is to nest a second
    repository inside theirs. The refusal names the enclosing root, so the owner is
    told what to attach rather than that their folder is bad."""
    from ouroboros.project_sources import enclosing_git_worktree, validate_attach_path

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    owner_repo, subdir = _repo_with_subdir(tmp_path)

    resolved, error = validate_attach_path(str(subdir), system_repo_dir=repo, drive_root=data)
    assert resolved is None
    assert str(owner_repo.resolve()) in error and "attach that root instead" in error

    # The two shapes A11/A12 exist to admit still pass: a plain folder (nothing
    # encloses it) and a worktree ROOT (git's toplevel IS the folder).
    plain = tmp_path / "plain"
    plain.mkdir()
    assert validate_attach_path(str(plain), system_repo_dir=repo, drive_root=data) == (plain.resolve(), "")
    assert validate_attach_path(str(owner_repo), system_repo_dir=repo, drive_root=data) == (
        owner_repo.resolve(), "",
    )
    assert enclosing_git_worktree(owner_repo) == ""
    assert enclosing_git_worktree(plain) == ""


def test_containment_survives_a_git_that_refuses_to_answer(tmp_path, monkeypatch):
    """The guard must not FAIL OPEN. It used to read `git rev-parse` and treat every
    non-zero exit as "nothing encloses this" — but git exits non-zero for reasons
    that have nothing to do with containment: `safe.directory` refusing a
    foreign-owned repo, an older git refusing unknown `extensions.*` (a sha256 or
    reftable repo), a hostile `[include]` stalling past the timeout. Each one turned
    UNKNOWN into ADMIT and let through exactly the nested shape the guard exists to
    refuse. The answer now comes off the filesystem, which always has one."""
    from ouroboros.project_sources import enclosing_git_worktree, validate_attach_path

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    owner_repo, subdir = _repo_with_subdir(tmp_path)

    # A `git` that answers rc=128 the way the dubious-ownership refusal does, and
    # one that never answers at all.
    shim = tmp_path / "shim"
    shim.mkdir()
    for name, body in (
        ("git", "#!/bin/sh\necho 'fatal: detected dubious ownership' >&2\nexit 128\n"),
    ):
        (shim / name).write_text(body, encoding="utf-8")
        (shim / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")

    assert enclosing_git_worktree(subdir) == str(owner_repo.resolve())
    _, error = validate_attach_path(str(subdir), system_repo_dir=repo, drive_root=data)
    assert str(owner_repo.resolve()) in error
    # A11/A12 still hold with the same broken git: a plain folder encloses nothing.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert validate_attach_path(str(plain), system_repo_dir=repo, drive_root=data) == (
        plain.resolve(), "",
    )


def test_containment_ignores_inherited_git_location_env(tmp_path, monkeypatch):
    """The probes ran with the ambient environment, so whatever launched Ouroboros
    decided which repository git thought it was standing in. `GIT_DIR`/`GIT_WORK_TREE`
    made a PLAIN folder report someone else's toplevel and get REFUSED — an A11/A12
    violation — and `GIT_CEILING_DIRECTORIES` stopped the upward search so a real
    repo subdirectory reported nothing and was ADMITTED. Both directions are wrong,
    and neither is a fact about the folder."""
    from ouroboros.project_sources import validate_attach_path

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    owner_repo, subdir = _repo_with_subdir(tmp_path)
    plain = tmp_path / "plain"
    plain.mkdir()

    monkeypatch.setenv("GIT_DIR", str(owner_repo / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(owner_repo))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(owner_repo))

    assert validate_attach_path(str(plain), system_repo_dir=repo, drive_root=data) == (
        plain.resolve(), "",
    )
    _, error = validate_attach_path(str(subdir), system_repo_dir=repo, drive_root=data)
    assert str(owner_repo.resolve()) in error


def test_git_storage_and_submodules_are_not_project_folders(tmp_path):
    """Two shapes that slipped the guard. A bare repository (and any interior of
    one) is git's STORAGE, not a folder to work in, and `.git` has no toplevel of
    its own to report. A submodule working directory is a repository by git's
    reckoning — `--show-toplevel` returns the submodule itself — so it answered
    "nothing encloses me" while sitting squarely inside the superproject's tree."""
    from ouroboros.project_sources import validate_attach_path

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    bare = tmp_path / "bare.git"
    subprocess.run(["git", "init", "-q", "--bare", str(bare)], check=True)
    _, err_bare = validate_attach_path(str(bare), system_repo_dir=repo, drive_root=data)
    assert "internal storage" in err_bare
    _, err_interior = validate_attach_path(
        str(bare / "refs" / "heads"), system_repo_dir=repo, drive_root=data
    )
    assert str(bare.resolve()) in err_interior

    superproject = tmp_path / "superproject"
    _init_git_repo(superproject)
    donor = tmp_path / "donor"
    _init_git_repo(donor)
    added = subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "submodule", "add", "-q",
         str(donor), "vendor/lib"],
        cwd=str(superproject), capture_output=True, text=True,
    )
    if added.returncode == 0:
        _, err_sub = validate_attach_path(
            str(superproject / "vendor" / "lib"), system_repo_dir=repo, drive_root=data
        )
        assert str(superproject.resolve()) in err_sub


def test_containment_refusal_does_not_recommend_an_ephemeral_root(tmp_path):
    """"Attach that root instead" is advice, and advice can be wrong. When the
    repository enclosing the folder is itself one of Ouroboros's removable
    checkouts, pointing the owner at it recommends a home that a `git worktree
    remove` or the retention sweep deletes. The refusal still names what it found;
    it just stops calling it a destination."""
    from ouroboros.project_sources import validate_attach_path

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    host = tmp_path / "host_repo"
    _init_git_repo(host)
    linked = tmp_path / "linked_wt"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "wt", str(linked)],
        cwd=str(host), check=True,
    )
    inner = linked / "pkg"
    inner.mkdir()

    _, error = validate_attach_path(str(inner), system_repo_dir=repo, drive_root=data)
    assert str(linked.resolve()) in error
    assert "attach that root instead" not in error
    assert "temporary checkout" in error


def test_promote_source_refuses_an_agent_supplied_ephemeral_checkout(tmp_path):
    """`resolve_promote_source` runs an AGENT-typed path through the attach guards
    only, and the paths an agent has in hand are exactly the checkouts Ouroboros
    makes for itself. A linked worktree passed every attach guard, so a project's
    PERMANENT home became a view that one `git worktree remove` deletes.
    `adopt_task_workspace` already applied the durable-place rule; this surface
    needs it more, because no owner ever looked at the path."""
    from ouroboros.projects_registry import get_project
    from ouroboros.promotion_source import resolve_promote_source

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    host = tmp_path / "host_repo"
    _init_git_repo(host)
    linked = tmp_path / "linked_wt"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "wt", str(linked)],
        cwd=str(host), check=True,
    )

    ctx = types.SimpleNamespace(DRIVE_ROOT=data, REPO_DIR=repo)
    folder, note, error, pid = resolve_promote_source(ctx, str(linked), "linkedproj")
    assert folder == "" and not note
    assert "linked git worktree" in error and str(host.resolve()) in error
    # Refused BEFORE any registry mutation — a placeless row is better than one
    # pointing at a folder that vanishes.
    assert get_project(data, pid) is None

    # The host repository itself is a perfectly good place and still passes.
    folder, _, error, _ = resolve_promote_source(ctx, str(host), "hostproj")
    assert error == "" and folder == str(host.resolve())


def test_create_never_git_inits_inside_someone_elses_repository(tmp_path):
    """The create dialog's `init_git` is a `git init` in whatever folder it was
    handed. Pointed at a repo subdirectory it produced a SHADOW repository nested in
    the owner's — after which the folder passed admission as a worktree root and
    every later diff, rollback and commit happened where the owner's real VCS shows
    only an untracked directory. The refusal comes BEFORE any registry mutation."""
    import asyncio

    from ouroboros.gateway.projects import api_projects_create
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    owner_repo, subdir = _repo_with_subdir(tmp_path)

    resp = asyncio.run(api_projects_create(
        _ProjectsReq({"name": "Nested", "path": str(subdir), "init_git": True},
                     drive_root=data, repo_dir=repo)
    ))
    assert resp.status_code == 400
    assert "inside the git repository at" in json.loads(resp.body)["error"]
    assert not (subdir / ".git").exists(), "no shadow repository inside the owner's repo"
    assert get_project(data, "nested") is None
    # The owner's own repository is untouched — no stray commit, no new tracked file.
    assert subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(owner_repo),
        capture_output=True, text=True, check=True,
    ).stdout.strip() == "?? packages/"


def test_init_git_route_refuses_a_working_dir_inside_another_repository(tmp_path):
    """The route whose ENTIRE job is to run `git init` in the folder. It re-runs the
    attach guards against the current working_dir, so a project whose row points
    inside a repository (registered before this guard, or hand-edited) still cannot
    have a shadow repo initialised in it."""
    import asyncio

    from ouroboros.gateway.projects import api_project_init_git
    from ouroboros.projects_registry import create_project, update_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    _owner_repo, subdir = _repo_with_subdir(tmp_path)

    create_project(data, "legacy", name="Legacy")
    update_project(data, "legacy", working_dir=str(subdir), provenance="attached")
    resp = asyncio.run(api_project_init_git(
        _ProjectsReq({}, drive_root=data, repo_dir=repo, path_params={"project_id": "legacy"})
    ))
    assert resp.status_code == 400
    assert "inside the git repository at" in json.loads(resp.body)["error"]
    assert not (subdir / ".git").exists()


def test_promote_attach_refuses_a_folder_nested_in_another_repository(tmp_path):
    """The agent-side attach inherits the guard from the same validator, so the
    subdirectory is never PERSISTED as the project's place. That matters on its own:
    a persisted repo subdir is a place task admission then refuses forever ("must be
    the git worktree root") with no offer attached and no route back."""
    from ouroboros.promotion_source import resolve_promote_source
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    owner_repo, subdir = _repo_with_subdir(tmp_path)

    ctx = types.SimpleNamespace(DRIVE_ROOT=data, REPO_DIR=repo)
    workspace, note, error, pid = resolve_promote_source(ctx, str(subdir), "nestedroom")
    assert workspace == "" and note == ""
    assert "inside the git repository at" in error
    assert get_project(data, pid) is None

    # The enclosing ROOT is exactly what the error tells them to use, and it works.
    workspace_ok, _note_ok, error_ok, pid_ok = resolve_promote_source(ctx, str(owner_repo), "rootroom")
    assert error_ok == "" and workspace_ok == str(owner_repo.resolve())
    assert get_project(data, pid_ok)["working_dir"] == str(owner_repo.resolve())


def test_adopt_refuses_a_task_folder_nested_in_another_repository(tmp_path):
    """The fourth surface: a converted task whose workspace was a repo subdirectory.
    The conversion still succeeds — that is its job — and the reason is disclosed."""
    from ouroboros.projects_registry import adopt_task_workspace, create_project, get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    _owner_repo, subdir = _repo_with_subdir(tmp_path)

    create_project(data, "sub", name="Sub")
    adopted, error = adopt_task_workspace(data, "sub", str(subdir), system_repo_dir=repo)
    assert adopted == ""
    assert "inside the git repository at" in error
    assert str(get_project(data, "sub").get("working_dir") or "") == ""


# --- the direct task API returns the decision instead of queueing ------------------

def test_api_tasks_create_returns_the_typed_git_offer_and_queues_nothing(tmp_path, monkeypatch):
    """The owner asked for file work in an untracked folder. Neither answer on its
    own is right: queueing it would mean editing files with no diff and no way back,
    and running `git init` for them would mutate a folder Ouroboros does not own. So
    the task is NOT queued and the browser gets a typed OFFER it can render."""
    from ouroboros.gateway.tasks import api_tasks_create

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    plain = tmp_path / "plain_workspace"
    plain.mkdir()

    enqueued: list = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: enqueued.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    resp = TestClient(app).post(
        "/api/tasks", json={"description": "edit the files", "workspace_root": str(plain)}
    )

    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error_code"] == "git_init_required"
    decision = payload["decision"]
    assert decision["decision"] == "git_init_required"
    assert decision["workspace_root"] == str(plain.resolve())
    assert decision["offer"] == "init_git"
    assert decision["enables"] == ["diff", "rollback", "branching"]
    assert "not tracked by git" in decision["message"]
    # The agent reads this same message on the promote path, and shell policy would
    # NOT stop it from running `git init` in an attached folder — so the message has
    # to say whose act this is, or "Ouroboros can start tracking it" reads as an
    # instruction to go and do it (A12).
    assert "Ouroboros will not run `git init` here" in decision["message"]
    assert enqueued == [], "the task must not be queued while the offer is unanswered"
    assert not (plain / ".git").exists(), "admission must NEVER initialise git itself"


def test_api_tasks_create_still_admits_a_git_worktree_root(tmp_path, monkeypatch):
    """Guard on the change above: the ordinary git workspace is untouched."""
    from ouroboros.gateway.tasks import api_tasks_create

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    ws = tmp_path / "tracked"
    _init_git_repo(ws)

    enqueued: list = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: enqueued.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    resp = TestClient(app).post(
        "/api/tasks", json={"description": "edit the files", "workspace_root": str(ws)}
    )
    assert resp.status_code == 200, resp.text
    assert enqueued and enqueued[0]["workspace_root"] == str(ws.resolve())


# --- the project-room promote path carries the same decision ----------------------

def test_promote_into_an_untracked_room_offers_git_and_does_not_queue(tmp_path, monkeypatch):
    """The agent-side sibling of the gateway case. The task is halted with the SAME
    decision object, and the owner-facing message says GIT_INIT_REQUIRED rather than
    WORKSPACE_UNUSABLE — the folder is not broken, the answer is simply missing."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, get_project, update_project

    # The drive root is a SIBLING of the owner's folder: a workspace overlapping the
    # Ouroboros data drive is refused by a guard that sits ahead of the git offer.
    drive = tmp_path / "data"
    drive.mkdir()
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    plain = tmp_path / "owner_folder"
    plain.mkdir()
    create_project(drive, "plainroom", name="Plain Room")
    update_project(drive, "plainroom", working_dir=str(plain), provenance="attached")

    enqueued: list = []
    sent: list = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
        send_with_budget=lambda chat_id, text: sent.append((chat_id, text)),
    )
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "untracked1",
        "objective": "Refactor the site",
        "project_id": "plainroom",
        "chat_id": 1,
    }, ctx)

    assert outcome["status"] == "needs_manual_target"
    assert outcome["reason"] == "git_init_required"
    assert outcome["decision"]["workspace_root"] == str(plain.resolve())
    assert outcome["decision"]["project_id"] == "plainroom"
    assert enqueued == []
    assert sent and "GIT_INIT_REQUIRED" in sent[0][1]
    assert not (plain / ".git").exists()
    # The attached folder is PRESERVED, never replaced by a fresh auto-provisioned
    # repo: auto-provisioning fires only for a project with no folder at all.
    assert get_project(drive, "plainroom")["working_dir"] == str(plain)
    result = json.loads((drive / "task_results" / "untracked1.json").read_text(encoding="utf-8"))
    assert result["reason_code"] == "git_init_required"


def test_the_workspace_block_tells_the_agent_not_to_answer_for_the_owner(tmp_path):
    """The rule goes where the existing rule already lives — the task text's own
    account of what git work is legitimate — because that is the sentence the agent
    reads while holding write+shell in the owner's folder. Shell policy permits
    `git init` there (it protects the Ouroboros runtime, not the owner's tree), so
    nothing but doctrine stands between the offer and the agent executing it."""
    from ouroboros.workspace_admission import compose_workspace_block

    block = compose_workspace_block(
        workspace_root=str(tmp_path), workspace_mode="external",
        memory_mode="forked", workspace_preflight={},
    )
    assert "Task-local git is allowed" in block
    assert "never run `git init` in the owner's project folder" in block
    assert "git_init_required" in block


def test_promote_workspace_none_opts_out_of_the_git_offer_too(tmp_path, monkeypatch):
    """A folder-less task in an untracked room is legitimate work (chat, research),
    so the explicit opt-out must sit AHEAD of the offer, not behind it."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import create_project, update_project

    drive = tmp_path / "data"
    drive.mkdir()
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    plain = tmp_path / "owner_folder2"
    plain.mkdir()
    create_project(drive, "plainroom2", name="Plain Room 2")
    update_project(drive, "plainroom2", working_dir=str(plain), provenance="attached")

    enqueued: list = []
    outcome = workers.promote_chat_to_task({
        "type": "promote_chat_to_task",
        "task_id": "untracked2",
        "objective": "Think about the site",
        "project_id": "plainroom2",
        "workspace": "none",
        "chat_id": 1,
    }, types.SimpleNamespace(
        enqueue_task=lambda task: enqueued.append(task),
        persist_queue_snapshot=lambda **_kwargs: True,
        load_state=lambda: {"owner_chat_id": 1},
    ))

    assert outcome["status"] == "scheduled"
    assert enqueued and not enqueued[0].get("workspace_root")


# --- the owner's YES ---------------------------------------------------------------

def test_init_git_route_answers_the_offer_and_then_the_task_admits(tmp_path):
    """The whole loop: attach a plain folder, get the offer instead of a queued
    task, say yes through the route, and the same folder now admits file work."""
    import asyncio

    from ouroboros.gateway.projects import api_project_init_git, api_projects_create
    from ouroboros.projects_registry import get_project
    from ouroboros.workspace_admission import GitInitRequiredError, validate_workspace_root

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    plain = tmp_path / "owner_site"
    plain.mkdir()
    (plain / "index.html") .write_text("<h1>hi</h1>\n", encoding="utf-8")

    created = asyncio.run(api_projects_create(
        _ProjectsReq({"name": "Site", "path": str(plain)}, drive_root=data, repo_dir=repo)
    ))
    assert created.status_code == 200
    pid = json.loads(created.body)["project"]["id"]

    # Before the answer: admission refuses with the offer, folder untouched.
    try:
        validate_workspace_root(str(plain), system_repo_dir=repo, drive_root=data)
        raise AssertionError("an untracked folder must not admit a file task")
    except GitInitRequiredError:
        pass
    assert not (plain / ".git").exists()

    resp = asyncio.run(api_project_init_git(
        _ProjectsReq({}, drive_root=data, repo_dir=repo, path_params={"project_id": pid})
    ))
    payload = json.loads(resp.body)
    assert resp.status_code == 200, payload
    assert payload["working_dir"] == str(plain.resolve())
    assert (plain / ".git").exists()
    # The owner's existing files are in the snapshot, not silently ignored.
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(plain), capture_output=True, text=True, check=True
    ).stdout.split()
    assert "index.html" in tracked
    # The folder binding is unchanged — saying yes tracks the place, it never moves it.
    assert get_project(data, pid)["working_dir"] == str(plain.resolve())
    # And the same folder now admits a file task.
    assert validate_workspace_root(
        str(plain), system_repo_dir=repo, drive_root=data
    ) == plain.resolve()


def test_init_git_route_refuses_what_it_cannot_safely_touch(tmp_path):
    """This route's whole job is to write into a folder, so it re-establishes the
    attach guards against the CURRENT working_dir instead of trusting a registry
    value that could have been edited or gone stale."""
    import asyncio

    from ouroboros.gateway.projects import api_project_init_git
    from ouroboros.projects_registry import create_project, update_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()

    def _call(pid):
        return asyncio.run(api_project_init_git(
            _ProjectsReq({}, drive_root=data, repo_dir=repo, path_params={"project_id": pid})
        ))

    assert _call("ghost").status_code == 404

    create_project(data, "fileless", name="Fileless")
    resp = _call("fileless")
    assert resp.status_code == 400
    assert json.loads(resp.body)["error_code"] == "no_working_dir"

    create_project(data, "gone", name="Gone")
    update_project(data, "gone", working_dir=str(tmp_path / "no-such-folder"))
    resp_gone = _call("gone")
    assert resp_gone.status_code == 400
    assert "does not exist" in json.loads(resp_gone.body)["error"]

    # A working_dir that overlaps the Ouroboros system repo is refused even though
    # the registry claims it: the guard is re-run, not remembered.
    create_project(data, "inrepo", name="InRepo")
    update_project(data, "inrepo", working_dir=str(repo))
    resp_repo = _call("inrepo")
    assert resp_repo.status_code == 400
    assert "Ouroboros system repo" in json.loads(resp_repo.body)["error"]
    assert not (repo / ".git").exists()


def test_init_git_route_keeps_credential_shaped_files_out_of_the_snapshot(tmp_path):
    """Same disclosed omission the create-dialog init_git makes: secrets are never
    baked into git history, and the owner is TOLD which files were left out."""
    import asyncio

    from ouroboros.gateway.projects import api_project_init_git
    from ouroboros.projects_registry import create_project, update_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    plain = tmp_path / "with_secrets"
    plain.mkdir()
    (plain / "app.py").write_text("print('x')\n", encoding="utf-8")
    (plain / ".env").write_text("API_KEY=supersecret\n", encoding="utf-8")

    create_project(data, "secrets", name="Secrets")
    update_project(data, "secrets", working_dir=str(plain), provenance="attached")
    resp = asyncio.run(api_project_init_git(
        _ProjectsReq({}, drive_root=data, repo_dir=repo, path_params={"project_id": "secrets"})
    ))
    payload = json.loads(resp.body)
    assert resp.status_code == 200, payload
    assert ".env" in payload["init_git_skipped"]
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=str(plain), capture_output=True, text=True, check=True
    ).stdout.split()
    assert "app.py" in tracked and ".env" not in tracked


# --- an auto-provisioned place is a place the owner can be shown -------------------

def test_autoprovisioned_folder_is_surfaced_as_genesis_not_silently(tmp_path, monkeypatch):
    """A11: a project always has a place, and the owner can always see WHERE and
    HOW it got one. The path was already on the row; what was missing was the fact
    that Ouroboros made the folder rather than the owner pointing at it, which left
    an auto-provisioned place indistinguishable from an unstamped attach."""
    import os

    from ouroboros.projects_registry import (
        create_project,
        ensure_project_workspace,
        get_project,
        projects_summary,
    )

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))

    create_project(data, "novel", name="Novel", origin="owner_ui")
    assert str(get_project(data, "novel").get("provenance") or "") == ""

    provisioned = ensure_project_workspace(data, "novel", repo)
    assert provisioned and os.path.isdir(provisioned)

    entry = get_project(data, "novel")
    assert entry["working_dir"] == provisioned
    assert entry["provenance"] == "genesis"
    row = next(p for p in projects_summary(data) if p["id"] == "novel")
    assert row["working_dir"] == provisioned and row["provenance"] == "genesis"

    # Idempotent: a second call returns the same tree and re-stamps nothing.
    assert ensure_project_workspace(data, "novel", repo) == provisioned
    assert get_project(data, "novel")["provenance"] == "genesis"


def test_autoprovisioning_never_relabels_an_existing_provenance(tmp_path, monkeypatch):
    """How a folder came to be is a historical fact. If a project's attached folder
    has gone missing, provisioning a replacement must not rewrite its history into
    "Ouroboros made this" — the owner attached something, and that is what happened.

    And the replacement has to actually LAND. Asserting the truthiness of the return
    was too weak to notice that the stale path was being handed back unchanged: a
    non-empty string is exactly what a REFUSED replacement returns too. The folder is
    the point, so the assertion is that a folder EXISTS there and that the row now
    names it."""
    from ouroboros.projects_registry import (
        create_project,
        ensure_project_workspace,
        get_project,
        update_project,
    )

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))

    vanished = tmp_path / "vanished"
    create_project(data, "moved", name="Moved", origin="owner_ui")
    update_project(data, "moved", working_dir=str(vanished), provenance="attached")
    result = ensure_project_workspace(data, "moved", repo)
    assert os.path.isdir(result)
    assert result != str(vanished)
    row = get_project(data, "moved")
    assert row["provenance"] == "attached"
    # The row must point at the tree that was just dug, or the genesis folder is an
    # orphan under a root the GC deliberately never prunes and the caller is holding
    # a path that does not exist.
    assert row["working_dir"] == result


# --- a project born from a task inherits that task's folder ------------------------

def _persist_task_result(drive_root: pathlib.Path, task_id: str, **fields) -> None:
    (drive_root / "task_results").mkdir(parents=True, exist_ok=True)
    (drive_root / "task_results" / f"{task_id}.json").write_text(
        json.dumps({"task_id": task_id, "status": "completed", **fields}), encoding="utf-8"
    )


def test_turn_into_project_adopts_the_tasks_working_folder(tmp_path):
    """A11: converting a card into a project used to drop the folder the task was
    working in, so the project came out placeless and its NEXT task provisioned a
    different empty tree — silently moving the work somewhere else."""
    import asyncio

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    ws = tmp_path / "site"
    _init_git_repo(ws)
    _persist_task_result(data, "t1", description="build the site", title="Site build",
                         workspace_root=str(ws))

    resp = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t1", "id": "site", "name": "Site"}, drive_root=data, repo_dir=repo)
    ))
    payload = json.loads(resp.body)
    assert resp.status_code == 200, payload
    assert payload["project"]["working_dir"] == str(ws.resolve())
    entry = get_project(data, "site")
    assert entry["working_dir"] == str(ws.resolve())
    assert entry["provenance"] == "attached" and entry["trusted_at"]


def test_turn_into_project_adopts_an_untracked_folder_too(tmp_path):
    """A12: the adopted folder is not required to be under git — a plain folder is
    a legitimate place, and the git question is asked at task admission."""
    import asyncio

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    plain = tmp_path / "notes"
    plain.mkdir()
    _persist_task_result(data, "t2", description="sort the notes", workspace_root=str(plain))

    resp = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t2", "id": "notes", "name": "Notes"}, drive_root=data, repo_dir=repo)
    ))
    assert resp.status_code == 200
    assert get_project(data, "notes")["working_dir"] == str(plain.resolve())
    assert not (plain / ".git").exists()


def test_turn_into_project_discloses_a_folder_it_cannot_adopt(tmp_path):
    """The conversion still succeeds — its job is to make the project — but a folder
    that has moved is REPORTED rather than quietly leaving a placeless project, and a
    path overlapping the Ouroboros roots is refused by the ordinary attach guards."""
    import asyncio

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    _persist_task_result(data, "t3", description="x", workspace_root=str(tmp_path / "vanished"))
    _persist_task_result(data, "t4", description="y", workspace_root=str(repo))

    gone = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t3", "id": "gonep", "name": "Gone"}, drive_root=data, repo_dir=repo)
    ))
    payload = json.loads(gone.body)
    assert gone.status_code == 200
    assert "was not adopted" in payload["working_dir_error"]
    assert str(get_project(data, "gonep").get("working_dir") or "") == ""

    overlapping = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t4", "id": "repop", "name": "Repo"}, drive_root=data, repo_dir=repo)
    ))
    assert "Ouroboros system repo" in json.loads(overlapping.body)["working_dir_error"]
    assert str(get_project(data, "repop").get("working_dir") or "") == ""


def test_from_task_response_types_both_folder_facts(tmp_path):
    """`working_dir_error` used to be free text: absent from the contracts, absent
    from the JS mirror, and read by nobody (the client takes `payload.project`), so
    a conversion that quietly produced a PLACELESS project was indistinguishable
    from one that worked. Both folder facts are now declared fields of
    `ProjectFromTaskResponse`, mirrored in api_types.js and rendered by the client."""
    import asyncio

    from ouroboros.gateway.contracts import ProjectFromTaskResponse
    from ouroboros.gateway.projects import api_project_from_task

    assert set(ProjectFromTaskResponse.__annotations__) >= {"working_dir", "working_dir_error"}
    mirror = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "api_types.js").read_text(
        encoding="utf-8"
    )
    assert "@typedef {Object} ProjectFromTaskResponse" in mirror
    client = (pathlib.Path(__file__).resolve().parent.parent / "web" / "modules" / "chat.js").read_text(
        encoding="utf-8"
    )
    assert "payload.working_dir_error" in client, "the disclosure must reach the owner"

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    ws = tmp_path / "adopted_site"
    _init_git_repo(ws)
    _persist_task_result(data, "t7", description="ship it", workspace_root=str(ws))

    resp = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t7", "id": "shipit", "name": "Ship"}, drive_root=data, repo_dir=repo)
    ))
    payload = json.loads(resp.body)
    assert resp.status_code == 200, payload
    assert payload["working_dir"] == str(ws.resolve())
    assert "working_dir_error" not in payload


def test_turn_into_project_never_replaces_an_existing_project_folder(tmp_path):
    """Conversion into an EXISTING project must not repoint that project at the
    converted task's folder — a project's place is not silently reassigned."""
    import asyncio

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import create_project, get_project, update_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    home_folder = tmp_path / "already"
    _init_git_repo(home_folder)
    other = tmp_path / "elsewhere"
    _init_git_repo(other)
    create_project(data, "keeper", name="Keeper", origin="owner_ui")
    update_project(data, "keeper", working_dir=str(home_folder), provenance="attached")
    _persist_task_result(data, "t5", description="z", workspace_root=str(other))

    resp = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t5", "id": "keeper", "name": "Keeper"}, drive_root=data, repo_dir=repo)
    ))
    assert resp.status_code == 200
    assert get_project(data, "keeper")["working_dir"] == str(home_folder)


def test_turn_into_project_refuses_an_ephemeral_checkout_as_a_durable_place(tmp_path):
    """A task's workspace_root is exactly where Ouroboros's OWN checkouts live. A
    linked worktree is removable with one command and a subagent checkout is swept
    by age, so adopting either would give the project a place that vanishes under
    it — and it would be a checkout of the system body, not the owner's work."""
    import asyncio

    from ouroboros.gateway.projects import api_project_from_task
    from ouroboros.projects_registry import get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    origin = tmp_path / "origin"
    _init_git_repo(origin)
    linked = tmp_path / "linked_checkout"
    subprocess.run(
        ["git", "worktree", "add", "-q", "-b", "wt-adopt", str(linked)],
        cwd=str(origin), check=True, capture_output=True,
    )
    _persist_task_result(data, "t6", description="branch work", workspace_root=str(linked))

    resp = asyncio.run(api_project_from_task(
        _ProjectsReq({"task_id": "t6", "id": "linkedp", "name": "Linked"}, drive_root=data, repo_dir=repo)
    ))
    payload = json.loads(resp.body)
    assert resp.status_code == 200, payload
    assert "linked git worktree" in payload["working_dir_error"]
    assert str(get_project(data, "linkedp").get("working_dir") or "") == ""


def test_adopt_refuses_the_subagent_and_thread_worktree_roots(tmp_path, monkeypatch):
    """The other half of the same rule, at the registry seam both conversion paths
    share: an acting subagent's self_worktree and a thread's branch-off checkout are
    Ouroboros's own trees, and the subagent root is age-GC'd."""
    from ouroboros.projects_registry import adopt_task_workspace, create_project, get_project

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    sub_root = tmp_path / "subagent_worktrees"
    thread_root = tmp_path / "thread_worktrees"
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(sub_root))
    monkeypatch.setenv("OUROBOROS_THREAD_WORKTREE_ROOT", str(thread_root))

    for pid, root, marker in (
        ("selfwt", sub_root / "task_abc", "acting-subagent worktree root"),
        ("threadwt", thread_root / "proj__1", "thread worktree root"),
    ):
        root.mkdir(parents=True)
        create_project(data, pid, name=pid)
        adopted, error = adopt_task_workspace(data, pid, str(root), system_repo_dir=repo)
        assert adopted == "", f"{pid} must not adopt {root}"
        assert marker in error, error
        assert str(get_project(data, pid).get("working_dir") or "") == ""


# --- a project's place is bound atomically -----------------------------------------

def test_set_working_dir_if_absent_binds_once_under_concurrency(tmp_path):
    """"Never overwrites an existing working_dir" was true of the code and false of
    the timeline: the check and the write were separate locked operations with a
    whole folder validation (or a whole genesis provisioning) between them. Exactly
    one of N concurrent writers may win, and every loser must learn the winner's
    folder rather than assume its own landed."""
    import threading

    from ouroboros.projects_registry import create_project, get_project, set_working_dir_if_absent

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "raced", name="Raced")
    folders = [tmp_path / f"candidate_{n}" for n in range(6)]
    for folder in folders:
        folder.mkdir()

    results: list = []
    start = threading.Barrier(len(folders))

    def _claim(folder):
        start.wait()
        results.append(set_working_dir_if_absent(
            data, "raced", str(folder), provenance="attached", trusted_at="2026-08-10T00:00:00Z"
        ))

    threads = [threading.Thread(target=_claim, args=(f,)) for f in folders]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    winners = [bound for bound, claimed in results if claimed]
    assert len(winners) == 1, f"exactly one writer may bind the place, got {winners}"
    bound_dir = get_project(data, "raced")["working_dir"]
    assert bound_dir == winners[0]
    # Every loser is TOLD the winner's folder — the whole point of the second value.
    assert {bound for bound, claimed in results if not claimed} in (set(), {bound_dir})
    assert get_project(data, "raced")["provenance"] == "attached"


def test_ensure_project_workspace_does_not_clobber_a_place_bound_mid_provision(tmp_path, monkeypatch):
    """The race that motivates the atomic setter, played out on the slow side: a
    conversion binds the project's folder while provisioning is still digging. The
    genesis tree must NOT replace it (the owner's real folder would be silently
    swapped), and the abandoned tree must be logged — it sits under the durable
    projects root, which is never GC-pruned."""
    from ouroboros import projects_registry
    from ouroboros.projects_registry import (
        create_project,
        ensure_project_workspace,
        get_project,
        set_working_dir_if_absent,
    )

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("OUROBOROS_SUBAGENT_PROJECTS_ROOT", str(tmp_path / "projects"))
    owner_folder = tmp_path / "owner_folder"
    owner_folder.mkdir()

    create_project(data, "slow", name="Slow", origin="owner_ui")
    real_provision = None

    def _slow_provision(**kwargs):
        # The competing writer lands DURING provisioning — the window the old
        # get_project → update_project pair left wide open.
        set_working_dir_if_absent(data, "slow", str(owner_folder), provenance="attached")
        return real_provision(**kwargs)

    import ouroboros.subagent_worktrees as sw

    real_provision = sw.provision_genesis_project
    monkeypatch.setattr(sw, "provision_genesis_project", _slow_provision)

    warnings: list = []
    monkeypatch.setattr(projects_registry.log, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg))

    result = ensure_project_workspace(data, "slow", repo)
    assert result == str(owner_folder), "the folder bound first must win"
    assert get_project(data, "slow")["working_dir"] == str(owner_folder)
    assert get_project(data, "slow")["provenance"] == "attached"
    assert any("abandoned" in str(text) for text in warnings), warnings


def test_ensure_project_scope_adopts_the_running_tasks_folder(tmp_path, monkeypatch):
    """The in-task sibling of the card conversion: a task that self-scopes mid-run
    hands its own folder to the project it just created."""
    import supervisor.workers as workers
    from ouroboros.projects_registry import get_project

    drive = tmp_path / "data"
    drive.mkdir()
    ws = tmp_path / "live_site"
    _init_git_repo(ws)
    monkeypatch.setattr(workers, "DRIVE_ROOT", drive)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")

    ctx = types.SimpleNamespace(
        RUNNING={"live1": {"task": {"id": "live1", "workspace_root": str(ws)}}},
        PENDING=[],
    )
    workers.ensure_project_scope(
        {"task_id": "live1", "project_id": "livesite", "project_name": "Live Site"}, ctx
    )
    assert get_project(drive, "livesite")["working_dir"] == str(ws.resolve())


# --- T4: the direct task API is an ADMISSION path too -----------------------------

def test_every_resolve_room_workspace_call_site_names_the_room(tmp_path):
    """A guard against the NEXT admission path forgetting `room_chat_id`.

    `resolve_room_workspace` grew that argument so it can tell WHICH thread is
    asking: a thread that branched off works in its own checkout, and a task
    admitted without the room's chat id is handed the PROJECT's folder instead —
    it then takes the project folder's writer lane and queues behind it. Branching
    bought the owner a second copy of their files and no concurrency, and nothing
    on any surface says so; the loss is visible only through the lane.

    The argument has a default, deliberately (the resolver is also asked
    project-wide questions), so a new caller that omits it compiles, passes every
    behavioural test it writes, and silently reintroduces the defect. This reads
    the SOURCE instead, because the fact under test is "no call site forgets",
    which no single behavioural test can express.
    """
    import re

    root = pathlib.Path(__file__).resolve().parents[1]
    call = re.compile(r"resolve_room_workspace\s*\(", re.MULTILINE)
    offenders = []
    inspected = 0
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if rel.parts[0] in {"tests", ".git", "build", "dist"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in call.finditer(text):
            # The call spans several lines; take the balanced argument list.
            depth, index = 0, match.end() - 1
            while index < len(text):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            args = text[match.end():index]
            if "def resolve_room_workspace" in text[max(0, match.start() - 40):match.start()]:
                continue
            inspected += 1
            if "room_chat_id" not in args:
                offenders.append(f"{rel}:{text[:match.start()].count(chr(10)) + 1}")
    assert inspected >= 2, (
        "the scan found no admission call sites at all — the pattern went stale and "
        "this guard is now vacuous"
    )
    assert offenders == [], (
        "these admission paths call resolve_room_workspace without room_chat_id, so "
        f"every thread of a project resolves to the project's folder: {offenders}"
    )


def test_the_direct_task_api_admits_a_branched_thread_into_its_own_checkout(
    tmp_path, monkeypatch,
):
    """The direct task API resolved NO room workspace at all (T4).

    `POST /api/tasks` with a `project_id` and no `workspace_root` queued a
    folder-less task, so a task born in a branched thread's room never reached that
    thread's checkout. Same admission question, same answer, whichever door the
    caller came through.
    """
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.projects_registry import create_project, create_thread
    from ouroboros.thread_worktrees import provision_thread_worktree

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    folder = tmp_path / "alpha"
    _init_git_repo(folder)
    create_project(data, "alpha", working_dir=str(folder))
    thread = create_thread(data, "alpha", name="Branched")
    handle = provision_thread_worktree(
        repo_dir=folder, project_id="alpha", thread_id=thread["id"],
        data_dir=data, worktree_root=tmp_path / "checkouts",
    )

    enqueued: list = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: enqueued.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    client = TestClient(app)

    resp = client.post("/api/tasks", json={
        "description": "work in my branch",
        "project_id": "alpha",
        "chat_id": int(thread["chat_id"]),
    })
    assert resp.status_code == 200, resp.text
    assert enqueued[-1]["workspace_root"] == str(pathlib.Path(handle.path).resolve())

    # ...and thread #0 of the same project still gets the project folder, so the
    # two are different LANES and can run at the same time (A7/A14).
    resp = client.post("/api/tasks", json={
        "description": "work in the folder",
        "project_id": "alpha",
        "chat_id": 0,
    })
    assert resp.status_code == 200, resp.text
    assert enqueued[-1]["workspace_root"] == str(folder.resolve())

    from ouroboros.project_lease import candidate_is_leasable, running_project_lanes

    held = running_project_lanes([{"task": enqueued[0]}])
    assert candidate_is_leasable(enqueued[1], held) is True
    assert candidate_is_leasable(dict(enqueued[0]), held) is False


def test_a_workspace_none_task_is_still_folder_less_through_the_direct_api(
    tmp_path, monkeypatch,
):
    """The opt-out survives the new resolution: `workspace="none"` is an explicit
    "I write nowhere", and resolving a folder for it would hand file access to a
    caller that asked not to have any."""
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.projects_registry import create_project

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    folder = tmp_path / "alpha"
    _init_git_repo(folder)
    create_project(data, "alpha", working_dir=str(folder))

    enqueued: list = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: enqueued.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    resp = TestClient(app).post("/api/tasks", json={
        "description": "think about it", "project_id": "alpha", "workspace": "none",
    })
    assert resp.status_code == 200, resp.text
    assert not enqueued[-1].get("workspace_root")


def test_the_direct_task_api_offers_git_for_an_untracked_project_folder(
    tmp_path, monkeypatch,
):
    """A12 through the second door: the project's own folder is untracked, so the
    task is not queued and the caller gets the same typed offer the explicit
    `workspace_root` path already returned."""
    from ouroboros.gateway.tasks import api_tasks_create
    from ouroboros.projects_registry import create_project

    repo = tmp_path / "repo"
    repo.mkdir()
    data = tmp_path / "data"
    (data / "memory").mkdir(parents=True)
    plain = tmp_path / "plain"
    plain.mkdir()
    create_project(data, "alpha", working_dir=str(plain))

    enqueued: list = []
    monkeypatch.setattr("supervisor.workers.WORKERS", {0: object()})
    monkeypatch.setattr("supervisor.workers._WORKER_POOL_DISABLED_REASON", "")
    monkeypatch.setattr("supervisor.queue.enqueue_task", lambda task: enqueued.append(dict(task)) or task)
    monkeypatch.setattr("supervisor.queue.persist_queue_snapshot", lambda reason="": True)

    app = Starlette(routes=[Route("/api/tasks", endpoint=api_tasks_create, methods=["POST"])])
    app.state.drive_root = data
    app.state.repo_dir = repo
    resp = TestClient(app).post("/api/tasks", json={
        "description": "edit the files", "project_id": "alpha",
    })
    assert resp.status_code == 400, resp.text
    assert resp.json()["error_code"] == "git_init_required"
    assert enqueued == []
    assert not (plain / ".git").exists()


# --------------------------------------------------------------------------- #
# P7 — the vanished-folder REPLACEMENT is a compare-and-swap
# --------------------------------------------------------------------------- #

def test_two_concurrent_replacements_leave_no_orphan_and_no_lie(tmp_path, monkeypatch):
    """P7: `ensure_project_workspace`'s replacement branch was an unconditional
    `update_project` — no `_file_write_lock`, no comparison against the observed
    stale value.

    Reproduced: two callers both observe the same vanished `working_dir`, both
    provision (`genesis_1`, `genesis_2`), both write. The registry ends on one and
    the OTHER is orphaned under the never-pruned durable projects root, while BOTH
    callers are handed their own path back — so one reports a binding that does not
    exist. Exactly the race `set_working_dir_if_absent` was written to close, on the
    one branch that opted out of it.
    """
    import threading

    from ouroboros import projects_registry
    from ouroboros.projects_registry import (
        create_project,
        ensure_project_workspace,
        get_project,
        update_project,
    )

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    vanished = tmp_path / "gone"
    create_project(data, "alpha", name="Alpha", origin="owner_ui")
    update_project(data, "alpha", working_dir=str(vanished), provenance="attached")
    assert not vanished.exists()

    made: list = []
    gate = threading.Barrier(2)
    counter = {"n": 0}
    lock = threading.Lock()

    class Handle:
        def __init__(self, path):
            self.path = path

    def fake_provision(*, repo_dir, task_id, data_dir, dir_name=""):
        with lock:
            counter["n"] += 1
            path = tmp_path / f"genesis_{counter['n']}"
        path.mkdir()
        made.append(str(path))
        gate.wait(timeout=10)  # force the interleave
        return Handle(path)

    import ouroboros.subagent_worktrees as sw

    monkeypatch.setattr(sw, "provision_genesis_project", fake_provision)
    warnings: list = []
    monkeypatch.setattr(
        projects_registry.log, "warning",
        lambda msg, *a, **k: warnings.append(msg % a if a else msg),
    )

    results: list = []

    def call():
        results.append(ensure_project_workspace(data, "alpha", repo))

    first = threading.Thread(target=call)
    second = threading.Thread(target=call)
    first.start(); second.start(); first.join(); second.join()

    final = str(get_project(data, "alpha")["working_dir"])
    assert len(made) == 2, "both callers really did provision their own tree"
    assert final in made
    # NOBODY is told a path that is not the binding: the loser reports the winner's.
    assert results == [final, final], (results, final)
    # ...and the abandoned tree is named rather than silently orphaned.
    assert any("abandoned" in str(text) for text in warnings), warnings
    # Provenance is a historical fact and survives the swap.
    assert get_project(data, "alpha")["provenance"] == "attached"


def test_a_newer_binding_is_never_overwritten_by_a_stale_replacement(tmp_path, monkeypatch):
    """The deterministic half: a binding written between the READ and the WRITE was
    silently overwritten, because the write compared against nothing."""
    from ouroboros.projects_registry import (
        create_project,
        ensure_project_workspace,
        get_project,
        update_project,
    )

    data = tmp_path / "data"
    data.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    vanished = tmp_path / "gone"
    create_project(data, "beta", name="Beta", origin="owner_ui")
    update_project(data, "beta", working_dir=str(vanished), provenance="attached")

    owner_folder = tmp_path / "the_owner_attached_this"
    owner_folder.mkdir()

    class Handle:
        def __init__(self, path):
            self.path = path

    def fake_provision(*, repo_dir, task_id, data_dir, dir_name=""):
        # While THIS caller digs, the owner attaches a real folder.
        update_project(data, "beta", working_dir=str(owner_folder))
        path = tmp_path / "genesis_beta"
        path.mkdir()
        return Handle(path)

    import ouroboros.subagent_worktrees as sw

    monkeypatch.setattr(sw, "provision_genesis_project", fake_provision)

    returned = ensure_project_workspace(data, "beta", repo)

    assert str(get_project(data, "beta")["working_dir"]) == str(owner_folder)
    assert returned == str(owner_folder), "the caller must be told the winner's path"


def test_replace_working_dir_if_unchanged_is_a_real_compare_and_swap(tmp_path):
    """The primitive on its own, beside `set_working_dir_if_absent`."""
    from ouroboros.projects_registry import (
        begin_project_deletion,
        create_project,
        get_project,
        replace_working_dir_if_unchanged,
        update_project,
    )

    data = tmp_path / "data"
    data.mkdir()
    create_project(data, "gamma", name="Gamma", origin="owner_ui")
    update_project(data, "gamma", working_dir="/w/old")

    # A stale expectation changes nothing and reports the truth.
    assert replace_working_dir_if_unchanged(data, "gamma", "/w/never", "/w/new") == (
        "/w/old", False,
    )
    assert get_project(data, "gamma")["working_dir"] == "/w/old"

    # The matching expectation swaps, once.
    assert replace_working_dir_if_unchanged(
        data, "gamma", "/w/old", "/w/new", provenance="genesis",
    ) == ("/w/new", True)
    assert get_project(data, "gamma")["working_dir"] == "/w/new"
    assert get_project(data, "gamma")["provenance"] == "genesis"
    # ...and a second attempt with the same now-stale expectation does not.
    assert replace_working_dir_if_unchanged(data, "gamma", "/w/old", "/w/other") == (
        "/w/new", False,
    )

    # Provenance is a historical fact: an existing one is never rewritten.
    assert replace_working_dir_if_unchanged(
        data, "gamma", "/w/new", "/w/third", provenance="attached",
    ) == ("/w/third", True)
    assert get_project(data, "gamma")["provenance"] == "genesis"

    # An inactive project answers ("", False), exactly as the sibling does.
    begin_project_deletion(data, "gamma")
    assert replace_working_dir_if_unchanged(data, "gamma", "/w/third", "/w/fourth") == (
        "", False,
    )
