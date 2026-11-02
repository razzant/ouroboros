"""What a writable session's engine grants it besides its copy (#1775, #2070).

The host makes short physical scratch (owned by the starting task's drive) and the shared download
cache's folders, and sends them as ``execution.auxiliaryWritableRoots`` only where the route's
catalog row negotiates ``accessControl.auxiliaryWritableRoots`` and the start is an Agent
``workspace_write`` one; its instructions name exactly those places. A full-access start, which needs no
grant, gets the same owned scratch named and declared disposable, with nothing sent. Elsewhere the rights
and the request stay as they were, an unsupported ``workspace_write`` route hearing a typed limitation.
An exact retry replays the recorded request; a continuation binds the rights of its own start.
Synthetic engines and isolated roots only: no live engine, provider, Settings or database.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import workspace_preflight, test_environment
from ouroboros.delegate_start_instructions import scratch_fact
from ouroboros.workspace_preflight import (
    PACKAGE_DOWNLOADS_DIRNAME,
    SESSION_SCRATCH_DIRNAME,
    ServiceDirectories,
    session_environment_facts,
    session_service_directories,
)
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)

GRANTED = {"accessControl": {"auxiliaryWritableRoots": True}}
CACHES = (("uv", "UV_CACHE_DIR"), ("pip", "PIP_CACHE_DIR"), ("npm", "npm_config_cache"))
_PLATFORM_TEMP_PARENT = test_environment.default_temp_parent


@pytest.fixture(autouse=True)
def _synthetic_temp_parent(tmp_path, monkeypatch):
    # All ordinary cases remain below the safe_test-owned pytest root. Only the
    # explicit real consumer test uses the platform's physically short parent.
    parent = tmp_path / "temp"
    parent.mkdir()
    monkeypatch.setattr(test_environment, "default_temp_parent", lambda: parent)


def _scratch_for(data, task, key):
    record = data / "task_drives" / task / SESSION_SCRATCH_DIRNAME / key / "scratch.json"
    return pathlib.Path(json.loads(record.read_text(encoding="utf-8"))["path"])


class _Catalog:
    """A catalog reader with one route row; counts reads so 'never asked' is observable."""

    engine_version = "3.22.0"

    def __init__(self, row=None, *, error=None):
        self.row, self.error, self.reads = row, error, 0

    def agent_capabilities(self, **_kw):
        self.reads += 1
        if self.error is not None:
            raise self.error
        return {"harnesses": [] if self.row is None else [{"id": "route", **self.row}]}


def _prepare(tmp_path, gateway, *, access="workspace_write", mode="agent", task_id="task-1", key="snap-1",
             execution=None):
    execution = execution or tmp_path / "copy"
    execution.mkdir(parents=True, exist_ok=True)
    return session_service_directories(gateway, "route", access=access, mode=mode, data_root=tmp_path / "data",
                                       drive_root=tmp_path / "data", task_id=task_id, key=key,
                                       execution_root=execution)


def _symlink_or_skip(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:  # e.g. Windows without the symlink privilege
        pytest.skip(f"symlinks unavailable here: {exc}")


def _negotiate(monkeypatch, engine_class, supported=True):
    """Give a suite's engine class the negotiated catalog bit on every route row."""
    original = engine_class.agent_capabilities

    def catalog(self, **_kw):
        body = original(self)
        for row in body["harnesses"]:
            row["accessControl"] = {"auxiliaryWritableRoots": supported}
        return body

    monkeypatch.setattr(engine_class, "agent_capabilities", catalog)


# --------------------------------------------------------------------------- the preparation seam

def test_a_negotiated_workspace_write_start_gets_only_its_exact_owned_directories(tmp_path):
    grant = _prepare(tmp_path, _Catalog(GRANTED))
    data = (tmp_path / "data").resolve()
    scratch = _scratch_for(data, "task-1", "snap-1")
    caches = [data / PACKAGE_DOWNLOADS_DIRNAME / name for name, _variable in CACHES]
    assert grant == ServiceDirectories(roots=tuple(map(str, (scratch, *caches))), scratch=str(scratch))
    for root in map(pathlib.Path, grant.roots):  # real, empty folders: a shallow mkdir, never an install
        assert root.is_dir() and not root.is_symlink() and not any(root.iterdir())
    made = sorted(path.relative_to(data).as_posix() for path in data.rglob("*"))
    assert made == sorted(["package-downloads", "package-downloads/npm", "package-downloads/pip",
                           "package-downloads/uv", "task_drives", "task_drives/task-1",
                           "task_drives/task-1/session_scratch_owners", "task_drives/task-1/session_scratch_owners/snap-1",
                           "task_drives/task-1/session_scratch_owners/snap-1/scratch.json",
                           "state", "state/session_scratch", "state/session_scratch/task-1",
                           "state/session_scratch/task-1/snap-1.json"])
    note = json.loads((data / "state/session_scratch/task-1/snap-1.json").read_text(encoding="utf-8"))
    assert note == {"schema_version": 1, "owner": str(data / "task_drives/task-1/session_scratch_owners/snap-1")}
    (scratch / "env-marker").write_text("kept\n", encoding="utf-8")  # the same key names the same folder again
    assert _prepare(tmp_path, _Catalog(GRANTED)) == grant
    assert (scratch / "env-marker").read_text(encoding="utf-8") == "kept\n"


@pytest.mark.parametrize("row", [
    None, {}, {"accessControl": {}}, {"accessControl": {"auxiliaryWritableRoots": False}},
    {"accessControl": {"auxiliaryWritableRoots": "true"}}, {"accessControl": True},
], ids=["route-missing", "engine-predates-row", "empty-control", "false", "string", "not-an-object"])
def test_without_the_negotiated_bit_nothing_is_prepared_or_claimed(tmp_path, row):
    grant = _prepare(tmp_path, _Catalog(row))
    assert grant.roots == () and grant.scratch == ""
    assert grant.limitation.startswith("auxiliary_roots_unsupported:")
    assert not (tmp_path / "data").exists()  # no host write for a route that grants nothing


def test_an_unreadable_catalog_is_no_support(tmp_path):
    grant = _prepare(tmp_path, _Catalog(GRANTED, error=OSError("catalog read failed")))
    assert grant.roots == () and grant.limitation.startswith("auxiliary_roots_unsupported:")
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("access, mode", [("readonly", "ask"), ("full", "ask"), ("workspace_write", "ask"),
                                          ("readonly", "agent")])
def test_readonly_and_non_agent_starts_negotiate_nothing(tmp_path, access, mode):
    catalog = _Catalog(GRANTED)
    assert _prepare(tmp_path, catalog, access=access, mode=mode) == ServiceDirectories()
    assert catalog.reads == 0 and not (tmp_path / "data").exists()


@pytest.mark.parametrize("row", [GRANTED, None, {**GRANTED, "writeMechanism": "tool_policy"}],
                         ids=["negotiating", "engine-predates-row", "tool-policy"])
def test_full_access_owns_its_scratch_and_negotiates_nothing(tmp_path, row):
    """Ownership is separate from the grant: full access reaches the scratch already, so the host makes
    and records it, whatever the route row says, and sends no root and makes no cache folder."""
    catalog = _Catalog(row)
    owned = _prepare(tmp_path, catalog, access="full")
    data = (tmp_path / "data").resolve()
    scratch = _scratch_for(data, "task-1", "snap-1")
    assert owned == ServiceDirectories(roots=(), scratch=str(scratch)) and catalog.reads == 0
    assert scratch.is_dir() and not any(scratch.iterdir()) and not (data / PACKAGE_DOWNLOADS_DIRNAME).exists()
    assert _prepare(tmp_path, catalog, access="full") == owned  # the same key names the same folder again


def test_a_full_access_scratch_that_cannot_be_made_is_a_typed_gap(tmp_path):
    (tmp_path / "data").write_text("not a folder\n", encoding="utf-8")
    gap = _prepare(tmp_path, _Catalog(GRANTED), access="full")
    assert gap.roots == () and gap.scratch == ""
    assert gap.limitation.startswith(workspace_preflight.SCRATCH_UNAVAILABLE + ":")
    text = workspace_preflight.session_service_facts(gap, tmp_path / "elsewhere")
    assert "no disposable scratch could be made for this run (scratch_unavailable:" in text
    assert f"({scratch_fact()}), where nothing owns or removes it, and report that gap" in text
    assert "your engine grants this run none" not in text


def test_an_unconfined_tool_policy_route_keeps_its_on_demand_behavior(tmp_path):
    """No filesystem fence confines its allowed shell, so a grant would change nothing: no field,
    no folders, no limitation (the confinement read decides relevance, never support)."""
    assert _prepare(tmp_path, _Catalog({**GRANTED, "writeMechanism": "tool_policy"})) == ServiceDirectories()
    assert not (tmp_path / "data").exists()
    assert _prepare(tmp_path, _Catalog({**GRANTED, "writeMechanism": "fs_sandbox"})).roots  # a fenced route


@pytest.mark.parametrize("linked", [
    "task_drives/task-1", "task_drives/task-1/session_scratch_owners", "task_drives/task-1/session_scratch_owners/snap-1",
    "package-downloads", "package-downloads/pip",
])
def test_a_link_never_turns_a_named_folder_into_a_broader_grant(tmp_path, linked):
    outside = tmp_path / "operator-home"
    outside.mkdir()
    link = tmp_path / "data" / linked
    link.parent.mkdir(parents=True, exist_ok=True)
    _symlink_or_skip(link, outside)
    grant = _prepare(tmp_path, _Catalog(GRANTED))
    assert grant.roots == () and grant.scratch == ""
    assert grant.limitation.startswith("service_directories_unavailable:") and "is not a real directory" in grant.limitation
    assert not any(outside.iterdir())  # nothing was made through the link


def test_a_folder_inside_a_git_tree_or_around_the_copy_grants_nothing(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    inside = session_service_directories(
        _Catalog(GRANTED), "route", access="workspace_write", mode="agent", data_root=repo / "data",
        drive_root=repo / "data", task_id="task-1", key="snap-1", execution_root=tmp_path / "copy")
    assert inside.roots == () and f"is inside the Git working tree {repo.resolve()}" in inside.limitation
    copy = tmp_path / "data" / "task_drives" / "task-1" / SESSION_SCRATCH_DIRNAME / "snap-1" / "copy"
    copy.mkdir(parents=True)
    around = _prepare(tmp_path, _Catalog(GRANTED), execution=copy)
    assert around.roots == () and "overlaps the execution tree" in around.limitation


@pytest.mark.parametrize("task_id, key", [("", "snap-1"), ("../escape", "snap-1"), ("task-1", "a/b"),
                                          ("task-1", "..")])
def test_an_unowned_task_or_key_grants_nothing(tmp_path, task_id, key):
    grant = _prepare(tmp_path, _Catalog(GRANTED), task_id=task_id, key=key)
    assert grant.roots == () and grant.limitation.startswith("service_directories_unavailable:")
    assert not (tmp_path / "escape").exists() and not (tmp_path / "data" / "escape").exists()


def test_new_start_replaces_missing_scratch_but_never_adopts_a_replacement(tmp_path):
    first = _prepare(tmp_path, _Catalog(GRANTED))
    sibling = _prepare(tmp_path, _Catalog(GRANTED), key="sibling")
    other = _prepare(tmp_path, _Catalog(GRANTED), task_id="task-2")
    scratch = pathlib.Path(first.scratch)
    assert len({first.scratch, sibling.scratch, other.scratch}) == 3
    for root in first.roots:
        assert not (tmp_path / "data" / "task_drives").is_relative_to(root)
        assert not pathlib.Path(sibling.scratch).is_relative_to(root)
        assert not pathlib.Path(other.scratch).is_relative_to(root)
    scratch.rmdir()
    lost = _prepare(tmp_path, _Catalog(GRANTED))
    assert lost.roots and lost.scratch != first.scratch
    assert not scratch.exists()  # a new start allocates; exact retry never comes here
    replacement = pathlib.Path(lost.scratch)
    replacement.rmdir()
    _symlink_or_skip(replacement, pathlib.Path(sibling.scratch))
    replaced = _prepare(tmp_path, _Catalog(GRANTED))
    assert replaced.roots == () and "not the recorded scratch" in replaced.limitation


@pytest.mark.parametrize("layout", ["direct", "headless", "nested"])
def test_existing_drive_settlement_retires_only_its_short_scratch(tmp_path, layout):
    from ouroboros import task_custody
    from tests._shared import retire_synthetic_retention

    data = tmp_path / "data"
    first = _prepare(tmp_path, _Catalog(GRANTED))
    held = _prepare(tmp_path, _Catalog(GRANTED), key="snap-2")
    sibling = _prepare(tmp_path, _Catalog(GRANTED), task_id="task-2")
    drive = data / "task_drives" / "task-1"
    # Exercise the existing retirement decision, not a test-only cleanup entrypoint.
    assert task_custody.settle_child_drive(data, "task-1", drive, live=lambda _tid: True,
                                           admission_rollback=True)["status"] == "retained"
    assert pathlib.Path(first.scratch).exists()
    assert task_custody.settle_child_drive(data, "task-1", drive, live=lambda _tid: False,
                                           admission_rollback=True)["status"] == "removed"
    [retired] = (data / "state" / "custody_trash").iterdir()
    if layout != "direct":
        # The trash owner also receives these existing child layouts when the outer drive retires.
        destination = retired / ("data" if layout == "headless" else "data/task_drives/nested")
        destination.mkdir(parents=True)
        (retired / SESSION_SCRATCH_DIRNAME).rename(destination / SESSION_SCRATCH_DIRNAME)
    reason = "producer could not prove process exit"
    test_environment.retain_tree(held.scratch, reason)
    # The producer's retention is never deleted, and no longer holds the custodied drive in the trash.
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "retained": [held.scratch]}
    assert not retired.exists() and not pathlib.Path(first.scratch).exists()
    assert (pathlib.Path(held.scratch) / test_environment.RETENTION_MARKER).is_file()
    assert pathlib.Path(sibling.scratch).is_dir()
    assert all(pathlib.Path(root).is_dir() for root in first.roots[1:])  # caches have a different lifetime
    assert retire_synthetic_retention(held.scratch, reason)


def test_unrelated_data_link_does_not_keep_a_retired_drive(tmp_path):
    from ouroboros import task_custody

    data = tmp_path / "data"
    retired = data / "state/custody_trash/task-1-retired"
    retired.mkdir(parents=True)
    outside = tmp_path / "unrelated"
    outside.mkdir()
    (outside / "keep.txt").write_text("untouched", encoding="utf-8")
    _symlink_or_skip(retired / "data", outside)
    assert task_custody.remove_custody_trash(data) == {"removed": 1, "stuck": {}, "retained": []}
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "untouched"


def test_failed_scratch_removal_keeps_its_ownership_record(tmp_path, monkeypatch):
    from ouroboros import force_remove, task_custody

    grant = _prepare(tmp_path, _Catalog(GRANTED))
    data = tmp_path / "data"
    drive = data / "task_drives/task-1"
    assert task_custody.settle_child_drive(data, "task-1", drive, live=lambda _tid: False,
                                          admission_rollback=True)["status"] == "removed"
    [retired] = (data / "state/custody_trash").iterdir()
    original_remove = force_remove.force_rmtree
    monkeypatch.setattr(force_remove, "force_rmtree",
                        lambda path: None if pathlib.Path(path) == pathlib.Path(grant.scratch) else original_remove(path))
    assert task_custody.remove_custody_trash(data)["stuck"]
    assert pathlib.Path(grant.scratch).is_dir()
    assert (retired / SESSION_SCRATCH_DIRNAME / "snap-1/scratch.json").is_file()


def test_replaced_short_directory_is_neither_granted_nor_deleted(tmp_path):
    from ouroboros import task_custody

    grant = _prepare(tmp_path, _Catalog(GRANTED))
    scratch = pathlib.Path(grant.scratch)
    displaced = scratch.with_name(scratch.name + "-original")
    scratch.rename(displaced)  # retain the inode so the new directory cannot reuse it
    scratch.mkdir()
    (scratch / "foreign.txt").write_text("not owned\n", encoding="utf-8")
    assert _prepare(tmp_path, _Catalog(GRANTED)).roots == ()
    data = tmp_path / "data"
    assert task_custody.settle_child_drive(data, "task-1", data / "task_drives/task-1",
                                           live=lambda _tid: False, admission_rollback=True)["status"] == "removed"
    assert task_custody.remove_custody_trash(data)["stuck"]
    assert (scratch / "foreign.txt").read_text(encoding="utf-8") == "not owned\n"
    assert displaced.is_dir()


@pytest.mark.serial
@pytest.mark.parametrize("consumer", ["delegate", "reviewer"])
def test_real_bus_through_safe_test_uses_the_actual_short_grant(tmp_path, monkeypatch, consumer):
    """The previously failing AF_UNIX consumer, under the grant named in an ordinary start.
    No engine/OS-boundary claim: native qualification separately verifies the applied grant."""
    from tests.test_delegate_continuation import _start
    from ouroboros.test_evidence import evidence_directory

    monkeypatch.setattr(test_environment, "default_temp_parent", _PLATFORM_TEMP_PARENT)
    if consumer == "delegate":
        request, payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-socket",
                                              harness_row=GRANTED)
        assert payload["status"] == "started", payload
        drive = pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id
    else:
        from ouroboros import review_checking as rc
        from ouroboros.subagents import delegated_run_shape

        author, data, source = _frozen_author(tmp_path)
        request = rc.session_checking(_Catalog(GRANTED), custody_drive=data, task_id="task-1", operation_id="socket",
                                      root=str(author), source=source, shape=delegated_run_shape(False, "workspace_write"),
                                      route_id="route")
        drive = data / "task_drives" / "task-1"
    scratch = pathlib.Path(request["execution"]["auxiliaryWritableRoots"][0])
    assert scratch.parent == _PLATFORM_TEMP_PARENT().resolve()
    assert str(scratch) in request["instructions"] and scratch.parent.as_posix() not in request["execution"]["auxiliaryWritableRoots"]
    repo = pathlib.Path(__file__).resolve().parents[1]
    evidence = evidence_directory(repo)  # the outer safe_test's durable destination, not tmp_path
    command = [sys.executable, "-I", "-S", str(repo / "scripts/safe_test.py"), "--temp-parent", str(scratch),
               "--evidence-dir", str(evidence / "consumer"), "--timeout", "90", "--", sys.executable,
               "-m", "pytest", "-q", "-o", "addopts=", "tests/test_replay_membership_real_bus.py"]
    (evidence / "consumer-command.json").write_text(json.dumps(command), encoding="utf-8")
    (evidence / "start-request.json").write_text(json.dumps(request), encoding="utf-8")
    try:
        result = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=120)
        (evidence / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (evidence / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        assert result.returncode == 0, result.stdout + result.stderr
        [run] = (evidence / "consumer").iterdir()
        record = json.loads((run / "result.json").read_text(encoding="utf-8"))
        assert record["returncode"] == 0 and pathlib.Path(record["scratch"]).parent == scratch
    finally:
        custody._CUSTODY.clear()
        workspace_preflight.release_session_scratch(drive)


def test_the_facts_name_exactly_the_granted_places_and_a_limitation_as_one(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace_preflight, "_host_tools", lambda: "tools observed")
    copy = tmp_path / "copy"
    folder = copy / "pkg"
    folder.mkdir(parents=True)
    data = tmp_path / "data"
    grant = _prepare(tmp_path, _Catalog(GRANTED), execution=copy)
    body = session_environment_facts(copy, data_root=data, body=True, write_root=folder, service=grant)
    assert f"({scratch_fact(launcher=True, granted=grant.scratch)});" in body
    assert f"pass --temp-parent {grant.scratch}, this run's disposable scratch" in body
    assert (f"disposable process state (in {grant.scratch}, this run's disposable scratch), since "
            f"this session writes only in {folder}, a folder of the copy") in body
    assert f"the scratch {grant.scratch} for disposable process state" in body
    for (_name, variable), root in zip(CACHES, grant.roots[1:]):
        assert f"{variable}={root}" in body
    assert "granted to this run" in body and "your harness may not permit writing it" not in body
    assert scratch_fact() not in body and scratch_fact(launcher=True) not in body  # no second, unpermitted place
    foreign = session_environment_facts(copy, data_root=data, service=grant)
    assert "safe_test" not in foreign and grant.scratch in foreign
    limited = session_environment_facts(copy, data_root=data, service=ServiceDirectories(
        limitation="auxiliary_roots_unsupported: fixture"))
    assert "your engine grants this run none (auxiliary_roots_unsupported: fixture)" in limited
    assert "report that limitation rather than widen access or work around it" in limited
    assert "your harness may not permit writing it" in limited and "granted to this run" not in limited
    assert session_environment_facts(copy, data_root=data) == session_environment_facts(
        copy, data_root=data, service=ServiceDirectories())  # nothing negotiated: the facts as before


# --------------------------------------------------------------------------- delegated starts

def test_a_supported_private_copy_start_sends_and_names_exactly_its_directories(tmp_path, monkeypatch):
    from tests.test_delegate_continuation import _start

    request, payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-granted",
                                           prompt="Change and check it", harness_row=GRANTED)
    assert payload["status"] == "started", payload
    held = custody.replay(tmp_path)["run-granted"]
    data = tmp_path.resolve()  # this context's drive and canonical data root
    scratch = _scratch_for(data, ctx.task_id, held.snapshot_id)
    cache = data / PACKAGE_DOWNLOADS_DIRNAME
    roots = request["execution"]["auxiliaryWritableRoots"]
    assert roots == [str(scratch), *(str(cache / name) for name, _variable in CACHES)]
    assert all(pathlib.Path(root).is_dir() for root in roots)
    # The grant only extends the live execution; access, mode and the copy stay as they were.
    assert (request["access"], request["mode"]) == ("workspace_write", "agent")
    assert {key: request["execution"][key] for key in ("isolation", "delegated")} == {"isolation": "live",
                                                                                      "delegated": True}
    copy = pathlib.Path(request["execution"]["workspaceRoot"]).resolve()
    assert all(copy not in pathlib.Path(root).parents and pathlib.Path(root) not in copy.parents for root in roots)
    text = request["instructions"]
    binding = text.split("\n\nDELEGATED EXECUTION BINDING", 1)[1].split("\n\nPROJECT ENVIRONMENT", 1)[0]
    assert f"keep it outside every Git working tree (in {scratch}, this run's disposable scratch)" in binding
    assert f"the scratch {scratch} for disposable process state" in text
    for name, variable in CACHES:
        assert f"{variable}={cache / name}" in text
    assert scratch_fact() not in text  # the one scratch place named is the one the request permits
    # The exact bytes the engine received are the ones custody recorded before the POST.
    assert custody.invocation_record(tmp_path, held.invocation_id)["request"] == request


@pytest.mark.parametrize("row", [None, {"accessControl": {"auxiliaryWritableRoots": False}}],
                         ids=["engine-predates-row", "route-declines"])
def test_an_unsupported_route_starts_unchanged_and_hears_its_limitation(tmp_path, monkeypatch, row):
    from tests.test_delegate_continuation import _start

    request, payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-plain", harness_row=row)
    assert payload["status"] == "started", payload
    assert (request["access"], request["mode"]) == ("workspace_write", "agent")
    assert set(request["execution"]) == {"isolation", "delegated", "workspaceRoot"}  # no unsupported field
    text = request["instructions"]
    assert "your engine grants this run none (auxiliary_roots_unsupported:" in text
    assert "report that limitation rather than widen access or work around it" in text
    assert f"keep it outside every Git working tree ({scratch_fact()})" in text  # the default place, as before
    assert not (tmp_path / "task_drives" / ctx.task_id / SESSION_SCRATCH_DIRNAME).exists()
    assert not (tmp_path / PACKAGE_DOWNLOADS_DIRNAME).exists()


def test_a_readonly_start_negotiates_nothing(tmp_path, monkeypatch):
    from tests.test_delegate_continuation import _start

    _select_transport_access(monkeypatch, "readonly")
    request, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=False, run_id="run-read",
                                            harness_row=GRANTED)
    assert payload["status"] == "started", payload
    assert (request["access"], request["mode"]) == ("readonly", "ask") and "execution" not in request
    assert "auxiliaryWritableRoots" not in json.dumps(request) and "grants this run" not in request["instructions"]
    assert not list(tmp_path.glob(f"task_drives/*/{SESSION_SCRATCH_DIRNAME}"))
    assert not (tmp_path / PACKAGE_DOWNLOADS_DIRNAME).exists()


def test_an_exact_retry_replays_the_recorded_roots_and_prepares_nothing(tmp_path, monkeypatch):
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from tests.test_delegate_continuation import _start

    original, unknown, _ctx, _calls = _start(
        tmp_path, monkeypatch, acting=True, run_id="run-retry", prompt="Change and check it", harness_row=GRANTED,
        start_error=ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503))
    roots = original["execution"]["auxiliaryWritableRoots"]
    assert unknown.get("pending_invocation_id"), unknown

    def forbidden(*_args, **_kwargs):
        raise AssertionError("a retry replays its recorded request; it never prepares directories again")

    monkeypatch.setattr(workspace_preflight, "session_service_directories", forbidden)
    monkeypatch.setattr(workspace_preflight, "_owned_directory", forbidden)
    monkeypatch.setattr(workspace_preflight, "_session_scratch", forbidden)
    # The recorded scratch may have been retired before a late retry; no new path replaces it.
    pathlib.Path(roots[0]).rmdir()
    # Even with the route no longer negotiating, the replay sends the recorded bytes exactly.
    retry, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-retry",
                                          prompt="Change and check it",
                                          start_kwargs={"retry_of": unknown["pending_invocation_id"]})
    assert payload["status"] == "started", payload
    assert retry == original and retry["execution"]["auxiliaryWritableRoots"] == roots
    # Persistence canonicalizes object-key order; every value and ordered root remains exact.
    assert json.dumps(retry, ensure_ascii=False, sort_keys=True).encode() == json.dumps(original, ensure_ascii=False, sort_keys=True).encode()
    assert not pathlib.Path(roots[0]).exists()


def test_a_continuation_binds_the_rights_of_its_own_start(tmp_path, monkeypatch):
    from tests.test_delegate_continuation import _settle, _start

    first, _payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-first",
                                          harness_row=GRANTED)
    _settle(tmp_path, "run-first", ctx.task_id)
    # Same tree (the predecessor was never captured, so its checkout is still there), still negotiated:
    # this start grants again the scratch of the snapshot it runs in.
    same, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-next", harness_row=GRANTED,
                                         start_kwargs={"continue_from": "run-first"})
    assert payload["continuation"]["workspace"] == "same_snapshot" and same["continueFrom"] == "run-first"
    assert same["execution"]["auxiliaryWritableRoots"] == first["execution"]["auxiliaryWritableRoots"]
    _settle(tmp_path, "run-next", ctx.task_id)
    # The route stopped negotiating: the continuation's own whole execution carries no grant, which
    # revokes it (only an execution object omitted altogether keeps the predecessor's).
    revoked, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-third",
                                            start_kwargs={"continue_from": "run-next"})
    assert payload["status"] == "started", payload
    assert revoked["continueFrom"] == "run-next" and revoked["execution"]["isolation"] == "live"
    assert "auxiliaryWritableRoots" not in revoked["execution"]
    assert "your engine grants this run none (auxiliary_roots_unsupported:" in revoked["instructions"]
    _settle(tmp_path, "run-third", ctx.task_id)
    # A read-only continuation prepares nothing and replaces the old execution,
    # rather than inheriting its live geometry or auxiliary writable roots.
    reader, payload, _ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-reader",
                                           access="readonly", harness_row=GRANTED,
                                           start_kwargs={"continue_from": "run-third"})
    assert payload["status"] == "started", payload
    assert (reader["access"], reader["mode"]) == ("readonly", "ask")
    assert reader["execution"] == {
        "isolation": "envelope", "delegated": False, "auxiliaryWritableRoots": []}


def test_scratch_is_never_captured_retained_or_released_with_the_copy(tmp_path, monkeypatch):
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
    from tests.test_delegate_continuation import _settle, _start

    request, _payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-scratch",
                                            harness_row=GRANTED)
    copy = pathlib.Path(request["execution"]["workspaceRoot"])
    scratch = pathlib.Path(request["execution"]["auxiliaryWritableRoots"][0])
    (scratch / "env").mkdir()
    (scratch / "env" / "marker.txt").write_text("environment\n", encoding="utf-8")
    (copy / "product.txt").write_text("work\n", encoding="utf-8")
    (copy / ".review-evidence").mkdir()
    (copy / ".review-evidence" / "check.log").write_text("passed\n", encoding="utf-8")
    _settle(tmp_path, "run-scratch", ctx.task_id)
    try:
        block = capture_terminal_patch_for_drive(tmp_path, custody.replay(tmp_path)["run-scratch"])
    finally:
        custody._CUSTODY.clear()
    assert block["status"] == "ready_with_changes", block
    patch = pathlib.Path(block["patch_artifact"]).read_text(encoding="utf-8")
    assert "product.txt" in patch and "check.log" not in patch and "marker.txt" not in patch
    assert [row["path"] for row in block["named_evidence"]["files"]] == [".review-evidence/check.log"]
    assert str(scratch) not in json.dumps(block)
    # The copy's checkout went with its saved capture; the scratch has its own lifetime.
    assert block["checkout_released"] is True and not copy.exists()
    assert (scratch / "env" / "marker.txt").read_text(encoding="utf-8") == "environment\n"


def test_a_folder_session_gets_a_legal_environment_place_without_its_source(tmp_path, monkeypatch):
    from ouroboros.subagent_worktrees import find_execution_snapshot
    from tests import test_investigation_start_roots as roots_suite

    _negotiate(monkeypatch, roots_suite.Engine)
    project, folder = roots_suite._project(tmp_path)
    ctx, engine, start = roots_suite._consumer(tmp_path, monkeypatch, "workspace_write", workspace=folder)
    result = start()
    assert result["status"] == "started", result
    [request] = engine.starts
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    custody._CUSTODY.clear()
    clone = pathlib.Path(find_execution_snapshot(held.snapshot_id)["path"]).resolve()
    start_folder = clone / "pkg" / "component"
    scratch = _scratch_for(pathlib.Path(ctx.drive_root).resolve(), "child", held.snapshot_id)
    granted = [pathlib.Path(root) for root in request["execution"]["auxiliaryWritableRoots"]]
    assert granted[0] == scratch and scratch.is_dir()
    for root in granted:  # never the copy, its folder, the live project or anything around them
        for tree in (clone, start_folder, project.resolve(), folder.resolve()):
            assert root != tree and tree not in root.parents and root not in tree.parents
    text = request["instructions"]
    assert (f"disposable process state (in {scratch}, this run's disposable scratch), since this "
            f"session writes only in {start_folder}, a folder of the copy") in text
    assert "safe_test" not in text  # a foreign project is sent to no launcher of the body
    assert pathlib.Path(request["execution"]["workspaceRoot"]).resolve() == start_folder


def test_full_access_names_its_owned_disposable_scratch_and_sends_no_grant(tmp_path, monkeypatch):
    from ouroboros.delegate_start_instructions import DISPOSABLE_SCRATCH
    from tests import test_investigation_start_roots as roots_suite

    _negotiate(monkeypatch, roots_suite.Engine)
    _project, folder = roots_suite._project(tmp_path)
    ctx, engine, start = roots_suite._consumer(tmp_path, monkeypatch, "full", workspace=folder)
    result = start()
    assert result["status"] == "started", result
    held = custody.replay(ctx.drive_root)[result["run_id"]]
    custody._CUSTODY.clear()
    [request] = engine.starts
    assert request["access"] == "full" and set(request["execution"]) == {"isolation", "delegated", "workspaceRoot"}
    assert engine.grants == [str(folder)]  # the existing full-access trust, unchanged: nothing more is sent
    scratch = _scratch_for(pathlib.Path(ctx.drive_root).resolve(), "child", held.snapshot_id)
    text = request["instructions"]
    assert scratch.is_dir() and not any(scratch.iterdir())
    # Declared before use: where it is, that it is disposable, who removes it, what does not come back.
    assert f"disposable process state (in {scratch}, this run's disposable scratch), since this session" in text
    assert f"no engine grant is sent, since full access already reaches it: the scratch {scratch} for" in text
    assert DISPOSABLE_SCRATCH in text and "your full access reaches it" in text
    assert "grants this run" not in text and scratch_fact() not in text


def test_a_payload_copy_negotiates_nothing_and_stays_pure(tmp_path, monkeypatch):
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw
    from ouroboros.tools.delegate_integration import payload_content_hash
    from tests.test_delegated_skill_payload import _exact_payload_start, _StartStub
    from tests.test_payload_skill_contract import _payload_world

    class _Negotiating(_StartStub):
        def agent_capabilities(self):
            body = super().agent_capabilities()
            body["harnesses"][0].update(GRANTED)
            return body

    ctx, _head, skill = _payload_world(tmp_path, monkeypatch, "shared")
    seen: dict = {}
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Negotiating(seen))
    delegate._CUSTODY.clear()
    try:
        payload = json.loads(_exact_payload_start(ctx, "edit the skill", root="skill_payload", bucket="external",
                                                  skill_name="alpha"))
        assert payload["status"] == "started", payload
        request = seen["request"]
        assert "auxiliaryWritableRoots" not in json.dumps(request) and "grants this run" not in request["instructions"]
        assert not list(pathlib.Path(ctx.drive_root).glob(f"task_drives/*/{SESSION_SCRATCH_DIRNAME}"))
        assert payload_content_hash(pathlib.Path(request["scope"]["root"])) == payload_content_hash(skill)
    finally:
        delegate._CUSTODY.clear()


# --------------------------------------------------------------------------- writable reviewer seats

def _frozen_author(tmp_path):
    from ouroboros import review_checking as rc

    author = tmp_path / "author"
    author.mkdir()
    (author / "pyproject.toml").write_text('[project]\nname = "fixture"\nversion = "1"\n', encoding="utf-8")
    for args in (("init", "-q"), ("add", "-A"), ("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "seed")):
        subprocess.run(["git", *args], cwd=author, check=True, capture_output=True)
    data = tmp_path / "data"
    return author, data, rc.freeze_checking_source(author, data_root=data, subject=rc.SUBJECT_WORKTREE)


def test_a_supported_writable_seat_sends_names_and_sizes_exactly_its_directories(tmp_path):
    from ouroboros import review_checking as rc
    from ouroboros.review_execution import SessionInvocation
    from ouroboros.review_session_preparation import prepare_review_session_request
    from ouroboros.subagents import delegated_run_shape

    author, data, source = _frozen_author(tmp_path)
    shape = delegated_run_shape(False, "workspace_write")
    binding = rc.session_checking(_Catalog(GRANTED), custody_drive=data, task_id="task-1", operation_id="op-1",
                                  root=str(author), source=source, shape=shape, route_id="route")
    scratch = _scratch_for(data.resolve(), "task-1", f"seat-{rc.seat_dir(data, 'task-1', 'op-1').name}")
    cache = data.resolve() / PACKAGE_DOWNLOADS_DIRNAME
    execution = binding["execution"]
    assert execution["auxiliaryWritableRoots"] == [str(scratch), *(str(cache / name) for name, _v in CACHES)]
    area = pathlib.Path(execution["workspaceRoot"])
    text = binding["instructions"]
    assert f"goes outside every Git working tree (in {scratch}, this run's disposable scratch)" in text
    # The copy is a linked worktree: its changes are evidence, so no environment goes inside it.
    assert (f"in {scratch}, this run's disposable scratch, never inside this copy, since its changes are "
            "retained as evidence") in text and ".gitignore ignores it" not in text
    assert "linked worktree of its repository: branches, tags, stash and Git config are the author's" in text
    assert f"UV_CACHE_DIR={cache / 'uv'}" in text and "granted to this run" in text
    # The request is prepared and measured with the final instructions and the grant in it.
    delivery: dict = {}
    invocation = SessionInvocation(task_id="task-1", surface="multi_model_review", slot_id="t1", timeout_sec=60.0,
                                   access="workspace_write", source_delivery=delivery)
    route = SimpleNamespace(route_id="route", model="fixture-model", effort="low", profile_id="")
    request = prepare_review_session_request(invocation, route, prompt="review", root=str(author), thread_id="",
                                             schema_asked=False, shape=shape, execution=execution,
                                             instructions=binding["instructions"],
                                             input_limit={"prompt_budget": 10 ** 6})
    assert request["execution"] == execution and request["instructions"] == binding["instructions"]
    assert delivery["input_limit"]["prompt_chars"] == len("review") + len(binding["instructions"])
    assert delivery["first_send_bytes"] == len(json.dumps(request, ensure_ascii=False).encode("utf-8"))
    # Its evidence is the copy's named folder; the scratch is neither retained nor released with the copy.
    (scratch / "pip-tmp").write_text("disposable\n", encoding="utf-8")
    (area / ".review-evidence").mkdir()
    (area / ".review-evidence" / "check.log").write_text("passed\n", encoding="utf-8")
    receipt = rc.capture_seat_evidence(data, "task-1", "op-1")
    assert receipt["status"] == "retained", receipt
    assert [row["path"] for row in receipt["named_evidence"]] == [".review-evidence/check.log"]
    assert str(scratch) not in json.dumps(receipt) and not area.exists()
    assert (scratch / "pip-tmp").read_text(encoding="utf-8") == "disposable\n"


@pytest.mark.parametrize("row", [None, {"accessControl": {"auxiliaryWritableRoots": False}}],
                         ids=["engine-predates-row", "route-declines"])
def test_unsupported_seats_keep_their_request_and_rights(tmp_path, row):
    from ouroboros import review_checking as rc
    from ouroboros.subagents import delegated_run_shape

    author, data, source = _frozen_author(tmp_path)
    binding = rc.session_checking(_Catalog(row), custody_drive=data, task_id="task-1", operation_id="op-1",
                                  root=str(author), source=source, shape=delegated_run_shape(False, "workspace_write"),
                                  route_id="route")
    area = rc.seat_dir(data, "task-1", "op-1") / "area.noindex"
    assert binding["execution"] == {"isolation": "live", "delegated": True, "workspaceRoot": str(area)}
    text = binding["instructions"]
    assert "your engine grants this run none (auxiliary_roots_unsupported:" in text
    assert "granted to this run" not in text and f"({scratch_fact()})" in text
    assert not list(data.glob(f"task_drives/*/{SESSION_SCRATCH_DIRNAME}"))
    assert not (data / PACKAGE_DOWNLOADS_DIRNAME).exists()
    rc.capture_seat_evidence(data, "task-1", "op-1")


def test_a_full_access_seat_owns_named_disposable_scratch_that_its_evidence_never_carries(tmp_path):
    from ouroboros import review_checking as rc
    from ouroboros.delegate_start_instructions import DISPOSABLE_SCRATCH
    from ouroboros.subagents import delegated_run_shape

    author, data, source = _frozen_author(tmp_path)
    gateway = _Catalog(GRANTED)
    gateway.ensure_full_access = lambda root: None
    binding = rc.session_checking(gateway, custody_drive=data, task_id="task-1", operation_id="op-1",
                                  root=str(author), source=source, shape=delegated_run_shape(False, "full"),
                                  route_id="route")
    area = rc.seat_dir(data, "task-1", "op-1") / "area.noindex"
    assert binding["execution"] == {"isolation": "live", "delegated": True, "workspaceRoot": str(area)}
    assert gateway.reads == 0  # nothing to negotiate at full access
    scratch = _scratch_for(data.resolve(), "task-1", f"seat-{rc.seat_dir(data, 'task-1', 'op-1').name}")
    text = binding["instructions"]
    assert f"goes outside every Git working tree (in {scratch}, this run's disposable scratch)" in text
    assert f"in {scratch}, this run's disposable scratch, never inside this copy" in text
    assert DISPOSABLE_SCRATCH in text and "granted to this run" not in text and scratch_fact() not in text
    assert not (data / PACKAGE_DOWNLOADS_DIRNAME).exists()
    (scratch / "venv-marker").write_text("disposable\n", encoding="utf-8")
    (area / ".review-evidence").mkdir()
    (area / ".review-evidence" / "check.log").write_text("passed\n", encoding="utf-8")
    receipt = rc.capture_seat_evidence(data, "task-1", "op-1")
    assert receipt["status"] == "retained" and str(scratch) not in json.dumps(receipt), receipt
    assert (scratch / "venv-marker").is_file()  # it goes with the review task's drive, not with the seat


def test_a_thread_seat_claims_no_grant_its_turns_cannot_carry(tmp_path):
    from ouroboros.review_session_preparation import seat_service_directories
    from ouroboros.review_thread_continuity import start_review_thread_turn
    from ouroboros.subagents import delegated_run_shape

    catalog = _Catalog(GRANTED)
    thread = seat_service_directories(catalog, "route", delegated_run_shape(False, "workspace_write"), tmp_path,
                                      "task-1", "op-1", str(tmp_path / "area"), "plan_review:seat")
    assert thread.roots == () and thread.limitation.startswith("thread_turns_carry_no_grant:")
    assert seat_service_directories(catalog, "route", delegated_run_shape(False, "full"), tmp_path, "task-1",
                                    "op-1", str(tmp_path / "area"), "plan_review:seat") == ServiceDirectories()
    assert catalog.reads == 0 and not (tmp_path / "task_drives").exists()
    sent = start_review_thread_turn(SimpleNamespace(start_thread_turn=lambda _tid, body, **_kw: body), "thread-1", {
        "prompt": "p", "execution": {"isolation": "live", "auxiliaryWritableRoots": ["/fixture"]}},
        idempotency_key="key")
    assert sent["execution"] == {"isolation": "live"}  # a turn carries geometry only
