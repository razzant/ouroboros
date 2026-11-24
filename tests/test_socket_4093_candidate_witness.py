"""#4093 on a GitHub-hosted Linux runner: the NORMAL local in-place scratch lifecycle, observed.

What this file witnesses, end to end, with the application's own policy intact:

* the host issues the run's disposable scratch by its own ordinary path, records it, and names
  exactly that one place in the instructions the engine received;
* the place handed out is INSIDE the fresh, private synthetic home this run created, and
  that home's whole physical ancestry from "/" qualifies for a strict consumer: every
  ancestor is a real directory owned by root or this user and writable by nobody but its
  owner. That is the property #4093's consumer needs, and it is checked PHYSICALLY, never by
  reading the application's names;
* the socket is real: a fresh, independent, benign process (stdlib only, ``-I -S``) re-reads
  the exact instructions the engine received, takes the ONE granted scratch out of them,
  re-checks every physical ancestor itself, and binds ONE AF_UNIX socket at the exact
  absolute address, under Linux's real 108-byte bound;
* the release is the host's ordinary one: after the run's ordinary completion event and its
  ordinary task result, the host retires exactly that scratch container (socket and all) and
  leaves a sibling beside it untouched.

What is deliberately NOT done here (and why): the placement decision and the byte budget
belong to the application and are never monkeypatched — the runner's own home chain satisfies
them, so the application's real policy decides everything. Only the ENGINE is a double (the
synthetic transport of the existing delegated-transport suite): no vendor session, no
credential, no network. The consumer is a benign process, not the strict consumer of the issue.

Nothing about the private parent's NAME is assumed: the witness takes the container from the
host's own ownership record and the granted workspace from the instructions the engine
received. It never looks for a ".obs" directory, and it never guesses a branch or a tree id.

Ordering rules this file obeys, so that a failure means what it says:

* the containment check (the granted container is strictly inside the fresh synthetic home)
  happens BEFORE any sibling is written and BEFORE any separate process is started. If the
  application placed the scratch outside that home, this test stops there and writes
  nothing outside;
* no path that came from the binding or from an ownership record is ever deleted by hand.
  The only directory this test removes itself is the synthetic home it created.

Linux only: the application budgets a private parent against AF_UNIX's real ``sun_path``, so
the witness needs the runner's short private home chain.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import pytest

from ouroboros import task_custody
from ouroboros.test_environment import SCRATCH_WORKSPACE
from ouroboros.workspace_preflight import SESSION_SCRATCH_DIRNAME
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
)
from tests.test_continuation_fresh_snapshot_binding import _GRANT_ROW, _ctx, _leg
from tests.test_delegate_continuation import _settle

LINUX = pytest.mark.skipif(sys.platform != "linux",
                           reason="this witness is a Linux-only CI check (the runner's private home chain)")

# The child: stdlib only, run with ``-I -S``. It knows two things — the exact instructions the
# engine received and the exact scratch the host issued — and derives everything else from the
# physical chain itself.
_CHILD = r'''
import json, os, re, socket, stat, sys

instructions, expected = sys.argv[1], sys.argv[2]
text = open(instructions, encoding="utf-8").read()

# 1. The grant: exactly ONE scratch is named in the in-place binding, and it is the one the
#    host issued. The private parent's name is never assumed.
binding = text.split("DELEGATED IN-PLACE BINDING", 1)[1].split("\n\n", 1)[0]
found = re.findall(r"in (\S+), this run's disposable scratch", binding)
assert found == [expected], f"the in-place binding names {found!r}, not exactly the one issued scratch {expected!r}"
assert len(re.findall(r"this run's disposable scratch", text)) == 1, "the instructions name a scratch more than once"

# 2. The path: absolute, physical, present, and the empty directory the host made.
assert os.path.isabs(expected), f"not an absolute path: {expected}"
assert os.path.realpath(expected) == expected, f"not a physical path: {expected}"
assert os.path.isdir(expected), f"the granted scratch is not a directory: {expected}"
assert os.listdir(expected) == [], f"the granted scratch is not empty: {os.listdir(expected)}"

# 3. The permission checks, re-read here from the physical chain: every component from "/"
#    down to the scratch is a real directory of root or this user, and none is writable by
#    group or others. This is the rule a strict socket consumer applies before it binds,
#    and it is the property #4093 is about.
chain, walk = [], expected
while True:
    chain.append(walk)
    parent = os.path.dirname(walk)
    if parent == walk:
        break
    walk = parent
assert chain[-1] == "/", chain
for walk in chain:
    seen = os.lstat(walk)
    assert stat.S_ISDIR(seen.st_mode), f"not a real directory: {walk}"
    assert seen.st_uid in (0, os.geteuid()), f"not owned by root or this user: {walk}"
    assert not seen.st_mode & 0o022, f"writable by group or others: {walk} ({oct(seen.st_mode & 0o777)})"
assert os.access(expected, os.W_OK | os.X_OK), "the granted scratch is not writable by this user"

# 4. The socket: ONE AF_UNIX socket, at the exact absolute address, under Linux's real bound.
sun = 108  # Linux's sun_path, the NUL included
address = os.path.join(expected, "ctl")
size = len(os.fsencode(address))
assert size < sun, f"the address does not fit the real bound: {size} >= {sun}"
sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
try:
    sock.bind(address)
finally:
    sock.close()
assert stat.S_ISSOCK(os.lstat(address).st_mode), "the bound path is not a socket"
assert sorted(os.listdir(expected)) == ["ctl"], os.listdir(expected)
print(json.dumps({"placed": "absolute", "socket": True, "address": address,
                  "entries": sorted(os.listdir(expected)), "sun_path": sun, "address_bytes": size,
                  "ancestry": chain}))
'''


def _runner_home() -> pathlib.Path:
    """The runner account's STABLE home, exported by the workflow before safe_test replaces HOME.

    The process HOME is never used: under ``scripts/safe_test.py`` it belongs to the
    boundary, not to the runner account, and the boundary lives under a disposable temp root
    the application refuses for a private parent. The workflow records the real home in
    ``SOCKET_4093_RUNNER_HOME``; nothing here guesses a path layout.
    """
    override = os.environ.get("SOCKET_4093_RUNNER_HOME", "").strip()
    assert override, ("SOCKET_4093_RUNNER_HOME is not set: export the runner account's stable home "
                      "(the workflow does this before safe_test replaces HOME)")
    home = pathlib.Path(override).resolve()
    assert home.is_dir(), f"the runner's stable home is not a directory: {home}"
    return home


def _fresh_private_home(stable: pathlib.Path) -> pathlib.Path:
    """A fresh, empty, private (0700) home this run owns, inside the runner's stable home.

    The name is short on purpose: ``tempfile.mkdtemp`` would otherwise add its default
    "tmp" prefix to the eight random characters, and every extra byte of the private
    parent's path is charged against the application's own AF_UNIX byte budget. An empty
    prefix and suffix give the eight-character name alone.

    The fresh home is never the live HOME, never a temp path and never an existing profile.
    """
    home = pathlib.Path(tempfile.mkdtemp(dir=stable, prefix="", suffix="")).resolve()
    os.chmod(home, 0o700)
    return home


def _owner_records(ctx) -> list[pathlib.Path]:
    """The host's own ownership records for this task's scratch: one per granted container."""
    owners = pathlib.Path(ctx.drive_root) / "task_drives" / ctx.task_id / SESSION_SCRATCH_DIRNAME
    return sorted(owners.glob("*/scratch.json"))


def _granted(ctx) -> tuple[pathlib.Path, pathlib.Path]:
    """The container the host recorded, and the workspace it granted inside it.

    The container comes from the host's own record; the workspace is the product's own
    ``SCRATCH_WORKSPACE`` inside it. No private-parent name (".obs" or anything else) and no
    remembered tree id enter this decision.
    """
    records = _owner_records(ctx)
    assert len(records) == 1, f"expected exactly one scratch ownership record, found {records}"
    row = json.loads(records[0].read_text(encoding="utf-8"))
    container = pathlib.Path(row["path"])
    inside = container / SCRATCH_WORKSPACE
    workspace = inside if inside.is_dir() else container
    assert workspace.is_dir(), f"the recorded scratch is not a directory: {workspace}"
    return container, workspace


def _require_inside_home(container: pathlib.Path, workspace: pathlib.Path, home: pathlib.Path) -> None:
    """The issued container must sit strictly inside the fresh synthetic home.

    This runs before any sibling is written and before any separate process is started:
    if the application placed the scratch outside that home, this witness stops here
    instead of writing a socket or a sibling somewhere else.
    """
    real_home = home.resolve()
    real_container = container.resolve()
    real_workspace = workspace.resolve()
    assert real_container != real_home, (
        f"the issued scratch IS the synthetic home itself: {real_container}")
    assert real_container.is_relative_to(real_home), (
        f"the issued scratch is outside the fresh synthetic home: {real_container} not under {real_home}")
    assert real_workspace == real_container or real_workspace.is_relative_to(real_container), (
        f"the granted workspace is not inside the issued container: {real_workspace} not under {real_container}")


@LINUX
def test_a_full_in_place_run_gets_a_private_scratch_a_real_socket_binds_in_it_and_the_host_releases_exactly_it(
        tmp_path, monkeypatch):
    """The #4093 normal-use witness on Linux: the real selector, creator, allocator, binding
    text, child process, socket and release. Doubled on purpose: the ENGINE only (the synthetic
    transport of the delegated-transport suite). Placement, the byte budget, the ancestor
    judgement and the AF_UNIX bound are the application's own, unmocked.
    """
    from ouroboros.workspace_preflight import release_finished_scratch
    from tests._store_shared import seed_result

    stable = _runner_home()
    home = _fresh_private_home(stable)
    container = sibling = None
    try:
        # The only environment change this test makes: the HOME the app reads. Everything the
        # app decides afterwards — placement, ancestors, the bound — is decided by the app.
        monkeypatch.setenv("HOME", str(home))
        ctx, data = _ctx(tmp_path, "git"), tmp_path.resolve()
        request, payload = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="edit here", access="full",
                                row=_GRANT_ROW, workspace="in_place")
        assert payload["workspace"] == "in_place" and payload["status"] == "started", payload
        assert "auxiliaryWritableRoots" not in request["execution"]  # ownership, never a new grant

        container, workspace = _granted(ctx)
        assert workspace.is_absolute() and workspace == workspace.resolve(), workspace

        # Before anything is written or started: the scratch the host issued must be inside
        # the fresh synthetic home this run created.
        _require_inside_home(container, workspace, home)

        # The host named exactly that one place in the instructions the engine received.
        binding = request["instructions"].split("DELEGATED IN-PLACE BINDING", 1)[1]
        assert f"in {workspace}, this run's disposable scratch" in binding, binding

        # A sibling beside the container: the host owns the container, never this folder.
        sibling = container.parent / f"sibling-{os.getpid()}"
        sibling.mkdir(mode=0o700)
        (sibling / "mine.txt").write_text("kept\n", encoding="utf-8")

        text, script = tmp_path / "instructions.txt", tmp_path / "child.py"
        text.write_text(request["instructions"], encoding="utf-8")
        script.write_text(_CHILD, encoding="utf-8")
        done = subprocess.run([sys.executable, "-I", "-S", str(script), str(text), str(workspace)],
                              capture_output=True, text=True, timeout=120)
        assert done.returncode == 0, done.stdout + done.stderr
        facts = json.loads(done.stdout.splitlines()[-1])
        assert facts["placed"] == "absolute" and facts["socket"] is True and facts["entries"] == ["ctl"], facts
        assert facts["address"] == str(workspace / "ctl"), facts
        assert facts["address_bytes"] < facts["sun_path"], facts

        # The ordinary end of an ordinary task: the run's terminal event, its task result, then
        # the host's own release of the scratch it owns.
        _settle(tmp_path, "run-1", ctx.task_id, state="succeeded", reason="")
        seed_result(data, ctx.task_id, {"status": "completed"})
        assert release_finished_scratch(data, lambda _task: False)["released"]
        assert task_custody.remove_custody_trash(data)["removed"] == 1
        assert not container.exists()  # exactly that container, socket and all
        assert sibling.is_dir() and (sibling / "mine.txt").read_text(encoding="utf-8") == "kept\n"
    finally:
        # Only the synthetic home this test created is removed by hand. No path that came
        # from the binding or from an ownership record is deleted here: the host's ordinary
        # release is what retires the container.
        shutil.rmtree(home, ignore_errors=True)
