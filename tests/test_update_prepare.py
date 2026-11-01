"""Preparing a unified Update beside the serving checkout: real Git, a miniature body.

The serving checkout is never written: its HEAD, branch, index file and working files
stay byte-identical through preparation, a conflicting target and a refusal. The
candidate is one installation-owned copy record whose checkout the GC retires only once
it is settled.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from ouroboros import subagent_worktrees
from supervisor import update_prepare
from tests.body_candidate_support import git, stage_then_edit_in_one_second
from tests.test_update_activation import STAGED, UNSTAGED, Scene

pytestmark = pytest.mark.serial


@pytest.fixture
def scene(tmp_path, monkeypatch):
    import supervisor.state as state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 0.0)  # no resolver starts in these tests
    return Scene(tmp_path, monkeypatch)


def _serving_bytes(scene):
    """The serving checkout as bytes: HEAD, the real index FILE, and every working file."""
    index = pathlib.Path(git(scene.serving, "rev-parse", "--absolute-git-dir")) / "index"
    files = {path.relative_to(scene.serving).as_posix(): path.read_bytes()
             for path in sorted(scene.serving.rglob("*")) if path.is_file() and ".git" not in path.parts}
    return {"head": git(scene.serving, "rev-parse", "HEAD"), "index": index.read_bytes(),
            "index_mtime": index.stat().st_mtime_ns, "files": files}


def test_preparation_reads_three_layers_without_writing_the_serving_checkout(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work()
    before = _serving_bytes(scene)

    row = scene.prepare()

    assert _serving_bytes(scene) == before  # not even an index refresh
    assert not (pathlib.Path(git(scene.serving, "rev-parse", "--absolute-git-dir")) / "MERGE_HEAD").exists()
    assert row["state"] == "ready" and row["kind"] == "update_candidate" and row["dirty"] is True
    layers = row["layers"]
    assert git(scene.serving, "show", f"{layers['new_index']}:ouroboros/mod_b.py") + "\n" == STAGED
    assert git(scene.serving, "show", f"{layers['new_files']}:ouroboros/mod_b.py") + "\n" == UNSTAGED
    assert git(scene.serving, "show", f"{layers['new_files']}:local-scratch.txt") == "mine, untracked"
    assert git(scene.serving, "show", f"{layers['new_commit']}:ouroboros/mod_a.py") == "GEN = 'GEN_CAND'"
    refs = git(scene.serving, "for-each-ref", "--format=%(refname)", row["refs"])
    assert {line.rsplit("/", 1)[1] for line in refs.splitlines()} == {"base", "k1", "k2", "c", "i_prime", "w_prime"}
    assert git(scene.serving, "branch", "--list", "candidate/*").strip().startswith("+ candidate/update-")
    projection = update_prepare.projection()
    assert projection["state"] == "ready" and projection["actions"] == ["activate", "discard_candidate"]
    assert projection["local_changes"] is True and projection["merge_kind"] == "clean"


@pytest.mark.parametrize("flag", ["skip-worktree", "assume-unchanged", "intent-to-add"])
def test_index_states_git_cannot_carry_refuse_with_their_paths(scene, flag):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    if flag == "intent-to-add":
        (scene.serving / "ouroboros" / "planned.py").write_text("PLANNED = 1\n")
        git(scene.serving, "add", "-N", "ouroboros/planned.py")
        path = "ouroboros/planned.py"
    else:
        git(scene.serving, "update-index", f"--{flag}", "ouroboros/mod_b.py")
        path = "ouroboros/mod_b.py"
    before = _serving_bytes(scene)
    with pytest.raises(update_prepare.PrepareRefused) as refused:
        scene.prepare()
    assert refused.value.code == "UPDATE_LOCAL_STATE_UNSUPPORTED"
    assert path in refused.value.text and flag.replace("-", "_") in refused.value.text
    assert _serving_bytes(scene) == before and update_prepare.current() is None


def test_conflicting_target_waits_in_its_workspace_and_serving_is_untouched(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_OFFICIAL'\n"})
    (scene.serving / "ouroboros" / "mod_a.py").write_text("GEN = 'GEN_LOCAL'\n")
    git(scene.serving, "commit", "-q", "-am", "local change of the same line")
    before = _serving_bytes(scene)

    row = scene.prepare()

    assert _serving_bytes(scene) == before
    # No model budget in this fixture: the adaptation waits instead of starting a resolver.
    assert row["state"] == "paused_budget" and row["resolver"]["conflicts"] == ["ouroboros/mod_a.py"]
    workspace = pathlib.Path(row["path"])
    assert "<<<<<<<" in (workspace / "ouroboros" / "mod_a.py").read_text()  # the conflict lives only here
    assert git(workspace, "rev-parse", "HEAD") == row["resolver"]["m0"]
    projection = update_prepare.projection()
    assert "activate" not in projection["actions"]
    # The merge with this installation's own commits is where this conflict is.
    assert projection["layer_conflicts"] == {"merge": ["ouroboros/mod_a.py"]} and projection["conflict_count"] == 1
    assert projection["merge_kind"] == "conflicting" and projection["adaptation"] == {"step": "merge"}


def test_a_layer_conflict_offers_the_explicit_update_without_local_changes(scene):
    scene.official({"ouroboros/mod_b.py": "GEN = 'GEN_OFFICIAL'\n"})
    (scene.serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_UNSTAGED'\n")
    before = _serving_bytes(scene)

    row = scene.prepare()

    assert row["state"] == "paused_budget" and row["resolver"]["step"] == "files"
    assert row["resolver"]["conflicts"] == ["ouroboros/mod_b.py"]
    projection = update_prepare.projection()
    assert "apply_without_local_changes" in projection["actions"]
    # The merge itself had nothing to combine: the conflict is the owner's uncommitted file, never a local commit's.
    assert projection["merge_kind"] == "fast_forward" and projection["adaptation"] == {"step": "files"}
    assert projection["layer_conflicts"] == {"files": ["ouroboros/mod_b.py"]}
    plain = scene.prepare(without_local_changes=True)
    assert _serving_bytes(scene) == before
    assert plain["state"] == "ready" and plain["without_local_changes"] is True
    c_tree = git(scene.serving, "rev-parse", f"{plain['layers']['new_commit']}^{{tree}}")
    assert plain["layers"]["new_index"] == plain["layers"]["new_files"] == c_tree
    # The owner's dropped work stays reachable on the candidate's private refs.
    assert git(scene.serving, "show", f"{plain['refs']}k2:ouroboros/mod_b.py") == "GEN = 'GEN_UNSTAGED'"
    assert update_prepare.find(row["candidate_id"])["state"] == "superseded"


def test_the_gc_retires_only_a_settled_candidate_with_its_checkout_and_refs(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    first = scene.prepare()
    second = scene.prepare()
    assert update_prepare.find(first["candidate_id"])["state"] == "superseded"

    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task: False)

    assert not pathlib.Path(first["path"]).exists() and update_prepare.find(first["candidate_id"]) is None
    assert git(scene.serving, "for-each-ref", first["refs"]) == ""
    assert pathlib.Path(second["path"]).is_dir() and update_prepare.current()["candidate_id"] == second["candidate_id"]
    update_prepare.discard(update_prepare.current())
    subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task: False)
    assert not pathlib.Path(second["path"]).exists() and update_prepare.current() is None


def test_space_is_measured_and_a_shortage_refuses_with_numbers(scene, monkeypatch):
    import collections

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    (scene.serving / "big-untracked.bin").write_bytes(b"x" * 4096)
    usage = collections.namedtuple("usage", "total used free")
    monkeypatch.setattr(update_prepare.shutil, "disk_usage", lambda _path: usage(10**6, 10**6 - 100, 100))
    with pytest.raises(update_prepare.PrepareRefused) as refused:
        scene.prepare()
    assert refused.value.code == "UPDATE_SPACE" and "untracked files 4096" in refused.value.text
    assert "100 free" in refused.value.text and update_prepare.current() is None


def test_a_target_the_checkout_already_contains_is_current(scene):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    git(scene.serving, "merge", "-q", "--ff-only", "official")
    with pytest.raises(update_prepare.PrepareRefused) as refused:
        scene.prepare()
    assert refused.value.code == "UPDATE_CURRENT"


@pytest.mark.parametrize("targets, prefix, launcher, kind", [
    (["--user"], "base", "handoff", "user_site"),  # the packaged interpreter's user site
    ([], "own-venv", "", "venv"),  # the owner's own virtual environment
    ([], "base", "", "private"),  # a bare interpreter the server runs without a launcher: a venv of its own
    ([], "base", "handoff", "private"),  # the same under a launcher that hands itself over to new packages
    ([], "base", "predating", ""),  # under an older launcher, whose restart installs the lock into its own packages
    ([], "private", "handoff", "private"),  # the private environment an Update selected, serving
])
def test_the_serving_packages_seam_is_a_user_site_a_venv_a_private_environment_or_none(monkeypatch, tmp_path, targets,
                                                                                      prefix, launcher, kind):
    import sys

    from ouroboros import platform_layer

    fixed = tmp_path / "data" / "state" / "python-private"
    prefixes = {"base": tmp_path / "base", "own-venv": tmp_path / "own-venv", "private": fixed}
    monkeypatch.setattr(platform_layer, "pip_install_target_args", lambda _python: targets)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "prefix", str(prefixes[prefix]))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "base"))
    monkeypatch.setenv("PYTHONUSERBASE", str(tmp_path / "userbase"))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    for name, value in (("OUROBOROS_MANAGED_BY_LAUNCHER", launcher and "1"),
                        (update_prepare.LAUNCHER_HANDOFF_ENV, launcher == "handoff" and "1")):
        monkeypatch.setenv(name, value) if value else monkeypatch.delenv(name, raising=False)
    found, path = update_prepare.serving_environment()
    assert found == kind
    assert path == {"user_site": tmp_path / "userbase", "private": fixed}.get(kind, prefixes[prefix])
    # Windows renames a user site and a venv under their running processes (measured on NTFS); a system Python's
    # entries run in their private environment as that venv's base Python (the hook's waiting parent, the launcher).
    monkeypatch.setattr(sys, "platform", "win32")
    assert update_prepare.serving_environment()[0] == kind


def test_a_venv_lock_installs_with_the_cloned_venvs_own_interpreter(monkeypatch, tmp_path):
    from supervisor import update_checks

    calls = []
    monkeypatch.setattr(update_checks, "run_tracked",
                        lambda cmd, cwd, timeout, env=None: calls.append((cmd, env)) or {"returncode": 0})
    target = tmp_path / "venv.update-x"
    update_prepare._install_lock(target, tmp_path / "lock.txt", "venv")
    cmd, env = calls[0]
    assert cmd[:4] == [str(target / "bin" / "python"), "-m", "pip", "install"] and "--user" not in cmd
    assert env["PYTHONNOUSERSITE"] == "1" and not {"PYTHONUSERBASE", "VIRTUAL_ENV", "PIP_USER"} & set(env)


def test_selection_points_pip_written_scripts_at_the_path_the_venv_runs_from(tmp_path):
    """Both forms distlib writes; links and unrelated files stay as they are; a resumed pass changes nothing."""
    from ouroboros import body_switch

    current, generation = tmp_path / "venv", tmp_path / "venv.update-x"
    (current / "bin").mkdir(parents=True)  # the generation, renamed into place by the switch
    plain = current / "bin" / "tool"
    plain.write_bytes(f"#!{generation}/bin/python\nimport tool\n".encode())
    plain.chmod(0o755)
    trampoline = current / "bin" / "long-tool"
    trampoline.write_bytes(f"#!/bin/sh\n'''exec' \"{generation}/bin/python\" \"$0\" \"$@\"\n' '''\n".encode())
    (current / "bin" / "python").symlink_to("/usr/bin/python3")
    (current / "bin" / "activate").write_text(f"VIRTUAL_ENV={current}\n")
    for _pass in range(2):
        body_switch._retarget_scripts(str(current), str(generation), str(current))
        assert plain.read_text().splitlines()[0] == f"#!{current}/bin/python" and plain.stat().st_mode & 0o111
        assert f'"{current}/bin/python"' in trampoline.read_text() and str(generation) not in trampoline.read_text()
    assert os.readlink(current / "bin" / "python") == "/usr/bin/python3"
    assert (current / "bin" / "activate").read_text() == f"VIRTUAL_ENV={current}\n"


def test_a_ready_candidate_lists_its_carried_paths_by_layer_and_keeps_the_versions_it_pinned(scene):
    """The work as the restart leaves it, read from the candidate's own trees; the versions as prepared."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "VERSION": "1.0.1\n"})
    scene.local_work()  # one local commit; mod_b staged, then edited again; one untracked file

    row = scene.prepare()

    assert row["state"] == "ready"
    projection = update_prepare.projection(latest_sha="f" * 40)  # the channel has moved on since
    assert (projection["target_version"], projection["base_version"]) == ("1.0.1", "1.0.0")
    assert projection["target_sha"] == scene.target and projection["newer_available"] is True
    assert projection["merge_kind"] == "clean" and projection["local_commits"] == 1
    assert projection["carried"] == {"index": {"count": 1, "paths": [["M", "ouroboros/mod_b.py"]]},
                                     "files": {"count": 2, "paths": [["A", "local-scratch.txt"], ["M", "ouroboros/mod_b.py"]]},
                                     "list": str(update_prepare.carried_list_path(row))}
    layers = row["layers"]  # exactly the differences of the candidate's own layer trees, C -> I' -> W'
    assert git(scene.serving, "diff-tree", "-r", "--name-only", layers["new_commit"], layers["new_index"]).split() == [
        "ouroboros/mod_b.py"]
    # Leaving the work out carries nothing, and says so with exact zero counts.
    plain = scene.prepare(without_local_changes=True)
    assert (plain["carried"]["index"]["count"], plain["carried"]["files"]["count"]) == (0, 0)
    assert plain["carried"]["index"]["paths"] == plain["carried"]["files"]["paths"] == []
    assert "list" not in plain["carried"] and not update_prepare.carried_list_path(plain).exists()


def _many_untracked(scene, count=250):
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    (scene.serving / "notes").mkdir()
    for index in range(count):
        (scene.serving / "notes" / f"n{index:03d}.md").write_text(f"note {index}\n")
    return scene.prepare()


def test_more_carried_paths_than_the_record_keeps_are_listed_once_in_full_for_download(scene, monkeypatch):
    """250 real untracked files: the record keeps the first 200 in Git's order and the exact count; every one, the
    later 50 included, is in the one list beside the copy, which its candidate's download serves and the GC removes."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway import files, update_activation

    row = _many_untracked(scene)

    monkeypatch.delenv("OUROBOROS_FILE_BROWSER_DEFAULT", raising=False)  # and the test client is not a loopback one
    carried = update_activation.candidate_view()["carried"]
    layer = carried["files"]
    assert row["state"] == "ready" and layer["count"] == 250 and len(layer["paths"]) == update_prepare._CARRIED_LIMIT
    assert ["A", "notes/n249.md"] not in layer["paths"]  # a later path the record itself does not hold
    listing = pathlib.Path(carried["list"])
    assert listing == update_prepare.carried_list_path(row) and listing.parent == pathlib.Path(row["path"]).parent
    assert carried["list_url"] == f"/api/files/download?carried_list={row['candidate_id']}"
    served = TestClient(Starlette(routes=[Route("/api/files/download", files.api_files_download)])).get(
        carried["list_url"])
    assert served.status_code == 200 and served.content == listing.read_bytes()
    lines = served.text.splitlines()
    every = [line.split("\t") for line in lines[lines.index("In the files: 250") + 1:]]
    assert len(every) == 250 and ["A", "notes/n249.md"] in every and ["A", "notes/n200.md"] in every
    assert every[:update_prepare._CARRIED_LIMIT] == layer["paths"]  # the record's excerpt is the list's beginning
    assert "Staged: 0" in lines and carried["index"] == {"count": 0, "paths": []}
    update_prepare.release_environment(row)  # what the GC calls once it retired the copy
    assert not listing.exists() and "list_url" not in update_activation.candidate_view()["carried"]


def _http(base, path, headers):
    """``(status, headers, body)`` of one real GET; an error status is an answer here, not an exception."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers), timeout=30) as reply:
            return reply.status, dict(reply.headers), reply.read()
    except urllib.error.HTTPError as reply:
        return reply.code, dict(reply.headers), reply.read()


def test_a_phone_signed_in_over_the_network_downloads_the_whole_list_without_a_files_root(scene, monkeypatch):
    """The actual consumers over real HTTP: the gateway's own routes behind the network gate, and a client that gate
    counts as remote (uvicorn's trusted loopback proxy forwards its address, as for a phone behind a tunnel), with no
    Files root and then with one that leaves the copies out. The Files link of before fails for it; the candidate's
    own download serves all 250 changes byte for byte, by that candidate's id and nothing else. The desktop
    launcher's unchanged file bridge saves the same bytes."""
    import json
    import socket
    import threading
    import time
    from urllib.parse import urlencode

    import uvicorn
    from starlette.applications import Starlette

    from ouroboros.gateway import files
    from ouroboros.gateway.router import collect_routes
    from ouroboros.server_auth import NetworkAuthGate
    from tests.test_ui_smoke_large_artifacts import _native_file_api

    row = _many_untracked(scene)
    listing = update_prepare.carried_list_path(row)
    monkeypatch.setattr(pathlib.Path, "home", classmethod(lambda cls: scene.tmp))  # the copies live under HOME
    monkeypatch.delenv("OUROBOROS_FILE_BROWSER_DEFAULT", raising=False)
    monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "phone-secret")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    app = NetworkAuthGate(Starlette(routes=collect_routes(data_dir=scene.data)))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning", forwarded_allow_ips="127.0.0.1"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(.01)
        assert server.started
        port = sock.getsockname()[1]
        phone = {"X-Forwarded-For": "198.51.100.23"}  # the gate sees this address: a remote client
        signed = {**phone, "X-Ouroboros-Password": "phone-secret"}

        def get(path, headers=signed):
            return _http(f"http://127.0.0.1:{port}", path, headers)

        def refusal(path):
            status, _headers, body = get(path)
            return status, json.loads(body).get("reason_code")

        # The panel's status names the link, under the same sign-in as every Updates request.
        assert get("/api/update/status", phone)[0] == 401
        status, _headers, body = get("/api/update/status")
        assert status == 200
        url = json.loads(body)["candidate"]["carried"]["list_url"]
        assert url == "/api/files/download?" + urlencode({"carried_list": row["candidate_id"]})
        assert get(url, phone)[0] == 401
        # The Files link of before needs a Files root this client does not have; the candidate's download does not.
        before = files.download_url_for_local_file(listing)
        assert before.startswith("/api/files/download?path=") and get(before)[0] == 400
        status, headers, served = get(url)
        assert status == 200 and served == listing.read_bytes()
        assert headers["content-disposition"] == f'attachment; filename="{listing.name}"'
        lines = served.decode().splitlines()
        every = [line.split("\t") for line in lines[lines.index("In the files: 250") + 1:]]
        assert len(every) == 250 and ["A", "notes/n249.md"] in every and ["A", "notes/n200.md"] in every
        # A Files root that leaves the copies out has no link for the list at all; the download stays the same.
        shared = scene.tmp / "shared"
        shared.mkdir()
        monkeypatch.setenv("OUROBOROS_FILE_BROWSER_DEFAULT", str(shared))
        assert files.download_url_for_local_file(listing) == "" and get(url)[2] == served
        # The desktop app: its launcher's file bridge, unchanged, takes this address and saves the same bytes.
        api, _digest = _native_file_api(port, [])
        saved = api.download_file_to_downloads(url, listing.name)
        assert saved["ok"] is True and pathlib.Path(saved["path"]).read_bytes() == served
        # That candidate's own list only: no other selector beside it, no other id or path, never a link elsewhere.
        cid = row["candidate_id"]
        for query in ("carried_list=", f"carried_list={cid}&path=notes", f"carried_list={cid}&carried_list={cid}",
                      f"carried_list={cid}&upload=x"):
            assert get(f"/api/files/download?{query}")[0] == 400, query
        for other in (str(listing), "../../worktrees", "update_000000000000_000000", cid.upper()):
            assert refusal("/api/files/download?" + urlencode({"carried_list": other})) == (409, "candidate_changed")
        kept = listing.with_name("kept.txt")
        listing.rename(kept)
        listing.symlink_to(scene.serving / "ouroboros" / "mod_a.py")
        assert refusal(url) == (404, "list_unavailable")
        listing.unlink()
        kept.rename(listing)
        update_prepare.release_environment(row)  # the GC's hook removes the list; the candidate is still live
        assert refusal(url) == (404, "list_unavailable")
        update_prepare.discard(update_prepare.current())
        assert refusal(url) == (409, "candidate_changed")
    finally:
        server.should_exit = True
        thread.join(10)
        sock.close()
        assert not thread.is_alive()


def _conflicted_twice(scene):
    """A, paused at a conflict of an uncommitted file and offering the exclusion; then B, prepared again meanwhile."""
    scene.official({"ouroboros/mod_b.py": "GEN = 'GEN_OFFICIAL'\n"})
    (scene.serving / "ouroboros" / "mod_b.py").write_text("GEN = 'GEN_UNSTAGED'\n")
    first = scene.prepare()
    second = scene.prepare()
    assert update_prepare.find(first["candidate_id"])["state"] == "superseded"
    assert second["state"] == "paused_budget" and "apply_without_local_changes" in update_prepare.projection()["actions"]
    return first, second


def test_leaving_the_local_work_out_is_bound_to_the_candidate_it_was_asked_for(scene):
    """A request from a screen that showed A changes nothing once B is current; the same request for B works."""
    first, second = _conflicted_twice(scene)
    before, rows = _serving_bytes(scene), update_prepare._rows()

    with pytest.raises(update_prepare.PrepareRefused) as stale:
        update_prepare.admit(first["candidate_id"])
    assert stale.value.code == "UPDATE_CANDIDATE_CHANGED" and not update_prepare._running.locked()
    with pytest.raises(update_prepare.PrepareRefused) as again:
        scene.prepare(without_local_changes=True, expected=first["candidate_id"])
    assert again.value.code == "UPDATE_CANDIDATE_CHANGED" and not update_prepare._running.locked()
    assert update_prepare._rows() == rows and _serving_bytes(scene) == before  # no copy, no record change

    plain = scene.prepare(without_local_changes=True, expected=second["candidate_id"])
    assert plain["state"] == "ready" and plain["without_local_changes"] is True
    assert update_prepare.find(second["candidate_id"])["state"] == "superseded" and _serving_bytes(scene) == before


def test_a_candidate_discarded_while_its_exclusion_was_admitted_is_refused_at_the_admission(scene, monkeypatch):
    """The request passed its check; another device discards A before the new copy is admitted: nothing replaces it."""
    first, second = _conflicted_twice(scene)
    before = _serving_bytes(scene)
    update_prepare.admit(second["candidate_id"])  # the request's own check passes and holds the slot
    provision = update_prepare._provision

    def discarded_meanwhile(*args):
        row = provision(*args)
        update_prepare.discard(update_prepare.find(second["candidate_id"]))  # the other device
        return row

    monkeypatch.setattr(update_prepare, "_provision", discarded_meanwhile)
    with pytest.raises(update_prepare.PrepareRefused) as changed:
        scene.prepare(without_local_changes=True, expected=second["candidate_id"], admitted=True)

    assert changed.value.code == "UPDATE_CANDIDATE_CHANGED" and not update_prepare._running.locked()
    assert update_prepare.current() is None and _serving_bytes(scene) == before
    made = [row for row in update_prepare._rows() if row["candidate_id"] not in (first["candidate_id"],
                                                                                second["candidate_id"])]
    assert [row["state"] for row in made] == ["discarded"] and made[0]["reason"].startswith("UPDATE_CANDIDATE_CHANGED")
    assert update_prepare.find(second["candidate_id"])["state"] == "discarded"  # by its owner, never superseded


def test_an_activation_that_began_while_the_copy_was_made_discards_it_and_keeps_the_candidate_current(scene,
                                                                                                    monkeypatch):
    """The admission's own refusal keeps its answer: an activation authorized while the new copy was made discards
    that copy at once and the candidate being activated stays current. The preparation raises that refusal like any
    other end without a candidate of its own, and the status says it beside the activating candidate (Astra F4)."""
    from ouroboros import body_adoption

    monkeypatch.setattr(update_prepare, "_last_refusal", {})  # this test's own panel memory
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    ready = scene.prepare()
    provision = update_prepare._provision

    def activation_meanwhile(*args):  # the owner's Restart to update, authorized once the copy exists
        row = provision(*args)
        body_adoption.authorize_update(scene.data, update_prepare.find(ready["candidate_id"]))
        return row

    monkeypatch.setattr(update_prepare, "_provision", activation_meanwhile)
    with pytest.raises(update_prepare.PrepareRefused) as refused:
        scene.prepare()

    made = [row for row in update_prepare._rows() if row["candidate_id"] != ready["candidate_id"]]
    assert [(row["state"], row["reason"]) for row in made] == [
        ("discarded", "UPDATE_ACTIVATION_PENDING: an activation started meanwhile")]
    assert refused.value.code == "UPDATE_ACTIVATION_PENDING" and not update_prepare._running.locked()
    assert update_prepare.current()["candidate_id"] == ready["candidate_id"]
    view = update_prepare.projection()
    assert (view["candidate_id"], view["refusal"]["code"]) == (ready["candidate_id"], "UPDATE_ACTIVATION_PENDING")


def _ended_worker():
    """The stamp a previous generation left: ``_worker()`` as another process wrote it; that process has exited."""
    import json
    import subprocess
    import sys

    code = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
            "from supervisor.update_prepare import _worker; print(json.dumps(_worker()))")
    done = subprocess.run([sys.executable, "-c", code, str(pathlib.Path(__file__).resolve().parents[1])],
                          capture_output=True, text=True, timeout=120, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_preparation_its_process_left_mid_step_offers_retry_or_discard_after_the_next_boot(scene):
    """Astra C1: a step runs on a thread of the serving process. When that process dies in the checks, the record
    stays ``checking`` with no action and the GC keeps it. The next boot's settlement (``settle_on_boot``, also with
    no handoff) fails it as interrupted once its worker provably ended, never while that worker may still run."""
    from ouroboros import body_adoption

    class Died(BaseException):  # the process ends inside the checks: no step records an outcome
        pass

    def checks(_row):
        raise Died

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work()
    with pytest.raises(Died):
        update_prepare.prepare(fetch=False, checks=checks)
    row = update_prepare.current()
    assert row["state"] == "checking" and update_prepare.projection()["actions"] == []
    assert row["worker"]["pid"] == os.getpid()
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)
    assert update_prepare.current()["state"] == "checking"  # its worker is this very process: not proven over
    retired = subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task: False)
    assert retired["removed"] == 0 and update_prepare.find(row["candidate_id"]) is not None

    update_prepare._update(row, worker=_ended_worker())  # as the generation that has since exited left it
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=True)

    projection = update_prepare.projection()
    assert projection["state"] == "failed" and projection["reason"].startswith("UPDATE_PREPARE_INTERRUPTED: ")
    assert projection["actions"] == ["prepare", "apply_without_local_changes", "discard_candidate"]
    assert update_prepare.reconcile_interrupted(scene.data) == []  # settled once
    update_prepare.discard(update_prepare.find(row["candidate_id"]))
    assert subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task: False)["removed"] == 1


def test_a_providerless_generation_retains_its_update_outcome_without_minting_supervisor_state(scene, monkeypatch):
    """Native q7ba/q7be: no provider, so no supervisor and no ``init_state``; its state is uninitialized and nothing
    may be written into it. The Last attempt is the Update's own record, and no supervisor state appears."""
    from ouroboros import body_adoption, body_switch
    from supervisor import state

    for name, path in {"DRIVE_ROOT": scene.data, "STATE_PATH": scene.data / "state/state.json",
                       "STATE_LAST_GOOD_PATH": scene.data / "state/state.last_good.json",
                       "STATE_LOCK_PATH": scene.data / "locks/state.lock"}.items():
        monkeypatch.setattr(state, name, path)
    with pytest.raises(state.StateUnavailable) as unavailable:
        state.update_state(lambda live: live)
    assert unavailable.value.reason == "uninitialized"  # the native runs' condition
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "VERSION": "1.0.1\n"})
    row = scene.prepare()
    scene.activate(row)
    assert scene.boot().returncode == 0
    body_switch._attest_loaded(str(scene.serving), scene.handoff())

    # As the providerless boot calls it: no supervisor, its recovery's readiness stands in (``server``).
    body_adoption.settle_on_boot(scene.data, scene.serving, supervisor_ready=False, providerless_ready=lambda: True)

    assert update_prepare.find(row["candidate_id"])["state"] == "activated"
    last = update_prepare.projection()["last_activation"]
    assert (last["outcome"], last["target_version"], last["candidate_id"]) == ("adopted", "1.0.1", row["candidate_id"])
    assert not (scene.data / "state" / "state.json").exists()
    assert not (scene.data / "state" / "state.last_good.json").exists()


def _app_candidate(scene, candidate_id, version, previous):
    """A candidate whose exchange left the replaced app as ``.<App>-<previous>-backup.app`` beside the app."""
    import time

    apps = scene.tmp / "Applications"
    installed, backup = apps / "Ouroboros.app", apps / f".Ouroboros-{previous}-backup.app"
    for app in (installed, backup):
        (app / "Contents").mkdir(parents=True, exist_ok=True)
    row = {"kind": "update_candidate", "candidate_id": candidate_id, "task_id": candidate_id, "state": "activating",
           "path": str(scene.tmp / "copies" / candidate_id), "repo_dir": str(scene.serving), "created_at": time.time(),
           "refs": f"refs/ouroboros/update/{candidate_id}/", "target": {"version": version},
           "native": {"status": "staged", "version": version, "previous_version": previous,
                      "installed": str(installed), "backup_app": str(backup)}}
    subagent_worktrees._put_record(row)
    return row, backup


def test_an_adopted_private_environment_is_recorded_so_its_absence_never_installs_into_the_interpreter(scene):
    """Once an Update adopted a bare interpreter's private environment, the retained record says so: that
    environment gone at a later cold entry or after a return is refused like a broken one, never the
    interpreter's own packages. Only an installation that never selected one keeps installing there."""
    import time

    from ouroboros import runtime_venv

    fixed = scene.data / "state" / runtime_venv.PRIVATE_NAME
    cid = "update_dddddddddddd_000004"
    row = {"kind": "update_candidate", "candidate_id": cid, "task_id": cid, "state": "activating",
           "path": str(scene.tmp / "copies" / cid), "repo_dir": str(scene.serving), "created_at": time.time(),
           "refs": f"refs/ouroboros/update/{cid}/", "target": {"version": "7.9.0"},
           "environment": {"generation": {"kind": "private", "current": str(fixed)}}}
    subagent_worktrees._put_record(row)
    bare = str(scene.tmp / "system" / "bin" / "python3")
    assert runtime_venv.dependency_python(scene.data, scene.serving, bare) == bare  # never selected: as always

    update_prepare.record_activation(scene.data, {"outcome": "adopted", "candidate_id": cid})

    assert update_prepare.retained_fact("private_environment", scene.data) == {"root": str(fixed)}
    update_prepare._forget_activation()  # the next admitted preparation keeps it
    assert update_prepare.retained_fact("private_environment", scene.data) == {"root": str(fixed)}
    assert "private_environment" not in (update_prepare.projection().get("last_activation") or {})
    with pytest.raises(runtime_venv.RuntimeVenvRefused, match="is missing"):
        runtime_venv.dependency_python(scene.data, scene.serving, bare)  # gone: never the interpreter's packages


def test_a_confirmed_app_adoption_retires_the_backup_an_earlier_one_left(scene):
    """Astra A4: every app replacement leaves its backup beside the app; once a later app adoption is confirmed,
    the earlier backup goes, so at most one stays (PERSISTENCE). A return removes nothing."""
    first, old = _app_candidate(scene, "update_aaaaaaaaaaaa_000001", "7.8.0", "7.7.0")
    update_prepare.record_activation(scene.data, {"outcome": "adopted", "candidate_id": first["candidate_id"]})
    assert old.is_dir()  # the only backup: kept for recovery
    update_prepare._forget_activation()  # the next admitted preparation
    second, new = _app_candidate(scene, "update_bbbbbbbbbbbb_000002", "7.9.0", "7.8.0")
    unrelated = old.parent / ".Ouroboros-notes"
    unrelated.mkdir()

    update_prepare.record_activation(scene.data, {"outcome": "adopted", "candidate_id": second["candidate_id"]})

    assert not old.exists() and new.is_dir() and unrelated.is_dir() and (old.parent / "Ouroboros.app").is_dir()
    assert update_prepare.projection()["last_activation"]["native"]["backup_app"] == str(new)
    third, _ = _app_candidate(scene, "update_cccccccccccc_000003", "7.10.0", "7.9.0")
    update_prepare.record_activation(scene.data, {"outcome": "returned", "candidate_id": third["candidate_id"]})
    assert new.is_dir()


def test_a_preparation_starts_no_child_once_the_generation_stop_began_and_settles_typed(scene, monkeypatch):
    """Producer -> Stop -> next spawn. A preparation still running when an owner Restart enters this generation's
    stop door (``owned_shutdown.begin_owned_stop``) starts no further child: every Git, copy and native-tool start
    goes through the shell's tracked admission, which that door closed. Its record settles ``failed`` with the typed
    reason, so the next start offers preparing again; the steps before the door ran their children as usual."""
    import subprocess

    from ouroboros import owned_shutdown

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # this test's own generation
    started = []

    class Recorded(subprocess.Popen):  # every child this process starts, with whether the door had opened
        def __init__(self, args, *rest, **kwargs):
            started.append((list(args) if isinstance(args, (list, tuple)) else [args], owned_shutdown.stop_begun()))
            super().__init__(args, *rest, **kwargs)

    real_merge = update_prepare._merge

    def merge_then_restart(*args, **kwargs):  # the owner's Restart enters its door once C exists
        merged = real_merge(*args, **kwargs)
        owned_shutdown.begin_owned_stop(scene.data, cause="owner_restart")
        return merged

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work()
    monkeypatch.setattr(subprocess, "Popen", Recorded)
    monkeypatch.setattr(update_prepare, "_merge", merge_then_restart)

    row = scene.prepare()

    assert any(args[:1] == ["git"] for args, stopping in started if not stopping)  # the workspace and C were made
    assert [args for args, stopping in started if stopping] == []
    assert row["state"] == "failed" and row["reason"].startswith(
        "UPDATE_PREPARE_STOPPED: this server generation's stop has begun"), row["reason"]
    assert update_prepare.find(row["candidate_id"])["state"] == "failed"
    assert "prepare" in update_prepare.projection()["actions"]


@pytest.mark.parametrize("door", ["shut", "opens_in_the_resolver"])
def test_the_carrier_step_of_the_merge_starts_its_git_through_the_tracked_admission(scene, monkeypatch, door):
    """Producer -> Stop -> next spawn inside C's carrier step. VERSION conflicts (this installation released 1.1.0, the
    official target is 1.0.1), so the shared carrier resolver runs Git in the workspace (three ``show``, ``merge-file``,
    ``add``). Door shut: it resolves VERSION to the official side and C is clean. When an owner Restart enters this
    generation's stop door once the resolver read VERSION's stages, its later steps start no child: VERSION stays
    unmerged with Git's markers in the workspace, and the record fails typed, never as a conflict to adapt."""
    import subprocess

    from ouroboros import owned_shutdown
    from ouroboros.tools import release_sync

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # this test's own generation
    started, real_popen, real_substitute = [], subprocess.Popen, release_sync.substitute_carrier_spans

    class Recorded(subprocess.Popen):  # every child this process starts, with whether the door had opened
        def __init__(self, args, *rest, **kwargs):
            started.append((list(args) if isinstance(args, (list, tuple)) else [args], owned_shutdown.stop_begun()))
            super().__init__(args, *rest, **kwargs)

    def substitute(*args, **kwargs):  # called once the resolver holds the three stages of the file
        if door != "shut" and not owned_shutdown.stop_begun():
            owned_shutdown.begin_owned_stop(scene.data, cause="owner_restart")
        return real_substitute(*args, **kwargs)

    scene.official({"VERSION": "1.0.1\n", "ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    (scene.serving / "VERSION").write_text("1.1.0\n")
    git(scene.serving, "commit", "-q", "-am", "local release")
    monkeypatch.setattr(release_sync, "substitute_carrier_spans", substitute)
    monkeypatch.setattr(subprocess, "Popen", Recorded)
    row = scene.prepare()
    monkeypatch.setattr(subprocess, "Popen", real_popen)

    carrier = [args[3:] for args, _stopping in started if args[:2] == ["git", "-C"]]
    assert [step[:2] for step in carrier[:3]] == [["show", f":{stage}:VERSION"] for stage in (1, 2, 3)]
    workspace = pathlib.Path(row["path"])
    if door == "shut":
        assert [step[:2] for step in carrier[3:5]] == [["merge-file", "-p"], ["add", "--"]]
        assert row["state"] == "ready" and row["merge"]["kind"] == "clean", row.get("reason")
        assert row["merge"]["carrier_paths"] == ["VERSION"]
        assert git(scene.serving, "show", f"{row['layers']['new_commit']}:VERSION") == "1.0.1"
        return
    assert [args for args, stopping in started if stopping] == []
    assert row["state"] == "failed" and row["reason"].startswith(
        "UPDATE_PREPARE_STOPPED: this server generation's stop has begun"), row["reason"]
    assert len(git(workspace, "ls-files", "-u", "--", "VERSION").splitlines()) == 3  # never staged after the door
    assert "<<<<<<<" in (workspace / "VERSION").read_text()


def _copies(scene):
    """The folder a new copy's checkout is born in (``unindexed.copy_path``)."""
    return scene.tmp / "worktrees" / "copies.noindex"


def deadline_at_checkout(monkeypatch):
    """Give a new copy's checkout (``reset --hard``, 300 s live) a deadline it cannot meet: the real Git starts and the
    bounded Git owner ends it, answering as at any deadline. Returns the undo."""
    from supervisor import git_ops

    bounded = git_ops._run_git_process_bounded

    def owner(cmd, *, timeout, **kwargs):
        if cmd == ["git", "reset", "--hard", "--quiet", "--no-recurse-submodules"]:
            timeout = 1e-6
        return bounded(cmd, timeout=timeout, **kwargs)

    monkeypatch.setattr(git_ops, "_run_git_process_bounded", owner)
    return lambda: monkeypatch.setattr(git_ops, "_run_git_process_bounded", bounded)


@pytest.mark.parametrize("step", ["remove", "add", "reset"])
def test_a_copy_whose_checkout_fails_after_its_record_settles_at_once_and_preparing_again_works(scene, monkeypatch,
                                                                                                step):
    """Whole-Astra review: the copy's record is written before its checkout exists, and each later step failed with
    that record ``preparing`` while this process lived, so nothing settled it and the panel offered no action. Each
    fails here through its real owner: the generation's stop door shut before the stale registration is removed,
    Git itself unable to make the checkout (the copies folder is a file), the checkout's ``reset --hard`` ended at
    the bounded Git owner's deadline. The copy settles at once with its typed reason, as the admission settles a
    copy it refuses; preparing again is offered, the serving checkout is untouched, the GC retires the copy with its
    branch, and preparing again (the next generation, for the stop) makes a ready candidate."""
    from ouroboros import owned_shutdown

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    scene.local_work()
    before, put, undo = _serving_bytes(scene), subagent_worktrees._put_record, lambda: None
    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # this test's own generation

    def put_then_stop(row, *args):  # the owner's Restart enters its door once the copy is recorded
        put(row, *args)
        if row.get("kind") == update_prepare.KIND and not owned_shutdown.stop_begun():
            owned_shutdown.begin_owned_stop(scene.data, cause="owner_restart")

    if step == "remove":
        monkeypatch.setattr(subagent_worktrees, "_put_record", put_then_stop)
    elif step == "add":
        _copies(scene).parent.mkdir(parents=True, exist_ok=True)
        _copies(scene).write_text("a file where the copies are born\n")
    else:
        undo = deadline_at_checkout(monkeypatch)

    with pytest.raises(Exception) as failed:
        scene.prepare()

    rows = update_prepare._rows()
    assert [row["state"] for row in rows] == ["discarded"], rows  # never left preparing while this process lives
    assert isinstance(failed.value, update_prepare.PrepareRefused), failed.value
    assert failed.value.code == ("UPDATE_PREPARE_STOPPED" if step == "remove" else "UPDATE_WORKSPACE_FAILED")
    assert {"remove": "stop has begun", "add": "git worktree add exited", "reset": "timed out"}[step] in \
        failed.value.text and rows[0]["reason"] == f"{failed.value.code}: {failed.value.text}"
    projection = update_prepare.projection()
    assert (projection["state"], projection["actions"]) == ("none", ["prepare"])
    assert update_prepare.reconcile_interrupted(scene.data) == [] and _serving_bytes(scene) == before

    monkeypatch.setattr(owned_shutdown, "_GENERATION_STOP", owned_shutdown._Stop())  # the next generation
    monkeypatch.setattr(subagent_worktrees, "_put_record", put)
    undo()
    if step == "add":
        _copies(scene).unlink()
    assert subagent_worktrees.prune_orphans(retention_days=0, live=lambda _task: False)["removed"] == 1
    assert update_prepare._rows() == [] and git(scene.serving, "branch", "--list", "candidate/*") == ""
    assert not pathlib.Path(rows[0]["path"]).exists()
    again = scene.prepare()
    assert again["state"] == "ready", again.get("reason")
    assert update_prepare.projection()["actions"] == ["activate", "discard_candidate"]
    assert _serving_bytes(scene) == before


def test_a_path_holding_unrecorded_work_refuses_before_any_record_and_keeps_that_work(scene, monkeypatch):
    """A refusal before the copy's record exists changes nothing, as always: the unrecorded folder at the copy's
    path keeps its files, and no record, branch or private ref is made."""
    import types
    import uuid

    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})
    monkeypatch.setattr(update_prepare, "uuid", types.SimpleNamespace(uuid4=lambda: uuid.UUID(hex="abcdef" + "0" * 26)))
    taken = _copies(scene) / f"update_{scene.target[:12]}_abcdef"
    taken.mkdir(parents=True)
    (taken / "notes.txt").write_text("unrecorded work\n")
    before = _serving_bytes(scene)

    with pytest.raises(update_prepare.PrepareRefused) as refused:
        scene.prepare()

    assert refused.value.code == "UPDATE_CANDIDATE_PATH_OCCUPIED" and update_prepare._rows() == []
    assert (taken / "notes.txt").read_text() == "unrecorded work\n" and _serving_bytes(scene) == before
    assert git(scene.serving, "branch", "--list", "candidate/*") == ""
    assert git(scene.serving, "for-each-ref", update_prepare.REF_PREFIX) == ""


def _listing_snapshot(repo):
    """The c9b46e48 measurement of the same layers: a scratch index from the real index LISTING (no stat data)."""
    import tempfile

    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    listing = subprocess_run(["git", "ls-files", "-s", "-z"], repo, env).stdout
    with tempfile.TemporaryDirectory() as scratch:
        env = {**env, "GIT_INDEX_FILE": str(pathlib.Path(scratch) / "index")}
        subprocess_run(["git", "-c", "core.splitIndex=false", "read-tree", "--empty"], repo, env)
        subprocess_run(["git", "-c", "core.splitIndex=false", "update-index", "-z", "--index-info"], repo, env, listing)
        index_tree = subprocess_run(["git", "-c", "core.splitIndex=false", "write-tree"], repo, env).stdout.strip()
        subprocess_run(["git", "-c", "core.splitIndex=false", "add", "-A", "--", "."], repo, env)
        files_tree = subprocess_run(["git", "-c", "core.splitIndex=false", "write-tree"], repo, env).stdout.strip()
    return index_tree.decode(), files_tree.decode()


def subprocess_run(argv, cwd, env, data=None):
    import subprocess

    return subprocess.run(argv, cwd=str(cwd), env=env, input=data, capture_output=True, check=True)


@pytest.mark.parametrize("split", [False, True], ids=["whole_index", "split_index"])
def test_the_snapshot_reads_one_copy_of_the_real_index_and_writes_nothing_beside_it(scene, split):
    """The layers come from a byte copy of the real index, whose stat data spares ``add -A`` rehashing unchanged
    files. A split index keeps its shared part in the repository: the copy reads it there and is written whole, so
    no shared-index file appears, disappears or changes beside the real one, which stays byte- and time-identical.
    I and W are exactly what the listing-based measurement makes of the same checkout."""
    if split:
        git(scene.serving, "config", "core.splitIndex", "true")
        git(scene.serving, "update-index", "--split-index")
    scene.local_work()  # a commit, a staged edit, a further unstaged edit and an untracked file
    for number in range(30):  # enough entries that a split index keeps most of them in its shared part
        (scene.serving / "docs" / f"note{number}.txt").parent.mkdir(exist_ok=True)
        (scene.serving / "docs" / f"note{number}.txt").write_text(f"note {number}\n")
    git(scene.serving, "add", "docs")
    (scene.serving / "docs" / "note3.txt").write_text("edited after staging\n")
    git_dir = pathlib.Path(git(scene.serving, "rev-parse", "--absolute-git-dir"))
    shared = sorted(git_dir.glob("sharedindex.*"))
    assert bool(shared) == split
    index = git_dir / "index"
    # Any read of a split index freshens its shared part's time (Git's own expiry clock): bytes are what count there.
    before = (index.read_bytes(), index.stat().st_mtime_ns, {path.name: path.read_bytes() for path in shared})

    snap = update_prepare.snapshot(scene.serving, "ouroboros")

    after = (index.read_bytes(), index.stat().st_mtime_ns, {path.name: path.read_bytes() for path in shared})
    assert after == before and sorted(git_dir.glob("sharedindex.*")) == shared  # nothing written beside it
    assert snap["refusals"] == [] and snap["dirty"] is True
    assert (snap["index_tree"], snap["files_tree"]) == _listing_snapshot(scene.serving)
    assert git(scene.serving, "show", f"{snap['index_tree']}:docs/note3.txt") == "note 3"
    assert git(scene.serving, "show", f"{snap['files_tree']}:docs/note3.txt") == "edited after staging"
    assert git(scene.serving, "show", f"{snap['files_tree']}:local-scratch.txt") == "mine, untracked"


@pytest.mark.parametrize("split", [False, True], ids=["whole_index", "split_index"])
def test_a_same_size_edit_in_the_second_its_index_was_written_reaches_the_files_layer(scene, split):
    """Git checks an entry stamped in its index file's own second by content (racily clean), so it sees a same-size
    edit made then. The copy keeps the real index's time, so W carries that edit as Git does and I keeps what was
    staged; the real index keeps its bytes and its time. A copy stamped with its own, later time trusted the stale
    stat data and lost the edit from W. LABELLED CONTROLLED TIMING (``stage_then_edit_in_one_second``): the stage
    and the edit share one second, and the snapshot starts after it."""
    if split:
        git(scene.serving, "config", "core.splitIndex", "true")
        git(scene.serving, "update-index", "--split-index")
    if not stage_then_edit_in_one_second(scene.serving, "ouroboros/mod_a.py", "GEN = 'GEN_OLD'\n", "GEN = 'GEN_NEW'\n"):
        pytest.skip("the stage and the edit never fit one second here: the race was not set")
    git_dir = pathlib.Path(git(scene.serving, "rev-parse", "--absolute-git-dir"))
    index, shared = git_dir / "index", sorted(git_dir.glob("sharedindex.*"))
    assert bool(shared) == split
    before = (index.read_bytes(), index.stat().st_mtime_ns, {path.name: path.read_bytes() for path in shared})

    snap = update_prepare.snapshot(scene.serving, "ouroboros")

    assert (index.read_bytes(), index.stat().st_mtime_ns, {path.name: path.read_bytes() for path in shared}) == before
    assert snap["refusals"] == [] and snap["dirty"] is True
    assert git(scene.serving, "show", f"{snap['files_tree']}:ouroboros/mod_a.py") == "GEN = 'GEN_NEW'"
    assert git(scene.serving, "show", f"{snap['index_tree']}:ouroboros/mod_a.py") == "GEN = 'GEN_OLD'"
    assert (snap["index_tree"], snap["files_tree"]) == _listing_snapshot(scene.serving)  # every file hashed
    assert git(scene.serving, "diff", "--name-only", env={"GIT_OPTIONAL_LOCKS": "0"}) == "ouroboros/mod_a.py"


@pytest.mark.skipif(os.name == "nt", reason="a POSIX shell stands in for an older Git")
def test_a_git_without_merge_tree_write_tree_carries_the_same_layers_in_the_workspace(scene, monkeypatch, tmp_path):
    """Git before 2.38 has no ``merge-tree --write-tree`` (and before 2.40 no ``--merge-base``): its old merge-tree
    rejects the call with its usage. Preparation then merges the same two sides, each a fresh commit of its tree on
    the exact base, in the candidate's own workspace, and puts the workspace back: the same I' and W' trees and the
    same conflicted paths as the objects-only merge, and the serving checkout untouched. No version is guessed."""
    import shutil

    real = shutil.which("git")
    shim, refused = tmp_path / "old-git", tmp_path / "refused.log"
    shim.mkdir()
    (shim / "git").write_text("#!/bin/sh\n"
                              'if [ "$1" = "merge-tree" ] && [ "$2" = "--write-tree" ]; then\n'
                              f'  echo "$*" >> "{refused}"\n'
                              '  echo "usage: git merge-tree <base-tree> <branch1> <branch2>" >&2; exit 129\n'
                              "fi\n"
                              f'exec "{real}" "$@"\n')
    (shim / "git").chmod(0o755)
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n", "docs/official.txt": "official\n"})
    scene.local_work()
    native = scene.prepare()
    update_prepare.discard(native)
    monkeypatch.setenv("PATH", f"{shim}{os.pathsep}{os.environ['PATH']}")
    before = _serving_bytes(scene)

    older = scene.prepare()

    assert refused.read_text().count("merge-tree --write-tree") >= 2  # each layer merge, then the capability probe
    assert older["state"] == "ready", older.get("reason")
    assert (older["layers"]["new_index"], older["layers"]["new_files"]) == (native["layers"]["new_index"],
                                                                           native["layers"]["new_files"])
    assert _serving_bytes(scene) == before
    workspace = pathlib.Path(older["path"])
    assert git(workspace, "status", "--porcelain") == "" and not (pathlib.Path(
        git(workspace, "rev-parse", "--absolute-git-dir")) / "MERGE_HEAD").exists()
    # A staged edit the official update also changes: the staged layer conflicts the same way on both paths.
    update_prepare.discard(older)
    git(scene.serving, "reset", "-q", "--hard")
    (scene.serving / "docs").mkdir(exist_ok=True)
    (scene.serving / "docs" / "official.txt").write_text("mine\n")
    git(scene.serving, "add", "docs/official.txt")
    conflicted = scene.prepare()
    monkeypatch.setenv("PATH", os.environ["PATH"].split(os.pathsep, 1)[1])
    update_prepare.discard(conflicted)
    expected = scene.prepare()
    assert conflicted["state"] == expected["state"] == "paused_budget"
    assert conflicted["resolver"]["conflicts"] == expected["resolver"]["conflicts"] == ["docs/official.txt"]
    assert conflicted["resolver"]["step"] == expected["resolver"]["step"] == "index"


@pytest.mark.parametrize("tests, reason", [
    ({"unverified": "the selected run did not complete (parallel: exit 124, 0 case(s))", "blocking": []},
     "checks_unverified"),
    ({"blocking": ["tests/test_a.py::test_x"], "already_failing": []}, "checks_failed")], ids=["unverified", "failed"])
def test_checks_that_did_not_complete_are_unverified_never_a_failed_test(scene, tests, reason):
    """A run that did not complete verified nothing: the candidate is not ready and says so as ``checks_unverified``,
    distinct from a measured failure (``checks_failed``); both keep the checks' own record and offer preparing again."""
    scene.official({"ouroboros/mod_a.py": "GEN = 'GEN_CAND'\n"})

    row = update_prepare.prepare(fetch=False, checks=lambda _row: {"ok": False, "tests": tests})

    assert (row["state"], row["reason"]) == ("failed", reason) and row["checks"]["tests"] == tests
    assert "prepare" in update_prepare.projection()["actions"]


@pytest.mark.parametrize("step", ["remote", "fetch", "resolve"])
def test_a_target_step_a_begun_stop_refused_or_ended_is_that_stop_never_an_unavailable_update(monkeypatch, step):
    """Resolving the pinned target runs three tracked Git steps (the remote's configuration, the fetch, the target's
    resolution). A begun stop refuses each start (127) or ends it (-15/-9): that is the stop's typed refusal, as for
    every other Git step of a preparation, never ``UPDATE_UNAVAILABLE``. Without a stop the same failure stays that."""
    from ouroboros.tools import shell_process
    from supervisor import git_ops

    said = "Emergency Stop has retired command admission"
    monkeypatch.setattr(git_ops, "_managed_update_target", lambda: ("managed", "ouroboros", "managed/ouroboros"))
    monkeypatch.setattr("ouroboros.update_channels.get_update_channel", lambda: "dev")
    monkeypatch.setattr(git_ops, "ensure_official_update_remote", lambda: (step != "remote", said))
    monkeypatch.setattr(git_ops, "git_fetch_bounded", lambda _remote: (127, "", said) if step == "fetch" else (0, "", ""))
    monkeypatch.setattr(git_ops, "_resolve_managed_update_target", lambda *_a: (
        ("", "", said) if step == "resolve" else ("managed/ouroboros", "a" * 40, "")))
    monkeypatch.setattr("supervisor.update_merge.managed_update_constitution_present", lambda _sha: True)
    for stop, code in ((False, "UPDATE_UNAVAILABLE"), (True, "UPDATE_PREPARE_STOPPED")):
        monkeypatch.setattr(shell_process, "_panic_requested", stop)
        with pytest.raises(update_prepare.PrepareRefused) as refused:
            update_prepare.resolve_target(fetch=True)
        assert refused.value.code == code, (stop, refused.value.text)
