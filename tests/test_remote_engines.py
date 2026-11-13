"""Named remote Claudexor engines (``ouroboros.remote_engines``, ``ouroboros.remote_workspace``,
``ouroboros.remote_stops``) against the stand in ``tests/_remote_engine_stand.py``: a stand-in OpenSSH on a real
process boundary and a bearer-checked loopback engine. No real SSH, host, engine or account is used.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import time
from types import SimpleNamespace

import pytest

from ouroboros import delegate_custody as custody
from ouroboros import delegate_custody_current as current
from ouroboros import execution_domain as domain
from ouroboros import remote_engines, remote_stops, remote_workspace
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from tests._remote_engine_stand import STORE, TOKEN, alive as _alive, tunnel_pids as _tunnel_pids

pytest_plugins = ("tests._remote_engine_stand",)


# --- grammar, rows, routes ---------------------------------------------------------------------------------

def test_the_route_grammar_names_a_remote_engine_and_keeps_the_owned_one():
    from ouroboros.llm_routing import resolve_target
    from ouroboros.provider_models import (claudexor_engine, claudexor_route, parse_claudexor_model,
                                           parse_model_route, provider_for_model)

    remote = parse_model_route("claudexor@vps::codex=gpt-6-remote")
    assert (remote.kind, remote.connection, remote.model, remote.valid) == ("claudexor", "vps", "codex=gpt-6-remote", True)
    assert parse_model_route("claudexor@default::codex=x").connection == ""  # the owned engine's other spelling
    assert not parse_model_route("claudexor@Bad Name::codex=x").valid
    assert provider_for_model("claudexor@vps::codex=gpt-6-remote") == "claudexor"
    assert parse_claudexor_model("claudexor@vps::codex=gpt-6-remote") == ("codex", "gpt-6-remote")
    assert parse_claudexor_model("claudexor::codex=gpt-6-remote") == ("codex", "gpt-6-remote")
    with pytest.raises(ValueError):
        parse_claudexor_model("openai::gpt-5")
    assert claudexor_engine("claudexor@vps::codex=m") == "vps" and claudexor_engine("claudexor::codex=m") == ""
    assert claudexor_route("vps", "codex", "m") == "claudexor@vps::codex=m"
    rows = [{"name": "vps", "ssh_host": "ouroboros@vds", "remote_root": "/srv/o"}]
    target = resolve_target("claudexor@vps::codex=gpt-6-remote", {remote_engines.ENGINE_CONNECTIONS_KEY: rows})
    assert target["connection"] == "vps" and target["usage_model"] == "claudexor@vps::codex=gpt-6-remote"
    assert resolve_target("claudexor::codex=gpt-6-remote", {})["connection"] == ""
    from ouroboros.llm_attempt import ConnectionUnavailable

    for settings, reason in (({}, "missing"), ({remote_engines.ENGINE_CONNECTIONS_KEY: [{**rows[0], "enabled": False}]},
                                                "disabled")):
        with pytest.raises(ConnectionUnavailable) as refused:  # before dispatch, never the owned engine instead
            resolve_target("claudexor@vps::codex=gpt-6-remote", settings)
        assert refused.value.reason == reason


@pytest.mark.parametrize("row, error", [
    ({"name": "default", "ssh_host": "h"}, "name must be"),
    ({"name": "vps", "ssh_host": "-oProxyCommand=evil"}, "ssh_host must be"),
    ({"name": "vps", "ssh_host": "h", "remote_root": "~/runs"}, "remote_root must be an absolute path"),
    ({"name": "vps", "ssh_host": "h", "remote_root": "/srv/../etc"}, "remote_root must name a folder"),
    ({"name": "vps", "ssh_host": "h", "workspaces": ["relative"]}, "must be an absolute path"),
    ({"name": "vps", "ssh_host": "h", "token": "x"}, "unknown field(s) token"),
])
def test_engine_connection_rows_refuse_what_no_host_could_honour(row, error):
    rows, problem = remote_engines.normalize_engine_connections([row])
    assert rows == [] and error in problem
    ok, none = remote_engines.normalize_engine_connections(
        [{"name": "vps", "ssh_host": "ouroboros@vds", "engine_home": "~/.claudexor/v3", "remote_root": "/srv/o"}])
    assert none == "" and ok[0]["engine_home"] == "~/.claudexor/v3"


def test_a_session_route_carries_its_engine_and_a_local_one_keeps_its_bytes():
    from ouroboros.configured_subagents import configured_subagents_dict, parse_configured_subagents

    local = {"subagent_id": "a", "recommended_use": "x", "access": "full",
             "route": {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": ""}}
    remote = {"subagent_id": "b", "recommended_use": "y", "access": "full",
              "route": {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": "remote-main",
                        "engine": "vps", "workspace": "/srv/project"}}
    parsed = parse_configured_subagents({"enabled": True, "items": [local, remote]})
    assert (parsed.items[0].route.engine, parsed.items[1].route.engine) == ("", "vps")
    assert parsed.items[1].route.workspace == "/srv/project"
    saved = configured_subagents_dict(parsed)["items"]
    assert "engine" not in saved[0]["route"] and saved[1]["route"]["engine"] == "vps"
    for bad, word in (({"engine": "vps", "workspace": "relative"}, "workspace"),
                      ({"workspace": "/srv/p"}, "needs route.engine")):
        with pytest.raises(ValueError, match=word):
            parse_configured_subagents({"enabled": True, "items": [{**remote, "route": {**local["route"], **bad}}]})
    with pytest.raises(ValueError, match="meaningful only for agent_session"):
        parse_configured_subagents({"enabled": True, "items": [{"subagent_id": "c", "recommended_use": "z", "route": {
            "kind": "api_model", "target_id": "claudexor::codex=m", "engine": "vps"}}]})


def test_engine_predicates_are_total_and_another_engines_absence_never_settles_a_remote_run(tmp_path):
    ref = {"kind": "remote", "connection": "vps", "ssh_host": "ouroboros@vds", "engine_home": "/e", "store": STORE}
    other = {**ref, "connection": "other"}
    assert domain.is_bound_ref(ref) and domain.is_remote_ref(ref) and not domain.is_work_ref(ref)
    assert domain.engine_key(ref)[1] == STORE and domain.engine_key({}) is None
    held = SimpleNamespace(engine_ref=ref)
    assert domain.same_engine(held, ref) and not domain.same_engine(held, other)
    # The same connection name re-pointed at another address or engine home is another engine, even with the
    # same store: its answers (a 404 included) never speak for a run recorded on the first, nor share its client.
    for moved in ({**ref, "ssh_host": "root@vds"}, {**ref, "engine_home": "/other"}):
        assert domain.engine_key(moved) != domain.engine_key(ref)
        assert not domain.same_engine(SimpleNamespace(engine_ref=moved), ref)
        groups = domain.engine_groups([SimpleNamespace(engine_ref=ref), SimpleNamespace(engine_ref=moved)])
        assert len([group for group in groups if group[0]]) == 2  # two engines, never one shared client
    assert not domain.same_engine(SimpleNamespace(engine_ref=None), ref)  # the owned engine never answers for it
    assert not domain.same_engine(held, {})  # nor the remote one for an owned run
    groups = domain.engine_groups([SimpleNamespace(engine_ref=ref), SimpleNamespace(engine_ref={}),
                                   SimpleNamespace(engine_ref=other)])
    assert [group[0].get("connection") for group in groups] == [None, "vps", "other"]
    drive = tmp_path / "drive"
    drive.mkdir()
    entry = custody.RunCustody(run_id="run-r", task_id="t", invocation_id="inv-r", route_id="codex", model="m",
                               access="readonly", mode="ask", ledger_root=str(drive), engine_ref=dict(ref),
                               daemon_store_id=STORE)
    assert custody.record_started(drive, entry)
    run = current.run(drive, "run-r")
    assert run.engine_ref == ref
    gone = ClaudexorUnavailable("run_not_found", "gone", status_code=404)
    gone.problem = {"context": {"storeId": STORE}}
    owned = SimpleNamespace(store_identity=STORE, engine_ref=None)
    assert not custody.daemon_says_absent(gone, gateway=owned, run=run)  # same store id, but not its engine
    assert custody.daemon_says_absent(gone, gateway=SimpleNamespace(store_identity=STORE, engine_ref=ref), run=run)


# --- the tunnel --------------------------------------------------------------------------------------------

def test_attach_reads_the_running_engine_publishes_a_private_descriptor_and_starts_nothing(host):
    client = remote_engines.gateway("vps")
    try:
        assert client.store_identity == STORE
        assert client.engine_ref == {"kind": "remote", "connection": "vps", "ssh_host": "ouroboros@vds",
                                     "engine_home": str(host.engine_home), "store": STORE,
                                     "remote_root": str(host.remote_root)}
    finally:
        client.close()
    published = remote_engines.published("vps")
    folder = pathlib.Path(published["tokenPath"]).parent
    assert published["host"] == "127.0.0.1" and published["port"] != json.loads(host.descriptor.read_text())["port"]
    assert TOKEN not in json.dumps(published) and TOKEN not in json.dumps(remote_engines.status())
    assert (folder / "token").stat().st_mode & 0o777 == 0o600
    calls = host.ssh_calls()
    commands = [" ".join(call["command"]) for call in calls if call["command"]]
    assert any("control-api.json" in command for command in commands)
    assert not any(word in command for command in commands for word in ("bootstrap", "claudexor", "stop", "kill"))
    forwards = [call for call in calls if call["forward"]]
    assert len(forwards) == 1 and forwards[0]["no_command"]
    for option in ("BatchMode=yes", "StrictHostKeyChecking=yes", "ControlMaster=no", "ControlPath=none"):
        assert option in forwards[0]["options"]
    # a second process view reuses the published tunnel: no second attach
    again = remote_engines.gateway("vps")
    again.close()
    assert len([call for call in host.ssh_calls() if call["forward"]]) == 1
    assert remote_engines.check("vps")["ok"] and len([c for c in host.ssh_calls() if c["forward"]]) == 1
    pid = _tunnel_pids("vps")[0]
    assert remote_engines.retire("vps")
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid) and remote_engines.published("vps") is None
    assert [call for call in host.state["calls"] if "/v2/runs" in call["path"]] == []  # nothing ran or stopped


def test_no_running_engine_or_a_prompt_refuses_typed_and_never_attaches(host):
    host.descriptor.unlink()
    with pytest.raises(ClaudexorUnavailable) as stopped:
        remote_engines.gateway("vps")
    assert stopped.value.code == "remote_engine_not_running"
    for destination, code in (("deny@vds", "remote_ssh_needs_interaction"),
                              ("unknown@vds", "remote_ssh_host_key_unverified")):
        host.set_rows([{**host.rows[0], "ssh_host": destination}])
        result = remote_engines.check("vps")
        assert result["ok"] is False and result["code"] == code
    assert not [call for call in host.ssh_calls() if call["forward"]]


def test_recorded_work_is_reached_only_through_its_own_address_home_and_store(host):
    client = remote_engines.gateway("vps")
    ref = dict(client.engine_ref)
    client.close()
    bound = remote_engines.bound_gateway(ref)
    assert bound.engine_ref["store"] == STORE
    bound.close()
    host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@other-vps"}])
    with pytest.raises(ClaudexorUnavailable) as repointed:
        remote_engines.bound_gateway(ref)
    assert repointed.value.code == "remote_engine_changed"
    host.set_rows(host.rows)
    host.state["store"] = "store-after-reinstall"
    remote_engines.retire("vps")
    with pytest.raises(ClaudexorUnavailable) as changed:
        remote_engines.bound_gateway(ref)
    assert changed.value.code == "remote_engine_changed"
    host.set_rows([*host.rows, {**host.rows[0], "name": "other"}])  # another connection to the same engine
    with pytest.raises(ClaudexorUnavailable):
        domain.bound_gateway({**ref, "connection": "vps"})  # still the recorded connection only, still refused


# --- workspaces --------------------------------------------------------------------------------------------

def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, text=True).stdout


def _local_copy(tmp_path):
    copy = tmp_path / "local" / "copy"
    (copy / "sub").mkdir(parents=True)
    _git(copy.parent, "init", "-q", str(copy))
    (copy / ".gitignore").write_text("build/\n")
    (copy / "a.txt").write_text("alpha\n")
    (copy / "sub" / "b.txt").write_text("beta\n")
    _git(copy, "add", "-A")
    _git(copy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    (copy / "c.txt").write_text("untracked kept\n")
    (copy / ".env").write_text("SECRET=1\n")
    (copy / "build").mkdir()
    (copy / "build" / "out.bin").write_text("ignored\n")
    return copy


def test_a_staged_start_sends_the_private_copy_and_its_inputs_and_the_result_comes_home(host, tmp_path):
    copy = _local_copy(tmp_path)
    scratch = tmp_path / "local" / "scratch"
    scratch.mkdir()
    selected = tmp_path / "local" / "retained" / "inputs" / "selected" / "abcd"
    selected.mkdir(parents=True)
    (selected / "brief.pdf").write_bytes(b"%PDF-1.4 owner brief")
    manifest = selected.parent.parent / "inputs.json"
    manifest.write_text(json.dumps([{"local": "inputs/selected/abcd/brief.pdf"}]))
    skill = tmp_path / "local" / "skills" / "ui_design"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text("read references/look.md")
    (skill / "references" / "look.md").write_text("judge by eye")
    request = {
        "prompt": f"Fix the bug in {copy}/a.txt", "access": "full",
        "instructions": f"The private execution root for this run is {copy}. Read {selected / 'brief.pdf'} "
                        f"(manifest {manifest}). The authority root /Users/owner/project stays read-only. "
                        f"HOST SKILL REFERENCES: read {skill / 'SKILL.md'}.",
        "scope": {"kind": "project", "root": "/Users/owner/project", "ephemeral": True},
        "execution": {"isolation": "live", "delegated": True, "workspaceRoot": str(copy),
                      "auxiliaryWritableRoots": [str(scratch)]},
        "harnesses": ["codex"], "primaryHarness": "codex",
        "_host_inputs": [{"path": str(selected / "brief.pdf"), "kind": "input"},
                         {"path": str(manifest), "kind": "input"}, {"path": str(skill), "kind": "reference"}],
    }
    client = remote_engines.gateway("vps")
    try:
        handle = client.start_run(request, idempotency_key="inv-remote-1")
    finally:
        client.close()
    assert handle["runId"] == "run-remote-1" and "runDir" not in handle and handle["remoteRunDir"]
    folder = host.remote_root / "inv-remote-1"
    work = folder / "work"
    sent = next(call for call in host.state["calls"] if call["path"] == "/v2/runs")
    body = sent["body"]
    assert sent["key"] == "inv-remote-1" and "_host_inputs" not in body
    assert body["scope"] == {"kind": "project", "root": str(work), "ephemeral": True}
    assert body["execution"]["workspaceRoot"] == str(work)
    assert body["execution"]["auxiliaryWritableRoots"] == [str(folder / "aux" / "0")]
    # the owner's and the host's words travel byte for byte; the typed fields and the appended map say where things are
    assert body["prompt"] == request["prompt"] and body["instructions"].startswith(request["instructions"])
    assert "HOST PATH MAP" in body["instructions"][len(request["instructions"]):]
    brief = folder / "inputs" / "0" / "brief.pdf"
    assert str(brief) in body["instructions"] and brief.read_bytes() == b"%PDF-1.4 owner brief"
    assert (folder / "inputs" / "1" / "inputs.json").exists() and (folder / "aux" / "0").is_dir()
    # exactly the copy's own files: no selection policy beyond the copy's (an untracked .env it holds travels too)
    assert {p.relative_to(work).as_posix() for p in work.rglob("*") if p.is_file() and ".git" not in p.parts} == {
        ".gitignore", "a.txt", "sub/b.txt", "c.txt", ".env"}
    reference = folder / "refs" / "2" / "ui_design"
    assert (reference / "references" / "look.md").read_text() == "judge by eye"
    assert f"read {skill / 'SKILL.md'}" in body["instructions"]  # a reference is mapped, never rewritten in prose
    assert f"{skill} -> {reference}" in body["instructions"] and f"{copy} -> {work}" in body["instructions"]
    trust = [call for call in host.state["calls"] if call["path"] == "/v2/trust" and call["method"] == "POST"]
    assert trust and trust[0]["body"]["repoRoot"] == str(work)  # the grant is the staged copy's, never a local root
    uploads = len([call for call in host.ssh_calls() if "tar -xf" in " ".join(call["command"])])
    replay = remote_engines.gateway("vps")
    replay.start_run(request, idempotency_key="inv-remote-1")  # an unknown-outcome replay finds the same copy
    replay.close()
    assert len([call for call in host.ssh_calls() if "tar -xf" in " ".join(call["command"])]) == uploads
    # the remote session works in its copy
    (work / "a.txt").write_text("alpha fixed\n")
    (work / "sub" / "b.txt").unlink()
    (work / "new.txt").write_text("added remotely\n")
    ref = {"kind": "remote", "connection": "vps", "ssh_host": "ouroboros@vds", "engine_home": str(host.engine_home),
           "store": STORE, "remote_root": str(host.remote_root)}
    entry = SimpleNamespace(execution_root=str(copy), invocation_id="inv-remote-1", snapshot_id="inv-remote-1")
    cap_dir = tmp_path / "capture"
    first = remote_workspace.mirror_result(entry, ref, cap_dir)
    assert first["state"] == "applied" and pathlib.Path(first["patch"]).exists()
    assert (copy / "a.txt").read_text() == "alpha fixed\n" and not (copy / "sub" / "b.txt").exists()
    assert (copy / "new.txt").read_text() == "added remotely\n" and (copy / ".env").exists()
    assert remote_workspace.mirror_result(entry, ref, cap_dir)["state"] == "already_applied"
    assert work.is_dir()  # the host copy stays for Continue and Keep: retirement is not the capture's
    (copy / "a.txt").write_text("local edit meanwhile\n")
    conflicted = remote_workspace.mirror_result(entry, ref, cap_dir)
    assert conflicted["state"] == "conflict" and pathlib.Path(conflicted["patch"]).exists()
    host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@elsewhere"}])
    assert remote_workspace.mirror_result(entry, ref, cap_dir)["code"] == "remote_engine_changed"


def test_a_missing_declared_input_refuses_the_start_before_anything_is_sent(host, tmp_path):
    copy = _local_copy(tmp_path)
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)},
               "_host_inputs": [str(tmp_path / "gone.txt")]}
    client = remote_engines.gateway("vps")
    try:
        with pytest.raises(ClaudexorUnavailable) as missing:
            client.start_run(request, idempotency_key="inv-missing")
    finally:
        client.close()
    assert missing.value.code == "remote_input_missing"
    assert not [call for call in host.state["calls"] if call["path"] == "/v2/runs"]
    assert not (host.remote_root / "inv-missing").exists()


def test_a_baseline_that_does_not_list_exactly_the_manifest_sent_commits_nothing_and_starts_nothing(
        host, tmp_path, monkeypatch):
    """The host commits the baseline only when its index lists exactly the manifest sent; here the manifest names
    one file twice, which no index can list, so the host refuses before the commit and nothing is POSTed. The same
    invocation key, staged again from the copy's real selection, then proves its baseline and starts."""
    copy = _local_copy(tmp_path)
    real = remote_workspace._selected_files

    def doubled(root, *, git):
        files, excluded = real(root, git=git)
        return files + files[:1], excluded
    monkeypatch.setattr(remote_workspace, "_selected_files", doubled)
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)}}
    folder = host.remote_root / "inv-mismatch"
    client = remote_engines.gateway("vps")
    try:
        with pytest.raises(ClaudexorUnavailable) as refused:
            client.start_run(request, idempotency_key="inv-mismatch")
    finally:
        client.close()
    assert refused.value.code == "remote_stage_mismatch"
    assert not [call for call in host.state["calls"] if call["path"] == "/v2/runs"]
    assert not (folder / remote_workspace.STAGED_MARKER).exists() and not (folder / "work").exists()
    monkeypatch.setattr(remote_workspace, "_selected_files", real)
    retry = remote_engines.gateway("vps")
    try:
        retry.start_run(request, idempotency_key="inv-mismatch")
    finally:
        retry.close()
    assert (folder / remote_workspace.STAGED_MARKER).exists()
    assert _git(folder / "work", "ls-files").split() == sorted(["a.txt", "sub/b.txt", "c.txt", ".env", ".gitignore"])
    assert [call["key"] for call in host.state["calls"] if call["path"] == "/v2/runs"] == ["inv-mismatch"]


@pytest.mark.parametrize("answer, removed", [((429, "rate_limited"), True), ((409, "idempotency_conflict"), False),
                                             ((503, "engine_busy"), False)])
def test_a_definite_refusal_of_the_post_removes_only_the_copy_this_call_staged(host, tmp_path, answer, removed):
    """The engine's own definite refusal of the POST (a 4xx) started nothing, so the folder THIS call staged for it goes
    again, as after a refusal before the POST; an answer that leaves a run possible (a 5xx, a key the engine already
    holds) keeps it for that run's custody, and a folder an earlier call staged (its answer lost) is never touched."""
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(_local_copy(tmp_path))}}

    def start(key):
        client = remote_engines.gateway("vps")
        try:
            with pytest.raises(ClaudexorUnavailable) as refused:
                client.start_run(request, idempotency_key=key)
        finally:
            client.close()
        return refused.value

    host.state["refuse_start"] = answer
    refused = start("inv-refused")
    assert refused.status_code == answer[0] and not host.state["runs"]
    assert (host.remote_root / "inv-refused").exists() is not removed
    assert ("was removed again" in str(refused)) is removed
    host.state["refuse_start"], host.state["lose_start_answer"] = None, True
    start("inv-earlier")  # accepted there, its answer lost: the copy is that run's
    host.state["refuse_start"], host.state["lose_start_answer"] = answer, False
    assert start("inv-earlier").status_code == answer[0]  # its retry refused: this call staged nothing
    assert (host.remote_root / "inv-earlier" / "work").is_dir() and len(host.state["runs"]) == 1


@pytest.mark.parametrize("harness", ["codex", "claude"])
def test_a_remote_session_of_either_harness_starts_on_that_engine_with_its_harness_unchanged(host, tmp_path, harness):
    """Nothing in the remote path is harness-specific: a configured Claude Code session row on a remote engine reaches
    that engine exactly as a Codex one does, and whether the host serves that harness is the engine's own answer."""
    from ouroboros.configured_subagents import parse_configured_subagents

    copy = _local_copy(tmp_path)
    parsed = parse_configured_subagents({"enabled": True, "items": [{
        "subagent_id": "r", "recommended_use": "x", "access": "full",
        "route": {"kind": "agent_session", "target_id": f"{harness}=model-x", "engine": "vps"}}]})
    gateway = domain.native_gateway(engine=parsed.items[0].route.engine)
    try:
        assert gateway.engine_ref["kind"] == "remote" and gateway.stages_workspace
        gateway.start_run({"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)},
                           "harnesses": [harness], "primaryHarness": harness}, idempotency_key=f"inv-{harness}")
    finally:
        gateway.close()
    [sent] = [call["body"] for call in host.state["calls"] if call["path"] == "/v2/runs"]
    assert (sent["harnesses"], sent["primaryHarness"]) == ([harness], harness)
    assert sent["scope"]["root"] == str(host.remote_root / f"inv-{harness}" / "work")


def test_an_in_place_start_on_a_remote_engine_is_refused_before_anything_is_recorded_or_sent(host):
    """``workspace='in_place'`` edits the owner's folder directly. A remote engine would edit a copy on its host that
    no in-place capture brings back (and that copy's retirement would then remove), so the start is refused typed
    where its scratch is prepared, before any row, scratch or POST; a local engine's in-place start is unchanged."""
    from ouroboros.subagent_work_order import in_place_scratch

    gateway = domain.native_gateway(engine="vps")
    try:
        with pytest.raises(ClaudexorUnavailable) as refused:
            in_place_scratch(SimpleNamespace(), gateway, SimpleNamespace(), SimpleNamespace(access="full"), "inv-ip",
                             "/owner/project")
    finally:
        gateway.close()
    assert refused.value.code == "remote_in_place_unsupported" and "Nothing was started" in str(refused.value)
    assert not _posts(host)
    local = SimpleNamespace(engine_ref=None)  # the owned engine keeps its deferred scratch preparation
    assert callable(in_place_scratch(SimpleNamespace(), local, SimpleNamespace(), SimpleNamespace(access="full"),
                                     "inv-ip", "/owner/project"))


def test_a_declared_remote_workspace_works_there_and_reports_its_remote_effects(host, tmp_path):
    copy = _local_copy(tmp_path)
    brief = tmp_path / "local" / "brief.md"
    brief.write_text("the owner's brief")
    request = {"prompt": f"Work in {copy}", "instructions": f"Root {copy}; brief {brief}.",
               "scope": {"kind": "project", "root": str(copy), "ephemeral": True},
               "execution": {"isolation": "live", "workspaceRoot": str(copy), "auxiliaryWritableRoots": ["/x"]},
               "_host_inputs": [str(brief)]}
    client = remote_engines.gateway("vps")
    client.engine_ref = {**client.engine_ref, "workspace": str(host.workspace / "app")}
    try:
        client.start_run(request, idempotency_key="inv-ws")
        client.start_run({**request, "continueFrom": "run-remote-1"}, idempotency_key="inv-ws-next")
        client.engine_ref = {**client.engine_ref, "workspace": "/srv/undeclared"}
        with pytest.raises(ClaudexorUnavailable) as undeclared:
            client.start_run(request, idempotency_key="inv-ws-2")
    finally:
        client.close()
    assert undeclared.value.code == "remote_workspace_undeclared"
    body = next(call for call in host.state["calls"] if call["path"] == "/v2/runs")["body"]
    target = str(host.workspace / "app")
    assert body["scope"]["root"] == target and "ephemeral" not in body["scope"]
    assert body["execution"]["workspaceRoot"] == target and "auxiliaryWritableRoots" not in body["execution"]
    sent = host.remote_root / "inv-ws" / "inputs" / "0" / "brief.md"
    assert sent.read_text() == "the owner's brief" and str(sent) in body["instructions"]
    assert body["prompt"] == request["prompt"]
    resumed = next(call["body"] for call in host.state["calls"] if call["key"] == "inv-ws-next")
    tail = resumed["prompt"][len(request["prompt"]):]  # a resumed session also learns this start's own inputs
    assert resumed["prompt"].startswith(request["prompt"])
    assert tail == resumed["instructions"][len(request["instructions"]):]
    assert f"{brief} -> {host.remote_root / 'inv-ws-next' / 'inputs' / '0' / 'brief.md'}" in tail
    assert not (host.remote_root / "inv-ws" / "work").exists()  # nothing of the local project travelled
    ref = {"kind": "remote", "connection": "vps", "ssh_host": "ouroboros@vds", "engine_home": str(host.engine_home),
           "store": STORE, "remote_root": str(host.remote_root), "workspace": target}
    block = remote_workspace.remote_workspace_block(SimpleNamespace(engine_ref=ref, run_id="run-remote-1",
                                                                    snapshot_id="", patch_disposed=""))
    # with nowhere to retain its returns, nothing came back: never a final projection, nothing released
    assert block["status"] == "failed" and block["retry"] == "transient" and block["execution_root"] == target
    assert block["remote_run"]["workProduct"] == {"kind": "git", "path": "final/patch.diff"}
    from ouroboros.delegate_capture_attempt import capture_required

    owed = SimpleNamespace(mode="ask", patch_disposed="", superseded_by="", resource_ref={}, execution_root=str(copy),
                           engine_ref=ref)
    assert not capture_required(owed) and capture_required(SimpleNamespace(**{**vars(owed), "engine_ref": {}}))


def test_ordinary_return_keeps_unlistable_remote_output_owed(host, tmp_path):
    """Git may exit zero while skipping a directory: both ordinary and reviewer captures must keep its result owed."""
    copy = _local_copy(tmp_path)
    client = remote_engines.gateway("vps")
    try:
        client.start_run({"prompt": "p", "scope": {"kind": "project", "root": str(copy)}},
                         idempotency_key="inv-unreadable")
        ref = dict(client.engine_ref)
    finally:
        client.close()
    remote = host.remote_root / "inv-unreadable" / "work"
    locked = remote / "locked"
    locked.mkdir()
    (locked / "unique.txt").write_text("must return or remain owed\n")
    (remote / "good.txt").write_text("also retained\n")
    locked.chmod(0)
    try:
        with pytest.raises(PermissionError):
            list(locked.iterdir())  # the real filesystem, not an invented Git error
        result = remote_workspace.mirror_result(
            SimpleNamespace(execution_root=str(copy), invocation_id="inv-unreadable"), ref, tmp_path / "capture")
        assert result["state"] == "unlisted", result
        assert not (copy / "good.txt").exists() and not (copy / "locked").exists()
    finally:
        locked.chmod(0o700)
    repaired = remote_workspace.mirror_result(
        SimpleNamespace(execution_root=str(copy), invocation_id="inv-unreadable"), ref, tmp_path / "capture")
    assert repaired["state"] == "applied"
    assert (copy / "locked" / "unique.txt").read_text() == "must return or remain owed\n"
    assert (copy / "good.txt").read_text() == "also retained\n"


# --- Panic, recovery, model operations ---------------------------------------------------------------------

def test_panic_never_waits_on_a_remote_engine_its_census_ends_the_forward_and_the_next_start_delivers_the_stop(
        host, tmp_path, monkeypatch):
    """Panic writes the owed stop, fires one unawaited stop per remote run (here into an engine that accepts and never
    answers) and returns at once; its census ends the SSH forward like any other installation process. The next
    start's ordinary sweep, with no task asking, delivers the stop on the run's own engine and the intent stays until
    a terminal read-back proves it."""
    import os

    from ouroboros import config
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs
    from ouroboros.process_scope import SCOPE_VALUE, installation_marker, panic_scope_census

    data = pathlib.Path(config.DATA_DIR)
    data.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(installation_marker(data, create=True), SCOPE_VALUE)  # this "server" planted its scope
    client = remote_engines.gateway("vps")
    ref = dict(client.engine_ref)
    client.close()
    for rid, bound in (("run-remote-live", ref), ("run-owned", {})):
        assert custody.record_started(data, custody.RunCustody(
            run_id=rid, task_id="t-panic", invocation_id=f"inv-{rid}", route_id="codex", model="m",
            access="full", mode="ask", ledger_root=str(data), engine_ref=dict(bound), daemon_store_id=STORE))
    pid = _tunnel_pids("vps")[0]
    import threading

    host.state["hang_control"] = threading.Event()  # the engine accepts the stop and never answers it
    started = time.monotonic()
    outcome = remote_stops.panic_stop_work(data)
    assert time.monotonic() - started < 1.0 and outcome["remote_stops_sent"] == 1  # nothing waited
    assert current.run(data, "run-remote-live").containment_fault["reason"] == remote_stops.PANIC_STOP_UNCONFIRMED
    assert not current.run(data, "run-owned").containment_fault
    census = panic_scope_census(data, exclude={os.getpid(), os.getppid(), 1})
    assert census["requested"] is True
    deadline = time.monotonic() + 5
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(pid)  # the forward is an ordinary member of the installation's census
    host.state.pop("hang_control").set()
    assert remote_stops.owed_stop_counts(data)["vps"] == {"runs": 1, "starts": 0}  # shown until read back stopped
    # the next start: the ordinary sweep re-attaches and delivers the stop, though the task is live and asks nothing
    remote_engines.retire("vps")
    calls = len(host.state["calls"])
    host.state["run_state"] = "running"
    owned_gateway = lambda: SimpleNamespace(store_identity="owned", engine_ref=None, close=lambda: None)  # noqa: E731
    outcomes = reconcile_orphaned_runs(data, running_task_ids={"t-panic"}, gateway_factory=owned_gateway)
    row = next(row for row in outcomes if row.get("run_id") == "run-remote-live")
    assert row["action"] == "cancelled" and row["outcome"] == custody.CANCEL_REQUESTED
    assert any(call["path"] == "/v2/runs/run-remote-live/control" for call in host.state["calls"][calls:])
    assert current.run(data, "run-remote-live").containment_fault["reason"] == remote_stops.PANIC_STOP_UNCONFIRMED
    host.state["run_state"] = "cancelled"  # a terminal read proves the stop: the intent is discharged
    reconcile_orphaned_runs(data, running_task_ids={"t-panic"}, gateway_factory=owned_gateway)
    assert not current.run(data, "run-remote-live").containment_fault
    assert remote_stops.owed_stop_counts(data).get("vps", {"runs": 0})["runs"] == 0


def test_an_owed_stop_outlasts_a_failed_read_and_every_other_incident_until_its_proof(host, tmp_path):
    """Emergency Stop's owed stop shares one incident slot with every other incident about the same run, or the same
    start before it has a run id. A transient failure to read the run on its own reachable engine, an unverified cancel
    or another writer's incident never replaces it: the run stays owed, its engine's row still counts it, and the next
    pass with the engine answering stops it; only a terminal read-back discharges it. An incident that is not an owed
    stop is replaced by the next one as before."""
    from ouroboros.delegate_custody_reconcile import _reconcile_one, reconcile_orphaned_runs

    data = tmp_path / "custody"
    ref = _remote_ref()

    def started(run_id):
        assert custody.record_started(data, custody.RunCustody(
            run_id=run_id, task_id="t-owed", invocation_id=f"inv-{run_id}", route_id="codex", model="m",
            access="full", mode="ask", ledger_root=str(data), engine_ref=dict(ref), daemon_store_id=STORE))
        return current.run(data, run_id)

    started("run-owed")
    _request_start(data, "inv-owed-start", ref, tmp_path, task="t-owed")
    assert remote_stops.record_owed_stops(data) == {"runs": 1, "starts": 1}

    def unreadable(*_args, **_kwargs):
        raise ClaudexorUnavailable("daemon_not_answering", "the engine did not answer this read")

    reading = SimpleNamespace(get_run=unreadable, engine_ref=dict(ref), store_identity=STORE)
    assert _reconcile_one(data, reading, current.run(data, "run-owed"))["action"] == "unreadable"
    custody.record_containment_fault(data, current.run(data, "run-owed"), "cancel_unverified", "another writer")
    assert custody.emit(data, custody.CONTAINMENT_FAULT, {"run_id": "", "invocation_id": "inv-owed-start",
                                                         "task_id": "t-owed", "reason": "start_unreadable"})
    assert remote_stops.owed(current.run(data, "run-owed")) and remote_stops.owed(_pending(data, "inv-owed-start")[0])
    assert remote_stops.owed_stop_counts(data)["vps"] == {"runs": 1, "starts": 1}  # its engine's row still shows them
    plain = started("run-plain")  # recorded after Emergency Stop: it owes nothing
    custody.record_containment_fault(data, plain, "reconcile_unreadable", "first")
    custody.record_containment_fault(data, current.run(data, "run-plain"), "cancel_unverified", "second")
    assert current.run(data, "run-plain").containment_fault["reason"] == "cancel_unverified"
    host.state["run_states"] = {"run-owed": "running"}  # the engine answers again: the stop is delivered
    outcomes = reconcile_orphaned_runs(data, running_task_ids={"t-owed"}, gateway_factory=_OWNED_GATEWAY)
    row = next(row for row in outcomes if row.get("run_id") == "run-owed")
    assert row["action"] == "cancelled" and remote_stops.owed(current.run(data, "run-owed"))
    assert any(call["path"] == "/v2/runs/run-owed/control" for call in host.state["calls"])
    host.state["run_states"] = {"run-owed": "cancelled"}  # a terminal read-back proves it
    reconcile_orphaned_runs(data, running_task_ids={"t-owed"}, gateway_factory=_OWNED_GATEWAY)
    assert not remote_stops.owed(current.run(data, "run-owed"))


def test_a_remote_model_operation_recovers_only_through_its_recorded_engine(host, tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.llm_claudexor import ModelTurnState, bind_turn_engine, recover_model_attempt
    from ouroboros.observability import persist_call

    data = pathlib.Path(config.DATA_DIR)
    client = remote_engines.gateway("vps")
    ref = dict(client.engine_ref)
    client.close()
    persist_call(data, task_id="t-op", call_id="att-1_model_request", call_type="llm_claudexor_request",
                 payload={"x": 1}, keep_raw=True,
                 manifest={"invocation_id": "att-1", "task_id": "t-op", "operation_id": "op-1", "engine_ref": ref})
    opened = []
    from ouroboros import claudexor_daemon

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway",
                        lambda: pytest.fail("the owned engine must not answer for a remote operation"))

    def fake_bound(bound_ref, **_kwargs):
        opened.append(bound_ref["connection"])
        return SimpleNamespace(get_model_operation=lambda *_a, **_k: {"state": "running"}, close=lambda: None)

    monkeypatch.setattr(remote_engines, "bound_gateway", fake_bound)
    assert recover_model_attempt(data, {"task_id": "t-op", "attempt_id": "att-1"}) is None
    assert opened == ["vps"]
    probed = []  # a caller's own probe policy receives the request and opens that engine itself
    assert recover_model_attempt(data, {"task_id": "t-op", "attempt_id": "att-1"}, gateway_factory=lambda request=None: (
        probed.append(request["engine_ref"]["connection"]) or fake_bound(request["engine_ref"]))) is None
    assert probed == ["vps"]
    slot = ModelTurnState({"turn": "owned-engine-token"})
    bind_turn_engine(slot, {"connection": ""})
    assert slot.envelope == {"turn": "owned-engine-token"}
    bind_turn_engine(slot, {"connection": "vps"})  # another engine: that token never travels there
    assert slot.envelope is None and slot.engine == "vps"


@pytest.mark.parametrize("change", ["store", "ssh_host", "engine_home"])
def test_a_tasks_conversation_stays_on_the_engine_it_began_on(host, tmp_path, change):
    """The real preparation and upload seam (``_ModelInvocation.prepare``). A task's turn slot binds to the engine
    identity of its first dispatch. The same connection name re-pointed to another store, address or engine home is
    another engine: every later request of that task (its turn token, the native continuations in its messages, the
    account it prefers, all issued by the first engine) is refused before its upload, typed, and nothing reaches the
    other engine. A new task binds to the engine the connection names now and keeps its own continuity there;
    pointing the connection back resumes the first task with its own token. A local route's slot is untouched."""
    import hashlib
    import shutil

    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched, ModelTurnState, _ModelInvocation, bind_turn_engine

    drive = tmp_path / "drive"
    drive.mkdir()
    target = {"connection": "vps", "source": "codex", "resolved_model": "gpt-6-remote"}

    def send(slot, key, payload):
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        invocation = _ModelInvocation(target, payload, {"model_turn_state": slot, "timeout": 30})
        try:
            invocation.prepare(SimpleNamespace(attempt_id=key, drive_root=drive), canonical_bytes=data,
                               sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        finally:
            invocation.close()

    def uploaded():
        return [item["bytes"] for item in host.state.get("uploads", {}).values() if item["bytes"] is not None]

    task = ModelTurnState()
    send(task, "att-1", {"x": 1, "nativeContinuation": None})  # the task's turn starts on this engine
    task.envelope = {"turn": "issued-by-the-first-engine"}  # what that engine's result handed back
    send(task, "att-2", {"x": 2, "nativeContinuation": dict(task.envelope)})  # the same engine: its token travels there
    assert b"issued-by-the-first-engine" in uploaded()[-1]
    count = len(uploaded())
    if change == "store":  # the same address and home, a reinstalled engine
        host.state["store"] = "store-remote-2"
        remote_engines.retire("vps")
    elif change == "ssh_host":
        host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@vds-moved"}])
    else:
        moved = tmp_path / "host" / "engine-moved"
        shutil.copytree(host.engine_home, moved)
        host.set_rows([{**host.rows[0], "engine_home": str(moved)}])
    native = {"route": {"source": "codex", "model": "gpt-6-remote", "credentialProfileId": "first-engine-account",
                        "accountFingerprint": "fp-first"}}
    for key, payload in (("att-3", {"x": 3, "nativeContinuation": dict(task.envelope)}),
                         ("att-4", {"x": 4, "nativeContinuation": None, "account": {"mode": "auto",
                                    "preferredProfileId": "first-engine-account"},
                                    "messages": [{"role": "assistant", "content": "a", "nativeContinuation": native}]})):
        with pytest.raises(ClaudexorModelNotDispatched) as refused:
            send(task, key, payload)
        assert refused.value.code == "model_turn_engine_changed", key
    assert len(uploaded()) == count  # nothing of the first engine's conversation reached the other one
    assert task.envelope == {"turn": "issued-by-the-first-engine"}  # kept for the engine that issued it
    fresh = ModelTurnState()  # a new task binds to the engine the connection names now
    send(fresh, "att-5", {"x": 5, "nativeContinuation": None})
    fresh.envelope = {"turn": "issued-by-the-new-engine"}
    send(fresh, "att-6", {"x": 6, "nativeContinuation": dict(fresh.envelope)})
    assert len(uploaded()) == count + 2 and b"issued-by-the-new-engine" in uploaded()[-1]
    if change != "store":  # the row pointed back names the first engine again: its task resumes with its own token
        host.set_rows(host.rows)
        send(task, "att-7", {"x": 7, "nativeContinuation": dict(task.envelope)})
        assert b"issued-by-the-first-engine" in uploaded()[-1]
    owned = ModelTurnState({"turn": "owned-engine-token"})
    bind_turn_engine(owned, {"connection": ""})
    assert owned.envelope == {"turn": "owned-engine-token"}  # a local route's continuity is as before


@pytest.mark.parametrize("change", ["store", "ssh_host"])
def test_a_cold_restored_conversation_stays_on_the_engine_that_issued_it(host, tmp_path, change):
    """A cold restore: the task's loop starts with a fresh turn slot (``loop.py``) and its transcript comes back from
    the working checkpoint (``working_checkpoint.save_round`` / ``read_working``) with the native continuations the
    first engine issued, whose route names the account its answers came from. The same connection name re-pointed
    meanwhile reaches another engine: the restored conversation's first request (its continuations, and the account
    preference the restore and the send read from them) is refused before its upload, typed, and nothing reaches the
    other engine. The same engine after the restore receives exactly the bytes it received before this binding
    existed; a new task binds to the engine the connection names now; pointing the name back resumes the first."""
    import copy
    import hashlib

    from ouroboros import working_checkpoint as wc
    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched, ModelTurnState, _ModelInvocation, _request
    from tests._budget_pause_exact_helpers import _loop_ctx

    drive = tmp_path / "drive"
    drive.mkdir()
    target = {"connection": "vps", "source": "codex", "resolved_model": "gpt-6-remote",
              "usage_model": "claudexor@vps::codex=gpt-6-remote"}
    auto = {"model_role": "main", "model_account_override": ""}
    route = {"source": "codex", "model": "gpt-6-remote", "credentialProfileId": "first-engine-account",
             "accountFingerprint": "fp-first"}

    def uploaded():
        return [item["bytes"] for item in host.state.get("uploads", {}).values() if item["bytes"] is not None]

    def build(messages, slot):
        return _request(dict(target), copy.deepcopy(messages), [], {**auto, "model_turn_state": slot})

    def upload(slot, key, payload):
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        invocation = _ModelInvocation(dict(target), payload, {"model_turn_state": slot, "timeout": 30})
        try:
            invocation.prepare(SimpleNamespace(attempt_id=key, drive_root=drive), canonical_bytes=data,
                               sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        except BaseException:
            invocation.close()
            raise
        return invocation

    # The first engine answers; its message, as the real result seam hands it to the loop, joins the transcript.
    first = ModelTurnState()
    invocation = upload(first, "att-1", build([{"role": "user", "content": "start"}], first))
    try:
        answer, _usage = invocation.finish({"outcome": "completed", "route": dict(route), "message": {
            "role": "assistant", "content": "first answer", "nativeContinuation": {
                "route": dict(route), "format": "opaque-v1", "payload": "issued-by-the-first-engine"}}})
    finally:
        invocation.close()
    ctx, limit_ctx = _loop_ctx(tmp_path, task_id="cold-task")
    limit_ctx.messages = [{"role": "user", "content": "start"}, answer]
    assert wc.save_round(limit_ctx, "post_batch")
    restored = wc.read_working(tmp_path, "cold-task", 1)[1]["messages"]  # what a cold restore reads back
    assert restored == limit_ctx.messages
    restored.append({"role": "user", "content": "go on"})
    # The wire of a restored conversation is the one it was before this binding: the transcript without the host's
    # own (``_``-prefixed) provenance on its continuations.
    legacy = copy.deepcopy(restored)
    for message in legacy:
        native_route = (message.get("nativeContinuation") or {}).get("route") or {}
        for key in [key for key in native_route if key.startswith("_")]:
            native_route.pop(key)

    same = ModelTurnState()  # the restored loop's fresh slot, and the engine that issued the conversation
    payload = build(restored, same)
    assert json.dumps(payload, sort_keys=True) == json.dumps(build(legacy, ModelTurnState()), sort_keys=True)
    assert payload["account"] == {"mode": "auto", "preferredProfileId": "first-engine-account"}
    upload(same, "att-2", payload).close()
    assert b"issued-by-the-first-engine" in uploaded()[-1] and b"first-engine-account" in uploaded()[-1]

    count = len(uploaded())
    if change == "store":  # the same address and home, a reinstalled engine
        host.state["store"] = "store-remote-2"
        remote_engines.retire("vps")
    else:
        host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@vds-moved"}])
    fresh = ModelTurnState()  # restored again: its first request after the re-point
    payload = build(restored, fresh)
    assert payload["account"] == {"mode": "auto", "preferredProfileId": "first-engine-account"}  # what it would send
    try:
        upload(fresh, "att-3", payload).close()
    except ClaudexorModelNotDispatched as refused:
        assert refused.code == "model_turn_engine_changed"
    else:
        leaked = uploaded()[-1]
        pytest.fail("the restored conversation reached the other engine: its continuation "
                    f"{b'issued-by-the-first-engine' in leaked}, the first engine's account {b'first-engine-account' in leaked}")
    assert len(uploaded()) == count  # nothing of the first engine's conversation reached the other one
    new = ModelTurnState()  # a new task binds to the engine the connection names now
    upload(new, "att-4", build([{"role": "user", "content": "new task"}], new)).close()
    assert len(uploaded()) == count + 1 and b"first-engine-account" not in uploaded()[-1]
    if change != "store":  # the name pointed back: the restored conversation resumes on the engine that issued it
        host.set_rows(host.rows)
        back = ModelTurnState()
        upload(back, "att-5", build(restored, back)).close()
        assert b"issued-by-the-first-engine" in uploaded()[-1]


@pytest.mark.parametrize("change", ["store", "ssh_host"])
def test_a_reviewers_rounds_without_a_turn_slot_stay_on_the_engine_that_issued_their_continuations(host, tmp_path,
                                                                                                     change):
    """A reviewer's native tool rounds keep no turn slot (``NativeToolRoundReviewExecutor._chat_kwargs`` names none),
    yet each round re-sends the continuations its engine handed back and the account their route names. The engine that
    issued them is proven against the gateway's own handshake before any upload, slot or none (the round's target holds
    it, as ``chat_claudexor`` passes one target to both seams): the same engine receives the stateless request it
    received before (no turn field, no host stamp); another engine behind the same name is refused before upload, also
    for a round rebuilt from its already stripped send copy; an unstamped (legacy) conversation is sent as before."""
    import copy
    import hashlib

    from ouroboros.llm_claudexor import ClaudexorModelNotDispatched, _ModelInvocation, _request

    drive = tmp_path / "drive"
    drive.mkdir()
    target = {"connection": "vps", "source": "codex", "resolved_model": "gpt-6-remote",
              "usage_model": "claudexor@vps::codex=gpt-6-remote"}
    params = {"model_role": "reviewer:seat", "model_account_override": "", "timeout": 30}  # no model_turn_state
    route = {"source": "codex", "model": "gpt-6-remote", "credentialProfileId": "issuer-account"}

    def uploaded():
        return [item["bytes"] for item in host.state.get("uploads", {}).values() if item["bytes"] is not None]

    def send(round_target, payload, key):  # the round's one target, given to the request and its invocation alike
        data = json.dumps(payload, sort_keys=True).encode("utf-8")
        invocation = _ModelInvocation(round_target, payload, dict(params))
        try:
            invocation.prepare(SimpleNamespace(attempt_id=key, drive_root=drive), canonical_bytes=data,
                               sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        except BaseException:
            invocation.close()
            raise
        return invocation

    first = dict(target)
    invocation = send(first, _request(first, [{"role": "user", "content": "review"}], [], dict(params)), "seat-1")
    try:
        answer, _usage = invocation.finish({"outcome": "completed", "route": dict(route), "message": {
            "role": "assistant", "content": "a tool call", "nativeContinuation": {
                "route": dict(route), "format": "opaque-v1", "payload": "issued-by-the-first-engine"}}})
    finally:
        invocation.close()
    rounds = [{"role": "user", "content": "review"}, answer, {"role": "user", "content": "the tool's result"}]
    same = dict(target)
    payload = _request(same, copy.deepcopy(rounds), [], dict(params))
    assert "nativeContinuation" not in payload  # a stateless round: no turn field, as before
    send(same, payload, "seat-2").close()
    assert b"issued-by-the-first-engine" in uploaded()[-1] and b"_host_engine" not in uploaded()[-1]
    count = len(uploaded())
    if change == "store":  # the same address and home, a reinstalled engine
        host.state["store"] = "store-remote-2"
        remote_engines.retire("vps")
    else:
        host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@vds-moved"}])
    moved = dict(target)
    payload = _request(moved, copy.deepcopy(rounds), [], dict(params))
    assert payload["account"] == {"mode": "auto", "preferredProfileId": "issuer-account"}  # what it would send
    for key, rebuilt in (("seat-3", payload), ("seat-4", _request(moved, payload["messages"], [], dict(params)))):
        with pytest.raises(ClaudexorModelNotDispatched) as refused:
            send(moved, rebuilt, key).close()
        assert refused.value.code == "model_turn_engine_changed", key
    assert len(uploaded()) == count  # nothing of the first engine's conversation reached the other one
    legacy = copy.deepcopy(rounds)  # recorded before an issuer was stamped: nothing names one, so it goes as before
    legacy[1]["nativeContinuation"]["route"].pop("_host_engine")
    unstamped = dict(target)
    send(unstamped, _request(unstamped, legacy, [], dict(params)), "seat-5").close()
    assert len(uploaded()) == count + 1


def test_an_owned_engine_conversation_is_restored_and_sent_as_before():
    """The owned engine's answer (its gateway carries no remote reference) is handed to the loop as the engine wrote
    it, and a restored local conversation's continuation and account are sent verbatim while a fresh slot stays
    unbound: exactly as before the remote binding existed."""
    from ouroboros.llm_claudexor import ModelTurnState, _ModelInvocation, _request

    target = {"connection": "", "source": "codex", "resolved_model": "gpt-6", "usage_model": "claudexor::codex=gpt-6"}
    route = {"source": "codex", "model": "gpt-6", "credentialProfileId": "local-main", "accountFingerprint": "fp"}
    written = {"role": "assistant", "content": "a", "nativeContinuation": {"route": dict(route), "payload": "local"}}
    invocation = _ModelInvocation(dict(target), {"messages": []}, {"timeout": 30})
    invocation.gateway = SimpleNamespace(engine_ref=None, close=lambda: None)
    answer, _usage = invocation.finish({"outcome": "completed", "route": dict(route), "message": json.loads(json.dumps(written))})
    assert answer == written
    messages = [{"role": "user", "content": "hi"}, answer, {"role": "user", "content": "go on"}]
    slot = ModelTurnState()
    payload = _request(dict(target), json.loads(json.dumps(messages)), [],
                       {"model_role": "main", "model_account_override": "", "model_turn_state": slot})
    assert payload["messages"] == messages and payload["account"] == {"mode": "auto", "preferredProfileId": "local-main"}
    assert slot.identity == ""


def test_a_remote_send_never_prefers_another_engines_account():
    """The soft account choice of an Auto send (``llm_substitution.send_account``) stays on its engine: the account a
    local answer or a plan built for the local route rests on is never preferred on a remote engine of the same source
    and model. A route that moves to the remote engine loses the local continuations at the loop's switch seam
    (``sanitize_reasoning_on_model_switch``), and a plan names an account only for its own engine's route. An
    explicit pin stays the pin. The same facts on the owned engine are the positive controls."""
    from ouroboros.llm import LLMClient
    from ouroboros.llm_substitution import send_account

    local_account = {"source": "codex", "model": "gpt-6", "credentialProfileId": "local-main", "accountFingerprint": "fp"}
    history = [{"role": "user", "content": "hi"},
               {"role": "assistant", "content": "a", "nativeContinuation": {"route": dict(local_account)}}]
    owned = {"source": "codex", "resolved_model": "gpt-6", "connection": ""}
    remote = {**owned, "connection": "vps"}
    auto = {"model_role": "main", "model_account_override": ""}
    plan = SimpleNamespace(model_route=dict(local_account), provider="claudexor", account_pin="", model_role="main",
                           window_witness_bound=True, stale=False, status="confirmed", model="claudexor::codex=gpt-6")
    assert send_account(owned, history, auto)[0] == {"mode": "auto", "preferredProfileId": "local-main"}  # control
    assert send_account(owned, [], auto, plan=plan)[0] == {"mode": "auto", "preferredProfileId": "local-main"}
    moved = LLMClient.sanitize_reasoning_on_model_switch(history, "claudexor::codex=gpt-6", "claudexor@vps::codex=gpt-6")
    assert send_account(remote, moved, auto)[0] == {"mode": "auto"}
    assert send_account(remote, moved, auto, plan=plan)[0] == {"mode": "auto"}  # a local plan lends the remote nothing
    remote_plan = SimpleNamespace(**{**vars(plan), "model": "claudexor@vps::codex=gpt-6",
                                     "model_route": {**local_account, "credentialProfileId": "remote-main"}})
    assert send_account(remote, moved, auto, plan=remote_plan)[0] == {"mode": "auto", "preferredProfileId": "remote-main"}
    assert send_account(remote, history, {"model_role": "main", "model_account_override": "remote-main"})[0] == {
        "mode": "pin", "profileId": "remote-main"}


def test_disconnect_never_reads_an_unconfirmed_tunnel_stop_as_closed(host, monkeypatch):
    """The native stop seam's outcome is the answer, never "a descriptor existed": a stop it could not confirm, or
    could not carry out, keeps the descriptor and the custody row, and the route, the engine status and a Save's
    warning say so; a confirmed stop closes, and a second Disconnect finds none running."""
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from ouroboros import process_custody
    from ouroboros.gateway.engine_connections import engine_connection_routes
    from ouroboros.gateway.settings import _retire_unsaved_engine_tunnels

    remote_engines.gateway("vps").close()
    pid = _tunnel_pids("vps")[0]
    real = process_custody.stop_ledgered_processes

    def unconfirmed_stop(drive_root, purposes, *, timeout_sec=5.0, unconfirmed=None, **_kwargs):
        unconfirmed.append(f"process {pid} tree exit unconfirmed")  # as the seam reports a survivor
        return []

    def unreadable_ledger(*_args, **_kwargs):
        raise OSError("ledger unreadable")

    with TestClient(Starlette(routes=engine_connection_routes())) as client:
        with monkeypatch.context() as patched:  # undone even on a failure, so cleanup stops through the real seam
            for failing in (unconfirmed_stop, unreadable_ledger):
                patched.setattr(process_custody, "stop_ledgered_processes", failing)
                assert client.post("/api/engine-connections/vps/disconnect", json={}).json() == {"name": "vps",
                                                                                                 "closed": None}
                assert remote_engines.published("vps") is not None and _alive(pid)  # evidence kept, never "closed"
                [row] = client.get("/api/engine-connections").json()["connections"]
                # the retained descriptor still answers, yet the status keeps the unconfirmed stop, never "connected"
                assert row["observation"]["state"] == "stop_unconfirmed" and row["observation"]["message"]
            warning = _retire_unsaved_engine_tunnels({remote_engines.ENGINE_CONNECTIONS_KEY: host.rows},
                                                     {remote_engines.ENGINE_CONNECTIONS_KEY: []})
            assert len(warning) == 1 and "could not confirm its tunnel to it stopped" in warning[0]
        assert process_custody.stop_ledgered_processes is real
        assert client.post("/api/engine-connections/vps/disconnect", json={}).json()["closed"] is True
        assert remote_engines.published("vps") is None and not _alive(pid)
        [row] = client.get("/api/engine-connections").json()["connections"]
        assert row["observation"]["state"] == "offline"  # a later confirmed stop is what resolves it
        assert client.post("/api/engine-connections/vps/disconnect", json={}).json()["closed"] is False


def _forwards(host):
    return len([call for call in host.ssh_calls() if call["forward"]])


@pytest.mark.parametrize("cause", ["survivor", "unreadable_ledger"])
def test_a_reattach_opens_no_second_forward_while_the_first_is_not_confirmed_stopped(host, monkeypatch, cause):
    """Re-pointing a connection (or re-attaching after its forward stopped answering) first stops the earlier forward
    through the custody seam. A stop the seam does not confirm (a survivor it keeps), or a custody ledger that cannot
    be read, is a typed refusal: the earlier descriptor stays, no second forward is spawned beside it, and the row says
    so. Once the stop is confirmed the same re-point attaches normally."""
    import os

    from ouroboros import config, process_custody

    remote_engines.gateway("vps").close()
    first = remote_engines.published("vps")
    pid = int(first["pid"])
    host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@vds-moved"}])  # the row now names another address
    ledger = process_custody.ledger_path(pathlib.Path(config.DATA_DIR))
    before = _forwards(host)
    with monkeypatch.context() as patched:
        if cause == "survivor":
            def survivor(drive_root, purposes, *, timeout_sec=5.0, unconfirmed=None, **_kwargs):
                unconfirmed.append(f"process {pid} tree exit unconfirmed")  # as the seam reports one it keeps
                return []

            patched.setattr(process_custody, "stop_ledgered_processes", survivor)
        else:
            os.chmod(ledger, 0)  # the seam reads it leniently as empty; custody's strict read says unreadable
        try:
            for _attempt in range(2):  # asking again does not spawn one either
                with pytest.raises(ClaudexorUnavailable) as refused:
                    remote_engines.tunnel("vps")
                assert refused.value.code == "remote_engine_stop_unconfirmed"
                assert _forwards(host) == before and remote_engines.published("vps") == first and _alive(pid)
            [row] = remote_engines.status()
            assert row["observation"]["state"] == "stop_unconfirmed"
        finally:
            os.chmod(ledger, 0o600)
    moved = remote_engines.tunnel("vps")  # the stop confirmed: the re-pointed row attaches
    assert _forwards(host) == before + 1 and moved["ssh_host"] == "ouroboros@vds-moved" and not _alive(pid)


def test_a_failed_handshake_whose_forward_is_not_confirmed_stopped_keeps_its_descriptor(host, monkeypatch):
    """A new forward whose engine handshake fails is stopped through the custody seam; if that stop is not confirmed
    its descriptor stays as the evidence of what may still run, the refusal says so (chained to the handshake's own),
    and no later attach spawns another forward beside it. Panic's close still drops the descriptor and token for
    safety, returns the unconfirmed stop and leaves the custody row; a confirmed stop then closes it."""
    from ouroboros import process_custody

    real = process_custody.stop_ledgered_processes
    before = _forwards(host)

    def refused_handshake(self, **_kwargs):
        raise ClaudexorUnavailable("compatibility_refused", "the engine refused this client")

    def survivor(drive_root, purposes, *, timeout_sec=5.0, unconfirmed=None, **kwargs):
        live = process_custody.pending_process_stops(drive_root, purposes)  # only a forward custody holds survives
        if not live:
            return real(drive_root, purposes, timeout_sec=timeout_sec, unconfirmed=unconfirmed, **kwargs)
        unconfirmed.extend(live)
        return []

    with monkeypatch.context() as patched:
        patched.setattr(remote_engines.RemoteEngineGateway, "handshake", refused_handshake)
        patched.setattr(process_custody, "stop_ledgered_processes", survivor)
        with pytest.raises(ClaudexorUnavailable) as refused:
            remote_engines.tunnel("vps")
        assert refused.value.code == "remote_engine_stop_unconfirmed"
        assert refused.value.__cause__.code == "compatibility_refused"
        kept = remote_engines.published("vps")
        assert kept is not None and _alive(int(kept["pid"])) and _forwards(host) == before + 1
        with pytest.raises(ClaudexorUnavailable):
            remote_engines.tunnel("vps")  # its descriptor proves no store: attaching again would spawn beside it
        assert _forwards(host) == before + 1
        assert remote_engines.status()[0]["observation"]["state"] == "stop_unconfirmed"
        closed = remote_engines.panic_close(["vps"])
        assert closed["tunnels"]["vps"]["unconfirmed"] and remote_engines.published("vps") is None
        assert _alive(int(kept["pid"]))  # Panic drops the token; the process stays in custody, never called gone
    assert remote_engines.retire("vps") is True and not _alive(int(kept["pid"]))  # a confirmed stop closes it


def test_a_tunnel_is_known_by_its_forwards_custody_never_by_whatever_holds_its_port(host, monkeypatch):
    """A tunnel is live while its forward's own custody row is (the published pid, its measured birth and command),
    never because something answers on its port: a status read opens no connection (none toward the host), and once
    the forward died, another local process listening on its old port is not mistaken for the tunnel: the next use
    attaches a new forward and never hands that process the engine's token."""
    import os
    import signal
    import socket
    import threading

    remote_engines.gateway("vps").close()
    first = remote_engines.published("vps")
    opened, real_connect = [], socket.create_connection
    with monkeypatch.context() as probing:
        probing.setattr(socket, "create_connection", lambda *a, **k: opened.append(a) or real_connect(*a, **k))
        [row] = remote_engines.status()  # the process that opened it: its own child
        assert row["observation"]["state"] == "connected"
        probing.setattr(remote_engines, "_HELD", {})  # any other process: the forward's custody row
        assert remote_engines.status()[0]["observation"]["state"] == "connected"
    assert opened == []  # passive: no socket, no host contact
    os.kill(int(first["pid"]), signal.SIGKILL)  # the forward dies on its own (no Disconnect)
    remote_engines._HELD["vps"].wait(timeout=10)
    with monkeypatch.context() as elsewhere:
        elsewhere.setattr(remote_engines, "_HELD", {})
        assert remote_engines.status()[0]["observation"]["state"] != "connected"  # its custody row is not live
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", int(first["port"])))  # another local process takes the port
    listener.listen(8)
    listener.settimeout(0.2)
    received, done = [], threading.Event()

    def accept():
        while not done.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(2)
                try:
                    received.append(conn.recv(65536))
                except OSError:
                    received.append(b"")

    taker = threading.Thread(target=accept, daemon=True)
    taker.start()
    try:
        assert remote_engines.status()[0]["observation"]["state"] != "connected"
        remote_engines.gateway("vps").close()  # attaches a new forward of its own
    finally:
        done.set()
        taker.join(timeout=5)
        listener.close()
    again = remote_engines.published("vps")
    assert again["pid"] != first["pid"] and again["port"] != first["port"]
    assert received == [] and not any(TOKEN.encode() in data for data in received)


def test_a_remote_catalog_read_asks_that_engine(host):
    from ouroboros.llm_capability_policy import model_catalog

    catalog = model_catalog("codex", None, engine="vps")
    assert catalog["models"] == [{"id": "gpt-6-remote"}]
    assert any(call["path"] == "/v2/model-sources/codex/models" for call in host.state["calls"])


def test_a_remote_reviewer_row_keeps_its_mark_and_its_seat_starts_on_that_engine(host, tmp_path):
    from ouroboros.configured_subagents import parse_configured_subagents
    from ouroboros.review_custody import _attempt_key
    from ouroboros.review_execution import session_route_for_review_slot
    from ouroboros.reviewer_slot_config import _catalog_row_slot

    row = {"subagent_id": "remote-reviewer", "recommended_use": "review", "access": "readonly", "review_eligible": True,
           "route": {"kind": "agent_session", "target_id": "codex=gpt-6", "credential_profile_id": "remote-main",
                     "engine": "vps"}}
    parsed = parse_configured_subagents({"enabled": True, "items": [row]})
    slot = _catalog_row_slot(parsed.items[0], {})
    assert slot.engine == "vps" and session_route_for_review_slot(slot).engine == "vps"
    request = SimpleNamespace(retry_key="", surface="review", task_id="t", call_type="", session_root="",
                              slot_messages={})
    local_slot = _catalog_row_slot(parse_configured_subagents({"enabled": True, "items": [
        {**row, "route": {**row["route"], "engine": ""}}]}).items[0], {})
    assert _attempt_key(request, slot) != _attempt_key(request, local_slot)  # where a seat runs is its identity
    checkout = _local_copy(tmp_path)
    (checkout / ".gitignore").write_text("build/\n.review-drive/\n")
    view = checkout / ".review-drive" / "t-review" / "turn.txt"
    view.parent.mkdir(parents=True)
    view.write_text("the paged review prompt the host materialized")
    facts, gateway = domain.seat_gateway(None, str(checkout), "vps")
    try:
        assert facts["engine_ref"]["connection"] == "vps" and facts["engine_ref"]["store"] == STORE
        assert gateway.stages_workspace  # the same staged transport a delegated run uses
        gateway.start_run({"prompt": "review", "instructions": f"Read {view}.", "access": "readonly",
                           "scope": {"kind": "project", "root": str(checkout), "ephemeral": True}},
                          idempotency_key="inv-seat")
    finally:
        gateway.close()
    staged = host.remote_root / "inv-seat" / "work" / ".review-drive" / "t-review" / "turn.txt"
    assert staged.read_text() == "the paged review prompt the host materialized"  # the ignored view travelled
    sent = next(call["body"] for call in host.state["calls"] if call["path"] == "/v2/runs" and call["key"] == "inv-seat")
    # the host's words stay as written; the map names where the view's folder is on the engine's host
    assert sent["instructions"].startswith(f"Read {view}.") and f"{checkout} -> {staged.parents[2]}" in sent["instructions"]


# --- Emergency Stop: owed stops, unknown starts and the re-adoption gate (``remote_stops``) ------------------------

def _remote_ref():
    client = remote_engines.gateway("vps")
    try:
        return dict(client.engine_ref)
    finally:
        client.close()


def _request_start(data, invocation, ref, root, task="t-lost"):
    pathlib.Path(data).mkdir(parents=True, exist_ok=True)
    body = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(root)},
            "harnesses": ["codex"], "primaryHarness": "codex"}
    assert custody.emit(data, custody.START_REQUESTED, {
        "invocation_id": invocation, "task_id": task, "root_task_id": task, "request": body, "engine_ref": ref,
        "idempotency_key": invocation, "route": "codex", "project_id": ""})
    return body


def _pending(data, invocation):
    return [row for row in custody.pending_invocations(data) if row["invocation_id"] == invocation]


def _posts(host):
    return [call for call in host.state["calls"] if call["path"] == "/v2/runs" and call["method"] == "POST"]


def _OWNED_GATEWAY():
    raise ClaudexorUnavailable("daemon_unreachable", "no owned engine answers in this test")


@pytest.mark.parametrize("task_groups", [True, False], ids=["task-group", "run-folder"])
def test_a_start_whose_answer_was_lost_is_found_on_its_store_and_stopped_never_sent_again(host, tmp_path, task_groups):
    """The POST was accepted and its answer lost; Emergency Stop then owes the start a stop. It is found on the store
    its invocation recorded (by its task group, or, on an engine without task groups, by its run folder), bound with
    the intent on its row and cancelled; a retry by its task is refused, and nothing is ever POSTed again."""
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"  # this test's own custody (the installation's tunnel stays under DATA_DIR)
    invocation = f"inv-lost-{'group' if task_groups else 'folder'}"
    host.state["task_groups"] = task_groups
    copy = _local_copy(tmp_path)
    ref = _remote_ref()
    body = _request_start(data, invocation, ref, copy, task=invocation)
    host.state["runs"].append({"runId": "run-other", "key": "other-key", "taskGroupId": remote_stops.task_group_id(
        "other"), "root": "/srv/other", "executionRoot": "/srv/other", "prompt": "q",
        "createdAt": "2026-11-10T00:00:00Z"})  # another client's run on the same engine
    client = remote_engines.gateway("vps")
    client.custody_root = data  # as the delegate start binds it: its phase facts ride its own invocation
    host.state["lose_start_answer"] = True
    try:
        with pytest.raises(ClaudexorUnavailable):
            client.start_run(body, idempotency_key=invocation)
    finally:
        client.close()
        host.state["lose_start_answer"] = False
    [pending] = _pending(data, invocation)
    assert pending["remote_start_phase"] == "posting"  # the POST may have left: sent-unknown, never "not sent"
    ours = host.state["runs"][-1]
    assert ours["key"] == invocation and bool(ours["taskGroupId"]) is task_groups  # accepted on the engine
    assert ours["taskGroupId"] in (None, remote_stops.task_group_id(invocation))
    outcome = remote_stops.panic_stop_work(data)  # the server's own door: intent first, then the addressed stop
    assert outcome["owed"] == {"runs": 0, "starts": 1}
    [pending] = _pending(data, invocation)
    assert remote_stops.owed(pending)
    with pytest.raises(ClaudexorUnavailable) as refused:  # its task's retry: never re-sent
        domain.start_engine(data, invocation)
    assert refused.value.code == "remote_stop_owed"
    posts = len(_posts(host))
    host.state["run_states"] = {ours["runId"]: "running"}
    # its owner is live and both the read and a replay are admitted: the owed path alone resolves it (never the
    # owner's watched read, which a remote start does not get, nor a replay)
    outcomes = reconcile_orphaned_runs(data, running_task_ids={invocation}, gateway_factory=_OWNED_GATEWAY,
                                       observe_pending=lambda: True, replay_pending=lambda: True)
    row = next(row for row in outcomes if row.get("run_id") == ours["runId"])
    assert row["action"] == "cancelled" and len(_posts(host)) == posts  # found, bound, stopped; no POST
    bound = current.run(data, ours["runId"])
    assert bound.invocation_id == invocation and remote_stops.owed(bound) and bound.engine_ref["store"] == STORE
    assert not _pending(data, invocation)
    stopped = [call["path"] for call in host.state["calls"] if call["path"].endswith("/control")]
    assert "/v2/runs/run-other/control" not in stopped and f"/v2/runs/{ours['runId']}/control" in stopped
    host.state["run_states"] = {ours["runId"]: "cancelled"}  # a terminal read proves the stop
    reconcile_orphaned_runs(data, running_task_ids={invocation}, gateway_factory=_OWNED_GATEWAY,
                            observe_pending=lambda: True, replay_pending=lambda: True)
    assert not remote_stops.owed(current.run(data, ours["runId"])) and len(_posts(host)) == posts


@pytest.mark.parametrize("consumer", ["orphan_sweep", "owned_sweep", "retry", "seat_rejoin"])
def test_a_lost_remote_start_is_never_looked_up_by_a_body_its_engine_did_not_accept(host, tmp_path, consumer):
    """#3138's noncreating read sends a start's RECORDED body, but a remote engine accepted the copy staged on its host
    (other roots, an appended host path map, its ``taskGroupId``): under the same key that is the engine's conflict.
    So even an engine that declares the read is never asked it by a remote start (``RemoteEngineGateway.lookup_run``
    refuses). Every reader looks for the start on its recorded store instead, by its task group and run folder
    (``remote_stops.locate_started``), and binds the run it finds once: an orphan's sweep (then reconciled as any
    orphan run), a live owner's sweep at its bounded pace, an explicit retry's read, a review seat's rejoin read.
    Nothing is cancelled, replayed or sent, and no second run is ever created."""
    from ouroboros.delegate_accepted import recover
    from ouroboros.delegate_shared import delegate_payload
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs
    from ouroboros.gateways.claudexor import ClaudexorGateway

    data, invocation = tmp_path / "custody", f"inv-lookup-{consumer}"
    host.state["run_lookup"] = True  # an engine that declares the read (no served engine does today)
    body = _request_start(data, invocation, _remote_ref(), _local_copy(tmp_path), task=invocation)
    client = remote_engines.gateway("vps")
    client.custody_root = data
    host.state["lose_start_answer"] = True  # accepted on the engine, its answer lost on the way back
    try:
        with pytest.raises(ClaudexorUnavailable):
            client.start_run(body, idempotency_key=invocation)
    finally:
        client.close()
        host.state["lose_start_answer"] = False
    ours, runs = host.state["runs"][-1], len(host.state["runs"])
    host.state["run_states"] = {ours["runId"]: "running"}
    probe = remote_engines.gateway("vps")
    try:
        with pytest.raises(ClaudexorUnavailable) as conflict:  # what the unadapted read would have sent: refused there
            ClaudexorGateway.lookup_run(probe, body, idempotency_key=invocation)
    finally:
        probe.close()
    assert conflict.value.status_code == 409 and conflict.value.code == "idempotency_conflict"
    reads = len([call for call in host.state["calls"] if call["path"] == "/v2/runs/lookup"])
    posts, calls = len(_posts(host)), len(host.state["calls"])
    if consumer.endswith("sweep"):
        live = set() if consumer == "orphan_sweep" else {invocation}
        outcomes = reconcile_orphaned_runs(data, running_task_ids=live, gateway_factory=_OWNED_GATEWAY,
                                           observe_pending=lambda: True, replay_pending=lambda: True)
        if consumer == "orphan_sweep":  # found and bound, then reconciled as any orphan: its owner's end is no verdict
            [row] = [row for row in outcomes if row.get("run_id") == ours["runId"]]
            assert row["action"] == "left_live", row
        else:  # its owner holds it: only what the engine accepted is read and bound, every decision stays the owner's
            [row] = [row for row in outcomes if row.get("invocation_id") == invocation]
            assert row["action"] == "accepted_start_bound" and row["run_id"] == ours["runId"], row
    elif consumer == "retry":
        from ouroboros.delegate_invocation import invocation_record
        from ouroboros.tools.delegate_integration import _accepted_retry

        answer = _accepted_retry(data, invocation, {**invocation_record(data, invocation), "request": body})
        started = delegate_payload(answer)
        assert started["status"] == "started" and started["run_id"] == ours["runId"], started
        assert started["recovered_without_start"] is True
    else:  # a review seat's recorded engine, then its own read (``review_session_custody.rejoin_recorded_review``)
        record = {**_pending(data, invocation)[0], "request": body}
        _facts, gateway = domain.seat_gateway(record, str(tmp_path), custody_root=data)
        try:
            held, read = recover(data, gateway, {**record, "invocation_id": invocation})
        finally:
            gateway.close()
        assert held is not None and held.run_id == ours["runId"] and read["action"] == "accepted_start_bound", read
    assert len([call for call in host.state["calls"] if call["path"] == "/v2/runs/lookup"]) == reads  # never read
    assert len(host.state["runs"]) == runs and len(_posts(host)) == posts  # nothing sent, never a second run
    assert not [call for call in host.state["calls"][calls:] if call["path"].endswith("/control")]  # nothing stopped
    bound = current.run(data, ours["runId"])
    assert bound.invocation_id == invocation and bound.engine_ref["store"] == STORE and not _pending(data, invocation)


def test_an_owed_start_nothing_lists_stays_owed_and_is_never_closed_by_time_or_an_empty_answer(host, tmp_path,
                                                                                               monkeypatch):
    """The engine lists only its currently served generations: an empty lookup is no proof, whatever its age. The
    start keeps its owed stop (visible on its engine's row), is looked for again only when due (the first pass of a
    process, then after a bounded pause, attaching when no tunnel is open), and is never re-sent."""
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"
    _request_start(data, "inv-never", _remote_ref(), _local_copy(tmp_path))
    custody.emit(data, custody.REMOTE_START_PHASE, {"run_id": "", "invocation_id": "inv-never", "phase": "staging"})
    custody.emit(data, custody.REMOTE_START_PHASE, {"run_id": "", "invocation_id": "inv-never", "phase": "posting"})
    custody.emit(data, custody.REMOTE_START_PHASE, {"run_id": "", "invocation_id": "inv-never", "phase": "staging"})
    remote_stops.record_owed_stops(data)
    [pending] = _pending(data, "inv-never")
    assert pending["remote_start_phase"] == "posting"  # a phase never moves back
    monkeypatch.setattr(remote_stops, "_LOOKED_UP", {})
    outcomes = reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    [row] = [row for row in outcomes if row.get("invocation_id") == "inv-never"]
    assert (row["action"], row["reason"]) == ("invocation_retained", "remote_start_not_found")
    assert row["lookup"]["task_group"] == "read" and remote_stops.owed(_pending(data, "inv-never")[0])
    lookups = len([call for call in host.state["calls"] if "/v2/task-groups/" in call["path"]])
    assert reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY) == []  # not due
    assert len([call for call in host.state["calls"] if "/v2/task-groups/" in call["path"]]) == lookups
    monkeypatch.setattr(remote_stops, "_RELOOK_SEC", 0.0)  # the bounded pause has passed: asked again
    assert reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    remote_engines.retire("vps")  # no tunnel is open now: the next due pass attaches by itself
    forwards = len([call for call in host.ssh_calls() if call["forward"]])
    again = reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    assert [(row["action"], row["reason"]) for row in again if row.get("invocation_id") == "inv-never"] == [
        ("invocation_retained", "remote_start_not_found")]
    assert len([call for call in host.ssh_calls() if call["forward"]]) == forwards + 1
    assert remote_stops.owed_stop_counts(data) == {"vps": {"runs": 0, "starts": 1}}  # still owed and visible
    assert not _posts(host)  # never re-sent


@pytest.mark.parametrize("first", ["unreachable", "empty"])
def test_an_owed_start_is_looked_for_again_after_its_tunnel_died_and_stopped_once_its_engine_answers(
        host, tmp_path, monkeypatch, first):
    """The first lookup reaches no engine, or the engine lists nothing yet; then the forward dies. With no task, no
    Check and no restart asking, the ordinary maintenance pass asks again after its bounded pause: it attaches, finds
    the accepted run, cancels it and keeps the stop owed until a terminal read-back. Nothing is POSTed again."""
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data, invocation = tmp_path / "custody", f"inv-relook-{first}"
    body = _request_start(data, invocation, _remote_ref(), _local_copy(tmp_path), task=invocation)
    client = remote_engines.gateway("vps")
    client.custody_root = data
    host.state["lose_start_answer"] = True  # accepted on the engine, its answer lost on the way back
    try:
        with pytest.raises(ClaudexorUnavailable):
            client.start_run(body, idempotency_key=invocation)
    finally:
        client.close()
        host.state["lose_start_answer"] = False
    ours = host.state["runs"][-1]
    host.state["run_states"] = {ours["runId"]: "running"}
    remote_stops.record_owed_stops(data)
    monkeypatch.setattr(remote_stops, "_LOOKED_UP", {})
    posts = len(_posts(host))

    def sweep():
        return reconcile_orphaned_runs(data, running_task_ids={invocation}, gateway_factory=_OWNED_GATEWAY)

    def forwards():
        return len([call for call in host.ssh_calls() if call["forward"]])

    if first == "unreachable":
        remote_engines.retire("vps")
        host.descriptor.rename(host.descriptor.with_suffix(".gone"))
        assert not [row for row in sweep() if row.get("run_id") == ours["runId"]]
        host.descriptor.with_suffix(".gone").rename(host.descriptor)
    else:
        host.state["runs"].remove(ours)  # the engine's current lens does not list it
        [row] = [row for row in sweep() if row.get("invocation_id") == invocation]
        assert (row["action"], row["reason"]) == ("invocation_retained", "remote_start_not_found")
        host.state["runs"].append(ours)
    assert remote_stops.owed(_pending(data, invocation)[0])
    remote_engines.retire("vps")  # the forward is gone
    before = forwards()
    assert sweep() == [] and forwards() == before  # within the pause: nothing is asked, nothing attached
    monkeypatch.setattr(remote_stops, "_RELOOK_SEC", 0.0)  # the pause has passed
    row = next(row for row in sweep() if row.get("run_id") == ours["runId"])
    assert row["action"] == "cancelled" and forwards() == before + 1  # the sweep re-attached by itself
    assert any(call["path"] == f"/v2/runs/{ours['runId']}/control" for call in host.state["calls"])
    assert remote_stops.owed(current.run(data, ours["runId"]))  # owed until its engine reads it back stopped
    host.state["run_states"] = {ours["runId"]: "cancelled"}
    sweep()
    assert not remote_stops.owed(current.run(data, ours["runId"])) and not _pending(data, invocation)
    assert len(_posts(host)) == posts  # never re-sent


def test_the_boot_pass_reaches_no_remote_engine_and_the_first_periodic_pass_does(host, tmp_path):
    """The boot's reconcile pass never waits on SSH (``server_maintenance._startup_custody_sweep``): every remote
    engine's work, an owed stop included, is left to the periodic sweep, off the loop, whose first pass follows
    readiness and attaches as needed. The boot reads and marks nothing remote, so that first pass still looks for an
    unknown start at once and delivers the owed stop."""
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"
    ref = _remote_ref()
    assert custody.record_started(data, custody.RunCustody(
        run_id="run-boot", task_id="t-boot", invocation_id="inv-run-boot", route_id="codex", model="m", access="full",
        mode="ask", ledger_root=str(data), engine_ref=dict(ref), daemon_store_id=STORE))
    _request_start(data, "inv-boot-start", ref, tmp_path, task="t-boot")
    remote_stops.record_owed_stops(data)
    remote_engines.retire("vps")  # a boot: no tunnel is open
    host.state["run_states"] = {"run-boot": "running"}
    ssh, calls = len(host.ssh_calls()), len(host.state["calls"])
    assert reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY,
                                   observe_pending=lambda: True, replay_pending=lambda: True,
                                   remote_engines=False) == []
    assert len(host.ssh_calls()) == ssh and len(host.state["calls"]) == calls  # no SSH, no engine request
    assert remote_stops.owed(current.run(data, "run-boot")) and remote_stops.owed(_pending(data, "inv-boot-start")[0])
    outcomes = reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY,
                                       observe_pending=lambda: True, replay_pending=lambda: True)
    assert next(row for row in outcomes if row.get("run_id") == "run-boot")["action"] == "cancelled"
    assert {"invocation_id": "inv-boot-start", "reason": "remote_start_not_found"}.items() <= next(
        row for row in outcomes if row.get("invocation_id") == "inv-boot-start").items()


def test_a_start_emergency_stop_cut_during_staging_is_closed_as_not_sent_without_asking_any_engine(host, tmp_path):
    """The start's own facts settle it: ``staging`` was recorded and ``posting`` never was, so its POST never left
    this computer. Closed as definitely not started, with no engine contacted and its host folder named."""
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"
    _request_start(data, "inv-cut", _remote_ref(), _local_copy(tmp_path))
    client = remote_engines.gateway("vps")
    client.custody_root = data
    client._record_phase("inv-cut", "staging")  # the door code's own fact; then the process died mid-staging
    client.close()
    remote_engines.retire("vps")
    remote_stops.record_owed_stops(data)
    [pending] = _pending(data, "inv-cut")
    assert remote_stops.owed(pending) and remote_stops.unsent(pending)
    calls, sshes = len(host.state["calls"]), len(host.ssh_calls())
    outcomes = reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    assert [(row["action"], row["reason"]) for row in outcomes] == [("invocation_closed_unsent", "remote_start_not_sent")]
    assert not _pending(data, "inv-cut") and len(host.state["calls"]) == calls and len(host.ssh_calls()) == sshes
    record = custody.invocation_record(data, "inv-cut")
    assert record["state"] == "failed_definite" and not _posts(host)


@pytest.mark.parametrize("change", ["store", "repointed"])
def test_an_owed_start_on_another_store_or_a_repointed_row_keeps_its_intent_and_stops_nothing(host, tmp_path, change):
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"
    ref = _remote_ref()
    if change == "store":
        ref = {**ref, "store": "store-of-another-engine"}
    else:
        host.set_rows([{**host.rows[0], "ssh_host": "ouroboros@elsewhere"}])
    _request_start(data, "inv-held", ref, _local_copy(tmp_path))
    host.state["runs"].append({"runId": "run-held", "key": "inv-held", "taskGroupId": remote_stops.task_group_id(
        "inv-held"), "root": "/x", "executionRoot": "/x", "prompt": "p", "createdAt": "2026-11-11T00:00:00Z"})
    remote_stops.record_owed_stops(data)
    calls = len(host.state["calls"])
    reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    assert not [call for call in host.state["calls"][calls:] if call["path"].endswith("/control")]
    assert not _posts(host)
    [pending] = _pending(data, "inv-held")
    assert remote_stops.owed(pending)  # still owed: no other engine or store answers for it


def test_a_launcher_panic_owes_remote_stops_at_restore_and_its_receipt_waits_for_them(host, tmp_path, monkeypatch):
    """A launcher's Panic never ran the server's remote duty: the boot that restores work marks every open remote run
    and start as owing a stop before any worker exists, and the receipt is not written while that mark is not
    durable (the input is kept for the next boot)."""
    from supervisor import worker_chat_lane
    from ouroboros import remote_stops as stops
    from ouroboros.installation_stop import PANIC_RECEIPT, installation_panic
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_state_authority import _prior

    q, state, workers = _install_queue(tmp_path, monkeypatch)
    state.save_state(_prior())
    ref = _remote_ref()
    assert custody.record_started(tmp_path, custody.RunCustody(
        run_id="run-launcher", task_id="t-l", invocation_id="inv-run-launcher", route_id="codex", model="m",
        access="full", mode="ask", ledger_root=str(tmp_path), engine_ref=dict(ref), daemon_store_id=STORE))
    _request_start(tmp_path, "inv-start-launcher", ref, _local_copy(tmp_path), task="t-l")
    assert q.persist_queue_snapshot(reason="before_panic")
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    workers.PENDING.clear()
    q.restore_pending_from_snapshot()
    assert stops.owed(current.run(tmp_path, "run-launcher")) and stops.owed(_pending(tmp_path, "inv-start-launcher")[0])
    custody.resolve_containment_fault(tmp_path, current.run(tmp_path, "run-launcher"), "test_reset")
    monkeypatch.setattr(stops, "record_owed_stops", lambda root: (_ for _ in ()).throw(OSError("store busy")))
    worker_chat_lane.auto_resume_after_restart()
    assert PANIC_RECEIPT not in state.load_state() and installation_panic(tmp_path)  # the input is kept
    monkeypatch.undo()
    q, state, workers = _install_queue(tmp_path, monkeypatch)
    host.set_rows(host.rows)  # the undo above also took the connection rows away
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state().get(PANIC_RECEIPT) and stops.owed(current.run(tmp_path, "run-launcher"))


def test_a_panic_boot_without_a_remote_engine_never_reads_anything_remote(tmp_path, monkeypatch):
    """Local Panic and its boot are as on base without a remote engine connection: a broken remote module or an
    unreadable custody store cannot hold the receipt, and nothing remote is imported or waited on. Only an
    installation that once reached a remote engine (its tunnel home, which no removal deletes) reads custody without
    a connection, and custody it cannot read is unknown, never none: that receipt waits and its input is kept."""
    import shutil

    from supervisor import worker_chat_lane
    from ouroboros.installation_stop import PANIC_RECEIPT, installation_panic
    from ouroboros.server_control import _remote_panic_step
    from tests._budget_pause_exact_helpers import _install_queue
    from tests.test_state_authority import _prior

    monkeypatch.delenv(remote_engines.ENGINE_CONNECTIONS_KEY, raising=False)
    q, state, workers = _install_queue(tmp_path, monkeypatch)
    state.save_state(_prior())
    (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")

    def broken(*_args, **_kwargs):
        raise OSError("custody unreadable")

    for name in ("record_owed_stops", "panic_stop_work", "cancel_open_model_operations"):
        monkeypatch.setattr(remote_stops, name, broken)
    monkeypatch.setattr(remote_engines, "panic_close", broken)
    workers.PENDING.clear()
    q.restore_pending_from_snapshot()
    monkeypatch.setattr(current, "open_held_runs", broken)  # the custody store itself cannot be read
    reached = tmp_path.joinpath(*remote_stops.TUNNEL_HOME)
    (reached / "removed").mkdir(parents=True)  # a connection this installation once reached, since removed
    worker_chat_lane.auto_resume_after_restart()
    assert PANIC_RECEIPT not in state.load_state() and installation_panic(tmp_path)
    shutil.rmtree(reached)  # as an installation that never reached a remote engine
    worker_chat_lane.auto_resume_after_restart()
    assert state.load_state().get(PANIC_RECEIPT) and not installation_panic(tmp_path)
    assert _remote_panic_step(tmp_path) == _remote_panic_step(tmp_path, settle=True) == {
        "requested": True, "remote_engines": 0}


def test_every_re_adoption_of_a_remote_run_carries_out_its_owed_stop_first(host, tmp_path):
    """The startup/adoption race: a task given its remote run again (``run_gateway``, a continued start) before the
    sweep ran gets it only after Emergency Stop's stop is proven; an unproven stop or an unreachable engine refuses,
    and a local run is untouched."""
    data = tmp_path / "custody"
    ref = _remote_ref()
    entry = custody.RunCustody(run_id="run-adopt", task_id="t-a", invocation_id="inv-adopt", route_id="codex",
                               model="m", access="full", mode="ask", ledger_root=str(data), engine_ref=dict(ref),
                               daemon_store_id=STORE)
    assert custody.record_started(data, entry)
    remote_stops.record_owed_stops(data)
    host.state["run_states"] = {"run-adopt": "running"}
    with pytest.raises(ClaudexorUnavailable) as live:
        domain.run_gateway(entry, lambda: pytest.fail("a remote run never falls back to the owned engine"))
    assert live.value.code == "remote_stop_owed"  # the stop was sent; the run is not handed back live
    assert any(call["path"] == "/v2/runs/run-adopt/control" for call in host.state["calls"])
    with pytest.raises(ClaudexorUnavailable) as continued:
        domain.start_engine(data, continuation_token="run-adopt")
    assert continued.value.code == "remote_stop_owed"
    remote_engines.retire("vps")
    host.descriptor.rename(host.descriptor.with_suffix(".gone"))  # the engine stops answering
    with pytest.raises(ClaudexorUnavailable) as unreachable:
        domain.run_gateway(entry, lambda: None)
    assert unreachable.value.code == "remote_engine_not_running" and remote_stops.owed(current.run(data, "run-adopt"))
    host.descriptor.with_suffix(".gone").rename(host.descriptor)
    host.state["run_states"] = {"run-adopt": "cancelled"}
    gateway = domain.run_gateway(entry, lambda: None)
    try:
        assert gateway.get_run("run-adopt")["summary"]["state"] == "cancelled"  # a stopped run, never a live one
    finally:
        gateway.close()
    assert not remote_stops.owed(current.run(data, "run-adopt"))
    owned = custody.RunCustody(run_id="run-local", task_id="t-a", engine_ref={})
    assert domain.run_gateway(owned, lambda: "owned-gateway") == "owned-gateway"


def test_remote_model_recovery_honours_retry_after_on_its_own_engine_through_the_maintenance_pass(
        host, tmp_path, monkeypatch):
    """The existing consumer (``terminal_cost_reconciliation.reconcile_abandoned_usage``) reads a remote operation only
    on the engine its request recorded, waits out a received Retry-After without asking anyone, and settles the
    engine's own price once the operation is terminal: no other engine, no new operation, no guessed zero."""
    import hashlib
    from datetime import datetime as real_datetime, timedelta, timezone

    from ouroboros import claudexor_daemon, terminal_cost_reconciliation as tcr, usage_accounting as usage
    from ouroboros.observability import persist_call
    from ouroboros.task_results import write_task_result
    from supervisor import queue
    from tests._usage_store_testing import ledger_rows

    root = tmp_path / "usage"
    root.mkdir()
    monkeypatch.setattr(queue, "task_has_live_ownership", lambda *_a, **_k: False)
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda: pytest.fail("the owned engine is never asked"))
    for task in ("child", "root"):
        write_task_result(root, task, "cancelled", result="done", root_task_id="root",
                          parent_task_id="" if task == "root" else "root",
                          delegation_role="root" if task == "root" else "subagent",
                          **({"root_phase_checkpoint": {"post_task_synthesis": "completed"}} if task == "root" else {}))
    reservation = usage.reserve_attempt(usage.AttemptRequest(
        model="claudexor@vps::codex=gpt-6-remote", provider="claudexor", drive_root=root, task_id="child",
        root_task_id="root", parent_task_id="root", reservation_usd=1.25, global_limit_usd=100.0))
    usage.mark_dispatched(reservation)
    ref = _remote_ref()
    persist_call(root, task_id="child", call_id=f"{reservation.attempt_id}_model_request",
                 call_type="llm_claudexor_request", payload={}, keep_raw=True,
                 manifest={"invocation_id": reservation.attempt_id, "operation_id": "op-remote", "engine_ref": ref})
    raw = json.dumps({"outcome": "completed", "usage": {}, "cost": {"knowledge": "exact", "cashUsd": 0.4}})
    opened, asked, busy = [], [], [True]

    class Remote:
        store_identity = STORE

        def get_model_operation(self, operation_id, **_kw):
            asked.append(operation_id)
            if busy[0]:
                refused = ClaudexorUnavailable("engine_busy", "busy", status_code=503)
                refused.retry_after = "120"
                raise refused
            return {"state": "succeeded", "dispatch": {"state": "response_received"},
                    "response": {"state": "ready", "ref": {"sha256": "sha256:" + hashlib.sha256(raw.encode()).hexdigest(),
                                                           "sizeBytes": len(raw.encode())}}}

        def get_model_result(self, *_a, **_k):
            return raw.encode()

        def acknowledge_model_result(self, *_a, **_k):
            return {}

        def create_model_operation(self, *_a, **_k):
            pytest.fail("recovery never creates an operation")

        def close(self):
            pass

    monkeypatch.setattr(remote_engines, "bound_gateway", lambda bound, **_k: opened.append(dict(bound)) or Remote())
    tcr.reconcile_abandoned_usage(root)
    row = next(row for row in ledger_rows(root) if row["attempt_id"] == reservation.attempt_id)
    assert asked == ["op-remote"] and row["recovery"]["basis"]["header"] == "Retry-After" and row["recovery"]["due_at"]
    busy[0] = False
    tcr.reconcile_abandoned_usage(root)  # before the received Retry-After: nobody is asked
    waiting = next(row for row in ledger_rows(root) if row["attempt_id"] == reservation.attempt_id)
    assert asked == ["op-remote"] and waiting["state"] != "settled" and len(opened) == 1

    class Later(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz or timezone.utc) + timedelta(seconds=200)

    monkeypatch.setattr(tcr, "datetime", Later)
    tcr.reconcile_abandoned_usage(root)
    settled = next(row for row in ledger_rows(root) if row["attempt_id"] == reservation.attempt_id)
    assert asked == ["op-remote", "op-remote"] and all(bound == ref for bound in opened) and len(opened) == 2
    assert settled["state"] == "settled" and settled["cost_usd"] == pytest.approx(0.4)  # the engine's own price


def test_after_panic_each_open_remote_model_operation_gets_one_best_effort_cancel_and_its_price_stays_owed(
        host, tmp_path):
    """No new ledger: the open attempt row and its request manifest (engine and operation id) are the authority. The
    boot after a Panic sends each operation one cancel on its own engine and settles nothing; a create whose answer
    was lost names no operation, so no route can address it and it stays open for recovery."""
    from ouroboros import usage_accounting as usage
    from ouroboros.observability import persist_call
    from tests._usage_store_testing import ledger_rows

    root = tmp_path / "usage"
    root.mkdir()
    ref = _remote_ref()
    attempts = []
    for operation in ("op-remote", ""):
        reservation = usage.reserve_attempt(usage.AttemptRequest(
            model="claudexor@vps::codex=gpt-6-remote", provider="claudexor", drive_root=root, task_id="t-op",
            root_task_id="t-op", parent_task_id="", reservation_usd=1.0, global_limit_usd=100.0))
        usage.mark_dispatched(reservation)
        persist_call(root, task_id="t-op", call_id=f"{reservation.attempt_id}_model_request",
                     call_type="llm_claudexor_request", payload={}, keep_raw=True,
                     manifest={"invocation_id": reservation.attempt_id, "engine_ref": ref,
                               **({"operation_id": operation} if operation else {})})
        attempts.append(reservation.attempt_id)
    remote_engines.retire("vps")  # after a Panic no forward is open: the boot attaches for this one cancel
    assert remote_stops.cancel_open_model_operations(root) == {"op-remote": "requested"}
    assert [call["path"] for call in host.state["calls"] if call["path"].startswith("/v2/model-operations/")] == [
        "/v2/model-operations/op-remote/control"]
    rows = {row["attempt_id"]: row for row in ledger_rows(root)}
    assert all(rows[attempt]["state"] == "dispatched" for attempt in attempts)  # price and outcome stay recovery's


# --- what returns: fact files, products and evidence (``remote_engines``, ``remote_workspace``) --------------------

def test_a_remote_runs_facts_are_read_through_its_artifact_routes_and_an_unread_file_is_never_absent(host, tmp_path):
    """Observed model, containment and session events come from the engine's own artifact routes, each read whole
    now into a snapshot of this read (a size never stands for content), the event log only once the run is terminal;
    a text fact the route refuses for its size (413) is read whole from the run's own folder over SSH. A read that is
    not complete names what it did not read and offers no run folder, never an earlier read's files as current."""
    from ouroboros.gateways.claudexor import attempt_containment, final_attempt_facts

    run = "run-facts"
    telemetry = (f"run_id: {run}\nfinal_attempt_id: a01\nattempts:\n  - attempt_id: a01\n    harness_id: codex\n"
                 "    observed_model: gpt-6-remote-2026\n    profile_id: remote-main\n").encode()
    attempt = (b"attempt_id: a01\nharness_home_isolated: true\nharness_home_dir: /home/ouroboros/.h/a01\n"
               b"confinement_mechanism: landlock\nconfinement_verified_denied_path: /etc/shadow\n")
    events = b'{"type": "tool_call"}\n' * (5 * 1024 * 1024 // 22 + 1)  # more than the engine's 4 MiB text cap
    host_run = host.engine_home / "runs" / run  # the run's own folder on the host
    (host_run / "attempts" / "a01").mkdir(parents=True)
    (host_run / "attempts" / "a01" / "events.jsonl").write_bytes(events)
    host.state["run_details"] = {run: {"runDir": str(host_run)}}
    host.state["artifacts"] = {(run, "final/telemetry.yaml"): telemetry, (run, "attempts/a01/attempt.yaml"): attempt,
                               (run, "attempts/a01/events.jsonl"): {"status": 413, "code": "artifact_too_large"},
                               (run, "final/secret.env"): b"x"}
    host.state["run_states"] = {run: "running"}
    client = remote_engines.gateway("vps")
    try:
        live = client.get_run(run)["summary"]
        assert live["remoteRunDir"] == str(host_run) and live["runDir"] != live["remoteRunDir"]
        assert sorted(live["remoteArtifacts"]["files"]) == ["attempts/a01/attempt.yaml", "final/telemetry.yaml"]
        assert final_attempt_facts({"summary": live}, run)["model"] == "gpt-6-remote-2026"
        host.state["run_states"] = {run: "succeeded"}
        detail = client.get_run(run)
        folder = pathlib.Path(detail["summary"]["runDir"])
        assert folder != pathlib.Path(live["runDir"]) and folder.parent.name == "final"
        [applied] = attempt_containment(str(folder))
        assert applied.home_isolated is True and applied.boundary_mechanism == "landlock"
        assert (folder / "attempts/a01/events.jsonl").read_bytes() == events  # whole, over SSH past the cap
        assert not (folder / "final/secret.env").exists()  # only the fact files the readers take
        host.state["artifacts"][(run, "final/telemetry.yaml")] = {"status": 409, "code": "credential_shaped"}
        refused = client.get_run(run)["summary"]
        assert refused["remoteArtifacts"]["state"] == "partial" and "runDir" not in refused
        assert refused["remoteArtifacts"]["unread"] == {"final/telemetry.yaml": "credential_shaped"}
        assert final_attempt_facts({"summary": refused}, run) == {}  # unread: no fact, and the reason is named
        host.state["artifacts_list_error"] = True
        unlisted = client.get_run(run)["summary"]
        assert "runDir" not in unlisted and unlisted["remoteArtifacts"] == {"state": "unread", "code": "not_found"}
        assert (folder / "final/telemetry.yaml").read_bytes() == telemetry  # a settled read's snapshot stays as read
    finally:
        client.close()


def test_a_fact_snapshot_belongs_to_one_engine_address_and_one_read(host, tmp_path):
    """Two engines behind one connection name (a re-pointed address or home, a reinstalled store) never share a
    run's fact snapshots; two reads of the same content share one, two of different content never do."""
    one = {"connection": "vps", "ssh_host": "ouroboros@vds", "engine_home": "/e", "store": "s1"}
    for other in ({**one, "ssh_host": "root@vds"}, {**one, "engine_home": "/f"}, {**one, "store": "s2"}):
        assert remote_engines.fact_mirror_root(one, "run-1") != remote_engines.fact_mirror_root(other, "run-1")
    run, path = "run-same", "final/telemetry.yaml"
    host.state["artifacts"] = {(run, path): b"model: before\n"}
    host.state["run_states"] = {run: "running"}
    client = remote_engines.gateway("vps")
    try:
        first, again = client.get_run(run)["summary"]["runDir"], client.get_run(run)["summary"]["runDir"]
        host.state["artifacts"][(run, path)] = b"model: after!\n"  # the same size, other content
        changed = client.get_run(run)["summary"]["runDir"]
    finally:
        client.close()
    assert first == again != changed and pathlib.Path(changed, path).read_bytes() == b"model: after!\n"
    assert pathlib.Path(first, path).read_bytes() == b"model: before\n"  # an earlier read's snapshot is never rewritten


def test_a_staged_result_returns_binary_directory_mode_link_and_evidence_whole(host, tmp_path):
    """Binary bytes, a new folder tree, a deletion, an executable bit and a symlink return through the result patch;
    the run's ``.review-evidence/`` (which Git ignores) returns whole beside it for the ordinary retention."""
    copy = _local_copy(tmp_path)
    (copy / ".gitignore").write_text("build/\n.review-evidence/\n")
    _git(copy, "add", ".gitignore")
    _git(copy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "ignore evidence")
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)}}
    client = remote_engines.gateway("vps")
    try:
        client.start_run(request, idempotency_key="inv-products")
    finally:
        client.close()
    work = host.remote_root / "inv-products" / "work"
    blob = bytes(range(256)) * 64
    (work / "assets" / "deep").mkdir(parents=True)
    (work / "assets" / "deep" / "logo.bin").write_bytes(blob)
    (work / "assets" / "notes.txt").write_text("a new folder\n")
    (work / "sub" / "b.txt").unlink()
    (work / "a.txt").chmod(0o755)
    (work / "link-to-a").symlink_to("a.txt")
    (work / ".review-evidence" / "shots").mkdir(parents=True)
    (work / ".review-evidence" / "report.md").write_text("what I found")
    (work / ".review-evidence" / "shots" / "page.png").write_bytes(b"\x89PNG\r\n\x1a\n" + blob[:100])
    ref = {**_remote_ref(), "remote_root": str(host.remote_root)}
    entry = SimpleNamespace(execution_root=str(copy), invocation_id="inv-products")
    result = remote_workspace.mirror_result(entry, ref, tmp_path / "capture-products")
    assert result["state"] == "applied", result
    assert (copy / "assets" / "deep" / "logo.bin").read_bytes() == blob
    assert (copy / "assets" / "notes.txt").read_text() == "a new folder\n" and not (copy / "sub" / "b.txt").exists()
    assert (copy / "a.txt").stat().st_mode & 0o111 and (copy / "link-to-a").is_symlink()
    assert result["evidence"]["state"] == "returned" and result["evidence"]["files"] == 2
    assert (copy / ".review-evidence" / "report.md").read_text() == "what I found"
    assert (copy / ".review-evidence" / "shots" / "page.png").read_bytes().startswith(b"\x89PNG")
    assert result["baseline"] and len(result["stage_identity"]) == 64  # the host copy's own recorded source


def test_the_result_waits_for_the_runs_evidence_and_never_carries_it_in_the_patch(host, tmp_path, monkeypatch):
    """Evidence the host holds alone is never left behind: while it cannot be brought back nothing is applied and
    the capture stays owed (the host copy keeps it); once it is here the patch applies, and the evidence folder,
    tracked by no ignore file on the host, rides beside the patch, never in it (as a local capture holds it out)."""
    copy = _local_copy(tmp_path)  # its .gitignore does not name .review-evidence/
    client = remote_engines.gateway("vps")
    try:
        client.start_run({"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)},
                          "harnesses": ["codex"], "primaryHarness": "codex"}, idempotency_key="inv-evidence")
    finally:
        client.close()
    work = host.remote_root / "inv-evidence" / "work"
    (work / "a.txt").write_text("alpha changed remotely\n")
    (work / ".review-evidence").mkdir()
    (work / ".review-evidence" / "log.txt").write_text("only on the host")
    ref = {**_remote_ref(), "remote_root": str(host.remote_root)}
    entry = SimpleNamespace(execution_root=str(copy), invocation_id="inv-evidence")
    real = remote_workspace._return_evidence
    monkeypatch.setattr(remote_workspace, "_return_evidence",
                        lambda *args: {"state": "unreachable", "code": "remote_ssh_timeout"})
    held = remote_workspace.mirror_result(entry, ref, tmp_path / "cap")
    assert held["state"] == "evidence_unread" and (copy / "a.txt").read_text() == "alpha\n"
    from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive  # noqa: F401 - the consumer
    monkeypatch.setattr(remote_workspace, "_return_evidence", real)
    import tarfile

    uploads = len([call for call in host.ssh_calls() if "tar -cf" in " ".join(call["command"])])
    with monkeypatch.context() as old_python:  # before 3.10.12 tarfile has no safe filter: nothing is extracted
        old_python.delattr(tarfile, "data_filter")
        unsafe = remote_workspace.mirror_result(entry, ref, tmp_path / "cap")
    assert unsafe["state"] == "evidence_unread" and unsafe["evidence"] == {
        "state": "unsupported", "code": "tar_data_filter_unavailable"}
    assert (copy / "a.txt").read_text() == "alpha\n" and not (copy / ".review-evidence").exists()
    assert len([call for call in host.ssh_calls() if "tar -cf" in " ".join(call["command"])]) == uploads
    done = remote_workspace.mirror_result(entry, ref, tmp_path / "cap")
    assert done["state"] == "applied" and (copy / "a.txt").read_text() == "alpha changed remotely\n"
    assert (copy / ".review-evidence" / "log.txt").read_text() == "only on the host"
    assert b".review-evidence" not in pathlib.Path(done["patch"]).read_bytes()
    assert remote_workspace.mirror_result(entry, ref, tmp_path / "cap")["state"] == "already_applied"  # idempotent
    plain = tmp_path / "plain"
    plain.mkdir()
    refused = remote_workspace.mirror_result(SimpleNamespace(execution_root=str(plain), invocation_id="x"), ref,
                                             tmp_path / "cap2")
    assert refused["state"] == "local_copy_not_git"  # an honest limitation, never "the copy is gone"


def test_returned_evidence_stays_in_its_folder_across_returns_and_a_link_is_never_made(host, tmp_path):
    """A remote run writes its own copy's ``.review-evidence/``; what returns is unpacked into a private folder and merged
    into the local evidence folder through real folders and regular files only. A link the run left there is named and
    never made, so a later return to the same copy (a continuation's) cannot pass through it into the local repository,
    and evidence that returned earlier stays. A local entry on the way that is not a real folder or a regular file
    refuses the return whole before anything local changes: nothing is applied and the capture stays owed. (Every
    file here is inert: no Git setting, hook or command.)"""
    def absent(path):
        return not path.is_symlink() and not path.exists()

    copy = _local_copy(tmp_path)
    (copy / ".gitignore").write_text("build/\n.review-evidence/\n")
    _git(copy, "add", ".gitignore")
    _git(copy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "ignore evidence")
    client = remote_engines.gateway("vps")
    try:
        client.start_run({"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)}},
                         idempotency_key="inv-links")
    finally:
        client.close()
    git_config = (copy / ".git" / "config").read_bytes()
    work = host.remote_root / "inv-links" / "work"
    evidence = work / ".review-evidence"
    evidence.mkdir()
    (evidence / "report.md").write_text("what I found")
    (evidence / "a").symlink_to("../.git")
    (evidence / "latest").symlink_to("report.md")
    ref = {**_remote_ref(), "remote_root": str(host.remote_root)}
    entry = SimpleNamespace(execution_root=str(copy), invocation_id="inv-links")
    first = remote_workspace.mirror_result(entry, ref, tmp_path / "capture-1")
    assert first["state"] == "no_changes" and first["evidence"]["state"] == "returned", first
    assert absent(copy / ".review-evidence" / "a") and absent(copy / ".review-evidence" / "latest"), "a link was made"
    assert sorted(gap["path"] for gap in first["evidence"]["gaps"]) == [".review-evidence/a", ".review-evidence/latest"]
    (evidence / "a").unlink()  # the continuation's run: a folder where the link was
    (evidence / "a").mkdir()
    (evidence / "a" / "config").write_bytes(b"[core]\n\trepositoryformatversion = 0\n# remote-evidence-marker\n")
    (evidence / "report.md").unlink()  # it returns only its own evidence: what came back earlier stays here
    second = remote_workspace.mirror_result(entry, ref, tmp_path / "capture-2")
    assert second["evidence"]["state"] == "returned" and second["evidence"]["files"] == 1, second
    assert (copy / ".git" / "config").read_bytes() == git_config  # nothing reached the repository
    assert (copy / ".review-evidence" / "a" / "config").read_bytes().endswith(b"# remote-evidence-marker\n")
    assert (copy / ".review-evidence" / "report.md").read_text() == "what I found"
    (copy / ".review-evidence" / "shots").symlink_to("../.git")  # a local link on the way (an older return's)
    (evidence / "shots").mkdir()
    (evidence / "shots" / "page.png").write_bytes(b"\x89PNG\r\n")
    (work / "a.txt").write_text("alpha changed remotely\n")
    third = remote_workspace.mirror_result(entry, ref, tmp_path / "capture-3")
    assert third["state"] == "evidence_unread" and third["evidence"]["state"] == "refused", third
    assert third["evidence"]["code"] == "evidence_local_conflict"
    assert absent(copy / ".git" / "page.png") and (copy / "a.txt").read_text() == "alpha\n"  # nothing moved or applied
    assert (copy / ".git" / "config").read_bytes() == git_config
    assert sorted(path.name for path in (tmp_path / "capture-3").iterdir()) == []  # its private unpacking is gone


def test_an_evidence_stream_with_a_member_outside_its_folder_is_refused_whole(tmp_path, monkeypatch):
    """A stream that is not what the host's ``tar -cf - .review-evidence`` writes (a member outside that folder, by a
    ``..`` part, an absolute name or another folder) is refused whole before anything is unpacked: the local evidence
    stays as it was and the capture stays owed."""
    import io
    import tarfile

    if not hasattr(tarfile, "data_filter"):
        pytest.skip("this interpreter has no safe tar filter, so no evidence is unpacked at all (``unsupported``)")
    local = tmp_path / "copy"
    (local / ".review-evidence").mkdir(parents=True)
    (local / ".review-evidence" / "kept.md").write_text("earlier evidence")
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in (".review-evidence/../escaped.txt", "/absolute.txt", "elsewhere/x.txt"):
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w") as bundle:
            for member in (".review-evidence/ok.md", name):
                info = tarfile.TarInfo(member)
                info.size = 5
                bundle.addfile(info, io.BytesIO(b"inert"))
        monkeypatch.setattr(remote_engines, "run_ssh", lambda *_a, sink=None, **_k: sink.write(stream.getvalue()))
        result = remote_workspace._return_evidence({"ssh_host": "h", "connection": "vps"}, "/remote", local, capture)
        assert result["state"] == "refused" and result["code"] == "evidence_member_outside", (name, result)
    assert sorted(path.name for path in (local / ".review-evidence").iterdir()) == ["kept.md"]
    assert not (tmp_path / "escaped.txt").exists() and not list(capture.iterdir())


def test_a_continuation_in_the_same_copy_brings_its_remote_predecessors_work_home_first(host, tmp_path, monkeypatch):
    """The ordinary Continue of a staged remote run, through its real claim (``claimed_start_request``): while the
    predecessor's result cannot be brought back the start is refused before any row or POST, nothing is superseded
    and that work stays owed on the host (a start that skipped the claim is refused at staging, unsent); once it can,
    the claim captures the predecessor first (never again once captured), the successor stages the copy WITH that
    work, and after the successor took the copy over both host folders are retired."""
    from ouroboros import delegate_continuation, owner_keep
    from ouroboros.delegate_capture_attempt import capture_required
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs
    from ouroboros.delegate_start_claims import claimed_start_request
    from ouroboros.subagent_worktrees import provision_execution_snapshot

    data = tmp_path / "custody"
    data.mkdir()
    target = tmp_path / "target"
    _git(tmp_path, "init", "-q", str(target))
    (target / "a.txt").write_text("alpha\n")
    _git(target, "add", "-A")
    _git(target, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base")
    snapshot = provision_execution_snapshot(target_root=target, task_id="t-c", snapshot_id="snap-c",
                                            worktree_root=tmp_path / "copies")
    copy = pathlib.Path(snapshot.path)
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)},
               "harnesses": ["codex"], "primaryHarness": "codex"}
    bound = {"route_id": "codex", "model": "m", "access": "full", "mode": "agent", "ledger_root": str(data),
             "daemon_store_id": STORE, "snapshot_id": "snap-c", "execution_root": str(copy),
             "baseline_sha": snapshot.baseline_sha, "target_root": str(target)}
    client = remote_engines.gateway("vps")
    try:
        client.custody_root = data
        ref = dict(client.engine_ref)
        first = client.start_run(request, idempotency_key="inv-1")["runId"]
    finally:
        client.close()
    assert custody.record_started(data, custody.RunCustody(run_id=first, task_id="t-c", invocation_id="inv-1",
                                                           engine_ref=dict(ref), **bound))
    (host.remote_root / "inv-1" / "work" / "a.txt").write_text("alpha, fixed by the first run\n")
    (host.remote_root / "inv-1" / "work" / ".review-evidence").mkdir()
    (host.remote_root / "inv-1" / "work" / ".review-evidence" / "notes.md").write_text("what the first run checked")
    assert custody.emit(data, custody.SETTLED, {"run_id": first, "task_id": "t-c", "state": "interrupted"})

    def claim():
        return claimed_start_request(
            data, run_id="", task_id="t-c", invocation_id="inv-2", idempotency_key="inv-2", route="codex",
            request={**request, "continueFrom": first}, continuation_of=first, capture_id="inv-2", project_id="",
            snapshot_id="snap-c", snapshot_task_id="t-c", execution_root=str(copy), engine_ref=ref)

    real, posts = remote_workspace._return_evidence, len(_posts(host))
    monkeypatch.setattr(remote_workspace, "_return_evidence", lambda *_a: {"state": "unreachable",
                                                                          "code": "remote_ssh_timeout"})
    started, refusal = claim()  # the negative control: a definite no-run, nothing recorded or sent
    assert not started and refusal["reason"] == "continuation_remote_result_unread", refusal
    assert refusal["remote_result"] == {"state": "evidence_unread"} and "Nothing was started" in refusal["detail"]
    assert not [row for row in custody.pending_invocations(data) if row["invocation_id"] == "inv-2"]
    prior = current.run(data, first)
    assert capture_required(prior) and not prior.patch_captured and not prior.superseded_by
    assert (copy / "a.txt").read_text() == "alpha\n" and len(_posts(host)) == posts
    client = remote_engines.gateway("vps")
    try:
        client.custody_root = data
        with pytest.raises(ClaudexorUnavailable) as skipped:  # the staging seam refuses a start that skipped the claim
            client.start_run({**request, "continueFrom": first}, idempotency_key="inv-skip")
    finally:
        client.close()
    assert skipped.value.code == "continuation_remote_result_unread" and skipped.value.unsent
    assert not (host.remote_root / "inv-skip").exists() and len(_posts(host)) == posts
    monkeypatch.setattr(remote_workspace, "_return_evidence", real)
    started, refusal = claim()
    assert started and refusal == {}, refusal
    assert current.run(data, first).patch_captured and (copy / "a.txt").read_text() == "alpha, fixed by the first run\n"
    assert (copy / ".review-evidence" / "notes.md").read_text() == "what the first run checked"
    asked = len(host.ssh_calls())
    assert delegate_continuation._remote_result_home(data, first) == {} and len(host.ssh_calls()) == asked
    client = remote_engines.gateway("vps")
    try:
        client.custody_root = data
        second = client.start_run({**request, "continueFrom": first}, idempotency_key="inv-2")["runId"]
    finally:
        client.close()
    assert (host.remote_root / "inv-2" / "work" / "a.txt").read_text() == "alpha, fixed by the first run\n"
    # a resumed native session may keep its first request's instructions (#2513): the prompt also names THIS copy
    bodies = {call["key"]: call["body"] for call in _posts(host)}
    assert bodies["inv-1"]["prompt"] == request["prompt"]  # a fresh start's words stay as they are
    resumed, work2 = bodies["inv-2"], host.remote_root / "inv-2" / "work"
    tail = resumed["prompt"][len(request["prompt"]):]
    assert resumed["prompt"].startswith(request["prompt"])
    assert tail == resumed["instructions"][len(request["instructions"]):]
    assert f"Its copy of the work is {work2};" in tail and f"{copy} -> {work2}" in tail and "inv-1" not in tail
    assert custody.record_started(data, custody.RunCustody(
        run_id=second, task_id="t-c", invocation_id="inv-2", engine_ref=dict(ref), continuation_of=first,
        capture_id="inv-2", snapshot_task_id="t-c", **bound))
    assert current.run(data, first).superseded_by == second
    assert custody.emit(data, custody.SETTLED, {"run_id": second, "task_id": "t-c", "state": "succeeded"})
    custody.record_patch_disposed(data, current.run(data, second), disposition="rejected")
    monkeypatch.setattr(owner_keep, "is_kept", lambda *_a, **_k: False)
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: False)
    outcomes = reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    for run_id, key in ((first, "inv-1"), (second, "inv-2")):
        assert {"run_id": run_id, "task_id": "t-c", "action": "remote_copy_retired"} in outcomes, outcomes
        assert not (host.remote_root / key).exists() and current.run(data, run_id).remote_copy_retired
    calls = len(host.state["calls"])
    reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
    assert len(host.state["calls"]) == calls  # nothing left to retire: no engine is opened


def test_a_superseded_run_whose_result_never_came_home_keeps_its_host_copy_and_opens_no_engine(
        host, tmp_path, monkeypatch):
    """A predecessor superseded without its result ever captured here (only a start that bypassed the hand-over, or a
    row from before it, leaves one): its host folder holds the only copy of that work, so the sweep keeps it
    (``result_not_returned``) and opens no engine for it, while its disposed successor's folder is retired."""
    from ouroboros import delegate_continuation, owner_keep
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"
    data.mkdir()
    ref = {**_remote_ref(), "remote_root": str(host.remote_root)}
    for key in ("inv-a", "inv-b"):
        (host.remote_root / key / "work").mkdir(parents=True)
    (host.remote_root / "inv-a" / "work" / "only-on-host.txt").write_text("never brought home\n")

    def row(run_id, invocation, **extra):
        return custody.RunCustody(run_id=run_id, task_id="t-chain", invocation_id=invocation, route_id="codex",
                                  model="m", access="full", mode="agent", ledger_root=str(data), engine_ref=dict(ref),
                                  daemon_store_id=STORE, snapshot_id="snap-1", execution_root=str(tmp_path), **extra)

    assert custody.record_started(data, row("run-a", "inv-a"))
    assert custody.emit(data, custody.SETTLED, {"run_id": "run-a", "task_id": "t-chain"})
    assert custody.record_started(data, row("run-b", "inv-b", continuation_of="run-a", capture_id="inv-b"))
    assert custody.emit(data, custody.SETTLED, {"run_id": "run-b", "task_id": "t-chain"})
    custody.record_patch_disposed(data, current.run(data, "run-b"), disposition="applied")
    monkeypatch.setattr(owner_keep, "is_kept", lambda *_a, **_k: False)
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: False)
    assert remote_workspace.copy_hold(data, current.run(data, "run-a")) == "result_not_returned"

    def sweep():
        calls = len(host.state["calls"])
        outcomes = reconcile_orphaned_runs(data, running_task_ids=set(), gateway_factory=_OWNED_GATEWAY)
        return [row for row in outcomes if row.get("run_id") in ("run-a", "run-b")], len(host.state["calls"]) - calls

    retired, _calls = sweep()
    assert retired == [{"run_id": "run-b", "task_id": "t-chain", "action": "remote_copy_retired"}], retired
    assert (host.remote_root / "inv-a" / "work" / "only-on-host.txt").exists()
    assert not (host.remote_root / "inv-b").exists()
    assert sweep() == ([], 0) and not current.run(data, "run-a").remote_copy_retired


def test_a_remote_workspace_run_retains_its_work_product_and_evidence_never_local_no_changes(host, tmp_path):
    workspace = host.workspace / "app"
    (workspace / ".review-evidence").mkdir(parents=True)
    (workspace / ".review-evidence" / "findings.md").write_text("checked on the host")
    product = b"diff --git a/x b/x\n" + bytes(range(256))
    host.state["artifacts"] = {("run-ws", "final/patch.diff"): product}
    ref = {**_remote_ref(), "workspace": str(workspace)}
    entry = SimpleNamespace(engine_ref=ref, run_id="run-ws", task_id="t-ws", invocation_id="inv-ws-products",
                            capture_id="", snapshot_id="")
    block = remote_workspace.remote_workspace_block(entry, drive=tmp_path / "drive")
    assert block["status"] == "remote_workspace" and block["remote_run"]["workProduct"]["path"] == "final/patch.diff"
    kept = block["retained"]
    assert pathlib.Path(kept["work_product"]["local"]).read_bytes() == product
    assert kept["work_product"]["size"] == len(product) and kept["evidence"]["state"] == "returned"
    assert (pathlib.Path(kept["folder"]) / ".review-evidence" / "findings.md").read_text() == "checked on the host"
    host.state["artifacts"] = {}  # the engine no longer serves it: named unread, never "no changes"
    again = remote_workspace.remote_workspace_block(entry, drive=tmp_path / "drive2")
    assert again["retained"]["work_product"]["unread"] == "artifact_not_found"
    assert again["status"] == "failed" and again["retry"] == "transient" and "disposition" not in again


def test_a_host_copy_is_retired_only_when_its_runs_retention_ends(host, tmp_path, monkeypatch):
    """Keep, a live Continue, an undisposed result and an unsettled run each keep the host folder; once none holds it
    the sweep removes exactly that folder (a lost removal is asked again on the next pass), its local fact mirror
    goes with it, and the row stops holding it."""
    from ouroboros import delegate_continuation, owner_keep
    from ouroboros.delegate_custody_reconcile import reconcile_orphaned_runs

    data = tmp_path / "custody"
    data.mkdir()
    ref = {**_remote_ref(), "remote_root": str(host.remote_root)}
    folder = host.remote_root / "inv-keep"
    (folder / "work").mkdir(parents=True)
    (folder / "inputs").mkdir()
    entry = custody.RunCustody(run_id="run-keep", task_id="t-keep", invocation_id="inv-keep", route_id="codex",
                               model="m", access="full", mode="agent", ledger_root=str(data), engine_ref=dict(ref),
                               daemon_store_id=STORE, snapshot_id="snap-keep")
    assert custody.record_started(data, entry)
    mirror = remote_engines.fact_mirror_root(ref, "run-keep")
    (mirror / "live" / "snap").mkdir(parents=True, exist_ok=True)
    (mirror / "final" / "snap").mkdir(parents=True, exist_ok=True)
    kept, continuable = [False], [False]
    monkeypatch.setattr(owner_keep, "is_kept", lambda _root, task: kept[0] and task == "t-keep")
    monkeypatch.setattr(delegate_continuation, "still_continuable", lambda *_a, **_k: continuable[0])

    def sweep():
        outcomes = reconcile_orphaned_runs(data, running_task_ids={"t-keep"}, gateway_factory=_OWNED_GATEWAY)
        return [row for row in outcomes if row.get("run_id") == "run-keep"]

    sweep()
    assert folder.is_dir()  # live: never removed
    host.state["run_states"] = {"run-keep": "completed"}
    assert custody.emit(data, custody.SETTLED, {"run_id": "run-keep", "task_id": "t-keep"})
    sweep()
    assert folder.is_dir() and remote_workspace.copy_hold(data, current.run(data, "run-keep")) == "result_not_disposed"
    custody.record_patch_disposed(data, current.run(data, "run-keep"), disposition="applied")
    kept[0] = True
    assert remote_workspace.copy_hold(data, current.run(data, "run-keep")) == "owner_keep"
    kept[0], continuable[0] = False, True
    assert remote_workspace.copy_hold(data, current.run(data, "run-keep")) == "continuable"
    sweep()
    assert folder.is_dir()
    continuable[0] = False
    remote_engines.retire("vps")
    host.descriptor.rename(host.descriptor.with_suffix(".gone"))  # the engine is unreachable: a lost cleanup
    sweep()
    assert folder.is_dir() and not current.run(data, "run-keep").remote_copy_retired
    host.descriptor.with_suffix(".gone").rename(host.descriptor)
    # decided at the effect: a Keep saved after the sweep's own read still holds it (inside the Keep gate), and so
    # does a continuation claimed under the run's disposition lock
    with monkeypatch.context() as effect:
        effect.setattr(remote_workspace, "copy_hold", lambda *_a, **_k: "")
        kept[0] = True
        assert {"run_id": "run-keep", "task_id": "t-keep", "action": "remote_copy_kept",
                "reason": "owner_keep"} in sweep()
        kept[0] = False
        assert custody.emit(data, custody.START_REQUESTED, {"invocation_id": "inv-next", "task_id": "t-keep",
                                                            "continuation_of": "run-keep",
                                                            "request": {"continueFrom": "x"}})
        assert {"run_id": "run-keep", "task_id": "t-keep", "action": "remote_copy_kept",
                "reason": "continuation_pending"} in sweep()
        assert custody.emit(data, custody.START_FAILED, {"invocation_id": "inv-next", "task_id": "t-keep",
                                                         "definite": True, "reason": "test"})
        # an unconfirmed removal is not asked again on the very next pass
        effect.setattr(remote_workspace, "retire_staged", lambda *_a: False)
        assert {"run_id": "run-keep", "task_id": "t-keep", "action": "remote_copy_kept",
                "reason": "removal_unconfirmed"} in sweep()
        calls = len(host.state["calls"])  # paused: no outcome, and its engine is not even opened
        assert remote_workspace.copy_due(data, current.run(data, "run-keep")) == "retry_not_due"
        assert sweep() == [] and len(host.state["calls"]) == calls
    monkeypatch.setattr(remote_workspace, "_RETIRE_DUE", {})
    retired = sweep()
    assert {"run_id": "run-keep", "task_id": "t-keep", "action": "remote_copy_retired"} in retired
    assert not folder.exists() and (host.remote_root).is_dir()  # exactly that folder
    assert current.run(data, "run-keep").remote_copy_retired and "run-keep" not in current.held_runs(data)
    assert not (mirror / "live").exists() and (mirror / "final" / "snap").is_dir()  # settled facts are never erased


def test_imported_remote_documents_are_released_only_through_their_recorded_engine(host, tmp_path):
    from ouroboros.delegate_documents import DOCUMENT_IMPORT_REQUESTED, _fold_owed
    from ouroboros.execution_domain import EngineClients

    ref = _remote_ref()
    owed = {}
    _fold_owed(owed, "doc.pdf", "key-1", DOCUMENT_IMPORT_REQUESTED, {
        "type": DOCUMENT_IMPORT_REQUESTED, "origin": "imported", "store_id": STORE, "engine_ref": ref,
        "request": {"lifetime": "managed"}})
    assert owed["key-1"]["engine_ref"] == ref  # the release obligation keeps the engine it was imported on
    clients = EngineClients(owned=lambda: pytest.fail("a remote import is never asked of the owned engine"))
    try:
        reached = clients.for_store(STORE, engine_ref=ref)
        assert reached.store_identity == STORE and reached.engine_ref["connection"] == "vps"
        alias = clients.for_store(STORE, engine_ref={**ref, "store": "store-elsewhere"})
        with pytest.raises(ClaudexorUnavailable):  # another store under the same name answers for nothing here
            alias.operations()
    finally:
        clients.drop()


@pytest.mark.parametrize("change", ["content", "mode", "link"])
def test_bytes_that_change_on_the_way_under_the_same_name_are_refused_before_the_post(host, tmp_path, monkeypatch,
                                                                                      change):
    """The negative control of the baseline check: the host receives the same paths, but one file's content, mode
    or link target is not what was hashed here. The host commits nothing and nothing is POSTed; the same key, sent
    the real bytes, then proves its baseline (paths, modes and blobs) and starts."""
    import shutil

    copy = _local_copy(tmp_path)
    (copy / "link").symlink_to("a.txt")
    _git(copy, "add", "link")
    _git(copy, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "link")
    altered = tmp_path / "altered"
    shutil.copytree(copy, altered, symlinks=True)
    if change == "content":
        (altered / "a.txt").write_text("alphA\n")  # same name and size, other bytes
    elif change == "mode":
        (altered / "a.txt").chmod(0o755)
    else:
        (altered / "link").unlink()
        (altered / "link").symlink_to("c.txt")
    real = remote_workspace._tar_feed
    monkeypatch.setattr(remote_workspace, "_tar_feed",
                        lambda root, files, extra=(), members=(): real(altered, files, extra, members))
    key = f"inv-altered-{change}"
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)}}
    client = remote_engines.gateway("vps")
    try:
        with pytest.raises(ClaudexorUnavailable) as refused:
            client.start_run(request, idempotency_key=key)
        assert refused.value.code == "remote_stage_mismatch"
        assert not [call for call in host.state["calls"] if call["path"] == "/v2/runs" and call.get("key") == key]
        assert not (host.remote_root / key / remote_workspace.STAGED_MARKER).exists()
        monkeypatch.setattr(remote_workspace, "_tar_feed", real)
        client.start_run(request, idempotency_key=key)
    finally:
        client.close()
    work = host.remote_root / key / "work"
    expected = remote_workspace._expected_index(copy, remote_workspace._selected_files(copy, git=True)[0], git=True)
    assert subprocess.run(["git", "-C", str(work), "ls-files", "-s", "-z"], capture_output=True).stdout == expected
    assert b"120000 " in expected and (work / "link").is_symlink()  # the link travelled as a link, mode and target


@pytest.mark.parametrize("mode", ["local-copy", "remote-workspace"])
def test_declared_inputs_and_references_travel_whole_and_owner_words_stay_byte_for_byte(host, tmp_path, mode):
    """Both modes carry the start's typed host sources: a binary input file and an input folder from outside the
    checkout land under ``inputs/<n>/`` with identical bytes, a governance reference under ``refs/<n>/``, each named
    by the appended HOST PATH MAP; the prompt and instructions (the owner's words, quotes, other scripts and every path
    in them) reach the engine byte for byte."""
    copy = _local_copy(tmp_path)
    outside = tmp_path / "retained" / "inputs"
    (outside / "bundle" / "nested").mkdir(parents=True)
    binary = bytes(range(256)) * 8
    (outside / "scan.bin").write_bytes(binary)
    (outside / "bundle" / "nested" / "data.csv").write_text("a,b\n1,2\n")
    design = tmp_path / "body" / "DESIGN.md"
    design.parent.mkdir()
    design.write_text("# design")
    owner = 'Владелец: «посмотри /Users/owner/Desktop/plan.md» — and keep "quotes" & $HOME as typed.'
    request = {"prompt": f"{owner} Use {outside / 'scan.bin'}.",
               "instructions": f"Inputs: {outside / 'bundle'}. Governance: {design}.",
               "scope": {"kind": "project", "root": str(copy)},
               "_host_inputs": [{"path": str(outside / "scan.bin"), "kind": "input"},
                                {"path": str(outside / "bundle"), "kind": "input"},
                                {"path": str(design), "kind": "reference"}]}
    key = f"inv-sources-{mode}"
    client = remote_engines.gateway("vps")
    if mode == "remote-workspace":
        client.engine_ref = {**client.engine_ref, "workspace": str(host.workspace)}
    try:
        client.start_run(request, idempotency_key=key)
    finally:
        client.close()
    body = next(call["body"] for call in host.state["calls"] if call["path"] == "/v2/runs" and call["key"] == key)
    folder = host.remote_root / key
    assert (folder / "inputs" / "0" / "scan.bin").read_bytes() == binary
    assert (folder / "inputs" / "1" / "bundle" / "nested" / "data.csv").read_text() == "a,b\n1,2\n"
    assert (folder / "refs" / "2" / "DESIGN.md").read_text() == "# design"
    assert body["prompt"] == request["prompt"] and body["instructions"].startswith(request["instructions"])
    for local, remote in ((outside / "scan.bin", folder / "inputs" / "0" / "scan.bin"),
                          (outside / "bundle", folder / "inputs" / "1" / "bundle"),
                          (design, folder / "refs" / "2" / "DESIGN.md")):
        assert f"{local} -> {remote}" in body["instructions"]
    assert body["scope"]["root"] == (str(host.workspace) if mode == "remote-workspace" else str(folder / "work"))
    assert "_host_inputs" not in body and body.get("taskGroupId") == remote_stops.task_group_id(key)


def test_a_host_source_inside_a_staged_root_travels_once_with_it_on_any_host(tmp_path, monkeypatch):
    """A declared host source inside one of the staged roots already travels in that copy, so it is neither sent nor
    mapped a second time. Containment is judged by the host's own path rules (``pathlib``), never by a ``/`` prefix: a
    sibling that only shares the root's name as a prefix still travels, and on a Windows host (its path rules stood in
    here) a source inside ``C:\\...\\proj`` is inside it."""
    root = tmp_path / "proj"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "spec.md").write_text("inside")
    sibling = tmp_path / "proj-notes.md"
    sibling.write_text("outside")
    request = {"_host_inputs": [{"path": str(root / "docs" / "spec.md")}, {"path": str(root), "kind": "reference"},
                                {"path": str(sibling)}]}
    _entries, path_map = remote_workspace._input_entries(request, "/host/runs/inv", [str(root)])
    assert [local for local, _remote in path_map] == [str(sibling)]
    monkeypatch.setattr(remote_workspace, "pathlib", SimpleNamespace(Path=pathlib.PureWindowsPath))
    windows = {"_host_inputs": [{"path": "C:\\Users\\me\\proj\\docs\\spec.md"}, {"path": "C:\\Users\\me\\proj"}]}
    assert remote_workspace._input_entries(windows, "/host/runs/inv", ["C:\\Users\\me\\proj"]) == ([], [])


def test_inputs_are_proven_on_the_host_and_a_reused_copy_must_be_the_one_selected(host, tmp_path, monkeypatch):
    """The inputs are re-hashed on the host against the blobs, links and entries sent (a byte changed on the way
    commits and moves nothing); a link that stays inside a declared folder travels as a link, one that leaves it, or
    an unreadable entry, refuses before anything is sent; a copy already staged for the invocation is reused only when
    it was made from exactly the files selected now; every refusal before the POST is marked unsent."""
    import os

    copy = _local_copy(tmp_path)
    folder_input = tmp_path / "prepared" / "set"
    (folder_input / "deep").mkdir(parents=True)
    (folder_input / "deep" / "data.txt").write_text("rows")
    os.symlink("deep/data.txt", folder_input / "alias")
    (folder_input / "empty").mkdir()
    request = {"prompt": "p", "instructions": "i", "scope": {"kind": "project", "root": str(copy)},
               "harnesses": ["codex"], "primaryHarness": "codex",
               "_host_inputs": [{"path": str(folder_input), "kind": "input"}]}

    def start(key, body=request):
        client = remote_engines.gateway("vps")
        try:
            return client.start_run(body, idempotency_key=key)
        finally:
            client.close()

    start("inv-proof")
    landed = host.remote_root / "inv-proof" / "inputs" / "0" / "set"
    assert (landed / "alias").is_symlink() and os.readlink(landed / "alias") == "deep/data.txt"
    assert (landed / "empty").is_dir() and (landed / "deep" / "data.txt").read_text() == "rows"
    # a byte changed on the way: the host's own re-hash refuses, nothing is committed or started
    real_feed = remote_workspace._tar_feed

    def altered(root, files, extra=(), members=()):
        members = [(arc, kind, src) for arc, kind, src in members]
        feed = real_feed(root, files, extra, members)

        def corrupt(stream):
            import io as _io

            buffer = _io.BytesIO()
            feed(buffer)
            stream.write(buffer.getvalue().replace(b"rows", b"rowz"))
        return corrupt

    monkeypatch.setattr(remote_workspace, "_tar_feed", altered)
    posts = len(_posts(host))
    with pytest.raises(ClaudexorUnavailable) as mismatch:
        start("inv-proof-altered")
    assert mismatch.value.code == "remote_stage_mismatch" and mismatch.value.unsent and len(_posts(host)) == posts
    assert not (host.remote_root / "inv-proof-altered").exists()  # nothing of the refused staging stays
    monkeypatch.setattr(remote_workspace, "_tar_feed", real_feed)
    # a link that leaves what was declared never travels, and neither does anything else of the start
    outside = tmp_path / "private.key"
    outside.write_text("secret")
    os.symlink(outside, folder_input / "escape")
    with pytest.raises(ClaudexorUnavailable) as escape:
        start("inv-escape")
    assert escape.value.code == "remote_input_link_outside" and escape.value.unsent
    assert not (host.remote_root / "inv-escape").exists()
    (folder_input / "escape").unlink()
    # the same invocation staged again from other files: the copy on the host is not reused
    start("inv-identity")
    (copy / "a.txt").write_text("changed locally\n")
    with pytest.raises(ClaudexorUnavailable) as changed:
        start("inv-identity")
    assert changed.value.code == "remote_stage_changed" and changed.value.unsent
    assert (host.remote_root / "inv-identity" / "work").is_dir()  # a copy an earlier call made is left as it is


ROSTER = {"profiles": [
    {"profile": {"harness_id": "codex", "profile_id": "remote-main", "display_name": "Remote main", "enabled": True},
     "status": {"verification": "passed", "availability": "available"}},
    {"profile": {"harness_id": "codex", "profile_id": "remote-spare", "enabled": False},
     "status": {"verification": "not_run", "availability": "unknown"}}], "accountPools": []}


def test_a_remote_catalog_carries_that_engines_own_roster_and_never_an_empty_one_for_unread(host):
    from ouroboros.gateway.engine_connections import _catalog

    remote_engines.gateway("vps").close()  # the catalog answers through a tunnel already open
    host.state["profiles"] = ROSTER
    catalog = _catalog("vps")
    assert catalog["accounts_state"] == "ok" and catalog["sources"][0]["credentialHarness"] == "codex"
    assert catalog["accounts"] == [
        {"harness_id": "codex", "profile_id": "remote-main", "label": "Remote main", "enabled": True,
         "verification": "passed", "availability": "available"},
        {"harness_id": "codex", "profile_id": "remote-spare", "label": "remote-spare", "enabled": False,
         "verification": "not_run", "availability": "unknown"}]
    host.state["profiles_error"] = True
    unread = _catalog("vps")
    assert unread["accounts_state"] == "unread" and unread["accounts_code"] == "accounts_unavailable"
    assert "accounts" not in unread
    host.state["profiles_error"], host.state["profiles"] = False, {"profiles": [{"profile": {}}], "accountPools": []}
    assert _catalog("vps")["accounts_code"] == "malformed_response"
    assert TOKEN not in json.dumps(catalog)


def test_a_review_panels_owed_start_is_never_re_sent_by_its_rejoin(host, tmp_path):
    """A review seat rejoins its pending start through ``seat_gateway(record)``: an owed one refuses before any POST."""
    data = tmp_path / "custody"
    ref = _remote_ref()
    _request_start(data, "inv-seat-owed", ref, _local_copy(tmp_path), task="t-seat")
    remote_stops.record_owed_stops(data)
    [record] = _pending(data, "inv-seat-owed")
    with pytest.raises(ClaudexorUnavailable) as refused:
        domain.seat_gateway({**record, "ledger_root": str(data)}, str(tmp_path), "vps")
    assert refused.value.code == "remote_stop_owed" and not _posts(host)
