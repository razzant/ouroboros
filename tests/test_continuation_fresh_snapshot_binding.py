"""#2513: a writing continuation in a NEW private snapshot must be told its own root.

A writer that changed nothing closes itself as ``no_changes`` and its copy goes. Continuing
it for implementation provisions a fresh snapshot at a new path, and the host binds that path
in the new request's instructions. The engine's ``auto`` carrier may resume the predecessor's
native session, which keeps the instructions of its FIRST request: the binding that named the
retired copy and told the child to stop on a root mismatch. ``delegate_continuation`` states
that facts which change reach the continuing child in its PROMPT for exactly this reason, so
the prompt must carry the binding of the root this run writes in.

The repair has two halves, checked below against the real request producer. The access line
every run's instructions persist scopes run-specific facts (access, edit/evidence/scratch and
copy-local locations, what a private copy makes read-only) to ONE run and names the current
one by identity, never by position. And EVERY ordinary continuation's prompt restates this
run's access and binding exactly as its own instructions carry them, after provisioning: a
resumed session may hold any earlier leg's instructions, the current ones, or both, and a
packet re-brief quotes earlier prompts after the current one. Request bytes prove the host
contract only; what a resumed vendor model does with them is not observed here.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from ouroboros import delegate_custody as custody
from ouroboros.delegate_start_instructions import (
    _ACCESS_PRECEDENCE,
    access_instruction,
    execution_binding_fingerprint,
    no_copy_binding_instruction,
)
from ouroboros.tools.delegate_integration import capture_terminal_patch_for_drive
from tests._delegated_transport_shared import (  # noqa: F401 -- autouse transport fixture
    _owned_gateway_uses_each_test_transport,
    _select_transport_access,
)
from tests.test_delegate_continuation import _settle, _start

_BOUND = "the private execution root for this run is "


def test_a_continuation_in_a_fresh_snapshot_names_its_new_root_where_a_resumed_session_reads(
        tmp_path, monkeypatch):
    first, _payload, ctx, _calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-orient",
                                          prompt="orient: read the code and report a plan")
    old_root = first["execution"]["workspaceRoot"]
    assert _BOUND + old_root in first["instructions"]
    _settle(tmp_path, "run-orient", ctx.task_id, state="succeeded", reason="completed")
    capture = capture_terminal_patch_for_drive(tmp_path, custody.replay(tmp_path)["run-orient"])
    assert capture["status"] == "ready_no_changes" and capture["disposition"] == custody.NO_CHANGES, capture
    assert not pathlib.Path(old_root).exists()  # the orientation copy is retired with its empty result

    request, payload, _ctx, calls = _start(tmp_path, monkeypatch, acting=True, run_id="run-impl",
                                           prompt="implement the plan",
                                           start_kwargs={"continue_from": "run-orient"})
    assert payload["status"] == "started" and len(calls) == 1, payload
    new_root = request["execution"]["workspaceRoot"]
    assert payload["continuation"]["workspace"] == "fresh" and new_root != old_root
    assert pathlib.Path(new_root).is_dir()
    # The host binds the new copy in the new instructions ...
    assert _BOUND + new_root in request["instructions"] and old_root not in request["instructions"]
    # ... yet with no carrier constraint the engine may resume the native session that keeps
    # the first request's instructions (``_BOUND + old_root``) ...
    assert request["continueFrom"] == "run-orient" and "continueCarrier" not in request
    # ... so the prompt, the one channel a resumed session is sure to read, must name the new root.
    assert new_root in request["prompt"], request["prompt"]


# --------------------------------------------------------------------------- the pairings

_HEADER = "\n\nTHIS RUN'S ACCESS AND WORKSPACE (current host facts of this turn):"
_BINDING = "\n\nDELEGATED EXECUTION BINDING"
_KINDS = {_BINDING: "copy", "\n\nDELEGATED IN-PLACE BINDING": "in_place",
          "\n\nDELEGATED DIRECTORY COPY BINDING": "folder_copy"}
_GRANTED = re.compile(r"in (\S+), the scratch your engine grants this run")
_GRANT_ROW = {"writeMechanism": "fs_sandbox", "accessControl": {"auxiliaryWritableRoots": True}}


class _Engine:
    """The serving engine as the start producer sees it: continues runs, every access profile,
    Git and folder work products, optionally the auxiliary-roots grant; records every body."""

    engine_version = "3.22.0"

    def __init__(self, run_id, calls, row, error):
        self.run_id, self.calls, self.row, self.error = run_id, calls, row, error

    def handshake(self, **_kw): return {}
    def agent_capabilities(self, **_kw):
        return {"runControlKeys": ["continueFrom"], "mutability": {"workspaceKinds": ["git", "directory"]},
                "harnesses": [{"id": "some-route", "enabled": True, "status": "ok",
                               "accessProfilesSupported": ["readonly", "workspace_write", "full"], **self.row}]}
    def quota_snapshots(self): return []
    def find_project_id(self, root): return "prj-existing"
    def register_project(self, root): return "prj-new"
    def remove_project(self, project_id): return {}
    def ensure_full_access(self, root): return {}
    def start_run(self, request, *, idempotency_key=""):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return {"runId": self.run_id, "runDir": f"/tmp/{self.run_id}"}
    def close(self): pass


def _ctx(tmp_path, kind):
    """A Git writer's task (its own worktree) or an ordinary-folder writer's (no Git)."""
    from ouroboros.tools.registry import ToolContext
    from tests._delegated_transport_shared import _delegating_ctx

    if kind == "git":
        return _delegating_ctx(tmp_path, acting=True, task_id="t-nanny-write")
    folder, data = tmp_path / "documents", tmp_path / "data"  # a folder inside the drive is refused
    for path in (tmp_path / "system", data, folder):
        path.mkdir(exist_ok=True)
    (folder / "notes.txt").write_text("seed\n", encoding="utf-8")
    return ToolContext(repo_dir=tmp_path / "system", drive_root=data, task_id="t-folder",
                       workspace_root=folder, workspace_mode="external")


def _leg(tmp_path, monkeypatch, ctx, run_id, *, prompt="carry on", access=None, row=None, error=None, **kwargs):
    """One real ``_delegate_start`` leg; ``access`` is the profile the selected row captures."""
    import ouroboros.tools.delegate as delegate
    from ouroboros.gateways import claudexor as gw

    calls = []
    _select_transport_access(monkeypatch, access)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    monkeypatch.setenv("OUROBOROS_SUBAGENT_WORKTREE_ROOT", str(tmp_path / "snap_root"))
    monkeypatch.setattr(gw, "ClaudexorGateway", lambda *a, **k: _Engine(run_id, calls, row or {}, error))
    delegate._CUSTODY.clear()
    payload = json.loads(delegate._delegate_start(ctx, prompt, **kwargs).text)
    delegate._CUSTODY.clear()
    if error is None:
        assert payload["status"] == "started" and len(calls) == 1, payload
    return (calls[-1] if calls else None), payload


def _actual_root(request):
    """Where the engine starts the harness: ``execution.workspaceRoot``, else ``scope.root``."""
    return (request.get("execution") or {}).get("workspaceRoot") or request["scope"]["root"]


def _run_blocks(text):
    """Every (access line, binding) a text states as a run's own: the turn's first, quoted ones after.
    Each binding renderer, and the no-copy line, is one paragraph after the one access line."""
    return [tuple(piece.split("\n\n")[:2]) for piece in text.split(_HEADER)[1:]]


def _kind(binding):
    return next((name for prefix, name in _KINDS.items() if ("\n\n" + binding).startswith(prefix)), "no_copy")


def _named_root(binding):
    """The root a binding names as its run's, or None (an engine folder copy names none)."""
    marker = {"copy": _BOUND, "in_place": "you work directly in ", "no_copy": "its working directory is "}.get(
        _kind(binding))
    return re.split(r"[:,]? ", binding.split(marker, 1)[1].rstrip("."), maxsplit=1)[0] if marker else None


def _current(request, block):
    """Identity, never position: the block whose root is where the engine starts THIS run and whose
    bytes the current instructions carry (or, with no renderer, the line naming that root)."""
    line, binding = block
    root, instructions = _actual_root(request), request["instructions"]
    if line != access_instruction(request["access"], investigation="PURPOSE: this is an evidence-only" in line):
        return False
    if _kind(binding) == "no_copy":
        return "\n\n" + binding == no_copy_binding_instruction(root)
    return "\n\n" + binding in instructions and _named_root(binding) in (root, None)


def _turn(request, caller, *, investigation=False):
    """Check one continuation turn against the request it rides in; return its run facts.

    Compares values, never substrings alone: the access line is the one ``request.access``
    renders, the binding is byte-identical to the one in the current instructions (so their
    digest and scratch agree), its root is where the engine starts the harness and its
    scratch is the first ``auxiliaryWritableRoots`` grant, or none is named."""
    prompt, instructions = request["prompt"], request["instructions"]
    assert prompt.count(_HEADER) == 1, prompt
    (line, binding), = _run_blocks(prompt)
    rest = prompt.split(_HEADER, 1)[1]
    assert rest == line + "\n\n" + binding + ("\n\n" + caller if caller else ""), rest
    assert line == access_instruction(request["access"], investigation=investigation) and line in instructions
    root, kind = _actual_root(request), _kind(binding)
    grants = (request.get("execution") or {}).get("auxiliaryWritableRoots") or []
    assert _current(request, (line, binding)), binding
    if kind == "no_copy":
        assert not any(prefix in instructions for prefix in _KINDS)  # no renderer covers this run
    assert _named_root(binding) == (None if kind == "folder_copy" else root)
    if kind == "copy":
        assert execution_binding_fingerprint(root, request["scope"]["root"]) in binding
        assert f"The stable authority/project root {request['scope']['root']} is a read-only identity" in binding
        assert _GRANTED.findall(binding) == grants[:1], grants
    if kind == "folder_copy":
        assert "Use only the engine-provided working directory/cwd" in binding
        assert "workspaceRoot" not in request["execution"]
    return {"kind": kind, "root": root, "block": (line, binding), "binding": binding}


def _pairings(legs, current, facts):
    """A resumed native session may keep ANY earlier leg's instructions (a packet leg opened a
    new session), the current ones, or both; a packet re-brief reads the current instructions
    and the current turn, then quotes earlier prompts after it. Every view leaves one reading."""
    roots = lambda text: re.findall(re.escape(_BOUND) + r"(\S+?):", text)  # noqa: E731
    own = [facts["root"]] if facts["kind"] == "copy" else []
    assert roots(current["instructions"]) == own == roots(_run_blocks(current["prompt"])[0][1])  # new only
    for earlier in legs:  # old only, or both: what an earlier set names is scoped to ITS run
        assert _ACCESS_PRECEDENCE in earlier["instructions"] and "the later DELEGATED" not in earlier["instructions"]
        assert set(roots(earlier["instructions"])) <= {_actual_root(earlier)}
    packet = (current["prompt"] + "\n\n[engine notice]\n\n## Original work order\n"
              + "\n\n".join(leg["prompt"] for leg in legs))
    blocks = _run_blocks(packet)
    assert blocks[0] == facts["block"] and len(blocks) == 1 + sum(_HEADER in leg["prompt"] for leg in legs)
    assert all(block == facts["block"] for block in blocks if _current(current, block)), blocks


def _write(request, name):
    root = pathlib.Path(_actual_root(request))
    (root / name).write_text(f"{name}\n", encoding="utf-8")


def _close(tmp_path, ctx, run_id, disposition):
    """Close a settled Git writer's leg the way its supervisor would."""
    from ouroboros.tools.subagent_integration_delegated import _integrate_delegated_patch

    capture = capture_terminal_patch_for_drive(tmp_path, custody.replay(tmp_path)[run_id])
    if disposition in ("applied", "rejected"):
        verdict = _integrate_delegated_patch(ctx, run_id=run_id, decision={"applied": "apply"}.get(disposition, "reject"))
        assert "⚠️" not in verdict, verdict
    custody._CUSTODY.clear()
    return capture


# --------------------------------------------------------------------------- the chains

@pytest.mark.parametrize("second", ["undisposed", "no_changes", "applied", "rejected"])
def test_every_leg_of_a_chain_states_its_own_root_access_and_scratch(tmp_path, monkeypatch, second):
    """R1 → R2 → R2 (the second leg's copy is still undisposed) and R1 → R2 → R3 (it was closed
    as no_changes, applied or rejected); the auxiliary grant gives every copy its own scratch."""
    ctx = _ctx(tmp_path, "git")
    legs = []
    first, _p = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="orient", row=_GRANT_ROW)
    legs.append(first)
    _settle(tmp_path, "run-1", ctx.task_id, state="succeeded", reason="completed")
    assert _close(tmp_path, ctx, "run-1", "no_changes")["disposition"] == custody.NO_CHANGES

    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="implement", row=_GRANT_ROW,
                            continue_from="run-1")
    facts = _turn(request, "implement")
    assert payload["continuation"]["workspace"] == "fresh" and facts["root"] != _actual_root(first)
    _pairings(legs, request, facts)
    legs.append(request)
    if second != "no_changes":
        _write(request, "second.txt")
    _settle(tmp_path, "run-2", ctx.task_id, state="succeeded", reason="completed")
    if second != "undisposed":
        _close(tmp_path, ctx, "run-2", second)

    third, payload = _leg(tmp_path, monkeypatch, ctx, "run-3", prompt="finish", row=_GRANT_ROW,
                          continue_from="run-2")
    facts = _turn(third, "finish")
    _pairings(legs, third, facts)
    roots = [_actual_root(leg) for leg in (*legs, third)]
    if second == "undisposed":  # R1 → R2 → R2: the turn restates R2 though the first session named R1
        assert payload["continuation"]["workspace"] == "same_snapshot" and roots[2] == roots[1] != roots[0]
        assert facts["block"] == _run_blocks(request["prompt"])[0]
    else:  # R1 → R2 → R3
        assert payload["continuation"]["workspace"] == "fresh" and len(set(roots)) == 3
    # Only the run's binding moves: the scope, the task contract and every host fact before it stay.
    assert third["scope"]["root"] == first["scope"]["root"] == str(pathlib.Path(ctx.workspace_root).resolve())
    assert len({leg["instructions"].split(_BINDING)[0] for leg in (*legs, third)}) == 1
    tree = {"undisposed": "SAME private snapshot", "no_changes": "changed no files to look for",
            "applied": "were APPLIED", "rejected": "were REJECTED"}[second]
    assert tree in third["prompt"].split(_HEADER, 1)[0]


def test_a_recreated_checkout_is_restated_at_the_same_path(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, "git")
    first, _p = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="start")
    _write(first, "work.txt")
    _settle(tmp_path, "run-1", ctx.task_id)
    assert _close(tmp_path, ctx, "run-1", "captured")["status"] == "ready_with_changes"
    assert not pathlib.Path(_actual_root(first)).exists()  # released with its saved capture
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="continue", continue_from="run-1")
    facts = _turn(request, "continue")
    assert payload["continuation"]["checkout"] == "recreated" and facts["root"] == _actual_root(first)
    assert pathlib.Path(facts["root"], "work.txt").is_file() and "restored from the saved result" in request["prompt"]
    _pairings([first], request, facts)


def test_an_investigation_continued_after_its_evidence_was_retained(tmp_path, monkeypatch):
    """A task without write authority investigates in a private copy; its successor gets a new one."""
    import subprocess

    from tests._delegated_transport_shared import _delegating_ctx

    ctx = _delegating_ctx(tmp_path, acting=False, task_id="t-nanny-read")
    for argv in (["init"], ["add", "-A"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "--allow-empty",
                                             "-m", "seed"]):  # the read tree it investigates is a Git tree
        subprocess.run(["git", *argv], cwd=ctx.repo_dir, capture_output=True, check=True)
    first, _p = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="investigate", access="workspace_write")
    assert "PURPOSE: this is an evidence-only investigation" in first["instructions"]
    _write(first, "probe.txt")
    _settle(tmp_path, "run-1", ctx.task_id, state="succeeded", reason="completed")
    assert _close(tmp_path, ctx, "run-1", "retained")["disposition"] == custody.EVIDENCE_RETAINED
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="look again", access="workspace_write",
                            continue_from="run-1")
    facts = _turn(request, "look again", investigation=True)
    assert payload["continuation"]["workspace"] == "fresh" and facts["root"] != _actual_root(first)
    assert "retained as evidence" in request["prompt"]
    _pairings([first], request, facts)


# --------------------------------------------------------------------------- access and place

def test_full_lowered_to_workspace_write_restates_access_and_the_newly_granted_scratch(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, "git")
    first, _p = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="start", access="full", row=_GRANT_ROW)
    assert first["access"] == "full" and not first["execution"].get("auxiliaryWritableRoots")
    _settle(tmp_path, "run-1", ctx.task_id)
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="continue", access="workspace_write",
                            row=_GRANT_ROW, continue_from="run-1")
    facts = _turn(request, "continue")
    assert payload["continuation"]["workspace"] == "same_snapshot" and facts["root"] == _actual_root(first)
    assert request["access"] == "workspace_write" and request["execution"]["auxiliaryWritableRoots"]
    assert _GRANTED.findall(facts["binding"]) and not _GRANTED.findall(first["instructions"])
    assert access_instruction("full") in first["instructions"] and access_instruction("full") not in request["prompt"]
    _pairings([first], request, facts)


def test_a_writer_lowered_to_readonly_names_the_root_it_reads(tmp_path, monkeypatch):
    ctx = _ctx(tmp_path, "git")
    first, _p = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="start", row=_GRANT_ROW)
    _write(first, "pending.txt")
    _settle(tmp_path, "run-1", ctx.task_id)
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="review", access="readonly",
                            row=_GRANT_ROW, continue_from="run-1")
    facts = _turn(request, "review")
    assert facts["kind"] == "no_copy" and request["access"] == "readonly" and payload["continuation"]["workspace"] == "fresh"
    assert facts["root"] == request["scope"]["root"] == str(pathlib.Path(ctx.workspace_root).resolve())
    assert "workspaceRoot" not in request["execution"] and _actual_root(first) not in request["prompt"]
    _pairings([first], request, facts)
    # A read-only run continued read-only is restated too, though nothing about its place changed.
    _settle(tmp_path, "run-2", ctx.task_id)
    again, _p = _leg(tmp_path, monkeypatch, ctx, "run-3", prompt="", access="readonly", continue_from="run-2")
    assert _turn(again, "")["binding"] == facts["binding"]


def test_in_place_after_a_closed_copy_and_a_copy_after_in_place(tmp_path, monkeypatch):
    """The copy's binding made the target read-only for ITS run; in place, the turn says so."""
    ctx = _ctx(tmp_path, "git")
    target = str(pathlib.Path(ctx.workspace_root).resolve())
    first, _p = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="orient")
    assert f"stable authority/project root {target} is a read-only" in first["instructions"]
    _settle(tmp_path, "run-1", ctx.task_id, state="succeeded", reason="completed")
    _close(tmp_path, ctx, "run-1", "no_changes")
    request, _payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="edit here", continue_from="run-1",
                             workspace="in_place")
    facts = _turn(request, "edit here")
    assert facts["kind"] == "in_place" and pathlib.Path(facts["root"]).resolve() == pathlib.Path(target)
    assert "what a private copy makes read-only belong to ONE run" in first["instructions"]
    _pairings([first], request, facts)
    _settle(tmp_path, "run-2", ctx.task_id)
    third, payload = _leg(tmp_path, monkeypatch, ctx, "run-3", prompt="now in a copy", continue_from="run-2")
    facts = _turn(third, "now in a copy")
    assert facts["kind"] == "copy" and facts["root"] != target and "wrote DIRECTLY" in third["prompt"]
    _pairings([first, request], third, facts)


@pytest.mark.parametrize("strategy", ["direct", "copy"])
def test_an_ordinary_folder_continuation_names_its_place(tmp_path, monkeypatch, strategy):
    """A direct folder run works in the folder itself; an engine copy's root is unknown before
    the POST, so its turn names the directory the engine starts it in."""
    ctx = _ctx(tmp_path, "folder")
    options = {"directory_strategy": strategy, "scope_paths": ["."]}
    first, payload = _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="tidy", **options)
    _settle(ctx.drive_root, "run-1", ctx.task_id)
    request, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="finish", continue_from="run-1", **options)
    facts = _turn(request, "finish")
    if strategy == "direct":
        assert facts["kind"] == "no_copy" and facts["root"] == request["execution"]["workspaceRoot"]
        assert pathlib.Path(facts["root"]).resolve() == pathlib.Path(ctx.workspace_root).resolve()
    else:
        assert facts["kind"] == "folder_copy" and request["execution"]["isolation"] == "envelope"
    _pairings([first], request, facts)


# --------------------------------------------------------------------------- what stays

def test_an_exact_retry_replays_the_recorded_turn_and_a_plan_turn_keeps_its_contract(tmp_path, monkeypatch):
    from ouroboros import delegate_continuation as continuation
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    ctx = _ctx(tmp_path, "git")
    _leg(tmp_path, monkeypatch, ctx, "run-1", prompt="start")
    _settle(tmp_path, "run-1", ctx.task_id)
    lost = ClaudexorUnavailable("daemon_unreachable", "connection reset", status_code=503)
    original, unknown = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="go on", error=lost, continue_from="run-1")
    facts = _turn(original, "go on")
    replayed, payload = _leg(tmp_path, monkeypatch, ctx, "run-2", prompt="go on",
                             retry_of=unknown["pending_invocation_id"])
    assert payload["status"] == "started" and replayed == original and _turn(replayed, "go on") == facts
    # Only an ordinary continuation is restated: a plan-thread turn and a plain start are untouched.
    plan = continuation.ContinuationStart(facts={"thread_id": "th-1", "continuation_of": "run-p"}, prompt="answer")
    assert continuation.with_run_binding(plan, None, "x", "/r") is plan
    assert continuation.with_run_binding(continuation.NO_CONTINUATION, None, "x", "/r") is continuation.NO_CONTINUATION


def test_the_persisted_line_scopes_run_facts_and_names_the_current_one_by_identity():
    from ouroboros.delegate_start_instructions import execution_binding_text

    for access in ("readonly", "workspace_write", "full"):
        assert _ACCESS_PRECEDENCE in access_instruction(access)
    for phrase in ("belong to ONE run", "quoted from an earlier run are history wherever they appear",
                   "the current run's DELEGATED EXECUTION BINDING", "within the same owner, task, target and access ceiling",
                   "the directory your harness started its current turn in",
                   "a path inside an earlier run's copy means the same relative path from there"):
        assert phrase in _ACCESS_PRECEDENCE, phrase
    assert "later" not in _ACCESS_PRECEDENCE  # position never decides which binding is current
    text = execution_binding_text("/copies/r2", "/authority")
    assert "Compare only the directory your harness started this run in, physically" in text
    assert "through a symlink, matches, and a later cd changes nothing" in text
    assert "If the harness cannot honor this binding" not in text
