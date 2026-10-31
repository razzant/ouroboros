"""What an Observe wake may ASK a tool to do, once it is allowed to call it.

Observe keeps a short list of mutating-table names visible because each has a
genuine read-only or own-work use (``OBSERVE_ARGUMENT_NARROWED``); withholding
the NAME would take that use away with the mutation. The narrowing is therefore
on the ARGUMENTS, at dispatch — the same seam ``test_consciousness_authority``
pins for a withheld name. These are the positive and negative halves of it,
including what a read-only descendant of such a wake may ask for in turn.
"""

from __future__ import annotations

import json

import pytest

from ouroboros import consciousness_authority as ca
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_consciousness_authority import _registry, _wake_task


_MUTATING_SHAPES = [
    ("schedule_subagent", {"objective": "write", "expected_output": "diff",
                           "write_surface": "external_workspace"}),
    ("schedule_subagent", {"objective": "write", "expected_output": "diff", "may_mutate": True}),
    ("delegate_start", {"prompt": "edit", "access": "workspace_write"}),
    ("delegate_start", {"prompt": "edit", "root": "skills/demo"}),
    ("write_file", {"root": "user_files", "path": "no.txt", "content": "no"}),
    ("write_file", {"path": "server.py", "content": "no"}),
    ("edit_text", {"root": "active_workspace", "path": "x.py", "old": "a", "new": "b"}),
    ("manage_schedules", {"action": "update", "schedule_id": "s1", "reason": "change time", "cron": "0 9 * * *"}),
]


@pytest.mark.parametrize(("name", "args"), _MUTATING_SHAPES)
def test_observe_refuses_a_mutating_shape_of_a_tool_it_still_sees(tmp_path, monkeypatch, name, args):
    """The negative half: the name is dispatchable, the mutating ARGUMENTS are not."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    wake = _registry(tmp_path, _wake_task("observe")["metadata"])
    assert wake.get_schema_by_name(name) is not None
    assert "RESOURCE_CONSTRAINT_BLOCKED" in wake.execute(name, args)


def test_observe_keeps_its_own_research_and_notes(tmp_path, monkeypatch):
    """The positive half: a read-only child, a read-only session and its own drive."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    wake = _registry(tmp_path, _wake_task("observe")["metadata"])
    for name in ("schedule_subagent", "delegate_start", "manage_schedules", "cancel_task"):
        assert wake.get_schema_by_name(name) is not None, name
    written = wake.execute("write_file", {"root": "task_drive", "path": "notes.txt",
                                          "content": "internal finding"})
    assert "BLOCKED" not in written
    assert (tmp_path / "drive" / "task_drives" / "turn-1" / "notes.txt").read_text() == "internal finding"
    # An omitted access on the nanny verb is the READ-ONLY shape, not a refusal:
    # withholding it would take the research session away with the write.
    from ouroboros import consciousness_authority as authority

    observe_meta = _wake_task("observe")["metadata"]
    assert authority.observe_argument_refusal(observe_meta, "delegate_start", {"prompt": "read"}) == ""
    assert authority.observe_argument_refusal(observe_meta, "delegate_start", {"prompt": "read", "access": "readonly"}) == ""
    assert authority.observe_argument_refusal(observe_meta, "schedule_subagent", {"objective": "read"}) == ""
    assert authority.observe_argument_refusal(observe_meta, "schedule_subagent",
                                              {"objective": "read", "write_surface": "read_only"}) == ""
    for action in ("list", "get", "disable", "delete", "restore"):
        assert authority.observe_argument_refusal(observe_meta, "manage_schedules", {"action": action}) == ""
    # An Act/Full turn is not narrowed at all by this seam.
    for level in ("act", "full"):
        assert authority.observe_argument_refusal(_wake_task(level)["metadata"], "write_file",
                                                  {"root": "user_files", "path": "x"}) == ""
    assert authority.observe_argument_refusal({"client_message_id": "cm"}, "write_file",
                                              {"root": "user_files", "path": "x"}) == ""


def test_the_write_root_observe_may_choose_still_confines_the_path_it_names(tmp_path, monkeypatch):
    """The level picks the BASE; the file tool still resolves the path inside it.

    ``observe_argument_refusal`` narrows ``root`` and leaves the positive path
    open on purpose, so the only thing standing between "Observe may write its
    task_drive" and "Observe may write anything" is that confinement. These are
    the bypasses it has to refuse: traversal, an absolute path, and a symlink
    inside the drive that points out of it.
    """
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    wake = _registry(tmp_path, _wake_task("observe")["metadata"])
    drive_dir = tmp_path / "drive" / "task_drives" / "turn-1"
    # The positive path first: without it, "refused" would prove nothing.
    assert "OK:" in wake.execute("write_file", {"root": "task_drive", "path": "notes.txt",
                                               "content": "internal finding"})
    outside = tmp_path / "outside.txt"
    outside.write_text("owner bytes\n", encoding="utf-8")
    drive_dir.mkdir(parents=True, exist_ok=True)
    (drive_dir / "escape.txt").symlink_to(outside)
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()
    (drive_dir / "escape_dir").symlink_to(outside_dir, target_is_directory=True)
    for label, path in (
        ("traversal", "../../../outside.txt"),
        ("absolute", str(outside)),
        ("absolute into the repo", str(tmp_path / "repo" / "README.md")),
        ("symlinked file", "escape.txt"),
        ("symlinked directory", "escape_dir/x.txt"),
    ):
        refused = wake.execute("write_file", {"root": "task_drive", "path": path, "content": "no"})
        assert refused.startswith("⚠️ WRITE_FILE_ERROR"), (label, refused)
    assert outside.read_text(encoding="utf-8") == "owner bytes\n"
    assert sorted(item.name for item in outside_dir.iterdir()) == []
    # edit_text lands on the same confinement through the same chosen root.
    assert "⚠️" in wake.execute("edit_text", {"root": "task_drive", "path": str(outside),
                                              "old": "owner", "new": "no"})
    assert outside.read_text(encoding="utf-8") == "owner bytes\n"


def test_observe_delegated_session_stays_readonly_when_access_is_omitted(tmp_path):
    """The nanny verb's SHAPE, not just its arguments: Observe hosts read-only."""
    from ouroboros.tools.delegate import _derive_authority

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="wake-1",
                      is_direct_chat=True, task_metadata=_wake_task("observe")["metadata"])
    for kwargs in ({}, {"access": "workspace_write"}, {"access": "full"}):
        shape = _derive_authority(ctx, **kwargs)
        assert shape.access == "readonly" and shape.isolation != "live", kwargs
        assert shape.delegated is False, kwargs


def test_schedule_control_is_root_authority_but_reading_is_not(tmp_path):
    """A delegated task may LIST; only the root turn may change a schedule."""
    # The child of an owner task: the Supervisor does not hold it, so the refusal is the schedule tool's own.
    child = _registry(tmp_path, {"delegation_role": "subagent", "work_kind": "owner"})
    listed = child.execute_result("manage_schedules", {"action": "list"})
    assert listed.status == "ok" and json.loads(listed.text)["tasks"] == []
    for action in ("update", "disable", "delete", "restore"):
        refused = child.execute_result("manage_schedules",
                                       {"action": action, "schedule_id": "s1", "reason": "because"})
        assert refused.status == "blocked", action
        assert refused.code in {"ACCESS_BLOCKED", "RESOURCE_CONSTRAINT_BLOCKED"}
    # Reading the schedule table is research, so it is in the read-only catalog.
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES

    assert "manage_schedules" in LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    # Registered handlers answer TEXT (ToolEntry.handler -> str); the typed result
    # rides the published sidecar, which is how the status above is still known.
    from ouroboros.tools.followup import _manage_schedules

    for args in ({"action": "list"}, {"action": "get", "schedule_id": "s1"},
                 {"action": "update", "schedule_id": "s1", "reason": "x", "objective": "Changed"},
                 {"action": "disable", "schedule_id": "s1", "reason": "x"}):
        assert isinstance(_manage_schedules(child._ctx, **args), str), args


def test_schedule_mutation_uses_resolved_subagent_profile_even_without_lineage_metadata(tmp_path):
    """A stale/missing delegation_role cannot turn a delegated profile into an owner."""
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.tools.followup import _manage_schedules

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    ctx = ToolContext(
        repo_dir=repo, drive_root=drive, task_id="child-1",
        task_constraint=TaskConstraint(mode="local_readonly_subagent"), task_metadata={},
    )
    blocked = _manage_schedules(ctx, action="disable", schedule_id="s1", reason="pause")
    assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked


def test_schedule_mutation_fails_closed_when_profile_resolution_is_unavailable(tmp_path, monkeypatch):
    """A profile lookup error cannot be treated as proof of root schedule authority."""
    from ouroboros.tools.followup import _manage_schedules

    repo = tmp_path / "repo"
    drive = tmp_path / "drive"
    repo.mkdir()
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="turn-1", task_metadata={})
    monkeypatch.setattr("ouroboros.tool_access.active_tool_profile",
                        lambda _ctx: (_ for _ in ()).throw(RuntimeError("profile unavailable")))
    blocked = _manage_schedules(ctx, action="disable", schedule_id="s1", reason="pause")
    assert "RESOURCE_CONSTRAINT_BLOCKED" in blocked


def test_presence_gets_no_schedule_authority_at_all(tmp_path, monkeypatch):
    """No Presence expansion: an admitted guest room neither reads nor writes it."""
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")  # the tool's own boundary, not Safety
    guest = _registry(tmp_path, {"presence": {"event": {"source_event_id": "presence-1"}}})
    for args in ({"action": "list"}, {"action": "get", "schedule_id": "s1"},
                 {"action": "update", "schedule_id": "s1", "reason": "x", "objective": "Changed"},
                 {"action": "disable", "schedule_id": "s1", "reason": "x"}):
        result = guest.execute_result("manage_schedules", args)
        assert result.status == "blocked"
        assert result.code in {"ACCESS_BLOCKED", "RESOURCE_CONSTRAINT_BLOCKED"}
# --- nested dispatch: what a read-only descendant may ask for ----------------------


def _readonly_child_registry(tmp_path, level="observe"):
    """A read-only research child of a consciousness wake, as dispatch sees it."""
    from ouroboros.contracts.task_constraint import TaskConstraint

    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "README.md").write_text("ok\n", encoding="utf-8")
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    metadata = dict(_wake_task(level)["metadata"])
    metadata.update(delegation_role="subagent", root_task_id="wake-1", parent_task_id="wake-1")
    reg.set_context(ToolContext(
        repo_dir=repo, drive_root=drive, task_id="child-1",
        task_constraint=TaskConstraint(mode="local_readonly_subagent"),
        task_metadata=metadata,
    ))
    return reg


def test_choosing_the_next_wake_stays_with_the_waking_mind_and_does_not_descend(tmp_path, monkeypatch):
    """``set_next_wakeup`` is an allow/deny rule, not an unclassified leftover.

    The approved wake prompt asks this mind to choose its own next interval, so
    Observe keeps the verb — managing a SCHEDULE RECORD and choosing when to
    wake next are different operations and the boundary says so out loud. A
    read-only descendant is not the waking mind and never gets it; keeping the
    verb is not permission to enable the alarm or move the owner's bounds.
    """
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    for level in ("observe", "act"):
        wake = _registry(tmp_path, _wake_task(level)["metadata"])
        assert wake.get_schema_by_name("set_next_wakeup") is not None, level
        assert "set_next_wakeup" not in ca.disabled_tools_for(level), level
    assert _readonly_child_registry(tmp_path).get_schema_by_name("set_next_wakeup") is None
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES

    assert "set_next_wakeup" not in LOCAL_READONLY_SUBAGENT_TOOL_NAMES


def test_the_observe_boundary_names_every_tool_it_narrows_rather_than_implying_one(tmp_path):
    """The narrowed list is the inventory; a name on it must have a dispatch rule.

    A tool kept VISIBLE for a read-only use is only safe while its mutating
    argument shape is actually refused. Pinning the two sets against each other
    means a later catalog addition cannot quietly join the visible half without
    the refusal half arriving with it.
    """
    narrowed = set(ca.OBSERVE_ARGUMENT_NARROWED)
    disabled = set(ca.disabled_tools_for("observe"))
    assert not (narrowed & disabled), "a narrowed tool is visible, so it cannot also be withheld"
    # Narrowed by something other than a refused argument SHAPE, each named here
    # so the exemption is a decision rather than a gap nobody noticed:
    #   journal_write / workpad_write — the owner approved these as Observe's own
    #     project notes; their bound is the note surface, not a refusal.
    #   cancel_task — narrowed to the children this wake itself started, which
    #     `test_observe_may_stop_its_own_child_but_not_the_owners_running_work` owns.
    bounded_elsewhere = {"journal_write", "workpad_write", "cancel_task"}
    covered = {name for name, _args in _MUTATING_SHAPES}
    assert narrowed <= covered | bounded_elsewhere, (
        f"argument-narrowed with no refusal pinned anywhere: {sorted(narrowed - covered - bounded_elsewhere)}")
    assert bounded_elsewhere <= narrowed, "an exemption for a name that is no longer narrowed"


def test_a_readonly_child_may_research_further_but_never_spawn_a_mutating_one(tmp_path, monkeypatch):
    """Recursion is allowed; the authority does not grow on the way down."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    from ouroboros.tool_access import active_tool_profile
    from ouroboros.tools.control_scheduling import _select_subagent_constraint

    child = _readonly_child_registry(tmp_path)
    assert active_tool_profile(child._ctx) == "local_readonly_subagent"
    # The descendant it MAY have: a read-only one, chosen by the same selector.
    allowed = _select_subagent_constraint("", "", False, None, "", caller_readonly=True, ctx=child._ctx)
    assert allowed["mode"] == "local_readonly_subagent"
    # The descendant it may NOT have: every acting surface, refused at dispatch.
    for surface in ("self_worktree", "external_workspace"):
        refused = child.execute("schedule_subagent", {
            "objective": "edit the repo", "expected_output": "a diff", "write_surface": surface})
        assert "BLOCKED" in refused, surface


def test_a_readonly_child_cannot_pass_a_mutating_delegation_budget_onward(tmp_path):
    """The legacy may_mutate flag is dropped rather than inherited by a descendant."""
    from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE, LOCAL_READONLY_SUBAGENT_MODE
    from ouroboros.tools.control_scheduling import _select_subagent_constraint, delegation_may_mutate

    assert delegation_may_mutate(True, LOCAL_READONLY_SUBAGENT_MODE) is False
    assert delegation_may_mutate(True, ACTING_SUBAGENT_MODE) is True
    assert delegation_may_mutate(True, "self_modification") is True
    assert delegation_may_mutate(False, "self_modification") is False
    # And the constraint selector refuses the acting child itself, not just its budget.
    refused = _select_subagent_constraint(
        "external_workspace", "", False, None, "", caller_readonly=True,
        ctx=_readonly_child_registry(tmp_path)._ctx)
    assert not isinstance(refused, dict)


def test_a_readonly_child_of_an_observe_wake_still_reads_schedules(tmp_path):
    """Inheritance carries the level down without taking the read away."""
    child = _readonly_child_registry(tmp_path)
    assert ca.is_observe_origin(child._ctx.task_metadata)
    listed = child.execute_result("manage_schedules", {"action": "list"})
    assert listed.status == "ok" and json.loads(listed.text)["tasks"] == []
    refused = child.execute_result(
        "manage_schedules", {"action": "disable", "schedule_id": "s", "reason": "why"})
    assert refused.status == "blocked"


def test_observe_may_stop_its_own_child_but_not_the_owners_running_work(tmp_path, monkeypatch):
    """The nanny's other half: starting a child implies being able to stop it.

    A wake is a TOP-LEVEL principal, whose cancel authority is ordinarily the
    whole queue. Observe is narrowed at the same custody seam a delegated task
    meets, so un-withholding the name grants own-child cancellation and nothing
    wider.
    """
    from ouroboros.tools import join_ledger

    wake = _registry(tmp_path, _wake_task("observe")["metadata"], task_id="wake-1")
    assert wake.get_schema_by_name("cancel_task") is not None  # the guard is at the tool
    monkeypatch.setattr(join_ledger, "_is_own_child", lambda *_a, **_k: False)
    refused = wake.execute("cancel_task", {"task_id": "owner-task", "reason": "stop it"})
    assert "ACCESS_BLOCKED" in refused or "not a child of this task" in refused
    # An owner turn at the same seam keeps the full top-level authority.
    owner = _registry(tmp_path, {"client_message_id": "cm"}, task_id="owner-turn")
    assert "not a child of this task" not in owner.execute(
        "cancel_task", {"task_id": "owner-task", "reason": "stop it"})
    # Its own child is cancellable: custody says yes, the level does not veto it.
    monkeypatch.setattr(join_ledger, "_is_own_child", lambda *_a, **_k: True)
    assert "not a child of this task" not in wake.execute(
        "cancel_task", {"task_id": "child-1", "reason": "enough"})


# --- disposing a captured patch is a world mutation, whoever captured it ---------


_DISPOSITION_SHAPES = [
    # a Git-root capture of an unrelated earlier task whose owner is terminal
    ("integrate_delegated_patch", {"run_id": "run-foreign-git", "decision": "apply"}),
    ("integrate_delegated_patch", {"run_id": "run-foreign-git", "decision": "reject",
                                   "reason": "release the orphan"}),
    # an installed skill payload: apply lands LIVE in the payload, no Git root involved
    ("integrate_delegated_patch", {"run_id": "run-foreign-payload", "decision": "apply",
                                   "reason": "adopt the payload edit"}),
    ("integrate_delegated_patch", {"run_id": "run-foreign-payload", "decision": "reject"}),
    ("integrate_subagent_patch", {"task_id": "child-of-someone-else", "decision": "apply"}),
    ("integrate_subagent_patch", {"task_id": "child-of-someone-else", "decision": "reject"}),
]


def _probe_handler(reg, name, calls):
    """Replace the tool's handler with a recorder, so a call that REACHES the
    handler is visible and a call the gate refuses provably never did."""
    import dataclasses

    entry = reg._entries[name]

    def probe(_ctx, **args):
        calls.append((name, dict(args)))
        return "PROBE_REACHED_HANDLER"

    projected = dataclasses.replace(entry, handler=probe)
    reg._entries[name] = projected
    if name in reg._scoped_entries:
        reg._scoped_entries[name] = projected


@pytest.mark.parametrize(("name", "args"), _DISPOSITION_SHAPES)
def test_observe_cannot_dispose_any_captured_patch_at_dispatch(tmp_path, monkeypatch, name, args):
    """Fable scope B1: ``orphan_disposition_status`` lets a live TOP-LEVEL task
    dispose a foreign terminal owner's capture, and the payload apply rebinds
    the skill through the caller's profile — so a wake's read-only-children rule
    protects nothing here. The two verbs are therefore WITHHELD at Observe and
    refused by the dispatcher before any custody row is read: the run's target
    (Git root or installed payload) and its owner cannot make the call succeed,
    because the call never reaches the handler that would consult them.
    """
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")  # the widest install still refuses
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    assert name in ca.OBSERVE_DISABLED
    calls: list = []
    wake = _registry(tmp_path, _wake_task("observe")["metadata"], task_id="wake-1")
    _probe_handler(wake, name, calls)
    assert wake.get_schema_by_name(name) is not None  # dispatch-only: the schema stays
    result = wake.execute(name, args)
    assert "RESOURCE_CONSTRAINT_BLOCKED" in result and name in result
    assert calls == [], "the disposition handler must never run under Observe"
    # The same verb from an Act wake DOES reach the handler: the refusal above is
    # the level's gate, not a broken tool.
    act_calls: list = []
    act = _registry(tmp_path, _wake_task("act")["metadata"], task_id="wake-2")
    _probe_handler(act, name, act_calls)
    # The independent Safety assessment is disabled; this checks the autonomy boundary.
    assert act.execute(name, args).startswith("PROBE_REACHED_HANDLER")
    assert act_calls == [(name, args)]


def test_evolution_stats_artifact_write_is_withheld_from_observe(tmp_path, monkeypatch):
    """The local export writes a task artifact; it is not an Observe-only read."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    assert "generate_evolution_stats" in ca.OBSERVE_DISABLED
    wake = _registry(tmp_path, _wake_task("observe")["metadata"])
    assert "RESOURCE_CONSTRAINT_BLOCKED" in wake.execute("generate_evolution_stats", {})
