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
import queue
from dataclasses import replace

import pytest

from ouroboros import consciousness_authority as ca
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests.test_consciousness_authority import _registry, _wake_task
from tests._cases import case_folders
from tests.test_place_focus import focus as focus


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
    ("configure_presence", {"action": "set_state", "place_id": "p1", "state": "paused"}),
    ("switch_model", {"scope": "room", "project_id": "alpha", "model": "vendor/chosen"}),
    ("switch_model", {"scope": "main", "model": "vendor/chosen"}),
]


def test_observe_refuses_a_mutating_shape_of_a_tool_it_still_sees(subtests, tmp_path, monkeypatch):
    """The negative half: the name is dispatchable, the mutating ARGUMENTS are not."""
    case_paths = case_folders(tmp_path)
    for name, args in _MUTATING_SHAPES:
        tmp_path = next(case_paths)
        with subtests.test(name=name, args=args), monkeypatch.context() as monkeypatch:
            monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
            wake = _registry(tmp_path, _wake_task("observe")["metadata"])
            assert wake.get_schema_by_name(name) is not None
            assert "RESOURCE_CONSTRAINT_BLOCKED" in wake.execute(name, args)


_PLACE_EDITS = [("connect", {"connector": "bridge"}), ("disconnect", {}), ("set_state", {"state": "paused"}),
                ("add_wake", {"fact": "mention"}), ("remove_wake", {"fact": "mention"}), ("add_note", {"text": "x"}),
                ("remove_note", {"note_id": "n"}), ("set_working_folder", {"working_folder": "/tmp"}),
                ("set_model", {"model": "m"}), ("confirm_owner_account", {"platform_actor_id": "a"}),
                ("revoke_owner_account", {"platform_actor_id": "a"}), ("update", {}), ("List", {}), ("list ", {}),
                ("create_circle", {"name": "Team", "model": "circle/model"}),
                ("edit_circle", {"circle_id": "c", "model": "circle/next"}), ("", {})]


def test_every_level_below_full_reads_places_and_changes_none(tmp_path, monkeypatch):
    """A wake below Full lists and inspects places; every other ``configure_presence`` action is refused.

    The two reads are the one reader of every room a place captured, which the wake's places block counts
    and names (16.09 withheld the whole name with the owner's place configuration). Both halves at the
    guard's seam: without the narrowing Act and Observe are refused ``list`` by name; without the refusal an
    edit reaches the place writer. Full, an owner turn and a child of an Act wake are pinned beside them.
    """
    from ouroboros import places
    from tests._safety_fixtures import disable_safety

    disable_safety(monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    for level, folder in zip(("observe", "act"), case_folders(tmp_path)):
        wake = _registry(folder, _wake_task(level)["metadata"])
        place = places.connect_place(folder / "drive", connector="bridge", provider="chat", conversation_id="c",
                                     label="Inbox", state="silent")
        listed = json.loads(wake.execute("configure_presence", {"action": "list"}))
        assert listed["ok"] and [row["place_id"] for row in listed["places"]] == [place.place_id], level
        inspected = json.loads(wake.execute("configure_presence", {"action": "inspect", "place_id": place.place_id}))
        assert inspected["ok"] and inspected["place"]["rooms"] == [], level
        for action, extra in _PLACE_EDITS:
            refused = wake.execute("configure_presence", {"action": action, "place_id": place.place_id, **extra})
            assert refused.startswith("⚠️ RESOURCE_CONSTRAINT_BLOCKED: below Full"), (level, action, refused)
        assert places.get_place(folder / "drive", place.place_id) == place  # nothing changed
    act = _wake_task("act")["metadata"]
    child = {**ca.consciousness_origin_metadata(act), "task_id": "child-1"}
    edit = {"action": "set_state", "place_id": "p", "state": "paused"}
    assert ca.observe_argument_refusal(child, "configure_presence", edit).startswith("⚠️")
    assert ca.observe_argument_refusal(_wake_task("full")["metadata"], "configure_presence", edit) == ""
    assert ca.observe_argument_refusal({"work_kind": "owner"}, "configure_presence", edit) == ""
    assert "configure_presence" not in ca.disabled_tools_for("act") + ca.disabled_tools_for("observe")


@pytest.fixture
def model_effects(monkeypatch):
    from tests._safety_fixtures import disable_safety

    disable_safety(monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    monkeypatch.setattr("ouroboros.tools.control_models.known_models",
                        lambda: [{"model": "vendor/chosen", "use_local": False}])
    monkeypatch.setattr("ouroboros.llm.LLMClient.available_models", lambda _self: ["vendor/chosen"])
    writes, recents = [], []
    monkeypatch.setattr("ouroboros.gateway.owner_effort._write_main_route",
                        lambda _request, choice, **kw: writes.append(("main", choice)) or {"ok": True})
    monkeypatch.setattr("ouroboros.composer_cognition.update_project_model_choice",
                        lambda root, pid, choice: writes.append((pid, choice)) or ({"kind": "main"}, choice))
    monkeypatch.setattr("ouroboros.gateway.ui_preferences.record_model_recent", lambda *a: recents.append(a))
    return writes, recents


def _place_model_registry(focus, monkeypatch, level):
    """Build the real place carrier, including focus_metadata's origin and autonomy."""
    from ouroboros import places
    from ouroboros.loop_messages import _initialize_owner_directives
    from ouroboros.owner_words import PLACE_CONFIRMED
    from ouroboros.presence_runner import _build_task

    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", level)
    root, place, admission, event = focus
    place = places.edit_place(root, place.place_id, "confirm_owner_account", author="owner", platform_actor_id="person")
    event = replace(event, text="Please use the other model.")
    task = _build_task(replace(admission, place=place), event, drive_root=root, staged_files=())
    metadata = task["metadata"]
    assert metadata["initiator"] == "consciousness" and metadata["consciousness_autonomy"] == level
    assert metadata["focus"]["actor"]["owner_confirmed"] is True
    reg = ToolRegistry(repo_dir=root, drive_root=root)
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=task["id"], current_task_type="presence",
                      is_direct_chat=True, current_chat_id=task["chat_id"], task_metadata=metadata,
                      task_contract=task["task_contract"], event_queue=queue.Queue())
    reg.set_context(ctx)
    _initialize_owner_directives(ctx, [{"role": "user", "content": "[FOCUS] the captured event"}])
    assert ctx._owner_directives[0]["source"] == PLACE_CONFIRMED
    return reg, task


@pytest.mark.parametrize("scope", ["room", "main"])
@pytest.mark.parametrize("level", ["observe", "act", "full"])
def test_place_cited_settings_choices_require_full_at_dispatch(focus, monkeypatch, model_effects, level, scope):
    reg, _task = _place_model_registry(focus, monkeypatch, level)
    writes, recents = model_effects
    assert reg.get_schema_by_name("switch_model") is not None
    args = {"scope": scope, "model": "vendor/chosen", "owner_words": "use the other model"}
    if scope == "room":
        args["project_id"] = "alpha"
    result = reg.execute_result("switch_model", args)
    if level == "full":
        assert result.status == "ok", result.text
        assert json.loads(result.text.splitlines()[0])["ok"] is True
        assert writes == [("alpha" if scope == "room" else "main",
                           {"kind": "model", "model": "vendor/chosen", "use_local": False})]
        assert len(recents) == 1
    else:
        assert result.status == "blocked" and result.code == "RESOURCE_CONSTRAINT_BLOCKED", result.text
        assert "below Full autonomy" in result.text and "Model capsule or by a word in Main" in result.text
        assert "a Full wake may carry the cite" in result.text
        assert not writes and not recents  # neither settings writer (and its act receipt) is reached
    assert reg._ctx.event_queue.empty()  # no supervisor model act was emitted


@pytest.mark.parametrize("scope", [None, "work"])
@pytest.mark.parametrize("level", ["observe", "act"])
def test_below_full_place_keeps_its_own_route_and_effort_switch(focus, monkeypatch, model_effects, level, scope):
    from ouroboros.owner_model import eligibility
    from ouroboros.task_results import write_task_result

    reg, task = _place_model_registry(focus, monkeypatch, level)
    args = {"model": "vendor/chosen", "effort": "high", **({"scope": scope} if scope else {})}
    result = reg.execute_result("switch_model", args)
    assert result.status == "ok", result.text
    assert reg._ctx.active_model_override == "vendor/chosen"
    assert reg._ctx.active_effort_override == "high"
    # A cited primary change still meets the separate durable-Presence refusal.
    assert eligibility(task) == "presence_model_unsupported"
    stored = write_task_result(reg._ctx.drive_root, task["id"], "running",
                               _presence_turn=task["_presence_turn"], metadata=task["metadata"])
    assert eligibility(stored) == "presence_model_unsupported"
    result = reg.execute_result("switch_model", {**args, "owner_words": "use the other model"})
    assert result.status == "error" and result.code == "TOOL_ARG_ERROR", result.text
    assert "no eligible durable root" in result.text
    assert model_effects == ([], []) and reg._ctx.event_queue.empty()


@pytest.mark.parametrize("level", ["observe", "act"])
def test_place_model_start_keeps_the_existing_level_boundary(focus, monkeypatch, model_effects, level):
    from ouroboros.tools import control_routing

    reg, _task = _place_model_registry(focus, monkeypatch, level)
    starts = []
    monkeypatch.setattr(control_routing, "_promotion_pool_disabled_from_snapshot", lambda _ctx: "")
    monkeypatch.setattr(control_routing, "_emit_and_wait_for_routing",
                        lambda _ctx, event: (starts.append(event) or "live", {"status": "scheduled"}))
    args = {"objective": "Write the report", "title": "Report", "destination": "place",
            "model": "vendor/chosen", "owner_words": "use the other model"}
    assert ca.observe_argument_refusal(reg._ctx.task_metadata, "start_task", args) == ""
    result = reg.execute_result("start_task", args)
    if level == "act":
        assert result.status == "ok", result.text
        assert starts[0]["metadata"]["lead"] == {
            "source": "owner_words", "model": "vendor/chosen", "use_local": False, "cited": "use the other model"}
    else:
        assert result.status == "blocked" and result.code == "RESOURCE_CONSTRAINT_BLOCKED"
        assert "disabled_tools" in result.text and "start_task" in result.text and not starts
    assert model_effects == ([], [])


@pytest.mark.parametrize("actor", ["main_chat", "project_chat", "owner_root"])
@pytest.mark.parametrize("source", ["initial_user", "live_call"])
@pytest.mark.parametrize("scope", ["work", "room", "main"])
def test_owner_foci_keep_cited_model_choices_at_dispatch(tmp_path, monkeypatch, model_effects, actor, source, scope):
    reg = _registry(tmp_path, {"work_kind": "owner", "client_message_id": "owner-message"})
    ctx = reg._ctx
    ctx.is_direct_chat = actor != "owner_root"
    ctx.project_id = "alpha" if actor == "project_chat" else ""
    ctx._owner_directives = [{"source": source, "content": "Please use the other model.", "utterance_id": "u1"}]
    requests = []
    monkeypatch.setattr("ouroboros.tools.control_models.request_work_model",
                        lambda _ctx, choice: (requests.append(choice) or '{"ok": true}', True))
    args = {"scope": scope, "model": "vendor/chosen", "owner_words": "use the other model"}
    if scope == "room":
        args["project_id"] = "alpha"
    result = reg.execute_result("switch_model", args)
    assert result.status == "ok" and json.loads(result.text.splitlines()[0])["ok"], result.text
    writes, recents = model_effects
    if scope == "work":
        assert requests == [{"kind": "model", "model": "vendor/chosen", "use_local": False, "cited": "use the other model"}]
        assert not writes and not recents
    else:
        assert not requests and len(recents) == 1
        assert writes == [("alpha" if scope == "room" else "main",
                           {"kind": "model", "model": "vendor/chosen", "use_local": False})]


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


def test_observe_cannot_dispose_any_captured_patch_at_dispatch(subtests, tmp_path, monkeypatch):
    """Fable scope B1: ``orphan_disposition_status`` lets a live TOP-LEVEL task
    dispose a foreign terminal owner's capture, and the payload apply rebinds
    the skill through the caller's profile — so a wake's read-only-children rule
    protects nothing here. The two verbs are therefore WITHHELD at Observe and
    refused by the dispatcher before any custody row is read: the run's target
    (Git root or installed payload) and its owner cannot make the call succeed,
    because the call never reaches the handler that would consult them.
    """
    case_paths = case_folders(tmp_path)
    for name, args in _DISPOSITION_SHAPES:
        tmp_path = next(case_paths)
        with subtests.test(name=name, args=args), monkeypatch.context() as monkeypatch:
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


def test_full_can_create_and_edit_circle_models_through_dispatch(tmp_path, monkeypatch):
    from ouroboros import places
    from tests._safety_fixtures import disable_safety
    disable_safety(monkeypatch)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.place_projection.place_declarations", lambda _root: [])
    wake = _registry(tmp_path, _wake_task("full")["metadata"])
    root = tmp_path / "drive"
    place = places.connect_place(root, connector="bridge", provider="chat")
    created = json.loads(wake.execute("configure_presence", {
        "action": "create_circle", "name": "Team", "model": "circle/model"}))["circle"]
    assert created["model"] == "circle/model"
    places.edit_place(root, place.place_id, "set_circle", author="owner", circle_id=created["circle_id"])
    edited = json.loads(wake.execute("configure_presence", {
        "action": "edit_circle", "circle_id": created["circle_id"], "model": "circle/next",
        "expected_revision": created["revision"]}))["circle"]
    assert edited["model"] == "circle/next" and edited["revision"] > created["revision"]
    inspected = json.loads(wake.execute("configure_presence", {"action": "inspect", "place_id": place.place_id}))
    assert inspected["place"]["model_route"] == {"value": "circle/next", "layer": "circle"}
    cleared = json.loads(wake.execute("configure_presence", {
        "action": "edit_circle", "circle_id": created["circle_id"], "model": "",
        "expected_revision": edited["revision"]}))["circle"]
    assert cleared["model"] == ""
