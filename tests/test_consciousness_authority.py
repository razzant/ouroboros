"""Authority levels of a consciousness wake-up and how they follow its work (P3).

Observe / Act / Full (owner decision В10', default Act) are carried as
``metadata.consciousness_autonomy`` and derived at task build into the
contract's ``disabled_tools`` and the per-task ``runtime_mode_cap`` (В21=A);
for a consciousness-origin task the disabled list binds at DISPATCH ONLY so the
wake's tool schemas and prompt prefix are byte-identical to an owner turn's
(В31=B, the I3 comparison below). The origin (label, ledger category, level)
is inherited by everything the wake starts; a wake speaks as a task through
``steer_task`` (ISSUER, PLAN 5.2a); ``/evolve off`` is sticky against the agent
tool (В12); a Full-level campaign stays inside the consciousness tree.
"""

from __future__ import annotations

import json
import pathlib
import types

import pytest

from ouroboros import consciousness_authority as ca
from ouroboros.tools.registry import ToolContext, ToolRegistry

WAKE_META = {
    "initiator": "consciousness", "usage_category": "consciousness",
    "wake_reason": "heartbeat", "consciousness_autonomy": "act", "model_role": "consciousness",
}


def _wake_task(level="act", **extra):
    task = {"id": "wake-1", "type": "task", "text": "wake", "_is_direct_chat": True, "chat_id": 1,
            "metadata": {**WAKE_META, "consciousness_autonomy": level, **extra}}
    return ca.apply_consciousness_authority(task)


def _registry(tmp_path, metadata=None, *, task_id="turn-1"):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "README.md").write_text("ok\n", encoding="utf-8")
    drive = tmp_path / "drive"
    drive.mkdir(exist_ok=True)
    reg = ToolRegistry(repo_dir=repo, drive_root=drive)
    reg.set_context(ToolContext(
        repo_dir=repo, drive_root=drive, task_id=task_id, is_direct_chat=True,
        task_metadata=dict(metadata or {}),
    ))
    return reg


# --- the level tables -------------------------------------------------------------


def test_levels_and_their_two_consequences():
    assert ca.LEVELS == ("observe", "act", "full")
    assert ca.disabled_tools_for("full") == []
    assert ca.disabled_tools_for("act") == list(ca.ACT_DISABLED)
    observe = ca.disabled_tools_for("observe")
    assert set(ca.ACT_DISABLED) <= set(observe)
    assert {"promote_chat_to_task", "run_command",
            "browser_action", "initiate_presence", "submit_skill_to_hub"} <= set(observe)
    # The names with a real read-only path are kept and narrowed on ARGUMENTS.
    assert ca.OBSERVE_ARGUMENT_NARROWED.isdisjoint(observe)
    assert {"schedule_subagent", "delegate_start", "cancel_task"} <= ca.OBSERVE_ARGUMENT_NARROWED
    # The nanny of a running campaign is never withheld, at any level.
    assert "steer_task" not in observe and "steer_task" not in ca.disabled_tools_for("act")
    # Observe is an EXCEPTION list: reading and talking stay available by default.
    for name in ("read_file", "web_search", "browse_page", "send_user_message", "escalate",
                 "knowledge_write", "update_scratchpad", "switch_model", "enable_tools"):
        assert name not in observe
    assert ca.runtime_mode_cap_for("act") == "light" == ca.runtime_mode_cap_for("observe")
    assert ca.runtime_mode_cap_for("full") == ""


def test_unknown_level_falls_back_to_the_owner_setting(monkeypatch):
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_AUTONOMY", "observe")
    assert ca.normalize_level("bogus") == "observe"
    assert ca.normalize_level("") == "observe"
    assert ca.normalize_level("FULL") == "full"


def test_observe_table_covers_every_registry_entry_marked_mutates_worktree(tmp_path):
    """The registry marker is the second source of the same fact; the table cannot drift."""
    reg = _registry(tmp_path)
    marked = {e.name for e in reg._entries.values() if e.mutates_worktree and not e.alias_for}
    assert marked, "the catalog carries mutates_worktree entries"
    # The argument-narrowed names have a read-only Observe path; what they may be
    # ASKED to do is checked at dispatch instead of hiding the whole tool.
    missing = marked - set(ca.OBSERVE_DISABLED) - ca.OBSERVE_ARGUMENT_NARROWED
    assert not missing, f"mutates_worktree entries missing from OBSERVE_WORLD_MUTATION_TOOLS: {sorted(missing)}"
    unknown = set(ca.OBSERVE_DISABLED) - {e.name for e in reg._entries.values()}
    # Skill/project tools are registered lazily (skills, journal); the built-in names must exist.
    assert unknown <= {"toggle_skill", "skill_owner_action", "journal_write", "workpad_write",
                       "configure_presence", "initiate_presence", "delegate_start"}, sorted(unknown)


# --- derivation at task build ---------------------------------------------------


def test_apply_consciousness_authority_derives_both_consequences_once():
    task = _wake_task("act")
    assert task["metadata"]["disabled_tools"] == list(ca.ACT_DISABLED)
    assert task["metadata"]["runtime_mode_cap"] == "light"
    full = _wake_task("full")
    assert full["metadata"]["disabled_tools"] == [] and full["metadata"]["runtime_mode_cap"] == ""
    # An explicit producer list stands; an owner turn is untouched.
    explicit = _wake_task("act", disabled_tools=["web_search"])
    assert explicit["metadata"]["disabled_tools"] == ["web_search"]
    owner = ca.apply_consciousness_authority({"id": "o", "metadata": {"client_message_id": "cm"}})
    assert "disabled_tools" not in owner["metadata"] and "runtime_mode_cap" not in owner["metadata"]


def test_contract_carries_the_derived_list_and_origin_helpers():
    from ouroboros.contracts.task_contract import attach_task_contract

    task = attach_task_contract(_wake_task("observe"))
    assert task["task_contract"]["disabled_tools"] == ca.disabled_tools_for("observe")
    assert "toggle_evolution" in ca.task_disabled_tools(task)
    origin = ca.consciousness_origin_metadata(task["metadata"])
    assert origin == {"initiator": "consciousness", "usage_category": "consciousness_task",
                      "consciousness_autonomy": "observe"}
    assert ca.consciousness_origin_metadata({"client_message_id": "cm"}) == {}
    assert ca.is_consciousness_origin(origin) and not ca.is_consciousness_origin(None)


@pytest.mark.parametrize(("install", "cap", "expected"), [
    ("cyber_pro", "light", "light"), ("pro", "light", "light"), ("advanced", "light", "light"),
    ("light", "light", "light"), ("light", "advanced", "light"), ("cyber_pro", "", "cyber_pro"),
    ("pro", "bogus", "pro"),
])
def test_effective_runtime_mode_is_the_stricter_of_install_and_cap(install, cap, expected):
    assert ca.effective_runtime_mode(install, {"runtime_mode_cap": cap}) == expected
    assert ca.effective_runtime_mode(install, None) == install


# --- dispatch-only enforcement (В31=B) -------------------------------------------


def test_consciousness_contract_keeps_the_full_schema_set_and_refuses_at_dispatch(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    main = _registry(tmp_path, {"client_message_id": "cm-1"})
    wake = _registry(tmp_path, _wake_task("act")["metadata"])
    # Same schemas, same advertised names, same initial envelope, same omission manifest.
    assert wake.schemas() == main.schemas()
    assert wake.available_tools() == main.available_tools()
    assert wake.initial_tool_names() == main.initial_tool_names()
    assert wake.capability_omissions() == main.capability_omissions()
    assert not any(item.get("reason") == "disabled_by_contract" for item in wake.capability_omissions())
    assert "toggle_evolution" in wake.available_tools()
    assert wake.get_schema_by_name("toggle_evolution") is not None
    assert wake.policy_hidden_reason("toggle_evolution") is None
    # The dispatcher is the mechanism: the withheld name is refused with the typed block.
    result = wake.execute("toggle_evolution", {"enabled": True, "objective": "x"})
    assert "RESOURCE_CONSTRAINT_BLOCKED" in result and "toggle_evolution" in result
    for name, args in (("request_restart", {}), ("set_tool_timeout", {"seconds": 30}),
                       ("toggle_consciousness", {"action": "stop"})):
        assert "RESOURCE_CONSTRAINT_BLOCKED" in wake.execute(name, args), name


def test_an_ordinary_contract_still_hides_its_disabled_tools(tmp_path):
    """The dispatch-only case is the consciousness special case, not a general change."""
    reg = _registry(tmp_path, {"disabled_tools": ["toggle_evolution"]})
    assert "toggle_evolution" not in reg.available_tools()
    assert all(s["function"]["name"] != "toggle_evolution" for s in reg.schemas())
    assert reg.get_schema_by_name("toggle_evolution") is None
    assert reg.policy_hidden_reason("toggle_evolution") == "disabled by this task's contract (disabled_tools)"
    assert any(item.get("reason") == "disabled_by_contract" for item in reg.capability_omissions())


def test_i3_serialized_request_prefix_matches_an_owner_turn(tmp_path, monkeypatch):
    """The provider request's cached prefix — the tool schema array and the two
    cached system blocks up to the dynamic boundary (context_fit) — is byte-identical
    for an owner turn and for a wake at Act and at Observe built from one snapshot."""
    from ouroboros.context import build_llm_messages
    from tests.test_cache_optimization import _make_env_and_memory

    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    env, memory = _make_env_and_memory(tmp_path)
    owner = {"id": "t-owner", "type": "task", "text": "hi", "_is_direct_chat": True, "chat_id": 1,
             "metadata": {"client_message_id": "cm-1"}}
    owner_msgs, _ = build_llm_messages(env=env, memory=memory, task=owner)
    owner_prefix = [json.dumps(owner_msgs[0]["content"][i], sort_keys=True) for i in (0, 1)]
    owner_reg = _registry(tmp_path, owner["metadata"], task_id="t-owner")
    owner_tools = json.dumps(owner_reg.schemas(), sort_keys=True)
    for level in ("act", "observe"):
        task = _wake_task(level)
        task.update(id="t-owner", text="hi")  # the same turn: the wake differs only in its metadata
        msgs, _ = build_llm_messages(env=env, memory=memory, task=task)
        prefix = [json.dumps(msgs[0]["content"][i], sort_keys=True) for i in (0, 1)]
        assert prefix == owner_prefix, level
        assert "cache_control" not in msgs[0]["content"][2]
        reg = _registry(tmp_path, task["metadata"], task_id="t-owner")
        assert json.dumps(reg.schemas(), sort_keys=True) == owner_tools, level
        assert reg.capability_omissions() == owner_reg.capability_omissions(), level


# --- the per-task mode cap (В21=A): level x install mode --------------------------


_BLOCKED_CALLS = (
    ("write_file", {"path": "README.md", "content": "changed\n"}),
    ("commit_reviewed", {"commit_message": "test"}),
    ("run_command", {"cmd": "touch x.py"}),
    ("start_service", {"cmd": ["sleep", "5"], "name": "svc"}),
)


@pytest.mark.parametrize("mode", ["light", "advanced", "pro", "cyber_pro"])
@pytest.mark.parametrize("level", ["act", "observe"])
@pytest.mark.parametrize(("tool_name", "args"), _BLOCKED_CALLS)
def test_act_and_observe_cannot_touch_the_repo_in_any_install_mode(tmp_path, monkeypatch, mode, level, tool_name, args):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    reg = _registry(tmp_path, _wake_task(level)["metadata"])
    result = reg.execute(tool_name, dict(args))
    expected = "RESOURCE_CONSTRAINT_BLOCKED" if level == "observe" and tool_name != "commit_reviewed" else "LIGHT_MODE_BLOCKED"
    assert expected in result, (mode, level, tool_name, result[:300])
    assert not (tmp_path / "repo" / "x.py").exists()
    assert (tmp_path / "repo" / "README.md").read_text(encoding="utf-8") == "ok\n"


@pytest.mark.parametrize("mode", ["advanced", "pro", "cyber_pro"])
def test_full_follows_the_install_mode(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", mode)
    reg = _registry(tmp_path, _wake_task("full")["metadata"])
    assert "LIGHT_MODE_BLOCKED" not in reg.execute("write_file", {"path": "scratch.txt", "content": "changed\n"})
    assert (tmp_path / "repo" / "scratch.txt").read_text(encoding="utf-8") == "changed\n"
    assert "LIGHT_MODE_BLOCKED" not in reg.execute("run_command", {"cmd": "touch x.py"})


def test_full_in_a_light_install_is_still_light(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "light")
    reg = _registry(tmp_path, _wake_task("full")["metadata"])
    assert "LIGHT_MODE_BLOCKED" in reg.execute("write_file", {"path": "README.md", "content": "x"})


@pytest.mark.parametrize(("level", "expected"), [("act", "LIGHT_MODE_BLOCKED"),
                                                 ("observe", "RESOURCE_CONSTRAINT_BLOCKED")])
def test_a_repo_path_reached_through_user_files_is_still_light_blocked(tmp_path, monkeypatch, level, expected):
    """The P3 residual: a cyber_pro install resolves ``user_files`` to a base that
    CONTAINS the repo, so the ROOT NAME alone cannot decide the light gate — the
    resolved target must. A light-capped wake writing a repository path under
    that root is refused; Observe never reaches the gate (the tool is withheld)."""
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    reg = _registry(tmp_path, _wake_task(level)["metadata"])
    target = tmp_path / "repo" / "x.py"
    result = reg.execute("write_file", {"root": "user_files", "path": "repo/x.py", "content": "print(1)\n"})
    assert expected in result, (level, result[:300])
    assert not target.exists()
    if level == "act":
        # The same root still writes a genuine user file outside the repo.
        assert "OK: wrote" in reg.execute(
            "write_file", {"root": "user_files", "path": "notes.txt", "content": "kept\n"})


def test_full_may_still_write_a_repo_path_through_user_files(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro")
    monkeypatch.setenv("OUROBOROS_USER_FILES_ROOT", str(tmp_path))
    reg = _registry(tmp_path, _wake_task("full")["metadata"])
    assert "LIGHT_MODE_BLOCKED" not in reg.execute(
        "write_file", {"root": "user_files", "path": "repo/x.py", "content": "print(1)\n"})
    assert (tmp_path / "repo" / "x.py").read_text(encoding="utf-8") == "print(1)\n"


def test_act_keeps_the_light_positive_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "pro")
    reg = _registry(tmp_path, _wake_task("act")["metadata"])
    for root in ("task_drive", "artifact_store"):
        result = reg.execute("write_file", {"root": root, "path": "notes.txt", "content": "kept\n"})
        assert "BLOCKED" not in result, (root, result[:300])
    assert "BLOCKED" not in reg.execute("read_file", {"path": "README.md"})


# --- the wake through the real lane --------------------------------------------


def test_the_lane_attaches_the_level_to_the_wake_contract(monkeypatch, tmp_path):
    import queue
    import threading

    from ouroboros import agent as agent_module
    from supervisor import workers
    from tests.test_consciousness_wake_lane import _lane, _wait_for

    _lane(monkeypatch, tmp_path, event_q=queue.Queue())
    seen: list = []
    done = threading.Event()

    class Actor:
        def handle_task(self, task):
            seen.append(task)
            done.set()
            return []

    monkeypatch.setattr(agent_module, "make_agent", lambda **kw: Actor())
    receipt = workers.handle_wake_direct(1, "wake", {**WAKE_META, "consciousness_autonomy": "act"})
    assert receipt["admitted"] is True
    assert done.wait(10) and _wait_for(lambda: bool(seen))
    task = seen[0]
    assert task["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)
    assert task["metadata"]["runtime_mode_cap"] == "light"


# --- ISSUER: a wake speaks as a task -------------------------------------------


def test_routing_issuer_keeps_wake_relays_task_authored_and_explicit_owner_ingress(tmp_path):
    from ouroboros.tools.control_routing import ISSUER_OWNER_TURN, ISSUER_TASK, _routing_issuer

    wake = types.SimpleNamespace(task_id="wake-1", is_direct_chat=True, last_owner_delivery=None,
                                 task_metadata=dict(_wake_task("act")["metadata"]))
    assert _routing_issuer(wake) == {"kind": ISSUER_TASK, "task_id": "wake-1", "root_task_id": "wake-1"}
    # An owner turn is the direct turn the owner door stamped; a bare direct context is not one.
    owner = types.SimpleNamespace(task_id="turn-1", is_direct_chat=True, last_owner_delivery=None,
                                  task_metadata={"origin_message_ref": {"chat_id": 1, "client_message_id": "cm-1"}})
    assert _routing_issuer(owner) == {"kind": ISSUER_OWNER_TURN}
    bare = types.SimpleNamespace(task_id="turn-2", is_direct_chat=True, last_owner_delivery=None, task_metadata={})
    assert _routing_issuer(bare) == {"kind": ISSUER_TASK, "task_id": "turn-2", "root_task_id": "turn-2"}
    # Draining real owner dialogue provides receipt identity, never authorship.
    relaying = types.SimpleNamespace(task_id="c-root", is_direct_chat=False,
                                     last_owner_delivery={"client_message_id": "cm-9", "text": "go"},
                                     task_metadata={"initiator": "consciousness"})
    assert _routing_issuer(relaying) == {"kind": ISSUER_TASK, "task_id": "c-root", "root_task_id": "c-root"}
    # A client id is not the door's stamp: a wake (or a Presence event, whose client id is the
    # provider's event id) keeps speaking as a task.
    client_id_only = types.SimpleNamespace(task_id="c-root", is_direct_chat=True, last_owner_delivery=None,
                                           task_metadata={"initiator": "consciousness", "client_message_id": "cm-2"})
    assert _routing_issuer(client_id_only) == {"kind": ISSUER_TASK, "task_id": "c-root", "root_task_id": "c-root"}


def test_steer_from_a_wake_is_written_as_an_independent_task_message(tmp_path, monkeypatch):
    from ouroboros.tools import control_routing

    sent: list = []
    monkeypatch.setattr(control_routing, "_send_task_message",
                        lambda ctx, issuer, target, msg, chat_id: sent.append((issuer, target, msg)) or "WRITTEN")
    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
        task_id="wake-1", is_direct_chat=True, last_owner_delivery=None,
        task_metadata=dict(_wake_task("act")["metadata"]),
    )
    assert control_routing._steer_task(ctx, task_id="r-1", message="please also check X") == "WRITTEN"
    assert sent == [({"kind": "task", "task_id": "wake-1", "root_task_id": "wake-1"}, "r-1", "please also check X")]
    assert ctx.pending_events == []


# --- origin inheritance: promote / followup / subagent -------------------------


@pytest.fixture
def _promote_root(tmp_path, monkeypatch):
    """The real promote admission path (tool -> supervisor handler -> worker_promotion)."""
    import ouroboros.config as cfg
    import supervisor.message_bus as mb
    import supervisor.queue as queue_mod
    from supervisor import workers

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue_mod, "DRIVE_ROOT", str(tmp_path))
    monkeypatch.setattr(queue_mod, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(mb, "get_bridge", lambda: types.SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr(workers, "_announce_created_project", lambda *a, **kw: None)
    # Origin admission is under test, not the asynchronous Git/toolchain scan.
    monkeypatch.setattr("ouroboros.workspace_admission.bounded_workspace_preflight",
                        lambda root: {"schema_version": 1, "workspace_root": str(root)})
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_promote_from_a_wake_mints_a_consciousness_root_through_the_real_admission(_promote_root):
    from ouroboros.tools.control_routing import _promote_chat_to_task
    from ouroboros.utils import append_jsonl
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    tmp_path = _promote_root
    enqueued: list = []
    captured: dict = {}
    supervisor = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], WORKERS={0: types.SimpleNamespace()},
        bridge=types.SimpleNamespace(send_routing_ack=lambda *a, **k: None, broadcast=lambda *a, **k: None),
        enqueue_task=lambda task: enqueued.append(task) or dict(task),
        persist_queue_snapshot=lambda **_k: True, load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=append_jsonl,
    )
    ctx = types.SimpleNamespace(
        pending_events=[], current_chat_id=1, drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_id="wake-1", is_direct_chat=True, last_owner_delivery=None, project_id="",
        task_metadata=dict(_wake_task("act")["metadata"]), task_contract={},
        event_queue=types.SimpleNamespace(
            put_nowait=lambda event: (captured.update(event), _handle_promote_chat_to_task(event, supervisor))),
    )
    out = _promote_chat_to_task(ctx, "audit the logs", workspace="none", predecessor_task_id="")
    assert out.startswith("OK: task"), out
    # The event carries the origin by value and nothing the wake asked for is stripped.
    assert captured["initiator"] == "consciousness" and captured["consciousness_autonomy"] == "act"
    assert captured["usage_category"] == "consciousness_task" and captured["workspace"] == "none"
    assert "presence" not in captured
    [root] = enqueued
    assert root["actor_id"] == "consciousness" and root["delegation_role"] == "root"
    assert root["metadata"]["initiator"] == "consciousness"
    assert root["metadata"]["usage_category"] == "consciousness_task"
    assert root["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)
    assert root["metadata"]["runtime_mode_cap"] == "light" and "_presence_origin" not in root


def test_route_to_project_from_a_wake_mints_a_consciousness_root_too(_promote_root):
    """P3d (live stand): the wake's route DID create a root in project redline-exodus, but
    with empty metadata — no origin, no ledger category, no level, no withheld tools — because
    only ``promote_chat_to_task`` stamped the origin onto its event. A route mints a root
    through the SAME admission door, so it carries the same origin by value."""
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control_routing import _route_to_project
    from ouroboros.utils import append_jsonl
    from supervisor.events_project_routing import _handle_promote_chat_to_task

    tmp_path = _promote_root
    create_project(tmp_path, "racer", name="Racer")
    enqueued: list = []
    captured: dict = {}
    supervisor = types.SimpleNamespace(
        DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], WORKERS={0: types.SimpleNamespace()},
        bridge=types.SimpleNamespace(send_routing_ack=lambda *a, **k: None, broadcast=lambda *a, **k: None),
        enqueue_task=lambda task: enqueued.append(task) or dict(task),
        persist_queue_snapshot=lambda **_k: True, load_state=lambda: {"owner_chat_id": 1},
        append_jsonl=append_jsonl,
    )
    ctx = types.SimpleNamespace(
        pending_events=[], current_chat_id=1, drive_root=tmp_path, budget_drive_root=str(tmp_path),
        task_id="wake-1", is_direct_chat=True, last_owner_delivery=None, project_id="",
        task_metadata=dict(_wake_task("act")["metadata"]), task_contract={},
        event_queue=types.SimpleNamespace(
            put_nowait=lambda event: (captured.update(event), _handle_promote_chat_to_task(event, supervisor))),
    )
    out = _route_to_project(ctx, "racer", "audit the logs", predecessor_task_id="")
    assert "Routed to project" in out, out
    assert captured["initiator"] == "consciousness"
    assert captured["usage_category"] == "consciousness_task"
    assert captured["consciousness_autonomy"] == "act"
    [root] = enqueued
    assert root["actor_id"] == "consciousness" and root["project_id"] == "racer"
    assert root["metadata"]["initiator"] == "consciousness"
    assert root["metadata"]["usage_category"] == "consciousness_task"
    assert root["metadata"]["runtime_mode_cap"] == "light"
    assert root["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)


def test_route_to_project_from_an_owner_turn_stays_unstamped(_promote_root):
    """The origin is inherited, never minted: an owner's own route keeps the plain root."""
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control_routing import _route_to_project

    tmp_path = _promote_root
    create_project(tmp_path, "racer", name="Racer")
    ctx = types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
        task_id="owner-turn", is_direct_chat=True, project_id="", task_contract={},
        task_metadata={"client_message_id": "cm-1"},
    )
    _route_to_project(ctx, "racer", "audit the logs", predecessor_task_id="")
    [evt] = ctx.pending_events
    assert "initiator" not in evt and "usage_category" not in evt


def test_promoted_root_is_stamped_and_its_contract_derives_the_level(tmp_path, monkeypatch):
    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    enqueued: list = []

    def enqueue(task):
        enqueued.append(task)
        return dict(task)

    ctx = types.SimpleNamespace(enqueue_task=enqueue, persist_queue_snapshot=lambda **_k: True,
                                load_state=lambda: {"owner_chat_id": 1})
    evt = {"type": "promote_chat_to_task", "task_id": "c0000001", "objective": "audit the logs",
           "chat_id": 1, "workspace": "none", "initiator": "consciousness",
           "usage_category": "consciousness_task", "consciousness_autonomy": "act"}
    assert workers.promote_chat_to_task(evt, ctx)["status"] == "scheduled"
    task = enqueued[0]
    assert task["actor_id"] == "consciousness" and task["delegation_role"] == "root"
    assert task["metadata"]["initiator"] == "consciousness"
    assert task["metadata"]["usage_category"] == "consciousness_task"
    assert task["metadata"]["consciousness_autonomy"] == "act"
    assert task["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)
    assert task["metadata"]["runtime_mode_cap"] == "light"
    assert task["source"] == "promote_chat_to_task" and "_presence_origin" not in task


def test_followup_template_inherits_the_origin_and_admission_derives_the_level(tmp_path, monkeypatch):
    from supervisor import queue
    from tests.test_schedule_followup import _ctx, _followup

    ctx = _ctx(tmp_path)
    ctx.task_metadata.update(_wake_task("act")["metadata"])
    assert _followup(ctx).startswith("FOLLOWUP_SCHEDULED")
    record = queue.list_scheduled_tasks(tmp_path / "data")["tasks"][0]
    meta = record["task"]["metadata"]
    assert meta["initiator"] == "consciousness" and meta["usage_category"] == "consciousness_task"
    assert meta["consciousness_autonomy"] == "act" and "task_contract" not in record["task"]
    monkeypatch.setattr(queue, "load_state", lambda: {"owner_chat_id": 1})
    task = queue._task_from_schedule(record)
    assert task["delegation_role"] == "root" and task["metadata"]["initiator"] == "consciousness"
    assert task["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)
    assert task["metadata"]["runtime_mode_cap"] == "light"


def test_owner_followup_template_carries_no_origin(tmp_path):
    from supervisor import queue
    from tests.test_schedule_followup import _ctx, _followup

    assert _followup(_ctx(tmp_path)).startswith("FOLLOWUP_SCHEDULED")
    meta = queue.list_scheduled_tasks(tmp_path / "data")["tasks"][0]["task"]["metadata"]
    assert "initiator" not in meta and "consciousness_autonomy" not in meta


def test_subagent_payload_lands_the_origin_on_the_child_metadata():
    from supervisor.task_dispatch import build_scheduled_task_payload

    fields = {"tid": "kid1", "chat_id": 1, "text": "x", "desc": "x", "role": "researcher",
              "root_task_id": "wake-1", "delegation_role": "subagent", "actor_id": "subagent:researcher",
              "origin_metadata": ca.consciousness_origin_metadata(_wake_task("act")["metadata"])}
    task = build_scheduled_task_payload(fields)
    assert task["metadata"]["initiator"] == "consciousness"
    assert task["metadata"]["usage_category"] == "consciousness_task"
    assert task["metadata"]["consciousness_autonomy"] == "act"
    plain = build_scheduled_task_payload({**fields, "origin_metadata": {}})
    assert "initiator" not in plain["metadata"]


def test_schedule_subagent_event_names_the_origin():
    """The tool stamps ``origin_metadata`` on the schedule event beside the envelope:
    the consciousness origin and the owner's words that caused the tree, by value."""
    source = pathlib.Path("ouroboros/tools/control_scheduling.py").read_text(encoding="utf-8")
    assert '"origin_metadata": {**consciousness_origin_metadata(metadata), **owner_origin},' in source
    handler = pathlib.Path("supervisor/events_schedule_task.py").read_text(encoding="utf-8")
    assert '"origin_metadata": evt.get("origin_metadata"),' in handler


# --- evolution: eligibility, sticky owner stop, campaign provenance ------------


def test_post_task_promotion_is_refused_when_toggle_evolution_is_withheld():
    from ouroboros.post_task_evolution import _eligible

    assert _eligible({"type": "task"}) is True
    assert _eligible({"type": "task", "task_contract": {"disabled_tools": ["toggle_evolution"]}}) is False
    assert _eligible({"type": "task", "metadata": {"disabled_tools": ["toggle_evolution"]}}) is False
    from ouroboros.contracts.task_contract import attach_task_contract

    assert _eligible(attach_task_contract(_wake_task("act"))) is False
    assert _eligible(attach_task_contract(_wake_task("full"))) is True


def test_globalized_promotion_view_keeps_the_contract(tmp_path, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.contracts.task_contract import attach_task_contract

    seen: list = []
    monkeypatch.setattr(pipeline, "_update_improvement_backlog", lambda env, entry: None)
    monkeypatch.setattr("ouroboros.post_task_evolution.maybe_promote",
                        lambda env, task, entry, llm: seen.append(task))
    task = attach_task_contract({**_wake_task("act"), "project_id": "lab"})
    pipeline._run_global_backlog_promotion_only(
        types.SimpleNamespace(drive_root=tmp_path), task,
        {"backlog_candidates": [{"summary": "tidy the logs"}]}, None,
    )
    assert seen and seen[0]["task_contract"]["disabled_tools"] == list(ca.ACT_DISABLED)


def test_request_file_and_pending_apply_carry_the_origin(tmp_path, monkeypatch):
    from ouroboros import post_task_evolution as pte

    task = _wake_task("full")
    pte._write_request(tmp_path, {"objective": "improve X", "requires_plan_review": False}, task)
    req = json.loads((tmp_path / pte._REQUEST_REL).read_text(encoding="utf-8"))
    assert req["initiator"] == "consciousness" and req["consciousness_autonomy"] == "full"
    assert req["usage_category"] == "consciousness_task"
    calls: list = []
    monkeypatch.setattr("ouroboros.config.get_post_task_evolution_enabled", lambda: True)
    monkeypatch.setattr("supervisor.evolution_lifecycle.evolution_block_reason", lambda: "")
    monkeypatch.setattr("supervisor.evolution_lifecycle.start_evolution_campaign",
                        lambda objective, source="", **kw: calls.append((objective, source, kw)) or {"id": "c1"})
    monkeypatch.setattr("supervisor.state.load_state", lambda: {
        "owner_chat_id": 7, "evolution_owner_stopped": False,
        "evolution_mode_enabled": False,
    })

    def _update_state(mutator):
        live = {"evolution_owner_stopped": False, "evolution_mode_enabled": False}
        mutator(live)
        return live

    monkeypatch.setattr("supervisor.state.update_state", _update_state)
    monkeypatch.setattr("ouroboros.config.get_post_task_evolution_budget_usd", lambda: 0.0)
    assert pte.apply_pending_request(tmp_path) is True
    assert calls == [("improve X", "post_task", {"origin": {
        "initiator": "consciousness", "usage_category": "consciousness_task", "consciousness_autonomy": "full"}})]


def _toggle_ctx(state, sent):
    return types.SimpleNamespace(load_state=state.load_state,
                                 send_with_budget=lambda cid, text, **kw: sent.append(text))


def test_agent_tool_enable_is_refused_while_the_owner_stop_stands(tmp_path, monkeypatch):
    """В12: /evolve off is sticky against toggle_evolution — the typed refusal, no campaign."""
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(owner_chat_id=7, evolution_owner_stopped=True))
    started: list = []
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(el, "start_evolution_campaign",
                        lambda objective, source="", **kw: started.append(source) or {"status": "active"})
    sent: list = []
    events_mod._handle_toggle_evolution({"enabled": True, "objective": "x"}, _toggle_ctx(state, sent))
    assert started == []
    assert bool(state.load_state().get("evolution_owner_stopped")) is True
    assert not state.load_state().get("evolution_mode_enabled")
    assert sent and "stayed OFF" in sent[0] and "sticky" in sent[0]


def test_agent_tool_enable_without_an_owner_stop_starts_a_campaign_with_the_origin(tmp_path, monkeypatch):
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    state.update_state(lambda live: live.update(owner_chat_id=7, evolution_owner_stopped=False))
    started: list = []
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(el, "start_evolution_campaign",
                        lambda objective, source="", **kw: started.append((source, kw)) or {"status": "active"})
    sent: list = []
    evt = {"enabled": True, "objective": "x", "initiator": "consciousness",
           "usage_category": "consciousness_task", "consciousness_autonomy": "full"}
    events_mod._handle_toggle_evolution(evt, _toggle_ctx(state, sent))
    assert started == [("agent_tool", {"origin": {
        "initiator": "consciousness", "usage_category": "consciousness_task", "consciousness_autonomy": "full"}})]
    live = state.load_state()
    assert live["evolution_mode_enabled"] is True and live["evolution_owner_stopped"] is False


def test_a_stop_the_agent_placed_itself_stays_undoable_by_the_agent(tmp_path, monkeypatch):
    """В12 binds the OWNER's stop; the agent's own toggle_evolution(False) is not an owner
    stop — it still blocks the post-task re-arm (the flag), but the agent may re-enable."""
    import supervisor.state as state
    from supervisor import events as events_mod
    from supervisor import evolution_lifecycle as el

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    # What the agent's own toggle_evolution(False) leaves behind (the disable path itself needs
    # the live supervisor; its state write is pinned in test_evolution_stop_and_cost).
    state.update_state(lambda live: live.update(owner_chat_id=7, evolution_owner_stopped=True,
                                                evolution_stop_source="agent_tool"))
    started: list = []
    monkeypatch.setattr(el, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(el, "start_evolution_campaign",
                        lambda objective, source="", **kw: started.append(source) or {"status": "active"})
    sent: list = []
    events_mod._handle_toggle_evolution({"enabled": True, "objective": "again"}, _toggle_ctx(state, sent))
    live = state.load_state()
    assert started == ["agent_tool"] and live["evolution_owner_stopped"] is False
    assert "evolution_stop_source" not in live and not [t for t in sent if "sticky" in t]
    # The owner's stop (/evolve off, panic, an owner-sourced toggle: no agent_tool source) stays sticky.
    state.update_state(lambda live: live.update(evolution_owner_stopped=True, evolution_stop_source="owner_chat"))
    events_mod._handle_toggle_evolution({"enabled": True, "objective": "x"}, _toggle_ctx(state, sent))
    assert started == ["agent_tool"] and sent and "sticky" in sent[-1]
    state.update_state(lambda live: (live.update(evolution_owner_stopped=True), live.pop("evolution_stop_source", None)))
    events_mod._handle_toggle_evolution({"enabled": True, "objective": "y"}, _toggle_ctx(state, sent))
    assert started == ["agent_tool"] and len([t for t in sent if "sticky" in t]) == 2


def test_observe_does_without_the_work_starting_review_verb():
    """`request_deep_self_review` enqueues a ROOT: Observe starts nothing (В10')."""
    assert "request_deep_self_review" in ca.OBSERVE_DISABLED and "request_deep_self_review" not in ca.ACT_DISABLED
    # The GitHub write verbs change the world beyond the repository (PLAN §5.4: Observe keeps the reads only).
    for verb in ("create_github_issue", "comment_on_issue", "comment_on_pr", "close_github_issue"):
        assert verb in ca.OBSERVE_DISABLED and verb not in ca.ACT_DISABLED, verb
    for verb in ("list_github_prs", "get_github_pr", "get_github_checks", "list_github_issues", "get_github_issue"):
        assert verb not in ca.OBSERVE_DISABLED, verb
    # The built-in execution verb (astra scope round 5): a skill's script.
    assert "skill_exec" in ca.OBSERVE_DISABLED and "skill_exec" not in ca.ACT_DISABLED


def test_deep_review_request_carries_the_origin_to_the_one_door(tmp_path, monkeypatch):
    """The tool's event names the caller's origin, the handler hands it to the queue, and the
    queued root carries it — so the admission door and the ledger see the tree (В11/В18)."""
    from ouroboros.tools.control_runtime import _request_deep_self_review
    from supervisor import queue, state
    from supervisor.events_runtime_controls import _handle_deep_self_review_request

    monkeypatch.setattr("ouroboros.deep_self_review.deep_review_route", lambda: ("", "reviewer-x"))
    ctx = types.SimpleNamespace(pending_events=[], task_metadata=dict(_wake_task("act")["metadata"]))
    assert _request_deep_self_review(ctx, "look again").startswith("Deep self-review requested")
    evt = ctx.pending_events[0]
    assert evt["type"] == "deep_self_review_request" and evt["initiator"] == "consciousness"
    owner = types.SimpleNamespace(pending_events=[], task_metadata={"client_message_id": "cm"})
    _request_deep_self_review(owner, "look")
    assert "initiator" not in owner.pending_events[0]

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    state.update_state(lambda live: live.update(owner_chat_id=1))
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("supervisor.workers._worker_pool_execution_state",
                        lambda: {"available": True, "disabled_reason": ""})
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window",
                        lambda root, now=None: {"status": "available", "limit_usd": 20.0, "accounted_usd": 0.0,
                                                "remaining_usd": 20.0, "unknown_unmetered": 0, "resets_at": ""})
    handed: list = []
    sup = types.SimpleNamespace(queue_deep_self_review_task=lambda **kw: handed.append(kw))
    _handle_deep_self_review_request(evt, sup)
    assert handed[0]["origin"] == {"initiator": "consciousness", "usage_category": "consciousness_task",
                                   "consciousness_autonomy": "act"}
    assert queue.queue_deep_self_review_task("look again", model="reviewer-x", origin=handed[0]["origin"])
    assert pending[0]["type"] == "deep_self_review" and pending[0]["metadata"]["initiator"] == "consciousness"
    assert pending[0]["metadata"]["usage_category"] == "consciousness_task"
    # The owner's own request stays unmarked.
    assert queue.queue_deep_self_review_task("mine", model="reviewer-x", force=True)
    assert "metadata" not in pending[1] or "initiator" not in pending[1]["metadata"]


def test_the_allowance_is_read_before_the_queue_lock(tmp_path, monkeypatch):
    """The ledger read takes the cross-process ledger lock; it must not run under the queue lock,
    or a contended ledger stalls every queue reader."""
    from supervisor import queue, state

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    seen: list = []

    def _window(root, now=None):
        seen.append(queue._queue_lock._is_owned())
        return {"status": "available", "limit_usd": 20.0, "accounted_usd": 0.0, "remaining_usd": 20.0,
                "unknown_unmetered": 0, "resets_at": ""}

    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window", _window)
    task = {"id": "root-1", "type": "task", "chat_id": 1, "text": "x", "metadata": dict(_wake_task("act")["metadata"])}
    assert not queue.enqueue_task(task).get("_admission_blocked")
    assert seen == [False]


def test_toggle_tool_stamps_the_turn_origin_on_its_event(monkeypatch):
    from ouroboros.tools.control_runtime import _toggle_evolution

    monkeypatch.setattr("supervisor.evolution_lifecycle.evolution_block_reason", lambda: "")
    ctx = types.SimpleNamespace(pending_events=[], task_metadata=dict(_wake_task("full")["metadata"]))
    assert _toggle_evolution(ctx, True, "improve X").startswith("OK")
    evt = ctx.pending_events[0]
    assert evt["type"] == "toggle_evolution" and evt["initiator"] == "consciousness"
    assert evt["consciousness_autonomy"] == "full"
    owner = types.SimpleNamespace(pending_events=[], task_metadata={"client_message_id": "cm"})
    _toggle_evolution(owner, True, "improve X")
    assert "initiator" not in owner.pending_events[0]


def test_campaign_keeps_the_origin_and_its_cycle_tasks_inherit_it(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    origin = {"initiator": "consciousness", "usage_category": "consciousness_task", "consciousness_autonomy": "full"}
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="agent_tool", origin=origin)
    assert campaign["initiator"] == "consciousness" and campaign["consciousness_autonomy"] == "full"
    # An agent-sourced resume keeps the recorded origin; the owner's explicit resume of a PAUSED
    # campaign ADOPTS it (see test_the_owners_start_adopts_a_paused_consciousness_campaign).
    campaign["status"] = "paused"
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True
    resumed = evolution_lifecycle.start_evolution_campaign("", source="agent_tool")
    assert resumed["initiator"] == "consciousness"
    state.update_state(lambda live: live.update(owner_chat_id=1, evolution_mode_enabled=True,
                                                evolution_owner_stopped=False))
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setattr(queue, "send_with_budget", lambda *a, **k: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window",
                        lambda root, now=None: {"status": "available", "limit_usd": 20.0, "accounted_usd": 0.0,
                                                "remaining_usd": 20.0, "unknown_unmetered": 0, "resets_at": ""})
    queue.enqueue_evolution_task_if_needed()
    assert len(pending) == 1
    task = pending[0]
    assert task["type"] == "evolution" and task["metadata"]["initiator"] == "consciousness"
    assert task["metadata"]["usage_category"] == "consciousness_task"
    assert task["task_contract"]["disabled_tools"] == [] and task["metadata"]["runtime_mode_cap"] == ""
    assert state.load_state()["evolution_cycle"] == 1
    # The ONE door refuses the next cycle (the tree's allowance is spent): the campaign is
    # paused ONCE with an owner line — no cycle bump, no transaction minted on every pass.
    pending.clear()
    sent: list = []
    monkeypatch.setattr(queue, "send_with_budget", lambda cid, text, **kw: sent.append(text))
    monkeypatch.setattr("ouroboros.consciousness_allowance.allowance_window",
                        lambda root, now=None: {"status": "exhausted", "limit_usd": 20.0, "accounted_usd": 21.0,
                                                "remaining_usd": 0.0, "unknown_unmetered": 0, "resets_at": "2027-01-01T00:00:00+00:00"})
    queue.enqueue_evolution_task_if_needed()
    queue.enqueue_evolution_task_if_needed()
    assert pending == [] and state.load_state()["evolution_cycle"] == 1
    paused = evolution_lifecycle._read_evolution_campaign()
    assert paused["status"] == "paused" and paused["pause_reason"] == "admission_refused:consciousness_allowance_exhausted"
    assert len(sent) == 1 and "Evolution paused" in sent[0] and "consciousness_allowance_exhausted" in sent[0]
    assert not state.load_state().get("evolution_mode_enabled")


def test_a_transient_refusal_never_pauses_the_campaign(tmp_path, monkeypatch):
    """Only a consciousness refusal (allowance, concurrency) pauses; anything else clears itself
    and the next pass retries — without recording a cycle (opus round 3)."""
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    pending: list = []
    queue.init_queue_refs(pending, {}, {"value": 0})
    monkeypatch.setattr(state, "TOTAL_BUDGET_LIMIT", 0.0)
    evolution_lifecycle.start_evolution_campaign("Improve", source="owner_chat")
    state.update_state(lambda live: live.update(owner_chat_id=1, evolution_mode_enabled=True,
                                                evolution_owner_stopped=False))
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    sent: list = []
    monkeypatch.setattr(queue, "send_with_budget", lambda cid, text, **kw: sent.append(text))
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(queue, "enqueue_task", lambda task, **kw: {**task, "_admission_blocked": "duplicate_task_id"})
    queue.enqueue_evolution_task_if_needed()
    assert pending == [] and sent == []
    assert evolution_lifecycle._read_evolution_campaign()["status"] == "active"
    live = state.load_state()
    assert live.get("evolution_mode_enabled") is True and not live.get("evolution_cycle")


def test_the_owners_start_adopts_a_paused_consciousness_campaign(tmp_path):
    """BIBLE P0: the owner's explicit /evolve start on a campaign the consciousness allowance paused
    makes it the owner's work — otherwise the door refuses and re-pauses it until the window frees."""
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    origin = {"initiator": "consciousness", "usage_category": "consciousness_task", "consciousness_autonomy": "full"}
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="agent_tool", origin=origin)
    campaign["status"], campaign["pause_reason"] = "paused", "admission_refused:consciousness_allowance_exhausted"
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True
    adopted = evolution_lifecycle.start_evolution_campaign("", source="owner_chat")
    assert adopted["status"] == "active" and adopted["adopted_by_owner_at"] and "pause_reason" not in adopted
    assert not any(key in adopted for key in origin)
    assert ca.consciousness_origin_metadata(adopted) == {}
    # The function's own default source is the owner too (grok round 4).
    campaign = evolution_lifecycle.start_evolution_campaign("Again", source="agent_tool", origin=origin)
    campaign["status"] = "paused"
    assert evolution_lifecycle._write_evolution_campaign(campaign) is True
    assert "initiator" not in evolution_lifecycle.start_evolution_campaign("")


def test_owner_campaign_carries_no_origin(tmp_path):
    from supervisor import evolution_lifecycle, queue, state

    state.init(tmp_path)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    queue.init(tmp_path)
    campaign = evolution_lifecycle.start_evolution_campaign("Improve", source="owner_chat")
    assert "initiator" not in campaign
    assert ca.consciousness_origin_metadata(campaign) == {}


# --- what a wake may address: the host routing manifest (P3c) -------------------


def _wake_routing_ctx(tmp_path, **metadata):
    """A wake's tool context: the P3 envelope plus whatever Main-lane routing facts the
    alarm clock merged into it (``main_lane_routing_metadata``)."""
    return types.SimpleNamespace(
        pending_events=[], event_queue=None, current_chat_id=1, drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_id="wake-1", is_direct_chat=True,
        last_owner_delivery=None, project_id="", task_contract={},
        task_metadata={**_wake_task("act")["metadata"], **metadata},
    )


def _addressable_result(tmp_path, task_id="racer-old"):
    """One settled root on disk, exactly as the Main manifest would preview it."""
    from ouroboros.server_routing_context import _task_result_ground_truth

    row = {"task_id": task_id, "status": "completed", "project_id": "racer",
           "title": "Racer prototype", "objective": "Build the racer prototype",
           "task_contract": {"objective": "Build the racer prototype", "context": "exact old context"}}
    results = tmp_path / "task_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / f"{task_id}.json").write_text(json.dumps({"_schema_version": 1, **row}), encoding="utf-8")
    return _task_result_ground_truth(row)


def test_a_wake_continues_a_result_its_routing_manifest_makes_addressable(tmp_path):
    """The first live wake chose a predecessor and got AUTHORITY_SOURCE_UNAVAILABLE twice,
    so the work it had decided on never started: its metadata carried no routing manifest
    (P3c). With the Main lane's own facts the named id is addressable, on both verbs."""
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control_routing import _promote_chat_to_task, _route_to_project

    create_project(tmp_path, "racer", name="Racer")
    preview = _addressable_result(tmp_path)
    facts = {"main_routing_manifest": {"final_results": [preview]}}

    routed = _wake_routing_ctx(tmp_path, **facts)
    out = _route_to_project(routed, "racer", "Continue the racer", predecessor_task_id="racer-old")
    assert out.startswith("⚠️ ROUTE_UNCONFIRMED"), out
    [route_evt] = routed.pending_events
    assert route_evt["predecessor_task_id"] == "racer-old"
    assert route_evt["predecessor_authority_source"] == preview["authority_source"]

    promoted = _wake_routing_ctx(tmp_path, **facts)
    _promote_chat_to_task(promoted, "Finish the racer", workspace="none", predecessor_task_id="racer-old")
    [promote_evt] = promoted.pending_events
    assert promote_evt["predecessor_task_id"] == "racer-old"
    assert promote_evt["predecessor_authority_source"] == preview["authority_source"]
    # The manifest never dilutes the wake's own origin.
    assert promote_evt["initiator"] == "consciousness"


def test_a_wake_starts_fresh_work_with_no_predecessor_and_needs_no_manifest(tmp_path):
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control_routing import _promote_chat_to_task, _route_to_project

    create_project(tmp_path, "racer", name="Racer")

    routed = _wake_routing_ctx(tmp_path)
    assert _route_to_project(routed, "racer", "Start a separate experiment",
                             predecessor_task_id="").startswith("⚠️ ROUTE_UNCONFIRMED")
    assert "predecessor_authority_source" not in routed.pending_events[0]

    promoted = _wake_routing_ctx(tmp_path)
    _promote_chat_to_task(promoted, "audit the logs", workspace="none", predecessor_task_id="")
    assert "predecessor_task_id" not in promoted.pending_events[0]


def test_a_wake_without_the_manifest_continues_a_settled_root_and_still_refuses_a_live_one(tmp_path):
    """The door judges the root, not the facts a wake was handed: with no manifest at all
    a wake continues a settled root on both verbs (the pointer is rebuilt from the durable
    result and equals the one the manifest would have shown), while a live root keeps its
    typed refusal toward steer_task and emits nothing."""
    from ouroboros.projects_registry import create_project
    from ouroboros.tools.control_routing import _promote_chat_to_task, _route_to_project

    create_project(tmp_path, "racer", name="Racer")
    preview = _addressable_result(tmp_path)

    routed = _wake_routing_ctx(tmp_path)
    out = _route_to_project(routed, "racer", "Continue the racer", predecessor_task_id="racer-old")
    assert out.startswith("⚠️ ROUTE_UNCONFIRMED"), out
    [route_evt] = routed.pending_events
    assert route_evt["predecessor_authority_source"] == preview["authority_source"]

    promoted = _wake_routing_ctx(tmp_path)
    _promote_chat_to_task(promoted, "Finish the racer", workspace="none", predecessor_task_id="racer-old")
    [promote_evt] = promoted.pending_events
    assert promote_evt["predecessor_task_id"] == "racer-old"
    assert promote_evt["initiator"] == "consciousness"

    (tmp_path / "task_results" / "racer-live.json").write_text(json.dumps({
        "_schema_version": 1, "task_id": "racer-live", "status": "running", "project_id": "racer",
    }), encoding="utf-8")
    refused = _wake_routing_ctx(tmp_path)
    out = _route_to_project(refused, "racer", "Continue the racer", predecessor_task_id="racer-live")
    assert out.startswith("⚠️ AUTHORITY_SOURCE_UNAVAILABLE (route_to_project)") and "steer_task" in out
    assert refused.pending_events == []
