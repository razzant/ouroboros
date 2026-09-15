"""v6.37.0 guard (C4.1): the in-task ensure_project_scope affordance — create/attach
a named Ouroboros project and scope the CURRENT running task into it, instead of the
cyber-racing fallback (bare `mkdir ~/Desktop`). Idempotent for the same project,
refuses to re-scope to a different one, rejects subagents."""

import logging
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _hermetic_bindings(tmp_path, monkeypatch):
    """R13: the guard reads the durable bindings at the canonical data dir, so no
    test in this module may reach the live root."""
    import ouroboros.config as cfg

    monkeypatch.setattr(cfg, "DATA_DIR", tmp_path)
    return tmp_path


def _ctx(**kw):
    base = dict(project_id="", task_metadata={}, task_contract={}, task_id="t1", event_queue=None,
                pending_events=[], drive_root=_hermetic_root())
    base.update(kw)
    return SimpleNamespace(**base)


def _hermetic_root():
    import ouroboros.config as cfg

    return cfg.DATA_DIR


def _live_ctx(supervisor, **kw):
    """The tool wired to the REAL supervisor handler, so its receipt wait reads
    exactly what the handler persisted (the rail contract)."""
    from supervisor.events_project_routing import _handle_ensure_project_scope

    ctx = _ctx(**kw)
    ctx.event_queue = SimpleNamespace(put_nowait=lambda event: _handle_ensure_project_scope(event, supervisor))
    return ctx


def _supervisor(tmp_path, running=None):
    from ouroboros.utils import append_jsonl

    return SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=dict(running or {}), PENDING=[], append_jsonl=append_jsonl)


def test_scope_call_emits_the_event_and_reports_unconfirmed_without_a_supervisor():
    """The deferred transport (no live supervisor) means NO processing, which is
    not success: the tool used to say "OK: created/attached" here before any
    bind existed. The in-memory scope moves so journal writes target the
    project meanwhile; the durable outcome is what the result reports."""
    from ouroboros.tools.control import _ensure_project_scope
    from ouroboros.project_facts import project_id_from_display_name

    ctx = _ctx()
    out = _ensure_project_scope(ctx, project_name="Cyber Racing")
    assert out.startswith("⚠️ SCOPE_UNCONFIRMED")
    assert "durably bound to no project" in out and "OK" not in out
    expected_pid = project_id_from_display_name("Cyber Racing")
    # the rest of THIS task is scoped immediately (journal/knowledge work now)
    assert ctx.project_id == expected_pid
    # a durable ensure_project_scope event is emitted for the supervisor, on the rail
    evs = [e for e in ctx.pending_events if e.get("type") == "ensure_project_scope"]
    assert len(evs) == 1
    assert evs[0]["task_id"] == "t1"
    assert evs[0]["project_id"] == expected_pid
    assert evs[0]["project_name"] == "Cyber Racing"
    assert evs[0]["client_message_id"] == f"agent-steer:{evs[0]['routing_token']}"


def test_a_landed_bind_is_reported_from_the_durable_binding(tmp_path, monkeypatch):
    import supervisor.message_bus as mb
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.projects_registry import project_binding_for_task
    from ouroboros.tools.control import _ensure_project_scope
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr(workers, "_announce_created_project", lambda *a, **kw: None)
    pid = project_id_from_display_name("Cyber Racing")
    supervisor = _supervisor(tmp_path, running={"t1": {"task": {"id": "t1", "project_id": ""}}})

    out = _ensure_project_scope(_live_ctx(supervisor), project_name="Cyber Racing")

    assert out.startswith(f"OK: this task is now durably bound to project '{pid}'")
    assert "created it" in out and f"durably bound to project '{pid}'" in out
    assert (project_binding_for_task(tmp_path, "t1") or {}).get("project_id") == pid
    assert supervisor.RUNNING["t1"]["task"]["project_id"] == pid


def test_idempotent_same_project_and_refuses_different():
    from ouroboros.tools.control import _ensure_project_scope
    from ouroboros.project_facts import project_id_from_display_name

    pid = project_id_from_display_name("Cyber Racing")
    ctx = _ctx(project_id=pid)
    out = _ensure_project_scope(ctx, project_name="Cyber Racing")
    assert "already scoped" in out
    assert not [e for e in ctx.pending_events if e.get("type") == "ensure_project_scope"]

    ctx2 = _ctx(project_id="other-project")
    out2 = _ensure_project_scope(ctx2, project_name="Cyber Racing")
    assert "cannot be re-scoped" in out2
    assert ctx2.project_id == "other-project"  # scope NOT changed


def test_bound_task_renames_its_project_instead_of_creating_a_second(tmp_path):
    """B4=A: the durable binding is the one truth. A task already bound to a
    project that asks to be scoped to a differently named one keeps its project
    and carries the requested name to it - the empty second project that split
    token-observatory off token-atlas is exactly what this refuses.

    The event keeps the REQUESTED id: the supervisor handler reads the same
    binding and owns the rename turn. Rewriting the id here made that branch
    unreachable, so no rename ever happened while this text claimed one had."""
    from ouroboros.projects_registry import bind_task_to_project, list_projects
    from ouroboros.tools.control import _ensure_project_scope

    bind_task_to_project(tmp_path, "t1", "token-atlas", 5150, origin={"absent": "system"})
    ctx = _ctx()
    out = _ensure_project_scope(ctx, project_name="Token Observatory")

    # No supervisor processed it, so the tool claims nothing about the rename:
    # the durable binding is what it states, and the event carries the request.
    assert out.startswith("⚠️ SCOPE_UNCONFIRMED")
    assert "durably bound to project 'token-atlas'" in out
    assert ctx.project_id == "token-atlas"
    evs = [e for e in ctx.pending_events if e.get("type") == "ensure_project_scope"]
    assert len(evs) == 1
    assert evs[0]["project_id"] == "token-observatory"
    assert evs[0]["project_name"] == "Token Observatory"
    assert [p["id"] for p in list_projects(tmp_path)] == ["token-atlas"]


def test_bound_task_scope_text_claims_no_rename_when_the_name_is_unchanged(tmp_path, monkeypatch):
    """The text must be TRUE: a request that names the project it is already called
    says nothing about a rename -- read from the handler's receipt, not guessed."""
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.tools.control import _ensure_project_scope
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    # The bound project already carries the requested display name, while the id the
    # name derives to is a different one.
    create_project(tmp_path, "token-atlas", name="Token Observatory")
    bind_task_to_project(tmp_path, "t1", "token-atlas", 5150, origin={"absent": "system"})
    supervisor = _supervisor(tmp_path, running={"t1": {"task": {"id": "t1", "project_id": "token-atlas"}}})

    out = _ensure_project_scope(_live_ctx(supervisor), project_name="Token Observatory")

    assert out.startswith("OK: this task stays durably bound to project 'token-atlas'")
    assert "no second project" in out
    assert "rename" not in out


def test_bound_task_rename_reaches_the_registry_through_the_real_handler(tmp_path, monkeypatch):
    """The two halves joined: real tool -> real event -> real supervisor handler.
    Each half was green on its own while the rename never happened in production."""
    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from ouroboros.projects_registry import bind_task_to_project, create_project, get_project
    from ouroboros.tools.control import _ensure_project_scope
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    project = create_project(tmp_path, "token-atlas", name="Token Atlas")
    bind_task_to_project(tmp_path, "t1", "token-atlas", project["chat_id"], origin={"absent": "system"})
    broadcasts: list = []
    announced: list = []
    monkeypatch.setattr(mb, "get_bridge",
                        lambda: SimpleNamespace(broadcast=lambda payload: broadcasts.append(payload)))
    monkeypatch.setattr(workers, "_announce_created_project",
                        lambda *a, **kw: announced.append(True))

    supervisor = _supervisor(tmp_path, running={"t1": {"task": {"id": "t1", "project_id": "token-atlas"}}})
    out = _ensure_project_scope(_live_ctx(supervisor), project_name="Token Observatory")

    assert get_project(tmp_path, "token-atlas")["name"] == "Token Observatory"
    assert [p["id"] for p in reg.list_projects(tmp_path)] == ["token-atlas"]
    assert broadcasts == [] and announced == []
    assert supervisor.RUNNING["t1"]["task"]["project_id"] == "token-atlas"
    # The text is true because it is the handler's receipt: the rename LANDED.
    assert out.startswith("OK: this task stays durably bound to project 'token-atlas'")
    assert "the requested name 'Token Observatory' was applied to it as a rename" in out


def test_binding_outranks_a_stale_ctx_scope_and_stays_idempotent(tmp_path):
    from ouroboros.projects_registry import bind_task_to_project
    from ouroboros.tools.control import _ensure_project_scope

    bind_task_to_project(tmp_path, "t1", "token-atlas", 5150, origin={"absent": "system"})
    ctx = _ctx(project_id="stale-scope")
    out = _ensure_project_scope(ctx, project_id="token-atlas")

    assert "already scoped" in out
    assert ctx.project_id == "token-atlas"
    assert not [e for e in ctx.pending_events if e.get("type") == "ensure_project_scope"]


def test_unreadable_bindings_are_disclosed_and_do_not_block(tmp_path, caplog):
    """Proportionality: an unreadable store reads as "no binding" (the behaviour
    before this seam existed) and says so once; only a READABLE binding refuses."""
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.tools.control import _ensure_project_scope

    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "project_task_bindings.json").write_text("{ not json", encoding="utf-8")

    ctx = _ctx()
    with caplog.at_level(logging.WARNING):
        out = _ensure_project_scope(ctx, project_name="Cyber Racing")

    # Not blocked: the event went out and the scope moved; with no supervisor
    # the outcome is honestly unconfirmed and the unreadable store is disclosed.
    assert out.startswith("⚠️ SCOPE_UNCONFIRMED") and "could not be read" in out
    assert [e["type"] for e in ctx.pending_events] == ["ensure_project_scope"]
    assert ctx.project_id == project_id_from_display_name("Cyber Racing")
    assert "project_binding_unreadable" in caplog.text


def test_rejects_subagent_and_requires_an_arg():
    from ouroboros.tools.control import _ensure_project_scope

    # delegation_role lives on task_metadata / contract lineage, not a ctx attr
    out = _ensure_project_scope(_ctx(task_metadata={"delegation_role": "subagent"}), project_name="X")
    assert "subagents" in out.lower()
    out_lineage = _ensure_project_scope(
        _ctx(task_contract={"lineage": {"delegation_role": "subagent"}}), project_name="X"
    )
    assert "subagents" in out_lineage.lower()

    out2 = _ensure_project_scope(_ctx())
    assert "TOOL_ARG_ERROR" in out2


def test_supervisor_handler_creates_binds_updates_running_and_broadcasts(monkeypatch):
    """C4.1 supervisor side (review F1/F2): the handler must create_project,
    bind the task, UPDATE the RUNNING map's task project_id (so the project lease
    counts it as a lane occupant), and broadcast projects_changed."""
    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from supervisor import workers

    calls = {"create": None, "bind": None, "touch": None, "broadcast": None}
    # Hermetic (R13): the handler now reads the durable binding first, and
    # workers.DRIVE_ROOT here is the LIVE data root.
    monkeypatch.setattr(reg, "project_id_for_task", lambda dr, tid, **kw: "")
    monkeypatch.setattr(reg, "create_project", lambda dr, pid, **kw: calls.__setitem__("create", (pid, kw)) or {"id": pid, "chat_id": 7})
    monkeypatch.setattr(
        reg,
        "bind_task_to_project",
        lambda dr, tid, pid, chat=None, *, origin: calls.__setitem__("bind", (tid, pid, chat, origin)),
    )
    monkeypatch.setattr(reg, "touch_project", lambda dr, pid: calls.__setitem__("touch", pid))

    class _Bridge:
        def broadcast(self, payload):
            calls["broadcast"] = payload

    monkeypatch.setattr(mb, "get_bridge", lambda: _Bridge())

    running = {"t1": {"task": {"id": "t1"}}}
    ctx = SimpleNamespace(RUNNING=running)
    workers.ensure_project_scope({"task_id": "t1", "project_id": "cyber-racing", "project_name": "Cyber Racing"}, ctx)

    assert calls["create"][0] == "cyber-racing"
    # An in-flight self-scope with no chat-born origin binds with the typed reason.
    assert calls["bind"] == ("t1", "cyber-racing", 7, {"absent": "mid_task_no_origin"})
    assert running["t1"]["task"]["project_id"] == "cyber-racing"  # F1: lease lane occupancy
    assert calls["broadcast"] == {"type": "projects_changed", "project_id": "cyber-racing", "chat_id": 7}


def test_supervisor_handler_refuses_before_side_effects_when_bound_elsewhere(monkeypatch, tmp_path):
    """B4=A: create precedes bind, so a task bound elsewhere used to leave a project
    row, a lease mark, a broadcast and a chat announcement behind before the immutable
    bind refused. The refusal is now first, and the requested name is carried to the
    project the task actually belongs to."""
    import json

    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    calls = {"create": None, "bind": None, "touch": None, "broadcast": None, "rename": None,
             "announce": None}
    monkeypatch.setattr(reg, "project_id_for_task", lambda dr, tid, **kw: "token-atlas")
    monkeypatch.setattr(reg, "create_project", lambda *a, **kw: calls.__setitem__("create", a))
    monkeypatch.setattr(reg, "bind_task_to_project", lambda *a, **kw: calls.__setitem__("bind", a))
    monkeypatch.setattr(reg, "touch_project", lambda *a, **kw: calls.__setitem__("touch", a))
    monkeypatch.setattr(reg, "update_project",
                        lambda dr, pid, **kw: calls.__setitem__("rename", (pid, kw)))
    monkeypatch.setattr(mb, "get_bridge", lambda: calls.__setitem__("broadcast", True))
    monkeypatch.setattr(workers, "_announce_created_project",
                        lambda *a, **kw: calls.__setitem__("announce", True))

    running = {"t1": {"task": {"id": "t1", "project_id": "token-atlas"}}}
    workers.ensure_project_scope(
        {"task_id": "t1", "project_id": "token-observatory", "project_name": "Token Observatory"},
        SimpleNamespace(RUNNING=running),
    )

    assert calls["create"] is None and calls["bind"] is None and calls["touch"] is None
    assert calls["broadcast"] is None and calls["announce"] is None
    assert calls["rename"] == ("token-atlas", {"name": "Token Observatory"})
    assert running["t1"]["task"]["project_id"] == "token-atlas"
    rows = [json.loads(line) for line in
            (tmp_path / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows[-1]["type"] == "project_binding_failed"
    assert rows[-1]["reason"] == "project_scope_conflict"
    assert rows[-1]["project_id"] == "token-observatory"


def test_supervisor_handler_stops_when_the_bind_raises(monkeypatch, tmp_path):
    """A refused bind must not be followed by the lease mark, the broadcast and the
    announcement: the task is not in that project, so nothing may say it is."""
    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    calls = {"broadcast": None, "announce": None}

    def _raise(*_a, **_kw):
        raise ValueError("project binding is immutable")

    monkeypatch.setattr(reg, "project_id_for_task", lambda dr, tid, **kw: "")
    monkeypatch.setattr(reg, "create_project", lambda dr, pid, **kw: {"id": pid, "chat_id": 7})
    monkeypatch.setattr(reg, "touch_project", lambda *a, **kw: None)
    monkeypatch.setattr(reg, "bind_task_to_project", _raise)
    monkeypatch.setattr(mb, "get_bridge", lambda: calls.__setitem__("broadcast", True))
    monkeypatch.setattr(workers, "_announce_created_project",
                        lambda *a, **kw: calls.__setitem__("announce", True))

    running = {"t1": {"task": {"id": "t1", "project_id": ""}}}
    workers.ensure_project_scope(
        {"task_id": "t1", "project_id": "cyber-racing", "project_name": "Cyber Racing"},
        SimpleNamespace(RUNNING=running),
    )

    assert running["t1"]["task"]["project_id"] == ""
    assert calls["broadcast"] is None and calls["announce"] is None


def test_supervisor_handler_treats_an_unreadable_store_as_unbound(monkeypatch, tmp_path, caplog):
    """Proportionality (D6-6): an unreadable bindings file behaves like "no binding"
    and is disclosed once; it does not stop the conversion."""
    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "project_task_bindings.json").write_text("{ not json", encoding="utf-8")
    bound = {}
    monkeypatch.setattr(reg, "create_project", lambda dr, pid, **kw: {"id": pid, "chat_id": 7})
    monkeypatch.setattr(reg, "touch_project", lambda *a, **kw: None)
    monkeypatch.setattr(reg, "bind_task_to_project",
                        lambda dr, tid, pid, chat=None, *, origin: bound.update({"pid": pid}))
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr(workers, "_announce_created_project", lambda *a, **kw: None)

    running = {"t1": {"task": {"id": "t1", "project_id": ""}}}
    with caplog.at_level(logging.WARNING):
        workers.ensure_project_scope(
            {"task_id": "t1", "project_id": "cyber-racing", "project_name": "Cyber Racing"},
            SimpleNamespace(RUNNING=running),
        )

    assert bound == {"pid": "cyber-racing"}
    assert running["t1"]["task"]["project_id"] == "cyber-racing"
    assert "project_binding_unreadable" in caplog.text


# --- the sibling half: one owner message, one Project -------------------------

_OWNER_TEXT = "Publish and merge the seven pull requests"


def _owner_ref(client_message_id: str = "msg-owner", chat_id: int = 1) -> dict:
    from ouroboros.project_dialogue import build_owner_message_ref

    return build_owner_message_ref(
        chat_id=chat_id, client_message_id=client_message_id,
        ts="2026-09-14T12:15:20+00:00", text=_OWNER_TEXT,
    )


def _bind_the_work(tmp_path, ref):
    """The turn's card was already turned into a Project; T2 is a DIFFERENT task id of
    the SAME owner message, itself unbound."""
    from ouroboros.projects_registry import bind_task_to_project, create_project

    room = create_project(tmp_path, "the-work", name="The Work")
    bind_task_to_project(tmp_path, "t-turn", "the-work", room["chat_id"],
                         origin={"ref": ref, "text": _OWNER_TEXT})
    return room


def test_tool_guard_keeps_an_explicit_name_as_the_models_own_choice(tmp_path):
    """P13: the origin answers an IMPLICIT act. A sibling task that explicitly asks for
    a differently named room gets it - the message's existing project is NOT renamed
    from a task that never belonged to it, and the guard's answer matches what the
    supervisor handler will do with the same event."""
    from ouroboros.project_facts import project_id_from_display_name
    from ouroboros.projects_registry import get_project, list_projects
    from ouroboros.tools.control import _ensure_project_scope

    ref = _owner_ref()
    _bind_the_work(tmp_path, ref)
    ctx = _ctx(task_id="t-root", task_metadata={"origin_message_ref": dict(ref)})
    out = _ensure_project_scope(ctx, project_name="Token Observatory")

    assert out.startswith("⚠️ SCOPE_UNCONFIRMED")  # no supervisor here: nothing landed yet
    assert ctx.project_id == project_id_from_display_name("Token Observatory")
    assert [p["id"] for p in list_projects(tmp_path)] == ["the-work"]      # the tool creates nothing
    assert get_project(tmp_path, "the-work")["name"] == "The Work"         # and renames nothing
    evs = [e for e in ctx.pending_events if e.get("type") == "ensure_project_scope"]
    assert len(evs) == 1 and evs[0]["project_id"] == "token-observatory"


def test_supervisor_handler_adopts_the_project_the_owner_message_already_has(
    tmp_path, monkeypatch,
):
    """One owner message spawns several task ids. When the turn's card was already
    turned into a Project and the ROOT's mid-run scope call names that same room, the
    root JOINS it: one durable binding for this exact task, its live lane moved onto
    the project it now belongs to, and no second room."""
    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    ref = _owner_ref()
    _bind_the_work(tmp_path, ref)
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr(workers, "_announce_created_project", lambda *a, **kw: None)

    # A bare-workspace promote leaves a DERIVED project id on the row; the adopt owns
    # the binding it writes, so the lane follows it (authority="binding").
    running = {"t-root": {"task": {"id": "t-root", "project_id": "proj_deadbeef1234",
                                   "origin_message_ref": dict(ref),
                                   "origin_message_text": _OWNER_TEXT}}}
    workers.ensure_project_scope(
        {"task_id": "t-root", "project_id": "the-work", "project_name": "The Work"},
        SimpleNamespace(RUNNING=running, PENDING=[]),
    )

    assert (reg.project_binding_for_task(tmp_path, "t-root") or {}).get("project_id") == "the-work"
    assert running["t-root"]["task"]["project_id"] == "the-work"
    assert [p["id"] for p in reg.list_projects(tmp_path)] == ["the-work"]


def test_supervisor_handler_lets_an_explicit_different_name_fork(tmp_path, monkeypatch):
    """The other half of P13: the same sibling asking for a DIFFERENT room creates it
    and binds itself there; the message's first project keeps its name and its task."""
    import ouroboros.projects_registry as reg
    import supervisor.message_bus as mb
    from supervisor import workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    ref = _owner_ref()
    _bind_the_work(tmp_path, ref)
    monkeypatch.setattr(mb, "get_bridge", lambda: SimpleNamespace(broadcast=lambda payload: None))
    monkeypatch.setattr(workers, "_announce_created_project", lambda *a, **kw: None)

    running = {"t-root": {"task": {"id": "t-root", "project_id": "",
                                   "origin_message_ref": dict(ref),
                                   "origin_message_text": _OWNER_TEXT}}}
    workers.ensure_project_scope(
        {"task_id": "t-root", "project_id": "token-observatory",
         "project_name": "Token Observatory"},
        SimpleNamespace(RUNNING=running, PENDING=[]),
    )

    assert (reg.project_binding_for_task(tmp_path, "t-root") or {}).get(
        "project_id") == "token-observatory"
    assert (reg.project_binding_for_task(tmp_path, "t-turn") or {}).get("project_id") == "the-work"
    assert reg.get_project(tmp_path, "the-work")["name"] == "The Work"
    assert sorted(p["id"] for p in reg.list_projects(tmp_path)) == ["the-work", "token-observatory"]


def test_the_retry_an_unconfirmed_bind_asks_for_reads_the_durable_outcome_not_its_own_scope():
    """The unconfirmed result tells the model to call again; that call must not
    turn the optimistic in-memory scope into "already scoped" — it binds again
    (the handler attaches to the same project) or reports the durable truth."""
    from ouroboros.tools.control import _ensure_project_scope

    ctx = _ctx()
    first = _ensure_project_scope(ctx, project_name="Cyber Racing")
    assert first.startswith("⚠️ SCOPE_UNCONFIRMED")
    second = _ensure_project_scope(ctx, project_name="Cyber Racing")
    assert "already scoped" not in second and second.startswith("⚠️ SCOPE_UNCONFIRMED")
    assert len([e for e in ctx.pending_events if e.get("type") == "ensure_project_scope"]) == 2
    # a different project while the first is pending is a re-scope of a pending scope, refused as before
    third = _ensure_project_scope(ctx, project_name="Other Racing")
    assert "cannot be re-scoped" in third


def test_a_conflict_receipt_names_the_project_the_task_is_actually_bound_to():
    """A conversion that won after the tool read the binding: the supervisor's
    receipt carries the real target; the worker scope follows it and the text
    never renders a rename status as a project id."""
    from ouroboros.tools.control_delegation import _scope_outcome_text

    ctx = _ctx(project_id="requested")
    out = _scope_outcome_text(
        ctx, {"status": "rejected", "reason": "project_scope_conflict", "detail": "renamed", "target": "existing"},
        mode="live", tid="t1", pid="requested", bound="", display_name="Cyber Racing", previous_scope="",
    )
    assert "durably bound to project 'existing'" in out and "'renamed'" not in out
    assert "applied to it as a rename" in out
    assert ctx.project_id == "existing"

