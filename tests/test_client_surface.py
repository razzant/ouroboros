"""Owner Surface Fact: per-message client_surface provenance behavior.

Covers the ingress normalizer, chat.jsonl persistence, bus survival, mailbox
round-trip, the loop's change-note, producer wiring pins, the history-endpoint
negative (the field never replays to the SPA), and the prompt-fact parity
binding SYSTEM.md's product claims to the client code that implements them.
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib
from types import SimpleNamespace

from ouroboros.client_surface import (
    CLIENT_SURFACE_UA_LIMIT,
    client_surface_identity,
    normalize_client_surface,
)

REPO = pathlib.Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Normalizer: closed keys, disclosed bounds, honest emptiness
# ---------------------------------------------------------------------------

def test_normalizer_keeps_closed_keys_and_drops_unknown():
    fact = normalize_client_surface({
        "pywebview": True,
        "ua": "  TestShell/1.0  ",
        "viewport": {"w": "1200", "h": 800},
        "narrow_layout": False,
        "coarse_pointer": True,
        "captured_at": "2026-08-18T00:00:00Z",
        "evil_extra": "dropped",
        "cookies": {"a": 1},
    })
    assert fact == {
        "pywebview": True,
        "ua": "TestShell/1.0",
        "viewport": {"w": 1200, "h": 800},
        "narrow_layout": False,
        "coarse_pointer": True,
        "captured_at": "2026-08-18T00:00:00Z",
    }


def test_normalizer_empty_and_non_dict_are_none():
    assert normalize_client_surface({}) is None
    assert normalize_client_surface(None) is None
    assert normalize_client_surface("desktop") is None
    assert normalize_client_surface({"unknown_only": True}) is None


def test_normalizer_bounds_are_disclosed_not_silent():
    fact = normalize_client_surface({"ua": "x" * 2000})
    assert fact is not None
    ua = fact["ua"]
    assert len(ua) <= CLIENT_SURFACE_UA_LIMIT
    # The strict-bound SSOT keeps the omission marker INSIDE the limit.
    assert "chars" in ua or "…" in ua, "oversized ua must carry a disclosed marker"


def test_normalizer_clamps_viewport_and_bounds_timestamp():
    fact = normalize_client_surface({
        "viewport": {"w": -5, "h": 10_000_000},
        "captured_at": "t" * 300,
    })
    assert fact["viewport"] == {"w": 0, "h": 100_000}
    assert len(fact["captured_at"]) <= 64


def test_normalizer_bad_viewport_is_dropped_not_fatal():
    fact = normalize_client_surface({"pywebview": True, "viewport": {"w": "nan", "h": None}})
    assert fact == {"pywebview": True}
    # stock json.loads accepts the Infinity literal; int(inf) raises
    # OverflowError which must be swallowed like every other bad viewport
    # (proven adversarial finding: an escape here dropped the whole message).
    fact = normalize_client_surface({"pywebview": True, "viewport": {"w": float("inf"), "h": 5}})
    assert fact == {"pywebview": True}
    fact = normalize_client_surface({"viewport": {"w": float("-inf"), "h": float("nan")}})
    assert fact is None


def test_normalizer_accepts_only_real_booleans():
    # "false"/"0" strings would coerce to True and invert the fact — drop them.
    assert normalize_client_surface({"pywebview": "false"}) is None
    assert normalize_client_surface({"coarse_pointer": "0", "narrow_layout": 1}) is None
    assert normalize_client_surface({"pywebview": False}) == {"pywebview": False}


# ---------------------------------------------------------------------------
# Identity: what counts as a surface CHANGE (resize/rotation never does)
# ---------------------------------------------------------------------------

def test_identity_excludes_viewport_and_narrow_layout():
    a = {"pywebview": False, "coarse_pointer": False, "ua": "U", "viewport": {"w": 100, "h": 50}, "narrow_layout": False}
    b = {"pywebview": False, "coarse_pointer": False, "ua": "U", "viewport": {"w": 900, "h": 50}, "narrow_layout": True}
    assert client_surface_identity(a) == client_surface_identity(b)


def test_identity_changes_on_bridge_pointer_ua_or_channel():
    base = {"pywebview": False, "coarse_pointer": False, "ua": "U"}
    assert client_surface_identity(base) != client_surface_identity({**base, "pywebview": True})
    assert client_surface_identity(base) != client_surface_identity({**base, "coarse_pointer": True})
    assert client_surface_identity(base) != client_surface_identity({**base, "ua": "V"})
    assert client_surface_identity({"channel": "cli"}) != client_surface_identity({"channel": "api_command"})
    assert client_surface_identity(None) is None
    assert client_surface_identity({}) is None
    # A fact carrying NONE of the identity keys has no identity at all — it
    # must not collapse to the all-empty tuple and fake a "DIFFERENT surface".
    assert client_surface_identity({"narrow_layout": True, "viewport": {"w": 1, "h": 1}}) is None


# ---------------------------------------------------------------------------
# chat.jsonl persistence (writer half)
# ---------------------------------------------------------------------------

def test_log_chat_persists_client_surface_column(tmp_path, monkeypatch):
    import supervisor.message_bus as mb

    monkeypatch.setattr(mb, "DATA_DIR", tmp_path)
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    fact = {"pywebview": True, "ua": "TestShell/1.0"}
    mb.log_chat("in", 1, 1, "hello", source="web", client_surface=fact)
    mb.log_chat("in", 1, 1, "bare", source="web")
    rows = [json.loads(line) for line in (tmp_path / "logs" / "chat.jsonl").read_text().splitlines()]
    assert rows[0]["client_surface"] == fact
    assert "client_surface" not in rows[1]


# ---------------------------------------------------------------------------
# Bus survival: nested in task_metadata through the get_updates whitelist
# ---------------------------------------------------------------------------

def test_client_surface_survives_bus_inside_task_metadata():
    from supervisor.message_bus import LocalChatBridge

    bridge = LocalChatBridge()
    fact = {"pywebview": False, "ua": "Phone/1.0", "received_at": "2026-08-18T00:00:00Z"}
    bridge.enqueue_local_message(
        "follow-up", chat_id=1, user_id=1, source="web",
        task_metadata={"client_surface": fact},
    )
    updates = bridge.get_updates(offset=0, timeout=1)
    assert updates, "message must come back out of the bus"
    meta = updates[0]["message"]["task_metadata"]
    assert meta["client_surface"] == fact


# ---------------------------------------------------------------------------
# Mailbox round-trip + the loop's change-note
# ---------------------------------------------------------------------------

def test_mailbox_round_trip_carries_client_surface(tmp_path):
    from ouroboros.owner_mailbox import drain_owner_entries, write_owner_message

    fact = {"pywebview": False, "coarse_pointer": True, "ua": "Phone/1.0"}
    assert write_owner_message(tmp_path, "from phone", "t1", msg_id="m1", client_surface=fact)
    assert write_owner_message(tmp_path, "bare", "t1", msg_id="m2")
    entries = drain_owner_entries(tmp_path, task_id="t1", seen_ids=set())
    by_id = {e["msg_id"]: e for e in entries}
    assert by_id["m1"]["client_surface"] == fact
    assert "client_surface" not in by_id["m2"]


def _ctx_with_surface(fact):
    return SimpleNamespace(task_metadata={"client_surface": fact} if fact else {})


def test_owner_surface_note_only_on_identity_change():
    from ouroboros.client_surface import owner_surface_note as _owner_surface_note

    desktop = {"pywebview": True, "coarse_pointer": False, "ua": "Shell/1"}
    phone = {"pywebview": False, "coarse_pointer": True, "ua": "Phone/1"}
    ctx = _ctx_with_surface(desktop)
    # Same identity as the task's own start fact: silence.
    assert _owner_surface_note(ctx, dict(desktop)) == ""
    # Real device change: a loud note carrying the raw fact.
    note = _owner_surface_note(ctx, phone)
    assert "DIFFERENT client surface" in note and "Phone/1" in note
    # Same new surface again: silence (baseline advanced to last-seen).
    assert _owner_surface_note(ctx, dict(phone)) == ""


def test_owner_surface_note_viewport_resize_is_not_a_change():
    from ouroboros.client_surface import owner_surface_note as _owner_surface_note

    base = {"pywebview": True, "coarse_pointer": False, "ua": "Shell/1",
            "viewport": {"w": 1400, "h": 900}, "narrow_layout": False}
    resized = {**base, "viewport": {"w": 700, "h": 900}, "narrow_layout": True}
    ctx = _ctx_with_surface(base)
    assert _owner_surface_note(ctx, resized) == ""


def test_owner_surface_note_first_fact_without_baseline_is_neutral():
    from ouroboros.client_surface import owner_surface_note as _owner_surface_note

    ctx = _ctx_with_surface(None)
    note = _owner_surface_note(ctx, {"pywebview": False, "ua": "Phone/1"})
    assert "sent from client surface" in note
    assert "DIFFERENT" not in note
    # And no note at all when there is nothing to compare or say.
    assert _owner_surface_note(None, {"ua": "x"}) == ""
    assert _owner_surface_note(ctx, None) == ""


# ---------------------------------------------------------------------------
# Producer wiring pins (the dead-wire class: a consumer test alone stays green
# while the producer never sends — pin the producers structurally)
# ---------------------------------------------------------------------------

def _function_calls(tree: ast.AST, func_name: str) -> set[str]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return {
                n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", "")
                for n in ast.walk(node)
                if isinstance(n, ast.Call)
            }
    raise AssertionError(f"function {func_name} not found")


def test_all_three_routing_producers_attach_client_surface():
    # v7 split: the three producers and the attacher live in the routing leaf;
    # tools/control.py re-exports them.
    tree = ast.parse((REPO / "ouroboros" / "tools" / "control_routing.py").read_text(encoding="utf-8"))
    for producer in ("_promote_chat_to_task", "_route_to_project", "_steer_task"):
        calls = _function_calls(tree, producer)
        assert "_attach_client_surface" in calls, (
            f"{producer} no longer attaches client_surface — the fact silently "
            "dies at this transition (the is_desktop dead-wire class)"
        )


def test_owner_surface_note_baseline_uses_the_shared_projection():
    """Codex scope C1: the mailbox baseline and the context render must read ONE
    projection — a CLI/API-admitted task (channel fact stamped at admission)
    whose first web follow-up arrives is a surface CHANGE, never a neutral
    first fact."""
    from ouroboros.client_surface import owner_client_fact, owner_surface_note

    ctx = SimpleNamespace(task_metadata={"client_surface": {"channel": "cli"}})
    note = owner_surface_note(ctx, {"pywebview": False, "ua": "Mozilla/5.0 test"})
    assert "DIFFERENT" in note, "known channel baseline must make a web fact a CHANGE"
    # The context render consumes the same projection function (no inline twin).
    context_src = (REPO / "ouroboros" / "context.py").read_text(encoding="utf-8")
    assert "owner_client_fact(" in context_src
    # The projection reads ONLY the producer-assembled fact: metadata.source is
    # overloaded (scheduler) and must never be dressed up as an owner surface.
    assert owner_client_fact({"client_surface": {"pywebview": True}}) == {"pywebview": True}
    for source in ("cli", "scheduled_task", "skill_scheduled_task", "web"):
        assert owner_client_fact({"source": source}) is None
    assert owner_client_fact({}) is None
    assert owner_client_fact(None) is None


def test_schedule_template_can_never_smuggle_a_surface_fact():
    """Codex scope round 3: POST /api/schedules accepted caller metadata and
    _task_from_schedule copied it — a forged client_surface (incl. a fake
    received_at) became a machine task's owner_client. The key is now in the
    RESERVED_TEMPLATE_FIELDS SSOT: admission rejects it loudly, and the
    producer filter strips it from records persisted before the rule."""
    from ouroboros.schedule_contract import RESERVED_TEMPLATE_FIELDS
    from supervisor import queue as squeue

    assert "client_surface" in RESERVED_TEMPLATE_FIELDS
    record = {
        "id": "sched-forged",
        "name": "forged surface",
        "task": {
            "text": "machine work",
            "metadata": {
                "client_surface": {"pywebview": True, "received_at": "forged"},
                "harmless": "kept",
            },
        },
    }
    task = squeue._task_from_schedule(record)
    assert "client_surface" not in task["metadata"], (
        "a schedule-fired machine task must never wear an owner surface"
    )
    assert task["metadata"]["harmless"] == "kept"
    from ouroboros.client_surface import owner_client_fact

    assert owner_client_fact(task["metadata"]) is None


def test_external_admission_stamps_channel_and_never_forwards_caller_surface():
    """Codex scope round 2 N3: /api/tasks metadata is caller-controlled and
    client_surface is not a reserved key — the admission must OVERWRITE any
    caller-built descriptor with its caller-declared channel stamp (a fake
    received_at would impersonate a host stamp)."""
    tasks_src = (REPO / "ouroboros" / "gateway" / "tasks.py").read_text(encoding="utf-8")
    stamp = 'metadata["client_surface"] = {"channel": str(metadata.get("source") or "api_task")}'
    assert stamp in tasks_src, "external admission no longer stamps/overwrites the channel fact"
    assert tasks_src.index('metadata.setdefault("source"') < tasks_src.index(stamp), (
        "the stamp must read the already-defaulted source"
    )


def test_ws_ingress_normalizes_and_stamps_received_at():
    ws_src = (REPO / "ouroboros" / "gateway" / "ws.py").read_text(encoding="utf-8")
    assert "normalize_client_surface" in ws_src
    assert 'client_surface["received_at"] = utc_now_iso()' in ws_src, (
        "the ws chat branch no longer stamps the host received_at beside the "
        "client-reported captured_at — the provenance pair breaks silently"
    )


def test_attach_client_surface_copies_by_value_and_skips_absence():
    from ouroboros.tools.control import _attach_client_surface

    fact = {"pywebview": True, "ua": "Shell/1"}
    evt: dict = {}
    _attach_client_surface(SimpleNamespace(task_metadata={"client_surface": fact}), evt)
    assert evt["client_surface"] == fact
    assert evt["client_surface"] is not fact, "must copy by value, never share the dict"
    for ctx in (
        SimpleNamespace(task_metadata={}),
        SimpleNamespace(task_metadata={"client_surface": {}}),
        SimpleNamespace(task_metadata={"client_surface": "bogus"}),
        SimpleNamespace(task_metadata=None),
        SimpleNamespace(),
    ):
        evt = {}
        _attach_client_surface(ctx, evt)
        assert "client_surface" not in evt


def test_promotion_lands_client_surface_under_metadata(monkeypatch, tmp_path):
    """Behavioral landing test: the promoted task carries the fact under
    task['metadata'] even when no metadata dict pre-existed (only force_plan
    creates one) — a top-level landing would be invisible to the renderer."""
    import types

    import supervisor.workers as workers

    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    pending = []
    ctx = types.SimpleNamespace(
        enqueue_task=lambda task: pending.append(dict(task)) or task,
        load_state=lambda: {"owner_chat_id": 1},
    )
    fact = {"pywebview": False, "coarse_pointer": True, "ua": "Phone/1"}
    workers.promote_chat_to_task(
        {"task_id": "surf-land", "objective": "Build", "client_surface": fact},
        ctx,
    )
    assert len(pending) == 1
    assert pending[0]["metadata"]["client_surface"] == fact
    assert "client_surface" not in pending[0], "fact must not land top-level"


def test_route_owner_message_stamps_channel_for_non_web_ingress(monkeypatch):
    """Behavioral producer test for the host {channel: source} stamp: a
    telegram-like bridge message renders a channel fact, a plain web message
    stays an honest absence (§9a №9)."""
    import server

    captured = []

    class ImmediateThread:
        def __init__(self, target, args=(), kwargs=None, daemon=False):
            self.target, self.args, self.kwargs = target, args, kwargs or {}

        def start(self):
            self.target(*self.args, **self.kwargs)

    monkeypatch.setattr(server.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(server, "_reserved_project_for_chat", lambda ctx, chat_id: {})
    monkeypatch.setattr(server, "_addressable_root_tasks", lambda ctx, _pid: [])
    monkeypatch.setattr("ouroboros.projects_registry.list_projects", lambda _root: [])
    monkeypatch.setattr(server, "_decision_turn_metadata", lambda ctx, chat_id, cmid, md: md)

    def make_ctx():
        return SimpleNamespace(
            DRIVE_ROOT=None,
            consciousness=SimpleNamespace(
                inject_observation=lambda *_: None, pause=lambda: None, resume=lambda: None
            ),
            get_chat_agent=lambda: SimpleNamespace(_busy=True),
            handle_chat_ephemeral=lambda *a, **kw: captured.append(kw.get("task_metadata")),
            handle_chat_direct=lambda *a, **kw: captured.append(kw.get("task_metadata")),
        )

    server._route_owner_message(SimpleNamespace(), make_ctx(), {
        "chat_id": 1, "text": "привет из телеги", "client_message_id": "m1",
        "task_metadata": None, "log_text": "привет из телеги",
        "origin_message_ref": {"chat_id": 1, "client_message_id": "m1"},
        "source": "skill:telegram",
    })
    assert captured[-1]["client_surface"] == {"channel": "skill:telegram"}

    server._route_owner_message(SimpleNamespace(), make_ctx(), {
        "chat_id": 1, "text": "обычный веб", "client_message_id": "m2",
        "task_metadata": None, "log_text": "обычный веб",
        "origin_message_ref": {"chat_id": 1, "client_message_id": "m2"},
        "source": "web",
    })
    assert "client_surface" not in (captured[-1] or {})

    # A real browser fact is never overwritten by the channel stamp.
    server._route_owner_message(SimpleNamespace(), make_ctx(), {
        "chat_id": 1, "text": "с фактом", "client_message_id": "m3",
        "task_metadata": {"client_surface": {"pywebview": True}}, "log_text": "с фактом",
        "origin_message_ref": {"chat_id": 1, "client_message_id": "m3"},
        "source": "skill:telegram",
    })
    assert captured[-1]["client_surface"] == {"pywebview": True}

    # A synthetic A2A chat (negative id) is machine traffic: same skill source,
    # NO owner surface stamp (codex scope round 2 N2).
    server._route_owner_message(SimpleNamespace(), make_ctx(), {
        "chat_id": -42, "text": "a2a ping", "client_message_id": "m4",
        "task_metadata": None, "log_text": "a2a ping",
        "origin_message_ref": {"chat_id": -42, "client_message_id": "m4"},
        "source": "skill:a2a",
    })
    assert "client_surface" not in (captured[-1] or {}), (
        "machine-to-machine traffic must never wear an owner_client fact"
    )


def test_steering_and_project_mailbox_writers_pass_client_surface():
    # Tripwire complement to the behavioral tests above (the mailbox writers are
    # exercised via write_owner_message round-trip; these pins catch a dropped
    # kwarg at the two forwarding call sites).
    steering = (REPO / "supervisor" / "steering.py").read_text(encoding="utf-8")
    # v7 split: the project-mailbox write moved to the owner-routing leaf while the
    # log_chat forwarding stayed in server.py, so the pin reads BOTH owners as one
    # surface — the point is that neither call site loses the kwarg.
    server_src = ((REPO / "server.py").read_text(encoding="utf-8")
                  + (REPO / "ouroboros" / "server_owner_routing.py").read_text(encoding="utf-8"))
    assert "client_surface=" in steering, "steer mailbox write dropped client_surface"
    # BOTH server call sites (project-mailbox write AND log_chat forwarding)
    # must carry the kwarg — a single-substring pin went false-green when one
    # of the two was dropped (final code review MAJOR).
    assert server_src.count("client_surface=(") >= 2, (
        "a server client_surface forwarding call site was dropped "
        f"(found {server_src.count('client_surface=(')}, expected >= 2)"
    )


def test_presentation_env_is_stripped_by_benchmark_server_runner():
    runner = (REPO / "devtools" / "benchmarks" / "common" / "server_runner.py").read_text(encoding="utf-8")
    assert '"OUROBOROS_PRESENTATION",' in runner, (
        "isolated benchmark servers must not inherit the operator's desktop posture"
    )


# ---------------------------------------------------------------------------
# History endpoint negative: the field never replays to the SPA (R3)
# ---------------------------------------------------------------------------

def test_chat_history_does_not_emit_client_surface(tmp_path):
    from ouroboros.gateway.history import make_chat_history_endpoint

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "progress.jsonl").write_text("", encoding="utf-8")
    (logs / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-08-18T00:00:00Z",
            "direction": "in",
            "chat_id": 1,
            "user_id": 1,
            "text": "hello from phone",
            "source": "web",
            "client_surface": {"pywebview": False, "ua": "Phone/1.0"},
        }) + "\n",
        encoding="utf-8",
    )
    endpoint = make_chat_history_endpoint(tmp_path)
    response = asyncio.run(endpoint(SimpleNamespace(query_params={"limit": "10"})))
    payload = json.loads(response.body.decode("utf-8"))["messages"]
    rec = next(item for item in payload if item.get("text") == "hello from phone")
    assert "client_surface" not in rec, (
        "client_surface is model/forensics provenance; the SPA knows its own "
        "surface and the history projection deliberately omits it"
    )


# ---------------------------------------------------------------------------
# Prompt-fact parity: SYSTEM.md product claims stay bound to the code (AC8)
# ---------------------------------------------------------------------------

def test_system_prompt_reload_claim_is_code_bound():
    system_md = (REPO / "prompts" / "SYSTEM.md").read_text(encoding="utf-8")
    if "reload-on-SHA" not in system_md and "reloads itself" not in system_md:
        return  # the claim was removed together with its obligation
    ws_js = (REPO / "web" / "modules" / "ws.js").read_text(encoding="utf-8")
    assert "sha-change" in ws_js or "Reload on SHA change" in ws_js, (
        "SYSTEM.md claims the UI auto-reloads on SHA change, but the ws.js "
        "reload path is gone — the prompt now states a false product fact "
        "(the exact failure class this feature exists to close)"
    )


def test_frontend_sends_raw_observables_without_device_taxonomy():
    chat_js = (REPO / "web" / "modules" / "chat.js").read_text(encoding="utf-8")
    surface_js = (REPO / "web" / "modules" / "client_surface.js").read_text(encoding="utf-8")
    # The send site imports AND actually spreads the snapshot field into the
    # frame (the import alone is the dead-wire class this feature closes); the
    # module measures at send time and never labels devices.
    assert "client_surface.js" in chat_js
    assert "...clientSurfaceField()," in chat_js
    assert "clientSurfaceSnapshot" in surface_js and "matchMedia" in surface_js
    assert "client_surface" in surface_js
    for label in ("mobile_browser", "desktop_app", "browser_tab"):
        for src_name, src in (("chat.js", chat_js), ("client_surface.js", surface_js)):
            assert label not in src, (
                f"device taxonomy label {label!r} crept into {src_name} — raw "
                "observables only, the model classifies"
            )
