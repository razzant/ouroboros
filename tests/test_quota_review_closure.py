"""Bounded review regressions, retained from independent synthetic witnesses."""
from __future__ import annotations

# Regressions derived from independent witnesses; no live-provider calls.
import json

import pytest


def test_inspector_resolves_the_main_actor_offered_by_empty_enabled_catalog(tmp_path, monkeypatch):
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
    from ouroboros.subagent_runtime import model_visible_subagent_catalog, select_subagent_snapshot
    from ouroboros.tools.registry import ToolRegistry

    from tests._shared import saved_settings

    settings = {
        "OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": []}),
        "OUROBOROS_MODEL": "openai/synthetic-main-model",
        "OPENROUTER_API_KEY": "synthetic-unused-credential",
    }
    saved_settings(monkeypatch, tmp_path, OUROBOROS_MODEL=settings["OUROBOROS_MODEL"],
                   OPENROUTER_API_KEY=settings["OPENROUTER_API_KEY"])  # route keys read the saved document
    catalog = model_visible_subagent_catalog(settings)
    handle = catalog["rows"][0]["subagent_id"]
    selected, _ = select_subagent_snapshot(settings, subagent_id=handle)
    assert selected["selected_subagent_id"] == "main"
    registry = ToolRegistry(tmp_path / "repo", tmp_path / "data")
    with task_settings_scope(task_settings_snapshot(settings, settings)):
        result = registry.execute_result("inspect_subagent", {"subagent_id": handle})
    assert result.status == "ok", f"Catalog offered {handle}; scheduler resolved main; inspector: {result.text}"


def test_inspection_effort_agrees_with_model_named_effort(monkeypatch):
    from ouroboros import config
    from ouroboros.gateway import claudexor_passive
    from ouroboros.settings_integrity import task_settings_scope, task_settings_snapshot
    from ouroboros.subagent_inspection import inspect_subagent
    from ouroboros.subagent_runtime import model_visible_subagent_catalog

    settings = {"OUROBOROS_SUBAGENTS": json.dumps({"enabled": True, "items": [{
        "subagent_id": "synthetic", "recommended_use": "synthetic fixture",
        "route": {"kind": "agent_session", "target_id": "cursor=cursor-grok-4.6-high"},
    }]})}
    monkeypatch.setattr(config, "load_settings", lambda: {})
    monkeypatch.setattr(claudexor_passive, "_quota_payload", lambda **kwargs: {
        "reads": {}, "read_errors": {}, "timings_ms": {}, "profiles": {},
        "quota": [], "quota_absences": [],
    })
    catalog = model_visible_subagent_catalog(settings)
    offered = catalog["rows"][0]
    with task_settings_scope(task_settings_snapshot(settings, settings)):
        actual = inspect_subagent(offered["subagent_id"])
    assert actual["configured"]["effort"] == offered["effort"]


def test_recursively_malformed_snapshot_is_unknown_instead_of_aborting_the_round(tmp_path):
    from ouroboros.skill_context_snapshot import _read_snapshot

    path = tmp_path / "snapshot.json"
    path.write_text("[" * 2000 + "0" + "]" * 2000)
    descriptor = {"revision": "synthetic", "eligible": True, "reason": "ready",
                  "state_dir": str(tmp_path), "path": "snapshot.json"}
    result = _read_snapshot("synthetic", descriptor, 1_800_000_000)
    assert result["state"] == "unknown"


@pytest.mark.parametrize("malformation", ["recursive_json", "symlink_loop"])
def test_malformed_snapshot_withdraws_without_aborting_real_round(tmp_path, monkeypatch, malformation):
    from types import SimpleNamespace

    from ouroboros import extension_loader, peer_roster
    from ouroboros.loop_round_limits import _CompactionRoundContext, _run_round_compaction
    from ouroboros.skill_context_snapshot import visible_snapshots
    from tests.test_skill_context_snapshot import _fixture, _write

    root, skill, ctx, path = _fixture(tmp_path)
    monkeypatch.setattr(peer_roster, "maybe_append_roster_note", lambda *args: False)
    round_ctx = _CompactionRoundContext(tools=SimpleNamespace(_ctx=ctx), drive_root=root,
        drive_logs=root / "logs", task_id="root", round_idx=2,
        event_queue=None, emit_progress=lambda value: None)
    try:
        messages = [{"role": "system", "content": "Synthetic stable core"},
                    {"role": "user", "content": "Synthetic original task"}]
        _write(path, "Synthetic current fact")
        messages = _run_round_compaction(messages, round_ctx)
        assert visible_snapshots(messages)[skill.name][1]["state"] == "fresh"
        if malformation == "recursive_json":
            path.write_text("[" * 2000 + "0" + "]" * 2000)
        else:
            path.unlink()
            path.symlink_to(path.name)
        messages = _run_round_compaction(messages, round_ctx)
        current = visible_snapshots(messages)[skill.name][1]
        assert current["state"] == "unknown"
        assert current["reason"] == "snapshot_malformed_or_unreadable"
        assert "text" not in current
        size = len(messages)
        messages = _run_round_compaction(messages, round_ctx)
        assert len(messages) == size  # withdrawal is not repeated every round
    finally:
        extension_loader.unload_extension(skill.name)


# Source: witness_owner_words_merge_into_snapshot.py. After an interrupted
# pre-send tail, incoming owner words stay on their own ordinary wire row.

import queue
from copy import deepcopy
from types import SimpleNamespace

from tests.test_skill_context_snapshot import _write
from tests.test_skill_context_snapshot import publisher as _publisher_fixture

publisher = _publisher_fixture


def test_owner_message_after_interrupted_round_stays_outside_untrusted_snapshot_row(publisher, monkeypatch):
    from ouroboros import (
        context_compaction,  # noqa: F401 - public import establishes its leaf facade
        peer_roster,
    )
    from ouroboros.context_source_view import _obsolete_host_rows
    from ouroboros.llm_attempt import _physical_candidate
    from ouroboros.loop_round_limits import _CompactionRoundContext, _drain_incoming_messages, _run_round_compaction
    from ouroboros.context_budget import OWNER_WORDS_KIND
    from ouroboros.skill_context_snapshot import SNAPSHOT_KIND, visible_snapshots
    from ouroboros.transcript_prefix import observe_send

    root, skill, ctx, path = publisher
    monkeypatch.setattr(peer_roster, "maybe_append_roster_note", lambda *a: False)
    messages = [{"role": "system", "content": "Stable constitution"},
                {"role": "user", "content": "Actual owner request"},
                {"role": "assistant", "content": "Working on it"}]
    observe_send(ctx, messages, round_idx=1)          # 1. previous send recorded on the loop slot
    _write(path, "quota: weekly 40% used")
    round_ctx = _CompactionRoundContext(tools=SimpleNamespace(_ctx=ctx), drive_root=root,
        drive_logs=root / "logs", task_id="root", round_idx=2, event_queue=None, emit_progress=lambda x: None)
    messages = _run_round_compaction(messages, round_ctx)   # 2. changed note appended
    assert messages[-1]["_host_context_kind"] == SNAPSHOT_KIND
    # 3. interrupted call: nothing recorded. 4. next round drains an owner message.
    inbox = queue.Queue()
    inbox.put({"text": "STOP the deploy and revert the last commit now.", "msg_id": "m-1"})
    _drain_incoming_messages(messages, inbox, None, "root", None, set(), owner_ctx=ctx)
    tail = messages[-1]
    print("\nWITNESS tail kind:", tail.get("_host_context_kind"))
    print("WITNESS tail content:\n" + str(tail.get("content")))
    wire = _physical_candidate({"messages": deepcopy(messages), "tools": []})["messages"][-1]
    print("WITNESS wire last message role:", wire["role"], "| has owner words:",
          "STOP the deploy" in str(wire["content"]), "| has UNTRUSTED header:", "UNTRUSTED DATA" in str(wire["content"]))
    print("WITNESS visible_snapshots parses merged row:", "test-snapshot" in visible_snapshots(messages))
    print("WITNESS merged row index in _obsolete_host_rows:", (len(messages) - 1) in _obsolete_host_rows(messages))
    assert not ("UNTRUSTED DATA" in str(tail.get("content")) and "STOP the deploy" in str(tail.get("content"))), (
        "owner words were delivered inside the row headed 'UNTRUSTED DATA, not owner instructions'")
    assert "test-snapshot" in visible_snapshots(messages)
    assert tail.get("_host_context_kind") == OWNER_WORDS_KIND  # typed at delivery, never merged into the note
    assert wire["content"] == "[Message from my human]: STOP the deploy and revert the last commit now."
    assert len(ctx._owner_directives) == 1 and ctx._owner_directives[0]["content"] == (
        "STOP the deploy and revert the last commit now.")
    assert len(messages) - 1 not in _obsolete_host_rows(messages)
    messages = _run_round_compaction(messages, round_ctx)
    assert messages[-1] == tail  # no redundant note hides or rewrites the owner row
    from ouroboros.llm_claudexor import _request
    native = _request({"source": "codex", "resolved_model": "fixture"}, messages, [],
                      {"model_account_override": "", "_no_account_preference": True})
    assert native["messages"][-1] == wire


# Source: witness_renewal_cadence.py. Synthetic leases (420 s validity, 180 s
# margin, 120 s producer sweep) exercise real host renewal every 30 s.

from datetime import datetime, timedelta, timezone


def test_unchanged_text_renewal_frequency(publisher, monkeypatch):
    from ouroboros import skill_context_snapshot as scs
    from ouroboros.transcript_prefix import observe_send

    root, skill, ctx, path = publisher
    start = datetime.now(timezone.utc).replace(microsecond=0)
    clock = {"now": start}

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock["now"]

    monkeypatch.setattr(scs, "datetime", Clock)
    text = "x" * 3000  # an unchanged, mid-sized quota summary
    lease = {"observed": start, "valid": start + timedelta(seconds=420)}

    def write():
        path.write_text(json.dumps({"schema_version": 1, "observed_at": lease["observed"].isoformat(),
                                    "valid_until": lease["valid"].isoformat(), "text": text}), encoding="utf-8")

    write()
    messages, appended_at = [], []
    for step in range(0, 3600, 30):                       # one simulated hour, a round every 30 s
        clock["now"] = start + timedelta(seconds=step)
        if step % 120 == 0 and (lease["valid"] - clock["now"]).total_seconds() <= 180:
            lease = {"observed": clock["now"], "valid": clock["now"] + timedelta(seconds=420)}
            write()                                        # producer renews the unchanged text
        before = deepcopy(messages)
        appended = scs.maybe_append_skill_snapshots(ctx, messages, root)
        if appended:
            appended_at.append(step)
        assert messages[:len(before)] == before  # metadata renewal never rewrites sent history
        assert observe_send(ctx, messages, round_idx=step // 30) is None
    full_notes = [m for m in messages if m.get("_host_context_kind") == scs.SNAPSHOT_KIND and text in m["content"]]
    print("\nWITNESS appended at seconds:", appended_at)
    print("WITNESS full-text notes in one hour for unchanged text:", len(full_notes),
          "| chars appended:", sum(len(m["content"]) for m in full_notes))
    # Once the dates in view run out one note says so and names the current text; later
    # renewals add nothing until the conversation is next rebuilt (#2333).
    assert len(full_notes) == 1 and len(messages) == 2
    value = json.loads(messages[-1]["content"].split("\n", 1)[1])
    assert "text" not in value and (value["state"], value["reason"]) == ("outdated", "validity_ended")
    assert scs.obsolete_snapshot_positions(messages) == {0}
    from ouroboros.llm_attempt import _physical_candidate
    from ouroboros.llm_claudexor import _request
    for wire in (_physical_candidate({"messages": messages, "tools": []}),
            _request({"source": "codex", "resolved_model": "fixture"}, messages, [],
                     {"model_account_override": "", "_no_account_preference": True})):
        assert sum(text in m["content"] for m in wire["messages"]) == 1


# Source: witness_pinned_engine_passive_freshness.py and tagged query-contract
# witness. 4.0.x accepts only view; 4.1.x also accepts model/requestKind. Both
# support constraint_freshness but not resources_freshness. No live engine.


from starlette.testclient import TestClient

from tests.test_claudexor_quota_passive import (  # noqa: F401 - pytest fixture import
    ENGINE_FRESHNESS_QUOTA,
    QUOTA_URL,
    _app,
    passive_engine as _passive_engine_fixture,
)

passive_engine = _passive_engine_fixture


def _legacy(value):
    value = deepcopy(value)
    for snapshot in value["snapshots"]:
        snapshot.pop("availability", None)
        for constraint in snapshot["constraints"]:
            constraint.pop("freshness", None)
    return value


def _install_legacy_freshness_engine(engine, allowed_params):
    def hook(request):
        view = request.url.params.get("view")
        engine.state["errors"].pop("/v2/quota", None)
        if set(request.url.params) - set(allowed_params) or view not in (None, "resources", "constraint_freshness"):
            engine.state["errors"]["/v2/quota"] = (
                400, {"code": "invalid_request", "message": "view must be resources or constraint_freshness"})
        elif view == "constraint_freshness":
            engine.state["responses"]["/v2/quota"] = deepcopy(ENGINE_FRESHNESS_QUOTA)
        else:
            engine.state["responses"]["/v2/quota"] = _legacy(ENGINE_FRESHNESS_QUOTA)
    engine.state["hooks"]["/v2/quota"] = hook


@pytest.mark.parametrize("allowed_params", [("view",), ("view", "model", "requestKind")], ids=["4.0.x", "4.1.x"])
def test_passive_view_preserves_legacy_constraint_freshness_without_scope_claim(passive_engine, allowed_params):
    passive_engine.provision()
    _install_legacy_freshness_engine(passive_engine, allowed_params)
    with TestClient(_app()) as client:
        payload = client.get(QUOTA_URL).json()
    sent = [params for _, path, params in passive_engine.calls if path == "/v2/quota"]
    words = [c.get("freshness") for c in payload["quota"][0]["constraints"]]
    print("\nWITNESS quota GET selectors sent:", sent)
    print("WITNESS reads:", payload["reads"])
    print("WITNESS per-constraint freshness delivered:", words)
    print("WITNESS snapshot aggregate freshness:", payload["quota"][0]["freshness"])
    # Assert the property the passive contract and the Quotas skill rely on.
    assert words == ["stale", "fresh"], (
        "per-constraint freshness lost on the legacy engine: the weekly window that the engine"
        " reports fresh now inherits the stale aggregate")
    assert sent == [{"view": "resources_freshness", "requestKind": "text"}, {"view": "constraint_freshness"}]
    assert "availability_scope" not in payload


@pytest.mark.parametrize("failure", ["timeout", 500, "malformed"])
def test_freshness_compatibility_stops_after_non_selector_failure(failure):
    from ouroboros.claudexor_observations import read_quota
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    class Gateway:
        def __init__(self): self.calls = []
        def quota_state(self, **scope):
            self.calls.append(scope)
            if scope.get("view") == "resources_freshness":
                raise ClaudexorUnavailable("http_error", "unsupported view", status_code=400)
            assert scope == {"view": "constraint_freshness"}
            if failure == "malformed": return {"snapshots": None}
            raise ClaudexorUnavailable("http_error", "unknown", status_code=500 if failure == 500 else 0)
    gateway = Gateway()
    with pytest.raises(ClaudexorUnavailable): read_quota(gateway, model="exact", request_kind="image")
    assert len(gateway.calls) == 2  # no plain GET on malformed/5xx/unknown transport


def test_renewals_keep_visible_full_base_through_reclaim_and_restore_it_when_absent(publisher, monkeypatch):
    from datetime import timedelta
    from types import SimpleNamespace

    from ouroboros import context_compaction, peer_roster
    from ouroboros import skill_context_snapshot as scs
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.context_source_view import emergency_address_view
    from ouroboros.llm_attempt import _physical_candidate
    from ouroboros.llm_claudexor import _request
    from ouroboros.loop_round_limits import _CompactionRoundContext, _run_round_compaction
    from ouroboros.observability import read_blob_ref
    root, skill, ctx, path = publisher
    start = datetime.now(timezone.utc)
    clock = {"now": start}
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return clock["now"]
    monkeypatch.setattr(scs, "datetime", Clock)
    monkeypatch.setattr(peer_roster, "maybe_append_roster_note", lambda *a: False)
    messages = [{"role": "system", "content": "Stable governance"}, {"role": "user", "content": "Real original request"}]
    text = "Complete facts including unknown and units " * 300
    _write(path, text, observed=start, valid=start + timedelta(seconds=1))
    assert scs.maybe_append_skill_snapshots(ctx, messages, root)
    base = deepcopy(messages[-1])
    for step in range(2, 42, 2):
        clock["now"] = start + timedelta(seconds=step)
        _write(path, text, observed=clock["now"], valid=clock["now"] + timedelta(seconds=1))
        assert scs.maybe_append_skill_snapshots(ctx, messages, root) is (step == 2)  # one note, then nothing
        assert messages[2] == base
    assert len(messages) == 4 and scs.visible_snapshots(messages)[skill.name][1]["state"] == "outdated"
    # The real host-copy reclaim removes the outdated full text and keeps the note naming the current one.
    before = deepcopy(messages)
    request = ContextReclaimRequest(route_fp="fixture", round_id="7", transcript_sha256=
        context_compaction.context_reclaim_transcript_sha256(messages), measurement_basis="cold_estimate",
        measurement_density=3.0, reclaim_goal_tokens=1)
    reclaimed, receipt = emergency_address_view(messages, request, rung="host_copies",
        drive_root=root, task_id="root", measure_candidate=lambda rows: len(json.dumps(rows)))
    assert receipt.status == "applied"
    assert base not in reclaimed and scs.visible_snapshots(reclaimed)[skill.name][1]["state"] == "outdated"
    assert messages == before
    retained = json.loads(base["content"].split("\n", 1)[1])["source"]["retained"]
    assert json.loads(read_blob_ref(root, retained, expected_kind="txt"))["text"] == text
    # A rebuilt conversation with no full text in view shows the current one in full again.
    cold = [*reclaimed[:2], reclaimed[-1]]
    assert scs.visible_snapshots(cold)[skill.name][1]["state"] == "outdated"
    fresh_ctx = SimpleNamespace(task_id="root", task_metadata={}, budget_drive_root=str(root))
    round_ctx = _CompactionRoundContext(tools=SimpleNamespace(_ctx=fresh_ctx), drive_root=root,
        drive_logs=root / "logs", task_id="root", round_idx=8, event_queue=None, emit_progress=lambda x: None)
    restored = _run_round_compaction(cold, round_ctx)
    raw = json.loads(restored[-1]["content"].split("\n", 1)[1])
    assert raw["text"] == text and "text_ref" not in raw
    for wire in (_physical_candidate({"messages": restored, "tools": []}), _request(
            {"source": "codex", "resolved_model": "fixture"}, restored, [],
            {"model_account_override": "", "_no_account_preference": True})):
        assert text in wire["messages"][-1]["content"]
    # Expiry withdraws the claim once, and cannot keep a hidden reference fresh.
    clock["now"] = start + timedelta(seconds=42)
    assert scs.maybe_append_skill_snapshots(fresh_ctx, restored, root)
    assert scs.visible_snapshots(restored)[skill.name][1]["state"] == "stale"
    assert "text" not in scs.visible_snapshots(restored)[skill.name][1]
    assert not scs.maybe_append_skill_snapshots(fresh_ctx, restored, root)


def test_old_visible_snapshot_header_survives_a_cold_code_update(publisher):
    from ouroboros import skill_context_snapshot as scs
    root, skill, ctx, path = publisher
    _write(path)
    messages = []
    scs.maybe_append_skill_snapshots(ctx, messages, root)
    messages[-1]["content"] = messages[-1]["content"].replace(scs.SNAPSHOT_HEADER, scs._LEGACY_SNAPSHOT_HEADER)
    assert scs.visible_snapshots(messages)[skill.name][1]["text"] == "A"
    assert not scs.maybe_append_skill_snapshots(ctx, messages, root)


def test_multibyte_snapshot_over_file_bound_is_explicit_unknown_not_clipped(publisher):
    from ouroboros import skill_context_snapshot as scs
    root, skill, ctx, path = publisher
    _write(path, "🧠" * 12000)
    assert path.stat().st_size > scs.MAX_SNAPSHOT_BYTES
    messages = []
    assert scs.maybe_append_skill_snapshots(ctx, messages, root)
    current = scs.visible_snapshots(messages)[skill.name][1]
    assert current["state"] == "unknown" and current["reason"] == "snapshot_oversized"
    assert "text" not in current


# An isolated root task's drive goes through normal settlement after its
# publisher moved on; the exact source still reads from the canonical root.


def test_exact_snapshot_source_survives_normal_isolated_drive_prune(publisher):
    import time

    from ouroboros import headless, observability
    from ouroboros import skill_context_snapshot as scs
    from ouroboros.task_results import write_task_result
    canonical, skill, ctx, path = publisher
    task = "snapshot-root-task"
    drive = headless.prepare_task_drive(canonical, task, "empty")
    ctx.task_id, ctx.task_metadata = task, {"budget_drive_root": str(canonical)}
    _write(path, "First exact snapshot")
    original = path.read_text(encoding="utf-8")
    messages = []
    assert scs.maybe_append_skill_snapshots(ctx, messages, drive)
    source = scs.visible_snapshots(messages)[skill.name][1]["source"]["retained"]
    observability.persist_call(drive, task_id=task, call_id="request", call_type="llm_request",
                               payload={"messages": messages})
    _write(path, "Later periodic publisher update")
    write_task_result(drive, task, "completed", result="answer", artifact_status="ready")
    headless.copy_child_task_result(canonical, {"id": task, "drive_root": str(drive)})
    assert headless.retry_child_task_refs(canonical, drive, task)["child_ref_promotion"]["status"] == "complete"
    report = headless.prune_headless_task_drives(canonical, retention_days=7, now=time.time() + 8 * 86400,
                                                 live=lambda _task: False)
    assert any(row["task_id"] == task for row in report["pruned"]) and not drive.exists()
    assert observability.read_call_payload(canonical, task_id=task, call_id="request")[1]["messages"] == messages
    assert observability.read_blob_ref(canonical, source, expected_kind="txt") == original


def test_route_health_reads_only_the_exact_default_subject_and_never_the_roster():
    """#244: no roster read guesses that a null subject is `<harness>-default`."""
    from ouroboros.subagent_route_health import _exhausted_window

    class Gateway:
        def __init__(self, subject):
            self.subject = subject

        def credential_profiles(self):
            raise AssertionError("route health must not read the roster to guess a quota subject")

        def quota_state(self, **scope):
            return {"snapshots": [{"subject": {"harness": "claude", "subject_id": self.subject},
                                   "freshness": "fresh",
                                   "availability": {"state": "exhausted", "resets_at": "2099-01-01T00:00:00Z"}}],
                    "availability_scope": {"model": None, "requestKind": "text", "basis": "quota_block"}}

    assert _exhausted_window(Gateway(None), "claude", pinned_profile="claude-default") == (False, "")
    assert _exhausted_window(Gateway("claude-default"), "claude", pinned_profile="claude-default") == (
        True, "2099-01-01T00:00:00Z")
