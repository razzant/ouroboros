"""Bounded review regressions, retained from independent synthetic witnesses."""
from __future__ import annotations

# Regressions derived from independent witnesses; no live-provider calls.
import json

import pytest


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

from copy import deepcopy

from tests.test_skill_context_snapshot import _write
from tests.test_skill_context_snapshot import publisher as _publisher_fixture

publisher = _publisher_fixture


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
