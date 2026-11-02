"""The coordination poll's purity: a poll answers, and changes nothing it could have been asked again.

This module once pinned the task-acceptance admission rails (the launch floor,
the acceptance wallet and its paid-cycle cap). Those rails are gone with the
acceptance machine. What stays is the poll every subagent bootstrap and nanny
wake runs (``delegate_supervision.coordination_live_context``): it writes
nothing of its own and inherits only the canonical usage-ledger reader's own
bounded maintenance — today: the torn-tail quarantine after a single crash
mid-append, pinned in that single-crash shape, the empty `state/` directory the
reader's lock lives in on a never-initialized root, and removal of a stale
`usage_attempts.lock` older than the reader's 90 s stale window
(`usage_ledger._locked` → `platform_layer.acquire_exclusive_file_lock`, whose
stale-age branch unlinks the lock file and retries) — each pinned by its own
regression below; every ledger state, absent included, goes through the
canonical reader. ``_KnownPricedLLM`` stays importable for the suites that use it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tests.test_acceptance_delivery import _EpisodeLLM
from tests._store_journal_shared import tree_state
from tests._store_shared import result_signature


class _KnownPricedLLM(_EpisodeLLM):
    """The delivery suite's scripted reviewer whose provider reports a FINAL price
    for each send: KNOWN spend, the only money a limit decides on (#1487). The
    suite's own replies carry no usable price, so they stay unresolved exposure."""

    def chat(self, **kwargs):
        from ouroboros import usage_accounting as ua

        request = ua.AttemptRequest(reservation_usd=self.reservation_usd, drive_root=self.drive_root,
                                    model="openai/fake-reviewer", provider="openrouter")
        return ua.execute_physical_attempt(request, lambda: self._reply(kwargs),
                                           extractor=lambda reply: (reply[1], self.reservation_usd, True))


def _deadline_root_ctx(monkeypatch, tmp_path, *, seconds_left, policy="adaptive"):
    """A REAL root task on a real deadline: the 120 s grace as the WHOLE reserve
    (pct 0) and one earlier event row in its events stream. Nothing is patched:
    the same contract production reads. ``max_improvement_passes`` is left unset,
    as shipped."""
    import queue
    from datetime import timedelta

    from ouroboros import task_pacing
    from ouroboros.contracts.task_contract import build_task_contract
    from ouroboros.deadline_utils import utc_now
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.utils import append_jsonl

    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "120")  # the whole reserve (pct 0)
    monkeypatch.delenv("OUROBOROS_TASK_ABS_CEILING_SEC", raising=False)
    now = utc_now()
    contract = build_task_contract({
        "deadline_at": (now + timedelta(seconds=seconds_left)).isoformat(),
        "budget_profile": {"improvement_policy": policy, "reserve_finalization_pct": 0},
    })
    metadata = {
        "root_task_id": "root-floor-band", "delegation_role": "root",
        "budget_drive_root": str(tmp_path), "task_contract": contract,
        "created_at": (now - timedelta(seconds=60)).isoformat(),
        "deadline_at": contract["deadline_at"],
    }
    ctx = SimpleNamespace(
        task_id="root-floor-band", root_task_id="root-floor-band", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_contract=contract, task_metadata=metadata,
        pending_events=[], event_queue=queue.Queue(),
    )
    write_task_result(tmp_path, "root-floor-band", STATUS_RUNNING, root_task_id="root-floor-band",
                      delegation_role="root", task_contract=contract)
    events = task_pacing.acceptance_timing_events_path(ctx)
    events.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(events, {"type": "task_received", "task_id": "root-floor-band"})  # an earlier row
    return ctx, events


# ---------------------------------------------------------------------------
# Purity: a poll answers, and changes nothing it could have been asked again.
# ---------------------------------------------------------------------------


def test_an_empty_store_polls_known_zero_through_the_reader_and_writes_nothing(tmp_path):
    """Owner R56 (2): on a root whose usage store holds no attempt the
    coordination poll answers a KNOWN zero settled spend THROUGH the canonical
    reader (no fast path answers ahead of it) and changes nothing at all. The
    store exists as in production: the supervisor's lifecycle job creates it
    before any worker (and so any poll) exists. Pinned as the EXACT observed
    tree, byte for byte, across two polls (each store as its rows and any
    write's trace: a WAL reader's own index and log files are no write,
    ``tree_state``); no events row, no ctx attribute.
    The relayed numbers are the reader's own projection, key by key."""
    import queue

    from ouroboros import delegate_custody as custody
    from ouroboros import store_import, usage_store
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.usage_accounting import usage_breakdown

    ctx = SimpleNamespace(
        task_id="root-empty", root_task_id="root-empty", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={
            "root_task_id": "root-empty", "budget_drive_root": str(tmp_path)},
        pending_events=[], event_queue=queue.Queue(),
    )
    root = custody.custody_root(ctx)
    assert sorted(root.rglob("*")) == []  # genuinely fresh
    usage_store.migrate_from_journal(root)  # the lifecycle jobs, before any worker
    store_import.ensure_current_store(root)
    attrs_before = set(vars(ctx))
    known_zero = {
        "state": "known", "settled_usd": 0.0, "accounted_usd": 0.0,
        "cost_final": True, "unknown_unmetered": 0, "integrity_degraded": False,
    }

    def _tree():
        return tree_state(root)

    before = _tree()
    events_before = custody.event_log_path(root).read_bytes()
    for _ in range(2):
        assert coordination_live_context(ctx)["settled_spend"] == known_zero
        assert _tree() == before  # the poll changed nothing
    assert custody.event_log_path(root).read_bytes() == events_before  # no events row
    assert set(vars(ctx)) == attrs_before
    assert ctx.event_queue.empty() and ctx.pending_events == []

    # The answer IS the reader's: the same projection, key by key.
    projection = usage_breakdown(root, root_task_id="root-empty")
    assert {key: projection[key] for key in known_zero if key != "state"} == {
        key: value for key, value in known_zero.items() if key != "state"}


@pytest.mark.parametrize("enforced", [False, True])
def test_a_stale_name_lock_is_the_readers_removal_and_the_poll_stays_known_zero(
        monkeypatch, tmp_path, enforced):
    """The maintenance a poll inherits from the canonical reader is bounded:
    on an installation without kernel file locks (the store's ``name`` tier)
    every store access runs under the money name lock, so a
    `state/usage_attempts.lock` left behind by a dead writer and older than the
    lock's 90 s stale window is REMOVED by the reader's own lock acquisition
    (`usage_ledger._named_lock` → `platform_layer.acquire_exclusive_file_lock`,
    whose stale-age branch unlinks the file and retries), never by the poll,
    which writes nothing of its own. On the ``enforced`` tier SQLite's own
    locks serialize the store and the same file is inert: left untouched.
    Pinned as the EXACT observed tree; the answer is the reader's own known
    zero; no events row; no ctx attribute; a second poll changes nothing."""
    import os
    import queue

    from ouroboros import delegate_custody as custody
    from ouroboros import platform_layer, store_import, usage_store
    from ouroboros.delegate_supervision import coordination_live_context

    ctx = SimpleNamespace(
        task_id="root-stale-lock", root_task_id="root-stale-lock", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={
            "root_task_id": "root-stale-lock", "budget_drive_root": str(tmp_path)},
        pending_events=[], event_queue=queue.Queue(),
    )
    root = custody.custody_root(ctx)
    monkeypatch.setattr(platform_layer, "kernel_file_locks_enforced", lambda _path, *, known=None: enforced)
    usage_store.migrate_from_journal(root)  # the lifecycle jobs decide the tier once
    store_import.ensure_current_store(root)
    usage_store.forget(root)

    def _tree():
        return tree_state(root)

    migrated = _tree()
    stale_lock = root / "state" / "usage_attempts.lock"
    stale_lock.write_text("pid=999999 ts=0\n", encoding="utf-8")
    os.utime(stale_lock, (0, 0))  # a dead writer's lock, aged to the epoch: far past 90 s
    with_stale = _tree()
    events_before = custody.event_log_path(root).read_bytes()
    attrs_before = set(vars(ctx))
    known_zero = {
        "state": "known", "settled_usd": 0.0, "accounted_usd": 0.0,
        "cost_final": True, "unknown_unmetered": 0, "integrity_degraded": False,
    }

    assert coordination_live_context(ctx)["settled_spend"] == known_zero
    after_first = _tree()
    if enforced:
        assert after_first == with_stale  # inert on the enforced tier
    else:
        assert not stale_lock.exists()  # the reader's stale-age branch removed it, then released its own
        assert after_first == migrated
    assert custody.event_log_path(root).read_bytes() == events_before  # no events row

    assert coordination_live_context(ctx)["settled_spend"] == known_zero
    assert _tree() == after_first  # the second poll changed nothing
    assert set(vars(ctx)) == attrs_before
    assert ctx.event_queue.empty() and ctx.pending_events == []


def test_a_directory_at_the_ledger_path_is_the_readers_verdict_not_known_zero(tmp_path):
    """Every ledger state goes through the canonical reader — there is no fast
    path ahead of it — so a DIRECTORY at the ledger path reports exactly the
    reader's own typed refusal (`UsageAccountingError` → `state: unknown`, no
    confident $0), on a repeat poll too — pinned against the reader called
    directly, so the fact cannot drift from the verdict it relays."""
    import queue

    from ouroboros import delegate_custody as custody
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.usage_accounting import usage_breakdown
    from ouroboros.usage_ledger import LEDGER_REL, UsageAccountingError

    ctx = SimpleNamespace(
        task_id="root-dir", root_task_id="root-dir", drive_root=tmp_path,
        budget_drive_root=str(tmp_path), task_metadata={
            "root_task_id": "root-dir", "budget_drive_root": str(tmp_path)},
        pending_events=[], event_queue=queue.Queue(),
    )
    root = custody.custody_root(ctx)
    (root / LEDGER_REL).mkdir(parents=True)
    with pytest.raises(UsageAccountingError):  # the reader's own verdict on a directory
        usage_breakdown(root, root_task_id="root-dir")

    for _ in range(2):
        spend = coordination_live_context(ctx)["settled_spend"]
        assert spend["state"] == "unknown" and spend["reason"] == "UsageAccountingError"
        assert spend["settled_usd"] is None and spend["cost_final"] is None


def _legacy_settings(monkeypatch, tmp_path):
    """Point config at a settings file whose retired auto-Low pair a
    `load_settings()` WOULD normalize and persist — the write this batch must
    keep out of every read path. Returns the path and its (bytes, mtime_ns)."""
    import json as _json

    from ouroboros import config as cfg

    settings_path = tmp_path / "settings-legacy.json"
    settings_path.write_text(_json.dumps({"OUROBOROS_CONTEXT_MODE": "low"}), encoding="utf-8")
    monkeypatch.setattr(cfg, "SETTINGS_PATH", settings_path, raising=True)
    return settings_path, (settings_path.read_bytes(), settings_path.stat().st_mtime_ns)


def test_the_whole_coordination_poll_writes_nothing_and_reports_an_unlatched_task(
        monkeypatch, tmp_path):
    """The poll every subagent bootstrap and nanny wake runs writes NOTHING on a
    healthy tree, in ALL of its facts (the bounded maintenance it inherits from
    the canonical ledger reader — today the empty `state/` directory on a
    never-initialized root, the stale-lock removal, and the usage ledger's
    torn-tail quarantine — is pinned by the absent-ledger and stale-lock tests
    above and the torn-ledger test below). On a task carrying an `adaptive`
    profile (the `until_deadline` alias is retired) AND no
    `created_at`/`started_at` (both writes armed), with the grace env ABSENT and
    a legacy context-mode settings file (the settings write armed): the settings
    bytes and mtime, the events stream, the task result and the event queue are
    byte-identical, no ctx attribute appears, and the `time` fact honestly
    answers `not_set` for a task whose window nobody has latched yet. The two
    controls below fire both writes from the paths that OWN them, so nothing
    passes vacuously."""
    from ouroboros import config as cfg, task_pacing
    from ouroboros.delegate_supervision import coordination_live_context

    ctx, events = _deadline_root_ctx(
        monkeypatch, tmp_path, seconds_left=620, policy="adaptive")  # 7.0 ABI: until_deadline retired
    del ctx.task_metadata["created_at"]  # metadata-poor: no anchor without a WRITE
    monkeypatch.delenv("OUROBOROS_FINALIZATION_GRACE_SEC", raising=False)
    settings_path, settings_before = _legacy_settings(monkeypatch, tmp_path)
    from ouroboros import store_import, usage_store

    usage_store.migrate_from_journal(tmp_path)  # the lifecycle jobs, before any worker
    store_import.ensure_current_store(tmp_path)
    before = (events.read_bytes(), result_signature(tmp_path, "root-floor-band"))
    attrs_before = set(vars(ctx))

    for _ in range(2):
        live = coordination_live_context(ctx)
        assert live["time"]["state"] == "not_set"  # no latched anchor, and none taken
        assert live["time"] == {"state": "not_set", "remaining_sec": None,
                                "reserve_sec": None, "inside_reserve": None, "expired": None}
    assert (settings_path.read_bytes(), settings_path.stat().st_mtime_ns) == settings_before
    assert (events.read_bytes(), result_signature(tmp_path, "root-floor-band")) == before
    assert ctx.event_queue.empty() and ctx.pending_events == []
    assert set(vars(ctx)) == attrs_before  # the EXACT attribute set: nothing latched, nothing cached
    assert not hasattr(ctx, "_time_budget_started_at")  # the one latch a poll could take

    # Controls: each armed write really is reachable from its owning path.
    assert cfg.load_settings(migrate=True).get("OUROBOROS_CONTEXT_MODE") == "max"
    assert settings_path.read_bytes() != settings_before[0]  # the boot's migrating read persists
    assert task_pacing.resolve_budget_profile(ctx)["improvement_policy"] == "adaptive"
    # (the third control of the upstream test — the alias deprecation row and
    # its `_acceptance_pacing_deprecation_emitted` latch — is void: the
    # ``until_deadline`` alias and its emission are retired, 7.0 ABI Q10=A, and
    # nothing in the tree writes that attribute any more)
    assert task_pacing.build_budget_snapshot(ctx).has_deadline is True
    assert getattr(ctx, "_time_budget_started_at", None) is not None


def test_a_single_crash_torn_journal_gets_the_quarantine_its_one_time_import_performs(
        monkeypatch, tmp_path):
    """The one CONTENT write among the bounded maintenance a poll inherits from
    the canonical reader: on a root whose store was exported back to a journal
    (the downgrade step) and whose older release then crashed mid-append, the
    first read runs the one-time import, and the import repairs the journal
    exactly as every journal reader did — truncate to the intact prefix, one
    quarantine row holding the torn bytes verbatim, one
    `usage_ledger_tail_quarantined` event — before retiring it as
    ``usage_attempts.jsonl.imported``; the poll then reports the survivors as
    degraded integrity rather than final cost. Nothing else moves: the
    settings bytes and mtime, every earlier event row, the task result, the
    queue and the ctx attributes are unchanged, and a second poll writes
    nothing at all."""
    import base64

    from ouroboros import delegate_custody as custody, usage_accounting as ua, usage_store
    from ouroboros.delegate_supervision import coordination_live_context
    from ouroboros.usage_ledger import LEDGER_REL, QUARANTINE_REL

    ctx, events = _deadline_root_ctx(monkeypatch, tmp_path, seconds_left=620)
    settings_path, settings_before = _legacy_settings(monkeypatch, tmp_path)
    # Seed the money the POLL itself will read (the custody root it resolves),
    # through the real attempt path, then export it as a journal.
    root = custody.custody_root(ctx)
    with ua.usage_scope(ua.UsageScope(
            drive_root=root, task_id="root-floor-band", root_task_id="root-floor-band")):
        ua.execute_physical_attempt(
            ua.AttemptRequest(model="local-review-test", provider="local", reservation_usd=0.0),
            lambda: ({"content": "seed"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}))
    usage_store.export_journal(root)
    ledger, quarantine = root / LEDGER_REL, root / QUARANTINE_REL
    intact, torn = ledger.read_bytes(), b'{"seq": 99, "attempt_id": "half-writ'
    ledger.write_bytes(intact + torn)  # the process died mid-append
    before = (events.read_bytes(), result_signature(tmp_path, "root-floor-band"), set(vars(ctx)))

    live = coordination_live_context(ctx)

    spend = live["settled_spend"]  # it ANSWERED from the repaired prefix, honestly
    assert "reason" not in spend  # no read error: the repair is not a failure path
    assert spend["state"] == "partial" and spend["integrity_degraded"] is True
    assert spend["cost_final"] is False  # a quarantined tail never claims final cost
    assert ledger.read_bytes() == intact  # imported and kept in place, truncated to the intact prefix
    rows = [json.loads(line) for line in quarantine.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and base64.b64decode(rows[0]["raw_base64"]) == torn
    assert rows[0]["source"] == str(ledger)
    after = events.read_bytes()
    assert after.startswith(before[0])  # every earlier event row is untouched
    added = [json.loads(line) for line in after[len(before[0]):].decode("utf-8").splitlines()]
    assert [row["type"] for row in added] == ["usage_ledger_tail_quarantined"]
    assert (settings_path.read_bytes(), settings_path.stat().st_mtime_ns) == settings_before
    assert result_signature(tmp_path, "root-floor-band") == before[1] and set(vars(ctx)) == before[2]
    assert ctx.event_queue.empty() and ctx.pending_events == []

    # The imported store is the healthy case again: the second poll writes nothing.
    steady = (ledger.read_bytes(), quarantine.read_bytes(), events.read_bytes(),
              (root / usage_store.STORE_REL).read_bytes())
    coordination_live_context(ctx)
    assert (ledger.read_bytes(), quarantine.read_bytes(), events.read_bytes(),
            (root / usage_store.STORE_REL).read_bytes()) == steady


def test_reading_the_finalization_grace_never_touches_the_settings_file(monkeypatch, tmp_path):
    """`get_finalization_grace_sec` is env → the supplied settings mapping →
    the shipped default. With the env absent it must answer from the default
    without reading settings from disk."""
    from ouroboros import config as cfg

    monkeypatch.delenv("OUROBOROS_FINALIZATION_GRACE_SEC", raising=False)
    settings_path, before = _legacy_settings(monkeypatch, tmp_path)
    assert cfg.get_finalization_grace_sec() == cfg.FINALIZATION_GRACE_DEFAULT_SEC
    assert cfg.get_finalization_grace_sec({"OUROBOROS_FINALIZATION_GRACE_SEC": "45"}) == 45
    monkeypatch.setenv("OUROBOROS_FINALIZATION_GRACE_SEC", "9999")
    assert cfg.get_finalization_grace_sec() == 300  # the clamp is unchanged
    assert (settings_path.read_bytes(), settings_path.stat().st_mtime_ns) == before
    # Control: the file really is one the boot's migrating read would rewrite.
    assert cfg.load_settings(migrate=True).get("OUROBOROS_CONTEXT_MODE") == "max"
    assert settings_path.read_bytes() != before[0]
