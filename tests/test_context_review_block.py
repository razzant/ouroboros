"""The ``## Review`` block: the review POOL a task's settings snapshot serves, in both context paths (#1548, PR-3).

It is read through the builder every review surface uses (``review_pool_slots``), so the mind sees the pool its own
commit, plan and acceptance review would run — never a last-execution receipt and never settings saved after the
task started. The pool is the catalog's marked rows (contract §1.3): there is no default panel any more.
"""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

from ouroboros.configured_subagents import SUBAGENTS_SETTING, roster_handles
from ouroboros.settings_integrity import task_settings_snapshot
from ouroboros.subagent_runtime import (
    COST_UNKNOWN_HINT,
    SESSION_SEAT_COST_HINT,
    review_facts_block,
    review_records_block,
)
from tests.test_doc_context import _make_env_and_memory

_HEADER = "## Review\n\n"


def _row(subagent_id: str, target: str, *, kind: str = "api_model", effort: str = "", marked: bool = True,
         **extra) -> dict:
    row = {"subagent_id": subagent_id, "recommended_use": "Reviews diffs.",
           "route": {"kind": kind, "target_id": target}, **extra}
    if effort:
        row["effort"] = effort
    if marked:
        row["review_eligible"] = True
    return row


def _roster(*rows: dict, enabled: bool = True) -> str:
    return json.dumps({"enabled": enabled, "items": list(rows)})


_POOL = _roster(
    _row("critic-key", "openai/gpt-5.6-terra", effort="medium"),
    _row("packet-key", "openai/gpt-5.6-sol", delivery="packet"),
    _row("session-key", "codex=gpt-5.6-sol", kind="agent_session", effort="high"),
    _row("helper-key", "openai/gpt-5.6-luna", marked=False),  # a free helper is never the panel
)


def _decode(text: str) -> dict:
    """The block inside a context part; it may sit between other sections."""
    assert _HEADER in text, "the ## Review block is missing"
    block, _end = json.JSONDecoder().raw_decode(text.split(_HEADER, 1)[1])
    return block


def _block(snapshot=None) -> tuple[dict, str]:
    text = review_facts_block(snapshot)
    assert text.startswith(_HEADER)
    return _decode(text), text


_RECORDS_HEADER = "## Review records\n\n"


def _records(tmp_path, task_id: str = "task-1") -> dict:
    """The changing-part block: this task's recent ledger records, never inside the cached prefix."""
    text = review_records_block(drive_root=tmp_path, task_id=task_id)
    assert text.startswith(_RECORDS_HEADER)
    block, _end = json.JSONDecoder().raw_decode(text.split(_RECORDS_HEADER, 1)[1])
    return block


def _ledger(monkeypatch, recent_records, archived_segments_exist=lambda drive_root: False) -> None:
    module = types.ModuleType("ouroboros.review_ledger")
    module.recent_records = recent_records
    module.archived_segments_exist = archived_segments_exist
    monkeypatch.setitem(sys.modules, "ouroboros.review_ledger", module)


@pytest.fixture(autouse=True)
def _clean_review_plane(monkeypatch):
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL",
                "OUROBOROS_REVIEW_ENFORCEMENT", "OUROBOROS_REVIEWER_SLOTS", SUBAGENTS_SETTING):
        monkeypatch.delenv(key, raising=False)
    # No ledger reader is the graceful case; a test that needs records installs one.
    monkeypatch.setitem(sys.modules, "ouroboros.review_ledger", None)


def _handles() -> dict[str, str]:
    from ouroboros.configured_subagents import parse_configured_subagents

    return roster_handles(parse_configured_subagents(os.environ[SUBAGENTS_SETTING]), dict(os.environ))


def test_the_pool_is_the_marked_rows_named_by_their_catalog_handle(tmp_path, monkeypatch):
    monkeypatch.setenv(SUBAGENTS_SETTING, _POOL)
    block, text = _block()
    handles = _handles()

    assert (block["source"], block["error"], block["pool_empty"]) == ("structured", "", False)
    first, second, third = block["pool"]
    # seat_id is the stored row id (the identity the records carry); subagent_id
    # is the roster handle ``## Available subagents`` shows.
    assert first == {"seat_id": "critic-key", "subagent_id": handles["critic-key"], "model": "openai/gpt-5.6-terra",
                     "effort": "medium", "delivery": "native", "cost_hint": first["cost_hint"]}
    assert second == {"seat_id": "packet-key", "subagent_id": handles["packet-key"], "model": "openai/gpt-5.6-sol",
                      "effort": "high", "delivery": "packet", "cost_hint": second["cost_hint"]}
    assert third == {"seat_id": "session-key", "subagent_id": handles["session-key"], "model": "codex=gpt-5.6-sol",
                     "effort": "high", "delivery": "session", "cost_hint": SESSION_SEAT_COST_HINT}
    assert "helper-key" not in text and "gpt-5.6-luna" not in text, "an unmarked row is not a reviewer"
    # An api seat's hint is the wave's own estimate or an honest unknown — never a price table.
    for row in (first, second):
        assert row["cost_hint"] == COST_UNKNOWN_HINT or row["cost_hint"].startswith("≈$")
    # The preflight and /review seat whichever enabled catalog row is named per call
    # (decision 3A): the block names that rule, not a second panel.
    assert "catalog row" in block["surfaces"]["preflight"] and "Main" in block["surfaces"]["system_review"]
    assert block["omitted"] == {"rows": 0} and "recent_records" not in block
    assert block["full_source"] == {"pool": "GET /api/review-pool", "records": "## Review records"}
    assert _records(tmp_path) == {"recent_records": [], "omitted": {"records": 0}, "full_source": "state/review_ledger/"}


def test_the_cost_hint_of_a_cold_seat_never_waits_on_the_provider_catalog(tmp_path, monkeypatch):
    """Context assembly is a hot path (every round builds it): an api seat whose window this
    process has not measured yet is priced from the evidence already held — the full window
    when there is none — and never through OpenRouter's live ``/models`` catalog. (The
    measurement used to reach `LLMClient._fetch_openrouter_capabilities`, a 5-second network
    wait per unevidenced seat, and left the catalog in the process caches for every later send.)"""
    from ouroboros.llm import LLMClient

    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(
        _row("cold-key", f"openai/never-measured-{tmp_path.name.lower()}", effort="medium")))
    monkeypatch.setattr(LLMClient, "_fetch_openrouter_capabilities",
                        classmethod(lambda cls: pytest.fail("the ## Review block reached the live provider catalog")))

    block, _text = _block()

    (seat,) = block["pool"]
    assert seat["seat_id"] == "cold-key"
    assert seat["cost_hint"] == COST_UNKNOWN_HINT or seat["cost_hint"].startswith("≈$")


def test_the_pool_ignores_the_catalog_switch_and_reads_only_enabled_rows(monkeypatch):
    # Delegation off (``enabled: false``) does not switch review off (F6); a
    # row's own ``enabled: false`` does take it out of the pool.
    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(
        _row("critic-key", "openai/gpt-5.6-terra", effort="medium"),
        _row("retired-key", "openai/gpt-5.6-sol", enabled=False),
        enabled=False,
    ))
    block, _ = _block()
    assert block["source"] == "structured"
    assert [row["seat_id"] for row in block["pool"]] == ["critic-key"]


def test_a_catalog_with_no_marked_row_is_an_empty_pool_a_loud_fact_not_a_default_panel(monkeypatch):
    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(_row("helper-key", "openai/gpt-5.6-luna", marked=False)))
    block, text = _block()

    assert (block["source"], block["error"], block["pool"], block["pool_empty"]) == ("empty", "", [], True)
    assert "default" not in (block["source"], text.split('"rule"')[0]), "no shipped panel is implied"
    assert block["rule"] and block["surfaces"]["commit_gate"].startswith("every pool row")
    # No catalog at all is a never-configured install: the read seam mints the factory
    # rows (`factory_review_rows`), so the block shows that pool — not an empty one.
    monkeypatch.delenv(SUBAGENTS_SETTING)
    never_configured, _ = _block()
    assert (never_configured["source"], never_configured["pool_empty"]) == ("structured", False)
    assert [row["seat_id"] for row in never_configured["pool"]] == ["review-1", "review-2", "review-3"]


def test_the_seats_without_credentials_are_a_block_fact_and_every_seat_is_the_loud_one(monkeypatch):
    """VD3-08: the block names the api seats whose model this install holds no credentials for
    (the same fact ``GET /api/review-pool`` carries as ``pool_without_credentials`` and Settings
    shows); a session seat logs in itself and is never listed; when it is every seat of the pool
    the block says so in one loud key. Funded, neither key appears."""
    from ouroboros import provider_models

    monkeypatch.setenv(SUBAGENTS_SETTING, _POOL)
    handles = _handles()
    monkeypatch.setattr(provider_models, "model_has_credentials_in_settings",
                        lambda model, settings: model != "openai/gpt-5.6-terra")
    some, _ = _block()
    assert some["pool_without_credentials"] == [handles["critic-key"]]
    assert "no_pool_row_has_credentials" not in some, "the session seat and a funded api seat still answer"

    monkeypatch.setattr(provider_models, "model_has_credentials_in_settings", lambda model, settings: False)
    unfunded, _ = _block()
    assert unfunded["pool_without_credentials"] == [handles["critic-key"], handles["packet-key"]]
    assert "no_pool_row_has_credentials" not in unfunded, "the session seat logs in itself"

    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(_row("critic-key", "openai/gpt-5.6-terra"),
                                                   _row("packet-key", "openai/gpt-5.6-sol", delivery="packet")))
    api_only, _ = _block()
    assert api_only["no_pool_row_has_credentials"] is True
    assert len(api_only["pool_without_credentials"]) == len(api_only["pool"]) == 2

    monkeypatch.setattr(provider_models, "model_has_credentials_in_settings", lambda model, settings: True)
    funded, _ = _block()
    assert "pool_without_credentials" not in funded and "no_pool_row_has_credentials" not in funded


def test_an_invalid_catalog_is_an_error_with_an_empty_pool_not_an_absent_one(monkeypatch):
    from ouroboros.reviewer_slot_config import review_pool_state

    monkeypatch.setenv(SUBAGENTS_SETTING, '{"enabled": true, "items": [{"subagent_id": "x"}]}')
    block, _ = _block()

    assert block["source"] == "error"
    assert block["error"] == review_pool_state(os.environ[SUBAGENTS_SETTING])["error"] and block["error"]
    assert (block["pool"], block["pool_empty"]) == ([], False)
    assert block["rule"] and block["surfaces"]["plan_review"] == "every pool row"


def test_the_block_never_reads_the_last_execution(monkeypatch):
    from ouroboros import reviewer_slot_config

    def forbidden(*_args, **_kwargs):
        raise AssertionError("## Review must not read reviewer_slots_last")

    monkeypatch.setattr(reviewer_slot_config, "reviewer_slot_last_executions", forbidden)
    monkeypatch.setattr(reviewer_slot_config, "_last_execution_path", forbidden)
    monkeypatch.setenv(SUBAGENTS_SETTING, _POOL)
    block, text = _block()
    assert block["source"] == "structured" and "reviewer_slots_last" not in text


def test_the_block_reads_the_task_snapshot_not_settings_saved_after_the_task_started(tmp_path, monkeypatch):
    from ouroboros import context
    from ouroboros.config import load_settings
    from ouroboros.settings_integrity import task_settings_scope

    env, memory = _make_env_and_memory(tmp_path)
    frozen = _roster(_row("critic-key", "openai/snapshot-model"))
    # A task snapshot carries the document AND its projected environment as
    # they were at task start; the catalog is a document key.
    snapshot = task_settings_snapshot({**load_settings(), SUBAGENTS_SETTING: frozen},
                                      {**os.environ, SUBAGENTS_SETTING: frozen})
    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(_row("critic-key", "openai/live-model")))
    with task_settings_scope(snapshot):
        core = context._capture_context_core(env, memory, {"id": "task-1", "type": "task", "text": "w"}, None, None)

    assert [row["model"] for row in _decode(core.semi_stable_text)["pool"]] == ["openai/snapshot-model"]
    assert "openai/live-model" not in core.semi_stable_text
    assert _block(snapshot)[0]["pool"][0]["model"] == "openai/snapshot-model"
    assert _block()[0]["pool"][0]["model"] == "openai/live-model"


def test_both_context_paths_carry_the_block(tmp_path, monkeypatch):
    from ouroboros import context

    monkeypatch.setenv(SUBAGENTS_SETTING, _POOL)
    env, memory = _make_env_and_memory(tmp_path)
    shared = context._capture_context_core(env, memory, {"id": "root", "type": "task", "text": "w"}, None, None)
    declared = context._capture_context_core(env, memory, {
        "id": "child", "type": "task", "delegation_role": "subagent", "text": "Q",
        "configured_subagent": {"route": {"kind": "api_model"}}, "task_contract": {"input_sources": "declared"},
    }, None, None)

    assert _decode(shared.semi_stable_text)["source"] == "structured"
    assert _decode(declared.semi_stable_text) == _decode(shared.semi_stable_text)
    assert _HEADER not in shared.dynamic_text + declared.dynamic_text
    # The task's records are a changing fact: both paths carry them in the dynamic part only.
    assert _RECORDS_HEADER in shared.dynamic_text and _RECORDS_HEADER in declared.dynamic_text
    assert _RECORDS_HEADER not in shared.semi_stable_text + declared.semi_stable_text


def test_root_and_child_acceptance_differ_and_a_seat_id_or_handle_is_the_child_selector(monkeypatch):
    from ouroboros.reviewer_slot_config import child_acceptance_slots, review_pool_slots

    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(
        _row("sol-key", "openai/gpt-5.6-sol"), _row("terra-key", "openai/gpt-5.6-terra"),
    ))
    block, _ = _block()
    acceptance = block["surfaces"]["task_acceptance"]
    slots = review_pool_slots()

    assert acceptance["root"] != acceptance["child"] and "name one" in acceptance["child"]
    assert child_acceptance_slots(slots)[1]["reason"] == "reviewer_selection_required"
    for seat in block["pool"]:
        for selector in (seat["seat_id"], seat["subagent_id"]):
            chosen, refusal = child_acceptance_slots(slots, selector)
            assert not refusal and [slot.slot_id for slot in chosen] == [seat["seat_id"]]


@pytest.mark.parametrize("enforcement, mode, blocks, opening", [
    ("blocking", "pro", True, "Blocking:"),
    ("advisory", "pro", False, "Advisory:"),
    ("blocking", "cyber_pro", False, "Cyber Pro:"),
    ("advisory", "cyber_pro", False, "Cyber Pro:"),
])
def test_one_rule_sentence_per_effective_authority(monkeypatch, enforcement, mode, blocks, opening):
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    monkeypatch.setattr("ouroboros.config.get_runtime_mode", lambda: mode)
    block, _ = _block()

    assert (block["enforcement"], block["mode"], block["enforcement_blocks"]) == (enforcement, mode, blocks)
    assert block["rule"].startswith(opening) and block["rule"].count(".") == 1
    if opening == "Blocking:":  # NOT_PERFORMED stops the commit too (an all-Packet pool, a failed coupling part)
        assert "a review that was not performed" in block["rule"]
    if opening == "Advisory:":  # the handback before Git effects, never an automatic commit (DEVELOPMENT 05)
        assert "before any Git effect" in block["rule"] and "continue explicitly" in block["rule"]
        assert "commit proceeds" not in block["rule"]


def test_a_fourteen_seat_pool_shrinks_its_rows_and_stays_within_four_kilobytes(tmp_path, monkeypatch):
    keys = [f"{'k' * 60}{index:04d}" for index in range(14)]
    monkeypatch.setenv(SUBAGENTS_SETTING, _roster(*[
        _row(key, f"openai/{'m' * 53}{index:04d}") for index, key in enumerate(keys)]))
    _ledger(monkeypatch, lambda drive_root, task_id="", limit=20, hot_only=False: [
        {"record_id": f"rv-{'r' * 40}-{index:02d}", "surface": "commit_gate",
         "ts": f"2026-10-07T12:{index:02d}:00.000000+00:00", "verdict": {"aggregate": "QUORUM_FAILED"}}
        for index in range(7)][:int(limit)])
    block, text = _block()
    rows = block["pool"]

    assert len(text.encode("utf-8")) <= 4096
    assert block["omitted"] == {"rows": 14} and "recent_records" not in block
    records = _records(tmp_path)
    assert records["omitted"]["records"] == "1+" and len(records["recent_records"]) == 5
    assert len(rows) == 14 and all(set(row) == {"seat_id", "model"} and len(row["seat_id"]) == 64 for row in rows)
    assert block["full_source"]["pool"] == "GET /api/review-pool"


_LANES_KEY = "OUROBOROS_REVIEWER_SLOTS"
_HELPERS_ONLY = _roster(_row("helper-key", "openai/gpt-5.6-luna", marked=False))  # not a pool yet: a migration subject
_SNAPSHOT = "state/review_migrations/20261008T101010Z-slots-to-pool.json"


def _migration_outcomes():
    """Real ``MigrationOutcome`` records, computed by the migration itself — the shape
    ``config.review_pool_migrations_seen()`` returns (a tuple, oldest first)."""
    from ouroboros import review_pool_migration as m

    finished = m.migrate_review_lanes({_LANES_KEY: json.dumps({
        "triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "x/one"}, "effort": "high"}],
        "scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "x/one"}, "effort": "high"}]}),
        SUBAGENTS_SETTING: _HELPERS_ONLY})
    refused = m.migrate_review_lanes({_LANES_KEY: '{"triad": [{"model": "x/y"}]}', SUBAGENTS_SETTING: _HELPERS_ONLY})
    noop = m.migrate_review_lanes({_LANES_KEY: "", SUBAGENTS_SETTING: _POOL})
    assert finished.error == "" and not finished.noop and finished.consumed_keys == (_LANES_KEY,)
    assert refused.error and refused.retained_keys == (_LANES_KEY,) and refused.catalog_after is None
    assert noop.noop and noop.error == ""
    return finished, refused, noop


_REFUSED_LANES = '{"triad": [{"model": "x/y"}]}'


def _task_snapshot(**document):
    """A task's settings snapshot over the live document with ``document`` keys as the task read them."""
    from ouroboros.config import load_settings

    settings = {**load_settings(), **document}
    for key, value in document.items():
        if value is None:
            settings.pop(key, None)
    return task_settings_snapshot(settings, {**os.environ, SUBAGENTS_SETTING: str(document.get(SUBAGENTS_SETTING) or "")})


def test_w4_a_refused_migration_is_the_error_of_the_task_whose_document_it_refused(monkeypatch):
    """The A↔C seam, in the shape package C really has: ``config.review_pool_migrations_seen()``
    returns the tuple of ``MigrationOutcome`` records this process computed, and the supervisor
    boot's record (``server_maintenance.review_pool_migration_records``) holds the snapshot path.
    The block binds the outcome to the DOCUMENT whose pool it shows (FIX5 W4): a task whose
    settings snapshot still carries the refused lanes key and the unmigrated catalog sees its
    refusal and its snapshot path; the live settings, a pool catalog, carry neither — however
    recent the refusal is in this process — and the process registry keeps its history."""
    from ouroboros import config as cfg
    from ouroboros import server_maintenance

    finished, refused, _noop = _migration_outcomes()
    monkeypatch.setenv(SUBAGENTS_SETTING, _POOL)
    monkeypatch.setattr(cfg, "review_pool_migrations_seen", lambda: (finished, refused))
    monkeypatch.setattr(server_maintenance, "review_pool_migration_records", lambda state=None: {
        refused.input_sha256: {"ts": "20261008T101010Z", "snapshot": _SNAPSHOT, "error": refused.error, "reported": None}})
    refused_task = _task_snapshot(**{_LANES_KEY: _REFUSED_LANES, SUBAGENTS_SETTING: _HELPERS_ONLY})
    block, _ = _block(refused_task)
    assert block["source"] == "error" and block["error"] == refused.error
    assert "triad[0] has unknown keys" in block["error"], "the refusal's own sentence, not a paraphrase"
    assert block["migration_snapshot"] == _SNAPSHOT and block["pool"] == []

    live, _ = _block()
    assert (live["source"], live["error"]) == ("structured", "") and "migration_snapshot" not in live
    assert [row["seat_id"] for row in live["pool"]] == ["critic-key", "packet-key", "session-key"]
    assert cfg.review_pool_migrations_seen() == (finished, refused), "history is bound, never erased"


def test_w4_two_task_snapshots_each_see_their_own_migration_fact(monkeypatch):
    """Two tasks interleaved on one process: one started on the refused document, the other on
    the repaired one whose migration finished. Each block carries the fact of ITS document —
    the error and snapshot of the refusal, the snapshot (no error) of the finished migration —
    whichever outcome the process computed last, and in either order of reading."""
    from ouroboros import config as cfg
    from ouroboros import server_maintenance

    finished, refused, _noop = _migration_outcomes()
    finished_path = "state/review_migrations/20261008T090909Z-slots-to-pool.json"
    monkeypatch.setattr(cfg, "review_pool_migrations_seen", lambda: (finished, refused))
    monkeypatch.setattr(server_maintenance, "review_pool_migration_records", lambda state=None: {
        finished.input_sha256: {"ts": "20261008T090909Z", "snapshot": finished_path, "error": "", "reported": None},
        refused.input_sha256: {"ts": "20261008T101010Z", "snapshot": _SNAPSHOT, "error": refused.error, "reported": None}})
    refused_task = _task_snapshot(**{_LANES_KEY: _REFUSED_LANES, SUBAGENTS_SETTING: _HELPERS_ONLY})
    repaired_task = _task_snapshot(**{_LANES_KEY: None, SUBAGENTS_SETTING: finished.catalog_after})

    for _ in range(2):
        refused_block, _ = _block(refused_task)
        repaired_block, _ = _block(repaired_task)
        assert (refused_block["source"], refused_block["error"]) == ("error", refused.error)
        assert refused_block["migration_snapshot"] == _SNAPSHOT
        assert (repaired_block["source"], repaired_block["error"]) == ("structured", "")
        assert repaired_block["migration_snapshot"] == finished_path
        assert {row["model"] for row in repaired_block["pool"]} == {"x/one"}


def test_w4_a_finished_migration_is_no_error_and_a_noop_leaves_no_fact(monkeypatch):
    """The other direction: a migration that finished is not an error (the block reads the
    migrated pool as ``structured``) and still points at its snapshot; a no-op outcome — the
    catalog was already a pool — is skipped, so a process that saw only no-ops shows no
    migration fact at all."""
    from ouroboros import config as cfg
    from ouroboros import server_maintenance

    finished, _refused, noop = _migration_outcomes()
    monkeypatch.setenv(SUBAGENTS_SETTING, finished.catalog_after)
    monkeypatch.setattr(server_maintenance, "review_pool_migration_records", lambda state=None: {
        finished.input_sha256: {"ts": "20261008T101010Z", "snapshot": _SNAPSHOT, "error": "", "reported": None}})

    monkeypatch.setattr(cfg, "review_pool_migrations_seen", lambda: (finished, noop))
    block, _ = _block()
    assert (block["source"], block["error"]) == ("structured", "")
    assert block["migration_snapshot"] == _SNAPSHOT
    assert block["pool"] and {row["model"] for row in block["pool"]} == {"x/one"}, "the migrated pool is what runs"

    monkeypatch.setattr(cfg, "review_pool_migrations_seen", lambda: (noop,))
    block, _ = _block()
    assert (block["source"], block["error"]) == ("structured", "") and "migration_snapshot" not in block

    monkeypatch.setattr(cfg, "review_pool_migrations_seen", lambda: ())
    block, _ = _block()
    assert "migration_snapshot" not in block and block["source"] == "structured"


def test_w4_the_block_and_the_owners_receipt_share_one_deciding_predicate(monkeypatch):
    """One owner of «this migration decided this document»: the block asks
    ``review_pool_receipts.outcome_decides_document`` — the predicate the owner's receipt
    applies — about every outcome the process saw, and ``subagent_runtime`` keeps no copy
    of it, so the two surfaces cannot drift apart on which document an outcome decided."""
    from ouroboros import config as cfg
    from ouroboros import review_pool_receipts, subagent_runtime

    assert not hasattr(subagent_runtime, "_migration_decided_this_document")
    finished, refused, noop = _migration_outcomes()
    monkeypatch.setattr(cfg, "review_pool_migrations_seen", lambda: (finished, refused, noop))
    monkeypatch.setenv(SUBAGENTS_SETTING, finished.catalog_after)
    asked = []
    decides = review_pool_receipts.outcome_decides_document

    def spy(outcome, settings):
        asked.append(outcome)
        return decides(outcome, settings)

    monkeypatch.setattr(review_pool_receipts, "outcome_decides_document", spy)
    block, _ = _block()
    assert asked == [finished, refused, noop], "the block asks the receipt's predicate, outcome by outcome"
    assert (block["source"], block["error"]) == ("structured", "")
    assert [decides(outcome, {SUBAGENTS_SETTING: finished.catalog_after}) for outcome in asked] == [True, False, False]


def test_w4_the_read_seam_refusal_reaches_the_model_and_the_owners_repair_clears_it(tmp_path, monkeypatch):
    """End to end, nothing patched between the two packages: the settings document on disk
    carries lanes the migration refuses; the read seam (``review_pool_migration.apply_at_read_seam``)
    records the outcome and keeps the lanes key, and the block the model reads carries that
    refusal (no snapshot path: the boot has not written one in this process). The owner then
    saves a catalog — the save drops the lanes key — and the repaired document's block carries
    no past migration error: the refusal stays in the process registry (history), bound to the
    document it refused, not to whatever document is read next."""
    from ouroboros import config as cfg
    from ouroboros import review_pool_migration as m

    monkeypatch.setattr(m, "_MIGRATIONS_SEEN", {})
    monkeypatch.delenv(SUBAGENTS_SETTING, raising=False)
    path = tmp_path / "settings.json"
    monkeypatch.setattr(cfg, "SETTINGS_PATH", path)
    path.write_text(json.dumps({_LANES_KEY: _REFUSED_LANES, SUBAGENTS_SETTING: _HELPERS_ONLY}), encoding="utf-8")
    block, _ = _block()
    (refusal,) = m.migrations_seen()
    assert refusal.error and block["source"] == "error" and block["error"] == refusal.error
    assert "triad[0] has unknown keys" in block["error"] and "migration_snapshot" not in block
    assert block["pool"] == []

    path.write_text(json.dumps({SUBAGENTS_SETTING: _POOL}), encoding="utf-8")  # the owner's repairing save
    repaired, _ = _block()
    assert (repaired["source"], repaired["error"]) == ("structured", "") and "migration_snapshot" not in repaired
    assert [row["seat_id"] for row in repaired["pool"]] == ["critic-key", "packet-key", "session-key"]
    assert m.migrations_seen() == (refusal,), "the registry keeps the refusal; the block no longer wears it"


def test_recent_records_are_the_readers_newest_five_of_this_task_from_a_bounded_hot_read(tmp_path, monkeypatch):
    calls = []
    newest_first = [{"record_id": f"r{index}", "surface": "commit_gate", "ts": f"2026-10-07T00:00:0{index}+00:00",
                     "verdict": {"aggregate": "FAIL" if index % 2 else "PASS"}} for index in reversed(range(7))]

    # review_ledger.recent_records' signature: the limit is an int, newest first; the context
    # capture reads the hot index only, with a six-row cap (never the whole history).
    def recent_records(drive_root, task_id="", limit=20, hot_only=False):
        calls.append((drive_root, task_id, int(limit), hot_only))
        return newest_first[:max(1, int(limit))]

    _ledger(monkeypatch, recent_records)
    block = _records(tmp_path, task_id="task-9")

    assert calls == [(tmp_path, "task-9", 6, True)]
    assert [record["record_id"] for record in block["recent_records"]] == ["r6", "r5", "r4", "r3", "r2"]
    assert block["recent_records"][:2] == [
        {"record_id": "r6", "surface": "commit_gate", "aggregate": "PASS", "ts": "2026-10-07T00:00:06+00:00"},
        {"record_id": "r5", "surface": "commit_gate", "aggregate": "FAIL", "ts": "2026-10-07T00:00:05+00:00"},
    ]
    assert block["omitted"]["records"] == "1+", "more rows than shown in the hot index: a bounded fact, not a count"
    calls.clear()
    empty = _records(tmp_path, task_id="")
    assert (empty["recent_records"], empty["omitted"]["records"], calls) == ([], 0, []), "empty selects every task"
    few = lambda drive_root, task_id="", limit=20, hot_only=False: newest_first[:3]  # noqa: E731
    _ledger(monkeypatch, few)
    assert _records(tmp_path, task_id="task-9")["omitted"]["records"] == 0, "nothing else in the hot index, no archive"
    _ledger(monkeypatch, few, archived_segments_exist=lambda drive_root: True)
    assert _records(tmp_path, task_id="task-9")["omitted"]["records"] == "unknown", "an archive may hold older records"


def test_an_unreadable_ledger_is_unknown_never_a_silent_zero(tmp_path, monkeypatch):
    def recent_records(drive_root, task_id="", limit=20, hot_only=False):
        raise OSError("index unreadable")

    _ledger(monkeypatch, recent_records)
    block = _records(tmp_path)
    assert block["recent_records"] == [] and block["omitted"]["records"] == "unknown"
