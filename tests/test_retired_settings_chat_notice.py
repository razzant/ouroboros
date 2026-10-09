"""The owner-facing chat notice about retired settings keys (D-07).

``config.normalize_settings_raw`` drops the keys a release retired and says so on the
module logger — a line an owner who never opens the Logs panel does not see. The
supervisor boot tells the OWNER once, in their chat, from the sets that read seam
recorded, with the same sentence, deduplicated durably per retired-key set in
``state.json``. These tests pin: emitted once for a document carrying a retired key,
not emitted without one, not repeated on a second boot, not sent (and not marked)
before an owner chat is bound, the successor named truthfully (the reviewer comma-lists
are replaced by the review pool), the review-lane keys consumed by the pool migration
never reported as a loss, and the boot wiring itself.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from ouroboros import config as cfg
from ouroboros import review_pool_migration as rpm
from ouroboros import server_maintenance
from supervisor import message_bus, state


@pytest.fixture
def boot_state(tmp_path, monkeypatch):
    """A supervisor state root at ``tmp_path`` with a fresh in-process retirement seam."""
    state.init(tmp_path)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "locks").mkdir(parents=True, exist_ok=True)
    state.save_state({})  # an initialized install: only explicit init creates state (#1307)
    cfg._RETIREMENT_NOTICE_SEEN.clear()
    rpm._MIGRATIONS_SEEN.clear()
    yield tmp_path
    cfg._RETIREMENT_NOTICE_SEEN.clear()
    rpm._MIGRATIONS_SEEN.clear()


@pytest.fixture
def sent(monkeypatch):
    rows: list = []
    monkeypatch.setattr(
        message_bus, "send_with_budget",
        lambda chat_id, text, *args, **kwargs: rows.append((chat_id, text, kwargs)),
    )
    return rows


def _bind_owner(chat_id: int = 1) -> None:
    state.update_state(lambda st: st.__setitem__("owner_chat_id", chat_id))


RETIRED_DOC = {
    "OUROBOROS_REVIEW_MODELS": "a/one,b/two",
    "OUROBOROS_SCOPE_REVIEW_MODEL": "c/three",
    "TOTAL_BUDGET": 10.0,
}


def test_notice_reaches_the_owner_chat_once_per_retired_key_set(boot_state, sent):
    _bind_owner(1)
    loaded = cfg.normalize_settings_raw(dict(RETIRED_DOC))

    server_maintenance._startup_retired_settings_notice(loaded)

    assert len(sent) == 1, sent
    chat_id, text, kwargs = sent[0]
    assert chat_id == 1
    assert kwargs == {"role": "system", "system_type": "retired_settings_notice"}
    for key in ("OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL"):
        assert key in text, key
    assert "NOT honored" in text
    assert "OUROBOROS_SUBAGENTS" in text, "the successor surface (the review pool) is named"
    assert "review pool" in text
    assert "TOTAL_BUDGET" not in text

    # The durable marker is keyed by the exact retired-key set.
    marker = state.load_state().get("retired_settings_notified")
    assert isinstance(marker, dict)
    assert list(marker) == ["OUROBOROS_REVIEW_MODELS,OUROBOROS_SCOPE_REVIEW_MODEL"]

    # A second apply (supervisor revival, next boot) does not repeat it.
    server_maintenance._startup_retired_settings_notice(loaded)
    assert len(sent) == 1


def test_no_notice_without_a_retired_key(boot_state, sent):
    _bind_owner(1)
    loaded = cfg.normalize_settings_raw({"TOTAL_BUDGET": 10.0})

    server_maintenance._startup_retired_settings_notice(loaded)

    assert sent == []
    assert "retired_settings_notified" not in state.load_state()


def test_the_durable_marker_survives_a_fresh_process(boot_state, sent):
    """The dedupe is the state file, not the in-process seam: a new process that reads
    the same document again (the seam's own set is empty there) still stays quiet."""
    _bind_owner(1)
    loaded = cfg.normalize_settings_raw(dict(RETIRED_DOC))
    server_maintenance._startup_retired_settings_notice(loaded)
    assert len(sent) == 1

    cfg._RETIREMENT_NOTICE_SEEN.clear()  # "fresh process"
    loaded = cfg.normalize_settings_raw(dict(RETIRED_DOC))
    server_maintenance._startup_retired_settings_notice(loaded)
    assert len(sent) == 1

    # A DIFFERENT retired-key set is its own loss and gets its own line. The key is taken
    # from the successor table, never spelled here: the grep-class retirement gate
    # (tests/test_legacy_timeout_retirement.py) keeps retired names out of live surfaces.
    from ouroboros.settings_defaults import RETIRED_SETTING_SUCCESSORS

    retired_key, successors = next(iter(RETIRED_SETTING_SUCCESSORS.items()))
    cfg.normalize_settings_raw({retired_key: "5"})
    server_maintenance._startup_retired_settings_notice(loaded)
    assert len(sent) == 2
    assert retired_key in sent[1][1]
    assert successors[0] in sent[1][1], "the successor table is read"


def test_nothing_is_sent_or_marked_before_an_owner_chat_is_bound(boot_state, sent):
    loaded = cfg.normalize_settings_raw(dict(RETIRED_DOC))

    server_maintenance._startup_retired_settings_notice(loaded)
    assert sent == []
    assert "retired_settings_notified" not in state.load_state()

    # The first boot that HAS an owner chat delivers it.
    _bind_owner(7)
    server_maintenance._startup_retired_settings_notice(loaded)
    assert [row[0] for row in sent] == [7]


# A structured panel the strict parser ACCEPTS (slot ids, typed routes, both groups).
AUTHORED_SLOTS = (
    '{"triad": [{"slot_id": "t1", "route": {"kind": "api_chat", "target_id": "x/y"}}], '
    '"scope": [{"slot_id": "s1", "route": {"kind": "api_chat", "target_id": "x/y"}}]}'
)


MALFORMED_SLOTS = '{"triad": [{"model": "x/y"}]}'  # a row without slot_id/route: rejected


def test_the_comma_list_clause_names_the_review_pool():
    """The sentence itself: the reviewer comma-lists are replaced by the rows of the
    subagent catalog marked Reviewer — one static fact, because which rows run is the
    review-pool migration's own report, not this notice's."""
    from ouroboros.settings_defaults import retired_setting_keys_notice

    text = retired_setting_keys_notice(("OUROBOROS_REVIEW_MODELS",))
    assert "OUROBOROS_REVIEW_MODELS" in text and "review pool" in text
    assert "OUROBOROS_SUBAGENTS" in text and "Settings → Agents" in text
    for stale in ("SHIPPED", "authored in that setting", "NO reviewer panel", "OUROBOROS_REVIEWER_SLOTS"):
        assert stale not in text, (stale, text)


def test_migrated_review_lane_keys_are_consumed_not_reported_as_a_loss(boot_state, sent, caplog):
    """An authored panel is migrated into the subagent catalog BEFORE the purge: the lane
    key and the surface effort keys leave the document as consumed, so neither the chat
    notice nor the read-seam log line lists them among the dropped keys — only the
    comma-lists, which the migration never read (ABI-10), are a loss to report."""
    import logging

    _bind_owner(1)
    doc = dict(RETIRED_DOC, OUROBOROS_REVIEWER_SLOTS=AUTHORED_SLOTS, OUROBOROS_EFFORT_REVIEW="medium")
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        loaded = cfg.normalize_settings_raw(doc)
    server_maintenance._startup_retired_settings_notice(loaded)

    assert "OUROBOROS_REVIEWER_SLOTS" not in loaded and "OUROBOROS_EFFORT_REVIEW" not in loaded
    catalog = json.loads(loaded["OUROBOROS_SUBAGENTS"])
    # t1 (packet, medium from the surface key) and s1 (reads, high) are two engines: two rows.
    assert [row["subagent_id"] for row in catalog["items"] if row.get("review_eligible")] == ["review-1", "review-2"]
    assert len(sent) == 1
    text = sent[0][1]
    assert "OUROBOROS_REVIEW_MODELS" in text
    assert "OUROBOROS_REVIEWER_SLOTS" not in text and "OUROBOROS_EFFORT_REVIEW" not in text
    log_lines = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(log_lines) == 1 and "OUROBOROS_REVIEWER_SLOTS" not in log_lines[0]
    assert list(state.load_state()["retired_settings_notified"]) == [
        "OUROBOROS_REVIEW_MODELS,OUROBOROS_SCOPE_REVIEW_MODEL"]


def test_a_malformed_reviewer_slots_setting_is_kept_for_the_owner_not_dropped(boot_state, sent, caplog):
    """A lane value the strict parser rejects cannot be migrated, and a key the owner must
    still repair is not a loss to announce: the migration keeps it in the document (its own
    report names the error), the purge leaves it alone, and the retired-keys notice lists
    only the comma-lists."""
    import logging

    from ouroboros.review_pool_migration import parse_reviewer_slots

    with pytest.raises(ValueError):
        parse_reviewer_slots({}, MALFORMED_SLOTS)

    _bind_owner(1)
    doc = dict(RETIRED_DOC, OUROBOROS_REVIEWER_SLOTS=MALFORMED_SLOTS)
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        loaded = cfg.normalize_settings_raw(doc)
    server_maintenance._startup_retired_settings_notice(loaded)

    assert loaded["OUROBOROS_REVIEWER_SLOTS"] == MALFORMED_SLOTS, "kept until the owner's catalog save"
    assert "OUROBOROS_SUBAGENTS" not in loaded, "no partial migration"
    assert len(sent) == 1 and "OUROBOROS_REVIEWER_SLOTS" not in sent[0][1]
    outcomes = cfg.review_pool_migrations_seen()
    assert len(outcomes) == 1 and outcomes[0].error and outcomes[0].retained_keys == ("OUROBOROS_REVIEWER_SLOTS",)
    not_migrated = [r.getMessage() for r in caplog.records if "not migrated" in r.getMessage()]
    assert len(not_migrated) == 1


def test_the_supervisor_boot_calls_the_notices_after_the_queue_restore():
    """The wiring pin: the notices run in ``server._run_supervisor`` once the message bus
    and the state file are initialised, next to the other boot-time owner notices; the
    environment fact and then the review-pool report follow the retired-keys notice."""
    source = (pathlib.Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    body = source.split("def _run_supervisor(settings: dict) -> None:", 1)[1].split("\ndef ", 1)[0]
    assert "_startup_retired_settings_notice(settings)" in body
    assert "_startup_review_pool_notice(settings)" in body
    assert body.index("restore_pending_from_snapshot(") < body.index("_startup_retired_settings_notice(settings)")
    assert body.index("_startup_retired_settings_notice(settings)") < body.index("_startup_environment_review_notice()")
    assert body.index("_startup_environment_review_notice()") < body.index("_startup_review_pool_notice(settings)")


# --- review keys set in the PROCESS ENVIRONMENT (D1-V03 / VD3-03) ---------------------------


ENV_REVIEW_KEYS = ("OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_EFFORT_REVIEW", "OUROBOROS_EFFORT_SCOPE_REVIEW",
                   "OUROBOROS_EFFORT_DEEP_SELF_REVIEW", "OUROBOROS_MODEL_DEEP_SELF_REVIEW",
                   "OUROBOROS_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODELS", "OUROBOROS_SCOPE_REVIEW_MODEL")


@pytest.fixture
def no_document(boot_state, monkeypatch):
    """No settings document: the never-configured install whose process environment is the
    only place an operator could have put review keys (a Docker unit, a Colab cell)."""
    for key in ENV_REVIEW_KEYS + ("OUROBOROS_SUBAGENTS", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", boot_state / "absent" / "settings.json")
    monkeypatch.setattr(server_maintenance, "DATA_DIR", boot_state)
    monkeypatch.setenv("OPENROUTER_API_KEY", "present")
    return boot_state


def test_review_keys_in_the_environment_are_one_loud_fact_not_a_configuration(no_document, monkeypatch, sent, caplog):
    """The lanes exported in the environment are NOT read (no env lanes reader comes back):
    the install runs the factory rows — and the boot says so, as a WARNING on the server log
    at every boot and ONCE in the owner chat (durable marker), naming the keys and the
    successor (``OUROBOROS_SUBAGENTS``)."""
    import logging

    monkeypatch.setenv("OUROBOROS_REVIEWER_SLOTS", AUTHORED_SLOTS)
    monkeypatch.setenv("OUROBOROS_EFFORT_REVIEW", "low")
    _bind_owner(1)
    loaded = cfg.load_settings_lock_held(_settings_lock_held=False)
    items = json.loads(loaded["OUROBOROS_SUBAGENTS"])["items"]
    assert "x/y" not in [row["route"]["target_id"] for row in items], "env lanes are not read"
    assert "OUROBOROS_REVIEWER_SLOTS" not in loaded and all(row["minted_from"] == "factory_default" for row in items)

    assert server_maintenance.environment_retired_review_keys() == ("OUROBOROS_REVIEWER_SLOTS", "OUROBOROS_EFFORT_REVIEW")
    with caplog.at_level(logging.WARNING, logger="server"):
        server_maintenance._startup_environment_review_notice()
        server_maintenance._startup_environment_review_notice()  # a supervisor revival
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "no longer read" in r.getMessage()]
    assert len(warnings) == 2, "the log line is loud at every boot"
    assert len(sent) == 1, sent
    chat_id, text, kwargs = sent[0]
    assert chat_id == 1 and kwargs == {"role": "system", "system_type": "retired_settings_notice"}
    assert text == warnings[0]
    assert "OUROBOROS_REVIEWER_SLOTS, OUROBOROS_EFFORT_REVIEW" in text and "are no longer read" in text
    assert "OUROBOROS_SUBAGENTS" in text and "Settings → Agents" in text and "not applied" in text
    assert list(state.load_state()["retired_settings_notified"]) == [
        "environment:OUROBOROS_REVIEWER_SLOTS,OUROBOROS_EFFORT_REVIEW"]

    # The migration's own report does not call this install settings-less: it names the keys.
    sent.clear()
    server_maintenance._startup_review_pool_notice(loaded)
    (pool_notice,) = [row[1] for row in sent if row[2].get("system_type") == "review_pool_migration_notice"]
    assert pool_notice.startswith("⚙️ Review pool initialized.") and "3 reviewer rows" in pool_notice
    assert "This install had no review settings" not in pool_notice
    assert ("settings document had no review settings" in pool_notice
            and "OUROBOROS_REVIEWER_SLOTS, OUROBOROS_EFFORT_REVIEW set in the process environment are no longer read"
            in pool_notice)


def test_no_environment_notice_and_the_plain_never_configured_report_without_env_review_keys(no_document, sent, caplog):
    """The other direction: an environment without review keys yields no log line, no chat
    message, no marker — and the never-configured report keeps its plain head."""
    import logging

    _bind_owner(1)
    loaded = cfg.load_settings_lock_held(_settings_lock_held=False)
    assert server_maintenance.environment_retired_review_keys() == ()
    with caplog.at_level(logging.WARNING, logger="server"):
        server_maintenance._startup_environment_review_notice()
    assert not [r for r in caplog.records if "no longer read" in r.getMessage()]
    assert sent == [] and "retired_settings_notified" not in state.load_state()

    server_maintenance._startup_review_pool_notice(loaded)
    (pool_notice,) = [row[1] for row in sent if row[2].get("system_type") == "review_pool_migration_notice"]
    assert pool_notice.startswith("⚙️ Review pool initialized. This install had no review settings (no review lanes, "
                                  "no subagent catalog)")
    assert "process environment" not in pool_notice


def test_the_environment_notice_names_the_retired_comma_lists_too(no_document):
    """An N-2 unit still exporting the reviewer comma-lists hits the same fact, one sentence."""
    environ = {"OUROBOROS_REVIEW_MODELS": "a/one,b/two", "OUROBOROS_REVIEWER_SLOTS": "   "}
    keys = server_maintenance.environment_retired_review_keys(environ)
    assert keys == ("OUROBOROS_REVIEW_MODELS",), "a blank value configures nothing and is not announced"
    text = server_maintenance.environment_review_notice(keys)
    assert "OUROBOROS_REVIEW_MODELS, which is no longer read" in text and "That value was not applied" in text
