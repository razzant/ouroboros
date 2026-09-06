"""The owner-facing chat notice about retired settings keys (D-07).

``config.normalize_settings_raw`` drops the keys a release retired and says so on the
module logger — a line an owner who never opens the Logs panel does not see. The
supervisor boot now tells the OWNER once, in their chat, from the sets that read seam
recorded, with the same sentence, deduplicated durably per retired-key set in
``state.json``. These tests pin: emitted once for a document carrying a retired key,
not emitted without one, not repeated on a second boot, not sent (and not marked)
before an owner chat is bound, the active panel source named truthfully in each of its three
states (absent / authored / invalid), and the boot wiring itself.
"""

from __future__ import annotations

import pathlib

import pytest

from ouroboros import config as cfg
from ouroboros import server_maintenance
from supervisor import message_bus, state


@pytest.fixture
def boot_state(tmp_path, monkeypatch):
    """A supervisor state root at ``tmp_path`` with a fresh in-process retirement seam."""
    state.init(tmp_path)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)
    (tmp_path / "locks").mkdir(parents=True, exist_ok=True)
    cfg._RETIREMENT_NOTICE_SEEN.clear()
    yield tmp_path
    cfg._RETIREMENT_NOTICE_SEEN.clear()


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
    assert "OUROBOROS_REVIEWER_SLOTS" in text, "the successor setting is named"
    assert "shipped" in text.lower(), "the ACTIVE panel source is named"
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


def test_a_malformed_reviewer_slots_setting_names_no_panel_and_the_parse_error(boot_state, sent, caplog):
    """Non-empty is not authored, and malformed is not absent either: the loader
    (``load_reviewer_slot_config``) RAISES on text the strict parser rejects, so no panel
    serves — commit review blocks, plan and skill review refuse — until the owner repairs
    the setting. The notice must say exactly that, with the row-precise parse error, in the
    chat and in the read-seam log line; it must claim neither the authored panel nor the
    shipped default (astra M4 finding 8: the earlier pin asserted "SHIPPED default" here)."""
    import logging

    from ouroboros.reviewer_slot_config import parse_reviewer_slots

    with pytest.raises(ValueError) as err:
        parse_reviewer_slots(MALFORMED_SLOTS)
    parse_error = str(err.value)

    _bind_owner(1)
    doc = dict(RETIRED_DOC, OUROBOROS_REVIEWER_SLOTS=MALFORMED_SLOTS)
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        loaded = cfg.normalize_settings_raw(doc)
    server_maintenance._startup_retired_settings_notice(loaded)

    assert len(sent) == 1
    text = sent[0][1]
    assert "OUROBOROS_REVIEWER_SLOTS" in text and "NO reviewer panel" in text
    assert "refused" in text and "repaired" in text
    assert parse_error in text, "the row-precise parse error is what the owner has to fix"
    assert "shipped" not in text.lower() and "authored in that setting" not in text
    log_lines = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(log_lines) == 1 and "NO reviewer panel" in log_lines[0] and parse_error in log_lines[0]


@pytest.mark.parametrize("slots_state,expected,forbidden", [
    (("absent", ""), "SHIPPED default", ("authored in that setting", "NO reviewer panel")),
    (("authored", ""), "authored in that setting", ("SHIPPED", "NO reviewer panel")),
    (("invalid", "OUROBOROS_REVIEWER_SLOTS: triad[0] is not an object"),
     "NO reviewer panel", ("SHIPPED", "authored in that setting")),
], ids=["absent", "authored", "invalid"])
def test_the_notice_sentence_has_three_honest_panel_states(slots_state, expected, forbidden):
    """The sentence itself, per state the reviewer-slot seam derives: absent -> the shipped
    default runs; authored -> that panel runs; invalid -> no panel, the parse error named."""
    from ouroboros.settings_defaults import retired_setting_keys_notice

    text = retired_setting_keys_notice(("OUROBOROS_REVIEW_MODELS",), reviewer_slots=slots_state)
    assert expected in text, text
    for phrase in forbidden:
        assert phrase not in text, (phrase, text)
    if slots_state[0] == "invalid":
        assert slots_state[1] in text

def test_an_authored_reviewer_slots_setting_is_named_as_the_active_panel(boot_state, sent, caplog):
    """The notice names what runs NOW: an owner who already authored the structured
    setting is not told they are on the shipped default panel — in the chat, and in the
    log line the read seam emits from the same sentence."""
    import logging

    _bind_owner(1)
    doc = dict(RETIRED_DOC, OUROBOROS_REVIEWER_SLOTS=AUTHORED_SLOTS)
    with caplog.at_level(logging.WARNING, logger="ouroboros.config"):
        loaded = cfg.normalize_settings_raw(doc)
    server_maintenance._startup_retired_settings_notice(loaded)

    assert len(sent) == 1
    text = sent[0][1]
    assert "authored" in text and "shipped" not in text.lower()
    log_lines = [r.getMessage() for r in caplog.records if "retired" in r.getMessage()]
    assert len(log_lines) == 1 and "shipped" not in log_lines[0].lower()


def test_the_supervisor_boot_calls_the_notice_after_the_queue_restore():
    """The wiring pin: the notice runs in ``server._run_supervisor`` once the message bus
    and the state file are initialised, next to the other boot-time owner notices."""
    source = (pathlib.Path(__file__).resolve().parents[1] / "server.py").read_text(encoding="utf-8")
    body = source.split("def _run_supervisor(settings: dict) -> None:", 1)[1]
    assert "_startup_retired_settings_notice(settings)" in body.split("\ndef ", 1)[0]
    assert body.index("restore_pending_from_snapshot()") < body.index("_startup_retired_settings_notice(settings)")
