"""A skill's Google sign-in offered in a conversation (owner's vision of 2026-11-27).

The mind gives the owner a button to the skill's card (a ``send_links`` place ``skill-sign-in/<skill>/<service>``,
``ouroboros/link_actions.py``); the owner signs in on that card; the conversation that offered it then hears once that
the skill is signed in (``ouroboros/sign_in_offers.py``). Telegram, which cannot finish a sign-in that returns to the
computer, says where to finish it instead of drawing a dead button. Offline: a fake provider, the real sign-in route.
"""
from __future__ import annotations

import pytest

from ouroboros import link_actions, sign_in_offers
from ouroboros.link_actions import LinkActionsValidationError, validate_link_actions
from tests.test_service_sign_in_loopback import CAL, _save_client, _shown, _sign_in, _token, world  # noqa: F401 -- the fixture

PLACE = "skill-sign-in/calendar/google"


def test_a_place_is_a_link_action_with_a_label_and_no_address():
    assert validate_link_actions([{"label": "Sign in to Google", "place": PLACE},
                                  {"label": "Docs", "url": "https://example.com/docs"}]) == [
        {"label": "Sign in to Google", "place": PLACE}, {"label": "Docs", "url": "https://example.com/docs"}]
    for bad in ({"label": "x", "place": "skill-sign-in/calendar"}, {"label": "x", "place": "https://example.com"},
                {"label": "x", "place": PLACE, "url": "https://example.com"}, {"label": " ", "place": PLACE},
                {"label": "x", "place": "skill-sign-in/../google"}):
        with pytest.raises(LinkActionsValidationError):  # the whole batch is refused, the good link too
            validate_link_actions([{"label": "Docs", "url": "https://example.com/docs"}, bad])
    assert link_actions.sign_in_place(PLACE) == ("calendar", "google")
    # The canonical names the host's sign-in answers give: a dot in a skill name, a service id that starts with "_".
    assert link_actions.sign_in_place("skill-sign-in/calendar.team/_work") == ("calendar.team", "_work")
    assert link_actions.sign_in_place("skill-sign-in/calendar/Google") is None
    # A model that fills both optional fields sends an empty place beside its address: an ordinary link.
    assert validate_link_actions([{"label": "Docs", "url": "https://example.com/docs", "place": ""}]) == [
        {"label": "Docs", "url": "https://example.com/docs"}]


def test_a_place_names_a_skill_that_declares_that_sign_in(world):  # noqa: F811
    from ouroboros.tools import core_artifacts

    core_artifacts._known_sign_in_places([{"label": "Sign in", "place": PLACE}])
    for typo in ("skill-sign-in/calendar/gogle", "skill-sign-in/calendr/google"):
        with pytest.raises(LinkActionsValidationError, match="declares no sign-in service"):
            core_artifacts._known_sign_in_places([{"label": "Sign in", "place": typo}])


def test_a_phone_only_mirror_still_hears_connected():
    from skills.telegram.plugin import _target_chat

    owner = {"TELEGRAM_CHAT_ID": "4242", "TELEGRAM_MIRROR_MODE": "telegram_only"}
    row = {"role": "system", "system_type": "sign_in_connected", "transport": {}}
    assert _target_chat(owner, row) == 4242, "an owner-act receipt, like a skill notice, reaches the phone"
    assert _target_chat(owner, {**row, "system_type": "storage_free_notice"}) == 0


def test_a_links_row_with_a_place_is_delivered_and_its_conversation_noted(tmp_path, monkeypatch):
    from supervisor import message_bus

    monkeypatch.setattr(message_bus, "DATA_DIR", tmp_path)
    bridge = message_bus.LocalChatBridge({})
    sent = []
    bridge._broadcast_fn = sent.append
    ok, why = bridge.send_links(4242, [{"label": "Sign in to Google", "place": PLACE}], title="Calendar needs Google")
    assert (ok, why) == (True, "ok") and sent[-1]["actions"] == [{"label": "Sign in to Google", "place": PLACE}]
    assert (tmp_path / "state" / "sign_in" / "offers.json").exists()
    said = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda *a, **k: said.append((a, k)))
    assert sign_in_offers.announce(tmp_path, "calendar", "google", "Google")
    (chat_id, text), kwargs = said[0]
    assert chat_id == 4242 and text == "calendar is signed in to Google now."
    assert (kwargs["role"], kwargs["system_type"]) == ("system", "sign_in_connected")
    assert not sign_in_offers.announce(tmp_path, "calendar", "google", "Google")
    assert len(said) == 1, "said once: the offer leaves when it is announced"


def test_a_completed_sign_in_says_connected_in_the_conversation_that_offered_it_and_nowhere_else(world, monkeypatch):  # noqa: F811
    provider, owner, host = world
    from ouroboros import config

    said = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: said.append((a, k)))
    _save_client(owner)
    _sign_in(provider, owner, CAL)  # nobody offered it: the card's own sign-in says nothing in a chat
    assert said == []
    sign_in_offers.note(config.DATA_DIR, 4242, [{"label": "Sign in", "place": PLACE}])
    _sign_in(provider, owner, CAL)
    assert [(args[0], kwargs["system_type"]) for args, kwargs in said] == [(4242, "sign_in_connected")]
    assert "@" not in said[0][0][1], "the line names no account"


def test_the_mind_is_told_it_can_offer_the_cards_sign_in_as_a_button(world):  # noqa: F811
    _provider, _owner, host = world
    answer = _token(host, CAL)
    assert answer.status_code == 404 and "place skill-sign-in/calendar/google" in answer.json()["hint"]


def test_telegram_says_where_to_finish_a_sign_in_instead_of_a_dead_button():
    from skills.telegram.lib import telegram_places
    from skills.telegram.lib.telegram_api import _LOCALIZED_TEXTS

    buttons, places = telegram_places.link_rows([{"label": "Sign in to Google", "place": PLACE},
                                                 {"label": "Docs", "url": "https://example.com/docs"}, "junk"])
    assert buttons == [("Docs", "https://example.com/docs")] and places == [("Sign in to Google", "calendar")]
    line = telegram_places.finish_lines(places, _LOCALIZED_TEXTS["en"])
    assert line.startswith("Sign in to Google: finish this on the computer running Ouroboros")
    assert "Extensions › Skills › calendar" in line


def test_a_replayed_return_never_answers_a_newer_offer(world, monkeypatch):  # noqa: F811
    """A stale tab's second Finish replays the filed receipt of an earlier sign-in: no new sign-in happened, so the
    newer offer waits for the real one."""
    provider, owner, _host = world
    from ouroboros import config

    said = []
    monkeypatch.setattr("supervisor.message_bus.send_with_budget", lambda *a, **k: said.append((a, k)))
    _save_client(owner)
    started = owner.post("/api/sign-in", json={"action": "begin", "skill": CAL, "service": "google",
                                               "scope_words": ["openid", "email", "calendar.events"],
                                               **_shown(owner, CAL, "")}).json()
    returned = provider.approve(started["url"])
    assert owner.post("/api/sign-in", json={"action": "complete", "returned_url": returned}).json()["state"] == "signed_in"
    sign_in_offers.note(config.DATA_DIR, 4242, [{"label": "Sign in", "place": PLACE}])
    owner.post("/api/sign-in", json={"action": "complete", "returned_url": returned})  # the stale tab's replay
    assert said == [], "a replayed receipt is not a new sign-in"
    _sign_in(provider, owner, CAL)
    assert [(args[0], kwargs["system_type"]) for args, kwargs in said] == [(4242, "sign_in_connected")]
