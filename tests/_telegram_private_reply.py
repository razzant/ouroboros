"""Fictional Places inputs for private-reply boundary tests."""
from types import SimpleNamespace

import pytest

from ouroboros import place_guard, places


_PRIVATE_FIXTURE_DIGEST = "sha256:" + "f" * 64


def unreadable_private_bridge(_root, *, strict=False):
    if strict:
        raise OSError("fixture role unavailable")
    return ""


@pytest.fixture
def confirmed_private_place(tmp_path, monkeypatch):
    """One fictional Place with an isolated bridge-role reader."""
    from ouroboros import telegram_owner_bot

    target = dict(provider="telegram", account_id="4503599627370501",
                  conversation_id="4503599627370502", thread_id="")
    connector = telegram_owner_bot.FEED_CONNECTOR
    selected = places.connect_place(tmp_path, connector=connector, provider=target["provider"],
                                    account_id=target["account_id"], conversation_id=target["conversation_id"], scope="conversation")
    selected = places.edit_place(tmp_path, selected.place_id, "confirm_owner_account", author="owner",
                                 platform_actor_id=target["conversation_id"], account_id=target["account_id"])
    monkeypatch.setattr(telegram_owner_bot, "owner_bridge_bot_id", lambda *_a, **_k: "")
    return SimpleNamespace(root=tmp_path, connector=connector, target=target, place=selected,
                           digest=_PRIVATE_FIXTURE_DIGEST)


@pytest.fixture
def confirmed_private_condition(confirmed_private_place):
    fixture = confirmed_private_place
    records = places.place_records(fixture.root, persist_import=False)
    condition = place_guard.capture_owner_dm(fixture.root, records, fixture.connector, fixture.target,
                                             place_id=fixture.place.place_id, admitted_revision=fixture.place.revision)
    assert condition is not None
    return condition
