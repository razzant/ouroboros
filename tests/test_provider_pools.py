"""Pools: a named connection that rotates over its accounts; no providers, no network."""
from __future__ import annotations

import base64

import pytest

from ouroboros.llm_attempt import ConnectionUnavailable
from ouroboros.llm_routing import resolve_target
from ouroboros.provider_connections import (
    connection_configured,
    connection_display,
    connection_values,
    connections_meta,
    normalize_connections,
    pool_members,
    settings_view,
)
from ouroboros.provider_models import CONNECTIONS_KEY, parse_model_route


def _key(client_id: str) -> str:
    """A GigaChat authorization key: base64("<client_id>:<secret>")."""
    return base64.b64encode(f"{client_id}:secret".encode()).decode()


SETTINGS = {
    CONNECTIONS_KEY: [
        {"kind": "gigachat", "name": "a", "label": "Keys", "credentials": _key("client-a")},
        {"kind": "gigachat", "name": "b", "credentials": _key("client-b")},
        {"kind": "gigachat", "name": "pool", "label": "Keys", "members": ["a", "b"]},
    ],
}
POOL = parse_model_route("gigachat@pool::GigaChat-3-Pro")


@pytest.fixture(autouse=True)
def fresh_rotation(monkeypatch):
    from ouroboros import provider_connections as pc

    monkeypatch.setattr(pc, "_ROTATION", {})


def test_a_pool_serves_each_call_through_the_next_member():
    read, rows = settings_view(SETTINGS)
    served = [connection_values(POOL, read, rows)("credentials") for _ in range(4)]
    assert served == [_key("client-a"), _key("client-b"), _key("client-a"), _key("client-b")]


def test_a_read_never_spends_a_rotation():
    read, rows = settings_view(SETTINGS)
    assert connection_values(POOL, read, rows, choose=False)("credentials") == _key("client-a")
    assert connection_values(POOL, read, rows, choose=False)("credentials") == _key("client-a")
    # The dispatch path still starts from the pool's first member.
    assert connection_values(POOL, read, rows)("credentials") == _key("client-a")


def test_a_disabled_member_is_skipped_not_fatal():
    settings = {CONNECTIONS_KEY: [
        {"kind": "gigachat", "name": "a", "credentials": _key("client-a")},
        {"kind": "gigachat", "name": "b", "credentials": _key("client-b"), "enabled": False},
        {"kind": "gigachat", "name": "c", "credentials": ""},
        {"kind": "gigachat", "name": "pool", "members": ["a", "b", "c"]},
    ]}
    read, rows = settings_view(settings)
    assert [connection_values(POOL, read, rows)("credentials") for _ in range(3)] == [_key("client-a")] * 3
    assert connection_configured(POOL, read, rows) is True


def test_a_pool_without_a_serveable_member_is_refused():
    settings = {CONNECTIONS_KEY: [
        {"kind": "gigachat", "name": "a", "credentials": ""},
        {"kind": "gigachat", "name": "pool", "members": ["a"]},
    ]}
    read, rows = settings_view(settings)
    with pytest.raises(ConnectionUnavailable):
        connection_values(POOL, read, rows)
    assert connection_configured(POOL, read, rows) is False


def test_the_route_grammar_serves_a_pool_with_a_real_target():
    targets = [resolve_target("gigachat@pool::GigaChat-3-Pro", settings=SETTINGS) for _ in range(2)]
    assert [target["api_key"] for target in targets] == [_key("client-a"), _key("client-b")]
    assert [target["provider"] for target in targets] == ["gigachat", "gigachat"]
    assert targets[0]["usage_model"] == "gigachat@pool::GigaChat-3-Pro"


def test_a_pool_says_how_many_accounts_it_holds():
    read, rows = settings_view(SETTINGS)
    assert connection_display("gigachat", "pool", rows).endswith("2 accounts")
    assert connection_display("gigachat", "a", rows).endswith("Keys")
    meta = {entry["name"]: entry for entry in connections_meta(SETTINGS, SETTINGS)}
    assert meta["pool"]["members"] == ["a", "b"]
    assert meta["pool"]["configured"] is True
    assert "members" not in meta["a"]


def test_a_pool_row_is_saved_as_written():
    rows, error = normalize_connections(SETTINGS[CONNECTIONS_KEY])
    assert error == ""
    assert pool_members(rows[2]) == ("a", "b")


@pytest.mark.parametrize("rows, message", [
    ([{"kind": "gigachat", "name": "pool", "members": []}], "non-empty list"),
    ([{"kind": "gigachat", "name": "a", "credentials": _key("a")},
      {"kind": "gigachat", "name": "pool", "members": ["a", "ghost"]}], "not a saved connection"),
    ([{"kind": "gigachat", "name": "a", "credentials": _key("a")},
      {"kind": "gigachat", "name": "pool", "members": ["a", "a"]}], "twice"),
    ([{"kind": "gigachat", "name": "a", "credentials": _key("a")},
      {"kind": "gigachat", "name": "pool", "members": ["a"], "credentials": _key("x")}],
     "carries only kind, name, label, enabled and members"),
    ([{"kind": "gigachat", "name": "pool", "members": ["default"]}], "must be a named connection"),
    ([{"kind": "gigachat", "name": "a", "credentials": _key("a")},
      {"kind": "gigachat", "name": "b", "credentials": _key("b")},
      {"kind": "gigachat", "name": "inner", "members": ["a", "b"]},
      {"kind": "gigachat", "name": "outer", "members": ["inner"]}], "is itself a pool"),
    ([{"kind": "openai", "name": "a", "api_key": "k"},
      {"kind": "gigachat", "name": "pool", "members": ["a"]}], "connection of another kind"),
])
def test_a_pool_row_that_cannot_serve_is_refused(rows, message):
    saved, error = normalize_connections(rows)
    assert saved == []
    assert message in error


def test_configured_checks_never_move_the_next_call():
    read, rows = settings_view(SETTINGS)
    for _ in range(3):
        assert connection_configured(POOL, read, rows) is True
        assert connection_display("gigachat", "pool", rows).endswith("2 accounts")
    assert connection_values(POOL, read, rows)("credentials") == _key("client-a")


def test_a_member_of_another_kind_refuses_the_pool_rather_than_being_skipped():
    settings = {CONNECTIONS_KEY: [
        {"kind": "openai", "name": "other", "api_key": "k"},
        {"kind": "gigachat", "name": "a", "credentials": _key("client-a")},
        {"kind": "gigachat", "name": "pool", "members": ["a", "other"]},
    ]}
    read, rows = settings_view(settings)
    with pytest.raises(ConnectionUnavailable):
        connection_values(POOL, read, rows)
    assert connection_configured(POOL, read, rows) is False


def test_a_pool_label_must_be_text():
    rows, error = normalize_connections([
        {"kind": "gigachat", "name": "a", "credentials": _key("client-a")},
        {"kind": "gigachat", "name": "pool", "members": ["a"], "label": 7},
    ])
    assert rows == [] and "label must be text" in error
