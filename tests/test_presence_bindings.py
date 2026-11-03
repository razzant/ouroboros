import json

import pytest

from ouroboros.presence_bindings import PRESENCE_BINDINGS_FILENAME, PresenceBinding, PresenceBindingError, PresenceEndpoint, list_presence_bindings
from tests.frozen_presence_bindings import load_presence_binding, new_presence_binding_id, save_presence_binding


def _binding(transport_skill="telegram-bot"):
    endpoint = PresenceEndpoint("telegram", "public-bot", "room-42")
    return PresenceBinding(
        new_presence_binding_id(),
        transport_skill,
        "community-moderator",
        endpoint,
        endpoint,
    )


def test_owner_binding_roundtrip_and_transport_ownership(tmp_path):
    binding = save_presence_binding(tmp_path, _binding())
    assert load_presence_binding(tmp_path, "telegram-bot", binding.binding_id) == binding
    with pytest.raises(PresenceBindingError) as caught:
        load_presence_binding(tmp_path, "slack-bridge", binding.binding_id)
    assert caught.value.code == "presence_binding_wrong_transport"


def test_disabled_and_corrupt_bindings_fail_closed(tmp_path):
    binding = _binding()
    save_presence_binding(tmp_path, PresenceBinding(**{**binding.__dict__, "enabled": False}))
    with pytest.raises(PresenceBindingError) as caught:
        load_presence_binding(tmp_path, "telegram-bot", binding.binding_id)
    assert caught.value.code == "presence_binding_disabled"

    path = tmp_path / "state" / PRESENCE_BINDINGS_FILENAME
    path.write_text(json.dumps({"schema_version": 1, "bindings": []}), encoding="utf-8")
    with pytest.raises(PresenceBindingError) as caught:
        load_presence_binding(tmp_path, "telegram-bot", binding.binding_id)
    assert caught.value.code == "presence_bindings_invalid_state"


def test_binding_list_preserves_owner_created_endpoints(tmp_path):
    binding = save_presence_binding(tmp_path, _binding())
    assert list_presence_bindings(tmp_path) == (binding,)


def test_account_wide_endpoint_is_explicit_and_cannot_pin_a_thread():
    assert PresenceEndpoint("slack", "workspace-1", "*") == PresenceEndpoint(
        "slack", "workspace-1", "*", ""
    )
    with pytest.raises(PresenceBindingError) as caught:
        PresenceEndpoint("slack", "workspace-1", "*", "thread-1")
    assert caught.value.code == "presence_binding_wildcard_thread_invalid"

    wildcard = PresenceEndpoint("slack", "workspace-1", "*")
    exact = PresenceEndpoint("slack", "workspace-1", "channel-1")
    PresenceBinding(new_presence_binding_id(), "slack-bridge", "helper", wildcard, exact)
    with pytest.raises(PresenceBindingError) as caught:
        PresenceBinding(
            new_presence_binding_id(),
            "slack-bridge",
            "helper",
            wildcard,
            wildcard,
        )
    assert caught.value.code == "presence_binding_destination_must_be_exact"
