"""Shared emergency acceptance, with synchronous stand-ins and no native stop."""
import threading
from types import SimpleNamespace

import pytest

from ouroboros import server_control


@pytest.fixture
def emergency(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr("ouroboros.config.resolve_data_dir", lambda: tmp_path)
    monkeypatch.setattr("ouroboros.panic_source.record_source",
                        lambda root, **evidence: calls.append(("source", evidence)))
    monkeypatch.setattr(threading, "Thread",
                        lambda *, target, **kw: SimpleNamespace(start=target))
    panic = server_control.PanicIngress(lambda: calls.append("bridge stop"))
    panic._owner = [True, (17, 23)]  # a previously positively observed binding
    return SimpleNamespace(panic=panic), calls


def test_bridge_acceptance_keeps_transport_evidence_and_never_uses_startup(emergency):
    bridge, calls = emergency
    evidence = {"ingress": "telegram", "owner_surface": "unknown", "skill_name": "telegram"}
    assert server_control.request_panic(bridge, "/panic now", source="skill:telegram", user_id=17, chat_id=23,
                                       evidence=evidence, startup_owner_command=lambda *a, **kw: pytest.fail("fallback"))
    assert calls == [("source", evidence), "bridge stop"]


@pytest.mark.parametrize("pair", [(0, 0), (99, 23), (17, 99)])
def test_non_owner_never_acquires_startup_stop_authority(emergency, pair):
    bridge, calls = emergency
    assert not server_control.request_panic(bridge, "/panic", source="skill:telegram",
        user_id=pair[0], chat_id=pair[1], startup_owner_command=lambda *a, **kw: pytest.fail("non-owner fallback"))
    assert calls == [] and not server_control._restart_stop_requested


@pytest.mark.parametrize("text,accepted", [("/panic", True), ("/panic trailing", False)])
def test_web_startup_uses_existing_callback_compatibility(emergency, text, accepted):
    _, calls = emergency
    evidence = {"ingress": "web_ui_ws", "owner_surface": "unknown"}

    def startup(command, **kwargs):
        assert kwargs == {"send_kwargs": {"broadcast": False}}
        return (lambda: calls.append("startup stop")) if command == "/panic" else None

    assert server_control.request_panic(None, text, startup_owner_command=startup, evidence=evidence) is accepted
    assert calls == ([("source", evidence), "startup stop"] if accepted else [])
    assert server_control._restart_stop_requested is accepted
