"""Locally imported fixture: no engine lifecycle or provider can escape a test.

Import this fixture into each suite that needs it; never register it globally
(other suites own their fake transports). Tests may supply fake transports; an unpatched attach/ensure records an unavailable
observation, and provisioning/model generation always fails the test.
"""
import pytest


@pytest.fixture(autouse=True)
def stub_delegate_engines(monkeypatch):
    from ouroboros import claudexor_daemon as daemon
    from ouroboros import claudexor_harness_install as installer
    from ouroboros.gateways import claudexor
    from ouroboros.llm import LLMClient

    calls = []

    def unavailable(name, code):
        def call(*_args, **_kwargs):
            calls.append(name)
            raise claudexor.ClaudexorUnavailable(code, "controlled transport absence")
        return call

    monkeypatch.setattr(daemon, "ensure_owned_gateway", unavailable("ensure", "daemon_recovery_only"))
    monkeypatch.setattr(daemon, "read_owned_gateway", unavailable("attach", "daemon_unreachable"))
    monkeypatch.setattr(claudexor, "ClaudexorGateway", unavailable("gateway", "daemon_unreachable"))
    monkeypatch.setattr(daemon, "get_owned_daemon", lambda **_kw: pytest.fail("real engine lifecycle forbidden"))
    monkeypatch.setattr(daemon, "resolve_claudexord", lambda: pytest.fail("real engine provisioning forbidden"))
    monkeypatch.setattr(daemon.OwnedClaudexorDaemon, "_spawn", lambda *_a, **_k: pytest.fail("real engine spawn forbidden"))
    monkeypatch.setattr(installer, "install_missing_harness_cli", lambda *_a: pytest.fail("real install forbidden"))
    monkeypatch.setattr(LLMClient, "chat_async", lambda *_a, **_k: pytest.fail("real provider forbidden"))
    monkeypatch.setattr(LLMClient, "chat", lambda *_a, **_k: pytest.fail("real provider forbidden"))
    return calls
