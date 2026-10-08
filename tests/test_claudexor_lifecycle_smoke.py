"""The fake-only capability bridge cannot discover or admit a vendor route."""
from types import SimpleNamespace

import pytest

from scripts import claudexor_lifecycle_smoke as smoke
from ouroboros.gateways.claudexor import ClaudexorGateway


@pytest.mark.parametrize('harness,kind,expected', [
    ('fake-hang', 'fake', ''),
    ('vendor', 'cli', 'unexpected_catalog_route'),
    ('fake-hang', 'cli', 'non_fake_manifest'),
])
def test_fixture_catalog_uses_only_addressed_fake_facts(monkeypatch, harness, kind, expected):
    def forbidden(*args, **kwargs):
        pytest.fail('The ordinary catalog probes vendor routes and must not run here.')

    monkeypatch.setattr(ClaudexorGateway, 'agent_capabilities', forbidden)
    calls = []

    def request(method, path, **kwargs):
        calls.append((method, path))
        return {'harnesses': [{'id': harness, 'status': 'ok', 'manifest': {
            'kind': kind, 'access_profiles_supported': ['readonly']}}]}

    client = SimpleNamespace(_request=request)
    try:
        with smoke.fixture_catalog():
            if expected:
                with pytest.raises(smoke.SmokeFailure) as error:
                    ClaudexorGateway.agent_capabilities(client)
                assert error.value.code == expected
            else:
                catalog = ClaudexorGateway.agent_capabilities(client)
                assert catalog['harnesses'][0]['accessProfilesSupported'] == ['readonly']
                assert catalog['harnesses'][0]['id'] == 'fake-hang'
        assert calls == [('GET', '/v2/harnesses?all=true&harness=fake-hang&harness=fake-implement')]
    finally:
        assert ClaudexorGateway.agent_capabilities is forbidden
