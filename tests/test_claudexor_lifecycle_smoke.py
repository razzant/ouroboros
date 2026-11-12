"""The fake-only capability bridge cannot discover or admit a vendor route."""
from pathlib import Path
from types import SimpleNamespace
import os
import shutil

import pytest
import yaml

from scripts import claudexor_lifecycle_smoke as smoke
from ouroboros.gateways.claudexor import ClaudexorGateway, ClaudexorUnavailable


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


def test_lifecycle_job_runs_the_smoke_with_the_environment_interpreter():
    """A bare inner `python` resolved outside the virtualenv on Windows (#1641)."""
    path = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "claudexor-platform-gate.yml"
    job = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["lifecycle"]
    assert "windows-latest" in job["strategy"]["matrix"]["os"]
    commands = "\n".join(str(step.get("run", "")) for step in job["steps"])
    assert '"$py" -I -S scripts/safe_test.py' in commands
    assert '-- "$py" scripts/claudexor_lifecycle_smoke.py' in commands
    assert "-- python scripts/claudexor_lifecycle_smoke.py" not in commands


def test_legacy_seed_copies_old_writer_bytes_and_keeps_native_identity(tmp_path):
    daemon = tmp_path / 'daemon'
    fixture = smoke.seed_legacy_fixture(daemon)
    source = smoke.LEGACY_FIXTURE / fixture['journal']['path']
    copied = daemon / fixture['journal']['path']
    assert copied.read_bytes() == source.read_bytes()
    assert fixture['baseline'] == '820e849cd15cca47758b700603e156ef29129fbf'
    assert fixture['expected']['runId'] != fixture['expected']['jobId']
    assert fixture['run_artifacts'] == []
    if os.name != 'nt':
        assert copied.stat().st_mode & 0o777 == 0o600
        assert copied.parent.stat().st_mode & 0o777 == 0o700
    with pytest.raises(smoke.SmokeFailure) as error:
        smoke.seed_legacy_fixture(daemon)
    assert error.value.code == 'legacy_fixture_requires_fresh_store'
    assert copied.read_bytes() == source.read_bytes()


def test_legacy_seed_rejects_corrupt_fixture_before_writing(tmp_path, monkeypatch):
    import json

    fixture_root = tmp_path / 'fixture'
    shutil.copytree(smoke.LEGACY_FIXTURE, fixture_root)
    manifest = json.loads((fixture_root / 'fixture.json').read_text(encoding='utf-8'))
    journal = fixture_root / manifest['journal']['path']
    data = bytearray(journal.read_bytes())
    data[-1] ^= 1
    journal.write_bytes(data)
    monkeypatch.setattr(smoke, 'LEGACY_FIXTURE', fixture_root)
    daemon = tmp_path / 'daemon'
    with pytest.raises(smoke.SmokeFailure) as error:
        smoke.seed_legacy_fixture(daemon)
    assert error.value.code == 'legacy_fixture_bytes_changed'
    assert not daemon.exists()


@pytest.mark.parametrize('response', ['retained', 'missing_native', 'missing_returns_record'])
def test_legacy_lookup_uses_public_native_id_without_job_fallback(tmp_path, monkeypatch, response):
    daemon = tmp_path / 'daemon'
    fixture = smoke.seed_legacy_fixture(daemon)
    # Unit filesystem preconditions only; this test does not qualify SQL startup.
    (daemon / 'journal').rename(daemon / 'journal-legacy')
    (daemon / 'engine.sqlite').touch()
    expected, requests = fixture['expected'], []
    gateway = object.__new__(ClaudexorGateway)

    def request(method, path, **kwargs):
        requests.append((method, path))
        if path.endswith('run-o6-not-present') and response != 'missing_returns_record':
            raise ClaudexorUnavailable('run_missing', 'fixture 404', status_code=404)
        if response == 'missing_native':
            raise ClaudexorUnavailable('run_missing', 'fixture 404', status_code=404)
        summary = {**expected, 'runId': expected['jobId']} if response == 'job_alias' else expected
        return {'summary': summary}

    monkeypatch.setattr(gateway, '_request', request)
    if response == 'retained':
        assert smoke.retained_legacy_run(gateway, daemon, fixture) == expected
    elif response == 'missing_native':
        with pytest.raises(ClaudexorUnavailable):
            smoke.retained_legacy_run(gateway, daemon, fixture)
    else:
        with pytest.raises(smoke.SmokeFailure) as error:
            smoke.retained_legacy_run(gateway, daemon, fixture)
        assert error.value.code == ('legacy_native_run_changed' if response == 'job_alias' else 'unknown_native_run_found')
    assert requests[0] == ('GET', '/v2/runs/' + expected['runId'])
    assert ('GET', '/v2/runs/' + expected['jobId']) not in requests


def test_legacy_lookup_refuses_missing_archive_even_if_the_record_answers(tmp_path):
    daemon = tmp_path / 'daemon'
    fixture = smoke.seed_legacy_fixture(daemon)
    (daemon / 'engine.sqlite').touch()

    def get_run(run_id):
        if run_id == fixture['expected']['runId']:
            return {'summary': fixture['expected']}
        raise ClaudexorUnavailable('run_missing', 'fixture 404', status_code=404)

    with pytest.raises(smoke.SmokeFailure) as error:
        smoke.retained_legacy_run(SimpleNamespace(get_run=get_run), daemon, fixture)
    assert error.value.code == 'legacy_source_not_preserved'


def test_every_runtime_name_the_witness_imports_inside_a_step_exists():
    """The witness imports runtime internals inside its steps, which only the platform gate runs; a removed or
    renamed name fails here instead (#3167: ``settle`` still imported the deleted ``_delegate_wait``)."""
    import ast
    import importlib
    import importlib.util

    tree = ast.parse(Path(smoke.__file__).read_text(encoding='utf-8'))
    steps = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    late = [(node.module, alias.name) for step in steps
            for node in ast.walk(step) if isinstance(node, ast.ImportFrom) and node.module
            and node.module.split('.')[0] in {'ouroboros', 'scripts', 'supervisor'} for alias in node.names]
    def found(module, name):  # an attribute, or a submodule of a package (a plain module has no __path__)
        parent = importlib.import_module(module)
        return hasattr(parent, name) or (hasattr(parent, '__path__')
                                         and importlib.util.find_spec(f'{module}.{name}') is not None)

    missing = [f'{module}.{name}' for module, name in late if not found(module, name)]
    assert late and missing == []
