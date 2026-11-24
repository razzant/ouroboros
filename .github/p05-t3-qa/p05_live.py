"""Loopback-only product servers. Source and interpreter come from the runner's sealed staging receipt."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
from ouroboros.process_containment import ProcessContainer
from ouroboros.server_process import read_service_bindings
from ouroboros.test_environment import isolated_environment, owned_process_temp
from tests.fixtures_mock_llm import MockLLMServer
from tests.test_ui_smoke_playwright import _free_port

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = json.loads((ROOT / 'subject.json').read_bytes())
ROLE = Path(sys.prefix).name
assert ROLE in {'base', 'candidate'} and Path(sys.prefix) == ROOT / 'venvs' / ROLE
ASSETS = ('modules/place_sheet.js', 'modules/settings_accounts.js', 'modules/settings_extension_forms.js')


def exact_tree(name):
    checkout = ROOT / name
    git = ['/usr/bin/git', '--no-optional-locks', '-C', str(checkout)]
    pins = SUBJECT['subjects'][name]
    assert (checkout / '.git').is_dir() and not (checkout / '.git').is_symlink()
    for ref, key in [('HEAD', 'commit'), ('HEAD^{tree}', 'tree')]:
        assert subprocess.check_output([*git, 'rev-parse', ref], text=True).strip() == pins[key]
    subprocess.run([*git, 'diff', '--quiet', 'HEAD', '--'], check=True)
    assert subprocess.check_output([*git, 'ls-files', '--others', '--exclude-standard', '-z']) == b''


def seed_mock_settings(data, llm_url, *, inline=False):
    mock = 'openai-compatible::mock-model'
    settings = dict.fromkeys(('OUROBOROS_MODEL', 'OUROBOROS_MODEL_LIGHT', 'OUROBOROS_MODEL_FALLBACKS'), mock)
    settings.update(OPENAI_COMPATIBLE_API_KEY='p05-synthetic', OPENAI_COMPATIBLE_BASE_URL=llm_url,
                    OUROBOROS_MAX_WORKERS=1, OUROBOROS_RUNTIME_MODE='light', OUROBOROS_SAFETY_MODE='off',
                    OUROBOROS_TASK_REVIEW_MODE='off', OUROBOROS_CONTEXT_MODE='low', TOTAL_BUDGET=10.0,
                    OUROBOROS_POST_TASK_EVOLUTION='false', OUROBOROS_UI_LANGUAGE='en',
                    OUROBOROS_STARTER_SKILLS='["builtin/telegram"]', OUROBOROS_STARTER_SKILLS_TURN_ON=False)
    if inline:
        settings.update(OUROBOROS_MODEL_CONSCIOUSNESS=mock, OUROBOROS_CONSCIOUSNESS_AUTONOMY='act',
                        OUROBOROS_CONSCIOUSNESS_DAILY_USD=10.0)
    (data / 'settings.json').write_text(json.dumps(settings), encoding='utf-8')


class Live:
    def __init__(self, root, llm, name, managed=False):
        self.name, self.repo = name, ROOT / name
        self.env = isolated_environment(root, self.repo, process_temp=owned_process_temp())
        self.data = root / 'data'
        self.log = root / 'server-stdio.log'
        self.port, host_port = _free_port(), _free_port()
        assert self.port != host_port
        self.url, self.host = f'http://127.0.0.1:{self.port}', f'http://127.0.0.1:{host_port}'
        self.env.update(OUROBOROS_SERVER_HOST='127.0.0.1', OUROBOROS_SERVER_PORT=str(self.port),
                        OUROBOROS_HOST_SERVICE_PORT=str(host_port), OUROBOROS_NETWORK_PASSWORD='p05-ui-only',
                        PYTHONDONTWRITEBYTECODE='1', GIT_TERMINAL_PROMPT='0')
        assert Path(self.env['OUROBOROS_DATA_DIR']) == self.data
        self.policy = None
        if managed:
            # Supported managed-deployment mode: the launch names a strict policy file outside the checkout and
            # this fixture's data root; the app pins it on boot and legitimately keeps source and deps read-only.
            from ouroboros import deployment_policy
            self.policy = root / 'admin' / 'deployment-policy.json'
            self.policy.parent.mkdir(exist_ok=True)
            self.policy_bytes = json.dumps({'version': 1}).encode('utf-8')
            self.policy.write_bytes(self.policy_bytes)
            assert self.repo not in self.policy.parents and self.data not in self.policy.parents
            self.env.pop(deployment_policy.GENERATION_ENV, None)
            self.env[deployment_policy.POLICY_ENV] = str(self.policy)
        seed_mock_settings(self.data, llm.base_url)
        self.proc = self.container = None

    def state(self):
        response = httpx.get(self.url + '/api/state', timeout=3, trust_env=False)
        response.raise_for_status()
        return response.json()

    def ready(self, prior_since=None):
        until = time.monotonic() + 85
        while time.monotonic() < until:
            exit_code = self.proc.poll()
            assert exit_code is None, f'owned server exited before the required boot was ready (status {exit_code})'
            try:
                row = self.state()
                since = (row.get('boot') or {}).get('since')
                if row.get('supervisor_ready') is True and since and since != prior_since:
                    binding = read_service_bindings(self.data)['main']
                    assert (binding['pid'], binding['port']) == (self.proc.pid, self.port)
                    assert Path(os.readlink(f'/proc/{self.proc.pid}/cwd')) == self.repo
                    exact_tree(self.name)
                    for rel in ASSETS:
                        raw = httpx.get(f'{self.url}/static/{rel}', timeout=6, trust_env=False)
                        raw.raise_for_status()
                        assert raw.content == (self.repo / 'web' / rel).read_bytes(), rel
                    return since
            except httpx.HTTPError:
                pass
            time.sleep(.2)
        raise AssertionError('owned loopback server did not become ready')

    def start(self):
        assert self.proc is self.container is None
        exact_tree(self.name)
        interpreter = ROOT / 'venvs' / self.name / 'bin' / 'python'
        if self.name != 'old':
            assert Path(sys.prefix) == interpreter.parent.parent
        self.container = ProcessContainer()
        with self.log.open('ab') as log:
            self.proc = self.container.spawn([str(interpreter), '-B', 'server.py'], cwd=self.repo,
                                             env=self.env, stdout=log, stderr=subprocess.STDOUT)
        return self.ready()

    def stop(self):
        proc, container = self.proc, self.container
        self.proc = self.container = None
        if container is None:
            return
        try:
            if proc is not None and proc.poll() is None:
                proc.terminate()  # exact test-owned handle within the disposable systemd unit
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            try:
                error = container.reap()
            finally:
                container.close()
            assert not error, 'loopback process ownership unsettled'
            if proc is not None:
                proc.wait(timeout=5)
        exact_tree(self.name)

    def restart(self, since):
        # The owned Linux server can re-exec before its accepted HTTP response drains. Never resend.
        try:
            with httpx.stream('POST', self.url + '/api/command', json={'cmd': '/restart', 'if_boot': since},
                              timeout=15, trust_env=False) as response:
                assert response.status_code == 200, 'restart was explicitly rejected'
                response.read()
                body = response.json()
                assert body['status'] == 'ok' and body.get('accepted') is not False
        except (httpx.ReadError, httpx.ReadTimeout, httpx.RemoteProtocolError):
            return self.ready(since)  # only a different ready boot on the same owned process qualifies
        return self.ready(since)

    def policy_governs(self):
        """Closed boolean: GET /api/settings reports the source, version and digest prefix of the fixture's exact policy bytes."""
        meta = httpx.get(self.url + '/api/settings', timeout=8, trust_env=False).json().get('_meta')
        state = meta.get('policy_state') if isinstance(meta, dict) else None
        seen = state.get('deployment') if isinstance(state, dict) else None
        return (self.policy is not None and isinstance(seen, dict)
                and (seen.get('source'), seen.get('version'), seen.get('digest')) == (
                    'deployment_policy', 1, hashlib.sha256(self.policy_bytes).hexdigest()[:16]))


@contextmanager
def live(root, name=ROLE, managed=False):
    with MockLLMServer() as llm:
        server = Live(root, llm, name, managed)
        try:
            yield server
        finally:
            server.stop()


def publish_fact(name, facts):
    path = ROOT / 'work' / 'proof' / f'{name}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(facts, stream, sort_keys=True)
        stream.write('\n')


def publish_step(name, step=None, **notes):
    """Where a live case stopped, so a missing witness names its step instead of vanishing.

    Only a closed step name and closed integer measurements leave the runner: no product
    text, no path, no payload. A call without a step keeps the step already recorded and
    only adds measurements, so a measurement can be taken before an assertion fails.
    """
    from p05_proof import NOTE_KEYS, STEP_VALUES

    assert step is None or step in STEP_VALUES, 'unknown step name'
    assert set(notes) <= NOTE_KEYS, 'unknown measurement'
    assert all(type(value) is int for value in notes.values()), 'measurement is not an integer'
    path = ROOT / 'work' / 'proof' / 'steps' / f'{name}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    current = {'step': None, 'notes': {}}
    if path.exists():
        try:
            stored = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            stored = {}
        if isinstance(stored, dict) and isinstance(stored.get('notes'), dict):
            current = {'step': stored.get('step') if stored.get('step') in STEP_VALUES else None,
                       'notes': dict(stored['notes'])}
    if step is not None:
        current['step'] = step
    current['notes'].update(notes)
    path.write_text(json.dumps(current, sort_keys=True) + '\n', encoding='utf-8')
