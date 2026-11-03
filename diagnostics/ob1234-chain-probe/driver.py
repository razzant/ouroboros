#!/usr/bin/env python3
"""OB#1234 in-container driver for ONE arm (v3). Runs as the unprivileged user ``qa``
inside ``dbus-run-session``, with the container's own system python (stdlib only).

It starts the PUBLISHED launcher through its public entry with a clean caller
environment, completes onboarding through the real API, submits one ordinary task
(POST /api/tasks) and lets the product's own loop call browse_page and
browser_action against a loopback HTTP fixture. It records raw facts only; it judges
nothing.

Provider. The wire shape copies the repository's own loopback model
(tests/system_e2e/harness.py::LoopbackModelServer: /models with max_model_len, JSON or
one-frame SSE + usage + [DONE]; canned Safety JSON for response_format json_object).
It is copied, not imported: harness.py imports pytest and product modules at import
time (lines 34, 47, 53-68), which the container's python does not have, and the probe
must not run HEAD code against the published 7.4.4/7.9.0 servers. Its review-organ
markers are HEAD literals, so they are not used for gating here.

Gating (nanny finding a). The script advances ONLY for the intended agent stream: the
run nonce is present AND both browse_page and browser_action are offered in ``tools``
AND the request is not a json_object (Safety) call. A step is complete only when the
request carries a tool result for the call id the provider itself issued; the final
answer is given only after results for EVERY issued id arrived. Every other request
(title, Safety, review, anything) gets a neutral answer and is saved raw.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = 'ob1234-scripted'
SLUG = f'openai-compatible::{MODEL}'


def fixture_page(nonce: str, frame_origin: str) -> str:
    # The cross-site iframe proves a second-origin REQUEST only. The product launches
    # Chromium with --disable-features=site-per-process (browser.py:303 at v7.4.4,
    # :347 at v7.9.0), so no separate renderer is promised; roles are what is observed.
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>OB1234 fixture</title></head>
<body><h1>OB1234_FIXTURE_{nonce}</h1>
<canvas id="gl" width="128" height="128"></canvas>
<iframe src="{frame_origin}/frame.html" width="200" height="80"></iframe>
<div id="done" style="display:none">OB1234_DONE</div>
<script>
(async () => {{
  const r = {{nonce: '{nonce}'}};
  try {{
    const gl = document.getElementById('gl').getContext('webgl');
    r.webgl = !!gl;
    if (gl) {{ r.renderer = gl.getParameter(gl.RENDERER); gl.clearColor(0.2, 0.6, 0.9, 1); gl.clear(gl.COLOR_BUFFER_BIT); }}
  }} catch (e) {{ r.webgl_error = String(e); }}
  try {{ await fetch('/beacon', {{method: 'POST', body: JSON.stringify(r)}}); }} catch (e) {{}}
  document.getElementById('done').style.display = 'block';
}})();
</script></body></html>"""


FRAME = '<!doctype html><html><body><p>OB1234_FRAME</p></body></html>'


class ScriptedProvider:
    """One loopback server: fixture pages, beacon and the gated scripted model."""

    def __init__(self, out: Path, nonce: str, public_url: str, hold: float):
        self.out, self.nonce, self.hold = out, nonce, hold
        (out / 'provider').mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.seq = 0
        self.issued: list[tuple[str, str, dict]] = []   # (call id, tool, args)
        self.hits, self.beacons = [], []
        self.final_served = threading.Event()
        self.ledger = (out / 'provider.jsonl').open('a', buffering=1)
        self.steps = [('browse_page', {'url': 'FIXTURE', 'output': 'text', 'wait_for': '#done', 'timeout': 30000}),
                      ('browser_action', {'action': 'screenshot'})]
        if public_url:
            self.steps.append(('browse_page', {'url': public_url, 'output': 'text', 'timeout': 30000}))
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                return

            def send(self, status, data: bytes, ctype='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', ctype)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()

            def do_GET(self):
                with provider.lock:
                    provider.hits.append({'ts': time.time(), 'path': self.path, 'host': self.headers.get('Host'),
                                          'ua': self.headers.get('User-Agent')})
                if self.path.rstrip('/').endswith('/models'):
                    # Same capability answer as harness.LoopbackModelServer (window evidence).
                    return self.send(200, json.dumps({'data': [{'id': MODEL, 'max_model_len': 2_000_000}]}).encode())
                if self.path.startswith('/fixture.html'):
                    return self.send(200, fixture_page(provider.nonce, f'http://localhost:{self.server.server_port}')
                                     .encode(), 'text/html; charset=utf-8')
                if self.path.startswith('/frame.html'):
                    return self.send(200, FRAME.encode(), 'text/html; charset=utf-8')
                return self.send(404, b'{}')

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
                if self.path == '/beacon':
                    with provider.lock:
                        provider.beacons.append({'ts': time.time(), 'body': raw.decode('utf-8', 'replace')})
                    return self.send(204, b'')
                try:
                    body = json.loads(raw or b'{}')
                except ValueError:
                    body = {}
                message = provider.answer(body if isinstance(body, dict) else {}, raw)
                payload = {'id': f'ob1234-{provider.seq}', 'object': 'chat.completion', 'model': MODEL,
                           'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}],
                           'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}}
                if not body.get('stream'):
                    return self.send(200, json.dumps(payload).encode())
                delta = dict(message)
                if delta.get('tool_calls'):
                    delta['tool_calls'] = [dict(c, index=i) for i, c in enumerate(delta['tool_calls'])]
                common = {'id': payload['id'], 'model': MODEL, 'object': 'chat.completion.chunk'}
                frames = [{**common, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': 'stop'}]},
                          {**common, 'choices': [], 'usage': payload['usage']}]
                data = ''.join('data: ' + json.dumps(f) + '\n\n' for f in frames) + 'data: [DONE]\n\n'
                return self.send(200, data.encode(), 'text/event-stream')

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def answer(self, body: dict, raw: bytes) -> dict:
        with self.lock:
            self.seq += 1
            seq = self.seq
        (self.out / 'provider' / f'{seq:04d}.json').write_bytes(raw)   # raw, every request
        tools = [((t or {}).get('function') or {}).get('name') for t in body.get('tools') or []]
        fmt = body.get('response_format')
        safety = isinstance(fmt, dict) and fmt.get('type') == 'json_object'
        has_nonce = self.nonce in raw.decode('utf-8', 'replace')
        intended = has_nonce and not safety and 'browse_page' in tools and 'browser_action' in tools
        results = {m.get('tool_call_id') for m in body.get('messages') or []
                   if isinstance(m, dict) and m.get('role') == 'tool'}
        row = {'seq': seq, 'ts': time.time(), 'model': body.get('model'), 'stream': bool(body.get('stream')),
               'tools_offered': len(tools), 'has_nonce': has_nonce, 'safety_json': safety, 'intended': intended}
        if safety:
            message = {'role': 'assistant', 'content': json.dumps({'status': 'SAFE', 'reason': 'ob1234 stub'})}
            row['answer'] = 'safety_SAFE'
        elif not intended:
            message = {'role': 'assistant', 'content': 'ok'}
            row['answer'] = 'neutral'
        else:
            with self.lock:
                answered = [cid for cid, _, _ in self.issued if cid in results]
                row['issued'] = [cid for cid, _, _ in self.issued]
                row['results_for_issued'] = answered
                pending = [item for item in self.issued if item[0] not in results]
                if pending:  # a resend without our result: repeat the SAME call, never advance
                    cid, tool, args = pending[0]
                    row['answer'] = f'repeat:{cid}'
                elif len(self.issued) < len(self.steps):
                    tool, args = self.steps[len(self.issued)]
                    args = {k: (f'http://127.0.0.1:{self.port}/fixture.html' if v == 'FIXTURE' else v)
                            for k, v in args.items()}
                    cid = f'ob1234_{self.nonce[:8]}_{len(self.issued)}'
                    self.issued.append((cid, tool, args))
                    row['answer'] = f'call:{cid}:{tool}'
                else:
                    cid = None
                    row['answer'] = 'final_after_all_results'
            if cid is None:
                time.sleep(self.hold)  # browser still open: the sampler re-reads Chromium maps
                message = {'role': 'assistant', 'content': f'OB1234 probe complete {self.nonce}'}
                self.final_served.set()
            else:
                message = {'role': 'assistant', 'content': 'still working',
                           'tool_calls': [{'id': cid, 'type': 'function',
                                           'function': {'name': tool, 'arguments': json.dumps(args)}}]}
        self.ledger.write(json.dumps(row) + '\n')
        return message

    def snapshot(self) -> dict:
        """Write what the provider has seen so far; written again at close (the lists only grow)."""
        with self.lock:
            seen = {'fixture-hits.json': list(self.hits), 'beacons.json': list(self.beacons),
                    'issued-calls.json': list(self.issued)}
        for name, value in seen.items():
            (self.out / name).write_text(json.dumps(value, indent=1))
        return {'issued': seen['issued-calls.json'], 'final_served': self.final_served.is_set(),
                'fixture_hits': len(seen['fixture-hits.json']), 'beacons': len(seen['beacons.json'])}

    def close(self) -> dict:
        self.server.shutdown()
        self.ledger.close()
        return self.snapshot()


def http(method, url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode('utf-8', 'replace')


def wait_for(read, what, timeout, process=None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f'{what}: launcher exited first with {process.returncode}')
        try:
            value = read()
            if value:
                return value
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f'timed out: {what}')


def run(args, env=None, timeout=20):
    try:
        done = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
        return {'argv': args, 'rc': done.returncode, 'stdout': done.stdout[-2000:], 'stderr': done.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'argv': args, 'rc': None, 'error': repr(exc)}


def bus_reach(env):
    """Did THIS user's client get an answer from each bus? A socket file is not reach.
    It says nothing about whether Chromium itself connected (see pw:browser stderr)."""
    call = ['call', 'org.freedesktop.DBus', '/org/freedesktop/DBus', 'org.freedesktop.DBus', 'GetId']
    return {'system': run(['busctl', '--system', *call], env, 5), 'session': run(['busctl', '--user', *call], env, 5),
            'session_address_present': bool(env.get('DBUS_SESSION_BUS_ADDRESS'))}


def payload_facts(payload: Path):
    facts = {'entry': str(payload / 'Ouroboros'), 'entry_is_script': False, 'internal_json': {}}
    try:
        facts['entry_is_script'] = (payload / 'Ouroboros').read_bytes()[:2] == b'#!'
    except OSError as exc:
        facts['entry_error'] = repr(exc)
    for path in sorted((payload / '_internal').glob('*.json')):
        if path.stat().st_size < 65536:
            try:
                facts['internal_json'][path.name] = json.loads(path.read_text())
            except ValueError:
                pass
    return facts


def copy_tree(src: Path, dst: Path) -> dict:
    """Bounded copy that records its own outcome; an absent source is recorded as absent."""
    if not src.is_dir():
        return {'present': False}
    done = run(['cp', '-a', str(src), str(dst)], timeout=20)
    if done.get('rc') != 0:
        done.setdefault('error', f"cp rc={done.get('rc')}")
    return {'present': True, **done}


def stop_launcher(process, grace: int) -> dict:
    if process is None:
        return {'started': False}
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)  # the launcher's ordinary quit path
        try:
            process.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            return {'returncode': None, 'error': f'still running {grace} s after SIGTERM'}
    return {'returncode': process.returncode}


class Deadline(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--payload', type=Path, required=True)
    ap.add_argument('--home', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--public-url', default='')
    ap.add_argument('--hold', type=float, default=8.0)
    ap.add_argument('--deadline', type=int, default=700, help='bound before teardown, seconds')
    args = ap.parse_args()
    out, home = args.out, args.home
    out.mkdir(parents=True, exist_ok=True)
    runtime = home / '.xdg-runtime'
    for d in (home, runtime, home / '.tmp'):
        d.mkdir(parents=True, exist_ok=True)
    runtime.chmod(0o700)
    nonce = uuid.uuid4().hex
    env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': str(home), 'USER': 'qa', 'LOGNAME': 'qa',
           'LANG': 'C.UTF-8', 'SHELL': '/bin/bash', 'TMPDIR': str(home / '.tmp'), 'XDG_RUNTIME_DIR': str(runtime),
           'BROWSER': '/bin/true', 'DEBUG': 'pw:browser'}
    if os.environ.get('DBUS_SESSION_BUS_ADDRESS'):
        env['DBUS_SESSION_BUS_ADDRESS'] = os.environ['DBUS_SESSION_BUS_ADDRESS']
    facts = {'arm': args.arm, 'nonce': nonce, 'driver_pid': os.getpid(), 'caller_env': env,
             'payload': payload_facts(args.payload), 'bus_reach_before_launch': bus_reach(env), 'steps': []}

    def step(name, **data):
        facts['steps'].append({'ts': time.time(), 'step': name, **data})
        (out / 'driver.json').write_text(json.dumps(facts, indent=1))

    def on_deadline(*_):
        raise Deadline(f'driver deadline {args.deadline}s')

    signal.signal(signal.SIGALRM, on_deadline)
    signal.signal(signal.SIGTERM, on_deadline)
    signal.alarm(args.deadline)
    provider = ScriptedProvider(out, nonce, args.public_url, args.hold)
    step('provider', port=provider.port)
    data = home / 'Ouroboros' / 'data'
    port_file, settings_path = data / 'state' / 'server_port', data / 'settings.json'
    port_file.unlink(missing_ok=True)
    process, task_id, base = None, '', ''
    try:
        if settings_path.exists():  # only the documented manual arm C reuses a configured HOME
            saved = json.loads(settings_path.read_text())
            step('configured_home_endpoint_refreshed', before=saved.get('OPENAI_COMPATIBLE_BASE_URL'))
            saved['OPENAI_COMPATIBLE_BASE_URL'] = f'http://127.0.0.1:{provider.port}/v1'
            settings_path.write_text(json.dumps(saved, indent=2))
        with (out / 'launcher-stdout.log').open('wb') as log:
            process = subprocess.Popen([str(args.payload / 'Ouroboros')], env=env, cwd=str(home), stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        step('launcher_started', container_pid=process.pid)
        port = wait_for(lambda: int(port_file.read_text()), 'server port', 600, process)
        base = f'http://127.0.0.1:{port}'
        health = wait_for(lambda: json.loads(http('GET', f'{base}/api/health', timeout=3)[1]), 'health', 300, process)
        step('health', port=port, health=health)
        status, _ = http('GET', f'{base}/api/onboarding')
        step('onboarding_state', status=status)
        if status != 204:
            # Keys verified at both tags: subscriptionsConnected/skipSubscriptionPresets
            # (settings_setup_contract.py:255-256 at v7.4.4, :274-275 at v7.9.0);
            # starter_skills read only by v7.9.0 (onboarding.py:920, skill_starter.py:347);
            # v7.4.4's validate_setup_payload reads named provider fields only, so an
            # unread key is ignored there. The saved settings are recorded below.
            body = {'OPENAI_COMPATIBLE_BASE_URL': f'http://127.0.0.1:{provider.port}/v1',
                    'OPENAI_COMPATIBLE_API_KEY': 'ob1234-fixture-not-a-credential',
                    'OUROBOROS_MODEL': SLUG, 'OUROBOROS_MODEL_LIGHT': SLUG,
                    'OUROBOROS_RUNTIME_MODE': 'advanced', 'OUROBOROS_REVIEW_ENFORCEMENT': 'advisory',
                    'subscriptionsConnected': False, 'skipSubscriptionPresets': True, 'starter_skills': []}
            status, text = http('POST', f'{base}/api/onboarding/complete', body, timeout=90)
            step('onboarding_complete', status=status, body=text[:4000])
            if status != 200:
                raise RuntimeError(f'onboarding refused: {status}')
        try:
            saved = json.loads(settings_path.read_text())
            step('saved_settings', values={k: ('<fixture>' if k.endswith('_KEY') else v) for k, v in saved.items()
                                           if k.startswith(('OUROBOROS_MODEL', 'OUROBOROS_SAFETY', 'OUROBOROS_TASK_REVIEW',
                                                            'OUROBOROS_RUNTIME_MODE', 'OUROBOROS_STARTER',
                                                            'OUROBOROS_POST_TASK', 'OPENAI_COMPATIBLE'))})
        except (OSError, ValueError) as exc:
            step('saved_settings_unreadable', error=repr(exc))
        seen = {}

        def admission_ready():
            # Both tags answer POST /api/tasks with 503 until supervisor_ready_event is set and the pool has
            # a worker (gateway/tasks.py::_supervisor_ready_error); /api/state reports that event as
            # supervisor_ready and len(WORKERS) as workers_total. Other refusals appear only in the POST answer.
            seen.clear()
            try:
                seen['status'], seen['body'] = http('GET', f'{base}/api/state', timeout=5)
                state = json.loads(seen['body'])
            except (OSError, ValueError) as exc:
                seen['error'] = repr(exc)
                return False
            return (seen['status'] == 200 and isinstance(state, dict) and state.get('supervisor_ready') is True
                    and isinstance(state.get('workers_total'), int) and state['workers_total'] > 0)

        try:
            wait_for(admission_ready, 'supervisor and worker admission readiness (/api/state)', 300, process)
        finally:
            step('admission_state', **seen)  # the exact last answer, ready or not
        try:
            status, text = http('POST', f'{base}/api/tasks', {
                'description': f'OB1234 chain probe {nonce}: open the local fixture with browse_page, take one '
                               'screenshot with browser_action, then answer.',
                'title': 'OB1234 chain probe', 'timeout_sec': 600}, timeout=60)
        except Exception as exc:  # admitted or not is unknown; never re-posted
            step('task_submit_outcome_unknown', error=repr(exc))
            raise
        try:
            task_id = str((json.loads(text) or {}).get('task_id') or '') if 200 <= status < 300 else ''
        except (ValueError, AttributeError):
            task_id = ''
        step('task_submitted', status=status, body=text, task_id=task_id)
        if not task_id:  # refused or unconfirmed: recorded once above, never re-posted
            raise RuntimeError(f'POST /api/tasks gave no task_id (HTTP {status}); not retried')
        while not provider.final_served.is_set():
            if process.poll() is not None:
                raise RuntimeError(f'launcher exited during task with {process.returncode}')
            time.sleep(1)
        step('final_served_after_all_issued_results')
    except Exception as exc:  # Deadline included: recorded first, teardown always follows
        step('error', error=repr(exc))
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # teardown below is bounded (about 110 s); run.sh has its own bound

        def guarded(name, action):  # each teardown step stands alone and records its own failure
            try:
                step(name, **action())
            except Exception as exc:
                step(name, error=repr(exc))

        guarded('issued_calls', provider.snapshot)  # accumulated provider facts first, before any HTTP
        if task_id and base:
            for name, method, path, body in (('task_detail', 'GET', f'/api/tasks/{task_id}', None),
                                             ('state_at_task_detail', 'GET', '/api/state', None),
                                             ('task_cancel', 'POST', f'/api/tasks/{task_id}/cancel', {})):
                guarded(name, lambda: dict(zip(('status', 'body'), http(method, base + path, body, timeout=5))))
        logs = data / 'logs'
        guarded('logs_before_stop', lambda: copy_tree(logs, out / 'logs-before-stop'))  # evidence BEFORE stopping
        guarded('launcher_exit', lambda: stop_launcher(process, 30))
        guarded('provider_closed', provider.close)
        guarded('logs_after_stop', lambda: copy_tree(logs, out / 'logs'))
        repo = home / 'Ouroboros' / 'repo'
        guarded('managed_repo', lambda: {'rev_parse_head': run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], timeout=10),
                                         'version_file': run(['cat', str(repo / 'VERSION')], timeout=5)})
        guarded('ps_after_quit', lambda: {'ps': run(['ps', '-u', 'qa', '-o', 'pid=,ppid=,stat=,lstart=,args='], timeout=10)})
    # No silent success: any recorded error, teardown included, makes the exit status non-zero.
    return 1 if any(s['step'] == 'error' or 'error' in s for s in facts['steps']) else 0


if __name__ == '__main__':
    sys.exit(main())
