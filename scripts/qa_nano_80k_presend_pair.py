"""One-shot paired startup on Ubuntu; exact sources and a disposable count-only venv.

The fence stops the built-in SDK at the HTTP transport, before a generation
request goes on the wire. The private vendor copy independently disables model
evaluation, sampling and generation. Only the stock /extras/measure_chat does
real formatter/tokenizer work. A fence hit is an observation, never a product
refusal, a successful response or a fit guarantee.
"""
from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request


SOURCES = {
    'base': ('1e2ad9737e25a1b8d438dd7dfc1f028728f66816', '52120a3dd3f1aa8f564082b5667a1aba7249c3d3'),
    'candidate': ('b176d771eeee6ad32905a29bd32943b979f32fdb', 'bb1a03790a01208a8187d777fcb3dd5fd88b01a3'),
}
FIXTURE_SHA = 'ee0c814bd781707fc0cda00d81294f6bf5b5c0d2fba1fd7fb4bd1c48581237b1'
LLAMA_SHA = '2e6cf4e2b1ed5ae04ecc57679c04240e0ef5bcec8932d07e068cc25fae05a2e0'
MODEL_FILE = 'Qwen3.5-0.8B-Q4_K_M.gguf'
BASE_URL = 'http://127.0.0.1:8765'
MEASURE_URL = 'http://127.0.0.1:8766/extras/measure_chat'
GUARD_MESSAGE = 'QA_NO_GENERATION'
EVENTS = 'qa-events.jsonl'
EVENT_BUDGET = 64 * 1024
HOOKED = False


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def code(value):
    return value if isinstance(value, str) and re.fullmatch(r'[a-zA-Z0-9_.:-]{1,100}', value) else None


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    temp.replace(path)


def event(kind, **facts):
    # One atomic append. Never retain HTTP bodies, headers, log lines or provider speech.
    path = Path(os.environ['NANO_PRESEND_CASE']) / EVENTS
    payload = (json.dumps({'kind': kind, 'mono_ns': time.monotonic_ns(), **facts}, separators=(',', ':'))
               + '\n').encode('ascii')
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if os.fstat(fd).st_size + len(payload) >= EVENT_BUDGET or os.write(fd, payload) != len(payload):
            raise OSError('qa_event_not_recorded')
    finally:
        os.close(fd)


def observe(kind, **facts):
    try:
        event(kind, **facts)
    except (OSError, ValueError):
        # Lack of observer evidence must neither change a real count nor open a generation path.
        pass


def request_facts(body):
    try:
        data = json.loads(body)
        messages = data.get('messages', [])
        return {'exact_fixture_rows': sum(
            isinstance(row, dict) and row.get('role') == 'user'
            and isinstance(text := row.get('content'), str) and sha(text) == FIXTURE_SHA for row in messages),
                'max_tokens': data['max_tokens'] if type(data.get('max_tokens')) is int else None}
    except (AttributeError, TypeError, ValueError):
        return {'exact_fixture_rows': None}


def install_hooks():
    """Startup hook only in the runner venv, only for the two isolated units."""
    global HOOKED
    if not os.environ.get('NANO_PRESEND_CASE') or HOOKED:
        return
    import httpx
    import requests

    original_http = httpx.HTTPTransport.handle_request
    original_send = requests.Session.send

    def guarded_http(self, request):
        url = request.url
        if (url.host in ('127.0.0.1', 'localhost') and url.port == 8766
                and request.method != 'GET' and str(url) != MEASURE_URL):
            # The actual sync SDK selected on both versions has arrived here;
            # there is no call to the original transport in this branch.
            try:
                event('qa_transport_stop', chat_route=url.path == '/v1/chat/completions',
                      **request_facts(request.read()))
            except (OSError, ValueError, httpx.StreamError):
                # Distinct, non-retryable QA failure when the witness cannot be written; never forward.
                return httpx.Response(422, request=request, json={'error': {
                    'message': 'QA_STOP_EVIDENCE_UNAVAILABLE', 'code': 'qa_stop_evidence_unavailable', 'type': 'qa_probe'}})
            return httpx.Response(400, request=request,
                json={'error': {'message': GUARD_MESSAGE, 'code': 'qa_no_generation', 'type': 'qa_probe'}})
        return original_http(self, request)

    def measured_send(self, request, **kwargs):
        if request.url != MEASURE_URL or request.method != 'POST':
            return original_send(self, request, **kwargs)
        binding = request_facts(request.body)
        observe('measure_started', **binding)
        try:
            response = original_send(self, request, **kwargs)
        except Exception as exc:
            observe('measure_failed', error_type=code(type(exc).__name__), **binding)
            raise
        # The stock call is non-streaming. Requests has already buffered this body.
        try:
            doc = response.json() if response.status_code == 200 else {}
            sample = {key: doc[key] for key in ('input_tokens', 'context_window', 'process_id')
                      if type(doc.get(key)) is int}
            sample.update(exact=doc.get('input_is_exact') is True, supported=doc.get('supported') is True,
                          reason=code(doc.get('reason')))
        except (AttributeError, TypeError, ValueError):
            sample = {'body_read': 'unavailable'}
        observe('measure_returned', http=response.status_code, **sample, **binding)
        return response

    httpx.HTTPTransport.handle_request = guarded_http
    requests.Session.send = measured_send
    HOOKED = True


def prepare(root):
    import sysconfig
    from importlib.metadata import version

    require(sys.platform == 'linux' and Path(sys.prefix).resolve() == root / 'venv', 'wrong_venv')
    site = Path(sysconfig.get_path('purelib')).resolve()
    require(site.is_relative_to(root / 'venv') and version('llama-cpp-python') == '0.3.38', 'wrong_vendor')
    vendor = site / 'llama_cpp' / 'llama.py'
    require(vendor.resolve().is_relative_to(site), 'vendor_outside_venv')
    original = vendor.read_bytes()
    require(sha(original) == LLAMA_SHA, 'vendor_sha_changed')
    text = original.decode('utf-8')
    cls, = [n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'Llama']
    guarded = {'eval', 'generate', 'sample'}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in guarded]
    require({n.name for n in methods} == guarded and len(methods) == len(guarded), 'vendor_guard_anchors')
    lines = text.splitlines(keepends=True)
    for fn in sorted(methods, key=lambda n: n.lineno, reverse=True):
        require(ast.get_docstring(fn) is not None, 'vendor_guard_docstring_anchor')
        lines.insert(fn.body[0].end_lineno, f'        raise RuntimeError({GUARD_MESSAGE!r})\n')
    patched = ''.join(lines)
    ast.parse(patched)
    vendor.write_text(patched, encoding='utf-8')
    helper = Path(__file__).read_bytes()
    with (site / '_qa_nano_presend_pair.py').open('xb') as out:
        out.write(helper)
    with (site / 'qa_nano_presend_pair.pth').open('x', encoding='utf-8') as out:
        out.write('import _qa_nano_presend_pair; _qa_nano_presend_pair.install_hooks()\n')

    from llama_cpp import Llama

    # The exact synthetic body from the preceding experiment; do not reuse its client/task id.
    vocab = Llama(model_path=str(root / MODEL_FILE), vocab_only=True, verbose=False)
    try:
        material = ' '.join(sha(f'nano-80k-observed-20261128-a quiet sample {i}')[:8] for i in range(7000))

        def prompt(n):
            return ('What object is named on the final line? Reply with that word only. The intervening labels are synthetic.\n'
                    'BEGIN QA TEXT\n' + material[:n] + '\nEND QA TEXT\nObject: LANTERN')

        def tokens(value):
            return len(vocab.tokenize(value.encode(), add_bos=False, special=False))

        lo, hi = 0, len(material)
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if tokens(prompt(mid)) >= 15_000:
                hi = mid
            else:
                lo = mid
        message = prompt(hi)
        require(sha(message) == FIXTURE_SHA and tokens(message) == 15_000, 'fixture_identity_or_count')
        (root / 'private-prompt.txt').write_text(message, encoding='utf-8')
    finally:
        vocab.close()
    workflow = Path(__file__).absolute().parent.parent / '.github/workflows/nano-80k-presend-pair.yml'
    write(root / 'out/prepared.json', {'script_sha256': sha(helper), 'workflow_sha256': sha(workflow.read_bytes()),
        'vendor_before_sha256': LLAMA_SHA, 'vendor_count_only_sha256': sha(patched),
        'fixture_sha256': FIXTURE_SHA, 'fixture_gguf_vocab_tokens': 15_000})


def bind_source(root, side):
    from importlib.util import find_spec

    repo = root / f'source-{side}'
    require(Path.cwd() == repo, 'wrong_source_cwd')
    sys.path.insert(0, str(repo))
    spec = find_spec('ouroboros.cli')
    require(spec is not None and Path(spec.origin).resolve() == repo / 'ouroboros/cli.py', 'wrong_module_origin')
    return repo


def boot(root, side):
    from _qa_nano_presend_pair import HOOKED as active

    require(active, 'qa_transport_hook_missing')
    repo = bind_source(root, side)
    from ouroboros import cli
    from ouroboros.config import resolve_data_dir

    require(Path(cli.__file__).resolve() == repo / 'ouroboros/cli.py', 'wrong_loaded_cli')
    require(resolve_data_dir().resolve() == root / side / 'data', 'wrong_live_data_root')
    event('bound_cli_import', module='ouroboros.cli', scope='verified_detached_source', data_root='fresh_case')
    sys.argv = ['ouroboros', 'server', '--host', '127.0.0.1', '--port', '8765']
    raise SystemExit(cli.main())


def read_local(path):
    with urllib.request.urlopen(BASE_URL + path, timeout=1.5) as response:
        return response.status, json.loads(response.read() or b'{}')


def time_ms(limit, cutoff):
    left = int((cutoff - time.time() - 8) * 1000)
    if left < 1:
        raise TimeoutError('qa_case_cutoff')
    return min(limit, left)


async def web_probe(ctx, cutoff, message, data):
    from playwright.async_api import async_playwright
    from ouroboros.context_attempt_reader import list_attempts
    from ouroboros.task_status import FINAL_STATUSES
    from ouroboros.task_store import list_documents, read_document

    exact_text = sha(' '.join(message.split()))
    owner = hint = bound = None
    frames = 0

    async def read(path):
        response = await ctx['browser_context'].request.get(BASE_URL + path, timeout=time_ms(2500, cutoff))
        try:
            return response.status, await response.json() if response.status == 200 else {}
        finally:
            await response.dispose()

    def matches_origin(ref):
        return (owner is not None and isinstance(ref, dict) and ref.get('chat_id') == 1
                and ref.get('client_message_id') == owner and ref.get('text_sha256') == exact_text)

    def system_fact(row):
        if not (isinstance(row, dict) and row.get('role') == 'system' and
                row.get('system_type') != 'initialization_notice' and
                (matches_origin(row.get('origin_message_ref')) or (bound and row.get('task_id') == bound))):
            return
        notice = row.get('notice') or {}
        fact = {k: code(row.get(k)) for k in ('system_type', 'task_terminal_status', 'reason_code')}
        if isinstance(notice, dict):
            fact['notice_code'] = code(notice.get('code'))
            fact['notice_step'] = code(notice.get('step'))
        if fact not in ctx['system_rows'] and len(ctx['system_rows']) < 12:
            ctx['system_rows'].append(fact)

    def socket(ws):
        sent_owner = None

        def sent(raw):
            nonlocal owner, frames, sent_owner
            row = decode(raw)
            if row.get('type') == 'chat' and row.get('content') == message:
                frames += 1
                cid = row.get('client_message_id')
                if isinstance(cid, str) and cid and owner is None and row.get('chat_id', 1) == 1:
                    owner = sent_owner = cid
                    ctx['client_id_sha256'] = sha(cid)
                ctx['matching_frames'] = frames

        def received(raw):
            row = decode(raw)
            if (sent_owner and row.get('client_message_id') == sent_owner and row.get('chat_id') == 1
                    and row.get('type') == 'chat' and row.get('role') == 'system'
                    and row.get('system_type') == 'initialization_notice'):
                ctx['initialization_notice'] = {'source': 'same_socket_client_message_id',
                                                'reason_code': code(row.get('reason_code'))}
            if (owner and row.get('client_message_id') == owner and row.get('chat_id') == 1
                    and row.get('role') == 'user' and row.get('ingress_accepted') is True):
                ctx['accepted_echo'] = True
            system_fact(row)

        ws.on('framesent', sent)
        ws.on('framereceived', received)

    def decode(raw):
        try:
            doc = json.loads(raw)
            return doc if isinstance(doc, dict) else {}
        except (TypeError, ValueError):
            return {}

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(executable_path='/usr/bin/google-chrome', chromium_sandbox=True,
                                           headless=True, timeout=time_ms(15_000, cutoff))
        try:
            ctx['browser_context'] = await browser.new_context(locale='en-US', viewport={'width': 1280, 'height': 850})
            page = await ctx['browser_context'].new_page()
            page.on('websocket', socket)
            await page.goto(BASE_URL + '/', wait_until='domcontentloaded', timeout=time_ms(15_000, cutoff))
            require(page.url == BASE_URL + '/', 'unexpected_browser_location')
            box = page.locator('#chat-input')
            await box.wait_for(state='visible', timeout=time_ms(15_000, cutoff))
            _, settings = await read('/api/settings')
            if settings.get('OUROBOROS_CONTEXT_MODE') != 'nano':
                await page.locator('.chat-context-mode .chat-context-head').click(timeout=time_ms(7000, cutoff))
                await page.locator('.chat-context-strip [role=radio][data-mode=nano]').click(timeout=time_ms(7000, cutoff))
            for _ in range(16):
                _, settings = await read('/api/settings')
                if settings.get('OUROBOROS_CONTEXT_MODE') == 'nano':
                    break
                await asyncio.sleep(0.25)
            require(settings.get('OUROBOROS_CONTEXT_MODE') == 'nano', 'nano_not_observed')
            ctx['phase'] = 'nano_confirmed'
            ctx['selected_context_mode'] = 'nano'
            save(ctx)
            await box.fill(message, timeout=time_ms(8000, cutoff))
            require(await box.input_value(timeout=time_ms(5000, cutoff)) == message, 'composer_did_not_retain_fixture')
            ctx['phase'] = 'one_click_attempted_no_retry'
            ctx['click_mono_ns'] = time.monotonic_ns()
            save(ctx)
            try:
                await page.locator('#chat-send').click(timeout=time_ms(8000, cutoff))
                ctx['click_returned'] = True
            except Exception as exc:
                ctx['click_observation_error'] = code(type(exc).__name__)

            while time.time() < cutoff - 13:
                try:
                    status, state = await read('/api/state')
                    ctx['state_http'] = status
                    if status == 200 and owner:
                        rows = [r for r in state.get('active_direct_turns', []) if r.get('chat_id') == 1
                                and r.get('kind') == 'direct_chat' and r.get('client_message_id') == owner]
                        ctx['active_snapshot'] = 'one' if len(rows) == 1 else 'none' if not rows else 'ambiguous'
                        if len(rows) == 1:
                            row = rows[0]
                            hint = row.get('activity_id') or hint
                            ctx['last_live'] = {k: code(row.get(k)) for k in ('phase', 'turn_phase', 'carrier')}
                            ctx['live_id_sha256'] = sha(hint) if isinstance(hint, str) else None
                except Exception as exc:
                    ctx['state_read_error'] = code(type(exc).__name__)
                try:
                    status, history = await read('/api/chat/history?chat_id=1&n_human=16&n_progress=0')
                    ctx['history_http'] = status
                    for row in history.get('messages', []):
                        system_fact(row)
                except Exception as exc:
                    ctx['history_read_error'] = code(type(exc).__name__)
                try:
                    if owner:
                        doc = read_document(data, bound or hint) if (bound or hint) else None
                        if not matches_origin((doc or {}).get('origin_message_ref')):
                            found = [item for item in list_documents(data) if matches_origin(item.get('origin_message_ref'))]
                            doc = found[0] if len(found) == 1 else None
                        if doc and isinstance(tid := doc.get('task_id'), str):
                            bound = tid
                            ctx['durable_task_sha256'] = sha(tid)
                            execution = (doc.get('outcome_axes') or {}).get('execution') or {}
                            if doc.get('status') in FINAL_STATUSES:
                                ctx.setdefault('terminal', {'status': code(doc.get('status')),
                                    'reason_code': code(doc.get('reason_code') or execution.get('reason_code')),
                                    'origin': code(doc.get('terminal_origin')), 'seen_mono_ns': time.monotonic_ns()})
                            attempts = list_attempts(data, tid)
                            r = attempts['attempts']
                            ctx['attempts_visible'] = len(r)
                            ctx['host_dispatch_marks_visible'] = sum(bool(a.get('ts_dispatched')) for a in r)
                            ctx['attempts_complete'] = not attempts.get('next_after')
                except Exception as exc:
                    ctx['task_or_attempt_read_error'] = code(type(exc).__name__)
                observation = safe_results(ctx['dir'], ctx)
                save(ctx, observation)
                if observation['outcome'] in DECISIVE_OUTCOMES:
                    return
                await asyncio.sleep(0.6)
        finally:
            await asyncio.wait_for(browser.close(), timeout=3)


def read_events(directory):
    try:
        raw = (directory / EVENTS).read_bytes()
        if len(raw) >= EVENT_BUDGET or not raw.endswith(b'\n'):
            return 'incomplete', []
        return 'read', [json.loads(line) for line in raw.splitlines()]
    except FileNotFoundError:
        return 'absent', []
    except (OSError, ValueError):
        return 'unreadable', []


DECISIVE_OUTCOMES = ('qa_boundary', 'native_ingress_refusal_observed', 'native_terminal_before_boundary')


def safe_results(directory, facts):
    coverage, events = read_events(directory)
    after_click = [r for r in events if r['mono_ns'] >= facts.get('click_mono_ns', 2**100)]
    own = [r for r in after_click if r.get('exact_fixture_rows') == 1]
    stops = [r for r in own if r.get('kind') == 'qa_transport_stop' and r.get('chat_route')]
    measure_kinds = ('measure_started', 'measure_returned', 'measure_failed')
    measures = [r for r in own if r.get('kind') in measure_kinds]
    single_send = facts.get('matching_frames') == 1
    compatible_ingress = (single_send and (facts.get('accepted_echo')
                          or facts.get('live_id_sha256') or facts.get('durable_task_sha256')))
    terminal = facts.get('terminal') or {}
    systems = facts.get('system_rows', [])
    terminal_system = any(row.get('task_terminal_status') in ('completed', 'failed', 'cancelled', 'rejected_duplicate')
                          or row.get('system_type') == 'task_not_started' for row in systems)
    any_stop = any(r.get('kind') == 'qa_transport_stop' for r in after_click)
    init = facts.get('initialization_notice')
    # Only these typed producers attest a pre-dispatch refusal on the pinned trees.
    # Generic host failures (including one after losing the QA witness) do not.
    native_reason = None
    if terminal.get('origin') == 'host_notice' and terminal.get('reason_code') == 'context_capacity':
        native_reason = 'context_capacity'
    elif any(row.get('system_type') == 'task_not_started' and row.get('notice_code') == 'turn_not_accepted'
             and row.get('notice_step') == 'send_after_restart' for row in systems):
        native_reason = 'turn_not_accepted'
    if stops and compatible_ingress:
        outcome = 'qa_boundary'  # Subsequent host terminal/notice is downstream of a test-only 400.
    elif single_send and init and not compatible_ingress and not any_stop:
        outcome = 'native_ingress_refusal_observed'  # Reports the keyed WS rejection, not whether anything was saved.
    elif single_send and native_reason and not any_stop:
        outcome = 'native_terminal_before_boundary'
    elif terminal or terminal_system:
        outcome = 'terminal_cause_unknown'
    elif init or any_stop:
        outcome = 'diagnostics_incomplete'
    elif facts.get('active_snapshot') == 'one' and facts.get('phase') in ('one_click_attempted_no_retry', 'observation_ended'):
        outcome = 'unfinished_at_last_live_read'
    else:
        outcome = 'diagnostics_incomplete'
    last_proven = ('qa_transport_before_wire' if outcome == 'qa_boundary' else
                   'bound_websocket_initialization_notice' if outcome == 'native_ingress_refusal_observed' else
                   'correlated_native_terminal' if outcome == 'native_terminal_before_boundary' else
                   'exact_count_returned' if any(r.get('kind') == 'measure_returned' and r.get('exact')
                                                 and r.get('supported') for r in measures) else
                   'counter_http_entered' if measures else
                   'turn_phase_observed' if (facts.get('last_live') or {}).get('turn_phase') else
                   'accepted_ingress' if compatible_ingress else
                   'browser_outbound' if facts.get('matching_frames') else facts.get('phase'))
    # Event entries have only synthetic facts from the hook, never a body or a log.
    return {'outcome': outcome, 'last_proven': last_proven,
        'native_refusal_reason': (native_reason if outcome == 'native_terminal_before_boundary' else
                                  init.get('reason_code') if outcome == 'native_ingress_refusal_observed' else None),
        'event_coverage': coverage, 'bound_cli_import': any(
        r.get('kind') == 'bound_cli_import' for r in events),
        'count_events_bound_to_fixture': measures[:12], 'count_events_bound_total': len(measures),
        'counter_observations_all_after_click': {k: sum(r.get('kind') == k for r in after_click) if coverage == 'read'
                                                  else None for k in measure_kinds},
        'counter_observations_all_case': {k: sum(r.get('kind') == k for r in events) if coverage == 'read'
                                          else None for k in measure_kinds},
        'probe_chat_transport_stops_for_fixture': len(stops) if coverage == 'read' else None,
        'probe_total_transport_stops': sum(r.get('kind') == 'qa_transport_stop' for r in events)
                                       if coverage == 'read' else None,
        'first_probe_transport_stop': stops[0] if stops else None}


def save(ctx, observed=None):
    facts = {k: v for k, v in ctx.items() if k not in ('browser_context', 'dir', 'path')}
    write(ctx['path'], {**facts, **(observed or safe_results(ctx['dir'], facts))})


def probe(root, side, cutoff):
    from _qa_nano_presend_pair import HOOKED as active

    directory = root / side
    ctx = {'dir': directory, 'path': root / 'out' / f'{side}.json', 'case': side, 'phase': 'preflight',
           'system_rows': [], 'matching_frames': 0, 'accepted_echo': False}
    save(ctx)
    server = None
    try:
        require(sys.platform == 'linux' and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'not_loopback_only')
        require(active and Path(sys.prefix).resolve() == root / 'venv', 'missing_qa_fence')
        require(os.environ.get('LOGITS_ALL') == 'false', 'wrong_shared_logits_setting')
        ctx['logits_all_launch_env'] = False
        repo = root / f'source-{side}'
        require(Path.cwd() == repo, 'wrong_source_cwd')
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD', 'HEAD^{tree}'], cwd=repo, timeout=4,
                                       text=True).splitlines()
        require(tuple(head) == SOURCES[side], 'wrong_subject')
        status = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=repo, timeout=4)
        require(not status, 'subject_not_clean')
        ctx.update(product_commit=head[0], product_tree=head[1], phase='source_verified')
        save(ctx)
        message = (root / 'private-prompt.txt').read_text(encoding='utf-8')
        require(sha(message) == FIXTURE_SHA, 'fixture_bytes_changed')
        data, home = directory / 'data', directory / 'home'
        require(not data.exists(), 'data_already_present')
        home.mkdir(mode=0o700)
        os.environ.update(HOME=str(home), XDG_CACHE_HOME=str(home / '.cache'), XDG_CONFIG_HOME=str(home / '.config'),
            OUROBOROS_DATA_DIR=str(data), OUROBOROS_SETTINGS_PATH=str(data / 'settings.json'), OUROBOROS_REPO_DIR=str(repo),
            LOCAL_MODEL_SOURCE=str(root / MODEL_FILE), LOCAL_MODEL_CONTEXT_LENGTH='80000', LOCAL_MODEL_N_GPU_LAYERS='0',
            LOCAL_MODEL_PORT='8766', USE_LOCAL_MAIN='true', USE_LOCAL_LIGHT='true', OUROBOROS_MODEL_FALLBACKS='',
            OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        bind_source(root, side)
        ctx['phase'] = 'starting_real_cli'
        save(ctx)
        with (directory / 'private-server.log').open('xb') as log:
            server = subprocess.Popen([sys.executable, str(Path(__file__).absolute()), 'boot', str(root), side],
                                      cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            startup_end = min(cutoff - 35, time.time() + 125)
            while time.time() < startup_end:
                require(server.poll() is None, 'cli_exited_early')
                try:
                    health = read_local('/api/health')[1].get('boot', {})
                    if health.get('dialogue_ready') and health.get('tasks_ready'):
                        ctx['phase'] = 'dialogue_and_tasks_ready'
                        ctx['onboarding_http'] = read_local('/api/onboarding')[0]
                        save(ctx)
                        require(ctx['onboarding_http'] == 204, 'ordinary_preconfigured_start_not_ready')
                        local = read_local('/api/local-model/status')[1]
                        ctx['local_status'] = code(local.get('status'))
                        save(ctx)
                        if local.get('status') == 'ready':
                            applied = local.get('settings_application', {})
                            require(local.get('model_path') == str(root / MODEL_FILE) and local.get('port') == 8766
                                    and applied.get('status') == 'ready' and not applied.get('pending_keys')
                                    and not applied.get('unknown_keys'), 'different_local_model_or_pending_settings')
                            ctx['phase'] = 'ordinary_local_ready'
                            save(ctx)
                            break
                except (OSError, ValueError) as exc:
                    if isinstance(exc, ValueError) and exc.args and code(exc.args[0]):
                        raise
                    ctx['last_boot_read_error'] = code(type(exc).__name__)
                    save(ctx)
                time.sleep(0.5)
            require(ctx['phase'] == 'ordinary_local_ready', 'startup_not_ready_within_slice')
            coverage, boot_events = read_events(directory)
            require(coverage == 'read' and any(r.get('kind') == 'bound_cli_import' for r in boot_events), 'cli_import_witness_missing')
            asyncio.run(web_probe(ctx, cutoff, message, data))
            ctx['observation_end'] = 'returned'  # a harness marker, not a task status
    except Exception as exc:
        ctx.update(probe_error_type=code(type(exc).__name__), probe_error_code=code(exc.args[0]) if exc.args else None)
    finally:
        save(ctx)
        if server is not None and server.poll() is None:
            server.terminate()  # exact child only; the unit owns all descendants
            try:
                server.wait(timeout=3)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=1)
    return 0 if safe_results(directory, ctx)['outcome'] in DECISIVE_OUTCOMES else 1


def collect(root):
    for side in SOURCES:
        directory = root / side
        path = root / 'out' / f'{side}.json'
        facts = json.loads(path.read_text())
        exit_path = directory / 'unit-exit'
        if exit_path.is_file():
            facts['unit_exit'] = int(exit_path.read_text())
        if facts.get('phase') or directory.is_dir():
            facts.update(safe_results(directory, facts))
            if not facts.get('phase'):
                facts['probe_did_not_record_start'] = True
        write(path, facts)


if __name__ == '__main__':
    command = sys.argv[1]
    root = Path(sys.argv[2]).resolve()
    if command == 'prepare':
        prepare(root)
    elif command == 'boot':
        boot(root, sys.argv[3])
    elif command == 'probe':
        raise SystemExit(probe(root, sys.argv[3], int(sys.argv[4])))
    elif command == 'collect':
        collect(root)
    else:
        raise SystemExit('unknown QA command')
