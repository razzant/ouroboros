"""One User/Nano qualification on a disposable Ubuntu runner; also its private vendor observer."""
from __future__ import annotations

import ast
import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import time
import urllib.request


PRODUCT = 'b176d771eeee6ad32905a29bd32943b979f32fdb'
BASE = 'http://127.0.0.1:8765'
FACTS = {'product': PRODUCT, 'send': 'not_clicked'}
STAGE = 'preflight'
WORD = 'LANTERN'
RULES = 'OUROBOROS_SAFETY_OWNER_RULES'
SMALL = 'Synthetic QA note: the sample workshop has a slate-blue shelf.'
WINDOW = 80_000
MIN_REPLY_TOKENS = 8_192
RUNTIME = re.compile(r'(?m)^## Runtime context\n\n(?=\{)')
VENDOR_LOG = 'private-nano-llama-stages.jsonl'
VENDOR_VERSION = '0.3.38'
VENDOR_CAP = 128_000
VENDOR_FILES = {
    'server/app.py': '81c5a5be463cfb89b6961fe5c0085234a9004f7d2e41571b224ede45dd47cba7',
    'llama.py': '2e6cf4e2b1ed5ae04ecc57679c04240e0ef5bcec8932d07e068cc25fae05a2e0',
}
REQUEST = ContextVar('nano_qa_only_chat', default=None)
ORDINAL = itertools.count(1)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(text):
    return sha_bytes(text.encode('utf-8'))


def digest_string(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def vendor_event(event, **facts):
    """Only our copied files in the runner venv import this. No inputs or headers are retained."""
    if root := os.getenv('NANO_QA_ONLY_DIR'):
        try:
            raw = (json.dumps({'event': event, 'at': utc(), 'pid': os.getpid(), **facts},
                              separators=(',', ':')) + '\n').encode('ascii')
            path = Path(root) / VENDOR_LOG
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                if os.fstat(fd).st_size < VENDOR_CAP:
                    os.write(fd, raw)  # one append from this private loopback server
            finally:
                os.close(fd)
        except (OSError, ValueError):
            pass  # absent evidence is not a reason to alter the owner's HTTP request


async def vendor_chat_received(request):
    """Existing authenticated/validated POST handler, before its own inference lock."""
    ordinal = next(ORDINAL)
    facts = {'request': ordinal, 'body_digest_observed': False}
    try:
        from ouroboros.request_wire_contract import physical_candidate_sha256

        body = await request.json()  # same parsed Request cache FastAPI just used, not an extra read from the wire
        if isinstance(body, dict):
            wanted = os.getenv('NANO_QA_ONLY_FRAGMENT_SHA256')
            matches = sum(row.get('role') == 'user' and isinstance(row.get('content'), str)
                          and sha(row['content']) == wanted for row in body.get('messages', [])
                          if isinstance(row, dict))
            facts.update(body_digest_observed=True, canonical_body_sha256=physical_candidate_sha256(body),
                         exact_user_rows=matches, user_fragment_sha256=wanted if matches == 1 else None,
                         streamed=body.get('stream') is True,
                         requested_max_tokens=body.get('max_tokens') if type(body.get('max_tokens')) is int else None)
    except Exception:
        pass  # never adopt an observer error as the vendor's error or record its text
    vendor_event('generation_handler_entered', **facts)
    return ordinal


@contextmanager
def vendor_chat_scope(ordinal):
    token = REQUEST.set(ordinal)
    try:
        yield
    finally:
        REQUEST.reset(token)


def vendor_generation_entered(model, tokens):
    if (ordinal := REQUEST.get()) is not None:
        try:
            n_ctx = model.n_ctx()
        except Exception:
            n_ctx = None  # Missing observer evidence must not affect generation.
        vendor_event('token_generator_entered', request=ordinal, input_tokens=len(tokens),
                     n_ctx=n_ctx if type(n_ctx) is int and n_ctx > 0 else None)


def instrument_vendor(app_src, llama_src):
    """Produce two narrow source insertions without replacing any auth/route/model options."""
    def inject(source, line, insertion, module_import):
        pre, post = source.split('\n', 1)
        assert pre == 'from __future__ import annotations'
        lines = (pre + '\n' + post).splitlines(keepends=True)
        lines.insert(line - 1, insertion)
        lines.insert(1, module_import + '\n')
        value = ''.join(lines)
        ast.parse(value)
        return value

    funcs = {node.name: node for node in ast.parse(app_src).body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    handler = funcs['create_chat_completion']
    assert isinstance(handler, ast.AsyncFunctionDef) and '/v1/chat/completions' in ast.unparse(handler.decorator_list[0])
    assert 'Depends(authenticate)' in ast.unparse(handler.decorator_list[0])
    assert ast.unparse(handler.body[0]) == 'body_model = body.model'
    assert ast.unparse(handler.body[-1]).startswith('async with contextlib.asynccontextmanager(get_llama_proxy)()')
    anchor = ast.unparse(handler.body[-1].body[-1])
    assert anchor == 'return await run_in_threadpool(llama.create_chat_completion, **kwargs)'
    # No scope or waits before the existing lock except a synchronous redacted fact append from the already parsed Request.
    app_src = app_src.replace('        ' + anchor + '\n',
                             '        with _nano_qa_scope(_nano_qa_ordinal):\n'
                             '            _nano_qa_answer = await run_in_threadpool(llama.create_chat_completion, **kwargs)\n'
                             '        _nano_qa_event("nonstream_handler_returned", request=_nano_qa_ordinal)\n'
                             '        return _nano_qa_answer\n', 1) if app_src.count('        ' + anchor + '\n') == 1 else None
    assert app_src is not None
    # Record that this process really constructed the app; the runtime's import-only readiness probe does not do so.
    app_src = inject(app_src, handler.body[0].lineno,
                     '    _nano_qa_ordinal = await _nano_qa_received(request)\n',
                     'from llama_cpp._nano_qa_only import (vendor_chat_received as _nano_qa_received,\n'
                     '    vendor_chat_scope as _nano_qa_scope, vendor_event as _nano_qa_event)')
    lines = app_src.splitlines(keepends=True)
    app_factory, = [n for n in ast.parse(app_src).body if isinstance(n, ast.FunctionDef) and n.name == 'create_app']
    assert ast.unparse(app_factory.body[-1]) == 'return app'
    lines.insert(app_factory.body[-1].lineno - 1, '    _nano_qa_event("model_app_constructed")\n')
    app_src = ''.join(lines)
    ast.parse(app_src)

    model, = [n for n in ast.parse(llama_src).body if isinstance(n, ast.ClassDef) and n.name == 'Llama']
    generator, = [n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'generate']
    assert ast.get_docstring(generator) and generator.args.args[1].arg == 'tokens'
    assert 'self._mirostat_mu =' in ast.unparse(generator.body[1])
    assert any(isinstance(n, ast.Yield) for n in ast.walk(generator))
    llama_src = inject(llama_src, generator.body[0].end_lineno + 1, '        _nano_qa_generate(self, tokens)\n',
                       'from llama_cpp._nano_qa_only import vendor_generation_entered as _nano_qa_generate')
    return app_src, llama_src


def install_vendor_observer():
    from importlib.metadata import version

    venv = (ROOT / 'venv').resolve()
    package = (Path(sysconfig.get_path('purelib')) / 'llama_cpp').resolve()
    assert Path(sys.prefix).resolve() == venv and package.is_relative_to(venv)
    assert version('llama-cpp-python') == VENDOR_VERSION
    originals = []
    for name, expected in VENDOR_FILES.items():
        path = package / name
        assert path.resolve().is_relative_to(venv)
        raw = path.read_bytes()
        actual = sha_bytes(raw)
        if actual != expected:
            record(vendor_observer={'state': 'source_pin_mismatch_no_launch', 'path': name,
                                    'expected_sha256': expected, 'observed_sha256': actual})
            raise ValueError('pinned vendor Python source changed')
        originals.append((path, raw.decode('utf-8')))
    changed = instrument_vendor(*(text for _, text in originals))
    helper = Path(__file__).read_bytes()
    with (package / '_nano_qa_only.py').open('xb') as out:
        out.write(helper)
    after = {}
    for (path, _), text in zip(originals, changed):
        raw = text.encode('utf-8')
        path.write_bytes(raw)
        after[str(path.relative_to(package))] = sha_bytes(raw)
    record(vendor_observer={'state': 'installed_in_runner_venv_only', 'version': VENDOR_VERSION,
                            'before_sha256': VENDOR_FILES, 'after_sha256': after,
                            'private_helper_sha256': sha_bytes(helper)})


def record(**facts):
    FACTS.update(facts)
    tmp = ROOT / 'facts.tmp'
    tmp.write_text(json.dumps(FACTS, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(ROOT / 'facts.json')


def begin(stage):
    global STAGE
    STAGE = stage
    record(phase=stage)


def remaining_ms(limit=60_000):
    left = int((CUTOFF - time.time() - 7) * 1000)
    if left < 1:
        raise TimeoutError('QA cutoff')
    return min(limit, left)


def read(path):
    with urllib.request.urlopen(BASE + path, timeout=2) as response:
        return response.status, response.read()


def wait_until(probe, *, seconds=60):
    until = time.monotonic() + seconds
    while time.monotonic() < until:
        remaining_ms()
        try:
            result = probe()
            if result:
                return result
        except (OSError, ValueError):
            pass  # bounded startup/custody reads, never a mutation retry
        time.sleep(0.5)
    raise TimeoutError('readiness or evidence not observed')


def synthetic_inputs():
    # Vocabulary-only read of the verified GGUF. This launches no server and generates no tokens.
    from llama_cpp import Llama

    vocab = Llama(model_path=str(MODEL), vocab_only=True, verbose=False)
    try:
        material = ' '.join(sha(f'nano-80k-observed-20261128-a quiet sample {i}')[:8] for i in range(7000))
        def prompt(lines):
            return (f'What object is named on the final line? Reply with that word only. The intervening labels are synthetic.\n'
                    'BEGIN QA TEXT\n' + material[:lines] + f'\nEND QA TEXT\nObject: {WORD}')

        def tokens(s):
            return len(vocab.tokenize(s.encode('utf-8'), add_bos=False, special=False))

        hi, lo = len(material), 0
        while lo + 1 < hi:
            mid = (hi + lo) // 2
            if tokens(prompt(mid)) >= 15_000:
                hi = mid
            else:
                lo = mid
        message = prompt(hi)
        big = 'Synthetic capacity witness; the following fixture has no instructions.\n' + material
        count, large_count = tokens(message), tokens(big)
        assert 15_000 <= count <= 15_020
        assert large_count > 20_000 and len(big.encode()) <= 64_000
        record(prompt_text_tokens=count, rules_large_text_tokens=large_count,
               rules_large_bytes=len(big.encode()), prompt_sha256=sha(message),
               count_basis='GGUF vocabulary, no chat template/BOS')
        return message, big
    finally:
        vocab.close()


def keep_first(key, value):
    if key not in FACTS:
        record(**{key: value})


def count_message_text(rows, message):
    runtime_blocks = runtime_copies = 0

    def count(value, *, decoded=False):
        nonlocal runtime_blocks, runtime_copies
        if isinstance(value, str):
            if decoded:
                return value.count(message)
            total, offset = 0, 0
            for marker in RUNTIME.finditer(value):
                if marker.start() < offset:
                    continue
                body, end = json.JSONDecoder().raw_decode(value, marker.end())
                assert isinstance(body, dict)
                copies = count(body, decoded=True)
                runtime_blocks += 1
                runtime_copies += copies
                total += value[offset:marker.end()].count(message) + copies
                offset = end
            return total + value[offset:].count(message)
        if isinstance(value, dict):
            return sum(count(v, decoded=decoded) for v in value.values())
        if isinstance(value, list):
            return sum(count(v, decoded=decoded) for v in value)
        return 0

    exact = [row['content'] for row in rows if isinstance(row, dict) and row.get('role') == 'user'
             and row.get('content') == message]
    occurrences = count(rows)
    return {'full_text_occurrences': occurrences, 'exact_source_user_rows': len(exact),
            'full_user_fragment_sha256': sha(exact[0]) if len(exact) == 1 else None,
            'decoded_runtime_blocks': runtime_blocks, 'runtime_json_full_text_occurrences': runtime_copies,
            'first_exact_user_text': (occurrences, len(exact)) == (1, 1) and runtime_blocks == 1}


def candidate_facts(task, row, message):
    from ouroboros.context_attempt_reader import read_attempt
    from ouroboros.observability import call_manifest_path
    from ouroboros.request_wire_contract import physical_candidate_sha256

    attempt = row['attempt_id']
    witness, body = read_attempt(DATA, task, attempt, report_facts=True)
    facts = {'seal_read': witness.get('body'), 'host_transport_observation': witness.get('transport_observation')}
    failure = witness.get('failure') or {}
    http_status = failure.get('provider_status_code')
    if type(http_status) is int and 100 <= http_status <= 599:
        facts['local_transport_reported_http_status'] = http_status
    if body is None:
        return facts
    facts['candidate_projection_sha256'] = sha_bytes(body)
    layout = json.loads(body)
    exclusions = witness.get('host_exclusions')
    if isinstance(exclusions, list) and all(isinstance(e, dict) for e in exclusions) and not any(
        e.get('class') not in ('transport_envelope', 'provider_side_transform')
        and (not isinstance(path := e.get('path'), str) or path == '$' or path.startswith(('$.messages', 'messages')))
        for e in exclusions
    ) and isinstance(rows := layout.get('messages'), list):
        try:
            facts.update(count_message_text(rows, message))
        except (ValueError, AssertionError):
            facts.update(runtime_json_status='not_decodable', first_exact_user_text=None)
    fit = witness.get('context_fit') or {}
    facts.update(observed_mode='nano' if fit.get('rendered_mode') == 'nano' else 'other_or_unknown',
                 wire_output_allowance=layout.get('max_tokens') if type(layout.get('max_tokens')) is int else None)
    # read_attempt above verifies the seal and binding. The same loose manifest can also
    # name the serving-formatter witness. Never copy the witness or request itself out.
    try:
        head = json.loads(call_manifest_path(DATA, task, attempt).read_bytes())
        seal = head.get('model_send_seal') or {}
        # read_attempt verified the pre-redaction seal against its attempt row.
        # Do not substitute the redacted CAS digest for that canonical identity.
        projection = head.get('redacted_projection_ref') or {}
        digest = seal.get('pre_redaction_sha256')
        if (head.get('task_id') == task and head.get('call_type') == 'physical_llm_candidate'
                and seal.get('attempt_id') == attempt and projection.get('sha256') == witness.get('sha256')
                and seal.get('canonical_basis') == 'canonical_json_v1' and digest_string(digest)):
            facts['host_canonical_body_sha256'] = digest
            facts['projection_canon_matches_host_seal'] = physical_candidate_sha256(layout) == digest
            floor = head.get('nano_minimum') or {}
            actual = floor.get('actual') or {}
            facts['first_precount'] = {
                'status': 'exact' if floor.get('status') == 'exact' else 'not_observed',
                **{key: actual[key] for key in ('input_tokens', 'sent_max_tokens') if type(actual.get(key)) is int},
                **{key: floor[key] for key in ('context_window', 'reply_floor_tokens') if type(floor.get(key)) is int},
            }
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return facts


def find_response(task, attempt):
    from ouroboros.observability import read_call_payload

    matches = []
    for path in (DATA / 'observability' / 'calls' / task).glob('*_response.json'):
        header = json.loads(path.read_bytes())
        if header.get('call_type') != 'llm_response' or header.get('provider') != 'local':
            continue
        _, body, _ = read_call_payload(DATA, task_id=task, call_id=path.stem)
        usage = body.get('usage') or {}
        if usage.get('provider') == 'local' and (usage.get('ledger_attempt_ids') or [None])[-1] == attempt:
            matches.append((body, usage))
    if len(matches) != 1:
        return {'model_response': 'not_observed' if not matches else 'ambiguous'}
    body, usage = matches[0]
    answer = (body.get('message') or {}).get('content')
    text = answer.strip() if isinstance(answer, str) else ''
    return {'model_response': 'received', 'model_reply_has_visible_text': bool(text),
            'model_reply_raw_sha256': sha(answer) if isinstance(answer, str) else None,
            'model_reply_sha256': sha(text) if text else None, 'model_reply_word_ok': text == WORD,
            **{f'reported_{key}': usage[key] for key in ('prompt_tokens', 'completion_tokens')
               if type(usage.get(key)) is int}}


def observe_attempts(task, message):
    from ouroboros.context_attempt_reader import list_attempts

    # Ledger order uses seq_first, not timestamps; pages are newest first. The
    # earliest dispatched row must not be replaced by a later, healthier retry.
    page = list_attempts(DATA, task)
    rows = list(page['attempts'])
    while page['next_after'] and len(rows) < 160:
        page = list_attempts(DATA, task, after=page['next_after'])
        rows.extend(page['attempts'])
    if page['next_after']:
        return {'attempt_observation': 'page_bound_reached', 'physical_order_complete': False}
    sent = [row for row in reversed(rows) if row.get('ts_dispatched')]
    facts = {'physical_order_complete': True, 'physical_attempts_observed': len(rows),
             'host_dispatches_observed': len(sent)}
    if rows:
        facts['earliest_reservation_at'] = rows[-1].get('ts_reserved')
        facts['earliest_reservation_state'] = rows[-1].get('state')
    if not sent:
        return facts
    first = sent[0]
    aid = first['attempt_id']
    facts.update(first_dispatched_attempt_sha256=sha(aid), first_host_dispatch_at=first['ts_dispatched'],
                 first_dispatch_reserved_at=first.get('ts_reserved'), first_ledger_state=first.get('state'),
                 first_local_route=first.get('provider') == 'local')
    if FACTS.get('first_dispatched_attempt_sha256') not in (None, sha(aid)):
        record(first_attempt_order_changed=True)
    # Publish lifecycle first so a concurrent seal/read failure never erases it.
    record(**facts)
    if first.get('provider') != 'local':
        return facts
    try:
        facts.update(candidate_facts(task, first, message))
    except Exception as error:
        facts['seal_read_error'] = type(error).__name__
    record(**facts)
    try:
        facts.update(find_response(task, aid))
    except Exception as error:
        facts['model_response_read_error'] = type(error).__name__
    return facts


def observe_vendor_log(root, facts):
    """Only the owned event file is read: no vendor logs, request bodies or headers."""
    events = []
    try:
        raw = (root / VENDOR_LOG).read_bytes()
    except FileNotFoundError:
        raw = b''
    except OSError:
        return {'vendor_stage_coverage': 'unreadable'}
    if len(raw) >= VENDOR_CAP or (raw and not raw.endswith(b'\n')):
        return {'vendor_stage_coverage': 'incomplete'}
    try:
        for line in raw.splitlines():
            event = json.loads(line)
            assert event['event'] in {'model_app_constructed', 'generation_handler_entered',
                                      'token_generator_entered', 'nonstream_handler_returned'}
            assert type(event['pid']) is int and isinstance(event['at'], str)
            events.append(event)
    except (AssertionError, ValueError, TypeError, KeyError):
        return {'vendor_stage_coverage': 'incomplete'}
    constructions = [r for r in events if r['event'] == 'model_app_constructed']
    receipts = [r for r in events if r['event'] == 'generation_handler_entered']
    result = {'vendor_stage_coverage': 'app_constructed' if constructions else 'app_construction_not_observed',
              'model_app_constructions': len(constructions), 'generation_handler_entries': len(receipts),
              'token_generator_entries': sum(r['event'] == 'token_generator_entered' for r in events),
              'first_server_binding': 'not_observed'}
    if not receipts:
        return result
    first = receipts[0]
    canonical = first.get('canonical_body_sha256')
    user_sha = first.get('user_fragment_sha256')
    result.update(first_handler_entry_at=first['at'], first_handler_user_rows=first.get('exact_user_rows'),
                  first_handler_user_sha256=user_sha if digest_string(user_sha) else None,
                  first_handler_canonical_sha256=canonical if digest_string(canonical) else None)
    if len(constructions) != 1 or first['pid'] != constructions[0]['pid']:
        return {**result, 'first_server_binding': 'server_instance_not_unique'}
    if len(receipts) != 1 or facts.get('first_attempt_order_changed'):
        return {**result, 'first_server_binding': 'ambiguous'}
    if not digest_string(canonical) or canonical != facts.get('host_canonical_body_sha256'):
        return {**result, 'first_server_binding': 'canonical_host_witness_missing_or_different'}
    if (not facts.get('origin_bound') or not digest_string(user_sha) or user_sha != facts.get('prompt_sha256')
            or first.get('exact_user_rows') != 1):
        return {**result, 'first_server_binding': 'durable_origin_or_full_fragment_missing'}
    group = [r for r in events if r['pid'] == first['pid'] and r.get('request') == first['request']]
    entries = [r for r in group if r['event'] == 'token_generator_entered']
    returned = [r for r in group if r['event'] == 'nonstream_handler_returned']
    result['first_server_binding'] = 'canonical_and_full_fragment'
    if len(returned) == 1:
        result['bound_handler_returned_at'] = returned[0]['at']  # value to framework, not delivery
    if len(entries) == 1:
        n = entries[0]['input_tokens']
        cap = first.get('requested_max_tokens')
        n_ctx = entries[0].get('n_ctx')
        if type(n_ctx) is not int or n_ctx <= 0:
            n_ctx = None
        fit_window = min(WINDOW, n_ctx) if n_ctx is not None else None
        result.update(bound_token_generator_entered_at=entries[0]['at'], bound_generator_input_tokens=n,
                      bound_server_requested_max_tokens=cap, bound_generator_n_ctx=n_ctx,
                      bound_generator_n_ctx_status='observed' if n_ctx is not None else 'unknown',
                      bound_fit_window=fit_window,
                      first_fit_observed=(facts.get('first_exact_user_text') is True
                          and facts.get('full_user_fragment_sha256') == user_sha
                          and fit_window is not None
                          and type(n) is int and type(cap) is int and cap >= MIN_REPLY_TOKENS
                          and n >= (facts.get('prompt_text_tokens') or WINDOW)
                          and n + cap <= fit_window and cap == facts.get('wire_output_allowance')))
    return result


def chat_last_evidence(facts):
    vendor = facts.get('vendor_observation') or {}
    if vendor.get('first_server_binding') == 'canonical_and_full_fragment':
        if vendor.get('bound_token_generator_entered_at'):
            if facts.get('visible_reply_matches_provider') is True:
                return 'own_visible_reply_and_bound_generator_entry'
            if facts.get('model_response') == 'received':
                return 'own_local_response_and_bound_generator_entry'
            if vendor.get('bound_handler_returned_at'):
                return 'bound_generator_entered_and_handler_returned'
            return 'bound_generator_entered'
        return 'bound_authenticated_handler_entered'
    if facts.get('visible_reply_matches_provider') is True:
        return 'own_visible_reply_server_entry_unbound'
    if facts.get('model_response') == 'received':
        return 'own_local_response_server_entry_unbound'
    if vendor.get('generation_handler_entries'):
        return 'unbound_handler_observed'
    if facts.get('first_host_dispatch_at'):
        return 'host_mark_before_transport'
    if facts.get('earliest_reservation_at'):
        return 'host_reserved'
    if facts.get('origin_bound') or facts.get('ingress_owner_echo_accepted_at'):
        return 'ingress_accepted'
    if facts.get('websocket_send_seen_at'):
        return 'browser_websocket_send'
    return facts.get('send', 'not_clicked')


def read_origin(owner_id, message, hint):
    from ouroboros.task_store import list_documents, read_document

    def match(doc):
        ref = (doc or {}).get('origin_message_ref') or {}
        return (ref.get('chat_id'), ref.get('client_message_id'), ref.get('text_sha256')) == (
            1, owner_id, sha(' '.join(message.split())))

    if hint and match(read_document(DATA, hint)):
        return hint
    matches = [doc['task_id'] for doc in list_documents(DATA) if doc.get('task_id') and match(doc)]
    return matches[0] if len(matches) == 1 else None


async def get_json(context, path):
    response = await context.request.get(BASE + path, timeout=remaining_ms(3_000))
    try:
        if response.status != 200:
            return response.status, {}
        return response.status, await response.json()
    finally:
        await response.dispose()


async def open_rules(page, *, limit=20_000):
    await page.locator('[data-nav-page=settings]:visible').click(timeout=remaining_ms(limit))
    await page.locator('[data-settings-open=rules]:visible').click(timeout=remaining_ms(limit))
    field = page.locator('#settings-rules')
    await field.wait_for(state='visible', timeout=remaining_ms(limit))
    await page.wait_for_function("() => document.querySelector('#settings-rules')?.disabled === false",
                                 timeout=remaining_ms(limit))
    return field


async def save_rules(page, field, text, label):
    fields = {f'{label}_save_response': 'not_observed'}
    posts = 0

    def submitted(request):
        nonlocal posts
        if request.url == BASE + '/api/owner/safety-rules' and request.method == 'POST':
            posts += 1
            record(**{f'{label}_save_posts_observed': posts})

    page.on('request', submitted)
    try:
        await field.fill(text, timeout=remaining_ms(8_000))
        async with page.expect_response(lambda r: r.url == BASE + '/api/owner/safety-rules' and r.request.method == 'POST',
                                        timeout=remaining_ms(65_000)) as event:
            record(**{f'{label}_save_click_attempted_at': utc()})
            await page.locator('[data-settings-page=rules] [data-editor-save]').click(timeout=remaining_ms(8_000))
        response = await event.value
        fields.update({f'{label}_http': response.status, f'{label}_response_seen_at': utc()})
        record(**fields)
        body = await asyncio.wait_for(response.json(), remaining_ms(8_000) / 1000)
        if not isinstance(body, dict):
            return
    except Exception as error:
        record(**fields, **{f'{label}_save_read_error': type(error).__name__})
        return
    finally:
        page.remove_listener('request', submitted)
    code = body.get('code')
    allowed = {'nano_publication_' + reason for reason in (
        'over_capacity', 'busy', 'unavailable', 'stale', 'bootstrap_pending', 'incomplete')}
    record(**{f'{label}_save_response': 'decoded', f'{label}_code': code if code in allowed else 'other' if code else None,
              f'{label}_saved_flag': body.get('saved') if type(body.get('saved')) is bool else None,
              f'{label}_ok_flag': body.get('ok') is True,
              f'{label}_rules_echo_exact': body.get('rules') == text,
              f'{label}_proposal_echo_exact': body.get('proposed_rules') == text})


async def fresh_rules_page(context):
    page = await context.new_page()
    await page.goto(BASE + '/', wait_until='domcontentloaded', timeout=remaining_ms(16_000))
    assert page.url == BASE + '/'
    return page, await open_rules(page, limit=8_000)


async def settings_readback(context, label, small, large):
    """Both reads run after each Save even when its POST/response timed out; neither read retries the write."""
    def classify(value):
        return ('small' if value == small else 'large' if value == large else
                'other' if isinstance(value, str) else 'not_observed')

    async def api():
        try:
            status, settings = await get_json(context, '/api/settings')
            active = classify(settings.get(RULES)) if status == 200 else 'not_observed'
            record(**{f'{label}_readback_http': status, f'{label}_active_api': active,
                      f'{label}_api_readback_at': utc()})
            return active
        except Exception as error:
            record(**{f'{label}_active_api': 'not_observed', f'{label}_api_read_error': type(error).__name__})
            return 'not_observed'

    async def reopened():
        try:
            page, field = await fresh_rules_page(context)
            shown = classify(await field.input_value(timeout=remaining_ms(4_000)))
            record(**{f'{label}_active_reopened_ui': shown, f'{label}_ui_readback_at': utc()})
            return shown, page, field
        except Exception as error:
            record(**{f'{label}_active_reopened_ui': 'not_observed', f'{label}_ui_read_error': type(error).__name__})
            return 'not_observed', None, None

    # Fresh documents avoid the draft living in the old JS module. Do not reload the
    # saving tab (which could abort an in-flight POST after an uncertain outcome).
    active, (shown, page, field) = await asyncio.gather(api(), reopened())
    return active, shown, page, field


async def settings_branch(context, go, large):
    await go.wait()
    try:
        record(settings_status='partial', settings_started_at=utc())
        page, field = await fresh_rules_page(context)
        if await field.input_value(timeout=remaining_ms(4_000)) == SMALL:
            record(large_outcome='not_attempted_small_fixture_already_present')
            return False
        try:
            await save_rules(page, field, SMALL, 'small')
        finally:
            active, shown, page, field = await settings_readback(context, 'small', SMALL, large)
        if ((active, shown) != ('small', 'small') or not FACTS.get('small_save_click_attempted_at')
                or FACTS.get('small_save_posts_observed', 0) > 1):
            record(large_outcome='not_attempted_without_confirmed_small_baseline')
            return False
        record(small_baseline_evidence='exact_api_and_reopened_ui', small_retained_sha256=sha(SMALL))
        try:
            await save_rules(page, field, large, 'large')
        finally:
            active, shown, _, _ = await settings_readback(context, 'large', SMALL, large)
        prior = (active, shown) == ('small', 'small')
        explicit_refusal = (FACTS.get('large_http') == 409 and FACTS.get('large_saved_flag') is False
                            and FACTS.get('large_code') == 'nano_publication_over_capacity')
        if 'large' in (active, shown):
            outcome = 'large_became_active_no_more_writes'
        elif prior:
            outcome = 'over_capacity_prior_preserved' if explicit_refusal else 'prior_observed_response_not_proven'
        else:
            outcome = 'readback_incomplete_or_disagrees'
        record(large_outcome=outcome, prior_rules_survived_api=None if active == 'not_observed' else active == 'small',
               prior_rules_survived_reopened_ui=None if shown == 'not_observed' else shown == 'small')
        ok = prior and explicit_refusal and FACTS.get('large_save_posts_observed') == 1
        record(settings_status='observed' if ok else 'partial')
        return ok
    except Exception as error:
        record(settings_status='partial', settings_error_type=type(error).__name__)
        return False


async def chat_branch(page, context, go, message):
    owner = None
    echo = False
    hint = None
    frames = 0

    def socket(ws):
        def sent(payload):
            nonlocal owner, frames
            try:
                row = json.loads(payload)
            except (ValueError, TypeError):
                return
            if isinstance(row, dict) and row.get('type') == 'chat' and row.get('content') == message:
                frames += 1
                if isinstance(cid := row.get('client_message_id'), str) and cid and owner is None:
                    owner = cid
                    record(client_message_id_sha256=sha(cid), websocket_send_seen_at=utc())
                record(matching_websocket_outbound_frames=frames)

        def received(payload):
            nonlocal echo, hint
            try:
                row = json.loads(payload)
            except (ValueError, TypeError):
                return
            if (owner and isinstance(row, dict) and row.get('chat_id') == 1
                    and row.get('client_message_id') == owner):
                if row.get('type') == 'chat' and row.get('role') == 'user' and row.get('ingress_accepted') is True:
                    echo = True
                    keep_first('ingress_owner_echo_accepted_at', utc())
                if row.get('type') == 'typing' and isinstance(row.get('task_id'), str):
                    hint = row['task_id']

        ws.on('framesent', sent)
        ws.on('framereceived', received)

    page.on('websocket', socket)
    try:
        await page.goto(BASE + '/', wait_until='domcontentloaded', timeout=remaining_ms(20_000))
        assert page.url == BASE + '/'  # no guessed sign-in or setup
        box = page.locator('#chat-input')
        assert await page.locator('#chat-messages .chat-bubble.assistant:not(.typing-bubble):not(.progress) > .message').count() == 0
        await box.wait_for(state='visible', timeout=remaining_ms(20_000))
        _, settings = await get_json(context, '/api/settings')
        if settings.get('OUROBOROS_CONTEXT_MODE') != 'nano':
            await page.locator('.chat-context-mode .chat-context-head').click(timeout=remaining_ms(8_000))
            await page.locator('.chat-context-strip [role=radio][data-mode=nano]').click(timeout=remaining_ms(8_000))
        until = time.monotonic() + 25
        while time.monotonic() < until:
            _, settings = await get_json(context, '/api/settings')
            if settings.get('OUROBOROS_CONTEXT_MODE') == 'nano':
                break
            await asyncio.sleep(0.5)
        assert settings.get('OUROBOROS_CONTEXT_MODE') == 'nano'
        record(saved_context_mode='nano')
        await box.fill(message, timeout=remaining_ms(15_000))
        assert await box.input_value(timeout=remaining_ms(8_000)) == message
        # Keep time for the two independent readbacks even if this sole Send stalls.
        assert remaining_ms(450_000) == 450_000, 'too little time for both branches'
        record(send='click_attempted_no_retry', click_at=utc())
        clicked = time.monotonic()
        try:
            await page.locator('#chat-send').click(timeout=remaining_ms(8_000))
            record(send='click_returned_no_retry')
        except Exception as error:
            record(click_error_type=type(error).__name__)
        task = None
        # Settings receives time even if the chat never answers. If an answer is
        # fast, release the second tab earlier to avoid competing for the counter.
        settings_release = min(clicked + 720, time.monotonic() + CUTOFF - time.time() - 380)
        while time.time() < CUTOFF - 30:
            try:
                if not owner:
                    data = await page.locator('#chat-messages .chat-bubble.user').evaluate_all(
                        "(nodes, text) => nodes.filter(n => n.querySelector('.message')?.innerText?.replace(/\\s+/g, ' ').trim() === text).map(n => n.dataset.clientMessageId).filter(Boolean)",
                        ' '.join(message.split()))
                    if len(set(data)) == 1 and data:
                        owner = data[0]
                        record(client_message_id_sha256=sha(owner), browser_user_row_observed=True)
                if owner and not task:
                    _, snapshot = await get_json(context, '/api/state')
                    live = [r for r in snapshot.get('active_direct_turns', []) if r.get('chat_id') == 1
                            and r.get('kind') == 'direct_chat' and r.get('client_message_id') == owner]
                    if len(live) == 1:
                        hint = live[0].get('activity_id') or hint
                        phase = live[0].get('phase')
                        record(live_turn_phase=phase if phase in ('preparing', 'working', 'unknown') else 'other')
                    task = await asyncio.to_thread(read_origin, owner, message, hint)
                    if task:
                        record(origin_bound=True, origin_task_sha256=sha(task), origin_bound_at=utc())
                if task:
                    record(attempt_binding='durable_origin')
                    record(**observe_attempts(task, message))
                elif hint:
                    record(attempt_binding='provisional_active_turn_only')

                vendor = observe_vendor_log(ROOT, FACTS)
                record(vendor_observation=vendor, chat_last_proven=chat_last_evidence({**FACTS, 'vendor_observation': vendor}))
                # Normally avoid competing with the inference lock. If it never returns,
                # hand Settings its own window anyway, after persisting this snapshot.
                if time.monotonic() >= settings_release or FACTS.get('model_response') == 'received':
                    go.set()
                if task:
                    # An assistant bubble from another task (or a typing/status row) is not the answer.
                    texts = await page.locator('#chat-messages .chat-bubble.assistant:not(.typing-bubble):not(.progress)').evaluate_all(
                        "(nodes, id) => nodes.filter(n => n.dataset.taskId === id && n.getClientRects().length).map(n => n.querySelector(':scope > .message')?.innerText?.trim()).filter(Boolean)", task)
                    if texts:
                        keep_first('assistant_dom_observed_at', utc())
                        expected_hash = FACTS.get('model_reply_sha256')
                        matched = any(sha(text) == expected_hash for text in texts)
                        record(assistant_dom_observed=True, visible_reply_matches_provider=matched,
                               visible_word_ok=(expected_hash == sha(WORD)) if matched else None)
                    if FACTS.get('visible_reply_matches_provider'):
                        go.set()
                        break
            except Exception as error:
                record(last_chat_read_error=type(error).__name__)
                if time.monotonic() >= settings_release:
                    go.set()
            await asyncio.sleep(1.5 if not task else 2.5)
        record(chat_observation_ended_at=utc(), ingress_echo_seen=echo)
        vendor = observe_vendor_log(ROOT, FACTS)
        record(vendor_observation=vendor, chat_last_proven=chat_last_evidence({**FACTS, 'vendor_observation': vendor}))
        ok = chat_complete(FACTS)
        record(chat_status='observed' if ok else 'partial')
        return ok
    except Exception as error:
        record(chat_status='partial', chat_error_type=type(error).__name__)
        return False
    finally:
        go.set()


def chat_complete(facts):
    vendor = facts['vendor_observation']
    return (facts.get('origin_bound') is True and facts.get('matching_websocket_outbound_frames') == 1
            and facts.get('first_local_route') is True and facts.get('observed_mode') == 'nano'
            and vendor.get('generation_handler_entries') == vendor.get('token_generator_entries') == 1
            and vendor.get('first_fit_observed') is True and vendor.get('bound_handler_returned_at')
            and facts.get('reported_prompt_tokens') == vendor.get('bound_generator_input_tokens')
            and facts.get('model_response') == 'received' and facts.get('model_reply_has_visible_text') is True
            and facts.get('visible_reply_matches_provider') is True)


def assert_ready():
    assert read('/api/onboarding')[0] == 204
    ready = json.loads(read('/api/health')[1]).get('boot', {})
    return ready if ready.get('dialogue_ready') and ready.get('tasks_ready') else None


async def scenario(message, large):
    from playwright.async_api import async_playwright

    boot = wait_until(assert_ready, seconds=190)
    record(onboarding_http=204, dialogue_ready=True, tasks_ready=True, boot_phase=boot.get('phase'))
    begin('local_model_readiness')

    def local_ready():
        row = json.loads(read('/api/local-model/status')[1])
        if row.get('status') == 'error':
            raise RuntimeError('model service did not reach ready')
        return row if row.get('status') == 'ready' else None

    local = wait_until(local_ready, seconds=250)
    assert local.get('port') == 8766 and local.get('model_path') == str(MODEL)
    applied = local.get('settings_application', {})
    assert applied.get('status') == 'ready' and not applied.get('pending_keys') and not applied.get('unknown_keys')
    record(local_service_ready=True, local_context_reported=local.get('context_length'))
    begin('parallel_browser_observation')
    record(chat_status='partial', settings_status='not_started')
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path='/usr/bin/google-chrome', chromium_sandbox=True,
                                          headless=True, timeout=remaining_ms(20_000))
        try:
            context = await browser.new_context(viewport={'width': 1280, 'height': 850}, locale='en-US')
            page = await context.new_page()
            go = asyncio.Event()
            chat, settings = await asyncio.gather(chat_branch(page, context, go, message),
                                                  settings_branch(context, go, large))
            record(status='observed' if chat and settings else 'partial', phase='finished')
            return chat and settings
        finally:
            await asyncio.wait_for(browser.close(), timeout=4)

def main():
    record(status='partial', phase=STAGE)
    try:
        assert sys.platform == 'linux' and ROOT.is_dir()
        assert {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'expected new loopback-only namespace'
        assert not DATA.exists() and MODEL.is_file()
        home = ROOT / 'home'
        home.mkdir(mode=0o700)
        # Ordinary environment-only deployment. No owner settings or setup/auth state is fabricated.
        os.environ.update(HOME=str(home), XDG_CACHE_HOME=str(home / '.cache'),
                          OUROBOROS_DATA_DIR=str(DATA), OUROBOROS_SETTINGS_PATH=str(DATA / 'settings.json'),
                          OUROBOROS_REPO_DIR=str(Path.cwd()), LOCAL_MODEL_SOURCE=str(MODEL),
                          LOCAL_MODEL_CONTEXT_LENGTH='80000', LOCAL_MODEL_N_GPU_LAYERS='0', LOCAL_MODEL_PORT='8766',
                          USE_LOCAL_MAIN='true', USE_LOCAL_LIGHT='true',
                          OUROBOROS_MODEL_FALLBACKS='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
        sys.path.insert(0, str(Path.cwd()))  # import the verified source checkout, not an installed app
        begin('install_only_runner_vendor_observer')
        install_vendor_observer()
        begin('count_fixture')
        # The server receives only the comparison hash through its own inherited test environment.
        message, large = synthetic_inputs()
        os.environ.update(NANO_QA_ONLY_DIR=str(ROOT), NANO_QA_ONLY_FRAGMENT_SHA256=sha(message))
        begin('headless_boot')
        with (ROOT / 'private-server.log').open('wb') as log:
            server = subprocess.Popen([sys.executable, '-m', 'ouroboros.cli', 'server',
                                       '--host', '127.0.0.1', '--port', '8765'], stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_until(lambda: read('/api/health')[0] == 200 if server.poll() is None else False, seconds=170)
                status, _ = read('/api/onboarding')
                record(onboarding_http=status)
                assert status == 204, 'documented readiness refused the supplied profile'
                bound = asyncio.run(scenario(message, large))
            finally:
                if server.poll() is None:
                    server.terminate()  # exact child only; outer unit confirmation stops remaining descendants
                    try:
                        server.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=1)
    except Exception as error:
        # Never upload the raw browser, model log, environment, DB, cookie or request body.
        record(status='partial', stopped_at=STAGE, failure_type=type(error).__name__)
        return 1
    return 0 if bound else 1


def collect_finished_probe(root):
    """Read-only recomposition apart from replacing the safe summary after the unit exits."""
    global ROOT
    ROOT = root
    path = root / 'facts.json'
    if not path.is_file():
        return 0
    FACTS.update(json.loads(path.read_bytes()))
    vendor = observe_vendor_log(root, FACTS)
    record(vendor_observation=vendor, chat_last_proven=chat_last_evidence({**FACTS, 'vendor_observation': vendor}))
    if FACTS.get('status') == 'observed' and not chat_complete(FACTS):
        record(status='partial', chat_status='partial', unit_final_evidence_changed=True)
        return 1
    return 0


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--collect':
        raise SystemExit(collect_finished_probe(Path(sys.argv[2]).resolve()))
    ROOT = Path(sys.argv[1]).resolve()
    CUTOFF = int(sys.argv[2])
    DATA = ROOT / 'data'
    MODEL = ROOT / 'Qwen3.5-0.8B-Q4_K_M.gguf'
    raise SystemExit(main())
