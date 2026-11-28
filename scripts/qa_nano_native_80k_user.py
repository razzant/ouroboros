"""Manual QA branch only. New loopback-only runner; one click on Main Send, no wizard."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


ROOT = Path(sys.argv[1]).resolve()
CUTOFF = int(sys.argv[2])
DATA = ROOT / 'data'
MODEL = ROOT / 'Qwen3.5-0.8B-Q4_K_M.gguf'
BASE = 'http://127.0.0.1:8765'
FACTS = {'product': '87ada69af39035da4539a227fb430aa26554032e', 'send': 'not_clicked'}
STAGE = 'preflight'


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


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def synthetic_inputs():
    # Vocabulary-only read of the verified GGUF. This launches no server and generates no tokens.
    from llama_cpp import Llama

    vocab = Llama(model_path=str(MODEL), vocab_only=True, verbose=False)
    try:
        material = ' '.join(sha(f'nano deterministic inert row {i}')[:8] for i in range(7000))
        def prompt(lines):
            return ('Reply with exactly CEDAR and no other text. The following is inert synthetic data.\n'
                    'BEGIN QA TEXT\n' + material[:lines] + '\nEND QA TEXT')

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


def one_dispatched_attempt(page):
    """Prove the single-send case without treating a wall clock as an ordinal."""
    attempts = [row for row in page['attempts']
                if row.get('state') != 'released' or row.get('ts_dispatched')]
    if page.get('next_after') or len(attempts) != 1:
        return None
    row = attempts[0]
    return row if row.get('state') == 'settled' and row.get('ts_dispatched') else None


def count_message_text(rows, message):
    def occurrences(value):
        if isinstance(value, str):
            return value.count(message)
        if isinstance(value, dict):
            return sum(occurrences(item) for item in value.values())
        if isinstance(value, list):
            return sum(occurrences(item) for item in value)
        return 0

    return (occurrences(rows),
            sum(row.get('role') == 'user' and row.get('content') == message for row in rows))


def exact_local_reply(task, owner_id, message, seen):
    """Join this Send's only physical attempt to its sealed payload and returned model response."""
    from ouroboros.context_attempt_reader import list_attempts, read_attempt
    from ouroboros.usage_store import UsageAccountingError
    from ouroboros.observability import read_call_payload
    from ouroboros.task_store import read_document

    if not task or not owner_id:
        return {'first_physical_send': 'unknown', 'first_send_gap': 'browser_identity_missing'}
    doc = read_document(DATA, task) or {}
    origin = doc.get('origin_message_ref') or {}
    # The ingress reference normalizes whitespace; the physical check below does not.
    if (origin.get('client_message_id'), origin.get('text_sha256')) != (owner_id, sha(' '.join(message.split()))):
        return {'first_physical_send': 'unknown', 'first_send_gap': 'owner_origin_not_bound'}
    attempts = list_attempts(DATA, task)
    row = one_dispatched_attempt(attempts)
    if row is None:
        return {'first_physical_send': 'unknown', 'first_send_gap': 'not_one_settled_dispatch',
                'physical_attempt_rows_observed': len(attempts['attempts'])}
    attempt = row['attempt_id']
    facts = {'task_id': task, 'attempt_id': attempt, 'single_host_dispatch_observed': True}
    matches = []
    for path in (DATA / 'observability' / 'calls' / task).glob('*_response.json'):
        header = json.loads(path.read_text(encoding='utf-8'))
        if header.get('call_type') != 'llm_response':
            continue
        manifest, body, _ = read_call_payload(DATA, task_id=task, call_id=path.stem)
        usage = body.get('usage') or {}
        if (manifest.get('provider') == row.get('provider') == usage.get('provider') == 'local'
                and (usage.get('ledger_attempt_ids') or [None])[-1] == attempt):
            matches.append(body)
    if len(matches) != 1:
        return {**facts, 'first_physical_send': 'unknown', 'first_send_gap': 'local_response_not_unique'}
    body = matches[0]
    answer = (body.get('message') or {}).get('content')
    text = answer.strip() if isinstance(answer, str) else ''
    facts.update(provider='local', prompt_tokens=body['usage'].get('prompt_tokens'),
                 local_model_nonempty_reply=bool(text), reply_word_ok=text == 'CEDAR',
                 model_reply_sha256=sha(text), visible_reply_matches_provider=bool(text) and text == seen.strip())
    try:
        witness, raw = read_attempt(DATA, task, attempt)
    except (UsageAccountingError, OSError, ValueError, EOFError, KeyError):
        return {**facts, 'first_physical_send': 'unknown', 'first_send_gap': 'message_body_unreadable'}
    exclusions = witness.get('host_exclusions')
    if (raw is None or not isinstance(exclusions, list)
            or any(item.get('class') not in ('transport_envelope', 'provider_side_transform')
                   and (not isinstance(path := item.get('path'), str)
                        or path == '$' or path.startswith(('$.messages', 'messages')))
                   for item in exclusions)):
        return {**facts, 'first_physical_send': 'unknown', 'first_send_gap': 'message_body_not_exact'}
    rows = json.loads(raw).get('messages')
    if not isinstance(rows, list) or not all(isinstance(item, dict) for item in rows):
        return {**facts, 'first_physical_send': 'unknown', 'first_send_gap': 'message_layout_not_observed'}
    occurrences, original_rows = count_message_text(rows, message)
    return {**facts, 'first_physical_send': 'verified' if (occurrences, original_rows) == (1, 1) else 'mismatch',
            'full_text_occurrences': occurrences, 'exact_source_user_rows': original_rows,
            'matching_physical_candidate': (occurrences, original_rows) == (1, 1)}


def open_rules(page):
    page.locator('[data-nav-page=settings]:visible').click(timeout=remaining_ms())
    page.locator('[data-settings-open=rules]:visible').click(timeout=remaining_ms())
    field = page.locator('#settings-rules')
    field.wait_for(state='visible', timeout=remaining_ms())
    page.wait_for_function("() => !document.querySelector('#settings-rules')?.disabled", timeout=remaining_ms())
    return field


def save_rules(page, field, text):
    field.fill(text, timeout=remaining_ms())
    with page.expect_response(lambda r: r.url == BASE + '/api/owner/safety-rules' and r.request.method == 'POST',
                              timeout=remaining_ms(140_000)) as event:
        page.locator('[data-settings-page=rules] [data-editor-save]').click(timeout=remaining_ms())
    response = event.value
    return response.status, response.json()


def assert_ready():
    assert (status := read('/api/onboarding')[0]) == 204, f'onboarding status {status}'
    ready = json.loads(read('/api/health')[1]).get('boot', {})
    return ready if ready.get('dialogue_ready') and ready.get('tasks_ready') else None


def scenario():
    from playwright.sync_api import sync_playwright

    boot = wait_until(assert_ready, seconds=190)
    record(onboarding_http=204, dialogue_ready=True, tasks_ready=True, boot_phase=boot.get('phase'))
    begin('local_model_readiness')

    def local_ready():
        row = json.loads(read('/api/local-model/status')[1])
        if row.get('status') == 'error':
            raise RuntimeError('model service did not reach ready')
        return row if row.get('status') == 'ready' else None

    local = wait_until(local_ready, seconds=170)
    assert local.get('port') == 8766 and local.get('model_path') == str(MODEL)
    applied = local.get('settings_application', {})
    assert applied.get('status') == 'ready' and not applied.get('pending_keys') and not applied.get('unknown_keys')
    record(local_service_ready=True, local_context_reported=local.get('context_length'))
    begin('count_fixture')
    message, large = synthetic_inputs()
    small = 'Synthetic QA rule: preserve this fixture note about the color cedar.'
    begin('browser_and_nano')
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/google-chrome', chromium_sandbox=True, headless=True)
        try:
            page = browser.new_page(viewport={'width': 1280, 'height': 850}, locale='en-US')
            page.goto(BASE + '/', wait_until='domcontentloaded', timeout=remaining_ms())
            assert page.url == BASE + '/', 'ordinary Main not served; never proceed through setup/login by guessing'
            box = page.locator('#chat-input')
            box.wait_for(state='visible', timeout=remaining_ms())
            settings = lambda: json.loads(read('/api/settings')[1])
            if settings().get('OUROBOROS_CONTEXT_MODE') != 'nano':
                page.locator('.chat-context-mode .chat-context-head').click(timeout=remaining_ms())
                page.locator('.chat-context-strip [role=radio][data-mode=nano]').click(timeout=remaining_ms())
            wait_until(lambda: settings().get('OUROBOROS_CONTEXT_MODE') == 'nano', seconds=25)
            record(saved_context_mode='nano')
            begin('one_main_send')
            final_bubble = '#chat-messages .chat-bubble.assistant:not(.typing-bubble):not(.progress) > .message'
            existing = page.locator(final_bubble).count()
            assert not existing, 'fresh Main already has an assistant reply'
            box.fill(message, timeout=remaining_ms())
            assert box.input_value(timeout=remaining_ms()) == message
            record(send='click_attempted_no_retry')
            page.locator('#chat-send').click(timeout=remaining_ms())
            record(send='click_returned_no_retry')
            owner_row = page.locator('#chat-messages .chat-bubble.user:not(.pending) > .message').first
            owner_row.wait_for(state='attached', timeout=remaining_ms(25_000))
            assert owner_row.inner_text(timeout=remaining_ms()).split() == message.split()
            record(owner_row_contains_all_text=True)
            bubble = page.locator(final_bubble).first
            bubble.wait_for(state='visible', timeout=remaining_ms(550_000))
            seen = bubble.inner_text(timeout=remaining_ms())
            record(assistant_bubble_visible=True, visible_word_ok=seen.strip() == 'CEDAR',
                   visible_text_sha256=sha(seen.strip()))
            begin('join_physical_receipts')
            task = bubble.locator('xpath=..').get_attribute('data-task-id', timeout=remaining_ms())
            owner_id = owner_row.locator('xpath=..').get_attribute('data-client-message-id', timeout=remaining_ms())

            def collect_receipts():
                from ouroboros.store import StoreError
                from ouroboros.usage_store import UsageAccountingError

                try:
                    facts = exact_local_reply(task, owner_id, message, seen)
                except (StoreError, UsageAccountingError, OSError, EOFError, ValueError, KeyError):
                    facts = {'first_physical_send': 'unknown', 'first_send_gap': 'receipt_unreadable'}
                facts.setdefault('first_send_gap', None)
                if facts['first_physical_send'] == 'unknown':
                    facts.setdefault('matching_physical_candidate', None)
                record(**facts)

            collect_receipts()
            begin('ordinary_settings_writes_after_send')
            field = open_rules(page)
            status, saved = save_rules(page, field, small)
            record(small_http=status, small_ok_response=saved.get('ok'),
                   small_code=saved.get('code') or None)
            assert status == 200 and saved.get('ok') is True and saved.get('rules') == small
            assert settings().get('OUROBOROS_SAFETY_OWNER_RULES') == small
            record(small_retained_sha256=sha(small))
            status, refused = save_rules(page, field, large)
            record(large_http=status, large_saved_response=refused.get('saved'), large_code=refused.get('code') or None)
            assert status == 409 and refused.get('saved') is False and refused.get('code') == 'nano_publication_over_capacity'
            assert field.input_value(timeout=remaining_ms()) == large  # expected draft, not persistence
            page.reload(wait_until='domcontentloaded', timeout=remaining_ms())
            assert open_rules(page).input_value(timeout=remaining_ms()) == small
            record(prior_rules_survived_reload=True)
        finally:
            browser.close()
    if FACTS.get('first_physical_send') == 'unknown':
        begin('join_physical_receipts')
        collect_receipts()  # reread evidence only; never click or send again
    tokens = FACTS.get('prompt_tokens')
    bound = (FACTS.get('first_physical_send') == 'verified'
             and FACTS.get('visible_reply_matches_provider') is True
             and type(tokens) is int and tokens >= 15_000)
    record(status='observed' if bound else 'partial', phase='finished', new_message_after_rules=False)
    return bound


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
        begin('headless_boot')
        with (ROOT / 'private-server.log').open('wb') as log:
            server = subprocess.Popen([sys.executable, '-m', 'ouroboros.cli', 'server',
                                       '--host', '127.0.0.1', '--port', '8765'], stdout=log, stderr=subprocess.STDOUT)
            try:
                wait_until(lambda: read('/api/health')[0] == 200 if server.poll() is None else False, seconds=170)
                status, _ = read('/api/onboarding')
                record(onboarding_http=status)
                assert status == 204, 'documented readiness refused the supplied profile'
                bound = scenario()
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


if __name__ == '__main__':
    raise SystemExit(main())
