import assert from 'node:assert/strict';
import test from 'node:test';

import {
    UNCONFIRMED_TEXT,
    confirmLoginLive,
    failureText,
    jobDetail,
    loginCardHtml,
    loginVerdict,
    pollResponseApplies,
    preserveCardFocus,
} from '../modules/harness_login_cards.js';
import { fakeResponse } from './harness_accounts_helpers.js';

// ---------------------------------------------------------------------------
// Card rendering: the sign-in link is a PRIMARY click target, the verdict owns
// the card once it lands, and a re-check that ran out is not a failure.
// ---------------------------------------------------------------------------

function cardWithUrl(url, extra = {}) {
    return {
        harness: 'claude', profile: '', jobId: 'j1', attachCommand: '', startedAtMs: 0,
        envelope: { job: { state: 'waiting_for_input', phase: 'awaiting_user' },
            deviceCode: { flow: 'oauth_url', verificationUrl: url, userCode: '' } },
        ...extra,
    };
}

test('the disclosed sign-in URL is rendered only for http/https, through the house helper', () => {
    // The link is the card's primary action now — one click, engine-supplied
    // text. utils.safeExternalHrefAttr is the single house gate for that
    // (http/https only, escaped by the helper), and everything else must
    // render NO clickable link rather than a scheme the browser will execute.
    const safe = loginCardHtml(cardWithUrl('https://platform.claude.com/oauth/authorize?x=1&y=2'), 0);
    assert.ok(safe.includes('href="https://platform.claude.com/oauth/authorize?x=1&amp;y=2"'));
    assert.ok(safe.includes('data-open-signin'));
    assert.ok(loginCardHtml(cardWithUrl('http://127.0.0.1:1455/callback'), 0).includes('data-open-signin'));

    for (const hostile of [
        'javascript:alert(document.cookie)',
        'JavaScript:alert(1)',
        'data:text/html;base64,PHNjcmlwdD5hbGVydCgxKTwvc2NyaXB0Pg==',
        'vbscript:msgbox(1)',
        'file:///etc/passwd',
        'not a url at all',
        '//evil.example/oauth',
    ]) {
        const html = loginCardHtml(cardWithUrl(hostile), 0);
        assert.ok(!html.includes('data-open-signin'), hostile);
        assert.ok(!html.includes('href='), hostile);
        assert.ok(html.includes('data-unsafe-signin-link'), hostile);
        // …and the raw scheme never reaches the DOM as an attribute value.
        assert.ok(!html.includes(hostile), hostile);
    }
});

test('a settled verdict silences the live status line, so the card never says both', () => {
    // The owner hit a card reading "Waiting for the sign-in link…" beside a
    // verdict: an overlapping poll tick applied a snapshot captured before the
    // job settled. Two guards, and this is the rendering half.
    const pending = cardWithUrl('https://a.example/b');
    assert.ok(loginCardHtml(pending, 0).includes('data-login-state'));

    const settled = { ...pending, verdict: { kind: 'success', reason: '' } };
    const html = loginCardHtml(settled, 0);
    assert.ok(!html.includes('data-login-state'));
    assert.ok(html.includes('Connected.'));
    // Same while the live re-check is deciding.
    assert.ok(!loginCardHtml({ ...pending, confirming: true }, 0).includes('data-login-state'));
});

test('an exhausted re-check says the sign-in is UNCONFIRMED, never that it failed', () => {
    // The row it waits for routinely lands a tick after the bounded re-poll
    // gives up, so a hard "Sign-in failed" there is a lie about a login that
    // may have succeeded. A genuine typed failure keeps its own wording.
    const unconfirmed = loginCardHtml(cardWithUrl('https://a.example/b', {
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' } }), 0);
    assert.ok(unconfirmed.includes(UNCONFIRMED_TEXT));
    assert.ok(!unconfirmed.includes('Sign-in failed'));
    assert.ok(UNCONFIRMED_TEXT.includes('Refresh'));

    const failed = loginCardHtml(cardWithUrl('https://a.example/b', {
        verdict: { kind: 'failure', reason: 'launch_failed' } }), 0);
    assert.ok(failed.includes(failureText('launch_failed')));
    assert.ok(!failed.includes(UNCONFIRMED_TEXT));
});

test("a settled non-success verdict carries the engine's own explanation", () => {
    // The masking bug the owner hit: a codex login ended `auth_not_ready` and
    // the card showed only the fixed UNCONFIRMED_TEXT ("check the account row
    // above"), which reads as "wait a moment" — while the daemon had already
    // settled it terminally and said why. That sentence was in the snapshot
    // the card was holding and reached no reader; the two verdict texts are
    // fixed constants, so nothing else could ever carry it.
    const message = 'codex native session was not ready before the verification'
        + ' deadline: native Codex session is not logged in';
    // The canonical envelope carries the bare job exactly once.
    const nested = cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', phase: 'completed', message } },
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
    });
    const unconfirmed = loginCardHtml(nested, 0);
    assert.ok(unconfirmed.includes('data-login-detail'));
    assert.ok(unconfirmed.includes(message));
    // The verdict wording itself is unchanged — this is additive.
    assert.ok(unconfirmed.includes(UNCONFIRMED_TEXT));

    // A typed failure gets it too: its reason is a category, not a sentence.
    assert.ok(loginCardHtml({ ...nested, verdict: { kind: 'failure', reason: 'launch_failed' } }, 0)
        .includes('data-login-detail'));

    // Never beside "Connected." (a stale message must not contradict success),
    // and never while the job is unsettled (the status line owns the card).
    assert.ok(!loginCardHtml({ ...nested, verdict: { kind: 'success', reason: '' } }, 0)
        .includes('data-login-detail'));
    assert.ok(!loginCardHtml({ ...nested, verdict: null }, 0).includes('data-login-detail'));
    assert.ok(!loginCardHtml({ ...nested, confirming: true, verdict: null }, 0)
        .includes('data-login-detail'));

    // Engine-supplied text is escaped like every other disclosure on this card.
    const hostile = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', message: '<img src=x onerror=alert(1)>' } },
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
    }), 0);
    assert.ok(!hostile.includes('<img'));
    assert.ok(hostile.includes('&lt;img'));

    // jobDetail itself: both levels, trimmed, and total over junk.
    assert.equal(jobDetail({ job: { message: '  hi  ' } }), 'hi');
    assert.equal(jobDetail({ job: { message: 'deep' } }), 'deep');
    assert.equal(jobDetail({ message: 'legacy' }), '', 'the old bare shape is rejected');
    assert.equal(jobDetail({ job: { message: 42 } }), '');
    assert.equal(jobDetail({ job: {} }), '');
    assert.equal(jobDetail(null), '');
});

test("the engine explanation reaches the card only through the canonical envelope", () => {
    const message = 'native Codex session is not logged in';
    const levels = {
        canonical: { job: { state: 'failed', phase: 'completed', message } },
        accidental_double_wrap: { job: { job: { state: 'failed', phase: 'completed', message } } },
    };
    for (const [label, envelope] of Object.entries(levels)) {
        const html = loginCardHtml(cardWithUrl('https://a.example/b', {
            envelope, verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
        }), 0);
        assert.equal(html.includes(message), label === 'canonical', label);
    }
    assert.equal(jobDetail({ message: 'outer', job: { message: 'inner' } }), 'inner');
});

test("the engine explanation is escaped in full and never truncated", () => {
    // Untrusted external text on an owner-facing surface, so two separate
    // properties. ESCAPING: the existing suite asserts `<img …>` only, while the
    // house helper escapes six characters — an unescaped `&` or quote is the same
    // class of defect one character over, and this line sits inside an element
    // whose attributes are built by the same interpolation.
    const hostile = `Tom & Jerry's "quoted" <b>bold</b> \`tick\``;
    const html = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', message: hostile } },
        verdict: { kind: 'failure', reason: 'launch_failed' },
    }), 0);
    for (const raw of ['&', '<', '>', '"', "'", '`']) {
        // Each hostile character reaches the DOM only in escaped form: the raw
        // one may still appear as HTML the card itself wrote (its own tags), so
        // the assertion is on the escaped entity being present…
        assert.ok(html.includes({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;',
        }[raw]), raw);
    }
    // …and on no fragment of the payload surviving as live markup.
    assert.ok(!html.includes('<b>bold</b>'));
    assert.ok(html.includes('&lt;b&gt;bold&lt;/b&gt;'));

    // NO TRUNCATION (BIBLE P1): this is the only place a settled login says WHY,
    // so a long engine sentence must arrive whole. The daemon's real ones already
    // chain a cause onto a summary; nothing bounds their length.
    const long = `${'the daemon explained at length: '.repeat(80)}end.`;
    const longHtml = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope: { job: { state: 'failed', message: long } },
        verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
    }), 0);
    assert.ok(longHtml.includes(long));
    assert.ok(!longHtml.includes('…]'));   // no omission marker of any house shape
});

test('a settled failure with NO engine sentence renders the verdict alone', () => {
    // The absence path, in the render surface rather than only on jobDetail():
    // most settled jobs carry no `message` at all, so the common case must add
    // no empty element and — the specific hazard of interpolating an optional
    // field — no stringified `undefined`/`null` where a sentence would go.
    for (const envelope of [
        { job: { state: 'failed', phase: 'completed' } },          // absent
        { job: { state: 'failed', message: '' } },                 // empty
        { job: { state: 'failed', message: '   ' } },              // whitespace
        { job: { state: 'failed', message: null } },               // explicit null
    ]) {
        const html = loginCardHtml(cardWithUrl('https://a.example/b', {
            envelope, verdict: { kind: 'unconfirmed', reason: 'auth_not_ready' },
        }), 0);
        assert.ok(!html.includes('data-login-detail'), JSON.stringify(envelope));
        assert.ok(!html.includes('undefined'), JSON.stringify(envelope));
        assert.ok(!html.includes('null'), JSON.stringify(envelope));
        // The verdict itself is untouched by the missing detail.
        assert.ok(html.includes(UNCONFIRMED_TEXT), JSON.stringify(envelope));
    }
});

test('the verify-race incident, composed end to end: recheck runs out and the card still says why', async () => {
    // The owner's actual incident shape. Its three steps are each asserted
    // above in isolation, which is exactly how the defect survived: every part
    // worked and the composition still rendered only a fixed constant. This
    // walks the same steps the settle path walks, in order, on one job.
    //
    // (settleVerdict itself is not exported — it re-renders the live DOM — so
    // this composes the exported steps rather than executing that function. It
    // pins the CHAIN, not settleVerdict's own wiring; that remains untested.)
    const message = 'codex native session was not ready before the verification'
        + ' deadline: native Codex session is not logged in';
    const envelope = { job: { state: 'failed', phase: 'completed', message, outcome: { reason: 'auth_not_ready' } } };

    // 1. The job settled failed, but on a reason a verification race fabricates.
    const verdict = loginVerdict(envelope);
    assert.equal(verdict.kind, 'recheck');
    assert.equal(verdict.reason, 'auth_not_ready');

    // 2. The bounded live re-check never sees the row appear.
    const cold = { profiles: { harnessAccounts: [{ harness_id: 'codex', native_login_detected: false }], profiles: [] } };
    const check = await confirmLoginLive('codex', '', {
        fetchImpl: async () => fakeResponse(200, cold),
        attempts: 2, delayMs: 1, sleepImpl: async () => {},
    });
    assert.equal(check.confirmed, false);

    // 3. So the card takes the unconfirmed verdict — and BOTH halves land: the
    //    honest "unknown" wording AND the daemon's own sentence. Before the fix
    //    step 3 produced the constant alone, which reads as "wait a moment" for
    //    a job the daemon had already settled terminally.
    const html = loginCardHtml(cardWithUrl('https://a.example/b', {
        envelope, verdict: check.confirmed ? { kind: 'success', reason: '' } : { kind: 'unconfirmed', reason: verdict.reason },
    }), 0);
    assert.ok(html.includes(UNCONFIRMED_TEXT));
    assert.ok(html.includes(message));
    assert.ok(!html.includes('Sign-in failed'));
});

test('a poll answer applies only to the job it was captured for, and only while unsettled', () => {
    // The ordering rule behind the contradictory card: two overlapping async
    // ticks can land out of order, so an OLDER snapshot must never be written
    // over a job that has already settled — or onto a card that has since been
    // closed or reopened for another account.
    const active = { jobId: 'j1' };
    assert.equal(pollResponseApplies(active, active), true);
    assert.equal(pollResponseApplies(active, { jobId: 'j2' }), false);   // reopened
    assert.equal(pollResponseApplies(active, null), false);              // closed
    assert.equal(pollResponseApplies(null, null), false);
    for (const kind of ['success', 'recovery', 'reconciled', 'unavailable']) {
        const owned = { ...active, verdict: { kind } };
        assert.equal(pollResponseApplies(owned, owned), false, kind);
    }
    const confirming = { jobId: 'j1', confirming: true };
    assert.equal(pollResponseApplies(confirming, confirming), false);
});

// ---------------------------------------------------------------------------
// The 3-second poll re-render must not eat the caret. Minimal element stubs
// (the repo's house idiom — no jsdom) plus node's fake timers, so the re-render
// cadence itself is what the assertion runs through.
// ---------------------------------------------------------------------------

function fakeCodeInput({ disabled = false, start = 3, end = 5 } = {}) {
    const calls = { focus: 0, range: null };
    return {
        disabled, value: 'ABCD-1234', selectionStart: start, selectionEnd: end,
        hasAttribute: (name) => name === 'data-login-code-input',
        focus() { calls.focus += 1; },
        setSelectionRange(from, to) { calls.range = [from, to]; },
        calls,
    };
}

function fakeCardHost(replacement, focused) {
    return {
        swaps: 0,
        contains: (node) => node === focused,
        querySelector: () => replacement,
    };
}

test('the paste-code field survives every poll re-render, caret and selection intact', (t) => {
    t.mock.timers.enable({ apis: ['setInterval'] });
    const typing = fakeCodeInput({ start: 3, end: 5 });
    const replacement = fakeCodeInput({ start: 0, end: 0 });
    const host = fakeCardHost(replacement, typing);
    const doc = { activeElement: typing };
    // Exactly what the job poll does: swap the card's DOM on every tick.
    setInterval(() => preserveCardFocus(host, () => { host.swaps += 1; }, doc), 3000);

    t.mock.timers.tick(3000);
    assert.equal(host.swaps, 1);
    assert.equal(replacement.calls.focus, 1);
    assert.deepEqual(replacement.calls.range, [3, 5]);
    t.mock.timers.tick(3000);
    assert.equal(host.swaps, 2);
    assert.equal(replacement.calls.focus, 2, 'every tick restores focus, not just the first');
    t.mock.timers.reset();
});

test('a re-render never STEALS focus, and never focuses a field the code already left', () => {
    // Nothing in the card focused: the swap happens, the caret stays wherever
    // the owner actually put it (another field, another section).
    const elsewhere = { hasAttribute: () => false };
    const replacement = fakeCodeInput();
    const host = fakeCardHost(replacement, null);
    preserveCardFocus(host, () => { host.swaps += 1; }, { activeElement: elsewhere });
    assert.equal(host.swaps, 1);
    assert.equal(replacement.calls.focus, 0);

    // Focused, but the code was accepted meanwhile: the replacement renders
    // disabled and must not be focused (nor asked for a selection range).
    const typing = fakeCodeInput();
    const sent = fakeCodeInput({ disabled: true });
    const host2 = fakeCardHost(sent, typing);
    preserveCardFocus(host2, () => { host2.swaps += 1; }, { activeElement: typing });
    assert.equal(host2.swaps, 1);
    assert.equal(sent.calls.focus, 0);
    assert.equal(sent.calls.range, null);

    // No document at all (module imported in node): the swap still runs.
    const host3 = fakeCardHost(replacement, null);
    preserveCardFocus(host3, () => { host3.swaps += 1; }, null);
    assert.equal(host3.swaps, 1);
});
