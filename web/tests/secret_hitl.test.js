// The integration-secret form: card-opened, status-first, Save or explicit Replace, unknown kept
// unknown, Close is not Cancel. Values live in the open form's controls and nowhere else.
import assert from 'node:assert/strict';
import test from 'node:test';
import { NodeStub, turn } from './chat_decision_fixture.js';
import { allowLines, createSecretRequestForms, readerLine, secretRequestOf } from '../modules/secret_hitl.js';

const CANARY = 'zq7-CANARY-Ωlj3~p0+kx/2';
const REQUEST = { plugin_name: 'telegram', fields: [{ id: 'botToken', setting_key: 'TELEGRAM_BOT_TOKEN' }] };
const QUIZ = { quizId: 'qz-s', taskId: 't-1', projectId: '', secretRequest: secretRequestOf({ secret_request: REQUEST }) };
const DECISION = 'quiz:t-1:qz-s';
const READERS = [{ name: 'telegram', enabled: true, reads: ['TELEGRAM_BOT_TOKEN'], granted: ['TELEGRAM_BOT_TOKEN'] },
    { name: 'mailer', enabled: false, reads: ['TELEGRAM_BOT_TOKEN'], granted: [] }];
const status = (configured, extra = {}) => ({ ok: true, decision_id: DECISION, state: 'open', secret_request: REQUEST,
    keys: [{ id: 'botToken', setting_key: 'TELEGRAM_BOT_TOKEN', configured }], consumers: READERS, ...extra });
const saved = (event = 'secret_filled') => ({ ok: true, saved: true, state: 'answered', decision_id: DECISION,
    duplicate: false, [event]: { pluginName: 'telegram', fields: ['botToken'] },
    secret_outcome: { event, pluginName: 'telegram', fields: ['botToken'], settings_keys: ['TELEGRAM_BOT_TOKEN'],
        consumers: READERS } });
const reply = (statusCode, body) => ({ ok: statusCode < 400, status: statusCode, json: async () => body });

function setup({ statusBody = status(false), save = () => reply(200, saved()), cancel = null, waitMs = 25 } = {}) {
    const prior = { document: globalThis.document, localStorage: globalThis.localStorage };
    globalThis.document = { createElement: (tag) => new NodeStub(tag) };
    const stored = [];
    globalThis.localStorage = { getItem: () => null, setItem: (...args) => stored.push(args), removeItem() {} };
    const host = new NodeStub('body');
    const calls = [], accepted = [], toasts = [], focus = [], reveals = [];
    const state = { statusBody };
    const forms = createSecretRequestForms({
        apiFetch: async (url, init) => {
            const body = JSON.parse(init.body);
            calls.push(body);
            if (body.action === 'status') return typeof state.statusBody === 'function' ? state.statusBody() : reply(200, state.statusBody);
            if (body.action === 'cancel') return cancel ? cancel(body) : reply(200, { ok: true, saved: false, state: 'closed',
                decision_id: DECISION, secret_cancelled: { pluginName: 'telegram', fields: ['botToken'] } });
            return save(body);
        },
        accept: (card, body) => { accepted.push(body); card.dataset.state = body.state; forms.sync(card, body.state, body); },
        showToast: (text) => toasts.push(text), host: () => host, waitMs,
        bindFocus: (panel, options) => { focus.push(options); return () => focus.push('released'); },
        bindReveal: (input, button, options) => { reveals.push({ input, button, options }); return { reset() { reveals.push('reset'); } }; },
    });
    const card = new NodeStub('div');
    card.dataset.state = 'open';
    return { forms, card, host, calls, accepted, toasts, focus, stored, reveals, state,
        restore() { globalThis.document = prior.document; globalThis.localStorage = prior.localStorage; } };
}
const q = (node, name) => node.querySelector(name);
const input = (fx) => q(fx.host, 'secret-request-input');
const ok = (fx) => q(fx.host, 'secret-request-ok');
const type = (field, value) => { field.value = value; field.listeners.get('input')?.({}); };
const everything = (node) => JSON.stringify(node, (key, value) => (key === 'parentNode' || key === 'listeners' ? undefined : value));
async function opened(fx) {
    fx.forms.mount(fx.card, QUIZ);
    fx.forms.sync(fx.card, fx.card.dataset.state, {});
    q(fx.card, 'chat-secret-open').click();
    await turn();
}

test('the card names fields and keys; the form opens only from it and reads status first', async () => {
    const fx = setup();
    try {
        fx.forms.mount(fx.card, QUIZ);
        assert.equal(fx.host.children.length, 0, 'arrival never opens a modal');
        assert.equal(q(fx.card, 'chat-secret-field-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN');
        assert.equal(q(fx.card, 'chat-secret-open').textContent, 'Enter secret…');
        await opened(fx);
        assert.deepEqual(fx.calls, [{ decision_id: DECISION, action: 'status' }]);
        const dialog = q(fx.host, 'secret-request-dialog');
        assert.equal(dialog.getAttribute('role'), 'dialog');
        assert.equal(dialog.getAttribute('aria-modal'), 'true');
        assert.equal(q(fx.host, 'secret-request-backdrop').getAttribute('data-i18n-authored'), '');
        assert.equal(q(fx.host, 'secret-request-title').textContent, 'telegram: secret needed');
        assert.equal(q(fx.host, 'secret-request-intro').textContent, 'The value is saved to Settings. It is not sent in this conversation.');
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · not set');
        assert.equal(input(fx).type, 'password');
        assert.equal(input(fx).autocomplete, 'off');
        assert.equal(fx.reveals[0].input, input(fx), 'Show/Hide is the Settings reveal over this local draft');
        assert.equal(q(fx.host, 'secret-request-readers-title').textContent, 'Skills that request this setting:');
        assert.deepEqual(fx.host.querySelectorAll('secret-request-reader').map((row) => row.textContent),
            ['telegram · enabled · asks for TELEGRAM_BOT_TOKEN · allowed to read it',
                'mailer · disabled · asks for TELEGRAM_BOT_TOKEN · not allowed yet']);
        assert.equal(ok(fx).textContent, 'Save');
        assert.equal(ok(fx).disabled, true);
        type(input(fx), CANARY);
        assert.equal(ok(fx).disabled, false);
        assert.equal(q(fx.host, 'secret-request-warning').textContent, '', 'no key yet: nothing to replace');
        assert.equal(fx.focus[0].initialFocus, input(fx));
    } finally { fx.restore(); }
});

test('Save posts the values once with what the form showed, then empties the controls and settles', async () => {
    const fx = setup();
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        ok(fx).click();  // a second press while the first is in flight sends nothing
        await turn();
        const posts = fx.calls.filter((body) => !body.action);
        assert.equal(posts.length, 1);
        assert.deepEqual(posts[0].values, { botToken: CANARY });
        assert.deepEqual(posts[0].observed, { botToken: false });
        assert.ok(posts[0].request_id);
        assert.equal(fx.host.children.length, 0, 'the form closed');
        assert.equal(fx.accepted.at(-1).state, 'answered');
        assert.match(q(fx.card, 'chat-secret-status').textContent, /^Saved to Settings: botToken\./);
        assert.match(q(fx.card, 'chat-secret-status').textContent, /did not grant access/);
        assert.equal(fx.card.querySelectorAll('chat-secret-reader').map((row) => row.textContent)[1],
            'mailer · was off · requested TELEGRAM_BOT_TOKEN · was not allowed to read it');
        assert.equal(q(fx.card, 'chat-secret-actions'), null, 'a settled card has no actions');
        assert.equal(fx.stored.length, 0);
        assert.ok(!everything(fx.host).includes(CANARY) && !everything(fx.card).includes(CANARY));
        assert.ok(!fx.toasts.join(' ').includes(CANARY));
    } finally { fx.restore(); }
});

test('a set key is Replace: other skills keep their permissions, the allowed readers are named', async () => {
    const fx = setup({ statusBody: status(true), save: () => reply(200, saved('secret_replaced')) });
    try {
        await opened(fx);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · a value is set (hidden)');
        assert.equal(ok(fx).textContent, 'Replace');
        assert.equal(q(fx.host, 'secret-request-warning').textContent, 'Replace overwrites the saved value. Other skills’ '
            + 'permissions stay the same; those already allowed to read it (telegram) may get the new value.');
        type(input(fx), CANARY);
        ok(fx).click();
        await turn();
        assert.deepEqual(fx.calls.at(-1).observed, { botToken: true });
        assert.match(q(fx.card, 'chat-secret-status').textContent, /^Replaced in Settings: botToken\..*did not grant access/);
    } finally { fx.restore(); }
});

test('a key saved elsewhere since the form opened turns Save into Replace and keeps the draft', async () => {
    let posts = 0;
    const fx = setup({ save: () => {
        posts += 1;
        return posts === 1 ? reply(409, { ok: false, saved: false, reason_code: 'secret_key_state_changed', state: 'open',
            decision_id: DECISION, keys: [{ id: 'botToken', setting_key: 'TELEGRAM_BOT_TOKEN', configured: true }],
            consumers: READERS }) : reply(200, saved('secret_replaced'));
    } });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        const field = input(fx);
        ok(fx).click();
        await turn();
        assert.equal(input(fx), field, 'the same control');
        assert.equal(field.value, CANARY, 'the draft survives the refusal');
        assert.equal(ok(fx).textContent, 'Replace');
        assert.match(q(fx.host, 'secret-request-error').textContent, /got a value in Settings after the form opened, so your entry was not saved/);
        assert.ok(q(fx.host, 'secret-request-warning').textContent.startsWith('Replace overwrites'));
        ok(fx).click();
        await turn();
        const [first, second] = fx.calls.filter((body) => !body.action);
        assert.deepEqual([first.observed, second.observed], [{ botToken: false }, { botToken: true }]);
        assert.equal(fx.host.children.length, 0);
    } finally { fx.restore(); }
});

test('a lost reply is unknown: Save stays blocked until status, which never claims nothing was saved', async () => {
    let posts = 0;
    const fx = setup({ save: () => { posts += 1; if (posts === 1) throw new TypeError('network'); return reply(200, saved('secret_replaced')); } });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await turn();
        assert.equal(ok(fx).disabled, true, 'no automatic or one-tap resubmit over an unknown outcome');
        assert.equal(q(fx.host, 'secret-request-check').hidden, false);
        assert.match(q(fx.host, 'secret-request-error').textContent, /its result is unknown/);
        assert.equal(q(fx.host, 'secret-request-key').textContent,
            'Saved in Settings as TELEGRAM_BOT_TOKEN · unknown until you check status', 'a stale read is not shown as current');
        assert.equal(fx.card.dataset.submission, 'unknown');
        const firstId = fx.calls.filter((body) => !body.action)[0].request_id;
        fx.state.statusBody = status(true, { save_unconfirmed: true });
        q(fx.host, 'secret-request-check').click();
        await turn();
        assert.match(q(fx.host, 'secret-request-error').textContent, /never confirmed; whether it stored a value is unknown/);
        assert.ok(!/nothing (is|was) saved/i.test(q(fx.host, 'secret-request-error').textContent));
        assert.equal(ok(fx).textContent, 'Replace', 'a set key cannot prove this attempt saved it; replacing is explicit');
        assert.equal(ok(fx).disabled, false);
        // A later lifecycle frame still showing the unconfirmed claim does not take the choice back.
        fx.forms.sync(fx.card, 'open', { secret_save: { state: 'writing' } });
        assert.equal(ok(fx).disabled, false);
        ok(fx).click();
        await turn();
        const again = fx.calls.filter((body) => !body.action)[1];
        assert.notEqual(again.request_id, firstId, 'saving again over an unconfirmed attempt is a new request');
    } finally { fx.restore(); }
});

test('a lost reply whose save landed settles from status', async () => {
    const fx = setup({ save: () => { throw new TypeError('network'); } });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await turn();
        fx.state.statusBody = saved();
        q(fx.host, 'secret-request-check').click();
        await turn();
        assert.equal(fx.host.children.length, 0);
        assert.equal(fx.accepted.at(-1).state, 'answered');
    } finally { fx.restore(); }
});

test('written but unconfirmed replies stay unknown, refusals that saved nothing keep the draft', async () => {
    for (const [code, saved_, unknown] of [['secret_save_unverified', true, true], ['secret_completion_unrecorded', true, true],
        ['secret_save_unconfirmed', null, true], ['settings_busy', false, false], ['secret_value_placeholder', false, false]]) {
        const fx = setup({ save: () => reply(code === 'settings_busy' ? 503 : code.startsWith('secret_value') ? 400 : 500,
            { ok: false, saved: saved_, reason_code: code, decision_id: DECISION, state: 'open' }) });
        try {
            await opened(fx);
            type(input(fx), CANARY);
            ok(fx).click();
            await turn();
            assert.equal(input(fx).value, CANARY, code);
            assert.equal(ok(fx).disabled, unknown, code);
            assert.equal(q(fx.host, 'secret-request-check').hidden, !unknown, code);
            const text = q(fx.host, 'secret-request-error').textContent;
            assert.equal(/nothing was saved/.test(text), !unknown && code !== 'secret_value_placeholder', `${code}: ${text}`);
            assert.ok(!text.includes(CANARY));
        } finally { fx.restore(); }
    }
});

test('Close and Escape only hide the form; Cancel request is the recorded cancel', async () => {
    const fx = setup();
    try {
        await opened(fx);
        type(input(fx), CANARY);
        q(fx.host, 'secret-request-cancel').click();
        assert.equal(fx.host.children.length, 0);
        assert.ok(fx.reveals.includes('reset'), 'a reveal of the draft is reset on close');
        assert.ok(!fx.calls.some((body) => body.action === 'cancel'));
        q(fx.card, 'chat-secret-open').click();
        await turn();
        assert.equal(input(fx).value, '', 'reopening starts empty');
        fx.focus.at(-1).onEscape();
        assert.equal(fx.host.children.length, 0);
        q(fx.card, 'chat-secret-cancel').click();
        await turn();
        const cancel = fx.calls.find((body) => body.action === 'cancel');
        assert.deepEqual(Object.keys(cancel).sort(), ['action', 'decision_id', 'request_id']);
        assert.equal(fx.accepted.at(-1).state, 'closed');
    } finally { fx.restore(); }
});

test('a finished task keeps the form usable and says that saving resumes nothing', async () => {
    const fx = setup({ statusBody: status(false, { state: 'expired_terminal' }) });
    try {
        fx.card.dataset.state = 'expired_terminal';
        await opened(fx);
        assert.equal(q(fx.card, 'chat-secret-open').disabled, false);
        assert.match(q(fx.host, 'secret-request-late').textContent, /has finished\. Saving still stores the key; it does not resume that work\./);
        type(input(fx), CANARY);
        assert.equal(ok(fx).disabled, false);
    } finally { fx.restore(); }
});

test('a re-rendered card keeps the open form, its draft and focus target; another window settles it', async () => {
    const fx = setup();
    try {
        await opened(fx);
        type(input(fx), CANARY);
        fx.forms.release(fx.card);
        const next = new NodeStub('div');
        next.dataset.state = 'open';
        fx.forms.mount(next, QUIZ);
        fx.forms.sync(next, 'open', {});
        assert.equal(fx.host.children.length, 1, 'the form outlives its card node');
        assert.equal(input(fx).value, CANARY);
        fx.forms.sync(next, 'answered', saved().secret_outcome && { secret_outcome: saved().secret_outcome });
        assert.equal(fx.host.children.length, 0);
        assert.equal(fx.toasts.length, 1);
        assert.ok(!fx.toasts[0].includes(CANARY));
    } finally { fx.restore(); }
});

const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done; }); return { promise, resolve }; };
const keyNowSet = () => reply(409, { ok: false, saved: false, reason_code: 'secret_key_state_changed', state: 'open',
    decision_id: DECISION, keys: [{ id: 'botToken', setting_key: 'TELEGRAM_BOT_TOKEN', configured: true }], consumers: READERS });

test('Close waits for a save in flight like Escape and a click outside; its refusal leaves a live form', async () => {
    const held = deferred();
    const fx = setup({ save: () => held.promise });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await turn();
        const close = q(fx.host, 'secret-request-close'), dismiss = q(fx.host, 'secret-request-cancel');
        assert.deepEqual([close.disabled, dismiss.disabled, q(fx.card, 'chat-secret-open').disabled], [true, true, true]);
        close.click();
        dismiss.click();
        fx.focus.at(-1).onEscape();
        assert.equal(fx.host.children.length, 1, 'the form waits for the reply that settles it');
        held.resolve(keyNowSet());
        await turn();
        assert.equal(ok(fx).textContent, 'Replace');
        assert.equal(ok(fx).disabled, false, "the reply's key facts are a fresh read: Save is live");
        assert.equal(q(fx.host, 'secret-request-check').hidden, true);
        assert.deepEqual([close.disabled, dismiss.disabled, q(fx.card, 'chat-secret-open').disabled], [false, false, false]);
        assert.equal(input(fx).value, CANARY);
    } finally { fx.restore(); }
});

test('Enter secret waits for a cancel in flight; after a failed cancel the form opens with a fresh read', async () => {
    const held = deferred();
    const fx = setup({ cancel: () => held.promise });
    try {
        fx.forms.mount(fx.card, QUIZ);
        fx.forms.sync(fx.card, 'open', {});
        q(fx.card, 'chat-secret-cancel').click();
        await turn();
        assert.equal(q(fx.card, 'chat-secret-open').disabled, true, 'no form is born while the request may be closing');
        held.resolve(reply(503, { ok: false, saved: false, reason_code: 'settings_busy' }));
        await turn();
        assert.equal(q(fx.card, 'chat-secret-open').disabled, false);
        q(fx.card, 'chat-secret-open').click();
        await turn();
        assert.deepEqual(fx.calls.map((body) => body.action), ['cancel', 'status']);
        type(input(fx), CANARY);
        assert.equal(ok(fx).disabled, false);
    } finally { fx.restore(); }
});

test('a reopened form never shows an earlier read as current when its status read fails', async () => {
    const fx = setup();
    try {
        await opened(fx);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · not set');
        q(fx.host, 'secret-request-close').click();
        fx.state.statusBody = () => { throw new TypeError('network'); };
        q(fx.card, 'chat-secret-open').click();
        await turn();
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · could not be read');
        type(input(fx), CANARY);
        assert.equal(ok(fx).disabled, true, 'no save on key facts this form did not read');
        assert.equal(q(fx.host, 'secret-request-check').hidden, false);
        fx.state.statusBody = status(true);
        q(fx.host, 'secret-request-check').click();
        await turn();
        assert.equal(ok(fx).textContent, 'Replace');
        assert.equal(ok(fx).disabled, false);
    } finally { fx.restore(); }
});

test('a saved card says its readers could not be read, never that none read the key', () => {
    const fx = setup();
    const readers = () => fx.card.querySelectorAll('chat-secret-reader').map((row) => row.textContent);
    try {
        fx.forms.mount(fx.card, QUIZ);
        const unknown = saved().secret_outcome;
        delete unknown.consumers;
        fx.forms.sync(fx.card, 'answered', { secret_outcome: { ...unknown, consumers_unknown: true } });
        assert.match(readers()[0], /requested these keys at the time could not be established/);
        fx.forms.sync(fx.card, 'answered', { secret_outcome: { ...unknown, consumers: [] } });
        assert.deepEqual(readers(), []);
    } finally { fx.restore(); }
});

test("the form's Show and Hide speak the installed language: the overlay never walks the form", async () => {
    const { applyPayload, currentPayload, flushMisses, CODE_PREFIX } = await import('../modules/i18n.js');
    const prior = currentPayload();
    const fx = setup();
    try {
        applyPayload({ language: 'fr', english: false, entries: {
            [CODE_PREFIX + 'secret.show']: { text: 'Afficher' }, [CODE_PREFIX + 'secret.hide']: { text: 'Masquer' } } });
        await opened(fx);
        assert.equal(q(fx.host, 'secret-request-backdrop').getAttribute('data-i18n-authored'), '');
        assert.deepEqual(fx.reveals[0].options?.copy, { show: 'Afficher', hide: 'Masquer' });
    } finally { applyPayload(prior); flushMisses(); fx.restore(); }
});

test('a reader line keeps the switch, the request and the granted right apart', () => {
    assert.equal(readerLine({ name: 'a', enabled: false, reads: ['K'], granted: ['K'] }), 'a · disabled · asks for K · allowed to read it');
    // Grant facts only (RV1 A11): no claim about whether the enabled skill runs, which the card cannot keep current.
    const partial = readerLine({ name: 'b', enabled: true, reads: ['FIRST', 'NEXT', 'LAST'], granted: ['FIRST'] });
    assert.equal(partial, 'b · enabled · asks for FIRST, NEXT, LAST · allowed to read FIRST, not NEXT, LAST');
    assert.doesNotMatch(partial, /can.t run|runs/);
    assert.equal(readerLine({ name: 'c', enabled: true, reads: ['K'], unsupported: true }), 'c · enabled · asks for K · this skill type receives no Settings keys');
    assert.equal(readerLine({ name: 'd', enabled: true, reads: ['K'], grant_unknown: true }), 'd · enabled · asks for K · its access could not be read');
});

// A stalled fetch or response body must release the owner, never trap them: the wait is bounded by
// `waitMs` (real timers, a few milliseconds here), nothing is resent, and the late reply still settles.
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const WAIT = 25;
const stalled = () => deferred();
const posts = (fx) => fx.calls.filter((body) => !body.action);
const bodyNever = (held) => ({ ok: true, status: 200, json: () => held.promise });
const unconfirmedText = /^The save was not confirmed; its result is unknown\./;

for (const [name, stall] of [['fetch', (held) => held.promise], ['response body', (held) => bodyNever(held)]]) {
    test(`a stalled ${name} releases Close and Check status after the wait; nothing is resent, the late reply settles`, async () => {
        const held = stalled();
        const fx = setup({ save: () => stall(held) });
        try {
            await opened(fx);
            type(input(fx), CANARY);
            ok(fx).click();
            await turn();
            const close = q(fx.host, 'secret-request-close'), dismiss = q(fx.host, 'secret-request-cancel');
            assert.deepEqual([close.disabled, dismiss.disabled, ok(fx).textContent], [true, true, 'Saving…']);
            await sleep(WAIT * 3);
            assert.match(q(fx.host, 'secret-request-error').textContent, unconfirmedText);
            assert.deepEqual([close.disabled, dismiss.disabled, ok(fx).disabled], [false, false, true]);
            assert.equal(q(fx.host, 'secret-request-check').hidden, false);
            assert.equal(fx.card.dataset.submission, 'unknown', 'the head leaves "sending"');
            assert.equal(input(fx).readOnly, false);
            ok(fx).click();                                   // a duplicate press on a live frame sends nothing
            input(fx).listeners.get('keydown')?.({ key: 'Enter', preventDefault() {} });
            await turn();
            assert.equal(posts(fx).length, 1, 'no automatic or manual resend of an unconfirmed save');
            held.resolve(name === 'fetch' ? reply(200, saved()) : saved());
            await sleep(5);
            assert.equal(fx.host.children.length, 0, 'the late success closed the live form');
            assert.equal(fx.accepted.at(-1).state, 'answered');
            assert.match(q(fx.card, 'chat-secret-status').textContent, /^Saved to Settings: botToken\./);
            assert.ok(!everything(fx.host).includes(CANARY) && !everything(fx.card).includes(CANARY));
            assert.equal(posts(fx).length, 1);
        } finally { fx.restore(); }
    });
}

test('a late success after Close still settles the card; Close during the unknown wait empties the controls', async () => {
    const held = stalled();
    const fx = setup({ save: () => held.promise });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await sleep(WAIT * 3);
        q(fx.host, 'secret-request-close').click();
        assert.equal(fx.host.children.length, 0, 'Close is available once the wait ended');
        assert.match(q(fx.card, 'chat-secret-status').textContent, /result is unknown/);
        held.resolve(reply(200, saved()));
        await sleep(5);
        assert.equal(fx.accepted.at(-1).state, 'answered');
        assert.match(q(fx.card, 'chat-secret-status').textContent, /^Saved to Settings: botToken\./);
        assert.ok(!everything(fx.card).includes(CANARY));
    } finally { fx.restore(); }
});

test('a late refusal for the live form ends the doubt; for a reopened form it changes nothing', async () => {
    const first = stalled(), second = stalled();
    const queue = [first, second];
    const fx = setup({ save: () => queue.shift().promise });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await sleep(WAIT * 3);
        first.resolve(keyNowSet());
        await sleep(5);
        assert.equal(ok(fx).textContent, 'Replace');
        assert.equal(ok(fx).disabled, false, 'the definite refusal saved nothing: Save is live on its facts');
        assert.equal(q(fx.host, 'secret-request-check').hidden, true);
        assert.equal(input(fx).value, CANARY);
        // Again, but Close and reopen before the late refusal arrives.
        ok(fx).click();
        await sleep(WAIT * 3);
        q(fx.host, 'secret-request-close').click();
        fx.state.statusBody = status(false);
        q(fx.card, 'chat-secret-open').click();
        await sleep(5);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · not set');
        second.resolve(keyNowSet());
        await sleep(5);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · not set',
            'the old save\'s facts are not the reopened form\'s read');
        assert.equal(ok(fx).textContent, 'Save');
    } finally { fx.restore(); }
});

test('a late status read of a closed form never describes the reopened one', async () => {
    const readA = stalled();
    let reads = 0;
    const fx = setup({ statusBody: () => { reads += 1; if (reads === 1) return readA.promise; throw new TypeError('network'); } });
    try {
        fx.forms.mount(fx.card, QUIZ);
        fx.forms.sync(fx.card, 'open', {});
        q(fx.card, 'chat-secret-open').click();
        await turn();
        q(fx.host, 'secret-request-close').click();            // dialog A closed while its read is outstanding
        q(fx.card, 'chat-secret-open').click();                // dialog B: its own read fails
        await sleep(5);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · could not be read');
        readA.resolve(reply(200, status(false)));
        await sleep(5);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · could not be read');
        type(input(fx), CANARY);
        assert.equal(ok(fx).disabled, true, 'A\'s read cannot enable Save in B');
        assert.equal(q(fx.host, 'secret-request-check').hidden, false);
    } finally { fx.restore(); }
});

test('an older status read never overrides a newer failed read of the same form', async () => {
    const readA = stalled();
    let reads = 0;
    const fx = setup({ statusBody: () => { reads += 1; if (reads === 1) return readA.promise; throw new TypeError('network'); } });
    try {
        fx.forms.mount(fx.card, QUIZ);
        fx.forms.sync(fx.card, 'open', {});
        q(fx.card, 'chat-secret-open').click();
        await sleep(WAIT * 3);                                 // read A stalled: released as unreadable
        assert.equal(q(fx.host, 'secret-request-check').hidden, false);
        q(fx.host, 'secret-request-check').click();            // read B fails
        await sleep(5);
        readA.resolve(reply(200, status(true)));
        await sleep(5);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · could not be read');
        type(input(fx), CANARY);
        assert.equal(ok(fx).disabled, true);
    } finally { fx.restore(); }
});

test('a stalled status read releases Check status and Close; its own late reply is then taken', async () => {
    const readA = stalled();
    let reads = 0;
    const fx = setup({ statusBody: () => { reads += 1; return reads === 1 ? readA.promise : reply(200, status(false)); } });
    try {
        fx.forms.mount(fx.card, QUIZ);
        fx.forms.sync(fx.card, 'open', {});
        q(fx.card, 'chat-secret-open').click();
        await turn();
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · checking Settings…');
        assert.equal(q(fx.host, 'secret-request-check').hidden, true);
        await sleep(WAIT * 3);
        assert.equal(q(fx.host, 'secret-request-key').textContent, 'Saved in Settings as TELEGRAM_BOT_TOKEN · could not be read');
        assert.equal(q(fx.host, 'secret-request-check').hidden, false);
        readA.resolve(reply(200, status(true)));
        await sleep(5);
        assert.equal(ok(fx).textContent, 'Replace', 'the current read\'s late reply is a normal read');
        assert.equal(q(fx.host, 'secret-request-check').hidden, true);
        assert.equal(reads, 1, 'no polling');
    } finally { fx.restore(); }
});

test('a stalled cancel releases the card after the wait; its late reply still settles, with no repeat', async () => {
    const held = stalled();
    const fx = setup({ cancel: () => held.promise });
    try {
        fx.forms.mount(fx.card, QUIZ);
        fx.forms.sync(fx.card, 'open', {});
        q(fx.card, 'chat-secret-cancel').click();
        await turn();
        assert.equal(q(fx.card, 'chat-secret-open').disabled, true);
        await sleep(WAIT * 3);
        assert.deepEqual([q(fx.card, 'chat-secret-open').disabled, q(fx.card, 'chat-secret-cancel').disabled], [false, false]);
        assert.equal(fx.toasts.filter((text) => /cancellation was not confirmed/.test(text)).length, 1);
        held.resolve(reply(200, { ok: true, saved: false, state: 'closed', decision_id: DECISION,
            secret_cancelled: { pluginName: 'telegram', fields: ['botToken'] } }));
        await sleep(5);
        assert.equal(fx.accepted.at(-1).state, 'closed');
        assert.equal(fx.calls.filter((body) => body.action === 'cancel').length, 1);
        assert.equal(fx.toasts.filter((text) => /cancellation was not confirmed/.test(text)).length, 1);
    } finally { fx.restore(); }
});

test('Close and dispose clear every wait they own', async () => {
    const live = new Set();
    const real = { setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout };
    globalThis.setTimeout = (fn, ms, ...rest) => { const id = real.setTimeout(fn, ms, ...rest); if (ms >= 10000) live.add(id); return id; };
    globalThis.clearTimeout = (id) => { live.delete(id); return real.clearTimeout(id); };
    const held = stalled();
    const fx = setup({ waitMs: 60000, statusBody: () => held.promise });
    try {
        fx.forms.mount(fx.card, QUIZ);
        fx.forms.sync(fx.card, 'open', {});
        q(fx.card, 'chat-secret-open').click();
        await turn();
        assert.equal(live.size, 1, 'the open form waits on its status read');
        q(fx.host, 'secret-request-close').click();
        assert.equal(live.size, 0, 'Close clears the form\'s wait');
        fx.forms.dispose();
        assert.equal(live.size, 0);
    } finally {
        globalThis.setTimeout = real.setTimeout; globalThis.clearTimeout = real.clearTimeout; fx.restore();
    }
});

test('dispose clears a save and a cancel in flight; their late replies change nothing', async () => {
    const live = new Set();
    const real = { setTimeout: globalThis.setTimeout, clearTimeout: globalThis.clearTimeout };
    globalThis.setTimeout = (fn, ms, ...rest) => { const id = real.setTimeout(fn, ms, ...rest); if (ms >= 10000) live.add(id); return id; };
    globalThis.clearTimeout = (id) => { live.delete(id); return real.clearTimeout(id); };
    const held = stalled();
    const fx = setup({ waitMs: 60000, save: () => held.promise });
    try {
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await turn();
        assert.equal(live.size, 1, 'the save in flight waits');
        fx.forms.dispose();
        assert.equal(live.size, 0);
        const before = fx.accepted.length;
        held.resolve(reply(200, saved()));
        await sleep(5);
        assert.equal(fx.accepted.length, before, 'a disposed surface settles nothing');
        assert.equal(fx.host.children.length, 0);
    } finally {
        globalThis.setTimeout = real.setTimeout; globalThis.clearTimeout = real.clearTimeout; fx.restore();
    }
});

// A remote lifecycle closes the form: focus follows only if the form held it, and goes to the question.
const withQuestion = (card) => {
    const question = new NodeStub('div');
    question.className = 'chat-quiz-question';
    card.append(question);
    return question;
};
const remoteAnswer = (fx, card) => fx.forms.sync(card, 'answered', { secret_outcome: saved().secret_outcome });

test('a remote settlement moves focus from the closed form to the request\'s question', async () => {
    const fx = setup();
    try {
        const question = withQuestion(fx.card);
        await opened(fx);
        input(fx).focus();
        remoteAnswer(fx, fx.card);
        assert.equal(fx.host.children.length, 0);
        assert.equal(globalThis.document.activeElement, question, 'not the body');
        assert.equal(fx.toasts.length, 1);
    } finally { fx.restore(); }
});

test('a remote settlement never takes focus the form did not hold', async () => {
    const fx = setup();
    try {
        withQuestion(fx.card);
        await opened(fx);
        const elsewhere = new NodeStub('textarea');
        elsewhere.focus();                       // the owner is typing in the composer
        remoteAnswer(fx, fx.card);
        assert.equal(fx.host.children.length, 0);
        assert.equal(globalThis.document.activeElement, elsewhere);
    } finally { fx.restore(); }
});

test('with a Main mirror and a Project card the settled focus lands on the connected question', async () => {
    const fx = setup();
    try {
        const mirror = new NodeStub('div');
        mirror.dataset.state = 'open';
        const mirrorQuestion = withQuestion(mirror);
        fx.forms.mount(mirror, QUIZ);
        fx.forms.sync(mirror, 'open', {});
        const projectQuestion = withQuestion(fx.card);
        await opened(fx);
        mirrorQuestion.remove();                 // the Main copy retired meanwhile
        input(fx).focus();
        remoteAnswer(fx, fx.card);
        assert.equal(globalThis.document.activeElement, projectQuestion);
    } finally { fx.restore(); }
});

test('dispose closes an open form without moving focus', async () => {
    const fx = setup();
    try {
        withQuestion(fx.card);
        await opened(fx);
        input(fx).focus();
        const held = globalThis.document.activeElement;
        fx.forms.dispose();
        assert.equal(fx.host.children.length, 0);
        assert.equal(globalThis.document.activeElement, held, 'a disposed surface takes no focus');
    } finally { fx.restore(); }
});

// The timed-out Save still settles, but cannot take focus after the owner dismissed its form.
test('a late save after dismissal keeps the composer focus', async () => {
    const held = stalled();
    const fx = setup({ save: () => held.promise });
    try {
        withQuestion(fx.card);
        await opened(fx);
        type(input(fx), CANARY);
        ok(fx).click();
        await sleep(WAIT * 3);
        q(fx.host, 'secret-request-close').click();
        const composer = new NodeStub('textarea');
        composer.focus();
        held.resolve(reply(200, saved()));
        await sleep(5);
        assert.equal(fx.accepted.at(-1).state, 'answered');
        assert.equal(globalThis.document.activeElement, composer);
    } finally { fx.restore(); }
});


// ---- Save and allow: one explicit extended action beside Save, for the request's own skill only.
const SKILL = { name: 'telegram', available: true, content_hash: 'sha256:files-shown', installation: 'inst-shown',
    admitted: true, supported: true, granted: [], waiting: [], switch: 'unset' };
const OTHERS = [{ name: 'telegram', enabled: false, reads: ['TELEGRAM_BOT_TOKEN'], granted: [] },
    { name: 'mailer', enabled: true, reads: ['TELEGRAM_BOT_TOKEN'], granted: ['TELEGRAM_BOT_TOKEN'] }];
const allowButton = (fx) => q(fx.host, 'secret-request-allow');
const note = (fx) => q(fx.host, 'secret-request-allow-note').textContent;
const outcomeLines = (fx) => fx.card.querySelectorAll('chat-secret-outcome-line').map((row) => row.textContent);
const allowed = (allow) => ({ ...saved(), secret_outcome: { ...saved().secret_outcome, allow: {
    skill: 'telegram', fields: ['botToken'], keys: ['TELEGRAM_BOT_TOKEN'], content_hash: SKILL.content_hash,
    enable: true, at: 't', ...allow } } });

test('a status naming the skill offers Save and allow beside Save; Enter and Save never send it', async () => {
    const fx = setup({ statusBody: status(false, { skill: { ...SKILL, switch: 'on' }, consumers: OTHERS }) });
    try {
        await opened(fx);
        assert.equal(allowButton(fx).hidden, false);
        assert.equal(q(fx.host, 'secret-request-allow-label').textContent, 'Save and allow telegram');
        assert.equal(ok(fx).dataset.primary, '', 'Save keeps the one primary slot');
        assert.equal(allowButton(fx).dataset.primary, undefined);
        assert.match(note(fx), /TELEGRAM_BOT_TOKEN.*telegram/);
        assert.ok(fx.host.querySelectorAll('secret-request-reader').map((row) => row.textContent)
            .includes('mailer · enabled · asks for TELEGRAM_BOT_TOKEN · allowed to read it'), 'who already reads it is named');
        assert.equal(allowButton(fx).disabled, true, 'nothing typed yet');
        type(input(fx), CANARY);
        assert.equal(allowButton(fx).disabled, false);
        input(fx).listeners.get('keydown')?.({ key: 'Enter', preventDefault() {} });
        await turn();
        const [post] = fx.calls.filter((body) => !body.action);
        assert.equal('allow' in post, false, 'Enter is the plain Save');
        assert.match(q(fx.card, 'chat-secret-status').textContent, /did not grant access/);
        assert.deepEqual(outcomeLines(fx), []);
    } finally { fx.restore(); }
});

test('Save, allow and turn on sends the shown files and fields; the card says save, access and switch apart', async () => {
    let sent = null;
    const fx = setup({ statusBody: status(false, { skill: SKILL }),
        save: (body) => { sent = body; return reply(200, allowed({ state: 'granted', granted_keys: ['TELEGRAM_BOT_TOKEN'], enabled: true })); } });
    try {
        await opened(fx);
        assert.equal(q(fx.host, 'secret-request-allow-label').textContent, 'Save, allow and turn on telegram');
        assert.match(note(fx), /TELEGRAM_BOT_TOKEN.*telegram/);
        type(input(fx), CANARY);
        allowButton(fx).click();
        await turn();
        assert.deepEqual(sent.allow, { content_hash: 'sha256:files-shown', installation: 'inst-shown', fields: ['botToken'], enable: true });
        assert.deepEqual(sent.values, { botToken: CANARY });
        assert.equal(fx.host.children.length, 0, 'the form closed');
        assert.equal(q(fx.card, 'chat-secret-status').textContent, 'Saved to Settings: botToken.');
        assert.match(outcomeLines(fx)[0], /Granted TELEGRAM_BOT_TOKEN to telegram/);
        assert.match(outcomeLines(fx)[1], /telegram was on/);
        assert.ok(!everything(fx.card).includes(CANARY));
    } finally { fx.restore(); }
});

test('an older server without skill facts, or a skill that already reads the key, offers Save alone', async () => {
    for (const extra of [{}, { skill: { ...SKILL, granted: ['TELEGRAM_BOT_TOKEN'] } }, { skill: { ...SKILL, supported: false } },
        { skill: { ...SKILL, installation: '' } }, { skill: { name: 'telegram', available: false } }]) {
        const fx = setup({ statusBody: status(false, extra) });
        try {
            await opened(fx);
            assert.equal(allowButton(fx).hidden, true, JSON.stringify(extra));
            assert.equal(note(fx), '');
        } finally { fx.restore(); }
    }
    const fx = setup({ statusBody: status(false, { skill: { ...SKILL, admitted: false, waiting: ['TELEGRAM_BOT_TOKEN'] } }) });
    try {
        await opened(fx);
        assert.equal(allowButton(fx).hidden, true);
        assert.match(note(fx), /telegram.*these exact files.*explicitly accepted/);
    } finally { fx.restore(); }
});

test('files not reviewed yet: the note and the card say the consent waits; an Off is never turned on', async () => {
    const fx = setup({ statusBody: status(true, { skill: { ...SKILL, admitted: false } }),
        save: () => reply(200, allowed({ state: 'deferred', reason: 'not_admitted', enabled: false, enable_reason: 'deferred' })) });
    try {
        await opened(fx);
        assert.equal(q(fx.host, 'secret-request-allow-label').textContent, 'Replace, allow and turn on telegram');
        assert.match(note(fx), /these exact files.*explicitly accepted.*does not approve/);
        assert.doesNotMatch(note(fx), /Other skills’ permissions/, 'the Replace warning, not the note, says other skills keep theirs');
        assert.match(q(fx.host, 'secret-request-warning').textContent, /Other skills’ permissions stay the same/);
        type(input(fx), CANARY);
        allowButton(fx).click();
        await turn();
        assert.equal(fx.accepted.at(-1).secret_outcome.allow.state, 'deferred');
        assert.deepEqual(fx.calls.at(-1).allow.fields, ['botToken']);
        assert.match(outcomeLines(fx)[0], /TELEGRAM_BOT_TOKEN.*exact files.*explicitly accepted/);
        assert.match(outcomeLines(fx)[1], /Turn-on waits.*Off/);
        assert.match(fx.card.querySelectorAll('chat-secret-reader')[1].textContent, /mailer.*was not allowed/);
    } finally { fx.restore(); }
    const off = setup({ statusBody: status(false, { skill: { ...SKILL, switch: 'off' } }) });
    try {
        await opened(off);
        assert.equal(q(off.host, 'secret-request-allow-label').textContent, 'Save and allow telegram');
        assert.match(note(off), /telegram stays off; it was turned off\.$/);
    } finally { off.restore(); }
});

test('a refused, unknown or pending allow is never shown as access, nor a failed turn-on as on', () => {
    const base = { skill: 'telegram', keys: ['TELEGRAM_BOT_TOKEN'], enable: true };
    const lines = (fields) => allowLines({ ...base, ...fields });
    const denied = lines({ state: 'refused', reason: 'skill_changed', enabled: false, enable_reason: 'not_allowed' });
    assert.deepEqual(denied.map(({ tone }) => tone), ['warn', 'warn']);
    assert.match(denied[0].text, /files changed/);
    assert.match(denied[1].text, /access was not granted/);
    assert.ok(lines({ state: 'unknown', enabled: null, enable_reason: 'unknown' }).every(({ text, tone }) => tone === 'warn' && /unknown/.test(text)));
    for (const state of ['requested', 'running']) assert.deepEqual(lines({ state }).map(({ tone }) => tone), ['neutral']);
    assert.match(lines({ enable: false, state: 'refused', reason: 'installation_ended' })[0].text, /installation.*ended/);
    const notStarted = lines({ enable: false, state: 'refused', reason: 'not_started' })[0].text;
    assert.match(notStarted, /grant step did not start/);  // neutral: a restart or a lane that never started its job
    assert.doesNotMatch(notStarted, /restart/i);
    assert.match(lines({ enable: false, state: 'refused', reason: 'update_rolled_back' })[0].text, /rolled back/);
    const partial = lines({ state: 'granted', keys: ['FIRST', 'NEXT'], granted_keys: ['FIRST'], enabled: false, enable_reason: 'conflict' });
    assert.match(partial[0].text, /Granted FIRST to telegram/);
    assert.doesNotMatch(partial[0].text, /NEXT/);
    assert.match(partial[1].text, /not turned on/);
    assert.match(lines({ state: 'granted', granted_keys: ['TELEGRAM_BOT_TOKEN'], enabled: false, enable_reason: 'owner_off' })[1].text, /stayed off/);
    assert.deepEqual(allowLines(undefined), []);
});

for (const [name, current, notice] of [
    ['files changed since the form read them: nothing saved, the draft stays, the next allow names the new files',
        { ...SKILL, content_hash: 'sha256:files-now' }, /files or installation changed/i],
    ['same files after reinstall: the form names the new installation before another allow',
        { ...SKILL, installation: 'inst-now' }, /different installation/i],
]) {
    test(name, async () => {
        const bodies = [];
        const fx = setup({ statusBody: status(false, { skill: SKILL }), save: (body) => {
            bodies.push(body);
            return bodies.length === 1
                ? reply(409, { ok: false, saved: false, reason_code: 'secret_allow_stale', decision_id: DECISION, state: 'open',
                    keys: [{ id: 'botToken', setting_key: 'TELEGRAM_BOT_TOKEN', configured: false }], consumers: READERS,
                    skill: current })
                : reply(200, allowed({ state: 'granted', enabled: true }));
        } });
        try {
            const revision = (skill) => skill.content_hash.replace(/^sha256:/, '').slice(0, 12);
            await opened(fx);
            assert.ok(note(fx).includes(SKILL.installation) && note(fx).includes(revision(SKILL)));
            assert.equal(q(fx.host, 'secret-request-allow-label').textContent, 'Save, allow and turn on telegram');
            type(input(fx), CANARY);
            allowButton(fx).click();
            await turn();
            assert.match(q(fx.host, 'secret-request-error').textContent, notice);
            assert.ok(note(fx).includes(current.installation) && note(fx).includes(revision(current)));
            assert.ok(!note(fx).includes(current.installation === SKILL.installation ? revision(SKILL) : SKILL.installation));
            assert.equal(input(fx).value, CANARY, 'the draft stays');
            assert.equal(bodies.length, 1, 'nothing is retried automatically');
            allowButton(fx).click();
            await turn();
            assert.deepEqual(bodies[1].allow, { content_hash: current.content_hash, installation: current.installation,
                fields: ['botToken'], enable: true });
            assert.notEqual(bodies[1].request_id, bodies[0].request_id, 'another recipient or revision is a new request');
        } finally { fx.restore(); }
    });
}
