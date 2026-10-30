// The room's chat model in the real composer (composer_model_picker.js; DESIGN "Composer owner controls"): the
// trigger stands first on the row and names what new work starts on, from the room's own readback only (a Project
// never borrows Main's choice); a choice posts once to its room's owner writer and says Saving until the readback;
// Send waits for that save while typed Panic never does; a refusal keeps the draft, says why and returns the trigger
// to the saved choice. The sheet itself and its geometry are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { composerMarkup } from '../modules/composer.js';
import { sourceMark } from '../modules/composer_model_picker.js';
import { ElementStub, installDom, restoreDom } from './chat_dom_fixture.js';

const SOL = 'claudexor::codex=gpt-6.1-sol';
const ASTRA = 'claudexor::codex=gpt-6-astra';
const FLASH = 'openai-compatible::flash';
const ROUTES = { main: { model: SOL, use_local: false }, fallback: { model: FLASH, use_local: false } };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function room({ chatId = 1, projectId = '' } = {}) {
    const saves = [];
    const env = installDom(async (url, init = {}) => {
        if (String(url).startsWith('/api/chat/history')) return { ok: true, json: async () => ({ messages: [], window: { complete: true } }) };
        if (String(url) === '/api/owner/chat-model') {
            return new Promise((resolve) => saves.push({ body: JSON.parse(init.body), resolve }));
        }
        return { ok: true, json: async () => ({ active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) };
    });
    const frames = [];
    let generation = 0;
    const instance = createChatInstance({
        ws: { on: () => () => {}, isConnected: () => true,
            send: (frame) => { frames.push(frame); return { status: 'sent', clientMessageId: `cm-${frames.length}` }; } },
        state: { activePage: 'chat', projectChatIds: new Set(chatId === 1 ? [] : [chatId]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, projectId, idPrefix: chatId === 1 ? 'chat' : `pchat-${projectId}`, mountEl: env.mount,
        asPanel: chatId !== 1,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const picker = instance.composer.modelPicker;
    const page = env.mount.children[0];
    const row = page.children.find((node) => node.classList.contains('chat-toolbar-row'));
    const prefix = chatId === 1 ? 'chat' : `pchat-${projectId}`;
    const input = document.byId.get(`${prefix}-input`);
    const send = document.byId.get(`${prefix}-send`);
    const press = (text) => {
        input.value = text;
        const event = { key: 'Enter', target: input, preventDefault() { event.defaultPrevented = true; } };
        for (const listener of input.listeners.get('keydown')) listener(event);
    };
    const scope = () => picker.trigger.children.find((node) => node.classList.contains('chat-model-scope'));
    const reply = (index, body, ok = true, status = 200) => saves[index].resolve({ ok, status, json: async () => body });
    return { instance, picker, row, input, send, press, frames, saves, scope, reply, close() { instance.destroy(); restoreDom(env.prior); } };
}

test('the trigger stands first on the row and names Main\'s model until the room chose its own', () => {
    const r = room();
    try {
        const markup = composerMarkup();
        assert.ok(markup.indexOf('chat-mobile-nav') < 0 || markup.indexOf('mobile-nav') < markup.indexOf('chat-model-picker'));
        assert.ok(markup.indexOf('id="chat-model-picker"') < markup.indexOf('id="chat-subagents-mode"'), 'before the capsules');
        assert.ok(r.picker.el.classList.contains('chat-model-picker') && r.picker.trigger, 'mounted in its slot');
        assert.equal(r.picker.el.dataset.known, 'false', 'nothing is named before the room\'s readback');
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'main' }, model_routes: ROUTES });
        assert.deepEqual([r.picker.nameEl().textContent, r.picker.el.dataset.known, r.picker.el.dataset.moved, r.scope().hidden],
            ['gpt-6.1-sol', 'true', 'false', true]);
        // No catalog read yet: the source is its id and the model its id, said as they are.
        assert.match(r.picker.trigger.getAttribute('aria-label'), /gpt-6\.1-sol, codex · Subscription\. Follows Main\./);
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'fallback' }, model_routes: ROUTES });
        assert.deepEqual([r.picker.nameEl().textContent, r.picker.el.dataset.moved, r.scope().hidden, r.scope().textContent],
            ['flash', 'true', false, 'this chat']);
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'fallback' }, model_routes: { main: ROUTES.main, fallback: null } });
        assert.equal(r.picker.nameEl().textContent, 'gpt-6.1-sol', 'no Fallback set: new work starts on Main, and says so');
        assert.match(r.picker.trigger.getAttribute('aria-label'), /Fallback is not set, so new work starts on Main\./);
    } finally { r.close(); }
});

test('a choice saves once; Send waits for it, Panic never does; a refusal keeps the draft and the saved choice', async () => {
    const r = room();
    try {
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'main' }, model_routes: ROUTES });
        void r.picker.choose({ kind: 'model', model: ASTRA });
        assert.deepEqual(r.saves.map((save) => save.body), [{ kind: 'model', model: ASTRA, use_local: false }]);
        assert.deepEqual([r.picker.nameEl().textContent, r.scope().textContent, r.picker.el.dataset.saving], ['gpt-6-astra', 'Saving', 'true']);
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'main' }, model_routes: ROUTES });
        assert.equal(r.picker.nameEl().textContent, 'gpt-6-astra', 'a stale state read never overwrites a save in flight');
        r.press('/panic');
        assert.deepEqual(r.frames.map((frame) => frame.content), ['/panic'], 'Panic is delivered before the save answers');
        r.frames.length = 0;
        r.press('start on Astra');
        await tick();
        assert.deepEqual([r.frames.length, r.send.textContent, r.send.disabled], [0, 'Saving', true], 'Send waits for the save');
        r.reply(0, { ok: true, project_id: '', model_choice: { kind: 'model', model: ASTRA, use_local: false }, previous_model_choice: { kind: 'main' } });
        await tick(); await tick();
        assert.deepEqual(r.frames.map((frame) => frame.content), ['start on Astra'], 'then the message goes');
        assert.deepEqual([r.scope().textContent, r.scope().hidden, r.picker.el.dataset.saving], ['this chat', false, 'false']);

        document.body = new ElementStub('body', document);  // the toast stack's host
        void r.picker.choose({ kind: 'model', model: 'anthropic::claude-opus-5-5' });
        r.press('still here');
        await tick();
        r.reply(1, { error: 'The model choice must be an object with a kind', saved: false, code: 'chat_model_invalid' }, false, 400);
        await tick(); await tick();
        assert.equal(r.frames.length, 1, 'a refused save sends nothing');
        assert.equal(r.input.value, 'still here', 'and keeps the draft');
        assert.deepEqual([r.picker.nameEl().textContent, r.picker.el.dataset.saving], ['gpt-6-astra', 'false'], 'the saved choice again');
        r.press('next');
        await tick();
        assert.deepEqual(r.frames.map((frame) => frame.content), ['start on Astra', 'next'], 'an earlier refusal never blocks later sends');
    } finally { r.close(); }
});

test('a Project reads and saves its own choice, never Main\'s', async () => {
    const r = room({ chatId: 7, projectId: 'p7' });
    try {
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'model', model: ASTRA, use_local: false }, model_routes: ROUTES, projects: [] });
        assert.equal(r.picker.el.dataset.known, 'false', 'Main\'s choice is never substituted for an unread room');
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'model', model: ASTRA, use_local: false }, model_routes: ROUTES,
            projects: [{ id: 'p7', chat_id: 7, cognition: { context_mode: 'max', subagents_mode: 'auto',
                effort_range: { min: 'low', recommended: 'high', max: 'max' } } }] });
        assert.deepEqual([r.picker.nameEl().textContent, r.picker.el.dataset.moved], ['gpt-6.1-sol', 'false'],
            'a row without its own choice follows Main\'s reference');
        void r.picker.choose({ kind: 'fallback' });
        assert.deepEqual(r.saves.map((save) => save.body), [{ kind: 'fallback', project_id: 'p7' }]);
        r.reply(0, { ok: true, project_id: 'p7', model_choice: { kind: 'fallback' }, previous_model_choice: { kind: 'main' } });
        await tick(); await tick();
        assert.deepEqual([r.picker.nameEl().textContent, r.scope().textContent], ['flash', 'this chat']);
    } finally { r.close(); }
});

test('an HTTP success without the saved-choice contract is not a successful save', async () => {
    const r = room();
    try {
        r.instance.hydrateStateSnapshot({ model_choice: { kind: 'main' }, model_routes: ROUTES });
        const pending = r.picker.choose({ kind: 'model', model: ASTRA });
        r.reply(0, { project_id: '' });
        assert.equal(await pending, false);
        assert.equal(r.picker.el.dataset.moved, 'false');
        assert.equal(r.picker.nameEl().textContent, 'gpt-6.1-sol');
    } finally { r.close(); }
});

test('a source draws its own sourced mark; one without a mark shows its initial, never an invented logo', () => {
    const api = (kind, label) => ({ id: kind, lane: 'api', kind, label });
    const openrouter = sourceMark(api('openrouter', 'OpenRouter'));
    assert.match(openrouter, /^<svg class="chat-model-mark" viewBox="0 0 24 24"/);
    assert.match(openrouter, /d="M16\.778 1\.844v1\.919/, 'OpenRouter\'s split arrows (simple-icons 16.36.0), not a boxed O');
    assert.match(sourceMark({ ...api('openrouter@work', 'OpenRouter · work'), kind: 'openrouter' }), /d="M16\.778/,
        'a named OpenRouter key keeps the mark');
    assert.match(sourceMark(api('openai', 'OpenAI')), /^<svg class="chat-model-mark"/);
    assert.match(sourceMark(api('zai', 'Z.ai (GLM)')), /chat-model-initial" aria-hidden="true">Z</);
});
