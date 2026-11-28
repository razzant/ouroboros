// The Main and Project composer (composer.js), mounted by every chat room: the same chat-* ids (namespaced in a
// Project panel), the room's draft kept across the room's teardown and cleared by Send, the draft API that later
// controls use, the input recall seeded once from the room's own history, and the "+" menu with its mount points.
// Enter and Swarm are in chat_composer_enter.test.js; the real engines are in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { composerMarkup, createPlusMenu } from '../modules/composer.js';
import { applyPayload } from '../modules/i18n.js';
import { ElementStub, installDom, memoryStorage, PREVIEW_SCOPE, restoreDom, roomKey } from './chat_dom_fixture.js';
import { installDomDouble } from './dom_double.js';

const tick = () => new Promise((resolve) => setImmediate(resolve));
const SUFFIXES = ['input-area', 'attachment-preview', 'model-mode', 'subagents-mode', 'context-mode', 'plus', 'file-input',
    'input', 'send'];

function fetchStub(history = []) {
    return async (url) => ({ ok: true, json: async () => (String(url).startsWith('/api/chat/history')
        ? { profile_scope: PREVIEW_SCOPE, messages: history, window: { complete: true } }
        : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true }) });
}

function mount(env, { chatId = 1, projectId = '', idPrefix = 'chat', frames = [], status = 'sent', handlers = {} } = {}) {
    let generation = 0;
    return createChatInstance({
        ws: { on: (event, fn) => { (handlers[event] ||= []).push(fn); return () => {}; }, isConnected: () => true,
            send: (frame) => { if (status !== 'failed') frames.push(frame); return { status, clientMessageId: `cm-${frames.length}` }; } },
        state: { activePage: 'chat', projectChatIds: new Set(chatId === 1 ? [] : [chatId]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId, projectId, idPrefix, mountEl: env.mount, asPanel: chatId !== 1,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
}

const PROJECT = { chatId: 7, projectId: 'p7', idPrefix: 'pchat-p7' };
const ownDraft = (chatId = 1) => JSON.parse(sessionStorage.getItem(roomKey('ouro_chat_text_v1', chatId)))?.text ?? null;
const node = (instance, id) => instance.page.querySelector(`[id="${id}"]`);

test('every room mounts the composer with the chat-* ids, namespaced in a Project panel with the mirror classes', () => {
    for (const options of [{}, PROJECT]) {
        const prefix = options.idPrefix || 'chat';
        const env = installDom(fetchStub());
        const instance = mount(env, options);
        try {
            for (const suffix of SUFFIXES) {
                const el = node(instance, `${prefix}-${suffix}`);
                assert.ok(el, `${prefix}-${suffix}`);
                if (prefix !== 'chat') {
                    assert.equal(el.classList.contains(`chat-${suffix}`), true, `${suffix}: the mirror class keeps the styling`);
                    assert.equal(node(instance, `chat-${suffix}`), null, `${suffix}: no Main id inside a Project panel`);
                }
            }
            assert.equal(instance.composer.input, node(instance, `${prefix}-input`), 'the composer owns the room\'s field');
            assert.equal(Object.hasOwn(node(instance, `${prefix}-plus`).dataset, 'composerPlus'), true);
            assert.equal(typeof instance.composer.plusMenu.setSections, 'function');
            assert.equal(Object.hasOwn(instance.composer.voiceSlot.dataset, 'composerVoice'), true);
            assert.equal(Object.hasOwn(instance.composer.chipsSlot.dataset, 'composerChips'), true);
        } finally { instance.destroy(); restoreDom(env.prior); }
    }
});

test('"+" stands where the paperclip stood; the voice and chips slots are empty, hidden and in their places', () => {
    const markup = composerMarkup();
    const tag = (pattern) => markup.match(pattern)?.[0] || '';
    const plus = tag(/<button class="chat-plus-btn[ "][^>]*>/);
    for (const attr of ['ui-hit-area', 'id="chat-plus"', 'type="button"', 'data-composer-plus', 'aria-label="More actions"',
        'aria-haspopup="menu"', 'aria-expanded="false"']) assert.ok(plus.includes(attr), attr);
    assert.doesNotMatch(markup, /chat-attach-btn|id="chat-attach"/, 'no paperclip beside it');
    assert.match(markup, /<span class="chat-voice-slot" data-composer-voice hidden><\/span>/);
    assert.match(markup, /<span class="chat-composer-chips" data-composer-chips hidden><\/span>/);
    // Text row: "+", the file input, the field, the voice slot, Send. Capsule row: Model, Subagents, Context, then the
    // chips slot (Effort mounts between them; Main's questions pill joins after). No one-shot Swarm pill anywhere, and
    // no model control but the capsule.
    assert.doesNotMatch(markup, /chat-model-picker|chat-model-trigger/);
    assert.doesNotMatch(markup, /chat-swarm|chat-composer-pills|data-armed/);
    const order = (names) => names.map((name) => markup.indexOf(name));
    const rising = (list) => list.every((value, i) => value >= 0 && (i === 0 || value > list[i - 1]));
    assert.ok(rising(order(['chat-plus-btn', 'chat-file-input', 'id="chat-input"', 'data-composer-voice', 'chat-send-group'])));
    assert.ok(rising(order(['chat-model-mode', 'chat-subagents-mode', 'chat-context-mode', 'data-composer-chips', 'chat-text-row'])));
});

const SOL = 'claudexor::codex=gpt-6.1-sol';
const ASTRA = 'claudexor::codex=gpt-6-astra';
const KIMI = 'moonshotai/kimi-k3';
const COGNITION = { context_mode: 'max', subagents_mode: 'auto', effort_range: { min: 'low', recommended: 'high', max: 'max' } };

/* A Project room whose Model capsule can be worked: its saves wait in `saves` to be answered (`reply`), the recent
   models are the rows its sheet offers, and `pick(model)` presses one of them as a pointer would. */
function modelRoom({ recents = [SOL, KIMI], own = { kind: 'model', model: ASTRA, use_local: false }, staged = null } = {}) {
    const saves = [];
    const history = [];  // the rows the room's next history read answers with
    const env = installDom(async (url, init = {}) => {
        if (String(url) === '/api/owner/chat-model') return new Promise((resolve) => saves.push({ body: JSON.parse(init.body), resolve }));
        if (String(url).startsWith('/api/ui/preferences')) return { ok: true, json: async () => ({ model_recents: recents.map((model) => ({ model, use_local: false })) }) };
        return fetchStub(history)(url);
    });
    // A Reply the room kept staged: the composer opens replying to that row.
    if (staged) sessionStorage.setItem(roomKey('ouro_chat_reply', 7), JSON.stringify(staged));
    const frames = [];
    const handlers = {};
    const instance = mount(env, { ...PROJECT, frames, handlers });
    const state = (choice = own) => ({ main_route: { model: SOL, use_local: false },
        projects: [{ id: 'p7', chat_id: 7, model_choice: choice, cognition: COGNITION }] });
    instance.hydrateStateSnapshot(state());
    document.defaultView = window;  // the sheet's place reads the page's window, as a browser's document has one
    document.body = new ElementStub('body', document);  // where the sheet and the toast stack stand
    const capsule = node(instance, 'pchat-p7-model-mode');
    const input = node(instance, 'pchat-p7-input');
    const send = node(instance, 'pchat-p7-send');
    const pick = async (model) => {
        for (const fn of capsule.querySelector('.chat-model-head').listeners.get('click')) fn({ detail: 1 });
        await tick(); await tick();  // the sheet's own read of the recent models
        const list = document.body.children.find((made) => made.classList.contains('model-sheet')).querySelector('.model-sheet-list');
        const row = list.querySelectorAll('.chat-model-row').find((made) => made.dataset.row === `route|${model}`);
        assert.ok(row, `a row for ${model}`);
        for (const fn of list.listeners.get('click')) fn({ target: { closest: (selector) => (selector === '.chat-model-row' ? row : null) } });
    };
    const press = (text) => {
        input.value = text;
        for (const listener of input.listeners.get('input') || []) listener({ target: input });
        const event = { key: 'Enter', target: input, preventDefault() { event.defaultPrevented = true; } };
        for (const listener of input.listeners.get('keydown')) listener(event);
    };
    const type = (text) => { input.value = text; for (const listener of input.listeners.get('input') || []) listener({ target: input }); };
    const reply = (index, body, ok = true, status = 200) => saves[index].resolve({ ok, status, json: async () => body });
    const saved = (index, model, token) => reply(index, { ok: true, project_id: 'p7', model_choice: { kind: 'model', model, use_local: false },
        previous_model_choice: own, choice_token: token });
    const fire = (event, message) => { for (const fn of handlers[event] || []) fn(message); };
    // The host's saved row of the `n`-th frame this room sent, taken as new work or not; `delivered: false` is a row a
    // host process took and ended with (neither dispatch fact).
    const savedRow = (n, { newWork = false, delivered = true } = {}) => ({ role: 'user', chat_id: 7, ts: `2026-01-02T03:04:0${n}+00:00`,
        source: 'web', sender_session_id: frames[n - 1].sender_session_id, client_message_id: `cm-${n}`, ingress_accepted: true,
        ...(delivered ? { ingress_dispatched: true } : {}), ...(newWork ? { new_work: true } : {}) });
    const copy = (n) => instance.page.querySelectorAll('.chat-bubble.user').find((bubble) => bubble.dataset.clientMessageId === `cm-${n}`);
    const actions = (n) => (copy(n)?.querySelectorAll('.msg-pending') || []).flatMap((note) => note.children.filter((made) => made.tagName === 'BUTTON'));
    return { env, instance, frames, saves, state, capsule, input, send, pick, press, type, reply, saved,
        /** A frame from the host, as the socket hands it to the room. */
        fire,
        /** That row as the host's live echo. */
        echo: (n, facts) => fire('chat', { type: 'chat', content: frames[n - 1].content, ...savedRow(n, facts) }),
        /** That row as the room's next history read answers it (a reconnect reads the room again). */
        async reread(n, facts) {
            history.push({ ...savedRow(n, facts), text: frames[n - 1].content, history_id: `h-cm-${n}`, history_position: { source: 'chat', offset: n } });
            fire('open', { previouslyConnected: true });
            for (let i = 0; i < 12; i += 1) await tick();
        },
        /** The actions the unconfirmed copy of the `n`-th message offers, by their words; `act` presses one. */
        actions: (n) => actions(n).map((made) => made.textContent),
        act: (n, word) => actions(n).find((made) => made.textContent === word).click(),
        value: () => capsule.querySelector('.chat-capsule-value').textContent,
        toasts: () => document.body.querySelectorAll('.toast, [data-toast]').map((made) => made.textContent),
        close() { instance.destroy(); restoreDom(env.prior); } };
}

test('Send waits for the room\'s model to be saved, Panic never does; a refused save keeps the draft', async () => {
    const r = modelRoom();
    try {
        assert.equal(r.value(), 'gpt-6-astra', 'the room\'s own model, from the room\'s own row');
        await r.pick(SOL);
        assert.deepEqual(r.saves.map((save) => save.body), [{ kind: 'model', model: SOL, use_local: false, project_id: 'p7' }]);
        r.instance.hydrateStateSnapshot(r.state());
        r.press('/panic');
        assert.deepEqual(r.frames.map((frame) => frame.content), ['/panic'], 'Panic is delivered before the save answers');
        r.frames.length = 0;
        r.press('start on Sol');
        await tick();
        assert.deepEqual([r.frames.length, r.send.textContent, r.send.disabled], [0, 'Saving', true], 'Send waits for the save');
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.new_work]), [['start on Sol', 'token-sol']],
            'then the message goes, as new work on the choice just saved');
        assert.equal(r.value(), 'gpt-6.1-sol', 'and the capsule shows the room\'s model');

        r.instance.hydrateStateSnapshot(r.state({ kind: 'model', model: SOL, use_local: false }));
        await r.pick(KIMI);
        r.press('still here');
        await tick();
        r.reply(1, { error: 'The project is not available', saved: false, code: 'project_unavailable' }, false, 404);
        await tick(); await tick();
        assert.equal(r.frames.length, 1, 'a refused save sends nothing');
        assert.equal(r.input.value, 'still here', 'and keeps the draft');
        assert.equal(r.value(), 'gpt-6.1-sol', 'the saved choice again');
        r.press('next');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.new_work]), [['start on Sol', 'token-sol'], ['next', undefined]],
            'an earlier refusal never blocks later sends, and arms none of them');
    } finally { r.close(); }
});

test('after a pick of another model Send and the empty field say the next message is a new task, before the press', async () => {
    const r = modelRoom();
    try {
        const said = () => [r.send.textContent, r.send.title, r.input.placeholder];
        const ORDINARY = ['Send', 'Send message', 'Message Ouroboros...'];
        const NEW_TASK = ['New task', 'Send as a new task on gpt-6.1-sol', 'New task...'];
        assert.deepEqual(said(), ORDINARY, 'nothing picked: an ordinary message');
        await r.pick(ASTRA);  // the row current now
        assert.deepEqual([r.saves.length, said()], [0, ORDINARY], 'the model the room already has arms nothing');
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        assert.deepEqual(said(), NEW_TASK);
        assert.ok(NEW_TASK[2].length <= ORDINARY[2].length, 'the armed field\'s words never outgrow the ordinary ones: no state adds a wrapped line');
        // A slash command is never new work, and a Reply to a work's own row reaches that work: Send says so, and
        // the choice waits for its own message.
        r.type('/status');
        assert.deepEqual(said(), ORDINARY);
        r.type('compare the drafts');
        assert.deepEqual(said().slice(0, 2), NEW_TASK.slice(0, 2));
        r.press('/status');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.new_work]), [['/status', undefined]], 'the command names no choice');
        assert.deepEqual(said(), NEW_TASK, 'and spends none');
        // The message goes naming the saved choice; until the host says how it took it, Send is an ordinary one again.
        r.press('compare the drafts');
        await tick();
        assert.deepEqual(r.frames.at(-1).new_work, 'token-sol');
        assert.deepEqual(said(), ORDINARY);
        r.press('and one more thing');
        await tick();
        assert.equal(r.frames.at(-1).new_work, undefined, 'one message names it, never the next too');
        // The host's saved row answers, by its live echo: given to a work as its Reply, the choice is not spent.
        // Taken as new work, it is. A row of a message that named no choice says nothing about it.
        r.echo(3);
        assert.deepEqual(said(), ORDINARY, 'another message\'s row is not this one\'s word');
        r.echo(2);
        assert.deepEqual(said(), NEW_TASK);
        r.press('now as new work');
        await tick();
        r.echo(r.frames.length, { newWork: true });
        assert.deepEqual([r.frames.at(-1).new_work, said()], ['token-sol', ORDINARY]);
        r.echo(r.frames.length);  // a repeated row without the mark is no second word
        assert.deepEqual(said(), ORDINARY);
    } finally { r.close(); }
});

test('a pick made before the host\'s word on the last message is the next message\'s own: that word never takes it', async () => {
    const r = modelRoom();
    try {
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        r.press('start on Sol');
        await tick();
        assert.deepEqual([r.frames.at(-1).new_work, r.send.textContent], ['token-sol', 'Send']);
        // The echo has not come (a slow host, or a socket that dropped it); the owner picks another model.
        await r.pick(KIMI);
        r.saved(1, KIMI, 'token-kimi');
        await tick(); await tick();
        assert.deepEqual([r.send.textContent, r.send.title], ['New task', 'Send as a new task on kimi-k3']);
        r.echo(1, { newWork: true });  // the first message was taken as new work on Sol
        assert.deepEqual([r.send.textContent, r.send.title], ['New task', 'Send as a new task on kimi-k3'],
            'its success spends its own choice, never the later one');
        r.press('and this on Kimi');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.new_work]),
            [['start on Sol', 'token-sol'], ['and this on Kimi', 'token-kimi']]);
        r.echo(2, { newWork: true });
        assert.equal(r.send.textContent, 'Send');
    } finally { r.close(); }
});

/* A room whose window picked Sol and sent `start on Sol` naming it (`cm-1`); the host's echo never came. */
async function sentUnheard() {
    const r = modelRoom();
    await r.pick(SOL);
    r.saved(0, SOL, 'token-sol');
    await tick(); await tick();
    r.press('start on Sol');
    await tick();
    assert.deepEqual([r.frames.at(-1).new_work, r.send.textContent], ['token-sol', 'Send']);
    return r;
}
const NEW_TASK_ON_SOL = ['New task', 'Send as a new task on gpt-6.1-sol'];

test('a message that named the choice and was never confirmed: Discard frees the choice, Send again keeps its one word', async () => {
    const discarded = await sentUnheard();
    try {
        discarded.fire('close');
        assert.deepEqual(discarded.actions(1), ['Send again', 'Discard']);
        assert.equal(discarded.send.textContent, 'Send', 'in doubt, its choice is still its own');
        discarded.act(1, 'Discard');
        assert.deepEqual([discarded.send.textContent, discarded.send.title], NEW_TASK_ON_SOL, 'given up, the choice waits for a message again');
        discarded.press('start on Sol after all');
        await tick();
        assert.deepEqual([discarded.frames.at(-1).content, discarded.frames.at(-1).new_work, discarded.send.textContent],
            ['start on Sol after all', 'token-sol', 'Send']);
        discarded.echo(1, { newWork: true });  // the given-up message's late row is not the new message's word
        assert.equal(discarded.send.textContent, 'Send');
        discarded.echo(2);  // the host gave the new one to a work: its choice waits again
        assert.deepEqual([discarded.send.textContent, discarded.send.title], NEW_TASK_ON_SOL);
    } finally { discarded.close(); }
    const repeated = await sentUnheard();
    try {
        repeated.fire('close');
        repeated.act(1, 'Send again');
        assert.deepEqual([repeated.frames.length, repeated.frames.at(-1).client_message_id, repeated.frames.at(-1).new_work, repeated.send.textContent],
            [2, 'cm-1', 'token-sol', 'Send'], 'the same message with the same id names the same choice, and Send promises no second task');
        repeated.echo(1, { newWork: true });
        repeated.press('and then this');
        await tick();
        assert.deepEqual([repeated.frames.at(-1).content, repeated.frames.at(-1).new_work], ['and then this', undefined], 'taken as new work, the choice is spent');
    } finally { repeated.close(); }
});

test('the saved row in history answers for the choice when the echo never came, as the echo would', async () => {
    const taken = await sentUnheard();
    try {
        await taken.reread(1, { newWork: true });
        assert.equal(taken.send.textContent, 'Send');
        taken.press('and then this');
        await tick();
        assert.equal(taken.frames.at(-1).new_work, undefined, 'taken as new work, the choice is spent');
    } finally { taken.close(); }
    const replied = await sentUnheard();
    try {
        await replied.reread(1);  // the host gave it to a work as its Reply
        assert.deepEqual([replied.send.textContent, replied.send.title], NEW_TASK_ON_SOL, 'the choice still waits for its own message');
    } finally { replied.close(); }
    // Taken as new work by a host process that ended: delivery is not known, and the one next step is a new message.
    // Its choice stays with it until the owner takes that step; then the new message is the new task again.
    const doubted = await sentUnheard();
    try {
        await doubted.reread(1, { newWork: true, delivered: false });
        assert.deepEqual([doubted.send.textContent, doubted.actions(1)], ['Send', ['Send as new']]);
        doubted.act(1, 'Send as new');  // its words go back to the composer (the browser lane reads them there)
        assert.deepEqual([document.activeElement === doubted.input, doubted.send.textContent, doubted.send.title], [true, ...NEW_TASK_ON_SOL]);
    } finally { doubted.close(); }
});

test('Send is shared with a conversation feature\'s entry: its face stands, and handed back the button says New task again', async () => {
    const observers = [];
    globalThis.MutationObserver = class {
        constructor(fn) { this.fn = fn; this.targets = []; this.watched = new Set(); observers.push(this); }
        observe(target) { this.targets.push(target); this.watched.add(target); }
        disconnect() { this.targets = []; }
        takeRecords() { return []; }
    };
    const changed = (target) => observers.filter((made) => made.targets.includes(target)).forEach((made) => made.fn([]));
    const r = modelRoom();
    try {
        const face = () => [r.send.textContent, r.send.title];
        const stateRead = () => r.instance.hydrateStateSnapshot(r.state({ kind: 'model', model: SOL, use_local: false }));
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        assert.deepEqual([...face(), r.input.placeholder], [...NEW_TASK_ON_SOL, 'New task...']);
        // The feature dresses the empty field's button as its entry; no state read writes over it or its title.
        Object.assign(r.send, { textContent: 'Talk', title: 'Talk with Ouroboros' });
        changed(r.send);
        stateRead();
        assert.deepEqual(face(), ['Talk', 'Talk with Ouroboros']);
        // The owner types: the feature hands the button back as a plain Send, and the composer says its own word.
        Object.assign(r.send, { textContent: 'Send', title: 'Send message' });
        changed(r.send);
        assert.deepEqual(face(), NEW_TASK_ON_SOL, 'the message still goes as a new task, and the button says so');
        // A call writes its own words in the empty field and gives the ordinary ones back when it ends.
        r.input.placeholder = 'Type to Ouroboros';
        stateRead();
        assert.equal(r.input.placeholder, 'Type to Ouroboros');
        r.input.placeholder = 'Message Ouroboros...';
        changed(r.input);
        assert.equal(r.input.placeholder, 'New task...');
        // The composer's own busy word ends in its own word, whatever stood there.
        r.press('start on Sol');
        await tick();
        assert.deepEqual([r.frames.at(-1).new_work, ...face()], ['token-sol', 'Send', 'Send message']);
    } finally { r.close(); delete globalThis.MutationObserver; }
    const own = observers.filter((made) => made.watched.has(r.send));
    assert.ok(own.length && own.every((made) => !made.targets.length), 'the room\'s end stops the watching');
});

test('in a translated interface the composer still knows its own Send words, shown in that language, from a feature\'s face', async () => {
    // The page's language overlay rewrites fixed words where they stand (i18n.js): what the composer wrote as `Send`
    // is on the button as its translation. Read before the room has a body, so no overlay walks the test's document;
    // `overlay()` below does what it would to the two nodes.
    const RU = { Send: 'Отправить', 'New task': 'Новая задача', 'Send message': 'Отправить сообщение',
        'Message Ouroboros...': 'Напишите Ouroboros...', 'New task...': 'Новая задача...' };
    applyPayload({ language: 'ru', english: false, entries: Object.fromEntries(Object.entries(RU).map(([key, text]) => [key, { text }])) });
    const observers = [];
    globalThis.MutationObserver = class {
        constructor(fn) { this.fn = fn; this.targets = []; observers.push(this); }
        observe(target) { this.targets.push(target); }
        disconnect() { this.targets = []; }
        takeRecords() { return []; }
    };
    const changed = (target) => observers.filter((made) => made.targets.includes(target)).forEach((made) => made.fn([]));
    const r = modelRoom();
    try {
        const overlay = () => {
            if (RU[r.send.textContent]) r.send.textContent = RU[r.send.textContent];
            if (RU[r.send.title]) r.send.title = RU[r.send.title];
            if (RU[r.input.placeholder]) r.input.placeholder = RU[r.input.placeholder];
        };
        const stateRead = () => r.instance.hydrateStateSnapshot(r.state({ kind: 'model', model: SOL, use_local: false }));
        overlay();
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['Отправить', 'Напишите Ouroboros...']);
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        // Its own word, in whatever language it stands, is the composer's to change: the pick says New task.
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['New task', 'New task...']);
        overlay();
        stateRead();
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['Новая задача', 'Новая задача...'],
            'a word already shown is not written again: the composer and the overlay never take turns');
        // A conversation feature's face is not the composer's word in any language: it stands.
        Object.assign(r.send, { textContent: 'Говорить', title: 'Говорить с Ouroboros' });
        changed(r.send);
        stateRead();
        assert.deepEqual([r.send.textContent, r.send.title], ['Говорить', 'Говорить с Ouroboros']);
        // Handed back as a plain Send (the feature's own English word, or its translation once the overlay saw it),
        // the button says the next message is a new task again.
        for (const handed of ['Send', 'Отправить']) {
            Object.assign(r.send, { textContent: 'Говорить', title: 'Говорить с Ouroboros' });
            changed(r.send);
            Object.assign(r.send, { textContent: handed, title: handed === 'Send' ? 'Send message' : 'Отправить сообщение' });
            changed(r.send);
            assert.deepEqual([r.send.textContent, r.send.title], NEW_TASK_ON_SOL, handed);
            overlay();
        }
        // A call's own words in the empty field stand; the ordinary ones given back in the page's language are the
        // composer's again.
        r.input.placeholder = 'Говорите или пишите';
        stateRead();
        assert.equal(r.input.placeholder, 'Говорите или пишите');
        r.input.placeholder = 'Напишите Ouroboros...';
        changed(r.input);
        assert.equal(r.input.placeholder, 'New task...');
    } finally {
        r.close();
        delete globalThis.MutationObserver;
        applyPayload({ language: 'en', english: true, entries: {} });
    }
});

test('another window\'s change of the room\'s model stops the message: the draft stays and nothing is sent on either model', async () => {
    const r = modelRoom();
    try {
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        assert.equal(r.send.textContent, 'New task');
        r.instance.hydrateStateSnapshot(r.state({ kind: 'model', model: KIMI, use_local: false }));  // another window's pick
        assert.equal(r.value(), 'kimi-k3', 'the capsule shows what is saved now');
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['Send', 'Message Ouroboros...'],
            'a pick another window overtook promises no new task on its model');
        r.press('compare the drafts');
        await tick();
        assert.deepEqual([r.frames.length, r.input.value, r.send.textContent], [0, 'compare the drafts', 'Send'],
            'nothing went out, the words stay, and the pick made here no longer stands');
        r.press('compare the drafts');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.new_work]), [['compare the drafts', undefined]],
            'pressed again it is an ordinary message of the room as it is');
        // A pick made after the other window's is a pick of another model again: the room is as that window left it.
        await r.pick(SOL);
        r.saved(1, SOL, 'token-sol-2');
        await tick(); await tick();
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['New task', 'New task...']);
    } finally { r.close(); }
});

// A Reply as the room stages it (chat_reply.js): a row inside a work's card names its task (`bubbleRef`), a card
// its `task:` address (`cardRef`), an ordinary bubble neither, so only the host knows whether its row is a work's.
const WORK_ROW = { address: 'row:7@2026-01-02T03:04:05+00:00#aaaaaaaaaaaa', chat_id: 7, role: 'assistant', kind: 'message',
    author_label: 'Ouroboros', snippet: 'the draft is ready', ts: '2026-01-02T03:04:05+00:00', task_id: 'work-a' };
const WORK_CARD = { address: 'task:work-a', chat_id: 7, role: 'assistant', kind: 'work_card', author_label: 'Draft review',
    snippet: 'reading', ts: '2026-01-02T03:04:05+00:00', task_id: 'work-a' };
const PLAIN_ROW = { address: 'row:7@2026-01-02T03:00:00+00:00#bbbbbbbbbbbb', chat_id: 7, role: 'user', kind: 'message',
    author_label: 'You', snippet: 'my earlier words', ts: '2026-01-02T03:00:00+00:00', task_id: '' };

test('a Reply to a work\'s card is its work\'s like a Reply to a row inside it', async () => {
    const r = modelRoom({ staged: WORK_CARD });
    try {
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['Send', 'Message Ouroboros...']);
        r.instance.hydrateStateSnapshot(r.state({ kind: 'model', model: KIMI, use_local: false }));
        r.press('stop after this file');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.reply_to, frame.new_work]), [['stop after this file', 'task:work-a', undefined]]);
    } finally { r.close(); }
});

test('a Reply to a work\'s row is never held back by a pick that no longer stands, and names no choice', async () => {
    const r = modelRoom({ staged: WORK_ROW });
    try {
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['Send', 'Message Ouroboros...'],
            'replying to a work\'s row, Send says an ordinary message: the same reading the send uses');
        r.instance.hydrateStateSnapshot(r.state({ kind: 'model', model: KIMI, use_local: false }));  // another window's pick
        r.press('about that draft');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.reply_to, frame.new_work]),
            [['about that draft', WORK_ROW.address, undefined]], 'the Reply went to its work at once, naming no choice');
        assert.deepEqual([r.input.value, r.toasts()], ['', []], 'nothing was held back and nothing was said');
        // The Reply left the field with its message; the pick made here still does not stand for plain words.
        r.press('and a plain one');
        await tick();
        assert.deepEqual([r.frames.length, r.input.value], [1, 'and a plain one'], 'the stale pick stops plain words once, the draft stays');
        r.press('and a plain one');
        await tick();
        assert.deepEqual([r.frames.at(-1).content, r.frames.at(-1).new_work], ['and a plain one', undefined]);
    } finally { r.close(); }
});

test('a Reply to an ordinary bubble names the choice and is the host\'s to place: never held back, never called a new task before', async () => {
    const r = modelRoom({ staged: PLAIN_ROW });
    try {
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        // Only the host knows whether the quoted row is a work's: Send promises nothing, the field says both outcomes.
        assert.deepEqual([r.send.textContent, r.send.title, r.input.placeholder],
            ['Send', 'Send the Reply: its work takes it, or it starts a new task on gpt-6.1-sol', 'Reply or new task...']);
        r.press('do this instead');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.reply_to, frame.new_work]),
            [['do this instead', PLAIN_ROW.address, 'token-sol']], 'a plain quote is no Reply to a work: it names the choice like any words');
        // The host gave it to a work as its Reply: the pick still waits for its own message.
        r.echo(1);
        assert.deepEqual([r.send.textContent, r.input.placeholder], ['New task', 'New task...']);
    } finally { r.close(); }
    // Another window changed the room's model: the Reply still goes at once, naming the choice made here, and the
    // host decides (its work takes it; a plain quote is refused and its words come back).
    const changed = modelRoom({ staged: PLAIN_ROW });
    try {
        await changed.pick(SOL);
        changed.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        changed.instance.hydrateStateSnapshot(changed.state({ kind: 'model', model: KIMI, use_local: false }));
        changed.press('do this instead');
        await tick();
        assert.deepEqual(changed.frames.map((frame) => [frame.content, frame.reply_to, frame.new_work]),
            [['do this instead', PLAIN_ROW.address, 'token-sol']]);
        assert.deepEqual([changed.input.value, changed.toasts()], ['', []], 'nothing held it back on the page');
    } finally { changed.close(); }
});

test('refused because the model changed elsewhere, the words return to the field and the next Send is an ordinary one', async () => {
    const r = modelRoom();
    try {
        await r.pick(SOL);
        r.saved(0, SOL, 'token-sol');
        await tick(); await tick();
        r.press('compare the drafts');  // this window has not heard of the other one's pick yet
        await tick();
        assert.deepEqual([r.frames.at(-1).new_work, r.input.value, r.send.textContent], ['token-sol', '', 'Send']);
        r.fire('chat', { type: 'chat', role: 'system', system_type: 'initialization_notice', chat_id: 7, client_message_id: 'cm-1',
            reason_code: 'model_choice_changed', content: 'The model for new work here was changed elsewhere before this message was saved.' });
        await tick();
        assert.deepEqual([r.input.value, r.send.textContent, r.input.placeholder], ['compare the drafts', 'Send', 'Message Ouroboros...'],
            'the words are back in the field, and the pick made here no longer stands');
        assert.equal(r.instance.page.querySelectorAll('.chat-bubble.user').filter((bubble) => bubble.dataset.clientMessageId === 'cm-1').length, 0,
            'its unsaved copy left the feed: the words stand once, in the field');
        r.press('compare the drafts');
        await tick();
        assert.deepEqual(r.frames.map((frame) => [frame.content, frame.new_work]),
            [['compare the drafts', 'token-sol'], ['compare the drafts', undefined]], 'sent again it is an ordinary message, never the refused intent');
        assert.equal(r.input.value, '');
    } finally { r.close(); }
});

test('a room keeps its own draft: setDraft and insertAtCursor keep it, a reopened room restores it, Send clears it', async () => {
    const env = installDom(fetchStub());
    const frames = [];
    let instance = mount(env, { ...PROJECT, frames });
    try {
        const { composer } = instance;
        composer.setDraft('hello world');
        assert.equal(composer.getDraft(), 'hello world');
        assert.equal(ownDraft(7), 'hello world', 'kept under the room\'s own key');
        assert.equal(ownDraft(), null, 'never under Main\'s');
        // A selection is replaced by the inserted words; the caret follows them in the focused field.
        const field = composer.input;
        let caret = null;
        field.setSelectionRange = (start, end) => { caret = [start, end]; };
        field.selectionStart = 6;
        field.selectionEnd = 11;
        composer.insertAtCursor('there');
        assert.equal(composer.getDraft(), 'hello there');
        assert.deepEqual(caret, [11, 11]);
        assert.equal(document.activeElement, field, 'the field has focus for the next word');
        assert.equal(ownDraft(7), 'hello there');
        // A collapsed caret inserts in place, nothing replaced.
        field.selectionStart = 5;
        field.selectionEnd = 5;
        composer.insertAtCursor(',');
        assert.equal(composer.getDraft(), 'hello, there');
        assert.deepEqual(caret, [6, 6]);

        instance.destroy();
        globalThis.sessionStorage = memoryStorage();
        instance = mount(env, { ...PROJECT, frames });
        assert.equal(instance.composer.getDraft(), 'hello, there', 'cold reopen restores its own draft');
        assert.deepEqual(frames, [], 'restoring the draft sends nothing');
        node(instance, 'pchat-p7-send').click();
        await tick();
        assert.equal(frames.length, 1);
        assert.deepEqual([frames[0].content, frames[0].chat_id, frames[0].project_id], ['hello, there', 7, 'p7']);
        assert.equal(instance.composer.getDraft(), '');
        assert.equal(ownDraft(7), null, 'Send clears the kept draft');
        instance.destroy();
        globalThis.sessionStorage = memoryStorage();
        instance = mount(env, { ...PROJECT, frames });
        assert.equal(instance.composer.getDraft(), '', 'sent text does not restore');
    } finally { instance.destroy(); restoreDom(env.prior); }
});

test('the input recall is seeded once from the room\'s own sent words, newest nearest', async () => {
    const row = (id, role, text) => ({ role, text, ts: `2026-10-17T10:0${id}:00.000Z`, history_id: `chat:${id}`,
        history_position: { source: 'chat', offset: id } });
    const env = installDom(fetchStub([row(1, 'user', 'first ask'), row(2, 'assistant', 'an answer'), row(3, 'user', 'second ask')]));
    const instance = mount(env, PROJECT);
    try {
        for (let i = 0; i < 10; i += 1) await tick();
        const field = instance.composer.input;
        field.setSelectionRange = (start, end) => { field.selectionStart = start; field.selectionEnd = end; };
        const press = (key) => { for (const listener of field.listeners.get('keydown')) listener({ key, target: field, preventDefault() {} }); };
        field.selectionStart = 0;
        field.selectionEnd = 0;
        press('ArrowUp');
        assert.equal(field.value, 'second ask');
        field.setSelectionRange(0, 0);
        press('ArrowUp');
        assert.equal(field.value, 'first ask', 'an assistant row is never recalled');
        press('ArrowDown');
        assert.equal(field.value, 'second ask');
        press('ArrowDown');
        assert.equal(field.value, '', 'past the newest: the words being written');
        assert.deepEqual(JSON.parse(sessionStorage.getItem('ouro_chat_input_history')), ['first ask', 'second ask']);
    } finally { instance.destroy(); restoreDom(env.prior); }
});

for (const key of ['ArrowUp', 'ArrowDown']) {
    test(`${key} at the history boundary does not replace another window's saved draft`, () => {
        const env = installDom(fetchStub());
        const firstTab = sessionStorage;
        const first = mount(env, PROJECT);
        let second, reopened;
        try {
            const field = first.composer.input;
            const caret = position => { field.selectionStart = field.selectionEnd = position; };
            field.setSelectionRange = caret;
            const press = key => {
                for (const listener of field.listeners.get('keydown')) listener({ key, target: field, preventDefault() {} });
            };
            first.composer.seedInputHistory([{ role: 'user', text: 'old message' }]);
            first.composer.setDraft('my draft');
            if (key === 'ArrowUp') {
                caret(0);
                press('ArrowUp');
                assert.equal(field.value, 'old message');
            }
            globalThis.sessionStorage = memoryStorage();
            second = mount(env, PROJECT);
            second.composer.setDraft('the other window’s draft');
            globalThis.sessionStorage = firstTab;
            caret(key === 'ArrowUp' ? 0 : field.value.length);
            press(key);
            globalThis.sessionStorage = memoryStorage();
            reopened = mount(env, PROJECT);
            assert.equal(reopened.composer.getDraft(), 'the other window’s draft', key);
        } finally { reopened?.destroy(); second?.destroy(); first.destroy(); restoreDom(env.prior); }
    });
}

function plusFixture() {
    const env = installDomDouble();
    const { doc, el } = env;
    const field = el('textarea');
    const button = el('button', { attrs: { 'aria-label': 'More actions', 'aria-expanded': 'false' } });
    doc.body.append(el('div', { className: 'chat-text-row' }, button, field));
    const chosen = [];
    const menu = createPlusMenu({ button, hostItems: [
        { key: 'attach', label: 'Attach file', icon: 'attach', onSelect: () => chosen.push('attach') },
    ] });
    const sheet = () => doc.body.children.find((child) => child.classList.contains('chat-plus-menu')) || null;
    const items = () => sheet()?.querySelectorAll('[role="menuitem"]') || [];
    return { ...env, field, button, menu, chosen, sheet, items, keys: () => items().map((item) => item.dataset.plusItem),
        key: (target, key) => target.emit('keydown', { key }),
        close() { menu.destroy(); env.restore(); } };
}

test('"+" by keyboard opens the menu with focus on Attach file; arrows move, Esc closes and returns focus', () => {
    const f = plusFixture();
    try {
        // "+" at the composer's left end, at the bottom of the window.
        f.button.rect = { left: 40, right: 76, top: 700, bottom: 736, width: 36, height: 36 };
        f.button.click();  // Enter or Space: a click with no pointer count
        const menu = f.sheet();
        assert.ok(menu, 'the menu is mounted outside the composer');
        assert.deepEqual([menu.css.get('--ui-popup-left'), menu.css.get('--ui-popup-top')], ['40px', '666px'],
            'it opens above "+", lined up with its start');
        assert.equal(menu.getAttribute('role'), 'menu');
        assert.equal(menu.getAttribute('aria-label'), 'More actions');
        assert.equal(menu.classList.contains('ui-popup'), true, 'the shared menu sheet and its viewport placement');
        assert.equal(f.button.getAttribute('aria-expanded'), 'true');
        assert.deepEqual(f.keys(), ['attach']);
        const [attach] = f.items();
        assert.equal(attach.textContent, 'Attach file');
        assert.equal(attach.children[0].getAttribute('aria-hidden'), 'true', 'the icon is decoration');
        assert.equal(f.doc.activeElement, attach);
        f.key(attach, 'Escape');
        assert.equal(f.sheet(), null);
        assert.equal(f.button.getAttribute('aria-expanded'), 'false');
        assert.equal(f.doc.activeElement, f.button, 'focus returns to "+"');
        f.button.click();
        f.items()[0].click();  // Enter or Space on the focused item
        assert.deepEqual(f.chosen, ['attach'], 'Attach file does what the paperclip did');
        assert.equal(f.sheet(), null, 'choosing closes the menu');
    } finally { f.close(); }
});

test('a pointer opens and chooses from "+" without taking the field\'s focus; Esc and an outside press close it', () => {
    const f = plusFixture();
    try {
        f.field.focus();
        assert.equal(f.button.emit('mousedown').defaultPrevented, true, 'pressing "+" keeps the field focused');
        f.button.emit('click', { detail: 1 });
        assert.ok(f.sheet());
        assert.equal(f.doc.activeElement, f.field, 'a phone keeps its keyboard');
        const escape = f.key(f.field, 'Escape');
        assert.equal(escape.stopped, true, 'Esc closes only the menu, not the field\'s own Esc');
        assert.equal(f.sheet(), null);
        assert.equal(f.doc.activeElement, f.field);
        f.button.emit('click', { detail: 1 });
        assert.equal(f.items()[0].emit('mousedown').defaultPrevented, true);
        f.items()[0].emit('click', { detail: 1 });
        assert.deepEqual([f.chosen, f.sheet(), f.doc.activeElement], [['attach'], null, f.field]);
        f.button.emit('click', { detail: 1 });
        f.doc.body.emit('pointerdown');
        assert.equal(f.sheet(), null, 'an outside press closes it');
        f.button.focus();
        f.button.emit('click', { detail: 1 });
        assert.equal(f.doc.activeElement, f.items()[0], 'a press on a focused "+" (a screen reader) moves focus in');
        f.key(f.items()[0], 'Escape');
        f.field.focus();
        f.button.emit('click', { detail: 1 });
        f.button.emit('click', { detail: 1 });
        assert.equal(f.sheet(), null, 'a second press on "+" closes it');
        // With the menu closed, Esc in the field is the field's again.
        assert.notEqual(f.key(f.field, 'Escape').stopped, true);
    } finally { f.close(); }
});

test('later controls add labelled sections after Attach file; a disabled item is skipped and an empty section omitted', () => {
    const f = plusFixture();
    try {
        const picked = [];
        f.menu.setSections([
            { key: 'templates', label: 'Templates', items: [
                { key: 'templates:simpler', label: 'Explain simpler', title: 'Ask for a simpler answer', onSelect: () => picked.push('simpler') },
                { key: 'templates:off', label: 'Unavailable here', disabled: true, onSelect: () => picked.push('off') },
            ] },
            { key: 'empty', label: 'Nothing yet', items: [] },
        ]);
        f.button.click();
        assert.deepEqual(f.keys(), ['attach', 'templates:simpler', 'templates:off']);
        const group = f.sheet().querySelector(':scope > .ui-popup-body').children[1];
        assert.deepEqual([group.getAttribute('role'), group.getAttribute('aria-label'), group.dataset.plusSection],
            ['group', 'Templates', 'templates']);
        assert.equal(group.children[0].textContent, 'Templates');
        assert.equal(f.sheet().querySelector(':scope > .ui-popup-body').children.length, 2, 'the empty section is not drawn');
        const [attach, simpler, off] = f.items();
        assert.equal(simpler.title, 'Ask for a simpler answer');
        assert.equal(off.disabled, true);
        f.key(attach, 'ArrowDown');
        assert.equal(f.doc.activeElement, simpler);
        f.key(simpler, 'ArrowDown');
        assert.equal(f.doc.activeElement, attach, 'the disabled item is skipped');
        f.key(attach, 'End');
        assert.equal(f.doc.activeElement, simpler, 'End reaches the last item that can be chosen');
        simpler.click();
        assert.deepEqual(picked, ['simpler']);
        f.button.click();
        f.menu.setSections([]);
        assert.deepEqual(f.keys(), ['attach', 'templates:simpler', 'templates:off'], 'an open menu keeps its list');
        f.button.click();
        f.button.click();
        assert.deepEqual(f.keys(), ['attach'], 'the next opening shows the new one');
        f.menu.destroy();
        assert.equal(f.sheet(), null, 'destroy removes an open menu');
    } finally { f.close(); }
});

test('refused Send keeps the current text and its cold copy without sending anything', async () => {
    const env = installDom(fetchStub()), frames = [];
    let instance = mount(env, { ...PROJECT, frames, status: 'failed' });
    try {
        instance.composer.setDraft('try again');
        node(instance, 'pchat-p7-send').click();
        await tick();
        assert.equal(instance.composer.getDraft(), 'try again');
        instance.destroy();
        globalThis.sessionStorage = memoryStorage();
        instance = mount(env, { ...PROJECT, frames });
        assert.equal(instance.composer.getDraft(), 'try again');
        assert.deepEqual(frames, []);
    } finally { instance.destroy(); restoreDom(env.prior); }
});

test('storage refusal leaves the new words in the field and does not replace the saved text', () => {
    const env = installDom(fetchStub());
    let instance = mount(env, PROJECT);
    try {
        instance.composer.setDraft('kept');
        localStorage.setItem = () => { throw new Error('storage refused'); };
        instance.composer.setDraft('not kept');
        assert.equal(instance.composer.getDraft(), 'not kept');
        instance.destroy();
        globalThis.sessionStorage = memoryStorage();
        instance = mount(env, PROJECT);
        assert.equal(instance.composer.getDraft(), 'kept');
    } finally { instance.destroy(); restoreDom(env.prior); }
});
