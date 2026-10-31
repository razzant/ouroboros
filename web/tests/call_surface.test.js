import './fixtures/call_modules.js';
import test from 'node:test';
import assert from 'node:assert/strict';
const { callPresentation, createTalkFacts } = await import('../../skills/live_conversation/ui/call_capsule.js');
const { callMemberOf, callSpan, createCallCards } = await import('../../skills/live_conversation/ui/call_card.js');
const { createCallLight } = await import('../../skills/live_conversation/ui/call_light.js');
const { callClock, markedCall } = await import('../../skills/live_conversation/ui/call_mark.js');
const { CALL_VISUAL, waveShape } = await import('../../skills/live_conversation/ui/call_aura.js');

const live = (extra = {}) => ({ state: 'live', code: '', detail: '', action: '', words: '', phase: '',
    muted: false, activity: { working: false, label: '' }, ...extra });
const ended = (code, extra = {}) => live({ state: 'closed', code, ...extra });

test('the live words follow the engine\'s turn and the microphone, never the gaps in the audio', () => {
    const words = (view, facts) => { const p = callPresentation(view, facts); return [p.title, p.sub, p.voice]; };
    // The owner's voice opens the gate before any caption: the words change at once.
    assert.deepEqual(words(live(), { owner: true }), ['Listening', 'You’re speaking', 'owner']);
    // An answer awaited (the owner finished, a tool's result or an addressed fact went in) reads Thinking.
    assert.deepEqual(words(live({ turn: 'awaiting' })), ['Thinking', '', 'owner']);
    assert.deepEqual(words(live({ turn: 'answering' })), ['Thinking', '', 'owner']);  // begun, nothing heard yet
    // Once heard, one answer stays Speaking through its pauses, in Ouroboros's colour.
    assert.deepEqual(words(live({ turn: 'answering' }), { answered: true }), ['Ouroboros', 'Speaking', 'ouroboros']);
    assert.deepEqual(words(live({ turn: 'idle' }), { answered: true }), ['Listening', 'Speak whenever you’re ready', 'owner']);
    // A tool batch ending inside an awaited answer is no Listening flash.
    assert.deepEqual(words(live({ turn: 'awaiting', activity: { working: true, label: '' } })), ['Working', 'Using a tool', 'owner']);
    assert.deepEqual(words(live({ turn: 'awaiting' }), { owner: true }), ['Listening', 'You’re speaking', 'owner']);  // a barge-in
    const facts = createTalkFacts();
    facts.heard(true);
    assert.equal(facts.facts.answered, true);
    facts.turn('answering');
    assert.equal(facts.facts.answered, true);
    facts.turn('awaiting');  // the next answer starts unheard
    assert.equal(facts.facts.answered, false);
});

test('every phase the call reports has its words; nothing claims a connection that is not happening', () => {
    assert.deepEqual(callPresentation({ state: 'idle' }), { mode: 'idle' });
    assert.deepEqual(callPresentation(live({ state: 'ended', code: 'owner_end' })), { mode: 'idle' });  // the owner's End
    const words = (view, facts) => { const p = callPresentation(view, facts); return [p.title, p.sub, p.light]; };
    assert.deepEqual(words(live({ state: 'permission' })), ['Waiting for the microphone', 'Allow it if your browser asks', 'trace']);
    assert.deepEqual(words(live({ state: 'admission' })), ['Starting the call…', '', 'trace']);
    assert.deepEqual(words(live({ state: 'preparing', phase: 'memory' })), ['Preparing memory…', 'Reading this room and my notes', 'trace']);
    assert.deepEqual(words(live({ state: 'preparing' })), ['Preparing the call…', '', 'trace']);
    // One engine: an old server's bare "connecting" and a new one's both read the same.
    assert.deepEqual(words(live({ state: 'connecting' })), ['Connecting…', '', 'trace']);
    assert.deepEqual(words(live({ state: 'connecting', engine: 'google' })), ['Connecting…', '', 'trace']);
    assert.deepEqual(words(live({ state: 'reconnecting', code: 'session_cap' })), ['Reconnecting…', 'Starting a fresh voice session', 'trace']);
    assert.deepEqual(words(live({ state: 'reconnecting', code: 'transport_lost:1006' })), ['Reconnecting…', 'The voice connection dropped', 'trace']);
    assert.deepEqual(words(live({ state: 'closing' })), ['Ending the call…', '', 'trace']);
    assert.deepEqual(words(live()), ['Listening', 'Speak whenever you’re ready', 'listen']);
    assert.deepEqual(words(live(), { ownerSpeaking: true }), ['Listening', 'You’re speaking', 'listen']);
    assert.deepEqual(words(live(), { thinking: true }), ['Thinking', '', 'think']);
    assert.deepEqual(words(live({ activity: { working: true, label: 'Searching the web' } })), ['Working', 'Searching the web', 'think']);
    assert.deepEqual(words(live({ activity: { working: true, label: '' } })), ['Working', 'Using a tool', 'think']);
    assert.deepEqual(words(live(), { heard: true, thinking: true }), ['Ouroboros', 'Speaking', 'listen']);
    assert.deepEqual(words(live({ muted: true })), ['Muted', 'Ouroboros can’t hear you', 'muted']);
    // Whose light may show: the owner's unless Ouroboros is heard (then echo never paints blue).
    assert.equal(callPresentation(live()).voice, 'owner');
    assert.equal(callPresentation(live(), { heard: true }).voice, 'ouroboros');
    assert.equal(callPresentation(live({ muted: true }), { heard: true }).voice, 'ouroboros');
    for (const state of ['permission', 'admission', 'preparing', 'connecting', 'live', 'reconnecting', 'closing'])
        assert.equal(callPresentation(live({ state })).mode, 'live', state);
});

test('each ending says what happened and offers its one next step', () => {
    const show = (view) => { const p = callPresentation(view); return [p.title, p.actions.map(a => a.label), p.tone]; };
    assert.deepEqual(show(ended('owner_bye')), ['Call ended', [], '']);
    assert.deepEqual(show(ended('owner_stop')), ['Call ended', [], '']);
    assert.deepEqual(show(ended('peers_left')), ['Everyone left the call', [], '']);
    assert.deepEqual(show(ended('budget_exceeded')), ['The spending limit is reached', ['Open Settings'], 'error']);
    for (const code of ['missing_credential', 'missing_credential_google'])
        assert.deepEqual(show(ended(code)), ['Calls need the selected engine’s API key — add it in Settings → Secrets', ['Open Settings'], 'error'], code);
    assert.deepEqual(show(ended('authentication')), ['The voice service refused the key', ['Open Settings'], 'error']);
    assert.deepEqual(show(ended('rate_limit', { detail: 'Slow down, please.' })), ['The voice service says: “Slow down, please.”', ['Try again'], 'error']);
    assert.deepEqual(show(ended('rate_limit')), ['The voice service asked to slow down', ['Try again'], 'error']);
    // Google's own sentence is quoted as Google's (a quota close is one of them); without one, plain words.
    assert.deepEqual(show(ended('provider_closed', { detail: 'You exceeded your current quota.' })),
        ['The voice service closed the call: “You exceeded your current quota.”', ['Try again'], 'error']);
    assert.deepEqual(show(ended('provider_closed')), ['The voice service closed the call', ['Try again'], 'error']);
    assert.deepEqual(show(ended('engine_unavailable')), ['Can’t reach the selected voice service', ['Try again'], 'error']);
    assert.deepEqual(show(ended('reconnect_failed')), ['Can’t reach the selected voice service', ['Try again'], 'error']);
    assert.deepEqual(show(ended('preparation_failed')), ['The call could not start; the room has the details', ['Try again'], 'error']);
    assert.deepEqual(show(ended('call_not_started')), ['The call could not start; the room has the details', ['Try again'], 'error']);
    for (const code of ['invalid_request', 'protocol', 'provider_error']) {
        const again = code === 'invalid_request' ? [] : ['Try again'];  // a refused setup would only repeat
        assert.deepEqual(show(ended(code, { detail: 'Bad config.' })), ['The voice service refused the call: “Bad config.”', again, 'error'], code);
        assert.deepEqual(show(ended(code)), ['The voice service refused the call', again, 'error'], code);
    }
    // The host's own refusals are the host's words, never framed as Google's.
    assert.deepEqual(show(ended('engine_refused', { detail: 'A live route requires the provider’s model ID.' })),
        ['The call could not start: A live route requires the provider’s model ID.', ['Open Settings'], 'error']);
    assert.deepEqual(show(ended('backpressure')), ['The call fell behind the voice stream', ['Try again'], 'error']);
    // The server's own action wins over the page's table; an action this page does not know offers nothing.
    assert.deepEqual(show(ended('engine_unavailable', { action: 'settings' })), ['Can’t reach the selected voice service', ['Open Settings'], 'error']);
    assert.deepEqual(show(ended('provider_closed', { action: 'other_engine' })), ['The voice service closed the call', [], 'error']);
    // Settings opens where the step is: the key among the skill-requested Secrets, the model in the extension settings.
    const target = (code) => callPresentation(ended(code)).actions.map(a => [a.tab, a.focus]);
    assert.deepEqual(target('missing_credential'), [['secrets', 'skill-requested-secrets']]);
    assert.deepEqual(target('authentication'), [['secrets', 'skill-requested-secrets']]);
    assert.deepEqual(target('engine_refused'), [['advanced', 'extension-settings-sections']]);
    assert.deepEqual(target('budget_exceeded'), [['behavior', '']]);
    // Nothing about another engine is left in the words.
    for (const code of Object.keys({ missing_credential: 1, authentication: 1, rate_limit: 1, engine_unavailable: 1, invalid_request: 1 }))
        assert.doesNotMatch(callPresentation(ended(code, { detail: 'x' })).title, /OpenAI/);
});

test('a reasonless or unknown ending still shows the capsule: the call ended, and what the room has', () => {
    for (const code of ['', 'finished']) {
        const p = callPresentation(ended(code));
        assert.deepEqual([p.mode, p.title, p.sub, p.small, p.actions], ['sheet', 'The call ended', 'The room has its transcript', '', []]);
    }
    const unknown = callPresentation(ended('a_code_from_the_future'));
    assert.deepEqual([unknown.mode, unknown.title, unknown.small], ['sheet', 'The call ended', '']);
    // Ending codes stay internal; the owner sees words.
    for (const code of ['owner_bye', 'max_minutes', 'peers_left', 'not_connected', 'budget_exceeded', 'missing_credential', 'authentication',
        'rate_limit', 'provider_closed', 'engine_unavailable', 'reconnect_failed',
        'preparation_failed', 'call_not_started', 'invalid_request', 'protocol', 'backpressure', 'provider_error',
        'engine_refused', 'socket_closed', 'socket_silent', 'microphone_busy', 'permission_denied', 'old_app']) {
        const p = callPresentation(ended(code));
        assert.doesNotMatch(p.title, /_/, code);
        assert.equal(p.small, '', code);
    }
});

test('what this page finds itself, and surfaces that cannot hold a call, get words and the step that helps', () => {
    const failed = (code, extra) => callPresentation(live({ state: 'failed', code, ...extra }));
    const denied = failed('permission_denied');
    assert.deepEqual([denied.mode, denied.tone, denied.light], ['sheet', 'notice', 'still']);  // words, not an alarm
    assert.deepEqual([failed('socket_silent').tone, failed('socket_silent').light], ['error', 'error']);
    assert.match(denied.title, /^Allow the microphone for Ouroboros in System Settings → Privacy & Security → Microphone \(or in the browser’s site settings\), then press Talk again\.$/);
    const old = failed('old_app');
    assert.match(old.title, /^This version of the Ouroboros app can’t use the microphone\. Talk works in your browser, or after you install the new app\.$/);
    assert.deepEqual(old.actions.map(a => [a.kind, a.label]), [['open_browser', 'Open in browser'], ['get_app', 'Get the new app']]);
    assert.deepEqual(failed('microphone_busy').actions.map(a => [a.kind, a.label]), [['reveal', 'Go to the call']]);
    assert.deepEqual([failed('socket_closed', { detail: 'Policy violation' }).title, failed('socket_closed').title],
        ['The call connection closed: “Policy violation”', 'The call ended unexpectedly']);
    assert.equal(failed('socket_silent').title, 'The call service did not answer');
    assert.match(failed('insecure').title, /HTTPS/);
    // An admission refusal keeps the server's words; the page adds the next step for its code.
    const refusal = failed('call_model_missing', { words: 'No live model is set for this engine in Settings (Models, Live calls).' });
    assert.deepEqual([refusal.title, refusal.actions.map(a => [a.kind, a.tab, a.focus])],
        ['No live model is set for this engine in Settings (Models, Live calls).', [['settings', 'advanced', 'extension-settings-sections']]]);
    const noKey = failed('missing_credential_google', { words: 'Calls need a selected engine’s API key — add it in Settings' });
    assert.deepEqual([noKey.title, noKey.actions.map(a => [a.kind, a.focus])],
        ['Calls need a selected engine’s API key — add it in Settings', [['settings', 'skill-requested-secrets']]]);
    assert.deepEqual(failed('repo_writer_gate_closed', { words: 'Ouroboros is applying an update; try the call again in a moment.' })
        .actions.map(a => a.kind), ['retry']);
});

test('thinking runs from the owner\'s final words to the first heard answer, on either engine\'s order', () => {
    // Gemini: the owner's words stream, end, then the answer is named and heard.
    const gemini = createTalkFacts();
    gemini.caption({ speaker: 'owner', text: 'Hello', final: false });
    assert.deepEqual(gemini.facts, { ownerSpeaking: true, thinking: false, answered: false });
    gemini.caption({ speaker: 'owner', text: 'Hello there', final: true });
    assert.deepEqual(gemini.facts, { ownerSpeaking: false, thinking: true, answered: false });
    gemini.caption({ speaker: 'ouroboros', text: '', final: false });  // the answer is named before its audio
    assert.equal(gemini.facts.thinking, true);
    gemini.heard(true);
    assert.deepEqual(gemini.facts, { ownerSpeaking: false, thinking: false, answered: true });
    // The owner's final words can land after the answer began (Gemini's, observed live): no thinking after it.
    const late = createTalkFacts();
    late.voice();
    late.caption({ speaker: 'ouroboros', text: '', final: false });
    late.caption({ speaker: 'ouroboros', text: 'Forty-two.', final: true });
    late.caption({ speaker: 'owner', text: 'What is six times seven?', final: true });
    assert.equal(late.facts.thinking, false);
    // The next question starts a new answer.
    late.voice();
    late.caption({ speaker: 'owner', text: 'And eight times nine?', final: true });
    assert.equal(late.facts.thinking, true);
    // Words heard over the answer, or no words at all, are no question.
    const heardNow = createTalkFacts();
    heardNow.caption({ speaker: 'owner', text: 'mm', final: true }, true);
    assert.equal(heardNow.facts.thinking, false);
    const empty = createTalkFacts();
    empty.caption({ speaker: 'owner', text: '  ', final: true });
    assert.equal(empty.facts.thinking, false);
    // An answer that was never heard (text only) still ends thinking when it is complete.
    const silent = createTalkFacts();
    silent.caption({ speaker: 'owner', text: 'Hi', final: true });
    silent.caption({ speaker: 'ouroboros', text: 'Hi.', final: true });
    assert.equal(silent.facts.thinking, false);
});

class FakeNode {
    constructor(tag, doc) {
        Object.assign(this, { tagName: tag.toUpperCase(), ownerDocument: doc, children: [], parentNode: null,
            dataset: {}, className: '', ownText: '', attributes: {}, listeners: {}, offsetTop: 0, offsetHeight: 0 });
    }
    get classList() {
        const names = () => this.className.split(/\s+/).filter(Boolean);
        return {
            contains: name => names().includes(name),
            add: name => { if (!names().includes(name)) this.className = [...names(), name].join(' '); },
            remove: name => { this.className = names().filter(other => other !== name).join(' '); },
        };
    }
    get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
    set textContent(value) { this.children = []; this.ownText = String(value); }
    get isConnected() { let node = this; while (node.parentNode) node = node.parentNode; return node === this.ownerDocument.feed; }
    get lastChild() { return this.children.at(-1) || null; }
    get previousElementSibling() { const kids = this.parentNode?.children || []; return kids[kids.indexOf(this) - 1] || null; }
    append(...nodes) { for (const node of nodes) { node.remove(); node.parentNode = this; this.children.push(node); } }
    replaceChildren(...nodes) { for (const child of [...this.children]) child.remove(); this.append(...nodes); }
    insertBefore(node, before) { node.remove(); node.parentNode = this; this.children.splice(this.children.indexOf(before), 0, node); }
    remove() { if (!this.parentNode) return; this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); this.parentNode = null; }
    querySelector(selector) {
        const name = selector.replace(/^\./, '');
        for (const child of this.children) {
            if (child.classList.contains(name)) return child;
            const deeper = child.querySelector(selector);
            if (deeper) return deeper;
        }
        return null;
    }
    setAttribute(name, value) { this.attributes[name] = String(value); }
    addEventListener(type, fn) { this.listeners[type] = fn; }
    click() { this.listeners.click?.(); }
}

function feedFixture({ live = () => 'c1' } = {}) {
    const doc = { createElement: tag => new FakeNode(tag, doc) };
    const feed = new FakeNode('div', doc);
    doc.feed = feed;
    Object.assign(feed, { scrollTop: 0, scrollHeight: 1000, clientHeight: 400 });
    const listeners = {};
    const ws = { on(type, fn) { (listeners[type] ||= new Set()).add(fn); return () => listeners[type].delete(fn); },
        emit(type, data) { for (const fn of listeners[type] || []) fn(data); }, listeners };
    let liveCall = live;
    const cards = createCallCards({ feed, ws, doc, live: () => liveCall() });
    const card = () => feed.children.find(node => node.className === 'call-card');
    const lines = () => card()?.querySelector('.call-card-lines').children.map(node => [node.className, node.textContent]) || [];
    // A room row as the chat mounts it: a bubble with its identity stamped.
    const row = (role, text, data = {}, ts = 0) => {
        const node = new FakeNode('div', doc);
        node.className = `chat-bubble ${role}`;
        const sender = new FakeNode('div', doc);
        sender.className = 'sender';
        sender.textContent = role === 'user' ? 'voice' : 'Ouroboros';
        const message = new FakeNode('div', doc);
        message.className = 'message';
        message.textContent = text;
        node.append(sender, message);
        Object.assign(node.dataset, data, ts ? { ts: String(ts) } : {});
        feed.append(node);
        return node;
    };
    return { doc, feed, ws, cards, card, lines, row, setLive: fn => { liveCall = fn; } };
}

test('a call is one card of the room: its rows fold into it, its captions stream at its end', () => {
    const { feed, ws, cards, card, lines, row, setLive } = feedFixture();
    const typed = row('user', 'Typed before the call', { clientMessageId: 'web-1', taskId: 't0' });
    const typing = new FakeNode('div', feed.ownerDocument);
    typing.className = 'chat-bubble assistant typing-bubble';
    feed.append(typing);
    // The owner speaks before any row of the call exists: the card appears at the room's end, live and
    // folded (the transcript is kept, never the call's focus); a click opens it.
    cards.update({ call_id: 'c1', speaker: 'owner', item_id: 'o1', text: 'Hello', final: false });
    cards.update({ call_id: 'c1', speaker: 'owner', item_id: 'o1', text: ', Ouroboros', final: false });
    assert.equal(feed.children.at(-1), typing);  // the typing indicator stays last
    assert.deepEqual([card().dataset.callCard, card().dataset.callOpen], ['live', '0']);
    card().querySelector('.call-card-head').click();
    assert.equal(card().dataset.callOpen, '1');
    card().querySelector('.call-card-head').click();
    assert.equal(card().dataset.callOpen, '0');
    assert.deepEqual(lines(), [['call-line owner call-caption', 'YouHello, Ouroboros']]);
    // The chat paints the owner's row (it listens first): the caption leaves, the row folds in as a line.
    const owner = row('user', 'Hello, Ouroboros!', { clientMessageId: 'call:c1:o1', taskId: 'call-task' }, 1000);
    ws.emit('chat', { type: 'chat', role: 'user', content: 'Hello, Ouroboros!', client_message_id: 'call:c1:o1' });
    cards.scan();
    assert.ok(owner.classList.contains('call-folded') && !typed.classList.contains('call-folded'));
    assert.deepEqual(lines(), [['call-line owner', 'YouHello, Ouroboros!']]);
    assert.equal(card().querySelector('.call-card-facts').textContent, 'Live · 1 line');
    cards.update({ call_id: 'c1', speaker: 'owner', item_id: 'o1', text: 'Hello, Ouroboros!', final: true });
    assert.equal(cards.count(), 0);  // a late caption of a settled item stays out
    // Ouroboros's spoken narration of the same task folds in; a link it sent stays a room message.
    cards.update({ call_id: 'c1', speaker: 'ouroboros', item_id: 'local:turn:3', text: 'Forty', final: false });
    assert.equal(lines().at(-1)[1], 'OuroborosForty');
    const narration = row('assistant', '💬 Forty-two.', { systemType: 'model_narration', taskId: 'call-task' }, 61000);
    ws.emit('chat', { type: 'chat', role: 'assistant', spoken: { call_id: 'c1', item_id: 'local:turn:3' } });
    const link = row('assistant', 'Here is the link: https://example.com', { taskId: 'call-task' }, 62000);
    const work = new FakeNode('div', feed.ownerDocument);
    work.className = 'chat-live-card';
    work.dataset.taskId = 'call-task';
    feed.append(work);
    const elsewhere = row('assistant', '💬 A text turn narrates.', { systemType: 'model_narration', taskId: 'other' });
    cards.scan();
    assert.ok(narration.classList.contains('call-folded') && work.classList.contains('call-folded'));
    assert.ok(!link.classList.contains('call-folded') && !elsewhere.classList.contains('call-folded'));
    assert.deepEqual(lines(), [['call-line owner', 'YouHello, Ouroboros!'], ['call-line ouroboros', 'OuroborosForty-two.']]);
    assert.equal(feed.children.indexOf(card()), feed.children.indexOf(owner) - 1);  // the card stands where the call began
    // The call ends: its closing notice reads under the words, and the card folds to one line.
    row('system', 'Live call c1 ended (owner_bye).', { taskId: 'call-task' }, 63000);
    setLive(() => '');
    cards.finish();
    cards.scan();
    assert.deepEqual(lines().at(-1), ['call-line notice', 'Live call c1 ended (owner_bye).']);
    assert.deepEqual([card().dataset.callCard, card().dataset.callOpen], ['ended', '0']);
    assert.equal(card().querySelector('.call-card-facts').textContent, '1 min 2 s · 2 lines');  // to the closing notice
    card().querySelector('.call-card-head').click();
    assert.equal(card().dataset.callOpen, '1');
    cards.destroy();
    assert.ok(!card() && !owner.classList.contains('call-folded'));
    assert.equal(ws.listeners.chat.size + ws.listeners.open.size, 0);
});

test('a call already in the room\'s history is one closed card, and an older page of it joins the same card', () => {
    const { feed, card, lines, row, doc } = feedFixture({ live: () => '' });
    row('user', 'Earlier typed words', { clientMessageId: 'web-2', taskId: 't1' });
    const first = row('user', 'One', { clientMessageId: 'call:c7:a', taskId: 'call-7' }, 5000);
    row('assistant', '💬 Two', { systemType: 'model_narration', taskId: 'call-7' }, 50000);
    const cards = createCallCards({ feed, doc, live: () => '' });
    assert.deepEqual([card().dataset.callCard, card().dataset.callOpen], ['ended', '0']);
    assert.equal(feed.children.indexOf(card()), feed.children.indexOf(first) - 1);
    // Paging up mounts an older row of the same call above: the card moves to stand before it.
    const older = new FakeNode('div', doc);
    older.className = 'chat-bubble user';
    const message = new FakeNode('div', doc);
    message.className = 'message';
    message.textContent = 'Zero';
    older.append(message);
    Object.assign(older.dataset, { clientMessageId: 'call:c7:z', taskId: 'call-7', ts: '1000' });
    feed.insertBefore(older, feed.children[0]);
    cards.scan();
    assert.equal(feed.children.indexOf(card()), 0);
    assert.deepEqual(lines().map(([, text]) => text), ['YouZero', 'YouOne', 'OuroborosTwo']);
    assert.equal(card().querySelector('.call-card-facts').textContent, '49 s · 3 lines');
    cards.destroy();
});

test('what a room row is to a call, and how a call\'s span reads', () => {
    const calls = new Map([['call-task', 'c1']]);
    const node = (className, data) => ({ classList: { contains: name => className.split(' ').includes(name) }, dataset: data });
    assert.deepEqual(callMemberOf(node('chat-bubble user', { clientMessageId: 'call:c1:x', taskId: 'call-task' }), new Map()),
        { task: 'call-task', kind: 'owner', call: 'c1' });
    assert.equal(callMemberOf(node('chat-bubble assistant', { systemType: 'model_narration', taskId: 'call-task' }), new Map()), null);
    assert.deepEqual(callMemberOf(node('chat-bubble assistant', { systemType: 'model_narration', taskId: 'call-task' }), calls),
        { task: 'call-task', kind: 'ouroboros' });
    assert.equal(callMemberOf(node('chat-bubble user', { clientMessageId: 'web-1', taskId: 'call-task' }), calls), null);
    assert.equal(callMemberOf(node('chat-bubble assistant', { taskId: 'call-task' }), calls), null);  // a link it sent
    assert.deepEqual([callSpan(45_000), callSpan(60_000), callSpan(210_400)], ['45 s', '1 min', '3 min 30 s']);
});

test('a caption follows the newest message only for a reader who is there', () => {
    const { feed, cards } = feedFixture();
    feed.scrollTop = 600;  // at the bottom (1000 - 600 - 400 = 0)
    cards.update({ call_id: 'c1', speaker: 'owner', item_id: '1', text: 'Hi', final: false });
    feed.scrollHeight = 1100;
    cards.update({ call_id: 'c1', speaker: 'owner', item_id: '1', text: ' there', final: false });
    assert.equal(feed.scrollTop, 1100);
    feed.scrollTop = 100;  // reading older messages
    feed.scrollHeight = 1200;
    cards.update({ call_id: 'c1', speaker: 'owner', item_id: '1', text: ' again', final: false });
    assert.equal(feed.scrollTop, 100);
});

function lightFixture({ reduce = false, shown = true } = {}) {
    let time = 0;
    const frames = [];
    const props = {};
    const heard = [];
    let voices = 0;
    let visible = shown;
    const env = {
        performance: { now: () => time },
        requestAnimationFrame: fn => { frames.push(fn); return frames.length; },
        cancelAnimationFrame: () => {},
        matchMedia: () => ({ matches: reduce }),
        document: { visibilityState: 'visible', addEventListener() {}, removeEventListener() {} },
    };
    const light = createCallLight({ env, visible: () => visible, heard: now => heard.push(now), voice: () => { voices += 1; } });
    light.attach({ style: { setProperty: (name, value) => { props[name] = Number(value); } } });
    light.setActive(true);
    const step = (ms = 16) => { time += ms; const run = frames.splice(0); for (const fn of run) fn(time); return run.length; };
    return { light, step, frames, props, heard, voices: () => voices, show: (value) => { visible = value; }, advance: ms => { time += ms; } };
}

test('the light answers the microphone at once and playback only when it leaves the speaker', () => {
    const f = lightFixture();
    f.light.level({ mic: 0.8, out: 0.7, playing: true, latency: 0.1 });
    assert.equal(f.voices(), 0);  // one loud moment moves the wave; the owner's voice is a state that holds (below)
    f.step(16);
    assert.ok(f.props['--call-mic'] > 0.2, f.props['--call-mic']);  // the first frame already moves
    assert.equal(f.props['--call-out'], 0);
    assert.deepEqual(f.heard, []);
    f.step(50);
    assert.equal(f.props['--call-out'], 0);  // 66 ms: still inside the 100 ms output latency
    f.step(50);
    assert.deepEqual(f.heard, [true]);  // heard once it is due
    f.step(16);
    assert.ok(f.props['--call-out'] > 0.2);
    // Silence: the envelope releases slowly, then the loop stops instead of spinning.
    f.light.level({ mic: 0, out: 0, playing: false, latency: 0 });
    let frames = 0;
    for (let i = 0; i < 200 && f.step(16); i++) frames += 1;
    assert.ok(frames > 10 && frames < 200, frames);
    assert.deepEqual([f.props['--call-mic'], f.props['--call-out']], [0, 0]);
    assert.equal(f.frames.length, 0);
    assert.deepEqual(f.heard, [true, false]);
    // Echo while Ouroboros is heard is not the owner's voice, however long it lasts.
    const echo = lightFixture();
    echo.light.level({ mic: 0, out: 0.7, playing: true, latency: 0 });
    echo.step(16);
    for (let t = 0; t < 600; t += 20) { echo.light.level({ mic: 0.5, out: 0.7, playing: true, latency: 0 }); echo.step(20); }
    assert.equal(echo.voices(), 0);
});

test('the owner\'s voice is a held state: a clatter or a pause changes nothing, a steady room stops counting', () => {
    const f = lightFixture();
    const feed = (mic, ms) => { for (let t = 0; t < ms; t += 20) { f.advance(20); f.light.level({ mic, out: 0, playing: false, latency: 0 }); } };
    feed(0, 400);
    feed(0.8, 100);  // a clatter moves the wave, but no voice begins
    assert.equal(f.light.speaking(), false);
    feed(0, 400);
    feed(0.6, 400);  // a voice that holds a quarter of a second begins
    assert.equal(f.light.speaking(), true);
    assert.equal(f.voices(), 1);
    feed(0, 500);  // a pause between words keeps it
    assert.equal(f.light.speaking(), true);
    feed(0, 600);  // most of a second of quiet ends it
    assert.equal(f.light.speaking(), false);
    feed(0.3, 8000);  // a steady background: within a few seconds the gate rises above it
    assert.equal(f.light.speaking(), false);
    feed(0.7, 400);  // and a voice over the background still begins
    assert.equal(f.light.speaking(), true);
});

test('reduced motion shows whose voice in calm steps; an off-screen room runs no frames', () => {
    const calm = lightFixture({ reduce: true });
    calm.light.level({ mic: 0.9, out: 0, playing: false, latency: 0 });
    calm.step(16);
    assert.equal(calm.props['--call-mic'], 0.6);
    calm.light.level({ mic: 0.3, out: 0, playing: false, latency: 0 });
    calm.step(16);
    assert.equal(calm.props['--call-mic'], 0.6);  // no swell with loudness
    calm.light.level({ mic: 0, out: 0, playing: false, latency: 0 });
    calm.step(16);
    assert.equal(calm.props['--call-mic'], 0);
    assert.equal(calm.step(16), 0);
    const hidden = lightFixture({ shown: false });
    hidden.light.level({ mic: 0.9, out: 0.9, playing: true, latency: 0 });
    assert.equal(hidden.frames.length, 0);
    assert.deepEqual(hidden.heard, [true]);  // the words and the mark still know who is heard
    hidden.show(true);
    hidden.light.wake();
    assert.equal(hidden.frames.length, 1);
    // The call ends: stillness, nothing heard.
    hidden.light.setActive(false);
    assert.deepEqual([hidden.props['--call-mic'], hidden.props['--call-out']], [0, 0]);
    assert.deepEqual(hidden.heard, [true, false]);
});

test('the light moves by itself only in moving phases, and an ended call draws its still state once', () => {
    const drawn = [];
    let time = 0;
    const frames = [];
    const env = { performance: { now: () => time }, requestAnimationFrame: fn => { frames.push(fn); return frames.length; },
        cancelAnimationFrame() {}, matchMedia: () => ({ matches: false }),
        document: { visibilityState: 'visible', addEventListener() {}, removeEventListener() {} } };
    const light = createCallLight({ env, draw: state => drawn.push(state) });
    light.attach({ style: { setProperty() {} } });
    const step = () => { time += 16; const run = frames.splice(0); for (const fn of run) fn(time); return run.length; };
    light.setActive(true);
    light.setMode('listen', 'owner');
    for (let i = 0; i < 5; i++) assert.equal(step(), 1);  // a gentle wave while listening, with no sound at all
    assert.deepEqual([drawn.at(-1).mode, drawn.at(-1).voice], ['listen', 'owner']);
    light.setMode('muted', 'owner');
    assert.equal(step(), 1);
    assert.equal(step(), 0);  // muted and silent: still, the loop stops
    light.setActive(false);
    const before = drawn.length;
    light.setMode('error', '');
    assert.equal(drawn.length, before + 1);  // drawn at once, without a frame loop
    assert.deepEqual([drawn.at(-1).mode, drawn.at(-1).mic, drawn.at(-1).out], ['error', 0, 0]);
    assert.equal(frames.length, 0);
});

test('the mark stands for the live call only while its room is off screen, with the call\'s own time', () => {
    const entry = (active, onScreen) => ({ session: { active }, onScreen: () => onScreen });
    assert.equal(markedCall(new Set()), null);
    assert.equal(markedCall(new Set([entry(false, false)])), null);
    assert.equal(markedCall(new Set([entry(true, true)])), null);
    const away = entry(true, false);
    assert.equal(markedCall(new Set([entry(false, true), away])), away);
    assert.equal(callClock(0, 5000), '');
    assert.equal(callClock(1000, 1000), '00:00');
    assert.equal(callClock(1000, 1000 + 192_000), '03:12');
    assert.equal(callClock(0 + 1, 1 + 3_725_000), '1:02:05');
});

test('the visual is one swappable contract: the wave in its slot, the glass behind the row, a small sign', () => {
    const made = [];
    const node = (tag) => {
        const element = { tag, className: '', attrs: {}, innerHTML: '', removed: false, props: {}, dataset: {}, children: [],
            clientWidth: 600, clientHeight: 50,
            style: { setProperty(name, value) { element.props[name] = value; }, removeProperty(name) { delete element.props[name]; } },
            setAttribute(name, value) { this.attrs[name] = value; }, remove() { this.removed = true; },
            getBoundingClientRect: () => ({ left: 0, width: 600 }) };
        made.push(element);
        return element;
    };
    const fills = [];
    const dashes = [];
    const doc = { createElement: tag => {
        const element = node(tag);
        if (tag === 'canvas') element.getContext = () => ({ setTransform() {}, clearRect() {}, beginPath() {}, moveTo() {},
            lineTo() {}, closePath() {}, arc() {}, fill() { fills.push(this.fillStyle); }, stroke() { dashes.push(this.strokeStyle); },
            setLineDash(pattern) { if (pattern.length) dashes.push('dash'); } });
        return element;
    } };
    const slot = { children: [], dataset: {}, clientWidth: 600, clientHeight: 50, getBoundingClientRect: () => ({ left: 0, width: 600 }),
        replaceChildren(...nodes) { this.children = nodes; } };
    const row = { appended: [], props: {}, append(child) { this.appended.push(child); }, querySelector: () => null,
        style: { setProperty(name, value) { row.props[name] = value; }, removeProperty(name) { delete row.props[name]; } } };
    const wrap = { dataset: { call: 'live', callView: 'voice' } };
    let clock = 0;
    const colours = { '--call-owner-rgb': '47, 107, 214', '--call-ouro-rgb': '201, 53, 69', '--call-amber-rgb': '180, 83, 9',
        '--neutral-rgb': '28, 33, 51' };
    const env = { devicePixelRatio: 3, getComputedStyle: () => ({ getPropertyValue: name => colours[name] || '' }),
        performance: { now: () => clock }, addEventListener() {}, removeEventListener() {} };
    const handle = CALL_VISUAL.mount({ slot, row, wrap, doc, env });
    const [canvas, work] = slot.children;
    assert.deepEqual([canvas.className, canvas.attrs['aria-hidden'], canvas.width, canvas.height], ['call-wave', 'true', 1200, 100]);
    assert.equal(work.className, 'call-work chat-live-typing');
    const [glass] = row.appended;  // the row's glass: its surface while live, tinted by the speaker's hue
    assert.equal(glass.className, 'call-glass');
    assert.match(glass.innerHTML, /call-glass-owner.*call-glass-ouroboros/);
    const run = (state, seconds) => { for (let t = 0; t < seconds; t += 1 / 60) { clock += 1000 / 60; handle.draw(state); } };
    const tints = () => [Number(row.props['--call-tint-owner']), Number(row.props['--call-tint-ouroboros'])];
    // The owner speaks: blue ink, a blue tint, the outline breathing out a few pixels.
    run({ mic: 0.8, out: 0, mode: 'listen', voice: 'owner', owner: true }, 2);  // the hue settles within about two seconds
    assert.ok(tints()[0] > 0.5 && tints()[1] === 0, String(tints()));
    assert.ok(Number(row.props['--call-breath']) > 1 && Number(row.props['--call-breath']) < 8, row.props['--call-breath']);
    const rgb = text => text.match(/[\d.]+/g).map(Number);
    fills.length = 0;
    run({ mic: 0.8, out: 0, mode: 'listen', voice: 'owner', owner: true }, 1 / 60);  // one settled frame
    const strongest = fills.map(rgb).filter(([, , , alpha]) => alpha > 0.9);
    assert.ok(strongest.length && strongest.every(([r, g, b]) => Math.abs(r - 47) <= 8 && Math.abs(g - 107) <= 8 && Math.abs(b - 214) <= 8),
        'the main strand in the owner\'s blue');
    // The turn passes: a slow direct cross-fade, blue out while crimson comes in, so both are lit for about
    // a second (a full-duplex room never flicks between them) and crimson holds the glass within a second.
    let both = 0;
    let arrived = Infinity;
    for (let t = 0; t < 3; t += 1 / 60) {
        clock += 1000 / 60;
        handle.draw({ mic: 0, out: 0.8, mode: 'listen', voice: 'ouroboros' });
        const [owner, ouroboros] = tints();
        if (owner > 0.1 && ouroboros > 0.1) both += 1;
        if (arrived === Infinity && ouroboros > 0.5) arrived = t;
    }
    assert.ok(both > 30 && both < 90, String([both, arrived]));
    assert.ok(arrived > 0.2 && arrived < 1, String([both, arrived]));
    assert.ok(tints()[1] > 0.5 && tints()[0] < 0.05, String(tints()));
    // Muted: one dashed line; an error: one still amber strand; reduced motion holds the outline still.
    fills.length = 0;
    run({ mic: 0.9, out: 0, mode: 'muted', voice: 'owner' }, 0.5);
    assert.ok(dashes.includes('dash'));
    run({ mode: 'error' }, 0.3);
    assert.ok(fills.at(-1).startsWith('rgba(180,83,9,'), fills.at(-1));
    run({ mic: 0.9, out: 0, mode: 'listen', voice: 'owner', owner: true, reduced: true }, 0.3);
    assert.equal(row.props['--call-breath'], '0.00');
    // Live work draws the strands in to the room's three dots, and lets them go again.
    wrap.dataset.callWork = '1';
    run({ mode: 'think', voice: '' }, 1);
    assert.equal(slot.dataset.work, '1');
    wrap.dataset.callWork = '';
    run({ mode: 'think', voice: '' }, 1);
    assert.equal(slot.dataset.work, '');
    const mark = CALL_VISUAL.mark(doc);
    assert.equal(mark.className, 'call-entry-mark');
    assert.equal((mark.innerHTML.match(/<path /g) || []).length, 3);  // the wave's three strands, still
    handle.destroy();
    assert.equal(slot.children.length, 0);
    assert.equal(glass.removed, true);
    assert.equal(row.props['--call-breath'], undefined);
    assert.throws(() => { CALL_VISUAL.name = 'lens'; }, TypeError);
});

test('the wave: whose colour from the state, its height from the voice, a pulse while preparing or thinking', () => {
    const shape = (state) => { const { colour, amp, pulse } = waveShape(state); return [colour, Number(amp.toFixed(2)), pulse]; };
    assert.deepEqual(shape({ mode: 'listen', time: 0 }), ['idle', 0.07, false]);  // a gentle low wave while listening
    // The wave reads a voice as if its level filled at -18 dBFS (owner, 18.10): 0.7 swings as 0.77.
    assert.deepEqual(shape({ mode: 'listen', mic: 0.7, owner: true }), ['owner', 0.77, false]);
    assert.deepEqual(shape({ mode: 'listen', mic: 0.95, owner: true }), ['owner', 1, false]);  // loud syllables reach the top
    assert.deepEqual(shape({ mode: 'listen', mic: 0.7, out: 0.6, voice: 'ouroboros' }), ['ouroboros', 0.66, false]);  // no echo
    // Whose colour is a held state, never the meter: a noisy room moves the wave in the room's ink, and
    // Ouroboros keeps its colour through a pause in its answer.
    assert.deepEqual(shape({ mode: 'listen', mic: 0.5 }), ['idle', 0.55, false]);
    assert.deepEqual(shape({ mode: 'listen', out: 0, voice: 'ouroboros' }), ['ouroboros', 0.07, false]);
    assert.deepEqual(shape({ mode: 'trace' }), ['idle', 0, true]);
    assert.deepEqual(shape({ mode: 'think' }), ['ouroboros', 0, true]);
    assert.deepEqual(shape({ mode: 'error', mic: 0.9 }), ['error', 0, false]);  // still, in amber
    assert.deepEqual(shape({ mode: 'muted', mic: 0.9, owner: true }), ['idle', 0, false]);
    assert.deepEqual(shape({ mode: 'muted', out: 0.5, voice: 'ouroboros' }), ['ouroboros', 0.55, false]);
    assert.deepEqual(shape({ mode: 'listen', mic: 0.7, owner: true, reduced: true }), ['owner', 0, false]);
    assert.deepEqual(shape({ mode: 'think', reduced: true }), ['ouroboros', 0, false]);
});

test('credential connect refusals offer the same no-key sheet and Settings action as missing_credential', () => {
    for (const state of ['failed', 'closed']) {
        assert.deepEqual(callPresentation({ state, code: 'credential' }),
            callPresentation({ state, code: 'missing_credential' }));
        const words = 'The selected engine needs a credential.';
        const refused = callPresentation({ state, code: 'credential', words });
        assert.equal(refused.title, words);
        assert.equal(refused.actions[0].kind, 'settings');
        assert.equal(callPresentation({ state, code: 'provider_error' }).actions[0].kind, 'retry');
    }
});

test('shared-limit endings use the same words and action for both host codes', () => {
    for (const state of ['failed', 'closed']) {
        const budget = callPresentation({state, code:'budget_exhausted'});
        assert.deepEqual(budget, callPresentation({state, code:'budget_exceeded'}));
        assert.equal(budget.title, 'The spending limit is reached');
        assert.equal(budget.actions[0].kind, 'settings');
        assert.equal(budget.actions[0].tab, 'behavior');
        assert.equal(callPresentation({state, code:'owner_bye'}).actions.length, 0);
    }
});

test('unknown ending codes show the server sentence and never the raw code', () => {
    for (const code of ['context_capacity', 'future_code', 'call_model_missing']) {
        const sentence = 'This room cannot fit the selected voice model.';
        const shown = callPresentation(ended(code, {detail:sentence, action:'settings'}));
        assert.equal(shown.title, sentence);
        assert.equal(shown.small, '');
        assert.equal(shown.actions[0].kind, 'settings');
        const bare = callPresentation(ended(code));
        assert.equal(bare.title, 'The call ended');
        assert.equal(bare.small, '');
    }
    assert.equal(callPresentation(ended('owner_bye', {detail:'Provider detail'})).title, 'Call ended');
});

test('the real admission opening echo folds into one call card live and after reload', () => {
    const { feed, doc, cards, card, lines, row } = feedFixture();
    // supervisor/message_ingress.py's /api/call/open echo has no task_id.
    const echo = { type: 'chat', role: 'system', content: 'The owner has just opened this call and can hear you now.',
        client_message_id: 'call:c1:open', system_type: 'call_open', chat_id: 1, source: 'web',
        text_placeholder: true, ingress_accepted: true };
    const opening = row(echo.role, echo.content, { clientMessageId: echo.client_message_id, systemType: echo.system_type });
    const ordinary = row('system', 'Unrelated notice', { clientMessageId: 'web:notice', systemType: 'notice' });
    cards.scan();
    assert.equal(opening.classList.contains('call-folded'), true);
    assert.equal(ordinary.classList.contains('call-folded'), false);
    assert.equal(card().dataset.callCard, 'live');
    assert.deepEqual(lines(), []); // opening metadata is not a spoken utterance
    const initial = card();
    row('user', 'Hello', { clientMessageId: 'call:c1:owner-1', taskId: 'task-1' });
    row('assistant', 'Welcome', { taskId: 'task-1', systemType: 'model_narration' });
    cards.scan();
    assert.equal(card(), initial);
    assert.equal(feed.children.filter(n => n.className === 'call-card').length, 1);
    assert.deepEqual(lines().map(([, text]) => text), ['YouHello', 'OuroborosWelcome']);
    cards.destroy();
    const reloaded = createCallCards({ feed, doc, live: () => '' });
    assert.equal(opening.classList.contains('call-folded'), true);
    assert.equal(card().dataset.callCard, 'ended');
    assert.deepEqual(lines().map(([, text]) => text), ['YouHello', 'OuroborosWelcome']);
    // Calls with only opening metadata remain independent; unrelated notices stay visible.
    const another = row('system', echo.content, { clientMessageId: 'call:c2:open', systemType: 'call_open' });
    reloaded.scan();
    assert.equal(another.classList.contains('call-folded'), true);
    assert.equal(feed.children.filter(n => n.className === 'call-card').length, 2);
    assert.equal(ordinary.classList.contains('call-folded'), false);
    reloaded.destroy();
});

test('obsolete time caps have no ending words or action; a deliberate goodbye names Ouroboros', () => {
    for (const code of ['max_minutes', 'invalid_call_max_minutes']) {
        const p = callPresentation(ended(code));
        assert.equal(p.title, 'The call ended');
        assert.deepEqual(p.actions, []);
    }
    const bye = callPresentation(ended('ouroboros_bye'));
    assert.equal(bye.title, 'Ouroboros ended the call');
    assert.equal(bye.tone, '');
    assert.equal(callPresentation(ended('owner_bye')).title, 'Call ended');
});

test('a call nobody joined folds to one notice line, and the engine line reads from the opening row', () => {
    const { feed, cards, card, lines, row } = feedFixture({ live: () => '' });
    const opening = row('system', 'The owner has just opened this call and can hear you now. Engine: Grok · xAI, grok-voice-3.0; no screen vision.',
        { clientMessageId: 'call:c9:open', systemType: 'call_open', rowAddress: 'row:1@t#9' }, 1000);
    const work = new FakeNode('div', feed.ownerDocument);
    work.className = 'chat-live-card';
    work.dataset.taskId = 't9';
    feed.append(work);
    // The closing notice answers the opening row: its quote names that row, and so the call.
    const notice = row('system', 'The call did not start: no one joined the call.', { taskId: 't9' }, 21000);
    const quote = new FakeNode('div', feed.ownerDocument);
    quote.className = 'chat-quote';
    quote.dataset.address = 'row:1@t#9';
    notice.append(quote);
    cards.scan();
    assert.ok(opening.classList.contains('call-folded') && work.classList.contains('call-folded') && notice.classList.contains('call-folded'));
    assert.equal(card().dataset.callCard, 'silent');
    assert.equal(card().querySelector('.call-card-title').textContent, 'The call did not start: no one joined the call.');
    assert.equal(card().querySelector('.call-card-facts').textContent, '');
    assert.equal(card().querySelector('.call-card-engine').textContent, 'Grok · xAI, grok-voice-3.0; no screen vision');
    card().querySelector('.call-card-head').click();
    assert.equal(card().dataset.callOpen, '0');  // one line has nothing to open
    // Words make it a call card again: the title returns, the engine line stays.
    row('user', 'Hello?', { clientMessageId: 'call:c9:o1', taskId: 't9' }, 2000);
    cards.scan();
    assert.deepEqual([card().dataset.callCard, card().querySelector('.call-card-title').textContent], ['ended', 'Call']);
    assert.equal(card().querySelector('.call-card-facts').textContent, '19 s · 1 line');
    assert.deepEqual(lines().map(([, text]) => text), ['YouHello?', 'The call did not start: no one joined the call.']);
});
