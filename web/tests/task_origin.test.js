// Where a work card's work came from (task_origin.js; DESIGN §5 "Conversation activity block"): the host's
// `place_origin` rides the same carriers as `initiator`, and a root card says it in one meta line under its name
// (the place unless the name already is it, what called the work, who), with every fact whole in Details. Nothing
// is inferred: a card without the fact is the card it was, and a nested child's origin is its parent's. Every
// place, account and scenario here is invented.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { applyPayload, createTranslator, setMissTransport } from '../modules/i18n.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { noteOrigin, paintOrigin, placeOriginParts } from '../modules/task_origin.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { installCardDom } from './helpers/card_dom.js';

// The flat fixture's querySelector does not descend; card internals need a real descendant lookup.
const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

const TS = '2026-11-16T12:00:00Z';
const TASK = 'focus-a';
const SEP = '\u00a0\u00b7 ';
const PLACE = 'GitHub · Lanternworks';
const ORIGIN = { place_id: 'github:example-org/lanternworks', place_label: PLACE, provider: 'github',
    actor: { id: '4242', label: 'tern-walker' }, wakes: ['my_thread'] };
const origin = (patch = {}) => ({ ...ORIGIN, ...patch });
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: history, window: { complete: true } }
            : String(url).startsWith('/api/tasks/') ? { status: 'running' } : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set([7]), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    const part = (id, name) => nodes(walkCard(messages, id)).filter((node) => node?.classList?.contains(name));
    return {
        instance, messages,
        card: (id = TASK) => walkCard(messages, id),
        // One span per part; a part's separator is its own tail, so the parts read as the line reads.
        line: (id = TASK) => { const [node] = part(id, 'chat-live-origin'); return node && !node.hidden ? node.children.map((span) => span.textContent).join('') : ''; },
        lineNodes: (id = TASK) => part(id, 'chat-live-origin'),
        title: (id = TASK) => part(id, 'chat-live-title')[0]?.textContent || '',
        facts: (id = TASK) => walkCard(messages, id)?.querySelector('[data-live-facts]')?.innerHTML || '',
        emit: (type, row) => handlers.get(type)({ chat_id: 1, ts: TS, ...row }),
        state: (data) => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
            active_chat_activities: [], ...data }),
        async replay() { await instance.refreshHistory({ revision: 1 }); await new Promise((resolve) => setTimeout(resolve, 0)); },
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const progress = (patch = {}) => ({ task_id: TASK, role: 'assistant', is_progress: true, narration: true, text: '💬 Reading the thread',
    content: '💬 Reading the thread', initiator: 'consciousness', place_origin: ORIGIN, ...patch });
const terminal = (patch = {}) => ({ role: 'system', system_type: 'task_summary', summary_kind: 'terminal_root_projection', task_id: TASK,
    chat_id: 1, ts: TS, text: 'Done', status: 'completed', outcome_phase: 'done', outcome_final: true,
    outcome_authority: 'canonical_task_result_after_finalization', initiator: 'consciousness', ...patch });

const placeOriginLine = (noted, title) => placeOriginParts(noted, title).map((part) => part.text).join(SEP);
const authored = (words) => `<span data-i18n-authored="">${words}</span>`;

// The record-level tests need a document too: Details rows are escaped as the DOM serializes them.
function withDocument(run) {
    const env = installDom();
    try { return run(); } finally { restoreDom(env.prior); }
}

test('the line names the place, what called the work in a heard message\'s words, and who', () => withDocument(() => {
    assert.equal(placeOriginLine(null), '');
    const noted = (patch) => { const record = { isSubagent: false }; noteOrigin(record, { place_origin: origin(patch) }); return record.placeOrigin; };
    const read = (patch, title = '') => placeOriginLine(noted(patch), title);
    assert.equal(read({}), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: ['new_issue'] }), [PLACE, 'new issue', 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: ['mention', 'my_thread'] }), [PLACE, 'mentions it and in its thread', 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: ['review_requested'], actor: { id: '7', label: 'rowan', owner: true } }),
        [PLACE, 'review requested', 'you'].join(SEP), 'the owner\'s confirmed account is «you»');
    // A kind the vocabulary does not know keeps its spelling; «every message» names no kind.
    assert.equal(read({ wakes: ['new_discussion'] }), [PLACE, 'new discussion', 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: ['any_message'] }), [PLACE, 'tern-walker'].join(SEP));
    // No kind matched: only a routed conversation is a reason; otherwise none is invented.
    assert.equal(read({ wakes: [], routed: true }), [PLACE, 'ongoing conversation', 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: [] }), [PLACE, 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: ['mention'], routed: true }), [PLACE, 'mentions it', 'tern-walker'].join(SEP));
    // An author the admission did not identify adds no word: never a person, never an id on the line.
    assert.equal(read({ actor: undefined }), [PLACE, 'in its thread'].join(SEP));
    assert.equal(read({ actor: { id: '4242' } }), [PLACE, 'in its thread'].join(SEP));
    assert.equal(read({ actor: { label: 'bot', owner: 'true', bot: 1 } }), [PLACE, 'in its thread', 'bot'].join(SEP), 'a flag counts only when true');
    // What someone outside wrote is marked as theirs; the host's own words (the reason, «you») are not.
    assert.deepEqual(placeOriginParts(noted({})).map((part) => part.authored), [true, false, true]);
    assert.deepEqual(placeOriginParts(noted({ actor: { id: '7', label: 'rowan', owner: true } })).map((part) => part.authored), [true, false, false]);
    // A kind its connector declared reads in those words, as written and marked theirs; words for a kind this work
    // did not match, or empty ones, say nothing.
    const discussion = { wakes: ['signal.new_discussion'], labels: { 'signal.new_discussion': 'New discussion', mention: 'Mentions you' } };
    assert.equal(read(discussion), [PLACE, 'New discussion', 'tern-walker'].join(SEP));
    assert.deepEqual(placeOriginParts(noted(discussion)).map((part) => part.authored), [true, true, true]);
    assert.deepEqual(noted(discussion).labels, { 'signal.new_discussion': 'New discussion' });
    assert.equal(read({ wakes: ['mention', 'signal.new_discussion'], labels: { 'signal.new_discussion': 'New discussion' } }),
        [PLACE, 'mentions it and New discussion', 'tern-walker'].join(SEP));
    assert.equal(read({ wakes: ['mention'], labels: { mention: '  ' } }), [PLACE, 'mentions it', 'tern-walker'].join(SEP));
}));

test('Details name a declared kind in its connector\'s words, as written, beside the host\'s own', () => withDocument(() => {
    const rows = (patch) => { const record = { isSubagent: false }; noteOrigin(record, { place_origin: origin(patch) }); return record.originRows; };
    const labels = { 'signal.new_discussion': 'new discussion thread' };
    assert.deepEqual(rows({ wakes: ['signal.new_discussion'], labels })[1], ['What called it', authored('new discussion thread')]);
    assert.deepEqual(rows({ wakes: ['mention', 'signal.new_discussion'], labels })[1], ['What called it', authored('Mentions and new discussion thread')]);
    assert.deepEqual(rows({ wakes: ['mention'] })[1], ['What called it', 'Mentions'], 'the host\'s words alone stay its own');
}));

test('the place is not repeated where the card\'s name already is it; a doubled stored label reads once', () => withDocument(() => {
    const record = { isSubagent: false };
    noteOrigin(record, { place_origin: origin({ place_label: 'GitHub · GitHub · both stands' }) });
    assert.equal(record.placeOrigin.label, 'GitHub · both stands');
    assert.equal(placeOriginLine(record.placeOrigin, 'Fix the export button'), ['GitHub · both stands', 'in its thread', 'tern-walker'].join(SEP));
    for (const title of ['GitHub · both stands', 'GitHub · GitHub · both stands', 'github ·  Both Stands']) {
        assert.equal(placeOriginLine(record.placeOrigin, title), ['in its thread', 'tern-walker'].join(SEP), title);
    }
    // A place the admission carried no label for names nothing on the line: its id is a Details fact.
    const bare = { isSubagent: false };
    noteOrigin(bare, { place_origin: origin({ place_label: undefined }) });
    assert.equal(placeOriginLine(bare.placeOrigin, 'Anything'), ['in its thread', 'tern-walker'].join(SEP));
    assert.equal(bare.originRows[0][1], `${authored('github:example-org/lanternworks')}`, 'the id alone, since it names its service');
}));

test('what is not the host\'s place fact is not an origin; a schedule\'s name keeps its own row', () => withDocument(() => {
    for (const raw of [undefined, null, 'GitHub', [], {}, { place_label: PLACE, wakes: ['mention'] }, { place_id: '  ' }]) {
        const record = { isSubagent: false };
        assert.equal(noteOrigin(record, { place_origin: raw }), false, JSON.stringify(raw));
        assert.equal(record.placeOrigin, undefined);
        assert.equal(record.originRows, undefined);
    }
    const record = { isSubagent: false };
    assert.equal(noteOrigin(record, { schedule_name: 'Daily check' }), true);
    assert.equal(record.scheduleName, 'Daily check');
    assert.equal(noteOrigin(record, { schedule_name: 'Daily check' }), false, 'an unchanged fact is no change');
    assert.equal(noteOrigin(record, null, ORIGIN), true);
    assert.equal(noteOrigin(record, null, { ...ORIGIN }), false);
    assert.equal(noteOrigin(null, { place_origin: ORIGIN }), false);
    assert.equal(paintOrigin({}), false, 'a record without a card paints nothing');
}));

test('every frame carries the fact, so whichever one mints the card names its origin', () => {
    assert.deepEqual(summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 'w', content: '💬 hi', place_origin: ORIGIN }).placeOrigin, ORIGIN);
    assert.deepEqual(summarizeChatLiveEvent({ type: 'tool_call_started', tool: 'read_file', task_id: 'w', place_origin: ORIGIN }).placeOrigin, ORIGIN);
    assert.equal('placeOrigin' in summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't', content: '💬 hi' }), false);
});

test('live: the card says its origin under its name from the first frame; Details keep it whole', () => {
    const f = fixture();
    try {
        f.emit('chat', progress({ suggested_name: 'Answer the export-button question' }));
        assert.equal(f.title(), 'Answer the export-button question');
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
        assert.equal(f.lineNodes().length, 1);
        // The line stands under the name, right before the episodes and the latest line (this stub keeps a card's parts flat).
        const parts = f.card().children.map((node) => ['chat-live-title', 'chat-live-origin', 'chat-live-episodes', 'chat-live-activity']
            .find((name) => node.classList.contains(name))).filter(Boolean);
        assert.deepEqual(parts, ['chat-live-title', 'chat-live-origin', 'chat-live-episodes', 'chat-live-activity']);
        // Its authored parts carry the boundary no translator crosses; the reason is the host's own word.
        assert.deepEqual(f.lineNodes()[0].children.map((span) => Object.hasOwn(span.dataset, 'i18nAuthored')), [true, false, true]);
        const facts = f.facts();
        assert.ok(facts.includes(`<dt>Started by</dt><dd>Consciousness</dd></div><div class="chat-live-fact"><dt>Place</dt><dd>${authored(PLACE)}</dd>`),
            'the place rows follow who started it');
        assert.ok(facts.includes('<dt>What called it</dt><dd>Replies in its threads</dd>'));
        assert.ok(facts.includes(`<dt>From</dt><dd>${authored('tern-walker')} · ${authored('4242')}</dd>`));
        // Later frames repeat the fact and change nothing; a frame without it takes nothing away.
        const node = f.lineNodes()[0];
        const [first] = node.children;
        f.emit('chat', progress({ text: '💬 Checking the code', content: '💬 Checking the code' }));
        f.emit('chat', progress({ text: '💬 Writing the answer', content: '💬 Writing the answer', place_origin: undefined }));
        assert.equal(f.lineNodes()[0], node, 'the same node, never a second line');
        assert.equal(node.children[0], first, 'an unchanged line keeps its own spans');
        assert.equal(f.facts(), facts, 'and an unchanged origin keeps its Details markup byte for byte');
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
    } finally { f.close(); }
});

test('live: a card named by its place says only why and who, and names the place once it is renamed', () => {
    const f = fixture();
    try {
        // The host's admission title is the place's label (task_finalization.wake_cause_title), published before the first frame.
        f.emit('task_named', { task_id: TASK, suggested_name: PLACE });
        f.emit('chat', progress());
        assert.equal(f.title(), PLACE);
        assert.equal(f.line(), ['in its thread', 'tern-walker'].join(SEP));
        f.emit('task_named', { task_id: TASK, suggested_name: 'Replied about the export button.' });
        assert.equal(f.title(), 'Replied about the export button.');
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
    } finally { f.close(); }
});

test('live: every authored word is text; Details say an unidentified author and a matched kind in the sheet\'s words', () => {
    const unsafe = '<img src=x onerror=bad()> & "quoted"';
    const escaped = '&lt;img src=x onerror=bad()&gt; &amp; "quoted"';
    const f = fixture();
    try {
        f.emit('chat', progress({ suggested_name: 'Work', place_origin: origin({ place_label: unsafe, provider: 'chat',
            actor: { id: unsafe, label: unsafe, bot: true }, wakes: ['review_requested', 'assigned'] }) }));
        assert.equal(f.line(), [unsafe, 'review requested and assigned to it', unsafe].join(SEP));
        assert.deepEqual(f.lineNodes()[0].children.map((span) => [span.tagName, span.children.length]), [['SPAN', 0], ['SPAN', 0], ['SPAN', 0]],
            'one text span per part, no markup from a name');
        const facts = f.facts();
        assert.ok(!facts.includes('<img'));
        assert.ok(facts.includes(`<dt>Place</dt><dd>${authored(escaped)} · ${authored('chat')}</dd>`),
            'a label that does not name its service has the service\'s id beside it');
        assert.ok(facts.includes('<dt>What called it</dt><dd>Review requests addressed to me and assignments addressed to me</dd>'));
        assert.ok(facts.includes(`<dt>From</dt><dd>${authored(escaped)} · bot</dd>`), 'an id equal to the name is said once');
    } finally { f.close(); }
    const g = fixture();
    try {
        g.emit('chat', progress({ suggested_name: 'Work', place_origin: origin({ actor: undefined, wakes: [], routed: true }) }));
        assert.equal(g.line(), [PLACE, 'ongoing conversation'].join(SEP));
        assert.ok(g.facts().includes('<dt>What called it</dt><dd>An ongoing conversation of its work; no checked kind matched</dd>'));
        assert.ok(g.facts().includes('<dt>From</dt><dd>Author not recorded</dd>'));
        assert.ok(g.facts().includes(`<dt>Place</dt><dd>${authored(PLACE)}</dd>`), 'a label that names its service says it once');
    } finally { g.close(); }
});

// The translator's own DOM needs: a text walker and each text node's parent element (the card DOM helper has neither).
function translatorDom(t) {
    const dom = installCardDom();
    const { doc } = dom;
    doc.createTextNode = (words) => { const holder = doc.createElement('span'); holder.textContent = words; return holder.childNodes[0]; };
    doc.createTreeWalker = (root) => {
        const texts = [];
        const visit = (node) => {
            if (node.nodeType === 3) {
                Object.defineProperty(node, 'parentElement', { configurable: true, get: () => node.parentNode });
                texts.push(node);
            } else (node.childNodes || []).forEach(visit);
        };
        visit(root);
        let index = 0;
        return { nextNode: () => texts[index++] || null };
    };
    t.after(() => dom.restore());
    return doc;
}

test('under the real translator a place, an account and an id stay as written while the host\'s own words translate', (t) => {
    const doc = translatorDom(t);
    setMissTransport(() => Promise.resolve(null));
    applyPayload({ language: 'fr', english: false, entries: {
        'code:places.heard.my_thread': { text: 'dans son fil' }, 'code:places.wake.my_thread': { text: 'réponses dans ses fils' },
        'code:task.fact.place': { text: 'Lieu' }, 'code:task.fact.called_by': { text: 'Ce qui l’a appelé' }, 'code:task.fact.from': { text: 'De' },
        'code:task.origin.bot': { text: 'robot' }, 'code:places.account.you': { text: 'vous' },
        // The same words as the names below: a translator that reached them would rewrite them.
        Settings: { text: 'Réglages' }, Files: { text: 'Fichiers' }, 'Settings-7': { text: 'Réglages-7' },
    } });
    t.after(() => applyPayload({ language: '', english: true, entries: {} }));
    const card = doc.createElement('div');
    const title = doc.createElement('span'), episodes = doc.createElement('div'), facts = doc.createElement('dl'), plain = doc.createElement('p');
    plain.textContent = 'Settings';  // the control: the same word outside an authored boundary
    card.append(title, episodes, facts, plain);
    doc.body.append(card);
    const paint = (place) => {
        const record = { isSubagent: false, titleEl: title, episodesEl: episodes };
        noteOrigin(record, { place_origin: place });
        // As the card's own painter does (chat_activity.js `paintLiveFacts`): a row with nothing to say is no row.
        facts.innerHTML = record.originRows.filter(([, value]) => value)
            .map(([label, value]) => `<div class="chat-live-fact"><dt>${label}</dt><dd>${value}</dd></div>`).join('');
        return record;
    };
    const record = paint({ place_id: 'p', place_label: 'Settings', provider: 'Files',
        actor: { id: 'Settings-7', label: 'Settings', bot: true }, wakes: ['my_thread'] });
    // The overlay as strict as it can be: it enters everything but an authored boundary, the card included.
    createTranslator({ skipRoots: '[data-i18n-authored]', userContent: '', excludeSelector: '' }).applyTo(doc.body);
    assert.equal(plain.textContent, 'Réglages', 'the translator is at work on this page');
    assert.deepEqual(record.originEl.children.map((span) => [span.textContent, span.hasAttribute('data-i18n-authored')]),
        [[`Settings${SEP}`, true], [`dans son fil${SEP}`, false], ['Settings', true]]);
    const rows = Object.fromEntries(facts.children.map((row) => [row.children[0].textContent, row.children[1]]));
    assert.deepEqual(Object.keys(rows), ['Lieu', 'Ce qui l’a appelé', 'De'], 'the labels are the host\'s and arrive translated');
    assert.deepEqual(rows.Lieu.children.map((span) => [span.textContent, span.hasAttribute('data-i18n-authored')]), [['Settings', true], ['Files', true]]);
    assert.equal(rows['Ce qui l’a appelé'].textContent, 'Réponses dans ses fils');
    assert.equal(rows.De.textContent, 'Settings · robot · Settings-7');
    assert.deepEqual(rows.De.children.map((span) => [span.textContent, span.hasAttribute('data-i18n-authored')]), [['Settings', true], ['Settings-7', true]]);
    // The owner's confirmed account is the host's word, translated; a gap is the host's word too.
    const mine = paint({ place_id: 'p', place_label: 'Settings', provider: 'settings', actor: { id: '9', label: 'Settings', owner: true }, wakes: [] });
    assert.deepEqual(mine.originEl.children.map((span) => [span.textContent, span.hasAttribute('data-i18n-authored')]),
        [[`Settings${SEP}`, true], ['vous', false]]);
    assert.deepEqual(facts.children.map((row) => row.children[0].textContent), ['Lieu', 'De'], 'no kind matched and none was routed: no reason row');
    assert.equal(facts.children[1].children[1].textContent, 'Settings · vous · 9');
});

test('a card without the fact is the card it was: no line node, no rows', () => {
    const f = fixture();
    try {
        f.emit('chat', progress({ suggested_name: 'Own time', place_origin: undefined }));
        assert.equal(f.lineNodes().length, 0);
        assert.doesNotMatch(f.facts(), /<dt>Place|What called it|<dt>From/);
        assert.match(f.facts(), /<dt>Started by<\/dt><dd>Consciousness<\/dd>/);
        f.emit('chat', { task_id: 'owner-work', role: 'assistant', is_progress: true, narration: true, text: '💬 Working', content: '💬 Working' });
        assert.equal(f.lineNodes('owner-work').length, 0);
    } finally { f.close(); }
});

test('replay: a progress row, a terminal row and an evidence row each name the origin; a later row adds it', async () => {
    const named = { suggested_name: 'Answer the export-button question' };
    const cases = {
        'progress row': [progress(named)],
        'terminal row': [progress({ ...named, place_origin: undefined }), terminal({ ...named, place_origin: ORIGIN })],
        'evidence row': [progress({ ...named, place_origin: undefined }), { role: 'system', system_type: 'task_summary',
            summary_kind: 'terminal_root_projection', task_id: TASK, chat_id: 1, ts: TS, text: '', is_progress: false, ...named, place_origin: ORIGIN,
            historical_terminal: { status: 'completed', phase: 'done', ts: TS } }],
        'a later progress row, after one without it': [progress({ ...named, place_origin: undefined }),
            progress({ ...named, text: '💬 Done reading', content: '💬 Done reading', task_terminal_status: 'completed', outcome_phase: 'done',
                outcome_final: true, outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } } })],
    };
    for (const [name, messages] of Object.entries(cases)) {
        const f = fixture(messages.map((row) => ({ chat_id: 1, ts: TS, ...row })));
        try {
            await f.replay();
            assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP), name);
            assert.match(f.facts(), /<dt>What called it<\/dt><dd>Replies in its threads<\/dd>/, name);
            assert.equal(f.lineNodes().length, 1, name);
        } finally { f.close(); }
    }
    // A declared kind, replayed: its connector's words on the line and in Details, each marked theirs.
    const declared = origin({ wakes: ['signal.new_discussion'], labels: { 'signal.new_discussion': 'New discussion' } });
    const f = fixture([progress({ ...named, place_origin: declared })].map((row) => ({ chat_id: 1, ts: TS, ...row })));
    try {
        await f.replay();
        assert.equal(f.line(), [PLACE, 'New discussion', 'tern-walker'].join(SEP));
        assert.deepEqual(f.lineNodes()[0].children.map((span) => span.dataset.i18nAuthored), ['', '', '']);
        assert.match(f.facts(), /<dt>What called it<\/dt><dd><span data-i18n-authored="">New discussion<\/span><\/dd>/);
    } finally { f.close(); }
});

test('a Main portal reads the origin from its census row: no frame of the work reaches Main', async () => {
    const f = fixture();
    try {
        // The card is in Main without the fact; its work then lives in a Project room, and only the census speaks for it.
        f.emit('chat', progress({ suggested_name: 'Answer the export-button question', place_origin: undefined }));
        assert.equal(f.lineNodes().length, 0);
        const census = { activity_id: TASK, chat_id: 7, kind: 'managed_task', phase: 'working', started_at: Date.parse(TS) / 1000,
            last_line: 'Reading the upload code.', last_line_at: TS, tool_calls: 3 };
        f.state({ task_bindings: { [TASK]: { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Exports' }],
            active_chat_activities: [{ ...census, place_origin: ORIGIN }] });
        await settle();
        assert.equal(f.card().dataset.portal, '1');
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
        // The next census repeats the fact, or omits it: the one line stays as it is.
        const node = f.lineNodes()[0];
        f.state({ task_bindings: { [TASK]: { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Exports' }],
            active_chat_activities: [{ ...census, tool_calls: 4, place_origin: ORIGIN }] });
        await settle();
        f.state({ task_bindings: { [TASK]: { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Exports' }],
            active_chat_activities: [{ ...census, tool_calls: 5 }] });
        await settle();
        assert.equal(f.lineNodes().length, 1);
        assert.equal(f.lineNodes()[0], node);
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
    } finally { f.close(); }
});

const STARTED = { role: 'system', system_type: 'project_started', task_id: TASK, project_id: 'p-1', project_name: 'Exports',
    task_name: 'Answer the export-button question', text: 'Exports › Answer the export-button question · Started', history_id: 'h-1' };
const world = (patch = {}) => ({ task_bindings: { [TASK]: { project_id: 'p-1', chat_id: 7 } }, projects: [{ id: 'p-1', name: 'Exports' }],
    active_chat_activities: [{ activity_id: TASK, chat_id: 7, kind: 'managed_task', phase: 'working', project_id: 'p-1', project_name: 'Exports',
        started_at: Date.parse(TS) / 1000, last_line: 'Reading the upload code.', last_line_at: TS, tool_calls: 3, place_origin: ORIGIN, ...patch }] });

test('a portal born in its Project reads the origin from the census: a start row binds its card, and no frame of the work is in Main', async () => {
    const said = [PLACE, 'in its thread', 'tern-walker'].join(SEP);
    const orders = {
        'a replayed start row, then the census': async (f) => {
            await f.replay();
            assert.equal(f.card()?.dataset.portal, '1');
            assert.equal(f.line(), '', 'the start row names no origin, and none is guessed from its words');
            f.state(world());
            await settle();
        },
        'the census, then the replayed start row': async (f) => { f.state(world()); await settle(); await f.replay(); await settle(); },
        'the census, then the live start row': async (f) => { f.state(world()); await settle(); f.emit('chat', { ...STARTED, content: STARTED.text }); await settle(); },
    };
    for (const [name, run] of Object.entries(orders)) {
        const f = fixture(name.includes('replayed') ? [{ chat_id: 1, ts: TS, ...STARTED }] : []);
        try {
            await run(f);
            assert.equal(f.card()?.dataset.portal, '1', name);
            assert.equal(f.title(), 'Answer the export-button question', name);
            assert.equal(f.line(), said, name);
            assert.equal(f.lineNodes().length, 1, name);
            assert.doesNotMatch(f.facts(), /What called it|<dt>Place/, `${name}: a portal discloses nothing, its room's card keeps the Details`);
            // A census that no longer carries the fact takes nothing away.
            f.state(world({ place_origin: undefined, tool_calls: 4 }));
            await settle();
            assert.equal(f.line(), said, name);
        } finally { f.close(); }
    }
});

test('a finished portal keeps its origin from the frozen completion row, with no census row to say it', async () => {
    // After a reload the census no longer lists finished work: the completion row froze `place_origin` with its delivery.
    const done = { role: 'system', system_type: 'project_completion_summary', task_id: TASK, project_id: 'p-1', project_name: 'Exports',
        task_name: STARTED.task_name, text: `Exports › ${STARTED.task_name} · Done`, history_id: 'h-2', place_origin: ORIGIN,
        outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } } };
    const said = [PLACE, 'in its thread', 'tern-walker'].join(SEP);
    const rows = (...items) => items.map((row, index) => ({ chat_id: 1, ts: `2026-11-16T12:0${index}:00Z`, ...row }));
    const orders = {
        'replayed in order': rows(STARTED, done), 'replayed, the completion first': rows(done, STARTED), live: [],
    };
    for (const [name, history] of Object.entries(orders)) {
        const f = fixture(history);
        try {
            if (history.length) await f.replay();
            else for (const row of [STARTED, done]) f.emit('chat', { ...row, content: row.text });
            await settle();
            assert.equal(f.card()?.dataset.portal, '1', name);
            assert.equal(f.line(), said, name);
            f.state({ task_bindings: { [TASK]: { project_id: 'p-1', chat_id: 7 } }, active_chat_activities: [] });
            await settle();
            assert.equal(f.line(), said, `${name}: a census without the work takes nothing away`);
            assert.equal(f.lineNodes().length, 1, name);
        } finally { f.close(); }
    }
});

test('a name another writer gives a portal is met at the next census: the place is never said twice', async () => {
    const f = fixture();
    try {
        // A frame minted the card unnamed, with the fact; the census made it a portal; its start row then names it by
        // the place's own label, writing the title itself.
        f.emit('chat', progress());
        f.state(world());
        await settle();
        assert.equal(f.card().dataset.portal, '1');
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
        f.emit('chat', { ...STARTED, task_name: PLACE, content: `Exports › ${PLACE} · Started` });
        await settle();
        assert.equal(f.title(), PLACE);
        f.state(world({ tool_calls: 4 }));
        await settle();
        assert.equal(f.line(), ['in its thread', 'tern-walker'].join(SEP));
        assert.equal(f.lineNodes().length, 1);
    } finally { f.close(); }
});

test('a nested child\'s origin is its parent\'s: it shows no line and no rows of its own', () => {
    const record = { isSubagent: true };
    assert.equal(noteOrigin(record, { place_origin: ORIGIN }), false);
    assert.equal(record.placeOrigin, undefined);
    assert.equal(record.originRows, undefined);
    const f = fixture();
    try {
        f.emit('chat', progress({ suggested_name: 'Root work' }));
        f.emit('chat', { task_id: 'child-1', parent_task_id: TASK, root_task_id: TASK, delegation_role: 'subagent', subagent_role: 'researcher',
            subagent_event: 'started', subagent_task_id: 'child-1', role: 'assistant', is_progress: true, text: 'started', content: 'started',
            initiator: 'consciousness', place_origin: ORIGIN });
        const child = f.card('child-1');
        assert.ok(child, 'the child card is drawn');
        assert.equal(f.lineNodes('child-1').length, 0);
        assert.doesNotMatch(child.querySelector('[data-live-facts]')?.innerHTML || '', /<dt>Place|What called it/);
        assert.equal(f.lineNodes().length, 1, 'the root keeps its own, and the child adds none inside it');
        assert.equal(f.line(), [PLACE, 'in its thread', 'tern-walker'].join(SEP));
    } finally { f.close(); }
});
