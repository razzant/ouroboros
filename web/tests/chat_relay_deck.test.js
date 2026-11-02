// Notes between works (chat_relay_deck.js) over the DOM double: a relay row's bubble is dressed as a note card and
// keeps its row (address, history identity, actions); a run of notes is one stack with the latest card in front, up to
// two ledges behind it and the older cards tucked (never `hidden`: a quote jump still lands on them); the head is an
// address whose receiver and sender read by room; a blue mark names tucked cards with the owner's words; a row between
// notes splits the run and taking it out joins it again, still open; a note without the teller's line opens with its
// own words in the meta ink; the stack fans out and folds back on its capsules with the keyboard's focus following;
// a note opens as written. Motion and layout run in the browser lane.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createRelayDecks } from '../modules/chat_relay_deck.js';
import { createChatReplies } from '../modules/chat_reply.js';
import { installDomDouble } from './dom_double.js';

const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
// The product's compact clock, read in UTC so the test does not depend on the machine's zone.
const clock = (iso) => {
    const d = new Date(iso);
    const hhmm = `${String(d.getUTCHours()).padStart(2, '0')}:${String(d.getUTCMinutes()).padStart(2, '0')}`;
    return { short: `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${hhmm}`, full: `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}, ${d.getUTCFullYear()} at ${hhmm}` };
};

function fixture(options = {}) {
    const env = installDomDouble();
    const { el } = env;
    const feed = el('div', { className: 'chat-messages' });
    env.doc.body.append(feed);
    const decks = createRelayDecks({ feed, lookupProject: async () => null, formatTime: clock, ...options });
    let seq = 0;
    const note = (text, { summary = '', sender = { name: 'Main', chat_id: 1 }, quote = '', source = null, target = null, ts = '' } = {}) => {
        seq += 1;
        ts ||= `2026-10-21T23:${String(10 + seq).padStart(2, '0')}:00Z`;
        const bubble = el('div', { className: 'chat-bubble assistant',
            dataset: { systemType: 'relay_entry', ts: String(Date.parse(ts)), rowAddress: `row:${seq}@r#${seq}` } },
        el('div', { className: 'sender', text: 'Ouroboros' }),
        ...(quote ? [el('div', { className: 'chat-quote', dataset: { mode: 'relay' }, text: quote })] : []),
        el('div', { className: 'message', text }),
        el('div', { className: 'msg-time', text: '23:10' }),
        el('div', { className: 'chat-message-actions' }, el('button', { className: 'chat-message-reply' }), el('button', { className: 'chat-message-copy' })));
        const relay = { relay_id: `r${seq}`, text: quote, quotes: quote ? [{ text: quote }] : [],
            ...(summary ? { summary } : {}), ...(sender ? { sender } : {}), ...(source ? { source } : {}), ...(target ? { target } : {}) };
        decks.adopt(bubble, { text, ts, relay }, Boolean(quote));  // dressed before it lands, as decorate does
        feed.append(bubble);
        return bubble;
    };
    const row = (text) => {
        const bubble = el('div', { className: 'chat-bubble user', text });
        feed.append(bubble);
        return bubble;
    };
    const visible = (bubble) => bubble.dataset.deckTucked !== '1';
    const act = (bubble) => bubble.querySelector('.chat-relay-act');
    const more = (bubble) => bubble.querySelector('.chat-relay-more');
    const from = (bubble) => bubble.querySelector('.chat-relay-from');
    const to = (bubble) => bubble.querySelector('.chat-relay-to');
    const time = (bubble) => bubble.querySelector('.chat-relay-time');
    const mark = (bubble) => bubble.querySelector('.chat-relay-words-inside');
    const ledges = (bubble) => bubble.querySelectorAll('.chat-relay-ledge');
    return { env, feed, decks, note, row, visible, act, more, from, to, time, mark, ledges, close() { decks.destroy(); env.restore(); } };
}

test('a run of notes is one stack of the rows themselves: the latest in front, two ledges behind, the older tucked but never hidden', async () => {
    const f = fixture();
    try {
        const rows = Array.from({ length: 8 }, (_, i) => f.note(`raw${i}:glued-work-words`, { summary: `Line ${i} for the owner` }));
        await settle();
        const front = rows[7];
        assert.equal(f.feed.children.length, 8, 'no element is added to the feed: the rows are the cards');
        assert.ok(rows.every((bubble) => bubble.classList.contains('chat-relay-note') && bubble.hidden === false), 'every row stays a row, none hidden');
        assert.ok(rows.every((bubble, i) => bubble.dataset.rowAddress === `row:${i + 1}@r#${i + 1}`), 'addresses unchanged');
        assert.ok(rows.slice(0, 7).every((bubble) => bubble.dataset.deckTucked === '1'), 'the older notes are tucked');
        assert.equal(f.visible(front), true);
        assert.equal(front.dataset.deck, 'closed');
        assert.equal(front.dataset.deckEdges, '2', 'the front card draws two ledges behind it');
        const layers = front.children.slice(0, 3);
        assert.deepEqual(layers.map((n) => n.className), ['chat-relay-ledge', 'chat-relay-ledge', 'chat-relay-face'], 'the layers lie first, behind the content');
        assert.ok(layers.every((n) => n.getAttribute('aria-hidden') === 'true'), 'the layers are decoration');
        assert.deepEqual([...f.ledges(front)].map((n) => n.dataset.level), ['2', '1']);
        assert.equal(f.from(front).textContent, 'Main');
        assert.equal(f.to(front).textContent, 'this work', 'in a work\'s room the receiver is this work');
        assert.equal(front.querySelector('.chat-relay-arrow').getAttribute('aria-hidden'), 'true');
        assert.equal(front.querySelector('.chat-relay-gist').textContent, 'Line 7 for the owner');
        assert.equal(f.act(front).hidden, false);
        assert.equal(f.act(front).textContent, '8 notes');
        assert.equal(f.act(front).getAttribute('aria-label'), 'Show 8 notes');
        assert.equal(f.act(front).getAttribute('aria-expanded'), 'false');
        assert.ok(f.act(front).className.split(' ').includes('ui-hit-area') && f.act(front).className.includes('btn-sm'), 'a real capsule with a finger\'s target');
        assert.equal(front.dataset.raw, '0', 'the note as written waits for a press');
    } finally { f.close(); }
});

test('a single note is a card without ledges or a capsule; two notes draw one ledge', async () => {
    const f = fixture();
    try {
        const lone = f.note('alone', { summary: 'One line' });
        f.row('an answer');
        const first = f.note('first');
        const second = f.note('second');
        await settle();
        assert.equal(lone.dataset.deck, undefined);
        assert.equal(lone.dataset.deckEdges, undefined, 'no ledge behind a card on its own');
        assert.equal(f.act(lone).hidden, true, 'no capsule on a card on its own');
        assert.equal(f.time(lone).textContent, 'Oct 21, 23:11');
        assert.equal(first.dataset.deckTucked, '1');
        assert.equal(second.dataset.deckEdges, '1', 'two notes: one ledge behind the front card');
        assert.equal(f.act(second).textContent, '2 notes');
    } finally { f.close(); }
});

test('the closed stack\'s time is a range: a shared day said once, two days both', async () => {
    const f = fixture();
    try {
        const sameDay = [f.note('a', { ts: '2026-10-21T22:27:00Z' }), f.note('b', { ts: '2026-10-21T23:13:00Z' })];
        f.row('break');
        const twoDays = [f.note('c', { ts: '2026-10-19T22:27:00Z' }), f.note('d', { ts: '2026-10-20T02:13:00Z' })];
        await settle();
        assert.equal(f.time(sameDay[1]).textContent, 'Oct 21, 22:27–23:13');
        assert.equal(f.time(twoDays[1]).textContent, 'Oct 19, 22:27–Oct 20, 02:13');
        f.act(twoDays[1]).click();
        assert.equal(f.time(twoDays[1]).textContent, 'Oct 20, 02:13', 'open, each card says its own time');
        assert.equal(f.time(twoDays[0]).textContent, 'Oct 19, 22:27');
    } finally { f.close(); }
});

test('the owner\'s words join a deck, while any other row ends a run', async () => {
    const f = fixture();
    try {
        const first = f.note('first');
        const second = f.note('second');
        f.row('owner says something');
        const quoted = f.note('third, passing his words', { quote: 'please check the logs', summary: 'Asks the work to check the logs' });
        const fourth = f.note('fourth');
        await settle();
        assert.equal(first.dataset.deckTucked, '1');
        assert.equal(second.dataset.deckEdges, '1');
        assert.equal(quoted.dataset.deckTucked, '1', 'his words join the next note in a deck');
        const kids = quoted.children;
        const quoteAt = kids.indexOf(quoted.querySelector('.chat-quote'));
        assert.ok(quoteAt > kids.indexOf(quoted.querySelector('.chat-relay-card-head'))
            && quoteAt < kids.indexOf(quoted.querySelector('.chat-relay-excerpt')), 'his words between the head and the line');
        assert.equal(fourth.dataset.deck, 'closed');
        assert.equal(f.act(fourth).textContent, '2 notes');
        assert.equal(f.mark(fourth).textContent, 'Your words in 1 earlier note');
        assert.equal(fourth.children.indexOf(f.mark(fourth)), fourth.children.indexOf(fourth.querySelector('.chat-relay-excerpt')) + 1,
            'the blue mark follows the excerpt');
    } finally { f.close(); }
});

test('a later note comes to the front; a row landing between notes splits the run and taking it out joins it, still open', async () => {
    const f = fixture();
    try {
        const one = f.note('one');
        const two = f.note('two');
        await settle();
        const three = f.note('three');
        await settle();
        assert.equal(three.dataset.deckEdges, '2');
        assert.equal(two.dataset.deckTucked, '1', 'the previous front card joins the deck behind the new one');
        assert.equal(f.act(three).textContent, '3 notes');
        const between = f.env.el('div', { className: 'chat-bubble user', text: 'between' });
        f.feed.insertBefore(between, two);
        f.decks.schedule();
        await settle();
        assert.equal(one.dataset.deck, undefined, 'one note before the row is a card on its own');
        assert.equal(f.visible(one), true);
        assert.equal(two.dataset.deckTucked, '1');
        assert.equal(three.dataset.deckEdges, '1');
        f.act(three).click();
        assert.equal(three.dataset.deck, 'open');
        between.remove();
        f.decks.schedule();
        await settle();
        assert.ok([one, two, three].every((bubble) => bubble.dataset.deck === 'open'), 'the joined deck stays open');
        assert.equal(one.dataset.deckFirst, '1', 'Show less moves to the new first card');
        assert.equal(f.act(one).textContent, 'Show less');
        assert.equal(f.act(two).hidden, true);
        assert.equal(f.act(three).hidden, true, 'the front card of an open column carries no control');
    } finally { f.close(); }
});

test('an ordinary reconcile rewrites nothing, and focus follows the deck\'s one control only when it had it', async () => {
    const f = fixture();
    try {
        const rows = [f.note('a'), f.note('b'), f.note('c')];
        await settle();
        const capsule = f.act(rows[2]);
        const head = rows[2].querySelector('.chat-relay-card-head');
        f.decks.schedule();
        await settle();
        assert.equal(f.act(rows[2]), capsule, 'an unchanged deck keeps its nodes');
        assert.equal(rows[2].querySelector('.chat-relay-card-head'), head);
        // A pointer opens the stack: nothing takes the focus.
        rows[2].click();
        assert.equal(rows[2].dataset.deck, 'open');
        assert.equal(f.env.doc.activeElement, f.env.doc.body, 'a press on the face moves no focus');
        f.act(rows[0]).click();
        assert.equal(rows[2].dataset.deck, 'closed');
        assert.equal(f.env.doc.activeElement, f.env.doc.body);
        // The keyboard opens it: focus moves to Show less on the first card, and back to the capsule on folding.
        capsule.focus();
        capsule.click();
        assert.equal(f.env.doc.activeElement, f.act(rows[0]), 'the opener had focus: Show less takes it');
        assert.equal(f.act(rows[0]).getAttribute('aria-expanded'), 'true');
        assert.equal(f.act(rows[0]).getAttribute('aria-label'), null, 'Show less says what it does');
        f.act(rows[0]).click();
        assert.equal(f.env.doc.activeElement, capsule, 'folded, the capsule has the focus again');
        assert.equal(capsule.getAttribute('aria-expanded'), 'false');
        // An older note becoming the first card of an open column takes the focused Show less with it.
        capsule.focus();
        capsule.click();
        const older = f.note('older');
        f.feed.insertBefore(older, rows[0]);
        f.decks.schedule();
        await settle();
        assert.equal(older.dataset.deckFirst, '1');
        assert.equal(f.act(rows[0]).hidden, true);
        assert.equal(f.env.doc.activeElement, f.act(older), 'Show less kept the focus on its new card');
    } finally { f.close(); }
});

test('a note read on its own opens the deck it joins, so the passage stays; Show less keeps a deck shut', async () => {
    const f = fixture();
    try {
        const first = f.note('a long note being read');
        await settle();
        f.more(first).click();
        assert.equal(first.dataset.raw, '1');
        const second = f.note('a second note arrives');
        await settle();
        assert.equal(first.dataset.deck, 'open', 'the deck it joins opens');
        assert.equal(f.visible(first), true, 'the note being read stays on screen');
        f.act(first).click();
        assert.equal(first.dataset.deckTucked, '1');
        f.note('a third note');
        await settle();
        assert.equal(second.dataset.deck, 'closed', 'after Show less a new note keeps the deck shut');
        assert.equal(first.dataset.raw, '1', 'its own open text stays open behind the front card');
    } finally { f.close(); }
});

test('a note without the teller\'s line opens with its own words, in the meta ink; a summary leads otherwise', async () => {
    const f = fixture();
    try {
        const raw = f.note('a0e1:research-done,no-safe-restore');
        f.row('break');
        const summarized = f.note('a0e1:research-done', { summary: 'Research is done; no safe fix yet' });
        await settle();
        assert.equal(raw.querySelector('.chat-relay-gist').dataset.raw, '1');
        assert.equal(raw.querySelector('.chat-relay-gist').textContent, 'a0e1:research-done,no-safe-restore');
        assert.equal(summarized.querySelector('.chat-relay-gist').dataset.raw, undefined, 'the teller\'s line reads in the voice\'s ink');
        assert.equal(summarized.querySelector('.chat-relay-gist').textContent, 'Research is done; no safe fix yet');
    } finally { f.close(); }
});

test('the sender reads by room: Main from a work\'s room, Another Main work inside Main, an unrecorded room never invented', async () => {
    const f = fixture();
    try {
        const main = f.note('from main', { sender: null, source: { chat_id: 1, label: 'Main' } });
        f.row('break');
        const unknown = f.note('from a project', { sender: null, source: { chat_id: 52601710, label: 'Main' } });
        f.row('break');
        const named = f.note('named', { sender: { name: 'Claudexor 478', chat_id: 52601710 } });
        f.row('break');
        const old = f.note('old', { sender: { task_id: 'missing-task' }, source: { chat_id: 42 } });
        await settle();
        assert.equal(f.from(main).textContent, 'Main');
        assert.equal(main.dataset.kind, 'main');
        assert.equal(f.from(unknown).textContent, 'Another work', 'never an invented room');
        assert.equal(unknown.dataset.kind, 'project');
        assert.equal(f.from(named).textContent, 'Claudexor 478');
        assert.equal(named.dataset.kind, 'project');
        assert.equal(f.from(old).textContent, 'Another work');
    } finally { f.close(); }
    const m = fixture({ chatId: 1 });
    try {
        const fromMain = m.note('a Main work writes to Main', { sender: { name: 'Main', chat_id: 1 } });
        m.row('break');
        const fromProject = m.note('a Project writes to Main', { sender: { name: 'Claudexor 478', chat_id: 52601710 } });
        await settle();
        assert.equal(m.from(fromMain).textContent, 'Another Main work', 'in Main a Main sender is another Main work, never Main → Main');
        assert.equal(m.from(fromProject).textContent, 'Claudexor 478');
    } finally { m.close(); }
});

test('in Main the receiver is the work the host stamped, or the count of works a closed stack addresses; elsewhere this work', async () => {
    const m = fixture({ chatId: 1 });
    try {
        const stamped = m.note('to one work', { target: { task_id: '9c3c', title: 'Возобновление задач и контроль кэша' } });
        m.row('break');
        const old = m.note('saved before the stamp');
        m.row('break');
        const run = [
            m.note('first', { target: { task_id: 'aaaa', title: 'Work A' } }),
            m.note('second', { target: { task_id: 'bbbb', title: 'Work B' } }),
            m.note('third', { target: { task_id: 'aaaa', title: 'Work A' } }),
        ];
        await settle();
        assert.equal(m.to(stamped).textContent, 'Возобновление задач и контроль кэша');
        assert.equal(m.to(old).textContent, 'this work', 'an old row without the stamp still reads');
        assert.equal(m.to(run[2]).textContent, '2 works', 'a closed stack naming two works says so');
        assert.notEqual(m.to(run[2]).textContent, 'Work A', 'never the front card\'s work as the stack\'s');
        m.act(run[2]).click();
        assert.deepEqual(run.map((b) => m.to(b).textContent), ['Work A', 'Work B', 'Work A'], 'open, each card names its own work');
        m.act(run[0]).click();
        m.row('break');
        const mixed = [m.note('unstamped'), m.note('stamped', { target: { task_id: 'cccc', title: 'Work C' } })];
        m.row('break');
        const shared = [m.note('one', { target: { task_id: 'dddd', title: 'Work D' } }), m.note('two', { target: { task_id: 'dddd', title: 'Work D' } })];
        await settle();
        assert.equal(m.to(mixed[1]).textContent, 'this work', 'a stack with a note of unknown receiver names no common work');
        assert.equal(m.to(shared[1]).textContent, 'Work D', 'every note names one work: the stack says it');
        m.act(mixed[1]).click();
        assert.deepEqual(mixed.map((b) => m.to(b).textContent), ['this work', 'Work C']);
    } finally { m.close(); }
    const p = fixture({ chatId: 42 });
    try {
        const note = p.note('to this work', { target: { task_id: '9c3c', title: 'A work title' } });
        await settle();
        assert.equal(p.to(note).textContent, 'this work', 'a work\'s room reads this work whatever the stamp says');
    } finally { p.close(); }
});

test('a closed stack written by several rooms says how many more; open, each card says its own', async () => {
    const f = fixture();
    try {
        const run = [
            f.note('from a project', { sender: { name: 'Claudexor 478', chat_id: 52601710 } }),
            f.note('from another project', { sender: { name: 'Community 2237', chat_id: 110770558 } }),
            f.note('from Main', { sender: { name: 'Main', chat_id: 1 } }),
        ];
        await settle();
        const rooms = (b) => b.querySelector('.chat-relay-rooms');
        assert.equal(rooms(run[2]).hidden, false);
        assert.equal(rooms(run[2]).textContent, 'and 2 more rooms');
        f.act(run[2]).click();
        assert.ok(run.every((b) => rooms(b).hidden), 'open, the cards speak for themselves');
        f.act(run[0]).click();
        f.note('from Main again', { sender: { name: 'Main', chat_id: 1 } });
        f.feed.removeChild(run[1]);
        f.decks.schedule();
        await settle();
        const front = f.feed.children[f.feed.children.length - 1];
        assert.equal(rooms(front).textContent, 'and 1 more room');
    } finally { f.close(); }
});

test('the capsule fans the deck out; more opens the note with the row\'s actions; Show less folds it', async () => {
    const f = fixture();
    try {
        const rows = [f.note('w1'), f.note('w2:as-written'), f.note('w3')];
        await settle();
        f.act(rows[2]).click();
        assert.ok(rows.every((bubble) => bubble.dataset.deck === 'open' && f.visible(bubble)), 'every card stands');
        assert.equal(rows[0].dataset.deckFirst, '1');
        assert.equal(f.act(rows[0]).textContent, 'Show less');
        assert.equal(rows[2].dataset.deckEdges, undefined, 'open, no ledges');
        assert.equal(f.act(rows[2]).hidden, true);
        const middle = rows[1];
        assert.equal(f.more(middle).textContent, 'more');
        assert.match(f.more(middle).getAttribute('aria-controls'), /^chat-relay-text-\d+-gist chat-relay-text-\d+$/);
        f.more(middle).click();
        assert.equal(middle.dataset.raw, '1', 'the note as written and its Reply and Copy show');
        assert.equal(f.more(middle).textContent, 'less');
        assert.equal(f.more(middle).getAttribute('aria-expanded'), 'true');
        f.more(middle).click();
        assert.equal(middle.dataset.raw, '0');
        f.act(rows[0]).click();
        assert.ok(rows.slice(0, 2).every((bubble) => bubble.dataset.deckTucked === '1'));
        assert.equal(f.act(rows[2]).textContent, '3 notes');
        assert.equal(f.act(rows[0]).hidden, true, 'folded, the first card\'s control goes');
    } finally { f.close(); }
});

test('a press on the front card fans the stack, except on its quote, a control or a selection', async () => {
    const f = fixture();
    try {
        const rows = [f.note('one'), f.note('two', { quote: 'his words' })];
        await settle();
        f.env.doc.selection = 'his';
        rows[1].click();
        assert.equal(rows[1].dataset.deck, 'closed', 'a selection is being made, not a press');
        f.env.doc.selection = null;
        rows[1].querySelector('.chat-quote').emit('click', { target: rows[1].querySelector('.chat-quote') });
        assert.equal(rows[1].dataset.deck, 'closed', 'the quote is its own control');
        f.more(rows[1]).click();
        assert.equal(rows[1].dataset.deck, 'closed', 'more opens only this card');
        assert.equal(rows[1].dataset.raw, '1');
        rows[1].click();
        assert.equal(rows[1].dataset.deck, 'open');
    } finally { f.close(); }
});

test('a quote\'s jump to a tucked note opens its deck at once; a visible note or any other row stays as it is', async () => {
    const f = fixture();
    try {
        const first = f.note('one');
        const second = f.note('two');
        const owner = f.row('mine');
        await settle();
        assert.equal(f.decks.surface(second), null, 'the front card is already visible');
        assert.equal(f.decks.surface(first), first);
        assert.equal(first.dataset.deck, 'open');
        assert.equal(f.visible(first), true);
        assert.equal(f.decks.surface(owner), null);
    } finally { f.close(); }
});

test('a relay row the room decorates is dressed as a note in its room; a bubble without the relay type stays a bubble', async () => {
    const env = installDomDouble();
    const feed = env.el('div', { className: 'chat-messages' });
    const input = env.el('textarea', { attrs: { id: 'chat-input' } });
    env.doc.body.append(feed, env.el('div', { className: 'chat-text-row' }, input));
    const replies = createChatReplies({ chatId: 1, feed, input, reading: () => ({ cancel() {}, scroll() {}, remember() {}, stick: true }) });
    try {
        const make = (dataset) => env.el('div', { className: 'chat-bubble assistant', dataset },
            env.el('div', { className: 'sender', text: 'Ouroboros' }), env.el('div', { className: 'message', text: 'words' }));
        const relayRow = make({ systemType: 'relay_entry' });
        const plain = make({});
        const relay = { relay_id: 'r', text: '', quotes: [], summary: 'Tells the work what to check',
            sender: { name: 'Main', chat_id: 1 }, target: { task_id: 't1', title: 'Receiving work' } };
        replies.decorate(relayRow, { row_address: 'row:1@a#1', role: 'assistant', text: 'words', relay });
        replies.decorate(plain, { row_address: 'row:2@a#2', role: 'assistant', text: 'words', relay });
        feed.append(relayRow, plain);
        await settle();
        assert.equal(relayRow.classList.contains('chat-relay-note'), true);
        assert.equal(relayRow.querySelector('.chat-relay-gist').textContent, 'Tells the work what to check');
        assert.equal(relayRow.querySelector('.chat-relay-from').textContent, 'Another Main work', 'the room (Main) reached the deck');
        assert.equal(relayRow.querySelector('.chat-relay-to').textContent, 'Receiving work');
        assert.equal(relayRow.querySelector('.chat-message-reply') !== null, true, 'the row keeps its own Reply');
        assert.equal(plain.classList.contains('chat-relay-note'), false, 'only a relay row becomes a note');
    } finally { replies.destroy(); env.restore(); }
});

test('the blue mark counts tucked cards with the owner\'s words, never the front card, and follows arrivals and splits', async () => {
    const f = fixture();
    try {
        const one = f.note('first', { quote: 'one' });
        const two = f.note('second', { quote: 'two' });
        const plain = f.note('third, no words');
        const front = f.note('latest', { quote: 'front words' });
        await settle();
        assert.equal(f.mark(front).textContent, 'Your words in 2 earlier notes', 'the front\'s own quote and the wordless card do not count');
        f.more(front).click();
        assert.equal(front.dataset.deck, 'closed', 'more opens only this card, not the deck');
        assert.equal(front.dataset.raw, '1');
        f.act(front).click();
        assert.equal(f.mark(front), null, 'open, the quotes show themselves');
        f.act(one).click();
        const latest = f.note('new arrival');
        await settle();
        assert.equal(f.mark(latest).textContent, 'Your words in 3 earlier notes');
        const boundary = f.row('boundary');
        f.feed.insertBefore(boundary, two);
        f.decks.schedule();
        await settle();
        assert.equal(f.mark(latest).textContent, 'Your words in 2 earlier notes');
        assert.equal(one.dataset.deck, undefined);
        assert.equal(plain.dataset.deckTucked, '1');
    } finally { f.close(); }
});

test('hidden rows and typing are transparent, and a quoted tucked row is surfaced by a jump', async () => {
    const f = fixture();
    try {
        const quoted = f.note('first', { quote: 'owner words' });
        f.row('hidden').hidden = true;
        f.feed.append(f.env.el('div', { className: 'typing-bubble' }));
        const front = f.note('second', { sender: { name: 'Another work', chat_id: 42 } });
        await settle();
        assert.equal(quoted.dataset.deckTucked, '1');
        assert.equal(front.dataset.deckEdges, '1');
        assert.equal(f.decks.surface(quoted), quoted);
        assert.equal(quoted.dataset.deckTucked, undefined);
    } finally { f.close(); }
});

test('known sender rooms use the Project reference, unknown rooms remain plain and lookups are shared', async () => {
    const lookups = [];
    const project = { id: 'sender', chat_id: 42, name: 'Sender room' };
    const f = fixture({ lookupProject: async id => { lookups.push(id); return id === 42 ? project : null; } });
    try {
        const known = f.note('known', { sender: { name: 'Saved name', chat_id: 42 } });
        f.note('same room', { sender: { name: 'Saved name', chat_id: 42 } });
        const absent = f.note('absent', { sender: { name: 'Gone room', chat_id: 99 } });
        const old = f.note('old', { sender: { task_id: 'missing-task' }, source: { chat_id: 42 } });
        await settle();
        const reference = known.querySelector('[data-intent="open-project"]');
        assert.equal(reference.getAttribute('aria-label'), 'Open project Sender room');
        assert.equal(reference.querySelector('.chat-live-project-name').textContent, 'Sender room');
        assert.equal(known.querySelector('.chat-relay-card-head').children[0], reference, 'the reference stands where the name stood');
        assert.equal(f.from(absent).textContent, 'Gone room');
        assert.equal(absent.querySelector('[data-intent="open-project"]'), null);
        assert.equal(f.from(old).textContent, 'Another work');
        assert.deepEqual(lookups, [42, 99], 'a quote source is never used as the sender destination');
    } finally { f.close(); }
});

test('a fanned column at the bottom of a room asks the room to place its first card under the header, and folding returns', async () => {
    const calls = [];
    const position = { stick: true };
    const f = fixture({ reading: () => position, clearance: () => 60, place: (node, opts = {}) => calls.push([node, Boolean(opts.follow)]) });
    try {
        f.feed.getBoundingClientRect = () => ({ top: 100, bottom: 700, height: 600 });
        const rows = [f.note('a'), f.note('b'), f.note('c')];
        await settle();
        // The column does not fit: its first card would stand above the band.
        rows[0].rect = { top: -400, bottom: -100, height: 300 };
        rows[1].rect = { top: -90, bottom: 210, height: 300 };
        rows[2].rect = { top: 220, bottom: 680, height: 460 };
        f.act(rows[2]).click();
        assert.deepEqual(calls, [[rows[0], false]], 'the room places the first card; the deck never scrolls itself');
        f.act(rows[0]).click();
        assert.deepEqual(calls.at(-1), [rows[2], true], 'folded, the reader who was following is returned to the newest message');
        // A reader in history: folding brings the front card back only when it left the band.
        calls.length = 0;
        position.stick = false;
        f.act(rows[2]).click();
        rows[2].rect = { top: 900, bottom: 1360, height: 460 };
        f.act(rows[0]).click();
        assert.deepEqual(calls, [[rows[0], false], [rows[2], false]]);
        // A column that fits asks for nothing.
        calls.length = 0;
        rows[0].rect = { top: 200, bottom: 300, height: 100 };
        rows[1].rect = { top: 300, bottom: 400, height: 100 };
        rows[2].rect = { top: 400, bottom: 500, height: 100 };
        f.act(rows[2]).click();
        f.act(rows[0]).click();
        assert.deepEqual(calls, []);
    } finally { f.close(); }
});

test('whether a fanned column fits is judged in the band a reader sees, not under the composer', async () => {
    const calls = [];
    let composerTop = 700;  // first no composer over the feed, then one over its last 100 px
    const f = fixture({ reading: () => ({ stick: true }), clearance: () => 60, band: () => ({ top: 100, bottom: composerTop }),
        place: (node, opts = {}) => calls.push([node, Boolean(opts.follow)]) });
    try {
        f.feed.getBoundingClientRect = () => ({ top: 100, bottom: 700, height: 600 });
        const rows = [f.note('a'), f.note('b'), f.note('c')];
        await settle();
        // A reader at the newest end: the front card ends at the composer's top, the column's first card is under the header.
        rows[0].rect = { top: 100, bottom: 250, height: 150 };
        rows[1].rect = { top: 250, bottom: 400, height: 150 };
        rows[2].rect = { top: 400, bottom: 600, height: 200 };
        f.act(rows[2]).click();
        assert.equal(calls.length, 0, 'a column that fits what the reader sees asks for nothing');
        f.act(rows[0]).click();
        composerTop = 600;
        f.act(rows[2]).click();
        assert.equal(calls.length, 1, 'with the composer over the feed the same column does not fit: its first card is placed');
        assert.ok(calls[0][0] === rows[0] && calls[0][1] === false);
    } finally { f.close(); }
});

test('a note arriving while the column is open becomes its front, and folding returns the follower to it', async () => {
    const calls = [];
    const position = { stick: true };
    const f = fixture({ reading: () => position, clearance: () => 60, place: (node, opts = {}) => calls.push([node, Boolean(opts.follow)]) });
    try {
        f.feed.getBoundingClientRect = () => ({ top: 100, bottom: 700, height: 600 });
        const rows = [f.note('a'), f.note('b'), f.note('c')];
        await settle();
        rows[0].rect = { top: -400, bottom: -100, height: 300 };
        rows[1].rect = { top: -90, bottom: 210, height: 300 };
        rows[2].rect = { top: 220, bottom: 680, height: 460 };
        f.act(rows[2]).click();
        const late = f.note('d');
        await settle();
        assert.equal(late.dataset.deck, 'open', 'the arrival joins the open column');
        f.act(rows[0]).click();
        const who = calls.map(([node, follow]) => [rows.indexOf(node) >= 0 ? `row ${rows.indexOf(node)}` : node === late ? 'late' : '?', follow]);
        assert.deepEqual(who, [['row 0', false], ['late', true]], 'the follow intent outlives the arrival');
    } finally { f.close(); }
});

test('disposing a room aborts a pending sender lookup and prevents a late DOM write', async () => {
    let resolve, signal;
    const f = fixture({ lookupProject: (_id, pending) => { signal = pending; return new Promise(done => { resolve = done; }); } });
    try {
        const note = f.note('pending', { sender: { name: 'Known name', chat_id: 42 } });
        await settle();
        f.decks.destroy();
        assert.equal(signal.aborted, true);
        resolve({ id: 'known', chat_id: 42, name: 'Resolved room' });
        await settle();
        assert.equal(note.querySelector('[data-intent="open-project"]'), null);
    } finally { f.close(); }
});
