// This browser's saved room text (chat_preview.js): what enters the bounded record, what a warm opening paints, and
// how the painted copy gives way to the canonical read without taking a passage the owner holds selected. The chat
// instance's admission paths (newest read, ↓'s latest page, arrivals, stale and failed reads) are in
// chat_history_integration.test.js and chat_history_arrivals.test.js; real engines in tests/test_phone_chat_reopen_browser.py.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createChatPreview, clearChatPreviews, forgetProjectPreviews, previewsRefused, PREVIEW_KEY } from '../modules/chat_preview.js';
import { senderLabel } from '../modules/chat_activity.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

function fixture(t) {
    const { prior, mount } = installDom();
    t.after(() => restoreDom(prior));
    const values = new Map();
    const storage = { getItem: key => values.get(key) ?? null, setItem: (key, value) => values.set(key, String(value)),
        removeItem: key => values.delete(key) };
    const make = ({ room = [1, ''], scope = '', kept = () => null } = {}) => createChatPreview({ feed: mount, room, scope,
        label: senderLabel, storage: () => storage, mutate: fn => fn(), kept });
    const record = () => JSON.parse(values.get(PREVIEW_KEY));
    const painted = () => mount.children.find(node => node.dataset.savedPreview === 'true');
    return { mount, values, storage, make, record, painted };
}
const row = (id, text, extra = {}) => ({ role: 'assistant', text, history_id: id, ...extra });

test('a new room instance paints only the saved words under their speakers: no controls, task facts or credentials', t => {
    const f = fixture(t), first = f.make();
    first.save([row('chat:1', 'An exact saved answer', { task_id: 'ended-work', cancelable: true,
        review_projection: { findings: ['not saved'] }, api_key: 'not saved', attachments: [{ name: 'file.pdf' }] })]);
    first.destroy();
    assert.doesNotMatch(f.values.get(PREVIEW_KEY), /ended-work|cancelable|api_key|findings|file\.pdf/);
    const fresh = f.make();
    assert.equal(fresh.show(), true);
    assert.equal(fresh.show(), false, 'painted once');
    const root = f.painted();
    assert.equal(root.children[0].textContent, 'Saved text preview · updating');
    assert.equal(root.querySelectorAll('.chat-bubble').length, 1);
    assert.equal(root.querySelectorAll('button').length, 0);
    assert.equal(root.querySelectorAll('[data-history-id]').length, 0, 'it carries no history identity');
    assert.equal(root.children[1].dataset.previewHistoryId, 'chat:1');
    fresh.retire();
    assert.equal(f.painted(), undefined);
    fresh.destroy();
});

test('speakers are frozen as the canonical bubble names them; words and names stay literal text', t => {
    const f = fixture(t), preview = f.make();
    preview.save([
        row('chat:1', 'First line\nSecond line', { role: 'user', source: 'telegram', sender_label: '<img src=x onerror=alert(1)> Mira' }),
        row('chat:2', 'From another window', { role: 'user', source: 'web', sender_session_id: 'another-window-private' }),
        row('chat:3', 'An autonomous thought', { initiator: 'consciousness' }),
        row('chat:4', 'A reviewed skill', { role: 'system', system_type: 'skill_review' }),
        row('chat:5', '<b>not markup</b>'),
    ]);
    assert.doesNotMatch(f.values.get(PREVIEW_KEY), /another-window-private|sender_session_id|initiator|source/);
    preview.destroy();
    const fresh = f.make();
    fresh.show();
    const bubbles = f.painted().children.slice(1);
    assert.deepEqual(bubbles.map(node => node.children[0].textContent),
        ['<img src=x onerror=alert(1)> Mira', 'You', 'Ouroboros · Consciousness', 'Skill Review', 'Ouroboros']);
    assert.equal(bubbles[0].children[1].textContent, 'First line\nSecond line');
    assert.equal(bubbles[4].children[1].textContent, '<b>not markup</b>');
    assert.equal(f.mount.querySelectorAll('img').length + f.mount.querySelectorAll('b').length, 0);
    fresh.destroy();
});

test('room, Project and installation scope never borrow each other\'s text', t => {
    const f = fixture(t);
    f.make().save([row('chat:1', 'Main words')]);
    assert.equal(f.make({ room: [2, ''] }).show(), false);
    assert.equal(f.make({ room: [1, 'other-project'] }).show(), false);
    assert.equal(f.make({ scope: 'another-installation' }).show(), false);
    assert.equal(f.make().show(), true);
});

test('only conversation rows with identity enter; refusals and unreadable streams keep the record; an empty read paints nothing', t => {
    const f = fixture(t), preview = f.make();
    preview.save([row('chat:1', 'Kept answer'), row('progress:2', 'narration', { is_progress: true }),
        row('chat:3', 'card row', { system_type: 'task_summary' }), row('', 'no identity'), row('chat:4', ''),
        { ...row('chat:5', 'a quiz'), system_type: 'quiz' }]);
    assert.deepEqual(f.record().rooms[0].rows.map(item => item.text), ['Kept answer']);
    const saved = f.values.get(PREVIEW_KEY);
    preview.save([row('chat:9', 'wrong')], { reason_code: 'history_source_unavailable' });
    preview.save([], { window: { complete: false, truncated_by: ['chat_source_unavailable'] } });
    assert.equal(f.values.get(PREVIEW_KEY), saved);
    preview.save([row('chat:9', 'bounded')], { window: { complete: false, truncated_by: ['quota'] } });
    assert.deepEqual(f.record().rooms[0].rows.map(item => item.text), ['bounded'], 'an incomplete window is still a window');
    preview.save([], { window: { complete: true } });
    assert.deepEqual(f.record().rooms[0].rows, [], 'read empty: the room keeps an empty record');
    assert.equal(f.make().show(), false);
});

test('arrivals merge by identity in place; a room without a record keeps no fragment', t => {
    const f = fixture(t), preview = f.make();
    preview.append([row('chat:1', 'Arrived before any newest read')]);
    assert.equal(f.values.get(PREVIEW_KEY), undefined);
    preview.save([]);
    preview.append([row('chat:1', 'Arrived since opening')]);
    preview.append([row('chat:1', 'Arrived since opening')]);
    preview.append([row('chat:1', 'Edited in place'), row('chat:2', 'Next', { role: 'user', sender_label: 'Mira', source: 'telegram' }),
        row('progress:3', 'narration', { is_progress: true })]);
    const rows = f.record().rooms[0].rows;
    assert.deepEqual(rows.map(item => [item.id, item.text, item.speaker]),
        [['chat:1', 'Edited in place', 'Ouroboros'], ['chat:2', 'Next', 'Mira']]);
});

test('each row goes where the feed puts its bubble: by time, a delayed arrival at its time, a missing time last', t => {
    // chat_render_batch.js insertTimelineNode draws rows by their time, and a delayed delivery keeps its earlier time
    // (gateway/history_paging.py latest_arrival): stream order is not the conversation's order. A row without a time is
    // stamped with the moment it is drawn (chat.js addMessage), after everything before it.
    const f = fixture(t), preview = f.make();
    const at = minute => `2026-11-03T12:${String(minute).padStart(2, '0')}:00Z`;
    const ids = () => f.record().rooms[0].rows.map(item => item.id);
    preview.save([row('chat:1', 'A', { ts: at(0) }), row('chat:2', 'B', { role: 'user', ts: at(2) }),
        row('chat:3', 'C, delivered late', { ts: at(1) }), row('chat:4', 'D, B\'s minute', { ts: at(2) }),
        row('chat:5', 'E, untimed')]);
    assert.deepEqual(ids(), ['chat:1', 'chat:3', 'chat:2', 'chat:4', 'chat:5']);
    // An arrival stamped before B joins at its time; a row edited in place keeps its place.
    preview.append([row('chat:6', 'F, delayed', { ts: at(1) }), row('chat:2', 'B edited', { role: 'user', ts: at(2) })]);
    assert.deepEqual(ids(), ['chat:1', 'chat:3', 'chat:6', 'chat:2', 'chat:4', 'chat:5']);
    preview.destroy();
    const fresh = f.make();
    fresh.show();
    assert.deepEqual(f.painted().children.slice(1).map(node => node.children[1].textContent),
        ['A', 'C, delivered late', 'F, delayed', 'B edited', 'D, B\'s minute', 'E, untimed'], 'the painted copy reads so');
    fresh.destroy();
    // The bound keeps the newest by time: a delayed old row at the stream's end gives way first.
    const many = Array.from({ length: 84 }, (_, n) => row(`chat:${n}`, `row ${n}`, { ts: `2026-11-03T13:${String(Math.floor(n / 60)).padStart(2, '0')}:${String(n % 60).padStart(2, '0')}Z` }));
    f.make().save([...many, row('chat:delayed', 'stamped before them all', { ts: at(0) })]);
    assert.deepEqual([ids()[0], ids().at(-1), ids().length], ['chat:4', 'chat:83', 80]);
});

test('equal times follow stream position exactly as the feed compares it, before the bound keeps the latest', t => {
    // insertTimelineNode breaks a tie with compareHistoryPosition over the bubble's dataset strings: the source compares
    // as text, so after a log rotation `chat:rotation-10` sorts before `chat:rotation-9`, whatever order they came in.
    const f = fixture(t), preview = f.make();
    const T = '2026-11-03T12:00:00Z', ids = () => f.record().rooms[0].rows.map(item => item.id);
    const at = (source, offset) => ({ ts: T, history_position: { source, offset } });
    preview.save([row('chat:a', 'Older log, later offset', at('chat:rotation-9', 500)),
        row('chat:b', 'Newer log, same millisecond', at('chat:rotation-10', 0)),
        row('chat:c', 'Same log, earlier offset', at('chat:rotation-9', 20))]);
    assert.deepEqual(ids(), ['chat:b', 'chat:c', 'chat:a']);
    // An arrival at the same time takes its place among them by position; one the room has keeps its place.
    preview.append([row('chat:d', 'Arrived, between them', at('chat:rotation-9', 100)),
        row('chat:b', 'Newer log, edited', at('chat:rotation-10', 0))]);
    assert.deepEqual(ids(), ['chat:b', 'chat:c', 'chat:d', 'chat:a']);
    // A time that does not parse leaves its bubble untimed: last, and never the place another row goes before.
    preview.save([row('chat:u', 'Unreadable time', { ts: 'not a time' }), row('chat:t', 'Timed', { ts: T })]);
    assert.deepEqual(ids(), ['chat:u', 'chat:t']);
    // The bound keeps the rows the placement puts last: 81 rows of one millisecond listed against their positions.
    preview.save(Array.from({ length: 81 }, (_, n) => row(`chat:${n}`, `row ${n}`, at('chat:rotation-1', 80 - n))));
    assert.deepEqual([ids()[0], ids().at(-1), ids().length], ['chat:79', 'chat:0', 80]);
});

test('the feed\'s own tie rule orders the preview: imported from chat_history_replay.js, never restated here', async () => {
    // Astra R2: a restated comparator agrees today (the test above pins it) and would silently fork with the next
    // change to compareHistoryPosition. chat_history_replay.js imports only task_checkpoints.js, whose start row reads
    // the Context capsule's sentences (context_observation.js → i18n.js → api_client.js → this module): every name
    // read across that loop is read inside a function, so the order the modules evaluate in does not matter.
    const source = await readFile(new URL('../modules/chat_preview.js', import.meta.url), 'utf8');
    assert.match(source, /^import \{ compareHistoryPosition \} from '\.\/chat_history_replay\.js';$/m);
    assert.doesNotMatch(source, /localeCompare/, 'no second copy of the stream comparison');
});

test('a Project the registry says is going away takes its saved text with it; every other room keeps its own', t => {
    const f = fixture(t);
    f.make({ room: [1, ''] }).save([row('chat:1', 'Main words')]);
    f.make({ room: [100, 'gate-0'] }).save([row('chat:1', 'Words of the deleted Project')]);
    f.make({ room: [100, 'gate-0'], scope: 'another-installation' }).save([row('chat:1', 'Its words in another scope')]);
    f.make({ room: [101, 'gate-1'] }).save([row('chat:1', 'Another Project keeps these')]);
    const others = f.record().rooms.filter(room => JSON.parse(room.key)[1][1] !== 'gate-0');
    forgetProjectPreviews(['gate-0'], f.storage);
    assert.deepEqual(f.record().rooms, others, 'the others keep their rows, order and bounds byte for byte');
    assert.doesNotMatch(f.values.get(PREVIEW_KEY), /deleted Project|another scope/);
    assert.equal(f.make({ room: [100, 'gate-0'] }).show(), false);
    assert.equal(f.make({ room: [101, 'gate-1'] }).show(), true);
    // Nothing to forget writes nothing; storage refused or a record unread changes nothing and throws nothing.
    const before = f.values.get(PREVIEW_KEY);
    const writes = [];
    forgetProjectPreviews(['unknown', ''], { ...f.storage, setItem: (...args) => writes.push(args) });
    forgetProjectPreviews([], { getItem() { throw new Error('not even read'); } });
    assert.deepEqual([writes, f.values.get(PREVIEW_KEY)], [[], before]);
    assert.doesNotThrow(() => forgetProjectPreviews(['gate-1'], { getItem() { throw new Error('denied'); } }));
    f.values.set(PREVIEW_KEY, '{');
    assert.doesNotThrow(() => forgetProjectPreviews(['gate-1'], f.storage));
    assert.equal(f.values.get(PREVIEW_KEY), '{');
});

test('a refusal anywhere in this document takes the record and every painted copy, selection or not; nothing else does', t => {
    const f = fixture(t), w = selectionWorld(f);
    f.make({ room: [1, ''] }).save([row('chat:1', 'Main words')]);
    f.make({ room: [2, ''] }).save([row('chat:2', 'Project words')]);
    const main = f.make({ room: [1, ''] }), project = f.make({ room: [2, ''] });
    main.show();
    project.show();
    const copies = () => f.mount.children.filter(node => node.dataset.savedPreview === 'true').length;
    w.select(f.painted().children[1].children[1]);  // the Project's copy, painted last and so first in the feed
    project.retire();
    main.retire();
    assert.equal(copies(), 1, 'the held copy stays, labelled');
    for (const status of [500, 503, 404, undefined]) assert.equal(previewsRefused({ status }, f.storage), false);
    assert.equal(copies(), 1, 'a failed read is no statement about the session');
    assert.ok(f.values.get(PREVIEW_KEY));
    assert.equal(previewsRefused({ status: 401 }, f.storage), true);
    assert.equal(copies(), 0, 'every room\'s copy goes, the held one too');
    assert.equal(f.values.get(PREVIEW_KEY), undefined);
    // A room's own refused read does the same for its neighbours; a destroyed preview is left alone.
    f.make({ room: [1, ''] }).save([row('chat:1', 'Again')]);
    const again = f.make({ room: [1, ''] }), gone = f.make({ room: [1, ''] });
    again.show();
    gone.destroy();
    const other = f.make({ room: [2, ''] });
    other.refused({ status: 403 });
    assert.equal(copies(), 0);
    assert.equal(f.values.get(PREVIEW_KEY), undefined);
});

test('storage refusal, corrupt data and unsupported versions open cold and leave the room working', t => {
    const f = fixture(t);
    for (const raw of ['{', '{"v":3,"rooms":[]}', `{"v":2,"rooms":[],"pad":"${'x'.repeat(200000)}"}`,
        '{"v":1,"rooms":[{"key":"[\\"\\",[1,\\"\\"]]","rows":[{"id":"chat:1","role":"user","text":"x"}]}]}',
        '{"v":2,"rooms":[{"key":"[\\"\\",[1,\\"\\"]]","rows":[{"id":"chat:1","role":"intruder","text":"x","speaker":"y"}]}]}']) {
        f.values.set(PREVIEW_KEY, raw);
        assert.equal(f.make().show(), false, raw.slice(0, 40));
    }
    const denied = createChatPreview({ feed: f.mount, room: [1, ''], label: senderLabel, mutate: fn => fn(),
        storage: () => { throw new Error('denied'); } });
    assert.equal(denied.show(), false);
    assert.doesNotThrow(() => denied.save([row('chat:1', 'x')]));
    assert.doesNotThrow(() => denied.failed({ status: 401 }));
    assert.doesNotThrow(() => clearChatPreviews({ removeItem() { throw new Error('denied'); } }));
});

test('the record stays under 200,000 characters and 8 rooms, keeps this room first and never cuts a person\'s text', t => {
    const f = fixture(t), text = 'long exact passage '.repeat(1000);
    for (let i = 0; i < 12; i += 1) {
        f.make({ room: [i + 1, ''] }).save(Array.from({ length: 100 }, (_, j) => row(`chat:${j}`, text, { role: 'user' })));
        const raw = f.values.get(PREVIEW_KEY), saved = JSON.parse(raw);
        assert.ok(raw.length <= 200000, `${raw.length} characters`);
        assert.equal(saved.rooms[0].key, JSON.stringify(['', [i + 1, '']]), 'the room just read is kept first');
        assert.ok(saved.rooms.every(room => room.rows.length <= 80 && room.rows.every(item => item.text === text)));
        assert.equal(saved.rooms[0].rows.at(-1).id, 'chat:99', 'its newest rows stay');
    }
    // Small rooms: the eight read most recently, newest first; a room read again moves to the front.
    const g = fixture(t);
    for (let i = 1; i <= 10; i += 1) g.make({ room: [i, ''] }).save([row('chat:1', `room ${i}`)]);
    assert.deepEqual(g.record().rooms.map(room => JSON.parse(room.key)[1][0]), [10, 9, 8, 7, 6, 5, 4, 3]);
    g.make({ room: [5, ''] }).save([row('chat:1', 'room 5 again')]);
    assert.deepEqual(g.record().rooms.map(room => JSON.parse(room.key)[1][0]), [5, 10, 9, 8, 7, 6, 4, 3]);
    clearChatPreviews(g.storage);
    assert.equal(g.make().show(), false);
});

// A document whose selection the test sets: `select(node)` holds a range over that node, `select()` releases it, and
// `changed()` fires selectionchange as an engine would.
function selectionWorld(f) {
    const doc = f.mount.ownerDocument, heard = new Set();
    let live = [];
    doc.addEventListener = (type, fn) => { if (type === 'selectionchange') heard.add(fn); };
    doc.removeEventListener = (type, fn) => heard.delete(fn);
    doc.getSelection = () => ({ get isCollapsed() { return !live.length; }, get rangeCount() { return live.length; },
        getRangeAt: index => live[index] });
    doc.createRange = () => ({ setStart(node, offset) { this.startContainer = node; this.startOffset = offset; },
        setEnd(node, offset) { this.endContainer = node; this.endOffset = offset; },
        intersectsNode(node) { return node.contains(this.startContainer) || node.contains(this.endContainer); } });
    return {
        heard,
        select(node = null) { live = node ? [{ startContainer: node, startOffset: 0, endContainer: node, endOffset: 1 }] : []; },
        changed() { heard.forEach(fn => fn()); },
    };
}

test('a selected passage keeps the retired preview, labelled, until released; the copy room_retire keeps holds it while covered', t => {
    const f = fixture(t), w = selectionWorld(f);
    let kept = null;
    const writer = f.make();
    writer.save([row('chat:1', 'Selected words')]);
    writer.destroy();
    const preview = f.make({ kept: () => kept });
    preview.show();
    const words = f.painted().children[1].children[1];
    w.select(words);
    preview.retire();
    assert.ok(f.painted());
    assert.equal(f.painted().children[0].textContent, 'Saved text preview · retained until selection is released');
    // The drawer covers the room: room_retire.js keeps the range, then WebKit drops the live one in the inert room.
    kept = [[words, 0, words, 1]];
    w.select();
    w.changed();
    assert.ok(f.painted(), 'the kept copy still holds the passage');
    // The drawer closes: the copy is given back as the live selection and the shell lets its copy go.
    w.select(words);
    kept = null;
    w.changed();
    assert.ok(f.painted());
    // A selection elsewhere, or none: the preview gives way.
    w.select(f.mount);  // a range on the feed that does not reach into the preview
    w.changed();
    assert.equal(f.painted(), undefined);
    preview.destroy();
    assert.equal(w.heard.size, 0, 'destroy releases its selection listener');
});

test('a refused session discards the record and the painted copy at once, selection or not; other failures only label it', t => {
    const f = fixture(t), w = selectionWorld(f);
    f.make().save([row('chat:1', 'Private words')]);
    const preview = f.make();
    preview.show();
    preview.failed({ status: 503 });
    assert.equal(f.painted().children[0].textContent, 'Saved text preview · not current');
    assert.ok(f.values.get(PREVIEW_KEY), 'a transient failure erases nothing');
    w.select(f.painted().children[1].children[1]);
    preview.retire();
    assert.ok(f.painted(), 'held by its selection');
    preview.failed({ status: 401 });
    assert.equal(f.painted(), undefined);
    assert.equal(f.values.get(PREVIEW_KEY), undefined);
    w.changed();
    assert.equal(f.painted(), undefined, 'nothing comes back');
    const other = f.make();
    other.save([row('chat:2', 'Again')]);
    other.show();
    other.failed({ status: 403 });
    assert.equal(f.painted(), undefined);
    assert.equal(f.values.get(PREVIEW_KEY), undefined);
});

test('any read the room makes, refused, discards a held copy at once; a page that merely failed changes nothing here', t => {
    // chat.js calls `refused` for every history read (a page, ↓, arrivals), `failed` for its newest read alone.
    const f = fixture(t), w = selectionWorld(f);
    f.make().save([row('chat:1', 'Held words')]);
    const preview = f.make();
    preview.show();
    w.select(f.painted().children[1].children[1]);
    preview.retire();
    const note = f.painted().children[0].textContent, saved = f.values.get(PREVIEW_KEY);
    for (const status of [500, 503, 404, undefined]) preview.refused({ status });
    assert.equal(f.painted().children[0].textContent, note, 'a failed page is no statement about the saved text');
    assert.equal(f.values.get(PREVIEW_KEY), saved);
    preview.refused({ status: 403 });
    assert.equal(f.painted(), undefined, 'selection or not');
    assert.equal(f.values.get(PREVIEW_KEY), undefined);
});

test('destroy removes the painted copy; later calls are inert', t => {
    const f = fixture(t);
    f.make().save([row('chat:1', 'Words')]);
    const preview = f.make();
    preview.show();
    const saved = f.values.get(PREVIEW_KEY);
    preview.destroy();
    assert.equal(f.painted(), undefined);
    assert.equal(preview.show(), false);
    preview.save([row('chat:9', 'late')]);
    preview.append([row('chat:9', 'late')]);
    assert.equal(f.values.get(PREVIEW_KEY), saved);
});

test('known limit: card structure is not rebuilt; a completion mirror and a child final stay flat text rows', t => {
    // Canonical replay draws these inside a work card or as a folded mirror with actions; the preview keeps only their
    // words under their speakers until that replay replaces it (DESIGN "History edges").
    const f = fixture(t), preview = f.make();
    preview.save([
        row('chat:1', 'Launch › Ship · Completed', { role: 'system', system_type: 'project_completion_summary',
            completion_answer: 'Done: PR #7 merged.', project_id: 'launch', task_id: 'final-task' }),
        row('chat:2', 'The child found the cause.', { delegation_role: 'subagent', task_id: 'child-1', parent_task_id: 'root-1' }),
    ]);
    const rows = f.record().rooms[0].rows.map(({ at, ...words }) => words);  // `at`: the moment an untimed row is drawn
    assert.deepEqual(rows, [
        { id: 'chat:1', role: 'system', text: 'Launch › Ship · Completed', speaker: 'System' },
        { id: 'chat:2', role: 'assistant', text: 'The child found the cause.', speaker: 'Ouroboros' }]);
    assert.doesNotMatch(f.values.get(PREVIEW_KEY), /completion_answer|Done: PR|parent_task_id|child-1/);
});
