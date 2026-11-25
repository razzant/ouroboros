// This browser's saved room text (chat_preview.js): what enters a room's bounded entry, what a warm opening paints,
// how the rooms share one total, which erasers take it, and how the painted copy gives way to the canonical read
// without taking a passage the owner holds selected. The chat instance's admission paths (newest read, ↓'s latest page,
// arrivals, stale and failed reads, the socket's late proof) are in chat_history_integration.test.js and
// chat_history_arrivals.test.js; real engines in tests/test_phone_chat_reopen_browser.py.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import vm from 'node:vm';
import { createChatPreview, clearChatPreviews, forgetProjectPreviews, forgetUnlistedProjectPreviews, previewsRefused,
    previewEntryKey, PREVIEW_FAMILY, PREVIEW_KEY } from '../modules/chat_preview.js';
import { senderLabel } from '../modules/chat_activity.js';
import { installDom, memoryStorage, restoreDom, PREVIEW_SCOPE } from './chat_dom_fixture.js';

// A second proven data profile: its rooms are not this one's, though the installation is the same.
const OTHER_PROFILE = '1234567890abcdef';
// DESIGN "History edges": a room's share of entry text (all v2's one record held) and every room's total, keys counted.
const ROOM_CHARS = 200000, TOTAL_CHARS = 1000000;

function fixture(t) {
    const { prior, mount } = installDom();
    t.after(() => restoreDom(prior));
    const storage = memoryStorage(), { values } = storage;
    const make = ({ room = [1, ''], scope = PREVIEW_SCOPE, kept = () => null } = {}) => createChatPreview({ feed: mount, room, scope,
        label: senderLabel, storage: () => storage, mutate: fn => fn(), kept });
    // A room's stored entry text, and it parsed: when the room was read and its rows.
    const raw = (room = [1, ''], scope = PREVIEW_SCOPE) => values.get(previewEntryKey(scope, room));
    const entry = (room, scope) => (raw(room, scope) === undefined ? undefined : JSON.parse(raw(room, scope)));
    const rows = (room, scope) => entry(room, scope)?.rows;
    const family = () => [...values.keys()].filter(key => key.startsWith(PREVIEW_FAMILY));
    const size = () => family().reduce((sum, key) => sum + key.length + values.get(key).length, 0);
    const painted = () => mount.children.find(node => node.dataset.savedPreview === 'true');
    return { mount, values, storage, make, raw, entry, rows, family, size, painted };
}
const row = (id, text, extra = {}) => ({ role: 'assistant', text, history_id: id, ...extra });

test('a new room instance paints only the saved words under their speakers: no controls, task facts or credentials', t => {
    const f = fixture(t), first = f.make();
    first.save([row('chat:1', 'An exact saved answer', { task_id: 'ended-work', cancelable: true,
        review_projection: { findings: ['not saved'] }, api_key: 'not saved', attachments: [{ name: 'file.pdf' }] })]);
    first.destroy();
    assert.doesNotMatch(f.raw(), /ended-work|cancelable|api_key|findings|file\.pdf/);
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
    assert.doesNotMatch(f.raw(), /another-window-private|sender_session_id|initiator|source/);
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

test('room, Project and data profile never borrow each other\'s text', t => {
    const f = fixture(t);
    f.make().save([row('chat:1', 'Main words')]);
    assert.equal(f.make({ room: [2, ''] }).show(), false);
    assert.equal(f.make({ room: [1, 'other-project'] }).show(), false);
    assert.equal(f.make({ scope: OTHER_PROFILE }).show(), false);
    assert.equal(f.make().show(), true);
});

test('the document proves its data profile: its own rooms paint, another profile and a document proving none open cold', t => {
    const f = fixture(t), doc = f.mount.ownerDocument;
    const meta = (profile, appearance = 'fedcba9876543210') => (selector) => {
        if (selector === 'meta[name="ouroboros-chat-preview-scope"]') return profile === null ? null : { content: profile };
        if (selector === 'meta[name="ouroboros-appearance-scope"]') return appearance === null ? null : { content: appearance };
        return null;
    };
    const open = () => createChatPreview({ feed: f.mount, room: [1, ''], label: senderLabel, mutate: fn => fn(),
        storage: () => f.storage });
    doc.querySelector = meta(PREVIEW_SCOPE);
    const first = open();
    first.save([row('chat:1', 'Profile words')]);
    first.destroy();
    const again = open();
    assert.equal(again.show(), true, 'the same proven profile paints its room');
    again.destroy();
    doc.querySelector = meta(OTHER_PROFILE);
    const other = open();
    assert.equal(other.show(), false, 'another profile is not this room');
    other.save([row('chat:2', 'Its own words')]);
    other.destroy();
    const texts = scope => f.rows([1, ''], scope).map(item => item.text);
    assert.deepEqual(texts(PREVIEW_SCOPE), ['Profile words'], 'each profile keeps its own');
    assert.deepEqual(texts(OTHER_PROFILE), ['Its own words']);
    // An installation scope, an older document and a malformed value are no profile proof: the room keeps to its
    // canonical history and the store gains nothing from it.
    const saved = JSON.stringify([...f.values]);
    for (const scope of [null, 'not-a-digest']) {
        doc.querySelector = meta(scope);
        const cold = open();
        assert.equal(cold.show(), false, `no proof (${scope})`);
        cold.save([row('chat:3', 'Never saved')]);
        cold.append([row('chat:4', 'Never appended')]);
        cold.destroy();
    }
    assert.equal(JSON.stringify([...f.values]), saved, 'no proof writes nothing');
    assert.doesNotMatch(saved, /Never saved|Never appended/);
});

test('only conversation rows with identity enter; refusals and unreadable streams keep the entry; an empty read paints nothing', t => {
    const f = fixture(t), preview = f.make();
    preview.save([row('chat:1', 'Kept answer'), row('progress:2', 'narration', { is_progress: true }),
        row('chat:3', 'card row', { system_type: 'task_summary' }), row('', 'no identity'), row('chat:4', ''),
        { ...row('chat:5', 'a quiz'), system_type: 'quiz' }]);
    assert.deepEqual(f.rows().map(item => item.text), ['Kept answer']);
    const saved = f.raw();
    preview.save([row('chat:9', 'wrong')], { reason_code: 'history_source_unavailable' });
    preview.save([], { window: { complete: false, truncated_by: ['chat_source_unavailable'] } });
    assert.equal(f.raw(), saved);
    preview.save([row('chat:9', 'bounded')], { window: { complete: false, truncated_by: ['quota'] } });
    assert.deepEqual(f.rows().map(item => item.text), ['bounded'], 'an incomplete window is still a window');
    preview.save([], { window: { complete: true } });
    assert.deepEqual(f.rows(), [], 'read empty: the room keeps an empty entry');
    assert.equal(f.make().show(), false);
});

test('arrivals merge by identity in place; a room without an entry keeps no fragment', t => {
    const f = fixture(t), preview = f.make();
    preview.append([row('chat:1', 'Arrived before any newest read')]);
    assert.deepEqual(f.family(), []);
    preview.save([]);
    preview.append([row('chat:1', 'Arrived since opening')]);
    preview.append([row('chat:1', 'Arrived since opening')]);
    preview.append([row('chat:1', 'Edited in place'), row('chat:2', 'Next', { role: 'user', sender_label: 'Mira', source: 'telegram' }),
        row('progress:3', 'narration', { is_progress: true })]);
    assert.deepEqual(f.rows().map(item => [item.id, item.text, item.speaker]),
        [['chat:1', 'Edited in place', 'Ouroboros'], ['chat:2', 'Next', 'Mira']]);
    // The entry gave way to rooms read since: what arrives next waits for the room's next newest window.
    f.values.delete(previewEntryKey(PREVIEW_SCOPE, [1, '']));
    preview.append([row('chat:3', 'Arrived after it gave way')]);
    assert.equal(f.raw(), undefined, 'no fragment stands for the room');
});

test('each row goes where the feed puts its bubble: by time, a delayed arrival at its time, a missing time last', t => {
    // chat_render_batch.js insertTimelineNode draws rows by their time, and a delayed delivery keeps its earlier time
    // (gateway/history_paging.py latest_arrival): stream order is not the conversation's order. A row without a time is
    // stamped with the moment it is drawn (chat.js addMessage), after everything before it.
    const f = fixture(t), preview = f.make();
    const at = minute => `2026-11-03T12:${String(minute).padStart(2, '0')}:00Z`;
    const ids = () => f.rows().map(item => item.id);
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
    const T = '2026-11-03T12:00:00Z', ids = () => f.rows().map(item => item.id);
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

test('a Project the registry says is going away takes its saved text in the profile that said so; every other room keeps its own', t => {
    // Astra 2: two data profiles of this origin can hold the same Project id; one profile's word takes its own room's.
    const f = fixture(t);
    f.make({ room: [1, ''] }).save([row('chat:1', 'Main words')]);
    f.make({ room: [100, 'gate-0'] }).save([row('chat:1', 'Words of the deleted Project')]);
    f.make({ room: [100, 'gate-0'], scope: OTHER_PROFILE }).save([row('chat:1', 'Its words in another profile')]);
    f.make({ room: [101, 'gate-1'] }).save([row('chat:1', 'Another Project keeps these')]);
    const gone = previewEntryKey(PREVIEW_SCOPE, [100, 'gate-0']);
    const others = [...f.values].filter(([key]) => key !== gone);
    forgetProjectPreviews(['gate-0'], PREVIEW_SCOPE, f.storage);
    assert.deepEqual([...f.values], others, 'the others keep their entries byte for byte, the same id in another profile too');
    assert.doesNotMatch(JSON.stringify([...f.values]), /deleted Project/);
    assert.equal(f.make({ room: [100, 'gate-0'] }).show(), false);
    assert.equal(f.make({ room: [100, 'gate-0'], scope: OTHER_PROFILE }).show(), true);
    assert.equal(f.make({ room: [101, 'gate-1'] }).show(), true);
    // A word for no proven profile takes nothing; nothing to forget writes nothing; storage refused throws nothing.
    const before = [...f.values];
    const writes = [];
    const watching = { ...f.storage, setItem: (...args) => writes.push(args), removeItem: (...args) => writes.push(args) };
    for (const scope of ['', 'installation', undefined]) forgetProjectPreviews(['gate-0'], scope, watching);
    forgetProjectPreviews(['unknown', ''], OTHER_PROFILE, watching);
    forgetProjectPreviews([], PREVIEW_SCOPE, { getItem() { throw new Error('not even read'); } });
    assert.deepEqual([writes, [...f.values]], [[], before]);
    assert.doesNotThrow(() => forgetProjectPreviews(['gate-1'], PREVIEW_SCOPE, { getItem() { throw new Error('denied'); } }));
});

test('a list that says it is complete under its profile takes that profile\'s rooms of the Projects it omits, saved before it was asked', t => {
    // Astra 1, 2 (owner 2A «копии надо стирать … при удалении проекта»): only an answer that says it is complete
    // (`complete: true`) and names its data profile (`profile_scope`) proves a Project gone, for that profile. Main,
    // another profile's rooms and a room saved after the list was asked (a Project created since) stay.
    const f = fixture(t);
    t.mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
    f.make({ room: [1, ''] }).save([row('chat:1', 'Main words')]);
    f.make({ room: [100, 'gone'] }).save([row('chat:1', 'Words of a Project deleted elsewhere')]);
    f.make({ room: [101, 'going'] }).save([row('chat:1', 'Words of a Project being deleted')]);
    f.make({ room: [102, 'kept'] }).save([row('chat:1', 'A listed Project keeps these')]);
    f.make({ room: [100, 'gone'], scope: OTHER_PROFILE }).save([row('chat:1', 'Another profile keeps its own')]);
    t.mock.timers.tick(1000);
    const asked = Date.now();
    t.mock.timers.tick(1000);
    f.make({ room: [103, 'new'] }).save([row('chat:1', 'A Project created after the list was asked')]);
    const keys = () => f.family().map(key => key.slice(PREVIEW_KEY.length + 1));
    const listed = [{ id: 'kept', lifecycle: 'active' }, { id: 'going', lifecycle: 'deleting' }];
    const all = keys();
    // No proof of absence: an incomplete snapshot that still lists its healthy neighbour, an older core's answer (neither
    // word), a complete one naming no profile or no proven one, one with no list, none at all.
    for (const answer of [{ projects: listed, complete: false, profile_scope: PREVIEW_SCOPE }, { projects: listed },
        { projects: listed, complete: true }, { projects: listed, complete: true, profile_scope: 'installation' },
        { projects: listed, complete: 'true', profile_scope: PREVIEW_SCOPE }, { complete: true, profile_scope: PREVIEW_SCOPE },
        undefined]) {
        forgetUnlistedProjectPreviews(answer, asked, f.storage);
        assert.deepEqual(keys(), all, `${JSON.stringify(answer)} takes nothing`);
    }
    // Another profile's complete list takes its own room alone; this profile's, the two of its Projects it omits.
    forgetUnlistedProjectPreviews({ projects: [], complete: true, profile_scope: OTHER_PROFILE }, asked, f.storage);
    assert.deepEqual(all.filter(key => !keys().includes(key)), [JSON.stringify([OTHER_PROFILE, [100, 'gone']])]);
    forgetUnlistedProjectPreviews({ projects: listed, complete: true, profile_scope: PREVIEW_SCOPE }, asked, f.storage);
    assert.deepEqual(keys(), ['', JSON.stringify([PREVIEW_SCOPE, [1, '']]), JSON.stringify([PREVIEW_SCOPE, [102, 'kept']]),
        JSON.stringify([PREVIEW_SCOPE, [103, 'new']])]);
    assert.doesNotThrow(() => forgetUnlistedProjectPreviews({ projects: [], complete: true, profile_scope: PREVIEW_SCOPE }, 1,
        { getItem() { throw new Error('denied'); } }));
});

test('v2 rooms moved into entries were saved before any list was asked: a complete list still takes the ones it omits', t => {
    // Astra 7: when a list's answer is the first reader of the family, the v2 record moves then; its rooms keep their
    // order below any time this page could have asked, so that answer, or any asked after, takes the Projects it omits.
    const f = fixture(t);
    t.mock.timers.enable({ apis: ['Date'], now: 2000 });
    const words = text => [{ id: 'chat:1', role: 'user', text, speaker: 'You' }];
    const v2 = () => f.values.set(PREVIEW_KEY, JSON.stringify({ v: 2, rooms: [
        { key: JSON.stringify([PREVIEW_SCOPE, [100, 'gone']]), rows: words('Words of a Project deleted since') },
        { key: JSON.stringify([PREVIEW_SCOPE, [102, 'kept']]), rows: words('A listed Project keeps these') },
        { key: JSON.stringify([PREVIEW_SCOPE, [1, '']]), rows: words('Main words') }] }));
    const list = { projects: [{ id: 'kept', lifecycle: 'active' }], complete: true, profile_scope: PREVIEW_SCOPE };
    const projects = () => f.family().filter(key => key !== PREVIEW_KEY)
        .map(key => JSON.parse(key.slice(PREVIEW_KEY.length + 1))[1][1]);
    v2();
    const asked = Date.now();
    t.mock.timers.tick(100);
    forgetUnlistedProjectPreviews(list, asked, f.storage);  // its answer moves the record at 2100
    assert.deepEqual(projects(), ['kept', ''], 'asked before the record moved');
    clearChatPreviews(f.storage);
    v2();
    assert.equal(f.make({ room: [1, ''] }).show(), true, 'moved by a show');
    t.mock.timers.tick(100);
    forgetUnlistedProjectPreviews(list, Date.now(), f.storage);
    assert.deepEqual(projects(), ['kept', ''], 'asked after the record moved');
});

test('a refusal anywhere in this document takes every entry and every painted copy, selection or not; nothing else does', t => {
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
    assert.equal(f.family().length, 3, 'the record and both rooms\' entries');
    assert.equal(previewsRefused({ status: 401 }, f.storage), true);
    assert.equal(copies(), 0, 'every room\'s copy goes, the held one too');
    assert.deepEqual(f.family(), []);
    // A room's own refused read does the same for its neighbours; a destroyed preview is left alone.
    f.make({ room: [1, ''] }).save([row('chat:1', 'Again')]);
    const again = f.make({ room: [1, ''] }), gone = f.make({ room: [1, ''] });
    again.show();
    gone.destroy();
    const other = f.make({ room: [2, ''] });
    other.refused({ status: 403 });
    assert.equal(copies(), 0);
    assert.deepEqual(f.family(), []);
});

test('storage refusal, corrupt data and unsupported versions open cold and leave the room working', t => {
    const f = fixture(t), entry = previewEntryKey(PREVIEW_SCOPE, [1, '']);
    const room = (rows) => JSON.stringify(JSON.stringify([PREVIEW_SCOPE, [1, '']])) + `,"rows":${JSON.stringify(rows)}`;
    for (const [record, text] of [['{', undefined], ['{"v":4}', '{"read":1,"rows":[]}'], [`{"v":2,"rooms":[],"pad":"${'x'.repeat(200000)}"}`, undefined],
        [`{"v":1,"rooms":[{"key":${room([{ id: 'chat:1', role: 'user', text: 'x', speaker: 'You' }])}}]}`, undefined],
        [`{"v":2,"rooms":[{"key":${room([{ id: 'chat:1', role: 'intruder', text: 'x', speaker: 'y' }])}}]}`, undefined],
        ['{"v":3}', '{'], ['{"v":3}', `{"read":1,"rows":[{"id":"chat:1","role":"user","text":"${'x'.repeat(ROOM_CHARS)}","speaker":"You"}]}`],
        ['{"v":3}', '{"read":1,"rows":[{"id":"chat:1","role":"intruder","text":"x","speaker":"y"}]}']]) {
        f.values.set(PREVIEW_KEY, record);
        if (text !== undefined) f.values.set(entry, text);
        assert.equal(f.make().show(), false, `${record.slice(0, 40)} ${String(text).slice(0, 40)}`);
    }
    const denied = createChatPreview({ feed: f.mount, room: [1, ''], scope: PREVIEW_SCOPE, label: senderLabel,
        mutate: fn => fn(), storage: () => { throw new Error('denied'); } });
    assert.equal(denied.show(), false);
    assert.doesNotThrow(() => denied.save([row('chat:1', 'x')]));
    assert.doesNotThrow(() => denied.failed({ status: 401 }));
    assert.doesNotThrow(() => clearChatPreviews({ removeItem() { throw new Error('denied'); } }));
    // A store that takes no writes (a full origin, a refused private window) keeps no older text for a newer read.
    const g = fixture(t);
    g.make().save([row('chat:1', 'Read before the store filled')]);
    const full = { ...g.storage, setItem() { throw Object.assign(new Error('full'), { name: 'QuotaExceededError' }); } };
    createChatPreview({ feed: g.mount, room: [1, ''], scope: PREVIEW_SCOPE, label: senderLabel, mutate: fn => fn(),
        storage: () => full }).save([row('chat:2', 'Read once it was full')]);
    assert.equal(g.raw(), undefined, 'the room opens cold rather than with its older words');
});

test('more than eight rooms keep their words; past the total the rooms saved least recently give way first', t => {
    const f = fixture(t);
    t.mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
    const read = (room, rows) => { f.make({ room }).save(rows); t.mock.timers.tick(1000); };
    // Twelve small rooms all keep their words (the one v2 record kept eight).
    for (let i = 1; i <= 12; i += 1) read([i, ''], [row('chat:1', `room ${i}`)]);
    for (let i = 1; i <= 12; i += 1) assert.equal(f.make({ room: [i, ''] }).show(), true, `room ${i}`);
    // Rooms that fill their own share (five 19,000-character rows each, about 95,000): every write stays in the total,
    // and the rooms that stand are the ones saved most recently.
    const text = 'long exact passage '.repeat(1000), order = [], sizes = new Map();
    const big = Array.from({ length: 5 }, (_, j) => row(`chat:${j}`, text, { role: 'user' }));
    for (let i = 1; i <= 12; i += 1) {
        read([i, 'p'], big);
        const key = previewEntryKey(PREVIEW_SCOPE, [i, 'p']);
        order.push(key);
        sizes.set(key, key.length + f.values.get(key).length);
        assert.ok(f.size() <= TOTAL_CHARS, `${f.size()} characters after room ${i}`);
    }
    const standing = order.filter(key => f.values.has(key)), gone = order.filter(key => !f.values.has(key));
    assert.ok(standing.length > 8 && gone.length > 0, `${standing.length} big rooms stand, ${gone.length} gave way`);
    assert.deepEqual(order.slice(-standing.length), standing, 'the rooms saved most recently stand');
    assert.equal(f.family().some(key => key.endsWith(',""]]')), false, 'every small room, saved before them, gave way first');
    // Only as much gives way as the total needs: the last room that went would not have fit beside them.
    assert.ok(f.size() + sizes.get(gone.at(-1)) > TOTAL_CHARS);
    // A room read (and so saved) again moves to the front: the next room to give way is the one saved least recently after it.
    const [oldest, second] = standing;
    read(JSON.parse(oldest.slice(PREVIEW_KEY.length + 1))[1], big);
    read([13, 'p'], big);
    assert.equal(f.values.has(oldest), true, 'read again, it stands');
    assert.equal(f.values.has(second), false, 'the one saved least recently since gave way');
    assert.ok(f.size() <= TOTAL_CHARS);
});

// Ten rooms of five 19,000-character rows each (about 95,000 characters with its key): together just under the total.
const BIG = Array.from({ length: 5 }, (_, j) => row(`chat:${j}`, 'long exact passage '.repeat(1000), { role: 'user' }));
function tenRooms(f, t) {
    t.mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
    const read = (room) => { f.make({ room }).save(BIG); t.mock.timers.tick(1000); };
    for (let i = 1; i <= 10; i += 1) read([i, 'p']);
    assert.ok(f.size() <= TOTAL_CHARS && f.family().length === 11, 'all ten stand');
    return read;
}
const entryOf = room => previewEntryKey(PREVIEW_SCOPE, room);

// Every write and removal `act` makes in the fixture's store.
function writesOf(f, act) {
    const writes = [], { setItem, removeItem } = f.storage;
    f.storage.setItem = (key, value) => { writes.push(['set', key]); setItem(key, value); };
    f.storage.removeItem = (key) => { writes.push(['remove', key]); removeItem(key); };
    try { act(); } finally { Object.assign(f.storage, { setItem, removeItem }); }
    return writes;
}

test('the order is when a room\'s copy was saved: a show or a read with nothing new writes nothing', t => {
    // Chair, review wave 2 (astra 5): each newest read saves its room, and that save is its place in the order; a show
    // or an arrivals read that brought no conversation row moves nothing and writes nothing.
    const f = fixture(t), read = tenRooms(f, t);
    const before = f.rows([2, 'p']);
    assert.deepEqual(writesOf(f, () => assert.equal(f.make({ room: [1, 'p'] }).show(), true, 'the room saved least recently')),
        [], 'a show writes nothing');
    t.mock.timers.tick(1000);
    assert.deepEqual(writesOf(f, () => f.make({ room: [2, 'p'] }).append([row('progress:9', 'narration only',
        { is_progress: true })])), [], 'nothing new writes nothing');
    t.mock.timers.tick(1000);
    read([11, 'p']);
    assert.deepEqual([1, 2, 3, 4].map(i => f.values.has(entryOf([i, 'p']))), [false, true, true, true],
        'the room saved least recently gave way, and only it, shown or not');
    assert.deepEqual(f.rows([2, 'p']), before, 'a room read with nothing new keeps its rows');
    assert.ok(f.size() <= TOTAL_CHARS);
    // A room without an entry notes nothing: it keeps no fragment.
    f.make({ room: [12, 'p'] }).append([]);
    assert.equal(f.values.has(entryOf([12, 'p'])), false);
});

test('a show and what follows it never undo another tab\'s erasure, nor put older words over its newer save', t => {
    // Astra 5: a write of this room between its read and what follows would bring back an entry another tab removed
    // since (its word that the Project is gone) or put this tab's older words over that tab's newer save.
    for (const [other, left] of [['removes the entry', undefined], ['saves newer words', ['Newer words of another tab']]]) {
        const f = fixture(t), key = entryOf([2, 'p']);
        f.make({ room: [2, 'p'] }).save([row('chat:1', 'Read by this tab')]);
        const act = () => (left ? f.make({ room: [2, 'p'] }).save([row('chat:2', left[0])]) : f.values.delete(key));
        // The other tab acts once, between this tab's read of the entry and the first write this tab makes to it.
        let acted = false;
        const { setItem } = f.storage;
        f.storage.setItem = (name, value) => { if (name === key && !acted) { acted = true; act(); } setItem(name, value); };
        const room = f.make({ room: [2, 'p'] });
        assert.equal(room.show(), true, `${other}: painted from this tab's read`);
        if (!acted) { acted = true; act(); }
        room.append([]);
        room.append([row('progress:3', 'narration only', { is_progress: true })]);
        room.retire();
        room.destroy();
        f.storage.setItem = setItem;
        assert.deepEqual(f.rows([2, 'p'])?.map(item => item.text), left, `another tab ${other}: that stands`);
    }
});

test('a room\'s own entry is written before any other gives way: one that fails takes no other room\'s', t => {
    // Grok 1: a write that does not land takes this room's older text (it never stands for a newer read) and nothing
    // else; only an origin too full to take it makes room (how much: the next test), and asks once more.
    const f = fixture(t), read = tenRooms(f, t);
    const target = entryOf([11, 'p']);  // an eleventh room: its text does not fit beside the ten
    const others = () => [...f.values].filter(([key]) => key !== target);
    const storing = setItem => createChatPreview({ feed: f.mount, room: [11, 'p'], scope: PREVIEW_SCOPE, label: senderLabel,
        mutate: fn => fn(), storage: () => ({ get length() { return f.storage.length; }, key: f.storage.key,
            getItem: f.storage.getItem, removeItem: f.storage.removeItem, setItem }) });
    const refusing = name => (key, value) => {
        if (key === target) throw Object.assign(new Error('refused'), { name });
        f.storage.setItem(key, value);
    };
    const kept = others();
    storing(refusing('SecurityError')).save(BIG);
    assert.deepEqual(others(), kept, 'a refusal that is not a full origin takes no other room\'s entry');
    assert.equal(f.values.has(target), false);
    read([11, 'p']);  // written: the room saved least recently gave way for it
    assert.equal(f.values.has(entryOf([1, 'p'])), false);
    storing(refusing('SecurityError')).save([]);
    assert.equal(f.values.has(target), false, 'a write that did not land leaves no older text for the newer read');
    read([11, 'p']);
    const stand = others();
    storing(refusing('QuotaExceededError')).save([...BIG].reverse());
    assert.equal(f.values.has(target), false, 'an origin that stays full: its older text goes too');
    assert.deepEqual(others(), stand, 'and a write that adds nothing to the origin frees nothing');
    assert.ok(f.size() <= TOTAL_CHARS);
});

// A browser's Storage that holds `capacity` characters of keys and values in all and refuses a write past it with
// QuotaExceededError, a replacement counted by what it replaces (an origin's quota, counted in characters).
function cappedStorage(capacity) {
    const storage = memoryStorage(), { values, setItem } = storage;
    storage.setItem = (key, value) => {
        let used = key.length + String(value).length;
        for (const [name, text] of values) if (name !== key) used += name.length + text.length;
        if (used > capacity) throw Object.assign(new Error('full'), { name: 'QuotaExceededError' });
        setItem(key, value);
    };
    return storage;
}

test('an origin too full for a write frees what the write adds, the rooms saved least recently first', t => {
    // Astra 6: a replacement is credited the text it replaces and a first insertion is charged its key too, so a
    // room grown by one character takes one room's place and a new room with a longer key still finds room. Each
    // origin here holds exactly what it stores.
    const { mount } = fixture(t);
    t.mock.timers.enable({ apis: ['Date'], now: 1_000_000 });
    const keys = storage => [...storage.values.keys()].filter(key => key !== PREVIEW_KEY);
    const saving = (storage, room, rows) => {
        createChatPreview({ feed: mount, room, scope: PREVIEW_SCOPE, label: senderLabel, mutate: fn => fn(),
            storage: () => storage }).save(rows);
        t.mock.timers.tick(1000);
    };
    const filled = (rooms) => {
        const storage = cappedStorage(Infinity);
        for (const [room, rows] of rooms) saving(storage, room, rows);
        const capped = cappedStorage([...storage.values].reduce((sum, [key, text]) => sum + key.length + text.length, 0));
        for (const [key, text] of storage.values) capped.values.set(key, text);
        return capped;
    };
    // Nine rooms of 10,000 characters, then one of 100,000 saved last; that one grows by a character.
    const small = Array.from({ length: 9 }, (_, i) => [[i + 1, 'p'], [row('chat:1', 'x'.repeat(10000))]]);
    const grown = filled([...small, [[10, 'p'], [row('chat:1', 'y'.repeat(100000))]]]);
    saving(grown, [10, 'p'], [row('chat:1', 'y'.repeat(100001))]);
    assert.equal(JSON.parse(grown.values.get(entryOf([10, 'p']))).rows[0].text.length, 100001, 'written');
    assert.deepEqual(keys(grown), [...small.slice(1).map(([room]) => entryOf(room)), entryOf([10, 'p'])],
        'one room gave way for one character: the one saved least recently');
    // Two rooms of equal text; a new room with the same text under a longer key needs both their places.
    const words = [row('chat:1', 'z'.repeat(1000))], longer = [3, 'a Project id longer than the others'];
    const first = filled([[[1, 'a'], words], [[2, 'b'], words]]);
    saving(first, longer, words);
    assert.deepEqual(keys(first), [entryOf(longer)], 'both gave way and the new room is saved');
    assert.deepEqual(JSON.parse(first.values.get(entryOf(longer))).rows.map(item => item.text), [words[0].text]);
});

test('a room over its own share keeps its newest whole rows; a person\'s text is never cut', t => {
    const f = fixture(t), text = 'long exact passage '.repeat(1000);
    f.make().save(Array.from({ length: 100 }, (_, j) => row(`chat:${j}`, text, { role: 'user' })));
    const kept = f.rows();
    assert.ok(f.raw().length <= ROOM_CHARS, `${f.raw().length} characters`);
    assert.ok(kept.length >= 9 && kept.every(item => item.text === text), `${kept.length} whole rows`);
    assert.deepEqual(kept.map(item => item.id), Array.from({ length: kept.length }, (_, j) => `chat:${100 - kept.length + j}`),
        'its newest rows, in order');
    // One row longer than the share is not cut to fit: the room keeps none of it.
    f.make().save([row('chat:200', 'x'.repeat(ROOM_CHARS))]);
    assert.deepEqual(f.rows(), []);
});

test('no room keeps less than v2 did: the longest answer its one record could hold paints, moved or read anew', t => {
    // Astra 4: v2 let a room alone fill its one 200,000-character record, its key and frame counted. That answer, moved
    // into its entry or read again, is kept whole and paints.
    const f = fixture(t), key = JSON.stringify([PREVIEW_SCOPE, [1, '']]);
    const shell = JSON.stringify({ v: 2, rooms: [{ key, rows: [{ id: 'chat:1', role: 'assistant', text: '', speaker: 'Ouroboros' }] }] });
    const answer = 'x'.repeat(ROOM_CHARS - shell.length);
    f.values.set(PREVIEW_KEY, shell.replace('"text":""', `"text":"${answer}"`));
    assert.equal(f.values.get(PREVIEW_KEY).length, ROOM_CHARS, 'a record v2 could write');
    const moved = f.make();
    assert.equal(moved.show(), true, 'moved into its entry, it paints');
    assert.equal(f.painted().children[1].children[1].textContent, answer, 'whole');
    moved.destroy();
    f.make().save([row('chat:2', answer)]);
    assert.deepEqual(f.rows().map(item => item.text), [answer], 'read anew, it is kept whole');
});

test('the one v2 record moves into entries once, the order it kept by time read', t => {
    const f = fixture(t);
    t.mock.timers.enable({ apis: ['Date'], now: 5_000_000 });
    const saved = text => [{ id: 'chat:1', role: 'user', text, speaker: 'You' }];
    f.values.set(PREVIEW_KEY, JSON.stringify({ v: 2, rooms: [
        { key: JSON.stringify([PREVIEW_SCOPE, [2, 'p']]), rows: saved('Read last') },
        { key: JSON.stringify([PREVIEW_SCOPE, [1, '']]), rows: saved('Read first') },
        { key: JSON.stringify([OTHER_PROFILE, [1, '']]), rows: [...saved('Another profile'), { id: 'chat:2', role: 'intruder' }] }] }));
    f.values.set('ouroboros.theme', 'dark');
    assert.equal(f.make({ room: [2, 'p'] }).show(), true);
    assert.equal(f.values.get(PREVIEW_KEY), '{"v":3}', 'the record names the form of entries');
    assert.deepEqual([f.rows([2, 'p']), f.rows([1, '']), f.rows([1, ''], OTHER_PROFILE)].map(rows => rows.map(item => item.text)),
        [['Read last'], ['Read first'], ['Another profile']]);
    assert.ok(f.entry([2, 'p']).read > f.entry([1, '']).read && f.entry([1, '']).read > f.entry([1, ''], OTHER_PROFILE).read);
    assert.equal(f.values.get('ouroboros.theme'), 'dark');
    // Once: later opens find the form in place and move nothing again (a show writes nothing).
    const moved = [...f.values];
    assert.equal(f.make({ room: [1, ''] }).show(), true);
    assert.deepEqual([...f.values], moved);
});

test('entries standing without the record are erased before any is read: an older version\'s eraser took the record', t => {
    // Every earlier sign-out page, sign-in clear and refusal removes PREVIEW_KEY alone.
    const f = fixture(t);
    f.make({ room: [1, ''] }).save([row('chat:1', 'Main words')]);
    f.make({ room: [2, 'p'] }).save([row('chat:1', 'Project words')]);
    f.values.set('ouroboros.theme', 'dark');
    f.values.delete(PREVIEW_KEY);
    assert.equal(f.make({ room: [2, 'p'] }).show(), false);
    assert.deepEqual([...f.values], [['ouroboros.theme', 'dark']], 'their words are gone, not hidden');
});

// The script the core's sign-in page runs on every sign-out (server_auth.py `_END_OF_SESSION_SCRIPT`, plain literals).
async function signInPageScript() {
    const source = await readFile(new URL('../../ouroboros/server_auth.py', import.meta.url), 'utf8');
    const literal = /^_END_OF_SESSION_SCRIPT = \(((?:\s*'[^'\\]*')+)\)/m.exec(source);
    assert.ok(literal, 'server_auth.py states the script as adjacent plain literals');
    return [...literal[1].matchAll(/'([^'\\]*)'/g)].map(match => match[1]).join('');
}

test('every eraser takes the record and every room\'s entry, and nothing else the browser keeps', async t => {
    const script = await signInPageScript();
    const erasers = {
        'the sign-in clear': f => clearChatPreviews(f.storage),
        'a refused read outside the room': f => previewsRefused({ status: 401 }, f.storage),
        'a room\'s refused page': f => f.make({ room: [9, ''] }).refused({ status: 403 }),
        'a room\'s refused newest read': f => f.make({ room: [9, ''] }).failed({ status: 401 }),
        'the sign-in page every sign-out reaches': f => vm.runInNewContext(script, { localStorage: f.storage }),
    };
    for (const [name, erase] of Object.entries(erasers)) {
        const f = fixture(t);
        f.values.set('ouroboros.theme', 'light');
        f.values.set('ouro_chat_draft:1', 'A draft');
        for (const [room, scope] of [[[1, ''], PREVIEW_SCOPE], [[2, 'p'], PREVIEW_SCOPE], [[1, ''], OTHER_PROFILE]]) {
            f.make({ room, scope }).save([row('chat:1', 'Private words')]);
        }
        f.values.set(previewEntryKey(PREVIEW_SCOPE, [3, 'listed nowhere']), '{"read":1,"rows":[]}');
        erase(f);
        assert.deepEqual([...f.values], [['ouroboros.theme', 'light'], ['ouro_chat_draft:1', 'A draft']], name);
    }
});

test('a canonical replay ends the opening: a later show paints nothing', t => {
    const f = fixture(t);
    f.make().save([row('chat:1', 'Saved words')]);
    const preview = f.make();
    assert.equal(preview.shown(), false);
    preview.retire();  // the room's first canonical replay, before any copy was shown
    assert.equal(preview.show(), false);
    assert.equal(f.painted(), undefined);
    const shown = f.make();
    assert.equal(shown.show(), true);
    assert.equal(shown.shown(), true, 'a painted copy stands');
    shown.retire();
    assert.equal(shown.shown(), false);
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

test('a refused session discards every entry and the painted copy at once, selection or not; other failures only label it', t => {
    const f = fixture(t), w = selectionWorld(f);
    f.make().save([row('chat:1', 'Private words')]);
    const preview = f.make();
    preview.show();
    preview.failed({ status: 503 });
    assert.equal(f.painted().children[0].textContent, 'Saved text preview · not current');
    assert.ok(f.raw(), 'a transient failure erases nothing');
    w.select(f.painted().children[1].children[1]);
    preview.retire();
    assert.ok(f.painted(), 'held by its selection');
    preview.failed({ status: 401 });
    assert.equal(f.painted(), undefined);
    assert.deepEqual(f.family(), []);
    w.changed();
    assert.equal(f.painted(), undefined, 'nothing comes back');
    const other = f.make();
    other.save([row('chat:2', 'Again')]);
    other.show();
    other.failed({ status: 403 });
    assert.equal(f.painted(), undefined);
    assert.deepEqual(f.family(), []);
});

test('any read the room makes, refused, discards a held copy at once; a page that merely failed changes nothing here', t => {
    // chat.js calls `refused` for every history read (a page, ↓, arrivals), `failed` for its newest read alone.
    const f = fixture(t), w = selectionWorld(f);
    f.make().save([row('chat:1', 'Held words')]);
    const preview = f.make();
    preview.show();
    w.select(f.painted().children[1].children[1]);
    preview.retire();
    const note = f.painted().children[0].textContent, saved = JSON.stringify([...f.values]);
    for (const status of [500, 503, 404, undefined]) preview.refused({ status });
    assert.equal(f.painted().children[0].textContent, note, 'a failed page is no statement about the saved text');
    assert.equal(JSON.stringify([...f.values]), saved);
    preview.refused({ status: 403 });
    assert.equal(f.painted(), undefined, 'selection or not');
    assert.deepEqual(f.family(), []);
});

test('destroy removes the painted copy; later calls are inert', t => {
    const f = fixture(t);
    f.make().save([row('chat:1', 'Words')]);
    const preview = f.make();
    preview.show();
    const saved = JSON.stringify([...f.values]);
    preview.destroy();
    assert.equal(f.painted(), undefined);
    assert.equal(preview.show(), false);
    preview.save([row('chat:9', 'late')]);
    preview.append([row('chat:9', 'late')]);
    assert.equal(JSON.stringify([...f.values]), saved);
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
    const rows = f.rows().map(({ at, ...words }) => words);  // `at`: the moment an untimed row is drawn
    assert.deepEqual(rows, [
        { id: 'chat:1', role: 'system', text: 'Launch › Ship · Completed', speaker: 'System' },
        { id: 'chat:2', role: 'assistant', text: 'The child found the cause.', speaker: 'Ouroboros' }]);
    assert.doesNotMatch(f.raw(), /completion_answer|Done: PR|parent_task_id|child-1/);
});

test('a host notice is saved as the line its bubble draws, never only its stored phrase (RV4 23)', t => {
    const f = fixture(t), preview = f.make();
    const line = 'The total budget is spent, so this turn did not start. Raise Total budget in Dashboard → Costs.';
    preview.save([row('chat:1', 'The total budget is spent, so this turn did not start.', { role: 'system',
        system_type: 'task_not_started', notice: { code: 'budget_exhausted', step: 'raise_total_budget', line } })]);
    assert.equal(f.rows()[0].text, line);
    preview.destroy();
});
