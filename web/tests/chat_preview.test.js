import test from 'node:test';
import assert from 'node:assert/strict';
import { createChatPreview, clearChatPreviews, PREVIEW_KEY } from '../modules/chat_preview.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

function fixture(t) {
    const { prior, mount } = installDom();
    t.after(() => restoreDom(prior));
    const values = new Map();
    const storage = { getItem: key => values.get(key), setItem: (key, value) => values.set(key, value),
        removeItem: key => values.delete(key) };
    const make = (room = [1, ''], scope = '') => createChatPreview({ feed: mount, room, scope,
        storage: () => storage, mutate: fn => fn() });
    return { mount, values, storage, make };
}
const data = (text = 'An exact saved answer') => ({ page_cursor: 'page:1', messages: [
    { role: 'assistant', text, history_id: 'chat:1', task_id: 'ended-work', cancelable: true,
        review_projection: { findings: ['not cached'] }, api_key: 'not cached' },
] });

test('a new room instance paints only source text, not cached controls or task facts', t => {
    const f = fixture(t), original = f.make();
    original.save(data()); original.destroy();
    const fresh = f.make();
    assert.equal(fresh.show(), true);
    assert.equal(f.mount.querySelectorAll('.chat-bubble').length, 1);
    assert.equal(f.mount.querySelectorAll('button').length, 0);
    assert.doesNotMatch(f.values.get(PREVIEW_KEY), /ended-work|cancelable|api_key|findings/);
    fresh.retire();
    assert.equal(f.mount.querySelectorAll('.chat-saved-preview').length, 0);
    fresh.destroy();
});

test('room, project and installation scope never borrow each other\'s preview', t => {
    const f = fixture(t); f.make().save(data());
    assert.equal(f.make([2, '']).show(), false);
    assert.equal(f.make([1, 'different-project']).show(), false);
    assert.equal(f.make([1, ''], 'another-install').show(), false);
});

test('failed and partial reads keep the previous record; a successful empty read clears it', t => {
    const f = fixture(t), preview = f.make();
    preview.save(data()); const saved = f.values.get(PREVIEW_KEY);
    preview.save({ ...data('wrong'), reason_code: 'history_source_unavailable' });
    preview.save({ messages: data('incomplete').messages });
    assert.equal(f.values.get(PREVIEW_KEY), saved);
    preview.save({ page_cursor: 'page:empty', messages: [] });
    assert.equal(f.make().show(), false);
});

test('storage failure, corrupt data and unsupported versions leave online chat usable', t => {
    const f = fixture(t);
    for (const raw of ['{', '{"v":2,"rooms":[]}', 'x'.repeat(200001), '{"v":1,"rooms":[{"key":"bad"}]}']) {
        f.values.set(PREVIEW_KEY, raw);
        assert.equal(f.make().show(), false);
    }
    const preview = createChatPreview({ feed: f.mount, room: 1, mutate: fn => fn(), storage: () => { throw Error('denied'); } });
    assert.equal(preview.show(), false);
    assert.doesNotThrow(() => preview.save(data()));
    assert.doesNotThrow(() => clearChatPreviews({ removeItem() { throw Error('denied'); } }));
});

test('the persistent record is bounded and never cuts text inside a message', t => {
    const f = fixture(t), text = 'long exact passage '.repeat(1000);
    for (let i = 0; i < 12; i++) f.make([i, '']).save({ page_cursor: 'page',
        messages: Array.from({ length: 100 }, (_, j) => ({ role: 'user', text, history_id: `chat:${j}` })) });
    const raw = f.values.get(PREVIEW_KEY), saved = JSON.parse(raw);
    assert.ok(raw.length <= 200000 && saved.rooms.length <= 8);
    assert.ok(saved.rooms.every(room => room.rows.length <= 80 && room.rows.every(row => row.text === text)));
    clearChatPreviews(f.storage);
    assert.equal(f.make().show(), false);
});
