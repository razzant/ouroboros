import assert from 'node:assert/strict';
import test from 'node:test';
import { createSavedDraft } from '../modules/composer_draft.js';
import { memoryStorage, roomKey } from './chat_dom_fixture.js';

const open = (local, session = memoryStorage(), key = roomKey, durable = true) => createSavedDraft({
    local: () => local, session: () => session, storeKey: key, durable,
});

test('cold restore takes only the latest saved text for the same origin, profile and room', () => {
    const local = memoryStorage();
    const main = open(local);
    const projectKey = base => roomKey(base, 7);
    main.save('unfinished\nmain');
    open(local, memoryStorage(), projectKey).save('unfinished project');
    assert.equal(open(local).restore(), 'unfinished\nmain');
    assert.equal(open(local, memoryStorage(), projectKey).restore(), 'unfinished project');
    assert.equal(open(local, memoryStorage(), base => `${base}:another-profile`).restore(), null);
    assert.equal(open(memoryStorage()).restore(), null, 'another origin');
    assert.equal(open(local, memoryStorage(), roomKey, false).restore(), null, 'unproven profile');
    main.save('');
    assert.equal(open(local).restore(), null);
});

test('each open tab keeps its own text through normal reload; late Send and clear leave a different newer revision', () => {
    for (const sent of [true, false]) {
        const local = memoryStorage(), leftTab = memoryStorage(), rightTab = memoryStorage();
        let left = open(local, leftTab), right = open(local, rightTab);
        left.save('my words');
        const sentId = left.revision();
        right.save('your words');
        left = open(local, leftTab);
        right = open(local, rightTab);
        assert.equal(left.restore(), 'my words');
        assert.equal(right.restore(), 'your words');
        if (sent) left.release(sentId); else left.save('');
        assert.equal(open(local).restore(), 'your words');
        assert.notEqual(open(local, leftTab).restore(), 'my words');
    }
});

test('equal words typed again are a different revision; retiring an observed predecessor never clears its successor', () => {
    const local = memoryStorage(), first = open(local), second = open(local);
    first.save('same');
    const firstId = first.revision();
    second.save('same');
    const secondId = second.revision();
    assert.notEqual(secondId, firstId);
    first.release(firstId);
    assert.equal(open(local).restore(), 'same');
    second.release(secondId);
    assert.equal(open(local).restore(), null);
});

test('an exact Send or clear in another window cannot revive the old text after a new write', () => {
    for (const interveningReload of [false, true]) {
        for (const clear of [false, true]) {
            const local = memoryStorage(), authorTab = memoryStorage();
            open(local, authorTab).save('old words');
            const other = open(local);
            assert.equal(other.restore(), 'old words');
            if (clear) other.save(''); else other.release(other.revision());
            if (interveningReload) assert.equal(open(local, authorTab).restore(), null);
            other.save('different words');
            assert.equal(open(local, authorTab).restore(), 'different words', `clear=${clear}, reload=${interveningReload}`);
        }
    }
});

test('a displaced draft survives reload until that exact draft is sent', () => {
    const local = memoryStorage(), authorTab = memoryStorage();
    open(local, authorTab).save('old words');
    const sender = open(local);
    assert.equal(sender.restore(), 'old words');
    const sent = sender.revision();
    sender.save('independent edit');
    assert.ok(![...local.values.values()].includes('old words'), 'no displaced words in the origin store');
    assert.equal(open(local, authorTab).restore(), 'old words', 'a different edit has not sent the old words');
    sender.release(sent);
    assert.equal(open(local, authorTab).restore(), 'independent edit');
    assert.equal(open(local).restore(), 'independent edit');
    assert.equal(local.length, 3, 'Send leaves no marker for the consumed revision');
});

test('an ordinary legacy tab draft becomes the first cold draft without replacing existing cold words', () => {
    const local = memoryStorage(), tab = memoryStorage(), legacyKey = roomKey('ouro_chat_draft');
    tab.setItem(legacyKey, 'older client\nkept these words');
    assert.equal(open(local, tab).restore(), 'older client\nkept these words');
    assert.equal(open(local).restore(), 'older client\nkept these words');
    assert.equal(tab.getItem(legacyKey), null);

    const anotherTab = memoryStorage();
    anotherTab.setItem(legacyKey, 'different old tab');
    assert.equal(open(local, anotherTab).restore(), 'different old tab');
    assert.equal(anotherTab.getItem(legacyKey), 'different old tab');
    assert.equal(open(local).restore(), 'older client\nkept these words');
});

test('legacy words stay in the tab when promotion is refused or the composition is tab-only', () => {
    const tab = memoryStorage(), local = memoryStorage(), legacyKey = roomKey('ouro_chat_draft');
    tab.setItem(legacyKey, 'still in my tab');
    const put = local.setItem;
    local.setItem = (key, value) => {
        if (key === roomKey('ouro_chat_text_v1')) throw new Error('refused');
        put(key, value);
    };
    assert.equal(open(local, tab).restore(), 'still in my tab');
    assert.equal(tab.getItem(legacyKey), 'still in my tab');
    assert.equal(open(local).restore(), null);
    local.setItem = put;
    assert.equal(open(local, tab).restore({ ordinary: false }), 'still in my tab');
    assert.equal(open(local).restore(), null);
    tab.setItem(legacyKey, '/panic'); // Only storage classification; no dispatch.
    assert.equal(open(local, tab).restore(), '/panic');
    assert.equal(open(local).restore(), null);
});

test('a refused body or selector write never clears another successfully saved text', () => {
    for (const refusal of ['body', 'selector']) {
        const local = memoryStorage(), a = open(local), b = open(local);
        a.save('known');
        b.save('other tab');
        const put = local.setItem;
        local.setItem = (key, value) => {
            if ((refusal === 'body') === key.includes(':edit:')) throw new Error('refused');
            put(key, value);
        };
        a.save('not kept');
        assert.equal(open(local).restore(), 'other tab');
        a.save('');
        assert.equal(open(local).restore(), 'other tab');
    }
});

test('ordinary typing and tab reload do not retain an edit history', () => {
    const local = memoryStorage(), tab = memoryStorage();
    for (let i = 0; i < 30; i++) {
        const draft = open(local, tab);
        if (i) assert.equal(draft.restore(), `last ${i - 1}`);
        draft.save(`first ${i}`);
        draft.save(`last ${i}`);
    }
    assert.equal(open(local).restore(), 'last 29', 'only the latest write is cold text');
    assert.equal(local.length, 3, 'one body, its live marker and its selector');
    const draft = open(local, tab);
    draft.restore();
    draft.release(draft.revision());
    assert.equal(local.length, 1, 'only the inert selector remains after Send');
});

test('immediate command is kept only by its original tab under the existing rules', () => {
    const local = memoryStorage(), tab = memoryStorage();
    const draft = open(local, tab);
    draft.save('/pani');
    draft.save('/panic');
    assert.equal(open(local, tab).restore(), '/panic');
    assert.equal(open(local).restore(), null);
    draft.clearTab('/panic');
    assert.equal(open(local, tab).restore(), null);
});

test('tab-only input replaces this tab’s edit, not another tab’s passively restored draft', () => {
    const local = memoryStorage(), firstTab = memoryStorage(), secondTab = memoryStorage();
    const first = open(local, firstTab), second = open(local, secondTab);
    first.save('unsent words');
    assert.equal(second.restore(), 'unsent words');
    // Storage only: the command is classified as tab-only; nothing is dispatched.
    second.save('/panic');
    assert.equal(open(local, secondTab).restore(), '/panic');
    assert.equal(open(local, firstTab).restore(), 'unsent words');
    assert.equal(open(local).restore(), 'unsent words');

    open(local, secondTab).save('');
    assert.equal(open(local).restore(), 'unsent words', 'clearing the replacement cannot release the adopted revision');
    const reloadedFirst = open(local, firstTab);
    assert.equal(reloadedFirst.restore(), 'unsent words');
    reloadedFirst.save('/panic');
    assert.equal(open(local).restore(), null, 'the originating tab supersedes its own saved edit');
});
