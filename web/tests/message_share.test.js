// Share's seams (message_share.js, message_actions.js, chat_media.js): it applies only to a row whose host recorded its
// words, its room's hold closes what it opened (closeTransient, release, destroy), the bot route is asked for exactly
// the shown words, and Telegram's chat picker is offered only to a client that has it.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatMedia } from '../modules/chat_media.js';
import { stampNodeTimestamp } from '../modules/chat_activity.js';
import { createMessageActions, messageRow, recordAnswerWords } from '../modules/message_actions.js';
import { PREPARE_SHARE_ROUTE, SHARE, prepareTelegramShare, registerMessageShare, telegramShareAvailable } from '../modules/message_share.js';

function bubble(role, address = 'row:1@2026-11-09T10:00:00Z#aaaaaaaaaaaa') {
    const classes = new Set(['chat-bubble', role]);
    return { dataset: { rowAddress: address }, classList: { contains: (name) => classes.has(name) },
        querySelector: () => ({ textContent: 'Shown words Copy' }) };
}

test('Share applies to the Ouroboros row whose host recorded its words, and to no other row', () => {
    const registry = createMessageActions();
    const win = { addEventListener() {}, removeEventListener() {} };
    const off = registerMessageShare({ registry, win, doc: {} });
    const answer = bubble('assistant');
    const owner = bubble('user');
    const card = bubble('assistant');  // a living card or a note between works: Copy, but no recorded words
    let held = null;
    recordAnswerWords(answer, '**Exact** words', (close) => { held = close; return () => { held = null; }; });
    assert.deepEqual(registry.actionsFor(messageRow(answer)).map((action) => action.key), [SHARE]);
    assert.equal(messageRow(answer).answerWords, '**Exact** words');
    assert.equal(messageRow(answer).text, 'Shown words Copy', 'what a skill sees is unchanged');
    assert.deepEqual(registry.actionsFor(messageRow(owner)), []);
    assert.deepEqual(registry.actionsFor(messageRow(card)), []);
    assert.equal(messageRow(card).holdTransient, null);
    assert.equal(typeof messageRow(answer).holdTransient, 'function');
    assert.equal(held, null, 'nothing is held before a press');
    off();
    assert.deepEqual(registry.actionsFor(messageRow(answer)), []);
});

test('the room closes what a row opened over it: covered, the row released, or the room gone', () => {
    const media = createChatMedia({
        durableChatMediaUrl: (value) => String(value || ''), formatMsgTime: () => null, insertMessageNode() {},
        senderLabel: () => 'Ouroboros', stampNodeTimestamp,
    });
    const row = {};
    const other = {};
    const closed = [];
    media.holdTransient(row, () => closed.push('covered'));
    media.closeTransient();
    assert.deepEqual(closed, ['covered']);
    media.closeTransient();
    assert.deepEqual(closed, ['covered'], 'a closed sheet is no longer held');
    media.holdTransient(row, () => closed.push('released'));
    media.release({ contains: (node) => node === other });
    assert.deepEqual(closed, ['covered'], 'another row’s release keeps it');
    media.release({ contains: (node) => node === row });
    assert.deepEqual(closed, ['covered', 'released']);
    const release = media.holdTransient(row, () => closed.push('late'));
    release();  // the sheet closed itself first
    media.holdTransient(row, () => closed.push('destroyed'));
    media.destroy();
    assert.deepEqual(closed, ['covered', 'released', 'destroyed']);
});

test('the bot is asked for exactly the shown words, and its sentence is the failure', async () => {
    const sent = [];
    const fetchImpl = async (url, init) => {
        sent.push({ url, init });
        return new Response(JSON.stringify({ id: 'prepared-1', expiration_date: 1_900_000_000 }), { status: 200 });
    };
    assert.deepEqual(await prepareTelegramShare('Line one\n  code  ', { fetchImpl }), { id: 'prepared-1', expiresAt: 1_900_000_000 });
    assert.equal(sent[0].url, PREPARE_SHARE_ROUTE);
    assert.equal(PREPARE_SHARE_ROUTE, '/api/extensions/telegram/miniapp/prepare-share');
    assert.deepEqual(JSON.parse(sent[0].init.body), { text: 'Line one\n  code  ' }, 'the words alone, nothing else');
    const refusing = async () => new Response(JSON.stringify({ error: 'No Telegram owner is bound to this bot yet.',
        reason_code: 'owner_not_bound' }), { status: 409 });
    await assert.rejects(prepareTelegramShare('x', { fetchImpl: refusing }), /No Telegram owner is bound/);
    const empty = async () => new Response(JSON.stringify({}), { status: 200 });
    await assert.rejects(prepareTelegramShare('x', { fetchImpl: empty }), /returned no message/);
});

test('Telegram’s chat picker is offered only to a Telegram client that has it (Bot API 8.0)', () => {
    const doc = { documentElement: { dataset: {} } };
    const app = (version, share = () => {}) => ({ initData: 'signed', shareMessage: share,
        isVersionAtLeast: (wanted) => Number(version) >= Number(wanted) });
    assert.equal(telegramShareAvailable({ win: { Telegram: { WebApp: app('8.0') } }, doc }), true);
    assert.equal(telegramShareAvailable({ win: { Telegram: { WebApp: app('7.10') } }, doc }), false, 'an older client');
    assert.equal(telegramShareAvailable({ win: { Telegram: { WebApp: { ...app('8.0'), shareMessage: undefined } } }, doc }), false);
    assert.equal(telegramShareAvailable({ win: {}, doc }), false, 'outside Telegram');
});
