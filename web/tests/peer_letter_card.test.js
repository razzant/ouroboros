// A letter's card facts and placement identity, without a room: two parts with the same words keep two placement keys
// (live and replay deduplicate by them), a sub-cent amount is never rounded away, and the card's own words say a
// reserved answer, an unconfirmed Stop and a cut failure reason. The room itself is exercised by the browser stand.
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildMessageKey } from '../modules/chat_activity.js';
import { usd, letterCardView } from '../modules/peer_letter_card.js';

const fact = (over = {}) => ({ letter_id: 'L1', place_id: 'p', to: 'Velm', to_seed: 'b', text: 'Count.', title: '',
    ceiling_usd: 2, spent_usd: null, spent_final: false, state: 'accepted_running', ...over });

test('two parts of one task with the same words keep two placement keys', () => {
    const opts = (part) => ({ taskId: 't', systemType: 'peer_letter', letterPartKey: `L1:${part}` });
    const a = buildMessageKey('assistant', 'Done.', null, opts('answer'));
    const b = buildMessageKey('assistant', 'Done.', null, { ...opts('answer'), letterPartKey: 'L2:answer' });
    const q = buildMessageKey('assistant', 'Done.', null, opts('question:q1'));
    assert.notEqual(a, b);
    assert.notEqual(a, q);
    assert.equal(a, buildMessageKey('assistant', 'Done.', null, opts('answer')));
});

test('a known sub-cent amount survives the card money format and an unknown stays empty', () => {
    assert.equal(usd(0.003), '$0.0030');
    assert.equal(usd(null), '');
    assert.equal(usd(undefined), '');
});

test('the card says a reserved answer, an unconfirmed Stop and a cut reason in its own words', () => {
    const reserved = letterCardView(fact({ state: 'needs_input', question: { question_id: 'q1', text: 'Which?', delivery_state: 'reserved' } }));
    assert.match(reserved.next, /reserved an answer/);
    assert.match(reserved.next, /not confirmed yet/);
    const stopped = letterCardView(fact({ stop: 'requested' }));
    assert.equal(stopped.word, 'Velm is working on it');
    const cut = letterCardView(fact({ state: 'failed', error: 'reason', error_chars: 9000 }));
    assert.match(cut.rest, /reason cut to its first part/);
    const held = letterCardView(fact({ state: 'needs_input', question: { question_id: 'q1', text: null, held: 'peer_silent' } }));
    assert.match(held.next, /has not released the question/);
});