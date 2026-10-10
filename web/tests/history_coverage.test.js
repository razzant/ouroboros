import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatHistoryPager, historyCoverage } from '../modules/chat_history.js';

const coverage = (from, to, upper = 100, extras = {}) => ({ v: 1, view: 'room',
    upper: { chat: upper, progress: 0 }, spans: {
        chat: { from, to, chain: 'retained', gaps: [], ...extras },
        progress: { from: 0, to: 0, chain: 'empty', gaps: [] },
    } });

test('page 3 at EOF plus recent is partial, despite overlapping row identities', () => {
    const recent = coverage(80, 100);
    assert.deepEqual(historyCoverage(recent, [coverage(0, 20)]), { complete: false, gaps: true, horizonGap: true });
    assert.equal(historyCoverage(recent, [coverage(0, 20), coverage(20, 80)]).complete, true);
    assert.equal(historyCoverage(recent, [coverage(20, 80)]).gaps, false, 'contiguous tail needs only ordinary load affordance');
});

test('read gaps, unknown metadata, disabled sources and incompatible coordinates never prove complete', () => {
    const recent = coverage(80, 100);
    for (const page of [coverage(0, 80, 100, { gaps: ['invalid_json'] }),
        coverage(0, 80, 100, { chain: 'replaced-prefix' }), { ...coverage(0, 80), view: 'other-view' }, null]) {
        const state = historyCoverage(recent, [page]);
        assert.equal(state.complete, false);
        assert.equal(state.gaps, true, 'unresolved bad metadata must remain disclosed at EOF');
    }
    assert.equal(historyCoverage({ ...recent, spans: { chat: null, progress: recent.spans.progress } }, []).complete, false);
    assert.equal(historyCoverage(null, [coverage(0, 100)]).complete, false);
});

test('a complete recent read cannot certify protected rows from a replaced archive chain', () => {
    const state = historyCoverage(coverage(0, 100), [coverage(0, 100, 100, { chain: 'replaced' })]);
    assert.equal(state.complete, false);
    assert.equal(state.gaps, true);
    const tainted = coverage(0, 50, 100, { gaps: ['invalid_json'] });
    assert.equal(historyCoverage(coverage(50, 100), [tainted]).gaps, true);
    assert.equal(historyCoverage(coverage(0, 100), [tainted]).complete, true, 'a clean reread heals its bytes');
});

test('a newer recent horizon does not heal bytes beyond the frozen old snapshot', () => {
    const recent = coverage(180, 200, 200);
    assert.equal(historyCoverage(recent, [coverage(0, 100)]).horizonGap, true);
    assert.equal(historyCoverage(recent, [coverage(0, 100), coverage(100, 180, 200)]).complete, true);
});

test('the first write of a previously empty source does not invent a gap', () => {
    const empty = coverage(0, 0, 0, { chain: 'empty' });
    assert.equal(historyCoverage(coverage(0, 100), [empty]).complete, true);
    for (const span of [null, { ...empty.spans.chat, gaps: ['read_error'] },
        { ...empty.spans.chat, from: 20, to: 20 }]) {
        assert.equal(historyCoverage(coverage(0, 100), [{ ...empty, spans: { ...empty.spans, chat: span } }]).gaps, true);
    }
});

test('a retained empty source keeps its zero frontier when its first growth is read truncated', () => {
    const empty = coverage(0, 0, 0, { chain: 'empty' });
    const grown = (from, to) => coverage(from, to, 100, { chain: 'a' });
    assert.deepEqual(historyCoverage(grown(40, 100), [empty]), { complete: false, gaps: true, horizonGap: true },
        'bytes written below the truncated read and above the retained rows are an undelivered gap');
    assert.deepEqual(historyCoverage(grown(0, 100), [empty]), { complete: true, gaps: false, horizonGap: false },
        'a first growth read from zero is the whole source');
    assert.deepEqual(historyCoverage(grown(40, 100), [empty, grown(0, 40)]), { complete: true, gaps: false, horizonGap: false },
        'paging the new chain delivers those bytes');
    assert.deepEqual(historyCoverage(grown(40, 100), [grown(40, 100)]), { complete: false, gaps: false, horizonGap: false },
        'without a retained frontier the same tail is ordinary pagination');
});

test('a same-rowcount re-read refreshes the span it certifies', async () => {
    let current = coverage(0, 50, 100, { gaps: ['invalid_json'] });
    const pager = createChatHistoryPager({
        fetchPage: () => ({ messages: [], page_cursor: 'p', next_cursor: null, has_more: false, coverage: current }),
        applyPage() {}, releasePage() {},
    });
    await pager.latest();
    assert.equal(historyCoverage(current, pager.getState().coverage).complete, false);
    current = coverage(0, 100);
    await pager.latest();
    assert.equal(historyCoverage(current, pager.getState().coverage).complete, true);
    pager.destroy();
});

test('narration never decides coverage: older-page narration spans are ignored, a failed recent span is a gap', () => {
    const recent = coverage(80, 100);
    const olderPage = { ...coverage(0, 80), spans: { chat: coverage(0, 80).spans.chat, progress: null } };
    assert.deepEqual(historyCoverage(recent, [olderPage]), { complete: true, gaps: false, horizonGap: false });
    const failed = { ...recent, spans: { ...recent.spans, progress: { from: 0, to: 0, chain: 'empty', gaps: ['read_error'] } } };
    assert.equal(historyCoverage(failed, [olderPage]).gaps, true);
    const unknown = { ...recent, spans: { chat: recent.spans.chat } };
    assert.equal(historyCoverage(unknown, [olderPage]).gaps, true, 'an unreadable narration span stays disclosed');
});

test('rotation keeps earlier prefix witnesses; a replaced later archive cannot complete coverage', () => {
    // Each span lists rolling prefix witnesses through its trailing segments
    // (symbolic here). A page is compatible when its own last witness is still
    // listed by the newest recent read.
    const at = (from, to, upper, chain) => coverage(from, to, upper, { chain });
    const pages = [at(0, 40, 100, 'a.ab.abc'), at(40, 80, 100, 'a.ab.abc')];
    assert.equal(historyCoverage(at(80, 100, 100, 'a.ab.abc'), pages).complete, true);
    assert.deepEqual(historyCoverage(at(80, 140, 140, 'ab.abc.abcd'), pages),
        { complete: true, gaps: false, horizonGap: false }, 'the live segment rotated and a new one began');
    const replaced = historyCoverage(at(80, 140, 140, "ab'.ab'c.ab'cd"), pages);
    assert.equal(replaced.complete, false);
    assert.equal(replaced.gaps, true, 'bytes read before a later archive was replaced are disclosed');
    assert.equal(historyCoverage(at(80, 140, 140, 'ab.abc.abcd'), [at(0, 80, 100, 'abc.x')]).gaps, true,
        'only the span\'s own last witness proves its coordinates');
});
