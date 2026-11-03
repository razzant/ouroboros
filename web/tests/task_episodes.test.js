// Continue episodes (task_episodes.js; owner QB = A): the host's admitted successor is the next episode of the SAME
// card. The chain is the host's explicit fact only; every episode is one meta line; the successor's card takes the
// predecessor's place and first name; the predecessor folds away.
import assert from 'node:assert/strict';
import test, { after, before } from 'node:test';
import { admittedSuccessor, createTaskEpisodes, episodeSpan, noteEpisodeFacts, renderEpisodes } from '../modules/task_episodes.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

// The browser's text serializer, as utils.escapeHtmlText reads it back.
let priorDocument;
before(() => {
    priorDocument = globalThis.document;
    globalThis.document = { createElement: () => ({ set textContent(value) {
        this.innerHTML = String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    } }) };
});
after(() => { globalThis.document = priorDocument; });

test('only the host\'s explicit, admitted chain counts', () => {
    assert.equal(admittedSuccessor({ continued_by: { state: 'admitted', successor_task_id: 's-1' } }), 's-1');
    assert.equal(admittedSuccessor({ continuation_offer: { state: 'admitted', successor_task_id: 's-2' } }), 's-2');
    assert.equal(admittedSuccessor({ continuation_offer: { state: 'bound', successor_task_id: 's-3' } }), '',
        'a bound but unadmitted claim is not an episode yet');
    // The claim names its successor before admission (owner_continue.claim_on_predecessor): bound is pending.
    assert.equal(admittedSuccessor({ continued_by: { state: 'bound', successor_task_id: 's-4', action_nonce: 'n-4' } }), '',
        'a bound continued_by claim is not an episode either');
    assert.equal(admittedSuccessor({ continued_by: { successor_task_id: 's-5' } }), '', 'a claim without its state proves nothing');
    assert.equal(admittedSuccessor({ task_name: 'Continue: Audit' }), '', 'a name is not a chain');
    assert.equal(admittedSuccessor(null), '');
});

test('one attempt\'s facts: when it started, when and how it ended, its cause in the shared words', () => {
    let episode = noteEpisodeFacts({}, { terminal_time: { attempt: { started_at: '2026-10-10T17:33:00Z' } } });
    assert.equal(episode.started, '2026-10-10T17:33:00Z');
    episode = noteEpisodeFacts(episode, { system_type: 'task_summary', status: 'failed', ts: '2026-10-10T18:15:00Z',
        reason_code: 'provider_failure', terminal_time: { occurred_at: '2026-10-10T18:15:00Z' } });
    assert.equal(episode.ended, '2026-10-10T18:15:00Z');
    assert.equal(episode.phase, 'error');
    assert.equal(episode.cause, 'The model provider failed to answer, so the task could not finish');
    assert.deepEqual(episode.codes, [], 'a cause in words leaves nothing for the details');
    // Owner 2A: a code no sentence describes yet reads neutrally; its exact code rides for the card's details.
    const unknown = noteEpisodeFacts(episode, { system_type: 'task_summary', status: 'failed', reason_code: 'example_new_reason' });
    assert.deepEqual([unknown.cause, unknown.codes], ['The recorded reason has no description yet.', ['example_new_reason']]);
    assert.equal(noteEpisodeFacts(episode, { suggested_name: 'Check totals' }).name, 'Check totals');
    assert.equal(noteEpisodeFacts({ name: 'Retained name' }, {}).name, 'Retained name');
});

test('an episode\'s time reads as the chat clock does', () => {
    const at = (h, m, d = 10) => new Date(2026, 9, d, h, m).getTime();
    assert.equal(episodeSpan(at(17, 33), at(18, 15)), '17:33–18:15');
    assert.equal(episodeSpan(at(19, 42), at(2, 59, 11)), '19:42–02:59 next day');
    assert.equal(episodeSpan(at(19, 42), ''), 'since 19:42');
    assert.equal(episodeSpan('', ''), '');
});

function lines() {
    let html = '';
    return { hidden: true, get innerHTML() { return html; }, set innerHTML(value) { html = value; } };
}

test('one episode is not a chain; several are one meta line each, the current one owning the chip', () => {
    const el = lines();
    assert.equal(renderEpisodes(el, [{ started: 1 }]), false);
    assert.equal(el.hidden, true);
    assert.equal(renderEpisodes(el, [
        { started: new Date(2026, 9, 10, 17, 33).getTime(), ended: new Date(2026, 9, 10, 18, 15).getTime(), phase: 'error',
            cause: 'The model provider failed' },
        { started: new Date(2026, 9, 10, 19, 42).getTime(), ended: '' },
    ]), true);
    assert.equal(el.hidden, false);
    assert.match(el.innerHTML, /Episode 1<\/span> · <span class="chat-live-episode-span">17:33–18:15<\/span> · <span class="chat-live-episode-outcome" data-tone="failed">Failed<\/span><span class="chat-live-episode-cause">: The model provider failed<\/span>/);
    assert.match(el.innerHTML, /data-episode="2" data-current="1"><span class="chat-live-episode-label">Episode 2<\/span> · <span class="chat-live-episode-span">since 19:42<\/span><\/div>/,
        'the current episode shows only its time: the chip is its');
    renderEpisodes(el, [{ started: 0, phase: 'error', cause: '<b>x</b>' }, {}]);
    assert.doesNotMatch(el.innerHTML, /<b>/, 'words are text, never markup');
    renderEpisodes(el, [{ name: '<b>old</b>' }, { name: 'Continue: check totals' }]);
    assert.match(el.innerHTML, /&lt;b&gt;old&lt;\/b&gt;/);
    assert.match(el.innerHTML, /chat-live-episode-name">Continue: check totals/);
    // An earlier episode's outcome takes the chips' tones: warnings succeed, a stop is neutral, a failure red.
    renderEpisodes(el, [{ phase: 'warn' }, { phase: 'cancelled' }, { phase: 'timeout' }, {}]);
    assert.deepEqual([...el.innerHTML.matchAll(/episode-outcome" data-tone="(\w+)">([^<]+)</g)].map((m) => `${m[2]}:${m[1]}`),
        ['Done with warnings:ok', 'Stopped:neutral', 'Failed:failed']);
});

function card(id, ts, name) {
    return { groupId: id, isSubagent: false, finished: false, suggestedName: name,
        root: { dataset: { ts: String(ts) }, hidden: false }, titleEl: { textContent: name }, episodesEl: lines() };
}

test('the successor\'s card takes the work\'s first place and name; the predecessor folds away; either arrival order', () => {
    for (const successorFirst of [false, true]) {
        const records = new Map();
        const moves = [];
        const episodes = createTaskEpisodes({ records, reanchor: (record, ts) => (moves.push([record.groupId, ts]), true),
            activity: () => ({ started_at: 1760117000 }) });
        const first = card('t-1', 1000, 'Audit the projects');
        const next = card('t-2', 5000, 'Continue: Audit the projects');
        const add = (record) => { records.set(record.groupId, record); episodes.sync(record); };
        if (successorFirst) add(next);
        add(first);
        episodes.note(first, { continued_by: { state: 'admitted', successor_task_id: 't-2' }, status: 'failed',
            system_type: 'task_summary', ts: '2026-10-10T18:15:00Z' });
        if (!successorFirst) add(next);
        assert.equal(first.root.hidden, true, 'the predecessor folds away');
        assert.equal(first.root.dataset.episodeOf, 't-2');
        assert.equal(next.titleEl.textContent, 'Audit the projects', 'the work keeps its first name');
        assert.equal(next.episodeName, 'Audit the projects');
        assert.deepEqual(moves.at(-1), ['t-2', 1000], 'the card stands where the work began');
        assert.equal(next.episodesEl.hidden, false);
        assert.match(next.episodesEl.innerHTML, /Episode 1.*Episode 2/s);
        assert.match(next.episodesEl.innerHTML, /chat-live-episode-name">Continue: Audit the projects/,
            'the successor keeps its own physical name separately from the work heading');
        assert.equal(episodes.sync({ groupId: 'other', isSubagent: false, root: { dataset: {} } }), false, 'not in a chain');
    }
});

test('an unnamed successor never takes the work\'s borrowed heading as its own episode name', () => {
    const records = new Map();
    const episodes = createTaskEpisodes({ records, reanchor: () => false, activity: () => null });
    const first = card('t-1', 1000, 'Audit the projects');
    const next = card('t-2', 5000, '');
    for (const record of [first, next]) { records.set(record.groupId, record); episodes.sync(record); }
    episodes.note(first, { status: 'failed', continued_by: { state: 'admitted', successor_task_id: 't-2' } });
    episodes.sync(next);  // a later sync reads the suggestedName the first one lent
    const own = () => next.episodesEl.innerHTML.split('data-episode="2"')[1];
    assert.equal(next.titleEl.textContent, 'Audit the projects', 'the work keeps its heading');
    assert.doesNotMatch(own(), /chat-live-episode-name/, 'the successor has no name of its own yet');
    episodes.rename(next, 'Continue: Audit the projects');
    assert.match(own(), /chat-live-episode-name">Continue: Audit the projects/);
    assert.equal(next.titleEl.textContent, 'Audit the projects');
});

test('a successor with no card here is handed on once admitted; a bound claim hands on and folds nothing', () => {
    const records = new Map();
    const handed = [];
    const episodes = createTaskEpisodes({ records, reanchor: () => false, activity: () => null,
        follow: (id, predecessor) => (handed.push([id, predecessor.groupId]), false) });
    const first = card('t-1', 1000, 'Audit the projects');
    records.set('t-1', first);
    const bound = { state: 'bound', successor_task_id: 't-2', action_nonce: 'n-1' };
    episodes.note(first, { status: 'failed', continued_by: bound, continuation_offer: { ...bound, eligible: false } });
    assert.deepEqual(handed, [], 'a bound claim is pending: nothing is handed on');
    assert.equal(first.root.hidden, false, 'and its card keeps its place');
    episodes.note(first, { status: 'failed', continued_by: { ...bound, state: 'admitted' } });
    assert.deepEqual(handed, [['t-2', 't-1']], 'the admitted successor, which has no card in this chat, is handed on');
    assert.equal(first.root.hidden, false, 'nothing folds before the successor has a card');
    // The successor's card is here (its frames reach this chat): the chain merges in place, nothing is handed on.
    const next = card('t-2', 5000, 'Continue: Audit the projects');
    next.root.isConnected = true;
    records.set('t-2', next);
    episodes.note(first, { status: 'failed', continued_by: { ...bound, state: 'admitted' } });
    assert.equal(handed.length, 1);
    assert.equal(first.root.hidden, true, 'the predecessor folds into the successor\'s card');
    assert.equal(next.titleEl.textContent, 'Audit the projects');
});

test('a bound claim never folds a card, even when the claimed successor\'s card is here; its admission does', () => {
    const records = new Map();
    const episodes = createTaskEpisodes({ records, reanchor: () => false, activity: () => null });
    const first = card('t-1', 1000, 'Audit the projects');
    const next = card('t-2', 5000, 'Continue: Audit the projects');
    for (const record of [first, next]) { records.set(record.groupId, record); episodes.sync(record); }
    const claim = { state: 'bound', successor_task_id: 't-2', action_nonce: 'n-1' };
    episodes.note(first, { status: 'failed', continued_by: claim });
    episodes.sync(next);
    assert.equal(first.root.hidden, false, 'a bound claim is pending: the work\'s card stays');
    assert.equal(next.titleEl.textContent, 'Continue: Audit the projects', 'and the claimed successor takes nothing');
    assert.equal(next.episodesEl.hidden, true);
    episodes.note(first, { status: 'failed', continued_by: { ...claim, state: 'admitted' } });
    assert.equal(first.root.hidden, true, 'admitted: the predecessor folds into the successor\'s card');
    assert.equal(next.titleEl.textContent, 'Audit the projects');
});

test('an unknown episode code is inside its disclosure, never in the summary, and updates without replacing content', () => {
    const { prior } = installDom();
    try {
        const first = card('t-1', 1000, 'Audit the projects');
        const next = card('t-2', 5000, 'Continue: Audit the projects');
        first.timelineEl = document.createElement('div');
        next.episodeHistoryEl = document.createElement('div');
        const records = new Map([[first.groupId, first], [next.groupId, next]]);
        const episodes = createTaskEpisodes({ records, reanchor: () => false, activity: () => null });
        episodes.note(first, { status: 'failed', system_type: 'task_summary', reason_code: 'example_new_reason',
            continued_by: { state: 'admitted', successor_task_id: next.groupId } });
        const disclosure = first.episodeDisclosure;
        assert.match(disclosure.firstElementChild.textContent, /The recorded reason has no description yet/);
        assert.doesNotMatch(disclosure.firstElementChild.textContent, /example_new_reason/);
        const codes = disclosure.querySelector('.chat-episode-codes');
        assert.equal(codes.textContent, 'Recorded reason: example_new_reason');
        assert.equal(codes.parentNode, disclosure);
        assert.equal(codes.hidden, false);
        assert.equal(first.timelineEl.parentNode, disclosure);
        episodes.note(first, { status: 'failed', system_type: 'task_summary', reason_code: 'provider_failure' });
        assert.equal(first.episodeDisclosure, disclosure);
        assert.equal(disclosure.querySelector('.chat-episode-codes'), codes);
        assert.equal(codes.hidden, true);
        assert.equal(codes.textContent, '');
        assert.equal(first.timelineEl.parentNode, disclosure);
    } finally { restoreDom(prior); }
});


test('the original Reviews reading door follows its host, never the old work controls', () => {
    const { prior } = installDom();
    try {
        const first = card('t-1', 1000, 'Audit');
        const next = card('t-2', 5000, 'Continue: Audit');
        const third = card('t-3', 9000, 'Continue: publish');
        first.timelineEl = document.createElement('div');
        first.reviewSummaryEl = document.createElement('button');
        first.reviewsHostEl = document.createElement('div');
        first.footEl = document.createElement('div');
        const oldStop = document.createElement('button');
        oldStop.dataset.cancelRun = 't-1';
        first.footEl.append(first.reviewSummaryEl, oldStop);
        next.episodeHistoryEl = document.createElement('div');
        third.episodeHistoryEl = document.createElement('div');
        const records = new Map([first, next, third].map(record => [record.groupId, record]));
        const episodes = createTaskEpisodes({ records, reanchor: () => false, activity: () => null });
        episodes.note(first, { status: 'failed', continued_by: { state: 'admitted', successor_task_id: next.groupId } });
        const disclosure = first.episodeDisclosure;
        const door = first.reviewSummaryEl;
        const host = first.reviewsHostEl;
        assert.deepEqual([...disclosure.children].slice(-3), [first.timelineEl, door, host]);
        assert.equal(oldStop.parentNode, first.footEl, 'old work control remains in its hidden physical shell');
        episodes.sync(next);
        assert.deepEqual([...disclosure.children].slice(-3), [first.timelineEl, door, host], 'sync never reorders or copies them');
        const late = document.createElement('p');
        host.appendChild(late);
        episodes.note(next, { status: 'failed', continued_by: { state: 'admitted', successor_task_id: third.groupId } });
        assert.equal(first.episodeDisclosure, disclosure);
        assert.equal(disclosure.parentNode, third.episodeHistoryEl);
        assert.equal(first.reviewSummaryEl, door);
        assert.equal(door.parentNode, disclosure);
        assert.equal(first.reviewsHostEl, host);
        assert.equal(late.parentNode, host, 'the original controller still owns late content');
        assert.equal(oldStop.parentNode, first.footEl);
    } finally { restoreDom(prior); }
});
