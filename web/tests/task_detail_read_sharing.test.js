// One current task-detail read per card: Details, the census reconciliation and the Continue lookup agree on it.
// Each GET below is held by the test and answers the store as it was when the request was sent.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { READ_EPOCH, syncContinueAction } from '../modules/task_continue.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
import { projection, work } from './work_fixture.js';

const TS = '2026-11-18T09:00:00Z';
const OK = { execution: { status: 'ok' }, objective: { status: 'not_evaluated' } };
const STOPPED = { execution: { status: 'ok', task_completion: { action: 'stop', rationale: 'Not ready' } }, objective: { status: 'not_evaluated' } };
const drain = () => new Promise((resolve) => setImmediate(resolve));
const offer = { eligible: true, cause: 'author_stop' };
const admitted = (successor) => ({ eligible: false, refusal: 'already_continued', state: 'admitted', successor_task_id: successor });

/** One root card whose GET /api/tasks/{id} answers are held in `f.gets` until the test settles each one. */
function taskCard(taskId, detail, chatId = 1) {
    const f = { detail, gets: [], hold: true };
    const { prior, mount } = installDom(async (url) => {
        const own = String(url) === `/api/tasks/${taskId}`, body = f.detail;
        if (String(url) === `/api/tasks/${taskId}/continue`) {
            return { ok: true, json: async () => ({ ok: true, successor_task_id: f.successor }) };
        }
        if (!own) return { ok: true, json: async () => (String(url).startsWith('/api/chat/history')
            ? { messages: [], window: { complete: true } } : { active_direct_turns: [] }) };
        const get = { body, settle: null };
        f.gets.push(get);
        if (f.hold) await new Promise((resolve) => { get.settle = resolve; });
        if (get.fail) return { ok: false, status: 500, json: async () => ({}) };
        return { ok: true, json: async () => body };
    });
    const handlers = new Map();
    let instance;
    f.close = () => { instance?.destroy(); restoreDom(prior); };
    try {
        instance = createChatInstance({
            ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(chatId === 1 ? [] : [chatId]), unreadCount: 0 },
            updateUnreadBadge() {}, chatId, idPrefix: 'chat', mountEl: mount,
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
                isCurrent: () => true, apply() {} },
        });
        handlers.get('chat')({ chat_id: chatId, role: 'assistant', is_progress: true, task_id: taskId, ts: TS, content: 'Checking', narration: true });
    } catch (error) { f.close(); throw error; }
    f.card = walkCard(globalThis.document.byId.get('chat-messages'), taskId);
    f.chat = (data) => handlers.get('chat')({ chat_id: chatId, task_id: taskId, ts: TS, ...data });
    f.log = (data) => handlers.get('log')({ chat_id: chatId, data: { task_id: taskId, ...data } });
    f.census = (attempt) => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
        active_chat_activities: attempt === null ? [] : [{ activity_id: taskId, chat_id: chatId, project_id: '', client_message_id: '',
            kind: 'managed_task', phase: 'running', detached: false, can_resume: false, cancelable: true,
            started_at: 1795000000, task_attempt: attempt }] });
    f.stopped = () => f.log({ type: 'task_done', status: 'completed', reason_code: 'author_stop', outcome_axes: STOPPED });
    f.settle = (index, fail = false) => { f.gets[index].fail = fail; f.gets[index].settle(); return drain(); };
    f.toggle = async () => { f.card.querySelector('[data-live-disclosure]').click(); await drain(); };
    f.button = () => f.card.querySelector('[data-continue-task]');
    f.retry = () => f.card.querySelector('[data-review-hydrate-retry]');
    f.finished = () => f.card.dataset.finished === '1';
    f.facts = () => f.card.querySelector('[data-live-facts]').innerHTML;
    return f;
}
const stored = (id, extra) => ({ task_id: id, status: 'completed', reason_code: 'author_stop', task_attempt: 1, outcome_axes: STOPPED, ...extra });
const running = (id, extra) => ({ task_id: id, status: 'running', task_attempt: 1, ...extra });
const done = (id, extra) => ({ task_id: id, status: 'completed', reason_code: 'final_message', task_attempt: 1, outcome_axes: OK, ...extra });
const finish = (f) => { f.hold = false; f.gets.forEach((g) => g.settle?.()); f.close(); };
const until = async (ok, what) => {
    for (let i = 0; i < 200 && !ok(); i += 1) await new Promise((resolve) => setTimeout(resolve, 5));
    assert.ok(ok(), what);
};
// The DOM stub's click does not bubble or match data selectors; hand the review host the press as a browser would.
function pressRetry(f) {
    const button = f.retry();
    let host = button.parentElement;
    while (host && !host.listeners.get('click')?.length) host = host.parentElement;
    for (const fn of host.listeners.get('click')) fn({ target: { closest: (sel) => (sel === '[data-review-hydrate-retry]' ? button : null) } });
}

test('a Continue lookup sent before the press cannot re-enable it when an eligible answer lands', async () => {
    const id = 'share-continue-press';
    const f = taskCard(id, stored(id, { continuation_offer: offer }));
    try {
        f.stopped();  // the terminal reads the offer once
        await drain();
        await f.settle(0);
        assert.equal(f.button()?.textContent, 'Continue');
        f.card.querySelector('[data-live-disclosure]').click();  // the open card reads its stored detail
        await drain();
        assert.equal(f.gets.length, 2);
        f.successor = 'share-successor';
        await f.button().onclick({ stopPropagation() {} });  // the press: the admission
        assert.equal(f.button().textContent, 'Continued');
        await f.settle(1);  // the read sent before the press answers the offer it found
        assert.equal(f.button().textContent, 'Continued', 'an older offer cannot undo a recorded admission');
        assert.equal(f.button().disabled, true);
        // Details stored through the same read apply as ever; a read sent after the press is later state.
        f.detail = stored(id, { continuation_offer: admitted(f.successor) });
        f.card.querySelector('[data-live-disclosure]').click();
        f.card.querySelector('[data-live-disclosure]').click();
        await drain();
        await f.settle(f.gets.length - 1);
        assert.equal(f.button().textContent, 'Continued');
    } finally { finish(f); }
});

test('an offer stated by a read sent before an admission never replaces it; a later read or an unnumbered row may', () => {
    const { prior } = installDom();
    try {
        const root = document.createElement('div');
        const record = { root, groupId: 'epochs', readEpoch: 2 };
        const at = (epoch, claim) => ({ continuation_offer: claim, ...(epoch === undefined ? {} : { [READ_EPOCH]: epoch }) });
        const label = () => root.querySelector('[data-continue-task]')?.textContent;
        syncContinueAction(record, at(2, admitted('next')));
        assert.equal(label(), 'Continued');
        assert.equal(record.admittedAfter, 2);
        syncContinueAction(record, at(1, offer));
        syncContinueAction(record, at(2, offer));
        assert.equal(label(), 'Continued', 'reads sent up to the admission are older than it');
        syncContinueAction(record, at(1, { eligible: false, refusal: 'hard_limit_reached', cause: 'round_limit' }));
        assert.equal(label(), 'Continued', 'no older answer of any kind');
        syncContinueAction(record, at(1, admitted('again')));
        assert.equal(label(), 'Continued', 'an older read naming the admission still says it');
        syncContinueAction(record, at(3, offer));
        assert.equal(label(), 'Continue', 'a read sent after the admission is later state');
        syncContinueAction(record, at(undefined, admitted('final')));
        assert.equal(label(), 'Continued', 'an unnumbered row says what it says');
    } finally { restoreDom(prior); }
});

test('a terminal under open Details reads the task detail once: the Continue lookup and Details are one GET', async () => {
    const id = 'share-terminal-open';
    const f = taskCard(id, running(id));
    try {
        await f.toggle();
        await f.settle(0);  // the open card has read the running record
        f.detail = stored(id, { continuation_offer: offer });
        f.stopped();
        await drain();
        assert.equal(f.gets.length, 2, 'the offer lookup and the settled Details are one read');
        await f.settle(1);
        await drain();
        assert.equal(f.button()?.textContent, 'Continue');
        assert.equal(f.gets.length, 2, 'and the one answer satisfied both');
        assert.equal(f.retry(), null);
    } finally { finish(f); }
});

test('a closed card still looks the offer up, without a Details read', async () => {
    const id = 'share-terminal-closed';
    const f = taskCard(id, stored(id, { continuation_offer: offer }));
    try {
        f.stopped();
        await drain();
        assert.equal(f.card.dataset.expanded, '0');
        assert.equal(f.gets.length, 1);
        await f.settle(0);
        assert.equal(f.button()?.textContent, 'Continue');
        assert.equal(f.gets.length, 1, 'nothing asked for the Details');
        assert.equal(f.retry(), null);
    } finally { finish(f); }
});

test('a lookup joining a snapshot sent before the terminal earns one authoritative read, not a loop', async () => {
    const id = 'share-lookup-older';
    const f = taskCard(id, running(id));
    try {
        await f.toggle();  // Details read of the running record, still in flight
        f.detail = stored(id, { continuation_offer: offer });
        f.stopped();  // the terminal's lookup joins it
        await drain();
        assert.equal(f.gets.length, 1);
        await f.settle(0);  // a running answer cannot say the offer
        await drain();
        assert.equal(f.gets.length, 2, 'one authoritative read follows');
        await f.settle(1);
        await drain();
        assert.equal(f.button()?.textContent, 'Continue');
        assert.equal(f.gets.length, 2);
    } finally { finish(f); }
});

test('Details first, then the census drops the work: the census joins that read and an older answer earns one more', async () => {
    const id = 'share-details-then-census';
    const f = taskCard(id, running(id));
    try {
        f.census(1);
        await f.toggle();  // the Details read of the running record is in flight
        f.detail = done(id);
        f.census(null);
        await drain();
        assert.equal(f.gets.length, 1, 'the census joins the Details read for this attempt');
        await f.settle(0);  // the running snapshot cannot settle the card
        await until(() => f.gets.length === 2, 'one authoritative read follows');
        await f.settle(1);
        await until(f.finished, 'the settled read finishes the card');
        assert.equal(f.retry(), null);
        assert.equal(f.gets.length, 2);
    } finally { finish(f); }
});

test('Details first, then the census drops the work, and the one read fails: one Retry, never beside a final', async () => {
    const id = 'share-details-then-census-failed';
    const f = taskCard(id, running(id));
    try {
        f.census(1);
        await f.toggle();
        f.detail = done(id);
        f.census(null);
        await drain();
        for (let i = f.gets.length - 1; i >= 0; i -= 1) await f.settle(i, i === 0);  // the earliest read fails
        await drain();
        assert.ok(!(f.finished() && f.retry()), 'a stale Retry beside a valid final');
        assert.ok(f.retry(), 'the failed read offers its Retry');
        const before = f.gets.length;
        pressRetry(f);
        await until(() => f.gets.length === before + 1, 'Retry reads once');
        await f.settle(f.gets.length - 1);
        await until(() => f.retry() === null, 'a success clears the failure');
        f.census(null);  // the card was never told the work ended: the next snapshot asks again
        await until(() => f.gets.length === before + 2, 'the census reads once more');
        await f.settle(f.gets.length - 1);
        await until(f.finished, 'the answer settles the card');
        assert.equal(f.retry(), null);
    } finally { finish(f); }
});

test('the census read of a newer attempt is a read of its own', async () => {
    const id = 'share-attempts';
    const f = taskCard(id, running(id));
    try {
        f.census(1);
        await f.toggle();  // attempt 1 is in flight
        f.detail = done(id, { task_attempt: 2 });
        f.census(2);
        f.census(null);
        await drain();
        assert.equal(f.gets.length, 2, 'a read of attempt 1 cannot answer attempt 2');
    } finally { finish(f); }
});

test('a read sent before a newer one that already answered speaks for no one: its late error shows no Retry', async () => {
    const id = 'share-superseded';
    const f = taskCard(id, stored(id, { continuation_offer: offer }));
    try {
        f.stopped();
        await drain();
        await f.settle(0);
        f.card.querySelector('[data-live-disclosure]').click();  // Details: a read in flight
        await drain();
        f.successor = 'share-superseded-next';
        f.detail = stored(id, { continuation_offer: admitted(f.successor) });  // the press is admitted in the store
        await f.button().onclick({ stopPropagation() {} });
        // The card's own chat reads the record now (work_portal.js); the stub's root does not bubble the press to it.
        for (const fn of document.byId.get('chat-messages').listeners.get('ouro:task-continued') || []) {
            fn({ detail: { taskId: id, successorId: f.successor } });
        }
        await drain();
        assert.equal(f.gets.length, 3, 'the press asked for a read of its own');
        await f.settle(2);  // the press's read, sent after the Details read, answers first
        await f.settle(1, true);  // the Details read fails afterwards
        assert.equal(f.retry(), null, 'a late failure of an older read leaves no Retry beside the newer answer');
        assert.equal(f.button().textContent, 'Continued');
    } finally { finish(f); }
});

// The same silence for the other way a card moves on: the census or a real retry names a newer attempt while a
// read of the old one is still in flight. Its answer belongs to no one, whoever asks, and it never owes a GET.
test('a read of a retired attempt speaks for no one: its late error leaves no Retry and asks for nothing more', async () => {
    const id = 'share-retired-attempt';
    const f = taskCard(id, running(id));
    try {
        f.census(1);
        await f.toggle();                       // Details: the read of attempt 1 is in flight
        assert.equal(f.gets.length, 1);
        f.detail = running(id, { task_attempt: 2 });
        f.census(2);                            // a real retry: this card is attempt 2 now
        await drain();
        await f.settle(0, true);                // the attempt-1 read fails late
        await drain();
        assert.equal(f.retry(), null, 'a retired attempt\'s failure is not this card\'s Retry');
        assert.equal(f.gets.length, 1, 'and it earns no read of its own');
    } finally { finish(f); }
});

// The census-first order applies a proved cancellation with one read; Details-first joins that same read, so the
// fact it carries (cancel_state pending) must satisfy it too, not ask for a second GET of the same task.
test('a cancellation the joined read already proved reaches the card without a second GET', async () => {
    const id = 'share-cancel-join';
    const f = taskCard(id, running(id, { cancel_state: 'pending' }));
    try {
        f.census(1);
        await f.toggle();                       // Details: the read of the cancelling task is in flight
        f.census(null);                         // the census drops the work: it joins that same read
        await drain();
        assert.equal(f.gets.length, 1, 'the census joins the Details read for this attempt');
        await f.settle(0);
        await until(() => f.card.querySelector('.chat-live-phase')?.textContent === 'Cancelling…',
            'the proved cancellation reaches the card');
        assert.equal(f.gets.length, 1, 'a fact the answer already carries needs no second read');
        assert.equal(f.finished(), false, 'a pending cancellation is not a final');
    } finally { finish(f); }
});

// A read is not a snapshot of the moment it was sent: its GET answers from the store as it is now. So a read sent
// while the card showed attempt 1 can truthfully carry attempt 2's stored record — the newer attempt's own
// evidence. The card takes it (rl-20261121T160100: a response naming the current attempt must stay usable), and
// that answered read owes nothing more. `liveRoot` answers each held GET from the store as it is when settled.
function liveRoot(taskId) {
    const f = { detail: running(taskId), gets: [], hold: true };
    const { prior, mount } = installDom(async (url) => {
        if (String(url) === `/api/tasks/${taskId}`) {
            const get = { settle: null, fail: false };
            f.gets.push(get);
            if (f.hold) await new Promise((resolve) => { get.settle = resolve; });
            return { ok: true, json: async () => f.detail };
        }
        if (String(url).startsWith('/api/chat/history')) return { ok: true, json: async () => ({ messages: [], window: { complete: true } }) };
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    let instance;
    f.close = () => { instance?.destroy(); restoreDom(prior); };
    instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: mount,
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    handlers.get('chat')({ chat_id: 1, role: 'assistant', is_progress: true, task_id: taskId, ts: TS, content: 'Checking', narration: true });
    f.card = walkCard(globalThis.document.byId.get('chat-messages'), taskId);
    f.census = (attempt) => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
        active_chat_activities: [{ activity_id: taskId, chat_id: 1, kind: 'managed_task', phase: 'running', detached: false,
            can_resume: false, cancelable: true, started_at: 1795000000, task_attempt: attempt }] });
    f.stopped = () => handlers.get('log')({ chat_id: 1, data: { type: 'task_done', task_id: taskId, ts: TS,
        status: 'completed', reason_code: 'author_stop', outcome_axes: STOPPED } });
    f.settle = (index, fail = false) => { f.gets[index].fail = fail; f.gets[index].settle(); return drain(); };
    f.toggle = async () => { f.card.querySelector('[data-live-disclosure]').click(); await drain(); };
    f.button = () => f.card.querySelector('[data-continue-task]');
    f.retry = () => f.card.querySelector('[data-review-hydrate-retry]');
    return f;
}

test('a read sent for the retired attempt whose answer names the current attempt reaches the card', async () => {
    const id = 'share-current-evidence';
    const f = liveRoot(id);
    try {
        await f.toggle();                       // Details: the read sent while attempt 1 was shown
        assert.equal(f.gets.length, 1);
        f.detail = running(id, { task_attempt: 2 });
        f.census(2);                            // a real retry: this card is attempt 2 now
        await drain();
        f.stopped();                            // attempt 2's Stop settles through the card's own frames
        await drain();
        const asked = f.gets.length;            // the frame naming attempt 2 may ask its own read once
        assert.ok(asked === 1 || asked === 2, `the current attempt owes at most one read of its own, got ${asked}`);
        f.detail = stored(id, { task_attempt: 2, continuation_offer: offer });
        await f.settle(0);                      // the retired read answers with the CURRENT attempt's record
        assert.equal(f.button()?.textContent, 'Continue', 'what it truthfully names as current is evidence');
        assert.equal(f.retry(), null, 'a usable success leaves no failure word');
        assert.equal(f.gets.length, asked, 'the accepted answer earns no read of its own');
        for (let i = 1; i < f.gets.length; i += 1) await f.settle(i);
        assert.equal(f.gets.length, asked, 'nor does its settled sibling');
    } finally { finish(f); }
});

// The allowance is bounded by the fact the answer states, not by the moment it was sent: a response naming the
// retired attempt (or naming none) still speaks for no one. Its Stop offer must not reopen the card, and the
// still-owed detail is asked for once more by the frame that needs it — never by the retired answer itself.
test('an answer naming the retired attempt stays silent: no late offer on the card, one ask for the owed detail', async () => {
    const id = 'share-retired-offer';
    const f = taskCard(id, running(id));
    try {
        await f.toggle();
        f.detail = running(id, { task_attempt: 2 });
        f.census(2);
        await drain();
        f.stopped();
        await drain();
        f.detail = stored(id, { continuation_offer: offer });  // attempt 1: the card's retired own record
        await f.settle(0);
        assert.equal(f.button(), null, 'the retired attempt\'s offer does not reach the card');
        assert.equal(f.retry(), null, 'nor does its retired success fabricate a failure');
        for (let i = 1; i < f.gets.length; i += 1) await f.settle(i);  // the current frame's own ask, same retired answer
        assert.equal(f.button(), null, 'an answer still naming attempt 1 is evidence for no one either');
        await drain();
        const seen = f.gets.length;
        await drain();
        assert.ok(seen <= 2, `silence asks at most the one read owed to the current attempt, saw ${seen}`);
        assert.equal(f.gets.length, seen, 'a silent answer never breeds a read chain');
        assert.equal(f.button(), null);
    } finally { finish(f); }
});

// ── The child's two settle readers join the card's one read ─────────────────────────────────────────────
// A root's terminal and a complete membership census reconcile each open child's detail. Since R10B their
// answers become handover facts for the child's card, so the same attempt must go through the card's read
// (`readCardDetail`): a Details GET in flight is joined, and two GETs of one attempt can no longer answer with
// different fates — a stale Retry never sits over a Done the other reader proved.
/** Root card with one open subagent child; every GET of the child is held until the test settles it. */
function childPair(rootId, childId, childDetail, chatId = 1) {
    const f = { childDetail, gets: [], hold: true, missing: false };
    const { prior, mount } = installDom(async (url) => {
        if (String(url) === `/api/tasks/${childId}`) {
            const get = { settle: null, fail: false };
            f.gets.push(get);
            if (f.hold) await new Promise((resolve) => { get.settle = resolve; });
            if (get.fail) return { ok: false, status: 500, json: async () => ({}) };
            if (f.missing) return { ok: false, status: 404, json: async () => ({ error: 'missing' }) };
            return { ok: true, json: async () => f.childDetail };
        }
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: [], window: { complete: true } }) };
        }
        if (String(url).startsWith('/api/tasks/')) return { ok: false, status: 404, json: async () => ({ error: 'missing' }) };
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    let instance;
    f.close = () => { instance?.destroy(); restoreDom(prior); };
    try {
        instance = createChatInstance({
            ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, chatId, idPrefix: 'chat', mountEl: mount,
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
                isCurrent: () => true, apply() {} },
        });
        handlers.get('chat')({ chat_id: chatId, role: 'assistant', is_progress: true, task_id: rootId, ts: TS,
            content: 'Root working', narration: true, cancelable: true });
        handlers.get('chat')({ chat_id: chatId, role: 'assistant', is_progress: true, task_id: childId, ts: TS,
            content: 'child working', subagent_task_id: childId, parent_task_id: rootId,
            delegation_role: 'subagent', subagent_role: 'reader', subagent_event: 'running' });
    } catch (error) { f.close(); throw error; }
    const childCard = walkCard(globalThis.document.byId.get('chat-messages'), childId);
    const deep = (el, sel) => el?.querySelector?.(sel)
        || (el?.children || []).reduce((hit, kid) => hit || deep(kid, sel), null);
    f.deep = sel => deep(childCard, sel);
    f.child = () => childCard;
    f.phase = () => deep(childCard, '[data-live-phase]')?.textContent || '';
    f.childRetry = () => deep(childCard, '[data-review-hydrate-retry]');
    f.toggleChild = async () => { deep(childCard, '[data-live-disclosure]').click(); await drain(); };
    f.rootStopped = () => handlers.get('log')({ chat_id: chatId, data: { type: 'task_done', task_id: rootId,
        status: 'completed', reason_code: 'author_stop', outcome_axes: STOPPED, ts: TS } });
    f.censusRootOnly = () => instance.hydrateStateSnapshot({ supervisor_ready: true, active_chat_activities_complete: true,
        ...projection(chatId, [work({ activity_id: rootId, chat_id: chatId, started_at: 1795000000 })]),
        active_chat_activities: [{ activity_id: rootId, chat_id: chatId, kind: 'managed_task', phase: 'running',
            detached: false, can_resume: false, cancelable: true, started_at: 1795000000, task_attempt: 1 }] });
    f.settle = (index, fail = false) => { f.gets[index].fail = fail; f.gets[index].settle(); return drain(); };
    return f;
}

test('a terminal root settles its open child through the child\'s one read: joining Details, one GET proves Done', async () => {
    const root = 'pair-root-terminal', kid = 'pair-child-shared';
    const f = childPair(root, kid, running(kid));
    try {
        await f.toggleChild();                  // the child's Details read is in flight
        assert.equal(f.gets.length, 1);
        f.rootStopped();                        // the root's terminal reconciles the open child
        await drain();
        assert.equal(f.gets.length, 1, 'the reconciliation joins the Details read of the same attempt');
        f.childDetail = done(kid);
        await f.settle(0);                      // the one answer carries the child's terminal
        await until(() => f.child().dataset.finished === '1' || f.phase() === 'Done',
            'the settled child shows its terminal');
        assert.equal(f.childRetry(), null, 'no stale Retry sits over the proved Done');
        assert.equal(f.gets.length, 1, 'and the settled fact earns no further read');
    } finally { finish(f); }
});

test('an absent child\'s membership read joins its open Details read; the shared 404 still proves absence', async () => {
    const root = 'pair-census-root', kid = 'pair-child-absent';
    const f = childPair(root, kid, running(kid));
    try {
        await f.toggleChild();                  // the child's Details read is in flight
        assert.equal(f.gets.length, 1);
        f.missing = true;                       // from here every read of the child is a confirmed 404
        f.censusRootOnly();                     // a complete body lists only the root: the child left membership
        await drain();
        assert.equal(f.gets.length, 1, 'the membership reconciliation joins the read in flight for this attempt');
        await f.settle(0);
        await until(() => f.deep('[data-live-phase]')?.hidden === true,
            'the proved absence reaches the child\'s card as it did through its own read');
        assert.equal(f.childRetry(), null, 'a genuine 404 is not a failed read: no Retry');
        assert.equal(f.gets.length, 1, 'and the shared answer asks for nothing more this round');
    } finally { finish(f); }
});
