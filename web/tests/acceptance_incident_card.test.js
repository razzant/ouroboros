import assert from 'node:assert/strict';
import test from 'node:test';

import { createChatInstance } from '../modules/chat.js';
import { taskReasonDetail } from '../modules/log_events.js';
import {
    acceptanceIncidentFromTaskDetail,
    mergeReviewGroup,
    renderReviewsSection,
    taskAcceptanceGroupFromTaskDetail,
} from '../modules/review_presentation.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// #1224: a LOCAL acceptance-evidence failure produces no panel at all, so the
// owner's card had nothing to read — the warning was invisible and the final
// borrowed a reviewer's words for a review that never ran. The incident reaches
// the card's Reviews group through the EXISTING carriers only: the review
// projection (live: review_reference → hydrate → keyed group merge; history
// and reconnect: the terminal record) and the acceptance decision. No card row
// of its own, no first alert, no toast.

const INCIDENT = {
    incident_id: 'acceptance-preparation:1f2e3d4c',
    status: 'failed',
    stage: 'preparation',
    attempts: 1,
    source_known: true,
    failure_kind: 'UnicodeDecodeError',
    failure_detail: "'utf-8' codec can't decode byte 0xe9",
    feedback_delivered: true,
};

const detailWith = (incident, extra = {}) => ({
    task_id: 'root',
    review_projection: { panels: [], acceptance_incident: incident },
    ...extra,
});

test('an incident with no panel at all still produces a review group', () => {
    const group = taskAcceptanceGroupFromTaskDetail(detailWith(INCIDENT), 'root');
    assert.ok(group, 'the group must exist on the incident alone');
    assert.equal(group.tone, 'warn');
    assert.match(group.warning, /could not be assembled locally/);
    assert.match(group.warning, /host attempt 1/);
    assert.match(group.warning, /no new reviewer was dispatched for this attempt/);
    assert.equal(group.attempts.length, 1);
    assert.equal(group.attempts[0].id, `acceptance-incident:${INCIDENT.incident_id}`);
});

test('the expanded detail carries the technical cause and names no reviewer verdict', () => {
    const group = taskAcceptanceGroupFromTaskDetail(detailWith(INCIDENT), 'root');
    const detailText = group.attempts[0].detailText;
    assert.match(detailText, /UnicodeDecodeError/);
    assert.match(detailText, /no new reviewer was dispatched for this attempt/);
    assert.match(detailText, /host attempts on this material: 1/);
    assert.match(detailText, /the author received this preparation failure/);
    assert.doesNotMatch(detailText, /reacted/); // Delivery is not a semantic author stance.
    assert.doesNotMatch(detailText, /PASS|FAIL/);
    const html = renderReviewsSection([group]);
    assert.match(html, /acceptance evidence/);
});

test('an earlier real reviewer FAIL keeps its own row beside the incident', () => {
    const panel = { surface: 'task_acceptance', panel_id: 'p1', aggregate_signal: 'FAIL',
        reason: 'A reviewer rejected the answer.', actors: [] };
    const group = taskAcceptanceGroupFromTaskDetail({
        task_id: 'root', review_projection: { panels: [panel], acceptance_incident: INCIDENT },
    }, 'root');
    assert.equal(group.attempts.length, 2);
    assert.equal(group.tone, 'warn');
    assert.equal(group.verdict, 'FAIL');            // the paid verdict is retained, not replaced
    assert.equal(group.attempts.at(-1).summary, 'A reviewer rejected the answer.');
});

test('the terminal acceptance decision carries the same incident after a reload', () => {
    const fromDecision = acceptanceIncidentFromTaskDetail({
        task_id: 'root',
        outcome_axes: { review: { acceptance_decision: { acceptance_incident: INCIDENT } } },
    });
    const fromProjection = acceptanceIncidentFromTaskDetail(detailWith(INCIDENT));
    assert.equal(fromDecision.id, fromProjection.id);
    assert.equal(fromDecision.attempts, fromProjection.attempts);
});

test('live, history and a reconnect state one row, never three', () => {
    const store = new Map();
    const group = () => taskAcceptanceGroupFromTaskDetail(detailWith(INCIDENT), 'root');
    mergeReviewGroup(store, group());
    mergeReviewGroup(store, group());
    const merged = mergeReviewGroup(store, group());
    assert.equal(merged.attempts.length, 1);
    assert.equal(merged.attempts[0].label, 'acceptance evidence · host attempt 1');
    assert.match(merged.warning, /host attempt 1/);
});

test('resolution clears the ACTIVE warning and keeps the row as history', () => {
    const store = new Map();
    mergeReviewGroup(store, taskAcceptanceGroupFromTaskDetail(detailWith(INCIDENT), 'root'));
    const resolved = { ...INCIDENT, status: 'resolved' };
    const merged = mergeReviewGroup(store, taskAcceptanceGroupFromTaskDetail(detailWith(resolved), 'root'));
    assert.equal(merged.warning, '');            // the active warning is gone
    assert.notEqual(merged.tone, 'warn');
    assert.equal(merged.attempts.length, 1);     // …and the fact is still recorded
    assert.match(merged.attempts[0].summary, /assembled after 1 failed host attempt/);
    assert.equal(merged.attempts[0].tone, 'neutral');
});

test('an unknown material identity is stated, not guessed at', () => {
    const group = taskAcceptanceGroupFromTaskDetail(
        detailWith({ ...INCIDENT, source_known: false }), 'root',
    );
    assert.match(group.attempts[0].detailText, /material identity: unknown/);
});

test('a record with no incident, or a zero-attempt one, is completely unchanged', () => {
    assert.equal(acceptanceIncidentFromTaskDetail({ task_id: 'root' }), null);
    assert.equal(acceptanceIncidentFromTaskDetail(detailWith({ ...INCIDENT, attempts: 0, status: 'resolved' })), null);
    assert.equal(taskAcceptanceGroupFromTaskDetail({ task_id: 'root', review_projection: { panels: [] } }, 'root'), null);
});

test('the final cause states the local failure AND the rail, and invents no rework', () => {
    const line = taskReasonDetail({
        status: 'completed',
        reason_code: 'budget_exhausted',
        outcome_axes: {
            execution: { status: 'degraded' },
            review: {
                status: 'degraded',
                acceptance_decision: {
                    status: 'finalized_unaccepted',
                    reason: 'acceptance_preparation_failed',
                    acceptance_incident: INCIDENT,
                },
            },
        },
    });
    assert.match(line, /could not assemble the evidence/);
    assert.match(line, /this preparation attempt dispatched no new reviewers/);
    assert.match(line, /ran out of budget/);
    assert.doesNotMatch(line, /rework|no reviewer ever saw/);
});

test('a genuine reviewer FAIL keeps its own sentence beside the incident', () => {
    const line = taskReasonDetail({
        status: 'completed',
        outcome_axes: {
            review: {
                acceptance_decision: {
                    status: 'finalized_unaccepted',
                    reason: 'reviewer_fail_no_capsule',
                    acceptance_incident: INCIDENT,
                },
            },
        },
    });
    assert.match(line, /A reviewer rejected the answer/);
    assert.match(line, /could not assemble the evidence/);
});

test('a resolved incident adds nothing to the final cause', () => {
    const line = taskReasonDetail({
        status: 'completed',
        outcome_axes: {
            review: {
                acceptance_decision: {
                    status: 'finalized_unaccepted', reason: 'author_finish',
                    acceptance_incident: { ...INCIDENT, status: 'resolved' },
                },
            },
        },
    });
    assert.equal(line, 'Ouroboros delivered this answer on its own judgement; the reviewers had not signed it off.');
});

// ── the ACTUAL chat consumer: review_reference → hydrate → keyed group merge ──

const chatId = 77;
const taskId = 'root';
const reference = (revision) => ({
    task_id: taskId, chat_id: chatId, presentation_owner_task_id: taskId,
    role: 'system', is_progress: true, system_type: 'review_reference',
    surface: 'task_acceptance', state_revision: revision.repeat(64),
    ts: '2026-09-23T10:00:01Z',
});

function chatFixture() {
    const data = { rows: [], detail: { task_id: taskId, status: 'running', review_projection: { panels: [] } } };
    const calls = [];
    const env = installDom(async (url) => {
        calls.push(String(url));
        if (String(url).startsWith('/api/chat/history')) return { ok: true, json: async () => ({ messages: data.rows }) };
        if (String(url).startsWith('/api/tasks/')) return { ok: true, json: async () => data.detail };
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {},
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
        chatId, idPrefix: 'chat', mountEl: env.mount, asPanel: true,
    });
    return {
        data, calls,
        card: () => walkCard(globalThis.document.byId.get('chat-messages'), taskId),
        send: (row) => handlers.get('chat')({ ...row, content: row.text || '' }),
        settle: () => new Promise((resolve) => setImmediate(resolve)),
        destroy() { instance.destroy(); restoreDom(env.prior); },
    };
}

test('the chat consumer shows the incident in the card Reviews group from a review reference and clears it on resolution', async () => {
    const fx = chatFixture();
    try {
        fx.send({ chat_id: chatId, role: 'system', is_progress: true, task_id: taskId,
            text: 'assembling acceptance evidence', ts: '2026-09-23T10:00:00Z' });
        const card = fx.card();
        assert.ok(card, 'the task card exists before its review reference arrives');

        // The host published the projection with the incident and no panel at
        // all; the existing review_reference invalidation makes the card read it.
        fx.data.detail = { task_id: taskId, status: 'running',
            review_projection: { panels: [], acceptance_incident: INCIDENT } };
        fx.send(reference('a'));
        await fx.settle();
        assert.ok(fx.calls.some((url) => url.startsWith(`/api/tasks/${taskId}`)), 'the reference re-read the task detail');
        const host = card.querySelector('[data-live-reviews-host]');
        assert.match(host.innerHTML, /chat-review-group warn/);
        assert.match(host.innerHTML, /acceptance evidence · host attempt 1/);
        assert.equal(card.dataset.expanded, '0');
        const summary = card.querySelector('[data-live-review-summary]');
        assert.equal(summary.hidden, false);
        assert.match(summary.textContent, /Acceptance unavailable: evidence preparation failed; no new reviewers/);

        // A replayed reference of the same publication changes nothing.
        fx.send(reference('a'));
        await fx.settle();
        assert.equal((host.innerHTML.match(/acceptance evidence · host attempt 1/g) || []).length, 1);

        // The next publication resolved it: the same keyed group updates in place.
        fx.data.detail = { task_id: taskId, status: 'running',
            review_projection: { panels: [], acceptance_incident: { ...INCIDENT, status: 'resolved' } } };
        fx.send(reference('b'));
        await fx.settle();
        assert.doesNotMatch(host.innerHTML, /chat-review-group warn/);
        assert.match(host.innerHTML, /assembled after 1 failed host attempt/);
        assert.equal((host.innerHTML.match(/acceptance evidence · host attempt 1/g) || []).length, 1);
    } finally { fx.destroy(); }
});

test('the chat consumer reads the same incident from a terminal record on replay', async () => {
    const fx = chatFixture();
    try {
        fx.send({ chat_id: chatId, role: 'system', is_progress: true, task_id: taskId,
            text: 'assembling acceptance evidence', ts: '2026-09-23T10:00:00Z' });
        fx.send({ chat_id: chatId, role: 'system', system_type: 'task_summary', task_id: taskId,
            task_terminal_status: 'completed', ts: '2026-09-23T10:00:05Z', text: 'Done',
            review_projection: { panels: [], acceptance_incident: INCIDENT },
            outcome_axes: { execution: { status: 'ok' }, review: { status: 'degraded', acceptance_decision: {
                status: 'finalized_unaccepted', reason: 'acceptance_preparation_failed', acceptance_incident: INCIDENT } } },
        });
        await fx.settle();
        const host = fx.card().querySelector('[data-live-reviews-host]');
        assert.match(host.innerHTML, /chat-review-group warn/);
        assert.match(host.innerHTML, /acceptance evidence · host attempt 1/);
    } finally { fx.destroy(); }
});
