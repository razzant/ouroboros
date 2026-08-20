import assert from 'node:assert/strict';
import test from 'node:test';

import {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    clearStickyCardState,
    projectCollapsedActivity,
} from '../modules/chat_card_state.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';

test('named root card shows the latest activity headline under the coined title', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: 'Data Analysis',
        headline: 'Analyzing the dataset',
    }), 'Analyzing the dataset');
});

test('unnamed root card suppresses the line (title already shows the activity)', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: '',
        headline: 'Analyzing the dataset',
    }), '');
    // Suppressed even when a previous activity was remembered.
    assert.equal(projectCollapsedActivity({
        suggestedName: '',
        headline: '',
        previous: 'Earlier step',
    }), '');
});

test('subagent card always feeds the line from the routed progress body', () => {
    assert.equal(projectCollapsedActivity({
        isSubagent: true,
        body: 'Running the migration script',
    }), 'Running the migration script');
    // No coined name is required — the subagent title keeps role·model·id.
    assert.equal(projectCollapsedActivity({
        isSubagent: true,
        suggestedName: '',
        body: 'Collecting evidence',
    }), 'Collecting evidence');
});

test('a frame without new activity keeps the previous text (Done never blanks it)', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: 'Data Analysis',
        headline: '',
        previous: 'Analyzing the dataset',
    }), 'Analyzing the dataset');
    assert.equal(projectCollapsedActivity({
        isSubagent: true,
        body: '',
        previous: 'Running the migration script',
    }), 'Running the migration script');
});

test('whitespace-only frames fall back to the previous activity', () => {
    assert.equal(projectCollapsedActivity({
        suggestedName: 'X',
        headline: '   ',
        previous: 'Real step',
    }), 'Real step');
});

test('clearStickyCardState resets the recycled record activity + cost (reusable slots)', () => {
    const record = {
        collapsedActivity: 'Old cycle activity',
        costMeta: { meta: ['cost=$1.00'], ts: 1, final: true },
        // Models the real element closely enough for attribute handling.
        activityEl: {
            textContent: 'Old cycle activity',
            title: '',
            removeAttribute(name) { if (name === 'title') this.title = ''; },
        },
    };
    record.latestActivityTs = '12:00:00';
    record.activityEl.title = 'Old cycle activity';
    clearStickyCardState(record);
    assert.equal(record.collapsedActivity, '');
    assert.equal(record.costMeta, null);
    assert.equal(record.activityEl.textContent, '');
    // The activity clock is cycle state too.
    assert.equal(record.latestActivityTs, '');
    assert.equal(record.activityEl.title, '');
});

test('a terminal subagent keeps its last narration as collapsed activity (replay)', () => {
    // On history replay terminal children are pre-marked before pass 1, so the
    // card is never re-driven through a working frame; the projection must
    // still return the remembered narration for the collapsed line.
    assert.equal(projectCollapsedActivity({
        isSubagent: true, body: 'Collecting evidence', previous: 'Collecting evidence',
    }), 'Collecting evidence');
    // A later empty frame does not blank it.
    assert.equal(projectCollapsedActivity({
        isSubagent: true, body: '', previous: 'Collecting evidence',
    }), 'Collecting evidence');
});

test('the collapsed projection is whitespace-normalized, bounded and explicit', () => {
    const long = 'x'.repeat(COLLAPSED_ACTIVITY_MAX + 250);
    const out = boundActivityPreview(long);
    assert.equal(out.length, COLLAPSED_ACTIVITY_MAX);
    assert.ok(out.endsWith('…'), 'the cut is visible, never silent');
    assert.equal(boundActivityPreview('  Reading\n  the   ledger  '), 'Reading the ledger');
    assert.equal(projectCollapsedActivity({ suggestedName: 'X', headline: long }), out);
});

test('root progress keeps a bounded activity preview and a complete timeline companion', () => {
    const full = `Inspecting evidence\n${'long detail '.repeat(60)}UNIQUE_ROOT_TAIL`;
    const summary = summarizeChatLiveEvent({
        type: 'send_message', is_progress: true, task_id: 'root-1', content: full,
    });
    assert.ok(summary.activityPreview.length <= COLLAPSED_ACTIVITY_MAX);
    assert.match(summary.activityPreview, /^Inspecting evidence/);
    assert.equal(summary.fullHeadline, full);
    assert.match(summary.fullHeadline, /UNIQUE_ROOT_TAIL$/);
});

test('subagent projection keeps identity, compact facts and complete disclosure', () => {
    const full = `${'Collecting evidence '.repeat(30)}UNIQUE_CHILD_TAIL`;
    const summary = summarizeChatLiveEvent({
        type: 'send_message',
        is_progress: true,
        delegation_role: 'subagent',
        subagent_task_id: 'child123456',
        parent_task_id: 'parent1',
        root_task_id: 'root1',
        subagent_role: 'researcher',
        model: 'anthropic/claude-fable-5',
        subagent_event: 'running',
        content: full,
        write_surface: 'workspace',
        status: 'running',
    });
    assert.match(summary.headline, /researcher · claude-fable-5 \(child123\) — running/);
    assert.ok(summary.activityPreview.length <= COLLAPSED_ACTIVITY_MAX);
    assert.match(summary.fullBody, /UNIQUE_CHILD_TAIL$/);
    assert.deepEqual(summary.meta, ['write=workspace', 'status=running']);
    assert.doesNotMatch(summary.meta.join(' '), /subagent|role=|parent=|root=/);
});
