import assert from 'node:assert/strict';
import test from 'node:test';

import { renderInstalledSkillCard } from '../modules/skill_card_renderer.js';

test('skill card shows the current review round over the runner-bounded ten-row window', () => {
    const history = Array.from({ length: 10 }, (_, idx) => ({
        status: 'clean',
        content_hash: `snapshot-${idx}`,
        group_id: 'manual:alpha',
        review_round: idx + 3,
        snapshot_attempt: 1,
        snapshot_revised: idx === 9,
        raw_actor_records: [{ raw_text: 'must stay private' }],
    }));
    const skill = {
        name: 'alpha',
        type: 'instruction',
        version: '1.0.0',
        description: 'test',
        source: 'external',
        enabled: false,
        review_status: 'clean',
        review_gate: { executable_review: true },
        review_findings: [],
        permissions: [],
        grants: {},
        skill_review: {
            current: history[history.length - 1],
            history,
        },
    };

    const html = renderInstalledSkillCard(skill);

    assert.match(html, /Skill review round 12 — snapshot snapshot-9 \(attempt 1\) — revised snapshot/);
    assert.match(html, /<details class="skills-review-history ui-rich-content">/);
    assert.match(html, /Skill Review history \(10\)/);
    assert.doesNotMatch(html, /must stay private/);
});

test('a bounded history window names the exact omitted remainder in its label', () => {
    const history = Array.from({ length: 10 }, (_, idx) => ({
        status: 'clean',
        content_hash: `snapshot-${idx}`,
        group_id: 'manual:alpha',
        review_round: idx + 15,
        snapshot_attempt: 1,
    }));
    const skill = {
        name: 'alpha',
        type: 'instruction',
        version: '1.0.0',
        description: 'test',
        source: 'external',
        enabled: false,
        review_status: 'clean',
        review_gate: { executable_review: true },
        review_findings: [],
        permissions: [],
        grants: {},
        skill_review: {
            current: history[history.length - 1],
            history,
            history_omitted: 14,
        },
    };

    const html = renderInstalledSkillCard(skill);
    assert.match(html, /Skill Review history \(10 of 24\)/);
});

function reviewCard(skillReview) {
    return renderInstalledSkillCard({
        name: 'alpha', type: 'instruction', version: '1.0.0', description: 'test',
        source: 'external', enabled: false, review_status: 'pending',
        review_gate: { executable_review: false }, review_findings: [],
        permissions: [], grants: {}, skill_review: skillReview,
    });
}

const REASON_ROW = '(?:<div class="skills-review-reason"( title="[^"]*")?>Reason: ([^<]*)</div>)?';

function currentOutcome(html) {
    const match = html.match(new RegExp(`<div class="skills-review-current"><strong>[^<]*</strong> · ([^<]*)</div>${REASON_ROW}`));
    assert.ok(match, 'current review line is rendered');
    return { line: match[1], reason: match[3] ?? null, title: match[2] ?? null };
}

function historyOutcomes(html) {
    return [...html.matchAll(new RegExp(`<li>Skill review round \\d+ — snapshot [^ ]+ \\(attempt \\d+\\) · ([^<]*)${REASON_ROW}(?:<details class="skills-review-reason">[\\s\\S]*?</details>)?</li>`, 'g'))]
        .map((match) => ({ line: match[1], reason: match[3] ?? null }));
}

const UNAVAILABLE_FAILED = 'review verdict unavailable · lifecycle failed';

// The job file keeps the lifecycle in `status` and the verdict in
// `review_status` (skill_review_runner._on_finished); a history row keeps the
// lifecycle in `job_status` and the verdict, or the lifecycle word when no
// verdict arrived, in `status` (_terminal_history_payload).
const OUTCOME_CASES = [
    {
        name: 'failed job without a verdict',
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'pending', terminal_reason: 'RuntimeError: reviewer offline' },
        history: { status: 'failed', job_status: 'failed', terminal_reason: 'RuntimeError: reviewer offline' },
        line: UNAVAILABLE_FAILED, reason: 'RuntimeError: reviewer offline',
    },
    {
        name: 'failed job whose result kept the pending verdict',
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'pending', terminal_reason: 'Skill file too large for one review pass' },
        history: { status: 'pending', job_status: 'failed', terminal_reason: 'Skill file too large for one review pass' },
        line: UNAVAILABLE_FAILED, reason: 'Skill file too large for one review pass',
    },
    {
        name: 'clean verdict whose dependency install failed',
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'clean', terminal_reason: 'pip exited with status 1' },
        history: { status: 'clean', job_status: 'failed', terminal_reason: 'pip exited with status 1' },
        line: 'clean · lifecycle failed', reason: 'pip exited with status 1',
    },
    {
        name: 'completed clean review',
        current: { status: 'completed', lifecycle_status: 'succeeded', review_status: 'clean', terminal_reason: 'succeeded' },
        history: { status: 'clean', job_status: 'succeeded', terminal_reason: 'succeeded' },
        line: 'clean', reason: null,
    },
    {
        name: 'completed job without a typed verdict',
        current: { status: 'completed', lifecycle_status: 'succeeded', review_status: 'pending', terminal_reason: 'succeeded' },
        history: { status: 'pending', job_status: 'succeeded', terminal_reason: 'succeeded' },
        line: 'pending', reason: null,
    },
    {
        name: 'interrupted job (dead owner)',
        current: { status: 'interrupted', lifecycle_status: 'interrupted', terminal_reason: 'owner_process_exited' },
        history: { status: 'interrupted', job_status: 'interrupted', terminal_reason: 'owner_process_exited' },
        line: 'review verdict unavailable · lifecycle interrupted', reason: 'owner_process_exited',
    },
    {
        name: 'timed-out job',
        current: { status: 'timeout', lifecycle_status: 'timeout', terminal_reason: 'TimeoutError: lifecycle deadline' },
        history: { status: 'timeout', job_status: 'timeout', terminal_reason: 'TimeoutError: lifecycle deadline' },
        line: 'review verdict unavailable · lifecycle timeout', reason: 'TimeoutError: lifecycle deadline',
    },
    {
        name: 'reason that only repeats the lifecycle word',
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'pending', terminal_reason: 'failed' },
        history: { status: 'failed', job_status: 'failed', terminal_reason: 'failed' },
        line: UNAVAILABLE_FAILED, reason: null,
    },
    {
        name: 'legacy verdict-only row',
        current: { status: 'pending' },
        history: { status: 'pending' },
        line: 'pending', reason: null,
    },
    {
        name: 'legacy row with no status at all',
        current: { review_round: 1 },
        history: { review_round: 1 },
        line: 'unknown', reason: null,
    },
];

test('a review run names its lifecycle outcome beside the verdict in both recorded shapes', () => {
    for (const item of OUTCOME_CASES) {
        const current = currentOutcome(reviewCard({ current: item.current, history: [] }));
        assert.deepEqual(
            { line: current.line, reason: current.reason },
            { line: item.line, reason: item.reason },
            `job file: ${item.name}`,
        );
        const rows = historyOutcomes(reviewCard({ current: { status: 'completed', review_status: 'clean' }, history: [item.history] }));
        assert.deepEqual(rows, [{ line: item.line, reason: item.reason }], `history row: ${item.name}`);
    }
});

test('a running review shows its lifecycle word with no reason', () => {
    const html = reviewCard({
        current: { status: 'running', lifecycle_status: 'running', terminal_reason: '' },
        history: [],
    });
    assert.deepEqual(currentOutcome(html), { line: 'running', reason: null, title: null });
});

test('a failed run never reads as pending and its history keeps the disclosed omission count', () => {
    const failed = { status: 'failed', job_status: 'failed', terminal_reason: 'RuntimeError: boom', review_round: 4 };
    const html = reviewCard({
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'pending', terminal_reason: 'RuntimeError: boom', review_round: 4 },
        history: [{ status: 'clean', job_status: 'succeeded', review_round: 3 }, failed],
        history_omitted: 2,
    });
    assert.equal(currentOutcome(html).line, UNAVAILABLE_FAILED);
    assert.doesNotMatch(html, /· pending</);
    assert.match(html, /Skill Review history \(2 of 4\)/);
    assert.deepEqual(historyOutcomes(html).map((row) => row.line), ['clean', UNAVAILABLE_FAILED]);
});

test('a recorded reason is escaped and bounded after the outcome it explains', () => {
    const hostile = '<img src=x onerror="alert(1)">';
    const escaped = currentOutcome(reviewCard({
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'clean', terminal_reason: hostile },
        history: [],
    }));
    assert.equal(escaped.line, 'clean · lifecycle failed');
    assert.equal(escaped.reason, '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');

    const giant = `ModuleNotFoundError: ${'x'.repeat(20_000)}`;
    const html = reviewCard({
        current: { status: 'failed', lifecycle_status: 'failed', review_status: 'pending', terminal_reason: giant },
        history: [{ status: 'failed', job_status: 'failed', terminal_reason: giant }],
    });
    const current = currentOutcome(html);
    assert.equal(current.line, UNAVAILABLE_FAILED);
    assert.ok(current.reason.length <= 420, `visible reason is bounded (${current.reason.length})`);
    assert.match(current.reason, /^ModuleNotFoundError: x+…\[truncated\]$/);
    assert.equal(current.title, ` title="${giant}"`);
    assert.ok(html.includes(`<summary>Full reason</summary><div>${giant}</div>`), 'complete reason is also touch/keyboard accessible');
    assert.ok(html.indexOf(UNAVAILABLE_FAILED) < html.indexOf('ModuleNotFoundError'));
    assert.match(historyOutcomes(html)[0].reason, /…\[truncated\]$/);
});
