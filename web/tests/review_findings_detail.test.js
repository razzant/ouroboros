import assert from 'node:assert/strict';
import test from 'node:test';

import {
    createReviewHydrator,
    createReviewPresentationController,
    planReviewGroupFromTaskDetail,
    renderReviewsSection,
    taskAcceptanceGroupFromTaskDetail,
} from '../modules/review_presentation.js';
import { reconcileReviewElementTree } from '../modules/review_dom_patch.js';

test('a full Plan wave renders findings, mapped dispositions, degraded reviewers and honesty stamps', () => {
    const fingerprint = 'd'.repeat(64);
    const group = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: { fingerprint, status: 'open' },
            waves: [{
                request_fingerprint: fingerprint,
                cycle_index: 2,
                aggregate: 'REVISE_PLAN',
                closed: false,
                paid: true,
                reason: 'blocking_slots_at_quorum:2/3',
                counts: { blocking: 1, note: 1, need_evidence: 0 },
                findings: [
                    {
                        finding_id: 'slot_1:f1', id: 'f1', class: 'blocking',
                        summary: 'The rollback path loses the stash', breaks: 'invariant_2',
                        locator: 'supervisor/update_merge.py',
                        recommendation: 'Restore the stash before reset',
                        slot: 'slot_1', model: 'anthropic/claude-opus-5',
                    },
                    {
                        finding_id: 'slot_2:f1', id: 'f1', class: 'note',
                        summary: 'Naming could be clearer', slot: 'slot_2',
                    },
                ],
                dispositions: [
                    { finding_id: 'slot_1:f1', decision: 'reject', rationale: 'stash is restored by boot finalize' },
                    { finding_id: 'slot_1:f1', decision: 'accept', rationale: 'second thoughts after re-reading' },
                    { finding_id: 'slot_9:gone', decision: 'accept', rationale: 'will fold into phase 2' },
                ],
                actors: [
                    { slot_id: 'slot_1', model: 'anthropic/claude-opus-5', ok: true },
                    { slot_id: 'slot_3', model: 'openai/gpt-5.6-sol', ok: false, failure_code: 'window_exhausted' },
                ],
                findings_paged: true,
                findings_total: 40,
                findings_texts_truncated: true,
                spec_body_truncated: true,
            }],
            waves_omitted: 0,
        },
    });

    const detail = group.attempts[0].detailText;
    assert.match(detail, /Findings: 1 blocking · 1 note · 0 need_evidence/);
    assert.match(detail, /\[blocking\] The rollback path loses the stash — breaks invariant_2 — at supervisor\/update_merge\.py — slot_1 · anthropic\/claude-opus-5/);
    assert.match(detail, /  fix: Restore the stash before reset/);
    assert.match(detail, /  agent: reject — stash is restored by boot finalize/);
    // Contradictory duplicate dispositions both render: the backend refuses
    // the pair and keeps the finding open, so hiding one would present the
    // other as operative.
    assert.match(detail, /  agent: accept — second thoughts after re-reading/);
    assert.match(detail, /\[note\] Naming could be clearer — slot_2/);
    assert.match(detail, /General dispositions:\n  slot_9:gone: accept — will fold into phase 2/);
    assert.match(detail, /^openai\/gpt-5\.6-sol · unavailable$/m);
    assert.match(detail, /Showing 2 of 40 findings \(per-slot page cap\)/);
    assert.match(detail, /Some finding texts were truncated at capture\./);
    assert.match(detail, /Spec body was truncated at capture\./);
    assert.match(detail, /Cost unavailable/);
    // The rendered section carries the finding text to the reader.
    const html = renderReviewsSection([group], {
        sectionExpanded: true,
        expandedGroups: new Set(['plan:root']),
        expandedAttempts: new Set([`plan:root:${group.attempts[0].id}`]),
    });
    assert.match(html, /The rollback path loses the stash/);
});

test('a compact Plan wave names its recorded counts and the immutable artifact remainder', () => {
    const fingerprint = 'e'.repeat(64);
    const group = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: {},
            waves: [{
                compact: true,
                request_fingerprint: fingerprint,
                cycle_index: 1,
                aggregate: 'GREEN',
                closed: true,
                counts: { findings: 5, dispositions: 3, blocking: 2 },
                wave_artifact: { root: 'artifact_store', path: 'w.json', sha256: 'abc123def4567890', bytes: 321 },
            }],
            waves_omitted: 0,
        },
    });
    const detail = group.attempts[0].detailText;
    assert.match(detail, /Recorded: 5 findings · 2 blocking · 3 dispositions/);
    assert.match(detail, /Finding bodies compacted · artifact sha256=abc123def456… \(321 bytes\)/);
    assert.doesNotMatch(detail, /w\.json/);
});

test('a revised plan selected after a critic wave is labelled as the earlier plan\'s review', () => {
    const critic = 'c'.repeat(64);
    const revised = 'r'.repeat(64);
    const authorSubject = {
        review_fingerprint: critic,
        source_ref: { root: 'artifact_store', path: 'plan-author.json', sha256: 'f'.repeat(64) },
        author_disposition: { action: 'finish', disposition: 'partial', rationale: 'Considered.', subject_hash: revised },
    };
    const detail = (fingerprint) => ({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: { fingerprint, status: 'open', reason: 'author_current_plan', author_subject: { ...authorSubject, author_disposition: { ...authorSubject.author_disposition, subject_hash: fingerprint } } },
            waves: [{ request_fingerprint: critic, cycle_index: 1, aggregate: 'GREEN', closed: true, paid: true, findings: [], counts: { blocking: 0, note: 0, need_evidence: 0 } }],
            waves_omitted: 0,
        },
    });
    const historical = planReviewGroupFromTaskDetail(detail(revised));
    assert.equal(historical.label, 'Plan review · earlier plan');
    assert.equal(historical.historicalCritic, true);
    assert.equal(historical.verdict, 'GREEN');  // the critic's real verdict, never a synthesized one
    assert.match(historical.authorDecisionText, /The verdict shown is the earlier plan's review; the selected plan r{64} has no verdict of its own/);
    const html = renderReviewsSection([historical], { sectionExpanded: true, expandedGroups: new Set(['plan:root']) });
    assert.match(html, /Plan review · earlier plan/);
    assert.match(html, /has no verdict of its own/);
    // Quiet side: the author selecting the reviewed plan itself keeps the plain group.
    const own = planReviewGroupFromTaskDetail(detail(critic));
    assert.equal(own.label, 'Plan review');
    assert.equal(own.historicalCritic, false);
    assert.doesNotMatch(own.authorDecisionText, /no verdict of its own/);
});

test('a plan wave ordered weaker than the owner setting names each seat, and a silent seat keeps its earlier finding listed', () => {
    const fingerprint = 'w'.repeat(64);
    const detail = (extra) => planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: { fingerprint, status: 'open' },
            waves: [{
                request_fingerprint: fingerprint, cycle_index: 2, aggregate: 'REVIEW_REQUIRED', closed: false, paid: true,
                counts: { blocking: 1, note: 0, need_evidence: 0 },
                findings: [{ finding_id: 'slot_1:f1', id: 'f1', class: 'blocking', summary: 'Friday is impossible', breaks: 'invariant_1', slot: 'slot_1', model: 'm/a', carried_absent_answer: true }],
                actors: [
                    { slot_id: 'slot_1', model: 'm/a', ok: false, error: 'transport died', effort: 'low', declared_effort: 'low', carried_findings: 1 },
                    { slot_id: 'slot_2', model: 'm/b', ok: true, effort: 'low', declared_effort: 'low' },
                    { slot_id: 'slot_3', model: 'cursor-grok-4.6-xhigh', ok: true, effort: 'xhigh', declared_effort: '' },
                ],
                ...extra,
            }],
            waves_omitted: 0,
        },
    }).attempts[0].detailText;
    const weaker = detail({ ordered_weaker: { slot_1: { effort: 'low', owner_effort: 'xhigh' }, slot_2: { effort: 'low', owner_effort: 'high' } } });
    assert.match(weaker, /^Reviewers ordered weaker than your setting: slot_1 low \(setting xhigh\), slot_2 low \(setting high\)$/m);
    assert.match(weaker, /^m\/a · unavailable · did not answer; its earlier finding is still listed$/m);
    assert.match(weaker, /\[blocking\] Friday is impossible — breaks invariant_1 — slot_1 · m\/a/);
    assert.doesNotMatch(weaker, /Verdict: GREEN/);
    // Quiet side: a wave with no weaker order (or an empty fact) draws no such line.
    assert.doesNotMatch(detail({}), /ordered weaker/);
    assert.doesNotMatch(detail({ ordered_weaker: {} }), /ordered weaker/);
});

test('the hydrator announces first load, failure and retry without narrating background refreshes', async () => {
    const events = [];
    let mode = 'ok';
    const hydrator = createReviewHydrator({
        fetchDetail: async () => {
            if (mode === 'reject') throw new Error('transport failed');
            return mode === 'missing' ? null : { ok: true };
        },
        applyDetail: () => true,
        onState: (taskId, status) => events.push(`${taskId}:${status}`),
    });

    mode = 'reject';
    assert.equal(await hydrator.hydrate('root', 'a'.repeat(64)), false);
    assert.deepEqual(events, ['root:loading', 'root:error']);

    events.length = 0;
    mode = 'ok';
    await hydrator.hydrate('root', 'a'.repeat(64));
    assert.deepEqual(
        events, ['root:loading', 'root:idle'],
        'a plain re-hydrate after failure re-fetches (no applied receipt was recorded) and announces',
    );

    events.length = 0;
    await hydrator.hydrate('root', 'b'.repeat(64));
    assert.deepEqual(events, ['root:idle'], 'a background refresh over applied content stays silent');

    events.length = 0;
    mode = 'missing';
    await hydrator.hydrate('gone', 'c'.repeat(64));
    assert.deepEqual(
        events, ['gone:loading', 'gone:idle'],
        'a genuinely absent record (404 → null) is not an error',
    );
});

test('the section renders hydration truth and the controller retries through onHydrate', () => {
    const errorHtml = renderReviewsSection([
        taskAcceptanceGroupFromTaskDetail({
            task_id: 'root',
            review_projection: { panels: [{ surface: 'task_acceptance', panel_id: 'p1', aggregate_signal: 'PASS' }] },
        }, 'root'),
    ], { sectionExpanded: true, hydrateStatus: 'error' });
    assert.match(errorHtml, /data-review-hydrate-status/);
    assert.match(errorHtml, /role="alert"/);
    assert.match(errorHtml, /<span>Review details failed to refresh/);
    assert.match(errorHtml, /data-review-hydrate-retry/);

    const loadingHtml = renderReviewsSection([
        taskAcceptanceGroupFromTaskDetail({
            task_id: 'root',
            review_projection: { panels: [{ surface: 'task_acceptance', panel_id: 'p1', aggregate_signal: 'PASS' }] },
        }, 'root'),
    ], { sectionExpanded: true, hydrateStatus: 'loading' });
    assert.match(loadingHtml, /Loading review details…/);

    const hydrated = [];
    let clickHandler = null;
    const statusNode = {
        setAttribute(key, value) { this[key] = value; },
        focus() { this.focused = true; },
    };
    const controller = createReviewPresentationController({
        host: {
            addEventListener: (_type, handler) => { clickHandler = handler; },
            querySelector: (selector) => (selector === '[data-review-hydrate-status]' ? statusNode : null),
        },
        summary: null,
        onHydrate: (...args) => hydrated.push(args),
    });
    controller.setHydrateStatus('error');
    clickHandler({
        target: {
            closest: (selector) => (selector === '[data-review-hydrate-retry]' ? {} : null),
        },
    });
    assert.deepEqual(hydrated, [[]]);
    assert.equal(statusNode.tabindex, '-1');
    assert.equal(statusNode.focused, true);
});

test('a failed first hydration renders the section shell with no groups', () => {
    assert.equal(renderReviewsSection([], {}), '');
    assert.equal(renderReviewsSection([], { hydrateStatus: 'loading' }), '',
        'a quiet first-load zero-group loading pass stays invisible (every card expand hydrates)');
    const errorShell = renderReviewsSection([], { sectionExpanded: true, hydrateStatus: 'error' });
    assert.match(errorShell, /data-review-section/);
    assert.match(errorShell, /Review details failed to refresh/);
    assert.match(errorShell, /data-review-hydrate-retry/);
    assert.match(errorShell, /chat-review-section-count">—</);

    // The error must be readable on a COLLAPSED section too: the status node
    // sits outside the hidden groups container.
    const collapsedShell = renderReviewsSection([], { hydrateStatus: 'error' });
    assert.match(collapsedShell, /data-expanded="0"/);
    const statusIndex = collapsedShell.indexOf('data-review-hydrate-status');
    const groupsIndex = collapsedShell.indexOf('chat-review-groups');
    assert.ok(statusIndex >= 0 && statusIndex < groupsIndex,
        'status node renders before (outside) the collapsible groups container');
    assert.match(collapsedShell, /Review details failed to refresh/);

    // A retry's own loading pass keeps the shell mounted (hadHydrateError),
    // so the recovery control cannot unmount mid-flight.
    const retryLoading = renderReviewsSection([], { hydrateStatus: 'loading', hadHydrateError: true });
    assert.match(retryLoading, /Loading review details…/);
    assert.equal(renderReviewsSection([], { hydrateStatus: 'loading', hadHydrateError: false }), '');
});

test('the hydrate status node swaps its message text across the loading→error patch', () => {
    class Node {
        constructor({ tag = 'div', dataset = {}, classes = [], attrs = {}, html = '', children = [] } = {}) {
            this.tagName = tag.toUpperCase();
            this.dataset = { ...dataset };
            this._attrs = new Map(Object.entries(attrs));
            for (const [key, value] of Object.entries(dataset)) {
                this._attrs.set(`data-${key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`)}`, String(value));
            }
            this._classes = new Set(classes);
            if (classes.length) this._attrs.set('class', classes.join(' '));
            this.classList = { contains: (name) => this._classes.has(name) };
            this._innerHTML = html;
            this.children = [];
            children.forEach((child) => this.insertBefore(child, null));
        }
        get attributes() { return [...this._attrs].map(([name, value]) => ({ name, value })); }
        get innerHTML() { return this._innerHTML; }
        set innerHTML(value) { this._innerHTML = String(value); this.children = []; }
        hasAttribute(name) { return this._attrs.has(name); }
        setAttribute(name, value) {
            this._attrs.set(name, String(value));
            if (name.startsWith('data-')) {
                this.dataset[name.slice(5).replace(/-([a-z])/g, (_a, c) => c.toUpperCase())] = String(value);
            }
        }
        removeAttribute(name) { this._attrs.delete(name); }
        insertBefore(child, before) {
            child.remove();
            const index = before ? this.children.indexOf(before) : -1;
            if (index >= 0) this.children.splice(index, 0, child); else this.children.push(child);
            child.parentElement = this;
            return child;
        }
        remove() {
            if (!this.parentElement) return;
            const index = this.parentElement.children.indexOf(this);
            if (index >= 0) this.parentElement.children.splice(index, 1);
            this.parentElement = null;
        }
        cloneNode(deep) {
            return new Node({
                tag: this.tagName,
                dataset: this.dataset,
                classes: [...this._classes],
                attrs: Object.fromEntries(this._attrs),
                html: this._innerHTML,
                children: deep ? this.children.map((child) => child.cloneNode(true)) : [],
            });
        }
    }
    const statusNode = (state) => new Node({
        dataset: { reviewHydrateStatus: '' },
        classes: [state === 'error' ? 'skill-review-error' : 'skill-review-loading'],
        children: state === 'error'
            ? [
                new Node({ tag: 'span', html: 'Review details failed to refresh — shown data may be incomplete. ' }),
                new Node({ tag: 'button', dataset: { reviewHydrateRetry: '' }, html: 'Retry' }),
            ]
            : [new Node({ tag: 'span', html: 'Loading review details…' })],
    });

    const current = statusNode('loading');
    assert.equal(reconcileReviewElementTree(current, statusNode('error')), true);
    assert.match(current.children[0].innerHTML, /failed to refresh/);
    assert.doesNotMatch(current.children[0].innerHTML, /Loading review details/);
    assert.equal(current.children.length, 2);
    assert.equal(current.children[1].innerHTML, 'Retry');

    assert.equal(reconcileReviewElementTree(current, statusNode('loading')), true);
    assert.match(current.children[0].innerHTML, /Loading review details/);
    assert.equal(current.children.length, 1);
});

test('a compact Plan wave still names reviewers ordered weaker than the setting', () => {
    const fingerprint = 'c'.repeat(64);
    const group = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: {},
            waves: [{
                compact: true, request_fingerprint: fingerprint, cycle_index: 1, aggregate: 'GREEN', closed: true,
                counts: { findings: 0, blocking: 0, dispositions: 0 },
                wave_artifact: { root: 'artifact_store', path: 'w.json', sha256: 'abc123def4567890', bytes: 321 },
                ordered_weaker: { slot_1: { effort: 'low', owner_effort: 'xhigh' } },
            }],
            waves_omitted: 0,
        },
    });
    const detail = group.attempts[0].detailText;
    assert.match(detail, /Reviewers ordered weaker than your setting: slot_1 low \(setting xhigh\)/);
    assert.match(detail, /Finding bodies compacted/);
});

test('a carried finding is explained on a not-sent seat and on a failed seat alike', () => {
    const fingerprint = 'e'.repeat(64);
    const detail = (actor) => planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: { fingerprint, status: 'open' },
            waves: [{
                request_fingerprint: fingerprint, cycle_index: 2, aggregate: 'REVIEW_REQUIRED', closed: false, paid: true,
                counts: { blocking: 1, note: 0, need_evidence: 0, parseable: 2, quorum: 2 }, findings: [],
                actors: [{ slot_id: 'slot_1', model: 'm/a', ...actor }],
            }],
            waves_omitted: 0,
        },
    }).attempts[0].detailText;
    assert.match(detail({ ok: false, operation_state: 'not_dispatched', error: 'health_skip', carried_findings: 1 }),
        /m\/a · not sent; its earlier finding is still listed/);
    assert.match(detail({ ok: false, error: 'transport died', reported_cause: 'transport died', carried_findings: 1 }),
        /m\/a · unavailable — "transport died" · did not answer; its earlier finding is still listed/);
    assert.doesNotMatch(detail({ ok: false, operation_state: 'not_dispatched', error: 'health_skip' }), /earlier finding/);
    assert.doesNotMatch(detail({ ok: false, error: 'transport died' }), /earlier finding/);
});

