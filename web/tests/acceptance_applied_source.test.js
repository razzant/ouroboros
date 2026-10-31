import assert from 'node:assert/strict';
import test from 'node:test';

import {
    createReviewHydrator,
    createReviewPresentationController,
    mergeReviewGroup,
    renderReviewsSection,
    reviewReferenceFromRow,
    taskAcceptanceGroupFromTaskDetail,
} from '../modules/review_presentation.js';

const hash = 'a'.repeat(64);
const panel = (overrides = {}) => ({
    surface: 'task_acceptance', panel_id: 'panel_1', aggregate_signal: 'FAIL',
    reason: 'The host applied the complete review.', task_attempt: 0,
    applied_source_status: 'available',
    applied_source_ref: { root: 'artifact_store', path: `source_handles/context_checkpoints/acceptance-${hash}.json`,
        sha256: hash, size: 75000, kind: 'task_source' },
    ...overrides,
});
const detail = (panels) => ({ task_id: 'root', review_projection: { panels } });
const group = (value) => taskAcceptanceGroupFromTaskDetail(detail([value]));

test('the complete applied source uses the existing artifact download route', () => {
    const projected = group(panel());
    assert.equal(projected.attempts[0].detailRef.url,
        `/api/tasks/root/artifacts/acceptance-${hash}.json?source=${encodeURIComponent(panel().applied_source_ref.path)}`);
    const html = renderReviewsSection([projected]);
    // A opens the exact saved applied snapshot in the existing reader; its
    // Download control is there, rather than a second card-level link.
    assert.match(html, /data-review-read-url="\/api\/tasks\/root\/artifacts\/acceptance-/);
    assert.match(html, />Read saved applied review<svg[^>]*>[\s\S]*?<\/svg><\/button>/);
    // The exact applied record stays in the group's Record facts, one labelled line per fact.
    assert.match(html, /<dt>Review panel panel_1<\/dt><dd>task_acceptance · authority=unspecified · verdict=FAIL/);
    assert.doesNotMatch(html, /full applied review unavailable/i);
    assert.equal(projected.attempts[0].timestamp, '');
});

test('legacy, unavailable and malformed sources remain explicitly unavailable', () => {
    for (const overrides of [
        { applied_source_status: undefined, applied_source_ref: undefined },
        { applied_source_status: 'unavailable' },
        { applied_source_ref: { ...panel().applied_source_ref, root: 'runtime_data' } },
        { applied_source_ref: { ...panel().applied_source_ref, path: '../private.json' } },
        { applied_source_ref: { ...panel().applied_source_ref, path: 'file%2Fprivate.json' } },
        { applied_source_ref: { ...panel().applied_source_ref, sha256: '' } },
        { applied_source_ref: { ...panel().applied_source_ref, size: '75000' } },
    ]) {
        const html = renderReviewsSection([group(panel(overrides))]);
        assert.match(html, /<span class="review-seat-meta">full applied review unavailable<\/span>/);
        assert.doesNotMatch(html, /data-review-read-url|<a /);
    }
});

test('the applied-source Read control hands the exact URL to the existing reader', () => {
    let click;
    const opened = [];
    const controller = createReviewPresentationController({
        host: { addEventListener(_event, handler) { click = handler; } },
        onRead: (...args) => opened.push(args),
    });
    const url = group(panel()).attempts[0].detailRef.url;
    const opener = { dataset: { reviewReadUrl: url, reviewReadName: 'Saved applied review' } };
    click({ target: { closest: selector => selector === '[data-review-read-url]' ? opener : null } });
    assert.equal(opened.length, 1);
    assert.equal(opened[0][0].reader.url, url);
    assert.equal(opened[0][0].source.durable, url);
    assert.equal(opened[0][0].mime, 'text/plain');
    assert.equal(opened[0][1], opener);
    assert.equal(controller.groups.size, 0, 'reading dispatches no new review');
});

test('retry identity uses an actual task attempt without inventing one for legacy rows', () => {
    const store = new Map();
    mergeReviewGroup(store, group(panel()));
    mergeReviewGroup(store, group(panel({ task_attempt: 1 })));
    assert.equal(store.get('task_acceptance:root').attempts.length, 2);
    assert.notEqual(group(panel()).attempts[0].id, group(panel({ task_attempt: 1 })).attempts[0].id);
    assert.equal(group(panel({ task_attempt: undefined })).attempts[0].id, 'panel_1');
});

test('a panel and its later application-failure record keep separate trace identities', () => {
    const projection = taskAcceptanceGroupFromTaskDetail(detail([
        panel({ panel_index: 0 }),
        panel({ panel_index: 1, aggregate_signal: 'DEGRADED' }),
    ]));
    const store = new Map();
    mergeReviewGroup(store, projection);
    mergeReviewGroup(store, projection);
    assert.equal(store.get('task_acceptance:root').attempts.length, 2);
    assert.equal(new Set(projection.attempts.map((attempt) => attempt.id)).size, 2);
    assert.deepEqual(projection.attempts.map((attempt) => attempt.verdict), ['FAIL', 'DEGRADED']);
});

test('acceptance invalidation joins a current read and refreshes to the newly applied source', async () => {
    const release = [];
    const applied = [];
    const hydrator = createReviewHydrator({
        fetchDetail: () => new Promise((resolve) => release.push(resolve)),
        applyDetail: (_id, value) => applied.push(group(value)),
    });
    const reference = (revision) => reviewReferenceFromRow({
        type: 'review_reference', surface: 'task_acceptance', task_id: 'root',
        state_revision: revision,
    });
    const first = reference('a'.repeat(64));
    const second = reference('b'.repeat(64));
    const one = hydrator.hydrate(first.presentationOwnerTaskId, first.stateRevision);
    await Promise.resolve();
    const two = hydrator.hydrate(second.presentationOwnerTaskId, second.stateRevision);
    assert.equal(release.length, 1);
    release[0](panel({ applied_source_status: 'unavailable' }));
    await one;
    await Promise.resolve();
    assert.equal(release.length, 2);
    release[1](panel({ publication_revision: 2 }));
    await two;
    assert.equal(applied.at(-1).attempts[0].detailRef.url,
        `/api/tasks/root/artifacts/acceptance-${hash}.json?source=${encodeURIComponent(panel().applied_source_ref.path)}`);
    await hydrator.hydrate(second.presentationOwnerTaskId, second.stateRevision);
    assert.equal(release.length, 2);
    assert.equal(reviewReferenceFromRow({ type: 'review_reference', surface: 'commit', task_id: 'root' }), null);
});

test('source handles use the bound-source selector on the existing download route', () => {
    const path = `source_handles/context_checkpoints/acceptance-${hash}.json`;
    const ref = { kind: 'task_source', root: 'artifact_store', path, size: 75000, sha256: hash };
    const projected = group(panel({ applied_source_ref: ref }));
    assert.equal(projected.attempts[0].detailRef.url,
        `/api/tasks/root/artifacts/acceptance-${hash}.json?source=${encodeURIComponent(path)}`);
    assert.match(renderReviewsSection([projected]), />Read saved applied review<svg/);
    for (const bad of [
        { ...ref, path: 'source_handles/context_checkpoints/../private.json' },
        { ...ref, path: 'source_handles/other/private.json' },
        { ...ref, size: '75000' }, { ...ref, sha256: '' },
        { ...ref, root: 'runtime_data' },
    ]) {
        assert.equal(group(panel({ applied_source_ref: bad })).attempts[0].detailRef.url, '');
    }
});
