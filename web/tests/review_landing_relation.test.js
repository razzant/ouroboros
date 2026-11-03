import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createReviewPresentationController, mergeReviewGroup, reviewGroupsFromTaskDetail, renderReviewsSection } from '../modules/review_presentation.js';
import { landingWords } from '../modules/review_record_card.js';

// #2242, owner 29.10 A: the opened Reviews row says how ITS record relates to the code that landed and was
// adopted, then the reason of the decision that governed THAT commit; the head only adds `earlier version`; the
// verdict stays the record's.
const CAND = '4f1c2a9e7b3d5a60c81e2f97d4b3a1c0e9f8d7a6';
const EARLY = '1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d';
const PREV = '9a8b7c6d5e4f30211f0e9d8c7b6a5f4e3d2c1b0a';
const SUBJECT = 'f'.repeat(64);
const LATER = 'e'.repeat(64);
const REASON = 'The Plus menu closes on send before the reply can land, so the sequence the reviewer describes cannot occur; '
    + 'the new test covers both send paths. '.repeat(6) + 'END-OF-REASON';
const seats = (verdict) => [{ slot_id: 's1', model: 'openai/gpt-5.5', observed_model: 'openai/gpt-5.5', transport_status: 'success',
    parse_status: 'valid', semantic_verdict: verdict, findings: [] }];
const landing = (overrides = {}) => ({ commit_sha: CAND, tree_sha: 'b'.repeat(40), parent_sha: PREV, reviewed_subject: 'same',
    author_subject: '', author_disposition: '', author_reason: '', recorded_at: '2026-10-28T14:12:10+00:00', ...overrides });
// What the ledger writes for a commit an author decision governed: that decision, by value (review_doctrine).
const governed = (overrides = {}) => landing({ author_subject: SUBJECT, author_disposition: 'rejected', author_reason: REASON, ...overrides });
const decision = { disposition: 'rejected', rationale: REASON, subject_hash: SUBJECT, reviewer_signal: 'FAIL', enforcement: 'advisory',
    recorded_at: '2026-10-28T14:09:02+00:00', landing: { kind: 'commit_landing', landed: true, subject_hash: SUBJECT } };
const panel = (overrides = {}) => ({ record_id: 'rec-1', surface: 'commit_gate', record_state: 'settled', revision: 3,
    subject: { kind: 'index', tree_sha: 'b'.repeat(40) }, goal: 'Return focus to the composer after a message is sent',
    aggregate_signal: 'PASS', enforcement_impact: 'advisory', actors: seats('PASS'),
    source_ref: { kind: 'review_ledger_record', path: 'state/review_ledger/rec-1.json' }, ...overrides });
const adopted = (overrides = {}) => ({ adoption_id: 'a3c91e5f2b07', outcome: 'adopted', commit_sha: CAND, previous_sha: PREV,
    branch: 'ouroboros', note: 'Adopted candidate commit 4f1c2a9e7b3d on ouroboros; previous commit 9a8b7c6d5e4f.',
    settled_at: '2026-10-28T14:13:27+00:00', revision: 2, commits: [CAND], commits_complete: true, ...overrides });
const detail = (panels, adoptions) => ({ task_id: 'task-a', review_projection: { panels, review_records_omitted: 0 },
    ...(adoptions ? { body_adoptions: adoptions } : {}) });
const id = 'review_record:task-a:record:rec-1';
const html = (groups, expanded = true) => renderReviewsSection(groups, { sectionExpanded: true, expandedGroups: new Set(expanded ? [id] : []) });
const body = (markup) => markup.match(/<div class="review-landing" data-review-landing>[\s\S]*?<\/div>/)?.[0] || '';
const words = (markup) => markup.replace(/<[^>]+>/g, '').replace(/&#39;/g, "'").replace(/&amp;/g, '&');
const head = (markup) => words(markup.match(/<button type="button" class="review-check-head"[\s\S]*?<\/button>/)?.[0] || '');
const facts = (markup) => markup.match(/<details class="review-facts"[\s\S]*?<\/details>/)?.[0] || '';
const said = (record, adoptions) => landingWords(record, adoptions).line;

test('the four chosen scenes say the relation inside the opened row and keep each verdict', () => {
    const scenes = [
        ['PASS on the exact bytes', panel({ commit_landings: [landing()] }), 'passed', 'Adopted 4f1c2a9e7b3d: the exact bytes this review passed.', ''],
        ['FAIL, author continued', panel({ aggregate_signal: 'FAIL', actors: seats('FAIL'), author_decision: decision,
            commit_landings: [governed()] }), 'rejected', "Adopted 4f1c2a9e7b3d despite this rejection, by Ouroboros's own decision.", REASON],
        ['PASS of an earlier version', panel({ commit_landings: [governed({ reviewed_subject: 'different', author_disposition: 'accepted' })] }),
            'passed', 'This review was for earlier code; it does not cover adopted 4f1c2a9e7b3d.', REASON],
        ['no verdict', panel({ aggregate_signal: 'QUORUM_FAILED', commit_landings: [governed()] }), 'no verdict',
            "Adopted 4f1c2a9e7b3d with no verdict from this review, by Ouroboros's own decision.", REASON],
    ];
    for (const [name, record, state, line, reason] of scenes) {
        const markup = html(reviewGroupsFromTaskDetail(detail([record], [adopted()])));
        const lead = words(body(markup));
        assert.ok(lead.startsWith(line), `${name}: ${lead}`);
        assert.equal(lead.includes(`Reason: ${REASON}`), Boolean(reason), name);
        const headMarkup = markup.match(/<button type="button" class="review-check-head"[\s\S]*?<\/button>/)[0];
        assert.match(headMarkup, new RegExp(`class="review-state ui-status" data-tone="(ok|error|warn)">${state}</span>`), name);
        assert.equal(head(markup).includes('earlier version'), name === 'PASS of an earlier version', name);
        assert.doesNotMatch(lead, /PASS|FAIL|QUORUM|DEGRADED|UNKNOWN/, name);  // tokens stay in Record facts
    }
    // An unreadable record says nothing about any commit; the adoption stays the task's own Details row.
    const gap = html(reviewGroupsFromTaskDetail({ task_id: 'task-a', body_adoptions: [adopted()],
        review_projection: { panels: [], gaps: [{ record_id: 'rec-1', review_gap: 'missing' }], review_records_omitted: 0 } }),
    false);
    assert.equal(body(gap), '');
    assert.match(words(gap), /could not be read/);
    assert.doesNotMatch(words(gap), /Adopted|Committed/);
});

test('the relation leads the seats only in the opened row, whole at any width; only the reason may be cut', () => {
    const groups = reviewGroupsFromTaskDetail(detail([panel({ commit_landings: [governed({ reviewed_subject: 'different' })] })], [adopted()]));
    const closed = html(groups, false);
    assert.match(closed, /<div class="chat-review-attempts review-check-body" hidden><div class="review-seats"><div class="review-landing"/);
    assert.equal(head(closed).startsWith('Code review · earlier version · Return focus'), true, head(closed));
    assert.doesNotMatch(head(closed), /cover|Adopted|Reason/);
    const opened = html(groups);
    // The relation is never clamped (`review-line-whole`); the reason keeps its two-line cut and its whole text in Record facts.
    assert.match(opened, /<div class="review-seats"><div class="review-landing" data-review-landing><p class="review-line review-line-whole"><span class="review-line-text">This review was for earlier code/);
    assert.match(body(opened), /<p class="review-line review-line-wrap review-line-quiet"><span class="review-line-by">Reason: <\/span>/);
    assert.ok(opened.indexOf('data-review-landing') < opened.indexOf('data-review-opinion='), 'the relation leads the seats');
});

test('"Adopted" needs the task\'s recorded adoption of that commit, exactly or within its adopted range', () => {
    const one = panel({ commit_landings: [landing()] });
    assert.equal(said(one), 'Committed 4f1c2a9e7b3d: the exact bytes this review passed.');
    for (const outcome of ['not_applied', 'superseded', 'unconfirmed']) {
        assert.equal(said(one, [adopted({ outcome, commits: undefined, commits_complete: undefined })]),
            'Committed 4f1c2a9e7b3d: the exact bytes this review passed.', outcome);
    }
    // Another adoption whose whole range is recorded did not take this commit.
    assert.equal(said(one, [adopted({ commit_sha: 'c'.repeat(40), commits: ['c'.repeat(40)] })]), 'Committed 4f1c2a9e7b3d: the exact bytes this review passed.');
    // An earlier commit of the adopted range was adopted as part of the adopted commit, in every relation.
    const early = (overrides) => panel({ commit_landings: [landing({ commit_sha: EARLY, ...overrides.landing })], ...overrides.panel });
    const range = [adopted({ commits: [CAND, EARLY] })];
    assert.equal(said(early({}), range), 'Adopted 1a2b3c4d5e6f as part of 4f1c2a9e7b3d: the exact bytes this review passed.');
    assert.equal(said(early({ panel: { aggregate_signal: 'FAIL' }, landing: { author_subject: SUBJECT } }), range),
        "Adopted 1a2b3c4d5e6f as part of 4f1c2a9e7b3d despite this rejection, by Ouroboros's own decision.");
    assert.equal(said(early({ panel: { aggregate_signal: 'QUORUM_FAILED' }, landing: { author_subject: SUBJECT } }), range),
        "Adopted 1a2b3c4d5e6f as part of 4f1c2a9e7b3d, with no verdict from this review, by Ouroboros's own decision.");
    assert.equal(said(early({ landing: { reviewed_subject: 'different' } }), range),
        'This review was for earlier code; it does not cover 1a2b3c4d5e6f, adopted as part of 4f1c2a9e7b3d.');
    assert.equal(said(early({ landing: { reviewed_subject: 'unknown' } }), range),
        'Adopted 1a2b3c4d5e6f as part of 4f1c2a9e7b3d; whether this review saw these bytes is not recorded.');
    for (const sentence of [said(early({ panel: { aggregate_signal: 'QUORUM_FAILED' } }), range), said(early({}), range)]) {
        assert.equal(sentence.match(/\bwith\b/g)?.length || 0, sentence.includes('no verdict') ? 1 : 0, sentence);
    }
    // A range whose list was cut (or never read) cannot say it lacked the commit: it says so, never "not adopted".
    for (const cut of [{ commits: [CAND], commits_complete: false, commits_omitted: 70 }, { commits: undefined, commits_complete: undefined }]) {
        assert.equal(said(early({}), [adopted(cut)]),
            'Committed 1a2b3c4d5e6f: the exact bytes this review passed. Whether the adoption of 4f1c2a9e7b3d included it is not recorded.');
    }
    // A record that saw two commits names the one the body adopted; without an adoption, the newest.
    const two = panel({ commit_landings: [landing(), landing({ commit_sha: EARLY })] });
    assert.equal(said(two, [adopted()]), 'Adopted 4f1c2a9e7b3d: the exact bytes this review passed.');
    assert.equal(said(two, []), 'Committed 1a2b3c4d5e6f: the exact bytes this review passed.');
});

test('each landing speaks with the decision that governed it, never a later one on the same record', () => {
    // Adopted A after a rejection, then B (another decision, other code) committed under the same record and not adopted.
    const a = governed({ commit_sha: EARLY, author_reason: 'Reason for adopted commit A' });
    const b = landing({ reviewed_subject: 'different', author_subject: LATER, author_disposition: 'accepted', author_reason: 'Reason for later commit B' });
    const record = panel({ aggregate_signal: 'FAIL', actors: seats('FAIL'), commit_landings: [a, b],
        author_decision: { ...decision, disposition: 'accepted', subject_hash: LATER, rationale: 'Reason for later commit B' } });
    const shown = landingWords(record, [adopted({ commit_sha: EARLY, commits: [EARLY] })]);
    assert.deepEqual([shown.line, shown.reason],
        ["Adopted 1a2b3c4d5e6f despite this rejection, by Ouroboros's own decision.", 'Reason for adopted commit A']);
    const newest = landingWords(record, []);
    assert.deepEqual([newest.line, newest.reason, newest.earlier],
        ['This review was for earlier code; it does not cover committed 4f1c2a9e7b3d.', 'Reason for later commit B', true]);
    // No governing decision: no attribution and no reason, whatever the record's current decision says.
    const plain = landingWords(panel({ aggregate_signal: 'FAIL', author_decision: decision, commit_landings: [landing()] }), [adopted()]);
    assert.deepEqual([plain.line, plain.reason], ['Adopted 4f1c2a9e7b3d despite this rejection.', '']);
    // A governing decision whose words were not kept is still named, without a reason it cannot show.
    const unworded = landingWords(panel({ aggregate_signal: 'FAIL', commit_landings: [landing({ author_subject: SUBJECT })] }), [adopted()]);
    assert.deepEqual([unworded.line, unworded.reason], ["Adopted 4f1c2a9e7b3d despite this rejection, by Ouroboros's own decision.", '']);
    const unknown = landingWords(panel({ commit_landings: [landing({ reviewed_subject: 'unknown' })] }), [adopted()]);
    assert.deepEqual([unknown.line, unknown.earlier], ['Adopted 4f1c2a9e7b3d; whether this review saw these bytes is not recorded.', false]);
    // A record written before landings were recorded (only the author's own landing) stays silent: unknown, never unreviewed.
    assert.deepEqual(landingWords(panel({ author_decision: decision }), [adopted()]), { line: '', reason: '', earlier: false });
    assert.deepEqual(landingWords(null, [adopted()]), { line: '', reason: '', earlier: false });
});

test('Record facts keep every shown landing with its own decision and whole reason, and name where the rest is', () => {
    const record = panel({ aggregate_signal: 'FAIL', author_decision: decision, commit_landings_omitted: 3,
        commit_landings: [landing({ commit_sha: EARLY, reviewed_subject: 'different', author_subject: LATER, author_disposition: 'accepted',
            author_reason: 'Reason for the earlier commit' }), governed()] });
    const markup = facts(html(reviewGroupsFromTaskDetail(detail([record], [adopted()]))));
    assert.match(markup, new RegExp(`<dt>Committed</dt><dd>commit=${EARLY} · tree=b{40} · parent=${PREV} · reviewed_subject=different · author_subject=${LATER} · author_disposition=accepted`));
    assert.match(markup, /<dt>Author reason<\/dt><dd>Reason for the earlier commit<\/dd>/);
    assert.match(markup, new RegExp(`<dt>Committed</dt><dd>commit=${CAND} · tree=b{40} · parent=${PREV} · reviewed_subject=same · author_subject=${SUBJECT} · author_disposition=rejected`));
    assert.match(markup, new RegExp(`<dt>Author decision</dt><dd>disposition=rejected · reviewer_signal=FAIL · enforcement=advisory · subject=${SUBJECT}`));
    assert.equal(words(markup).split(`Author reason${REASON}`).length, 2, 'the whole reason, up to its END-OF-REASON, once');
    assert.match(markup, /<dt>Earlier landings<\/dt><dd>3 not shown · state\/review_ledger\/rec-1\.json<\/dd>/);
    assert.match(markup, /<dt>Verdict<\/dt><dd>FAIL/);
});

test('a later source without the adoption fact never erases what an earlier read said', () => {
    const store = new Map();
    const record = panel({ commit_landings: [landing()] });
    const [first] = reviewGroupsFromTaskDetail(detail([record], [adopted()]));
    mergeReviewGroup(store, first);
    const [summary] = reviewGroupsFromTaskDetail(detail([record]));  // a summary row or a history page: no field
    assert.equal('adoptions' in summary, false);
    mergeReviewGroup(store, summary);
    assert.match(words(body(html(store))), /^Adopted 4f1c2a9e7b3d: the exact bytes this review passed\./);
    const [revised] = reviewGroupsFromTaskDetail(detail([record], [adopted({ outcome: 'superseded' })]));
    mergeReviewGroup(store, revised);  // a field that IS present replaces the earlier one
    assert.match(words(body(html(store))), /^Committed 4f1c2a9e7b3d/);
});

test('a history row read before the adoption settled cannot take the Reviews back to an earlier revision', () => {
    // Boot order: the server serves before it settles, so a history read can hold the unconfirmed row (1), the
    // task detail then brings the adoption (2), and that late history row arrives last. Both reach the card
    // through the same controller (chat.js: the summary row and the detail hydration).
    const record = panel({ commit_landings: [landing()] });
    const waiting = adopted({ outcome: 'unconfirmed', revision: 1, commits: undefined, commits_complete: undefined,
        note: 'Body 4f1c2a9e7b3d is running, but the native host did not confirm its artifact; the adoption stays unconfirmed.' });
    const other = adopted({ adoption_id: 'b7d2e4c6a8f0', commit_sha: 'c'.repeat(40), commits: ['c'.repeat(40)], revision: 1 });
    const controller = createReviewPresentationController({});
    const read = (adoptions) => controller.updateMany(reviewGroupsFromTaskDetail(detail([record], adoptions), 'task-a'));
    const shown = () => words(body(html(controller.groups)));
    const known = () => controller.groups.get('review_record:task-a').adoptions?.map((row) => [row.adoption_id, row.outcome, row.revision]);
    read([waiting]);
    assert.match(shown(), /^Committed 4f1c2a9e7b3d: the exact bytes this review passed\./);
    read([adopted()]);
    assert.match(shown(), /^Adopted 4f1c2a9e7b3d: the exact bytes this review passed\./);
    const late = read([waiting]);
    assert.match(shown(), /^Adopted 4f1c2a9e7b3d: the exact bytes this review passed\./);
    assert.equal(late, false, 'the late row is nothing newer');
    // Each adoption keeps its own highest revision; one a source does not name stays known, absence adds nothing.
    read([other]);
    read(undefined);
    assert.deepEqual(known(), [['a3c91e5f2b07', 'adopted', 2], ['b7d2e4c6a8f0', 'adopted', 1]]);
    read([adopted({ outcome: 'superseded', revision: 3, commits: undefined, commits_complete: undefined }), waiting]);
    assert.deepEqual(known(), [['a3c91e5f2b07', 'superseded', 3], ['b7d2e4c6a8f0', 'adopted', 1]]);
    assert.match(shown(), /^Committed 4f1c2a9e7b3d/);
    const fresh = new Map();
    mergeReviewGroup(fresh, reviewGroupsFromTaskDetail(detail([record]))[0]);
    mergeReviewGroup(fresh, reviewGroupsFromTaskDetail(detail([record]))[0]);
    assert.equal('adoptions' in fresh.get('review_record:task-a'), false, 'no source said anything: unknown, never "none adopted"');
});

// The Python side (tests/test_review_commit_landing.py) writes what two REAL commits under one record left: the
// stored task detail and the history page's summary row a cold reload reads. This consumer renders both.
test('two real commits under one record: the adopted one keeps its own decision after a cold reload', { skip: !process.env.OURO_TEST_LANDING_SOURCES }, () => {
    const sources = JSON.parse(readFileSync(process.env.OURO_TEST_LANDING_SOURCES, 'utf8'));
    for (const [where, source] of [['task detail', sources.detail], ['history summary row', sources.summary]]) {
        const [group] = reviewGroupsFromTaskDetail(source, sources.task_id).filter((row) => row.surface === 'review_record');
        assert.ok(group, where);
        const markup = renderReviewsSection([group], { sectionExpanded: true, expandedGroups: new Set([group.id]) });
        const lead = words(body(markup));
        assert.equal(lead, `Adopted ${sources.adopted.slice(0, 12)} despite this rejection, by Ouroboros's own decision.`
            + `Reason: ${sources.adopted_reason}`, where);
        const record = words(facts(markup));
        assert.ok(record.includes(`Author reason${sources.adopted_reason}`) && record.includes(`Author reason${sources.later_reason}`), where);
        assert.ok(record.includes(`commit=${sources.later}`) && !lead.includes(sources.later_reason), where);
    }
});
