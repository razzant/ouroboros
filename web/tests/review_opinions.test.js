import test from 'node:test';
import assert from 'node:assert/strict';
import { createReviewPresentationController, reviewGroupFromHistoryRow, reviewGroupsFromTaskDetail, renderReviewsSection } from '../modules/review_presentation.js';
import { taskReviewSourceUrl, taskSkillReviewSourceUrl } from '../modules/api_client.js';
import { formatReviewProjection, reviewStateWords, seatStateView } from '../modules/review_record_card.js';

// Selected Reviews A (owner 25.10, quiz answer A): rows say what was reviewed and each reviewer's outcome in
// words; the recorded tokens, identifiers and codes stay in the one Record facts disclosure.
const panel = {
    record_id: 'record-a', surface: 'commit_gate', revision: 2, review_wave_id: 'wave-a', record_state: 'settled',
    subject: { kind: 'worktree', tree_sha: 'old-tree' }, goal: 'Readable saved Reviews',
    aggregate_signal: 'PASS', enforcement_impact: 'advisory', panel_facts: { strictness: 'advisory', pool: ['a', 'b'], seats: 2, chosen_by: 'ouroboros' },
    actors: [
        { slot_id: 'seat-one', model: 'same/model', observed_model: 'same/model', transport_status: 'success', parse_status: 'valid', semantic_verdict: 'PASS',
            findings: [{ item: 'A useful note', reason: 'Still present with PASS' }], findings_omitted: 4,
            sources: [{ role: 'response', part: 'change,coupling', status: 'retained', sha256: 'a'.repeat(64) }] },
        { slot_id: 'seat-two', model: '', requested_model: 'same/model', observed_model: '', transport_status: 'error', parse_status: 'none',
            failure_code: 'native_transcript_cap_exceeded', reason: 'native_transcript_cap_exceeded',
            reported_cause: 'Turn too long: 900000 characters, limit 850000', failure_details: { units: 'characters', limit: 850000, measured: 900000 } },
        { slot_id: 'seat-three', model: 'same/model', observed_model: 'same/model', transport_status: 'success', parse_status: 'valid', semantic_verdict: 'PASS', findings: [] },
    ],
};
const detail = (overrides = {}, projection = {}) => ({ task_id: 'task-a', review_projection: { panels: [{ ...panel, ...overrides }], review_records_omitted: '1+', ...projection } });
const groupId = 'review_record:task-a:record:record-a';
const open = (groups, id = groupId) => renderReviewsSection(groups, { sectionExpanded: true, expandedGroups: new Set([id]) });
// What the owner reads on the rows, without the Record facts that keep the recorded tokens.
const rows = (markup) => markup.replace(/<details class="review-facts"[\s\S]*?<\/details>/g, '');
const facts = (markup) => markup.match(/<details class="review-facts"[\s\S]*?<\/details>/)?.[0] || '';
// The words the rows show, without markup or attributes.
const visible = (markup) => rows(markup).replace(/<[^>]+>/g, ' ');
// Every state word the rows show (status sentence or the moving mark's word).
const states = (markup) => [...rows(markup).matchAll(/class="review-state[^"]*" data-tone="[^"]+">(?:<span class="status-mark"[^>]*>(?:<i><\/i>){3}<\/span><span class="review-state-word">)?([^<]*)</g)].map((m) => m[1]);
const seat = (markup, slot) => markup.match(new RegExp(`<div class="review-seat" data-review-opinion="[^"]*:${slot}">[\\s\\S]*?</div></div>`))?.[0] || '';

test('a reviewed subject opens to one compact line per reviewer in words, the tokens kept in Record facts', () => {
    const html = open(reviewGroupsFromTaskDetail(detail()));
    assert.match(html, /<span class="review-check-subject">Code review · Readable saved Reviews<\/span><span class="review-state ui-status" data-tone="ok">passed<\/span>/);
    assert.match(html, /<span class="review-check-meta">[^<]*2 of 3 counted answered · 1 unavailable<\/span>/);
    // Same model twice stays two reviewers; a PASS keeps its notes.
    assert.match(seat(html, 'seat-one'), /<span class="review-seat-name">same\/model<\/span><span class="review-seat-role">counted<\/span><span class="review-state ui-status" data-tone="ok">passed it · 5 notes<\/span>/);
    assert.match(seat(html, 'seat-one'), /<span class="review-sev" data-tone="neutral">Note<\/span><span class="review-line-text">A useful note: Still present with PASS<\/span>/);
    assert.match(seat(html, 'seat-three'), /data-tone="ok">passed it<\/span>[\s\S]*No finding reported\./);
    // A refusal: the requested model is labelled as requested, the engine's own sentence and its typed limit.
    const refused = seat(html, 'seat-two');
    assert.match(refused, /<span class="review-seat-name">same\/model<\/span><span class="review-seat-note">requested · not reported<\/span>/);
    assert.match(refused, /data-tone="warn">unavailable<\/span>/);
    assert.match(refused, /<span class="review-line-by">Engine report: <\/span><span class="review-line-text">Turn too long: 900000 characters, limit 850000<\/span>/);
    assert.match(refused, /<span class="review-seat-meta">900,000 characters, limit 850,000 characters<\/span>/);
    assert.doesNotMatch(visible(html), /native_transcript_cap_exceeded|seat-one|record-a/);
    assert.ok(states(html).every((word) => !/PASS|FAIL|DEGRADED|UNKNOWN/.test(word)), String(states(html)));
    assert.doesNotMatch(html, /Show details|retry.*commit/i);
    // Read hands the exact retained source to the reader; no path, no regenerated review.
    const url = taskReviewSourceUrl('task-a', 'record-a', 'seat-one', { role: 'response', part: 'change,coupling', sha256: 'a'.repeat(64) });
    assert.ok(seat(html, 'seat-one').includes(`data-review-read-url="${url.replaceAll('&', '&amp;')}"`));
    assert.match(seat(html, 'seat-one'), /aria-label="Read same\/model answer">Read<svg/);
    // Record facts: one disclosure for the group, with the record, the verdict token and requested vs observed.
    assert.equal((html.match(/data-review-facts=/g) || []).length, 1);
    assert.match(facts(html), /<dt>Record<\/dt><dd>record-a · revision 2<\/dd>/);
    assert.match(facts(html), /<dt>Verdict<\/dt><dd>PASS<\/dd>/);
    assert.match(facts(html), /<dt>Reviewer seat-two model<\/dt><dd>requested same\/model · observed not reported<\/dd>/);
    assert.match(facts(html), /<dt>Reviewer seat-two failure<\/dt><dd>failure_code=native_transcript_cap_exceeded/);
    assert.match(facts(html), /<dt>Reviewer seat-one saved answer<\/dt><dd>part=change,coupling · status=retained · sha256=a{64}<\/dd>/);
    assert.match(html, /<span class="review-check-line">Advisory · chosen by Ouroboros · 2 of 2 ticked reviewers<\/span>/);
});

test('no aggregate or seat token reaches a row: every reduction reads in the shared words', () => {
    const words = { PASS: ['passed', 'ok'], FAIL: ['rejected', 'error'], QUORUM_FAILED: ['no verdict', 'warn'],
        NOT_PERFORMED: ['no verdict', 'warn'], NOT_DISPATCHED: ['not sent', 'warn'], UNKNOWN: ['verdict not recorded', 'warn'] };
    for (const [token, [word, tone]] of Object.entries(words)) {
        const html = open(reviewGroupsFromTaskDetail(detail({ aggregate_signal: token })));
        assert.match(html, new RegExp(`class="review-state ui-status" data-tone="${tone}">${word}</span>`), token);
        assert.ok(states(html).every((word) => !word.includes(token)), `${token}: ${states(html)}`);
        assert.match(facts(html), new RegExp(`<dt>Verdict</dt><dd>${token}`), token);
    }
    for (const [verdict, word] of [['DEGRADED', 'inconclusive'], ['UNKNOWN', 'outcome unknown']]) {
        const actor = { slot_id: 's', semantic_verdict: verdict, parse_status: 'valid', transport_status: 'success' };
        assert.equal(seatStateView(actor, panel).word, word);
    }
    assert.equal(reviewStateWords('PASS so far · 2 of 3 answered'), 'passed so far · 2 of 3 answered');
    assert.equal(reviewStateWords('2 objections may hold'), '2 objections may hold');
});

test('an owed seat reads with the existing wait owners, never a second definition', () => {
    const settled = { record_state: 'settled' };
    const pending = { record_state: 'pending' };
    const since = '2026-10-25T09:05:00+00:00';
    assert.match(seatStateView({ operation_state: 'pending_dispatch', transport_status: 'pending', awaiting_since: since }, pending).word, /^awaiting · since /);
    assert.equal(seatStateView({ operation_state: 'in_flight', transport_status: 'pending' }, pending, { live: true }).tone, 'working');
    assert.equal(seatStateView({ operation_state: 'in_flight', transport_status: 'pending' }, pending).tone, 'neutral');
    // A settled record's lost seat is terminal: its outcome is unknown, never awaited.
    assert.deepEqual(seatStateView({ operation_state: 'custody_lost' }, settled), { word: 'outcome unknown', tone: 'neutral', findings: 0 });
    assert.equal(seatStateView({ operation_state: 'custody_lost' }, pending).word, 'awaiting');
    assert.equal(seatStateView({ transport_status: 'not_dispatched' }, settled).word, 'not sent');
    // The same record's facts call the same seats awaited.
    const html = open(reviewGroupsFromTaskDetail({ task_id: 'task-a', status: 'running', review_projection: { panels: [{ ...panel, record_state: 'pending',
        aggregate_signal: 'NOT_PERFORMED', actors: [panel.actors[0], { slot_id: 'late', operation_state: 'in_flight', transport_status: 'pending', parse_status: 'none', semantic_verdict: 'pending' }] }] } }));
    assert.match(html, /class="review-check-head"[\s\S]*?class="review-state" data-tone="working"><span class="status-mark"[^>]*data-shape="moving"[\s\S]*?review-state-word">in progress · 1 of 2 answered</);
    assert.match(seat(html, 'late'), /class="review-state" data-tone="working">[\s\S]*?review-state-word">awaiting</);
    assert.match(facts(html), /<dt>Awaiting an answer<\/dt><dd>late<\/dd>/);
});

test('one unreadable record stays a local gap beside its neighbours, and Logs say so too', () => {
    const projection = { panels: [panel], gaps: [{ record_id: 'broken', review_gap: 'missing' }], review_records_omitted: 0 };
    const groups = reviewGroupsFromTaskDetail({ task_id: 'task-a', review_projection: projection });
    const html = renderReviewsSection(groups, { sectionExpanded: true, expandedGroups: new Set(['review_record:task-a:gap:broken']) });
    assert.match(html, /Code review · Readable saved Reviews/);
    assert.match(html, /<span class="review-check-subject">Review record unavailable<\/span><span class="review-state ui-status" data-tone="warn">could not be read<\/span>/);
    assert.match(html, /<span class="review-check-meta">record missing<\/span>/);
    assert.match(html, /This saved review could not be read\. The readable reviews beside it are unaffected\./);
    assert.match(facts(html.slice(html.indexOf('gap:broken'))), /Review record unavailable \(missing\)<\/dt><dd>broken<\/dd>|<dd class="review-facts-note">Review record unavailable \(missing\): broken<\/dd>/);
    const only = reviewGroupsFromTaskDetail({ task_id: 'task-gap', review_projection: { panels: [], gaps: [{ record_id: 'broken', review_gap: 'unreadable' }], review_records_omitted: 0 } });
    assert.equal(only.length, 1);
    assert.match(formatReviewProjection(projection), /\nReview record unavailable \(missing\): broken$/);
});

test('an omitted count alone makes no Reviews group, and uncertain history says may', () => {
    for (const omitted of ['unknown', '1+', 3, 0]) {
        assert.deepEqual(reviewGroupsFromTaskDetail({ task_id: 'quiet', review_projection: { panels: [], review_records_omitted: omitted } }), [], String(omitted));
    }
    const certain = renderReviewsSection(reviewGroupsFromTaskDetail(detail()), { sectionExpanded: true, expandedGroups: new Set(), historyTask: 'task-a', historyMore: true, historyCertain: true });
    assert.match(certain, /Earlier reviews outside this view<\/span><button type="button" class="review-read ui-hit-area" data-review-history>Read earlier reviews<\/button>/);
    const maybe = renderReviewsSection(reviewGroupsFromTaskDetail(detail({}, { review_records_omitted: 'unknown' })),
        { sectionExpanded: true, expandedGroups: new Set(), historyTask: 'task-a', historyMore: true, historyCertain: false });
    assert.match(maybe, /Earlier reviews may be outside this view/);
    const [group] = reviewGroupsFromTaskDetail(detail({}, { review_records_omitted: 0 }));
    assert.equal(group.historyMore, false);
});

test('source selectors are identity-bound and contain no filesystem location', () => {
    const source = taskReviewSourceUrl('t', 'record', 'slot', { role: 'failure', part: 'change', sha256: 'f'.repeat(64) });
    assert.match(source, /^\/api\/tasks\/t\/reviews\/record\/slot\?/);
    assert.match(source, /role=failure/);
    assert.doesNotMatch(source, /path=/);
    // A saved Skill review is read through its task's own review door (the desktop file bridge serves /api/tasks).
    assert.equal(taskSkillReviewSourceUrl('child task', 'job/1', 'alpha beta'), '/api/tasks/child%20task/reviews/skill/job%2F1?skill=alpha+beta&format=text');
});

test('two saved texts on one seat get two distinct Read names', () => {
    const twice = { ...panel.actors[0], sources: [
        { role: 'response', part: 'change,coupling', status: 'retained', sha256: 'a'.repeat(64) },
        { role: 'failure', part: 'change,coupling', status: 'retained', sha256: 'b'.repeat(64) },
    ] };
    const html = open(reviewGroupsFromTaskDetail(detail({ actors: [twice] })));
    assert.match(html, /aria-label="Read same\/model answer">Read answer<svg/);
    assert.match(html, /aria-label="Read same\/model failure note">Read failure note<svg/);
    // An answer opens as Markdown in the reader (Source keeps it exact); a failure note stays plain text.
    assert.match(html, /data-review-read-sha="a{64}" data-review-read-kind="markdown"/);
    assert.match(html, /data-review-read-sha="b{64}" data-review-read-kind="text"/);
});

test('a plan participant contributes and objects; nobody votes and the host fold decides what stands', () => {
    const groups = reviewGroupsFromTaskDetail({ task_id: 'plan-owner', status: 'completed', plan_record: { schema: 3,
        revisions: [{ n: 1, goal: 'Plan reader', done_when: [] }], asks: [{ ask_id: 'ask-a', revision: 1, host_added: ['critic-two'], participants: [
            { seat_id: 'critic', model: 'same/model', status: 'answered', owner_marked: true,
                reply: { prose: 'The plan needs one concrete check.', objections: [{ id: 'o1', text: 'Missing check' }], withdrawn: ['o0'] }, reply_ref: { ref: { sha256: 'b'.repeat(64) } } },
            { seat_id: 'critic-two', model: 'other/model', status: 'answered', owner_marked: false, reply: { prose: 'Fine.', objections: [{ id: 'o2', text: 'Naming' }] } },
        ] }] },
        plan_objections: { open_objections: {
            critic: [{ id: 'o1', text: 'Missing check', ask_id: 'ask-a', revision: 1, holds: true, not_adjudicated: true }],
            'critic-two': [{ id: 'o2', text: 'Naming', ask_id: 'ask-a', revision: 1, holds: false }],
        } },
    });
    const html = open(groups, 'plan:plan-owner');
    assert.match(html, /<span class="review-check-subject">Plan · revision 1<\/span><span class="review-state ui-status" data-tone="warn">1 objection may hold<\/span>/);
    assert.match(html, /1 ask · 2 participants · nobody votes/);
    assert.match(seat(html, 'critic'), /<span class="review-seat-role">counts<\/span><span class="review-state ui-status" data-tone="warn">1 objection may hold · 1 withdrawn<\/span>/);
    assert.match(seat(html, 'critic'), /<span class="review-sev" data-tone="warn">May hold<\/span><span class="review-line-text">Missing check<\/span>/);
    assert.match(seat(html, 'critic-two'), /added — never holds<\/span><span class="review-state ui-status" data-tone="neutral">1 objection, never holds<\/span>/);
    assert.match(seat(html, 'critic'), /domain=plan/);
    assert.ok(states(html).every((word) => !/PASS|FAIL/.test(word)));
    // The host's own facts stay whole in Record facts: who it added, the fold's standing, the withdrawal.
    assert.match(facts(html), /<dt>first ask on a body plan<\/dt><dd>the host added critic-two<\/dd>/);
    assert.match(facts(html), /\(not adjudicated\)/);
    assert.match(facts(html), /<dt>Withdrawn by critic<\/dt><dd>o0<\/dd>/);
    assert.match(facts(html), /<dt>Participant critic<\/dt><dd>status=answered · model=same\/model · reply_sha256=b{64}<\/dd>/);
});

test('a saved Skill review reads its normalized text through the initiating task, labelled as saved', () => {
    const row = (attempt) => ({ system_type: 'skill_review', skill: 'alpha', status: 'clean', job_id: 'job-2',
        review_group: { surface: 'skill', id: 'task:root:alpha', presentation_owner_task_id: 'root', projected_attempt_count: 1,
            count_is_authoritative: true, attempts: [{ job_id: 'job-2', skill: 'alpha', status: 'clean', review_round: 1, snapshot_attempt: 1, ...attempt }] } });
    for (const [attempt, owner] of [[{ initiator_task_id: 'initiator-child' }, 'initiator-child'], [{}, 'root']]) {
        const group = reviewGroupFromHistoryRow(row(attempt));
        const html = renderReviewsSection([group], { sectionExpanded: true, expandedGroups: new Set([group.id]) });
        assert.ok(html.includes(`data-review-read-url="${taskSkillReviewSourceUrl(owner, 'job-2', 'alpha').replaceAll('&', '&amp;')}"`), owner);
        assert.match(html, /aria-label="Read the saved skill review of alpha">Read saved review<svg/);
        assert.match(html, /data-review-read-name="alpha · saved skill review"/);
        assert.doesNotMatch(html, /\/api\/skills\//);
    }
});

test('Read opens an answer as Markdown and a failure note as plain text, never regenerating a review', async () => {
    let click;
    const opened = [];
    const controller = createReviewPresentationController({
        host: { addEventListener(_event, handler) { click = handler; } },
        onRead: (...args) => opened.push(args[0]),
    });
    for (const [kind, name] of [['markdown', 'openai/gpt-5.6-sol · answer'], ['text', 'same/model · failure note']]) {
        const opener = { dataset: { reviewReadUrl: `/api/tasks/t/reviews/r/s?sha256=${'a'.repeat(64)}`, reviewReadName: name, reviewReadKind: kind } };
        await click({ target: { closest: selector => (selector === '[data-review-read-url]' ? opener : null) } });
    }
    assert.deepEqual(opened.map(file => [file.kind, file.mime, file.filename]), [
        ['markdown', 'text/markdown', 'openai-gpt-5.6-sol · answer.md'],
        ['text', 'text/plain', 'same-model · failure note.txt'],
    ]);
    assert.equal(controller.groups.size, 0);
});
