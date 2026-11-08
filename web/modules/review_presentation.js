import { acceptanceGroupWithIncident, acceptanceIncidentFromTaskDetail } from './acceptance_incident_presentation.js';
export { acceptanceIncidentFromTaskDetail } from './acceptance_incident_presentation.js';
import { setInertCardPresentation } from './task_phase_chip.js';
import { escapeHtmlAttr, sinceLocalTime } from './utils.js';
import { isStoreMigrating, taskSourceDownloadUrl } from './api_client.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import { reconcileReviewMarkup } from './review_dom_patch.js';
import {
    actorAnswered, actorAwaiting, actorUnresolved, authorDoctrineLines, formatReviewProjection, seatAdviser, seatOwed,
} from './review_record_card.js';
import { planRecordEngaged, planRecordGroup } from './plan_record_presentation.js';
import { renderOpinionReviews, reviewViews } from './review_opinions.js';
import { getTaskReviewHistory } from './api_client.js';
export { formatReviewProjection } from './review_record_card.js';

const escapeHtmlText = escapeHtmlAttr;

const SURFACE_ORDER = new Map([
    ['skill', 0],
    ['plan', 1],
    ['task_acceptance', 2],
    ['review_record', 3],
]);

const ACTIVE_STATES = new Set(['queued', 'running', 'open', 'working', 'pending']);
const ERROR_STATES = new Set(['failed', 'error', 'blocked', 'blockers', 'timeout']);
const WARNING_STATES = new Set([
    'warnings', 'warning', 'degraded', 'interrupted', 'cancelled',
    'review_required', 'revise_plan', 'rail_degraded', 'cycles_exhausted',
]);
const SUCCESS_STATES = new Set([
    'clean', 'passed', 'pass', 'succeeded', 'success', 'completed', 'closed', 'green',
]);
const TERMINAL_STATES = new Set([
    ...SUCCESS_STATES, 'terminal',
    ...ERROR_STATES, ...WARNING_STATES,
]);
// These values belong to the lifecycle of the review job, not to the
// review's semantic verdict. Keep them terminal for activity purposes while
// preventing a lifecycle-only row from looking like a review verdict.
const LIFECYCLE_SUCCESS_STATES = new Set(['succeeded', 'success', 'completed']);
const LIFECYCLE_TERMINAL_STATES = new Set([
    ...LIFECYCLE_SUCCESS_STATES,
    'failed', 'error', 'timeout', 'interrupted', 'cancelled',
]);

const text = (value) => String(value ?? '').trim();
const finiteCount = (value) => {
    if (value === null || value === undefined || value === '') return null;
    const number = Number(value);
    return Number.isFinite(number) && number >= 0 ? Math.trunc(number) : null;
};
// A roster with an unanswered slot reports how far it got, never a bare verdict;
// a slot that is neither answered nor awaited keeps the warning tone.
const heldProgress = (lead, roster, answered, awaited = actorAwaiting) => {
    const unavailable = roster.filter((actor) => !awaited(actor) && !answered(actor)).length;
    return [`${lead} · ${roster.filter(answered).length} of ${roster.length} answered${unavailable ? ` · ${unavailable} unavailable` : ''}`, unavailable ? 'warn' : ''];
};

export function setReviewAnchor(record, enabled, writePhase) {
    if (!record || (record.historicalUnavailable || record.historicalUnconfirmed) || Boolean(record.reviewAnchor) === enabled) return false;
    record.reviewAnchor = enabled;
    setInertCardPresentation(record, enabled);
    if (enabled) {
        record.titleEl.textContent = record.suggestedName || 'Reviews';
    } else {
        writePhase(record, 'working');
        if (!record.suggestedName && !record.lastHumanHeadline) {
            record.titleEl.textContent = '';  // the chip says Working; the title waits for a name (#1369)
        }
    }
    return true;
}

function normalizedState(value, fallback = 'unavailable') {
    const state = text(value).toLowerCase();
    if (ACTIVE_STATES.has(state)) {
        if (state === 'pending') return 'queued';
        return state === 'open' || state === 'working' ? 'running' : state;
    }
    if (TERMINAL_STATES.has(state)) return 'terminal';
    if (state === 'unavailable' || state === 'absent') return 'unavailable';
    if (state === 'superseded') return 'superseded';
    return fallback;
}

function statusTone(state, verdict = '') {
    const token = text(verdict || state).toLowerCase();
    if (ERROR_STATES.has(token) || token === 'fail') return 'error';
    if (WARNING_STATES.has(token)) return 'warn';
    if (ACTIVE_STATES.has(token) || state === 'running' || state === 'queued') return 'working';
    if (SUCCESS_STATES.has(token)) return 'done';
    return 'neutral';
}

function lifecycleOnlyTone(lifecycleStatus, state) {
    const token = text(lifecycleStatus).toLowerCase();
    // Successful job completion is deliberately not a semantic PASS.  Other
    // lifecycle failures still carry a useful severity fact even though their
    // review verdict is unavailable.
    if (LIFECYCLE_SUCCESS_STATES.has(token)) return 'neutral';
    return statusTone(state, token);
}

function lifecycleFailure(status) {
    const token = text(status).toLowerCase();
    return LIFECYCLE_TERMINAL_STATES.has(token)
        && !LIFECYCLE_SUCCESS_STATES.has(token);
}

function reviewTone(state, verdict, lifecycleStatus = '') {
    return lifecycleFailure(lifecycleStatus)
        ? lifecycleOnlyTone(lifecycleStatus, state)
        : statusTone(state, verdict);
}

export function lifecycleMeta(status) {
    const token = text(status).toLowerCase();
    return lifecycleFailure(token) ? `lifecycle ${token}` : '';
}

export function hasSemanticVerdict(verdict) {
    const token = text(verdict).toLowerCase();
    return Boolean(token)
        && !ACTIVE_STATES.has(token)
        && !LIFECYCLE_TERMINAL_STATES.has(token)
        && !['terminal', 'unavailable', 'absent'].includes(token);
}

function attemptIdentity(attempt, fallback = '') {
    return text(
        attempt?.id || attempt?.job_id || attempt?.wave_id || attempt?.panel_id
        || attempt?.request_fingerprint || fallback,
    );
}

function normalizedExecutions(value, legacyExecution = null) {
    const rows = Array.isArray(value) ? value : [];
    if (rows.length) return rows.filter((item) => item && typeof item === 'object');
    const executed = legacyExecution?.executed;
    return executed && typeof executed === 'object' ? [executed] : [];
}

function executionsFromReviewRecord(record) {
    if (!record || typeof record !== 'object') return [];
    const direct = normalizedExecutions(record.executions, record.execution);
    const actorExecutions = (Array.isArray(record.actors) ? record.actors : [])
        .flatMap((actor) => normalizedExecutions(actor?.executions, actor?.execution));
    return [...direct, ...actorExecutions];
}

function uniformAttemptInitiator(attempts, fallback = '') {
    if (!attempts.length) return text(fallback);
    const initiators = attempts.map((attempt) => text(attempt.initiatorTaskId));
    const known = initiators.filter(Boolean);
    if (!known.length) return text(fallback);
    if (known.length !== initiators.length) return '';
    const unique = new Set(initiators);
    return unique.size === 1 ? initiators[0] : '';
}

function normalizeSkillAttempt(attempt, defaults = {}, ordinal = 0) {
    if (!attempt || typeof attempt !== 'object') return null;
    const skill = text(attempt.skill || defaults.skill);
    const jobId = text(attempt.job_id || attempt.jobId);
    const recordId = text(attempt.record_id || attempt.recordId);
    const id = attemptIdentity(attempt, [
        text(attempt.content_hash), text(attempt.review_round),
        text(attempt.snapshot_attempt), String(ordinal + 1),
    ].join(':'));
    if (!id) return null;
    const lifecycleStatus = text(attempt.lifecycle_status || attempt.job_status);
    const explicitStatus = text(
        attempt.review_status || attempt.review_verdict || attempt.status || defaults.status,
    );
    const lifecycleOnly = attempt.lifecycle_only === true
        || (!hasSemanticVerdict(explicitStatus)
            && LIFECYCLE_TERMINAL_STATES.has(lifecycleStatus.toLowerCase()))
        || (!attempt.review_status && !attempt.review_verdict
            && LIFECYCLE_TERMINAL_STATES.has(explicitStatus.toLowerCase()));
    const rawStatus = lifecycleOnly ? '' : explicitStatus;
    const state = lifecycleOnly
        ? normalizedState(lifecycleStatus || explicitStatus, defaults.state || 'unavailable')
        : normalizedState(rawStatus, defaults.state || 'unavailable');
    return {
        id,
        surface: 'skill',
        state,
        tone: lifecycleOnly
            ? lifecycleOnlyTone(lifecycleStatus || explicitStatus, state)
            : reviewTone(state, rawStatus, lifecycleStatus),
        verdict: rawStatus,
        lifecycleStatus,
        timestamp: text(
            attempt.ts || attempt.timestamp || attempt.finished_at
            || attempt.started_at || attempt.queued_at,
        ),
        ordinal,
        label: [
            attempt.review_round != null ? `round ${attempt.review_round}` : '',
            attempt.snapshot_attempt != null ? `attempt ${attempt.snapshot_attempt}` : '',
        ].filter(Boolean).join(' · ') || `attempt ${ordinal + 1}`,
        summary: text(attempt.summary || attempt.text || attempt.terminal_reason || attempt.error)
            || (lifecycleOnly ? 'Review verdict unavailable.' : ''),
        superseded: Boolean(attempt.superseded),
        replayed: Boolean(attempt.replayed || attempt.replay || attempt.replay_of || attempt.replayed_from_ts),
        revised: Boolean(attempt.snapshot_revised || attempt.revised_snapshot || attempt.revised),
        initiatorTaskId: text(
            attempt.initiator_task_id || attempt.origin_task_id || attempt.task_id,
        ),
        executions: normalizedExecutions(attempt.executions, attempt.execution),
        execution: attempt.execution && typeof attempt.execution === 'object' ? attempt.execution : null,
        lifecycleOnly,
        detailRef: skill && jobId ? { surface: 'skill', skill, jobId, ...(recordId ? { recordId } : {}) } : null,
        detailText: '',
    };
}

function normalizeSkillGroup(group, row = {}, { allowRowTaskIdFallback = true } = {}) {
    if (!group || typeof group !== 'object' || text(group.surface) !== 'skill') return null;
    const id = text(group.id || group.group_id);
    const ownerTaskId = text(group.presentation_owner_task_id);
    if (!id || !ownerTaskId) return null;
    const groupStatus = text(group.status);
    const lifecycleStatus = text(
        group.lifecycle_status || row.lifecycle_status || row.job_status
        || groupStatus || row.status,
    );
    const groupVerdict = text(group.verdict);
    const rowVerdict = text(row.review_status || row.review_verdict || row.status);
    const lifecycleOnly = group.lifecycle_only === true
        || (!hasSemanticVerdict(groupVerdict || rowVerdict)
            && LIFECYCLE_TERMINAL_STATES.has(lifecycleStatus.toLowerCase()))
        || (!group.review_status && !group.review_verdict
            && groupVerdict.toLowerCase() === lifecycleStatus.toLowerCase()
            && LIFECYCLE_TERMINAL_STATES.has(lifecycleStatus.toLowerCase()));
    const defaults = {
        skill: text(group.skill || row.skill),
        status: lifecycleOnly
            ? text(group.review_status || group.review_verdict)
            : text(group.review_status || group.verdict || group.status || row.status),
        state: normalizedState(
            group.state || (lifecycleOnly ? lifecycleStatus : group.status || row.status),
        ),
        initiatorTaskId: text(
            group.initiator_task_id || group.origin_task_id || row.origin_task_id
            || (allowRowTaskIdFallback ? row.task_id : ''),
        ),
    };
    const rawAttempts = Array.isArray(group.attempts) ? group.attempts : [];
    const attempts = rawAttempts
        .map((attempt, index) => normalizeSkillAttempt(attempt, defaults, index))
        .filter(Boolean);
    if (!attempts.length && (row.job_id || group.job_id || group.lifecycle_id)) {
        const fallback = normalizeSkillAttempt({
            ...row,
            ...group,
            job_id: group.job_id || row.job_id || group.lifecycle_id,
            lifecycle_only: lifecycleOnly,
        }, defaults, 0);
        if (fallback) attempts.push(fallback);
    }
    const latest = attempts[attempts.length - 1] || null;
    const state = normalizedState(
        group.state || group.lifecycle_status || group.status || latest?.state,
        latest?.state || 'unavailable',
    );
    const projectedCount = finiteCount(group.projected_attempt_count);
    const count = projectedCount ?? attempts.length;
    return {
        id,
        surface: 'skill',
        label: 'Skill review',
        subject: defaults.skill,
        presentationOwnerTaskId: ownerTaskId,
        subjectTaskId: text(group.subject_task_id),
        initiatorTaskId: uniformAttemptInitiator(attempts, defaults.initiatorTaskId),
        state,
        tone: lifecycleOnly
            ? lifecycleOnlyTone(
                lifecycleStatus || latest?.lifecycleStatus || group.review_status || group.review_verdict,
                state,
            )
            : reviewTone(
                state,
                text(group.verdict || group.review_status || group.status || latest?.verdict),
                lifecycleStatus || latest?.lifecycleStatus,
            ),
        verdict: lifecycleOnly
            ? text(group.review_status || group.review_verdict || latest?.verdict)
            : text(group.verdict || group.review_status || group.status || latest?.verdict),
        summary: text(group.summary || latest?.summary)
            || (lifecycleOnly ? 'Review verdict unavailable.' : ''),
        lifecycleOnly,
        lifecycleStatus,
        activeCount: finiteCount(group.active_count) ?? (state === 'queued' || state === 'running' ? 1 : 0),
        attemptCount: Math.max(count, attempts.length),
        countIsAuthoritative: group.count_is_authoritative === true,
        attempts,
    };
}

export function reviewGroupFromHistoryRow(row) {
    return normalizeSkillGroup(row?.review_group, row);
}

export function classifyReviewLifecycle(row) {
    const lifecycle = row?.lifecycle || row?.progress_meta?.lifecycle;
    if (!lifecycle || typeof lifecycle !== 'object' || text(lifecycle.kind) !== 'review') {
        return { classification: 'not_review', group: null };
    }
    const groupId = text(lifecycle.group_id || lifecycle.review_group_id);
    const ownerTaskId = text(lifecycle.presentation_owner_task_id);
    if (!groupId || !ownerTaskId) {
        return { classification: 'source_incomplete', group: null };
    }
    const status = text(lifecycle.status || lifecycle.phase || 'queued');
    const reviewStatus = text(lifecycle.review_status || lifecycle.review_verdict);
    const lifecycleOnly = !reviewStatus;
    const timestamp = text(
        lifecycle.finished_at || lifecycle.started_at || lifecycle.queued_at
        || lifecycle.ts || row?.ts || row?.timestamp,
    );
    const group = normalizeSkillGroup({
        surface: 'skill',
        id: groupId,
        presentation_owner_task_id: ownerTaskId,
        subject_task_id: lifecycle.subject_task_id,
        initiator_task_id: lifecycle.initiator_task_id || lifecycle.origin_task_id,
        skill: lifecycle.target || lifecycle.skill,
        state: status,
        status: reviewStatus,
        review_status: reviewStatus,
        lifecycle_status: status,
        lifecycle_only: lifecycleOnly,
        active_count: ['pending', 'queued', 'running'].includes(status.toLowerCase()) ? 1 : 0,
        projected_attempt_count: lifecycle.projected_attempt_count ?? (lifecycle.job_id || lifecycle.id ? 1 : 0),
        count_is_authoritative: lifecycle.count_is_authoritative === true,
        attempts: lifecycle.job_id || lifecycle.id ? [{
            ...lifecycle,
            ts: timestamp,
            job_id: lifecycle.job_id || lifecycle.id,
            status: reviewStatus,
            review_status: reviewStatus,
            lifecycle_status: status,
            lifecycle_only: lifecycleOnly,
        }] : [],
    }, row, { allowRowTaskIdFallback: false });
    return {
        classification: group ? 'source_complete' : 'source_incomplete',
        group,
    };
}

export function reviewGroupFromLifecycle(row) {
    return classifyReviewLifecycle(row).group;
}

function lifecyclePointerPayload(row) {
    const direct = row?.lifecycle_pointer;
    if (direct && typeof direct === 'object') return direct;
    const nested = row?.progress_meta?.lifecycle_pointer;
    return nested && typeof nested === 'object' ? nested : null;
}

export function classifyReviewLifecyclePointer(row) {
    const pointer = lifecyclePointerPayload(row);
    if (!pointer || text(pointer.kind) !== 'review') {
        return { classification: 'not_pointer', group: null };
    }
    const groupId = text(pointer.group_id || pointer.review_group_id);
    const ownerTaskId = text(pointer.presentation_owner_task_id);
    if (!groupId || !ownerTaskId) {
        return { classification: 'source_incomplete', group: null };
    }
    const status = text(pointer.status || 'queued');
    const reviewStatus = text(pointer.review_status || pointer.review_verdict);
    const lifecycleOnly = !reviewStatus;
    const initiatorTaskId = text(pointer.initiator_task_id || pointer.origin_task_id);
    const timestamp = text(
        pointer.finished_at || pointer.started_at || pointer.queued_at
        || pointer.ts || row?.ts || row?.timestamp,
    );
    const group = normalizeSkillGroup({
        surface: 'skill',
        id: groupId,
        presentation_owner_task_id: ownerTaskId,
        subject_task_id: pointer.subject_task_id,
        initiator_task_id: initiatorTaskId,
        skill: pointer.target || pointer.skill,
        state: status,
        status: reviewStatus,
        review_status: reviewStatus,
        lifecycle_status: status,
        lifecycle_only: lifecycleOnly,
        active_count: ['pending', 'queued', 'running'].includes(status.toLowerCase()) ? 1 : 0,
        projected_attempt_count: pointer.projected_attempt_count ?? (pointer.job_id ? 1 : 0),
        count_is_authoritative: pointer.count_is_authoritative === true,
        attempts: pointer.job_id ? [{
            ...pointer,
            ts: timestamp,
            initiator_task_id: initiatorTaskId,
            status: reviewStatus,
            review_status: reviewStatus,
            lifecycle_status: status,
            lifecycle_only: lifecycleOnly,
        }] : [],
    }, row, { allowRowTaskIdFallback: false });
    return {
        classification: group ? 'source_complete' : 'source_incomplete',
        group,
    };
}

function planAttempt(wave, index, isCurrent, live) {
    if (!wave || typeof wave !== 'object') return null;
    const fingerprint = text(wave.request_fingerprint);
    const id = attemptIdentity(wave, `${fingerprint || 'wave'}:${wave.cycle_index ?? index + 1}`);
    if (!id) return null;
    const verdict = text(wave.aggregate || 'UNKNOWN');
    const superseded = Boolean(wave.superseded) || !isCurrent;
    // Gate closure and physical reviewer custody are independent: a partial wave can stay in flight after the
    // author selects a different plan — but only while its own task still runs to collect it; afterwards it is a gap.
    const custodyPending = wave.custody_pending === true;
    const state = custodyPending && live ? 'running' : superseded ? 'superseded' : 'terminal';
    const roster = (Array.isArray(wave.actors) ? wave.actors : [])
        .filter((actor) => actor && typeof actor === 'object');
    // A wave whose reviewers may still answer reports how far it got; a settled wave
    // whose reviewers were too few for a verdict reports the same counts in the neutral
    // tone (the stored DEGRADED word is the host's placeholder and never paints).
    const settledNoQuorum = !custodyPending && verdict === 'DEGRADED';
    const [progress, heldTone] = custodyPending || settledNoQuorum ? heldProgress(
        custodyPending && roster.some(actorUnresolved) && !roster.some(actorAwaiting) ? 'unresolved' : (custodyPending && live ? 'in progress' : 'no verdict'), roster, (actor) => actor.ok === true) : ['', ''];
    return {
        id,
        surface: 'plan',
        state,
        progress,
        tone: settledNoQuorum ? 'neutral' : heldTone || statusTone(state, custodyPending ? '' : verdict),
        verdict,
        timestamp: text(wave.reviewed_at || wave.ts || wave.timestamp || wave.closed_at),
        ordinal: index,
        label: wave.cycle_index != null ? `wave ${wave.cycle_index}` : `wave ${index + 1}`,
        summary: text(wave.reason || wave.summary),
        superseded,
        compact: Boolean(wave.compact),
        replayed: Boolean(wave.replayed || wave.cached),
        revised: Boolean(wave.revised),
        initiatorTaskId: '',
        executions: executionsFromReviewRecord(wave),
        execution: null,
        detailRef: null,
        detailText: `${planWaveDetail(wave)}\nCost unavailable`.trim(),
    };
}

function currentPlanAttempt(current, index) {
    const fingerprint = text(current?.fingerprint);
    const status = text(current?.status).toLowerCase();
    if (!fingerprint || !status) return null;
    const state = normalizedState(status);
    const detailText = [
        `Status: ${status}`,
        current.reason ? `Reason: ${text(current.reason)}` : '',
        'Review result unavailable.',
        'Cost unavailable',
    ].filter(Boolean).join('\n');
    return {
        id: fingerprint,
        surface: 'plan',
        state,
        tone: statusTone(state, status),
        verdict: status,
        timestamp: text(current.ts || current.timestamp),
        ordinal: index,
        label: 'current attempt',
        summary: text(current.reason),
        superseded: false,
        compact: false,
        replayed: false,
        revised: false,
        initiatorTaskId: '',
        executions: normalizedExecutions(current.executions, current.execution),
        execution: null,
        detailRef: null,
        detailText,
    };
}

function planFindingLines(wave) {
    const findings = (Array.isArray(wave.findings) ? wave.findings : [])
        .filter((item) => item && typeof item === 'object');
    const dispositions = (Array.isArray(wave.dispositions) ? wave.dispositions : [])
        .filter((item) => item && typeof item === 'object');
    // EVERY disposition row renders: the backend refuses duplicate rows for
    // one finding as contradictory intent and keeps the finding open, so
    // showing only the first would present a refused decision as operative.
    const dispositionsByFinding = new Map();
    for (const disposition of dispositions) {
        const key = text(disposition.finding_id);
        const bucket = dispositionsByFinding.get(key) || [];
        bucket.push(disposition);
        dispositionsByFinding.set(key, bucket);
    }
    const dispositionLine = (disposition, prefix) => (
        `${prefix}${text(disposition.decision) || 'disposition'}${text(disposition.rationale) ? ` — ${text(disposition.rationale)}` : ''}`
    );
    const lines = [];
    for (const finding of findings) {
        const head = [
            `[${text(finding.class) || 'finding'}] ${text(finding.summary) || '(no summary)'}`,
            text(finding.breaks) ? `breaks ${text(finding.breaks)}` : '',
            text(finding.locator) ? `at ${text(finding.locator)}` : '',
        ].filter(Boolean).join(' — ');
        const source = [text(finding.slot), text(finding.model)].filter(Boolean).join(' · ');
        lines.push(source ? `${head} — ${source}` : head);
        if (text(finding.recommendation)) lines.push(`  fix: ${text(finding.recommendation)}`);
        const findingId = text(finding.finding_id);
        if (findingId) {
            for (const disposition of dispositionsByFinding.get(findingId) || []) {
                lines.push(dispositionLine(disposition, '  agent: '));
            }
            dispositionsByFinding.delete(findingId);
        }
    }
    if (dispositionsByFinding.size) {
        lines.push('General dispositions:');
        for (const [findingId, bucket] of dispositionsByFinding) {
            for (const disposition of bucket) {
                lines.push(dispositionLine(disposition, `  ${findingId || '(no finding id)'}: `));
            }
        }
    }
    return lines;
}

function planActorAvailabilityLines(wave) {
    // The bug report's own bar: a result that was never received must say so
    // explicitly instead of contributing silently-zero findings. A row names the model and
    // quotes the engine's sentence; failure code and slot id stay in the task detail and Logs.
    const lines = [];
    for (const actor of (Array.isArray(wave.actors) ? wave.actors : [])) {
        if (!actor || typeof actor !== 'object' || actor.ok !== false) continue;
        const model = text(actor.model) || 'reviewer';
        const cause = text(actor.reported_cause).split(/\s+/).join(' ');
        const carried = finiteCount(actor.carried_findings) ? '; its earlier finding is still listed' : '';
        if (actorAwaiting(actor)) lines.push(`${model} · awaiting${sinceLocalTime(actor.awaiting_since)}`);
        else if (actorUnresolved(actor)) lines.push(`${model} · no answer${cause ? ` — "${cause}"` : ''}${sinceLocalTime(actor.awaiting_since)}`);
        else if (text(actor.operation_state) === 'not_dispatched') lines.push(`${model} · not sent${carried}`);
        else lines.push(`${model} · unavailable${cause ? ` — "${cause}"` : ''}${carried ? ` · did not answer${carried}` : ''}`);
    }
    return lines;
}

function planWaveDetail(wave) {
    const lines = [
        wave.custody_pending === true
            ? 'Verdict: none (wave held open)'
            : (text(wave.aggregate) === 'DEGRADED' ? 'Verdict: none — fewer reviewers answered than needed' : (wave.aggregate ? `Verdict: ${wave.aggregate}` : '')),
        wave.closed != null ? `Closed: ${wave.closed ? 'yes' : 'no'}` : '',
        wave.paid != null ? `Reviewer panel dispatched: ${wave.paid ? 'yes' : 'no'}` : '',
        wave.quorum_unreachable ? 'Quorum unavailable' : '',
        wave.cycles_exhausted ? 'Review cycles exhausted' : '',
        wave.reason ? `Reason: ${text(wave.reason)}` : '',
    ];
    const counts = wave.counts && typeof wave.counts === 'object' ? wave.counts : {};
    // A panel ordered weaker than the owner's setting says so seat by seat (typed fact; the
    // verdict token is never recoloured), on compact waves too since compaction keeps the fact.
    const weaker = wave.ordered_weaker && typeof wave.ordered_weaker === 'object'
        ? Object.entries(wave.ordered_weaker).filter(([, row]) => row && typeof row === 'object') : [];
    if (weaker.length) {
        lines.push(`Reviewers ordered weaker than your setting: ${weaker
            .map(([sid, row]) => `${sid} ${text(row.effort) || '?'} (setting ${text(row.owner_effort) || '?'})`).join(', ')}`);
    }
    if (wave.compact) {
        // A compacted wave keeps counts while its finding bodies moved to the
        // immutable wave artifact; name that remainder instead of rendering a
        // bound that looks like the whole record.
        const recorded = [
            finiteCount(counts.findings) != null ? `${finiteCount(counts.findings)} findings` : '',
            finiteCount(counts.blocking) != null ? `${finiteCount(counts.blocking)} blocking` : '',
            finiteCount(counts.dispositions) != null ? `${finiteCount(counts.dispositions)} dispositions` : '',
        ].filter(Boolean).join(' · ');
        if (recorded) lines.push(`Recorded: ${recorded}`);
        const artifact = wave.wave_artifact && typeof wave.wave_artifact === 'object' ? wave.wave_artifact : {};
        const sha = text(artifact.sha256);
        lines.push(`Finding bodies compacted${sha ? ` · artifact sha256=${sha.slice(0, 12)}…` : ''}${finiteCount(artifact.bytes) != null ? ` (${finiteCount(artifact.bytes)} bytes)` : ''}`);
        return lines.filter(Boolean).join('\n');
    }
    const countParts = ['blocking', 'note', 'need_evidence']
        .filter((key) => finiteCount(counts[key]) != null)
        .map((key) => `${finiteCount(counts[key])} ${key}`);
    if (countParts.length) lines.push(`Findings: ${countParts.join(' · ')}`);
    lines.push(...planFindingLines(wave));
    lines.push(...planActorAvailabilityLines(wave));
    const findingsShown = (Array.isArray(wave.findings) ? wave.findings : []).length;
    if (wave.findings_paged && finiteCount(wave.findings_total) != null) {
        lines.push(`Showing ${findingsShown} of ${finiteCount(wave.findings_total)} findings (per-slot page cap)`);
    }
    if (wave.findings_texts_truncated) lines.push('Some finding texts were truncated at capture.');
    if (wave.spec_body_truncated) lines.push('Spec body was truncated at capture.');
    const authorLine = authorDispositionText(wave.author_disposition);
    if (authorLine) lines.push(authorLine);
    return lines.filter(Boolean).join('\n');
}

function legacyPlanReviewGroup(owner, stateRecord) {
    const projection = stateRecord?.legacy_v1_projection;
    if (!projection || typeof projection !== 'object') return null;
    const status = text(projection.status).toLowerCase();
    if (!status || status === 'absent') return null;
    const outcome = text(projection.outcome);
    const fingerprint = text(projection.fingerprint);
    const attempts = fingerprint && outcome ? [{
        id: fingerprint,
        surface: 'plan',
        state: 'terminal',
        tone: statusTone('terminal', outcome),
        verdict: outcome,
        timestamp: '',
        ordinal: 0,
        label: 'legacy wave',
        summary: text(projection.reason),
        superseded: false,
        compact: false,
        replayed: false,
        revised: false,
        initiatorTaskId: '',
        executions: [],
        execution: null,
        detailRef: null,
        detailText: [
            `Status: ${status}`,
            `Verdict: ${outcome}`,
            projection.closed != null ? `Closed: ${projection.closed ? 'yes' : 'no'}` : '',
            projection.reason ? `Reason: ${text(projection.reason)}` : '',
            'Cost unavailable',
        ].filter(Boolean).join('\n'),
    }] : [];
    const pending = status === 'pending';
    const controlVerdict = status === 'rail_degraded' ? status : outcome || status;
    const state = pending ? 'queued'
        : (status === 'open' && !outcome ? 'unavailable' : 'terminal');
    return {
        id: `plan:${owner}`,
        surface: 'plan',
        label: 'Plan review',
        subject: '',
        presentationOwnerTaskId: owner,
        subjectTaskId: owner,
        initiatorTaskId: owner,
        state,
        tone: statusTone(state, outcome || (status === 'open' ? 'unavailable' : controlVerdict)),
        verdict: controlVerdict,
        summary: text(projection.reason),
        activeCount: pending ? 1 : 0,
        attemptCount: attempts.length,
        countIsAuthoritative: false,
        attempts,
    };
}

export function planReviewGroupFromTaskDetail(detail, ownerTaskId = '') {
    const owner = text(ownerTaskId || detail?.task_id);
    if (owner && planRecordEngaged(detail?.plan_record)) return planRecordGroup(owner, detail.plan_record, detail);
    const stateRecord = detail?.plan_review_state;
    if (!owner || !stateRecord || typeof stateRecord !== 'object') return null;
    if (stateRecord.schema_version === 1) return legacyPlanReviewGroup(owner, stateRecord);
    const current = stateRecord.current_attempt && typeof stateRecord.current_attempt === 'object'
        ? stateRecord.current_attempt : {};
    const recordedWaves = Array.isArray(stateRecord.waves) ? stateRecord.waves : [];
    if (!recordedWaves.length && !text(current.status)) return null;
    const currentFingerprint = text(current.fingerprint);
    const authorSubject = current.author_subject;
    const author = authorDispositionText(authorSubject?.author_disposition);
    const reviewFingerprint = author ? text(authorSubject.review_fingerprint) : currentFingerprint;
    // The critic reviewed the EARLIER plan when the author selected a revised one: the group
    // is labelled as that plan's review and the selected plan is named unreviewed — the same
    // `historical_critic` fact the gate projection carries, never a synthesized verdict.
    const historicalCritic = Boolean(author) && Boolean(reviewFingerprint) && reviewFingerprint !== currentFingerprint;
    const authorDecisionText = author ? [author,
        `Critic plan: ${reviewFingerprint}`,
        historicalCritic ? `The verdict shown is the earlier plan's review; the selected plan ${currentFingerprint} has no verdict of its own` : '',
        authorSubject.source_ref?.path ? `Current plan source: ${text(authorSubject.source_ref.root)}:${text(authorSubject.source_ref.path)}` : '',
        authorSubject.source_ref?.sha256 ? `Source sha256=${text(authorSubject.source_ref.sha256)}` : '',
    ].filter(Boolean).join('\n') : '';
    const typedCurrentStatus = text(current.status).toLowerCase();
    // C-09: a compact row proves history, not reusable authority. While the
    // same envelope is being reviewed again, replace that stale projection
    // with current_attempt; the eventual full wave reuses the same identity.
    const waves = recordedWaves.filter((wave) => !(
        wave?.compact && !author
        && currentFingerprint
        && typedCurrentStatus
        && text(wave.request_fingerprint) === currentFingerprint
        && (typedCurrentStatus === 'open' || wave.closed === true)
    ));
    const currentWaveIndex = reviewFingerprint
        ? waves.findIndex((wave) => text(wave?.request_fingerprint) === reviewFingerprint)
        : (waves.length ? waves.length - 1 : -1);
    const attempts = waves
        .map((wave, index) => planAttempt(
            wave,
            index,
            index === currentWaveIndex, text(detail?.status).toLowerCase() === 'running',
        ))
        .filter(Boolean);
    let currentAttempt = currentWaveIndex >= 0
        ? attempts.find((attempt) => attempt.id === attemptIdentity(
            waves[currentWaveIndex],
            `${currentFingerprint || 'wave'}:${waves[currentWaveIndex]?.cycle_index ?? currentWaveIndex + 1}`,
        )) || null
        : null;
    const currentStatus = typedCurrentStatus || (currentAttempt ? 'closed' : 'open');
    let state = 'terminal';
    let activeCount = 0;
    if (!currentAttempt && !author) {
        const unmatchedAttempt = currentPlanAttempt(current, attempts.length);
        if (unmatchedAttempt) {
            attempts.push(unmatchedAttempt);
            currentAttempt = unmatchedAttempt;
            state = unmatchedAttempt.state;
            activeCount = state === 'queued' || state === 'running' ? 1 : 0;
        } else if (currentStatus) {
            state = normalizedState(currentStatus);
        }
    }
    let currentVerdict = text(currentAttempt?.verdict || (author ? 'unavailable' : currentStatus));
    // The typed Plan gate releases an open wave when the task-wide deadline
    // rail degrades. Preserve the wave as semantic review evidence, but mirror
    // the backend precedence in the group header. A closed wave remains final.
    const terminalControl = typedCurrentStatus === 'rail_degraded'
        ? typedCurrentStatus
        : (waves[currentWaveIndex]?.cycles_exhausted === true ? 'cycles_exhausted' : '');
    if (currentAttempt && terminalControl && waves[currentWaveIndex]?.closed !== true) {
        state = normalizedState(terminalControl);
        activeCount = 0;
        currentVerdict = terminalControl;
    }
    const activeAttempts = attempts.filter((attempt) => ['queued', 'running'].includes(attempt.state));
    if (activeAttempts.length) {
        state = activeAttempts.some((attempt) => attempt.state === 'running') ? 'running' : 'queued';
        activeCount = activeAttempts.length;
    }
    // The group header speaks for its CURRENT wave: a progress phrase while that wave
    // has unanswered slots, the recorded verdict once every slot is collected.
    const progress = text(currentAttempt?.progress);
    return {
        id: `plan:${owner}`,
        surface: 'plan',
        label: historicalCritic ? 'Plan review · earlier plan' : 'Plan review',
        historicalCritic,
        subject: '',
        presentationOwnerTaskId: owner,
        subjectTaskId: owner,
        initiatorTaskId: owner,
        state,
        progress,
        tone: progress ? currentAttempt.tone : statusTone(state, currentVerdict),
        verdict: currentVerdict,
        summary: text(current.reason || currentAttempt?.summary),
        authorDecisionText,
        activeCount,
        attemptCount: attempts.length + (finiteCount(stateRecord.waves_omitted) || 0),
        countIsAuthoritative: finiteCount(stateRecord.waves_omitted) === 0,
        attempts,
    };
}

function authorDispositionText(author, label = '') {
    if (!author || typeof author !== 'object' || (!text(author.disposition) && !text(author.action))) return '';
    const actionLabel = label || `Author ${author.action === 'stop' ? 'stop' : 'finish'}`;
    return [
        `${actionLabel}${text(author.disposition) ? `: ${text(author.disposition)}` : ''}`,
        text(author.reviewer_signal) ? `reviewer signal=${text(author.reviewer_signal)}` : '',
        text(author.rationale),
        text(author.subject_hash) ? `subject_hash=${text(author.subject_hash)}` : '',
        text(author.source) ? `source=${text(author.source)}` : '',
        ...authorDoctrineLines(author),
    ].filter(Boolean).join(' · ');
}

/** Read-only: a task-acceptance record stored before the result review replaced that machine. */
export function taskAcceptanceGroupFromTaskDetail(detail, ownerTaskId = '') {
    const owner = text(ownerTaskId || detail?.task_id);
    const projection = detail?.review_projection;
    const panels = (Array.isArray(projection?.panels) ? projection.panels : [])
        .filter((panel) => text(panel?.surface) === 'task_acceptance');
    const incident = acceptanceIncidentFromTaskDetail(detail);
    // A local preparation failure produces NO panel: the group has to exist on
    // the incident alone, or the owner sees nothing at all.
    if (!owner || (!panels.length && !incident)) return null;
    const acceptanceDecision = detail?.outcome_axes?.review?.acceptance_decision
        || detail?.review_status?.acceptance_decision;
    const decisionAuthor = acceptanceDecision?.author_disposition;
    // A task's final author subject can differ from every historical panel:
    // Advisory may finish edited h2 after the critic reviewed h1. Keep this
    // decision at task level; it never belongs to a panel merely by recency.
    const authorDecisionText = authorDispositionText(decisionAuthor)
        || authorDispositionText({
            disposition: acceptanceDecision?.agent_disposition,
            rationale: acceptanceDecision?.agent_rationale
                || acceptanceDecision?.author_rationale || acceptanceDecision?.rationale,
            source: acceptanceDecision?.source,
        }, 'Author stance (unbound)');
    const attempts = panels.map((panel, index) => {
        const verdict = text(panel?.aggregate_signal || 'UNKNOWN');
        const roster = (Array.isArray(panel?.actors) ? panel.actors : []).filter((actor) => actor && typeof actor === 'object');
        const awaited = roster.some(actorAwaiting);
        // An awaited panel is activity only while its own task still runs to collect it; afterwards an unanswered
        // slot is a recorded gap. A settled panel without a verdict says so in words; DEGRADED stays in its detail.
        const live = awaited && text(detail?.status).toLowerCase() === 'running' && !panel?.superseded && index === panels.length - 1;
        const held = { FAIL: 'FAIL', PASS: live ? 'PASS so far' : 'PASS' }[verdict.toUpperCase()] || (live ? 'in progress' : 'no verdict');
        const [progress, heldTone] = awaited || verdict.toUpperCase() === 'DEGRADED' ? heldProgress(held, roster, (actor) => text(actor.transport_status) === 'success') : ['', ''];
        // The host composed this sentence; the card prints it verbatim and
        // leads the detail with it, because the renderer shows detailText over
        // summary. The panel's own verdict and identity are untouched.
        const late = panel?.late_settlement && typeof panel.late_settlement === 'object' ? panel.late_settlement : null;
        const lateNote = late ? text(late.note) : '';
        return {
            id: [attemptIdentity(panel, `panel:${index + 1}`),
                panel.task_attempt == null ? '' : `task-attempt:${panel.task_attempt}`,
                panel.panel_index == null ? '' : `panel-index:${panel.panel_index}`].filter(Boolean).join(':'),
            surface: 'task_acceptance',
            state: live ? 'running' : (panel?.superseded ? 'superseded' : 'terminal'),
            progress,
            tone: !awaited || held === 'FAIL' ? statusTone('terminal', verdict)
                : (heldTone || (live ? 'working' : (held === 'PASS' ? statusTone('terminal', verdict) : 'neutral'))),
            verdict,
            timestamp: text(panel?.ts || panel?.timestamp),
            ordinal: index,
            label: `panel ${text(panel?.panel_id || index + 1)}${late ? ' · settled after the task ended' : ''}`,
            summary: lateNote || text(panel?.reason),
            superseded: Boolean(panel?.superseded),
            replayed: false,
            revised: false,
            initiatorTaskId: owner,
            executions: executionsFromReviewRecord(panel),
            execution: null,
            detailRef: { surface: 'task_acceptance', url: panel.applied_source_status === 'available'
                ? taskSourceDownloadUrl(owner, panel.applied_source_ref) : '' },
            detailText: [lateNote, formatReviewProjection({ panels: [panel] }),
                authorDispositionText(panel.author_disposition), 'Cost unavailable'].filter(Boolean).join('\n'),
        };
    });
    return acceptanceGroupWithIncident({ owner, attempts, incident, authorDecisionText, statusTone });
}

/** The task's own review-ledger records (panels carrying `record_id`), one attempt per record, oldest first. */
export function reviewRecordGroupFromTaskDetail(detail, ownerTaskId = '') {
    const owner = text(ownerTaskId || detail?.task_id);
    const projection = detail?.review_projection;
    const panels = (Array.isArray(projection?.panels) ? projection.panels : []).filter((panel) => text(panel?.record_id));
    const gaps = (Array.isArray(projection?.gaps) ? projection.gaps : []).filter((gap) => gap && typeof gap === 'object');
    // Only a record or a gap makes a group: a count of older records alone (`review_records_omitted`,
    // unknown once the index rotated) is no review of this task.
    if (!owner || (!panels.length && !gaps.length)) return null;
    const live = text(detail?.status).toLowerCase() === 'running';
    const attempts = panels.map((panel, index) => {
        const verdict = text(panel.aggregate_signal) || 'UNKNOWN';
        // A pending record whose counted seats are still owed reports how far it got, like a stored panel:
        // in progress while its task runs, no verdict once it ended; NOT_PERFORMED stays in Record facts.
        const counted = (Array.isArray(panel.actors) ? panel.actors : []).filter((actor) => actor && typeof actor === 'object' && !seatAdviser(actor));
        const owed = text(panel.record_state) === 'pending' && counted.some((actor) => seatOwed(actor, panel));
        const [progress, heldTone] = owed
            ? heldProgress(live ? 'in progress' : 'no verdict', counted, actorAnswered, (actor) => seatOwed(actor, panel)) : ['', ''];
        return {
            id: `record:${text(panel.record_id)}`, panel, surface: 'review_record', state: owed && live ? 'running' : 'terminal', progress,
            // NOT_PERFORMED, QUORUM_FAILED and NOT_DISPATCHED reached no verdict: a warning, never a quiet neutral.
            tone: owed ? (heldTone || (live ? 'working' : 'neutral')) : ({ PASS: 'done', FAIL: 'error' }[verdict] || 'warn'),
            verdict, timestamp: text(panel.ts), ordinal: index, live,
            label: `${text(panel.surface) || 'review'} record ${text(panel.record_id)}`, summary: text(panel.reason),
            superseded: false, replayed: false, revised: false, initiatorTaskId: owner,
            executions: executionsFromReviewRecord(panel), execution: null, detailRef: null,
            detailText: [formatReviewProjection({ panels: [panel] }), text(panel.source_ref?.path) ? `Full record: ${text(panel.source_ref.path)}` : ''].filter(Boolean).join('\n'),
        };
    });
    for (const gap of gaps) attempts.push({
        id: `gap:${text(gap.record_id) || text(gap.review_gap)}`, surface: 'review_record', state: 'terminal', tone: 'warn', verdict: 'could not be read',
        label: 'Review record unavailable', summary: 'This saved review could not be read. The readable reviews beside it are unaffected.',
        gap: true, gapReason: text(gap.review_gap), detailText: formatReviewProjection({ gaps: [gap] }), executions: [],
    });
    const latest = attempts.at(-1) || { tone: 'neutral', verdict: '', summary: '' };
    return {
        // Earlier records exist ('1+' or a count) or may exist in a rotated index ('unknown'); 0 says none.
        historyTask: owner, historyMore: ![0, undefined, null, ''].includes(projection.review_records_omitted),
        historyCertain: projection.review_records_omitted !== 'unknown',
        id: `review_record:${owner}`, surface: 'review_record', label: 'Review records', subject: '',
        presentationOwnerTaskId: owner, subjectTaskId: owner, initiatorTaskId: owner,
        state: 'terminal', progress: '', tone: latest.tone, verdict: latest.verdict, summary: latest.summary,
        warning: '', authorDecisionText: '', activeCount: 0, attemptCount: attempts.length,
        // Older records stay in the ledger: the count is exact only when the projection says none exist.
        countIsAuthoritative: projection.review_records_omitted === 0, attempts,
        // The task's own settled adoptions, joined with each record's landed commit; a source without the
        // field leaves what an earlier read said (absent is unknown, never "none adopted").
        ...(Array.isArray(detail?.body_adoptions) ? { adoptions: detail.body_adoptions } : {}),
    };
}

// One row per adoption at its highest revision across two reads: a history row read before a later outcome settled
// cannot take that adoption back, nor drop one the other read named.
function mergeAdoptions(prior, incoming) {
    const revision = (row) => (Number.isSafeInteger(row?.revision) ? row.revision : 0);
    const rows = new Map();
    for (const row of [...prior, ...incoming]) {
        const id = text(row?.adoption_id) || JSON.stringify(row);
        if (!(revision(rows.get(id)) > revision(row))) rows.set(id, row);
    }
    return [...rows.values()];
}

export function reviewGroupsFromTaskDetail(detail, ownerTaskId = '') {
    return [
        planReviewGroupFromTaskDetail(detail, ownerTaskId),
        taskAcceptanceGroupFromTaskDetail(detail, ownerTaskId),
        reviewRecordGroupFromTaskDetail(detail, ownerTaskId),
    ].filter(Boolean);
}

export function mergeReviewGroup(store, incoming) {
    if (!(store instanceof Map) || !incoming?.id || !incoming.presentationOwnerTaskId) return null;
    const prior = store.get(incoming.id);
    if (!prior) {
        const created = { ...incoming, attempts: [...incoming.attempts] };
        store.set(created.id, created);
        return created;
    }
    const incomingIds = new Set(incoming.attempts.map((attempt) => attempt.id));
    const priorById = new Map(prior.attempts.map((attempt) => [attempt.id, attempt]));
    const mergedById = new Map(priorById);
    let hasStaleActiveAttempt = false;
    let introducedActiveAttempt = false;
    for (const attempt of incoming.attempts) {
        const previous = priorById.get(attempt.id);
        const previousTerminal = previous?.state === 'terminal' || previous?.state === 'superseded';
        const incomingActive = attempt.state === 'queued' || attempt.state === 'running';
        const replacesCompactedPlan = (
            incoming.surface === 'plan'
            && previous?.compact === true
            && attempt.compact === false
            && incomingActive
        );
        // Attempt ids are immutable domain references. A delayed lifecycle row
        // for the same attempt may arrive after its terminal history row, but
        // it cannot make that physical attempt non-terminal again.
        let mergedAttempt = previousTerminal && incomingActive && !replacesCompactedPlan
            ? { ...attempt, ...previous }
            : { ...(previous || {}), ...attempt };
        // A lifecycle terminal frame can arrive after the domain history row
        // for the same attempt.  It contributes timing/execution facts, but
        // its `succeeded`/`completed` word is not a review verdict and must
        // not erase an already-proved semantic result.
        if (
            attempt.lifecycleOnly
            && previousTerminal
            && hasSemanticVerdict(previous?.verdict)
        ) {
            mergedAttempt = {
                ...mergedAttempt,
                state: previous.state,
                tone: lifecycleFailure(attempt.lifecycleStatus)
                    ? lifecycleOnlyTone(attempt.lifecycleStatus, previous.state)
                    : previous.tone,
                verdict: previous.verdict,
                summary: previous.summary || attempt.summary,
                lifecycleOnly: false,
            };
        }
        // A lifecycle frame for an already-semantic attempt carries a transport
        // timestamp, not a new attempt timestamp. Keep the domain row's time as
        // the ordering key so a late frame cannot move that attempt past newer
        // semantic history.
        if (attempt.lifecycleOnly && previous && !previous.lifecycleOnly) {
            mergedAttempt.timestamp = previous.timestamp || mergedAttempt.timestamp;
        }
        mergedById.set(attempt.id, mergedAttempt);
        if (incomingActive && previousTerminal && !replacesCompactedPlan) {
            hasStaleActiveAttempt = true;
        }
        if (incomingActive && (!previous || replacesCompactedPlan)) introducedActiveAttempt = true;
    }
    if (incoming.surface === 'plan') {
        // Plan task detail owns one canonical current fingerprint. An omitted
        // prior unmatched attempt was replaced, not merely absent from a
        // bounded history window; retain its reason but retire its liveness.
        for (const [id, attempt] of mergedById) {
            if (
                !incomingIds.has(id)
                && (attempt.state === 'queued' || attempt.state === 'running')
            ) {
                mergedById.set(id, {
                    ...attempt, state: 'superseded', tone: 'neutral', superseded: true, progress: '',
                });
            }
        }
    }
    const priorIds = prior.attempts.map((attempt) => attempt.id);
    const priorOnly = priorIds.filter((id) => !incomingIds.has(id));
    // A projection that contains every known attempt owns their order (for
    // example, terminal Skill history arriving after one live row). A bounded
    // projection that omits a known attempt keeps established order; a new
    // timestamped identity is inserted before a later known timestamp.
    let order;
    if (priorOnly.length === 0) {
        order = incoming.attempts.map((attempt) => attempt.id);
    } else {
        order = [...priorIds];
        for (const attempt of incoming.attempts) {
            if (priorById.has(attempt.id) || order.includes(attempt.id)) continue;
            const timestamp = text(attempt.timestamp);
            const insertBefore = timestamp
                ? order.findIndex((id) => {
                    const existing = text(mergedById.get(id)?.timestamp);
                    return existing && existing > timestamp;
                })
                : -1;
            if (insertBefore >= 0) order.splice(insertBefore, 0, attempt.id);
            else order.push(attempt.id);
        }
    }
    const staleActiveRegression = (
        (prior.state === 'terminal' || prior.state === 'superseded')
        && (incoming.state === 'queued' || incoming.state === 'running')
        && hasStaleActiveAttempt
        && !introducedActiveAttempt
    );
    const merged = {
        ...prior,
        ...incoming,
        attempts: order.map((id) => mergedById.get(id)).filter(Boolean),
        attemptCount: Math.max(prior.attemptCount || 0, incoming.attemptCount || 0, mergedById.size),
    };
    if (Array.isArray(prior.adoptions) && Array.isArray(incoming.adoptions)) merged.adoptions = mergeAdoptions(prior.adoptions, incoming.adoptions);
    if (staleActiveRegression) {
        merged.state = prior.state;
        merged.verdict = prior.verdict;
        merged.summary = prior.summary;
        merged.activeCount = prior.activeCount;
        merged.lifecycleStatus = prior.lifecycleStatus;
        merged.tone = prior.progress ? prior.tone : reviewTone(merged.state, merged.verdict, merged.lifecycleStatus);
    }
    const activeAttempts = merged.attempts.filter(
        (attempt) => attempt.state === 'queued' || attempt.state === 'running',
    );
    // A bounded/history projection can omit the newest live attempt. Preserve
    // that attempt's liveness until a source explicitly updates the same
    // immutable attempt id to a terminal state.
    if (activeAttempts.length) {
        const active = activeAttempts.at(-1);
        merged.state = activeAttempts.some((attempt) => attempt.state === 'running') ? 'running' : 'queued';
        merged.tone = (active.progress && active.tone) || 'working';
        merged.verdict = active.verdict || (merged.lifecycleOnly ? '' : merged.state);
        merged.summary = active.summary || merged.summary;
        merged.activeCount = activeAttempts.length;
    }
    if (incoming.lifecycleOnly && incoming.attempts.length && !activeAttempts.length) {
        // Project a lifecycle-only group from the latest merged attempt.  A
        // late frame for an existing attempt inherits that attempt's typed
        // verdict; a genuinely new attempt stays neutral instead of borrowing
        // the previous attempt's group-level PASS.
        const latestAttempt = merged.attempts.at(-1);
        if (latestAttempt) {
            merged.state = latestAttempt.state;
            merged.tone = reviewTone(
                latestAttempt.state,
                latestAttempt.verdict,
                latestAttempt.lifecycleStatus,
            );
            merged.verdict = latestAttempt.verdict || '';
            merged.summary = latestAttempt.summary || merged.summary;
            merged.lifecycleOnly = Boolean(latestAttempt.lifecycleOnly);
        }
    }
    const priorLatestAttempt = prior.attempts.at(-1);
    const mergedLatestAttempt = merged.attempts.at(-1);
    const newAttemptHasProvenablyNewerTimestamp = (
        priorLatestAttempt?.timestamp
        && mergedLatestAttempt
        && !priorById.has(mergedLatestAttempt.id)
        && !mergedLatestAttempt.lifecycleOnly
        && hasSemanticVerdict(mergedLatestAttempt.verdict)
        && text(mergedLatestAttempt.timestamp) > text(priorLatestAttempt.timestamp)
    );
    const omittedLatestLifecycle = (
        !activeAttempts.length
        && priorLatestAttempt?.lifecycleOnly
        && !incomingIds.has(priorLatestAttempt.id)
        && !newAttemptHasProvenablyNewerTimestamp
    );
    const delayedSemanticProjection = (
        incoming.surface === 'skill'
        && priorOnly.length > 0
        && !activeAttempts.length
        && priorLatestAttempt
        && !priorLatestAttempt.lifecycleOnly
        && hasSemanticVerdict(priorLatestAttempt.verdict)
        && !incomingIds.has(priorLatestAttempt.id)
        && mergedLatestAttempt
        && !mergedLatestAttempt.lifecycleOnly
        && hasSemanticVerdict(mergedLatestAttempt.verdict)
        && (
            !text(priorLatestAttempt.timestamp)
            || !text(mergedLatestAttempt.timestamp)
            || text(mergedLatestAttempt.timestamp) <= text(priorLatestAttempt.timestamp)
        )
    );
    if (omittedLatestLifecycle || delayedSemanticProjection) {
        // A stale refresh can omit the newest terminal lifecycle row or carry
        // an older semantic row first seen after the newer one arrived live.
        // Both keep the header on the newest proved attempt; an unresolved
        // lifecycle still cannot borrow the older semantic verdict.
        merged.state = priorLatestAttempt.state;
        merged.tone = reviewTone(
            priorLatestAttempt.state,
            priorLatestAttempt.verdict,
            priorLatestAttempt.lifecycleStatus,
        );
        merged.verdict = priorLatestAttempt.verdict || '';
        merged.summary = priorLatestAttempt.summary || merged.summary;
        merged.lifecycleOnly = Boolean(priorLatestAttempt.lifecycleOnly);
        merged.lifecycleStatus = priorLatestAttempt.lifecycleStatus || merged.lifecycleStatus;
    }
    merged.initiatorTaskId = uniformAttemptInitiator(
        merged.attempts,
        incoming.initiatorTaskId || prior.initiatorTaskId,
    );
    // The progress phrase belongs to the attempt the header speaks for.
    if ('progress' in merged) merged.progress = text((activeAttempts.at(-1) || merged.attempts.at(-1))?.progress);
    if (JSON.stringify(prior) === JSON.stringify(merged)) return prior;
    store.set(merged.id, merged);
    return merged;
}

export function orderedReviewGroups(store) {
    const groups = store instanceof Map ? [...store.values()] : Array.isArray(store) ? [...store] : [];
    return groups.sort((a, b) => (SURFACE_ORDER.get(a.surface) ?? 99) - (SURFACE_ORDER.get(b.surface) ?? 99));
}

export function reviewGroupCounts(store) {
    const groups = orderedReviewGroups(store);
    return {
        groupCount: groups.length,
        activeCount: groups.reduce((total, group) => total + (finiteCount(group.activeCount) || 0), 0),
    };
}

function reviewRevision(value) {
    const revision = text(value).toLowerCase();
    return /^[0-9a-f]{64}$/.test(revision) ? revision : null;
}

// One queue for every card's review detail on the page (issue #1771): a room with
// many review owners otherwise opened one GET per owner at once and starved the
// browser's few connections per host (WebKit: 6), videos included. Newest request
// first: a room opens at its newest cards, and an expansion is the latest ask. A
// read keeps its slot until it settles; one whose response never comes is aborted
// at the slot's bound and fails like any lost read (its card offers Retry), so a dead
// connection neither stalls every card's details nor stays open beside the next
// reads. A closed owner's queued reads never start.
export const REVIEW_DETAIL_READS = 2;
export const REVIEW_DETAIL_SLOT_MS = 30000;
const detailQueue = [];
let detailReads = 0;
function drainDetailReads() {
    while (detailReads < REVIEW_DETAIL_READS && detailQueue.length) {
        const job = detailQueue.pop();
        detailReads += 1;
        const abort = new AbortController();
        const lost = new Promise((_resolve, reject) => abort.signal.addEventListener('abort',
            () => reject(new Error('review detail read lost its connection')), { once: true }));
        const timer = setTimeout(() => abort.abort(), REVIEW_DETAIL_SLOT_MS);
        Promise.race([new Promise(resolve => resolve(job.read(abort.signal))), lost]).then(job.resolve, job.reject)
            .finally(() => { clearTimeout(timer); detailReads -= 1; drainDetailReads(); });
    }
}
const queueDetailRead = (owner, read) => new Promise((resolve, reject) => {
    detailQueue.push({ owner, read, resolve, reject });
    drainDetailReads();
});

// Revisions are opaque SHA-256 tokens; one distinct token may trail a live GET.
export function createReviewHydrator({ fetchDetail, applyDetail, onState = () => {}, onSettled = () => {} } = {}) {
    const states = new Map();
    const start = (taskId, state, revision, onDomWrite) => {
        clearTimeout(state.migrationTimer); state.migrationTimer = null;
        const generation = ++state.generation;
        state.inFlightRevision = revision;
        const write = typeof onDomWrite === 'function' ? onDomWrite : (mutate) => mutate();
        // First load/retry announces; routine refresh over applied content stays quiet.
        const notify = (status) => {
            state.lastStatus = status;
            return write(() => onState(taskId, status));
        };
        const request = Promise.resolve()
            .then(() => {
                if (!state.everApplied || state.lastStatus === 'error') notify('loading');
                return queueDetailRead(states, (signal) => fetchDetail(taskId, { signal }));
            })
            .then((detail) => {
                // The strict seam rejects failures; null means genuinely absent (404).
                if (detail === null || detail === undefined) return false;
                return write(() => applyDetail(taskId, detail));
            })
            .then((applied) => {
                if (applied !== false && revision !== null) state.appliedRevision = revision;
                state.everApplied = true;
                notify('idle');
                return applied;
            })
            .catch(error => {
                if (states.get(taskId) !== state) return false;
                if (isStoreMigrating(error)) {
                    notify('loading');
                    state.migrationTimer = setTimeout(() => start(taskId, state, revision, onDomWrite), 700);
                    return false;
                }
                notify('error');
                return false;
            })
            .finally(() => {
                if (states.get(taskId) !== state || state.inFlight !== request || state.generation !== generation) return;
                state.inFlight = null;
                state.inFlightRevision = null;
                const pending = state.pending;
                state.pending = null;
                if (pending && pending.revision !== state.appliedRevision) {
                    start(taskId, state, pending.revision, pending.onDomWrite);
                }
                onSettled(taskId);
            });
        state.inFlight = request;
        return request;
    };

    return {
        // Saved Review: errors await Retry; a successful empty/404 permits fallback.
        ready(taskId) {
            const state = states.get(taskId);
            return !state || (!state.inFlight && !state.migrationTimer && state.lastStatus !== 'error');
        },
        hydrate(taskIdValue, revisionValue = null, { onDomWrite = null } = {}) {
            const taskId = text(taskIdValue);
            if (!taskId) return Promise.resolve(false);
            const revision = reviewRevision(revisionValue);
            let state = states.get(taskId);
            if (!state) {
                state = {
                    appliedRevision: null,
                    inFlightRevision: null,
                    pending: null,
                    inFlight: null,
                    generation: 0,
                    everApplied: false,
                    lastStatus: 'idle',
                };
                states.set(taskId, state);
            }
            if (revision !== null && revision === state.appliedRevision) {
                return Promise.resolve(false);
            }
            if (state.inFlight) {
                if (revision !== null && revision === state.pending?.revision) {
                    return state.inFlight.then(() => state.inFlight || false);
                }
                if (revision !== null && revision !== state.inFlightRevision) {
                    state.pending = { revision, onDomWrite };
                    return state.inFlight.then(() => state.inFlight || false);
                }
                return state.inFlight;
            }
            return start(taskId, state, revision, onDomWrite);
        },
        invalidateApplied(taskIdValue = '') {
            const taskId = text(taskIdValue);
            if (taskId) {
                const state = states.get(taskId);
                if (state) state.appliedRevision = null;
                return;
            }
            // Reset the presentation receipt after rebuild; keep joining any physical GET.
            for (const state of states.values()) state.appliedRevision = null;
        },
        clear() {
            for (const state of states.values()) clearTimeout(state.migrationTimer);
            states.clear();
            for (const [index, job] of [...detailQueue.entries()].reverse()) {
                if (job.owner === states) detailQueue.splice(index, 1)[0].reject(new Error('review detail read cancelled'));
            }
        },
    };
}

/**
 * Return only an explicit executed receipt. Requested/effective route intent
 * is deliberately ignored: identity markup must not manufacture execution.
 */
export function reviewExecutionEvidence(execution) {
    const wrapped = execution?.executed && typeof execution.executed === 'object';
    const executed = wrapped
        ? execution.executed
        : execution;
    if (!executed || typeof executed !== 'object') return null;
    const kind = text(executed.kind || executed.route_kind || executed.channel).toLowerCase();
    const harness = text(executed.harness_id || executed.harness);
    const api = ['api', 'api_chat', 'api_model', 'native'].includes(kind);
    const harnessReceipt = kind === 'harness' || kind === 'agent_session' || (wrapped && Boolean(harness));
    if (!api && (!harnessReceipt || !harness)) return null;
    return {
        harness: api ? 'api' : harness,
        channel: api ? 'api' : '',
        label: kind === 'native' ? 'API · native tool rounds' : text(executed.label),
        model: text(executed.model || executed.model_id),
    };
}

export function reviewExecutionEvidenceList(executions, legacyExecution = null) {
    const rows = normalizedExecutions(executions, legacyExecution);
    const seen = new Set();
    const evidence = [];
    for (const row of rows) {
        const normalized = reviewExecutionEvidence(row);
        if (!normalized) continue;
        const key = [normalized.harness, normalized.channel, normalized.label, normalized.model].join('\u0000');
        if (seen.has(key)) continue;
        seen.add(key);
        evidence.push(normalized);
    }
    return evidence;
}

export function reviewReferenceFromRow(row) {
    const directType = text(row?.system_type || row?.type || row?.event);
    const direct = directType === 'review_reference' ? row : null;
    const nested = row?.review_reference && typeof row.review_reference === 'object'
        ? row.review_reference
        : (row?.progress_meta?.review_reference && typeof row.progress_meta.review_reference === 'object'
            ? row.progress_meta.review_reference
            : null);
    const reference = direct || nested;
    if (!reference || !['plan_review', 'task_acceptance'].includes(text(reference.surface))) return null;
    const presentationOwnerTaskId = text(
        reference.presentation_owner_task_id || reference.task_id
        || row?.presentation_owner_task_id || row?.task_id,
    );
    if (!presentationOwnerTaskId) return null;
    return {
        surface: text(reference.surface),
        presentationOwnerTaskId,
        stateRevision: text(reference.state_revision),
        reviewFingerprint: text(reference.review_fingerprint),
    };
}

export function renderReviewsSection(groupsInput, disclosure = {}) {
    return renderOpinionReviews(orderedReviewGroups(groupsInput), disclosure, {
        executions: (attempt, { showModel = true } = {}) => reviewExecutionEvidenceList(attempt.executions, attempt.execution).map(execution =>
            harnessIdentityMarkup(execution.harness, { channel: execution.channel, label: execution.label, className: 'chat-review-execution-identity' })
            + (showModel && execution.model ? `<span class="chat-review-execution-model">${escapeHtmlText(execution.model)}</span>` : '')).join(''),
    });
}

// The controls a reader can hold focus on, each named by the attribute that identifies it across updates.
const FOCUS_KEYS = ['reviewGroupToggle', 'reviewReadUrl', 'reviewHistory', 'reviewHydrateRetry', 'reviewAttemptDetail'];
const focusSelector = (key, value) => `[data-${key.replace(/[A-Z]/g, (char) => `-${char.toLowerCase()}`)}${value === '' ? '' : `="${value}"`}]`;

/**
 * Interactive Reviews renderer. Chat owns disclosure and hydration callbacks;
 * review events update content without taking disclosure ownership.
 */
export function createReviewPresentationController({
    host,
    summary,
    disclosure,
    onHydrate = () => {},
    onRead = () => {},
    onDomWrite = (mutate) => mutate(),
} = {}) {
    const groups = new Map();
    const state = disclosure || {};
    if (!(state.expandedGroups instanceof Set)) state.expandedGroups = new Set();
    if (state.sectionExpanded !== true) state.sectionExpanded = false;

    // Keyed reconciliation keeps a focused control in place; a node an update had to replace hands
    // focus to its successor (the same group, source, facts or action), never to the page.
    const focusedControl = () => {
        const active = host?.ownerDocument?.activeElement;
        if (!active || !host?.contains?.(active)) return null;
        const owner = active.closest?.(FOCUS_KEYS.map((key) => focusSelector(key, '')).join(',')) || active;
        const key = FOCUS_KEYS.find((name) => Object.hasOwn(owner.dataset || {}, name));
        if (key) return { key, value: owner.dataset[key], node: active, summary: active.matches?.('summary') === true };
        const facts = active.closest?.('[data-review-facts]');
        return facts ? { key: 'reviewFacts', value: facts.dataset.reviewFacts, node: active, summary: true } : null;
    };
    const restoreFocus = (focused) => {
        if (!focused || focused.node?.isConnected) return;
        const value = String(focused.value ?? '').replaceAll('"', '\\"');
        const target = host.querySelector?.(focusSelector(focused.key, value));
        (focused.summary ? target?.querySelector?.('summary') || target : target)?.focus?.();
    };

    const render = () => onDomWrite(() => {
        if (!host || !summary) return;
        const focused = focusedControl();
        const { activeCount } = reviewGroupCounts(groups);
        const groupCount = reviewViews(orderedReviewGroups(groups)).length;
        const failedEmpty = groupCount === 0 && (
            state.hydrateStatus === 'error'
            || (state.hydrateStatus === 'loading' && state.hadHydrateError === true)
        );
        summary.hidden = groupCount === 0 && !failedEmpty;
        // The door's words are its one text node, before the chevron its card draws (chat.js), as a status chip's are.
        let words = summary.firstChild;
        if (words?.nodeType !== 3) words = summary.insertBefore(summary.ownerDocument.createTextNode(''), words || null);
        const said = groupCount ? `Reviews ${groupCount}${activeCount ? ` · ${activeCount} active` : ''}` : (failedEmpty ? 'Reviews' : '');
        if (words.data !== said) words.data = said;
        summary.setAttribute?.('aria-expanded', state.sectionExpanded ? 'true' : 'false');
        const history = [...groups.values()].find(group => group.historyTask);
        if (history) {
            state.historyTask = history.historyTask;
            state.historyCertain = history.historyCertain !== false;
            if (state.historyOffset === undefined) state.historyMore = history.historyMore;
        }
        reconcileReviewMarkup(host, renderReviewsSection(groups, state));
        restoreFocus(focused);
    });

    summary?.addEventListener?.('click', () => {
        state.sectionExpanded = !state.sectionExpanded;
        render();
        if (state.sectionExpanded) onHydrate();
    });
    host?.addEventListener('click', async (event) => {
        const read = event.target?.closest?.('[data-review-read-url]');
        if (read) {
            const url = read.dataset.reviewReadUrl;
            // The reader names the file by the source's words, never a path a model chose. A reviewer's answer
            // reads as Markdown (its Source view and Download keep the exact text); saved detail stays plain text.
            const markdown = read.dataset.reviewReadKind === 'markdown';
            const filename = `${String(read.dataset.reviewReadName || 'Saved review').replace(/[\\/:]+/g, '-')}.${markdown ? 'md' : 'txt'}`;
            onRead({ filename, mime: markdown ? 'text/markdown' : 'text/plain', kind: markdown ? 'markdown' : 'text', source: { durable: url }, reader: { url },
                provenance: read.dataset.reviewReadSha ? `Saved source ${read.dataset.reviewReadSha}` : 'Saved review text' }, read, event);
            return;
        }
        const history = event.target?.closest?.('[data-review-history]');
        if (history) {
            if (state.historyLoading) return;
            state.historyLoading = true; state.historyError = false; render();
            try {
                const page = await getTaskReviewHistory(state.historyTask, state.historyOffset || 0);
                const incoming = reviewGroupsFromTaskDetail(page, state.historyTask);
                for (const group of incoming) mergeReviewGroup(groups, group);
                state.historyOffset = page.next_offset;
                state.historyMore = page.next_offset !== null && page.next_offset !== undefined;
            } catch { state.historyError = true; }
            state.historyLoading = false; render(); return;
        }
        const hydrateRetry = event.target?.closest?.('[data-review-hydrate-retry]');
        if (hydrateRetry) {
            // A failed pass never records an applied revision, so a plain
            // revision-less re-hydrate always re-issues the physical GET.
            onHydrate();
            const status = host.querySelector?.('[data-review-hydrate-status]');
            if (status) {
                status.setAttribute?.('tabindex', '-1');
                status.focus?.();
            }
            return;
        }
        const groupToggle = event.target?.closest?.('[data-review-group-toggle]');
        if (!groupToggle) return;
        const groupId = groupToggle.dataset.reviewGroupToggle || '';
        if (!groupId) return;
        if (state.expandedGroups.has(groupId)) state.expandedGroups.delete(groupId);
        else state.expandedGroups.add(groupId);
        render();
        if (state.expandedGroups.has(groupId)) onHydrate();
    });

    return {
        groups,
        render,
        update(group) {
            const prior = groups.get(group?.id), merged = mergeReviewGroup(groups, group);
            if (merged && merged !== prior) render();
            return merged === prior ? false : merged;
        },
        updateMany(nextGroups) {
            let changed = false;
            for (const group of Array.isArray(nextGroups) ? nextGroups : []) {
                const prior = groups.get(group?.id), merged = mergeReviewGroup(groups, group);
                if (merged && merged !== prior) changed = true;
            }
            if (changed) render();
            return changed;
        },
        setHydrateStatus(statusValue) {
            const statusVisible = (value) => groups.size > 0 || value === 'error'
                || (value === 'loading' && state.hadHydrateError === true);
            const status = text(statusValue);
            if (state.hydrateStatus === status) return false;
            const wasVisible = statusVisible(state.hydrateStatus);
            // Remember a failure across the retry's loading pass so the
            // zero-group shell stays mounted until the retry settles.
            if (status === 'error') state.hadHydrateError = true;
            else if (status === 'idle') state.hadHydrateError = false;
            state.hydrateStatus = status;
            const isVisible = statusVisible(status);
            if (wasVisible || isVisible) render();
            return wasVisible || isVisible;
        },
    };
}
