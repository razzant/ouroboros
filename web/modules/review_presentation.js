import { escapeHtmlAttr } from './utils.js';
import { taskSourceDownloadUrl } from './api_client.js';
import { harnessIdentityMarkup } from './harness_presentation.js';
import { reconcileReviewMarkup } from './review_dom_patch.js';

const escapeHtmlText = escapeHtmlAttr;

const SURFACE_ORDER = new Map([
    ['skill', 0],
    ['plan', 1],
    ['task_acceptance', 2],
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

export function setReviewAnchor(record, enabled, writePhase) {
    if (!record || Boolean(record.reviewAnchor) === enabled) return false;
    record.reviewAnchor = enabled;
    record.phaseEl.hidden = enabled;
    if (enabled) {
        record.titleEl.textContent = record.suggestedName || 'Reviews';
        record.inlineTypingEl.style.display = 'none';
    } else {
        writePhase(record, 'working');
        if (!record.suggestedName && !record.lastHumanHeadline) {
            record.titleEl.textContent = 'Working...';
        }
        record.inlineTypingEl.style.display = '';
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

function lifecycleMeta(status) {
    const token = text(status).toLowerCase();
    return lifecycleFailure(token) ? `lifecycle ${token}` : '';
}

function hasSemanticVerdict(verdict) {
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
        detailRef: skill && jobId ? { surface: 'skill', skill, jobId } : null,
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

function planAttempt(wave, index, isCurrent) {
    if (!wave || typeof wave !== 'object') return null;
    const fingerprint = text(wave.request_fingerprint);
    const id = attemptIdentity(wave, `${fingerprint || 'wave'}:${wave.cycle_index ?? index + 1}`);
    if (!id) return null;
    const verdict = text(wave.aggregate || 'UNKNOWN');
    const superseded = Boolean(wave.superseded) || !isCurrent;
    // A persisted wave is a completed reviewer call even when the plan gate
    // intentionally remains open (`closed=false`). Liveness belongs only to a
    // current attempt for which no matching wave has landed yet.
    const state = superseded ? 'superseded' : 'terminal';
    return {
        id,
        surface: 'plan',
        state,
        tone: statusTone(state, verdict),
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
    // explicitly instead of contributing silently-zero findings.
    const lines = [];
    for (const actor of (Array.isArray(wave.actors) ? wave.actors : [])) {
        if (!actor || typeof actor !== 'object' || actor.ok !== false) continue;
        const identity = [text(actor.slot_id), text(actor.model)].filter(Boolean).join(' · ') || 'reviewer';
        const cause = text(actor.failure_code) || text(actor.error) || 'no parseable verdict';
        lines.push(`Reviewer unavailable: ${identity} — ${cause}`);
    }
    return lines;
}

function planWaveDetail(wave) {
    const lines = [
        wave.aggregate ? `Verdict: ${wave.aggregate}` : '',
        wave.closed != null ? `Closed: ${wave.closed ? 'yes' : 'no'}` : '',
        wave.paid != null ? `Reviewer panel dispatched: ${wave.paid ? 'yes' : 'no'}` : '',
        wave.quorum_unreachable ? 'Quorum unavailable' : '',
        wave.cycles_exhausted ? 'Review cycles exhausted' : '',
        wave.reason ? `Reason: ${text(wave.reason)}` : '',
    ];
    const counts = wave.counts && typeof wave.counts === 'object' ? wave.counts : {};
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
    const stateRecord = detail?.plan_review_state;
    if (!owner || !stateRecord || typeof stateRecord !== 'object') return null;
    if (stateRecord.schema_version === 1) return legacyPlanReviewGroup(owner, stateRecord);
    const current = stateRecord.current_attempt && typeof stateRecord.current_attempt === 'object'
        ? stateRecord.current_attempt : {};
    const recordedWaves = Array.isArray(stateRecord.waves) ? stateRecord.waves : [];
    if (!recordedWaves.length && !text(current.status)) return null;
    const currentFingerprint = text(current.fingerprint);
    const typedCurrentStatus = text(current.status).toLowerCase();
    // C-09: a compact row proves history, not reusable authority. While the
    // same envelope is being reviewed again, replace that stale projection
    // with current_attempt; the eventual full wave reuses the same identity.
    const waves = recordedWaves.filter((wave) => !(
        wave?.compact
        && currentFingerprint
        && typedCurrentStatus
        && text(wave.request_fingerprint) === currentFingerprint
        && (typedCurrentStatus === 'open' || wave.closed === true)
    ));
    const currentWaveIndex = currentFingerprint
        ? waves.findIndex((wave) => text(wave?.request_fingerprint) === currentFingerprint)
        : (waves.length ? waves.length - 1 : -1);
    const attempts = waves
        .map((wave, index) => planAttempt(
            wave,
            index,
            index === currentWaveIndex,
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
    if (!currentAttempt) {
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
    let currentVerdict = text(currentAttempt?.verdict || currentStatus);
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
    return {
        id: `plan:${owner}`,
        surface: 'plan',
        label: 'Plan review',
        subject: '',
        presentationOwnerTaskId: owner,
        subjectTaskId: owner,
        initiatorTaskId: owner,
        state,
        tone: statusTone(state, currentVerdict),
        verdict: currentVerdict,
        summary: text(current.reason || currentAttempt?.summary),
        activeCount,
        attemptCount: attempts.length + (finiteCount(stateRecord.waves_omitted) || 0),
        countIsAuthoritative: finiteCount(stateRecord.waves_omitted) === 0,
        attempts,
    };
}

function compactCoverage(coverage) {
    if (!coverage || typeof coverage !== 'object') return '';
    return Object.entries(coverage)
        .filter(([, value]) => value !== '' && value !== null && value !== undefined)
        .map(([key, value]) => `${key}=${String(value)}`)
        .join(', ');
}

export function formatReviewProjection(projection) {
    const panels = Array.isArray(projection?.panels) ? projection.panels : [];
    const lines = [];
    panels.forEach((panel, panelIndex) => {
        if (!panel || typeof panel !== 'object') return;
        const quorum = panel.quorum && typeof panel.quorum === 'object' ? panel.quorum : {};
        const panelId = String(panel.panel_id || `panel-${panelIndex + 1}`);
        lines.push(
            `Review panel ${panelId}: ${String(panel.surface || 'review')} · authority=${String(panel.authority || 'unspecified')} · verdict=${String(panel.aggregate_signal || 'UNKNOWN')} · transport=${String(panel.transport_status || 'unknown')} · parse=${String(panel.parse_status || 'unknown')} · quorum=${String(quorum.contributed ?? 0)}/${String(quorum.configured ?? 0)} (required ${String(quorum.required ?? 0)}) · enforcement=${String(panel.enforcement_impact || 'unknown')}${panel.single_reviewer_no_diversity ? ' · single-reviewer (no diversity)' : ''}${panel.dialogue && panel.dialogue.status ? ` · dialogue=${String(panel.dialogue.status)}` : ''}${panel.superseded ? ' · superseded' : ''}`,
        );
        if (panel.reason) lines.push(`Panel reason: ${String(panel.reason)}`);
        const coverage = compactCoverage(panel.coverage);
        if (coverage) lines.push(`Panel coverage: ${coverage}`);
        const binding = [
            panel.candidate_hash ? `candidate_hash=${String(panel.candidate_hash)}` : '',
            panel.evidence_revision ? `evidence_revision=${String(panel.evidence_revision)}` : '',
            panel.fence_hash ? `fence_hash=${String(panel.fence_hash)}` : '',
            panel.binding_hash ? `binding_hash=${String(panel.binding_hash)}` : '',
        ].filter(Boolean);
        if (binding.length) lines.push(`Panel binding: ${binding.join(' · ')}`);
        (Array.isArray(panel.actors) ? panel.actors : []).forEach((actor) => {
            if (!actor || typeof actor !== 'object') return;
            const slotId = String(actor.slot_id || '?');
            lines.push(
                `Reviewer ${slotId}: role=${String(actor.actor_role || 'reviewer')} · provider=${String(actor.provider || 'unknown')} · model=${String(actor.model || 'unknown')} · transport=${String(actor.transport_status || 'unknown')} · parse=${String(actor.parse_status || 'unknown')} · verdict=${String(actor.semantic_verdict || 'none')}${actor.outcome_tier ? ` · outcome_tier=${String(actor.outcome_tier)}` : ''}${actor.dialogue_status ? ` · dialogue=${String(actor.dialogue_status)}` : ''} · quorum=${actor.quorum_contribution ? 'contributes' : 'abstains'} · enforcement=${String(actor.enforcement_impact || 'unknown')}`,
            );
            const actorCoverage = compactCoverage(actor.coverage);
            if (actorCoverage) lines.push(`Reviewer ${slotId} coverage: ${actorCoverage}`);
            if (actor.reason) lines.push(`Reviewer ${slotId} reason: ${String(actor.reason)}`);
            if (Array.isArray(actor.findings)) {
                for (const finding of actor.findings) {
                    if (!finding || typeof finding !== 'object') continue;
                    const label = [text(finding.severity), text(finding.verdict)]
                        .filter(Boolean).join(' ') || 'finding';
                    const title = text(finding.item) || text(finding.summary) || '(no item)';
                    const summaryText = text(finding.summary);
                    const body = [
                        `[${label}]${text(finding.id) ? ` ${text(finding.id)}` : ''} ${title}`,
                        summaryText && summaryText !== title ? `summary: ${summaryText}` : '',
                        text(finding.reason) ? `reason: ${text(finding.reason)}` : '',
                        text(finding.evidence) ? `evidence: ${text(finding.evidence)}` : '',
                        text(finding.recommendation) ? `fix: ${text(finding.recommendation)}` : '',
                    ].filter(Boolean).join(' — ');
                    lines.push(`Reviewer ${slotId} finding: ${body}`);
                }
                const omitted = finiteCount(actor.findings_omitted);
                if (omitted) lines.push(`Reviewer ${slotId} findings omitted: ${omitted}`);
            }
            // P1: name the durable full copy unconditionally — bounded rows,
            // per-string truncation markers and pre-findings-era projections
            // all resolve through the same observability call.
            const callId = text(actor.response_ref?.call_id);
            if (callId) {
                lines.push(`Reviewer ${slotId} full response: observability call ${callId}`);
            }
        });
    });
    return lines.join('\n');
}

export function taskAcceptanceGroupFromTaskDetail(detail, ownerTaskId = '') {
    const owner = text(ownerTaskId || detail?.task_id);
    const projection = detail?.review_projection;
    const panels = (Array.isArray(projection?.panels) ? projection.panels : [])
        .filter((panel) => text(panel?.surface) === 'task_acceptance');
    if (!owner || !panels.length) return null;
    const attempts = panels.map((panel, index) => {
        const verdict = text(panel?.aggregate_signal || 'UNKNOWN');
        return {
            id: [attemptIdentity(panel, `panel:${index + 1}`),
                panel.task_attempt == null ? '' : `task-attempt:${panel.task_attempt}`,
                panel.panel_index == null ? '' : `panel-index:${panel.panel_index}`].filter(Boolean).join(':'),
            surface: 'task_acceptance',
            state: panel?.superseded ? 'superseded' : 'terminal',
            tone: statusTone('terminal', verdict),
            verdict,
            timestamp: text(panel?.ts || panel?.timestamp),
            ordinal: index,
            label: `panel ${text(panel?.panel_id || index + 1)}`,
            summary: text(panel?.reason),
            superseded: Boolean(panel?.superseded),
            replayed: false,
            revised: false,
            initiatorTaskId: owner,
            executions: executionsFromReviewRecord(panel),
            execution: null,
            detailRef: { surface: 'task_acceptance', url: panel.applied_source_status === 'available'
                ? taskSourceDownloadUrl(owner, panel.applied_source_ref, 'task_acceptance_review') : '' },
            detailText: `${formatReviewProjection({ panels: [panel] })}\nCost unavailable`.trim(),
        };
    });
    const latest = attempts.at(-1);
    return {
        id: `task_acceptance:${owner}`,
        surface: 'task_acceptance',
        label: 'Task acceptance',
        subject: '',
        presentationOwnerTaskId: owner,
        subjectTaskId: owner,
        initiatorTaskId: owner,
        state: 'terminal',
        tone: statusTone('terminal', latest?.verdict),
        verdict: text(latest?.verdict),
        summary: text(latest?.summary),
        activeCount: 0,
        attemptCount: attempts.length,
        countIsAuthoritative: true,
        attempts,
    };
}

export function reviewGroupsFromTaskDetail(detail, ownerTaskId = '') {
    return [
        planReviewGroupFromTaskDetail(detail, ownerTaskId),
        taskAcceptanceGroupFromTaskDetail(detail, ownerTaskId),
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
                    ...attempt, state: 'superseded', tone: 'neutral', superseded: true,
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
    if (staleActiveRegression) {
        merged.state = prior.state;
        merged.verdict = prior.verdict;
        merged.summary = prior.summary;
        merged.activeCount = prior.activeCount;
        merged.lifecycleStatus = prior.lifecycleStatus;
        merged.tone = reviewTone(merged.state, merged.verdict, merged.lifecycleStatus);
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
        merged.tone = 'working';
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

// Revisions are opaque SHA-256 tokens; one distinct token may trail a live GET.
export function createReviewHydrator({ fetchDetail, applyDetail, onState = () => {} } = {}) {
    const states = new Map();

    const start = (taskId, state, revision, onDomWrite) => {
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
                return fetchDetail(taskId);
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
            .catch(() => {
                notify('error');
                return false;
            })
            .finally(() => {
                if (state.inFlight !== request || state.generation !== generation) return;
                state.inFlight = null;
                state.inFlightRevision = null;
                const pending = state.pending;
                state.pending = null;
                if (pending && pending.revision !== state.appliedRevision) {
                    start(taskId, state, pending.revision, pending.onDomWrite);
                }
            });
        state.inFlight = request;
        return request;
    };

    return {
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
                if (
                    revision !== null
                    && revision !== state.inFlightRevision
                ) {
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
            // A full DOM rebuild discards the projection that an applied
            // revision hydrated, but it must retain and join any physical GET
            // already in flight. Reset only the applied presentation receipt.
            for (const state of states.values()) state.appliedRevision = null;
        },
        clear() {
            states.clear();
        },
    };
}

function attemptMeta(attempt) {
    return [
        attempt.timestamp,
        attempt.verdict || (attempt.lifecycleOnly
            ? (['queued', 'running'].includes(attempt.state) ? attempt.state : 'review verdict unavailable')
            : ''),
        lifecycleMeta(attempt.lifecycleStatus),
        attempt.superseded ? 'superseded' : '',
        attempt.replayed ? 'replay' : '',
        attempt.revised ? 'revised snapshot' : '',
    ].filter(Boolean).join(' · ');
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
    const groups = orderedReviewGroups(groupsInput);
    // A failed FIRST hydration has no groups to hang the error on: the shell
    // renders anyway and stays mounted through the retry's own loading pass
    // (hadHydrateError) so the recovery control cannot unmount mid-flight. A
    // quiet first-load zero-group loading pass stays invisible (every card
    // expand hydrates, most tasks have no reviews).
    const hydrateStatus = text(disclosure.hydrateStatus);
    const emptyShell = !groups.length && (
        hydrateStatus === 'error'
        || (hydrateStatus === 'loading' && disclosure.hadHydrateError === true)
    );
    if (!groups.length && !emptyShell) return '';
    const expandedGroups = disclosure.expandedGroups instanceof Set ? disclosure.expandedGroups : new Set();
    const expandedAttempts = disclosure.expandedAttempts instanceof Set ? disclosure.expandedAttempts : new Set();
    const sectionExpanded = disclosure.sectionExpanded === true;
    const { groupCount, activeCount } = reviewGroupCounts(groups);
    const countText = groupCount
        ? `${groupCount}${activeCount ? ` · ${activeCount} active` : ''}`
        : '—';
    const groupHtml = groups.map((group) => {
        const groupExpanded = expandedGroups.has(group.id);
        const shown = group.countIsAuthoritative ? `${group.attemptCount}` : `${group.attempts.length} shown`;
        const attempts = group.attempts.map((attempt) => {
            const attemptKey = `${group.id}:${attempt.id}`;
            const attemptExpanded = expandedAttempts.has(attemptKey);
            const skillRef = attempt.detailRef?.surface === 'skill' ? attempt.detailRef : null;
            const detailAttrs = skillRef
                ? ` data-skill-review-skill="${escapeHtmlAttr(skillRef.skill)}" data-skill-review-job="${escapeHtmlAttr(skillRef.jobId)}"`
                : '';
            const source = attempt.detailRef?.surface === 'task_acceptance' ? attempt.detailRef : null;
            const fullSource = source ? (source.url
                ? `<a class="btn btn-default" href="${escapeHtmlAttr(source.url)}" download>Download full applied review</a>`
                : '<span>Full applied review unavailable.</span>') : '';
            const detail = skillRef ? '' : `<span>${escapeHtmlText(attempt.detailText || attempt.summary || 'No additional detail projected.')}</span>${fullSource ? `<div>${fullSource}</div>` : ''}`;
            const executions = reviewExecutionEvidenceList(attempt.executions, attempt.execution);
            const executionMarkup = executions.map((execution) => (
                harnessIdentityMarkup(execution.harness, {
                    channel: execution.channel,
                    label: execution.label,
                    className: 'chat-review-execution-identity',
                }) + (execution.model
                    ? `<span class="chat-review-execution-model">${escapeHtmlText(execution.model)}</span>`
                    : '')
            )).join('');
            const attemptInitiator = (
                attempt.initiatorTaskId
                && attempt.initiatorTaskId !== group.presentationOwnerTaskId
                && attempt.initiatorTaskId !== group.initiatorTaskId
            )
                ? `<div class="chat-review-initiator">Initiated by task ${escapeHtmlText(attempt.initiatorTaskId)}</div>`
                : '';
            return `
                <div class="chat-review-attempt ${escapeHtmlAttr(attempt.tone)}${attempt.superseded ? ' superseded' : ''}" data-review-attempt="${escapeHtmlAttr(attemptKey)}">
                    <div class="chat-review-attempt-main">
                        <span class="chat-review-attempt-label">${escapeHtmlText(attempt.label)}</span>
                        <span class="chat-review-attempt-meta">${escapeHtmlText(attemptMeta(attempt))}</span>
                        ${executionMarkup ? `<span class="chat-review-execution">${executionMarkup}</span>` : ''}
                    </div>
                    ${attemptInitiator}
                    <button type="button" class="chat-review-detail-toggle" data-review-attempt-toggle="${escapeHtmlAttr(attemptKey)}" aria-expanded="${attemptExpanded ? 'true' : 'false'}">${attemptExpanded ? 'Hide details' : 'Show details'}</button>
                    <div class="chat-review-attempt-detail" data-review-attempt-detail="${escapeHtmlAttr(attemptKey)}"${detailAttrs} aria-busy="false"${attemptExpanded ? '' : ' hidden'}>${detail}</div>
                </div>`;
        }).join('');
        const initiator = group.initiatorTaskId && group.initiatorTaskId !== group.presentationOwnerTaskId
            ? `<div class="chat-review-initiator">Initiated by task ${escapeHtmlText(group.initiatorTaskId)}</div>`
            : '';
        return `
            <div class="chat-review-group ${escapeHtmlAttr(group.tone)}" data-review-group="${escapeHtmlAttr(group.id)}">
                <button type="button" class="chat-review-group-toggle" data-review-group-toggle="${escapeHtmlAttr(group.id)}" aria-expanded="${groupExpanded ? 'true' : 'false'}">
                    <span class="chat-review-group-main">
                        <span class="chat-review-group-label">${escapeHtmlText(group.label)}</span>
                        ${group.subject ? `<span class="chat-review-subject">${escapeHtmlText(group.subject)}</span>` : ''}
                    </span>
                    <span class="chat-review-group-meta">${escapeHtmlText([group.verdict || (group.lifecycleOnly
                        ? (['queued', 'running'].includes(group.state) ? group.state : 'review verdict unavailable')
                        : group.state), lifecycleMeta(group.lifecycleStatus), shown].filter(Boolean).join(' · '))}</span>
                </button>
                <div class="chat-review-attempts"${groupExpanded ? '' : ' hidden'}>
                    <div class="chat-review-group-cost">Cost unavailable</div>
                    ${initiator}${attempts}
                </div>
            </div>`;
    }).join('');
    // Section-level hydration truth (typed controller state, no per-attempt
    // FSM): a first load announces itself, a failed refresh names itself and
    // offers Retry instead of silently presenting stale-or-missing detail.
    // The message rides inside a <span>: the keyed status node is patched in
    // place across loading↔error, and the DOM patcher syncs text only through
    // childless-element innerHTML — a bare text node beside the Retry button
    // would survive the transition stale.
    const hydrateHtml = hydrateStatus === 'loading'
        ? '<div class="skill-review-loading" data-review-hydrate-status role="status" aria-live="polite"><span>Loading review details…</span></div>'
        : (hydrateStatus === 'error'
            ? '<div class="skill-review-error" data-review-hydrate-status role="alert"><span>Review details failed to refresh — shown data may be incomplete. </span><button type="button" class="skill-review-retry" data-review-hydrate-retry>Retry</button></div>'
            : '');
    // The status node sits OUTSIDE the collapsible groups container: a failed
    // refresh stays visible on a collapsed section and the keyed node survives
    // loading↔error transitions. Disclosure stays user-owned — nothing expands.
    return `
        <section class="chat-live-reviews" data-review-section data-expanded="${sectionExpanded ? '1' : '0'}">
            <button type="button" class="chat-review-section-toggle" data-review-section-toggle aria-expanded="${sectionExpanded ? 'true' : 'false'}">
                <span>Reviews</span><span class="chat-review-section-count">${escapeHtmlText(countText)}</span>
            </button>
            ${hydrateHtml}<div class="chat-review-groups"${sectionExpanded ? '' : ' hidden'}>${groupHtml}</div>
        </section>`;
}

/**
 * Interactive Reviews renderer. Chat owns disclosure and hydration callbacks;
 * review events update content without taking disclosure ownership.
 */
export function createReviewPresentationController({
    host,
    summary,
    disclosure,
    onHydrate = () => {},
    onLoadSkillDetail = () => {},
    onDomWrite = (mutate) => mutate(),
} = {}) {
    const groups = new Map();
    const state = disclosure || {};
    if (!(state.expandedGroups instanceof Set)) state.expandedGroups = new Set();
    if (!(state.expandedAttempts instanceof Set)) state.expandedAttempts = new Set();
    if (state.sectionExpanded !== true) state.sectionExpanded = false;

    const focusedControl = () => {
        const active = host?.ownerDocument?.activeElement;
        if (!active || !host?.contains?.(active)) return null;
        if (active.matches?.('[data-review-section-toggle]')) return { kind: 'section', key: '' };
        if (active.dataset?.reviewGroupToggle) return { kind: 'group', key: active.dataset.reviewGroupToggle };
        if (active.dataset?.reviewAttemptToggle) return { kind: 'attempt', key: active.dataset.reviewAttemptToggle };
        const detail = active.closest?.('[data-review-attempt-detail]');
        if (detail?.dataset?.reviewAttemptDetail) {
            return { kind: 'attempt', key: detail.dataset.reviewAttemptDetail };
        }
        return null;
    };

    const restoreFocus = (focused) => {
        if (!focused) return;
        let target = null;
        if (focused.kind === 'section') target = host.querySelector?.('[data-review-section-toggle]');
        const attribute = focused.kind === 'group'
            ? 'reviewGroupToggle'
            : (focused.kind === 'attempt' ? 'reviewAttemptToggle' : '');
        if (attribute) {
            target = Array.from(host.querySelectorAll?.(`[data-${focused.kind === 'group' ? 'review-group-toggle' : 'review-attempt-toggle'}]`) || [])
                .find((candidate) => candidate.dataset?.[attribute] === focused.key);
        }
        target?.focus?.();
    };

    const render = () => onDomWrite(() => {
        if (!host || !summary) return;
        const focused = focusedControl();
        const { groupCount, activeCount } = reviewGroupCounts(groups);
        const failedEmpty = groupCount === 0 && (
            state.hydrateStatus === 'error'
            || (state.hydrateStatus === 'loading' && state.hadHydrateError === true)
        );
        summary.hidden = groupCount === 0 && !failedEmpty;
        summary.textContent = groupCount
            ? `Reviews ${groupCount}${activeCount ? ` · ${activeCount} active` : ''}`
            : (failedEmpty ? 'Reviews' : '');
        const reconciled = reconcileReviewMarkup(host, renderReviewsSection(groups, state));
        const active = host?.ownerDocument?.activeElement;
        if (!reconciled || !active || !host.contains?.(active)) restoreFocus(focused);
        for (const detail of host.querySelectorAll?.('[data-review-attempt-detail]') || []) {
            if (
                !detail?.hidden
                && detail.dataset?.skillReviewSkill
                && detail.dataset?.skillReviewJob
            ) onLoadSkillDetail(detail);
        }
    });

    host?.addEventListener('click', (event) => {
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
        const retry = event.target?.closest?.('[data-skill-review-retry]');
        if (retry) {
            const detail = retry.closest?.('[data-review-attempt-detail]');
            if (detail?.dataset?.skillReviewSkill && detail?.dataset?.skillReviewJob) {
                onLoadSkillDetail(detail, { retry: true });
                detail.setAttribute?.('tabindex', '-1');
                detail.focus?.();
            }
            return;
        }
        const sectionToggle = event.target?.closest?.('[data-review-section-toggle]');
        if (sectionToggle) {
            state.sectionExpanded = !state.sectionExpanded;
            render();
            if (state.sectionExpanded) onHydrate();
            return;
        }
        const groupToggle = event.target?.closest?.('[data-review-group-toggle]');
        if (groupToggle) {
            const groupId = groupToggle.dataset.reviewGroupToggle || '';
            if (!groupId) return;
            if (state.expandedGroups.has(groupId)) state.expandedGroups.delete(groupId);
            else state.expandedGroups.add(groupId);
            render();
            if (state.expandedGroups.has(groupId)) onHydrate();
            return;
        }
        const attemptToggle = event.target?.closest?.('[data-review-attempt-toggle]');
        if (!attemptToggle) return;
        const attemptKey = attemptToggle.dataset.reviewAttemptToggle || '';
        if (!attemptKey) return;
        const opening = !state.expandedAttempts.has(attemptKey);
        if (opening) state.expandedAttempts.add(attemptKey);
        else state.expandedAttempts.delete(attemptKey);
        render();
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
