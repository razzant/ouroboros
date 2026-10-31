// Read-only projection of an old record's local preparation incident (stored before the result
// review replaced task acceptance; nothing writes one now); no reviewer or lifecycle authority.
const text = (value) => String(value ?? '').trim();

// The host's own LOCAL acceptance-evidence failure, read from whichever carrier
// the stored record holds: the review projection an old checkpoint published,
// or the terminal acceptance decision. One incident id, one host attempt count —
// a replayed frame therefore states the same row rather than a second incident.
export function acceptanceIncidentFromTaskDetail(detail) {
    const fromProjection = detail?.review_projection?.acceptance_incident;
    const decision = detail?.outcome_axes?.review?.acceptance_decision
        || detail?.review_status?.acceptance_decision;
    const incident = (fromProjection && typeof fromProjection === 'object' ? fromProjection : null)
        || (decision?.acceptance_incident && typeof decision.acceptance_incident === 'object'
            ? decision.acceptance_incident : null);
    if (!incident || !text(incident.incident_id)) return null;
    const attempts = Number(incident.attempts || 0) || 0;
    // The host projects nothing for a preparation that never failed; a frame
    // that still carries a zero count is no incident either (twin guard).
    if (attempts <= 0) return null;
    const resolved = text(incident.status) === 'resolved';
    return {
        id: text(incident.incident_id),
        attempts,
        resolved,
        stage: text(incident.stage) || 'preparation',
        sourceKnown: incident.source_known !== false,
        failureKind: text(incident.failure_kind),
        failureDetail: text(incident.failure_detail),
        exposed: incident.feedback_delivered === true,
        retryBasis: text(incident.retry?.basis),
        // The ACTIVE warning is what a resolution clears; the row itself stays,
        // so the history of the failure is never rewritten away.
        warning: resolved ? '' : (
            `Acceptance evidence could not be assembled locally${
                incident.failure_kind ? ` (${text(incident.failure_kind)})` : ''
            } — host attempt ${attempts}; no new reviewer was dispatched for this attempt.`),
        summary: resolved
            ? `Acceptance evidence assembled after ${attempts} failed host attempt${attempts === 1 ? '' : 's'}.`
            : `Acceptance evidence could not be assembled locally — host attempt ${attempts}.`,
    };
}

export function acceptanceIncidentAttempt(incident) {
    return {
        // Stable across live delivery, history and a reconnect: mergeReviewGroup
        // keys on it, so repeated publications update ONE row.
        id: `acceptance-incident:${incident.id}`,
        surface: 'task_acceptance',
        state: 'terminal',
        progress: '',
        tone: incident.resolved ? 'neutral' : 'warn',
        verdict: incident.resolved ? 'resolved' : 'no verdict',
        timestamp: '',
        ordinal: -1,
        label: `acceptance evidence · host attempt ${incident.attempts}`,
        summary: incident.summary,
        superseded: false,
        replayed: false,
        revised: false,
        executions: [],
        execution: null,
        detailRef: { surface: 'task_acceptance', url: '' },
        detailText: [
            incident.summary,
            `incident: ${incident.id}`,
            `stage: ${incident.stage} (no new reviewer was dispatched for this attempt; earlier panels keep their own rows)`,
            `host attempts on this material: ${incident.attempts}`,
            incident.sourceKnown ? '' : 'material identity: unknown (the source could not be read)',
            incident.failureKind ? `failure: ${incident.failureKind}` : '',
            incident.failureDetail ? `detail: ${incident.failureDetail}` : '',
            incident.exposed ? 'the author received this preparation failure' : '',
            incident.retryBasis ? `explicit retry: ${incident.retryBasis}` : '',
        ].filter(Boolean).join('\n'),
    };
}

// Both simultaneous facts of a locally failed acceptance: the host's own evidence
// assembly failed AND some rail ended the task. Either one alone used to replace
// the other, which is how a record with no new reviewer read as "the requested
// rework never happened". The clause speaks for THIS preparation attempt only;
// an earlier real reviewer FAIL keeps its own sentence beside it. Present only
// when the decision carries the typed incident, so historical records read
// exactly as before; a resolved incident states nothing. `phrases` is the
// caller's own cause vocabulary — the browser twin of
// project_dialogue._incident_cause_clauses.
export function acceptanceIncidentClauses(decision, reason, phrases) {
    const incident = decision?.acceptance_incident;
    if (!incident || typeof incident !== 'object' || String(incident.status || '') === 'resolved') return [];
    // `phrases` is the caller's cause lookup: a function (the install-language seam) or the
    // raw English table, so no presenter reads the table past the translation seam.
    const phrase = typeof phrases === 'function' ? phrases : (key) => (phrases || {})[key];
    const rail = reason && reason !== 'final_message' ? (phrase(reason) || '') : '';
    return [phrase('acceptance_preparation_failed'), rail];
}

// The Reviews group of ONE task's acceptance: the panel rows the task really
// has, with an unresolved local preparation failure prepended as its own row.
// That failure produces NO panel, so the group exists on the incident alone or
// the owner sees nothing at all. `statusTone` is the caller's shared tone
// vocabulary; this builder adds no verdict, lifecycle or cost authority.
export function acceptanceGroupWithIncident({ owner, attempts, incident, authorDecisionText, statusTone }) {
    if (incident) attempts.unshift(acceptanceIncidentAttempt(incident));
    const latest = attempts.at(-1);
    return {
        id: `task_acceptance:${owner}`,
        surface: 'task_acceptance',
        label: 'Task acceptance',
        subject: '',
        presentationOwnerTaskId: owner,
        subjectTaskId: owner,
        initiatorTaskId: owner,
        state: latest?.state === 'running' ? 'running' : 'terminal',
        progress: text(latest?.progress),
        // An unresolved local failure colours the group; a resolution clears
        // that active warning and leaves the row as history.
        tone: incident?.warning ? 'warn' : (latest?.tone || statusTone('terminal', latest?.verdict)),
        verdict: text(latest?.verdict),
        summary: incident?.warning || text(latest?.summary),
        // ALWAYS present: a merge spreads the incoming group over the prior one,
        // so an omitted key would keep a cleared warning alive forever.
        warning: incident?.warning || '',
        authorDecisionText,
        activeCount: latest?.state === 'running' ? 1 : 0,
        attemptCount: attempts.length,
        countIsAuthoritative: true,
        attempts,
    };
}
