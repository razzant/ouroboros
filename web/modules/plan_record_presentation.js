// The planning organ's record (plan_record, schema 3) as one Reviews group: revisions,
// asks, participants, contributions, open objections and done_when. Nobody votes and no
// verdict exists, so the group reports facts — the current revision, how many objections
// stand open (and whether an owner-marked reviewer's may hold) and whether answers are still
// on their way. Which objections stand comes from the host's fold in the task detail
// (plan_objections); this module never re-derives it. The retired organ's v1/v2 records keep
// their read-only branches in review_presentation.js.

const text = (value) => String(value ?? '').trim();
// What an ask was about, as the host names it (plan_record.ask_subject): an ask before any plan
// carries no revision (null) and is a reading of the request.
const askSubject = (revision) => (revision == null ? 'the reading of the request' : `revision ${revision}`);

export const planRecordEngaged = (record) => record && typeof record === 'object'
    && ((Array.isArray(record.revisions) && record.revisions.length) || (Array.isArray(record.asks) && record.asks.length));

function hostObjections(detail) {
    // Which objections stand is the host's fold (plan_record.fold_objections, projected by the task
    // detail as plan_objections: open objections by seat, imported earlier-schema findings included).
    // The panel renders it and re-derives nothing; null when the detail carries no projection.
    const view = detail?.plan_objections;
    const bySeat = view && typeof view === 'object' && view.open_objections && typeof view.open_objections === 'object'
        ? view.open_objections : null;
    if (!bySeat) return null;
    const rows = [];
    for (const [seat, items] of Object.entries(bySeat)) {
        for (const item of (Array.isArray(items) ? items : [])) {
            if (!item || typeof item !== 'object') continue;
            rows.push({ seat: text(seat), id: text(item.id), text: text(item.text), askId: text(item.ask_id), revision: item.revision,
                holds: item.holds === true, notAdjudicated: item.not_adjudicated === true, legacy: item.legacy === true });
        }
    }
    return rows;
}


function planDoneWhenLines(revision) {
    const rows = Array.isArray(revision?.done_when) ? revision.done_when : [];
    if (!rows.length) return ['done_when: (none stated yet)'];
    return ['done_when:', ...rows.map((row) => `  ${text(row.id)}: ${text(row.claim)}${text(row.support) ? ` — shown by: ${text(row.support)}` : ''}`)];
}

export function planRecordGroup(owner, record, detail) {
    const revisions = (Array.isArray(record.revisions) ? record.revisions : []).filter((r) => r && typeof r === 'object');
    const asks = (Array.isArray(record.asks) ? record.asks : []).filter((a) => a && typeof a === 'object');
    const latest = revisions.at(-1) || null;
    const hostRows = hostObjections(detail);
    const known = hostRows !== null;
    const objections = hostRows || [];
    const holding = objections.filter((o) => o.holds);
    const imported = objections.filter((o) => o.legacy);
    const legacyUnreadable = detail?.plan_objections && typeof detail.plan_objections === 'object'
        ? detail.plan_objections.legacy_unreadable : null;
    const live = text(detail?.status).toLowerCase() === 'running';
    const attempts = [];
    if (latest) {
        attempts.push({
            id: `revision:${latest.n}`, surface: 'plan', state: 'terminal', tone: 'neutral',
            verdict: `revision ${latest.n}${latest.body ? ' · body plan' : ''}`,
            timestamp: text(latest.ts), ordinal: 0, label: `plan revision ${latest.n} of ${revisions.length}`,
            summary: text(latest.goal), superseded: false, compact: false, replayed: false, revised: revisions.length > 1,
            initiatorTaskId: '', executions: [], execution: null, detailRef: null,
            detailText: [
                `goal: ${text(latest.goal) || '(none)'}`,
                ...planDoneWhenLines(latest),
                `affected_paths: ${JSON.stringify(Array.isArray(latest.affected_paths) ? latest.affected_paths : [])}`,
                latest.body ? `body plan: ${text(latest.body_note) || 'changes Ouroboros\'s own body'}` : '',
                ...revisions.slice(0, -1).map((r) => `earlier revision ${r.n} (${text(r.ts) || '?'}): ${
                    (Array.isArray(r.done_when) ? r.done_when : []).map((c) => `${text(c.id)}: ${text(c.claim)}`).join('; ') || 'no done_when'}`),
                ...imported.map((o) => `imported objection ${o.seat}/${o.id} from the previous plan review${o.holds ? ' (may hold)' : ''}${
                    o.notAdjudicated ? ' (not adjudicated)' : ''}: ${o.text}`),
                known ? '' : 'open objections: not projected by the host in this detail',
                legacyUnreadable && typeof legacyUnreadable === 'object'
                    ? `earlier plan-review state unreadable (${text(legacyUnreadable.source)}): ${text(legacyUnreadable.error)}` : '',
                'Cost unavailable',
            ].filter(Boolean).join('\n'),
        });
    }
    asks.forEach((ask, index) => {
        const participants = (Array.isArray(ask.participants) ? ask.participants : []).filter((p) => p && typeof p === 'object');
        const pending = participants.filter((p) => p.status === 'pending');
        const answered = participants.filter((p) => p.status === 'answered');
        const raised = objections.filter((o) => o.askId === text(ask.ask_id));
        const running = pending.length > 0 && live;
        const lines = [
            `ask ${text(ask.ask_id)} on ${askSubject(ask.revision)}${text(ask.message) ? ` — «${text(ask.message)}»` : ''}`,
            ...(Array.isArray(ask.host_added) && ask.host_added.length
                ? [`${ask.revision == null ? 'a reading about Ouroboros\'s own body' : 'first ask on a body plan'}: the host added ${
                    ask.host_added.join(', ')}`] : []),
            ...participants.map((p) => {
                const who = `${text(p.seat_id)} (${text(p.model) || '?'}${p.owner_marked ? '' : ', added — never holds'})`;
                if (p.status === 'answered') {
                    const reply = p.reply || {};
                    const n = (Array.isArray(reply.objections) ? reply.objections : []).length;
                    const w = (Array.isArray(reply.withdrawn) ? reply.withdrawn : []).length;
                    const prose = text(reply.prose);
                    if (ask.revision == null) {  // a reading's points stay out of the fold: shown here as advice, with their text
                        const points = (Array.isArray(reply.objections) ? reply.objections : [])
                            .map((o) => text(o && typeof o === 'object' ? (o.text || o.claim || o.id) : o)).filter(Boolean);
                        return `${who}: answered${prose ? `: ${prose}` : ''}${points.length ? ` — advice, never holds: ${points.join('; ')}` : ''}`;
                    }
                    return `${who}: answered — ${n} objection${n === 1 ? '' : 's'}${w ? `, ${w} withdrawn` : ''}${prose ? `: ${prose}` : ''}`;
                }
                if (p.status === 'not_dispatched') return `${who}: not sent — ${text(p.not_dispatched_reason) || 'not sendable'}`;
                if (p.status === 'pending') return `${who}: awaiting`;
                return `${who}: ${text(p.status)}${text(p.error || p.parse_error) ? ` — ${text(p.error || p.parse_error)}` : ''}`;
            }),
            ...raised.map((o) => `objection ${o.seat}/${o.id}${o.holds ? '' : ' (never holds)'}${o.notAdjudicated ? ' (not adjudicated)' : ''}: ${o.text}`),
            'Cost unavailable',
        ];
        attempts.push({
            id: `ask:${text(ask.ask_id)}`, surface: 'plan', state: running ? 'running' : 'terminal',
            progress: pending.length ? `${answered.length} of ${participants.length} answered` : '',
            tone: running ? 'working' : (raised.length ? 'warn' : 'done'),
            verdict: raised.length ? `${raised.length} objection${raised.length === 1 ? '' : 's'}` : (pending.length ? 'awaiting' : 'no objection'),
            timestamp: text(ask.ts), ordinal: index + 1, label: `ask ${text(ask.ask_id)} · ${askSubject(ask.revision)}`,
            summary: text(ask.message), superseded: false, compact: false, replayed: false, revised: false,
            // The host's fold for this ask (which objections stand), or null when the detail did not project it.
            initiatorTaskId: '', executions: [], execution: null, detailRef: null, participants, objections: known ? raised : null,
            detailText: lines.join('\n'),
        });
    });
    const awaiting = asks.some((ask) => (Array.isArray(ask.participants) ? ask.participants : []).some((p) => p?.status === 'pending'));
    const state = awaiting && live ? 'running' : 'terminal';
    const verdict = !known ? 'objections not projected'
        : holding.length ? `${holding.length} objection${holding.length === 1 ? '' : 's'} may hold`
            : objections.length ? `${objections.length} open objection${objections.length === 1 ? '' : 's'}`
                : awaiting ? 'awaiting answers' : (asks.length ? 'no open objection' : 'not asked yet');
    return {
        id: `plan:${owner}`, surface: 'plan', label: 'Plan', subject: latest ? `revision ${latest.n}` : '',
        presentationOwnerTaskId: owner, subjectTaskId: owner, initiatorTaskId: owner,
        state, progress: '', tone: !known ? 'neutral' : holding.length ? 'warn' : (objections.length ? 'warn' : (state === 'running' ? 'working' : 'done')),
        verdict, summary: text(latest?.goal), authorDecisionText: '',
        activeCount: state === 'running' ? 1 : 0, attemptCount: attempts.length, countIsAuthoritative: true, attempts,
    };
}
