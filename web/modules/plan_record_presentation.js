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


// What the own room reached a seat: the rows its delivery carried inline and what its file reads covered of the
// exact snapshot. Separate facts, neither a measure of understanding: no percentage, no tone, and an unobserved read
// says nothing at all. The rows were counted before any send: a seat never sent was delivered none, and a pending
// outcome says the delivery may not have gone. A stored row with no provenance or measured status (an older
// projector's paged brief or blank 0 of N) measured nothing of the room, so it says nothing either.
const roomCount = (value) => Number(value || 0).toLocaleString('en-US');
const READ_PROVENANCE_WORDS = { host_observed: 'observed by the host', harness_observed: 'inferred from the harness journal' };

function planRoomLine(participant) {
    const inline = participant.status !== 'not_dispatched' && participant.room_delivery && typeof participant.room_delivery === 'object'
        ? participant.room_delivery : null;
    const read = participant.room_read_coverage && typeof participant.room_read_coverage === 'object' ? participant.room_read_coverage : null;
    const provenance = READ_PROVENANCE_WORDS[text(read?.provenance)];
    const facts = [
        inline ? `${roomCount(inline.conversation_inline_rows)} of ${roomCount(inline.conversation_rows)} conversation rows inline${
            Number(inline.other_rows) > 0 ? `, ${roomCount(inline.other_rows)} other rows only in the snapshot file` : ''}` : '',
        inline && participant.status === 'pending' ? 'counted for a delivery that may not have been sent' : '',
        provenance && ['complete', 'incomplete'].includes(text(read.status))
            ? `file reads covered ${roomCount(read.covered_chars)} of ${roomCount(read.complete_chars)} characters · ${provenance}` : '',
    ].filter(Boolean);
    return facts.length ? `Participant ${text(participant.seat_id)} room: ${facts.join(' · ')}` : '';
}

// One ask's evidence as its row recorded it (attach, attached, omissions; the host's plan_record.ask_evidence): each
// locator attached whole or in part, which means prepared for the participants, never that one was sent it (its own
// status says that) or read it, or not attached, each with every reason recorded for it. A cut, a repeat or a history
// gap stays beside content that attached. A field an older writer did not record is not recorded, never none.
const PART_REASONS = ['truncated_to_', 'chat_history_gap:'];

export function askEvidence(ask) {
    const listed = (key) => (Array.isArray(ask?.[key]) ? ask[key] : null);
    const declared = listed('attach');
    const attached = listed('attached');
    const omissions = listed('omissions');
    const rows = new Map((attached || []).filter((row) => row && typeof row === 'object').map((row) => [text(row.locator), row]));
    const reasons = new Map();
    for (const row of omissions || []) {
        if (!row || typeof row !== 'object') continue;
        reasons.set(text(row.locator), [...(reasons.get(text(row.locator)) || []), text(row.reason)]);
    }
    const items = [];
    for (const locator of new Set([...(declared || []).map(text), ...rows.keys(), ...reasons.keys()])) {
        const row = rows.get(locator);
        const why = reasons.get(locator) || [];
        let state;
        if (!locator) continue;
        if (row) state = !omissions ? 'attached_unknown' : why.some((reason) => PART_REASONS.some((prefix) => reason.startsWith(prefix))) ? 'part' : 'whole';
        else if (!attached) state = 'unknown';
        else if (why.length || !omissions) state = 'not_attached';
        else state = 'unknown';
        items.push({ locator, state, reasons: why, bytes: row?.bytes ?? null, sha256: text(row?.sha256) });
    }
    const count = (state) => items.filter((item) => item.state === state).length;
    return {
        recorded: { attach: declared !== null, attached: attached !== null, omissions: omissions !== null },
        items, counts: { whole: count('whole'), part: count('part'), attached_unknown: count('attached_unknown'), not_attached: count('not_attached'), unknown: count('unknown') },
    };
}

// Owner words for the reasons the host records; Record facts keep each code beside them, and an unlisted code is shown
// as it is.
const REASON_WORDS = [
    ['duplicate_locator', 'listed more than once'], ['url_not_fetched', 'a link: links are listed, never fetched'],
    ['denied_path', 'settings or runtime data outside this task\'s own stores'],
    ['outside_allowed_roots', 'outside the folders this task can read'], ['sensitive', 'a credential-like file name'],
    ['missing', 'not found'], ['directory', 'a folder, not a file'], ['binary', 'not text'], ['unreadable', 'could not be read'],
    ['symlink_loop', 'a link that loops'], ['budget_exhausted', 'the ask\'s evidence budget was already spent'],
    ['task_not_found', 'no saved result for that task'], ['chat_not_found', 'no such conversation'],
    ['unsupported_locator', 'not a locator plan_task reads'], ['read_target_differs', 'that address reads another file'],
    ['too_large:', 'too large to attach'], ['read_blocked:', 'refused by the read rules'],
    ['task_unreadable:', 'could not be read'], ['chat_unreadable:', 'could not be read'],
    ['chat_history_gap:', 'the conversation history has a gap'], ['source_unretained:', 'its whole source was not kept'],
    ['invalid_selector', 'the selection is not valid'], ['selector_out_of_range', 'the selection is outside the source'],
    ['symbol_', 'the symbol could not be selected'],
];
const byteCount = (value) => Number(value).toLocaleString('en-US');

function reasonWords(reason, item) {
    const cut = /^truncated_to_(\d+)$/.exec(reason);
    if (cut) return `the first ${byteCount(cut[1])}${item.bytes != null ? ` of ${byteCount(item.bytes)}` : ''} bytes (${reason})`;
    const words = REASON_WORDS.find(([code]) => (code.endsWith(':') || code.endsWith('_') ? reason.startsWith(code) : reason === code))?.[1];
    return words ? `${words} (${reason})` : reason;
}

function planEvidenceLines(ask) {
    const id = text(ask.ask_id);
    const { recorded, items, counts } = askEvidence(ask);
    if (!items.length) {
        return recorded.attach || (recorded.attached && recorded.omissions) ? [] : [`Ask ${id} evidence: not recorded`];
    }
    const describe = (item) => [item.locator, item.state === 'attached_unknown' ? 'completeness not recorded' : '',
        item.bytes != null && !item.reasons.some((reason) => reason.startsWith('truncated_to_')) ? `source ${byteCount(item.bytes)} bytes` : '',
        ...item.reasons.map((reason) => `${item.state === 'part' && PART_REASONS.some((prefix) => reason.startsWith(prefix))
            ? 'in part: ' : ''}${reasonWords(reason, item)}`),
        item.sha256 ? `sha256 ${item.sha256}` : ''].filter(Boolean).join(' · ');
    const group = (label, states) => {
        const rows = items.filter((item) => states.includes(item.state)).map(describe);
        return rows.length ? [`Ask ${id} ${label}: ${rows[0]}`, ...rows.slice(1).map((row) => `  ${row}`)] : [];
    };
    return [
        `Ask ${id} evidence: prepared for its participants: ${evidenceCounts(counts, items.length)}`,
        ...group('attached', ['whole', 'part', 'attached_unknown']), ...group('not attached', ['not_attached']),
        ...group('declared, attachment not recorded', ['unknown']),
        recorded.omissions ? '' : `Ask ${id} evidence cuts and absences: not recorded`,
    ].filter(Boolean);
}

const evidenceCounts = (counts, total) => [`${counts.whole} of ${total} attached whole`, counts.part ? `${counts.part} in part` : '',
    counts.attached_unknown ? `${counts.attached_unknown} attached, completeness not recorded` : '',
    counts.not_attached ? `${counts.not_attached} not attached` : '', counts.unknown ? `${counts.unknown} not recorded` : '']
    .filter(Boolean).join(', ');

// The ask's one evidence sentence on its opened row, only when what was prepared for the participants is incomplete;
// which locators and why stay in Record facts.
function planEvidenceNotice(ask) {
    const { items, counts } = askEvidence(ask);
    return counts.part || counts.attached_unknown || counts.not_attached || counts.unknown ? `Evidence for this ask: ${evidenceCounts(counts, items.length)}.` : '';
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
            ...planEvidenceLines(ask),
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
            ...participants.map(planRoomLine).filter(Boolean),
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
            evidenceNotice: planEvidenceNotice(ask), detailText: lines.join('\n'),
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
