/**
 * The failed-check row of a task card (owner decision 13.10, #1585).
 *
 * A `verify_and_record` call can complete normally while the criterion it checks is
 * not met. The call stays one ordinary call in the folded evidence row (no error, no
 * task status change); the recorded verdict gets its own expandable diagnostic row,
 * live and on replay, keyed beside the call so a later settlement never retires it
 * and a later passing check never erases it. Every word comes from typed facts: the
 * producer's `verification` object (live frames carry it directly, the settled row
 * inside `tool_result_meta`) and the process facts beside it — never the result text.
 * `placeReplayedEvidence` is the replay side: it draws the evidence row, these rows and any refused steps from
 * the host's replayed observations, under the keys the live frames used.
 */
import { replayedEvidenceTs, updateLiveTimelineItem, upsertToolFoldRow } from './chat_render_batch.js';
import { fmt, tr } from './i18n.js';
import { failedStepHeadline, handledRefusalHeadline, heldCallPhrase, refusedByStop, refusedStepPhrase } from './task_reason_words.js';

const MISSED_MATCH = {
    substring: ['task.check.text_not_found', 'expected text not found'],
    exact: ['task.check.output_not_exact', 'output did not match exactly'],
    exact_line: ['task.check.line_not_exact', 'no output line matched exactly'],
    json_equals: ['task.check.json_not_equal', 'JSON output not equal'],
};

const oneLine = (text, max) => {
    const s = String(text || '').replace(/\s+/g, ' ').trim();
    return s.length > max ? `${s.slice(0, max - 3)}...` : s;
};

function extractCommandText(args) {
    const cmd = args?.cmd;
    if (Array.isArray(cmd)) return cmd.map((part) => String(part || '').trim()).filter(Boolean).join(' ');
    return typeof cmd === 'string' ? cmd.trim() : '';
}

/** The same invocation diagnostic for a live settlement and a bounded history observation. */
export function toolStepDiagnostic(evt, callKey) {
    const command = extractCommandText(evt.args);
    const result = String(evt.result_preview || evt.error || '').trim();
    const refusal = refusedStepPhrase(evt);
    const held = heldCallPhrase(evt.tool, evt.status) || (evt.status === 'blocked' && refusal);
    const extra = !held && refusal;  // an independent failure keeps its own headline
    const warning = held || handledRefusalHeadline(evt) || evt.status === 'non_zero_exit';
    return {
        phase: refusedByStop(evt) ? 'info' : warning ? 'warn' : evt.status === 'interrupted' ? 'info' : 'error',
        headline: failedStepHeadline(evt),
        body: oneLine([extra, command && `Command: ${oneLine(command, 120)}`, result].filter(Boolean).join(' '), 220),
        fullBody: [extra, command && `Command: ${command}`, result].filter(Boolean).join('\n\n'),
        visible: true, hostRefusedStep: Boolean(refusal), dedupeKey: callKey, sourceTs: evt.ts || '',
    };
}

/** What the check's own process did, said first so a failed criterion never reads as a
 *  failed command: a completed check whose criterion missed, a check that exited
 *  non-zero, or one the host stopped or a signal killed (process facts, not text). Exit 0
 *  says "not met" only where a comparison ran and missed (`matched` is false); a comparison
 *  that reached no verdict (a bytes_equal operand missing or refused) is not confirmed. */
export function checkProcessSentence(evt, check = {}) {
    if (check.contract_kind === 'artifact_observation') return '';
    if (evt.timed_out === true) return tr('task.check.process_timed_out', 'The host stopped the check at its time limit; no comparison ran.');
    if (evt.signal) return fmt('The check process was killed by {signal}.', { signal: evt.signal });
    if (Number.isInteger(evt.exit_code) && evt.exit_code !== 0) return fmt('The check ran and exited with code {code}.', { code: evt.exit_code });
    if (evt.exit_code !== 0) return '';
    return check.matched === false
        ? tr('task.check.process_completed', 'The check ran successfully (exit 0); its criterion was not met.')
        : tr('task.check.process_unconfirmed', 'The check ran successfully (exit 0); its comparison reached no verdict, so the criterion is not confirmed.');
}

/** Why the check did not pass, in the order the facts establish it: a host kill or a
 *  failing exit says the comparison is not the verdict; otherwise the mode's miss.
 *  A bytes_equal detail is the producer's own sentence and may name a refused operand,
 *  a missing file or a comparison failure (then no `matched`), so it is shown as recorded.
 *  Without a recorded miss the criterion is unconfirmed, never asserted unmet. */
export function checkFailureCause(evt, check) {
    if (check.contract_kind === 'artifact_observation' || check.kind === 'artifact_observation') {
        if (check.paths_named == null) return tr('task.check.not_confirmed', 'criterion not confirmed');
        return check.paths_named ? tr('task.check.file_not_found', 'a named file was not found') : tr('task.check.no_files_named', 'no files were named');
    }
    if (evt.timed_out === true) return tr('task.check.timed_out', 'timed out');
    if (evt.signal) return fmt('killed by {signal}', { signal: evt.signal });
    if (Number.isInteger(evt.exit_code) && evt.exit_code !== 0) return fmt('exit code {code}', { code: evt.exit_code });
    if (check.expected_match === 'bytes_equal') {
        return String(check.detail || '').split('\n')[0].trim()
            || (check.matched === false ? tr('task.check.bytes_not_equal', 'files not byte-equal') : tr('task.check.comparison_unconfirmed', 'comparison not confirmed'));
    }
    if (check.matched === false) return tr(...(MISSED_MATCH[check.expected_match] || MISSED_MATCH.substring));
    return tr('task.check.not_confirmed', 'criterion not confirmed');
}

/** The task's detail sentence for its recorded red check (`outcome_axes.objective.verification_red`, the host's
 *  `_outcome_receipts.red_check_fact`): the cause in this row's own words and whether any check passed after it, or ''
 *  when the record holds no such fact (its stored sentence stands). Twin: tool_error_words.red_check_phrase. */
export function redCheckPhrase(objective) {
    const fact = objective?.verification_red;
    if (!fact || typeof fact !== 'object') return '';
    const count = (value) => (Number.isInteger(value) && value > 0 ? value : 0);
    const cause = fact.kind === 'artifact_observation'  // the host observed the named files and ran no command
        ? (fact.paths_named ? tr('task.check.file_not_found', 'a named file was not found') : tr('task.check.no_files_named', 'no files were named'))
        : !fact.signal && !Number.isInteger(fact.exit_code) ? tr('task.check.no_exit_code', 'no exit code')
            : fact.signal || fact.exit_code !== 0 ? checkFailureCause(fact, fact)  // the row beside says the process exited 0; this line says it itself
                : fmt('exit code 0; {cause}', { cause: checkFailureCause(fact, fact) });
    const after = count(fact.passed_after), more = count(fact.more);
    const head = after === 1 ? fmt('A recorded check did not pass ({cause}); 1 check passed after it, none recorded as a rerun of it', { cause })
        : after ? fmt('A recorded check did not pass ({cause}); {n} checks passed after it, none recorded as a rerun of it', { cause, n: after })
            : fmt('The last recorded check did not pass ({cause}); no check passed after it', { cause });
    const tail = more === 1 ? tr('task.check.more_one', '; 1 more failed check has no recorded passing rerun')
        : more ? fmt('; {n} more failed checks have no recorded passing rerun', { n: more }) : '';
    return `${head}${tail}.`;
}

/** The diagnostic row for one failed check, or null. `callKey` is the call's own
 *  timeline key; a failed CALL (is_error) keeps its existing error row instead. */
export function checkFailureView(evt, callKey) {
    const check = evt?.verification || evt?.tool_result_meta?.verification;
    if (!callKey || evt.is_error || check?.status !== 'fail') return null;
    const process = checkProcessSentence(evt, check);
    const subject = [
        check.check ? fmt('Check: {check}', { check: check.check }) : '',
        check.criterion_id ? fmt('Criterion: {criterion}', { criterion: check.criterion_id }) : '',
    ].filter(Boolean).join(' · ');
    const preview = String(evt.result_preview || '').trim();
    return {
        phase: 'warn',
        headline: fmt('Check did not pass: {cause}', { cause: checkFailureCause(evt, check) }),
        body: oneLine([process, subject, check.contract_kind === 'artifact_observation' ? check.detail : ''].filter(Boolean).join(' '), 220),
        fullBody: [process, subject, preview || check.detail || ''].filter(Boolean).join('\n\n'),
        visible: true,
        dedupeKey: `${callKey}:check`,
        sourceTs: String(evt.ts || ''),
    };
}

/** Draw a marked settlement with the live call formatter. Its command arguments are not part of
 *  the history projection; its recorded time, invocation key and bounded result preview are. */
export function refusedStepView(seen, key, presentCall) {
    if (!key || seen?.fact !== 'settled' || seen.tool_result_meta?.refused_step !== true
        || !['held', 'error'].includes(seen.status)) return null;
    const { phase, headline, body, fullBody } = presentCall({ type: 'tool_call', is_error: true,
        tool: seen.tool, status: seen.recorded_status, tool_result_meta: seen.tool_result_meta, result_preview: seen.result_preview });
    return { phase, headline, body, fullBody, visible: true, dedupeKey: key, sourceTs: String(seen.ts || '') };
}

/** A settled invocation's diagnostic keeps its live key, even after the task ended. */
export function placeToolDiagnostic(record, view, ts, rawTs = view.sourceTs) {
    const { timelineUpdate } = updateLiveTimelineItem(record, view,
        { ts, rawTs, syntheticKey: view.dedupeKey, headline: view.headline, inPlaceByKey: true });
    return !['none', 'duplicate-skip'].includes(timelineUpdate);
}

/** Replayed host evidence (`tool_call_log.replay_evidence`) draws the rows the live frames drew:
 *  the folded evidence row (`fold`, its view) at its first call's time, then failed checks and refused steps
 *  under their live keys. `carrierTs` is the history row that delivered the evidence and
 *  `formatTs` the card's timestamp formatter. True when the timeline changed. */
export function placeReplayedEvidence(record, fold, evidence, carrierTs, formatTs, presentCall) {
    const at = replayedEvidenceTs(evidence, carrierTs);
    let changed = !['none', 'duplicate-skip'].includes(upsertToolFoldRow(record, fold, formatTs(at), at).timelineUpdate);
    for (const seen of evidence?.observations || []) {
        const settled = record.toolFold?.calls.get(seen.key)?.settlement;
        if (seen.fact !== 'settled' || !settled || settled.status !== seen.status || settled.hostError !== Boolean(seen.hostError)) continue;
        const view = refusedStepView(seen, seen.key, presentCall) || (seen.diagnostic ? toolStepDiagnostic({ ...seen.diagnostic, tool: seen.tool, ts: seen.ts,
            is_error: seen.status === 'error' || seen.status === 'interrupted' }, seen.key) : checkFailureView(seen, seen.key));
        if (view) changed = placeToolDiagnostic(record, view, formatTs(view.sourceTs)) || changed;
    }
    return changed;
}
