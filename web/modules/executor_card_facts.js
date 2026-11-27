// A delegated child's executor facts as its card and Dashboard Logs read them: the executor its delegation selected (a
// request, never evidence), the dispatch chip a frame states, the chip a card keeps across later frames and attempts,
// and the model and request facts its line and Details print. It reads projected frames and terminal records only;
// identity (marks, names, route grammar) is harness_presentation's, and no route, billing or account choice is made here.

import { META_PART_SEPARATOR, compactModel, coordinatorFacts, displayApiModelTarget, formatCoordinatorProvenance, harnessPresentation,
    joinMetaParts, routeHarness, routeIdentity, selectedModel, selectedRoute, taskAttempt } from './harness_presentation.js';
import { fmt } from './i18n.js';
import { UNSTARTED_REASONS } from './task_reason_words.js';

// Durable task detail is allowed to finish a card only at one of the task
// result store's genuinely-settled statuses. In particular, interrupted and
// the legacy cancel_requested latch remain retryable rather than becoming a
// fabricated Done/Stopped projection.
export const TERMINAL_TASK_DETAIL_STATUSES = new Set([
    'completed', 'failed', 'cancelled', 'rejected_duplicate',
]);
// Statuses that make the OUTCOME known, whether or not the card may close yet.
export const OBSERVED_OUTCOME_STATUSES = new Set([...TERMINAL_TASK_DETAIL_STATUSES, 'done', 'cancel_requested']);

export function normalizeTaskTerminalRecord(evt) {
    if (!evt || typeof evt !== 'object') return evt || {};
    const terminalStatus = String(evt.task_terminal_status || '').trim();
    return terminalStatus ? { ...evt, status: terminalStatus } : evt;
}

// Phase 6, owner directive #1: «бейдж точно нужен, но не рекламный … что ТУТ
// бабл \ субагент на codex». A small chip naming the harness this bubble or
// subagent was DISPATCHED to — icon plus the short harness name, in the style of
// the agent account rows on the Agents tab, never a promotional badge.
// Dispatch, not receipt: `executor_route` is resolved once when the work is sent,
// so the chip says where it was sent; a landing below that ask is disclosed on
// `capability_delta`. Only a DELEGATED route is worth a chip: the native API path
// is the ordinary case and prints nothing (absent fact -> null -> no element).
// The completion seam's typed substrate claim (subagents.actual_substrate) rides
// the terminal frame beside the counts and joins the detail sentence; the enum
// never travels bare.
const SUBSTRATE_NOTE = {
    harness_used: 'custody evidence confirms a harness run',
    harness_attempted: 'harness attempted; no successful run confirmed',
    native_only: 'no harness run recorded',
};

// Quiet label (owner choice #360 A). `label` is the helper's name, with one short
// ` · phrase` only where the bare name would read as a receipt nothing issued: a
// progress actor (`last update`), a refused pin (`blocked`, frozen), unreadable
// evidence, no run, runs not settled, failed or unverified runs, an undisclosed
// outcome split. The ordinary case — an assigned route while the card's own phase
// chip says Working, or every settled run ok — is the bare name: it names the
// selected helper and never proves its session computes now. `title` is the
// detail sentence (counts, model and its source, spend, substrate, access);
// the expanded card prints it in its Executor details row, because a hover
// title alone is invisible on touch, to AT and in copies. `hasEvidence`, `ended`,
// `observation`, `sourceTs` and the revision feed keepStickyExecutorChip; `dispatched` cardExecutorChip.
export function executorChip(evt, known = null) {
    const coordinatorProvenance = formatCoordinatorProvenance(evt);
    // The delegation's request, read before any evidence below: the frame's, else the one its card keeps (`known`).
    const selected = requestedExecutor(evt) || known;
    if (selected?.kind === 'api_model') return null;  // an API model runs on no harness: a harness it calls is its tool
    const record = normalizeTaskTerminalRecord(evt), sourceTs = Date.parse(evt?.ts || evt?.timestamp || '') || 0;
    const status = String(record.outcome_axes?.lifecycle?.status || record.status || evt?.subagent_event || '').toLowerCase();
    const ended = OBSERVED_OUTCOME_STATUSES.has(status) || status === 'rejected';
    // A frame names its task attempt (`task_attempt`); an observation of another known attempt is not its own (the host's rule).
    const raw = evt?.executor_observation, attempt = taskAttempt(evt?.task_attempt), seen = taskAttempt(raw?.task_attempt);
    const observation = raw && raw.task_id === String(evt?.subagent_task_id || evt?.task_id || '') && raw.run_id && raw.attempt_id
        && raw.harness_id && Number.isInteger(raw.revision) && (attempt === null || seen === null || attempt === seen) ? raw : null;
    let report = '';
    if (observation) {
        // A well-formed observation is the run's own progress report. A live frame without execution evidence keeps `last
        // update` (an observation, not a receipt) and its requested/observed model source, or `model unconfirmed`, in the
        // detail; a terminal record or receipt instead speaks for the run's end below and keeps this report as history.
        const name = harnessPresentation(observation.harness_id).label;
        const model = observation.model_source === 'observed' || observation.model_source === 'requested'
            ? compactModel(observation.model, false) : '';  // a harness reports no lane
        const asked = selectedModel(selected, observation.harness_id);  // a model-less report keeps its own harness's request
        report = `Latest observed activity: ${name}, ${model ? `model ${model} (${observation.model_source})` : asked ? `model ${asked} (requested)` : 'model unconfirmed'}, run ${observation.run_id}, attempt ${observation.attempt_id}, revision ${observation.revision}. This is progress, not a terminal execution receipt.`;
        if (!ended && !evt?.execution_evidence) return {
            harness: observation.harness_id,
            hasEvidence: false,
            observation,
            sourceTs,
            label: `${name} · last update`,
            title: report,
            ...(coordinatorProvenance ? { coordinatorProvenance } : {}), ...(attempt !== null && { attempt }),
        };
    }
    // The route id is OPAQUE (`harness[=model]`): only its harness part is printed, a head that is no harness id (an
    // API model target) names none, and before dispatch the selection names its harness.
    const routed = routeHarness(evt?.executor_route), harness = routed || selected?.harness || observation?.harness_id || '';
    if (!harness) return null;
    const name = harnessPresentation(harness).label, asked = selectedModel(selected, harness);
    const modelClause = asked ? `, model ${asked} (requested)` : '';
    const base = {  // a route, a recorded start or a run's own report proves the work was sent; a selection alone does not
        harness, label: name, sourceTs, dispatched: Boolean(routed || observation || status === 'running'),
        ...(coordinatorProvenance ? { coordinatorProvenance } : {}), ...(attempt !== null && { attempt }),
    };
    const chip = (hasEvidence, phrase, title) => ({ ...base, hasEvidence, label: phrase ? `${name} · ${phrase}` : name, title });
    // LAYERED TRUTH: identity (mark + product name) comes from the harness_presentation SSOT; the route is a DISPATCH
    // decision; whether a delegated run happened is EVIDENCE, reconciled once at the completion seam
    // (subagents.envelope_from_task -> execution_evidence) on the terminal frame.
    const evidence = evt?.execution_evidence && typeof evt.execution_evidence === 'object' ? evt.execution_evidence : null;
    // The substrate clause is a completion-seam claim coupled to evidence: never attach it to an evidence-less frame, where
    // it could contradict the dispatch. A receipt ends the run: its card keeps the run's last report as history (cardExecutorChip).
    const substrateNote = evidence ? (SUBSTRATE_NOTE[String(evt?.actual_substrate || '')] || '') : '';
    const withSubstrate = (title) => (substrateNote ? `${title} — ${substrateNote}` : title);
    const history = report && { lastUpdate: report, ...((seen ?? attempt) !== null && { lastUpdateAttempt: seen ?? attempt }) };
    const receipt = (made) => ({ ...made, ended, ...(history && { ...history, title: `${made.title}. ${report}` }) });
    if (UNSTARTED_REASONS.has(String(evt?.reason_code || ''))) {
        // The typed $0 terminal of a helper the host could not start (#363): its route was refused,
        // or its delegate tools were hidden or uncheckable; the route names the pin, the code says it never ran.
        // Checked before the evidence branches — an empty custody read would
        // otherwise print "no run yet" over a child never dispatched at all.
        // Evidence-grade so a later dispatch-shaped frame cannot downgrade it.
        return { ...chip(true, 'blocked', `Pinned to ${name}, but this subagent could not start — the task was NOT run (no metered API spend, no delegated run)`), dispatched: false };
    }
    if (!evidence) {
        // A dispatched route's terminal record without run evidence stays explicitly unknown. A selection that ended before any
        // dispatch (cancelled while queued, refused at admission) had no run to lose: it keeps its executor and request (`undispatched`,
        // worded at paint: executorTitle), no missing-receipt warning or promise of one, unless its card saw a start (`dispatchedEnd`, cardExecutorChip).
        const unknown = { ...chip(false, 'run outcome unknown', `The ${name} route was assigned${modelClause}, but this terminal record has no delegated-run evidence; the run outcome is unknown${report && `. ${report}`}`), ended, dispatched: true, ...history };
        if (ended) return base.dispatched ? unknown : { ...chip(false, '', `Selected ${name}${modelClause}; no dispatch was recorded before it ended`), ended, undispatched: { model: asked || '' }, dispatchedEnd: unknown };
        // Evidence rides TERMINAL frames only, so a live frame proves nothing
        // either way, and under the pre-start charter the leaf usually IS running
        // by now: an evidence-grade negative ("no run yet") would be false for most
        // of the live phase. The bare name states the assignment; only work already sent is promised its receipt.
        // The supervising agent's route and billing ride beside the sentence, for the card's Coordinator row.
        return { ...chip(false, '', `${routed ? 'Dispatched to' : 'Selected'} ${name}${modelClause}${base.dispatched ? ' — run evidence arrives with the terminal receipt' : ''}`),
            coordinator: coordinatorFacts(evt) };
    }
    const started = Number(evidence.delegated_runs_started || 0);
    const settled = Number(evidence.delegated_runs_settled || 0);
    // Historical frames (v6.94–v6.99) carry delegated_runs_succeeded without
    // delegated_runs_failed: reconstruct the exact complement rather than
    // rendering a clean receipt over an all-failed delegation. A frame with
    // neither counter disclosed no outcome split, so it never reads clean.
    let failed = Number(evidence.delegated_runs_failed ?? NaN);
    if (!Number.isFinite(failed)) {
        const succeeded = Number(evidence.delegated_runs_succeeded ?? NaN);
        failed = Number.isFinite(succeeded) ? Math.max(0, settled - succeeded) : 0;
    }
    failed = Math.max(0, failed);
    if (evidence.evidence_read_failed) {
        // The custody log EXISTS but could not be (fully) read: the counts are
        // UNKNOWN, not an established fact (sol finding, b49f8192 wave) — past
        // recorded starts too, where the partial work-order replay sets the flag.
        // No substrate clause: the seam never claims one over unreadable evidence.
        return receipt(chip(true, 'evidence unavailable', started
            ? `The ${name} route was assigned and at least ${started} delegated run(s) started, but the evidence could not be fully read — final counts are unknown`
            : `The ${name} route was assigned, but the delegated-run evidence could not be read — whether a run happened is unknown, not "none"`));
    }
    if (!started) {
        return receipt(chip(true, 'no run yet', withSubstrate(`The ${name} route was assigned, but there is no durable record of a delegated run for this subagent`)));
    }
    if (!settled) {
        // Terminal-frame material: started-but-unsettled means the run(s) never
        // settled (orphaned or lost), not "still executing" — a present-tense
        // "running" on a finished card would be a lie.
        return receipt(chip(true, 'not settled', withSubstrate(`Delegated to your ${name} account — ${started} run(s) started, none settled`)));
    }
    const cost = evidence.subscription_cost_usd;
    const approx = evidence.subscription_cost_estimated ? '~' : '';
    const costPart = (cost === null || cost === undefined)
        ? 'subscription spend undisclosed'
        : evidence.subscription_cost_partial
            ? `Tracked: ${approx ? 'up to ' : ''}$${Number(cost).toFixed(2)} (total unknown)`
            : `${approx}$${Number(cost).toFixed(2)} subscription`;
    const runsPart = `${settled} run${settled === 1 ? '' : 's'}`;
    // Unverified work-order coverage is an honesty fact: those runs sit inside the
    // aggregate failed count, discounted from ok, so label and detail name them
    // instead of claiming a physical failure, and a mix says "failed or unverified".
    // Nothing orders failures and recoveries in time: the counts carry no chronology.
    // The detail keeps the owner dictionary "N ok, M failed" (plan D9) when either
    // counter is disclosed. Access rides the detail — requested surface from the
    // frame, applied access from the settlement receipts when the engine disclosed
    // it (empty = predates the receipt).
    const counted = Number.isFinite(Number(evidence.delegated_runs_failed ?? NaN))
        || Number.isFinite(Number(evidence.delegated_runs_succeeded ?? NaN));
    const unresolved = Number(evidence.delegated_runs_source_unresolved || 0);
    const notOk = Math.max(failed, unresolved);
    const ok = Math.max(0, settled - notOk);
    // Coverage can be incomplete on a non-successful run; equal aggregate counts
    // cannot erase that terminal fact. Legacy frames without states stay count-based.
    const terminalFailure = Array.isArray(evidence.delegated_run_failure_states)
        && evidence.delegated_run_failure_states.some(state => typeof state === 'string' && state && state !== 'source_incomplete');
    const notOkWord = unresolved && (terminalFailure || unresolved < notOk) ? 'failed or unverified'
        : unresolved ? 'unverified' : 'failed';
    const okPart = `${ok} ok${notOk ? `, ${notOk} ${notOkWord}` : ''}`;
    const unsettled = Math.max(0, started - settled);
    const phrase = notOk ? (settled === 1 ? `run ${notOkWord}` : `${notOk} of ${settled} runs ${notOkWord}`)
        : unsettled ? `${unsettled} of ${started} runs not settled` : (counted ? '' : 'run outcome unknown');
    const accessClauses = [];
    const surface = String(evt?.write_surface || '').trim();
    if (surface) accessClauses.push(`write surface ${surface}`);
    const applied = Array.isArray(evidence.applied_access_profiles)
        ? evidence.applied_access_profiles.filter(Boolean) : [];
    if (applied.length) accessClauses.push(`access applied: ${applied.join(', ')}`);
    const accessPart = accessClauses.length ? ` — ${accessClauses.join('; ')}` : '';
    const unresolvedTitle = unresolved
        ? ` — ${unresolved} run(s) settled with unverified work-order coverage (not counted ok)`
        : '';
    const unsettledTitle = unsettled ? ` — ${unsettled} more run(s) started, not settled` : '';
    const models = Array.isArray(evidence.harness_models)
        ? [...new Set(evidence.harness_models.filter(model => typeof model === 'string' && model.trim()))] : [];
    // The union names settled runs, not which one answered: only an unambiguous single-run success owns the line.
    const successes = evidence.delegated_runs_succeeded, failures = evidence.delegated_runs_failed;
    const singleSuccess = started === 1 && settled === 1 && notOk === 0 && !evidence.delegated_run_failure_states?.length
        && (successes === 1 || (successes == null && failures === 0)) && (failures == null || failures === 0);
    const settledRunModel = singleSuccess && models.length === 1 ? models[0] : '';
    const modelPart = models.length ? ` — ${settledRunModel ? 'successful run model' : 'models observed across settled runs'}: ${models.join(', ')}` : '';
    return receipt({ ...chip(true, phrase, withSubstrate(`Delegated to your ${name} account — ${runsPart} settled${counted ? ` (${okPart})` : ''}, ${costPart}`)
        + unresolvedTitle + unsettledTitle + accessPart + modelPart), settledRunModel });
}

// Sticky-card precedence (adversarial wave B-ADV-2): an evidence-bearing (receipt) chip is never downgraded by a later
// evidence-less (dispatch) frame — the history sync after justFinished anchors on a mid-run row.
export function keepStickyExecutorChip(prior, next) {
    if (!prior || !next) return false;
    if (prior.sourceTs && next.sourceTs && next.sourceTs < prior.sourceTs) return true;
    // A run's report speaks for its own attempt alone: another known attempt's frame decides by attempt order.
    const before = prior.observation, after = next.observation, attemptOf = (chip, report) => taskAttempt(report?.task_attempt) ?? taskAttempt(chip.attempt);
    const was = attemptOf(prior, before), now = attemptOf(next, after);
    if (before && was !== null && now !== null && was !== now) return now < was;
    if (before && after) return before.task_id === after.task_id && before.task_attempt === after.task_attempt
        && before.run_id === after.run_id && after.revision < before.revision;
    // A typed live observation outranks any later evidence-less dispatch label, even when the route name changes while
    // the same card is reconciled; a terminal record outranks it, and no later non-terminal frame outranks that record.
    return !next.hasEvidence && Boolean(prior.hasEvidence || (before || prior.ended) && !next.ended);
}

/** The executor a delegation selected: `requested_executor_route`, frozen with the child's row at scheduling, else
 *  the model an older frame's full dispatch route spells (`codex=gpt-5`); a bare legacy harness names none. It is a
 *  request, never evidence: {kind: api_model|agent_session, harness, model}, or null. */
export function requestedExecutor(evt) {
    const declared = selectedRoute(evt?.requested_executor_route), target = declared?.target_id || '';
    if (declared?.kind === 'api_model') return { kind: 'api_model', harness: '', model: target };
    const route = declared ? target : String(evt?.executor_route || '').trim();
    const harness = routeHarness(route);
    if (!harness || (route !== target && !route.includes('='))) return null;
    return { kind: 'agent_session', harness, model: route.split('=').slice(1).join('=').split(':')[0].trim() };
}

/** The selection as the card's Details state it when its compact line does not already (`Codex · gpt-5.5`). */
export function requestedFact(selected, chip, lineModel, execution) {
    if (!selected) return '';
    const api = selected.kind === 'api_model';
    const target = api ? displayApiModelTarget(selected.model) : selected.model;
    const harness = selected.kind === 'agent_session' ? selected.harness : '';
    const asked = selectedModel(selected, harness);
    const sameExecutor = harness ? String(chip?.harness || '').toLowerCase() === harness : !chip;
    const bare = (text) => String(text || '').replace(/^requested /, '').replace(/ \(local\)$/i, '');
    const solve = api && execution?.source === 'usable_solve_response' ? execution : null;
    const switchedLane = typeof solve?.used_local === 'boolean' && solve.used_local !== target.endsWith(' (local)');
    const update = chip?.observation;
    const model = chip?.settledRunModel || (['observed', 'requested'].includes(update?.model_source) && update.model)
        || solve?.reported_model || solve?.used_model || lineModel;
    // A display abbreviation is not a new model, but two recorded qualified identities may have the same short name.
    // A solve's route is its provider and model (routeIdentity) however it is spelled: two spellings of one stay one
    // identity, another provider or vendor does not; a reported short label alone proves no other route. The chosen and
    // used routes keep the dispatch grammar; only the provider's own id (`reported_model`) is read in the provider that
    // served it, never as a bare (OpenRouter) route of its own.
    const choice = bare(target), qualified = /\/|::/, identity = (id) => (api ? routeIdentity(bare(id)) : bare(id));
    const served = solve?.reported_model && model === solve.reported_model && solve.used_model && !String(model).includes('::')
        ? routeIdentity(bare(solve.used_model)).split('::')[0] : '';
    const named = served ? routeIdentity(`${served}::${bare(model)}`) : identity(model);
    const differs = qualified.test(choice) && qualified.test(bare(model)) && named !== identity(choice)
        || Boolean(solve?.used_model) && identity(solve.used_model) !== identity(choice);
    const sameName = bare(compactModel(lineModel)) === bare(asked);
    if (sameExecutor && (!asked || (!switchedLane && !differs && sameName))) return '';
    return joinMetaParts([harness && harnessPresentation(harness).label, differs && sameName ? target : asked]);
}

/** A parent-addressed admission refusal (`subagent_admission_notice`) is a row of its parent naming its own refused
 *  request: gateway/history.py keeps it a separate request, so none of its lineage or selection reaches a card's
 *  identity, and an admitted child under the same id stays as it is. */
export function admissionRefusalRow(msg) {
    const selected = requestedExecutor(msg), content = String(msg.content || msg.text || '');
    const asked = selected && joinMetaParts([selected.harness && harnessPresentation(selected.harness).label, selected.model]);
    return { task_id: msg.task_id, ts: msg.ts, chat_id: msg.chat_id, role: msg.role, system_type: msg.system_type, narration: msg.narration,
        is_progress: true, content: asked ? `${content}${META_PART_SEPARATOR}${fmt('requested {model}', { model: asked })}` : content };
}

/** Whether a run's last report (of attempt `report`) speaks for an end of attempt `end`: a known attempt on both sides
 *  decides. A card that knows its latest attempt (task_phase_chip's `childAttempt`, even from its first numbered census)
 *  dates an end that names none to it unless that one was interrupted, so a numbered report of an earlier attempt stays
 *  out; a card that crossed an attempt boundary (`childRetried`, `interruptedAttempt`) also keeps out an unnumbered
 *  report it cannot place after the boundary; with no boundary crossed, such a report stays. */
function reportOfEnd(report, end, { childRetried, childAttempt, interruptedAttempt } = {}) {
    const latest = taskAttempt(childAttempt), interrupted = taskAttempt(interruptedAttempt);
    const at = end ?? (latest !== null && !(interrupted !== null && interrupted >= latest) ? latest : null);
    if (at !== null && report !== null) return at === report;
    return !childRetried || (report !== null && interrupted !== null && report > interrupted);
}

/** The chip a card keeps after a frame whose chip it adopts (`next`, else none: keepStickyExecutorChip decides), from
 *  projected facts only. An API-model selection runs on no harness, so its card has none: its own model and solve
 *  response name it, and a harness it calls is its tool. A terminal record that found no dispatch reads as the dispatched
 *  route's end (`dispatchedEnd`) once the card saw a route, a start or a run's report, and a terminal chip keeps the
 *  run's last progress report of its own attempt (reportOfEnd) in its detail sentence as history, never on the line. */
export function cardExecutorChip(card, next) {
    const { executorChip: prior, requestedExecutor: selected } = card;
    if (selected?.kind === 'api_model') return null;
    if (!next) return placedReport(card, prior) || null;
    const chip = (prior?.dispatched || prior?.observation) && next.dispatchedEnd || next;
    const seen = prior?.observation ? prior.title : prior?.lastUpdate;
    const of = prior?.observation ? taskAttempt(prior.observation.task_attempt) ?? taskAttempt(prior.attempt) : taskAttempt(prior?.lastUpdateAttempt);
    return placedReport(card, chip.ended && seen && !chip.lastUpdate && reportOfEnd(of, taskAttempt(chip.attempt), card)
        ? { ...chip, title: `${chip.title}. ${seen}`, lastUpdate: seen, ...(of !== null && { lastUpdateAttempt: of }) } : chip);
}

// A live report the card cannot place at its latest attempt stops naming the executor's model: the line returns to the
// frozen selection until the new attempt reports, and the report stays history. A numbered report is compared with the
// latest known attempt whichever arrived first: an interrupted or older attempt's is out, the latest's or a newer one's
// (its census lags) stays on the line; an unnumbered one follows reportOfEnd. Live only: a terminal that names no attempt
// still takes a report as its history only through reportOfEnd (cardExecutorChip), never across a known retry unmatched.
function placedReport(card, chip) {
    const report = chip?.observation, of = taskAttempt(report?.task_attempt) ?? taskAttempt(chip?.attempt);
    const latest = taskAttempt(card.childAttempt), interrupted = taskAttempt(card.interruptedAttempt);
    const live = of === null ? reportOfEnd(of, null, card)
        : !(interrupted !== null && of <= interrupted) && (latest === null ? reportOfEnd(of, null, card) : of >= latest);
    if (!report || live) return chip;
    const { observation, ...rest } = chip, name = harnessPresentation(chip.harness).label;
    const asked = selectedModel(card.requestedExecutor, chip.harness);
    return { ...rest, label: name, dispatched: true, lastUpdate: chip.title, ...(of !== null && { lastUpdateAttempt: of }),
        title: `Selected ${name}${asked ? `, model ${asked} (requested)` : ''} — run evidence arrives with the terminal receipt` };
}

/** The facts of a card's Coordinator (else Agent model) row: a session child's recorded coordinator (recordedCoordinator,
 *  kept past later chips that carry none) and its provenance, else the card's own agent model (`fallback`). */
export function cardCoordinator(record, chip, session, fallback = '') {
    const recorded = session ? { ...chip.coordinator, ...record.coordinator } : {};
    const provenance = session ? formatCoordinatorProvenance(recorded.provenance || chip?.coordinatorProvenance || chip?.snapshot || '') : '';
    return { route: recorded.route || fallback, billing: recorded.billing || '', provenance };
}

/** The model the executor itself ran, as far as its evidence says: the single successful
 *  settled run's model, else the run's own progress
 *  report with its source, else the model the delegation selected for this same executor
 *  (`selectedModel`), which says `requested`; an unconfirmed one is absent (the detail
 *  sentence says `model unconfirmed`). A null chip is an executor without a harness. On a
 *  `settled` card a progress report's observed model is history (its Executor details), never
 *  the answer; a harness reports no lane, so its model's words never mark one. */
export function executorModelFact(chip, selected = null, settled = false) {
    if (chip?.settledRunModel) return { model: chip.settledRunModel };
    const observation = chip?.observation;
    const model = compactModel(observation?.model, false);
    if (model && observation.model_source === 'observed' && !settled) return { model };
    if (model && observation.model_source === 'requested') return { model, requested: true };
    const asked = selectedModel(selected, chip?.harness);
    return asked ? { model: asked, requested: true } : { model: '' };
}

/** A model fact in the record's English words (`requested …` for a request), as Details compare it. */
export const modelText = ({ model = '', requested = false } = {}) => (requested ? `requested ${model}` : model);
export const executorModelText = (chip, selected = null, settled = false) => modelText(executorModelFact(chip, selected, settled));
