// How the window says a host sentence (DESIGN principle 7 and "Chat authorship and System rows").
//
// A host notice (ouroboros/host_notice.py) is drawn as it is: the one line its server assembled, the phrase and its
// next step in the install's language, which Telegram and Copy say too (`noticeLine`); its raw details sit behind
// Details. A notice has no pictograph to hide.
//
// Rows written before notices keep the display-only table below: a producer opened its sentence with a decorative
// pictograph, and the stored row keeps its bytes (history, message keys, Copy, Telegram, the model's context). A
// closed table, keyed by the row's typed `system_type`; each entry is one producer's exact source prefix. Only a row
// whose type and opening both match loses the leading mark and its space. Every other row renders as stored: another
// type, an untyped or pre-typing row, authored words (reminder, skill notice, Project mirrors), a prefix not listed
// here. A producer that moves to a notice moves its prefixes to RETIRED_HOST_NOTICE_PREFIXES for the rows it wrote.
import { fmt, tr, tx } from './i18n.js';

export const HOST_NOTICE_PREFIXES = new Map(Object.entries({
    startup_notice: ['♻️ Restored pending queue: ', '♻️ Settling ', '♻️ The server restarted itself after ',
        '♻️ Ouroboros did not start its recovery turn '],
    command_reply: ['⚠️ Command ignored: ', '✅ Owner chat registered.', '♻️ Restarting.', '⚠️ Restart cancelled: ',
        '🧬 Evolution campaign: ', '🧬 Evolution stayed OFF: ', '🧬 Evolution did not turn on: ',
        '🧠 Background consciousness ', '🧠 Consciousness off: ', '🧠 Consciousness on: ', '🧠 Consciousness on, but ',
        '🧠 Consciousness is already ', '🧠 Consciousness: '],
    restart_notice: ['♻️ Restart requested by agent: ', '⏳ Restart drain: ', '🧬 Restart cancelled: ',
        '⚠️ Restart skipped: '],
    consciousness_notice: ['🧠 Background consciousness ', '🧠 Consciousness off: ', '🧠 Consciousness on: ', '🧠 Consciousness on, but '],
    evolution_notice: ['🧬 Evolution stayed OFF: ', '🧬 Evolution stayed off: ',
        '🧬 Evolution: ', '🧬 Evolution campaigns are self-modification work',
        '🛑 Stopped evolution task(s): ', '⚠️ Evolution stop INCOMPLETE — still live: '],
    promotion_notice: ['❌ Evolution promotion refused: '],
    managed_update_notice: ['🔒 An update is using the repository.'],
    supervisor_failure: ['🛑 The event-bus process of this server exited', '🛑 The task loop failed three times in a row'],
    retired_settings_notice: ['⚙️ Settings: '],
    review_pool_migration_notice: ['⚙️ Review pool: ', '⚙️ Review pool initialized. ', '⚙️ Review settings migrated. ',
        '⚙️ Review settings could not be migrated automatically: '],
    optional_bounds_notice: ['⚙️ Task limits on this install: '],
    legacy_memory_notice: ['🧠 Memory: '],
    command_timeout_notice: ['⏱️ The separate per-call ceiling'],
    review_enforcement_notice: ['⚙️ This install had reviews set to Blocking.'],
    model_cap_notice: ['⚙️ "Requests at once per model" held 3 '],
    subagent_cap_notice: ['⚙️ "At once, for one task" held 6 '],
    light_local_box_notice: ['⚙️ An empty Light model now follows its “Use the local model for this role” box everywhere.'],
    budget_notice: ['🚫 Model budget reached. '],
    runtime_liveness_notice: ['⚠️ A chat turn looks wedged '],
    initialization_notice: ['⚠️ System is still initializing.', '⚠️ The data store is still being prepared ',
        '⚠️ Startup failed before the chat could accept messages', '⚠️ The chat is not accepting messages right now',
        '⚠️ Your message was not saved: '],
    worker_failure: ['❌ Task failed after '],
    // Progress rows: a task card's timeline line (`log_events.summarizeChatLiveEvent`).
    task_scheduled: ['🗓️ Scheduled subagent ', '🗓️ Scheduled task '],
    skill_review: ['✅ Skill review round ', '⚠️ Skill review round ', '❌ Skill review round ', '⏳ Skill review round '],
    host_progress: ['🌐 Connection restored ', '🌐 The new attempt could not reach the provider ',
        '🌐 Provider connection ', '🌐 The primary provider temporarily refused ', '🌐 No local model slot was free',
        '🌐 The local Claudexor engine ', '🌐 The request did not reach the provider ', '🌐 Primary provider answered after ',
        '🌐 Model response received after ', '🌐 Stopped waiting for ', '🌐 Still waiting for ', '⚡ Fallback: ',
        '📐 Plan: ', '✅ PR #', '⚠️ PR #',
        // web_search legs (`tools/search.py`) and a declared-output baseline (`tools/shell_outputs.py`).
        '🔍 Trying ', '↪ Trying ', '🔍 Searching: ', '🔍 Answer from ', '⚠️ Web search ', '⚠️ Output baseline unavailable: '],
}));

// Producers this install no longer runs: rows they already wrote keep losing only the mark at
// display (host_notice_text.test.js pins that none of them is written again).
export const RETIRED_HOST_NOTICE_PREFIXES = new Map(Object.entries({
    startup_notice: ['♻️ Restored pending queue from snapshot: ', '♻️ Cancelling '],
    // The /bg status reply before the switch's words (owner 15.11, 5A) said «🧠 Background consciousness: …».
    command_reply: ['🧠 Background consciousness:'],
    supervisor_failure: ['🛑 Supervisor loop died after repeated crashes; '],
    worker_failure: ['⚠️ Frequent worker crashes.', '❌ worker process crashed ',
        '❌ Deep self-review failed: worker process crashed ', '❌ Task `'],
    // Producers that now write a host notice (ouroboros/host_notice.py): a phrase without a mark, its raw facts behind Details.
    restart_notice: ['🧬 Body adoption: '],
    evolution_cycle_outcome: ['🧬 Evolution cycle absorbed ', '⚠️ Evolution cycle abandoned '],
    promotion_notice: ['❌ Failed to promote to stable: ', '✅ Promoted: '],
    managed_update_notice: ['📦 Managed update: '],
    deep_self_review_queued: ['🔎 Deep self-review queued: '],
    terminal_incident: ['🔌 Task ', '⚠️ Sending a message for task '],
    task_reaper_notice: ['⚠️ A timed-out worker (task '],
    worker_readiness_notice: ['⚠️ Worker checkout sampled after initialization ', '⚠️ Worker slot '],
    task_error: ['⚠️ Error: '],
    finalization_notice: ['⏳ Task ', '▶️ Task '],
    attachment_notice: ['📎 Attachment staging report for ', '⚠️ Task not started: every attachment was rejected.',
        '⚠️ Some declared attachments could not be staged; '],
    cancellation_notice: ['❌ cancel '],
    task_admission_notice: ['✅ Repair task ', '⚠️ Cost accounting is unavailable. ', '🚫 Budget exhausted. '],
    task_started: ['🧬 Evolution task ', '🔎 Review task '],
    evolution_notice: ['🛑 Cancelled evolution task(s): ', '🧬⚠️ Evolution paused: ', '💸 Evolution stopped: ',
        '🧬 Evolution paused: '],
    host_progress: ['🌐 Could not establish a provider connection ', '📐 Plan review:', '📐 Plan review so far:',
        '🌐 The local Claudexor engine answered again after '],
}));

const prefixFor = (text, type) => [...(HOST_NOTICE_PREFIXES.get(String(type || '')) || []),
    ...(RETIRED_HOST_NOTICE_PREFIXES.get(String(type || '')) || [])].find(entry => text.startsWith(entry));

// `text` is the row as stored and decides the match; `shown` is what the caller would
// otherwise paint (the translated sentence for a `tx` type), which loses the same mark.
export function hostNoticeDisplayText(text, systemType, shown = text) {
    const source = String(text ?? '');
    const display = String(shown ?? '');
    const prefix = prefixFor(source, systemType);
    if (!prefix) return display;
    const mark = prefix.slice(0, prefix.indexOf(' ') + 1);
    // The receipt's leading mark also carries severity; keep that meaning in words.
    if (prefix === '⚠️ PR #' && display.startsWith(mark)) return `${tr('task.host.warning', 'Warning')}: ${display.slice(mark.length)}`;
    return display.startsWith(mark) ? display.slice(mark.length) : display;
}

// A tool call the host typed as failed (`is_error`) opens its result with its warning mark before the error's code
// (loop_tool_execution.py "⚠️ TOOL_ERROR (tool): …", mcp_client.py "⚠️ MCP_…"): the failure is said by its chip, so
// the mark is hidden at display, as a known host prefix is; the record keeps it.
const TOOL_ERROR_MARK = '⚠️ ';
export const toolErrorDisplayText = (evt, text) => (evt?.is_error === true && String(text ?? '').startsWith(TOOL_ERROR_MARK)
    ? String(text).slice(TOOL_ERROR_MARK.length) : text);

// A pause notice's heads before the words its press recorded, which run up to the notice's retained clause
// (ouroboros/pause_notices.py notice_text; web/tests/fixtures/pause_notice_parity.json pins both sides). The words
// are their author's: never translated (DESIGN "Language").
const PAUSE_HEADS = ['You paused this task: ', 'Ouroboros paused this task: '];
const PAUSE_RETAINED = ' Its work is retained; Resume continues it.';

/** A relayed host sentence in the install's language: by its exact text (`tx`), except a pause notice's recorded words,
 *  filled unchanged into its translated frame (`fmt`). */
export function hostSentenceText(text) {
    const source = String(text ?? '');
    const head = PAUSE_HEADS.find((entry) => source.startsWith(entry));
    const end = head ? source.lastIndexOf(PAUSE_RETAINED) : -1;
    if (!head || end <= head.length) return tx(source);
    return fmt(`${head}{words}${source.slice(end)}`, { words: source.slice(head.length, end) });
}

/** A task-mailbox message the host wrote (a silence fact, a control receipt): system provenance and no sending
 *  task, so its row says who wrote it instead of «Message from task unknown». */
export const hostTaskMessage = (evt) => evt?.provenance === 'system' && (!evt.source_task_id || evt.source_task_id === 'unknown');

/** A host notice's line, the one its server assembled for every reader (`notice.line`); '' for any other row. */
export const noticeLine = (row) => (row?.role === 'system' && row.narration !== true
    && typeof row.notice?.line === 'string' ? row.notice.line : '');
// A notice's raw details open from its card line as plain text (`fullBody`), never in a toast.
export const noticeBody = (row) => (noticeLine(row) && row.notice.details ? { fullBody: String(row.notice.details), bodyFormat: 'text' } : {});

// Authored narration outranks a stale host type, on both bubble and timeline paths.
export function hostRowDisplayText(row, shown = row.text ?? row.content ?? '') {
    if (row.role !== 'system' || row.narration === true) return shown;
    if (noticeLine(row)) return noticeLine(row);
    let text = String(row.text ?? row.content ?? '');
    // A progress envelope is not authored content. Remove it only together with
    // a recognized host opening; unknown and authored rows keep every character.
    if (row.is_progress === true && text.startsWith('💬 ') && prefixFor(text.slice('💬 '.length), row.system_type)) {
        text = text.slice('💬 '.length);
        if (String(shown).startsWith('💬 ')) shown = String(shown).slice('💬 '.length);
    }
    return hostNoticeDisplayText(text, row.system_type, shown);
}
