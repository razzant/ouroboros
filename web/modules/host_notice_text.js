// Display-only form of a host sentence that its producer opens with a decorative
// pictograph (DESIGN "Chat authorship and System rows"). The stored row keeps its bytes:
// history, message keys, Copy, Telegram, the model's context and outcomes.py, which
// reads their warning marks, see exactly what the producer wrote.
//
// A closed table, keyed by the row's typed `system_type`; each entry is one producer's
// exact source prefix. Only a row whose type and opening both match loses the leading
// mark and its space. Every other row renders as stored: another type, an untyped or
// pre-typing row, authored words (reminder, skill notice, Project mirrors), a prefix
// not listed here. Future producer changes must keep this display mapping in sync.
import { tr } from './i18n.js';

export const HOST_NOTICE_PREFIXES = new Map(Object.entries({
    startup_notice: ['♻️ Restored pending queue: ', '♻️ Settling '],
    command_reply: ['⚠️ Command ignored: ', '✅ Owner chat registered.', '♻️ Restarting.', '⚠️ Restart cancelled: ',
        '🧬 Evolution campaign: ', '🧬 Evolution stayed OFF: ', '🧬 Evolution did not turn on: ',
        '🧠 Background consciousness ', '🧠 Background consciousness:'],
    restart_notice: ['♻️ Restart requested by agent: ', '⏳ Restart drain: ', '🧬 Restart cancelled: ',
        '⚠️ Restart skipped: ', '🧬 Body adoption: '],
    consciousness_notice: ['🧠 Background consciousness '],
    evolution_notice: ['🧬⚠️ Evolution paused: ', '🧬 Evolution stayed OFF: ', '🧬 Evolution stayed off: ', '🧬 Evolution paused: ',
        '🧬 Evolution: ', '🧬 Evolution campaigns are self-modification work', '💸 Evolution stopped: ',
        '🛑 Stopped evolution task(s): ', '⚠️ Evolution stop INCOMPLETE — still live: '],
    evolution_cycle_outcome: ['🧬 Evolution cycle absorbed ', '⚠️ Evolution cycle abandoned '],
    promotion_notice: ['❌ Evolution promotion refused: ', '❌ Failed to promote to stable: ', '✅ Promoted: '],
    managed_update_notice: ['📦 Managed update: ', '🔒 An update is using the repository.'],
    supervisor_failure: ['🛑 Supervisor loop died after repeated crashes; '],
    retired_settings_notice: ['⚙️ Settings: '],
    review_pool_migration_notice: ['⚙️ Review pool: ', '⚙️ Review pool initialized. ', '⚙️ Review settings migrated. ',
        '⚙️ Review settings could not be migrated automatically: '],
    optional_bounds_notice: ['⚙️ Task limits on this install: '],
    legacy_memory_notice: ['🧠 Memory: '],
    task_admission_notice: ['🚫 Budget exhausted. '],
    budget_notice: ['🚫 Model budget reached. '],
    deep_self_review_queued: ['🔎 Deep self-review queued: '],
    runtime_liveness_notice: ['⚠️ A chat turn looks wedged '],
    initialization_notice: ['⚠️ System is still initializing.', '⚠️ The data store is still being prepared ',
        '⚠️ Startup failed before the chat could accept messages', '⚠️ The chat is not accepting messages right now',
        '⚠️ Your message was not saved: '],
    terminal_incident: ['🔌 Task ', '⚠️ Sending a message for task '],
    task_reaper_notice: ['⚠️ A timed-out worker (task '],
    worker_failure: ['⚠️ Frequent worker crashes.', '❌ worker process crashed ',
        '❌ Deep self-review failed: worker process crashed ', '❌ Task failed after ', '❌ Task `'],
    worker_readiness_notice: ['⚠️ Worker checkout sampled after initialization ', '⚠️ Worker slot '],
    // Progress rows: a task card's timeline line (`log_events.summarizeChatLiveEvent`).
    task_scheduled: ['🗓️ Scheduled subagent ', '🗓️ Scheduled task '],
    task_error: ['⚠️ Error: '],
    finalization_notice: ['⏳ Task ', '▶️ Task '],
    skill_review: ['✅ Skill review round ', '⚠️ Skill review round ', '❌ Skill review round ', '⏳ Skill review round '],
    host_progress: ['🌐 Connection restored ', '🌐 The new attempt could not reach the provider ',
        '🌐 Provider connection ', '🌐 The primary provider temporarily refused ',
        '🌐 The request did not reach the provider ', '🌐 Primary provider answered after ',
        '🌐 Stopped waiting for ', '🌐 Still waiting for ', '⚡ Fallback: ', '📐 Plan: ', '✅ PR #', '⚠️ PR #',
        // web_search legs (`tools/search.py`) and a declared-output baseline (`tools/shell_outputs.py`).
        '🔍 Trying ', '↪ Trying ', '🔍 Searching: ', '🔍 Answer from ', '⚠️ Web search ', '⚠️ Output baseline unavailable: '],
}));

// Producers this install no longer runs: rows they already wrote keep losing only the mark at
// display (host_notice_text.test.js pins that none of them is written again).
export const RETIRED_HOST_NOTICE_PREFIXES = new Map(Object.entries({
    startup_notice: ['♻️ Restored pending queue from snapshot: ', '♻️ Cancelling '],
    attachment_notice: ['📎 Attachment staging report for ', '⚠️ Task not started: every attachment was rejected.',
        '⚠️ Some declared attachments could not be staged; '],
    cancellation_notice: ['❌ cancel '],
    task_admission_notice: ['✅ Repair task ', '⚠️ Cost accounting is unavailable. '],
    task_started: ['🧬 Evolution task ', '🔎 Review task '],
    evolution_notice: ['🛑 Cancelled evolution task(s): '],
    host_progress: ['🌐 Could not establish a provider connection ', '📐 Plan review:', '📐 Plan review so far:'],
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

// Authored narration outranks a stale host type, on both bubble and timeline paths.
export function hostRowDisplayText(row, shown = row.text ?? row.content ?? '') {
    if (row.role !== 'system' || row.narration === true) return shown;
    let text = String(row.text ?? row.content ?? '');
    // A progress envelope is not authored content. Remove it only together with
    // a recognized host opening; unknown and authored rows keep every character.
    if (row.is_progress === true && text.startsWith('💬 ') && prefixFor(text.slice('💬 '.length), row.system_type)) {
        text = text.slice('💬 '.length);
        if (String(shown).startsWith('💬 ')) shown = String(shown).slice('💬 '.length);
    }
    return hostNoticeDisplayText(text, row.system_type, shown);
}
