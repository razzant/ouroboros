/* Every settings row, as data: no functions, no layout, no imports (it survives JSON.parse(JSON.stringify(ROWS))).
   A row has one home (`page`: a Settings path or a door id such as 'dashboard:costs') and is drawn and written by
   settings_controls.js (settingsRow, commitSetting); `writer`, `codec`, `confirm` and `hiddenWhen` name entries of
   its registries. A row that needs a function is hand-written in its page and listed in HAND_WRITTEN by id and keys.
   Append a page's rows as one block under its own comment. */

export const ROWS = [
    // Dashboard › Costs: the limits card, then its fold.
    { id: 'TOTAL_BUDGET', keys: ['TOTAL_BUDGET'], page: 'dashboard:costs', fold: false,
        name: 'Total budget', meta: 'Paid work stops when known spend reaches it.', more: '',
        control: { kind: 'number', unit: '$', from: 'setup:totalBudget' },
        aliases: ['Total Budget (USD)', 'Total budget ($)', 'spending limit'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_PER_TASK_COST_USD', keys: ['OUROBOROS_PER_TASK_COST_USD'], page: 'dashboard:costs', fold: false,
        name: 'Per-task cost cap', meta: 'A task and its helpers stop when their known spend reaches it.', more: '',
        control: { kind: 'number', unit: '$', from: 'setup:perTaskCostUsd' },
        aliases: ['Per-Task Cost Cap (USD)', 'Per-task cost cap ($)', 'one task'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_CONSCIOUSNESS_DAILY_USD', keys: ['OUROBOROS_CONSCIOUSNESS_DAILY_USD'], page: 'dashboard:costs', fold: false,
        name: 'Consciousness per 24 hours',
        meta: 'Its wake-ups, replies in places and the work they start, over a rolling 24 hours.', more: '',
        control: { kind: 'number', unit: '$', min: 0 },
        aliases: ['Consciousness per 24 hours ($)', 'Spending limit over 24 hours', 'allowance'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_MAX_ROUNDS', keys: ['OUROBOROS_MAX_ROUNDS'], page: 'dashboard:costs', fold: true,
        name: 'Max rounds per task', meta: 'Model rounds one task may take.', more: '',
        control: { kind: 'number', noLimit: true },
        aliases: ['Max Rounds per Task', 'round limit'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_TASK_ABS_CEILING_SEC', keys: ['OUROBOROS_TASK_ABS_CEILING_SEC'], page: 'dashboard:costs', fold: true,
        name: 'Task lifetime limit', meta: 'Seconds one task may run.', more: '',
        control: { kind: 'number', unit: 's', noLimit: true },
        aliases: ['Task Lifetime Limit (s)', 'longest task'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },

    // Dashboard › Consciousness: what it does on its own, its two doors, then its fold. The strip's labels are
    // AUTONOMY_LEVELS (consciousness_autonomy.js); a test keeps the two equal.
    { id: 'OUROBOROS_CONSCIOUSNESS_AUTONOMY', keys: ['OUROBOROS_CONSCIOUSNESS_AUTONOMY'], page: 'dashboard:consciousness', fold: false,
        name: 'How far it goes', meta: 'What it may do when nobody asked.',
        more: 'Advises: On its own it studies, keeps its memory and tells you; it changes nothing outside. '
            + 'Acts: Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself. '
            + 'Evolves: Also changes itself: its own code, its settings, connecting new places.',
        control: { kind: 'strip', options: [{ value: 'observe', label: 'Advises' }, { value: 'act', label: 'Acts' },
            { value: 'full', label: 'Evolves' }] },
        aliases: ['Autonomy', 'Consciousness autonomy'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_BG_WAKEUP', keys: ['OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX'], page: 'dashboard:consciousness', fold: false,
        name: 'Wakes every', meta: 'It picks a moment inside this range.', more: '',
        control: { kind: 'pair', unit: 's', min: 60, labels: ['Shortest wait, in seconds', 'Longest wait, in seconds'] },
        aliases: ['Shortest wake interval (s)', 'Longest wake interval (s)', 'wake interval'],
        writer: 'wakeRange', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', keys: ['OUROBOROS_CONSCIOUSNESS_MAX_TASKS'], page: 'dashboard:consciousness', fold: false,
        name: 'Own tasks at once', meta: 'Zero: it thinks and writes, and starts no tasks.', more: 'Work you asked for never counts.',
        control: { kind: 'number', min: 0, max: 32 },
        aliases: ['Max concurrent tasks', 'its own tasks'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'consciousness-allowance', keys: [], page: 'dashboard:consciousness', fold: false,
        name: 'Daily allowance', meta: 'With the other limits, in Costs.', more: '',
        control: { kind: 'door' },
        aliases: [],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'consciousness-model', keys: [], page: 'dashboard:consciousness', fold: false,
        name: 'Thinks with', meta: 'Its model, chosen in Models.', more: '',
        control: { kind: 'door' },
        aliases: [],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_PRESENCE_MAX_ACTIVE', keys: ['OUROBOROS_PRESENCE_MAX_ACTIVE'], page: 'dashboard:consciousness', fold: true,
        name: 'Concurrent conversations', meta: 'Places it talks in at the same time. Others wait their turn.', more: '',
        control: { kind: 'number', min: 1, max: 20 },
        aliases: ['conversations at once'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },

    // Dashboard › Evolution: whether a cycle may follow your tasks (one choice over two keys), its floor and its
    // standing direction.
    { id: 'OUROBOROS_POST_TASK_EVOLUTION', keys: ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE'],
        page: 'dashboard:evolution', fold: false,
        name: 'After finishing your tasks', meta: 'A cycle it chooses runs later, once it is idle.',
        more: 'After each task, it asks itself whether one improvement is worth a cycle. Every N tasks, it asks after '
            + 'every Nth task. Every finished task counts, chats included.',
        control: { kind: 'custom', options: [{ value: 'off', label: 'Off' }, { value: 'llm', label: 'After each task' },
            { value: 'every_n', label: 'Every N tasks' }] },
        aliases: ['Post-Task Self-Evolution', 'Every N Tasks', 'evolution cadence'],
        writer: 'evolutionWhen', codec: 'evolutionWhen', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD', keys: ['OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD'], page: 'dashboard:evolution', fold: false,
        name: 'Start only above', meta: 'Budget that must be left before a cycle starts.',
        more: '0 leaves it to the other limits. A running cycle still keeps to the per-task cap and the total budget.',
        control: { kind: 'number', unit: '$', min: 0 },
        aliases: ['Post-task Start Floor (USD)', 'start floor'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE', keys: ['OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE'], page: 'dashboard:evolution', fold: false,
        name: 'What to work on', meta: 'A standing direction added to every cycle. Empty leaves the choice to Ouroboros.', more: '',
        control: { kind: 'text' },
        aliases: ['Standing Objective (optional)', 'standing objective'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },

    // Dashboard › Updates: which official branch Update follows. A deployment that updates the installation itself
    // leaves the row out.
    { id: 'OUROBOROS_UPDATE_CHANNEL', keys: ['OUROBOROS_UPDATE_CHANNEL'], page: 'dashboard:updates', fold: false,
        name: 'Channel', meta: 'Which official branch Update follows.',
        more: 'Stable follows released code on main, the default. QA follows ouroboros-stable. Development follows '
            + 'ouroboros. Your own work branch stays ouroboros.',
        control: { kind: 'strip', options: [{ value: 'stable', label: 'Stable' }, { value: 'qa', label: 'QA' },
            { value: 'development', label: 'Development' }] },
        aliases: ['Update Channel', 'release channel'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: 'managedDeployment' },

    // Dashboard › Storage: how long finished work is kept, in its Cleanup part.
    { id: 'OUROBOROS_GC_RETENTION_DAYS', keys: ['OUROBOROS_GC_RETENTION_DAYS'], page: 'dashboard:storage', fold: false,
        name: 'Keep finished work for', meta: 'Older working copies and task files are removed. Project folders stay.',
        more: 'It covers the working copies agents used, the folders of finished tasks and leftover service logs.',
        control: { kind: 'number', unit: 'days', min: 1, max: 365 },
        aliases: ['GC Retention (days)', 'retention'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
];

/** Rows drawn by their page's own code: `{ id, keys, page }`, read by search and by the "one key, one control" test. */
export const HAND_WRITTEN = [];

/** A page's rows in order; `fold` true or false keeps only the rows behind (or before) the page's fold. */
export function rowsOf(page, { fold } = {}) {
    return ROWS.filter((row) => row.page === page && (fold === undefined || row.fold === fold));
}

export function rowById(id) {
    return ROWS.find((row) => row.id === id) || null;
}
