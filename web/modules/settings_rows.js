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

    // Settings › Agents: the roster (its page draws the switch and the list of agents and writes them), then the fold.
    { id: 'OUROBOROS_SUBAGENTS', keys: ['OUROBOROS_SUBAGENTS'], page: 'agents', fold: false,
        name: 'Hand work to helpers', meta: 'Ouroboros decides when. Off stops it in every room.', more: '',
        control: { kind: 'custom' },
        aliases: ['Delegation', 'Available subagents', 'Subagents', 'Reviewers', 'Agents list'],
        writer: 'checker', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS', keys: ['OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS'], page: 'agents', fold: true,
        name: 'Helpers may change files', meta: 'Auto follows Access.',
        more: 'Whether a helper may write in its own copy of the work, an outside folder or a new project. Helpers that only read stay available.',
        control: { kind: 'strip', options: [{ value: 'off', label: 'Off' }, { value: 'auto', label: 'Auto' }, { value: 'on', label: 'On' }] },
        aliases: ['Allow mutative subagents', 'write access'],
        writer: 'generic', codec: 'helperMutation', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT', keys: ['OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT'], page: 'agents', fold: true,
        name: 'At once, for one task', meta: 'Helpers one task may run at the same time.', more: '',
        control: { kind: 'number', min: 1, max: 500 },
        aliases: ['Active subagents per root', 'Active at once per task'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_MAX_SUBAGENT_DEPTH', keys: ['OUROBOROS_MAX_SUBAGENT_DEPTH'], page: 'agents', fold: true,
        name: 'Helpers of helpers', meta: 'How many levels deep. 0 turns helpers off.', more: '',
        control: { kind: 'number', min: 0, max: 10 },
        aliases: ['Subagent depth', 'Depth'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_SUBAGENT_WORKTREE_ROOT', keys: ['OUROBOROS_SUBAGENT_WORKTREE_ROOT'], page: 'agents', fold: true,
        name: 'Folder for working copies', meta: 'Blank: ~/Ouroboros/subagent_worktrees. Old copies are cleaned up.', more: '',
        control: { kind: 'text' },
        aliases: ['Subagent worktree root', 'worktrees'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_SUBAGENT_PROJECTS_ROOT', keys: ['OUROBOROS_SUBAGENT_PROJECTS_ROOT'], page: 'agents', fold: true,
        name: 'Folder for new projects', meta: 'Blank: ~/Ouroboros/projects. Projects are kept.', more: '',
        control: { kind: 'text' },
        aliases: ['Subagent projects root (genesis)', 'projects root'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
];

/** Rows drawn by their page's own code: `{ id, keys, page }`, read by search and by the "one key, one control" test. */
export const HAND_WRITTEN = [];

// The home of Settings (settings_home.js): its six groups in this order, each row a door to a Settings page or out of
// Settings (`to`: a path or a door id). Old tab names are aliases, so search still finds them.
export const HOME_ROWS = [
    { id: 'home:models', group: 'Mind', name: 'Models', to: 'models', glyph: 'sparkle', aliases: ['Model', 'Main model'] },
    { id: 'home:agents', group: 'Mind', name: 'Agents', to: 'agents', glyph: 'bot', aliases: ['Subagents', 'Helpers'] },
    { id: 'home:consciousness', group: 'Life', name: 'Consciousness', to: 'dashboard:consciousness', glyph: 'brain', aliases: ['Background consciousness'] },
    { id: 'home:evolution', group: 'Life', name: 'Evolution', to: 'dashboard:evolution', glyph: 'projects', aliases: ['Self-evolution'] },
    { id: 'home:circles', group: 'Life', name: 'Circles', to: 'page:circles', glyph: 'circles', aliases: ['Places'] },
    { id: 'home:rules', group: 'Limits', name: 'Your rules', to: 'rules', glyph: 'book', aliases: ['Rules'] },
    { id: 'home:permissions', group: 'Limits', name: 'Permissions and checks', to: 'permissions', glyph: 'check', aliases: ['Behavior', 'Access', 'Safety'] },
    { id: 'home:costs', group: 'Limits', name: 'Costs', to: 'dashboard:costs', glyph: 'gauge', aliases: ['Budget', 'Spending'] },
    { id: 'home:accounts', group: 'Connections', name: 'Accounts and keys', to: 'accounts', glyph: 'globe', aliases: ['Providers', 'Accounts', 'Secrets', 'API keys'] },
    { id: 'home:tools', group: 'Connections', name: 'Tools', to: 'tools', glyph: 'plug', aliases: ['MCP servers', 'Tool servers'] },
    { id: 'home:skills', group: 'Connections', name: 'Skills', to: 'page:skills', glyph: 'skills', aliases: ['Extensions'] },
    { id: 'home:appearance', group: 'Interface', name: 'Appearance', to: 'appearance', glyph: 'template', aliases: ['Theme', 'Language'] },
    { id: 'home:notifications', group: 'Interface', name: 'Notifications', to: 'notifications', glyph: 'inbox', aliases: [] },
    { id: 'home:voice', group: 'Interface', name: 'Voice and calls', to: 'voice', glyph: 'mic', aliases: ['Calls', 'Dictation'] },
    { id: 'home:updates', group: 'System', name: 'Updates', to: 'dashboard:updates', glyph: 'share', aliases: ['Update channel'] },
    { id: 'home:storage', group: 'System', name: 'Storage', to: 'dashboard:storage', glyph: 'files', aliases: ['Cleanup'] },
    { id: 'home:computer', group: 'System', name: 'This computer', to: 'computer', glyph: 'settings', aliases: ['Advanced', 'About', 'Network'] },
];

/** A page's rows in order; `fold` true or false keeps only the rows behind (or before) the page's fold. */
export function rowsOf(page, { fold } = {}) {
    return ROWS.filter((row) => row.page === page && (fold === undefined || row.fold === fold));
}

export function rowById(id) {
    return ROWS.find((row) => row.id === id) || null;
}
