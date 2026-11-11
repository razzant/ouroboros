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

    // Skill sources (settings_extension_forms.js), opened from the Skills page's sources view and from search.
    { id: 'OUROBOROS_SKILLS_REPO_PATH', keys: ['OUROBOROS_SKILLS_REPO_PATH'], page: 'skill-sources', fold: false,
        name: 'Skills folder', meta: 'One more folder of skills on this computer. Ouroboros reads it and never clones or pulls it.', more: '',
        control: { kind: 'text' },
        aliases: ['External Skills Repo', 'Skills Repo Path', 'skills repository'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_CLAWHUB_REGISTRY_URL', keys: ['OUROBOROS_CLAWHUB_REGISTRY_URL'], page: 'skill-sources', fold: false,
        name: 'ClawHub address', meta: 'Change it only for a mirror of your own. The host must be clawhub.ai or localhost.', more: '',
        control: { kind: 'text' },
        aliases: ['ClawHub Marketplace', 'Registry URL', 'ClawHub Registry URL'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
];

/** Rows drawn by their page's own code: `{ id, keys, page }`, read by search and by the "one key, one control" test. */
export const HAND_WRITTEN = [];

// The home of Settings (settings_home.js): its six groups in this order, each row a door to a Settings page or out of
// Settings (`to`: a path or a door id). `glyph` names its picture; the tabs Settings had are aliases, so search still
// finds them.
export const HOME_ROWS = [
    { id: 'home:models', group: 'Mind', name: 'Models', to: 'models', glyph: 'sparkle', aliases: ['Model', 'Main model'] },
    { id: 'home:agents', group: 'Mind', name: 'Agents', to: 'agents', glyph: 'bot', aliases: ['Subagents', 'Helpers'] },
    { id: 'home:consciousness', group: 'Life', name: 'Consciousness', to: 'dashboard:consciousness', glyph: 'brain', aliases: ['Background consciousness'] },
    { id: 'home:evolution', group: 'Life', name: 'Evolution', to: 'dashboard:evolution', glyph: 'growth', aliases: ['Self-evolution', 'Post-Task Self-Evolution'] },
    { id: 'home:circles', group: 'Life', name: 'Circles', to: 'page:circles', glyph: 'circles', aliases: ['Places'] },
    { id: 'home:rules', group: 'Limits', name: 'Your rules', to: 'rules', glyph: 'rules', aliases: ['Rules', 'Safety rules'] },
    { id: 'home:permissions', group: 'Limits', name: 'Permissions and checks', to: 'permissions', glyph: 'check', aliases: ['Behavior', 'Access', 'Safety', 'Review'] },
    { id: 'home:costs', group: 'Limits', name: 'Costs', to: 'dashboard:costs', glyph: 'gauge', aliases: ['Budget', 'Spending'] },
    { id: 'home:accounts', group: 'Connections', name: 'Accounts and keys', to: 'accounts', glyph: 'key', aliases: ['Providers', 'Accounts', 'Secrets', 'API keys'] },
    { id: 'home:tools', group: 'Connections', name: 'Tools', to: 'tools', glyph: 'plug', aliases: ['MCP servers', 'Tool servers'] },
    { id: 'home:skills', group: 'Connections', name: 'Skills', to: 'page:skills', glyph: 'skills', aliases: ['Extensions'] },
    { id: 'home:appearance', group: 'Interface', name: 'Appearance', to: 'appearance', glyph: 'moon', aliases: ['Theme', 'Language', 'Interface size'] },
    { id: 'home:notifications', group: 'Interface', name: 'Notifications', to: 'notifications', glyph: 'bell', aliases: ['Sound'] },
    { id: 'home:voice', group: 'Interface', name: 'Voice and calls', to: 'voice', glyph: 'mic', aliases: ['Calls', 'Dictation'] },
    { id: 'home:updates', group: 'System', name: 'Updates', to: 'dashboard:updates', glyph: 'refresh', aliases: ['Update channel'] },
    { id: 'home:storage', group: 'System', name: 'Storage', to: 'dashboard:storage', glyph: 'drive', aliases: ['Cleanup', 'GC Retention'] },
    { id: 'home:computer', group: 'System', name: 'This computer', to: 'computer', glyph: 'monitor', aliases: ['Advanced', 'About', 'Network', 'Startup'] },
];

// Doors search offers to pages of Settings that no home row opens (settings_home.js searchEntries): the list of the
// forms skills registered, which stood under Advanced as "Extension Settings".
export const SEARCH_DOORS = [
    { id: 'door:skill', name: 'Skill settings', to: 'skill', glyph: 'skills', aliases: ['Extension Settings', 'Advanced', 'Settings of skills'] },
];

/** A page's rows in order; `fold` true or false keeps only the rows behind (or before) the page's fold. */
export function rowsOf(page, { fold } = {}) {
    return ROWS.filter((row) => row.page === page && (fold === undefined || row.fold === fold));
}

export function rowById(id) {
    return ROWS.find((row) => row.id === id) || null;
}
