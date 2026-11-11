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

    // Dashboard › Consciousness: the card "On its own" (the first four, then its fold), then the doors of "Around it".
    // The strip's labels are AUTONOMY_LEVELS (consciousness_autonomy.js); a test keeps the two equal. A row without a
    // meta says its value in its own words on the tab (the chosen level, the range in minutes or hours).
    { id: 'OUROBOROS_CONSCIOUSNESS_AUTONOMY', keys: ['OUROBOROS_CONSCIOUSNESS_AUTONOMY'], page: 'dashboard:consciousness', fold: false,
        name: 'How far it goes', meta: '', more: '',
        control: { kind: 'strip', options: [{ value: 'observe', label: 'Advises' }, { value: 'act', label: 'Acts' },
            { value: 'full', label: 'Evolves' }] },
        aliases: ['Autonomy', 'Consciousness autonomy'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_BG_WAKEUP', keys: ['OUROBOROS_BG_WAKEUP_MIN', 'OUROBOROS_BG_WAKEUP_MAX'], page: 'dashboard:consciousness', fold: false,
        name: 'Wakes every', meta: '', more: '',
        control: { kind: 'pair', unit: 's', min: 60, labels: ['Shortest wait, in seconds', 'Longest wait, in seconds'] },
        aliases: ['Shortest wake interval (s)', 'Longest wake interval (s)', 'wake interval'],
        writer: 'wakeRange', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_CONSCIOUSNESS_MAX_TASKS', keys: ['OUROBOROS_CONSCIOUSNESS_MAX_TASKS'], page: 'dashboard:consciousness', fold: false,
        name: 'Own tasks at once', meta: 'Zero: it thinks and writes, and starts no tasks.', more: '',
        control: { kind: 'number', min: 0, max: 32 },
        aliases: ['Max concurrent tasks', 'its own tasks'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'consciousness-allowance', keys: [], page: 'dashboard:consciousness', fold: false,
        name: 'Daily allowance', meta: 'With the other limits, in Costs.', more: '',
        control: { kind: 'door' },
        aliases: [],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_PRESENCE_MAX_ACTIVE', keys: ['OUROBOROS_PRESENCE_MAX_ACTIVE'], page: 'dashboard:consciousness', fold: true,
        name: 'Concurrent conversations', meta: 'Places it talks in at the same time. Others wait their turn.', more: '',
        control: { kind: 'number', min: 1, max: 20 },
        aliases: ['conversations at once'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'consciousness-model', keys: [], page: 'dashboard:consciousness', fold: false,
        name: 'Thinks with', meta: '', more: '',
        control: { kind: 'door' },
        aliases: [],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'consciousness-places', keys: [], page: 'dashboard:consciousness', fold: false,
        name: 'Where it lives', meta: 'Its places, their words and their circles.', more: '',
        control: { kind: 'door' },
        aliases: [],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },

    // Dashboard › Evolution: whether a cycle may follow your tasks (one choice over two keys, said in its own words on
    // the tab), its floor, and its standing direction (a card of its own).
    { id: 'OUROBOROS_POST_TASK_EVOLUTION', keys: ['OUROBOROS_POST_TASK_EVOLUTION', 'OUROBOROS_POST_TASK_EVOLUTION_CADENCE'],
        page: 'dashboard:evolution', fold: false,
        name: 'After finishing your tasks', meta: '', more: '',
        control: { kind: 'custom', options: [{ value: 'off', label: 'Off' }, { value: 'llm', label: 'After each task' },
            { value: 'every_n', label: 'Every N tasks' }] },
        aliases: ['Post-Task Self-Evolution', 'Every N Tasks', 'evolution cadence'],
        writer: 'evolutionWhen', codec: 'evolutionWhen', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD', keys: ['OUROBOROS_POST_TASK_EVOLUTION_BUDGET_USD'], page: 'dashboard:evolution', fold: false,
        name: 'Start only above', meta: 'Budget that must be left before a cycle starts. 0 sets no floor.', more: '',
        control: { kind: 'number', unit: '$', min: 0 },
        aliases: ['Post-task Start Floor (USD)', 'start floor'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE', keys: ['OUROBOROS_EVOLUTION_PERSISTENT_OBJECTIVE'], page: 'dashboard:evolution', fold: false,
        name: 'What to work on', meta: 'A standing direction added to every cycle. Empty leaves the choice to Ouroboros.', more: '',
        control: { kind: 'text' },
        aliases: ['Standing Objective (optional)', 'standing objective'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },

    // Dashboard › Updates: which official branch Update follows, each choice with its sentence. A deployment that
    // updates the installation itself leaves the row out.
    { id: 'OUROBOROS_UPDATE_CHANNEL', keys: ['OUROBOROS_UPDATE_CHANNEL'], page: 'dashboard:updates', fold: false,
        name: 'Which updates', meta: '', more: '',
        control: { kind: 'custom', options: [
            { value: 'stable', label: 'Stable', note: 'Released versions. The main branch.' },
            { value: 'qa', label: 'QA', note: 'Release candidates. The ouroboros-stable branch.' },
            { value: 'development', label: 'Development', note: 'Every change as it lands. The ouroboros branch.' }] },
        aliases: ['Update Channel', 'Channel', 'release channel'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: 'managedDeployment' },

    // Dashboard › Storage: how long finished work is kept, the first row of its Cleanup card.
    { id: 'OUROBOROS_GC_RETENTION_DAYS', keys: ['OUROBOROS_GC_RETENTION_DAYS'], page: 'dashboard:storage', fold: false,
        name: 'Keep finished work for', meta: 'Older working copies and task files are removed. Project folders stay.', more: '',
        control: { kind: 'number', unit: 'days', min: 1, max: 365 },
        aliases: ['GC Retention (days)', 'retention'],
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

    // Settings › Agents: the roster (its page draws the switch and the list of agents and writes them), then the fold.
    { id: 'OUROBOROS_SUBAGENTS', keys: ['OUROBOROS_SUBAGENTS'], page: 'agents', fold: false,
        name: 'Hand work to helpers', meta: 'Ouroboros decides when. Off stops it in every room.', more: '',
        control: { kind: 'custom' },
        aliases: ['Delegation', 'Available subagents', 'Subagents', 'Reviewers', 'Agents list'],
        writer: 'checker', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS', keys: ['OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS'], page: 'agents', fold: true,
        name: 'Helpers may change files', meta: 'Auto follows Access.',
        more: 'Whether a helper may write in its own copy of the work, an outside folder or a new project. Helpers that only read stay available.',
        control: { kind: 'strip', options: [{ value: 'false', label: 'Off' }, { value: '', label: 'Auto' }, { value: 'true', label: 'On' }] },
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
