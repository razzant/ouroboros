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

    // Permissions and checks: what it may change, who watches its actions, who checks its results, what it is
    // trusted with. The Safety scale is shown as Off / Outgoing / + Commands (stored off / light / full; the words are
    // consciousness_words.js SAFETY_LEVELS). A door's `control.to` is a Settings path or a door id.
    { id: 'OUROBOROS_RUNTIME_MODE', keys: ['OUROBOROS_RUNTIME_MODE'], page: 'permissions', fold: false,
        name: 'Access', meta: 'How far it reaches into its own code and this computer.', more: '',
        control: { kind: 'choices', options: [
            { value: 'light', label: 'Light', note: 'Works on your projects. Never changes its own code.' },
            { value: 'advanced', label: 'Advanced', note: 'May grow its own skills and outer layer.' },
            { value: 'pro', label: 'Pro', note: 'May change its protected core. Every change passes review.' },
            { value: 'cyber_pro', label: 'Cyber Pro', note: 'Full say over itself and this computer. Reviews advise, it decides.' }] },
        aliases: ['Runtime mode', 'self-modification', 'own code', 'Light', 'Advanced', 'Pro', 'Cyber Pro'],
        writer: 'runtimeMode', codec: 'access', confirm: 'raiseAccess', hiddenWhen: '' },
    { id: 'OUROBOROS_SAFETY_MODE', keys: ['OUROBOROS_SAFETY_MODE'], page: 'permissions', fold: false,
        name: 'Work you gave it', meta: 'What a second model reads before a step of your tasks runs.',
        more: 'Outside Cyber Pro Ouroboros cannot lower its own checks; under Cyber Pro it may set them too. Every step a lower level lets through is recorded.',
        control: { kind: 'strip', options: [
            { value: 'off', label: 'Off', note: 'No model checks. Fixed rules still guard protected files.' },
            { value: 'light', label: 'Outgoing', note: 'Messages, posts, skills, connectors and starting work are read first.' },
            { value: 'full', label: '+ Commands', note: 'Shell commands and scripts are read first too. File edits are not.' }] },
        aliases: ['Safety Supervisor', 'Safety', 'Your work', 'safety level', 'Off', 'Light', 'Full'],
        writer: 'safetyMode', codec: '', confirm: 'lowerSafety', hiddenWhen: '' },
    { id: 'OUROBOROS_CONSCIOUSNESS_SAFETY_MODE', keys: ['OUROBOROS_CONSCIOUSNESS_SAFETY_MODE'], page: 'permissions', fold: false,
        name: 'Its own life', meta: 'What a second model reads before a step it takes on its own, places included.', more: '',
        control: { kind: 'strip', options: [
            { value: 'off', label: 'Off', note: 'No model checks. Fixed rules still guard protected files.' },
            { value: 'light', label: 'Outgoing', note: 'Messages, posts, skills, connectors and starting work are read first.' },
            { value: 'full', label: '+ Commands', note: 'Shell commands and scripts are read first too. File edits are not.' }] },
        aliases: ['Safety Supervisor', 'Safety', 'Consciousness', 'safety level', 'Check consciousness fully'],
        writer: 'safetyMode', codec: '', confirm: 'lowerSafety', hiddenWhen: '' },
    { id: 'permissions:rules', keys: [], page: 'permissions', fold: false,
        name: 'Your rules', meta: 'Ouroboros reads them every time; the Supervisor reads them before actions.', more: '',
        control: { kind: 'door', to: 'rules' }, aliases: ['rules'], writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'permissions:supervisor', keys: [], page: 'permissions', fold: false,
        name: 'The Supervisor thinks with', meta: '', more: '',
        control: { kind: 'door', to: 'models/role/safety' }, aliases: ['Checked by', 'Safety Supervisor model'],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'permissions:second-check', keys: [], page: 'permissions', fold: false,
        name: 'Second check', meta: '', more: '',
        control: { kind: 'door', to: 'agents' }, aliases: ['re-check', 'recheck', 'flagged actions'],
        writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_TASK_REVIEW_MODE', keys: ['OUROBOROS_TASK_REVIEW_MODE'], page: 'permissions', fold: false,
        name: 'Results of your tasks', meta: 'Whether a finished task gets a second look.',
        more: 'Required also reviews an answer its task requires checked. With Advisory, Ouroboros decides after a review whether a revised result '
            + 'needs another; with Blocking, a result that did not pass is held until a review passes it. A review already running can still answer after Off.',
        control: { kind: 'strip', options: [
            { value: 'off', label: 'Off', note: 'Not reviewed. No new review starts, a requested one included.' },
            { value: 'auto', label: 'Auto', note: 'Ouroboros asks for a review when it wants one.' },
            { value: 'required', label: 'Required', note: 'Every result a task produced is reviewed.' }] },
        aliases: ['Task Result Review', 'Review of task results', 'acceptance'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_REVIEW_ENFORCEMENT', keys: ['OUROBOROS_REVIEW_ENFORCEMENT'], page: 'permissions', fold: false,
        name: 'When reviewers find a serious problem', meta: 'Whether a review can stop a change or only advise.',
        more: 'Blocking stops a commit and the turning on of a reviewed skill while a critical finding stands. Under Cyber Pro a review only advises.',
        control: { kind: 'strip', options: [
            { value: 'advisory', label: 'Advisory', note: 'It reads the finding and decides.' },
            { value: 'blocking', label: 'Blocking', note: 'Changes to itself and new skills wait until a review passes.' }] },
        aliases: ['Review Enforcement', 'Reviews', 'Advisory', 'Blocking'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'permissions:reviewers', keys: [], page: 'permissions', fold: false,
        name: 'Reviewers', meta: 'The agents that read finished work.', more: '',
        control: { kind: 'door', to: 'agents' }, aliases: ['review pool'], writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_REVIEW_MAX_CYCLES', keys: ['OUROBOROS_REVIEW_MAX_CYCLES'], page: 'permissions', fold: true,
        name: 'Paid review rounds for one change', meta: 'After the last one it may still correct its work.',
        more: 'Counts paid review waves, a failed one included: result review and commit review per task, skill review per task or snapshot. '
            + 'Collecting and replaying a review is free. ∞ removes the count; deadlines and budgets still apply.',
        control: { kind: 'strip', options: [{ value: '1', label: '1' }, { value: '2', label: '2' }, { value: '3', label: '3' },
            { value: '5', label: '5' }, { value: 'unlimited', label: '∞' }] },
        aliases: ['Max Review Cycles', 'Paid review rounds', 'review cycles', 'no limit', 'unlimited'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS', keys: ['OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS'], page: 'permissions', fold: false,
        name: 'Give a reviewed skill its keys', meta: 'After a review of exactly that version. Off asks you for every key and permission.', more: '',
        control: { kind: 'switch' }, aliases: ['Auto-grant reviewed skills', 'auto-grant', 'grant', 'Skills'],
        writer: 'autoGrant', codec: 'bool', confirm: 'autoGrantOn', hiddenWhen: '' },

    // Your rules: one text with its own Save (settings_safety.js draws and writes it through its owner endpoint).
    { id: 'OUROBOROS_SAFETY_OWNER_RULES', keys: ['OUROBOROS_SAFETY_OWNER_RULES'], page: 'rules', fold: false,
        name: 'Your rules', meta: 'Standing rules in your own words.', more: '',
        control: { kind: 'custom' }, aliases: ['rules', 'never', 'always', 'forbid'], writer: 'safetyRules', codec: '', confirm: '', hiddenWhen: '' },

    // Appearance: theme and size are this device's (theme.js, interface_size.js); the language is the install's and
    // writes through its own endpoint (settings_language.js).
    { id: 'appearance:theme', keys: [], page: 'appearance', fold: false, name: 'Theme', meta: '', more: '',
        control: { kind: 'custom' }, aliases: ['Light', 'Dark', 'System', 'dark mode'], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'appearance:size', keys: [], page: 'appearance', fold: false, name: 'Size', meta: 'Everything larger or smaller.', more: '',
        control: { kind: 'custom' }, aliases: ['Interface size', 'zoom', 'text size'], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_UI_LANGUAGE', keys: ['OUROBOROS_UI_LANGUAGE'], page: 'appearance', fold: false,
        name: 'Language', meta: 'Menus and system messages, on every device. Ouroboros answers in the language you write.', more: '',
        control: { kind: 'custom' }, aliases: ['Interface language', 'translation', 'Other'], writer: 'language', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'appearance:translations', keys: [], page: 'appearance', fold: true,
        name: 'Translations', meta: 'Import a translation file, export this language, or generate its translations again.', more: '',
        control: { kind: 'custom' }, aliases: ['Import JSON', 'Export', 'Regenerate'], writer: '', codec: '', confirm: '', hiddenWhen: '' },

    // Notifications: this device's choices (notifications.js writes them; `control.pref` names its preference).
    { id: 'notify:enabled', keys: [], page: 'notifications', fold: false, name: 'Notify me on this device', meta: '', more: '',
        control: { kind: 'custom', pref: 'enabled' }, aliases: ['Enable notifications'], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:needs_answer', keys: [], page: 'notifications', fold: false, name: 'A question waits for you', meta: '', more: '',
        control: { kind: 'custom', pref: 'needs_answer' }, aliases: ['A question or decision is waiting for you'], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:task_done', keys: [], page: 'notifications', fold: false, name: 'A task finished or stopped', meta: '', more: '',
        control: { kind: 'custom', pref: 'task_done' }, aliases: [], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:important', keys: [], page: 'notifications', fold: false, name: 'Ouroboros writes while it works',
        meta: 'Its messages while it works, its reminders and skill notices.', more: '',
        control: { kind: 'custom', pref: 'important' }, aliases: ['reminders', 'skill notices'], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:main_reply', keys: [], page: 'notifications', fold: false, name: 'Ordinary replies in Main', meta: '', more: '',
        control: { kind: 'custom', pref: 'main_reply' }, aliases: [], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:sound', keys: [], page: 'notifications', fold: false, name: 'Sound', meta: 'Off asks the system for silent alerts.', more: '',
        control: { kind: 'custom', pref: 'sound' }, aliases: [], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:tone_volume', keys: [], page: 'notifications', fold: false, name: 'App tone volume',
        meta: 'Only the soft tone the app plays itself. The system sets the volume of its own alerts.', more: '',
        control: { kind: 'custom' }, aliases: ['Volume', 'Preview tone'], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:show_text', keys: [], page: 'notifications', fold: false, name: 'Show the message text',
        meta: 'Off shows only what kind of event it is.', more: '',
        control: { kind: 'custom', pref: 'show_text' }, aliases: [], writer: 'device', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:test', keys: [], page: 'notifications', fold: false, name: 'Try it', meta: '', more: '',
        control: { kind: 'custom' }, aliases: ['Send a test notification'], writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'notify:places', keys: [], page: 'notifications', fold: false, name: 'Telegram and other places',
        meta: 'What Ouroboros sends there is set with each place, in Circles.', more: '',
        control: { kind: 'door', to: 'page:circles' }, aliases: [], writer: '', codec: '', confirm: '', hiddenWhen: '' },

    // Voice and calls: no setting of its own. Two rows say the saved choice and where it is made; a skill that
    // registered a live conversation gets a door to its own form, drawn by the page from the contributions store.
    { id: 'voice:dictation', keys: [], page: 'voice', fold: false, name: 'Dictation',
        meta: 'Chosen at the message field: +, then Add dictation.', more: '',
        control: { kind: 'custom' }, aliases: ['speech to text', 'microphone', 'Whisper'], writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'voice:read-aloud', keys: [], page: 'voice', fold: false, name: 'Read aloud',
        meta: 'Chosen in a reply’s menu: Read aloud, then its voice.', more: '',
        control: { kind: 'custom' }, aliases: ['voice', 'text to speech'], writer: '', codec: '', confirm: '', hiddenWhen: '' },

    // This computer: the host's two switches (settings_autostart.js), who can reach it, and doors to its two pages.
    { id: 'computer:autostart', keys: [], page: 'computer', fold: false, name: 'Start when I sign in', meta: '', more: '',
        control: { kind: 'custom' }, aliases: ['Start on sign-in', 'autostart', 'startup'], writer: 'host', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_DESKTOP_KEEP_RUNNING', keys: ['OUROBOROS_DESKTOP_KEEP_RUNNING'], page: 'computer', fold: false,
        name: 'Keep running when the window is closed',
        meta: 'Tasks, schedules and Telegram keep working. An icon brings the window back or quits.', more: '',
        control: { kind: 'custom' }, aliases: ['background', 'Keep running when the window closes'], writer: 'host', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_SERVER_HOST', keys: ['OUROBOROS_SERVER_HOST'], page: 'computer', fold: false,
        name: 'Open Ouroboros from', meta: '', more: '',
        control: { kind: 'strip', options: [
            { value: '127.0.0.1', label: 'Only here', note: 'Only this computer.' },
            { value: '0.0.0.0', label: 'My network', note: 'Any device on your network, with the password.' }] },
        aliases: ['Server Bind Host', 'Network Gate', 'Open to the network', 'LAN', '0.0.0.0', '127.0.0.1'],
        writer: 'generic', codec: 'network', confirm: 'openNetwork', hiddenWhen: '' },
    { id: 'OUROBOROS_NETWORK_PASSWORD', keys: ['OUROBOROS_NETWORK_PASSWORD'], page: 'computer', fold: false,
        name: 'Password', meta: 'Asked on every other device.', more: '',
        control: { kind: 'secret' }, aliases: ['Network Password'], writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'computer:address', keys: [], page: 'computer', fold: false, name: 'Address for other devices',
        meta: 'Type it in a browser on the same network.', more: '',
        control: { kind: 'custom' }, aliases: ['LAN URL', 'Copy link'], writer: '', codec: '', confirm: '', hiddenWhen: 'noLanAddress' },
    { id: 'computer:advanced', keys: [], page: 'computer', fold: false, name: 'Advanced',
        meta: 'Workers, the interface folder, certificates.', more: '',
        control: { kind: 'door', to: 'computer/advanced' }, aliases: [], writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'computer:about', keys: [], page: 'computer', fold: false, name: 'About', meta: '', more: '',
        control: { kind: 'door', to: 'computer/about' }, aliases: ['version', 'licences'], writer: '', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_MAX_WORKERS', keys: ['OUROBOROS_MAX_WORKERS'], page: 'computer/advanced', fold: false,
        name: 'Tasks at once', meta: 'Worker processes that run tasks side by side.', more: '',
        control: { kind: 'number', min: 1, max: 50 }, aliases: ['Max Workers', 'workers'], writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_UI_DIR', keys: ['OUROBOROS_UI_DIR'], page: 'computer/advanced', fold: false,
        name: 'Interface folder', meta: 'Empty serves the built-in interface.',
        more: 'none: no main page, for an interface hosted separately behind the same address. Otherwise an absolute or ~ path, on the '
            + 'computer that runs Ouroboros, to a trusted folder with its own index.html; a path that cannot be served keeps the built-in '
            + 'interface and says why. After the restart the built-in page reloads into the new choice; a page from a folder or another host '
            + 'refreshes itself and only reconnects. To return to the built-in interface, clear this field (from any interface, or in '
            + 'settings.json while Ouroboros is stopped), restart, and open the Ouroboros address again. An OUROBOROS_UI_DIR environment '
            + 'value shows here until a choice is saved; a saved choice, empty included, wins.',
        control: { kind: 'text' }, aliases: ['Interface directory', 'Main Interface', 'OUROBOROS_UI_DIR', 'Serving now'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'OUROBOROS_EXTRA_CA_BUNDLE', keys: ['OUROBOROS_EXTRA_CA_BUNDLE'], page: 'computer/advanced', fold: false,
        name: 'Extra certificate bundle', meta: 'A PEM file on this computer, for a network with its own authority.',
        more: 'Only for a network whose certificates the default bundle does not trust: a proxy that inspects TLS, or an endpoint '
            + 'behind a national authority. Leave empty unless a provider fails with a certificate error.',
        control: { kind: 'text' }, aliases: ['Extra CA bundle', 'Extra CA Certificates', 'PEM', 'certificate'],
        writer: 'generic', codec: '', confirm: '', hiddenWhen: '' },
    { id: 'computer:reset', keys: [], page: 'computer/about', fold: true, name: 'Reset all data',
        meta: 'Deletes state, memory, logs and settings, then restarts. The code stays.', more: '',
        control: { kind: 'custom' }, aliases: ['Danger Zone', 'Erase all data', 'Reset All Data'], writer: '', codec: '', confirm: '', hiddenWhen: '' },

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

/** A page's rows in order; `fold` true or false keeps only the rows behind (or before) the page's fold. */
export function rowsOf(page, { fold } = {}) {
    return ROWS.filter((row) => row.page === page && (fold === undefined || row.fold === fold));
}

export function rowById(id) {
    return ROWS.find((row) => row.id === id) || null;
}
