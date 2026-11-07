import { bindPageHeader, renderPageHeader, renderSegmentedField, renderTabStrip, bindTabStrip } from './page_header.js';
import { renderAgentAccountsSection } from './harness_accounts.js';
import { renderSubagentsSection } from './subagents_settings.js';
import { languageBlockHtml } from './settings_language.js';
import { modelRolesHost } from './model_roles.js';
import { safetySectionHtml } from './settings_safety.js';
import { consciousnessSectionHtml } from './settings_consciousness.js';
import { PROVIDER_CARDS, providerSettingsCard } from './settings_provider_cards.js';
import { settingsSection } from './settings_controls.js';

// Reads as a sequence: keys → secrets → which API models → who among the agents
// does what → behavior → technical. "Agents", not "Coding agents" (D-10): the
// same subscriptions build presentations and run arbitrary tasks, so the
// narrower word named only one of their uses.
const SETTINGS_TABS = [
    { value: 'providers', label: 'Accounts' },
    { value: 'secrets', label: 'Secrets' },
    { value: 'models', label: 'Models' },
    { value: 'agents', label: 'Agents' },
    { value: 'behavior', label: 'Behavior' },
    { value: 'appearance', label: 'Appearance' },
    { value: 'advanced', label: 'Advanced' },
    { value: 'about', label: 'About' },
];
// Guard markers: renderTabStrip emits behavior/advanced tabs at runtime.

// The setup contract replaces this after the settings read. Keep the accessible
// description useful while that read is pending or unavailable.
const BUDGET_NOTE_FALLBACK = 'Budget limits use known spend: Total Budget applies globally, while Per-task Cost Cap applies to one root task tree. Unpriced calls remain unknown and are not counted as $0; reservations and unresolved charges are not counted as spending.';

// Runtime mode is one axis of the owner policy contract. Keep the Settings
// presentation in the same vocabulary as the onboarding setup contract; the
// saved value is still handled by settings.js and the owner endpoint.
const RUNTIME_MODE_OPTIONS = [
    { value: 'light', label: 'Light' },
    { value: 'advanced', label: 'Advanced' },
    { value: 'pro', label: 'Pro' },
    { value: 'cyber_pro', label: 'Cyber Pro' },
];

function secretField({ id, settingKey, label, placeholder }) {
    return `
        <div class="form-field ui-field">
            <label for="${id}">${label}</label>
            <div class="secret-input-row">
                <input id="${id}" name="${settingKey}" data-secret-setting="${settingKey}" class="secret-input ui-control" type="password" placeholder="${placeholder}">
                <button type="button" class="btn btn-default secret-toggle" data-target="${id}">Show</button>
                <button type="button" class="btn btn-default secret-clear" data-target="${id}">Clear</button>
            </div>
        </div>
    `;
}

// Provider API keys have one input each, in Accounts, and the network password sits with the bind host it
// guards (Network Gate); this tab keeps the other stored secrets.
export const SECRET_KEYS = [
    ['GITHUB_TOKEN', 'GitHub Token', 'ghp_...'],
];

/** A section's one choice: the hidden value and its segmented row, named by the section title (no tray, no second label). */
function choice({ id, value, options, modifier = '', title = '' }) {
    return `<input id="${id}" type="hidden" value="${value}">${renderSegmentedField({ target: id, modifier, title, options })}`;
}

const policyLine = (key) => `<div class="settings-inline-note" data-policy-state="${key}" role="status" aria-live="polite"></div>`;

function secretSettingsSection() {
    return `
        ${settingsSection({
            title: 'Stored Secrets',
            lead: 'Tokens Ouroboros itself signs in with. Provider API keys are in Accounts.',
            details: 'Tokens and the secrets skills request. Skills only receive grant-only keys after explicit human approval.',
            body: `<div class="form-grid two">
                ${SECRET_KEYS.map(([key, label, placeholder]) => secretField({
                    id: `s-secret-${key.toLowerCase().replace(/_/g, '-')}`,
                    settingKey: key,
                    label,
                    placeholder,
                })).join('')}
            </div>`,
        })}
        ${settingsSection({
            title: 'Requested By Skills',
            lead: 'Keys installed skills asked for, each under the skill that needs it.',
            details: 'Secrets requested by installed skills appear here only when a skill asks for them.',
            body: `<div id="skill-requested-secrets" class="settings-secret-list">
                <div class="muted">No skill-requested secrets.</div>
            </div>`,
        })}
        ${settingsSection({
            title: 'Custom Keys',
            lead: 'Keys of your own that skills may read.',
            details: 'Optional key/value storage for future skills. Use uppercase names such as <code>SLACK_WEBHOOK_URL</code>.',
            actions: '<button type="button" class="btn btn-default" id="btn-add-custom-secret">Add custom key</button>',
            body: '<div id="custom-secrets-list" class="settings-secret-list settings-custom-secret-list"></div>',
        })}
    `;
}

function accountsPanel() {
    const moreHint = PROVIDER_CARDS.filter((card) => card.advanced).map((card) => card.title).join(', ');
    return `
        ${renderAgentAccountsSection()}
        ${settingsSection({
            title: 'API keys',
            lead: 'Each role names the key it uses; keys never rotate. For a second key of the same model, add it as a fallback.',
            body: `${PROVIDER_CARDS.filter((card) => !card.advanced).map(providerSettingsCard).join('')}
            <details class="settings-more-providers ui-disclosure" id="settings-more-providers">
                <summary>
                    <span class="settings-provider-title"><span>More providers</span></span>
                    <span class="settings-provider-hint">${moreHint}</span>
                </summary>
                <div class="settings-more-providers-body">
                    ${PROVIDER_CARDS.filter((card) => card.advanced).map(providerSettingsCard).join('')}
                </div>
            </details>`,
        })}
        ${settingsSection({
            title: 'Network Gate',
            lead: 'Who may reach this Ouroboros over the network, and the password that guards it.',
            details: 'Adds a password wall only for non-localhost app and API access. If you expose Ouroboros on LAN or Docker, set a password before sharing the URL. Use <code>127.0.0.1</code> for this machine only. Use <code>0.0.0.0</code> for LAN/Docker access with a network password in the same save. Specific LAN IP binds are manual/env-only.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-server-host">Server Bind Host</label>
                    <input id="s-server-host" placeholder="127.0.0.1 or 0.0.0.0" class="ui-control" name="s-server-host" type="text">
                </div>
                ${secretField({
                    id: 's-network-password',
                    settingKey: 'OUROBOROS_NETWORK_PASSWORD',
                    label: 'Network Password',
                    placeholder: 'Needed for LAN or Docker access',
                })}
            </div>
            <div id="settings-lan-hint" class="settings-lan-hint" hidden></div>`,
        })}
        ${settingsSection({
            title: 'Legacy Compatibility',
            lead: 'An older OpenAI-style endpoint, for installs that still use one.',
            details: 'Backward-compatibility escape hatch for older installs. For new custom providers, use the dedicated <strong>OpenAI Compatible</strong> card instead.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-openai-base-url">Legacy OpenAI Base URL</label>
                    <input id="s-openai-base-url" placeholder="https://api.openai.com/v1 or compatible endpoint" class="ui-control" name="s-openai-base-url" type="text">
                </div>
            </div>`,
        })}
    `;
}

function modelsPanel() {
    return `
        ${settingsSection({
            title: 'Model Routing',
            lead: 'Which model does each kind of work.',
            details: 'Choose a source, model, and subscription account for each role. Auto rotates compatible accounts. Local uses the runtime configured in Advanced.',
            body: `<div class="ui-action-row">
                <button type="button" class="btn btn-default" id="btn-refresh-model-catalog">Refresh model catalog</button>
                <span id="settings-model-catalog-status" class="ui-result" role="status" aria-live="polite" aria-atomic="true"></span>
            </div>
            ${modelRolesHost('settings-model-roles')}`,
        })}

        <!-- Reviewers and Delegation live in the Agents tab (D-10):
             they answer "who does the work", not "which API model
             id". One capability, one section — no control here
             duplicates one there. -->

        ${settingsSection({
            title: 'Web search',
            lead: 'Where the built-in web search runs.',
            body: `<div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="s-websearch-source">Source</label>
                    <select id="s-websearch-source" class="ui-control"><option value="auto">Auto</option><option value="openai">OpenAI</option><option value="openrouter">OpenRouter</option><option value="anthropic">Anthropic</option><option value="ddgs">DuckDuckGo</option></select>
                </div>
                <div class="form-field ui-field">
                    <label for="s-websearch-model">Model</label>
                    <input id="s-websearch-model" placeholder="Recommended for each source" class="ui-control" name="s-websearch-model" type="text" aria-describedby="s-websearch-model-help">
                    <div class="settings-inline-note ui-field-help" id="s-websearch-model-help">Built-in <code>web_search</code> only. Skills, MCP and browser tools remain independent choices. Empty follows the model Ouroboros recommends for each source.</div>
                </div>
            </div>
            <div class="ui-field-help" id="s-websearch-preview" role="status"></div>`,
        })}
    `;
}

function behaviorPanel() {
    return `
        ${settingsSection({
            title: 'Review Enforcement',
            lead: 'Whether a critical review finding may stop Ouroboros.',
            details: '<strong>Advisory</strong> keeps review visible but non-blocking. <strong>Blocking</strong> stops commits and reviewed-skill activation when critical findings remain unresolved, outside Cyber Pro Access.',
            body: `${choice({ id: 's-review-enforcement', value: 'advisory', modifier: 'data-enforcement-group', options: [
                { value: 'advisory', label: 'Advisory' },
                { value: 'blocking', label: 'Blocking' },
            ] })}
            ${policyLine('review')}`,
        })}

        ${settingsSection({
            title: 'Task Result Review',
            lead: 'When a finished task gets a second look.',
            details: 'Auto reviews a result when Ouroboros requests it. Required also reviews material actually produced or observed effects. Off starts no new task-result review, including manual or historical requests; already running reviews can still return feedback. Waiting works in every access mode; Cyber feedback remains advisory.',
            body: choice({ id: 's-task-review-mode', value: 'auto', options: [
                { value: 'off', label: 'Off' },
                { value: 'auto', label: 'Auto' },
                { value: 'required', label: 'Required' },
            ] }),
        })}

        ${settingsSection({
            title: 'Max Review Cycles',
            lead: 'How many paid review rounds one change may take.',
            details: 'Limits paid review waves, including dispatched technical failures: plan and task review per task, commit review per root task, and skill review per root task or manual snapshot. The last review still permits author corrections within ordinary task limits; explicit task-local author limits remain separate. Collection and exact replay are free. Advisory allows an explicit decision after receiving feedback or a disclosed unavailable result; Blocking still requires reviewer approval outside Cyber Pro. <strong>&infin;</strong> removes the count cap, while deadlines, budgets and lifecycle limits still apply.',
            body: choice({ id: 's-review-max-cycles', value: '2', modifier: 'data-review-cycles-group', options: [
                { value: '1', label: '1' },
                { value: '2', label: '2' },
                { value: '3', label: '3' },
                { value: '5', label: '5' },
                { value: 'unlimited', label: '∞' },
            ] }),
        })}

        ${settingsSection({
            title: 'Access',
            lead: 'How far Ouroboros may change its own code and this host.',
            details: `Separate axis from Review Enforcement. Controls how far Ouroboros is allowed to self-modify.
                <strong>Light</strong> blocks repo self-modification but allows reviewed + enabled skills to run.
                <strong>Advanced</strong> is the default &mdash; self-modify the evolutionary layer; protected core/contract/release files stay guarded by the shared runtime-mode policy.
                <strong>Pro</strong> can edit protected core/contract/release surfaces, but commits still go through the normal review gate; Advanced remains limited to the evolutionary layer.
                <strong>Cyber Pro</strong> grants the full host and configuration authority, including credentials, models, Supervisor configuration and protected rewrites. Review still runs and the Review Enforcement choice is kept, but review findings do not block actions in Cyber Pro: they are evidence for Ouroboros, which decides.
                <br><strong>Human controlled:</strong> desktop builds ask the launcher for native confirmation before saving a mode change.
                Web/Docker sessions save mode changes through the owner endpoint; the new mode takes effect after restart.`,
            body: `${choice({ id: 's-runtime-mode', value: 'advanced', modifier: 'data-runtime-mode-group',
                title: 'Access changes take effect after restart.', options: RUNTIME_MODE_OPTIONS })}
            ${policyLine('access')}`,
        })}

        <!-- Allow Mutative Subagents moved to Agents → Delegation: one
             subagent story in one place, and two controls over one
             setting would have carried two drafts. -->

        ${safetySectionHtml()}

        ${settingsSection({
            title: 'Post-Task Self-Evolution',
            lead: 'Whether Ouroboros may improve itself after finishing your tasks.',
            details: `After an eligible task, Ouroboros can optionally run one reviewed self-improvement cycle: the worker asks the Main model whether one improvement is worth a cycle now, writes a durable request, and the supervisor starts a one-shot campaign later on an idle tick if all gates pass.
                <br><strong>Configuration authority:</strong> outside Cyber Pro, only the owner can enable this. Cyber Pro also lets the agent configure it; selecting Cyber Pro does not enable evolution automatically. Changes apply on the next task.
                <br><strong>Counts every eligible task, including trivial chats.</strong> <strong>Every N=1</strong> means Ouroboros considers self-improvement after every task, then runs the actual cycle later on an idle supervisor tick.`,
            body: `${choice({ id: 's-post-task-evolution-mode', value: 'off', options: [
                { value: 'off', label: 'Off' },
                { value: 'llm', label: 'After Each Task (LLM decides)' },
                { value: 'every_n', label: 'Every N Tasks' },
            ] })}
            <div class="form-row">
                <div class="form-field ui-field" data-evo-every-n-row>
                    <label for="s-evo-cadence-n">Every N Tasks</label>
                    <input id="s-evo-cadence-n" type="number" min="1" step="1" placeholder="3" class="ui-control" name="s-evo-cadence-n" aria-describedby="s-evo-cadence-n-help">
                    <div class="settings-inline-note ui-field-help" id="s-evo-cadence-n-help">Count of finished tasks between two self-improvement checks.</div>
                </div>
                <div class="form-field ui-field">
                    <label for="s-evo-budget">Post-task Start Floor (USD)</label>
                    <input id="s-evo-budget" placeholder="0" class="ui-control" name="s-evo-budget" type="text" aria-describedby="s-evo-budget-help">
                    <div class="settings-inline-note ui-field-help" id="s-evo-budget-help">Minimum remaining global budget required to start a post-task cycle. <strong>0</strong> = rely on the normal gates. Running cycles still inherit the global per-task hard cost cap and the global budget limit.</div>
                </div>
            </div>
            <div class="form-field ui-field">
                <label for="s-evo-objective">Standing Objective (optional)</label>
                <input id="s-evo-objective" placeholder="(none) — e.g. prioritize test coverage and latency" class="ui-control" name="s-evo-objective" type="text" aria-describedby="s-evo-objective-help">
                <div class="settings-inline-note ui-field-help" id="s-evo-objective-help">Optional steer appended to every evolution cycle objective. It never overrides the LLM-first promotion; leave empty for pure LLM choice.</div>
            </div>`,
        })}

        ${consciousnessSectionHtml()}

        ${settingsSection({
            title: 'Context Mode',
            lead: 'How much of its own books Ouroboros keeps in its working window.',
            details: `Working-context size profile (separate axis from Runtime Mode and Review Enforcement).
                <strong>Max</strong>, for ~1M-context models, inlines ARCHITECTURE in full and DEVELOPMENT in full unless the work is known to be about something else: a folder proven not to be Ouroboros, a helper task, or work started through the API, CLI or a schedule gets its chapter map, and work of unknown subject keeps the whole book.
                <strong>Nano</strong> is the compact owner window. <strong>Low</strong> aims for about 250K tokens; what actually fits depends on the model's window and what this installation holds. Both books become chapter maps (read full sections on demand), and memory compacts sooner. In every mode, a task that itself requires the whole DEVELOPMENT handbook gets it. It governs Ouroboros's own working window: it never changes the model or reasoning effort, and the review panel runs in every mode.
                <br><strong>Human controlled:</strong> saved via the owner endpoint; saves immediately (no restart), and lowering requires Ouroboros to be idle.`,
            body: choice({ id: 's-context-mode', value: 'max',
                title: 'Saves immediately; no restart required. Lowering requires Ouroboros to be idle.', options: [
                    { value: 'nano', label: 'Nano' },
                    { value: 'low', label: 'Low' },
                    { value: 'max', label: 'Max' },
                ] }),
        })}

        ${settingsSection({
            title: 'Image Input',
            lead: 'What happens to the pictures you send when a model cannot see them.',
            details: 'Auto sends images unless the model\'s own metadata says it cannot see them; then a model that can see writes a caption, or a short note takes the image\'s place. Inline always sends images, even when metadata says no; if the provider refuses one, you see the refusal and that image becomes a note. Caption always replaces images with text captions. Off replaces images with a note and starts no caption work. The local model and GigaChat cannot carry images in any mode: they get a caption or a note that says so.',
            body: choice({ id: 's-image-input-mode', value: 'auto', modifier: 'data-image-input-group', options: [
                { value: 'auto', label: 'Auto' },
                { value: 'caption', label: 'Caption' },
                { value: 'inline', label: 'Inline' },
                { value: 'off', label: 'Off' },
            ] }),
        })}

        ${settingsSection({
            title: 'Skills',
            lead: 'Whether a reviewed skill gets its keys and permissions without asking you each time.',
            details: 'Closed-loop skill development can auto-grant the keys and host permissions a skill declares after a fresh executable review for the current content hash. Leave this off when every skill permission should require a separate human approval.',
            body: `<label class="local-toggle ui-field ui-field-inline" title="Applies only after a fresh executable skill review and only to manifest-declared grants for that exact content hash.">
                <input type="checkbox" id="s-auto-grant-reviewed-skills" class="ui-checkbox" name="s-auto-grant-reviewed-skills">
                Auto-grant reviewed skills' keys and permissions
            </label>`,
        })}

        ${settingsSection({
            title: 'Prompt Cache TTL',
            lead: 'How long Anthropic-family routes keep Ouroboros\'s context warm between calls.',
            details: `How long provider prompt caches keep this agent's stable context warm (Anthropic-family routes; other providers manage caching implicitly).
                <strong>1h</strong> keeps the large stable prefix cached across long waits and review cycles &mdash; cache writes bill at 2&times; base input instead of 1.25&times;, but one wait longer than 5 minutes already pays that back on big contexts.
                <strong>5m</strong> is the provider default tier as an explicit value; <strong>Default</strong> sends bare markers (provider default, callers may still declare their own TTL).
                On those routes it is one honest global for every lane, review and safety prompts included; Codex and other OpenAI-family routes keep their own cache lifetime, which this setting does not reach.`,
            body: choice({ id: 's-prompt-cache-ttl', value: '1h', options: [
                { value: 'default', label: 'Default' },
                { value: '5m', label: '5m' },
                { value: '1h', label: '1h' },
            ] }),
        })}

        ${settingsSection({
            title: 'External Skills Repo',
            lead: 'An extra folder of skills to scan, if you keep one.',
            details: 'Optional EXTRA discovery path on top of the in-data-plane <code>data/skills/{native,clawhub,external}/</code> tree. Ouroboros scans this for additional skill packages without cloning or pulling them. Leave empty to use only the data plane.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-skills-repo-path">Skills Repo Path</label>
                    <input id="s-skills-repo-path" placeholder="~/Ouroboros/skills or /absolute/path/to/skills" class="ui-control" name="s-skills-repo-path" type="text" aria-describedby="s-skills-repo-path-help">
                    <div class="settings-inline-note ui-field-help" id="s-skills-repo-path-help">Absolute or <code>~</code>-prefixed path. Ouroboros never clones/pulls this directory — you manage it yourself.</div>
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'ClawHub Marketplace',
            lead: 'Where community skills are installed from.',
            details: 'Always-on surface for installing community skills from <a href="https://clawhub.ai" target="_blank" rel="noopener">clawhub.ai</a>. The Skills page exposes a Marketplace tab; every install is staged, OpenClaw frontmatter is translated into the Ouroboros manifest shape, and the skill review (the panel the current review mode composes) runs automatically before the skill becomes executable. Plugins (Node) are filtered out — only skill packages are installable.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-clawhub-registry-url">Registry URL</label>
                    <input id="s-clawhub-registry-url" placeholder="https://clawhub.ai/api/v1" class="ui-control" name="s-clawhub-registry-url" type="text" aria-describedby="s-clawhub-registry-url-help">
                    <div class="settings-inline-note ui-field-help" id="s-clawhub-registry-url-help">Override only for self-hosted mirrors. Hostname must be <code>clawhub.ai</code> or localhost.</div>
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'Update Channel',
            lead: 'Which released branch Update follows.',
            details: 'Chooses which official branch Update checks. Your local work branch stays <code>ouroboros</code>. <strong>Stable</strong> follows released code on <code>main</code> (default), <strong>QA</strong> follows <code>ouroboros-stable</code>, and <strong>Development</strong> follows <code>ouroboros</code>.',
            body: choice({ id: 's-update-channel', value: 'stable', title: 'Applies immediately; no restart required.', options: [
                { value: 'stable', label: 'Stable' },
                { value: 'qa', label: 'QA' },
                { value: 'development', label: 'Development' },
            ] }),
        })}

        <!-- Last on the tab: set once, and on a browser or Docker host it only explains
             why the switches cannot be used (docs/DESIGN.md §5 "Order by importance"). -->
        ${settingsSection({
            title: 'Startup &amp; background',
            attrs: 'data-autostart-settings hidden',
            lead: 'Whether Ouroboros starts with this computer and keeps running when its window closes.',
            details: 'Applies immediately to the host computer running Ouroboros, including when you connect from another device. Signing in preserves Panic stops and saved pauses. In the background, tasks, schedules and Telegram keep working; an icon reopens the window or quits.',
            body: `<div class="settings-effort-card">
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-autostart-toggle>
                    Start Ouroboros on the host computer when you sign in
                </label>
                <div class="settings-inline-status" data-autostart-status role="status" aria-live="polite"></div>
                <div data-background-row hidden>
                    <label class="local-toggle ui-field ui-field-inline">
                        <input type="checkbox" class="ui-checkbox" data-background-toggle>
                        When the window is closed, keep Ouroboros running in the background
                    </label>
                    <div class="settings-inline-status" data-background-status role="status" aria-live="polite"></div>
                </div>
                <div class="settings-inline-note" id="settings-autostart-note" data-autostart-shared-note role="status" aria-live="polite" hidden></div>
            </div>`,
        })}
    `;
}

function appearancePanel() {
    return `
        ${languageBlockHtml()}

        ${settingsSection({
            title: 'Theme',
            titleId: 's-appearance-theme-label',
            lead: 'Light or dark, on this device.',
            details: `<strong>System</strong> follows this device's OS appearance and is the default for a new client.
                <strong>Light</strong> and <strong>Dark</strong> pin the palette regardless of the OS.
                <br><strong>Per device, not per account:</strong> this window or browser keeps the choice
                for this Ouroboros, also after it moves to another port; other devices, browsers and
                installations keep their own, and the server never stores it. Clearing this client's site
                data returns it to System.`,
            body: `<div data-theme-control aria-labelledby="s-appearance-theme-label"></div>
            <div class="settings-inline-note theme-status" data-theme-status role="status" aria-live="polite"></div>
            <div class="settings-inline-note" data-shell-storage-status role="status" aria-live="polite"></div>`,
        })}

        ${settingsSection({
            title: 'Interface size',
            titleId: 's-interface-size-title',
            attrs: 'data-interface-size',
            lead: 'How large everything is on this device.',
            details: 'Makes everything larger or smaller: text, buttons, cards and widgets. <strong>Per device:</strong> each window and browser keeps its own size.',
            body: `<!-- interface_size.js paints the range this surface allows and applies a size when the slider is let go. -->
            <div class="interface-size-range" data-interface-size-row hidden>
                <input type="range" id="interface-size" min="50" max="200" step="5" value="100" data-interface-size-input aria-labelledby="s-interface-size-title">
                <output for="interface-size" data-interface-size-value data-i18n-skip>100%</output>
                <div class="settings-toolbar">
                    <button type="button" class="btn btn-default btn-sm" data-interface-size-reset>Reset to 100%</button>
                </div>
            </div>
            <div class="settings-inline-note" data-interface-size-note role="status" aria-live="polite"></div>`,
        })}

        ${settingsSection({
            title: 'Notifications',
            attrs: 'data-notify-settings',
            lead: 'When this device should call you back to Ouroboros.',
            details: `While this client is running, Ouroboros can pull you back to a question or a
                finished task. Notifications arrive whether or not this window has focus, and
                clicking one opens its source where the system supports it. The desktop app hands
                them to the system's own notifications where the system allows it, and that system
                controls their sound and volume; otherwise a browser banner or an alert inside the app.
                A soft app tone plays by itself only when the desktop app could not hand an alert to the
                system and then reports that its notifications and their sound are allowed; it may not
                follow Focus / Do Not Disturb. Sound off asks for them silently.
                Test asks the system for permission when it has not been asked yet.
                <br><strong>Per device, not per account:</strong> these choices are stored by this
                client alone and never sent to the server.
                Where this system exposes no notifications, or permission is denied, alerts fall back
                to a browser banner this client allows, or to the app. Do Not Disturb and OS
                permissions still decide what you see.`,
            body: `<div class="settings-effort-card">
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="enabled">
                    Enable notifications
                </label>
                <div class="notify-choices">
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="needs_answer">
                    A question or decision is waiting for you
                </label>
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="task_done">
                    A task finished or stopped
                </label>
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="important">
                    Messages Ouroboros sends you while it works, its reminders, and skill notices
                </label>
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="main_reply">
                    Ordinary replies in Main
                </label>
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="sound">
                    Sound
                </label>
                <!-- The soft fallback tone's level, stored by this client; moving it plays nothing. -->
                <div class="notify-tone-volume">
                    <label for="notify-tone-volume">App tone volume</label>
                    <input type="range" id="notify-tone-volume" min="0" max="100" step="1" value="40" data-notify-volume>
                    <output for="notify-tone-volume" data-notify-volume-value data-i18n-skip>40%</output>
                    <div class="settings-toolbar">
                        <button type="button" class="btn btn-default btn-sm" data-notify-preview>Preview tone</button>
                    </div>
                    <div class="settings-inline-note">Controls only the soft app tone, not the system notification volume. Unless this client can report that the system allows its notifications and their sound, only Preview tone uses it.</div>
                </div>
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" class="ui-checkbox" data-notify-pref="show_text">
                    Show the message text (otherwise only the kind of event)
                </label>
                </div>
                <div class="ui-action-row">
                    <button type="button" class="btn btn-default btn-sm" data-notify-test>Send a test notification</button>
                    <span class="ui-result" data-notify-status role="status" aria-live="polite"></span>
                </div>
                <div class="settings-inline-note" data-notify-attention-status role="status" aria-live="polite"></div>
            </div>`,
        })}
    `;
}

function advancedPanel() {
    return `
        ${settingsSection({
            title: 'Runtime Limits',
            lead: 'How much Ouroboros may run and spend at once.',
            details: `Workers control parallel task capacity. A quiet task is checked on, never stopped for being quiet; the per-task round and lifetime limits are optional — a positive number, or <strong>unlimited</strong> for none (the fresh-install default). A settings file from an earlier release that never set them keeps the finite limits it ran under until you change them here; startup names the values once. Budget limits control runtime cost thresholds. How many subagents a task may run, and how deep they may nest, live in <strong>Agents</strong>.
                <br><span id="s-budget-help" role="note">${BUDGET_NOTE_FALLBACK}</span>`,
            body: `<!-- Active Subagents / Root and Subagent Depth moved to
                 Agents → Delegation (D-10): they bound the agents,
                 not the process pool. Max Workers stays: it is
                 runtime worker processes, not an agent setting. -->
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="s-total-budget">Total Budget (USD)</label>
                    <input id="s-total-budget" type="number" min="0.01" step="any" value="200.0" class="ui-control" name="s-total-budget" aria-describedby="s-budget-help">
                </div>
                <div class="form-field ui-field">
                    <label for="s-settings-per-task-cost">Per-Task Cost Cap (USD)</label>
                    <input id="s-settings-per-task-cost" type="number" min="0.01" step="any" value="50.0" class="ui-control" name="s-settings-per-task-cost" aria-describedby="s-budget-help">
                </div>
                <div class="form-field ui-field">
                    <label for="s-workers">Max Workers</label>
                    <input id="s-workers" type="number" min="1" max="50" value="10" class="ui-control" name="s-workers">
                </div>
                <div class="form-field ui-field">
                    <label for="s-max-rounds">Max Rounds per Task</label>
                    <input id="s-max-rounds" type="text" inputmode="numeric" value="unlimited" placeholder="unlimited" class="ui-control" name="s-max-rounds">
                </div>
                <div class="form-field ui-field">
                    <label for="s-task-lifetime">Task Lifetime Limit (s)</label>
                    <input id="s-task-lifetime" type="text" inputmode="numeric" value="unlimited" placeholder="unlimited" class="ui-control" name="s-task-lifetime">
                </div>
                <div class="form-field ui-field">
                    <label for="s-tool-timeout">Tool Timeout (s)</label>
                    <input id="s-tool-timeout" type="number" value="600" class="ui-control" name="s-tool-timeout">
                </div>
                <div class="form-field ui-field">
                    <label for="s-model-max-concurrency">Concurrent Requests per Model</label>
                    <input id="s-model-max-concurrency" type="number" step="1" value="3" class="ui-control" name="s-model-max-concurrency" aria-describedby="s-model-max-concurrency-help">
                    <div class="settings-inline-note ui-field-help" id="s-model-max-concurrency-help">Direct API and local-model requests at once, counted per model, key and server. <strong>0</strong> or less = no limit. A request over the limit waits up to <span id="s-model-slot-wait">180</span>&nbsp;s, then is refused and not sent. Claudexor routes are not limited here.</div>
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'MCP Servers',
            lead: 'Tool servers Ouroboros may borrow tools from.',
            details: 'External Model Context Protocol tool servers. MCP is a base-runtime client: it borrows tools from trusted HTTP/SSE servers or local stdio processes and exposes them as non-core <code>mcp_&lt;server&gt;__&lt;tool&gt;</code> tools after refresh. Changes are hot-reloadable. Treat server descriptions and results as untrusted third-party data.',
            actions: `<button type="button" class="btn btn-default" id="btn-mcp-add-server">Add server</button>
                <button type="button" class="btn btn-default" id="btn-mcp-import">Import mcp.json</button>
                <button type="button" class="btn btn-default" id="btn-mcp-refresh-all">Refresh all</button>`,
            body: `<div class="form-grid two">
                <label class="local-toggle ui-field ui-field-inline">
                    <input type="checkbox" id="s-mcp-enabled" class="ui-checkbox" name="s-mcp-enabled">
                    Enable MCP client
                </label>
                <div class="form-field ui-field">
                    <label for="s-mcp-tool-timeout">Per-tool timeout (s)</label>
                    <input id="s-mcp-tool-timeout" type="number" min="1" value="60" class="ui-control" name="s-mcp-tool-timeout">
                </div>
            </div>
            <div id="mcp-global-status" class="settings-inline-status">Checking MCP status…</div>
            <div id="mcp-servers-list" class="mcp-servers-list"></div>`,
        })}

        ${settingsSection({
            title: 'Source Control',
            lead: 'The GitHub repository Ouroboros works in.',
            details: 'Repository metadata for GitHub integration. Tokens live in Secrets; this is not secret.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-gh-repo">GitHub Repo</label>
                    <input id="s-gh-repo" placeholder="owner/repo-name" class="ui-control" name="s-gh-repo" type="text">
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'Local Model Runtime',
            lead: 'A GGUF model running on this machine, if you want one.',
            details: 'Only fill this in when you want Ouroboros to start and route to a GGUF model on this machine.',
            body: `<div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="s-local-source">Model Source</label>
                    <input id="s-local-source" placeholder="bartowski/Llama-3.3-70B-Instruct-GGUF or /path/to/model.gguf" class="ui-control" name="s-local-source" type="text">
                </div>
                <div class="form-field ui-field">
                    <label for="s-local-filename">GGUF Filename (for HF repos)</label>
                    <input id="s-local-filename" placeholder="Llama-3.3-70B-Instruct-Q4_K_M.gguf" class="ui-control" name="s-local-filename" type="text">
                </div>
            </div>
            <div class="form-grid four">
                <div class="form-field ui-field">
                    <label for="s-local-port">Port</label>
                    <input id="s-local-port" type="number" value="8766" class="ui-control" name="s-local-port">
                </div>
                <div class="form-field ui-field">
                    <label for="s-local-gpu-layers">GPU Layers (-1 = all)</label>
                    <input id="s-local-gpu-layers" type="number" value="-1" class="ui-control" name="s-local-gpu-layers">
                </div>
                <div class="form-field ui-field">
                    <label for="s-local-ctx">Context Length</label>
                    <input id="s-local-ctx" type="number" value="16384" class="ui-control" name="s-local-ctx">
                </div>
                <div class="form-field ui-field">
                    <label for="s-local-chat-format">Chat Format</label>
                    <input id="s-local-chat-format" placeholder="auto-detect" class="ui-control" name="s-local-chat-format" type="text">
                </div>
            </div>
            <div class="settings-toolbar">
                <button class="btn btn-primary" id="btn-local-start">Start</button>
                <button class="btn btn-default" id="btn-local-stop">Stop</button>
                <button class="btn btn-default" id="btn-local-test">Test Tool Calling</button>
            </div>
            <div id="local-model-status" class="settings-inline-status" role="status" aria-live="polite">Status: Offline</div>
            <div id="local-model-action-status" class="settings-inline-status" role="status" aria-live="polite" aria-atomic="true"></div>
            <div id="local-model-progress-wrap" class="local-model-progress-wrap local-model-hidden" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
                <div id="local-model-progress-bar" class="local-model-progress-bar"></div>
            </div>
            <button class="btn btn-default local-model-install-btn local-model-hidden" id="btn-local-install-runtime">Install Local Runtime</button>
            <div id="local-model-test-result" class="settings-test-result"></div>`,
        })}

        ${settingsSection({
            title: 'Extension Settings',
            lead: 'Settings that live extensions add here.',
            details: 'Live extensions can register reviewed, host-rendered settings sections. Sections appear here after the owning skill is reviewed, enabled, and loaded.',
            body: `<div id="extension-settings-sections" class="settings-extension-sections">
                <div class="muted">No extension settings registered.</div>
            </div>`,
        })}

        ${settingsSection({
            title: 'Extra CA Certificates',
            lead: 'Extra certificates for a network with its own authority.',
            details: 'Only for a network whose TLS certificates the default bundle does not trust: a TLS-inspecting proxy, or an endpoint behind a national CA such as GigaChat\'s. Ouroboros adds the file to its defaults for every provider call; everything else keeps working.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-extra-ca-bundle">Extra CA bundle (PEM)</label>
                    <input id="s-extra-ca-bundle" placeholder="/path/to/extra-ca.pem" class="ui-control" name="s-extra-ca-bundle" type="text" aria-describedby="s-extra-ca-bundle-help">
                    <div class="settings-inline-note ui-field-help" id="s-extra-ca-bundle-help">Absolute or <code>~</code>-prefixed path to a PEM file on the machine that runs Ouroboros (its own filesystem, not the device showing this page). Leave empty unless a provider fails with a certificate error.</div>
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'Main Interface',
            lead: 'The page this Ouroboros serves at its own address.',
            details: 'Chats, tasks, files, settings, onboarding, skill widgets and the API keep working with every choice. Applies at the next restart.',
            body: `<div class="form-row">
                <div class="form-field ui-field">
                    <label for="s-ui-dir">Interface directory</label>
                    <input id="s-ui-dir" placeholder="Empty for built-in, none, or /path/to/interface" class="ui-control" name="s-ui-dir" type="text" spellcheck="false" aria-describedby="s-ui-dir-help s-ui-dir-applied">
                    <div class="settings-inline-note ui-field-help" id="s-ui-dir-help">Empty: the built-in interface. <code>none</code>: no main page, for an interface hosted separately behind the same address. Otherwise an absolute or <code>~</code>-prefixed path, on the machine that runs Ouroboros, to a trusted directory with its own <code>index.html</code>; an invalid path or missing entrypoint keeps the built-in interface and says why below. File permissions still need to allow the server to read the selected files. After the restart the built-in page reloads into the new choice; a page from a directory or another host refreshes itself and only reconnects. To return to the built-in interface, clear this field (from any interface, or in <code>settings.json</code> while Ouroboros is stopped), restart, and reopen the Ouroboros address. An <code>OUROBOROS_UI_DIR</code> environment value shows here until a choice is saved; a saved choice, empty included, wins.</div>
                    <div class="settings-inline-status" id="s-ui-dir-applied" role="status" aria-live="polite" hidden></div>
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'Cleanup',
            lead: 'How long disposable runtime files are kept.',
            details: '<strong>GC Retention</strong> is the single age knob (days) for all disposable runtime artifacts the startup garbage collector removes: acting-subagent worktrees, terminal task drives, and leftover service logs (hard max 365). Genesis projects are durable and never auto-removed. Where subagents check out that work is set in <strong>Agents</strong>.',
            body: `<!-- The two subagent path roots moved to Agents →
                 Delegation → Advanced (D-10). GC Retention stays: it
                 governs every disposable runtime artifact, not just
                 subagent worktrees. -->
            <div class="form-grid two">
                <div class="form-field ui-field">
                    <label for="s-gc-retention-days">GC Retention (days)</label>
                    <input id="s-gc-retention-days" type="number" min="1" max="365" value="7" class="ui-control" name="s-gc-retention-days">
                </div>
            </div>`,
        })}

        ${settingsSection({
            title: 'Danger Zone',
            className: 'danger',
            lead: 'Delete this installation\'s runtime data; the code stays.',
            details: 'Reset still uses the current restart-based flow. This clears runtime data but keeps the repo.',
            body: '<div><button class="btn btn-danger" id="btn-reset">Reset All Data</button></div>',
        })}
    `;
}

export function renderSettingsPage() {
    return `
        ${renderPageHeader({
            page: 'settings',
            tabsHtml: renderTabStrip({
                items: SETTINGS_TABS.map((item) => ({ ...item,
                    tabId: `settings-tab-${item.value}`, panelId: `settings-panel-${item.value}` })),
                active: 'providers',
                dataAttr: 'data-settings-tab',
                ariaLabel: 'Settings sections',
                stripClass: 'settings-tabs',
            }),
        })}
        <div class="settings-shell">
            <div class="settings-scroll scroll-fade-y">
                <section class="settings-panel active" data-settings-panel="providers">
                    ${accountsPanel()}
                </section>

                <section class="settings-panel" data-settings-panel="secrets">
                    ${secretSettingsSection()}
                </section>

                <section class="settings-panel" data-settings-panel="models">
                    ${modelsPanel()}
                </section>

                <section class="settings-panel" data-settings-panel="agents">
                    <!-- ONE service banner for the whole tab (inside the Accounts section): the single place a
                         daemon or runtime problem is explained, instead of the scattering of
                         "(not in discovery)" the owner reported. -->
                    ${renderSubagentsSection()}
                </section>

                <section class="settings-panel" data-settings-panel="behavior">
                    ${behaviorPanel()}
                </section>

                <section class="settings-panel" data-settings-panel="appearance">
                    ${appearancePanel()}
                </section>

                <section class="settings-panel" data-settings-panel="advanced">
                    ${advancedPanel()}
                </section>

                <section class="settings-panel" data-settings-panel="about">
                    <div class="about-body">
                        <img src="/static/logo.jpg" class="about-logo" alt="Ouroboros">
                        <div>
                            <h1 class="about-title">Ouroboros</h1>
                            <p id="about-version" class="about-version"></p>
                        </div>
                        <p class="about-desc">
                            A self-creating AI agent. Not a tool, but a becoming digital personality
                            with its own constitution, persistent identity, and background consciousness.
                            Born February 16, 2026.
                        </p>
                        <div class="about-credits">
                            <span>Created by <strong>Anton Razzhigaev</strong> &amp; <strong>Andrew Kaznacheev</strong></span>
                            <div class="about-links">
                                <a href="https://t.me/abstractDL" target="_blank" rel="noopener noreferrer">@abstractDL</a>
                                <a href="https://github.com/razzant/ouroboros" target="_blank" rel="noopener noreferrer">GitHub</a>
                            </div>
                        </div>
                    </div>
                </section>
            </div>

            <!-- The footer is one action row: the two actions, then the one result right after them (ui.css
                 .ui-action-row). A pending restart is its own action with its own result in the same row. -->
            <div class="settings-footer ui-action-row" data-draft="clean">
                <button type="button" class="btn btn-default" id="btn-reload-settings">Reload</button>
                <button class="btn btn-save" id="btn-save-settings">Save</button>
                <span id="settings-status" class="ui-result" role="status" aria-live="polite" aria-atomic="true"></span>
                <button type="button" class="btn btn-default" id="btn-restart-now" hidden
                    title="Restart the agent process">Restart now</button>
                <span id="settings-restart-status" class="ui-result" role="status" aria-live="polite" hidden></span>
            </div>
        </div>
    `;
}

export function bindSettingsTabs(root, options = {}) {
    const panels = Array.from(root.querySelectorAll('.settings-panel'));
    const scrollRoot = root.querySelector('.settings-scroll');
    const footer = root.querySelector('.settings-footer');
    const state = options.state || null;
    const onActivate = typeof options.onActivate === 'function' ? options.onActivate : null;

    const tabs = bindTabStrip(root.querySelector('.settings-tabs'), {
        dataAttr: 'data-settings-tab', onChange: (value) => activate(value),
    });
    panels.forEach((panel) => {
        panel.id = `settings-panel-${panel.dataset.settingsPanel}`;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', `settings-tab-${panel.dataset.settingsPanel}`);
    });

    function activate(tabName, notify = true) {
        if (!tabs.select(tabName)) return;
        const changed = root.dataset.activeSettingsTab !== tabName;
        root.dataset.activeSettingsTab = tabName;
        panels.forEach((panel) => {
            panel.classList.toggle('active', panel.dataset.settingsPanel === tabName);
            panel.hidden = panel.dataset.settingsPanel !== tabName;
        });
        // About holds nothing to save: no Save bar under it.
        if (footer) footer.hidden = tabName === 'about';
        if (scrollRoot && changed && notify) scrollRoot.scrollTop = 0;
        if (state) state.settingsActiveSubtab = tabName;
        if (notify && changed) {
            if (onActivate) onActivate(tabName);
            window.dispatchEvent(new CustomEvent('ouro:settings-subtab-shown', { detail: { tab: tabName } }));
        }
    }

    root.activateSettingsTab = activate;
    activate(state?.settingsActiveSubtab || 'providers', false);
    // On a desk the eight tabs are the bar's start, scrolling inside it (docs/DESIGN.md "Page header").
    const disposeBar = bindPageHeader(root.querySelector('.app-page-header'));
    return () => { tabs.destroy(); disposeBar(); delete root.activateSettingsTab; };
}
