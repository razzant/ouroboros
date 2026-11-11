/** The section builders of the old Settings tabs, kept unmounted while the lanes of the rework recompose them into
 * pages (settings_pages.js): the shell in settings.js no longer draws any of them. */
import { renderSegmentedField } from './page_header.js';
import { renderAgentAccountsSection } from './harness_accounts.js';
import { languageBlockHtml } from './settings_language.js';
import { safetySectionHtml } from './settings_safety.js';
import { consciousnessSectionHtml } from './settings_consciousness.js';
import { PROVIDER_CARDS, providerSettingsCard } from './settings_provider_cards.js';
import { settingsSection } from './settings_controls.js';
import { apiFetch } from './api_client.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { showToast } from './toast.js';
import { copyTextWithToast } from './ui_helpers.js';
import { escapeHtmlAttr as escapeHtml } from './utils.js';

// The setup contract replaces this after the settings read. Keep the accessible
// description useful while that read is pending or unavailable.
const BUDGET_NOTE_FALLBACK = 'Budget limits use known spend: Total Budget applies globally, while Per-task Cost Cap applies to one root task tree. Unpriced calls remain unknown and are not counted as $0; reservations and unresolved charges are not counted as spending.';

// Runtime mode is one axis of the owner policy contract. Keep the Settings
// presentation in the same vocabulary as the onboarding setup contract; the
// saved value is still handled by settings.js and the owner endpoint.
export const RUNTIME_MODE_OPTIONS = [
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
function choice({ id, value, options, modifier = '' }) {
    return `<input id="${id}" type="hidden" value="${value}">${renderSegmentedField({ target: id, modifier, options })}`;
}

const policyLine = (key) => `<div class="settings-inline-note" data-policy-state="${key}" role="status" aria-live="polite"></div>`;

export function secretSettingsSection() {
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

export function accountsPanel() {
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
                    <div class="settings-inline-note" data-legacy-base-url-fixed hidden></div>
                </div>
            </div>`,
        })}
    `;
}

export function behaviorPanel() {
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
            details: 'Auto reviews a result when Ouroboros requests it. Required also reviews what a task actually produced, and an answer its task requires checked. With Advisory, Ouroboros decides after a review whether a revised result needs another; with Blocking, a reviewed result that did not pass is held until a review passes it. Off starts no new task-result review, including manual or historical requests; already running reviews can still return feedback. Waiting works in every access mode; Cyber feedback remains advisory.',
            body: choice({ id: 's-task-review-mode', value: 'auto', options: [
                { value: 'off', label: 'Off' },
                { value: 'auto', label: 'Auto' },
                { value: 'required', label: 'Required' },
            ] }),
        })}

        ${settingsSection({
            title: 'Max Review Cycles',
            lead: 'How many paid review rounds one change may take.',
            details: 'Limits paid review waves, including dispatched technical failures: result review and commit review per root task, and skill review per root task or manual snapshot. The last review still permits author corrections within ordinary task limits; explicit task-local author limits remain separate. Collection and exact replay are free. Advisory allows an explicit decision after receiving feedback or a disclosed unavailable result; Blocking still requires reviewer approval outside Cyber Pro. <strong>&infin;</strong> removes the count cap, while deadlines, budgets and lifecycle limits still apply.',
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
                options: RUNTIME_MODE_OPTIONS })}
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
            body: choice({ id: 's-context-mode', value: 'max', options: [
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
            body: choice({ id: 's-update-channel', value: 'stable', options: [
                { value: 'stable', label: 'Stable' },
                { value: 'qa', label: 'QA' },
                { value: 'development', label: 'Development' },
            ] }) + '<div class="settings-inline-note" data-update-channel-fixed hidden>Set by your organization: its administrator updates this installation.</div>',
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

export function appearancePanel() {
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

export function advancedPanel() {
    return `
        ${settingsSection({
            title: 'Runtime Limits',
            lead: 'How much Ouroboros may run and spend at once.',
            details: `Workers: memory admits new project processes; an optional number caps acting tasks. Sleeping tasks yield capacity, and warm returns may exceed that cap. Conversation runs separately. A quiet task is checked on, never stopped for being quiet; the per-task round and lifetime limits are optional — a positive number, or <strong>unlimited</strong> for none (the fresh-install default). A settings file from an earlier release that never set them keeps the finite limits it ran under until you change them here; startup names the values once. Budget limits control runtime cost thresholds. How many subagents a task may run, and how deep they may nest, live in <strong>Agents</strong>.
                <br><span id="s-budget-help" role="note">${BUDGET_NOTE_FALLBACK}</span>`,
            body: `<!-- The acting-task cap lives in Runtime Limits;
                 per-root child and depth limits live in Agents. -->
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
                    <input id="s-workers" type="number" min="1" step="1" placeholder="memory" class="ui-control" name="s-workers" aria-describedby="s-workers-help">
                    <div class="settings-inline-note ui-field-help" id="s-workers-help">Empty: the limit is this machine's memory; a new project process starts while one more fits. A number caps acting tasks, excluding accepted waits and sleeps. Warm returns can exceed it.</div>
                </div>
                <div class="form-field ui-field">
                    <label for="s-max-rounds">Max Rounds per Task</label>
                    <input id="s-max-rounds" type="text" inputmode="numeric" value="unlimited" placeholder="unlimited" class="ui-control" name="s-max-rounds">
                </div>
                <div class="form-field ui-field">
                    <label for="s-task-lifetime">Task Lifetime Limit (s)</label>
                    <input id="s-task-lifetime" type="text" inputmode="numeric" value="unlimited" placeholder="unlimited" class="ui-control" name="s-task-lifetime">
                </div>
            </div>`,
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

// The static About content; utils.js fills `#about-version` wherever it stands.
export function aboutPanel() {
    return `
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
    `;
}

// What the core serves at its own address right now (`served_ui` on /api/state),
// beside the saved choice that applies at the next restart; a refused choice
// carries the core's own sentence naming it and why. A core older than the choice
// reports nothing, and nothing is shown.
export function servedUiStatus(servedUi) {
    const label = { builtin: 'the built-in interface', directory: 'the interface directory', none: 'no main page' }[servedUi?.source];
    if (!label) return null;
    return servedUi.error
        ? { text: `Serving now: ${label}. ${servedUi.error}`, tone: 'warn' }
        : { text: `Serving now: ${label}.`, tone: 'muted' };
}

// The hint under the bind host (`#settings-lan-hint`), from a settings read's `_meta`.
export function paintNetworkHint(hint, meta) {
    if (!hint || !meta) return;
    if (meta.reachability === 'loopback_only') {
        hint.innerHTML = 'Bound to localhost: only this machine can reach Ouroboros. Set Server Bind Host to <code>0.0.0.0</code>, save, and restart for LAN access.';
        hint.dataset.tone = 'info';
        hint.hidden = false;
    } else if (meta.reachability === 'lan_reachable') {
        const url = escapeHtml(meta.recommended_url || '');
        const warning = escapeHtml(meta.warning || '');
        // Status first, action right (ui.css .ui-action-row). The link keeps its own words; the button
        // carries the address this read drew, so a click after any later read copies that read's address.
        const copy = url ? `<button type="button" class="btn btn-default btn-sm ui-hit-area" data-copy-lan-url="${url}">Copy link</button>` : '';
        hint.innerHTML = `<div class="ui-action-row"><span>LAN URL: <a href="${url}" target="_blank" rel="noopener">${url}</a>${warning ? ' — <strong>' + warning + '</strong>' : ''}</span>${copy}</div>`;
        hint.dataset.tone = meta.warning ? 'warn' : 'ok';
        hint.hidden = false;
    } else if (meta.reachability === 'host_ip_unknown') {
        const url = escapeHtml(meta.recommended_url || '');
        const warning = escapeHtml(meta.warning || '');
        hint.innerHTML = `Server is listening on non-localhost but LAN IP could not be detected automatically. Try <code>${url}</code>.${warning ? ' <strong>' + warning + '</strong>' : ''}`;
        hint.dataset.tone = 'warn';
        hint.hidden = false;
    } else {
        hint.hidden = true;
    }
}

// Every read rebuilds the hint, so one listener lives as long as the hint and reads the address off the
// button that was clicked; the shared copy owns the fallback and reports through the toast host.
export function bindNetworkHintCopy(hint, { copy = copyTextWithToast } = {}) {
    const onClick = (event) => {
        const url = event.target?.closest?.('[data-copy-lan-url]')?.dataset.copyLanUrl;
        if (url) void copy(url, { okMessage: 'LAN URL copied.' });
    };
    hint?.addEventListener('click', onClick);
    return () => hint?.removeEventListener('click', onClick);
}

// Reset All Data (the Danger Zone's button): asks first, then deletes the runtime data and restarts.
export async function resetAllData() {
    const confirmedReset = await openConfirmDialog({
        title: 'Reset runtime data',
        body: 'This will delete all runtime data (state, memory, logs, settings) and restart.\nThe repo (agent code) will be preserved.\nYou will need to re-enter your provider settings.\n\nContinue?',
        confirmLabel: 'Delete and restart',
        danger: true,
    });
    if (!confirmedReset) return;
    try {
        const res = await apiFetch('/api/reset', { method: 'POST' });
        const data = await res.json();
        if (data.status === 'ok') {
            await openConfirmDialog({
                title: 'Reset complete',
                body: 'Deleted: ' + (data.deleted.join(', ') || 'nothing') + '\nRestarting...',
                alert: true,
            });
        } else {
            await openConfirmDialog({
                title: 'Reset failed',
                body: 'Error: ' + (data.error || 'unknown'),
                alert: true,
            });
        }
    } catch (e) {
        showToast('Reset failed: ' + e.message, 'error');
    }
}
