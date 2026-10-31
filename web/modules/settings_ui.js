import { bindPageHeader, renderPageHeader, renderSegmentedField, renderTabStrip, bindTabStrip } from './page_header.js';
import { renderAgentAccountsSection, renderAgentsServiceBanner } from './harness_accounts.js';
import { renderSubagentsSection } from './subagents_settings.js';
import { languageBlockHtml } from './settings_language.js';
import { modelRolesHost } from './model_roles.js';
import { safetySectionHtml } from './settings_safety.js';
import { consciousnessSectionHtml } from './settings_consciousness.js';
import { PROVIDER_CARDS, providerSettingsCard } from './settings_provider_cards.js';

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

// Provider API keys have one input each, in Accounts; this tab keeps the other stored secrets.
export const SECRET_KEYS = [
    ['GITHUB_TOKEN', 'GitHub Token', 'ghp_...'],
    ['OUROBOROS_NETWORK_PASSWORD', 'Network Password', 'Required for LAN/Docker binds'],
];

function secretSettingsSection() {
    return `
        <section class="settings-card">
            <h3>Stored Secrets</h3>
            <div class="settings-section-copy">
                Tokens, passwords and the secrets skills request. Provider API keys are in Accounts.
                Skills only receive grant-only keys after explicit human approval.
            </div>
            <div class="form-grid two">
                ${SECRET_KEYS.map(([key, label, placeholder]) => secretField({
                    id: `s-secret-${key.toLowerCase().replace(/_/g, '-')}`,
                    settingKey: key,
                    label,
                    placeholder,
                })).join('')}
            </div>
        </section>
        <section class="settings-card">
            <h3>Requested By Skills</h3>
            <div class="settings-section-copy">
                Secrets requested by installed skills appear here only when a skill asks for them.
            </div>
            <div id="skill-requested-secrets" class="settings-secret-list">
                <div class="muted">No skill-requested secrets.</div>
            </div>
        </section>
        <section class="settings-card">
            <div class="settings-card-head">
                <div>
                    <h3>Custom Keys</h3>
                    <div class="settings-section-copy">
                        Optional key/value storage for future skills. Use uppercase names such as <code>SLACK_WEBHOOK_URL</code>.
                    </div>
                </div>
                <button type="button" class="btn btn-default btn-sm" id="btn-add-custom-secret">Add custom key</button>
            </div>
            <div id="custom-secrets-list" class="settings-secret-list settings-custom-secret-list"></div>
        </section>
    `;
}

export function renderSettingsPage() {
    return `
        ${renderPageHeader({
            page: 'settings',
            tabsHtml: `
                <div class="settings-tabs-bar">
                    ${renderTabStrip({
                        items: SETTINGS_TABS.map((item) => ({ ...item,
                            tabId: `settings-tab-${item.value}`, panelId: `settings-panel-${item.value}` })),
                        active: 'providers',
                        dataAttr: 'data-settings-tab',
                        ariaLabel: 'Settings sections',
                        stripClass: 'settings-tabs',
                        tabClass: 'settings-tab',
                    })}
                </div>
            `,
        })}
        <div class="settings-shell">
            <div class="settings-scroll scroll-fade-y">
                <section class="settings-panel active" data-settings-panel="providers">
                    <div class="settings-section-copy">
                        Connect subscriptions and API keys here, then choose their roles in Models and Agents.
                        Adding an account keeps your existing assignments.
                    </div>
                    ${renderAgentsServiceBanner()}
                    ${renderAgentAccountsSection()}
                    <h3>API keys</h3>
                    <div class="settings-section-copy">Each role names the key it uses; keys never rotate. For a second key of the same model, add it as a fallback.</div>
                    ${PROVIDER_CARDS.filter((card) => !card.advanced).map(providerSettingsCard).join('')}
                    <details class="settings-more-providers" id="settings-more-providers">
                        <summary>
                            <span class="settings-provider-title"><span>More providers</span></span>
                            <span class="settings-provider-hint">Cloud.ru Foundation Models, MiniMax, DeepSeek, and GigaChat</span>
                        </summary>
                        <div class="settings-more-providers-body">
                            ${PROVIDER_CARDS.filter((card) => card.advanced).map(providerSettingsCard).join('')}
                        </div>
                    </details>
                    <div class="form-section compact">
                        <h3>Network Gate</h3>
                        <div class="form-row">${secretField({
                            id: 's-network-password',
                            settingKey: 'OUROBOROS_NETWORK_PASSWORD',
                            label: 'Network Password (optional)',
                            placeholder: 'Leave blank to keep the network surface open',
                        })}</div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-server-host">Server Bind Host</label>
                                <input id="s-server-host" placeholder="127.0.0.1 or 0.0.0.0" class="ui-control" name="s-server-host" type="text" aria-describedby="s-server-host-help">
                                <div class="settings-inline-note ui-field-help" id="s-server-host-help">Use <code>127.0.0.1</code> for this machine only. Use <code>0.0.0.0</code> for LAN/Docker access with a Network Password in the same save. Specific LAN IP binds are manual/env-only.</div>
                            </div>
                        </div>
                        <div class="settings-inline-note">Adds a password wall only for non-localhost app and API access. If you expose Ouroboros on LAN or Docker, set a password before sharing the URL.</div>
                        <div id="settings-lan-hint" class="settings-lan-hint" hidden></div>
                    </div>
                    <div class="form-section compact">
                        <h3>Legacy Compatibility</h3>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-openai-base-url">Legacy OpenAI Base URL</label>
                                <input id="s-openai-base-url" placeholder="https://api.openai.com/v1 or compatible endpoint" class="ui-control" name="s-openai-base-url" type="text">
                            </div>
                        </div>
                        <div class="settings-inline-note">Backward-compatibility escape hatch for older installs. For new custom providers, use the dedicated <code>OpenAI Compatible</code> card instead.</div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="secrets">
                    ${secretSettingsSection()}
                </section>

                <section class="settings-panel" data-settings-panel="models">
                    <div class="form-section">
                        <h3>Model Routing</h3>
                        <div class="settings-section-copy">
                            Choose a source, model, and subscription account for each role. Auto rotates
                            compatible accounts. Local uses the runtime configured in Advanced.
                        </div>
                        <div class="settings-action-row">
                            <span id="settings-model-catalog-status" class="settings-inline-status" role="status" aria-live="polite" aria-atomic="true">Model catalog is optional and failure-tolerant.</span>
                            <button type="button" class="btn btn-default" id="btn-refresh-model-catalog">Refresh Model Catalog</button>
                        </div>
                        ${modelRolesHost('settings-model-roles')}
                    </div>

                    <!-- Reviewers and Delegation live in the Agents tab (D-10):
                         they answer "who does the work", not "which API model
                         id". One capability, one section — no control here
                         duplicates one there. -->

                    <div class="form-section">
                        <h3>Web search</h3>
                        <div class="form-grid two">
                            <div class="form-field ui-field">
                                <label for="s-websearch-source">Source</label>
                                <select id="s-websearch-source" class="ui-control"><option value="auto">Auto</option><option value="openai">OpenAI</option><option value="openrouter">OpenRouter</option><option value="anthropic">Anthropic</option><option value="ddgs">ddgs</option></select>
                            </div>
                            <div class="form-field ui-field">
                                <label for="s-websearch-model">Model</label>
                                <input id="s-websearch-model" placeholder="Auto: source defaults" class="ui-control" name="s-websearch-model" type="text" aria-describedby="s-websearch-model-help">
                                <div class="settings-inline-note ui-field-help" id="s-websearch-model-help">Built-in <code>web_search</code> only. Skills, MCP and browser tools remain independent choices. Empty uses source defaults.</div>
                            </div>
                        </div>
                        <div class="ui-field-help" id="s-websearch-preview" role="status"></div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="agents">
                    <div class="settings-section-copy">
                        Configure the subagents and reviewers Ouroboros works with. Subscriptions and
                        API keys are in Accounts; global model roles are in Models.
                    </div>
                    <!-- ONE service banner for the whole tab: the single place a
                         daemon or runtime problem is explained, instead of the
                         scattering of "(not in discovery)" the owner reported. -->
                    ${renderSubagentsSection()}
                </section>

                <section class="settings-panel" data-settings-panel="behavior">
                    <div class="form-section">
                        <h3>Review Enforcement</h3>
                        <div class="settings-section-copy"><code>Advisory</code> keeps review visible but non-blocking. <code>Blocking</code> stops commits and reviewed-skill activation when critical findings remain unresolved, outside Cyber Pro Access.</div>
                        <div class="settings-effort-card">
                            <label>Enforcement Mode</label>
                            <input id="s-review-enforcement" type="hidden" value="advisory">
                            ${renderSegmentedField({
                                target: 's-review-enforcement',
                                modifier: 'data-enforcement-group',
                                options: [
                                    { value: 'advisory', label: 'Advisory' },
                                    { value: 'blocking', label: 'Blocking' },
                                ],
                            })}
                            <div class="settings-inline-note" data-policy-state="review" role="status" aria-live="polite"></div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Task Result Review</h3>
                        <div class="settings-section-copy">Auto reviews a result when Ouroboros requests it. Required also reviews material actually produced or observed effects. Off starts no new task-result review, including manual or historical requests; already running reviews can still return feedback. Waiting works in every access mode; Cyber feedback remains advisory.</div>
                        <div class="settings-effort-card">
                            <label>Task Result Review</label>
                            <input id="s-task-review-mode" type="hidden" value="auto">
                            ${renderSegmentedField({
                                target: 's-task-review-mode',
                                options: [
                                    { value: 'off', label: 'Off' },
                                    { value: 'auto', label: 'Auto' },
                                    { value: 'required', label: 'Required' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Max Review Cycles</h3>
                        <div class="settings-section-copy">Limits paid review waves, including dispatched technical failures: plan and task review per task, commit review per root task, and skill review per root task or manual snapshot. The last review still permits author corrections within ordinary task limits; explicit task-local author limits remain separate. Collection and exact replay are free. Advisory allows an explicit decision after receiving feedback or a disclosed unavailable result; Blocking still requires reviewer approval outside Cyber Pro. <code>&infin;</code> removes the count cap, while deadlines, budgets and lifecycle limits still apply.</div>
                        <div class="settings-effort-card">
                            <label>Max Review Cycles</label>
                            <input id="s-review-max-cycles" type="hidden" value="2">
                            ${renderSegmentedField({
                                target: 's-review-max-cycles',
                                modifier: 'data-review-cycles-group',
                                options: [
                                    { value: '1', label: '1' },
                                    { value: '2', label: '2' },
                                    { value: '3', label: '3' },
                                    { value: '5', label: '5' },
                                    { value: 'unlimited', label: '\u221E' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Access</h3>
                        <div class="settings-section-copy">
                            Separate axis from Review Enforcement. Controls how far Ouroboros is allowed to self-modify.
                            <code>Light</code> blocks repo self-modification but allows reviewed + enabled skills to run.
                            <code>Advanced</code> is the default &mdash; self-modify the evolutionary layer; protected core/contract/release files stay guarded by the shared runtime-mode policy.
                            <code>Pro</code> can edit protected core/contract/release surfaces, but commits still go through the normal review gate; Advanced remains limited to the evolutionary layer.
                            <code>Cyber Pro</code> grants the full host and configuration authority, including credentials, models, Supervisor configuration and protected rewrites. Review still runs and the Review Enforcement choice is kept, but review findings do not block actions in Cyber Pro: they are evidence for Ouroboros, which decides.
                            <br><strong>Human controlled:</strong> desktop builds ask the launcher for native confirmation before saving a mode change.
                            Web/Docker sessions save mode changes through the owner endpoint; the new mode takes effect after restart.
                        </div>
                        <div class="settings-effort-card">
                            <label>Access level</label>
                            <input id="s-runtime-mode" type="hidden" value="advanced">
                            ${renderSegmentedField({
                                target: 's-runtime-mode',
                                modifier: 'data-runtime-mode-group',
                                title: 'Access changes take effect after restart.',
                                options: RUNTIME_MODE_OPTIONS,
                            })}
                            <div class="settings-inline-note" data-policy-state="access" role="status" aria-live="polite"></div>
                        </div>
                    </div>

                    <!-- Allow Mutative Subagents moved to Agents → Delegation: one
                         subagent story in one place, and two controls over one
                         setting would have carried two drafts. -->

                    <div class="form-section">
                        <h3>Safety Supervisor</h3>
                        ${safetySectionHtml()}
                    </div>

                    <div class="form-section">
                        <h3>Post-Task Self-Evolution</h3>
                        <div class="settings-section-copy">
                            After an eligible task, Ouroboros can optionally run one reviewed self-improvement cycle: the worker asks the Main model whether one improvement is worth a cycle now, writes a durable request, and the supervisor starts a one-shot campaign later on an idle tick if all gates pass.
                            <br><strong>Configuration authority:</strong> outside Cyber Pro, only the owner can enable this. Cyber Pro also lets the agent configure it; selecting Cyber Pro does not enable evolution automatically. Changes apply on the next task.
                        </div>
                        <div class="settings-effort-card">
                            <label>Self-Improvement Trigger</label>
                            <input id="s-post-task-evolution-mode" type="hidden" value="off">
                            ${renderSegmentedField({
                                target: 's-post-task-evolution-mode',
                                options: [
                                    { value: 'off', label: 'Off' },
                                    { value: 'llm', label: 'After Each Task (LLM decides)' },
                                    { value: 'every_n', label: 'Every N Tasks' },
                                ],
                            })}
                            <div class="settings-inline-note"><strong>Counts every eligible task, including trivial chats.</strong> <code>Every N=1</code> means Ouroboros considers self-improvement after every task, then runs the actual cycle later on an idle supervisor tick.</div>
                        </div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <div data-evo-every-n-row>
                                <label for="s-evo-cadence-n">Every N Tasks</label>
                                <input id="s-evo-cadence-n" type="number" min="1" step="1" placeholder="3" class="ui-control" name="s-evo-cadence-n" aria-describedby="s-evo-cadence-n-help">
                                <div class="settings-inline-note ui-field-help" id="s-evo-cadence-n-help">Visible only when Self-Improvement Trigger = Every N Tasks.</div>
                                </div>
                            </div>
                            <div class="form-field ui-field">
                                <label for="s-evo-budget">Per-Cycle Budget Reserve (USD)</label>
                                <input id="s-evo-budget" placeholder="0" class="ui-control" name="s-evo-budget" type="text" aria-describedby="s-evo-budget-help">
                                <div class="settings-inline-note ui-field-help" id="s-evo-budget-help">Minimum remaining global budget required to start a post-task cycle. <code>0</code> = rely on the normal gates. Running cycles still inherit the global per-task hard cost cap and the supervisor's reserved-budget floor.</div>
                            </div>
                        </div>
                        <div class="form-field ui-field">
                            <label for="s-evo-objective">Standing Objective (optional)</label>
                            <input id="s-evo-objective" placeholder="(none) — e.g. prioritize test coverage and latency" class="ui-control" name="s-evo-objective" type="text" aria-describedby="s-evo-objective-help">
                            <div class="settings-inline-note ui-field-help" id="s-evo-objective-help">Optional steer appended to every evolution cycle objective. It never overrides the LLM-first promotion; leave empty for pure LLM choice.</div>
                        </div>
                    </div>

                    <div class="form-section" id="settings-consciousness">
                        <h3>Consciousness</h3>
                        ${consciousnessSectionHtml()}
                    </div>

                    <div class="form-section">
                        <h3>Context Mode</h3>
                        <div class="settings-section-copy">
                            Working-context size profile (separate axis from Runtime Mode and Review Enforcement).
                            <code>Max</code> inlines ARCHITECTURE and DEVELOPMENT in full &mdash; for ~1M-context models (today's behavior).
                            <code>Nano</code> is the compact owner window. <code>Low</code> fits ~250K / local models: ARCHITECTURE becomes a navigation map (read full sections on demand), DEVELOPMENT stays full for normal runnable tasks unless a structured non-development caller opts out, and memory compacts sooner. It governs Ouroboros's own working window: it never changes the model or reasoning effort, and the review panel runs in every mode.
                            <br><strong>Human controlled:</strong> saved via the owner endpoint; saves immediately (no restart), and lowering requires Ouroboros to be idle.
                        </div>
                        <div class="settings-effort-card">
                            <label>Context Mode</label>
                            <input id="s-context-mode" type="hidden" value="max">
                            ${renderSegmentedField({
                                target: 's-context-mode',
                                title: 'Saves immediately; no restart required. Lowering requires Ouroboros to be idle.',
                                options: [
                                    { value: 'nano', label: 'Nano' },
                                    { value: 'low', label: 'Low' },
                                    { value: 'max', label: 'Max' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Image Input</h3>
                        <div class="settings-section-copy">Auto sends images unless the model's own metadata says it cannot see them; then a model that can see writes a caption, or a short note takes the image's place. Inline always sends images, even when metadata says no; if the provider refuses one, you see the refusal and that image becomes a note. Caption always replaces images with text captions. Off replaces images with a note and starts no caption work. The local model and GigaChat cannot carry images in any mode: they get a caption or a note that says so.</div>
                        <div class="settings-effort-card">
                            <label>Image Input Mode</label>
                            <input id="s-image-input-mode" type="hidden" value="auto">
                            ${renderSegmentedField({
                                target: 's-image-input-mode',
                                modifier: 'data-image-input-group',
                                options: [
                                    { value: 'auto', label: 'Auto' },
                                    { value: 'caption', label: 'Caption' },
                                    { value: 'inline', label: 'Inline' },
                                    { value: 'off', label: 'Off' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Skills</h3>
                        <div class="settings-section-copy">
                            Closed-loop skill development can auto-grant the keys and host permissions a skill declares after a fresh executable review for the current content hash.
                            Leave this off when every skill permission should require a separate human approval.
                        </div>
                        <label class="local-toggle ui-field ui-field-inline" title="Applies only after a fresh executable skill review and only to manifest-declared grants for that exact content hash.">
                            <input type="checkbox" id="s-auto-grant-reviewed-skills" class="ui-checkbox" name="s-auto-grant-reviewed-skills">
                            Auto-grant reviewed skills' keys and permissions
                        </label>
                    </div>

                    <div class="form-section">
                        <h3>Prompt Cache TTL</h3>
                        <div class="settings-section-copy">
                            How long provider prompt caches keep this agent's stable context warm (Anthropic-family routes; other providers manage caching implicitly).
                            <code>1h</code> keeps the large stable prefix cached across long waits and review cycles &mdash; cache writes bill at 2&times; base input instead of 1.25&times;, but one wait longer than 5 minutes already pays that back on big contexts.
                            <code>5m</code> is the provider default tier as an explicit value; <code>Default</code> sends bare markers (provider default, callers may still declare their own TTL).
                            One honest global: it applies to every lane, including review and safety prompts.
                        </div>
                        <div class="settings-effort-card">
                            <label>Prompt Cache TTL</label>
                            <input id="s-prompt-cache-ttl" type="hidden" value="1h">
                            ${renderSegmentedField({
                                target: 's-prompt-cache-ttl',
                                options: [
                                    { value: 'default', label: 'Default' },
                                    { value: '5m', label: '5m' },
                                    { value: '1h', label: '1h' },
                                ],
                            })}
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>External Skills Repo</h3>
                        <div class="settings-section-copy">
                            Optional EXTRA discovery path on top of the in-data-plane
                            <code>data/skills/{native,clawhub,external}/</code> tree.
                            Ouroboros scans this for additional skill packages without
                            cloning or pulling them. Leave empty to use only the data plane.
                        </div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-skills-repo-path">Skills Repo Path</label>
                                <input id="s-skills-repo-path" placeholder="~/Ouroboros/skills or /absolute/path/to/skills" class="ui-control" name="s-skills-repo-path" type="text" aria-describedby="s-skills-repo-path-help">
                                <div class="settings-inline-note ui-field-help" id="s-skills-repo-path-help">Absolute or <code>~</code>-prefixed path. Ouroboros never clones/pulls this directory — you manage it yourself.</div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>ClawHub Marketplace</h3>
                        <div class="settings-section-copy">
                            Always-on surface for installing community skills from
                            <a href="https://clawhub.ai" target="_blank" rel="noopener">clawhub.ai</a>.
                            The Skills page exposes a Marketplace tab; every install is
                            staged, OpenClaw frontmatter is translated into the
                            Ouroboros manifest shape, and the skill review (the panel the current
                            review mode composes) runs automatically before the skill becomes
                            executable. Plugins (Node)
                            are filtered out — only skill packages are installable.
                        </div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-clawhub-registry-url">Registry URL</label>
                                <input id="s-clawhub-registry-url" placeholder="https://clawhub.ai/api/v1" class="ui-control" name="s-clawhub-registry-url" type="text" aria-describedby="s-clawhub-registry-url-help">
                                <div class="settings-inline-note ui-field-help" id="s-clawhub-registry-url-help">Override only for self-hosted mirrors. Hostname must be <code>clawhub.ai</code> or localhost.</div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Update Channel</h3>
                        <div class="settings-section-copy">
                            Chooses which official branch Update checks. Your local work branch stays <code>ouroboros</code>.
                            <code>Stable</code> follows released code on <code>main</code> (default),
                            <code>QA</code> follows <code>ouroboros-stable</code>, and
                            <code>Development</code> follows <code>ouroboros</code>.
                        </div>
                        <div class="settings-effort-card">
                            <label>Official Update Source</label>
                            <input id="s-update-channel" type="hidden" value="stable">
                            ${renderSegmentedField({
                                target: 's-update-channel',
                                title: 'Applies immediately; no restart required.',
                                options: [
                                    { value: 'stable', label: 'Stable' },
                                    { value: 'qa', label: 'QA' },
                                    { value: 'development', label: 'Development' },
                                ],
                            })}
                        </div>
                    </div>

                    <!-- Last on the tab: set once, and on a browser or Docker host it only explains
                         why the switches cannot be used (docs/DESIGN.md §5 "Order by importance"). -->
                    <div class="form-section" data-autostart-settings hidden>
                        <h3>Startup &amp; background</h3>
                        <div class="settings-section-copy">
                            Applies immediately to the host computer running Ouroboros, including when you
                            connect from another device. Signing in preserves Panic stops and saved pauses.
                            In the background, tasks, schedules and Telegram keep working; an icon reopens
                            the window or quits.
                        </div>
                        <div class="settings-effort-card">
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
                        </div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="appearance">
                    ${languageBlockHtml()}

                    <div class="form-section">
                        <h3>Theme</h3>
                        <div class="settings-section-copy">
                            <code>System</code> follows this device's OS appearance and is the default for a new client.
                            <code>Light</code> and <code>Dark</code> pin the palette regardless of the OS.
                            <br><strong>Per device, not per account:</strong> this window or browser keeps the choice
                            for this Ouroboros, also after it moves to another port; other devices, browsers and
                            installations keep their own, and the server never stores it. Clearing this client's site
                            data returns it to System.
                        </div>
                        <div class="settings-effort-card">
                            <label class="theme-choice-label" id="s-appearance-theme-label">Theme</label>
                            <div data-theme-control aria-labelledby="s-appearance-theme-label"></div>
                            <div class="settings-inline-note theme-status" data-theme-status role="status" aria-live="polite"></div>
                            <div class="settings-inline-note" data-shell-storage-status role="status" aria-live="polite"></div>
                        </div>
                    </div>

                    <div class="form-section" data-interface-size>
                        <h3>Interface size</h3>
                        <div class="settings-section-copy">
                            Makes everything larger or smaller: text, buttons, cards and widgets.
                            <br><strong>Per device:</strong> each window and browser keeps its own size.
                        </div>
                        <div class="settings-effort-card">
                            <!-- interface_size.js paints the range this surface allows and applies a size when the slider is let go. -->
                            <div class="interface-size-range" data-interface-size-row hidden>
                                <label for="interface-size">Size</label>
                                <input type="range" id="interface-size" min="50" max="200" step="5" value="100" data-interface-size-input>
                                <output for="interface-size" data-interface-size-value data-i18n-skip>100%</output>
                                <div class="settings-toolbar">
                                    <button type="button" class="btn btn-default btn-sm" data-interface-size-reset>Reset to 100%</button>
                                </div>
                            </div>
                            <div class="settings-inline-note" data-interface-size-note role="status" aria-live="polite"></div>
                        </div>
                    </div>

                    <div class="form-section" data-notify-settings>
                        <h3>Notifications</h3>
                        <div class="settings-section-copy">
                            While this client is running, Ouroboros can pull you back to a question or a
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
                            permissions still decide what you see.
                        </div>
                        <div class="settings-effort-card">
                            <label class="local-toggle ui-field ui-field-inline">
                                <input type="checkbox" class="ui-checkbox" data-notify-pref="enabled">
                                Enable notifications
                            </label>
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
                            <div class="settings-toolbar">
                                <button type="button" class="btn btn-default btn-sm" data-notify-test>Send a test notification</button>
                            </div>
                            <div class="settings-inline-note" data-notify-status role="status" aria-live="polite"></div>
                            <div class="settings-inline-note" data-notify-attention-status role="status" aria-live="polite"></div>
                        </div>
                    </div>
                </section>

                <section class="settings-panel" data-settings-panel="advanced">
                    <div class="form-section">
                        <h3>Runtime Limits</h3>
                        <!-- Active Subagents / Root and Subagent Depth moved to
                             Agents → Delegation (D-10): they bound the agents,
                             not the process pool. Max Workers stays: it is
                             runtime worker processes, not an agent setting. -->
                        <div class="settings-section-copy">Workers control parallel task capacity. A quiet task is checked on, never stopped for being quiet; the per-task round and lifetime limits are optional — a positive number, or <code>unlimited</code> for none (the fresh-install default). A settings file from an earlier release that never set them keeps the finite limits it ran under until you change them here; startup names the values once. Budget limits control runtime cost thresholds. How many subagents a task may run, and how deep they may nest, live in <code>Agents</code>.</div>
                        <div class="form-grid two">
                            <div class="form-field ui-field">
                                <label for="s-total-budget">Total Budget (USD)</label>
                                <input id="s-total-budget" type="number" min="0.01" step="any" value="200.0" class="ui-control" name="s-total-budget">
                            </div>
                            <div class="form-field ui-field">
                                <label for="s-settings-per-task-cost">Per-Task Cost Cap (USD)</label>
                                <input id="s-settings-per-task-cost" type="number" min="0.01" step="any" value="50.0" class="ui-control" name="s-settings-per-task-cost">
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
                        </div>
                    </div>

                    <div class="form-section">
                        <div class="settings-card-head">
                            <div>
                                <h3>MCP Servers</h3>
                                <div class="settings-section-copy">
                                    External Model Context Protocol tool servers. MCP is a base-runtime client:
                                    it borrows tools from trusted HTTP/SSE servers or local stdio processes and exposes them as non-core
                                    <code>mcp_&lt;server&gt;__&lt;tool&gt;</code> tools after refresh. Changes are hot-reloadable.
                                    Treat server descriptions and results as untrusted third-party data.
                                </div>
                            </div>
                            <div class="settings-toolbar">
                                <button type="button" class="btn btn-default btn-sm" id="btn-mcp-add-server">Add server</button>
                                <button type="button" class="btn btn-default btn-sm" id="btn-mcp-import">Import mcp.json</button>
                                <button type="button" class="btn btn-default btn-sm" id="btn-mcp-refresh-all">Refresh all</button>
                            </div>
                        </div>
                        <div class="form-grid two">
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
                        <div id="mcp-servers-list" class="mcp-servers-list"></div>
                    </div>

                    <div class="form-section">
                        <h3>Source Control</h3>
                        <div class="settings-section-copy">Repository metadata for GitHub integration. Tokens live in Secrets; this is not secret.</div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-gh-repo">GitHub Repo</label>
                                <input id="s-gh-repo" placeholder="owner/repo-name" class="ui-control" name="s-gh-repo" type="text">
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Local Model Runtime</h3>
                        <div class="settings-section-copy">Only fill this in when you want Ouroboros to start and route to a GGUF model on this machine.</div>
                        <div class="form-grid two">
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
                        <button class="btn btn-secondary local-model-install-btn local-model-hidden" id="btn-local-install-runtime">Install Local Runtime</button>
                        <div id="local-model-test-result" class="settings-test-result"></div>
                    </div>

                    <div class="form-section">
                        <h3>Extension Settings</h3>
                        <div class="settings-section-copy">
                            Live extensions can register reviewed, host-rendered settings sections.
                            Sections appear here after the owning skill is reviewed, enabled, and loaded.
                        </div>
                        <div id="extension-settings-sections" class="settings-extension-sections">
                            <div class="muted">No extension settings registered.</div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Extra CA Certificates</h3>
                        <div class="settings-section-copy">Only for a network whose TLS certificates the default bundle does not trust: a TLS-inspecting proxy, or an endpoint behind a national CA such as GigaChat's. Ouroboros adds the file to its defaults for every provider call; everything else keeps working.</div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-extra-ca-bundle">Extra CA bundle (PEM)</label>
                                <input id="s-extra-ca-bundle" placeholder="/path/to/extra-ca.pem" class="ui-control" name="s-extra-ca-bundle" type="text" aria-describedby="s-extra-ca-bundle-help">
                                <div class="settings-inline-note ui-field-help" id="s-extra-ca-bundle-help">Absolute or <code>~</code>-prefixed path to a PEM file on the machine that runs Ouroboros (its own filesystem, not the device showing this page). Leave empty unless a provider fails with a certificate error.</div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Main Interface</h3>
                        <div class="settings-section-copy">The page this Ouroboros serves at its own address. Chats, tasks, files, settings, onboarding, skill widgets and the API keep working with every choice. Applies at the next restart.</div>
                        <div class="form-row">
                            <div class="form-field ui-field">
                                <label for="s-ui-dir">Interface directory</label>
                                <input id="s-ui-dir" placeholder="Empty for built-in, none, or /path/to/interface" class="ui-control" name="s-ui-dir" type="text" spellcheck="false" aria-describedby="s-ui-dir-help s-ui-dir-applied">
                                <div class="settings-inline-note ui-field-help" id="s-ui-dir-help">Empty: the built-in interface. <code>none</code>: no main page, for an interface hosted separately behind the same address. Otherwise an absolute or <code>~</code>-prefixed path, on the machine that runs Ouroboros, to a trusted directory with its own <code>index.html</code>; an invalid path or missing entrypoint keeps the built-in interface and says why below. File permissions still need to allow the server to read the selected files. After the restart the built-in page reloads into the new choice; a page from a directory or another host refreshes itself and only reconnects. To return to the built-in interface, clear this field (from any interface, or in <code>settings.json</code> while Ouroboros is stopped), restart, and reopen the Ouroboros address. An <code>OUROBOROS_UI_DIR</code> environment value shows here until a choice is saved; a saved choice, empty included, wins.</div>
                                <div class="settings-inline-status" id="s-ui-dir-applied" role="status" aria-live="polite" hidden></div>
                            </div>
                        </div>
                    </div>

                    <div class="form-section">
                        <h3>Cleanup</h3>
                        <!-- The two subagent path roots moved to Agents →
                             Delegation → Advanced (D-10). GC Retention stays: it
                             governs every disposable runtime artifact, not just
                             subagent worktrees. -->
                        <div class="settings-section-copy">
                            <strong>GC Retention</strong> is the single age knob (days) for all disposable runtime artifacts the startup garbage collector removes: acting-subagent worktrees, terminal task drives, and leftover service logs (hard max 365). Genesis projects are durable and never auto-removed. Where subagents check out that work is set in <code>Agents</code>.
                        </div>
                        <div class="form-grid two">
                            <div class="form-field ui-field">
                                <label for="s-gc-retention-days">GC Retention (days)</label>
                                <input id="s-gc-retention-days" type="number" min="1" max="365" value="7" class="ui-control" name="s-gc-retention-days">
                            </div>
                        </div>
                    </div>

                    <div class="form-section danger">
                        <h3>Danger Zone</h3>
                        <div class="settings-inline-note">Reset still uses the current restart-based flow. This clears runtime data but keeps the repo.</div>
                        <button class="btn btn-danger" id="btn-reset">Reset All Data</button>
                    </div>
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

            <div class="settings-footer">
                <div class="settings-footer-actions">
                    <button type="button" class="btn btn-secondary" id="btn-reload-settings">Reload Settings</button>
                    <button class="btn btn-save" id="btn-save-settings">Save Settings</button>
                    <button type="button" class="btn btn-secondary" id="btn-restart-now" hidden
                        title="Restart the agent process">Restart now</button>
                </div>
                <div class="settings-footer-status">
                    <span id="settings-unsaved-indicator" class="settings-inline-status settings-unsaved-indicator">Unsaved changes</span>
                    <div id="settings-restart-status" class="settings-inline-status" role="status" aria-live="polite" hidden></div>
                    <div id="settings-status" class="settings-inline-status" role="status" aria-live="polite" aria-atomic="true"></div>
                </div>
            </div>
        </div>
    `;
}

export function bindSettingsTabs(root, options = {}) {
    const panels = Array.from(root.querySelectorAll('.settings-panel'));
    const scrollRoot = root.querySelector('.settings-scroll');
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
