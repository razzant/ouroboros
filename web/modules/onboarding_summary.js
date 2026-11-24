// The first-run wizard's Summary step: what the one Start saves, read before it is saved. The choices that shape the
// first hour lead, one row each in the rail's order: the connections, the Main model, Access, Consciousness where it is
// offered, Skills (the starter list paints its own row and Change into the node left for it) and the limits. Every
// assignment Start saves stands behind one disclosure, each model role whole as the model and agent editors name it
// (what it inherits, its account or rotation, processing, a window the owner set), local routing, the agents and the
// reviewers. Late reads (sources, catalog, preview) repaint the rows and notes in place. The disclosure
// opens by itself only when a note under the card starts asking to check the assignments, else it stays as the
// owner left it. The step reads the wizard's draft and writes nothing of it.
import { escapeHtmlAttr as escapeHtml, formatUsd2 } from './utils.js';
import { resolveModelRoleAssignments } from './model_roles.js';
import { familyLabels } from './onboarding_agents_step.js';
import { sourcePresentation } from './harness_presentation.js';
import { languageBlockHtml } from './settings_language.js';
import { apiProviderLabel, isLocalFilesystemSource, processingIntentLabel } from './route_editor_primitives.js';
import { PACKET_ONLY_POOL_WARNING, packetOnlyReviewPool, reviewPoolRows } from './subagents_settings.js';

const trim = (value) => String(value || '').trim();
const rowsHtml = (rows) => rows.map(([label, value]) => `<div class="summary-kv"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>`).join('');
// The keys a draft can hold, in Accounts' words; a value beside the label replaces «configured».
const KEY_ROWS = [['anthropicKey', 'Anthropic'], ['zaiKey', 'Z.ai (GLM)'], ['deepseekKey', 'DeepSeek'], ['minimaxKey', 'MiniMax'],
    ['cloudruKey', 'Cloud.ru'], ['openaiKey', 'OpenAI'], ['openrouterKey', 'OpenRouter']];

function localArtifact(draft) {
    // The direct file wins even if the draft still holds a filename from a Hub repository.
    const source = trim(draft.LOCAL_MODEL_SOURCE);
    const filename = source && !isLocalFilesystemSource(source) ? trim(draft.LOCAL_MODEL_FILENAME) : '';
    return { source: filename ? `${source} / ${filename}` : source,
        name: (filename || source).split(/[\\/]/).filter(Boolean).at(-1) || '' };
}

const noteHtml = (attr, text) => `<div class="wizard-inline-note" ${attr}${text ? '' : ' hidden'}>${escapeHtml(text)}</div>`;

/**
 * The Summary step over the wizard's draft. `contract`: the setup contract (its model slots, provider profiles, local
 * routing choices and the step's words); `state`: the draft, read only; `consciousness`: that step (its row, and
 * Summary's lead where a first state is still to come); `localSlot(slot)`: the local runtime serves the role;
 * `accessLabel(mode)`: an Access level in words; `keyWords(key)`: a built-in key's choices in words; `reads()`: what the
 * wizard has read of this draft, `{ api, subscription, local, profile, connected, sources, draft, catalog, agents:
 * { snapshot, catalogKnown }, mainBinding }`.
 */
export function createSummaryStep({ contract = {}, state, consciousness, localSlot, accessLabel, keyWords, reads }) {
    const slots = contract.modelSlots || [];
    const profiles = contract.providerProfiles || {};
    const meta = (contract.steps || []).find((step) => step.id === 'summary') || {};
    let open = false;   // the assignments, as the owner left them
    let shownNotes = {};

    // One model role as Start saves it. The route's effective lane decides: the local runtime, else the subscription
    // the full descriptor read names (whether or not it serves this setup), else the API source, always in words.
    // `compact` keeps the source, the model and a pinned account: Main's row among the choices.
    function assignment(role, read, localName, compact = false) {
        const { slot, local, source, model, account, inherits, context, processing, inheritedProcessing } = role;
        const subscription = !local && source.startsWith('subscription:');
        const facets = [local ? (localName ? `Local · ${localName}` : 'Local model')
            : `${subscription ? sourcePresentation(source.slice(13), read.sources).label : apiProviderLabel(source, profiles)} · ${model}`];
        if (subscription && (account || !compact)) facets.push(account ? `Account: ${account}` : 'Auto rotation');
        if (compact) return facets.join(' · ');
        facets.push(`Processing: ${processingIntentLabel(processing, inheritedProcessing, { helper: slot === 'fallback' })}`);
        if (context > 0) facets.push(`${context.toLocaleString('en-US')} tokens, set by you`);
        const from = slots.find((entry) => entry.slot === inherits)?.label.replace(/ Model$/, '') || 'Main';
        return `${inherits ? `Uses ${from} · ` : ''}${facets.join(' · ')}`;
    }

    function agentsValue(read) {
        const labels = familyLabels(state.agentsConnected, read.agents.snapshot, { catalogKnown: read.agents.catalogKnown });
        const count = (read.catalog?.items || []).length;
        const actors = `${count} Available subagent${count === 1 ? '' : 's'}`;
        if (!labels.length) return `${actors} · API/local access only`;
        if (state.skipSubscriptionPresets) return `${actors} · ${labels.join(', ')} connected · automatic subscription preset skipped`;
        return `${actors} · ${labels.join(', ')} connected`;
    }

    const reviewer = (row) => [row.route?.target_id || row.subagent_id,
        row.route?.credential_profile_id ? `Account: ${row.route.credential_profile_id}` : '', `Effort: ${row.effort || 'Auto'}`,
        `Processing: ${processingIntentLabel(row.processing_preference, state.processingPreference)}`].filter(Boolean).join(' · ');

    function parts(read) {
        const connected = read.subscription && !read.api
            ? `${read.connected.filter((source) => state.agentsConnected.includes(source.credentialHarness))
                .map((source) => sourcePresentation(source.id, read.connected).label).join(', ')} subscription · no API key`
            : read.local && !read.api ? (profiles.local?.label || 'Local-first')
                : profiles[read.profile]?.label || profiles.openrouter?.label || 'OpenRouter';
        const local = localArtifact(read.draft);
        // OPENAI_BASE_URL belongs to the built-in OpenAI key independently of which pair the compatible route selects.
        const legacyAddress = trim(read.draft.OPENAI_BASE_URL);
        const special = { zaiKey: trim(state.zaiPlan) === 'coding' ? 'configured · coding plan' : '',
            openaiKey: legacyAddress ? `configured for the legacy address ${legacyAddress}` : '' };
        const keys = KEY_ROWS.filter(([key]) => trim(state[key]))
            .map(([key, label]) => [label, [special[key] || 'configured', ...keyWords(key)].join(' · ')]);
        const roles = resolveModelRoleAssignments(read.draft, slots, localSlot).filter((role) => role.local || role.model);
        const main = roles.find((role) => role.slot === 'main');
        const usesSubscription = roles.some(({ local, source }) => !local && source.startsWith('subscription:'));
        const routing = (contract.localRoutingModes || []).find((mode) => mode.value === state.localRoutingMode)?.label || 'Cloud models only';
        return {
            choices: [['Connected access', connected],
                ...(read.local ? [['Local source', local.source]] : []),
                ...keys,
                ['Main model', main ? assignment(main, read, local.name, true) : 'None'],
                ['Access level', accessLabel(state.runtimeMode)],
                ...(trim(state.skillsRepoPath) ? [['Skills repo', trim(state.skillsRepoPath)]] : []),
                ...consciousness.summaryRows()],
            limits: [...(read.api || !read.subscription ? [['Total API budget', formatUsd2(state.totalBudget)],
                ['Per-task API cap', formatUsd2(state.perTaskCostUsd)]] : []),
                ...(usesSubscription ? [['When quota runs out', roles.some((role) => role.slot === 'fallback') ? 'Tries the Fallback, then waits for the reset'
                    : 'Waits for the reset, then continues']] : [])],
            assignments: [...(read.local ? [['Local routing', routing]] : []),
                ...slots.map((slot) => [slot.slot === 'consciousness' ? 'Consciousness model' : slot.label.replace(/ Model$/, ''),
                    roles.filter((role) => role.slot === slot.slot).map((role) => assignment(role, read, local.name)).join(' → ') || 'None']),
                ['Agents', agentsValue(read)],
                ['Reviewers', reviewPoolRows(read.catalog).map(reviewer).join(' · ') || 'None — the main model reviews alone']],
        };
    }

    function assignmentNotes(read) {
        const recovery = state.recoveryPrepared ? `Automatic subscription presets were skipped. ${state.recoveryMain === read.mainBinding
            ? 'Reviewers were assigned to Main.' : 'Main changed; reviewers keep the assignments shown above. Use Main for reviewers again if you want to update them.'} Check the assignments, then Start Ouroboros to save this draft. Later changes in Settings are manual.` : '';
        return { 'data-review-pool-warning': packetOnlyReviewPool(read.catalog) ? PACKET_ONLY_POOL_WARNING : '', 'data-recovery-note': recovery };
    }

    function rememberNotes(notes) {
        const startsAsking = Object.entries(notes).some(([attr, text]) => text && !shownNotes[attr]);
        shownNotes = notes;
        if (startsAsking) open = true;
        return startsAsking;
    }

    return {
        /** The step: its header, the card (the choices, the Skills node, the limits, the assignments' disclosure),
         * the notes that ask to check the assignments, and the staged Language control. */
        html() {
            const read = reads();
            const { choices, limits, assignments } = parts(read);
            const notes = assignmentNotes(read);
            rememberNotes(notes);
            return `
            <div class="step-header">
                <div>
                    <h2 class="step-title">${escapeHtml(meta.title || 'Review before launch')}</h2>
                    <p class="step-copy">${escapeHtml(consciousness.summaryLead() || meta.copy || '')}</p>
                </div>
            </div>
            <div class="summary-card">
                <div class="summary-card-rows" data-summary-part="choices">${rowsHtml(choices)}</div>
                <div data-starter-skills="row"></div>
                <div class="summary-card-rows" data-summary-part="limits">${rowsHtml(limits)}</div>
                <details class="ui-disclosure summary-assignments" data-summary-assignments${open ? ' open' : ''}>
                    <summary>Models, agents and reviewers</summary>
                    <div class="summary-card-rows" data-summary-part="assignments">${rowsHtml(assignments)}</div>
                </details>
            </div>
            ${noteHtml('data-tone="warn" data-review-pool-warning', notes['data-review-pool-warning'])}
            ${noteHtml('data-recovery-note', notes['data-recovery-note'])}
            ${languageBlockHtml({ onboarding: true })}`;
        },
        /** The owner's open or close of the assignments is kept across renders. */
        bind(root) {
            const details = root.querySelector('[data-summary-assignments]');
            details?.addEventListener('toggle', () => { if (details.isConnected) open = details.open; });
        },
        /** Patch only the mounted Summary. The Language control and the disclosure retain their nodes. */
        refresh(root) {
            const details = root.querySelector('[data-summary-assignments]');
            if (!details) return;
            const read = reads();
            for (const [part, rows] of Object.entries(parts(read))) {
                const host = root.querySelector(`[data-summary-part="${part}"]`);
                if (host) host.innerHTML = rowsHtml(rows);
            }
            const notes = assignmentNotes(read);
            for (const [attr, text] of Object.entries(notes)) {
                const host = root.querySelector(`[${attr}]`);
                if (host && text !== shownNotes[attr]) Object.assign(host, { textContent: text, hidden: !text });
            }
            if (rememberNotes(notes)) details.open = true;
        },
    };
}
