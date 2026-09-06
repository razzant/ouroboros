import {
    escapeHtmlAttr as escapeHtml,
    grantReady,
    isRateLimitError,
    preflightFailed,
    preflightFailedStale,
    preflightFindingText,
    reviewReady,
    safeExternalHrefAttr as safeExternalUrl,
} from './utils.js';
import { formatRelativeAge, installedTime, renderToneBadge } from './ui_helpers.js';
import { hubListingRowFor, hubSyncVerdict } from './hub_sync.js';

function hasSkillUiTab(skill, live = {}) {
    return (live?.ui_tabs || []).some((tab) => (tab?.skill || tab?.skill_name || tab?.extension || '') === skill.name);
}

function statusBadge(status, gate = null, profile = '') {
    // Owner-attested skills are executable but the expensive LLM review was SKIPPED — show a
    // distinct warning-toned badge so it never reads as a full LLM-clean verdict.
    if (profile === 'owner_attested') {
        return renderToneBadge('owner-attested', 'warn');
    }
    const executable = gate && typeof gate.executable_review === 'boolean'
        ? gate.executable_review
        : ['clean', 'warnings'].includes(status);
    const tone = status === 'blockers' ? 'danger' : executable ? 'ok' : status === 'warnings' ? 'warn' : 'muted';
    return renderToneBadge(status || 'pending', tone);
}

function missingGrantLoadError(skill) {
    return !grantReady(skill) && String(skill.load_error || '').includes('missing owner grants');
}

function repairableSource(skill) {
    const source = (skill.source || 'native').toLowerCase();
    const payloadRoot = String(skill.payload_root || '');
    // self_authored payloads live under skills/external/ (config
    // SKILL_SOURCE_SUBDIRS has no self_authored bucket), so the payload-root
    // gate already matches them — only the source list kept them out (#335).
    return ['clawhub', 'ouroboroshub', 'external', 'self_authored'].includes(source)
        && /^skills\/(external|clawhub|ouroboroshub)\//.test(payloadRoot);
}

function repairReady(skill) {
    return repairableSource(skill)
        && (skill.review_status === 'blockers'
            || preflightFailed(skill)
            || (Boolean(skill.load_error) && !missingGrantLoadError(skill)));
}

// D11: the recorded preflight FAIL is STALE for the current payload bytes —
// Re-review stays primary; Repair is offered from the menu, honestly labeled
// as based on the last recorded preflight run.
function staleRepairOffer(skill) {
    return repairableSource(skill) && preflightFailedStale(skill) && !preflightFailed(skill);
}

function skillConflictReason(skill) {
    const conflict = skill?.conflict && typeof skill.conflict === 'object' ? skill.conflict : null;
    if (!conflict || conflict.code !== 'skill_conflict') return '';
    const names = Array.isArray(conflict.skills) ? conflict.skills.filter(Boolean) : [];
    if (!names.length) return 'conflicts with another enabled skill';
    const omitted = Number(conflict.omitted || 0);
    return `conflicts with ${names.join(', ')}${omitted > 0 ? ` (+${omitted} more)` : ''}`;
}

function primaryAction(skill, reviewInProgress, repairInProgress, live) {
    if (reviewInProgress) return { label: 'Reviewing...', disabled: true };
    if (repairInProgress) return { label: 'Repairing...', disabled: true };
    if (skill.lifecycle_virtual && (skill.load_error || isRateLimitError(skill.load_error)) && (skill.source || '').toLowerCase() === 'clawhub') {
        return { action: 'retry_install', label: isRateLimitError(skill.load_error) ? 'Retry later' : 'Retry install' };
    }
    // A deterministic preflight FAIL persists as pending: Review/Re-review
    // would deterministically fail again, so a repairable payload offers
    // Repair first (#335). Non-repairable sources keep the Review CTA.
    if ((skill.load_error && !missingGrantLoadError(skill))
        || (skill.review_status === 'blockers' && !reviewReady(skill))
        || (preflightFailed(skill) && repairReady(skill))) {
        return repairReady(skill) ? { action: 'repair', label: 'Repair' } : { label: '', disabled: true };
    }
    if (!reviewReady(skill)) return { action: skill.review_stale ? 'rereview' : 'review', label: skill.review_stale ? 'Re-review' : 'Review' };
    const grants = skill.grants || {};
    const missing = [
        ...(grants.missing_keys || grants.requested_keys || []),
        ...(grants.missing_permissions || grants.requested_permissions || []),
    ];
    if (skill.is_self_authored && !skill.enabled) return { action: 'approve_enable', label: 'Approve & enable', keys: missing.join(',') };
    if (!grantReady(skill)) return { action: 'grant', label: 'Grant access', keys: missing.join(',') };
    if (skill.enabled && skill.type === 'extension' && skill.live_loaded && hasSkillUiTab(skill, live)) {
        return { action: 'open_widgets', label: 'Open widgets' };
    }
    return { label: '' };
}

const LIFECYCLE_PENDING_LABELS = {
    disable: 'Disabling…',
    enable: 'Enabling…',
    uninstall: 'Uninstalling…',
    update: 'Updating…',
    install: 'Installing…',
    review: 'Reviewing…',
    deps: 'Installing deps…',
    delete: 'Deleting…',
};

function lifecyclePendingLabel(kind) {
    return LIFECYCLE_PENDING_LABELS[String(kind || '')] || 'Working…';
}

function lifecycleFailedLabel(kind) {
    const base = String(kind || '').trim();
    if (!base) return 'Lifecycle failed';
    return `${base.charAt(0).toUpperCase()}${base.slice(1)} failed`;
}

function statusChip(skill, action, live) {
    // An in-flight lifecycle job (serialized lane) takes precedence so the card
    // shows e.g. "Disabling…" instead of the stale persisted state.
    if (skill.lifecycle_pending) {
        return `<span class="skills-status-chip skills-status-warn">${escapeHtml(lifecyclePendingLabel(skill.lifecycle_kind))}</span>`;
    }
    if (skill.lifecycle_status === 'failed') {
        return `<span class="skills-status-chip skills-status-danger">${escapeHtml(lifecycleFailedLabel(skill.lifecycle_kind))}</span>`;
    }
    let status = { tone: 'muted', label: 'Off' };
    const conflictReason = skillConflictReason(skill);
    if (!grantReady(skill)) status = { tone: 'warn', label: 'Needs access grant' };
    else if (skill.lifecycle_virtual && isRateLimitError(skill.load_error)) status = { tone: 'warn', label: 'Rate limited' };
    else if (skill.load_error) status = { tone: 'danger', label: 'Failed to load' };
    else if (conflictReason) status = { tone: 'danger', label: conflictReason.charAt(0).toUpperCase() + conflictReason.slice(1) };
    else if (preflightFailed(skill)) status = { tone: 'danger', label: 'Preflight failed' };
    else if (!reviewReady(skill)) status = { tone: 'warn', label: 'Needs review' };
    else if (skill.enabled && skill.type === 'extension') {
        status = skill.live_loaded && (skill.dispatch_live || hasSkillUiTab(skill, live))
            ? { tone: 'ok', label: 'Loaded' }
            : { tone: 'warn', label: skill.live_loaded ? 'Loaded — UI tab pending' : 'Enabled — not loaded' };
    } else if (skill.enabled) status = { tone: 'ok', label: 'Enabled' };
    const process = ['server', 'worker'].includes(skill.process) ? skill.process : '';
    if (skill.type === 'extension' && process) status.label += ` · ${process}`;
    const attrs = action.action ? `data-skill="${escapeHtml(skill.name)}" data-skill-action="${escapeHtml(action.action)}" role="button" tabindex="0"` : '';
    return `<span class="skills-status-chip skills-status-${status.tone} ${action.action ? 'is-clickable' : ''}" ${attrs}>${escapeHtml(status.label)}</span>`;
}

/**
 * Display-only OuroborosHub sync badges for an installed card. The verdict
 * comes from the shared hub_sync helper: "Update available" when the live
 * catalog serves a different version for a hub-bucket skill, "Published vX"
 * on the server's byte-exact verification, and "Submitted PR #N" from the
 * local publish receipt while the catalog does not confirm it. Without a
 * catalog snapshot (fetch failed or not passed) only listing-plane facts
 * ("Published vX") may be claimed.
 */
function hubSyncBadges(skill, options = {}) {
    const map = options.hubCatalogByName instanceof Map ? options.hubCatalogByName : null;
    const verdict = hubSyncVerdict(
        hubListingRowFor(skill),
        map ? (map.get(skill.name) || null) : null,
        { catalogUnavailable: options.hubCatalogAvailable !== true },
    );
    const facts = verdict.copy_facts;
    const out = [];
    if (verdict.badges.includes('update_available')) {
        out.push('<span class="skills-badge skills-badge-warn">Update available</span>');
    }
    if (verdict.badges.includes('published')) {
        out.push(`<span class="skills-badge skills-badge-ok">Published v${escapeHtml(skill.version || '')}</span>`);
    }
    if (verdict.badges.includes('submitted_pr') && facts.receipt_pr !== null) {
        out.push(`<span class="skills-badge skills-badge-warn">Submitted PR #${escapeHtml(String(facts.receipt_pr))}</span>`);
    }
    return out.join(' ');
}

function sourceChip(skill) {
    const source = (skill.source || 'native').toLowerCase();
    const map = {
        clawhub: ['ClawHub', 'warn'],
        ouroboroshub: ['OuroborosHub', 'ok'],
        self_authored: ['Authored', 'ok'],
        external: ['External', 'muted'],
        user_repo: ['User repo', 'muted'],
    };
    if (!map[source]) return '';
    const [label, tone] = map[source];
    return `<span class="skills-source-chip skills-source-${tone}">${escapeHtml(label)}</span>`;
}

function reviewFindings(skill) {
    const findings = Array.isArray(skill.review_findings) ? skill.review_findings : [];
    if (!findings.length) return '';
    const rows = findings.map((f) => {
        const preflight = preflightFindingText(f);
        const reason = preflight || f.reason || f.message || JSON.stringify(f);
        return `<li><strong>${escapeHtml(f.verdict || f.severity || '')}</strong> ${escapeHtml(f.item || f.check || f.title || 'finding')}: ${escapeHtml(reason)}</li>`;
    }).join('');
    return `<details class="skills-review-findings ui-rich-content"><summary class="muted">${findings.length} review finding${findings.length === 1 ? '' : 's'}</summary><ul>${rows}</ul></details>`;
}

function reviewRunTitle(run) {
    const round = Number(run?.review_round || 1);
    const attempt = Number(run?.snapshot_attempt || 1);
    const snapshot = String(run?.content_hash || '').slice(0, 12) || 'unknown';
    const revised = run?.snapshot_revised ? ' — revised snapshot' : '';
    return `Skill review round ${round} — snapshot ${snapshot} (attempt ${attempt})${revised}`;
}

function reviewHistory(skill) {
    const review = skill.skill_review && typeof skill.skill_review === 'object'
        ? skill.skill_review : {};
    // The backend projection already bounds history to its ten-row window and
    // discloses the exact omitted count; a second client-side slice would be
    // an undisclosed bound on top of a disclosed one.
    const history = Array.isArray(review.history) ? review.history : [];
    const current = review.current && Object.keys(review.current).length
        ? review.current : history[history.length - 1];
    if (!current) return '';
    const rows = history.map((run) => {
        const status = run.review_status || run.status || run.job_status || 'unknown';
        const source = run.source ? ` · ${run.source}` : '';
        return `<li>${escapeHtml(reviewRunTitle(run))} · ${escapeHtml(status)}${escapeHtml(source)}</li>`;
    }).join('');
    const omitted = Number(review.history_omitted);
    const historyLabel = Number.isFinite(omitted) && omitted > 0
        ? `${history.length} of ${history.length + omitted}`
        : `${history.length}`;
    const currentStatus = current.review_status || current.status || current.job_status || 'unknown';
    return `<div class="skills-review-current"><strong>${escapeHtml(reviewRunTitle(current))}</strong> · ${escapeHtml(currentStatus)}</div>
        ${rows ? `<details class="skills-review-history ui-rich-content"><summary class="muted">Skill Review history (${historyLabel})</summary><ol>${rows}</ol></details>` : ''}`;
}

function grantBlock(skill) {
    const grants = skill.grants || {};
    const requested = [...(grants.requested_keys || []), ...(grants.requested_permissions || [])];
    if (!requested.length) return '';
    const missing = [...(grants.missing_keys || []), ...(grants.missing_permissions || [])];
    const granted = [...(grants.granted_keys || []), ...(grants.granted_permissions || [])];
    const tone = grants.unsupported_for_skill_type ? 'muted' : missing.length ? 'warn' : 'ok';
    const status = grants.unsupported_for_skill_type ? 'This skill type cannot receive keys or host permissions.' : missing.length ? 'This skill needs your permission to use the keys and permissions above.' : 'Access granted.';
    return `<div class="skills-access skills-access-${tone}">
        <div class="skills-access-row"><span class="skills-access-label">Needs access</span> ${requested.map((k) => `<code>${escapeHtml(k)}</code>`).join(' ')}</div>
        ${granted.length ? `<div class="skills-access-row"><span class="skills-access-label">Granted</span> ${granted.map((k) => `<code>${escapeHtml(k)}</code>`).join(' ')}</div>` : ''}
        <div class="skills-access-status">${escapeHtml(status)}</div>
    </div>`;
}

function provenanceBlock(prov) {
    if (!prov || typeof prov !== 'object') return '';
    const rows = [];
    if (prov.slug) rows.push(`<span>slug: <code>${escapeHtml(prov.slug)}</code></span>`);
    if (prov.sha256) rows.push(`<span>sha256: <code>${escapeHtml(String(prov.sha256).slice(0, 12))}…</code></span>`);
    if (prov.license) rows.push(`<span>license: ${escapeHtml(prov.license)}</span>`);
    const href = safeExternalUrl(prov.homepage);
    if (href) rows.push(`<a href="${href}" target="_blank" rel="noopener noreferrer">homepage</a>`);
    const warnings = (prov.adapter_warnings || []).map((msg) => `<li>${escapeHtml(msg)}</li>`).join('');
    return (rows.length ? `<div class="skills-card-provenance muted">${rows.join(' · ')}</div>` : '')
        + (warnings ? `<details class="skills-card-warnings"><summary class="muted">adapter warnings</summary><ul>${warnings}</ul></details>` : '');
}

function presenceRuntimeBlock(skill) {
    const runtime = skill.presence_runtime;
    if (!runtime || typeof runtime !== 'object') return '';
    if (runtime.error) {
        return `<div class="skills-load-error">Presence runtime unavailable: ${escapeHtml(runtime.error)}</div>`;
    }
    const defaults = runtime.defaults || {};
    const overrides = runtime.overrides || {};
    const modelOverride = ['main', 'light'].includes(overrides.model_slot) ? overrides.model_slot : '';
    const roundsOverride = Number.isInteger(overrides.inline_max_rounds) ? overrides.inline_max_rounds : '';
    const fingerprint = String(runtime.state_fingerprint || '');
    return `<form class="skills-presence-runtime" data-presence-runtime-form data-skill-name="${escapeHtml(skill.name)}" data-state-fingerprint="${escapeHtml(fingerprint)}">
        <div class="skills-presence-runtime-title">Presence runtime</div>
        <label>Model
            <select name="model_slot">
                <option value="" ${modelOverride ? '' : 'selected'}>Reviewed default (${escapeHtml(defaults.model_slot || 'main')})</option>
                <option value="main" ${modelOverride === 'main' ? 'selected' : ''}>Main</option>
                <option value="light" ${modelOverride === 'light' ? 'selected' : ''}>Light</option>
            </select>
        </label>
        <label>Inline rounds
            <input name="inline_max_rounds" type="number" min="1" step="1" value="${escapeHtml(roundsOverride)}" placeholder="${escapeHtml(defaults.inline_max_rounds || 10)}">
        </label>
        <div class="skills-presence-runtime-actions">
            <button type="submit" class="btn btn-default btn-sm">Save</button>
            <button type="button" class="btn btn-ghost btn-sm" data-presence-runtime-reset>Use reviewed defaults</button>
        </div>
        <div class="muted">Applies to new Presence turns only.</div>
    </form>`;
}

export function renderInstalledSkillCard(skill, reviewingSkills = new Set(), repairingSkills = new Set(), live = {}, options = {}) {
    const safeName = escapeHtml(skill.name);
    const reviewInProgress = reviewingSkills.has(skill.name);
    const repairInProgress = repairingSkills.has(skill.name);
    const action = primaryAction(skill, reviewInProgress, repairInProgress, live);
    const actionAttrs = action.action ? `data-skill="${safeName}" data-skill-action="${escapeHtml(action.action)}" role="button" tabindex="0"` : '';
    const lockReason = !skill.enabled && (skillConflictReason(skill) || (skill.review_gate?.executable_review === false && (skill.review_gate.summary || skill.review_gate.blocking_reason)) || (skill.review_stale ? 'review is stale — re-review the skill first' : ''));
    const source = (skill.source || 'native').toLowerCase();
    const market = source === 'clawhub' || source === 'ouroboroshub';
    const payloadRoot = skill.payload_root || '';
    const localDelete = (source === 'self_authored' || source === 'external') && payloadRoot.startsWith('skills/external/');
    const prov = market ? skill.provenance : null;
    const hubBadges = hubSyncBadges(skill, options);
    const submit = submitHubReady(skill, Boolean(options.githubTokenConfigured));
    // Instruction skills from a marketplace/external bucket can be converted into
    // runnable script skills by the repair agent (it authors scripts/<file> and
    // flips type instruction->script, then re-reviews). Offer it as a secondary
    // action so the normal review/grant/enable CTA stays primary.
    const makeRunnable = skill.type === 'instruction'
        && ['clawhub', 'ouroboroshub', 'external', 'self_authored'].includes(source)
        && /^skills\/(external|clawhub|ouroboroshub)\//.test(payloadRoot)
        && !repairInProgress;
    // Owner-attestation: let the owner SKIP the expensive LLM review for THEIR OWN skill,
    // plus hash-verified official OuroborosHub payloads (freshly rechecked by the backend).
    // Offered only when a review is actually outstanding, and not once already owner-attested.
    // Mirror the backend source gate (skill_owner_attestation.review_skill_owner_attest):
    // native/ClawHub are never attestable, and OuroborosHub must carry a backend
    // owner_attestable/official_hub_verified hint. The endpoint still re-verifies.
    const officialHubHint = source === 'ouroboroshub'
        && (skill.owner_attestable === true || skill.official_hub_verified === true);
    const ownSourceHint = source !== 'clawhub'
        && source !== 'native'
        && source !== 'ouroboroshub'
        && (skill.owner_attestable === true || source === 'external' || source === 'self_authored' || skill.is_self_authored);
    const thirdPartySource = source === 'clawhub' || source === 'native' || (source === 'ouroboroshub' && !officialHubHint);
    const ownerAttestable = !reviewInProgress
        && !thirdPartySource
        && !(skill.review_profile === 'owner_attested' && !skill.review_stale)
        && (officialHubHint || ownSourceHint)
        // A deterministic preflight FAIL makes attestation a guaranteed 409
        // (run_owner_attestation reruns the preflight); hide the dead end (#335).
        && !preflightFailed(skill)
        && (!reviewReady(skill) || skill.review_stale);
    const menu = (market || localDelete || !reviewInProgress || submit.visible || makeRunnable || ownerAttestable)
        ? `<div class="skills-card-menu"><button type="button" class="skills-card-menu-trigger" aria-label="More actions" aria-haspopup="menu" aria-expanded="false" data-skill-menu-trigger>⋮</button><dialog class="skills-card-menu-dialog" role="menu">
            ${makeRunnable ? `<button type="button" role="menuitem" class="skills-menu-item skills-make-runnable" data-skill="${safeName}" data-skill-action="repair" title="Author a runnable script for this instruction skill via the repair agent">Make runnable</button>` : ''}
            ${!reviewInProgress && !(preflightFailed(skill) && repairReady(skill)) ? `<button type="button" role="menuitem" class="skills-menu-item skills-review" data-skill="${safeName}">${skill.review_status === 'pending' ? 'Review' : (skill.review_stale ? 'Re-review' : 'Review again')}</button>` : ''}
            ${!reviewInProgress && !repairInProgress && staleRepairOffer(skill) ? `<button type="button" role="menuitem" class="skills-menu-item skills-repair-stale" data-skill="${safeName}" data-skill-action="repair" title="Repair based on the last recorded preflight — the payload changed since that run, so Re-review would recheck it first">Repair</button>` : ''}
            ${ownerAttestable ? `<button type="button" role="menuitem" class="skills-menu-item skills-attest-review skills-attest-warn" data-skill="${safeName}" title="Skip the expensive LLM review for your own or verified official-hub skill. The deterministic safety preflight still runs, and this is logged for audit.">⚠️ Skip review</button>` : ''}
            ${submit.visible ? `<button type="button" role="menuitem" class="skills-menu-item skills-submit-hub ${submit.disabled ? 'is-disabled' : ''}" data-skill="${safeName}" title="${escapeHtml(submit.reason)}" data-submit-disabled="${submit.disabled ? 'true' : 'false'}" data-submit-reason="${escapeHtml(submit.reason)}" data-submit-state="${escapeHtml(submit.state || '')}" data-publication-ready="${submit.publication_ready === true ? 'true' : 'false'}" aria-disabled="${submit.disabled ? 'true' : 'false'}">Publish to OuroborosHub</button>` : ''}
            ${market ? `<button type="button" role="menuitem" class="skills-menu-item skills-update" data-skill="${safeName}" data-source="${escapeHtml(source)}">Update</button><button type="button" role="menuitem" class="skills-menu-item skills-uninstall" data-skill="${safeName}" data-source="${escapeHtml(source)}">Uninstall</button>` : ''}
            ${localDelete ? `<button type="button" role="menuitem" class="skills-menu-item skills-delete-local" data-skill="${safeName}" data-payload-root="${escapeHtml(payloadRoot)}">Delete</button>` : ''}
        </dialog></div>` : '';
    const primary = action.action ? `<button type="button" class="btn btn-primary skills-primary-action" data-skill="${safeName}" data-skill-action="${escapeHtml(action.action)}" ${action.keys ? `data-keys="${escapeHtml(action.keys)}"` : ''} ${action.disabled ? 'disabled' : ''}>${escapeHtml(action.label)}</button>` : '';
    // While a lifecycle job for this skill is queued/running, reflect the in-flight
    // intent and lock the control, so the toggle handler's re-render cannot snap it
    // back to the stale persisted state with no feedback.
    const lifecyclePending = Boolean(skill.lifecycle_pending);
    const toggleOn = skill.lifecycle_kind === 'disable' && lifecyclePending ? false
        : (skill.lifecycle_kind === 'enable' && lifecyclePending ? true : skill.enabled);
    const toggleLocked = Boolean(lockReason) || lifecyclePending;
    const toggleTitle = lifecyclePending ? lifecyclePendingLabel(skill.lifecycle_kind)
        : (lockReason ? `Locked: ${lockReason}` : (skill.enabled ? 'Turn skill off' : 'Turn skill on'));
    const toggle = skill.lifecycle_virtual ? '' : `<label class="skills-switch ${toggleLocked ? 'is-locked' : ''}" ${lockReason && action.action ? actionAttrs : ''} title="${escapeHtml(toggleTitle)}">
        <input type="checkbox" class="skills-toggle" role="switch" data-skill="${safeName}" ${toggleOn ? 'checked' : ''} ${toggleLocked ? 'disabled' : ''} aria-checked="${toggleOn ? 'true' : 'false'}" aria-label="${escapeHtml(lifecyclePending ? `${skill.name} (${lifecyclePendingLabel(skill.lifecycle_kind)})` : (lockReason ? `${skill.name} (locked: ${lockReason})` : skill.name))}">
        <span class="skills-switch-track" aria-hidden="true"><span class="skills-switch-thumb"></span></span>
    </label>`;
    const details = `<details class="skills-details"><summary>Show details</summary>
        <div class="skills-detail-row"><span class="skills-detail-label">Type</span><code>${escapeHtml(skill.type || 'skill')}</code> · version ${escapeHtml(skill.version || '—')} · source ${escapeHtml(source)}</div>
        <div class="skills-detail-row"><span class="skills-detail-label">Review</span>${statusBadge(skill.review_status, skill.review_gate, skill.review_profile)}${skill.review_stale ? ' <span class="skills-badge skills-badge-warn">stale</span>' : ''}</div>
        <div class="skills-detail-row"><span class="skills-detail-label">Permissions</span>${(skill.permissions || []).map((p) => `<code>${escapeHtml(p)}</code>`).join(' ') || '<i class="muted">none</i>'}</div>
        ${presenceRuntimeBlock(skill)}
        ${provenanceBlock(prov)}
    </details>`;
    return `<article class="skills-card" data-skill="${safeName}" ${reviewInProgress ? 'data-reviewing="1"' : ''} ${repairInProgress ? 'data-repairing="1"' : ''}>
        <header class="skills-card-head">
            <div class="skills-card-title"><h3>${safeName}${sourceChip(skill) ? ` ${sourceChip(skill)}` : ''}${hubBadges ? ` ${hubBadges}` : ''}</h3>${skill.description ? `<p class="skills-card-desc">${escapeHtml(skill.description)}</p>` : ''}${formatRelativeAge(installedTime(skill)) ? `<div class="skills-card-installed muted">${escapeHtml(formatRelativeAge(installedTime(skill)))}</div>` : ''}</div>
            <div class="skills-card-toggle">${statusChip(skill, action, live)}${primary}${toggle}${menu}</div>
        </header>
        ${lockReason ? `<div class="skills-lock-hint ${action.action ? 'is-clickable' : ''}" title="${escapeHtml(lockReason)}" ${actionAttrs}>Locked: ${escapeHtml(lockReason)}</div>` : ''}
        ${reviewInProgress ? '<div class="skills-review-progress" role="status" aria-live="polite"><span class="skills-review-spinner" aria-hidden="true"></span><span>Review in progress</span></div>' : ''}
        ${repairInProgress ? '<div class="skills-review-progress skills-repair-progress" role="status" aria-live="polite"><span class="skills-review-spinner" aria-hidden="true"></span><span>Repair task is being queued</span></div>' : ''}
        ${grantBlock(skill)}
        ${reviewHistory(skill)}
        ${reviewFindings(skill)}
        ${skill.lifecycle_status === 'failed' && skill.lifecycle_error ? `<div class="skills-load-error">${escapeHtml(skill.lifecycle_error)}</div>` : ''}
        ${skill.load_error && !missingGrantLoadError(skill) ? `<div class="skills-load-error">${escapeHtml(skill.load_error)}</div>` : ''}
        ${skill.health_regressed ? `<div class="skills-load-error">Regression: was live at ${escapeHtml(String((skill.last_known_good || {}).version || '?'))} (${escapeHtml(String((skill.last_known_good || {}).sha || '').slice(0, 12))}); broken after a code update.</div>` : ''}
        <footer class="skills-card-actions">${details}</footer>
    </article>`;
}

function submitHubReady(skill, githubTokenConfigured = false) {
    // Prefer the host's SSOT verdict. The additive task_start_allowed fact owns
    // admission when present; disabled remains only a compatibility projection.
    if (skill.submit_hub && typeof skill.submit_hub === 'object') {
        const submit = skill.submit_hub;
        return {
            ...submit,
            disabled: typeof submit.task_start_allowed === 'boolean'
                ? !submit.task_start_allowed
                : (typeof submit.disabled === 'boolean' ? submit.disabled : true),
        };
    }
    // Older payloads cannot know publication readiness. Keep the selected preflight
    // reachable for every otherwise supported source; it owns review/repair truth.
    const source = (skill.source || 'native').toLowerCase();
    const visible = ['external', 'self_authored', 'user_repo', 'ouroboroshub', 'clawhub'].includes(source);
    if (!visible) return { visible: false, disabled: true, reason: '' };
    if (!githubTokenConfigured) return { visible: true, disabled: true, reason: 'Configure GITHUB_TOKEN in Settings -> Secrets' };
    return {
        visible: true,
        publication_ready: false,
        task_start_allowed: true,
        disabled: false,
        reason: 'Run the selected publish preflight',
    };
}
