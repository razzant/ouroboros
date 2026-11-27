// Telegram group chooser inside Circles -> Add a place (wire-v2 Core + Hub single-skill `telegram-bot` contract):
// - Branch A ("My Telegram bot"): reads `GET /api/extensions/telegram/group_feed` only when the owner opens the
//   Telegram choice, requires explicit owner opt-in (`POST /api/extensions/telegram/group_feed` with
//   `{enabled: true, expected_account_id}`) after showing what group members see, then reads the host's `owner_bot`
//   projection source (`GET /api/extensions/telegram-bot/owner_bot_status`) and connects exact groups through
//   `place_connect.js` (`POST /api/places` with `contribution_id: 'owner_bot'`).
// - Branch B ("Another bot with its own @name"): uses the ordinary `telegram-bot` (`place`) connector and its
//   `GET /api/extensions/telegram-bot/status` route (`TELEGRAM_PUBLIC_BOT_TOKEN`), connecting exact groups with the
//   legacy `POST /api/places` request shape.
// - Skill installation, updates, enabling, and permission grants route through Extensions -> Skills / Browse.
import { fetchJson, withinRequestLifetime } from './api_client.js';
import { connectorMatchesPlace, failureText, isOwnerBotPair, PLACES_CHANGED_EVENT } from './consciousness_api.js';
import { fmt, tr } from './i18n.js';
import { openDashboard, openSkillsFor, statusLine } from './place_sheet.js';
import { skillIconHtml } from './skill_presentation.js';
import { escapeHtmlAttr as escapeHtml, grantBlocks } from './utils.js';

/** Owner words for timed-out/cancelled bounded reads or unreadable replies, without leaking internal `state read` text. */
export const requestFailureText = (error) => (error?.name === 'StateReadEnded'
    ? (error.reason === 'expired'
        ? tr('places.error.timed_out', 'The request took too long.')
        : tr('places.error.cancelled', 'The request did not finish.'))
    : error?.unreadable
        ? tr('places.error.unreadable', 'Ouroboros returned an unreadable reply.')
        : failureText(error));

const TELEGRAM_BOT_CONNECTOR = 'telegram-bot';
const TELEGRAM_NATIVE_SKILL = 'telegram';
const TELEGRAM_PUBLIC_KEY = 'TELEGRAM_PUBLIC_BOT_TOKEN';
const BOT_TOKEN_MISSING_CODE = 'Bot token missing';
const PASSIVE_LIFETIME_MS = 4000;
const STATUS_LIFETIME_MS = 10000;
const HANDLE_RE = /^@?[A-Za-z0-9_]{3,64}$/;
const GROUP_CHAT_ID_RE = /^-\d+$/;
const PRIVATE_CHAT_ID_RE = /^\d+$/;

const plainObject = (value) => (value && typeof value === 'object' && !Array.isArray(value) ? value : null);
const offline = (connector) => Boolean(connector?.readiness && connector.readiness.tone !== 'ok');
const btn = (act, label, attrs = '', tone = 'btn-default') =>
    `<button type="button" class="btn ${tone} btn-sm" data-connect-act="${act}"${attrs ? ` ${attrs}` : ''}>${escapeHtml(label)}</button>`;

/** Confirm a Telegram bot identity from `identity.sends_as`. `requireHandle` enforces a valid `@username` for Branch A. */
export function confirmedTelegramBot(identity, { expectedAccountId = '', requireHandle = false } = {}) {
    const sendsAs = plainObject(identity)?.sends_as;
    if (!plainObject(sendsAs) || sendsAs.kind !== 'bot') return null;
    const id = typeof sendsAs.id === 'string' ? sendsAs.id.trim() : '';
    const rawLabel = typeof sendsAs.label === 'string' ? sendsAs.label.trim() : '';
    if (!id || id === '*' || id.length > 200 || !rawLabel || rawLabel.length > 200) return null;
    if (expectedAccountId && id !== expectedAccountId) return null;
    const validHandle = HANDLE_RE.test(rawLabel);
    if (requireHandle && !validHandle) return null;
    const username = validHandle ? rawLabel.replace(/^@/, '') : '';
    const label = validHandle ? `@${username}` : rawLabel;
    return { id, label, username };
}

/** Only accept `https://t.me/<verified_handle>?startgroup=<non-empty>` matching the confirmed bot `@username`. */
export function validatedTelegramInviteUrl(rawUrl, botLabel) {
    if (typeof rawUrl !== 'string' || !rawUrl.trim()) return '';
    const handle = typeof botLabel === 'string' ? botLabel.trim().replace(/^@/, '') : '';
    if (!/^[A-Za-z0-9_]{3,64}$/.test(handle)) return '';
    let parsed;
    try { parsed = new URL(rawUrl.trim()); } catch { return ''; }
    if (parsed.protocol !== 'https:' || parsed.hostname.toLowerCase() !== 't.me') return '';
    if (parsed.username || parsed.password || parsed.port || parsed.hash) return '';
    if (parsed.pathname.toLowerCase() !== `/${handle.toLowerCase()}`) return '';
    const startgroup = parsed.searchParams.get('startgroup');
    if (!startgroup || !startgroup.trim()) return '';
    for (const key of parsed.searchParams.keys()) {
        if (key !== 'startgroup') return '';
    }
    return parsed.toString();
}

/** Exact negative-ID Telegram group rows for one confirmed bot account; private DMs are ignored, bad/foreign rows fail closed. */
export function telegramGroupChoices(data, expectedAccountId, { strictAccount = false } = {}) {
    const account = typeof expectedAccountId === 'string' ? expectedAccountId.trim() : '';
    if (!plainObject(data) || !account || account === '*' || !Array.isArray(data.connect)) {
        return { state: 'unavailable', rows: [] };
    }
    const rows = [];
    let hadInvalid = false;
    for (const row of data.connect) {
        if (!plainObject(row)) { hadInvalid = true; continue; }
        const rowAccount = typeof row.account_id === 'string' ? row.account_id.trim() : '';
        const conv = typeof row.conversation_id === 'string' ? row.conversation_id.trim() : '';
        const label = typeof row.label === 'string' ? row.label.trim() : '';
        if (!rowAccount || rowAccount !== account || !conv || conv === '*' || !label || label.length > 300) {
            hadInvalid = true;
            continue;
        }
        if (GROUP_CHAT_ID_RE.test(conv)) {
            rows.push({ accountId: account, conversationId: conv, label });
        } else if (!PRIVATE_CHAT_ID_RE.test(conv)) {
            hadInvalid = true;
        }
    }
    if (strictAccount && hadInvalid) return { state: 'unavailable', rows: [] };
    if (rows.length) return { state: 'known', rows };
    return { state: hadInvalid ? 'unavailable' : 'empty', rows: [] };
}

/** Parse Core v2 `GET /api/extensions/telegram/group_feed`. */
export function parseTelegramGroupFeed(raw) {
    const data = plainObject(raw);
    if (!data || data.v !== 1 || typeof data.state !== 'string'
        || typeof data.owner_bound !== 'boolean' || typeof data.enabled !== 'boolean' || typeof data.capable !== 'boolean') {
        return { state: 'unavailable', bot: null, ownerBound: false, enabled: false, capable: false, inviteUrl: '' };
    }
    // A saved consent can still be revoked when the provider identity cannot be confirmed. It is not identity evidence.
    const consent = data.consent_account_id;
    const savedConsent = typeof consent === 'string' && /^[1-9]\d*$/.test(consent) ? { consentAccountId: consent }
        : Object.hasOwn(data, 'consent_account_id') ? { consentAccountId: consent === '' ? '' : null } : {};
    const separateHeld = data.connector_token_held === true;
    // Old Core did not publish this fact. Only absence permits the old, exact Hub status read as the witness.
    const intakeReady = Object.hasOwn(data, 'intake_ready') ? data.intake_ready === true : undefined;
    if (['no_token', 'bridge_not_ready', 'identity_unconfirmed'].includes(data.state)) {
        return { state: data.state, bot: null, ownerBound: data.owner_bound, enabled: data.enabled, capable: data.capable, separateHeld, inviteUrl: '', ...savedConsent };
    }
    if (data.state !== 'ready') {
        return { state: 'unavailable', bot: null, ownerBound: false, enabled: false, capable: false, inviteUrl: '' };
    }
    const accountId = typeof data.account_id === 'string' ? data.account_id.trim() : '';
    const bot = accountId ? confirmedTelegramBot(data.identity, { expectedAccountId: accountId, requireHandle: true }) : null;
    if (!bot) {
        return { state: 'identity_unconfirmed', bot: null, ownerBound: data.owner_bound, enabled: data.enabled, capable: data.capable, separateHeld, inviteUrl: '', ...savedConsent };
    }
    if (!data.owner_bound) {
        return { state: 'unpaired', bot, ownerBound: false, enabled: data.enabled, capable: data.capable, separateHeld, inviteUrl: '', ...savedConsent };
    }
    return {
        state: 'ready',
        accountId: bot.id,
        bot,
        ownerBound: true,
        enabled: data.enabled,
        capable: data.capable,
        ...(intakeReady !== undefined ? { intakeReady } : {}),
        separateHeld,
        ...savedConsent,
        inviteUrl: validatedTelegramInviteUrl(data.invite_url, bot.label),
    };
}

/** Parse Hub v2 `GET /api/extensions/telegram-bot/owner_bot_status`. */
export function parseOwnerBotStatus(raw, expectedAccountId = '') {
    const data = plainObject(raw);
    if (!data || data.v !== 1) {
        return { state: 'unavailable', reason: 'invalid_payload', bot: null, inviteUrl: '', choices: { state: 'unavailable', rows: [] } };
    }
    const reason = typeof data.reason === 'string' ? data.reason.trim() : '';
    if (data.enabled === false && reason === 'feed_off' && data.connect === null) {
        return { state: 'feed_off', reason, bot: null, inviteUrl: '', choices: { state: 'unavailable', rows: [] } };
    }
    if (data.enabled !== true || reason !== '') {
        return { state: 'unavailable', reason: reason || 'unavailable', bot: null, inviteUrl: '', choices: { state: 'unavailable', rows: [] } };
    }
    const accountId = typeof data.account_id === 'string' ? data.account_id.trim() : '';
    if (!accountId || (expectedAccountId && accountId !== expectedAccountId)) {
        return { state: 'account_mismatch', reason: 'account_mismatch', bot: null, inviteUrl: '', choices: { state: 'unavailable', rows: [] } };
    }
    const bot = confirmedTelegramBot(data.identity, { expectedAccountId: accountId, requireHandle: true });
    if (!bot) {
        return { state: 'unavailable', reason: 'identity_unconfirmed', bot: null, inviteUrl: '', choices: { state: 'unavailable', rows: [] } };
    }
    const choices = telegramGroupChoices(data, bot.id, { strictAccount: true });
    if (choices.state === 'unavailable') {
        return { state: 'unavailable', reason: 'invalid_connect', bot, inviteUrl: '', choices };
    }
    return {
        state: choices.rows.length ? 'known' : 'empty',
        reason: '',
        accountId: bot.id,
        bot,
        inviteUrl: validatedTelegramInviteUrl(data.invite_url, bot.label),
        choices,
    };
}

/** Parse Hub `GET /api/extensions/telegram-bot/status` for Branch B (`TELEGRAM_PUBLIC_BOT_TOKEN`).
 *  Follows typed `link` (`waiting_for_token`, `connecting`, ...) and typed `connect_unavailable === 'Bot token missing'`
 *  without parsing human prose; preserves unknown (`null`/absent `connect`) vs known `[]`. */
export function parseSeparateBotStatus(raw) {
    const data = plainObject(raw);
    if (!data || data.error) {
        return { state: 'unavailable', link: '', bot: null, readiness: null, choices: { state: 'unavailable', rows: [] } };
    }
    const readiness = typeof data.text === 'string' && data.text.trim()
        ? { text: data.text.trim(), tone: typeof data.tone === 'string' && data.tone ? data.tone : 'muted', authored: true } : null;
    const link = typeof data.link === 'string' ? data.link.trim() : '';
    const missingToken = link === 'waiting_for_token'
        || (!link && ((data.connect === null && data.connect_unavailable === BOT_TOKEN_MISSING_CODE)
            || data.runtime_state === 'waiting_for_token'));
    if (missingToken) {
        return { state: 'unconfigured', link: link || 'waiting_for_token', bot: null, readiness, choices: { state: 'unavailable', rows: [] } };
    }
    const connecting = link === 'connecting' || link === 'not_started' || (!link && data.bot_unconfirmed === true);
    const bot = data.bot_unconfirmed === true ? null : confirmedTelegramBot(data.identity);
    if (!bot) {
        return connecting
            ? { state: 'connecting', link: link || 'connecting', bot: null, readiness, choices: { state: 'unavailable', rows: [] } }
            : { state: 'unavailable', link, bot: null, readiness, choices: { state: 'unavailable', rows: [] } };
    }
    if (!Array.isArray(data.connect)) {
        return { state: connecting ? 'connecting' : 'unavailable', link, bot, readiness, choices: { state: 'unavailable', rows: [] } };
    }
    const choices = telegramGroupChoices(data, bot.id);
    return {
        state: choices.state === 'unavailable' ? 'unavailable' : (choices.rows.length ? 'known' : 'empty'),
        link,
        bot,
        readiness,
        choices,
    };
}

function classifyNativeFeedFailure(httpStatus, tgNativeSkill) {
    if (tgNativeSkill?.enabled === false) return 'telegram_disabled';
    if (tgNativeSkill?.enabled === true && tgNativeSkill?.live_loaded === false) return 'telegram_not_live';
    if (httpStatus === 404 && tgNativeSkill?.enabled === true && tgNativeSkill?.live_loaded === true) return 'old_core';
    return 'unavailable';
}

const isTelegramBotGroupPlace = (place) => place.connector === TELEGRAM_BOT_CONNECTOR
    && ['', 'place', 'owner_bot'].includes(place.contributionId ?? place.contribution_id ?? '')
    && GROUP_CHAT_ID_RE.test(place.conversationId);

export function telegramFacts(view = {}) {
    const connectors = Array.isArray(view.connectors) ? view.connectors : [];
    const places = Array.isArray(view.places) ? view.places : [];
    const ownerBotConnector = connectors.find((c) => isOwnerBotPair(c)) || null;
    const tgBotConnector = connectors.find((c) => c.connector === TELEGRAM_BOT_CONNECTOR && !isOwnerBotPair(c)) || null;
    const skills = Array.isArray(view.tg?.skills) ? view.tg.skills : null;
    const tgBotEntry = skills?.find((s) => s?.name === TELEGRAM_BOT_CONNECTOR) || null;
    const tgBotQueued = Boolean(tgBotEntry && (tgBotEntry.lifecycle_virtual === true || tgBotEntry.lifecycle_pending));
    const tgBotSkill = tgBotEntry && !tgBotQueued ? tgBotEntry : null;
    const tgNativeSkill = skills?.find((s) => s?.name === TELEGRAM_NATIVE_SKILL && !s?.lifecycle_virtual) || null;
    const groupPlaces = places.filter(isTelegramBotGroupPlace);
    const shown = Boolean(ownerBotConnector || tgBotConnector || tgBotEntry || tgNativeSkill || groupPlaces.length);
    const icon = ownerBotConnector?.icon || tgBotConnector?.icon || tgBotEntry?.icon || tgNativeSkill?.icon || null;
    return { shown, icon, ownerBotConnector, tgBotConnector, tgBotSkill, tgBotQueued, tgNativeSkill, groupPlaces };
}

export function telegramGroupsEntryHtml(info, places = []) {
    if (!info?.shown) return '';
    const n = info.groupPlaces?.length ?? places.filter(isTelegramBotGroupPlace).length;
    const ready = Boolean(info.ownerBotConnector || (info.tgBotConnector && !offline(info.tgBotConnector)));
    const state = n ? fmt(n === 1 ? 'Connected · {n} group' : 'Connected · {n} groups', { n })
        : ready ? tr('places.tg.state.ready', 'Choose a bot') : tr('places.tg.state.setup', 'Set up a bot');
    // This is a count of saved connections, not a liveness claim for either bot.
    const tone = 'muted';
    const action = btn('open-tg-groups', n ? tr('places.tg.choose_group', 'Choose group') : tr('places.tg.choose_bot', 'Choose bot'));
    return `<div class="harness-account-row place-connector-row" data-connector-entry="telegram-groups">
        <div class="harness-account-main"><span class="place-glyph" aria-hidden="true">${skillIconHtml(info.icon, 18)}</span><strong>${escapeHtml(tr('places.tg.title', 'Telegram Groups'))}</strong></div>
        <div class="harness-account-meta"><span class="ui-status" data-tone="${tone}">${escapeHtml(state)}</span></div>
        <div class="harness-account-actions">${action}</div>
    </div>`;
}

function groupRowsHtml(places, connector, rows, busy, connectDisabled = false) {
    const blocked = busy || connectDisabled;
    return `<div class="place-accounts" data-tg-group-list>${rows.map((row, index) => {
        const existing = places.find((place) => connectorMatchesPlace(connector, place)
            && place.accountId === row.accountId && place.conversationId === row.conversationId);
        const state = existing ? `<div class="harness-account-meta"><span class="ui-status" data-tone="ok">${tr('places.connected', 'Connected')}</span></div>` : '';
        const action = existing
            ? btn('adopt', tr('places.choose', 'Choose'), `data-place-id="${escapeHtml(existing.id)}"${busy ? ' disabled' : ''}`)
            : btn('connect', tr('places.connect.action', 'Connect'), `data-choice="${index}"${blocked ? ' disabled' : ''}`,
                rows.length === 1 && !connectDisabled ? 'btn-primary' : 'btn-default');
        return `<div class="harness-account-row place-connector-row">
            <div class="harness-account-main"><strong data-i18n-authored>${escapeHtml(row.label)}</strong></div>
            ${state}<div class="harness-account-actions">${action}</div>
        </div>`;
    }).join('')}</div>`;
}

function personalBodyHtml(view, info, connectDisabled = false) {
    const { tg = {}, places = [], busy = false } = view;
    const feed = tg.groupFeed;
    const { ownerBotConnector, tgBotConnector, tgBotSkill, tgBotQueued } = info;
    if (tg.consentPending) {
        return `${statusLine({ text: tr('places.tg.consent_unconfirmed', 'The group discovery change is not confirmed yet.'), tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '')}</div>`;
    }
    if (!feed) {
        return `<p class="place-sheet-note" role="status">${escapeHtml(tr('places.tg.personal.loading', 'Checking your personal Telegram bot…'))}</p>`;
    }
    if (feed.state === 'telegram_disabled') {
        return `<p class="place-sheet-note">${escapeHtml(tr('places.tg.personal.skill_off', 'Your personal Telegram bridge is off in Skills. Turn it on there, or choose another bot with its own @name.'))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.connector.enable', 'Enable in Skills'), 'data-skill="telegram"')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (feed.state === 'telegram_not_live') {
        return `<p class="place-sheet-note">${escapeHtml(tr('places.tg.personal.skill_not_live', 'Your personal Telegram bridge is enabled in Skills, but it is not running right now. Check Telegram in Skills, or choose another bot with its own @name.'))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.tg.open_telegram', 'Open Telegram in Skills'), 'data-skill="telegram"')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (feed.state === 'old_core') {
        return `${statusLine({ text: tr('places.tg.personal.old_core', 'Sharing your personal Telegram bot with groups needs an Ouroboros update. Another bot with its own @name works now.'), tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-open-updates', tr('places.tg.open_updates', 'Open Updates'))}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (feed.state === 'no_token' || feed.state === 'unpaired') {
        return `<p class="place-sheet-note">${escapeHtml(tr('places.tg.personal.unpaired', 'Pair your personal Telegram bot in a private chat first so Ouroboros can confirm its @name, or use another bot with its own @name.'))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.tg.open_telegram', 'Open Telegram in Skills'), 'data-skill="telegram"')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (feed.state === 'bridge_not_ready' || feed.state === 'identity_unconfirmed') {
        return `<p class="place-sheet-note" role="status">${escapeHtml(tr('places.tg.personal.starting', 'Your personal Telegram bot is still confirming its @name. Check again in a moment, or use another bot with its own @name.'))}</p>
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}${btn('tg-open-skill', tr('places.tg.open_telegram', 'Open Telegram in Skills'), 'data-skill="telegram"', 'btn-ghost')}</div>`;
    }
    if (feed.state !== 'ready' || !feed.bot) {
        return `${statusLine({ text: tr('places.tg.personal.unavailable', 'Could not read your personal Telegram bot status.'), tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.retry', 'Try again'))}</div>`;
    }
    if (consentUnreadable(feed)) {
        return `${statusLine({ text: tr('places.tg.consent_unreadable', 'Could not read the saved group discovery setting.'), tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '')}</div>`;
    }

    const handle = feed.bot.label;
    if (tgBotQueued) {
        return `<p class="place-sheet-note" role="status">${escapeHtml(fmt('Telegram Bot is still installing or updating in Skills for {bot}. Wait for it to finish there, then check again.', { bot: handle }))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.tg.open_in_skills', 'Open in Skills'), 'data-skill="telegram-bot"')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (!tgBotSkill && !tgBotConnector && !ownerBotConnector) {
        return `<p class="place-sheet-note">${escapeHtml(fmt('Group places use the official Telegram Bot connector. Install telegram-bot from Browse to use {bot} in groups; your private chat stays private.', { bot: handle }))}</p>
            <div class="place-sheet-actions">${btn('tg-browse-bot', tr('places.tg.find_bot', 'Find telegram-bot in Extensions'), '', 'btn-primary')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (tgBotSkill && tgBotSkill.enabled === false) {
        return `<p class="place-sheet-note">${escapeHtml(fmt('Turn on Telegram Bot in Skills so {bot} can receive group messages here. No second bot token is needed.', { bot: handle }))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.connector.enable', 'Enable in Skills'), 'data-skill="telegram-bot"', 'btn-primary')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    // A running legacy connector may still have historical review findings, or no local verdict at all.
    // Those are not a repair requirement for the running version; its missing owner_bot contribution needs an update.
    const oldHub = !ownerBotConnector && tgBotSkill?.live_loaded === true && !grantBlocks(tgBotSkill) && !tgBotSkill.load_error;
    if (!ownerBotConnector && tgBotSkill && !oldHub && (grantBlocks(tgBotSkill) || tgBotSkill.review_status === 'blockers' || tgBotSkill.review_status === 'pending' || tgBotSkill.load_error)) {
        const blockedWords = grantBlocks(tgBotSkill)
            ? fmt('Approve access for Telegram Bot in Skills to use {bot} in groups. Your personal bot token stays in Telegram — do not enter a second token.', { bot: handle })
            : fmt('Telegram Bot needs review or repair in Skills before {bot} can connect groups here.', { bot: handle });
        return `<p class="place-sheet-note">${escapeHtml(blockedWords)}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.tg.review_in_skills', 'Review in Skills'), 'data-skill="telegram-bot"', 'btn-primary')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (!ownerBotConnector || offline(ownerBotConnector) || !feed.capable) {
        const text = oldHub ? fmt('Telegram Bot needs an update in Skills before {bot} can connect groups here.', { bot: handle })
            : fmt('Telegram Bot is not ready to list groups for {bot} yet.', { bot: handle });
        const door = oldHub ? tr('places.tg.update_in_skills', 'Update in Skills') : tr('places.tg.open_in_skills', 'Open in Skills');
        return `${statusLine({ text, tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-open-skill', door, 'data-skill="telegram-bot"')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }

    const ownerStatus = tg.ownerStatus;
    if (!feed.enabled) {
        if (offAccount(feed)) return `<p class="place-sheet-note">${escapeHtml(tr('places.tg.revoke_previous', 'Turn off groups for the previous bot before allowing this bot.'))}</p>`;
        return `<p class="place-sheet-note">${escapeHtml(fmt('Allowing group updates lets Ouroboros list groups where {bot} is a member. You connect each group separately.', { bot: handle }))}</p>
            <div class="place-sheet-actions">${btn('tg-opt-in', fmt('Allow {bot} in groups', { bot: handle }), busy ? 'disabled' : '', 'btn-primary')}</div>`;
    }
    if (feed.intakeReady === false) {
        return `<p class="place-sheet-note" role="status">${escapeHtml(fmt('Groups are allowed for {bot}. Check again before sending a message in the group.', { bot: handle }))}</p>
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '', 'btn-primary')}</div>`;
    }
    if (!ownerStatus) {
        return `<p class="place-sheet-note" role="status">${escapeHtml(fmt('Reading groups seen by {bot}…', { bot: handle }))}</p>`;
    }
    // Never resubmit opt-in POST when native `group_feed` already reports `enabled: true`; offer refresh instead.
    if (ownerStatus.state === 'feed_off') {
        return `<p class="place-sheet-note" role="status">${escapeHtml(fmt('{bot} is allowed in groups; waiting for Telegram Bot to pick up the group feed.', { bot: handle }))}</p>
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '', 'btn-primary')}</div>`;
    }
    if (ownerStatus.state === 'account_mismatch' || ownerStatus.state === 'unavailable') {
        const msg = ownerStatus.reason === 'update_required'
            ? fmt('Telegram Bot needs an update in Skills to read {bot}’s groups.', { bot: handle })
            : ownerStatus.reason === 'account_mismatch'
                ? tr('places.tg.personal.mismatch', 'Your paired Telegram bot changed while reading groups. Check again to refresh.')
                : fmt('Could not read {bot}’s group list yet.', { bot: handle });
        const updateDoor = ownerStatus.reason === 'update_required' ? btn('tg-open-skill', tr('places.tg.update_in_skills', 'Update in Skills'), 'data-skill="telegram-bot"') : '';
        return `${statusLine({ text: msg, tone: 'warn' })}
            <div class="place-sheet-actions">${updateDoor}${btn('tg-refresh', tr('places.retry', 'Try again'))}</div>`;
    }

    const inviteUrl = ownerStatus.inviteUrl || feed.inviteUrl;
    const inviteLink = inviteUrl
        ? `<a class="btn btn-default btn-sm place-connect-link" href="${escapeHtml(inviteUrl)}" target="_blank" rel="noopener noreferrer" data-connect-invite="personal">${escapeHtml(fmt('Add {bot} to a Telegram group', { bot: handle }))}</a>` : '';
    if (ownerStatus.state === 'empty') {
        return `<ol class="place-connect-steps">
                <li>${escapeHtml(fmt('Add {bot} to the Telegram group you want to connect.', { bot: handle }))}</li>
                <li>${escapeHtml(fmt('Send a message or @mention {bot} in that group so Telegram delivers an update to it.', { bot: handle }))}</li>
                <li>${escapeHtml(tr('places.tg.step_connect', 'Press Check again here, then press Connect beside the group.'))}</li>
            </ol>
            <div class="place-sheet-actions">${inviteLink}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '')}</div>`;
    }
    return `${groupRowsHtml(places, ownerBotConnector, ownerStatus.choices.rows, busy, connectDisabled)}
        <div class="place-sheet-actions">${inviteLink}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '')}</div>`;
}

const discoveryOn = (feed) => feed?.state === 'ready' && feed.enabled
    && (feed.consentAccountId === undefined || feed.consentAccountId === feed.bot.id);
const canListPersonalGroups = (feed, connector) => discoveryOn(feed) && feed.capable && feed.intakeReady !== false
    && connector && !offline(connector);
const consentUnreadable = (feed) => Boolean(feed?.state === 'ready' && (
    feed.consentAccountId === null
    || (feed.enabled && (feed.consentAccountId ?? feed.bot.id) !== feed.bot.id)
    || (!feed.enabled && feed.consentAccountId === feed.bot.id)
));
const offAccount = (feed) => feed?.consentAccountId === undefined
    ? (feed?.enabled && feed.state === 'ready' ? feed.bot.id : '') : feed.consentAccountId;
const offLabel = (feed) => {
    const id = offAccount(feed);
    if (feed?.bot && id === feed.bot.id) return fmt('Turn off groups for {bot}', { bot: feed.bot.label });
    if (feed?.bot && id && id !== feed.bot.id) return fmt('Turn off groups for the previous bot (ID {id})', { id });
    return fmt('Turn off groups for bot ID {id}', { id });
};
const separateHeld = (tg) => tg?.separateStatus?.link === 'conflict_with_owner_bot'
    || (tg?.separateStatus?.state === 'unconfigured' && tg?.groupFeed?.separateHeld === true);

function separateBodyHtml(view, info, connectDisabled = false) {
    const { tg = {}, places = [], busy = false } = view;
    const { tgBotConnector, tgBotSkill, tgBotQueued } = info;
    if (tgBotQueued) {
        return `<p class="place-sheet-note" role="status">${escapeHtml(tr('places.tg.separate.installing', 'Telegram Bot is still installing or updating in Skills. Wait for it to finish there, then check again.'))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.tg.open_in_skills', 'Open in Skills'), 'data-skill="telegram-bot"')}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (!tgBotSkill && !tgBotConnector) {
        return `<ol class="place-connect-steps">
                <li>${escapeHtml(tr('places.tg.separate.step_botfather', 'Create a new bot with @BotFather in Telegram (/newbot) so it has its own @name.'))}</li>
                <li>${escapeHtml(tr('places.tg.separate.step_token', 'Save its token as TELEGRAM_PUBLIC_BOT_TOKEN in Settings › Accounts and keys (never reuse your personal bot token).'))}</li>
                <li>${escapeHtml(tr('places.tg.separate.step_install', 'Install telegram-bot in Extensions, add the bot to your group, and press Check again.'))}</li>
            </ol>
            <div class="place-sheet-actions">${btn('tg-browse-bot', tr('places.tg.find_bot', 'Find telegram-bot in Extensions'), '', 'btn-primary')}${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if ((tgBotSkill && (tgBotSkill.enabled === false || grantBlocks(tgBotSkill) || tgBotSkill.load_error || tgBotSkill.live_loaded === false))
        || (!tgBotConnector && tgBotSkill)) {
        const label = grantBlocks(tgBotSkill) ? tr('places.tg.review_in_skills', 'Review in Skills')
            : tgBotSkill?.enabled === false ? tr('places.connector.enable', 'Enable in Skills')
                : tr('places.tg.open_in_skills', 'Open in Skills');
        return `<p class="place-sheet-note">${escapeHtml(tr('places.tg.separate.enable_first', 'Make sure Telegram Bot is running in Skills and TELEGRAM_PUBLIC_BOT_TOKEN is saved in Settings › Accounts and keys (never reuse your personal bot token).'))}</p>
            <div class="place-sheet-actions">${btn('tg-open-skill', label, 'data-skill="telegram-bot"', 'btn-primary')}${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }

    const settingsDoor = tgBotConnector?.settings ? btn('tg-bot-settings', tr('places.connector.settings', 'Open connector settings')) : '';
    const sep = tg.separateStatus;
    if (separateHeld(tg)) {
        return `${statusLine({ text: tr('places.tg.separate.held', 'The separate bot is held because its token belongs to your personal bot.'), tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    const canReadSeparate = Boolean(tgBotConnector?.source?.url && (!offline(tgBotConnector) || tgBotSkill?.live_loaded === true));
    if (canReadSeparate && !sep) {
        return `<p class="place-sheet-note" role="status">${escapeHtml(tr('places.tg.separate.loading', 'Reading the separate Telegram bot…'))}</p>`;
    }
    if (!sep) {
        return `${tgBotConnector?.readiness ? statusLine(tgBotConnector.readiness) : ''}
            <div class="place-sheet-actions">${btn('tg-open-skill', tr('places.tg.open_in_skills', 'Open in Skills'), 'data-skill="telegram-bot"')}${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (sep.state === 'unconfigured') {
        const readiness = sep.readiness || tgBotConnector?.readiness;
        return `${readiness ? statusLine(readiness) : ''}
            <ol class="place-connect-steps">
                <li>${escapeHtml(tr('places.tg.separate.step_botfather', 'Create a new bot with @BotFather in Telegram (/newbot) so it has its own @name.'))}</li>
                <li>${escapeHtml(tr('places.tg.separate.step_token', 'Save its token as TELEGRAM_PUBLIC_BOT_TOKEN in Settings › Accounts and keys (never reuse your personal bot token).'))}</li>
            </ol>
            <div class="place-sheet-actions">${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'), '', 'btn-primary')}${settingsDoor}${btn('tg-refresh', tr('places.tg.check_again', 'Check again'))}</div>`;
    }
    if (sep.state === 'connecting') {
        return `${sep.readiness ? statusLine(sep.readiness) : ''}
            <p class="place-sheet-note" role="status">${escapeHtml(tr('places.tg.separate.connecting', 'Telegram Bot is connecting and confirming its @name. Check again in a moment.'))}</p>
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '', 'btn-primary')}${settingsDoor}${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}</div>`;
    }
    if (sep.state === 'unavailable') {
        const status = sep.readiness || { text: tr('places.tg.separate.unavailable', 'Could not read the separate Telegram bot status.'), tone: 'warn' };
        return `${statusLine(status)}
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.retry', 'Try again'))}${settingsDoor}${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}</div>`;
    }

    const botLabel = sep.bot.label;
    if (sep.state === 'empty') {
        return `<ol class="place-connect-steps">
                <li>${escapeHtml(fmt('Open your Telegram group and add {bot} as a member.', { bot: botLabel }))}</li>
                <li>${escapeHtml(fmt('Send a message or @mention {bot} in that group so Telegram delivers an update to it.', { bot: botLabel }))}</li>
                <li>${escapeHtml(tr('places.tg.step_connect', 'Press Check again here, then press Connect beside the group.'))}</li>
            </ol>
            <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '', 'btn-primary')}${btn('tg-open-keys', tr('places.tg.open_keys', 'Open Accounts and keys'))}${settingsDoor}</div>`;
    }
    return `${groupRowsHtml(places, tgBotConnector, sep.choices.rows, busy, connectDisabled)}
        <div class="place-sheet-actions">${btn('tg-refresh', tr('places.tg.check_again', 'Check again'), busy ? 'disabled' : '')}${settingsDoor}</div>`;
}

export function telegramGroupsHtml(view, silentNote, circlePickerHtml = '', connectDisabled = false) {
    const info = telegramFacts(view);
    const branch = view.tgBranch === 'separate' ? 'separate' : 'personal';
    const feed = view.tg?.groupFeed;
    if (view.tgConnectPending) {
        const pendingConnect = view.tgConnectPending;
        const text = fmt('Could not confirm whether {group} was connected.', { group: pendingConnect.choice.label });
        return `<section class="place-sheet-section" data-tg-groups-sheet>
            <div class="place-sheet-actions">${btn('back', tr('places.connect.back', 'Back'), view.busy ? 'disabled' : '')}</div>
            ${statusLine({ text, tone: 'warn' })}
            <div class="place-sheet-actions">${btn('tg-check-connect', tr('places.tg.check_again', 'Check again'), view.busy ? 'disabled' : '')}</div></section>`;
    }
    const personalMeta = feed?.state === 'ready' && feed.bot
        ? fmt('Group members see {bot}; your private chat stays private.', { bot: feed.bot.label })
        : feed?.state === 'old_core' ? tr('places.tg.meta.old_core', 'Needs Ouroboros update')
            : feed?.state === 'telegram_disabled' ? tr('places.tg.meta.off', 'Off in Skills')
                : feed?.state === 'telegram_not_live' ? tr('places.tg.meta.not_live', 'Not running in Skills')
                    : feed?.state === 'no_token' || feed?.state === 'unpaired' ? tr('places.tg.meta.unpaired', 'Pair in private chat first')
                        : tr('places.tg.meta.personal', 'Uses the bot paired to your private Telegram chat');
    const sepBot = view.tg?.separateStatus?.bot;
    const separateMeta = sepBot ? fmt('{bot} · separate bot for groups', { bot: sepBot.label })
        : tr('places.tg.meta.separate', 'A separate bot just for groups');
    const optionRow = (key, title, meta) => {
        const chosen = branch === key;
        const label = chosen ? tr('places.tg.chosen', 'Chosen') : tr('places.tg.use_bot', 'Use this bot');
        return `<div class="harness-account-row place-connector-row" data-tg-option="${key}">
            <div class="harness-account-main"><strong>${escapeHtml(title)}</strong></div>
            <div class="harness-account-meta"><span class="meta-fact">${escapeHtml(meta)}</span></div>
            <div class="harness-account-actions">${btn('tg-branch', label, `data-branch="${key}" aria-pressed="${chosen}"${view.busy ? ' disabled' : ''}`)}</div>
        </div>`;
    };
    return `<section class="place-sheet-section" data-tg-groups-sheet>
        <div class="place-sheet-actions">${btn('back', tr('places.connect.back', 'Back'))}</div>
        <h3>${escapeHtml(tr('places.where', 'Where'))} · ${escapeHtml(tr('places.tg.title', 'Telegram Groups'))}</h3>
        <p class="place-sheet-note">${escapeHtml(silentNote)}</p>
        ${circlePickerHtml}
        <div class="place-connect-group" role="group" aria-label="${escapeHtml(tr('places.tg.which_bot', 'Which bot joins the group?'))}">
            <p class="place-connect-subheading">${escapeHtml(tr('places.tg.which_bot', 'Which bot joins the group?'))}</p>
            <div class="place-accounts">
                ${optionRow('personal', tr('places.tg.branch.personal', 'My Telegram bot'), personalMeta)}
                ${optionRow('separate', tr('places.tg.branch.separate', 'Another bot with its own @name'), separateMeta)}
            </div>
        </div>
        <div class="place-connect-group" data-tg-branch-panel="${branch}">
            ${branch === 'separate' ? separateBodyHtml(view, info, connectDisabled) : personalBodyHtml(view, info, connectDisabled)}
            ${branch === 'personal' && !view.tg?.consentPending && !consentUnreadable(feed) && offAccount(feed)
                ? `<div class="place-sheet-actions">${btn('tg-opt-out', offLabel(feed), view.busy ? 'disabled' : '')}</div>
                <p class="place-sheet-note">${escapeHtml(tr('places.tg.off_scope', 'Turning groups off stops receiving group messages and new Ouroboros replies in all groups using that bot. Your private chat is unchanged.'))}</p>` : ''}
            ${branch === 'separate' && info.tgBotConnector && !offline(info.tgBotConnector) && !separateHeld(view.tg)
                ? `<div class="place-sheet-actions">${btn('tg-other-chats', tr('places.tg.other_chats', 'Other Telegram chats'), view.busy ? 'disabled' : '')}</div>` : ''}
        </div>
        ${statusLine(view.status)}
    </section>`;
}

export function createTelegramGroupsController({
    view, sheet, signal, read, readStatus, openSettings, openConnectorSettings, paint, enrich,
    nextGen, getGen, telegram, enabledByDefault = true, onSetup,
}) {
    if (telegram === false || (!telegram && !enabledByDefault)) {
        return {
            readPassive: async () => {},
            refresh: async () => {},
            cancelInFlight: () => {},
            handleAction: async (act) => {
                if (act === 'open-tg-groups') { view.tgOpen = true; paint(); return true; }
                if (act === 'tg-branch') { view.tgBranch = 'separate'; paint(); return true; }
                return false;
            },
        };
    }
    const api = {
        readExtensions: telegram?.readExtensions || ((init) => fetchJson('/api/extensions', { cache: 'no-store', ...init })),
        readGroupFeed: telegram?.readGroupFeed || ((init) => fetchJson('/api/extensions/telegram/group_feed', { cache: 'no-store', ...init }, { rejectOkFalse: true })),
        writeGroupFeed: telegram?.writeGroupFeed || ((enabled, expectedAccountId, { signal } = {}) =>
            fetchJson('/api/extensions/telegram/group_feed', {
                method: 'POST', headers: { 'Content-Type': 'application/json' }, signal,
                body: JSON.stringify({ enabled, expected_account_id: expectedAccountId }),
            }, { rejectOkFalse: true, requireBody: true })),
    };
    let statusAbort = null;
    let skillsSeq = 0;
    const cancelInFlight = () => { statusAbort?.abort(); statusAbort = null; };
    signal?.addEventListener?.('abort', cancelInFlight, { once: true });
    const freshSignal = () => {
        cancelInFlight();
        statusAbort = new AbortController();
        return statusAbort.signal;
    };

    const syncSelectedChoices = () => {
        const info = telegramFacts(view);
        if (!view.tgOpen || view.tgOther) return;
        if (view.tgConnectPending) {
            view.selected = null; view.choices = null; view.rawStatus = null;
        } else if (view.tgBranch === 'personal' && !view.tg.consentPending && canListPersonalGroups(view.tg.groupFeed, info.ownerBotConnector) && view.tg.ownerStatus?.state === 'known') {
            view.selected = info.ownerBotConnector;
            view.choices = view.tg.ownerStatus.choices;
            view.rawStatus = view.tg.rawOwnerStatus;
        } else if (view.tgBranch === 'separate' && info.tgBotConnector && view.tg.separateStatus?.state === 'known') {
            view.selected = info.tgBotConnector;
            view.choices = view.tg.separateStatus.choices;
            view.rawStatus = view.tg.rawSeparateStatus;
        } else {
            view.selected = null;
            view.choices = null;
            view.rawStatus = null;
        }
    };

    const reconcileFeedFailureWithSkills = () => {
        const feed = view.tg.groupFeed;
        if (!feed || feed.httpStatus === undefined) return;
        const { tgNativeSkill } = telegramFacts(view);
        feed.state = classifyNativeFeedFailure(feed.httpStatus, tgNativeSkill);
    };

    // Advisory `/api/extensions` read: never blocks branch discovery or status reads, and never overwrites newer reads.
    const kickSkillsRead = (lifetimeMs = PASSIVE_LIFETIME_MS) => {
        const seq = ++skillsSeq;
        return (async () => {
            try {
                const data = await withinRequestLifetime((transport) => api.readExtensions({ signal: transport }), { signal, lifetimeMs });
                if (sheet.closed || signal?.aborted || seq !== skillsSeq || !Array.isArray(data?.skills)) return;
                view.tg.skills = data.skills;
                reconcileFeedFailureWithSkills();
                if (view.tgOpen) {
                    syncSelectedChoices();
                    paint();
                } else {
                    enrich();
                }
            } catch { /* passive skills index is advisory; connector projections and routes remain authoritative */ }
        })();
    };

    const readSeparateStatus = async (runGen, activeSignal) => {
        const info = telegramFacts(view);
        const target = info.tgBotConnector;
        const canRead = Boolean(target?.source?.url && (!offline(target) || info.tgBotSkill?.live_loaded === true));
        if (!canRead) {
            syncSelectedChoices();
            paint();
            return;
        }
        view.tg.separateStatus = null; view.tg.rawSeparateStatus = null; syncSelectedChoices(); paint();
        try {
            const raw = await withinRequestLifetime((transport) => readStatus(target, { signal: transport }),
                { signal: activeSignal, lifetimeMs: STATUS_LIFETIME_MS });
            if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
            view.tg.rawSeparateStatus = raw;
            view.tg.separateStatus = parseSeparateBotStatus(raw);
        } catch {
            if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
            view.tg.separateStatus = { state: 'unavailable', link: '', bot: null, readiness: null, choices: { state: 'unavailable', rows: [] } };
        }
        // Only when B reports `waiting_for_token`, check native `group_feed.connector_token_held` so a duplicate token
        // withheld by core is explained without making a healthy separate bot depend on A.
        if (view.tg.separateStatus?.state === 'unconfigured' && !view.tg.groupFeed) {
            const { tgNativeSkill } = telegramFacts(view);
            if (tgNativeSkill?.enabled !== false && !(tgNativeSkill?.enabled === true && tgNativeSkill?.live_loaded === false)) {
                try {
                    const rawFeed = await withinRequestLifetime((transport) => api.readGroupFeed({ signal: transport }),
                        { signal: activeSignal, lifetimeMs: PASSIVE_LIFETIME_MS });
                    if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
                    const parsed = parseTelegramGroupFeed(rawFeed);
                    if (parsed.state !== 'unavailable') view.tg.groupFeed = parsed;
                } catch { /* Advisory held-token check: waiting_for_token stays usable without native group_feed. */ }
            }
        }
        syncSelectedChoices();
        paint();
    };

    const readBranchStatus = async (runGen, activeSignal) => {
        if (view.tgBranch === 'separate') {
            await readSeparateStatus(runGen, activeSignal);
            return;
        }

        // Branch A (`personal`): recover initial native `group_feed` discovery whenever `groupFeed` is still unset,
        // and re-read read-only while an unconfirmed consent write is still pending (e.g. after returning from B).
        if (!view.tg.groupFeed || view.tg.consentPending) {
            syncSelectedChoices();
            paint();
            const { tgNativeSkill } = telegramFacts(view);
            if (tgNativeSkill?.enabled === false) {
                view.tg.groupFeed = { state: 'telegram_disabled', httpStatus: 0, bot: null, ownerBound: false, enabled: false, capable: false, inviteUrl: '' };
            } else if (tgNativeSkill?.enabled === true && tgNativeSkill?.live_loaded === false) {
                view.tg.groupFeed = { state: 'telegram_not_live', httpStatus: 0, bot: null, ownerBound: false, enabled: false, capable: false, inviteUrl: '' };
            } else {
                try {
                    const rawFeed = await withinRequestLifetime((transport) => api.readGroupFeed({ signal: transport }),
                        { signal: activeSignal, lifetimeMs: STATUS_LIFETIME_MS });
                    if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
                    view.tg.groupFeed = parseTelegramGroupFeed(rawFeed);
                } catch (error) {
                    if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
                    const httpStatus = error?.status ?? 0;
                    view.tg.groupFeed = {
                        state: classifyNativeFeedFailure(httpStatus, telegramFacts(view).tgNativeSkill),
                        httpStatus,
                        bot: null, ownerBound: false, enabled: false, capable: false, inviteUrl: '',
                    };
                }
            }
        }

        const info = telegramFacts(view);
        const feed = view.tg.groupFeed;
        settleConsent(feed);
        if (view.tg.consentPending || !canListPersonalGroups(feed, info.ownerBotConnector)) {
            syncSelectedChoices();
            paint();
            return;
        }
        view.tg.ownerStatus = null; view.tg.rawOwnerStatus = null; syncSelectedChoices(); paint();
        try {
            const raw = await withinRequestLifetime((transport) => readStatus({ ...info.ownerBotConnector, accountId: feed.bot.id }, { signal: transport }),
                { signal: activeSignal, lifetimeMs: STATUS_LIFETIME_MS });
            if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
            view.tg.rawOwnerStatus = raw;
            const status = parseOwnerBotStatus(raw, feed.bot.id);
            view.tg.ownerStatus = status.bot && status.bot.username.toLowerCase() !== feed.bot.username.toLowerCase()
                ? { state: 'account_mismatch', reason: 'account_mismatch' } : status;
        } catch {
            if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
            view.tg.ownerStatus = { state: 'unavailable', reason: 'read_failed', bot: null, inviteUrl: '', choices: { state: 'unavailable', rows: [] } };
        }
        syncSelectedChoices();
        paint();
    };

    const refreshTelegramChoice = async ({ reloadPlaces = false } = {}) => {
        const runGen = nextGen();
        const activeSignal = freshSignal();
        view.tg.groupFeed = null;
        view.tg.ownerStatus = null; view.tg.rawOwnerStatus = null;
        view.tg.separateStatus = null; view.tg.rawSeparateStatus = null;
        syncSelectedChoices();
        paint();

        if (reloadPlaces || !Array.isArray(view.tg.skills)) {
            void kickSkillsRead(STATUS_LIFETIME_MS);
        }
        if (reloadPlaces) {
            try {
                const latest = await withinRequestLifetime((transport) => read({ signal: transport }),
                    { signal: activeSignal, lifetimeMs: STATUS_LIFETIME_MS });
                if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
                Object.assign(view, latest);
            } catch (error) {
                if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
                view.status = { text: requestFailureText(error), tone: 'error' };
                paint();
                // `/api/places` could be delayed while the native consent read is already possible.
            }
        }
        await readBranchStatus(runGen, activeSignal);
    };

    const changedBot = () => ({ text: tr('places.tg.account_changed', 'Your paired Telegram bot changed. Check the @name shown below before changing group discovery.'), tone: 'warn' });
    const groupFeedChanged = () => globalThis.window?.dispatchEvent?.(new CustomEvent(PLACES_CHANGED_EVENT,
        { detail: { telegramGroupFeed: true } }));
    const consentConfirmed = (feed, { accountId, enabled, legacy }) => enabled
        ? discoveryOn(feed) && feed.bot.id === accountId
        : feed?.consentAccountId === '' || (legacy && feed?.consentAccountId === undefined
            && feed?.state === 'ready' && feed.bot.id === accountId && !feed.enabled);
    const consentUnchanged = (feed, { accountId, enabled, legacy }) => {
        if (!feed || feed.state === 'unavailable') return false;
        if (enabled) {
            return feed.consentAccountId === ''
                || (legacy && feed.consentAccountId === undefined && feed.state === 'ready' && feed.bot?.id === accountId && !feed.enabled);
        }
        return feed.consentAccountId === accountId
            || (legacy && feed.consentAccountId === undefined && feed.state === 'ready' && feed.bot?.id === accountId && feed.enabled);
    };
    const settleConsent = (feed) => {
        const pending = view.tg.consentPending;
        if (!pending) return;
        const changed = pending.enabled ? feed?.state === 'ready' && feed.bot.id !== pending.accountId
            : Boolean(feed?.consentAccountId && feed.consentAccountId !== pending.accountId);
        if (changed || consentConfirmed(feed, pending)) {
            view.tg.consentPending = null;
            view.status = changed ? changedBot() : {};
            groupFeedChanged();
        }
    };
    const changeConsent = async (enabled) => {
        const info = telegramFacts(view);
        const feed = view.tg.groupFeed;
        if (view.tg.consentPending) return;
        if (enabled && (feed?.state !== 'ready' || feed.enabled || feed.consentAccountId === null || offAccount(feed)
            || !feed.capable || !info.ownerBotConnector || offline(info.ownerBotConnector))) return;
        const accountId = enabled ? feed.bot.id : offAccount(feed);
        if (!accountId) return;
        const pending = { accountId, enabled, legacy: feed.consentAccountId === undefined };
        const runGen = nextGen();
        const activeSignal = freshSignal();
        view.busy = true;
        view.status = { text: tr('places.tg.saving_discovery', 'Saving group discovery…'), tone: 'muted' };
        paint();
        try {
            const reply = await withinRequestLifetime((transport) => api.writeGroupFeed(enabled, accountId, { signal: transport }),
                { signal: activeSignal, lifetimeMs: STATUS_LIFETIME_MS });
            if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
            const observed = parseTelegramGroupFeed(reply);
            if (consentConfirmed(observed, pending)) {
                view.tg.groupFeed = observed;
                view.status = {};
                groupFeedChanged();
                await readBranchStatus(runGen, activeSignal);
                return;
            }
        } catch (error) {
            if (sheet.closed || activeSignal.aborted || getGen() !== runGen) return;
            const status = error?.status;
            const okFalse = error?.body?.ok === false;
            const code = typeof error?.body?.code === 'string' ? error.body.code : '';
            const observed = error?.body ? parseTelegramGroupFeed(error.body) : null;
            if (status === 409 && okFalse && ['account_changed', 'owner_not_bound', 'connector_unavailable'].includes(code)) {
                view.status = code === 'account_changed' ? changedBot() : { text: requestFailureText(error), tone: 'warn' };
                view.busy = false;
                await refreshTelegramChoice({ reloadPlaces: true });
                return;
            }
            if (status === 409 && okFalse && observed && consentUnchanged(observed, pending)) {
                view.tg.groupFeed = observed;
                view.status = { text: requestFailureText(error), tone: 'warn' };
                syncSelectedChoices();
                return;
            }
            if (status === 503 && okFalse && code === 'consent_unknown') {
                view.status = { text: requestFailureText(error), tone: 'warn' };
                if (observed && observed.state !== 'unavailable') {
                    view.tg.groupFeed = observed;
                    syncSelectedChoices();
                } else {
                    view.busy = false;
                    await refreshTelegramChoice();
                }
                return;
            }
            if (status === 503 && okFalse && code === 'unwritable' && observed && observed.state !== 'unavailable') {
                if (consentConfirmed(observed, pending)) {
                    view.tg.groupFeed = observed;
                    view.status = {};
                    view.busy = false;
                    groupFeedChanged();
                    await readBranchStatus(runGen, activeSignal);
                    return;
                }
                if (consentUnchanged(observed, pending)) {
                    view.tg.groupFeed = observed;
                    view.status = { text: requestFailureText(error), tone: 'error' };
                    syncSelectedChoices();
                    return;
                }
            }
            if (status === 400 && okFalse && code === 'invalid') {
                view.status = { text: requestFailureText(error), tone: 'error' };
                return;
            }
        } finally {
            view.busy = false;
            paint();
        }
        // A lost, timed-out, unreadable, or untyped reply may follow a saved change. Only reread the exact native consent fact.
        view.tg.consentPending = pending;
        view.status = {};
        await refreshTelegramChoice();
    };

    return {
        refresh: refreshTelegramChoice,
        async readPassive(_connectors, { isCurrent } = {}) {
            await kickSkillsRead(PASSIVE_LIFETIME_MS);
            if (sheet.closed || signal?.aborted || (isCurrent && !isCurrent())) return;
        },
        cancelInFlight,
        async handleAction(act, control) {
            if (act === 'open-tg-groups') {
                view.tgOpen = true;
                view.status = {};
                if (view.tgConnectPending) { paint(); return true; }
                await refreshTelegramChoice({ reloadPlaces: false });
                return true;
            }
            if (act === 'tg-branch') {
                const nextBranch = control.dataset.branch === 'separate' ? 'separate' : 'personal';
                if (view.tgBranch === nextBranch) return true;
                view.tgBranch = nextBranch;
                view.status = {};
                const runGen = nextGen();
                const activeSignal = freshSignal();
                await readBranchStatus(runGen, activeSignal);
                return true;
            }
            if (act === 'tg-refresh') {
                view.status = {};
                await refreshTelegramChoice({ reloadPlaces: true });
                return true;
            }
            if (act === 'tg-opt-in' || act === 'tg-opt-out') {
                await changeConsent(act === 'tg-opt-in');
                return true;
            }
            if (act === 'tg-browse-bot') {
                onSetup?.();
                sheet.close();
                openSkillsFor({ query: TELEGRAM_BOT_CONNECTOR });
                return true;
            }
            if (act === 'tg-open-skill') {
                const skill = control.dataset.skill || TELEGRAM_BOT_CONNECTOR;
                onSetup?.();
                sheet.close();
                openSkillsFor({ skill });
                return true;
            }
            if (act === 'tg-open-updates') {
                onSetup?.();
                sheet.close();
                openDashboard({ tab: 'updates' });
                return true;
            }
            if (act === 'tg-open-keys') {
                if (!openSettings) return true;
                onSetup?.();
                sheet.close();
                await openSettings('accounts', { row: TELEGRAM_PUBLIC_KEY });
                return true;
            }
            if (act === 'tg-bot-settings') {
                const { tgBotConnector } = telegramFacts(view);
                onSetup?.();
                sheet.close();
                if (tgBotConnector) await openConnectorSettings(tgBotConnector, openSettings);
                return true;
            }
            return false;
        },
    };
}
