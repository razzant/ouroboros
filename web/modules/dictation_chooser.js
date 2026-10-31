// "+" → Add dictation… (docs/DESIGN.md "Voice controls"): the ways to dictate on this install, each saying where the
// audio goes. An engine that is live here is chosen at once; an installed built-in engine that is off (Browser speech)
// is turned on through the Skills toggle, then chosen; a built-in engine not installed yet and an OuroborosHub engine
// (Whisper) are reached through the Skills page — the installed entry, or the Browse search where it installs.
// Choosing saves `voice.stt`; the mic appears once the engine can run here. Nothing here installs code or sends audio.
import { fetchJson } from './api_client.js';
import { fmt, tr } from './i18n.js';
import { showToast } from './toast.js';
import { bindMenu } from './ui_interactions.js';
import { destinationLabel, getSpeech } from './speech_providers.js';

/** OuroborosHub engines offered before they are installed: catalog data, a fixed Skills-page entry until the Hub
 * catalog carries `contributes` (SPEC §3). */
export const HUB_STT_ENGINES = Object.freeze([
    Object.freeze({ slug: 'whisper-voice', label: 'Whisper', destination: 'host', install_size: '~500 MB' }),
]);

/** Ask the Skills page for one installed skill's entry (`{skill}`) or a Browse search (`{query}`); skills.js
 * listens. */
export function openSkills(detail) {
    globalThis.window?.dispatchEvent?.(new CustomEvent('ouro:open-skills', { detail }));
}

/**
 * The chooser's rows: live engines (`live`), built-in engines that are off or not installed (`offer`; `installed:
 * false` from the server when the built-in is not installed) and OuroborosHub engines not live here (`hub`, with
 * `installed` from the Skills listing when it could be read). Pure: `speech` is the hub's reading.
 */
export function chooserOptions({ speech, installed = null, location = globalThis.location }) {
    const chosen = speech.selected('stt');
    const live = speech.list('stt');
    const rows = live.map((row) => ({ kind: 'live', key: row.key, label: row.label, chosen: row.key === chosen.key,
        line: row.available ? row.destinationLabel : row.reason, disabled: !row.available,
        hint: row.key === chosen.key ? tr('voice.chooser.chosen', 'Chosen') : tr('voice.chooser.use', 'Use') }));
    const liveKeys = new Set(live.map((row) => row.key));
    for (const offer of speech.offers('stt')) {
        if (liveKeys.has(`${offer.skill}:${offer.id}`)) continue;
        rows.push({ kind: 'offer', key: `${offer.skill}:${offer.id}`, label: offer.label, offer,
            line: offer.title || destinationLabel(offer.destination, location),
            hint: offer.installed === false ? fmt('Find {label} in Skills', { label: offer.label })
                : tr('voice.chooser.turn_on', 'Turn on and use') });
    }
    const liveSkills = new Set(live.map((row) => row.skill));
    for (const engine of HUB_STT_ENGINES) {
        if (liveSkills.has(engine.slug)) continue;
        const skill = installed?.find((item) => item?.name === engine.slug) || null;
        rows.push({ kind: 'hub', key: `hub:${engine.slug}`, label: engine.label, engine, installed: Boolean(skill),
            line: [destinationLabel(engine.destination, location), engine.install_size, 'OuroborosHub'].filter(Boolean).join(' · '),
            hint: skill ? fmt('Open {label} in Skills', { label: engine.label }) : fmt('Find {label} in Skills', { label: engine.label }) });
    }
    return { rows, chosen };
}

/** Choose one row: live → save; offer → toggle on, then save (not installed: Browse); hub → the Skills page.
 * Resolves to a notice line. */
export async function chooseOption(row, { speech, open = openSkills }) {
    if (row.kind === 'live') {
        await speech.choose('stt', row.key);
        return fmt('Dictation: {label}. Press the mic to dictate.', { label: row.label });
    }
    if (row.kind === 'offer' && row.offer.installed === false) {
        open({ query: row.offer.skill });
        return '';
    }
    if (row.kind === 'offer') {
        await speech.enableOffer(row.offer);
        return fmt('{label} is on. Press the mic to dictate.', { label: row.label });
    }
    open(row.installed ? { skill: row.engine.slug } : { query: row.engine.slug });
    return '';
}

/**
 * Open the chooser under the room's "+" (or its field). `listing()` reads the installed skills (Skills page index);
 * a failed read still offers the Browse search.
 */
export async function openDictationChooser({ composer, speech = getSpeech(),
    listing = () => fetchJson('/api/extensions', { cache: 'no-store' }),
    notice = (text, tone = 'info') => showToast(text, tone) } = {}) {
    const input = composer?.input;
    const doc = input?.ownerDocument || globalThis.document;
    const anchor = input?.closest?.('.chat-text-row')?.querySelector?.('[data-composer-plus]') || input;
    let installed = null;
    try { installed = (await listing())?.skills || null; } catch { installed = null; }
    const { rows, chosen } = chooserOptions({ speech, installed });
    const menu = doc.createElement('div');
    menu.className = 'ui-popup voice-menu voice-chooser';
    menu.setAttribute('role', 'menu');
    menu.setAttribute('aria-label', tr('voice.chooser.label', 'Add dictation'));
    const head = doc.createElement('div');
    head.className = 'voice-menu-head';
    head.textContent = tr('voice.chooser.title', 'Choose how to dictate');
    menu.append(head);
    if (chosen.key && !chosen.available) {
        const why = doc.createElement('div');
        why.className = 'voice-menu-note';
        why.textContent = fmt('Your chosen engine cannot dictate here: {reason}', { reason: chosen.reason });
        menu.append(why);
    }
    let binding = null;
    for (const row of rows) {
        const item = doc.createElement('button');
        item.type = 'button';
        item.className = 'voice-menu-item';
        item.setAttribute('role', 'menuitem');
        item.dataset.voiceOption = row.key;
        if (row.disabled) item.setAttribute('aria-disabled', 'true');
        for (const [className, text] of [['voice-menu-name', row.label], ['voice-menu-line', row.line],
            ['voice-menu-hint', row.hint]]) {
            if (!text) continue;
            const part = doc.createElement('span');
            part.className = className;
            part.textContent = text;
            item.append(part);
        }
        item.addEventListener('click', () => {
            if (row.disabled) return;
            binding?.close({ restoreFocus: true });
            chooseOption(row, { speech }).then((said) => { if (said) notice(said, 'ok'); }, (error) => notice(
                fmt('{label} did not turn on: {reason}', { label: row.label, reason: String(error?.message || error) }),
                'danger'));
        });
        menu.append(item);
    }
    doc.body.append(menu);
    binding = bindMenu(menu, { anchor, align: 'start', onClose: () => menu.remove() });
    return binding;
}
