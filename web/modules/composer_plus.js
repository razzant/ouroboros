// The composer's "+" menu filled from skills' contributions (docs/DESIGN.md "Composer owner controls", "Skills in the
// page"): after the host's Attach file, one section per skill under its label with its `composer.plus` items in the
// owner's arrangement, then the host's own items: "Add dictation…" while no speech engine can dictate here, once the
// voice controls have offered their chooser, and "Make a button with Ouroboros…", which starts a request in the draft for
// the owner to finish and send. Every item does its contribution's verb on this room's draft; none sends a message.
import { tr } from './i18n.js';
import { draftPort, getContributions, runContributionAction } from './contributions.js';

const dictation = { offer: null, listeners: new Set() };
const dictationChanged = () => { for (const listener of [...dictation.listeners]) listener(); };
// A room's chips that have no room in its capsule row, not even as "+N" (composer_chips.js): listed in "+" under their
// skill's name. One entry per composer: `{items, sync}`, whichever of the two binds first creates it.
const chipOverflow = new WeakMap();
const overflowOf = (composer) => {
    if (!chipOverflow.has(composer)) chipOverflow.set(composer, { items: [], sync: null });
    return chipOverflow.get(composer);
};

/** The chips of `composer`'s room that have no room in the row: `[{key, skill, skill_label, label, title, onSelect}]`. */
export function setChipOverflow(composer, items) {
    if (!composer) return;
    const entry = overflowOf(composer);
    entry.items = Array.isArray(items) ? items : [];
    entry.sync?.();
}

/**
 * The voice controls' chooser behind "Add dictation…": `open({composer, draft})`, or `{open, shown()}` when the voice
 * controls decide themselves when the item shows (by default: while no speech-to-text engine is chosen). The item is
 * absent until one is set. Returns the unset.
 */
export function setDictationOffer(offer) {
    const next = typeof offer === 'function' ? { open: offer } : (typeof offer?.open === 'function' ? offer : null);
    dictation.offer = next;
    dictationChanged();
    return () => {
        if (dictation.offer !== next) return;
        dictation.offer = null;
        dictationChanged();
    };
}

/** What "Make a button with Ouroboros…" starts in the draft (the bundled templates skill teaches the rest). */
export const MAKE_A_BUTTON_TEXT = 'Make me a button in the + menu that ';

/**
 * Keep one room's `composer.plusMenu` in step with the contributions and the dictation offer. `notice(text, tone)`
 * says how a verb went (a toast when omitted). Returns the disposer.
 */
export function bindComposerPlus({ composer, store = getContributions(), notice = undefined, run = runContributionAction }) {
    const draft = draftPort(() => ({ composer }));
    const verb = (item) => () => { void run(item, { draft, notice }); };

    function sections() {
        const bySkill = new Map();
        for (const item of store.byPlacement('composer.plus')) {
            if (!bySkill.has(item.skill)) {
                bySkill.set(item.skill, { key: `skill:${item.skill}`, label: item.skill_label || item.skill, items: [] });
            }
            bySkill.get(item.skill).items.push({ key: item.key, label: item.label, title: item.title || '',
                icon: item.icon || '', onSelect: verb(item) });
        }
        for (const chip of overflowOf(composer).items) {
            if (!bySkill.has(chip.skill)) {
                bySkill.set(chip.skill, { key: `skill:${chip.skill}`, label: chip.skill_label || chip.skill, items: [] });
            }
            bySkill.get(chip.skill).items.push({ key: `chip:${chip.key}`, label: chip.label, title: chip.title || '',
                icon: '', onSelect: chip.onSelect });
        }
        const offer = dictation.offer;
        const host = [];
        if (offer && (offer.shown ? offer.shown() : !store.voice('stt')?.key)) {
            host.push({ key: 'dictation', label: tr('composer.plus.dictation', 'Add dictation…'), icon: 'mic',
                onSelect: () => offer.open({ composer, draft }) });
        }
        host.push({ key: 'make-button', label: tr('composer.plus.make_button', 'Make a button with Ouroboros…'),
            title: tr('composer.plus.make_button_title', 'Ask Ouroboros for a button of your own in this menu'),
            icon: 'sparkle',
            onSelect: () => { draft.insert(tr('composer.plus.make_button_text', MAKE_A_BUTTON_TEXT)); } });
        // The host's own items close the menu under Ouroboros's name, apart from the skills' sections above them.
        return [...bySkill.values(), { key: 'host', label: tr('composer.plus.host', 'Ouroboros'), items: host }];
    }

    const sync = () => composer.plusMenu.setSections(sections());
    const off = store.onChange(sync);
    dictation.listeners.add(sync);
    const overflow = overflowOf(composer);
    overflow.sync = sync;
    sync();
    return () => {
        off();
        dictation.listeners.delete(sync);
        if (overflow.sync === sync) overflow.sync = null;
    };
}
