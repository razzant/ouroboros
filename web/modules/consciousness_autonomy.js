// The consciousness autonomy words (the owner's final names, 21.10): stored observe / act / full. A leaf
// without imports, so the first-run wizard shows the level without loading the main app's presentation
// modules; consciousness_words.js re-exports both names, so a relabel is still one line, here.

/** Consciousness autonomy: stored observe / act / full. */
export const AUTONOMY_LEVELS = Object.freeze([
    { value: 'observe', label: 'Advises', note: 'On its own it studies, keeps its memory and tells you; it changes nothing outside.' },
    { value: 'act', label: 'Acts', note: 'Also acts on its own: starts work, runs commands, speaks and acts in places. It does not change itself.' },
    { value: 'full', label: 'Evolves', note: 'Also changes itself: its own code, its settings, connecting new places.' },
]);

export const autonomyOf = (value) => AUTONOMY_LEVELS.find((level) => level.value === value) || null;
