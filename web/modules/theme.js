/** Light/dark theme: the one writer of the root `data-theme` attribute.
    The durable source is the `theme` UI preference (ouroboros/gateway/
    ui_preferences.py); localStorage only mirrors it so the inline head
    script in index.html / onboarding_template.html can paint the right
    theme before the preference fetch returns. */
import { apiClient } from './api_client.js';

export const THEME_STORAGE_KEY = 'ouro.theme';
export const THEME_CHANGED_EVENT = 'ouro:theme-changed';

/** Anything but the literal 'light' is the dark default. */
export function normalizeTheme(value) {
    return value === 'light' ? 'light' : 'dark';
}

export function currentTheme() {
    return document.documentElement.dataset.theme === 'light' ? 'light' : 'dark';
}

export function applyTheme(value) {
    const theme = normalizeTheme(value);
    const root = document.documentElement;
    if (theme === 'light') root.dataset.theme = 'light';
    else delete root.dataset.theme;
    try { localStorage.setItem(THEME_STORAGE_KEY, theme); } catch { /* storage blocked: server value still wins on boot */ }
    document.dispatchEvent(new CustomEvent(THEME_CHANGED_EVENT, { detail: { theme } }));
    return theme;
}

/** Settings -> Behavior segmented control. Independent of the settings Save
    button: the click applies + persists immediately and never touches the
    settings draft (no data-effort-* hooks, no input/change events). */
export function bindThemeSegments(root) {
    const buttons = Array.from(root.querySelectorAll('[data-theme-value]'));
    if (!buttons.length) return;
    const sync = () => {
        const active = currentTheme();
        buttons.forEach((button) => {
            const on = button.dataset.themeValue === active;
            button.classList.toggle('active', on);
            button.setAttribute('aria-pressed', String(on));
        });
    };
    buttons.forEach((button) => {
        button.addEventListener('click', () => {
            const theme = applyTheme(button.dataset.themeValue);
            apiClient.saveUiPreferences({ theme }).catch((err) => {
                // A silent failure leaves the localStorage mirror and the durable
                // preference disagreeing, and the launcher then paints the other
                // theme's window chrome on the next start.
                console.warn('Failed to save theme', err);
            });
        });
    });
    document.addEventListener(THEME_CHANGED_EVENT, sync);
    sync();
}
