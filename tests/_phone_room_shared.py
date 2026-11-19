"""Phone-room fixtures for the production SPA under the static-server/API fixture (tests/test_subscription_setup_browser.py):
saved-history pages, a visual viewport the test moves (and pinches), a Telegram SDK stand-in whose fullscreen state and
safe areas the test drives, and screenshots kept in ``OUROBOROS_UI_EVIDENCE_DIR``. Emulated touch, viewport and SDK
signals exercise the page's own owners; they are not a native iPhone keyboard, IME or Telegram client."""
from __future__ import annotations

import json
import os
from pathlib import Path


def history(texts, *, after: str = 'after:recent', start: int = 0) -> dict:
    """One room's newest page of saved assistant rows, complete and continuable by ``after``."""
    return {'messages': [{'role': 'assistant', 'text': text, 'ts': f'2026-11-03T17:{(start + i) // 60:02d}:{(start + i) % 60:02d}Z',
                          'history_id': f'chat:{start + i}',
                          'history_position': {'source': 'chat:fixture', 'offset': start + i}}
                         for i, text in enumerate(texts)],
            'page_cursor': 'page:recent', 'next_cursor': None, 'after_cursor': after, 'has_more': False,
            'window': {'complete': True, 'truncated_by': []},
            'coverage': {'v': 1, 'view': 'phone', 'upper': {'chat': 100},
                         'spans': {'chat': {'chain': 'fixture', 'from': 0, 'to': 100, 'gaps': []}, 'progress': None}}}


def shot(page, name: str, clip: dict | None = None) -> Path | None:
    """A screenshot kept as evidence when the run names a directory; the test still asserts its own facts."""
    root = os.environ.get('OUROBOROS_UI_EVIDENCE_DIR')
    if not root:
        return None
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f'{name}.png'
    page.screenshot(path=str(path), **({'clip': clip} if clip else {}))
    return path


def facts(name: str, value: dict) -> None:
    """Measured facts beside the screenshots, and in the test log."""
    root = os.environ.get('OUROBOROS_UI_EVIDENCE_DIR')
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f'{name}.json').write_text(json.dumps(value, indent=1, sort_keys=True), encoding='utf-8')
    print(name, json.dumps(value, sort_keys=True))


# The visual viewport the page reads (app.js updateVvh, layout.js pinchMagnified): `__viewport(h, emit)` moves its
# height, with or without the resize event an engine may never send; `__pinch()` magnifies it.
VIEWPORT = """(() => {
    let height = innerHeight, pinched = false;
    const viewport = new EventTarget();
    Object.defineProperties(viewport, {height: {get: () => height}, scale: {get: () => pinched ? 2 : 1},
        width: {get: () => document.documentElement.clientWidth / (pinched ? 2 : 1)}});
    Object.defineProperty(window, 'visualViewport', {configurable: true, value: viewport});
    window.__viewport = (h, emit = true) => { height = h; if (emit) viewport.dispatchEvent(new Event('resize')); };
    window.__pinch = () => { pinched = true; viewport.dispatchEvent(new Event('resize')); };
})()"""


def telegram_sdk(*, version: str = '9.1', fullscreen: str = 'grant', insets: dict | None = None,
                 content: dict | None = None, late_ms: int | None = None, init_data: str = 'phone-fixture') -> str:
    """A Telegram WebApp stand-in installed before the page's scripts (``late_ms``: that much later, as an async tag
    loads). ``fullscreen``: ``grant`` enters it, ``refuse`` reports UNSUPPORTED, ``absent`` has no method; the test
    reads ``__tg.calls`` and drives ``__tg.exit()``, ``__tg.insets(safe, content)``."""
    config = json.dumps({'version': version, 'fullscreen': fullscreen, 'late': late_ms, 'initData': init_data,
                         'safe': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0, **(insets or {})},
                         'content': {'top': 0, 'bottom': 0, 'left': 0, 'right': 0, **(content or {})}})
    return """((config) => {
    const listeners = new Map(), calls = [];
    const button = () => ({isVisible: false, show() { this.isVisible = true; }, hide() { this.isVisible = false; },
        onClick(fn) { this.callback = fn; }, offClick() { this.callback = null; }});
    const fire = (name, detail) => setTimeout(() => listeners.get(name)?.(detail), 30);
    const app = {version: config.version, platform: 'ios', initData: config.initData, colorScheme: 'light',
        themeParams: {}, isExpanded: true, isFullscreen: false,
        safeAreaInset: {...config.safe}, contentSafeAreaInset: {...config.content},
        BackButton: button(), SettingsButton: button(), HapticFeedback: {impactOccurred() {}}, openLink() {},
        isVersionAtLeast(v) {
            const a = config.version.split('.').map(Number), b = String(v).split('.').map(Number);
            for (let i = 0; i < Math.max(a.length, b.length); i += 1) if ((a[i] || 0) !== (b[i] || 0)) return (a[i] || 0) > (b[i] || 0);
            return true;
        },
        onEvent(name, fn) { listeners.set(name, fn); }, offEvent(name) { listeners.delete(name); }};
    for (const name of ['ready', 'expand', 'disableVerticalSwipes', 'setHeaderColor', 'setBackgroundColor',
        'setBottomBarColor', 'hideKeyboard']) app[name] = (...args) => { calls.push(name); };
    if (config.fullscreen !== 'absent') app.requestFullscreen = () => {
        calls.push('requestFullscreen');
        if (config.fullscreen === 'refuse') return fire('fullscreenFailed', {error: 'UNSUPPORTED'});
        // Telegram answers later, with its own state and the edges fullscreen exposes.
        setTimeout(() => { app.isFullscreen = true; listeners.get('fullscreenChanged')?.(); }, 30);
    };
    window.__tg = {calls, listeners, app,
        exit() { app.isFullscreen = false; listeners.get('fullscreenChanged')?.(); },
        insets(safe, content) { Object.assign(app.safeAreaInset, safe || {}); Object.assign(app.contentSafeAreaInset, content || {});
            listeners.get('safeAreaChanged')?.(); listeners.get('contentSafeAreaChanged')?.(); }};
    // When the SDK appeared and when the page first said its fullscreen state: the order is the test's fact.
    new MutationObserver(() => { if (window.__tg.boundAt == null && document.documentElement?.dataset.telegramFullscreen)
        window.__tg.boundAt = performance.now(); }).observe(document, {subtree: true, attributes: true,
        attributeFilter: ['data-telegram-fullscreen']});
    const install = () => { window.Telegram = {WebApp: app}; window.__tg.installedAt = performance.now(); };
    if (config.late == null) install(); else setTimeout(install, config.late);
})(%s)""" % config
