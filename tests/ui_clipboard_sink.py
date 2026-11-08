"""An inert stand-in for the clipboard in browser tests: nothing here reaches the machine's clipboard.

`INSTALL` replaces the async Clipboard API and `document.execCommand` in the page with recorders on `window.__sink`.
`api` is how the async API answers ('ok' | 'refused' | 'missing': there is no `navigator.clipboard`), `exec` what the
engine answers to a copy. `selections` notes what the focused field, or the page selection, held when the engine was
asked to copy, which is exactly what it would have put on the clipboard. Use it as an init script before navigation
(`page.add_init_script(INSTALL)`) or on a loaded page (`page.evaluate(INSTALL)`), then flip it with `set_sink`.
"""
from __future__ import annotations

INSTALL = """(() => {
    const sink = window.__sink = {api: 'ok', exec: true, attempts: 0, writes: [], selections: []};
    Object.defineProperty(navigator, 'clipboard', {configurable: true, get: () => sink.api === 'missing' ? undefined : {
        writeText: async (text) => {
            sink.attempts += 1;
            if (sink.api === 'refused') throw new DOMException('Synthetic permission denial', 'NotAllowedError');
            sink.writes.push(text);
        },
    }});
    document.execCommand = (command) => {
        const node = document.activeElement;
        const text = node && 'value' in node ? node.value.slice(node.selectionStart, node.selectionEnd) : String(getSelection());
        sink.selections.push({command, text, tag: node?.tagName || '', readOnly: Boolean(node?.readOnly),
            inDialog: Boolean(node?.closest?.('dialog[open]'))});
        return sink.exec;
    };
})()"""


def set_sink(page, api=None, exec=None, reset=False):
    """Change how the recorder answers (None leaves a value as it is); `reset` also forgets what it recorded."""
    page.evaluate("""({api, exec, reset}) => {
        const sink = window.__sink;
        if (api !== null) sink.api = api;
        if (exec !== null) sink.exec = exec;
        if (reset) Object.assign(sink, {attempts: 0, writes: [], selections: []});
    }""", {"api": api, "exec": exec, "reset": reset})


def recorded(page):
    """What the recorder saw: `{api, exec, attempts, writes, selections}`."""
    return page.evaluate("() => window.__sink")
