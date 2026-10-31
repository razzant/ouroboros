"""A finite ESM graph used unchanged by the loader and real browser consumer checks."""
from __future__ import annotations

import base64
from pathlib import Path

NAME = "esm_read_files"
PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")
BINARY = bytes(range(256))
RENDER = {"kind": "module", "entry_type": "module", "entry": "app/main.js", "start": "auto",
          "assets": ["app/lib.js", "app/cycle.js", "app/late.mjs", "style.css", "nested/palette.css",
                     "images/pixel.png", "data.bin", "data.json"]}


def install(root: Path, *, reviewed=True, enabled=True) -> Path:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_enabled, save_review_state

    skill = root / "skills" / "external" / NAME
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"""---
name: {NAME}
description: Browser file consumer fixture.
version: 0.1.0
type: extension
entry: plugin.py
permissions: [widget, route]
---
Finite browser files.
""", encoding="utf-8", newline="\n")
    (skill / "plugin.py").write_text(f"""hits = 0
async def ping(request):
    return {{'ok': True}}
async def disposed(request):
    global hits
    hits += 1
    return {{'hits': hits}}
async def count(request):
    return {{'hits': hits}}
def register(api):
    api.register_route('ping', ping)
    api.register_route('disposed', disposed, methods=('POST',))
    api.register_route('count', count)
    api.register_ui_tab('main', 'Module files', render={RENDER!r})
    api.register_ui_tab('other', 'Separate files', render={{'kind':'module', 'entry_type':'module', 'entry':'other.js', 'start':'auto'}})
    api.register_ui_tab('classic', 'Classic entry', render={{'kind':'module', 'entry':'classic.mjs', 'start':'auto'}})
""", encoding="utf-8", newline="\n")
    files = {
        "app/main.js": """import { value, cycle } from './lib.js';
const root = document.getElementById('root');
root.innerHTML = '<h2>Module files</h2><p id="summary">Loading</p><img id="pixel" width="24" height="24"><button id="late">Load detail</button>';
const style = document.createElement('link'); style.rel = 'stylesheet'; style.href = '../style.css';
// Styles resolve from the document base, whereas module-relative addresses use import.meta.url.
style.href = new URL('../style.css', import.meta.url); document.head.append(style);
document.querySelector('#pixel').src = new URL('../images/pixel.png', import.meta.url);
const data = new Uint8Array(await (await fetch(new URL('../data.bin', import.meta.url))).arrayBuffer());
const json = await (await fetch('./data.json')).json();
const api = await (await fetch('/api/extensions/esm_read_files/ping')).json();
window.proof = {value, cycle: cycle(), size: data.length, last: data[255], json: json.value, api: api.ok,
    url: import.meta.url, base: document.baseURI, origin: window.origin, top: false, cookie: false};
try { window.proof.top = !!parent.document; } catch {}
try { window.proof.cookie = !!document.cookie; } catch {}
root.dataset.ready = '1'; document.querySelector('#summary').textContent = `${value} · ${data.length} bytes`;
document.querySelector('#late').onclick = async () => {
    const part = './late.mjs'; root.dataset.late = (await import(part)).late;
};
window.__ouroWidgetOnDispose(async () => { await fetch('/api/extensions/esm_read_files/disposed', {method:'POST'}); });
""",
        "app/lib.js": "import { read } from './cycle.js'; export const value = 'Module ready'; export const cycle = () => read();",
        "app/cycle.js": "import { value } from './lib.js'; export const read = () => value;",
        "app/late.mjs": "export const late = 'loaded';",
        "style.css": "@import './nested/palette.css'; body{font:16px system-ui;padding:16px} #summary{background:url('./images/pixel.png') no-repeat right center;padding-right:24px}",
        "nested/palette.css": "body{background:#17191d;color:#f2f3f5} h2{font-size:20px} button{padding:8px 12px}",
        "data.json": '{"value":"declared"}',
        "other.js": "document.getElementById('root').textContent='Separate module'; window.proof={url:import.meta.url,origin:window.origin};",
        "classic.mjs": "window.classicThis = this === window; var classicGlobal = 'kept'; document.getElementById('root').textContent='Classic ready';",
        "unlisted.txt": "not granted to the widget",
    }
    for name, source in files.items():
        file = skill / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(source, encoding="utf-8", newline="\n")  # browser text is UTF-8 on every host
    (skill / "images").mkdir()
    (skill / "images/pixel.png").write_bytes(PNG)
    (skill / "data.bin").write_bytes(BINARY)
    save_enabled(root, NAME, enabled)
    if reviewed:
        save_review_state(root, NAME, SkillReviewState(status="pass", content_hash=compute_content_hash(skill)))
    return skill
