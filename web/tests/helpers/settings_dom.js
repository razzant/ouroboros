// A DOM and a server for the node tests of Settings: the dashboard fixture's document with an element that parses
// markup as a browser nests it, the home's passive reads answered, and the reads that start something left out.
import { forgetSettingRows } from '../../modules/settings_controls.js';
import { createLayerStack } from '../../modules/layers.js';
import { bindPageBack } from '../../modules/page_header.js';
import { NodeStub, setup } from '../dashboard_dom_fixture.js';

// The fixture's element, reading a self-closed tag (an SVG `<path … />` of a glyph) as childless, so a row drawn
// after a glyph is that glyph's sibling here as it is in a browser.
export class Stub extends NodeStub {
    // No layout in this fixture; a geometry test supplies its own boxes.
    getBoundingClientRect() { return { x: 0, y: 0, top: 0, right: 0, bottom: 0, left: 0, width: 0, height: 0 }; }
    toggleAttribute(name, force = !this.hasAttribute(name)) {
        if (force) this.setAttribute(name, ''); else this.removeAttribute(name);
        return force;
    }
    get innerHTML() { return super.innerHTML; }
    set innerHTML(html) {
        this.textContent = '';
        const stack = [this];
        for (const token of String(html).match(/<[^>]+>|[^<]+/g) || []) {
            if (token.startsWith('</')) { stack.pop(); continue; }
            if (!token.startsWith('<')) { stack.at(-1)._text += token; continue; }
            const tag = token.match(/^<([\w-]+)/)?.[1];
            if (!tag) continue;
            const node = new Stub(tag);
            for (const attr of token.slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) node.setAttribute(attr[1], attr[2] ?? '');
            stack.at(-1).appendChild(node);
            if (!['input', 'br', 'hr', 'img'].includes(tag) && !token.endsWith('/>')) stack.push(node);
        }
    }
}

export const ANSWERS = {
    '/api/settings': { OUROBOROS_MODEL: 'openrouter::z-ai/glm-5.3', OPENROUTER_API_KEY: 'sk-...', OUROBOROS_SAFETY_OWNER_RULES: '', _meta: {} },
    '/api/mcp/status': { enabled: true, servers: [] },
    '/api/extensions': { skills: [{ name: 'one' }], live: { settings_sections: [] } },
    '/api/places': { places: [], connectors: [] },
    '/api/update/status': { managed: true, check_ok: true, current_version: '7.10.2' },
    '/api/storage': { count: null },
    '/api/desktop/autostart': { state: 'unavailable' },
    // The engine's roster and windows as last observed, the passive view Costs reads too: it starts and refreshes
    // nothing. Every other read of the status (the full one, with models) is a start below.
    '/api/claudexor/status?view=quota': { view: 'quota', profiles: {}, quota: [], reads: { catalog: 'not_read', accounts: 'not_read', quota: 'not_read' } },
};
export const STARTS = ['/api/model-catalog', '/api/review-pool', '/api/claudexor/wake', '/api/claudexor/status', '/api/mcp/refresh', '/api/providers/test',
    '/api/update/check', '/api/storage/measure'];
export const settle = async () => { for (let round = 0; round < 12; round += 1) await new Promise((resolve) => setImmediate(resolve)); };

export function world(t, { history } = {}) {
    forgetSettingRows();
    // A Settings shell mounted in this world is put away after its test (its `pagehide`, with the globals it was mounted
    // with, which a second world of the same test may have replaced): a shell left mounted keeps the module-wide
    // lifecycle subscription and reads in the next test.
    let env = null;
    t.after(() => {
        const kept = { window: globalThis.window, document: globalThis.document };
        Object.assign(globalThis, { window: env.window, document: env.document });
        try { env.window.dispatchEvent({ type: 'pagehide', persisted: false }); } finally { Object.assign(globalThis, kept); }
    });
    env = setup(t);
    const posts = [];
    for (const [url, body] of Object.entries(ANSWERS)) {
        env.routes.set(url, async (_url, init = {}) => {
            if (init.method && init.method !== 'GET') posts.push(url);
            return { ok: true, status: 200, json: async () => body };
        });
    }
    env.document.createElement = (tag) => new Stub(tag);
    const content = new Stub('div');
    content.id = 'content';
    env.document.body.appendChild(content);
    const state = { activePage: 'chat' };
    const shown = [];
    const layers = createLayerStack({ doc: env.document, win: env.window, history });
    const off = bindPageBack({ layers, doc: env.document, win: env.window,
        currentHeader: () => content.querySelector(`#page-${state.activePage}`)?.querySelector('.app-page-header') });
    t.after(() => { off(); layers.destroy(); });
    const showPage = async (name, options = {}) => {
        shown.push([name, options]);
        layers.batch(() => {
            if (state.activePage !== name) {
                state.activePage = name;
                env.window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: name } }));
            }
            options.enter?.();
        });
        return true;
    };
    return { env, posts, content, state, shown, showPage, layers };
}

