// Page half of the system WKWebView composer-clearance check (composer_clearance.py). The host injects this at
// document start in the page's own world. It keeps the app's WebSocket so a growing answer can be delivered the
// way the browser tests deliver one, and gives the host measurements and user-like actions. It never edits the
// product: the negative layouts are a separate stylesheet the host adds and removes.
(() => {
    'use strict';
    if (window.__composerCheck) return;
    const NativeSocket = window.WebSocket;
    const sockets = [];
    window.WebSocket = class extends NativeSocket {
        constructor(...args) {
            super(...args);
            sockets.push(this);
        }
    };

    const frame = () => new Promise((resolve) => requestAnimationFrame(() => resolve()));
    // Checked once per frame of the page itself, never for longer than `ms`.
    async function until(probe, label, ms) {
        const deadline = performance.now() + ms;
        for (;;) {
            const value = probe();
            if (value) return value;
            if (performance.now() > deadline) throw new Error(`timed out after ${ms} ms waiting for ${label}`);
            await frame();
        }
    }
    const shown = (el) => Boolean(el && el.getClientRects().length) && el.getBoundingClientRect().height > 0;
    // The match on screen: a closed room's instance can linger, hidden, before the reopened one.
    const find = (selector) => [...document.querySelectorAll(selector)].find(shown) || document.querySelector(selector);
    const feedOf = (selector) => {
        const feed = find(selector);
        if (!feed) throw new Error(`no transcript ${selector}`);
        return feed;
    };
    const fieldOf = (selector) => {
        const room = feedOf(selector).parentElement;
        const field = room.querySelector('#chat-input-area textarea, .chat-input-area textarea')
            || room.querySelector('textarea');
        if (!field) throw new Error(`no composer field beside ${selector}`);
        return field;
    };

    // Where the newest row ends and the composer begins, in viewport pixels, beside the reserve's own CSS.
    function geometry(feed) {
        const composer = feed.parentElement.querySelector('#chat-input-area, .chat-input-area');
        const box = feed.getBoundingClientRect();
        const bottoms = [...feed.children]
            .filter((el) => shown(el) && !/^(absolute|fixed)$/.test(getComputedStyle(el).position))
            .map((el) => el.getBoundingClientRect().bottom);
        const style = getComputedStyle(feed);
        const end = getComputedStyle(feed, '::after');
        return {
            scrollTop: feed.scrollTop, scrollHeight: feed.scrollHeight, clientHeight: feed.clientHeight,
            feedTop: box.top, feedBottom: box.bottom, rows: bottoms.length,
            lastBottom: bottoms.length ? Math.max(...bottoms) : null,
            composerTop: shown(composer) ? composer.getBoundingClientRect().top : null,
            composerHeight: shown(composer) ? composer.getBoundingClientRect().height : null,
            reserve: style.getPropertyValue('--chat-input-reserve').trim(),
            paddingBottom: style.paddingBottom,
            spacer: { content: end.content, flexBasis: end.flexBasis, flexShrink: end.flexShrink },
        };
    }

    const check = {
        // Up: a socket is open and Main shows its saved history.
        async ready({ text }) {
            await until(() => sockets.some((socket) => socket.readyState === 1), 'an open WebSocket', 60000);
            await check.showing({ feed: '#chat-messages', text });
            if (document.fonts?.ready) await document.fonts.ready;
            return check.facts();
        },
        async showing({ feed, text }) {
            await until(() => [...document.querySelectorAll(`${feed} .chat-bubble`)]
                .some((el) => shown(el) && el.textContent.includes(text)), `${feed} showing "${text}"`, 30000);
            return true;
        },
        async gone({ selector }) {
            await until(() => ![...document.querySelectorAll(selector)].some(shown), `${selector} to close`, 15000);
            return true;
        },
        async viewport({ width, height }) {
            const layout = width >= window.ouroLayout.dockedMinWidth ? 'docked' : 'stack';
            await until(() => innerWidth === width && innerHeight === height
                && document.documentElement.dataset.layout === layout, `a ${width}x${height} ${layout} viewport`, 15000);
            return { innerWidth, innerHeight, layout };
        },
        // Fonts loaded, then the transcript, its composer and its reserve unchanged for `still` frames in a row.
        async settle({ feed, least = 14, still = 8, most = 300 }) {
            if (document.fonts?.ready) await document.fonts.ready;
            let last = '';
            let same = 0;
            for (let count = 1; count <= most; count += 1) {
                await frame();
                const node = find(feed);
                const g = node && geometry(node);
                const key = g ? JSON.stringify([g.scrollTop, g.scrollHeight, g.clientHeight, g.composerHeight,
                    g.reserve, node.hasAttribute('data-resizing'), innerWidth, innerHeight]) : 'absent';
                same = key === last ? same + 1 : 0;
                last = key;
                if (count >= least && same >= still) return { frames: count, settled: true };
            }
            return { frames: most, settled: false };
        },
        position({ feed }) {
            return { scrollTop: feedOf(feed).scrollTop };
        },
        // A negative collapses the scroll range and the engine clamps the position; this hands it back.
        restore({ feed, scrollTop }) {
            const node = feedOf(feed);
            node.scrollTop = scrollTop;
            return { scrollTop: node.scrollTop };
        },
        // As the app left it, then at the end of the scroll range this engine gives the transcript.
        measure({ feed }) {
            const node = feedOf(feed);
            const rest = geometry(node);
            node.scrollTop = node.scrollHeight;
            return { rest, end: geometry(node), layout: document.documentElement.dataset.layout || null,
                innerWidth, innerHeight };
        },
        // Typing as a user does: focus, then the field's own input event.
        fill({ feed, text }) {
            const field = fieldOf(feed);
            field.focus();
            field.value = text;
            field.dispatchEvent(new Event('input', { bubbles: true }));
            return field.value.length;
        },
        draft({ feed }) {
            return fieldOf(feed).value;
        },
        click({ selector }) {
            const control = document.querySelector(selector);
            if (!control) throw new Error(`no control ${selector}`);
            control.click();
            return true;
        },
        // A frame the server would push, delivered on the app's own open socket.
        emit({ message }) {
            const socket = sockets.filter((item) => item.readyState === 1).at(-1);
            if (!socket) throw new Error('no open WebSocket');
            socket.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(message) }));
            return true;
        },
        // One deliberately broken layout over the product's, or none.
        variant({ css }) {
            document.getElementById('composer-check-variant')?.remove();
            if (!css) return false;
            const style = document.createElement('style');
            style.id = 'composer-check-variant';
            style.textContent = css;
            document.head.append(style);
            return true;
        },
        // Main behind another page keeps its reserve, and the first frame back clears the composer. `before` is where
        // Main was left: that frame is evidence only when it was its newest message. The click back is its own task,
        // as a user's click arrives between rendering updates; made inside one, it would be laid out by that update
        // before the room's own landing. `presented` is Main as the first rendering update that shows it lays it out:
        // its ResizeObserver step runs after that update's animation frame callbacks and layout, just before paint.
        // `presentedFrame` counts this script's callbacks by then (0: an update laid Main out before any). `first` is
        // the start of the first frame, before the app's own callbacks in it: the system WKWebView keeps no scroll
        // position for a hidden size container, the transcript included, and the room lands on its newest message
        // within that update.
        async leaveAndReturn({ away, page }) {
            const feed = feedOf('#chat-messages');
            const before = geometry(feed);
            check.click({ selector: away });
            await until(() => shown(document.querySelector(page)) && !shown(feed), `${page} over Main`, 15000);
            await check.settle({ feed: '#chat-messages' });
            const hidden = getComputedStyle(feed).getPropertyValue('--chat-input-reserve').trim();
            let frames = 0, presented = null;
            const observer = new ResizeObserver(() => {
                if (!presented && shown(feed)) presented = { presentedFrame: frames, ...geometry(feed) };
            });
            observer.observe(feed);
            await frame(); await frame();  // its first report, of the hidden Main, is behind us
            const first = await new Promise((resolve) => setTimeout(() => {
                check.click({ selector: '[data-nav-page="chat"]' });
                requestAnimationFrame(() => { frames = 1; resolve(geometry(feed)); });
            }, 0));
            for (let count = 0; !presented && count < 60; count += 1) {
                await frame();
                frames += 1;
            }
            observer.disconnect();
            return { before, shown: before.reserve, hidden, first, presented };
        },
        // Whether this engine scrolls a container's end padding: the defect the transcript's spacer replaced.
        probe() {
            const layouts = {
                block: 'display:block',
                flex_column: 'display:flex;flex-direction:column',
                flex_column_container: 'display:flex;flex-direction:column;container-type:inline-size',
            };
            const out = {};
            for (const [name, layout] of Object.entries(layouts)) {
                const box = document.createElement('div');
                box.style.cssText = 'position:absolute;left:-10000px;top:0;width:200px;height:200px;'
                    + `box-sizing:border-box;overflow:auto;padding:0 0 100px;${layout}`;
                const row = document.createElement('div');
                row.style.cssText = 'height:150px;flex:0 0 150px';
                box.append(row);
                document.body.append(box);
                out[name] = { scrollHeight: box.scrollHeight, clientHeight: box.clientHeight,
                    endPaddingScrolls: box.scrollHeight >= 249 };
                box.remove();
            }
            return out;
        },
        facts() {
            return { url: location.href, userAgent: navigator.userAgent, platform: navigator.platform,
                dpr: devicePixelRatio, innerWidth, innerHeight, visibility: document.visibilityState,
                layout: document.documentElement.dataset.layout || null, sockets: sockets.length,
                fieldSizing: CSS.supports('field-sizing', 'content'),
                composerSizing: document.querySelector('#chat-input')?.dataset.sizing || null };
        },
    };
    window.__composerCheck = check;
})();
