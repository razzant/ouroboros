// Observe the production room on animation frames. All times in this log share performance.now().
(() => {
    const {prefix, room, anchorKey, scenario} = window.traceConfig;
    const log = [];
    const field = () => document.getElementById(prefix + '-input');
    const feed = () => document.getElementById(prefix + '-messages');
    const row = () => feed()?.querySelector('[data-history-id="' + anchorKey + '"]');
    const round = x => Number.isFinite(x) ? Math.round(x * 4) / 4 : null;
    const time = () => round(performance.now());
    const mark = (event, data = {}) => {
        if (event === 'activation-touch') {
            if (beforeTouch) log.push(beforeTouch);
            recordingCycle = true;
        }
        log.push({t: time(), event, ...data});
    };
    const box = el => {
        if (!el?.getClientRects().length) return null;
        const r = el.getBoundingClientRect();
        return [round(r.top), round(r.bottom)];
    };
    let revision = 0, signature, firstAnchor, glyph, frameNumber = 0, timer;
    let beforeTouch, stopped = false, recordingCycle = false;
    // The same first printed character distinguishes a text line's movement from changes to the bubble around it.
    function firstLine(anchor) {
        const message = anchor?.querySelector('.message');
        if (!message) return null;
        if (!glyph || glyph.startContainer.nodeType !== Node.TEXT_NODE || !message.contains(glyph.startContainer)) {
            const nodes = document.createTreeWalker(message, NodeFilter.SHOW_TEXT);
            let node;
            while ((node = nodes.nextNode())) {
                const at = node.data.search(/\S/);
                if (at < 0) continue;
                glyph = document.createRange();
                glyph.setStart(node, at);
                glyph.setEnd(node, at + 1);
                return box(glyph);
            }
            return null;
        }
        return box(glyph);
    }
    function sample() {
        const f = feed(), input = field(), area = document.getElementById(prefix + '-input-area');
        const bar = document.querySelector(prefix === 'chat' ? '#page-chat .chat-page-header' : '#project-panel .project-panel-bar');
        const rows = Array.from(f?.querySelectorAll('[data-history-id]') || []);
        const ids = rows.map(n => n.dataset.historyId);
        // Changes to the identity, order or words must be visible even if the number of rows stays constant.
        const next = JSON.stringify(rows.map((n, i) => [ids[i], n.querySelector('.message')?.textContent]));
        if (signature !== next) {
            mark('transcript', {revision: ++revision, ids});
            signature = next;
        }
        const anchor = rows.find(n => n.dataset.historyId === anchorKey);
        firstAnchor ||= anchor;
        const vv = window.visualViewport;
        return {
            keyboard: document.body?.classList.contains('keyboard-open') ?? null,
            vvH: round(vv?.height), offsetTop: round(vv?.offsetTop), offsetLeft: round(vv?.offsetLeft),
            scale: round(vv?.scale), innerH: innerHeight, scrollY: round(scrollY),
            bar: box(bar), feed: box(f), field: box(input), composer: box(area),
            anchor: box(anchor), line: firstLine(anchor), sameNode: anchor ? firstAnchor === anchor : null,
            focused: !!input && document.activeElement === input, quoted: !!area?.querySelector('.chat-quote--composer'),
            top: round(f?.scrollTop), extent: f?.scrollHeight ?? null, height: f?.clientHeight ?? null,
            gap: f ? round(f.scrollHeight - f.clientHeight - f.scrollTop) : null,
            hydrated: f?.dataset.historyHydrated === 'true', revision
        };
    }
    function tick() {
        if (stopped) return;
        const point = {n: ++frameNumber, t: time(), frame: sample()};
        // Keep the last idle callback and every callback through dismissal, even with identical geometry.
        if (recordingCycle) log.push(point);
        else beforeTouch = point;
        timer = requestAnimationFrame(tick);
    }
    let waiting = [], hold = scenario === 'project-held';
    if (hold) {
        const fetchPage = window.fetch.bind(window);
        window.fetch = async (request, init) => {
            const url = new URL(request.url || request, location.href);
            const response = await fetchPage(request, init);
            if (hold && response.ok && url.pathname === '/api/chat/history' && url.searchParams.get('chat_id') === String(room)) {
                mark('history-held');
                await new Promise(resolve => waiting.push(resolve));
            }
            return response;
        };
    }
    window.motion = {
        field, feed, row, mark, sample, get pending() { return waiting.length; },
        uikit(sequence, name) {
            mark('uikit', {sequence, name, geometry: sample()});
            if (name === 'UIKeyboardDidShowNotification' && hold && waiting.length) {
                mark('release', {pending: waiting.length});
                hold = false;
                waiting.splice(0).forEach(resolve => resolve());
            }
        },
        stop() {
            if (!stopped) {
                cancelAnimationFrame(timer);
                mark('stop', {geometry: sample()});
                stopped = true;
            }
            return {log, anchorKey, scenario};
        }
    };
    mark('start');
    timer = requestAnimationFrame(tick);
})();
