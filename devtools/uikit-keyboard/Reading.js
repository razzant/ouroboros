// Establish each entry condition before XCTest's finger, and check it again at the actual touch.
(() => {
    const m = window.motion;
    if (!m || window.Telegram?.WebApp?.initData !== 'phone-fixture' || window.__viewport || !window.visualViewport) return;
    const {scenario, projectId, room} = window.traceConfig;
    const project = room !== 1, held = scenario === 'project-held';
    const post = (phase, aim) => window.webkit.messageHandlers.reading.postMessage({phase, aim: aim || []});
    const target = () => project ? m.field() : m.row()?.querySelector('.chat-message-reply');
    let previous, stillSince = performance.now(), lastPhase, opened = false, touched = false, focusSeen = false;
    function historyLoaded(feed) {
        if (feed?.dataset.historyHydrated !== 'true' || feed.querySelectorAll('[data-history-id]').length !== 10 || !m.row()) return false;
        const first = feed.querySelector('[data-history-id="chat:' + (project ? 20 : 0) + '"] .message');
        return !!first?.textContent.includes('Synthetic room ' + room + ', entry 0.');
    }
    function position() {
        const f = m.feed(), control = target(), field = m.field();
        if (!f || !control || !field || control.closest('[hidden], [inert]') ||
            document.documentElement.dataset.telegramFullscreen !== 'active') return null;
        const vv = window.visualViewport, r = control.getBoundingClientRect();
        const centerX = r.x + r.width / 2, y = r.y + r.height / 2;
        const center = document.elementFromPoint(centerX, y);
        const centerHit = control.contains(center);
        // On a coarse pointer the time beside Main's Reply can cover its centre.
        const x = centerHit ? centerX : [0.75, 0.25].map(part => r.left + part * r.width)
            .find(candidate => control.contains(document.elementFromPoint(candidate, y)));
        let replyCenter;
        if (!project) {
            const coveredBy = center?.closest('.msg-time') ? 'msg-time' : 'other';
            replyCenter = centerHit ? 'reply' : coveredBy;
        }
        const signature = [f.scrollTop, vv.height, vv.offsetTop, x, y].join('/');
        if (signature !== previous) { stillSince = performance.now(); previous = signature; }
        const loaded = historyLoaded(f);
        const empty = f.dataset.historyHydrated !== 'true' && m.pending > 0 && !f.querySelector('[data-history-id]');
        const closed = document.activeElement !== field && !document.body.classList.contains('keyboard-open');
        const hit = r.width > 0 && r.height > 0 && x !== undefined;
        return {hit, replyCenter, closed, loaded: !!loaded, empty, gap: f.scrollHeight - f.clientHeight - f.scrollTop,
            stillMs: performance.now() - stillSince, aim: hit ? [(x - vv.offsetLeft) * vv.scale, (y - vv.offsetTop) * vv.scale] : []};
    }
    function phase(p) {
        if (!p?.hit || !p.closed || p.stillMs < 180) return null;
        if (held) return p.empty ? 'project-held' : null;
        if (!p.loaded || (!project && !m.row()?.classList.contains('is-folded'))) return null;
        if (Math.abs(p.gap) <= 6) return project ? 'project-ready' : 'latest';
        return !project && p.gap > 48 ? 'reading' : null;
    }
    const needed = project ? scenario : scenario === 'main-reading' ? 'reading' : 'latest';
    const timer = setInterval(() => {
        if (project && !opened && document.getElementById('chat-messages')?.dataset.historyHydrated === 'true' &&
            document.documentElement.dataset.telegramFullscreen === 'active') {
            opened = true;  // Same Project reference entry as the room tests. Only the navigation setup is scripted.
            window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: {id: projectId}, pointer: true}}));
        }
        if (!touched) {
            const p = position();
            const next = phase(p);
            if (next && next !== lastPhase) {
                m.mark('position', {phase: next, gap: p.gap, replyCenter: p.replyCenter});
                post(next, p.aim);
            }
            lastPhase = next;
        } else if (focusSeen && document.activeElement === m.field() && historyLoaded(m.feed()) &&
                   (project || !!document.querySelector('#chat-input-area .chat-quote--composer'))) {
            m.mark('focused-history', {geometry: m.sample()});
            post('focused-history');
            clearInterval(timer);
        }
    }, 60);
    window.addEventListener('touchstart', e => {
        if (!e.isTrusted) return;
        if (touched && m.feed()?.contains(e.target)) { m.mark('dismiss-touch'); return; }
        if (touched || !target()?.contains(e.target)) return;
        const t = e.changedTouches[0];
        const p = position();
        const accepted = phase(p) === needed && target().contains(document.elementFromPoint(t.clientX, t.clientY));
        m.mark('activation-touch', {accepted, hydrated: p?.loaded === true, pending: m.pending, geometry: m.sample(),
            gap: p?.gap ?? null, replyCenter: p?.replyCenter, x: t.clientX, y: t.clientY});
        touched = true;
        post(accepted ? 'entry-touch' : 'wrong-entry');
    }, {capture: true, passive: true});
    window.addEventListener('input', e => {
        if (e.isTrusted && touched && focusSeen && e.target === m.field() && e.target.value.length > 0) {
            m.mark('field-input');
            post('field-input');
        }
    }, {capture: true, passive: true});
    for (const name of ['focusin', 'focusout']) {
        window.addEventListener(name, e => {
            if (e.target !== m.field()) return;
            m.mark(name);
            if (name === 'focusin' && touched) { focusSeen = true; post('field-focus'); }
        }, {capture: true, passive: true});
    }
})();
