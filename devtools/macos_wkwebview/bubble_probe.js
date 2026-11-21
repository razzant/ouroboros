    const rows = [...document.querySelectorAll(a.sel)].filter(row => row.dataset.clientMessageId === a.id);
    if (a.address) {
        const row = rows[0];
        if (rows.length !== 1 || row.dataset.rowAddress !== a.address) return {count: rows.length, visible_accepted: false};
        const reply = (visibility, shown) => ({count: 1, visible_accepted: shown ? true : null, visibility});
        if (getComputedStyle(row).visibility !== 'visible') return reply('hidden', false);
        let opacity = 1;
        for (let node = row; node; node = node.parentElement) {
            const style = getComputedStyle(node);
            if (style.display === 'none') return reply('hidden', false);
            opacity *= Number(style.opacity);
        }
        if (!(opacity >= 0.95)) return reply('transparent', false);
        const box = row.getBoundingClientRect(), feed = row.closest(a.feed)?.getBoundingClientRect();
        if (!row.getClientRects().length || !box.width || !box.height || !feed) return reply('hidden', false);
        const left = Math.max(box.left, feed.left, 0), right = Math.min(box.right, feed.right, innerWidth);
        const top = Math.max(box.top, feed.top, 0), bottom = Math.min(box.bottom, feed.bottom, innerHeight);
        if (!(left < right && top < bottom)) return reply('clipped', false);
        const points = [[.5, .5], [.25, .25], [.75, .25], [.25, .75], [.75, .75]];
        const exposed = points.some(([x, y]) => {
            const hit = document.elementFromPoint(left + (right - left) * x, top + (bottom - top) * y);
            return !!hit && row.contains(hit);
        });
        return reply(exposed ? 'exposed' : 'covered', exposed);
    }
    const buttons = rows.length === 1 ? rows[0].querySelectorAll('[data-unconfirmed-action=retry]') : [];
    return {count: rows.length, retry: buttons.length === 1 && !buttons[0].disabled &&
        !!buttons[0].getClientRects().length};
