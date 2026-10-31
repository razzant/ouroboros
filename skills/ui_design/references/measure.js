// measure.js — what the current screen of a rendered page shows, as data. Evidence for a designer's
// judgement, never a gate: it reports observations and their sizes, no severities, no scores.
// One function expression, read-only, no dependencies. Run it after the page has painted (take a
// screenshot first: WebKit can report a different layout before its first paint).
//   Playwright: page.evaluate(`(${src})(${JSON.stringify({subject: '.card'})})`) -- call it; Node never calls a
//   function given as a string. Ouroboros: browser_action(action="evaluate", value=src) runs it with defaults.
// Options: subject (CSS selector of the element under comparison), minFont (11 px), target (24 px),
// touch (3 px gap), nearMiss (3 px), limit (examples kept per kind; facts.counts has every total).
// Returns {viewport, facts, findings: [{kind, message, where, value}]}. Kinds: page-overflow,
// container-overflow, text-clipped, text-truncated, text-clipped-by-ancestor, text-covered,
// pinned-over-content, controls-touching, controls-overlap, target-small, text-tiny, edges-near-miss,
// contrast-low, contrast-unknown, font-fallback, image-broken.
// Not modelled: clip-path (a clipped element still counts as covering what lies under its box).
((opts) => {
  opts = opts || {};
  const minFont = opts.minFont || 11, target = opts.target || 24, touch = opts.touch ?? 3;
  const nearMiss = opts.nearMiss || 3, limit = opts.limit || 6;
  const counts = {}, findings = [];
  const add = (kind, message, where, value) => {
    counts[kind] = (counts[kind] || 0) + 1;
    if (counts[kind] <= limit) findings.push({ kind, message, where: where || '', value: value ?? null });
  };
  const r1 = v => Math.round(v * 10) / 10;
  const box = b => ({ x: r1(b.left), y: r1(b.top), w: r1(b.width), h: r1(b.height) });
  const cut = (a, b) => { const l = Math.max(a.left, b.left), t = Math.max(a.top, b.top);
    const r = Math.min(a.right, b.right), d = Math.min(a.bottom, b.bottom);
    return { left: l, top: t, right: r, bottom: d, width: Math.max(0, r - l), height: Math.max(0, d - t) }; };
  const path = (el, prefix) => {
    const parts = [];
    for (let e = el; e && e.nodeType === 1 && parts.length < 4; e = e.parentElement) {
      if (e.id) { parts.unshift(e.tagName.toLowerCase() + '#' + e.id); break; }
      parts.unshift(e.tagName.toLowerCase() + [...e.classList].slice(0, 2).map(c => '.' + c).join(''));
    }
    return (prefix || '') + parts.join(' > ');
  };
  const ownText = el => [...el.childNodes].filter(n => n.nodeType === 3).map(n => n.nodeValue).join(' ').replace(/\s+/g, ' ').trim();
  const swatch = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  const color = s => { const m = /^rgba?\(([^)]+)\)$/.exec(String(s));   // other syntaxes (oklch, color()) go through a canvas
    if (m) { const p = m[1].split(/[\s,/]+/).filter(Boolean).map(Number); return { r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1 }; }
    if (!swatch || !s) return null;
    swatch.fillStyle = '#000'; swatch.fillStyle = String(s); swatch.clearRect(0, 0, 1, 1); swatch.fillRect(0, 0, 1, 1);
    const d = swatch.getImageData(0, 0, 1, 1).data; return { r: d[0], g: d[1], b: d[2], a: d[3] / 255 }; };
  const lum = c => { const f = v => (v /= 255) <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b); };
  const clipsX = cs => cs.overflowX !== 'visible', clipsY = cs => cs.overflowY !== 'visible';
  const paints = (el, cs) => { const bg = color(cs.backgroundColor);
    return (bg && bg.a > 0.1) || cs.backgroundImage !== 'none' || /^(IMG|SVG|CANVAS|VIDEO|INPUT|TEXTAREA)$/i.test(el.tagName) || !!ownText(el); };
  const texts = [], controls = [], pinned = [], frames = [];
  const vw = innerWidth, vh = innerHeight, ALL = { left: -1e9, top: -1e9, right: 1e9, bottom: 1e9, width: 2e9, height: 2e9 };

  // Walk the document and its same-origin frames. Every box is cut by its clipping ancestors and by its
  // frame, so what is scrolled away inside a container is not on screen.
  function scan(doc, win, offset, scale, prefix, frameClip, depth) {
    const clips = new Map();
    const toTop = b => { const x = { left: offset.x + b.left * scale, top: offset.y + b.top * scale, width: b.width * scale, height: b.height * scale };
      x.right = x.left + x.width; x.bottom = x.top + x.height; return x; };
    const clipOf = el => {
      const p = el.parentElement;
      if (!p || win.getComputedStyle(el).position === 'fixed') return { rect: frameClip, anc: ALL, by: null };
      if (clips.has(p)) return clips.get(p);
      let c = clipOf(p);
      const pcs = win.getComputedStyle(p);
      if (p !== doc.body && p !== doc.documentElement && (clipsX(pcs) || clipsY(pcs))) {
        const pr = toTop(p.getBoundingClientRect());
        c = { rect: cut(c.rect, pr), anc: cut(c.anc, pr), by: p };
      }
      clips.set(p, c); return c;
    };
    const hidden = new Map();
    const faded = el => { if (!el || el.nodeType !== 1) return false; if (hidden.has(el)) return hidden.get(el);
      const cs = win.getComputedStyle(el), p = el.parentElement;
      const v = +cs.opacity === 0 || (p && p.tagName === 'DETAILS' && !p.open && el.tagName !== 'SUMMARY') || faded(p);
      hidden.set(el, v); return v; };
    for (const el of doc.body ? doc.body.querySelectorAll('*') : []) {
      const cs = win.getComputedStyle(el);
      if (cs.display === 'none' || cs.visibility === 'hidden' || faded(el) || el.closest('[aria-hidden="true"],[inert]')) continue;
      const raw = el.getBoundingClientRect();
      if (raw.width < 1 || raw.height < 1) continue;
      const full = toTop(raw), clip = clipOf(el), vis = cut(full, clip.rect), byAnc = cut(full, clip.anc);
      if (vis.width < 1 || vis.height < 1) continue;
      const item = { el, cs, win, doc, full, vis, byAnc, clipBy: clip.by, prefix, offset, scale, sel: path(el, prefix),
        onScreen: vis.bottom > 0 && vis.top < vh && vis.right > 0 && vis.left < vw };
      if (el.tagName === 'IFRAME') {
        let inner = null; try { inner = el.contentDocument; } catch (e) { inner = null; }
        if (inner && inner.body && depth < 3) {
          const s = el.offsetWidth ? (raw.width / el.offsetWidth) * scale : scale;
          frames.push(el);
          scan(inner, el.contentWindow, { x: full.left + (parseFloat(cs.borderLeftWidth) || 0) * s, y: full.top + (parseFloat(cs.borderTopWidth) || 0) * s }, s, item.sel + ' >> ', vis, depth + 1);
        }
        continue;
      }
      if (el.tagName === 'IMG' && el.complete && el.naturalWidth === 0) add('image-broken', 'image did not load', item.sel, el.getAttribute('src'));
      if (/fixed|sticky/.test(cs.position) && !pinned.some(p => p.el.contains(el))) pinned.push(item);
      if (el.matches('a[href],button,input:not([type=hidden]),select,textarea,summary,[role=button],[role=tab],[tabindex]:not([tabindex="-1"])')) {
        const bg = color(cs.backgroundColor), bd = color(cs.borderTopColor);
        item.boxed = (bg && bg.a > 0.1) || (parseFloat(cs.borderTopWidth) > 0 && bd && bd.a > 0.1) || /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName);
        controls.push(item);
      }
      if (clipsX(cs) || clipsY(cs)) overflowOf(item);
      item.text = ownText(el);
      if (item.text) { item.fs = parseFloat(cs.fontSize) * (el.offsetWidth ? (raw.width / el.offsetWidth) * scale : scale); texts.push(item); }
    }
  }

  function overflowOf({ el, cs, sel, prefix }) {
    if (/^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || ownText(el)) return;
    for (const axis of ['x', 'y']) {
      const over = axis === 'x' ? el.scrollWidth - el.clientWidth : el.scrollHeight - el.clientHeight;
      const mode = (axis === 'x' ? cs.overflowX : cs.overflowY);
      if (over <= 1 || mode === 'visible' || (axis === 'y' && /auto|scroll/.test(mode))) continue;
      const b = el.getBoundingClientRect(), edge = axis === 'x' ? 'right' : 'bottom';
      const kids = [...el.children].filter(k => k.getBoundingClientRect()[edge] > b[edge] + 1)
        .sort((p, q) => q.getBoundingClientRect()[edge] - p.getBoundingClientRect()[edge]);
      add('container-overflow', `content runs ${over}px past the ${axis === 'x' ? 'right' : 'bottom'} edge; it ${/auto|scroll/.test(mode) ? 'scrolls' : 'is cut off'}`,
        sel, { px: over, axis, mode, culprit: kids[0] ? path(kids[0], prefix) : '' });
    }
  }

  scan(document, window, { x: 0, y: 0 }, 1, '', { left: 0, top: 0, right: vw, bottom: Infinity, width: vw, height: Infinity }, 0);
  const pinnedOf = el => pinned.find(p => p.el === el || p.el.contains(el));
  const hitAt = (x, y) => { let hit = document.elementFromPoint(x, y);
    while (hit && hit.tagName === 'IFRAME') { let d = null; try { d = hit.contentDocument; } catch (e) { d = null; }
      if (!d) break; const fb = hit.getBoundingClientRect(), s = hit.offsetWidth ? fb.width / hit.offsetWidth : 1;
      const inner = d.elementFromPoint((x - fb.left) / s, (y - fb.top) / s); if (!inner) break; hit = inner; }
    return hit; };

  const sw = document.documentElement.scrollWidth;
  if (sw > vw + 1) {
    const wide = [...document.body.querySelectorAll('*')].map(e => [e, e.getBoundingClientRect().right]).filter(p => p[1] > vw + 1).sort((p, q) => q[1] - p[1]);
    add('page-overflow', `the page scrolls sideways by ${sw - vw}px at width ${vw}`, wide[0] ? path(wide[0][0]) : 'html', sw - vw);
  }

  // Text: clipped, truncated, cut by an ancestor, covered, tiny, contrast.
  let tiny = 0, unknownContrast = 0; const sizes = new Set();
  for (const t of texts) {
    if (!t.onScreen) continue;
    const { el, cs } = t;
    sizes.add(Math.round(t.fs));
    if (t.fs < minFont) { tiny++; add('text-tiny', `"${t.text.slice(0, 40)}" renders at ${r1(t.fs)}px`, t.sel, r1(t.fs)); }
    const overX = el.scrollWidth - el.clientWidth, overY = el.scrollHeight - el.clientHeight;
    if ((overX > 1 && /hidden|clip/.test(cs.overflowX)) || (overY > 1 && /hidden|clip/.test(cs.overflowY))) {
      const ellipsis = cs.textOverflow === 'ellipsis' || (cs.webkitLineClamp && cs.webkitLineClamp !== 'none');
      add(ellipsis ? 'text-truncated' : 'text-clipped', `"${t.text.slice(0, 40)}" ${ellipsis ? 'ends in an ellipsis' : 'is cut off by its own box'}`, t.sel, Math.max(overX, overY));
    }
    if (t.clipBy) {
      const pcs = t.clipBy.ownerDocument.defaultView.getComputedStyle(t.clipBy);
      const hx = t.full.width - t.byAnc.width, hy = /hidden|clip/.test(pcs.overflowY) ? t.full.height - t.byAnc.height : 0;
      if (hx > 1 || hy > 1) add('text-clipped-by-ancestor', `"${t.text.slice(0, 40)}" is cut ${hx > 1 ? 'sideways' : 'at the bottom or top'} by its container`,
        `${t.sel} | ${path(t.clipBy, t.prefix)}`, { hiddenPx: r1(Math.max(hx, hy)), axis: hx > 1 ? 'x' : 'y' });
    }
    const node = [...el.childNodes].find(n => n.nodeType === 3 && n.nodeValue.trim());
    const line = node && (() => { const r = t.doc.createRange(); r.selectNodeContents(node); return r.getClientRects()[0]; })();
    if (line) for (const f of [0.25, 0.5, 0.75]) {
      const x = t.offset.x + (line.left + line.width * f) * t.scale, y = t.offset.y + (line.top + line.height / 2) * t.scale;
      if (x < Math.max(0, t.vis.left) || x >= Math.min(vw, t.vis.right) || y < Math.max(0, t.vis.top) || y >= Math.min(vh, t.vis.bottom)) continue;
      const hit = hitAt(x, y);
      if (!hit || hit === el || el.contains(hit) || hit.contains(el) || pinnedOf(hit) || !paints(hit, hit.ownerDocument.defaultView.getComputedStyle(hit))) continue;
      add('text-covered', `"${t.text.slice(0, 40)}" is drawn under another element`, `${t.sel} | ${path(hit)}`, null); break;
    }
    const fg = color(cs.color);
    if (!fg || fg.a <= 0.5) continue;
    let base = null, why = '';
    const layers = [];
    for (let e = el; e && !base && !why; e = e.parentElement) {
      const ecs = t.win.getComputedStyle(e);
      if (ecs.backgroundImage !== 'none') why = 'image or gradient';
      else if (ecs.backdropFilter && ecs.backdropFilter !== 'none') why = 'backdrop filter';
      else { const c = color(ecs.backgroundColor); if (c && c.a > 0) { layers.push(c); if (c.a >= 0.99) base = c; } }
    }
    if (why) { unknownContrast++; add('contrast-unknown', `"${t.text.slice(0, 30)}" sits on ${why}`, t.sel, 'unknown'); continue; }
    let bg = base || { r: 255, g: 255, b: 255 };
    for (const c of (base ? layers.slice(0, -1) : layers).reverse()) bg = { r: c.r * c.a + bg.r * (1 - c.a), g: c.g * c.a + bg.g * (1 - c.a), b: c.b * c.a + bg.b * (1 - c.a) };
    const L1 = lum(fg), L2 = lum(bg), ratio = r1((Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05));
    const large = t.fs >= 24 || (t.fs >= 18.6 && +cs.fontWeight >= 600), level = large ? 3 : 4.5;
    if (ratio < level) add('contrast-low', `"${t.text.slice(0, 30)}" has contrast ${ratio}:1 (WCAG AA for this size is ${level}:1)`, t.sel, ratio);
  }

  // Pinned (fixed or sticky) layers drawn over text or controls that are not part of them.
  for (const p of pinned) {
    if (!p.onScreen) continue;
    for (const c of [...texts, ...controls]) {
      if (!c.onScreen || p.el.contains(c.el) || c.el.contains(p.el)) continue;
      const o = cut(c.vis, p.vis);
      if (o.width < 2 || o.height < 2) continue;
      const hit = hitAt(o.left + o.width / 2, o.top + o.height / 2);
      if (hit && (hit === p.el || p.el.contains(hit))) add('pinned-over-content', `a ${p.cs.position} element covers ${r1(o.width)}×${r1(o.height)}px of content`, `${p.sel} | ${c.sel}`, [r1(o.width), r1(o.height)]);
    }
  }

  // Controls: touching neighbours, overlaps, small hit targets.
  const live = controls.filter(c => c.onScreen);
  for (let i = 0; i < live.length; i++) {
    const a = live[i], b0 = a.vis;
    if (b0.width < target && b0.height < target && !(a.el.tagName === 'A' && a.el.closest('p,li'))) add('target-small', `${r1(b0.width)}×${r1(b0.height)}px hit target`, a.sel, [r1(b0.width), r1(b0.height)]);
    for (let j = i + 1; j < live.length; j++) {
      const b = live[j];
      if (!a.boxed || !b.boxed || a.el.contains(b.el) || b.el.contains(a.el) || a.el.className === b.el.className) continue;
      const ox = Math.min(a.vis.right, b.vis.right) - Math.max(a.vis.left, b.vis.left);
      const oy = Math.min(a.vis.bottom, b.vis.bottom) - Math.max(a.vis.top, b.vis.top);
      if (ox > 1 && oy > 1) add('controls-overlap', `two controls overlap by ${r1(ox)}×${r1(oy)}px`, `${a.sel} | ${b.sel}`, [r1(ox), r1(oy)]);
      else if ((oy > 4 && ox <= 0 && -ox < touch) || (ox > 4 && oy <= 0 && -oy < touch)) add('controls-touching', `two controls are ${r1(-Math.max(ox, oy))}px apart`, `${a.sel} | ${b.sel}`, r1(-Math.max(ox, oy)));
    }
  }

  // Alignment inside the subject (or the whole screen): distinct left edges and near misses between relatives.
  const topRect = el => { const b = el.getBoundingClientRect(); let r = { left: b.left, top: b.top, width: b.width, height: b.height };
    for (let w = el.ownerDocument.defaultView; w.frameElement; w = w.parent) {
      const f = w.frameElement, fb = f.getBoundingClientRect(), s = f.offsetWidth ? fb.width / f.offsetWidth : 1;
      if (w.parent.getComputedStyle(f).visibility === 'hidden') return null;
      r = { left: fb.left + r.left * s, top: fb.top + r.top * s, width: r.width * s, height: r.height * s };
    }
    return { ...r, right: r.left + r.width, bottom: r.top + r.height }; };
  let region = null;
  if (opts.subject) for (const doc of [document, ...frames.map(f => f.contentDocument)]) {
    for (const n of doc.querySelectorAll(opts.subject)) { const r = n.getBoundingClientRect().width > 0 && topRect(n); if (r) { region = r; break; } }
    if (region) break;
  }
  const inside = t => t.onScreen && (!region || (t.vis.left >= region.left - 1 && t.vis.right <= region.right + 1 && t.vis.top >= region.top - 1 && t.vis.bottom <= region.bottom + 1));
  const items = [...texts.filter(inside), ...live.filter(inside)];
  const edges = new Map();
  for (const it of items) { const k = Math.round(it.vis.left); edges.set(k, (edges.get(k) || 0) + 1); }
  const related = (a, b) => { let e = a.el; for (let i = 0; i < 4 && e; i++, e = e.parentElement) if (e.contains(b.el)) return true; return false; };
  for (let i = 0; i < items.length; i++) for (let j = i + 1; j < items.length; j++) {
    const a = items[i], b = items[j], d = Math.abs(a.vis.left - b.vis.left);
    if (a.el === b.el || a.el.contains(b.el) || b.el.contains(a.el) || !related(a, b) || d < 0.75 || d > nearMiss) continue;
    if (a.vis.bottom <= b.vis.top + 1 || b.vis.bottom <= a.vis.top + 1) add('edges-near-miss', `left edges ${r1(d)}px apart`, `${a.sel} | ${b.sel}`, [r1(a.vis.left), r1(b.vis.left)]);
  }

  if (document.fonts && document.fonts.check) {
    const generic = /^(serif|sans-serif|monospace|system-ui|ui-[a-z-]+|-apple-system|blinkmacsystemfont|cursive|fantasy|emoji|math)$/i;
    for (const fam of new Set(texts.filter(t => t.onScreen).map(t => t.cs.fontFamily.split(',')[0].trim().replace(/^["']|["']$/g, '')))) {
      let ok = true; try { ok = !fam || generic.test(fam) || document.fonts.check(`16px "${fam}"`); } catch (e) { ok = true; }
      if (!ok) add('font-fallback', `"${fam}" is declared but not loaded; another face is painted`, '', fam);
    }
  }

  return {
    viewport: { w: vw, h: vh },
    facts: {
      page: { scrollWidth: sw, scrollHeight: document.documentElement.scrollHeight, screens: r1(document.documentElement.scrollHeight / vh),
        links: document.querySelectorAll('a[href]').length, controls: live.length },
      textSizes: [...sizes].sort((a, b) => a - b), tinyText: tiny, contrastUnknown: unknownContrast,
      subject: region ? box(region) : null, leftEdges: [...edges.entries()].sort((a, b) => a[0] - b[0]), counts,
    },
    findings,
  };
})
