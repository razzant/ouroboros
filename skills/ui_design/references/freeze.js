// freeze.js — turn the page as it is rendered now into one self-contained HTML document: the real
// product code and data, frozen, so it can become a frame of a one-file comparison page.
// One async function expression, no dependencies; it reads the page and never changes it.
//   Playwright: const {html, missing} = await page.evaluate(`(${src})()`), then write html to a file.
//   Through Ouroboros's browser_action a long frame arrives as a bounded view of its kept text; write it to a file (see browser.md).
// It inlines same-origin stylesheets (url() assets become data: URIs), images, canvases and live form
// values; scroll offsets become data-freeze-scroll="top,left" attributes (the comparison template
// restores them); scripts, inline event handlers, preloads and <base> are removed. Anything it could not
// inline is listed in `missing`; a cross-origin stylesheet or a nested frame stays a gap, never a fetch.
// Sheets are written from their live rules, so CSSOM edits stay. A rule CSSOM cannot write back whole is
// taken from its sheet's own text when it can be, and `missing` names it all the same: the serialized
// rules cannot prove that this text still holds the live value.
(async () => {
  const missing = [], cache = new Map();
  const dataUrl = async (url) => {
    if (!url || url.startsWith('data:')) return url;
    if (!cache.has(url)) cache.set(url, (async () => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(String(res.status));
        const blob = await res.blob();
        return await new Promise((done) => { const fr = new FileReader(); fr.onload = () => done(fr.result); fr.readAsDataURL(blob); });
      } catch (e) { missing.push(url); return null; }
    })());
    return cache.get(url);
  };
  const inlineUrls = async (css, base) => {   // each url() token is replaced once, by its own match span
    const re = /url\(\s*(['"]?)([^'")]+)\1\s*\)/gi, found = new Map();
    for (const [, , ref] of css.matchAll(re)) {
      if (!ref.startsWith('data:') && !ref.startsWith('#') && !found.has(ref)) found.set(ref, await dataUrl(new URL(ref, base).href));
    }
    return css.replace(re, (token, q, ref) => (found.get(ref) ? `url(${q}${found.get(ref)}${q})` : token));
  };
  // A var() shorthand beside one of its own longhands serializes its other longhands empty (`padding-top: ;`) in
  // every engine, so a frame written from such a rule would lose them. Its sheet is copied from its own text instead,
  // comments dropped, when that text, inlined as the frame needs it, parses to the same rules; otherwise (a sheet
  // edited through CSSOM since, or without a text of its own) the rules stay and those longhands are lost. The match
  // cannot see var() tokens: a CSSOM edit to them, or a linked file rebuilt since load that the fetch reads anew
  // (observed in WebKit), looks the same. So `missing` names each such declaration either way, copied or lost.
  // An inline style is exact as it is: a CSSOM edit to it rewrites its attribute, where a lost longhand shows.
  const LOST = '(a var() shorthand CSSOM cannot serialize)';
  const COPIED = "(a var() shorthand CSSOM cannot serialize, copied from its sheet's text: compare it with the live page)";
  const emptied = (style, text) => [...(style || [])].filter((name) => !name.startsWith('--')
    && (` ${text}`.includes(` ${name}: ;`) || ` ${text}`.includes(` ${name}:  !important;`)));
  const emptiedRules = (rules) => rules.flatMap((rule) => {
    const names = emptied(rule.style, rule.style?.cssText);
    return [...(names.length ? [`${names.join(', ')} of ${rule.cssText.split('{')[0].trim()}`] : []), ...emptiedRules([...(rule.cssRules || [])])];
  });
  const ownText = async (sheet, rules, base) => {
    if (document.compatMode !== 'CSS1Compat') return null;   // a frame is standards mode: quirky text would read differently
    try {
      const node = sheet.ownerNode, source = node?.localName === 'style' ? node.textContent
        : node || sheet.ownerRule ? await fetch(sheet.href, { cache: 'force-cache' }).then((res) => (res.ok ? res.text() : '')) : '';
      const text = source.replace(/\/\*[\s\S]*?\*\//g, '');
      if (!text.trim() || /@import/i.test(text)) return null;   // an @import in a frame would fetch
      const own = await inlineUrls(text, base), parsed = new CSSStyleSheet();
      parsed.replaceSync(own);
      const written = [...parsed.cssRules].map((rule) => rule.cssText).join('\n');
      return written === await inlineUrls(rules.map((rule) => rule.cssText).join('\n'), base) ? own : null;
    } catch (e) { return null; }
  };
  const sheetText = async (sheet) => {
    let rules;
    try { rules = [...sheet.cssRules]; } catch (e) { missing.push(sheet.href || 'stylesheet'); return ''; }
    const base = sheet.href || location.href, lost = emptiedRules(rules);
    const own = lost.length ? await ownText(sheet, rules, base) : null;
    missing.push(...lost.map((what) => `${what} in ${sheet.href || 'inline CSS'} ${own ? COPIED : LOST}`));
    if (own) return own;
    const parts = [];
    for (const rule of rules) parts.push(rule.styleSheet ? await sheetText(rule.styleSheet) : rule.cssText);
    return inlineUrls(parts.join('\n'), base);
  };
  const live = [...document.documentElement.querySelectorAll('*')];
  const clone = document.documentElement.cloneNode(true);
  const copies = [...clone.querySelectorAll('*')];
  const twin = new Map(live.map((el, i) => [el, copies[i]]));
  for (const el of live) {
    const c = twin.get(el);
    if (el.scrollTop || el.scrollLeft) c.setAttribute('data-freeze-scroll', `${Math.round(el.scrollTop)},${Math.round(el.scrollLeft)}`);
    if (el.tagName === 'TEXTAREA') c.textContent = el.value;
    else if (el.tagName === 'SELECT') [...el.options].forEach((o, i) => c.options[i].toggleAttribute('selected', o.selected));
    else if (el.tagName === 'INPUT') { if (/^(checkbox|radio)$/.test(el.type)) c.toggleAttribute('checked', el.checked); else c.setAttribute('value', el.value); }
    else if (el.tagName === 'CANVAS') { try { const img = document.createElement('img'); img.src = el.toDataURL(); img.width = el.width; img.height = el.height; img.style.cssText = el.style.cssText; img.className = el.className; c.replaceWith(img); } catch (e) { missing.push('canvas'); } }
    else if (el.tagName === 'IFRAME') { missing.push(el.src || 'iframe'); c.removeAttribute('src'); c.removeAttribute('srcdoc'); }
    if (el.shadowRoot) missing.push('shadow root in ' + el.tagName.toLowerCase());
  }
  for (const el of [document.documentElement, ...live]) {   // a script-written inline style has no other text to copy
    const names = el.hasAttribute('style') ? emptied(el.style, el.getAttribute('style')) : [];
    if (names.length) missing.push(`${names.join(', ')} of the inline style of ${el.localName}${el.id ? `#${el.id}` : ''} ${LOST}`);
  }
  if (document.documentElement.scrollTop || document.documentElement.scrollLeft) {
    clone.setAttribute('data-freeze-scroll', `${Math.round(document.documentElement.scrollTop)},${Math.round(document.documentElement.scrollLeft)}`);
  }
  // outerHTML does not escape style's raw text: an HTML end tag must stay CSS, including in copied source.
  // A CSS slash escape preserves string values and cannot close the HTML style element.
  const htmlSafeStyleText = (css) => css.replace(/<\/style/gi, (end) => '<\\/' + end.slice(2));
  for (const sheet of [...document.styleSheets]) {
    const owner = sheet.ownerNode && twin.get(sheet.ownerNode);
    if (!owner) continue;
    const style = document.createElement('style');
    style.textContent = await sheetText(sheet);
    if (sheet.media && sheet.media.mediaText) style.setAttribute('media', sheet.media.mediaText);
    owner.replaceWith(style);
  }
  for (const sheet of document.adoptedStyleSheets || []) {
    const style = document.createElement('style'); style.textContent = await sheetText(sheet); clone.querySelector('head')?.append(style);
  }
  for (const img of clone.querySelectorAll('img[src]')) {
    const data = await dataUrl(new URL(img.getAttribute('src'), location.href).href);
    if (data) img.setAttribute('src', data);
    img.removeAttribute('srcset'); img.removeAttribute('loading');
  }
  clone.querySelectorAll('picture source, script, noscript, base, link').forEach((n) => n.remove());
  for (const el of [clone, ...clone.querySelectorAll('*')]) {
    for (const a of [...el.attributes]) {
      if (/^on/i.test(a.name)) el.removeAttribute(a.name);
      else if (a.name === 'style' && a.value.includes('url(')) el.setAttribute('style', await inlineUrls(a.value, location.href));
    }
  }
  // Escape once at the serialization boundary, including cloned styles with no sheet and template contents.
  const escapeStyles = (root) => {
    for (const style of root.querySelectorAll('style')) {
      if (style.namespaceURI === 'http://www.w3.org/1999/xhtml') style.textContent = htmlSafeStyleText(style.textContent);
    }
    for (const template of root.querySelectorAll('template')) if (template.content) escapeStyles(template.content);
  };
  escapeStyles(clone);
  const html = '<!doctype html>\n' + clone.outerHTML;
  return { html, missing: [...new Set(missing)] };
})
