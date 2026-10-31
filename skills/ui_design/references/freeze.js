// freeze.js — turn the page as it is rendered now into one self-contained HTML document: the real
// product code and data, frozen, so it can become a frame of a one-file comparison page.
// One async function expression, no dependencies; it reads the page and never changes it.
//   Playwright: const {html, missing} = await page.evaluate(`(${src})()`), then write html to a file.
//   Not through Ouroboros's browser_action: it cuts results at 20,000 characters (see browser.md).
// It inlines same-origin stylesheets (url() assets become data: URIs), images, canvases and live form
// values; scroll offsets become data-freeze-scroll="top,left" attributes (the comparison template
// restores them); scripts, inline event handlers, preloads and <base> are removed. Anything it could not
// inline is listed in `missing`; a cross-origin stylesheet or a nested frame stays a gap, never a fetch.
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
    const re = /url\(\s*(['"]?)([^'")]+)\1\s*\)/g, found = new Map();
    for (const [, , ref] of css.matchAll(re)) {
      if (!ref.startsWith('data:') && !ref.startsWith('#') && !found.has(ref)) found.set(ref, await dataUrl(new URL(ref, base).href));
    }
    return css.replace(re, (token, q, ref) => (found.get(ref) ? `url(${q}${found.get(ref)}${q})` : token));
  };
  const sheetText = async (sheet) => {
    let rules;
    try { rules = [...sheet.cssRules]; } catch (e) { missing.push(sheet.href || 'stylesheet'); return ''; }
    const parts = [];
    for (const rule of rules) parts.push(rule.styleSheet ? await sheetText(rule.styleSheet) : rule.cssText);
    return inlineUrls(parts.join('\n'), sheet.href || location.href);
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
  if (document.documentElement.scrollTop || document.documentElement.scrollLeft) {
    clone.setAttribute('data-freeze-scroll', `${Math.round(document.documentElement.scrollTop)},${Math.round(document.documentElement.scrollLeft)}`);
  }
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
  const html = '<!doctype html>\n' + clone.outerHTML;
  return { html, missing: [...new Set(missing)] };
})
