/* What Ouroboros thinks of your LLMs: renders the rating and the editions list from the edition data.
   No dependencies, no network requests outside this folder. Data: data/latest.js (embedded, also works from a local file),
   data/latest.json and data/editions/<id>/edition.json (fetched over http). Strings: assets/strings.js. */
(function () {
  'use strict';
  var STR = window.OUROBOROS_STRINGS || {};
  var EMBED = window.OUROBOROS_RATING || null;
  var EDS = window.OUROBOROS_EDITIONS || null;
  var doc = document, root = doc.documentElement, body = doc.body;
  var app = doc.getElementById('app');
  if (!app) return;
  var PAGE = body.getAttribute('data-page') || 'rating';
  var SITE = body.getAttribute('data-site') || '';
  var REPO = body.getAttribute('data-repo') || '';
  var SOURCE = body.getAttribute('data-source') || '';
  var CANON = body.getAttribute('data-canonical') || '';
  var ICON = body.getAttribute('data-icon') || 'assets/icon.png';
  var LANG_KEY = 'ouroboros-rating-lang', THEME_KEY = 'ouroboros-rating-theme';
  // the table opens on my ranking: the models in my order and the letters of the main columns
  var state = { lang: 'en', E: EMBED, group: 'rating', open: null, notice: null };
  var M = {}, W = {}, G = {}, CELL = {}, CASE = {}, RATED = [], MAIN = [];
  var live = null;

  // ---------- small helpers
  function query(name) {
    var q = window.location.search.replace(/^\?/, '').split('&');
    for (var i = 0; i < q.length; i++) {
      var kv = q[i].split('=');
      try {
        if (decodeURIComponent(kv[0]) === name) return decodeURIComponent(kv.slice(1).join('=') || '');
      } catch (e) { /* a malformed pair from some other tool: skip it */ }
    }
    return null;
  }
  function stored(key) { try { return window.localStorage.getItem(key); } catch (e) { return null; } }
  function store(key, value) { try { window.localStorage.setItem(key, value); } catch (e) { /* private mode */ } }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; });
  }
  function ru() { return state.lang === 'ru'; }
  function L(o) { return o ? (o[state.lang] || o.en || '') : ''; }
  function num(n) {
    var s = String(Math.round(n)), out = '', sep = ru() ? ' ' : ',';
    for (var i = 0; i < s.length; i++) { if (i && (s.length - i) % 3 === 0 && s.charAt(i - 1) !== '-') out += sep; out += s.charAt(i); }
    return out;
  }
  function dec(x, digits) { var s = Number(x).toFixed(digits); return ru() ? s.replace('.', ',') : s; }
  function share(p) {
    if (p > 0 && p < 0.1) return '<' + dec(0.1, 1) + '%';
    return (p >= 10 || p === Math.round(p) ? String(Math.round(p)) : dec(p, 1)) + '%';
  }
  function plural(n, forms) {
    n = Math.abs(Math.round(n));
    if (!ru()) return n === 1 ? forms[0] : forms[forms.length - 1];
    var k = n % 10, h = n % 100;
    if (k === 1 && h !== 11) return forms[0];
    if (k >= 2 && k <= 4 && !(h >= 12 && h <= 14)) return forms[1];
    return forms[forms.length - 1];
  }
  function fmt(s, v) {
    var out = '', i = 0, n = s.length, j, key, parts, x;
    while (i < n) {
      var ch = s.charAt(i);
      if (ch === '{') {
        j = s.indexOf('}', i);
        if (j > i) { key = s.slice(i + 1, j); x = v[key]; out += x == null ? '' : (typeof x === 'number' ? num(x) : esc(x)); i = j + 1; continue; }
      }
      if (ch === '[') {
        j = s.indexOf(']', i);
        if (j > i) { parts = s.slice(i + 1, j).split('|'); if (parts.length > 2 && v[parts[0]] != null) { out += plural(v[parts[0]], parts.slice(1)); i = j + 1; continue; } }
      }
      out += ch; i++;
    }
    return out;
  }
  function raw(key) { var e = STR[key]; return e ? (e[state.lang] || e.en || '') : key; }
  function t(key, vars) { return vars ? fmt(raw(key), vars) : raw(key); }
  function forms(key, n) { return plural(n, raw(key).split('|')); }
  function compact(n) {
    var steps = [[1e12, 'unit_trillion'], [1e9, 'unit_billion'], [1e6, 'unit_million'], [1e3, 'unit_thousand']];
    for (var i = 0; i < steps.length; i++) {
      if (n >= steps[i][0]) {
        var v = n / steps[i][0], digits = v >= 20 ? 0 : v >= 10 ? 1 : 2, b = dec(v, digits);
        while (digits && b.charAt(b.length - 1) === '0') { b = b.slice(0, -1); digits--; }
        if (!digits && (b.charAt(b.length - 1) === '.' || b.charAt(b.length - 1) === ',')) b = b.slice(0, -1);
        return [b, raw(steps[i][1])];
      }
    }
    return [num(n), ''];
  }
  function compactText(n) { var c = compact(n); return c[0] + (c[1] ? (ru() ? ' ' : '') + c[1] : ''); }
  function ymd(iso) { var p = String(iso).split('-'); return [Number(p[0]), Number(p[1]), Number(p[2])]; }
  function months(key) { return raw(key).split('|'); }
  function dateLong(iso) { var d = ymd(iso), m = months('months_long')[d[1] - 1]; return ru() ? d[2] + ' ' + m + ' ' + d[0] : m + ' ' + d[2] + ', ' + d[0]; }
  function dateDay(iso) { var d = ymd(iso), m = months('months_long')[d[1] - 1]; return ru() ? d[2] + ' ' + m : m + ' ' + d[2]; }
  function dateRange(a, b, year) {
    var x = ymd(a), y = ymd(b), mo, s;
    if (ru()) {
      mo = months('months_long');
      s = (x[0] === y[0] && x[1] === y[1]) ? x[2] + '–' + y[2] + ' ' + mo[x[1] - 1] : x[2] + ' ' + mo[x[1] - 1] + ' – ' + y[2] + ' ' + mo[y[1] - 1];
      return s + (year ? ' ' + y[0] : '');
    }
    mo = months('months_short');
    s = (x[0] === y[0] && x[1] === y[1]) ? mo[x[1] - 1] + ' ' + x[2] + '–' + y[2] : mo[x[1] - 1] + ' ' + x[2] + ' – ' + mo[y[1] - 1] + ' ' + y[2];
    return s + (year ? ', ' + y[0] : '');
  }
  function nb(s) { return String(s).replace(/ /g, ' '); }
  // A case carries its own date, so the first and the last day of the cases are read from the cases themselves.
  function caseSpan(E) {
    var a = '', b = '';
    (E.cases || []).forEach(function (c) { var last = c.to || c.from; if (!a || c.from < a) a = c.from; if (last > b) b = last; });
    return a ? [a, b] : null;
  }
  // The day an edition speaks for: the last day of its logs or of its cases, whichever is later. The ranking is an
  // opinion, so the head carries this one date; the two exact ranges are told in "How to read".
  function asOf(E) { var span = caseSpan(E); return span && span[1] > E.period.to ? span[1] : E.period.to; }
  function caseWhen(c) {
    var a = ymd(c.from), b = c.to ? ymd(c.to) : null, mo;
    if (ru()) {
      if (!b) return a[2] + '.' + pad(a[1]);
      return a[1] === b[1] ? a[2] + '–' + b[2] + '.' + pad(a[1]) : a[2] + '.' + pad(a[1]) + '–' + b[2] + '.' + pad(b[1]);
    }
    mo = months('months_short');
    if (!b) return mo[a[1] - 1] + ' ' + a[2];
    return a[1] === b[1] ? mo[a[1] - 1] + ' ' + a[2] + '–' + b[2] : mo[a[1] - 1] + ' ' + a[2] + ' – ' + mo[b[1] - 1] + ' ' + b[2];
  }
  function pad(n) { return n < 10 ? '0' + n : String(n); }
  function lower(s) { return String(s).toLowerCase(); }
  var WIDE = '(min-width: 1240px)';
  function wide() { return window.matchMedia(WIDE).matches; }

  var ICON_SUN = '<svg class="sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  var ICON_MOON = '<svg class="moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>';
  var ICON_X = '<svg viewBox="0 0 14 14" aria-hidden="true"><path d="M2 2l10 10M12 2L2 12"/></svg>';

  // ---------- language and theme
  function pickLang() {
    var q = query('lang');
    if (q === 'ru' || q === 'en') return q;
    var s = stored(LANG_KEY);
    return (s === 'ru' || s === 'en') ? s : 'en';
  }
  function setLang(l) {
    state.lang = l;
    store(LANG_KEY, l);
    var parts = [], ed = query('edition');
    if (ed) parts.push('edition=' + encodeURIComponent(ed));
    if (l !== 'en') parts.push('lang=' + l);
    try { window.history.replaceState(null, '', window.location.pathname + (parts.length ? '?' + parts.join('&') : '') + window.location.hash); } catch (e) { /* local file */ }
    var wrap = app.querySelector('.mxwrap'), left = wrap ? wrap.scrollLeft : 0;
    render();
    wrap = app.querySelector('.mxwrap');
    if (wrap) wrap.scrollLeft = left;
    var b = app.querySelector('[data-lang="' + l + '"]');
    if (b) b.focus();
  }
  function theme() { return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light'; }
  function applyTheme(th) {
    root.setAttribute('data-theme', th);
    var meta = doc.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', th === 'dark' ? '#0d0b0f' : '#ffffff');
  }
  function withLang(href) { return state.lang === 'en' ? href : href + (href.indexOf('?') < 0 ? '?' : '&') + 'lang=' + state.lang; }

  // ---------- shared chrome
  function bar() {
    var here = PAGE === 'editions';
    return '<header class="shell bar">' +
      '<a class="brand" href="' + esc(SITE) + '" aria-label="' + esc(t('nav_home')) + '"><img src="' + esc(ICON) + '" alt="" width="34" height="34"><span>Ouroboros</span></a>' +
      '<nav aria-label="Ouroboros">' +
      '<a href="' + esc(withLang('index.html')) + '"' + (here ? '' : ' aria-current="page"') + '>' + esc(t('nav_rating')) + '</a>' +
      '<a href="' + esc(withLang('editions.html')) + '"' + (here ? ' aria-current="page"' : '') + '>' + esc(t('nav_editions')) + '</a>' +
      '<a class="far" href="' + esc(SITE) + 'benchmarks/">' + esc(t('nav_benchmarks')) + '</a>' +
      '<a class="far" href="' + esc(SITE) + 'paper/">' + esc(t('nav_paper')) + '</a>' +
      '<a class="far" href="' + esc(SITE) + 'install/">' + esc(t('nav_install')) + '</a>' +
      '<a class="far" href="' + esc(REPO) + '">' + esc(t('nav_github')) + '</a>' +
      '</nav>' +
      '<div class="tools"><div class="seg" role="group" aria-label="' + esc(t('lang_label')) + '">' +
      '<button type="button" data-lang="en" lang="en" aria-pressed="' + (state.lang === 'en') + '">EN</button>' +
      '<button type="button" data-lang="ru" lang="ru" aria-pressed="' + (state.lang === 'ru') + '">RU</button></div>' +
      '<button class="icon-btn" type="button" data-theme-toggle aria-label="' + esc(t('theme_label')) + '" title="' + esc(t('theme_label')) + '">' + ICON_SUN + ICON_MOON + '</button></div>' +
      '</header>';
  }
  function foot(E) {
    var built = E ? '<p>' + t('foot_built', { id: E.edition, date: dateLong(E.built) }) + '</p>' : '';
    return '<footer class="foot"><div>' + built +
      '<p><a href="' + esc(REPO) + '">' + esc(t('foot_by')) + '</a></p></div>' +
      '<nav aria-label="' + esc(t('title')) + '">' +
      '<a href="' + esc(withLang(PAGE === 'editions' ? 'index.html' : 'editions.html')) + '">' + esc(PAGE === 'editions' ? t('back_to_rating') : t('all_editions')) + '</a>' +
      '<a href="' + esc(SOURCE) + '/METHOD.md">' + esc(t('foot_method')) + '</a>' +
      '<a href="' + esc(SOURCE) + '/CHANGELOG.md">' + esc(t('foot_changelog')) + '</a>' +
      '<a href="' + esc(SOURCE) + '/SCHEMA.md">' + esc(t('foot_schema')) + '</a>' +
      '<a href="data/latest.json">' + esc(t('foot_data')) + '</a>' +
      '<a href="' + esc(SITE) + '">' + esc(t('foot_site')) + '</a>' +
      '</nav></footer>';
  }
  function setHead(title) {
    root.setAttribute('lang', state.lang);
    doc.title = title;
    var d = doc.querySelector('meta[name="description"]');
    if (d) d.setAttribute('content', t('description'));
    var og = doc.querySelector('meta[property="og:image"]'), tw = doc.querySelector('meta[name="twitter:image"]');
    var img = CANON + 'card/og-' + state.lang + '.png';
    if (og) og.setAttribute('content', img);
    if (tw) tw.setAttribute('content', img);
  }

  // ---------- rating page
  function index(E) {
    M = {}; W = {}; G = {}; CELL = {}; CASE = {}; RATED = []; MAIN = [];
    E.models.forEach(function (m) { M[m.id] = m; if (m.status === 'rated') RATED.push(m); });
    E.works.forEach(function (w) { W[w.id] = w; });
    E.groups.forEach(function (g) { G[g.id] = g; if (g.main) MAIN.push(g); });
    E.cells.forEach(function (c) { CELL[c.model + '|' + c.work] = c; });
    E.cases.forEach(function (c) { CASE[c.id] = c; });
  }
  // a short head: the title, one sentence that says whose opinion this is, and the edition. The table comes next
  function hero(E) {
    var draft = E.status !== 'final';
    return '<section class="head"><div class="head-main">' +
      '<p class="eyebrow">' + t('eyebrow', { id: E.edition }) + '</p>' +
      '<h1>' + esc(t('title')) + '</h1><p class="lead">' + esc(t('lead')) + '</p></div>' +
      '<div class="head-side"><div class="status' + (draft ? ' draft' : '') + '">' + t(draft ? 'status_draft' : 'status_final', { id: E.edition }) + '</div>' +
      '<div class="sub">' + nb(t('as_of', { date: dateLong(asOf(E)) })) + '</div>' +
      (draft ? '<div class="sub">' + esc(t('status_sub_draft')) + '</div>' : '') +
      '<div class="sub"><a href="' + esc(withLang('editions.html')) + '">' + esc(t('all_editions')) + '</a></div></div></section>' +
      (state.notice ? '<p class="notice" role="status">' + esc(t(state.notice.key, state.notice.args)) + '</p>' : '');
  }
  function tile(tone, big, unit, label, sub) {
    return '<div class="stat ' + tone + '"><b>' + esc(big) + (unit ? '<small>' + esc(unit) + '</small>' : '') + '</b>' +
      '<span class="label">' + esc(label) + '</span><span class="sub">' + sub + '</span></div>';
  }
  function tokenText(tok, n) {
    return n === tok.read && tok.read_display ? L(tok.read_display) : compactText(n) + (n === tok.read && tok.lower_bound ? '+' : '');
  }
  // the tiles sit under the table and count what stands behind it; nothing in them is a rate of success
  function tiles(E) {
    var T = E.totals, tok = T.tokens || {}, out = [], c, sub;
    out.push(tile('t-blue', String(T.models.rated), '', forms('tile_models', T.models.rated), t('tile_models_sub', { apart: T.models.earlier + T.models.trial + T.models.untried })));
    out.push(tile('t-green', String(T.work_types), '', forms('tile_works', T.work_types), t('tile_works_sub', { columns: T.columns })));
    out.push(tile('t-violet', num(T.cases), '', forms('tile_cases', T.cases), esc(t('tile_cases_sub'))));
    if (tok.status === 'final' && tok.read != null) {
      c = compact(tok.read);
      sub = [t('tile_tokens_since', { since: dateLong(tok.since) })];
      if (tok.cache_share != null) sub.push(t('tile_tokens_cache', { share: Math.round(100 * tok.cache_share) + '%' }));
      if (tok.written != null) sub.push(t('tile_tokens_written', { written: compactText(tok.written) }));
      // a side fact about how much I worked, not about any model: the tile is quiet, without a tone
      out.push(tile('quiet', c[0], c[1] + (tok.lower_bound ? '+' : ''), t('tile_tokens'), sub.join('; ')));
    } else {
      out.push(tile('quiet pending', '—', '', t('tile_tokens'), esc(t('tile_tokens_pending'))));
    }
    return '<section class="stats" aria-label="' + esc(t('title')) + '">' + out.join('') + '</section>';
  }
  // a coloured letter is my pick; a dash stands in the few cells that have no letter on purpose
  function chip(g, cls) { return '<span class="t t-' + (g || 'N') + (cls ? ' ' + cls : '') + '">' + (g || '–') + '</span>'; }
  function chipWords(g) { return g ? g + ', ' + t('tier_' + g) : t('unrated_short'); }
  function legend() {
    return '<ul class="legend">' + state.E.scale.map(function (s) {
      return '<li title="' + esc(L(s.name)) + '">' + chip(s.id) + esc(t('card_tier_' + s.id)) + '</li>';
    }).join('') + '</ul>';
  }
  // who holds the top letter in each column of the main table
  function picks(E) {
    var top = E.scale[0].id, items = E.picks.filter(function (p) { return p.models.length; }).map(function (p) {
      return '<li><span class="g">' + esc(L(G[p.group].name)) + '</span><b>' + p.models.map(function (id) { return esc(M[id].short); }).join(', ') + '</b></li>';
    });
    return items.length ? '<div class="picks"><h3>' + chip(top) + esc(t('picks_title')) + '</h3><ul>' + items.join('') + '</ul></div>' : '';
  }
  function cellBtn(kind, key, r, k, g, who, what) {
    var words = chipWords(g);
    return '<button class="t t-' + (g || 'N') + '" type="button" data-' + kind + '="' + esc(key) + '" data-r="' + r + '" data-k="' + k + '" aria-expanded="false" aria-label="' +
      esc(who + ', ' + what + ': ' + words) + '" title="' + esc(what + ': ' + words) + '">' + (g || '–') + '</button>';
  }
  // three views of one table: my ranking with the letters of the main columns, every kind of work, or one group opened
  function matrix(E) {
    var mode = state.group, rating = mode === 'rating';
    var groups = rating ? MAIN : mode === 'all' ? E.groups : E.groups.filter(function (g) { return g.id === mode; });
    var tpl = 'var(--namew)', cols = [], sp = '<i class="sp" aria-hidden="true"></i>', h = [];
    if (rating) {
      tpl += ' repeat(' + groups.length + ', minmax(var(--cw), 1fr))';
      groups.forEach(function (g) { cols.push({ g: g }); });
    } else {
      groups.forEach(function (g, i) {
        if (i) tpl += ' var(--gapw)';
        // a group of two kinds of work must still hold its name: "Challenge" does not fit over two narrow columns
        var need = Math.ceil((L(g.name).length * 7.6 + 10) / g.works.length);
        tpl += ' repeat(' + g.works.length + ', minmax(max(var(--cw), ' + need + 'px), 1fr))';
        g.works.forEach(function (w, k) { cols.push({ w: W[w], gap: i > 0 && k === 0 }); });
      });
    }
    var corner = '<div class="corner" role="columnheader"><span class="rk" aria-hidden="true">#</span><span>' + esc(t('corner')) + '</span></div>';
    h.push('<div class="mx" data-group="' + esc(mode) + '" style="--tpl:' + tpl + '" role="table" aria-label="' + esc(t('table_label')) + '">');
    if (rating) {
      h.push('<div class="hr hr2" role="row">' + corner);
    } else {
      h.push('<div class="hr hr1" role="row">' + corner);
      groups.forEach(function (g, i) {
        if (i) h.push(sp);
        h.push('<div class="gh" style="grid-column:span ' + g.works.length + '" role="columnheader"><span>' + esc(L(g.name)) + '</span></div>');
      });
      h.push('</div><div class="hr hr2" role="row">');
    }
    cols.forEach(function (c) {
      if (c.gap) h.push(sp);
      if (c.g) {
        h.push('<div class="wh" role="columnheader"><button class="wbtn" type="button" data-g="' + esc(c.g.id) + '" title="' + esc(t('open_group_title') + ': ' + L(c.g.name)) +
          '"><span class="wl">' + esc(L(c.g.name)) + '</span></button></div>');
      } else {
        h.push('<div class="wh" role="columnheader"><button class="wbtn" type="button" data-w="' + esc(c.w.id) + '" aria-expanded="false" title="' +
          esc(L(c.w.name) + '. ' + L(c.w.what)) + '"><span class="wl">' + esc(L(c.w.name)) + '</span></button></div>');
      }
    });
    h.push('</div>');
    RATED.forEach(function (m, r) {
      h.push('<div class="r" role="row"><div class="m" role="rowheader"><button class="mbtn" type="button" data-m="' + esc(m.id) + '" aria-expanded="false" title="' +
        esc(m.name) + '" aria-label="' + esc(t('rank_label', { n: m.rank }) + ': ' + m.name) + '"><span class="rk">' + m.rank + '</span><span class="nm">' + esc(m.short) + '</span></button></div>');
      cols.forEach(function (c, k) {
        if (c.gap) h.push(sp);
        var inner = c.g ? cellBtn('gc', m.id + '|' + c.g.id, r, k, m.groups[c.g.id].grade, m.short, L(c.g.name))
          : cellBtn('c', m.id + '|' + c.w.id, r, k, CELL[m.id + '|' + c.w.id].grade, m.short, L(c.w.name));
        h.push('<div class="c" role="cell">' + inner + '</div>');
      });
      h.push('</div>');
    });
    h.push('</div>');
    return h.join('');
  }
  function chipBtn(id, name, n) {
    return '<button class="chip" type="button" data-g="' + esc(id) + '" aria-pressed="' + (state.group === id) + '">' + esc(name) + '<small>' + n + '</small></button>';
  }
  function board(E) {
    var chips = chipBtn('rating', t('view_rating'), MAIN.length) + chipBtn('all', t('all_works'), E.works.length);
    E.groups.forEach(function (g) { chips += chipBtn(g.id, L(g.name), g.works.length); });
    return '<section class="board" aria-labelledby="board-h"><h2 id="board-h" class="sr">' + esc(t('board_title')) + '</h2>' +
      '<div class="board-head">' + legend() + '<p class="note">' + esc(t('board_hint')) + '</p></div>' +
      '<div class="chips" role="group" aria-label="' + esc(t('group_filter')) + '">' + chips + '</div>' +
      '<p class="hint">' + esc(t('scroll_hint')) + '</p><div class="mxwrap">' + matrix(E) + '</div>' + picks(E) + '</section>';
  }
  function setGroup(id) {
    closeDetail(false);
    state.group = id;
    var chips = app.querySelectorAll('.chip'), wrap = app.querySelector('.mxwrap'), on = null;
    for (var i = 0; i < chips.length; i++) {
      var pressed = chips[i].getAttribute('data-g') === id;
      chips[i].setAttribute('aria-pressed', String(pressed));
      if (pressed) on = chips[i];
    }
    wrap.innerHTML = matrix(state.E);
    wrap.scrollLeft = 0;
    fit();
    if (on && !app.contains(doc.activeElement)) on.focus({ preventScroll: true });
  }
  function factsList(list) {
    return list.length ? '<ul class="facts">' + list.map(function (f) { return '<li>' + f + '</li>'; }).join('') + '</ul>' : '';
  }
  function casesList(ids) {
    if (!ids || !ids.length) return '<p class="none">' + esc(t('cases_none')) + '</p>';
    return '<ul class="cases">' + ids.map(function (id) {
      var c = CASE[id];
      if (!c) return '';
      // a case that several models share names them with their exact versions: its text may say just "Fable",
      // and the Fable of that case need not be the Fable of the ranking
      var who = c.models.length > 1 ? '<small>' + c.models.map(function (m) { return esc(M[m] ? M[m].short : m); }).join(' · ') + '</small>' : '';
      return '<li><time datetime="' + esc(c.from) + '">' + esc(caseWhen(c)) + '</time><span>' + esc(L(c.text)) + who + '</span></li>';
    }).join('') + '</ul>';
  }
  // records from my logs: what happened and how often, never a rate of correctness
  function cellFacts(E, c) {
    var s = c.stats || {}, out = [], v, line;
    if (s.review) {
      v = s.review;
      out.push(t('f_review', { seats: v.seats, answered: v.answered }));
      if (v.crit) out.push(t('f_review_crit', { crit: v.crit }));
      if (v.sole) out.push(t('f_review_sole', { sole: v.sole, fixed: v.sole_fixed }));
      if (v.missed) out.push(t('f_review_missed', { a: v.missed[0], b: v.missed[1] }));
      if (v.eye) out.push(t('f_review_eye', { n: v.eye[0], real: v.eye[1], rare: v.eye[2], opt: v.eye[3], wrong: v.eye[4] }));
      if (v.route) {
        line = !v.route.api ? t('f_route_agent') : !v.route.agent ? t('f_route_api') : t('f_route', { agent: v.route.agent, api: v.route.api });
        if (v.route.api > v.route.agent) line += ' ' + t('f_route_note');
        out.push(line);
      }
    }
    if (s.plan) {
      v = s.plan;
      out.push(t('f_plan', { answers: v.answers, zero: v.zero }));
      if (v.objections) out.push(t('f_plan_obj', { obj: v.objections, lifted: v.lifted, open: v.open }));
      if (v.unparseable >= 3) out.push(t('f_plan_unparsed', { n: v.unparseable }));
    }
    if (s.safety) out.push(t('f_safety', { n: s.safety.calls }));
    if (s.lead) out.push(t('f_lead', { roots: s.lead.roots }));
    if (s.first_pass) {
      v = s.first_pass;
      out.push(t('f_first_pass', { chains: v.chains, first: v.first }) + (v.self_judged ? ' ' + t('f_first_pass_self') : ''));
    }
    return out;
  }
  function modelFacts(E, m) {
    var s = m.stats || {}, out = [];
    if (s.records) out.push(t('f_model_records', { n: s.records }));
    if (s.review) out.push(t('f_model_review', { seats: s.review.seats, crit: s.review.crit }));
    if (s.plan) out.push(t('f_model_plan', { answers: s.plan.answers, obj: s.plan.objections }));
    (s.routes || []).forEach(function (r) {
      var line = t('f_model_route', { harness: r.harness, n: r.n, pct: r.ok_pct + '%' });
      if (r.top && r.top.length) line += t('f_model_route_fail', { list: r.top.map(function (x) { return t('fail_' + x.kind) + ' ' + num(x.n); }).join(', ') });
      out.push(line + '.');
    });
    return out;
  }
  function split(w, me) {
    var segs = '', keys = '';
    w.split.forEach(function (s) {
      var on = s.model === me, short = M[s.model] ? M[s.model].short : s.model;
      segs += '<i class="seg-b' + (on ? ' on' : '') + '" style="flex:' + Math.max(s.share, 0.6) + ' 0 0"></i>';
      keys += '<li' + (on ? ' class="on"' : '') + '>' + esc(short) + ' ' + esc(s.n && !s.share ? share(0.01) : share(s.share)) + '</li>';
    });
    if (w.other) {
      segs += '<i class="seg-b rest" style="flex:' + Math.max(w.other.share, 0.6) + ' 0 0"></i>';
      keys += '<li>' + esc(t('dist_others')) + ' ' + esc(share(w.other.share)) + '</li>';
    }
    return '<div class="dist"><div class="cap">' + esc(t('dist_caption')) + '</div><div class="barrow" role="img" aria-label="' + esc(t('dist_label')) + '">' + segs + '</div><ul class="keys">' + keys + '</ul></div>';
  }
  function dhead(badge, title, sub) {
    return '<div class="dh">' + badge + '<div class="dt"><h3 id="detail-h">' + title + '</h3><p class="dsub">' + sub + '</p></div>' +
      '<button class="x" type="button" data-close aria-label="' + esc(t('close')) + '">' + ICON_X + '</button></div>';
  }
  function why(text) { return text ? '<p class="why mine"><b>' + esc(t('judge_label')) + '</b> ' + esc(L(text)) + '</p>' : ''; }
  // a kind of work has a note of its own, or it simply takes the letter of its column, and then the column's reason
  function reasonOf(c) {
    if (c.note) return c.note;
    var g = G[W[c.work].group], x = M[c.model].groups[g.id];
    return g.main && c.grade && x && x.grade === c.grade ? x.reason : null;
  }
  function holders(wid) {
    return RATED.filter(function (m) { return CELL[m.id + '|' + wid].grade; }).map(function (m) { return m.short; }).join(', ');
  }
  function seatNote(w) { return '<p class="why note">' + t('unrated_' + w.unrated, { holder: holders(w.id) }) + '</p>'; }
  // who has which letter: one line per letter, models in my order
  function tierLines(pairs) {
    var by = {}, lines = '';
    pairs.forEach(function (x) { if (x.grade) (by[x.grade] = by[x.grade] || []).push(x.short); });
    state.E.scale.forEach(function (s) {
      if (by[s.id]) lines += '<li>' + chip(s.id) + '<span>' + by[s.id].map(esc).join(', ') + '</span></li>';
    });
    return lines ? '<ul class="tierlines">' + lines + '</ul>' : '';
  }
  function workLetters(wid) { return tierLines(RATED.map(function (m) { return { short: m.short, grade: CELL[m.id + '|' + wid].grade }; })); }
  function groupLetters(gid) { return tierLines(RATED.map(function (m) { return { short: m.short, grade: m.groups[gid].grade }; })); }
  function recordScope(wid) {
    var key = wid === 'mind' ? 'records_scope_mind' : wid === 'rev_safety' ? 'records_scope_safety' : null;
    return key ? '<p class="quiet">' + esc(t(key)) + '</p>' : '';
  }
  function cellHtml(E, key) {
    var c = CELL[key], m = M[c.model], w = W[c.work], g = c.grade, u = c.use;
    var sub = esc(g ? t('tier_long_' + g) : t('unrated_' + c.unrated + '_long')) + '. ' + esc(L(w.what));
    var use = '<p class="why use"><b>' + esc(t('use_label')) + '</b> ' + esc(t('use_line_' + (u.band || 'never'))) +
      (u.n ? ' ' + t('use_records', { n: u.n, share: !u.share ? share(0.01) : share(u.share) }) : '') + '</p>' + recordScope(c.work);
    return dhead(chip(g), esc(m.short) + ' · ' + esc(L(w.name)), sub) +
      '<div class="dg"><div class="dcol">' + (g ? why(reasonOf(c)) : seatNote(w)) + use + (u.n ? split(w, m.id) : '') + factsList(cellFacts(E, c)) + '</div>' +
      '<div class="dcol"><h4>' + esc(t('letters_title')) + '</h4>' + workLetters(w.id) +
      (c.cases ? '<h4 class="in">' + esc(t('cases_title')) + '</h4>' + casesList(c.cases) : '') + '</div></div>';
  }
  function groupCellHtml(E, key) {
    var i = key.indexOf('|'), m = M[key.slice(0, i)], g = G[key.slice(i + 1)], x = m.groups[g.id];
    // the kinds of work inside the column; where my advice differs from the column, its note stands right there
    var told = {}, inside = g.works.map(function (w) {
      var c = CELL[m.id + '|' + w], text = c.note && c.grade !== x.grade ? L(c.note) : '';
      if (told[text]) text = '';   // a note that several kinds of work share is printed once, at the first of them
      told[text] = true;
      return '<li>' + chip(c.grade) + '<span>' + esc(L(W[w].name)) + (text ? '<small>' + esc(text) + '</small>' : '') + '</span></li>';
    }).join('');
    return dhead(chip(x.grade), esc(m.short) + ' · ' + esc(L(g.name)), esc(t('tier_long_' + x.grade)) + '.') +
      '<div class="dg"><div class="dcol">' + why(x.reason) +
      '<h4 class="in">' + esc(t('inside_title')) + '</h4><ul class="tierlines inside">' + inside + '</ul>' +
      '<p class="more"><button class="link" type="button" data-g="' + esc(g.id) + '">' + esc(t('open_group')) + '</button></p></div>' +
      '<div class="dcol"><h4>' + esc(t('letters_title')) + '</h4>' + groupLetters(g.id) +
      (x.cases ? '<h4 class="in">' + esc(t('cases_title')) + '</h4>' + casesList(x.cases) : '') + '</div></div>';
  }
  function groupPills(m) {
    var out = '';
    MAIN.forEach(function (g) { out += '<span class="gp">' + chip(m.groups[g.id].grade) + esc(L(g.name)) + '</span>'; });
    return out ? '<div class="gps">' + out + '</div>' : '';
  }
  function trustCheck(m) {
    return '<p class="tc"><b>' + esc(t('trust_label')) + '</b> ' + esc(L(m.trust)) + '</p><p class="tc"><b>' + esc(t('check_label')) + '</b> ' + esc(L(m.check)) + '</p>';
  }
  function pickLine(m) {
    if (!m.picks || !m.picks.length) return '';
    return '<p class="tc"><b>' + esc(t('picks_for')) + '</b> ' + esc(m.picks.map(function (id) { return lower(L(G[id].name)); }).join(', ')) + '</p>';
  }
  function noteBox(m) { return m.note ? '<p class="why note"><b>' + esc(t('note_label')) + '</b> ' + esc(L(m.note)) + '</p>' : ''; }
  function rankBadge(m) { return '<span class="t rkb" aria-label="' + esc(t('rank_label', { n: m.rank })) + '">' + m.rank + '</span>'; }
  function modelHtml(E, id) {
    var m = M[id];
    return dhead(rankBadge(m), esc(m.name), esc(m.maker) + '. ' + esc(L(m.one_liner))) +
      '<div class="dg"><div class="dcol"><p class="dlong">' + esc(L(m.verdict)) + '</p>' + groupPills(m) + trustCheck(m) + pickLine(m) + noteBox(m) + factsList(modelFacts(E, m)) + '</div>' +
      '<div class="dcol"><h4>' + esc(t('cases_title')) + '</h4>' + casesList(m.cases) + '</div></div>';
  }
  function workHtml(E, id) {
    var w = W[id];
    var k = '<div class="kpis"><div><b>' + num(w.total) + '</b><span>' + esc(forms('kpi_logged', w.total)) + '</span></div></div>';
    return dhead('', esc(L(w.name)), esc(L(w.what))) +
      '<div class="dg"><div class="dcol"><h4>' + esc(t('letters_title')) + '</h4>' + workLetters(id) + (w.unrated ? seatNote(w) : '') + '</div>' +
      '<div class="dcol">' + k + split(w, null) + recordScope(w.id) + '<p class="quiet">' + esc(t('use_moves_nothing')) + '</p></div></div>';
  }
  // the models in the order of the ranking, each with its portrait
  function modelsList(E) {
    var rows = RATED.map(function (m) {
      return '<li class="vrow"><div class="who"><h3><span class="rk">' + m.rank + '</span>' + esc(m.name) + '</h3><div class="mk">' + esc(m.maker) + '</div>' + groupPills(m) + '</div>' +
        '<div class="say"><p class="liner">' + esc(L(m.one_liner)) + '</p><p class="long">' + esc(L(m.verdict)) + '</p>' + trustCheck(m) +
        '<details><summary>' + esc(t('numbers_and_cases')) + '</summary>' + noteBox(m) + factsList(modelFacts(E, m)) + '<h4>' + esc(t('cases_title')) + '</h4>' + casesList(m.cases) + '</details></div></li>';
    });
    var ps = E.notes && E.notes.postscript ? '<p class="ps"><span>' + esc(L(E.notes.postscript)) + '</span></p>' : '';
    return '<section class="sec" aria-labelledby="models-h"><h2 id="models-h">' + esc(t('models_title')) + '</h2><p class="note">' + esc(t('models_note')) + '</p><ol class="vlist">' + rows.join('') + '</ol>' + ps + '</section>';
  }
  // earlier versions and newcomers stand apart: a note each, no letters
  function apartList(E, status) {
    var out = '';
    E.models.forEach(function (m) {
      if (m.status !== status) return;
      out += '<li><b>' + esc(m.name) + '.</b> ' + esc(L(m.note)) +
        (m.cases && m.cases.length ? '<details><summary>' + esc(t('cases_only')) + '</summary>' + casesList(m.cases) + '</details>' : '') + '</li>';
    });
    return out;
  }
  function outside(E) {
    var earlier = apartList(E, 'earlier'), trial = apartList(E, 'trial'), never = [];
    E.models.forEach(function (m) { if (m.status === 'untried') never.push(m.name); });
    if (never.length) trial += '<li><b>' + esc(t('never_tried')) + '</b> ' + esc(never.join(', ')) + '.</li>';
    return '<section class="sec pad" aria-labelledby="outside-h"><h2 id="outside-h">' + esc(t('outside_title')) + '</h2>' +
      (earlier ? '<h3 class="subh">' + esc(t('earlier_title')) + '</h3><p class="note">' + esc(t('earlier_note')) + '</p><ul class="plain">' + earlier + '</ul>' : '') +
      (trial ? '<h3 class="subh">' + esc(t('trial_title')) + '</h3><p class="note">' + esc(t('trial_note')) + '</p><ul class="plain">' + trial + '</ul>' : '') + '</section>';
  }
  function how(E) {
    var tok = E.totals.tokens || {}, items = [], line, span = caseSpan(E), extra = [], seat = null;
    E.groups.forEach(function (g) { if (!g.main) extra.push(L(g.name)); });
    E.works.forEach(function (w) { if (w.unrated && !seat) seat = w; });
    items.push(t('how_frame'));
    items.push(t('how_order'));
    items.push(t('how_grade'));
    items.push(t('how_works', { works: E.totals.work_types, columns: E.totals.columns, extra: extra.join(', ') }));
    if (seat) items.push(t('how_seat', { work: L(seat.name), holder: holders(seat.id) }));
    items.push(t('how_cases', { n: E.totals.cases }) + (span ? ' ' + t('how_cases_span', { from: dateDay(span[0]), to: dateLong(span[1]) }) : ''));
    items.push(t('how_records', { from: dateDay(E.period.from), to: dateLong(E.period.to) }));
    if (tok.status === 'final' && tok.definition) {
      line = t('how_tokens', { read: tokenText(tok, tok.read).replace('+', ''), since: dateLong(tok.since), cache: Math.round(100 * (tok.cache_share || 0)) + '%', written: compactText(tok.written), definition: L(tok.definition) });
      if (tok.period_read) line += ' ' + t('how_tokens_period', { tokens: compactText(tok.period_read), note: tok.period_read_note ? L(tok.period_read_note) : '' });
      items.push(line);
    } else {
      items.push(t('how_tokens_pending'));
    }
    items.push(t('how_left_out'));
    items.push(E.status === 'final' ? t('how_made_final') : t('how_made_draft'));
    return '<section class="sec pad" id="how" aria-labelledby="how-h"><h2 id="how-h">' + esc(t('how_title')) + '</h2><ul class="plain">' +
      items.map(function (x) { return '<li>' + x + '</li>'; }).join('') +
      '<li><a href="' + esc(SOURCE) + '/METHOD.md">' + esc(t('full_method')) + '</a></li></ul></section>';
  }
  function renderRating() {
    var E = state.E;
    if (state.group !== 'rating' && state.group !== 'all' && !E.groups.some(function (g) { return g.id === state.group; })) state.group = 'rating';
    index(E);
    setHead(t('title'));
    app.innerHTML = bar() + '<main class="shell page">' + hero(E) + board(E) + tiles(E) + modelsList(E) +
      '<div class="two">' + outside(E) + how(E) + '</div>' + foot(E) + '</main>';
    fit();
    var wrap = app.querySelector('.mxwrap');
    if (wrap && window.ResizeObserver) { if (live) live.disconnect(); live = new window.ResizeObserver(fit); live.observe(wrap); }
    if (state.open) { var b = find(state.open); state.open = null; if (b) openDetail(b, true); }
  }
  // column names lie flat while a column is wide enough for them; in a narrow table they stand upright
  function fit() {
    var mx = app.querySelector('.mx'), wrap = app.querySelector('.mxwrap');
    if (!mx || !wrap) return;
    var cols = mx.querySelectorAll('.wh').length, side = mx.querySelector('.corner').offsetWidth;
    mx.classList.toggle('hz', state.group !== 'all' && cols > 0 && (wrap.clientWidth - side) / cols >= 84);
    var hint = app.querySelector('.hint');
    if (hint) hint.hidden = wrap.scrollWidth <= wrap.clientWidth + 1;
  }

  // ---------- detail
  var current = null;
  function find(o) {
    var list = app.querySelectorAll('[data-' + o.kind + ']');
    for (var i = 0; i < list.length; i++) if (list[i].getAttribute('data-' + o.kind) === o.id) return list[i];
    return null;
  }
  function closeDetail(refocus) {
    if (!current) return;
    var c = current;
    current = null; state.open = null;
    if (c.scrim) c.scrim.remove();
    c.el.remove();
    root.classList.remove('locked');
    if (c.btn && doc.contains(c.btn)) {
      c.btn.classList.remove('sel');
      c.btn.setAttribute('aria-expanded', 'false');
      if (refocus) c.btn.focus();
    }
  }
  function openDetail(btn, quiet) {
    var kind = btn.hasAttribute('data-c') ? 'c' : btn.hasAttribute('data-gc') ? 'gc' : btn.hasAttribute('data-m') ? 'm' : 'w', id = btn.getAttribute('data-' + kind);
    if (current && current.kind === kind && current.id === id) { closeDetail(true); return; }
    closeDetail(false);
    var E = state.E, html = kind === 'c' ? cellHtml(E, id) : kind === 'gc' ? groupCellHtml(E, id) : kind === 'm' ? modelHtml(E, id) : workHtml(E, id);
    var inline = wide(), el = doc.createElement('div'), scrim = null;
    el.innerHTML = '<div class="dbox" tabindex="-1" aria-labelledby="detail-h" role="' + (inline ? 'region' : 'dialog') + '"' + (inline ? '' : ' aria-modal="true"') + '>' + html + '</div>';
    if (inline) {
      el.className = 'detail';
      var row = btn.closest('.r') || app.querySelector('.hr2');
      row.after(el);
    } else {
      el.className = 'sheet';
      scrim = doc.createElement('div');
      scrim.className = 'scrim';
      scrim.addEventListener('click', function () { closeDetail(true); });
      body.appendChild(scrim);
      body.appendChild(el);
      root.classList.add('locked');
    }
    btn.classList.add('sel');
    btn.setAttribute('aria-expanded', 'true');
    current = { el: el, btn: btn, kind: kind, id: id, scrim: scrim, inline: inline };
    state.open = { kind: kind, id: id };
    el.addEventListener('click', function (e) {
      var gb = e.target.closest('[data-g]');
      if (gb) { e.stopPropagation(); setGroup(gb.getAttribute('data-g')); return; }
      if (e.target.closest('[data-close]')) closeDetail(true);
    });
    if (!quiet) {
      el.firstChild.focus({ preventScroll: true });
      if (inline) {
        var r = el.getBoundingClientRect(), vh = window.innerHeight || root.clientHeight;
        var smooth = !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (r.bottom > vh - 8) el.scrollIntoView({ block: r.height > vh - 200 ? 'start' : 'nearest', behavior: smooth ? 'smooth' : 'auto' });
      }
    }
  }
  function moveFocus(btn, dr, dk) {
    var r = Number(btn.getAttribute('data-r')), k = Number(btn.getAttribute('data-k'));
    for (var step = 0; step < 64; step++) {
      r += dr; k += dk;
      if (r < 0 || k < 0 || r >= RATED.length || k > 63) return;
      var next = app.querySelector('.t[data-r="' + r + '"][data-k="' + k + '"]');
      if (next) { next.focus(); return; }
    }
  }

  // ---------- editions page
  function renderEditions() {
    setHead(t('editions_title') + ' · ' + t('title'));
    var list = (EDS && EDS.editions) || [], latest = EDS && EDS.latest, items = '';
    list.forEach(function (e) {
      var isLatest = e.edition === latest, draft = e.status !== 'final';
      var card = isLatest ? 'card/latest-' + state.lang + '-' + theme() + '.png' : e.card;
      var page = isLatest ? 'index.html' : 'index.html?edition=' + encodeURIComponent(e.edition);
      items += '<li class="ed"><div><h2>' + esc(e.edition) + (isLatest ? '<span class="pill">' + esc(t('ed_latest')) + '</span>' : '') +
        (draft ? '<span class="pill draft">' + esc(t('ed_draft')) + '</span>' : '') + '</h2>' +
        '<dl><dt>' + esc(t('ed_graded')) + '</dt><dd>' + t('ed_graded_value', { models: e.models_rated, works: e.work_types }) + '</dd>' +
        (e.top && e.top.length ? '<dt>' + esc(t('ed_top')) + '</dt><dd>' + esc(e.top.join(', ')) + '</dd>' : '') +
        '<dt>' + esc(t('ed_published')) + '</dt><dd>' + esc(e.published ? dateLong(e.published) : t('ed_not_published')) + '</dd>' +
        '<dt>' + esc(t('ed_period')) + '</dt><dd>' + esc(dateRange(e.period.from, e.period.to, true)) + '</dd>' +
        '<dt>' + esc(t('ed_method')) + '</dt><dd>v' + esc(e.method_version) + '</dd></dl>' +
        '<p class="moved"><b>' + esc(t('ed_moved')) + '.</b> ' + esc(L(e.changelog)) + '</p>' +
        '<div class="links"><a href="' + esc(withLang(page)) + '">' + esc(t('ed_page')) + '</a><a href="' + esc(e.data) + '">' + esc(t('ed_data')) + '</a>' +
        (card ? '<a href="' + esc(card) + '">' + esc(t('ed_card')) + '</a>' : '') + '</div></div>' +
        (card ? '<figure><a href="' + esc(withLang(page)) + '"><img src="' + esc(card) + '" alt="' + esc(t('title') + ', ' + e.edition) + '" width="1600" height="900" loading="lazy"></a></figure>' : '') + '</li>';
    });
    app.innerHTML = bar() + '<main class="shell page"><section class="head"><div class="head-main"><p class="eyebrow">' + esc(t('eyebrow_editions')) + '</p>' +
      '<h1>' + esc(t('editions_title')) + '</h1><p class="lead">' + esc(t('editions_lead')) + ' ' + esc(t('ed_method_note')) + '</p></div></section>' +
      '<ul class="edlist">' + items + '</ul>' + foot(null) + '</main>';
  }
  function render() { closeDetailSilently(); if (PAGE === 'editions') renderEditions(); else renderRating(); }
  function closeDetailSilently() { var keep = state.open; closeDetail(false); state.open = keep; }

  // ---------- events
  app.addEventListener('click', function (e) {
    var b = e.target.closest('button');
    if (!b || !app.contains(b)) return;
    if (b.hasAttribute('data-lang')) { if (b.getAttribute('data-lang') !== state.lang) setLang(b.getAttribute('data-lang')); return; }
    if (b.hasAttribute('data-theme-toggle')) {
      var next = theme() === 'dark' ? 'light' : 'dark';
      applyTheme(next); store(THEME_KEY, next);
      if (PAGE === 'editions') render();
      return;
    }
    if (b.hasAttribute('data-g')) { setGroup(b.getAttribute('data-g')); return; }
    if (b.hasAttribute('data-c') || b.hasAttribute('data-gc') || b.hasAttribute('data-m') || b.hasAttribute('data-w')) openDetail(b, false);
  });
  doc.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && current) { e.preventDefault(); closeDetail(true); return; }
    if (e.key === 'Tab' && current && !current.inline) {
      var f = current.el.querySelectorAll('button, a[href], summary');
      if (!f.length) return;
      var first = f[0], last = f[f.length - 1], a = doc.activeElement;
      if (!current.el.contains(a)) { e.preventDefault(); first.focus(); }
      else if (e.shiftKey && (a === first || a === current.el.firstChild)) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && a === last) { e.preventDefault(); first.focus(); }
      return;
    }
    var tgt = e.target;
    if (tgt && tgt.hasAttribute && tgt.hasAttribute('data-k') && !(e.altKey || e.ctrlKey || e.metaKey || e.shiftKey)) {
      if (e.key === 'ArrowRight') { e.preventDefault(); moveFocus(tgt, 0, 1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); moveFocus(tgt, 0, -1); }
      else if (e.key === 'ArrowDown') { e.preventDefault(); moveFocus(tgt, 1, 0); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); moveFocus(tgt, -1, 0); }
    }
  });
  if (window.matchMedia) {
    var mw = window.matchMedia(WIDE);
    var onWide = function () {
      if (!current || current.inline === wide()) return;
      var keep = { kind: current.kind, id: current.id };
      closeDetail(false);
      var b = find(keep);
      if (b) openDetail(b, true);
    };
    if (mw.addEventListener) mw.addEventListener('change', onWide); else if (mw.addListener) mw.addListener(onWide);
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onScheme = function (ev) { if (!stored(THEME_KEY)) { applyTheme(ev.matches ? 'dark' : 'light'); if (PAGE === 'editions') render(); } };
    if (mq.addEventListener) mq.addEventListener('change', onScheme); else if (mq.addListener) mq.addListener(onScheme);
  }

  // ---------- boot
  function validEdition(s) {
    if (typeof s !== 'string' || s.length < 1 || s.length > 12) return false;
    for (var i = 0; i < s.length; i++) {
      var ch = s.charAt(i), digit = ch >= '0' && ch <= '9';
      if (!digit && !(ch === '.' && i > 0 && i < s.length - 1)) return false;
    }
    return true;
  }
  function load(url, done, fail) {
    if (!window.fetch) { fail(); return; }
    window.fetch(url, { cache: 'no-cache', credentials: 'omit' }).then(function (r) { if (!r.ok) throw new Error(String(r.status)); return r.json(); }).then(done, fail);
  }
  function show(d) {
    var before = state.E, open = state.open;
    try { state.E = d; state.open = null; render(); return true; }
    catch (e) { state.E = before; state.open = open; render(); return false; }
  }
  state.lang = pickLang();
  if (PAGE === 'editions') { render(); return; }
  if (!EMBED) { app.textContent = 'The edition data did not load.'; return; }
  var want = query('edition'), local = window.location.protocol === 'file:';
  if (want && !validEdition(want)) want = null;
  if (want && want !== EMBED.edition && local) state.notice = { key: 'notice_edition', args: { want: want, shown: EMBED.edition } };
  render();
  if (local) return;
  if (want && want !== EMBED.edition) {
    load('data/editions/' + want + '/edition.json', function (d) { if (!show(d)) { state.notice = { key: 'notice_missing', args: { want: want, shown: EMBED.edition } }; render(); } },
      function () { state.notice = { key: 'notice_missing', args: { want: want, shown: EMBED.edition } }; render(); });
  } else if (!want) {
    load('data/latest.json', function (d) { if (d && (d.edition !== EMBED.edition || d.built !== EMBED.built || d.status !== EMBED.status)) show(d); }, function () { /* the embedded copy stays */ });
  }
})();
