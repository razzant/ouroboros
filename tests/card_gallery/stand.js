// Card gallery test stand (tests/card_gallery_stand.py serves it; the product never does).
// One scene = one card family in one recorded state, drawn by the product's own entry point for that family from
// producer output; the stand writes no card markup. A scene with more than one recorded frame offers
// "Next frame", which delivers the next frame through the same entry point the live socket uses.
// A scene is ready (`window.gallery.ready`, and its "All scenes" link `aria-current`, which the ui_look stand reads)
// only once it shows the state its address asks for: every read it started answered, every wait it names met. A wait
// that runs out, a page that says its read failed and an address naming no scene are said on the page instead, and
// the scene is never marked ready.
import { apiFetch } from '/static/modules/api_client.js';

const params = new URLSearchParams(location.search);
const root = document.querySelector('[data-gallery-root]');
const element = (tag, attrs = {}, text = '') => {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text) node.textContent = text;
    return node;
};
const corpus = async (name) => {
    const response = await fetch(`/gallery/corpus/${name}.json`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`/gallery/corpus/${name}.json answered ${response.status}`);
    return response.json();
};

// ---- the families: what each state is, where it comes from, how the product draws it

const WORK_CASES = {
    held_continue: 'A Continue the server holds: the work waits, with the reason it gives.',
    working_call: 'A work running one exact call.',
    direct_question: 'A direct turn waiting for the owner\'s answer.',
    paused_by_owner: 'A work the owner paused.',
    model_access: 'A work waiting for model access.',
    pausing_with_queued_helper: 'A work pausing while its helper is still queued.',
};

const FAMILIES = [
    {
        id: 'quiz', label: 'Question card', source: 'chat_decision.js buildQuizCard; quiz_state through applyQuizStateFrame',
        corpus: 'card_gallery.quiz',
        states: {
            open: 'An open question with no offered answers: the owner writes one.',
            long: 'A question whose words run long.',
            waiting: 'A question its task is waiting on.',
            closed: 'A question closed before an answer (second frame: quiz_state).',
            answered: 'An answered question (second frame: quiz_state with the owner\'s comment).',
        },
    },
    {
        id: 'living_card', label: 'Living card', source: 'chat_cards.js buildCard; card_state through applyCardState',
        corpus: 'card_gallery.living_card',
        states: {
            long: 'A card with a long title and body.',
            updated: 'A card the producer updates: revision 1, then card_state with revision 2.',
            missing: 'A card whose body is gone: GET /api/cards/{id} answers 404.',
        },
    },
    {
        id: 'reviews', label: 'Reviews in a task card',
        source: 'review_presentation.js createReviewPresentationController from the task detail\'s review_projection',
        corpus: 'card_gallery.reviews',
        states: {
            empty: 'No review panel yet.',
            waiting: 'A panel still waiting for a reviewer.',
            failure: 'A panel with a blocking failure and many findings.',
            missing_record: 'A review whose record is missing.',
        },
    },
    {
        id: 'work', label: 'Room work status',
        source: 'room_work.js createRoomWork over work_status.js reconcileWorkState; the second read is card_gallery.work_reads',
        corpus: 'work_projection_cases',
        states: {
            ...WORK_CASES,
            incomplete_read: 'A running work, then an incomplete read that does not list it: kept, unconfirmed.',
            confirmed_absent: 'A running work, then a complete read without it: the room is idle.',
        },
    },
    {
        id: 'consciousness', label: 'Consciousness tab', source: 'consciousness_dashboard.js initConsciousnessDashboard',
        corpus: 'consciousness_contract',
        states: {
            snapshot: 'The tab over the recorded consciousness facts.',
            read_failed: 'GET /api/consciousness fails (503); the tab says so.',
        },
    },
    {
        id: 'widgets', label: 'Widgets board', source: 'widgets.js initWidgets (Widgets page shown)',
        corpus: 'card_gallery.widgets',
        states: {
            empty: 'No widgets.',
            pinned_long: 'A pinned living card with a long body.',
        },
    },
    {
        id: 'skills', label: 'Skills page', source: 'skills.js initSkills (Skills page shown) over GET /api/extensions',
        corpus: 'card_gallery.skills',
        states: {
            empty: 'No installed skills.',
            review_blocked: 'A skill whose review blocks it.',
        },
    },
    {
        id: 'login', label: 'Sign-in card',
        source: 'harness_login_cards.js createLoginCardController: its start answered with the recorded job through its own transport, then its poll',
        corpus: 'card_gallery.login',
        states: {
            waiting: 'A synthetic engine login job waiting for the owner, with no link or code yet.',
            failed: 'A synthetic engine login job that failed.',
        },
    },
    {
        id: 'task_card', label: 'Task card in Main',
        source: 'chat.js createChatInstance over GET /api/chat/history, /api/state and /api/tasks/{id}; chat frames through its socket handlers',
        corpus: 'card_gallery.task_card',
        states: {
            running: 'A running task: its narration, its clock and its controls (next frame: a new line arrives live).',
            done: 'A finished task: its long answer under the card, its Details opened by its own control.',
            failed: 'A task the model provider did not answer: its reason, Continue, and its Details opened.',
            paused: 'A task the owner paused: the room says so and offers Resume.',
        },
    },
    {
        id: 'share', label: 'Share card',
        source: 'message_share.js registered as app.js does, on the answer in Main; message_share_card.js renderShareCard draws the Card',
        corpus: 'card_gallery.share',
        states: {
            card: 'A finished task\'s answer shared as a Card: the picture that leaves, drawn in the room\'s colours.',
            table: 'An answer with a table: the Card refuses it and says why; Text carries it whole.',
        },
    },
    {
        id: 'connection', label: 'Setup: connection card',
        source: 'onboarding_starter_skills.js createStarterSkillsStep (the setup\'s Skills step) over GET /api/onboarding/starter-skills; the host\'s recorded identity answer through the card\'s own check (checkIdentity), nothing posted',
        corpus: 'card_gallery.connection',
        states: {
            qr: 'A token the service accepted: its account, the QR and link to connect it, and the skill\'s option.',
            refused: 'A token the service refused: the card says so and will not save it.',
        },
    },
    {
        id: 'provider_card', label: 'Setup: provider card',
        source: 'settings_provider_cards.js providerSettingsCard and settings_connections.js createConnectionsEditor, as the setup wizard\'s providers panel composes and loads them, over its bootstrap',
        corpus: 'card_gallery.provider_card',
        states: {
            closed: 'No compatible server yet: the card stays closed, with Add key.',
            saved: 'A compatible server with its key saved: the card open on its Default key, the key never shown.',
        },
    },
    {
        id: 'agents', label: 'Settings › Agents', source: 'subagents_settings.js, mounted as section_levels.js mounts a Settings page, over GET /api/settings and /api/review-pool',
        corpus: 'card_gallery.agents',
        states: {
            empty_pool: 'One agent, none marked Reviewer: the list shows it and the pool is empty.',
            disabled_reviewer: 'A Reviewer switched off: it leaves the pool and the page says so.',
        },
    },
];

const sceneNames = () => FAMILIES.flatMap((family) => Object.keys(family.states).map((state) => `${family.id}.${state}`));
const firstScene = (family) => `${family.id}.${Object.keys(family.states)[0]}`;

// What the address asks for. An option it leaves out takes its default (the first scene, the browser's colour scheme,
// every frame); one it names that the stand does not have is said, never replaced by another.
function selection() {
    const scene = params.has('scene') ? params.get('scene') : firstScene(FAMILIES[0]);
    const [familyId, state, ...rest] = scene.split('.');
    const family = FAMILIES.find((each) => each.id === familyId);
    const known = Boolean(family && state && !rest.length && Object.hasOwn(family.states, state));
    const theme = params.get('theme');
    const themes = ['dark', 'light'];
    const frames = params.get('frames');
    const faults = [];
    if (!known) faults.push(`there is no scene "${scene}"`);
    if (theme !== null && !themes.includes(theme)) faults.push(`there is no theme "${theme}" (dark or light)`);
    if (frames !== null && !/^\d+$/.test(frames)) faults.push(`frames is a number of frames, not "${frames}"`);
    return {
        scene, family: known ? family : null, state: known ? state : '', faults,
        theme: themes.includes(theme) ? theme : (matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark'),
        frames: frames === null || !/^\d+$/.test(frames) ? null : Number(frames),
    };
}

// ---- the reads: each one this tab sends to the stand names its scene and frame

// The scene and the frame being drawn ride on every request the page sends to this stand's /api/ (X-Gallery-Scene,
// X-Gallery-Step), so two tabs of one browser, which share its cookies, each read their own scene. A request to another
// origin or outside /api/ goes out exactly as the page made it; the bare document signs nothing. `open` holds the
// signed requests not answered yet.
const reads = { scene: '', frame: 0, open: new Map(), next: 1 };

function signReads(scene) {
    const send = window.fetch.bind(window);
    reads.scene = scene;
    window.fetch = (input, init) => {
        let url = null;
        try {
            url = new URL(input instanceof Request ? input.url : String(input), document.baseURI);
        } catch { /* fetch itself says why */ }
        if (!url || url.origin !== location.origin || !url.pathname.startsWith('/api/')) return send(input, init);
        let request;
        try {
            request = new Request(input, init);  // its method, body, signal and the rest as the page gave them
        } catch (error) {
            return Promise.reject(error);
        }
        request.headers.set('X-Gallery-Scene', reads.scene);
        request.headers.set('X-Gallery-Step', String(reads.frame));
        const id = reads.next;
        reads.next += 1;
        reads.open.set(id, `${request.method} ${url.pathname}`);
        return send(request).finally(() => reads.open.delete(id));
    };
}

// ---- waiting: bounded, and said when it runs out

// Up to ten seconds, by the page's clock, for what a scene waits on (a card drawn, a control there); `ready` may throw
// for a page that says it failed. Past that the scene is not what was asked for, and says so.
async function until(ready, what, ms = 10000) {
    const end = performance.now() + ms;
    for (;;) {
        const met = ready();
        if (met) return met;
        if (performance.now() >= end) throw new Error(`waited ${ms / 1000} s for ${what}`);
        await new Promise((resolve) => setTimeout(resolve, 40));
    }
}

// Two animation frames after the last write, so the product's own frames have run. A tab the browser does not paint
// (one in the background) runs none, so each is waited for at most a quarter second.
async function settle() {
    for (let i = 0; i < 2; i += 1) {
        await new Promise((resolve) => {
            const timer = setTimeout(resolve, 250);
            requestAnimationFrame(() => { clearTimeout(timer); resolve(); });
        });
    }
}

// Every read the frame started answered and drawn, and none started by what those answers drew.
async function answered(ms = 10000) {
    const end = performance.now() + ms;
    for (;;) {
        if (!reads.open.size) {
            await settle();
            if (!reads.open.size) return;
        }
        if (performance.now() >= end) {
            throw new Error(`waited ${ms / 1000} s for its reads: ${[...reads.open.values()].join(', ')} not answered`);
        }
        await new Promise((resolve) => setTimeout(resolve, 40));
    }
}

// ---- the chrome: a quiet bar outside the product frame

function chrome(pick) {
    const { family, state, theme } = pick;
    const bar = element('header', { class: 'gallery-bar', 'aria-label': 'Card gallery' });
    const title = element('p', { class: 'gallery-title' }, 'Card gallery');
    title.append(element('span', { class: 'gallery-tag' }, 'test stand'));
    const select = (label, name, options, current) => {
        const field = element('label', { class: 'gallery-field' });
        field.append(element('span', { class: 'gallery-field-label' }, label));
        const control = element('select', { class: 'ui-control', 'data-gallery-pick': name, 'aria-label': label });
        for (const [value, words] of options) {
            const option = element('option', { value }, words);
            if (value === current) option.selected = true;
            control.append(option);
        }
        field.append(control);
        return field;
    };
    const go = (scene, chosen) => { location.search = `?scene=${encodeURIComponent(scene)}&theme=${chosen}`; };
    // An address naming no scene leaves the card unchosen: the picker offers the cards there are.
    const familyPick = select('Card', 'family', [...(family ? [] : [['', 'Choose a card']]),
        ...FAMILIES.map((f) => [f.id, f.label])], family?.id || '');
    familyPick.querySelector('select').addEventListener('change', (event) => {
        const next = FAMILIES.find((f) => f.id === event.target.value);
        if (next) go(firstScene(next), theme);
    });
    bar.append(title, familyPick);
    if (family) {
        const statePick = select('State', 'state', Object.keys(family.states).map((s) => [s, s.replaceAll('_', ' ')]), state);
        statePick.querySelector('select').addEventListener('change', (event) => go(`${family.id}.${event.target.value}`, theme));
        bar.append(statePick);
    }
    const themePick = select('Theme', 'theme', [['dark', 'Dark'], ['light', 'Light']], theme);
    themePick.querySelector('select').addEventListener('change', (event) => go(pick.scene, event.target.value));
    bar.append(themePick);
    root.append(bar);
    if (family) {
        const what = element('p', { class: 'gallery-what', 'data-gallery-what': '' }, family.states[state]);
        // Where the scene comes from opens on demand: on a phone it would push the card a screen down.
        const source = element('details', { class: 'gallery-source' });
        source.append(element('summary', {}, 'Source'), element('span', {}, `${family.source} · ${family.corpus}`));
        root.append(what, source);
    }
    // Every scene, one link each: the same address the pickers go to. The ui_look stand navigates by these
    // (data-nav-page) and takes the one with aria-current for the scene shown, so only a drawn scene's link has it.
    const index = element('details', { class: 'gallery-index' });
    index.open = !family;
    index.append(element('summary', {}, `All scenes (${sceneNames().length})`));
    const nav = element('nav', { 'aria-label': 'Card gallery scenes' });
    let open = null;
    for (const each of FAMILIES) {
        const group = element('p', { class: 'gallery-index-family' }, `${each.label}:`);
        for (const name of Object.keys(each.states)) {
            const scene = `${each.id}.${name}`;
            const link = element('a', { href: `?scene=${scene}&theme=${theme}`, 'data-nav-page': scene }, name.replaceAll('_', ' '));
            if (each === family && name === state) open = link;
            group.append(' ', link);
        }
        nav.append(group);
    }
    index.append(nav);
    root.append(index);
    const mark = (shown) => { if (shown) open?.setAttribute('aria-current', 'page'); else open?.removeAttribute('aria-current'); };
    if (!family) return { mark };
    const actions = element('div', { class: 'gallery-actions ui-action-row' });
    const next = element('button', { type: 'button', class: 'btn btn-default btn-sm', 'data-gallery-next': '' });
    const counter = element('span', { class: 'gallery-counter', 'aria-live': 'polite', 'data-gallery-counter': '' });
    actions.append(next, counter);
    root.append(actions);
    return { next, counter, mark };
}

// What went wrong, said where the scene's description stands; the scene is not ready.
function fault(error) {
    const message = String(error?.message || error);
    root.querySelector('[data-gallery-fault]')?.remove();
    const line = element('p', { class: 'gallery-fault', role: 'alert', 'data-gallery-fault': '' },
        `The stand could not draw this scene: ${message}`);
    const after = root.querySelector('[data-gallery-what]') || root.querySelector('.gallery-bar');
    if (after) after.after(line);
    else root.append(line);
    const stack = String(error?.stack || '');  // WebKit's stack does not repeat the message; Chromium's starts with it
    window.gallery = { ...window.gallery, ready: false, scenes: sceneNames,
        failed: stack.includes(message) ? stack : [message, stack].filter(Boolean).join('\n') };
}

// ---- the scenes: each returns { frames: n, apply(k) } and mounts the subject into `stage`

const quietWs = () => ({ on() {}, off() {}, send() {}, isConnected: () => true });

async function chatColumn(stage) {
    const { createChatMedia } = await import('/static/modules/chat_media.js');
    const feed = element('div', { id: 'chat-messages', class: 'chat-messages gallery-feed' });
    stage.append(feed);
    const media = createChatMedia({ chatSessionId: 'gallery', durableChatMediaUrl: (x) => x || '',
        formatMsgTime: () => ({ short: '12:00', full: '20 November 2026, 12:00' }),
        insertMessageNode: (node) => feed.append(node), senderLabel: () => 'Ouroboros', stampNodeTimestamp: () => {} });
    return { feed, media };
}

const SCENES = {
    async quiz(stage, entry) {
        const { feed, media } = await chatColumn(stage);
        const { createChatDecision } = await import('/static/modules/chat_decision.js');
        const { mountChatMarkdown, enhanceChatMarkdown } = await import('/static/modules/chat_markdown.js');
        const decision = createChatDecision({ apiFetch, fetchDetail: async () => null, frameNode: media.bubbleFrameNode,
            mountMarkdown: mountChatMarkdown, enhanceMarkdown: enhanceChatMarkdown, showToast: () => {},
            isMain: true, chatId: 1 });
        const frames = entry.frames || [];
        return {
            frames: frames.length,
            apply(k) {
                if (k === 0) {
                    const card = decision.buildQuizCard(frames[0]);
                    card.dataset.gallerySubject = '';
                    feed.append(card);
                } else decision.applyQuizStateFrame(feed, frames[k]);
            },
        };
    },
    async living_card(stage, entry) {
        const { feed, media } = await chatColumn(stage);
        const { createChatCards } = await import('/static/modules/chat_cards.js');
        const cards = createChatCards({ frameNode: media.bubbleFrameNode, composer: () => null });
        const frames = entry.frames || [];
        return {
            frames: frames.length,
            apply(k) {
                if (k === 0) {
                    const card = cards.buildCard(frames[0]);
                    card.dataset.gallerySubject = '';
                    feed.append(card);
                } else cards.applyCardState(frames[k]);
            },
        };
    },
    async reviews(stage, entry) {
        const { createReviewPresentationController, reviewGroupsFromTaskDetail } = await import('/static/modules/review_presentation.js');
        const subject = element('div', { class: 'gallery-reviews', 'data-gallery-subject': '' });
        // The task card's door and host as chat.js names them; the card around them is chat.js's (see README).
        const summary = element('button', { type: 'button', class: 'chat-live-review-summary ui-hit-area', 'aria-expanded': 'false' });
        const host = element('div', { 'data-live-reviews-host': '' });
        subject.append(summary, host);
        stage.append(subject);
        const controller = createReviewPresentationController({ host, summary, disclosure: { sectionExpanded: true } });
        return {
            frames: 1,
            apply() {
                const detail = { task_id: entry.task_id, review_projection: entry.review_projection };
                controller.updateMany(reviewGroupsFromTaskDetail(detail, entry.task_id));
            },
        };
    },
    async work(stage, _entry, state, corpora) {
        const { renderPageHeader } = await import('/static/modules/page_header.js');
        const { mainHeaderActionsHtml } = await import('/static/modules/chat_header.js');
        const { createRoomWork } = await import('/static/modules/room_work.js');
        const { emptyWorkState, reconcileWorkState, roomStatusView } = await import('/static/modules/work_status.js');
        // A server case is its one read. The two reads after a running work are its read (the working_call case),
        // then the work fields the server answers next, recorded whole in card_gallery.work_reads.
        const served = Object.hasOwn(WORK_CASES, state);
        const first = corpora.work_projection_cases.cases?.[served ? state : 'working_call'];
        const second = served ? null : corpora.card_gallery.work_reads?.[state];
        if (!first || typeof first !== 'object') {
            throw new Error(`work_projection_cases records no case ${served ? state : 'working_call'}`);
        }
        if (!served && (!second || typeof second !== 'object')) {
            throw new Error(`the card corpus records no card_gallery.work_reads.${state}`);
        }
        // The room the server's case speaks for: Main (1) draws Main's bar, a Project room the panel's status bar,
        // both as chat.js writes them.
        const chatId = Number(Object.keys(first.rooms || {})[0] || first.works?.[0]?.room || 1);
        const asPanel = chatId !== 1;
        const page = element('div', { id: asPanel ? 'panel-gallery' : 'page-chat', 'data-gallery-subject': '',
            class: `${asPanel ? 'chat-instance-panel' : 'page active'} gallery-room` });
        page.innerHTML = `${asPanel ? '<div class="chat-panel-statusbar"><span id="chat-status" class="status-badge offline">Connecting…</span></div>'
            : renderPageHeader({ title: 'Main Chat', variant: 'overlay', className: 'chat-page-header',
                toolbarClass: 'ui-chrome-group', actionsHtml: mainHeaderActionsHtml(), showMobileNav: false })}
            <div id="chat-messages"></div>`;
        stage.append(page);
        // Each read as a connected, ready host answers it (supervisor_ready), with every field it recorded.
        const body = (read) => ({ supervisor_ready: true, ...read });
        const bodies = served ? [body(first)] : [body(first), body(second)];
        let workState = emptyWorkState();
        const badge = page.querySelector('#chat-status');
        const roomWork = createRoomWork({ page, asPanel, statusBadge: badge,
            messagesDiv: page.querySelector('#chat-messages'), reading: { stick: false, cancel() {}, scroll() {} },
            works: () => workState });
        return {
            frames: bodies.length,
            async apply(k) {
                workState = reconcileWorkState(workState, bodies[k], chatId);
                // The badge's words and mark as chat.js setStatus writes roomStatusView's answer (a connected,
                // ready host); room_work.js draws the control and the list around it.
                const view = roomStatusView(workState, { connected: true, ready: true });
                badge.className = `status-badge ${view.kind}`;
                badge.textContent = view.text;
                Object.assign(badge.dataset, { tone: view.tone, mark: view.mark, owner: view.owner ? '1' : '',
                    amber: view.amber, step: view.step || '' });
                roomWork.update();
                await settle();
                // The list is the subject: open it with its own control, as the owner does.
                const trigger = page.querySelector('.rw-trigger');
                if (k === bodies.length - 1 && trigger && !trigger.hidden && trigger.getAttribute('aria-expanded') !== 'true') {
                    trigger.click();
                    await settle();
                    // A scripted click reads as a keyboard visit and rings the row it focuses; a pointer open does not.
                    if (page.contains(document.activeElement)) document.activeElement.blur();
                }
            },
        };
    },
    async consciousness(stage) {
        const { initConsciousnessDashboard } = await import('/static/modules/consciousness_dashboard.js');
        const mount = element('div', { class: 'gallery-tab', 'data-gallery-subject': '' });
        stage.append(mount);
        const tab = initConsciousnessDashboard({ mount, ws: quietWs(), openSettings() {}, openDashboardTab() {},
            state: { activePage: 'dashboard', dashboardActiveSubtab: 'consciousness' } });
        return { frames: 1, async apply() { await tab.refresh(); } };
    },
    async widgets(stage, entry, state) {
        const { initWidgets } = await import('/static/modules/widgets.js');
        const { widgetKey } = await import('/static/modules/widget_list.js');
        const tabs = entry.responses?.[0]?.ui_tabs;
        if (!Array.isArray(tabs)) throw new Error(`the card corpus records no widget list for widgets.${state}`);
        const keys = tabs.map(widgetKey);
        stage.id = 'content';
        initWidgets({ ws: quietWs() });
        const page = stage.querySelector('#page-widgets');
        page.classList.add('active');
        page.dataset.gallerySubject = '';
        const list = page.querySelector('#widgets-list');
        const failure = page.querySelector('#widgets-list-error');
        const failed = () => (failure.querySelector('[data-widget-list-error]') || failure).textContent.trim();
        return {
            frames: 1,
            async apply() {
                window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'widgets' } }));
                const loading = list.innerHTML;  // what the page shows while its list is read
                // Its recorded list drawn: those cards, each with its body mounted, or the page's own empty state, never
                // the loading face; a read the page says failed fails the scene, in the page's words.
                const what = keys.length ? `its ${keys.length} recorded card(s)` : 'its empty state';
                await until(() => {
                    if (!failure.hidden) throw new Error(`the Widgets page says: ${failed()}`);
                    const cards = [...list.querySelectorAll('[data-widget-key]:not([data-widget-removed])')];
                    if (!keys.length) return !cards.length && list.innerHTML !== loading;
                    const drawn = new Set(cards.map((card) => card.dataset.widgetKey));
                    return cards.length === keys.length && keys.every((key) => drawn.has(key))
                        && cards.every((card) => card.querySelector('[data-widget-mount]')?.childElementCount > 0);
                }, `the Widgets page to draw ${what}`);
            },
        };
    },
    async skills(stage) {
        const { initSkills } = await import('/static/modules/skills.js');
        stage.id = 'content';
        initSkills({ ws: quietWs(), state: { activePage: 'skills' } });
        const page = stage.querySelector('#page-skills');
        page.classList.add('active');
        page.dataset.gallerySubject = '';
        return {
            frames: 1,
            async apply() {
                window.dispatchEvent(new CustomEvent('ouro:page-shown', { detail: { page: 'skills' } }));
                await settle();
                // The installed list is the subject: open it with its own row, as the owner does.
                const installed = await until(() => page.querySelector('[data-extensions-open="installed"]'),
                    'the Skills page\'s Installed row');
                installed.click();
                const list = page.querySelector('#skills-list');
                const empty = page.querySelector('#skills-empty');
                const status = page.querySelector('#skills-status');
                await until(() => {
                    if (status?.dataset.tone === 'danger' && status.textContent.trim()) {
                        throw new Error(`the Skills page says: ${status.textContent.trim()}`);
                    }
                    return list?.childElementCount || empty?.hidden === false;
                }, 'the Skills page to draw its installed list');
            },
        };
    },
    async login(stage, entry, state) {
        const { createLoginCardController, loginVerdict } = await import('/static/modules/harness_login_cards.js');
        const created = entry.responses?.[0];
        const harness = String(created?.job?.harness || '');
        if (!harness) throw new Error(`the card corpus records no sign-in job for login.${state}`);
        // The controller's own transport seam answers its start with the recorded job, so nothing is posted and a stand
        // that answers every write itself (the ui_look tool's does) still shows this job. Every other request, its job
        // poll included, goes to the stand as the page sends it.
        const recorded = () => new Response(JSON.stringify(created), { headers: { 'Content-Type': 'application/json' } });
        const fetchImpl = async (url, init = {}) => (url === '/api/claudexor/login' && /^post$/i.test(init.method || '')
            ? recorded() : apiFetch(url, init));
        const host = element('div', { class: 'gallery-login', 'data-gallery-subject': '' });
        stage.append(host);
        const controller = createLoginCardController({ host, store: null, fetchImpl });
        // The face of the recorded job: its state line while it waits, its verdict once it ended. Any other (the error
        // face of a start the card refused) is not this scene.
        const face = loginVerdict(created).kind === 'pending' ? '[data-login-state]' : '[data-login-verdict]';
        return {
            frames: 1,
            async apply() {
                await controller.start(harness, '');
                if (!host.querySelector(face)) {
                    const shown = host.innerText.trim().replace(/\s+/g, ' ');
                    throw new Error(`the sign-in card does not show the recorded job: ${shown}`);
                }
            },
        };
    },
    async task_card(stage, entry, state) {
        const room = await mainRoom(stage, entry);
        room.page.dataset.gallerySubject = '';
        const frames = entry.frames || [];
        return {
            frames: 1 + frames.length,
            async apply(k) {
                if (k === 0) {
                    await room.open();
                    // A settled card's Details is the subject of those states: opened with its own control.
                    if (['done', 'failed'].includes(state)) {
                        await until(() => room.card()?.dataset.finished === '1', 'the task card to settle');
                        const details = await until(() => room.card()?.querySelector('[data-live-disclosure]'),
                            'the card\'s Details control');
                        details.click();
                        await until(() => room.card()?.dataset.expanded === '1', 'the card\'s Details to open');
                        if (room.page.contains(document.activeElement)) document.activeElement.blur();
                    }
                } else room.deliver(frames[k - 1]);
                await settle();
            },
        };
    },
    async share(stage, entry) {
        // Share as app.js registers it, on the finished task's answer in Main: its own action, then Card.
        const { registerMessageShare } = await import('/static/modules/message_share.js');
        registerMessageShare();
        const room = await mainRoom(stage, entry);
        return {
            frames: 1,
            async apply() {
                await room.open();
                const action = () => [...room.page.querySelectorAll('.chat-bubble.assistant [data-message-action="share"]')].at(-1);
                (await until(action, 'the answer\'s Share action')).click();
                (await until(() => document.querySelector('.message-share [data-share-format="card"]'), 'Share\'s Card')).click();
                // The picture loaded, or the Card's own note saying why it will not draw one (its preview's alert).
                await until(() => {
                    const picture = document.querySelector('.message-share-card');
                    return (picture?.complete && picture.naturalWidth > 0)
                        || document.querySelector('.message-share .document-reader-status.is-error');
                }, 'the Card drawn or refused');
                document.querySelector('.message-share')?.setAttribute('data-gallery-subject', '');
            },
        };
    },
    async connection(stage, entry) {
        const { createStarterSkillsStep } = await import('/static/modules/onboarding_starter_skills.js');
        const { checkIdentity } = await import('/static/modules/onboarding_connection_card.js');
        // The step's own seam for its card: the host's recorded identity answer reaches checkIdentity as if the
        // request had come back (nothing is posted), and one pairing code every time, so the QR reads alike.
        const { status = 200, body = null } = entry.identity || {};
        const check = (skill, token) => checkIdentity(skill, token, async () => new Response(JSON.stringify(body),
            { status, headers: { 'Content-Type': 'application/json' } }));
        const step = createStarterSkillsStep({ card: { check, random: (bytes) => bytes.fill(7) } });
        const page = element('div', { class: 'gallery-setup', 'data-gallery-subject': '' });
        page.innerHTML = step.html();
        stage.append(page);
        return {
            frames: 1,
            async apply() {
                step.mount(page.querySelector('[data-starter-skills="step"]'));
                await step.loading;
                // Connect, as the owner presses it; the token typed, then the field left: the card checks it.
                (await until(() => page.querySelector('[data-connection-toggle]'), 'the card\'s Connect')).click();
                const field = await until(() => page.querySelector('[data-connection-token]'), 'the card\'s token field');
                field.value = entry.token || '';
                field.dispatchEvent(new Event('change', { bubbles: true }));
                await until(() => page.querySelector('[data-connection-status], .skills-connection-qr svg'),
                    'the card\'s answer to the token');
                if (page.contains(document.activeElement)) document.activeElement.blur();
            },
        };
    },
    async provider_card(stage, entry) {
        const { PROVIDER_CARDS, providerSettingsCard } = await import('/static/modules/settings_provider_cards.js');
        const { createConnectionsEditor } = await import('/static/modules/settings_connections.js');
        const { onboardingSettingsDraft } = await import('/static/modules/onboarding_agents_step.js');
        const { bindSecretInputs } = await import('/static/modules/settings_secrets.js');
        const bootstrap = entry.responses?.[0] || {};
        const state = bootstrap.initialState || {};
        const contract = bootstrap.contract || {};
        const compatible = PROVIDER_CARDS.find((card) => card.kind === 'openai-compatible');
        // The setup wizard's providers panel and its one provider card (onboarding_wizard.js renderProvidersStep). The
        // wizard keeps the wiring below to itself (its lines after createConnectionsEditor): the same calls, in its order.
        const root = element('div', { class: 'gallery-setup', 'data-gallery-subject': '' });
        root.innerHTML = `<div data-settings-panel="providers">${providerSettingsCard({ ...compatible,
            title: 'Custom OpenAI-compatible servers',
            open: Boolean(state.compatibleBaseUrl || state.legacyCompatibleBaseUrl || state.connections?.length) })}</div>`;
        stage.append(root);
        return {
            frames: 1,
            apply() {
                const panel = root.querySelector('[data-settings-panel="providers"]');
                const fields = contract.providerFields || [];
                const draft = onboardingSettingsDraft({ state, providerFields: fields,
                    budgetFields: contract.budgetFields || [], modelSlots: contract.modelSlots || [] });
                const editor = createConnectionsEditor({ root });
                for (const field of compatible.fields) {  // the built-in key's fields: the draft, and what is saved
                    const input = panel.querySelector(`#${field.id}`);
                    input.value = draft[field.settingKey] || '';
                    input.dataset.appliedValue = state[fields.find((each) => each.settingKey === field.settingKey)?.stateKey] || '';
                }
                bindSecretInputs(panel);
                editor.load({ ...draft, _meta: { policy_state: { deployment: bootstrap.deployment } } },
                    { modelSlots: contract.modelSlots || [] });
            },
        };
    },
    async agents(stage) {
        // The column Settings opens a page in (section_levels.js pageElement), and the mount it calls.
        const { settingsPageFor } = await import('/static/modules/settings_pages.js');
        await import('/static/modules/subagents_settings.js');
        const { readSettings } = await import('/static/modules/settings_controls.js');
        const page = element('section', { class: 'settings-page gallery-settings', 'data-settings-page': 'agents',
            'aria-label': 'Agents', 'data-gallery-subject': '' });
        page.innerHTML = '<div class="settings-column"><div class="settings-page-body"></div></div>';
        stage.append(page);
        const ctx = { readSettings, openSettings() {}, openDashboardTab() {}, showPage() {}, layers: null, ws: quietWs(),
            state: { activePage: 'settings', settingsPath: 'agents' }, onScreen: () => true };
        return {
            frames: 1,
            async apply() {
                await settingsPageFor('agents').mount(page.querySelector('.settings-page-body'), { path: 'agents', row: '', view: '', ctx });
                // The roster's switch is operable once the settings answer is adopted; the pool read it starts then is
                // one of the frame's reads, answered before the scene is ready.
                await until(() => page.querySelector('[data-setting-row="OUROBOROS_SUBAGENTS"] [data-setting-control]')?.disabled === false,
                    'the roster to adopt the recorded settings');
            },
        };
    },
};

// Main as app.js creates it (its own page, feed and composer) over a task's recorded reads. Its socket handlers stay
// here, so a recorded frame reaches the handler the live socket would call; the census is read and handed over as
// app.js hands it.
async function mainRoom(stage, entry) {
    const { createChatInstance } = await import('/static/modules/chat.js');
    const handlers = new Map();
    const ws = { ws: { readyState: 1 }, isConnected: () => true,
        on(type, fn) { handlers.set(type, [...(handlers.get(type) || []), fn]); return () => {}; },
        send(frame) { return { status: 'sent', clientMessageId: frame.client_message_id || 'gallery' }; } };
    let chat = null;
    const snapshots = { begin: () => ({ generation: 1, requestedAt: Date.now() }), isCurrent: () => true,
        gate() { return Promise.resolve(this.begin()); }, apply(_request, data) { chat?.hydrateStateSnapshot(data); } };
    stage.id = 'content';
    chat = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, openSettings() {}, openDashboardTab() {}, chatId: 1, mountEl: stage, stateSnapshots: snapshots });
    const page = stage.querySelector('#page-chat');
    page.classList.add('gallery-room');
    const card = () => page.querySelector(`.chat-live-card[data-task-id="${entry.task_id}"]`);
    return {
        page, card,
        async open() {
            await until(() => chat.hasPaintedHistory(), 'Main to paint its history');
            const census = await apiFetch('/api/state', { cache: 'no-store' });
            if (!census.ok) throw new Error(`GET /api/state answered ${census.status}`);
            chat.hydrateStateSnapshot(await census.json());
            await until(() => card(), `the card of ${entry.task_id}`);
        },
        deliver(frame) { (handlers.get(frame.type) || []).forEach((fn) => fn(frame)); },
    };
}

// ---- boot

async function boot() {
    const pick = selection();
    document.documentElement.dataset.theme = pick.theme;
    if (params.get('bare') === '1') {
        window.gallery = { ready: true, bare: true, scenes: sceneNames };
        return;
    }
    window.gallery = { ready: false, scene: pick.scene, scenes: sceneNames };
    const bar = chrome(pick);
    if (pick.faults.length) throw new Error(pick.faults.join('; '));
    const { family, state } = pick;
    signReads(pick.scene);
    const corpora = Object.fromEntries(await Promise.all(['card_gallery', 'work_projection_cases', 'consciousness_contract']
        .map(async (name) => [name, await corpus(name)])));
    const entry = family.corpus.startsWith('card_gallery.') ? corpora.card_gallery[family.id]?.[state] : {};
    if (!entry || typeof entry !== 'object') throw new Error(`the card corpus records no ${family.corpus}.${state}`);
    const stage = element('main', { class: 'gallery-stage', 'data-gallery-stage': '', 'data-scene': pick.scene });
    root.append(stage);
    const scene = await SCENES[family.id](stage, entry, state, corpora);
    const upTo = pick.frames === null ? scene.frames : Math.max(1, Math.min(pick.frames, scene.frames));
    let shown = 0;
    const paintCounter = () => {
        bar.counter.textContent = scene.frames > 1 ? `Frame ${shown} of ${scene.frames}` : '';
        bar.next.textContent = shown < scene.frames ? 'Next frame' : 'Replay';
        bar.next.hidden = scene.frames < 2;
    };
    // One frame: its reads name it, and it counts once they are answered and drawn.
    const step = async () => {
        reads.frame = shown;
        await scene.apply(shown);
        await answered();
        shown += 1;
        paintCounter();
    };
    const drawn = () => {
        window.gallery = { ready: true, scene: pick.scene, frames: scene.frames, shown: () => shown, scenes: sceneNames };
        bar.mark(true);
        bar.next.disabled = false;
    };
    bar.next.addEventListener('click', () => {
        if (shown >= scene.frames) {
            location.search = `?scene=${pick.scene}&theme=${pick.theme}&frames=1`;
            return;
        }
        window.gallery.ready = false;
        bar.mark(false);
        bar.next.disabled = true;
        step().then(drawn, (error) => { fault(error); throw error; });
    });
    while (shown < upTo) await step();
    paintCounter();
    drawn();
}

boot().catch((error) => {
    fault(error);
    throw error;
});
