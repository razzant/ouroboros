"""Focused shell integration against real documents/modules and synthetic API data.

These tests cover geometry and interactions, not native host bridges or backend work.
The existing subscription fixture owns the static server, browser and cleanup.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def open_app(ui, path="/"):
    page = ui["page"]
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback()
               if urlparse(route.request.url).netloc == origin else route.abort())
    page.goto(ui["url"] + path)
    page.wait_for_selector("#chat-input", state="attached")
    return page


def hit_box(locator):
    return locator.evaluate("""el => {
        const r = el.getBoundingClientRect();
        const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
        return {x:r.x,y:r.y,width:r.width,height:r.height,right:r.right,bottom:r.bottom,
            reachable: !!hit && (hit === el || el.contains(hit)), hit:hit?.id || hit?.className};
    }""")


@pytest.mark.parametrize("width", [390, 1440])
def test_chat_header_decoration_does_not_clip_menu_and_system_actions_keep_gap(subscription_ui, width):
    ui = subscription_ui
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": 600})
    rows = [{"role": "system", "text": text, "markdown": markdown,
             "system_type": "project_completion_summary", "project_id": "fixture-project",
             "project_name": "Fixture project", "ts": f"2026-09-09T00:00:0{i}Z"}
            for i, (text, markdown) in enumerate([
                ("Plain completion text\nSecond line.", False),
                ("**Markdown completion**\n\nA second paragraph.", True)])]
    rows.extend({"role": "assistant", "text": f"Reading line {i}. " + "Useful content continues below the header. " * 6,
                 "ts": f"2026-09-09T00:01:{i:02}Z"} for i in range(12))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps({"messages": rows, "progress": []})))
    open_app(ui)
    page.wait_for_selector(".system-message-actions")
    actions = page.locator("#chat-messages .system-message-actions")
    assert actions.count() == 2
    for action in actions.all():
        metrics = action.evaluate("""el => {
            const prose = el.previousElementSibling;
            return {previous:prose.className, nested:!!el.closest('.message'),
                gap:el.querySelector('button').getBoundingClientRect().top - prose.getBoundingClientRect().bottom,
                below:el.getBoundingClientRect().bottom - el.querySelector('button').getBoundingClientRect().bottom};
        }""")
        assert metrics["previous"] == "message" and not metrics["nested"]
        assert metrics["gap"] == pytest.approx(12, abs=0.5), metrics
        assert metrics["below"] == pytest.approx(12, abs=0.5), metrics
    header = page.locator(".chat-page-header")
    paint = header.evaluate("""el => ({mask:getComputedStyle(el).maskImage,
        overflow:getComputedStyle(el).overflow, fade:getComputedStyle(el,'::before').maskImage,
        blur:getComputedStyle(el,'::before').backdropFilter,
        pointer:getComputedStyle(el,'::before').pointerEvents})""")
    assert paint["mask"] == "none" and paint["overflow"] == "visible", paint
    assert "gradient" in paint["fade"] and "blur" in paint["blur"], paint
    assert paint["pointer"] == "none"
    page.locator("#chat-messages").evaluate("el=>el.scrollTop=0")
    setup_browser.capture(page, f"shell-chat-actions-{width}")
    page.locator("#chat-messages").evaluate("el=>el.scrollTop=160")
    page.locator(".chat-header-more > summary").click()
    for item in page.locator(".chat-header-menu-item").all():
        box = hit_box(item)
        assert box["reachable"] and box["bottom"] <= 600 and box["right"] <= width, box
    setup_browser.capture(page, f"shell-chat-menu-{width}")
    page.locator(".chat-header-more > summary").click()


def test_scroll_fade_tracks_edges_late_content_and_stops_after_disposal(subscription_ui):
    page = open_app(subscription_ui)
    # Mount only a bounded scroll region; its observer and mask are the production ones.
    page.evaluate("""async () => {
        const {bindScrollFade} = await import('/static/modules/scroll_fade.js');
        const el = document.createElement('div');
        el.id='shell-fade-probe'; el.className='scroll-fade-y';
        el.style.cssText='position:fixed;z-index:100;left:20px;top:160px;width:300px;height:100px;overflow:auto;background:#17131a';
        const child=document.createElement('div'); child.textContent='First visible line';
        el.append(child); document.body.append(el);
        window.shellFade={el,child,dispose:bindScrollFade(el)};
    }""")
    state = """() => {const e=shellFade.el;return [e.hasAttribute('data-scroll-above'),e.hasAttribute('data-scroll-below')]}"""
    page.wait_for_function("""() => getComputedStyle(shellFade.el).getPropertyValue('--scroll-fade-top').trim()==='0px'""")
    assert page.evaluate(state) == [False, False]
    page.evaluate("shellFade.child.style.height='500px'")
    page.wait_for_function("shellFade.el.hasAttribute('data-scroll-below')")
    assert page.evaluate(state) == [False, True]
    page.evaluate("shellFade.el.scrollTop=200")
    page.wait_for_function("shellFade.el.hasAttribute('data-scroll-above')")
    assert page.evaluate(state) == [True, True]
    setup_browser.capture(page, "shell-scroll-middle")
    page.evaluate("shellFade.el.scrollTop=shellFade.el.scrollHeight")
    page.wait_for_function("!shellFade.el.hasAttribute('data-scroll-below')")
    assert page.evaluate(state) == [True, False]
    page.evaluate("shellFade.el.replaceChildren(document.createTextNode('New short content'))")
    page.wait_for_function("!shellFade.el.hasAttribute('data-scroll-above')")
    assert page.evaluate(state) == [False, False]
    page.evaluate("""() => {shellFade.dispose();shellFade.el.append(shellFade.child);
        shellFade.child.style.height='1000px';shellFade.el.scrollTop=100;
        shellFade.el.dispatchEvent(new Event('scroll'));}""")
    page.wait_for_timeout(100)  # beyond the observer/rAF turns that would repaint without disposal
    assert page.evaluate(state) == [False, False]
    page.evaluate("shellFade.el.remove()")


def test_matrix_respects_reduced_motion_changes_and_page_cleanup(subscription_ui):
    page = subscription_ui["page"]
    page.emulate_media(reduced_motion="reduce")
    open_app(subscription_ui)
    page.wait_for_selector("#matrix-rain", state="attached")
    sample = "document.querySelector('#matrix-rain').toDataURL()"
    first = page.evaluate(sample)
    page.wait_for_timeout(200)
    assert page.evaluate(sample) == first
    setup_browser.capture(page, "shell-rain-reduced")
    page.emulate_media(reduced_motion="no-preference")
    page.wait_for_function("before => document.querySelector('#matrix-rain').toDataURL() !== before", arg=first)
    page.emulate_media(reduced_motion="reduce")
    page.wait_for_timeout(100)
    frozen = page.evaluate(sample)
    page.wait_for_timeout(200)
    assert page.evaluate(sample) == frozen
    page.evaluate("""() => {window.shellRain=document.querySelector('#matrix-rain');
        window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:false}));}""")
    assert page.locator("#matrix-rain").count() == 0
    before = page.evaluate("[shellRain.width,shellRain.height,shellRain.toDataURL()]")
    page.set_viewport_size({"width": 700, "height": 400})
    page.emulate_media(reduced_motion="no-preference")
    page.wait_for_timeout(200)
    assert page.evaluate("[shellRain.width,shellRain.height,shellRain.toDataURL()]") == before


@pytest.mark.parametrize("height", [260, 600])
def test_photo_menu_escapes_gallery_crop_downloads_and_cleans_up(subscription_ui, height):
    page = subscription_ui["page"]
    page.set_viewport_size({"width": 390, "height": height})
    page.route("**/api/chat/history*", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
        "messages": [{"role": "assistant", "text": "Photo fixture is ready", "ts": "2026-09-09T00:00:00Z"}]})))
    open_app(subscription_ui)
    page.get_by_text("Photo fixture is ready", exact=True).wait_for()
    # Actual media renderer, gallery grouping, menu and CSS. Callbacks only bind it
    # to this document's real message region; no copied photo/menu markup.
    page.evaluate("""async () => {
        const {createChatMedia}=await import('/static/modules/chat_media.js');
        const feed=document.querySelector('#chat-messages');
        const canvas=document.createElement('canvas');canvas.width=32;canvas.height=12;
        const c=canvas.getContext('2d');c.fillStyle='#7ba1c4';c.fillRect(0,0,32,12);
        const image=canvas.toDataURL('image/png').split(',')[1];
        const media=createChatMedia({chatSessionId:'shell-proof',durableChatMediaUrl:()=>'',
            formatMsgTime:()=>null,senderLabel:()=> 'Fixture',stampNodeTimestamp:()=>{},
            insertMessageNode:node=>feed.append(node)});
        window.shellPhoto={media,add:()=>{for(let i=0;i<2;i++){
            const msg={type:'photo',task_id:'shell-gallery',role:'assistant',image_base64:image,mime:'image/png'};
            media.buildGallery('photos',msg,media.buildMediaBubble(msg));
        }}};
        shellPhoto.add();
    }""")
    trigger = page.locator(".chat-photo-actions summary").first
    trigger.scroll_into_view_if_needed()
    page.evaluate("() => new Promise(r => requestAnimationFrame(()=>requestAnimationFrame(r)))")
    page.evaluate("""() => {
        window.shellPhotoEvents=[];
        for (const type of ['scroll','focusin','click']) window.addEventListener(type,e=>
            shellPhotoEvents.push({type,target:e.target?.id||e.target?.className||e.target?.tagName,
                scroll:document.querySelector('#chat-messages').scrollTop,
                active:document.activeElement?.outerHTML.slice(0,100),time:performance.now()}),true);
    }""")
    trigger.click()
    menu = page.locator('body > .chat-photo-menu[role="menu"]')
    setup_browser.capture(page, f"shell-photo-open-{height}")
    assert menu.is_visible(), json.dumps(page.evaluate("shellPhotoEvents"))
    assert menu.evaluate("el=>el.closest('.chat-gallery-item')===null")
    assert page.locator(".chat-gallery-item").first.evaluate("el=>getComputedStyle(el).overflow") == "hidden"
    for item in menu.locator('[role="menuitem"]').all():
        item.scroll_into_view_if_needed()
        box = hit_box(item)
        assert box["reachable"] and box["x"] >= 0 and box["right"] <= 390 and box["bottom"] <= height, box
    page.keyboard.press("End")
    assert menu.locator('[data-photo-action="copy"]').evaluate("el=>document.activeElement===el")
    setup_browser.capture(page, f"shell-photo-menu-{height}")
    page.keyboard.press("Escape")
    assert not menu.count()
    assert trigger.evaluate("el=>document.activeElement===el")
    trigger.click()
    with page.expect_download() as download:
        page.locator('body > .chat-photo-menu [data-photo-action="download"]').click()
    assert download.value.suggested_filename == "image.png"
    assert download.value.failure() is None
    assert not menu.count()
    trigger.click()
    assert menu.is_visible()
    page.evaluate("shellPhoto.media.reset()")
    assert not menu.count() and page.locator(".chat-gallery-item").count() == 0
    page.evaluate("shellPhoto.add()")
    page.locator(".chat-photo-actions summary").first.click()
    assert menu.is_visible()
    page.evaluate("shellPhoto.media.destroy()")
    assert not menu.count() and page.locator(".chat-gallery-item").count() == 0


def test_settings_footer_controls_are_reachable_at_content_breakpoints(subscription_ui):
    page = open_app(subscription_ui, "/#settings")
    page.wait_for_selector("#btn-save-settings")
    # Exercise the existing footer's geometry in each meaningful controller state.
    # Domain save/validation transitions belong to the forms track's separate flows.
    states = [
        ("clean", "", False, False), ("dirty", "Unsaved changes", False, False),
        ("saving", "Saving…", True, False),
        ("error", "Error: Could not save your settings. Your edits are still available.", False, False),
        ("restart", "Saved. Restart required before the new runtime settings take effect.", False, True),
    ]
    for width in [640, 641, 760, 768, 980, 981]:
        page.set_viewport_size({"width": width, "height": 600})
        if width <= 640:
            page.wait_for_function("document.querySelector('#primary-sidebar').getBoundingClientRect().right<=1")
        for sidebar in [220, 280]:
            page.evaluate("value => document.documentElement.style.setProperty('--sidebar-width',value+'px')", sidebar)
            for name, message, busy, restart in states:
                page.evaluate("""s => {
                    document.querySelector('#settings-status').textContent=s.message;
                    document.querySelector('#settings-unsaved-indicator').classList.toggle('is-visible',s.name==='dirty');
                    document.querySelector('#btn-save-settings').disabled=s.busy;
                    document.querySelector('#btn-restart-now').hidden=!s.restart;
                }""", {"name": name, "message": message, "busy": busy, "restart": restart})
                for selector in ["#btn-reload-settings", "#btn-save-settings"] + (["#btn-restart-now"] if restart else []):
                    box = hit_box(page.locator(selector))
                    valid = box["reachable"] and box["x"] >= 0 and box["right"] <= width + 1 and box["bottom"] <= 600
                    if not valid:
                        setup_browser.capture(page, f"shell-footer-failure-{width}-{sidebar}-{name}")
                    assert valid, json.dumps([width, sidebar, name, selector, box])
                if sidebar == 280 and name in {"dirty", "restart"}:
                    setup_browser.capture(page, f"shell-footer-{width}-{name}")


def test_short_sidebar_bounds_projects_list_and_cost_cards_keep_local_table_overflow(subscription_ui):
    ui = subscription_ui
    page = ui["page"]
    projects = [{"id": f"project-{i}", "name": f"Project {i} with a descriptive title", "chat_id": 100+i,
                 "lifecycle": "active", "visible_revision": 0} for i in range(12)]
    page.route("**/api/projects", lambda route: route.fulfill(content_type="application/json", body=json.dumps({"projects": projects})))
    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
        "supervisor_ready": True, "active_chat_activities": [], "projects": projects,
        "project_chat_ids": [p["chat_id"] for p in projects]})))
    data = {"total_cost": 27.4, "total_calls": 42, "accounting": {"available": True,
            "accounted_usd": 27.4, "confirmed_usd": 25, "reserved_usd": 2,
            "unresolved_upper_bound_usd": 0.4, "unknown_unmetered": 1,
            "limit_usd": 200, "cost_final": False},
            "by_model": {"provider/a-long-model-name-for-real-table-overflow": {"calls": 42, "cost": 27.4}}}
    page.route("**/api/cost-breakdown", lambda route: route.fulfill(content_type="application/json", body=json.dumps(data)))
    page.set_viewport_size({"width": 844, "height": 320})
    open_app(ui)
    page.wait_for_selector(".nav-project-row")
    sidebar = page.locator("#primary-sidebar")
    scrolling = sidebar.evaluate("""el => [...el.querySelectorAll('*')].filter(e =>
        ['auto','scroll'].includes(getComputedStyle(e).overflowY) && e.scrollHeight>e.clientHeight+1).map(e=>e.className)""")
    # The projects list carries its own bounded window, so the column itself
    # stays the same height whatever the project count.
    assert set(scrolling) == {"sidebar-scroll", "nav-projects-list"}, scrolling
    nav = page.locator('[data-nav-page="settings"]')
    nav.scroll_into_view_if_needed()
    assert hit_box(nav)["reachable"]
    setup_browser.capture(page, "shell-sidebar-short-bottom")
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="costs"]').click()
    page.wait_for_function("document.querySelector('#cost-confirmed').textContent==='$25.00'")
    for width in [390, 641, 768, 981]:
        page.set_viewport_size({"width": width, "height": 600})
        if width <= 640:
            page.wait_for_function("document.querySelector('#primary-sidebar').getBoundingClientRect().right<=1")
        for card in page.locator(".costs-stats-grid > .stat-card").all():
            card.scroll_into_view_if_needed()
            box = hit_box(card)
            valid = box["reachable"] and box["x"] >= 0 and box["right"] <= width + 1
            if not valid:
                setup_browser.capture(page, f"shell-cost-failure-{width}")
            assert valid, json.dumps([width, box])
        for table in page.locator(".costs-tables-grid > div").all():
            assert table.evaluate("el=>getComputedStyle(el).overflowX") == "auto"
            assert table.evaluate("el=>el.getBoundingClientRect().right<=innerWidth+1")
        if width == 390:
            first = page.locator("#cost-by-model").locator("..")
            assert first.evaluate("el => el.scrollWidth > el.clientWidth"), "long table should scroll locally"
            first.evaluate("el=>el.scrollLeft=el.scrollWidth")
            assert first.evaluate("el=>el.scrollLeft>0")
        page.locator(".costs-stats-grid").scroll_into_view_if_needed()
        setup_browser.capture(page, f"shell-costs-{width}")


def test_mcp_transport_fields_use_named_shared_controls_without_changing_drafts(subscription_ui):
    page = open_app(subscription_ui, '/#settings')
    page.locator('[data-settings-tab="advanced"]').click()
    servers = [dict(id='web', name='HTTP server', transport='streamable_http', url='https://example.test/mcp',
                    auth_token='***', auth_header='Authorization', custom_option='retained'),
               dict(id='local', name='Local server', transport='stdio', command='npx', args=['-y','server'],
                    env={'PORT':'8080'}, env_from_settings={'TOKEN':'TOKEN_KEY'})]
    page.evaluate('''async servers=>{
        const mcp=await import('/static/modules/mcp_settings.js');
        mcp.applyMcpSettings({MCP_ENABLED:true,MCP_SERVERS:servers,MCP_TOOL_TIMEOUT_SEC:60});
        window.mcpProof=mcp;
    }''', servers)
    cards=page.locator('[data-mcp-card]')
    assert cards.count()==2
    for control in cards.locator('[data-mcp-field]:not([type="checkbox"])').all():
        assert 'ui-control' in control.get_attribute('class')
        assert control.get_attribute('aria-label').startswith('MCP server ')
        assert control.evaluate('el=>Array.from(el.labels||[]).length') == 1
    page.get_by_label('MCP server 1: Server URL', exact=True).fill('https://edited.test/mcp')
    draft=page.evaluate('mcpProof.collectMcpSettings()')
    assert draft['MCP_SERVERS'][0]['url']=='https://edited.test/mcp'
    assert draft['MCP_SERVERS'][0]['auth_token']=='***'
    assert draft['MCP_SERVERS'][0]['custom_option']=='retained'
    assert draft['MCP_SERVERS'][1]['args']==['-y','server']
    assert draft['MCP_SERVERS'][1]['env_from_settings']=={'TOKEN':'TOKEN_KEY'}
    assert page.evaluate('mcpProof.validateMcpSettings().length') == 0
    field = page.get_by_label('MCP server 2: Environment (JSON, optional)', exact=True)
    for invalid in ['{unfinished', '[]', '5', '{"PORT":8080}', '{"":"value"}']:
        field.fill(invalid)
        assert page.evaluate('mcpProof.validateMcpSettings().map(e=>e.input.dataset.mcpField)') == ['env']
        assert field.input_value() == invalid
    field.fill('null')
    assert page.evaluate('mcpProof.validateMcpSettings().length') == 0  # backend accepts omitted maps
    field.fill('{"PORT":"8080"}')
    assert page.evaluate('mcpProof.validateMcpSettings().length') == 0
    capture = setup_browser.capture
    cards.first.scroll_into_view_if_needed()
    capture(page,'shell-mcp-shared-fields')


# One structural reading of a quiz card for the "same form" comparison: every element's class,
# its own text and its disabled state, in document order; the Main-only Project chip is skipped.
CARD_FORM_JS = """card => { const out = []; const walk = (el) => {
    if (el.classList.contains('chat-quiz-project')) return;
    out.push([[...el.classList].filter(name => name !== 'project-question-card').join(' '),
        el.children.length ? '' : el.textContent.trim(), el.disabled ?? null, el.getAttribute('placeholder')]);
    [...el.children].forEach(walk); }; walk(card); return out; }"""


@pytest.mark.parametrize('width,height', [(1100, 722), (390, 844)])
def test_question_mirrors_full_form_settle_and_reload(subscription_ui, width, height):
    """Real SPA readers and WebSocket handlers; disposable source data, no owner writes.

    A realistic burst of one task's Project questions in Main: every unanswered, passed, finished
    or replaced question is the Project's own form plus one Project chip; an answered one never
    enters Main. The first confirmed answer from any source — a press in Main, the Project form
    over quiz_state, a keyboard press — shows the result for five seconds and removes only the
    Main copy, keeping the reading position. A copy that learns its form together with its answer
    gets the same five seconds; stale snapshots and a reconnect never bring a removed copy back,
    also after Main's bounded question memory forgot it (one canonical task-detail read decides)."""
    import copy
    from urllib.parse import parse_qs

    ui = subscription_ui
    page = ui['page']
    page.set_viewport_size({'width': width, 'height': height})
    project = {'id': 'question-proof', 'name': 'Evidence project with a deliberately long descriptive name',
               'chat_id': 42, 'lifecycle': 'active', 'visible_revision': 0}
    details = ['Preserve the full original measurements.', 'Compare the independent run.']
    blocks = {
        'exact-question': {'state': 'answered', 'question': 'Which evidence should we retain?', 'answered_index': 0,
            'options': ['Keep the primary source', 'Use the replication'], 'option_details': details,
            'wait_for_answer': True, 'comment': 'Retain the provenance.', 'asked_at': '2026-09-16T00:00:00Z'},
        'second': {'state': 'open', 'question': 'Second of three: publish the interim table as well?',
            'options': ['Publish it now', 'Hold it until the replication lands'], 'option_details': ['', ''],
            'wait_for_answer': True, 'asked_at': '2026-09-16T00:01:00Z'},
        'passed': {'state': 'open', 'question': 'Which figure format keeps the appendix small?', 'options': ['PNG', 'WebP'],
            'option_details': ['Lossless, larger', 'Smaller at quality 82'], 'assumption': 'WebP at quality 82',
            'recommended_index': 1, 'asked_at': '2026-09-16T00:02:00Z'},
        'finished': {'state': 'expired_terminal', 'question': 'Should the archive keep the raw instrument logs?',
            'options': ['Keep them', 'Drop them'], 'option_details': ['', ''], 'wait_for_answer': True,
            'asked_at': '2026-09-16T00:02:30Z'},
        'replaced': {'state': 'superseded', 'question': 'Which appendix order reads best?', 'options': ['By date', 'By topic'],
            'option_details': ['', ''], 'assumption': 'By date', 'asked_at': '2026-09-16T00:02:45Z'},
        'waiting': {'state': 'open', 'wait_for_answer': True, 'recommended_index': 0, 'asked_at': '2026-09-16T00:03:00Z',
            'question': 'Third of three. The **licence** of the external dataset forbids redistribution, so the archive '
                        'can either ship without it and link to the source, or wait for written permission, which the '
                        'maintainers usually grant within a week. Which way do we go?',
            'options': ['Ship without it and link the source', 'Wait for written permission'],
            'option_details': ['Readers follow one extra link.', 'Publication slips by about a week.'],
            'stake': 'Whether the archive ships this week.'},
    }
    asked = copy.deepcopy(blocks)
    wait = {'quiz_id': 'waiting', 'state': 'waiting'}
    decisions, activities, history_reads, sockets, detail_reads = [], [], [], [], []
    mode = {'stale': False}
    # A second task of the same Project whose form reaches Main only after its wait did.
    late = {'state': 'open', 'question': 'Which licence notice goes on the cover?', 'options': ['Short notice', 'Full notice'],
            'option_details': ['One line with a link.', 'The whole licence text.'], 'stake': 'The cover layout.',
            'wait_for_answer': True, 'asked_at': '2026-09-16T00:06:00Z'}

    def connect(ws):
        sockets.append(ws)
        ws.send(json.dumps({'type': 'heartbeat'}))
    page.route_web_socket('**/ws', connect)
    page.route('**/api/projects', lambda r: r.fulfill(json={'projects': [project]}))
    # A stable served SHA: a reconnect re-reads history in place instead of reloading the window.
    page.route('**/api/state', lambda r: r.fulfill(json={'sha': 'browser-fixture', 'supervisor_ready': True,
        'active_chat_activities': activities, 'projects': [project], 'project_chat_ids': [42]}))

    def detail(route):
        # The canonical record (GET /api/tasks/{id}) the Project form and a revalidating Main read.
        task_id = urlparse(route.request.url).path.rsplit('/', 1)[-1]
        detail_reads.append(task_id)
        quizzes = blocks if task_id == 'proof-task' else {'late-form': late}
        route.fulfill(json={'task_id': task_id, 'project_id': project['id'],
            'owner_quiz': {qid: {'quiz_id': qid, **block} for qid, block in quizzes.items()},
            'owner_wait': wait if task_id == 'proof-task' else {'quiz_id': 'late-form', 'state': 'resumed'}})
    page.route('**/api/tasks/proof-task', detail)
    page.route('**/api/tasks/late-task', detail)

    def pointer(qid, block):
        # The row the Python producer emits (project_question_pointer): complete for the form.
        return {'role': 'system', 'system_type': 'project_question_pointer', 'task_id': 'proof-task', 'quiz_id': qid,
            'quiz_state': block['state'], 'project_id': project['id'], 'project_name': project['name'],
            'project_chat_id': 42, 'chat_id': 1, 'ts': block['asked_at'],
            **({'owner_wait_state': wait['state']} if qid == wait['quiz_id'] else
               {'owner_wait_state': 'resumed'} if activities and block.get('wait_for_answer') else {}),
            **{key: block[key] for key in ('question', 'options', 'option_details', 'stake', 'assumption',
                                           'recommended_index', 'wait_for_answer', 'answered_index', 'comment') if key in block}}

    def decide(route):
        sent = route.request.post_data_json
        decisions.append(sent)
        qid = sent['decision_id'].split(':')[2]
        blocks[qid].update(state='answered', answered_index=sent['option_index'])
        if qid == wait['quiz_id']:
            wait['state'] = 'resumed'
            activities[0]['required_question'].update(quiz_state='answered', answered_index=sent['option_index'],
                                                      owner_wait_state='resumed')
        route.fulfill(json={'ok': True, 'state': 'answered', 'answered_index': sent['option_index'],
                            **({'answered_after_terminal': True, 'forwarded': True} if qid == 'finished' else {})})
    page.route('**/api/decisions', decide)

    def history(route):
        history_reads.append(route.request.url)
        chat_id = parse_qs(urlparse(route.request.url).query).get('chat_id', ['1'])[0]
        if chat_id == '42':
            # Source questions are outside this window: exact navigation must read detail.
            rows = [{'role': 'assistant', 'text': 'Later retained project message.', 'ts': '2026-09-16T01:00:00Z'}]
        else:
            source = asked if mode['stale'] else blocks
            rows = [pointer(qid, block) for qid, block in source.items()]
            # Later Main reading below the questions, so a removal can happen above the reader.
            rows += [{'role': 'assistant', 'text': f'Later Main note {i}. ' + 'Reading continues below the questions. ' * 4,
                      'ts': f'2026-09-16T00:1{i}:00Z'} for i in range(8)]
        route.fulfill(json={'messages': rows, 'progress': []})
    page.route('**/api/chat/history*', history)
    open_app(ui)
    mirrors = page.locator('#chat-messages .chat-bubble.project-question')
    card = lambda qid: page.locator(f'#chat-messages .project-question-card[data-quiz-id="{qid}"]')
    ids = lambda: mirrors.evaluate_all("els => els.map(el => el.querySelector('.chat-quiz-card').dataset.quizId)")
    card('waiting').wait_for()
    assert ids() == ['second', 'passed', 'finished', 'replaced', 'waiting'], 'an answered question never enters Main'
    forms = page.evaluate("""() => [...document.querySelectorAll('#chat-messages .project-question-card')].map(el => ({
        id: el.dataset.quizId, status: el.querySelector('.chat-quiz-status-text').textContent,
        chip: el.querySelector('.chat-quiz-project .chat-live-project-name')?.textContent,
        question: el.querySelector('.chat-quiz-question').textContent.trim().slice(0, 24),
        details: [...el.querySelectorAll('.chat-quiz-option-detail')].map(n => n.textContent),
        stake: el.querySelector('.chat-quiz-stake')?.textContent.trim() || '',
        own: !!el.querySelector('.chat-quiz-comment'), disabled: [...el.querySelectorAll('.chat-quiz-option')].every(b => b.disabled),
        recommended: [...el.querySelectorAll('.chat-quiz-option')].findIndex(b => b.querySelector('.chat-quiz-option-recommended'))}))""")
    by_id = {form['id']: form for form in forms}
    assert {form['chip'] for form in forms} == {project['name']}
    assert by_id['waiting']['status'] == 'Waiting for your answer' and by_id['waiting']['own']
    assert by_id['waiting']['stake'] == 'At stake: Whether the archive ships this week.'
    assert by_id['waiting']['details'] == ['Readers follow one extra link.', 'Publication slips by about a week.']
    assert by_id['waiting']['recommended'] == 0 and by_id['passed']['recommended'] == 1
    assert by_id['passed']['details'] == ['Lossless, larger', 'Smaller at quality 82'] and by_id['passed']['own']
    assert by_id['finished']['status'].startswith('Unanswered · the task finished') and by_id['finished']['own']
    assert by_id['replaced']['status'] == 'Replaced by a newer question'
    assert by_id['replaced']['disabled'] and not by_id['replaced']['own'], 'a replaced question is a read-only record'
    assert card('waiting').locator('.chat-quiz-question strong').inner_text() == 'licence'
    geometry = page.evaluate("""() => {
        const scroller = document.querySelector('#chat-messages');
        const cards = [...document.querySelectorAll('#chat-messages .project-question-card')];
        const box = el => el.getBoundingClientRect();
        return {overflow: scroller.scrollWidth - scroller.clientWidth, page: document.documentElement.scrollWidth - innerWidth,
            right: Math.max(...cards.map(el => box(el).right)), viewport: innerWidth,
            chipInside: cards.every(el => box(el.querySelector('.chat-quiz-project')).right <= box(el).right + 0.5),
            chipCut: (() => { const n = cards[0].querySelector('.chat-quiz-project .chat-live-project-name'); return n.scrollWidth > n.clientWidth; })(),
            sizes: [...new Set(cards.flatMap(el => [el.querySelector('.chat-quiz-status'), el.querySelector('.chat-quiz-project')])
                .map(el => getComputedStyle(el).fontSize))]};
    }""")
    print(json.dumps({'question_mirror_geometry': geometry, 'viewport': [width, height]}))
    assert geometry['overflow'] <= 1 and geometry['page'] <= 0 and geometry['right'] <= width, geometry
    assert geometry['chipInside'] and geometry['sizes'] == ['12px'], geometry
    card('second').scroll_into_view_if_needed()
    setup_browser.capture(page, f'question-mirrors-burst-{width}')

    # Both asks arrived before the single wait was published. Its next ordinary census names
    # only the last quiz: the older wait ends in place, the question stays answerable.
    read_count = len(history_reads)
    activities.append({'activity_id': 'proof-task', 'chat_id': 42, 'project_id': project['id'],
        'kind': 'direct_chat', 'phase': 'working', 'required_question': pointer('waiting', blocks['waiting'])})
    page.wait_for_function("() => document.querySelector('[data-quiz-id=\"second\"] .chat-quiz-status-text')"
                           "?.textContent.includes('task continued')", timeout=15000)
    assert len(history_reads) == read_count, 'census freshness must not require history refetch'
    assert card('second').locator('.chat-quiz-comment').count() == 1
    assert card('second').locator('.chat-quiz-wait-ended').count() == 1

    # The chip opens the exact question in its Project: the same form, node for node.
    card('waiting').locator('.chat-quiz-project').click()
    own = page.locator('.chat-quiz-card[data-task-id="proof-task"][data-quiz-id="waiting"]:not(.project-question-card)')
    own.get_by_text('Publication slips by about a week.', exact=True).wait_for()
    assert own.evaluate(CARD_FORM_JS) == card('waiting').evaluate(CARD_FORM_JS)
    setup_browser.capture(page, f'question-mirror-same-form-{width}')
    page.locator('#project-panel-close').click()

    # One touch answers from Main: the recorded result reads for five seconds, then only the
    # Main copy goes. A duplicate live frame and the census's own answer never restart it.
    card('waiting').locator('.chat-quiz-option').nth(1).click()
    page.locator('#chat-messages .project-question-card[data-quiz-id="waiting"][data-state="answered"]').wait_for()
    confirmed = page.evaluate('performance.now()')
    assert card('waiting').locator('.chat-quiz-option.chosen').inner_text().startswith('Wait for written permission')
    assert card('waiting').locator('.chat-quiz-status-text').inner_text() == 'You answered'
    assert card('waiting').locator('.chat-quiz-comment').count() == 0
    assert [(sent['decision_id'], sent['option_index'], 'comment' in sent) for sent in decisions] == [
        ('quiz:proof-task:waiting', 1, False)]
    card('waiting').evaluate("el => el.scrollIntoView({block: 'center'})")
    setup_browser.capture(page, f'question-mirror-answered-{width}')
    page.wait_for_timeout(3000)
    sockets[-1].send(json.dumps({'type': 'quiz_state', 'task_id': 'proof-task', 'quiz_id': 'waiting', 'state': 'answered',
                                 'answered_index': 1, 'ts': '2026-09-16T00:04:00Z'}))
    assert card('waiting').count() == 1, 'the result stays readable for the whole moment'
    page.wait_for_function("() => !document.querySelector('#chat-messages [data-quiz-id=\"waiting\"]')", timeout=10000)
    elapsed = page.evaluate('performance.now()') - confirmed
    assert 4700 <= elapsed <= 6500, elapsed
    setup_browser.capture(page, f'question-mirror-removed-{width}')
    # Only the Main copy went: the Project keeps its answered card.
    page.evaluate("""p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p,
        task_id: 'proof-task', quiz_id: 'waiting'}}))""", project)
    own.locator('.chat-quiz-option.chosen').filter(has_text='Wait for written permission').wait_for()
    page.locator('#project-panel-close').click()

    # The Project form answers another question while the owner reads further down: the copy
    # above the reader settles and leaves without moving the reading position.
    card('finished').evaluate("el => el.scrollIntoView({block: 'start'})")
    anchor = card('finished').evaluate('el => el.getBoundingClientRect().top')
    frame = {'type': 'quiz_state', 'task_id': 'proof-task', 'quiz_id': 'passed', 'state': 'answered', 'answered_index': 0,
             'comment': 'PNG for the print edition.', 'ts': '2026-09-16T00:05:00Z'}
    blocks['passed'].update(state='answered', answered_index=0, comment=frame['comment'])
    sockets[-1].send(json.dumps(frame))
    card('passed').locator('.chat-quiz-answer').get_by_text("Owner's answer: PNG for the print edition.", exact=True).wait_for()
    started = page.evaluate('performance.now()')
    page.wait_for_timeout(2500)
    sockets[-1].send(json.dumps(frame))
    page.wait_for_function("() => !document.querySelector('#chat-messages [data-quiz-id=\"passed\"]')", timeout=10000)
    elapsed = page.evaluate('performance.now()') - started
    assert 4500 <= elapsed <= 6500, elapsed
    assert abs(card('finished').evaluate('el => el.getBoundingClientRect().top') - anchor) <= 2

    # A keyboard answer: focus stays in the settled copy, then moves on to the next question.
    card('finished').locator('.chat-quiz-question').focus()
    page.keyboard.press('Alt+Tab')
    assert page.evaluate("document.activeElement.classList.contains('chat-quiz-option')"), \
        page.evaluate('document.activeElement.outerHTML')
    page.keyboard.press('Enter')
    page.locator('#chat-messages .project-question-card[data-quiz-id="finished"][data-state="answered"]').wait_for()
    assert page.evaluate("document.activeElement.closest('.chat-quiz-card')?.dataset.quizId") == 'finished'
    page.wait_for_function("() => !document.querySelector('#chat-messages [data-quiz-id=\"finished\"]')", timeout=10000)
    assert page.evaluate("document.activeElement.closest('.chat-quiz-card')?.dataset.quizId") == 'replaced'
    assert ids() == ['second', 'replaced']

    # The census names another task's wait before any row carries its form: a partial copy with
    # its way to the Project. The complete row then arrives already answered (the Project form
    # answered meanwhile): the whole form shows the recorded result for five seconds, then goes.
    base = {'role': 'system', 'system_type': 'project_question_pointer', 'task_id': 'late-task', 'quiz_id': 'late-form',
            'project_id': project['id'], 'project_name': project['name'], 'project_chat_id': 42, 'chat_id': 1,
            'ts': late['asked_at'], 'wait_for_answer': True}
    activities.append({'activity_id': 'late-task', 'chat_id': 42, 'project_id': project['id'], 'kind': 'direct_chat',
        'phase': 'working', 'required_question': {**base, 'quiz_state': 'open', 'owner_wait_state': 'waiting'}})
    card('late-form').wait_for(timeout=15000)
    assert card('late-form').locator('.chat-quiz-option').count() == 0
    assert card('late-form').locator('.chat-quiz-question').inner_text() == 'Open the original question for its text.'
    assert card('late-form').locator('.chat-quiz-project').count() == 1
    setup_browser.capture(page, f'question-mirror-partial-{width}')
    late.update(state='answered', answered_index=1)
    sockets[-1].send(json.dumps({'type': 'chat', 'content': f"You answered in {project['name']}", 'is_progress': False,
        'markdown': False, **base, **{key: late[key] for key in ('question', 'options', 'option_details', 'stake',
                                                                    'answered_index')},
        'quiz_state': 'answered', 'owner_wait_state': 'resumed'}))
    page.locator('#chat-messages .project-question-card[data-quiz-id="late-form"][data-state="answered"]'
                 ' .chat-quiz-option.chosen').wait_for()
    started = page.evaluate('performance.now()')
    assert card('late-form').locator('.chat-quiz-option').count() == 2
    assert card('late-form').locator('.chat-quiz-option.chosen').inner_text().startswith('Full notice')
    assert card('late-form').locator('.chat-quiz-status-text').inner_text() == 'You answered'
    assert card('late-form').locator('.chat-quiz-stake').inner_text() == 'At stake: The cover layout.'
    card('late-form').evaluate("el => el.scrollIntoView({block: 'center'})")
    setup_browser.capture(page, f'question-mirror-late-form-answered-{width}')
    page.wait_for_function("() => !document.querySelector('#chat-messages [data-quiz-id=\"late-form\"]')", timeout=10000)
    elapsed = page.evaluate('performance.now()') - started
    assert 4500 <= elapsed <= 6500, elapsed
    assert ids() == ['second', 'replaced']
    assert 'late-task' not in detail_reads, 'a question this tab remembers is never re-read'

    # Stale snapshots — the census and a reconnect's history re-read taken before the answers —
    # never bring a removed copy back, also after Main's bounded question memory (2000) let the
    # answers go: every quiz_state frame is one more remembered question.
    for index in range(2001):
        sockets[-1].send(json.dumps({'type': 'quiz_state', 'task_id': 'noise-task', 'quiz_id': f'noise-{index}',
                                     'state': 'open'}))
    page.wait_for_timeout(500)
    detail_count = len(detail_reads)
    mode['stale'] = True
    activities[0]['required_question'] = pointer('waiting', asked['waiting'])
    read_count = len(history_reads)
    page.evaluate('window.sameDocument = true')
    sockets[-1].close()
    deadline = page.evaluate('performance.now()') + 15000
    while len(history_reads) == read_count and page.evaluate('performance.now()') < deadline:
        page.wait_for_timeout(200)
    assert len(history_reads) > read_count and len(sockets) == 2, 'the reconnect re-read history'
    page.wait_for_timeout(3500)
    assert page.evaluate('window.sameDocument === true'), 'the reconnect re-read history in place'
    card('second').scroll_into_view_if_needed()
    setup_browser.capture(page, f'question-mirrors-after-stale-reconnect-{width}')
    assert ids() == ['second', 'replaced']
    # The forgotten questions were settled by their canonical record, one read per task.
    assert {'proof-task', 'late-task'} <= set(detail_reads[detail_count:]), detail_reads[detail_count:]
    assert len(decisions) == 2, 'observing answers never sends one'

    # A reload paints durable history: answered questions stay out, the rest keep their form.
    mode['stale'] = False
    activities[0]['required_question'] = pointer('waiting', blocks['waiting'])
    activities[1]['required_question'].update(quiz_state='answered', answered_index=1, owner_wait_state='resumed')
    page.reload()
    page.wait_for_selector('#chat-input', state='attached')
    card('replaced').wait_for()
    assert ids() == ['second', 'replaced']
    card('second').scroll_into_view_if_needed()
    setup_browser.capture(page, f'question-mirrors-reloaded-{width}')
    assert len(decisions) == 2, 'a reload and a navigation never answer anything'
