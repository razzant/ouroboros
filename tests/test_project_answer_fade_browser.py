"""The approved Main mirror fade through the real chat consumer, with static gateway facts.

No runtime or live storage: reuse the isolated HTTP/API fixture. The excerpt is from the
accepted owner's preview; generated line counts exercise the two fold boundaries precisely.
"""
import json
import re

import pytest

from tests.test_project_work_card_browser import _evidence_dir
from tests.test_project_work_card_browser import project_chat as project_chat

pytestmark = [pytest.mark.ui_browser]

ANSWER = """Вечерний GitHub-проход завершён. Семь ответов опубликованы от `ouroboros-agent`; каждый независимо перечитан по тексту, автору и адресу.

Главное:

- **[#2214: старый Linux-launcher не принимает новые Settings.](https://github.com/razzant/ouroboros/issues/2214)** С Sol сверили исторический и нынешний код: старый список ролей действительно отвергает `safety`, даже со значением Auto. Точный установщик автора и отсутствие traceback ещё не воспроизводили. Попросил имеющиеся сведения об архиве; удалять настройки или повторять неудачный запуск не советовал.
- **Discussion #2213: ошибка запуска на Windows.** Осмотрел PDF со скриншотом. Видно падение локальной команды настройки Git, но не её причина. Ответил по-русски и попросил две безопасные команды чтения, без Reset, удаления папок или отправки ключей.

**Один вопрос тебе остаётся открытым:** рекомендую взять #2214 первым, затем узкие #2199/#2211. Новые UI-возможности обсудить отдельно. До выбора инженеров не запускаю."""

# Old clamp geometry, used only as a measurement oracle; the consumer and DOM stay production.
BASELINE = """.chat-bubble.project-answer > .message {
    max-height: var(--project-answer-fold) !important; margin-bottom: 0 !important;
}"""
GEOMETRY = """b => {
    const origin = b.getBoundingClientRect();
    const box = selector => { const r = b.querySelector(selector).getBoundingClientRect();
        return {x:r.x-origin.x, y:r.y-origin.y, width:r.width, height:r.height}; };
    const m = b.querySelector('.message'), css = getComputedStyle(m);
    return {height:origin.height, message:box('.message'), project:box('.system-message-actions'),
        time:box('.msg-time'), actions:box('.chat-message-actions'),
        folded:b.classList.contains('is-folded'), mask:css.maskImage || css.webkitMaskImage,
        margin:parseFloat(css.marginBottom), line:parseFloat(css.lineHeight),
        overflow:b.scrollWidth > b.clientWidth + 1, scrollWidth:b.scrollWidth, clientWidth:b.clientWidth,
        viewportOverflow:origin.left < 0 || origin.right > innerWidth};
}"""


def emit(page, key, answer, limitation=False, *, project_name="Project"):
    row = {"role": "system", "system_type": "project_completion_summary", "chat_id": 1,
           "task_id": key, "project_id": "alpha", "project_name": project_name, "task_name": "GitHub pass",
           "content": "Project completed", "completion_answer": answer,
           "ts": "2026-10-22T21:28:00Z", "history_id": key,
           "terminal_time": {"v": 1, "source": "executor_terminal", "occurred_at": "2026-10-22T21:28:00Z", "attempt": {}},
           "row_address": f"row:1@2026-10-22T21:28:00Z#{key}"}
    if limitation:
        row.update(completion_reason="The Windows measurements are still missing.", completion_notice={
            "phase": "failed", "cause": "The Windows measurements are still missing.",
            "undescribed": ["windows_measurements_missing"], "salvage": ""})
    page.evaluate("row => emit('chat', row)", row)
    bubble = page.locator(f'#chat-messages > .project-answer[data-task-id="{key}"]')
    bubble.wait_for()
    bubble.scroll_into_view_if_needed()
    refresh(page, bubble)
    return bubble


def refresh(page, bubble):
    bubble.evaluate("""async b => {
        const {refreshProjectAnswerFold} = await import('/static/modules/project_answer.js');
        refreshProjectAnswerFold(b);
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }""")


def stable_footprint(page, bubble):
    expected_style = page.add_style_tag(content=BASELINE)
    before = bubble.evaluate(GEOMETRY)
    expected_style.evaluate("e=>e.remove()")
    refresh(page, bubble)
    after = bubble.evaluate(GEOMETRY)
    for key in ("height", "project", "time", "actions"):
        assert after[key] == before[key], (key, before, after)
    assert after["overflow"] == before["overflow"], (before, after)
    assert not after["viewportOverflow"], after
    return after


def capture(page, bubble, path):
    # A generic scroll-into-view can leave the footer behind the floating composer.
    # Centre it as a reader would, then verify the whole card is visible before capture.
    bubble.evaluate("b=>b.scrollIntoView({block:'center', behavior:'instant'})")
    page.wait_for_function("""b => {
        const r=b.getBoundingClientRect(), input=document.querySelector('#chat-input').getBoundingClientRect();
        return r.top >= 50 && r.bottom < input.top - 48;
    }""", arg=bubble.element_handle())
    page.screenshot(path=str(path), animations="disabled")
    page.screenshot(path=str(path.with_stem(path.stem + '-card')), clip=bubble.bounding_box(), animations="disabled")


@pytest.mark.parametrize("project_chat", [
    ("webkit", "light", 1100), ("webkit", "dark", 1100),
    ("webkit", "light", 390), ("webkit", "dark", 390),
    ("chromium", "light", 1100), ("chromium", "dark", 1100),
], indirect=True)
def test_folded_fade_keeps_main_mirror_footprint_and_actions(project_chat, tmp_path):
    chat = project_chat
    page, expect = chat["page"], chat["expect"]
    page.set_viewport_size({"width": chat["width"], "height": 750})
    evidence = _evidence_dir(tmp_path)
    stem = f"fade-{chat['engine']}-{chat['theme']}-{chat['width']}"
    facts = {}
    bubble = emit(page, "fade-long", ANSWER)
    facts["long"] = stable_footprint(page, bubble)
    # Seven lines, the last two fading on the shared edge (ui.css [data-fold-clipped]); nothing paints past the fold.
    assert facts["long"]["message"]["height"] == 168
    assert facts["long"]["margin"] == 0
    assert facts["long"]["folded"]
    assert facts["long"]["mask"].startswith("linear-gradient(") and "rgba(0, 0, 0, 0.5)" in facts["long"]["mask"], facts["long"]
    assert bubble.locator('.message a').get_attribute('tabindex') == '-1'
    capture(page, bubble, evidence / f"{stem}.png")
    # Real controls retain the complete answer and Project destination.
    bubble.locator('.chat-message-copy').click()
    assert page.evaluate('copiedText') == ANSWER
    bubble.locator('.chat-message-reply').click()
    expect(page.get_by_role('button', name='Cancel reply', exact=True)).to_be_visible()
    assert page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply')).address") == bubble.get_attribute('data-row-address')
    assert ANSWER.split('**')[-1] in bubble.locator('.message').inner_text()
    chip = bubble.locator('[data-intent="open-project"]')
    if page.evaluate("matchMedia('(hover: hover)').matches"):
        # Reply can move the chip under the existing pointer; establish a true non-hover baseline.
        page.mouse.move(0, 0)
        assert not chip.evaluate('b=>b.matches(":hover")')
        chip.evaluate('async b=>{await Promise.all(b.getAnimations().map(a=>a.finished));}')
        normal = chip.evaluate('b=>getComputedStyle(b).backgroundImage')
        chip.evaluate('''b=>{b.dataset.tintTransition='';const onRun=e=>{
            if(e.propertyName==='--project-reference-fill'){b.dataset.tintTransition='yes';b.removeEventListener('transitionrun',onRun);}
        };b.addEventListener('transitionrun',onRun);}''')
        chip.hover()
        expect(chip).to_have_attribute('data-tint-transition', 'yes')
        chip.evaluate('async b=>{await Promise.all(b.getAnimations().map(a=>a.finished));}')
        assert chip.evaluate('b=>getComputedStyle(b).backgroundImage') != normal
    bubble.locator('[data-intent="open-project"]').click()
    assert page.evaluate('openedProject.project.id') == 'alpha'
    page.get_by_role('button', name='Cancel reply', exact=True).click()
    # The same mounted answer settles at a different width and returns without oscillation.
    page.set_viewport_size({"width": 390 if chat["width"] == 1100 else 1100, "height": 750})
    refresh(page, bubble)
    facts["resized"] = stable_footprint(page, bubble)
    page.set_viewport_size({"width": chat["width"], "height": 750})
    refresh(page, bubble)

    # Fractional type scales and content on either side of both boundaries. Repeated refreshes
    # must never alternate the class or move footer controls. Then shrink the same mounted node.
    for scale in (16, 17):
        bubble.evaluate("(b,size)=>b.style.setProperty('--type-section',size+'px')", scale)
        for lines in (2, 7, 8, 9, 12, 2):
            bubble.locator('.message').evaluate("(m,n)=>{m.textContent=Array.from({length:n},()=> 'Result line').join('\\n');m.style.whiteSpace='pre-wrap';}", lines)
            refresh(page, bubble)
            expected = stable_footprint(page, bubble)
            assert expected["folded"] == (lines > 7), (scale, lines, expected)
            assert (expected["mask"] != "none") == (lines > 7)
            for _ in range(4):
                refresh(page, bubble)
                assert bubble.evaluate(GEOMETRY) == expected
            facts[f"{scale}px-{lines}-lines"] = expected

    # During width movement the production owner defers JS measurement: the fold and its fade keep the geometry
    # BEFORE a settle refresh.
    bubble.locator('.message').evaluate("m=>{m.textContent='A visible result with enough words to wrap. '.repeat(12);m.style.whiteSpace='normal';}")
    refresh(page, bubble)
    bubble.evaluate("b=>b.parentElement.setAttribute('data-resizing','')")
    for width in (220, 300, 240):
        bubble.evaluate('(b,w)=>{b.style.width=w+"px";}', width)
        moving = bubble.evaluate(GEOMETRY)
        assert moving['margin'] == 0
        original = page.add_style_tag(content=BASELINE)
        before = bubble.evaluate(GEOMETRY)
        original.evaluate('e=>e.remove()')
        for key in ('height', 'project', 'time', 'actions'):
            assert moving[key] == before[key], (key, moving, before)
    bubble.evaluate("b=>{b.style.removeProperty('width');b.parentElement.removeAttribute('data-resizing');}")
    refresh(page, bubble)
    stable_footprint(page, bubble)

    warning = emit(page, "fade-warning", ANSWER, limitation=True)
    facts["warning"] = stable_footprint(page, warning)
    assert facts["warning"]["message"]["height"] == 168
    assert facts["warning"]["margin"] == 0
    assert facts["warning"]["mask"] == facts["long"]["mask"], 'a limitation under the answer changes nothing in its fade'
    assert warning.locator('.project-answer-limitation').bounding_box()['y'] >= warning.locator('.message').bounding_box()['y'] + 168
    warning.locator('.project-ending-details > summary').click()
    expect(warning.locator('.project-ending-details > p')).to_be_visible()
    stable_footprint(page, warning)
    capture(page, warning, evidence / f"{stem}-warning.png")
    (evidence / f"{stem}.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")


# Check both the visible controls and their invisible ::after reach (ui.css "Touch reach"). A centre tap alone
# misses the corner of a long Project reference that Copy used to cover.
FINGER = """b => {
    const own = node => node?.closest('.chat-message-actions > button');
    const time = b.querySelector(':scope > .msg-time').getBoundingClientRect();
    const reference = b.querySelector(':scope > .system-message-actions [data-intent="open-project"]');
    const visible = reference.getBoundingClientRect();
    const at = (x, y) => document.elementFromPoint(x, y);
    const reach = (node, rect) => {
        const style = getComputedStyle(node), after = getComputedStyle(node, '::after');
        // Absolute inset starts at the padding edge, so account for the button border.
        const inset = side => after.content === 'none' ? 0 : Math.min(0,
            parseFloat(after[side]) + parseFloat(style.getPropertyValue(`border-${side}-width`)));
        return {left: rect.left + inset('left'), right: rect.right - inset('right'),
                top: rect.top + inset('top'), bottom: rect.bottom - inset('bottom')};
    };
    const link = reach(reference, visible);
    const hits = box => [.05, .5, .95].flatMap(fy => [.05, .5, .95].map(fx =>
        at(box.left + (box.right - box.left) * fx, box.top + (box.bottom - box.top) * fy)?.closest('button') === reference));
    return {coarse: matchMedia('(pointer: coarse)').matches, timeRight: time.right,
        reference: {visible: visible.toJSON(), reach: link, hits: hits(visible), reachHits: hits(link),
                    tap: {x: visible.left + visible.width * .95, y: visible.top + visible.height * .95}},
        controls: [...b.querySelectorAll(':scope > .chat-message-actions > button')].map(node => {
            const r = node.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2, half = 22;
            const hit = (px, py) => own(at(px, py)) === node;
            return {name: node.dataset.messageAction || node.className.split(' ')[0], left: r.left, right: r.right,
                    x, y, reach: reach(node, r),
                    hits: [hit(x, y), hit(x - half + 1, y), hit(x + half - 1, y), hit(x, y - half + 1), hit(x, y + half - 1)]};
        })};
}"""


@pytest.mark.parametrize("project_chat", [
    ("webkit", "light", 402), ("webkit", "dark", 402), ("chromium", "light", 320),
], indirect=True)
def test_folded_answer_footer_keeps_touch_targets_separate(project_chat, tmp_path):
    chat = project_chat
    page, expect = chat["page"], chat["expect"]
    page.set_viewport_size({"width": chat["width"], "height": 874})
    page.evaluate("async () => (await import('/static/modules/message_share.js')).registerMessageShare()")
    evidence = _evidence_dir(tmp_path)
    for size, name in (("short", "Project"), ("long", "Conference notes and drawings for a very long project name")):
        sample = f"The {size} map is ready for review."
        bubble = emit(page, f"finger-{size}", "\n\n".join([sample] * 12), project_name=name)
        expect(bubble.locator("[data-message-action='share']")).to_be_visible()
        expect(bubble.locator('[data-intent="open-project"]')).to_have_attribute('aria-label', f'Open project {name}')
        expect(bubble.locator('.msg-time > span[aria-hidden]')).to_be_visible()
        geo = bubble.evaluate(GEOMETRY)
        assert geo['folded'] and not geo['viewportOverflow'], geo
        stem = f"finger-{chat['engine']}-{chat['theme']}-{chat['width']}-{size}"
        capture(page, bubble, evidence / f"{stem}.png")
        found = bubble.evaluate(FINGER)
        (evidence / f"{stem}.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
        assert found['coarse'], found
        reference, controls = found['reference'], found['controls']
        assert reference['hits'] == reference['reachHits'] == [True] * 9, found
        assert [c['name'] for c in controls] == ['chat-message-reply', 'share', 'chat-message-copy'], found
        assert found['timeRight'] <= controls[0]['left'], found
        r = reference['reach']
        assert r['right'] - r['left'] >= 44 and r['bottom'] - r['top'] >= 44, found
        for control in controls:
            assert control['hits'] == [True] * 5, (control['name'], found)
            c = control['reach']
            assert c['right'] - c['left'] >= 44 and c['bottom'] - c['top'] >= 44, found
            assert r['right'] <= c['left'] or r['left'] >= c['right'] or r['bottom'] <= c['top'] or r['top'] >= c['bottom'], found
        page.touchscreen.tap(reference['tap']['x'], reference['tap']['y'])
        assert page.evaluate('window.openedProject?.project') == {'id': 'alpha', 'name': name}
        reply = controls[0]
        page.touchscreen.tap(reply['x'], reply['y'])
        page.wait_for_function("() => document.activeElement?.id === 'chat-input'")
        cancel = page.get_by_role('button', name='Cancel reply', exact=True)
        expect(cancel).to_be_visible()
        quote = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
        assert quote['address'] == bubble.get_attribute('data-row-address')
        assert quote['snippet'].startswith(sample)
        cancel.tap()


@pytest.mark.parametrize("project_chat", [("webkit", "light", 402), ("chromium", "dark", 402)], indirect=True)
def test_system_project_row_over_copy_keeps_the_answers_clearance(project_chat, tmp_path):
    """A System Project row (an ending with no answer; started and handoff rows share its action row) above a copy control keeps the same touch
    clearance as an answer's: its long reference's right end opens the Project, and Reply and Copy hit themselves."""
    chat = project_chat
    page, expect = chat["page"], chat["expect"]
    page.set_viewport_size({"width": chat["width"], "height": 874})
    name = "Conference notes and drawings for a very long project name"
    page.evaluate("row => emit('chat', row)", {
        "role": "system", "system_type": "project_completion_summary", "chat_id": 1, "task_id": "sibling-row",
        "project_id": "alpha", "project_name": name, "task_name": "GitHub pass", "content": "Project completed",
        "completion_reason": "The work stopped before it wrote an answer.",
        "ts": "2026-10-22T21:28:00Z", "history_id": "sibling-row",
        "terminal_time": {"v": 1, "source": "executor_terminal", "occurred_at": "2026-10-22T21:28:00Z", "attempt": {}},
        "row_address": "row:1@2026-10-22T21:28:00Z#sibling-row"})
    bubble = page.locator('#chat-messages > .chat-bubble.system[data-task-id="sibling-row"]')
    bubble.wait_for()
    bubble.scroll_into_view_if_needed()
    expect(bubble.locator('.chat-message-actions > .chat-message-copy')).to_be_visible()
    found = bubble.evaluate(FINGER)
    (_evidence_dir(tmp_path) / f"sibling-{chat['engine']}-{chat['theme']}.json").write_text(json.dumps(found, indent=2), encoding="utf-8")
    assert found['coarse'], found
    reference, controls = found['reference'], found['controls']
    assert [c['name'] for c in controls] == ['chat-message-reply', 'chat-message-copy'], found
    assert reference['hits'] == reference['reachHits'] == [True] * 9, found
    r = reference['reach']
    for control in controls:
        assert control['hits'] == [True] * 5, (control['name'], found)
        c = control['reach']
        assert r['right'] <= c['left'] or r['left'] >= c['right'] or r['bottom'] <= c['top'] or r['top'] >= c['bottom'], found
    page.touchscreen.tap(reference['tap']['x'], reference['tap']['y'])
    assert page.evaluate('window.openedProject?.project') == {'id': 'alpha', 'name': name}


@pytest.mark.parametrize("project_chat", [
    ("webkit", "light", 1100), ("chromium", "light", 1100),
], indirect=True)
def test_folded_project_answer_stops_and_restores_keyboard_tabindex(project_chat):
    """Mirrored project answers stop every keyboard focus/scroller target while folded without inert holes,
    keep late overflow stops held while folded, and restore original/scroller tabindex on unfold."""
    chat = project_chat
    page = chat["page"]
    page.set_viewport_size({"width": chat["width"], "height": 750})
    wide_line = " ".join(f"project_wide_code_{i:02d}" for i in range(1, 16))
    rich_answer = "\n\n".join([
        "Project summary with [Project link](https://example.com/project).",
        "```text\nshort code\n```",
        "```text\ninitially short code\n```",
        f"```text\n{wide_line}\n```",
        "| Col A | Col B | Col C | Col D | Col E | Col F | Col G | Col H |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
        "| [Table ref](https://example.com/tref) | Alpha01 | Beta02 | Gamma03 | Delta04 | Epsilon05 | Zeta06 | Eta07 |",
        "Closing paragraph below the seven-line fold.",
    ])
    bubble = emit(page, "fade-keyboard", rich_answer)
    # Add an authored tabindex target inside the message and refresh so the fold owner captures it cleanly.
    bubble.evaluate("""b => {
        const span = document.createElement('span');
        span.className = 'authored-stop';
        span.setAttribute('tabindex', '2');
        span.textContent = 'Authored stop';
        b.querySelector('.message').appendChild(span);
    }""")
    refresh(page, bubble)
    state_js = """b => {
        const m = b.querySelector('.message'), pres = m.querySelectorAll('pre');
        const nodes = {
            link: m.querySelector('a[href="https://example.com/project"]'),
            tableLink: m.querySelector('a[href="https://example.com/tref"]'),
            shortPre: pres[0], latePre: pres[1], widePre: pres[2],
            tableWrap: m.querySelector('.md-table-wrap'),
            authored: m.querySelector('.authored-stop'),
        };
        return {
            folded: b.classList.contains('is-folded'),
            inertCount: m.querySelectorAll('[inert], [data-fold-inert]').length,
            stopped: Object.entries(nodes).filter(([, el]) => el.hasAttribute('data-fold-stop')).map(([k]) => k),
            tabs: Object.fromEntries(Object.entries(nodes).map(([k, el]) => [k, el.getAttribute('tabindex')])),
        };
    }"""
    all_keys = ["link", "tableLink", "shortPre", "latePre", "widePre", "tableWrap", "authored"]
    all_stopped = {"folded": True, "inertCount": 0, "stopped": all_keys, "tabs": {k: "-1" for k in all_keys}}
    assert bubble.evaluate(state_js) == all_stopped

    # Overflow appearing while folded (real ResizeObserver via viewport change) must never leak tabindex="0",
    # and must restore tabindex="0" once the fold lifts.
    bubble.evaluate("(b, wide) => { b.querySelectorAll('.message pre code')[1].textContent = wide; }", wide_line)
    page.set_viewport_size({"width": 420, "height": 750})
    page.wait_for_function(
        "b => b.querySelectorAll('.message pre')[1]?.getAttribute('role') === 'region'",
        arg=bubble.element_handle(),
    )
    assert bubble.evaluate(state_js) == all_stopped

    # Unfolding the same mounted answer restores original/scroller/authored tabindex verbatim.
    bubble.evaluate("b => b.style.setProperty('--project-answer-lines', '80')")
    refresh(page, bubble)
    assert bubble.evaluate(state_js) == {
        "folded": False, "inertCount": 0, "stopped": [],
        "tabs": {"link": None, "tableLink": None, "shortPre": None, "latePre": "0", "widePre": "0", "tableWrap": "0", "authored": "2"},
    }

    # Refolding stops every target again without inert holes.
    bubble.evaluate("b => b.style.removeProperty('--project-answer-lines')")
    refresh(page, bubble)
    assert bubble.evaluate(state_js) == all_stopped


# The answer's own width decides its fold, whatever moved it: no window resize, no settled-width event, no refresh by
# hand. A boundary answer (just over seven lines at the narrow width, four at the wide one) must lose its fade and give
# its link back to Tab when the room widens, and take both again when it narrows; while the feed's width is still
# moving ([data-resizing]) a visible answer crossing the boundary is re-folded at once, in both directions. A real
# sidebar-handle drag in the full app is covered by the sprint's stand probe (notes: handle_c.py); this fixture mounts
# the chat alone, so the room's width is moved directly.
FOLD_STATE = """b => { const m = b.querySelector('.message'), a = m.querySelector('a');
    return {mask: m.hasAttribute('data-fold-clipped'), folded: b.classList.contains('is-folded'), tab: a.getAttribute('tabindex'),
        content: m.scrollHeight, height: m.clientHeight}; }"""


@pytest.mark.parametrize("project_chat", [("chromium", "dark", 1100), ("webkit", "light", 1100)], indirect=True)
def test_the_fold_follows_the_answers_own_width_in_both_directions(project_chat):
    chat = project_chat
    page = chat["page"]
    page.set_viewport_size({"width": chat["width"], "height": 750})
    bubble = emit(page, "fold-width", "Start")
    room = "document.querySelector('#page-chat')"
    page.evaluate(f"{room}.style.width = '420px'")
    bubble.evaluate("""b => { const m = b.querySelector('.message'), a = document.createElement('a');
        a.href = '#details'; a.textContent = 'Read details'; const words = document.createTextNode('');
        m.replaceChildren(words, a);
        for (let i = 0; i < 600 && m.scrollHeight < m.clientHeight + 20; i++) words.data += 'room detail '; }""")
    settle = "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(r, 120))))"
    page.evaluate(settle)
    narrow = bubble.evaluate(FOLD_STATE)
    assert narrow["mask"] and narrow["folded"] and narrow["tab"] == "-1", narrow
    page.evaluate(f"{room}.style.width = '1000px'")
    page.evaluate(settle)
    wide = bubble.evaluate(FOLD_STATE)
    assert wide["content"] <= wide["height"] + 1, wide
    assert not wide["mask"] and not wide["folded"] and wide["tab"] is None, ('widened: whole, no fade, back in Tab', wide)
    page.evaluate(f"{room}.style.width = '420px'")
    page.evaluate(settle)
    assert bubble.evaluate(FOLD_STATE) == narrow, 'narrowed again: folded, faded, out of Tab'
    # While the width is still moving (chat_reading_position.js marks the feed), a visible answer re-folds at once.
    page.evaluate("document.querySelector('#chat-messages').setAttribute('data-resizing', '')")
    for width, folded in (("1000px", False), ("420px", True)):
        page.evaluate(f"{room}.style.width = '{width}'")
        page.evaluate(settle)
        moving = bubble.evaluate(FOLD_STATE)
        assert moving["mask"] is folded and moving["folded"] is folded, (width, moving)
    page.evaluate("document.querySelector('#chat-messages').removeAttribute('data-resizing')")


# A room's edge as the browser computes it (DESIGN "One edge", "A room's edges"): at the top and at the bottom, two
# blur planes whose filters are the shared tokens, whatever rule tries to override them, and the tint above them; the
# top clears toward the feed below the bar, the bottom toward the feed above the composer.
ROOM_EDGE = """() => {
    const root = getComputedStyle(document.documentElement);
    const token = name => root.getPropertyValue(name).trim();
    const side = selector => [...document.querySelector(selector).querySelectorAll(':scope > .room-edge')].map(n => {
        const s = getComputedStyle(n);
        return {edge: n.dataset.edge, filter: s.backdropFilter || s.webkitBackdropFilter || 'none',
            mask: s.maskImage || s.webkitMaskImage, position: s.maskPosition || s.webkitMaskPosition,
            events: s.pointerEvents, hidden: n.getAttribute('aria-hidden')};
    });
    return {soft: token('--edge-blur-soft'), deep: token('--edge-blur-deep'),
        top: side('#page-chat > .chat-page-header'), bottom: side('#chat-input-area')};
}"""


@pytest.mark.parametrize("project_chat", [("chromium", "dark", 1100), ("webkit", "light", 390)], indirect=True)
def test_a_rooms_edge_is_two_shared_blur_planes_under_its_tint_clearing_toward_the_feed(project_chat):
    page = project_chat["page"]
    found = page.evaluate(ROOM_EDGE)
    assert found["soft"] and found["deep"] and found["soft"] != found["deep"], found
    first = lambda piece: piece["mask"].split("), linear-gradient(")[0]  # noqa: E731 - the ramp layer
    for name in ("top", "bottom"):
        pieces = found[name]
        assert [p["edge"] for p in pieces] == ["soft", "deep", "tint"], (name, found)
        assert [p["filter"] for p in pieces] == [found["soft"], found["deep"], "none"], (name, 'the shared tokens', found)
        assert all(p["events"] == "none" and p["hidden"] == "true" for p in pieces), (name, found)
        for piece in pieces:
            clears_upward = bool(re.search(r"to top|\b0deg", first(piece)))
            assert clears_upward is (name == "bottom"), (name, 'which end clears', piece)
            # The ramp layer lies at the end that clears, the solid layer at the other: the top ramps out below the
            # bar, the bottom ramps out above the composer, so no hard seam lands over the feed or the controls.
            rows = [layer.split()[-1] for layer in piece["position"].split(",")]
            ends = [1 if row == "100%" else 0 if row in ("0px", "0%") else None for row in rows]
            assert ends == ([1, 0] if name == "top" else [0, 1]), (name, 'ramp and solid layers in place', piece)


# The owner's criterion for a room's edge (27.11): it does not reach past the capsules. At the top nothing of it lies
# lower than the bar's capsules' lower side plus the bar's own padding; at the bottom nothing higher than the capsule
# row's upper side less that padding. The edge that reached one edge length below the bar fails it.
EDGE_EXTENT = """() => {
    const bar = document.querySelector('#page-chat > .chat-page-header'), area = document.querySelector('#chat-input-area');
    const shown = n => n.getClientRects().length && getComputedStyle(n).visibility !== 'hidden';
    const capsules = [...bar.querySelectorAll('.app-page-actions button, .cx-chip, .rw-trigger')].filter(shown);
    const chips = [...area.querySelectorAll('.chat-toolbar-row > *')].filter(shown);
    const nodes = host => [...host.querySelectorAll(':scope > .room-edge')].map(n => n.getBoundingClientRect());
    return {pad: parseFloat(getComputedStyle(bar).paddingBottom),
        capsulesEnd: Math.max(...capsules.map(n => n.getBoundingClientRect().bottom)),
        chipsStart: Math.min(...chips.map(n => n.getBoundingClientRect().top)),
        topEnd: Math.max(...nodes(bar).map(r => r.bottom)), bottomStart: Math.min(...nodes(area).map(r => r.top))};
}"""


@pytest.mark.parametrize("project_chat", [("chromium", "light", 1100), ("webkit", "dark", 390)], indirect=True)
def test_a_rooms_edge_ends_at_its_capsules_top_and_bottom(project_chat):
    found = project_chat["page"].evaluate(EDGE_EXTENT)
    assert found["capsulesEnd"] > 0 and found["chipsStart"] > found["capsulesEnd"], found
    assert found["topEnd"] <= found["capsulesEnd"] + found["pad"] + 0.5, ('the top edge ends at the capsules', found)
    assert found["bottomStart"] >= found["chipsStart"] - found["pad"] - 0.5, ('the bottom edge starts at the capsule row', found)
