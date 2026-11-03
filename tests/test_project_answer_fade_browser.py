"""The approved Main mirror fade through the real chat consumer, with static gateway facts.

No runtime or live storage: reuse the isolated HTTP/API fixture. The excerpt is from the
accepted owner's preview; generated line counts exercise the two fold boundaries precisely.
"""
import json

import pytest

from tests.test_project_work_card_browser import _evidence_dir
from tests.test_project_work_card_browser import project_chat as project_chat

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

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


def emit(page, key, answer, limitation=False):
    row = {"role": "system", "system_type": "project_completion_summary", "chat_id": 1,
           "task_id": key, "project_id": "alpha", "project_name": "Project", "task_name": "GitHub pass",
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
def test_extended_fade_keeps_main_mirror_footprint_and_actions(project_chat, tmp_path):
    chat = project_chat
    page, expect = chat["page"], chat["expect"]
    page.set_viewport_size({"width": chat["width"], "height": 750})
    evidence = _evidence_dir(tmp_path)
    stem = f"fade-{chat['engine']}-{chat['theme']}-{chat['width']}"
    facts = {}
    bubble = emit(page, "fade-long", ANSWER)
    facts["long"] = stable_footprint(page, bubble)
    assert facts["long"]["message"]["height"] == 216
    assert facts["long"]["margin"] == -48
    assert facts["long"]["folded"]
    assert bubble.locator('.message a').get_attribute('tabindex') == '-1'
    capture(page, bubble, evidence / f"{stem}.png")
    # Real controls retain the complete answer and Project destination, even below the painted tail.
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

    # During width movement the production owner defers JS measurement. The CSS fallback
    # must preserve the old geometry BEFORE a settle refresh, including a stale long-tail margin.
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
    assert warning.locator('.project-answer-limitation').bounding_box()['y'] >= warning.locator('.message').bounding_box()['y'] + 168
    warning.locator('.project-ending-details > summary').click()
    expect(warning.locator('.project-ending-details > p')).to_be_visible()
    stable_footprint(page, warning)
    capture(page, warning, evidence / f"{stem}-warning.png")
    (evidence / f"{stem}.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
