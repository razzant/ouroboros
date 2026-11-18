"""Main navigation and explicit resume on the actual baseline SPA, with synthetic IO only."""
import pytest
from playwright.sync_api import expect

from tests import test_room_work_browser as rw

from .capture import Frames
from .scene import BASE, CHILD, NATIVE_QUIZ, NATIVE_TASK, NAMES, PASSAGE, PROJECT_QUIZ, PROJECT_TASK, ROOT, Scene

subscription_ui = rw.subscription_ui
pytestmark = pytest.mark.ui_browser
PILL = "#page-chat .chat-question-nav"
CARD = f'#chat-messages .chat-live-card[data-task-id="{ROOT}"]'
HELPER = f'[data-activity-section="queue"] [data-act="task-control"][data-id="{CHILD}"][data-resume="1"]'


def action_row(task):
    return f'#chat-messages .owner-action[data-owner-action="{task}"]'


@pytest.fixture(params=["dark", "light"])
def theme(request):
    return request.param


def setup(ui, theme, variant, history_mode="loaded", *, helper_first=False, questions=None, width=1100, baseline=BASE):
    page = ui["new_page"](viewport={"width": width, "height": 844 if width == 390 else 722},
                          has_touch=width == 390, locale="en-US", timezone_id="UTC", service_workers="block")
    page.set_default_timeout(10000)
    scene = Scene(page, variant, history_mode, questions=questions, helper_first=helper_first, baseline=baseline)
    attention = scene.snapshot["main_attention"]
    first = CHILD if helper_first else ROOT
    if variant == "A":  # stock A orders actions by the first appearance of each System row
        scene.snapshot["main_attention"] = [row for row in attention if row["task_id"] == first]
    rw.open_app({**ui, "page": page}, scene, theme=theme)
    if variant == "A":
        expect(page.locator(action_row(first))).to_be_attached()
        scene.snapshot["main_attention"] = attention
        scene.refresh()
    expect(page.locator("html")).to_have_attribute("data-theme", theme)
    passage = page.locator(f'#chat-messages [data-client-message-id="{PASSAGE}"]')
    passage.wait_for(state="attached")
    passage.evaluate("el => el.scrollIntoView({block:'center'})")
    rw.settle_frames(page)
    return page, scene


CASES = [
    ("a", "A", "loaded"), ("b", "B", "loaded"), ("now", "Now", "loaded"),
    ("a-archive", "A", "archived"), ("b-archive", "B", "archived"),
    ("a-helper", "A", "loaded"), ("b-helper", "B", "loaded"), ("b-absent", "B", "absent"),
]


@pytest.mark.parametrize("name,variant,history_mode", CASES, ids=[case[0] for case in CASES])
def test_main(subscription_ui, theme, name, variant, history_mode):
    helper = name.endswith("-helper")
    target = CHILD if helper else ROOT
    page, scene = setup(subscription_ui, theme, variant, history_mode, helper_first=helper)
    frames = Frames(page, theme, name)
    pill, card = page.locator(PILL), page.locator(CARD)
    resume = card.locator("[data-resume-run]")
    row = page.locator(action_row(target))
    own_rows = page.locator("#chat-messages .owner-action")
    baseline_reads = len(scene.history_log)
    if variant == "Now":
        trigger = rw.trigger(page)
        frames.add("В исходном Main статус комнаты открывает список работ.", trigger, "Открыть работы Main")
        expect(pill).to_be_hidden()
        expect(own_rows).to_have_count(0)
        expect(card).to_have_count(1)
        trigger.click()
        go = rw.row_main(page, ROOT)
        expect(go).to_have_attribute("data-target", "card")
        frames.add("Открой исходную карточку через список работ.", go, "К карточке")
        go.click()
    else:
        frames.add("Две работы ждут действия. Нажми плашку над полем сообщения.", pill, "Показать ожидающее действие")
        expect(pill.locator(".chat-question-nav-count")).to_have_text("2")
        expect(pill.locator(".chat-question-nav-word")).to_have_text("actions")
        expect(own_rows).to_have_count(2 if variant == "A" else 0)
        expect(card).to_have_count(int(history_mode == "loaded"))
        reading_top = page.locator("#chat-messages").evaluate("el => el.scrollTop")
        pill.click()
        if variant == "A":
            expect(row).to_be_focused()
            resume = row.locator("[data-resume-run]")
        elif helper:
            control = page.locator(HELPER)
            expect(control).to_be_focused()
            assert control.get_attribute("data-root") is None
            frames.add(f"В Activity выбран помощник «{NAMES[CHILD]}».", control, "Открыть действия помощника")
            control.click()
            resume = page.locator('body > .task-control-menu [data-task-control="resume"]')
            expect(resume).to_have_attribute("role", "menuitem")
        elif history_mode == "absent":
            expect(page.get_by_text("No message for this task in Main.", exact=True)).to_be_visible()
            expect(card).to_have_count(0)
            assert abs(page.locator("#chat-messages").evaluate("el => el.scrollTop") - reading_top) <= 1
            frames.add("В истории нет этой карточки. Main сообщает об этом и оставляет чтение на месте.")
        else:
            expect(resume).to_be_focused()

    reads = scene.history_log[baseline_reads:]
    lookup = variant == "B" and history_mode != "loaded"
    anchor = ("absent" if history_mode == "absent" else "found") if lookup else "not_needed"
    assert [r for r in reads if r["around"]] == (
        [{"chat": 1, "around": f"task:{ROOT}", "cursor": "", "result": anchor}] if lookup else [])
    assert not any(r["cursor"] for r in reads)
    assert not scene.posts_wire and not any(method != "GET" for method, _ in scene.requests)
    expect(own_rows).to_have_count(2 if variant == "A" else 0)
    if history_mode == "absent":
        frames.finish(anchor=anchor)
        return

    expect(resume).to_be_enabled()
    expect(resume).to_have_text("Resume")
    if variant == "A":
        assert history_mode == "loaded" or card.count() == 0
        expect(row.locator(".model-wait-heading strong")).to_have_text(NAMES[target])
        description = f"В конце переписки появилась System-строка «{NAMES[target]}» с Resume."
    elif helper:
        description = "Это меню одного помощника. Resume продолжит только его."
    else:
        description = f"Resume находится в исходной карточке «{NAMES[ROOT]}»."
    frames.add(description, resume, "Нажать Resume")
    path = f"/api/tasks/{target}/resume"
    with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(path)) as reply:
        resume.click()
    assert reply.value.ok
    expect(page.get_by_text("Resuming: the task returns to the queue.", exact=True)).to_be_visible()
    frames.add("Фикстура подтвердила Resume. Адрес запроса показан под кадром.")
    assert scene.posts_wire == [("POST", path, "", None)]
    assert [(m, p) for m, p in scene.requests if m != "GET"] == [("POST", path)]
    frames.finish(anchor=anchor, resume={"path": path, "scope": scene.controls[target].get("scope") or "root"})


@pytest.mark.parametrize("variant", ["Now", "A", "B"])
@pytest.mark.parametrize("kind", ["mirror", "native"])
def test_question_preserves_navigation(subscription_ui, variant, kind):
    page, scene = setup(subscription_ui, "dark", variant, questions=kind)
    task, quiz = (PROJECT_TASK, PROJECT_QUIZ) if kind == "mirror" else (NATIVE_TASK, NATIVE_QUIZ)
    selector = f'.chat-quiz-card[data-task-id="{task}"][data-quiz-id="{quiz}"]'
    card = page.locator("#chat-messages " + selector)
    expect(card).to_have_count(1)
    page.locator(PILL).click()
    expect(card.locator(".chat-quiz-question")).to_be_focused()
    if kind == "mirror":
        card.locator("xpath=ancestor::*[contains(@class,'chat-bubble')][1]").locator(".chat-quiz-project").click()
        original = page.locator("#project-panel " + selector)
        expect(original).to_be_visible()
        assert original.locator(".chat-quiz-question").inner_text() == card.locator(".chat-quiz-question").inner_text()
        expect(card).to_have_count(1)
    else:
        expect(card.locator(".chat-quiz-comment")).not_to_be_focused()
    assert not scene.posts_wire


@pytest.mark.parametrize('held', [False, True], ids=['queued', 'held'])
@pytest.mark.parametrize('proposed', [False, True], ids=['stock', 'proposal'])
def test_evolution_status(subscription_ui, theme, held, proposed, monkeypatch):
    """The same replenished balance with and without an existing explicit pause."""
    from supervisor import queue
    from tests import test_evolution_browser as evo

    item = {'id': 'atlas-cycle', 'type': 'evolution', 'chat_id': 1}
    if held:
        item['_budget_pause'] = {'reason': 'budget', 'status': 'paused_before_dispatch', 'scope': 'global',
                                 'physical_calls': 0, 'auto_resume': False, 'replay_safe': True,
                                 'resume_policy': 'manual_same_generation'}
    monkeypatch.setattr(queue, 'PENDING', [item])
    monkeypatch.setattr(queue, 'RUNNING', {})
    monkeypatch.setattr(queue, 'load_state', lambda: {'evolution_mode_enabled': True, 'owner_chat_id': 1,
                                                    'evolution_cycle': 12})
    monkeypatch.setattr(queue, 'budget_remaining', lambda *_a, **_k: 25.0)
    monkeypatch.setattr(queue, '_read_evolution_campaign', lambda: {'objective': 'Improve the atlas index',
                                                                  'cycles_done': 3, 'absorbed_cycles_done': 1})
    stock = queue.get_evolution_status_snapshot()
    assert (stock['status'], stock['queued_task_id'], stock['budget_remaining_usd']) == ('queued', item['id'], 25.0)
    shown = dict(stock)
    if proposed and held:
        shown.update(status='budget_stopped', detail='This cycle is paused at a spending limit. To continue, resume the task in Activity.')
    state = {**evo.STATE, 'evolution_enabled': True, 'evolution_state': shown}
    page = subscription_ui['page']
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    page = evo.open_evolution(subscription_ui, evo.history(), state=state, scheme=theme, width=1100, height=722)
    chip = page.locator('#evo-campaign-chip')
    expect(page.locator('html')).to_have_attribute('data-theme', theme)
    expect(chip).to_have_text('Budget' if proposed and held else 'Queued')
    expect(chip).to_have_attribute('data-tone', 'warn' if proposed and held else 'ok')
    expect(page.locator('#evo-campaign-note')).to_have_text(shown['detail'])
    campaign = page.get_by_role('region', name='Evolution campaign', exact=True)
    expect(campaign).to_be_visible()
    expect(campaign.get_by_role('button', name='Resume', exact=True)).to_have_count(0)
    assert not any(path.endswith('/resume') for path, _ in subscription_ui['posts'])
    scene_id = ('evo-held' if held else 'evo-queue') + ('-design' if proposed else '')
    frames = Frames(page, theme, scene_id)
    frames.add(('У цикла уже есть денежная пауза. ' if held else 'Обычная очередь, денежной паузы нет. ')
               + ('Предложенный статус: ' if proposed else 'Нынешний статус: ') + chip.inner_text() + '.')
    frames.finish(anchor='not_needed')
