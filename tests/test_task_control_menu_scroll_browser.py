"""A task-control menu is dismissed by a scroll that can have moved its trigger, and by no other.

The room's work list opens its row menu as a portal over a floating sheet while the chat feed behind the sheet re-anchors on
its own (a late history page, a text refit). A scroll of that feed cannot move the trigger, yet it dismissed the menu within
a millisecond of opening it (CI: test_real_row_hurry_wrap_up_stop_and_restart_with_page_open, the menu never visible). A
scroller that holds the trigger, and the page itself, still dismiss (owner decision 1D, test_s3_task_control_browser.py:
"only scroll that can stale anchor geometry invokes 1D close").

The real module in the real SPA page (the static-server/API fixture of test_subscription_setup_browser), in both engines;
no runtime. The menus open from real pointer clicks, the container scroll is a real wheel, and every state is screenshotted
(OUROBOROS_UI_EVIDENCE_DIR). The first check is the one that fails without the fix; the other two keep what must still close.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

# A scroller behind a floating sheet (the chat feed), the sheet holding a trigger, and a scroller that holds its trigger.
STAGE = """async () => {
    const { openTaskControlMenu } = await import('/modules/task_control_menu.js');
    const box = (id, css, html) => {
        const node = document.createElement('div');
        node.id = id; node.style.cssText = css; node.innerHTML = html;
        document.body.append(node);
        return node;
    };
    const feed = box('ob-feed', 'position:fixed;inset:0;overflow:auto;z-index:1', '<div style="height:4000px"></div>');
    box('ob-sheet', 'position:fixed;top:60px;left:60px;width:260px;height:60px;z-index:60;background:#222',
        '<button id="ob-trigger" style="margin:12px">trigger</button>');
    const holder = box('ob-holder', 'position:fixed;top:200px;left:60px;width:260px;height:80px;overflow:auto;z-index:60;'
        + 'background:#222', '<div style="height:400px;padding:12px"><button id="ob-inner">inner</button></div>');
    for (const id of ['ob-trigger', 'ob-inner']) {
        const button = document.getElementById(id);
        button.addEventListener('click', () => openTaskControlMenu(button, { actions: ['hurry', 'stop_now'], onAction() {} }));
    }
    window.__ob = { scrolls: { feed: 0, holder: 0 }, menus: () => document.querySelectorAll('body > .task-control-menu').length };
    feed.addEventListener('scroll', () => { window.__ob.scrolls.feed += 1; });
    holder.addEventListener('scroll', () => { window.__ob.scrolls.holder += 1; });
}"""
# Two frames and a turn after a scroll: the scroll event it queued has been dispatched.
SETTLED = "() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(done, 60))))"
MENU = "body > .task-control-menu"


@pytest.fixture(params=["chromium", "webkit"])
def staged(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    page = ui["page"]
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    page.evaluate(STAGE)
    return page, request.param


def _shot(page, engine, name):
    setup_browser.capture(page, f"menu-scroll-{engine}-{name}")


def test_a_scroll_that_cannot_have_moved_the_trigger_leaves_its_menu_open(staged):
    page, engine = staged
    page.locator("#ob-trigger").click()
    page.locator(MENU).wait_for(state="visible")
    _shot(page, engine, "1-open")
    # The feed behind the sheet re-anchors (it holds no trigger): the menu stays, and so does its trigger's state.
    page.evaluate("() => { document.getElementById('ob-feed').scrollTop = 240; }")
    page.evaluate(SETTLED)
    _shot(page, engine, "2-after-the-feed-scrolled")
    assert page.evaluate("() => window.__ob.scrolls.feed") >= 1, "the scroll under test never fired"
    assert page.evaluate("() => window.__ob.menus()") == 1, "a scroll that cannot move the trigger dismissed its menu"
    assert page.locator("#ob-trigger").get_attribute("aria-expanded") == "true"
    assert page.locator(MENU).is_visible()


def test_a_scroll_of_the_container_holding_the_trigger_still_dismisses_the_menu(staged):
    page, engine = staged
    page.locator("#ob-inner").click()
    page.locator(MENU).wait_for(state="visible")
    _shot(page, engine, "3-open-inside-a-scroller")
    box = page.locator("#ob-holder").bounding_box()
    # A wheel over the holder's top-right corner (clear of the menu, which opens beneath its trigger at the left).
    page.mouse.move(box["x"] + box["width"] - 10, box["y"] + 10)
    page.mouse.wheel(0, 60)
    page.locator(MENU).wait_for(state="detached")
    _shot(page, engine, "4-after-the-holder-scrolled")
    assert page.evaluate("() => window.__ob.scrolls.holder") >= 1, "the scroll under test never fired"
    assert page.locator("#ob-inner").get_attribute("aria-expanded") == "false"


def test_a_page_scroll_still_dismisses_the_menu(staged):
    page, engine = staged
    page.locator("#ob-trigger").click()
    page.locator(MENU).wait_for(state="visible")
    page.evaluate("() => document.dispatchEvent(new Event('scroll'))")  # the document is the event's target
    page.locator(MENU).wait_for(state="detached")
    _shot(page, engine, "5-after-a-page-scroll")
    assert page.locator("#ob-trigger").get_attribute("aria-expanded") == "false"


def test_popups_scroll_only_their_content_and_keep_their_actions(staged):
    page, engine = staged
    page.set_viewport_size({"width": 390, "height": 244})
    page.evaluate("""async () => {
        const {openTaskControlMenu} = await import('/modules/task_control_menu.js');
        const {bindMenu} = await import('/modules/ui_interactions.js');
        const holder = document.getElementById('ob-holder');
        holder.style.top = '105px';
        const launcher = document.createElement('button');
        launcher.id = 'ob-task'; launcher.textContent = 'Task options';
        holder.firstElementChild.replaceChildren(launcher);
        launcher.onclick = () => openTaskControlMenu(launcher, {
            actions: ['finalize', 'hurry', 'pause', 'stop_earlier', 'stop_now', 'resume'],
            onAction: action => window.chosen = action,
        });
        const ordinary = document.createElement('button');
        ordinary.id = 'ob-ordinary'; ordinary.textContent = 'View options';
        document.getElementById('ob-sheet').append(ordinary);
        ordinary.onclick = () => {
            const menu = document.createElement('div');
            menu.className = 'ui-popup'; menu.setAttribute('role', 'menu');
            menu.setAttribute('aria-label', 'Evidence options');
            for (let n = 1; n <= 12; n++) {
                const row = document.createElement('button'); row.setAttribute('role', 'menuitem');
                row.textContent = `Review source ${n} and the notes its owner kept beside it`;
                row.onclick = () => { window.chosen = row.textContent; binding.close({restoreFocus:true}); };
                menu.append(row);
            }
            document.body.append(menu);
            const binding = bindMenu(menu, {anchor: ordinary, onClose: () => menu.remove()});
        };
    }""")
    # WebKit delivers the viewport's resize after set_viewport_size returns; task menus dismiss on resize.
    page.wait_for_function('innerWidth === 390 && innerHeight === 244')
    page.evaluate(SETTLED)
    for theme, opener, selector, action, answer in [
        ('dark', '#ob-task', MENU, '[data-task-control="resume"]', 'resume'),
        ('light', '#ob-ordinary', 'body > [aria-label="Evidence options"]:not([inert])',
         '[role="menuitem"]:last-child', 'Review source 12 and the notes its owner kept beside it'),
    ]:
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        button = page.locator(opener)
        button.focus()
        page.keyboard.press('Enter')
        popup = page.locator(selector)
        popup.wait_for()
        content = popup.locator(':scope > .ui-popup-body[data-scroll-below]')
        content.wait_for(timeout=5000)
        assert popup.evaluate("el => getComputedStyle(el).maskImage") == 'none'
        assert content.evaluate("el => getComputedStyle(el).maskImage.includes('linear-gradient')")
        if theme == 'light':  # the shared menu at its capped width: its rows wrap there and stay inside its own scrollbar
            fit = content.evaluate("""el => [el.scrollWidth - el.clientWidth, el.parentElement.offsetWidth,
                parseFloat(el.parentElement.style.getPropertyValue('--ui-popup-max-width'))]""")
            assert fit[0] <= 0 and fit[1] >= fit[2] - 1, (engine, 'no sideways scroll at the capped width', fit)
        _shot(page, engine, f'painted-{theme}')
        box = content.element_handle()
        if theme == 'light':
            page.keyboard.press('End')
        else:
            box.evaluate('el => { el.scrollTop = el.scrollHeight; }')  # menu must not dismiss while its contents scroll
        page.wait_for_function("el => el.hasAttribute('data-scroll-above') && !el.hasAttribute('data-scroll-below')", arg=box)
        option = popup.locator(action)
        assert option.evaluate("el => { const b = el.parentElement.getBoundingClientRect(), r = el.getBoundingClientRect();"
                               " return r.top >= b.top && r.bottom <= b.bottom + 1; }")
        option.click()
        assert page.evaluate('window.chosen') == answer
        popup.wait_for(state='detached')
        assert button.evaluate('el => document.activeElement === el'), theme
        assert box.evaluate("el => !el.hasAttribute('data-scroll-above') && !el.hasAttribute('data-scroll-below')")


def test_open_source_picker_gains_its_faded_edge_when_catalog_rows_arrive(staged):
    page, engine = staged
    page.set_viewport_size({"width": 390, "height": 270})
    page.evaluate("""async () => {
        const {bindSourcePickers} = await import('/modules/source_picker.js');
        const field = document.getElementById('ob-sheet');
        field.innerHTML = '<label class="ui-field"><select data-model-role-source aria-label="Model source">'
                        + '<option value="auto">Auto</option><option value="direct">Direct model access</option></select></label>';
        window.releasePicker = bindSourcePickers(field);
    }""")
    page.wait_for_function('innerWidth === 390 && innerHeight === 270')
    page.evaluate(SETTLED)
    button = page.get_by_role('combobox', name='Model source')
    button.click()
    popup = page.get_by_role('listbox', name='Model source')
    popup.get_by_role('option', name='Direct model access').wait_for()
    assert popup.evaluate("el => !el.querySelector('[data-scroll-below]') && getComputedStyle(el).maskImage === 'none'")
    popup_handle = popup.element_handle()
    page.evaluate("""() => {
        const sources = document.querySelector('[data-model-role-source]');
        for (let n = 1; n <= 25; n++) sources.add(new Option(`Additional source ${n}`, `source-${n}`));
    }""")
    content = popup.locator(':scope > .ui-popup-body[data-scroll-below]')
    content.wait_for()
    assert popup.evaluate('(el, old) => el === old', popup_handle)
    assert popup.evaluate("el => getComputedStyle(el).maskImage === 'none' && el.getBoundingClientRect().bottom < innerHeight")
    _shot(page, engine, 'late-catalog')
    box = content.element_handle()
    button.press('End')
    last = popup.get_by_role('option', name='Additional source 25')
    assert button.get_attribute('aria-activedescendant') == last.get_attribute('id')
    last.wait_for()
    page.wait_for_function("el => !el.hasAttribute('data-scroll-below')", arg=box)
    button.press('Enter')
    assert page.locator('[data-model-role-source]').input_value() == 'source-25'
    popup.wait_for(state='detached')
    assert button.evaluate('el => el === document.activeElement')
    page.evaluate('window.releasePicker()')


@pytest.fixture
def nested_fades(staged):
    page, _engine = staged
    page.set_viewport_size({"width": 390, "height": 600})
    page.evaluate("""async () => {
        for (const id of ['ob-feed', 'ob-sheet', 'ob-holder']) document.getElementById(id).remove();
        const {createLiveCardTimelineRenderer} = await import('/modules/chat_render_batch.js');
        const {buildTimelineItemHtml} = await import('/modules/chat_activity.js');
        const {createChatMedia} = await import('/modules/chat_media.js');
        const {createTextBubbles} = await import('/modules/chat_text_bubble.js');
        const {createReplyQuote, releaseQuoteFades} = await import('/modules/chat_reply_quote.js');
        const {createReviewPresentationController, reviewGroupsFromTaskDetail} = await import('/modules/review_presentation.js');
        const host = document.createElement('div'); host.id = 'nested-fade-fixture';
        document.getElementById('chat-messages').append(host);
        const prose = 'The reviewer checked the same saved source and set aside one unrelated observation.\\n\\n'.repeat(45);
        const mount = () => {
            const root = document.createElement('div');
            root.innerHTML = '<div data-expanded="1"><div data-timeline></div></div><button data-summary>Reviews</button><div data-reviews></div>';
            host.append(root);
            const record = { root: root.firstElementChild, timelineEl: root.querySelector('[data-timeline]'), groupId:'notes',
                expandedLineKeys:new Set(['full']), items:[{lineKey:'full', phase:'warn', headline:'Saved review notes',
                    body:'Preview', fullRef:'test-task', truncated:true, fetchedFull:prose, bodyFormat:'text'}] };
            const writer = createLiveCardTimelineRenderer({ withStableViewport: write => write(), buildTimelineItemHtml });
            writer.renderLiveCardTimeline(record);
            const media = createChatMedia({ chatSessionId:'nested-life', insertMessageNode:node=>root.append(node), senderLabel:()=> 'System' });
            const draw = createTextBubbles({ chatMedia:media, replies:{ decorate() {} }, feed:root });
            const { bubble } = draw('Skill review: lookup exercises\\n\\n' + prose, 'system', {meta:{systemType:'skill_review'}});
            root.append(bubble);
            bubble.querySelector('[data-skill-review-toggle]').click();
            const quote = createReplyQuote({ address:'row:1@2026-09-16T00:00:00Z#0123456789ab', role:'user',
                excerpt:{ start_utf16:0, text:'Copied from the visible record. '.repeat(60) } },
                { onJump: ref => { (window.quoteJumps ??= []).push(ref.address); } });
            root.append(quote);
            quote.querySelector('.chat-clamp-toggle').click();
            const review = createReviewPresentationController({ host:root.querySelector('[data-reviews]'), summary:root.querySelector('[data-summary]'),
                disclosure:{ sectionExpanded:true, expandedGroups:new Set(['review_record:notes:record:stored']) } });
            review.updateMany(reviewGroupsFromTaskDetail({task_id:'notes', review_projection:{ panels:[{
                record_id:'stored', surface:'commit_gate', record_state:'settled', revision:1, aggregate_signal:'PASS',
                actors:Array.from({length:14}, (_, n) => ({slot_id:`seat-${n}`, model:`example/model-${n}`,
                    observed_model:`example/model-${n}`, transport_status:'success', parse_status:'valid', semantic_verdict:'PASS'}))
            }] }}));
            root.querySelector('.review-facts').open = true;
            const boxes = ['.chat-live-line-body-full', '.skill-review-full', '.chat-quote-text', '.review-facts-body']
                .map(selector => root.querySelector(selector));
            return { boxes, patchOutput() {
                const item = record.items[0];
                item.fetchedFull += '\\nA late line.';
                writer.patchLastTimelineItem(item, record);
            }, retire() { writer.releaseTimeline(record); review.destroy(); releaseQuoteFades(root);
                media.release(bubble); media.destroy(); root.remove(); } };
        };
        window.nestedLife = {current:mount(), mount};
    }""")
    ready = "() => nestedLife.current.boxes.every(el => el?.hasAttribute('data-scroll-below'))"
    page.wait_for_function(ready)
    yield staged
    page.evaluate("() => nestedLife.current.retire()")


def test_nested_fade_owners_release_live_output_review_and_quote_before_remount(nested_fades):
    page, _engine = nested_fades
    page.evaluate("""() => { const first = nestedLife.current; nestedLife.old = first.boxes;
        first.retire();
        // Keep the exact old scrollers alive long enough to see any subscriber repaint in response to a scroll.
        nestedLife.old.forEach(el => { el.scrollTop = 50; el.dispatchEvent(new Event('scroll')); });
        nestedLife.current = nestedLife.mount();
    }""")
    page.wait_for_function("() => nestedLife.current.boxes.every(el => el?.hasAttribute('data-scroll-below'))")
    assert page.evaluate("() => nestedLife.old.every(el => !el.isConnected && !el.hasAttribute('data-scroll-below'))")


@pytest.mark.parametrize(("selector", "label"), [
    (".chat-live-line-body-full", "Full task output"),
    (".skill-review-full", "Skill review text"),
    (".chat-quote-text", "Selected quotation"),
], ids=["task-output", "skill-review", "quotation"])
def test_nested_fade_body_is_tab_reachable_only_when_it_scrolls(nested_fades, selector, label):
    page, engine = nested_fades
    box = page.locator(f"#nested-fade-fixture {selector}")
    bounds = """el => {
        const before = document.createElement('button'); before.id = 'scroll-start'; before.textContent = 'Before';
        const after = document.createElement('button'); after.id = 'scroll-end'; after.textContent = 'After';
        el.before(before); el.after(after); before.focus(); el.scrollTop = 0;
    }"""
    box.evaluate(bounds)
    assert box.get_attribute("role") == "region" and box.get_attribute("aria-label") == label
    tab = "Alt+Tab" if engine == "webkit" else "Tab"
    page.keyboard.press(tab)
    assert box.evaluate("el => document.activeElement === el && el.matches(':focus-visible') && el.scrollTop === 0")
    page.keyboard.press("PageDown")
    page.wait_for_function("selector => document.querySelector('#nested-fade-fixture ' + selector).scrollTop > 0", arg=selector, timeout=5000)
    _shot(page, engine, f"{selector[1:]}-keyboard")
    if selector == ".chat-live-line-body-full":
        page.evaluate("() => { document.getElementById('scroll-start').remove(); document.getElementById('scroll-end').remove(); }")
        box.evaluate("el => { window.sameOutput = el; nestedLife.current.patchOutput(); }")
        assert box.evaluate("el => window.sameOutput === el && document.activeElement === el && el.tabIndex === 0")
        box.evaluate(bounds)
    box.evaluate("el => { el.style.maxBlockSize = 'none'; }")
    page.wait_for_function("selector => !document.querySelector('#nested-fade-fixture ' + selector).hasAttribute('tabindex')", arg=selector)
    assert box.get_attribute("role") is None and box.get_attribute("aria-label") is None
    page.locator("#scroll-start").focus()
    page.keyboard.press(tab)
    assert page.locator("#scroll-end").evaluate("el => document.activeElement === el")


def test_expanded_quote_jump_control_is_separate_and_keeps_both_activations(nested_fades):
    page, engine = nested_fades
    quote = page.locator("#nested-fade-fixture .chat-quote[data-excerpt]")
    jump = quote.locator(".chat-quote-main")
    passage = quote.locator(".chat-quote-text")
    assert passage.evaluate("el => !el.closest('button')")
    passage.click(position={"x": 10, "y": 10})
    jump.click()
    jump.focus()
    page.keyboard.press("Enter")
    assert page.evaluate("() => window.quoteJumps") == ["row:1@2026-09-16T00:00:00Z#0123456789ab"] * 3
    # A drag that selected words is a selection, not a jump: the click a drag ends with keeps the words selected.
    box = passage.bounding_box()
    page.mouse.move(box["x"] + 6, box["y"] + 12)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] - 10, box["y"] + 12, steps=12)
    page.mouse.up()
    assert page.evaluate("() => getSelection().toString()").strip(), "the drag selected no words to read"
    assert page.evaluate("() => window.quoteJumps.length") == 3, "a drag that selected words jumped"
    _shot(page, engine, "quote-drag-selection")
    # A plain press on the words, with nothing selected, still jumps.
    page.evaluate("() => getSelection().removeAllRanges()")
    passage.click(position={"x": 10, "y": 10})
    assert page.evaluate("() => window.quoteJumps.length") == 4


def test_plain_confirm_fade_region_only_joins_tab_cycle_when_long(staged):
    page, engine = staged
    page.set_viewport_size({'width': 390, 'height': 480})
    page.evaluate("""async () => {
        const {openConfirmDialog} = await import('/modules/confirm_dialog.js');
        window.showConfirm = (body) => { window.confirmResult = openConfirmDialog({title:'Confirm example', body}); };
        window.showConfirm('A long explanation should be readable before deciding. '.repeat(140));
    }""")
    dialog = page.get_by_role('dialog', name='Confirm example')
    body = dialog.locator('.ui-modal-body')
    ok = dialog.get_by_role('button', name='Continue')
    assert ok.evaluate('el => document.activeElement === el')
    page.wait_for_function("() => document.querySelector('.confirm-dialog .ui-modal-body').hasAttribute('data-scroll-below')")
    assert body.get_attribute('role') == 'region' and body.get_attribute('aria-label') == 'Confirmation text'
    page.keyboard.press('Alt+Tab' if engine == 'webkit' else 'Tab')
    assert body.evaluate('el => document.activeElement === el && el.scrollTop === 0')
    page.keyboard.press('PageDown')
    page.wait_for_function("() => document.querySelector('.confirm-dialog .ui-modal-body').scrollTop > 0", timeout=5000)
    page.keyboard.press('Escape')
    assert page.evaluate('window.confirmResult') is False
    page.evaluate("window.showConfirm('Short text.')")
    page.evaluate(SETTLED)
    assert body.get_attribute('tabindex') is None and body.get_attribute('aria-label') is None
    assert ok.evaluate('el => document.activeElement === el')
    page.keyboard.press('Escape')


def test_model_citation_fade_region_scrolls_when_expanded(staged):
    page, engine = staged
    page.evaluate("""async () => {
        for (const id of ['ob-feed', 'ob-sheet', 'ob-holder']) document.getElementById(id).remove();
        const {paintCardModel, openCardModel} = await import('/modules/card_model.js');
        const {createModelCatalogStore} = await import('/modules/model_catalog_store.js');
        const metaEl = document.createElement('div');
        document.getElementById('chat-messages').append(metaEl);
        const model = 'z-ai/glm-5.3';
        const store = createModelCatalogStore({readCatalog:async () => ({items:[]}), readPreferences:async () => ({}),
            readSettings:async () => ({OUROBOROS_MODEL:model}), readProcessing:async () => ({})});
        const record = {groupId:'cited', metaEl};
        const lead = {model, basis:'owner_words', cited:{text:'The original model choice is preserved here. '.repeat(220)}};
        paintCardModel(record, {target:'cited', lead, main:{model}}, {store, coarse:() => false,
            place:() => ({reposition(){}, destroy(){}})});
        const { popup } = openCardModel(record);
        // The other tests mount the sheet via the room. This isolated mount supplies its placement only.
        popup.style.cssText = 'position:fixed; top:40px; left:40px; width:320px; --ui-popup-max-height:360px;';
    }""")
    sheet = page.locator('.model-sheet')
    line = sheet.locator('.card-model-pop-line')
    toggle = sheet.locator('.chat-clamp-toggle')
    assert line.get_attribute('tabindex') is None
    line.evaluate("el => { el.style.width = '270px'; }")
    page.wait_for_function("() => !document.querySelector('.model-sheet .chat-clamp-toggle').hidden", timeout=5000)
    toggle.click(timeout=5000)
    line.evaluate("""el => {
        const before = document.createElement('button'); before.textContent='Before';
        el.before(before); before.focus(); el.scrollTop=0;
    }""")
    page.wait_for_function("() => document.querySelector('.card-model-pop-line').getAttribute('aria-label') === 'Quoted model choice'", timeout=5000)
    page.keyboard.press('Alt+Tab' if engine == 'webkit' else 'Tab')
    assert line.evaluate('el => el === document.activeElement && el.scrollTop === 0')
    page.keyboard.press('PageDown')
    page.wait_for_function("() => document.querySelector('.card-model-pop-line').scrollTop > 0", timeout=5000)
    toggle.click()
    assert line.get_attribute('tabindex') is None and line.evaluate('el => el.scrollTop === 0')


def test_nested_fade_region_focus_ring_clears_its_mask(nested_fades):
    page, engine = nested_fades
    for selector in (".chat-live-line-body-full", ".skill-review-full", ".chat-quote-text"):
        box = page.locator(f"#nested-fade-fixture {selector}")
        assert box.evaluate("el => getComputedStyle(el).maskImage !== 'none' && el.scrollTop === 0")
        box.evaluate("""el => {
            const before = document.createElement('button'); before.textContent = 'Before';
            el.before(before); before.focus();
        }""")
        page.keyboard.press("Alt+Tab" if engine == "webkit" else "Tab")
        assert box.evaluate("""el => { const style = getComputedStyle(el);
            return document.activeElement === el && el.matches(':focus-visible') && el.hasAttribute('data-scroll-below')
                && style.outlineStyle !== 'none' && parseFloat(style.outlineWidth) > 0 && style.maskImage === 'none';
        }""")
        _shot(page, engine, f"{selector[1:]}-focus-ring")
        box.evaluate("el => el.previousElementSibling.remove()")
