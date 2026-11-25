"""Passage coordinates, visibility and bidi selection in the real chat renderer."""
import json

import pytest

from tests.test_project_work_card_browser import project_chat as project_chat

pytestmark = [pytest.mark.ui_browser, pytest.mark.parametrize(
    "project_chat", [("chromium", "light", 1100), ("webkit", "dark", 1100)], indirect=True,
    ids=["chromium", "webkit"])]


def row(n, words, **extra):
    ts = extra.pop("ts", f"2026-10-01T17:00:{n:02d}Z")
    return {"role": "assistant", "text": words, "chat_id": 1, "ts": ts, "history_id": f"excerpt-{n}",
            "row_address": f"row:1@{ts}#{n:012x}", **extra}


def ref(source, excerpt=None):
    result = {"address": source["row_address"], "role": "assistant", "chat_id": 1, "snippet": "Source"}
    if excerpt is not None:
        result["excerpt"] = excerpt
    return result


def bubble(page, source):
    return page.locator(f'#chat-messages .chat-bubble[data-row-address={json.dumps(source["row_address"])}]')


def select_contents(locator):
    locator.scroll_into_view_if_needed()
    locator.evaluate("""node => { const r = document.createRange(); r.selectNodeContents(node);
        getSelection().removeAllRanges(); getSelection().addRange(r); }""")


def test_rendered_mapping_is_reused_until_the_body_changes(project_chat):
    page = project_chat["page"]
    source = row(1, "North **trail**.")
    project_chat["rows"][:] = [source]
    page.reload()
    facts = bubble(page, source).evaluate("""async node => {
        const {readExcerpt, excerptRange} = await import('/static/modules/chat_reply_excerpt.js');
        const body = node.querySelector('.message'), range = document.createRange();
        const original = window.getComputedStyle;
        let reads = 0;
        window.getComputedStyle = (...args) => { reads++; return original(...args); };
        try {
            range.selectNodeContents(body);
            const first = readExcerpt(node, range), built = reads;
            const second = readExcerpt(node, range), landed = excerptRange(node, first)?.toString(), reused = reads;
            body.querySelector('strong').firstChild.data = 'gate';
            range.selectNodeContents(body);
            const changed = readExcerpt(node, range), rebuilt = reads;
            return {first, second, landed, built, reused, changed, rebuilt};
        } finally { window.getComputedStyle = original; }
    }""")
    assert facts["first"] == facts["second"] == {"text": "North trail.", "start_utf16": 0}
    assert facts["landed"] == "North trail." and facts["built"] > 0
    assert facts["reused"] == facts["built"] and facts["rebuilt"] > facts["reused"]
    assert facts["changed"] == {"text": "North gate.", "start_utf16": 0}


@pytest.mark.parametrize("media", [False, True], ids=["words", "caption"])
def test_live_and_reloaded_selection_keep_the_host_recorded_speaker(project_chat, media):
    page = project_chat["page"]
    source = row(1, "Meet at noon.", role="user", source="presence:telegram", reply_author_label="Mira",
                 **({"msg_type": "photo", "caption": "Meet at noon."} if media else {}))
    page.evaluate("""([kind, source]) => emit(kind, {...source, content:source.text,
        ...(kind === 'photo' ? {image_base64:'aW1hZ2U=', mime:'image/png'} : {})})""", ["photo" if media else "chat", source])
    for reloaded in (False, True):
        if reloaded:
            project_chat["rows"][:] = [source]
            page.reload()
        body = bubble(page, source).locator('[data-reply-caption]' if media else '.message[data-reply-body]')
        select_contents(body)
        page.get_by_role("button", name="Reply to quote", exact=True).click()
        staged = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
        assert staged["author_label"] == "Mira" and staged["excerpt"]["text"] == "Meet at noon."
        page.get_by_role("button", name="Cancel reply").click()


def test_selected_list_code_and_backward_link_drag(project_chat):
    page = project_chat["page"]
    items = row(1, "- first\n- second\n\n```\na\n  b\n```")
    prefix, words = "🌙 Guide: The ", "blue lamp by the east gate"
    proactive = row(2, "🌙 Guide: **The blue lamp** by [the east gate](https://example.invalid/harbor)",
                    system_type="proactive_message")
    project_chat["rows"][:] = [items, proactive]
    page.reload()
    body = bubble(page, items).locator(".message[data-reply-body]")
    body.locator("li").last.wait_for()
    facts = body.evaluate("""async body => {
        const {readExcerpt, excerptRange} = await import('/static/modules/chat_reply_excerpt.js');
        const target = body.closest('.chat-bubble');
        const [first, second] = body.querySelectorAll('li'), code = body.querySelector('pre code');
        const r = document.createRange();
        r.setStart(first.firstChild,0); r.setEnd(second.firstChild,6);
        const both = readExcerpt(target,r);
        r.selectNodeContents(second); const item = readExcerpt(target,r);
        r.selectNodeContents(code); const raw = readExcerpt(target,r);
        return {both,item, round:excerptRange(target,item).toString(), raw, code:code.textContent};
    }""")
    assert facts["both"] == {"text": "first\nsecond", "start_utf16": 0}
    assert facts["item"] == {"text": "second", "start_utf16": 6} and facts["round"] == "second"
    assert facts["raw"]["text"] == facts["code"] and "a\n  b" in facts["code"]

    body = bubble(page, proactive).locator(".message[data-reply-body]")
    body.scroll_into_view_if_needed()
    start, end = body.evaluate("""body => {
        // At the link's trailing caret, just outside its hit box, dragging selects text instead of dragging its URL.
        const link = body.querySelector('a').getBoundingClientRect();
        const strong = body.querySelector('strong').firstChild, i = strong.data.indexOf('blue');
        const r = document.createRange(); r.setStart(strong,i); r.setEnd(strong,i+1);
        const word = r.getBoundingClientRect();
        return [[link.right+.5,(link.top+link.bottom)/2], [word.left+.2,(word.top+word.bottom)/2]];
    }""")
    assert start[0] > end[0]
    page.mouse.move(*start)
    page.mouse.down()
    page.mouse.move(*end, steps=16)
    page.mouse.up()
    selected = body.evaluate("""body => {
        const s = getSelection();
        return [s.toString(), body.querySelector('a').contains(s.anchorNode), body.querySelector('strong').contains(s.focusNode)];
    }""")
    assert selected == [words, True, True]
    page.get_by_role("button", name="Reply to quote", exact=True).click()
    page.locator("#chat-input").fill("Tell me more.")
    page.locator("#chat-send").click()
    page.wait_for_function("sentFrames.some(f => f.reply_excerpt)")
    sent = page.evaluate("sentFrames.find(f => f.reply_excerpt)")
    assert sent["reply_to"] == proactive["row_address"]
    assert sent["reply_excerpt"] == {"text": words, "start_utf16": len(prefix.encode("utf-16-le")) // 2}


def test_folded_passage_and_replaced_highlights(project_chat):
    page = project_chat["page"]
    lines = [f"Observation {i:02}" for i in range(12)]
    source = row(1, "Finished", role="system", system_type="project_completion_summary", project_id="alpha",
                 project_name="Project Alpha", completion_answer="  \n".join(lines))
    first = {"text": lines[0], "start_utf16": 0}
    wide = {"text": "\n".join(lines), "start_utf16": 0}
    marked = row(2, "A reply to a visible line.", role="user", reply_to=ref(source, first))
    hidden = row(3, "A reply spanning the fold.", role="user", reply_to=ref(source, wide))
    whole = row(4, "A reply to the whole answer.", role="user", reply_to=ref(source))
    stale = row(5, "A reply whose words changed.", role="user",
                reply_to=ref(source, {"text": "An earlier version", "start_utf16": 0}))
    project_chat["rows"][:] = [source, marked, hidden, whole, stale]
    page.reload()
    target = bubble(page, source)
    page.wait_for_function("document.querySelector('#chat-messages .project-answer.is-folded')")
    facts = target.evaluate("""async target => {
        const {readExcerpt} = await import('/static/modules/chat_reply_excerpt.js');
        const body = target.querySelector('.message');
        const r = document.createRange(); r.selectNodeContents(body);
        const hidden = readExcerpt(target,r);
        const first = document.createTreeWalker(body,NodeFilter.SHOW_TEXT).nextNode();
        r.setStart(first,0); r.setEnd(first,first.length);
        return [hidden,readExcerpt(target,r)];
    }""")
    assert facts == [None, first]

    def jump(reply):
        bubble(page, reply).locator('.chat-quote-main').click()

    def visible_mark():
        page.wait_for_function("CSS.highlights.get('reply-excerpt')?.size === 1")

    jump(marked)
    visible_mark()
    jump(whole)
    assert page.evaluate("!CSS.highlights.has('reply-excerpt')")
    jump(marked)
    visible_mark()
    jump(hidden)
    page.wait_for_function("document.querySelector('#toast-stack')?.textContent.includes('folded part')")
    assert page.evaluate("!CSS.highlights.has('reply-excerpt')")
    assert target.evaluate("n => n.classList.contains('is-quote-target')")
    jump(marked)
    visible_mark()
    jump(stale)
    page.wait_for_function("document.querySelector('#toast-stack')?.textContent.includes('could not be located')")
    assert page.evaluate("!CSS.highlights.has('reply-excerpt')")

    # Browsers without Custom Highlight use a Selection: clear it on the next jump unless the reader replaced it.
    page.evaluate("Object.defineProperty(window, 'Highlight', {configurable:true,value:undefined})")
    jump(marked)
    page.wait_for_function("expected => getSelection().toString() === expected", arg=lines[0])
    # Keyboard activation keeps a real Selection in place until jump itself decides what it owns.
    whole_button = bubble(page, whole).locator('.chat-quote-main')
    whole_button.evaluate("el => el.click()")
    assert page.evaluate("getSelection().isCollapsed")
    jump(marked)
    page.wait_for_function("expected => getSelection().toString() === expected", arg=lines[0])
    bubble(page, whole).locator('.message').evaluate("""n => {
        const r = document.createRange(); r.selectNodeContents(n);
        getSelection().removeAllRanges(); getSelection().addRange(r);
    }""")
    whole_button.evaluate("el => el.click()")
    assert page.evaluate("getSelection().toString()") == whole["text"]


def test_visible_inline_offsets_and_return_of_button_after_scroll(project_chat):
    page = project_chat["page"]
    rich = row(1, "Find   **copper   and**    [copper](https://example.invalid/x)   again.")
    plain = row(2, "Find   copper   and    copper   again.", role="user")
    project_chat["rows"][:] = [rich, plain] + [row(n, "Filler for the scrollbar.") for n in range(3, 45)]
    page.reload()
    feed = page.locator('#chat-messages')
    body = bubble(page, rich).locator(".message")
    project_chat['expect'](bubble(page, project_chat['rows'][-1])).to_be_in_viewport()
    feed.hover()
    page.mouse.wheel(0, -feed.evaluate('el => el.scrollHeight'))
    project_chat['expect'](body).to_be_in_viewport()
    second = body.locator("a")
    second.evaluate("""node => { const r = document.createRange(); r.selectNodeContents(node);
        getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    action = page.get_by_role("button", name="Reply to quote", exact=True)
    action.wait_for()
    page.mouse.wheel(0, feed.evaluate('el => el.scrollHeight'))
    action.wait_for(state="hidden")
    project_chat['expect'](body).not_to_be_in_viewport()
    page.mouse.wheel(0, -feed.evaluate('el => el.scrollHeight'))
    project_chat['expect'](body).to_be_in_viewport()
    action.wait_for()
    action.click()
    assert page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply')).excerpt") == {
        "text": "copper", "start_utf16": 16}
    facts = page.evaluate("""async ([rich, plain]) => {
        const {readExcerpt, excerptRange} = await import('/static/modules/chat_reply_excerpt.js');
        const bubble = id => [...document.querySelectorAll('.chat-bubble')].find(b => b.dataset.rowAddress === id);
        const s = getSelection(), r = document.createRange(), target = bubble(rich), body = target.querySelector('.message');
        r.selectNodeContents(body.querySelector('p')); s.removeAllRanges(); s.addRange(r);
        const richText = [s.toString(), readExcerpt(target, r), excerptRange(target, {text:'copper', start_utf16:16}).startContainer.parentElement.tagName];
        const mine = bubble(plain).querySelector('.message'); r.selectNodeContents(mine);
        return [richText, readExcerpt(bubble(plain), r)];
    }""", [rich["row_address"], plain["row_address"]])
    assert facts == [["Find copper and copper again.", {"text": "Find copper and copper again.", "start_utf16": 0}, "A"],
                     {"text": plain["text"], "start_utf16": 0}]


def test_image_label_is_not_part_of_the_author_range(project_chat):
    page = project_chat['page']
    source = row(1, '![map](https://example.invalid/image.png) north')
    project_chat['rows'][:] = [source]
    page.reload()
    facts = bubble(page, source).evaluate("""async node => {
        const {readExcerpt, excerptRange} = await import('/static/modules/chat_reply_excerpt.js');
        const link = node.querySelector('.md-image-ref'), label = link.querySelector('.md-image-label');
        const r = document.createRange(), alt = link.lastChild, north = link.nextSibling;
        r.setStart(alt,0); r.setEnd(alt,alt.length);
        const map = readExcerpt(node,r);
        r.setStart(north,1); r.setEnd(north,north.length);
        const after = readExcerpt(node,r);
        r.selectNodeContents(label);
        return {shown:link.textContent, map, after, label:readExcerpt(node,r),
            landed:excerptRange(node, {text:'north', start_utf16:4})?.toString()};
    }""")
    assert facts == {'shown': 'Image: map', 'map': {'text': 'map', 'start_utf16': 0},
                     'after': {'text': 'north', 'start_utf16': 4}, 'label': None, 'landed': 'north'}


def test_caption_only_replay_keeps_live_excerpt_coordinates(project_chat):
    page = project_chat['page']
    caption = '🌙 Jetty  map'
    photo = row(1, 'Generated photo metadata', msg_type='photo', caption=caption)
    video = row(2, 'Generated video metadata', msg_type='video', caption=caption)
    selected = {'text': 'map', 'start_utf16': 10}
    read = """async bubble => {
        const {excerptBody, readExcerpt, excerptRange} = await import('/static/modules/chat_reply_excerpt.js');
        const body = excerptBody(bubble), r = document.createRange();
        if (!body) return null;
        r.setStart(body.firstChild, body.textContent.indexOf('map')); r.setEnd(body.firstChild, body.textContent.length);
        const excerpt = readExcerpt(bubble, r);
        return {words:body.textContent, excerpt, landed:excerptRange(bubble, excerpt)?.toString()};
    }"""
    page.evaluate("""([photo,video]) => {
        emit('photo', {...photo, image_base64:'aW1hZ2U=', mime:'image/png'});
        emit('video', {...video, video_base64:'T2dnUw==', mime:'video/ogg'});
    }""", [photo, video])
    expected = {'words': caption, 'excerpt': selected, 'landed': 'map'}
    for media in (photo, video):
        assert bubble(page, media).evaluate(read) == expected
    project_chat['rows'][:] = [photo, video]
    page.reload()
    for media in (photo, video):
        assert bubble(page, media).evaluate(read) == expected
        assert bubble(page, media).locator('img, video').count() == 0


def test_two_grouped_captions_keep_separate_addresses_and_authored_i18n(project_chat):
    page = project_chat["page"]

    def photo(n, caption):
        return row(n, caption or "Photo attachment", ts=f"2026-10-01T17:0{n}:00Z",
                   msg_type="photo", caption=caption, task_id="media",
                   mime="image/png", image_base64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j2ZsAAAAASUVORK5CYII=")
    first, second, empty = photo(1, "Dawn"), photo(2, "Settings"), photo(3, "")
    project_chat["rows"][:] = [first, second, empty]
    page.reload()
    group = bubble(page, first)
    caption = group.locator('figcaption', has_text='Settings')
    caption.wait_for()
    assert group.locator('figure').count() == 3 and group.locator('figcaption').count() == 2
    assert group.locator('[data-reply-caption]').count() == 2
    page.evaluate("""async () => {
        const i18n = await import('/static/modules/i18n.js');
        i18n.applyPayload({language:'ru', entries:{Settings:{text:'Настройки'},
            'code:chat.reply.sender_selection':{text:'Выбор отправителя · Источник:'}}});
    }""")
    caption.evaluate("""node => { const r = document.createRange(); r.selectNodeContents(node);
        getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    page.get_by_role('button', name='Reply to quote', exact=True).click()
    staged = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
    assert staged['address'] == second['row_address'] and staged['excerpt'] == {'text': 'Settings', 'start_utf16': 0}
    assert staged['ts'] == second['ts']
    strip = page.locator('.chat-quote--composer[data-excerpt] .chat-quote-main')
    assert strip.get_attribute('aria-label').endswith(': Settings') and 'Выбор отправителя' in strip.get_attribute('aria-label')
    when = page.evaluate("async ts => (await import('/static/modules/chat_activity.js')).formatMsgTime(ts).short", second['ts'])
    assert f', {when}: Settings' in strip.get_attribute('aria-label')
    project_chat['expect'](strip.locator('.chat-quote-text')).to_have_text('Settings', use_inner_text=True)
    page.locator('#chat-input').fill('Clarify the setting.')
    page.locator('#chat-send').click()
    saved = page.locator('#chat-messages .chat-bubble.user .chat-quote-main')
    project_chat['expect'](saved.locator('.chat-quote-text')).to_have_text('Settings', use_inner_text=True)
    assert saved.get_attribute('aria-label').endswith(': Settings')
    assert page.evaluate("""async () => !(await import('/static/modules/i18n.js')).pendingMisses()
        .some(key => key.includes(': Settings'))""")
    saved.click()
    page.wait_for_function("""() => [...CSS.highlights.get('reply-excerpt') || []]
        .some(r => r.startContainer.parentElement.matches('figcaption') && r.toString() === 'Settings')""")


def test_mermaid_source_fallback_and_render_share_passage_coordinates(project_chat):
    page = project_chat["page"]
    source = row(1, "🌙 Before\n\n```mermaid\ngraph TD\nA --> B\n```\n\ngraph TD was the first line.")
    project_chat["rows"][:] = [source]
    page.route('**/static/mermaid.min.js', lambda route: route.abort())
    page.reload()
    body = bubble(page, source).locator('.message')
    project_chat['expect'](body.locator('.md-diagram-error-note')).to_have_text('Diagram library failed to load.')
    facts = body.evaluate("""async body => {
        const {readExcerpt, excerptRange} = await import('/static/modules/chat_reply_excerpt.js');
        const row = body.closest('.chat-bubble'), r = document.createRange();
        r.selectNodeContents(body.querySelector('.md-diagram-error-note'));
        const note = readExcerpt(row, r);
        r.selectNodeContents(body.querySelector('code'));
        const code = readExcerpt(row, r);
        const prose = body.querySelector('p:last-child').firstChild;
        r.setStart(prose, 0); r.setEnd(prose, 8);
        return {note, code, landed: excerptRange(row, code)?.toString(), prose: readExcerpt(row, r)};
    }""")
    assert facts == {"note": None, "code": {"text": "graph TD\nA --> B\n", "start_utf16": 10},
                     "landed": "graph TD\nA --> B\n", "prose": {"text": "graph TD", "start_utf16": 27}}
    code = body.locator('pre code')
    code.evaluate("""node => { const t = node.firstChild, r = document.createRange();
        r.setStart(t, 0); r.setEnd(t, 8); getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    page.get_by_role('button', name='Reply to quote', exact=True).click()
    selected = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply')).excerpt")
    assert selected == {"text": "graph TD", "start_utf16": 10}
    code_reply = row(2, 'About the diagram code', role='user', reply_to=ref(source, selected))
    prose_reply = row(3, 'About the prose', role='user', reply_to=ref(source, facts['prose']))
    project_chat['rows'].extend([code_reply, prose_reply])
    page.unroute('**/static/mermaid.min.js')
    page.reload()
    body.locator('.md-mermaid svg').wait_for()
    bubble(page, code_reply).locator('.chat-quote-main').click()
    project_chat['expect'](page.locator('#toast-stack')).to_contain_text('The quoted passage could not be located.')
    assert not page.evaluate("CSS.highlights.has('reply-excerpt')")
    assert bubble(page, source).evaluate("node => node.classList.contains('is-quote-target')")
    bubble(page, prose_reply).locator('.chat-quote-main').click()
    page.wait_for_function("""() => [...CSS.highlights.get('reply-excerpt') || []].some(range =>
        range.startContainer.parentElement.matches('.message > p:last-child') && range.toString() === 'graph TD')""")


def test_rendered_math_keeps_source_units_off_the_prose(project_chat):
    page = project_chat["page"]
    # The formula's source `x^2` is also duplicated in the prose right after it, at the exact
    # offset a passage saved on the fallback text resolves to once KaTeX replaces the formula:
    # that passage must never highlight the prose copy in its place.
    source = row(1, "\\(x^2\\)??x^2")
    project_chat["rows"][:] = [source]
    page.route("**/static/vendor/katex/**", lambda route: route.abort())
    page.reload()
    body = bubble(page, source).locator(".message")
    project_chat["expect"](body.locator(".katex")).to_have_count(0)
    facts = body.evaluate("""async body => {
        const {readExcerpt} = await import('/static/modules/chat_reply_excerpt.js');
        const target = body.closest('.chat-bubble'), text = body.querySelector('p').firstChild;
        const r = document.createRange(), pick = (a, b) => {
            r.setStart(text, a); r.setEnd(text, b); return readExcerpt(target, r); };
        return {words: text.data, formula: pick(2, 5), prose: pick(7, 12)};
    }""")
    assert facts == {"words": "\\(x^2\\)??x^2", "formula": {"text": "x^2", "start_utf16": 2},
                     "prose": {"text": "??x^2", "start_utf16": 7}}
    body.evaluate("""body => { const text = body.querySelector('p').firstChild; const r = document.createRange();
        r.setStart(text, 2); r.setEnd(text, 5); getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    page.get_by_role("button", name="Reply to quote", exact=True).click()
    selected = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply')).excerpt")
    assert selected == facts["formula"]
    code_reply = row(2, 'About the formula', role='user', reply_to=ref(source, selected))
    prose_reply = row(3, 'About the prose', role='user', reply_to=ref(source, facts["prose"]))
    project_chat["rows"].extend([code_reply, prose_reply])
    page.unroute("**/static/vendor/katex/**")
    page.reload()
    body.locator(".katex").wait_for()
    # Coordinates survive the render: the prose passage, saved on the fallback text, still
    # reads at the same offset over the rendered formula's reserved source units.
    rendered = body.evaluate("""async body => {
        const {readExcerpt} = await import('/static/modules/chat_reply_excerpt.js');
        const target = body.closest('.chat-bubble'), prose = [...target.querySelectorAll('p')][0].lastChild;
        const r = document.createRange(); r.setStart(prose, 0); r.setEnd(prose, prose.length);
        return readExcerpt(target, r);
    }""")
    assert rendered == facts["prose"]
    bubble(page, code_reply).locator('.chat-quote-main').click()
    project_chat["expect"](page.locator("#toast-stack")).to_contain_text('The quoted passage could not be located.')
    assert not page.evaluate("CSS.highlights.has('reply-excerpt')")
    assert bubble(page, source).evaluate("node => node.classList.contains('is-quote-target')")
    bubble(page, prose_reply).locator('.chat-quote-main').click()
    page.wait_for_function("""() => [...CSS.highlights.get('reply-excerpt') || []]
        .some(range => range.toString() === '??x^2')""")


def test_a_formula_katex_cannot_parse_keeps_its_source_units(project_chat):
    page = project_chat["page"]
    # KaTeX shows a formula it cannot parse as its TeX alone, four units shorter than the delimited
    # source. The second `map` stands exactly those four units after the first: a passage saved on
    # the first before the renderer ran must still land on the first, never on its neighbour.
    source = row(1, "$$x^$$ map map")
    project_chat["rows"][:] = [source]
    page.route("**/static/vendor/katex/**", lambda route: route.abort())
    page.reload()
    body = bubble(page, source).locator(".message")
    body.locator("p").wait_for()
    project_chat["expect"](body.locator(".katex-error")).to_have_count(0)
    facts = body.evaluate("""async body => {
        const {readExcerpt} = await import('/static/modules/chat_reply_excerpt.js');
        const target = body.closest('.chat-bubble'), text = body.querySelector('p').firstChild;
        const r = document.createRange(), pick = (a, b) => {
            r.setStart(text, a); r.setEnd(text, b); return readExcerpt(target, r); };
        return {words: text.data, tex: pick(2, 4), first: pick(7, 10), across: pick(0, 4)};
    }""")
    assert facts == {"words": "$$x^$$ map map", "tex": {"text": "x^", "start_utf16": 2},
                     "first": {"text": "map", "start_utf16": 7}, "across": {"text": "$$x^", "start_utf16": 0}}
    body.evaluate("""body => { const text = body.querySelector('p').firstChild; const r = document.createRange();
        r.setStart(text, 7); r.setEnd(text, 10); getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    page.get_by_role("button", name="Reply to quote", exact=True).click()
    assert page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply')).excerpt") == facts["first"]
    replies = {name: row(n, f"About {name}", role="user", reply_to=ref(source, facts[name]))
               for n, name in enumerate(("first", "tex", "across"), start=2)}
    project_chat["rows"].extend(replies.values())
    page.unroute("**/static/vendor/katex/**")
    page.reload()
    body.locator(".katex-error").wait_for()
    assert body.locator(".katex").count() == 0
    rendered = body.evaluate("""async body => {
        const {readExcerpt} = await import('/static/modules/chat_reply_excerpt.js');
        const target = body.closest('.chat-bubble'), error = body.querySelector('.katex-error');
        const prose = error.closest('p').lastChild, r = document.createRange();
        const pick = (from, a, to, b) => { r.setStart(from, a); r.setEnd(to, b); return readExcerpt(target, r); };
        return {shown: error.closest('p').textContent, tex: pick(error.firstChild, 0, error.firstChild, 2),
                first: pick(prose, 1, prose, 4), second: pick(prose, 5, prose, 8),
                across: pick(error.firstChild, 0, prose, 4)};
    }""")
    assert rendered == {"shown": "x^ map map", "tex": facts["tex"], "first": facts["first"],
                        "second": {"text": "map", "start_utf16": 11}, "across": None}

    def landed(inside, start):
        page.wait_for_function("""([address, inside, start]) => {
            const body = [...document.querySelectorAll('#chat-messages .chat-bubble')]
                .find(node => node.dataset.rowAddress === address).querySelector('.message');
            const error = body.querySelector('.katex-error');
            const node = inside ? error.firstChild : error.closest('p').lastChild;
            return [...CSS.highlights.get('reply-excerpt') || []].some(range =>
                range.startContainer === node && range.startOffset === start);
        }""", arg=[source["row_address"], inside, start])

    # A passage that took in a delimiter the failed formula no longer shows has no place to land.
    bubble(page, replies["across"]).locator('.chat-quote-main').click()
    project_chat["expect"](page.locator("#toast-stack")).to_contain_text('The quoted passage could not be located.')
    assert not page.evaluate("CSS.highlights.has('reply-excerpt')")
    bubble(page, replies["first"]).locator('.chat-quote-main').click()
    landed(False, 1)
    bubble(page, replies["tex"]).locator('.chat-quote-main').click()
    landed(True, 0)


def test_retained_badge_survives_reload_after_its_upload_completes(project_chat):
    page, expect = project_chat["page"], project_chat["expect"]
    project_chat["rows"][:] = [row(1, "Copper lantern")]
    held = []
    page.route("**/api/owner/context-mode", lambda route: held.append(route))

    def upload(route):
        upload_name = f'{1:032x}_note.txt'
        route.fulfill(json={"ok": True, "filename": upload_name, "display_name": "note.txt", "mime": "text/plain",
                            "view": {"name": "note.txt", "kind": "file", "available": True, "size": 4,
                                     "url": f"/api/files/download?upload={upload_name}"}})

    page.route("**/api/chat/upload", upload)
    page.reload()
    select_contents(bubble(page, project_chat["rows"][0]).locator(".message p"))
    page.get_by_role("button", name="Reply to quote", exact=True).click()
    page.locator("#chat-file-input").set_input_files({"name": "note.txt", "mimeType": "text/plain", "buffer": b"note"})
    composer = page.locator("#chat-input")
    composer.fill("Read the attachment.")
    # A held context save holds the send; the owner edits the words while the badge stays staged.
    page.locator("#chat-context-mode .chat-context-head").click()
    page.locator(".chat-context-mode .chat-seg[data-mode='low']").click()
    expect(page.locator("#chat-context-mode")).to_have_attribute("data-disabled", "true")
    page.locator("#chat-send").click()
    expect(composer).to_have_value("Read the attachment.")  # the send waits for the room's save
    composer.fill("Edited words, same file.")
    held[0].fulfill(json={"ok": True, "context_mode": "low"})
    page.wait_for_function("sentFrames.length === 1")
    sent = page.evaluate("sentFrames[0]")
    assert sent["attachments"][0]["filename"].endswith("_note.txt")
    expect(page.locator(".attach-name")).to_have_text("note.txt")  # the badge stays in the newer draft
    # A reload restores the retained badge from its completed upload's ref, with the newer words.
    page.reload()
    expect(composer).to_have_value("Edited words, same file.")
    expect(page.locator(".attach-name")).to_have_text("note.txt")
    assert page.evaluate("sentFrames") == [], "a fresh page sends nothing"
    staged = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
    assert staged["excerpt"] == sent["reply_excerpt"]


def test_right_click_fenced_code_replies_only_without_an_explicit_selection(project_chat):
    page, expect = project_chat['page'], project_chat['expect']
    source = row(1, 'Example\n\n```text\nwords for the clipboard\n```')
    project_chat['rows'][:] = [source]
    page.reload()
    code = bubble(page, source).locator('pre code')
    select_contents(code)
    point = page.evaluate("""() => { const line = getSelection().getRangeAt(0).getClientRects()[0];
        return [line.left + line.width / 2, line.top + line.height / 2]; }""")
    page.evaluate("document.addEventListener('contextmenu', e => { window.menuPrevented = e.defaultPrevented; }, {once:true})")
    page.mouse.click(*point, button='right')
    assert page.evaluate('window.menuPrevented') is False  # the browser still owns native Copy
    assert page.evaluate('getSelection().toString().trim()') == 'words for the clipboard'
    menu = page.locator('.chat-row-menu')
    expect(menu).to_have_count(0)
    page.evaluate('getSelection().removeAllRanges()')
    page.mouse.click(*point, button='right')
    menu.get_by_role('menuitem', name='Reply', exact=True).click()
    kept = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
    assert kept['address'] == source['row_address'] and 'excerpt' not in kept


def test_grouped_document_quotes_use_the_items_time_live_and_after_reload(project_chat):
    page = project_chat['page']
    project_chat['rows'].clear()
    page.reload()
    page.wait_for_function('typeof window.emit === "function"')
    docs = [row(n, words, msg_type="document", filename=f"{words}.txt", mime="text/plain", caption=words,
                ts=f"2026-10-01T17:0{n}:00Z", task_id="maps") for n, words in enumerate(("Before", "After"), 1)]
    for doc in docs:
        page.evaluate("doc => emit('document', {...doc, type:'document', history_id:undefined})", doc)
    project_chat['rows'][:] = docs
    for reload in (False, True):
        if reload:
            page.reload()
        group = bubble(page, docs[0])
        project_chat['expect'](group.locator('.chat-file-item')).to_have_count(2)
        group.locator('.chat-file-caption', has_text='After').evaluate("""node => {
            const r = document.createRange(); r.selectNodeContents(node);
            getSelection().removeAllRanges(); getSelection().addRange(r);
        }""")
        page.get_by_role('button', name='Reply to quote', exact=True).click()
        staged = page.evaluate("JSON.parse(sessionStorage.getItem('ouro_chat_reply'))")
        assert (staged['address'], staged['ts']) == (docs[1]['row_address'], docs[1]['ts'])
        when = page.evaluate("async ts => (await import('/static/modules/chat_activity.js')).formatMsgTime(ts).short", docs[1]['ts'])
        assert f', {when}: After' in page.locator('.chat-quote--composer .chat-quote-main').get_attribute('aria-label')
        page.get_by_role('button', name='Cancel reply').click()


def test_selected_send_frees_the_field_and_its_bubble_resends_after_reload(project_chat):
    page, expect = project_chat["page"], project_chat["expect"]
    project_chat["rows"][:] = [row(1, "Copper lantern")]
    uploads = []

    def upload(route):
        name = 'one.txt' if not uploads else 'two.txt'
        uploads.append(name)
        upload_name = f'{len(uploads):032x}_{name}'
        route.fulfill(json={"ok": True, "filename": upload_name, "display_name": name, "mime": "text/plain",
            "view": {"name": name, "kind": "file", "available": True, "size": 3,
                     "url": f"/api/files/download?upload={upload_name}"}})

    page.route('**/api/chat/upload', upload)
    page.reload()
    body = bubble(page, project_chat["rows"][0]).locator('.message')
    body.evaluate("""node => { const r = document.createRange(); r.selectNodeContents(node);
        getSelection().removeAllRanges(); getSelection().addRange(r); }""")
    page.get_by_role('button', name='Reply to quote', exact=True).click()
    composer = page.locator('#chat-input')
    files = page.locator('#chat-file-input')
    send = page.locator('#chat-send')
    composer.fill('Read the attachment.')
    files.set_input_files({'name': 'one.txt', 'mimeType': 'text/plain', 'buffer': b'one'})
    send.click()
    page.wait_for_function('sentFrames.length === 1')
    first = page.evaluate('sentFrames[0]')
    sent_files = [item['filename'] for item in first['attachments']]
    assert first['reply_to'] and first['reply_excerpt'] and sent_files == [f'{1:032x}_one.txt']
    # Taken by the socket, the words, the file and the quote leave the field: the next message answers nothing.
    expect(composer).to_have_value('')
    expect(page.locator('.attach-name')).to_have_count(0)
    expect(page.locator('.chat-quote--composer')).to_have_count(0)
    page.evaluate("""f => emit('chat',{role:'system',chat_id:1,system_type:'initialization_notice',
        content:'Not saved yet',client_message_id:f.client_message_id})""", first)
    resend = page.get_by_role('button', name='Send again', exact=True)
    expect(resend).to_have_count(1)

    # After a reload its bubble still holds the complete send: the same id, words, quote and files, uploaded once.
    page.reload()
    expect(composer).to_have_value('')
    expect(page.locator('.chat-quote--composer')).to_have_count(0)
    resend.click()
    page.wait_for_function('sentFrames.length === 1')
    again = page.evaluate('sentFrames[0]')
    keys = ('client_message_id', 'content', 'reply_to', 'reply_excerpt')
    assert {k: again.get(k) for k in keys} == {k: first.get(k) for k in keys}
    assert [item['filename'] for item in again['attachments']] == sent_files and uploads == ['one.txt']

    accept = """f => emit('chat',{type:'chat',role:'user',source:'web',chat_id:1,
        content:f.content,client_message_id:f.client_message_id,ts:'2026-10-01T17:00:30Z',
        ingress_accepted:true,ingress_dispatched:true,reply_to:{address:f.reply_to,excerpt:f.reply_excerpt}})"""
    composer.fill('And this one.')
    files.set_input_files({'name': 'two.txt', 'mimeType': 'text/plain', 'buffer': b'two'})
    page.unroute('**/api/chat/upload', upload)

    def upload_with_late_ack(route):
        # The first message's late acceptance leaves the newer draft and its upload alone.
        page.evaluate(accept, first)
        assert composer.input_value() == 'And this one.' and page.locator('.attach-name').inner_text() == 'two.txt'
        upload(route)

    page.route('**/api/chat/upload', upload_with_late_ack)
    send.click()
    page.wait_for_function('sentFrames.length === 2')
    next_frame = page.evaluate('sentFrames[1]')
    # The harness coins ids per page load, so after the reload the frame itself tells the two messages apart.
    assert next_frame['content'].startswith('And this one.') and next_frame['attachments'] != first['attachments']
    assert not next_frame.get('reply_to') and not next_frame.get('reply_excerpt')
    expect(composer).to_have_value('')
    expect(page.locator('.attach-name')).to_have_count(0)
    expect(resend).to_have_count(0)


def test_send_as_new_recovers_the_old_quote_only_when_the_composer_is_free(project_chat):
    page, expect = project_chat['page'], project_chat['expect']
    old, new = row(1, 'The blue lamp.'), row(2, 'The amber lamp.')
    project_chat['rows'][:] = [old, new]
    page.reload()
    composer = page.locator('#chat-input')
    stage = page.get_by_role('button', name='Reply to quote', exact=True)
    select_contents(bubble(page, old).locator('.message p'))
    stage.click()
    composer.fill('Was it shipped?')
    page.locator('#chat-send').click()
    page.wait_for_function('sentFrames.length === 1')
    first, = page.evaluate('sentFrames')
    # The sent words and quote leave the field; the closing strip is gone before the next quote opens.
    expect(composer).to_have_value('')
    expect(page.locator('.chat-quote--composer')).to_have_count(0)

    select_contents(bubble(page, new).locator('.message p'))
    stage.click()
    composer.fill('Ship this one tomorrow.')
    page.locator('#chat-file-input').set_input_files({'name': 'invoice.txt', 'mimeType': 'text/plain', 'buffer': b'new'})
    page.evaluate("""([frame, reply_to, address]) => emit('chat', { ...frame, type:'chat', role:'user', source:'web',
        chat_id:1, ts:'2026-10-01T17:00:03Z', row_address:address, ingress_accepted:true, reply_to })""",
        [first, ref(old, first['reply_excerpt']), row(3, 'Was it shipped?')['row_address']])
    again = page.get_by_role('button', name='Send as a new message', exact=True)
    again.click()
    expect(page.locator('#toast-stack')).to_contain_text('Finish or clear the current draft')
    expect(composer).to_have_value('Ship this one tomorrow.')
    draft_quote = page.locator('.chat-quote--composer:not([inert]) .chat-quote-text')
    expect(draft_quote).to_have_text(new['text'])
    expect(page.locator('.attach-name')).to_have_text(['invoice.txt'])

    # Even an attachment-only draft is occupied; neither reply donates its content to the other.
    page.get_by_role('button', name='Cancel reply').click()
    composer.fill('')
    again.click()
    expect(composer).to_have_value('')
    expect(draft_quote).to_have_count(0)
    expect(page.locator('.attach-name')).to_have_text(['invoice.txt'])
    page.get_by_role('button', name='Remove attachment invoice.txt').click()
    composer.fill('A plain draft.')
    again.click()
    expect(composer).to_have_value('A plain draft.')
    expect(draft_quote).to_have_count(0)

    composer.fill('')
    again.click()
    expect(composer).to_have_value('Was it shipped?')
    expect(draft_quote).to_have_text(old['text'])
    assert page.evaluate('sentFrames') == [first], 'preparing does not send'
    page.locator('#chat-send').click()
    page.wait_for_function('sentFrames.length === 2')
    second = page.evaluate('sentFrames[1]')
    assert second['client_message_id'] != first['client_message_id'] and second['content'] == first['content']
    assert second['reply_to'] == old['row_address'] and second['reply_excerpt'] == first['reply_excerpt']
    expect(composer).to_have_value('')


def test_a_selected_sends_delivery_doubt_leaves_its_bubble_when_the_host_confirms_dispatch(project_chat):
    """A saved row of a host process that has ended says its delivery is not confirmed and offers Send as new; the
    running host's own dispatch fact on that row takes the warning and the action off the bubble, not only off storage."""
    page, expect = project_chat['page'], project_chat['expect']
    project_chat['rows'][:] = [row(1, 'Copper lantern')]
    page.reload()
    select_contents(bubble(page, project_chat['rows'][0]).locator('.message'))
    page.get_by_role('button', name='Reply to quote', exact=True).click()
    page.locator('#chat-input').fill('Which one?')
    page.locator('#chat-send').click()
    page.wait_for_function('sentFrames.length === 1')
    first = page.evaluate('sentFrames[0]')
    saved = """([frame, facts]) => emit('chat', { ...frame, type:'chat', role:'user', source:'web', chat_id:1,
        ts:'2026-10-01T17:00:30Z', ingress_accepted:true, ...facts, reply_to:{address:frame.reply_to, excerpt:frame.reply_excerpt} })"""
    page.evaluate(saved, [first, {}])  # no dispatch fact: the process that saved it has since ended
    sent = page.locator('#chat-messages .chat-bubble.user')
    expect(sent.locator("[data-ingress-saved='unconfirmed']")).to_contain_text('Saved; delivery not confirmed.')
    again = page.get_by_role('button', name='Send as a new message', exact=True)
    expect(again).to_have_count(1)
    page.evaluate(saved, [first, {'ingress_dispatched': True}])  # the running host entered its dispatch
    expect(sent.locator("[data-ingress-saved='unconfirmed']")).to_have_count(0)
    expect(again).to_have_count(0)
    expect(sent.locator('[data-ingress-saved]')).to_have_count(1)  # the plain saved tick stands in its place


def test_a_long_sent_quote_keeps_its_show_all_clear_of_the_rounded_edge(project_chat):
    page, expect = project_chat['page'], project_chat['expect']
    source = row(1, 'First line.\nSecond line.\nThird line.')
    excerpt = {'text': 'First line.\nSecond line.\nThird line.', 'start_utf16': 0}
    reply = row(2, 'About these lines.', role='user', source='web', reply_to=ref(source, excerpt))
    project_chat['rows'][:] = [source, reply]
    page.reload()
    toggle = bubble(page, reply).locator('.chat-quote[data-excerpt] .chat-clamp-toggle')
    expect(toggle).to_be_visible()
    inset = toggle.evaluate("""node => {
        const quote = node.closest('.chat-quote'), box = node.getBoundingClientRect(), edge = quote.getBoundingClientRect();
        return getComputedStyle(quote).direction === 'rtl' ? box.left - edge.left : edge.right - box.right;
    }""")
    assert inset >= 7.5, f'Show all sits {inset}px from the quote edge; the composer strip keeps --space-2 (8px)'


def test_a_file_items_wrapper_wears_the_cards_radius_so_its_target_ring_follows_it(project_chat):
    page, expect = project_chat['page'], project_chat['expect']
    project_chat['rows'].clear()
    page.reload()
    page.wait_for_function('typeof window.emit === "function"')
    doc = row(1, 'Plan', msg_type='document', filename='plan.txt', mime='text/plain', caption='Plan', task_id='maps')
    page.evaluate("doc => emit('document', {...doc, type:'document', history_id:undefined})", doc)
    item = bubble(page, doc).locator('.chat-file-item').first
    expect(item).to_be_visible()
    radius, token = item.evaluate("""node => [getComputedStyle(node).borderTopLeftRadius,
        getComputedStyle(document.documentElement).getPropertyValue('--radius-inner').trim()]""")
    assert radius == token and radius != '0px', f'wrapper radius {radius} should be the inner radius {token}'
