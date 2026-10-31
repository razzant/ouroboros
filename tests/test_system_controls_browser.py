"""Approved System material and six button consumers in the real SPA.

Synthetic history/API data, production renderers and styles, isolated subscription
server. Clipboard writes are recorded at the browser boundary; downloads are real
browser downloads. Native desktop bridges are outside this test's scope.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from tests import test_room_work_browser as rw
from tests.test_dark_voice_material_contrast_browser import COMPOSITE

subscription_ui = rw.subscription_ui
room_ui = rw.room_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
PROJECT = {"id": "system-proof", "name": "Release evidence", "chat_id": 42,
           "lifecycle": "active", "visible_revision": 0}
REMINDER = ("Reminder · Ouroboros · written 21 October 2026, 18:00 · for 22 October 2026, 09:00\n\n"
            + "Before the call, compare the saved report with the source files. Keep the original notes, "
              "including unresolved questions and the evidence still needed from the next run. " * 4)
FILE_NAME = "Release evidence and outstanding questions from the October verification archive.zip"


def system(text, index, **facts):
    return {"role": "system", "text": text, "ts": f"2026-10-22T06:00:{index:02}Z", **facts}


def complete_history(page, transport):
    def respond(route):
        chat_id = int(parse_qs(urlparse(route.request.url).query).get("chat_id", ["1"])[0])
        rows = transport.history.get(chat_id, [])
        route.fulfill(json={"messages": rows, "progress": [], "has_more": False, "page_cursor": "proof",
            "coverage": {"v": 1, "view": "proof", "upper": {"chat": len(rows), "progress": 0}, "spans": {
                "chat": {"from": 0, "to": len(rows), "chain": "proof", "gaps": []},
                "progress": {"from": 0, "to": 0, "chain": "empty", "gaps": []}}}})
    page.route("**/api/chat/history*", respond)
    page.set_default_timeout(8000)


# Finish colour transitions before measuring, without forcing a simulated state.
# Token references are resolved on this exact consumer, including voice-local tokens.
PAINT = """el => {
    for (const a of document.getAnimations()) if (a.constructor.name === 'CSSTransition') a.finish();
    const s=getComputedStyle(el), r=el.getBoundingClientRect();
    const hit=document.elementFromPoint(r.x+r.width/2,r.y+r.height/2);
    const token=(name, property) => {const p=document.createElement('i');
        p.style[property]=`var(${name})`;el.append(p);const v=getComputedStyle(p)[property];p.remove();return v;};
    return {bg:s.backgroundColor, color:s.color, radius:s.borderTopLeftRadius, shadow:s.boxShadow,
        borders:['Top','Right','Bottom','Left'].map(k=>[s[`border${k}Width`],s[`border${k}Color`]]),
        outline:s.outlineWidth, outlineStyle:s.outlineStyle, outlineColor:s.outlineColor,
        filter:s.filter, transform:s.transform, glass:s.backdropFilter || s.webkitBackdropFilter,
        width:r.width,height:r.height,x:r.x,y:r.y,right:r.right,bottom:r.bottom,
        reachable:!!hit && (hit===el || el.contains(hit)),overflow:el.scrollWidth-el.clientWidth,
        rest:token('--elev-rest','boxShadow'),float:token('--elev-float','boxShadow'),
        capsule:token('--surface-capsule','backgroundColor'),hover:token('--surface-capsule-hover','backgroundColor'),
        neutralHover:token('--bg-hover','backgroundColor'),quietHover:token('--tone-quiet-hover','backgroundColor'),
        focus:token('--focus-accent-border','color'),
        ok:token('--status-ok-fg','color'), contrast:window.__contrast.of(el)};
}"""


def paint(locator):
    return locator.evaluate(PAINT)


def reveal(page, locator):
    page.evaluate("getSelection().removeAllRanges()")
    locator.evaluate("el=>(el.closest('.chat-bubble,.chat-live-card')||el).scrollIntoView({block:'center'})")
    rw.settle_frames(page)


def borderless(value):
    assert all(width == "0px" or color == "rgba(0, 0, 0, 0)" for width, color in value["borders"]), value


def button_states(page, control, *, quiet=False, floating=False, card=False, disable=True):
    reveal(page, control)
    page.mouse.move(0, 0)
    rest = paint(control)
    assert rest["reachable"] and rest["x"] >= 0 and rest["right"] <= page.viewport_size["width"] + 1, rest
    borderless(rest)
    assert rest["radius"] == ("12px" if card else "999px"), rest
    assert rest["shadow"] == ("none" if quiet else rest["float"] if floating else rest["rest"]), rest
    control.hover()
    hover = paint(control)
    if not quiet and not floating:
        assert hover["bg"] == hover["hover"], hover
    page.mouse.down()
    pressed = paint(control)
    # Release away from the control: inspect the actual active state without invoking its action.
    page.mouse.move(0, 0)
    page.mouse.up()
    page.evaluate("getSelection().removeAllRanges()")
    page.keyboard.press("Tab")
    control.focus()
    focus = paint(control)
    assert (focus["outline"], focus["outlineStyle"], focus["outlineColor"]) == ("2px", "solid", focus["focus"]), focus
    for state in [hover, pressed, focus]:
        borderless(state)
        assert state["height"] == pytest.approx(rest["height"], abs=0.1), state
        assert state["shadow"] == rest["shadow"] and state["transform"] == "none", state
    for state in [rest, hover, pressed]:
        assert state["contrast"]["ratio"] >= 4.5, state
    assert "brightness" in pressed["filter"], pressed
    if floating:
        assert hover["filter"].count('brightness') == 1 and pressed["filter"].count('brightness') == 2
    if quiet:
        assert hover["bg"] == hover["quietHover"], hover
    result = {"rest": rest, "hover": hover, "press": pressed, "focus": focus}
    if disable:
        control.evaluate("el=>el.disabled=true")
        control.hover(force=True)
        result["disabled"] = paint(control)
        assert result["disabled"]["shadow"] == "none", result
        control.evaluate("el=>el.disabled=false")
    return result


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1055, 390])
def test_system_main_project_history_live_and_nested_review(room_ui, theme, width):
    page = room_ui["page"]
    rows = [system("Background consciousness enabled; next wake-up at 23:39.", 1,
                   system_type="consciousness_state"),
            system(REMINDER, 2, system_type="reminder"),
            system("Release evidence · Stopped", 3, system_type="project_completion_summary",
                   project_id=PROJECT["id"], project_name=PROJECT["name"], task_id="ended-proof",
                   task_name="Compare release evidence", completion_notice={"phase": "cancelled",
                       "cause": "Stopped at the owner's request.", "salvage": "Preserved output: the saved archive."}),
            system("Skill review: archive-reader", 4, system_type="skill_review", skill="archive-reader", job_id="proof")]
    transport = rw.Transport(page, projects=[PROJECT], history={1: rows, 42: rows[:2]})
    complete_history(page, transport)
    attempts = []

    def review_read(route):
        attempts.append(route.request.url)
        route.fulfill(status=503 if len(attempts) == 1 else 200,
                      json={"error": "Temporarily unavailable"} if len(attempts) == 1
                      else {"markdown": "## Findings\n\nReview evidence is available again."})

    page.route("**/api/skills/archive-reader/review-history/proof", review_read)
    rw.open_app(room_ui, transport, theme=theme, viewport={"width": width, "height": 764})
    page.evaluate(COMPOSITE)
    page.locator("#chat-messages .chat-bubble.system").first.wait_for()
    prefix = f"system-{theme}-{width}"
    page.locator("#chat-messages").evaluate("el=>el.scrollTop=0")
    rw.capture(page, prefix + "-main")
    values = page.locator("#chat-messages .chat-bubble.system").evaluate_all(
        "els=>els.map(el=>({bg:getComputedStyle(el).backgroundColor,border:getComputedStyle(el).borderLeftWidth,"
        "shadow:getComputedStyle(el).boxShadow,radius:getComputedStyle(el).borderTopLeftRadius,"
        "overflow:el.scrollWidth-el.clientWidth,sender:__contrast.of(el.querySelector('.sender')),"
        "body:__contrast.of(el.querySelector('.message'))}))")
    assert len(values) == 4, values
    for value in values:
        assert value["bg"] == ("rgba(255, 255, 255, 0.055)" if theme == "dark" else "rgba(245, 158, 11, 0.12)"), value
        assert value["border"] == "0px" and value["radius"] == "18px" and value["shadow"] != "none", value
        assert value["overflow"] <= 1 and value["sender"]["ratio"] >= 4.5, value
        assert value["body"]["ratio"] >= 4.5, value
    ending = page.locator(".message.project-ending")
    reveal(page, ending)
    assert paint(ending)["shadow"] == "none"
    assert paint(ending)["radius"] == "0px"
    rw.capture(page, prefix + "-ending")
    review = page.locator(".chat-skill-review")
    review.locator('[data-skill-review-toggle]').click()
    retry = review.locator('[data-skill-review-retry]')
    retry.wait_for()
    retry_states = button_states(page, retry)
    rw.capture(page, prefix + "-retry")
    retry.click()
    review.locator('[data-skill-review-full][data-state="loaded"]').wait_for()
    assert len(attempts) == 2
    full = paint(review.locator('[data-skill-review-full]'))
    assert full["radius"] == "0px" and full["shadow"] == "none", full
    rw.open_project(page, PROJECT)
    project_rows = page.locator("#project-panel .chat-bubble.system")
    project_rows.first.wait_for()
    reveal(page, project_rows.first)
    rw.capture(page, prefix + "-project")
    for row in project_rows.all():
        borderless(paint(row))
        assert paint(row)["shadow"] == paint(row)["rest"]
    transport.send({"type": "chat", "chat_id": 42, "role": "system", "system_type": "reminder",
                    "content": "Reminder · archive-reader · keep the saved evidence.", "ts": "2026-10-22T06:01:00Z"})
    live = page.locator("#project-panel .chat-bubble.system").filter(has_text="keep the saved evidence")
    live.wait_for()
    assert paint(live)["bg"] == values[0]["bg"]
    rw.evidence(prefix, {"history": values, "retry": retry_states, "reviewDetail": full,
                         "liveProject": paint(live), "console": transport.console})


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("width", [1055, 390])
def test_shared_controls_material_and_real_actions(room_ui, theme, width):
    page = room_ui["page"]
    owner = rw.owner("Please keep the archive with its full name.", "owner-file", 1)
    owner["attachments"] = [{"name": FILE_NAME, "kind": "file", "mime": "application/zip", "size": 1234,
                              "available": True, "url": "/api/files/download?upload=" + "a" * 32 + "_archive.zip"},
                             {"name": "Earlier archive.zip", "kind": "file", "available": False}]
    transport = rw.Transport(page, activities=[rw.activity("controls-work")], history={1: [
        owner, rw.assistant("Saved commands:\n\n```sh\nprintf 'evidence saved'\n```", 2),
        rw.progress("controls-work", "Checking the archive evidence.", 3, suggested_name="Archive evidence review")]})
    complete_history(page, transport)
    update_status = {"available": True, "local_changes": "absent",
                     "current_version": "1", "latest_version": "2"}
    page.route("**/api/update/status", lambda route: route.fulfill(json=update_status))
    room_ui["backend"]["downloads"] = {owner["attachments"][0]["url"]: b"archive evidence"}
    page.route("**/api/files/download?*", lambda route: route.continue_())
    clip = (Path(__file__).parent / "fixtures/media-recovery.mp4").read_bytes()
    video_source = {"available": False, "requests": []}

    def video_read(route):
        available = video_source["available"]
        video_source["requests"].append(available)
        route.fulfill(status=200 if available else 503, body=clip if available else b"Unavailable",
                      content_type="video/mp4")

    page.route("**/api/proof-video.mp4", video_read)
    rw.open_app(room_ui, transport, theme=theme, viewport={"width": width, "height": 764})
    page.evaluate(COMPOSITE)
    page.evaluate("""async name => {
        const {createChatMedia}=await import('/static/modules/chat_media.js');
        const {createReviewPresentationController}=await import('/static/modules/review_presentation.js');
        const feed=document.querySelector('#chat-messages');
        const media=createChatMedia({chatSessionId:'controls-proof',durableChatMediaUrl:x=>x||'',
            formatMsgTime:()=>null,senderLabel:()=> 'Ouroboros',stampNodeTimestamp:()=>{},
            insertMessageNode:node=>feed.append(node)});
        const canvas=document.createElement('canvas');canvas.width=320;canvas.height=120;
        const c=canvas.getContext('2d');c.fillStyle='#7ba1c4';c.fillRect(0,0,320,120);
        feed.append(media.buildMediaBubble({type:'photo',role:'assistant',
            image_base64:canvas.toDataURL('image/png').split(',')[1],mime:'image/png'}));
        feed.append(media.buildDocumentBubble({type:'document',role:'assistant',filename:name,
            file_base64:btoa('archive evidence'),mime:'application/zip'}));
        feed.append(media.buildMediaBubble({type:'video',role:'assistant',
            download_url:'/api/proof-video.mp4',mime:'video/mp4'}));
        const card=feed.querySelector('.chat-live-card');
        const host=document.createElement('div'), summary=document.createElement('span');card.append(summary,host);
        window.proofRetry=0;
        const review=createReviewPresentationController({host,summary,onHydrate:()=>{proofRetry++;review.setHydrateStatus('loading');}});
        review.setHydrateStatus('error');
        window.proofImages=[];
        navigator.clipboard.write=async items=>{for(const item of items) for(const type of item.types)
            proofImages.push({type,size:(await item.getType(type)).size});};
        window.controlsProof={media,review};
    }""", FILE_NAME)
    prefix = f"controls-{theme}-{width}"
    results = {}
    selectors = {
        "copy": '.md-code-copy', "photo": '.chat-photo-actions summary',
        "incomingFile": '.chat-bubble.user .chat-file-card',
        "outgoingFile": '.chat-bubble.assistant .chat-file-card',
        "retry": '[data-review-hydrate-retry]', "player": '[data-media-action="play"]',
    }
    for name, selector in selectors.items():
        control = page.locator(selector).first
        reveal(page, control)
        rw.capture(page, prefix + "-" + name)
        # Copy and the Reviews tray's Retry read (a `.review-read` link-button since #2278) are quiet controls.
        results[name] = button_states(page, control, quiet=name in ("copy", "retry"), floating=name == "photo",
                                      card=name.endswith("File"), disable=name != "photo")
    disabled = page.locator('.chat-bubble.user .chat-file-card:disabled')
    assert paint(disabled)["shadow"] == "none"
    assert paint(disabled)["radius"] == "12px"
    results["unavailableFile"] = paint(disabled)
    copy = page.locator('.md-code-copy')
    copy.click()
    page.wait_for_function("document.querySelector('.md-code-copy').classList.contains('is-copied')")
    assert paint(copy)["color"] == paint(copy)["ok"]
    assert page.evaluate("window.__copied.at(-1)") == "printf 'evidence saved'\n"
    results["copied"] = paint(copy)
    rw.capture(page, prefix + "-copied")
    photo = page.locator('.chat-photo-actions summary')
    photo.click()
    menu = page.locator('body > .chat-photo-menu[role="menu"]:not([inert])')
    menu.wait_for()
    rw.capture(page, prefix + "-photo-menu")
    page.keyboard.press('End')
    assert menu.locator('[data-photo-action="copy"]').evaluate('el=>el===document.activeElement')
    page.keyboard.press('Escape')
    assert photo.evaluate('el=>el===document.activeElement')
    photo.click()
    menu.locator('[data-photo-action="copy"]').click()
    page.wait_for_function('proofImages.length===1')
    assert page.evaluate('proofImages[0].type') == 'image/png'
    assert page.evaluate('proofImages[0].size') > 0
    photo.click()
    with page.expect_download() as download:
        menu.locator('[data-photo-action="download"]').click()
    assert download.value.failure() is None
    for selector in [selectors["incomingFile"], selectors["outgoingFile"]]:
        page.locator(selector).first.click()
        with page.expect_download() as download:
            page.locator('dialog[open]:not([inert]) [data-file-action="download"]').click()
        assert download.value.failure() is None
        assert Path(download.value.path()).read_bytes() == b"archive evidence"
    page.locator('[data-review-hydrate-retry]').click()
    assert page.evaluate('proofRetry') == 1
    review_tray = paint(page.locator('.chat-live-card .chat-live-reviews').last)
    assert review_tray["radius"] == "0px" and review_tray["shadow"] == "none", review_tray
    results["nestedReview"] = review_tray
    # A real failed decoder recovers on the existing Play gesture, on the same element.
    page.wait_for_function("document.querySelector('video').error?.code===4")
    page.evaluate("""() => {window.proofVideo=document.querySelector('video');window.proofLoads=0;
        const load=proofVideo.load.bind(proofVideo);proofVideo.load=()=>{proofLoads++;load();};}""")
    page.locator('[data-media-action="speed"]').click()
    assert page.evaluate('proofVideo.playbackRate') == 1.5
    video_source["available"] = True
    page.locator('[data-media-action="play"]').click()
    page.wait_for_function('!proofVideo.error && !proofVideo.paused && proofVideo.currentTime > 0.25')
    assert page.evaluate('proofLoads') == 1
    assert page.evaluate("document.querySelector('video')===proofVideo && proofVideo.videoWidth===160")
    page.locator('[data-media-action="play"]').click()
    page.wait_for_function('proofVideo.paused')
    rate = page.locator('.chat-media-rate-menu')
    rate_style = paint(rate)
    assert any(w != '0px' and c != 'rgba(0, 0, 0, 0)' for w, c in rate_style['borders']), rate_style
    if rate.is_visible():
        rate.select_option('2')
        assert page.evaluate('proofVideo.playbackRate') == 2
    results["rateField"] = rate_style
    results["videoRecovery"] = video_source
    reveal(page, page.locator('.chat-media-player'))
    rw.capture(page, prefix + "-recovered-player")
    # Update remains the real sidebar consumer, including in the mobile drawer.
    if width == 390:
        page.locator('[data-mobile-nav-toggle]').first.click()
    update = page.locator('#update-pill')
    update.wait_for()
    results["update"] = button_states(page, update)
    # The pill's palette: neutral, the owner's amber (local changes) and the failure's red. A confirmed success has no
    # capsule (owner 1B), so no green tone is drawn for it.
    for tone, token in [('neutral', '--surface-capsule'), ('warn', '--tone-amber'), ('error', '--tone-red')]:
        update.evaluate('(el,tone)=>el.dataset.tone=tone', tone)
        update.hover()
        value = paint(update)
        expected = update.evaluate("""(el,token)=>{const p=el.appendChild(document.createElement('i'));
            p.style.backgroundColor=`var(${token}-hover)`;const v=getComputedStyle(p).backgroundColor;p.remove();return v;}""", token)
        assert value["bg"] == expected, (tone, value, expected)
        results["update-" + tone] = value
    update.evaluate("el=>{el.dataset.tone='neutral';el.classList.add('has-local');}")
    results["update-local"] = paint(update)
    assert results["update-local"]["bg"] == results["update-warn"]["bg"]
    update.evaluate("el=>{el.dataset.tone='error';}")
    assert paint(update)["bg"] == results["update-error"]["bg"]
    # Exercise the current controller as well as its palette: real stage/outcome refreshes keep the shared button
    # material, a failed or unconfirmed outcome keeps its close, and a confirmed success draws no capsule at all while
    # the panel retains its record.
    for outcome, text, tone in [('', 'Checking update…', 'neutral'),
                                ('unknown', 'Update outcome unconfirmed', 'neutral'),
                                ('failed', 'Update failed', 'error')]:
        update_status['update_progress'] = {'operation_id': 'material-proof',
            'active': not outcome, 'stage': 'checking', 'outcome': outcome,
            'concluded_by': 'boot', 'current_generation': True}
        transport.send({'type': 'update_progress_changed'})
        page.wait_for_function("text=>document.querySelector('#update-pill')?.textContent===text", arg=text)
        assert update.get_attribute('data-tone') == tone
        assert 'has-local' not in (update.get_attribute('class') or '')
        assert page.locator('.update-notice-close').is_visible() == bool(outcome), 'only a concluded outcome offers its close'
        borderless(paint(update))
        results['controller-' + (outcome or 'checking')] = paint(update)
    rw.capture(page, prefix + "-update")
    update_status['update_progress'] = {'operation_id': 'material-proof', 'active': False, 'stage': 'checking',
        'outcome': 'applied', 'concluded_by': 'boot', 'current_generation': True}
    transport.send({'type': 'update_progress_changed'})
    # A confirmed success wears nothing and never masks the newer version this fixture announces ...
    page.wait_for_function("document.querySelector('#update-pill')?.textContent==='Update available'")
    assert update.get_attribute('data-tone') is None
    assert page.locator('.update-notice-close').count() == 1 and not page.locator('.update-notice-close').is_visible()
    # ... and with nothing newer to announce there is no capsule at all.
    update_status['available'] = False
    transport.send({'type': 'update_progress_changed'})
    page.wait_for_function("document.querySelector('#update-pill')?.hidden===true")
    assert not [t for t in page.locator('.toast').all_inner_texts() if 'update' in t.lower()], 'a confirmed success announces nothing'
    update_status['available'] = True
    # The capsule's destination: with a failure showing, pressing it opens Dashboard -> Updates, and hiding the notice
    # there changes nothing in the record.
    update_status['update_progress']['outcome'] = 'failed'
    transport.send({'type': 'update_progress_changed'})
    page.wait_for_function("document.querySelector('#update-pill')?.textContent==='Update failed'")
    update.click()
    page.locator('#page-dashboard').wait_for()
    if width == 390:
        rw.setup_browser.wait_drawer_closed(page)
    updates_selected = page.locator('[data-dashboard-tab="updates"]').get_attribute('aria-selected') == 'true'
    assert updates_selected
    assert page.locator('#updates-attempt-dismiss').count() == 0, 'the hide lives on the notice, never in the panel'
    assert 'The update did not complete.' in page.locator('#page-dashboard').inner_text()
    results["updateNavigation"] = {"dashboardVisible": True, "updatesSelected": updates_selected}
    rw.capture(page, prefix + "-update-destination")
    assert not [path for path, _ in room_ui["posts"] if '/update/' in path]
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    rw.evidence(prefix, {"states": results, "console": transport.console})
