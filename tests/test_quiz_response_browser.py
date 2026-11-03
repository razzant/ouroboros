"""Real quiz modules/styles, synthetic I/O; persistent browser close/reopen, not native app QA."""
from __future__ import annotations

import functools
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from tests._shared import producer_evidence_dir

pytestmark = [pytest.mark.ui_browser]
WEB = Path(__file__).resolve().parents[1] / "web"


def _evidence_dir(tmp_path):
    return producer_evidence_dir(tmp_path / "evidence", "quiz-response", env="QUIZ_BROWSER_EVIDENCE")

BOOT = """async ({id, theme}) => {
    document.documentElement.className='ouro-ui'; document.documentElement.dataset.theme=theme;
    document.head.innerHTML='<meta charset="utf-8"><link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">';
    document.body.innerHTML='<main style="max-width:680px;margin:24px auto;padding:16px" id="quiz-host"></main>';
    document.body.style.overflow='auto';
    const {createChatDecision}=await import('/modules/chat_decision.js');
    window.qaPosts=[]; window.qaReads=0; window.qaMode='hold';
    window.qaQuiz={quiz_id:id,state:'open',question:'When should this change ship?',
        options:[{label:'Ship after checks',detail:'Keep the answer and finish independent review.'},
            {label:'Wait for more evidence',detail:'Keep the current release unchanged.'}]};
    window.qaDecision=createChatDecision({isMain:true,
        frameNode:(_msg,node)=>node,showToast:()=>{},
        apiFetch:(_url,init)=>{qaPosts.push(JSON.parse(init.body));
            if(qaMode==='hold')return new Promise(resolve=>window.qaFinish=resolve);
            return Promise.resolve({ok:true,status:200,json:async()=>({ok:true,state:'answered',
                answered_index:qaPosts.at(-1).option_index,comment:qaPosts.at(-1).comment})});},
        fetchDetail:async()=>{qaReads++;return {task_id:'browser-quiz-task',
            question_rooms:{[id]:{state:'known',project_id:'',chat_id:1}},owner_quiz:{[id]:qaQuiz}};}});
    const card=qaDecision.buildQuizCard({...qaQuiz,task_id:'browser-quiz-task'});
    document.querySelector('#quiz-host').append(card);
    await Promise.all([...document.querySelectorAll('link')].map(l=>l.sheet?null:new Promise(r=>l.onload=r)));
}"""


@pytest.mark.parametrize("engine,theme,width", [("chromium", "dark", 1146), ("webkit", "light", 390)])
def test_quiz_pending_persistent_profile_recovery_and_keyboard(tmp_path, engine, theme, width):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 for the browser proof")
    playwright = pytest.importorskip("playwright.sync_api")
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    profile = tmp_path / "browser-profile"
    evidence = _evidence_dir(tmp_path)
    context = None
    pw = None
    observations = {}
    try:
        pw = playwright.sync_playwright().start()
        browser = getattr(pw, engine)
        def launch():
            return browser.launch_persistent_context(str(profile), headless=True,
                viewport={"width": width, "height": 844})
        def mount(context, quiz_id):
            page = context.new_page()
            page.goto(origin + "/tests/fixtures/question_presentation_parity.json")
            page.evaluate(BOOT, {"id": quiz_id, "theme": theme})
            return page
        context = launch()
        page = mount(context, "persistent")
        text = "  Keep my exact words\nand line break.  "
        page.locator(".chat-quiz-comment").fill(text)
        # Safari may blur a textarea without focusing a pointer-pressed button.
        # Capture native click-time focus before the component's target handler;
        # the pending paint must preserve that, not manufacture Chromium focus.
        page.evaluate("document.addEventListener('click',()=>window.qaClickFocus=document.activeElement,true)")
        page.locator(".chat-quiz-option").nth(1).click()
        pending = page.evaluate("""() => {const c=document.querySelector('.chat-quiz-card'),b=c.querySelectorAll('.chat-quiz-option')[1];
            return {state:c.dataset.state,phase:c.dataset.submission,pending:b.classList.contains('pending-choice'),
                chosen:b.classList.contains('chosen'),focusPreserved:document.activeElement===window.qaClickFocus,readonly:c.querySelector('textarea').readOnly,
                posts:qaPosts.length,overflow:c.scrollWidth>c.clientWidth,marked:getComputedStyle(b).backgroundColor!==getComputedStyle(c.querySelectorAll('.chat-quiz-option')[0]).backgroundColor};}""")
        observations["pending"] = pending
        page.screenshot(path=str(evidence / f"{engine}-{theme}-pending.png"))
        assert pending == {"state": "open", "phase": "sending", "pending": True, "chosen": False,
                           "focusPreserved": True, "readonly": True,
                           "posts": 1, "overflow": False, "marked": True}, pending
        original = page.evaluate("qaPosts[0]")
        page.screenshot(path=str(evidence / f"{engine}-{theme}-pending.png"))
        context.close()  # closes the browser process, not just the page
        context = launch()  # same on-disk browser profile and origin
        page = mount(context, "persistent")
        page.wait_for_function("qaReads===1 && document.querySelector('.chat-quiz-card').dataset.submission==='unknown'")
        assert page.evaluate("qaPosts.length") == 0, "reopening only reads; no automatic POST"
        assert page.locator(".chat-quiz-comment").input_value() == text
        page.screenshot(path=str(evidence / f"{engine}-{theme}-restored.png"))
        page.evaluate("qaMode='success'")
        page.locator(".chat-quiz-retry").click()
        page.wait_for_function("document.querySelector('.chat-quiz-card').dataset.state==='answered'")
        assert page.evaluate("qaReads") == 2
        assert page.evaluate("qaPosts[0]") == original
        assert page.evaluate("localStorage.length") == 0
        assert page.locator(".chat-quiz-answer").inner_text() == "Owner's answer: " + text
        page.close()

        # Restoring unsent words is programmatic: no input event or census may
        # be needed before the published quiet Close protects them.
        page = context.new_page()
        page.goto(origin + '/tests/fixtures/question_presentation_parity.json')
        page.evaluate("localStorage.setItem('ouro_quiz_reply_v1:'+JSON.stringify(['browser-quiz-task','draft-restore']), JSON.stringify({v:1,draft:'Restored unsent words',reason:'',attempt:null}))")
        page.evaluate(BOOT, {"id": "draft-restore", "theme": theme})
        assert page.locator('.chat-quiz-comment').input_value() == 'Restored unsent words'
        assert page.locator('.chat-quiz-close').is_disabled()
        # No separate control clears a draft (owner 19.10, A): emptying the field by hand releases it.
        assert page.locator('.chat-quiz-clear').count() == 0
        page.locator('.chat-quiz-comment').fill('')
        assert page.locator('.chat-quiz-close').is_enabled()
        assert page.evaluate("localStorage.length") == 0
        page.close()

        page = mount(context, "keyboard")
        field = page.locator(".chat-quiz-comment")
        field.fill("First line")
        field.press("Shift+Enter")
        field.press("End")
        page.keyboard.insert_text("Second line")
        assert page.evaluate("qaPosts.length") == 0
        field.press("Enter")
        assert page.evaluate("qaPosts.length") == 1
        assert page.evaluate("'option_index' in qaPosts[0]") is False
        assert field.evaluate("el=>el.readOnly") is True
        assert field.evaluate("el=>document.activeElement===el") is True
        # A positive WS fact beats the original request's later HTTP error.
        page.evaluate("""() => {qaDecision.applyQuizStateFrame(document.body,{task_id:'browser-quiz-task',quiz_id:'keyboard',
            state:'answered',answered_index:0,comment:'Winning server words'});
            qaFinish({ok:false,status:500,json:async()=>({})});}""")
        page.wait_for_function("document.querySelector('.chat-quiz-card').dataset.state==='answered'")
        assert page.locator(".chat-quiz-answer").inner_text() == "Owner's answer: Winning server words"
        assert page.locator(".chat-quiz-status-text").inner_text() == "You answered"
        observations = {"pending": pending, "restored_request": original, "reopened_without_post": True,
                        "explicit_retry_same_bytes": True, "keyboard_and_ws_error_precedence": True}
        # Real second window and real browser storage events, deliberately held
        # at the capture boundary to exercise typing before the component sees them.
        other = context.new_page()
        other.add_init_script("""window.qaQueued=[]; window.qaHoldStorage=true;
            addEventListener('storage',e=>{if(qaHoldStorage){qaQueued.push(e);e.stopImmediatePropagation();}},true);""")
        other.goto(origin + '/tests/fixtures/question_presentation_parity.json')
        other.evaluate(BOOT, {"id": "two-windows", "theme": theme})
        sender = mount(context, 'two-windows')
        sender.locator('.chat-quiz-comment').fill('Frozen across windows')
        sender.locator('.chat-quiz-option').nth(1).click()
        other.wait_for_function('qaQueued.length>0')
        assert other.locator('.chat-quiz-card').get_attribute('data-submission') == ''
        other.locator('.chat-quiz-comment').fill('Edit before queued notification')
        assert other.locator('.chat-quiz-comment').input_value() == 'Frozen across windows'
        assert other.evaluate('qaPosts.length') == 0
        frozen = sender.evaluate('qaPosts[0]')
        assert other.evaluate("JSON.parse(localStorage.getItem('ouro_quiz_reply_v1:'+JSON.stringify(['browser-quiz-task','two-windows'])+':attempt')).attempt") == frozen
        other.evaluate("qaHoldStorage=false; for(const e of qaQueued)dispatchEvent(e); qaQuiz={...qaQuiz,state:'answered',answered_index:1,comment:'Frozen across windows'}")
        sender.evaluate("qaFinish({ok:true,status:200,json:async()=>({ok:true,state:'answered',answered_index:1,comment:'Frozen across windows'})})")
        sender.wait_for_function("document.querySelector('.chat-quiz-card').dataset.state==='answered'")
        other.wait_for_function("document.querySelector('.chat-quiz-card').dataset.state==='answered'")
        observations['two_windows_delayed_real_storage_events'] = True
        sender.close(); other.close()

        forget = mount(context, 'forget')
        forget.locator('.chat-quiz-option').first.click()
        assert forget.locator('.chat-quiz-retry').is_visible() is False
        forget.evaluate('qaFinish({ok:false,status:500,json:async()=>({})})')
        forget.wait_for_function("document.querySelector('.chat-quiz-card').dataset.submission==='unknown'")
        forget.screenshot(path=str(evidence / f'{engine}-{theme}-unknown.png'))
        forget.locator('.chat-quiz-forget').click()
        assert 'does not cancel or retract' in forget.locator('[role=dialog]').inner_text()
        forget.locator('[data-confirm-cancel]').last.click()
        assert forget.locator('.chat-quiz-card').get_attribute('data-submission') == 'unknown'
        forget.locator('.chat-quiz-forget').click()
        forget.locator('[data-confirm-ok]').click()
        assert forget.locator('.chat-quiz-comment').evaluate('el=>el.readOnly') is False
        assert forget.evaluate('qaPosts.length') == 1
        assert forget.evaluate('localStorage.length') == 0
        forget.evaluate("qaDecision.applyQuizStateFrame(document.body,{task_id:'browser-quiz-task',quiz_id:'forget',state:'answered',answered_index:0,comment:'Server already saved this'})")
        assert forget.locator('.chat-quiz-answer').inner_text() == "Owner's answer: Server already saved this"
        observations['explicit_local_forget_cancel_then_confirm_no_post'] = True
        observations['forget_does_not_retract_later_canonical_answer'] = True
        context.close()
        context = None
    finally:
        if context is not None:
            context.close()
        if pw is not None:
            pw.stop()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        (evidence / f"{engine}-{theme}.json").write_text(json.dumps(observations, ensure_ascii=False, indent=2), encoding="utf-8")
