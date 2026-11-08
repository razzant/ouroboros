"""Real room-work overlay and text reflow: independent-review regressions.

Production modules and styles run in two headless engines. The page is a small local
consumer, not the whole SPA or a physical desktop; all requests are fulfilled offline.
"""
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.ui_browser
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def geometry_evidence(request, tmp_path):
    from tests.ci_evidence import output_dir

    root = output_dir(request.config)
    if root is None:
        return tmp_path
    target = root / "geometry" / request.node.name
    target.mkdir(parents=True, exist_ok=True)
    return target


@pytest.fixture(params=["chromium", "webkit"])
def page(request):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = getattr(playwright, request.param).launch(headless=True)
        try:
            yield browser.new_page(viewport={"width": 1000, "height": 760}, reduced_motion="reduce")
        finally:
            browser.close()


def _serve(page, html):
    def respond(route):
        name = urlsplit(route.request.url).path
        if name == "/":
            route.fulfill(status=200, content_type="text/html", body=html)
            return
        path = (ROOT / "web" / name.lstrip("/")).resolve()
        if not path.is_relative_to(ROOT / "web") or not path.is_file():
            route.fulfill(status=404, body="Not in the local geometry fixture")
            return
        kind = "text/javascript" if path.suffix == ".js" else "text/css"
        route.fulfill(status=200, content_type=kind, body=path.read_bytes())
    page.route("http://shared.test/**", respond)
    page.goto("http://shared.test/")


_WORK = '''<!doctype html><html data-theme="light" data-layout="docked"><head>
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css"><link rel="stylesheet" href="/room_work.css">
<style>html,body{margin:0;width:100%;height:100%}#page-chat{display:block;position:relative;width:900px;height:680px;margin:0}
.chat-page-header{height:56px;padding:0 24px;box-sizing:border-box}.app-page-title-row{display:flex}
#chat-messages{position:absolute;top:0;left:0;right:0;height:600px;overflow:auto;padding:0 24px;box-sizing:border-box}
#subject{position:absolute;top:78px;left:36px;width:740px;min-height:110px;margin:0;box-sizing:border-box}
#chat-input-area{position:absolute;top:610px;left:24px;right:24px;height:60px}
</style></head><body><div id="page-chat" class="page active">
<div id="chat-messages" class="chat-messages"><div id="subject" class="chat-bubble assistant"
 data-row-address="row:1@time#shared" data-room-revisions='{"1":7,"5551":3}'>
<div class="sender">Ouroboros</div><div class="message">This shared message must stay unread while the room's work list hides its words. Please read the complete instructions before continuing.</div></div></div>
<header class="chat-page-header"><div class="app-page-title-block"><div class="app-page-title-row">Main Chat</div></div>
<div class="app-page-actions"><span id="badge" class="status-badge thinking" data-tone="live" data-mark="moving">Working</span><button>Panic</button></div></header>
<div id="chat-input-area"><textarea id="chat-input" placeholder="Message Ouroboros"></textarea></div></div></body></html>'''


def test_room_work_sheet_blocks_read_and_dismissal_resamples(page, geometry_evidence):
    tmp_path = geometry_evidence
    _serve(page, _WORK)
    page.evaluate('''async () => {
        const {createRoomWork} = await import('/modules/room_work.js');
        const {createSharedReadTracker} = await import('/modules/read_ahead.js');
        const {readBand} = await import('/modules/project_read_state.js');
        const root=document.querySelector('#page-chat'), feed=document.querySelector('#chat-messages');
        const works = Array.from({length:3},(_,i)=>({work_id:'task'+i, room:1, name:'An existing task '+i,
          kind:'task', episode:{task_id:'task'+i,live:true}, activity:'working',transition:'none',call:'',
          because:[],tails:[],turn:[],release:'self',members:{ids:['task'+i]},controls:[],confirmed:true,mark:'moving',live:true}));
        window.workList=createRoomWork({page:root,statusBadge:document.querySelector('#badge'),messagesDiv:feed,
            reading:{stick:false,cancel(){},scroll(){}},works:()=>({works,complete:true,room:{activity:'working'}})});
        window.posts=[];
        window.tracker=createSharedReadTracker({chatId:1,feed,
            band:()=>readBand(feed,document.querySelector('.chat-page-header'),document.querySelector('#chat-input-area')),
            shown:()=>true,post:async address=>{window.posts.push(address);return true;}});
    }''')
    try:
        page.locator('.rw-trigger').click()
        page.wait_for_function("document.querySelector('.rw-shade').getBoundingClientRect().height > 150")
        facts = page.evaluate('''() => {
            const bubble=document.querySelector('#subject'), sheet=document.querySelector('.rw-shade');
            const box=bubble.getBoundingClientRect(), shade=sheet.getBoundingClientRect();
            const x=(box.left+box.right)/2, y=Math.min(box.bottom-10,shade.bottom-10);
            window.tracker.sample();
            return {posts:[...window.posts],role:sheet.getAttribute('role'),aria_modal:sheet.getAttribute('aria-modal'),
                top_is_sheet:sheet.contains(document.elementFromPoint(x,y)), box:box.toJSON(), shade:shade.toJSON()};
        }''')
        (tmp_path / "covered.json").write_text(json.dumps(facts, indent=2))
        page.screenshot(path=str(tmp_path / "covered.png"))
        assert facts["top_is_sheet"] and facts["role"] == "dialog"
        assert facts["aria_modal"] != "true", "exercise the actual nonmodal work-list path"
        assert facts["posts"] == [], "the words hidden by the work list were not read"
        page.evaluate("window.workList.close()")
        # No manual tracker.sample(): dismissal itself must create the next reading moment.
        page.wait_for_function("window.posts.length === 1")
        assert page.evaluate("window.posts") == ["row:1@time#shared"]
        page.screenshot(path=str(tmp_path / "uncovered.png"))
    finally:
        page.evaluate("window.tracker.dispose(); window.workList.destroy()")


_REFLOW = '''<!doctype html><html><style>
body{margin:0;font:16px/24px monospace}#feed{position:absolute;left:20px;top:100px;width:700px;height:96px;overflow:auto;background:#eee}
#bubble{position:relative;background:#dce8ff}#message{margin:0}#tail{height:500px}
</style><div id="feed"><div id="bubble" data-row-address="row:1@reflow#shared" data-room-revisions='{"1":7,"5551":3}'>
<div id="message" class="message"></div></div><div id="tail"></div></div></html>'''


def test_width_change_cannot_combine_coverage_of_different_word_layouts(page, geometry_evidence):
    tmp_path = geometry_evidence
    _serve(page, _REFLOW)
    result = page.evaluate('''async()=>{
        const {createSharedReadTracker}=await import('/modules/read_ahead.js');
        const feed=document.querySelector('#feed'),bubble=document.querySelector('#bubble'),message=document.querySelector('#message');
        const words=Array.from({length:56},(_,i)=>'step'+String(i+1).padStart(2,'0'));
        message.textContent=words.join(' ');
        function positions(width){
            bubble.style.width=width+'px';
            return words.map((word,i)=>{const range=document.createRange();range.setStart(message.firstChild,i*7);range.setEnd(message.firstChild,i*7+6);
                const rect=range.getBoundingClientRect();return {word,top:rect.top-bubble.getBoundingClientRect().top,bottom:rect.bottom-bubble.getBoundingClientRect().top};});
        }
        let selected;
        for(let oldWidth=400;oldWidth<680&&!selected;oldWidth+=1){
            const before=positions(oldWidth),height=bubble.getBoundingClientRect().height;
            if(height<=96)continue;
            for(let newWidth=oldWidth+1;newWidth<Math.min(oldWidth+80,700);newWidth+=1){
                const after=positions(newWidth),newHeight=bubble.getBoundingClientRect().height;
                const missed=before.filter((word,i)=>word.top>=96&&after[i].bottom<=96);
                if(height===newHeight&&missed.length){selected={oldWidth,newWidth,height,missed};break;}
            }
        }
        if(!selected)throw new Error('No stable-height word reflow found');
        bubble.style.width=selected.oldWidth+'px';
        const posts=[];
        const tracker=createSharedReadTracker({chatId:1,feed,band:()=>feed.getBoundingClientRect(),shown:()=>true,
            post:async address=>{posts.push(address);return true;}});
        tracker.sample(); const beforePosts=[...posts];
        bubble.style.width=selected.newWidth+'px';feed.scrollTop=96;
        tracker.sample(); const afterPosts=[...posts];
        feed.scrollTop=0;tracker.sample();const rereadPosts=[...posts];tracker.dispose();
        return {...selected,beforePosts,afterPosts,rereadPosts};
    }''')
    (tmp_path / "reflow.json").write_text(json.dumps(result, indent=2))
    page.screenshot(path=str(tmp_path / "reflow.png"))
    assert result["missed"] and result["beforePosts"] == []
    assert result["afterPosts"] == [], "unchanged height does not mean unchanged word layout"
    assert result["rereadPosts"] == ["row:1@reflow#shared"]
