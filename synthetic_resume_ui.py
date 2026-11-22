"""One private browser -> real HTTP Resume -> saved grant -> actual consumption.

Only the delegated engine and worker queue endpoint are stand-ins. Run as a scratch
pytest module under the candidate's tests/conftest.py through scripts/safe_test.py.
The fixture server binds an ephemeral loopback socket; no running installation is used.

The scenario: five distinct unknown helper starts across two members of one paused
tree, one owner press that refuses (409) and names all five, a Cancel that changes
nothing, then exactly one "Resume anyway" (200) for the same saved task. Screenshots
and the synthetic facts go to the evidence directory safe_test selected; only the one
PNG is ever meant to leave the runner. The consumed proof names the grant's OWN five
choice_ids after the worker pass — the refusal's HTTP offer is kept only as an
equality check (offer == sent == 200 reply == consumed grant), never as the source
of the recorded list. Every address below is a synthetic fixture value.
"""
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest


@pytest.mark.ui_browser
def test_addressed_resume_from_activity(tmp_path, monkeypatch):
    import uvicorn
    from playwright.sync_api import expect, sync_playwright
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Mount, Route
    from starlette.staticfiles import StaticFiles
    from ouroboros import delegate_custody as dc
    from ouroboros.gateway.tasks import api_task_resume, _queue_snapshot
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from tests._owner_pause_ended_helpers import _tail_tree, _tick
    from tests._work_projection_shared import state_fields
    from tests.test_activity_outcomes_browser import HTML, WEB
    from tests.test_delegated_start_resume import _consume

    # safe_test owns this run: the boundary marker is set and the queue's drive root is
    # this test's own tmp_path, so no real installation's data root can be reached.
    assert os.environ.get('OUROBOROS_PYTEST_ACTIVE') == '1'
    # The launcher declares the persistent UI evidence directory: safe_test passes
    # OUROBOROS_UI_EVIDENCE_DIR through its boundary (and the CI step declares it
    # explicitly), so the picture and the synthetic facts land in one known place.
    evidence_dir = os.environ.get('OUROBOROS_UI_EVIDENCE_DIR')
    assert evidence_dir, 'the launcher must declare OUROBOROS_UI_EVIDENCE_DIR before the scenario runs'
    evidence = Path(evidence_dir).resolve()
    evidence.mkdir(exist_ok=True, parents=True)
    req = {'prompt':'Count three synthetic ribbons.', 'access':'readonly', 'mode':'ask', 'primaryHarness':'fixture', 'maxSeconds':30}
    choices = [('root-1' if i % 2 == 0 else 'child-dead', f'fixture-request-{i:02d}') for i in range(5)]
    assert len({iid for _, iid in choices}) == 5, "the five helper starts need five distinct invocation ids"
    def add(root):
        for member, iid in choices:
            assert dc.record_start_requested(root, task_id=member, root_task_id='root-1', invocation_id=iid,
                idempotency_key=iid, request=req, project_id='', project_owned=False, route='fixture',
                daemon_store_id='store-a', store_identity_unproven=False)
    queue, workers, engine = _tail_tree(tmp_path, monkeypatch, chat_id=1, tail=add)
    assert queue.DRIVE_ROOT == tmp_path
    assert request_owner_pause('root-1', request_id='synthetic-pause')['ok']
    _tick(queue)
    # Use the actual custody observation at the worker's park. The old helper supplied [].
    from ouroboros import budget_pause
    from tests._budget_pause_exact_helpers import _loop_ctx, _fast_hold, _supervisor_ctx
    from supervisor.events_budget import install_exact_budget_pause
    ctx,limit = _loop_ctx(tmp_path,'root-1'); ctx.root_task_id='root-1'
    with monkeypatch.context() as park:
        _fast_hold(park,budget_pause)
        with pytest.raises(budget_pause.BudgetPauseRequested) as stopped:
            budget_pause.enter_owner_pause(limit)
        budget_pause.end_dispatch_fence('root-1')
    row = stopped.value.pause
    (evidence/'park-unknown.json').write_text(json.dumps({'state':row.get('state'), 'settlement':row.get('settlement'), 'external_runs':row.get('external_runs')}))
    assert {r['invocation_id'] for r in row['external_runs']['runs']} == {i for task,i in choices if task=='root-1'}
    install_exact_budget_pause(_supervisor_ctx(tmp_path,workers,queue,[],[]),'root-1',budget_pause.exact_pause_marker(row)['checkpoint'])
    _tick(queue)
    assert queue.persist_queue_snapshot(reason='fixture for a real browser request')
    posts = []
    async def activity(request): return HTMLResponse(HTML)
    async def state(request):
        return JSONResponse({'supervisor_ready':True,'bg_consciousness_enabled':False,'projects':[],
             'project_chat_ids':[], **state_fields(tmp_path,direct_turns=[]),'active_chat_activities_complete':True})
    async def tasks(request): return JSONResponse({'tasks':[],'queue':_queue_snapshot(tmp_path)})
    async def detail(request):
        row=load_task_result(tmp_path,request.path_params['task_id'])
        return JSONResponse(public_task_result(row) if row else {}, status_code=200 if row else 404)
    async def schedules(request): return JSONResponse({'tasks':[]})
    async def resume(request):
        assert request.path_params['task_id']=='root-1'
        sent=await request.body()
        result=await api_task_resume(request)
        posts.append({'sent':json.loads(sent) if sent else None,'status':result.status_code,'reply':json.loads(result.body)})
        return result
    app=Starlette(routes=[Route('/fixture',activity),Route('/api/state',state),Route('/api/tasks',tasks),
       Route('/api/tasks/{task_id}/resume',resume,methods=['POST']),Route('/api/tasks/{task_id}',detail),
       Route('/api/schedules',schedules),Mount('/',app=StaticFiles(directory=WEB))])
    sock=socket.socket(); sock.bind(('127.0.0.1',0)); sock.listen(16)
    origin=f'http://127.0.0.1:{sock.getsockname()[1]}'
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=0,ws='none',lifespan='off',log_level='warning',access_log=False))
    worker=threading.Thread(target=server.run, kwargs={'sockets':[sock]},daemon=True); worker.start()
    page_errors=[]
    try:
        for _ in range(100):
            if server.started or not worker.is_alive(): break
            time.sleep(.05)
        assert server.started
        with sync_playwright() as pw:
            # Ask Playwright for Chromium's own sandbox wherever the platform has one
            # (the Linux CI runner); never a --no-sandbox fallback and never a host sysctl.
            browser=pw.chromium.launch(headless=True, chromium_sandbox=True)
            try:
                page=browser.new_page(viewport={'width':1100,'height':750}, reduced_motion='reduce')
                page.route('**/*',lambda route: route.continue_() if route.request.url.startswith(origin+'/') else route.abort())
                page.on('pageerror',lambda error:page_errors.append(str(error)))
                page.goto(origin+'/fixture?theme=light'); page.wait_for_function('window.ready === true')
                activity_text=page.locator('body').inner_text()
                (evidence/'activity-open.txt').write_text(activity_text)
                def open_dialog():
                    page.locator('[data-act="task-control"][data-id="root-1"]').click()
                    page.get_by_role('menuitem',name='Resume',exact=True).click()
                    d=page.get_by_role('dialog').filter(has=page.get_by_text('Resume anyway',exact=True)); expect(d).to_be_visible()
                    return d
                dialog=open_dialog()
                before=dialog.inner_text(); page.screenshot(path=str(evidence/'unknown-starts-confirm.png'),animations='disabled')
                (evidence/'unknown-starts-dialog.txt').write_text(before,encoding='utf-8')
                listed=posts[-1]['reply']['acceptable_unknown']; assert len(listed)==len(set(listed))==5
                for member,iid in choices:
                    assert member in before and iid in before, (member,iid,before)
                    assert any(member in cid and iid in cid for cid in listed)
                assert 'spend' in before.lower() and ('running' in before.lower() or 'working' in before.lower())
                dialog.get_by_role('button',name='Cancel',exact=True).click(); expect(dialog).to_have_count(0)
                assert len(posts)==1 and not load_task_result(tmp_path,'root-1')['budget_pause'].get('grant')
                dialog=open_dialog()
                with page.expect_response(lambda r:r.url.endswith('/api/tasks/root-1/resume') and r.request.method=='POST') as ack:
                    dialog.get_by_role('button',name='Resume anyway',exact=True).click()
                assert ack.value.status==200, ack.value.text()
                assert [p['status'] for p in posts]==[409,409,200],posts
                assert set(posts[-1]['sent']['accept_unknown'])==set(listed) and posts[-1]['sent']['reason'].strip()
                assert posts[-1]['reply']['ok'] and posts[-1]['reply']['task_id']=='root-1'
            finally:
                browser.close()
    finally:
        server.should_exit=True; worker.join(10); sock.close()
        (evidence/'unknown-starts-http.json').write_text(json.dumps({'posts':posts,'page_errors':page_errors},indent=2),encoding='utf-8')
    assert not worker.is_alive() and not page_errors
    _consume(tmp_path,workers,monkeypatch)
    saved=load_task_result(tmp_path,'root-1')['budget_pause']
    assert saved['grant']['consumed_at'] and saved['state']=='resumed'
    assert all(dc.invocation_record(tmp_path,iid)['state']=='pending' for _,iid in choices)
    # The proof is built from the CONSUMED grant, not the HTTP offer: the 200 reply's own
    # decisions and the grant saved under this same root-1 must carry exactly the same five
    # distinct addresses the refusal named and the button sent — nothing synthesized after.
    responded=[d['choice_id'] for d in posts[-1]['reply']['accepted_start_risks']]
    consumed=[d['choice_id'] for d in saved['grant']['accepted_start_risks']]
    assert len(consumed)==len(set(consumed))==5 and len(responded)==len(set(responded))==5
    assert set(consumed)==set(responded)==set(posts[-1]['sent']['accept_unknown'])==set(listed)
    assert all(d['resuming_task_id']=='root-1' for d in saved['grant']['accepted_start_risks'])
    (evidence/'unknown-starts-consumed.json').write_text(json.dumps({'same_id':'root-1','phase':saved['state'],
        'grant_id':saved['grant']['grant_id'],'consumed_at':saved['grant']['consumed_at'],
        'accepted_ids':consumed,'creates':engine.starts,'cancel_requests':engine.cancels},indent=2),encoding='utf-8')

    assert '5 external runs may still be running' in activity_text, activity_text
