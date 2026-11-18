"""Disposable Linux Stop checks. All browser inputs are invented tiles; no actual device input, provider or extension delivery."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

# A single reviewed binding in the one-off workflow, not a moving branch tip.
BASE = os.environ.get('STOP_BASE_SHA', '')
TREE = os.environ.get('STOP_EXPECTED_TREE', '')
TASK, TOOL = 'public-mosaic-example', 'ext_example_check_tile'
OWNING = '''admission_close_unconfirmed batch4_extension_completion builtin_skill_source chat_logs_ui
conversation_activity_block extension_call_stop extension_plugin_api extension_plugin_api_matrix extension_process_runner
late_review_marks lc2_owner_facades live_contracts loop_tool_control memory_reader_plugin owner_facing_honesty
observability_outcomes_v2 outcome_tool_error_axes place_guard post_task_inputs project_plain_rows rail_cause_coverage record_contract reflection_trace_listing
result_review_desktop_effects task_summary task_tool_metrics tool_call_log tool_result unix_computer_use_ax_exposure
unix_computer_use_call_snapshot unix_computer_use_effects unix_computer_use_element_act unix_computer_use_key_stop
unix_computer_use_one_process unix_computer_use_paste_chord unix_computer_use_remote_targeting unix_computer_use_skill
unix_computer_use_targeting unix_computer_use_window_capture verification_check_facts verify_artifact_lifecycle windows_computer_use_delivery'''.split()
STRUCTURE = '''architecture_facts contracts docs_sync domain_manifest domain_report gateway_parity generated_inventories
hermetic_data_root packaging_sync preflight_process_containment process_custody safety_policy smoke'''.split()
SIZE = ['smoke', 'reference_book_budgets', 'repo_health_smoke']
NODE = 'tool_call_record verification_check_row owner_stop_card log_category history_retention host_notice_presentation i18n_catalog'.split()
PY_GROUPS = {'owning': OWNING, 'structure': STRUCTURE, 'size': SIZE}
# These four require an actual macOS framework, compiler or pasteboard; they are not Linux obligations.
MAC_ONLY = {
    ('unix_computer_use_one_process', 'test_an_axurl_value_converts_to_its_absolute_string'),
    ('unix_computer_use_paste_chord', 'test_the_paste_chord_program_compiles_without_running'),
    ('unix_computer_use_paste_chord', 'test_the_local_clipboard_write_is_utf8_and_tells_pbcopy_so_whatever_locale_the_process_inherited'),
    ('unix_computer_use_window_capture', 'test_real_pyobjc_bridge_registers_handlers_configures_encodes_and_waits'),
}
# The changed branches must actually execute, including every discriminating parameter, not merely be collected.
MUST_PASS = {
    'test_async_handler_freezes_its_receipt_before_loop_callbacks': 4,
    'test_async_timeout_result_stays_while_custody_tracks_actual_unwind': 4,
    'test_a_plain_step_refusal_is_no_ledger_failure_while_an_independent_error_stays_one': 1,
    'test_a_started_send_refused_a_further_step_carries_its_destination_facts': 1,
    'test_child_result_cap_and_stop_are_independent': 12, 'test_reader_enforces_payload_limit_independently_of_writer': 3,
    'test_child_deadline_keeps_its_timeout_before_any_result_handoff': 1, 'test_small_tool_result_uses_the_same_stop_allowance_on_write_and_read': 1,
    'test_the_calls_record_decides_again_after_the_read_whatever_the_read_found': 6,
    'test_missing_task_id_keeps_its_lifecycle_binding_across_tool_transport': 2,
    'test_actual_key_keeps_stop_and_independent_failure_distinct': 3, 'test_failure_and_timeout_remain_unresolved_beside_the_host_refusal': 2,
    'test_a_refusal_preserves_report_failure_and_keeps_a_non_error_completion_blocked': 2,
    'test_key_refused_before_the_mac_activation_helper_starts': 3, 'test_a_stop_before_the_first_chord_sends_nothing_and_a_switch_before_it_keeps_its_mark': 3,
    'test_osworld_a_stop_before_the_first_chord_sends_nothing_and_the_switch_keeps_its_mark': 1,
    'test_ssh_key_checks_after_inventory_and_again_before_input': 2, 'test_x11_key_checks_stop_after_resolution_and_before_input': 2,
    'test_host_refused_step_is_no_tool_error_but_an_independent_failure_stays_one': 1,
    'test_a_clean_stop_reads_policy_denied_in_the_trace_summary': 1, 'test_a_window_capture_is_capture_only_and_keyed_to_its_window': 1,
}
RESEED = 'test_call_skill_release_reseeds_the_old_payload_but_preserves_owner_edits'
SEED = 'e471e380e6de1e7f728e0fffbb4457667422ebf4'
LATE_TESTS = ['reload draws the refused-step row under its live key, and a late live frame stays one row',
              'a refusal settling after the terminal keeps its own row, without reopening the task',
              'an independent failure settling after the terminal keeps its red row, refusal in its body',
              'a plain late failure without the host mark keeps folding, with no row of its own',
              'an independent error beside the refusal stays red on reload, with the refusal in its body']


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def identity():
    if not __debug__ or sys.platform != 'linux' or Path.cwd().resolve() != Path(git('rev-parse','--show-toplevel')).resolve():
        raise ValueError('Use the disposable Linux product checkout with assertions enabled')
    if not all(re.fullmatch('[0-9a-f]{40}', v) for v in (BASE,TREE)): raise ValueError('Missing explicit base/tree binding')
    head = git('rev-parse', 'HEAD')
    assert git('write-tree') == TREE and git('cat-file', '-t', BASE) == 'commit'
    if os.environ.get('GITHUB_ACTIONS') == 'true' or os.environ.get('STOP_PRODUCT_SHA'):
        assert head == os.environ['STOP_PRODUCT_SHA'] and git('rev-parse', 'HEAD^{tree}') == TREE
    subprocess.run(['git', 'diff', '--quiet'], check=True)
    return {'head': head, 'tree': TREE, 'base': BASE}


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def pytest_report(path, phase):
    rows = list(ET.parse(path).iter('testcase'))
    passed, skips, failed, modules = [], [], [], set()
    for row in rows:
        cls, name = row.get('classname', ''), row.get('name','')
        module = next((v for v in cls.split('.') if v.startswith('test_')), cls).removeprefix('test_')
        tag = name.split('[', 1)[0]; key = (module, tag); modules.add(module)
        skip = row.find('skipped')
        if skip is not None: skips.append({'module':module, 'test':name, 'reason':skip.get('message','')[:240]})
        elif row.find('error') is not None or row.find('failure') is not None: failed.append(f'{cls}::{name}')
        else: passed.append((key, name))
    counts = Counter(key[1] for key,_ in passed)
    missing = sorted(set(PY_GROUPS[phase]) - {key[0] for key,_ in passed})
    issues = ['no passing witness in '+m for m in missing]
    issues += ['unexpected skip: '+r['test'] for r in skips if (r['module'],r['test'].split('[',1)[0]) not in MAC_ONLY]
    if phase == 'owning':
        issues += [f'{test}: {counts[test]} passed, expected {n}' for test,n in MUST_PASS.items() if counts[test] != n]
        for edited in ('False','True'):
            wanted = f'{RESEED}[{edited}-unix_computer_use-{SEED}-0.6.0]'
            if not any(k[0]=='builtin_skill_source' and s==wanted for k,s in passed): issues.append('reseed case absent: '+wanted)
    if phase == 'size' and len(passed) != 7: issues.append(f'{len(passed)} passed size cases, expected 7')
    return {'tests_in_xml':len(rows), 'passed_including_subtests':len(passed), 'failed':failed, 'skipped':skips,
            'executed_modules':sorted(modules), 'issues':issues}


def node_report(path):
    text = path.read_text(errors='replace'); counts = {}
    for key in ('tests','pass','fail','cancelled','skipped','todo'):
        match = re.findall(rf'^# {key} (\d+)\s*$', text, re.M)
        counts[key] = int(match[0]) if len(match)==1 else None
    # These modules use top-level node:test declarations. Added cases raise the floor;
    # a renamed late-row scenario must still be deliberately reviewed below.
    pattern = re.compile(r'''^test\(\s*['\"]''', re.M)
    declared = sum(len(pattern.findall(Path('web/tests', n+'.test.js').read_text())) for n in NODE)
    good = declared >= 68 and counts['tests'] is not None and counts['tests'] >= declared and counts['pass']==counts['tests'] and all(counts[k]==0 for k in ('fail','cancelled','skipped','todo'))
    issues = [] if good else [f'TAP totals {counts}, explicit test declarations {declared}']
    for title in LATE_TESTS:
        if not re.search(r'^ok \d+ - '+re.escape(title)+r'\s*$', text, re.M): issues.append('missing PASS: '+title)
    return {'counts':counts,'failed':[x[:260] for x in text.splitlines() if x.startswith('not ok ')],'issues':issues}


def phase(name, out):
    binding = identity(); assert Path.cwd() not in out.parents and out != Path.cwd()
    raw, public = out/'raw'/name, out/'public'; raw.mkdir(parents=True, exist_ok=False); public.mkdir(parents=True,exist_ok=True)
    result = {**binding, 'phase':name, 'status':'failed'}
    try:
        if name in PY_GROUPS:
            # Only the separate synthetic browser phase opens a UI. No provider, integration, or native-device lane.
            mark = 'size_ratchet' if name=='size' else 'not size_ratchet and not integration and not browser and not ui_browser and not ui_browser_docker and not portable_detail and not skill_smoke and not required_provider and not optional_provider'
            cmd = [sys.executable,'-m','pytest','-n','0','-o','addopts=','--continue-on-collection-errors','-q','--tb=short','-rs',
                   '--timeout=180' if name=='size' else '--timeout=120','-m',mark,'--junitxml='+str(raw/'junit.xml'),
                   *['tests/test_'+v+'.py' for v in PY_GROUPS[name]]]
            if name=='size':
                # The workflow push carries the QA overlay, not the change being measured.
                # Run the repository's explicit-base comparison on the detached product, in blocking mode.
                cmd = ['env', '-u', 'GITHUB_EVENT_NAME', 'OURO_SIZE_RATCHET_BASE_REF='+BASE, 'OURO_BOOK_GROWTH_MODE=block', *cmd]
        elif name=='node':
            cmd = [shutil.which('node') or 'node','--test','--test-concurrency=1','--test-reporter=tap',*['web/tests/'+v+'.test.js' for v in NODE]]
        elif name=='browser': cmd = [sys.executable,'-I',str(Path(__file__).resolve()),'_browser',str(public/'mosaic')]
        else: raise ValueError(name)
        with (raw/'stdout').open('xb') as so, (raw/'stderr').open('xb') as se:
            code = subprocess.run([sys.executable,'-I','-S','scripts/safe_test.py','--timeout','2600' if name=='owning' else '1200',
                                   '--evidence-dir',str(raw/'records'),'--',*cmd],stdout=so,stderr=se).returncode
        result['exit'] = code
        if name in PY_GROUPS: result.update(pytest_report(raw/'junit.xml',name))
        if name=='owning':
            tested={'tests/test_'+v+'.py' for v in OWNING+STRUCTURE+SIZE}|{'web/tests/'+v+'.test.js' for v in NODE}
            delta=git('diff','--diff-filter=ACMRT','--name-only',BASE,TREE).splitlines()
            homes={v for v in delta if re.fullmatch(r'tests/test_[^/]+\.py|web/tests/[^/]+\.test\.js',v)}
            result['issues']+=['changed test not selected: '+v for v in sorted(homes-tested)]
        if name=='node': result.update(node_report(raw/'stdout'))
        if name=='browser':
            obs = json.loads((public/'mosaic'/'observations.json').read_text()); states = {(x['phase'],x['theme']) for x in obs['states']}
            assert states=={(p,t) for p in ('live','done-before-late','first-late','reload','duplicate-same-key') for t in ('light','dark')}
            assert len(obs['history'])==3 and not obs['page_errors'] and not obs['stop_unconfirmed'] and obs['browser_complete']
            assert {(s['phase'],s['theme']) for s in obs['states'] if s.get('collapsed',{}).get('phase')=='Done'}=={(p,t) for p in ('first-late','reload') for t in ('light','dark')}
        if code or result.get('failed') or result.get('issues'): raise RuntimeError('Phase or coverage check failed')
        identity(); result['status']='passed'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'[:700]
        # The raw streams are from a clean public checkout on a disposable machine. Keep them in the job log, not in the downloadable bundle.
        for stream in ('stdout','stderr'):
            p=raw/stream
            if p.is_file(): print(f'--- {name} {stream} tail ---\n'+p.read_text(errors='replace')[-4800:])
        raise
    finally:
        dump(public/(name+'.json'), result)
        print(name, result['status'], 'exit', result.get('exit','not read'), 'coverage',result.get('issues',[]), flush=True)


def writer_case(root, data, color, message, reason, failed):
    """Only the harmless producer and its Stop fact are fixtures. The executor writes the actual rows and both live frames."""
    from queue import Empty, SimpleQueue
    from threading import Lock
    from ouroboros import extension_plugin_api as api, loop_tool_execution as execution
    from ouroboros.tools.extension_dispatch import _extension_completion
    from supervisor.log_addressing import address_task_event
    from supervisor import message_bus
    from tests._tool_call_log_shared import _Registry
    logs=root/color/'logs'; logs.mkdir(parents=True)
    signal={'phase_lock':Lock(),api.STEP_STOP:{'reason':reason,'note':api._STOP_WORDS[reason],'refusals':1}}
    body=json.dumps({'ok':not failed, 'tile':color, 'message':message})
    registry=_Registry(root/color, lambda name,args: _extension_completion(body,'',handoff=signal))
    q=SimpleQueue(); registry._ctx.event_queue=q; registry._ctx.task_metadata={'root_task_id':TASK,'delegation_role':'root'}
    result=execution._execute_with_timeout(registry, {'id':'tile-'+color,'function':{'name':TOOL,'arguments':'{}'}},logs,20,TASK)
    recorded=[json.loads(s) for s in (logs/'tools.jsonl').read_text().splitlines() if s]
    assert [r['type'] for r in recorded]==['tool_call_started','tool_call']; settle=recorded[1]
    bucket, code = ('tool_reported_failure','TOOL_REPORTED_FAILURE') if failed else ('blocked','OWNER_OPERATION_REFUSED')
    meta={'refused_step':True,'control_reason':reason}; inv=settle['invocation_id']; assert settle['tool_result_code']==result['tool_result'].code==code
    frames=[]; bridge=message_bus.LocalChatBridge(); bridge._broadcast_fn=frames.append
    while True:
        try: evt=q.get_nowait()
        except Empty: break
        if evt['type']=='log_event':
            tagged=address_task_event({TASK:{'task':{'id':TASK,'chat_id':1,'root_task_id':TASK}}},data,dict(evt['data']))
            bridge.push_log(tagged)  # the actual supervisor → browser envelope
    assert [f['data']['type'] for f in frames]==['tool_call_started','tool_call','tool_call_finished']
    assert all((f['type'],f['chat_id'],f['data']['task_id'],f['data']['invocation_id'])==('log',1,TASK,inv) for f in frames)
    assert {k:settle['tool_result_meta'][k] for k in meta}==meta and frames[-1]['data']['tool_result_meta']==meta
    assert 'tool_result_code' not in frames[-1]['data'] and frames[-2]['data']['tool_result_code']==code
    assert result['tool_result'].producer_text==body and len(result['tool_result'].host_annotations)==1
    for row in (settle,frames[-2]['data'],frames[-1]['data']):
        assert row['is_error'] is True and row['status']==bucket and message in row['result_preview']
        assert 'OWNER_OPERATION_REFUSED:' in row['result_preview']
    return {'color':color,'message':message,'reason':reason,'bucket':bucket,'code':code,'inv':inv,'rows':recorded,'frames':frames}


def browser(out):
    info=identity(); assert os.environ.get('OUROBOROS_PYTEST_ACTIVE')=='1'
    scratch=Path(os.environ['OUROBOROS_TEST_TEMP_ROOT']); sys.path.insert(0,str(Path.cwd()))
    from types import SimpleNamespace
    from devtools.benchmarks.common.server_runner import seed_owner_state
    from ouroboros import dialogue_store
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import write_task_result
    from ouroboros.terminal_projection import _project_row
    from ouroboros.tool_call_log import append_call_row
    from playwright.sync_api import expect, sync_playwright
    from supervisor import message_bus
    from tests.system_e2e.harness import keyless_settings, start_server
    from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _OBSERVE_STATE_READS, _emit_ws_frame, _wait_socket_open_quiescent
    out.mkdir(exist_ok=False); instance=scratch/'mosaic-instance'; data=instance/'data'; data.mkdir(parents=True)
    # All model slots point to an unused local endpoint or are empty. There is no real provider key or queued task.
    settings=keyless_settings(SimpleNamespace(base_url='http://127.0.0.1:9/v1'),OUROBOROS_MAX_WORKERS=1,
                              OUROBOROS_STARTER_SKILLS_TURN_ON=False,OUROBOROS_SKILL_SOURCES='[]',MCP_ENABLED=False,MCP_SERVERS=[])
    run={**info,'states':[],'history':[],'page_errors':[],'browser_complete':False,'stop_unconfirmed':None}
    server=None; page=None
    try:
        with acting_as(data): seed_owner_state(data)
        server=start_server(Path.cwd(),instance,settings); assert server.attestation['ok'] and server.attestation['repo_head']==info['head']
        with acting_as(data), sync_playwright() as pw:
            message_bus.DATA_DIR=data  # the routing address is this disposable install
            location=os.environ.get('PLAYWRIGHT_BROWSERS_PATH'); binary=Path(pw.chromium.executable_path)
            if location: assert binary.is_file() and binary.resolve().is_relative_to(Path(location).resolve()), 'missing pinned Chromium inside safe_test'
            elif not binary.is_file(): binary=Path(shutil.which('chromium') or '/nonexistent-chromium')
            assert binary.is_file() and os.access(binary,os.X_OK)
            node=os.environ.get('PLAYWRIGHT_NODEJS_PATH'); assert node and Path(node).is_file() and os.access(node,os.X_OK)
            run['node']=subprocess.check_output([node,'--version'],text=True).strip(); run['python']=sys.version.split()[0]
            client=pw.chromium.launch(headless=True,executable_path=str(binary)); run['chromium']=client.version
            try:
                page=client.new_page(viewport={'width':1280,'height':960},locale='en-US',reduced_motion='reduce')
                page.on('pageerror',lambda e:run['page_errors'].append(str(e)[:300])); page.add_init_script(f'({_CAPTURE_TEST_SOCKET})(); ({_OBSERVE_STATE_READS})();')
                page.goto(server.base_url,wait_until='domcontentloaded'); _wait_socket_open_quiescent(page)
                card=page.locator(f'.chat-live-card[data-task-id="{TASK}"]'); titles=card.locator('.chat-live-line-title')
                def emit(frame): _emit_ws_frame(page,frame)
                def persist(row): assert append_call_row({},data/'logs',dict(row))['task_log']
                def capture(stage, cases, done, fold):
                    expect(card).to_be_visible(timeout=30000)
                    if card.get_attribute('data-expanded')!='1': card.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
                    if done: expect(card.locator('[data-live-phase]').first).to_have_text('Done'); expect(card).to_have_attribute('data-finished','1')
                    for theme in ('light','dark'):
                        page.evaluate('v=>document.documentElement.dataset.theme=v',theme)
                        card.screenshot(path=str(out/f'{stage}-{theme}.png'),animations='disabled')
                        for case in cases:
                            tone='error' if case['bucket']=='tool_reported_failure' else ('warn' if case['reason']=='stop_unreadable' else 'info')
                            title=('One of the steps failed' if tone=='error' else ("A further step was refused: the task's Stop could not be read" if tone=='warn' else 'A further step was refused by Stop'))+' · '+TOOL
                            row=card.locator('.chat-live-line.'+tone).filter(has=page.locator('.chat-live-line-title',has_text=re.compile('^'+re.escape(title)+'$')))
                            expect(row).to_have_count(1); expect(row).to_contain_text(case['message'])
                            if tone=='error': expect(row).to_contain_text('A further step was refused by Stop')
                        special=card.locator('.chat-live-line').filter(has=page.locator('.chat-live-line-title',has_text=re.compile('^(One of the steps failed|A further step was refused)')))
                        expect(special).to_have_count(len(cases)); keys=special.evaluate_all('(rs)=>rs.map(r=>r.dataset.liveLineKey)'); assert len(set(keys))==len(keys) and all(keys)
                        if fold: expect(titles.filter(has_text=re.compile('^'+re.escape(fold)+'$'))).to_have_count(1)
                        state={'phase':stage,'theme':theme,'titles':titles.all_text_contents(),'individual_rows':len(keys)}
                        if stage in ('first-late','reload'):
                            door=card.locator(':scope > [data-live-summary-button] [data-live-disclosure]'); summary_bar=card.locator(':scope > [data-live-summary-button]')
                            expect(door).to_have_attribute('aria-expanded','true'); door.click(); expect(card).to_have_attribute('data-expanded','0'); expect(door).to_have_attribute('aria-expanded','false')
                            expect(card.locator('[data-live-phase]').first).to_have_text('Done'); expect(card.locator('[data-live-phase]').first).to_be_visible()
                            card.screenshot(path=str(out/f'{stage}-{theme}-collapsed.png'),animations='disabled'); state['collapsed']={'phase':'Done','header':summary_bar.inner_text()}
                            door.click(); expect(card).to_have_attribute('data-expanded','1'); expect(door).to_have_attribute('aria-expanded','true'); expect(special).to_have_count(len(cases)); expect(titles.filter(has_text=re.compile('^'+re.escape(fold)+'$'))).to_have_count(1)
                        run['states'].append(state)
                progress={'task_id':TASK,'chat_id':1,'content':'Checking a mosaic of three colored tiles.','ts':datetime.now(timezone.utc).isoformat(),'narration':True}
                dialogue_store.append(data,'progress',progress); emit({**progress,'type':'chat','role':'assistant','is_progress':True})
                examples=[('blue','Blue recorded; amber not started.','cancelled',False),('green','Green failed its format check.','cancelled',True),('violet','Violet left untouched.','stop_unreadable',False)]
                cases=[writer_case(scratch/'writer',data,*v) for v in examples]
                for row in cases[0]['rows']: persist(row)
                for frame in cases[0]['frames']: emit(frame)
                capture('live',cases[:1],False,'1 tool call'); assert card.get_attribute('data-finished')!='1'
                for later in cases[1:]: persist(later['rows'][0]); emit(later['frames'][0])
                msg='The example mosaic is closed.'; message_bus.log_chat('out',1,1,msg,task_id=TASK,drive_root=data,require_write=True)
                terminal=write_task_result(data,TASK,'completed',chat_id=1,root_task_id=TASK,delegation_role='root',_is_direct_chat=True,result=msg,tool_calls=3,tool_errors=1)
                summary=_project_row(TASK,terminal,{}, {'chat_id':1}); dialogue_store.append(data,'chat',summary)
                emit({**summary,'type':'chat','role':'system','system_type':'task_summary','content':summary['text']})
                capture('done-before-late',cases[:1],True,None)
                # Delay the writer's two real settlements until the DOM proves Done. No projected history can supply them yet.
                for later in cases[1:]: emit(later['frames'][1]); emit(later['frames'][2])
                capture('first-late',cases,True,'3 tool calls · 1 error')
                for later in cases[1:]: persist(later['rows'][1])
                page.reload(wait_until='domcontentloaded'); _wait_socket_open_quiescent(page); capture('reload',cases,True,'3 tool calls · 1 error')
                reply=page.request.get(server.base_url+'/api/chat/history?chat_id=1'); assert reply.ok
                history={o['key']:o for m in reply.json()['messages'] if m.get('task_id')==TASK for o in (m.get('tool_evidence') or {}).get('observations',[]) if o.get('fact')=='settled'}
                assert set(history)=={'tool:'+TASK+':'+c['inv'] for c in cases}
                for c in cases:
                    h=history['tool:'+TASK+':'+c['inv']]; assert h['recorded_status']==c['bucket'] and h['status']==('error' if c['bucket']=='tool_reported_failure' else 'held')
                    assert h['result_preview']==c['rows'][1]['result_preview'] and h['tool_result_meta']=={'refused_step':True,'control_reason':c['reason']}
                    run['history'].append({'tile':c['color'],'writer_code':c['code'],'recorded_status':h['recorded_status'],'history_status':h['status'],'message':c['message']})
                for c in cases: emit(c['frames'][2])  # byte-for-byte duplicate of each original finished frame, never another call
                capture('duplicate-same-key',cases,True,'3 tool calls · 1 error'); assert not run['page_errors']; run['browser_complete']=True
            finally: client.close()
    finally:
        if server is not None: server.stop(); run['stop_unconfirmed']=server.stop_unconfirmed
        dump(out/'observations.json',run)
        if run['stop_unconfirmed']: raise RuntimeError('Disposable server exit not confirmed')


def bundle(out):
    # Ship only the public test summaries and these islanded, synthetic card pictures.
    public=out/'public'; phases=(*PY_GROUPS,'node','browser'); names=['binding.json',*[n+'.json' for n in phases],'mosaic/observations.json']
    for p in ('live','done-before-late','first-late','reload','duplicate-same-key'):
        for t in ('light','dark'):
            names.append(f'mosaic/{p}-{t}.png')
            if p in ('first-late','reload'): names.append(f'mosaic/{p}-{t}-collapsed.png')
    missing=[]
    with zipfile.ZipFile(out/'stop-check.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:
            f=public/n
            if not f.is_file() or f.is_symlink() or not f.resolve().is_relative_to(public.resolve()): missing.append(n); continue
            z.write(f,arcname=n)
        z.writestr('inventory.json',json.dumps({'expected':names,'missing':missing,'base':BASE,'tree':TREE,'type':'synthetic Linux QA'},indent=2))
    if missing: raise RuntimeError('Evidence not produced: '+', '.join(missing))
    if any(json.loads((public/(n+'.json')).read_text()).get('status')!='passed' for n in phases): raise RuntimeError('One or more independent phases failed')


if __name__=='__main__':
    if len(sys.argv)!=3: raise SystemExit('Use: python stop_lane.py PHASE OUTPUT_DIRECTORY')
    name, path=sys.argv[1],Path(sys.argv[2]).resolve()
    if name=='_browser': browser(path)
    elif name=='bundle': bundle(path)
    elif name in (*PY_GROUPS,'node','browser'): phase(name,path)
    else: raise SystemExit('Unknown phase')
