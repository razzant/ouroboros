"""#4287 lifecycle witness: exact git archive, disposable roots, no model/provider.
The tested product commit arrives as --candidate (one full 40-hex sha); the QA
workflow derives it from the immutable QA-commit -> product link (git rev-parse HEAD^).
Panic mode is intentionally NOT supported on a workstation. See qa/README.md.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from urllib.parse import urlencode
from urllib.request import build_opener, ProxyHandler, Request
from xml.etree import ElementTree

SELF = Path(__file__).resolve()
QA_REF = 'refs/heads/qa/4287-room-history-reader'
QA_PATHS = {
    b'.github/workflows/qa-room-history-reader.yml', b'qa/README.md',
    b'qa/witness.py', b'qa/windows_reader_close_rename.py',
}
WINDOWS_CASES = (
    'tests/test_room_history_reader.py::test_actual_reader_reuses_bytes_and_reports_fallback',
    'tests/test_room_history_reader.py::test_real_503_is_not_retried_in_server',
    'tests/test_room_history_reader.py::test_ingress_witnesses_are_the_same_in_the_reader_and_server',
    'tests/test_room_history_reader.py::test_each_lease_releases_canonical_and_nested_root_handles',
    'tests/test_room_history_reader.py::test_history_uses_the_direct_venv_interpreter_without_retargeting_other_jobs',
    'qa/windows_reader_close_rename.py::test_reader_normal_close_releases_store_files_for_rename',
)


def emit(door, **fields):
    print(json.dumps(dict(door=door, **fields), ensure_ascii=False), flush=True)


def require_hosted(os_name):
    expected = {
        'GITHUB_ACTIONS': 'true', 'RUNNER_ENVIRONMENT': 'github-hosted',
        'RUNNER_OS': os_name, 'GITHUB_EVENT_NAME': 'push', 'GITHUB_REF': QA_REF,
    }
    if platform.system() != os_name or any(os.environ.get(k) != v for k, v in expected.items()):
        raise RuntimeError(f'QA_RUNNER_UNVERIFIED: requires GitHub-hosted {os_name} push of {QA_REF}')


def qa_product_sha(repo):
    """Check the actual pushed checkout; do not trust a moving branch or a wide path glob."""
    if repo != Path(os.environ['GITHUB_WORKSPACE']).resolve() or SELF.parent.parent != repo:
        raise RuntimeError('QA_CHECKOUT_UNVERIFIED: witness and checkout are not the Actions workspace')

    def git(*argv):
        return subprocess.check_output(['git', '-C', str(repo), *argv])

    head = git('rev-parse', 'HEAD').strip()
    if head.decode('ascii') != os.environ.get('GITHUB_SHA'):
        raise RuntimeError('QA_CHECKOUT_UNVERIFIED: HEAD differs from the pushed commit')
    if git('status', '--porcelain=v1', '-z', '--untracked-files=all'):
        raise RuntimeError('QA_CHECKOUT_UNVERIFIED: checkout is not clean')
    # Read the raw commit header: shallow graph traversal can hide a parent.
    headers = git('cat-file', 'commit', 'HEAD').split(b'\n\n', 1)[0].splitlines()
    parents = [line[7:] for line in headers if line.startswith(b'parent ')]
    if len(parents) != 1 or re.fullmatch(b'[0-9a-f]{40}', parents[0]) is None:
        raise RuntimeError('QA_PRODUCT_PARENT_UNVERIFIED: QA commit needs exactly one product parent')
    product = parents[0].decode('ascii')
    raw = git('diff-tree', '-r', '--no-commit-id', '--no-renames', '--raw', '-z', product, 'HEAD')
    fields = raw.split(b'\x00')
    if len(fields) != 2 * len(QA_PATHS) + 1 or fields[-1] != b'' or set(fields[1:-1:2]) != QA_PATHS:
        raise RuntimeError('QA_PATHS_UNVERIFIED: product-parent diff needs all and only the four QA files')
    for header in fields[:-1:2]:
        parts = header.split()
        if len(parts) != 5 or parts[:2] != [b':000000', b'100644'] or parts[-1] != b'A':
            raise RuntimeError('QA_PATHS_UNVERIFIED: each QA file must be a regular-file addition')
    return product


def prove_qa_head(os_name):
    require_hosted(os_name)
    product = qa_product_sha(Path.cwd().resolve())
    with Path(os.environ['GITHUB_OUTPUT']).open('a', encoding='utf-8') as out:
        out.write(f'product_sha={product}\n')
    emit('QA_HEAD_BOUND', qa_sha=os.environ['GITHUB_SHA'], product_sha=product, ref=QA_REF)


def require_windows_results(report):
    try:
        result = ElementTree.parse(report)
    except (OSError, ElementTree.ParseError) as exc:
        raise RuntimeError('WINDOWS_CASE_EVIDENCE_UNAVAILABLE: no readable JUnit readback') from exc
    expected = {(Path(node).stem, name) for node, name in (case.split('::') for case in WINDOWS_CASES)}
    seen = set()
    for case in result.iter('testcase'):
        ident = (case.get('classname', '').split('.')[-1], case.get('name', ''))
        if ident not in expected or ident in seen:
            raise RuntimeError(f'WINDOWS_CASE_COVERAGE_MISMATCH: duplicate or unexpected {ident}')
        for label in ('skipped', 'failure', 'error'):
            if case.find(label) is not None:
                raise RuntimeError(f'WINDOWS_CASE_NOT_PASSED: {ident} {label}')
        seen.add(ident)
    if seen != expected:
        raise RuntimeError(f'WINDOWS_CASE_COVERAGE_MISMATCH: missing {sorted(expected - seen)}')


def run_windows_cases():
    require_hosted('Windows')
    product = qa_product_sha(Path.cwd().resolve())
    if product != os.environ.get('PRODUCT_SHA'):
        raise RuntimeError('QA_PRODUCT_PARENT_UNVERIFIED: execution parent differs from checked parent')
    shipped = Path(os.environ['OUROBOROS_CI_INTERPRETER']).resolve()
    if (not shipped.is_file() or sys.prefix == sys.base_prefix
            or Path(sys._base_executable).resolve() != shipped):
        raise RuntimeError('WINDOWS_SHIPPED_INTERPRETER_UNVERIFIED: pytest is not in the shipped Python venv')
    with tempfile.TemporaryDirectory(prefix='hist4287-win-', dir=os.environ['RUNNER_TEMP']) as scratch:
        report = Path(scratch) / 'selected.xml'
        result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-rA', '-o', 'addopts=',
                                 '-p', 'no:randomly', '--junitxml', str(report), *WINDOWS_CASES], check=False)
        if result.returncode:
            raise RuntimeError(f'WINDOWS_PYTEST_FAILED_OR_NOT_RUN: exit {result.returncode}')
        require_windows_results(report)
    emit('WINDOWS_SIX_CASES_VERIFIED', product_sha=product, python=sys.version.split()[0],
         sqlite=sqlite3.sqlite_version, cases=list(WINDOWS_CASES))


def wait_for(test, timeout=12, label='observation timed out'):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if out := test():
            return out
        time.sleep(.035)
    raise AssertionError(label)


def print_file(path):
    return json.loads(path.read_text()) if path.exists() else None


def db_check(root, name='dialog'):
    try:
        with contextlib.closing(sqlite3.connect((root / f'state/{name}.sqlite').as_uri() + '?mode=ro', uri=True)) as conn:
            if conn.execute('pragma integrity_check').fetchall() != [('ok',)]:
                raise RuntimeError(f'STORE_INTEGRITY_FAILED: {name}.sqlite')
    except sqlite3.Error as exc:
        raise RuntimeError(f'STORE_READ_UNVERIFIED: {name}.sqlite') from exc
    return 'ok'


def fresh_profile_check(root):
    # Lifespan publishes current and usage even providerless. The first
    # untouched dialogue store can be absent; readback must not create it.
    checks = {}
    for name in ('current', 'usage', 'dialog'):
        path = root / 'state' / f'{name}.sqlite'
        try:
            path.stat()
        except FileNotFoundError:
            if name != 'dialog':
                raise RuntimeError(f'LINUX_REQUIRED_STORE_MISSING: {name}.sqlite') from None
            checks[name] = 'not_created_in_fresh_profile'
        else:
            checks[name] = db_check(root, name)
    return checks


def seed(root):
    from ouroboros import dialogue_store
    from tests._dialogue import confirm_first_use, import_all
    from ouroboros.store_import import ensure_current_store

    confirm_first_use(root)
    import_all(root)
    ensure_current_store(root)
    dialogue_store.append(root, 'chat', {'direction':'in', 'text':'SYNTHETIC_4287', 'chat_id':1,
                                        'ts':'2026-11-25T19:00:00Z'})


def page(reader, root):
    from ouroboros.gateway import history_reader as hr
    from ouroboros.gateway.history import make_chat_history_endpoint
    assert hr.reader is reader

    async def read():
        paths = []
        token = hr.READ_PATH.set(paths.append)
        try:
            response = await make_chat_history_endpoint(root)(SimpleNamespace(query_params={}))
        finally:
            hr.READ_PATH.reset(token)
        return response.status_code, json.loads(response.body), paths

    status, result, paths = asyncio.run(read())
    rows = result.get('messages', [])
    return dict(status=status, path=paths, seed=any(row.get('text')=='SYNTHETIC_4287' for row in rows))


def bind_real(reader, root, repo):
    reader.start(root, repo)
    assert reader.ready.wait(12), reader._failure
    assert reader.pid is not None and reader.pid != os.getpid()
    return reader._process


def settle(reader, process):
    reader._thread.join(8)
    assert not reader.live_children(), reader.live_children()
    assert process.poll() is not None
    assert process.stdin.closed and process.stdout.closed
    return dict(pid=process.pid, wait_status=process.returncode, pipes_closed=True, children=[])


def controller(mode, root, repo):
    from ouroboros.gateway import history_reader as hr
    from ouroboros import startup_historical_audit as sha
    from ouroboros import store, store_switch
    from ouroboros.process_custody import current_custody_session_id

    assert current_custody_session_id()
    seed(root)
    reader = hr.reader
    if mode == 'start-race':
        real_spawn = sha.spawn_maintenance_job
        arrived, release = threading.Event(), threading.Event()
        seen = []
        def delayed(*args, on_spawn, **kwargs):
            def entrance(process):
                seen.append(process)
                arrived.set()
                assert release.wait(8)
                on_spawn(process)
            return real_spawn(*args, on_spawn=entrance, **kwargs)
        sha.spawn_maintenance_job = delayed
        reader.start(root, repo)
        try:
            assert arrived.wait(8)
            assert seen[0].poll() is None
            kids = reader.live_children()
            assert kids == [hr.PURPOSE + ':launch_or_reap'], kids
            reader.stop()  # Only this controller's reader; not global Panic.
            assert not reader.ready.is_set()
        finally:
            release.set()
        emit(mode, before=kids, after=settle(reader, seen[0]), database=db_check(root), ready=False)
        return

    process = bind_real(reader, root, repo)
    try:
        before = page(reader, root)
        assert before == dict(status=200, seed=True, path=['child']), before
        if mode == 'handoff':
            # Execute the actual switch composition and real store hand-off. The
            # tree callback intentionally refuses before effects; it never runs Git.
            events = []
            hand_off = store_switch._hand_off_stores
            def observed_handoff(arg):
                before_handoff = settle(reader, process)
                events.append(dict(at='before_real_store_handoff', **before_handoff))
                return hand_off(arg)
            store_switch._hand_off_stores = observed_handoff
            expected_head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
            class Refused(RuntimeError):
                before_tree_effects = True
            def refuse():
                fenced = page(reader, root)
                assert fenced == dict(status=503, path=['handoff'], seed=False), fenced
                events.append(dict(at='before_any_tree_effect', **fenced))
                raise Refused('fixture refused before touching tree')
            try:
                store_switch.switch_or_reset(root, repo, target_sha=expected_head, target_ref='HEAD',
                                             reason='synthetic lifecycle test', door='ui_rollback', reset=refuse)
            except Refused:
                pass
            else:
                raise AssertionError('typed pre-mutation refusal not reached')
            after = page(reader, root)
            assert after == dict(status=200, seed=True, path=['server']), after
            assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==expected_head
            assert not store._HANDED_OFF, store._HANDED_OFF
            emit(mode, before=before, after=after, events=events, database=db_check(root), head=expected_head)
        elif mode == 'parent-crash':
            (root.parent / 'reader.json').write_text(json.dumps(dict(reader_pid=process.pid, host_pid=os.getpid(), before=before)))
            threading.Event().wait(30)  # external driver kills only this Popen, not its group
            raise AssertionError('parent-crash driver did not act')
        elif mode == 'in-flight':
            # Freeze only this own Popen, after real readiness and one real GET.
            process.send_signal(signal.SIGSTOP)
            async def exercise():
                original, sent = reader._send, threading.Event()
                def observe(*a):
                    original(*a)
                    sent.set()  # successful write+flush of a real frame to the real reader's pipe
                reader._send = observe
                from ouroboros.gateway.history import make_chat_history_endpoint
                paths = []
                token = hr.READ_PATH.set(paths.append)
                try:
                    task = asyncio.create_task(make_chat_history_endpoint(root)(SimpleNamespace(query_params={})))
                    assert await asyncio.to_thread(sent.wait, 5)
                    assert not task.done() and reader._requests and process.poll() is None
                    reader.stop()
                    result = await asyncio.wait_for(task, 5)
                    assert result.status_code == 503 and paths == ['handoff'], (result.status_code, paths)
                    return dict(status=result.status_code,path=paths)
                finally:
                    hr.READ_PATH.reset(token)
            response = asyncio.run(exercise())
            emit(mode, before=before, in_flight=response, after=settle(reader, process), database=db_check(root))
        else:
            call = process.kill if mode == 'reader-crash' else reader.stop
            call()
            exited = settle(reader, process)
            after = page(reader,root)
            expected = dict(status=200, seed=True, path=['server']) if mode=='reader-crash' else dict(status=503, seed=False,path=['handoff'])
            assert after == expected, after
            emit(mode, before=before, after=after, exit=exited, database=db_check(root))
    finally:
        reader.stop()
        settle(reader, process)


def http_get(port, path):
    with build_opener(ProxyHandler({})).open(f'http://127.0.0.1:{port}{path}', timeout=.7) as r:
        return json.load(r)


def live_host(root, repo):
    # Reuse the providerless real server (Starlette/uvicorn), preventing optional
    # background service provision only. The serving reader is created per server
    # generation (history_reader.begin_generation at lifespan startup), so the marker
    # wraps each generation's original read at its creation, not a mock and not a
    # pre-boot module singleton that no request would reach.
    from ouroboros import server_boot, server_entrypoint
    from ouroboros.gateway import history_reader as hr
    server_boot.start_background_services = lambda *_a, **_kw: None
    bind = server_entrypoint.bound_service_socket
    @contextlib.contextmanager
    def report_bind(root, kind, *args, **kwargs):
        with bind(root, kind, *args, **kwargs) as listener:
            if kind == 'main':
                emit('main_socket', port=listener.getsockname()[1])
            yield listener
    server_entrypoint.bound_service_socket = report_bind
    begin = hr.begin_generation
    def observed_begin(generation_root):
        generation = begin(generation_root)
        read = generation.read
        async def observed(*args, **kwargs):
            paths=[]
            token = hr.READ_PATH.set(paths.append)
            try:
                result = await read(*args, **kwargs)
            finally:
                hr.READ_PATH.reset(token)
            emit('http_history', path=paths, reader_pid=generation.pid, status=result[0])
            return result
        generation.read = observed
        return generation
    hr.begin_generation = observed_begin
    import server
    sys.argv = ['server.py','--host','127.0.0.1','--port','0']
    code = server.main()
    process = hr.reader._process
    assert process is not None
    emit('lifespan_exit', **settle(hr.reader,process), host_pid=os.getpid(), code=code)
    return code


def child_status(pid):
    # Single exact recorded reader PID. `ps` is observation, not a signal.
    assert type(pid) is int and pid > 1
    completed = subprocess.run(['ps','-o','stat=','-p',str(pid)], capture_output=True,text=True)
    if completed.returncode == 1 and not completed.stdout.strip() and not completed.stderr:
        return 'absent'
    assert completed.returncode == 0 and completed.stdout.strip(), completed.stderr
    return completed.stdout.strip()


def child_active(pid):
    status=child_status(pid)
    return status != 'absent' and not status.startswith('Z')


def driver(src, work, candidate, hosted_panic=False):
    assert sys.platform in ('darwin','linux'), 'POSIX witness only'
    if hosted_panic:
        require_hosted('Linux')
        if candidate != qa_product_sha(src) or work.parent != Path(os.environ['RUNNER_TEMP']).resolve():
            raise RuntimeError('QA_PRODUCT_OR_SCRATCH_UNVERIFIED: need this product parent and fresh runner scratch')
    if re.fullmatch('[0-9a-f]{40}', candidate or '') is None:
        raise RuntimeError('candidate must be one full 40-hex product commit id')
    work.mkdir()  # Fail if an existing location was supplied; nothing is used in place.
    repo=work/'repo'; repo.mkdir()
    subprocess.run(['git','-C',str(src),'cat-file','-e',candidate+'^{commit}'],check=True)  # the checkout must already hold the exact product commit
    archive=work/'source.tar'
    with archive.open('xb') as out:
        subprocess.run(['git','-C',str(src),'archive',candidate],stdout=out,check=True)
    subprocess.run(['tar','-xf',str(archive),'-C',str(repo)],check=True)
    # Supply repository identity in the disposable archive. On the shallow
    # hosted checkout fetch its advertised local HEAD plus the checked parent;
    # an upload-pack need not accept a raw SHA for an unadvertised parent.
    git=['git','-C',str(repo)]
    subprocess.run([*git,'init','-q'],check=True)
    depth, wanted = ('2', 'HEAD') if hosted_panic else ('1', candidate)
    subprocess.run([*git,'-c','protocol.file.allow=always','fetch','-q','--no-tags',f'--depth={depth}',
                    '--update-shallow',str(src),wanted],check=True)
    subprocess.run([*git,'update-ref','--no-deref','HEAD',candidate],check=True)
    sys.path.insert(0,str(repo))
    from ouroboros.test_environment import isolated_environment
    lineage = dict(candidate=candidate, platform=platform.platform(), python=sys.version.split()[0],sqlite=sqlite3.sqlite_version)
    emit('provenance', **lineage)
    modes=('live-host',) if hosted_panic else ('idle','reader-crash','start-race','in-flight','handoff','parent-crash','live-host')
    for mode in modes:
        area=work/mode
        env=isolated_environment(area,repo)
        env.update(PYTHONPATH=str(repo),OUROBOROS_SERVER_PORT='0',OUROBOROS_HOST_SERVICE_PORT='0',
                   OUROBOROS_LAUNCHER_STOP_STDIN='1')
        root=area/'data'
        (root/'settings.json').write_text(json.dumps({'OUROBOROS_STARTER_SKILLS':'[]','OUROBOROS_SUBAGENTS':'[]',
                                                     'OUROBOROS_MAX_WORKERS':1}))
        logpath=area/'probe.log'
        with logpath.open('wb') as log:
            proc = subprocess.Popen([sys.executable,'-B','-u',str(SELF),'--child',mode,str(root),str(repo)],
                                    env=env,cwd=repo,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,
                                    start_new_session=True)
            try:
                if mode=='parent-crash':
                    state=wait_for(lambda: print_file(area/'reader.json'))
                    assert state['host_pid']==proc.pid and child_active(state['reader_pid'])
                    proc.kill()
                    assert proc.wait(timeout=5)==-signal.SIGKILL
                    wait_for(lambda: not child_active(state['reader_pid']),timeout=15)
                    emit(mode, **state, parent_wait_status=proc.returncode, reader_running_after=False,
                         reader_status_after=child_status(state['reader_pid']), database=db_check(root))
                elif mode=='live-host':
                    def socket_port():
                        for line in logpath.read_text().splitlines():
                            if line.startswith('{"door": "main_socket"'):
                                return json.loads(line)['port']
                        return False
                    port=wait_for(socket_port,timeout=20,label='LINUX_SERVER_BIND_UNVERIFIED')
                    def health():
                        try: r=http_get(port,'/api/health')
                        except OSError: return False
                        return r if r.get('boot',{}).get('tasks_ready') else False
                    ready=wait_for(health,timeout=20,label='LINUX_SERVER_NOT_READY')
                    wait_for(lambda: 'room_history_reader ready pid=' in logpath.read_text(),timeout=12,
                             label='LINUX_CHILD_NOT_READY')
                    page0=http_get(port,'/api/chat/history?'+urlencode({'chat_id':1}))
                    assert isinstance(page0.get('messages'), list), 'LINUX_HISTORY_BODY_UNVERIFIED'
                    http_line=next((json.loads(x) for x in logpath.read_text().splitlines()
                                    if x.startswith('{"door": "http_history"')), {})
                    assert (http_line.get('path'), http_line.get('status'))==(['child'],200), 'LINUX_CHILD_GET_UNVERIFIED'
                    pid=http_line['reader_pid']
                    assert pid!=proc.pid and child_active(pid), 'LINUX_LIVE_CHILD_PID_UNVERIFIED'
                    if hosted_panic:
                        assert not (root/'state/panic_stop.flag').exists() and proc.poll() is None, 'LINUX_PRE_PANIC_STATE_UNVERIFIED'
                        assert port > 1024 and port not in (8765,8766), 'LINUX_BOUND_EPHEMERAL_PORT_UNVERIFIED'
                        outgoing=Request(f'http://127.0.0.1:{port}/api/command',
                                         data=b'{"cmd":"/panic"}',method='POST',
                                         headers={'Content-Type':'application/json'})
                        with build_opener(ProxyHandler({})).open(outgoing,timeout=8) as response:
                            ack=dict(status=response.status,body=json.load(response))
                        assert ack==dict(status=200,body={'status':'ok'}), ('LINUX_PANIC_ACK_UNVERIFIED', ack)
                        assert proc.wait(timeout=15)==99, ('LINUX_PANIC_SERVER_EXIT_UNVERIFIED', proc.returncode)
                        assert (root/'state/panic_stop.flag').is_file(), 'LINUX_PANIC_FLAG_MISSING'
                        wait_for(lambda: not child_active(pid),timeout=8,label='LINUX_CHILD_EXIT_UNCONFIRMED')
                        emit('guest-api-panic',host_pid=proc.pid, reader_pid=pid, reader_status=child_status(pid),
                             server_wait_status=proc.returncode, ack=ack, flag=True,
                             history=http_line,database=fresh_profile_check(root))
                    else:
                        proc.stdin.write(b'quit\n'); proc.stdin.flush(); proc.stdin.close()
                        assert proc.wait(timeout=14)==0
                        observed=next(json.loads(x) for x in logpath.read_text().splitlines() if x.startswith('{"door": "lifespan_exit"'))
                        assert observed['pid']==pid and not child_active(pid)
                        databases=fresh_profile_check(root)
                        assert databases['current']==databases['usage']=='ok', databases
                        emit('providerless-host-quit', host_pid=proc.pid, port=port, boot=ready['boot'], history=http_line,
                             exit=observed, database=databases, server_wait_status=proc.returncode)
                else:
                    assert proc.wait(timeout=30)==0, logpath.read_text()
                    text=logpath.read_text()
                    for line in text.splitlines():
                        if line.startswith('{"door":') or 'room_history_reader_fallback' in line: print(line,flush=True)
            finally:
                # Never sweep a machine, process name, PID list, or another process group.
                if proc.poll() is None:
                    proc.kill(); proc.wait(timeout=5)
                if proc.stdin and not proc.stdin.closed: proc.stdin.close()
    emit('LIFECYCLE_GUEST_PANIC_VERIFIED' if hosted_panic else 'LIFECYCLE_LOCAL_VERIFIED')


if __name__=='__main__':
    if not __debug__:
        raise RuntimeError('QA_ASSERTIONS_DISABLED: do not use PYTHONOPTIMIZE or -O')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if sys.argv[1:2] == ['--prove-qa-head'] and len(sys.argv) == 3:
        prove_qa_head(sys.argv[2])
    elif sys.argv[1:] == ['--windows-cases']:
        run_windows_cases()
    elif sys.argv[1:2]==['--child']:
        mode, root, repo=sys.argv[2:]; root=Path(root); repo=Path(repo)
        sys.path.insert(0,str(repo))
        if mode=='live-host': raise SystemExit(live_host(root,repo))
        controller(mode,root,repo)
    else:
        args=argparse.ArgumentParser()
        args.add_argument('source_repo',type=Path)
        args.add_argument('fresh_scratch',type=Path)
        args.add_argument('--candidate',required=True,
                          help='full 40-hex product commit to archive and test')
        args.add_argument('--hosted-panic-only', action='store_true')
        ns=args.parse_args()
        driver(ns.source_repo.resolve(),ns.fresh_scratch.resolve(),ns.candidate,ns.hosted_panic_only)
