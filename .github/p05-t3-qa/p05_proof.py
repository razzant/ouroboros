"""Execute the pinned matrix or package its closed, synthetic proof set after teardown.

The pack step runs as the same unprivileged UID with the work tree read-only. Only
normalized fields and named screenshots go in the ZIP; never raw pytest/TAP/JUnit.
"""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import re
import socket
import stat
import struct
import subprocess
import sys
import zipfile
import zlib

MIB = 1 << 20
PY_FLAGS = ['-ra', '--tb=short', '-o', 'addopts=', '-p', 'no:cacheprovider']
ENGINES = {'chromium': (1280, 920), 'webkit': (390, 844)}
CHOICES = ('keep', 'add')
TRANSITIONS = {'old-same-wakes': 'before-connector', 'typed-then-old-same-wakes': 'after-connector',
               'no-old-writer': 'none'}
READINESS = {f'first-dm-{order}-{kind}': (order, kind)
             for order in ('core-first', 'connector-first') for kind in ('plain', 'inbox-signal')}
PAIRED_FIRST_DM = 'first-dm-core-first-plain'
BASE_LIVE = {
    'test_p05_upgrade.py': (f'test_first_dm_across_publication[{PAIRED_FIRST_DM}]', 'test_public_gateway_dns_verdict'),
    'test_p05_browser.py': ('test_telegram_form_and_accounts_key[chromium]', 'test_telegram_form_and_accounts_key[webkit]'),
}
LIVE = {'test_p05_upgrade.py': len(TRANSITIONS) + len(READINESS) + 1, 'test_p05_browser.py': 6}
OUTCOMES = ('message', 'silent', 'tool_delivered', 'deferred')
HISTORY_COUNTS = {'test_reseed_from_a_real_previous_seed': 3,
                  'test_call_skill_release_reseeds_the_old_payload_but_preserves_owner_edits': 14}
# Local-transport proofs that a new public address is verified before the Telegram menu moves, and that only a
# name-error answer for the exact host, not a temporary failure, counts as absence. Floors, so added cases pass.
NEW_MENU_FLOORS = {'test_public_dns_recovery_updates_the_menu_only_after_verification': 4,
                   'test_fresh_tunnel_can_verify_during_registration': 2,
                   'test_delayed_menu_install_needs_a_fresh_marker_without_resetting_grace': 2}
OLD_MENU_COUNTS = {'test_a_cold_start_restores_an_older_releases_owner_before_exposing_the_new_one': 2,
                   'test_a_chat_only_record_without_this_bots_button_restores_nothing': 4,
                   'test_a_restore_interrupted_after_the_derivation_resumes_from_its_ledger': 1}


def read_regular(root, rel, limit=MIB):
    """Read by descriptor, anchoring every directory and final open without following links."""
    if not root.is_absolute() or rel.startswith('/'):
        raise ValueError('unexpected proof path')
    parts = (*root.parts[1:], *rel.split('/'))
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError('unexpected proof component')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    parent = os.open('/', flags)
    try:
        for component in parts[:-1]:
            child = os.open(component, flags, dir_fd=parent)
            os.close(parent)
            parent = child
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=parent)
        with os.fdopen(fd, 'rb') as stream:
            seen = os.fstat(stream.fileno())
            if not stat.S_ISREG(seen.st_mode) or seen.st_nlink != 1 or not 0 < seen.st_size <= limit:
                raise ValueError('expected one bounded regular file')
            data = stream.read(limit + 1)
            if len(data) != seen.st_size or os.fstat(stream.fileno()).st_mtime_ns != seen.st_mtime_ns:
                raise ValueError('proof changed during read')
            return data
    finally:
        os.close(parent)


def object_at(root, rel, limit=MIB):
    value = json.loads(read_regular(root, rel, limit))
    if not isinstance(value, dict):
        raise ValueError('expected a proof object')
    return value


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode('utf-8')


def put(path, value):
    with path.open('xb') as stream:
        stream.write(json_bytes(value))


def boundary(root, collecting):
    state = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
    mounts = Path('/proc/self/mountinfo').read_text().splitlines()
    routes = Path('/proc/net/route').read_text().splitlines()[1:]
    outer = os.environ['P05_OUTER_PID_NS']
    assert re.fullmatch(r'pid:\[\d+\]', outer) and os.readlink('/proc/self/ns/pid') != outer
    assert os.geteuid() == os.getuid() != 0
    assert set(os.getgroups()) <= {os.getgid()}
    assert state['NoNewPrivs'].strip() == '1' and int(state['CapEff'], 16) == 0
    assert [name for _, name in socket.if_nameindex()] == ['lo']
    assert not any(row.split()[1] == '00000000' for row in routes)
    assert any(row.split()[4] == '/run' and ' - tmpfs ' in row for row in mounts)
    assert not any(Path(p).exists() for p in ('/run/docker.sock', '/run/systemd/private', '/run/dbus/system_bus_socket'))
    assert not any(k.startswith(('GITHUB_', 'ACTIONS_', 'SSH_', 'GH_')) or k.endswith(('_API_KEY', '_TOKEN'))
                   for k in os.environ)
    assert os.statvfs(root).f_flag & os.ST_RDONLY
    assert bool(os.statvfs(root / 'work').f_flag & os.ST_RDONLY) == collecting
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        with socket.create_connection(listener.getsockname(), timeout=1):
            peer, _ = listener.accept()
            peer.close()
    return {'loopback': True, 'private_run': True, 'private_pids': True, 'non_root': True,
            'no_new_privileges': True, 'effective_caps': 0, 'subject_readonly': True}


def groups(subject):
    home = subject['test_matrix']
    result = {f'candidate-{name}': ('candidate', name, home[name], limit) for name, limit in (
        ('presence', 1250), ('queue', 620), ('migration', 450), ('telegram', 630), ('history', 250), ('structural', 480), ('size', 240))}
    result['base-controls'] = ('base', 'pytest', home['base_controls'], 240)
    result['base-live'] = ('base', 'live', list(BASE_LIVE), 420)
    for file in home['node']:
        key = 'node-' + Path(file).name.removesuffix('.test.js')
        assert key not in result
        result[key] = ('candidate', 'node', [file], 110)
    result['candidate-live'] = ('candidate', 'live', list(LIVE), 650)
    return result


def exact_sources(root, subject):
    for role, pins in subject['subjects'].items():
        git = ['/usr/bin/git', '--no-optional-locks', '-C', str(root / role)]
        for ref, key in [('HEAD', 'commit'), ('HEAD^{tree}', 'tree')]:
            assert subprocess.check_output([*git, 'rev-parse', ref], text=True).strip() == pins[key]
        subprocess.run([*git, 'diff', '--quiet', 'HEAD', '--'], check=True)
        assert subprocess.check_output([*git, 'ls-files', '--others', '-z']) == b''


def command(root, subject, role, kind, files, raw):
    py = str(root / 'venvs' / role / 'bin' / 'python')
    if kind == 'node':
        return [str(root / 'node-standalone/bin/node'), '--test', '--test-reporter=tap', *files]
    args = [py, '-m', 'pytest', *PY_FLAGS, '--timeout=130']
    if kind == 'size':
        # safe_test scrubs upstream bytes and GITHUB_EVENT inputs; set the exact comparison base only inside pytest.
        base = subject['subjects']['base']['commit']
        args = [py, '-c', 'import os,runpy,sys; os.environ["OURO_SIZE_RATCHET_BASE_REF"]=sys.argv.pop(1); '
                'runpy.run_module("pytest", run_name="__main__")', base, *PY_FLAGS, '--timeout=130', '-m', 'size_ratchet']
    if kind == 'live':
        args += ['--timeout=330', '-p', 'tests.conftest', '-c', str(root / role / 'pyproject.toml')]
        if role == 'base':
            files = [f'{root / "harness" / file}::{case}' for file in files for case in BASE_LIVE[file]]
        else:
            files = [str(root / 'harness' / file) for file in files]
    return [*args, *files, '--ci-evidence-dir', str(raw), '--junitxml', str(raw / 'private-junit.xml')]


def run(root):
    proof = boundary(root, collecting=False)  # before the first product import
    work = root / 'work'
    subject = object_at(root, 'subject.json')
    exact_sources(root, subject)
    (work / 'proof/screens').mkdir()
    put(work / 'proof/boundary.json', proof)
    success = True
    for label, (role, kind, files, timeout) in groups(subject).items():
        raw = work / 'raw' / label
        raw.mkdir()
        env = os.environ.copy()
        if kind == 'live':
            env.update(OUROBOROS_RUN_UI_SMOKE='1', OUROBOROS_EXPECT_BROWSER_ENGINES='chromium,webkit',
                       OUROBOROS_UI_EVIDENCE_DIR=str(work / 'proof'))
        launcher = [str(root / 'venvs' / role / 'bin/python'), '-I', '-S', '-B', 'scripts/safe_test.py',
                    '--temp-parent', '/tmp', '--evidence-dir', str(raw / 'safe'), '--timeout', str(timeout), '--']
        with (raw / 'stdio.out').open('xb') as out, (raw / 'stdio.err').open('xb') as err:
            rc = subprocess.run([*launcher, *command(root, subject, role, kind, files, raw)], cwd=root / role,
                                env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, check=False).returncode
        attempts = list((raw / 'safe').iterdir()) if (raw / 'safe').is_dir() else []
        settled = False
        if len(attempts) == 1:
            try:
                receipt = object_at(root, f'work/raw/{label}/safe/{attempts[0].name}/result.json')
                settled = (receipt['execution'] == 'executed' and receipt['containment_error'] == ''
                           and receipt['evidence_error'] == '' and isinstance(receipt['cleanup'], dict)
                           and not receipt['source_gap'] and receipt['returncode'] == rc)
            except (KeyError, ValueError, OSError):
                pass
        put(raw / 'phase.json', {'exit': rc, 'settled': settled})
        if not settled:
            break  # no next group alongside work whose custody is unknown
        success &= rc == 0
    else:
        exact_sources(root, subject)
        return 0 if success else 1
    return 1


def node_counts(data):
    totals = re.findall(rb'^# (tests|pass|fail|skipped|cancelled|todo) (\d+)$', data, re.M)
    counts = {key.decode('ascii'): int(value) for key, value in totals}
    good = (len(totals) == 6 and set(counts) == {'tests', 'pass', 'fail', 'skipped', 'cancelled', 'todo'}
            and counts['tests'] > 0 and counts['pass'] == counts['tests']
            and not any(counts[key] for key in ('fail', 'skipped', 'cancelled', 'todo')) and b'Bail out!' not in data)
    return {'counts': counts, 'complete': bool(good)}


def pytest_counts(record, files, kind, root, role):
    selected = record['selected_nodeids']
    if not isinstance(selected, list) or not selected or not all(isinstance(node, str) for node in selected):
        raise ValueError('missing pytest selection')
    expected = set(selected)
    hits, names = Counter(), Counter()
    for node in selected:
        path, *identity = node.split('[', 1)[0].split('::')
        if kind == 'live':
            path = Path(path).name
        if path not in files or not identity:
            raise ValueError('wrong tests collected')
        hits[path] += 1
        names[identity[-1]] += 1
    reports = {node: Counter() for node in selected}
    nonpass = skipped = 0
    cases = {node: i for i, node in enumerate(selected)}
    known_names, faults = {}, []
    for row in record['reports']:
        node, phase, outcome = row['nodeid'], row['phase'], row['outcome']
        if node not in expected:
            raise ValueError('unselected test report')
        reports[node][phase] += not row['is_subtest']
        nonpass += outcome != 'passed' or 'wasxfail' in row
        skipped += outcome == 'skipped'
        if outcome != 'passed' or 'wasxfail' in row:
            file, *parts = node.split('[', 1)[0].split('::')
            if kind == 'live':
                file = Path(file).name
            if file not in known_names:
                public = read_regular(root, ('harness' if kind == 'live' else role) + '/' + file, 4 * MIB)
                known_names[file] = {f.name for f in ast.walk(ast.parse(public))
                                     if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
            if phase not in ('setup', 'call', 'teardown') or outcome not in ('passed', 'failed', 'skipped'):
                raise ValueError('unexpected pytest phase or outcome')
            # Only a public filename and function, plus an ordinal: no parameter ID, traceback or raw response data.
            faults.append({'file': file, 'function': parts[-1] if parts[-1] in known_names[file] else None,
                           'index': cases[node], 'phase': phase, 'outcome': outcome})
    expected_phases = {'setup': 1, 'call': 1, 'teardown': 1}
    fully_run = sum(dict(phases) == expected_phases for phases in reports.values())
    complete = (record['format'] == 1 and record['session_exit_code'] == 0
                and record['tests_collected'] == len(selected) == len(expected) == fully_run
                and set(hits) == set(files) and not nonpass and not record['collection_failures']
                and not record['collection_skips'] and not record.get('worker_crash'))
    if kind == 'history':
        complete &= all(names[name] == count for name, count in HISTORY_COUNTS.items())
    if kind == 'telegram':
        complete &= all(names[name] == count for name, count in OLD_MENU_COUNTS.items())
        complete &= all(names[name] >= count for name, count in NEW_MENU_FLOORS.items())
    if kind == 'size':
        complete &= len(selected) == 7
    if kind == 'live':
        if role == 'base':
            expected_cases = {f'{file}::{case}' for file, cases in BASE_LIVE.items() for case in cases}
            complete &= {Path(node.split('::', 1)[0]).name + '::' + node.split('::', 1)[1]
                         for node in selected} == expected_cases
        else:
            complete &= (dict(hits) == LIVE and names['test_first_dm_across_publication'] == len(READINESS)
                         and names['test_public_gateway_dns_verdict'] == 1)
    return {'selected': len(selected), 'fully_run': fully_run, 'nonpass_reports': nonpass,
            'skipped_reports': skipped, 'modules': dict(hits), 'nonpass': faults, 'complete': bool(complete)}


def wanted_live():
    for engine, dimensions in ENGINES.items():
        yield (f'telegram-{engine}', ('skill-form', 'accounts-empty-editor'), dimensions)
        yield (f'base-telegram-{engine}', ('skill-form', 'accounts-empty-editor'), dimensions)
        for choice in CHOICES:
            yield (f'place-{engine}-{choice}', ('wakes-before', 'wakes-reopened'), dimensions)
    for name in (*TRANSITIONS, *READINESS, 'base-' + PAIRED_FIRST_DM, 'base-public-dns', 'candidate-public-dns'):
        yield (name, (), None)


def public_fact(name, incoming):
    # Do not publish arbitrary string values from product responses or from browser consoles.
    if name == 'base-' + PAIRED_FIRST_DM:
        result = {'order': 'core-first', 'event_kind': 'plain', 'surface': 'host-asgi-inline',
                  'unpublished': {'http': 200, 'status': 'completed', 'outcome': 'silent', 'arrivals': 1,
                                  'focus_starts': 0, 'task_absent': True, 'saved_wakes': ['legacy_addressed']},
                  'replay': {'http': 200, 'same_turn': True, 'outcome': 'silent', 'arrivals': 1,
                             'focus_starts': 0, 'task_absent': True, 'places_unchanged': True}}
    elif name in ('base-public-dns', 'candidate-public-dns'):
        result = {'nxdomain': 'host_absent' if name == 'candidate-public-dns' else 'observer_unavailable',
                  'servfail': 'observer_unavailable', 'foreign_question': 'observer_unavailable',
                  'nxdomain_with_answers': 'observer_unavailable'}
    elif name in READINESS:
        order, kind = READINESS[name]
        outcome = incoming['accepted']['outcome']
        if outcome not in ('message', 'silent'):
            raise ValueError('unexpected completed focus outcome')
        matched = ['direct_message'] + (['signal.priority_inbox'] if kind == 'inbox-signal' else [])
        result = {'order': order, 'event_kind': kind, 'surface': 'host-asgi-inline',
                  'unpublished': {'http': 429, 'ok': False, 'code': 'wake_import_declarations_pending',
                                  'disposition': 'retry', 'retry_after': 60, 'focus_starts': 0,
                                  'arrivals': 0 if kind == 'inbox-signal' else 1,
                                  'import': 'pending', 'task_absent': True, 'wakes_unchanged': True},
                  'accepted': {'http': 200, 'status': 'completed', 'outcome': outcome, 'turn_ref_exact': True,
                               'task': 'completed', 'arrivals': 1, 'focus_starts': 1, 'matched_wakes': matched,
                               'import': 'completed', 'saved_wakes': ['legacy_addressed', 'direct_message',
                                                                     'signal.priority_inbox']},
                  'replay': {'http': 200, 'same_turn': True, 'arrivals': 1, 'focus_starts': 1,
                             'places_unchanged': True}}
    elif name.startswith(('telegram-', 'base-telegram-')):
        result = {'settings_get': 200, 'saved_miniapp': 'off', 'save_enabled': True,
                  'accounts_editor_empty': True, 'settings_writes': 0,
                  'setup_copy': 'secrets' if name.startswith('base-') else 'accounts_and_keys'}
    elif name.startswith('place-'):
        choice = name.rsplit('-', 1)[1]
        result = {'old_same_list_http_save': True, 'choice': choice, 'reopened': True,
                  'removed_saved': 1 if choice == 'keep' else 2, 'cleared': True, 'declared_additions_only': True}
    else:
        old = TRANSITIONS[name] != 'none'
        status, outcome = incoming['status'], incoming['outcome']
        if status not in ('completed', 'continuing') or outcome not in OUTCOMES or (old and (status, outcome) != ('completed', 'silent')):
            raise ValueError('unexpected HTTP envelope')
        result = {'http': 200, 'ok': True, 'status': status, 'outcome': outcome, 'turn_ref_exact': True,
                  'arrivals': 1, 'task': 'absent' if old else 'completed', 'matched_wakes': [] if old else ['direct_message'],
                  'old_owner_save': old, 'old_save_time': TRANSITIONS[name], 'ordinary_restart': not old,
                  'passive_place_reads': True}
    if json_bytes(incoming) != json_bytes(result):
        raise ValueError('missing or unexpected synthetic witness fields')
    return result


def clean_png(data, dimensions):
    """Keep only pixel chunks; reject unexpected dimensions, broken structure or trailing bytes."""
    signature = b'\x89PNG\r\n\x1a\n'
    if not data.startswith(signature):
        raise ValueError('not PNG')
    pos, result, seen = len(signature), [signature], []
    while pos + 12 <= len(data):
        size, kind = struct.unpack_from('!I4s', data, pos)
        end = pos + 12 + size
        if end > len(data) or zlib.crc32(data[pos + 4:end - 4]) != struct.unpack_from('!I', data, end - 4)[0]:
            raise ValueError('PNG chunk damaged')
        if not seen:
            if kind != b'IHDR' or size != 13 or struct.unpack_from('!II', data, pos + 8) != dimensions:
                raise ValueError('unexpected witness geometry')
        if kind in (b'IHDR', b'PLTE', b'tRNS', b'IDAT', b'IEND'):
            result.append(data[pos:end])
        elif not kind[:1].islower():
            raise ValueError('unknown critical PNG chunk')
        seen.append(kind)
        pos = end
        if kind == b'IEND':
            if size == 0 and pos == len(data) and seen.count(b'IHDR') == 1 and b'IDAT' in seen:
                return b''.join(result)
            break
    raise ValueError('incomplete screenshot')


def pack(root, run_rc):
    box = boundary(root, collecting=True)
    subject = object_at(root, 'subject.json')  # created by staging, outside producer's writable subtree
    if object_at(root, 'work/proof/boundary.json') != box:
        return 2
    ready, missing = {}, []
    for label, (role, kind, files, _) in groups(subject).items():
        prefix = f'work/raw/{label}'
        try:
            phase = object_at(root, prefix + '/phase.json')
            if type(phase['exit']) is not int or not -255 <= phase['exit'] <= 255 or type(phase['settled']) is not bool:
                raise ValueError('bad phase record')
            try:
                result = (node_counts(read_regular(root, prefix + '/stdio.out', 3 * MIB)) if kind == 'node' else
                          pytest_counts(object_at(root, prefix + '/results.json', 8 * MIB), files, kind, root, role))
            except (KeyError, TypeError, ValueError, OSError):
                result = {'complete': False, 'projection': 'unavailable'}
            result.update(exit=phase['exit'], settled=phase['settled'], subject=role)
            result['complete'] &= phase['exit'] == 0 and phase['settled']
            ready[f'groups/{label}.json'] = json_bytes(result)
            if not result['complete']:
                missing.append(label + ': not qualified')
        except (KeyError, TypeError, ValueError, OSError):
            missing.append(label + ': missing phase result')
    for name, scenes, dimensions in wanted_live():
        rel = f'work/proof/{name}.json'
        try:
            ready[f'live/{name}.json'] = json_bytes(public_fact(name, object_at(root, rel, 16_384)))
        except (KeyError, TypeError, ValueError, OSError):
            missing.append(f'live/{name}.json')
        for scene in scenes:
            filename = f'screens/{name}-{scene}.png'
            try:
                ready[filename] = clean_png(read_regular(root, 'work/proof/' + filename, 4 * MIB), dimensions)
            except (TypeError, ValueError, OSError, struct.error):
                missing.append(filename)
    if not ready:
        return 2  # A subject, summary or boundary alone is not a result artifact.
    complete = run_rc == 0 and not missing
    summary = {'format': 1, 'complete': complete, 'unit_exit': run_rc, 'boundary': box,
               'missing_or_not_qualified': missing, 'proof_members': sorted(ready)}
    ready.update({'subject.json': json_bytes(subject), 'summary.json': json_bytes(summary)})
    archive = sys.stdout.buffer  # Actions traps stdout in a new runner-private regular file, not a job log.
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for filename, data in sorted(ready.items()):
            bundle.writestr(filename, data)
    return 0  # A valid partial archive is publishable, not a successful test run.


def check_archive(root, filename, rc):
    """The runner reads back only the private stream file, never a work-tree path."""
    subject = object_at(root, 'subject.json')
    allowed = {f'groups/{label}.json' for label in groups(subject)}
    for name, scenes, _ in wanted_live():
        allowed.add(f'live/{name}.json')
        allowed.update(f'screens/{name}-{scene}.png' for scene in scenes)
    with zipfile.ZipFile(filename) as bundle:
        names = bundle.namelist()
        assert len(names) == len(set(names)) and {'subject.json', 'summary.json'} <= set(names)
        members = set(names) - {'subject.json', 'summary.json'}
        assert members and members <= allowed
        assert all(0 < info.file_size <= 4 * MIB for info in bundle.infolist())
        assert bundle.testzip() is None
        summary = json.loads(bundle.read('summary.json'))
        assert bundle.read('subject.json') == json_bytes(subject)
        assert summary['unit_exit'] == rc and members == set(summary['proof_members'])
        for name, _scenes, _dimensions in wanted_live():
            member = f'live/{name}.json'
            if member in members:
                fact = bundle.read(member)
                assert len(fact) <= 16_384 and fact == json_bytes(public_fact(name, json.loads(fact)))
        if summary['complete']:
            assert rc == 0 and members == allowed and not summary['missing_or_not_qualified']
        print(f"Checked proof archive: {len(members)} proof members, complete={summary['complete']}")
    return 0 if summary['complete'] else 3  # distinct from malformed archive / generic exception


def main():
    mode, directory, *args = sys.argv[1:]
    assert re.fullmatch(r'/var/lib/p05-[0-9]+-[0-9]+', directory)
    root = Path(directory)
    if mode == 'run' and not args:
        return run(root)
    if mode == 'pack' and len(args) == 1:
        return pack(root, int(args[0]))
    if mode == 'check-archive' and len(args) == 2:
        return check_archive(root, args[0], int(args[1]))
    return 2


if __name__ == '__main__':
    sys.exit(main())
