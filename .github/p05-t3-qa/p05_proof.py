"""Execute the pinned matrix or package its closed, synthetic proof set after teardown.

The pack step runs as the same unprivileged UID with the work tree read-only. Only
normalized fields and named screenshots go in the ZIP; never raw pytest/TAP/JUnit.

Diagnostics contract (package 05, second focused round). Every group and every missing
witness now says WHAT the producer left behind, instead of one opaque "unavailable":

* The producer (tests/ci_evidence.py in the pinned subject) writes results.json only in
  pytest_sessionfinish, and fills selected_nodeids only in pytest_collection_modifyitems.
  A session killed before its end therefore leaves NO record, and a session that failed
  during collection leaves a record WITHOUT a selection. Both were previously collapsed
  into projection="unavailable"; they are now record_missing and no_selection.
* A group that ran but did not finish every selected test is "truncated"; a group whose
  tests failed in setup, call or teardown is red_setup / red_call / red_teardown. The
  default stays strict: only exit 0, a settled custody receipt, every selected test run
  and no non-passing report make a group complete.
* A red node carries only bounded public facts: the public test file, the public test
  function, its ordinal in the selection, the pytest phase, the outcome, a closed
  exception bucket and the definition line of that function in the public file. No
  assertion text, traceback, stdout, path, payload or credential ever leaves the runner.
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
import time
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
LIVE = {'test_p05_upgrade.py': len(TRANSITIONS) + len(READINESS) + 1, 'test_p05_browser.py': 8}
# The live matrix runs in two parts per role: a killed browser session no longer hides the
# upgrade session's own record, and a browser failure no longer hides the upgrade result.
LIVE_PARTS = (('upgrade', 'test_p05_upgrade.py'), ('browser', 'test_p05_browser.py'))
# Paired controls: the same public tests, on the base and on the candidate, under the SAME
# confinement. They say whether a red belongs to the confinement or to the candidate change.
# The names are public test names; the projection adds only the validated source line and
# the typed outcome, never a call, a payload or raw output.
PROCESS_CONTROL_FILES = ('tests/test_preflight_process_containment.py', 'tests/test_process_custody.py')
PROCESS_CONTROL_TESTS = ('test_a_detached_child_is_still_found_after_its_root_exits',
                          'test_process_container_kills_a_descendant_that_left_the_group',
                          'test_update_quiesce_kills_service_recorded_by_worker_process',
                          'test_lifeline_kills_child_when_parent_dies')
RESPONSIVENESS_CONTROL_FILES = ('tests/test_host_service_responsiveness.py',)
RESPONSIVENESS_CONTROL_TESTS = ('test_blocking_checks_run_off_the_event_loop',)
PAIRED_CONTROLS = (
    ('process', PROCESS_CONTROL_FILES, PROCESS_CONTROL_TESTS, 360),
)
# The base and candidate run their respective responsiveness tests, not a same-source pair.
DIAGNOSTIC_OBSERVATIONS = (
    ('responsiveness', RESPONSIVENESS_CONTROL_FILES, RESPONSIVENESS_CONTROL_TESTS, 180),
)
SAME_SOURCE_PAIR = 'same_source_pair'
NON_EXACT_DIAGNOSTIC = 'non_exact_diagnostic'
OUTCOMES = ('message', 'silent', 'tool_delivered', 'deferred')
HISTORY_COUNTS = {'test_reseed_from_a_real_previous_seed': 4,
                  'test_call_skill_release_reseeds_the_old_payload_but_preserves_owner_edits': 22}
# Local-transport proofs that a new public address is verified before the Telegram menu moves, and that only a
# name-error answer for the exact host, not a temporary failure, counts as absence. Floors, so added cases pass.
NEW_MENU_FLOORS = {'test_public_dns_recovery_updates_the_menu_only_after_verification': 4,
                   'test_fresh_tunnel_can_verify_during_registration': 2,
                   'test_delayed_menu_install_needs_a_fresh_marker_without_resetting_grace': 2}
OLD_MENU_COUNTS = {'test_a_cold_start_restores_an_older_releases_owner_before_exposing_the_new_one': 2,
                   'test_a_chat_only_record_without_this_bots_button_restores_nothing': 4,
                   'test_a_restore_interrupted_after_the_derivation_resumes_from_its_ledger': 1}
# Closed exception buckets: the producer's error type is a class name, never its message.
ERROR_BUCKETS = {
    'AssertionError': 'assertion', 'Failed': 'assertion', 'Skipped': 'skipped',
    'TimeoutError': 'timeout', 'OSError': 'os', 'PermissionError': 'os', 'FileNotFoundError': 'os',
    'ValueError': 'value', 'RuntimeError': 'runtime', 'SystemExit': 'exit',
    'SubprocessError': 'subprocess', 'Exception': 'exception', 'BaseException': 'exception',
}
# Closed step names a live case may report: where its witness stopped.
STEP_VALUES = frozenset({
    'entered_settings', 'skill_form_captured', 'accounts_door_clicked', 'accounts_level_attached',
    'secret_row_found', 'secret_opener_label_ok', 'secret_editor_visible', 'secret_editor_empty',
    'accounts_captured', 'fact_published',
    'sheet_open', 'wakes_before_captured', 'wakes_pending_captured', 'choice_saved', 'sheet_reopened', 'wakes_cleared',
    'server_ready', 'restart_requested', 'restart_ready', 'dm_sent', 'dm_settled', 'witness_published',
})
# Closed measurement keys a live case may report: integers only, no text from the product.
NOTE_KEYS = frozenset({'scroll_width', 'inner_width', 'viewport_width', 'viewport_height', 'http_status'})
# The existing measurements are pixels or an HTTP status: 0..NOTE_MAX, never an arbitrary integer.
NOTE_MAX = 65_535


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


def present(root, rel):
    """Whether one bounded regular file exists: the base checkout may lack a new test module."""
    try:
        read_regular(root, rel, 4 * MIB)
    except (OSError, ValueError):
        return False
    return True


def groups(root, subject):
    home = subject['test_matrix']
    result = {f'candidate-{name}': ('candidate', name, home[name], limit, None) for name, limit in (
        ('presence', 1400), ('queue', 620), ('migration', 600), ('telegram', 780), ('history', 420),
        ('structural', 480), ('size', 240))}
    result['base-controls'] = ('base', 'pytest', home['base_controls'], 240, None)
    for part, file in LIVE_PARTS:
        result[f'candidate-live-{part}'] = ('candidate', 'live', [file], 650, None)
        result[f'base-live-{part}'] = ('base', 'live', [f'{file}::{case}' for case in BASE_LIVE[file]], 420, None)
    for file in home['node']:
        key = 'node-' + Path(file).name.removesuffix('.test.js')
        assert key not in result
        result[key] = ('candidate', 'node', [file], 110, None)
    for set_name, files_of_set, tests_of_set, budget in PAIRED_CONTROLS:
        for role in ('base', 'candidate'):
            chosen = [file for file in files_of_set
                      if present(root, role + '/' + file) and names_in(root, role + '/' + file, tests_of_set)]
            result[f'{role}-{set_name}'] = (role, 'pytest', chosen, budget,
                                            ' or '.join(tests_of_set))
    for set_name, files_of_set, tests_of_set, budget in DIAGNOSTIC_OBSERVATIONS:
        for role in ('base', 'candidate'):
            chosen = [file for file in files_of_set
                      if present(root, role + '/' + file) and names_in(root, role + '/' + file, tests_of_set)]
            result[f'{role}-{set_name}-diagnostic'] = (role, 'pytest', chosen, budget,
                                                       ' or '.join(tests_of_set))
    return result


def names_in(root, rel, names):
    """Which of the public test names this file holds: the paired control runs only those."""
    try:
        data = read_regular(root, rel, 4 * MIB).decode('utf-8', 'replace')
    except (OSError, ValueError):
        return False
    return any(name in data for name in names)


def exact_sources(root, subject):
    for role, pins in subject['subjects'].items():
        git = ['/usr/bin/git', '--no-optional-locks', '-C', str(root / role)]
        for ref, key in [('HEAD', 'commit'), ('HEAD^{tree}', 'tree')]:
            assert subprocess.check_output([*git, 'rev-parse', ref], text=True).strip() == pins[key]
        subprocess.run([*git, 'diff', '--quiet', 'HEAD', '--'], check=True)
        assert subprocess.check_output([*git, 'ls-files', '--others', '-z']) == b''


def live_files(root, files):
    """The harness paths of a live group: a plain file name or file::case."""
    out = []
    for entry in files:
        file, _, case = entry.partition('::')
        out.append(f'{root / "harness" / file}::{case}' if case else str(root / 'harness' / file))
    return out


def command(root, subject, role, kind, files, raw, select=None):
    py = str(root / 'venvs' / role / 'bin' / 'python')
    if kind == 'node':
        return [str(root / 'node-standalone/bin/node'), '--test', '--test-reporter=tap', *files]
    args = [py, '-m', 'pytest', *PY_FLAGS, '--timeout=130']
    if select:
        args += ['-k', select]
    if kind == 'size':
        # safe_test scrubs upstream bytes and GITHUB_EVENT inputs; set the exact comparison base only inside pytest.
        base = subject['subjects']['base']['commit']
        args = [py, '-c', 'import os,runpy,sys; os.environ["OURO_SIZE_RATCHET_BASE_REF"]=sys.argv.pop(1); '
                'runpy.run_module("pytest", run_name="__main__")', base, *PY_FLAGS, '--timeout=130', '-m', 'size_ratchet']
    if kind == 'live':
        args += ['--timeout=330', '-p', 'tests.conftest', '-c', str(root / role / 'pyproject.toml')]
        files = live_files(root, files)
    return [*args, *files, '--ci-evidence-dir', str(raw), '--junitxml', str(raw / 'private-junit.xml')]


def reap_orphans(keep):
    """Reap the terminated children this process inherited but did not start.

    The test unit runs with PrivatePIDs, so this runner is the init of a private PID
    namespace: every descendant the test tree abandons is reparented to it. An abandoned
    descendant that exits stays a zombie until that init waits for it, and a zombie still
    answers a liveness probe, so an artefact of the environment would decide a process test.
    The product's own CI runs no private PID namespace, where a host init reaps instead.
    Only children other than the one being waited for are reaped here.
    """
    reaped = 0
    while True:
        try:
            info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except (ChildProcessError, OSError):
            return reaped
        if info is None or not info.si_pid or info.si_pid == keep:
            return reaped
        try:
            os.waitpid(info.si_pid, os.WNOHANG)
        except (ChildProcessError, OSError):
            return reaped
        reaped += 1


def wait_with_reaping(proc, poll=0.01):
    """Wait for the run's own child while reaping what the test tree leaves behind."""
    total = peak = 0
    while True:
        reaped = reap_orphans(proc.pid)
        total += reaped
        peak = max(peak, reaped)
        try:
            pid, status = os.waitpid(proc.pid, os.WNOHANG)
        except ChildProcessError:
            return (proc.returncode if isinstance(proc.returncode, int) else 1), total, peak
        if pid == proc.pid:
            return os.waitstatus_to_exitcode(status), total, peak
        time.sleep(poll)


def run(root):
    proof = boundary(root, collecting=False)  # before the first product import
    work = root / 'work'
    subject = object_at(root, 'subject.json')
    exact_sources(root, subject)
    (work / 'proof/screens').mkdir()
    put(work / 'proof/boundary.json', proof)
    success = True
    for label, (role, kind, files, timeout, select) in groups(root, subject).items():
        raw = work / 'raw' / label
        raw.mkdir()
        env = os.environ.copy()
        if kind == 'live':
            env.update(OUROBOROS_RUN_UI_SMOKE='1', OUROBOROS_EXPECT_BROWSER_ENGINES='chromium,webkit',
                       OUROBOROS_UI_EVIDENCE_DIR=str(work / 'proof'))
        if not files:
            # The paired control has nothing to run on this base: it says so, it does not pass.
            put(raw / 'phase.json', {'exit': 0, 'settled': True, 'skipped': 'no_matching_tests'})
            continue
        launcher = [str(root / 'venvs' / role / 'bin/python'), '-I', '-S', '-B', 'scripts/safe_test.py',
                    '--temp-parent', '/tmp', '--evidence-dir', str(raw / 'safe'), '--timeout', str(timeout), '--']
        with (raw / 'stdio.out').open('xb') as out, (raw / 'stdio.err').open('xb') as err:
            proc = subprocess.Popen([*launcher, *command(root, subject, role, kind, files, raw, select)], cwd=root / role,
                                    env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        rc, reaped, peak = wait_with_reaping(proc)
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
        put(raw / 'phase.json', {'exit': rc, 'settled': settled, 'reaped_orphans': reaped, 'reaped_peak': peak})
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


def error_bucket(error_type):
    """A closed bucket for the producer's exception class: never the class name itself."""
    if not isinstance(error_type, str) or not error_type:
        return 'none'
    return ERROR_BUCKETS.get(error_type, 'other')


def record_summary(record):
    """Only safe typed facts about a producer record: numbers, booleans and closed enums."""
    selected = record.get('selected_nodeids')
    reports = record.get('reports')
    failures = record.get('collection_failures')
    skips = record.get('collection_skips')
    summary = {
        'format': record.get('format') if record.get('format') in (0, 1) else None,
        'session_exit_code': record.get('session_exit_code') if type(record.get('session_exit_code')) is int else None,
        'tests_collected': record.get('tests_collected') if type(record.get('tests_collected')) is int else None,
        'report_count': len(reports) if isinstance(reports, list) else None,
        'collection_failures': len(failures) if isinstance(failures, list) else None,
        'collection_skips': len(skips) if isinstance(skips, list) else None,
        'has_selection': isinstance(selected, list),
        'selected_count': len(selected) if isinstance(selected, list) else None,
        'worker_crash': bool(record.get('worker_crash')),
    }
    phases, outcomes = Counter(), Counter()
    if isinstance(reports, list):
        for row in reports:
            if isinstance(row, dict):
                phase = row.get('phase')
                outcome = row.get('outcome')
                phases[phase if phase in ('setup', 'call', 'teardown', 'crash') else 'other'] += 1
                outcomes[outcome if outcome in ('passed', 'failed', 'skipped') else 'other'] += 1
    summary['report_phases'] = dict(phases)
    summary['report_outcomes'] = dict(outcomes)
    return summary


def pytest_counts(record, files, kind, root, role, live_cases=None, control_tests=None):
    """The group's projection. Strict as before, but every way it can fail is named."""
    summary = record_summary(record)
    result = {'selected': summary['selected_count'], 'fully_run': 0, 'nonpass_reports': 0,
              'skipped_reports': 0, 'modules': {}, 'nonpass': [], 'complete': False, 'record': summary}
    selected = record.get('selected_nodeids')
    if not isinstance(selected, list) or not selected or not all(isinstance(node, str) for node in selected):
        # The producer fills the selection in pytest_collection_modifyitems: without it the
        # session never got past collection (an import error, a lane that collected nothing).
        result['projection'] = 'no_selection'
        return result
    expected = set(selected)
    hits, names = Counter(), Counter()
    for node in selected:
        path, *identity = node.split('[', 1)[0].split('::')
        if kind == 'live':
            path = Path(path).name
        if path not in files or not identity:
            result['projection'] = 'selection_mismatch'
            result['selection_mismatch'] = {'selected_count': len(selected)}
            return result
        hits[path] += 1
        names[identity[-1]] += 1
    reports = {node: Counter() for node in selected}
    nonpass = skipped = 0
    cases = {node: i for i, node in enumerate(selected)}
    known_names, faults = {}, []
    unexpected = Counter()
    rows = record.get('reports')
    if not isinstance(rows, list):
        result['projection'] = 'record_invalid'
        return result
    for row in rows:
        if not isinstance(row, dict):
            unexpected['malformed'] += 1
            continue
        node, phase, outcome = row.get('nodeid'), row.get('phase'), row.get('outcome')
        if not isinstance(node, str) or node not in expected:
            unexpected['unselected'] += 1
            continue
        if phase not in ('setup', 'call', 'teardown') or outcome not in ('passed', 'failed', 'skipped'):
            # A crashed worker or another phase the producer names: counted, never named.
            unexpected['phase_or_outcome'] += 1
            continue
        reports[node][phase] += not row.get('is_subtest')
        nonpass += outcome != 'passed' or 'wasxfail' in row
        skipped += outcome == 'skipped'
        if outcome != 'passed' or 'wasxfail' in row:
            file, *parts = node.split('[', 1)[0].split('::')
            if kind == 'live':
                file = Path(file).name
            if file not in known_names:
                public = read_regular(root, ('harness' if kind == 'live' else role) + '/' + file, 4 * MIB)
                known_names[file] = {node.name: node.lineno for node in ast.walk(ast.parse(public))
                                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
            function = parts[-1] if parts[-1] in known_names[file] else None
            # Only a public filename and function, plus an ordinal, the phase, the outcome, a
            # closed exception bucket and the definition line of that function: no parameter
            # ID, traceback, assertion message or raw response data.
            faults.append({'file': file, 'function': function,
                           'line': known_names[file].get(function) if function else None,
                           'index': cases[node], 'phase': phase, 'outcome': outcome,
                           'error': error_bucket(row.get('error_type'))})
    expected_phases = {'setup': 1, 'call': 1, 'teardown': 1}
    fully_run = sum(dict(phases) == expected_phases for phases in reports.values())
    complete = (record.get('format') == 1 and record.get('session_exit_code') == 0
                and record.get('tests_collected') == len(selected) == len(expected) == fully_run
                and set(hits) == set(files) and not nonpass and not record.get('collection_failures')
                and not record.get('collection_skips') and not record.get('worker_crash'))
    if kind == 'history':
        complete &= all(names[name] == count for name, count in HISTORY_COUNTS.items())
    if kind == 'telegram':
        complete &= all(names[name] == count for name, count in OLD_MENU_COUNTS.items())
        complete &= all(names[name] >= count for name, count in NEW_MENU_FLOORS.items())
    if kind == 'size':
        complete &= len(selected) == 7
    if kind == 'live':
        if role == 'base':
            expected_cases = set(live_cases or ())
            complete &= {Path(node.split('::', 1)[0]).name + '::' + node.split('::', 1)[1]
                         for node in selected} == expected_cases
        else:
            expected_hits = {name: count for name, count in LIVE.items() if name in files}
            complete &= dict(hits) == expected_hits
            if 'test_p05_upgrade.py' in files:
                complete &= (names['test_first_dm_across_publication'] == len(READINESS)
                             and names['test_public_gateway_dns_verdict'] == 1)
            if 'test_p05_browser.py' in files:
                complete &= (names['test_telegram_form_and_accounts_key'] == len(ENGINES)
                             and names['test_place_sheet_keeps_or_accepts_wakes'] == len(ENGINES) * len(CHOICES)
                             and names['test_place_sheet_pending_unpublished_signal'] == len(ENGINES))
    if unexpected:
        result['projection'] = 'unexpected_report'
        result['unexpected_reports'] = dict(unexpected)
        complete = False
    if control_tests is not None:
        missing = set(control_tests) - set(names)
        extra = set(names) - set(control_tests)
        if missing or extra:
            # Every declared test of a paired set must be selected, and nothing else: a file
            # that holds only part of the set, or a fuzzy -k match, is not that pair.
            result.setdefault('projection', 'control_missing' if missing else 'control_extra')
            complete = False
    result.update({'selected': len(selected), 'fully_run': fully_run, 'nonpass_reports': nonpass,
                   'skipped_reports': skipped, 'modules': dict(hits), 'nonpass': faults, 'complete': bool(complete)})
    return result


def group_outcome(result, phase):
    """One closed word for what the group did: setup, collection, truncation or a real red."""
    if not phase['settled']:
        return 'unsettled'
    projection = result.get('projection')
    if projection:
        return projection
    faults = result.get('nonpass') or []
    if faults:
        phases = {fault['phase'] for fault in faults}
        for name in ('setup', 'call', 'teardown'):
            if name in phases:
                return 'red_' + name
        return 'red_other'
    selected, fully_run = result.get('selected'), result.get('fully_run')
    if isinstance(selected, int) and isinstance(fully_run, int) and fully_run != selected:
        return 'truncated'
    if phase['exit'] != 0:
        return 'session_exit'
    return 'passed' if result.get('complete') else 'not_qualified'


def wanted_live():
    for engine, dimensions in ENGINES.items():
        yield (f'telegram-{engine}', ('skill-form', 'accounts-empty-editor'), dimensions)
        yield (f'base-telegram-{engine}', ('skill-form', 'accounts-empty-editor'), dimensions)
        for choice in CHOICES:
            yield (f'place-{engine}-{choice}', ('wakes-before', 'wakes-reopened'), dimensions)
        yield (f'place-pending-{engine}', ('wakes-pending',), dimensions)
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
    elif name.startswith('place-pending-'):
        result = {'wake_review': 'pending', 'readiness': 'off_in_skills',
                  'offered_facts': ['direct_message', 'signal.priority_inbox'],
                  'unpublished_signals': ['signal.priority_inbox'],
                  'custom_signal_disabled': True, 'custom_signal_waiting': True,
                  'core_direct_enabled': True, 'saved_legacy_removable': True,
                  'bulk_add_disabled': True, 'keep_enabled': True,
                  'place_writes': 0, 'places_unchanged': True}
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
        if not old:  # the only real Restart case is a managed same-program Restart, never an unmanaged Update
            result['restart_mode'] = 'managed_same_program'
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


def read_step(root, name):
    """The closed step a live case reached, with its closed measurements: nothing else."""
    try:
        value = object_at(root, f'work/proof/steps/{name}.json', 16_384)
    except (KeyError, TypeError, ValueError, OSError):
        return None
    step, notes = value.get('step'), value.get('notes')
    if not isinstance(step, str) or step not in STEP_VALUES or not isinstance(notes, dict):
        return None
    if any(key not in NOTE_KEYS or isinstance(number, bool) or not isinstance(number, int)
           or not 0 <= number <= NOTE_MAX for key, number in notes.items()):
        return None
    return {'step': step, 'notes': dict(notes)}


def pack(root, run_rc):
    box = boundary(root, collecting=True)
    subject = object_at(root, 'subject.json')  # created by staging, outside producer's writable subtree
    if object_at(root, 'work/proof/boundary.json') != box:
        return 2
    ready, missing = {}, []
    comparisons = {}
    controls = {f'{role}-{name}': tests for name, _files, tests, _budget in PAIRED_CONTROLS
                for role in ('base', 'candidate')}
    diagnostics = {f'{role}-{name}-diagnostic': tests
                   for name, _files, tests, _budget in DIAGNOSTIC_OBSERVATIONS
                   for role in ('base', 'candidate')}
    declared = {**controls, **diagnostics}
    for label, (role, kind, files, _, _) in groups(root, subject).items():
        prefix = f'work/raw/{label}'
        try:
            phase = object_at(root, prefix + '/phase.json')
            if type(phase['exit']) is not int or not -255 <= phase['exit'] <= 255 or type(phase['settled']) is not bool:
                raise ValueError('bad phase record')
            if 'skipped' in phase:
                # Only the producer's exact closed word for a declared observation with no matching
                # public test in this tree is a known state. The phase file lives in the
                # test-writable run tree, so any other value is a typed gap, never echoed.
                known = (label in declared and not files
                         and phase.get('skipped') == 'no_matching_tests'
                         and phase['exit'] == 0 and phase['settled'] is True)
                projection = 'no_matching_tests' if known else 'invalid_skip'
                result = {'complete': False, 'projection': projection}
                result.update(exit=phase['exit'], settled=phase['settled'], subject=role)
                ready[f'groups/{label}.json'] = json_bytes(result)
                missing.append(f'{label}: {projection}')
                continue
            names = sorted({entry.partition('::')[0] for entry in files})
            try:
                if kind == 'node':
                    result = node_counts(read_regular(root, prefix + '/stdio.out', 3 * MIB))
                else:
                    try:
                        record = object_at(root, prefix + '/results.json', 8 * MIB)
                    except FileNotFoundError:
                        # The producer writes results.json only at sessionfinish: no record means
                        # the session never finished (a kill or a timeout), not a green run.
                        result = {'complete': False, 'projection': 'record_missing'}
                    except (ValueError, OSError):
                        result = {'complete': False, 'projection': 'record_invalid'}
                    else:
                        result = pytest_counts(record, names, kind, root, role,
                                               live_cases={entry for entry in files if '::' in entry}
                                               if kind == 'live' and role == 'base' else None,
                                               control_tests=declared.get(label))
            except (KeyError, TypeError, ValueError, OSError):
                result = {'complete': False, 'projection': 'record_invalid'}
            result.update(exit=phase['exit'], settled=phase['settled'], subject=role)
            for key in ('reaped_orphans', 'reaped_peak'):
                if type(phase.get(key)) is int:
                    result[key] = phase[key]
            result['complete'] &= phase['exit'] == 0 and phase['settled']
            outcome = group_outcome(result, phase)
            result['outcome'] = outcome
            classify(root, result, label, files, role)
            if 'comparison' in result:
                comparisons[label] = result['comparison']
            ready[f'groups/{label}.json'] = json_bytes(result)
            if not result['complete']:
                missing.append(f'{label}: {outcome}')
        except (KeyError, TypeError, ValueError, OSError):
            missing.append(f'{label}: missing phase result')
    for name, scenes, dimensions in wanted_live():
        rel = f'work/proof/{name}.json'
        try:
            ready[f'live/{name}.json'] = json_bytes(public_fact(name, object_at(root, rel, 16_384)))
        except FileNotFoundError:
            missing.append(f'live/{name}.json: observation_missing')
        except (KeyError, TypeError, ValueError, OSError):
            missing.append(f'live/{name}.json: observation_invalid')
        step = read_step(root, name)
        if step is not None:
            ready[f'steps/{name}.json'] = json_bytes(step)
        for scene in scenes:
            filename = f'screens/{name}-{scene}.png'
            try:
                ready[filename] = clean_png(read_regular(root, 'work/proof/' + filename, 4 * MIB), dimensions)
            except FileNotFoundError:
                missing.append(f'{filename}: screenshot_missing')
            except (TypeError, ValueError, OSError, struct.error):
                missing.append(f'{filename}: screenshot_invalid')
    if not ready:
        return 2  # A subject, summary or boundary alone is not a result artifact.
    complete = run_rc == 0 and not missing
    summary = {'format': 1, 'complete': complete, 'unit_exit': run_rc, 'boundary': box,
               'missing_or_not_qualified': missing, 'proof_members': sorted(ready),
               'comparisons': comparisons}
    ready.update({'subject.json': json_bytes(subject), 'summary.json': json_bytes(summary)})
    archive = sys.stdout.buffer  # Actions traps stdout in a new runner-private regular file, not a job log.
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for filename, data in sorted(ready.items()):
            bundle.writestr(filename, data)
    return 0  # A valid partial archive is publishable, not a successful test run.


def source_bytes(root, files, role):
    """Whether this role's own copy of each file is the other role's bytes."""
    other = 'candidate' if role == 'base' else 'base'
    pairs = {}
    for file in files:
        try:
            pairs[file] = read_regular(root, role + '/' + file, 4 * MIB) == read_regular(root, other + '/' + file, 4 * MIB)
        except (OSError, ValueError):
            pairs[file] = None
    return pairs


def comparison_slot(label):
    """The declared comparison of one group, independently of execution outcome."""
    for role in ('base', 'candidate'):
        for name, _files, _tests, _budget in PAIRED_CONTROLS:
            if label == f'{role}-{name}':
                return SAME_SOURCE_PAIR
        for name, _files, _tests, _budget in DIAGNOSTIC_OBSERVATIONS:
            if label == f'{role}-{name}-diagnostic':
                return NON_EXACT_DIAGNOSTIC
    return None


def classify(root, result, label, files, role):
    """Attach the closed comparison word and the byte measurement behind it."""
    slot = comparison_slot(label)
    if slot is None:
        return result
    evidence = source_bytes(root, files, role)
    if not evidence:
        return result
    result['comparison'] = slot
    result['source_bytes'] = evidence
    return result


def check_archive(root, filename, rc):
    """The runner reads back only the private stream file, never a work-tree path."""
    subject = object_at(root, 'subject.json')
    required = {f'groups/{label}.json' for label in groups(root, subject)}
    optional = set()
    for name, scenes, _ in wanted_live():
        required.add(f'live/{name}.json')
        optional.add(f'steps/{name}.json')
        required.update(f'screens/{name}-{scene}.png' for scene in scenes)
    allowed = required | optional
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
            step = f'steps/{name}.json'
            if step in members:
                value = json.loads(bundle.read(step))
                assert isinstance(value, dict) and set(value) == {'step', 'notes'}
                assert isinstance(value['step'], str) and value['step'] in STEP_VALUES
                notes = value['notes']
                assert isinstance(notes, dict) and all(
                    key in NOTE_KEYS and isinstance(number, int) and not isinstance(number, bool)
                    and 0 <= number <= NOTE_MAX for key, number in notes.items())
                assert bundle.read(step) == json_bytes(value)
        if summary['complete']:
            assert rc == 0 and required <= members and not summary['missing_or_not_qualified']
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
