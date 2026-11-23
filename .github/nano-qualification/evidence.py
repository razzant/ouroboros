"""One-shot workflow glue. Consume the checkout's CI projections; never export them raw."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import zlib


HERE = Path(__file__).resolve().parent
PHASES = ('setup', 'call', 'teardown')
MAIN_ACTIONS = ('recipe', 'checkout', 'bind', 'python', 'uv', 'node', 'dependencies',
                'critical', 'paired', 'context', 'size', 'collection', 'selected', 'final_checkout')
BROWSER_ACTIONS = ('recipe', 'checkout', 'bind', 'python', 'uv', 'node', 'dependencies',
                   'install_browsers', 'jsunit', 'browser', 'existing', 'final_checkout')
BROWSER_PHASES = {'settings-js': 'jsunit', 'browser-observe': 'browser', 'browser-existing': 'existing'}
TARGETED = 'targeted_rerun'
TARGETED_ACTIONS = (*BROWSER_ACTIONS[:7], 'size', 'seams', 'inventory',
                    *(step for step in BROWSER_ACTIONS[7:] if step != 'existing'))
TARGETED_PHASES = {'size': 'size', 'seams': 'seams', 'inventory': 'inventory',
                   'settings-js': 'jsunit', 'browser-observe': 'browser'}
ACTIONS = {'main': MAIN_ACTIONS, 'browser': BROWSER_ACTIONS, 'targeted': TARGETED_ACTIONS}
REPORT_SCOPES = {'main': 'selected_linux_qualification_not_release_proof',
                 'browser': 'selected_linux_browser_observation_not_release_proof',
                 'targeted': 'targeted_linux_rerun_not_release_proof'}
REPORT_FILES = {'main': 'nano-qualification.json', 'browser': 'nano-qualification-browser.json',
                'targeted': 'nano-qualification-targeted.json'}
INVENTORY_CHECK = 'scripts/regenerate_inventories.py'
SEAMS_SECONDS = 1200  # the one pytest run of the targeted seams; the workflow's step ceiling is this plus two minutes
BROWSER_ENV = {'OUROBOROS_RUN_UI_SMOKE': '1', 'OUROBOROS_EXPECT_BROWSER_ENGINES': 'chromium',
               'OUROBOROS_UI_BROWSER_ENGINE': 'chromium'}
PNG = b'\x89PNG\r\n\x1a\n'
STEP_STATES = {'success', 'failure', 'skipped', 'cancelled'}
SAMPLE_LIMIT = 32
MAX_SOURCE = 64 * 1024 * 1024
MAX_FRAME = 4 * 1024 * 1024
KINDS = ('page', 'card')  # the two frames every owner keeps: the viewport and the card
FORMAT = 2  # a report of another format is not this recipe's
KILLED = (124, 137, 143)  # exit codes of a producer stopped by a timeout or a signal
PAIRED = 'paired'
# The projection never carries a word the recipe did not declare: a checkout fact is one of these.
CHECKOUT_WHY = ('bound', 'other_commit', 'other_tree', 'dirty_tree', 'git_unreadable')
UNATTRIBUTED = 'unattributed_base_not_checked'


def excise(node):
    return node.split('[', 1)[0]


def dict_of(value):
    return value if isinstance(value, dict) else {}


def known_tests(plan):
    """Every test module a public node name may come from: the head's own and the overlay the recipe places."""
    return set(plan['all_tests']) | {plan['observation']['test']}


def matches(node, selector):
    if '[' in selector:
        return node == selector
    bare = excise(node)
    return bare == selector or bare.startswith(selector + '::')


def names(nodes, allowed):
    """Keep tracked relative module names and symbol identifiers, never parameter text."""
    result = []
    for node in list(nodes)[:SAMPLE_LIMIT]:
        bare = excise(node)
        if (bare.split('::')[0] not in allowed or len(bare) > 240
                or not re.fullmatch(r'tests/[A-Za-z0-9_/-]+\.py(?:::[A-Za-z_][A-Za-z0-9_]*)*', bare)):
            result.append('<node withheld>')
        elif bare != node:
            result.append(bare + '[sha256:' + hashlib.sha256(node.encode()).hexdigest()[:16] + ']')
        else:
            result.append(bare)
    return result


def read_json(path):
    try:
        with path.open('rb') as stream:
            raw = stream.read(MAX_SOURCE + 1)
        if len(raw) > MAX_SOURCE:
            return None, 'oversized'
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None, 'invalid'
        return obj, 'read'
    except FileNotFoundError:
        return None, 'missing'
    except (OSError, ValueError, RecursionError):
        return None, 'invalid'


def result_rows(rows):
    return (isinstance(rows, list) and all(isinstance(row, dict)
            and isinstance(row.get('nodeid'), str)
            and row.get('phase') in (*PHASES, 'crash')
            and row.get('outcome') in ('passed', 'failed', 'skipped')
            for row in rows))


def journal(path):
    rows, running, dropped = [], {}, 0
    try:
        with path.open('rb') as stream:
            data = stream.read(MAX_SOURCE + 1)
    except FileNotFoundError:
        return [], [], 'missing', None
    except OSError:
        return [], [], 'invalid', None
    state = 'partial' if len(data) <= MAX_SOURCE else 'truncated'
    for line in data[:MAX_SOURCE].splitlines():
        try:
            row = json.loads(line)
        except (ValueError, RecursionError):
            dropped += 1  # An unfinished write is evidence of neither success nor failure.
            continue
        node = row.get('nodeid') if isinstance(row, dict) else None
        if not isinstance(node, str):
            dropped += 1
        elif row.get('event') == 'start':
            running[node] = None
        elif row.get('event') == 'finish':
            running.pop(node, None)
        elif row.get('event') == 'report':
            if result_rows([row]):
                rows.append(row)
                if row['phase'] == 'crash':
                    running.pop(node, None)
            else:
                dropped += 1
    return rows, sorted(running), state, dropped


def case_results(rows):
    """(pytest outcome, responsible phase); a setup or call alone never proves a pass."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['nodeid']].append(row)
    cases = {}
    for node, group in grouped.items():
        whole = [row for row in group if not row.get('is_subtest')]
        phases = [row['phase'] for row in whole]
        failed = next((row for row in group if row['outcome'] == 'failed' or row['phase'] == 'crash'), None)
        skipped = next((row for row in group if row['outcome'] == 'skipped'), None)
        if failed:
            cases[node] = ('failed', failed['phase'])
        elif len(phases) != len(set(phases)) or not {'setup', 'teardown'} <= set(phases):
            cases[node] = ('unknown', 'execution')
        elif skipped:
            label = 'xfailed' if skipped.get('wasxfail') is not None else 'skipped'
            cases[node] = (label, skipped['phase'])
        elif set(phases) == set(PHASES):
            cases[node] = ('xpassed' if any(row.get('wasxfail') is not None for row in group) else 'passed', 'call')
        else:
            cases[node] = ('unknown', 'execution')
    return cases


def observed_bool(value):
    return value if type(value) is bool else None


def exit_code(value):
    return value if type(value) is int and 0 <= value <= 255 else None


def load_session(directory):
    data, state = read_json(directory / 'results.json')
    if state == 'read' and (data.get('format') != 1 or not result_rows(data.get('reports'))
                            or exit_code(data.get('session_exit_code')) is None
                            or type(data.get('tests_collected')) is not int or data['tests_collected'] < 0
                            or not isinstance(data.get('collection_failures'), list)
                            or not isinstance(data.get('collection_skips'), list)):
        state = 'invalid'
    events, in_flight, journal_state, dropped = journal(directory / 'events.jsonl') if state != 'read' else ([], [], 'unused', None)
    return (data if state == 'read' else None, state,
            case_results(data['reports'] if state == 'read' else events), in_flight, journal_state, dropped)


def _git(*args):
    return subprocess.run(['git', *args], capture_output=True, check=True, text=True, timeout=30).stdout


def checkout_state(subject):
    """``(bound, why, unowned)``: the checkout is the subject only when HEAD and its tree are the subject's and nothing else
    is in the tree. ``why`` is one of ``CHECKOUT_WHY``; ``unowned`` counts entries in the way (never their names)."""
    try:
        found = _git('rev-parse', 'HEAD', 'HEAD^{tree}').split()
        dirty = _git('status', '--porcelain=v1', '--untracked-files=all').splitlines()
    except (subprocess.SubprocessError, OSError):
        return False, 'git_unreadable', 0
    if found[:1] != [subject['commit']]:
        return False, 'other_commit', len(dirty)
    if found[1:] != [subject['tree']]:
        return False, 'other_tree', len(dirty)
    return (False, 'dirty_tree', len(dirty)) if dirty else (True, 'bound', 0)


def bound_git(subject):
    return checkout_state(subject)[0]


def bind_anew(subject):
    """Put the checkout on ``subject`` with one plain ``git switch --detach``: never a reset, a clean or a forced switch, so
    nothing the recipe does not own is touched, and a tree that still is not the subject's afterwards is reported as such.
    Returns ``already``, ``restored`` or ``failed``."""
    if bound_git(subject):
        return 'already'
    try:
        _git('switch', '--detach', subject['commit'])
    except (subprocess.SubprocessError, OSError):
        return 'failed'
    return 'restored' if bound_git(subject) else 'failed'


def clear_own_overlay(plan):
    """A phase killed during its run leaves the overlay it placed. It is removed only when it is byte for byte the planned
    file at the planned path: anything else at that path is not the recipe's and stays where it is."""
    seen = dict_of(plan.get('observation'))
    try:
        target = Path(seen['test'])
        if (target.is_file() and not target.is_symlink()
                and hashlib.sha256(target.read_bytes()).hexdigest() == seen['sha256']):
            target.unlink()
            return True
    except (KeyError, TypeError, OSError):
        pass
    return False


def put_json(path, body):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(body, ensure_ascii=True, indent=2) + '\n', encoding='ascii')
    temp.replace(path)


def receipt(root, label, subject):
    data, _ = read_json(root / label / 'receipt.json')
    if data and all(data.get(field) == value for field, value in subject.items()):
        return data
    return {}


def phase_specs(plan, scope='main'):
    if scope == 'targeted':
        return {label: ('head', step) for label, step in TARGETED_PHASES.items()}
    if scope == 'browser':
        return {label: ('head', step) for label, step in BROWSER_PHASES.items()}
    specs = {label: ('head', step) for label, step in [('critical', 'critical'), ('context', 'context'),
             ('size', 'size'), ('collection', 'collection'), ('parallel', 'selected'), ('exclusive', 'selected')]}
    for role in ('base', 'head'):
        specs.update({f'{role}-{label}': (role, 'paired') for label, _node in plan['controls']})
    return specs


def spec(plan, label):
    if plan.get('selection_profile') == TARGETED:
        return phase_specs(plan, 'targeted')[label]
    return phase_specs(plan, 'browser' if label in BROWSER_PHASES else 'main')[label]


def execute(root, plan, label, command, *, cwd=None, env=None, overlay=None):
    """Run one producer in a clean, bound checkout. ``overlay`` is ``(path, bytes)``: a QA-only file placed for the run
    and removed before the checkout is judged again, so no phase ever leaves the tested tree different.

    A head phase that is not one of the paired controls first binds the head anew when an earlier phase (a failed or killed
    one) left the checkout elsewhere or left the recipe's own overlay behind; the earlier failure stays in its own receipt,
    and a tree that cannot be bound is not tested (the receipt says why, by count and by a fixed word)."""
    role, _ = spec(plan, label)
    subject = plan[role]
    directory = root / label
    directory.mkdir()
    cleared = clear_own_overlay(plan)
    rebound = bind_anew(subject) if role == 'head' and not label.startswith('head-') else 'not_needed'
    start, why, unowned = checkout_state(subject)
    state = {**subject, 'checkout_before': start, 'checkout_after': None, 'producer_exit': None,
             'checkout_rebound': rebound, 'checkout_why': why, 'checkout_unowned_entries': unowned,
             'leftover_overlay_removed': cleared}
    if overlay:
        state['overlay'] = {'path': overlay[0], 'sha256': hashlib.sha256(overlay[1]).hexdigest()}
    put_json(directory / 'receipt.json', state)
    if not start:
        return
    placed = None
    try:
        if overlay:
            target = Path(overlay[0])
            with target.open('xb') as stream:  # an existing file is a collision, never ours to remove
                stream.write(overlay[1])
            placed = target
        with (directory / 'console.log').open('xb') as stream:
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, cwd=cwd, env=env).returncode
        state['producer_exit'] = code if code >= 0 else 128 - code
        put_json(directory / 'receipt.json', state)  # Keep the exit even if the subsequent Git check fails.
    finally:
        if placed is not None:
            placed.unlink(missing_ok=True)
    state['checkout_after'] = bound_git(subject)
    put_json(directory / 'receipt.json', state)


def pytest_command(root, label, limit, args, *, base=None):
    directory = root / label
    setup = ['env', '-u', 'GITHUB_EVENT_NAME', '-u', 'GITHUB_REF',
             'OURO_SIZE_RATCHET_BASE_REF=' + base, 'OURO_BOOK_GROWTH_MODE=block'] if base else []
    return [sys.executable, '-I', '-S', 'scripts/safe_test.py', '--timeout', str(limit),
            '--evidence-dir', str(directory / 'private'), '--', *setup, sys.executable, '-m', 'pytest',
            '-o', 'addopts=', '-p', 'no:cacheprovider', '-q', '--tb=no',
            '--ci-evidence-dir', str(directory), *args]


def session_issues(data, cases, *, collection=False, interrupted=False):
    errors = []
    if data is None:
        return ['final_results_missing_or_invalid']
    if not (isinstance(data.get('environment'), dict) and data['environment'].get('platform') == 'Linux'):
        errors.append('linux_not_observed')
    actual = data.get('github')
    if any(os.environ.get('GITHUB_' + k.upper()) and (not isinstance(actual, dict)
            or actual.get(k) != os.environ['GITHUB_' + k.upper()]) for k in ('sha', 'run_id', 'run_attempt')):
        errors.append('report_not_from_this_actions_attempt')
    if not interrupted and data['session_exit_code'] != 0:
        errors.append('pytest_nonzero_exit')
    if data['collection_failures']:
        errors.append('collection_failed')
    if not collection and (len(cases) > data['tests_collected'] or
                           not interrupted and len(cases) != data['tests_collected']):
        errors.append('collected_reported_count_differs')
    return errors


def membership(expected, parallel, exclusive, plan):
    actual = set(parallel) | set(exclusive)
    found = {node.split('::')[0] for node in expected}
    passed = {node.split('::')[0] for node, (outcome, _) in {**parallel, **exclusive}.items() if outcome == 'passed'}
    failures = sorted(node for node, (outcome, _) in {**parallel, **exclusive}.items()
                      if outcome in ('unknown', 'failed'))
    is_witness = exclusive.get(plan['exclusive']) == ('passed', 'call') and plan['exclusive'] not in parallel
    return {'missing': sorted(expected - actual), 'unexpected': sorted(actual - expected),
            'duplicate': sorted(set(parallel) & set(exclusive)), 'failed_or_unsettled': failures,
            'required_without_pass': sorted(set(plan['required_default']) - passed),
            'unplanned_modules': sorted(found - set(plan['selected'])),
            'no_default_nodes': sorted(set(plan['selected']) - found), 'exclusive_passed': is_witness}


def requirements(cases, selectors):
    """A family may be selected without an old cardinality; every matching node must settle green."""
    absent = []
    for node, count in selectors:
        selected = [outcome for case, (outcome, _phase) in cases.items() if matches(case, node)]
        if not selected or count is not None and len(selected) != count or any(v != 'passed' for v in selected):
            absent.append(node)
    return absent


def context_rounds(root):
    verdict, state = read_json(root / 'verdict.json')
    rounds = {}
    if state == 'read' and isinstance(verdict.get('rounds'), list):
        for row in verdict['rounds']:
            if (isinstance(row, dict) and isinstance(row.get('dir'), str)
                    and re.fullmatch(r'\.|retry-(?:[1-9]|10)', row['dir']) and row['dir'] not in rounds):
                code = exit_code(row.get('exit'))
                if code is None:
                    return {}, 'invalid', 'unknown'
                rounds[row['dir']] = code
            else:
                return {}, 'invalid', 'unknown'
    elif state == 'read':
        state = 'invalid'
    outcome = verdict.get('verdict') if verdict else None
    if state == 'read':
        order = ['.', *(f'retry-{n}' for n in range(1, len(rounds)))]
        if set(order) != set(rounds):
            return {}, 'invalid', 'unknown'
        rounds = {name: rounds[name] for name in order}
    return rounds, state, outcome if outcome in ('green', 'flaky', 'accepted', 'red') else 'unknown'


def context_coverage(directory, selectors):
    """Project native ci_retry decisions against this attempt's actual case evidence."""
    rounds, state, verdict = context_rounds(directory)
    if state != 'read' or verdict not in ('green', 'flaky'):
        return ['context_gate_verdict_unconfirmed'], list(selectors)
    first, _, cases, *_ = load_session(directory)
    expected = set(cases)
    issues = []
    if first:
        crash = first.get('worker_crash')
        # Context gate uses no shard. Its crash record names the rest for ci_retry.
        if isinstance(crash, dict) and crash.get('collection_available') is True:
            unfinished = crash.get('unfinished')
            if isinstance(unfinished, list) and all(isinstance(n, str) for n in unfinished):
                expected.update(unfinished)
        if len(expected) != first['tests_collected']:
            issues.append('context_first_collection_unconfirmed')
    final = {}
    for attempt, code in rounds.items():
        data, _, observed, *_ = (load_session(directory / attempt) if attempt != '.'
                                 else (first, None, cases))
        # Nonzero attempts may have left a partial collection/run. The first
        # assignment and all later reports must still reconcile by exact node ID.
        pending = expected - {n for n, (outcome, _) in final.items() if outcome not in ('failed', 'unknown')}
        if (session_issues(data, observed, interrupted=code != 0) or not observed.keys() & pending
                or data['session_exit_code'] != code or data['tests_collected'] > len(expected)
                or not observed.keys() <= expected):
            issues.append('context_first_attempt_unproven' if attempt == '.' else 'context_retry_not_proven')
        for node, result in observed.items():
            if node not in pending:
                continue
            prior = final.get(node)
            if prior and prior[0] == 'failed' and result[0] != 'passed':
                continue  # A skip is not a successful retry after failure.
            final[node] = result
    if (final.keys() != expected or not final or any(
            outcome in ('failed', 'unknown') for outcome, _ in final.values())):
        issues.append('context_case_unsettled')
    if any(not any(matches(case, selector) for selector in selectors) for case in final):
        issues.append('unexpected_context_node')
    return issues, [node for node in selectors if not any(matches(case, node) for case in final)]


def tap_summary(path):
    """The footer counts of ``node --test --test-reporter=tap``; None when no complete footer was printed."""
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None
    counts = {key: int(value) for key, value in re.findall(r'^# (tests|suites|pass|fail|cancelled|skipped|todo) (\d+)$', text, re.M)}
    return counts if {'tests', 'pass', 'fail', 'cancelled', 'skipped', 'todo'} <= counts.keys() else None


def jsunit_issues(directory, plan):
    if not plan['jsunit']['files']:
        return []  # no browser-module test follows from this delta; the report says so
    counts = tap_summary(directory / 'console.log')
    if counts is None:
        return ['jsunit_summary_missing']
    return ([] if counts['tests'] >= 1 and counts['pass'] == counts['tests'] else ['jsunit_not_all_passed']) \
        + (['jsunit_failed_or_cancelled'] if counts['fail'] or counts['cancelled'] else []) \
        + (['jsunit_skipped_or_todo'] if counts['skipped'] or counts['todo'] else [])


def same(seen, wanted):
    return type(seen) is type(wanted) and seen == wanted


def facts_differ(row, want):
    """The names of the facts a recorded row shows otherwise than the plan expects (None: recorded, not asserted)."""
    row = dict_of(row)
    return [name for name, wanted in dict_of(want).items() if wanted is not None and not same(row.get(name), wanted)]


def frame_name(owner, kind):
    return f'{owner}-{kind}.png'


def frame_owners(plan):
    """Every owner of a frame, from the plan's own tables (a state, a Test line, a journey step that keeps frames): the one list
    the producer's names, the reader's checks and the upload allowlist all come from."""
    seen = plan['observation']
    return [*seen['expect'], *(f'test-{outcome}' for outcome in seen['lines']),
            *(step for step, row in seen['journey'].items() if row['frames'])]


def allowed_frames(plan):
    """The only screenshot names a public report may carry: both frames of every owner and ``failed-<place>-page.png``."""
    seen = plan['observation']
    return {frame_name(owner, kind) for owner in frame_owners(plan) for kind in KINDS} | {
        f'failed-{place}-page.png' for place in (*seen['expect'], *seen['journey'], 'setup')}


def png_shape(data):
    """``(width, height, owner label, distinct bytes of the pixel rows)`` of a well-formed PNG, else None. The rows are inflated
    under a cap, so a frame cannot be a bomb; one flat colour holds a handful of distinct bytes, a drawn page many."""
    try:
        if data[:8] != PNG or data[12:16] != b'IHDR':
            return None
        width, height = struct.unpack('>II', data[16:24])
        position, pixels, label = 8, b'', ''
        while position + 12 <= len(data):
            size, kind = struct.unpack('>I4s', data[position:position + 8])
            body = data[position + 8:position + 8 + size]
            if kind == b'IDAT':
                pixels += body
            elif kind == b'tEXt' and body.startswith(b'qa-owner\0'):
                label = body[9:].decode('ascii')
            position += 12 + size
        inflater = zlib.decompressobj()
        rows = inflater.decompress(pixels, 64 << 20)
        return None if inflater.unconsumed_tail else (width, height, label, len(set(rows)))
    except (struct.error, zlib.error, UnicodeDecodeError):
        return None


def png_ok(folder, frame, name, viewport=None):
    """A frame is its own when it is the planned file, a PNG of the recorded size and digest that was stamped with that very
    name, drawn (more than one flat colour) and, for a page frame, the viewport it was measured at (any screen density: the
    proportions and at least the width)."""
    try:
        data = (folder / name).read_bytes() if re.fullmatch(r'[A-Za-z0-9._-]+\.png', name) else b''
    except OSError:
        return False
    shape = png_shape(data)
    if not (isinstance(frame, dict) and frame.get('file') == name and 0 < len(data) <= MAX_FRAME and frame.get('bytes') == len(data)
            and frame.get('sha256') == hashlib.sha256(data).hexdigest() and shape and shape[2] == name[:-4] and shape[3] > 8):
        return False
    return viewport is None or (shape[0] >= viewport[0] and shape[0] * viewport[1] == shape[1] * viewport[0])


def frames_ok(folder, row, owner):
    """The row keeps exactly its own two frames."""
    frames, measure = dict_of(dict_of(row).get('frames')), dict_of(dict_of(row).get('measure'))
    viewport = measure.get('viewport')
    viewport = viewport if isinstance(viewport, list) and len(viewport) == 2 and all(type(n) is int and n > 0 for n in viewport) else None
    return sorted(frames) == sorted(KINDS) and all(
        png_ok(folder, frames[kind], frame_name(owner, kind), viewport if kind == 'page' else None) for kind in KINDS)


RECEIPTS = {'states': 'observation.json', 'lines': 'observation-lines.json', 'journey': 'observation-journey.json',
            'failure': 'observation-failure.json'}


def observation_files(root):
    """The observation's own receipts, each as read: ``(folder, {kind: data or None})``."""
    folder = root / 'observation'
    return folder, {kind: read_json(folder / name)[0] for kind, name in RECEIPTS.items()}


def measure_issues(measure):
    """What a recorded measure says, judged from its numbers (never from the test's own verdict): the Settings page or wizard
    content that scrolls, the document, and what the page clips. A measure that is not there is not green."""
    measure = dict_of(measure)
    container, document = dict_of(measure.get('container')), dict_of(measure.get('document'))
    numbers = (container.get('scroll_width'), container.get('client_width'), document.get('scroll_width'), document.get('inner_width'))
    viewport = measure.get('viewport')
    if (not all(type(n) is int and n >= 0 for n in numbers) or not isinstance(measure.get('clipped'), list)
            or not (isinstance(viewport, list) and len(viewport) == 2 and all(type(n) is int and n > 0 for n in viewport))):
        return ['not_measured']
    issues = []
    if numbers[0] > numbers[1] + 1:
        issues.append('container_overflow')
    if numbers[2] > numbers[3] + 1:
        issues.append('document_overflow')
    if measure['clipped']:
        issues.append('clipped')
    return issues


def narrow(measure):
    """The measure was taken on a phone-wide viewport (the one the phone states and steps are for)."""
    viewport = dict_of(measure).get('viewport')
    return isinstance(viewport, list) and len(viewport) == 2 and type(viewport[0]) is int and viewport[0] <= 430


def observation_checkout_issue(part, plan):
    got = dict_of(part.get('checkout'))
    if not all(isinstance(got.get(key), str) and re.fullmatch(r'[0-9a-f]{40}', got[key]) for key in ('commit', 'tree')):
        return 'observation_checkout_unconfirmed'
    if any(got[key] != plan['head'][key] for key in ('commit', 'tree')):
        return 'observation_ran_on_another_checkout'
    return None


def row_issues(row, owner, want, folder, framed=True, worded=True):
    """Everything wrong with one recorded row (a state, a Test line or a journey step) against the plan, whatever the test said of
    itself: the facts it shows, the page it was measured on, and the frames that are its own. ``worded``: the row also carries the
    test's own verdict on words the plan cannot hold (a note, a Test line), which must be ok."""
    row = dict_of(row)
    issues = [f'{name}_differs' for name in facts_differ(row, want)] + ([] if not worded or row.get('ok') is True else ['not_ok'])
    if framed:
        issues += measure_issues(row.get('measure'))
        if owner.endswith('-phone') and not narrow(row.get('measure')):
            issues.append('phone_viewport_not_applied')
        if not frames_ok(folder, row, owner):
            issues.append('frame_missing_or_not_a_png')
    return issues


def judge(parts, folder, plan):
    """Everything wrong with what the browser recorded, from its receipts (or their projection) and the frames beside them:
    the plan's states, Test lines and journey steps, each row in the plan's order with its facts, measure and own frames."""
    seen, problems = plan['observation'], []
    owed = {'states': ('states', 'state', [(n, n, want, True, True) for n, want in seen['expect'].items()]),
            'lines': ('lines', 'outcome', [(o, f'test-{o}', seen['line_expect'], True, True) for o in seen['lines']]),
            'journey': ('steps', 'step', [(s, s, row['expect'], row['frames'], False) for s, row in seen['journey'].items()])}
    for kind, (key, ident, wanted) in owed.items():
        part = dict_of(parts.get(kind))
        rows = part.get(key) if isinstance(part.get(key), list) else []
        done = part.get('complete') is True
        problems += [] if done else [f'{kind}_incomplete']
        if [dict_of(row).get(ident) for row in rows] != [row[0] for row in wanted][:len(rows)] or done and len(rows) != len(wanted):
            problems.append(f'{kind}_rows_differ_from_the_plan')
        for row, (_, owner, want, framed, worded) in zip(rows, wanted):
            problems += [f'{owner}:{issue}' for issue in row_issues(row, owner, want, folder, framed, worded)]
        if issue := observation_checkout_issue(part, plan):
            problems.append(issue)
    if not same(dict_of(parts.get('states')).get('reserved_host_requests'), 0):
        problems.append('network_to_reserved_host')
    if not same(dict_of(parts.get('lines')).get('saves_during_test'), 0):
        problems.append('a_test_saved_something')
    return list(dict.fromkeys(problems))


def failure_rows(root, plan):
    """What a recorded failure may say in public: the node and the place only if the plan declared them, a code only from the
    declared vocabulary, a screenshot only if it is that place's own, verified. Nothing else of it is kept."""
    folder, kinds = observation_files(root)
    seen = plan['observation']
    functions = {node.split('::', 1)[1] for node in seen['nodes']}
    places = {*seen['expect'], *seen['journey'], 'setup'}
    rows = []
    for row in (dict_of(kinds['failure']).get('failures') or [])[:8]:
        row = dict_of(row)
        frame, place = row.get('frame'), row.get('state') if row.get('state') in places else 'withheld'
        rows.append({'node': row.get('node') if row.get('node') in functions else 'withheld', 'state': place,
                     'code': row.get('code') if row.get('code') in seen['codes'] else 'other',
                     'frame': {key: frame.get(key) for key in ('file', 'bytes', 'sha256')}
                     if place != 'withheld' and isinstance(frame, dict) and png_ok(folder, frame, f'failed-{place}-page.png') else None})
    return rows


def observation_issues(root, plan):
    folder, parts = observation_files(root)
    return [*judge(parts, folder, plan), *(['observation_failure_recorded'] if failure_rows(root, plan) else [])]


def existing_issues(cases, data, plan):
    files = plan['observation']['existing']
    if not files:
        return [], []
    problems = ([] if cases else ['existing_suites_ran_no_case']) \
        + (['existing_case_not_passed'] if any(outcome != 'passed' for outcome, _ in cases.values()) else [])
    absent = [f for f in files if not any(case.startswith(f + '::') and outcome == 'passed' for case, (outcome, _) in cases.items())]
    return problems, absent


def check(root, plan, label):
    role, _ = spec(plan, label)
    state = receipt(root, label, plan[role])
    directory = root / label
    data, _, cases, *_ = load_session(directory)
    is_collection = label == 'collection'
    problems = [] if label in ('context', 'settings-js', 'inventory') else session_issues(data, cases, collection=is_collection)
    if (exit_code(state.get('producer_exit')) != 0 or state.get('checkout_after') is not True
            or state.get('checkout_before') is not True):
        problems.append('producer_or_checkout_unconfirmed')
    detail = {}
    if is_collection:
        ids = data.get('selected_nodeids') if data else None
        if not (isinstance(ids, list) and ids and all(isinstance(n, str) for n in ids)
                and len(set(ids)) == len(ids) == data['tests_collected'] and not cases):
            problems.append('collection_membership_unavailable')
        elif not set(n.split('::')[0] for n in ids) <= set(plan['selected']):
            problems.append('collection_left_the_rendered_selection')
        elif plan['exclusive'] not in ids:
            problems.append('exclusive_witness_not_collected')
    elif label == 'critical':
        specs = [(node, count) for _, node, count in plan['critical']]
        detail['required_without_pass'] = requirements(cases, specs)
        if any(not any(matches(node, selector) for selector, _ in specs) for node in cases):
            problems.append('unexpected_critical_node')
    elif label == 'size':
        detail['required_without_pass'] = requirements(cases, [(node, 1) for node in plan['size']])
        if plan.get('selection_profile') == TARGETED and set(cases) != set(plan['size']):
            problems.append('size_witnesses_differ_from_plan')
        if any(outcome != 'passed' for outcome, _ in cases.values()) or data and data['collection_skips']:
            problems.append('size_case_not_passed')
    elif label == 'seams':
        detail['required_without_pass'] = requirements(cases, [(node, None) for node in plan['seams']])
        if (any(not any(matches(case, node) for node in plan['seams']) for case in cases)
                or data and data['collection_skips']):
            problems.append('seams_witnesses_differ_from_plan')
    elif label == 'inventory':
        pass  # the repository's read-only generator signals its verdict by exit, not pytest
    elif label == 'context':
        issues, detail['required_without_report'] = context_coverage(directory, plan['context'])
        problems.extend(issues)
    elif label == 'settings-js':
        problems.extend(jsunit_issues(directory, plan))
    elif label == 'browser-observe':
        nodes = plan['observation']['nodes']
        detail['required_without_pass'] = requirements(cases, [(node, 1) for node in nodes])
        if any(not any(matches(case, node) for node in nodes) for case in cases):
            problems.append('unexpected_observation_node')
        problems.extend(observation_issues(root, plan))
    elif label == 'browser-existing':
        more, detail['files_without_pass'] = existing_issues(cases, data, plan)
        problems.extend(more)
    elif label == 'exclusive':
        detail['required_without_pass'] = requirements(cases, [(plan['exclusive'], 1)])
    elif label not in ('parallel', 'exclusive'):
        node = dict(plan['controls'])[label.split('-', 1)[1]]
        detail['required_without_pass'] = requirements(cases, [(node, 1)])
        if len(cases) != 1:
            problems.append('wrong_paired_control_count')
    if any(detail.values()):
        problems.append('required_witness_unsettled')
    return problems, detail


def paired_selection(root, plan):
    """Per control: the one case each side collected, and whether the two sides collected the same one (how the control's bytes
    relate in the two trees is proved at render, in the plan, and is not a fact of the run)."""
    rows = {}
    for label, _node in plan['controls']:
        sides = {role: sorted(load_session(root / f'{role}-{label}')[2]) for role in ('base', 'head')}
        rows[label] = {'base_cases': len(sides['base']), 'head_cases': len(sides['head']),
                       'same_single_case': len(sides['head']) == 1 and sides['base'] == sides['head']}
    return rows


def run_step(root, plan, step):
    def tested(label, command, **options):
        execute(root, plan, label, command, **options)
        issues, _ = check(root, plan, label)
        subject = plan[spec(plan, label)[0]]
        status = receipt(root, label, subject)
        # A red outside the six paired controls was run on the head alone: the base was not asked, so the line says so.
        words = (' (' + UNATTRIBUTED + ')' if not label.startswith(('base-', 'head-')) else '') if issues else ''
        print(f'{label}: {subject["commit"]} producer_exit={status.get("producer_exit")} '
              f'coverage={"red" if issues else "ok"}{words}' + (': ' + ', '.join(issues) if issues else ''), flush=True)
        if status.get('checkout_rebound') in ('restored', 'failed'):
            print(f'{label}: checkout {status["checkout_rebound"]} before this phase '
                  f'({status.get("checkout_why")}; {status.get("checkout_unowned_entries")} entries in the way)', flush=True)
        # Already-completed phases have a bounded public trace even if a later cancellation prevents the final
        # Actions report/upload from running: counts and declared names only, never a console or a message.
        if issues or label == 'context':
            directory = root / label
            rounds, _, verdict = context_rounds(directory) if label == 'context' else ({}, None, None)
            for attempt in dict.fromkeys(['.', *rounds]):
                row = safe_attempt(directory / attempt, known_tests(plan), rounds.get(attempt))
                row['outliers'] = row['outliers'][:8]
                for field in ('collection_failures', 'collection_skips', 'in_flight'):
                    row[field]['sample'] = row[field]['sample'][:8]
                print(json.dumps({'phase': label, 'attempt': attempt, 'verdict': verdict, **row},
                                 ensure_ascii=True), flush=True)
            if label in BROWSER_PHASES:
                for failure in failure_rows(root, plan):
                    print(json.dumps({'phase': label, 'observation_failure': failure['code'], 'node': failure['node'],
                                      'state': failure['state']}, ensure_ascii=True), flush=True)
        return bool(issues)

    nothing = [sys.executable, '-c', 'pass']  # a planned-away producer still leaves a bound, honest receipt
    if step == 'paired':
        # Each side is bound before its controls; a failed or killed control never strands the next side or a later phase on
        # the base: the head is bound again at the end, and each later head phase binds it anew before it runs (``execute``).
        red = False
        try:
            for role in ('base', 'head'):
                moved = bind_anew(plan[role])
                print(f'paired: {role} {moved}', flush=True)
                for label, node in plan['controls']:
                    phase = f'{role}-{label}'
                    red |= tested(phase, pytest_command(root, phase, 80, ['--timeout=60', node]))
        finally:
            print(f'paired: head {bind_anew(plan["head"])} after the controls', flush=True)
        selection = paired_selection(root, plan)
        different = sorted(label for label, row in selection.items() if not row['same_single_case'])
        print('Paired selection: ' + ('the same single case on both sides for every control' if not different
                                      else 'DIFFERS for ' + ', '.join(different)), flush=True)
        return red or bool(different)
    if step == 'context':
        return tested(step, [sys.executable, 'scripts/context_gate.py', '--workers', '2',
                             '--evidence-dir', str(root / step)])
    if step == 'critical':
        return tested(step, pytest_command(root, step, 465, ['--timeout=45', *[node for _, node, _ in plan['critical']]]))
    if step == 'size':
        if plan.get('selection_profile') == TARGETED:
            modules = sorted({node.split('::')[0] for node in plan['size']})
            return tested(step, pytest_command(root, step, 240, ['-m', 'size_ratchet', *modules], base=plan['base']['commit']))
        return tested(step, pytest_command(root, step, 530, ['-m', 'size_ratchet', 'tests/'], base=plan['base']['commit']))
    if step == 'seams':
        return tested(step, pytest_command(root, step, SEAMS_SECONDS, plan['seams']))
    if step == 'inventory':
        return tested(step, [sys.executable, '-I', '-S', 'scripts/safe_test.py', '--timeout', '120',
                             '--evidence-dir', str(root / step / 'private'), '--', sys.executable, INVENTORY_CHECK, '--check'])
    if step == 'jsunit':
        files = plan['jsunit']['files']
        return tested('settings-js', ['node', '--test', '--test-reporter=tap', *files] if files else nothing,
                      cwd=plan['jsunit']['cwd'])
    if step == 'browser':
        folder = root / 'observation'
        folder.mkdir()
        overlay = (root / 'observation_test.py').read_bytes()
        if hashlib.sha256(overlay).hexdigest() != plan['observation']['sha256']:
            raise ValueError('the staged observation is not the planned file')
        target = plan['observation']['test']
        return tested('browser-observe', pytest_command(root, 'browser-observe', 960, [
            '--timeout=240', '--tb=line', '-m', 'ui_browser', target]),
            env={**os.environ, **BROWSER_ENV, 'OUROBOROS_UI_EVIDENCE_DIR': str(folder)}, overlay=(target, overlay))
    if step == 'existing':
        files = plan['observation']['existing']
        return tested('browser-existing', pytest_command(root, 'browser-existing', 2300, [
            '--timeout=900', '--tb=line', '-m', 'ui_browser', '-k', 'chromium', *files]) if files else nothing,
            env={**os.environ, **BROWSER_ENV})
    focused = plan.get('selection_profile') == 'focused_rerun'
    scope = plan['selected'] if focused else ['tests/', *['--ignore=' + module for module in plan['ignore']]]
    if focused and (not scope or not set(scope) <= known_tests(plan) - {plan['observation']['test']}):
        raise ValueError('focused selection needs explicit frozen test modules')
    exclusions = plan['exclusions']
    if step == 'collection':
        return tested(step, pytest_command(root, step, 180 if focused else 370, ['--collect-only', '-m', exclusions, *scope]))
    if step != 'selected':
        raise ValueError('unknown producer')
    parallel = tested('parallel', pytest_command(root, 'parallel', 720 if focused else 1750, ['-m', 'not exclusive and ' + exclusions,
                  '-n', '2', '--dist', 'loadgroup', '--max-worker-restart=0', '--timeout=300', *scope]))
    exclusive = tested('exclusive', pytest_command(root, 'exclusive', 360 if focused else 620, ['-m', 'exclusive and ' + exclusions,
                    '--timeout=300', *scope]))
    collected, *_ = load_session(root / 'collection')
    if check(root, plan, 'collection')[0]:
        return True
    diff = membership(set(collected['selected_nodeids']), load_session(root / 'parallel')[2],
                      load_session(root / 'exclusive')[2], plan)
    print('Selected coverage: ' + ', '.join(f'{key}={len(value) if isinstance(value, list) else value}'
                                            for key, value in diff.items()), flush=True)
    return parallel or exclusive or not diff['exclusive_passed'] or any(
        diff[k] for k in ('missing', 'unexpected', 'duplicate', 'failed_or_unsettled',
                          'required_without_pass', 'unplanned_modules'))


def safe_attempt(directory, known, runner_exit=None):
    data, state, cases, inflight, journal_state, dropped = load_session(directory)
    collection = [r.get('nodeid') for r in data['collection_failures'] if isinstance(r, dict)] if data else []
    skips = [n for n in data['collection_skips'] if isinstance(n, str)] if data else []
    problem_rows = sorted((node, outcome, phase) for node, (outcome, phase) in cases.items() if outcome != 'passed')
    # Show failures first even in a large skipped/platform-specific suite.
    problem_rows.sort(key=lambda row: (row[1] not in ('failed', 'unknown'), row))
    observed = data is not None or journal_state in ('partial', 'truncated')
    return {'results': state, 'journal': journal_state, 'unreadable_journal_rows': dropped,
            'pytest_exit': data['session_exit_code'] if data else None, 'runner_exit': runner_exit,
            'collected': data['tests_collected'] if data else None,
            'counts': dict(Counter(outcome for outcome, _ in cases.values())) if observed else None,
            'outlier_count': len(problem_rows) if observed else None, 'outliers': [dict(test=test, outcome=row[1], phase=row[2])
                for test, row in zip(names([n for n, *_ in problem_rows], known), problem_rows)],
            'collection_failures': {'count': len(data['collection_failures']) if data is not None else None,
                                    'sample': names([n for n in collection if isinstance(n, str)], known)},
            'collection_skips': {'count': len(data['collection_skips']) if data is not None else None,
                                 'sample': names(skips, known)},
            'in_flight': {'count': len(inflight) if journal_state == 'partial' and dropped == 0 else None,
                          'sample': names(inflight, known)}}


def text(value, limit=400):
    return value.encode('ascii', 'backslashreplace').decode()[:limit] if isinstance(value, str) else None


def measure_facts(measure):
    """The numbers of a measure and the names of what it found clipped (a fixed vocabulary), nothing else."""
    measure = dict_of(measure)
    container, document = dict_of(measure.get('container')), dict_of(measure.get('document'))
    numbers = lambda row, keys: {key: row.get(key) if type(row.get(key)) is int else None for key in keys}  # noqa: E731
    clipped = measure.get('clipped')
    return {'viewport': measure.get('viewport') if isinstance(measure.get('viewport'), list) else None,
            'container': numbers(container, ('scroll_width', 'client_width')),
            'document': numbers(document, ('scroll_width', 'inner_width')),
            'clipped': [text(name, 20) for name in clipped][:8] if isinstance(clipped, list) else None,
            'issues': measure_issues(measure)}


def frame_facts(row):
    return {name: {k: f.get(k) for k in ('file', 'bytes', 'sha256')} for name, f in dict_of(dict_of(row).get('frames')).items()
            if isinstance(f, dict)}


def bounded(value):
    """A value the public report may carry: a bounded ASCII string, a bool, an int, null, or a short list of those."""
    if isinstance(value, list):
        return [None if isinstance(item, list) else bounded(item) for item in value[:8]]
    return text(value) if isinstance(value, str) else value if type(value) in (bool, int) or value is None else None


def project(part, key):
    """One receipt in the report, in the receipt's own shape so that ``judge`` can read it again: plain names with bounded
    values, each measure as numbers, each frame as its file, size and digest."""
    part = dict_of(part)
    rows = []
    for row in part.get(key) if isinstance(part.get(key), list) else []:
        if isinstance(row, dict):
            kept = {name: bounded(value) for name, value in row.items()
                    if name not in ('measure', 'frames') and re.fullmatch(r'[a-z_]{1,40}', name)}
            rows.append({**kept, **({'measure': measure_facts(row['measure'])} if 'measure' in row else {}),
                         **({'frames': frame_facts(row)} if 'frames' in row else {})})
    counters = {name: part[name] for name in ('reserved_host_requests', 'saves_during_test') if type(part.get(name)) is int}
    return {'complete': part.get('complete') is True, **counters, key: rows,
            'checkout': {k: text(dict_of(part.get('checkout')).get(k), 40) for k in ('commit', 'tree')}}


def observation_summary(root, plan):
    """Bounded facts of what the browser showed, shaped like the receipts so that the reader judges them again with ``judge``."""
    folder, got = observation_files(root)
    states = dict_of(got['states'])
    return {'engine': text(states.get('engine'), 20), 'browser_version': text(states.get('browser_version'), 40),
            'parts': {kind: project(got[kind], key) for kind, key in (('states', 'states'), ('lines', 'lines'), ('journey', 'steps'))},
            'failures': failure_rows(root, plan),
            'issues': observation_issues(root, plan) if folder.is_dir() else ['observation_not_run']}


def public_frames(root, plan, summary):
    """The allowlisted screenshots a report names, each verified as its own: ``{name: bytes}``. Anything else in the observation
    folder (a half-written file, a log, a name the plan never declared) is never read into the result."""
    folder, allowed = root / 'observation', allowed_frames(plan)
    wanted = [frame for part in summary['parts'].values() for rows in part.values() if isinstance(rows, list)
              for row in rows for frame in dict_of(dict_of(row).get('frames')).values()]
    wanted += [row['frame'] for row in summary['failures'] if row['frame']]
    kept = {}
    for frame in wanted:
        name = dict_of(frame).get('file')
        if name in allowed and png_ok(folder, frame, name):
            kept[name] = (folder / name).read_bytes()
    return kept


def identity(env):
    """The run, attempt and commit the workflow reports, each only when it has the shape of one."""
    return {key: env.get('GITHUB_' + key.upper()) if re.fullmatch(pattern, env.get('GITHUB_' + key.upper(), '')) else None
            for key, pattern in (('sha', r'[0-9a-f]{40}'), ('run_id', r'[0-9]{1,20}'), ('run_attempt', r'[0-9]{1,20}'))}


def header(scope, env):
    """What every report says first, whatever else it could or could not project: format, scope, run, platform and how each
    workflow step ended."""
    steps = {key: env.get('OUT_' + key.upper()) if env.get('OUT_' + key.upper()) in STEP_STATES else 'unknown'
             for key in ACTIONS[scope]}
    name = platform.system()
    return {'format': FORMAT, 'scope': REPORT_SCOPES[scope],
            **identity(env), 'platform': name if name in ('Linux', 'Darwin', 'Windows') else None, 'steps': steps}


def plan_digest(plan):
    """The plan a report belongs to: a report is read only against the plan it was projected from."""
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def outcome_of(steps, phases):
    """How the run ended, in one word: ``interrupted`` (a step cancelled, a producer stopped by a timeout or a signal),
    ``failed``, ``incomplete`` (a step skipped or never reported, a producer that never finished) or ``complete``. It only names
    the way the run ended; the reader judges whether it was good."""
    if 'cancelled' in steps.values() or any(phase.get('producer_exit') in KILLED for phase in phases.values()):
        return 'interrupted'
    if 'failure' in steps.values() or any(phase['issues'] for phase in phases.values()):
        return 'failed'
    return 'complete' if set(steps.values()) == {'success'} and phases and all(
        phase['producer_exit'] is not None for phase in phases.values()) else 'incomplete'


def report(root, plan, env, scope='main'):
    known = known_tests(plan)
    result = {**header(scope, env), 'plan_sha256': plan_digest(plan), 'phases': {}}
    states = result['steps']
    for label, (role, step) in phase_specs(plan, scope).items():
        directory = root / label
        bound = receipt(root, label, plan[role])
        issues, detail = check(root, plan, label)
        exit_seen = exit_code(bound.get('producer_exit'))
        red = bool(issues) or states[step] != 'success' or exit_seen != 0
        why = bound.get('checkout_why')
        result['phases'][label] = phase = {**plan[role], 'step': states[step],
              'checkout_before': observed_bool(bound.get('checkout_before')),
              'checkout_after': observed_bool(bound.get('checkout_after')), 'producer_exit': exit_seen,
              # How the checkout was found before the phase and whether it had to be bound anew (fixed words and a count).
              'checkout_rebound': bound.get('checkout_rebound') if bound.get('checkout_rebound') in (
                  'not_needed', 'already', 'restored', 'failed') else None,
              'checkout_why': why if why in CHECKOUT_WHY else None,
              'checkout_unowned_entries': bound.get('checkout_unowned_entries')
              if type(bound.get('checkout_unowned_entries')) is int else None,
              # Only the six paired controls ran on the base; any other red was found on the head alone.
              'attribution': PAIRED if label.startswith(('base-', 'head-')) else (UNATTRIBUTED if red else 'not_needed'),
              'issues': sorted(set(issues)), 'witness_gaps': {k: {'count': len(v), 'sample': names(v, known)} for k, v in detail.items()},
              'attempts': {'first': safe_attempt(directory, known)}}
        if label == 'context':
            rounds, retry_state, verdict = context_rounds(directory)
            phase.update(retry_receipt=retry_state, retry_verdict=verdict)
            phase['attempts']['first']['runner_exit'] = rounds.get('.')
            for number in range(1, 11):
                retry = f'retry-{number}'
                if retry in rounds or (directory / retry).is_dir():
                    phase['attempts'][retry] = safe_attempt(directory / retry, known, rounds.get(retry))
        elif label == 'settings-js':
            phase['tap'] = tap_summary(directory / 'console.log')
    result['unattributed'] = [label for label, phase in result['phases'].items() if phase['attribution'] == UNATTRIBUTED]
    if scope != 'main':
        result['observation'] = observation_summary(root, plan)
    else:
        result['selected_coverage'] = 'unknown'
        if not check(root, plan, 'collection')[0]:
            expected = set(load_session(root / 'collection')[0]['selected_nodeids'])
            diff = membership(expected, load_session(root / 'parallel')[2], load_session(root / 'exclusive')[2], plan)
            result['selected_coverage'] = {'expected_nodes': len(expected), 'exclusive_passed': diff.pop('exclusive_passed'),
                    **{key: {'count': len(value), 'sample': names(value, known)} for key, value in diff.items()}}
        result['paired_selection'] = paired_selection(root, plan)
    result['outcome'] = outcome_of(result['steps'], result['phases'])
    return result


def put_report(root, scope, public, frames=None):
    """The one file (and, for the browser, the one folder of allowlisted, verified frames) the upload takes: no receipt, no
    half-written file, no console, no unnamed file of the run leaves."""
    if scope != 'main':
        shelf = root.parent / 'nano-qualification-public'
        (shelf / 'observation').mkdir(parents=True, exist_ok=True)
        for name, data in (frames or {}).items():
            (shelf / 'observation' / name).write_bytes(data)
        put_json(shelf / REPORT_FILES[scope], public)
    else:
        put_json(root.parent / REPORT_FILES[scope], public)


def selected_scope_note(plan):
    if plan.get('selection_profile') == TARGETED:
        return 'Targeted rerun only: seven size checks, the planned seams, the generator check, the planned JS files and three Chromium cases; old runs are not inherited.'
    if plan.get('selection_profile') != 'focused_rerun':
        return None
    narrowed, wider = len(plan['selected']), plan['wider_selection_modules']
    prompt = ('PhysicalAttemptContext prompt-source witnesses are in scope.' if plan['prompt_source_witnesses']
              else 'The frozen head predates PhysicalAttemptContext prompt-source witnesses; they are not covered.')
    return (f'Focused rerun: {narrowed} explicit default-lane modules of {wider} in the wider selection. '
            f'{prompt} This scope does not complete or replace the earlier interrupted run.')


def write_summary(public, plan, scope):
    path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not path:
        return
    with open(path, 'a', encoding='utf-8') as stream:
        print('## Nano Ollama — ' + {'browser': 'Settings in Chromium and unit run', 'main': 'selected check',
                                     'targeted': 'targeted rerun'}[scope]
              + ' (not a release or full battery)', file=stream)
        print(f"Outcome: **{public['outcome']}**. Platform: {public['platform']}; workflow {public['sha']}; "
              f"run {public['run_id']}/{public['run_attempt']}", file=stream)
        print(f"\nBase `{plan['base']['commit']}`; candidate `{plan['head']['commit']}`.", file=stream)
        if note := selected_scope_note(plan):
            print('\n' + note, file=stream)
        print('\nSteps: ' + ', '.join(f'{k}={v}' for k, v in public['steps'].items()) + '.', file=stream)
        print('\n| Phase | Producer exit | Test evidence | Coverage issues | Attribution |', file=stream)
        print('| --- | --- | --- | --- | --- |', file=stream)
        for label, item in public['phases'].items():
            evidence = item['attempts']['first']
            print(f"| {label} | {item['producer_exit']} | {evidence['results']}/{evidence['journal']} | "
                  f"{', '.join(item['issues']) or 'none'} | {item['attribution']} |", file=stream)
        print('\n' + ('No tests ran on the base.' if scope == 'targeted' else 'Only the six paired controls ran on the base.')
              + ' An unattributed red is not told apart from a pre-existing or environmental red.', file=stream)
        if public['unattributed']:
            print('\nUnattributed reds: ' + ', '.join(public['unattributed']) + '.', file=stream)
        if scope != 'main':
            seen = public['observation']
            print(f"\nObserved {len(seen['parts']['states']['states'])} states and {len(seen['parts']['journey']['steps'])} journey "
                  f"steps in {seen['browser_version']}; issues: " + (', '.join(seen['issues']) or 'none'), file=stream)
            for failure in seen['failures']:
                print(f"\nObservation failure: node {failure['node']}, at {failure['state']}, code {failure['code']}.", file=stream)
        else:
            coverage = public['selected_coverage']
            print('\nSelected coverage: ' + (coverage if isinstance(coverage, str) else ', '.join(
                f'{key}={value["count"] if isinstance(value, dict) else value}' for key, value in coverage.items())), file=stream)
            print('\nPaired selection: ' + ', '.join(f"{k}={'same' if v['same_single_case'] else 'DIFFERENT'}"
                                                      for k, v in public['paired_selection'].items()), file=stream)
        print('\nOnly the bounded projection is retained. No exit or missing projection is inferred as a pass.', file=stream)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('critical', 'paired', 'context', 'size', 'collection', 'selected',
                                          'seams', 'inventory', 'jsunit', 'browser', 'existing', 'report'))
    parser.add_argument('--scope', choices=('main', 'browser', 'targeted'), default='main')
    arguments = parser.parse_args()
    stage, scope = arguments.stage, arguments.scope
    try:
        if stage != 'report':
            plan = json.loads((HERE / 'plan.json').read_bytes())
            if plan.get('selection_profile') == TARGETED and stage not in TARGETED_PHASES.values():
                raise ValueError('stage not in the targeted rerun')
            return int(run_step(HERE, plan, stage))
        try:
            plan = json.loads((HERE / 'plan.json').read_bytes())
            if (plan.get('selection_profile') == TARGETED) != (scope == 'targeted'):
                raise ValueError('report scope does not match the plan')
            public = report(HERE, plan, os.environ, scope)
            put_report(HERE, scope, public, public_frames(HERE, plan, public['observation']) if scope != 'main' else None)
        except Exception:
            # A projection that cannot be made still leaves an upload with an explicit outcome, never a missing one.
            print('report: the projection failed; the upload carries only the workflow steps and the outcome report_failed', file=sys.stderr)
            put_report(HERE, scope, {**header(scope, os.environ), 'phases': {}, 'outcome': 'report_failed', 'report_gap': 'projection_failed'})
            return 0
        write_summary(public, plan, scope)
        return 0
    except KeyboardInterrupt:
        print(f'{stage}: interrupted; a missing exit or final test projection remains unknown', file=sys.stderr)
        return 130
    except Exception:
        print(f'{stage}: qualification driver or evidence export failed; no raw diagnostics published', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
