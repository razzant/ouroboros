"""Temporary runner projection: exact subject, small typed outcomes, named synthetic files only."""
import ast
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import re
import shutil
import sys

RAW = Path(os.environ["QA_RAW"])
PRODUCT = Path(os.environ["QA_PRODUCT"])
LANES = (
    'costs-chromium', 'costs-webkit', 'evolution-layout', 'evolution-chromium', 'evolution-webkit',
    'main-a', 'project-find', 'share-chromium', 'share-webkit', 'comparison', 'python', 'node',
    'floor-structure', 'size', 'context',
)
SETUP = ('PRODUCT_STAGE', 'DEPS_SETUP', 'NODE_SETUP', 'BROWSER_SETUP', 'PRODUCT_UI', 'SHARE_DESIGN',
         'PYTHON_STEP', 'NODE_STEP', 'STRUCTURE_STEP')
OID = re.compile(r'[a-f0-9]{40}')
MODULE = re.compile(r'tests/(?:[A-Za-z0-9_]+/)?test_[A-Za-z0-9_]+\.py')
SELECTOR = re.compile(MODULE.pattern + r'(?:::[A-Za-z0-9_]+){0,3}')
TAP_SOURCE = re.compile(r'web/tests/[A-Za-z0-9_.-]+\.test\.js')
EVIDENCE_LIMIT = 1024 * 1024


def read_json(path, max_bytes=16 * 1024 * 1024):
    try:
        if path.is_file() and path.stat().st_size <= max_bytes:
            payload = json.loads(path.read_text(encoding='utf-8'))
            return payload if isinstance(payload, dict) else None
    except (OSError, ValueError):
        pass
    return None


def floor():
    source = ast.parse((PRODUCT / 'ouroboros/test_selection.py').read_text(encoding='utf-8'))
    matches = [item.value for item in source.body if isinstance(item, ast.Assign)
               and any(isinstance(key, ast.Name) and key.id == 'FLOOR_TESTS' for key in item.targets)]
    if len(matches) != 1:
        return 2
    homes = ast.literal_eval(matches[0])
    if not isinstance(homes, dict) or not homes or any(
        not MODULE.fullmatch(path) or not (PRODUCT / path).is_file() for path in homes
    ):
        return 2
    print('\n'.join(sorted(homes)))
    return 0


def instrument(scene, selector):
    """Append only captures inside one known test in the UI-only checkout; never alter assertions."""
    path, delim, function = selector.partition('::')
    if not delim or not MODULE.fullmatch(path):
        return 2
    if scene == 'main':
        anchor = '    assert not transport.posts and not transport.around_reads\n'
        capture = '        capture(page, "main-a-" + task_id)\n'
        before = True
    elif scene == 'evolution':
        anchor = '    expect(note).to_contain_text("Resume the task in Activity")\n'
        capture = '    setup_browser.capture(page, "budget-after-replenish")\n'
        before = False
    else:
        return 2
    target = Path(os.environ['QA_UI']) / path
    text = target.read_text(encoding='utf-8')
    nodes = [node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == function]
    if len(nodes) != 1:
        return 2
    lines = text.splitlines(keepends=True)
    start, end = nodes[0].lineno - 1, nodes[0].end_lineno
    body = ''.join(lines[start:end])
    if body.count(anchor) != 1:
        return 2
    body = body.replace(anchor, capture + anchor if before else anchor + capture, 1)
    result = ''.join(lines[:start]) + body + ''.join(lines[end:])
    ast.parse(result)
    target.write_text(result, encoding='utf-8')
    return 0


def pytest_report(lane, targets):
    facts = read_json(RAW / lane / 'results.json') or {}
    selected, reports = facts.get('selected_nodeids'), facts.get('reports')
    if not isinstance(selected, list) or not all(isinstance(node, str) for node in selected) or not isinstance(reports, list):
        return None, [{'kind': 'missing_or_invalid_test_report'}]
    counts = {'tests': len(selected), 'passed': 0, 'failed': 0, 'skipped': 0, 'fixture_skips': 0, 'xfailed': 0}
    issues, by_node = [], defaultdict(list)
    paths = {target.split('::', 1)[0] for target in targets}

    def note(node, kind):
        path, *parts = node.split('[', 1)[0].split('::')
        detail = {'file': path} if path in paths else {}
        if detail and parts and re.fullmatch(r'test_[A-Za-z0-9_]{1,160}', parts[-1]):
            detail['test'] = parts[-1]
        issues.append({'kind': kind, **detail})

    for row in reports:
        if (not isinstance(row, dict) or not isinstance(row.get('nodeid'), str)
            or row.get('phase') not in {'setup', 'call', 'teardown', 'crash'}
            or row.get('outcome') not in {'passed', 'skipped', 'failed'}):
            return None, [{'kind': 'invalid_report_row'}]
        by_node[row['nodeid']].append(row)
    selection = set(selected)
    if len(selection) != len(selected):
        counts['failed'] += 1
        note('', 'duplicate_selection')
    for node in sorted(selection | by_node.keys()):
        rows = by_node[node]
        phases = {row['phase']: row for row in rows if not row.get('is_subtest')}
        setup, call, teardown = (phases.get(phase, {}) for phase in ('setup', 'call', 'teardown'))
        bad = next((row for row in rows if row['outcome'] == 'failed' or row['phase'] == 'crash'), None)
        if (node not in selection or len(phases) != sum(not row.get('is_subtest') for row in rows)):
            counts['failed'] += 1
            note(node, 'unexpected_or_duplicate_test')
        elif bad:
            counts['failed'] += 1
            note(node, bad['phase'] + '_failure')
        elif setup.get('outcome') == 'skipped' and not call and teardown.get('outcome') == 'passed':
            counts['fixture_skips'] += 1
            note(node, 'fixture_skip')
        elif set(phases) != {'setup', 'call', 'teardown'}:
            counts['failed'] += 1
            note(node, 'missing_test_phase')
        elif (call['outcome'] == 'skipped' and isinstance(call.get('wasxfail'), str)
              and call.get('error_type') != 'XFailed' and setup['outcome'] == teardown['outcome'] == 'passed'
              and all(row['outcome'] == 'passed' for row in rows if row is not call)):
            # Match ci_verdicts.expected_failures: a completed assertion xfail, never setup/run=False or explicit pytest.xfail.
            counts['xfailed'] += 1
            note(node, 'expected_failure')
        elif any(row.get('wasxfail') is not None and row['outcome'] == 'passed' for row in rows):
            counts['failed'] += 1
            note(node, 'unexpected_pass')
        elif any(row['outcome'] == 'skipped' for row in rows):
            counts['skipped'] += 1
            note(node, 'skip_or_explicit_xfail')
        else:
            counts['passed'] += 1
    for collection in ('collection_failures', 'collection_skips'):
        for item in facts.get(collection, []):
            node = item.get('nodeid', '') if isinstance(item, dict) else item
            counts['failed' if collection == 'collection_failures' else 'skipped'] += 1
            note(node if isinstance(node, str) else '', collection)
    for target in targets:
        if not any(node == target or node.startswith((target + '::', target + '[')) for node in selection):
            counts['failed'] += 1
            note(target, 'target_not_executed')
    if lane in {'main-a', 'project-find', 'evolution-layout', 'comparison'}:
        for engine in ('chromium', 'webkit'):
            if not any(re.search(r'\[[^\]]*\b' + engine + r'\b[^\]]*\]$', node) for node in selection):
                counts['failed'] += 1
                note('', f'engine_not_selected_{engine}')
    if facts.get('session_exit_code') != 0:
        counts['failed'] += 1
        note('', 'nonzero_or_unreported_pytest_exit')
    return counts, issues


def node_report(lane, targets):
    path = RAW / f'{lane}.log'
    if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        return None, [{'kind': 'missing_or_oversize_tap'}]
    text = path.read_text(encoding='utf-8', errors='replace')
    keys = {'tests', 'pass', 'fail', 'skipped', 'cancelled', 'todo'}
    rows = re.findall(r'^# (tests|pass|fail|skipped|cancelled|todo) ([0-9]+)[ \t]*$', text, re.M)
    if len(rows) != len(keys) or set(dict(rows)) != keys:
        return None, [{'kind': 'missing_or_ambiguous_tap_footer'}]
    counts = {key: int(n) for key, n in rows}
    if sum(counts[key] for key in keys - {'tests'}) != counts['tests']:
        return None, [{'kind': 'inconsistent_tap_totals'}]
    issues = []
    for case in re.finditer(r'(?m)^\s*not ok \d+ - ([^\n]*)\n[ \t]+---\n(.*?)(?m:^[ \t]+\.\.\.$)', text, re.S):
        title, detail = case.groups()
        row = {'kind': 'test_failure'}
        where = re.search(r'(?m)^\s*location: ([^\n]+)', detail)
        for target in targets:
            at = re.search(r'(?:^|/)' + re.escape(target) + r':([0-9]{1,8})(?=[:\'" ]|$)', where[1]) if where else None
            if at or title.strip() == target:
                row['file'] = target
                if at:
                    row['line'] = int(at[1])
                break
        issues.append(row)
    return counts, list({tuple(row.items()): row for row in issues}.values())


def context_report(lane, _targets):
    value = read_json(RAW / lane / 'verdict.json') or {}
    counts = value.get('counts')
    fields = {'passed', 'flaky', 'failed', 'accepted', 'not_run', 'skipped'}
    if not isinstance(counts, dict) or set(counts) != fields or not all(type(n) is int and n >= 0 for n in counts.values()):
        return None, [{'kind': 'missing_or_invalid_context_verdict'}]
    label = value.get('verdict')
    if label not in {'green', 'flaky', 'accepted', 'red'}:
        return None, [{'kind': 'unknown_context_verdict'}]
    return counts, [{'kind': 'context_gate_' + label}]


def teardown(lane, code):
    parent = RAW / ('context' if lane == 'context' else 'safe-' + lane)
    records = list(parent.glob('run-*/result.json'))
    value = read_json(records[0], 1024 * 1024) if len(records) == 1 else None
    if not value or not isinstance(clean := value.get('cleanup'), dict):
        return {'status': 'unconfirmed', 'scratch': 'unknown'}
    scratch = 'retained' if clean.get('retained') else 'released'
    status = ('containment_failed' if value.get('containment_error') else
              'evidence_incomplete' if any(value.get(k) for k in ('evidence_error', 'evidence_fallback', 'source_gap')) else
              'not_executed' if value.get('execution') != 'executed' else
              'exit_mismatch' if value.get('returncode') != code else 'settled')
    return {'status': status, 'scratch': scratch}


def record(lane, kind, code, targets):
    readers = {'pytest': pytest_report, 'tap': node_report, 'context': context_report}
    if (lane not in LANES or kind not in readers or not all(
        (SELECTOR if kind == 'pytest' else TAP_SOURCE).fullmatch(target) for target in targets
    )):
        return 2
    counts, issues = readers[kind](lane, targets)
    custody = teardown(lane, code)
    strict = lane not in {'python', 'floor-structure', 'context'}
    healthy = False
    observations = False
    if counts:
        if kind == 'tap':
            healthy = counts['pass'] > 0 and not any(counts[k] for k in ('fail', 'cancelled', 'skipped', 'todo'))
        elif kind == 'context':
            healthy = counts['passed'] + counts['flaky'] > 0 and not counts['not_run'] and issues[0]['kind'] in {
                'context_gate_green', 'context_gate_flaky'} and not counts['failed']
            observations = any(counts[k] for k in ('flaky', 'skipped'))
        else:
            observations = any(counts[k] for k in ('skipped', 'fixture_skips', 'xfailed'))
            # Fixture/explicit skips stay incomplete even for the ordinary and structural suite.
            healthy = (counts['passed'] > 0 and not counts['failed'] and not counts['skipped']
                       and not counts['fixture_skips'] and (not strict or not observations))
    if code:
        status = 'failed'
        reason = 'timeout' if code == 124 else 'interrupted' if code >= 128 else 'nonzero_exit'
    elif not healthy or custody['status'] != 'settled':
        status, reason = 'incomplete', 'missing_or_unsettled_evidence'
    else:
        status, reason = ('passed_with_observations' if observations else 'passed'), None
    result = {'status': status, 'reason': reason, 'exit_code': code, 'teardown': custody, 'counts': counts,
              'observations': issues[:32], 'observations_omitted': max(0, len(issues) - 32)}
    (RAW / f'{lane}.json').write_text(json.dumps(result), encoding='utf-8')
    print(f'{lane}: {status}; exit {code}; teardown {custody["status"]}; counts {json.dumps(counts)}')
    return 0 if status.startswith('passed') else 1


def frame_names():
    result = []
    for engine in ('chromium', 'webkit'):
        result += [f'costs/{engine}/hourly-{size}.png' for size in ('1280x900', '390x844')]
        result += [f'evolution/strip-{engine}-{width}-{theme}-evolution-measures.png'
                   for width in (1280, 320) for theme in ('dark', 'light')]
        result += [f'evolution-parked/{engine}/budget-after-replenish.png']
        result += [f'main/room-work/{engine}/main-a-{subject}.png' for subject in ('atlas', 'symbols', 'catalog')]
        result += [f'share/{engine}/{name}' for name in ('share-inline-spaces.png', 'share-words-before-formula.png',
                                                        'answer-points-1100-light.html')]
        result += [f'comparison/compare-{engine}-{width}.png' for width in (900, 390)]
        result += [f'find/{engine}-project-landed-{theme}.png' for theme in ('dark', 'light')]
        result += [f'find/find-sheet-keyboard-{height}-{room}-{engine}.{ext}' for height in (508, 250)
                   for room in ('main', 'project') for ext in ('png', 'json')]
    return result


def publish_frames(output):
    source = RAW / 'visual'
    facts = {}
    for name in frame_names():
        f = source / name
        if not f.is_file():
            facts[name] = 'missing'
        elif f.resolve() != source.resolve() / name or not 0 < f.stat().st_size <= EVIDENCE_LIMIT:
            facts[name] = 'refused_path_or_size'
        else:
            target = output / 'frames' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, target)
            facts[name] = {'bytes': target.stat().st_size}
    return facts


def summarize():
    output = Path(os.environ['QA_OUTPUT'])
    runner = os.environ['GITHUB_SHA']
    if not OID.fullmatch(runner):
        return 2
    identity = RAW / 'product-identity'
    parts = identity.read_text(encoding='ascii').splitlines() if identity.is_file() else []
    product = dict(zip(('commit', 'tree', 'base'), parts)) if len(parts) == 3 and all(OID.fullmatch(p) for p in parts) else None
    stages = {key.lower(): value if (value := os.environ.get('QA_' + key)) in
              {'success', 'failure', 'skipped', 'cancelled'} else 'unknown' for key in SETUP}
    lanes = {lane: read_json(RAW / f'{lane}.json', EVIDENCE_LIMIT) or {'status': 'not_run'} for lane in LANES}
    frame_stages = {}
    for scene in ('main', 'evolution'):
        path = RAW / f'frame-stage-{scene}'
        state = path.read_text(encoding='ascii').strip() if path.is_file() else None
        frame_stages[scene] = state if state in {'instrumented', 'capture_anchor_unavailable'} else 'not_run'
    result = {'scope': 'focused product diagnostic, not full browser/release qualification', 'runner_sha': runner,
              'product': product, 'stages': stages, 'lanes': lanes, 'frames': publish_frames(output),
              'frame_adapters': frame_stages,
              'find_frames': 'Desktop match landing and phone search with emulated visual viewport at 508 and 250 px: synthetic frame and measured rectangles.'}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a', encoding='utf-8') as summary:
        print('## Focused product QA\n\n| Lane | Outcome |\n| --- | --- |', file=summary)
        for lane, observation in lanes.items():
            print(f'| {lane} | {observation["status"]} |', file=summary)
    return 0


if __name__ == '__main__':
    mode, *args = sys.argv[1:]
    if mode == 'floor':
        raise SystemExit(floor())
    if mode == 'instrument':
        raise SystemExit(instrument(*args))
    if mode == 'record':
        lane, kind, code, *targets = args
        raise SystemExit(record(lane, kind, int(code), targets))
    if mode == 'summarize':
        raise SystemExit(summarize())
    raise SystemExit(2)
