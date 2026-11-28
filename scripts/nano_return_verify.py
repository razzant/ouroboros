"""Ephemeral, keyless Ubuntu probe. Only the two unedited cold-return tests are instrumented."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


REF = 'refs/heads/qa/nano-return-verification'
COLD_PATH = 'tests/system_e2e/test_cold_return_window_park.py'
COLD = {
    's47': 'test_s47_a_returned_work_that_no_longer_fits_parks_then_resumes_once_it_fits',
    's50': 'test_s50_the_chosen_overview_outlives_an_ordinary_stop_after_its_grant_was_spent',
}
STAGES = {'s47': {'GROWN': ('update_identity',), 'SHORTENED': ('update_identity',)},
          's50': {'OVERVIEW_HELD': ('knowledge_write',),
                  'OVERVIEW_CURRENT_AND_ROOMY': ('knowledge_write', 'update_identity')}}
SLOTS = {'update_identity': ('identity', 'memory/identity.md', 'text'),
         'knowledge_write': ('overview', 'memory/knowledge/overview.md', 'body')}
TOOLS = frozenset((*SLOTS, 'enable_tools'))
SIZE = ('tests/test_smoke.py::test_no_oversized_modules',
        'tests/test_smoke.py::test_size_ratchet_manifest_matches_live_tree',
        'tests/test_smoke.py::test_size_ratchet_transition_against_explicit_base',
        'tests/test_repo_health_smoke.py::test_checked_in_manifest_generator_check_is_exact')
FILTER = ('tests/test_context_gate.py::test_workflow_filters_cover_the_lane_and_request_builders',
          'tests/test_context_gate.py::test_context_gate_has_recovery_homes')
CHRONICLE = ('tests/test_chronicle_tools.py::test_forged_quote_is_refused_and_an_exact_quote_by_its_speaker_passes',
             'tests/test_nano_activation_boundaries.py::test_normal_mark_and_overview_share_the_same_root_limit',
             'tests/test_nano_knowledge_chronicle_publication.py::test_mark_projection_and_rollback_via_real_tool',
             'tests/test_nano_knowledge_chronicle_publication.py::test_shared_broker_refuses_growth_and_retains_transaction_not_journal')
PHASES = ('setup', 'call', 'teardown')
OUTCOMES = ('passed', 'failed', 'skipped')
CODES = {'OK', 'TOOL_REPORTED_FAILURE', 'TOOL_ARG_ERROR', 'RESOURCE_CONSTRAINT_BLOCKED',
         'EXECUTOR_ERROR', 'LEGACY_WARNING', 'SAFETY_BLOCKED', 'TOOL_NOT_FOUND',
         'TOOL_POLICY_BLOCKED', 'TOOL_UNAVAILABLE', 'TOOL_TIMEOUT', 'TOOL_DENIED'}
STATUSES = {'ok', 'error', 'blocked', 'exception', 'started', 'timeout', 'refused', 'warn', 'warning'}
REFUSALS = ('revision_conflict', 'revision_required', 'history_unavailable', 'replace_failed',
            'over_capacity', 'busy', 'unavailable', 'stale', 'incomplete', 'bootstrap_pending')


def token(value, pattern):
    return value if isinstance(value, str) and re.fullmatch(pattern, value) else None


def digest(value):
    return hashlib.sha256(value).hexdigest()


def enum(value, choices, fallback='unknown'):
    return value if isinstance(value, str) and value in choices else fallback


def snapshot(root, relative, expected, exact):
    # Refuse observation of symlinks and unexpected slots; the ordinary test is unaffected.
    path = root / relative
    answer = {'state': 'unreadable', 'sha256': None, 'expected_match': None}
    if path.resolve() != path or not path.is_relative_to(root):
        return answer
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 1_000_000:
            return answer
        raw = path.read_bytes()
    except FileNotFoundError:
        return {**answer, 'state': 'absent', 'expected_match': False}
    except OSError:
        return answer
    return {'state': 'observed', 'sha256': digest(raw),
            'expected_match': raw == expected if exact else expected in raw}


def refusal_kind(preview):
    for reason in REFUSALS:
        if (preview.startswith(f'{reason}:')
            or preview.startswith(f'⚠️ RESOURCE_CONSTRAINT_BLOCKED: {reason};')
            or preview.startswith(f'⚠️ RESOURCE_CONSTRAINT_BLOCKED: The active text was not changed.\n{reason}:')
            or preview.startswith(f'⚠️ TOOL_ERROR: Knowledge write not completed: {reason}.')
            or preview.startswith(f'⚠️ TOOL_ERROR: {reason} ')):
            return reason if reason in REFUSALS[:4] else f'nano_{reason}'
    return 'blocked_other'


def receipt(row, root, expected_slot, after):
    tool = row['tool']
    status = enum(row.get('status'), STATUSES)
    code = enum(row.get('tool_result_code'), CODES, None)
    meta = row.get('tool_result_meta')
    meta = meta if isinstance(meta, dict) else {}
    preview = row.get('result_preview') or ''
    preview = preview if isinstance(preview, str) else ''
    rev, address, kind = None, 'not_observed', 'unknown'
    if status != 'ok' and (row.get('is_error') is True or status in {'blocked', 'error', 'refused', 'exception', 'timeout'}):
        kind = refusal_kind(preview)
    elif status == 'ok' and row.get('is_error') is False:
        if tool == 'update_identity':
            match = re.match(r'OK: identity (created|updated|unchanged) \([^\n]+\)[^\n]*revision=([0-9a-f]{64})\.', preview, re.I)
            if match:
                kind = 'unchanged' if match[1].lower() == 'unchanged' else 'saved'
                rev = match[2]
        elif tool == 'knowledge_write':
            kind = enum(meta.get('knowledge_write_reason'), {'saved', 'unchanged'})
            source = meta.get('knowledge_source')
            if isinstance(source, dict):
                rev = token(source.get('revision'), r'[0-9a-f]{64}')
                if isinstance(source.get('path'), str) and isinstance(source.get('canonical_root'), str):
                    # Compare only. Never read from the reported destination.
                    address = ('same' if Path(source['path']) == root / expected_slot
                               and Path(source['canonical_root']) == root else 'different')
        elif tool == 'enable_tools':
            kind = 'tool_load_returned_ok'
    return {'kind': kind, 'status': status, 'code': code, 'receipt_revision': rev, 'target_address': address,
            'revision_matches_observed_file': (rev == after['sha256']
                                               if rev and after and after['sha256'] else None)}


class Observer:
    """Wrap only test helpers, always call the original once and propagate its return/error."""

    def __init__(self, lane, owned, proof, pytest):
        self.lane, self.owned, self.proof = lane, owned, proof
        self.context = None

        @pytest.hookimpl(wrapper=True)
        def pytest_runtest_protocol(item):
            return (yield from self.protocol(item))

        self.pytest_runtest_protocol = pytest_runtest_protocol

    def protocol(self, item):
        if item.nodeid != f'{COLD_PATH}::{COLD[self.lane]}':
            return (yield)
        module = item.module
        stub = module._LocalLaneStub
        originals = (module._write_memory, module.submit_running, module.wait_durable_result, stub._answer)
        had_answer = '_answer' in vars(stub)
        original_write, original_submit, original_wait, original_answer = originals

        def answer(instance, body, seq):
            frame = original_answer(instance, body, seq)
            context = self.context
            if context is not None and context['instance'] is instance:
                self.observe(context, self.model_step, module, body, frame)
            return frame

        def submit(server, objective, *args, **kwargs):
            task_id = original_submit(server, objective, *args, **kwargs)
            context = self.context
            if context is not None and objective == 'Rewrite memory as instructed. [' + context['mark'] + ']':
                context['task_id'] = token(task_id, r'[0-9a-f]{8,64}(?:-[0-9a-f]{8,64}){0,4}')
            return task_id

        def wait(oracle, task_id, *args, **kwargs):
            result = original_wait(oracle, task_id, *args, **kwargs)
            context = self.context
            if context is not None and task_id == context['task_id']:
                self.observe(context, self.terminal, result)
            return result

        def write(server, instance, oracle, name, *writes):
            context = {'instance': instance, 'mark': 'COLD_RETURN_MEMORY_' + name,
                       'stage': enum(name, STAGES[self.lane]), 'task_id': None, 'fault': False,
                       'frames': [], 'slots': {}, 'terminal': 'not_observed'}
            self.context = context
            self.observe(context, self.before, server, writes)
            result = 'did_not_return'
            try:
                value = original_write(server, instance, oracle, name, *writes)
                result = 'returned'
                return value
            finally:
                # The actual asserts (and wait) run even when the observer fails. No attempts are repeated.
                self.context = None
                self.observe(context, self.close, oracle, result)

        module._write_memory, module.submit_running, module.wait_durable_result, stub._answer = write, submit, wait, answer
        try:
            return (yield)
        finally:
            module._write_memory, module.submit_running, module.wait_durable_result = originals[:3]
            if had_answer:
                stub._answer = original_answer
            else:
                del stub._answer

    @staticmethod
    def observe(context, action, *args):
        try:
            action(context, *args)
        except Exception:
            context['fault'] = True

    def before(self, context, server, writes):
        root = Path(server.data_root).resolve(strict=True)
        if not root.is_relative_to(self.owned) or tuple(w[0] for w in writes) != STAGES[self.lane].get(context['stage']):
            raise ValueError('Unexpected synthetic profile or stages')
        context['root'] = root
        for tool, _, check in writes:
            name, relative, field = SLOTS[tool]
            expected = inspect.getclosurevars(check).nonlocals[field].encode('utf-8')
            exact = tool == 'update_identity'
            stored = snapshot(root, relative, expected, exact)
            context['slots'][tool] = (name, relative, expected, exact, stored)

    @staticmethod
    def model_step(context, module, body, frame):
        if module._asked(body, (context['mark'],)) != context['mark']:
            return
        if len(context['frames']) >= 32:
            context['fault'] = True
            return
        offered = {tool.get('function', {}).get('name') for tool in body.get('tools') or [] if isinstance(tool, dict)}
        kind, message = frame
        calls = []
        for call in message.get('tool_calls') or []:
            fn = call.get('function') or {}
            name = enum(fn.get('name'), TOOLS, 'other')
            call_id = token(call.get('id'), r'call_[0-9]+')
            args = json.loads(fn.get('arguments') or '{}') if name in TOOLS else {}
            if name == 'knowledge_write' and (args.get('topic'), args.get('scope')) != ('overview', 'global'):
                name = 'other'
            calls.append({'tool': name, 'call_id': call_id,
                          'expected_revision': token(args.get('expected_revision'), r'[0-9a-f]{64}'),
                          'expected_revision_set': 'expected_revision' in args})
        context['frames'].append({'kind': enum(kind, {'agent', 'final', 'finalization'}),
                                  'has_tools': bool(body.get('tools')),
                                  'target_schemas': sorted(TOOLS & offered), 'issued': calls})

    @staticmethod
    def terminal(context, result):
        context['terminal'] = enum(result.get('status'), {'completed', 'failed', 'cancelled', 'scheduled'})

    def close(self, context, oracle, result):
        row = {'stage': context['stage'], 'aux_task_id': context['task_id'], 'ordinary_helper': result,
               'aux_terminal': context['terminal'], 'model_frames': context['frames'], 'slots': {}, 'calls': [],
               'observer_complete': False, 'settlement_read': 'not_observed'}
        self.proof['writes'].append(row)
        try:
            root = context['root']
            observed = {}
            for tool, (name, relative, expected, exact, before) in context['slots'].items():
                after = snapshot(root, relative, expected, exact)
                observed[tool] = (relative, after)
                row['slots'][name] = {'before': before, 'after': after}
            task_id = context['task_id']
            if task_id:
                log = oracle.task_drive(task_id).tools_rows()
                issued = {c['call_id']: c for f in context['frames'] for c in f['issued'] if c['call_id']}
                relevant = [r for r in log if r.get('type') == 'tool_call' and r.get('task_id') == task_id
                            and r.get('tool') in TOOLS]
                if len(relevant) > 32:
                    context['fault'] = True
                row['settlement_read'] = 'settlements_observed' if relevant else 'no_settlements_observed'
                seen = set()
                for entry in relevant[:32]:
                    name = entry['tool']
                    call_id = token(entry.get('tool_call_id'), r'call_[0-9]+')
                    expected_slot, after = observed.get(name, (None, None))
                    fact = receipt(entry, root, expected_slot, after)
                    model_call = issued.get(call_id)
                    bound = bool(model_call and model_call['tool'] == name)
                    if bound:
                        seen.add((call_id, name))
                    row['calls'].append({'tool': name, 'call_id': call_id, 'stub_issued': bound, **fact})
                for call_id, step in issued.items():
                    if (call_id, step['tool']) not in seen:
                        row['calls'].append({'tool': step['tool'], 'call_id': call_id, 'stub_issued': True,
                                             'kind': 'receipt_not_observed'})
            row['observer_complete'] = not context['fault'] and bool(row['slots']) and all(
                s[k]['state'] != 'unreadable' for s in row['slots'].values() for k in ('before', 'after'))
        finally:
            save(self.proof)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(proof):
    with tempfile.NamedTemporaryFile(mode='w', dir=DEST.parent, encoding='utf-8', delete=False) as out:
        json.dump(proof, out, indent=2)
        out.write('\n')
    Path(out.name).replace(DEST)


def qualify(report, wanted, exit_code, github):
    checked = report.get('selected_nodeids')
    selected = (isinstance(checked, list) and len(checked) == len(wanted) and sorted(checked) == sorted(wanted))
    observed = {node: {phase: [] for phase in PHASES} for node in wanted}
    valid = report.get('format') == 1 and report.get('github') == github and (
        type(report.get('session_exit_code')) is int and report['session_exit_code'] == exit_code)
    rows = report.get('reports')
    if not isinstance(rows, list):
        valid, rows = False, []
    for row in rows:
        node, phase, state = (row.get(k) for k in ('nodeid', 'phase', 'outcome'))
        if node not in observed or phase not in PHASES or state not in OUTCOMES or row.get('wasxfail') is not None:
            valid = False
        elif row.get('is_subtest') is True:
            valid &= state == 'passed'
        else:
            observed[node][phase].append(state)
    return {'exit': exit_code, 'exact_collection': selected, 'report_bound': valid,
            'nodes': observed, 'accepted': exit_code == 0 and selected and valid and all(
                facts == ['passed'] for phases in observed.values() for facts in phases.values())}


def main(args):
    global DEST
    dest, overlay, run_id, attempt, ref, lane, base, base_tree, product, product_tree, expected_driver = args
    DEST = Path(dest).resolve()
    github = {'sha': overlay, 'run_id': run_id, 'run_attempt': attempt}
    if (lane not in {'base', 'candidate', *COLD} or ref != REF or attempt != '1' or not sys.platform.startswith('linux')
            or os.environ.get('GITHUB_ACTIONS') != 'true' or os.environ.get('GITHUB_EVENT_NAME') != 'push'
            or os.environ.get('GITHUB_REF') != ref
            or any(os.environ.get('GITHUB_' + key.upper()) != value for key, value in github.items())
            or not all(token(x, r'[0-9a-f]{40}') for x in (overlay, base, base_tree, product, product_tree))
            or not token(run_id, r'[0-9]+') or digest(Path(__file__).read_bytes()) != expected_driver):
        return 3
    owned = Path(os.environ['OUROBOROS_TEST_TEMP_ROOT']).resolve(strict=True)
    if not all(Path(os.environ[k]).resolve().is_relative_to(owned) for k in ('HOME', 'TMPDIR', 'OUROBOROS_DATA_DIR')):
        return 3
    # Prove the only interface is the local synthetic model transport, also after launcher scrubbing.
    net_devices = {row.split(':', 1)[0].strip() for row in Path('/proc/net/dev').read_text().splitlines()[2:]}
    if net_devices != {'lo'} or len(Path('/proc/net/route').read_text().splitlines()) != 1:
        return 3
    subject, tree = (base, base_tree) if lane == 'base' else (product, product_tree)
    git_env = dict(os.environ, GIT_OPTIONAL_LOCKS='0', GIT_NO_LAZY_FETCH='1',
                   GIT_ALLOW_PROTOCOL='file', GIT_NO_REPLACE_OBJECTS='1')

    def git(*argv):
        return subprocess.check_output(['git', *argv], env=git_env, text=True).strip()

    def clean():
        return git('rev-parse', 'HEAD') == subject and git('rev-parse', 'HEAD^{tree}') == tree and not git(
            'status', '--porcelain', '--untracked-files=all')

    if (not clean() or git('show', '-s', '--format=%P', overlay) != product
            or git('show', '-s', '--format=%P', product) != base
            or git('rev-parse', f'{base}^{{tree}}') != base_tree or git('rev-parse', f'{product}^{{tree}}') != product_tree):
        return 3
    # Execute clean source from the working directory rather than the external QA script's directory.
    sys.path.insert(0, str(Path.cwd()))
    import pytest

    proof = {'schema': 1, 'synthetic_only': True, 'lane': lane, 'github': github, 'base': base,
             'base_tree': base_tree, 'product': product, 'product_tree': product_tree, 'subject': subject,
             'subject_tree': tree, 'driver_sha256': expected_driver, 'writes': [], 'checks': {}, 'accepted': False}
    save(proof)
    check_root = owned / 'nano-return-observation'
    check_root.mkdir()
    if lane in COLD:
        os.environ['OUROBOROS_E2E_DEEP'] = 'mock'
        phases = [('cold', (f'{COLD_PATH}::{COLD[lane]}',), ('-m', 'integration'), [Observer(lane, owned, proof, pytest)])]
    else:
        os.environ['OURO_SIZE_RATCHET_BASE_REF'] = base
        os.environ['OURO_BOOK_GROWTH_MODE'] = 'block'
        # Both lanes compare the same stock gate nodes against the same base.
        # Base has no Nano Chronicle tests; their absence is not represented as a passing result.
        phases = [('gates', (*SIZE, *FILTER, *(CHRONICLE if lane == 'candidate' else CHRONICLE[:1])), (), [])]
    for name, nodes, switches, plugins in phases:
        evidence = check_root / name
        evidence.mkdir()
        rc = int(pytest.main(['-o', 'addopts=', '-p', 'no:cacheprovider', '-q', '--tb=no',
                              '--ci-evidence-dir', str(evidence), *switches, *nodes], plugins=plugins))
        try:
            report = read_json(evidence / 'results.json')
        except (OSError, ValueError):
            report = {}
        proof['checks'][name] = qualify(report, nodes, rc, github)
        save(proof)
    if lane in COLD:
        complete = all(row['observer_complete'] for row in proof['writes'])
        observed = [row['stage'] for row in proof['writes']]
        proof['observation_complete'] = complete and observed == list(STAGES[lane]) and all(
            row['aux_terminal'] == 'completed' and row['aux_task_id'] and row['model_frames'] and all(
                row['slots'].get(SLOTS[tool][0], {}).get('after', {}).get('expected_match') is True
                and any(call['tool'] == tool and call['stub_issued'] and call['kind'] in {'saved', 'unchanged'}
                        and call['target_address'] != 'different' and call['revision_matches_observed_file'] is True
                        for call in row['calls']) for tool in STAGES[lane][row['stage']])
            for row in proof['writes'])
    proof['checkout_still_clean'] = clean()
    proof['accepted'] = proof['checkout_still_clean'] and proof.get('observation_complete', True) and all(
        check['accepted'] for check in proof['checks'].values())
    save(proof)
    return 0 if proof['accepted'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
