#!/usr/bin/env python3
"""OB#1234 raw-facts collector (v3). It computes NO verdict, hypothesis or PASS.

Run on the runner host; it reads files only. For every arm it joins the sampler's process
records, the driver's facts and the product's own tools.jsonl into one ``facts.json``. An
input it cannot read is named in ``input_errors`` and never becomes an empty observation.
The parent judges from it. Package ownership of mapped paths is not attributed here.

Exit statuses: none are witnessed. The only exit information is whatever
DEBUG=pw:browser printed for the MAIN browser PID (raw lines kept in pw-browser.txt).
No "no SIGTRAP" statement can be derived from these files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REFERENCE_ROLES = ('browser', 'zygote', 'renderer', 'gpu-process', 'utility:network.mojom.NetworkService')
EXIT_STATEMENT = ('Child exits UNWITNESSED. Only DEBUG=pw:browser output for the main browser PID exists '
                  '(pw-browser.txt); no statement about SIGTRAP in any child process is possible.')
BROWSER_NAMES = ('chrome', 'chrome-headless-shell', 'headless_shell')
LOG_COPIES = ('logs-at-cleanup', 'logs', 'logs-before-stop')  # newest copy first
PROCESS_KEYS = ('chromium_executables_from_run', 'chains_from_chromium_browser', 'chromium_roles_observed',
                'reference_roles_not_observed', 'chromium_processes_zero_or_failed_maps_reads',
                'processes_with_unreadable_environ', 'loader_env_by_process', 'chromium_processes')


def load(path: Path, default, errors: dict):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        errors[path.name] = repr(exc)
        return default


def role(rec):
    # Tokens, because a rewritten cmdline may be one space-joined string. An unreadable or
    # flagless cmdline of anything but a browser binary stays unknown.
    tokens = ' '.join(rec.get('argv') or []).split()
    value = lambda flag: next((t.split('=', 1)[1] for t in tokens if t.startswith(flag + '=')), '')
    kind = value('--type')
    if kind == 'utility':
        return 'utility:' + (value('--utility-sub-type') or '?')
    if kind:
        return kind
    base = (rec.get('exe') or '').rsplit('/', 1)[-1]
    return 'browser' if tokens and base in BROWSER_NAMES else f'unknown:{base or "?"}'


def label(rec, launcher_ns_pid):
    exe, argv = rec.get('exe') or '', rec.get('argv') or []
    base = exe.rsplit('/', 1)[-1]
    if rec.get('chromium'):
        return 'chromium:' + role(rec)
    if base == 'node' and any('run-driver' in a for a in argv):
        return 'node-driver'
    if base.startswith('python'):
        if any(a.endswith('/server.py') for a in argv):
            return 'server'
        if any(a.endswith('/launcher.py') for a in argv):
            return 'launcher(source)'
        return 'python(role unknown)'
    if str(rec.get('ns_pid')) == str(launcher_ns_pid):
        return 'launcher'
    return base or rec.get('comm', '?')


def path_class(path, payload, chromium_dir):
    if chromium_dir and path.startswith(chromium_dir + '/'):
        return 'chromium_dir'
    if '/_internal/linux-gtk/' in path:
        return 'payload_linux_gtk'
    if path.startswith(payload + '/_internal/'):
        return 'payload_internal'
    if path.startswith(payload + '/'):
        return 'payload_other'
    return 'outside_payload'


def read_lines(path: Path, errors: dict):
    try:
        return path.read_text(errors='replace').splitlines()
    except OSError as exc:
        errors[path.name] = repr(exc)
        return None


def load_jsonl(path: Path, errors: dict) -> list:
    rows, bad = [], 0
    for line in read_lines(path, errors) or []:
        try:
            row = json.loads(line)
        except ValueError:
            row = None
        if isinstance(row, dict):
            rows.append(row)
        else:
            bad += 1
    if bad:
        errors[f'{path.name}:bad_lines'] = bad
    return rows


def chosen_log(arm_dir: Path, name: str, errors: dict):
    """The one copy of a product log used for counts: the largest, the newest on a tie. All copies stay on disk."""
    sizes = {}
    for copy in LOG_COPIES:
        try:
            sizes[copy] = (arm_dir / copy / name).stat().st_size
        except OSError:
            pass
    best = max(sizes, key=sizes.get, default=None)
    if best is None:
        errors[name] = 'no copy present'
    return (arm_dir / best / name if best else None), {'chosen': best, 'sizes': sizes}


def tool_rows(rows: list, task_id: str) -> list:
    # Result rows (#1316 reader): type "tool_call" or a legacy row without type.
    keep = ('ts', 'tool', 'tool_call_id', 'task_id', 'is_error', 'status', 'args', 'result_preview')
    return [{k: row.get(k) for k in keep} for row in rows if str(row.get('type') or 'tool_call') == 'tool_call'
            and row.get('tool') in ('browse_page', 'browser_action') and (not task_id or row.get('task_id') == task_id)]


def arm_facts(arm_dir: Path, payload: str):
    errors = {}  # every unreadable input by name: absent or corrupt is never an empty observation
    driver = load(arm_dir / 'driver.json', {}, errors)
    steps = {s['step']: s for s in driver.get('steps', [])}
    loaded = load(arm_dir / 'sampler' / 'processes.json', None, errors)
    if loaded is not None and not isinstance(loaded, list):
        errors['processes.json'] = 'expected a list of process records'
        loaded = None
    records = loaded or []
    meta = load(arm_dir / 'sampler' / 'sampler-meta.json', {}, errors)
    launcher_ns = (steps.get('launcher_started') or {}).get('container_pid')
    by_key = {}
    for rec in records:
        rec['label'] = label(rec, launcher_ns)
        by_key[rec['key']] = rec
    chromium = [r for r in records if r.get('chromium')]
    chromium_dir = chromium[0]['exe'].rsplit('/', 1)[0] if chromium else ''

    def chain(rec):
        # A link needs the parent's PID and the birth captured at identification; otherwise it ends as unknown.
        out, seen = [], set()
        while rec['key'] not in seen:
            seen.add(rec['key'])
            out.append({k: rec.get(k) for k in ('label', 'key', 'ns_pid', 'ppid', 'parent_starttime', 'exe')})
            born = rec.get('parent_starttime')
            parent = by_key.get(f"{rec.get('ppid')}:{born}") if born is not None else None
            if parent is None:
                out.append({'parent_not_linked': {'ppid': rec.get('ppid'), 'parent_starttime': born}})
                break
            rec = parent
        return out

    chromium_out = []
    for r in chromium:
        maps = [{'path': m['path'], 'first_seen': m['first_seen'],
                 'class': path_class(m['path'].replace(' (deleted)', ''), payload, chromium_dir)}
                for m in r.get('maps', [])]
        chromium_out.append({'label': r['label'], 'host_pid': r['host_pid'], 'starttime': r['starttime'],
                             'ppid': r['ppid'], 'parent_starttime': r.get('parent_starttime'), 'exe': r['exe'],
                             'argv': r['argv'], 'argv_history': r.get('argv_history'),
                             'pre_exec_images': r.get('images'), 'environ_readable': r.get('environ_readable'),
                             'env_values': r.get('env_values'), 'maps_reads': r.get('maps_reads'),
                             'maps_attempts': r.get('maps_attempts'), 'maps_errors': r.get('maps_errors'),
                             'read_errors': r.get('read_errors'), 'exe_reread_failures': r.get('exe_reread_failures'),
                             'first_seen': r.get('first_seen'), 'gone_seen_at': r.get('gone_seen_at'), 'maps': maps})
    observed_roles = sorted({role(r) for r in chromium})
    agent_path, agent_source = chosen_log(arm_dir, 'agent_stdout.log', errors)
    tools_path, tools_source = chosen_log(arm_dir, 'tools.jsonl', errors)
    # Two different streams: counted apart, never summed. Agent-log counts come from one chosen copy.
    streams = {name: lines for name, path in (('agent_stdout', agent_path), ('launcher_stdout', arm_dir / 'launcher-stdout.log'))
               if path and (lines := read_lines(path, errors)) is not None}
    count = lambda text: {name: sum(text in line for line in lines) for name, lines in streams.items()}
    (arm_dir / 'pw-browser.txt').write_text(''.join(f'{name}\t{line}\n' for name, lines in streams.items()
                                                    for line in lines if 'pw:browser' in line))
    task_id = (steps.get('task_submitted') or {}).get('task_id') or ''
    rows = tool_rows(load_jsonl(tools_path, errors) if tools_path else [], task_id)
    provider = load_jsonl(arm_dir / 'provider.jsonl', errors)
    console = read_lines(arm_dir / 'driver-console.log', errors) or []
    cleanup = arm_dir / 'cleanup-errors.txt'
    result = {
        'status': 'ATTEMPTED', 'input_errors': errors,
        'log_sources': {'agent_stdout.log': agent_source, 'tools.jsonl': tools_source},
        'exit_statement': EXIT_STATEMENT,
        'payload': driver.get('payload'), 'health': (steps.get('health') or {}).get('health'),
        'managed_repo': steps.get('managed_repo'), 'saved_settings': (steps.get('saved_settings') or {}).get('values'),
        'package_owner_attribution': 'NOT_PERFORMED',
        'chromium_runtime_version': 'UNKNOWN unless a raw line in pw-browser.txt or product_browser_tool_rows reports it; the probe runs no browser itself',
        'chromium_release_metadata': 'sources.lock.json sbom_components: release metadata, not a runtime observation',
        'chromium_executables_from_run': sorted({r['exe'] for r in chromium}),
        'admission': {k: steps.get(k) for k in ('admission_state', 'task_submit_outcome_unknown', 'task_submitted',
                                                'task_detail', 'state_at_task_detail')},
        'chains_from_chromium_browser': [chain(r) for r in chromium if role(r) == 'browser'],
        'chromium_roles_observed': observed_roles,
        'reference_roles_not_observed': [x for x in REFERENCE_ROLES if x not in observed_roles],
        'site_isolation_note': 'product passes --disable-features=site-per-process; a cross-site iframe request '
                               'does not imply a separate renderer',
        'chromium_processes_zero_or_failed_maps_reads': [
            {'host_pid': r['host_pid'], 'label': r['label'], 'maps_reads': r.get('maps_reads'),
             'maps_errors': r.get('maps_errors')} for r in chromium if not r.get('maps_reads') or r.get('maps_errors')],
        'processes_with_unreadable_environ': [
            {'host_pid': r['host_pid'], 'label': r['label'], 'errno': r.get('environ_errno')}
            for r in records if r.get('environ_readable') is False],
        'loader_env_by_process': [
            {'label': r['label'], 'host_pid': r['host_pid'], 'environ_readable': r.get('environ_readable'),
             'env_values': r.get('env_values')}
            for r in records if r['label'] in ('launcher', 'launcher(source)', 'server', 'python(role unknown)', 'node-driver')
            or r['label'] == 'chromium:browser'],
        'chromium_processes': chromium_out,
        'bus_reach_by_probe_client': driver.get('bus_reach_before_launch'),
        'pw_browser_lines_by_stream': count('pw:browser'),
        'failed_to_connect_to_the_bus_lines_by_stream': count('Failed to connect to the bus'),
        'provider': {'requests': len(provider), 'intended': sum(1 for p in provider if p.get('intended')),
                     'neutral': sum(1 for p in provider if p.get('answer') == 'neutral'),
                     'safety_json': sum(1 for p in provider if p.get('safety_json')),
                     'issued_and_final': next((steps[k] for k in ('provider_closed', 'issued_calls')
                                               if 'issued' in steps.get(k, {})), None)},
        'product_browser_tool_rows': {'source': tools_source['chosen'], 'task_id_filter': task_id or None, 'rows': rows},
        'fixture': {'hits': load(arm_dir / 'fixture-hits.json', [], errors), 'beacons': load(arm_dir / 'beacons.json', [], errors)},
        'driver_errors': [s for s in driver.get('steps', []) if s['step'] == 'error' or 'error' in s],
        'driver_console_tail': console[-5:],
        'cleanup_errors': cleanup.read_text().splitlines() if cleanup.exists() else [],
        'sampler': {k: meta.get(k) for k in ('stop_reason', 'records', 'polls', 'still_live_at_stop', 'exit_status', 'max_poll_gap_s',
                                             'list_errors', 'stat_unread', 'unclassified', 'other_ns_processes', 'limits')},
    }
    if loaded is None:  # no sampler snapshot: process facts are unknown, not empty
        result.update(dict.fromkeys(PROCESS_KEYS))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--container-arms-root', default='/opt/ob1234')
    args = ap.parse_args()
    facts = {'schema': 'ob1234-chain-probe/facts/3', 'verdicts': 'none by design; the parent judges', 'arms': {},
             'disk_stops': {p.name: load(p, None, {}) for p in sorted((args.out / 'facts').glob('disk-stop-*.json'))}}
    payload_of = {'A': 'A', 'B': 'B', 'C': 'A'}
    for arm, p in payload_of.items():
        if (args.out / arm).is_dir():
            facts['arms'][arm] = arm_facts(args.out / arm, f'{args.container_arms_root}/{p}/Ouroboros')
        elif arm != 'C':  # a lane that did not run is named, never just absent
            facts['arms'][arm] = {'status': 'NOT_ATTEMPTED'}
    (args.out / 'facts.json').write_text(json.dumps(facts, indent=1))
    print(json.dumps({arm: {'status': v['status'], 'roles': v.get('chromium_roles_observed'),
                            'tool_rows': len(v.get('product_browser_tool_rows', {}).get('rows', [])),
                            'driver_errors': len(v.get('driver_errors', []))} for arm, v in facts['arms'].items()}))


if __name__ == '__main__':
    main()
