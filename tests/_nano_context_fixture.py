"""October owner-shaped producers: synthetic prose, counts and repeated-field sizes only."""
import copy
from dataclasses import replace
from types import SimpleNamespace

from ouroboros.context_sections import _build_installed_skills_section, runtime_host_section
from ouroboros.review_state import AdvisoryReviewState, CommitAttemptRecord, CommitReadinessDebtItem, make_repo_key
from ouroboros.subagent_runtime import review_pool_host_section, subagent_catalog_section
from tests import _memory_scale_fixture as scale


def catalog():
    rows = []
    for group, count, chars in [('opus', 2, 444), ('sonnet', 3, 368), ('fable', 2, 607), ('haiku', 2, 311)]:
        advice = scale.text(f'{group}: authored advice. ', chars)
        for n in range(count):
            rows.append(dict(subagent_id=f'{group}-{n}', route_class='api', effort='high', review_eligible=n == 0,
                             recommended_use=(f'Fallback route: {advice} Keep this qualification.' if n else advice)))
    return {'rows': rows}


def pool():
    return dict(enforcement='advisory', mode='advanced', source='structured', error='',
                pool=[dict(subagent_id=f'{name}-0', seat_id=f'seat-{name}', model=name,
                           cost_hint='subscription', last_review={'status': 'unknown'},
                           delivery='session', detail=scale.text('Recorded pool facts', 800))
                      for name in ('opus', 'sonnet', 'fable', 'haiku')])


def review_state(root):
    key = make_repo_key(root)
    debts = [CommitReadinessDebtItem(f'crd-{n}', 'validation', f'Unresolved debt {n}: verify the retained source.',
             repo_key=key, evidence=[f'Unresolved debt {n}: verify the retained source.']) for n in range(3)]
    runs = [CommitAttemptRecord('2026-09-20T00:00:00Z',
                f'v7.4.0 historical execution {n}: ' + 'detail ' * 90, 'failed',
                snapshot_hash=f'{n:064x}', repo_key=key) for n in range(5)]
    return AdvisoryReviewState(attempts=runs, commit_readiness_debts=debts,
        last_stale_from_edit_ts='2026-10-23T12:00:00Z', last_stale_reason='The checkout changed.', last_stale_repo_key=key)


def runtime_data():
    from ouroboros.gateway.work_status import work_tails
    foreign = [dict(work_id=f'foreign-{n}', title=scale.text(f'Foreign work {n}', 280),
                    room={'chat_id': n, 'project_id': f'project-{n}'},
                    root_status='failed' if n < 10 else 'running', helpers_running=0, helpers_waiting=0,
                    tails=work_tails([], [dict(kind='run', state='unknown' if n < 2 else 'live',
                                              run_id=f'run-{n}-' + 'r' * 64)])) for n in range(12)]
    receipt = {'schema': 1, 'checks': [dict(name=name, status='passed', source='author_declared',
               detail='Execution receipt ' + 'retained evidence ' * 60) for name in ('full_battery', 'cost_contract')],
               'review': {'observed': 'unknown', 'declared': 'PASS'}, 'post_merge': {'status': 'not_run', 'detail': 'Not yet run.'}}
    qualification = dict(base_sha='base', target_sha='target',
        commits=[dict(sha=f'commit-{n}', subject=f'Change {n}', date='2026-10-23', receipt=copy.deepcopy(receipt),
                      body_read=True) for n in range(2)],
        github={'status': 'observed', 'missing_shas': ['commit-absent'],
                'runs': [dict(sha=f'commit-{n}', status='completed', conclusion='failure', url='https://fixture.invalid/run',
                              run_id=n, attempt=1) for n in range(2)]})
    return dict(task_authority_origin={'text': 'Exact authority'}, budget={'remaining_usd': 10, 'limit_usd': 12},
        roots={'system_repo': '/fixture/body', 'active_workspace': '/fixture/work'},
        unknown_observation={'state': 'unknown', 'reason': 'No observation'},
        vitals={'state': 'unknown', 'compact': 'producer-declared; unchanged'},
        live_works_present={'works': [{'task_id': 'own-root', 'state': 'running'}], 'other_live_work': foreign},
        official_update={'status': 'update_available', 'local_changes': 'unknown',
                         'letter': {'text': 'Authored update letter. ' * 80, 'qualification': qualification}})


def skills_section(root, monkeypatch):
    rows = [dict(name=f'skill-{n}', enabled=True, executable_review=True, type='behavior', version='1',
                 review_status='advisory', review_gate={'author_accepted': True},
                 description='Skill capability description.', when_to_use='When requested.') for n in range(8)]
    monkeypatch.setattr('ouroboros.skill_loader.summarize_skills', lambda *_: {'skills': rows})
    return _build_installed_skills_section(SimpleNamespace(drive_root=root), skills=[])


def host_sections(root, monkeypatch):
    from ouroboros.agent_task_pipeline import build_review_context
    monkeypatch.setattr('ouroboros.review_state.load_state', lambda *_: review_state(root))
    monkeypatch.setattr('ouroboros.review_ledger.latest_preflight_record',
                        lambda *_a, **_kw: {'ts': '2026-10-23T11:00:00Z'})
    continuity = build_review_context(SimpleNamespace(drive_root=root, repo_dir=root))
    return (runtime_host_section(runtime_data()), review_pool_host_section(pool()), continuity,
            subagent_catalog_section(catalog()), skills_section(root, monkeypatch))


def core(root, monkeypatch):
    captured = scale.core(root)
    replacements = {p.section_id: p for p in host_sections(root, monkeypatch)}
    dynamic = tuple(replacements.pop(p.section_id, p) for p in captured.dynamic_sections) + tuple(replacements.values())
    return replace(captured, dynamic_sections=dynamic, dynamic_text='\n\n'.join(dynamic))


def tools():
    from ouroboros.tools.compact_context import get_tools
    return [dict(type='function', function=get_tools()[0].schema) if row['function']['name'] == 'compact_context'
            else row for row in scale.tools()]
