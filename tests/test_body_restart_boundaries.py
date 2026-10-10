"""Cold-entry replacement, adoption identity and current shutdown evidence."""
import pathlib
import shutil

import pytest

from ouroboros import body_adoption, body_candidate, owned_shutdown
from tests.body_candidate_support import candidate_commit, git, isolate, make_ctx, make_serving, run_entry
from tests.test_body_adoption_repairs import _armed

pytestmark = pytest.mark.serial


@pytest.mark.parametrize('kind', ['file', 'symlink'])
@pytest.mark.parametrize('foreign', [False, True])
def test_cold_switch_replaces_directory_with_historical_cache_only(tmp_path, monkeypatch, kind, foreign):
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    candidate_commit(serving, files={'pkg/current.py': 'CURRENT = 1\n'})
    ctx = make_ctx(serving, data, 'directory')
    body_candidate.prepare(ctx)
    candidate = pathlib.Path(ctx.repo_dir)
    shutil.rmtree(candidate / 'pkg')
    if kind == 'symlink':
        (candidate / 'pkg').symlink_to('ouroboros', target_is_directory=True)
    else:
        (candidate / 'pkg').write_text('now a file\n')
    git(candidate, 'add', '-A')
    git(candidate, 'commit', '-qm', 'replace directory')
    sha = git(candidate, 'rev-parse', 'HEAD')
    body_candidate.record_reviewed_commit(ctx, sha)
    _armed(serving, data, ctx, sha)
    cache = serving / 'pkg/__pycache__'
    cache.mkdir()
    (cache / 'removed_in_earlier_release.cpython-310.pyc').write_bytes(b'historical')
    if foreign:
        (cache / 'notes.txt').write_text('foreign work')
    result = run_entry(serving)
    assert result.returncode == 0, result.stderr
    if foreign:
        assert (cache / 'notes.txt').read_text() == 'foreign work'
        assert (serving / 'pkg/current.py').exists()
        assert body_adoption.read(data)['phase'] == 'abandoned'
    else:
        assert git(serving, 'rev-parse', 'HEAD') == sha
        assert body_adoption.read(data)['phase'] == 'switched'
        assert (serving / 'pkg').is_symlink() if kind == 'symlink' else (serving / 'pkg').read_text() == 'now a file\n'
        again = run_entry(serving)
        assert again.returncode == 0, again.stderr


def test_evolution_cleanup_only_abandons_its_own_authorization(tmp_path, monkeypatch):
    from supervisor import evolution_lifecycle
    serving = make_serving(tmp_path)
    data = isolate(monkeypatch, tmp_path)
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'pro')
    monkeypatch.setattr(evolution_lifecycle, '_evolution_campaign_path', lambda: data / 'state/evolution_campaign.json')
    root = make_ctx(serving, data, 'root')
    evolution = make_ctx(serving, data, 'evolution')
    body_candidate.prepare(root)
    body_candidate.prepare(evolution)
    sha = candidate_commit(root.repo_dir)
    body_candidate.record_reviewed_commit(root, sha)
    handoff = body_adoption.authorize(root, sha, reason='root restart')
    tx = {'task_id': 'evolution', 'base_head': git(serving, 'rev-parse', 'HEAD')}
    evolution_lifecycle._cleanup_worktree_after_cycle(tx, 'evolution')
    assert body_adoption.read(data) == handoff
    assert tx['cleanup_status'] == 'candidate_retained'
    evolution_lifecycle._cleanup_worktree_after_cycle(dict(tx, task_id='root'), 'root')
    assert body_adoption.read(data) == {}


def test_cached_stop_after_late_registration_cannot_arm_adoption(tmp_path, monkeypatch):
    from ouroboros import workspace_executor
    from tests.test_owned_shutdown import _sleeper, _reap
    from tests.test_body_adoption import EXITED, Scene
    scene = Scene(tmp_path, monkeypatch)
    scene.commit()
    scene.authorize()
    initial = owned_shutdown.stop_owned_work(scene.data)
    assert initial['state'] == 'completed' and initial['targets'] == 0
    late = _sleeper()
    try:
        workspace_executor._register_process(scene.data, {'record_type': 'foreground',
            'executor_type': 'local', 'executor_id': 'host', 'host_pid': late.pid})
        current = owned_shutdown.stop_owned_work(scene.data)
        assert current['targets'] == 0  # the one stop is joined, never restarted
        assert current['state'] == 'unconfirmed' and current['unconfirmed']
        assert late.poll() is None
        import server
        import multiprocessing
        import threading
        from supervisor import workers
        assert body_adoption.bind_restart(lambda **kw: (True, 'ok'), scene.data, 'adopt it')()[0]
        restart = threading.Event()
        restart.set()
        monkeypatch.setattr(server, 'DATA_DIR', scene.data)
        monkeypatch.setattr(server, '_restart_requested', restart)
        monkeypatch.setattr(server, '_owner_restart_requested', threading.Event())
        monkeypatch.setattr(server, '_stop_owned_daemon_for_new_pin', lambda: None)
        monkeypatch.setattr(workers, 'kill_workers', lambda **kw: None)
        monkeypatch.setattr(workers, 'last_worker_exit_census', lambda: EXITED)
        monkeypatch.setattr(multiprocessing, 'active_children', lambda: [])
        server._emergency_process_cleanup(port_sweep=False)
        assert body_adoption.read(scene.data)['phase'] == 'authorized'
        assert not scene.pointer().exists()
        assert owned_shutdown.finish_unconfirmed_stops(scene.data)['confirmed'] == 1
    finally:
        _reap(late)


def test_crash_recovery_has_actionable_status_without_automatic_adoption(tmp_path, monkeypatch):
    from tests.test_body_adoption_consumers import _crash_recovered_cycle
    from supervisor import queue, state
    scene = _crash_recovered_cycle(tmp_path, monkeypatch, 'evo-crash')
    monkeypatch.setattr(state, 'control_value', lambda *args: (True, True))
    snapshot = queue.get_evolution_status_snapshot()
    assert snapshot['status'] == 'waiting_for_restart_verify'
    assert 'prepare_self_change' in snapshot['detail'] and 'request_restart' in snapshot['detail']
    assert scene.commit_sha in snapshot['detail'] and 'evo-crash' in snapshot['detail']
    assert body_adoption.read(scene.data) == {}
    assert not (scene.data / 'state/pending_restart_verify.json').exists()
