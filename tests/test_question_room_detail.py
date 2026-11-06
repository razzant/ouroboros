"""The real task-detail producer feeds the shared browser question reader."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros.gateway.tasks import api_task_get
from ouroboros.owner_quiz import record_asked, record_publication
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result


def question_world(root):
    a, b = (create_project(root, name, name=name) for name in ('context-a', 'near-b'))
    rows = {'context-main': {'chat_id': 1, 'project_id': a['id']},
            'hidden-main': {'chat_id': 0}, 'plain-main': {'chat_id': 1},
            'project-local': {'chat_id': a['chat_id'], 'project_id': a['id']},
            'converted': {'chat_id': 1, 'project_id': a['id'], 'parent_task_id': 'mid', 'root_task_id': 'root'},
            'unpublished-hidden': {'chat_id': 0}, 'missing-address': {},
            'missing-parent': {'chat_id': 1, 'parent_task_id': 'absent'},
            'foreign': {'chat_id': 17}}
    for tid in ('root', 'mid'):
        write_task_result(root, tid, 'completed', chat_id=1, parent_task_id='root' if tid == 'mid' else '')
    bind_task_to_project(root, 'root', a['id'], origin={'absent': 'system'})
    bind_task_to_project(root, 'mid', b['id'], origin={'absent': 'system'})
    for tid, fields in rows.items():
        write_task_result(root, tid, 'completed', result='Retained question', **fields)
        record_asked(root, tid, quiz_id='q', question='Which source?', options=['Notes', 'Replication'])
        if tid not in {'unpublished-hidden', 'missing-address', 'missing-parent'}:
            record_publication(root, tid, 'q', a['chat_id'] if tid == 'project-local' else 1)
    return a, b


def detail_client(root):
    app = Starlette(routes=[Route('/api/tasks/{task_id}', api_task_get)])
    app.state.drive_root = root
    return TestClient(app)


@pytest.mark.parametrize('task_id', ['context-main', 'hidden-main', 'missing-parent'])
def test_actual_task_detail_recovers_shared_question_form(tmp_path, task_id):
    a, b = question_world(tmp_path)
    detail = detail_client(tmp_path).get('/api/tasks/' + task_id).json()
    project = b['id'] if task_id == 'converted' else a['id'] if task_id == 'project-local' else ''
    run = subprocess.run([shutil.which('node'), 'web/tests/question_room_detail_consumer.mjs'],
                         input=json.dumps({'detail': detail, 'project': project}), text=True,
                         capture_output=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
    assert run.returncode == 0, run.stdout + run.stderr
    if task_id != 'missing-parent':
        assert detail['owner_quiz']['q']['publication_chat_id'] == (a['chat_id'] if task_id == 'project-local' else 1)


@pytest.mark.parametrize('task_id', ['unpublished-hidden', 'missing-address', 'foreign'])
def test_unknown_or_foreign_detail_cannot_authorize_main(tmp_path, task_id):
    question_world(tmp_path)
    detail = detail_client(tmp_path).get('/api/tasks/' + task_id).json()
    run = subprocess.run([shutil.which('node'), 'web/tests/question_room_detail_consumer.mjs'],
                         input=json.dumps({'detail': detail, 'project': '', 'refuse': True}), text=True,
                         capture_output=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
    assert run.returncode == 0, run.stdout + run.stderr


def test_unreadable_room_authority_can_recover_within_same_epoch(tmp_path, monkeypatch):
    from ouroboros import projects_registry as pr
    question_world(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(pr, '_load_bindings', lambda *a, **k: (_ for _ in ()).throw(ValueError('unreadable')))
        unknown = detail_client(tmp_path).get('/api/tasks/context-main').json()
    known = detail_client(tmp_path).get('/api/tasks/context-main').json()
    run = subprocess.run([shutil.which('node'), 'web/tests/question_room_detail_consumer.mjs'],
                         input=json.dumps({'detail': known, 'unknown': unknown, 'project': ''}), text=True,
                         capture_output=True, timeout=15, cwd=Path(__file__).resolve().parents[1])
    assert run.returncode == 0, run.stdout + run.stderr
