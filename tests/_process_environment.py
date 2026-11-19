"""Shared isolated registry and process environment for process-tool consumers."""

import pytest

from ouroboros import mcp_client
from ouroboros.tools import services
from ouroboros.tools.registry import ToolContext, ToolRegistry


@pytest.fixture
def process_context(tmp_path, monkeypatch):
    system, workspace, data = (tmp_path / name for name in ('system', 'workspace', 'data'))
    for path in (system, workspace, data):
        path.mkdir()
    monkeypatch.setenv('OUROBOROS_RUNTIME_MODE', 'advanced')
    monkeypatch.setattr('ouroboros.safety.check_safety', lambda *args, **kwargs: (True, ''))
    context = ToolContext(repo_dir=system, drive_root=data, workspace_root=workspace,
                          workspace_mode='external', task_id='process-env')
    registry = ToolRegistry(system, data)
    registry.set_context(context)
    yield registry, context, workspace, data
    services.kill_all_services(data)
    mcp_client.reset_manager_for_tests()
