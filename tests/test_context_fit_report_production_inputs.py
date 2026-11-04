"""The offline report measures the inputs production selects: its tools, caller kind, reserve and room folder."""
import json
import re

import pytest

from devtools.context_fit_report import Unmeasurable, main, report
from tests import _memory_scale_fixture as scale
from tests.test_context_fit_report import fingerprint, offline_fixture

WINDOW, REPLY_CEILING = 250_000, 65_536
# Low cannot start beside Main's 65,536 reply in 150K on this population, Nano can; 40K refuses even Nano's fixed input.
SMALL_WINDOW, REFUSING_WINDOW = 150_000, 40_000
CALL_WINDOW, CALL_RESERVE = 92_000, 10_000


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    """One synthetic install: the scale population plus a room with a registered folder (and its folder)."""
    from ouroboros.projects_registry import create_project

    base = tmp_path_factory.mktemp('fit-source')
    root = scale.write_root(base / 'data')
    folder = base / 'room-folder'
    folder.mkdir()
    chat = create_project(root, 'fit-folder', name='Folder room', working_dir=str(folder))['chat_id']
    offline_fixture(root)
    return root, folder.resolve(), chat


def cli(capsys, root, *args, room=1, mode='low', window=WINDOW, reserve=REPLY_CEILING):
    code = main(['--data', str(root), '--room', str(room), '--window', str(window), '--reserve', str(reserve),
                 '--mode', mode, *args])
    output = capsys.readouterr()
    return code, output.out, output.err


def test_cli_default_is_a_direct_turn_with_the_nano_floor_and_discloses_its_tool_source(source, capsys):
    from ouroboros import context_budget
    from ouroboros.memory_view import start_room, view_role

    root, _folder, _chat = source
    before = fingerprint(root)
    code, out, _err = cli(capsys, root, '--json', mode='nano')
    result = json.loads(out)
    assert code == 0 and fingerprint(root) == before
    # Production's initial Nano reserve is the reply floor, not the route's whole ceiling.
    assert result['mode'] == 'nano' and result['reserve'] == context_budget.NANO_MIN_HEADROOM_TOKENS
    assert result['reserve'] == context_budget.context_mode_limits('nano', 'nano', REPLY_CEILING)[1] < REPLY_CEILING
    assert result['arithmetic']['room_after_reserve_tokens'] == WINDOW - result['reserve'] - result['input_tokens']
    task = {'_is_direct_chat': True}
    assert result['caller']['kind'] == 'direct' and result['caller']['role'] == view_role(task) == 'integrator'
    assert result['work_room_levels'] == start_room(task, 'integrator') == 1
    tools = result['tools']
    assert tools['selector'].startswith('ouroboros.tool_policy.initial_tool_schemas(context_mode=')
    # A standalone registry is never presented as the live catalog, and its error has no assumed direction.
    assert tools['verified_complete'] is False and len(tools['absent']) == 2
    assert sum(tools['by_surface'].values()) == tools['count'] and tools['by_surface']['builtin'] > 0
    notes = ' '.join(result['diagnostic_notes'])
    assert 'smaller' in notes and 'larger' in notes and 'never provider tokens' in notes
    assert result['estimator'].startswith('estimate_context_prompt_tokens')
    # Production hands its registry to the capture: Nano's request names every callable tool by namespace.
    connections = next(row for row in result['blocks'] if row['block'] == 'connections')
    assert connections['form'] == 'by namespace' and connections['tokens'] > 0


def test_cli_root_caller_keeps_the_two_start_work_rooms_production_gives_a_queued_root(source, capsys, monkeypatch):
    from ouroboros import context_budget, context_fit
    from ouroboros.memory_view import start_room, view_role

    root, _folder, _chat = source
    planned = []
    plan = context_fit.build_context_fit_plan
    monkeypatch.setattr(context_fit, 'build_context_fit_plan',
                        lambda env, core, task, **kw: planned.append(task) or plan(env, core, task, **kw))
    before = fingerprint(root)
    code, out, _err = cli(capsys, root, '--json', '--caller', 'root')
    result = json.loads(out)
    code_direct, out_direct, _err = cli(capsys, root, '--json')
    direct = json.loads(out_direct)
    assert code == code_direct == 0 and fingerprint(root) == before
    task = {'_is_direct_chat': False}
    levels = start_room(task, view_role(task))
    assert levels == context_budget.start_room_levels('integrator', direct_chat=False) == 2
    assert result['caller'] == {'kind': 'root', 'role': 'integrator', 'task_fields': ['_is_direct_chat']}
    assert result['work_room_levels'] == levels and direct['work_room_levels'] == 1
    assert result['start_work_room_tokens'] > direct['start_work_room_tokens']
    # The plan reads the very task the core was captured for, not a direct turn made up by the measurer.
    assert [row['_is_direct_chat'] for row in planned] == [False, True]


def test_a_live_call_is_a_direct_caller_and_unknown_caller_kinds_are_not_guessed(source, capsys):
    root, _folder, _chat = source
    for args, message in ((('--caller', 'root', '--session-provider', 'gemini_live'), 'direct caller'),
                          (('--caller', 'wake'), 'invalid choice')):
        with pytest.raises(SystemExit) as stop:
            cli(capsys, root, *args)
        assert stop.value.code == 2 and message in capsys.readouterr().err


def test_tools_are_the_production_initial_selection_not_the_registry_listing(source, monkeypatch):
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES, META_TOOL_NAMES
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools import registry as registry_module

    root, _folder, _chat = source
    made = []

    class Spy(registry_module.ToolRegistry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

        def on_demand_schema_names(self):
            # What an unpinned MCP server's tools are: listed by the registry, never loaded at the start.
            names = sorted(s['function']['name'] for s in self.schemas()
                           if s['function']['name'] not in CORE_TOOL_NAMES | META_TOOL_NAMES)
            return frozenset(names[-2:])

    monkeypatch.setattr(registry_module, 'ToolRegistry', Spy)
    result = report(root, 1, WINDOW, REPLY_CEILING, 'low')
    registry = made[-1]
    selected = initial_tool_schemas(registry, context_mode='low')
    row = next(row for row in result['blocks'] if row['block'] == 'tools')
    assert row['count'] == len(selected) == len(registry.schemas()) - 2  # the registry's listing is two schemas longer
    assert result['tools']['count'] == len(selected) and result['tools']['on_demand_withheld'] == 2


def test_tools_row_is_the_serialized_projection_on_the_physical_estimator(tmp_path):
    import json as _json

    from devtools.context_fit_report import measure_core
    from ouroboros import context_fit as cf
    from ouroboros.tool_policy import select_tool_schemas
    from ouroboros.utils import estimate_tokens

    result = measure_core(scale.core(tmp_path), scale.tools(), root=tmp_path, window=92000, reserve=10000)
    sent = list(select_tool_schemas(scale.tools(), context_mode=result['mode']).schemas)
    row = next(row for row in result['blocks'] if row['block'] == 'tools')
    compact = estimate_tokens(_json.dumps(sent, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    # The request serializes compactly; the plan-side estimator counts separator spaces the wire does not carry.
    assert abs(row['tokens'] - compact) <= 2 and row['tokens'] < cf.tool_schema_tokens(sent)
    assert result['estimator'].startswith('estimate_context_prompt_tokens') and row['count'] == len(sent)
    assert sum(block['tokens'] for block in result['blocks']) == result['input_tokens']


def test_project_room_carries_its_registered_folder_by_caller_kind_and_writes_nothing(source, monkeypatch, capsys):
    from ouroboros import context

    root, folder, chat = source
    captured = []
    capture = context._capture_context_core
    monkeypatch.setattr(context, '_capture_context_core', lambda env, memory, task, *a, **kw: captured.append(
        (task, a[-1])) or capture(env, memory, task, *a, **kw))
    before = fingerprint(root)
    code, out, _err = cli(capsys, root, room=chat)
    assert code == 0 and 'Caller: direct (role integrator)' in out
    assert report(root, chat, WINDOW, REPLY_CEILING, 'low', caller='root')['caller']['kind'] == 'root'
    (direct, registry), (queued, _registry) = captured
    # The capture receives the registry whose context is this very task, as production's does.
    assert registry._ctx.task_id == direct['id'] and registry._ctx.project_id == direct['project_id']
    # A direct turn uses the room lens; a queued root runs in the room's workspace. No invented note stands in for either.
    assert direct['metadata'] == {'_project_room_dir': str(folder)} and 'workspace_root' not in direct
    assert queued['workspace_root'] == str(folder) and queued['workspace_mode'] == 'external' and 'metadata' not in queued
    assert fingerprint(root) == before and not (root / 'task_drives').exists()


def test_project_room_without_a_usable_folder_is_refused_in_words_not_imitated(source, capsys):
    from ouroboros.projects_registry import list_projects

    root, _folder, _chat = source
    before = fingerprint(root)
    room = next(p['chat_id'] for p in list_projects(root) if not p.get('working_dir'))
    code, out, err = cli(capsys, root, room=room)
    assert code == 2 and not out and 'no usable registered folder' in err and 'PermissionError' not in err
    with pytest.raises(Unmeasurable, match='read-only report cannot model'):
        report(root, room, WINDOW, REPLY_CEILING, 'low', caller='root')
    assert fingerprint(root) == before and not (root / 'task_drives').exists()


def test_a_plan_that_falls_to_nano_is_summarized_by_the_selection_it_measured(source, capsys, monkeypatch):
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools import registry as registry_module

    root, _folder, _chat = source
    made = []

    class Registry(registry_module.ToolRegistry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(registry_module, 'ToolRegistry', Registry)
    result = report(root, 1, SMALL_WINDOW, REPLY_CEILING, 'low')
    resident, requested = (len(initial_tool_schemas(made[-1], context_mode=mode)) for mode in ('nano', 'low'))
    row = next(row for row in result['blocks'] if row['block'] == 'tools')
    tools = result['tools']
    # The control is real: Low was requested and the plan rendered Nano, whose request carries far fewer schemas.
    assert (result['requested_mode'], result['mode'], result['profile']) == ('low', 'nano', 'task_local_nano')
    assert resident < requested
    # One list: the summary and the table row are the Nano selection; the requested Low list is named, not mistaken for it.
    assert tools['count'] == row['count'] == resident and sum(tools['by_surface'].values()) == resident
    assert (tools['requested_mode'], tools['requested_count']) == ('low', requested)
    assert "'nano'" in tools['selector'] and 'select_tool_schemas' in tools['selector']
    # The public CLI prints the same numbers in the table and in the summary line.
    code, text, _err = cli(capsys, root, window=SMALL_WINDOW)
    table = re.search(r'^tools \| selected \| \d+ \| (\d+)$', text, re.M)
    line = re.search(r'^Caller: .*; tools: (\d+) in nano \(the requested low selection has (\d+)\)', text, re.M)
    assert code == 0 and table and line
    assert (table.group(1), line.group(1), line.group(2)) == (str(resident), str(resident), str(requested))
    # Control: a window Low fits in changes nothing, and the requested list is not repeated.
    unlowered = report(root, 1, WINDOW, REPLY_CEILING, 'low')
    assert unlowered['mode'] == 'low' and unlowered['tools']['count'] == requested
    assert 'requested_count' not in unlowered['tools']


@pytest.fixture
def session_spy(monkeypatch):
    """What the offline call path hands its plan, its refusal bound and its call host; every spy calls through."""
    import socket
    from types import SimpleNamespace

    from devtools import context_fit_report as report_module
    from ouroboros import context_fit
    from ouroboros.live_conversation import ConversationHost
    from ouroboros.tool_capabilities import CORE_TOOL_NAMES, META_TOOL_NAMES
    from ouroboros.tools import registry as registry_module
    from skills.live_conversation.driver import INITIAL_TOOLS
    from tests._socket_seal import block_network_except_socketpair

    # The offline call ends its host on an event loop. Windows builds one through connect() (a TCP socketpair,
    # simulated here on every OS), so only that self-pipe may pass the guard.
    monkeypatch.setattr(socket, 'socketpair', getattr(socket, '_fallback_socketpair', socket.socketpair))
    block_network_except_socketpair(monkeypatch, lambda *a, **kw: pytest.fail('provider connection'))
    spy = SimpleNamespace(registries=[], plans=[], bounds=[], hosts=[])

    class Registry(registry_module.ToolRegistry):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spy.registries.append(self)

        def on_demand_schema_names(self):
            # What an unpinned MCP server's tools are: listed by the registry, never loaded at the start.
            names = sorted(s['function']['name'] for s in self.schemas()
                           if s['function']['name'] not in CORE_TOOL_NAMES | META_TOOL_NAMES | set(INITIAL_TOOLS))
            return frozenset(names[-2:])

    build, measure, init = context_fit.build_context_fit_plan, report_module.measure_core, ConversationHost.__init__

    def build_spy(env, core, task, **kwargs):
        spy.plans.append(SimpleNamespace(task=task, tools=[s['function']['name'] for s in kwargs['tool_schemas']]))
        return build(env, core, task, **kwargs)

    def measure_spy(core, schemas, **kwargs):
        if kwargs.get('prepared') is None:  # a refusal bound, not the call host's measured candidate
            spy.bounds.append(SimpleNamespace(task=kwargs.get('task'), tools=[s['function']['name'] for s in schemas]))
        return measure(core, schemas, **kwargs)

    def init_spy(host, *args):
        init(host, *args)
        spy.hosts.append(host)

    monkeypatch.setattr(registry_module, 'ToolRegistry', Registry)
    monkeypatch.setattr(context_fit, 'build_context_fit_plan', build_spy)
    monkeypatch.setattr(report_module, 'measure_core', measure_spy)
    monkeypatch.setattr(ConversationHost, '__init__', init_spy)
    return spy


def test_call_path_plans_and_hosts_the_capture_task_with_the_initial_selection(source, session_spy, capsys):
    from ouroboros import live_conversation

    root, folder, chat = source
    before = fingerprint(root)
    code, out, _err = cli(capsys, root, '--json', '--session-provider', 'gemini_live', room=chat, mode='max',
                          window=CALL_WINDOW, reserve=CALL_RESERVE)
    result = json.loads(out)
    registry, plan, host = session_spy.registries[-1], session_spy.plans[0], session_spy.hosts[0]
    withheld = registry.on_demand_schema_names()
    # The plan site gets the initial selection: the registry's listing is two on-demand schemas longer.
    assert code == 0 and len(withheld) == 2 and not withheld & set(plan.tools)
    # The report ends the host it built: a live call left registered is met by the next settings write in this process.
    assert host not in live_conversation._active_hosts
    # One task, the capture's own folder-bound room, for the plan, the host and the host's tool context.
    assert plan.task['project_id'] == 'fit-folder' and plan.task['metadata'] == {'_project_room_dir': str(folder)}
    assert host.task == plan.task and host.ctx.project_id == 'fit-folder'
    assert host.ctx.task_metadata == plan.task['metadata'] and host.ctx.is_direct_chat
    assert not str(host.ctx.drive_root).startswith(str(root))  # the host writes under scratch, never the measured root
    # The call report carries the same caller and tool-source facts as the standalone report of that room.
    code_standalone, out_standalone, _err = cli(capsys, root, '--json', room=chat, mode='max', window=WINDOW)
    standalone = json.loads(out_standalone)
    assert code_standalone == 0 and result['caller'] == standalone['caller']
    assert result['caller']['task_fields'] == ['_is_direct_chat', 'metadata._project_room_dir', 'project_id']
    tools = result['tools']
    assert tools['verified_complete'] is False and tools['absent'] == standalone['tools']['absent']
    # The summary is the host's measured list, the call's own tools counted apart from the registry's.
    row = next(row for row in result['blocks'] if row['block'] == 'tools')
    assert result['phase'] == 'session_candidate' and tools['count'] == row['count'] == len(host.schemas)
    assert tools['by_surface']['call'] == len(host.call_tools) > 0 and sum(tools['by_surface'].values()) == tools['count']
    assert 'requested_count' not in tools and tools['selector'].startswith('the call host: ')
    assert fingerprint(root) == before and not (root / 'task_drives').exists()


def test_call_path_refusal_bound_measures_the_plans_selection_under_the_plans_task(source, session_spy, capsys):
    from ouroboros.tool_policy import initial_tool_schemas

    root, _folder, _chat = source
    code, text, _err = cli(capsys, root, '--session-provider', 'gemini_live', mode='max',
                           window=REFUSING_WINDOW, reserve=CALL_RESERVE)
    registry, plan = session_spy.registries[-1], session_spy.plans[0]
    withheld = registry.on_demand_schema_names()
    (bound,) = session_spy.bounds
    # Even Nano's fixed input is refused, so the call host seam is never reached and the bound is all there is.
    assert code == 1 and 'MinimumViewUnfit' in text and not session_spy.hosts
    # The bound is measured on the initial selection and the plan's task, not the registry listing and a made-up task.
    assert len(withheld) == 2 and not withheld & set(bound.tools) and bound.tools == plan.tools
    assert bound.task == plan.task and all(other.task == plan.task for other in session_spy.plans)
    # The printed summary is the resident Nano list the bound measured, with the requested list named once.
    resident = len(initial_tool_schemas(registry, context_mode='nano'))
    table = re.search(r'^tools \| selected \| \d+ \| (\d+)$', text, re.M)
    line = re.search(r'^Caller: .*; tools: (\d+) in nano \(the requested max selection has (\d+)\)', text, re.M)
    assert table and line and (table.group(1), line.group(1)) == (str(resident), str(resident))
    assert int(line.group(2)) == len(plan.tools) > resident

