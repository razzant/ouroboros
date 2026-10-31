"""The installed-skills legend bounds a skill's tool list and says so."""
from types import SimpleNamespace

from ouroboros.context_sections import _build_installed_skills_section


def _section(monkeypatch, tmp_path, surfaces):
    rows = [dict(name='unix_computer_use', enabled=True, executable_review=True, type='extension', version='0.5.2',
                 review_status='pending', description='Desktop tools.', when_to_use='When asked.',
                 tool_surfaces=[{'name': f'ext_19_r_unix_computer_use_{n}'} for n in surfaces])]
    monkeypatch.setattr('ouroboros.skill_loader.summarize_skills', lambda *_: {'skills': rows})
    return _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])


def test_a_bounded_tool_list_names_the_rest_and_where_to_read_them(monkeypatch, tmp_path):
    """The 27.10 call read the first eight names as the whole skill, enabled seven desktop tools and told the
    owner that click and key did not exist."""
    names = ['list_connections', 'add_connection', 'test_connection', 'activate_connection', 'use_local',
             'clear_active_connection', 'capabilities', 'screenshot', 'click', 'key', 'type_text', 'scroll']
    [line] = [line for line in _section(monkeypatch, tmp_path, names).splitlines() if line.startswith('  Tools:')]
    assert line.endswith("ext_19_r_unix_computer_use_screenshot; and 4 more: "
                         "list_available_tools(namespace='unix_computer_use') names them all")
    assert 'ext_19_r_unix_computer_use_click' not in line


def test_a_short_tool_list_is_complete_and_says_nothing_more(monkeypatch, tmp_path):
    [line] = [line for line in _section(monkeypatch, tmp_path, ['screenshot', 'click']).splitlines()
              if line.startswith('  Tools:')]
    assert line == '  Tools: ext_19_r_unix_computer_use_screenshot, ext_19_r_unix_computer_use_click'
