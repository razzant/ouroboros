"""The installed-skills legend names every tool of a skill, in an order the cache can keep."""
from types import SimpleNamespace

from ouroboros.context_sections import _build_installed_skills_section


def _section(monkeypatch, tmp_path, surfaces):
    rows = [dict(name='unix_computer_use', enabled=True, executable_review=True, type='extension', version='0.5.2',
                 review_status='pending', description='Desktop tools.', when_to_use='When asked.',
                 tool_surfaces=[{'name': f'ext_19_r_unix_computer_use_{n}'} for n in surfaces])]
    monkeypatch.setattr('ouroboros.skill_loader.summarize_skills', lambda *_: {'skills': rows})
    return _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])


def test_the_tool_list_names_every_tool_sorted_whatever_the_registration_order(monkeypatch, tmp_path):
    """The 27.10 call read the first eight names as the whole skill, enabled seven desktop tools and told the
    owner that click and key did not exist. Every name is listed now, and sorted: the section sits in the
    cached prefix, so two registration orders must give the same bytes."""
    names = ['list_connections', 'add_connection', 'test_connection', 'activate_connection', 'use_local',
             'clear_active_connection', 'capabilities', 'screenshot', 'click', 'key', 'type_text', 'scroll']
    first = _section(monkeypatch, tmp_path, names)
    assert first == _section(monkeypatch, tmp_path, list(reversed(names)))
    [line] = [line for line in first.splitlines() if line.startswith('  Tools:')]
    assert line == '  Tools: ' + ', '.join(f'ext_19_r_unix_computer_use_{n}' for n in sorted(names))
    assert 'more' not in line
    assert first.splitlines()[1] == ('A "Tools" line that lists a skill\'s registered tools names all of them; '
                                     'enable_tools("<skill name>") loads them in one call where their schemas are '
                                     'not in the request yet.')


def test_the_loading_hint_reaches_both_forms_only_beside_registered_tools(monkeypatch, tmp_path):
    """A skill_exec skill registers no tool, so its name loads nothing: alone it gets no loading hint, beside an
    extension the hint is in the full and the compact form, and every skill keeps its name in both forms."""
    hint = 'enable_tools("<skill name>") loads'
    script = dict(name='script_tool', enabled=True, executable_review=True, type='script', version='1',
                  review_status='pass', description='Runs a script.', when_to_use='When asked.',
                  runnable_via_skill_exec=True, tool_surfaces=[])
    monkeypatch.setattr('ouroboros.skill_loader.summarize_skills', lambda *_: {'skills': [script]})
    alone = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])
    assert '  Tools: skill_exec' in alone and hint not in alone and hint not in alone.operative_view
    assert all('script_tool' in form for form in (alone, alone.operative_view))

    extension = _section(monkeypatch, tmp_path, ['screenshot'])
    rows = [script, dict(name='unix_computer_use', enabled=True, executable_review=True, type='extension',
                         description='Desktop tools.', when_to_use='When asked.',
                         tool_surfaces=[{'name': 'ext_19_r_unix_computer_use_screenshot'}])]
    monkeypatch.setattr('ouroboros.skill_loader.summarize_skills', lambda *_: {'skills': rows})
    mixed = _build_installed_skills_section(SimpleNamespace(drive_root=tmp_path), skills=[])
    for form in (extension, extension.operative_view, mixed, mixed.operative_view):
        assert hint in form and 'unix_computer_use' in form
    assert 'script_tool' in mixed.operative_view


def test_an_extension_without_registered_tools_keeps_its_name_and_adds_no_loading_hint(monkeypatch, tmp_path):
    text = _section(monkeypatch, tmp_path, [])
    assert '  Tools:' not in text and 'enable_tools' not in text and 'enable_tools' not in text.operative_view
    assert all('unix_computer_use' in form for form in (text, text.operative_view))
