"""Protocol bytes and qualifier decision seams; live X11 is parent-run only."""
import ctypes as C
import json
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
reader = runpy.run_path(str(ROOT / 'scripts/x11_window_identity.py'))


def test_xlib_format_32_is_an_array_of_native_longs_and_always_freed():
    values = (C.c_ulong * 2)(0x40000e, 0x600013)
    freed = []

    def get_property(display, window, atom, offset, length, delete, requested, kind, format_, count, after, data):
        assert (offset, length, delete, requested) == (0, 16384, False, 0)
        C.cast(kind, C.POINTER(C.c_ulong))[0] = 33
        C.cast(format_, C.POINTER(C.c_int))[0] = 32
        C.cast(count, C.POINTER(C.c_ulong))[0] = 2
        C.cast(after, C.POINTER(C.c_ulong))[0] = 0
        C.cast(data, C.POINTER(C.POINTER(C.c_ubyte)))[0] = C.cast(values, C.POINTER(C.c_ubyte))
        return 0

    x = reader['X11'].__new__(reader['X11'])
    x.lib = SimpleNamespace(XGetWindowProperty=get_property, XFree=lambda data: freed.append(bool(data)))
    x.display = 1
    x.atom = lambda _: 33
    x.checked = lambda: None
    observed = x.property(1, '_NET_CLIENT_LIST')
    assert observed['values'] == [0x40000e, 0x600013]  # packed uint32 gives [first, 0] on LP64
    assert freed == [True]
    x.checked = lambda: (_ for _ in ()).throw(RuntimeError('BadWindow'))
    with pytest.raises(RuntimeError, match='BadWindow'):
        x.property(1, '_NET_CLIENT_LIST')
    assert freed == [True, True]


def test_absent_empty_and_failed_properties_remain_distinct():
    decode = reader['property_value']
    absent = dict(status=0, type=0, format=0, bytes_after=0)
    empty = dict(status=0, type=17, format=8, bytes_after=0, hex='')
    assert decode(absent, 17, 8) is None
    assert decode(empty, 17, 8) == b''
    for changed in ({'status': 3}, {'type': 18}, {'format': 16}, {'bytes_after': 1}):
        with pytest.raises(RuntimeError):
            decode({**empty, **changed}, 17, 8)


def observation(title='Ouroboros — Setup'):
    return {'status': 'ok', 'windows': [dict(id=0x40000e, pid=120, wm_class=['Ouroboros', 'Ouroboros', ''],
        map_state=2, width=980, height=780, title=title)]}


def test_visible_window_requires_real_pid_class_geometry_and_exact_unicode_title():
    select = reader['select_window']
    expected = observation()
    assert select(expected, 120, 'Ouroboros — Setup')['id'] == 0x40000e
    # The public shell entry executes the renamed frozen binary. GTK's actual
    # WM_CLASS in the retained native run is this binary's name, not its wrapper.
    for actual, expected_class in [(['Ouroboros.bin', 'Ouroboros.bin', ''], 'Ouroboros.bin'),
                                   (['launcher.py', 'Launcher.py', ''], 'Launcher.py')]:
        packaged = observation()
        packaged['windows'][0]['wm_class'] = actual
        assert select(packaged, 120, 'Ouroboros — Setup', expected_class)['id'] == 0x40000e
        assert select(packaged, 999, 'Ouroboros — Setup', expected_class) is None
        assert select(packaged, 120, 'Ouroboros — Setup', 'other') is None
    for field, value in [('pid', 999), ('wm_class', ['other']), ('map_state', 0), ('width', 0),
                         ('title', None), ('title', ''), ('title', 'Ouroboros - Setup'),
                         ('title', 'Ouroboros — Setup Required')]:
        changed = observation()
        changed['windows'][0][field] = value
        assert select(changed, 120, 'Ouroboros — Setup') is None
    with pytest.raises(RuntimeError, match='Ambiguous'):
        select({'status': 'ok', 'windows': expected['windows'] * 2}, 120, 'Ouroboros — Setup')
    with pytest.raises(RuntimeError, match='observation failed'):
        select({'status': 'error', 'error': 'XOpenDisplay refused'}, 120, 'Ouroboros — Setup')


@pytest.mark.parametrize('title', [b'Ouroboros \xe2\x80\x94 Setup', 'Заголовок 🐍'.encode(), b'', None, b'\xff'])
def test_observe_decodes_only_utf8_property_bytes_and_retains_raw_evidence(title):
    atoms = {'WINDOW': 33, 'CARDINAL': 6, 'STRING': 31, 'UTF8_STRING': 99}
    props = {
        '_NET_CLIENT_LIST': dict(status=0, type=33, format=32, bytes_after=0, values=[0x40000e]),
        '_NET_WM_PID': dict(status=0, type=6, format=32, bytes_after=0, values=[120]),
        'WM_CLASS': dict(status=0, type=31, format=8, bytes_after=0, hex=b'Ouroboros\0Ouroboros\0'.hex()),
        '_NET_WM_NAME': dict(status=0, type=99, format=8, bytes_after=0, hex=(title or b'').hex())
            if title is not None else dict(status=0, type=0, format=0, bytes_after=0),
    }
    x = SimpleNamespace(display=1, lib=SimpleNamespace(XDefaultRootWindow=lambda _: 5), atom=atoms.get,
        property=lambda _, name: props[name], attributes=lambda _: dict(map_state=2, width=980, height=780))
    result = {}
    if title == b'\xff':
        with pytest.raises(UnicodeDecodeError):
            reader['observe'](x, result)
        assert result['windows'][0]['properties']['_NET_WM_NAME']['hex'] == 'ff'
    else:
        reader['observe'](x, result)
        assert result['windows'][0]['title'] == (None if title is None else title.decode('utf-8'))
    del props['_NET_WM_NAME']
    props['_NET_CLIENT_LIST'] = dict(status=0, type=0, format=0, bytes_after=0)
    with pytest.raises(RuntimeError, match='cannot establish client absence'):
        reader['observe'](x, {})


@pytest.mark.parametrize('returncode,body', [(0, observation()), (0, {'status': 'ok', 'windows': []}),
                                           (2, {'status': 'error', 'error': 'BadWindow'}), (0, 'not JSON')])
def test_qualifier_records_every_observation_and_never_retries_reader_errors_as_absence(tmp_path, monkeypatch, returncode, body):
    qualify = runpy.run_path(str(ROOT / 'scripts/qualify_linux_appimage.py'))
    raw = (json.dumps(body) if not isinstance(body, str) else body).encode()
    monkeypatch.setitem(qualify['find_window'].__globals__, 'subprocess', SimpleNamespace(
        run=lambda *args, **kw: SimpleNamespace(returncode=returncode, stdout=raw, stderr=b''),
        CalledProcessError=__import__('subprocess').CalledProcessError))
    if returncode or isinstance(body, str):
        with pytest.raises(RuntimeError):
            qualify['wait_for'](lambda: qualify['find_window']('Ouroboros — Setup', tmp_path, 120), 'Setup', timeout=1)
    else:
        result = qualify['find_window']('Ouroboros — Setup', tmp_path, 120)
        assert bool(result) == bool(body['windows'])
    evidence = json.loads((tmp_path / 'window-search.json').read_text())
    assert evidence['returncode'] == returncode and evidence['stdout'] == raw.decode()
    assert len((tmp_path / 'window-search.jsonl').read_text().splitlines()) == 1


def test_native_server_environment_oracle_fails_on_gui_leaks_and_preserves_empty():
    check = runpy.run_path(str(ROOT / 'scripts/qualify_linux_appimage.py'))['assert_server_environment']
    record = {'argv': ['python3', '/fixture/server.py'], 'environment': {'LD_LIBRARY_PATH': ''}}
    check([record], {'LD_LIBRARY_PATH': ''})
    with pytest.raises(AssertionError):
        check([record], {})
    record['environment'] = {'GI_TYPELIB_PATH': '/bundle/linux-gtk'}
    with pytest.raises(AssertionError):
        check([record], {})
    with pytest.raises(AssertionError, match='No actual server'):
        check([], {})
