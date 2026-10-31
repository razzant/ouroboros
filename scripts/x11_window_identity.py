#!/usr/bin/env python3
"""Byte-exact X11 client identity for the isolated native qualifier.

EWMH defines _NET_CLIENT_LIST as WINDOW/32, _NET_WM_PID as CARDINAL/32,
and _NET_WM_NAME as UTF8_STRING/8. Xlib returns format-32 data in C longs
(including on LP64), not packed uint32s. See packaging/appimage/README.md.
No X text/locale conversion, title regex or xdotool search is involved.
"""
from __future__ import annotations

import ctypes as C
import json
import os
import sys


class Attributes(C.Structure):
    _fields_ = [(key, C.c_int) for key in ('x', 'y', 'width', 'height', 'border_width', 'depth')] + [
        ('visual', C.c_void_p), ('root', C.c_ulong), ('window_class', C.c_int),
        ('bit_gravity', C.c_int), ('win_gravity', C.c_int), ('backing_store', C.c_int),
        ('backing_planes', C.c_ulong), ('backing_pixel', C.c_ulong), ('save_under', C.c_int),
        ('colormap', C.c_ulong), ('map_installed', C.c_int), ('map_state', C.c_int),
        ('all_event_masks', C.c_long), ('your_event_mask', C.c_long), ('do_not_propagate_mask', C.c_long),
        ('override_redirect', C.c_int), ('screen', C.c_void_p)]


class XError(C.Structure):
    _fields_ = [('type', C.c_int), ('display', C.c_void_p), ('resourceid', C.c_ulong),
                ('serial', C.c_ulong), ('error_code', C.c_ubyte), ('request_code', C.c_ubyte),
                ('minor_code', C.c_ubyte)]


def property_value(record, expected_type, expected_format):
    """None means absent; empty values are present; protocol/type errors raise."""
    if record['status'] != 0:
        raise RuntimeError(f"XGetWindowProperty failed: {record}")
    if record['type'] == 0 and record['format'] == 0 and record['bytes_after'] == 0:
        return None
    if record['type'] != expected_type or record['format'] != expected_format or record['bytes_after']:
        raise RuntimeError(f"Unexpected or truncated X11 property: {record}")
    if expected_format == 8:
        return bytes.fromhex(record['hex'])
    return record['values']


class X11:
    def __init__(self):
        self.lib = C.CDLL('libX11.so.6')
        signatures = {
            'XOpenDisplay': (C.c_void_p, [C.c_char_p]),
            'XDefaultRootWindow': (C.c_ulong, [C.c_void_p]),
            'XInternAtom': (C.c_ulong, [C.c_void_p, C.c_char_p, C.c_int]),
            'XGetWindowProperty': (C.c_int, [C.c_void_p, C.c_ulong, C.c_ulong, C.c_long, C.c_long,
                C.c_int, C.c_ulong, C.POINTER(C.c_ulong), C.POINTER(C.c_int), C.POINTER(C.c_ulong),
                C.POINTER(C.c_ulong), C.POINTER(C.POINTER(C.c_ubyte))]),
            'XGetWindowAttributes': (C.c_int, [C.c_void_p, C.c_ulong, C.POINTER(Attributes)]),
            'XFree': (C.c_int, [C.c_void_p]), 'XSync': (C.c_int, [C.c_void_p, C.c_int]),
            'XCloseDisplay': (C.c_int, [C.c_void_p]),
        }
        for name, (restype, argtypes) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = restype, argtypes
        self.errors = []
        callback = C.CFUNCTYPE(C.c_int, C.c_void_p, C.POINTER(XError))

        def on_error(_display, event):
            e = event.contents
            self.errors.append({'code': e.error_code, 'request': e.request_code, 'resource': e.resourceid})
            return 0

        self.handler = callback(on_error)  # keep callback alive until display close
        self.lib.XSetErrorHandler.argtypes = [callback]
        self.lib.XSetErrorHandler.restype = C.c_void_p
        self.lib.XSetErrorHandler(self.handler)
        self.display = self.lib.XOpenDisplay(None)
        if not self.display:
            raise RuntimeError(f"XOpenDisplay failed for DISPLAY={os.environ.get('DISPLAY')!r}")

    def atom(self, name):
        return self.lib.XInternAtom(self.display, name.encode('ascii'), False)

    def checked(self):
        self.lib.XSync(self.display, False)
        if self.errors:
            raise RuntimeError(f'X11 protocol error: {self.errors}')

    def property(self, window, name):
        kind, format_, count, after = C.c_ulong(), C.c_int(), C.c_ulong(), C.c_ulong()
        data = C.POINTER(C.c_ubyte)()
        status = self.lib.XGetWindowProperty(self.display, window, self.atom(name), 0, 16384, False, 0,
            C.byref(kind), C.byref(format_), C.byref(count), C.byref(after), C.byref(data))
        try:
            self.checked()
            record = {'status': status, 'type': kind.value, 'format': format_.value,
                      'items': count.value, 'bytes_after': after.value}
            if format_.value == 8:
                record['hex'] = C.string_at(data, count.value).hex()
            elif format_.value == 32:
                record['values'] = list(C.cast(data, C.POINTER(C.c_ulong))[:count.value])
            return record
        finally:
            if data:
                self.lib.XFree(data)

    def attributes(self, window):
        attributes = Attributes()
        status = self.lib.XGetWindowAttributes(self.display, window, C.byref(attributes))
        self.checked()
        if not status:
            raise RuntimeError(f'XGetWindowAttributes failed: {window:#x}')
        return {key: getattr(attributes, key) for key in ('x', 'y', 'width', 'height', 'map_state')}


def observe(x, result):
    root = x.lib.XDefaultRootWindow(x.display)
    atoms = {name: x.atom(name) for name in ('WINDOW', 'CARDINAL', 'STRING', 'UTF8_STRING')}
    result.update(root=root, atoms=atoms, clients=x.property(root, '_NET_CLIENT_LIST'), windows=[])
    clients = property_value(result['clients'], atoms['WINDOW'], 32)
    if clients is None:
        raise RuntimeError('Window manager has no _NET_CLIENT_LIST; cannot establish client absence')
    for window in clients:
        record = {'id': window, 'properties': {}}
        result['windows'].append(record)
        record.update(x.attributes(window))
        for name in ('WM_CLASS', '_NET_WM_PID', '_NET_WM_NAME'):
            record['properties'][name] = x.property(window, name)
        props = record['properties']
        klass = property_value(props['WM_CLASS'], atoms['STRING'], 8)
        pid = property_value(props['_NET_WM_PID'], atoms['CARDINAL'], 32)
        title = property_value(props['_NET_WM_NAME'], atoms['UTF8_STRING'], 8)
        if pid is not None and len(pid) != 1:
            raise RuntimeError(f'Invalid _NET_WM_PID: {pid}')
        record.update(wm_class=None if klass is None else klass.decode('latin-1').split('\0'),
                      pid=None if pid is None else pid[0],
                      title=None if title is None else title.decode('utf-8', errors='strict'))


def select_window(observation, pid, title, wm_class='Ouroboros'):
    if observation.get('status') != 'ok':
        raise RuntimeError(f'X11 identity observation failed: {observation}')
    matches = [r for r in observation['windows'] if r['pid'] == pid and r['map_state'] == 2
               and r['wm_class'] is not None
               and wm_class in r['wm_class']
               and r['title'] == title and r['width'] > 0 and r['height'] > 0]
    if len(matches) > 1:
        raise RuntimeError(f'Ambiguous visible launcher windows: {matches}')
    return matches[0] if matches else None


def main():
    result = {'status': 'error', 'DISPLAY': os.environ.get('DISPLAY')}
    x = None
    try:
        x = X11()
        observe(x, result)
        result['status'] = 'ok'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        if x is not None:
            x.lib.XCloseDisplay(x.display)
    # ASCII JSON carries every UTF-8 byte as hex too, irrespective of stdout locale.
    print(json.dumps(result, ensure_ascii=True))
    return 0 if result['status'] == 'ok' else 2


if __name__ == '__main__':
    sys.exit(main())
