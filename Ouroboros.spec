# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Ouroboros (macOS, Linux, Windows).

Bundles launcher.py as the entry point. The app ships an embedded managed git
bootstrap artifact (``repo.bundle`` + ``repo_bundle_manifest.json``) and still
includes the repo data tree needed by the launcher/runtime itself (web assets,
docs, tests, bundled skills, etc.). On first run the launcher materializes a
real git repo under ``~/Ouroboros/repo`` from the embedded bundle; the embedded
python-standalone interpreter then runs the agent as a subprocess.
"""

import os
import pathlib as _pathlib
import sys

from ouroboros.tool_module_inventory import (
    FROZEN_TOOL_MANIFEST_NAME as _FROZEN_TOOL_MANIFEST_NAME,
    build_frozen_tool_manifest as _build_frozen_tool_manifest,
)

block_cipher = None

# ---------------------------------------------------------------------------
# Platform-specific settings
# ---------------------------------------------------------------------------
_is_macos = sys.platform == "darwin"
_is_windows = sys.platform == "win32"

if _is_windows:
    _icon = 'assets/icon.ico' if os.path.exists('assets/icon.ico') else None
    _console = False
elif _is_macos:
    _icon = 'assets/icon.icns'
    _console = False
else:
    _icon = None
    _console = False

# ---------------------------------------------------------------------------
# Strip dev-only files from python-standalone before bundling.
# python-build-standalone ships symlinks (lib/pkgconfig, etc.) that break
# PyInstaller's BUNDLE step on macOS.
# ---------------------------------------------------------------------------
import shutil as _shutil
for _sub in ('include', 'share', 'lib/pkgconfig'):
    _p = os.path.join('python-standalone', _sub)
    if os.path.islink(_p):
        os.remove(_p)
    elif os.path.isdir(_p):
        _shutil.rmtree(_p)

# ---------------------------------------------------------------------------
# On Windows, pythonnet/clr_loader ship native DLLs that PyInstaller
# does not collect automatically. Gather them before Analysis.
# ---------------------------------------------------------------------------
from PyInstaller.utils.hooks import collect_all as _collect_all

_extra_datas = []
_extra_binaries = []
_extra_hiddenimports = []

# One side-effect-free source scan owns both PyInstaller's complete direct
# tools-package closure and the transient handler manifest read by a genuinely
# frozen ToolRegistry. The manifest is generated only after the build's clean
# repository gate and is written below the ignored PyInstaller work directory.
_frozen_tool_manifest_path = (
    _pathlib.Path("build") / "generated" / _FROZEN_TOOL_MANIFEST_NAME
)
_tool_module_inventory = _build_frozen_tool_manifest(
    _pathlib.Path("ouroboros") / "tools",
    _frozen_tool_manifest_path,
)
_extra_datas.append((str(_frozen_tool_manifest_path), "ouroboros"))
_extra_hiddenimports.extend(_tool_module_inventory.package_modules)

# Bundle the official, notarized Node.js runtime (pruned to bin/node[.exe]) so
# skill payloads with runtime=node and the `node --check` preflight work out of
# the box. The build scripts run scripts/download_node_standalone.* before
# PyInstaller; the signing pass re-signs node under the hardened runtime so
# macOS does not code-signing-kill it. The build scripts guarantee its presence
# in CI/release builds; for ad-hoc dev builds we warn-and-continue (rather than
# hard-fail) so a local PyInstaller run without node still produces an app —
# node-runtime skills simply fall back to PATH node there.
if os.path.isdir('node-standalone'):
    _extra_datas.append(('node-standalone', 'node-standalone'))
else:
    print('WARNING: node-standalone/ not found — bundled node will be absent and '
          'node-runtime skills will rely on PATH node. Run '
          'scripts/download_node_standalone.sh (or .ps1) before PyInstaller for a release build.')

if os.path.isdir('ripgrep-standalone'):
    _extra_datas.append(('ripgrep-standalone', 'ripgrep-standalone'))
else:
    print('WARNING: ripgrep-standalone/ not found — bundled rg will be absent and '
          'search_code will rely on PATH rg or the Python fallback. Run '
          'scripts/download_ripgrep_standalone.sh (or .ps1) before PyInstaller for a release build.')

# Seed the exact reviewed Claudexor engine closure. The archive stays compressed
# inside the app and is extracted into Ouroboros's writable data plane on first
# delegated use; Node remains a separate host-owned bundled runtime.
if os.path.isdir('claudexor-runtime'):
    _extra_datas.append(('claudexor-runtime', 'claudexor-runtime'))
else:
    print('WARNING: claudexor-runtime/ not found — managed Claudexor installation '
          'will download the exact pinned archive on first use. Release builds run '
          'scripts/fetch_claudexor_runtime.py before PyInstaller and must bundle it.')

if _is_windows:
    for _pkg in ('pythonnet', 'clr_loader'):
        try:
            _d, _b, _h = _collect_all(_pkg)
            _extra_datas += _d
            _extra_binaries += _b
            _extra_hiddenimports += _h
        except Exception:
            pass

# tree-sitter-language-pack ships native grammar binaries (.so/.dylib/.pyd) +
# data that PyInstaller does not collect automatically. Bundle them on every
# platform so polyglot code intelligence (query_code / inventory symbols for
# Go/Rust/Java/Ruby/C/... and JS/TS) works out of the box (WS3, v6.33.0). For
# ad-hoc dev builds we warn-and-continue: structural extraction then degrades
# VISIBLY (structural_unavailable:<lang>) rather than silently regex-guessing.
for _pkg in ('tree_sitter', 'tree_sitter_language_pack'):
    try:
        _d, _b, _h = _collect_all(_pkg)
        _extra_datas += _d
        _extra_binaries += _b
        _extra_hiddenimports += _h
    except Exception as _exc:
        print(f'WARNING: could not collect {_pkg} for bundling ({_exc}); polyglot '
              'structural extraction will degrade visibly unless the dep is importable at runtime.')

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=_extra_binaries,
    datas=[
        ('VERSION', '.'),
        ('repo.bundle', '.'),
        ('repo_bundle_manifest.json', '.'),
        ('.gitignore', '.'),
        ('BIBLE.md', '.'),
        ('README.md', '.'),
        ('requirements-runtime.lock', '.'),
        ('uv.lock', '.'),
        ('pyproject.toml', '.'),
        ('Makefile', '.'),
        ('server.py', '.'),
        ('ouroboros', 'ouroboros'),
        ('supervisor', 'supervisor'),
        ('prompts', 'prompts'),
        ('web', 'web'),
        ('docs', 'docs'),
        ('tests', 'tests'),
        ('assets', 'assets'),
        ('skills', 'skills'),
        ('python-standalone', 'python-standalone'),
    ] + _extra_datas,
    hiddenimports=[
        'webview',
        'ouroboros.config',
    ] + _extra_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['scripts/pyi_rth_pythonnet.py'] if _is_windows else [],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Ouroboros',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=_console,
    disable_windowed_traceback=False,
    icon=_icon,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Ouroboros',
)

# macOS application bundle (skipped on Linux/Windows)
if _is_macos:
    app = BUNDLE(
        coll,
        name='Ouroboros.app',
        icon='assets/icon.icns',
        bundle_identifier='com.ouroboros.agent',
        info_plist={
            'CFBundleShortVersionString': open('VERSION').read().strip(),
            'CFBundleVersion': open('VERSION').read().strip(),
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '12.0',
        },
    )
