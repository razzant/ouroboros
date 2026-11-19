"""The bundled Python pin and what must agree with it.

``scripts/download_python_standalone.sh`` pins the CPython the desktop packages, the Docker
images and the shipped CI lanes run. Each build freezes its launcher from a venv of that same
interpreter and refuses a python-standalone of another version. Pillow 13 (decoder security
fixes) has no Python 3.10 wheels, so an installed app still on 3.10 keeps updating on 12.3.
"""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement

REPO = pathlib.Path(__file__).resolve().parents[1]
CHECKER = REPO / "scripts" / "check_python_build.py"


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _pin() -> str:
    return re.search(r'^PY_VERSION="([0-9.]+)"$', _read("scripts/download_python_standalone.sh"), re.MULTILINE)[1]


def test_every_carrier_of_the_bundled_python_names_the_pin():
    version = _pin()
    assert f'$PyVersion = "{version}"' in _read("scripts/download_python_standalone.ps1")
    upstream = json.loads(_read("scripts/windows_signing_catalog.json"))["pythonUpstream"]
    assert upstream["pythonVersion"] == version
    assert upstream["archive"].startswith(f"cpython-{version}+{upstream['release']}-x86_64-pc-windows-msvc-")
    docker = _read("Dockerfile")
    assert f'FILENAME="cpython-{version}+{upstream["release"]}-' in docker
    assert f"assert version=='{version}', version" in docker
    # Every older app's Update preview parses this exact sentence from its update target (update_impact.py).
    assert re.search(rf"^Desktop downloads bundle CPython {re.escape(version)} and SQLite \d+\.\d+\.\d+\.",
                     _read("README.md"), re.MULTILINE)
    for page in ("site/install/index.html", "docs/install/index.html"):
        assert f"Desktop downloads bundle CPython {version} and SQLite " in _read(page)
    assert f"(CPython {version}, SQLite " in _read("README.md")  # the Docker paragraph
    deployment = _read("docs/DEPLOYMENT.md")
    assert f"Packages built from this source bundle CPython {version};" in deployment
    assert f"CPython {version} and SQLite " in deployment
    assert f"expect {version} and Pillow" in deployment  # the transition check an operator runs
    # The managed container's runtime venv is named for the image's minor (runtime_venv refuses another minor's venv).
    minor = ".".join(version.split(".")[:2])
    for text in (deployment, _read("Dockerfile")):  # each names it: a revert to the unversioned path fails too
        venvs = re.findall(r"/ouroboros/body/venv-(\d+\.\d+)", text)
        assert venvs and set(venvs) == {minor}, venvs


def _environment(python: str) -> dict:
    return {"python_full_version": python, "python_version": ".".join(python.split(".")[:2]),
            "implementation_name": "cpython", "platform_python_implementation": "CPython",
            "sys_platform": "linux", "platform_machine": "x86_64"}


@pytest.mark.parametrize(("python", "pillow"), [
    ("3.10.0", "12.3.0"), ("3.10.22", "12.3.0"), ("3.11.0", "13.0.0"), ("3.12.15", "13.0.0"),
])
def test_each_python_takes_exactly_one_pillow_line(python, pillow):
    block = _read("pyproject.toml").split("dependencies = [", 1)[1].split("\n]", 1)[0]
    declared = [Requirement(text) for text in re.findall(r'^\s*"([Pp]illow[^"]*)"', block, re.MULTILINE)]
    active = [req for req in declared if req.marker is None or req.marker.evaluate(_environment(python))]
    assert len(active) == 1, [str(req) for req in active]
    assert active[0].specifier.contains(pillow)
    # 3.10 is never handed Pillow 13 (no cp310 wheels), 3.11+ never the vulnerable 12.x line.
    assert not active[0].specifier.contains("13.0.0" if pillow == "12.3.0" else "12.3.0")
    locked = []
    for line in _read("requirements-runtime.lock").splitlines():
        if line.startswith("pillow=="):
            requirement, _, marker = line.partition(";")
            if not marker.strip() or Marker(marker.strip()).evaluate(_environment(python)):
                locked.append(requirement.strip().split("==", 1)[1])
    assert len(locked) == 1 and locked[0].split(".")[0] == pillow.split(".")[0]


def _check(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECKER), *args], capture_output=True, text=True, timeout=120)


def test_the_build_check_passes_the_pinned_python_and_refuses_another():
    sys.path.insert(0, str(CHECKER.parent))
    try:
        import check_python_build
    finally:
        sys.path.remove(str(CHECKER.parent))
    assert check_python_build.pinned_version() == _pin()
    own = "%d.%d.%d" % sys.version_info[:3]
    passed = _check("--expect", own)
    assert passed.returncode == 0, passed.stderr
    stale = _check("--expect", "3.99.0")
    assert stale.returncode == 1 and f"is Python {own}; this checkout pins 3.99.0" in stale.stderr


@pytest.mark.parametrize(("script", "embedded", "venv", "freeze"), [
    ("build.sh", "python-standalone/bin/python3", 'python-standalone/bin/python3 -m venv "$BUILD_VENV"',
     '"$BUILD_PYTHON" -m PyInstaller Ouroboros.spec'),
    ("build_windows.ps1", '& "python-standalone\\python.exe"', '& "python-standalone\\python.exe" -m venv $BuildVenv',
     "& $BuildPython -m PyInstaller Ouroboros.spec"),
    ("build_linux.sh", '"$PORTABLE_PYTHON"', '"$PORTABLE_PYTHON" -m venv "$BUILD_VENV"',
     '"$BUILD_PYTHON" -m PyInstaller Ouroboros.spec'),
])
def test_every_build_checks_its_python_and_freezes_from_a_venv_of_it(script, embedded, venv, freeze):
    """The shipped interpreter runs the pin check itself before anything is installed, then is the base of the
    venv that freezes the launcher: no PATH Python takes part."""
    src = _read(script)
    check = src.index("scripts/check_python_build.py")
    line = src[src.rindex("\n", 0, check) + 1:src.index("\n", check)]
    assert line.strip().startswith(embedded), line
    first_install = min(src.index(marker) for marker in ("uv pip install", "-m venv") if marker in src)
    assert check < first_install, "the pin is checked before anything is installed"
    assert venv in src and freeze in src
    assert src.index(venv) < src.index(freeze)
    assert not re.search(r"^\s*(?:python3?|& python) -m PyInstaller", src, re.MULTILINE), "no PATH Python freezes"
