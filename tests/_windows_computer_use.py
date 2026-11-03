"""Import the bundled ``skills/windows_computer_use`` payload as a package for tests.

The payload's modules import one another relatively, as the extension loader
stages them; registering the package by file location keeps ``skills/`` off
``sys.path``, where ``skills/telegram`` would shadow other imports.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

SKILL_DIR = pathlib.Path(__file__).resolve().parents[1] / "skills" / "windows_computer_use"
PACKAGE = "windows_computer_use"


def load_package():
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, SKILL_DIR / "__init__.py", submodule_search_locations=[str(SKILL_DIR)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    return sys.modules[PACKAGE]


load_package()
