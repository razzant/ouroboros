"""Installed interpreters carry a real script through dispatch and artifact delivery."""

from __future__ import annotations

import pathlib
import shutil
import sys

import pytest

from ouroboros.artifacts import registered_task_artifact
from ouroboros.tools.registry import ToolContext, ToolRegistry


pytestmark = pytest.mark.serial

_PYTHON = "import sys\nfrom pathlib import Path\nprint(*sys.argv[1:], sep='\\n')\nwith Path('once.txt').open('a', newline='') as f: f.write('once\\n')\n"
_SCRIPTS = {
    "python3": _PYTHON,
    "python_absolute": _PYTHON,
    "node": "console.log(process.argv.slice(2).join('\\n')); require('node:fs').appendFileSync('once.txt', 'once\\n');",
    "perl": "print join(\"\\n\", @ARGV), \"\\n\"; open(my $f, '>>', 'once.txt') or die $!; binmode($f); print {$f} \"once\\n\"; close($f);",
    "zsh": "printf '%s\\n' \"$@\"\nprintf '%s\\n' once >> once.txt\n",
    "lua": "print(table.concat(arg, '\\n')); local f = assert(io.open('once.txt', 'a')); f:write('once\\n'); f:close()",
}


@pytest.mark.parametrize("runtime", list(_SCRIPTS))
def test_installed_script_runs_once_and_delivers_its_output(tmp_path, monkeypatch, runtime):
    interpreter = sys.executable if runtime == "python_absolute" else runtime
    if runtime not in {"python3", "python_absolute", "node"} and not shutil.which(interpreter):
        pytest.skip(f"{runtime} is not installed")
    repo = tmp_path / "repo"
    drive = tmp_path / "data"
    repo.mkdir()
    drive.mkdir()
    ctx = ToolContext(repo_dir=repo, drive_root=drive, task_id="script-capability")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    monkeypatch.setattr(
        "ouroboros.safety._run_llm_check",
        lambda *_args, **_kwargs: pytest.fail("Safety Off must not call a model"),
    )
    registry = ToolRegistry(repo_dir=repo, drive_root=drive)
    registry.set_context(ctx)
    args = ["alpha beta", ";$(literal)", "--literal-flag"]

    result = registry.execute("run_script", {
        "interpreter": interpreter, "script": _SCRIPTS[runtime], "args": args,
        "cwd": "task_drive", "outputs": ["once.txt"],
    })

    assert "ARTIFACT_OUTPUTS" in result, result
    assert "\n".join(args) in result, result
    record = registered_task_artifact(drive, ctx.task_id, "once.txt")
    assert record and record["kind"] == "process_output"
    assert pathlib.Path(record["path"]).read_bytes() == b"once\n"
    assert not list(ctx.task_drive_root().glob("tmp_scripts/script_*"))
