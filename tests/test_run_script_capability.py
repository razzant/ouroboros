"""Installed interpreters carry a real script through dispatch and artifact delivery."""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

import pytest

from ouroboros.artifacts import registered_task_artifact
from ouroboros.tools.registry import ToolContext, ToolRegistry
from tests._cases import case_folders


def _registry(tmp_path, monkeypatch):
    repo, drive = tmp_path / "repo", tmp_path / "data"
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
    return registry, ctx, repo, drive


_PYTHON = "import sys\nfrom pathlib import Path\nprint(*sys.argv[1:], sep='\\n')\nwith Path('once.txt').open('a', newline='') as f: f.write('once\\n')\n"
_SCRIPTS = {
    "python3": _PYTHON,
    "python_absolute": _PYTHON,
    "node": "console.log(process.argv.slice(2).join('\\n')); require('node:fs').appendFileSync('once.txt', 'once\\n');",
    "perl": "print join(\"\\n\", @ARGV), \"\\n\"; open(my $f, '>>', 'once.txt') or die $!; binmode($f); print {$f} \"once\\n\"; close($f);",
    "zsh": "printf '%s\\n' \"$@\"\nprintf '%s\\n' once >> once.txt\n",
    "lua": "print(table.concat(arg, '\\n')); local f = assert(io.open('once.txt', 'a')); f:write('once\\n'); f:close()",
}


def test_installed_script_runs_once_and_delivers_its_output(subtests, tmp_path, monkeypatch):
    case_paths = case_folders(tmp_path)
    for runtime in list(_SCRIPTS):
        tmp_path = next(case_paths)
        with subtests.test(runtime=runtime), monkeypatch.context() as monkeypatch:
            interpreter = sys.executable if runtime == "python_absolute" else runtime
            if runtime not in {"python3", "python_absolute", "node"} and not shutil.which(interpreter):
                pytest.skip(f"{runtime} is not installed")
            registry, ctx, _, drive = _registry(tmp_path, monkeypatch)
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


@pytest.mark.parametrize("package_type", ["module", "commonjs", None])
def test_node_script_uses_the_staging_packages_module_format(tmp_path, monkeypatch, package_type):
    registry, _, repo, _ = _registry(tmp_path, monkeypatch)
    # A private package.json in the staging directory would break the workspace's #imports.
    package = {"imports": {"#value": "./value.cjs"}}
    if package_type:
        package["type"] = package_type
    (repo / "package.json").write_text(json.dumps(package), encoding="utf-8")
    (repo / "value.cjs").write_text("module.exports = 'native';", encoding="utf-8")
    for esm in (False, True):
        load = "import value from '#value';" if esm else "const value = require('#value');"
        script = load + "console.log(value, process.argv.slice(2).join(','), process.argv[1].endsWith('.js'));"
        args = ["two words", "--flag"]
        # Node versions differ on syntax detection in a typeless package. Compare
        # the same bytes with the actual selected Node and the same package scope.
        (repo / "reference.js").write_text(script, encoding="utf-8")
        direct = registry.execute("run_command", {"cmd": ["node", "reference.js", *args], "cwd": "active_workspace"})
        inline = registry.execute("run_script", {"interpreter": "node", "script": script, "args": args, "cwd": "active_workspace"})
        ok = "native two words,--flag true"
        if (esm and package_type == "module") or (not esm and package_type != "module") or (package_type is None and ok in direct.splitlines()):
            assert ok in direct.splitlines(), direct
            assert ok in inline.splitlines(), inline
        else:
            rejection = "Cannot use import statement" if esm else "require is not defined"
            assert rejection in direct and rejection in inline, (direct, inline)
        assert not list((repo / ".ouroboros").rglob("script.js"))


def test_named_root_reads_only_the_nearest_in_scope_node_declaration(tmp_path):
    from ouroboros.tools.shell import _node_stage_suffix

    (tmp_path / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    root = tmp_path / "root"
    sub = root / "src"
    sub.mkdir(parents=True)
    assert _node_stage_suffix(sub, root) == ".js"  # the outer package isn't part of this root
    (root / "package.json").write_text('{"type":"module"}', encoding="utf-8")
    assert _node_stage_suffix(sub, root) == ".mjs"
    for value, suffix in [('"commonjs"', '.cjs'), ('"CommonJS"', '.js'), ('42', '.js'), ('', '.js')]:
        (sub / "package.json").write_text('{"type":' + value + '}', encoding="utf-8")
        assert _node_stage_suffix(sub, root) == suffix
    modules = root / "node_modules" / "pkg"
    modules.mkdir(parents=True)
    assert _node_stage_suffix(modules, root) == ".js"


@pytest.mark.parametrize("package_type", ["module", "commonjs", None])
def test_node_named_root_staging_keeps_the_selected_package_module_format(tmp_path, monkeypatch, package_type):
    """Task-drive staging carries the selected package type without writing the script there."""
    registry, ctx, repo, _ = _registry(tmp_path, monkeypatch)
    package = {} if package_type is None else {"type": package_type}
    (repo / "package.json").write_text(json.dumps(package), encoding="utf-8")
    suffix = {"module": "mjs", "commonjs": "cjs"}.get(package_type, "js")
    args = ["two words", "--flag"]
    # Another invocation's staging scope must neither change the selected type nor be overwritten.
    sibling_package = ctx.task_drive_root() / "tmp_scripts" / "package.json"
    sibling_package.parent.mkdir(parents=True, exist_ok=True)
    sibling_package.write_text('{"type":"module"}', encoding="utf-8")

    # The declared format travels in the staged extension: a body in THAT format runs
    # from the named root and names the extension it ran from.
    body = (
        ("import { argv } from 'node:process';" if package_type == "module"
         else "const { argv } = require('node:process');")
        + "console.log('ran', argv.slice(2).join(','), argv[1].endsWith('.mjs') ? 'mjs'"
          " : argv[1].endsWith('.cjs') ? 'cjs' : 'js');"
    )
    result = registry.execute("run_script", {
        "interpreter": "node", "script": body, "args": args, "cwd": "system_repo",
    })
    assert f"ran two words,--flag {suffix}" in result, result

    # An ESM body behaves EXACTLY like the same bytes as a checked-in .js of the package
    # — Node's syntax detection in a typeless scope varies by version, so the native run
    # with the selected Node is the oracle, not a hardcoded error string.
    esm_body = "import { argv } from 'node:process'; console.log('esm ran', argv.slice(2).join(','));"
    (repo / "reference.js").write_text(esm_body, encoding="utf-8")
    direct = registry.execute("run_command", {"cmd": ["node", "reference.js", *args], "cwd": "system_repo"})
    inline = registry.execute("run_script", {"interpreter": "node", "script": esm_body, "args": args, "cwd": "system_repo"})
    assert ("esm ran" in direct) == ("esm ran" in inline), (direct, inline)
    assert not list(ctx.task_drive_root().glob("tmp_scripts/script_*"))
    assert sibling_package.read_text(encoding="utf-8") == '{"type":"module"}'


def test_node_package_scoped_imports_stay_an_active_workspace_affordance(tmp_path, monkeypatch):
    """Only the workspace stage is inside this package; imports resolve relative to the script."""
    registry, _, repo, _ = _registry(tmp_path, monkeypatch)
    (repo / "package.json").write_text(json.dumps({
        "type": "module", "imports": {"#value": "./value.mjs"},
    }), encoding="utf-8")
    (repo / "value.mjs").write_text("export default 'native';", encoding="utf-8")
    body = "import value from '#value'; console.log('value is', value);"
    inside = registry.execute("run_script", {"interpreter": "node", "script": body, "cwd": "active_workspace"})
    assert "value is native" in inside, inside
    outside = registry.execute("run_script", {"interpreter": "node", "script": body, "cwd": "system_repo"})
    assert "value is native" not in outside, outside
    assert "ERR_PACKAGE_IMPORT_NOT_DEFINED" in outside, outside
