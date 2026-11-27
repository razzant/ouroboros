#!/usr/bin/env python3
"""One-shot QA launcher: diagnose the existing reference CLI without editing it."""
import ast
import json
from pathlib import Path
import re
import runpy
import sys


PROFILES = {"fresh_bare", "fresh_recommended", "mature_recommended"}
MODELS = {"qwen35", "gemma4"}
CALL_STAGES = {
    "read_pin": "read_pin", "reference_counter": "reference_counter", "working_text": "working_text",
    "isolated_environment": "isolated_environment", "captured_root": "captured_root",
    "profile_result": "profile_result", "dumps": "report_serialization",
}
EXCEPTIONS = {"FileNotFoundError", "PermissionError", "ImportError", "ModuleNotFoundError", "IndexError",
              "AttributeError", "KeyError", "OSError", "RuntimeError", "TypeError", "ValueError",
              "UnicodeDecodeError", "AssertionError", "MinimumViewUnfit", "UndefinedError", "TemplateRuntimeError",
              "TemplateSyntaxError", "PackageNotFoundError", "JSONDecodeError"}
NUMBERS = ("estimated_input_tokens", "response_reserve_tokens", "window_tokens", "deficit_tokens")


def publish_exception(frame, exc, stage, root, destination):
    sites = []
    tb = exc.__traceback__
    while tb is not None:
        name = tb.tb_frame.f_code.co_filename
        if name.startswith(str(root) + "/"):
            rel = name[len(str(root)) + 1:]
            if (rel == "scripts/nano_reference_capacity.py"
                    or re.fullmatch(r"(?:ouroboros|supervisor)/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py", rel)):
                sites.append({"file": rel, "line": tb.tb_lineno})
        tb = tb.tb_next
    values = frame.f_locals if frame is not None else {}
    label, model = values.get("label"), values.get("model")
    profile = label if label in PROFILES else None
    model = model if model in MODELS else label if label in MODELS else None
    name = type(exc).__name__
    counts = {}
    if name == "MinimumViewUnfit":
        for key in NUMBERS:
            value = vars(exc).get(key)
            if type(value) is int:
                counts[key] = value
    destination.write_text(json.dumps({"stage": stage, "profile": profile, "model": model,
                                       "exception": name if name in EXCEPTIONS else "other",
                                       "sites": sites[-24:], "counts": counts}, indent=2) + "\n", encoding="utf-8")


def main():
    diagnostic = Path(sys.argv[1])
    root = Path.cwd().resolve()
    target = root / "scripts/nano_reference_capacity.py"
    tree = ast.parse(target.read_bytes())
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    handler, = [node for node in ast.walk(entry) if isinstance(node, ast.ExceptHandler) and node.name == "exc"]
    catch_line = handler.body[0].lineno
    stages = {}
    for call in ast.walk(entry):
        if isinstance(call, ast.Call):
            callee = call.func
            name = callee.id if isinstance(callee, ast.Name) else (
                "dumps" if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name)
                and callee.value.id == "json" and callee.attr == "dumps" else None)
            if name in CALL_STAGES:
                stages[call.lineno] = CALL_STAGES[name]
    stage = "main_entry"
    frame_seen = None

    def local_trace(frame, event, _arg):
        nonlocal stage
        if event != "line":
            return local_trace
        line = frame.f_lineno
        if line == catch_line:
            publish_exception(frame, frame.f_locals["exc"], stage, root, diagnostic)
        elif line in stages:
            stage = stages[line]
        return local_trace

    def global_trace(frame, event, _arg):
        nonlocal frame_seen
        if event == "call" and frame.f_code.co_filename == str(target) and frame.f_code.co_name == "main":
            frame_seen = frame
            return local_trace
        return None

    # Mirror direct file execution: argv[0] and sys.path[0] name the product script.
    sys.argv = [str(target), *sys.argv[2:]]
    sys.path[0] = str(target.parent)
    sys.settrace(global_trace)
    try:
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit:
        raise
    except Exception as exc:
        if not diagnostic.exists():
            publish_exception(frame_seen, exc, stage if frame_seen is not None else "before_main", root, diagnostic)
        raise
    finally:
        sys.settrace(None)


if __name__ == "__main__":
    main()
