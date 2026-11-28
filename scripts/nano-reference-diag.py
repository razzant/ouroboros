#!/usr/bin/env python3
"""Run the checked-out reference CLI; preserve complete synthetic rows and its first failure stage."""
import ast
import json
from pathlib import Path
import runpy
import sys


PROFILES = ("fresh_bare", "fresh_recommended", "mature_recommended", "preseeded_overflow")
MODELS = ("qwen35", "gemma4")
CALL_STAGES = {name: name for name in (
    "read_pin", "reference_counter", "working_text", "isolated_environment", "captured_root", "profile_result",
)}
CALL_STAGES["dumps"] = "report_serialization"
TRACE_STAGES = {
    **CALL_STAGES,
    "reference_instruction": "reference_instruction",
    "check_instruction_views": "instruction_view",
    "grow_guarded_memory": "guarded_growth",
    "publication_sample": "publication_sample",
    "guarded_write": "guarded_writer",
}
WRITERS = ("identity", "scratchpad", "overview", "knowledge_index", "mark")
# Exact suffixes from the public synthetic guarded_writer, never the exception body.
WRITER_REASONS = {
    "proposal was already active when the counter saw it": "active_before_count",
    "proposal did not reach the exact counter": "exact_counter_not_reached",
    "initial Settings bytes changed": "fixed_settings_changed",
    "counted, but active bytes or publication revision did not match": "counted_but_publication_did_not_match",
    "capacity refusal failed to preserve bytes/revision or prove memory pressure": "refusal_preservation_or_pressure_not_proven",
}
PROFILE_REASONS = {
    "unexpected profile binding": "unexpected_profile_binding",
    "initial rules failed ordinary readback": "fixed_rules_readback_mismatch",
    "initial rules absent, duplicated or not isolated by ablation": "fixed_rules_occurrence_mismatch",
    "initial Settings bytes changed": "fixed_settings_changed",
}
TYPED_REFUSALS = {"busy", "unavailable", "stale", "incomplete", "bootstrap_pending", "over_capacity"}
PROFILE_STAGES = {"guarded_profile." + label for label in ("settings_bootstrap", "initial", "seeded", "boundary")}
EXCEPTIONS = {
    "AssertionError", "AttributeError", "FileNotFoundError", "ImportError", "IndexError", "JSONDecodeError",
    "KeyError", "MinimumViewUnfit", "ModuleNotFoundError", "NanoPublicationRefused", "OSError",
    "PackageNotFoundError", "PermissionError", "RuntimeError", "SyntaxError", "TemplateRuntimeError",
    "TemplateSyntaxError", "TypeError", "UndefinedError", "UnicodeDecodeError", "ValueError",
}


def main():
    diagnostic = Path(sys.argv[1])
    target = Path.cwd().resolve() / "scripts/nano_reference_capacity.py"
    complete = {}
    first_failure = False
    stage, frame_seen = "launcher_setup", None

    def record(*, error=None, frame=None):
        nonlocal first_failure
        if first_failure:
            return
        result_stage, reason = stage, None
        if error is not None:
            first_failure = True
            tb = error.__traceback__
            while tb is not None:
                where = tb.tb_frame
                function = where.f_code.co_name
                if where.f_code.co_filename == str(target) and function in TRACE_STAGES:
                    result_stage = TRACE_STAGES[function]
                    writer = where.f_locals.get("label")
                    if function == "guarded_write" and writer in WRITERS:
                        result_stage += "." + writer
                        if type(error) is ValueError and len(error.args) == 1 and isinstance(error.args[0], str):
                            for phrase, code in WRITER_REASONS.items():
                                if error.args[0] == f"{writer}: {phrase}":
                                    reason = code
                                    break
                tb = tb.tb_next
            if type(error) is ValueError and len(error.args) == 1 and isinstance(error.args[0], str):
                prefix, _, why = error.args[0].partition(": ")
                if prefix in PROFILE_STAGES:
                    result_stage = prefix
                    reason = PROFILE_REASONS.get(why) or (why if why in TYPED_REFUSALS or why in EXCEPTIONS else None)
                elif prefix in {"guarded_writer." + label for label in WRITERS}:
                    result_stage = prefix
                    reason = WRITER_REASONS.get(why) or (why if why in TYPED_REFUSALS else None)
                elif error.args[0] == "guarded fresh start: wrong occurrence count of the initial rules":
                    result_stage, reason = "guarded_fresh_start", "fixed_rules_occurrence_mismatch"
        local = frame.f_locals if frame is not None else {}
        label, model = local.get("label"), local.get("model")
        exception = type(error).__name__ if error is not None else None
        evidence = {
            "state": "exception" if error is not None else "checkpoint",
            "stage": result_stage,
            "profile": label if label in PROFILES else None,
            "model": label if label in MODELS else model if label in PROFILES and model in MODELS else None,
            "exception": exception if exception is None or exception in EXCEPTIONS else "other",
            "reason": reason,
            "completed_profiles": complete,
        }
        pending = diagnostic.with_suffix(".tmp")
        pending.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        pending.replace(diagnostic)

    try:
        tree = ast.parse(target.read_bytes())
        entry, = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"]
        handler, = [node for node in ast.walk(entry) if isinstance(node, ast.ExceptHandler) and node.name == "exc"]
        catch_line = handler.body[0].lineno
        stages = {}
        for call in ast.walk(entry):
            if not isinstance(call, ast.Call):
                continue
            callee = call.func
            name = callee.id if isinstance(callee, ast.Name) else (
                "dumps" if isinstance(callee, ast.Attribute) and isinstance(callee.value, ast.Name)
                and callee.value.id == "json" and callee.attr == "dumps" else None)
            if name in CALL_STAGES:
                stages[call.lineno] = CALL_STAGES[name]

        def trace_main(frame, event, _arg):
            nonlocal stage
            if event == "line":
                line = frame.f_lineno
                if line == catch_line:
                    record(error=frame.f_locals["exc"], frame=frame)
                elif line in stages:
                    stage = stages[line]
            return trace_main

        def trace_profile(_frame, event, result):
            # An exception unwind also emits return, but its value is None. Never publish an in-progress row.
            if event == "return" and isinstance(result, dict) and {"F", "F0", "B", "work_pair", "joint_witness"} <= result.keys():
                profile, model = (frame_seen.f_locals[key] for key in ("label", "model"))
                if profile not in PROFILES or model not in MODELS:
                    raise RuntimeError("unexpected reference profile")
                complete.setdefault(profile, {})[model] = result
                record(frame=frame_seen)
            return trace_profile

        def global_trace(frame, event, _arg):
            nonlocal frame_seen
            if event == "call" and frame.f_code.co_filename == str(target):
                if frame.f_code.co_name == "main":
                    frame_seen = frame
                    return trace_main
                if frame_seen is not None and frame.f_code.co_name == "profile_result" and frame.f_back is frame_seen:
                    frame.f_trace_lines = False
                    return trace_profile
            return None

        # The locked environment deliberately does not install the project. Only the bound checkout supplies its imports.
        sys.path[:1] = [str(target.parent), str(target.parent.parent)]
        sys.argv = [str(target), *sys.argv[2:]]
        stage = "before_main"
        sys.settrace(global_trace)
        runpy.run_path(str(target), run_name="__main__")
    except Exception as exc:
        record(error=exc, frame=frame_seen)
        raise
    finally:
        sys.settrace(None)


if __name__ == "__main__":
    main()
