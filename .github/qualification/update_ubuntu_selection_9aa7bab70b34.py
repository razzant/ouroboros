"""Equipment for one Ubuntu diagnostic of update candidate 9aa7bab70b34: not product code and not a test.

Run by ``.github/workflows/update-ubuntu-selection-9aa7bab70b34.yml``, only under the candidate's own
``scripts/safe_test.py``. Both modes write their records beneath ``OUROBOROS_TEST_EVIDENCE_ROOT`` (that owner's
run directory, outside the scratch it releases) and their working files beneath its short root, nowhere else.

``node`` runs exactly :data:`NODE` once, with the serial pass's flags, through ``pytest.main`` from the checkout
as ``python -m pytest`` would; its exit status is pytest's own. The candidate's test and product bytes run as
they are. The one addition is an observer plugin object: around this node only it wraps the test helper
``Scene``'s ``activate`` and ``boot`` (tests/test_update_activation.py) and appends one JSON line per point to
the record file it opened before pytest started: after activation (the environment record authorization
returned; the test then removes the generation), before the boot (the test has made it again) and after the
boot (its exit and output). Each point holds ``os.lstat`` of the recorded paths, the names beside them, the raw
handoff.json and the toy report. Between activation and boot it only reads and appends to that open file: it
creates no file or directory there. The selection runs in the boot's own process on the candidate's
body_switch; this sees only what that process leaves.

A failure of the observer's own output is a gap in the record, never the test's outcome. An ordinary I/O error of
the record file (write, flush, fsync, final close) or of the stderr line that reports a gap
(``SELECTION_OBSERVATION_INCOMPLETE``) goes to ``observer_errors`` of the end record and to stderr, whichever still
works (a close failure comes after the end record, so only stderr can carry it). It does not replace the boot's or
the test's exception or return value, stop the boot, or change pytest's status or the command's exit status: the
gap line is written past the buffer of ``sys.__stderr__``, to its descriptor, so a line that fails leaves nothing
there for the interpreter's final flush of stderr to fail on again (exit 120). Interrupts and cancellation are not
such errors and pass through untouched. Opening the record file before pytest starts is preparation: if that fails,
no test has run.

``recreate`` is the separately labelled control, not the product test: beneath its own safe_test root it makes
one directory shaped like the prepared generation (``lib/toydep-2.0``), removes it with ``shutil.rmtree``, makes
the same path again and records ``os.lstat`` before and after. One observation; it has no verdict.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import time

NODE = ("tests/test_update_environment_selection.py::"
        "test_a_selection_that_cannot_complete_returns_the_whole_transition_before_any_entry")
# The candidate's serial pass (ouroboros/preflight_runner.py SERIAL_PASS_FLAGS), the pass R4's red came from.
SERIAL_PASS_FLAGS = ("--timeout=300",)
RUN_MARKER = "OUROBOROS_TEST_EVIDENCE_RUN.txt"  # ouroboros/test_evidence.py EVIDENCE_RUN_MARKER
RECORDED = ("current", "generation", "previous")


def _owned_evidence() -> Path | None:
    """safe_test's run directory for this command; ``None`` (refused) anywhere else."""
    evidence = os.environ.get("OUROBOROS_TEST_EVIDENCE_ROOT", "")
    if not (os.environ.get("OUROBOROS_TEST_TEMP_ROOT") and evidence and (Path(evidence) / RUN_MARKER).is_file()):
        print("refused: run under scripts/safe_test.py with OUROBOROS_TEST_EVIDENCE_ROOT set", file=sys.stderr)
        return None
    return Path(evidence)


def _lstat(path) -> dict:
    """What is at ``path`` itself; ``identity`` is read as body_switch._path_identity reads it: ``[st_dev, st_ino]``."""
    try:
        seen = os.lstat(path)
    except FileNotFoundError:
        return {"path": str(path), "identity": None}
    except OSError as exc:
        return {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
    return {"path": str(path), "identity": [seen.st_dev, seen.st_ino], "mode": stat.filemode(seen.st_mode),
            "nlink": seen.st_nlink, "mtime_ns": seen.st_mtime_ns, "ctime_ns": seen.st_ctime_ns}


def _names(path, depth=2):
    """Names beneath ``path``, ``depth`` levels down, a directory's ending in ``/``; ``None`` when nothing is there."""
    try:
        with os.scandir(path) as listing:
            entries = sorted(listing, key=lambda entry: entry.name)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return [f"<unreadable: {type(exc).__name__}: {exc}>"]
    names = []
    for entry in entries:
        folder = entry.is_dir(follow_symlinks=False)
        names.append(entry.name + ("/" if folder else ""))
        if folder and depth > 1:
            names.extend(f"{entry.name}/{name}" for name in _names(entry.path, depth - 1) or ())
    return names


def _text(path):
    """A small file's raw text; ``None`` when nothing is there."""
    try:
        return Path(path).read_bytes().decode("utf-8", "replace")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return f"<unreadable: {type(exc).__name__}: {exc}>"


class _Observer:
    """The record of the one node: a JSON line per point, on disk before the test goes on."""

    def __init__(self, stream):
        self.stream, self.seen, self.errors, self.environment = stream, {}, [], {}

    def gap(self, what):
        """Say on the real stderr that the record has a gap. A stderr that fails is one more gap, for the end record.

        The line goes past the buffer of ``sys.__stderr__``, straight to its descriptor, so a line that cannot be
        written is dropped there and then. Left in that buffer, it would fail again in the interpreter's final flush of
        stderr, and the process would end with 120 instead of pytest's status. A stream with no binary buffer beneath
        it (an in-memory one) is printed to."""
        line = f"SELECTION_OBSERVATION_INCOMPLETE {what}"
        try:
            if getattr(sys.__stderr__, "buffer", None) is None:
                print(line, file=sys.__stderr__, flush=True)
            else:
                data, fd = (line + "\n").encode(sys.__stderr__.encoding, "backslashreplace"), sys.__stderr__.fileno()
                while data:  # a descriptor may take part of the line at a time
                    data = data[os.write(fd, data):]
        except (OSError, ValueError) as exc:  # a full, broken or closed stderr; an interrupt is not caught here
            self.errors.append(f"stderr: {type(exc).__name__}: {exc}")

    def write(self, point, facts):
        row = {"point": point, "monotonic_ns": time.monotonic_ns(), **facts}
        self.seen[point] = row
        try:
            self.stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except (OSError, TypeError, ValueError) as exc:
            self.errors.append(f"{point}: {type(exc).__name__}: {exc}")
            self.gap(f"{point}: {exc}")

    def observe(self, point, collect):
        try:
            facts = collect()
        except Exception as exc:  # noqa: BLE001 - a gap in the record, never a failure of the test
            self.errors.append(f"{point}: {type(exc).__name__}: {exc}")
            facts = {"error": f"{type(exc).__name__}: {exc}"}
        self.write(point, facts)

    def state(self, scene, handoff_file):
        """The recorded paths, the names beside them, the handoff and the toy report as they are now."""
        current = self.environment.get("current") or ""
        return {"paths": {key: _lstat(self.environment[key]) for key in RECORDED if self.environment.get(key)},
                "names": {"generation": _names(self.environment.get("generation") or ""), "current": _names(current),
                          "beside_current": _names(os.path.dirname(current), depth=1) if current else None},
                "handoff_json": _text(handoff_file), "report_jsonl": _text(scene.report)}

    def activated(self, scene, handoff, handoff_file):
        self.environment = json.loads(json.dumps(handoff.get("environment") or {}, default=str))
        return {"tmp": str(scene.tmp), "environment": self.environment, **self.state(scene, handoff_file)}

    def install(self, item):
        """Wrap this node's ``Scene.activate`` and ``boot``; returns the undo. Nothing else is touched."""
        module = item.module
        scene_class = module.Scene
        originals = {name: scene_class.__dict__[name] for name in ("activate", "boot")}

        def handoff_file(scene):
            return module.body_adoption.helper_dir(scene.data) / module.body_switch.HANDOFF_NAME

        def activate(scene, *args, **kwargs):
            handoff = originals["activate"](scene, *args, **kwargs)
            self.observe("after_activate", lambda: self.activated(scene, handoff, handoff_file(scene)))
            return handoff

        def boot(scene, *args, **kwargs):
            self.observe("before_boot", lambda: self.state(scene, handoff_file(scene)))
            try:
                result = originals["boot"](scene, *args, **kwargs)
            except BaseException as exc:
                self.observe("after_boot", lambda: {"raised": type(exc).__name__,
                                                    **self.state(scene, handoff_file(scene))})
                raise
            self.observe("after_boot", lambda: {"returncode": result.returncode, "stdout": result.stdout,
                                                "stderr": result.stderr, **self.state(scene, handoff_file(scene))})
            return result

        def undo():
            for name, original in originals.items():
                setattr(scene_class, name, original)

        scene_class.activate, scene_class.boot = activate, boot
        return undo

    def end(self):
        """The points seen, the observer's own gaps, and plain comparisons of the raw records above."""
        def identity(point, key):
            return ((self.seen.get(point) or {}).get("paths") or {}).get(key, {}).get("identity")

        def handoff(point):
            try:
                value = json.loads((self.seen.get(point) or {}).get("handoff_json") or "{}")
            except ValueError:
                return {}
            return value if isinstance(value, dict) else {}

        def same(one, other):
            return None if one is None or other is None else one == other

        armed, after = handoff("before_boot"), handoff("after_boot")
        recorded = ((armed.get("environment") or {}).get("identity") or {}).get("generation")
        made_again, current = identity("before_boot", "generation"), identity("after_boot", "current")
        derived = {
            "authorized_generation_identity": (self.environment.get("identity") or {}).get("generation"),
            "armed_generation_identity": recorded,
            "generation_identity_after_activate": identity("after_activate", "generation"),
            "generation_identity_before_boot": made_again,
            "made_again_equals_armed": same(made_again, recorded),
            "generation_identity_after_boot": identity("after_boot", "generation"),
            "current_identity_after_boot": current,
            "current_after_boot_equals_armed": same(current, recorded),
            "boot_returncode": (self.seen.get("after_boot") or {}).get("returncode"),
            "phase_after_boot": after.get("phase"),
            "environment_after_boot": {key: (after.get("environment") or {}).get(key) for key in ("state", "foreign")},
            "return_after_boot": after.get("return"),
            "entries_after_boot": after.get("entries"),
        }
        self.write("end", {"seen": sorted(self.seen), "observer_errors": list(self.errors), "derived": derived})

    def around(self, item):
        """The hook wrapper's body: this node only; the outcome is never read or changed."""
        if item.nodeid != NODE:
            yield
            return
        self.observe("start", lambda: {"nodeid": item.nodeid, "module_file": str(item.module.__file__),
                                       "cwd": os.getcwd(), "temp_root": os.environ.get("OUROBOROS_TEST_TEMP_ROOT")})
        undo = None
        try:
            undo = self.install(item)
        except Exception as exc:  # noqa: BLE001 - the node then runs unobserved, and the record says so
            self.errors.append(f"install: {type(exc).__name__}: {exc}")
        try:
            yield
        finally:
            try:
                if undo is not None:
                    undo()
                self.end()
            except Exception as exc:  # noqa: BLE001 - never turns the node's result into an error
                self.gap(f"end: {type(exc).__name__}: {exc}")

    def close(self):
        """Close the record once pytest is done. A close that cannot flush is a gap (only stderr can carry it: the
        end record is written), never the test's result."""
        try:
            self.stream.close()
        except (OSError, ValueError) as exc:  # the flush that failed before fails again; an interrupt is not caught
            self.gap(f"close: {type(exc).__name__}: {exc}")


def node(argv) -> int:
    """Run :data:`NODE` once, observed; the exit status is pytest's own."""
    if argv:
        print(f"node takes no arguments: it runs exactly {NODE}", file=sys.stderr)
        return 2
    evidence = _owned_evidence()
    if evidence is None:
        return 2
    import pytest

    class Plugin:
        def __init__(self, observer):
            self.observer = observer

        @pytest.hookimpl(hookwrapper=True)
        def pytest_runtest_call(self, item):
            yield from self.observer.around(item)

    (evidence / "observation").mkdir()
    # Opened before pytest starts: during the node the observer only appends to this one file. A failure to open it
    # is a failure of the preparation (no test has run), not a gap in the record.
    observer = _Observer(open(evidence / "observation" / "node.jsonl", "x", encoding="utf-8"))
    try:
        status = pytest.main([NODE, *SERIAL_PASS_FLAGS, "--ci-evidence-dir", str(evidence / "pytest" / "node")],
                             plugins=[Plugin(observer)])
    finally:
        observer.close()  # its I/O error is a gap: it replaces neither pytest's status nor an exception leaving main
    return int(status)


def recreate(argv) -> int:
    """The control: one directory made again at its path, as the node's lines 118-119 do; not the product test."""
    if argv:
        print("recreate takes no arguments", file=sys.stderr)
        return 2
    evidence = _owned_evidence()
    if evidence is None:
        return 2
    root = Path(os.environ["OUROBOROS_TEST_TEMP_ROOT"])
    (evidence / "observation").mkdir()
    work = Path(tempfile.mkdtemp(prefix="r", dir=root))
    generation = work / "generation"
    (generation / "lib").mkdir(parents=True)
    (generation / "lib" / "toydep-2.0").write_text("new\n")
    before = _lstat(generation)
    shutil.rmtree(generation)
    generation.mkdir()
    after = _lstat(generation)
    record = {"label": "control: one directory made again at its path; not the product test, no verdict",
              "steps": ["mkdir <path>/lib (with parents)", "write <path>/lib/toydep-2.0", "lstat <path>",
                        "shutil.rmtree(<path>)", "<path>.mkdir()", "lstat <path>"],
              "root": _lstat(root), "before": before, "after": after,
              "same_identity": None if before.get("identity") is None or after.get("identity") is None
              else before["identity"] == after["identity"]}
    with open(evidence / "observation" / "recreate.json", "x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=1, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    shutil.rmtree(work)  # nothing undeclared stays beneath safe_test's root
    print(json.dumps({key: record[key] for key in ("before", "after", "same_identity")}, sort_keys=True))
    return 0


def main(argv) -> int:
    modes = {"node": node, "recreate": recreate}
    if not argv or argv[0] not in modes:
        print("usage: node | recreate", file=sys.stderr)
        return 2
    return modes[argv[0]](list(argv[1:]))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
