"""One pushed revision on one disposable GitHub-hosted Mac. All simctl mutations use the newly created ID."""
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from urllib.request import ProxyHandler, build_opener

from report import CASES, summarize, testcase_results

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REF = "refs/heads/probe/telegram-uikit-keyboard"
BUNDLE = "org.ouroboros.MotionProbeR1"
SIM = ["xcrun", "simctl"]
REFERENCE_WEB = "2140e2b19a92bf016aa4c19bf0b86f45cff50232"
REFERENCE_DATA = "9a55aec19a44cd6d6d50dcc524b341f3505051901152459ea904888e893390d3"


def output(*args):
    return subprocess.check_output(args, text=True).strip()


def select_phone(catalog, sdk):
    def version(s):
        return tuple(int(v) for v in s.split("."))
    runtimes = [r for r in catalog["runtimes"] if r.get("isAvailable") and
                r["identifier"].startswith("com.apple.CoreSimulator.SimRuntime.iOS-") and
                (17,) <= version(r["version"])[:2] <= version(sdk)[:2]]
    for runtime in sorted(runtimes, key=lambda r: version(r["version"]), reverse=True):
        available = {d.get("deviceTypeIdentifier") for d in catalog["devices"].get(runtime["identifier"], [])
                     if d.get("isAvailable")}
        available.update(d.get("identifier") for d in runtime.get("supportedDeviceTypes", []))
        phones = [d for d in catalog["devicetypes"] if d["identifier"] in available and d["name"].startswith("iPhone ")]
        if phones:
            # A standard portrait iPhone, if advertised for this runtime.
            def rank(d):
                gen = re.fullmatch(r"iPhone (\d+)", d["name"])
                return bool(gen), int(gen[1]) if gen else 0, d["name"]
            return runtime, max(phones, key=rank)
    raise RuntimeError("No installed, available iPhone runtime compatible with the selected simulator SDK")


def save(path, body):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(body, indent=2 if path.name == "summary.json" else None, allow_nan=False) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def run():
    test_deadline = time.monotonic() + 27 * 60  # The hosted job has 35 minutes including setup and upload.
    if (sys.platform != "darwin" or os.getenv("GITHUB_ACTIONS") != "true" or
            os.getenv("PROBE_RUNNER") != "github-hosted" or os.getenv("GITHUB_REF") != REF or
            os.getenv("GITHUB_EVENT_NAME") != "push"):
        raise SystemExit("Requires the dedicated GitHub-hosted push job")
    sha = os.environ["GITHUB_SHA"]
    if not re.fullmatch("[0-9a-f]{40}", sha) or output("git", "-C", REPO, "rev-parse", "HEAD") != sha:
        raise SystemExit("Checkout does not match the push")
    root = Path(os.environ["RUNNER_TEMP"])
    export = root / "uikit-observation"
    export.mkdir()  # Never mix attempts.
    private = Path(tempfile.mkdtemp(dir=root, prefix="keyboard-run-"))
    stamp = str(uuid.uuid4())
    result = private / "Keyboard.xcresult"
    web = output("git", "-C", REPO, "rev-parse", "HEAD:web")
    report = {"purpose": "UIKit keyboard observation", "productVerdict": "unassessed", "commit": sha,
              "webTree": web, "matchesReferenceWeb": web == REFERENCE_WEB, "ref": REF, "cases": {},
              "probeTree": output("git", "-C", REPO, "rev-parse", "HEAD:devtools/uikit-keyboard"),
              "capture": "incomplete", "stage": "toolchain"}
    traces = {}
    def checkpoint(stage):
        report["stage"] = stage
        save(export / "summary.json", report)
    server, udid = None, None
    cleanup_errors = []
    checkpoint("toolchain")
    try:
        report["xcode"] = output("xcodebuild", "-version").splitlines()
        report["macOS"] = output("sw_vers", "-productVersion")
        sdk = output("xcrun", "--sdk", "iphonesimulator", "--show-sdk-version")
        runtime, phone = select_phone(json.loads(output(*SIM, "list", "-j")), sdk)
        report["simulator"] = {"runtime": runtime["identifier"], "version": runtime["version"],
                               "build": runtime.get("buildversion"), "deviceType": phone["identifier"], "sdk": sdk}
        checkpoint("create-device")
        udid = str(uuid.UUID(output(*SIM, "create", "Keyboard-" + stamp, phone["identifier"], runtime["identifier"]))).upper()
        checkpoint("boot-device")
        subprocess.run([*SIM, "boot", udid], check=True)
        subprocess.run([*SIM, "bootstatus", udid, "-b"], check=True)
        # Mark the swipe-typing introduction seen inside this new iPhone only.
        tutorial = subprocess.run([*SIM, "spawn", udid, "defaults", "write", "com.apple.keyboard.preferences",
                                   "DidShowContinuousPathIntroduction", "-bool", "YES"])
        report["tutorialPreferenceWritten"] = tutorial.returncode == 0

        checkpoint("fixture")
        port_file = private / "port"
        server = subprocess.Popen([sys.executable, "-B", HERE / "fixture.py", port_file])
        for _ in range(100):
            if server.poll() is not None:
                raise RuntimeError("The loopback fixture exited during startup")
            if port_file.exists():
                break
            time.sleep(0.1)
        port = int(port_file.read_text())
        if not 0 < port < 65536:
            raise RuntimeError("Invalid bound fixture port")
        project, = HERE.glob("*.xcodeproj")
        scheme, = (project / "xcshareddata/xcschemes").glob("*.xcscheme")
        checkpoint("xctest")
        stopped = None
        try:
            test = subprocess.run(["xcodebuild", "test", "-quiet", "-project", project, "-scheme", scheme.stem,
                                   "-configuration", "Debug", "-destination", f"platform=iOS Simulator,id={udid}",
                                   "-parallel-testing-enabled", "NO", "-maximum-concurrent-test-simulator-destinations", "1",
                                   "-only-testing:MotionUITests/KeyboardTests", "-derivedDataPath", private / "derived",
                                   "-resultBundlePath", result, f"MOTION_UDID={udid}", f"MOTION_PORT={port}",
                                   f"MOTION_RUN_ID={stamp}", "CODE_SIGN_IDENTITY=-"],
                                   timeout=max(0, test_deadline - time.monotonic()))
            report["xcodebuildExit"] = test.returncode
            if test.returncode in (-signal.SIGTERM, -signal.SIGINT):
                report["interruption"] = "cancelled"
        except subprocess.TimeoutExpired:
            report["interruption"] = "xcodebuild-timeout"
        except (SystemExit, KeyboardInterrupt) as exc:
            report["interruption"] = "cancelled"
            stopped = exc
        checkpoint("collect")
        testcases = {}
        # A partially written xcresult may be unavailable. Preserve the app's separate files regardless.
        limit = 2 if report.get("interruption") == "cancelled" else 10
        docs = None
        try:
            container = subprocess.run([*SIM, "get_app_container", udid, BUNDLE, "data"], capture_output=True,
                                       text=True, timeout=limit)
            if container.returncode == 0 and Path(container.stdout.strip()).is_absolute():
                docs = Path(container.stdout.strip()) / "Documents"
        except (OSError, subprocess.TimeoutExpired):
            pass

        def export_cases(outcomes, route_reads):
            for case in CASES:
                path = docs / f"motion-{stamp}-{case}.json" if docs else None
                outcome, failure = outcomes.get(case, ("Unresolved", None))
                try:
                    raw = json.loads(path.read_text()) if path and path.is_file() else {}
                    summary, trace = summarize(case, raw, outcome, route_reads.get(case, {}), udid, runtime["version"])
                except (ValueError, OSError, TypeError, KeyError, AttributeError):
                    summary, trace = {"capture": "incomplete", "xctest": outcome, "source": "unreadable"}, {}
                if docs is None:
                    summary["source"] = "container-unavailable"
                if failure:
                    summary["xctestFailure"] = failure
                report["cases"][case], traces[case] = summary, trace
            save(export / "trace.json", traces)
            checkpoint("collect")

        # Save observations before asking XCTest for a verdict: cancellation can reach either step.
        if docs is None:
            report["caseFiles"] = "container-unavailable"
        export_cases({}, {})
        if result.is_dir():
            try:
                xc = subprocess.run(["xcrun", "xcresulttool", "get", "test-results", "tests", "--path", result],
                                    capture_output=True, text=True, timeout=limit)
                if xc.returncode == 0:
                    testcases = testcase_results(json.loads(xc.stdout))
            except (OSError, ValueError, TypeError, AttributeError, subprocess.TimeoutExpired):
                report["testResults"] = "unreadable"
        opener = build_opener(ProxyHandler({}))
        reads = {}
        report["sameFixtureAsReference"] = False
        try:
            with opener.open(f"http://127.0.0.1:{port}/probe/reads", timeout=min(limit, 5)) as response:
                receipt = json.load(response)
            reads = receipt["cases"]
            digest = receipt["fixtureSha256"]
            if isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest):
                report["fixtureSha256"] = digest
                report["sameFixtureAsReference"] = digest == REFERENCE_DATA
        except (OSError, ValueError, KeyError, TypeError):
            report["routeEvidence"] = "unavailable"
        export_cases(testcases, reads)
        captures = {c["capture"] for c in report["cases"].values()}
        if report.get("xcodebuildExit") == 0 and report["sameFixtureAsReference"] and captures <= {"complete", "observed-with-sampling-gaps"}:
            report["capture"] = "complete" if captures == {"complete"} else "observed-with-sampling-gaps"
        checkpoint("finished")
        if stopped is not None:
            raise stopped
    finally:
        try:
            if server is not None and server.poll() is None:
                server.terminate()
                grace = 1 if report.get("interruption") == "cancelled" else 5
                try:
                    server.wait(timeout=grace)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=grace)
        finally:
            if udid:
                for action in ("shutdown", "delete"):
                    try:
                        subprocess.run([*SIM, action, udid], check=True,
                                       timeout=2 if report.get("interruption") == "cancelled" else 15)
                    except (OSError, subprocess.SubprocessError):
                        cleanup_errors.append(action)
            report["cleanupErrors"] = cleanup_errors
            if not (export / "trace.json").exists():
                save(export / "trace.json", traces)
            save(export / "summary.json", report)
    for case, row in report["cases"].items():
        print(case, row["capture"], "unmet checks", [k for k, ok in row.get("checks", {}).items() if not ok],
              "XCTest", row.get("xctestFailure", row.get("xctest")))
    return 0 if report["capture"] != "incomplete" and not cleanup_errors else 1


if __name__ == "__main__":
    def cancelled(_signum, _frame):
        raise SystemExit("Hosted job interrupted")
    signal.signal(signal.SIGTERM, cancelled)
    sys.exit(run())
