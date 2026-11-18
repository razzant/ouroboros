"""Export only numeric geometry, known test events and synthetic row identities."""
import math
import re

CASES = {
    "main-reading": "testMainReading", "main-latest": "testMainLatest",
    "project-ready": "testProjectReady", "project-held": "testProjectHeld",
}
SHOW = "UIKeyboardDidShowNotification"
HIDE = "UIKeyboardDidHideNotification"
WILL_SHOW = "UIKeyboardWillShowNotification"
WILL_HIDE = "UIKeyboardWillHideNotification"
NOTICES = {SHOW, HIDE, WILL_SHOW, WILL_HIDE,
           "UIKeyboardWillChangeFrameNotification", "UIKeyboardDidChangeFrameNotification"}
MARKS = {"start", "stop", "position", "transcript", "history-held", "release", "activation-touch",
         "focusin", "focusout", "uikit", "focused-history", "field-input", "dismiss-touch"}
MILESTONES = {"started", "latest", "reading", "project-ready", "project-held", "entry-touch", "wrong-entry",
              "field-focus", "focused-history", "field-input", "did-show", "did-hide", "partial", "joined", "write-error", "saved",
              "reading-script-unavailable", "loopback-rules-unavailable", "recorder-unavailable"}
FAILURE_REASONS = {"initial-position", "reading-position", "target-coordinates", "target-offscreen", "entry-touch",
                   "field-focus", "no-software-keyboard", "no-did-show", "keyboard-offscreen", "focused-history",
                   "no-hittable-letter", "no-field-input", "no-did-hide", "trace-unsaved", "unjoined-notifications",
                   "keyboard-still-present"}
# A hole this large leaves a visible part of a typical keyboard transition unobserved.
SAMPLING_GAP_MS = 100
GEOMETRY = ("vvH", "offsetTop", "offsetLeft", "scale", "innerH", "scrollY", "top", "extent", "height", "gap", "revision")
BOXES = ("bar", "feed", "field", "anchor", "line", "composer")


def number(value):
    return value if type(value) in (float, int) and math.isfinite(value) else None


def vector(value, length):
    if isinstance(value, list) and len(value) == length and all(number(v) is not None for v in value):
        return value
    return None


def geometry(frame):
    clean = {k: number(frame.get(k)) for k in GEOMETRY}
    clean.update((k, vector(frame.get(k), 2)) for k in BOXES)
    for key in ("keyboard", "focused", "quoted", "sameNode", "hydrated"):
        clean[key] = frame.get(key) if type(frame.get(key)) is bool else None
    return clean


def clean_trace(raw):
    native, timeline = [], []
    for row in raw.get("uikit", []):
        if row.get("name") not in NOTICES:
            continue
        native.append({"name": row["name"], **{k: number(row.get(k)) for k in
                       ("sequence", "uptimeMs", "durationSeconds", "animationCurve", "overlapPt")},
                       **{k: vector(row.get(k), 4) for k in ("begin", "end")}})
    for row in (raw.get("recorder") or {}).get("log", []):
        t = number(row.get("t"))
        if t is None:
            continue
        if "frame" in row and type(row.get("n")) is int and row["n"] > 0:
            timeline.append({"t": t, "n": row["n"], "frame": geometry(row["frame"])})
        elif row.get("event") in MARKS:
            event = {"t": t, "event": row["event"]}
            if row.get("phase") in {"latest", "reading", "project-held", "project-ready"}:
                event["phase"] = row["phase"]
            if row.get("name") in NOTICES:
                event["name"] = row["name"]
            if row.get("replyCenter") in {"reply", "msg-time", "other"}:
                event["replyCenter"] = row["replyCenter"]
            for key in ("revision", "sequence", "pending", "gap", "x", "y"):
                if number(row.get(key)) is not None:
                    event[key] = row[key]
            for key in ("accepted", "hydrated"):
                if type(row.get(key)) is bool:
                    event[key] = row[key]
            if "ids" in row:
                event["ids"] = [s if isinstance(s, str) and re.fullmatch(r"chat:\d{1,3}", s) else "unexpected"
                                for s in row["ids"]]
            if "geometry" in row:
                event["geometry"] = geometry(row["geometry"])
            timeline.append(event)
    return {"uikit": native, "timeline": timeline}


def testcase_results(tree):
    """Read executed leaf outcomes, including a duplicate or mixed result as unresolved."""
    results = {case: [] for case in CASES}

    reasons = {}
    methods = {method: case for case, method in CASES.items()}

    def visit(node, case=None):
        if node.get("nodeType") == "Test Case":
            identifier = node.get("nodeIdentifier", "").split("/")[-1].removesuffix("()")
            case = methods.get(identifier)
            if case:
                result = node.get("result", node.get("testResult"))
                results[case].append(result if result in {"Passed", "Failed", "Skipped"} else "Unknown")
        elif case and node.get("nodeType") == "Failure Message":
            tag = re.search(r"\[probe:([a-z-]+)\]", node.get("name", ""))
            if tag and tag[1] in FAILURE_REASONS:
                reasons.setdefault(case, tag[1])
        for child in node.get("children", []):
            visit(child, case)

    for root in tree.get("testNodes", []):
        visit(root)
    return {case: (rows[0] if len(rows) == 1 else "Unresolved",
                   reasons.get(case, "unclassified") if rows == ["Failed"] else None)
            for case, rows in results.items()}


def minus(a, b):
    return round(a - b, 2) if a is not None and b is not None else None


def positions(frame):
    def coord(key, index=0):
        return frame[key][index] if frame[key] else None
    anchor_feed = minus(coord("anchor"), coord("feed"))
    return {
        "anchorVisualY": minus(coord("anchor"), frame["offsetTop"]),
        "lineVisualY": minus(coord("line"), frame["offsetTop"]),
        "anchorInContentY": anchor_feed + frame["top"] if anchor_feed is not None and frame["top"] is not None else None,
        "fieldVisualY": minus(coord("field"), frame["offsetTop"]),
        "fieldBottomFromVisibleBottom": minus(minus(coord("field", 1), frame["offsetTop"]), frame["vvH"]),
    }


def switches(frames):
    edges = []
    for old, new in zip(frames, frames[1:]):
        a, b = old["frame"], new["frame"]
        if a["keyboard"] is None or b["keyboard"] is None or a["keyboard"] == b["keyboard"]:
            continue
        before, after = positions(a), positions(b)
        gap = minus(new["t"], old["t"])
        edges.append({"keyboardOpen": b["keyboard"], "frames": [old["n"], new["n"]], "timeMs": [old["t"], new["t"]],
                      "gapMs": gap, "samplingGap": gap >= SAMPLING_GAP_MS,
                      "vvHeight": [a["vvH"], b["vvH"]], "offsetTop": [a["offsetTop"], b["offsetTop"]],
                      "anchorVisualY": [before["anchorVisualY"], after["anchorVisualY"]],
                      "lineVisualY": [before["lineVisualY"], after["lineVisualY"]],
                      "fieldVisualY": [before["fieldVisualY"], after["fieldVisualY"]],
                      "sameRevision": a["revision"] == b["revision"], "sameAnchorNode": a["sameNode"] and b["sameNode"],
                      "deltaCssPx": {"vvHeight": minus(b["vvH"], a["vvH"]), "offsetTop": minus(b["offsetTop"], a["offsetTop"]),
                                     "scrollTop": minus(b["top"], a["top"]),
                                     **{key: minus(after[key], value) for key, value in before.items()}}})
    return edges


def phase_coverage(notices, will_name, did, delivery, frames):
    """Measure between WebKit's receipts on its clock. UIKit's interval has its own clock."""
    will = {}
    for notice in reversed(notices):
        if notice["name"] in (SHOW, HIDE):
            break
        if notice["name"] == will_name:
            will = notice  # Include the whole interval if UIKit announces the change twice.
    native_ms = minus(did.get("uptimeMs"), will.get("uptimeMs"))
    begin, end = delivery(will).get("t"), delivery(did).get("t")
    result = {"status": "window-unavailable", "nativeSequences": [will.get("sequence"), did.get("sequence")],
              "nativeIntervalMs": native_ms, "deliveryWindowMs": [begin, end],
              "rAFCount": None, "largestGapMs": None}
    if begin is None or end is None or end < begin:
        return result
    times = [r["t"] for r in frames if begin < r["t"] < end]
    points = [begin, *times, end]
    gap = max(b - a for a, b in zip(points, points[1:]))
    sampled = (len(times) >= 2 and gap < SAMPLING_GAP_MS and native_ms is not None and
               abs(native_ms - (end - begin)) < SAMPLING_GAP_MS)
    result.update(status="sampled" if sampled else "sampling-gap", rAFCount=len(times), largestGapMs=gap)
    return result


def summarize(case, raw, testcase, reads, device, expected_ios):
    trace = clean_trace(raw)
    log, native = trace["timeline"], trace["uikit"]
    frames = [r for r in log if "frame" in r]
    by_event = lambda kind: [r for r in log if r.get("event") == kind]
    touch = next(iter(by_event("activation-touch")), {})
    start = touch.get("t", math.inf)
    stop = next(iter(by_event("stop")), {})
    end = stop.get("t", -1)
    ready = next(iter(by_event("focused-history")), {})
    ready_time = ready.get("t", math.inf)
    project = case.startswith("project")
    held = case == "project-held"
    first = 20 if project else 0
    wanted_ids = [f"chat:{i}" for i in range(first, first + 10)]
    snapshots = by_event("transcript")
    before = touch.get("geometry", {})
    ready_frame = ready.get("geometry", {})
    joined = {(r["sequence"], r["name"]): r for r in by_event("uikit") if "sequence" in r and "name" in r}
    def delivery(notice):
        return joined.get((notice.get("sequence"), notice.get("name")), {})
    visible_shows = [r for r in native if r["name"] == SHOW and (r["overlapPt"] or 0) > 100 and r["begin"] and r["end"]]
    show = next((r for r in visible_shows if delivery(r).get("t", -1) >= start), next(iter(visible_shows), {}))
    after_show = native[native.index(show) + 1:] if show else native
    hide = next((r for r in after_show if r["name"] == HIDE), {})
    show_time = delivery(show).get("t", math.inf)
    hide_time = delivery(hide).get("t", -math.inf)
    enter = phase_coverage(native[:native.index(show)] if show else native, WILL_SHOW, show, delivery, frames)
    exit_phase = phase_coverage(after_show[:after_show.index(hide)] if hide else after_show, WILL_HIDE, hide, delivery, frames)
    after_hide_rafs = sum(r["t"] > hide_time for r in frames) if delivery(hide) else None
    expected_version = tuple(int(n) for n in expected_ios.split(".")) if expected_ios else ()
    version = raw.get("ios", "")
    actual_version = tuple(int(n) for n in version.split(".")) if re.fullmatch(r"\d+(\.\d+){1,2}", version) else ()
    observed_gap = touch.get("gap")
    gap_ok = observed_gap is not None and (observed_gap > 48 if case == "main-reading" else abs(observed_gap) <= 6)
    required_routes = {"document", "/static/app.js", "/static/style.css", "/ws", "/api/onboarding",
                       "/api/state", "/api/chat/history/1"}
    if project:
        required_routes.update({"/api/projects/by-id", "/api/chat/history/101"})
    routes = {k: reads.get(k, 0) for k in sorted(required_routes)}
    focus_at_touch = any(start <= r["t"] <= hide_time for r in by_event("focusin"))
    dismiss = next((r for r in by_event("dismiss-touch") if show_time <= r["t"] <= hide_time), {})
    after_hide = bool(delivery(hide) and end >= hide_time and stop.get("geometry", {}).get("focused") is False)
    held_in_order = (any(r["t"] <= start for r in by_event("history-held")) and
                     any(show_time <= r["t"] <= ready_time and r.get("pending", 0) > 0 for r in by_event("release")))
    times = [r["uptimeMs"] for r in native]
    native_clock = all(t is not None for t in times) and times == sorted(times) and all(
        r["sequence"] == i for i, r in enumerate(native))
    def measured(f):
        return all(f.get(k) is not None for k in ("vvH", "offsetTop", "feed", "field", "composer"))
    real_case = (str(raw.get("device", "")).upper() == device.upper() and raw.get("scenario") == case and
                 (raw.get("recorder") or {}).get("scenario") == case and
                 (raw.get("recorder") or {}).get("anchorKey") == wanted_ids[7] and
                 bool(expected_version and actual_version) and
                 (expected_version + (0, 0))[:3] == (actual_version + (0, 0))[:3])
    checks = {
        "deviceRuntimeRoom": real_case, "servedSpaApiWs": all(n > 0 for n in routes.values()),
        "xctestCycleWithHittableKey": testcase == "Passed",
        "trustedTouchInPosition": touch.get("accepted") is True and gap_ok and before.get("focused") is False and before.get("keyboard") is False,
        "entryHistory": (touch.get("pending", 0) > 0 and before.get("anchor") is None and touch.get("hydrated") is False
                         if held else touch.get("hydrated") is True and before.get("sameNode") is True),
        "joinedDidShowHide": bool(hide.get("end") and native_clock and start <= show_time <= hide_time),
        "trustedInputBeforeDismissal": any(ready_time <= r["t"] <= dismiss.get("t", -1) for r in by_event("field-input")),
        "focusedAndDismissedFromFeed": focus_at_touch and start <= ready_time < hide_time and bool(dismiss) and after_hide,
        "measuredAtTouchShowHideStop": all(measured(f) for f in
            (before, delivery(show).get("geometry", {}), delivery(hide).get("geometry", {}), stop.get("geometry", {}))),
        "focusedLoadedHistory": ready_frame.get("focused") is True and ready_frame.get("sameNode") is True and ready_frame.get("line") is not None and
            ready_frame.get("hydrated") is True and (project or ready_frame.get("quoted") is True),
    }
    if held:
        checks["historyReleasedAfterDidShow"] = held_in_order
    first_full = next((r for r in snapshots if r["t"] >= show_time and r.get("ids") == wanted_ids), {})
    revision = first_full.get("revision") if held else before.get("revision")
    data_from = first_full.get("t", math.inf) if held else start
    samples = [(r["t"], r.get("frame", r.get("geometry"))) for r in log if "frame" in r or "geometry" in r]
    stable_samples = [f for t, f in samples if t >= data_from]
    exact_ids = any(r.get("revision") == revision and r.get("ids") == wanted_ids for r in snapshots)
    later_revisions = {r.get("revision") for r in snapshots if r["t"] >= data_from}
    changed_data = exact_ids and (bool(later_revisions - {None, revision}) or any(
        f["sameNode"] is False or (f["revision"] is not None and f["revision"] != revision)
        for f in stable_samples))
    stable_data = bool(stable_samples and exact_ids and later_revisions <= {revision} and all(
        f["revision"] == revision and f["sameNode"] is True for f in stable_samples))
    checks["sameRowsWordsAndAnchorNode"] = stable_data
    edges = [edge for edge in switches(frames) if start <= edge["timeMs"][1]]
    span = lambda key: [min(values), max(values)] if (values := [f[key] for t, f in samples
                        if t >= start and f[key] is not None]) else None
    capture = "incomplete"
    if all(checks.values()):
        capture = ("complete" if enter["status"] == exit_phase["status"] == "sampled" and after_hide_rafs
                   and not any(edge["samplingGap"] for edge in edges)
                   else "observed-with-sampling-gaps")
    result = {"capture": capture,
              "xctest": testcase, "checks": checks, "sameRowsWordsAndAnchorNode": stable_data,
              "historyStability": "stable" if stable_data else "changed" if changed_data else "unverified",
              "milestones": [name for name in raw.get("milestones", []) if name in MILESTONES],
              "unservedApiReads": reads.get("unserved-api-reads", 0),
              "initialGapCssPx": observed_gap, "historyAnchor": wanted_ids[7], "routes": routes,
              "ios": version if actual_version else None, "keyboardOverlapPt": show["overlapPt"] if show else None,
              "sampling": {"entry": enter, "exit": exit_phase, "rAFsAfterDidHideDelivery": after_hide_rafs},
              "vvHeightRangeCssPx": span("vvH"), "offsetTopRangeCssPx": span("offsetTop"),
              "classSwitches": edges}
    if not project:
        result["replyCenterAtTouch"] = touch.get("replyCenter")
    return result, trace
