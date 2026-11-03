// Finite, experimental public PID route. NOT imported by the skill. No activation, focus, primer,
// private API, cursor move/restore, clipboard, or global HID fallback. The post API has no delivery result.
// PIDInputProbe <pid> <window_id> <click x y|key>
// click: public mouse-window fields are hypotheses to measure, not a guaranteed window-addressed API.
// key: a single 'k' down/up to the PID; window_id is the intended window only (CGEvent has no key-window selector).
import ApplicationServices
import Foundation

func finish(_ row: [String: Any], _ code: Int32) -> Never {
    let data = try! JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
    FileHandle.standardOutput.write(data + Data([0x0a]))
    exit(code)
}

let args = CommandLine.arguments
guard args.count >= 4, let pid = Int32(args[1]), pid > 0, let wid = UInt32(args[2]), wid > 0,
      (args[3] == "click" && args.count == 6) || (args[3] == "key" && args.count == 4) else {
    finish(["status": "not_posted", "reason": "usage: PIDInputProbe <pid> <window_id> <click x y|key>"], 2)
}
let mode = args[3]
guard let windows = CGWindowListCopyWindowInfo([.optionIncludingWindow], wid) as? [[String: Any]],
      windows.contains(where: { ($0[kCGWindowNumber as String] as? NSNumber)?.uint32Value == wid &&
                                ($0[kCGWindowOwnerPID as String] as? NSNumber)?.int32Value == pid }) else {
    finish(["status": "not_posted", "reason": "window_does_not_belong_to_pid"], 2)
}
guard CGPreflightPostEventAccess() else {
    finish(["status": "not_posted", "reason": "post_events_denied"], 3)
}
guard let source = CGEventSource(stateID: .privateState) else {
    finish(["status": "not_posted", "reason": "no_event_source"], 2)
}
var events: [CGEvent] = []
if mode == "click" {
    guard let x = Double(args[4]), let y = Double(args[5]), x.isFinite, y.isFinite else {
        finish(["status": "not_posted", "reason": "invalid_coordinates"], 2)
    }
    for type in [CGEventType.leftMouseDown, .leftMouseUp] {
        guard let event = CGEvent(mouseEventSource: source, mouseType: type, mouseCursorPosition: CGPoint(x: x, y: y), mouseButton: .left) else {
            finish(["status": "not_posted", "reason": "no_mouse_event"], 2)
        }
        event.setIntegerValueField(.mouseEventClickState, value: 1)
        event.setIntegerValueField(.mouseEventWindowUnderMousePointer, value: Int64(wid))
        event.setIntegerValueField(.mouseEventWindowUnderMousePointerThatCanHandleThisEvent, value: Int64(wid))
        events.append(event)
    }
} else {
    for down in [true, false] {
        guard let event = CGEvent(keyboardEventSource: source, virtualKey: 40, keyDown: down) else {
            finish(["status": "not_posted", "reason": "no_key_event"], 2)
        }
        var units: [UniChar] = [107]  // k, independently recognizable beside decoy letters
        event.keyboardSetUnicodeString(stringLength: 1, unicodeString: &units)
        events.append(event)
    }
}
let began = ProcessInfo.processInfo.systemUptime
for event in events {
    event.flags = []
    event.setIntegerValueField(.eventSourceUserData, value: 42001)
    event.postToPid(pid)
}
finish(["status": "posted_unverified", "route": "CGEventPostToPid", "mode": mode, "pid": pid,
        "intended_window_id": wid, "window_addressing": mode == "click" ? "public_mouse_fields_unverified" : "pid_only",
        "post_count": events.count, "tag": 42001, "began": began,
        "post_ms": (ProcessInfo.processInfo.systemUptime - began) * 1000], 0)
