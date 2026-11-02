// Owned AppKit fixture for tests/test_macos_computer_use_native.py (opt-in; nothing runs it by default).
//
//   QAFixture target <title> <log.jsonl>   field "Amount", Commit, status, count:N, two "Twin" buttons
//   QAFixture front  <title> <log.jsonl>   a text view that takes foreground typing and observes the desktop
//   QAFixture paste  <title> <log.jsonl>   a standard Edit menu (Paste = cmd+v -> paste:), a text view that logs
//                                          every paste action it performs, and a decoy field
//
// stdin, one command per line: "mark <label>" (log a state snapshot), "replace" (target: swap Commit for a
// fresh instance), "type <interval_ms> <letters>" (front: post the letters as keystrokes while it stays the
// active app), "focus <paste|decoy|window>", "clear", "expect <token>" (paste: the clipboard is compared with
// the token, never logged), "quit". EOF quits too, so a fixture never outlives its test.
//
// Every observation is one JSON line in the log, stamped with ProcessInfo.systemUptime (one clock for both
// processes). Both roles log their own activation, key-window changes and every application activation the
// workspace announces, so a focus taken and given back between two samples still shows, and every key event
// it receives itself with its modifiers (none of the keyboard of other apps). The front role also logs mouse
// events and the clipboard's change count, never its contents. The target's snapshots carry Commit's frame in
// global top-left points, the space clicks are sent in. The paste role also logs key-up and modifier events,
// the characters AppKit derived for each key event, the input source in use, every action its menu sent and
// every paste action its text view performed, so a chord that arrived but pasted nothing is told apart from
// one that never arrived.

import AppKit
import Carbon.HIToolbox

let arguments = CommandLine.arguments
guard arguments.count == 4, ["target", "front", "paste"].contains(arguments[1]) else {
    FileHandle.standardError.write(Data("usage: QAFixture <target|front|paste> <title> <log.jsonl>\n".utf8))
    exit(2)
}
let role = arguments[1]
let title = arguments[2]
let logHandle: FileHandle = {
    guard FileManager.default.createFile(atPath: arguments[3], contents: nil),
          let handle = FileHandle(forWritingAtPath: arguments[3]) else {
        FileHandle.standardError.write(Data("cannot write \(arguments[3])\n".utf8))
        exit(2)
    }
    return handle
}()

func log(_ event: String, _ fields: [String: Any] = [:]) {
    var row = fields
    row["event"] = event
    row["role"] = role
    row["pid"] = Int(getpid())
    row["t"] = ProcessInfo.processInfo.systemUptime
    if let data = try? JSONSerialization.data(withJSONObject: row, options: [.sortedKeys]) {
        logHandle.write(data + Data([0x0a]))
    }
}

/// The id of the input source in use now (`keyboard`) and of the keyboard layout under it (`layout`; the same for a
/// plain layout, the layout an input method types through otherwise); "" when unread.
func inputSources() -> [String: String] {
    func identifier(_ source: TISInputSource?) -> String {
        guard let source = source, let raw = TISGetInputSourceProperty(source, kTISPropertyInputSourceID) else { return "" }
        return Unmanaged<CFString>.fromOpaque(raw).takeUnretainedValue() as String
    }
    return ["keyboard": identifier(TISCopyCurrentKeyboardInputSource()?.takeRetainedValue()),
            "layout": identifier(TISCopyCurrentKeyboardLayoutInputSource()?.takeRetainedValue())]
}

/// The paste role's text view: every paste action it performs is logged with who sent it, then performed.
final class PasteView: NSTextView {
    func logged(_ action: String, _ sender: Any?) {
        let item = sender as? NSMenuItem
        log("paste_action", ["action": action, "sender": item.map { "menu:" + $0.title } ?? String(describing: sender.map { type(of: $0) }),
                             "text_before": string, "selection_before": [selectedRange().location, selectedRange().length]])
    }

    override func paste(_ sender: Any?) {
        logged("paste:", sender)
        super.paste(sender)
    }

    override func pasteAsPlainText(_ sender: Any?) {
        logged("pasteAsPlainText:", sender)
        super.pasteAsPlainText(sender)
    }

    override func pasteAsRichText(_ sender: Any?) {
        logged("pasteAsRichText:", sender)
        super.pasteAsRichText(sender)
    }
}

final class Fixture: NSObject, NSApplicationDelegate, NSTextFieldDelegate, NSTextViewDelegate {
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 460, height: 240),
                          styleMask: [.titled], backing: .buffered, defer: false)
    let field = NSTextField(string: "")
    let status = NSTextField(labelWithString: "idle")
    let counter = NSTextField(labelWithString: "count:0")
    let typed = NSTextView(frame: NSRect(x: 20, y: 20, width: 420, height: 200))
    let pasted = PasteView(frame: NSRect(x: 20, y: 60, width: 420, height: 160))
    let decoy = NSTextField(string: "")
    var token: String?
    var commit: NSButton!
    var count = 0
    var pending: [Character] = []
    var typer: Timer?
    var input = ""
    var monitors: [Any] = []

    func button(_ title: String, _ identifier: String, _ action: Selector, _ frame: NSRect, tag: Int = 0) -> NSButton {
        let button = NSButton(title: title, target: self, action: action)
        button.frame = frame
        button.tag = tag
        button.setAccessibilityIdentifier(identifier)
        window.contentView!.addSubview(button)
        return button
    }

    func place(_ view: NSView, _ identifier: String, _ frame: NSRect) {
        view.frame = frame
        view.setAccessibilityIdentifier(identifier)
        window.contentView!.addSubview(view)
    }

    func applicationDidFinishLaunching(_ note: Notification) {
        window.title = title
        let visible = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 800)
        window.setFrameTopLeftPoint(NSPoint(x: visible.minX + (role == "target" ? 40 : 540), y: visible.maxY - 40))
        if role == "target" {
            field.placeholderString = "Amount"
            field.setAccessibilityLabel("Amount")
            field.delegate = self
            place(field, "qa.field", NSRect(x: 20, y: 190, width: 200, height: 24))
            commit = button("Commit", "qa.commit", #selector(commitPressed(_:)), NSRect(x: 240, y: 186, width: 100, height: 32), tag: 1)
            place(status, "qa.status", NSRect(x: 20, y: 150, width: 400, height: 20))
            place(counter, "qa.count", NSRect(x: 20, y: 120, width: 200, height: 20))
            _ = button("Twin", "qa.twinA", #selector(twinPressed(_:)), NSRect(x: 20, y: 70, width: 100, height: 32))
            _ = button("Twin", "qa.twinB", #selector(twinPressed(_:)), NSRect(x: 140, y: 70, width: 100, height: 32))
            window.orderFrontRegardless()  // shown without taking the front: it is acted on in the background
        } else if role == "paste" {
            installEditMenu()
            pasted.delegate = self
            pasted.isRichText = false
            place(pasted, "qa.paste", pasted.frame)
            decoy.placeholderString = "Decoy"
            decoy.delegate = self
            place(decoy, "qa.decoy", NSRect(x: 20, y: 20, width: 200, height: 24))
            // Every key event the app receives, key-up and modifier changes included, with what AppKit made of it.
            let mask: NSEvent.EventTypeMask = [.keyUp, .flagsChanged]
            if let more = NSEvent.addLocalMonitorForEvents(matching: mask, handler: { self.key($0); return $0 }) {
                monitors.append(more)
            }
            NotificationCenter.default.addObserver(forName: NSMenu.didSendActionNotification, object: nil, queue: .main) { note in
                let item = note.userInfo?["MenuItem"] as? NSMenuItem
                log("menu_sent_action", ["title": item?.title ?? "", "key_equivalent": item?.keyEquivalent ?? "",
                                         "modifier_mask": Int(item?.keyEquivalentModifierMask.rawValue ?? 0),
                                         "first_responder": self.responder()])
            }
        } else {
            typed.delegate = self
            place(typed, "qa.typing", typed.frame)
            let mask: NSEvent.EventTypeMask = [.mouseMoved, .leftMouseDown, .leftMouseUp, .leftMouseDragged,
                                               .rightMouseDown, .rightMouseUp, .otherMouseDown, .otherMouseUp, .scrollWheel]
            if let global = NSEvent.addGlobalMonitorForEvents(matching: mask, handler: { self.pointer($0, "global") }) {
                monitors.append(global)
            }
            if let local = NSEvent.addLocalMonitorForEvents(matching: mask, handler: { self.pointer($0, "local"); return $0 }) {
                monitors.append(local)
            }
            log("pointer_monitor", ["installed": monitors.count == 2])
        }
        if let keys = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { self.key($0); return $0 }) {
            monitors.append(keys)
        }
        let center = NotificationCenter.default
        for (name, event) in [(NSApplication.didBecomeActiveNotification, "app_became_active"),
                              (NSApplication.didResignActiveNotification, "app_resigned_active"),
                              (NSWindow.didBecomeKeyNotification, "window_became_key"),
                              (NSWindow.didResignKeyNotification, "window_resigned_key")] {
            center.addObserver(forName: name, object: nil, queue: .main) { _ in log(event, self.state()) }
        }
        NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification,
                                                          object: nil, queue: .main) { note in
            let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication
            log("workspace_activated", ["activated_pid": Int(app?.processIdentifier ?? -1),
                                        "activated_bundle": app?.bundleIdentifier ?? ""])
        }
        if role == "front" || role == "paste" {
            window.makeKeyAndOrderFront(nil)
            window.makeFirstResponder(role == "front" ? typed : pasted)
            NSApp.activate(ignoringOtherApps: true)
        }
        FileHandle.standardInput.readabilityHandler = { handle in
            let data = handle.availableData
            DispatchQueue.main.async { self.receive(data) }
        }
        log("ready", state())
    }

    func state() -> [String: Any] {
        let mouse = NSEvent.mouseLocation
        var out: [String: Any] = [
            "app_active": NSApp.isActive,
            "key_window": window.isKeyWindow,
            "first_responder": responder(),
            "front_pid": Int(NSWorkspace.shared.frontmostApplication?.processIdentifier ?? -1),
            "mouse": [mouse.x.rounded(), mouse.y.rounded()],
        ]
        if role == "target" {
            out["field"] = field.stringValue
            out["count"] = count
            out["status"] = status.stringValue
            out["commit_instance"] = commit?.tag ?? 0
            if let commit = commit, let primary = NSScreen.screens.first {
                let frame = window.convertToScreen(commit.convert(commit.bounds, to: nil))
                out["commit_frame"] = [frame.minX, primary.frame.maxY - frame.maxY, frame.width, frame.height]
                    .map { $0.rounded() }
            }
        } else if role == "paste" {
            out["text"] = pasted.string
            out["selection"] = [pasted.selectedRange().location, pasted.selectedRange().length]
            out["decoy"] = decoy.stringValue
            out["input_source"] = inputSources()
            out["pasteboard_change_count"] = NSPasteboard.general.changeCount
            if let token = token {  // a comparison with the expected token: the contents are never logged
                out["pasteboard_is_token"] = NSPasteboard.general.string(forType: .string) == token
            }
        } else {
            out["text"] = typed.string
            out["pasteboard_change_count"] = NSPasteboard.general.changeCount  // a counter, not the contents
        }
        return out
    }

    /// A standard Edit menu, as an AppKit application template has it: Paste is cmd+v and sends paste: to the first
    /// responder; Paste and Match Style is cmd+opt+shift+v; Find Next (cmd+g) stands in for an application command.
    func installEditMenu() {
        let main = NSMenu()
        let appItem = NSMenuItem()
        appItem.submenu = NSMenu(title: "QAFixture")  // no Quit key: a stray chord must never end the fixture
        main.addItem(appItem)
        let editItem = NSMenuItem()
        let edit = NSMenu(title: "Edit")
        editItem.submenu = edit
        main.addItem(editItem)
        edit.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
        edit.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
        edit.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
        edit.addItem(withTitle: "Paste and Match Style", action: #selector(NSTextView.pasteAsPlainText(_:)),
                     keyEquivalent: "v").keyEquivalentModifierMask = [.command, .option, .shift]
        edit.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
        edit.addItem(withTitle: "Find Next", action: #selector(findNext(_:)), keyEquivalent: "g").target = self
        NSApp.mainMenu = main
    }

    @objc func findNext(_ sender: Any?) {
        log("find_next", ["sender": (sender as? NSMenuItem)?.title ?? "", "first_responder": responder()])
    }

    func responder() -> String {
        let first = window.firstResponder
        if first === typed { return "qa.typing" }
        if field.currentEditor() != nil && first === field.currentEditor() { return "qa.field" }
        if decoy.currentEditor() != nil && first === decoy.currentEditor() { return "qa.decoy" }
        if let view = first as? NSView, !view.accessibilityIdentifier().isEmpty { return view.accessibilityIdentifier() }
        return first.map { String(describing: type(of: $0)) } ?? "none"
    }

    func pointer(_ event: NSEvent, _ scope: String) {
        let at = NSEvent.mouseLocation
        log("pointer_event", ["type": Int(event.type.rawValue), "scope": scope, "mouse": [at.x.rounded(), at.y.rounded()]])
    }

    func key(_ event: NSEvent) {
        let flags = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        let held = [(NSEvent.ModifierFlags.command, "cmd"), (.control, "ctrl"), (.option, "alt"), (.shift, "shift")]
            .filter { flags.contains($0.0) }.map { $0.1 }
        if event.type == .flagsChanged {  // a modifier went down or up: it has no characters to read
            log("flags_changed", ["key_code": Int(event.keyCode), "modifiers": held, "flags": Int(flags.rawValue),
                                  "input_source": inputSources()])
            return
        }
        var row: [String: Any] = ["chars_ignoring": event.charactersIgnoringModifiers ?? "", "key_code": Int(event.keyCode),
                                  "modifiers": held]
        if role == "paste" {
            row.merge(["characters": event.characters ?? "", "flags": Int(flags.rawValue), "repeat": event.isARepeat,
                       "input_source": inputSources(), "first_responder": responder()]) { $1 }
        }
        log(event.type == .keyUp ? "key_up" : "key_down", row)
    }

    @objc func commitPressed(_ sender: NSButton) {
        count += 1
        status.stringValue = "committed:" + field.stringValue
        counter.stringValue = "count:\(count)"
        log("commit", ["instance": sender.tag, "field": field.stringValue, "count": count,
                       "app_active": NSApp.isActive, "key_window": window.isKeyWindow])
    }

    @objc func twinPressed(_ sender: NSButton) {
        log("twin", ["identifier": sender.accessibilityIdentifier(), "app_active": NSApp.isActive])
    }

    func controlTextDidChange(_ note: Notification) {
        // Every edit through the field editor lands here: keystrokes, and also a value set through Accessibility
        // while the field holds the field editor (seen natively); key_down rows tell keystrokes apart.
        if (note.object as AnyObject?) === decoy {
            log("decoy_text", ["decoy": decoy.stringValue])
            return
        }
        log("field_typed", ["field": field.stringValue])
    }

    func textDidChange(_ note: Notification) {
        if (note.object as AnyObject?) === pasted {
            log("paste_text", ["text": pasted.string, "selection": [pasted.selectedRange().location, pasted.selectedRange().length]])
            return
        }
        log("front_text", ["text": typed.string])
    }

    func receive(_ data: Data) {
        if data.isEmpty {
            FileHandle.standardInput.readabilityHandler = nil
            quit()
            return
        }
        input += String(decoding: data, as: UTF8.self)
        while let newline = input.firstIndex(of: "\n") {
            let line = String(input[..<newline])
            input.removeSubrange(...newline)
            command(line.split(separator: " ", maxSplits: 2).map(String.init))
        }
    }

    func command(_ words: [String]) {
        switch words.first ?? "" {
        case "mark":
            log("mark", state().merging(["label": words.count > 1 ? words[1] : ""]) { $1 })
        case "replace" where role == "target":
            let fresh = button("Commit", "qa.commit", #selector(commitPressed(_:)), commit.frame, tag: commit.tag + 1)
            commit.removeFromSuperview()
            commit = fresh
            log("commit_replaced", ["instance": fresh.tag])
        case "type" where role == "front" && words.count == 3:
            startTyping(Int(words[1]) ?? 60, words[2])
        case "focus" where role == "paste" && words.count == 2:
            let views: [String: NSResponder] = ["paste": pasted, "decoy": decoy]  // "window": the window itself
            log("focused", ["asked": words[1], "made": window.makeFirstResponder(views[words[1]]), "first_responder": responder()])
        case "clear" where role == "paste":
            pasted.string = ""
            decoy.stringValue = ""
            log("cleared", state())
        case "expect" where role == "paste" && words.count == 2:
            token = words[1]
            log("expecting", ["pasteboard_is_token": NSPasteboard.general.string(forType: .string) == token])
        case "quit":
            quit()
        default:
            log("unknown_command", ["words": words])
        }
    }

    func startTyping(_ intervalMs: Int, _ text: String) {
        guard CGPreflightPostEventAccess() else {
            log("typing_unsupported", ["reason": "post_events_denied"])
            return
        }
        pending = Array(text.filter { $0.isASCII && $0.isLetter })
        log("typing_started", ["chars": pending.count, "interval_ms": intervalMs])
        typer = Timer.scheduledTimer(withTimeInterval: Double(max(intervalMs, 10)) / 1000, repeats: true) { _ in
            self.typeNext()
        }
    }

    func typeNext() {
        if pending.isEmpty {
            stopTyping("typing_finished")
            return
        }
        // A posted keystroke goes to whichever app the window server finds active: post only while this one
        // is, so a stolen focus stops the typing (and says so) rather than typing into another app's window.
        guard NSApp.isActive, window.isKeyWindow, window.firstResponder === typed else {
            stopTyping("typing_stopped")
            return
        }
        var units = Array(String(pending.removeFirst()).utf16)
        for down in [true, false] {
            guard let event = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: down) else { continue }
            event.keyboardSetUnicodeString(stringLength: units.count, unicodeString: &units)
            event.post(tap: .cghidEventTap)
        }
    }

    func stopTyping(_ event: String) {
        typer?.invalidate()
        typer = nil
        log(event, state().merging(["left": pending.count]) { $1 })
        pending = []
    }

    func quit() {
        monitors.forEach(NSEvent.removeMonitor)
        monitors = []
        log("quit")
        NSApp.terminate(nil)
    }
}

let app = NSApplication.shared
let fixture = Fixture()
app.delegate = fixture
app.setActivationPolicy(.regular)
app.run()
