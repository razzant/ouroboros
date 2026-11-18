import XCTest

final class KeyboardTests: XCTestCase {
    @MainActor private func waitFor(_ phase: String, on trace: XCUIElement, seconds: TimeInterval = 20) -> Bool {
        let condition = NSPredicate(format: "value CONTAINS %@", "|\(phase)|")
        return XCTWaiter.wait(for: [XCTNSPredicateExpectation(predicate: condition, object: trace)],
                              timeout: seconds) == .completed
    }

    // One fresh nonpersistent WKWebView per case. The same XCTest owns the real finger and the AX keyboard check.
    @MainActor private func observe(_ scenario: String) throws {
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launchEnvironment["MOTION_CASE"] = scenario
        app.launch()
        let trace = app.buttons["save-trace"]
        addTeardownBlock {
            await MainActor.run {
                if app.state == .runningForeground && trace.exists && !(trace.value as? String ?? "").contains("|saved|") {
                    trace.tap()
                    _ = self.waitFor("saved", on: trace, seconds: 4)
                }
            }
        }
        let window = app.windows.firstMatch
        let web = app.webViews.firstMatch
        func readOlder(above bottom: CGFloat, distance: CGFloat) {
            let frame = web.frame.intersection(window.frame)
            let height = min(bottom, frame.maxY) - frame.minY
            let relative = CGVector(dx: frame.minX - window.frame.minX + frame.width * 0.38,
                                    dy: frame.minY - window.frame.minY + height * 0.40)
            let finger = window.coordinate(withNormalizedOffset: .zero).withOffset(relative)
            finger.press(forDuration: 0.06, thenDragTo: finger.withOffset(CGVector(dx: 0, dy: distance)),
                         withVelocity: .slow, thenHoldForDuration: 0.18)
        }

        let initial = scenario.hasPrefix("main-") ? "latest" : scenario
        XCTAssertTrue(waitFor(initial, on: trace, seconds: 90), "[probe:initial-position]")
        if scenario == "main-reading" {
            readOlder(above: window.frame.maxY, distance: 100)
            XCTAssertTrue(waitFor("reading", on: trace), "[probe:reading-position]")
        }
        let value = try XCTUnwrap(trace.value as? String, "[probe:target-coordinates]")
        let aim = try XCTUnwrap(value.components(separatedBy: "|").last, "[probe:target-coordinates]")
        let xy = aim.split(separator: ",").compactMap { Double($0) }
        guard xy.count == 2 else { return XCTFail("[probe:target-coordinates]") }
        let point = CGPoint(x: xy[0], y: xy[1])
        XCTAssertTrue(web.frame.contains(point), "[probe:target-offscreen]")
        window.coordinate(withNormalizedOffset: .zero).withOffset(
            CGVector(dx: point.x - window.frame.minX, dy: point.y - window.frame.minY)).tap()
        XCTAssertTrue(waitFor("entry-touch", on: trace), "[probe:entry-touch]")
        XCTAssertTrue(waitFor("field-focus", on: trace), "[probe:field-focus]")

        let keyboard = app.keyboards.firstMatch
        XCTAssertTrue(keyboard.waitForExistence(timeout: 30), "[probe:no-software-keyboard]")
        XCTAssertTrue(waitFor("did-show", on: trace), "[probe:no-did-show]")
        XCTAssertGreaterThan(keyboard.frame.intersection(window.frame).height, 100, "[probe:keyboard-offscreen]")
        XCTAssertTrue(waitFor("focused-history", on: trace), "[probe:focused-history]")
        let letters = keyboard.keys.matching(NSPredicate(format: "label MATCHES %@", "\\p{L}"))
        let key = try XCTUnwrap(letters.allElementsBoundByIndex.first { $0.isHittable }, "[probe:no-hittable-letter]")
        key.tap()
        XCTAssertTrue(waitFor("field-input", on: trace), "[probe:no-field-input]")
        readOlder(above: keyboard.frame.minY, distance: 72)
        XCTAssertTrue(waitFor("did-hide", on: trace), "[probe:no-did-hide]")
        XCTAssertTrue(waitFor("saved", on: trace), "[probe:trace-unsaved]")
        XCTAssertTrue(waitFor("joined", on: trace, seconds: 1), "[probe:unjoined-notifications]")
        XCTAssertFalse(keyboard.exists, "[probe:keyboard-still-present]")
    }

    @MainActor func testMainReading() throws { try observe("main-reading") }
    @MainActor func testMainLatest() throws { try observe("main-latest") }
    @MainActor func testProjectReady() throws { try observe("project-ready") }
    @MainActor func testProjectHeld() throws { try observe("project-held") }
}
