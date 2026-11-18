import UIKit
import WebKit

@main final class App: UIResponder, UIApplicationDelegate {}

final class Scene: UIResponder, UIWindowSceneDelegate {
    var window: UIWindow?
    func scene(_ scene: UIScene, willConnectTo session: UISceneSession, options: UIScene.ConnectionOptions) {
        guard let scene = scene as? UIWindowScene else { return }
        let window = UIWindow(windowScene: scene)
        window.rootViewController = MotionHost()
        self.window = window
        window.makeKeyAndVisible()
    }
}

final class MotionHost: UIViewController, WKScriptMessageHandler, WKNavigationDelegate {
    private let save = UIButton(type: .system)
    private var web: WKWebView!
    private var replay: URL?
    private var file: URL?
    private var flags: [String] = []
    private var native: [[String: Any]] = []
    private var report: [String: Any]?
    private var exporting = false
    private let device = ProcessInfo.processInfo.environment["SIMULATOR_UDID"] ?? ""
    private let scenario = ProcessInfo.processInfo.environment["MOTION_CASE"] ?? ""
    private var aim = ""

    override func viewDidLoad() {
        super.viewDidLoad()
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .nonPersistent()
        config.userContentController.add(self, name: "reading")
        web = WKWebView(frame: view.bounds, configuration: config)
        web.navigationDelegate = self
        web.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        web.scrollView.contentInsetAdjustmentBehavior = .never
        view.addSubview(web)

        // Independent of the document and its layout. XCTest uses this button only to finish an incomplete trace.
        save.setTitle("Trace", for: .normal)
        save.accessibilityIdentifier = "save-trace"
        save.addTarget(self, action: #selector(finishTrace), for: .touchUpInside)
        save.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(save)
        NSLayoutConstraint.activate([save.trailingAnchor.constraint(equalTo: view.safeAreaLayoutGuide.trailingAnchor),
                                     save.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor)])

        let info = Bundle.main.infoDictionary ?? [:]
        guard let expected = info["MotionDevice"] as? String, let id = UUID(uuidString: expected),
              UUID(uuidString: device) == id, let run = info["MotionRun"] as? String, UUID(uuidString: run) != nil,
              ["main-reading", "main-latest", "project-ready", "project-held"].contains(scenario),
              let portText = info["MotionPort"] as? String, let port = UInt16(portText), port > 0 else {
            flag("wrong-destination-or-configuration")
            return
        }
        file = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("motion-\(run)-\(scenario).json")
        replay = URL(string: "http://127.0.0.1:\(port)/?case=\(scenario)")
        flag("started")
        _ = persist()
        guard let script = Bundle.main.url(forResource: "Reading", withExtension: "js"),
              let source = try? String(contentsOf: script, encoding: .utf8) else {
            fail("reading-script-unavailable"); return
        }
        web.configuration.userContentController.addUserScript(WKUserScript(source: source, injectionTime: .atDocumentEnd,
                                                                          forMainFrameOnly: true))
        for name in [UIResponder.keyboardWillShowNotification, UIResponder.keyboardWillChangeFrameNotification,
                     UIResponder.keyboardDidChangeFrameNotification, UIResponder.keyboardDidShowNotification,
                     UIResponder.keyboardWillHideNotification, UIResponder.keyboardDidHideNotification] {
            NotificationCenter.default.addObserver(self, selector: #selector(keyboard(_:)), name: name, object: nil)
        }
        let rules = """
        [{"trigger":{"url-filter":"^https?://"},"action":{"type":"block"}},
         {"trigger":{"url-filter":"^wss?://"},"action":{"type":"block"}},
         {"trigger":{"url-filter":"^http://127[.]0[.]0[.]1:\(port)/"},"action":{"type":"ignore-previous-rules"}},
         {"trigger":{"url-filter":"^ws://127[.]0[.]0[.]1:\(port)/"},"action":{"type":"ignore-previous-rules"}}]
        """
        WKContentRuleListStore.default().compileContentRuleList(forIdentifier: "fixture-loopback",
                                                               encodedContentRuleList: rules) { list, _ in
            guard let list, let url = self.replay else { self.fail("loopback-rules-unavailable"); return }
            self.web.configuration.userContentController.add(list)
            self.web.load(URLRequest(url: url))
        }
    }

    func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                 decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        decisionHandler(action.targetFrame?.isMainFrame == true && action.request.url == replay ? .allow : .cancel)
    }

    private func flag(_ name: String) {
        if !flags.contains(name) { flags.append(name) }
        save.accessibilityValue = "|" + flags.joined(separator: "|") + "|" + aim
    }

    func userContentController(_ controller: WKUserContentController, didReceive message: WKScriptMessage) {
        if message.frameInfo.isMainFrame, message.frameInfo.request.url == replay,
           let body = message.body as? [String: Any], let phase = body["phase"] as? String,
           ["latest", "reading", "project-ready", "project-held", "entry-touch", "wrong-entry",
            "field-focus", "focused-history", "field-input"].contains(phase) {
            if let xy = body["aim"] as? [NSNumber], xy.count == 2 {
                let local = CGPoint(x: xy[0].doubleValue, y: xy[1].doubleValue)
                let point = web.convert(local, to: view.window)
                aim = "\(point.x),\(point.y)"
            }
            flag(phase)
            _ = persist()
        }
    }

    @objc private func keyboard(_ note: Notification) {
        let ms = ProcessInfo.processInfo.systemUptime * 1000
        guard report == nil, let screen = web.window?.windowScene?.screen else { return }
        let data = note.userInfo ?? [:]
        var event: [String: Any] = ["sequence": native.count, "name": note.name.rawValue, "uptimeMs": ms]
        event["durationSeconds"] = data[UIResponder.keyboardAnimationDurationUserInfoKey]
        event["animationCurve"] = data[UIResponder.keyboardAnimationCurveUserInfoKey]
        for (label, key) in [("begin", UIResponder.keyboardFrameBeginUserInfoKey),
                             ("end", UIResponder.keyboardFrameEndUserInfoKey)] {
            guard let screenRect = (data[key] as? NSValue)?.cgRectValue else { continue }
            let r = web.convert(screenRect, from: screen.coordinateSpace)
            event[label] = [r.origin.x, r.origin.y, r.width, r.height]
            if label == "end" {
                let overlap = web.bounds.intersection(r)
                event["overlapPt"] = overlap.isNull ? 0 : overlap.height
                if note.name == UIResponder.keyboardDidShowNotification && !overlap.isNull && overlap.height > 100 {
                    flag("did-show")
                }
            }
        }
        native.append(event)
        web.callAsyncJavaScript("window.motion?.uikit(sequence, name)",
                                arguments: ["sequence": native.count - 1, "name": note.name.rawValue],
                                in: nil, in: .page, completionHandler: nil)
        if note.name == UIResponder.keyboardDidHideNotification && flags.contains("did-show") {
            flag("did-hide")
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { self.finishTrace() }
        }
        _ = persist()
    }

    private func persist() -> Bool {
        guard let file else { return false }
        var output: [String: Any] = ["device": device, "scenario": scenario, "ios": UIDevice.current.systemVersion,
                                     "milestones": flags, "uikit": native,
                                     "recorder": NSNull()]
        if let report { output["recorder"] = report }
        do {
            try JSONSerialization.data(withJSONObject: output).write(to: file, options: .atomic)
            return true
        } catch {
            flag("write-error")
            return false
        }
    }

    private func fail(_ reason: String) {
        flag(reason)
        flag("partial")
        _ = persist()
    }

    @objc private func finishTrace() {
        guard file != nil, !exporting, report == nil else { return }
        exporting = true
        web.evaluateJavaScript("window.motion ? JSON.stringify(motion.stop()) : null") { value, _ in
            self.exporting = false
            guard let json = value as? String, let parsed = try? JSONSerialization.jsonObject(with: Data(json.utf8)),
                  let trace = parsed as? [String: Any], let log = trace["log"] as? [[String: Any]] else {
                self.fail("recorder-unavailable")
                return
            }
            self.report = trace
            let delivered = log.filter { $0["event"] as? String == "uikit" }.compactMap { $0["name"] as? String }
            if delivered.contains(UIResponder.keyboardDidShowNotification.rawValue) &&
                delivered.contains(UIResponder.keyboardDidHideNotification.rawValue) { self.flag("joined") }
            if self.persist() { self.flag("saved"); _ = self.persist() }
        }
    }
}
