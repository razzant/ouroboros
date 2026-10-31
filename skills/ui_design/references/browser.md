# Render, look, measure

Look before you measure. A screenshot is a picture you must actually open; a measurement is evidence, never a verdict.
`W×H` below is the viewer's window from the brief or the owner-client facts, not a constant.

## Ouroboros, with its own tools

- `browse_page(url, output="screenshot", viewport="WxH", engine="webkit")` renders the page and attaches the pixels to your context in the same round. `file://` paths work. Name the engine you used: WebKit shows what a native macOS window shows but may download on first use; Chromium is bundled.
- `browser_action(action="screenshot")` keeps the capture in the browser session but does not show it to you; follow it with `analyze_screenshot`, or take the picture with `browse_page` instead.
- `browser_action(action="evaluate", value=<the text of measure.js>)` calls the snippet on the current page and returns its facts and findings. Take the screenshot first so the page has painted; WebKit can report one width before the first paint and another after. The tool cuts any result at 20,000 characters, so a frozen page never comes back whole this way: freeze with the script below and write the frame to a file.
- Detail: judge it on a crop at device scale 2 from the script below, never on a stitched or scaled sheet.

## A script, for Ouroboros or a delegated session

- Open the delivered file the way the owner will: copy the lone file into an empty folder and open it from there, so anything it still fetches shows up as a failed request.
- Viewport and engine come from the brief (the viewer's window, device scale factor 2, their engine). `PLAYWRIGHT_BROWSERS_PATH` or the host facts in your instructions name an existing browser cache; make one bounded check for a Python or Node that imports Playwright, and stop there. No disk hunting.
- Screenshot, then read the PNG yourself with your native image tool. Then evaluate the measurement snippet. Then crops of the compared element.
- A readonly session that cannot run a browser says so and returns the file for the parent to look at. Never write "I looked" about a picture you did not see.

### Python

```python
import json, pathlib, shutil, tempfile
from playwright.sync_api import sync_playwright

W, H = 1280, 800                                                            # the viewer's window
src = pathlib.Path("page.html").resolve()
refs = pathlib.Path("<absolute path to skills/ui_design/references>")
call = lambda name, opts: f"({(refs / name).read_text()})({json.dumps(opts)})"
tmp = pathlib.Path(tempfile.mkdtemp()); shutil.copy(src, tmp / src.name)   # opened alone, as delivered
with sync_playwright() as pw:
    browser = pw.webkit.launch()                                            # or pw.chromium; say which
    page = browser.new_context(viewport={"width": W, "height": H},
                               device_scale_factor=2, color_scheme="light").new_page()
    failed = []; page.on("requestfailed", lambda r: failed.append(r.url))
    page.goto((tmp / src.name).as_uri(), wait_until="load"); page.wait_for_timeout(500)
    page.screenshot(path="first-screen.png")                                # look at this file yourself
    report = page.evaluate(call("measure.js", {"subject": ".card"}))
    page.locator(".card").screenshot(path="subject@2x.png")
    # To freeze a live page as a comparison frame instead, open its URL and run:
    #   frozen = page.evaluate(call("freeze.js", {}))
    #   pathlib.Path("frame.html").write_text(frozen["html"])
    print(json.dumps(report, ensure_ascii=False, indent=1), failed)
    browser.close()
shutil.rmtree(tmp, ignore_errors=True)
```

### Node

```js
const { webkit } = require("playwright");                                  // or chromium; say which
const fs = require("fs"), os = require("os"), path = require("path"), { pathToFileURL } = require("url");
const [W, H] = [1280, 800];                                                 // the viewer's window
const refs = "<absolute path to skills/ui_design/references>";
const call = (name, opts) => `(${fs.readFileSync(path.join(refs, name), "utf8")})(${JSON.stringify(opts)})`;
const src = path.resolve("page.html"), tmp = fs.mkdtempSync(path.join(os.tmpdir(), "look-"));
fs.copyFileSync(src, path.join(tmp, path.basename(src)));                   // opened alone, as delivered
(async () => {
  const browser = await webkit.launch();
  const context = await browser.newContext({ viewport: { width: W, height: H },
                                             deviceScaleFactor: 2, colorScheme: "light" });
  const page = await context.newPage();
  const failed = []; page.on("requestfailed", r => failed.push(r.url()));
  await page.goto(pathToFileURL(path.join(tmp, path.basename(src))).href, { waitUntil: "load" });
  await page.waitForTimeout(500);
  await page.screenshot({ path: "first-screen.png" });                       // look at this file yourself
  const report = await page.evaluate(call("measure.js", { subject: ".card" }));
  await page.locator(".card").screenshot({ path: "subject@2x.png" });
  console.log(JSON.stringify(report, null, 1), failed);
  await browser.close();
})();
```

## What to do with the result

Read the first screen as a stranger: what is the primary thing, what does each box look like it is, is anything clipped, covered or stuck. Then read the findings and keep the ones the viewer would notice. Fix, look again, and stop when the frame is right. Report the engine, the viewport, what you looked at and what you measured, in that order.
