/* Mouse, wheel, keys and text on a Mac as CoreGraphics events, locally (osascript -l JavaScript <this file> '<spec>')
   and over SSH (the same source through osascript -e on that Mac, the spec as the word after it). JavaScript for
   Automation is part of macOS: nothing is installed, and posting needs only the grant to post input events
   (Accessibility; no Automation grant). The spec is JSON {"ops": [...], "budget_ms": n} built by mac_input.py:
     click {x, y, button: left|right|middle, count: 1..3}  move {x, y}  drag {from: [x, y], to: [x, y], button}
     button {down: bool, button, x?, y?}  scroll {dy, dx} (wheel lines; positive dy = up, positive dx = left)
     position {}  type {text, interval_ms}  keys {mods, code | char}  hold {mods, code? | char?, ms}
   Every event carries the skill's mark in kCGEventSourceUserData (field 42), so a receiver can tell them apart.
   Refusals use the words the receipt composer (place._finish) reads, and nothing is posted after either: a key
   character no key of the active layout gives (alone or with Shift) -> "Unable to get key code for <character>"; no
   right to post -> "Accessibility privileges not enabled". All events of the call exist before the first is posted, so
   one the system cannot make fails the call with nothing sent. An error once something went down, or the program's own
   deadline (budget_ms, below the caller's timeout), lets up every key and button it holds and ends with that error:
   osascript exits 1, never an ok receipt. A finished mouse button "down" stays down: that is what it was asked for.
   Text goes one grapheme per key-down/up pair (a surrogate pair in one event; a grapheme over 20 UTF-16 units as its
   code points) with CGEventKeyboardSetUnicodeString, on the key that gives the character in the active layout when
   one does (an application that ignores the event's string still reads that key), else on a neutral key, counted as
   override_only. Every function is bound here with its C signature; CGPoint (two doubles) is passed as two doubles,
   which is the same call on arm64 and x86_64. The answer on stdout is one JSON object. */
ObjC.import("Foundation"); ObjC.import("AppKit"); ObjC.import("Carbon"); ObjC.import("CoreServices"); ObjC.import("CoreGraphics");
var TAG = 0x4f55524f; /* "OURO": the skill's mark on every event (kCGEventSourceUserData) */
var NEUTRAL_KEY = 0x5a; /* kVK_F20: gives no character, so it carries a string no key of the layout gives */
var MODS = {cmd: [55, 0x100000], shift: [56, 0x20000], alt: [58, 0x80000], ctrl: [59, 0x40000], fn: [63, 0x800000]};
var BUTTONS = {left: [0, 1, 2, 6], right: [1, 3, 4, 7], middle: [2, 25, 26, 27]}; /* number, down, up, dragged */
var MOVED = 5, SESSION_TAP = 1, HID_TAP = 0, LINE = 1;
function run(argv) {
  var spec = JSON.parse(argv && argv.length === 1 ? String(argv[0]) : "null");
  if (!spec || !Array.isArray(spec.ops) || !spec.ops.length) {
    throw new Error("the input program takes one JSON spec with a non-empty ops list; nothing was sent");
  }
  var started = Date.now(), budget = Math.max(500, Number(spec.budget_ms) || 8000);
  [["TISCopyCurrentKeyboardLayoutInputSource", ["void*", []]], ["TISGetInputSourceProperty", ["void*", ["void*", "id"]]],
   ["CFDataGetBytePtr", ["void*", ["void*"]]], ["CFRelease", ["void", ["void*"]]], ["LMGetKbdType", ["unsigned char", []]],
   ["UCKeyTranslate", ["int", ["void*", "unsigned short", "unsigned short", "unsigned int", "unsigned int", "unsigned int",
                               "void*", "unsigned long", "void*", "void*"]]],
   ["CGPreflightPostEventAccess", ["bool", []]], ["CGEventSourceCreate", ["void*", ["int"]]],
   ["CGEventSourceButtonState", ["bool", ["int", "unsigned int"]]],
   ["CGEventCreateMouseEvent", ["void*", ["void*", "unsigned int", "double", "double", "unsigned int"]]],
   ["CGEventCreateKeyboardEvent", ["void*", ["void*", "unsigned short", "bool"]]],
   ["CGEventCreateScrollWheelEvent2", ["void*", ["void*", "unsigned int", "unsigned int", "int", "int", "int"]]],
   ["CGEventKeyboardSetUnicodeString", ["void", ["void*", "unsigned long", "void*"]]],
   ["CGEventSetIntegerValueField", ["void", ["void*", "unsigned int", "long long"]]],
   ["CGEventSetFlags", ["void", ["void*", "unsigned long long"]]], ["CGEventPost", ["void", ["unsigned int", "void*"]]]
  ].forEach(function (f) { ObjC.bindFunction(f[0], f[1]); });
  function where() { /* the pointer in global top-left points: AppKit counts up from the primary screen's bottom */
    var at = $.NSEvent.mouseLocation, primary = $.NSScreen.screens.objectAtIndex(0).frame;
    return [Math.round(at.x), Math.round(primary.size.height - at.y)];
  }
  if (spec.ops.length === 1 && spec.ops[0].op === "position") {
    var now = where();
    return JSON.stringify({x: now[0], y: now[1]});
  }
  var keys = null, layoutId = "";
  function keyFor(ch) { /* [code, needs Shift] of the key giving ch alone or with Shift alone, else null */
    if (keys === null) { keys = readLayout(); }
    return Object.prototype.hasOwnProperty.call(keys, ch) ? keys[ch] : null;
  }
  function readLayout() {
    var found = {}, source = $.TISCopyCurrentKeyboardLayoutInputSource(), data = null;
    if (source) {
      var name = $.TISGetInputSourceProperty(source, $("TISPropertyInputSourceID"));
      layoutId = name ? ObjC.unwrap(ObjC.castRefToObject(name)) : "";
      data = $.TISGetInputSourceProperty(source, $("TISPropertyUnicodeKeyLayoutData"));
    }
    var layout = data ? $.CFDataGetBytePtr(data) : null;
    for (var state = 0; layout && state <= 2; state += 2) { /* 0 no modifier, 2 shiftKey >> 8; the lowest code wins */
      for (var code = 0; code < 128; code++) {
        var dead = $.NSMutableData.dataWithLength(4), count = $.NSMutableData.dataWithLength(8), text = $.NSMutableData.dataWithLength(8);
        if ($.UCKeyTranslate(layout, code, 0, state, $.LMGetKbdType(), 1, dead.mutableBytes, 4, count.mutableBytes, text.mutableBytes) !== 0) { continue; }
        var ch = ObjC.unwrap($.NSString.alloc.initWithDataEncoding(text, $.NSUTF16LittleEndianStringEncoding)).replace(/\u0000+$/, "");
        if (ch && !Object.prototype.hasOwnProperty.call(found, ch)) { found[ch] = [code, state === 2]; }
      }
    }
    if (source) { $.CFRelease(source); }
    return found;
  }
  var chordKeys = [];
  for (var i = 0; i < spec.ops.length; i++) { /* every key named by its character is found before anything is posted */
    var op = spec.ops[i], key = null;
    if ((op.op === "keys" || op.op === "hold") && typeof op.code === "number") { key = [op.code, false]; }
    if ((op.op === "keys" || op.op === "hold") && typeof op.char === "string" && !(key = keyFor(op.char))) {
      console.log("Unable to get key code for " + op.char + " (no key of " + (layoutId || "the active keyboard layout") +
                  " gives " + op.char + " alone or with Shift); nothing was sent");
      return "";
    }
    chordKeys.push(key);
  }
  if (!$.CGPreflightPostEventAccess()) {
    console.log("WARNING: Accessibility privileges not enabled (no right to post input events); nothing was sent");
    return "";
  }
  var events = $.CGEventSourceCreate(1), made = [], plan = [], held = {}, last = null, answer = {posted: 0};
  function own(event, what) {
    if (!event) { throw new Error("the system made no " + what + " event; nothing was sent"); }
    made.push(event);
    $.CGEventSetIntegerValueField(event, 42, TAG);
    return event;
  }
  /* A step: the event, its tap, the pause after it, and what it presses (down: [name, value]) or lets up (up: name). */
  function step(event, tap, wait, fields) { plan.push(Object.assign({event: event, tap: tap, wait: wait}, fields || {})); }
  function mouse(button, kind, x, y, clicks) { /* kind: 1 down, 2 up, 3 dragged, 0 moved */
    var b = BUTTONS[button];
    if (!b) { throw new Error("unknown mouse button " + JSON.stringify(button) + "; nothing was sent"); }
    var event = own($.CGEventCreateMouseEvent(events, kind ? b[kind] : MOVED, x, y, b[0]), "mouse");
    if (clicks) { $.CGEventSetIntegerValueField(event, 1, clicks); } /* kCGMouseEventClickState */
    return event;
  }
  function keyEvent(code, down, flags, text) {
    var event = own($.CGEventCreateKeyboardEvent(events, code, down), "key");
    $.CGEventSetFlags(event, flags); /* exactly these flags: what the keyboard holds never rides along */
    if (text !== undefined) {
      var units = $.NSMutableData.dataWithData($.NSString.alloc.initWithString(text).dataUsingEncoding($.NSUTF16LittleEndianStringEncoding));
      $.CGEventKeyboardSetUnicodeString(event, text.length, units.mutableBytes);
    }
    return event;
  }
  function chord(mods, key, holdMs) { /* modifiers down in order, the key down and up with all their flags, modifiers up */
    var flags = 0, names = mods.slice();
    if (key && key[1] && names.indexOf("shift") < 0) { names.push("shift"); }
    names.forEach(function (m) {
      if (!MODS[m]) { throw new Error("unknown modifier " + JSON.stringify(m) + "; nothing was sent"); }
      flags |= MODS[m][1];
      step(keyEvent(MODS[m][0], true, flags), SESSION_TAP, 0.01, {down: ["k" + MODS[m][0], MODS[m][0]]});
    });
    if (key) {
      step(keyEvent(key[0], true, flags), SESSION_TAP, holdMs === undefined ? 0.02 : holdMs / 1000, {down: ["k" + key[0], key[0]]});
      step(keyEvent(key[0], false, flags), SESSION_TAP, 0.01, {up: "k" + key[0]});
    } else if (holdMs !== undefined && plan.length) { plan[plan.length - 1].wait = holdMs / 1000; }
    names.slice().reverse().forEach(function (m) {
      flags &= ~MODS[m][1];
      step(keyEvent(MODS[m][0], false, flags), SESSION_TAP, 0.01, {up: "k" + MODS[m][0]});
    });
  }
  function pieces(text) {
    var parts = (typeof Intl !== "undefined" && Intl.Segmenter) ?
      Array.from(new Intl.Segmenter(undefined, {granularity: "grapheme"}).segment(text), function (s) { return s.segment; }) : Array.from(text);
    return parts.reduce(function (out, g) { return out.concat(g.length > 20 ? Array.from(g) : [g]); }, []);
  }
  try {
    spec.ops.forEach(function (op, index) {
      var button = op.button || "left";
      if (op.op === "click") {
        step(mouse(button, 0, op.x, op.y), HID_TAP, 0.02, {at: [op.x, op.y]});
        for (var n = 1; n <= Math.max(1, Math.min(3, op.count || 1)); n++) {
          step(mouse(button, 1, op.x, op.y, n), HID_TAP, 0.02, {at: [op.x, op.y], down: ["b" + button, button]});
          step(mouse(button, 2, op.x, op.y, n), HID_TAP, 0.06, {at: [op.x, op.y], up: "b" + button});
        }
      } else if (op.op === "move") { /* a button held since an earlier call makes the move a drag */
        var pressed = ["left", "right", "middle"].filter(function (b) { return $.CGEventSourceButtonState(1, BUTTONS[b][0]); })[0];
        step(pressed ? mouse(pressed, 3, op.x, op.y) : mouse("left", 0, op.x, op.y), HID_TAP, 0.02, {at: [op.x, op.y]});
      } else if (op.op === "drag") {
        var a = op.from, b = op.to;
        step(mouse(button, 0, a[0], a[1]), HID_TAP, 0.02, {at: a});
        step(mouse(button, 1, a[0], a[1], 1), HID_TAP, 0.05, {at: a, down: ["b" + button, button]});
        for (var s = 1; s <= 12; s++) { /* the way in between, so the receiver sees the pointer travel */
          var p = [a[0] + (b[0] - a[0]) * s / 12, a[1] + (b[1] - a[1]) * s / 12];
          step(mouse(button, 3, p[0], p[1]), HID_TAP, 0.02, {at: p});
        }
        step(mouse(button, 2, b[0], b[1], 1), HID_TAP, 0, {at: b, up: "b" + button});
      } else if (op.op === "button") {
        var at = typeof op.x === "number" ? [op.x, op.y] : where();
        step(mouse(button, op.down ? 1 : 2, at[0], at[1], 1), HID_TAP, 0.02,
             op.down ? {at: at, down: ["b" + button, button]} : {at: at, up: "b" + button});
      } else if (op.op === "scroll") { /* one wheel line per detent, as a wheel sends them */
        var ny = Math.abs(op.dy || 0), nx = Math.abs(op.dx || 0);
        for (var l = 0; l < Math.max(ny, nx); l++) {
          step(own($.CGEventCreateScrollWheelEvent2(events, LINE, 2, l < ny ? Math.sign(op.dy) : 0, l < nx ? Math.sign(op.dx) : 0, 0),
                   "scroll"), HID_TAP, 0.02);
        }
      } else if (op.op === "type") {
        var parts = pieces(String(op.text || "")), wait = Math.max(0.004, (op.interval_ms || 0) / 1000);
        answer = {posted: 0, method: "unicode_events", chars_total: parts.length, chars_sent: 0, override_only: 0};
        parts.forEach(function (piece) {
          var ch = piece === "\n" ? "\r" : piece, key = ch === "\r" ? [36, false] : (ch === "\t" ? [48, false] : keyFor(ch));
          if (!key) { answer.override_only++; }
          var code = key ? key[0] : NEUTRAL_KEY, flags = key && key[1] ? MODS.shift[1] : 0;
          step(keyEvent(code, true, flags, ch), SESSION_TAP, 0.002, {down: ["k" + code, code]});
          step(keyEvent(code, false, flags, ch), SESSION_TAP, wait, {up: "k" + code, typed: true});
        });
      } else if (op.op === "keys" || op.op === "hold") {
        chord(op.mods || [], chordKeys[index], op.op === "hold" ? Math.max(0, Number(op.ms) || 0) : undefined);
      } else {
        throw new Error("unknown input op " + JSON.stringify(op.op) + "; nothing was sent");
      }
    });
    function deadline() { return new Error("the input program reached its own deadline (" + budget + " ms) after " + answer.posted + " event(s)"); }
    plan.forEach(function (s) {
      if (Date.now() - started >= budget) { throw deadline(); }
      $.CGEventPost(s.tap, s.event);
      answer.posted++;
      if (s.at) { last = s.at; }
      if (s.down) { held[s.down[0]] = s.down[1]; }
      if (s.up) { delete held[s.up]; }
      if (s.typed) { answer.chars_sent++; }
      if (s.wait && Date.now() - started + s.wait * 1000 >= budget) { throw deadline(); } /* never a shortened hold */
      if (s.wait) { delay(s.wait); }
    });
  } catch (error) {
    Object.keys(held).reverse().forEach(function (name) { /* let up what went down, last first; the error is rethrown */
      try {
        var up = name[0] === "k" ? keyEvent(held[name], false, 0) :
          own($.CGEventCreateMouseEvent(events, BUTTONS[held[name]][2], last[0], last[1], BUTTONS[held[name]][0]), "mouse");
        $.CGEventPost(name[0] === "k" ? SESSION_TAP : HID_TAP, up);
      } catch (ignored) { /* nothing more can be done from here */ }
    });
    if (answer.method) { console.log("typed " + answer.chars_sent + " of " + answer.chars_total + " character(s) before the error"); }
    throw error;
  } finally {
    made.forEach(function (event) { $.CFRelease(event); });
    if (events) { $.CFRelease(events); }
  }
  return JSON.stringify(answer);
}
