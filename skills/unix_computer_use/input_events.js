/* Mouse, wheel, keys and text on a Mac as CoreGraphics events, locally (osascript -l JavaScript <this file> '<spec>')
   and over SSH (the same source through osascript -e on that Mac, the spec as the word after it). JavaScript for
   Automation is part of macOS: nothing is installed, and posting needs only the grant to post input events
   (Accessibility; no Automation grant). The spec is JSON {"ops": [...], "budget_ms": n} built by mac_input.py:
     click {x, y, button: left|right|middle, count: 1..3}  move {x, y}  drag {from: [x, y], to: [x, y], button}
     button {down: bool, button, x?, y?}  scroll {dy, dx} (wheel lines; positive dy = up, positive dx = left)
     position {}  resolve {chars, command_chars}  type {text, interval_ms}
     keys {mods, code | char | system}  hold {mods, code? | char? | system?, ms}
   position and resolve post nothing and need no grant. Everything else first checks the right to post input events
   ("Accessibility privileges not enabled" and nothing posted when it is missing), then finds every character a chord
   names in the active layout, alone or with Shift, Option or both, dead keys excluded ("Unable to get key code for
   <character>" and nothing posted when none gives it): those are the words the receipt composer (place._finish)
   reads. resolve answers which characters no key gives (typing plane, and Command plane as paste_chord.js finds it), so
   a key sequence is refused before its first step. A system key (play, volume, brightness, keyboard light) is the
   NSSystemDefined event (subtype 8) a keyboard's media keys send. Every event carries the skill's mark in
   kCGEventSourceUserData (field 42); a key keeps the class bits a keyboard gives it (arrows Fn and NumericPad, F-keys
   and Home/End/Page/forward delete Fn, the keypad NumericPad), never a modifier the person happens to hold. All events
   of the call exist before the first is posted, each press beside its release, so one the system cannot make fails
   the call with nothing sent. An error once something went down, or the program's own deadline (budget_ms, below the
   caller's timeout), posts each press's own prepared release of every key and button still down (in its catch,
   last pressed first; counted in "posted"), prints "__POSTED__ <json>" (how many events went out and how many
   characters went down) and ends with that error: osascript
   exits 1, never an ok receipt. A finished mouse button "down" stays down: that is what it was asked for.
   Text goes one grapheme per key-down/up pair (a surrogate pair in one event; CRLF is one line break; a grapheme over
   20 UTF-16 units as its code points) with CGEventKeyboardSetUnicodeString, on the key that gives the character in
   the active layout when one does (an application that ignores the event's string still reads that key), else on a
   neutral key, counted as override_only. Every C function is bound here with its C signature; CGPoint (two doubles)
   is passed as two doubles, the same call on arm64 and x86_64. The answer on stdout is one JSON object. */
ObjC.import("Foundation"); ObjC.import("AppKit"); ObjC.import("Carbon"); ObjC.import("CoreServices"); ObjC.import("CoreGraphics");
var TAG = 0x4f55524f; /* "OURO": the skill's mark on every event (kCGEventSourceUserData) */
var NEUTRAL_KEY = 0x5a; /* kVK_F20: gives no character, so it carries a string no key of the layout gives */
var MODS = {cmd: [55, 0x100000], shift: [56, 0x20000], alt: [58, 0x80000], ctrl: [59, 0x40000], fn: [63, 0x800000]};
var BUTTONS = {left: [0, 1, 2, 6], right: [1, 3, 4, 7], middle: [2, 25, 26, 27]}; /* number, down, up, dragged */
var MOVED = 5, SESSION_TAP = 1, HID_TAP = 0, LINE = 1, SYSTEM_DEFINED = 14;
var CLASS_BITS = (function () { /* kCGEventFlagMaskNumericPad 0x200000, kCGEventFlagMaskSecondaryFn 0x800000 */
  var bits = {};
  [123, 124, 125, 126].forEach(function (c) { bits[c] = 0xa00000; });
  [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106, 115, 119, 116, 121, 117, 114, 71]
    .forEach(function (c) { bits[c] = 0x800000; }); /* 71 keypad clear: a keyboard gives it Fn, not NumericPad */
  [65, 67, 69, 75, 76, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92].forEach(function (c) { bits[c] = 0x200000; });
  return bits;
})();
function run(argv) {
  var spec = JSON.parse(argv && argv.length === 1 ? String(argv[0]) : "null");
  if (!spec || !Array.isArray(spec.ops) || !spec.ops.length) {
    throw new Error("the input program takes one JSON spec with a non-empty ops list; nothing was sent");
  }
  var started = Date.now(), budget = Math.max(500, Number(spec.budget_ms) || 8000);
  [["TISCopyCurrentKeyboardLayoutInputSource", ["void*", []]], ["TISGetInputSourceProperty", ["void*", ["void*", "id"]]],
   ["CFDataGetBytePtr", ["void*", ["void*"]]], ["CFRelease", ["void", ["void*"]]], ["CFRetain", ["void*", ["void*"]]],
   ["LMGetKbdType", ["unsigned char", []]],
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
  var layoutId = "", planes = {}, layoutRead = true;
  function typed(layout, code, state, options) { /* the key's characters at a modifier state, else "" */
    var dead = $.NSMutableData.dataWithLength(4), count = $.NSMutableData.dataWithLength(8), text = $.NSMutableData.dataWithLength(8);
    if ($.UCKeyTranslate(layout, code, 0, state, $.LMGetKbdType(), options, dead.mutableBytes, 4, count.mutableBytes, text.mutableBytes) !== 0) { return ""; }
    /* A dead-key state left nonzero (read as its four bytes) means the key arms a composition: not that character's key. */
    if (options === 0 && /[^\u0000]/.test(ObjC.unwrap($.NSString.alloc.initWithDataEncoding(dead, $.NSUTF16LittleEndianStringEncoding)))) { return ""; }
    return ObjC.unwrap($.NSString.alloc.initWithDataEncoding(text, $.NSUTF16LittleEndianStringEncoding)).replace(/\u0000+$/, "");
  }
  function plane(state, options) { /* {chars: {character: lowest code}, codes: [character by code]}, read once */
    var name = state + ":" + options;
    if (planes[name]) { return planes[name]; }
    var found = {chars: {}, codes: []}, source = $.TISCopyCurrentKeyboardLayoutInputSource(), data = null;
    try {
      if (source) {
        var id = $.TISGetInputSourceProperty(source, $("TISPropertyInputSourceID"));
        layoutId = id ? ObjC.unwrap(ObjC.castRefToObject(id)) : "";
        data = $.TISGetInputSourceProperty(source, $("TISPropertyUnicodeKeyLayoutData"));
      }
      var layout = data ? $.CFDataGetBytePtr(data) : null;
      if (!layout) { layoutRead = false; } /* no layout data: nothing is known, which is not "no key gives it" */
      for (var code = 0; layout && code < 128; code++) {
        var ch = typed(layout, code, state, options);
        found.codes[code] = ch;
        if (ch && !Object.prototype.hasOwnProperty.call(found.chars, ch)) { found.chars[ch] = code; }
      }
    } finally {
      if (source) { $.CFRelease(source); }
    }
    planes[name] = found;
    return found;
  }
  /* Typing planes: none, Shift, Option, Shift+Option (state = modifier >> 8). Options 0 keeps dead keys out: a dead
     key types nothing on its own, so an application that reads key codes would start a composition on it. */
  var STATES = [[0, []], [2, ["shift"]], [8, ["alt"]], [10, ["shift", "alt"]]];
  function keyFor(ch) { /* {code, mods} of the key giving ch, else null */
    for (var i = 0; i < STATES.length; i++) {
      var chars = plane(STATES[i][0], 0).chars;
      if (Object.prototype.hasOwnProperty.call(chars, ch)) { return {code: chars[ch], mods: STATES[i][1]}; }
    }
    return null;
  }
  function commandKey(ch) { /* as paste_chord.js finds a Command shortcut's key: Command, else Command+Shift */
    var cmd = plane(1, 1).codes, both = plane(3, 1).codes, shifted = plane(2, 1).codes;
    for (var code = 0; code < 128; code++) {
      if (cmd[code] === ch || (both[code] && both[code] === cmd[code] ? shifted[code] : both[code]) === ch) { return true; }
    }
    return false;
  }
  if (spec.ops.length === 1 && spec.ops[0].op === "position") {
    var now = where();
    return JSON.stringify({x: now[0], y: now[1]});
  }
  if (spec.ops.length === 1 && spec.ops[0].op === "resolve") { /* the grant first, then the layout (if it reads) */
    var r = spec.ops[0];
    if (!$.CGPreflightPostEventAccess()) { return JSON.stringify({post_events: false}); }
    plane(0, 0);
    if (!layoutRead) { return JSON.stringify({post_events: true, layout_unread: true}); }
    return JSON.stringify({layout: layoutId, post_events: true,
                           missing: (r.chars || []).filter(function (ch) { return !keyFor(ch); }),
                           missing_command: (r.command_chars || []).filter(function (ch) { return !commandKey(ch); })});
  }
  if (!$.CGPreflightPostEventAccess()) {
    console.log("WARNING: Accessibility privileges not enabled (no right to post input events); nothing was sent");
    return "";
  }
  var chordKeys = [];
  for (var i = 0; i < spec.ops.length; i++) { /* every key named by its character is found before anything is posted */
    var op = spec.ops[i], key = null;
    if (op.op === "keys" || op.op === "hold") {
      if (typeof op.code === "number") { key = {code: op.code, mods: []}; }
      if (typeof op.system === "number") { key = {system: op.system, mods: []}; }
      if (typeof op.char === "string" && !(key = keyFor(op.char)) && !layoutRead) {
        console.log("WARNING: the active keyboard layout could not be read (no layout data); nothing was sent");
        return "";
      }
      if (typeof op.char === "string" && !key) {
        console.log("Unable to get key code for " + op.char + " (no key of " + (layoutId || "the active keyboard layout") +
                    " gives " + op.char + " alone or with Shift or Option); nothing was sent");
        return "";
      }
    }
    chordKeys.push(key);
  }
  var events = $.CGEventSourceCreate(1), made = [], kept = [], plan = [], held = {}, open = {}, last = null;
  var answer = {posted: 0};
  function own(event, what) {
    if (!event) { throw new Error("the system made no " + what + " event; nothing was sent"); }
    made.push(event);
    $.CGEventSetIntegerValueField(event, 42, TAG);
    return event;
  }
  /* A step: the event, its tap, the pause after it, what it presses (down: [name, value]) or lets up (up: name). A
     release is registered as the prepared event that lets that key or button up. */
  function step(event, tap, wait, fields) {
    var s = Object.assign({event: event, tap: tap, wait: wait}, fields || {});
    if (s.down) { (open[s.down[0]] = open[s.down[0]] || []).push(s); }
    if (s.up && open[s.up] && open[s.up].length) { open[s.up].pop().release = s; } /* this press's own release */
    plan.push(s);
  }
  function mouse(button, kind, x, y, clicks) { /* kind: 1 down, 2 up, 3 dragged, 0 moved */
    var b = BUTTONS[button];
    if (!b) { throw new Error("unknown mouse button " + JSON.stringify(button) + "; nothing was sent"); }
    var event = own($.CGEventCreateMouseEvent(events, kind ? b[kind] : MOVED, x, y, b[0]), "mouse");
    if (clicks) { $.CGEventSetIntegerValueField(event, 1, clicks); } /* kCGMouseEventClickState */
    return event;
  }
  function keyEvent(code, down, flags, text) {
    var event = own($.CGEventCreateKeyboardEvent(events, code, down), "key");
    /* exactly these flags and the key's own class bits: what the keyboard holds never rides along */
    $.CGEventSetFlags(event, flags | (CLASS_BITS[code] || 0));
    if (text !== undefined) {
      var units = $.NSMutableData.dataWithData($.NSString.alloc.initWithString(text).dataUsingEncoding($.NSUTF16LittleEndianStringEncoding));
      $.CGEventKeyboardSetUnicodeString(event, text.length, units.mutableBytes);
    }
    return event;
  }
  function systemEvent(n, down) { /* NX key type n as a keyboard's media key sends it */
    var ns = $.NSEvent.otherEventWithTypeLocationModifierFlagsTimestampWindowNumberContextSubtypeData1Data2(
      SYSTEM_DEFINED, {x: 0, y: 0}, down ? 0xa00 : 0xb00, 0, 0, null, 8, (n << 16) | ((down ? 0xa : 0xb) << 8), -1);
    /* Measured on macOS 26.4 (21.11.2026, nothing posted): reading ns.CGEvent as a property gives a Ref that holds
       NULL (a void* call then acts on nothing); asking the object for it by its selector gives the object, which
       castObjectToRef turns into the real pointer. The wrapper is kept so the object lives as long as the program. */
    var obj = ns && !ns.isNil() ? ns.performSelector("CGEvent") : null;
    var ref = obj && !obj.isNil() ? ObjC.castObjectToRef(obj) : null;
    if (!ref) { throw new Error("the system made no system key event; nothing was sent"); }
    kept.push(obj);
    /* -CGEvent hands back a new autoreleased event the NSEvent does not keep: this program holds its own reference,
       released with every other event it made. */
    return own($.CFRetain(ref), "system key");
  }
  function chord(mods, key, holdMs) { /* modifiers down in order, the key down and up with all their flags, modifiers up */
    var flags = 0, names = mods.slice();
    (key && key.mods || []).forEach(function (m) { if (names.indexOf(m) < 0) { names.push(m); } });
    names.forEach(function (m) {
      if (!MODS[m]) { throw new Error("unknown modifier " + JSON.stringify(m) + "; nothing was sent"); }
      flags |= MODS[m][1];
      step(keyEvent(MODS[m][0], true, flags), SESSION_TAP, 0.01, {down: ["k" + MODS[m][0], MODS[m][0]]});
    });
    var wait = holdMs === undefined ? 0.02 : holdMs / 1000;
    if (key && key.system !== undefined) {
      step(systemEvent(key.system, true), HID_TAP, wait, {down: ["s" + key.system, key.system]});
      step(systemEvent(key.system, false), HID_TAP, 0.01, {up: "s" + key.system});
    } else if (key) {
      step(keyEvent(key.code, true, flags), SESSION_TAP, wait, {down: ["k" + key.code, key.code]});
      step(keyEvent(key.code, false, flags), SESSION_TAP, 0.01, {up: "k" + key.code});
    } else if (holdMs !== undefined && plan.length) { plan[plan.length - 1].wait = wait; }
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
        var parts = pieces(String(op.text || "")), wait = Math.max(0.004, (op.interval_ms || 0) / 1000); /* after each up only */
        answer = {posted: 0, method: "unicode_events", chars_total: parts.length, chars_sent: 0, override_only: 0};
        parts.forEach(function (piece) {
          var ch = piece === "\n" || piece === "\r\n" ? "\r" : piece;
          var key = ch === "\r" ? {code: 36, mods: []} : (ch === "\t" ? {code: 48, mods: []} : keyFor(ch));
          var code = key ? key.code : NEUTRAL_KEY, flags = 0;
          (key ? key.mods : []).forEach(function (m) { flags |= MODS[m][1]; });
          step(keyEvent(code, true, flags, ch), SESSION_TAP, 0, {down: ["k" + code, code], typed: true, override: !key});
          step(keyEvent(code, false, flags, ch), SESSION_TAP, wait, {up: "k" + code});
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
      s.posted = true;
      if (s.at) { last = s.at; }
      if (s.down) { held[s.down[0]] = s; }
      if (s.up) { delete held[s.up]; }
      if (s.typed) { answer.chars_sent++; } /* a character counts once its key went down */
      if (s.override) { answer.override_only++; }
      if (s.wait && Date.now() - started + s.wait * 1000 >= budget) { throw deadline(); } /* never a shortened hold */
      if (s.wait) { delay(s.wait); }
    });
  } catch (error) {
    Object.keys(held).reverse().forEach(function (name) { /* let up what went down, last first; the error is rethrown */
      try {
        var down = held[name], value = down.down[1];
        if (down.release && !down.release.posted) { /* the press's own prepared release */
          $.CGEventPost(down.release.tap, down.release.event);
        } else if (name[0] === "k") {
          $.CGEventPost(SESSION_TAP, keyEvent(value, false, 0));
        } else if (name[0] === "b") {
          $.CGEventPost(HID_TAP, own($.CGEventCreateMouseEvent(events, BUTTONS[value][2], last[0], last[1], BUTTONS[value][0]), "mouse"));
        } else {
          $.CGEventPost(HID_TAP, systemEvent(value, false));
        }
        answer.posted++;
      } catch (ignored) { /* nothing more can be done from here */ }
    });
    console.log("__POSTED__ " + JSON.stringify(answer));
    throw error;
  } finally {
    made.forEach(function (event) { $.CFRelease(event); });
    if (events) { $.CFRelease(events); }
  }
  return JSON.stringify(answer);
}
