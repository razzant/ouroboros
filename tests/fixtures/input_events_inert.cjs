"use strict";
/* Runs skills/unix_computer_use/input_events.js byte for byte with inert stand-ins for the parts of the JavaScript for
   Automation bridge it uses. Node has no ObjC, AppKit, Carbon or CoreGraphics: nothing here can post input, move the
   pointer or read a real keyboard layout. A scenario says what the stand-ins answer: "spec" (the one word osascript
   hands run; "argv" overrides the whole list), "layout" ({id, planes: {state: {code: characters}}, dead: {state:
   [codes]}}: a dead key answers its spacing character only when asked without dead keys), "post": false
   (no right to post), "pointer" ([x, y] in AppKit's bottom-left points) with "screen_height", "held_buttons" (button
   numbers the HID state holds), "fail" ("event:2" = the third event the system is asked for is not made, "post:1" =
   the second CGEventPost throws; a list for several), "tick_ms" (how far the clock moves per bridge call; delays move it by their length).
   The trace says what the program did: every posted event with its fields, what is still held, what leaked.
   usage: node input_events_inert.cjs <program.js> '<JSON list of scenarios>'  ->  a JSON list of traces on stdout */
const fs = require("fs");
const vm = require("vm");

const UTF16LE = 0x94000100; // NSUTF16LittleEndianStringEncoding
const KEYBOARD_FLAGS = 0x80000; // a new key event carries what the keyboard holds; here Option is held
// ...and, as CoreGraphics does, the class bits of its key: arrows Fn and NumericPad, F-keys and the navigation block
// Fn, the keypad NumericPad (the program must keep these and drop the held Option).
const CLASS = (code) => ([123, 124, 125, 126].includes(code) ? 0xa00000
  : [122, 120, 99, 118, 96, 97, 98, 100, 101, 109, 103, 111, 105, 107, 113, 106, 115, 119, 116, 121, 117, 114, 71].includes(code) ? 0x800000
    : [65, 67, 69, 75, 76, 78, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91, 92].includes(code) ? 0x200000 : 0);
// An Objective-C property that returns a CoreFoundation object (NSEvent.CGEvent) comes back as an object wrapper,
// as JXA's bridge gives it; a C function bound with void* would receive NULL from one. Only ObjC.castObjectToRef
// turns it into the pointer: the stand-ins of void* parameters refuse a wrapper outright.
const wrapper = (object) => ({kind: "ObjectSpecifier", wrapped: object, isNil: () => object === null});
const nullRef = () => ({kind: "NullRef"});

function trace(program, scenario) {
  const t = {calls: [], log: [], posted: [], delays: [], error: null, returned: null};
  const owned = [], bridged = [], held = new Set(), bound = new Set(), count = {event: 0, post: 0};
  let clock = 0;
  const fault = (what) => [].concat(scenario.fail || []).includes(`${what}:${count[what]++}`); // one or a list
  const own = (kind, fields) => { const object = {kind, alive: true, fields: {}, ...fields}; owned.push(object); return object; };
  const live = (object, kind, use) => {
    if (!object || object.kind !== kind || !object.alive) throw new Error(`${use}: no live ${kind}`);
    return object;
  };
  const string = (value) => ({kind: "NSString", value, dataUsingEncoding: (encoding) => {
    if (encoding !== UTF16LE) throw new Error("dataUsingEncoding: not UTF-16LE");
    return {kind: "NSData", bytes: Buffer.from(value, "utf16le")};
  }});
  const event = (fields) => (fault("event") ? null : own("CGEvent", fields));
  const deliver = (tap, e) => {
    const f = e.fields, row = {tap, tag: f[42] === undefined ? null : f[42]};
    if (e.type === "system") {
      Object.assign(row, {system: e.n, down: e.down, flags: e.flags, subtype: e.subtype});
    } else if (e.type === "key") {
      Object.assign(row, {key: e.code, down: e.down, flags: e.flags});
      if (e.unicode !== undefined) row.unicode = e.unicode;
      if (e.down) held.add(`k${e.code}`); else held.delete(`k${e.code}`);
    } else if (e.type === "scroll") {
      Object.assign(row, {scroll: [e.dy, e.dx], units: e.units, wheels: e.wheels});
    } else {
      Object.assign(row, {mouse: e.mouseType, at: [e.x, e.y], button: e.button});
      if (f[1] !== undefined) row.clicks = f[1];
      if ([1, 3, 25].includes(e.mouseType)) held.add(`b${e.button}`);
      if ([2, 4, 26].includes(e.mouseType)) held.delete(`b${e.button}`);
    }
    t.posted.push(row);
  };
  const C = {
    TISCopyCurrentKeyboardLayoutInputSource: () => (scenario.layout ? own("TISInputSource", {layout: scenario.layout}) : null),
    TISGetInputSourceProperty: (source, key) => {
      const {layout} = live(source, "TISInputSource", "TISGetInputSourceProperty");
      if (key.value === "TISPropertyInputSourceID") return {kind: "CFString", owner: source, value: layout.id};
      return key.value === "TISPropertyUnicodeKeyLayoutData" && layout.planes ? {kind: "CFData", owner: source} : null;
    },
    CFDataGetBytePtr: (data) => ({kind: "uchr", owner: live(data.owner, "TISInputSource", "CFDataGetBytePtr")}),
    LMGetKbdType: () => 198,
    UCKeyTranslate: (layout, code, action, mods, kbdType, options, dead, max, length, text) => {
      const {planes, dead: deadKeys, arming} = live(layout.owner, "TISInputSource", "UCKeyTranslate on layout data").layout;
      const isDead = options === 0 && ((deadKeys || {})[mods] || []).includes(code);
      // "arming": a key that types its character and leaves a composition armed (a nonzero dead-key state).
      if (options === 0 && (isDead || ((arming || {})[mods] || []).includes(code))) dead.bytes.writeUInt32LE(1, 0);
      const out = isDead ? "" : ((planes[mods] || {})[code] || "").slice(0, max);
      Buffer.from(out, "utf16le").copy(text.bytes);
      length.bytes[0] = out.length;
      return 0;
    },
    CFRelease: (object) => {
      if (object && object.bridged) { // an event the bridge handed out: released only as often as it was retained
        if (object.retains <= 0) throw new Error("CFRelease of a system key event never retained (an over-release)");
        object.retains -= 1;
        return;
      }
      if (!owned.includes(object)) throw new Error(`CFRelease of ${object ? object.kind + ", not its own" : "NULL"}`);
      live(object, object.kind, "CFRelease").alive = false;
    },
    CFRetain: (object) => {
      if (!object || !object.bridged || !object.alive) throw new Error("CFRetain of something not a live bridged event");
      object.retains += 1;
      return object;
    },
    CGPreflightPostEventAccess: () => scenario.post !== false,
    CGEventSourceCreate: (state) => { t.source_state = state; return own("CGEventSource", {}); },
    CGEventSourceButtonState: (state, button) => (scenario.held_buttons || []).includes(button),
    CGEventCreateMouseEvent: (source, mouseType, x, y, button) => {
      live(source, "CGEventSource", "CGEventCreateMouseEvent");
      if (typeof x !== "number" || typeof y !== "number") throw new Error("CGEventCreateMouseEvent: the point is two doubles");
      return event({type: "mouse", mouseType, x, y, button});
    },
    CGEventCreateKeyboardEvent: (source, code, down) => {
      live(source, "CGEventSource", "CGEventCreateKeyboardEvent");
      return event({type: "key", code, down, flags: KEYBOARD_FLAGS | CLASS(code)});
    },
    CGEventCreateScrollWheelEvent2: (source, units, wheels, dy, dx, dz) => {
      live(source, "CGEventSource", "CGEventCreateScrollWheelEvent2");
      return event({type: "scroll", units, wheels, dy, dx, dz});
    },
    CGEventKeyboardSetUnicodeString: (e, length, units) => {
      live(e, "CGEvent", "CGEventKeyboardSetUnicodeString");
      e.unicode = units.bytes.subarray(0, length * 2).toString("utf16le");
    },
    CGEventSetIntegerValueField: (e, field, value) => { live(e, "CGEvent", "CGEventSetIntegerValueField").fields[field] = value; },
    CGEventSetFlags: (e, flags) => { live(e, "CGEvent", "CGEventSetFlags").flags = flags; },
    CGEventPost: (tap, e) => {
      live(e, "CGEvent", "CGEventPost");
      if (fault("post")) throw new Error("injected post failure");
      deliver(tap, e);
    },
  };
  const [px, py] = scenario.pointer || [0, 0];
  const Foundation = {
    NSMutableData: {
      dataWithLength: (n) => { const bytes = Buffer.alloc(n); return {bytes, mutableBytes: {bytes}}; },
      dataWithData: (data) => { const bytes = Buffer.from(data.bytes); return {bytes, mutableBytes: {bytes}}; },
    },
    NSString: {alloc: {
      initWithDataEncoding: (data, encoding) => {
        if (encoding !== UTF16LE) throw new Error("initWithDataEncoding: not UTF-16LE");
        return string(data.bytes.toString("utf16le"));
      },
      initWithString: (value) => string(String(value)),
    }},
    NSUTF16LittleEndianStringEncoding: UTF16LE,
    NSEvent: {
      get mouseLocation() { t.calls.push("mouseLocation"); return {x: px, y: py}; },
      // A system key: the NSEvent owns its CGEvent (not the program's to release; the stand-in refuses a CFRelease).
      otherEventWithTypeLocationModifierFlagsTimestampWindowNumberContextSubtypeData1Data2:
        (type, at, flags, time, window, context, subtype, data1, data2) => {
          if (type !== 14 || subtype !== 8 || data2 !== -1 || typeof at.x !== "number") throw new Error("not a system-defined key event");
          if (fault("event")) return null;
          const cg = {kind: "CGEvent", alive: true, fields: {}, type: "system", n: data1 >> 16, down: ((data1 >> 8) & 0xff) === 0xa,
                      flags, subtype, bridged: true, retains: 0};
          bridged.push(cg);
          // As measured on macOS 26.4: the property hands out a Ref holding NULL (no isNil; a void* call with it is
          // refused here), the selector hands out the object, and only castObjectToRef of the object is the pointer.
          return {isNil: () => false, get CGEvent() { return nullRef(); },
                  performSelector: (sel) => (sel === "CGEvent" ? wrapper(cg) : wrapper(null))};
        },
    },
    NSScreen: {screens: {objectAtIndex: (i) => ({frame: {origin: {x: 0, y: 0}, size: {width: 1728, height: scenario.screen_height || 1117}}})}},
  };
  const ObjC = {
    import: () => {},
    bindFunction: (name, signature) => {
      if (!Object.hasOwn(C, name)) throw new Error(`no inert stand-in for ${name}`);
      bound.add(name);
      if (name === "CGEventCreateMouseEvent") t.mouse_signature = signature;
    },
    castRefToObject: (ref) => { live(ref.owner, "TISInputSource", "castRefToObject"); return string(ref.value); },
    castObjectToRef: (object) => {
      if (object && object.kind === "NullRef") return object;  // a Ref already: the cast keeps its NULL
      if (!object || object.kind !== "ObjectSpecifier") throw new Error("castObjectToRef of something not an object wrapper");
      return object.wrapped;
    },
    unwrap: (object) => (object && object.kind === "NSString" ? object.value : object),
  };
  const $ = new Proxy((value) => string(String(value)), {get: (target, name) => {
    if (typeof name !== "string") return undefined;
    if (Object.hasOwn(Foundation, name)) return Foundation[name];
    if (!bound.has(name)) throw new Error(`$.${name} without ObjC.bindFunction`);
    return (...args) => {
      if (args.some((arg) => arg && (arg.kind === "ObjectSpecifier" || arg.kind === "NullRef"))) {
        throw new Error(`${name}: an object wrapper or a NULL Ref where a pointer goes (the call would act on nothing)`);
      }
      t.calls.push(name); clock += scenario.tick_ms || 0; return C[name](...args);
    };
  }});
  const context = vm.createContext({ObjC, $, Intl, Date: {now: () => clock},
    delay: (seconds) => { t.delays.push(seconds); clock += seconds * 1000; },
    console: {log: (...parts) => { t.log.push(parts.join(" ")); }}});
  try {
    vm.runInContext(program, context, {filename: "input_events.js", timeout: 5000});
    const argv = scenario.argv !== undefined ? scenario.argv : [JSON.stringify(scenario.spec)];
    t.returned = vm.runInContext(`run(${JSON.stringify(argv)})`, context, {timeout: 5000});
  } catch (error) {
    t.error = error && error.message !== undefined ? String(error.message) : String(error);
  }
  t.held = [...held];
  t.leaked = owned.filter((object) => object.alive).map((object) => object.kind)
    .concat(bridged.filter((object) => object.retains > 0).map(() => "system CGEvent (retained, never released)"));
  t.retained = bridged.map((object) => object.retains);
  return t;
}

const [path, scenarios] = process.argv.slice(2);
const program = fs.readFileSync(path, "utf8");
process.stdout.write(JSON.stringify(JSON.parse(scenarios).map((scenario) => trace(program, scenario))));
