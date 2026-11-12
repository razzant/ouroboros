"use strict";
/* Runs skills/unix_computer_use/paste_chord.js byte for byte with inert stand-ins for the parts of the JavaScript for
   Automation bridge it uses. Node has no ObjC, Carbon or CoreGraphics: nothing here can post input or read a real
   keyboard layout or clipboard. A scenario says what the stand-ins answer (the layout's characters by UCKeyTranslate
   modifier state, the right to post, an event the system cannot make, a call that throws: fail "post:1" = the
   second CGEventPost throws; "landed" = after the event reached the system; "argv" = the words osascript hands run,
   absent for none); the trace says what the program did.
   usage: node paste_chord_inert.cjs <program.js> '<JSON list of scenarios>'  ->  a JSON list of traces on stdout */
const fs = require("fs");
const vm = require("vm");

const UTF16LE = 0x94000100; // NSUTF16LittleEndianStringEncoding
const KEYBOARD_FLAGS = 0x80000; // a new event carries what the keyboard holds; here Option is held

function trace(program, scenario) {
  const t = {calls: [], log: [], translated: [], posted: [], released: [], error: null, returned: null};
  const owned = [], held = new Set(), bound = new Set(), count = {event: 0, flags: 0, post: 0, delay: 0};
  const fault = (what) => scenario.fail === `${what}:${count[what]++}`;
  const own = (kind, fields) => { const object = {kind, alive: true, ...fields}; owned.push(object); return object; };
  const live = (object, kind, use) => {
    if (!object || object.kind !== kind || !object.alive) throw new Error(`${use}: no live ${kind}`);
    return object;
  };
  const string = (value) => ({kind: "NSString", value});
  const deliver = (tap, event) => {
    t.posted.push([tap, event.code, event.down, event.flags]);
    if (event.down) held.add(event.code); else held.delete(event.code);
  };
  const C = {
    TISCopyCurrentKeyboardLayoutInputSource: () =>
      (scenario.layout ? own("TISInputSource", {layout: scenario.layout}) : null),
    TISGetInputSourceProperty: (source, key) => { // the Get rule: what it returns belongs to the source
      const {layout} = live(source, "TISInputSource", "TISGetInputSourceProperty");
      if (key.value === "TISPropertyInputSourceID") return {kind: "CFString", owner: source, value: layout.id};
      return key.value === "TISPropertyUnicodeKeyLayoutData" && layout.planes ? {kind: "CFData", owner: source} : null;
    },
    CFDataGetBytePtr: (data) => ({kind: "uchr", owner: live(data.owner, "TISInputSource", "CFDataGetBytePtr")}),
    LMGetKbdType: () => 198,
    UCKeyTranslate: (layout, code, action, mods, kbdType, options, dead, max, length, text) => {
      const {planes} = live(layout.owner, "TISInputSource", "UCKeyTranslate on layout data").layout;
      t.translated.push([code, action, mods, kbdType, options, max]);
      const out = ((planes[mods] || {})[code] || "").slice(0, max);
      Buffer.from(out, "utf16le").copy(text.bytes);
      length.bytes[0] = out.length;
      return 0;
    },
    CFRelease: (object) => {
      if (!owned.includes(object)) throw new Error(`CFRelease of ${object ? object.kind + ", not its own" : "NULL"}`);
      live(object, object.kind, "CFRelease").alive = false;
      t.released.push(object.kind);
    },
    CGPreflightPostEventAccess: () => scenario.post !== false,
    CGEventSourceCreate: (state) => {
      t.source_state = state;
      return scenario.no_event_source ? null : own("CGEventSource", {});
    },
    CGEventCreateKeyboardEvent: (source, code, down) => {
      if (source !== null) live(source, "CGEventSource", "CGEventCreateKeyboardEvent");
      return fault("event") ? null : own("CGEvent", {code, down, flags: KEYBOARD_FLAGS});
    },
    CGEventSetFlags: (event, flags) => {
      live(event, "CGEvent", "CGEventSetFlags");
      if (fault("flags")) throw new Error("injected flags failure");
      event.flags = flags;
    },
    CGEventPost: (tap, event) => {
      live(event, "CGEvent", "CGEventPost");
      if (!fault("post")) return deliver(tap, event);
      if (scenario.landed) deliver(tap, event);
      throw new Error("injected post failure");
    },
  };
  const Foundation = {
    NSMutableData: {dataWithLength: (n) => { const bytes = Buffer.alloc(n); return {bytes, mutableBytes: {bytes}}; }},
    NSString: {alloc: {initWithDataEncoding: (data, encoding) => {
      if (encoding !== UTF16LE) throw new Error("initWithDataEncoding: not UTF-16LE");
      return string(data.bytes.toString("utf16le"));
    }}},
    NSUTF16LittleEndianStringEncoding: UTF16LE,
  };
  const ObjC = {
    import: () => {},
    bindFunction: (name) => {
      if (!Object.hasOwn(C, name)) throw new Error(`no inert stand-in for ${name}`);
      bound.add(name);
    },
    castRefToObject: (ref) => { live(ref.owner, "TISInputSource", "castRefToObject"); return string(ref.value); },
    unwrap: (object) => (object && object.kind === "NSString" ? object.value : object),
  };
  const $ = new Proxy((value) => string(String(value)), {get: (target, name) => {
    if (typeof name !== "string") return undefined;
    if (Object.hasOwn(Foundation, name)) return Foundation[name];
    if (!bound.has(name)) throw new Error(`$.${name} without ObjC.bindFunction`);
    return (...args) => { t.calls.push(name); return C[name](...args); };
  }});
  const context = vm.createContext({ObjC, $,
    delay: () => { t.calls.push("delay"); if (fault("delay")) throw new Error("injected delay failure"); },
    console: {log: (...parts) => { t.calls.push("console.log"); t.log.push(parts.join(" ")); }}});
  try {
    vm.runInContext(program, context, {filename: "paste_chord.js", timeout: 5000});
    const argv = scenario.argv === undefined ? "" : JSON.stringify(scenario.argv); // a JSON list is a JavaScript literal
    t.returned = vm.runInContext(`run(${argv})`, context, {timeout: 5000});
  } catch (error) {
    t.error = error && error.message !== undefined ? String(error.message) : String(error);
  }
  t.held = [...held];
  t.leaked = owned.filter((object) => object.alive).map((object) => object.kind);
  return t;
}

const [path, scenarios] = process.argv.slice(2);
const program = fs.readFileSync(path, "utf8");
process.stdout.write(JSON.stringify(JSON.parse(scenarios).map((scenario) => trace(program, scenario))));
