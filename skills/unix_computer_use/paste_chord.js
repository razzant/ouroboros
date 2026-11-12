/* The paste chord of type_text(via="paste") on a Mac, locally (osascript -l JavaScript <this file>) and over SSH
   (the same source through osascript -e on that Mac): Command plus the key that gives "v" UNDER COMMAND in the
   active keyboard layout, posted with exactly the Command flag. JavaScript for Automation is part of macOS and posts
   through CoreGraphics under the Accessibility grant cliclick needs (no Automation grant, nothing to install).
   Measured (#2589, macOS 26.4 guest, AppKit Edit menu): the menu matched the Command-plane characters of the event,
   so typing the "v" of the layout itself missed Paste on Russian - PC (Option added) and Dvorak - QWERTY Command ("." under
   Command), and a fixed US-V key missed it on Dvorak ("k"); this key pasted on ABC, the three Russian layouts and
   both Dvorak ones. key() sends a Command shortcut of one character the same way (8.11.2026, desktop Slack
   under a Cyrillic layout: cliclick, typing the layout's own "g" under Command, refused cmd+g with "Unable to
   get key code for g"): the character arrives as the program's own argument (osascript hands the words after
   the program to run), never written into this source; without one it is "v" and the words are the paste
   chord's. Its refusals use the words the receipt composer (place._finish) reads: no such key -> "Unable to get
   key code for <character>" (untypable_characters), no right to post -> "Accessibility privileges not enabled"
   (permission_denied); nothing is posted either way. A posted chord names its key on stderr; shlex quoting
   preserves this source as one shell word. */
ObjC.import("Foundation"); ObjC.import("Carbon"); ObjC.import("CoreServices"); ObjC.import("CoreGraphics");
function run(argv) {
  var named = Boolean(argv && argv.length), wanted = named ? String(argv[0]) : "v";
  var what = named ? "Command chord" : "paste chord";
  if (named && (argv.length !== 1 || !/^[^\s\p{C}]$/u.test(wanted))) { /* one code point, neither space nor control */
    throw new Error("the Command chord takes one printable character, not " + JSON.stringify(argv) + "; nothing was sent");
  }
  [["TISCopyCurrentKeyboardLayoutInputSource", ["void*", []]], ["TISGetInputSourceProperty", ["void*", ["void*", "id"]]],
   ["CFDataGetBytePtr", ["void*", ["void*"]]], ["CFRelease", ["void", ["void*"]]], ["LMGetKbdType", ["unsigned char", []]],
   ["UCKeyTranslate", ["int", ["void*", "unsigned short", "unsigned short", "unsigned int", "unsigned int", "unsigned int",
                               "void*", "unsigned long", "void*", "void*"]]],
   ["CGPreflightPostEventAccess", ["bool", []]], ["CGEventSourceCreate", ["void*", ["int"]]],
   ["CGEventCreateKeyboardEvent", ["void*", ["void*", "unsigned short", "bool"]]],
   ["CGEventSetFlags", ["void", ["void*", "unsigned long long"]]], ["CGEventPost", ["void", ["unsigned int", "void*"]]]
  ].forEach(function (f) { ObjC.bindFunction(f[0], f[1]); });
  var source = $.TISCopyCurrentKeyboardLayoutInputSource(), id = "", data = null, key = -1;
  if (source) {
    var name = $.TISGetInputSourceProperty(source, $("TISPropertyInputSourceID"));
    id = name ? ObjC.unwrap(ObjC.castRefToObject(name)) : "";
    data = $.TISGetInputSourceProperty(source, $("TISPropertyUnicodeKeyLayoutData"));
  }
  var layout = data ? $.CFDataGetBytePtr(data) : null;
  function typed(code, state) { /* the key's characters at a modifier state (kUCKeyActionDown, no dead keys), else null */
    var dead = $.NSMutableData.dataWithLength(4), count = $.NSMutableData.dataWithLength(8), text = $.NSMutableData.dataWithLength(8);
    return $.UCKeyTranslate(layout, code, 0, state, $.LMGetKbdType(), 1, dead.mutableBytes, 4, count.mutableBytes, text.mutableBytes) === 0 ?
      ObjC.unwrap($.NSString.alloc.initWithDataEncoding(text, $.NSUTF16LittleEndianStringEncoding)).replace(/\u0000+$/, "") : null;
  }
  function find(reads) { /* the first key that reads as the character, else -1 */
    for (var code = 0; layout && code < 128; code++) { if (reads(code) === wanted) { return code; } }
    return -1;
  }
  /* Modifier state cmdKey >> 8 = 1, Command alone. A named character no key gives so is looked for once more as an app
     reads Command+Shift (12.11.2026 guest matrix): its characters (state 3), or where the layout ignores Shift under
     Command (state 3 the same as 1: ABC, Dvorak) the Shift plane's (2), which AppKit's key equivalents matched then
     (cmd+} on ABC is Command, Shift and ]). Never the typing plane alone, so cmd+п is refused; paste is Command-V only. */
  key = find(function (code) { return typed(code, 1); });
  var shift = named && key < 0 && (key = find(function (code) {
    var both = typed(code, 3);
    return both !== null && both === typed(code, 1) ? typed(code, 2) : both;
  })) >= 0, held = shift ? "Command+Shift" : "Command";
  if (source) { $.CFRelease(source); }
  if (key < 0) {
    console.log("Unable to get key code for " + wanted + " (" + (layout ? "no key of " + id + " gives " + wanted + " under Command" +
                                                                    (named ? " or Command+Shift" : "") : "the active keyboard layout could not be read") +
                "); the " + what + " was not sent");
    return "";
  }
  if (!$.CGPreflightPostEventAccess()) {
    console.log("WARNING: Accessibility privileges not enabled (no right to post input events); the " + what + " was not sent");
    return "";
  }
  /* All events exist before the first is posted: one the system cannot make fails the chord with nothing sent.
     An error once a key went down still lets up what went down (the key, then Shift, then Command; each at most
     once, nothing goes down again) and ends the program with that same error: osascript exits 1, never an ok receipt. */
  var flags = shift ? 0x120000 : 0x100000; /* exactly Command (kCGEventFlagMaskCommand), and Shift (0x20000) when found so */
  var steps = [[55, true, 0x100000]].concat(shift ? [[56, true, flags]] : [], [[key, true, flags], [key, false, flags]],
                                            shift ? [[56, false, 0x100000]] : [], [[55, false, 0]]); /* 55 Command, 56 Shift */
  var events = $.CGEventSourceCreate(1), chord = [], tried = 0, posted = 0; /* kCGEventSourceStateHIDSystemState */
  try {
    steps.forEach(function (step) {
      var event = $.CGEventCreateKeyboardEvent(events, step[0], step[1]);
      if (!event) {
        throw new Error("CGEventCreateKeyboardEvent made no event for key code " + step[0] + "; the " + what + " was not sent");
      }
      chord.push(event);
      $.CGEventSetFlags(event, step[2]);
    });
    chord.forEach(function (event) { tried++; $.CGEventPost(1, event); posted++; delay(0.02); }); /* session tap */
  } catch (error) {
    for (var down = steps.length / 2 - 1; down >= 0; down--) { /* a down at i has its up at length - 1 - i */
      if (tried > down && posted <= steps.length - 1 - down) {
        try { $.CGEventPost(1, chord[steps.length - 1 - down]); } catch (ignored) { /* best effort: the error that counts is rethrown */ }
      }
    }
    throw error;
  } finally {
    chord.forEach(function (event) { $.CFRelease(event); });
    if (events) { $.CFRelease(events); }
  }
  console.log(what + ": " + held + " + key code " + key + " (" + wanted + " under " + held + " in " + id + ")");
  return "";
}
