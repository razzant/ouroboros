/* The paste chord of type_text(via="paste") on a Mac, locally (osascript -l JavaScript <this file>) and over SSH
   (the same source through osascript -e on that Mac): Command plus the key that gives "v" UNDER COMMAND in the
   active keyboard layout, posted with exactly the Command flag. JavaScript for Automation is part of macOS and posts
   through CoreGraphics under the Accessibility grant cliclick needs (no Automation grant, nothing to install).
   Measured (#2589, macOS 26.4 guest, AppKit Edit menu): the menu matched the Command-plane characters of the event,
   so typing the "v" of the layout itself missed Paste on Russian - PC (Option added) and Dvorak - QWERTY Command ("." under
   Command), and a fixed US-V key missed it on Dvorak ("k"); this key pasted on ABC, the three Russian layouts and
   both Dvorak ones. Its refusals use the words the receipt composer (place._finish) reads: no such key ->
   "Unable to get key code for v" (untypable_characters), no right to post -> "Accessibility privileges not enabled"
   (permission_denied); nothing is posted either way. A posted chord names its key on stderr;
   shlex quoting preserves this source as one shell word. */
ObjC.import("Foundation"); ObjC.import("Carbon"); ObjC.import("CoreServices"); ObjC.import("CoreGraphics");
function run() {
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
  for (var code = 0; layout && key < 0 && code < 128; code++) {
    var dead = $.NSMutableData.dataWithLength(4), count = $.NSMutableData.dataWithLength(8), text = $.NSMutableData.dataWithLength(8);
    /* kUCKeyActionDown 0, modifier state cmdKey >> 8 = 1 (Command alone), kUCKeyTranslateNoDeadKeysMask 1 */
    if ($.UCKeyTranslate(layout, code, 0, 1, $.LMGetKbdType(), 1, dead.mutableBytes, 4, count.mutableBytes, text.mutableBytes) === 0 &&
        ObjC.unwrap($.NSString.alloc.initWithDataEncoding(text, $.NSUTF16LittleEndianStringEncoding)).replace(/\u0000+$/, "") === "v") {
      key = code;
    }
  }
  if (source) { $.CFRelease(source); }
  if (key < 0) {
    console.log("Unable to get key code for v (" + (layout ? "no key of " + id + " gives v under Command"
                                                           : "the active keyboard layout could not be read") + "); the paste chord was not sent");
    return "";
  }
  if (!$.CGPreflightPostEventAccess()) {
    console.log("WARNING: Accessibility privileges not enabled (no right to post input events); the paste chord was not sent");
    return "";
  }
  /* All four events exist before the first is posted: one the system cannot make fails the chord with nothing sent.
     An error once Command or the key went down still lets up what went down (the key, then Command; each at most
     once, nothing goes down again) and ends the program with that same error: osascript exits 1, never an ok receipt. */
  var events = $.CGEventSourceCreate(1), chord = [], tried = 0, posted = 0; /* kCGEventSourceStateHIDSystemState */
  try {
    [[55, true, 0x100000], [key, true, 0x100000], [key, false, 0x100000], [55, false, 0]].forEach(function (step) {
      var event = $.CGEventCreateKeyboardEvent(events, step[0], step[1]); /* 55: the Command key */
      if (!event) {
        throw new Error("CGEventCreateKeyboardEvent made no event for key code " + step[0] + "; the paste chord was not sent");
      }
      chord.push(event);
      $.CGEventSetFlags(event, step[2]); /* exactly Command */
    });
    chord.forEach(function (event) { tried++; $.CGEventPost(1, event); posted++; delay(0.02); }); /* session tap */
  } catch (error) {
    [[1, 2], [0, 3]].forEach(function (pair) { /* [a down, its up] */
      if (tried > pair[0] && posted <= pair[1]) {
        try { $.CGEventPost(1, chord[pair[1]]); } catch (ignored) { /* best effort: the error that counts is rethrown */ }
      }
    });
    throw error;
  } finally {
    chord.forEach(function (event) { $.CFRelease(event); });
    if (events) { $.CFRelease(events); }
  }
  console.log("paste chord: Command + key code " + key + " (v under Command in " + id + ")");
  return "";
}
