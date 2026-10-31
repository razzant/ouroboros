// One layout per batch (docs/development/11-design-system.md "Chat viewport invariant"). A layout read right after a
// DOM write lays the whole document out again; code that reads and writes row by row pays that once per row (a page of
// history: hundreds of layouts, seconds each on a busy Mac). Inside a batch (`batch(fn)`: the reading owner's write is
// one, chat_reading_position.js `mutate`) `measure(read, apply)` defers its read: once `fn` has returned, every
// deferred read runs (one layout serves them all), then every `apply(value)` in the order asked. An `apply` that needs
// the layout it changed asks `measure` again (the next round). Outside a batch `apply(read())` runs at once. `read`
// only reads, `apply` only writes; whatever the reading owner restores after its write sees every round applied.
let queue = null;

export function measure(read, apply) {
    if (!queue) return void apply(read());
    queue.push([read, apply]);
}

// `settle()` runs before each round's reads: the reading owner wakes the rows its write changed (chat_row_sleep.js), so
// no read lays a sleeping row out on its own.
export function batch(fn, settle = () => {}) {
    const outer = !queue;
    if (outer) queue = [];
    try {
        const result = fn();
        while (queue.length) {
            settle();
            const round = queue.splice(0);
            const values = round.map(([read]) => read());
            round.forEach(([, apply], index) => apply(values[index]));
        }
        return result;
    } finally {
        if (outer) queue = null;
    }
}
