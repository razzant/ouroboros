// Virtual time for the bounded /api/state reads (#2343). STATE_READ_LIFETIME_MS is an engineering tolerance,
// not a target: tests jump past it instead of waiting.

export const flush = async (turns = 8) => {
    for (let i = 0; i < turns; i += 1) await new Promise((resolve) => setImmediate(resolve));
};

/** Mock setTimeout, setInterval and Date; the returned `advance(ms)` ticks in `step`s, settling promises between. */
export function clock(t, step = 1000) {
    t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'], now: 1_700_000_000_000 });
    return async (ms, by = step) => {
        await flush();
        for (let done = 0; done < ms; done += by) {
            t.mock.timers.tick(by);
            await flush();
        }
    };
}

/** Whether `promise` has settled yet, and how. */
export function observe(promise) {
    const box = { settled: false, value: undefined, error: undefined };
    promise.then((value) => { box.settled = true; box.value = value; }, (error) => { box.settled = true; box.error = error; });
    return box;
}
