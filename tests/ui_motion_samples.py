"""Sample a real CSS transition's geometry without depending on the runner's frame rate."""

# Observe the production state change before the next paint, then seek the actual transitions. No duration,
# easing, keyframes or geometry is supplied by the test: removing the transition produces one jump, not samples.
# Then let the motion run normally, observing frames and DOM changes too: seeking must not bypass
# production timers or hide a room whose rows are disposed before its exit has finished.
SAMPLE_TRANSITION = """({root, target, ready, property, act, measure}) => new Promise((resolve, reject) => {
    const values = [measure()];
    const live = [];
    let frame = 0;
    let observing = null;
    const observer = new MutationObserver(sample);
    const cleanup = () => { observer.disconnect(); observing?.disconnect(); cancelAnimationFrame(frame); clearTimeout(timer); };
    const timer = setTimeout(() => { cleanup(); reject(new Error('Motion never reached its requested state or finished')); }, 5000);
    function sample() {
        if (!ready()) return;
        const node = target();
        const animations = node?.getAnimations() || [];
        const motion = animations.find(a => a.transitionProperty === property);
        observer.disconnect();
        const duration = motion?.effect.getComputedTiming().endTime || 0;
        if (duration > 0) {
            const times = animations.map(a => a.currentTime);
            animations.forEach(a => a.pause());
            for (let i = 0; i <= 20; i += 1) {
                animations.forEach(a => { a.currentTime = duration * i / 20; });
                values.push(measure());
            }
            animations.forEach((a, i) => { a.currentTime = times[i]; a.play(); });
            const observe = () => live.push(measure());
            observing = new MutationObserver(observe);
            observing.observe(root, {attributes: true, childList: true, subtree: true});
            const tick = () => { observe(); frame = requestAnimationFrame(tick); };
            tick();
            Promise.all(animations.map(a => a.finished.catch(() => {}))).then(() => requestAnimationFrame(() => {
                observe(); cleanup(); resolve({values, live, duration});
            }));
        } else {
            values.push(measure());
            cleanup(); resolve({values, live, duration});
        }
    }
    observer.observe(root, {attributes: true, childList: true, subtree: true});
    act();
    sample();
})"""
