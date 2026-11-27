// Subscription bookkeeping for DOM doubles that don't lay out or deliver resize events.
export class ResizeObserverStub {
    targets = new Set();
    observe(target) { this.targets.add(target); }
    unobserve(target) { this.targets.delete(target); }
    disconnect() { this.targets.clear(); }
}
