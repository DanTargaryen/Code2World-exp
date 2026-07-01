// Random policy with action persistence (inertia) so trajectories look like
// purposeful wandering rather than per-step jitter. Deterministic given seed.
import { makeRng } from '../engine/rng.js';

export function makeRandomPolicy(seed, schema, opts = {}) {
  const hold = opts.hold ?? 0.85;      // prob of keeping previous action
  const rng = makeRng((seed ^ 0x9e3779b9) >>> 0);
  const n = schema.n;
  let last = 1 + Math.floor(rng.next() * (n - 1)); // start non-noop
  return function act(/* state */) {
    if (rng.next() < hold) return last;
    last = Math.floor(rng.next() * n); // can include noop
    return last;
  };
}
