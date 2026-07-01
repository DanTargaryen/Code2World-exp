// Self-check: (1) determinism — same seed twice -> identical frames+states.
//            (2) counterfactual — changing player_speed changes the trajectory.
import path from 'path';
import fs from 'fs';
import crypto from 'crypto';
import { fileURLToPath } from 'url';
import { recordEpisode } from './engine/recorder.mjs';
import { makeRandomPolicy } from './policies/random.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const gameFile = path.join(__dirname, 'games', 'coin_collection.js');
const tmp = path.join(__dirname, '.verify_tmp');
const policyFactory = (seed, schema) => makeRandomPolicy(seed, schema, { hold: 0.85 });

function md5dir(dir) {
  const files = fs.readdirSync(path.join(dir, 'frames')).sort();
  const h = crypto.createHash('md5');
  for (const f of files) h.update(fs.readFileSync(path.join(dir, 'frames', f)));
  h.update(fs.readFileSync(path.join(dir, 'states.jsonl')));
  return h.digest('hex');
}

async function run(name, params, seed = 0, steps = 120) {
  const outDir = path.join(tmp, name);
  fs.rmSync(outDir, { recursive: true, force: true });
  fs.mkdirSync(outDir, { recursive: true });
  await recordEpisode({ gameFile, params, seed, steps, width: 128, height: 128, policyFactory, outDir });
  return outDir;
}

console.log('== determinism: same seed twice ==');
const a = await run('det_a', {}, 7, 120);
const b = await run('det_b', {}, 7, 120);
const ha = md5dir(a), hb = md5dir(b);
console.log('  run A:', ha);
console.log('  run B:', hb);
console.log('  MATCH:', ha === hb ? 'PASS ✓' : 'FAIL ✗');

console.log('== counterfactual: change player_speed ==');
const base = await run('cf_base', { player_speed: 0.18 }, 7, 120);
const fast = await run('cf_fast', { player_speed: 0.40 }, 7, 120);
const hbase = md5dir(base), hfast = md5dir(fast);
console.log('  speed 0.18:', hbase);
console.log('  speed 0.40:', hfast);
console.log('  DIFFERS:', hbase !== hfast ? 'PASS ✓' : 'FAIL ✗');

// Show last-state divergence to prove trajectory changed.
const lastState = (dir) => JSON.parse(fs.readFileSync(path.join(dir, 'states.jsonl'), 'utf8').trim().split('\n').pop());
const sb = lastState(base), sf = lastState(fast);
console.log(`  final player pos  base=(${sb.player.x},${sb.player.y}) fast=(${sf.player.x},${sf.player.y})`);
console.log(`  final score       base=${sb.score} fast=${sf.score}`);

console.log('== counterfactual: swap left/right action mapping ==');
// Same seed -> identical coin layout AND identical action_id sequence (the random
// policy is seed-only). Swapping the dirs for left(3)/right(4) must change the
// trajectory, proving action semantics are a configurable rule dimension.
const SWAP_LR_DIRS = [
  [0, 0], [0, 1], [0, -1], [1, 0], [-1, 0],   // 3 and 4 swapped
  [-0.7071, 0.7071], [0.7071, 0.7071], [-0.7071, -0.7071], [0.7071, -0.7071],
];
const normal = await run('km_normal', {}, 7, 120);
const swapped = await run('km_swap', { dirs: SWAP_LR_DIRS }, 7, 120);
const hnorm = md5dir(normal), hswap = md5dir(swapped);
console.log('  normal dirs :', hnorm);
console.log('  swapped dirs:', hswap);
console.log('  DIFFERS:', hnorm !== hswap ? 'PASS ✓' : 'FAIL ✗');
const sn = lastState(normal), ss = lastState(swapped);
console.log(`  final player pos  normal=(${sn.player.x},${sn.player.y}) swapped=(${ss.player.x},${ss.player.y})`);

fs.rmSync(tmp, { recursive: true, force: true });
console.log('done.');
