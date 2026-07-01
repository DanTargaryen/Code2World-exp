// CLI: generate a batch of episodes for one game.
//   node generate.mjs --game coin_collection --episodes 3 --steps 300 --seed-start 0 --out episodes
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import { recordEpisode } from './engine/recorder.mjs';
import { makeRandomPolicy } from './policies/random.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 ? process.argv[i + 1] : def;
}

const game = arg('game', 'coin_collection');
const episodes = parseInt(arg('episodes', '3'), 10);
const steps = parseInt(arg('steps', '300'), 10);
const seedStart = parseInt(arg('seed-start', '0'), 10);
const width = parseInt(arg('width', '256'), 10);
const height = parseInt(arg('height', '256'), 10);
const variant = arg('variant', 'base');
const outRoot = arg('out', path.join(__dirname, 'episodes'));
// Config file for complex overrides (dirs, key_binding, etc.)
const configFile = arg('config', null);
const fileConfig = configFile ? JSON.parse(fs.readFileSync(configFile, 'utf8')) : {};
// --params still works for quick single-value overrides; it takes precedence over --config.
const paramsOverride = { ...fileConfig, ...JSON.parse(arg('params', '{}')) };

const gameFile = path.join(__dirname, 'games', game + '.js');
if (!fs.existsSync(gameFile)) { console.error('no such game:', gameFile); process.exit(1); }

const policyFactory = (seed, schema) => makeRandomPolicy(seed, schema, { hold: 0.85 });

console.log(`generating ${episodes} episodes of ${game} (variant=${variant}, steps=${steps})`);
for (let i = 0; i < episodes; i++) {
  const seed = seedStart + i;
  const outDir = path.join(outRoot, game, `${variant}_seed${seed}`);
  fs.mkdirSync(outDir, { recursive: true });
  const t0 = process.hrtime.bigint();
  const meta = await recordEpisode({
    gameFile, params: paramsOverride, seed, steps, width, height, policyFactory, outDir,
  });
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log(`  [${i + 1}/${episodes}] seed=${seed} steps=${meta.actual_steps} score=${meta.final_score} reason=${meta.terminate_reason} (${ms.toFixed(0)}ms) -> ${outDir}`);
}
console.log('done.');
