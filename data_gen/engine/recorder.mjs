// Generic recorder: loads a game + policy into headless chromium, drives the
// deterministic step loop, captures one PNG per step, and dumps the full
// closed-loop dataset (frames, actions, states, events, rewards, source, meta).
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

function readVendor(p) { return fs.readFileSync(path.join(ROOT, p), 'utf8'); }

const PAGE_HTML = (w, h) => `<!doctype html><html><head><meta charset="utf-8"></head>
<body style="margin:0;background:#000">
<canvas id="c" width="${w}" height="${h}"></canvas>
<script type="module">
import { Game } from '/game.js';
const canvas = document.getElementById('c');
window.__init = (params) => { window.__game = new Game(canvas, params); window.__ready = true; };
window.__reset = (seed) => window.__game.reset(seed);
window.__step = (a) => window.__game.step(a);
window.__render = () => window.__game.render();
window.__state = () => window.__game.getState();
window.__schema = () => window.__game.getActionSchema();
window.__grab = () => canvas.toDataURL('image/png');
</script>
</body></html>`;

export async function recordEpisode({
  gameFile, params = {}, seed = 0, steps = 300,
  width = 256, height = 256, policyFactory, outDir,
}) {
  const threeSrc = readVendor('vendor/three.module.js');
  const rngSrc = readVendor('engine/rng.js');
  const gameSrc = fs.readFileSync(gameFile, 'utf8');

  const browser = await chromium.launch({
    executablePath: process.env.PW_CHROME || undefined,
    args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
           '--no-sandbox', '--disable-dev-shm-usage'],
  });
  const page = await browser.newPage({ viewport: { width, height } });
  page.on('pageerror', e => console.log('  [pageerror]', e.message));

  await page.route('**/three.module.js', r => r.fulfill({ contentType: 'application/javascript', body: threeSrc }));
  await page.route('**/engine/rng.js', r => r.fulfill({ contentType: 'application/javascript', body: rngSrc }));
  await page.route('**/game.js', r => r.fulfill({ contentType: 'application/javascript', body: gameSrc }));
  await page.route('**/index.html', r => r.fulfill({ contentType: 'text/html', body: PAGE_HTML(width, height) }));
  await page.goto('http://localhost/index.html', { waitUntil: 'load' });
  await page.evaluate((p) => window.__init(p), params);
  await page.waitForFunction('window.__ready === true', null, { timeout: 15000 });

  await page.evaluate((s) => window.__reset(s), seed);
  const schema = await page.evaluate('window.__schema()');
  const policy = policyFactory(seed, schema);

  // dirs
  const framesDir = path.join(outDir, 'frames');
  fs.mkdirSync(framesDir, { recursive: true });
  const actionsF = fs.createWriteStream(path.join(outDir, 'actions.jsonl'));
  const statesF = fs.createWriteStream(path.join(outDir, 'states.jsonl'));
  const eventsF = fs.createWriteStream(path.join(outDir, 'events.jsonl'));
  const rewardsF = fs.createWriteStream(path.join(outDir, 'rewards.jsonl'));

  const grabAndSave = async (idx) => {
    const dataUrl = await page.evaluate('window.__grab()');
    const buf = Buffer.from(dataUrl.split(',')[1], 'base64');
    fs.writeFileSync(path.join(framesDir, String(idx).padStart(6, '0') + '.png'), buf);
  };

  // Frame 0 = initial observation (state before any action).
  const s0 = await page.evaluate('window.__state()');
  statesF.write(JSON.stringify(s0) + '\n');
  await grabAndSave(0);

  let cumScore = 0, done = false, terminateReason = null, actualSteps = 0;
  for (let t = 0; t < steps; t++) {
    const state = await page.evaluate('window.__state()');
    const action = policy(state);
    const res = await page.evaluate((a) => { const r = window.__step(a); window.__render(); return r; }, action);

    actionsF.write(JSON.stringify({ t, action }) + '\n');
    cumScore += res.reward;
    rewardsF.write(JSON.stringify({ t, reward: round(res.reward), cum_score: round(cumScore), done: res.done }) + '\n');
    for (const ev of res.events) eventsF.write(JSON.stringify(ev) + '\n');

    const ns = await page.evaluate('window.__state()');
    statesF.write(JSON.stringify(ns) + '\n');
    await grabAndSave(t + 1);

    actualSteps = t + 1;
    if (res.done) { done = true; terminateReason = ns.terminate_reason; break; }
  }

  await new Promise(r => { let n = 4; const d = () => (--n === 0) && r(); for (const f of [actionsF, statesF, eventsF, rewardsF]) f.end(d); });

  // source + meta.
  fs.writeFileSync(path.join(outDir, 'source.js'), gameSrc);
  const effectiveParams = await page.evaluate('window.__game.params');
  const meta = {
    game: path.basename(gameFile, '.js'),
    seed, params, effective_params: effectiveParams, action_schema: schema,
    resolution: [width, height], requested_steps: steps,
    actual_steps: actualSteps, done, terminate_reason: terminateReason,
    final_score: round(cumScore), frame_count: actualSteps + 1,
  };
  fs.writeFileSync(path.join(outDir, 'meta.json'), JSON.stringify(meta, null, 2));

  await browser.close();
  return meta;
}

function round(v) { return Math.round(v * 1e4) / 1e4; }
