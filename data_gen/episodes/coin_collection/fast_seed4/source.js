// coin_collection — top-down 2D world rendered with three.js (orthographic).
// This file IS the grounded rule condition AND the simulator. Fully deterministic:
// (seed, params, action sequence) uniquely determines every frame, state and event.
//
// World rules:
//   - Player moves on a square arena [-A, A]^2 at constant speed per step.
//   - Action sets a velocity direction; position integrates by player_speed * dt.
//   - Walls clamp the player (no wrap).
//   - Coins are scattered at reset. Collecting (dist < coin_radius+player_radius)
//     removes the coin, emits "coin_collected", adds coin_reward to score.
//   - Episode terminates when all coins collected (success) or time_limit reached.
import * as THREE from '/three.module.js';
import { makeRng } from '/engine/rng.js';

// Discrete action space defaults. action_id -> velocity direction (unit).
// 0 = noop, 1..8 = 8 compass directions. These now live in DEFAULT_PARAMS so a
// config can remap action semantics (e.g. swap left/right, reverse all, drop
// diagonals). action_names / key_binding are exposed via the schema so a
// code-conditioned model can read the (possibly remapped) action semantics.
const DEFAULT_DIRS = [
  [0, 0],            // 0 noop
  [0, 1],            // 1 up      (W / ArrowUp)
  [0, -1],           // 2 down    (S / ArrowDown)
  [-1, 0],           // 3 left    (A / ArrowLeft)
  [1, 0],            // 4 right   (D / ArrowRight)
  [-0.7071, 0.7071], // 5 up-left
  [0.7071, 0.7071],  // 6 up-right
  [-0.7071, -0.7071],// 7 down-left
  [0.7071, -0.7071], // 8 down-right
];

const DEFAULT_ACTION_NAMES = [
  'noop', 'up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right',
];

const DEFAULT_KEY_BINDING = {
  noop: [],
  up: ['KeyW', 'ArrowUp'],
  down: ['KeyS', 'ArrowDown'],
  left: ['KeyA', 'ArrowLeft'],
  right: ['KeyD', 'ArrowRight'],
};

export const DEFAULT_PARAMS = {
  arena: 8.0,
  player_speed: 0.18,     // units per step
  player_radius: 0.35,
  coin_count: 12,
  coin_radius: 0.30,
  coin_reward: 1.0,
  step_penalty: 0.0,      // optional per-step cost
  time_limit: 300,        // steps
  dt: 1.0,                // step is the time unit; speed already per-step
  // Action semantics — overridable via config (the configurable "rule prompt").
  dirs: DEFAULT_DIRS,                 // action_id -> [dx, dy] direction
  action_names: DEFAULT_ACTION_NAMES, // action_id -> human-readable name
  key_binding: DEFAULT_KEY_BINDING,   // semantic name -> physical keys (metadata)
};

export class Game {
  constructor(canvas, params = {}) {
    this.params = { ...DEFAULT_PARAMS, ...params };
    this.canvas = canvas;

    const A = this.params.arena;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setClearColor(0x10131a);
    this.scene = new THREE.Scene();
    // Orthographic top-down camera spanning a bit beyond the arena.
    const m = A + 1.0;
    this.camera = new THREE.OrthographicCamera(-m, m, m, -m, 0.1, 100);
    this.camera.position.set(0, 0, 10);
    this.camera.lookAt(0, 0, 0);

    // Arena floor + border.
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(2 * A, 2 * A),
      new THREE.MeshBasicMaterial({ color: 0x1b2230 })
    );
    this.scene.add(floor);
    const border = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.PlaneGeometry(2 * A, 2 * A)),
      new THREE.LineBasicMaterial({ color: 0x3a4a66 })
    );
    border.position.z = 0.01;
    this.scene.add(border);

    // Player.
    this.playerMesh = new THREE.Mesh(
      new THREE.CircleGeometry(this.params.player_radius, 24),
      new THREE.MeshBasicMaterial({ color: 0x46d3ff })
    );
    this.playerMesh.position.z = 0.2;
    this.scene.add(this.playerMesh);

    this.coinMeshes = [];
  }

  reset(seed = 0) {
    this.seed = seed >>> 0;
    const rng = makeRng(this.seed);
    const A = this.params.arena - 0.6;
    this.t = 0;
    this.score = 0;
    this.done = false;
    this.terminateReason = null;

    // Player starts at center.
    this.player = { x: 0, y: 0, vx: 0, vy: 0 };
    this.playerMesh.position.set(0, 0, 0.2);

    // Scatter coins.
    for (const m of this.coinMeshes) this.scene.remove(m);
    this.coinMeshes = [];
    this.coins = [];
    for (let i = 0; i < this.params.coin_count; i++) {
      const x = rng.range(-A, A);
      const y = rng.range(-A, A);
      this.coins.push({ x, y, alive: true });
      const mesh = new THREE.Mesh(
        new THREE.CircleGeometry(this.params.coin_radius, 16),
        new THREE.MeshBasicMaterial({ color: 0xffd23f })
      );
      mesh.position.set(x, y, 0.1);
      this.scene.add(mesh);
      this.coinMeshes.push(mesh);
    }
    this._pendingEvents = [];
    this.render();
    return this.getState();
  }

  // action: integer action_id in [0, 8]. Returns {reward, done, events}.
  step(action) {
    if (this.done) return { reward: 0, done: true, events: [] };
    const p = this.params;
    const a = (action | 0);
    const dirs = this.params.dirs;
    const dir = dirs[a] || dirs[0];
    const events = [];

    // Integrate position.
    this.player.vx = dir[0] * p.player_speed;
    this.player.vy = dir[1] * p.player_speed;
    let nx = this.player.x + this.player.vx * p.dt;
    let ny = this.player.y + this.player.vy * p.dt;

    // Wall collision: clamp to arena, emit wall_hit if clamped.
    const lim = p.arena - p.player_radius;
    let hit = false;
    if (nx < -lim) { nx = -lim; hit = true; }
    if (nx > lim) { nx = lim; hit = true; }
    if (ny < -lim) { ny = -lim; hit = true; }
    if (ny > lim) { ny = lim; hit = true; }
    if (hit) events.push({ t: this.t, type: 'wall_hit' });
    this.player.x = nx; this.player.y = ny;

    // Coin collection.
    let reward = -p.step_penalty;
    const cr = p.coin_radius + p.player_radius;
    const cr2 = cr * cr;
    let aliveCount = 0;
    for (let i = 0; i < this.coins.length; i++) {
      const c = this.coins[i];
      if (!c.alive) continue;
      const dx = c.x - nx, dy = c.y - ny;
      if (dx * dx + dy * dy < cr2) {
        c.alive = false;
        this.coinMeshes[i].visible = false;
        this.score += p.coin_reward;
        reward += p.coin_reward;
        events.push({ t: this.t, type: 'coin_collected', coin_id: i, x: c.x, y: c.y });
      } else {
        aliveCount++;
      }
    }

    this.t++;
    // Termination.
    if (aliveCount === 0) { this.done = true; this.terminateReason = 'all_coins'; }
    else if (this.t >= p.time_limit) { this.done = true; this.terminateReason = 'time_limit'; }
    if (this.done) events.push({ t: this.t, type: 'terminate', reason: this.terminateReason });

    // Sync visuals.
    this.playerMesh.position.set(this.player.x, this.player.y, 0.2);
    this._pendingEvents = events;
    return { reward, done: this.done, events };
  }

  render() { this.renderer.render(this.scene, this.camera); }

  getState() {
    return {
      t: this.t,
      player: { x: round(this.player.x), y: round(this.player.y), vx: round(this.player.vx), vy: round(this.player.vy) },
      coins: this.coins.map(c => ({ x: round(c.x), y: round(c.y), alive: c.alive })),
      coins_remaining: this.coins.filter(c => c.alive).length,
      score: this.score,
      done: this.done,
      terminate_reason: this.terminateReason,
    };
  }

  getActionSchema() {
    return {
      type: 'discrete',
      n: this.params.dirs.length,
      actions: this.params.action_names,
      key_binding: this.params.key_binding,
      semantics: 'action_id sets player velocity direction; magnitude = player_speed per step',
    };
  }
}

function round(v) { return Math.round(v * 1e4) / 1e4; }
