# Code-Conditioned World Model — Dataset Generation (Stage 1)

Self-contained pipeline that generates the **code – action – state – video** closed-loop
data described in the blog (`plans/code_conditioned_world_model_blog.pdf`, Stage 1).

Games are small parametric three.js worlds. Each game file is BOTH the grounded rule
condition (`source.js` saved per episode) AND a fully deterministic simulator, so
`(seed, params, action_trace)` uniquely determines every frame / state / event.

## Layout
```
data_gen/
  vendor/three.module.js     three.js r160 (local, offline)
  engine/rng.js              deterministic seeded RNG (mulberry32)
  engine/recorder.mjs        playwright driver: step loop + frame capture + dump
  games/coin_collection.js   game = rule condition + simulator
  configs/                   rule-variant JSON configs (action remapping, etc.)
  policies/random.js         random policy with action persistence
  generate.mjs               batch CLI
  verify.mjs                 determinism + counterfactual self-check
  make_video.py              frames -> mp4 (opencv)
  run_env.sh                 sets PW_CHROME + LD_LIBRARY_PATH, runs a node script
  .condalibs/                libasound.so.2 (headless chromium dep, no root)
```

## Usage
```bash
# generate episodes
./run_env.sh generate.mjs --game coin_collection --episodes 20 --steps 300 --seed-start 0

# counterfactual variant (e.g. faster player) — same seeds, different rule
./run_env.sh generate.mjs --game coin_collection --episodes 20 --variant fast \
  --params '{"player_speed":0.40}'

# rule variant via config file (e.g. remap action semantics: A=right -> A=left)
./run_env.sh generate.mjs --game coin_collection --episodes 20 --variant swap_lr \
  --config configs/swap_left_right.json

# self-check (determinism + speed + action-mapping counterfactuals)
./run_env.sh verify.mjs

# render a video for human inspection
python3 make_video.py episodes/coin_collection/base_seed0 30
```

## Configurable rules (the "rule prompt")
Game rules are a structured JSON config. Everything in a game's `DEFAULT_PARAMS`
is overridable, including **action semantics**:
- `dirs` — `action_id -> [dx, dy]` velocity direction. Remap to swap/reverse
  directions or drop diagonals (fewer actions).
- `action_names` — `action_id -> name` (must match `dirs` length).
- `key_binding` — semantic name -> physical keys (metadata exposed in the schema).
- numeric params: `player_speed`, `player_radius`, `coin_count`, `coin_radius`,
  `coin_reward`, `step_penalty`, `time_limit`, `arena`, `dt`.

Two ways to override (they merge; `--params` wins over `--config`):
- `--config <file.json>` — full config with nested structures (dirs, key_binding).
- `--params '<json>'` — quick single-value override, e.g. `'{"player_speed":0.3}'`.

Prebuilt configs in `configs/`:
- `swap_left_right.json` — swaps the left/right action directions.
- `reverse_all.json` — reverses every direction (up<->down, left<->right).
- `wasd_only.json` — drops diagonals, leaving 5 actions (noop + 4 directions).

Because the random policy is **seed-only** (it ignores state), the same seed
yields the *same action_id sequence* under any config — so a config change is a
clean counterfactual: identical coin layout + identical actions, different rule.

## Per-episode output
```
episodes/<game>/<variant>_seed<N>/
  meta.json       game, seed, params (overrides), effective_params (merged full), action_schema, resolution, outcome
  source.js       full game source = grounded rule condition
  frames/NNNNNN.png   frame 0 = initial obs; frame t+1 = state after action t
  actions.jsonl   {t, action}
  states.jsonl    {t, player{x,y,vx,vy}, coins[{x,y,alive}], coins_remaining, score, done}
  events.jsonl    coin_collected / wall_hit / terminate
  rewards.jsonl   {t, reward, cum_score, done}
  video.mp4       optional, via make_video.py
```

## Verified properties
- **Determinism**: same seed twice → identical frame+state md5 (basis for counterfactual eval).
- **Counterfactual sensitivity (speed)**: changing `player_speed` changes trajectory/score/frames.
- **Counterfactual sensitivity (action mapping)**: swapping left/right `dirs` changes trajectory/frames.

## Environment notes (no root)
- headless chromium-1223 (playwright 1.52) renders WebGL via swiftshader.
- Only missing system lib was `libasound.so.2`, supplied from conda-forge `alsa-lib` in `.condalibs/`.
- The bundled playwright ffmpeg is webm-only and cannot read PNG; video uses system opencv instead.

## Next steps
- Add `navigation.js` (goal-reaching with walls).
- Add scripted/greedy policies for stronger action-following coverage.
- Scale batch (e.g. 20 ep × 300 steps per game) + systematic config variant sets
  (dirs / speed / collision_radius / coin_reward / key_binding) for Stage-3 ablation.
