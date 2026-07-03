"""Self-check for the config-driven CoinRun options.

Runs three checks (mirrors the spirit of the old data_gen verify.mjs):
  1. NO-DRIFT: base config (no overrides) is frame-for-frame identical to a
     vanilla ProcgenEnv with NO coinrun_config -> options-ization changed nothing.
  2. DETERMINISM: same config + same seed + same actions -> identical frames.
  3. COUNTERFACTUAL: a changed mechanic (fast / lowgrav) produces DIFFERENT frames
     from base under the identical seed+action stream -> code sensitivity works.

Run inside the procgen pod after rebuilding procgen:
    python -u dataset/verify_config.py
"""
import os, sys, hashlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from game_config import load_config

from procgen import ProcgenEnv

ENV = "coinrun"
NUM_ENVS = 4
STEPS = 120
LEVEL = 0
SEED_ACTIONS = 12345
ACTION_SET = np.array([1, 2, 4, 5, 7, 8], dtype=np.int32)  # same restricted set as collect_one


def fixed_actions(n_steps, num_envs):
    """Deterministic action stream, independent of any env, so two runs are
    driven identically and only the rule differs."""
    rng = np.random.RandomState(SEED_ACTIONS)
    return [rng.choice(ACTION_SET, size=num_envs).astype(np.int32) for _ in range(n_steps)]


def rollout_md5(coinrun_config):
    """Run a fixed level+action rollout, return md5 over all rgb frames."""
    kwargs = dict(num_envs=NUM_ENVS, env_name=ENV, num_levels=1,
                  start_level=LEVEL, distribution_mode="hard")
    if coinrun_config is not None:
        kwargs["coinrun_config"] = coinrun_config
    venv = ProcgenEnv(**kwargs)
    obs = venv.reset()
    h = hashlib.md5()
    h.update(np.ascontiguousarray(obs["rgb"]).tobytes())
    for acts in fixed_actions(STEPS, NUM_ENVS):
        obs, _, _, _ = venv.step(acts)
        h.update(np.ascontiguousarray(obs["rgb"]).tobytes())
    venv.close()
    return h.hexdigest()


def cfg(name):
    return load_config(os.path.join(HERE, "configs", f"{name}.yaml"))["coinrun_config"]


def main():
    print("== 1. NO-DRIFT: base config vs vanilla (no coinrun_config) ==")
    vanilla = rollout_md5(None)
    base = rollout_md5(cfg("base"))
    print(f"  vanilla     : {vanilla}")
    print(f"  base config : {base}")
    print(f"  IDENTICAL   : {'PASS ✓' if vanilla == base else 'FAIL ✗'}")

    print("== 2. DETERMINISM: base twice ==")
    base2 = rollout_md5(cfg("base"))
    print(f"  MATCH       : {'PASS ✓' if base == base2 else 'FAIL ✗'}")

    print("== 3. COUNTERFACTUAL: base vs fast / lowgrav ==")
    fast = rollout_md5(cfg("fast"))
    lowgrav = rollout_md5(cfg("lowgrav"))
    print(f"  fast    differs from base: {'PASS ✓' if fast != base else 'FAIL ✗'}")
    print(f"  lowgrav differs from base: {'PASS ✓' if lowgrav != base else 'FAIL ✗'}")

    ok = (vanilla == base) and (base == base2) and (fast != base) and (lowgrav != base)
    print("\nRESULT:", "ALL PASS ✓" if ok else "SOME FAILED ✗")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
