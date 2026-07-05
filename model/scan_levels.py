"""Scan CoinRun levels to find object-rich scenes for strong-visual-signal variants.

For each level seed we roll out base + each "turn-off" config (no_crate/no_monster/
no_pit) with a fixed rightward action stream, and measure real-frame pixel diff vs
base. A large diff for an object means that object EXISTS on this level (so turning
it off is visible). Picks levels where the most object types are present.

    python scan_levels.py --n 40 --top 10
"""
import argparse, sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "dataset"))
from game_config import load_config
from procgen import ProcgenEnv

ACTS = [7, 7, 8, 7, 7, 5, 7, 7, 7, 8, 7, 7, 4, 7, 7, 8, 7, 7, 7, 5, 7, 7, 8, 7, 7, 5, 7, 7]


def rollout(level, cfg):
    v = ProcgenEnv(num_envs=1, env_name="coinrun", num_levels=1,
                   start_level=level, distribution_mode="hard", coinrun_config=cfg)
    fr = [v.reset()["rgb"][0].astype(np.float32)]
    for a in ACTS:
        o, _, _, _ = v.step(np.array([a], np.int32))
        fr.append(o["rgb"][0].astype(np.float32))
    del v
    return np.stack(fr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="scan levels 0..n-1")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--thresh", type=float, default=0.5, help="min Δ to count object as present")
    args = ap.parse_args()

    offs = {k: (load_config(os.path.join(HERE, "dataset", "configs", f"no_{k}.yaml"))["coinrun_config"] or None)
            for k in ["crate", "monster", "pit"]}

    rows = []
    for lv in range(args.n):
        base = rollout(lv, None)
        ds = {k: float(np.abs(rollout(lv, c) - base).mean()) for k, c in offs.items()}
        ntypes = sum(v > args.thresh for v in ds.values())
        rows.append((ntypes, sum(ds.values()), lv, ds))
    rows.sort(reverse=True)

    print(f"scanned {args.n} levels | crateΔ monsterΔ pitΔ (Δ>{args.thresh} => object present)")
    print("rank level | crate  monster  pit   | #types  totalΔ")
    for i, (nt, tot, lv, ds) in enumerate(rows[:args.top]):
        print(f"  {i+1:2d}  {lv:4d} | {ds['crate']:5.2f}  {ds['monster']:6.2f}  {ds['pit']:5.2f} |   {nt}    {tot:6.2f}")


if __name__ == "__main__":
    main()
