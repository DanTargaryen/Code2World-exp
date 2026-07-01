"""Driver: for each variant, write modified coinrun.cpp, run collect_one.py as a
fresh subprocess (so Procgen rebuilds), then restore the original source.

The original coinrun.cpp is always restored, even on crash.
"""
import os, sys, subprocess, shutil, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from variants import VARIANT_NAMES, make_variant_source

COINRUN = "/mnt/pfs/users/huangzehuan/projects/linming/workspace/procgen/procgen/src/games/coinrun.cpp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc")
    ap.add_argument("--unpaired", type=int, default=2200)
    ap.add_argument("--paired-count", type=int, default=300)
    ap.add_argument("--eval-count", type=int, default=100)
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--variants", nargs="*", default=VARIANT_NAMES)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    original = open(COINRUN).read()
    backup = COINRUN + ".orig_backup"
    if not os.path.exists(backup):
        shutil.copy(COINRUN, backup)

    try:
        for v in args.variants:
            print(f"\n========== variant: {v} ==========", flush=True)
            modified = make_variant_source(original, v)
            with open(COINRUN, "w") as f:
                f.write(modified)
            cmd = [sys.executable, "-u", os.path.join(HERE, "collect_one.py"),
                   "--variant", v, "--out", args.out, "--src", COINRUN,
                   "--unpaired", str(args.unpaired),
                   "--paired-count", str(args.paired_count),
                   "--eval-count", str(args.eval_count),
                   "--action-repeat", str(args.action_repeat)]
            r = subprocess.run(cmd)
            if r.returncode != 0:
                print(f"!! variant {v} failed (code {r.returncode})", flush=True)
                break
    finally:
        with open(COINRUN, "w") as f:
            f.write(original)
        print("\nrestored original coinrun.cpp", flush=True)

    # write variants manifest
    import json
    from variants import VARIANTS
    manifest = {}
    for v in args.variants:
        spec = VARIANTS[v]
        manifest[v] = {
            "change": None if spec is None else {"from": spec[0].strip(), "to": spec[1].strip()},
            "source": os.path.join(args.out, v, "source.cpp"),
        }
    with open(os.path.join(args.out, "variants.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.out}/variants.json", flush=True)


if __name__ == "__main__":
    main()
