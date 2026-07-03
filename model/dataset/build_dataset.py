"""Driver: build a Code2World dataset via the config-driven route.

For each variant we run collect_one.py as a subprocess, passing its YAML config
(dataset/configs/<variant>.yaml). collect_one loads the config into procgen
options at collection time — NO coinrun.cpp edit, NO recompile (unlike the old
variants.py string-patch route). The config.yaml is also stored per variant as
the code condition (precompute --code-source yaml).

Prereq: procgen must be built once with the coinrun options plumbing
(feat/coinrun-config-options); `coinrun_gravity` etc. must be recognized.

Modes:
  default       unpaired + paired(seeds) + eval(seeds) per variant (random levels)
  --single-scene  overfit ONE fixed level; train/eval differ only by action stream
                  (matches the code2world_act6_pf overfit dataset)

Examples:
  # single-scene overfit on base (the current training setup)
  python -u build_dataset.py --single-scene --variants base \
    --unpaired 20000 --eval-unpaired 1000 --out <root>

  # multi-variant, random levels
  python -u build_dataset.py --variants base fast no_hazards --out <root>
"""
import os, sys, subprocess, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# the coinrun.cpp is stored per variant as source.cpp too (a second, raw form of the
# code condition). With the config-driven route it is NEVER modified — all variants
# share the same source; mechanics come from the config.
COINRUN = "/mnt/pfs/users/huangzehuan/projects/linming/workspace/procgen/procgen/src/games/coinrun.cpp"


def config_path(variant):
    return os.path.join(HERE, "configs", f"{variant}.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/pfs/data/huangzehuan/datasets/code2world_cfg")
    ap.add_argument("--variants", nargs="*", default=["base"])
    ap.add_argument("--action-repeat", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=20, help="max latent steps per episode")
    # random-level mode
    ap.add_argument("--unpaired", type=int, default=2000)
    ap.add_argument("--paired-count", type=int, default=300)
    ap.add_argument("--eval-count", type=int, default=100)
    # single-scene overfit mode
    ap.add_argument("--single-scene", action="store_true",
                    help="overfit one fixed level; train/eval differ only by action stream")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--eval-unpaired", type=int, default=1000,
                    help="--single-scene: #eval episodes (same scene, held-out action streams)")
    ap.add_argument("--num-envs", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # every requested variant must have a config (the config-driven route needs it)
    for v in args.variants:
        if not os.path.exists(config_path(v)):
            print(f"!! no config for variant {v!r}: {config_path(v)}", flush=True)
            sys.exit(1)

    for v in args.variants:
        print(f"\n========== variant: {v} ==========", flush=True)
        cmd = [sys.executable, "-u", os.path.join(HERE, "collect_one.py"),
               "--variant", v, "--out", args.out, "--src", COINRUN,
               "--config", config_path(v),
               "--action-repeat", str(args.action_repeat),
               "--max-steps", str(args.max_steps),
               "--num-envs", str(args.num_envs)]
        if args.single_scene:
            cmd += ["--single-scene", "--level", str(args.level),
                    "--unpaired", str(args.unpaired),
                    "--eval-unpaired", str(args.eval_unpaired)]
        else:
            cmd += ["--unpaired", str(args.unpaired),
                    "--paired-count", str(args.paired_count),
                    "--eval-count", str(args.eval_count)]
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"!! variant {v} failed (code {r.returncode})", flush=True)
            sys.exit(r.returncode)

    # manifest: precompute.py reads variants from here (keys) + records the code condition
    manifest = {v: {"config": f"configs/{v}.yaml",
                    "config_stored": os.path.join(args.out, v, "config.yaml"),
                    "source": os.path.join(args.out, v, "source.cpp")}
                for v in args.variants}
    with open(os.path.join(args.out, "variants.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {args.out}/variants.json ({len(args.variants)} variants)", flush=True)


if __name__ == "__main__":
    main()
