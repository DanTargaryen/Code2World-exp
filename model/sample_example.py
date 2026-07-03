"""Render one dataset episode into a self-contained sample bundle under examples/.

The bundle shows the full code -> action -> frame loop for a single episode, so
one folder tells the whole story (independent of any model checkpoint):

  examples/<tag>/
    video.mp4        upscaled playback of the episode
    grid.png         every frame as a tile, labelled with the action that produced it
    config.yaml      declarative code condition (the game rules, if a matching cfg exists)
    source.cpp       source code condition (coinrun.cpp variant the data was collected with)
    actions.json     per-frame action ids + labels (aligned to frames[1:])
    meta.json        shapes / counts / provenance
    README.md        one-paragraph explanation of the bundle

    python -u sample_example.py --ep 0 --variant base
"""
import os, sys, json, argparse, subprocess, shutil
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))

# per-frame action id -> short label (compact CoinRun set; ids are raw Procgen)
ALABEL = {1: "left", 2: "left+jump", 4: "stay", 5: "jump", 7: "right", 8: "right+jump", 0: "noop"}
ASHORT = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U", 0: "-"}


def save_video(frames, path, fps, scale):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames[0].shape[:2]
    OH, OW = H * scale, W * scale
    p = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "16", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        img = cv2.resize(f, (OW, OH), interpolation=cv2.INTER_NEAREST)
        p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
    p.stdin.close(); p.wait()


def save_grid(frames, actions, path, per_row=9, s=3):
    W = 64 * s; tiles = []
    for i in range(len(frames)):
        cell = cv2.resize(frames[i], (W, W), interpolation=cv2.INTER_NEAREST)
        cell = cv2.copyMakeBorder(cell, 16, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        lbl = "init" if i == 0 else f"{i}:{ASHORT.get(int(actions[i-1]), str(int(actions[i-1])))}"
        cv2.putText(cell, lbl, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(cell)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + per_row], 1) for r in range(0, len(tiles), per_row)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


README = """# Sample bundle: {tag}

One CoinRun episode from `{root}` ({variant} / {split}, episode {ep}). Self-contained
demonstration of the **code -> action -> frame** loop the world model learns.

- **code condition** — what defines the game rules. Two equivalent forms:
  - `config.yaml`: declarative mechanics (the config-driven route){cfg_note}.
  - `source.cpp`: the exact coinrun.cpp variant this data was collected with.
- **action condition** — `actions.json`: {n_act} per-frame action ids ({ar} per latent
  step), aligned to `frames[1:]` (the action that produced each frame). Actions persist
  in segments and are NOT aligned to latent boundaries, so a latent's {ar} frames may
  span an action switch.
- **frames** — {n_frames} frames (64x64): `video.mp4` (playback) and `grid.png`
  (every frame tiled + labelled). Frame 0 is the init observation.

A latent = {ar} frames (Wan VAE temporal 4x). This episode = {K} latent steps.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_eval")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--out", default="../examples")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=6)
    args = ap.parse_args()

    tag = f"{args.variant}_{args.split}_ep{args.ep}"
    bundle = os.path.join(args.out, tag)
    os.makedirs(bundle, exist_ok=True)

    npz = np.load(os.path.join(args.root, args.variant, f"{args.split}.npz"))
    el = npz["episode_lengths"]
    ar = int(npz["action_repeat"])
    f0 = int(el[:args.ep].sum()) + args.ep          # frame start (each ep has K+1 frames)
    a0 = int(el[:args.ep].sum()) * ar               # action start (ar*K per ep)
    K = int(el[args.ep])
    n_frames = K * ar + 1                           # 4K+1 frames for K latent steps
    frames = npz["frames"][f0: f0 + n_frames]       # (n_frames, 64, 64, 3)
    actions = npz["actions"][a0: a0 + K * ar]       # (ar*K,) per-frame, aligned to frames[1:]
    n_act = len(actions)

    # 1) video + grid
    save_video(list(frames), os.path.join(bundle, "video.mp4"), args.fps, args.scale)
    save_grid(frames, actions, os.path.join(bundle, "grid.png"))

    # 2) per-frame actions (id + human label), aligned to frames[1:]
    with open(os.path.join(bundle, "actions.json"), "w") as f:
        json.dump({
            "aligned_to": "frames[1:] (action i produced frame i+1)",
            "action_repeat": ar,
            "n_actions": n_act,
            "actions": [int(a) for a in actions],
            "labels": [ALABEL.get(int(a), str(int(a))) for a in actions],
        }, f, indent=2)

    # 3) code condition: source.cpp (from dataset) + config.yaml (if a matching cfg exists)
    src = os.path.join(args.root, args.variant, "source.cpp")
    has_src = os.path.exists(src)
    if has_src:
        shutil.copy(src, os.path.join(bundle, "source.cpp"))
    cfg = os.path.join(HERE, "dataset", "configs", f"{args.variant}.yaml")
    has_cfg = os.path.exists(cfg)
    if has_cfg:
        shutil.copy(cfg, os.path.join(bundle, "config.yaml"))

    # 4) meta + README
    with open(os.path.join(bundle, "meta.json"), "w") as f:
        json.dump({
            "tag": tag, "root": args.root, "variant": args.variant,
            "split": args.split, "episode": args.ep,
            "latent_steps_K": K, "action_repeat": ar,
            "n_frames": n_frames, "frame_size": [64, 64, 3],
            "n_per_frame_actions": n_act,
            "fps": args.fps, "scale": args.scale,
            "has_source_cpp": has_src, "has_config_yaml": has_cfg,
        }, f, indent=2)
    cfg_note = "" if has_cfg else " (no matching config.yaml for this variant)"
    with open(os.path.join(bundle, "README.md"), "w") as f:
        f.write(README.format(tag=tag, root=args.root, variant=args.variant, split=args.split,
                              ep=args.ep, n_act=n_act, ar=ar, n_frames=n_frames, K=K,
                              cfg_note=cfg_note))

    files = sorted(os.listdir(bundle))
    print(f"saved bundle {bundle}/ ({n_frames} frames, {K} latent steps, {n_act} actions)", flush=True)
    print("  " + "  ".join(files), flush=True)


if __name__ == "__main__":
    main()
