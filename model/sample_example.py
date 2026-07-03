"""Render one dataset episode to examples/: mp4 video + frame grid PNG.

A quick way to eyeball what the collected CoinRun data looks like (real frames +
per-frame actions), independent of any model checkpoint.

    python -u sample_example.py --ep 0            # writes ../examples/
"""
import os, sys, argparse, subprocess
import numpy as np
import cv2

# per-frame action id -> short label (compact CoinRun set; ids are raw Procgen)
ALABEL = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U", 0: "-"}


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
    os.makedirs(args.out, exist_ok=True)

    npz = np.load(os.path.join(args.root, args.variant, f"{args.split}.npz"))
    el = npz["episode_lengths"]
    ar = int(npz["action_repeat"])
    # locate this episode: frames stored as sum(K+1) per episode; actions as ar*K
    f0 = int(el[:args.ep].sum()) + args.ep          # frame start (each ep has K+1 frames)
    a0 = int(el[:args.ep].sum()) * ar               # action start (ar*K per ep)
    K = int(el[args.ep])
    n_frames = K * ar + 1                           # 4K+1 frames for K latent steps
    frames = npz["frames"][f0: f0 + n_frames]       # (n_frames, 64, 64, 3)
    actions = npz["actions"][a0: a0 + K * ar]       # (ar*K,) per-frame, aligned to frames[1:]

    tag = f"{args.variant}_{args.split}_ep{args.ep}"
    # 1) mp4
    mp4 = os.path.join(args.out, f"{tag}.mp4")
    save_video(list(frames), mp4, args.fps, args.scale)
    print(f"saved {mp4} ({n_frames} frames, {n_frames/args.fps:.1f}s @ {args.fps}fps)", flush=True)

    # 2) frame grid PNG: one tile per frame, labelled with the action that produced it
    s = 3; W = 64 * s; per_row = 9; tiles = []
    for i in range(n_frames):
        cell = cv2.resize(frames[i], (W, W), interpolation=cv2.INTER_NEAREST)
        cell = cv2.copyMakeBorder(cell, 16, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        # frame 0 = init (no action); frame i>=1 produced by actions[i-1]
        lbl = "init" if i == 0 else f"{i}:{ALABEL.get(int(actions[i-1]), str(int(actions[i-1])))}"
        cv2.putText(cell, lbl, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(cell)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + per_row], 1) for r in range(0, len(tiles), per_row)]
    grid = np.concatenate(rows, 0)
    png = os.path.join(args.out, f"{tag}_grid.png")
    cv2.imwrite(png, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"saved {png} ({n_frames} frames, {per_row}/row)", flush=True)


if __name__ == "__main__":
    main()
