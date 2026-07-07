"""Sample a preview video for EVERY procgen game (16 total) with random actions.

All procgen games share a Discrete(15) action space, so a random action stream is
enough to see each game's visuals/dynamics. For each game we run a fixed level with
a seeded random action stream (segments held a few frames for smoother motion),
letting the env auto-reset on done so the clip keeps showing content. Outputs:

  <game>.mp4            each game alone (upscaled)
  all_games_grid.mp4    4x4 grid of all 16 games
  README.txt            provenance

Ground-truth procgen frames (NO model). Run inside a pod that has procgen.
"""
import os, sys, argparse, subprocess
import numpy as np
import cv2
from procgen.env import ENV_NAMES
from procgen import ProcgenEnv

SEG_MIN, SEG_MAX = 3, 10   # hold a random action a few frames for smoother motion


def make_action_stream(seed, n, n_actions=15):
    rng = np.random.RandomState(seed)
    acts = []
    while len(acts) < n:
        a = int(rng.randint(0, n_actions))
        seg = int(rng.randint(SEG_MIN, SEG_MAX + 1))
        acts += [a] * seg
    return acts[:n]


def rollout(env_name, level, actions, seed):
    """Run a random action stream; allow auto-reset so the clip stays full length."""
    venv = ProcgenEnv(num_envs=1, env_name=env_name, num_levels=1,
                      start_level=level, distribution_mode="hard")
    frames = [venv.reset()["rgb"][0].copy()]
    for a in actions:
        obs, _, done, _ = venv.step(np.array([a], np.int32))
        frames.append(obs["rgb"][0].copy())
    del venv
    return frames


def upscale(f, scale, label=None):
    im = cv2.resize(f, (f.shape[1] * scale, f.shape[0] * scale),
                    interpolation=cv2.INTER_NEAREST)
    if label:
        cv2.putText(im, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2, cv2.LINE_AA)
    return im


def write_video(path, frame_iter, w, h, fps):
    ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()
    p = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for im in frame_iter:
        p.stdin.write(np.ascontiguousarray(im, np.uint8).tobytes())
    p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--grid-scale", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    actions = make_action_stream(args.seed, args.max_frames)
    seqs = {}
    for g in ENV_NAMES:
        fr = rollout(g, args.level, actions, args.seed)
        seqs[g] = fr
        print(f"  {g:12s}: {len(fr):3d} frames", flush=True)
        write_video(os.path.join(args.out, f"{g}.mp4"),
                    (upscale(f, args.scale, g) for f in fr),
                    64 * args.scale, 64 * args.scale, args.fps)

    # 4x4 grid of all 16 games
    games = list(ENV_NAMES)
    cols, rows = 4, 4
    s = args.grid_scale
    cell = 64 * s
    maxlen = max(len(fr) for fr in seqs.values())

    def grid_frame(i):
        canvas = np.zeros((rows * cell, cols * cell, 3), np.uint8)
        for idx, g in enumerate(games):
            fr = seqs[g]
            f = fr[i] if i < len(fr) else fr[-1]
            im = upscale(f, s, g)
            r, c = idx // cols, idx % cols
            canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = im
        return canvas

    write_video(os.path.join(args.out, "all_games_grid.mp4"),
                (grid_frame(i) for i in range(maxlen)), cols * cell, rows * cell, args.fps)

    with open(os.path.join(args.out, "README.txt"), "w") as f:
        f.write("procgen all-games preview (ground-truth frames, NO model)\n")
        f.write(f"games ({len(games)}): {', '.join(games)}\n")
        f.write(f"level={args.level} seed={args.seed} max_frames={args.max_frames} "
                f"fps={args.fps} scale={args.scale} grid_scale={args.grid_scale}\n")
        f.write("actions: random Discrete(15), held 3..10 frames per segment; auto-reset on done\n")
    print(f"saved {len(games)} solo + 1 grid ({maxlen} frames) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
