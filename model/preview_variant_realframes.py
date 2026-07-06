"""Collect REAL procgen frames (NO model) for base + hazard-off variants on a fixed
level, to preview the visual signal each config change produces BEFORE retraining.

For each variant we run the SAME fixed per-frame action stream on the SAME level,
record every env frame until `done` (episode end) or max_frames, and write:
  <variant>.mp4                    each variant alone
  compare_<...>_3col.mp4           side-by-side [base | no_crate | no_monster],
                                   shorter runs hold their last frame + "DONE" tag
  README.txt                       full provenance

These are ground-truth procgen outputs (not model generation), so there is no
weight/ar/block here — the README says so explicitly.
"""
import os, sys, argparse, subprocess
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "dataset"))
from game_config import load_config
from procgen import ProcgenEnv

ACTION_SET = np.array([1, 2, 4, 5, 7, 8], dtype=np.int32)   # L L+U STAY UP R R+U
# rightward-biased so the player actually traverses the level (sees crates/monsters)
WEIGHTS = np.array([0.05, 0.08, 0.05, 0.12, 0.35, 0.35])


def make_action_stream(seed, n):
    rng = np.random.RandomState(seed)
    acts, i = [], 0
    while len(acts) < n:
        a = int(rng.choice(ACTION_SET, p=WEIGHTS))
        seg = int(rng.randint(2, 9))            # hold 2..8 frames (matches ActionStream)
        acts += [a] * seg
    return acts[:n]


def rollout(level, cfg, actions, env_opts=None):
    """Run the fixed action stream on `level`; stop at done (episode end). Returns
    list of (64,64,3) frames and whether it ended by death/goal."""
    venv = ProcgenEnv(num_envs=1, env_name="coinrun", num_levels=1,
                      start_level=level, distribution_mode="hard",
                      coinrun_config=cfg, **(env_opts or {}))
    frames = [venv.reset()["rgb"][0].copy()]
    ended = False
    for a in actions:
        obs, _, done, _ = venv.step(np.array([a], np.int32))
        if done[0]:                              # next frame would be a NEW level -> stop
            ended = True
            break
        frames.append(obs["rgb"][0].copy())
    del venv
    return frames, ended


def upscale(f, scale, label=None, tag=None):
    im = cv2.resize(f, (64 * scale, 64 * scale), interpolation=cv2.INTER_NEAREST)
    if label:
        cv2.putText(im, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    if tag:
        cv2.putText(im, tag, (4, im.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2, cv2.LINE_AA)
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
    ap.add_argument("--level", type=int, default=30)
    ap.add_argument("--variants", nargs="+", default=["base", "no_crate", "no_monster"])
    ap.add_argument("--astream-seed", type=int, default=7)
    ap.add_argument("--max-frames", type=int, default=240)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    actions = make_action_stream(args.astream_seed, args.max_frames)
    seqs = {}
    for v in args.variants:
        c = load_config(os.path.join(HERE, "dataset", "configs", f"{v}.yaml"))
        fr, ended = rollout(args.level, c["coinrun_config"] or None, actions,
                            env_opts=c["env_opts"] or None)
        seqs[v] = (fr, ended)
        print(f"  {v:12s}: {len(fr):3d} frames{' (ended: death/goal)' if ended else ' (survived max)'}", flush=True)
        write_video(os.path.join(args.out, f"{v}.mp4"),
                    (upscale(f, args.scale, v) for f in fr),
                    64 * args.scale, 64 * args.scale, args.fps)

    # side-by-side; pad shorter runs by holding last frame + DONE tag
    maxlen = max(len(fr) for fr, _ in seqs.values())
    H = 64 * args.scale
    def col_frame(i):
        cells = []
        for v in args.variants:
            fr, ended = seqs[v]
            if i < len(fr):
                cells.append(upscale(fr[i], args.scale, v))
            else:
                cells.append(upscale(fr[-1], args.scale, v, tag="DONE"))
        return np.concatenate(cells, axis=1)
    tag = "-".join(args.variants)
    write_video(os.path.join(args.out, f"compare_{tag}_{len(args.variants)}col.mp4"),
                (col_frame(i) for i in range(maxlen)), H * len(args.variants), H, args.fps)
    print(f"saved {len(args.variants)} solo + 1 compare video ({maxlen} frames, {maxlen/args.fps:.1f}s) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
