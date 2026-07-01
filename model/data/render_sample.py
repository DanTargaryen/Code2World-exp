"""Render a longer CoinRun episode to mp4 (both 64x64 original and 32x32 training res)."""
import numpy as np
import cv2
import os
from procgen import ProcgenEnv

def collect_long_episode(seed=0, max_steps=400, num_envs=64):
    """Collect a long episode by picking the env that survives longest."""
    venv = ProcgenEnv(num_envs=num_envs, env_name='coinrun',
                      num_levels=0, start_level=seed, distribution_mode='easy')
    obs = venv.reset()
    rgb = obs['rgb']
    # Track frames per env until first done
    all_frames = [[] for _ in range(num_envs)]
    done_flag = [False]*num_envs
    for ei in range(num_envs):
        all_frames[ei].append(rgb[ei].copy())

    # Bias actions toward moving right + jumping (more purposeful than pure random)
    rng = np.random.RandomState(seed)
    for t in range(max_steps):
        # action 7 = right, 9 = jump, with some randomness
        acts = rng.choice([7, 7, 7, 9, 8, 1, 4], size=num_envs)
        obs, rew, done, info = venv.step(acts)
        rgb = obs['rgb']
        for ei in range(num_envs):
            if not done_flag[ei]:
                all_frames[ei].append(rgb[ei].copy())
                if done[ei]:
                    done_flag[ei] = True
    # Pick the longest episode
    lengths = [len(f) for f in all_frames]
    best = int(np.argmax(lengths))
    print(f"Longest episode: env {best}, {lengths[best]} frames")
    return all_frames[best]

def save_video(frames, path, fps=15, scale=1):
    import subprocess, imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    h, w = frames[0].shape[:2]
    H, W = h*scale, w*scale
    cmd = [ffmpeg, '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        img = f
        if scale != 1:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)
        proc.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    print(f"Saved {path} ({len(frames)} frames, {W}x{H})")

if __name__ == "__main__":
    out_dir = "sample_video"
    os.makedirs(out_dir, exist_ok=True)

    frames64 = collect_long_episode(seed=7, max_steps=400)

    # 64x64 original, upscaled 6x for viewing
    save_video(frames64, os.path.join(out_dir, "coinrun_64.mp4"), fps=15, scale=6)

    # 32x32 training res, upscaled 12x for viewing (same final size)
    frames32 = [cv2.resize(f, (32, 32), interpolation=cv2.INTER_AREA) for f in frames64]
    save_video(frames32, os.path.join(out_dir, "coinrun_32.mp4"), fps=15, scale=12)

    # Side-by-side: 64 (left) vs 32-upscaled (right), both at 384x384
    sbs_frames = []
    for f64, f32 in zip(frames64, frames32):
        left = cv2.resize(f64, (384, 384), interpolation=cv2.INTER_NEAREST)
        right = cv2.resize(f32, (384, 384), interpolation=cv2.INTER_NEAREST)
        sep = np.zeros((384, 4, 3), dtype=np.uint8); sep[:] = 255
        sbs_frames.append(np.concatenate([left, sep, right], axis=1))
    save_video(sbs_frames, os.path.join(out_dir, "coinrun_sbs.mp4"), fps=15, scale=1)
    print("done")
