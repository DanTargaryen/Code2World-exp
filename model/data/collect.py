"""Collect CoinRun episodes from Procgen, resize to 32x32, save as npz."""
import numpy as np
import cv2
import os
import time
import argparse

def collect(env_name="coinrun", num_episodes=8000, max_steps=128,
            target_size=32, num_envs=128, out_dir="procgen_data"):
    from procgen import ProcgenEnv

    os.makedirs(out_dir, exist_ok=True)

    venv = ProcgenEnv(
        num_envs=num_envs,
        env_name=env_name,
        num_levels=0,
        start_level=0,
        distribution_mode="hard",
    )

    all_frames = []
    all_actions = []
    all_rewards = []
    all_dones = []

    episode_count = 0
    t0 = time.time()

    obs = venv.reset()
    if isinstance(obs, dict):
        obs = obs["rgb"]
    ep_frames = [[] for _ in range(num_envs)]
    ep_actions = [[] for _ in range(num_envs)]
    ep_rewards = [[] for _ in range(num_envs)]
    ep_dones = [[] for _ in range(num_envs)]

    for ei in range(num_envs):
        frame = obs[ei] if target_size == 64 else cv2.resize(obs[ei], (target_size, target_size), interpolation=cv2.INTER_AREA)
        ep_frames[ei].append(frame)

    step = 0
    while episode_count < num_episodes:
        actions = np.random.randint(0, 15, size=num_envs)
        obs, rewards, dones, infos = venv.step(actions)
        if isinstance(obs, dict):
            obs = obs["rgb"]

        for ei in range(num_envs):
            frame = obs[ei] if target_size == 64 else cv2.resize(obs[ei], (target_size, target_size), interpolation=cv2.INTER_AREA)
            ep_frames[ei].append(frame)
            ep_actions[ei].append(actions[ei])
            ep_rewards[ei].append(rewards[ei])
            ep_dones[ei].append(dones[ei])

            if dones[ei] or len(ep_actions[ei]) >= max_steps:
                if len(ep_actions[ei]) >= 4:
                    all_frames.append(np.array(ep_frames[ei], dtype=np.uint8))
                    all_actions.append(np.array(ep_actions[ei], dtype=np.int32))
                    all_rewards.append(np.array(ep_rewards[ei], dtype=np.float32))
                    all_dones.append(np.array(ep_dones[ei], dtype=bool))
                    episode_count += 1

                    if episode_count % 500 == 0:
                        elapsed = time.time() - t0
                        total_frames = sum(len(f) for f in all_frames)
                        print(f"  [{episode_count}/{num_episodes}] episodes, "
                              f"{total_frames} frames, {elapsed:.0f}s")

                    if episode_count >= num_episodes:
                        break

                ep_frames[ei] = [frame]
                ep_actions[ei] = []
                ep_rewards[ei] = []
                ep_dones[ei] = []

        step += 1

    total_frames = sum(len(f) for f in all_frames)
    print(f"Collected {episode_count} episodes, {total_frames} total frames")

    # Save episode lengths for reconstruction
    ep_lengths = np.array([len(a) for a in all_actions], dtype=np.int32)

    # Flatten for storage
    flat_frames = np.concatenate(all_frames, axis=0)
    flat_actions = np.concatenate(all_actions, axis=0)
    flat_rewards = np.concatenate(all_rewards, axis=0)
    flat_dones = np.concatenate(all_dones, axis=0)

    out_path = os.path.join(out_dir, f"{env_name}_{target_size}x{target_size}.npz")
    np.savez_compressed(out_path,
                        frames=flat_frames,
                        actions=flat_actions,
                        rewards=flat_rewards,
                        dones=flat_dones,
                        episode_lengths=ep_lengths)

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Saved to {out_path} ({size_mb:.1f} MB)")
    print(f"  frames: {flat_frames.shape}, actions: {flat_actions.shape}")
    print(f"  episodes: {len(ep_lengths)}, avg length: {ep_lengths.mean():.1f}")

    # Save a few sample frames for visual inspection
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(min(20, len(all_frames))):
        for t_idx in [0, len(all_frames[i])//2, -1]:
            cv2.imwrite(
                os.path.join(sample_dir, f"ep{i}_t{t_idx}.png"),
                cv2.cvtColor(all_frames[i][t_idx], cv2.COLOR_RGB2BGR))
    print(f"Saved sample frames to {sample_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="coinrun")
    parser.add_argument("--episodes", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--out", default="procgen_data")
    args = parser.parse_args()
    collect(args.env, args.episodes, args.max_steps, args.size, args.num_envs, args.out)
