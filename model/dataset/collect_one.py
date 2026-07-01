"""Collect one CoinRun variant: unpaired (random levels) + paired (fixed seeds
replayed across all variants) + eval-paired (held-out seeds).

Run as a fresh process per variant so Procgen rebuilds with the current source.
Assumes the driver has already written the modified coinrun.cpp.

action_repeat: each sampled action is held for `action_repeat` env steps. We record
EVERY intermediate frame, so an episode of K actions has 4K+1 frames (for repeat=4),
which the Wan 2.1 causal VAE temporal-compresses to exactly K+1 latents
(1 action <-> 1 latent <-> 4 frames). reward is summed over the repeat window,
done is OR-ed. A chunk in which the env resets (done mid-window) is DROPPED so the
4K+1 invariant holds and no chunk straddles a reset.
"""
import os, sys, time, shutil, argparse
import numpy as np
import cv2

# Restricted action set: keep only horizontal moves + jump + noop, drop the three
# "downward" actions (0=down-left, 3=down, 6=down-right) which are meaningless in a
# platformer (gravity handles falling). IDs stay original (Procgen semantics fixed).
#   1=left  2=up-left  4=noop  5=up/jump  7=right  8=up-right
ACTION_SET = np.array([1, 2, 4, 5, 7, 8], dtype=np.int32)
MIN_ACTIONS = 4   # minimum actions (latent steps) for an episode to be kept


def sample_actions_for_seed(level_seed, max_len):
    """Deterministic action sequence for a level seed (reused across variants)."""
    rng = np.random.RandomState(level_seed)
    return rng.choice(ACTION_SET, size=max_len).astype(np.int32)


def to64(obs):
    return obs  # already 64x64 from Procgen


def collect_unpaired(env_name, n_episodes, max_actions, num_envs, action_repeat):
    from procgen import ProcgenEnv
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name,
                      num_levels=0, start_level=0, distribution_mode="hard")
    obs = venv.reset()["rgb"]
    # per-env buffers: ep_f holds 4K+1 frames, ep_a/r/d hold K per-action entries
    ep_f = [[obs[i].copy()] for i in range(num_envs)]
    ep_a = [[] for _ in range(num_envs)]
    ep_r = [[] for _ in range(num_envs)]
    ep_d = [[] for _ in range(num_envs)]
    out_f, out_a, out_r, out_d = [], [], [], []
    count = 0
    rng = np.random.RandomState(0)

    def flush(i):
        nonlocal count
        if len(ep_a[i]) >= MIN_ACTIONS:
            out_f.append(np.array(ep_f[i], np.uint8))
            out_a.append(np.array(ep_a[i], np.int32))
            out_r.append(np.array(ep_r[i], np.float32))
            out_d.append(np.array(ep_d[i], np.bool_))
            count += 1

    while count < n_episodes:
        acts = rng.choice(ACTION_SET, size=num_envs).astype(np.int32)
        chunk_f = [[] for _ in range(num_envs)]
        chunk_r = np.zeros(num_envs, np.float32)
        chunk_done = np.zeros(num_envs, bool)
        for _k in range(action_repeat):
            obs, rew, done, _ = venv.step(acts)
            obs = obs["rgb"]
            for i in range(num_envs):
                chunk_f[i].append(obs[i].copy())
                chunk_r[i] += rew[i]
                chunk_done[i] |= bool(done[i])
        for i in range(num_envs):
            if chunk_done[i]:
                # env auto-reset inside this window -> drop the straddling chunk,
                # close the episode, restart buffer from the current (post-reset) frame.
                flush(i)
                ep_f[i] = [obs[i].copy()]; ep_a[i] = []; ep_r[i] = []; ep_d[i] = []
                if count >= n_episodes:
                    break
            else:
                ep_f[i].extend(chunk_f[i])              # +action_repeat frames
                ep_a[i].append(acts[i])
                ep_r[i].append(chunk_r[i])
                ep_d[i].append(False)
                if len(ep_a[i]) >= max_actions:
                    flush(i)
                    ep_f[i] = [obs[i].copy()]; ep_a[i] = []; ep_r[i] = []; ep_d[i] = []
                    if count >= n_episodes:
                        break
    return out_f, out_a, out_r, out_d


def collect_paired(env_name, seeds, max_actions, action_repeat):
    """One episode per seed; replay the seed's deterministic action sequence.
    Each action held action_repeat steps; reward summed, done OR-ed over the window.
    A window containing a reset is dropped (episode ends at last complete action)."""
    from procgen import ProcgenEnv
    out_f, out_a, out_r, out_d, out_seed = [], [], [], [], []
    for s in seeds:
        venv = ProcgenEnv(num_envs=1, env_name=env_name,
                          num_levels=1, start_level=int(s), distribution_mode="hard")
        obs = venv.reset()["rgb"]
        actions = sample_actions_for_seed(s, max_actions)
        f = [obs[0].copy()]; a = []; r = []; d = []
        for t in range(max_actions):
            act = np.array([actions[t]], np.int32)
            chunk_f = []; chunk_r = 0.0; chunk_done = False
            for _k in range(action_repeat):
                obs, rew, done, _ = venv.step(act)
                obs = obs["rgb"]
                chunk_f.append(obs[0].copy()); chunk_r += float(rew[0]); chunk_done |= bool(done[0])
            if chunk_done:
                break                              # drop straddling window; episode ends here
            f.extend(chunk_f); a.append(actions[t]); r.append(chunk_r); d.append(False)
        if len(a) >= MIN_ACTIONS:
            out_f.append(np.array(f, np.uint8))
            out_a.append(np.array(a, np.int32))
            out_r.append(np.array(r, np.float32))
            out_d.append(np.array(d, np.bool_))
            out_seed.append(int(s))
        venv.close() if hasattr(venv, "close") else None
    return out_f, out_a, out_r, out_d, out_seed


def save_npz(path, fs, as_, rs, ds, seeds=None, action_repeat=1):
    lengths = np.array([len(a) for a in as_], np.int32)   # K (= latent steps) per episode
    flat_f = np.concatenate(fs, 0)
    flat_a = np.concatenate(as_, 0)
    flat_r = np.concatenate(rs, 0)
    flat_d = np.concatenate(ds, 0)
    kw = dict(frames=flat_f, actions=flat_a, rewards=flat_r, dones=flat_d,
              episode_lengths=lengths, action_repeat=np.int32(action_repeat))
    if seeds is not None:
        kw["seeds"] = np.array(seeds, np.int64)
    np.savez_compressed(path, **kw)
    mb = os.path.getsize(path) / 1e6
    print(f"  saved {os.path.basename(path)}: {len(lengths)} eps, "
          f"{len(flat_f)} frames, {mb:.0f}MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--env", default="coinrun")
    ap.add_argument("--out", required=True)              # dataset root
    ap.add_argument("--src", required=True)              # current coinrun.cpp (already modified)
    ap.add_argument("--unpaired", type=int, default=2200)
    ap.add_argument("--max-steps", type=int, default=60, help="max ACTIONS (latent steps) per episode")
    ap.add_argument("--action-repeat", type=int, default=4, help="env steps per action (frames/action)")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--paired-start", type=int, default=10000)
    ap.add_argument("--paired-count", type=int, default=300)
    ap.add_argument("--eval-start", type=int, default=20000)
    ap.add_argument("--eval-count", type=int, default=100)
    args = ap.parse_args()

    vdir = os.path.join(args.out, args.variant)
    os.makedirs(vdir, exist_ok=True)
    t0 = time.time()
    ar = args.action_repeat
    print(f"[{args.variant}] collecting... (action_repeat={ar})", flush=True)

    fs, as_, rs, ds = collect_unpaired(args.env, args.unpaired, args.max_steps, args.num_envs, ar)
    save_npz(os.path.join(vdir, "episodes_train.npz"), fs, as_, rs, ds, action_repeat=ar)

    pseeds = list(range(args.paired_start, args.paired_start + args.paired_count))
    fs, as_, rs, ds, sd = collect_paired(args.env, pseeds, args.max_steps, ar)
    save_npz(os.path.join(vdir, "episodes_paired.npz"), fs, as_, rs, ds, sd, action_repeat=ar)

    eseeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    fs, as_, rs, ds, sd = collect_paired(args.env, eseeds, args.max_steps, ar)
    save_npz(os.path.join(vdir, "episodes_eval.npz"), fs, as_, rs, ds, sd, action_repeat=ar)

    shutil.copy(args.src, os.path.join(vdir, "source.cpp"))
    print(f"[{args.variant}] done in {time.time()-t0:.0f}s -> {vdir}", flush=True)


if __name__ == "__main__":
    main()
