"""Collect one CoinRun variant: unpaired (random levels) + paired (fixed seeds
replayed across all variants) + eval-paired (held-out seeds).

Run as a fresh process per variant so Procgen rebuilds with the current source.
Assumes the driver has already written the modified coinrun.cpp.

PER-FRAME ACTION mode (this branch): each ENV STEP (frame) gets its own action, but
actions are sampled in SEGMENTS that persist a random number of frames (SEG_MIN..
SEG_MAX) and DO NOT align to latent (4-frame) boundaries. So a single latent's 4
frames may span an action switch. We still record EVERY frame: an episode of K
latents has 4K+1 frames (Wan VAE temporal-compresses to K+1 latents), and now
stores 4K PER-FRAME actions (aligned to frames[1:], i.e. the action that PRODUCED
each frame). FLOW-ONLY: we store ONLY frames + per-frame actions (no reward/done
data fields). A latent window straddling a reset is still dropped so the 4K+1
invariant holds — the env `done` signal drives that control flow but is not saved.
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
SEG_MIN, SEG_MAX = 2, 8   # per-frame action segment length range (frames), latent-unaligned


class ActionStream:
    """Per-frame action generator: holds one action for a random run of SEG_MIN..SEG_MAX
    frames, then resamples. Segment boundaries are independent of the 4-frame latent
    grid, so latents routinely contain an intra-frame action switch."""
    def __init__(self, rng):
        self.rng = rng
        self._cur = int(rng.choice(ACTION_SET))
        self._left = int(rng.randint(SEG_MIN, SEG_MAX + 1))

    def next(self):
        if self._left <= 0:
            self._cur = int(self.rng.choice(ACTION_SET))
            self._left = int(self.rng.randint(SEG_MIN, SEG_MAX + 1))
        self._left -= 1
        return self._cur


def sample_actions_for_seed(level_seed, n_frames):
    """Deterministic PER-FRAME action sequence for a level seed (reused across variants).
    Returns n_frames actions produced by a seeded ActionStream."""
    stream = ActionStream(np.random.RandomState(level_seed))
    return np.array([stream.next() for _ in range(n_frames)], dtype=np.int32)


def to64(obs):
    return obs  # already 64x64 from Procgen



def collect_unpaired(env_name, n_episodes, max_actions, num_envs, action_repeat):
    """Per-frame actions from a per-env ActionStream. Buffers hold, per latent step,
    `action_repeat` frames and `action_repeat` per-frame actions. ep_a therefore ends
    with 4K per-frame actions. FLOW-ONLY: only frames + per-frame actions are stored;
    the env `done` signal is used solely to drop reset-straddling windows / close
    episodes, not saved. Random levels (num_levels=0). For single fixed scene use
    collect_single_scene."""
    from procgen import ProcgenEnv
    venv = ProcgenEnv(num_envs=num_envs, env_name=env_name,
                      num_levels=0, start_level=0, distribution_mode="hard")
    obs = venv.reset()["rgb"]
    rng = np.random.RandomState(0)
    streams = [ActionStream(np.random.RandomState(1000 + i)) for i in range(num_envs)]
    # per-env buffers: ep_f holds 4K+1 frames, ep_a holds 4K per-frame actions
    ep_f = [[obs[i].copy()] for i in range(num_envs)]
    ep_a = [[] for _ in range(num_envs)]
    ep_k = [0 for _ in range(num_envs)]              # completed latent steps per env
    out_f, out_a = [], []
    count = 0

    def flush(i):
        nonlocal count
        if ep_k[i] >= MIN_ACTIONS:
            out_f.append(np.array(ep_f[i], np.uint8))
            out_a.append(np.array(ep_a[i], np.int32))
            count += 1

    def reset_buf(i):
        ep_f[i] = [obs[i].copy()]; ep_a[i] = []; ep_k[i] = 0

    while count < n_episodes:
        # one latent window = action_repeat frames, each frame its own per-frame action
        chunk_f = [[] for _ in range(num_envs)]
        chunk_a = [[] for _ in range(num_envs)]
        chunk_done = np.zeros(num_envs, bool)
        for _k in range(action_repeat):
            acts = np.array([streams[i].next() for i in range(num_envs)], np.int32)
            obs, rew, done, _ = venv.step(acts)
            obs = obs["rgb"]
            for i in range(num_envs):
                chunk_f[i].append(obs[i].copy())
                chunk_a[i].append(int(acts[i]))
                chunk_done[i] |= bool(done[i])
        for i in range(num_envs):
            if chunk_done[i]:
                # env auto-reset inside this window -> drop the straddling chunk,
                # close the episode, restart buffer from the current (post-reset) frame.
                flush(i)
                reset_buf(i)
                if count >= n_episodes:
                    break
            else:
                ep_f[i].extend(chunk_f[i])              # +action_repeat frames
                ep_a[i].extend(chunk_a[i])              # +action_repeat per-frame actions
                ep_k[i] += 1
                if ep_k[i] >= max_actions:
                    flush(i)
                    reset_buf(i)
                    if count >= n_episodes:
                        break
    return out_f, out_a


def collect_single_scene(env_name, n_episodes, max_actions, num_envs, action_repeat,
                         level, stream_seed0):
    """Collect episodes that ALL start from the SAME fixed CoinRun level.

    CoinRun's vecenv auto-resets a done env to a NEW random level (num_levels=1 only
    constrains the INITIAL levels), so we cannot rely on auto-reset to stay on `level`.
    Instead we run num_envs parallel envs all seeded to `level`, keep only each env's
    FIRST episode (before any auto-reset), then DESTROY and REBUILD the venv for the
    next batch — every rebuild starts cleanly at `level` (verified: identical init
    frames). Per-frame actions come from per-env ActionStreams seeded stream_seed0+i,
    advanced across batches so trajectories stay diverse. FLOW-ONLY: only frames +
    per-frame actions are stored; the env `done` signal only ends the first episode."""
    from procgen import ProcgenEnv
    out_f, out_a = [], []
    streams = [ActionStream(np.random.RandomState(stream_seed0 + i)) for i in range(num_envs)]
    batch = 0
    while len(out_f) < n_episodes:
        venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=1,
                          start_level=int(level), distribution_mode="hard")
        obs = venv.reset()["rgb"]
        ep_f = [[obs[i].copy()] for i in range(num_envs)]
        ep_a = [[] for _ in range(num_envs)]
        ep_k = [0] * num_envs
        finalized = [False] * num_envs           # env done its first episode -> stop recording
        for t in range(max_actions):
            all_done = True
            chunk_f = [[] for _ in range(num_envs)]
            chunk_a = [[] for _ in range(num_envs)]
            chunk_done = np.zeros(num_envs, bool)
            for _j in range(action_repeat):
                acts = np.array([streams[i].next() for i in range(num_envs)], np.int32)
                obs, rew, done, _ = venv.step(acts)
                obs = obs["rgb"]
                for i in range(num_envs):
                    if finalized[i]:
                        continue
                    chunk_f[i].append(obs[i].copy()); chunk_a[i].append(int(acts[i]))
                    chunk_done[i] |= bool(done[i])
            for i in range(num_envs):
                if finalized[i]:
                    continue
                all_done = False
                if chunk_done[i]:
                    finalized[i] = True            # drop straddling window; first episode ends
                else:
                    ep_f[i].extend(chunk_f[i]); ep_a[i].extend(chunk_a[i]); ep_k[i] += 1
            if all_done:
                break
        for i in range(num_envs):
            if ep_k[i] >= MIN_ACTIONS and len(out_f) < n_episodes:
                out_f.append(np.array(ep_f[i], np.uint8))
                out_a.append(np.array(ep_a[i], np.int32))
        del venv
        batch += 1
        if batch % 20 == 0:
            print(f"    [single-scene] {len(out_f)}/{n_episodes} eps ({batch} batches)", flush=True)
    return out_f[:n_episodes], out_a[:n_episodes]


def collect_paired(env_name, seeds, max_actions, action_repeat):
    """One episode per seed; replay the seed's deterministic PER-FRAME action stream.
    Each latent window = action_repeat frames (each its own action). A window
    containing a reset is dropped (episode ends at last complete latent). ep_a stores
    per-frame actions (4K for K latents). FLOW-ONLY: only frames + per-frame actions
    are stored; the env `done` signal only ends the episode at a reset boundary."""
    from procgen import ProcgenEnv
    out_f, out_a, out_seed = [], [], []
    for s in seeds:
        venv = ProcgenEnv(num_envs=1, env_name=env_name,
                          num_levels=1, start_level=int(s), distribution_mode="hard")
        obs = venv.reset()["rgb"]
        # deterministic per-frame action stream for the whole episode (frame-aligned)
        pf_actions = sample_actions_for_seed(s, max_actions * action_repeat)
        f = [obs[0].copy()]; a = []; k = 0
        for t in range(max_actions):
            chunk_f = []; chunk_a = []; chunk_done = False
            for j in range(action_repeat):
                act = np.array([pf_actions[t * action_repeat + j]], np.int32)
                obs, rew, done, _ = venv.step(act)
                obs = obs["rgb"]
                chunk_f.append(obs[0].copy()); chunk_a.append(int(act[0]))
                chunk_done |= bool(done[0])
            if chunk_done:
                break                              # drop straddling window; episode ends here
            f.extend(chunk_f); a.extend(chunk_a); k += 1
        if k >= MIN_ACTIONS:
            out_f.append(np.array(f, np.uint8))
            out_a.append(np.array(a, np.int32))
            out_seed.append(int(s))
        venv.close() if hasattr(venv, "close") else None
    return out_f, out_a, out_seed


def save_npz(path, fs, as_, seeds=None, action_repeat=1):
    # episode_lengths = K latent steps per episode. actions are PER-FRAME (4K per
    # episode, aligned to frames[1:]), so K = len(actions)//action_repeat. FLOW-ONLY:
    # no reward/done fields are stored.
    lengths = np.array([len(a) // action_repeat for a in as_], np.int32)  # K per episode
    flat_f = np.concatenate(fs, 0)
    flat_a = np.concatenate(as_, 0)
    assert len(flat_a) == action_repeat * int(lengths.sum()), \
        f"per-frame actions {len(flat_a)} != {action_repeat}*latent steps {int(lengths.sum())}"
    kw = dict(frames=flat_f, actions=flat_a,
              episode_lengths=lengths, action_repeat=np.int32(action_repeat),
              per_frame_actions=np.bool_(True))
    if seeds is not None:
        kw["seeds"] = np.array(seeds, np.int64)
    np.savez_compressed(path, **kw)
    mb = os.path.getsize(path) / 1e6
    print(f"  saved {os.path.basename(path)}: {len(lengths)} eps, "
          f"{len(flat_f)} frames, {len(flat_a)} per-frame actions, {mb:.0f}MB", flush=True)


def store_conditions(vdir, args):
    """Persist the code condition(s) into the variant dir: source.cpp always,
    plus config.yaml when a --config is given (the declarative code condition the
    text encoder reads via precompute --code-source yaml)."""
    shutil.copy(args.src, os.path.join(vdir, "source.cpp"))
    if args.config:
        shutil.copy(args.config, os.path.join(vdir, "config.yaml"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--env", default="coinrun")
    ap.add_argument("--out", required=True)              # dataset root
    ap.add_argument("--src", required=True)              # current coinrun.cpp (already modified)
    ap.add_argument("--config", default=None,            # config.yaml = declarative code condition
                    help="YAML config for this variant; stored as <out>/<variant>/config.yaml "
                         "and fed to the text encoder (precompute --code-source yaml)")
    ap.add_argument("--unpaired", type=int, default=2200)
    ap.add_argument("--max-steps", type=int, default=60, help="max ACTIONS (latent steps) per episode")
    ap.add_argument("--action-repeat", type=int, default=4, help="env steps per action (frames/action)")
    ap.add_argument("--num-envs", type=int, default=128)
    ap.add_argument("--paired-start", type=int, default=10000)
    ap.add_argument("--paired-count", type=int, default=300)
    ap.add_argument("--eval-start", type=int, default=20000)
    ap.add_argument("--eval-count", type=int, default=100)
    # single-scene overfit: fix ONE CoinRun level; train/eval differ only by action stream
    ap.add_argument("--single-scene", action="store_true",
                    help="overfit one fixed level; skip paired; eval = same scene, held-out action streams")
    ap.add_argument("--level", type=int, default=0, help="fixed CoinRun level id for --single-scene")
    ap.add_argument("--eval-unpaired", type=int, default=1000,
                    help="--single-scene: #eval episodes (same scene, held-out action streams)")
    args = ap.parse_args()

    vdir = os.path.join(args.out, args.variant)
    os.makedirs(vdir, exist_ok=True)
    t0 = time.time()
    ar = args.action_repeat

    if args.single_scene:
        # single fixed scene: train + eval share the level, differ only by action stream.
        # eval streams are held-out (disjoint seed range) so we test NEW action sequences
        # on the SAME scene — the honest "action following" generalization test.
        print(f"[{args.variant}] single-scene overfit: level={args.level} "
              f"action_repeat={ar} train={args.unpaired} eval={args.eval_unpaired}", flush=True)
        fs, as_ = collect_single_scene(args.env, args.unpaired, args.max_steps,
                                       args.num_envs, ar, args.level, stream_seed0=1000)
        save_npz(os.path.join(vdir, "episodes_train.npz"), fs, as_, action_repeat=ar)
        # held-out action streams: seed range far from train's [1000, 1000+num_envs)
        fs, as_ = collect_single_scene(args.env, args.eval_unpaired, args.max_steps,
                                       args.num_envs, ar, args.level, stream_seed0=900000)
        save_npz(os.path.join(vdir, "episodes_eval.npz"), fs, as_, action_repeat=ar)
        store_conditions(vdir, args)
        print(f"[{args.variant}] done in {time.time()-t0:.0f}s -> {vdir}", flush=True)
        return

    print(f"[{args.variant}] collecting... (action_repeat={ar})", flush=True)

    fs, as_ = collect_unpaired(args.env, args.unpaired, args.max_steps, args.num_envs, ar)
    save_npz(os.path.join(vdir, "episodes_train.npz"), fs, as_, action_repeat=ar)

    pseeds = list(range(args.paired_start, args.paired_start + args.paired_count))
    fs, as_, sd = collect_paired(args.env, pseeds, args.max_steps, ar)
    save_npz(os.path.join(vdir, "episodes_paired.npz"), fs, as_, sd, action_repeat=ar)

    eseeds = list(range(args.eval_start, args.eval_start + args.eval_count))
    fs, as_, sd = collect_paired(args.env, eseeds, args.max_steps, ar)
    save_npz(os.path.join(vdir, "episodes_eval.npz"), fs, as_, sd, action_repeat=ar)

    store_conditions(vdir, args)
    print(f"[{args.variant}] done in {time.time()-t0:.0f}s -> {vdir}", flush=True)


if __name__ == "__main__":
    main()
