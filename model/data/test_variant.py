import numpy as np
from procgen import ProcgenEnv

def run_fixed(seed=0, steps=40):
    venv = ProcgenEnv(num_envs=1, env_name='coinrun', num_levels=1,
                      start_level=seed, distribution_mode='hard')
    venv.reset()
    actions_seq = [7,7,7,9,7,7,7,7,9,7]*4
    frames = []
    for t in range(steps):
        a = np.array([actions_seq[t]])
        obs, rew, done, info = venv.step(a)
        frames.append(obs['rgb'][0].astype(np.int64))
    sig = np.array([f.sum() for f in frames])
    return sig

sig = run_fixed(seed=0, steps=40)
print("SIG", " ".join(str(x) for x in sig[:10]))
print("CHECKSUM", sig.sum())
