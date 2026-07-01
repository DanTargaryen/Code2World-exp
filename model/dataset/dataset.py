"""Code2World dataset: loads precomputed VAE latents + Qwen code embeds, samples
T-frame windows. Each item = (latents, actions, rewards, dones, code) for one
window from one (variant, episode).

Splits:
  - train:  unpaired (episodes_train) + paired (episodes_paired) across all variants
  - eval:   episodes_eval (held-out paired seeds)
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset


class Code2WorldDataset(Dataset):
    def __init__(self, root, split="train", window=32, variants=None):
        """
        split: "train" -> episodes_train + episodes_paired ; "eval" -> episodes_eval
        window: T frames per sample (input T + target T means T+1 latents).
        """
        self.root = root
        self.window = window
        self.code_embeds = torch.load(os.path.join(root, "code_embeds.pt"))
        if variants is None:
            variants = list(self.code_embeds.keys())
        self.variants = variants

        if split == "train":
            split_files = ["episodes_train", "episodes_paired"]
        elif split == "eval":
            split_files = ["episodes_eval"]
        else:
            raise ValueError(split)

        # build a flat index of (variant, latent_tensor_ref, ep_start, ep_len)
        self.shards = []       # list of dicts holding loaded tensors
        self.index = []        # (shard_id, ep_frame_start, ep_act_start, ep_len)
        lat_dir = os.path.join(root, "latents")
        for v in variants:
            for sf in split_files:
                pt = os.path.join(lat_dir, f"{v}__{sf}.pt")
                if not os.path.exists(pt):
                    continue
                d = torch.load(pt)
                shard_id = len(self.shards)
                self.shards.append({"variant": v,
                                    "latents": d["latents"],          # (Nframe, z, h, w) fp16
                                    "actions": d["actions"],
                                    "rewards": d["rewards"],
                                    "dones": d["dones"]})
                ep_len = d["episode_lengths"].tolist()
                f_cursor = 0   # frames: each ep has len+1 frames
                a_cursor = 0   # actions: each ep has len actions
                for L in ep_len:
                    if L >= window:   # need at least `window` actions
                        self.index.append((shard_id, f_cursor, a_cursor, L))
                    f_cursor += L + 1
                    a_cursor += L

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        shard_id, f0, a0, L = self.index[i]
        sh = self.shards[shard_id]
        T = self.window
        # random window start within the episode
        off = np.random.randint(0, L - T + 1)
        fs = f0 + off
        as_ = a0 + off
        lat = sh["latents"][fs: fs + T + 1].float()       # (T+1, z, h, w)
        act = sh["actions"][as_: as_ + T].long()           # (T,)
        rew = sh["rewards"][as_: as_ + T].float()
        done = sh["dones"][as_: as_ + T].long()
        code = self.code_embeds[sh["variant"]].float()     # (N_tok, 896)
        return {"latents": lat, "actions": act, "rewards": rew,
                "dones": done, "code": code, "variant": sh["variant"]}


def collate(batch):
    """Pad code to the max length in the batch (variants may have different token counts)."""
    T = batch[0]["latents"].shape[0]
    latents = torch.stack([b["latents"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])
    rewards = torch.stack([b["rewards"] for b in batch])
    dones = torch.stack([b["dones"] for b in batch])
    maxN = max(b["code"].shape[0] for b in batch)
    D = batch[0]["code"].shape[1]
    code = torch.zeros(len(batch), maxN, D)
    code_mask = torch.zeros(len(batch), maxN, dtype=torch.bool)
    for j, b in enumerate(batch):
        n = b["code"].shape[0]
        code[j, :n] = b["code"]
        code_mask[j, :n] = True
    return {"latents": latents, "actions": actions, "rewards": rewards,
            "dones": dones, "code": code, "code_mask": code_mask,
            "variant": [b["variant"] for b in batch]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world")
    ap.add_argument("--window", type=int, default=32)
    args = ap.parse_args()
    ds = Code2WorldDataset(args.root, split="train", window=args.window)
    print(f"train windows: {len(ds)}")
    s = ds[0]
    for k, v in s.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {v}")
    ev = Code2WorldDataset(args.root, split="eval", window=args.window)
    print(f"eval windows: {len(ev)}")
