"""Code2World dataset: loads precomputed VAE latents + Qwen code embeds, samples
T-latent windows. Each item = (latents, actions, code) for one window from one
(variant, episode). Flow-only: reward/done are not loaded (mechanism removed).

PER-FRAME actions: an episode of K latents stores 4K per-frame actions (action_repeat
frames per latent). A window of T latents therefore carries T+1 latents and
action_repeat*T per-frame actions (aligned to the window's frames[1:]).

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
        window: T latents per sample (input T + target means T+1 latents).
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
                ar = int(d.get("action_repeat", 1))
                # per-frame if actions count == action_repeat * total latent steps
                n_lat_steps = int(d["episode_lengths"].sum())
                self.per_frame = bool(d.get("per_frame_actions", False)) or \
                    (len(d["actions"]) == ar * n_lat_steps and ar > 1)
                self.action_repeat = ar if self.per_frame else 1
                shard_id = len(self.shards)
                self.shards.append({"variant": v,
                                    "latents": d["latents"],          # (Nframe, z, h, w) fp16
                                    "actions": d["actions"]})
                ep_len = d["episode_lengths"].tolist()
                f_cursor = 0   # latents: each ep has L+1 latents
                a_cursor = 0   # actions: action_repeat*L per ep (per-frame) or L (legacy)
                for L in ep_len:
                    if L >= window:   # need at least `window` latents
                        self.index.append((shard_id, f_cursor, a_cursor, L))
                    f_cursor += L + 1
                    a_cursor += L * self.action_repeat

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        shard_id, f0, a0, L = self.index[i]
        sh = self.shards[shard_id]
        T = self.window
        ar = self.action_repeat
        # random window start within the episode (in latent steps)
        off = np.random.randint(0, L - T + 1)
        fs = f0 + off
        lat = sh["latents"][fs: fs + T + 1].float()          # (T+1, z, h, w)
        # actions: per-frame -> ar*T for this window (a0 already counts per-frame);
        # per-latent (ar=1) -> T. Only the in-episode offset scales by ar.
        as_ = a0 + off * ar
        act = sh["actions"][as_: as_ + T * ar].long()        # (ar*T,) or (T,)
        code = self.code_embeds[sh["variant"]].float()       # (N_tok, 896)
        return {"latents": lat, "actions": act,
                "code": code, "variant": sh["variant"]}


def collate(batch):
    """Pad code to the max length in the batch (variants may have different token counts)."""
    latents = torch.stack([b["latents"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])
    maxN = max(b["code"].shape[0] for b in batch)
    D = batch[0]["code"].shape[1]
    code = torch.zeros(len(batch), maxN, D)
    code_mask = torch.zeros(len(batch), maxN, dtype=torch.bool)
    for j, b in enumerate(batch):
        n = b["code"].shape[0]
        code[j, :n] = b["code"]
        code_mask[j, :n] = True
    return {"latents": latents, "actions": actions,
            "code": code, "code_mask": code_mask,
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
