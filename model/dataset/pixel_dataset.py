"""Pixel-space dataset: reads raw frames from the existing npz (no VAE precompute),
slides a W+1-frame window, returns patchified+normalized "latents" (W+1,192,8,8)
plus per-frame actions and the code embedding.

Every frame is modeled directly (no temporal compression), so a window of W+1
frames = W+1 "latent steps". action_repeat=4 in the data means every 4 consecutive
frames share one action: window position j (j>=1, i.e. episode-frame off+j) is
driven by actions[(off+j-1)//action_repeat].

Frames are kept as uint8 in RAM (~1.1GB/shard) and converted per item. Defaults to
the `base` variant only to bound RAM for the quick path-C experiment; pass more
variants to include them (code cross-attn still batches via the padding mask).
"""
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from einops import rearrange

PATCH = 8


class PixelDataset(Dataset):
    def __init__(self, root, split="train", window=41, variants=("base",),
                 mean=None, std=None):
        self.root = root
        self.window = window                              # frames-1; sample = window+1 frames
        self.patch = PATCH
        self.code_embeds = torch.load(os.path.join(root, "code_embeds.pt"))
        self.variants = list(variants)

        # normalization: standard bounded [0,1] -> [-1,1] (mean 0.5, std 0.5).
        # Robust for pixels (no outliers); the flow target z1-eps stays well-scaled.
        if mean is None:
            mean = [0.5, 0.5, 0.5]
        if std is None:
            std = [0.5, 0.5, 0.5]
        self.mean = np.asarray(mean, np.float32).reshape(1, 1, 3)   # broadcast over (H,W,3)
        self.std = np.asarray(std, np.float32).reshape(1, 1, 3)

        if split == "train":
            split_files = ["episodes_train", "episodes_paired"]
        elif split == "eval":
            split_files = ["episodes_eval"]
        else:
            raise ValueError(split)

        self.shards = []
        self.index = []       # (shard_id, ep_frame_start, ep_action_start, K, action_repeat)
        for v in self.variants:
            for sf in split_files:
                npz = os.path.join(root, v, sf + ".npz")
                if not os.path.exists(npz):
                    continue
                d = np.load(npz)
                ar = int(d["action_repeat"]) if "action_repeat" in d else 1
                sid = len(self.shards)
                self.shards.append({"variant": v,
                                    "frames": d["frames"],        # (Nf,H,W,3) uint8
                                    "actions": d["actions"]})     # (Na,) per-action
                epl = d["episode_lengths"].tolist()
                f_cursor = 0     # frames: each ep has 4K+1
                a_cursor = 0     # actions: each ep has K
                for K in epl:
                    nfr = ar * K + 1
                    # need window+1 frames within this episode
                    if nfr >= window + 1:
                        self.index.append((sid, f_cursor, a_cursor, K, ar))
                    f_cursor += nfr
                    a_cursor += K

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        sid, f0, a0, K, ar = self.index[i]
        sh = self.shards[sid]
        W = self.window
        nfr = ar * K + 1
        off = np.random.randint(0, nfr - (W + 1) + 1)          # window start within episode
        fs = f0 + off
        frames = sh["frames"][fs: fs + W + 1]                  # (W+1,H,W,3) uint8
        # normalize + patchify -> (W+1, 192, 8, 8)
        x = frames.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x).permute(0, 3, 1, 2)            # (W+1,3,H,W)
        lat = rearrange(x, "t c (h p1) (w p2) -> t (c p1 p2) h w",
                        p1=self.patch, p2=self.patch)          # (W+1,192,8,8)
        # per-frame actions: window position j (j=1..W) = ep-frame off+j, action idx (off+j-1)//ar
        act = np.empty(W, np.int64)
        for j in range(1, W + 1):
            act[j - 1] = sh["actions"][a0 + (off + j - 1) // ar]
        act = torch.from_numpy(act)
        code = self.code_embeds[sh["variant"]].float()
        return {"latents": lat.float(), "actions": act, "code": code,
                "variant": sh["variant"]}


def collate(batch):
    latents = torch.stack([b["latents"] for b in batch])
    actions = torch.stack([b["actions"] for b in batch])
    maxN = max(b["code"].shape[0] for b in batch)
    D = batch[0]["code"].shape[1]
    code = torch.zeros(len(batch), maxN, D)
    code_mask = torch.zeros(len(batch), maxN, dtype=torch.bool)
    for j, b in enumerate(batch):
        n = b["code"].shape[0]
        code[j, :n] = b["code"]; code_mask[j, :n] = True
    # pixel path has no reward/done supervision (frames only); return zeros for API compat
    B, T = actions.shape
    rewards = torch.zeros(B, T)
    dones = torch.zeros(B, T, dtype=torch.long)
    return {"latents": latents, "actions": actions, "rewards": rewards, "dones": dones,
            "code": code, "code_mask": code_mask, "variant": [b["variant"] for b in batch]}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc")
    ap.add_argument("--window", type=int, default=41)
    ap.add_argument("--variants", nargs="*", default=["base"])
    ap.add_argument("--stats", action="store_true", help="compute+cache per-channel pixel stats")
    args = ap.parse_args()

    if args.stats:
        # stats over the base train frames (representative; variants share look)
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.pixel_codec import compute_pixel_stats
        npz = np.load(os.path.join(args.root, "base", "episodes_train.npz"))
        mean, std = compute_pixel_stats(npz["frames"])
        out = {"mean": [float(x) for x in mean], "std": [float(x) for x in std]}
        json.dump(out, open(os.path.join(args.root, "pixel_stats.json"), "w"), indent=2)
        print("saved pixel_stats.json:", out)
    else:
        ds = PixelDataset(args.root, "train", args.window, args.variants)
        ev = PixelDataset(args.root, "eval", args.window, args.variants)
        print(f"train windows: {len(ds)} | eval windows: {len(ev)}")
        s = ds[0]
        for k, v in s.items():
            print(f"  {k}:", tuple(v.shape) if torch.is_tensor(v) else v)
