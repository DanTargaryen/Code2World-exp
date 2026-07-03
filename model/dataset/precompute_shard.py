"""Sharded latent precompute: split one split's episodes into N contiguous shards,
encode shard `i` on its own GPU, save latents_shard_{i}.pt. A separate --merge pass
concatenates shards in order into the final {variant}__{split}.pt (dataset format).

Usage (per GPU worker):
  CUDA_VISIBLE_DEVICES=k python precompute_shard.py --root R --variant base \
      --split episodes_train --shard k --nshards 8 --device cuda:0
Then merge:
  python precompute_shard.py --root R --variant base --split episodes_train \
      --nshards 8 --merge
"""
import os, sys, time, argparse
import numpy as np, torch
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # model/  (for models.vae)
sys.path.insert(0, _HERE)                      # model/dataset/  (for precompute)
from models.vae import WanVAEWrapper
from precompute import encode_frames_temporal, encode_frames


def frame_offset(ep_lengths, ar, ep_start):
    """flat frame index where episode `ep_start` begins (sum of 4K+1 over prior eps)."""
    c = 0
    for K in ep_lengths[:ep_start]:
        c += ar * int(K) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_train")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    lat_dir = os.path.join(args.root, "latents")
    os.makedirs(lat_dir, exist_ok=True)
    npz = os.path.join(args.root, args.variant, args.split + ".npz")
    d = np.load(npz)
    ep_lengths = d["episode_lengths"]
    ar = int(d["action_repeat"]) if "action_repeat" in d else 1
    n_ep = len(ep_lengths)

    if args.merge:
        # concat shards in order + attach per-episode fields (unchanged, full split)
        parts = []
        for s in range(args.nshards):
            p = os.path.join(lat_dir, f"_shard_{args.variant}__{args.split}_{s}of{args.nshards}.pt")
            parts.append(torch.load(p)["latents"])
        lat = torch.cat(parts, 0)
        exp = int(ep_lengths.sum()) + n_ep     # sum(K+1)
        assert lat.shape[0] == exp, f"merged latents {lat.shape[0]} != expected {exp}"
        out = os.path.join(lat_dir, f"{args.variant}__{args.split}.pt")
        torch.save({"latents": lat,
                    "actions": torch.from_numpy(d["actions"]),
                    "rewards": torch.from_numpy(d["rewards"]),
                    "dones": torch.from_numpy(d["dones"]),
                    "episode_lengths": torch.from_numpy(ep_lengths),
                    "action_repeat": ar,
                    "seeds": torch.from_numpy(d["seeds"]) if "seeds" in d else None},
                   out)
        print(f"merged {args.nshards} shards -> {out} ({lat.shape}, "
              f"{os.path.getsize(out)/1e6:.0f}MB)", flush=True)
        for s in range(args.nshards):
            os.remove(os.path.join(lat_dir, f"_shard_{args.variant}__{args.split}_{s}of{args.nshards}.pt"))
        return

    # worker: encode this shard's contiguous ep range
    per = (n_ep + args.nshards - 1) // args.nshards
    e0 = args.shard * per
    e1 = min(n_ep, e0 + per)
    if e0 >= n_ep:
        print(f"[shard {args.shard}] empty, skip", flush=True)
        return
    ep_slice = ep_lengths[e0:e1]
    f0 = frame_offset(ep_lengths, ar, e0)
    f1 = frame_offset(ep_lengths, ar, e1)
    frames = d["frames"][f0:f1]
    t0 = time.time()
    vae = WanVAEWrapper(args.vae, device=args.device)
    if ar > 1:
        lat = encode_frames_temporal(vae, frames, ep_slice, ar, device=args.device)
    else:
        lat = encode_frames(vae, frames, device=args.device)
    out = os.path.join(lat_dir, f"_shard_{args.variant}__{args.split}_{args.shard}of{args.nshards}.pt")
    torch.save({"latents": lat}, out)
    print(f"[shard {args.shard}] ep[{e0}:{e1}] -> {lat.shape} {os.path.getsize(out)/1e6:.0f}MB "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
