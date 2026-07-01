"""Precompute pass over the collected dataset:
  1. VAE-encode all frames -> fp16 latents (16, 8, 8), stored per variant/split.
     - per_frame mode (legacy): each frame -> 1 latent (T=1 chunks, no temporal compression).
     - temporal mode (action_repeat>1): each episode's 4K+1 frames -> K+1 latents via the
       Wan causal 3D VAE (1 action <-> 1 latent <-> 4 frames). episode_lengths stays K, so
       frames-per-ep = K+1 latents, matching the per_frame layout the dataset expects.
  2. Qwen-encode each variant's source.cpp -> code_embeds.pt {variant: (N_tok, 896)}.

Mode is auto-detected from the npz `action_repeat` field (>1 -> temporal).
Run on the GPU pod after build_dataset.py finishes.
"""
import os, sys, time, json, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # model/
from models.vae import WanVAEWrapper

SPLITS = ["episodes_train", "episodes_paired", "episodes_eval"]


@torch.no_grad()
def encode_frames(vae, frames_uint8, batch=256, device="cuda:0"):
    """Per-frame encode: frames_uint8 (N, H, W, 3) -> latents (N, z, h, w) fp16."""
    N = len(frames_uint8)
    outs = []
    for i in range(0, N, batch):
        chunk = frames_uint8[i:i+batch]
        x = torch.from_numpy(chunk).permute(0, 3, 1, 2).float() / 255.0  # (b,3,H,W)
        x = x.unsqueeze(1).to(device)                                     # (b,1,3,H,W)
        lat = vae.encode(x)                                               # (b,1,z,h,w)
        outs.append(lat[:, 0].half().cpu())
    return torch.cat(outs, 0)  # (N, z, h, w) fp16


@torch.no_grad()
def encode_frames_temporal(vae, frames_uint8, ep_lengths, action_repeat, device="cuda:0",
                           ep_batch=16):
    """Temporal-compressing per-episode encode, BATCHED.

    Each episode has K actions -> 4K+1 frames -> K+1 latents. Episodes are grouped
    into mini-batches of `ep_batch`; within a batch every clip is right-padded (in
    time) to the batch's max frame length and encoded in one VAE call, then sliced
    back to its true K+1 latents. The Wan VAE is causal in time, so trailing pad
    frames cannot corrupt earlier latents (verified: exact match). Returns a flat
    (sum(K_i+1), z, h, w) fp16 tensor matching the legacy per-frame layout."""
    ep_lengths = [int(k) for k in ep_lengths.tolist()]
    # episode frame slices into the flat array
    starts, c = [], 0
    for K in ep_lengths:
        starts.append(c); c += action_repeat * K + 1
    outs = []
    for i in range(0, len(ep_lengths), ep_batch):
        Ks = ep_lengths[i:i + ep_batch]
        clips = []
        for j, K in enumerate(Ks):
            s = starts[i + j]; n = action_repeat * K + 1
            x = torch.from_numpy(frames_uint8[s:s + n]).permute(0, 3, 1, 2).float() / 255.0
            clips.append(x)                                  # (n,3,H,W)
        Tmax = max(c.shape[0] for c in clips)
        H, W = clips[0].shape[2:]
        # right-pad in time, stack -> (b, 3, Tmax, H, W) in [-1,1]
        bx = torch.zeros(len(clips), 3, Tmax, H, W)
        for j, x in enumerate(clips):
            bx[j, :, :x.shape[0]] = x.permute(1, 0, 2, 3)
        bx = (bx.to(device, vae.dtype) * 2.0 - 1.0)
        lat = vae.vae.encode(bx).latent_dist.mode()          # (b, z, Lmax, h, w)
        lat = (lat - vae.latents_mean) / vae.latents_std
        for j, K in enumerate(Ks):
            outs.append(lat[j, :, :K + 1].permute(1, 0, 2, 3).contiguous().half().cpu())  # (K+1,z,h,w)
    return torch.cat(outs, 0)


def precompute_latents(root, vae_path, variants, device):
    vae = WanVAEWrapper(vae_path, device=device)
    lat_dir = os.path.join(root, "latents")
    os.makedirs(lat_dir, exist_ok=True)
    for v in variants:
        for split in SPLITS:
            npz = os.path.join(root, v, split + ".npz")
            if not os.path.exists(npz):
                continue
            d = np.load(npz)
            ep_lengths = d["episode_lengths"]
            ar = int(d["action_repeat"]) if "action_repeat" in d else 1
            t0 = time.time()
            if ar > 1:
                lat = encode_frames_temporal(vae, d["frames"], ep_lengths, ar, device=device)
                mode = f"temporal(ar={ar})"
            else:
                lat = encode_frames(vae, d["frames"], device=device)
                mode = "per_frame"
            out = os.path.join(lat_dir, f"{v}__{split}.pt")
            torch.save({"latents": lat,                       # (sum(K+1), z, h, w) fp16
                        "actions": torch.from_numpy(d["actions"]),
                        "rewards": torch.from_numpy(d["rewards"]),
                        "dones": torch.from_numpy(d["dones"]),
                        "episode_lengths": torch.from_numpy(ep_lengths),
                        "action_repeat": ar,
                        "seeds": torch.from_numpy(d["seeds"]) if "seeds" in d else None},
                       out)
            mb = os.path.getsize(out) / 1e6
            print(f"  {v}/{split} [{mode}]: {lat.shape} -> {mb:.0f}MB ({time.time()-t0:.0f}s)", flush=True)


def precompute_code(root, qwen_path, variants, device, max_len=5120):
    from transformers import AutoModel, AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(qwen_path)
    model = AutoModel.from_pretrained(qwen_path, torch_dtype=torch.float32).to(device).eval()
    embeds = {}
    for v in variants:
        src_path = os.path.join(root, v, "source.cpp")
        src = open(src_path).read()
        ids = tok(src, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = model(**ids).last_hidden_state[0]  # (N_tok, 896)
        embeds[v] = out.half().cpu()
        print(f"  code[{v}]: {tuple(embeds[v].shape)}", flush=True)
    torch.save(embeds, os.path.join(root, "code_embeds.pt"))
    print(f"  saved code_embeds.pt", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--qwen", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/Qwen2.5-0.5B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", choices=["latents", "code", "both"], default="both")
    args = ap.parse_args()

    manifest = json.load(open(os.path.join(args.root, "variants.json")))
    variants = list(manifest.keys())
    print(f"variants: {variants}", flush=True)

    if args.only in ("code", "both"):
        print("== precompute code embeds ==", flush=True)
        precompute_code(args.root, args.qwen, variants, args.device)
    if args.only in ("latents", "both"):
        print("== precompute latents ==", flush=True)
        precompute_latents(args.root, args.vae, variants, args.device)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
