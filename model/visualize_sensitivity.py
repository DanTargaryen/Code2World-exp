"""Visualize code sensitivity: fix the SAME init frame + SAME action sequence,
run all 7 variant codes through the model, produce a side-by-side video.

The ONLY thing differing across columns is the code condition, so any visible
difference between columns is pure code sensitivity.

Layout per frame:  [ GT(base真值) | base | fast | slow | lowgrav | highgrav | highjump | lowjump ]
Runs several seeds (episodes) so you have multiple examples to inspect.
"""
import os, sys, argparse, subprocess
import numpy as np
import torch
import torch.nn.functional as F
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.causal_dit import CausalDiT
from models.vae import WanVAEWrapper

VARIANTS = ["base", "fast", "slow", "lowgrav", "highgrav", "highjump", "lowjump"]


def load_ep(root, variant, ep_idx, split="episodes_eval"):
    npz = np.load(os.path.join(root, variant, f"{split}.npz"))
    el = npz["episode_lengths"]
    f0 = int(el[:ep_idx].sum()) + ep_idx
    a0 = int(el[:ep_idx].sum())
    L = int(el[ep_idx])
    frames = npz["frames"][f0:f0 + L + 1]
    actions = npz["actions"][a0:a0 + L]
    lat = torch.load(os.path.join(root, "latents", f"{variant}__{split}.pt"),
                     map_location="cpu")["latents"][f0:f0 + L + 1].float()
    return frames, actions, lat, L


@torch.no_grad()
def batched_rollout(model, init_lat, actions, codes, masks, num_actions, dev, context):
    """init_lat (B,z,h,w) same init repeated; actions (L,); codes (B,N,D); masks (B,N).
    Returns pred latents (B,L,z,h,w)."""
    B = init_lat.shape[0]
    L = len(actions)
    hist = init_lat.unsqueeze(1)              # (B,1,z,h,w)
    preds = []
    for t in range(L):
        inp = hist[:, -context:]
        tlen = inp.shape[1]
        a = torch.from_numpy(actions[max(0, t + 1 - context):t + 1].astype(np.int64)).to(dev)
        a_oh = F.one_hot(a, num_actions).float().unsqueeze(0).expand(B, -1, -1)  # (B,tlen,A)
        pred, _, _ = model(inp, a_oh, codes, masks)
        nxt = pred[:, -1:]
        preds.append(nxt)
        hist = torch.cat([hist, nxt], dim=1)
    return torch.cat(preds, dim=1)            # (B,L,z,h,w)


def save_video(frames_uint8, path, fps=10):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames_uint8[0].shape[:2]
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames_uint8:
        p.stdin.write(np.ascontiguousarray(f, np.uint8).tobytes())
    p.stdin.close(); p.wait()


def label_cell(img, text, cell=128):
    img = cv2.resize(img, (cell, cell), interpolation=cv2.INTER_NEAREST)
    bar = np.full((16, cell, 3), 20, np.uint8)
    cv2.putText(bar, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
    return np.concatenate([bar, img], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world")
    ap.add_argument("--ckpt", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world/ckpt_final.pt")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--eps", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--cell", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="sensitivity_videos")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    ck = torch.load(args.ckpt, map_location=dev)
    ca = ck.get("args", {})
    code_bank = torch.load(os.path.join(args.root, "code_embeds.pt"), map_location="cpu")
    code_dim = next(iter(code_bank.values())).shape[1]
    model = CausalDiT(latent_dim=16, embed_dim=ca.get("embed_dim", 512),
                      num_layers=ca.get("num_layers", 12), num_heads=ca.get("num_heads", 8),
                      num_actions=args.num_actions, spatial_size=8,
                      max_frames=ca.get("window", 32), code_dim=code_dim).to(dev)
    # strict=False: old checkpoints carry removed reward/done head weights.
    model.load_state_dict(ck["model"], strict=False); model.eval()
    vae = WanVAEWrapper(args.vae, device=dev)
    print(f"loaded model step {ck.get('step')}", flush=True)

    # pad 7 variant codes to same length + mask
    maxN = max(code_bank[v].shape[0] for v in VARIANTS)
    codes = torch.zeros(len(VARIANTS), maxN, code_dim)
    masks = torch.zeros(len(VARIANTS), maxN, dtype=torch.bool)
    for i, v in enumerate(VARIANTS):
        c = code_bank[v].float(); n = c.shape[0]
        codes[i, :n] = c; masks[i, :n] = True
    codes = codes.to(dev); masks = masks.to(dev)

    for ep in args.eps:
        bframes, bactions, blat, bL = load_ep(args.root, "base", ep)
        init = blat[0:1].to(dev).expand(len(VARIANTS), -1, -1, -1).contiguous()  # (7,z,h,w)
        pred = batched_rollout(model, init, bactions, codes, masks,
                               args.num_actions, dev, args.context)     # (7,L,z,h,w)
        # decode each variant
        gen = {}
        for i, v in enumerate(VARIANTS):
            gl = torch.cat([blat[0:1].to(dev), pred[i]], 0).unsqueeze(0)   # (1,L+1,z,h,w)
            gf = vae.decode(gl)[0]
            gen[v] = [(gf[t].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                      for t in range(gf.shape[0])]
        # GT(base) reference = true base frames
        gt = [bframes[t] for t in range(len(bframes))]

        L = min(len(gt), *[len(gen[v]) for v in VARIANTS])
        out_frames = []
        for t in range(L):
            cols = [label_cell(gt[t], "GT(base)", args.cell)]
            for v in VARIANTS:
                cols.append(label_cell(gen[v][t], v, args.cell))
            sep = np.full((cols[0].shape[0], 2, 3), 255, np.uint8)
            row = cols[0]
            for c in cols[1:]:
                row = np.concatenate([row, sep, c], axis=1)
            out_frames.append(row)
        path = os.path.join(args.out, f"sensitivity_ep{ep}.mp4")
        save_video(out_frames, path, fps=10)
        print(f"  saved {path}  ({L} frames, {out_frames[0].shape[1]}x{out_frames[0].shape[0]})", flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    main()
