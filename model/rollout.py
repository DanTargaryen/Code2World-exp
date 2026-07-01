"""Autoregressive rollout: given frame_0 + action sequence + code, generate a video
by feeding the model's own predicted latents back in (true AR, not teacher forcing).

Outputs three videos per episode:
  - original.mp4   : true Procgen frames (ground truth)
  - generated.mp4  : model AR rollout, decoded
  - compare.mp4    : side-by-side [original | generated]
"""
import os, sys, argparse, subprocess
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.causal_dit import CausalDiT
from models.vae import WanVAEWrapper


def load_episode(root, variant, split, ep_idx):
    """Load raw frames + precomputed latents + actions for one episode."""
    npz = np.load(os.path.join(root, variant, f"{split}.npz"))
    ep_len = npz["episode_lengths"]
    f0 = int(ep_len[:ep_idx].sum()) + ep_idx     # frames have +1 per ep
    a0 = int(ep_len[:ep_idx].sum())
    L = int(ep_len[ep_idx])
    frames = npz["frames"][f0:f0 + L + 1]         # (L+1, H, W, 3) uint8  TRUE frames
    actions = npz["actions"][a0:a0 + L]           # (L,)
    # precomputed latents
    lat_all = torch.load(os.path.join(root, "latents", f"{variant}__{split}.pt"),
                         map_location="cpu")
    latents = lat_all["latents"][f0:f0 + L + 1].float()  # (L+1, z, h, w)
    return frames, actions, latents, L


def save_video(frames_uint8, path, fps=10, scale=6):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames_uint8[0].shape[:2]
    OH, OW = H * scale, W * scale
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import cv2
    for f in frames_uint8:
        img = cv2.resize(f, (OW, OH), interpolation=cv2.INTER_NEAREST)
        p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
    p.stdin.close(); p.wait()


@torch.no_grad()
def rollout(model, vae, init_latent, actions, code, num_actions, dev, context):
    """Autoregressive rollout.
    init_latent: (1, z, h, w) frame-0 latent
    actions: (L,) int
    code: (1, N, code_dim)
    Returns predicted latents (L, z, h, w) for frames 1..L.
    """
    z, h, w = init_latent.shape[1:]
    L = len(actions)
    hist = init_latent.unsqueeze(1)               # (1, 1, z, h, w) growing buffer
    preds = []
    for t in range(L):
        # use the last `context` frames as input window
        inp = hist[:, -context:]                   # (1, t', z, h, w)
        tlen = inp.shape[1]
        act_win = actions[max(0, t + 1 - context): t + 1]
        # pad action window to tlen (left side already aligned to inp frames)
        a = torch.from_numpy(act_win.astype(np.int64)).to(dev)
        a_oh = F.one_hot(a, num_actions).float().unsqueeze(0)   # (1, tlen, A)
        pred, _, _ = model(inp, a_oh, code)        # (1, tlen, z, h, w)
        next_lat = pred[:, -1:]                    # (1, 1, z, h, w) last-frame prediction
        preds.append(next_lat)
        hist = torch.cat([hist, next_lat], dim=1)
    pred_lat = torch.cat(preds, dim=1)[0]          # (L, z, h, w)
    return pred_lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world")
    ap.add_argument("--ckpt", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world/ckpt_final.pt")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_eval")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--context", type=int, default=32, help="AR context window (<= train window)")
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--out", default="rollout_out")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    # --- load checkpoint + infer config ---
    ck = torch.load(args.ckpt, map_location=dev)
    cargs = ck.get("args", {})
    z = 16; h = 8
    code_bank = torch.load(os.path.join(args.root, "code_embeds.pt"), map_location="cpu")
    code_dim = next(iter(code_bank.values())).shape[1]
    model = CausalDiT(latent_dim=z, embed_dim=cargs.get("embed_dim", 512),
                      num_layers=cargs.get("num_layers", 12), num_heads=cargs.get("num_heads", 8),
                      num_actions=args.num_actions, spatial_size=h,
                      max_frames=cargs.get("window", 32), code_dim=code_dim).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')})", flush=True)

    vae = WanVAEWrapper(args.vae, device=dev)

    # --- data ---
    frames, actions, latents, L = load_episode(args.root, args.variant, args.split, args.ep)
    print(f"episode: variant={args.variant} ep={args.ep} len={L}", flush=True)
    code = code_bank[args.variant].float().unsqueeze(0).to(dev)   # (1, N, code_dim)
    init_latent = latents[0:1].to(dev)                            # (1, z, h, w)

    # --- AR rollout ---
    pred_lat = rollout(model, vae, init_latent, actions, code,
                       args.num_actions, dev, args.context)        # (L, z, h, w)

    # --- decode generated; assemble videos ---
    gen_lat = torch.cat([init_latent, pred_lat], dim=0)           # (L+1, z, h, w) incl frame0
    gen_frames = vae.decode(gen_lat.unsqueeze(0))[0]              # (L+1, 3, H, W)
    gen_uint8 = [(gen_frames[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                 for i in range(gen_frames.shape[0])]
    orig_uint8 = [frames[i] for i in range(len(frames))]

    save_video(orig_uint8, os.path.join(args.out, f"{args.variant}_ep{args.ep}_original.mp4"))
    save_video(gen_uint8, os.path.join(args.out, f"{args.variant}_ep{args.ep}_generated.mp4"))
    # side-by-side
    sbs = []
    for o, g in zip(orig_uint8, gen_uint8):
        sep = np.ones((o.shape[0], 2, 3), np.uint8) * 255
        sbs.append(np.concatenate([o, sep, g], axis=1))
    save_video(sbs, os.path.join(args.out, f"{args.variant}_ep{args.ep}_compare.mp4"))
    print(f"saved videos to {args.out}/  (original | generated | compare)", flush=True)


if __name__ == "__main__":
    main()
