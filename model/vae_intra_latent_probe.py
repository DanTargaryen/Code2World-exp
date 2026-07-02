"""VAE go/no-go probe for the per-frame-action plan.

Question: Wan 2.1 VAE compresses 4 frames -> 1 latent. If we want per-frame actions
to be learnable, a single latent MUST retain the INTRA-LATENT motion (frame-to-frame
change within its 4 frames). This probes whether encode_video->decode_video preserves
that motion, by comparing intra-latent frame differences before vs after roundtrip.

We pick, from real eval episodes, the 4-frame chunks with the LARGEST internal motion
(these are the hardest / most informative), roundtrip them, and measure how much of the
within-chunk motion survives.
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vae import WanVAEWrapper


def intra_motion(frames):
    # frames: (T,3,H,W) float [0,1] -> mean abs frame-to-frame diff, per adjacent pair
    d = (frames[1:] - frames[:-1]).abs().mean(dim=(1, 2, 3))   # (T-1,)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc/base/episodes_eval.npz")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--n_chunks", type=int, default=16, help="top-motion 4-frame chunks to test")
    ap.add_argument("--out", default="outputs/vae_probe")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    d = np.load(args.npz, allow_pickle=True)
    frames_all = torch.from_numpy(d["frames"]).float().permute(0, 3, 1, 2) / 255.0  # (N,3,H,W)
    ep_len = d["episode_lengths"]         # K actions per episode; frames per ep = 4K+1
    ar = int(d["action_repeat"])
    print(f"frames {tuple(frames_all.shape)} | action_repeat={ar} | episodes {len(ep_len)}", flush=True)

    # enumerate NON-overlapping 4-frame chunks aligned to latent boundaries WITHIN episodes.
    # ep layout: frame0=init, then chunks [1..4],[5..8],... (each -> 1 latent). We test the
    # 5-frame window [c0, c0+4] so encode_video(5 frames)->2 latents, decode->5 frames; the
    # 2nd latent is the chunk of interest and its intra-motion = frames diffs in [c0+1..c0+4].
    chunks = []          # (motion, frame_start) ; frame_start = c0 (the pre-chunk anchor)
    off = 0
    for K in ep_len:
        K = int(K)
        base0 = off
        # latent l (l>=1) covers frames [base0 + 4(l-1)+1 .. base0 + 4l]
        for l in range(1, K + 1):
            c0 = base0 + 4 * (l - 1)              # anchor frame (last of previous latent)
            seg = frames_all[c0:c0 + 5]           # 5 frames -> 2 latents
            if seg.shape[0] < 5:
                break
            m = intra_motion(seg[1:]).mean().item()   # motion inside the target chunk
            chunks.append((m, c0))
        off += 4 * K + 1
    chunks.sort(reverse=True)
    top = chunks[:args.n_chunks]
    print(f"total chunks {len(chunks)} | testing top-{len(top)} by intra-motion "
          f"(motion range {top[-1][0]:.4f}..{top[0][0]:.4f})", flush=True)

    vae = WanVAEWrapper(args.vae, device=dev)

    rows = []
    stats = {"pre": [], "post": [], "recon_mse": []}
    for rank, (m, c0) in enumerate(top):
        seg = frames_all[c0:c0 + 5].to(dev)            # (5,3,H,W)  frame0=anchor, 1..4=chunk
        lat = vae.encode_video(seg)                    # (2, z, h, w)
        rec = vae.decode_video(lat)                    # (5,3,H,W)
        pre = intra_motion(seg[1:].cpu())              # (3,) diffs within chunk frames 1..4
        post = intra_motion(rec[1:].cpu())
        recon = (seg - rec).abs().mean().item()
        stats["pre"].append(pre.mean().item())
        stats["post"].append(post.mean().item())
        stats["recon_mse"].append(recon)
        rows.append((rank, c0, pre.mean().item(), post.mean().item(),
                     post.mean().item() / max(pre.mean().item(), 1e-6), recon))

    print("\nrank  c0     intra_pre  intra_post  ratio(post/pre)  recon_L1", flush=True)
    for r in rows:
        print(f"{r[0]:4d}  {r[1]:5d}  {r[2]:9.4f}  {r[3]:10.4f}  {r[4]:14.3f}  {r[5]:.4f}", flush=True)

    pre = np.array(stats["pre"]); post = np.array(stats["post"])
    print(f"\n=== SUMMARY (top-{len(top)} highest-motion chunks) ===", flush=True)
    print(f"intra-latent motion  pre  mean={pre.mean():.4f}", flush=True)
    print(f"intra-latent motion  post mean={post.mean():.4f}", flush=True)
    print(f"motion RETENTION ratio (post/pre) mean={ (post/np.maximum(pre,1e-6)).mean():.3f} "
          f"median={np.median(post/np.maximum(pre,1e-6)):.3f}", flush=True)
    print(f"reconstruction L1 mean={np.mean(stats['recon_mse']):.4f}", flush=True)
    print("\nInterpretation: ratio ~1.0 => VAE keeps intra-latent motion (per-frame action "
          "has a learnable target). ratio << 1 => motion smeared out (per-frame action moot).",
          flush=True)

    # dump a visual: for the top-3 chunks, GT 5 frames vs recon 5 frames
    try:
        import cv2
        s = 4
        vis_rows = []
        for (m, c0) in top[:3]:
            seg = frames_all[c0:c0 + 5].to(dev)
            rec = vae.decode_video(vae.encode_video(seg)).cpu()
            def strip(fr):
                imgs = [(fr[i].permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8) for i in range(fr.shape[0])]
                imgs = [cv2.resize(im,(64*s,64*s),interpolation=cv2.INTER_NEAREST) for im in imgs]
                return np.concatenate(imgs, axis=1)
            gt = strip(seg.cpu()); rc = strip(rec)
            sep = np.ones((4, gt.shape[1], 3), np.uint8)*255
            vis_rows.append(np.concatenate([gt, sep, rc], axis=0))
        big = np.concatenate([np.concatenate([r, np.ones((8,r.shape[1],3),np.uint8)*128], 0) for r in vis_rows], 0)
        p = os.path.join(args.out, "intra_latent_roundtrip.png")
        cv2.imwrite(p, cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        print(f"saved visual (top-3: GT strip over recon strip, 5 frames each) -> {p}", flush=True)
    except Exception as e:
        print(f"visual skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
