"""Generate one rollout sample from a trained ckpt: block-AR GT-vs-gen comparison.

Reuses the exact, training-validated path (dump_sample / block_ar_generate with
per-frame + compact actions), so it stays in sync with how sample_*.png were made
during training. Loads one eval clip, generates from its init latent + GT actions,
decodes both to RGB, writes a stacked GT|gen image.

    python rollout_sample.py \
      --ckpt <...>/ckpt_final.pt --root <dataset> --out ../examples/rollout_final.png
"""
import os, sys, argparse, subprocess
import numpy as np
import torch
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset.dataset import Code2WorldDataset, collate
from models.causal_dit import CausalDiT, block_ar_generate
from models.vae import WanVAEWrapper
from action_space import remap_to_compact

ASHORT = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U", 0: "-"}


def to_u8(t):
    return (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_video(frames, path, fps, scale):
    """frames: list of (H,W,3) uint8 -> mp4 via ffmpeg (nearest-neighbor upscaled)."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames[0].shape[:2]
    OH, OW = H * scale, W * scale
    p = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "16", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames:
        img = cv2.resize(f, (OW, OH), interpolation=cv2.INTER_NEAREST)
        p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
    p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--ep", type=int, default=0, help="which eval episode to roll out")
    ap.add_argument("--n_latents", type=int, default=21, help="how many latents to roll out & show")
    ap.add_argument("--flow_steps", type=int, default=16)
    ap.add_argument("--out", default="../examples/rollout_final",
                    help="output dir; writes video.mp4 + grid.png")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=6, help="video upscale factor")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=dev)
    ca = ck.get("args", {})
    compact = bool(ca.get("action_compact", True))
    num_actions = int(ca.get("num_actions", 6))
    block_size = int(ca.get("block_size", 3))

    eval_ds = Code2WorldDataset(args.root, split="eval", window=args.window,
                                variants=[args.variant])
    code_dim = next(iter(eval_ds.code_embeds.values())).shape[1]
    z, h = 16, 8
    model = CausalDiT(latent_dim=z, embed_dim=ca.get("embed_dim", 768),
                      num_layers=ca.get("num_layers", 24), num_heads=ca.get("num_heads", 16),
                      num_actions=num_actions, spatial_size=h,
                      max_frames=ca.get("window", args.window) + 1, code_dim=code_dim,
                      block_size=block_size, action_mode=ca.get("action_mode", "crossattn"),
                      action_window=ca.get("action_window", 3)).to(dev)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')}) | compact={compact} num_actions={num_actions}", flush=True)

    vae = WanVAEWrapper(args.vae, device=dev)

    # locate a SPECIFIC eval episode (args.ep) in the precomputed latents (not the
    # dataset's random-window sampler), so --ep picks a deterministic clip.
    lat_all = torch.load(os.path.join(args.root, "latents", f"{args.variant}__episodes_eval.pt"),
                         map_location="cpu")
    epl = lat_all["episode_lengths"].numpy()
    R = int(lat_all.get("action_repeat", 4))
    l0 = int(epl[:args.ep].sum()) + args.ep                    # latent start (each ep has K+1)
    a0 = int(epl[:args.ep].sum()) * R                          # action start (R*K per ep)
    K_ep = int(epl[args.ep])
    K = min(args.n_latents, K_ep + 1)                          # latents to roll out (incl init)
    lat = lat_all["latents"][l0: l0 + K].float().to(dev)      # (K, z, h, w)
    raw_actions = lat_all["actions"][a0: a0 + (K - 1) * R].numpy()   # (R*(K-1),) raw ids
    acts = remap_to_compact(torch.as_tensor(raw_actions)).numpy() if compact else raw_actions
    code = eval_ds.code_embeds[args.variant].float().unsqueeze(0).to(dev)   # (1, N, Dc)
    init = lat[:1].unsqueeze(0)                                # (1,1,z,h,w)
    gen = block_ar_generate(model, init, acts, code, num_actions, dev,
                            block_size, args.flow_steps)[0]    # (K, z, h, w)

    gt_fr = vae.decode_video(lat[:K])                          # (4*(K-1)+1, 3, H, W)
    gen_fr = vae.decode_video(gen[:K])
    nfr = gt_fr.shape[0]

    # --- video: per-frame [GT | gen] side-by-side ---
    sbs = []
    for i in range(nfr):
        g = to_u8(gt_fr[i]); p = to_u8(gen_fr[i])
        sep = np.ones((g.shape[0], 2, 3), np.uint8) * 255
        sbs.append(np.concatenate([g, sep, p], axis=1))        # left GT | right gen
    mp4 = os.path.join(args.out, "video.mp4")
    save_video(sbs, mp4, args.fps, args.scale)
    print(f"saved rollout video ({nfr} frames, {nfr/args.fps:.1f}s @ {args.fps}fps, "
          f"left=GT right=gen) -> {mp4}", flush=True)

    # --- grid: one tile per latent (representative frame), top=GT / bottom=gen ---
    s = 3; W = 64 * s; per_row = 7; tiles = []
    for li in range(K):
        fi = min(0 if li == 0 else 4 * (li - 1) + 1, nfr - 1)
        g = cv2.resize(to_u8(gt_fr[fi]), (W, W), interpolation=cv2.INTER_NEAREST)
        p = cv2.resize(to_u8(gen_fr[fi]), (W, W), interpolation=cv2.INTER_NEAREST)
        sep = np.ones((3, W, 3), np.uint8) * 255
        tile = np.concatenate([g, sep, p], 0)                 # top GT | bottom gen
        tile = cv2.copyMakeBorder(tile, 16, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        lbl = "init" if li == 0 else f"{li}:{ASHORT.get(int(raw_actions[4*(li-1)]), '')}"
        cv2.putText(tile, lbl, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + per_row], 1) for r in range(0, len(tiles), per_row)]
    grid = os.path.join(args.out, "grid.png")
    cv2.imwrite(grid, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"saved rollout grid ({K} latents, per tile: top=GT / bottom=gen) -> {grid}", flush=True)


if __name__ == "__main__":
    main()
