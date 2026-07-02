"""Bidirectional flow rollout: fix the INIT latent, feed a hand-crafted action
sequence, denoise the WHOLE latent sequence jointly, decode with the TEMPORAL VAE
(1 latent -> 4 frames), dump an mp4 + annotated grid.

Length is fixed by the sequence (init + n_actions latents), best kept at the trained
window. Target: ~10s @16fps -> 41 actions -> 42 latents -> 165 frames.
Action ids (CoinRun set_action_xy, act6 subset):
  7=右  8=右上(跳+右)  5=上(原地跳)  1=左  2=左上  4=停
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.bidir_dit import BidirDiT, full_seq_generate
from models.vae import WanVAEWrapper
from action_space import remap_to_compact, NUM_ACTIONS_COMPACT

ALABEL = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc")
    ap.add_argument("--ckpt", default="runs/c2w_fm/ckpt_final.pt")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_eval")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--flow_steps", type=int, default=16)
    ap.add_argument("--n_actions", type=int, default=41, help="latent steps to generate (->10s @16fps)")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--out", default="outputs/custom_rollout_fm")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    # hand-crafted action sequence (length n_actions), in RAW Procgen ids: 右/右上/上
    base = ([7] * 6 + [8] * 6 + [5] * 4 + [7] * 4 + [8] * 4 + [5] * 4 + [7] * 4)
    ACTIONS_RAW = (base * ((args.n_actions // len(base)) + 1))[:args.n_actions]
    ACTIONS = remap_to_compact(torch.as_tensor(ACTIONS_RAW, dtype=torch.long)).tolist()
    print(f"action seq (len {len(ACTIONS)})", flush=True)

    ck = torch.load(args.ckpt, map_location=dev)
    cargs = ck.get("args", {})
    if len(ACTIONS) != cargs.get("window", 41):
        print(f"  [note] n_actions={len(ACTIONS)} != train window={cargs.get('window', 41)}; "
              f"bidir is fixed-length, results best at the trained length", flush=True)
    z, h = 16, 8
    code_bank = torch.load(os.path.join(args.root, "code_embeds.pt"), map_location="cpu")
    code_dim = next(iter(code_bank.values())).shape[1]
    model = BidirDiT(latent_dim=z, embed_dim=cargs.get("embed_dim", 512),
                     num_layers=cargs.get("num_layers", 12), num_heads=cargs.get("num_heads", 8),
                     num_actions=NUM_ACTIONS_COMPACT, spatial_size=h,
                     max_frames=cargs.get("window", 41) + 1, code_dim=code_dim,
                     action_window=cargs.get("action_window", 3)).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')})", flush=True)

    vae = WanVAEWrapper(args.vae, device=dev)

    lat_all = torch.load(os.path.join(args.root, "latents", f"{args.variant}__{args.split}.pt"),
                         map_location="cpu")
    # init latent = first latent of episode `ep`
    ep_len = lat_all["episode_lengths"].numpy()                  # K (latent steps) per ep
    l0 = int(ep_len[:args.ep].sum()) + args.ep                   # latents per ep = K+1
    init_latent = lat_all["latents"][l0:l0 + 1].float().unsqueeze(0).to(dev)   # (1,1,z,h,w)
    code = code_bank[args.variant].float().unsqueeze(0).to(dev)

    gen = full_seq_generate(model, init_latent, ACTIONS, code, NUM_ACTIONS_COMPACT,
                            dev, args.flow_steps)[0]
    frames = vae.decode_video(gen)                              # (4*(L-1)+1, 3, H, W)
    imgs = [(frames[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
            for i in range(frames.shape[0])]
    print(f"generated {gen.shape[0]} latents -> {len(imgs)} frames "
          f"({len(imgs)/args.fps:.1f}s @ {args.fps}fps)", flush=True)

    import cv2
    # annotated grid (one tile per latent step, sampled every 4 frames + init)
    s = 4; H = W = 64 * s; per_row = 8
    labels = ["init"] + [ALABEL.get(a, str(a)) for a in ACTIONS_RAW]
    tiles = []
    for li in range(gen.shape[0]):
        fi = 0 if li == 0 else 4 * (li - 1) + 1                 # representative frame of latent li
        im = imgs[min(fi, len(imgs) - 1)]
        t = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(t, f"{li}:{labels[li]}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + per_row], 1) for r in range(0, len(tiles), per_row)]
    grid_path = os.path.join(args.out, f"{args.variant}_grid.png")
    cv2.imwrite(grid_path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"saved grid -> {grid_path}", flush=True)

    # full-fps mp4 (every decoded frame)
    try:
        import imageio_ffmpeg, subprocess
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        OH = OW = 64 * 6
        p = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                              "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                              "yuv420p", "-crf", "18",
                              os.path.join(args.out, f"{args.variant}.mp4")],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for im in imgs:
            p.stdin.write(np.ascontiguousarray(
                cv2.resize(im, (OW, OH), interpolation=cv2.INTER_NEAREST), np.uint8).tobytes())
        p.stdin.close(); p.wait()
        print(f"saved mp4 ({args.fps}fps)", flush=True)
    except Exception as e:
        print(f"mp4 skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
