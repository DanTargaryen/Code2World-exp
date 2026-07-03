"""Custom-action AR rollout: fix frame-0, feed a HAND-CRAFTED action sequence,
autoregressively generate frames, dump an annotated frame grid (+mp4).

Action ids (CoinRun set_action_xy, act6 subset):
  7 = 右 (vx=+1, vy=0)
  8 = 右上 (vx=+1, vy=+1)  跳+右
  5 = 上   (vx=0,  vy=+1)  原地跳
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.causal_dit import CausalDiT
from models.vae import WanVAEWrapper

ALABEL = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U"}


@torch.no_grad()
def rollout(model, init_latent, actions, code, num_actions, dev, context):
    L = len(actions)
    hist = init_latent.unsqueeze(1)
    preds = []
    for t in range(L):
        inp = hist[:, -context:]
        act_win = actions[max(0, t + 1 - context): t + 1]
        a = torch.from_numpy(np.asarray(act_win, np.int64)).to(dev)
        a_oh = F.one_hot(a, num_actions).float().unsqueeze(0)
        pred, _, _ = model(inp, a_oh, code)
        next_lat = pred[:, -1:]
        preds.append(next_lat)
        hist = torch.cat([hist, next_lat], dim=1)
    return torch.cat(preds, dim=1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6")
    ap.add_argument("--ckpt", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world_act6/ckpt_final.pt")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_eval")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--out", default="../outputs/custom_rollout")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    # hand-crafted action sequence: 右 / 右上 / 上 交替
    ACTIONS = ([7]*6 + [8]*6 + [5]*4 + [7]*4 + [8]*4 + [5]*4 + [7]*4)
    print(f"action seq (len {len(ACTIONS)}): {ACTIONS}", flush=True)

    ck = torch.load(args.ckpt, map_location=dev)
    cargs = ck.get("args", {})
    z, h = 16, 8
    code_bank = torch.load(os.path.join(args.root, "code_embeds.pt"), map_location="cpu")
    code_dim = next(iter(code_bank.values())).shape[1]
    model = CausalDiT(latent_dim=z, embed_dim=cargs.get("embed_dim", 512),
                      num_layers=cargs.get("num_layers", 12), num_heads=cargs.get("num_heads", 8),
                      num_actions=args.num_actions, spatial_size=h,
                      max_frames=cargs.get("window", 32), code_dim=code_dim).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()
    print(f"loaded {args.ckpt} (step {ck.get('step')})", flush=True)

    vae = WanVAEWrapper(args.vae, device=dev)

    lat_all = torch.load(os.path.join(args.root, "latents", f"{args.variant}__{args.split}.pt"),
                         map_location="cpu")
    ep_len = np.load(os.path.join(args.root, args.variant, f"{args.split}.npz"))["episode_lengths"]
    f0 = int(ep_len[:args.ep].sum()) + args.ep
    init_latent = lat_all["latents"][f0:f0+1].float().to(dev)
    code = code_bank[args.variant].float().unsqueeze(0).to(dev)

    pred_lat = rollout(model, init_latent, ACTIONS, code, args.num_actions, dev, args.context)
    gen_lat = torch.cat([init_latent, pred_lat], dim=0)
    gen = vae.decode(gen_lat.unsqueeze(0))[0]                 # (T+1,3,H,W) [0,1]
    imgs = [(gen[i].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8)
            for i in range(gen.shape[0])]

    # annotated grid
    import cv2
    s = args.scale; H = W = 64*s; per_row = 8
    labels = ["init"] + [ALABEL.get(a, str(a)) for a in ACTIONS]
    tiles = []
    for i, im in enumerate(imgs):
        t = cv2.resize(im, (W, H), interpolation=cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20,20,20))
        cv2.putText(t, f"{i}:{labels[i]}", (4,14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0,255,0), 1, cv2.LINE_AA)
        tiles.append(t)
    while len(tiles) % per_row: tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r+per_row], 1) for r in range(0, len(tiles), per_row)]
    grid = np.concatenate(rows, 0)
    grid_path = os.path.join(args.out, f"{args.variant}_customseq_grid.png")
    cv2.imwrite(grid_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"saved grid -> {grid_path}", flush=True)

    # mp4 (best-effort)
    try:
        import imageio_ffmpeg, subprocess
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        OH = OW = 64*6
        p = subprocess.Popen([ff,"-y","-f","rawvideo","-pix_fmt","rgb24","-s",f"{OW}x{OH}",
                              "-r","6","-i","-","-c:v","libx264","-pix_fmt","yuv420p","-crf","18",
                              os.path.join(args.out, f"{args.variant}_customseq.mp4")],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for im in imgs:
            p.stdin.write(np.ascontiguousarray(
                cv2.resize(im,(OW,OH),interpolation=cv2.INTER_NEAREST), np.uint8).tobytes())
        p.stdin.close(); p.wait()
        print("saved mp4", flush=True)
    except Exception as e:
        print(f"mp4 skipped: {e}", flush=True)


if __name__ == "__main__":
    main()
