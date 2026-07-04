"""Generate one rollout sample from a trained ckpt: block-AR GT-vs-gen comparison.

Reuses the exact, training-validated path (dump_sample / block_ar_generate with
per-frame + compact actions), so it stays in sync with how sample_*.png were made
during training. Loads one eval clip, generates from its init latent + GT actions,
decodes both to RGB, writes a stacked GT|gen image.

    python rollout_sample.py \
      --ckpt <...>/ckpt_final.pt --root <dataset> --out ../examples/rollout_final.png
"""
import os, sys, argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--n_latents", type=int, default=21, help="how many latents to roll out & show")
    ap.add_argument("--flow_steps", type=int, default=16)
    ap.add_argument("--out", default="../examples/rollout_final.png")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

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

    # one eval clip: init latent + GT per-frame actions
    b = collate([eval_ds[0]])
    lat = b["latents"][0].to(dev)                              # (L, z, h, w)
    raw_actions = b["actions"][0].cpu().numpy()                # (R*(L-1),) raw ids
    acts = remap_to_compact(torch.as_tensor(raw_actions)).numpy() if compact else raw_actions
    code = b["code"][:1].to(dev)
    init = lat[:1].unsqueeze(0)                                # (1,1,z,h,w)
    K = min(args.n_latents, lat.shape[0])
    gen = block_ar_generate(model, init, acts[: (K - 1) * 4], code, num_actions, dev,
                            block_size, args.flow_steps)[0]    # (K, z, h, w)

    gt_fr = vae.decode_video(lat[:K])                          # (4*(K-1)+1, 3, H, W)
    gen_fr = vae.decode_video(gen[:K])
    nfr = gt_fr.shape[0]

    # grid: one tile per latent (representative frame), top=GT / bottom=gen, action label
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
    cv2.imwrite(args.out, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
    print(f"saved rollout grid ({K} latents, per tile: top=GT / bottom=gen) -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
