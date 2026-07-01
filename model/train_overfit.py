"""Overfit sanity check: take ONE short clip, train the DiT to predict next-frame
latents until loss -> ~0, then decode the rollout to confirm the pipeline works.

Conditions on the REAL frozen Qwen code embed for the clip's variant (loaded from
the precomputed code_embeds.pt). The DiT's code_proj maps 896-d Qwen features into
the model width; the code itself is frozen (not optimized).
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vae import WanVAEWrapper
from models.causal_dit import CausalDiT


def load_clip(npz_path, clip_len=16, episode_idx=0):
    d = np.load(npz_path)
    frames = d["frames"]; actions = d["actions"]
    rewards = d["rewards"]; dones = d["dones"]; ep_len = d["episode_lengths"]
    # locate episode boundaries (frames has +1 per episode, actions does not)
    f_start = int(ep_len[:episode_idx].sum()) + episode_idx  # +1 init frame each prior ep
    a_start = int(ep_len[:episode_idx].sum())
    L = int(ep_len[episode_idx])
    clip_len = min(clip_len, L)
    f = frames[f_start:f_start + clip_len + 1]      # (clip+1, H, W, 3)
    a = actions[a_start:a_start + clip_len]          # (clip,)
    r = rewards[a_start:a_start + clip_len]
    dn = dones[a_start:a_start + clip_len]
    return f, a, r, dn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--code_embeds", default=None,
                    help="path to code_embeds.pt (default: <root>/code_embeds.pt "
                         "inferred from --data)")
    ap.add_argument("--variant", default=None,
                    help="variant key into code_embeds.pt (default: parent dir name of --data)")
    ap.add_argument("--code_max_tok", type=int, default=0,
                    help="truncate code to first N tokens (0 = keep all)")
    ap.add_argument("--clip", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="overfit_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    # --- data ---
    f, a, r, dn = load_clip(args.data, args.clip)
    T = len(a)
    print(f"Clip: {len(f)} frames, {T} actions, frame {f.shape[1]}x{f.shape[2]}")
    frames = torch.from_numpy(f).permute(0, 3, 1, 2).float() / 255.0  # (T+1,3,H,W)
    frames = frames.unsqueeze(0).to(dev)                              # (1,T+1,3,H,W)
    actions = torch.from_numpy(a.astype(np.int64)).to(dev)            # (T,)
    rewards = torch.from_numpy(r).to(dev)
    dones_t = torch.from_numpy(dn.astype(np.int64)).to(dev)

    # --- VAE: encode all frames once (frozen) ---
    vae = WanVAEWrapper(args.vae, device=dev)
    with torch.no_grad():
        all_lat = vae.encode(frames)            # (1, T+1, z, h, w)
    z, h, w = all_lat.shape[2:]
    print(f"Latent: z={z}, spatial={h}x{w}")
    inp_lat = all_lat[:, :T]                    # frames 0..T-1  (input)
    tgt_lat = all_lat[:, 1:T+1]                 # frames 1..T    (target = next frame)

    num_actions = 15
    act_onehot = F.one_hot(actions, num_actions).float().unsqueeze(0)  # (1,T,15)
    reward_cls = (torch.sign(rewards) + 1).long().unsqueeze(0)         # (1,T) in {0,1,2}
    done_cls = dones_t.unsqueeze(0)                                    # (1,T)

    # --- real frozen Qwen code embed for this clip's variant ---
    variant = args.variant or os.path.basename(os.path.dirname(os.path.abspath(args.data)))
    code_path = args.code_embeds or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.data))), "code_embeds.pt")
    code_bank = torch.load(code_path, map_location="cpu")
    if variant not in code_bank:
        raise KeyError(f"variant '{variant}' not in {code_path}; "
                       f"available: {list(code_bank.keys())}")
    code = code_bank[variant].float()                     # (N_tok, 896)
    if args.code_max_tok and code.shape[0] > args.code_max_tok:
        code = code[:args.code_max_tok]
    code = code.unsqueeze(0).to(dev)                       # (1, N, 896) frozen
    code_dim = code.shape[-1]
    print(f"Code: variant='{variant}', {tuple(code.shape)} (frozen)")

    # --- model ---
    model = CausalDiT(latent_dim=z, embed_dim=512, num_layers=12, num_heads=8,
                      num_actions=num_actions, spatial_size=h, max_frames=T,
                      code_dim=code_dim).to(dev)

    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params")

    t0 = time.time()
    for step in range(args.steps):
        model.train()
        pred, rew_logits, done_logits = model(inp_lat, act_onehot, code)
        loss_lat = F.mse_loss(pred, tgt_lat)
        loss_rew = F.cross_entropy(rew_logits.reshape(-1, 3), reward_cls.reshape(-1))
        loss_done = F.cross_entropy(done_logits.reshape(-1, 2), done_cls.reshape(-1))
        loss = loss_lat + 0.1 * loss_rew + 0.1 * loss_done

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % 100 == 0 or step == args.steps - 1:
            dt = time.time() - t0
            print(f"  step {step:5d} | loss {loss.item():.5f} "
                  f"(lat {loss_lat.item():.5f}, rew {loss_rew.item():.3f}, "
                  f"done {loss_done.item():.3f}) | {dt:.0f}s", flush=True)

    # --- evaluate: decode prediction vs ground truth ---
    model.eval()
    with torch.no_grad():
        pred, _, _ = model(inp_lat, act_onehot, code)
        pred_frames = vae.decode(pred)          # (1,T,3,H,W) predicted next frames
        gt_frames = vae.decode(tgt_lat)         # (1,T,3,H,W) GT next frames (VAE roundtrip)
        mse = F.mse_loss(pred_frames, gt_frames).item()
        psnr = -10 * np.log10(mse + 1e-8)
        print(f"\nFinal latent MSE: {loss_lat.item():.6f}")
        print(f"Decoded pred-vs-GT pixel MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")

    # save a comparison strip: [input | GT-next | pred-next] for a few frames
    from PIL import Image
    def to_img(t):  # (3,H,W) -> uint8 HWC
        return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    rows = []
    for ti in range(min(T, 8)):
        inp_img = to_img(vae.decode(inp_lat[:, ti:ti+1])[0, 0])
        gt_img = to_img(gt_frames[0, ti])
        pr_img = to_img(pred_frames[0, ti])
        sep = np.ones((inp_img.shape[0], 2, 3), dtype=np.uint8) * 255
        rows.append(np.concatenate([inp_img, sep, gt_img, sep, pr_img], axis=1))
    grid = np.concatenate(rows, axis=0)
    grid = np.kron(grid, np.ones((3, 3, 1))).astype(np.uint8)  # 3x upscale
    Image.fromarray(grid).save(os.path.join(args.out, "overfit_compare.png"))
    print(f"Saved {args.out}/overfit_compare.png  (cols: input | GT-next | pred-next)")


if __name__ == "__main__":
    main()
