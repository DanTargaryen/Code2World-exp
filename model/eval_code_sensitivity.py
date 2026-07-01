"""Code sensitivity evaluation.

Uses the paired eval set (same seed + same actions across all 7 variants, identical
init frame) so the ONLY variable is the code condition.

Produces:
  1. Confusion matrix M[v][w] = MSE(rollout with variant-v code, variant-w GT)
     + diagonal hit-rate (does v-code rollout best match v's GT?)
  2. Motion magnitude per variant (directional correctness: fast > base > slow?)
  3. Code ablation (zero / shuffled code -> latent MSE degradation)
  4. Side-by-side videos: same seed, all variants generated, for visual inspection
  5. confusion_matrix.png heatmap
"""
import os, sys, json, argparse, subprocess
import numpy as np
import torch
import torch.nn.functional as F

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
def ar_rollout(model, init_lat, actions, code, num_actions, dev, context):
    """init_lat (1,z,h,w), actions (L,), code (1,N,D) -> pred latents (L,z,h,w)."""
    L = len(actions)
    hist = init_lat.unsqueeze(1)
    preds = []
    for t in range(L):
        inp = hist[:, -context:]
        tlen = inp.shape[1]
        a = torch.from_numpy(actions[max(0, t + 1 - context):t + 1].astype(np.int64)).to(dev)
        a_oh = F.one_hot(a, num_actions).float().unsqueeze(0)
        pred, _, _ = model(inp, a_oh, code)
        nxt = pred[:, -1:]
        preds.append(nxt)
        hist = torch.cat([hist, nxt], dim=1)
    return torch.cat(preds, dim=1)[0]  # (L,z,h,w)


def save_video(frames_uint8, path, fps=10, scale=4):
    import imageio_ffmpeg, cv2
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames_uint8[0].shape[:2]
    OH, OW = H * scale, W * scale
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames_uint8:
        p.stdin.write(np.ascontiguousarray(cv2.resize(f, (OW, OH), interpolation=cv2.INTER_NEAREST), np.uint8).tobytes())
    p.stdin.close(); p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world")
    ap.add_argument("--ckpt", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world/ckpt_final.pt")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--n_eps", type=int, default=20, help="eval episodes for confusion matrix")
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--vis_ep", type=int, default=0, help="episode to render videos for")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="code_sens_out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device

    # model + code bank + vae
    ck = torch.load(args.ckpt, map_location=dev)
    ca = ck.get("args", {})
    code_bank = torch.load(os.path.join(args.root, "code_embeds.pt"), map_location="cpu")
    code_dim = next(iter(code_bank.values())).shape[1]
    model = CausalDiT(latent_dim=16, embed_dim=ca.get("embed_dim", 512),
                      num_layers=ca.get("num_layers", 12), num_heads=ca.get("num_heads", 8),
                      num_actions=args.num_actions, spatial_size=8,
                      max_frames=ca.get("window", 32), code_dim=code_dim).to(dev)
    model.load_state_dict(ck["model"]); model.eval()
    vae = WanVAEWrapper(args.vae, device=dev)
    codes = {v: code_bank[v].float().unsqueeze(0).to(dev) for v in VARIANTS}
    nV = len(VARIANTS)
    print(f"loaded model (step {ck.get('step')}), {nV} variants", flush=True)

    # =========================================================
    # 1. Confusion matrix M[v][w] = MSE(rollout(v-code, w-init+actions), w-GT)
    #    Since paired: init frame & actions are shared across variants for a given
    #    seed. We use variant w's init+actions (its true trajectory's inputs) and
    #    swap in code v. Compare against w's GT latents.
    # =========================================================
    print("== confusion matrix ==", flush=True)
    M = np.zeros((nV, nV))
    for ep in range(args.n_eps):
        # load each variant's GT once
        gt = {}
        for w in VARIANTS:
            frames, actions, lat, L = load_ep(args.root, w, ep)
            gt[w] = {"init": lat[0:1].to(dev), "actions": actions,
                     "tgt": lat[1:L + 1].to(dev), "L": L}
        for vi, v in enumerate(VARIANTS):
            for wi, w in enumerate(VARIANTS):
                g = gt[w]
                pred = ar_rollout(model, g["init"], g["actions"], codes[v],
                                  args.num_actions, dev, args.context)  # (L,z,h,w)
                M[vi, wi] += F.mse_loss(pred, g["tgt"]).item()
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{args.n_eps}", flush=True)
    M /= args.n_eps
    diag_hits = sum(int(np.argmin(M[vi]) == vi) for vi in range(nV))
    hit_rate = diag_hits / nV
    print(f"diagonal hit-rate: {diag_hits}/{nV} = {hit_rate:.1%} (random={1/nV:.1%})", flush=True)

    # =========================================================
    # 2. Motion magnitude per variant (directional correctness)
    #    Generate with each variant's OWN code+init+actions; measure mean frame diff.
    # =========================================================
    print("== motion magnitude ==", flush=True)
    motion_gen, motion_gt = {}, {}
    for v in VARIANTS:
        frames, actions, lat, L = load_ep(args.root, v, args.vis_ep)
        pred = ar_rollout(model, lat[0:1].to(dev), actions, codes[v], args.num_actions, dev, args.context)
        gen_lat = torch.cat([lat[0:1].to(dev), pred], 0).unsqueeze(0)
        gen_f = vae.decode(gen_lat)[0].cpu().numpy()      # (L+1,3,H,W)
        gt_f = frames.astype(np.float32) / 255.0          # (L+1,H,W,3)
        gen_f = np.transpose(gen_f, (0, 2, 3, 1))
        motion_gen[v] = float(np.abs(np.diff(gen_f, axis=0)).mean())
        motion_gt[v] = float(np.abs(np.diff(gt_f, axis=0)).mean())

    # =========================================================
    # 3. Code ablation: zero code / shuffled code -> latent MSE
    # =========================================================
    print("== code ablation ==", flush=True)
    abl = {}
    for tag in ["real", "zero", "shuffled"]:
        tot, n = 0.0, 0
        for ep in range(min(args.n_eps, 10)):
            for v in VARIANTS:
                _, actions, lat, L = load_ep(args.root, v, ep)
                c = codes[v]
                if tag == "zero":
                    c = torch.zeros_like(c)
                elif tag == "shuffled":
                    idx = torch.randperm(c.shape[1], device=dev)
                    c = c[:, idx]
                pred = ar_rollout(model, lat[0:1].to(dev), actions, c, args.num_actions, dev, args.context)
                tot += F.mse_loss(pred, lat[1:L+1].to(dev)).item(); n += 1
        abl[tag] = tot / n
    print(f"  ablation latent MSE: {abl}", flush=True)

    # =========================================================
    # 4. Visual: same seed (vis_ep), generate with ALL variant codes,
    #    side-by-side video grid. Each variant uses base's init+actions but its own code.
    # =========================================================
    print("== visual rollout (counterfactual: same init/actions, swap code) ==", flush=True)
    bframes, bactions, blat, bL = load_ep(args.root, "base", args.vis_ep)
    init = blat[0:1].to(dev)
    per_variant_frames = {}
    for v in VARIANTS:
        pred = ar_rollout(model, init, bactions, codes[v], args.num_actions, dev, args.context)
        gen_lat = torch.cat([init, pred], 0).unsqueeze(0)
        gen_f = vae.decode(gen_lat)[0]
        per_variant_frames[v] = [(gen_f[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                                 for i in range(gen_f.shape[0])]
    # build a grid video: 7 variants in a row, labeled
    import cv2
    Lv = min(len(per_variant_frames[v]) for v in VARIANTS)
    grid_frames = []
    for t in range(Lv):
        cols = []
        for v in VARIANTS:
            img = per_variant_frames[v][t]
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_NEAREST)
            cv2.putText(img, v[:6], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
            cols.append(img)
            cols.append(np.ones((96, 2, 3), np.uint8) * 255)
        grid_frames.append(np.concatenate(cols[:-1], axis=1))
    save_video(grid_frames, os.path.join(args.out, "counterfactual_grid.mp4"), fps=10, scale=1)
    print(f"  saved {args.out}/counterfactual_grid.mp4", flush=True)

    # =========================================================
    # 5. Confusion matrix heatmap
    # =========================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(nV)); ax.set_xticklabels(VARIANTS, rotation=45, ha="right")
    ax.set_yticks(range(nV)); ax.set_yticklabels(VARIANTS)
    ax.set_xlabel("ground-truth variant (w)")
    ax.set_ylabel("code used for rollout (v)")
    ax.set_title(f"Confusion: MSE(rollout_v, GT_w)\ndiag hit-rate {hit_rate:.0%} (random {1/nV:.0%})")
    for i in range(nV):
        for j in range(nV):
            ax.text(j, i, f"{M[i,j]:.4f}", ha="center", va="center",
                    color="white" if M[i,j] < M.mean() else "black", fontsize=6)
    plt.colorbar(im); plt.tight_layout()
    plt.savefig(os.path.join(args.out, "confusion_matrix.png"), dpi=130)
    print(f"  saved {args.out}/confusion_matrix.png", flush=True)

    # dump json
    result = {
        "confusion_matrix": M.tolist(),
        "variants": VARIANTS,
        "diag_hit_rate": hit_rate,
        "random_baseline": 1 / nV,
        "motion_generated": motion_gen,
        "motion_gt": motion_gt,
        "ablation_latent_mse": abl,
    }
    json.dump(result, open(os.path.join(args.out, "results.json"), "w"), indent=2)
    print("== motion (gen vs gt) ==")
    for v in VARIANTS:
        print(f"  {v:9s} gen={motion_gen[v]:.4f}  gt={motion_gt[v]:.4f}")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
