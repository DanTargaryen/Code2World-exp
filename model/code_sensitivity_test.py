"""Zero-shot code-sensitivity test: does swapping the code condition change the
generated video, holding the initial frame + action stream fixed?

Setup (single-scene, existing base-only ckpt):
  - one eval episode's init latent + GT per-frame actions are FIXED across variants
  - for each config (base + counterfactuals) we Qwen-encode its full spec -> code,
    then block-AR generate with the SAME ckpt / init / actions, only `code` differs
  - decode to RGB; measure per-variant mean-abs pixel diff vs base generation

Caveat: this ckpt only trained on `base`, so counterfactual configs are OUT of
distribution — a lack of change means "never taught this rule", not "mechanism
broken". Use as a baseline before deciding on multi-variant retraining.

    python code_sensitivity_test.py --ckpt <...> --root <dataset> --ep 462 \
      --configs base lowgrav highgrav fast slow highjump --out ../examples/sens_ep462
"""
import os, sys, argparse, subprocess
import numpy as np
import torch
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dataset.dataset import Code2WorldDataset
from dataset.game_config import load_config, render_code_condition
from models.causal_dit import CausalDiT, block_ar_generate
from models.vae import WanVAEWrapper
from action_space import remap_to_compact


def to_u8(t):
    return (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def encode_code(cfg_name, qwen, tok, dev):
    """Qwen-encode a config's full spec -> (N,896) code embedding."""
    path = os.path.join(HERE, "dataset", "configs", f"{cfg_name}.yaml")
    text = render_code_condition(load_config(path))
    ids = tok(text, return_tensors="pt", truncation=True, max_length=5120).to(dev)
    with torch.no_grad():
        return qwen(**ids).last_hidden_state[0].half().float().unsqueeze(0).to(dev)  # (1,N,896)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--qwen", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/Qwen2.5-0.5B")
    ap.add_argument("--train-variant", default="base", help="variant whose latents/actions to use as the fixed clip")
    ap.add_argument("--configs", nargs="+", default=["base", "lowgrav", "highgrav", "fast", "slow"])
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--n_latents", type=int, default=21)
    ap.add_argument("--flow_steps", type=int, default=16)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--scale", type=int, default=6)
    ap.add_argument("--out", default="../examples/code_sensitivity")
    ap.add_argument("--seed", type=int, default=0, help="fixed seed for flow sampling noise (reproducible)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.ckpt, map_location=dev)
    ca = ck.get("args", {})
    compact = bool(ca.get("action_compact", True))
    num_actions = int(ca.get("num_actions", 6))
    block_size = int(ca.get("block_size", 3))

    # fixed clip: init latent + GT actions from one eval episode of the train variant
    lat_all = torch.load(os.path.join(args.root, "latents", f"{args.train_variant}__episodes_eval.pt"),
                         map_location="cpu")
    epl = lat_all["episode_lengths"].numpy(); R = int(lat_all.get("action_repeat", 4))
    l0 = int(epl[:args.ep].sum()) + args.ep
    a0 = int(epl[:args.ep].sum()) * R
    K = min(args.n_latents, int(epl[args.ep]) + 1)
    lat = lat_all["latents"][l0:l0 + K].float().to(dev)
    raw_actions = lat_all["actions"][a0:a0 + (K - 1) * R].numpy()
    acts = remap_to_compact(torch.as_tensor(raw_actions)).numpy() if compact else raw_actions
    init = lat[:1].unsqueeze(0)

    code_dim = 896
    model = CausalDiT(latent_dim=16, embed_dim=ca.get("embed_dim", 768),
                      num_layers=ca.get("num_layers", 24), num_heads=ca.get("num_heads", 16),
                      num_actions=num_actions, spatial_size=8,
                      max_frames=ca.get("window", 20) + 1, code_dim=code_dim,
                      block_size=block_size, action_mode=ca.get("action_mode", "crossattn"),
                      action_window=ca.get("action_window", 3)).to(dev)
    model.load_state_dict(ck["model"], strict=False); model.eval()

    from transformers import AutoModel, AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    tok = AutoTokenizer.from_pretrained(args.qwen)
    qwen = AutoModel.from_pretrained(args.qwen, torch_dtype=torch.float32).to(dev).eval()
    vae = WanVAEWrapper(args.vae, device=dev)

    # GT frames (for the top reference row)
    gt_fr = vae.decode_video(lat[:K]); nfr = gt_fr.shape[0]

    gens, base_frames = {}, None
    print(f"ep={args.ep} K={K} | ckpt trained on '{args.train_variant}' only (counterfactuals are OOD)", flush=True)
    for name in args.configs:
        torch.manual_seed(args.seed)   # SAME sampling noise per variant -> diff is purely from code
        code = encode_code(name, qwen, tok, dev)
        gen = block_ar_generate(model, init, acts, code, num_actions, dev, block_size, args.flow_steps)[0]
        fr = vae.decode_video(gen[:K])
        gens[name] = fr
        if name == args.configs[0]:
            base_frames = fr
        # quantify difference vs the first (base) generation
        diff = (fr.float() - base_frames.float()).abs().mean().item() if base_frames is not None else 0.0
        print(f"  {name:10s}: mean|Δ| vs {args.configs[0]} = {diff:.5f}", flush=True)

    # video: rows = GT + each variant, per frame side by side
    def label_row(frames, txt):
        out = []
        for i in range(nfr):
            im = cv2.resize(to_u8(frames[i]), (64 * args.scale, 64 * args.scale), interpolation=cv2.INTER_NEAREST)
            cv2.putText(im, txt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            out.append(im)
        return out
    # ---- 1) base alone (single video) ----
    base_name = args.configs[0]
    H = 64 * args.scale
    ff = __import__("imageio_ffmpeg").get_ffmpeg_exe()

    def write_video(path, per_frame_imgs, w, h):
        p = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
                              "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", path],
                             stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for im in per_frame_imgs:
            p.stdin.write(np.ascontiguousarray(im, np.uint8).tobytes())
        p.stdin.close(); p.wait()

    base_row = label_row(gens[base_name], base_name)
    write_video(os.path.join(args.out, "base.mp4"), base_row, H, H)
    print(f"saved {args.out}/base.mp4 ({base_name} alone)", flush=True)

    # ---- 2) 3x2 grid comparison: GT + up to 5 variants (6 tiles) ----
    tiles = ["GT"] + list(args.configs)          # GT then base, lowgrav, ...
    tiles = tiles[:6]                            # 3x2 holds 6
    tile_rows = {"GT": label_row(gt_fr, "GT")}
    for n in args.configs:
        tile_rows[n] = label_row(gens[n], n)
    ncol, nrow = 3, 2
    grid_w, grid_h = H * ncol, H * nrow
    blank = np.zeros((H, H, 3), np.uint8)

    def grid_frame(i):
        cells = []
        for t in tiles:
            cells.append(tile_rows[t][i])
        while len(cells) < ncol * nrow:
            cells.append(blank)
        rows = [np.concatenate(cells[r * ncol:(r + 1) * ncol], axis=1) for r in range(nrow)]
        return np.concatenate(rows, axis=0)

    write_video(os.path.join(args.out, "compare_3x2.mp4"), (grid_frame(i) for i in range(nfr)),
                grid_w, grid_h)
    print(f"saved {args.out}/compare_3x2.mp4 (3x2: {', '.join(tiles)})", flush=True)


if __name__ == "__main__":
    main()
