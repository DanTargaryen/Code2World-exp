"""Pixel-space variant of train_fm.py — identical block-AR flow objective, but the
"latent" is a 192-d 8x8 raw-pixel patch (PixelCodec) instead of a Wan VAE latent.

Only three things differ from train_fm.py:
  - data: PixelDataset (reads raw frames, patchify+normalize on the fly)
  - model dims: latent_dim=192, spatial_size=8 (64 tokens/frame, same as VAE geometry)
  - sample dump: decode via PixelCodec.unpatchify (1 latent == 1 frame, no temporal VAE)

The flow loss / per-block tau / forward_state logic is imported unchanged from
train_fm.  Every frame is modeled (no temporal compression): window=41 -> 42
latents = 2.6s @16fps, a quick sharpness check vs the VAE-latent run.

Example:
    python -u train_fm_pixel.py --window 41 --block_size 3 --batch_size 8 \
        --steps 20000 --out /mnt/.../checkpoints/code2world_act6_tc_pixel
"""
import os, sys, time, math, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.pixel_dataset import PixelDataset, collate
from models.causal_dit import CausalDiT, block_ar_generate
from models.pixel_codec import PixelCodec
from train_fm import compute_loss, evaluate, prep_batch  # reused, batch-dict agnostic


def build_loaders(args):
    tr = PixelDataset(args.root, "train", args.window, args.variants, stride=args.stride)
    ev = PixelDataset(args.root, "eval", args.window, args.variants, stride=args.stride)
    print(f"train windows: {len(tr)} | eval windows: {len(ev)}", flush=True)
    code_dim = next(iter(tr.code_embeds.values())).shape[1]
    train_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_dl = DataLoader(ev, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate,
                         drop_last=False, pin_memory=True)
    return tr, ev, train_dl, eval_dl, code_dim


@torch.no_grad()
def dump_sample(model, eval_ds, codec, num_actions, dev, block_size, flow_steps, out_png,
                n_latents=8):
    """Block-AR generate from one eval clip's init + GT actions; compare to GT frames.
    Pixel path: 1 latent == 1 frame (PixelCodec.decode_video)."""
    from PIL import Image
    b = collate([eval_ds[0]])
    lat = b["latents"][0].to(dev)                          # (L,192,8,8)
    actions = b["actions"][0].cpu().numpy()
    code = b["code"][:1].to(dev)
    model.eval()
    init = lat[:1].unsqueeze(0)
    gen = block_ar_generate(model, init, actions, code, num_actions, dev,
                            block_size, flow_steps)[0]     # (L,192,8,8)
    k = min(n_latents, gen.shape[0])
    gen_frames = codec.decode_video(gen[:k])               # (k,3,H,W)
    gt_frames = codec.decode_video(lat[:k])
    model.train()

    def to_img(t):
        return (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    rows = []
    for ti in range(gen_frames.shape[0]):
        g = to_img(gt_frames[ti]); p = to_img(gen_frames[ti])
        sep = np.ones((g.shape[0], 2, 3), np.uint8) * 255
        rows.append(np.concatenate([g, sep, p], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc")
    ap.add_argument("--out", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world_act6_tc_pixel")
    ap.add_argument("--window", type=int, default=41, help="steps-1 per window; latents = window+1")
    ap.add_argument("--block_size", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1, help="frame subsample; =action_repeat(4) -> 4fps 1frame/action")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=["base"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    train_ds, eval_ds, train_dl, eval_dl, code_dim = build_loaders(args)
    s0 = train_ds[0]
    L0, z, h, w = s0["latents"].shape
    print(f"pixel latents/window={L0} (window={args.window}+1) z={z} spatial={h}x{w} "
          f"| code_dim={code_dim} | block_size={args.block_size}", flush=True)
    assert z == 192 and h == 8, f"expected 192-d 8x8 pixel patch, got z={z} spatial={h}"

    model = CausalDiT(latent_dim=z, embed_dim=args.embed_dim, num_layers=args.num_layers,
                      num_heads=args.num_heads, num_actions=args.num_actions,
                      spatial_size=h, max_frames=L0, code_dim=code_dim,
                      block_size=args.block_size).to(dev)
    print(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        print(f"resumed from {args.resume} @ step {start_step}", flush=True)

    codec = None  # lazy PixelCodec for sample dumps

    def infinite(dl):
        while True:
            for b in dl:
                yield b
    it = infinite(train_dl)

    model.train()
    t0 = time.time()
    run = {"loss": 0.0, "fm": 0.0, "rew": 0.0, "done": 0.0, "n": 0}
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = next(it)
        loss, l_fm, l_rew, l_done = compute_loss(model, b, args.num_actions, dev, amp_dtype, args.block_size)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        run["loss"] += loss.item(); run["fm"] += l_fm.item()
        run["rew"] += l_rew.item(); run["done"] += l_done.item(); run["n"] += 1

        if step % args.log_every == 0 or step == args.steps - 1:
            n = max(run["n"], 1); dt = time.time() - t0
            ips = (step - start_step + 1) / max(dt, 1e-6)
            print(f"step {step:6d} | lr {lr_at(step):.2e} | loss {run['loss']/n:.5f} "
                  f"(fm {run['fm']/n:.5f} rew {run['rew']/n:.3f} done {run['done']/n:.3f}) "
                  f"| {ips:.2f} it/s", flush=True)
            run = {"loss": 0.0, "fm": 0.0, "rew": 0.0, "done": 0.0, "n": 0}

        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, eval_dl, args.num_actions, dev, amp_dtype, args.block_size)
            print(f"  [eval] loss {ev['loss']:.5f} (fm {ev['fm']:.5f})", flush=True)

        if step > 0 and step % args.sample_every == 0:
            if codec is None:
                codec = PixelCodec(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=dev)
            png = os.path.join(args.out, f"sample_{step:06d}.png")
            try:
                dump_sample(model, eval_ds, codec, args.num_actions, dev,
                            args.block_size, args.flow_steps, png)
                print(f"  [sample] saved {png} (cols: GT | block-AR gen, pixel)", flush=True)
            except Exception as e:
                print(f"  [sample] skipped: {e}", flush=True)

        if step > 0 and step % args.save_every == 0:
            ckpt = os.path.join(args.out, f"ckpt_{step:06d}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)}, ckpt)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)}, os.path.join(args.out, "ckpt_last.pt"))
            print(f"  [ckpt] {ckpt}", flush=True)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": args.steps, "args": vars(args)},
               os.path.join(args.out, "ckpt_final.pt"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
