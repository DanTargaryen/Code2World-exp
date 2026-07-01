"""Train the code-conditioned Causal DiT world model on the full Code2World dataset.

Consumes precomputed VAE latents + frozen Qwen code embeds (via Code2WorldDataset),
teacher-forced next-frame latent prediction with auxiliary reward/done heads.

Cross-attention honors the per-sample code padding mask (variants have different
token counts), so batching across variants is correct.

Single-GPU. Example:
    python train.py \
        --root /mnt/pfs/data/huangzehuan/datasets/code2world \
        --vae  /mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers \
        --window 16 --batch_size 8 --steps 20000 --out runs/c2w_v0
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.dataset import Code2WorldDataset, collate
from models.causal_dit import CausalDiT


def build_loaders(args):
    train_ds = Code2WorldDataset(args.root, split="train", window=args.window)
    eval_ds = Code2WorldDataset(args.root, split="eval", window=args.window)
    print(f"train windows: {len(train_ds)} | eval windows: {len(eval_ds)}", flush=True)
    code_dim = next(iter(train_ds.code_embeds.values())).shape[1]
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate,
                         drop_last=False, pin_memory=True)
    return train_ds, eval_ds, train_dl, eval_dl, code_dim


def split_batch(batch, num_actions, dev):
    """latents (B,T+1,..) -> inp (B,T,..), tgt (B,T,..); build aux targets."""
    lat = batch["latents"].to(dev, non_blocking=True)       # (B, T+1, z, h, w) fp32
    T = lat.shape[1] - 1
    inp_lat = lat[:, :T]
    tgt_lat = lat[:, 1:T + 1]
    actions = batch["actions"].to(dev, non_blocking=True).long()        # (B, T)
    if int(actions.max()) >= num_actions:
        raise ValueError(f"action id {int(actions.max())} >= num_actions={num_actions}; "
                         f"raise --num_actions")
    act_onehot = F.one_hot(actions, num_actions).float()               # (B, T, A)
    rewards = batch["rewards"].to(dev, non_blocking=True)              # (B, T)
    dones = batch["dones"].to(dev, non_blocking=True)                  # (B, T)
    reward_cls = (torch.sign(rewards) + 1).long()                     # {0,1,2}
    done_cls = dones.long()                                            # {0,1}
    code = batch["code"].to(dev, non_blocking=True)                    # (B, N, code_dim)
    code_mask = batch["code_mask"].to(dev, non_blocking=True)         # (B, N) bool
    return inp_lat, tgt_lat, act_onehot, reward_cls, done_cls, code, code_mask


def compute_loss(model, b, num_actions, dev, amp_dtype):
    inp_lat, tgt_lat, act_onehot, reward_cls, done_cls, code, code_mask = \
        split_batch(b, num_actions, dev)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        pred, rew_logits, done_logits = model(inp_lat, act_onehot, code, code_mask)
        loss_lat = F.mse_loss(pred.float(), tgt_lat)
        loss_rew = F.cross_entropy(rew_logits.float().reshape(-1, 3), reward_cls.reshape(-1))
        loss_done = F.cross_entropy(done_logits.float().reshape(-1, 2), done_cls.reshape(-1))
    loss = loss_lat + 0.1 * loss_rew + 0.1 * loss_done
    return loss, loss_lat.detach(), loss_rew.detach(), loss_done.detach()


@torch.no_grad()
def evaluate(model, eval_dl, num_actions, dev, amp_dtype, max_batches=20):
    model.eval()
    tot = {"loss": 0.0, "lat": 0.0, "rew": 0.0, "done": 0.0, "n": 0}
    for i, b in enumerate(eval_dl):
        if i >= max_batches:
            break
        loss, l_lat, l_rew, l_done = compute_loss(model, b, num_actions, dev, amp_dtype)
        tot["loss"] += loss.item(); tot["lat"] += l_lat.item()
        tot["rew"] += l_rew.item(); tot["done"] += l_done.item(); tot["n"] += 1
    model.train()
    n = max(tot["n"], 1)
    return {k: tot[k] / n for k in ("loss", "lat", "rew", "done")}


@torch.no_grad()
def dump_sample(model, eval_ds, vae, num_actions, dev, amp_dtype, out_png, n_frames=8):
    """Decode teacher-forced next-frame predictions vs GT for one eval clip."""
    from PIL import Image
    b = collate([eval_ds[0]])
    inp_lat, tgt_lat, act_onehot, _, _, code, code_mask = split_batch(b, num_actions, dev)
    model.eval()
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        pred, _, _ = model(inp_lat, act_onehot, code, code_mask)
    pred = pred.float()
    pred_frames = vae.decode(pred)        # (1,T,3,H,W)
    gt_frames = vae.decode(tgt_lat)
    model.train()

    def to_img(t):
        return (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    T = pred.shape[1]
    rows = []
    for ti in range(min(T, n_frames)):
        gt_img = to_img(gt_frames[0, ti]); pr_img = to_img(pred_frames[0, ti])
        sep = np.ones((gt_img.shape[0], 2, 3), dtype=np.uint8) * 255
        rows.append(np.concatenate([gt_img, sep, pr_img], axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--out", default="runs/c2w_v0")
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=8)
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
    ap.add_argument("--resume", default=None, help="checkpoint .pt to resume from")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    train_ds, eval_ds, train_dl, eval_dl, code_dim = build_loaders(args)

    # infer latent geometry from one sample
    s0 = train_ds[0]
    _, z, h, w = s0["latents"].shape
    print(f"latent z={z} spatial={h}x{w} | code_dim={code_dim}", flush=True)

    model = CausalDiT(latent_dim=z, embed_dim=args.embed_dim, num_layers=args.num_layers,
                      num_heads=args.num_heads, num_actions=args.num_actions,
                      spatial_size=h, max_frames=args.window, code_dim=code_dim).to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params", flush=True)

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

    vae = None  # lazy-load only for sample dumps

    def infinite(dl):
        while True:
            for b in dl:
                yield b
    it = infinite(train_dl)

    model.train()
    t0 = time.time()
    run = {"loss": 0.0, "lat": 0.0, "rew": 0.0, "done": 0.0, "n": 0}
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = next(it)
        loss, l_lat, l_rew, l_done = compute_loss(model, b, args.num_actions, dev, amp_dtype)
        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

        run["loss"] += loss.item(); run["lat"] += l_lat.item()
        run["rew"] += l_rew.item(); run["done"] += l_done.item(); run["n"] += 1

        if step % args.log_every == 0 or step == args.steps - 1:
            n = max(run["n"], 1); dt = time.time() - t0
            ips = (step - start_step + 1) / max(dt, 1e-6)
            print(f"step {step:6d} | lr {lr_at(step):.2e} | loss {run['loss']/n:.5f} "
                  f"(lat {run['lat']/n:.5f} rew {run['rew']/n:.3f} done {run['done']/n:.3f}) "
                  f"| {ips:.2f} it/s", flush=True)
            run = {"loss": 0.0, "lat": 0.0, "rew": 0.0, "done": 0.0, "n": 0}

        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, eval_dl, args.num_actions, dev, amp_dtype)
            print(f"  [eval] loss {ev['loss']:.5f} (lat {ev['lat']:.5f} "
                  f"rew {ev['rew']:.3f} done {ev['done']:.3f})", flush=True)

        if step > 0 and step % args.sample_every == 0:
            if vae is None:
                from models.vae import WanVAEWrapper
                vae = WanVAEWrapper(args.vae, device=dev)
            png = os.path.join(args.out, f"sample_{step:06d}.png")
            try:
                dump_sample(model, eval_ds, vae, args.num_actions, dev, amp_dtype, png)
                print(f"  [sample] saved {png} (cols: GT-next | pred-next)", flush=True)
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
