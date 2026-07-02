"""Train the code-conditioned Causal DiT as a BLOCK-AUTOREGRESSIVE flow model.

Single-stream Diffusion Forcing over the temporally-compressed latent sequence:
each window is L = window+1 latents (e.g. 42 = 14 blocks of 3). Every latent gets an
INDEPENDENT tau~U(0,1) and eps~N(0,I); the init latent (index 0) is held clean
(tau=1) and excluded from the loss. Temporal attention is block-causal so a block of
`block_size` latents denoises jointly conditioned on earlier (cleaner) blocks.

  z_tau = (1-tau)*eps + tau*z1
  v     = forward_flow(z_tau, tau, action, code)          # per-latent velocity
  L_fm  = || v - (z1 - eps) ||^2     over latents 1..L-1   # init excluded
  state = forward_state(clean z1, ...) -> reward/done      # separate clean pass, tau-free
  L     = L_fm + 0.1*CE(reward) + 0.1*CE(done)

action alignment: latent i (i>=1) is produced by action a[i-1]; sequence position 0
(init) gets a null (all-zero) action.

Example (10s @16fps -> 165 frames -> 42 latents):
    python -u train_fm.py \
        --root /mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc \
        --window 41 --block_size 3 --batch_size 8 --steps 20000 --out runs/c2w_fm
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.dataset import Code2WorldDataset, collate
from models.causal_dit import CausalDiT, block_ar_generate
from action_space import remap_to_compact, NUM_ACTIONS_COMPACT, NUM_ACTIONS_FULL


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


def prep_batch(batch, num_actions, dev, compact=False):
    """Returns full-sequence tensors (L = window+1 latents).
      lat  (B,L,z,h,w)   act_full (B,L,A)  pos0 null
      reward_cls/done_cls (B,L) long, pos0 = -100 (ignored by CE)
      code (B,N,Dc)      code_mask (B,N) bool
    compact=True remaps RAW Procgen ids [1,2,4,5,7,8] -> dense [0..5] (6-d one-hot)."""
    lat = batch["latents"].to(dev, non_blocking=True)                 # (B, L, z, h, w)
    B, L = lat.shape[:2]
    actions = batch["actions"].to(dev, non_blocking=True).long()      # (B, L-1) RAW ids
    if compact:
        actions = remap_to_compact(actions)
    if int(actions.max()) >= num_actions:
        raise ValueError(f"action id {int(actions.max())} >= num_actions={num_actions}")
    act_full = torch.zeros(B, L, num_actions, device=dev)
    act_full[:, 1:] = F.one_hot(actions, num_actions).float()         # pos0 stays null
    rewards = batch["rewards"].to(dev, non_blocking=True)             # (B, L-1)
    dones = batch["dones"].to(dev, non_blocking=True)                 # (B, L-1)
    reward_cls = torch.full((B, L), -100, dtype=torch.long, device=dev)
    done_cls = torch.full((B, L), -100, dtype=torch.long, device=dev)
    reward_cls[:, 1:] = (torch.sign(rewards) + 1).long()             # {0,1,2}
    done_cls[:, 1:] = dones.long()                                    # {0,1}
    code = batch["code"].to(dev, non_blocking=True)
    code_mask = batch["code_mask"].to(dev, non_blocking=True)
    return lat, act_full, reward_cls, done_cls, code, code_mask


def compute_loss(model, b, num_actions, dev, amp_dtype, block_size=3, compact=False):
    lat, act_full, reward_cls, done_cls, code, code_mask = prep_batch(b, num_actions, dev, compact)
    B, L = lat.shape[:2]
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        # per-BLOCK independent noise level (all latents in a block share tau, matching
        # block-AR inference which denoises a whole block jointly); tau independent across
        # blocks (Diffusion Forcing). eps stays per-latent. init (pos 0) held clean.
        n_blocks = (L + block_size - 1) // block_size
        tau_blk = torch.rand(B, n_blocks, device=dev)                 # (B, n_blocks)
        blk_id = torch.arange(L, device=dev) // block_size            # (L,)
        tau = tau_blk[:, blk_id]                                      # (B, L) broadcast per block
        tau[:, 0] = 1.0
        eps = torch.randn_like(lat)
        tau_b = tau[:, :, None, None, None]
        z_tau = (1.0 - tau_b) * eps + tau_b * lat
        v_star = lat - eps
        v = model.forward_flow(z_tau, tau, act_full, code, code_mask)
        loss_fm = F.mse_loss(v[:, 1:].float(), v_star[:, 1:])         # exclude init
        rew_logits, done_logits = model.forward_state(lat, act_full, code, code_mask)
        loss_rew = F.cross_entropy(rew_logits.float().reshape(-1, 3),
                                   reward_cls.reshape(-1), ignore_index=-100)
        loss_done = F.cross_entropy(done_logits.float().reshape(-1, 2),
                                    done_cls.reshape(-1), ignore_index=-100)
    loss = loss_fm + 0.1 * loss_rew + 0.1 * loss_done
    return loss, loss_fm.detach(), loss_rew.detach(), loss_done.detach()


@torch.no_grad()
def evaluate(model, eval_dl, num_actions, dev, amp_dtype, block_size=3, max_batches=20, compact=False):
    model.eval()
    tot = {"loss": 0.0, "fm": 0.0, "rew": 0.0, "done": 0.0, "n": 0}
    for i, b in enumerate(eval_dl):
        if i >= max_batches:
            break
        loss, l_fm, l_rew, l_done = compute_loss(model, b, num_actions, dev, amp_dtype, block_size, compact)
        tot["loss"] += loss.item(); tot["fm"] += l_fm.item()
        tot["rew"] += l_rew.item(); tot["done"] += l_done.item(); tot["n"] += 1
    model.train()
    n = max(tot["n"], 1)
    return {k: tot[k] / n for k in ("loss", "fm", "rew", "done")}


@torch.no_grad()
def dump_sample(model, eval_ds, vae, num_actions, dev, block_size, flow_steps, out_png,
                n_latents=8, compact=False):
    """Block-AR generate from one eval clip's init + GT actions; compare to GT frames.
    Decodes via the TEMPORAL VAE (decode_video) so 1 latent -> 4 frames."""
    from PIL import Image
    b = collate([eval_ds[0]])
    lat = b["latents"][0].to(dev)                          # (L, z, h, w)
    actions = b["actions"][0].cpu().numpy()                # (L-1,) RAW ids
    if compact:
        actions = remap_to_compact(torch.as_tensor(actions)).numpy()
    code = b["code"][:1].to(dev)
    model.eval()
    init = lat[:1].unsqueeze(0)                            # (1,1,z,h,w)
    gen = block_ar_generate(model, init, actions, code, num_actions, dev,
                            block_size, flow_steps)[0]     # (L, z, h, w)
    # show first n_latents latents decoded (each -> 4 frames via temporal VAE)
    k = min(n_latents, gen.shape[0])
    gen_frames = vae.decode_video(gen[:k])                 # (4*(k-1)+1, 3, H, W)
    gt_frames = vae.decode_video(lat[:k])
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
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--out", default="runs/c2w_fm")
    ap.add_argument("--window", type=int, default=41, help="actions per window; latents = window+1")
    ap.add_argument("--block_size", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--action_compact", action="store_true",
                    help="remap raw ids [1,2,4,5,7,8]->[0..5] and use 6-d one-hot "
                         "(drops the always-zero 0/3/6 dims); overrides --num_actions to 6")
    ap.add_argument("--action_mode", choices=["bias", "crossattn"], default="bias",
                    help="how actions enter each DiT block: additive bias (baseline) "
                         "or Matrix-Game-style window cross-attention")
    ap.add_argument("--action_window", type=int, default=3,
                    help="crossattn: #latents of action history per token (incl. current)")
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=8, help="Euler steps per block at sample time")
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
    if args.action_compact:
        args.num_actions = NUM_ACTIONS_COMPACT     # 6, override (dims 0/3/6 dropped)
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    train_ds, eval_ds, train_dl, eval_dl, code_dim = build_loaders(args)

    s0 = train_ds[0]
    L0, z, h, w = s0["latents"].shape
    print(f"latents/window={L0} (window={args.window}+1) z={z} spatial={h}x{w} "
          f"| code_dim={code_dim} | block_size={args.block_size}", flush=True)
    if (L0 % args.block_size) != 0:
        print(f"  [warn] L={L0} not divisible by block_size={args.block_size}", flush=True)

    model = CausalDiT(latent_dim=z, embed_dim=args.embed_dim, num_layers=args.num_layers,
                      num_heads=args.num_heads, num_actions=args.num_actions,
                      spatial_size=h, max_frames=L0, code_dim=code_dim,
                      block_size=args.block_size, action_mode=args.action_mode,
                      action_window=args.action_window).to(dev)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {n_params:.1f}M params | action_mode={args.action_mode}"
          f"{f' window={args.action_window}' if args.action_mode=='crossattn' else ''}",
          flush=True)

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

    vae = None

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
        loss, l_fm, l_rew, l_done = compute_loss(model, b, args.num_actions, dev, amp_dtype, args.block_size, args.action_compact)
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
            ev = evaluate(model, eval_dl, args.num_actions, dev, amp_dtype, args.block_size,
                          compact=args.action_compact)
            print(f"  [eval] loss {ev['loss']:.5f} (fm {ev['fm']:.5f} "
                  f"rew {ev['rew']:.3f} done {ev['done']:.3f})", flush=True)

        if step > 0 and step % args.sample_every == 0:
            if vae is None:
                from models.vae import WanVAEWrapper
                vae = WanVAEWrapper(args.vae, device=dev)
            png = os.path.join(args.out, f"sample_{step:06d}.png")
            try:
                dump_sample(model, eval_ds, vae, args.num_actions, dev,
                            args.block_size, args.flow_steps, png, compact=args.action_compact)
                print(f"  [sample] saved {png} (cols: GT | block-AR gen)", flush=True)
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
