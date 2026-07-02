"""Multi-GPU (DDP) version of train_fm_pixel.py — same block-AR flow objective.

Launch with torchrun:
    torchrun --nproc_per_node=4 train_fm_pixel_ddp.py --patch 4 --stride 4 \
        --block_size 1 --batch_size 2 --steps 10000 --out .../ckpt_dir

Design notes (why a wrapper):
  DDP only installs its gradient-sync hooks on the module's forward(). Our training
  does TWO sub-forwards (forward_flow + forward_state) that bypass CausalDiT.forward,
  so wrapping CausalDiT directly would NOT sync those grads. We wrap it in
  FlowTrainWrapper whose forward() runs both sub-forwards and returns the losses, then
  DDP wraps THAT — so every training grad flows through one forward() and syncs.

Per-rank batch_size is `--batch_size`; effective batch = batch_size * accum * world_size.
Rank 0 does all logging / eval / sample / checkpointing. Single-process (no torchrun)
still works: world_size=1, behaves like train_fm_pixel.py.
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.pixel_dataset import PixelDataset, collate
from models.causal_dit import CausalDiT, block_ar_generate
from models.pixel_codec import PixelCodec
from train_fm import prep_batch


class FlowTrainWrapper(nn.Module):
    """Runs both sub-forwards + loss inside one forward() so DDP syncs all grads."""
    def __init__(self, net, block_size):
        super().__init__()
        self.net = net
        self.block_size = block_size

    def forward(self, lat, act_full, reward_cls, done_cls, code, code_mask):
        B, L = lat.shape[:2]
        bs = self.block_size
        n_blocks = (L + bs - 1) // bs
        tau_blk = torch.rand(B, n_blocks, device=lat.device)
        blk_id = torch.arange(L, device=lat.device) // bs
        tau = tau_blk[:, blk_id]
        tau[:, 0] = 1.0
        eps = torch.randn_like(lat)
        tau_b = tau[:, :, None, None, None]
        z_tau = (1.0 - tau_b) * eps + tau_b * lat
        v_star = lat - eps
        v = self.net.forward_flow(z_tau, tau, act_full, code, code_mask)
        loss_fm = F.mse_loss(v[:, 1:].float(), v_star[:, 1:])
        rew_logits, done_logits = self.net.forward_state(lat, act_full, code, code_mask)
        loss_rew = F.cross_entropy(rew_logits.float().reshape(-1, 3),
                                   reward_cls.reshape(-1), ignore_index=-100)
        loss_done = F.cross_entropy(done_logits.float().reshape(-1, 2),
                                    done_cls.reshape(-1), ignore_index=-100)
        loss = loss_fm + 0.1 * loss_rew + 0.1 * loss_done
        return loss, loss_fm.detach(), loss_rew.detach(), loss_done.detach()


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc")
    ap.add_argument("--out", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world_act6_tc_pixel_ddp")
    ap.add_argument("--window", type=int, default=41)
    ap.add_argument("--block_size", type=int, default=3)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8, help="PER-RANK batch; effective = bs*accum*world")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--embed_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=12)
    ap.add_argument("--num_heads", type=int, default=8)
    ap.add_argument("--flow_steps", type=int, default=8)
    ap.add_argument("--variants", nargs="*", default=["base"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--sample_every", type=int, default=2000)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    # ---- distributed init (torchrun sets RANK/WORLD_SIZE/LOCAL_RANK) ----
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    distributed = world > 1
    if distributed:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"
    is_main = (rank == 0)
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]

    def log(*a):
        if is_main:
            print(*a, flush=True)

    if is_main:
        os.makedirs(args.out, exist_ok=True)

    # ---- data ----
    tr = PixelDataset(args.root, "train", args.window, args.variants, stride=args.stride, patch=args.patch)
    ev = PixelDataset(args.root, "eval", args.window, args.variants, stride=args.stride, patch=args.patch)
    code_dim = next(iter(tr.code_embeds.values())).shape[1]
    log(f"train windows: {len(tr)} | eval windows: {len(ev)} | world={world}")
    train_sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True,
                                       drop_last=True) if distributed else None
    train_dl = DataLoader(tr, batch_size=args.batch_size, shuffle=(train_sampler is None),
                          sampler=train_sampler, num_workers=args.num_workers, collate_fn=collate,
                          drop_last=True, pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_dl = DataLoader(ev, batch_size=args.batch_size, shuffle=False, num_workers=2,
                         collate_fn=collate, drop_last=False, pin_memory=True)

    s0 = tr[0]
    L0, z, h, w = s0["latents"].shape
    assert z == 3 * args.patch ** 2 and h == 64 // args.patch
    log(f"pixel latents/window={L0} z={z} spatial={h}x{w} patch={args.patch} block={args.block_size}")

    net = CausalDiT(latent_dim=z, embed_dim=args.embed_dim, num_layers=args.num_layers,
                    num_heads=args.num_heads, num_actions=args.num_actions, spatial_size=h,
                    max_frames=L0, code_dim=code_dim, block_size=args.block_size).to(dev)
    log(f"Model: {sum(p.numel() for p in net.parameters())/1e6:.1f}M params")
    wrapper = FlowTrainWrapper(net, args.block_size).to(dev)
    # find_unused_parameters: CausalDiT keeps a legacy MSE head (output_proj) that the
    # flow/state path never uses -> its grads are absent, so DDP must be told to expect it.
    model = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=True) if distributed else wrapper

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        log(f"resumed from {args.resume} @ step {start_step}")

    def infinite(dl, sampler):
        ep = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(ep)
            for b in dl:
                yield b
            ep += 1
    it = infinite(train_dl, train_sampler)

    codec = None
    model.train()
    t0 = time.time()
    run = {"loss": 0.0, "fm": 0.0, "rew": 0.0, "done": 0.0, "n": 0}
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        acc = {"loss": 0.0, "fm": 0.0, "rew": 0.0, "done": 0.0}
        for a_i in range(args.accum):
            b = next(it)
            lat, act_full, reward_cls, done_cls, code, code_mask = prep_batch(b, args.num_actions, dev)
            # skip DDP all-reduce on non-final accum micro-steps
            sync_ctx = model.no_sync() if (distributed and a_i < args.accum - 1) else _nullctx()
            with sync_ctx, torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                loss, l_fm, l_rew, l_done = model(lat, act_full, reward_cls, done_cls, code, code_mask)
            scaled = loss / args.accum
            (scaler.scale(scaled) if scaler.is_enabled() else scaled).backward()
            acc["loss"] += loss.item() / args.accum; acc["fm"] += l_fm.item() / args.accum
            acc["rew"] += l_rew.item() / args.accum; acc["done"] += l_done.item() / args.accum
        if scaler.is_enabled():
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
            opt.step()

        run["loss"] += acc["loss"]; run["fm"] += acc["fm"]
        run["rew"] += acc["rew"]; run["done"] += acc["done"]; run["n"] += 1

        if is_main and (step % args.log_every == 0 or step == args.steps - 1):
            n = max(run["n"], 1); dt = time.time() - t0
            ips = (step - start_step + 1) / max(dt, 1e-6)
            log(f"step {step:6d} | lr {lr_at(step):.2e} | loss {run['loss']/n:.5f} "
                f"(fm {run['fm']/n:.5f} rew {run['rew']/n:.3f} done {run['done']/n:.3f}) | {ips:.2f} it/s")
            run = {"loss": 0.0, "fm": 0.0, "rew": 0.0, "done": 0.0, "n": 0}

        if is_main and step > 0 and step % args.eval_every == 0:
            net.eval()
            with torch.no_grad():
                tot = {"loss": 0.0, "fm": 0.0, "n": 0}
                for i, b in enumerate(eval_dl):
                    if i >= 20:
                        break
                    lat, act_full, reward_cls, done_cls, code, code_mask = prep_batch(b, args.num_actions, dev)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                        loss, l_fm, _, _ = wrapper(lat, act_full, reward_cls, done_cls, code, code_mask)
                    tot["loss"] += loss.item(); tot["fm"] += l_fm.item(); tot["n"] += 1
            net.train()
            nn_ = max(tot["n"], 1)
            log(f"  [eval] loss {tot['loss']/nn_:.5f} (fm {tot['fm']/nn_:.5f})")

        if is_main and step > 0 and step % args.save_every == 0:
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)}, os.path.join(args.out, f"ckpt_{step:06d}.pt"))
            torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                        "step": step, "args": vars(args)}, os.path.join(args.out, "ckpt_last.pt"))
            log(f"  [ckpt] step {step}")

    if is_main:
        torch.save({"model": net.state_dict(), "opt": opt.state_dict(),
                    "step": args.steps, "args": vars(args)}, os.path.join(args.out, "ckpt_final.pt"))
        log("done.")
    if distributed:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
