"""8-GPU DDP training for the per-frame code-conditioned Causal DiT flow model.

Why a wrapper: DDP only hooks gradient sync inside forward(). Our flow training runs
forward_flow and BYPASSES CausalDiT.forward(), so we wrap the flow pass + loss into
FlowTrainWrapper.forward() and DDP-wrap THAT. The loss math is identical to
train_fm.compute_loss (imported prep_batch keeps it in sync).

Launch (single node, 8 GPUs):
  torchrun --nproc_per_node=8 train_fm_ddp.py \
    --root /mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf \
    --window 20 --block_size 3 --batch_size 8 --steps 10000 --eval_every 500 \
    --action_mode crossattn --action_window 3 --action_compact --out <dir>
Effective batch = batch_size * nproc.  rank0 owns logging / eval / sampling / ckpt.
"""
import os, sys, time, math, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset.dataset import Code2WorldDataset, collate
from models.causal_dit import CausalDiT, block_ar_generate
from action_space import remap_to_compact, NUM_ACTIONS_COMPACT
from train_fm import prep_batch, evaluate, dump_sample   # reuse identical loss/eval logic


class FlowTrainWrapper(torch.nn.Module):
    """Wrap forward_flow + flow loss into ONE forward() so DDP syncs grads.
    Mirrors train_fm.compute_loss exactly (flow objective only)."""
    def __init__(self, model, block_size, compact):
        super().__init__()
        self.model = model
        self.block_size = block_size
        self.compact = compact

    def forward(self, batch, dev, num_actions):
        lat, act_pf, code, code_mask = prep_batch(
            batch, num_actions, dev, self.compact)
        B, L = lat.shape[:2]
        bs = self.block_size
        # per-BLOCK independent tau (init pos 0 held clean); eps per-latent
        n_blocks = (L + bs - 1) // bs
        tau_blk = torch.rand(B, n_blocks, device=dev)
        blk_id = torch.arange(L, device=dev) // bs
        tau = tau_blk[:, blk_id]
        tau[:, 0] = 1.0
        eps = torch.randn_like(lat)
        tau_b = tau[:, :, None, None, None]
        z_tau = (1.0 - tau_b) * eps + tau_b * lat
        v_star = lat - eps
        v = self.model.forward_flow(z_tau, tau, act_pf, code, code_mask)
        loss_fm = F.mse_loss(v[:, 1:].float(), v_star[:, 1:])
        return loss_fm, loss_fm.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--out", default="runs/c2w_pf_ddp")
    ap.add_argument("--window", type=int, default=20, help="actions per window; latents = window+1")
    ap.add_argument("--block_size", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8, help="PER-GPU batch")
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--action_compact", action="store_true")
    ap.add_argument("--action_mode", choices=["bias", "crossattn"], default="crossattn")
    ap.add_argument("--action_window", type=int, default=3)
    ap.add_argument("--embed_dim", type=int, default=768)
    ap.add_argument("--num_layers", type=int, default=24)
    ap.add_argument("--num_heads", type=int, default=16)
    ap.add_argument("--flow_steps", type=int, default=16)
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
    if args.action_compact:
        args.num_actions = NUM_ACTIONS_COMPACT

    # ---- DDP init ----
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"
    is_main = rank == 0
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}[args.amp]
    def log(*a):
        if is_main:
            print(*a, flush=True)
    if is_main:
        os.makedirs(args.out, exist_ok=True)

    # ---- data (DistributedSampler shards train across ranks) ----
    train_ds = Code2WorldDataset(args.root, split="train", window=args.window)
    eval_ds = Code2WorldDataset(args.root, split="eval", window=args.window)
    code_dim = next(iter(train_ds.code_embeds.values())).shape[1]
    train_sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                          num_workers=args.num_workers, collate_fn=collate, drop_last=True,
                          pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, collate_fn=collate, drop_last=False,
                         pin_memory=True)
    log(f"train windows {len(train_ds)} | eval {len(eval_ds)} | world={world} "
        f"| effective batch={args.batch_size*world}")

    s0 = train_ds[0]
    L0, z, h, w = s0["latents"].shape
    model = CausalDiT(latent_dim=z, embed_dim=args.embed_dim, num_layers=args.num_layers,
                      num_heads=args.num_heads, num_actions=args.num_actions,
                      spatial_size=h, max_frames=L0, code_dim=code_dim,
                      block_size=args.block_size, action_mode=args.action_mode,
                      action_window=args.action_window).to(dev)
    if is_main:
        n = sum(p.numel() for p in model.parameters()) / 1e6
        log(f"Model {n:.1f}M | L={L0} z={z} spatial={h}x{w} | action_mode={args.action_mode} "
            f"window={args.action_window} num_actions={args.num_actions}")

    wrapper = FlowTrainWrapper(model, args.block_size, args.action_compact).to(dev)
    # CausalDiT keeps legacy output_proj unused in the flow path -> find_unused_parameters
    ddp = DDP(wrapper, device_ids=[local_rank], find_unused_parameters=True)
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
        model.load_state_dict(ck["model"], strict=False)  # tolerate old reward/done heads
        opt.load_state_dict(ck["opt"])
        start_step = ck.get("step", 0)
        log(f"resumed from {args.resume} @ step {start_step}")

    def infinite(dl):
        ep = 0
        while True:
            train_sampler.set_epoch(ep)   # reshuffle each epoch across ranks
            for b in dl:
                yield b
            ep += 1
    it = infinite(train_dl)

    vae = None
    model.train()
    t0 = time.time()
    run = {"fm": 0.0, "n": 0}
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        b = next(it)
        loss, l_fm = ddp(b, dev, args.num_actions)
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

        run["fm"] += l_fm.item(); run["n"] += 1

        if is_main and (step % args.log_every == 0 or step == args.steps - 1):
            n = max(run["n"], 1); dt = time.time() - t0
            ips = (step - start_step + 1) / max(dt, 1e-6)
            log(f"step {step:6d} | lr {lr_at(step):.2e} | fm {run['fm']/n:.5f} "
                f"| {ips:.2f} it/s (x{world})")
            run = {"fm": 0.0, "n": 0}

        if is_main and step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, eval_dl, args.num_actions, dev, amp_dtype, args.block_size,
                          compact=args.action_compact)
            log(f"  [eval] fm {ev['fm']:.5f}")
            model.train()

        if is_main and step > 0 and step % args.sample_every == 0:
            if vae is None:
                from models.vae import WanVAEWrapper
                vae = WanVAEWrapper(args.vae, device=dev)
            png = os.path.join(args.out, f"sample_{step:06d}.png")
            try:
                dump_sample(model, eval_ds, vae, args.num_actions, dev,
                            args.block_size, args.flow_steps, png, compact=args.action_compact)
                log(f"  [sample] saved {png}")
            except Exception as e:
                log(f"  [sample] skipped: {e}")
            model.train()

        if is_main and step > 0 and step % args.save_every == 0:
            ckpt = os.path.join(args.out, f"ckpt_{step:06d}.pt")
            payload = {"model": model.state_dict(), "opt": opt.state_dict(),
                       "step": step, "args": vars(args)}
            torch.save(payload, ckpt)
            torch.save(payload, os.path.join(args.out, "ckpt_last.pt"))
            log(f"  [ckpt] {ckpt}")

    if is_main:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": args.steps, "args": vars(args)},
                   os.path.join(args.out, "ckpt_final.pt"))
        log("done.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

