"""base eval episode: GT vs block-AR 生成对比。
用真实 episode 的 init latent + 真实动作序列做 block-AR 生成,
与真实帧对比,输出:
  1) 并排 mp4(逐真实帧 [GT | gen],16fps)—— 主产物
  2) 概览 grid png(每 latent 取代表帧,上 GT 下 gen)
"""
import os, sys, argparse, subprocess
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.causal_dit import CausalDiT, block_ar_generate
from models.vae import WanVAEWrapper
from models.pixel_codec import PixelCodec


def to_u8(t):
    return (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def save_video(frames_u8, path, fps, scale):
    """frames_u8: list of (H,W,3) uint8 -> mp4 via ffmpeg."""
    import imageio_ffmpeg
    import cv2
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    H, W = frames_u8[0].shape[:2]
    OH, OW = H * scale, W * scale
    p = subprocess.Popen(
        [ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-crf", "18", path],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for im in frames_u8:
        img = cv2.resize(im, (OW, OH), interpolation=cv2.INTER_NEAREST)
        p.stdin.write(np.ascontiguousarray(img, np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc")
    ap.add_argument("--ckpt", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world_act6_tc_fm/ckpt_final.pt")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_eval")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--max_latents", type=int, default=42, help="截到多少 latent(含 init)")
    ap.add_argument("--flow_steps", type=int, default=16)
    ap.add_argument("--num_actions", type=int, default=9)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--scale", type=int, default=6, help="视频每帧放大倍数")
    ap.add_argument("--out", default="outputs/custom_rollout_fm")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pixel", action="store_true",
                    help="pixel-space model: use PixelCodec (192-d 8x8 patch) not the VAE; "
                         "latents file is <variant>__<split>_pixel.pt (1 latent==1 frame)")
    ap.add_argument("--stride", type=int, default=1,
                    help="pixel-only frame subsample; =action_repeat(4) -> 4fps (1 frame/action)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = args.device
    tag = "_pixel" if args.pixel else ""

    # --- 模型 ---
    ck = torch.load(args.ckpt, map_location=dev)
    cargs = ck.get("args", {})
    pixel = args.pixel
    z, h = (192, 8) if pixel else (16, 8)
    code_bank = torch.load(os.path.join(args.root, "code_embeds.pt"), map_location="cpu")
    code_dim = next(iter(code_bank.values())).shape[1]
    model = CausalDiT(latent_dim=z, embed_dim=cargs.get("embed_dim", 512),
                      num_layers=cargs.get("num_layers", 12), num_heads=cargs.get("num_heads", 8),
                      num_actions=args.num_actions, spatial_size=h,
                      max_frames=cargs.get("window", 41) + 1, code_dim=code_dim,
                      block_size=cargs.get("block_size", 3)).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    if pixel:
        codec = PixelCodec(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], device=dev)
    else:
        codec = WanVAEWrapper(args.vae, device=dev)

    # --- 定位 episode 的 GT latent 与动作 ---
    if pixel:
        # pixel path: GT from raw frames; 1 latent == 1 frame; action_repeat maps 4 frames/action
        d = np.load(os.path.join(args.root, args.variant, f"{args.split}.npz"))
        epl = d["episode_lengths"]
        ar = int(d["action_repeat"]) if "action_repeat" in d else 1
        st = args.stride                                        # =ar -> 4fps (1 frame/action)
        f0 = int(sum(ar * int(k) + 1 for k in epl[:args.ep]))   # frame start of ep
        K = int(epl[args.ep])
        nfr_ep = ar * K + 1
        nsub = (nfr_ep - 1) // st + 1                           # subsampled steps available
        Ltot = min(nsub, args.max_latents)                      # subsampled steps == latent steps
        fidx = f0 + np.arange(Ltot) * st                        # raw frame indices of subsampled steps
        frames_np = d["frames"][fidx]                           # (Ltot,H,W,3) uint8
        gt_frames_in = torch.from_numpy(frames_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
        gt_lat = codec.encode_frames(gt_frames_in).cpu()        # (Ltot,192,8,8)
        # action for subsampled step q (1..Ltot-1): episode action (q*st-1)//ar
        actions = np.array([d["actions"][(q * st - 1) // ar] for q in range(1, Ltot)], np.int64)
    else:
        lat_all = torch.load(os.path.join(args.root, "latents", f"{args.variant}__{args.split}.pt"),
                             map_location="cpu")
        epl = lat_all["episode_lengths"].numpy()          # 每 ep 的 K(动作数),latent 数 = K+1
        l0 = int(epl[:args.ep].sum()) + args.ep            # 该 ep 起始 latent 下标(每 ep 多 1 个 init)
        a0 = int(epl[:args.ep].sum())                      # 该 ep 起始 action 下标
        K = int(epl[args.ep])
        Ltot = min(K + 1, args.max_latents)                # 实际用的 latent 数(含 init)
        gt_lat = lat_all["latents"][l0:l0 + Ltot].float()  # (Ltot, z, h, w)
        actions = lat_all["actions"][a0:a0 + (Ltot - 1)].numpy()   # (Ltot-1,)
    code = code_bank[args.variant].float().unsqueeze(0).to(dev)
    init = gt_lat[:1].unsqueeze(0).to(dev)             # (1,1,z,h,w) 只喂 init

    # --- block-AR 生成 ---
    gen_lat = block_ar_generate(model, init, actions, code, args.num_actions, dev,
                                cargs.get("block_size", 3), args.flow_steps)[0].cpu()  # (Ltot,z,h,w)
    assert gen_lat.shape[0] == Ltot, f"gen {gen_lat.shape[0]} != Ltot {Ltot}"

    # --- 解码全部帧(pixel: 1 latent->1 帧; vae: 时间维展开 4x) ---
    gt_frames = codec.decode_video(gt_lat.to(dev))
    gen_frames = codec.decode_video(gen_lat.to(dev))
    nfr = gt_frames.shape[0]
    assert gen_frames.shape[0] == nfr, f"帧数不一致 gt {nfr} vs gen {gen_frames.shape[0]}"

    # --- 主产物:逐帧 [GT | gen] 并排 mp4 ---
    import cv2
    sbs = []
    for i in range(nfr):
        g = to_u8(gt_frames[i]); p = to_u8(gen_frames[i])
        sep = np.ones((g.shape[0], 2, 3), np.uint8) * 255
        sbs.append(np.concatenate([g, sep, p], axis=1))    # 左 GT | 右 gen
    mp4 = os.path.join(args.out, f"{args.variant}_ep{args.ep}{tag}_gt_vs_gen.mp4")
    save_video(sbs, mp4, args.fps, args.scale)
    dur = nfr / args.fps
    print(f"K={K} Ltot={Ltot} frames={nfr} ({dur:.1f}s@{args.fps}fps) | 左=GT 右=gen | saved {mp4}",
          flush=True)

    # --- 概览 grid(每 latent 取代表帧,上 GT 下 gen) ---
    ALABEL = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U"}
    labels = ["init"] + [ALABEL.get(int(a), str(int(a))) for a in actions]
    frames_per_lat = 1 if pixel else 4          # pixel: 1 latent==1 帧; vae: 时间 4x
    s = 4; W = 64 * s; per_row = 8; tiles = []
    for li in range(Ltot):
        fi = 0 if li == 0 else frames_per_lat * (li - 1) + 1
        fi = min(fi, nfr - 1)
        g = cv2.resize(to_u8(gt_frames[fi]), (W, W), interpolation=cv2.INTER_NEAREST)
        p = cv2.resize(to_u8(gen_frames[fi]), (W, W), interpolation=cv2.INTER_NEAREST)
        sep = np.ones((3, W, 3), np.uint8) * 255
        tile = np.concatenate([g, sep, p], 0)              # 上 GT 下 gen
        tile = cv2.copyMakeBorder(tile, 16, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(tile, f"{li}:{labels[li]}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + per_row], 1) for r in range(0, len(tiles), per_row)]
    grid = np.concatenate(rows, 0)
    gpath = os.path.join(args.out, f"{args.variant}_ep{args.ep}{tag}_gt_vs_gen_grid.png")
    cv2.imwrite(gpath, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"grid saved {gpath}", flush=True)


if __name__ == "__main__":
    main()
