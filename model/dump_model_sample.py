"""Dump ONE training sample exactly as the model consumes it, into examples/.

Shows the full model-facing sample = 4 tensors that prep_batch/compute_loss feed
the flow model, plus human-readable views of each:

  text (code condition)  -> config.yaml (raw) + Qwen embedding `code` (N,896)
  initial frame          -> latent[0] (the clean init condition) + its decoded RGB
  target video           -> latent[1:] (what the model predicts) + decoded RGB frames
  actions                -> per-frame one-hot act_pf (R*L, A), R=4 frames/latent

Runs the real preprocessing (Wan VAE encode + Qwen encode) so every tensor is the
actual thing the model sees. Writes a self-contained bundle to examples/<tag>/.

    python -u dump_model_sample.py --root /tmp/sample_ds --variant base --split episodes_train --ep 0
"""
import os, sys, json, argparse, subprocess
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "dataset"))
from action_space import remap_to_compact

ALABEL = {1: "left", 2: "left+jump", 4: "stay", 5: "jump", 7: "right", 8: "right+jump", 0: "noop"}
ASHORT = {1: "L", 2: "L+U", 4: "STAY", 5: "UP", 7: "R", 8: "R+U", 0: "-"}


def save_grid(frames, labels, path, per_row=9, s=3):
    import cv2
    W = 64 * s; tiles = []
    for i, fr in enumerate(frames):
        cell = cv2.resize(fr, (W, W), interpolation=cv2.INTER_NEAREST)
        cell = cv2.copyMakeBorder(cell, 16, 2, 2, 2, cv2.BORDER_CONSTANT, value=(20, 20, 20))
        cv2.putText(cell, labels[i], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(cell)
    while len(tiles) % per_row:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[r:r + per_row], 1) for r in range(0, len(tiles), per_row)]
    cv2.imwrite(path, cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))


def to_u8(t):
    return (t.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--split", default="episodes_train")
    ap.add_argument("--ep", type=int, default=0)
    ap.add_argument("--window", type=int, default=20, help="L-1 target latents (window)")
    ap.add_argument("--vae", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers")
    ap.add_argument("--qwen", default="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/Qwen2.5-0.5B")
    ap.add_argument("--out", default="../examples")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num-actions", type=int, default=6, help="compact action count")
    args = ap.parse_args()
    dev = args.device
    R = 4  # vae temporal ratio (frames per latent)

    tag = f"model_sample_{args.variant}_{args.split}_ep{args.ep}"
    bundle = os.path.join(args.out, tag)
    os.makedirs(bundle, exist_ok=True)

    # ---- locate one episode's raw frames + per-frame actions ----
    npz = np.load(os.path.join(args.root, args.variant, f"{args.split}.npz"))
    el = npz["episode_lengths"]; ar = int(npz["action_repeat"])
    assert ar == R, f"this dump assumes vae_ratio {R}, dataset action_repeat={ar}"
    f0 = int(el[:args.ep].sum()) + args.ep
    a0 = int(el[:args.ep].sum()) * ar
    K = int(el[args.ep])                       # latent steps in this episode
    L = min(args.window, K) + 1                # sequence length incl init
    n_frames = (L - 1) * ar + 1                # frames feeding this window
    frames = npz["frames"][f0: f0 + n_frames]              # (n_frames,64,64,3) u8
    raw_actions = npz["actions"][a0: a0 + (L - 1) * ar]    # (R*(L-1),) raw ids

    # ---- text (code condition): FULL rendered rulebook (not the sparse yaml) ----
    from game_config import load_config, render_code_condition
    from transformers import AutoModel, AutoTokenizer
    os.environ["HF_HUB_OFFLINE"] = "1"
    cfg_path = os.path.join(args.root, args.variant, "config.yaml")
    cfg_text = render_code_condition(load_config(cfg_path))   # exactly what precompute feeds
    tok = AutoTokenizer.from_pretrained(args.qwen)
    qwen = AutoModel.from_pretrained(args.qwen, torch_dtype=torch.float32).to(dev).eval()
    ids = tok(cfg_text, return_tensors="pt", truncation=True, max_length=5120).to(dev)
    with torch.no_grad():
        code = qwen(**ids).last_hidden_state[0].half().cpu()   # (N,896) = the `code` tensor

    # ---- frames -> VAE latents (init + targets) ----
    from models.vae import WanVAEWrapper
    vae = WanVAEWrapper(args.vae, device=dev)
    fr = torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2)  # (n,3,64,64)
    lat = vae.encode_video(fr.to(dev)).cpu()               # (L,16,8,8) = the `latents` tensor
    init_lat = lat[:1]                                     # (1,16,8,8) clean init condition
    target_lat = lat[1:]                                   # (L-1,16,8,8) what the model predicts

    # ---- actions -> per-frame one-hot act_pf (R*L, A) ----
    acts_c = remap_to_compact(torch.from_numpy(raw_actions.astype(np.int64)))  # -> [0..5]
    act_pf = torch.zeros(R * L, args.num_actions)
    act_pf[R:] = torch.nn.functional.one_hot(acts_c, args.num_actions).float()  # first R = init null

    # ---- decode latents back to RGB for eyeballing ----
    dec = vae.decode_video(lat.to(dev))                    # (n_frames,3,64,64) reconstructed
    dec_u8 = [to_u8(dec[i]) for i in range(dec.shape[0])]

    # save the actual tensors the model sees
    torch.save({"latents": lat, "code": code, "act_pf": act_pf,
                "init_latent": init_lat, "target_latents": target_lat,
                "raw_actions": torch.from_numpy(raw_actions)},
               os.path.join(bundle, "model_input_tensors.pt"))

    # code condition forms: the sparse config the user wrote, the FULL rendered
    # rulebook that is actually encoded, and the raw source.cpp for reference.
    import shutil
    shutil.copy(cfg_path, os.path.join(bundle, "config.yaml"))          # sparse (user-authored)
    with open(os.path.join(bundle, "code_condition.txt"), "w") as f:    # what Qwen encodes
        f.write(cfg_text)
    src = os.path.join(args.root, args.variant, "source.cpp")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(bundle, "source.cpp"))

    # initial frame (raw + vae-reconstructed)
    import cv2
    cv2.imwrite(os.path.join(bundle, "initial_frame.png"),
                cv2.cvtColor(cv2.resize(frames[0], (384, 384), interpolation=cv2.INTER_NEAREST),
                             cv2.COLOR_RGB2BGR))

    # target video (raw frames 1..n) as mp4
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    OW = OH = 64 * 6
    p = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}",
                          "-r", "8", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                          "-crf", "16", os.path.join(bundle, "target_video.mp4")],
                         stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames[1:]:
        p.stdin.write(np.ascontiguousarray(cv2.resize(f, (OW, OH), interpolation=cv2.INTER_NEAREST), np.uint8).tobytes())
    p.stdin.close(); p.wait()

    # frame grid with per-frame action labels (raw frames)
    labels = ["init"] + [f"{i}:{ASHORT.get(int(raw_actions[i-1]), int(raw_actions[i-1]))}"
                         for i in range(1, n_frames)]
    save_grid(list(frames), labels, os.path.join(bundle, "frames_grid.png"))

    # per-frame actions (id + label), aligned to frames[1:]
    with open(os.path.join(bundle, "actions.json"), "w") as f:
        json.dump({"aligned_to": "frames[1:] (action i produced frame i+1)",
                   "raw_action_ids": [int(a) for a in raw_actions],
                   "compact_ids": [int(a) for a in acts_c.tolist()],
                   "labels": [ALABEL.get(int(a), str(int(a))) for a in raw_actions]},
                  f, indent=2)

    # meta: the sample's exact tensor shapes = what the model sees
    meta = {
        "tag": tag, "variant": args.variant, "split": args.split, "episode": args.ep,
        "sequence_length_L": L, "target_latents": L - 1, "vae_ratio_R": R,
        "n_raw_frames": n_frames,
        "model_input_tensors": {
            "latents  (L,16,8,8)":  list(lat.shape),
            "  init_latent[0]":     list(init_lat.shape),
            "  target_latents[1:]": list(target_lat.shape),
            "code     (N,896)":     list(code.shape),
            "act_pf   (R*L,A)":     list(act_pf.shape),
        },
        "code_condition": {"config_yaml": cfg_path, "qwen_tokens_N": int(code.shape[0])},
        "n_per_frame_actions": int(len(raw_actions)),
    }
    with open(os.path.join(bundle, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"saved model-sample bundle -> {bundle}/", flush=True)
    for k, v in meta["model_input_tensors"].items():
        print(f"  {k}: {v}", flush=True)
    print("  files: " + "  ".join(sorted(os.listdir(bundle))), flush=True)


if __name__ == "__main__":
    main()
