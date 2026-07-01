"""Verify Wan 2.1 VAE reconstruction quality on Procgen CoinRun 32x32 frames."""
import torch
import numpy as np
import os
import argparse
from PIL import Image


def load_frames(data_path, n_samples=200):
    """Load a subset of frames from collected data."""
    data = np.load(data_path)
    frames = data["frames"]  # (N, 32, 32, 3) uint8
    idx = np.random.choice(len(frames), min(n_samples, len(frames)), replace=False)
    return frames[idx]


def frames_to_tensor(frames):
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32 in [0, 1]"""
    t = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    return t


def compute_psnr(original, reconstructed):
    """Compute PSNR between two tensors in [0, 1]."""
    mse = ((original - reconstructed) ** 2).mean(dim=(1, 2, 3))
    psnr = -10 * torch.log10(mse + 1e-8)
    return psnr


def test_wan_vae(data_path, model_name="/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers",
                 n_samples=200, out_dir="vae_check", device="cuda:0"):
    """Load Wan 2.1 VAE, encode-decode Procgen frames, measure quality."""
    from diffusers import AutoencoderKLWan

    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading frames from {data_path}...")
    frames = load_frames(data_path, n_samples)
    print(f"  {frames.shape} frames loaded")

    print(f"Loading Wan 2.1 VAE from {model_name}...")
    vae = AutoencoderKLWan.from_pretrained(model_name, subfolder="vae",
                                            torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    print(f"  VAE loaded. z_dim: {vae.config.z_dim}")

    x = frames_to_tensor(frames).to(device)  # (N, 3, 32, 32)
    print(f"  Input tensor: {x.shape}, range [{x.min():.2f}, {x.max():.2f}]")

    # Wan VAE expects video input: (B, C, T, H, W)
    # For single frames, add T=1 dimension
    x_video = x.unsqueeze(2)  # (N, 3, 1, 32, 32)

    batch_size = 16
    all_recon = []
    all_latents = []

    print("Encoding and decoding...")
    with torch.no_grad():
        for i in range(0, len(x_video), batch_size):
            batch = x_video[i:i+batch_size]
            # Normalize to [-1, 1] as Wan VAE expects
            batch_norm = batch * 2.0 - 1.0

            latent_dist = vae.encode(batch_norm)
            latent = latent_dist.latent_dist.sample()
            recon = vae.decode(latent).sample

            # Back to [0, 1]
            recon = (recon + 1.0) / 2.0
            recon = recon.clamp(0, 1)

            all_recon.append(recon.squeeze(2).cpu())  # remove T dim
            all_latents.append(latent.cpu())

            if i == 0:
                print(f"  Latent shape: {latent.shape}")
                print(f"  Latent stats: mean={latent.mean():.3f}, std={latent.std():.3f}")

    recon_all = torch.cat(all_recon, dim=0)  # (N, 3, 32, 32)
    latents_all = torch.cat(all_latents, dim=0)

    # Compute metrics
    psnr_values = compute_psnr(x.cpu(), recon_all)
    mean_psnr = psnr_values.mean().item()
    mse_value = ((x.cpu() - recon_all) ** 2).mean().item()

    print(f"\n=== Results ===")
    print(f"  Mean PSNR: {mean_psnr:.2f} dB")
    print(f"  Mean MSE:  {mse_value:.6f}")
    print(f"  PSNR range: [{psnr_values.min():.1f}, {psnr_values.max():.1f}] dB")
    print(f"  Latent shape per frame: {latents_all.shape[1:]} "
          f"(spatial: {latents_all.shape[3]}x{latents_all.shape[4]}, "
          f"channels: {latents_all.shape[1]})")

    # Save comparison images
    print(f"\nSaving visual comparisons to {out_dir}/...")
    n_vis = min(20, len(frames))
    for i in range(n_vis):
        orig = (x[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        rec = (recon_all[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Scale up 4x for visibility
        orig_big = np.kron(orig, np.ones((4, 4, 1))).astype(np.uint8)
        rec_big = np.kron(rec, np.ones((4, 4, 1))).astype(np.uint8)
        diff = np.abs(orig_big.astype(float) - rec_big.astype(float)).astype(np.uint8) * 3

        comparison = np.concatenate([orig_big, rec_big, diff], axis=1)
        Image.fromarray(comparison).save(os.path.join(out_dir, f"compare_{i:03d}.png"))

    # Summary
    verdict = "PASS ✓" if mean_psnr >= 20.0 else "FAIL ✗ (consider fine-tune or VQ-VAE fallback)"
    print(f"\n  Verdict: {verdict}")
    print(f"  Threshold: PSNR >= 20 dB for key objects (player, coin, platforms) to be recognizable")

    return mean_psnr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to collected .npz")
    parser.add_argument("--model", default="Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--out", default="vae_check")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    test_wan_vae(args.data, args.model, args.samples, args.out, args.device)
