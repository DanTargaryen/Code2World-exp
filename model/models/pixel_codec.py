"""Pixel-space codec: replace the Wan VAE with a parameter-free 8x8 patchify.

A 64x64x3 frame -> 8x8 grid of 8x8x3 patches -> "latent" (192, 8, 8), where the
192 = 8*8*3 raw pixels of each patch become the channel dim. This mirrors the VAE
latent geometry (64 tokens/frame, 8x8 spatial) exactly, so CausalDiT is reused
unchanged with latent_dim=192, spatial_size=8. patchify/unpatchify are pure
reshapes (lossless).

Pixels are per-channel normalized to ~unit variance so the flow target z1-eps
(eps~N(0,I)) is well-scaled. Stats (mean/std over the RGB channels) are computed
once from the dataset and passed in; decode reverses the normalization.

decode_video() mirrors WanVAEWrapper.decode_video's signature/semantics
((L,192,8,8) normalized -> (T,3,H,W) in [0,1]) EXCEPT there is NO temporal
expansion here: L latents -> L frames (pixel path models every frame directly).
"""
import torch
import torch.nn as nn
from einops import rearrange

P = 8  # patch size (8x8 -> 8x8 grid on 64x64)


class PixelCodec(nn.Module):
    def __init__(self, mean=None, std=None, patch=P, device="cuda"):
        """mean/std: per-channel (3,) tensors over pixels in [0,1]. If None, identity
        (mean 0, std 1) — but you should pass real stats for training."""
        super().__init__()
        self.patch = patch
        if mean is None:
            mean = torch.zeros(3)
        if std is None:
            std = torch.ones(3)
        # store as (1,3,1,1) for broadcasting over (N,3,H,W)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.device = device
        self.to(device)

    @torch.no_grad()
    def encode_frames(self, frames):
        """frames: (T, 3, H, W) in [0,1] -> latent (T, 3*p*p, H/p, W/p) normalized.

        Channel layout of the patch: (c p1 p2) with c the RGB channel — chosen so
        unpatchify is the exact inverse."""
        f = frames.to(self.device, torch.float32)
        f = (f - self.mean) / self.std                       # per-channel normalize
        lat = rearrange(f, "t c (h p1) (w p2) -> t (c p1 p2) h w", p1=self.patch, p2=self.patch)
        return lat                                            # (T, 192, 8, 8)

    @torch.no_grad()
    def decode_video(self, latent):
        """latent: (L, 3*p*p, h, w) normalized -> frames (L, 3, H, W) in [0,1].

        No temporal expansion (pixel path is 1 latent == 1 frame). Name mirrors
        WanVAEWrapper.decode_video so rollout code can call either."""
        lat = latent.to(self.device, torch.float32)
        f = rearrange(lat, "t (c p1 p2) h w -> t c (h p1) (w p2)",
                      c=3, p1=self.patch, p2=self.patch)
        f = f * self.std + self.mean                          # denormalize
        return f.clamp(0, 1)


def compute_pixel_stats(frames_uint8, sample=200000):
    """frames_uint8: (N,H,W,3) uint8 -> per-channel (mean,std) in [0,1].
    Subsamples up to `sample` frames for speed."""
    import numpy as np
    N = len(frames_uint8)
    idx = np.arange(N) if N <= sample else np.random.RandomState(0).choice(N, sample, replace=False)
    x = frames_uint8[idx].astype("float32") / 255.0          # (n,H,W,3)
    mean = x.reshape(-1, 3).mean(0)
    std = x.reshape(-1, 3).std(0)
    return mean, std
