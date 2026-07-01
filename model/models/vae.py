"""Wan 2.1 VAE wrapper: frozen, encode/decode single frames to/from 8x8x16 latent.

Handles the per-channel latent normalization (latents_mean/std) so that the DiT
operates on roughly unit-variance latents.
"""
import torch
import torch.nn as nn


class WanVAEWrapper(nn.Module):
    def __init__(self, model_path, device="cuda", dtype=torch.float32):
        super().__init__()
        from diffusers import AutoencoderKLWan
        self.vae = AutoencoderKLWan.from_pretrained(
            model_path, subfolder="vae", torch_dtype=dtype)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad_(False)

        z_dim = self.vae.config.z_dim  # 16
        mean = torch.tensor(self.vae.config.latents_mean).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std).view(1, z_dim, 1, 1, 1)
        self.register_buffer("latents_mean", mean)
        self.register_buffer("latents_std", std)
        self.z_dim = z_dim
        self.device = device
        self.dtype = dtype
        self.to(device)

    @torch.no_grad()
    def encode(self, frames):
        """frames: (B, T, 3, H, W) in [0,1]  ->  latent (B, T, z_dim, h, w) normalized.

        Wan VAE is a video VAE (B, C, T, H, W). We treat each frame independently
        by collapsing T into the temporal axis with temporal compression handled
        per-frame (T=1 chunks).
        """
        B, T, C, H, W = frames.shape
        x = frames.reshape(B * T, C, 1, H, W)            # (B*T, 3, 1, H, W)
        x = x.to(self.device, self.dtype) * 2.0 - 1.0     # [0,1] -> [-1,1]
        lat = self.vae.encode(x).latent_dist.mode()       # (B*T, z, 1, h, w)
        # normalize
        lat = (lat - self.latents_mean) / self.latents_std
        _, z, _, h, w = lat.shape
        lat = lat.reshape(B, T, z, h, w)
        return lat

    @torch.no_grad()
    def decode(self, latent):
        """latent: (B, T, z_dim, h, w) normalized  ->  frames (B, T, 3, H, W) in [0,1]."""
        B, T, z, h, w = latent.shape
        lat = latent.reshape(B * T, z, 1, h, w).to(self.device, self.dtype)
        lat = lat * self.latents_std + self.latents_mean   # denormalize
        rec = self.vae.decode(lat).sample                  # (B*T, 3, 1, H, W)
        rec = (rec + 1.0) / 2.0
        rec = rec.clamp(0, 1)
        _, C, _, H, W = rec.shape
        return rec.reshape(B, T, C, H, W)

    @torch.no_grad()
    def encode_video(self, frames):
        """TEMPORAL-COMPRESSING encode for one clip.

        frames: (T, 3, H, W) in [0,1]  ->  latent (L, z_dim, h, w) normalized,
        where L = 1 + (T-1)//4 (Wan 2.1 causal 3D VAE: first frame -> 1 latent,
        then every 4 frames -> 1 latent). Requires T == 4*(L-1)+1.

        Unlike encode() (per-frame, T=1 chunks), this feeds the whole temporal
        axis at once so the VAE's temporal downsampling is actually used. One
        action_repeat=4 chunk of 4 frames therefore maps to exactly 1 latent step.
        """
        T, C, H, W = frames.shape
        x = frames.to(self.device, self.dtype) * 2.0 - 1.0   # [-1,1]
        x = x.permute(1, 0, 2, 3).unsqueeze(0)                # (1, 3, T, H, W)
        lat = self.vae.encode(x).latent_dist.mode()           # (1, z, L, h, w)
        lat = (lat - self.latents_mean) / self.latents_std
        lat = lat[0].permute(1, 0, 2, 3).contiguous()         # (L, z, h, w)
        return lat

    @torch.no_grad()
    def decode_video(self, latent):
        """Temporal-expanding decode (inverse of encode_video).

        latent: (L, z_dim, h, w) normalized  ->  frames (T, 3, H, W) in [0,1],
        T = 1 + (L-1)*4. Decodes the whole temporal axis jointly."""
        L, z, h, w = latent.shape
        lat = latent.permute(1, 0, 2, 3).unsqueeze(0).to(self.device, self.dtype)  # (1,z,L,h,w)
        lat = lat * self.latents_std + self.latents_mean
        rec = self.vae.decode(lat).sample                     # (1, 3, T, H, W)
        rec = ((rec + 1.0) / 2.0).clamp(0, 1)
        return rec[0].permute(1, 0, 2, 3).contiguous()        # (T, 3, H, W)

