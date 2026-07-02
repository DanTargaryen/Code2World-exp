"""Bidirectional DiT for code-conditioned world modeling — flow matching.

Sequence: L latents, each with S = h*w spatial tokens (8x8 = 64) -> (B, L, S, D).
Attention is fully BIDIRECTIONAL (non-causal): every latent attends every other.
The whole clip is denoised jointly; there is no autoregressive rollout.

Wan 2.1 temporal compression gives 1 action <-> 1 latent <-> 4 frames. The first
latent (index 0) is the INIT (encodes only frame 0), always given clean, never a
target. Generation denoises indices 1..L-1 from noise conditioned on the init,
the per-latent actions and the code.

Each DiT block applies, in order:
  1. tau bias         (timestep embedding -> per-latent additive bias, zero-init)
  2. spatial self-attention   (full attention within each latent, over S tokens)
  3. temporal self-attention  (full attention across L latents, per spatial pos)
  4. cross-attention to code tokens
  5. action window cross-attention (Matrix-Game style: windowed actions as K/V)
  6. FFN

The backbone IS the denoiser (single-stream flow matching):
  forward_flow(z_tau, tau, action, code)     -> per-latent velocity v = z1 - eps
  forward_state(clean_latents, action, code)  -> reward/done logits (tau-free)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000.0):
    """Sinusoidal embedding for tau in [0,1]. t: (...) -> (..., dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float().unsqueeze(-1) * 1000.0 * freqs   # scale [0,1] -> [0,1000]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class RMSNorm(nn.Module):
    """RMSNorm for QK normalisation (Matrix-Game style)."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * self.weight


def rope_freqs_1d(T, head_dim, device, theta=256.0):
    """1D rotary freqs over a length-T axis. Returns (cos, sin) each (T, head_dim)."""
    half = head_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(T, device=device).float()
    ang = torch.outer(pos, inv)                       # (T, half)
    cos = torch.cos(ang).repeat(1, 2)                 # (T, head_dim)
    sin = torch.sin(ang).repeat(1, 2)
    return cos, sin


def apply_rope(x, cos, sin):
    """Apply RoPE to x (..., T, head_dim) with cos/sin (T, head_dim). rotate_half style."""
    hd = x.shape[-1]
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    shape = [1] * (x.dim() - 2) + list(cos.shape)     # broadcast over leading dims
    return x * cos.view(*shape) + rot * sin.view(*shape)


class SpatialSelfAttention(nn.Module):
    """Full attention among the S spatial tokens within each latent."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, S, D)
        B, T, S, D = x.shape
        qkv = self.qkv(x).reshape(B, T, S, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(3)  # each (B, T, S, nh, hd)
        q = q.permute(0, 1, 3, 2, 4)  # (B, T, nh, S, hd)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)
        out = F.scaled_dot_product_attention(q, k, v)  # (B, T, nh, S, hd)
        out = out.permute(0, 1, 3, 2, 4).reshape(B, T, S, D)
        return self.proj(out)


class TemporalSelfAttention(nn.Module):
    """Full (bidirectional) attention across L latents, per spatial position."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, T, S, D)
        B, T, S, D = x.shape
        qkv = self.qkv(x).reshape(B, T, S, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(3)  # (B, T, S, nh, hd)
        q = q.permute(0, 2, 4, 1, 3)  # (B, S, nh, T, hd)
        k = k.permute(0, 2, 4, 1, 3)
        v = v.permute(0, 2, 4, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 3, 1, 4, 2).reshape(B, T, S, D)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Cross-attention from visual tokens (query) to code tokens (key/value)."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, code, code_mask=None):
        # x: (B, T, S, D)   code: (B, N, D)   code_mask: (B, N) bool, True = real token
        B, T, S, D = x.shape
        N = code.shape[1]
        nh, hd = self.num_heads, D // self.num_heads
        q = self.q(x).reshape(B, T * S, nh, hd).transpose(1, 2)        # (B, nh, T*S, hd)
        kv = self.kv(code).reshape(B, N, 2, nh, hd)
        k, v = kv.unbind(2)
        k = k.transpose(1, 2)  # (B, nh, N, hd)
        v = v.transpose(1, 2)
        attn_mask = code_mask[:, None, None, :] if code_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, nh, T*S, hd)
        out = out.transpose(1, 2).reshape(B, T, S, D)
        return self.proj(out)


class ActionWindowCrossAttention(nn.Module):
    """Window-conditioned action injection (Matrix-Game style, adapted).

    Code2world is time-compressed offline (1 latent <-> 1 action <-> 4 frames), so
    the action window is taken directly in LATENT units: for latent i we gather
    actions [i-(W-1) .. i], giving each latent the current action plus the previous
    W-1. The windowed features become per-latent K/V; visual tokens are the query,
    attending over the temporal axis per spatial position (bidirectional).

    Matrix-Game alignment: QK use RMSNorm, q/k carry 1D RoPE over the temporal axis
    (theta=256), and the left window pad repeats the FIRST action (not zeros).
    Output projection is zero-init so an untrained block reduces to the identity.
    """
    def __init__(self, dim, num_heads, num_actions, window=3, act_hidden=128,
                 rope_theta=256.0):
        super().__init__()
        self.num_heads = num_heads
        self.window = window
        self.rope_theta = rope_theta
        self.act_embed = nn.Sequential(
            nn.Linear(num_actions, act_hidden, bias=True), nn.SiLU(),
            nn.Linear(act_hidden, act_hidden, bias=True))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(act_hidden * window, dim * 2)
        self.q_norm = RMSNorm(dim // num_heads)
        self.k_norm = RMSNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, action_onehot):
        # x: (B,T,S,D)  action_onehot: (B,T,A)
        B, T, S, D = x.shape
        nh, hd = self.num_heads, D // self.num_heads
        W = self.window
        a = self.act_embed(action_onehot)                       # (B,T,H)
        # left-pad W-1 by REPEATING the first action, then sliding window concat
        a_pad = torch.cat([a[:, :1].expand(B, W - 1, a.shape[-1]), a], dim=1)  # (B,T+W-1,H)
        win = torch.stack([a_pad[:, i:i + T] for i in range(W)], dim=2)  # (B,T,W,H)
        win = win.reshape(B, T, -1)                            # (B,T,H*W)
        kv = self.kv(win).reshape(B, T, 2, nh, hd)
        k, v = kv.unbind(2)                                    # each (B,T,nh,hd)
        q = self.q_norm(self.q(x).reshape(B, T, S, nh, hd))
        k = self.k_norm(k)
        # RoPE over the temporal axis (per latent position t in [0,T))
        cos, sin = rope_freqs_1d(T, hd, x.device, self.rope_theta)
        q = apply_rope(q.permute(0, 2, 3, 1, 4), cos, sin)     # (B,S,nh,T,hd)
        k = apply_rope(k.permute(0, 2, 1, 3), cos, sin)        # (B,nh,T,hd)
        k = k.unsqueeze(1).expand(B, S, nh, T, hd)
        v = v.permute(0, 2, 1, 3).unsqueeze(1).expand(B, S, nh, T, hd)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 3, 1, 2, 4).reshape(B, T, S, D)   # (B,T,S,D)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, num_actions, action_window=3, mlp_ratio=4.0):
        super().__init__()
        self.tau_proj = nn.Linear(dim, dim)            # injects timestep embedding
        nn.init.zeros_(self.tau_proj.weight); nn.init.zeros_(self.tau_proj.bias)
        self.ln_sp = nn.LayerNorm(dim)
        self.spatial = SpatialSelfAttention(dim, num_heads)
        self.ln_tp = nn.LayerNorm(dim)
        self.temporal = TemporalSelfAttention(dim, num_heads)
        self.ln_cr = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads)
        self.ln_act = nn.LayerNorm(dim)
        self.action_cross = ActionWindowCrossAttention(dim, num_heads, num_actions,
                                                       window=action_window)
        self.ln_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, action_onehot, code, code_mask, t_emb):
        # x: (B,T,S,D)  action_onehot: (B,T,A)  t_emb: (B,T,D)
        x = x + self.tau_proj(t_emb).unsqueeze(2)                 # noise-level conditioning
        x = x + self.spatial(self.ln_sp(x))
        x = x + self.temporal(self.ln_tp(x))
        x = x + self.cross(self.ln_cr(x), code, code_mask)
        x = x + self.action_cross(self.ln_act(x), action_onehot)  # Matrix-Game position
        x = x + self.ff(self.ln_ff(x))
        return x


class BidirDiT(nn.Module):
    def __init__(self, latent_dim=16, embed_dim=512, num_layers=12, num_heads=8,
                 num_actions=6, spatial_size=8, max_frames=64, code_dim=896,
                 action_window=3):
        super().__init__()
        self.spatial_size = spatial_size
        S = spatial_size * spatial_size
        self.S = S
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.action_window = action_window

        # project frozen code-encoder embeds (e.g. Qwen, 896-d) into model width
        self.code_proj = (nn.Linear(code_dim, embed_dim)
                          if code_dim != embed_dim else nn.Identity())
        self.code_norm = nn.LayerNorm(embed_dim)

        self.input_proj = nn.Linear(latent_dim, embed_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, S, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, 1, embed_dim))

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, num_actions, action_window=action_window)
            for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)
        self.flow_out = nn.Linear(embed_dim, latent_dim)        # velocity head
        nn.init.zeros_(self.flow_out.weight); nn.init.zeros_(self.flow_out.bias)

        # timestep MLP for tau (shared across DiT blocks)
        self.tau_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        # auxiliary heads (per-latent: pool spatial tokens of the clean pass)
        self.reward_head = nn.Linear(embed_dim, 3)   # -1 / 0 / +1
        self.done_head = nn.Linear(embed_dim, 2)

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def _run_backbone(self, latents, action_onehot, code, code_mask, t_emb):
        """latents: (B,L,C,h,w) -> features (B,L,S,D). action_onehot: (B,L,A)."""
        B, L, C, h, w = latents.shape
        S = h * w
        code = self.code_norm(self.code_proj(code))            # (B, N, D)
        x = latents.permute(0, 1, 3, 4, 2).reshape(B, L, S, C)  # (B,L,S,C)
        x = self.input_proj(x)                                  # (B,L,S,D)
        x = x + self.spatial_pos + self.temporal_pos[:, :L]
        for blk in self.blocks:
            x = blk(x, action_onehot, code, code_mask, t_emb)
        return self.ln_out(x)                                   # (B,L,S,D)

    def forward_flow(self, z_tau, tau, action_onehot, code, code_mask=None):
        """Predict per-latent velocity v = z1 - eps.
        z_tau: (B,L,C,h,w) noised latents (init held clean)
        tau:   (B,L) in [0,1]   action_onehot: (B,L,A) (pos 0 = null action)
        returns velocity (B,L,C,h,w)."""
        B, L, C, h, w = z_tau.shape
        t_emb = self.tau_mlp(timestep_embedding(tau, self.embed_dim))     # (B,L,D)
        x = self._run_backbone(z_tau, action_onehot, code, code_mask, t_emb)
        return self.flow_out(x).reshape(B, L, h, w, C).permute(0, 1, 4, 2, 3)

    def forward_state(self, latents, action_onehot, code, code_mask=None):
        """reward/done from a CLEAN, tau-free pass.
        returns reward_logits (B,L,3), done_logits (B,L,2)."""
        t_emb = self.tau_mlp(timestep_embedding(torch.ones(*latents.shape[:2],
                                                           device=latents.device),
                                                self.embed_dim))
        x = self._run_backbone(latents, action_onehot, code, code_mask, t_emb)
        pooled = x.mean(dim=2)
        return self.reward_head(pooled), self.done_head(pooled)


@torch.no_grad()
def full_seq_generate(model, init_latent, actions, code, num_actions, dev,
                      flow_steps, code_mask=None):
    """Bidirectional generation: denoise the WHOLE sequence jointly.

    The init (index 0) is held clean; every other latent starts from noise and is
    integrated with one shared tau via Euler steps. Length is fixed by len(actions)
    (= training window), NOT arbitrary.

    init_latent: (1, 1, C, h, w) clean INIT. actions: length-K -> total L = K+1.
    Returns (1, K+1, C, h, w) including the init.
    """
    L = len(actions) + 1
    Cshape = init_latent.shape[2:]
    a_full = torch.zeros(1, L, num_actions, device=dev)
    for i in range(1, L):
        a_full[0, i, int(actions[i - 1])] = 1.0
    z = torch.randn(1, L, *Cshape, device=dev)
    z[:, :1] = init_latent                                         # slot 0 = clean init
    dt = 1.0 / flow_steps
    for s in range(flow_steps):
        tau = torch.full((1, L), s * dt, device=dev)
        tau[:, 0] = 1.0                                            # init always clean
        v = model.forward_flow(z, tau, a_full, code, code_mask)
        z[:, 1:] = z[:, 1:] + dt * v[:, 1:].float()               # integrate non-init only
    return z                                                      # (1, L, C, h, w)
