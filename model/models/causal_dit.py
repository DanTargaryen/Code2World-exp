"""Causal DiT for code-conditioned world modeling — BLOCK-AUTOREGRESSIVE flow.

Sequence: L latents, each with S = h*w spatial tokens (8x8 = 64) -> (B, L, S, D).
Latents are grouped into BLOCKS of `block_size` (default 3). Temporal attention is
BLOCK-CAUSAL: latent i attends latent j iff block(j) <= block(i) — within a block
bidirectional, across blocks causal. Generation is block-autoregressive: a whole
block of `block_size` latents is produced jointly per AR step, conditioned on all
earlier (clean) blocks.

Wan 2.1 temporal compression gives 1 action <-> 1 latent <-> 4 frames. The first
latent (index 0) is the INIT (encodes only frame 0), always given, never a target.
With 42 latents = 14 blocks of 3, block 0 = {init, x1, x2}.

Each DiT block applies, in order:
  1. action additive bias  (Linear(num_actions, D), broadcast over S; per latent)
     + optional tau bias    (Linear(D, D) on the timestep embedding, zero-init)
  2. spatial self-attention   (full attention within each latent, over S tokens)
  3. temporal block-causal attention (block-causal over L, per spatial position)
  4. cross-attention to code tokens
  5. FFN

Two objectives share one backbone:
  - Legacy MSE path  forward():  next-frame latent via output_proj; per-latent
    causal mask (block_size=1), no tau. Kept for the old checkpoint.
  - Block-AR flow (primary): the backbone IS the denoiser (single-stream
    Diffusion Forcing). Every latent carries its own noise level tau; the init is
    held clean (tau=1). No leak: same-block neighbours are only ever seen NOISED.
      forward_flow(z_tau, tau, action, code)    -> per-latent velocity v = z1 - eps
      forward_state(clean_latents, action, code) -> reward/done logits (tau-free)
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


def block_causal_allow(T, block_size, device):
    """(T,T) bool mask, True = latent i may attend latent j (block(j) <= block(i))."""
    blk = (torch.arange(T, device=device) // block_size)
    return blk[None, :] <= blk[:, None]    # (T_query, T_key)


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


class TemporalBlockAttention(nn.Module):
    """Block-causal attention across L latents, independently per spatial position.

    `allow` is a (L, L) bool mask (True = attend); block_size=1 recovers plain
    causal. Bidirectional within a block, causal across blocks.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, allow):
        # x: (B, T, S, D)   allow: (T, T) bool
        B, T, S, D = x.shape
        qkv = self.qkv(x).reshape(B, T, S, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(3)  # (B, T, S, nh, hd)
        q = q.permute(0, 2, 4, 1, 3)  # (B, S, nh, T, hd)
        k = k.permute(0, 2, 4, 1, 3)
        v = v.permute(0, 2, 4, 1, 3)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[None, None, None])
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
        attn_mask = None
        if code_mask is not None:
            attn_mask = code_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)  # (B, nh, T*S, hd)
        out = out.transpose(1, 2).reshape(B, T, S, D)
        return self.proj(out)


class ActionWindowCrossAttention(nn.Module):
    """Window-conditioned action injection (Matrix-Game style, adapted).

    Matrix-Game gives one action per RAW frame and pulls a window of
    vae_ratio*windows_size raw frames per latent. Code2world is already
    time-compressed offline (1 latent <-> 1 action <-> 4 frames), so the window
    is taken directly in LATENT units: for latent i we gather actions
    [i-(W-1) .. i] (left-padded with zeros = "no history"), giving each latent
    the current action plus the previous W-1. That W-latent window equals the
    3-latent (=12 raw-frame) context Matrix-Game uses at W=3.

    The windowed action features become per-latent K/V; the visual tokens are the
    query. Attention runs over the temporal axis per spatial position, masked by
    the SAME block-causal `allow` used by temporal self-attention (Matrix-Game is
    non-causal; we keep it causal to avoid AR leakage). The output projection is
    zero-init so an untrained block reduces to the identity (stable start, on par
    with the additive-bias path).
    """
    def __init__(self, dim, num_heads, num_actions, window=3, act_hidden=128):
        super().__init__()
        self.num_heads = num_heads
        self.window = window
        self.act_embed = nn.Sequential(
            nn.Linear(num_actions, act_hidden, bias=True), nn.SiLU(),
            nn.Linear(act_hidden, act_hidden, bias=True))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(act_hidden * window, dim * 2)
        self.q_norm = nn.LayerNorm(dim // num_heads)
        self.k_norm = nn.LayerNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    def forward(self, x, action_onehot, allow):
        # x: (B,T,S,D)  action_onehot: (B,T,A)  allow: (T,T) bool
        B, T, S, D = x.shape
        nh, hd = self.num_heads, D // self.num_heads
        a = self.act_embed(action_onehot)                       # (B,T,H)
        W = self.window
        # left-pad W-1 with zeros, then sliding window concat -> (B,T,H*W)
        a_pad = F.pad(a, (0, 0, W - 1, 0))                      # (B, T+W-1, H)
        win = torch.stack([a_pad[:, i:i + T] for i in range(W)], dim=2)  # (B,T,W,H)
        win = win.reshape(B, T, -1)                            # (B,T,H*W)
        kv = self.kv(win).reshape(B, T, 2, nh, hd)
        k, v = kv.unbind(2)                                    # each (B,T,nh,hd)
        q = self.q(x).reshape(B, T, S, nh, hd)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # attend over time per spatial position: q (B,S,nh,T,hd) x k (B,S,nh,T,hd)
        q = q.permute(0, 2, 3, 1, 4)                           # (B,S,nh,T,hd)
        k = k.permute(0, 2, 1, 3).unsqueeze(1).expand(B, S, nh, T, hd)
        v = v.permute(0, 2, 1, 3).unsqueeze(1).expand(B, S, nh, T, hd)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allow[None, None, None])
        out = out.permute(0, 3, 1, 2, 4).reshape(B, T, S, D)   # (B,T,S,D)
        return self.proj(out)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, num_actions, mlp_ratio=4.0,
                 action_mode="bias", action_window=3):
        super().__init__()
        self.action_mode = action_mode
        if action_mode == "bias":
            self.action_proj = nn.Linear(num_actions, dim, bias=False)
        elif action_mode == "crossattn":
            self.ln_act = nn.LayerNorm(dim)
            self.action_cross = ActionWindowCrossAttention(
                dim, num_heads, num_actions, window=action_window)
        else:
            raise ValueError(f"unknown action_mode {action_mode}")
        self.tau_proj = nn.Linear(dim, dim)            # injects timestep embedding
        nn.init.zeros_(self.tau_proj.weight); nn.init.zeros_(self.tau_proj.bias)
        self.ln_sp = nn.LayerNorm(dim)
        self.spatial = SpatialSelfAttention(dim, num_heads)
        self.ln_tp = nn.LayerNorm(dim)
        self.temporal = TemporalBlockAttention(dim, num_heads)
        self.ln_cr = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads)
        self.ln_ff = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, action_onehot, code, allow, code_mask=None, t_emb=None):
        # action_onehot: (B, T, num_actions); t_emb: (B, T, D) or None
        if self.action_mode == "bias":
            # action bias (+ tau) injected up front, before all attention
            bias = self.action_proj(action_onehot)
            if t_emb is not None:
                bias = bias + self.tau_proj(t_emb)
            x = x + bias.unsqueeze(2)
            x = x + self.spatial(self.ln_sp(x))
            x = x + self.temporal(self.ln_tp(x), allow)
            x = x + self.cross(self.ln_cr(x), code, code_mask)
            x = x + self.ff(self.ln_ff(x))
        else:  # crossattn: action injected AFTER cross-attn, before FFN (Matrix-Game
                # position, model.py cross_attn_ffn). tau stays up front so the noise
                # level still conditions self-/temporal-attention.
            if t_emb is not None:
                x = x + self.tau_proj(t_emb).unsqueeze(2)
            x = x + self.spatial(self.ln_sp(x))
            x = x + self.temporal(self.ln_tp(x), allow)
            x = x + self.cross(self.ln_cr(x), code, code_mask)
            x = x + self.action_cross(self.ln_act(x), action_onehot, allow)
            x = x + self.ff(self.ln_ff(x))
        return x


class CausalDiT(nn.Module):
    def __init__(self, latent_dim=16, embed_dim=512, num_layers=12, num_heads=8,
                 num_actions=15, spatial_size=8, max_frames=64, code_dim=896,
                 block_size=3, action_mode="bias", action_window=3):
        super().__init__()
        self.spatial_size = spatial_size
        S = spatial_size * spatial_size
        self.S = S
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.action_mode = action_mode
        self.action_window = action_window

        # project frozen code-encoder embeds (e.g. Qwen, 896-d) into model width
        self.code_proj = (nn.Linear(code_dim, embed_dim)
                          if code_dim != embed_dim else nn.Identity())
        self.code_norm = nn.LayerNorm(embed_dim)

        self.input_proj = nn.Linear(latent_dim, embed_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, S, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, max_frames, 1, embed_dim))

        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, num_actions,
                     action_mode=action_mode, action_window=action_window)
            for _ in range(num_layers)])
        self.ln_out = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, latent_dim)     # legacy MSE path
        self.flow_out = nn.Linear(embed_dim, latent_dim)        # velocity head
        nn.init.zeros_(self.flow_out.weight); nn.init.zeros_(self.flow_out.bias)

        # timestep MLP for tau (shared across DiT blocks)
        self.tau_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        # auxiliary heads (per-latent: pool spatial tokens of the CLEAN pass)
        self.reward_head = nn.Linear(embed_dim, 3)   # -1 / 0 / +1
        self.done_head = nn.Linear(embed_dim, 2)

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)

    def _run_backbone(self, latents, action_onehot, code, code_mask, block_size, t_emb):
        """latents: (B,L,C,h,w) -> features (B,L,S,D). action_onehot: (B,L,A)."""
        B, L, C, h, w = latents.shape
        S = h * w
        code = self.code_norm(self.code_proj(code))            # (B, N, D)
        x = latents.permute(0, 1, 3, 4, 2).reshape(B, L, S, C)  # (B,L,S,C)
        x = self.input_proj(x)                                  # (B,L,S,D)
        x = x + self.spatial_pos + self.temporal_pos[:, :L]
        allow = block_causal_allow(L, block_size, x.device)     # (L,L) bool
        for blk in self.blocks:
            x = blk(x, action_onehot, code, allow, code_mask, t_emb)
        return self.ln_out(x)                                   # (B,L,S,D)

    def forward(self, latents, action_onehot, code, code_mask=None):
        """LEGACY MSE path: per-latent causal, predict NEXT-frame latent.
        returns pred (B,T,C,h,w), reward_logits (B,T,3), done_logits (B,T,2)."""
        B, T, C, h, w = latents.shape
        x = self._run_backbone(latents, action_onehot, code, code_mask, 1, None)
        pred = self.output_proj(x).reshape(B, T, h, w, C).permute(0, 1, 4, 2, 3)
        pooled = x.mean(dim=2)
        return pred, self.reward_head(pooled), self.done_head(pooled)

    def forward_flow(self, z_tau, tau, action_onehot, code, code_mask=None):
        """Block-AR flow: predict per-latent velocity v = z1 - eps.
        z_tau: (B,L,C,h,w) noised latents (init held clean)
        tau:   (B,L) in [0,1]   action_onehot: (B,L,A) (pos 0 = null action)
        returns velocity (B,L,C,h,w)."""
        B, L, C, h, w = z_tau.shape
        t_emb = self.tau_mlp(timestep_embedding(tau, self.embed_dim))     # (B,L,D)
        x = self._run_backbone(z_tau, action_onehot, code, code_mask, self.block_size, t_emb)
        return self.flow_out(x).reshape(B, L, h, w, C).permute(0, 1, 4, 2, 3)

    def forward_state(self, latents, action_onehot, code, code_mask=None):
        """reward/done from a CLEAN, tau-free pass (per-latent causal).
        returns reward_logits (B,L,3), done_logits (B,L,2)."""
        x = self._run_backbone(latents, action_onehot, code, code_mask, 1, None)
        pooled = x.mean(dim=2)
        return self.reward_head(pooled), self.done_head(pooled)


@torch.no_grad()
def block_ar_generate(model, init_latent, actions, code, num_actions, dev,
                      block_size, flow_steps, code_mask=None):
    """Block-autoregressive rollout for the flow model.

    init_latent: (1, 1, C, h, w) clean INIT latent (index 0).
    actions:     length-K sequence; produces K latents -> total L = K+1 (<= max_frames).
    Each AR step denoises the next block of latents (block 0 emits block_size-1
    because the init fills one slot; later blocks emit block_size) jointly via Euler
    integration from noise, with the clean history held at tau=1.
    Returns the full latent sequence (1, K+1, C, h, w) including the init.
    """
    L_target = len(actions) + 1
    hist = init_latent.clone()                                    # (1, 1, C, h, w) clean
    Cshape = init_latent.shape[2:]
    dt = 1.0 / flow_steps
    while hist.shape[1] < L_target:
        cur = hist.shape[1]
        b = block_size - (cur % block_size) if (cur % block_size) else block_size
        b = min(b, L_target - cur)
        seqlen = cur + b
        a_full = torch.zeros(1, seqlen, num_actions, device=dev)
        for i in range(1, seqlen):
            a_full[0, i, int(actions[i - 1])] = 1.0
        z_block = torch.randn(1, b, *Cshape, device=dev)
        for s in range(flow_steps):
            seq = torch.cat([hist, z_block], dim=1)              # (1, seqlen, C, h, w)
            tau = torch.ones(1, seqlen, device=dev)
            tau[:, cur:] = s * dt                                 # history clean, block noised
            v = model.forward_flow(seq, tau, a_full, code, code_mask)[:, cur:cur + b]
            z_block = z_block + dt * v.float()
        hist = torch.cat([hist, z_block], dim=1)
    return hist                                                   # (1, L_target, C, h, w)
