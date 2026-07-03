# Code-Conditioned Causal DiT World Model — 训练计划

> 基于 Procgen CoinRun (64×64)，Wan 2.1 VAE (8× spatial, 16ch) + Causal DiT + Qwen2.5-0.5B Code Encoder，teacher forcing 直接训练 AR 视频世界模型

## 1. 目标

训练一个能「读懂游戏规则描述 + 接收玩家动作 → 自回归预测下一帧画面」的神经游戏引擎原型。

核心验证指标：
- **Action following**: 给定 action，生成帧是否反映了对应动作
- **Code sensitivity**: 改变规则描述后，预测帧是否跟着变（反事实测试）
- **Rollout stability**: 自回归生成 50+ 步后画面是否仍合理

## 2. 数据源

**OpenAI Procgen — CoinRun** (`github.com/openai/procgen`)

| 属性 | 值 |
|------|-----|
| 游戏 | CoinRun（2D 横版平台跳跃，程序化生成关卡） |
| 原始分辨率 | 64×64 RGB |
| 训练分辨率 | 64×64 RGB（Procgen 原始输出，不下采样） |
| 动作空间 | Discrete(15)，CoinRun 主要用移动+跳跃 |
| 数据量目标 | ~1M transitions（≈8K episodes × 128 steps） |
| 采集策略 | 随机策略 |

## 3. 整体架构

```
Procgen 64×64 → Wan 2.1 VAE encode (8× spatial) → 8×8×16 continuous latent
                                                                      ↓
    Qwen2.5-0.5B (冻结) ← coinrun.cpp ──→ code tokens ──→ cross-attention ──┐
                                                                           ↓
    action (Discrete 15) ──→ Linear → additive bias ──→ Causal DiT ──→ predict next latent
                                                           ↑                    ↓
                                                     block-causal          MSE loss
                                                     self-attention     (+ reward/done)
                                                                            ↓
                                              Wan 2.1 VAE decode ← predicted latent → 64×64 frame
```

**关键设计决策：**
- **Wan 2.1 VAE（非 2.2）**：2.1 的空间压缩 = 8×，64×64 → 8×8×16 = 64 spatial positions × 16 channels
- **latent channels = 16**：Wan VAE 统一输出 16 通道 latent
- **不用 VQ-VAE**：直接用 Wan 的连续 VAE，省掉 codebook 训练
- **不用 diffusion**：MSE 直接回归下一帧 latent（8×8 latent 空间仍然低维，不需要扩散）
- **不用 bidir→AR 转换**：从头 causal 训练，teacher forcing
- **不用大 LLM**：Qwen2.5-0.5B 冻结做 text encoder，代码理解够、显存可控

## 4. 组件详解

### 4.1 VAE — Wan 2.1 3D Causal VAE（冻结/微调）

| 属性 | 值 |
|------|-----|
| 来源 | Wan 2.1 预训练 VAE（A14B 版本） |
| 空间压缩 | **8×** (64×64 → 8×8) |
| 时间压缩 | 4× (单帧模式下不涉及) |
| latent 维度 | **8×8×16** (64 spatial positions × 16 channels) |
| latent 类型 | 连续 (非 VQ 离散) |
| 训练策略 | 先冻结测试重建质量；如果 Procgen 像素风重建差，则 fine-tune |

**为什么用 2.1 而非 2.2**：Wan 2.2 的 VAE 空间压缩为 16× 或 32×（含 patchify），64×64 帧会被压成 4×4 或 2×2 latent，信息损失较大。Wan 2.1 的 8× 空间压缩 → 8×8 latent 保留更充分的空间信息。

**需先验证**：Wan 2.1 VAE 对 Procgen 像素风格帧的 encode-decode 重建 PSNR。如果 < 20dB 或关键物体（玩家、金币、平台）不可辨，则改为从头训小 VQ-VAE（备选方案，详见附录）。

### 4.2 Code Encoder — Qwen2.5-0.5B（冻结）

| 属性 | 值 |
|------|-----|
| 模型 | Qwen2.5-0.5B (纯文本) |
| 策略 | 全程冻结，只取最后一层 hidden states |
| 输入 | **`coinrun.cpp` 完整 C++ 源码**（524 行，~2000-3000 tokens） |
| 输出 | code_tokens: (N, 896)，N = token 数，896 = Qwen hidden dim |
| 投影 | Linear(896, 512) 把 Qwen dim 投影到 DiT embed_dim |

**为什么直接喂源码而非 JSON 规则描述：**
- `coinrun.cpp` 是 `rules.json` 的**超集**——包含精确数值（`gravity=0.2f`、`maxspeed=0.5`、`max_jump=1.5`）、完整物理逻辑（`update_agent_velocity()`）、碰撞判定、关卡生成算法、动作语义（`action_vx = move_action / 3 - 1`）
- 直接喂可执行代码是博客核心主张——"code grounds rules"
- 反事实测试时直接改源码数值（如 `gravity = 0.2f → 0.5f`），比改 JSON 更 grounded
- 源码来自 Procgen 仓库 `procgen/src/games/coinrun.cpp`，现成可用

**源码中的关键参数（反事实测试可修改的位置）：**
- 物理：`gravity = 0.2f` (L419), `max_jump = 1.5` (L420), `maxspeed = 0.5` (L422), `air_control = 0.15f` (L421)
- 奖励：`GOAL_REWARD = 10.0f` (L12)
- 碰撞：`handle_agent_collision()` — 碰敌人/锯子 → done (L123-131)
- 动作映射：`set_action_xy()` — `action_vx = move_action / 3 - 1` (L452-453)

### 4.3 Causal DiT — 主模型（从头训练）

#### 注意力模式：Block-Causal

```
帧序列: [f0: 64 spatial tokens] [f1: 64 tokens] ... [fT: 64 tokens]

帧内: full self-attention (同一帧的 16 个 spatial token 互相看)
帧间: causal (frame t 只能 attend to frame 0..t-1)
```

训练时所有前序帧用 GT latent（teacher forcing），推理时自回归用预测 latent。

#### DiT Block 结构（分离式，每层一个 block）

```
┌─────────────────────────────────────────────┐
│ DiT Block ℓ                                 │
│                                             │
│  x += action_bias_ℓ(action)                │  ← action additive bias
│  x += SpatialSelfAttn(LN(x))              │  ← 帧内 full attention
│  x += TemporalCausalAttn(LN(x))           │  ← 帧间 causal attention
│  x += CrossAttn(LN(x), code_tokens)       │  ← code condition
│  x += FFN(LN(x))                          │  ← feed-forward
│                                             │
└─────────────────────────────────────────────┘
```

- **Spatial Self-Attention**: 每帧 64 个 spatial token 之间 full attention，学习空间结构
- **Temporal Causal Attention**: 同一 spatial 位置跨帧 causal attend，学习时间动态
- **Cross-Attention**: query=视觉 tokens, K/V=code_tokens，注入规则条件
- **Action Bias**: `Linear(15, embed_dim, bias=False)` 每层独立，broadcast 到帧内 64 tokens

选分离式而非合并式：与 Wan 架构一致，后续可用 Wan 预训练权重 warm start。

#### 三路输入对齐（全部投影到 512 维）

三路输入原始维度不同，各自通过一个可训练的 Linear 投影到 DiT 的 embed_dim=512，在同一 embedding space 里交互：

```
VAE latent (8×8×16)    → Linear(16, 512)  → 512-dim spatial tokens → DiT self-attention 的输入
Action (15-dim onehot) → Linear(15, 512)  → 512-dim bias          → additive bias 加到 tokens 上
Code text (N, 896)     → Linear(896, 512) → 512-dim code tokens   → DiT cross-attention 的 K/V
```

| 输入 | 原始维度 | 投影 | 目标维度 | 注入方式 |
|------|---------|------|---------|---------|
| VAE latent | 16 channels | `Linear(16, 512)` 可训练 | 512 | 成为 self-attention 的 token |
| Action | 15 (one-hot) | `Linear(15, 512)` **每层独立**，可训练 | 512 | additive bias 加到 token 上 |
| Code text | 896 (Qwen2.5 hidden) | `Linear(896, 512)` 可训练 | 512 | cross-attention 的 K/V |

- VAE 和 Qwen 本身**冻结**，只有三个 projection 层参与训练
- Action 的 Linear **每个 DiT layer 独立一个**（共 12 个），让不同层从 action 中提取不同层次的信息
- 输出时对称投影回去：`Linear(512, 16)` 把 DiT output 映射回 VAE latent 空间的 16 channels

#### 模型规模

| 参数 | 值 | 说明 |
|------|-----|------|
| embed_dim | 512 | 通过 Linear(16, 512) 从 latent 投影上来 |
| num_layers | 12 | 中等深度 |
| num_heads | 8 | head_dim=64 |
| spatial_tokens | 64 (8×8) | Wan 2.1 VAE 8× 压缩，64×64 → 8×8 |
| latent_channels | 16 | Wan VAE 输出固定 16 通道 |
| input_proj | Linear(16, 512) | latent 16ch → DiT embed_dim |
| output_proj | Linear(512, 16) | DiT embed_dim → latent 16ch (预测头) |
| temporal_frames | 16 | 上下文窗口（64×64 下序列较长，先用 16 帧） |
| 序列长度 | 64×16 = 1024 tokens | spatial 64 × temporal 16 |
| code_tokens | 可变 (源码长度) | Qwen2.5 输出序列 |
| 总参数量 | ~80-100M | 适配 8 GPU DDP（64×64 显存约 4× 于 32×32，batch 相应调小） |

### 4.4 损失函数

```
L = L_latent + λ₁·L_reward + λ₂·L_done
```

- `L_latent`: **MSE** between predicted latent and GT latent（主要损失）
  - 8×8 latent 空间仍然低维，MSE 不会产生像素级模糊
  - 如果后续发现多模态分布问题，可升级为 1-step flow matching
- `L_reward`: 奖励预测 cross-entropy (3 类: -1/0/+1)
- `L_done`: 终止预测 cross-entropy (2 类)

### 4.5 训练配置

| 参数 | 值 |
|------|-----|
| batch_size | 64 × N_GPU |
| optimizer | AdamW, lr=1e-4, weight_decay=0.01 |
| schedule | cosine decay, warmup 1K steps |
| grad_clip | 1.0 |
| 多卡 | PyTorch DDP |
| 训练步数 | ~100K steps |
| 预计耗时 | 4-8 小时 (4×GPU) |

## 5. Code Condition — 直接喂 C++ 源码

训练时将 `procgen/src/games/coinrun.cpp` 完整源码（524 行）作为 code condition 喂给 Qwen2.5-0.5B 编码。不再需要人工撰写 `rules.json`。

反事实测试时，直接修改源码中的具体数值或逻辑后重新编码，例如：
- 改 `gravity = 0.2f` → `gravity = 0.5f`
- 改 `GOAL_REWARD = 10.0f` → `GOAL_REWARD = 1.0f`
- 交换 `action_vx = move_action / 3 - 1` 的正负号（反转左右方向）

## 6. Code Sensitivity 验证（反事实测试）

| 扰动类型 | 做法 | 期望结果 |
|----------|------|---------|
| 改动作语义 | 源码中 `action_vx = move_action / 3 - 1` 改为 `1 - move_action / 3`（反转左右） | 同一 action 下移动方向反转 |
| 改物理参数 | `gravity = 0.2f` → `0.5f` | 下落更快，跳跃弧线变低 |
| 改奖励 | `GOAL_REWARD = 10.0f` → `1.0f` | 不直接影响画面，但验证模型是否读了这个值 |
| 改碰撞规则 | 注释掉 `if (obj->type == ENEMY) { step_data.done = true; }` | 碰敌人不再死亡 |
| 空白 code | zero out cross-attention output | 预测退化，证明 code 有信息增益 |

## 7. 文件结构

```
workspace/Code2world/model/
├── configs/
│   └── coinrun_64.yaml           # 训练超参配置
├── data/
│   ├── collect.py                # Procgen 数据采集 (随机策略, 64×64)
│   └── dataset.py                # PyTorch Dataset (episodes → frame sequences)
├── models/
│   ├── vae.py                    # Wan VAE 加载/冻结/微调 wrapper
│   ├── text_encoder.py           # Qwen2.5-0.5B 冻结 wrapper + projection (编码 C++ 源码)
│   ├── causal_dit.py             # Causal DiT (spatial + temporal causal + cross-attn)
│   ├── action_module.py          # Action additive bias (per-layer Linear)
│   └── world_model.py            # 组合: VAE + DiT + TextEncoder + ActionModule
├── train.py                      # 训练脚本 (DDP 多卡, teacher forcing)
├── evaluate.py                   # 评估: action following + rollout + code sensitivity
├── rules/
│   └── coinrun.cpp               # 直接从 procgen/src/games/coinrun.cpp 拷贝（反事实时修改此文件）
└── README.md
```

## 8. 实施步骤

| 阶段 | 内容 | 预计耗时 | 依赖 |
|------|------|---------|------|
| Step 1 | 采集 Procgen CoinRun 数据 (~1M transitions, 64×64) | 30s (CPU) ✅ 已完成 | procgen |
| Step 2 | 验证 Wan 2.1 VAE 重建质量 (encode→decode Procgen 帧) | ✅ 已完成 PSNR 29.5dB | Wan 2.1 VAE 权重 |
| Step 2b | ~~(如果重建差) 训练小 VQ-VAE~~ 不需要，VAE 直接冻结用 | — | — |
| Step 3 | 准备 CoinRun C++ 源码 + code 变体（多游戏/参数变体） | 1h | procgen 源码可编译 ✅ |
| Step 4 | 训练 Causal DiT World Model (DDP) | 4-8h (4×GPU) | Step 1-3 |
| Step 5 | 评估 (action following + rollout + code sensitivity) | 1h | Step 4 |
| **总计** | | **~1-2 天** | |

## 9. 数据源:Procgen CoinRun(单一路线）

> 早期曾有 `data_gen` 的 three.js 平替玩具(`coin_collection` + playwright 渲染),
> 现已删除。code condition 统一走原版 Procgen(`workspace/procgen`)CoinRun,
> 与模型实际训练/评估的数据源一致,信号最纯。

| 环节 | 实现 |
|------|------|
| 帧/action/reward | Procgen CoinRun(`coinrun.cpp` 确定性模拟器) |
| code condition | `coinrun.cpp` 源码(经 Qwen 编码,真实可执行代码) |
| 规则扰动 | code perturbation(改源码机制 → code sensitivity) |
| 模型评估 | `eval_code_sensitivity.py` / `rollout_*.py` |

## 10. 关键依赖

```
pip install procgen torch torchvision einops wandb
# Wan VAE: 从 Wan 2.1 仓库加载预训练权重（8× spatial, 16ch）
# Qwen2.5-0.5B: transformers 库加载
```

## 11. 核心参考

- **IRIS** (`github.com/eloialonso/iris`): VQ-VAE + AR Transformer, ~850 行核心代码，参考 world_model.py 的序列构建和 loss
- **ReactiveGWM** (`arxiv.org/abs/2605.15256`): Wan 底座 + action additive bias + cross-attn NPC prompt，直接复用 action 注入和 code 注入设计
- **Wan 2.1** (`github.com/Wan-Video/Wan2.1`): 3D Causal VAE (**8× spatial, 16ch latent**) + DiT，参考 VAE 和 DiT block 结构。用 2.1 的 VAE 而非 2.2（2.2 压缩率 16×/32×，64×64 帧只剩 4×4 或 2×2）

## 附录 A：Wan VAE 版本对比

| Wan VAE 版本 | 空间压缩 | 64×64 帧 → latent | 可用性 |
|---|---|---|---|
| **Wan 2.1 (A14B) ✅** | **8×** | 8×8×16 = 64 positions | ✅ 适合，实测 PSNR 29.5dB |
| Wan 2.2 base | 16× | 4×4×16 = 16 positions | ⚠️ 信息损失较大 |
| Wan 2.2 + patchify | 32× | 2×2×16 = 4 positions | ❌ 太小 |

## 附录 B：备选 VAE 方案（当前不需要）

> ✅ Wan 2.1 VAE 实测 PSNR 29.5dB，重建质量优秀，直接冻结使用。以下方案仅作备档。

如果 Wan VAE 对某些游戏重建质量不达标 (PSNR < 20dB)，退回到自训 VQ-VAE：

| 组件 | 规格 |
|------|------|
| Encoder | 3 层 Conv2d(stride=2), 64×64 → 8×8 |
| Codebook | 512 entries, embed_dim=256 |
| Decoder | 对称 ConvTranspose2d |
| tokens/帧 | 64 (8×8) |
| 损失 | L1 reconstruction + commitment + LPIPS |
| 训练 | ~50K steps, batch=256, Adam lr=3e-4, 单卡 1-2h |
