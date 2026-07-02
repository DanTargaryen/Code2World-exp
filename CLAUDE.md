# Code2world — Code-Conditioned **Bidirectional** DiT 世界模型

> **本文件是 `exp/action-bidir` 分支的独立说明**(与 `main` 的 block-AR 版不同,勿混用)。
> 神经游戏引擎原型:模型**读懂游戏规则的可执行源码 + 接收玩家动作 → 生成整段画面**,
> 验证「改源码 → 生成跟着变」(code sensitivity / 反事实)。核心主张:*code grounds rules*。

## 0. 本分支定位(与 main 的根本差异)

`main` 是 **block-AR + 加性 bias action** 的因果自回归版。本分支 `exp/action-bidir`
是**彻底纯化**的另一条路线,照搬 **Matrix-Game-3** 的 action 注入 + **全局双向(非因果)** 架构:

| | `main`(block-AR) | `exp/action-bidir`(本分支) |
|---|---|---|
| 时间注意力 | block-causal(块内双向、块间因果) | **全双向(非因果)** |
| action 注入 | 每层 `Linear(9→512)` 加性 bias、只看当前帧 | **窗口 cross-attn**(Matrix-Game 式) |
| 生成方式 | `block_ar_generate` 一块块自回归、**可任意长** | `full_seq_generate` 整段一次去噪、**定长** |
| one-hot 维度 | 9(0/3/6 恒零) | **6**(紧凑,只留有效动作) |
| 代码风格 | 多开关兼容 | **无开关,单一路径** |

**取舍**:bidir 放弃了任意长 rollout(定长 = 训练窗口),换取更贴 Matrix-Game 的整段联合建模。

## 1. 目标

基于 Procgen **CoinRun**(64×64),用 **Wan 2.1 VAE + Bidir DiT + Qwen2.5-0.5B 源码编码**
训练 flow-matching 世界模型。三个核心验证指标:

- **Action following**:给定 action,生成帧是否反映对应动作
- **Code sensitivity**(最核心命题):改规则源码后,预测帧是否跟着变(反事实)
- **画质/时序一致**:整段生成是否锐利、动作连贯(bidir 无长程 AR 漂移,但受定长约束)

## 2. 架构(`models/bidir_dit.py`,`BidirDiT`,78.5M)

```
CoinRun 64×64 ──Wan2.1 VAE encode(8×space+4×time)──> latent 序列 ─┐
coinrun.cpp ──Qwen2.5-0.5B(冻结)──> code tokens ─cross-attn──────┤
action(6维one-hot,窗口) ──window cross-attn──────────────────────┤
                                                                 ↓
                                    Bidir DiT(12层,全双向) → 整段 velocity
                                                                 ↓
                                    full_seq_generate(整段Euler去噪)→ Wan VAE decode
```

- 序列 = **L 个 latent**(默认 42 = window 41 + init),每 latent **S=8×8=64 token**,宽 **D=512**。
- **1 latent ↔ 1 action ↔ 4 帧**(Wan VAE 时间压缩已离线做好);latent 0 是 **init**(编码首帧,恒 clean、不预测)。
- VAE 与 Qwen **全程冻结、离线预编码**,只训 DiT。

**DiTBlock 固定顺序**(无分支):
```
1. tau bias        (timestep embedding → per-latent 加性 bias, zero-init)
2. spatial self-attn   (帧内 64 token 全注意)
3. temporal self-attn  (跨 L latent 全双向, 按 spatial 位)
4. cross-attn to code  (visual=Q, code=K/V)
5. action window cross-attn  (Matrix-Game 位置: cross-attn 后、FFN 前)
6. FFN
```

**action window cross-attn**(`ActionWindowCrossAttention`,对齐 Matrix-Game 细节):
- 动作 embed `Linear(6→128)+SiLU+Linear`,在 **latent 粒度**滑窗 concat `W`(默认3)个
  (当前+前 W-1),做 per-latent K/V;visual token 做 Q,沿时间轴双向 attend。
- **QK 用 RMSNorm**;q/k 带 **1D RoPE**(时间轴,theta=256);左 pad **重复首帧动作**(非补零)。
- **适配点**:Code2world 已离线时间压缩,滑窗按 latent(不像 Matrix-Game 乘 vae_ratio×4)。
- proj **zero-init** → 未训练时该子层恒等,起步稳。

**训练目标**(`train_fm.py`,单流 flow matching):
- **整段统一 τ**:每样本所有非-init latent 共享一个 `τ~U(0,1)`(匹配整段同步去噪),init 恒 τ=1、排除出 loss。
- `z_τ=(1-τ)ε+τ·z1`,`v=forward_flow(...)`,`L_fm=‖v-(z1-ε)‖²`(latents 1..L-1)。
- reward/done 走 `forward_state` clean 前向,`L = L_fm + 0.1·CE(reward) + 0.1·CE(done)`。

**推理**(`full_seq_generate`):init 恒 clean,其余从纯噪声,整段 **Euler 16 步**联合积分。定长 = len(actions)+1。

## 3. Action 空间(`action_space.py`)

- Procgen CoinRun 完整 15 维,有效 0~8(9维),采集只留 **6 个**动作(剔除向下 0/3/6):
  `1=左 2=左跳 4=停 5=跳 7=右 8=右跳`。
- 数据集存**原始 Procgen id**;模型边界 `remap_to_compact` 映射到紧凑 `[0..5]`,**6 维 one-hot**(无死维)。
- 本分支恒用 compact(非开关)。

## 4. 数据集 `code2world_act6_tc`(复用,未改)

路径:`/mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc/`。7 变体配对版,
`action_repeat=4`(4K+1 帧 → Wan 时间压缩 → K+1 latent),window=41→42 latent。
详见 `main` 分支说明与 `dataset/`。**本分支只改模型/训练,数据集零改动**。

## 5. 当前进度(2026-07-02)

- ✅ **架构改造代码完成,本地自检全过,未训练**。分支 commit 链(从 main 起):
  - `47e6904` action_compact:one-hot 9→6
  - `0db8f8c`/`8ae8880`/`f001a2c` 逐步加 window cross-attn → 对齐 Matrix-Game 注入位置 → 补 RoPE/RMSNorm/首帧pad
  - `ed7f5a8` 加 bidir(非因果)架构
  - `317be31` **纯化**:删所有开关与 9 个 legacy 文件,`causal_dit.py`→`bidir_dit.py`,`CausalDiT`→`BidirDiT`
- 自检覆盖:syntax、fwd/bwd、zero-init 恒等、**双向性**(改 idx4 动作影响到过去 idx1,与因果相反)、
  `full_seq_generate`(init 保留/定长)、`compute_loss` 端到端。全尺寸 **78.5M** 参数。
- ⏳ **待办**:上 pod 训练验证(`--window 41 --batch_size 8 --steps 10000 --eval_every 500`),
  对比 bidir 画质 vs main 的 block-AR;若优则跑满 20k 出定稿。

## 6. 目录结构(本分支已精简)

```
model/
├── models/  bidir_dit.py (BidirDiT + full_seq_generate)  vae.py (Wan VAE, 冻结)
├── train_fm.py      # bidir flow 训练(无开关)
├── rollout_fm.py    # 整段生成 → decode → mp4+grid
├── action_space.py  # ACTION_SET / remap_to_compact (6维)
├── render_gt.py
└── dataset/  build_dataset collect_one dataset precompute variants
```
> 已删(纯化):MSE 版 train.py/rollout.py/train_overfit.py/rollout_custom.py、
> pixel 版 train_fm_pixel/pixel_codec/pixel_dataset/rollout_compare、旧 eval/visualize_sensitivity。
> (这些在 `main` 及其他分支仍在,需要时回那边取。)

## 7. 关键参考

- **Matrix-Game-3**(`workspace/Matrix-Game/Matrix-Game-3`,`wan/modules/action_module.py`):
  本分支 action 注入的直接蓝本(keyboard 分支:窗口动作做 K/V、img 做 Q、RoPE+RMSNorm)。
- **Wan 2.1**:3D Causal VAE(8× 空间,16ch,4× 时间)+ DiT。
- **ReactiveGWM**(arxiv 2605.15256):Wan 底座 + action 注入 + cross-attn。

## 8. 协作约定

- 默认中文、简明。训练在 K8s pod(非跳板机),`/mnt/pfs` 共享盘落日志/ckpt。
- 进 pod:`kubectl exec hzh-easygo-1n-2-master-0 -c pytorch -- bash -lc '<cmd>'`;
  后台长任务 `cd model && CUDA_VISIBLE_DEVICES=0 nohup python -u ... > xxx.log 2>&1 &`。
- **筛选阶段 10k steps**(后半 eval 躺平、边际低);确认最优再跑满 20k。
- 访问 GitHub/HF 先设代理 `export http_proxy=http://192.168.48.17:18000`(https 同)。

### 训练/推理命令(本分支)

```bash
# 训练(整段 bidir flow, 6维动作恒 compact)
CUDA_VISIBLE_DEVICES=0 python -u train_fm.py \
  --root /mnt/pfs/data/huangzehuan/datasets/code2world_act6_tc \
  --window 41 --batch_size 8 --steps 10000 --eval_every 500 \
  --action_window 3 --out <ckpt_dir> > train_bidir.log 2>&1 &

# 推理(整段生成 42 latent → 165 帧 → 16fps ≈10s)
python -u rollout_fm.py --root <same_root> --ckpt <ckpt_dir>/ckpt_final.pt --flow_steps 16
```
