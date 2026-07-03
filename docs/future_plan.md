# Code2world — Future Plan(Stage 2+:从 MSE 到 Flow Matching)

> 当前(Stage 1 / act6)是纯 MSE 回归下一帧 latent,画面偏糊。本计划以 **flow matching 预测 velocity 为主线**
> 解决模糊,并预留 state 联合预测与 Transfusion 式统一序列的演进路径。

## 0. 现状与问题

- 现训练:`train.py` 一步回归 `pred`,`L = MSE(pred, tgt_lat) + 0.1·CE(reward) + 0.1·CE(done)`。
- 糊的两大主因(见 `stage1_summary.html`):① MSE 回归均值(下一帧多解,平均即糊);② 自回归误差累积。
- 方向:**latent 预测从"回归均值"换成"建模分布"** —— flow matching,采样出某个锐利的合理结果。

---

## 1. 主线:Flow Matching velocity(优先级最高)

### 1.1 训练目标(rectified flow)

每个 (sample, frame) 位置独立采样 `τ~U(0,1)`、`ε~N(0,I)`,GT 下一帧 latent 记 `z₁`:

```
z_τ = (1-τ)·ε + τ·z₁
v*  = z₁ - ε                      # 目标 velocity
L_fm = ‖ v_θ(z_τ, τ, c_t) - v* ‖²  # 替代原来的 latent MSE
```

- `c_t` = backbone 对**干净** `inp_lat`(teacher forcing)+ action + code 编出的每帧条件。
- τ 注入:sinusoidal → MLP → **AdaLN-Zero**(zero-init,初始等价恒等,训练稳)。
- 推理:从 `z₀=ε` 用 Euler 积分 1~N 步;1-step 即 `ẑ₁ = ε + v_θ(ε, τ=0, c_t)`,再 VAE decode。

### 1.2 关键 trick:per-frame 独立噪声(Diffusion Forcing)

自回归下同一帧 latent 既是"被去噪的目标"又是"下一帧的干净条件",矛盾点靠逐帧独立 τ 解决:

- 只对要预测的帧加噪,**历史帧当条件时用 clean(τ→1)**。
- 训练时各帧并行、各自 τ;推理时"历史 clean、当前从噪声起"。
- ⚠️ 不要用单一全局 τ,否则训练/推理的噪声假设对不上。

### 1.3 落地步骤

1. `CausalDiT` 拆出 `encode()`(出 clean 条件 `c_t`)+ `flow_head(z_τ, τ, c_t)`(出 velocity)。
2. output_proj 从"出 pred"改成"出 velocity";新增 τ 的 AdaLN 调制。
3. 起 `train_fm.py`(**不动正在跑的 `train.py`**),复用 dataset / backbone。
4. 验证顺序:先确认 flow 跑通、rollout 锐度提升,再叠加 state(见下)。
5. 可并行:延长当前 MSE 版到 50K steps 作对照,确认糊不是单纯训练不足。

> **已完成(2026-07-01)**:实际落地为 **block-AR 单流 Diffusion Forcing**(比原计划的两流 encode+flow_head 更适配 block 结构):
> backbone 即去噪器,temporal 改 block-causal,per-chunk τ(块内共享/块间独立)、ε 逐 latent,state 走独立 clean 前向。
> 结果见 CLAUDE.md §4:20k steps,eval fm 0.138,**AR 41 步画面仍锐利、彻底解决糊**,残留问题是长程语义漂移(下 §1.4)。

### 1.4 下一步:压长程漂移(scheduled sampling / partial rollout)

现象:flow 版 rollout **画面锐利但后期(~24 步后)内容语义漂移**(地面收窄、偏色、角色淡化)。
根因是训练/推理的**历史来源 gap**:

- 训练:历史块是「**GT** latent(加了 per-chunk 噪声)」。
- 推理:历史块是「**模型自己生成**的 latent(τ=1 当 clean)」——自生成误差逐块累积,训练里从没见过。

Diffusion Forcing 的独立 τ 只缓解了「噪声水平」的 gap,没缓解「自生成历史」的 gap。方案(按代价从小到大):

1. **Scheduled sampling / partial rollout(推荐先试)**:训练时以概率 `p`(随训练步 anneal,如 0→0.5)把**部分历史块**替换成模型自己前向采样的结果(no-grad),让 backbone 见过"带累积误差的历史"。
   - 实现:在 `train_fm.py` 里,对历史块偶尔用 `block_ar_generate` 式的一步/少步采样产出 `ẑ`,拼回当条件;当前块照常算 flow loss。
   - 代价:每步多若干次前向;`p` 不宜过大(否则早期噪声历史带偏训练)。
2. **更长 window / 训练更长 horizon**:直接把 window 拉到覆盖目标 rollout 长度(已是 41=10s),或叠加 rollout 微调阶段。
3. **推理端**:增大 `flow_steps`(更精积分)、或对历史做轻度重加噪一致化(consistency-style),属治标。

先做 1,对照当前 `ckpt_final.pt` 看 24 步后漂移是否收敛。

---

## 2. State 联合预测

### 2.1 解耦原则(无论哪种架构都成立)

**state(reward/done/物理量)预测必须基于干净上下文 `c_t`,不能吃 τ/噪声。**
否则同一帧在不同采样 τ 下给出不一致的 state,监督自相矛盾、推理还得选 τ。

### 2.2 方案 A(先做):外挂 head

- reward/done head 直接挂在 clean `c_t` 上(即现有做法),与 flow 分支并存:
  `L = L_fm + 0.1·CE(reward|c_t) + 0.1·CE(done|c_t)`。
- state 推理 O(1) 一次前向;latent 推理走 ODE 积分。两者独立。
- 够用、简单。**只要 state 是标量 reward/done,这就是首选。**

### 2.3 数据缺口提醒

- act6 的 npz 只有 `frames/actions/rewards/dones`,**没有结构化物理 state**(player x/y/vx/vy)。
- 若要回归 player 坐标/速度/事件,需先从 procgen 的 `info`/state 导出存进数据集,再加一路回归 head(MSE)挂到 `c_t`(CoinRun 需单独补采结构化 state)。

---

## 3. 演进方向:Transfusion-lite 统一序列(同时出 state 与 velocity)

> 参考 Transfusion / BAGEL 的"单 backbone、两种目标共存",让 **state 与画面互为条件**。
> 当 state 升级为结构化/语言(而非标量)时才真正划算。

### 3.1 序列布局(time-major)

```
[CODE tokens(prefix,clean)]
 t=0: [STATE₀ tokens][FRAME₀ latents×64]
 t=1: [STATE₁ tokens][FRAME₁ latents×64]
 ...
        ↑离散 AR+CE      ↑连续 flow velocity
```

### 3.2 注意力 mask(混合)

| 位置 | attend 到 |
|---|---|
| CODE | 自身(prefix) |
| STATEₜ | CODE + 所有 `FRAME_{<t}`(clean 历史) + 自身前序 state(causal) |
| FRAMEₜ | CODE + `FRAME_{<t}` + **`STATEₜ`** + 帧内 64 token 互看(bidir) |
| 帧间 | 一律 causal |

- **STATEₜ 排在 FRAMEₜ 之前** → 画面能 condition 于已确定的语义状态("done=1 才生成死亡帧")。
- STATEₜ 只看 clean 历史、看不到同步 noisy FRAMEₜ → 噪声解耦由序列顺序自动实现。
- τ 的 AdaLN **只调制 FRAME 位置**,STATE 位置不给 τ。

### 3.3 损失

```
L = E_{τ,ε}‖v_θ(z_τ,τ,ctx) - (z₁-ε)‖²        # FRAME 位置:flow
  + λ · CE(state tokens)                       # STATE 位置:AR
```

### 3.4 务实判断

- 只要 reward/done → 词表仅几个 token,AR 约等于分类 head,统一序列收益主要在"state condition 画面"。
- **真正划算**:state 含 player x/y、事件、或自然语言(预测/解释下一步)→ AR 的变长结构化能力才用得上,
  也对应"神经引擎能用语言解释状态"的研究故事。
- 规模提醒:Transfusion/BAGEL 是 7B+ 海量预训练涌现;本项目 63.6M 原型只借**范式**(单 backbone + 双目标 + 混合 mask),
  不复现涌现;BAGEL 的 MoT 双 expert 对小模型偏重,暂不碰。

---

## 4. 推进顺序(建议)

1. **[主线] flow matching velocity**:`train_fm.py`,FRAME 分支 MSE→flow,state 暂用现有外挂 head。验证 rollout 锐度。
2. 并行延长 MSE 版到 50K steps 作对照。
3. flow 跑通后,按需补结构化 state 数据 + 回归 head。
4. 若要"同时出 state 与 velocity 且互为条件" → 上 Transfusion-lite 统一序列(`train_transfusion.py`)。
5. 全程为 Stage 3 code sensitivity 评估保留对照 checkpoint。

## 5. 参考

- **Transfusion**(Meta, 2024):单 Transformer,文本 AR(CE)+ 图像连续 latent diffusion;文本 causal / 图像块 bidir;`L=L_LM+λL_diff`。
- **BAGEL**(ByteDance, 2025):decoder-only Mixture-of-Transformers,理解/生成 modality-specific expert,共享注意力。
- **Diffusion Forcing**(Chen et al.):逐帧独立噪声 level,自回归视频/序列扩散的关键调度。
- **Wan 2.1 / ReactiveGWM / IRIS**:见 `model_plan.md`。
