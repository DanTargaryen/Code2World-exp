# Code2world — Code-Conditioned Causal DiT 世界模型

> 神经游戏引擎原型:让模型**读懂游戏规则的可执行源码 + 接收玩家动作 → 自回归预测下一帧画面**,
> 并验证「改源码 → 生成跟着变」(code sensitivity / 反事实)。核心主张:*code grounds rules*。

## 1. 目标

基于 Procgen **CoinRun**(64×64),用 **Wan 2.1 VAE + Causal DiT + Qwen2.5-0.5B 源码编码** 训练
teacher-forcing 的自回归视频世界模型。三个核心验证指标:

- **Action following**:给定 action,生成帧是否反映对应动作
- **Code sensitivity**(最核心命题):改规则源码后,预测帧是否跟着变(反事实)
- **Rollout stability**:自回归生成 50+ 步后画面是否仍合理

## 2. 架构

```
CoinRun 64×64 ──Wan2.1 VAE encode(8×)──> 8×8×16 latent ─┐
coinrun.cpp ──Qwen2.5-0.5B(冻结)──> code tokens ─cross-attn─┤
action(0~8) ──per-layer Linear──> additive bias ──────────┤
                                                           ↓
                                          Causal DiT(12层,63.6M) → 预测下一帧 latent
                                          (帧内 full attn + 帧间 causal attn)
                                                           ↓
                                          Wan2.1 VAE decode → 64×64 frame
```

三路输入都投到 **embed_dim=512**:VAE latent(16→512) 做 token、action(9→512) 做每层独立 bias、
code(896→512) 做 cross-attn 的 K/V。**VAE 与 Qwen 全程冻结、离线预编码**,只训 DiT + 三个投影层。

- **Loss** = MSE(latent) + 0.1·CE(reward, 3类) + 0.1·CE(done, 2类),teacher forcing。
- **模型**:embed 512 / 12 层 / 8 head / spatial 8×8=64 token / window 32 帧 / 63.6M 全可训练。

## 3. 数据集 `code2world_act6`(7 变体配对版)

路径:`/mnt/pfs/data/huangzehuan/datasets/code2world_act6/`

单游戏 CoinRun,围绕 3 个物理参数做 **7 个固定档位单变量变体**(便于 code sensitivity 对比):

| 变体 | 改动参数 | 默认→变体 | 视觉效果 |
|------|---------|----------|---------|
| base | —（默认） | — | 基准 |
| fast / slow | maxspeed | 0.5→0.9 / 0.5→0.25 | 水平更快 / 更慢 |
| lowgrav / highgrav | gravity | 0.2→0.1 / 0.2→0.35 | 下落飘 / 下落沉 |
| highjump / lowjump | max_jump | 1.5→2.5 / 1.5→0.9 | 跳更高 / 更矮 |

- **Code condition**:每变体存完整 `source.cpp`,改动行加 `// [VARIANT] changed from <old> (<说明>)` 注释让 Qwen 注意到那几个数字(变体间 99% 源码相同)。Qwen 编码后存 `code_embeds.pt`,训练查表。
- **VAE latent** 离线预编码存 fp16(`latents/`),训练不跑 VAE。
- **训练集** = 非配对(各变体随机 seed+随机动作)+ 部分配对;**评估集** = 全配对 held-out seed。
  配对样本 = 同关卡 seed + 同动作序列跑全 7 变体,是 code sensitivity 最干净的对照
  (Procgen 在 seed+动作固定时确定性)。
- **动作采样范围**("act6" 之名即 6 个有效动作):采集时**剔除 vy=-1 的向下动作**(id 0/3/6,平台游戏无意义),
  只保留 **6 个**——`1=左 2=左+跳 4=停 5=跳 7=右 8=右+跳`(映射 `vx=a/3-1, vy=a%3-1`,见 `set_action_xy`)。
  模型侧 `num_actions=9` 仍做 9 维 one-hot(0/3/6 维恒为 0)。7 变体全只含这 6 个值、分布均匀。
- 规模:~2.13M 帧 / latent 4.4GB / window=32 → 训练 ~16855 窗口、评估 ~696 窗口。

## 4. 当前进度(2026-07-01)

- ✅ **Stage 2 完成 — Flow + 时间压缩 + Block-AR 训练跑通(数据集 `code2world_act6_tc`)**:
  - 20000 steps @ pod `hzh-easygo-1n-2-master-0`(`pytorch` 容器,1×A800),1.21 it/s、~4.6h。
  - 最终 **train fm 0.148 / eval fm 0.138~0.142**,eval 全程单调降(0.359→0.138)且 **eval≤train,零过拟合**。
    ⚠️ fm loss 与旧 MSE 版(0.02 量级)**不可直接比**:目标是 velocity `z₁−ε`,方差≈2,量纲不同;0.14 约解释掉 ~93% velocity 方差。
  - 配置:window 41(→42 latent=14 block×3)/ block_size 3 / batch 8 / bf16。产物 `checkpoints/code2world_act6_tc_fm/ckpt_final.pt`,日志 `model/train_fm_tc.log`。
  - **噪声实现**:per-chunk 独立 —— block 内 3 帧共享同一 τ(σ)、block 间独立采;init 恒 τ=1;**ε 仍逐 latent 独立**(共享 ε 会制造帧间伪相关泄漏)。匹配"整块联合去噪"的推理。
- ✅ **完整 10s 自回归 rollout 验证**(`rollout_fm.py`,41 步→42 latent→`decode_video` 165 帧→16fps→10.3s):
  - 从 init 真·AR 生成(block-AR:历史块 τ=1、当前块从噪声 Euler 16 步积分)。
  - 现象:**前 ~15 步结构成立;后期(24~41 步)语义漂移(地面收窄、偏蓝灰、角色淡化)但全程保持锐利**。
  - **关键结论**:flow 解决了"糊" —— AR 到 41 步画面仍锐利清晰,**不再**出现旧 MSE 版"~20 步坍塌成渐变色块"。
    问题从「画面糊」变成「长程内容漂移」(误差累积表现为语义偏离而非模糊),更本质、也更好办 → 下一步 scheduled sampling(见 `future_plan.md §1.4`)。
  - 产物:`model/outputs/custom_rollout_fm/base_blockar.mp4` + `_grid.png`。

### 实验分支:pixel-space(路径 C,2026-07-01,与 latent 版并行对比)

验证「去掉 Wan VAE、直接 pixel-space 生成是否更锐/artifact 更少」。核心技巧:**8×8 pixel patch 当一个 token**
→ 每 token=8·8·3=192 维原始像素、grid 8×8=64 token/帧,**几何与 latent 版完全一致**,`CausalDiT` 零架构改动
(仅 `latent_dim=192, spatial_size=8`)。VAE 换成无参数 patchify/unpatchify(纯 reshape,无损)。

- **路径 C = 每帧建模、不做时间压缩**:window=41 帧(+init=42 latent step=14 block×3)=2.6s@16fps,快速验锐度。
- 新文件:`models/pixel_codec.py`(patchify + [0,1]→[-1,1] 归一化)、`dataset/pixel_dataset.py`(直接读 npz `frames`,
  per-frame 动作映射 `(i-1)//action_repeat`,无需重采/预编码)、`train_fm_pixel.py`(复用 `train_fm` 的 loss/τ/state)、
  `rollout_compare.py --pixel`。loss/per-block τ/forward_state 逻辑与 latent 版一致,唯一变量=「16维学习latent vs 192维原始像素」。
- ⚠️ **归一化坑(已解决)**:最初用 per-channel 单位方差归一化,遇 CoinRun「多暗+少量亮」重尾分布产生大 outlier
  (亮像素→+7),fm loss 飙到 ~20。改用标准 **[0,1]→[-1,1]**(mean/std=0.5)后 fm 起点回到 ~1.3(与 latent 版同量级)。
- 显存实测:8×8 patch batch8 ~0.9 it/s、峰值 45GB(可跑);4×4 patch(256 token/帧)batch4 即 OOM,路径 C 不用。

### 结论(2026-07-02):**pixel-space 更糊,latent 空间胜出**

跑了两版 pixel 对照(均 20k、收敛):
- **pixel@16fps**(block_size=3,注入连续4帧,覆盖2.6s):eval fm 0.08,step6k 预览已见糊(中途停,未取满)。
- **pixel@4fps**(分支 `exp/pixel-4fps`,block_size=1,1帧/动作,覆盖10.5s):20k done,**eval fm 0.054**(收敛),
  产物视频 `model/outputs/pixel4fps_cmp/base_ep98_pixel_gt_vs_gen.mp4`(左GT右生成,4fps)+ grid。

**三方对比(同 ep98)**:

| | latent 版 | pixel@16fps | pixel@4fps |
|---|---|---|---|
| 收敛 eval fm | 0.14 | 0.08 | **0.054** |
| **AR 生成锐度** | **前~15步锐利、贴GT** | step1 起糊 | **step1 即糊成绿块** |

- **关键**:pixel 版 loss 更低(0.05<0.14)但**画面明显更糊** —— 印证"fm loss 不可跨表示比"(velocity 目标方差不同)。
- **糊不是训练不足**:4fps loss 已躺平(10k 后基本不动)、仍糊 → 是方法上限。
- **诊断**:① 4fps 帧间跳变大(0.25s/帧),单步 flow 目标分布过宽→退回糊均值;② 无 VAE 语义压缩,DiT 要在原始像素里同时管语义+几何+锐度,8×8 粗 patch 吃不消;③ block_size=1 少了块内联合去噪。
- **决定**:**Wan VAE 的 latent 空间对生成质量真有价值,pixel-space 不是免费替代**。主线回 latent 版,pixel 路线搁置(若要救:16fps 跑满收敛再比 / 4×4 细 patch 但贵)。

### 分支 `exp/pixel-4x4`:细 patch + 多卡 DDP(2026-07-02,进行中)

试"更细 patch 能否救糊"。git 已建仓,实验按约定开分支(`main`=baseline,`exp/pixel-4fps`、`exp/pixel-4x4`)。
- **4×4 patch 可配置化**:`--patch 4` → 每 token 48 维、grid 16×16=**256 token/帧**(8×8 是 192维/64token)。
  DiT **结构零改动**:patch 大小只改 `latent_dim`/`spatial_size` 两个构造参数,被第一层 `input_proj` 统一投到 512 维吸收;
  代价是 spatial attn O(S²) 从 64²→256²=**16×**,显存重(实测 4×4 batch1=23.7GB vs 8×8 batch8=45GB)。
  `pixel_codec/pixel_dataset/train_fm_pixel/rollout_compare` 全部通 `--patch`。
- **梯度累积** `--accum`:4×4 显存重,batch2×accum4=有效 batch8。
- **多卡 DDP** `train_fm_pixel_ddp.py`(`torchrun --nproc_per_node=N`):
  关键——DDP 只在 `forward()` 挂梯度同步钩子,而训练走 `forward_flow`+`forward_state` 两次前向绕过 forward,
  故用 `FlowTrainWrapper` 把两次前向+loss 收进一个 `forward()` 再 DDP 包;`find_unused_parameters=True`
  (CausalDiT 的 legacy `output_proj` 在 flow 路径不用);DistributedSampler+set_epoch;rank0 独占 log/eval/save。
  2 卡已验证跑通。有效 batch = batch_size×accum×world。
- **现状**:单卡 4×4(`1n-master-0` GPU1,`--patch 4 --stride 4 --block_size 1 --batch2 --accum4`)后台跑,
  **但集群 8 pod×8 卡全 100% 被别人占**,只 ~0.13 it/s、10k 要 ~19h(loss 正常降,fm 0.13@step900)。
  产物 `checkpoints/code2world_act6_tc_pixel4x4/`,日志 `model/train_pixel4x4.log`。
- **待办**:等集群空出 ≥2 卡 → 停单卡、用 DDP 快速重跑(~2-3h)→ 对比 4×4 vs 8×8 vs latent 锐度。

### 历史进度(Stage 1 / act6,已被 Stage 2 取代)

- ✅ **Stage 1(PoC,数据集 `code2world`)**:20K steps 跑通端到端,eval loss 0.0040,**无过拟合**
  (eval<train),能生成连贯但**偏糊**的视频。结论:code→action→video 的 AR 闭环成立。
  - 糊的主因:① MSE 回归均值 ② 自回归误差累积(预期现象,非 bug)。
  - checkpoint:`checkpoints/code2world/`(ckpt_final.pt 等)。
- ✅ **`code2world_act6` 训练完成**(7 变体配对版,为 Stage 3 准备):
  - 20000 steps,在 K8s pod `hzh-easygo-1n-2-master-0`(container `pytorch`)训练,~3.57 it/s、~1.9h。
  - 最终 train loss **0.00460**、eval loss **0.00422**(latent 主导,reward/done≈0),eval<train 无过拟合。
  - 配置:window 32 / batch 8 / num_actions 9 / bf16 / cuda:0。
  - 产物:`checkpoints/code2world_act6/ckpt_final.pt`,日志 `model/train_act6.log`。
- ✅ **自回归 rollout 验证**(`model/rollout_custom.py`,自定义"右/右上/上"动作序列):
  - 推理是**真 AR**(把自己预测的 latent 拼回历史当输入),动作**逐帧、每层独立 bias** 注入。
  - 现象:**前 ~8 步动作跟手、结构成立;~10 步后漂移、~20 步后场景坍塌成渐变色块**。
    完全符合"MSE 均值化 + AR 误差累积"的预期 → 直接动机:上 flow matching。
  - 产物:`model/outputs/custom_rollout/`(帧网格 PNG + mp4)。
- ⏳ **进行中(下一步,见 `docs/future_plan.md`)**:① loss 改 flow matching;② VAE 时间压缩重采数据(下文)。

### 改造已落地(2026-07-01,代码完成待训练验证)

目标:**生成 ~10s @16fps 视频**,用 Wan2.1 VAE 做时间+空间压缩,**block-AR(每步生成 3 个 latent)+ flow matching**。

**时间账(钉死的)**:10s×16fps=160→录 **165 帧**(=4×41+1)→Wan 时间 4× → **42 latent**(=14 block×3,÷3 满足)。
1 action↔1 latent↔4 帧;**latent 0 是 init**(首帧单独编码,恒给定,不预测),block 0={init, x₁, x₂}。

1. **Block-AR 单流 flow** —— `models/causal_dit.py`(已整体重写):
   - **temporal attention 改 block-causal**:`block(j)≤block(i)`(块内双向、块间 causal),`block_size=1` 退化成原 causal。
   - **backbone 即去噪器**(单流 Diffusion Forcing,非两流):`forward_flow(z_τ,τ,action,code)→逐 latent velocity`。
     **每个 latent 独立 τ、init 恒 τ=1**;块内邻居只以**带噪**形态互看 → 天然无泄漏(这是选单流而非两流的原因:init 混在 block 0 里也不漏)。
   - τ 经 sinusoidal→MLP→**逐 latent 加性 bias**(`tau_proj` zero-init,初始≈旧行为);`flow_out` zero-init 稳起步。
   - **state 解耦**:`forward_state(clean,…)→reward/done`,单独 clean 前向、τ-free。
   - `forward()`(旧 MSE 路径)保留,旧 ckpt 用 `strict=False` 可加载(rollout*.py 已改)。
   - **action 对齐**:latent i(i≥1)由 a_{i-1} 产生,序列 pos0 给 null action。
   - `block_ar_generate()`:推理时历史块 τ=1(clean)、当前块从噪声 Euler 积分联合去噪;block 0 出 2 个、之后每块 3 个,41 动作→42 latent。
2. **训练 `train_fm.py`**(整体重写,不动 `train.py`):整窗 L=42 latent,逐 latent τ~U(0,1)(init=1),
   `L_fm=‖v-(z₁-ε)‖²`(排除 init);reward/done CE 挂 `forward_state` 的 clean 输出。sample dump=block-AR 生成 vs GT(`decode_video`)。
   关键参数:`--window 41`(→42 latent)、`--block_size 3`。
3. **推理 `rollout_fm.py`**(重写):`block_ar_generate`+`decode_video`(1 latent→4 帧)+**16fps 导出**,41 步→165 帧≈10s。
4. **VAE 时间压缩**(`vae.py`):`encode_video`(4K+1 帧→K+1 latent 整段)/`decode_video`(逆)。
5. **数据集 `code2world_act6_tc`**(新目录):`collect_one.py --action-repeat 4 --max-steps 60`
   (录全 4K+1 帧、reward 窗口求和、done OR、跨 reset 丢弃);`precompute.py` 按 `action_repeat` 自动走 temporal 模式逐 episode `encode_video`,
   flat 后每 ep 仍 K+1 latent → `dataset.py` 零改动(window=41 取 42-latent 窗口,block 网格在窗口局部下标上、采样偏移无关)。

> 自检通过:block-causal mask 正确、forward_flow/forward_state/block_ar_generate shape 对、训练步+反传 OK、41 动作→42 latent。
> 下一步(K8s pod):① `build_dataset.py`(默认 out=`code2world_act6_tc`)→ `precompute.py`;② `train_fm.py --window 41 --block_size 3`;③ `rollout_fm.py` 看 10s 锐度。

### 已确认的关键事实(供改造用)

- **Wan VAE 是 causal 3D VAE,时间压缩规律 `T 帧 → 1+(T-1)/4 个 latent`**(实测:1→1,5→2,9→3,128→32;
  decode 反之 `L → 1+(L-1)·4`)。即首帧单独成 1 latent,之后每 4 帧压成 1 latent。
- 当前 `models/vae.py` 仍是**逐帧编码**(T=1 chunk,只用空间 8×、未用时间压缩);改造需按 episode 整段编码。

## 5. 后续规划

详见 `docs/future_plan.md`(以 flow matching velocity 为主线)。当前正在推进两件**耦合**改动:

1. **Loss 改 flow matching**:latent 预测从 MSE 回归均值 → **rectified flow 预测 velocity**
   `v*=z₁−ε`(建模分布而非均值,解决糊);per-frame 独立噪声(Diffusion Forcing),τ 经 AdaLN-Zero 注入;
   reward/done state head 仍挂 clean condition(与 τ/噪声解耦)。
2. **VAE 时间压缩重采数据**:`action_repeat=4`(每动作持续 4 env steps)→ `4K+1 帧` 经 Wan VAE 时间压缩
   → `K+1 个 latent`,**一个 action ↔ 一个 latent ↔ 4 帧**。序列短 4×、长程误差累积少 4×。
   - 改造面:`collect_one.py`(action repeat + reward 聚合 sum / done 聚合 any)、
     `vae.py`(逐帧→整段时间压缩)、`precompute.py`/`dataset.py`(latent 时间维=action 数)、
     `causal_dit.py`(时序维换成 latent step,模型结构基本不动)。
   - 新数据集独立目录(不覆盖 `code2world_act6`)。

后续(数据/loss 改造跑通后):

- **Stage 3 — Code sensitivity 量化**(核心科学问题,数据已就绪):同 held-out seed + 同动作喂不同变体源码,
  测生成轨迹是否随 code 改变方向正确;对照:code 置零/打乱看预测是否退化。已有 `eval_code_sensitivity.py` / `visualize_sensitivity.py`。
- **Stage 4 — 鲁棒性/扩展**:long-horizon rollout、window 加长、DDP 多卡、跨游戏、连续参数采样。
- **演进**:Transfusion-lite 统一序列(同 backbone 同时出 state 与 velocity,互为条件),见 `future_plan.md` §3。

## 6. 目录结构

```
workspace/Code2world/
├── CLAUDE.md              # 本文件
├── data_gen/              # Stage 1 数据生成(three.js + playwright,确定性引擎)
│   ├── generate.mjs       # 批量采集 CLI;recorder.mjs / rng.js(mulberry32)/ verify.mjs
│   └── games/coin_collection.js
├── model/                 # 训练 + 模型 + 评估
│   ├── train.py           # teacher-forcing 训练(act6 用 train_act6.log)
│   ├── dataset/           # collect_one.py(采集,含 ACTION_SET=6动作)/ precompute.py(VAE+Qwen 预编码)
│   │   ├── build_dataset.py / variants.py / dataset.py
│   ├── models/            # causal_dit.py / vae.py(Wan VAE wrapper,当前逐帧)
│   ├── rollout.py / rollout_custom.py   # AR 推理(后者支持自定义动作序列)
│   ├── eval_code_sensitivity.py / visualize_sensitivity.py
│   ├── outputs/           # 推理产物(custom_rollout/ 等)
│   └── *.log              # train_act6.log / train_run.log 等
└── docs/                  # 设计文档与报告(html notebook,localhost:8000/docs/ 查看)
    ├── future_plan.md           # Stage2+ 改造计划(flow matching 主线 + state + Transfusion-lite)
    ├── model_plan.md/.html      # 训练计划全文
    ├── dataset_spec.html        # 数据集设计规格(变体/配对/动作采样)
    ├── training_pipeline.html   # 训练 pipeline 逐步详解
    ├── stage1_summary.html      # 第一阶段训练总结(曲线/指标/糊根因/规划)
    ├── todo.html                # 对照博客 6 阶段路线图的进度
    └── viz.html                 # 数据可视化
```

## 7. 关键参考

- **IRIS**(github.com/eloialonso/iris):VQ-VAE + AR Transformer,参考序列构建与 loss。
- **ReactiveGWM**(arxiv 2605.15256):Wan 底座 + action additive bias + cross-attn,复用 action/code 注入设计。
- **Wan 2.1**(github.com/Wan-Video/Wan2.1):3D Causal VAE(8× spatial,16ch)+ DiT。用 2.1 而非 2.2
  (2.2 压缩 16×/32×,64×64 帧只剩 4×4 或 2×2,信息损失大)。

## 8. 协作约定

- 默认中文、简明扼要。除非明确要求不输出 HTML 正文(报告类用 `docs/` 下 notebook 模板)。
- 训练在 K8s pod 内跑,非这台跳板机;后台长任务用 `python -u … > xxx.log 2>&1 &` 落盘日志。
- 访问 GitHub/HuggingFace 等先设代理 `export http_proxy=http://192.168.48.17:18000`(https 同)。
- **实验默认 10k steps**(观察:后半程 eval loss 躺平、边际极低;糊是方法上限、加步数救不了)。筛选阶段一律 `--steps 10000 --eval_every 500` 快速判优劣;某方案确认最优、要出定稿模型时再跑满 20k。
- **新试验开新 git 分支**开发(`git checkout -b exp/xxx`),主线 `main` 只留已验证的 baseline;产物(outputs/日志/*.pt/*.mp4/*.png)已 gitignore。

## 9. K8s pod 操作流程(实测可用)

**这台是跳板机、不能跑训练**(装不下 14B VAE)。计算全在 pod 里,`/mnt/pfs` 是**跳板机与 pod 共享盘**
(日志/数据/ckpt 落这里,跳板机可直接 `tail`/`ls`,无需再 exec)。

- **进 pod**(非交互式,别用 `-it`):`kubectl get pods` 看列表;GPU pod 是 `hzh-easygo-1n-2-master-0`,
  容器 `-c pytorch`(8×A800-80GB)。跑命令:
  ```bash
  kubectl exec hzh-easygo-1n-2-master-0 -c pytorch -- bash -lc '<命令>'
  ```
- **起后台长任务**(训练/预编码):在 pod 里 `cd model && CUDA_VISIBLE_DEVICES=0 nohup python -u … > xxx.log 2>&1 &`;
  日志在共享盘,跳板机 `tail -f .../model/xxx.log` 直接看;进度/存活用 `kubectl exec … nvidia-smi` + `ps aux|grep`。
  ⚠️ 跳板机 bash 工具默认 120s 超时,`sleep` 别超 110s(否则 exit 143 被杀),长等要显式设 timeout。
- **一次性小验证**用 heredoc:`kubectl exec … -- bash -lc 'python - <<"PY" … PY'`;GPU 选空闲卡
  (`nvidia-smi` 看,GPU1 常被占,验证用 `CUDA_VISIBLE_DEVICES=2`)。

### `code2world_act6_tc` 数据集重建流程(2026-07-01 实测)

1. **采集**(procgen 逐变体重建源码,~1.5min/变体,×7≈11min):
   `python -u dataset/build_dataset.py --action-repeat 4`(默认 out=`…/datasets/code2world_act6_tc`,`--max-steps 60`)。
   跑完自动还原 `coinrun.cpp`、写 `variants.json`。
2. **预编码**(`--only latents` 跳过已存的 code_embeds;temporal VAE **批量** encode ~5min):
   `CUDA_VISIBLE_DEVICES=0 python -u dataset/precompute.py --root …/code2world_act6_tc --only latents`。
   - 关键提速:`encode_frames_temporal` 按 `ep_batch=16` 分组、右 pad 到批内最长帧、单次 VAE 前向后按各自 K+1 切回。
     **Wan VAE 时间上 causal**(实测尾部 pad 帧对前面 latent 零影响,exact 0.0),故 pad-batch 安全、比逐 episode 快 ~10×。
3. **训练**(block-AR flow,67.3M 参数,~1.2 it/s ≈ 4.7h/20k):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python -u train_fm.py --root …/code2world_act6_tc \
     --window 41 --block_size 3 --batch_size 8 --steps 20000 \
     --out …/checkpoints/code2world_act6_tc_fm > train_fm_tc.log 2>&1 &
   ```
   实测:train 10352 窗口 / eval 431;step200 时 fm 0.75、reward/done 已收敛。
4. **推理**:`python -u rollout_fm.py --root …/code2world_act6_tc --ckpt …/code2world_act6_tc_fm/ckpt_final.pt`
   → block-AR 生成 42 latent、`decode_video` 出 165 帧、16fps mp4(≈10.3s)。
