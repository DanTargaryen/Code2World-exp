# Code2world — 逐帧 action + 因果 block-AR flow DiT(main）

> 神经游戏引擎原型:模型**读懂游戏规则(config/源码)+ 接收玩家动作 → 自回归预测画面**,
> 验证「改规则 → 生成跟着变」(code sensitivity)。核心主张:*code grounds rules*。
>
> **分支状态(2026-07-03 重构后)**:所有工作已合入 `main`,`exp/*` 分支已删除,
> 平行探索(bidir/pixel/mgpos）归档在 `archive/*` tag。data_gen 三维玩具(three.js
> coin_collection）已删除;code condition 统一走原版 procgen CoinRun。
> **训练目标 flow-only**:reward/done 机制已从模型/训练/数据全链路移除,只留 flow(velocity）loss。

## 0. 定位与目标

**目标**:忠实复刻 **Matrix-Game-3** 的 action 注入机制(逐帧窗口 cross-attn),
验证它在 **code-conditioned** 场景的效果。两条关键决策:

1. **逐帧 action**(非逐 latent):每帧记录真实动作,一个 latent(4帧)携带 4 个逐帧 action,
   回到 Matrix-Game 原始的 `act_hidden × vae_ratio × window` 维度。当前「1 action=1 latent」
   发挥不出 window cross-attn 的价值。
2. **因果 block-AR 外层**(档次1):复用 block-causal + `block_ar_generate`,**不含** memory/
   KV-cache/sliding window。bidir 是另一条偏离路线(归档 `archive/action-bidir`),此处不用。

**action_cross** 用 LayerNorm、无 RoPE(bidir 链上的 RoPE/RMSNorm 在 `archive/action-bidir`)。

## 1. 三个核心验证指标

- **Action following**:逐帧动作是否被生成帧反映(尤其 latent 内动作切换)
- **Code sensitivity**(最核心):改规则源码后,预测帧是否跟着变
- **Rollout stability**:block-AR 自回归 50+ 步是否仍合理

## 2. 架构(`models/causal_dit.py`,`CausalDiT`)

```
CoinRun 64×64 ──Wan2.1 VAE(8×space+4×time)──> latent 序列 ─┐
coinrun.cpp ──Qwen2.5-0.5B(冻结)──> code ─cross-attn───────┤
逐帧 action(6维,窗口 R×W) ──window cross-attn──────────────┤
                                                          ↓
                              Causal DiT(block-causal) → 逐 latent velocity
                                                          ↓
                              block_ar_generate(块自回归) → Wan VAE decode
```

- 序列 = **L 个 latent**(默认 42=window41+init),每 latent **S=8×8=64 token**。
- **1 latent ↔ 4 帧 ↔ 4 个逐帧 action**;latent 0 是 init(恒 clean、不预测,其 4 帧 action=null)。
- **模型规模(2026-07-02 定)**:24 层 / **width 768** / 16 head(DiT-L 标准配置),
  **385M 参数**(ckpt bf16≈770MB)。由 12/512 加深加宽而来(觉得 12 层表达不够、资源不缺)。

**DiTBlock 顺序**(block-causal mask `allow`,`action_mode=crossattn`):
```
1. tau bias(zero-init)  2. spatial self-attn  3. temporal block-causal attn
4. cross-attn(code)      5. action window cross-attn(逐帧)  6. FFN
```
> 注:action 注入在 block 最前(action_mode 分支内),与归档 bidir 分支的「cross后」位置不同。

**逐帧 action window cross-attn**(`ActionWindowCrossAttention`):
- 输入 per-frame `(B, R×L, A)`(R=vae_ratio=4);每 latent i 取 `[R·i : R·(i+W)]` 共 **R×W 个**
  逐帧 action embed,concat 做 K/V → kv 维度 `act_hidden × R × W = 128×4×3 = 1536`(**=Matrix-Game**)。
- 左 pad `R×(W-1)` 帧(重复首帧);visual token 做 Q,沿时间轴 block-causal attend;proj zero-init。

## 3. Action 空间(`action_space.py`)

- CoinRun 有效动作剔除向下 0/3/6,留 **6 个**:`1=左 2=左跳 4=停 5=跳 7=右 8=右跳`。
- 数据集存原始 Procgen id;模型边界 `remap_to_compact` → 紧凑 `[0..5]`,**6 维 one-hot**。

## 4. 数据集 `code2world_act6_pf`(逐帧版)

路径:`/mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf/`(不覆盖旧 `_tc`)。
- **采集**(`dataset/collect_one.py`):`action_repeat=4` 帧结构不变,但**每帧记录真实 action**;
  动作用 `ActionStream` **分段持续**(每段随机 2~8 帧、段边界不对齐 latent),
  实测 **~47% 的 latent 内含动作切换**(逐帧方案的可学 target 来源)。
- **存储**(flow-only):每 ep `4K+1 帧` + `4K 个逐帧 action`(对齐 frames[1:]),
  加 `per_frame_actions` 标记;`episode_lengths` = `len(actions)//action_repeat`。
  **reward/done 已不再采集/存储**。paired/eval 用 seed 确定性重放逐帧动作流。
- **precompute**(`precompute.py`/`precompute_shard.py`):actions 透传,`.pt` 只存
  latents/actions/episode_lengths/action_repeat/seeds;**dataset.py** 自动检测 per_frame,
  窗口切 `R×T` 个逐帧 action,item = latents/actions/code(无 reward/done)。

### 4.1 config-driven 数据生成(新框架,取代改源码重编译)

- procgen CoinRun 的机制常量已提成 **float options**(`coinrun_gravity/max_jump/max_speed/
  air_control/mixrate/goal_reward`),默认 = 原版硬编码值 → 改参数**无需重编译**。
  改动在 procgen 分支 `feat/coinrun-config-options`(vecoptions 加 `consume_float`、
  game.h/game.cpp 加 options、coinrun.cpp 运行时读、env.py `coinrun_config` 透传)。
- **YAML config 当 code condition**:`dataset/game_config.py` 载入 `dataset/configs/*.yaml`
  → `coinrun_config` dict → `ProcgenEnv(coinrun_config=...)`。可读的声明式机制描述取代
  「喂原始 coinrun.cpp」。`dataset/verify_config.py` 自检:base config 与 vanilla 逐帧 md5
  一致(零漂移)+ 确定性 + fast/lowgrav counterfactual(pod 上全过)。
- **旧变体做法**(`variants.py` 字符串替换 coinrun.cpp + 每变体重编译)仍在,接 config
  驱动采集是下一步。

## 5. 当前进度(2026-07-03)

- ✅ **VAE go/no-go 前置实验通过**(归档 `archive/action-bidir`):VAE 把 4 帧压 1 latent 后,
  latent 内帧间运动**保留 ~95%**(post/pre=0.957)→ 逐帧 action 有物理可学 target。判定 **GO**。
- ✅ **逐帧采集 + 逐帧 action 注入**:action_cross kv 维度回到 Matrix-Game 的 1536,
  逐帧因果性/zero-init/block_ar_generate 端到端自检全过。
- ✅ **模型规模定为 24/768/16(385M)**。
- ✅ **数据集 `code2world_act6_pf` 就绪**(单场景 overfit):`--single-scene --level 0`,
  128 env 全固定 level、只取首个 episode、销毁重建 venv,保证每 ep init 帧完全相同。
  train 20000 ep / eval 1000 ep(同场景 + held-out 动作流 seed 900000 段)。
  预编码用 `precompute_shard.py`(按 ep 切 N 片、每卡一片、`--merge` 拼回)。
- ✅ **8 卡 DDP 训练脚本**(`train_fm_ddp.py`):FlowTrainWrapper 把 forward_flow + flow loss
  收进一个 forward() 再 DDP 包。
- ✅ **仓库重构(2026-07-03)**:合并 perframe 到 main + 删 exp 分支(archive tag 归档)+
  删 data_gen 玩具 + 删 pixel 死代码线 + 删 legacy train.py / data/ 遗留工具。
- ✅ **flow-only**:移除 reward/done —— 模型删 reward_head/done_head/forward_state;
  train_fm/train_fm_ddp loss=loss_fm;dataset/collect/precompute 不再读写 reward/done;
  加载 `strict=False` 兼容旧 ckpt。pod 冒烟(3步)+ 采集自检通过。
- ✅ **config-driven 数据生成框架**(见 §4.1):procgen options 化 + YAML config,验证零漂移。
- ⏳ **训练待重启**:数据/脚本/模型就绪。之前 8 卡 DDP 首启 CUDA OOM 是显存被占,非代码问题;
  换到有空闲显存的 1n pod 即可(`hzh-easygo-1n-2-master-0` 有 procgen 编译环境)。

### ⏭️ 下一步

1. **重启 flow-only 训练**(重采数据后):
   ```bash
   cd model && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
     --master_port=29530 train_fm_ddp.py \
     --root /mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf \
     --window 20 --block_size 3 --batch_size 8 --steps 10000 --eval_every 500 \
     --action_mode crossattn --action_window 3 --action_compact \
     --out <ckpt_dir> > ../logs/train_pf_ddp.log 2>&1 &
   ```
   显存不够降 `--batch_size 4`。有效 batch = batch × 卡数。
2. 训完看 `logs/train_pf_ddp.log` 的 fm loss(能否压到极低=overfit 成立)+ `sample_*.png`
   (block-AR rollout vs GT,单场景能否复现)。
3. **接 config 驱动采集**:把 `collect_one.py`/`build_dataset.py` 从 variants.py 字符串替换
   切到 `game_config.py` 的 YAML config;扩展更多机制(pit/敌人/布局,多为 int/bool)。
4. ablation:`--action_mode bias`(逐 latent)对比 crossattn 逐帧,验证窗口 cross-attn 价值。
5. 多变体 code sensitivity(多 config 采集、非单场景)。

## 6. Ablation 设计(验证机制价值的关键)

复刻忠实度不等于机制有效,必须靠对照测出价值:
| 配置 | `--action_mode` | 说明 |
|---|---|---|
| baseline | `bias` | 逐 latent 单动作加性 bias(折叠逐帧取末帧),main 老做法 |
| **主对象** | `crossattn` | 逐帧窗口 cross-attn(Matrix-Game 机制) |
| (可选)隔离 | `crossattn --action_window 1` | 只喂当前 latent,隔离「窗口历史」vs「cross-attn」贡献 |

## 7. 关键参考

- **Matrix-Game-3**(`workspace/Matrix-Game/Matrix-Game-3/wan/modules/action_module.py`):
  action 注入直接蓝本(keyboard 分支:窗口逐帧动作做 K/V、img 做 Q)。
- **Wan 2.1**:3D Causal VAE(8× 空间,16ch,4× 时间)。
- **ViT/DiT 规模协议**:24 层对应 width 1024(ViT-L/DiT-L);本任务小,取 768 折中。

## 8. 协作约定 & 命令

- 默认中文、简明。训练/采集在 K8s pod(非跳板机,跳板机无 procgen/gym3),
  `/mnt/pfs` 共享盘落日志/ckpt。
- 进 pod:`kubectl exec hzh-easygo-1n-2-master-0 -c pytorch -- bash -lc '<cmd>'`
  (这台有 procgen 编译环境);后台长任务
  `cd model && CUDA_VISIBLE_DEVICES=N nohup python -u ... > ../logs/xxx.log 2>&1 &`。
- **所有日志统一落到仓库根的 `logs/`(本地,已 gitignore)**,不要散落在 model/ 或各处。
- **procgen 重编译**(改了 C++ 后):`.build` 常被 root 占用,先
  `chown -R 1013:1013 .../procgen/procgen/.build`,再
  `cd procgen/.build/relwithdebinfo && cmake --build . --config relwithdebinfo`,
  root 编译后再 chown 回来。gym3 自动把 python float 编码成 FLOAT32 → C++ `consume_float`。
- 访问 GitHub/HF 先设代理 `export http_proxy=http://192.168.48.17:18000`(https 同)。
- **筛选 10k steps**,确认最优再跑满 20k。

```bash
# 0.(改了 coinrun options 才需)pod 内重编译 procgen,见上
# 1. 采集(逐帧)—— 现走 variants.py;config 驱动见 §4.1,自检 dataset/verify_config.py
python -u dataset/build_dataset.py --action-repeat 4   # 默认 out=code2world_act6_pf
# 2. 预编码
CUDA_VISIBLE_DEVICES=0 python -u dataset/precompute.py --root <pf_root> --only latents
# 3. 训练(24/768/16, crossattn 逐帧, flow-only)
CUDA_VISIBLE_DEVICES=0 python -u train_fm.py --root <pf_root> \
  --window 20 --block_size 3 --batch_size 8 --steps 10000 --eval_every 500 \
  --action_mode crossattn --action_window 3 --action_compact \
  --out <ckpt_dir> > ../logs/train_pf.log 2>&1 &
```
