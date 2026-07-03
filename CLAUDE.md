# Code2world — 逐帧 action + 因果 block-AR DiT(exp/action-perframe 分支)

> **本文件是 `exp/action-perframe` 分支的独立说明**。
> 神经游戏引擎原型:模型**读懂游戏规则源码 + 接收玩家动作 → 自回归预测画面**,
> 验证「改源码 → 生成跟着变」(code sensitivity)。核心主张:*code grounds rules*。

## 0. 本分支定位与目标

**目标 (a)**:忠实复刻 **Matrix-Game-3** 的 action 注入机制(逐帧窗口 cross-attn),
验证它在 **code-conditioned** 场景的效果。据讨论确定的两条关键决策:

1. **逐帧 action**(非逐 latent):每帧记录真实动作,一个 latent(4帧)携带 4 个逐帧 action,
   回到 Matrix-Game 原始的 `act_hidden × vae_ratio × window` 维度。当前「1 action=1 latent」
   发挥不出 window cross-attn 的价值。
2. **因果 block-AR 外层**(档次1):复用 block-causal + `block_ar_generate`,**不含** memory/
   KV-cache/sliding window。bidir 是另一条偏离路线(`exp/action-bidir`),此处不用。

**分支关系**:从 `exp/action-window`(crossattn + block-causal)起,**不含** bidir 链上的
RoPE/RMSNorm(那些在 `exp/action-bidir`)。本分支 action_cross 用 LayerNorm、无 RoPE。

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
> 注:本分支 action 注入在 block 最前(action_mode 分支内),与 bidir 分支的「cross后」位置不同。

**逐帧 action window cross-attn**(`ActionWindowCrossAttention`):
- 输入 per-frame `(B, R×L, A)`(R=vae_ratio=4);每 latent i 取 `[R·i : R·(i+W)]` 共 **R×W 个**
  逐帧 action embed,concat 做 K/V → kv 维度 `act_hidden × R × W = 128×4×3 = 1536`(**=Matrix-Game**)。
- 左 pad `R×(W-1)` 帧(重复首帧);visual token 做 Q,沿时间轴 block-causal attend;proj zero-init。

## 3. Action 空间(`action_space.py`)

- CoinRun 有效动作剔除向下 0/3/6,留 **6 个**:`1=左 2=左跳 4=停 5=跳 7=右 8=右跳`。
- 数据集存原始 Procgen id;模型边界 `remap_to_compact` → 紧凑 `[0..5]`,**6 维 one-hot**。

## 4. 数据集 `code2world_act6_pf`(逐帧版,新采)

路径:`/mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf/`(不覆盖旧 `_tc`)。
- **采集**(`collect_one.py`):`action_repeat=4` 帧结构不变,但**每帧记录真实 action**;
  动作用 `ActionStream` **分段持续**(每段随机 2~8 帧、段边界不对齐 latent),
  实测 **~47% 的 latent 内含动作切换**(逐帧方案的可学 target 来源)。
- **存储**:每 ep `4K+1 帧` + `4K 个逐帧 action`(对齐 frames[1:]),reward/done 仍逐 latent(K),
  加 `per_frame_actions` 标记。paired/eval 用 seed 确定性重放逐帧动作流。
- **precompute 无需改**(actions 透明透传);**dataset.py** 自动检测 per_frame,窗口切 `R×T` 个逐帧 action。

## 5. 当前进度(2026-07-02)

- ✅ **VAE go/no-go 前置实验通过**(`vae_intra_latent_probe.py`,在 `exp/action-bidir` 分支上做):
  VAE 把 4 帧压 1 latent 后,latent 内帧间运动**保留 ~95%**(post/pre=0.957),
  → 逐帧 action 有物理可学 target。判定 **GO**。
- ✅ **采集侧改造完成**(`e84f350`):逐帧存储 + 分段持续,格式/对齐自检全过。
- ✅ **模型侧逐帧注入完成**(`a4336f2`):action_cross kv 维度回到 Matrix-Game 的 1536,
  逐帧因果性/zero-init/block_ar_generate/prep_batch+compute_loss 端到端自检全过。
- ✅ **模型规模定为 24/768/16(385M)**:train_fm 默认已改。
- ✅ **数据集 `code2world_act6_pf` 就绪**(单场景 overfit,`456ce4a`/`44f24ef`):
  - **只采 base 单变体、固定 level 0 单场景**(用户要求先 overfit base、不学场景随机性):
    `--single-scene --level 0`,128 env 全固定 level、只取首个 episode、销毁重建 venv,
    保证每 ep init 帧完全相同(实测唯一数=1)。逐帧 action、window 20→21 latent。
  - train **20000 ep** / eval **1000 ep**(同场景 + held-out 动作流 seed 900000 段);
    实测 99% 满 20 latent、latent 内动作切换 47%。
  - **latent 预编码**:单卡满卡争抢会卡死→写了 `precompute_shard.py`(按 ep 切 N 片、
    每卡一片、`--merge` 拼回)。8 卡并行几分钟编完。产物:
    `latents/base__episodes_train.pt`(418942 latent/866MB)+ `__eval.pt`(20931/43MB)、
    `code_embeds.pt`(base 4615×896)。
- ✅ **8 卡 DDP 训练脚本就绪**(`train_fm_ddp.py`,`ac96cbb`):FlowTrainWrapper 把
  forward_flow+forward_state+loss 收进一个 forward() 再 DDP 包;冒烟测试(2卡4步)通过。
  同 commit 修了 `dataset.py` 逐帧索引双重计数 bug(a0 已含 ar 又乘一次→action 切空)。
- ⏳ **训练待重启**:8 卡 DDP 首次启动即 **CUDA OOM** —— 不是代码问题,是启动时
  `hzh-easygo-1n-2-master-0` 8 卡全被别人占满(每卡剩 92MB)。**另一 1n pod
  `hzh-easygo-1n-master-0` 每卡剩 ~42GB 空闲**,可换到那台跑(util 100% 会慢但显存够)。

### ⏭️ 下一步(换机器后从这里继续)

1. **重启训练**(数据/脚本/模型全就绪,只差 GPU 显存):
   ```bash
   cd model && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
     --master_port=29530 train_fm_ddp.py \
     --root /mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf \
     --window 20 --block_size 3 --batch_size 8 --steps 10000 --eval_every 500 \
     --action_mode crossattn --action_window 3 --action_compact \
     --out <ckpt_dir> > train_pf_ddp.log 2>&1 &
   ```
   显存不够就降 `--batch_size 4`,或换到 `hzh-easygo-1n-master-0`。有效 batch=batch×卡数。
2. 训完看 `train_pf_ddp.log` 的 fm loss(能否压到极低=overfit 成立)+ `sample_*.png`
   (block-AR rollout vs GT,单场景能否复现)。
3. ablation 对照:再跑 `--action_mode bias`(逐 latent 单动作)对比 crossattn 逐帧,
   验证「窗口 cross-attn 机制」是否真有价值。
4. (后续)多变体 code sensitivity(需另采 7 变体、非单场景)。

> **换机器提示**:数据集/ckpt 在共享盘 `/mnt/pfs`,代码在 git。GitHub 拉下来后代码即最新
> (本分支 `exp/action-perframe`);数据集路径 `/mnt/pfs/data/huangzehuan/datasets/code2world_act6_pf`
> 需该机器能挂载同一 `/mnt/pfs` 共享盘。VAE/Qwen 权重路径见脚本默认值(同在 /mnt/pfs)。

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

- 默认中文、简明。训练在 K8s pod(非跳板机),`/mnt/pfs` 共享盘落日志/ckpt。
- 进 pod:`kubectl exec hzh-easygo-1n-2-master-0 -c pytorch -- bash -lc '<cmd>'`;
  后台长任务 `cd model && CUDA_VISIBLE_DEVICES=N nohup python -u ... > xxx.log 2>&1 &`。
- 访问 GitHub/HF 先设代理 `export http_proxy=http://192.168.48.17:18000`(https 同)。
- **筛选 10k steps**,确认最优再跑满 20k。

```bash
# 1. 采集(逐帧, 7 变体)
python -u dataset/build_dataset.py --action-repeat 4   # 默认 out=code2world_act6_pf
# 2. 预编码
CUDA_VISIBLE_DEVICES=0 python -u dataset/precompute.py --root <pf_root> --only latents
# 3. 训练(24/768/16, crossattn 逐帧)
CUDA_VISIBLE_DEVICES=0 python -u train_fm.py --root <pf_root> \
  --window 41 --block_size 3 --batch_size 8 --steps 10000 --eval_every 500 \
  --action_mode crossattn --action_window 3 --action_compact \
  --out <ckpt_dir> > train_pf.log 2>&1 &
```
