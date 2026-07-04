# 评估产物：loss 曲线 & rollout sample

训练完成后，用这两个脚本从 ckpt/日志产出可视化结果。产物默认落 `examples/`（本地，
gitignore），脚本在 `model/`。

## 1. loss 曲线图 — `model/plot_loss.py`

从训练日志解析 `step N | fm X`（train）与 `[eval] fm X`（eval），画 fm loss vs step
（log-y）。eval 行无 step，对齐到其前最近的 train step。

```bash
cd model
python plot_loss.py --log ../logs/pipeline_base.log --out ../loss_curve.png
```

产物：`loss_curve.png`（train + eval 两条曲线，图例标注 final loss）。
本次 base 单场景 overfit 结果：train fm 1.693 → 0.025，eval fm 0.188 → 0.023。

## 2. rollout sample — `model/rollout_sample.py`

加载 ckpt + 一段 eval clip，从 init latent + GT 逐帧动作做 block-AR 生成，GT 与
生成各自 VAE decode 回 RGB，拼成网格（每格上 GT / 下 gen，标注每 latent 的动作）。
复用训练里验证过的 `block_ar_generate` + compact 动作路径，与训练时的 `sample_*.png`
同源。ckpt 的超参（compact/num_actions/block_size 等）从 ckpt["args"] 自动读取。

```bash
cd model
CUDA_VISIBLE_DEVICES=0 python rollout_sample.py \
  --ckpt /mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world_base_spec_ddp/ckpt_final.pt \
  --root /mnt/pfs/data/huangzehuan/datasets/code2world_base_spec \
  --variant base --window 20 --n_latents 21 --out ../examples/rollout_final.png
```

产物：`examples/rollout_final.png`（21 latent，7 列网格，top=GT / bottom=gen）。

> 注：都在 pod（如 `hzh-easygo-1n-2-master-0 -c pytorch`）上跑，需要 matplotlib/cv2/
> torch + VAE/ckpt。跳板机无 GPU/依赖。产物 root 属主时记得 `chown 1013:1013`。
