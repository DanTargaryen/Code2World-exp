#!/usr/bin/env bash
# One-shot pipeline: collect (base, single-scene, 20000 ep) -> precompute code +
# sharded latents -> 8-GPU DDP flow training. Run inside pod 1n-2-master-0.
# Logs to ../logs/. Each stage aborts the pipeline on failure.
set -euo pipefail

cd "$(dirname "$0")"                       # model/
ROOT=/mnt/pfs/data/huangzehuan/datasets/code2world_base_spec
VAE=/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/FastVideo/Wan2.1-VSA-T2V-14B-720P-Diffusers
QWEN=/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/Qwen2.5-0.5B
CKPT=/mnt/pfs/users/huangzehuan/projects/linming/checkpoints/code2world_base_spec_ddp
LOG=../logs
mkdir -p "$LOG" "$CKPT"

echo "==== [1/4] collect base single-scene (20000 train / 1000 eval) ===="
python -u dataset/build_dataset.py --single-scene --variants base \
  --out "$ROOT" --level 0 --num-envs 128 \
  --unpaired 20000 --eval-unpaired 1000 --max-steps 20

echo "==== [2/4] precompute code embeds (Qwen over full spec) ===="
CUDA_VISIBLE_DEVICES=0 python -u dataset/precompute.py --root "$ROOT" \
  --only code --code-source yaml --qwen "$QWEN" --device cuda:0

echo "==== [3/4] precompute latents (8-GPU sharded) ===="
for split in episodes_train episodes_eval; do
  echo "  -- $split: 8 shards --"
  for k in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$k python -u dataset/precompute_shard.py --root "$ROOT" \
      --variant base --split $split --shard $k --nshards 8 --vae "$VAE" --device cuda:0 &
  done
  wait
  echo "  -- $split: merge --"
  python -u dataset/precompute_shard.py --root "$ROOT" \
    --variant base --split $split --nshards 8 --merge
done

echo "==== [4/4] DDP flow training (8 GPU, 10000 steps) ===="
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=29531 \
  train_fm_ddp.py --root "$ROOT" --vae "$VAE" \
  --window 20 --block_size 3 --batch_size 8 --steps 10000 \
  --eval_every 500 --save_every 2000 --sample_every 2000 \
  --action_mode crossattn --action_window 3 --action_compact --num_actions 6 \
  --out "$CKPT"

echo "==== pipeline done -> ckpt in $CKPT ===="
