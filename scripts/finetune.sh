#!/bin/bash

# Primary finetune path: use Accelerate for multi-GPU DDP.
# Choose devices via shell-level CUDA_VISIBLE_DEVICES or your local Accelerate config.
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# Single-GPU fallback: replace the launcher with `python main.py` and keep `--gpu_ids`.

# accelerate launch main.py \
python main.py \
  --gpu_ids 0 \
  --split_type Normal \
  --dataset_name BRATS2023 \
  --model_name UniME \
  --batch_size 4 \
  --amp \
  --use_ema \
  --compile \
  --base_lr 3e-4 \
  --layer_decay 0.75 \
  --iter_per_epoch 250 \
  --num_epochs 600 \
  --crop_size 96 \
  --seed 42 \
  --uni_encoder_name UniEncoder \
  --uni_encoder_checkpoint log_pretrain/BRATS2023-42-Normal/UniEncoder/ema_best_checkpoint.pth \
  --wandb_mode online \
  --wandb_project UniME \
  --wandb_run_name UniME-BRATS2023-Finetune
