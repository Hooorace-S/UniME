#!/bin/bash

python pretrain.py  \
    --gpu_ids 0 \
    --model_name UniEncoderBase \
    --batch_size 8  \
    --num_workers 8 \
    --persistent_workers \
    --prefetch_factor 2 \
    --amp \
    --compile \
    --crop_size 96 \
    --dataset_name "BRATS2023" \
    --mask_ratio 0.75 \
    --seed 42 \
    --warmup_lr 1e-5 \
    --warmup_ratio 0.05 \
    --base_lr 5e-4 \
    --min_lr 1e-5 \
    --use_ema \
    --ema_validate_interval 1 \
    --ema_validate_start_epoch 500 \
    --iter_per_epoch 500 \
    --wandb_mode online \
    --wandb_project UniME \
    --wandb_run_name UniME-BRATS2023-Pretrain


# Primary finetune path: use Accelerate for multi-GPU DDP.
# Choose devices via shell-level CUDA_VISIBLE_DEVICES or your local Accelerate config.
# Single-GPU fallback: replace the launcher with `python main.py` and keep `--gpu_ids`.

# accelerate launch main.py \
python main.py \
  --gpu_ids 0 \
  --split_type Normal \
  --dataset_name BRATS2023 \
  --model_name UniMEBase \
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
  --uni_encoder_name UniEncoderBase \
  --uni_encoder_checkpoint log_pretrain/BRATS2023-42-Normal/UniEncoderBase/ema_best_checkpoint.pth \
  --wandb_mode online \
  --wandb_project UniME \
  --wandb_run_name UniME-BRATS2023-Finetune
