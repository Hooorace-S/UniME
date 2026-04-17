# UniME for Brain Tumor Segmentation with Missing Modalities

Official implementation of **Uni-Encoder Meets Multi-Encoders: Representation Before Fusion for Brain Tumor Segmentation with Missing Modalities**.

<div align="center">
  <img src="figure/fig2.png" alt="Overview" width="80%">
</div>


***UniME overview.** Stage 1 pretrains a single ViT Uni-Encoder with masked self-supervision and a lightweight auxiliary decoder (discarded after pretraining). Stage 2 introduces parallel modality-specific encoders and performs multi-scale feature fusion for segmentation. <img src="figure/icon/archive.png" alt="Archive" width="16"/> marks modules initialized from pretrained weights, while <img src="figure/icon/flame.png" alt="Flame" width="16"/> indicates modules trained from scratch.*

This repository now ships multiple released model scales in addition to the original paper configuration. `UniME` with `UniEncoder` corresponds to the original setting, while `UniMEBase`, `UniMESmall`, `UniMETiny`, and `UniMENano` are released scale variants with tuned training scripts and hyperparameters.

## Model Variants

| Segmentation model | Pretrain encoder | Description |
| --- | --- | --- |
| `UniME` | `UniEncoder` | Original configuration and reproduction path |
| `UniMEBase` | `UniEncoderBase` | Released Base variant with tuned training scripts |
| `UniMESmall` | `UniEncoderSmall` | Released Small variant with tuned training scripts |
| `UniMETiny` | `UniEncoderTiny` | Released Tiny variant with tuned training scripts |
| `UniMENano` | `UniEncoderNano` | Released Nano variant with tuned training scripts |

## Pre-requisites

#### 1. Set Up the Python Environment

  ```bash
conda create -n unime python=3.13.5 -y
conda activate unime
conda install nvidia::cuda-toolkit==12.8.1 -y
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt
  ```

#### 2. Prepare Datasets

Download the **BraTS 2023** dataset and process it using the provided script:

  ```bash
python scripts/brats23_process.py --raw-dir /path/to/raw/brats2023
  ```

After processing, organize the data in the following structure:

  ```text
.
└── data
    └── BRATS2023
        ├── seg/
        ├── vol/
        ├── train.txt
        ├── val.txt
        └── test.txt
  ```

where

  - `vol/` contains the processed MRI volumes.
  - `seg/` contains the corresponding segmentation labels.
  - `train.txt`, `val.txt`, and `test.txt` define the default dataset splits.

#### 3. Configuration

First, review the GPU and wandb settings in the shell scripts you plan to use. The codebase supports multi-GPU training through Hugging Face Accelerate. If you have not configured Accelerate yet, run `accelerate config` once and keep the generated settings locally. When launching with Accelerate, select devices through shell-level `CUDA_VISIBLE_DEVICES` or your local Accelerate config; `--gpu_ids` remains as the direct `python ...` fallback knob. Under Accelerate, `--batch_size` is **per process**, so the effective global batch size is `batch_size * num_processes`. If you want to monitor training with [Weights & Biases](https://wandb.ai), configure your account:

  ```bash
export WANDB_API_KEY=your_wandb_api_key
  ```

Then update the project and entity fields in the scripts. Using wandb will introduce **additional training overhead**. If you prefer not to use it, simply set:

  ```code
--wandb_mode disabled
  ```

in the scripts to avoid logging.

## Training Scripts

The repository provides both standalone stage scripts and one-click end-to-end scripts.

| Script | What it runs | When to use it |
| --- | --- | --- |
| `scripts/pipeline/` | End-to-end launchers that run pretraining first and then fine-tuning for a matched model scale | Use when you want a one-command workflow |
| `scripts/pipeline/train_base.sh` | Pretrains `UniEncoderBase`, then fine-tunes `UniMEBase` | Released `Base` variant |
| `scripts/pipeline/train_small.sh` | Pretrains `UniEncoderSmall`, then fine-tunes `UniMESmall` | Released `Small` variant |
| `scripts/pipeline/train_tiny.sh` | Pretrains `UniEncoderTiny`, then fine-tunes `UniMETiny` | Released `Tiny` variant |
| `scripts/pipeline/train_nano.sh` | Pretrains `UniEncoderNano`, then fine-tunes `UniMENano` | Released `Nano` variant |
| `scripts/pipeline/train_original.sh` | Pretrains `UniEncoder`, then fine-tunes `UniME` | Reproduces the original model setting |
| `scripts/pretrain.sh` | Runs only stage-1 pretraining for `UniEncoder` | Standalone pretraining |
| `scripts/finetune.sh` | Runs only stage-2 fine-tuning for `UniME` | Standalone fine-tuning from an existing pretrained checkpoint |

## Script Usage

#### 1. Pre-training

Run the pretraining stage with:

  ```bash
chmod +x ./scripts/pretrain.sh
bash ./scripts/pretrain.sh
  ```

`scripts/pretrain.sh` executes the pretraining stage independently. By default it launches `pretrain.py` for `UniEncoder`. If you want to switch to multi-GPU training, replace the direct `python pretrain.py ...` command with the commented `accelerate launch pretrain.py ...` pattern in the script. If you want a single-GPU fallback without Accelerate, keep the direct `python pretrain.py` command and point `--gpu_ids` to the device you want to use.

#### 2. Fine-tuning

Before fine-tuning, point `--uni_encoder_checkpoint` in `scripts/finetune.sh` at the saved pretraining checkpoint. We recommend using the `ema_best_checkpoint.pth` produced by pretraining. `scripts/finetune.sh` executes the fine-tuning stage independently and does not automatically run pretraining first, so the checkpoint path should always be reviewed before launch. You do not need to edit `models/UniME/networks.py` for the fine-tune path. After updating the checkpoint path, run:

  ```bash
chmod +x ./scripts/finetune.sh
bash ./scripts/finetune.sh
  ```

#### 3. One-Click End-to-End Runs
If you want a matched pretraining + fine-tuning workflow in one command, use the scripts in `scripts/pipeline/`. These scripts first run pretraining for the corresponding `UniEncoder*`, then launch fine-tuning for the matching `UniME*`.

To reproduce the original `UniEncoder` + `UniME` configuration described in the paper, run:

```bash
bash ./scripts/pipeline/train_original.sh
```

To run the released scale variants with tuned training scripts and hyperparameters, run:

```bash
bash ./scripts/pipeline/train_base.sh
bash ./scripts/pipeline/train_small.sh
bash ./scripts/pipeline/train_tiny.sh
bash ./scripts/pipeline/train_nano.sh
```


#### 4. Gradient Checkpointing

To better balance training speed and GPU memory usage across different hardware setups, the codebase supports gradient checkpointing in the encoder, regularizer, and decoder modules. These switches are configured in `models/__init__.py` through `MODEL_CHECKPOINT_CONFIGS`.

## Acknowledgements

We would like to thank the following repositories for their valuable contributions:

- [RFNet](https://github.com/dyh127/RFNet.git)
- [mmFormer](https://github.com/YaoZhang93/mmFormer.git)
- [M3AE](https://github.com/zhjohnchan/M3AE.git)
- [MONAI](https://github.com/Project-MONAI/MONAI.git)
- [medpy](https://github.com/loli/medpy.git)

Our implementation is inspired by and builds upon the codebases of these repositories.