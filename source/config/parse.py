from dataclasses import dataclass, asdict
import argparse
from typing import Any, Literal, cast

import os
import yaml

SplitType = Literal['Normal', 'Split1', 'Split2', 'Split3']


@dataclass
class TrainingConfig:
    """
    Training configuration for incomplete multi-modal segmentation.
    """
    # data settings
    batch_size: int = 4
    data_path: str = './data'
    dataset_name: str = 'BRATS2018'
    split_type: SplitType = 'Normal'
    train_ratio: Literal[50, 20, 10, 5] | None = None

    # model settings
    model_name: str = 'mmFormer'
    num_classes: int = 4
    crop_size: int = 80
    uni_encoder_name: str | None = None
    uni_encoder_checkpoint: str | None = None

    # save settings
    log_root: str | None = None

    # wandb settings
    wandb_mode: str = "online"
    wandb_project: str = "U-Bench"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None

    # training settings
    base_lr: float = 2e-4
    warmup_lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_ratio: float = 0.05

    weight_decay: float = 1e-4
    layer_decay: float | None = None

    num_epochs: int = 1000
    iter_per_epoch: int = 300
    log_freq: int = 5

    # dataloader settings
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2

    gpu_ids: str = ""  # e.g., "0,1,3" to use GPU 0,1,3
    seed: int = 1024

    # Auxiliary settings
    auxiliary_start_epoch: int = 0

    # mixed precision training settings
    amp: bool = True
    bfloat16: bool = True  # whether to use bfloat16, only valid when amp is True

    # complie setting
    compile: bool = True  # whether to use torch.compile

    # top-k model management
    top_k: int = 3

    # EMA settings
    use_ema: bool = False
    ema_decay: float = 0.999

    @classmethod
    def from_args(cls) -> "TrainingConfig":
        """Create configuration from command line arguments"""
        parser = argparse.ArgumentParser(
            description='Training script for incomplete multi-modal tumor segmentation'
        )

        # set data settings
        parser.add_argument('--batch_size', default=1, type=int, help='batch size')
        parser.add_argument('--data_path', default='./data', type=str, help='data path')
        parser.add_argument('--dataset_name', default='BRATS2018', type=str, help='dataset name')
        parser.add_argument(
            '--split_type', default='Split1', choices=['Normal', 'Split1', 'Split2', 'Split3'], help='split type'
        )
        parser.add_argument(
            '--train_ratio', default=None, type=int, choices=[50, 20, 10, 5],
            help='Few-shot train ratio for BRATS2020/BRATS2021 (use full set when omitted)'
        )

        # set model settings
        parser.add_argument('--model_name', default='mmFormer', type=str, help='model name')
        parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
        parser.add_argument('--crop_size', default=80, type=int, help='crop size')
        parser.add_argument(
            '--uni_encoder_name',
            default=None,
            type=str,
            help='Uni-Encoder name resolved via the default pretrain checkpoint layout',
        )
        parser.add_argument(
            '--uni_encoder_checkpoint',
            default=None,
            type=str,
            help='Explicit Uni-Encoder checkpoint path; overrides --uni_encoder_name when both are provided',
        )

        # set save settings
        parser.add_argument('--log_root', default='log', type=str, help='checkpoints save path')

        parser.add_argument(
            '--wandb_mode',
            type=str,
            default="online",
            choices=["disabled", "online", "offline"],
            help='W&B mode: disabled/online/offline',
        )
        parser.add_argument(
            '--wandb_project',
            type=str,
            default="U-Bench",
            help='W&B project name'
        )
        parser.add_argument(
            '--wandb_entity',
            type=str,
            default=None,
            help='W&B entity (team/user), optional'
        )
        parser.add_argument(
            '--wandb_run_name',
            type=str,
            default=None,
            help='W&B run name, optional'
        )
        parser.add_argument(
            '--wandb_group',
            type=str,
            default=None,
            help='W&B group name, optional'
        )

        # set training settings
        parser.add_argument('--base_lr', default=2e-4, type=float, help='base learning rate')
        parser.add_argument('--warmup_lr', default=1e-5, type=float, help='warmup learning rate')
        parser.add_argument('--min_lr', default=1e-6, type=float, help='min learning rate')
        parser.add_argument('--warmup_ratio', default=0.05, type=float, help='warmup ratio')

        parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
        parser.add_argument(
            '--layer_decay',
            default=None,
            type=float,
            help='layer-wise learning rate decay for Uni-Encoder finetuning (disabled when omitted)'
        )

        parser.add_argument('--num_epochs', default=1000, type=int, help='training epochs')
        parser.add_argument('--iter_per_epoch', default=300, type=int, help='iterations per epoch')
        parser.add_argument('--log_freq', default=5, type=int, help='log frequency')

        parser.add_argument('--gpu_ids', type=str, default="0", help='GPU ids')
        parser.add_argument('--seed', default=1024, type=int, help='random seed')

        # set auxiliary settings
        parser.add_argument('--auxiliary_start_epoch', default=0, type=int, help='use auxiliarier')

        # set mixed precision training settings
        parser.add_argument('--amp', dest='amp', action='store_true', help='use mixed precision training')
        parser.add_argument('--bfloat16', dest='bfloat16', action='store_true', help='use bfloat16')
        parser.add_argument('--no-amp', dest='amp', action='store_false', help='disable mixed precision training')
        parser.set_defaults(amp=True)

        # set compile settings
        parser.add_argument('--compile', dest='compile', action='store_true', help='use torch.compile')
        parser.add_argument('--no-compile', dest='compile', action='store_false', help='disable torch.compile')
        parser.set_defaults(compile=True)

        # dataloader settings
        parser.add_argument('--num_workers', default=4, type=int, help='number of dataloader workers')
        parser.add_argument('--persistent_workers', dest='persistent_workers', action='store_true')
        parser.add_argument('--no-persistent_workers', dest='persistent_workers', action='store_false')
        parser.set_defaults(persistent_workers=True)
        parser.add_argument('--prefetch_factor', default=2, type=int, help='prefetch factor for dataloader workers')

        # top-k model management
        parser.add_argument('--top_k', default=3, type=int, help='number of top models to keep')

        # EMA settings
        parser.add_argument(
            '--use_ema',
            action='store_true',
            help='use EMA weights for validation and checkpoint saving'
        )
        parser.add_argument(
            '--ema_decay',
            default=0.999,
            type=float,
            help='EMA decay rate'
        )

        args = parser.parse_args()

        return cls(**vars(args))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_yaml(self, file_path: str | None = None) -> None:
        """
        Save configuration to yaml file.

        Args:
            file_path (str, optional): Path to the yaml file. Defaults to None.

        Returns:
            str: Message indicating the configuration is saved.
        """
        if file_path is None:
            return
        config_dict = self.to_dict()
        path = cast(str, file_path)

        # check if path is directory or file
        if os.path.isdir(path) or path.endswith('/'):
            # if path is directory, add default file name
            path = os.path.join(path, 'config.yaml')

        # ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, sort_keys=False, default_flow_style=False)

    def __str__(self) -> str:
        """Print configuration information"""
        return "\n".join(f"{k}: {v}" for k, v in self.to_dict().items())


def get_args() -> TrainingConfig:
    """
    Get arguments from command line
    """
    return TrainingConfig.from_args()
