import argparse
from dataclasses import dataclass, asdict
import os
from typing import Any, Literal
import yaml

SplitType = Literal['Normal', 'Split1', 'Split2', 'Split3']


def _str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


@dataclass
class PretrainConfig:
    """
    Masking pretraining config (matching the reference behavior).
    """
    # Data
    batch_size: int = 4
    data_path: str = './data'
    dataset_name: str = 'BRATS2018'
    split_type: SplitType = 'Normal'
    train_ratio: Literal[50, 20, 10, 5] | None = None
    crop_size: int = 128  # reference uses 128^3 crops

    # Logging
    log_root: str = 'log_pretrain'

    # wandb settings
    wandb_mode: str = "online"
    wandb_project: str = "U-Bench"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_group: str | None = None

    # Optim / schedule
    base_lr: float = 5e-4       # reference default
    warmup_lr: float = 1e-5
    min_lr: float = 1e-6
    warmup_ratio: float = 0.05
    weight_decay: float = 1e-4   # reference used 0 for Adam (can be 1e-4 safely)
    num_epochs: int = 600
    iter_per_epoch: int = 250
    log_freq: int = 5

    # Loader
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2
    profile_data_pipeline: bool = False
    profile_warmup_steps: int = 20

    # System
    gpu_ids: str = "0"
    seed: int = 1024

    # AMP
    amp: bool = True
    bfloat16: bool = False  # reference used fp16 O1; keep bf16 off by default

    # compile, flash_attn (kept for parity)
    compile: bool = True
    flash_attn: bool = False

    # EMA
    use_ema: bool = False
    ema_decay: float = 0.999
    ema_validate_interval: int = 1
    ema_validate_start_epoch: int = 500

    # Masking specifics (reference behavior) ---
    mask_ratio: float = 0.75
    num_mask_modalities: int = 1  # max modal dropping during masking
    original_shape: tuple[int, int, int] = (155, 240, 240)  # BraTS canonical
    deep_supervised: bool = False

    # Loss
    regulization_rate: float = 0.005

    # Descriptor name for log folder
    model_name: str = 'M3AE'

    @classmethod
    def from_args(cls) -> "PretrainConfig":
        """Create configuration from command line arguments"""
        p = argparse.ArgumentParser(description="masking pretraining (BraTS).")

        # data
        p.add_argument(
            '--batch_size',
            type=int,
            default=4,
            help='Per-process batch size. Global effective batch size scales with Accelerate world size.',
        )
        p.add_argument('--data_path', type=str, default='./data')
        p.add_argument('--dataset_name', type=str, default='BRATS2021')
        p.add_argument('--split_type', choices=['Normal', 'Split1', 'Split2', 'Split3'], default='Normal')
        p.add_argument('--train_ratio', type=int, choices=[50, 20, 10, 5], default=None)
        p.add_argument('--crop_size', type=int, default=128)

        # logging
        p.add_argument('--log_root', type=str, default='log_pretrain')

        p.add_argument(
            '--wandb_mode',
            type=str,
            default="online",
            choices=["disabled", "online", "offline"],
            help='W&B mode: disabled/online/offline',
        )
        p.add_argument(
            '--wandb_project',
            type=str,
            default="U-Bench",
            help='W&B project name'
        )
        p.add_argument(
            '--wandb_entity',
            type=str,
            default=None,
            help='W&B entity (team/user), optional'
        )
        p.add_argument(
            '--wandb_run_name',
            type=str,
            default=None,
            help='W&B run name, optional'
        )
        p.add_argument(
            '--wandb_group',
            type=str,
            default=None,
            help='W&B group name, optional'
        )

        # optim/schedule
        p.add_argument('--base_lr', type=float, default=5e-4)
        p.add_argument('--warmup_lr', type=float, default=1e-5)
        p.add_argument('--min_lr', type=float, default=1e-6)
        p.add_argument('--warmup_ratio', type=float, default=0.05)
        p.add_argument('--weight_decay', type=float, default=1e-4)
        p.add_argument('--num_epochs', type=int, default=600)
        p.add_argument(
            '--iter_per_epoch',
            type=int,
            default=250,
            help='Optimizer steps per process per epoch.',
        )
        p.add_argument('--log_freq', type=int, default=5)

        # loader
        p.add_argument('--num_workers', type=int, default=4)
        p.add_argument('--persistent_workers', dest='persistent_workers', action='store_true')
        p.add_argument('--no-persistent_workers', dest='persistent_workers', action='store_false')
        p.set_defaults(persistent_workers=True)
        p.add_argument('--prefetch_factor', type=int, default=2)
        p.add_argument('--profile_data_pipeline', dest='profile_data_pipeline', action='store_true')
        p.add_argument('--no-profile_data_pipeline', dest='profile_data_pipeline', action='store_false')
        p.set_defaults(profile_data_pipeline=False)
        p.add_argument('--profile_warmup_steps', type=int, default=20)

        # system
        p.add_argument(
            '--gpu_ids',
            type=str,
            default="0",
            help='GPU ids for direct python launches; ignored when launched by Accelerate.',
        )
        p.add_argument('--seed', type=int, default=1024)

        # amp
        p.add_argument('--amp', dest='amp', action='store_true')
        p.add_argument('--no-amp', dest='amp', action='store_false')
        p.set_defaults(amp=True)
        p.add_argument('--bfloat16', dest='bfloat16', action='store_true')
        p.add_argument('--no-bfloat16', dest='bfloat16', action='store_false')
        p.set_defaults(bfloat16=False)

        # compile/flash
        p.add_argument('--compile', dest='compile', action='store_true')
        p.add_argument('--no-compile', dest='compile', action='store_false')
        p.set_defaults(compile=False)
        p.add_argument('--flash_attn', dest='flash_attn', action='store_true')
        p.add_argument('--no-flash_attn', dest='flash_attn', action='store_false')
        p.set_defaults(flash_attn=False)

        # EMA
        p.add_argument(
            '--use_ema',
            type=_str_to_bool,
            nargs='?',
            const=True,
            default=False,
            help='Whether to enable EMA.',
        )
        p.add_argument(
            '--ema_decay',
            type=float,
            default=0.999,
            help='The decay rate for EMA updates.',
        )
        p.add_argument(
            '--ema_validate_interval',
            type=int,
            default=1,
            help='After the specified start epoch, run EMA validation every N epochs.',
        )
        p.add_argument(
            '--ema_validate_start_epoch',
            type=int,
            default=500,
            help='The epoch from which EMA validation begins.',
        )

        # masking
        p.add_argument('--mask_ratio', type=float, default=0.75)
        p.add_argument('--num_mask_modalities', type=int, default=1)
        p.add_argument('--original_shape', type=int, nargs=3, default=(160, 180, 210))

        # loss
        p.add_argument('--regulization_rate', type=float, default=0.005)

        # model name
        p.add_argument('--model_name', type=str, default='M3AE')

        args = p.parse_args()
        return cls(**vars(args))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def to_yaml(self, file_path: str | None) -> None:
        """Save configuration to yaml file"""
        if file_path is None:
            return
        if os.path.isdir(file_path) or file_path.endswith('/'):
            file_path = os.path.join(file_path, 'config.yaml')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)

    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.to_dict().items())


def get_pretrain_args() -> PretrainConfig:
    """Get arguments from command line"""
    return PretrainConfig.from_args()
