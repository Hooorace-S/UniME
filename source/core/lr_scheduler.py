from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler

if TYPE_CHECKING:
    from source.config import TrainingConfig
    from source.pretrain.parse import PretrainConfig


def setup_scheduler(
    args: TrainingConfig | PretrainConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int | None = None,
) -> Scheduler:
    """
    Build learning rate scheduler in step-level:
        - Linear warmup: warmup_lr_init -> base_lr, default to 5% of total steps
        - Cosine decay: base_lr -> min_lr

    Args:
        args (TrainingConfig): Training configuration
        optimizer (torch.optim.Optimizer): Optimizer
        steps_per_epoch (int): Actual steps per epoch, if None uses args.iter_per_epoch

    Returns:
        Scheduler: Learning rate scheduler
    """
    if steps_per_epoch is None:
        steps_per_epoch = args.iter_per_epoch

    num_steps = int(args.num_epochs * steps_per_epoch)
    warmup_steps = int(args.warmup_ratio * num_steps)

    lr_scheduler = CosineLRScheduler(
        optimizer=optimizer,
        t_initial=num_steps,
        lr_min=args.min_lr,
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False
    )

    return lr_scheduler


def setup_lldr_scheduler(
    args: TrainingConfig,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int | None = None
) -> Scheduler:
    """
    Build the cosine scheduler for Uni-Encoder when LLDR is enabled.

    This wraps ``setup_scheduler`` to keep the existing behavior while
    providing a dedicated entry-point that can evolve independently if the
    LLDR pathway requires custom scheduling logic in the future.
    """
    return setup_scheduler(args, optimizer, steps_per_epoch)
