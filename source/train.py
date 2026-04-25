import os
import time
from contextlib import contextmanager
from typing import Dict, Iterator, cast

from accelerate import Accelerator
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from source.config import TrainingConfig
from source.core import (
    setup_optimizer,
    setup_scheduler,
    setup_lldr_optimizer,
    setup_lldr_scheduler,
    LossStrategy
)
from source.logger import Logger
from source.utils import wandb_utils, unwrap_model
from source.utils.runtime import clone_state_dict, set_model_is_training
from source.validate import validate_model


def _swap_named_tensors(
    named_tensors: Iterator[tuple[str, torch.Tensor]],
    shadow_tensors: Dict[str, torch.Tensor]
) -> None:
    """
    Swap model tensors with shadow tensors in-place.
    """
    for name, tensor in named_tensors:
        shadow_tensor = shadow_tensors.get(name)
        if shadow_tensor is None:
            shadow_tensors[name] = tensor.detach().clone()
            continue
        swap_buffer = tensor.detach().clone()
        tensor.copy_(shadow_tensor.to(device=tensor.device, dtype=tensor.dtype))
        shadow_tensor.copy_(swap_buffer)


class _EMAWeights:
    """
    Keep EMA parameters/buffers independent from trainable model weights.
    """
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow_params: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self.shadow_buffers: Dict[str, torch.Tensor] = {
            name: buffer.detach().clone()
            for name, buffer in model.named_buffers()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        one_minus_decay = 1.0 - self.decay
        for name, param in model.named_parameters():
            shadow_param = self.shadow_params.get(name)
            if shadow_param is None:
                self.shadow_params[name] = param.detach().clone()
                continue
            shadow_param.mul_(self.decay).add_(param.detach(), alpha=one_minus_decay)

        for name, buffer in model.named_buffers():
            shadow_buffer = self.shadow_buffers.get(name)
            if shadow_buffer is None:
                self.shadow_buffers[name] = buffer.detach().clone()
                continue
            shadow_buffer.copy_(buffer.detach())

    @contextmanager
    def use_ema_weights(self, model: torch.nn.Module):
        with torch.no_grad():
            _swap_named_tensors(model.named_parameters(), self.shadow_params)
            _swap_named_tensors(model.named_buffers(), self.shadow_buffers)
        try:
            yield
        finally:
            with torch.no_grad():
                _swap_named_tensors(model.named_parameters(), self.shadow_params)
                _swap_named_tensors(model.named_buffers(), self.shadow_buffers)


class _NullSummaryWriter:
    """Rank-local TensorBoard no-op used outside the main process."""

    def add_scalar(self, *args, **kwargs) -> None:
        _ = args, kwargs

    def close(self) -> None:
        return None


def _to_metric_tensor(value: torch.Tensor | float | int | None, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return torch.tensor(float(value), device=device, dtype=torch.float32)
    return value.detach().float()


def train_brats(
    args: TrainingConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    validation_results_csv: str,
    logger: Logger,
    accelerator: Accelerator
) -> list[tuple[int, str, float]]:
    """
    Train the incomplete multi-modal segmentation model with BRATS dataset

    Args:
        args (TrainingConfig): Training configuration
        model (nn.Module): model class
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        validation_results_csv (str): Path to validation results CSV file
        logger (Logger): Logger instance

    Returns:
        list: List of checkpoints for testing as tuples (epoch, checkpoint_path, composite_score),
            containing top-k checkpoints plus the final-epoch checkpoint.
    """
    assert args.log_root is not None
    # Setup directories
    log_dir = os.path.join(
        args.log_root, f"{args.dataset_name}-{args.seed}-{args.split_type}", args.model_name
    )
    tensorboard_dir = os.path.join(log_dir, "tensorboard")
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Save configuration
    if accelerator.is_main_process:
        args.to_yaml(file_path=os.path.join(log_dir, 'config.yaml'))

    # Setup optimizer and scheduler
    unime_series = args.model_name.lower().startswith("unime")
    layer_decay_value = getattr(args, "layer_decay", None)

    if unime_series and layer_decay_value is not None:
        configure_layer_decay = getattr(model, "configure_layer_decay", None)
        if callable(configure_layer_decay):
            configure_layer_decay(layer_decay_value)
        elif hasattr(model, "layer_decay"):
            setattr(model, "layer_decay", layer_decay_value)

    use_unime_lldr = (
        unime_series
        and layer_decay_value is not None
        and bool(getattr(model, "layerwise_lr_decay_enabled", False))
    )

    if use_unime_lldr:
        assert layer_decay_value is not None
        optimizer = setup_lldr_optimizer(args, model)
        logger.write(f"LLDR enabled for UniME (layer_decay={layer_decay_value:.4f}).\n")
        layer_schedule = getattr(optimizer, "unime_layer_schedule", None)
        non_uni_encoder_names = getattr(optimizer, "unime_non_uni_encoder_names", None)
        if layer_schedule and accelerator.is_main_process:
            decay_config_path = os.path.join(log_dir, "decay_config.yaml")
            decay_payload = {
                "layer_decay": float(layer_decay_value),
                "layer_schedule": [
                    {
                        "layer_id": entry["layer_id"],
                        "label": entry["label"],
                        "lr_scale": float(entry["lr_scale"]),
                        "parameter_names": entry["parameter_names"],
                    }
                    for entry in layer_schedule
                ],
            }
            if non_uni_encoder_names:
                decay_payload["non_uni_encoder_parameters"] = non_uni_encoder_names
            with open(decay_config_path, "w", encoding="utf-8") as decay_file:
                yaml.safe_dump(decay_payload, decay_file, sort_keys=False)
    else:
        optimizer = setup_optimizer(args, model)
        if unime_series and layer_decay_value is not None and not getattr(model, "layerwise_lr_decay_enabled", False):
            logger.write("Layer decay provided but UniME is not in finetune mode; LLDR disabled.\n")

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    scheduler = (
        setup_lldr_scheduler(args, optimizer, steps_per_epoch=args.iter_per_epoch)
        if use_unime_lldr else
        setup_scheduler(args, optimizer, steps_per_epoch=args.iter_per_epoch)
    )

    # Setup loss function
    loss_function = LossStrategy(required_auxiliary=False).to(accelerator.device)

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=tensorboard_dir) if accelerator.is_main_process else _NullSummaryWriter()
    base_model = unwrap_model(model)

    use_ema = bool(getattr(args, "use_ema", False))
    ema_weights: _EMAWeights | None = None
    if use_ema:
        if not (0.0 < float(args.ema_decay) < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {args.ema_decay}.")
        ema_weights = _EMAWeights(model=base_model, decay=float(args.ema_decay))
        logger.write(f"EMA enabled: decay={float(args.ema_decay):.6f}\n")
    else:
        logger.write("EMA disabled.\n")
    logger.flush()

    # Top-k checkpoint management
    topk_checkpoints: list[tuple[int, str, float]] = []
    final_checkpoint: tuple[int, str, float] | None = None

    # Training loop
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        # Check if we should use auxiliary losses
        if epoch >= args.auxiliary_start_epoch:
            loss_function.required_auxiliary = True

        # Training phase
        model.train()
        set_model_is_training(model, True)

        epoch_start_time = time.time()

        # Accumulate epoch losses
        epoch_losses = {
            'fusion': torch.zeros((), device=accelerator.device, dtype=torch.float32),
            'deep': torch.zeros((), device=accelerator.device, dtype=torch.float32),
            'aux': torch.zeros((), device=accelerator.device, dtype=torch.float32),
            'total': torch.zeros((), device=accelerator.device, dtype=torch.float32)
        }
        batch_count = 0

        # Create cycling iterator for train_loader
        train_iter = iter(train_loader)

        # Progress bar for training iterations
        pbar = tqdm(
            range(args.iter_per_epoch),
            desc=f"Epoch {epoch}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        current_lr = optimizer.param_groups[0]['lr']

        for i in pbar:
            # Calculate step for this iteration
            step = (i + 1) + (epoch - 1) * args.iter_per_epoch

            # Get next batch, cycling if necessary
            try:
                batch_data = next(train_iter)
            except StopIteration:
                # Reset iterator if we've exhausted the dataset
                train_iter = iter(train_loader)
                batch_data = next(train_iter)

            # Parse batch data
            images, labels, masks, _ = batch_data

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with AMP
            with accelerator.autocast():
                fusion_output, auxiliary_outputs, deep_supervision_outputs = model(images, masks)

                # Calculate losses
                loss_fusion, loss_deep, loss_aux = loss_function(
                    fusion_output=fusion_output,
                    deep_supervision_outputs=deep_supervision_outputs,
                    auxiliary_outputs=auxiliary_outputs,
                    target=labels
                )

                # Total loss
                loss_total = loss_fusion + loss_deep + loss_aux

            if torch.isnan(loss_total) or torch.isinf(loss_total):
                accelerator.print("NaN or Inf detected in loss, skipping this batch and freeing graph.")
                scheduler.step_update(step - 1)
                try:
                    del fusion_output, deep_supervision_outputs, auxiliary_outputs
                    del loss_total, loss_fusion, loss_deep, loss_aux
                    del batch_data
                except Exception as exc:
                    accelerator.print(f"Error during cleanup: {exc}")
                continue

            # Backward pass
            grad_norm: torch.Tensor | float | int | None = None
            accelerator.backward(loss_total)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_step_taken = not bool(getattr(accelerator, "optimizer_step_was_skipped", False))

            if ema_weights is not None and optimizer_step_taken:
                ema_weights.update(base_model)

            # Step-level scheduler update (use step directly as it's already calculated correctly)
            scheduler.step_update(step - 1)  # step_update expects 0-indexed
            global_step = step

            # Accumulate losses
            epoch_losses['fusion'] += loss_fusion.detach().float()
            epoch_losses['deep'] += loss_deep.detach().float()
            epoch_losses['aux'] += loss_aux.detach().float()
            epoch_losses['total'] += loss_total.detach().float()
            batch_count += 1

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # Update progress bar
            if accelerator.is_local_main_process:
                pbar.set_postfix({
                    'Loss': f'{loss_total.item():.4f}',
                    'Fusion': f'{loss_fusion.item():.4f}',
                    'Deep': f'{loss_deep.item():.4f}',
                    'Auxiliary': f'{loss_aux.item():.4f}',
                    'LR': f'{current_lr:.2e}'
                })

            # Logging at specified frequency
            if (i + 1) % args.log_freq == 0:
                reduced_loss_fusion = cast(
                    torch.Tensor,
                    accelerator.reduce(loss_fusion.detach().float(), reduction="mean")
                )
                reduced_loss_deep = cast(
                    torch.Tensor,
                    accelerator.reduce(loss_deep.detach().float(), reduction="mean")
                )
                reduced_loss_aux = cast(
                    torch.Tensor,
                    accelerator.reduce(loss_aux.detach().float(), reduction="mean")
                )
                reduced_loss_total = cast(
                    torch.Tensor,
                    accelerator.reduce(loss_total.detach().float(), reduction="mean")
                )

                reduced_grad_norm = _to_metric_tensor(grad_norm, accelerator.device)
                if reduced_grad_norm is not None:
                    reduced_grad_norm = cast(torch.Tensor, accelerator.reduce(reduced_grad_norm, reduction="mean"))

                if accelerator.is_main_process:
                    # Log to TensorBoard
                    writer.add_scalar('Loss/fusion', reduced_loss_fusion.item(), global_step)
                    writer.add_scalar('Loss/deep_supervision', reduced_loss_deep.item(), global_step)
                    writer.add_scalar('Loss/auxiliary', reduced_loss_aux.item(), global_step)
                    writer.add_scalar('Loss/total', reduced_loss_total.item(), global_step)
                    writer.add_scalar('Learning_Rate', current_lr, global_step)

        # Epoch summary
        epoch_time = time.time() - epoch_start_time

        # Calculate average losses
        reduced_batch_count = cast(
            torch.Tensor,
            accelerator.reduce(
                torch.tensor(float(batch_count), device=accelerator.device, dtype=torch.float32),
                reduction="sum"
            )
        ).clamp_min(1.0)

        avg_losses = {
            key: float(
                (
                    cast(torch.Tensor, accelerator.reduce(value, reduction="sum")) / reduced_batch_count
                ).item()
            )
            for key, value in epoch_losses.items()
        }

        # Log structured metrics for the epoch
        logger.log_metrics({
            'type': 'train',
            'epoch': epoch,
            'lr': current_lr,
            'total_loss': avg_losses['total'],
            'fusion_loss': avg_losses['fusion'],
            'deep_loss': avg_losses['deep'],
            'aux_loss': avg_losses['aux'],
            'time': epoch_time
        })
        wandb_utils.log(
            {
                "epoch": epoch,
                "train/epoch_avg_total_loss": avg_losses["total"],
                "train/epoch_avg_fusion_loss": avg_losses["fusion"],
                "train/epoch_avg_deep_loss": avg_losses["deep"],
                "train/epoch_avg_aux_loss": avg_losses["aux"],
                "train/epoch_time": epoch_time,
            },
        )

        # Log epoch summary to text log
        logger.write(
            f"Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"Avg Loss: {avg_losses['total']:.4f}\n"
        )
        logger.flush()

        # Validation
        if _valid_strategy(epoch, args.num_epochs):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.write(f"Validation at epoch {epoch}...\n")

                # Run validation and get composite score
                if ema_weights is not None:
                    with ema_weights.use_ema_weights(base_model):
                        main_writer = cast(SummaryWriter, writer)
                        dice_separate, dice_evaluation, composite_score = validate_model(
                            model=base_model,
                            args=args,
                            val_loader=val_loader,
                            epoch=epoch,
                            writer=main_writer,
                            logger=logger,
                            validation_results_csv=validation_results_csv
                        )
                        checkpoint_model_state = cast(
                            Dict[str, torch.Tensor],
                            accelerator.get_state_dict(model)
                        )
                        checkpoint_model_state = clone_state_dict(checkpoint_model_state)
                else:
                    main_writer = cast(SummaryWriter, writer)
                    dice_separate, dice_evaluation, composite_score = validate_model(
                        model=base_model,
                        args=args,
                        val_loader=val_loader,
                        epoch=epoch,
                        writer=main_writer,
                        logger=logger,
                        validation_results_csv=validation_results_csv
                    )
                    checkpoint_model_state = cast(
                        Dict[str, torch.Tensor],
                        accelerator.get_state_dict(model)
                    )
                    checkpoint_model_state = clone_state_dict(checkpoint_model_state)
                wandb_utils.log(
                    {
                        "epoch": epoch,
                        "val/dice_WT": float(dice_evaluation[0]),
                        "val/dice_TC": float(dice_evaluation[1]),
                        "val/dice_ET": float(dice_evaluation[2]),
                        "val/dice_ET_postpro": float(dice_evaluation[3]),
                        "val/composite_score": float(composite_score),
                    },
                )

                # Top-k checkpoint management
                checkpoint_path = os.path.join(
                    log_dir,
                    f"epoch_{epoch:04d}_composite_{composite_score:.6f}.pth"
                )

                checkpoint_payload = {
                    'epoch': epoch,
                    'model_state_dict': checkpoint_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'composite_score': composite_score,
                    'dice_evaluation': dice_evaluation,
                    'dice_separate': dice_separate
                }

                # Determine if we should save this checkpoint
                should_save = False
                if len(topk_checkpoints) < args.top_k:
                    should_save = True
                elif len(topk_checkpoints) > 0 and composite_score > topk_checkpoints[-1][2]:
                    # Remove worst checkpoint
                    worst_checkpoint = topk_checkpoints[-1]
                    if os.path.exists(worst_checkpoint[1]):
                        os.remove(worst_checkpoint[1])
                        logger.write(f"Removed checkpoint: {os.path.basename(worst_checkpoint[1])}\n")
                    topk_checkpoints.pop()
                    should_save = True

                if should_save:
                    # Save checkpoint
                    torch.save(checkpoint_payload, checkpoint_path)

                    logger.write(f"Saved checkpoint: {os.path.basename(checkpoint_path)}\n")
                    wandb_utils.log(
                        {
                            "epoch": epoch,
                            "checkpoint/epoch": int(epoch),
                            "checkpoint/composite_score": float(composite_score),
                        },
                    )
                    run = wandb_utils.run()

                    # Add to top-k list and sort
                    topk_checkpoints.append((epoch, checkpoint_path, composite_score))
                    topk_checkpoints.sort(key=lambda x: x[2], reverse=True)
                    if topk_checkpoints:
                        best_epoch, _best_path, best_score = topk_checkpoints[0]
                        if run is not None:
                            try:
                                run.summary["best_val_composite"] = float(best_score)
                                run.summary["best_epoch"] = int(best_epoch)
                            except Exception:
                                pass

                    # Log current top-k models
                    logger.write(f"Current top-{args.top_k} models:\n")
                    for i, (ep, path, score) in enumerate(topk_checkpoints):
                        logger.write(f"  {i+1}. Epoch {ep}: {score:.6f} saved at {path}\n")

                if epoch == args.num_epochs:
                    if not os.path.exists(checkpoint_path):
                        torch.save(checkpoint_payload, checkpoint_path)
                        logger.write(f"Saved final-epoch checkpoint: {os.path.basename(checkpoint_path)}\n")
                    else:
                        logger.write(f"Final-epoch checkpoint already saved: {os.path.basename(checkpoint_path)}\n")
                    final_checkpoint = (epoch, checkpoint_path, composite_score)

                logger.flush()
            accelerator.wait_for_everyone()

    # Training complete
    logger.write("Training completed!\n")
    logger.write(f"Total epochs: {args.num_epochs}\n")
    logger.write(f"Final model saved at: {log_dir}\n")

    # Close writer but don't close logger yet (will be closed in main)
    writer.close()

    checkpoints_for_test = topk_checkpoints.copy()
    if final_checkpoint is not None:
        existing_paths = {path for _, path, _ in checkpoints_for_test}
        if final_checkpoint[1] not in existing_paths:
            checkpoints_for_test.append(final_checkpoint)
            logger.write("Added final-epoch checkpoint to testing queue.\n")
        else:
            logger.write("Final-epoch checkpoint is already part of top-k queue.\n")
    else:
        logger.write("WARNING: Final-epoch checkpoint was not captured for testing.\n")
    logger.flush()

    # Return checkpoints to test (top-k + final epoch)
    return checkpoints_for_test if accelerator.is_main_process else []


def _valid_strategy(
    epoch: int,
    num_epochs: int
) -> bool:
    """
    Validate the model with the validation data loader
    """
    return (
        # (epoch <= 500 and epoch % 25 == 0) or
        (epoch > 450 and epoch % 10 == 0) or
        (epoch > num_epochs - 10 and epoch % 1 == 0) or
        epoch == num_epochs
    )
