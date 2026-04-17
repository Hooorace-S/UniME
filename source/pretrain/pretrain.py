import glob
import math
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Tuple, Mapping, cast

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from source.core import setup_optimizer, setup_scheduler
from source.logger import Logger
from source.pretrain.dataset import setup_pretrain_dataloader
from source.pretrain.loss_function import ReconstructionLoss
from source.pretrain.parse import PretrainConfig
from source.utils import clone_state_dict, unwrap_model, wandb_utils


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


def _should_run_ema_validation(args: PretrainConfig, epoch: int) -> bool:
    """
    Check whether EMA validation should run at this epoch.
    """
    return (
        epoch >= int(args.ema_validate_start_epoch)
        and (epoch - int(args.ema_validate_start_epoch)) % int(args.ema_validate_interval) == 0
    )


def _summarize_timings(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(float(v) for v in values)
    p95_idx = min(len(ordered) - 1, max(0, math.ceil(0.95 * len(ordered)) - 1))
    return {
        "mean": float(sum(ordered) / len(ordered)),
        "p95": float(ordered[p95_idx]),
        "max": float(ordered[-1]),
    }


def _sync_for_timing(enabled: bool, device: torch.device) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


class _NullSummaryWriter:
    """Rank-local TensorBoard no-op used outside the main process."""

    def add_scalar(self, *args, **kwargs) -> None:
        _ = args, kwargs

    def close(self) -> None:
        return None


def _evaluate_epoch_avg_recon_loss(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_fn: ReconstructionLoss,
    accelerator: Accelerator,
    desc: str
) -> float:
    """
    Evaluate epoch-average reconstruction loss over the full dataloader.
    """
    was_training = model.training
    model.eval()
    device = accelerator.device
    running_loss = torch.zeros((), device=device, dtype=torch.float32)
    batch_count = 0

    with torch.no_grad():
        pbar = tqdm(data_loader, desc=desc, leave=False, disable=not accelerator.is_local_main_process)
        for images, locs in pbar:
            images = images.to(device, non_blocking=True)
            locations = _pack_locations(locs, device)

            with accelerator.autocast():
                recon, prior = model(images, location=locations)
                target = images.to(dtype=recon.dtype)
                prior = prior.to(dtype=recon.dtype)
                loss = loss_fn(recon, prior, target)

            running_loss += loss.detach().float()
            batch_count += 1
            pbar.set_postfix({"Recon": f"{float(loss.detach().item()):.4f}"})

    if was_training:
        model.train()

    return float((running_loss / max(1, batch_count)).item())


def pretrain_brats(
    args: PretrainConfig,
    model: torch.nn.Module,
    train_loader: DataLoader,
    logger: Logger,
    accelerator: Accelerator,
) -> None:
    """
    Pretrain the model with the BRATS dataset.
    """
    log_dir = os.path.join(args.log_root, f"{args.dataset_name}-{args.seed}-{args.split_type}", args.model_name)
    tb_dir = os.path.join(log_dir, "tensorboard")
    if accelerator.is_main_process:
        os.makedirs(tb_dir, exist_ok=True)
        args.to_yaml(os.path.join(log_dir, "config.yaml"))

    optimizer = setup_optimizer(args, model)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    steps_per_epoch = len(train_loader)
    scheduler = setup_scheduler(args, optimizer, steps_per_epoch=steps_per_epoch)
    loss_fn = ReconstructionLoss(regulization_rate=args.regulization_rate, eps=1e-6).to(accelerator.device)

    writer = SummaryWriter(tb_dir) if accelerator.is_main_process else _NullSummaryWriter()

    device = accelerator.device
    profile_data_pipeline = bool(getattr(args, "profile_data_pipeline", False))
    profile_warmup_steps = max(0, int(getattr(args, "profile_warmup_steps", 0)))
    base_model = unwrap_model(model)

    logger.log_metrics({
        "type": "pretrain_dataset_info",
        "train_batches_per_rank": int(steps_per_epoch),
        "train_samples": len(train_loader.dataset),  # type: ignore[attr-defined]
        "batch_size_per_process": int(args.batch_size),
        "world_size": int(accelerator.num_processes),
    })
    wandb_utils.log(
        {
            "data/train_batches_per_rank": int(steps_per_epoch),
            "data/train_samples": len(train_loader.dataset),  # type: ignore[attr-defined]
            "data/batch_size_per_process": int(args.batch_size),
            "data/world_size": int(accelerator.num_processes),
        }
    )

    use_ema = bool(getattr(args, "use_ema", False))
    ema_weights: _EMAWeights | None = None
    ema_loader: DataLoader | None = None
    if use_ema:
        if not (0.0 < float(args.ema_decay) < 1.0):
            raise ValueError(f"ema_decay must be in (0, 1), got {args.ema_decay}.")
        if int(args.ema_validate_interval) <= 0:
            raise ValueError(f"ema_validate_interval must be > 0, got {args.ema_validate_interval}.")
        if int(args.ema_validate_start_epoch) <= 0:
            raise ValueError(f"ema_validate_start_epoch must be > 0, got {args.ema_validate_start_epoch}.")
        ema_weights = _EMAWeights(model=base_model, decay=float(args.ema_decay))
        logger.write(
            f"EMA enabled: decay={float(args.ema_decay):.6f}, "
            f"start_epoch={int(args.ema_validate_start_epoch)}, "
            f"interval={int(args.ema_validate_interval)}\n"
        )
    else:
        logger.write("EMA disabled.\n")
    logger.flush()

    best_ckpt_path = os.path.join(log_dir, "best_checkpoint.pth")
    last_ckpt_path = os.path.join(log_dir, "last_checkpoint.pth")
    best_train_loss = float("inf")
    best_epoch = 0
    ema_best_ckpt_path = os.path.join(log_dir, "ema_best_checkpoint.pth")
    ema_last_ckpt_path = os.path.join(log_dir, "ema_last_checkpoint.pth")
    ema_best_recon_loss = float("inf")
    ema_best_epoch = 0

    for legacy_ckpt_path in glob.glob(os.path.join(log_dir, "epoch_*.pth")):
        if not os.path.isfile(legacy_ckpt_path):
            continue
        try:
            os.remove(legacy_ckpt_path)
            logger.write(f"Removed legacy checkpoint: {os.path.basename(legacy_ckpt_path)}\n")
        except OSError as exc:
            logger.write(f"WARNING: Failed to remove legacy checkpoint {legacy_ckpt_path}: {exc}\n")

    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = torch.zeros((), device=device, dtype=torch.float32)
        batch_count = 0
        fetch_times: list[float] = []
        h2d_times: list[float] = []
        forward_backward_times: list[float] = []
        optimizer_times: list[float] = []

        train_iter = iter(train_loader)
        pbar = tqdm(
            range(steps_per_epoch),
            desc=f"Pretrain Epoch {epoch}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )

        for i in pbar:
            _sync_for_timing(profile_data_pipeline, device)
            fetch_start = time.perf_counter()
            try:
                images, locs = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, locs = next(train_iter)
            if profile_data_pipeline:
                fetch_times.append(time.perf_counter() - fetch_start)

            _sync_for_timing(profile_data_pipeline, device)
            h2d_start = time.perf_counter()
            images = images.to(device, non_blocking=True)
            locations = _pack_locations(locs, device)
            if profile_data_pipeline:
                _sync_for_timing(True, device)
                h2d_times.append(time.perf_counter() - h2d_start)

            _sync_for_timing(profile_data_pipeline, device)
            forward_backward_start = time.perf_counter()
            with accelerator.autocast():
                recon, prior = model(images, location=locations)
                target = images.to(dtype=recon.dtype)
                prior = prior.to(dtype=recon.dtype)
                loss = loss_fn(recon, prior, target)

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            if profile_data_pipeline:
                _sync_for_timing(True, device)
                forward_backward_times.append(time.perf_counter() - forward_backward_start)

            optimizer_start = time.perf_counter() if profile_data_pipeline else 0.0
            optimizer.step()
            optimizer_step_taken = not bool(getattr(accelerator, "optimizer_step_was_skipped", False))

            if ema_weights is not None and optimizer_step_taken:
                ema_weights.update(base_model)

            step_idx = (i + 1) + (epoch - 1) * steps_per_epoch
            scheduler.step_update(step_idx - 1)
            global_step = step_idx

            if profile_data_pipeline:
                _sync_for_timing(True, device)
                optimizer_times.append(time.perf_counter() - optimizer_start)

            loss_detached = loss.detach().float()
            running_loss += loss_detached
            batch_count += 1

            if (i + 1) % args.log_freq == 0:
                reduced_loss = cast(
                    torch.Tensor,
                    accelerator.reduce(loss_detached, reduction="mean")
                )
                loss_value = float(reduced_loss.item())
                if accelerator.is_main_process:
                    writer.add_scalar("Pretrain/Train_Loss", loss_value, global_step)
                    writer.add_scalar("Pretrain/LR", optimizer.param_groups[0]["lr"], global_step)
                if accelerator.is_local_main_process:
                    pbar.set_postfix({"Loss": f"{loss_value:.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.2e}"})

        reduced_batch_count = cast(
            torch.Tensor,
            accelerator.reduce(
                torch.tensor(float(batch_count), device=device, dtype=torch.float32),
                reduction="sum",
            )
        ).clamp_min(1.0)
        reduced_running_loss = cast(
            torch.Tensor,
            accelerator.reduce(running_loss, reduction="sum")
        )

        avg_loss = float((reduced_running_loss / reduced_batch_count).item())
        epoch_time = time.time() - epoch_start
        logger.log_metrics({
            "type": "pretrain",
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_recon_loss": avg_loss,
            "time": epoch_time,
        })

        profile_metrics: dict[str, float] = {}
        if profile_data_pipeline and accelerator.is_main_process:
            steady_fetch_times = fetch_times[min(profile_warmup_steps, len(fetch_times)):] or fetch_times
            fetch_summary = _summarize_timings(fetch_times)
            steady_fetch_summary = _summarize_timings(steady_fetch_times)
            h2d_summary = _summarize_timings(h2d_times)
            forward_backward_summary = _summarize_timings(forward_backward_times)
            optimizer_summary = _summarize_timings(optimizer_times)
            profile_metrics = {
                "first_batch_fetch_time": float(fetch_times[0]) if fetch_times else 0.0,
                "fetch_time_mean": fetch_summary["mean"],
                "fetch_time_p95": fetch_summary["p95"],
                "fetch_time_max": fetch_summary["max"],
                "fetch_time_steady_mean": steady_fetch_summary["mean"],
                "fetch_time_steady_p95": steady_fetch_summary["p95"],
                "fetch_time_steady_max": steady_fetch_summary["max"],
                "h2d_time_mean": h2d_summary["mean"],
                "forward_backward_time_mean": forward_backward_summary["mean"],
                "optimizer_time_mean": optimizer_summary["mean"],
            }
            logger.log_metrics({
                "type": "pretrain_profile",
                "epoch": epoch,
                **profile_metrics,
            })

        wandb_utils.log(
            {
                "epoch": epoch,
                "pretrain/epoch_avg_recon_loss": avg_loss,
                "pretrain/epoch_time": epoch_time,
                **{f"pretrain/{name}": value for name, value in profile_metrics.items()},
            },
        )
        profile_suffix = ""
        if profile_metrics:
            profile_suffix = (
                f" | fetch mean {profile_metrics['fetch_time_mean']:.4f}s"
                f" | fetch p95 {profile_metrics['fetch_time_p95']:.4f}s"
                f" | batch1 fetch {profile_metrics['first_batch_fetch_time']:.4f}s"
            )
        logger.write(f"[Pretrain] Epoch {epoch} | avg recon: {avg_loss:.4f} | time {epoch_time:.2f}s{profile_suffix}\n")
        logger.flush()
        writer.add_scalar("Pretrain/Train_Epoch_Loss", avg_loss, epoch)
        if profile_metrics:
            writer.add_scalar("Pretrain/Fetch_Time_Mean", profile_metrics["fetch_time_mean"], epoch)
            writer.add_scalar("Pretrain/Fetch_Time_P95", profile_metrics["fetch_time_p95"], epoch)
            writer.add_scalar("Pretrain/First_Batch_Fetch_Time", profile_metrics["first_batch_fetch_time"], epoch)

        accelerator.wait_for_everyone()
        is_best = float(avg_loss) < best_train_loss
        if accelerator.is_main_process:
            model_state_dict = cast(
                Mapping[str, torch.Tensor],
                accelerator.get_state_dict(model),
            )
            checkpoint_payload = {
                "epoch": epoch,
                "model_state_dict": clone_state_dict(model_state_dict),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": float(avg_loss),
            }

            torch.save(checkpoint_payload, last_ckpt_path)
            logger.write(f"Updated last checkpoint at epoch {epoch}: {os.path.basename(last_ckpt_path)}\n")

            if is_best:
                best_train_loss = float(avg_loss)
                best_epoch = epoch
                torch.save(checkpoint_payload, best_ckpt_path)
                logger.write(f"Updated best checkpoint at epoch {epoch}: {os.path.basename(best_ckpt_path)}\n")

        logger.flush()
        wandb_payload = {
            "epoch": epoch,
            "pretrain/last_checkpoint_epoch": int(epoch),
            "pretrain/last_checkpoint_train_loss": float(avg_loss),
        }
        if is_best:
            wandb_payload.update(
                {
                    "pretrain/best_checkpoint_epoch": int(best_epoch),
                    "pretrain/best_checkpoint_train_loss": float(best_train_loss),
                }
            )
        wandb_utils.log(wandb_payload)

        run = wandb_utils.run()
        if run is not None:
            try:
                run.summary["best_pretrain_loss"] = float(best_train_loss)
                run.summary["best_pretrain_epoch"] = int(best_epoch)
                run.summary["last_pretrain_loss"] = float(avg_loss)
                run.summary["last_pretrain_epoch"] = int(epoch)
            except Exception:
                pass
        accelerator.wait_for_everyone()

        if ema_weights is not None and _should_run_ema_validation(args, epoch):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.write(f"[Pretrain][EMA] Running validation at epoch {epoch}...\n")
                logger.flush()

                if ema_loader is None:
                    ema_loader = setup_pretrain_dataloader(args, full_dataset_pass=True)

                ema_avg_loss = float("inf")
                ema_is_best = False
                with ema_weights.use_ema_weights(base_model):
                    ema_avg_loss = _evaluate_epoch_avg_recon_loss(
                        model=base_model,
                        data_loader=ema_loader,
                        loss_fn=loss_fn,
                        accelerator=accelerator,
                        desc=f"EMA Validation Epoch {epoch}/{args.num_epochs}",
                    )
                    ema_checkpoint_payload = {
                        "epoch": epoch,
                        "model_state_dict": clone_state_dict(base_model.state_dict()),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": float(ema_avg_loss),
                        "checkpoint_variant": "ema",
                        "ema_decay": float(args.ema_decay),
                        "ema_validate_interval": int(args.ema_validate_interval),
                        "ema_validate_start_epoch": int(args.ema_validate_start_epoch),
                        "ema_epoch_avg_recon_loss": float(ema_avg_loss),
                    }
                    torch.save(ema_checkpoint_payload, ema_last_ckpt_path)

                    ema_is_best = float(ema_avg_loss) < ema_best_recon_loss
                    if ema_is_best:
                        ema_best_recon_loss = float(ema_avg_loss)
                        ema_best_epoch = epoch
                        torch.save(ema_checkpoint_payload, ema_best_ckpt_path)

                logger.log_metrics({
                    "type": "pretrain_ema",
                    "epoch": epoch,
                    "ema_epoch_avg_recon_loss": float(ema_avg_loss),
                    "ema_best_recon_loss": float(ema_best_recon_loss),
                    "ema_validate_interval": int(args.ema_validate_interval),
                    "ema_validate_start_epoch": int(args.ema_validate_start_epoch),
                })
                writer.add_scalar("Pretrain/EMA_Epoch_Loss", ema_avg_loss, epoch)
                writer.add_scalar("Pretrain/EMA_Best_Loss", ema_best_recon_loss, epoch)
                logger.write(
                    f"[Pretrain][EMA] Epoch {epoch} | avg recon: {ema_avg_loss:.4f} | "
                    f"best recon: {ema_best_recon_loss:.4f}\n"
                )
                logger.write(f"[Pretrain][EMA] Updated last checkpoint: {os.path.basename(ema_last_ckpt_path)}\n")
                if ema_is_best:
                    logger.write(f"[Pretrain][EMA] Updated best checkpoint: {os.path.basename(ema_best_ckpt_path)}\n")
                logger.flush()

                ema_wandb_payload = {
                    "epoch": epoch,
                    "pretrain/ema_epoch_avg_recon_loss": float(ema_avg_loss),
                    "pretrain/ema_last_checkpoint_epoch": int(epoch),
                    "pretrain/ema_last_recon_loss": float(ema_avg_loss),
                    "pretrain/ema_best_recon_loss": float(ema_best_recon_loss),
                    "pretrain/ema_validation_interval": int(args.ema_validate_interval),
                    "pretrain/ema_validation_start_epoch": int(args.ema_validate_start_epoch),
                }
                if ema_is_best:
                    ema_wandb_payload.update(
                        {
                            "pretrain/ema_best_checkpoint_epoch": int(ema_best_epoch),
                        }
                    )
                wandb_utils.log(ema_wandb_payload)

                run = wandb_utils.run()
                if run is not None:
                    try:
                        run.summary["ema_best_pretrain_loss"] = float(ema_best_recon_loss)
                        run.summary["ema_best_pretrain_epoch"] = int(ema_best_epoch)
                        run.summary["ema_last_pretrain_loss"] = float(ema_avg_loss)
                        run.summary["ema_last_pretrain_epoch"] = int(epoch)
                    except Exception:
                        pass
            accelerator.wait_for_everyone()

    writer.close()
    logger.write("Pretraining completed.\n")


Location3DTorch = Tuple[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
]


def _pack_locations(
    batch_locs: Any, device: torch.device
) -> Location3DTorch:
    """
    Convert a list of per-sample ((d0,d1),(h0,h1),(w0,w1)) into the structure
    """

    def to_long_tensor(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=torch.long, non_blocking=True)

    if (
        isinstance(batch_locs, (tuple, list))
        and len(batch_locs) == 3
        and all(isinstance(x, (tuple, list)) and len(x) == 2 for x in batch_locs)
        and isinstance(batch_locs[0][0], torch.Tensor)
    ):
        (d0, d1), (h0, h1), (w0, w1) = batch_locs
        return (
            (to_long_tensor(d0), to_long_tensor(d1)),
            (to_long_tensor(h0), to_long_tensor(h1)),
            (to_long_tensor(w0), to_long_tensor(w1)),
        )

    d0 = torch.tensor([loc[0][0] for loc in batch_locs], device=device, dtype=torch.long)
    d1 = torch.tensor([loc[0][1] for loc in batch_locs], device=device, dtype=torch.long)
    h0 = torch.tensor([loc[1][0] for loc in batch_locs], device=device, dtype=torch.long)
    h1 = torch.tensor([loc[1][1] for loc in batch_locs], device=device, dtype=torch.long)
    w0 = torch.tensor([loc[2][0] for loc in batch_locs], device=device, dtype=torch.long)
    w1 = torch.tensor([loc[2][1] for loc in batch_locs], device=device, dtype=torch.long)
    return ((d0, d1), (h0, h1), (w0, w1))
